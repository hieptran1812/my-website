---
title: "Building Images Fast and Securely in CI: BuildKit, Remote Cache, and Rootless Builders"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Building an image on your laptop is easy; building it fast, reproducibly, and securely inside a cold CI job is where it gets hard — here is how to wire BuildKit, a remote cache, a daemonless builder, and digest pinning so your CI build is 90 seconds, not 11 minutes, and never hands a poisoned dependency root on the runner."
tags:
  [
    "ci-cd",
    "devops",
    "buildkit",
    "docker",
    "kaniko",
    "build-cache",
    "rootless",
    "reproducible-builds",
    "supply-chain",
    "github-actions",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/building-images-fast-and-securely-in-ci-1.png"
---

The first time I owned a CI pipeline that actually mattered, I inherited an image-build job that took eleven minutes. Not because the image was huge — it was a fairly ordinary Node API with a Python sidecar — but because every single CI run started from nothing. The runner was ephemeral: GitHub spun up a fresh VM, ran the job, threw the VM away. Nothing survived between runs. So every build re-pulled the base image, re-ran `npm ci` and downloaded the entire dependency tree from the network, re-installed system packages with `apt-get`, and recompiled everything from a cold start. Eleven minutes, forty times a day, across a team of fifteen engineers. We were burning roughly seven CI-hours a day on work the build had already done yesterday, and the day before, and the day before that.

That was the *speed* problem, and it was the one everyone complained about. But underneath it were two quieter problems that nobody was looking at, and that would have hurt far more. The first: that job built images by mounting the host's Docker socket — `/var/run/docker.sock` — into the CI container, because that was the path of least resistance to "just run `docker build`." What that one line of YAML actually did was hand every CI job, including every pull request from every contributor, effective root on the runner host. The second: the image was not reproducible. Build the same git commit twice and you got two different images, because the build pulled `node:18` (a moving tag), ran `apt-get install` with no pins against a live package mirror, and baked the wall-clock timestamp into every layer. We were signing and promoting an artifact we could not reproduce, which means we could not actually prove what was in it.

This post is about fixing all three at once, because in CI they are not three separate problems — they are three faces of the same problem: **a build that runs as a cold, ephemeral job inside shared infrastructure cannot rely on any of the conveniences your laptop gives you for free.** Your laptop has a warm cache, a trusted single user, and a stable toolchain. A CI runner has none of those. So you have to *engineer back in* what the laptop gave you: a real build backend that understands caching and parallelism, a shared cache so cold runners are not slow, a daemonless builder so you do not hand the build root on the host, and pinning so the image is reproducible. The figure below is the spine of the whole post — the four jobs a CI image build has to do at once, in the order we will tackle them.

![Stacked diagram showing a CI image build stage from cold runner through build backend, remote cache, daemonless builder, secret mount, and pinned base down to a reproducible ninety second image](/imgs/blogs/building-images-fast-and-securely-in-ci-1.png)

This is part of the series **"CI/CD & Cloud-Native Delivery, From Commit to Production."** If you have not read [the CI/CD pipeline map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) yet, start there — it lays out the spine the whole series returns to: **commit → build → test → package → deploy → operate**, governed by **"build once, promote everywhere"** and **"everything as code,"** measured by the four DORA metrics. This post lives in the *package* stage, and it is the sharp end of "build once": you cannot promote one immutable artifact through five environments if your CI cannot build that artifact quickly, reproducibly, and without compromising the runner. It is the operational sequel to [the build stage post](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable), which covered *why* a build should be a pure function of its inputs; here we get our hands dirty making that true specifically *inside CI*, and it pairs with [writing a production Dockerfile](/blog/software-development/ci-cd/writing-a-production-dockerfile). By the end you will be able to set up BuildKit with cache mounts and secret mounts, wire a remote cache so cold runners reuse warm layers, replace the Docker-socket anti-pattern with a rootless or kaniko build, pin and timestamp-strip an image so two CI runs of the same commit produce the same bytes, and reason honestly about the cache-hit-rate math that makes it all pay off.

## 1. Why the laptop build betrays you in CI

Let me define the moving parts first, because everything downstream depends on them. A **container image** is a stack of read-only filesystem layers plus a small JSON config; each layer is the diff produced by one build instruction. A **registry** is the store you push images to — GHCR, ECR, Google Artifact Registry, Harbor. The **build backend** is the engine that actually executes your `Dockerfile` and produces those layers. On your laptop the backend is usually the Docker daemon — a long-lived root process that holds a *layer cache*: when you rebuild, it sees that a layer's instruction and inputs are unchanged and reuses the cached layer instead of redoing the work. That layer cache is the single biggest reason your laptop builds feel fast. Change one line near the bottom of your `Dockerfile` and only that layer and the ones after it rebuild; everything above is a cache hit.

Now put that exact build in CI and watch the cache evaporate. A modern CI runner is **ephemeral** — a fresh VM or container per job, discarded at the end. There is no warm daemon, no leftover layer cache, no `node_modules` from last time. The build that took ten seconds on your laptop on the second run takes the *full* cold-start time on every CI run, because every CI run *is* a first run. This is the **cold-cache problem**, and it is the defining constraint of building images in CI. It is not a bug in your `Dockerfile`; it is the nature of ephemeral infrastructure. Engineers who have only built locally are routinely shocked that their "fast" build is slow in CI, and they reach for the wrong fix — a bigger runner, more parallelism within the build — when the actual problem is that the cache is gone.

There is a second betrayal hiding behind the first. On your laptop, `docker build` talks to a daemon running as root that you implicitly trust, because it is *your* machine and *your* code. In CI, the build runs arbitrary instructions — including `RUN` steps that execute code from your dependencies — on shared infrastructure, often triggered by pull requests from people who are not you. The trust model is completely different, and the convenient way to give a CI job the ability to build images (mounting the host Docker socket) is exactly the wrong move under that trust model. We will come back to this in section 4; for now, hold the thought that *speed* and *security* in CI are both consequences of the same fact: the runner is cold and shared, not warm and trusted.

There is a third betrayal, quieter still, and it is the one that bit the team in the intro hardest. On your laptop, you rarely care whether two builds of the same code produce the same bytes — you build once and run it. In CI, the build *fabricates the artifact you ship*, and that artifact then flows, unchanged, through every environment under "build once, promote everywhere." If the build is not deterministic, you cannot answer the most basic supply-chain question — "is the image in production the one we built from this reviewed, scanned, signed commit?" — because rebuilding the commit gives you a *different* image, so there is nothing to compare against. The laptop lets you be sloppy about determinism because the artifact is disposable. CI cannot, because the artifact is the product. We cover this in section 5. The through-line of the whole post is that the three things a laptop gives you for free — a warm cache, a trusted single user, and a disposable artifact — all *invert* in CI, and each inversion is a job you have to do deliberately.

So the work of this post is to re-create, deliberately and as code, the three things the laptop gave you for free: a fast incremental build (sections 2, 3, and 6), a build that does not require trusting the job with the host (section 4), and a build whose output is reproducible (section 5). Let me start with the engine.

## 2. BuildKit: the engine that builds the graph, not the file

The legacy Docker builder treated your `Dockerfile` as a linear script: run instruction 1, then 2, then 3, each producing a layer, strictly in order. **BuildKit** — the modern build backend, default in current Docker and the engine behind `docker buildx` — treats your `Dockerfile` as a *dependency graph* instead. It parses the whole file, figures out which stages and instructions actually depend on which others, and then does only the work that is required, in parallel where it can. This single change in model is what unlocks everything else in this post, so it is worth understanding what it actually does.

Three behaviors matter for CI. First, **parallel execution of independent stages**. If you have a multi-stage build where one stage compiles your Go binary and another, independent stage builds your frontend assets, the legacy builder runs them one after the other; BuildKit runs them at the same time, because nothing in the graph says it must not. Second, **automatic build-graph skipping**: BuildKit computes a content hash for each node in the graph from its instruction *and its inputs*, so if the inputs to a node have not changed, it reuses the prior result — even pulling that result from a remote cache, which is the trick in section 3. Third, and this is the one teams underuse most, **cache mounts and secret mounts** via the `RUN --mount` syntax. The figure below shows the shape of a BuildKit build: not a ladder, a graph.

![Graph diagram of a BuildKit build where a pinned base feeds a deps stage and an asset stage in parallel, a compile stage, then a merged final stage producing a content addressed output image](/imgs/blogs/building-images-fast-and-securely-in-ci-3.png)

### Cache mounts: persist `~/.cache` across builds

Here is a distinction that trips up almost everyone, and it is the difference between a build that is merely "layer-cached" and one that is genuinely fast in CI. The **layer cache** caches the *output* of an instruction — the resulting filesystem layer. A **cache mount** (`RUN --mount=type=cache`) caches a *directory used during* an instruction, and persists it across builds without baking it into any layer. These are different caches solving different problems, and you want both.

Picture `npm ci`. The layer cache helps you only if `package-lock.json` is byte-for-byte unchanged — change one dependency and the whole `npm ci` layer is invalidated and re-runs from scratch, re-downloading *every* package, not just the changed one. A cache mount fixes precisely that: you mount npm's download cache (`~/.npm`) as a cache mount, so even when the layer is invalidated and `npm ci` re-runs, npm finds most packages already in its on-disk cache and skips the network. The same applies to `~/.cache/pip` for Python, `/root/.m2` for Maven, the Go build cache, Cargo's registry cache, and so on. Here is what it looks like in a `Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM node:18.20.4-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
# Cache mount: npm's download cache survives across builds.
# Even when this layer is invalidated, packages are reused from cache.
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm ci --prefer-offline --no-audit

FROM python:3.12.4-slim-bookworm AS pydeps
WORKDIR /svc
COPY requirements.txt ./
# Cache mount for pip's wheel cache — no re-download on lockfile changes.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --require-hashes -r requirements.txt
```

Two things to notice. `sharing=locked` serializes concurrent writers to the same cache mount, which matters when BuildKit runs stages in parallel and two of them would otherwise race on the cache directory. And the cache mount is *not part of the image* — nothing in `/root/.npm` ends up in a layer, so it neither bloats the image nor leaks into what you ship. It exists only during the build and persists on the BuildKit daemon's storage (or, in CI, in the remote cache — section 3) between builds.

To make the payoff vivid: without a cache mount, a one-line dependency bump invalidates the `npm ci` layer and re-downloads *all* of your dependencies — for a medium Node project that is hundreds of packages and tens of megabytes over the network, often 60–120 seconds of pure download. With the cache mount, npm finds nearly every package already extracted in `/root/.npm` and fetches only the one that changed: the same step drops to a handful of seconds. The cache mount turns "dependency layer invalidated" from a worst case into a near-no-op, which matters enormously in CI where lockfile-touching commits are common (every Renovate or Dependabot PR is one). It is the difference between "we only added one package, why did the build take 11 minutes" and a build that barely notices.

A common mistake worth flagging: people set the cache mount target to the wrong directory and wonder why it does nothing. The target must be the path the tool *actually* reads and writes its cache to, which is not always obvious. For npm it is `~/.npm` (the download cache), not `node_modules` (the install output — that one you usually do *not* want to cache-mount, because it is the thing the layer should produce). For pip it is `~/.cache/pip`. For Go it is the build cache (`/root/.cache/go-build`) and the module cache (`/go/pkg/mod`). For Maven it is `~/.m2/repository`. Cache the tool's *download/compile cache*, not its *output*; mixing the two up is how you get a build that is either slow (cached nothing useful) or subtly wrong (cached output that should have been rebuilt).

### Secret mounts: use a private token without baking it in

The other `--mount` you must know is `type=secret`, and it fixes one of the oldest image-security footguns: leaking a credential into a layer. Suppose your build needs a token to pull from a private package registry. The naive approaches all leak it. `ENV NPM_TOKEN=...` bakes the token into the image config, readable by anyone who pulls the image. `ARG NPM_TOKEN` with `--build-arg` leaks it into the build history and often into the layer too. Even `COPY .npmrc` followed by `RUN ... && rm .npmrc` leaves the file in an intermediate layer — deleting it in a later layer does not remove it from the earlier one, and `docker history` or a layer extractor recovers it trivially.

A **secret mount** exposes the secret as a file mounted into the `RUN` step's filesystem *only for the duration of that instruction*, and it is never written to any layer or to the build history:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM node:18.20.4-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
# The token is a file at /run/secrets/npm_token only inside this RUN.
# It is NOT baked into the layer or recoverable from image history.
RUN --mount=type=secret,id=npm_token \
    --mount=type=cache,target=/root/.npm,sharing=locked \
    NPM_TOKEN="$(cat /run/secrets/npm_token)" npm ci --prefer-offline
```

You pass the secret from the build command (`docker buildx build --secret id=npm_token,env=NPM_TOKEN ...`) and BuildKit wires it through. There is also `--mount=type=ssh`, which forwards your SSH agent into a `RUN` step so a `git clone` of a private repo or a `go mod download` against a private VCS works without the private key ever touching the image. The principle in all three cases is the same: **a build-time credential should be available *during* a build step and present in *zero* layers afterward.** If a secret is in your image's history, treat it as compromised — registries are pulled by more people than you think.

### How BuildKit's content-addressed cache works

The reason all this composes is that BuildKit's cache is **content-addressed**. Each node in the build graph gets a cache key derived from the operation plus the digests of its inputs — the parent layer, the files a `COPY` brought in, the literal command string. If two builds produce the same cache key for a node, they are guaranteed to produce the same output, so the output can be reused. This is also why *order matters* in a `Dockerfile`: put the instructions that change rarely (installing dependencies) above the ones that change every commit (copying your source), and the dependency layers stay cache-valid across most builds. It is the same content-addressing idea that lets BuildKit pull a cached layer from a *remote* store as if it had built it locally — which is the fix for the cold-cache problem, and where we go next.

It is worth being precise about *what* goes into a cache key, because the difference between a build that caches well and one that thrashes is almost always a misunderstanding here. For a `COPY` or `ADD`, BuildKit hashes the *content* of the files being copied — not their names, not their timestamps, the actual bytes. So `COPY package.json package-lock.json ./` produces a cache hit as long as those two files are byte-identical, regardless of when or where you build. That is exactly why the canonical fast `Dockerfile` copies the lockfiles and runs the install *before* copying the rest of the source: the install layer's cache key depends only on the lockfiles, so it survives every commit that does not touch them. For a `RUN`, by contrast, BuildKit (by default) hashes only the *command string and the parent layer* — it does not look inside the `RUN` to see what files it reads, because it cannot. This is the source of the most common cache surprise: a `RUN` that downloads something from the network gets a cache *hit* even though the network content changed, because the command string is identical. That is a feature for speed and a footgun for correctness — and it is precisely why pinning (section 5) matters: you want the command string to fully determine the output, so that a cache hit is also a *correct* hit.

There is a subtle interaction between cache mounts and cache keys worth calling out, because it confuses people the first time they hit it. A cache mount (`--mount=type=cache`) is *not* part of the `RUN`'s cache key. The contents of `/root/.npm` do not change whether the `RUN npm ci` layer is a hit or a miss — the layer's hit/miss is decided by the lockfile content (the `COPY` above it) and the command string. The cache mount only changes how *fast* the `RUN` executes when it does run. So the two caches stack cleanly: the layer cache decides *whether* to run the step, and the cache mount makes the step *cheap* when it does run. Internalizing that two-level model is what lets you reason about why your build is slow: a slow build with a warm remote cache almost always means a layer is being invalidated (a cache-key miss) — and the question becomes "which input changed," not "why is npm slow."

## 3. The cold-cache problem and the remote cache fix

Here is the central tension of building in CI, stated plainly. BuildKit's caching is brilliant, but it caches to *local* storage by default — and in CI there is no persistent local storage, because the runner is ephemeral. So all the cache cleverness from section 2 buys you exactly nothing on a fresh runner: the first build of the day and the fortieth are both cold. The fix is to move the cache *off* the runner and into a shared store that every runner can read and write: a **remote cache** (also called an external cache). BuildKit supports this directly with two flags — `--cache-to` (export this build's cache to the store) and `--cache-from` (import cache from the store before building).

The shape of the solution is in the figure below: one runner does a full build and *exports* its layers and cache mounts to a shared backend; a later, cold runner *imports* them and only rebuilds what actually changed.

![Graph diagram showing runner A exporting build layers via cache-to into a shared registry cache and a cold runner B importing them via cache-from to reach a reused ninety second build at eighty five percent cache hit](/imgs/blogs/building-images-fast-and-securely-in-ci-7.png)

### Inline vs registry vs gha vs s3

There are a few flavors of remote cache, and choosing wrong is a common reason teams set up a remote cache and still see slow builds. The figure and table below lay them out.

![Matrix comparing inline, registry, gha, and S3 remote cache backends across scope, whether intermediate layers are kept, and what each is best for](/imgs/blogs/building-images-fast-and-securely-in-ci-4.png)

| Cache backend | How invoked | Keeps intermediate layers? | Best for |
| --- | --- | --- | --- |
| **inline** | `--cache-to type=inline` | No — only final-image layers travel inside the image | One simple single-stage image where the final layers are most of the cost |
| **registry** | `--cache-to type=registry,ref=...,mode=max` | Yes, with `mode=max` — all stages including build-only stages | Any CI that can push to a registry; the default good choice |
| **gha** | `--cache-to type=gha,mode=max` | Yes — stored in the GitHub Actions cache (10 GB repo cap) | GitHub Actions specifically; zero extra infra |
| **s3 / azblob** | `--cache-to type=s3,region=...,bucket=...` | Yes, no hard cap you do not set yourself | Self-hosted runner fleets and large monorepos |

The single most important distinction is **`mode=max` versus the default `mode=min`**. With `mode=min` (the default), BuildKit exports cache only for the layers that end up in the final image. That sounds fine until you have a multi-stage build, where the expensive work — installing the full toolchain, compiling, running a heavy `npm install` in a `build` stage — happens in a stage that gets *discarded* before the final image. Those stages produce no final-image layers, so `mode=min` does not cache them, so they rebuild cold every time, and your "remote cache" mysteriously does nothing for your slowest steps. `mode=max` exports cache for *every* stage. The rule: **for multi-stage builds, always use `mode=max`.** The cost is a larger cache, which is exactly the trade-off you want.

Inline cache is the seductive trap. It is the simplest to set up — it rides along inside the image you push, no separate cache reference — but it can only carry final-image layers (it is effectively `mode=min`), so it is useless for the multi-stage builds that dominate real projects. Use a separate `registry` cache ref (or `gha` on GitHub Actions) for anything non-trivial.

There is a second axis to choose on: **cache scope and eviction**. A remote cache is a store with finite size and a replacement policy, and that policy quietly determines your real-world hit rate. The `gha` backend has a 10 GB per-repository cap and evicts least-recently-used entries; if your cache is large (big multi-stage builds, multiple architectures, many branches each writing their own scope), you can blow the cap and find that yesterday's warm layers were evicted overnight, so your "remote cache" delivers a cold build on Monday morning. The fixes are to scope deliberately (one cache ref per long-lived branch or per architecture, not per ephemeral PR branch — PR branches should *read* `main`'s cache, not each write their own and thrash the LRU) and to prune what you cache (you rarely need to cache test-only stages). A `registry` cache on a registry you control has no hard cap but does cost storage, so you set a retention policy on the cache repository instead. Either way, the mental shift is that a cache is a *budget* you manage, not a magic box — and an unmanaged cache regresses to cold silently, which is the worst failure mode because nobody notices until the bill or the build time creeps back up.

A related decision is **whether PR builds should write to the cache at all.** The safe default: PR builds read the `main`-branch cache (`cache-from`) but do *not* write to it (`cache-to` only on pushes to `main`). This stops a contributor's experimental PR — or a malicious one — from poisoning the shared cache that everyone else's builds import. It also keeps the LRU dominated by the layers that matter (the mainline ones). The build job in section 3 does exactly this with `push: ${{ github.event_name == 'push' }}`-style gating; extend the same idea to the cache export. This is a small policy choice with an outsized effect on both hit rate and supply-chain safety, and it is the kind of detail that separates a remote cache that helps from one that merely exists.

### A real GitHub Actions build job with a remote cache

Here is a complete, copy-and-adapt build job. It uses `docker buildx` with the `gha` cache backend, passes a secret without baking it, and pushes with credentials scoped by OIDC rather than a long-lived password.

```yaml
name: build-image
on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read
  packages: write
  id-token: write   # required for OIDC

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Buildx (BuildKit)
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: ${{ github.event_name == 'push' }}
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          # Remote cache: import last run's layers, export this run's.
          cache-from: type=gha
          cache-to: type=gha,mode=max
          # Secret mount — token is available during build, never in a layer.
          secrets: |
            npm_token=${{ secrets.NPM_TOKEN }}
          provenance: true   # generate SLSA provenance attestation
          sbom: true         # generate an SBOM attestation
```

Three details earn their keep. `cache-from`/`cache-to: type=gha` with `mode=max` gives every job the warm cache of the last job, across ephemeral runners. `secrets:` wires the npm token to the `--mount=type=secret,id=npm_token` in the `Dockerfile` from section 2. And `provenance: true` plus `sbom: true` make `buildx` emit a SLSA provenance attestation and a software bill of materials alongside the image — the supply-chain foundation that [image scanning and signing](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface) (the next post in this track) builds on. We are getting fast *and* setting up secure in the same job.

### The cache-hit-rate payoff

The reason a remote cache is worth the setup is arithmetic, and it is worth doing the arithmetic so you can defend the change. Let $t_{\text{cold}}$ be the cold build time, $t_{\text{warm}}$ the time when the dependency layers are reused, and $h$ the **cache-hit rate** — the fraction of builds in which the cached layers are valid. The expected build time is:

$$ \mathbb{E}[t] = h \cdot t_{\text{warm}} + (1 - h) \cdot t_{\text{cold}} $$

If $t_{\text{cold}} = 11$ minutes, $t_{\text{warm}} = 1.5$ minutes, and your dependency layers change only on the minority of commits that touch a lockfile so $h = 0.85$, then $\mathbb{E}[t] = 0.85 \times 1.5 + 0.15 \times 11 = 1.275 + 1.65 = 2.925$ minutes — roughly **3 minutes average instead of 11**. Across 40 builds a day that is $40 \times (11 - 2.9) \approx 324$ runner-minutes saved per day, or about 5.4 runner-hours daily that you stop paying for. The hit rate $h$ is the lever: anything that invalidates the dependency layers (reordering the `Dockerfile`, an unpinned base tag moving, a noisy lockfile) drops $h$ and the average climbs back toward cold. This is why the build hygiene from [the build stage post](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) — stable instruction order, lockfiles, pinned bases — is not separate from speed; it *is* the speed, because it protects $h$.

#### Worked example: a cold CI build cut from 11 minutes to 90 seconds

Take the real job I inherited. The cold build broke down roughly like this: pull base image 40s, `apt-get install` system packages 70s, `npm ci` (full network download) 210s, `pip install` 150s, compile + asset build 160s, copy + finalize 30s — about 660 seconds, 11 minutes. Here is where each second went and how we removed it:

1. **Remote cache (`type=gha,mode=max`).** The base-image layer, the `apt-get` layer, and the *result* of `npm ci`/`pip install` are all imported from the last run on a cold runner — provided their inputs are unchanged. On a typical commit that touches only application code, those layers are cache hits. That alone removes the 40 + 70 + 210 + 150 = 470 seconds of dependency work on a warm hit.
2. **Cache mounts.** On the *minority* of commits that *do* change a lockfile, the dependency layer is invalidated, so `npm ci` re-runs — but the `--mount=type=cache,target=/root/.npm` means npm reuses already-downloaded packages and only fetches the delta. The 210s `npm ci` becomes maybe 35s; the 150s `pip install` becomes ~25s. So even cache *misses* on dependencies are cheap.
3. **Parallel multi-stage.** The Node deps stage and the Python deps stage are independent, so BuildKit runs them concurrently rather than back to back — on a lockfile-change commit that overlaps the two install times instead of summing them.
4. **What is left to actually build** on a normal commit is the compile + asset step (160s) only if the source changed — which on most commits it did, but on docs-only or config-only commits even that is a cache hit.

On the common path (app code changed, dependencies unchanged): import warm layers (~10s of cache import), rebuild the compile/asset stage incrementally (~60s), finalize (~20s) — about **90 seconds**. On the occasional lockfile-change path it climbs to ~3–4 minutes, still a fraction of 11. The before-and-after is in the figure below.

![Before and after diagram contrasting a cold fresh runner that re-pulls and re-downloads everything for an eleven minute build against a runner using a registry remote cache that reuses dependency layers for a ninety second build](/imgs/blogs/building-images-fast-and-securely-in-ci-2.png)

The honest caveat: the *first* build after the cache is empty (or after a cache eviction — the `gha` backend has a 10 GB repo cap and evicts least-recently-used entries) is still cold. A remote cache changes the *average*, not the worst case. Measure $h$ in practice by logging which layers were `CACHED` in your `buildx` output over a week; if $h$ is low, your invalidation hygiene is the problem, not the cache.

## 4. Daemonless builders and the Docker-socket disaster

Now the security half, and it is the part I care about most, because the speed problem costs you money while the security problem can cost you the company. To build an image you need to run `docker build` — but the Docker CLI is just a client; the actual building is done by the Docker **daemon**, a privileged process running as root. So when a CI job needs to build images, the question is: *how does the job reach a daemon?* The convenient, widely-copied, and dangerous answer is **Docker-in-Docker via the host socket** — mount `/var/run/docker.sock` from the runner host into the CI container.

### Why mounting the socket hands over root

Here is the mechanism, and it is worth being precise because people wave it away as theoretical. The Docker socket is the daemon's full control API. Anything that can talk to that socket can tell the daemon to do anything the daemon can do — and the daemon runs as root on the host. In particular, a process with socket access can ask the daemon to run a new container with the host's root filesystem mounted and the `privileged` flag set, then `chroot` into it. At that point it is root on the *runner host*, not just inside the build container. The container boundary is gone. This is not an exploit chain; it is the documented, intended capability of the socket, abused. The figure below contrasts it with the safe path.

![Before and after diagram showing a CI job that mounts the docker socket and runs as host root where a poisoned dependency owns the runner, against a rootless kaniko build that confines the blast radius to the job](/imgs/blogs/building-images-fast-and-securely-in-ci-5.png)

Now layer on the CI trust model. A `RUN` step executes whatever your build does — and your build executes your dependencies' install scripts, your `Makefile`, code from packages you do not control. Modern attacks specifically target this: a compromised transitive dependency (the **dependency-confusion** and **typosquatting** class of attacks) runs malicious code at build time. If that build has socket access, the malicious code does not just exfiltrate a secret from one build — it owns the runner. From the runner it can read the credentials of *every other job* that shares it, push backdoored images under your organization's identity, and pivot into your registry and cloud. This is the shape of the worst supply-chain compromises of the last decade: the build environment was the soft target.

### The safer builders

The fix is to **never give the build a privileged daemon at all** — use a builder that constructs images in *userspace*, with no root and no daemon. The figure below is the taxonomy; the table compares the practical options.

![Tree diagram of CI image builder choices branching into daemon based Docker in Docker which mounts the socket versus daemonless options kaniko, buildah or podman, and rootless BuildKit](/imgs/blogs/building-images-fast-and-securely-in-ci-6.png)

| Builder | Daemon? | Root needed? | How it fits CI |
| --- | --- | --- | --- |
| **Docker-in-Docker (host socket)** | Yes, host daemon | Yes — effectively root on host | The anti-pattern. Avoid. |
| **Docker-in-Docker (privileged dind)** | Yes, nested daemon | Privileged container | Better isolation than the socket, but still a privileged container — a meaningful risk |
| **kaniko** | No | No (runs as an unprivileged container) | Executes the `Dockerfile` in userspace inside the cluster; ideal for Kubernetes-based CI |
| **buildah / podman** | No | No (rootless via user namespaces) | Daemonless; `buildah bud` builds from a `Dockerfile`; great on self-hosted Linux runners |
| **BuildKit rootless** | No (rootless `buildkitd`) | No (user namespaces) | Keeps BuildKit's caching and `--mount` features, no privilege |

The trade-off to be honest about: kaniko and buildah do not have *every* BuildKit feature, and historically kaniko's caching was weaker than BuildKit's (it caches to a registry but the semantics differ). If you depend heavily on BuildKit cache mounts and the full remote-cache machinery, **rootless BuildKit** is the option that keeps the speed work from section 3 *and* drops the privilege. If you are on Kubernetes-native CI (Tekton, Argo Workflows, GitLab Kubernetes executor), kaniko is the most common choice because it needs no special node configuration.

How do daemonless builders actually build an image without root? The trick is **user namespaces** — a Linux kernel feature that lets a process *think* it is root (uid 0) inside its own namespace while being an ordinary unprivileged user (say, uid 100000) on the host. Inside that namespace the builder can do the root-like things a build needs — create files owned by uid 0, run package managers that expect root, set up the layered filesystem — but none of that maps to any privilege on the host. If a malicious `RUN` step tries to escape, it finds it is uid 100000 with no real capabilities. kaniko takes a slightly different approach: it runs as an unprivileged container and executes the `Dockerfile` instructions *in its own filesystem*, snapshotting changes after each instruction to produce layers, never asking a daemon to do anything. Both approaches share the security property that matters: there is no privileged, long-lived daemon for a compromised build to hijack, and the build cannot reach outside its own sandbox. The cost is some performance and feature surface — rootless overlay filesystems can be slightly slower, and a few exotic `Dockerfile` features behave differently — but for the overwhelming majority of builds it is a clean swap, and the security gain is enormous.

One more practical note that surprises teams migrating off Docker-in-Docker: the *output* is identical. A kaniko or rootless-BuildKit build produces a standard OCI image, indistinguishable from one the Docker daemon would build, pushed to the same registry, pulled by the same Kubernetes. You are not adopting a different image format or a different deploy path — you are only changing the *builder*, which is exactly why the migration is usually low-risk. The hardest part is almost always disentangling the one job that "needs" the socket for some side task (running tests against a real database container during the build, say) — and the answer there is to move that side task *out* of the image build into a separate CI step with its own service container, not to keep the socket.

### A kaniko build job with no privilege and OIDC-scoped push

Here is a kaniko build as a Kubernetes Pod (the shape GitLab CI and Tekton both use), running unprivileged, building from the `Dockerfile`, and pushing with short-lived OIDC-federated credentials instead of a static registry password:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kaniko-build
spec:
  # No privileged, no hostPath docker.sock, no extra capabilities.
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
  containers:
    - name: kaniko
      image: gcr.io/kaniko-project/executor:v1.23.2
      args:
        - "--dockerfile=Dockerfile"
        - "--context=git://github.com/acme/api#refs/heads/main"
        - "--destination=ghcr.io/acme/api:$(CI_COMMIT_SHA)"
        # Remote cache to a registry ref — warm layers across runners.
        - "--cache=true"
        - "--cache-repo=ghcr.io/acme/api/cache"
        - "--reproducible"   # strip timestamps for a deterministic image
      env:
        - name: CI_COMMIT_SHA
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['commit-sha']
      volumeMounts:
        - name: docker-config
          mountPath: /kaniko/.docker
  volumes:
    # Push creds injected by an OIDC-federated token, not a static secret.
    - name: docker-config
      projected:
        sources:
          - serviceAccountToken:
              audience: ghcr.io
              expirationSeconds: 3600
              path: token
  restartPolicy: Never
```

Notice what is *absent*: no `hostPath` mount of the socket, no `privileged: true`, no static long-lived registry password. The push credential is a one-hour OIDC token scoped to the registry, so even if the build is compromised, the worst it can do is push to one repo for one hour — not assume the runner's full identity. The `--reproducible` flag is the bridge to the next section.

#### Worked example: the Docker-socket privilege-escalation, and the fix

Let me make the attack concrete, because the abstract version under-sells the danger. A team builds images in CI by mounting `/var/run/docker.sock`. A developer adds a new npm dependency; that dependency has a transitive sub-dependency whose maintainer account was compromised, and the bad actor publishes a patch version with a malicious `postinstall` script. The next CI run — triggered by an ordinary, well-meaning pull request — runs `npm ci`, which runs the `postinstall`, which runs as part of the build, which has access to the mounted socket. The script asks the daemon to launch a privileged container with the host root mounted, reads the runner's environment and the credentials of other jobs from the host, finds the registry push token, and builds and pushes a *backdoored* image of the team's flagship service to the registry under the organization's own identity. Because it is pushed by the legitimate CI, it sails through whatever trust the deploy pipeline places in "images our CI built." The backdoor reaches production.

Now run the same poisoned dependency through the kaniko Pod above. The `postinstall` still runs — you cannot stop a dependency from running its own install script during a build. But there is no socket to talk to, so it cannot escape the build container; it runs as an unprivileged user inside an unprivileged Pod. The push credential it can reach is a one-hour OIDC token scoped to a single repo, so it cannot impersonate the runner or push to other services. The blast radius is one job and one repo for one hour, and the [image scan gate](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface) plus image signing (so the deploy only accepts images signed by the *expected* pipeline) catch the tampering before deploy. The difference between "the company is breached" and "one job ran some junk and got caught" is entirely the build's privilege model. This is why [securing the pipeline itself](/blog/software-development/ci-cd/securing-the-pipeline-itself) treats the build environment as a primary attack surface, not an afterthought.

The rule, stated bluntly: **never mount the host Docker socket into a CI job.** If you remember one thing from this post, make it that.

## 5. Reproducible image builds in CI

The third face of the problem. Recall from the intro that we were signing and promoting an image we could not reproduce. **Reproducible** means: build the same git commit twice, on two different machines at two different times, and get a *byte-for-byte identical* image — same content digest. This is not a purist's luxury. It is the foundation of "build once, promote everywhere": if you cannot reproduce the artifact, you cannot prove that the thing you scanned and signed is the thing you deployed, and you cannot verify a [provenance attestation](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) against the source. A non-reproducible build means your supply-chain guarantees are theater.

Builds are non-reproducible because they read undeclared inputs. Three sources dominate in container builds, and each has a fix.

### Pin the base image by digest, not by tag

`FROM node:18` is a moving target. The `18` tag points at whatever the maintainers most recently pushed for Node 18 — a new patch, a rebased base OS, a different set of system libraries. Two builds a week apart can resolve `node:18` to two different images and silently produce different artifacts. The fix is to pin by **digest**, the immutable content hash of a specific image:

```dockerfile
# Pinned by digest — this is ALWAYS the same bytes, forever.
FROM node:18.20.4-bookworm-slim@sha256:6326b52a... AS base
```

The digest is the cryptographic identity of the image; it cannot drift. Renovate or Dependabot can bump the digest in a controlled PR (so you still get security updates), but each individual build is pinned. Pin *every* `FROM`, including intermediate stages and the final base.

### Strip nondeterministic timestamps with `SOURCE_DATE_EPOCH`

The sneakiest source of non-reproducibility is *time*. Every file written during a build gets a modification timestamp, and image layers record the wall-clock time they were created. Build the same source at 9:00 and at 10:00 and the timestamps differ, so the layer digests differ, so the image digest differs — even though the *contents* are identical. The cross-tool standard for fixing this is the `SOURCE_DATE_EPOCH` environment variable: a Unix timestamp that build tools honor as "pretend everything was created at this instant." Set it to the commit's own timestamp so it is deterministic per commit:

```bash
# Use the commit's author date as the canonical build time.
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

docker buildx build \
  --build-arg SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  --output type=image,name=ghcr.io/acme/api:"$GITHUB_SHA",rewrite-timestamp=true \
  --cache-from type=gha --cache-to type=gha,mode=max \
  .
```

BuildKit's `rewrite-timestamp=true` output option rewrites layer timestamps to `SOURCE_DATE_EPOCH`, and kaniko's `--reproducible` flag does the equivalent. With timestamps normalized and the base pinned by digest, the two biggest sources of drift are gone.

### Kill network-at-build-time nondeterminism

The last source is the network. `apt-get install curl` with no version pin fetches *whatever version the mirror serves today*; run it next month and you get a different version, hence different bytes. `RUN curl https://example.com/install.sh | sh` is worse — you are executing whatever lives at that URL right now, with no integrity check at all, which is both non-reproducible and a supply-chain hole. The fixes: pin package versions (`apt-get install -y curl=7.88.1-10+deb12u5`), use lockfiles with integrity hashes for language deps (`npm ci` against a committed `package-lock.json`, `pip install --require-hashes`), and verify checksums for any binary you download. The principle from [the build stage post](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) holds: **a reproducible build reads only its declared, pinned inputs.** Below is a comparison of the three drift sources and their fixes.

| Drift source | Symptom | Fix |
| --- | --- | --- |
| Moving base tag (`FROM node:18`) | Same commit resolves to different base over time | Pin by digest `@sha256:...`; bump via Renovate PR |
| Wall-clock timestamps in layers | Identical contents, different layer digests by time of day | `SOURCE_DATE_EPOCH` + `rewrite-timestamp=true` (or kaniko `--reproducible`) |
| Unpinned network fetches (`apt-get`, `curl \| sh`) | Different package versions on different days; no integrity | Pin versions; use lockfiles with hashes; verify checksums |

How do you *prove* reproducibility rather than hope for it? Build the same commit twice on two runners and diff the resulting image digests; if they match, the build is reproducible for that commit. A periodic CI job that does exactly this — "rebuild last release's commit and assert the digest matches what we shipped" — is the honest way to keep reproducibility from rotting. It is the same instinct as a backup you actually restore.

#### Worked example: a reproducibility-verification job that catches drift

Reproducibility is the kind of property that decays silently: it works the day you set it up, then six weeks later someone adds an unpinned `apt-get install`, or a tool starts embedding a hostname in a config file, and now your builds drift again — but nobody notices, because nothing *fails*. The fix is to make non-reproducibility *fail loudly* with a scheduled job that rebuilds a known commit and compares digests:

```yaml
name: verify-reproducible
on:
  schedule:
    - cron: "0 6 * * *"   # nightly
jobs:
  rebuild-and-compare:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ vars.LAST_RELEASE_SHA }}   # the commit we actually shipped
      - uses: docker/setup-buildx-action@v3
      - name: Rebuild the released commit deterministically
        run: |
          export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
          docker buildx build \
            --build-arg SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
            --output type=image,name=local/verify,rewrite-timestamp=true,push=false \
            --metadata-file meta.json .
      - name: Compare to the shipped digest
        run: |
          built=$(jq -r '.["containerimage.digest"]' meta.json)
          shipped="${{ vars.LAST_RELEASE_DIGEST }}"
          test "$built" = "$shipped" \
            || { echo "DRIFT: $built != $shipped"; exit 1; }
```

If the rebuilt digest differs from the digest you actually shipped, the job fails and you have caught a reproducibility regression the day it landed — not during the incident six weeks later when you are trying to figure out what is actually running in production. The cost is one cheap nightly build; the value is that "build once, promote everywhere" stays *true* rather than aspirational. I treat a failing reproducibility job the same as a failing test: someone broke a contract, and the build is the most expensive contract to have silently broken.

## 6. Multi-arch builds and the emulation tax

A growing reason CI image builds are slow has nothing to do with caching: **multi-architecture builds.** With ARM in the data center (AWS Graviton, Ampere, Apple Silicon dev laptops) and x86 still dominant elsewhere, many teams now ship a single multi-arch image (a manifest list pointing at one image per platform) so the same tag runs on `linux/amd64` and `linux/arm64`. The question is *how* you build the non-native platform, and the answer hides a large performance tax.

The default `buildx` approach uses **QEMU emulation**: the runner emulates the foreign CPU in software to run the build steps for the other architecture. It works with zero extra infrastructure — `docker buildx build --platform linux/amd64,linux/arm64` just works on an amd64 runner via a QEMU shim. But emulated builds are *slow*, often 3–10× slower for CPU-heavy steps (compilation especially), because every instruction is being interpreted. A Go or Rust compile that takes 2 minutes native can take 10–20 minutes emulated, and you pay it every cold build.

The alternative is **native runners per architecture**: build amd64 on an amd64 runner and arm64 on an arm64 runner (GitHub now offers ARM-hosted runners; most clouds offer ARM instances), then merge the two into one manifest list. This is faster but needs two runners and an extra merge step. The trade-off:

| Approach | Speed | Infra | When to use |
| --- | --- | --- | --- |
| **QEMU emulation** | Slow (3–10× on CPU-heavy steps) | None — one runner | Small images, infrequent builds, or you just need it to work |
| **Native per-arch runners + merge** | Fast (native speed each) | Two runner types + merge job | Hot path, CPU-heavy builds, large monorepos |

Here is the native-per-arch pattern in GitHub Actions: a matrix builds each platform on its own runner, pushes a digest, and a final job assembles the manifest list:

```yaml
jobs:
  build:
    strategy:
      matrix:
        include:
          - platform: linux/amd64
            runner: ubuntu-latest
          - platform: linux/arm64
            runner: ubuntu-24.04-arm   # native ARM runner — no QEMU
    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        id: build
        with:
          platforms: ${{ matrix.platform }}
          outputs: type=image,name=ghcr.io/acme/api,push-by-digest=true
          cache-from: type=gha,scope=${{ matrix.platform }}
          cache-to: type=gha,mode=max,scope=${{ matrix.platform }}

  merge:
    needs: build
    runs-on: ubuntu-latest
    steps:
      # docker buildx imagetools assembles one manifest list from
      # the two per-arch digests pushed above.
      - run: |
          docker buildx imagetools create \
            -t ghcr.io/acme/api:${{ github.sha }} \
            ghcr.io/acme/api@${{ needs.build.outputs.amd64-digest }} \
            ghcr.io/acme/api@${{ needs.build.outputs.arm64-digest }}
```

Note the `scope=${{ matrix.platform }}` on the cache — without distinct scopes the two architectures fight over the same cache key and thrash. The honest recommendation: if you only deploy to one architecture, **do not build multi-arch at all** — it doubles your build cost for an image nobody runs. Build multi-arch only for the platforms you actually deploy to or ship to others.

## 7. Measuring the build so the speedup is real, not anecdotal

Everything above is a hypothesis until you measure it, and "the build feels faster" is not a metric an engineering manager can budget against. The discipline this series insists on — *measured proof* — applies hardest to the build, because the build is where the time and money actually go. So before we assemble the final job, let me lay out how to measure it honestly, because the wrong measurement will lie to you.

The first thing to measure is **wall-clock build duration, split into cold and warm**. A single average is misleading: you want the distribution. Tag each CI build with whether its dependency layers were `CACHED` (BuildKit prints this for every step) and bucket the durations. You will typically see a bimodal distribution — a cluster of warm builds near $t_{\text{warm}}$ and a smaller cluster of cold ones near $t_{\text{cold}}$ — and the *ratio* of the two clusters is your cache-hit rate $h$ in the wild. If you only ever see the cold cluster, your remote cache is not working (the usual culprit: `mode=min` on a multi-stage build, or a cache ref the runner cannot read). If the warm cluster is not as fast as you expected, a layer is being invalidated that should not be.

The second thing is **cache-hit rate per layer, not just per build**. BuildKit's `--progress=plain` output lists each step and whether it was cached. Parse it and you can see *which* layer is the one that keeps missing. I have lost count of the times a "cache is not working" complaint turned out to be a single `COPY . .` placed too high in the `Dockerfile`, invalidating everything below it on every commit because the `.` includes a file that changes every build (a generated version file, a `.git` directory, a build timestamp). A `.dockerignore` that excludes the noisy paths often recovers more cache-hit rate than any amount of remote-cache tuning. Measure per-layer and the fix becomes obvious.

The third thing is **CI cost, in money**. Build minutes are not free: hosted runners bill per minute, and larger runners bill more per minute. Multiply your average build time by builds-per-day by your per-minute rate and you get a number a budget owner cares about. The 11-minute-to-90-second change in the worked example, at 40 builds/day on a runner that costs roughly \$0.008/minute, is the difference between about \$105/month and \$14/month in build minutes *for one job* — and a real org has dozens of such jobs. Frame the speedup in money and the afternoon of caching work pays for itself in the first week.

```bash
# Emit BuildKit's plain progress so you can parse cache hits per step.
docker buildx build --progress=plain . 2>&1 | tee build.log

# Count cached vs run steps to estimate this build's layer hit rate.
cached=$(grep -c 'CACHED' build.log)
ran=$(grep -cE '^#[0-9]+ DONE' build.log)
echo "cached steps: $cached  ran steps: $ran"
```

A word on what *not* to measure: do not chase the absolute fastest possible build at the cost of correctness. A build that skips a pinned-version check to shave ten seconds, or caches a layer it should rebuild, is fast and *wrong* — and a wrong artifact in production costs orders of magnitude more than a slow build. The goal is the fastest build that is still reproducible and secure, which is exactly why those three properties belong in one conversation.

#### Worked example: shrinking the image to make the build (and the cache) faster too

Speed is not only about caching the build; the *size* of what you produce feeds back into how fast every downstream step runs and how much cache you push around. Take the same Node + Python service. Built naively — `FROM node:18` (the full image, ~1 GB), the whole toolchain left in the final image, dev dependencies and build artifacts all shipped — it came out at about 1.9 GB. Here is the reduction, step by step, and why each step also helps the build:

1. **Multi-stage build.** Do the compile and `npm ci` in a `build` stage; copy only the built output into a slim runtime stage. The toolchain (compilers, headers, dev dependencies) never reaches the final image. That alone took it from 1.9 GB to about 320 MB.
2. **Slim or distroless runtime base.** Swap `node:18` for `node:18.20.4-bookworm-slim` (~120 MB base) or, for a static binary, `gcr.io/distroless/static` (~2 MB, no shell, no package manager — also a smaller attack surface). The runtime stage dropped to about 95 MB.
3. **Prune what you copy.** `npm ci --omit=dev` in the runtime stage's dependency install, and a tight `.dockerignore` so build context does not drag in `node_modules`, `.git`, and test fixtures. Final image: about **80 MB**.

The image went 1.9 GB → 80 MB, a 24× reduction. The build got faster too, for three compounding reasons: a smaller final image means fewer and smaller layers to export to the remote cache (so `cache-to` is cheaper), the runtime base pulls faster on a cold runner, and every environment that later pulls this image to deploy it pulls 80 MB instead of 1.9 GB — which is the deploy-time half of the win. This is the multi-stage and distroless discipline from [the production Dockerfile post](/blog/software-development/ci-cd/writing-a-production-dockerfile); I include it here because in CI, image size and build speed are not independent — the smaller image is the faster build and the faster deploy.

## 8. Putting it together: the fast, secure, reproducible CI build

We now have all four pieces. Stacked together they form a CI image-build job that is fast (remote cache + cache mounts + parallel stages), secure (rootless or kaniko, secret mounts not baked, OIDC-scoped push, no socket), and reproducible (digest-pinned base + `SOURCE_DATE_EPOCH`), feeding forward to scanning and signing. The figure maps each goal to the mechanism that delivers it.

![Matrix mapping the three goals fast, secure, and reproducible to their mechanisms remote cache with cache mounts, rootless build with secret mount, and digest pin with SOURCE_DATE_EPOCH, each with its measured payoff](/imgs/blogs/building-images-fast-and-securely-in-ci-8.png)

Here is the combined job — rootless BuildKit (to keep the cache features) on a self-hosted runner, with everything wired:

```yaml
name: image
on: { push: { branches: [main] } }
permissions:
  contents: read
  packages: write
  id-token: write   # OIDC for keyless push + signing

jobs:
  build:
    runs-on: self-hosted-rootless   # rootless buildkitd, no socket
    steps:
      - uses: actions/checkout@v4

      - name: Compute reproducible build time
        run: echo "SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)" >> "$GITHUB_ENV"

      - uses: docker/setup-buildx-action@v3
        with:
          driver: remote                  # talk to a rootless buildkitd
          endpoint: tcp://buildkitd:1234

      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: ${{ github.actor }}, password: ${{ secrets.GITHUB_TOKEN }} }

      - uses: docker/build-push-action@v6
        id: push
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=registry,ref=ghcr.io/${{ github.repository }}/cache
          cache-to: type=registry,ref=ghcr.io/${{ github.repository }}/cache,mode=max
          secrets: |
            npm_token=${{ secrets.NPM_TOKEN }}
          build-args: |
            SOURCE_DATE_EPOCH=${{ env.SOURCE_DATE_EPOCH }}
          outputs: type=image,rewrite-timestamp=true
          provenance: mode=max
          sbom: true

      # Forward to the next track post: scan + sign the image we just built.
      - name: Scan
        run: trivy image --exit-code 1 --severity HIGH,CRITICAL ghcr.io/${{ github.repository }}:${{ github.sha }}
      - name: Sign (keyless, OIDC)
        run: cosign sign --yes ghcr.io/${{ github.repository }}@${{ steps.push.outputs.digest }}
```

That one file is the whole post: `cache-from`/`cache-to ... mode=max` (fast), rootless `buildkitd` driver with no socket and a secret mount (secure), digest-pinned base in the `Dockerfile` plus `SOURCE_DATE_EPOCH` and `rewrite-timestamp` (reproducible), and the `trivy` scan + `cosign` sign that [the scanning post](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface) details. Build once, fast and clean, then promote that exact signed digest everywhere.

### Stress-testing the design

A design is only as good as its behavior under the bad cases. Let me push on it.

**What if the cache is cold (empty or evicted)?** The build falls back to a full cold build — correctly. The remote cache improves the *average*, never the worst case; never let a deploy depend on the cache being warm. A cold build should still *succeed*, just slowly.

**What if the cache backend is down mid-build?** `cache-from` failing to import must be a soft failure: BuildKit logs a warning and builds cold. `cache-to` failing to export must not fail the build either — the image is still correct without the cache being saved. Verify your CI does not hard-fail on a cache miss; some misconfigurations do.

**What if two PRs build at once and write the same cache ref?** With `sharing=locked` cache mounts and a content-addressed registry cache, concurrent writers are safe — last-writer-wins on the cache manifest, and since cache keys are content hashes, a "wrong" write is still a *valid* cache entry. Per-branch or per-platform `scope=` avoids cross-contamination.

**What if a secret leaks despite the secret mount?** The secret mount means it is not in the image, but it is still in CI's secret store. The defense in depth is short-lived credentials (OIDC tokens that expire in an hour) and scoping (a push token that can write one repo, not impersonate the runner), so a leak is bounded in both time and blast radius. Rotate on suspicion.

**What if the base image digest you pinned is later found vulnerable?** Pinning does not mean freezing forever — Renovate opens a PR to bump the digest, the scan gate catches known CVEs, and you ship the new pinned digest through the same pipeline. Pinning makes each build reproducible; the update flow keeps you patched.

**What if reproducibility breaks silently?** A nightly job that rebuilds the last release's commit and asserts the digest matches what shipped catches drift the moment a dependency or tool introduces nondeterminism — far better than discovering it during an incident.

## War story: the build environment as the soft target

The pattern in the worked example above is not hypothetical; it is the recurring shape of the worst software-supply-chain compromises. In the **SolarWinds** incident (2020), attackers did not tamper with the published source — they compromised the *build system* and injected the SUNBURST backdoor during the build of Orion, so the malicious code was present in the officially built, signed, and shipped artifact while the source repository looked clean. The lesson the industry took from it is the one this whole post is built on: the build environment is a privileged, high-value target, and "the source is clean" is not the same as "the artifact is clean" unless your build is reproducible and your build environment is hardened.

The **Codecov** breach (2021) is the more directly relevant cautionary tale for CI. Attackers altered Codecov's Bash uploader script — the kind of `curl ... | bash` line teams paste into their CI — so that when it ran inside customers' CI pipelines, it exfiltrated environment variables, which in CI means *secrets*: cloud keys, registry tokens, signing keys. Thousands of pipelines leaked credentials because a build step ran an unpinned, unverified script from the network with access to the job's full environment. Every fix in section 5 (pin and verify what you fetch) and section 4 (scope and shorten the credentials a build can reach) is a direct response to exactly this.

And the **dependency-confusion** research (2021) showed how a malicious package can run code in a build simply by being installed — pushing the attack surface all the way out to your transitive dependencies. Combine that with a build that mounts the Docker socket and you have the full chain from "a sub-dependency got compromised" to "a backdoored image got pushed under our identity." None of these required a zero-day in Docker or in the CI platform. They exploited the *convenient defaults*: trust the build with broad credentials, fetch from the network without pinning, and run the build with more privilege than it needs. The defensible posture is the opposite of the convenient one, which is precisely why it has to be engineered deliberately rather than left to the path of least resistance.

## How to reach for this (and when not to)

Every technique here has a cost, and a senior engineer's job is to know when the cost is not worth paying.

**Always do these, even on a small project:** pin your base image by digest, and never mount the host Docker socket into a CI job. Both are nearly free and both prevent a class of disasters. There is no project small enough to justify `/var/run/docker.sock` in CI.

**Do these as soon as build time hurts:** set up a remote cache (`gha` is zero-infra on GitHub Actions; `registry` cache everywhere else) and add cache mounts for your package managers. The payoff is immediate and the setup is an afternoon. Do *not*, however, reach for a bigger runner or build sharding before you have set up caching — caching is almost always the higher-leverage fix, and a sharded build with a cold cache is just paying for more cold builds in parallel.

**Do these when you have a real supply-chain posture to protect:** rootless/kaniko builds with OIDC-scoped push, secret mounts, and reproducible builds with verification. If you ship software other people run, or you are in a regulated space, this is table stakes. If you are a three-person startup shipping an internal tool, building rootless and pinning the base is plenty; you do not need a reproducible-build verification job on day one.

**Be skeptical of multi-arch by default.** If every target you deploy to is amd64, building arm64 via QEMU on every commit is pure waste. Build multi-arch only for the architectures you actually run or distribute, and prefer native runners over emulation once it is on your hot path.

**When NOT to over-engineer:** if you are on a managed PaaS that builds images for you (a buildpack-based platform, a Cloud Run source deploy), you may not own the build at all — and re-creating a hand-rolled BuildKit pipeline to replace it is usually a step backward. The techniques here are for teams that own their CI image build. Owning it is a real cost; make sure you are getting real value (speed, security, or reproducibility you actually need) before you take it on.

The deeper trade-off, the one worth internalizing: **speed, security, and reproducibility are not in tension in CI — they reinforce each other.** A reproducible build (pinned, lockfiled) has stable cache keys, which raises your cache-hit rate, which makes it fast. A rootless build that does not trust the host is also one that forces you to scope credentials, which is exactly what bounds a breach. The same hygiene that makes the build trustworthy makes it quick. That is why "fast" and "secure" stop being a choice once you build the pipeline correctly.

## Key takeaways

- **The cold-cache problem is the defining constraint of CI builds.** Ephemeral runners have no warm cache, so every build is cold until you give them a *remote* cache (`--cache-from`/`--cache-to`). The fix lives off the runner, not on it.
- **Use BuildKit and learn its mounts.** `RUN --mount=type=cache` persists dependency download caches across builds (distinct from the layer cache); `RUN --mount=type=secret` uses a token at build time without baking it into any layer. `--mount=type=ssh` does the same for private VCS access.
- **For multi-stage builds, remote cache with `mode=max`.** The default `mode=min` only caches final-image layers and silently fails to speed up your slowest, discarded build stages. Inline cache is the same trap.
- **Never mount the host Docker socket into a CI job.** It hands the job — and any poisoned dependency it runs — effective root on the runner. Use kaniko, buildah/podman, or rootless BuildKit: daemonless, unprivileged, no socket.
- **Scope and shorten the build's credentials.** OIDC-federated, one-hour, single-repo push tokens bound the blast radius of a compromised build to one job and one repo, not your whole registry and cloud.
- **Reproducibility is a supply-chain control, not a nicety.** Pin bases by digest, set `SOURCE_DATE_EPOCH` and rewrite layer timestamps, and pin/verify everything you fetch — so the artifact you scan and sign is provably the artifact you ship.
- **The cache-hit rate is the metric that pays the bill.** $\mathbb{E}[t] = h \cdot t_{\text{warm}} + (1-h)\,t_{\text{cold}}$ — protect $h$ with stable instruction order, lockfiles, and pinned bases. Measure how many layers were `CACHED` over a week.
- **Don't build multi-arch you don't deploy.** QEMU emulation can be 3–10× slower on CPU-heavy steps; reach for native per-arch runners only on the hot path, and only for architectures you actually run.

## Further reading

- [From commit to production: the CI/CD pipeline map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series map and the commit→build→test→package→deploy→operate spine this post sits inside.
- [The build stage: reproducible, fast, and cacheable](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) — the principle behind this post: why a build should be a pure function of its declared inputs, and the three levels of caching.
- [Writing a production Dockerfile](/blog/software-development/ci-cd/writing-a-production-dockerfile) — multi-stage builds, distroless bases, and a minimal attack surface for the image this job produces.
- [Image security scanning and a minimal attack surface](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface) — the next track post: gating on Trivy/Grype, SBOMs, and signing the digest this job built.
- [Securing the pipeline itself](/blog/software-development/ci-cd/securing-the-pipeline-itself) — treating the build environment and CI credentials as a primary attack surface, the lesson of SolarWinds and Codecov.
- The **BuildKit** documentation and the `docker buildx build` reference (cache backends, `--mount` types, multi-platform), and the **kaniko** project README for in-cluster, daemonless builds.
- The **SLSA** framework (supply-chain levels for software artifacts) and **Sigstore/cosign** docs for provenance, attestation, and keyless signing of the artifacts your pipeline produces.
- The **`SOURCE_DATE_EPOCH`** specification (reproducible-builds.org) for the cross-tool standard that makes builds time-deterministic.

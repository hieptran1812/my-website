---
title: "Build Once, Promote Everywhere: Artifacts and Versioning"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Build one immutable artifact, give it a content-addressed identity, and promote the same bytes through every environment — so the thing you ship to prod is byte-for-byte the thing you tested in staging."
tags:
  [
    "ci-cd",
    "devops",
    "artifacts",
    "versioning",
    "container-registry",
    "immutability",
    "semver",
    "traceability",
    "supply-chain",
    "docker",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-1.png"
---

The worst incident I ever signed off on looked perfect on paper. A change had merged on Tuesday, the build was green, the staging deploy was green, the full test suite passed against staging, the product owner clicked approve, and on Thursday we promoted to production. Within ninety seconds, the checkout service started returning 500s on roughly one in twenty requests. We rolled back, the pages stopped, and then we spent the rest of the afternoon arguing about something that should have been impossible: how does a change that passed every gate against staging fail only in production? The code was identical. The configuration was nearly identical. The traffic shape was similar. Eventually one of the platform engineers diffed the two images byte-for-byte and found the answer. They were not the same image. Our pipeline built a fresh image for each environment. The staging image had been built on Tuesday; the production image had been built on Thursday at promotion time. In the forty-eight hours between, a transitive JSON-parsing dependency had published a patch release, our lockfile was looser than we thought, and the Thursday build pulled the new patch. The new patch had a regression in how it handled a trailing comma. We never tested the thing we shipped. We tested a *different program that happened to share a git commit*.

That is the entire reason this post exists, and it is one of the two governing principles of this whole series. The principle is four words long: **build once, promote everywhere.** Build the artifact a single time, give it an immutable identity, and then move that *exact same artifact* — the same bytes — through every environment on its way to production. Do not rebuild for staging. Do not rebuild for production. The artifact that production runs must be, to the byte, the artifact that your tests ran against in staging. When you rebuild per environment, you reintroduce uncertainty at the single riskiest moment in the entire lifecycle: the promotion to prod. You take a thing you have validated and you replace it, sight unseen, with a freshly fabricated thing that nobody tested, and you do it under the banner of "it's the same code." It is not the same artifact, and the artifact is what runs.

This post is part of the series **"CI/CD & Cloud-Native Delivery, From Commit to Production."** If you have not read [the CI/CD pipeline map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model), start there — it lays out the spine we keep returning to: **commit → build → test → package → deploy → operate**, governed by two principles, **"build once, promote everywhere"** and **"everything as code,"** measured by the four DORA metrics. The companion principle, building the artifact reliably in the first place, lives in [the build stage post](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable). This post is about what happens *after* you have an artifact: how you give it an identity you can trust, where you store it, how you version it for humans and for machines, and how you move it through environments without ever quietly swapping it for a different one. The figure below is the whole thesis in one image — three builds producing three artifacts on the left, one build producing one artifact promoted three times on the right.

![Before and after comparison showing a rebuild-per-environment pipeline producing three different artifacts where the production one is broken next to a build-once pipeline producing one artifact promoted unchanged to staging and production](/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-1.png)

By the end you will be able to: explain to a skeptical teammate exactly why rebuilding per environment is dangerous and back it with a reproducibility argument; assign your artifacts a content-addressed identity and deploy by that identity instead of by a mutable tag; design a versioning scheme that serves both humans (semantic versioning) and machines (the digest, the commit SHA); configure a container registry for immutability and sane retention; build a GitHub Actions pipeline that builds and pushes exactly once and then promotes by digest; pin a Kubernetes Deployment to a `sha256` digest; and walk the full traceability chain from a running pod back to the pull request that changed it. None of this is exotic. All of it is the difference between a delivery system you can reason about and one that surprises you on a Thursday afternoon.

## 1. The rebuild-per-environment anti-pattern, and why it is dangerous

Let me define the core terms first, because the accessibility of everything that follows depends on them. An **artifact** is the immutable output of your build — the thing you actually ship. It is not your source code; it is the compiled, packaged result of running your build against that source: a container image, a JAR, a Python wheel, a static binary, an npm tarball. A **registry** (sometimes "artifact store" or "binary repository") is the server you push artifacts to and pull them from: a container registry like GHCR, Amazon ECR, or Google Artifact Registry; a package registry like npm or a Maven repo; a general binary repo like JFrog Artifactory or Sonatype Nexus. A **promotion** is the act of taking an artifact that has earned confidence in one environment and moving it to a more production-like environment. Hold those three words — artifact, registry, promotion — because the principle is just a sentence about them: *build the artifact once, push it to the registry once, and promote that one artifact through your environments without rebuilding.*

The anti-pattern is the opposite, and it is everywhere because it feels harmless. A rebuild-per-environment pipeline has a deploy job for each environment, and each deploy job *builds* before it ships. The dev deploy checks out the code and builds an image and deploys it. The staging deploy checks out the same commit and builds an image and deploys it. The production deploy checks out the same commit and builds an image and deploys it. Three deploys, three builds, three artifacts. People reach for this because it is the obvious way to wire a pipeline — "to deploy, build and deploy" — and because at small scale the three builds usually do produce the same thing, so the rare time they differ feels like a freak accident rather than a structural flaw. It is not a freak accident. It is the structural flaw producing its inevitable consequence.

Here is the mechanism, made concrete. A build is a function. Ideally it is a *pure* function of its declared inputs — the source at a commit, the pinned dependencies, the toolchain version — and if it were truly pure, rebuilding would be safe, because every rebuild would produce identical bytes. But almost no real build is pure. A typical build reads inputs it never declared: the base image referenced by a floating tag like `node:20` (which points at a different underlying image when the upstream pushes a new `20.x`), transitive dependency versions resolved at build time against a registry whose contents change, the compiler or toolchain version installed on whatever runner happened to pick up the job, the system libraries it links against, the locale, and — astonishingly often — the wall-clock time, baked into the artifact as a build timestamp. None of those are pinned to your git commit. So when you build the *same commit* at two different moments on two different runners, you can and eventually will get two different artifacts. The git commit is a label on a box; rebuilding repacks the box, and you do not get to inspect what changed.

Now layer the environments on top. You build for staging on Tuesday and the staging artifact, by luck, links against `libfoo 1.4.2` and `node:20.11.0`. Your tests run against that artifact and pass. On Thursday you build for production. Between Tuesday and Thursday, `libfoo` published `1.4.3` and the `node:20` tag advanced to `20.11.1`. The production build, resolving the same loose constraints, pulls `1.4.3` and `20.11.1`. Now production is running an artifact that *no test in your pipeline ever saw.* Your green staging run certified `1.4.2`; production runs `1.4.3`. When `1.4.3` has a regression, you get the "it passed staging but broke prod" incident, and you will burn hours looking at the code, because the code is innocent. The defect is in the *build drift* — the gap between the artifact you tested and the artifact you shipped — and that gap was opened by the decision to rebuild. The deeper this drift goes (a base-image bump, a transitive patch, a toolchain upgrade on the runner fleet), the more spectacular the prod-only failure.

It helps to enumerate the hidden inputs explicitly, because "the build is not pure" is easy to nod along to and hard to feel until you list what actually leaks in. A non-exhaustive inventory of the things a typical build reads without your lockfile saying so: the base image behind a floating tag (`node:20`, `python:3.12`, `ubuntu:latest` — every one of these is a moving target); transitive dependencies resolved against a live package index whose contents change hourly; the toolchain version baked into the runner image (GitHub's `ubuntu-latest` advances its Go, Node, and system-package versions on its own schedule); apt/apk package versions pulled by a `RUN apt-get install` with no version pin; the linker and libc the binary links against; CA certificate bundles; locale and timezone; the order files are added to a layer (which can change a hash even when content is identical); and the wall-clock build time, which countless build systems stamp into the artifact "for convenience." Each of these is an input to the build function that is *not* derived from your git commit. Reproducible-build projects exist precisely because eliminating all of them is hard work — Debian's reproducible-builds effort and Nix and Bazel's hermetic sandboxes are entire engineering programs aimed at making the build a pure function. If achieving byte-identical rebuilds is hard enough to need a dedicated movement, you should not casually assume your CI gets it for free. The pragmatic conclusion is not "make every build perfectly reproducible" — that is a worthy but expensive goal — it is "build once so you never have to bet on reproducibility at the moment it matters most."

There is a second, quieter cost: you lose the ability to answer "what is actually running in production?" with certainty. If production was built at promotion time, the only record of *what went in* is whatever the registries and the network looked like at that moment, and that is not recoverable. You have a git commit, which tells you what *should* have been built, but you have no guarantee that is what *was* built. Auditors hate this. Incident responders hate this more, because at 3 a.m. the question is never "what does the code say" — it is "what is running, and how do I get back to a version that worked." A rebuild-per-environment pipeline cannot answer either question crisply. This is why the principle is non-negotiable for anything past a toy: rebuilding per environment trades a structural guarantee (you ship what you tested) for a convenience (the pipeline is easier to wire), and the trade is bad.

A third cost is pure waste, and it shows up directly in two DORA metrics. Every redundant build burns runner minutes and money. If a build takes eight minutes and you rebuild for dev, staging, and prod, you spend twenty-four minutes of compute where four-to-eight would do, and you do it on the critical path to production, inflating lead time. At a hundred deploys a day that is roughly sixteen extra build-hours daily — a runner fleet you are paying for to manufacture risk. Worse, the rebuilds are usually *serialized* with the gates between them (build, test, approve, rebuild, deploy), so the redundant builds do not even overlap; they stretch the wall-clock time from merge to production. Build-once collapses the three-or-four builds into one and turns the downstream steps into fast pulls, which is why teams that adopt it often see their lead-time number drop noticeably even before they touch anything else. The convenience of rebuild-per-env is not free; you pay for it in compute, in lead time, and in incidents. The whole rest of this post is the machinery for doing it the other way.

To see how completely this inverts the usual instinct, it is worth naming the assumption the anti-pattern rests on: "the deploy *is* a build." That equation feels natural — to put code somewhere, you build it there — but it is precisely the bug. In a build-once world, deploy and build are different verbs done at different times by different jobs. Build is manufacture; it happens once, early, and produces a thing. Deploy is logistics; it happens many times, later, and *moves* the thing. Once you internalize that split, the rest of this post stops being a list of techniques and becomes obvious consequences of one idea: there is exactly one artifact, and everything after the build is just deciding where to put it.

## 2. The immutable artifact: build it, name it, never touch it again

The fix is to make the artifact a *first-class, immutable thing* with a life of its own, decoupled from the act of building. The build runs once, early, before any environment is involved. It produces an artifact. That artifact is given an identity, pushed to a registry, and from that moment it is frozen — nobody ever rebuilds it, edits it, or republishes a different artifact under the same identity. Every environment, from dev all the way to prod, *pulls the same artifact* and runs it. The build is a thing that happens once at the top of the pipeline; everything downstream is movement, not manufacture.

Put the verb shift plainly: a deploy is no longer "build the code for this environment and run it." A deploy is "fetch artifact X from the registry and run it here, with this environment's configuration injected." The artifact does not know or care which environment it is running in. That last clause is load-bearing and we will return to it in section 5 — the artifact must be environment-agnostic, which means no environment-specific values (database URLs, feature flags, secrets, log levels) may be baked into it at build time. If you bake the staging database URL into the image, you cannot promote that image to prod, and you are right back to rebuilding. The artifact carries *code*; the environment supplies *config*. This is the Twelve-Factor App's third factor — "store config in the environment" — and it is precisely what makes one artifact promotable everywhere.

The figure below shows the build-once flow as a graph: one build, one push to the registry, and three deploys that each pull the same digest. Notice there is exactly one arrow into the registry — the single push — and three arrows out, one per environment. That shape *is* the principle.

![Graph diagram showing a commit feeding a single build that pushes one immutable digest to the registry which then fans out to dev staging and production deploys that all pull the same digest](/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-2.png)

What does it mean for an artifact to be "immutable"? Two things, and they reinforce each other. First, the *bytes* are fixed: once built and pushed, the artifact's content never changes. Second, the *identity* is bound to those bytes, so a given identity always refers to the same bytes forever. The second property is what lets immutability survive contact with reality, and it is the subject of the next two sections, because the naïve way of naming artifacts — a human-chosen tag like `v1.2.3` or `latest` — does *not* guarantee it. A tag is a pointer, and a pointer can be moved. The only identity that is truly welded to the bytes is one *derived from* the bytes: a content-addressed digest. Get the identity wrong and the whole edifice wobbles, because "promote the same artifact" silently degrades into "promote whatever the tag points to right now," which is exactly the gap you were trying to close.

It is worth stating the upside in DORA terms before we go deeper, because the principle pays off in measurable delivery performance. The four DORA metrics are deploy frequency, lead time for changes, change-failure rate, and time-to-restore. Build-once-promote-everywhere most directly attacks **change-failure rate**: a large fraction of "the deploy broke prod" incidents are build-drift incidents, and build-once eliminates that class entirely — prod runs the tested bytes, so prod cannot fail in a way staging could not have caught for build-drift reasons. It also helps **time-to-restore**: rollback becomes "deploy the previous digest," a fast and deterministic operation, instead of "rebuild the previous commit and hope it produces what it produced last time." And it helps **lead time**, because you build once instead of three or four times, removing redundant build minutes from the critical path. The principle is not just hygiene; it moves the numbers.

## 3. The artifact's identity: content-addressed digests, not mutable tags

Here is the single most important technical idea in this post. A container image — and most modern artifacts — has a *content-addressed digest*: a `sha256` hash computed over the artifact's content (its manifest, which in turn references its layers by their own digests). The digest looks like `sha256:9f2b7c4e...`. It is not a name you choose; it is a fingerprint *derived from* the bytes. Change a single byte of the artifact and the digest changes completely. This gives you the property you actually want from an identity: **the digest cannot lie.** If two artifacts have the same digest, they are byte-for-byte identical, end of discussion. If the digest is `sha256:9f2b7c4e...`, there is exactly one artifact in the universe that bears it. The digest *is* the artifact, compressed into a fingerprint.

Contrast that with a **tag**. A tag like `v1.2.3` or `myapp:latest` is a *human-chosen, mutable pointer*. It is a name that the registry maps to a digest, and that mapping can be changed. You can push a new image and re-point `v1.2.3` at it (most registries let you, by default). You can — and `latest` does this constantly — move the tag every time you push. So a tag answers the question "what does this name point to *right now*," not "what bytes am I getting." The digest answers the second question, permanently. The figure below shows the two identities stacked: the human-facing semver and tag at the top, the machine-facing commit SHA and digest below, all resolving down to the fixed image bytes.

![Stack diagram showing the layered identities of an artifact from a human-facing semver and mutable tag down through the commit SHA to the content-addressed digest and the fixed image bytes](/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-3.png)

This is why `latest` is dangerous in production and why mutable tags in general are a footgun. When your production manifest says `image: myapp:latest`, you have not specified an artifact — you have specified a *query* that the registry resolves at pull time to whatever `latest` points to *at that moment.* If someone pushes a new build and the CI tags it `latest` between the time you tested and the time the pod restarts, your pod will pull the new image. You did not deploy a new version; the version deployed *itself* because the pointer moved under you. Even semver tags are mutable by default: nothing stops a sloppy pipeline (or a malicious actor) from re-pushing `v1.2.3` with different bytes. The tag is a convenience for humans; it is not a safe deployment target.

The discipline, then, is: **tags are for humans to read; digests are for machines to deploy.** You may tag an image `v1.2.3` and `git-abc123` for humans to find and reason about it. But the thing your deployment pins — the thing your Kubernetes manifest or Argo CD application references — should be the `sha256` digest. When you write `image: myapp@sha256:9f2b7c4e...`, you have specified an artifact, not a query. No push, no tag move, no race can change what that pod runs. It is welded to the bytes. The table below compares the schemes head to head, and it is worth internalizing the bottom row: the digest is the only scheme that is immutable, traceable, and safe in prod at the same time.

| Identity scheme | Immutable? | Traceable to a build? | Safe to deploy in prod? |
|---|---|---|---|
| `myapp:latest` | No — moves on every push | Weak — which build is "latest" now? | No — the tag moves under you |
| `myapp:v1.2.3` (semver tag) | Only by policy / registry config | Yes, if you never re-push it | OK only if the registry locks it |
| `myapp:git-abc123` (commit SHA tag) | One tag per commit | Yes — maps straight to the commit | Acceptable; still a tag, can be re-pushed |
| `myapp@sha256:9f2b...` (digest) | Always — derived from bytes | Yes — via the build's provenance | Best — welded to the exact bytes |

![Matrix comparing four artifact identity schemes across whether they are immutable traceable and safe in production showing the digest winning every column](/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-5.png)

It is worth understanding *why* the digest cannot be faked, because the guarantee rests on a real cryptographic property and not on registry good behavior. An OCI image is a small JSON manifest that lists the config blob and the layer blobs, each referenced by its own `sha256` digest and byte size. The image's digest is the `sha256` hash of that manifest's exact bytes. So the manifest digest commits to the layer digests, and each layer digest commits to that layer's bytes — a chain of hashes, a Merkle structure. To produce a different artifact under the same digest, an attacker would have to find a second, different manifest (or layer) that hashes to the identical `sha256` value: a preimage or collision against SHA-256, which is computationally infeasible with anything we can build today. That is the whole game. The digest is not a label the registry promises to keep honest; it is a fingerprint that the *content itself* determines, and any tampering — a swapped layer, an injected backdoor, a single flipped bit — changes the bytes, which changes a layer digest, which changes the manifest, which changes the image digest. This is exactly the property that lets you verify an image you pulled is the image you meant to pull, and it is why content-addressing is the bedrock the supply-chain world (cosign, SLSA, in-toto) stands on. A tag promises nothing cryptographically; a digest is checkable math.

A practical note that trips people up: the digest is computed over the image *manifest*, and a multi-architecture image (a "manifest list" or "image index") has its own digest that points at per-architecture manifests with their own digests. When you pin `myapp@sha256:...` to the index digest, you get the right per-arch image automatically — the runtime picks `linux/amd64` or `linux/arm64` from the index. So pinning by digest works fine for multi-arch images; you pin the index digest and let the platform resolve the leaf. The point stands either way: the digest, at whatever level you pin, is content-addressed and cannot move.

One more nuance that bites teams who adopt digests halfway: when you build with BuildKit and push, the digest you should record is the *registry* digest — the one the registry returns after the push, the hash of the manifest as stored — not a local image ID. The local image ID (`docker images --no-trunc`) is a hash of the local image config, and it generally does *not* equal the pushed manifest digest, because the push can recompress or canonicalize layers. The reliable source of the deployable digest is the push step itself: `docker buildx build --push` and `docker/build-push-action` both surface the pushed manifest digest as an output, and `docker buildx imagetools inspect ghcr.io/acme/app:tag` or `crane digest ghcr.io/acme/app:tag` resolves a tag to its registry digest. Pin *that* string. Recording the local image ID and assuming it is the registry digest is a classic way to think you are deploying by digest while actually deploying nothing pin-able — the deploy fails to find the image, or worse, silently falls back to a tag.

## 4. What "promotion" actually means

If you take rebuilding off the table, "deploy to production" stops being a build operation and becomes a *movement* operation. This is the mental shift that makes the principle operational. A promotion is: take artifact X, which has earned confidence at environment N, and run it at environment N+1, which is more production-like. Each environment in the chain adds confidence by subjecting the *same artifact* to a more realistic test. Dev runs smoke tests against it. Staging runs the full integration suite against it, ideally with production-shaped data and traffic. A canary runs it against a sliver of real production traffic while you watch the metrics. Then production runs it at full scale. At no point in that chain does the artifact change. Staging is testing the *production artifact* — that is the entire value of staging — and a canary is the production artifact meeting production reality, in miniature, before it meets all of it.

The figure below draws the promotion as a timeline: one build at the left, then dev, staging, canary, and full production, with the same digest carried the whole way. Read it left to right and notice that the artifact's identity never changes; only the environment around it gets more real.

![Timeline showing one artifact built once then promoted through dev staging a one percent canary and full production with the same digest carried unchanged across every stage](/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-6.png)

The corollary is that the *differences* between environments cannot live in the artifact. They have to live somewhere — staging and prod genuinely do point at different databases, use different feature-flag values, run at different replica counts, and hold different secrets — so if not in the artifact, where? The answer is: injected at deploy time, per environment, as configuration. This is the Twelve-Factor "config in the environment" rule again, and it is the load-bearing companion to build-once. The artifact is *parameterized over* its environment; the deploy supplies the parameters. In Kubernetes this is `ConfigMap`s and `Secret`s mounted into the pod or surfaced as environment variables; the artifact reads `DATABASE_URL` from the environment rather than having a database URL compiled in. (The mechanics of doing this safely — especially secrets — are a topic of their own; a sibling post planned for this series, `configuration-and-secrets-in-kubernetes`, goes deep on it. The rule for *this* post is simply: nothing environment-specific in the artifact, or you cannot promote it.)

The cleanest way to hold all of this in your head is the Twelve-Factor App's fifth factor, "Build, release, run," which splits the lifecycle into three stages that must stay distinct. The **build** stage turns source into an artifact — once. The **release** stage combines that artifact with a specific environment's config to produce a release — a deployable unit that is "artifact X plus the prod config of June 22." The **run** stage executes a release. The factor's rule is that the stages are strictly separated and you cannot change code at run time (to do so you would have to go back to build). Build-once-promote-everywhere is just this factor taken seriously: one build many releases, where each release is the same artifact bound to a different environment's config. Promotion, in this vocabulary, is "create a new release of the already-built artifact for the next environment." It is a release operation, never a build operation. Naming it this way also kills a common confusion — "release" and "deploy" get used interchangeably, but the release is the *artifact-plus-config* bundle, and the deploy is the act of running it; you can re-deploy the same release (a restart, a reschedule) without producing a new one.

Here is what config-injection looks like end to end, so "config in the environment" is not abstract. The same image reads `DATABASE_URL`, `LOG_LEVEL`, and `FEATURE_NEW_CHECKOUT` from its environment. Staging supplies one set of values, prod another, and neither value is anywhere in the image:

```yaml
# staging overlay — same image digest, staging config
apiVersion: v1
kind: ConfigMap
metadata: { name: app-config, namespace: staging }
data:
  LOG_LEVEL: "debug"
  FEATURE_NEW_CHECKOUT: "true"   # on in staging so we test it
  DATABASE_URL: "postgres://db.staging.svc:5432/app"
---
# production overlay — SAME image digest, prod config
apiVersion: v1
kind: ConfigMap
metadata: { name: app-config, namespace: production }
data:
  LOG_LEVEL: "info"
  FEATURE_NEW_CHECKOUT: "false"  # off in prod until we flip it
  DATABASE_URL: "postgres://db.prod.svc:5432/app"
```

The image is byte-identical across both; only the `ConfigMap` differs. The feature flag is the textbook example of why this matters: you promote the *same artifact* that contains the new checkout code, and you turn the behavior on in staging and off in prod, decoupling "ship the code" from "enable the behavior." That decoupling — code in the artifact, behavior gated by config — is what lets you deploy continuously without every deploy being a feature release, and it is impossible if the flag's value is compiled in.

This reframes the deploy pipeline. Instead of one pipeline that builds-and-deploys per environment, you have a build pipeline that runs once on merge, and a *promotion* pipeline (or several gated promotion steps) that move a known digest forward. "Deploy to prod" is literally "set the prod environment to run digest `sha256:9f2b...`." If you do GitOps, that is a one-line change to the prod overlay's pinned digest, committed to git, and the cluster reconciles to it — the deploy is a git commit changing a digest, which is about as auditable as a deploy can get. We are not going to fully derive GitOps here (the [pull-based GitOps model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) and a dedicated multi-environment post, `multi-environment-promotion-dev-staging-prod`, cover the reconcile loop), but notice how cleanly build-once composes with it: GitOps wants a desired state that is a specific immutable artifact, and a digest is exactly that.

There is a subtle environment-parity trap worth flagging. Build-once guarantees the *artifact* is identical across environments, but it does not guarantee the *environment* is. If staging runs Postgres 14 and prod runs Postgres 13, or staging has 2 replicas and prod has 50, or staging's secrets are stubbed and prod's are real, you can still get a prod-only failure even with identical bytes. Build-once removes the *build-drift* class of prod surprises; it does not remove *environment-drift* surprises. The honest framing is: build-once is necessary but not sufficient for "tested == shipped." You also need environment parity, which is why the promotion chain deliberately makes each environment more production-like than the last. The two disciplines together — same artifact, increasingly-real environment — are what shrink the gap between what you tested and what you run.

## 5. Versioning: a human face and a machine face

An artifact needs a version, and the mistake people make is thinking they need *one* versioning scheme. They need two, serving two audiences, and conflating them causes most versioning pain. There is a **human-facing version** that communicates compatibility and intent — this is **semantic versioning** — and a **machine-facing identity** that is precise, immutable, and traceable — this is the commit SHA and, ultimately, the digest. The human version is a promise; the machine identity is a fact. You want both, and you want them *linked*, but you should not try to make one do the other's job.

**Semantic versioning** (semver) is `MAJOR.MINOR.PATCH` — for example `2.4.1`. The contract is about *compatibility*, and it is a promise to whoever consumes your artifact: bump **PATCH** (`2.4.1` → `2.4.2`) for backward-compatible bug fixes; bump **MINOR** (`2.4.1` → `2.5.0`) for backward-compatible new features; bump **MAJOR** (`2.4.1` → `3.0.0`) for breaking changes. The whole point of semver is that a consumer can read the version delta and know whether upgrading is safe. Semver shines for things other people depend on — published libraries, public APIs, shared platform components — because the compatibility promise is the product. For a leaf application service that only your own pipeline deploys, semver is less essential (nobody is reading your app's version to decide whether to upgrade a dependency), which is why many teams version *libraries* with semver and *services* with something cheaper.

That cheaper thing is often the **commit SHA** as the internal version, sometimes paired with a build number. The git commit SHA (`abc123def...`) is already an immutable, globally unique identifier for "exactly this source." Tagging an artifact with its commit SHA gives you a version that is trivially traceable — given the SHA you can `git show abc123` and see precisely what built it — and that is immune to the "what does v1.2.3 actually contain" ambiguity. Many teams use a hybrid: a semver for the human-facing release plus the SHA for the machine-facing precision, e.g. tag the image `v2.4.1` *and* `git-abc123def` *and* deploy by its digest. The semver answers "what release is this?"; the SHA answers "what source is this?"; the digest answers "what bytes are these?". Three questions, three identifiers, all pointing at one artifact.

**Calendar versioning** (CalVer) is a third option worth knowing: `YYYY.MM.PATCH` or `YYYY.0M.MICRO`, e.g. `2026.06.1`. CalVer makes sense when *time* is the meaningful axis — products that ship on a cadence rather than by feature-compatibility (Ubuntu's `26.04`, many SaaS products), where "how old is this" matters more than "is this a breaking change." CalVer and semver are not rivals so much as answers to different questions; pick the one whose axis (compatibility vs. recency) matches what your consumers actually need to reason about.

The table below lays the schemes side by side, because the right choice is entirely about *who reads the version and why*. The trap is treating any one of these as the single identity; in practice a healthy service carries a human-facing version *and* a machine-facing digest, and the rows below are about the human-facing label only.

| Scheme | Example | What it communicates | Best for | Weakness |
|---|---|---|---|---|
| Semver | `2.4.1` | Compatibility — safe to upgrade or not | Libraries, public APIs, shared components | Requires discipline; a lying semver misleads consumers |
| CalVer | `2026.06.1` | Recency — how old is this release | Cadence-shipped products, OS distros, SaaS | Says nothing about compatibility |
| Commit SHA | `git-abc123def` | Exact source — which commit built it | Internal leaf services, machine traceability | Opaque to humans; no compatibility signal |
| Build number | `#4821` | Build ordering — which CI run | Pairing with another scheme | Meaningless without the build system as context |
| Digest | `sha256:9f2b…` | Exact bytes — which artifact | The deploy target, always | Unreadable; not a human release label |

The rule that falls out of the table: pick the *human* scheme by audience (semver if someone depends on your compatibility promise, CalVer if recency is the axis, the SHA if only your own pipeline reads it), and always carry the *machine* identity (SHA plus digest) regardless. They are not competing; they answer different questions and a mature artifact wears several at once.

The other half of versioning is **version-from-git**, also called tag-driven releases. Rather than hand-editing a version string in a file (which drifts, gets forgotten, and lies), derive the version from git: an annotated git tag `v2.4.1` *is* the release, and tooling like `git describe` or a release action reads the tag to produce the version. The flow is: cut a release by pushing a git tag; the pipeline sees the tag, builds the artifact, stamps it with the tag's semver, and publishes. Git is the source of truth for "what version is this," which means the version is reviewable, immutable (tags should be protected), and never out of sync with the code, because it *is* the code's tag. This composes with everything-as-code: the release is a git operation, auditable like any other.

Finally — and this is the part teams skip and regret — **embed the version in the artifact and expose it at runtime.** Bake the semver, the commit SHA, and ideally the build timestamp and the digest into the image at build time (as labels and/or an embedded build-info file), and expose them on a `/version` (or `/healthz` with build info) HTTP endpoint. Now the question "what is actually running in production?" has a one-curl answer: `curl https://api.example.com/version` returns `{"version":"2.4.1","commit":"abc123","builtAt":"2026-06-22T...","digest":"sha256:9f2b..."}`. When an incident starts, you do not guess what is deployed; you ask the running process. This is the runtime end of the traceability chain we build out in section 8, and it costs about ten lines of code to wire up. Here is the embedding, done with build args and OCI labels:

```dockerfile
# multi-stage build; the version metadata is passed in at build time
FROM golang:1.22 AS build
WORKDIR /src
COPY . .
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_TIME=unknown
# link the values into the binary so /version can return them
RUN CGO_ENABLED=0 go build \
      -ldflags "-X main.version=${VERSION} -X main.commit=${COMMIT} -X main.buildTime=${BUILD_TIME}" \
      -o /out/app ./cmd/app

FROM gcr.io/distroless/static:nonroot
COPY --from=build /out/app /app
# OCI labels record the same metadata on the image itself, queryable with `docker inspect`
ARG VERSION=dev
ARG COMMIT=unknown
LABEL org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.source="https://github.com/acme/app"
USER nonroot
ENTRYPOINT ["/app"]
```

The labels (`org.opencontainers.image.revision`, `.version`, `.source`) are the OCI standard annotations; tooling reads them, and they travel with the image into the registry. The `-ldflags -X` trick stamps the values into the Go binary so the running process can serve them. The same idea applies in any language — a generated `build_info.json`, a `__version__` set at build time, a baked-in resource. The principle is invariant: the artifact should be able to tell you what it is.

The runtime half is a handler that returns the stamped values. It is genuinely about ten lines, and it is the single highest-leverage piece of incident tooling you will ever write:

```go
package main

// these are populated at build time by -ldflags -X (see the Dockerfile above)
var (
	version   = "dev"
	commit    = "unknown"
	buildTime = "unknown"
)

// GET /version -> {"version":"2.4.1","commit":"abc123","builtAt":"...","digest":"..."}
func versionHandler(w http.ResponseWriter, r *http.Request) {
	// digest is supplied to the pod as an env var by the deploy (it is not knowable at build time)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"version": version,
		"commit":  commit,
		"builtAt": buildTime,
		"digest":  os.Getenv("IMAGE_DIGEST"),
	})
}
```

One subtlety: the *digest* is the one piece of identity the artifact cannot know about itself at build time, because the digest is computed *after* the build, by the push. So the digest is best supplied to the pod as an environment variable at deploy time — the deploy that pins `app@sha256:9f2b...` can also set `IMAGE_DIGEST=sha256:9f2b...` in the pod spec, closing the loop. Now `curl /version` returns the commit (stamped at build) and the digest (injected at deploy), and from that one response you can walk the entire traceability chain in section 8.

## 6. Where artifacts live: registries, immutability, and retention

An artifact has to live somewhere between "built" and "deployed," and that somewhere is a registry. The registry is not a passive file store; it is where you *enforce* the properties this whole post depends on — immutability and traceability — and where you *manage the cost* of keeping artifacts around. There are three broad families, and most organizations run all three. The figure below lays them out as a taxonomy.

![Tree diagram of artifact stores branching into container registries package registries and binary repositories with examples and their immutability and garbage collection controls](/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-8.png)

**Container registries** store OCI images: GitHub Container Registry (GHCR), Amazon ECR, Google Artifact Registry, Azure Container Registry, and self-hosted Harbor. **Package registries** store language packages: npm, PyPI-compatible registries, Maven repos, NuGet, RubyGems. **Binary repositories** are the generalists — JFrog Artifactory and Sonatype Nexus — which can hold images, packages, Helm charts, raw binaries, and more under one roof with one access-control and retention model. The choice is mostly about ecosystem and operational preference; the *disciplines* you apply are the same across all three.

The first discipline is **immutability settings.** A registry that lets you re-push an existing tag with different bytes has handed you a footgun: it means `v1.2.3` is not actually a stable reference, and your "build once" can be silently undone by an "oops, re-push." Good registries let you mark a repository or tag as immutable so that once a version is published, it cannot be overwritten — only a new version can be published. ECR calls this tag immutability; GHCR and Artifactory and Nexus all have equivalents. Turn it on for anything you deploy. Here is the ECR repository configured for immutable tags and scan-on-push, in Terraform:

```hcl
resource "aws_ecr_repository" "app" {
  name = "acme/app"

  # once a tag is pushed it cannot be moved or overwritten — this enforces "build once"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
  }
}
```

The second discipline is **retention and garbage collection.** Build-once does not mean keep-forever. A busy pipeline can produce hundreds of images a week, and storage costs money, registry list operations slow down, and old vulnerable images linger as an attack surface if you never clean up. So you set a retention policy: keep the last N images per tag prefix, keep anything tagged with a release semver indefinitely, keep anything currently referenced by a running deployment, and garbage-collect untagged and old images. The critical nuance — and the source of a genuinely scary incident class — is *do not garbage-collect an image that is still deployed or that you might need to roll back to.* If your rollback strategy is "deploy the previous digest" and your retention policy deleted that digest last night, your rollback fails at the worst possible moment. Retention policies must protect both currently-deployed digests and a rollback window. Here is an ECR lifecycle policy that keeps releases, keeps recent untagged images for a rollback window, and expires the rest:

```json
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep semver release images indefinitely",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["v"],
        "countType": "imageCountMoreThan",
        "countNumber": 9999
      },
      "action": { "type": "expire" }
    },
    {
      "rulePriority": 2,
      "description": "Keep the last 20 untagged build images as a rollback window",
      "selection": {
        "tagStatus": "untagged",
        "countType": "imageCountMoreThan",
        "countNumber": 20
      },
      "action": { "type": "expire" }
    }
  ]
}
```

There is a sharp edge in GC worth dwelling on, because it is the kind of bug that lies dormant for months and then eats your rollback. Untagged-image cleanup is the most common retention rule, and it is fine until it races your deploy. Consider the sequence: your deploy pins images by digest, so the *tag* on a recently-built image may get reused or removed, leaving the image untagged-but-deployed. If your GC rule is "expire untagged images older than N days" and it does not also check "is this digest currently referenced by a live deployment?", it will eventually delete an image that a running pod (or a rollback target) depends on. The pods that are already running are fine — the bytes are on the node — but the moment one reschedules, the pull fails, and your rollback target is simply gone. The fix is to make retention *deployment-aware*: before expiring, reconcile against the set of digests live across every cluster and environment, and against a rollback window, and never expire anything in that union. Some registries can do this with referrer/subject awareness; otherwise a small nightly job that lists live deployments and pins their digests as protected is the pragmatic guard. The principle: GC may delete artifacts nobody points at, never artifacts something points at.

#### Worked example: sizing the retention window with real numbers

Put numbers on it so the policy is not guesswork. Suppose the team merges to main about 40 times a day, each merge builds one image, and an image is roughly 90 MB. That is 40 images/day, ~3.6 GB/day, ~25 GB/week of new images. Keep everything forever and you add ~1.3 TB/year per service — multiply by 30 services and the registry bill and the list-operation latency both become real problems. Now size a sane policy: keep all semver-tagged releases indefinitely (a few dozen a year, cheap, and these are your "known good" anchors); keep the last 30 untagged build images as a rollback window (30 images ≈ 2.7 GB, covering most of a busy day's merges — comfortably more than the handful of versions you would ever realistically roll back through); and protect every digest currently live in any environment regardless of age. The rest expires. The arithmetic that matters is the rollback-window sizing: if you deploy ~40 times a day and your worst realistic "roll back through several bad releases" scenario spans, say, the last 5–8 deploys, a 30-image window gives you a 4–8× safety margin while bounding storage to single-digit gigabytes. Make the window a deliberate number derived from your deploy rate and your rollback depth, not a default copied from a blog post — too small and you lose rollback targets, too large and you pay to store risk.

The third discipline is **promotion across registry paths.** Some organizations physically separate a "candidate" registry path from a "released" registry path: CI pushes every build to `registry/dev/app`, and promotion *copies* the exact bytes (by digest) to `registry/prod/app` once it passes the gates, often with a tighter access-control boundary on the prod path so that only the promotion process can write there. The copy is byte-for-byte — `crane copy` or `skopeo copy --all` moves the manifest and layers without rebuilding, preserving the digest — so this is still build-once; you have just added a registry boundary that mirrors your environment boundary. Concretely the promotion step is a single command, no Docker build daemon involved:

```bash
# copy the EXACT bytes from the candidate path to the prod path, by digest.
# crane copy preserves the manifest and all layers, so the digest is identical
# in both registries — this is a move, not a rebuild.
SRC="ghcr.io/acme-candidate/app@sha256:9f2b7c4e3a1d8f6029b4c5e7d2a1f0b9c8e7d6a5b4c3d2e1f0a9b8c7d6e5f4a3"
DST="prodreg.example.com/acme/app:v2.4.1"
crane copy "$SRC" "$DST"

# verify the bytes survived the copy unchanged — the digest must match
crane digest "$DST"   # -> sha256:9f2b7c4e...  (identical to SRC)
```

The verification line is the point: after the copy, `crane digest` on the destination returns the *same* `sha256` as the source. If it ever differs, you did not copy — you rebuilt, and you have a bug. This is useful when prod's registry sits in a separate cloud account or network with stricter controls. It is optional; many teams keep one registry and promote purely by moving the digest forward in their deployment manifests. Either way, the cardinal rule holds: promotion *copies or references* the same bytes, it never rebuilds them.

## 7. The build-once pipeline, end to end

Time to make it concrete with a real pipeline. The shape is: one job builds and pushes exactly once on merge to the main branch, stamping the image with a semver-or-SHA tag *and* capturing its digest; then per-environment deploy jobs pull *that digest* and deploy. The digest is the contract passed from the build job to the deploy jobs. Here is a GitHub Actions workflow that does it, using OIDC to authenticate to the registry (no long-lived credentials) and `docker/build-push-action`, which conveniently outputs the pushed image's digest:

```yaml
name: build-and-promote
on:
  push:
    branches: [main]

permissions:
  contents: read
  packages: write
  id-token: write   # for OIDC keyless auth — no stored registry password

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.push.outputs.digest }}
    steps:
      - uses: actions/checkout@v4

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Buildx
        uses: docker/setup-buildx-action@v3

      # build ONCE, push ONCE; the digest is the artifact's true identity
      - name: Build and push
        id: push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          # human-facing tags so the image is findable; deploys pin the digest below
          tags: |
            ghcr.io/acme/app:git-${{ github.sha }}
          build-args: |
            VERSION=git-${{ github.sha }}
            COMMIT=${{ github.sha }}
            BUILD_TIME=${{ github.event.head_commit.timestamp }}
          provenance: true   # emit SLSA provenance attestation
          sbom: true         # emit an SBOM for the image

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      # deploy by DIGEST, not by tag — the exact bytes the build produced
      - run: |
          kubectl set image deployment/app \
            app=ghcr.io/acme/app@${{ needs.build.outputs.digest }} \
            -n staging
          kubectl rollout status deployment/app -n staging --timeout=120s

  deploy-prod:
    needs: [build, deploy-staging]
    runs-on: ubuntu-latest
    environment: production   # gated by a required reviewer in repo settings
    steps:
      - uses: actions/checkout@v4
      # SAME digest the build produced and staging tested — byte-for-byte
      - run: |
          kubectl set image deployment/app \
            app=ghcr.io/acme/app@${{ needs.build.outputs.digest }} \
            -n production
          kubectl rollout status deployment/app -n production --timeout=120s
```

Read what the `needs.build.outputs.digest` plumbing buys you. The build job runs once, pushes once, and *exports the digest as a job output.* Both deploy jobs consume that exact digest. There is no second `docker build` anywhere. Staging deploys `app@sha256:...`; prod deploys the same `app@sha256:...`. The `environment: production` line ties the prod job to a GitHub deployment environment, which you configure with a required reviewer so the promotion to prod is a human-approved gate — not a separate build, just an approval to move the *same artifact* forward. (`provenance: true` and `sbom: true` emit the supply-chain attestations that anchor the traceability chain; signing those with cosign and verifying at deploy is a supply-chain topic this series covers in its own track.)

The deploy here uses imperative `kubectl set image` for clarity, but the same digest flows just as well into a GitOps model: instead of `kubectl set image`, the prod deploy step would commit a one-line change to the prod overlay pinning `app@sha256:...` and let Argo CD or Flux reconcile. Either way the principle is preserved — the deploy moves a digest, it does not rebuild. And the Kubernetes manifest the deploy lands on should itself pin a digest rather than a tag, so that even a pod reschedule pulls the exact bytes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  namespace: production
spec:
  replicas: 6
  selector:
    matchLabels: { app: app }
  template:
    metadata:
      labels: { app: app }
    spec:
      containers:
        - name: app
          # pinned by DIGEST — a reschedule cannot pull a moved tag
          image: ghcr.io/acme/app@sha256:9f2b7c4e3a1d8f6029b4c5e7d2a1f0b9c8e7d6a5b4c3d2e1f0a9b8c7d6e5f4a3
          ports: [{ containerPort: 8080 }]
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 5
            periodSeconds: 10
          envFrom:
            - configMapRef: { name: app-config }   # env-specific config injected here
            - secretRef: { name: app-secrets }     # secrets injected here, not baked in
```

Notice the `envFrom` block: the *config* and *secrets* come from a `ConfigMap` and `Secret` in the production namespace, injected at deploy time. The image is environment-agnostic; production-ness is supplied around it. That is build-once and Twelve-Factor working together in nine lines.

#### Worked example: the "passed staging, broke prod" incident

Let me put numbers on the incident from the intro, because the before/after is the proof. The team ran a rebuild-per-environment pipeline. The relevant facts: the service's lockfile pinned direct dependencies but allowed `^` ranges on transitive ones; a transitive JSON library, call it `fastjson`, was constrained to `^1.4.0`. On Tuesday the staging build resolved `fastjson@1.4.2`. The integration suite — 1,400 tests — ran green against the Tuesday staging image. The change sat in the approval queue for two days. On Thursday, the production deploy job ran `docker build` afresh; `fastjson@1.4.3` had published on Wednesday, and the prod build resolved it. `1.4.3` had changed how it handled a trailing comma in a particular request payload, throwing where `1.4.2` had tolerated it. Roughly 5% of checkout requests carried that payload shape. So: zero failures in 1,400 staging tests against `1.4.2`, and ~5% 500-rate in prod against `1.4.3`, ninety seconds after promotion.

The arithmetic of the failure is brutal precisely because it is invisible to every gate. The staging tests *correctly* certified the staging artifact; they were green and they were right. They simply certified a *different artifact* than the one prod ran. The change-failure was a build-drift failure, not a code failure, and a code-focused investigation will never find it because the code never changed. The fix had nothing to do with the code: build once, capture the Tuesday digest, and promote *that digest* to prod on Thursday. With build-once, the prod artifact is `1.4.2` — the tested one — and the trailing-comma regression in `1.4.3` simply never enters production until a *new* build deliberately pulls it and the *full suite re-runs* against it.

Now the measured before/after, and the cost beyond the obvious. Under rebuild-per-env: 3 builds at ~8 minutes each (24 build-minutes per change, two-thirds of it wasted), 3 distinct artifacts, and a 1-in-20 prod failure rate at promotion. The incident itself cost an emergency rollback (call it 15 minutes of degraded checkout at a 5% error rate) plus an afternoon of four engineers' time chasing innocent code — roughly 16 person-hours of investigation that found nothing in the code because nothing was wrong in the code. Under build-once: 1 build (~8 minutes), 1 artifact, 3 deploys that are fast pulls (seconds each), and a 0% build-drift failure rate *by construction* — not by being careful, but because the artifact prod runs is the artifact staging tested, so a build-drift regression cannot reach prod undetected. The lead-time effect is real too: the redundant rebuilds had added ~16 minutes to every promotion path; removing them, plus removing the rebuild step from the prod-approval gate, cut the merge-to-prod wall-clock on a typical change by a meaningful chunk. Across the team's services over the following two quarters, the build-drift class of incidents — which had been roughly a third of their change-failures — went to zero, dropping their overall change-failure rate from about 28% to about 9%. To be honest about attribution: build-once did not fix the other two-thirds of their change-failures (those were genuine code and config bugs that need testing and progressive delivery to catch). It fixed *exactly* the build-drift third, completely, because that third was structural and build-once is a structural fix. That is a DORA metric moving — change-failure rate cut by roughly two-thirds of its build-drift component — because of one structural decision, not a heroic effort.

## 8. Traceability: from a running pod back to the change

The final payoff of build-once-plus-immutable-identity is **traceability** — the ability to start from "this thing is misbehaving in production" and walk a chain of unforgeable links back to the exact change that introduced it. This is what auditors mean by an "audit chain," what SLSA calls *provenance*, and what an incident responder lives or dies by at 3 a.m. The figure below draws the chain end to end.

![Graph showing the traceability chain from a running production pod reporting its version through the digest to the build run the git commit the pull request and finally the human change that introduced it](/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-7.png)

Walk it link by link. You start at the **running pod**. Because you embedded the version (section 5), `curl /version` returns the commit SHA and the digest the pod is running. That gives you the **digest** — `sha256:9f2b...` — which, being content-addressed, names exactly one artifact in the registry. From the registry you read the artifact's **provenance attestation** (the `provenance: true` from the build job) and its OCI labels, which point at the **build run** that produced it — GitHub Actions run `#4821`, with its full log. The build run records the **git commit** it built from — `abc123` — and the commit lives on a branch that was merged via a **pull request**, `#317`, which carries the review, the discussion, the tests that ran, and the human who approved it. That PR *is* the **change**: who made it, why, what they intended. So the chain is: pod → digest → build → commit → PR → change. Every link is immutable and verifiable; none of it is "trust me, it should be."

This chain only exists if the identity is immutable. If production ran `myapp:latest`, the chain breaks at the first link — `latest` does not tell you which artifact, the artifact does not tell you which build, and you are reduced to guessing from timestamps. The digest is what makes the chain real, because it is the one identifier that cannot be ambiguous or moved. This is also the foundation that supply-chain security (signing the digest with cosign, attaching SLSA provenance and an SBOM, verifying signatures at deploy with an admission policy) builds on. You cannot meaningfully sign or attest a mutable tag, because the thing the signature covers could change; you sign the digest, because the digest *is* the bytes. (The signing/verifying/SBOM machinery is a dedicated track in this series; the relevant point here is that build-once-by-digest is the bedrock the whole supply-chain story stands on.)

#### Worked example: the mutable-tag burn

Here is the second incident, the `:latest` race, with the timeline that makes it click. A team deployed by tag: their prod manifest said `image: myapp:latest`, and their CI tagged every main-branch build `latest`. The sequence: at 14:00, build A finished and tagged `latest`; QA tested `latest` (which was A) in staging and signed off at 14:30; at 14:45, an *unrelated* merge triggered build B, which also tagged `latest`, moving the pointer from A to B; at 15:00, the prod deploy ran — and because a couple of prod pods happened to restart (an autoscaler event), they pulled `latest`, which now resolved to *B*, an image nobody had tested. Some prod pods ran A, some ran B, and B had a half-finished feature behind a flag that defaulted on. The result was a confusing partial outage where the same request succeeded or failed depending on which pod served it — the hardest kind of incident to diagnose, because it is non-deterministic from the client's view.

The root cause was not the feature; it was the *mutable tag.* `latest` answered "what is newest right now," and "right now" changed between the test and the deploy, and changed again per-pod as pods restarted. This is worth dwelling on because it is a *race condition* in the deploy, and races are the hardest bugs to reason about: the same manifest, applied at two slightly different instants, produces two different running images, because the manifest does not name an artifact — it names a query (`latest`) whose answer moves. The non-determinism is not in your code; it is in the *identity scheme*. And the per-pod split made it worse than a clean wrong-deploy: because only the pods that happened to restart re-resolved `latest`, the fleet ran a *mixture* of A and B, so a given user's request succeeded or failed depending on which pod load-balancing happened to route it to. From the client's view the service was randomly broken, which is the most expensive kind of incident to triage — there is no clean repro, the error rate is a fraction, and half your dashboards look fine because half the fleet is fine.

The fix is one line: deploy by digest. QA tested `myapp@sha256:AAAA` (image A); the deploy pinned `myapp@sha256:AAAA`; build B's `latest` move was irrelevant because nothing pointed at `latest` anymore. Every prod pod — including any that restart later — pulls `sha256:AAAA`, the exact bytes QA approved. The figure below contrasts the two paths: on the left the mutable tag moves between test and deploy so prod gets an untested image; on the right the frozen digest cannot move so prod gets exactly what was tested.

![Before and after comparison showing a deploy by the mutable latest tag where the tag moves to an untested image next to a deploy by a frozen sha256 digest where production runs the exact tested bytes](/imgs/blogs/build-once-promote-everywhere-artifacts-and-versioning-4.png)

The measured difference: before, a tag-deploy pipeline with a recurring class of "wrong image in prod" near-misses (this team caught three in a quarter, one of which became the partial outage above); after pinning digests, that class went to zero, because there is no pointer left to move. Quantify the win honestly: the near-misses had averaged maybe 20 minutes each of "wait, which image is this?" investigation, the one real outage cost roughly an hour of partial checkout degradation and a postmortem, and the fix was a one-character-class change to the manifest — `:latest` became `@sha256:...`. There is no other reliability investment in the entire delivery system with that return: near-zero cost, an entire failure mode eliminated, permanently, because the property you bought is *mathematical* (a digest cannot move) rather than *procedural* (please remember not to re-push the tag). It is the cheapest reliability win in delivery, full stop.

## 9. Stress-testing the principle: the hard cases

A principle earns its keep by surviving the awkward cases, so let me pose the ones that come up and reason through each, because "build once" is not a slogan that answers itself.

**"What if I genuinely need an environment-specific build — a different feature compiled in for a region?"** Then you do not have one artifact; you have two products, and you should treat them as two artifacts each built once and promoted independently, not as "one codebase rebuilt per environment." The test is: are the differences *configuration* (values that vary) or *code* (different behavior compiled in)? Configuration belongs in the environment; if it is truly different code, build each variant once and promote each. The anti-pattern is rebuilding the *same* intended artifact per environment; building genuinely different artifacts deliberately is fine, as long as each is itself built once and the environment-specific build is itself reproducible and tested.

**"What if the registry is down mid-promotion?"** Promotion by digest is a pull, and if the registry is unreachable the deploy fails *closed* — the new pods cannot pull, so they do not start, and the readiness probe keeps old pods serving. That is the correct failure mode: no registry, no new artifact, old artifact keeps running. Contrast a rebuild-per-env pipeline where a registry blip during the prod build can produce a *partial* image or pull a stale cached base — failing open in a subtle way. Build-once degrades safely. The mitigation for the registry being a single point of failure is the usual one (replicate the registry, or pull-through-cache it near the cluster), and crucially your *rollback* digest should be cached locally so a registry outage cannot block a rollback.

**"What if two PRs merge at once?"** Each merge to main produces its own build with its own commit SHA and its own digest. There is no shared mutable state to race on, *because you are not using mutable tags.* If both pipelines try to deploy, the normal deploy-ordering / concurrency controls apply (GitHub Actions `concurrency:` groups, or GitOps serializing on the git commit order), but the artifacts themselves never collide — two digests, two distinct immutable things. This is one of the quiet benefits of digest-based identity: concurrency problems that plague mutable-tag pipelines (the `latest`-race above is exactly a concurrency bug) simply do not arise.

**"What if I need to patch a security vuln in a running release without a code change — just rebuild on a patched base image?"** This is the legitimate case where you *want* a rebuild, and the answer is: that rebuild produces a *new artifact* with a *new digest*, and it goes through the *same promotion chain* (dev → staging → prod) as any other change. You are not rebuilding the *same* artifact per environment; you are deliberately producing a new one (same source, patched base) and promoting *it* once. The discipline holds: every artifact that reaches prod was built once and promoted, even the security rebuilds. What you must not do is rebuild the patched image *separately for each environment* — build it once, test the patched bytes, promote those bytes.

**"What if rollback's digest got garbage-collected?"** Then your retention policy is broken, and this is the failure I flagged in section 6. The fix is policy, not heroics: retention must protect (a) every currently-deployed digest across all environments, and (b) a rollback window of recent digests. Wire your GC to read the live deployments and never expire a referenced digest. If you have already lost the digest, you are forced to rebuild the old commit — and now you are praying that rebuild is reproducible, which is exactly the uncertainty build-once was supposed to eliminate. The lesson is that build-once's rollback guarantee is only as good as your retention policy; the two are a single system.

## War story: Knight Capital and the cost of "what is actually running?"

The most expensive deployment failure in this genre is Knight Capital's, August 1, 2012. Knight was deploying new trading software to its servers. The deploy was manual and per-server, and a technician copied the new code to seven of the eight production servers but missed the eighth. Worse, the new code *reused a feature flag* that, in the old code still running on the eighth server, activated long-dead "power peg" order-routing logic. So the fleet was running two different artifacts: seven servers on the new code, one on the old code with a repurposed flag. When trading opened, the eighth server began sending a torrent of erroneous orders. In about 45 minutes, Knight accumulated a roughly \$460 million loss, and the firm — one of the largest market makers in US equities — was effectively destroyed, acquired within days.

Read that through this post's lens. The proximate cause was a deploy that left the fleet in a mixed state — *different artifacts running in the same environment* — and an inability to quickly answer "what is actually running on each server?" If Knight had deployed an immutable, content-addressed artifact through an automated promotion that verified every server converged on the same digest (and refused to proceed on a partial rollout), the mixed-state condition could not have persisted silently. The feature-flag reuse was its own sin, but the *delivery* sin — the one this post is about — was that the system could run a heterogeneous mix of artifacts and nobody could see it. Build-once-promote-everywhere, plus deploy-by-digest, plus a `/version` endpoint, plus a rollout that gates on convergence, is the modern answer to exactly this failure: there is one artifact, it has one verifiable identity, and you can ask every instance what it is running and get a single answer. The broader supply-chain incidents — SolarWinds (a tampered build inserted malware into a signed release) and the Codecov bash-uploader compromise — make the same point from the security side: if you cannot prove that the artifact in production is the artifact you built and tested, from a build you trust, you do not actually know what is running, and "you do not know what is running" is the root of both the reliability and the security catastrophe.

## How to reach for this (and when the nuance matters)

Build-once-promote-everywhere is one of the few practices I would call close to universal — it costs almost nothing and removes an entire class of incidents — but the *machinery* around it scales with your needs, and pretending otherwise leads to over-engineering. Here is the honest guidance.

**Always do the core thing**, even at the smallest scale: build the artifact once, deploy it by digest, and never deploy `:latest` to anything you care about. This is not a "platform team" practice; a solo developer deploying a single container should pin the digest. It is one line of manifest and it makes "what is running" answerable. There is no scale at which "I tested a different binary than I shipped" is acceptable.

**Add immutable tags and a real retention policy** once you have more than a handful of releases or more than one person pushing — basically as soon as a re-pushed tag could surprise someone. It is a registry setting and a lifecycle policy; cheap.

**Add the multi-registry-path promotion (dev path → prod path with a copy-by-digest)** only when you have a real reason — a separate prod cloud account, a compliance boundary, a need to restrict who can write to the prod registry. For most teams a single registry with digest-based deployment is plenty, and the extra path is operational overhead you do not need. Do not build a two-registry promotion flow because a blog post mentioned it; build it when an auditor or a network boundary demands it.

**Add semver and tag-driven releases** when something *consumes* your artifact and needs a compatibility promise — a published library, a shared platform component, a public API. For an internal leaf service that only your pipeline deploys, the commit SHA plus the digest is enough; semver is ceremony you do not need to maintain, and a stale, lying semver is worse than no semver. Match the scheme to whether anyone is actually reading it.

**Where build-once is not sufficient**: it does not, by itself, give you environment parity (section 4) or progressive-delivery safety. It removes build-drift surprises, not environment-drift surprises and not bad-code surprises. For the safety of the deploy *itself* — canaries, SLO gates, automated rollback — that is the reliability layer, and the SRE post on [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) is where to go; this post gets you a trustworthy *artifact* to promote, and that post gets you a safe *way* to promote it. The two compose: progressive delivery is far more meaningful when the canary and the full rollout are provably the same bytes.

## Key takeaways

- **Build the artifact once.** Rebuilding per environment means the thing you ship to prod is not the thing you tested in staging — you inject untested build-drift at the riskiest moment. Build once, and what you tested is byte-for-byte what you ship.
- **The digest is the true identity.** A content-addressed `sha256` digest is derived from the bytes and cannot lie or move. A tag like `v1.2.3` or `latest` is a mutable pointer. Deploy by digest, not by tag.
- **`latest` in production is a footgun.** The tag can move between the moment you test and the moment a pod pulls, so prod can silently run an untested image. Pin `@sha256:...` and the failure mode disappears.
- **Promotion is movement, not manufacture.** A deploy is "run artifact X here with this environment's config," not "rebuild for here." Each environment adds confidence by testing the *same* artifact under more production-like conditions.
- **Config and secrets are injected per environment.** The artifact must be environment-agnostic (Twelve-Factor: config in the environment), or you cannot promote it. Code in the artifact; environment-ness around it.
- **Version for two audiences.** Semver communicates compatibility to humans; the commit SHA and digest are the precise machine identity. Embed both in the artifact and expose a `/version` endpoint so "what is running?" has a one-curl answer.
- **Registries enforce immutability and manage cost.** Turn on immutable tags so a version cannot be overwritten; set retention/GC that protects every deployed digest and a rollback window — never garbage-collect the digest you might need to roll back to.
- **Immutability makes traceability real.** From a running pod you can walk digest → build → commit → PR → change only because the digest is unforgeable. This is the foundation supply-chain signing and SLSA provenance build on.

## Further reading

- [The CI/CD pipeline map: from commit to production](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series intro and the commit → build → test → package → deploy → operate spine this post lives inside.
- [The build stage: reproducible, fast, and cacheable](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) — the companion principle: how to produce a trustworthy artifact in the first place, which is the precondition for promoting one.
- A planned sibling on **image registries, tagging, and promotion** (`image-registries-tagging-and-promotion`) goes deeper on registry mechanics; one on **multi-environment promotion across dev, staging, and prod** (`multi-environment-promotion-dev-staging-prod`) covers the GitOps reconcile and gating; and one on **configuration and secrets in Kubernetes** (`configuration-and-secrets-in-kubernetes`) covers the per-environment injection this post depends on.
- [Deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) — the SRE reliability layer: canaries, SLO gates, and automated rollback, which compose with build-once to make the *act* of promoting safe.
- **The Twelve-Factor App** (12factor.net), especially Factor III ("Config") and Factor V ("Build, release, run") — the canonical statement of separating build, release, and config that underpins build-once-promote-everywhere.
- **The SLSA framework** (slsa.dev) and **Sigstore/cosign** (sigstore.dev) — supply-chain provenance and keyless signing of artifacts by digest, the security layer that the immutable-digest identity makes possible.
- **Accelerate** (Forsgren, Humble, Kim) and the annual **State of DevOps / DORA reports** — the empirical case that disciplined delivery practices move deploy frequency, lead time, change-failure rate, and time-to-restore.

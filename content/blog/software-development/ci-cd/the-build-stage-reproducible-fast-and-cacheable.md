---
title: "The Build Stage: Reproducible, Fast, and Cacheable"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Make your build a pure function of its inputs, cache it on three levels, and cut a 38-minute pipeline to 6 — without ever shipping a non-deterministic artifact again."
tags:
  [
    "ci-cd",
    "devops",
    "build",
    "reproducible-builds",
    "caching",
    "hermetic-builds",
    "bazel",
    "docker",
    "github-actions",
    "supply-chain",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-1.png"
---

A few years ago I watched a payments team chase a bug that did not exist. A customer reported a checkout failure that their on-call engineer could not reproduce. The engineer ran the service locally: it worked. They ran it in the staging cluster: it worked. They pulled the exact image tag that was live in production, ran *that* locally, and — it worked. Four hours and two engineers later, someone finally diffed the bytes of the production image against an image built from the same git commit on a fresh machine. The two were not the same. Same commit, same `Dockerfile`, two different artifacts. The production image had been built three weeks earlier on a CI runner that happened to have an older base image cached, plus a transitive dependency that had silently published a new patch version in between. The "bug" was that nobody actually knew what was running in production, because the build that produced it was not reproducible. The git commit was a label on a box whose contents could not be recovered.

That story is the whole reason this post exists. The build is the stage of the pipeline where most of the time goes and where most of the lies hide. If you measure a typical pipeline — and you should, with real timing data — the build usually dwarfs everything downstream. Tests run in minutes; the build can run in tens of minutes. And unlike a test, a build does not just take time: it *fabricates the thing you ship*. If the fabrication is not deterministic, every promise the rest of your pipeline makes is built on sand. You can test an artifact, scan it, sign it, and promote it through five environments, but if the artifact you tested is not byte-for-byte the artifact you deploy, you tested a different program. The figure below shows where the wall-clock time actually goes in a representative pipeline run and why the build is the stage worth obsessing over.

![Stacked diagram of the commit build test package deploy operate pipeline showing the build stage consuming roughly thirty minutes and flagged as slow and non-deterministic while later stages are fast](/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-1.png)

This post is part of the series **"CI/CD & Cloud-Native Delivery, From Commit to Production."** If you have not read [the CI/CD pipeline map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) yet, start there — it lays out the spine we keep returning to: **commit → build → test → package → deploy → operate**, governed by two principles, **"build once, promote everywhere"** and **"everything as code,"** and measured by the four DORA metrics. The build stage is where "build once, promote everywhere" is *earned*. You cannot promote one artifact through your environments if you cannot produce one artifact you trust. So my claim, and the spine of everything below, is this: **a great build is three things at once — reproducible, fast, and cacheable — and the three are not separate goals but the same goal seen from three angles.** A reproducible build is one whose output depends only on its declared inputs; a build like that is *cacheable*, because you can safely reuse the output whenever the inputs are unchanged; and a build that caches well is *fast*, because it stops redoing work it already did. By the end you will be able to diagnose why your build is non-deterministic, make it hermetic, set up dependency caching, layer caching, and build-graph caching with real config, reason about cache-hit-rate as a metric, and cut a slow pipeline by 5× or more — and you will understand the supply-chain reason this matters far beyond speed.

## 1. Why "works on my machine but not in CI" happens

Let me define the terms first, because the accessibility of the rest depends on it. A **build** is the process that turns your source code plus its dependencies into a runnable **artifact** — a compiled binary, a JAR, a Python wheel, a container image. An **artifact** is the immutable output you actually ship: not the source, but the thing produced from it. A **registry** is the store you push artifacts to (a container registry like GHCR or ECR, or a package registry like a Maven repo or npm). When we say a build should be *reproducible*, we mean: run it twice, on two machines, from the same source, and you get the same artifact. That sounds obvious. It is shockingly rare.

The reason it is rare is that almost every build reads inputs it never declared. Your source code is a declared input — it is in git, pinned to a commit. But your build also reads, often invisibly, the version of the compiler installed on the machine, the system libraries it links against, the contents of the network at the moment dependencies are fetched, environment variables, the locale and timezone, and — astonishingly often — the actual wall-clock time, baked into the artifact as a build timestamp. None of those are in git. None of them are pinned. The build is a function, but it is a function of far more than its declared arguments, and the undeclared arguments differ between your laptop and the CI runner. That is the entire mechanism behind "works on my machine."

Put plainly: a build that reads undeclared inputs is not a function of its source; it is a function of its source *and the accident of where it ran*. The tree below classifies the usual suspects.

![Tree diagram classifying undeclared build inputs into toolchain environment network dependencies and the wall clock with leaves such as unpinned compiler version host system libraries locale and timezone floating latest tags and baked in build dates](/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-2.png)

Walk the branches, because each is a real outage waiting to happen:

- **Toolchain version.** Your laptop has Node 20.11; the runner has Node 20.9; a CI cache from last month has Node 18. A minor compiler or interpreter difference changes optimization, changes minified output, occasionally changes behavior. The artifact differs even though the source did not.
- **System libraries.** A Go or Rust binary that dynamically links `glibc` will link against whatever `glibc` is on the build host. Build on Ubuntu 22.04, run on Alpine, and you get a `not found` at startup that has nothing to do with your code. C extensions in Python wheels are the classic case.
- **Network-fetched dependencies.** If your build runs `npm install` without a lockfile, or `pip install requests` without a pinned version, or `FROM node:latest` in a `Dockerfile`, then the *internet at build time* is an input. A dependency maintainer publishes a patch, and your "unchanged" build produces a changed artifact. This is the most common source of build drift in the wild.
- **Environment, locale, timezone.** A build that sorts file names will sort them differently under `LANG=C` versus `LANG=en_US.UTF-8`. A test that formats a date will produce different output in `UTC` versus `America/New_York`. These leak into generated code and into the artifact.
- **The wall clock.** This is the one that bites hardest and surprises people most. If your build embeds the current time — a `BUILD_DATE` label, a `new Date()` baked into a generated config, a `__TIMESTAMP__` in a compiled file, a gzip stream that records modification times — then *every single build produces a different artifact by construction*, even with everything else pinned. Two CI runs of the identical commit ten seconds apart produce two different images. We will fix exactly this in Section 8.

Here is why you should care beyond intellectual tidiness. The principle "build once, promote everywhere" says you build a single artifact, test it, and then promote that *same* artifact through staging and production rather than rebuilding per environment. If your build is non-deterministic, you literally cannot honor that principle, because every environment that triggers a rebuild gets a different program. The non-reproducible build I opened with had quietly turned "promote one artifact" into "rebuild three times and hope," and the production image was a stranger nobody could recreate. Reproducibility is not a nice-to-have for purists. It is the precondition for trusting your own pipeline.

## 2. Hermetic builds: declare and pin every input

The fix has a name: a **hermetic build**. A hermetic build is one that is sealed off from undeclared inputs — it can only see what you explicitly give it. The mantra is *declare and pin every input* so that the build becomes a pure function: same declared inputs in, same artifact out, anywhere. The before-and-after below contrasts a leaky build with a hermetic one.

![Before and after diagram comparing a leaky build that uses the host toolchain and fetches latest dependencies at build time and produces varying output against a hermetic build that pins the toolchain by digest vendors dependencies with a lockfile and produces the same output every time](/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-3.png)

Hermeticity is a discipline applied to every undeclared input from Section 1. Concretely:

**Pin the toolchain.** Do not say "Node 20." Say the exact version, and ideally pin the container that provides it by digest, not by tag. A tag like `node:20` is a moving pointer; a digest like `node:20@sha256:abc123…` is immutable. The same goes for your language toolchain, your build tool, and any CLI the build invokes.

```dockerfile
# Pin the base image by DIGEST, not a floating tag.
# `node:20` moves; this exact digest never does.
FROM node:20.11.1-bookworm-slim@sha256:1d3f6f0e... AS build

# Pin the package manager too — corepack lets us nail the exact pnpm version.
RUN corepack enable && corepack prepare pnpm@9.1.0 --activate

WORKDIR /app
```

**Use lockfiles, and install from them strictly.** A lockfile (`package-lock.json`, `pnpm-lock.yaml`, `poetry.lock`, `Cargo.lock`, `go.sum`) records the exact resolved version *and content hash* of every transitive dependency. The crucial detail most teams miss is that `npm install` may *update* the lockfile; you want the command that *refuses* to deviate from it.

```bash
# WRONG in CI — may resolve new versions, may mutate the lockfile:
npm install

# RIGHT in CI — installs exactly what the lockfile pins, fails if it can't:
npm ci

# pnpm equivalent:
pnpm install --frozen-lockfile

# Python (uv) equivalent — install exactly the locked set:
uv sync --frozen
```

**Cut off the network during the build proper.** This is the part that feels extreme until you have lived through a dependency-confusion incident. The strongest hermetic builds *fetch all dependencies into a content-addressed cache first*, then run the actual compilation with no network access at all. Bazel does this by design; you can approximate it with Docker by vendoring dependencies or pre-populating a layer. If the build cannot reach the network, the network cannot be an input.

**Vendor or content-address third-party code.** Vendoring means committing your dependencies' source into your repo (Go's `vendor/`, for instance). Content-addressing means referring to each dependency by the hash of its contents, so if the bytes change, the reference changes — you can never silently get different bytes under the same name. Lockfiles with integrity hashes give you most of this; vendoring gives you all of it at the cost of repo size.

**Neutralize the environment.** Set `LANG`, `LC_ALL`, `TZ`, and `SOURCE_DATE_EPOCH` explicitly in the build environment so locale, timezone, and the clock are declared inputs rather than ambient ones.

The payoff of hermeticity is not only reproducibility. A hermetic build is **debuggable** (you can recreate exactly what shipped, three weeks later, from the commit alone), **trustworthy** (the artifact provably corresponds to the source — Section 8), and crucially **cacheable** (because if the output depends only on declared inputs, you can hash those inputs and safely reuse a prior output whenever the hash matches). That last property is the bridge to the entire rest of this post. *Reproducibility is what makes caching correct.* You can only skip rebuilding a target if you are certain its inputs have not changed, and you can only be certain of that if the build is a function of declared inputs. A non-hermetic build cannot be safely cached, because you can never be sure the cached output reflects the current undeclared inputs. This is why teams that try to bolt aggressive caching onto a leaky build get the worst of both worlds: stale, wrong artifacts served from cache. Fix hermeticity first; then caching is not a gamble.

#### Worked example: the lockfile that wasn't enforced

A team had a `package-lock.json` committed and assumed they were safe. Their CI step ran `npm install`. One morning, two PRs that touched nothing in the dependency tree produced subtly different bundles, and a flaky visual-regression test started failing intermittently. The cause: a transitive dependency had published a new patch within the SemVer range, and `npm install` happily resolved it — and even rewrote the lockfile in CI, which nobody noticed because CI does not commit. The artifact was a function of *when the build ran*, not of the source. The fix was one word: change `npm install` to `npm ci`. After that, the lockfile became authoritative; CI failed loudly if the lockfile and `package.json` disagreed, instead of silently drifting. The flaky test stopped flaking, because the input that was actually changing — the dependency set — was now pinned. The lesson generalizes: having a lockfile is necessary but not sufficient; you must use the *frozen* install command, or the lock is decorative.

## 3. Reproducible builds: bit-identical is the gold standard

Hermeticity gets you "same inputs, same output." The strongest form of that is a **reproducible build** in the strict sense: the artifact is **bit-identical** across independent rebuilds. Same commit, built on your machine and on mine and on a CI runner in another datacenter, yields a file with the same SHA-256 digest, byte for byte. This is the gold standard, and it is harder than hermeticity because it requires eliminating *every* source of nondeterminism in the output, including subtle ones: file ordering in archives, embedded timestamps, absolute paths baked into debug info, non-deterministic map iteration order in code generators, random temp-directory names that leak into output.

Why chase bit-identical when "same behavior" is usually enough? Two reasons, one practical and one about trust.

The practical reason is caching, again. A content-addressed cache keys on the hash of inputs and stores the output. If two builds with the same inputs can produce different *outputs*, then downstream consumers of the artifact (the test stage, the deploy stage) see a changed artifact and redo their work even though nothing meaningful changed. Bit-identical output makes the *artifact itself* content-addressable, so caches further down also hit. Determinism compounds.

The trust reason is the supply chain, and it is the deeper one. If your build is bit-reproducible, then *anyone* can take your published source at a commit, rebuild it independently, and check that the resulting digest matches the artifact you shipped. That is **provenance verification**: proof that the binary running in production was built from the source you claim, not from source someone tampered with. This is the foundation of frameworks like **SLSA** (Supply-chain Levels for Software Artifacts) and the reason the reproducible-builds movement exists in projects like Debian, Tor, and Bitcoin. We will return to it in Section 8 with the verification flow. For now, hold the idea: a reproducible build is not just convenient, it is *checkable* by a third party, and checkability is what lets you trust software you did not personally compile.

The practical work of getting to bit-identical is a checklist of specific, well-understood sources of nondeterminism, and it helps to see them concretely so the goal stops feeling magical:

- **Embedded timestamps.** The big one, addressed at length in Section 8: any `BUILD_DATE`, gzip/tar modification time, or `__DATE__`/`__TIME__` macro. The fix is `SOURCE_DATE_EPOCH`, an environment variable that build tools honor in place of the wall clock, set to the commit time.
- **File ordering in archives.** A `tar`, `zip`, JAR, or wheel built by globbing a directory picks up files in filesystem order, which varies by machine. The fix is to sort entries explicitly and to use deterministic archivers (for example, `tar --sort=name --mtime=@$SOURCE_DATE_EPOCH`).
- **Absolute paths in debug info and metadata.** A binary that bakes in `/home/alice/project/...` versus `/home/ci-runner/...` differs byte-for-byte. The fix is path remapping (`-ffile-prefix-map` for GCC/Clang, `-trimpath` for Go) so the recorded path is relative or canonical.
- **Non-deterministic ordering in code generators.** A generator that iterates a hash map emits declarations in a random order. The fix is to sort outputs deterministically inside the generator, or to pin the map's seed.
- **Build-host fingerprints.** Usernames, hostnames, build numbers, or a random nonce leaked into the artifact. The fix is to refuse to embed anything host-specific; if you need a build identifier, derive it from the commit, never from the machine.

You do not have to fix all of these at once. The honest path is to add a CI step that builds the artifact *twice* on different runners and diffs the two digests; when they differ, the diff tool (`diffoscope` is the canonical one) tells you exactly which bytes disagree, and you chase down one source at a time. Each fix permanently removes a class of nondeterminism and nudges the hit-rate up.

Most teams do not need strict bit-for-bit reproducibility on day one. What every team needs is hermeticity (Section 2), and what every team eventually wants — once they are scanning and signing artifacts — is reproducibility, because it closes the loop between source and binary. Treat hermetic as the floor and bit-identical as the ceiling you climb toward, removing one source of nondeterminism at a time.

## 4. Caching: the single biggest CI speedup lever

Now to speed, and the single biggest lever you have: **caching**. The premise is simple and the discipline is everything. *Caching means not redoing work whose inputs did not change.* If you can identify the inputs of a unit of work precisely, hash them, and store the output keyed on that hash, then the next time you encounter the same inputs you fetch the result instead of recomputing it. The reason caching is the biggest lever is arithmetic: most of what a build does on any given commit is *re-doing work that did not change*. You touched 3 files; the build recompiled 2,000. Caching is how you stop paying for the 1,997 you did not touch.

There are three distinct layers of caching, and they are frequently confused, which leads to teams "having caching" while their builds stay slow. They attack different redundant work and key on different inputs. The matrix lays them out.

![Matrix comparing dependency cache Docker layer cache and build graph cache across what each skips what it is keyed on and its shared scope showing the build graph cache keyed on content hash and shared remotely across all developers](/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-4.png)

**Layer 1 — Dependency caching.** This caches the *downloaded dependencies*: the contents of `~/.m2` (Maven), `node_modules` or the pnpm store, `~/.cargo` (Rust), the pip wheel cache, the Go module cache. These are expensive to fetch over the network and almost never change between runs, because they are pinned by your lockfile. The cache key is therefore the **hash of the lockfile**: if the lockfile is unchanged, the dependency set is unchanged, so restore it; if the lockfile changed, recompute. This is the easiest win and the one to do first. On GitHub Actions it is one block.

```yaml
# .github/workflows/ci.yml — dependency cache keyed on the lockfile hash
name: ci
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: "20.11.1"

      # Cache the pnpm store, keyed on the EXACT lockfile.
      # A different lockfile -> different key -> cold restore -> recompute.
      - name: Cache pnpm store
        uses: actions/cache@v4
        with:
          path: ~/.local/share/pnpm/store
          key: pnpm-store-${{ runner.os }}-${{ hashFiles('pnpm-lock.yaml') }}
          # restore-keys is a FALLBACK prefix: if the exact key misses,
          # restore the newest cache that shares this prefix, then top it up.
          restore-keys: |
            pnpm-store-${{ runner.os }}-

      - run: corepack enable && corepack prepare pnpm@9.1.0 --activate
      - run: pnpm install --frozen-lockfile
      - run: pnpm build
```

**Layer 2 — Build-layer caching.** This is the Docker layer cache: each instruction in a `Dockerfile` produces a layer, and if the inputs to a layer are unchanged, the layer is reused. We treat the container-image build in depth in the planned sibling post *Building images fast and securely in CI*; here the key idea is ordering. Put the steps whose inputs change *least often* (installing dependencies) *before* the steps that change *most often* (copying your source), so that a code change does not bust the dependency-install layer. We will show the ordering trick in Section 5.

**Layer 3 — Build-graph caching.** This is the most powerful and least understood. Tools like **Bazel**, **Gradle**, **Nx**, and **Turborepo** model your build as a graph of *targets* — a library, a binary, a test, a bundle — where each target declares its inputs (its own sources plus the targets it depends on). The tool hashes each target's inputs *transitively* and stores the output **keyed on that content hash** in a cache. When you build, it walks the graph, computes each target's hash, and for any target whose hash matches a cached entry, it *fetches the output instead of rebuilding the target*. This is **incremental build**: you do not rebuild the world, you rebuild only the targets whose inputs actually changed, plus everything downstream of them.

The game-changer is the **remote cache**: the content-addressed store is shared across CI runners *and developer laptops*. The first person (or CI job) to build a target at a given hash pays the cost and uploads the result; everyone else who needs that exact target — same inputs, same hash — downloads it in milliseconds. The flow looks like this.

![Graph diagram of a remote build graph cache where changed source representing twelve of two hundred targets is hashed into a content address looked up in a shared remote cache producing one hundred eighty eight cache hits and twelve cache misses that are built and merged into a three minute artifact down from eighteen](/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-5.png)

Here is a Turborepo configuration that turns on remote caching for a JavaScript/TypeScript monorepo. The `inputs` and `outputs` declarations are what make caching *correct*: they tell the tool exactly which files feed each task and which files the task produces.

```json
{
  "$schema": "https://turbo.build/schema.json",
  "remoteCache": { "enabled": true },
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tsconfig.json", "package.json"],
      "outputs": ["dist/**"]
    },
    "test": {
      "dependsOn": ["build"],
      "inputs": ["src/**", "test/**", "vitest.config.ts"],
      "outputs": ["coverage/**"]
    },
    "lint": {
      "inputs": ["src/**", ".eslintrc.cjs"]
    }
  }
}
```

And the equivalent idea in Bazel, whose remote cache is configured in `.bazelrc`. Bazel is hermetic by construction, which is precisely why its caching is safe to share across machines.

```bash
# .bazelrc — point the build at a shared remote cache.
# Because Bazel builds are hermetic, a cache key computed on one
# machine is valid on every other machine, so CI and laptops share hits.
build --remote_cache=grpcs://cache.internal.example.com
build --remote_upload_local_results=true
# Don't let a single missing-network event fail the whole build:
build --remote_local_fallback=true
# Hash file CONTENTS, not mtimes, so timestamps never bust the cache:
startup --host_jvm_args=-Dbazel.DigestFunction=SHA256
```

The thing to internalize: dependency caching saves you the *download*; build-graph caching saves you the *compile*. They are not substitutes. A team that caches `node_modules` but rebuilds every package on every commit has fixed the small problem and left the big one. The compile is usually where the minutes are.

## 5. Cache-key discipline: the part everyone gets wrong

A cache is only as good as its key, and the key is where almost every caching bug lives. The rule is exact and unforgiving: **the cache key must capture exactly the inputs the work depends on — no more, no less.** Two failure modes sit on either side of that rule, and they are mirror images.

A **too-broad key** — one that does not include some real input — serves *stale* results. The classic example: keying a dependency cache on the branch name instead of the lockfile hash. You change a dependency, the lockfile changes, but the branch name is the same, so CI restores the old `node_modules` and your build runs against the wrong dependencies. The build is fast and *wrong*, which is worse than slow, because you trust the green check. Too-broad keys are dangerous precisely because they fail silently.

A **too-narrow key** — one that includes inputs the work does not actually depend on — *never hits*. The classic example: including the git commit SHA in a dependency-cache key. Every commit has a unique SHA, so every commit gets a unique key, so every run is a cache miss, so you pay full price every time while believing you have caching. Too-narrow keys are merely wasteful, not dangerous, which is why they survive in codebases for years — the build is slow but correct, and slow-but-correct does not page anyone.

The discipline, then, is to make the key the hash of the *true* inputs:

- For a **dependency cache**, key on the hash of the lockfile(s): `hashFiles('pnpm-lock.yaml')`. The lockfile *is* the complete declaration of the dependency set, so its hash is exactly the right key. If you have multiple lockfiles, hash all of them: `hashFiles('**/package-lock.json')`.
- For a **layer cache**, the key is implicit in the layer's inputs — which is why instruction ordering in the `Dockerfile` matters so much.
- For a **build-graph cache**, the tool computes the key for you by transitively hashing each target's declared inputs. Your job is to *declare the inputs correctly* (the `inputs` arrays in the Turborepo config above). Forget to declare an input and you get a too-broad key (stale outputs); declare a spurious input — like a `.log` file your tool regenerates — and you get a too-narrow key (perpetual misses).

The Docker ordering trick deserves its own example, because it is the most common place a layer cache is accidentally defeated.

```dockerfile
# BAD ORDERING — copies all source before installing deps.
# Any code change invalidates the COPY layer, which busts every
# layer after it, so `npm ci` re-runs on every single commit.
FROM node:20.11.1-bookworm-slim AS bad
WORKDIR /app
COPY . .
RUN npm ci
RUN npm run build
```

```dockerfile
# GOOD ORDERING — copy ONLY the lockfile + manifest first, install,
# THEN copy source. A code change busts only the source COPY layer;
# the expensive `npm ci` layer is reused because its inputs (the
# lockfile and manifest) did not change.
FROM node:20.11.1-bookworm-slim@sha256:1d3f6f0e... AS build
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY . .
RUN pnpm build
```

The difference between those two `Dockerfile`s is the difference between a 30-second incremental build and a 6-minute one, on the same hardware, for the same one-line code change. Nothing about the source differs — only the *order* in which inputs are declared to the cache. Cache-key discipline is mostly the discipline of telling the cache the truth about what each step depends on.

## 6. Cache-hit-rate is the metric

If you want to manage caching, measure it, and the metric is **cache-hit-rate**: of all the cacheable units of work in a build, what fraction were served from cache rather than recomputed? Most build-graph tools report this directly. Bazel will tell you `X processes, Y remote cache hit`; Turborepo prints `>>> FULL TURBO` when every task hit; Gradle's build scans show cache hit/miss per task. Watch this number the way you watch a test pass rate.

The reason hit-rate is *the* metric is that build time is roughly linear in the work you actually do, and hit-rate determines how much work you actually do. Let me make the arithmetic explicit, because the payoff is more dramatic than intuition suggests.

Suppose a cold build (zero cache hits) takes $T_{\text{cold}}$ and the work splits into a fixed overhead $f$ (checkout, cache restore, final packaging — work you always do) and the cacheable build work, which is the rest. If the cache-hit-rate is $h$ — the fraction of cacheable work served from cache — then the time you spend is approximately:

$$T(h) = f + (1 - h)\,(T_{\text{cold}} - f)$$

Set some numbers. Say a cold build is $T_{\text{cold}} = 38$ minutes with fixed overhead $f = 3$ minutes, so the cacheable work is 35 minutes. At a hit-rate of $h = 0.8$ (80% of targets cached, which is *normal* for a small change in a large monorepo where you touched a handful of files), the cacheable work shrinks to $0.2 \times 35 = 7$ minutes, and total time is $3 + 7 = 10$ minutes. At $h = 0.94$ — touched 12 of 200 targets — the cacheable work is $0.06 \times 35 \approx 2.1$ minutes, total about 5 minutes. The relationship is the reason a high-hit-rate build feels almost instant: as $h \to 1$, the build time collapses toward the fixed overhead $f$. The speedup factor is approximately $\frac{T_{\text{cold}}}{f + (1-h)(T_{\text{cold}} - f)}$, which at 80% hit-rate gives $\frac{38}{10} \approx 3.8\times$ and at 94% gives roughly $7\times$.

Two consequences fall out of that formula and they shape how you optimize:

First, **fixed overhead becomes your floor**. Once hit-rate is high, you are no longer fighting compile time — you are fighting checkout, cache *restore* time (a huge `node_modules` cache can take a minute just to download and extract), and packaging. At that point the next win is shrinking `f`: shallow git clones, partial checkouts, smaller cache payloads, faster artifact upload. Chasing more compile caching when you are already at 94% is optimizing the wrong term.

Second, **hit-rate is fragile to cache busts**. A single spurious input that changes every run — a generated timestamp file declared as an input, a `.env` that differs per runner — can drag hit-rate from 94% to near zero, because it sits high in the dependency graph and invalidates everything downstream. This is where reproducibility (Sections 2–3) and hit-rate meet: nondeterminism in a target's output becomes nondeterminism in the *next* target's input hash, and the cache misses cascade. A non-reproducible build cannot maintain a high hit-rate, full stop. The two are the same property.

#### Worked example: cutting a 38-minute build to 6 minutes

This is the headline example, and I want to show every minute. A backend monorepo — 200 build targets across a dozen services and shared libraries — had a CI build that took 38 minutes from cold. The team batched their merges and dreaded the queue. Here is the decomposition of the cold build and the three interventions, in the order we applied them.

| Stage | Cold (no cache) | Intervention | Warm result |
| --- | --- | --- | --- |
| Dependency download | 8 min | dependency cache keyed on lockfile hash | ~0 min (restore) |
| Compile all 200 targets | 18 min | remote build-graph cache; rebuild only the 12 changed | 3 min |
| Lint + typecheck (serial) | 7 min | run independent targets in parallel across 4 cores | 2 min (overlapped) |
| Fixed overhead (checkout, package) | 5 min | shallow clone, smaller cache payload | 1 min |
| **Total** | **38 min** | | **~6 min** |

Walk the arithmetic. The dependency cache removes the 8-minute download on every warm run — the lockfile rarely changes, so this is nearly always a hit. The remote build-graph cache is the big one: a typical PR touches around 12 of the 200 targets, so the cache-hit-rate on the compile step is $\frac{200 - 12}{200} = 94\%$, and the 18-minute full compile collapses to about 3 minutes of building only the changed targets and their dependents — *and the first CI runner to build those 12 uploads them, so a developer rebasing the same branch on their laptop gets them for free.* Parallelism overlaps lint and typecheck with compile rather than running them end-to-end. And shrinking the fixed overhead (shallow clone, a leaner cache archive) trims the floor. The build went from 38 to 6 minutes — about a 6× speedup — and the before/after below shows the headline numbers side by side.

![Before and after diagram contrasting a cold build with eight minutes of dependency download eighteen minutes compiling all two hundred targets and a thirty eight minute total against a warm cached build with near zero dependency restore three minutes compiling twelve of two hundred targets and a six minute total six times faster](/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-6.png)

How would you *honestly* measure this rather than cherry-pick? Report the *median* warm-build time over a week of real PRs, not your best run, and report it alongside the *cold*-build time (cache fully evicted) so the reader knows the worst case. Track cache-hit-rate as a time series; a sudden drop means someone broke determinism. And watch CI cost: this team's CI bill dropped from roughly \$18k/mo to around \$6k/mo, not only because runs were shorter but because shorter runs meant fewer concurrent runners spun up during the workday. Faster builds are cheaper builds.

A caution on the numbers above: they are illustrative of the *shape* of the result, not a benchmark of a specific named product. The 38-minute cold build, the 94% hit-rate, and the \$18k-to-\$6k bill are the kind of figures I have repeatedly seen on real monorepos, but your cold time, your hit-rate, and your cost depend entirely on your codebase's graph shape and your runner pricing. The point to take is not "expect exactly 6 minutes" — it is the *relationship*: build time falls toward fixed overhead as hit-rate rises, and cost falls roughly in proportion to total runner-minutes. Measure your own three numbers (cold time, median warm time, hit-rate), publish them on a dashboard, and treat a regression in any of the three as a bug to be triaged, not a fact of life. The teams that sustain fast builds are the ones who watch the hit-rate the way they watch test coverage: a number with an owner, an alert when it drops, and a culture that treats a busted cache key as a defect rather than a nuisance.

## 7. Speed beyond caching: parallelism, incrementality, and right-sizing

Caching stops you from redoing work. The other half of speed is doing the *necessary* work faster, and there are four levers.

**Parallelism.** A build is a graph; independent targets can build at the same time. Build-graph tools do this automatically up to the number of cores you give them — Bazel's `--jobs`, Turborepo's `--concurrency`, `make -j`. At the *pipeline* level, you parallelize across runners with a matrix: build for multiple platforms or shard your test suite across N machines. GitHub Actions matrix jobs fan out:

```yaml
# Matrix build: fan out across platforms and Node versions in parallel.
# Each cell is its own runner, so wall-clock time is the SLOWEST cell,
# not the SUM. fail-fast: false so one red cell doesn't cancel the rest.
jobs:
  build-matrix:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        node: ["20.11.1", "22.2.0"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}
      - uses: actions/cache@v4
        with:
          path: ~/.local/share/pnpm/store
          key: pnpm-${{ matrix.os }}-${{ hashFiles('pnpm-lock.yaml') }}
      - run: corepack enable && pnpm install --frozen-lockfile
      - run: pnpm build && pnpm test
```

The win from parallelism is bounded by Amdahl's law: if a fraction $p$ of the work is parallelizable across $n$ workers, the speedup is $\frac{1}{(1-p) + p/n}$. A build that is 90% parallelizable maxes out at $10\times$ no matter how many cores you throw at it, because the serial 10% — the final link, the slowest single target on the critical path — dominates. The practical implication: parallelism helps until the *longest single chain through the build graph* (the critical path) is your wall-clock floor. After that, more runners just cost money. Find your critical path before you buy more parallelism.

**Incremental compilation.** Within a single target, many compilers can reuse prior compilation state — TypeScript's `--incremental` and project references, Rust's incremental compilation, `ccache` for C/C++, Gradle's incremental annotation processing. This is caching at a finer grain than the target level, and it is mostly free to turn on. It matters most for the "warm laptop" loop where a developer changes one file and recompiles.

**Not rebuilding the world.** This is the build-graph discipline restated as a workflow rule: when you change a leaf utility used by everything, you *will* rebuild the world, and that is correct. But most changes are local, and the build should reflect that. The failure mode is a build that has no real graph — a monolithic `make build` that recompiles everything regardless of what changed. Introducing a build graph (even a coarse one) so that "touched the docs service" does not rebuild the payments service is often a bigger win than any cache tuning.

**Splitting the build and right-sizing the runner.** Two structural levers. *Splitting* means decomposing a monolithic pipeline so that only the affected parts run — only build and test the services downstream of the changed files. (This is the heart of the planned sibling post *Monorepo vs polyrepo and scaling the pipeline*; the short version is "affected-detection," computing the set of targets reachable from the diff and running only those.) *Right-sizing the runner* means matching the machine to the work: a build that is CPU-bound and parallel wants many cores; one that is I/O-bound on a huge `node_modules` wants fast local SSD and more RAM for the filesystem cache. Putting a parallel compile on a 2-core runner wastes the parallelism; putting a serial build on a 32-core runner wastes the money. Measure where the build is bottlenecked — CPU, memory, disk, network — before you change the runner.

A practical caution on all four: speed work has sharply diminishing returns, and the order matters. *Cache first, then parallelize, then split, then right-size.* Parallelizing a build that redoes all its work just redoes all its work faster on more expensive hardware. Splitting a pipeline whose build is uncached just means more uncached builds. The cache is the lever with the best return because it attacks the largest term — the redundant work — and the others attack what is left.

## 8. The reproducibility bug, and the supply-chain payoff

Let me make the reproducibility failure concrete with the kind of bug that breaks "build once, promote everywhere" in a way that is maddening to diagnose, then show the fix and the bigger reason it matters.

#### Worked example: the build that embedded the clock

A service set its own version banner at build time. Somewhere in the build, a generated file did the moral equivalent of:

```javascript
// build-info.generated.js — written during the build step.
// This single line makes EVERY build produce a different artifact.
export const BUILD_INFO = {
  version: process.env.GIT_SHA,
  builtAt: new Date().toISOString(), // <-- the clock is now an input
};
```

The `Dockerfile` also used `FROM node:20` — a floating tag. Two consequences followed. First, two CI runs of the identical commit produced two different images, because `builtAt` differed by however many seconds elapsed between runs, and occasionally because `node:20` had advanced to a new patch. Second, and this is the part that broke delivery: the team's deploy automation rebuilt the image at promotion time (a separate, well-meaning "release build") rather than promoting the image that CI had already tested. So the image that ran in production had a *different digest* than the image that passed tests. They were promoting a different program than the one they had validated — the exact failure I opened the post with. A test passing on the staging image told them nothing reliable about the production image, because the two were not the same bytes.

The fix had three parts:

```dockerfile
# 1) Pin the base image by digest (kills the floating-tag drift).
FROM node:20.11.1-bookworm-slim@sha256:1d3f6f0e... AS build
WORKDIR /app

# 2) Declare the clock as an input. SOURCE_DATE_EPOCH is the
#    cross-tool standard: build tools that honor it use THIS value
#    instead of the wall clock for any embedded timestamp.
ARG SOURCE_DATE_EPOCH
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
ENV TZ=UTC LANG=C.UTF-8

COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY . .
RUN pnpm build
```

```yaml
# 3) In CI, derive SOURCE_DATE_EPOCH from the COMMIT time, not "now",
#    and build the image ONCE. Downstream stages PROMOTE this exact
#    digest; nothing ever rebuilds at deploy time.
- name: Compute deterministic build date from the commit
  run: echo "SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)" >> "$GITHUB_ENV"

- name: Build once, by digest
  run: |
    docker buildx build \
      --build-arg SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
      --provenance=true \
      --output=type=image,name=ghcr.io/acme/web,push=true \
      .
```

The generated `build-info` file changed to read `SOURCE_DATE_EPOCH` instead of calling `new Date()`. After that, the wall clock was a *declared* input pinned to the commit time, so two builds of the same commit produced the same `builtAt`, the floating base tag was pinned, and — critically — the deploy stopped rebuilding and started promoting the tested digest. Two CI runs of the same commit now produced byte-identical images. The "is the staging artifact the production artifact?" question had a provable yes.

Now the bigger payoff, the one that makes this worth doing even when your builds are fast enough. Once a build is reproducible, the artifact is *verifiable against its source*. An independent party — your security team, an auditor, an open-source community — can take the published source at the commit, rebuild it hermetically, and compare digests. If they match, you have proof the artifact was not tampered with between source and binary. If they differ, you have caught either a non-reproducibility bug or an actual compromise. The flow is below.

![Graph diagram showing pinned source at a commit feeding a hermetic build using SOURCE_DATE_EPOCH to produce an artifact with a sha256 digest while the same source feeds an independent rebuild and the two paths converge on a verify step where matching digests mean verified and differing digests mean investigate](/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-8.png)

This is the basis of software supply-chain security frameworks. **SLSA** defines levels of build integrity, the higher of which require a hermetic, reproducible build that emits signed **provenance** — a signed statement of *what source and what builder produced this artifact*. Tools like cosign (Sigstore) sign the artifact and its provenance; consumers verify the signature and, ideally, reproduce the build to confirm the provenance is honest. We cover signing and SBOMs in the supply-chain posts of this series; the point here is upstream of all of it. *You cannot meaningfully attest "this binary came from this source" if rebuilding the source produces a different binary.* Reproducibility is the precondition for trustworthy provenance, the same way it was the precondition for correct caching. One property — output depends only on declared inputs — buys you speed, debuggability, and supply-chain trust simultaneously. That is why I keep insisting the three goals are one goal.

## 9. Build once, promote everywhere: the principle the build stage exists to serve

Everything above converges on one delivery principle that is worth stating directly, because it is the reason the build stage matters so much: **build once, promote everywhere.** You build a single artifact, exactly once, from a single commit. You test *that* artifact. You scan *that* artifact. You sign *that* artifact. And then you take *that same artifact* — the same bytes, identified by the same content digest — and promote it through your environments: dev, then staging, then production. At no point does any environment rebuild from source. The artifact is immutable; what changes between environments is only the *configuration* injected at deploy time (environment variables, secrets, replica counts, feature flags), never the artifact itself.

The alternative — and it is astonishingly common, usually because it grew organically rather than by decision — is **rebuild per environment**: a "staging build" produces a staging image, and a separate "production build" produces a production image. The two builds may use the same `Dockerfile` and the same commit, but as we have seen at length, unless the build is hermetic and reproducible they will produce *different artifacts*, and you will deploy to production something you never actually tested. The non-reproducible bug from Section 8 was exactly a rebuild-per-environment setup hiding behind a shared `Dockerfile`. The comparison below makes the trade-offs explicit.

| Dimension | Rebuild per environment | Build once, promote everywhere |
| --- | --- | --- |
| What production runs | A *fresh* build — possibly different bytes than tested | The *exact* artifact that passed every gate |
| Confidence from a green test | Low: you tested a different artifact | High: you tested *these* bytes |
| Dependence on reproducibility | Hidden, fatal: drift silently ships untested code | Explicit: one build, no drift possible |
| Build cost | N builds per release (one per env) | One build per release |
| Rollback | Rebuild the old version (slow, may drift again) | Re-point to the prior digest (instant, identical) |
| Provenance | Ambiguous: which build is "the" build? | One digest, one provenance statement |
| Failure mode | "Worked in staging, broke in prod" | Configuration bugs only, isolated and obvious |

The reason "build once" is correct is almost tautological once you see it: *the only way to be sure production runs what you tested is to ship the literal artifact you tested.* Any rebuild, no matter how careful, reintroduces the entire surface of non-determinism from Section 1. And the reason teams nonetheless rebuild per environment is usually that their build was never trustworthy enough to promote — the artifact felt disposable, so they treated it as disposable, rebuilding whenever they needed it. This is a vicious circle: a non-reproducible build makes "build once" feel unsafe, and rebuilding per environment guarantees the build stays untrusted. The way out is the first half of this post: make the build hermetic and reproducible, *then* the artifact becomes trustworthy enough to promote, *then* you build once. The build stage is, in the end, the place where the artifact earns the right to be promoted rather than rebuilt.

There is a measurable payoff too. A team that rebuilds per environment pays for N builds per release and absorbs the lead-time cost of each one; "build once" pays for a single build and reduces release lead time accordingly. More importantly, rollback becomes trivial: because every released version is an immutable, content-addressed artifact still sitting in the registry, rolling back is re-pointing the deployment at the *prior digest* — milliseconds, and provably identical to what ran before — rather than rebuilding the old commit and praying it reproduces. That single property turns rollback from a risky 20-minute rebuild into a near-instant, deterministic operation, which is one of the largest single contributors to a low time-to-restore. The build stage's discipline directly buys you a DORA metric three stages downstream.

#### Worked example: lead-time decomposition before and after

Make the lead-time win concrete. **Lead time for changes** is the time from a commit landing to that commit running in production. Decompose it for a team that rebuilds per environment:

$$T_{\text{lead}} = T_{\text{build,CI}} + T_{\text{test}} + T_{\text{build,staging}} + T_{\text{verify}} + T_{\text{build,prod}} + T_{\text{deploy}}$$

With their slow, uncached, rebuilt-thrice setup: $38 + 6 + 38 + 10 + 38 + 5 = 135$ minutes, and that is the *happy path* with no requeue. Now apply both principles from this post. Cache the build so each build is 6 minutes instead of 38, and adopt build-once so there is exactly *one* build whose artifact is promoted:

$$T_{\text{lead}} = T_{\text{build,once}} + T_{\text{test}} + T_{\text{promote,staging}} + T_{\text{verify}} + T_{\text{promote,prod}} + T_{\text{deploy}}$$

which becomes $6 + 6 + 0.2 + 10 + 0.2 + 5 \approx 27$ minutes — a promotion is just re-pointing a digest, so it costs seconds, not a rebuild. Lead time fell from 135 minutes to about 27, a 5× improvement, from two changes that are really one change: *make the build a trustworthy, cacheable pure function so you can build it once and promote it fast.* Measure this honestly by instrumenting your pipeline with per-stage timestamps and reporting the median over real changes; the two terms that dominate are almost always the build (fixed by caching) and the redundant rebuilds (fixed by build-once).

## 10. The slow-build death spiral

There is a failure mode that is not about any single build run but about what slow builds do to a *team* over months, and it is worth naming because it is self-reinforcing and most teams are somewhere on it without realizing.

It starts innocuously. The build is slow — say 38 minutes. Waiting 38 minutes for feedback is painful, so engineers, being rational, *batch* their changes: rather than push a one-line fix and wait, they accumulate several changes and push them together to amortize the wait. Batching makes each diff *bigger*. Bigger diffs are harder to review, so reviews slow down and get shallower. Bigger diffs also touch more files, so they *conflict* more with other people's bigger diffs, producing merge conflicts and rework. The rework triggers more slow builds. And eventually the most corrosive adaptation appears: people start *skipping* CI — merging with admin override, testing only locally, "I'll just hotfix it" — because the pipeline is an obstacle rather than a safety net. Now you have large, unreviewed, conflict-prone changes going to production with the safety checks routed around. The timeline below traces the loop.

![Timeline diagram of the slow build death spiral progressing from a slow thirty eight minute build to batching changes to bigger diffs that are harder to review to merge conflicts and rework to skipping CI with local shortcuts and finally to breaking the spiral with caching and parallelism](/imgs/blogs/the-build-stage-reproducible-fast-and-cacheable-7.png)

The reason this matters in DORA terms is direct. Three of the four DORA metrics degrade along the spiral. **Lead time for changes** rises because changes sit batched, waiting. **Change-failure rate** rises because big, shallowly-reviewed, conflict-laden changes break more often. And **deploy frequency** falls because nobody deploys a half-finished batch. The fourth, time-to-restore, gets worse too, because when a big batch breaks, you cannot tell *which* of the bundled changes did it. A slow build is not a local inconvenience; it is an upstream cause of poor delivery performance across the board. This is also why the [continuous integration discipline of merging early and often](/blog/software-development/ci-cd/continuous-integration-merge-early-merge-often) and a fast build are two sides of one practice: small batches *require* fast feedback to be tolerable, and fast feedback *enables* small batches. You cannot ask engineers to merge ten times a day into a pipeline that takes 38 minutes; the economics of their attention will not allow it.

Breaking the spiral is the work of this entire post: cache the build (Sections 4–6) so feedback is fast, make it reproducible (Sections 2–3) so the cache is correct and the artifact is trustworthy, and parallelize and split (Section 7) so it stays fast as the codebase grows. The order is "fast feedback first, everything else follows," because every other delivery improvement is downstream of being able to get an answer in minutes.

## 11. A complete, copy-and-adapt CI build job

Let me assemble the pieces into one realistic GitHub Actions workflow that builds reproducibly, caches on three levels, and builds once for promotion. This is close to what I would actually ship, with the reasoning in comments.

```yaml
# .github/workflows/build.yml
# Goals: hermetic + reproducible artifact, three-layer caching,
# build ONCE then promote the same digest. Tuned for a TS monorepo.
name: build
on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read
  packages: write
  id-token: write # for OIDC keyless signing later in the pipeline

concurrency:
  # Cancel superseded runs on the same ref so we don't pay for stale builds.
  group: build-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1 # shallow clone shrinks the fixed-overhead floor

      - uses: actions/setup-node@v4
        with:
          node-version: "20.11.1" # pinned toolchain, not "20"

      # LAYER 1: dependency cache, keyed on the lockfile hash.
      - name: Cache pnpm store
        uses: actions/cache@v4
        with:
          path: ~/.local/share/pnpm/store
          key: pnpm-${{ runner.os }}-${{ hashFiles('pnpm-lock.yaml') }}
          restore-keys: pnpm-${{ runner.os }}-

      # LAYER 3: build-graph remote cache (Turborepo) so unchanged
      # targets are fetched, not rebuilt — shared across CI and laptops.
      - name: Cache turbo build graph
        uses: actions/cache@v4
        with:
          path: .turbo
          key: turbo-${{ runner.os }}-${{ github.sha }}
          restore-keys: turbo-${{ runner.os }}-

      - run: corepack enable && corepack prepare pnpm@9.1.0 --activate
      - run: pnpm install --frozen-lockfile # frozen = hermetic deps

      # Deterministic clock: pin embedded timestamps to the commit time.
      - name: Deterministic build date
        run: echo "SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)" >> "$GITHUB_ENV"

      # Build only affected targets; the graph cache handles the rest.
      - run: pnpm turbo run build test lint --concurrency=4

      # LAYER 2: build the image ONCE, with BuildKit layer caching to
      # the registry. Downstream jobs PROMOTE this digest; never rebuild.
      - name: Build & push image once
        run: |
          docker buildx build \
            --build-arg SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
            --cache-from=type=registry,ref=ghcr.io/acme/web:buildcache \
            --cache-to=type=registry,ref=ghcr.io/acme/web:buildcache,mode=max \
            --provenance=true \
            --tag ghcr.io/acme/web:${{ github.sha }} \
            --output=type=image,push=true \
            .

      # Emit the digest so promotion steps reference bytes, not a tag.
      - name: Record image digest
        run: |
          digest=$(docker buildx imagetools inspect \
            ghcr.io/acme/web:${{ github.sha }} \
            --format '{{json .Manifest.Digest}}')
          echo "image: ghcr.io/acme/web@${digest}" >> "$GITHUB_STEP_SUMMARY"
```

A few things to notice that distinguish this from a naive workflow. The toolchain is pinned to a patch version. Dependencies install frozen. Three cache layers are wired with correct keys — lockfile hash for deps, the graph cache for compile, registry-backed BuildKit cache for image layers. The clock is pinned via `SOURCE_DATE_EPOCH`. `concurrency` cancels superseded runs so a flurry of pushes does not pile up cost. And the image is built *once* and identified by digest so that everything downstream promotes the tested bytes. That last point connects to the next stage of the pipeline: the planned sibling *Building images fast and securely in CI* goes deep on multi-stage builds, distroless base images, and registry cache backends; here the build's job is to emit one trustworthy, content-addressed artifact for the rest of the pipeline to carry forward unchanged.

## 12. War story: the dependency that wasn't there twice

Two real-world episodes drive home why "the network at build time is an input" is not academic.

The first is the **`event-stream` incident** of 2018, an npm supply-chain attack. A widely-used package changed maintainership; the new maintainer published a version that added a malicious dependency designed to steal cryptocurrency wallet keys. Any build that resolved dependencies *fresh from the network* without a strict lockfile, during the window the malicious version was live, pulled the compromised code into its artifact. Builds that installed from a committed lockfile with integrity hashes — and that did not happen to update the lockfile during the window — were protected, because the lockfile pinned the *content hash* of the known-good version, and a different payload would have produced a different hash and failed integrity verification. The lesson is exactly Section 2's: a lockfile with integrity hashes turns "fetch whatever the registry serves" into "fetch exactly these bytes or fail," which is the difference between a build that is a pure function of declared inputs and one that is a function of the internet's current mood.

The second is **dependency-confusion**, demonstrated publicly by a researcher in 2021 against dozens of major companies. The attack exploits build tools that, given a package name, search *both* a private internal registry and the public one and pick the highest version. By publishing a malicious package on the public registry with the same name as a company's internal package but a higher version number, an attacker could get the build to fetch the public, malicious version instead of the intended private one. The root cause, again, is an undeclared, uncontrolled input: the build's *resolution* of which registry and which version to trust was ambient, not pinned. The fixes — scoped registries, explicit registry configuration per scope, lockfiles that pin the exact source, and pull-through caches that never fall back to public — are all instances of *declare and pin every input*. A build that can only fetch from a vetted, content-addressed source cannot be confused, because there is no ambient choice left to make.

There is a third story worth a sentence, from the other end of the spectrum: **Knight Capital**, 2012, which lost \$440 million in 45 minutes. It is usually told as a deploy story, and it is — old code was left on one of eight servers because the deploy was manual and inconsistent — but the build/artifact lesson is real: there was no single, immutable, promoted artifact whose presence on every server could be verified. The fix that whole class of incident demands is the spine of this series: *one artifact, built once, promoted everywhere, with provenance you can check.* The build stage is where that one artifact is born, and if it is born non-reproducible, none of the downstream guarantees hold.

## 13. How to reach for this (and when not to)

Every practice in this post has a cost, and a senior engineer's job is to know which to pay for and when. Here is my honest decision guide.

**Always do this, even on day one, even on a tiny team:** pin your toolchain version, commit a lockfile, and use the *frozen* install command (`npm ci`, `pnpm install --frozen-lockfile`, `uv sync --frozen`). This is nearly free, it eliminates the single most common class of build drift, and skipping it buys you nothing. There is no team too small for a lockfile honored strictly. Likewise, **dependency caching keyed on the lockfile hash** is a few lines of YAML and an immediate win; do it as soon as your build downloads anything.

**Do this once your build is slow enough to hurt or your repo is large enough to have a real graph:** adopt a build-graph tool (Bazel/Gradle/Nx/Turborepo) with declared inputs and outputs, and turn on **remote caching** shared across CI and developers. This has real setup cost and ongoing maintenance — you must keep input declarations honest, run a cache server, and teach the team — so it pays off when you have many targets and frequent partial changes (a monorepo, a large service). For a single small service with a 4-minute build, it is over-engineering; a simple dependency cache plus good `Dockerfile` ordering is enough.

**Do this when you ship software others must trust, or you are subject to supply-chain requirements:** pursue strict bit-reproducibility and signed provenance (`SOURCE_DATE_EPOCH`, stripping nondeterminism, cosign, SLSA). The effort to chase the last sources of nondeterminism is real and can be tedious. It is unambiguously worth it for software distributed to users, for regulated industries, and for anything where "prove the binary matches the source" is a requirement. For an internal-only service consumed by your own cluster, hermeticity (recoverable, consistent builds) is the floor that matters, and you can climb toward bit-identical as the security program matures rather than front-loading it.

**Do NOT** parallelize or buy bigger runners before you have cached the build — you will just pay more to redo the same work faster. **Do NOT** add a remote build-graph cache to a build that is not hermetic — you will serve stale, wrong artifacts and trust them. **Do NOT** rebuild per environment to "be safe"; that *is* the unsafe path, because it discards the artifact you tested. And **do NOT** chase bit-reproducibility on a three-person internal tool that ships twice a year — the return is not there yet; spend the effort on a lockfile and a dependency cache and move on. The ordering rule that governs all of it: *reproducible first (so caching is correct), then cache (the biggest lever), then parallelize and split (what's left), then right-size (the floor).*

## 14. Stress-testing the design

A design is only as good as its behavior under the bad cases. Let me push on the build stage the way I would in a review.

**What if the cache is cold?** A cold cache — first build on a new runner, or after a cache eviction — must still produce a correct artifact, just slowly. This is why you always report the cold-build time alongside the warm one: the cold path is your worst case and your disaster-recovery time. A cold build that takes 38 minutes is acceptable as a rare event; a cold build that *fails* (because it depended on something only present in a warm cache) is a latent bug. Test the cold path periodically by evicting the cache on purpose.

**What if the remote cache is down or unreachable?** The build must *fall back to building locally*, not fail. This is why Bazel's `--remote_local_fallback=true` exists and why you should configure the equivalent. A cache is an optimization; treating it as a hard dependency makes your build *less* reliable than no cache at all. The correct failure mode for a cache outage is "slow," never "broken."

**What if a cache entry is poisoned or stale?** This is the nightmare and the reason reproducibility is non-negotiable for cached builds. If a too-broad key let a stale or wrong output into the cache, every consumer gets the wrong artifact and *trusts it*. Defenses: content-addressed keys (the key is the hash of the inputs, so a different input cannot collide onto the same entry), integrity verification on restore, and the ability to bust the entire cache by bumping a key prefix (`turbo-v2-…`) when you suspect poisoning. Build-graph tools that key on content hashes are structurally resistant to this; hand-rolled caches keyed on branch names are structurally vulnerable.

**What if two PRs build the same target concurrently?** With a content-addressed remote cache, this is fine and even beneficial — both compute the same hash, both may build the target (a small duplicated effort), and both upload an identical result to the same key. The cache is *write-idempotent* because the key is the content hash: writing the same bytes to the same key twice is harmless. This is a property you get for free from content-addressing and would have to engineer carefully in a mutable cache.

**What if the build is fast but the artifact is wrong?** This is the worst outcome and the one a too-broad cache key produces: a green, fast build serving a stale dependency set or a mis-cached output. It is worse than a slow build because it destroys trust in the green check. The defense is the whole first half of this post: hermeticity plus correct, content-addressed keys plus periodic cold-build verification. *Fast and wrong is a regression from slow and right.* Never optimize for speed in a way that can produce a wrong artifact; correctness is not negotiable and caching, done right, does not require trading it away.

## 15. Key takeaways

- **The build is the highest-leverage stage.** It dominates pipeline wall-clock time and hides most non-determinism, so fixing it pays off everywhere downstream.
- **Reproducible, fast, and cacheable are one property, not three.** A build whose output depends only on declared inputs is cacheable; a cacheable build is fast; and that same property makes the artifact verifiable. Get hermeticity right and the rest follows.
- **"Works on my machine" is always an undeclared input.** Toolchain version, system libs, network-fetched deps, locale, timezone, or the wall clock. Declare and pin every one.
- **Pin the toolchain by digest, install dependencies frozen, and cut off the network during the build.** A lockfile is necessary but useless unless you use the *frozen* install command.
- **Cache on three independent layers.** Dependency cache (keyed on the lockfile hash) saves the download; layer cache (via `Dockerfile` ordering) saves the install; build-graph remote cache (keyed on content hash, shared across CI and laptops) saves the compile. They compose; they do not substitute.
- **The cache key must capture exactly the true inputs.** Too broad serves stale, wrong results silently; too narrow never hits and quietly wastes money. Key on content, not on branch names or commit SHAs.
- **Cache-hit-rate is the metric.** Build time collapses toward fixed overhead as hit-rate rises; an 80% hit-rate is roughly a 4× speedup, 94% roughly 7×. A sudden hit-rate drop means someone broke determinism.
- **Cache before you parallelize, parallelize before you split, split before you buy bigger runners.** Each later lever attacks a smaller term; doing them out of order wastes money redoing work faster.
- **A non-reproducible build breaks "build once, promote everywhere" and poisons the supply chain.** Pin the base image, strip embedded timestamps with `SOURCE_DATE_EPOCH`, and build once for promotion so the artifact you tested is the artifact you ship.
- **Reproducibility is the precondition for trust.** It makes the cache correct *and* lets a third party rebuild your source and verify the binary, which is the foundation of SLSA provenance and supply-chain security.

## 16. Further reading

- *Accelerate: The Science of Lean Software and DevOps* (Forsgren, Humble, Kim) and the annual **State of DevOps / DORA** reports — the empirical case that fast, reliable delivery (which starts with fast builds) predicts organizational performance.
- The **reproducible-builds.org** project — the canonical guide to eliminating sources of nondeterminism, including the `SOURCE_DATE_EPOCH` specification used above.
- The **SLSA** framework (slsa.dev) — build-integrity levels, provenance, and why hermetic reproducible builds underpin supply-chain security; pair it with the **Sigstore/cosign** docs for signing and verification.
- The **Bazel** remote-caching docs and the **Turborepo** remote-cache docs — the two clearest treatments of content-addressed build-graph caching shared across CI and developers.
- The **GitHub Actions `actions/cache`** documentation and the **Docker BuildKit** cache-backend docs — the practical caching primitives the workflows above are built on.
- The **Twelve-Factor App** (factor II, Dependencies) — the discipline of explicitly declaring and isolating dependencies, which is hermeticity stated as an app-design principle.
- Within this series: [the CI/CD pipeline map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) for the commit-to-production spine; [continuous integration: merge early, merge often](/blog/software-development/ci-cd/continuous-integration-merge-early-merge-often) for why fast builds and small batches are one practice; the planned *Building images fast and securely in CI* for multi-stage and distroless container builds; and the planned *Monorepo vs polyrepo and scaling the pipeline* for affected-detection and pipeline sharding.
- Out of series: [progressive delivery as an SLO-gated reliability practice](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) for what happens to the artifact after the build; [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability) at the service-fleet level; and [the flaky test: find it, fix it, or quarantine it](/blog/software-development/debugging/the-flaky-test-find-it-fix-it-or-quarantine-it) for the test-stage analogue of build non-determinism.

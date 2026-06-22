---
title: "Writing a Production Dockerfile: Small, Fast, and Hard to Break Into"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The Dockerfile most teams ship is a 2 GB security liability that rebuilds from scratch on every commit. A handful of techniques — multi-stage builds, layer-cache ordering, minimal bases, and a nonroot user — take it to 80 MB and a clean scan. This is the field manual."
tags:
  [
    "ci-cd",
    "devops",
    "docker",
    "dockerfile",
    "containers",
    "multi-stage-build",
    "distroless",
    "image-security",
    "supply-chain",
    "build-cache",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/writing-a-production-dockerfile-1.png"
---

A security scanner once told me an image we shipped to production every day had two hundred and eighty known vulnerabilities, eleven of them rated critical. My first reaction was to argue with the scanner. We did not write two hundred and eighty bugs. We barely wrote enough code to have eleven bugs. But the scanner was right and I was wrong, and the reason is the whole point of this post: almost none of those vulnerabilities were in our code. They were in the operating system, the language runtime, the build toolchain, the compilers, the package manager, the dev dependencies, and the dozens of system libraries that came along for the ride because our `Dockerfile` started with `FROM node` and ended with `COPY . .` and `npm install`. We were shipping an entire Linux distribution and a full software development kit to run a thirty-megabyte web service. The image was 1.9 GB. It pulled slowly, it rebuilt from scratch on every single commit, it ran as root, and it carried a CVE count that made our security team wince. And here is the part that should bother you: this was not an unusually bad Dockerfile. It was the *normal* one. It was the one that works, the one that the tutorial gives you, the one that ships at thousands of companies right now.

That is the gap this post closes. The Dockerfile most teams ship is a 2 GB security liability that rebuilds from scratch every time, and fixing it is the single highest-leverage, lowest-effort delivery win available to most engineering teams. You do not need a platform team, a new tool, or a migration. You need to understand four ideas and apply them in about forty lines of a file you already have. The four ideas are: split the build toolchain from the runtime so you ship only the artifact; order your layers so the cache actually helps you; pick the smallest base image you can debug; and drop root. Apply them and the same service that shipped as a 1.9 GB image with two hundred and eighty CVEs ships as an 80 MB image with a handful — pulls in seconds, rebuilds in seconds when only code changed, and has almost nothing in it for an attacker to exploit if they get in.

![Before and after comparison of a naive single-stage Docker image at 1.9 GB with hundreds of CVEs running as root next to a multi-stage distroless image at 80 MB with almost no CVEs running as a nonroot user](/imgs/blogs/writing-a-production-dockerfile-1.png)

This post is part of the series **"CI/CD & Cloud-Native Delivery, From Commit to Production."** If you have not read [the CI/CD pipeline map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) yet, start there — it lays out the spine the whole series returns to: **commit → build → test → package → deploy → operate**, governed by two principles, **"build once, promote everywhere"** and **"everything as code,"** measured by the four DORA metrics. The Dockerfile sits at the *package* stage, and it is where "build once, promote everywhere" becomes a physical object. The image you produce here is the immutable artifact that flows, unchanged, through staging and into production. If that artifact is bloated and insecure, every environment inherits the bloat and the insecurity. If it is small and hardened, every environment inherits that instead. So my claim — the spine of everything below — is this: **a production Dockerfile is small, fast to build, cache-friendly, and minimal in attack surface, and a handful of techniques get you all four at once, because they are the same goal seen from different angles.** A multi-stage build makes the image small *and* shrinks the attack surface. A smaller base makes the image small *and* faster to pull *and* cheaper to patch. Ordering layers for the cache makes the build fast *and* cheaper in CI minutes. By the end you will be able to read a `docker history`, diagnose why your image is huge and slow, write a multi-stage production Dockerfile with a distroless runtime and a nonroot user, and defend every line of it on size, speed, and security grounds.

A quick word on vocabulary, because we will use it constantly. An **image** is a read-only, layered filesystem plus metadata that you build once and run many times. A **layer** is the filesystem diff produced by a single build instruction — each `RUN`, `COPY`, or `ADD` makes one. A **base image** is the image you start `FROM`; everything you build sits on top of it. A **container** is a running instance of an image. A **registry** (GHCR, ECR, Docker Hub, Artifact Registry, Harbor) is where images are stored and pulled from. A **CVE** is a publicly catalogued vulnerability; a scanner counts how many known CVEs are present in the packages inside your image. **Attack surface** is the sum of everything in the image that an attacker could exploit — every package, binary, and shell. The whole game of a production Dockerfile is to keep all of those numbers small.

## 1. Why the normal Dockerfile is a liability

Let us start with the image that everybody ships and pull it apart honestly. Here is a single-stage Node service Dockerfile of the kind you will find in a thousand repositories:

```dockerfile
# the naive single-stage Dockerfile — DO NOT ship this
FROM node:20

WORKDIR /app

COPY . .

RUN npm install

EXPOSE 3000

CMD ["node", "server.js"]
```

It works. It builds, it runs, it serves traffic. And it is wrong in at least five ways that compound into a real cost.

First, the base. `FROM node:20` without a suffix pulls the *full* Debian-based Node image, which is roughly a gigabyte before you add a byte of your own code. It contains a complete operating system: `apt`, `bash`, `git`, `curl`, `python3` (because `node-gyp` wants it), build-essential, a C compiler, and shared libraries you will never call. None of that is needed to *run* a Node server. All of it is needed to *build* one, maybe, and you have conflated the two.

Second, `COPY . .` copies your entire working directory into the image — including `node_modules` if it exists locally, your `.git` history, your `.env` file if you have one, your test fixtures, your CI logs, and whatever else is lying around. This bloats the image and, worse, can bake a secret into a layer where it lives forever in the image history even if a later instruction deletes it.

Third, `COPY . .` *before* `npm install` is the cache-busting mistake. Because the `COPY` instruction changes whenever any source file changes, every commit invalidates that layer and every layer after it — including `npm install`. So every push reinstalls every dependency from scratch even when you only changed one line of one file. We will quantify this in section 4; for now, hold the thought that a one-character change triggers a full dependency reinstall.

Fourth, the dev dependencies. `npm install` installs `devDependencies` — your test framework, your linter, your TypeScript compiler, your bundler — all of which are needed to build but not to run. They sit in the production image, adding size and CVEs, doing nothing.

Fifth, and most dangerously, the container runs as **root**. The Node base image does not switch users, so `CMD ["node", "server.js"]` runs as UID 0. If an attacker finds a remote-code-execution bug in your service or one of its dependencies, they get a root shell inside a container that has a full operating system, a package manager, and a compiler — everything they need to escalate, pivot, and persist. And container root is not as isolated as people assume: by default it maps to root on the host kernel, so a container-escape vulnerability turns "root in the container" into "root on the node."

Put those together and you have the image I described in the intro: 1.9 GB, two hundred and eighty CVEs, root, and a full rebuild on every commit. None of those properties is a bug you introduced. They are the *defaults*. A production Dockerfile is, more than anything, a deliberate refusal of the defaults.

### The cost is real and measurable

It is tempting to wave this off as theoretical. It is not. Here is what each property costs you in a delivery pipeline you actually operate:

- **Pull time.** A 1.9 GB image has to be pulled by every node that schedules your pod. On a Kubernetes cluster scaling up under load, that is the difference between a pod that is ready in eight seconds and one that is ready in ninety. During an autoscaling event or a node failure, that delay is the difference between absorbing a traffic spike and dropping requests.
- **Build minutes.** Rebuilding every dependency on every commit means your image build is the slow stage of CI. We will see a build go from three minutes to eight seconds in section 4. Multiply the saved minutes by your commit volume and your runner cost and it is real money.
- **Patch toil.** Every CVE in the image is something your security team has to triage. Two hundred and eighty CVEs is two hundred and eighty rows in a dashboard, most of them in packages you never call but cannot easily remove. Drop to three and the dashboard becomes actionable.
- **Blast radius.** The root-plus-full-OS combination is what turns a single application bug into a host compromise. The smaller and less privileged the image, the less an attacker can do with a foothold.

So the four levers — multi-stage, layer ordering, minimal base, nonroot — are not aesthetic. Each one moves a number you measure: image size, build time, CVE count, or blast radius. The figure above is the destination. The rest of this post is the road.

## 2. Multi-stage builds: the single biggest lever

If you do only one thing from this entire post, do this. The multi-stage build is the single biggest lever on both size and attack surface, and it costs you nothing but a few extra lines.

The idea is simple once you see it. A Dockerfile can contain more than one `FROM`. Each `FROM` starts a new **stage** with its own filesystem. You give stages names with `AS`. You do all the heavy work — install the SDK, pull dev dependencies, compile, bundle, run codegen — in a `build` stage that has the full toolchain. Then you start a *fresh, tiny* `runtime` stage and `COPY --from=build` only the finished artifact into it. Everything in the build stage that you did not explicitly copy forward is **discarded**. The compilers, the dev dependencies, the `node_modules` you needed to bundle, the intermediate files — none of it ends up in the shipped image, because the shipped image is the runtime stage, and the runtime stage only ever received the one thing you copied.

![Graph showing source and lockfile flowing into a build stage with the full toolchain that compiles an artifact which is copied into a small distroless runtime stage producing the shipped image while the toolchain is left behind](/imgs/blogs/writing-a-production-dockerfile-2.png)

The figure above states the idea in one picture: the source and the toolchain flow into the build stage, the build stage produces an artifact, and only the artifact crosses into the runtime stage. The build stage is a scaffold you tear down. The shipped image never knew the compiler existed.

### The pattern for compiled languages

For compiled languages — Go, Rust, C, anything that produces a self-contained binary — the multi-stage win is dramatic, because the runtime needs *nothing* except the binary. Here is a Go service:

```dockerfile
# --- build stage: full Go toolchain ---
FROM golang:1.22 AS build
WORKDIR /src

# copy go.mod/go.sum first so the module-download layer caches
COPY go.mod go.sum ./
RUN go mod download

# now copy source and build a static binary
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/app ./cmd/server

# --- runtime stage: nothing but the binary ---
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/app /app
USER nonroot:nonroot
ENTRYPOINT ["/app"]
```

The build stage is a 900 MB Go image with the compiler and the standard library. The runtime stage is `gcr.io/distroless/static-debian12`, which is a couple of megabytes and contains no shell, no package manager, and no libc beyond what a static binary needs — and the `:nonroot` variant even ships a nonroot user for you. `CGO_ENABLED=0` makes the binary fully static so it does not need any shared libraries at runtime, which is what lets it run on the near-empty distroless base. The resulting image is the binary plus a sliver of base: often under 20 MB. For Rust the pattern is identical — a `rust:1.77` build stage, `cargo build --release`, then `COPY --from=build /src/target/release/app` into distroless or even `scratch` (literally the empty image) if the binary is fully static.

### The pattern for interpreted languages

For interpreted languages — Python, Node, Ruby — there is no single binary; the runtime needs the interpreter and the *production* dependencies. So the runtime stage is a slim image with the interpreter, and you copy forward only the installed production dependencies, not the dev ones and not the build tooling. Here is the Node version that replaces the naive Dockerfile from section 1:

```dockerfile
# --- build stage: full toolchain, dev deps, build the app ---
FROM node:20-bookworm AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# --- deps stage: production deps only ---
FROM node:20-bookworm AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# --- runtime stage: slim base + prod deps + built artifact ---
FROM node:20-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY --from=deps /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY package.json ./
USER node
EXPOSE 3000
ENTRYPOINT ["node", "dist/server.js"]
```

There is a small subtlety worth naming: there are *three* stages here, not two. The `build` stage compiles or bundles your code (it needs `devDependencies` such as TypeScript or a bundler). A separate `deps` stage installs production dependencies *only* with `npm ci --omit=dev`. The `runtime` stage then takes the production `node_modules` from `deps` and the built `dist` from `build`. Why split `deps` out instead of pruning in the build stage? Because the `deps` stage depends only on `package.json` and `package-lock.json`, so its expensive `npm ci` layer caches independently of your source — it only re-runs when your dependencies actually change. This is the multi-stage build and the cache-ordering principle working together, which is the recurring theme of this whole post: the levers compound.

The compiled-versus-interpreted distinction is worth holding in your head as a table, because it tells you which runtime base to reach for:

| Language | Build stage produces | Runtime needs | Runtime base |
|---|---|---|---|
| Go / Rust (static) | one static binary | nothing but the kernel | `scratch` or `distroless/static` |
| Go / Rust (CGO/dynamic) | binary + a few libs | glibc + linked libs | `distroless/base` |
| Java | a fat JAR | a JRE | `distroless/java` or a slim JRE |
| Python | wheels + source | interpreter + prod deps | `python:3.x-slim` or `distroless/python3` |
| Node | bundled `dist` | Node runtime + prod deps | `node:20-slim` or `distroless/nodejs20` |

The pattern is identical across the column: build with everything, ship with the minimum that runs. For compiled languages the minimum is almost nothing, which is why a Go service can ship as a 15 MB image on `scratch`. For interpreted languages the minimum is the interpreter plus production dependencies, which is why those land around 80–180 MB. Either way the build toolchain — the part with the most CVEs and the most for an attacker to use — never makes it into the shipped image, because the runtime stage only ever received what you explicitly copied forward.

### Why the runtime carries no build tooling

The security payoff deserves to be stated precisely, because it is the part people undersell. In the naive single-stage image, every tool used to *build* the app is also present to be *exploited* at runtime: the compiler, `apt`, `git`, the dev dependencies. If an attacker gets code execution, those tools are weapons — a compiler lets them build an exploit on the box, a package manager lets them install a backdoor, a shell lets them explore. In the multi-stage image, the runtime stage *never contained* those tools, so there is nothing to weaponize. This is not "we removed the tools at the end." It is "the tools were never in this image." A `RUN rm -rf` that deletes a tool in a single-stage build does not actually shrink the image, because the deleted bytes still live in the earlier layer's history; a multi-stage `COPY --from` genuinely leaves them behind, in a stage that is thrown away. That distinction — discarded stage versus deleted-but-still-in-history — is the entire reason multi-stage builds beat the old `RUN install && build && cleanup` one-liner.

#### Worked example: a Python service, single-stage to multi-stage

Suppose you have a Flask API. The naive image is `FROM python:3.12` (about 1.0 GB), `COPY . .`, `RUN pip install -r requirements.txt`, where `requirements.txt` includes `pytest`, `black`, `mypy`, and `ipython` alongside Flask and Gunicorn. Image size lands around 1.1 GB; the scanner reports roughly 200 CVEs (mostly in the OS and the full Python build). Now go multi-stage. The build stage `FROM python:3.12` installs everything and runs any build step. A runtime stage `FROM python:3.12-slim` (about 130 MB) installs only the runtime requirements — split your `requirements.txt` into `requirements.txt` (Flask, Gunicorn) and `requirements-dev.txt` (the rest) and `pip install` only the former. The slim base has no compiler, so if you have native wheels you build them in the build stage with `pip wheel` and `COPY --from=build` the wheels into the runtime stage. Result: roughly 180 MB and around 30 CVEs. Same code, same behavior, one-sixth the size, one-seventh the CVEs, and the test framework no longer ships to production. The arithmetic is mundane and the payoff is large: that is the signature of a high-leverage technique.

## 3. Layer-cache ordering: order least-changing first

The second lever does not change what is in your image at all. It changes how fast you can build it — and on a busy team, build speed is a daily tax that compounds. The rule is one sentence: **order your instructions from least-changing to most-changing.** Understand why and you will never write a cache-busting Dockerfile again.

![Stacked layer diagram showing the base image at the bottom rarely changing then the lockfile copy then the cached dependency install then the source copy that changes every commit then the artifact build on top](/imgs/blogs/writing-a-production-dockerfile-3.png)

The figure above shows the order you want: the base at the bottom (rarely changes), the lockfile copy and dependency install next (change weekly at most), the source copy near the top (changes every commit), and the build last. The instructions that change rarely sit below the instructions that change often, so a frequent change only invalidates the layers above it.

### How Docker's layer cache actually works

Here is the machinery, because the rule only makes sense once you know it. When Docker builds an image, it executes instructions top to bottom, and each instruction produces a layer. For each instruction, Docker computes a **cache key**. For most instructions (`RUN`, `ENV`, `WORKDIR`) the key is essentially the instruction text plus the identity of the layer beneath it. For `COPY` and `ADD`, the key *also* includes a checksum of the files being copied. Docker checks: do I already have a cached layer for this exact key, built on top of this exact parent layer? If yes, it reuses the cached layer instantly and moves on. If no, it executes the instruction, builds a fresh layer, and — this is the crucial part — **every instruction after it must also rebuild**, because their parent layer just changed, so their cache keys no longer match.

That cascade is the whole story. The cache is a prefix match: it holds until the first instruction whose inputs changed, and from that point down, everything rebuilds. So the question for every Dockerfile is: *where is the first layer that changes on a typical commit, and how much expensive work sits below versus above it?* You want the expensive work (dependency install) below the frequently-changing input (your source), so the expensive work stays cached.

The naive Dockerfile gets this exactly backwards. `COPY . .` includes your source, so its checksum changes on every commit, which busts that layer, which busts the `RUN npm install` that comes after it. The fix is to copy *only the lockfile* first, install dependencies, and *then* copy the source:

```dockerfile
FROM node:20-slim
WORKDIR /app

# 1. copy ONLY the manifest + lockfile (changes rarely)
COPY package.json package-lock.json ./

# 2. install deps — this layer caches until the lockfile changes
RUN npm ci --omit=dev

# 3. NOW copy the source (changes every commit)
COPY . .

# 4. build (cheap relative to install)
RUN npm run build

USER node
ENTRYPOINT ["node", "dist/server.js"]
```

Now a source-only change busts the `COPY . .` layer and the `RUN npm run build` after it — but `npm ci` sits *above* the source copy and its inputs (the lockfile) did not change, so it is reused from cache. You install dependencies only when dependencies actually change.

### The cache-hit-rate payoff, quantified

You can put a number on this. Let the cost of a cold build be $T_{cold}$ — everything runs, including the expensive install step that takes $T_{deps}$. Let $h$ be the fraction of builds that are *code-only* (no dependency change), which is the common case; on most teams $h$ is well over 0.9 because you change code far more often than you change dependencies. With cache-friendly ordering, the expected build time is:

$$ T_{expected} = (1 - h)\,T_{cold} + h\,(T_{cold} - T_{deps}) = T_{cold} - h\,T_{deps} $$

The savings scale directly with how dominant the dependency step is ($T_{deps}$) and how often you skip it ($h$). If installing dependencies is two minutes of a three-minute build and 95% of your builds are code-only, you save $0.95 \times 2 = 1.9$ minutes on almost every build. That is the difference between a build that gates a fast feedback loop and one that does not.

### The `.dockerignore` — speed and safety at once

There is a companion file that does two jobs at once: `.dockerignore`. Before Docker builds anything, it sends the **build context** — the directory you point `docker build` at — to the build daemon. If that directory contains `node_modules`, `.git`, build outputs, and logs, all of it gets shipped to the daemon before the build even starts, which is slow, and then `COPY . .` may copy it into the image, which is both bloat and a leak risk. A `.dockerignore` excludes paths from the context, exactly like `.gitignore` excludes them from commits:

```bash
# .dockerignore
.git
.gitignore
node_modules
npm-debug.log
dist
coverage
.env
.env.*
*.md
Dockerfile
.dockerignore
.github
**/__pycache__
**/*.pyc
.venv
```

Three wins. **Speed**: a smaller context transfers faster, especially over a remote build daemon or in CI. **Correctness**: excluding `node_modules` and `dist` means `COPY . .` cannot accidentally copy a stale local build into the image, which is a classic source of "works locally, breaks in the image" confusion. **Security**: excluding `.env`, `.git`, and credential files means you cannot accidentally bake a secret into a layer. That last one is not hypothetical — a `.git` directory copied into an image leaks your entire commit history, including any secret ever committed and later removed, because Git keeps the old blobs. The `.dockerignore` is five minutes of work that prevents a whole category of incidents.

One subtlety about how `.dockerignore` interacts with multi-stage caching: the build context is sent once, before any stage runs, and the patterns in `.dockerignore` apply to *every* `COPY` from the context across all stages. So a `.dockerignore` that excludes `node_modules` protects both your `COPY package*.json` step and your `COPY . .` step. It does *not* affect `COPY --from=<stage>` instructions, because those copy from another stage's filesystem, not from the context — which is exactly why a build stage can produce `node_modules` and the runtime stage can still `COPY --from=deps` them in. The rule to keep separate is: `.dockerignore` governs what enters the build from your machine; `COPY --from` governs what moves between stages inside the build. Confusing the two leads to the frustrating "I excluded it but it is still in my image" puzzle, which is almost always a `COPY --from` pulling the thing in from a stage rather than the context.

## 4. Worked example: the cache-busting reorder, 3 minutes to 8 seconds

Let us make the cache lever concrete with real numbers, because this is the one that pays you back every single day.

#### Worked example: reordering a Node build for the cache

A team I worked with had this build in their pipeline, and it took about three minutes on every push:

```dockerfile
# BEFORE — cache-busting order
FROM node:20
WORKDIR /app
COPY . .
RUN npm install
RUN npm run build
CMD ["node", "dist/server.js"]
```

![Before and after comparison of a Dockerfile that copies all files then installs dependencies rebuilding everything every commit versus one that copies the lockfile first so the install layer stays cached and rebuilds in eight seconds](/imgs/blogs/writing-a-production-dockerfile-5.png)

The figure above shows the two orderings side by side. On the "before" side, `COPY . .` changes whenever any file changes, so on every commit the cache is busted at that line and `npm install` re-runs in full — about two minutes and forty seconds of the three-minute build. The team committed dozens of times a day. Most of those commits changed one or two source files and touched no dependencies at all. They were paying the full dependency-install cost on every one.

Here is the reorder:

```dockerfile
# AFTER — cache-friendly order
FROM node:20-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY . .
RUN npm run build
USER node
ENTRYPOINT ["node", "dist/server.js"]
```

Now the build measures like this. The first build after a dependency change is still about three minutes — you cannot avoid installing dependencies when they actually change. But every *subsequent* build that touches only source code: the `FROM`, `COPY package*.json`, and `RUN npm ci` layers are all cache hits (their inputs did not change), so Docker skips straight to `COPY . .` and `RUN npm run build`. Those two steps — copying source and running an incremental TypeScript build — took about eight seconds. The build went from three minutes to eight seconds on the common path, with one reordering and zero new tools. Over a day of, say, forty code-only commits, that is roughly forty times two minutes fifty seconds saved versus forty times eight seconds — on the order of an hour and a half of runner time reclaimed *per day*, which is real money and, more importantly, a feedback loop that is fast enough to keep developers in flow.

Two honest caveats. First, this assumes the build runs somewhere the layer cache persists. On your laptop it always does. In CI, ephemeral runners start with a cold cache unless you wire up cache reuse — `docker buildx` with a registry or GitHub Actions cache backend (`--cache-to type=gha --cache-from type=gha`), or BuildKit's inline cache. We cover that in depth in the planned sibling post on building images fast and securely in CI; for now, know that the reorder is necessary but you also need the CI cache wired up to realize the gain. Second, `npm ci` (clean install from the lockfile) is the right command for CI and images — it is deterministic and fails if `package.json` and `package-lock.json` disagree — whereas `npm install` may mutate the lockfile and is for local development. Using `npm ci` is itself a small reproducibility win.

The stress test is worth a sentence too. *What if the build cache is cold* — a fresh runner, a cache eviction, a base-image bump? Then you pay the full cold build, exactly as you did before; the reorder never makes a cold build slower, it only makes warm builds dramatically faster. The reorder is strictly dominant: it cannot lose, and on the common path it wins big.

## 5. Minimal base images: the spectrum from ubuntu to scratch

The third lever is the base image, and it is a spectrum, not a binary choice. Every step down the ladder makes the image smaller, faster to pull, and cheaper to patch — and at some point, harder to debug. Knowing where to stand on the ladder is a judgment call, and this section gives you the basis for it.

![Matrix comparing five base image options from full ubuntu through debian slim alpine distroless and scratch across image size CVE surface whether a shell is present and the best use case for each](/imgs/blogs/writing-a-production-dockerfile-4.png)

The figure above lays out the spectrum. Here is the same comparison as a table you can reason about line by line.

| Base | Typical size | CVE surface | Shell / pkg manager | Best for |
|---|---|---|---|---|
| `ubuntu` / full `node` / `python` | 70 MB–1 GB+ | Large | Yes (`bash`, `apt`) | Build stages, interactive debugging |
| `debian:bookworm-slim` / `-slim` tags | ~30–130 MB | Medium | Yes (`bash`, `apt`) | Interpreted-language runtimes |
| `alpine` | ~6–50 MB | Small | Yes (`ash`, `apk`) | Small images where musl is fine |
| `gcr.io/distroless/*` | ~2–25 MB | Tiny | No | Production runtimes |
| `scratch` | 0 (empty) | None of its own | No | Fully static binaries (Go/Rust) |

Walk down it. The **full base** (`ubuntu`, unsuffixed `node`, `python`) is a complete OS — great for a build stage where you genuinely need the tools, terrible as a runtime because it carries all of them into production. The **slim** variants strip the OS down to what a runtime usually needs (`-slim`, `bookworm-slim`); for interpreted languages this is often the sweet spot for the runtime stage, because you still get the interpreter and a normal libc but shed the build tooling. **Alpine** goes further by using `musl` libc and `busybox` instead of glibc and GNU coreutils, which makes it tiny — but musl is a real gotcha, covered next. **Distroless** (Google's `gcr.io/distroless` images) contains your language's runtime and its essential shared libraries and *nothing else* — no shell, no package manager, no `ls`. **Scratch** is the empty image: zero bytes of base, suitable only for a fully static binary that needs nothing but the kernel.

Why does smaller matter so concretely? Three reasons, all measurable. **Pull speed**: fewer bytes means faster pulls, which means faster scale-up and faster recovery when a node dies. **Patch surface**: a CVE in `libxml2` is only your problem if `libxml2` is in your image; the fewer packages you ship, the fewer CVEs you inherit and the fewer you have to patch. A distroless image often has single-digit CVEs simply because there is almost nothing in it to be vulnerable. **Attack surface**: no shell means an attacker who gets code execution cannot drop into `bash` and explore; no package manager means they cannot `apt install` their toolkit. The image is hostile to them, not hospitable.

### The alpine / musl gotcha

Alpine is seductively small, and for many workloads it is fine. But it uses `musl` libc instead of the `glibc` that almost everything else assumes, and that mismatch bites in specific ways you should know before you reach for it. Precompiled binaries built against glibc may not run. Python wheels are the classic pain: PyPI's `manylinux` wheels are built against glibc, so on Alpine `pip` often cannot use them and falls back to compiling from source, which means you need a full build toolchain in the image (defeating the size win) and your builds get slow and flaky. There have also been historical DNS-resolution differences in musl that surprised people running services with lots of outbound calls. The honest guidance: Alpine is great for Go and other statically-linked binaries and for genuinely small footprints where you have tested the workload; for Python and other ecosystems that lean on glibc-built wheels, a `-slim` Debian base or distroless is usually less trouble for the same or better security posture. Smaller is not automatically better if it costs you a flaky build and a glibc-shaped bug.

### The distroless tradeoff

Distroless is, for most production runtimes, the right destination — but it has one real cost, and you should go in with eyes open. There is **no shell**. You cannot `kubectl exec` into a distroless container and run `bash`, because there is no `bash`, no `sh`, no `ls`, no `cat`. For engineers used to debugging by shelling into a container, this feels like losing a hand. The answer is that the modern way to debug a container is *not* to shell into the production image anyway — it is to use an **ephemeral debug container** (`kubectl debug --image=busybox --target=mypod`), which attaches a throwaway container with a full toolkit into the same process namespace as your distroless pod, without ever putting a shell in the production image. You get the debugging power exactly when you need it and none of the attack surface the rest of the time. The tradeoff, then, is: distroless gives you a far smaller CVE count and attack surface in exchange for changing *how* you debug, not *whether* you can. For a production service that is almost always the right trade. For a base image in a build stage, or for an exploratory spike, keep the shell — distroless is a runtime choice, not a universal one.

## 6. Pin the base by digest: reproducibility and supply chain

A subtle but important point sits underneath the base-image choice: *which exact image* are you starting from? A tag like `node:20-slim` is a moving target. The maintainers re-publish it every time they rebuild — to pick up OS security patches, a new minor Node version, whatever — so the `node:20-slim` you built on Monday is not necessarily the bytes you build on Friday. That is good for getting patches and bad for reproducibility: two builds of the same commit can produce different images because the base moved underneath them. This is exactly the "same commit, different artifact" failure that the [build-once-promote-everywhere principle](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) exists to prevent, and it is the same class of bug the [build stage post](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) digs into.

The fix is to pin the base by its **digest** — the content-addressed SHA-256 hash of the exact image manifest — instead of (or alongside) a mutable tag:

```dockerfile
# pinned by digest: this is byte-for-byte the same base every build
FROM node:20-slim@sha256:6f3d6e3...c4a91b AS runtime
```

A digest is immutable: `@sha256:...` always refers to the exact same bytes, forever, no matter how many times the tag is re-pushed. Pin it and your build is reproducible with respect to the base — the same commit produces the same image. This also closes a supply-chain hole. If an attacker compromised the registry and re-pushed a malicious image under the `node:20-slim` tag, a tag-based build would silently pull the poisoned base; a digest-based build would not, because the digest would not match. Pinning by digest is one of the cheapest supply-chain hardening steps you can take, and it ties directly into "build once": you cannot promote one trusted artifact if its very foundation can shift between builds.

The obvious objection: if I pin by digest, how do I get security patches to the base? You do not pin and forget. You let an automated dependency bot — Renovate or Dependabot — watch the tag and open a pull request that bumps the digest when the base is rebuilt. Now the update is a reviewable, tested, version-controlled change instead of a silent drift. You get reproducibility *and* a controlled patch cadence. That is the whole philosophy of "everything as code" applied to your base image: the version you depend on is a value in a file, changed by a PR, not a tag that mutates behind your back.

## 7. Drop root: least privilege in the image

The fourth lever is the one with the biggest security payoff per line of effort: do not run as root. By default, a container runs its process as UID 0 — root inside the container. People assume the container is a strong boundary, so root-in-container feels harmless. It is not as strong as you think, and the gap is exactly where breaches happen.

![Matrix comparing a container running as root against one running as a numeric nonroot user with a read-only filesystem across host privilege on escape ability to write the image filesystem and whether it passes an admission policy gate](/imgs/blogs/writing-a-production-dockerfile-7.png)

The figure above states the contrast bluntly. Here is the reasoning behind it. The Linux kernel is shared between the host and all containers; container isolation is built from namespaces and cgroups, not a separate kernel. By default, root in the container maps to root on the host (unless you have configured user-namespace remapping, which many setups do not). So if there is a container-escape vulnerability — and there have been several over the years, in the runtime, in the kernel, in misconfigured mounts — then "root in the container" becomes "root on the node," and now an attacker who compromised one pod owns the host and every other pod scheduled on it. Running as a nonroot user does not prevent the escape bug, but it dramatically reduces what the attacker can do if they get in: an unprivileged user cannot trivially escalate, cannot write to most of the filesystem, cannot bind low ports, cannot load kernel modules. Least privilege means the foothold is shallow.

So drop root. In the Dockerfile:

```dockerfile
FROM node:20-slim AS runtime
WORKDIR /app

# create a dedicated, numeric, unprivileged user
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=deps --chown=10001:10001 /app/node_modules ./node_modules
COPY --from=build --chown=10001:10001 /app/dist ./dist

# switch to it BEFORE the process starts
USER 10001:10001

ENTRYPOINT ["node", "dist/server.js"]
```

Two details matter. First, use a **numeric UID** in `USER`, not just a name. Kubernetes' `runAsNonRoot: true` security context check works on the UID, and a named user without a clear numeric UID can confuse the admission check; a numeric `USER 10001` is unambiguous. Distroless makes this trivial — `gcr.io/distroless/static-debian12:nonroot` already runs as UID 65532, so you write `USER nonroot:nonroot` and you are done. Second, the `--chown` on the `COPY` ensures the files are owned by your nonroot user, so the process can read them.

### Read-only root filesystem

Closely related and equally cheap: make the root filesystem **read-only**. A web service almost never needs to write to its own image filesystem at runtime; if it does, it should write to a mounted volume or an explicit `tmpfs`. Setting the container's root filesystem read-only means that even if an attacker gets code execution, they cannot drop a binary into the image, cannot overwrite your application code, cannot persist a backdoor on disk. You configure this in the Kubernetes pod spec rather than the Dockerfile, but it is part of the same hardening posture, so I include it here for completeness:

```yaml
# kubernetes pod securityContext — the runtime half of image hardening
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
  seccompProfile:
    type: RuntimeDefault
```

`runAsNonRoot: true` is an admission gate — Kubernetes will refuse to start a pod whose image runs as root, which means your nonroot Dockerfile and your cluster policy enforce each other. `readOnlyRootFilesystem`, `allowPrivilegeEscalation: false`, and `capabilities: drop ALL` strip the remaining privileges. The Dockerfile sets the user; the pod spec enforces the rest. Together they implement least privilege in depth. The reliability and policy reasoning behind admission gates lives in the [site-reliability-engineering series](/blog/software-development/site-reliability-engineering/production-readiness-reviews); here we just make sure the image is built to pass them.

## 8. ENTRYPOINT, PID 1, and graceful shutdown

A production image has to behave well as a process, not just exist as a filesystem. Two failure modes hide here, and both are about how your process runs as **PID 1** inside the container.

First, **exec form versus shell form**. There are two ways to write `ENTRYPOINT` and `CMD`:

```dockerfile
# shell form — DON'T: wraps your process in /bin/sh -c, breaking signals
ENTRYPOINT node dist/server.js

# exec form — DO: your process is PID 1 directly
ENTRYPOINT ["node", "dist/server.js"]
```

The shell form (`ENTRYPOINT node dist/server.js`) runs your command as `/bin/sh -c "node dist/server.js"`, which means the *shell* is PID 1 and your `node` process is its child. When Kubernetes wants to stop the pod, it sends `SIGTERM` to PID 1 — the shell — and many shells do not forward signals to their children. Your `node` process never hears the `SIGTERM`, never gets to finish in-flight requests or close database connections, and after the grace period Kubernetes sends `SIGKILL` and your process dies abruptly mid-request. The exec form (the JSON-array syntax) makes your process PID 1 directly, so it receives signals and can shut down gracefully. **Always use the exec form for `ENTRYPOINT` and `CMD` in production.** As a bonus, distroless has no shell, so the shell form would not even work — distroless forces you into the correct form.

Second, **zombie reaping**. PID 1 has a special kernel duty: it must "reap" orphaned child processes (call `wait()` on them) or they become zombies that accumulate in the process table. Most application runtimes were not written to be PID 1 and do not reap. If your app spawns child processes (shelling out, running a worker pool), you can leak zombies. The fix is a tiny init process that *is* designed to be PID 1 and reaps for you. The standard one is `tini`:

```dockerfile
FROM node:20-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*
USER node
ENTRYPOINT ["/usr/bin/tini", "--", "node", "dist/server.js"]
```

`tini` becomes PID 1, forwards signals to your app (preserving graceful shutdown), and reaps zombies. Docker has `--init` to inject `tini` automatically, and Kubernetes setups often rely on the runtime providing it, so you do not always need it in the image — but if your app spawns children and you are seeing zombie processes or signals not reaching your app, `tini` is the fix. For a simple single-process service that uses the exec form, you may not need `tini` at all; know the failure mode so you recognize it when it appears.

Graceful shutdown is the payoff of getting PID 1 right. When your process receives `SIGTERM`, it should stop accepting new connections, finish in-flight requests within the grace period, close connections cleanly, and exit. That is what turns a deploy from "drops a handful of requests on every rollout" into "zero-downtime rollout." The Dockerfile's job is to make sure the signal actually reaches your handler; your application code's job is to handle it. The deeper rollout-safety mechanics — readiness probes, `preStop` hooks, connection draining — belong to the deployment, and the [microservices deployment-strategies post](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) covers them; the image's contribution is the exec-form `ENTRYPOINT`.

## 9. The finishing touches: HEALTHCHECK, labels, and not baking secrets

Three smaller production concerns round out a real Dockerfile.

**HEALTHCHECK.** A Dockerfile `HEALTHCHECK` tells Docker how to know whether the container is healthy, and Docker uses it to mark the container `healthy`/`unhealthy`:

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["/app", "healthcheck"]
```

Note the exec form again, and note that on a distroless image you cannot `CMD curl ...` because there is no `curl` — so you either build a tiny healthcheck subcommand into your binary (the `/app healthcheck` pattern above) or rely on Kubernetes probes instead. In Kubernetes, the `livenessProbe` and `readinessProbe` in the pod spec are what actually gate traffic and restarts, and they generally supersede the Dockerfile `HEALTHCHECK`. So for a Kubernetes deployment, the probes matter more than the `HEALTHCHECK` directive — but if you also run the image under plain Docker or Compose, the `HEALTHCHECK` earns its place.

**Labels and OCI annotations.** Stamp the image with metadata so that, six months from now, you can answer "what commit is this, who built it, when?" from the image alone:

```dockerfile
ARG GIT_SHA
ARG BUILD_DATE
LABEL org.opencontainers.image.source="https://github.com/acme/api" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="acme-api"
```

The `org.opencontainers.image.*` keys are the standard OCI annotations that registries and tools understand. This is "everything as code" applied to provenance: the image carries its own lineage, which is the first step toward the signed, attested supply chain the planned scanning sibling covers.

**Do not bake secrets into layers.** This one has burned real companies. Anything you `COPY` or set with `ARG` or `ENV` during the build is recorded in the image's layer history and is recoverable by anyone who pulls the image — even if a later instruction deletes the file or unsets the variable. So this is a leak:

```dockerfile
# NEVER do this — the token lives in the layer history forever
ARG NPM_TOKEN
RUN echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > ~/.npmrc \
 && npm ci \
 && rm ~/.npmrc
```

The `rm` does nothing for security: the `RUN` that wrote `.npmrc` is a layer, and `docker history` plus a layer extraction recovers the token. The correct way to pass a build-time secret is BuildKit's **secret mount**, which makes the secret available to a single `RUN` step and never writes it to a layer:

```dockerfile
# syntax=docker/dockerfile:1
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc npm ci
COPY . .
RUN npm run build
```

```bash
# build, passing the secret without baking it into any layer
DOCKER_BUILDKIT=1 docker build \
  --secret id=npmrc,src="$HOME/.npmrc" \
  -t acme/api:"$GIT_SHA" .
```

The secret is mounted only for that `RUN`, used to authenticate the install, and never persisted in a layer. Runtime secrets are a separate matter entirely — those should be injected at deploy time via the External Secrets Operator, a cloud secrets manager, or Kubernetes Secrets, never built into the image. The principle is the same as build-once-promote-everywhere: the image is environment-agnostic, and the secrets that differ per environment arrive at runtime, not bake time.

## 10. Worked example: the full 1.9 GB to 80 MB transformation

Now let us put every lever together and walk the full transformation step by step, watching the numbers move. This is the headline worked example, and it is the most useful exercise you can do with your own image.

![Timeline showing an image shrinking from 1.9 GB with two hundred eighty CVEs through adding a dockerignore then a slim base then a multi-stage split then a distroless runtime then a nonroot user with a pinned digest down to 80 MB and hardened](/imgs/blogs/writing-a-production-dockerfile-8.png)

The figure above is the path as a timeline; here is each step with its arithmetic.

#### Worked example: shrinking a Node service from 1.9 GB to 80 MB

**Step 0 — the baseline.** The naive Dockerfile from section 1: `FROM node:20`, `COPY . .`, `npm install`. Measured: **1.9 GB, ~280 CVEs, runs as root, ~3 minute rebuild on every commit.** This is where most teams start.

**Step 1 — add a `.dockerignore`.** Exclude `node_modules`, `.git`, `dist`, `.env`, logs. The local `node_modules` (which `COPY . .` was dragging in) stops being copied, and the `.git` history stops being shipped. Measured: **~1.6 GB.** Small size win, but a real security win — the `.git` and any local `.env` are no longer in the image. Effort: one file, five minutes.

**Step 2 — switch the base to `-slim`.** Change `FROM node:20` to `FROM node:20-slim`. The full Debian OS, with its compilers and `apt` and the rest, shrinks to the slim variant. Measured: **~900 MB, CVE count drops to roughly 120.** Effort: one word.

**Step 3 — go multi-stage.** Split into `build`, `deps`, and `runtime` stages as in section 2. The build stage installs `devDependencies` and bundles; the `deps` stage runs `npm ci --omit=dev`; the runtime stage copies only the production `node_modules` and the built `dist`. Now the test framework, the TypeScript compiler, and the build tooling are gone from the shipped image. Measured: **~240 MB, ~40 CVEs.** This is the single biggest jump, which is why multi-stage is the headline lever. Effort: restructure the Dockerfile, maybe twenty minutes.

**Step 4 — distroless runtime.** Change the runtime stage base from `node:20-slim` to `gcr.io/distroless/nodejs20-debian12`. The runtime OS — `bash`, `apt`, coreutils, the package database — disappears; only the Node runtime and essential libraries remain. Measured: **~85 MB, ~3 CVEs.** The CVE count collapses because there is almost nothing left to be vulnerable. Effort: one line, plus learning the ephemeral-debug-container workflow for the day you need to poke inside.

**Step 5 — nonroot and pin the digest.** Use the `:nonroot` distroless variant (UID 65532) and pin the base by `@sha256:` digest. Add the exec-form `ENTRYPOINT`, the OCI labels, and the read-only-filesystem pod spec. Measured: **~80 MB, ~3 CVEs, runs as nonroot, reproducible base, passes the `runAsNonRoot` admission gate.** Effort: a few lines.

The scoreboard:

| Step | Change | Size | CVEs | User |
|---|---|---|---|---|
| 0 | naive single-stage | 1.9 GB | ~280 | root |
| 1 | `.dockerignore` | 1.6 GB | ~280 | root |
| 2 | `-slim` base | 900 MB | ~120 | root |
| 3 | multi-stage split | 240 MB | ~40 | root |
| 4 | distroless runtime | 85 MB | ~3 | root |
| 5 | nonroot + digest pin | 80 MB | ~3 | nonroot |

A note on honesty: the exact numbers vary by language, dependency tree, and scanner, and the figures above are representative of real Node services I have shrunk, not a single benchmarked artifact — treat the *shape* of the curve (the big jump at multi-stage, the CVE collapse at distroless) as the durable lesson, not the specific megabytes. The way to make this real for your own service is to measure it. Run `docker history --no-trunc <image>` to see where the bytes are by layer — it prints each layer's size and the instruction that created it, and the fat layers jump out. Run `docker images` to see total size. Run a scanner (`trivy image <image>` or `grype <image>`) to count CVEs before and after each step. Watch the three numbers move as you apply each lever. That measurement loop is the entire methodology, and it is the proof the kit asks for: a before-and-after you generated yourself, on your own image, that you can defend.

### The artifacts: the bad Dockerfile and the fixed one

For completeness, here is the full production Dockerfile that the transformation lands on — copy and adapt it. First, the bad one, for reference:

```dockerfile
# BAD: single-stage, root, cache-busting, full base, ~1.9 GB
FROM node:20
WORKDIR /app
COPY . .
RUN npm install
EXPOSE 3000
CMD ["node", "server.js"]
```

And the fixed, production-grade version:

```dockerfile
# syntax=docker/dockerfile:1
# --- build stage: full toolchain, dev deps, bundle ---
FROM node:20-bookworm AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# --- deps stage: production deps only (caches independently) ---
FROM node:20-bookworm AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# --- runtime stage: distroless, nonroot, pinned by digest ---
FROM gcr.io/distroless/nodejs20-debian12:nonroot@sha256:6f3d6e3...c4a91b AS runtime
WORKDIR /app
ENV NODE_ENV=production
ARG GIT_SHA
LABEL org.opencontainers.image.source="https://github.com/acme/api" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.title="acme-api"
COPY --from=deps --chown=nonroot:nonroot /app/node_modules ./node_modules
COPY --from=build --chown=nonroot:nonroot /app/dist ./dist
COPY --chown=nonroot:nonroot package.json ./
USER nonroot:nonroot
EXPOSE 3000
ENTRYPOINT ["/nodejs/bin/node", "dist/server.js"]
```

And the `.dockerignore` that goes with it:

```bash
.git
.gitignore
node_modules
npm-debug.log
dist
build
coverage
.nyc_output
.env
.env.*
*.md
Dockerfile
.dockerignore
.github
.vscode
**/*.test.js
**/*.spec.ts
```

Every line in the fixed Dockerfile is one of the levers: the three stages are the multi-stage split; the lockfile-first `COPY` ordering is the cache lever; the distroless base is the minimal-base lever; the `@sha256:` is the digest pin; the `USER nonroot:nonroot` is the drop-root lever; the exec-form `ENTRYPOINT` is the PID-1 lever; the `LABEL`s are provenance. There is nothing exotic here. It is forty lines, and it converts a 2 GB liability into an 80 MB asset.

## 11. War story: the secrets that shipped in the layer

Let me tell you about a class of incident that is more common than it should be, because it makes the "do not bake secrets" rule visceral. The pattern: a team needed a private package-registry token to `npm install` (or `pip install`, or `bundle install`) a private dependency during the build. The obvious move is to pass the token as a build arg and write it to a config file. Someone, reasonably, added a `rm` at the end to "clean up" the token. The image shipped. The token looked gone — `docker run` showed no `.npmrc`. But the token was in the *layer history*. Anyone who could pull the image — and for a public image on Docker Hub, that is everyone on Earth — could run `docker history`, identify the layer, extract it with `docker save` and `tar`, and read the token in plaintext. There are public write-ups of exactly this: long-lived cloud credentials and registry tokens recovered from the published layers of popular images, sometimes years after they shipped, still valid.

The lesson is not "be careful with secrets." It is structural: **a deleted file in an earlier layer is not deleted from the image.** Layers are append-only diffs; a later layer that removes a file just records a whiteout, while the bytes remain recoverable in the earlier layer. The only correct fixes are (a) BuildKit secret mounts, which never write the secret to any layer, as shown in section 9, or (b) multi-stage builds where the secret is used only in an early stage that is discarded and never copied forward. This is, incidentally, another reason multi-stage builds are the gift that keeps giving: a token used in the build stage to fetch private dependencies simply does not exist in the runtime stage, because the runtime stage only received the `COPY --from` artifact, not the build stage's filesystem.

The broader supply-chain dimension connects to incidents the whole industry learned from. The Codecov breach (2021) showed how a compromised build script can exfiltrate every secret present in CI environments — a strong argument for minimizing what secrets ever touch a build and for keeping images free of long-lived credentials. The dependency-confusion class of attacks showed how an image that pulls dependencies at build time inherits the trust of every registry it talks to — a strong argument for pinning, for digest-pinned bases, and for scanning the resulting image. The image-hardening work in this post is one layer of that defense; the signing, SBOM, and scanning layers are covered in the planned sibling post on image security scanning and a minimal attack surface, and the reliability-side reasoning lives in the SRE series' production-readiness material. The point that ties them together: a minimal image is not just smaller, it is *less to trust and less to defend*, and that is a supply-chain win as much as a performance one.

## 12. Stress-testing the production Dockerfile

A practice you cannot break is a practice you do not understand. Let us poke at the production Dockerfile from a few angles.

**What if the build cache is cold?** Then you pay the full cold build — install all dependencies, compile everything. The cache-ordering reorder never makes the cold build slower; it only speeds up warm builds. In CI, the cold-cache case happens on every fresh runner unless you wire up a persistent cache backend (registry cache or the Actions cache), so the realistic posture is: the reorder is necessary, and a CI cache backend is what lets you cash in on it. The worst case is no worse than the naive image; the common case is far better.

**What if a native dependency needs a compiler that distroless does not have?** This is the real edge case with distroless and slim bases. Some packages have native extensions that compile against system libraries at install time. The answer is the multi-stage split doing its job: compile the native module *in the build stage* (which has the full toolchain), producing a prebuilt artifact or wheel, then `COPY --from=build` the compiled output into the distroless runtime. If the compiled module dynamically links a system library that distroless lacks, either statically link it in the build stage or move from `distroless/static` to `distroless/base` (which includes glibc and a few common libraries). The discipline is: the runtime stage gets the *output* of compilation, never the *means* of compilation.

**What if I need to debug the running container and there is no shell?** Use an ephemeral debug container: `kubectl debug -it <pod> --image=busybox:1.36 --target=<container>`. This attaches a throwaway container with a full toolkit into the target pod's namespaces, so you can inspect the process, the network, and shared volumes — without ever putting a shell in the production image. You get debugging power on demand and zero shell in production the rest of the time. (Plain Docker has `docker debug` in newer versions; the principle is the same.)

**What if the base image's digest I pinned has a critical CVE published against it?** Then Renovate or Dependabot opens a PR bumping the digest to the rebuilt, patched base, your CI builds and tests against the new base, and you merge. The digest pin does not trap you on a vulnerable base; it makes the upgrade an explicit, reviewable event rather than silent drift. If the CVE is urgent, you bump the digest by hand and ship. The pin is a control, not a cage.

**What if two stages need the same expensive setup?** Factor it into a shared base stage: `FROM node:20-bookworm AS base` that does the common `WORKDIR`, `COPY package*.json`, and then `FROM base AS build` and `FROM base AS deps` both inherit it. BuildKit builds the shared stage once and reuses it, and you avoid duplicating instructions. This keeps a multi-stage Dockerfile DRY without sacrificing the cache structure.

**What if my image must run on multiple CPU architectures?** Use `docker buildx build --platform linux/amd64,linux/arm64` to produce a multi-arch image (a manifest list). Multi-stage builds work fine with this; just be aware that cross-compilation in the build stage is faster than emulation, so for Go use `GOOS`/`GOARCH` build args and for other languages prefer native runners per architecture if emulation is slow. This is beyond the core four levers but worth knowing exists.

## 13. The four levers as one checklist

Step back and the four levers form a single, memorable checklist for any production image.

![Tree diagram of a production image checklist branching into small from a multi-stage build cache-friendly from lockfile-first ordering hardened from a nonroot user and read-only filesystem and reproducible from pinning the base by digest](/imgs/blogs/writing-a-production-dockerfile-6.png)

The figure above is the whole post in one diagram: a production image is **small** (multi-stage build, minimal base), **cache-friendly** (lockfile-first ordering, `.dockerignore`), **hardened** (nonroot user, read-only filesystem, no shell), and **reproducible** (base pinned by digest, deterministic `npm ci`). Each branch is independent — you can adopt them one at a time — but they compound. Multi-stage gives you small *and* less attack surface. Distroless gives you small *and* fewer CVEs *and* no shell to exploit. Lockfile-first ordering gives you fast builds *and* fewer CI minutes. The reason a handful of techniques get you so far is that none of them is single-purpose; each one moves several of the numbers you care about at once.

It is worth being explicit about how these connect to the series' two principles. **Build once, promote everywhere**: the image you build here is *the* artifact: pinned, reproducible, immutable. It flows unchanged through every environment, which is only safe if it is genuinely reproducible (hence the digest pin and `npm ci`) and environment-agnostic (hence no baked-in secrets — those arrive at runtime). **Everything as code**: the Dockerfile, the `.dockerignore`, the base-image digest, and the dependency versions are all values in version-controlled files, changed by reviewable pull requests, never mutated by hand or by a moving tag. A production Dockerfile is "everything as code" applied to the one artifact your whole pipeline exists to produce.

## 14. How to reach for this (and when not to)

Every practice has a cost, and a field manual that does not tell you when to skip a practice is selling you something. Here is the honest version.

**Always do these — they are free wins with no real downside:**

- **Multi-stage builds.** There is no production image that benefits from shipping its build toolchain. Always split. The only time a single stage is fine is a genuinely trivial image (a static-site `nginx` serving prebuilt files) where there is no build step at all.
- **Lockfile-first cache ordering.** It is strictly dominant — never slower, often dramatically faster. There is no argument against it.
- **A `.dockerignore`.** Five minutes, prevents a category of leaks and bloat. Always.
- **A nonroot `USER` with a numeric UID.** The security payoff per line is the highest in the file. The only exception is the rare workload that genuinely needs a privileged capability (binding port 80 directly without `CAP_NET_BIND_SERVICE`, certain low-level networking) — and even then, prefer granting the specific capability over running as full root.

**Reach for these when the context justifies the cost:**

- **Distroless.** The right default for a production runtime, but it changes how you debug. If your team relies heavily on shelling into containers and has not learned ephemeral debug containers, adopt distroless alongside that workflow change, not before it. For a build stage, keep the full base.
- **Digest pinning + a dependency bot.** Worth it for any image you ship to production, because reproducibility and supply-chain hardening compound over a service's life. For a throwaway spike or a local-only image, a tag is fine.
- **`tini` / init.** Only if your app spawns child processes or you observe zombies or signal-handling problems. A simple single-process service with an exec-form `ENTRYPOINT` usually does not need it.
- **Multi-arch builds.** Only if you actually deploy to multiple architectures (e.g., ARM-based cloud instances *and* x86). Otherwise it just slows your build.

**When NOT to over-engineer this:**

- **Don't chase the last megabyte.** The jump from 1.9 GB to 80 MB is transformative; the jump from 80 MB to 60 MB is usually not worth fighting Alpine's musl gotchas for. Get to distroless and stop, unless you have a measured reason (extreme pull-time sensitivity at huge scale) to go further.
- **Don't move to Alpine for an ecosystem that fights it.** For Python with native wheels, Alpine often costs you more in build flakiness and slowness than it saves in size. A slim Debian base or distroless is usually the better trade. Smaller is not the goal; small-*and*-reliable is.
- **Don't bake your whole monorepo into one giant image.** If a Dockerfile is dragging in services it does not run, the problem is the build context, not the base image. Scope the context (and the `.dockerignore`) to the one service.
- **Don't treat the Dockerfile as the whole security story.** A hardened image still needs scanning, signing, an SBOM, runtime policy, and network controls. The image is necessary, not sufficient. This post hardens the artifact; the planned scanning sibling and the SRE series cover the rest of the chain.

The meta-rule: the four core levers (multi-stage, ordering, minimal base, nonroot) are nearly universal and nearly free — do them by default. The refinements (the exact base on the spectrum, init, multi-arch) are judgment calls you make with a measurement in hand. Measure first, optimize second, and stop when the curve flattens.

## 15. Key takeaways

- **The default Dockerfile is a liability.** `FROM node`, `COPY . .`, `npm install`, run-as-root gives you a ~2 GB image with hundreds of CVEs that rebuilds on every commit. None of that is a bug you wrote — it is the defaults. A production Dockerfile is a deliberate refusal of the defaults.
- **Multi-stage builds are the single biggest lever.** Do the heavy work in a `build` stage with the full toolchain; `COPY --from=build` only the artifact into a tiny `runtime` stage. The shipped image never contained the compilers, so it is smaller *and* far less exploitable. Discarded stages genuinely leave bytes behind; a `RUN rm` does not.
- **Order layers least-changing-first.** Copy the lockfile and install dependencies *before* copying source, so a code-only change reuses the cached install layer. The cache is a prefix match — it holds until the first changed input, then everything below rebuilds. This is what turns a 3-minute build into an 8-second one.
- **A `.dockerignore` is five minutes that prevents leaks and bloat.** Exclude `node_modules`, `.git`, `.env`, and build outputs from the build context — faster builds, no accidental secret in a layer, no stale local build copied in.
- **Pick the smallest base you can debug.** The spectrum runs full → slim → alpine (musl gotchas) → distroless (no shell, tiny CVE count) → scratch (static binaries). Smaller means faster pulls, fewer CVEs to patch, and less for an attacker to use. Distroless is the right default runtime; debug it with ephemeral containers, not a baked-in shell.
- **Drop root.** Run as a numeric nonroot UID with a read-only root filesystem and dropped capabilities. Container root maps to host root on escape; least privilege keeps a foothold shallow. Pair `USER 10001` with a `runAsNonRoot` admission gate so the image and the cluster enforce each other.
- **Pin the base by digest.** A tag is a moving target; `@sha256:` is immutable, which makes builds reproducible and blocks a poisoned-registry attack. Let Renovate or Dependabot bump the digest by PR so you still get patches — reproducibility *and* a controlled cadence.
- **Use the exec-form `ENTRYPOINT`.** The JSON-array form makes your process PID 1 so it receives `SIGTERM` and shuts down gracefully; the shell form hides signals behind a shell and gets your process `SIGKILL`-ed mid-request. Add `tini` only if you spawn children.
- **Never bake secrets into a layer.** A deleted file lives on in the layer history forever. Use BuildKit secret mounts (`--mount=type=secret`) or a discarded build stage, and inject runtime secrets at deploy time.
- **Measure, don't guess.** `docker history` for where the bytes are, `docker images` for total size, `trivy`/`grype` for CVE count — before and after each lever. The 1.9 GB → 80 MB story is real only when you can reproduce it on your own image.

## Further reading

- [From Commit to Production: The CI/CD Mental Model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series map: the commit → build → test → package → deploy → operate spine, build-once-promote-everywhere, and the four DORA metrics this Dockerfile work serves.
- [Build Once, Promote Everywhere: Artifacts and Versioning](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) — why the image you build is the immutable artifact that flows through every environment, and why reproducibility (the digest pin) is what makes that safe.
- [The Build Stage: Reproducible, Fast, and Cacheable](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) — the deeper treatment of build reproducibility and the three levels of caching that the layer-cache section here builds on.
- Containers from first principles for delivery (planned sibling, slug `containers-from-first-principles-for-delivery`) — namespaces, cgroups, layers, and the OCI image format underneath everything in this post.
- Building images fast and securely in CI (planned sibling, slug `building-images-fast-and-securely-in-ci`) — wiring `docker buildx` and persistent layer-cache backends so the cache-ordering wins survive ephemeral CI runners.
- Image security scanning and a minimal attack surface (planned sibling, slug `image-security-scanning-and-a-minimal-attack-surface`) — Trivy/Grype scanning, SBOMs with Syft, cosign signing, and SLSA provenance on top of the minimal image this post produces.
- [Deployment strategies: blue-green, canary, feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) — how the graceful-shutdown image behaves during a rollout, and the connection-draining mechanics the exec-form `ENTRYPOINT` enables.
- [Production readiness reviews](/blog/software-development/site-reliability-engineering/production-readiness-reviews) — the reliability and policy-gate reasoning behind `runAsNonRoot`, read-only filesystems, and the admission controls a hardened image must pass.
- Docker's official multi-stage build and BuildKit documentation, Google's distroless project, and the OCI image-spec annotations reference — the canonical sources for the directives used throughout this post.

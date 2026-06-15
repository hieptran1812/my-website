---
title: "Containerizing Microservices: Docker Best Practices for Images You Won't Be Ashamed Of"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How to turn the bloated, root-running, CVE-riddled 1.2GB image most teams ship first into a 22MB multi-stage distroless image — and why your container is a security and cost surface, not just a package."
tags:
  [
    "microservices",
    "docker",
    "containers",
    "multi-stage-builds",
    "distroless",
    "container-security",
    "devops",
    "distributed-systems",
    "software-architecture",
    "backend",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/containerizing-microservices-docker-best-practices-1.webp"
---

The first Docker image I ever shipped to production was 1.4 gigabytes. It ran as root. It contained a full Ubuntu userland, a C compiler, the Node.js build toolchain, `git`, `curl`, `vim` (I genuinely do not know why), and an attack surface roughly the size of a small country. It also worked, which was the dangerous part — it passed CI, it ran in staging, it served traffic in production, and for about four months nobody noticed anything wrong. Then a security scanner got pointed at our registry as part of a compliance push, and the image came back with 312 known vulnerabilities, eleven of them rated critical. The same week, our cloud bill flagged the container registry as a surprise line item: we were storing forty-plus services, each a gigabyte-plus, each pulled fresh onto dozens of nodes on every deploy, and the egress and storage were adding up to real money. The image had been "working" the whole time. It was also a liability sitting quietly in the middle of our infrastructure, waiting.

That image is the villain of this post, and its redemption arc is the spine of everything we'll cover. We are going to take the order service from "ShopFast," a fictional e-commerce company whose engineers keep appearing throughout this series, and follow its Dockerfile through the exact evolution most teams eventually make — usually the hard way, after an incident. We start with the naive single-stage root image that weighs 1.2GB and carries a fistful of critical CVEs. We end with a 22MB multi-stage image built on a distroless base, running as a non-root user with a read-only root filesystem and every Linux capability dropped. Same service, same behavior, same binary doing the same work. A 55× smaller artifact, a tiny fraction of the vulnerabilities, and a deploy that is faster, cheaper, and far harder to exploit. The figure below shows the two ends of that journey side by side; the rest of the post is how you get from the left column to the right one.

![A before and after comparison of a bloated single stage root image versus a slim multi stage distroless non root image](/imgs/blogs/containerizing-microservices-docker-best-practices-1.webp)

By the end of this post you will be able to write a production-grade Dockerfile from scratch: a multi-stage build that compiles in a fat builder and ships only the binary, ordered so the layer cache turns a four-minute CI build into a twenty-five-second one, on a minimal base chosen deliberately rather than by copy-paste, running as a non-root user under a read-only filesystem with dropped capabilities, scanned for vulnerabilities in CI, and pinned by digest for reproducibility. More importantly, you will understand *why* each of those choices matters in a microservices fleet specifically — because when you have forty services instead of one, every one of these decisions multiplies by forty, and the difference between a sloppy image and a careful one stops being a code-review nitpick and becomes a meaningful line on your cloud bill and a meaningful entry in your threat model.

This post opens Track 6 of the series, the deployment and infrastructure track. It assumes you have already internalized what a single service looks like — if not, the [anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice) is the place to start, and [what microservices are and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them) is the honest framing of the whole tradeoff. The container is the unit you'll be deploying; everything that comes after — [Kubernetes](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials), [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability), [deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags), and [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management) — assumes a good image as its starting point. Get the image right first.

## Why containers and microservices are a natural fit

Before we touch a Dockerfile, it's worth being precise about *why* containers won, because the reasons aren't cosmetic and understanding them tells you which Docker practices actually matter versus which are cargo-culted. Microservices and containers grew up together, and they reinforce each other, but they are not the same idea. Microservices is an architectural decision about how to decompose a system. Containers are a packaging and isolation technology. The reason nearly every microservices team converges on containers is that microservices create three specific problems and containers solve all three cleanly.

The first problem is **the consistent artifact**. When you have one monolith, "how do we package and run this" is a problem you solve once. When you have forty services written by six teams in three languages — a Go order service, a Java payment service, a Python recommendation service, a Node.js notification service — "how do we package and run this" becomes forty problems, and if each team solves it differently you have built an operations nightmare. A container image solves this by being a single, uniform packaging format regardless of language: the operations team learns *one* thing (how to run a container), and every service, whatever it's written in, ships as the same kind of object. The platform — Kubernetes, your CI/CD system, your registry — only ever sees images. The polyglot reality of a microservices fleet is exactly what makes a uniform artifact format so valuable.

The second problem is **dev/prod parity**. The oldest bug report in software is "it works on my machine." With microservices the surface for that bug explodes, because a request might touch eight services and each one has its own runtime, its own dependencies, its own version of `libssl`. A container freezes the entire userland — the base OS libraries, the language runtime, the dependencies, the binary — into one immutable image. The image that passed your tests is byte-for-byte the image that runs in production. The build flow in the figure later in this post makes that explicit: the same digest flows from CI through staging to production, and "works on my machine" becomes "works in the image, and the image is the same everywhere." This is one of the principles in the [twelve-factor app](https://12factor.net/) methodology — keep development, staging, and production as similar as possible — and the container is the mechanism that makes it nearly free.

The third problem is **immutable, independent deploys**. The whole promise of microservices is that teams can deploy independently — the payment team ships at 2pm without coordinating with the order team. For that to be safe, a deploy has to be atomic and reversible: you replace running instances with new ones, and if something's wrong you replace them back. Mutable infrastructure (SSH in, `apt upgrade`, restart) makes that nearly impossible to do reliably across a fleet, because every server drifts into its own slightly-unique state. An immutable image makes it trivial: deploy means "schedule this new image," rollback means "schedule the old image's digest." Nothing is ever patched in place. The artifact you tested is the artifact that runs, and the previous artifact is one digest away if you need it back.

So containers didn't win because they were trendy. They won because microservices manufacture exactly the problems — heterogeneity, environment drift, risky in-place deploys — that containers were built to eliminate.

### Containers versus virtual machines, precisely

Juniors often arrive with a fuzzy sense that "a container is a lightweight VM." It isn't, and the difference is the whole reason containers are fast and dense enough to make a microservices fleet economical. A virtual machine virtualizes *hardware*: the hypervisor presents virtual CPUs, virtual disks, and virtual network cards to a complete *guest operating system*, kernel and all. Every VM boots its own kernel, runs its own init system, and reserves its own slice of memory for the OS before your application gets a single byte. That's why a VM takes tens of seconds to boot and why you can pack only a handful onto a host.

A container virtualizes the *operating system*, not the hardware. There is no guest kernel. A container is just a normal Linux process running directly on the *host* kernel, with two kernel features wrapped around it: **namespaces**, which give the process its own isolated view of the filesystem, network, process tree, and user IDs (so it thinks it's alone on the machine), and **cgroups** (control groups), which limit and account for the CPU, memory, and I/O the process can use. That's the entire trick. A container is a process the kernel has been told to lie to about what else exists, and to throttle if it gets greedy. There's no OS to boot, so it starts in milliseconds; there's no per-container kernel hogging memory, so you can pack dozens onto a node. The figure below shows the layering: your app process, wrapped in namespaces and cgroups, running on a container runtime, sitting directly on the one shared host kernel.

![A layered stack showing an app process wrapped in namespaces and cgroups on a container runtime sharing one host kernel](/imgs/blogs/containerizing-microservices-docker-best-practices-2.webp)

The practical consequences of "shared kernel, isolated process" are exactly the properties a microservices fleet needs. **Fast start**: a container is ready in tens of milliseconds, which is what makes autoscaling and rapid rescheduling possible — Kubernetes can kill and restart your pod or scale from ten to fifty replicas in seconds, which it could never do if each instance booted a kernel. **Density**: because there's no per-instance OS overhead, you can run thirty or forty containers on a node that might hold three or four VMs, and density translates directly into cost. **Lightweight isolation**: each container gets its own filesystem, network namespace, and resource limits, so the Go order service and the Java payment service can't see each other's files or steal each other's memory, even on the same host.

The one honest caveat — and a senior must know it — is that the shared kernel is also the security boundary, and it's a *softer* boundary than a VM's hardware virtualization. A kernel exploit (a container escape via a bug in the shared kernel) can, in principle, break out of the namespace isolation in a way that a VM escape never could, because there's no second kernel in the way. This is precisely why the hardening we'll do later — non-root, dropped capabilities, read-only root filesystem, minimal base — matters so much. You are reducing what a process can do *if* it gets compromised, because the wall between it and its neighbors is thinner than people assume. The lightness that makes containers economical is the same lightness that makes hardening non-optional.

## Docker fundamentals: just enough to build well

You can build excellent images while treating most of Docker as a black box, but four concepts are load-bearing and you cannot skip them: the image as a stack of layers, the registry, the difference between a tag and a digest, and the build cache.

An **image is a stack of read-only layers**. Each instruction in a Dockerfile that changes the filesystem — `FROM`, `COPY`, `RUN`, `ADD` — produces a new layer, which is a tarball of the filesystem *changes* that instruction made on top of the previous layer. The final image is those layers stacked, presented to the running container as a single merged filesystem via a union filesystem (overlayfs). When the container writes to a file, it gets a private copy-on-write layer on top; the underlying image layers stay immutable and shared. This layering is not a curiosity — it is the single most important fact for building good images, because it drives both the build cache and how efficiently images are stored and pulled. Two images built from the same base share that base layer on disk and over the wire; you pull it once.

A **registry** is where images live and are distributed. Docker Hub, Amazon ECR, Google Artifact Registry, GitHub Container Registry — they all do the same job: store image layers, indexed by repository name and reference, and serve them on pull. When you `docker push`, you upload the layers your registry doesn't already have; when a node does `docker pull`, it downloads only the layers it's missing. This is why layer sharing matters at fleet scale: if forty services share a 2MB distroless base layer, a node caches that layer once and reuses it for all forty, instead of pulling forty copies of a 120MB Ubuntu base.

A **tag versus a digest** is a distinction juniors routinely get wrong, and getting it wrong is how you ship non-reproducible builds and mysterious "it changed by itself" incidents. A **tag** like `golang:1.22` or `myservice:latest` is a *mutable, human-friendly pointer*. The owner of that repository can re-point `golang:1.22` at a new image whenever they publish a patch, and they do, constantly. So `golang:1.22` today and `golang:1.22` next week can be two different images. A **digest** like `golang@sha256:a1b2c3...` is the *immutable, content-addressed identity* of one specific image — it's the SHA-256 hash of the image's manifest, so it can never refer to anything else. If you build against a tag, your build is not reproducible: the same Dockerfile produces different images on different days. If you build against a digest, it's bit-for-bit reproducible forever. We'll pin digests later; for now, just hold the distinction: tags are convenient, digests are correct.

Finally, the **build cache**. When Docker executes a Dockerfile, it caches the result of each instruction. On the next build, for each instruction, it checks whether the instruction *and all its inputs* are unchanged from a cached layer; if so, it reuses the cached layer instead of re-running the instruction. The catch — and the entire art of layer ordering — is that **the cache is invalidated for an instruction and every instruction after it the moment any input changes.** Change a file that an early `COPY` touches and every layer below it rebuilds, even if those later instructions are identical. This is why the *order* of your Dockerfile instructions, which feels cosmetic, is actually the difference between a 25-second rebuild and a 4-minute one. We'll exploit this deliberately in the layer-ordering section.

It's worth making the layer-sharing economics concrete, because they explain several best practices at once. Suppose your fleet of forty services all build their runtime stage `FROM gcr.io/distroless/static-debian12:nonroot`. That base is one set of layers with one content hash. A node that has already pulled it for the order service does *not* re-pull it for the payment service or the inventory service — it recognizes the identical layer by its digest and reuses the bytes already on disk. So forty services share a few megabytes of base instead of each carrying its own copy. The same logic runs in reverse on the registry side: pushing a new version of a service only uploads the layers that *changed* (typically just the thin layer with your new binary), not the whole image. This is why a disciplined fleet that standardizes on a small set of shared base images pulls and stores dramatically less than a fleet where every service picks its own random full-OS base — layer sharing only helps when the layers are actually shared. Standardizing the base image across services isn't just tidiness; it's a direct lever on storage, pull time, and patch velocity.

## The naive Dockerfile: an anti-pattern, annotated

Let's meet the villain in full. Here is the kind of Dockerfile that ShopFast's order service shipped with for its first four months in production. It's a Go service, but the anti-patterns are language-agnostic — you've seen the Python and Node equivalents. Every line that's a problem is annotated.

```dockerfile
# THE ANTI-PATTERN — do not ship this
FROM golang:1.22                    # full Debian + Go toolchain = ~900MB base, runs as root
WORKDIR /app
COPY . .                            # copies EVERYTHING incl .git, tests, secrets, node_modules
RUN apt-get update && \
    apt-get install -y curl git vim # "might be handy" — pure attack surface, no cleanup
RUN go build -o /app/orderservice . # compiler + all build artifacts stay in the final image
EXPOSE 8080
CMD ["/app/orderservice"]           # runs as root (uid 0), full shell, full toolchain present
```

Build that and you get a 1.2GB image. Let's count the sins, because each one maps to a fix later. **`FROM golang:1.22`** ships the entire Go build toolchain — the compiler, the standard library sources, the linker, plus a full Debian userland — into the *runtime* image, even though none of it is needed once the binary is compiled. That's hundreds of megabytes of pure dead weight that also happens to be hundreds of megabytes of CVE surface. **`COPY . .`** with no `.dockerignore` copies your entire working directory into the image: the `.git` history, your test fixtures, your local `.env` file if you have one, build caches, everything. This bloats the image *and* is a genuine secret-leak risk — plenty of credentials have shipped to public registries inside a `.git` directory. **`apt-get install curl git vim`** installs tools "in case we need to debug," and without `rm -rf /var/lib/apt/lists/*` the package index stays in the layer too; every one of those tools is a capability an attacker gets for free if they land a shell. **The binary is built in the same image it runs in**, so the compiler and all intermediate build artifacts ride along forever. And **it runs as root**, the default, which means a compromise of the process is a compromise with root privileges inside the container — and given how soft the kernel boundary is, root-in-container is uncomfortably close to root-on-host.

The maddening thing, again, is that this image *works*. It serves traffic correctly. Nothing about its behavior tells you it's a problem. That's exactly why bad images persist: the cost is invisible until a scanner, a bill, or a breach makes it visible. The senior framing is to treat the image as a security and cost surface from day one, not as an inert package — and the rest of this post is the set of moves that shrink both surfaces.

## Multi-stage builds: the dramatic size win

The single highest-leverage change you can make — the one that takes the image from 1.2GB toward 20-something MB in one move — is the **multi-stage build**. The insight is simple once you see it: the tools you need to *build* a service (compiler, build dependencies, package manager) are almost entirely different from the tools you need to *run* it (just the binary and whatever it links against at runtime). A single-stage build conflates them and ships the build tools to production. A multi-stage build separates them: you compile in a fat "builder" stage that has the full toolchain, then you copy *only the compiled artifact* into a clean, tiny "runtime" stage that has none of it. The builder stage is discarded; it never gets shipped.

Here's the same ShopFast order service, built the right way. This is the target Dockerfile — multi-stage, distroless base, non-root — and it's worth reading slowly because almost every best practice in this post is in it.

```dockerfile
# ---- Build stage: full toolchain, thrown away after ----
FROM golang:1.22@sha256:<pinned-digest> AS build
WORKDIR /src

# Copy only the dependency manifests first (cache-friendly ordering)
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# Now copy source and build a static binary
COPY . .
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /out/orderservice ./cmd/orderservice

# ---- Runtime stage: distroless, non-root, binary only ----
FROM gcr.io/distroless/static-debian12:nonroot@sha256:<pinned-digest>
WORKDIR /app
COPY --from=build /out/orderservice /app/orderservice
USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["/app/orderservice"]
```

Walk through what changed. The `FROM golang:1.22 ... AS build` line names the first stage `build`; everything in it — the toolchain, the source, the module cache, the intermediate object files — lives only in that stage. The second `FROM` starts a *fresh* stage on the distroless base. The crucial line is `COPY --from=build /out/orderservice /app/orderservice`: it reaches into the build stage and copies *only* the compiled binary across the stage boundary. Nothing else makes it over — not the compiler, not the source, not the module cache. When Docker assembles the final image, only the last stage is kept. The builder, all 900MB-plus of it, is discarded.

The `CGO_ENABLED=0` flag tells Go to build a fully static binary with no dynamic links to system C libraries, which is what lets us run on `distroless/static` (and would let us run on `scratch`). The `-ldflags="-s -w"` strips the debug symbol table and DWARF debugging info from the binary, shaving several more megabytes off — fine for production, where you'd debug with a separate symbol-laden build or a core dump analyzed elsewhere. The figure below shows the flow: a heavy builder stage produces a small static binary, which is copied into a distroless runtime stage to become the final image, while the builder is thrown away.

![A graph showing source feeding a heavy builder stage that produces a small binary copied into a distroless runtime stage while the builder is discarded](/imgs/blogs/containerizing-microservices-docker-best-practices-3.webp)

#### Worked example: the size win and what it costs you at 40 pods

Let's put real numbers on the win, because the abstraction "smaller is better" doesn't land until you cost it out at fleet scale. The single-stage image is 1.2GB. The multi-stage distroless image is 22MB. That's a 55× reduction. Now multiply by a real microservices fleet: ShopFast runs 40 distinct services, and the order service runs at 40 replicas (pods) across the cluster for its peak load.

Consider a fresh node coming online — a common event under autoscaling, during a node pool upgrade, or when the cluster reschedules after a node failure. That node has to pull every image scheduled onto it. With the fat image at 1.2GB and, say, 10 pods landing on the node across various services, that's up to 12GB to pull before anything can start (layer sharing helps if they share bases, but the fat single-stage images barely shared anything). On a node with a 1 Gbps link doing maybe 100MB/s of effective registry throughput, 12GB is roughly **two minutes of pure image pull** before the first container even starts. With the slim images at 22MB each, that's 220MB — about **two seconds**. Two minutes versus two seconds of cold-start delay per node, every scale-up, every node replacement, every spot-instance reclamation.

The storage and transfer cost compounds it, and this is exactly the kind of fleet-scale arithmetic that [back-of-the-envelope estimation](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) is for — a quick multiply-it-out before you commit. Forty services, each with a 1.2GB image and, say, 20 historical tags retained in the registry for rollback, is 40 × 1.2GB × 20 = 960GB of registry storage. At roughly \$0.10 per GB-month for registry storage that's about \$96/month just in stored bytes — modest, but it's pure waste. The bigger cost is data transfer: every deploy and every scale event pulls images, and cross-AZ or internet egress is billed. If your fleet pulls, conservatively, 5TB of image data per month moving the fat images around and a meaningful slice of that crosses a billed network boundary at \$0.08-\$0.12 per GB, you're looking at several hundred dollars a month evaporating into moving bytes you didn't need. Shrink the images 55× and that line item nearly disappears. None of this changes what the service *does*. It just stops being expensive and slow for no reason.

## Minimal base images: distroless, alpine, scratch

The multi-stage build gets you a tiny *runtime stage*, but the runtime stage still sits on *some* base image, and the choice of base is the next big lever — for both size and, more importantly, security. The base image determines two things: how many bytes of OS userland you ship, and how many of those bytes contain known vulnerabilities and exploitable tools. There are four meaningful choices, in descending order of size and attack surface.

A **full OS base** (`debian:12`, `ubuntu:22.04`) gives you a complete Linux userland: a shell, a package manager, `coreutils`, dozens of shared libraries, the works. It's about 80-120MB and it's *comfortable* — you can `docker exec` in and get a bash shell, run `apt install` to debug, use familiar tools. That comfort is exactly the problem: everything that makes it easy for *you* to poke around makes it easy for an *attacker* who lands a foothold. A full Debian base routinely carries dozens to over a hundred known CVEs in its packages at any given time, most of which your service never even uses.

An **alpine base** (`alpine:3.20`) is the popular "small" choice at around 5-8MB. It's tiny because it uses `musl` libc instead of `glibc` and `busybox` instead of full `coreutils`. It still gives you a shell (`ash`) and a package manager (`apk`), so it's debuggable. The catch a senior must know: `musl` is *not* `glibc`, and the difference occasionally bites — DNS resolution behaves differently, some compiled wheels and native libraries that assume `glibc` break or perform worse, and there have been real production incidents traced to musl's DNS and threading behavior. Alpine is a fine choice, but "alpine because it's small" without understanding the musl tradeoff is how teams get a 3am surprise.

A **distroless base** (Google's `gcr.io/distroless/*`) is the sweet spot for most production services and what I reach for by default. "Distroless" means it contains your application's runtime dependencies and *nothing else* — no shell, no package manager, no `coreutils`, no `busybox`. The `static` variant is a couple of megabytes; the ones bundling a language runtime (Java, Python, Node) are larger but still far slimmer than a full OS. It uses `glibc`, so you don't get the musl surprises. Because there's no shell and no package manager, an attacker who achieves code execution can't spawn a shell, can't download tools, can't `apt install` a rootkit — the standard post-exploitation playbook simply doesn't have the pieces it needs. The cost is debuggability: you can't `docker exec -it ... sh` because there's no `sh`. You debug distroless containers from the outside, with ephemeral debug containers (`kubectl debug`) that attach a tools-laden container sharing the target's namespaces.

A **scratch base** is the empty image — literally nothing, zero bytes. For a fully static binary (a Go binary built with `CGO_ENABLED=0`, or a Rust binary built against musl), you can `FROM scratch` and your image is *just your binary* plus whatever you explicitly copy in (typically TLS root certificates and timezone data). This is the absolute minimum: maximum density, minimum attack surface, nothing to exploit because there's nothing there. The cost is that it's all-or-nothing — you must handle CA certs, timezone data, and `/etc/passwd` for the non-root user yourself, and there is *zero* debuggability. The decision matrix below lays out the four bases across the dimensions that actually matter.

![A decision matrix comparing full OS alpine distroless and scratch base images across size attack surface debuggability and compatibility](/imgs/blogs/containerizing-microservices-docker-best-practices-4.webp)

Read the matrix as a recommendation engine. **Distroless** wins for most compiled and runtime-based services: tiny, glibc-compatible, minimal attack surface, and the lost shell access is a debugging inconvenience you solve with ephemeral debug containers rather than a real blocker. **Scratch** wins specifically for fully static binaries where you want the absolute floor and your team is disciplined about debugging from outside — it's beautiful for a Go gateway or a Rust proxy. **Alpine** wins when you genuinely need an in-container shell *and* your dependencies are musl-clean (a lot of interpreted services with native extensions are not) — and when you're willing to test for the musl gotchas. **Full OS** wins only when you have a hard dependency that demands a full glibc userland and a pile of system packages, and even then you should be reaching for distroless first and asking hard questions before settling for Debian. The default, when you're not sure, is distroless.

#### Worked example: CVE count, full OS versus distroless

Numbers make the security argument concrete. Take the *same* ShopFast order service binary and put it on three different bases, then scan each. On a `debian:12` base, a typical scan turns up on the order of 100-180 known CVEs — almost none in your code, virtually all in the base OS packages you're not even using (a vulnerable `libxml2` you never call, a CVE in a coreutils tool you'll never run). On `alpine:3.20`, the count drops dramatically, often to a single digit to low double digits, because there's just so much less there. On `gcr.io/distroless/static`, it's frequently **zero to three**, and the ones that appear are usually in glibc or the TLS cert bundle — things you actually depend on and that get patched at the base.

Now connect that to operational reality. Your security team runs a scanner against the registry and files a ticket for every HIGH and CRITICAL CVE. With 40 services on a Debian base, that's potentially *thousands* of CVE tickets across the fleet, the overwhelming majority of which are noise — vulnerabilities in packages your services never invoke. Your engineers spend their week triaging false-positive security tickets instead of building features, and the real vulnerabilities drown in the noise. With 40 services on distroless, the scanner returns a near-empty result, and the handful of real findings are *visible* because they're not buried under a hundred irrelevant ones. The smaller base isn't just smaller — it's the difference between a security process that works and one that everyone learns to ignore. Minimizing the base is the cheapest, highest-leverage security control you have, and it costs you essentially nothing but a slightly harder debug story.

## Layer ordering and caching: fast rebuilds

Now the build-speed lever, which is pure economics for your CI pipeline and your developers' patience. Recall the cache rule: Docker caches each instruction's result, and changing any instruction's input invalidates that instruction *and every instruction after it*. The art is to order your Dockerfile so the things that change *rarely* come early (and stay cached) and the things that change *often* come late (so only they and their successors rebuild).

The canonical mistake is `COPY . .` early, before installing dependencies — which is what the anti-pattern did. Because your source code changes on *every single commit*, `COPY . .` invalidates the cache on every commit, which means the expensive dependency-installation step that comes after it *also* re-runs on every commit, even though `go.mod` (or `package.json`, or `requirements.txt`, or `pom.xml`) didn't change. You pay the full dependency download and install on every build for no reason.

The fix is to split the copy: first copy *only the dependency manifest* and install dependencies, *then* copy the source. Look again at the target Dockerfile — `COPY go.mod go.sum ./` then `go mod download`, and only after that `COPY . .` and `go build`. Now the dependency-download layer's only input is `go.mod`/`go.sum`, which change a few times a month, not a few times an hour. The figure below shows the ordering as a build sequence, with the slow dependency step protected behind the rarely-changing manifest copy.

![A timeline of a cache friendly build ordering with the base and dependency manifest copied first the slow install cached and source copied last](/imgs/blogs/containerizing-microservices-docker-best-practices-6.webp)

The same pattern applies in every language, and it's worth seeing them side by side because the principle transfers exactly. Here's the cache-friendly ordering for three common stacks:

```dockerfile
# --- Node.js ---
COPY package.json package-lock.json ./
RUN npm ci                      # cached unless lockfile changes
COPY . .
RUN npm run build

# --- Python ---
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt   # cached unless reqs change
COPY . .

# --- Java (Maven) ---
COPY pom.xml ./
RUN mvn dependency:go-offline   # cached unless pom changes
COPY src ./src
RUN mvn package -o
```

The other technique in the target Dockerfile is the **cache mount**: `RUN --mount=type=cache,target=/go/pkg/mod go mod download`. This (a BuildKit feature) gives the build step a persistent cache directory that survives *across builds* without becoming part of any image layer. Even when the dependency layer *is* invalidated, the actual download is fast because the module cache is still warm on the build host. It's the difference between "re-resolve the dependency graph from the index" and "verify the already-downloaded modules." Cache mounts are the single best speedup for dependency-heavy builds, and they keep the cached artifacts *out* of your image, so they don't bloat it.

#### Worked example: cutting a 4-minute CI build to 25 seconds

Here's the build-time math for ShopFast's order service. With the naive ordering — `COPY . .` before dependency install — every commit rebuilds from the source copy down: re-download all Go modules (about 90 seconds for this service's dependency tree), recompile from scratch (about 120 seconds), package the image (about 30 seconds). Roughly **four minutes** per build, on *every* push, even a one-line README change.

With cache-friendly ordering plus cache mounts, a typical commit only changes source code, not `go.mod`. So the base layer is a cache hit (instant), the `COPY go.mod go.sum` layer is a cache hit (instant), the `go mod download` layer is a cache hit (instant — `go.mod` didn't change), and only the `COPY . .` and `go build` layers re-run. With the build cache mount warm, the recompile only rebuilds the packages that actually changed, so the build drops to about 20 seconds plus a few seconds to package. Roughly **25 seconds**. That's a 10× speedup on the common case.

Across a team pushing 50 commits a day to this one service, that's 50 × (240 − 25) = about 10,750 seconds — roughly **three hours of CI compute saved per day, on one service.** Multiply across 40 services and you're saving real money on CI runners and, more valuably, giving every engineer a 25-second feedback loop instead of a four-minute coffee break. The slow build was never about the work the build *had* to do — it was about the cache invalidating layers that didn't need to rebuild. Fix the ordering and the work disappears.

## Running as non-root: the cheapest security upgrade

By default, a container's process runs as **root** (UID 0) inside the container. Juniors often assume "inside the container" makes root harmless, but it does not — container root *is* root on the host kernel, mapped through the user namespace, and several classes of container-escape and privilege-escalation attacks are only possible *because* the process is root. Running as a non-root user is the single cheapest, highest-impact hardening you can apply, and most teams skip it purely out of inertia.

There are two ways to drop root. The clean way is to bake it into the image with a `USER` directive, as the target Dockerfile does with `USER nonroot:nonroot` (the distroless `:nonroot` variant ships a UID 65532 user precisely for this). For a `scratch` or custom base where no non-root user exists, you create one explicitly:

```dockerfile
# In the build stage, create a passwd entry for a non-root user
RUN echo "appuser:x:65532:65532::/nonexistent:/sbin/nologin" > /etc/passwd_minimal

FROM scratch
COPY --from=build /etc/passwd_minimal /etc/passwd
COPY --from=build /out/orderservice /app/orderservice
USER 65532:65532
ENTRYPOINT ["/app/orderservice"]
```

But baking `USER` into the image is necessary, not sufficient — a privileged deployment can still override it, and defense in depth means enforcing it at the platform too. The second layer is the runtime security context, enforced by Kubernetes when it schedules the image (which is, again, why the image is what the orchestrator schedules — see the [Kubernetes essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials) post for the full picture). Here's the locked-down pod spec for the order service: non-root enforced, root filesystem read-only, *all* Linux capabilities dropped, and privilege escalation forbidden.

```yaml
# Kubernetes Deployment snippet — runtime hardening
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true          # refuse to run if image would run as root
        runAsUser: 65532
        fsGroup: 65532
        seccompProfile:
          type: RuntimeDefault      # restrict syscalls to a sane allowlist
      containers:
        - name: orderservice
          image: registry.example.com/orderservice@sha256:<digest>
          securityContext:
            allowPrivilegeEscalation: false   # no setuid/setgid escalation
            readOnlyRootFilesystem: true      # filesystem is immutable at runtime
            capabilities:
              drop: ["ALL"]                   # surrender every Linux capability
          volumeMounts:
            - name: tmp
              mountPath: /tmp       # mount a writable tmpfs for scratch space
      volumes:
        - name: tmp
          emptyDir: {}
```

The same hardening at `docker run` time, for local testing or a non-Kubernetes host, looks like this:

```bash
docker run \
  --user 65532:65532 \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --tmpfs /tmp \
  registry.example.com/orderservice@sha256:<digest>
```

Each flag closes a specific door. **`--read-only`** makes the container's root filesystem immutable — an attacker who gets code execution can't write a backdoor binary, can't modify your application files, can't drop a cron job. (You give back a writable `/tmp` as a tmpfs for the legitimate scratch-space needs.) **`--cap-drop ALL`** surrenders every Linux capability: the process can't bind privileged ports, can't load kernel modules, can't change file ownership, can't do raw network operations. A typical web service needs *zero* capabilities, so dropping all of them costs nothing and removes a huge slice of what a compromise could do. **`--security-opt no-new-privileges`** prevents the process from ever gaining more privileges than it started with, neutering setuid-binary escalation. The figure below stacks these runtime constraints, each one removing a way a compromised process could move.

![A stack of runtime hardening layers showing non root user read only root filesystem dropped capabilities no new privileges and a minimal distroless base](/imgs/blogs/containerizing-microservices-docker-best-practices-7.webp)

The payoff is a brutal reduction in blast radius. Picture the attacker who finds a remote code execution bug in your service. On the naive image, they're root, the filesystem is writable, they have a full shell, a package manager to pull tools, and every capability available — they can establish persistence, pivot, and probe for a kernel escape at leisure. On the hardened image, they're an unprivileged user (65532) with no shell to spawn, a read-only filesystem they can't write to, no capabilities to abuse, no ability to escalate, and a distroless base with no tools to download anything. They've got code execution in a box with smooth walls and no exits. The RCE is still a serious problem — but it's a contained one, and containment is the entire game in defense.

## The `.dockerignore` you should always have

A small file with an outsized effect. The `.dockerignore` controls what gets sent to the Docker daemon as "build context" and therefore what `COPY . .` can possibly include. Without it, `COPY . .` rakes in your `.git` directory (history, possibly committed secrets), `node_modules` (which you're about to rebuild anyway), test fixtures, local environment files, build artifacts, and editor cruft — bloating the image and creating real secret-leak risk. Here's a sane default:

```
# .dockerignore
.git
.gitignore
.dockerignore
Dockerfile
*.md
.env
.env.*
node_modules
dist
build
coverage
.vscode
.idea
*.test
*.log
**/*_test.go
tmp/
```

The `.dockerignore` does double duty: it shrinks the build context (faster builds, because less is sent to the daemon), and it makes `COPY . .` *safe* by guaranteeing the things that shouldn't ship can't ship. Even with a multi-stage build where the final image only gets the binary, you want this in the build stage — a leaked `.env` in the *build* stage's layers is still a leak if someone pulls the intermediate image or if your CI logs the context. Treat `.dockerignore` as mandatory, the same way you treat `.gitignore` as mandatory.

### The secrets-in-layers trap

There's a subtler version of the secret-leak problem that catches even experienced engineers, and it's worth a dedicated warning because it bites quietly. Because an image is a stack of layers and *each layer is permanent*, a secret that appears in *any* layer is in the image forever — even if a later layer deletes it. This is the cardinal sin of Docker secret handling:

```dockerfile
# NEVER do this — the secret is baked into a layer permanently
COPY .npmrc /root/.npmrc            # contains a registry auth token
RUN npm ci
RUN rm /root/.npmrc                 # too late: the token is still in the COPY layer
```

Deleting the file in a later `RUN` removes it from the *final merged filesystem*, but the layer that added it is still in the image. Anyone who pulls the image and runs `docker history --no-trunc` or extracts the individual layers can recover the token. The same trap applies to `ARG` and `ENV` values used for secrets — they're recorded in the image metadata and visible in `docker history`. The correct approach is BuildKit's secret mounts, which expose a secret to a single `RUN` step without ever writing it to a layer:

```dockerfile
# Mount the secret only for this RUN — never persisted to any layer
RUN --mount=type=secret,id=npmtoken \
    NPM_TOKEN=$(cat /run/secrets/npmtoken) npm ci
```

You pass it at build time with `docker build --secret id=npmtoken,src=$HOME/.npmrc`, and the token never touches a layer or `docker history`. The rule to internalize: **build-time secrets use secret mounts; runtime secrets come from the environment or a mounted volume at deploy time** (the subject of the configuration and secrets post). A secret should never appear in a `COPY`, an `ARG`, an `ENV`, or a `RUN` argument, because layers are forever.

## Image size and security: scanning, SBOMs, pinning

Smaller and non-root are foundational, but production-grade means *verifying* those properties continuously and making your builds reproducible. Three practices close the loop: vulnerability scanning, the software bill of materials, and digest pinning.

**Vulnerability scanning** means running a tool that compares the packages in your image against known-CVE databases and fails the build if it finds anything serious. The tool I reach for is [Trivy](https://github.com/aquasecurity/trivy) (open source, fast, accurate), though `docker scout`, Grype, and Snyk do the same job. You wire it into CI as a gate: build the image, scan it, and refuse to push if there are unfixed HIGH or CRITICAL vulnerabilities.

```bash
# Scan the built image; fail CI on fixable HIGH/CRITICAL CVEs
trivy image \
  --severity HIGH,CRITICAL \
  --ignore-unfixed \
  --exit-code 1 \
  registry.example.com/orderservice@sha256:<digest>

# Docker's built-in scanner does the same
docker scout cves registry.example.com/orderservice:latest
```

The `--ignore-unfixed` flag is a pragmatic choice: it skips CVEs that have no patch available yet, so the gate fails only on vulnerabilities you can actually *do* something about, rather than blocking deploys on findings nobody can fix. This is exactly where the minimal base pays off operationally — on distroless the scan returns almost nothing, so the gate is quiet and meaningful; on Debian it returns a hundred findings and the team learns to bypass the gate, which defeats the purpose.

A **software bill of materials (SBOM)** is a machine-readable inventory of everything in your image — every package, every version, every license. It's the artifact that lets you answer "are we affected by this newly-disclosed CVE?" *across the whole fleet in seconds* instead of rebuilding and rescanning forty services. When the next Log4Shell-scale vulnerability drops, the team with SBOMs queries their inventory and knows in minutes which services contain the vulnerable package and which versions; the team without them spends two days grepping Dockerfiles and re-scanning. You generate one at build time:

```bash
# Generate an SBOM in SPDX format and attach it as an attestation
docker buildx build \
  --sbom=true \
  --provenance=true \
  -t registry.example.com/orderservice:1.4.2 \
  --push .

# Or with trivy / syft directly
trivy image --format spdx-json --output sbom.json \
  registry.example.com/orderservice@sha256:<digest>
```

**Pinning by digest** is how you make the whole thing reproducible. We covered the tag-versus-digest distinction; here's how you actually pin. In your Dockerfile, pin your *base* images by digest so a base re-tag can't silently change your build:

```dockerfile
# Pin the base by digest — reproducible forever
FROM golang:1.22@sha256:7f3c...e91a AS build
FROM gcr.io/distroless/static-debian12:nonroot@sha256:2b8f...c40d
```

And in your Kubernetes manifests and deploy tooling, reference your *own* images by digest, not by a mutable tag like `:latest` or even `:1.4.2`:

```yaml
# Deploy the exact image, not whatever :latest happens to point at
image: registry.example.com/orderservice@sha256:9a1c...44ef
```

Digest pinning eliminates an entire class of "it worked yesterday" incidents. `:latest` is the worst offender — it means "whatever was pushed most recently," so two nodes pulling at slightly different times can run *different code*, and a rollback to "the same tag" can pull a different image than you tested. Pinning by digest guarantees every pod, in every environment, today and after the next rollback, runs the byte-identical artifact. The hardening scorecard below contrasts the naive image against the hardened one across every one of these controls.

![A matrix scorecard contrasting the naive image and the hardened image across size non root read only filesystem CVE scanning digest pinning health check and SBOM](/imgs/blogs/containerizing-microservices-docker-best-practices-8.webp)

The scorecard is the checklist version of this whole post. Naive fails every row: 1.2GB, root, writable filesystem, no scan, `:latest`, no health check, no SBOM. Hardened passes every row: 22MB, non-root UID 65532, read-only filesystem, a Trivy gate in CI, digest pinning, a defined health check, and an attached SBOM. The gap between those two columns is not exotic — it's a multi-stage build, a distroless base, a few lines of `securityContext`, a `.dockerignore`, and a scan step. A day of work that pays back forever, multiplied by every service you run.

## HEALTHCHECK and the orchestrator's view

An image isn't truly production-grade until it can tell the platform whether it's healthy. Docker's `HEALTHCHECK` instruction defines a command the runtime runs periodically to decide if the container is alive:

```dockerfile
HEALTHCHECK --interval=10s --timeout=2s --start-period=20s --retries=3 \
  CMD ["/app/orderservice", "healthcheck"]
```

Note that on a distroless base there's no shell, so you can't do `HEALTHCHECK CMD curl ...` — there is no `curl` and no `sh` to run it. Instead you invoke a subcommand of your own static binary (a common Go pattern: the binary, called as `orderservice healthcheck`, makes a localhost request to its own `/healthz` and exits 0 for healthy or 1 for unhealthy). This is one of the small adaptations the minimal base demands, and it's a clean one — your binary already knows how to talk to itself, so it's a few lines of code that ship for free. The same pattern works for `scratch` images: the health check is just another mode of the one binary you shipped.

In a Kubernetes world, the `HEALTHCHECK` instruction is largely superseded by Kubernetes' own liveness, readiness, and startup probes, which are richer (separate readiness from liveness, with separate timing and separate actions) and what the orchestrator actually acts on. The image's job is to *expose* the health endpoints; the platform's job is to *probe* them and act on the answers. This division of labor matters because the same endpoint, probed by the wrong kind of probe, can be dangerous. This is a deep topic with sharp edges — a liveness probe that checks a database can turn a brief dependency blip into a fleet-wide restart storm — and it gets the full treatment in [health checks, readiness, liveness, and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing). For our purposes here, the rule is simply: your image must ship health endpoints, and your binary should be able to probe itself for the no-shell base case.

## The build-to-deploy flow and one process per container

Step back and look at the whole lifecycle, because two principles only make sense at this altitude. The figure below traces an image from CI build through the scan gate, into the registry tagged and digested, then pulled by digest into staging and production, where Kubernetes schedules it across forty pods.

![A graph of the build to deploy flow from CI build through a scan gate into the registry and pulled by digest into staging and production pods scheduled by Kubernetes](/imgs/blogs/containerizing-microservices-docker-best-practices-5.webp)

The first principle the flow makes visible is **immutability and dev/prod parity in action**: one image, one digest, flowing unchanged through every environment. The image staging tested is the *exact same digest* production runs. This is the property that makes [independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability) and safe [blue-green and canary rollouts](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) possible — you're promoting an artifact, not rebuilding one. If you rebuild per environment, you've broken parity and reintroduced "works in staging, breaks in prod."

The second principle is **one process per container** (sometimes phrased "one concern per container"). A container should run a single primary process — your service — not a process manager juggling your app plus a log shipper plus a cron daemon plus an nginx sidecar all stuffed into one image. There are concrete reasons. The container runtime's lifecycle is built around PID 1: when PID 1 exits, the container stops, and the orchestrator restarts or reschedules it. If you cram multiple processes in and one of them dies silently, the container looks healthy while a piece of it is broken. Scaling also breaks: if your app and a batch job share a container, you can't scale them independently — scaling for app load also scales the batch job, wastefully. The microservices-native pattern is to keep the container to one process and push the auxiliary concerns out: logging goes to stdout/stderr and a node-level collector ships it; cross-cutting network concerns (mTLS, retries) go to a sidecar in the same pod but a *separate* container (the [service mesh](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) model). The image stays a clean, single-purpose unit, which is exactly what makes it composable in a fleet.

This also reinforces a configuration principle worth stating explicitly: **config comes from the environment, not baked into the image** — the [twelve-factor](https://12factor.net/config) rule. The same immutable image must run in dev, staging, and prod, which is only possible if its configuration (database URLs, feature flags, log levels) is injected at runtime via environment variables and mounted secrets, never compiled in. Bake a staging database URL into the image and you've destroyed the one-artifact-everywhere property. How to do this safely — env vars, ConfigMaps, and especially secrets that don't end up in image layers or `docker history` — is the subject of [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management). The container best practice is the negative rule: secrets and environment-specific config never go in the Dockerfile.

## Optimization: making the build production-grade

We've already hit the two biggest optimizations — multi-stage builds for size and layer ordering for cache hits — but a production build pipeline at fleet scale has a few more levers worth pulling, each with measurable payoff.

**Cache mounts** (covered above) keep dependency caches warm across builds without bloating images, and they're the biggest win for dependency-heavy languages. **Smaller base images** speed up not just storage but *pull time* on every node, which directly reduces pod cold-start latency under autoscaling. **Parallel stage builds**: BuildKit (Docker's modern build engine, default in recent versions) automatically parallelizes independent stages. If your build has, say, a stage that compiles the binary and an independent stage that builds frontend assets, BuildKit runs them concurrently rather than serially. You enable BuildKit's full power with `docker buildx` and structure your Dockerfile so independent work lives in independent stages.

**Remote/shared build cache** is the fleet-scale optimization most teams miss. By default the build cache is local to whichever machine ran the build. In CI, where every job often starts on a fresh ephemeral runner with a cold cache, that means *no* cache reuse — every build is a cold build. BuildKit can export and import the cache to/from a registry, so a fresh CI runner imports the warm cache from the last build:

```bash
docker buildx build \
  --cache-from type=registry,ref=registry.example.com/orderservice:buildcache \
  --cache-to type=registry,ref=registry.example.com/orderservice:buildcache,mode=max \
  -t registry.example.com/orderservice:1.4.2 \
  --push .
```

With registry-backed cache, a CI runner that has never built this service before still gets the cache-hit speedup, because it pulls the cached layers from the registry. This is what turns "every CI build is a cold four-minute build" into "every CI build is a warm twenty-five-second build," across ephemeral runners. **Measure the win** the way you'd measure any optimization: track CI build duration (p50 and p95), image size per service, node image-pull time (visible in pod scheduling latency), and registry storage and egress cost. Each optimization should move a specific one of those numbers, and if it doesn't, you've cargo-culted it.

## Stress test: a CVE drops in your base image at 3am

Now the problem-solving narrative. A best practice is only worth anything if it holds up under the stress it was designed for, so let's pose the scenario these practices exist to survive, then reason through whether the hardened setup actually wins.

**The scenario.** It's 3am. A critical CVE is disclosed in a widely-used base image — exactly the kind of "drop everything" zero-day that happens a couple of times a year. Your fleet is 40 services. Every one of them needs to be rebuilt on a patched base and redeployed, fast, because the vulnerability is being actively exploited in the wild. The question that determines whether your night is a fifteen-minute non-event or an eight-hour scramble is: **how fast can you rebuild and redeploy 40 services?**

Walk through it on the hardened setup. First, *do you even know which services are affected?* If you have SBOMs, you query your inventory and in seconds you know exactly which of the 40 contain the vulnerable component — maybe it's all of them, maybe it's the 28 that share a particular base. Without SBOMs, you're grepping Dockerfiles at 3am. Second, *the fix is one edit.* Because you pinned base images by digest, patching means bumping the base digest — ideally in a shared base-image definition or a renovate-bot PR that touches all the affected Dockerfiles at once. Third, *the rebuild is fast.* Because of cache-friendly ordering and registry-backed build cache, changing only the base layer means your dependency and source layers are still cache hits — only the base and the final assembly rebuild. A service that takes four minutes cold rebuilds in well under a minute warm. Fourth, *the scan re-gates automatically.* Your CI re-runs Trivy against the rebuilt image; if the patched base cleared the CVE, the gate passes; if not, you know immediately. Fifth, *the deploy is a rolling update of a new digest* — no manual server patching, no SSH, just the orchestrator replacing pods with the new image. The figure below times this out: from CVE disclosure to 40 patched services in about fifteen minutes.

![A timeline of patching a base image CVE across the fleet from disclosure to digest bump rebuild rescan and a rolling deploy of forty patched services](/imgs/blogs/containerizing-microservices-docker-best-practices-9.webp)

Now run the *same* scenario on the naive setup, and watch every shortcut become a wound. You don't have SBOMs, so you don't know which services are affected — you check all 40 by hand. Your Dockerfiles pin `FROM ubuntu:22.04` by tag, not digest, so "rebuild on the patched base" means hoping the upstream re-tag has landed and praying nothing else changed in the meantime. Your layer ordering is bad, so every rebuild is a cold four-minute build, and on ephemeral CI runners with no shared cache it's four minutes *each*, serially, because your CI can't parallelize the way a clean setup can — 40 × 4 minutes is over two and a half hours of pure build time. You have no scan gate, so you don't actually *know* the rebuild fixed the CVE; you're shipping on faith. And because some of those services were patched in place last quarter during a different fire, their running state has drifted from their Dockerfiles, so a rebuild produces a subtly different image that breaks in ways nobody can explain at 3:40am. The naive fleet doesn't patch in fifteen minutes. It patches in a day, with incidents.

This is the senior point made operational. Every best practice in this post — the minimal base, digest pinning, cache-friendly ordering, the scan gate, the SBOM, immutable deploys — is individually a small discipline. Together they're the difference between a CVE being a routine maintenance task and a CVE being a multi-team, multi-hour emergency. You don't adopt these practices because a linter told you to. You adopt them because they're what makes the 3am page boring, and boring is the highest compliment you can pay an on-call rotation.

## Stress test: the image is 1GB and cold starts are slow

A second, quieter failure mode, because not every problem is a security drill. ShopFast's recommendation service is built on a fat single-stage Python image — `python:3.12` base, the full build toolchain for some native ML dependencies, the works — and it weighs 1.1GB. The symptom isn't a breach; it's that autoscaling doesn't work well enough. When traffic spikes (a marketing email goes out, everyone hits the site at once), the Horizontal Pod Autoscaler tries to scale the recommendation service from 8 pods to 30. But each new pod lands on a node that may not have the 1.1GB image cached, so it has to pull it first — and at ~100MB/s that's *eleven seconds of image pull* before the container can even start, on top of the service's own slow boot. By the time the new pods are serving, the traffic spike is half over and users have already hit slow pages. The autoscaler is fighting with one hand tied behind its back.

Reason through the fix using exactly the tools in this post. Move the recommendation service to a multi-stage build: a fat builder with the ML toolchain that compiles the native dependencies, and a slim runtime stage on a distroless Python base that ships only the wheels and the app code. The image drops from 1.1GB to about 180MB (Python services with real ML deps don't get to 22MB, but a 6× reduction is enormous). Now a cold pull is under two seconds instead of eleven. Layer-share the common distroless Python base across the fleet so most nodes already have it cached, often making the pull near-instant. The autoscaler can now bring up new pods fast enough to actually absorb the spike. The cost dimension improves in lockstep — less storage, less egress, faster deploys.

The reusable lesson: **image size is not just a tidiness concern, it's a latency and elasticity concern.** In a microservices fleet that relies on autoscaling and rescheduling, the image is on the critical path of every scale-up and every node replacement, so a bloated image directly degrades your ability to respond to load. The same multi-stage, minimal-base discipline that shrinks your attack surface also sharpens your autoscaling. One set of practices, two payoffs.

## Case studies

Theory is cheaper than scar tissue, so here are three real-world threads that show these practices mattering at scale, with the lesson each one teaches.

**Google and distroless.** The `distroless` project came out of Google's own internal practice of building images that contain only an application and its runtime dependencies — no shell, no package manager, no extraneous userland. Google open-sourced the base images (`gcr.io/distroless/*`) and the tooling, and the motivation is exactly the security-and-size argument made above: at Google's scale, the difference between a base with a hundred packages and a base with the bare minimum is the difference between a constant stream of CVE noise and a manageable signal, multiplied across an enormous fleet. The lesson distroless teaches is that **debuggability is a tradeoff you can engineer around, but attack surface is a cost you pay continuously.** Google chose to lose the in-container shell — and to build the `kubectl debug` ephemeral-container workflow to compensate — because the security and operational win of a minimal base outweighs the convenience of `exec`-ing in. When the default base for the industry is "a full OS," consciously choosing distroless is the move that separates a careful team from a default one.

**A supply-chain and scanning win.** The 2021 Log4Shell vulnerability (CVE-2021-44228 in Log4j) was the moment SBOMs stopped being a compliance checkbox and became an operational necessity. When the vulnerability dropped, the universal question every engineering org asked was "are we affected, and where?" — and the answer was painful precisely because most organizations had no machine-readable inventory of what was inside their running images. Teams that had been generating SBOMs could query "which images contain a vulnerable Log4j version?" and get an answer in minutes; teams that hadn't spent days manually auditing dependency trees across hundreds of services and images, often re-scanning everything from scratch under time pressure. The lesson is that **you cannot respond to a supply-chain vulnerability you can't inventory.** The investment in scanning and SBOM generation looks like overhead right up until the moment a fleet-wide vulnerability drops, at which point it's the difference between a fifteen-minute query and a multi-day fire drill. Log4Shell is also why software supply-chain security (SBOMs, provenance attestations, signed images) moved from "nice to have" to a board-level concern across the industry.

**The fat-image cost story.** This pattern is common enough across the industry to be a genre: a team containerizes a polyglot microservices fleet by the most direct path — each service gets a single-stage Dockerfile on a full-language base — ships dozens of services each weighing 800MB to 1.5GB, and only later discovers the bill. The costs show up in three places at once. Registry storage balloons because every service retains many tagged versions for rollback. Network egress climbs because every deploy and every autoscale event pulls these images, and a meaningful fraction crosses billed network boundaries. And CI compute is wasted because every build is a slow cold build. The fix is always the same — multi-stage builds and minimal bases — and the reported savings are routinely *order-of-magnitude* on image size, with the storage, transfer, and build-time costs falling proportionally. The lesson: **the easy way to containerize and the right way to containerize diverge sharply at fleet scale, and the gap is denominated in dollars that don't show up until you're running forty services, not four.** The discipline that feels like over-engineering for one service is straightforwardly economical for forty.

## When to reach for which base (and the honest caveats)

A decisive recommendation, since the whole post builds toward one. Default to **distroless** for the great majority of production microservices — it's the right balance of tiny, secure, and glibc-compatible, and the lost shell is a debugging inconvenience you solve with ephemeral debug containers rather than a real blocker. Reach for **scratch** specifically when you have a fully static binary (Go with `CGO_ENABLED=0`, Rust against musl) and you want the absolute minimum surface and your team is comfortable debugging from the outside — it's ideal for gateways, proxies, and small static services. Choose **alpine** when you genuinely need an in-container shell during normal operations *and* you've verified your dependencies are musl-clean and you're prepared for the musl DNS/threading gotchas — it's a fine middle ground, just not a thoughtless default. Settle for a **full OS base** only when you have a hard requirement for a full glibc userland and a pile of system packages that distroless can't satisfy, and treat that as a smell worth investigating rather than a comfortable default.

And the meta-recommendation, the one that outlives any specific base: **always multi-stage, always non-root, always scanned, always digest-pinned, always a `.dockerignore`.** Those five are not base-dependent — they apply no matter what you put in the final `FROM`. The base is a tuning knob; those five are the floor. If you do nothing else from this post, do those five, and you've already moved your fleet from "embarrassing" to "defensible."

This is also a small lesson in [evolutionary architecture](/blog/software-development/system-design/evolutionary-architecture-designing-for-change): you don't have to build the perfect image on day one, but you do have to leave the seams that let it evolve cheaply — the multi-stage structure, the pinned base, the scan gate — so that hardening later is a clean refactor rather than a rewrite under fire.

When is the *naive* approach acceptable? Honestly: a throwaway local dev image, a one-off script you'll run once and delete, a prototype you're certain will never see production or a registry. The moment an image is going to run in production, sit in a registry, get pulled across a fleet, or be subject to a security review, the naive approach has costs that compound, and the hardened approach is a day of work that pays back forever. The trap is that the naive image *works*, so the deadline pressure says ship it — and then it quietly accrues cost and risk until an incident forces the rework under worse conditions. Pay the day up front.

## Key takeaways

- **Your image is a security and cost surface, not just a package.** Every megabyte and every package is something you store, transfer, scan, and defend, multiplied by every replica of every service. Treat the image as a liability to minimize, not an artifact to ignore.
- **A container is an isolated process on a shared kernel, not a lightweight VM.** That's why it starts in milliseconds and packs densely — and why the kernel boundary is soft, which is why non-root, dropped capabilities, and a read-only filesystem are non-optional, not nice-to-haves.
- **Multi-stage builds are the single biggest win.** Compile in a fat builder, ship only the binary into a tiny runtime stage. This alone takes a Go service from 1.2GB to ~22MB and removes the entire toolchain from your attack surface.
- **The base image is your biggest security lever.** Distroless or scratch cut CVE counts from triple digits to near zero, which is the difference between a scan gate that works and one everyone learns to bypass. Default to distroless.
- **Layer ordering is the difference between a 25-second and a 4-minute build.** Copy the dependency manifest and install dependencies *before* copying source, so a code-only change rebuilds in seconds. Add cache mounts and registry-backed cache for ephemeral CI runners.
- **Run as non-root with a read-only filesystem and all capabilities dropped.** It's nearly free and it shrinks the blast radius of any compromise from "root with a shell and a package manager" to "an unprivileged process in a box with no exits."
- **Pin by digest, scan in CI, generate an SBOM.** Digests make builds reproducible and rollbacks honest; the scan gate catches real vulnerabilities; the SBOM is what lets you answer "are we affected?" in minutes instead of days when the next Log4Shell drops.
- **Config and secrets come from the environment, never the image.** One immutable artifact runs everywhere only if its configuration is injected at runtime. A baked-in secret or staging URL destroys dev/prod parity and leaks into image layers.
- **Image size is a latency and elasticity concern, not just tidiness.** A bloated image is on the critical path of every autoscale event and node replacement, so shrinking it directly improves how fast you can respond to load.
- **The senior move is the boring 3am page.** Adopt these practices so a fleet-wide CVE is a fifteen-minute digest bump and rolling deploy, not an eight-hour scramble. The disciplines are individually small; together they make the worst night routine.

## Further reading

- **Docker's official best-practices guide** — [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) and the [multi-stage build docs](https://docs.docker.com/build/building/multi-stage/). The canonical reference, kept current with BuildKit features.
- **Google's distroless project** — [GoogleContainerTools/distroless](https://github.com/GoogleContainerTools/distroless). The base images, the rationale, and the variants for each runtime.
- **Trivy** — [aquasecurity/trivy](https://github.com/aquasecurity/trivy). The open-source scanner used in this post's CI gate; the docs cover image, filesystem, and SBOM scanning.
- **The Twelve-Factor App** — [12factor.net](https://12factor.net/). Especially the [config](https://12factor.net/config) and [dev/prod parity](https://12factor.net/dev-prod-parity) factors, which the container model makes nearly free.
- **Sam Newman, *Building Microservices* (2nd ed.)** — the deployment and operations chapters frame the image-as-deployment-unit thinking that underpins this whole track.
- **Series cross-links**: [anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice) for the unit you're packaging, [health checks and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) for the probes your image must expose, [Kubernetes for microservices](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials) for the orchestrator that schedules these images, [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability) for the pipeline that builds and ships them, [deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) for promoting one digest safely, and [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management) for keeping config out of the image.

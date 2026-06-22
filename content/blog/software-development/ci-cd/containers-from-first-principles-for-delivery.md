---
title: "Containers From First Principles, For Delivery: Why the Image Is the Unit You Ship"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A container is not a tiny VM. It is a normal Linux process with its view of the world restricted by the kernel, packaged with its dependencies into an immutable image. Understand that mechanism and every delivery decision downstream stops being magic."
tags:
  [
    "ci-cd",
    "devops",
    "containers",
    "docker",
    "oci",
    "namespaces",
    "cgroups",
    "build-once-promote-everywhere",
    "image-layers",
    "delivery",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/containers-from-first-principles-for-delivery-1.png"
---

A few years ago I watched a service that passed every test in CI fall over thirty seconds after it landed in production. The build was green. The image was identical to the one staging had been running happily for a week. The only thing that changed was the production node had 64 GB of RAM and a hard 512 MB memory limit on the container, and the JVM inside it had no idea the limit existed. It looked at the box, saw 64 GB, sized its heap accordingly, allocated past 512 MB, and the kernel reached in and killed it. Exit code 137. The pod restarted, did the same thing, restarted again, and the crash-loop pager went off at 2 a.m. The "broken" thing was not the code. It was that nobody on the team could actually say what a container *was* — they treated it as a small virtual machine, a box with its own everything, and so the idea that the process inside could see the host's full 64 GB while being limited to 512 MB simply did not compute.

That is the gap this post closes. In the [CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) that anchors this series, the spine is **commit → build → test → package → deploy → operate**, governed by two principles: *build once, promote everywhere*, and *everything as code*. The container is the thing that flows down that spine. It is the **unit of delivery** — the immutable artifact you build once, test once, and promote unchanged into every environment. If you do not understand what it actually is, you cannot reason about why "works on my machine" disappears, why an image is safe to promote by digest, why a 1.9 GB image is a delivery liability, or why your JVM got OOM-killed. So we are going to take the container apart, all the way down to the kernel, and put it back together as a delivery artifact you can reason about.

Here is the thesis, stated once and defended for the rest of the post: **a container is not a tiny VM. It is a normal Linux process running on the host kernel, with its view of the world restricted by a handful of kernel features, packaged together with its dependencies into an immutable, content-addressed image.** Three kernel features do the restricting — *namespaces* (what the process can see), *cgroups* (what the process can use), and a *union/overlay filesystem* (what the process reads and writes). The packaging format is the *OCI image*. Get those four ideas and the rest of cloud-native delivery stops being magic.

![A stacked diagram showing a container as an ordinary process whose view is narrowed by namespaces, cgroups, and an overlay filesystem on a shared host kernel](/imgs/blogs/containers-from-first-principles-for-delivery-1.png)

By the end you will be able to: read an OCI image manifest and explain what each piece does; predict an OOM kill from a memory limit and fix it; explain to a skeptical colleague exactly why a container shares the host kernel and a VM does not; break down why a 75 MB image is a better delivery unit than a 1.9 GB one, layer by layer; and decide, with reasons, when a plain container's isolation is good enough and when you need gVisor or a micro-VM. We will not re-derive Kubernetes orchestration — that belongs to the [Kubernetes-for-delivery](/blog/software-development/ci-cd/kubernetes-for-delivery-the-objects-that-matter) post and to [Kubernetes for microservices](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials). This post is about the brick. Those posts are about the wall.

## 1. The core misconception: a container is not a tiny VM

Let me start by killing the wrong model, because it causes more delivery bugs than any other single misunderstanding.

The wrong model says: a virtual machine is a whole computer simulated in software, and a container is the same idea, just lighter — a smaller box with its own operating system, its own kernel, its own everything, that happens to boot faster. People reach for this because the *experience* of using a container resembles using a tiny machine. You "run" it, you "ssh into it" (well, `docker exec`), it has its own process list, its own network address, its own filesystem. It quacks like a small VM, so the mind files it as one.

It is wrong in the one way that matters most for delivery: **a container does not have its own kernel.** A virtual machine virtualizes *hardware* — the hypervisor presents fake CPUs, fake memory, fake disks, and a full guest operating system boots on top of that fake hardware, with its own kernel managing those fake resources. A container virtualizes nothing. The process inside a container is running directly on the *host's* kernel, the same kernel every other container on that machine is using, the same kernel the host's own processes use. There is exactly one kernel on the box. The container is just a process (or a small tree of processes) that the kernel has been told to show a restricted view of the world to.

This single fact explains an enormous amount:

- **Why containers start in milliseconds and VMs take tens of seconds.** A VM has to boot an operating system — POST, bootloader, kernel init, systemd, the works. A container has nothing to boot; the kernel is already running. Starting a container is `fork`/`exec` of a process with some extra flags. That is why the container is the right unit for a delivery system that scales pods up and down constantly: starting another copy is nearly free.
- **Why you can pack hundreds of containers on a box but only a handful of VMs.** Every VM carries the dead weight of a full guest OS in memory — easily 512 MB to 1 GB before your app does anything. Containers share the one kernel, so the only memory cost is your app's actual footprint. Density is a delivery economics question, and the kernel-sharing is what makes the economics work.
- **Why a container built for Linux cannot run natively on a Mac or Windows kernel.** It is a Linux process using Linux kernel features. On macOS and Windows, Docker Desktop quietly runs a small Linux VM and runs your "containers" inside *that*. The container is not magic cross-platform — it is Linux-kernel-shaped, and the desktop tools hide a VM to give you Linux on a non-Linux host.
- **Why the security boundary is weaker than a VM's.** This is the trade-off that bites people, and we will spend a whole section on it. If there is one kernel and a hundred containers share it, then a kernel vulnerability is a shared blast radius. A VM's guest kernel is a real wall; a container's namespace is a restricted view, and views can sometimes be escaped.

Put plainly: a VM is *isolation by simulation* — fake hardware, real second kernel. A container is *isolation by restriction* — real hardware, one kernel, a narrowed view. Hold onto "narrowed view of the world," because the next three sections are precisely the three things the kernel narrows.

## 2. Namespaces: the kernel feature that narrows what a process can see

A **namespace** is a Linux kernel feature that gives a process its own private view of one particular global resource. Normally every process on a machine shares one global view — one list of process IDs, one network stack, one filesystem tree, one hostname. A namespace wraps one of those global resources so that a process inside the namespace sees only its slice and is blind to everything outside it. The process is still on the host kernel, still scheduled by the host scheduler, still using host RAM. It just *cannot see* the rest of the machine.

There are several namespace types, and a "container" is what you get when you create a process inside a fresh set of all of them at once.

![A taxonomy tree of the six Linux namespaces grouped into identity-and-naming namespaces and resource-view namespaces](/imgs/blogs/containers-from-first-principles-for-delivery-2.png)

- **PID namespace** gives the process its own process-ID tree. Inside the namespace, the first process is PID 1, and it can see only the processes in its own namespace. Run `ps aux` inside a container and you see a tiny list — your app and maybe a couple of children — not the host's hundreds of processes. From the *host's* point of view, those same processes have ordinary, large PIDs and are perfectly visible. It is the same processes, two different views. (This is also why PID 1 in a container matters: PID 1 has special signal-handling responsibilities, and a naive PID 1 that doesn't reap zombies or forward `SIGTERM` causes the "my container takes 10 seconds to stop" delivery annoyance.)
- **Network (net) namespace** gives the process its own network stack: its own interfaces, its own routing table, its own `iptables` rules, its own loopback. A container's `eth0` is a virtual interface, usually one end of a `veth` pair whose other end lives in the host and is bridged out. This is why two containers can both bind port 8080 without colliding — they have *different* network stacks. The port conflict you'd get running two copies of a server on a host simply does not exist between containers, because each has its own private set of ports.
- **Mount (mnt) namespace** gives the process its own filesystem mount table. This is what lets a container have a completely different root filesystem from the host — its `/` is the image's filesystem, not the host's. Mounts made inside the namespace (like a tmpfs, or a volume) are invisible to the host and to other containers. The mount namespace is the hook the overlay filesystem (section 4) hangs on.
- **UTS namespace** gives the process its own hostname and domain name. Tiny, but it is why `hostname` inside a container returns the container's name and not the host's.
- **IPC namespace** gives the process its own System V IPC and POSIX message queues — its own shared-memory segments. Two containers can't accidentally stomp on each other's shared memory.
- **User namespace** maps user and group IDs between the container and the host. This is the one with the biggest security implications: with a user namespace, UID 0 (root) *inside* the container can be mapped to an unprivileged UID *outside* on the host. So a process that thinks it is root, and behaves like root inside its restricted view, is actually a nobody on the host. Rootless containers lean entirely on this. (By default, plain Docker does **not** turn on user namespaces, which is exactly why root-in-container is a real risk — more in section 8.)

### Seeing namespaces without Docker at all

The cleanest way to feel that a container is "just namespaces" is to make one by hand, with no container runtime in sight, using `unshare` — a coreutils tool that runs a program in new namespaces.

```bash
# Run a shell in a new PID + mount + UTS namespace, as a fake root via a user namespace.
# --fork is needed because the new PID namespace's PID 1 must be a child.
sudo unshare --pid --mount --uts --fork --mount-proc bash

# Now, inside that shell:
hostname container-by-hand     # UTS namespace: change hostname freely
ps aux                          # PID namespace: you are PID 1, you see almost nothing
echo $$                         # prints 1 — you are the init of this namespace
```

That shell is, for all practical purposes, the inside of a container — its own process tree, its own hostname, its own mount table — and you built it with a single coreutils command. No image, no Docker, no daemon. There is nothing else in a container's *isolation* that this doesn't show; everything Docker adds on top is packaging, networking plumbing, and ergonomics.

You can also peek the *other* direction — from the host into a running container's namespaces — with `nsenter`:

```bash
# Find the host PID of a running container's main process.
PID=$(docker inspect --format '{{ .State.Pid }}' my-api)

# Enter that container's network + PID namespaces from the host, run a command.
sudo nsenter --target "$PID" --net --pid ip addr      # see the container's eth0
sudo nsenter --target "$PID" --net --pid ss -tlnp     # see what's listening inside
```

`nsenter` is one of the most useful debugging tools in delivery work: when a container ships without `curl`, `ip`, or a shell (as a good distroless image will — section 6), you can still enter its network namespace *from the host* using host tools and inspect what it's actually doing. The container's lack of a debug toolchain doesn't blind you, because the isolation is a view you can step into, not a wall you have to breach.

The delivery lesson of namespaces: the reason "it works on my machine" stops being a thing is partly the image (section 5) and partly *this* — the process runs with the same restricted, predictable view of the world everywhere, because that view is constructed from the image, not inherited from whatever junk happens to be on the host. The host's other processes, ports, mounts, and hostname are simply invisible. The environment is the image plus a clean set of namespaces, and that is identical in CI, on staging, and in production.

There's one more thing namespaces clarify that trips people up constantly: **namespaces are composable and not all-or-nothing.** A container doesn't have to take a fresh copy of every namespace. You can deliberately *share* a namespace between a container and the host, or between two containers, and runtimes do this on purpose. When you run `docker run --network=host`, the container *shares the host's net namespace* — it has no private network stack at all, it binds directly to host ports, and the port-isolation benefit vanishes. When you run `--pid=host`, the container can see (and signal) every process on the host. A Kubernetes *pod* is precisely a group of containers that *share a net and IPC namespace* (so they reach each other on `localhost` and share an IP) while keeping *separate mount and PID namespaces* by default — that's why a sidecar container can talk to the main container over `localhost:8080` but can't see its files. Once you understand that a container is a set of namespaces, "what's a pod?" answers itself: it's a shared-namespace bundle of containers. (The orchestration semantics of pods are the [Kubernetes-for-delivery](/blog/software-development/ci-cd/kubernetes-for-delivery-the-objects-that-matter) post; the *mechanism* is right here.)

This composability is also a security knob you'll touch in section 8. Every namespace you share with the host is a piece of isolation you gave back. `--network=host` is sometimes justified for performance, `--pid=host` occasionally for a monitoring agent that must see all processes — but each is a deliberate hole in the fence, and a delivery platform should make sharing host namespaces an explicit, reviewed exception, never a default someone copy-pastes from a Stack Overflow answer to "fix" a networking problem they didn't understand.

## 3. Cgroups: the kernel feature that narrows what a process can use

Namespaces control what a process can *see*. They do nothing about what it can *use*. A process in a fresh PID namespace can still spin every CPU core to 100% and allocate every byte of host RAM — it just can't see the other processes while it does it. That is the job of the second kernel feature: **control groups**, universally shortened to **cgroups**.

A cgroup is a kernel mechanism that groups processes together and then *limits and accounts for* their resource usage — CPU time, memory, block I/O, the number of PIDs, and more. You put a process into a cgroup, set limits on that cgroup, and the kernel enforces them. "Account" matters as much as "limit": cgroups are also how the kernel and your monitoring know how much CPU and memory a specific container is actually using, which is the data your autoscaler and your cost dashboards run on.

When you run a container with a memory limit, the runtime is creating a cgroup, putting the container's process into it, and writing the limit into the cgroup's control files. There is no virtualization, no fake hardware. It is the host kernel saying "this group of processes may use at most this much, and I am counting."

### The most important cgroup behavior for delivery: the OOM kill

Here is the behavior that produces the most surprising production incidents. When a process tries to allocate memory that would push its cgroup over its memory limit, the kernel does not gently return an error to the application (most apps don't check `malloc` for failure anyway). It invokes the **OOM (out-of-memory) killer** scoped to that cgroup, picks a victim process in the group, and kills it. For a single-process container that victim is your app. The process dies with signal 9 (`SIGKILL`); the container exits with code **137** (128 + 9). No graceful shutdown, no flush, no log line from your app saying goodbye. Just gone.

This is the mechanism behind the war story in the intro, and it is so common it deserves its own worked example.

#### Worked example: the JVM that ignored its 512 MB limit

A team containerized a Java service. The Dockerfile ran it with a generous-looking heap. Locally and in staging (small boxes), it was fine. In production it crash-looped within seconds of starting under load.

The numbers:

- Container memory limit (the cgroup): **512 MB**.
- Production host RAM: **64 GB**.
- The JVM in use predated container-awareness, so it sized its default heap at **1/4 of what it saw as "available RAM."** It read the *host's* `/proc/meminfo`, saw 64 GB, and happily set a max heap of about **16 GB**.

So the JVM believed it had a 16 GB ceiling while the cgroup had a 512 MB ceiling. The first time the heap grew past 512 MB — which under any real load is immediate — the cgroup OOM killer fired and the process died with exit 137. Restart, repeat, crash-loop.

![A timeline of a container being out-of-memory killed because the JVM read the host RAM instead of its cgroup memory limit](/imgs/blogs/containers-from-first-principles-for-delivery-7.png)

Why did the JVM read 64 GB? Because an *old* JVM didn't look at the cgroup — it looked at the host. The container's *view* of memory was not namespaced the way PIDs and the network are; historically `/proc/meminfo` inside a container showed the host's totals. The application has to be *told*, or *taught*, about the cgroup limit. The fixes, in order of preference:

```bash
# Fix 1 (best, modern JVMs >= 10/11): let the JVM read the cgroup, and size the heap
# as a PERCENTAGE of the cgroup limit, not the host.
docker run --memory=512m --memory-swap=512m \
  myorg/api:1.4.2 \
  java -XX:MaxRAMPercentage=75.0 -jar /app/app.jar
# 75% of 512MB = ~384MB heap, leaving headroom for metaspace, threads, off-heap.

# Fix 2 (explicit, works everywhere): just set the max heap by hand, below the limit.
docker run --memory=512m \
  myorg/api:1.4.2 \
  java -Xmx384m -jar /app/app.jar
```

Modern JVMs (10+, and solidly from 11) are *cgroup-aware*: by default they detect the container memory limit and size the heap from it. The flag `-XX:MaxRAMPercentage` is the clean lever — "use up to 75% of whatever my container limit is" — and it follows the limit automatically if you later raise or lower it. The off-by-default-on-old-runtimes hazard generalizes far beyond Java: **Node's `--max-old-space-size`, Go's `GOMEMLIMIT`, Python multiprocessing pool sizing, and `nproc`/CPU detection in many runtimes all historically read the host, not the cgroup.** Whenever a runtime auto-sizes something from "available resources," ask: available to *whom* — the host, or my cgroup? Getting that wrong is a top-five cause of container crash-loops.

### Requests, limits, and how they map down to cgroups

In Kubernetes you express resource needs as `requests` and `limits`. People treat these as orchestration concepts, but they bottom out in cgroups on the node:

- A **memory limit** becomes the cgroup memory ceiling. Exceed it and you get the OOM kill, exit 137, exactly as above.
- A **memory request** is used by the scheduler to decide which node has room; it is the floor the scheduler reserves. It is not a hard runtime cap, but it shapes packing.
- A **CPU limit** becomes a cgroup CPU bandwidth quota (CFS quota/period). Here the behavior is *different* from memory: exceeding a CPU limit does not kill you, it **throttles** you — the kernel pauses your process until the next scheduling period. That's why a CPU-limited service shows up as latency spikes and tail-latency pain, not crashes. (A subtlety: a too-low CPU limit on a latency-sensitive service can throttle it into SLO violations even at low average utilization — a real, documented trap.)
- A **CPU request** is a scheduler hint and a cgroup *shares* weight — under contention, the kernel gives CPU time proportional to shares.

The delivery takeaway: when you set `resources.limits.memory: 512Mi` in a manifest, you are writing a number into a kernel cgroup file on whatever node the pod lands on, and you are arming the OOM killer at that threshold. Treat the number as a contract your app must respect — and make your runtime cgroup-aware so it actually does.

### cgroups v1 vs v2, and seeing the files yourself

It's worth knowing there are two generations. **cgroups v1** had a separate hierarchy per controller (one tree for memory, another for CPU, another for I/O), which made consistent grouping awkward. **cgroups v2** unified them into a single hierarchy with a cleaner interface, and it's the default on modern distros and what Kubernetes increasingly assumes. The practical reason to care: some older container-aware logic and some monitoring tools read v1 paths, and on a v2-only host they silently misread limits — another flavor of the "auto-sizing reads the wrong thing" bug. If a runtime reports the wrong memory limit on a new node, a cgroup-version mismatch is a prime suspect.

You don't have to take any of this on faith — the cgroup is a set of files in a pseudo-filesystem, and you can read them. On a cgroups-v2 host:

```bash
# Inside a container with --memory=512m, the kernel exposes the limit as a plain file.
cat /sys/fs/cgroup/memory.max          # prints 536870912  (= 512 * 1024 * 1024)
cat /sys/fs/cgroup/memory.current      # prints current usage in bytes
cat /sys/fs/cgroup/cpu.max             # prints e.g. "150000 100000" -> 1.5 CPUs

# How many times has this cgroup hit its memory limit and triggered reclaim/OOM?
cat /sys/fs/cgroup/memory.events       # look for the "oom_kill" counter ticking up
```

When you `cat memory.max` and see exactly `536870912`, the limit stops being a vague concept: there is a literal number in a literal file, written by the runtime, enforced by the kernel. The `memory.events` file is gold for delivery debugging — if `oom_kill` is climbing, you have your answer for the crash-loop, and you didn't need to guess. This is also exactly the data your monitoring scrapes; container CPU/memory metrics in Prometheus are these cgroup files, surfaced. The reliability practice of *alerting* on them lives in SRE territory; the *mechanism* is a handful of readable files.

## 4. The union filesystem: layers, copy-on-write, and why images are cheap

The third kernel-backed mechanism is the **union (overlay) filesystem**, and it is the one that makes images cacheable, shareable, and fast — the properties delivery cares about most.

A container's root filesystem is not a copy of an OS sitting on disk. It is a *stack of layers* unioned together by a filesystem driver (on Linux, almost always `overlay2`). The bottom layers are **read-only** — they come from the image and are shared, byte-for-byte, by every container started from that image. On top of them the runtime adds one thin **writable layer**, private to the running container. The union filesystem merges them into a single view: reads fall through the stack to the first layer that has the file; writes go to the top layer using **copy-on-write**.

![A stack diagram of overlay filesystem layers showing read-only image layers shared and cached beneath a thin per-container writable layer](/imgs/blogs/containers-from-first-principles-for-delivery-4.png)

Copy-on-write (CoW) is the trick. When a container reads `/usr/lib/libssl.so`, the union FS finds it in a read-only image layer and serves it directly — no copy. When the container *writes* to a file, the union FS first copies that file up into the writable layer, then applies the write there, leaving the read-only layer untouched. Delete a file and the union FS adds a "whiteout" marker in the top layer that hides it. The read-only layers never change; the container's mutations live only in its private thin top layer.

Two delivery consequences fall straight out of this:

1. **Layers are shared across containers and across images.** If ten containers run from the same image, the read-only layers exist *once* on disk and *once* in page cache — ten containers, one copy of the base OS, one copy of the dependencies. Start an eleventh container and it shares them too. This is the density win again, now on the storage and memory-cache axis, not just RAM.
2. **Layers are content-addressed and cached at every hop.** Each layer is identified by the SHA-256 of its contents. If two images share a base layer (the same `python:3.12-slim`, say), the registry stores it once and a node that already has it does not re-pull it. This is the whole reason `docker pull` of your hundredth microservice image takes seconds, not minutes — it shares base and dependency layers with the ninety-nine before it. We'll formalize the build-cache payoff in the [build stage](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) and [Dockerfile](/blog/software-development/ci-cd/writing-a-production-dockerfile) posts; the kernel-level reason the cache *can* exist is this layered, content-addressed structure.

The ordering of layers in your Dockerfile is therefore a *cache-hit* decision, and it directly shapes delivery speed. Put the things that change rarely (base image, system packages, then language dependencies) low in the stack, and the thing that changes on every commit (your app code) high. Then a code-only change invalidates only the top layer; everything below is a cache hit. Invert that order — copy your source before installing dependencies — and every one-line code change re-installs the entire dependency tree, turning a 20-second rebuild into a 4-minute one. Same files, same app, 12× the build time, purely from layer order. The overlay filesystem is why that lever exists.

There is one gotcha worth stating plainly: **the writable layer is ephemeral.** When the container is removed, its top layer is discarded. Anything the app wrote to the container filesystem — uploaded files, a SQLite database, logs written to disk — is gone. That is not a bug; it is the point. The image is immutable; the container's state is disposable. Anything that must survive a restart goes into a **volume** (a mount, from the mount namespace, backed by host disk or a network volume) — explicitly outside the layered image. "Don't write important state to the container filesystem" is a first-principles consequence of copy-on-write, not an arbitrary rule.

## 5. The OCI image: a manifest, a config, and ordered layers

We now have the running mechanism. The packaging format that travels through your pipeline is the **image**, and since 2017 it has a vendor-neutral standard: the **OCI (Open Container Initiative) image spec**. "OCI" is what lets you build with Docker or BuildKit or Buildah or Kaniko, push to GHCR or ECR or Harbor, and run with containerd or CRI-O or Podman, all interoperably. It is the standard interface that makes "any registry, any runtime" true — and standard interfaces are what let a delivery toolchain be assembled from interchangeable parts.

An OCI image is not one file. It is a small graph of content-addressed blobs:

- A **manifest** — a small JSON document that lists, by digest, the config blob and the ordered layer blobs that make up the image for a given platform. The manifest itself has a digest; that digest *is* the image's identity.
- A **config** blob — JSON describing how to *run* the image: the default entrypoint and command, environment variables, working directory, exposed ports, the user to run as, and the ordered list of layer `diff_id`s with the build history.
- One or more **layer** blobs — gzip-compressed tarballs, each the filesystem diff of one build step. These are the read-only layers the overlay FS stacks.

![A branching graph showing an OCI manifest referencing a config blob and three layer tarballs, all content-addressed by digest and pulled from a registry to run identically across environments](/imgs/blogs/containers-from-first-principles-for-delivery-5.png)

Everything is **content-addressed by digest** — a `sha256:...` hash of the bytes. This is the property that makes the image a trustworthy delivery artifact, so it is worth being precise about what it buys you:

- **Immutability you can prove.** A tag like `myorg/api:1.4.2` is a mutable human label — someone can re-push a different image to the same tag tomorrow. A digest like `myorg/api@sha256:9b2c...e1` is the *content*. If even one byte of any layer or the config changes, the digest changes. So when your pipeline records and promotes a *digest*, "the artifact I tested" and "the artifact I shipped" are provably the same bytes. This is the kernel-of-truth under **build once, promote everywhere**: you don't rebuild per environment (which reintroduces drift); you promote one pinned digest from CI through staging to prod.
- **Portability.** The manifest and blobs are a standard wire format. Any OCI registry can store them; any OCI runtime can pull and run them. The image doesn't care what cloud or what runtime it lands on.
- **Cacheability.** Because each blob is named by its content hash, a registry stores each unique layer once and a node skips re-pulling layers it already has. Shared base layers cost nothing extra.

### Reading a real manifest

Here is a (trimmed) OCI image manifest as a registry serves it. You can pull your own with `docker buildx imagetools inspect --raw myorg/api:1.4.2` or `crane manifest`:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "digest": "sha256:7a1d...c0ff",
    "size": 1472
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:base11...aa",
      "size": 31518340
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:deps22...bb",
      "size": 41280091
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:app333...cc",
      "size": 204813
    }
  ]
}
```

Three layers — a base, a dependencies layer, a small app layer — each with its own digest and size, plus a tiny config blob the manifest points to. Notice the app layer is 200 KB while the dependencies layer is 41 MB: a code change ships 200 KB over the wire, because the 41 MB deps layer's digest is unchanged and already cached on the node. That is the layer cache paying off at deploy time, visible right in the manifest.

The config blob the manifest references looks like this (also trimmed):

```json
{
  "architecture": "amd64",
  "os": "linux",
  "config": {
    "User": "10001:10001",
    "Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "PORT=8080"],
    "Entrypoint": ["/app/api"],
    "ExposedPorts": { "8080/tcp": {} }
  },
  "rootfs": {
    "type": "layers",
    "diff_ids": ["sha256:b...", "sha256:d...", "sha256:a..."]
  },
  "history": [
    { "created_by": "FROM gcr.io/distroless/static" },
    { "created_by": "COPY deps" },
    { "created_by": "COPY --chown=10001 app /app/api" }
  ]
}
```

That config is the entire "how to run me" contract: run as user 10001 (non-root — section 8), entrypoint `/app/api`, port 8080. The runtime reads this, sets up the namespaces and cgroups, stacks the layers via overlay, and starts the process. No more, no less.

#### Worked example: pinning a digest to stop a deploy-drift bug

A concrete reason digests matter, not just a principle. A team deployed by tag: their pipeline pushed `myorg/api:latest`, and the Kubernetes manifest referenced `myorg/api:latest`. One Tuesday, a hotfix pushed a new `latest`. That evening, an unrelated node failure caused Kubernetes to reschedule a pod — which *re-pulled* `latest` and got the *new* hotfix image, not the one that had been running and was certified. Now the cluster ran a mix: old pods on the previously-pulled image, the rescheduled pod on the new one. Two versions of the same "deployment," silently, because a tag is a mutable pointer and "the image named `latest`" changed underneath them. Debugging took two hours because the manifest said `latest` and gave no clue which bytes were actually running.

The fix is one line of discipline, enforced by the mechanism:

```yaml
# Don't deploy by mutable tag. Deploy by immutable digest.
# Bad: image: myorg/api:latest          # pointer can move underneath you
# Good:
containers:
  - name: api
    image: myorg/api@sha256:9b2cf1e0a4d7e2c1b8f3a6d5e4c3b2a1908f7e6d5c4b3a2918f7e6d5c4b3a2e1
```

With a digest, a reschedule pulls the *exact same bytes*, every node runs the *same* image, and "which version is running?" has a single, provable answer. The build pipeline records the digest at build time and writes it into the manifest (often via a GitOps commit). Tags stay as human-friendly *aliases* for discoverability; the *deployment* references the digest. This is build-once-promote-everywhere made literal: the artifact you tested is, byte-for-byte, the artifact every node runs, because the digest *is* the bytes. Arithmetic of trust: a 32-byte hash collapses "are these two images the same?" from a fuzzy human question into a single equality check.

### Image vs container — the distinction that confuses everyone

These two words get used interchangeably and they are not the same thing, and the difference is exactly the difference between a frozen template and a running instance:

- An **image** is the immutable, on-disk, content-addressed *template*: read-only layers + config + manifest. It is not running. It is a file artifact in a registry or in your local store. You build images, push images, scan images, sign images, promote images.
- A **container** is a *running instance* of an image: the image's layers, plus a fresh writable layer on top, started as a namespaced + cgrouped process. It has state, a lifecycle, a PID. You run containers, stop containers, kill containers, exec into containers.

The relationship is class-to-object, recipe-to-meal, template-to-instance. One image spawns many containers, each with its own throwaway writable layer and its own set of namespaces, all sharing the same read-only layers underneath. When delivery people say "the artifact flows through the pipeline," they mean the *image* (by digest). The *containers* are spun up and torn down constantly at the operate stage; they are cattle. The image is the thing you version and promote.

### What `docker run` actually does

With all the pieces named, "run a container from an image" decomposes into concrete steps the runtime performs:

```bash
docker run \
  --name api \
  --memory=512m --cpus=1.5 \          # cgroups: memory + CPU limits
  --pid=private --network=bridge \    # namespaces: own PID tree, own net stack
  --user 10001:10001 \                # run as non-root (matches the image config)
  --read-only \                       # mount the rootfs read-only (harden the FS)
  --tmpfs /tmp \                       # a writable scratch mount where needed
  -p 8080:8080 \                      # publish container port to host
  myorg/api@sha256:9b2c...e1          # the IMAGE, pinned by DIGEST not tag
```

Step by step, the runtime:

1. **Resolves the image** by digest, pulling any layers the node doesn't already have (cache hit on shared layers).
2. **Creates a new set of namespaces** — a fresh PID, net, mount, UTS, IPC namespace (and a user namespace if configured) — giving the soon-to-start process its restricted view.
3. **Stacks the image's read-only layers** via overlay and adds the thin writable layer, mounting that union as the new root in the mount namespace.
4. **Creates a cgroup**, puts the process into it, and writes the limits (512 MB memory ceiling, 1.5-CPU quota).
5. **Wires the network** — a `veth` pair, the container end becoming `eth0` in the net namespace, the host end on a bridge, with a port-forward rule for `8080`.
6. **Executes the entrypoint** from the image config as PID 1 inside the new PID namespace, running as user 10001.

That is the whole magic trick. A namespaced, cgrouped, overlay-rooted process. Notice I pinned the image by **digest**, not tag — for anything past your laptop, promote and deploy by digest so the bytes are provably the ones you tested.

## 6. Why containers won for delivery

We can now answer the question the whole series cares about: *why did the container, specifically, become the unit of delivery?* Not "why is it cool" — why did it win the delivery problem over shipping VMs, shipping tarballs, or shipping "a deploy script that installs things on the box."

The answer is five properties, and every one of them traces directly to a mechanism we just built up.

**1. Dependency isolation — the death of "works on my machine."** The image carries the app *and its entire userland*: the right Python or Node version, the exact library versions, the system packages, the locale data — everything except the kernel. The classic deploy failure was environment drift: the box had Python 3.9, the dev laptop had 3.11; the box's `libssl` was a patch behind; a transitive C library was missing. The container ends that class of bug by *shipping the dependencies inside the artifact*. The runtime environment is the image plus a clean namespace set, identical everywhere.

**2. The immutable artifact — build once, promote everywhere.** Because the image is content-addressed and immutable, you can build it exactly once, in CI, then promote that *same digest* through staging into production with zero rebuilds. No "we rebuilt for prod and a dependency floated to a new version." The bytes you tested are the bytes you ship. This is the principle the whole series rests on, and the image's digest is what makes it enforceable.

![A before-and-after comparison contrasting a 1.9 GB virtual-machine tarball rebuilt per environment against a 75 MB OCI image built once and promoted by digest across all environments](/imgs/blogs/containers-from-first-principles-for-delivery-6.png)

**3. Fast start — it's a process, not a booting OS.** No guest kernel to boot. Starting a container is starting a process with extra flags, measured in milliseconds. A delivery system that scales replicas up under load, drains and replaces pods during a rolling update, or restarts a crashed instance, leans entirely on this. Slow starts make rolling deploys slow and autoscaling sluggish; container start speed is a delivery-latency property.

**4. Density — share the one kernel.** Hundreds of containers per node because they share the kernel and (via overlay) share read-only layers in page cache. Density is delivery economics: more workloads per node is a smaller bill, and a smaller scheduling unit is easier to pack and bin into a cluster.

**5. The standard interface — OCI.** Any registry, any runtime, any cloud. The OCI image and runtime specs mean your toolchain is built from interchangeable parts: swap GitHub Actions for GitLab CI, GHCR for ECR, Docker for containerd, and the artifact in the middle is unchanged. Standard interfaces are what let a delivery pipeline be assembled, debugged, and replaced piece by piece without a rewrite.

#### Worked example: 1.9 GB tarball-of-a-VM vs a 75 MB image

Make the contrast concrete with a real, common before→after. A team had a Python service they shipped as a VM image: spin up an Ubuntu VM, `apt install` a pile of build tools and runtime libraries, `pip install` the app, snapshot the VM, ship the snapshot. The snapshot was **1.9 GB**. It worked — until it didn't, because the snapshot drifted (someone `apt upgrade`'d the base before re-snapshotting), and because moving 1.9 GB per deploy was slow and the "rebuild for prod" step re-resolved dependencies and occasionally pulled a newer patch.

They rebuilt it as a multi-stage OCI image. (The Dockerfile craft is its own [post](/blog/software-development/ci-cd/writing-a-production-dockerfile); here we care about the *result* and *why* the mechanism makes it small.)

```dockerfile
# Stage 1: a fat builder with compilers and dev headers — thrown away at the end.
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
# Build wheels (may need gcc); install into a clean target dir we can copy out.
RUN pip install --no-cache-dir --target=/install -r requirements.txt
COPY src/ ./src/

# Stage 2: a tiny runtime image — no compilers, no shell, no package manager.
FROM gcr.io/distroless/python3-debian12
WORKDIR /app
COPY --from=builder /install /usr/lib/python3/dist-packages
COPY --from=builder /app/src ./src
USER 10001
ENTRYPOINT ["python", "-m", "src.main"]
```

The layer breakdown of the result, readable with `docker history`:

| Layer | Source | Size | Why |
|---|---|---|---|
| `distroless/python3` base | runtime base | ~50 MB | Python + libc, **no** shell/apt/coreutils |
| Python dependencies | `pip install` | ~24 MB | the app's third-party wheels |
| App source | `COPY src` | ~1 MB | the actual code |
| **Total** | | **~75 MB** | vs 1.9 GB before |

That is a **25× shrink**, and the reasons are exactly the mechanisms we built:

- **The build tools never ship.** Stage 1 had `gcc`, dev headers, build caches — hundreds of MB. The multi-stage build copies only the *artifacts* (compiled wheels, source) into stage 2, and stage 1 is discarded. The 1.9 GB VM carried all that build cruft forever because a VM can't separate "what I needed to build" from "what I need to run."
- **The base is distroless.** No shell, no package manager, no coreutils, no man pages, no full OS — just the language runtime and libc. That's ~50 MB instead of a ~700 MB Ubuntu base. Smaller base = smaller attack surface too (section 8), and `nsenter` from the host means you don't lose debuggability.
- **The layers are shared.** That 50 MB distroless base and the 24 MB deps layer are shared, by digest, with every other Python service on the node and across deploys. The *marginal* cost of a code-only deploy is the ~1 MB app layer.

The delivery payoff compounds:

| Metric | 1.9 GB VM tarball | 75 MB OCI image |
|---|---|---|
| Artifact size | 1.9 GB | 75 MB (25× smaller) |
| Deploy data moved (code change) | full 1.9 GB | ~1 MB (app layer only) |
| Cold pull on a fresh node | minutes | seconds (and shared base) |
| Rebuild per environment? | yes — drift | no — promote one digest |
| "Works on my machine"? | recurring | gone |
| Start time | ~30 s VM boot | <1 s process start |

You don't have to obsess over 75 MB versus 90 MB — past a point the returns flatten and a debuggable base is worth a few MB. The point of the example is the *kind* of difference and *why the mechanism produces it*: the VM ships a whole computer; the image ships a layered, deduplicated, immutable process bundle.

### The cache-hit economics of layers, quantified

There's a quantifiable delivery payoff hiding in "shared layers," and it's worth doing the arithmetic because it explains why container-based delivery scales to hundreds of services without the storage and bandwidth bill exploding. Suppose you run 50 Python microservices, each as its own image. If every image were a self-contained 1.9 GB VM tarball with no sharing, your registry stores $50 \times 1.9\text{ GB} = 95\text{ GB}$, and a fresh node pulling all 50 moves 95 GB over the network.

Now layer them properly: a shared 50 MB distroless base, a per-service dependency layer averaging 24 MB, and a tiny app layer averaging 1 MB. The base is stored *once* (content-addressed — same digest, same bytes). Registry storage becomes roughly $50\text{ MB} + 50 \times (24 + 1)\text{ MB} = 50 + 1250 = 1.3\text{ GB}$ — a **73× reduction** in stored bytes versus the no-sharing case. A node that already has the base and a few common dependency layers (Flask, requests, the usual suspects) pulls only the deltas. And the *deploy* cost of a one-line code change is the ~1 MB app layer, because every layer below it is a cache hit by digest. The cache-hit rate $h$ directly scales deploy data: bytes moved $\approx (1 - h) \times \text{image size}$, and for a code-only change against a warm node, $h$ is near 1 and the bytes moved are tiny. That is the difference between a deploy that takes seconds and one that takes minutes, multiplied across every deploy of every service every day — and it falls straight out of content-addressed, shared, read-only layers. The mechanism isn't a nicety; it's the thing that makes fleet-scale container delivery economically sane.

## 7. Containers vs VMs, precisely — and when you actually want stronger isolation

We've leaned on "shared kernel vs guest OS" throughout. Let's make the trade-off explicit, because choosing between them (and choosing the sandboxed middle ground) is a real delivery and platform decision.

![A before-and-after comparison contrasting a virtual machine running a full guest operating system on a hypervisor against a container running as a process on the shared host kernel](/imgs/blogs/containers-from-first-principles-for-delivery-3.png)

| Dimension | Virtual machine | Container |
|---|---|---|
| Kernel | Its own guest kernel | Shares the host kernel |
| Isolated by | Virtualized hardware (hypervisor) | Namespaces + cgroups (kernel features) |
| Boot/start time | Seconds to tens of seconds | Milliseconds to ~1 second |
| Overhead per instance | A full guest OS in RAM (~0.5–1 GB) | Just the app's footprint |
| Density per host | Tens | Hundreds to thousands |
| Image size | Gigabytes (full OS) | Tens to hundreds of MB (layered) |
| Isolation strength | Strong (separate kernels, hardware boundary) | Weaker by default (one shared kernel) |
| Right when | Hostile multi-tenant, kernel-level isolation, different guest OSes | Your own trusted services, fast scaling, dense packing |

The container's *every* advantage — speed, density, small images — comes from the one shared kernel. And the container's *one* real disadvantage — weaker isolation — comes from the exact same fact. There is no free lunch; it is one trade-off seen from two sides.

For most internal delivery — your own services, your own code, your own cluster — the container's isolation is *good enough*, and the speed/density/cost wins are decisive. That is why containers won the mainstream delivery problem. But there are workloads where "good enough by default" isn't: running **untrusted code** (CI runners executing arbitrary user-submitted builds, a multi-tenant SaaS running customers' code, serverless functions from strangers). For those, the industry built a middle ground that keeps the container *interface* but hardens the *isolation*:

![A decision matrix comparing plain containers, gVisor, and Kata or Firecracker micro-VMs across kernel sharing, isolation strength, start time, and best fit](/imgs/blogs/containers-from-first-principles-for-delivery-8.png)

- **gVisor** (Google) puts a user-space kernel between the container and the host kernel. The container's syscalls hit gVisor's reimplementation of the Linux syscall surface instead of the host kernel directly, dramatically shrinking the host-kernel attack surface. Cost: some syscall-heavy and I/O-heavy workloads run slower.
- **Kata Containers** runs each container (or pod) inside a lightweight, hardware-virtualized **micro-VM** with its own minimal guest kernel — but presents the standard OCI/CRI interface, so your pipeline and orchestrator don't know the difference. You get VM-grade isolation with much of the container ergonomics.
- **Firecracker** (AWS) is the micro-VM monitor behind Lambda and Fargate: it boots a stripped micro-VM in ~125 ms with a tiny memory footprint, giving per-tenant hardware isolation at near-container start speed. It's the proof that "VM-strength isolation" and "fast, dense" aren't strictly opposed if you strip the VM down far enough.

The decision rule for delivery: **plain containers for code you trust; sandboxed runtimes for code you don't.** If you're a platform team building a CI service that runs arbitrary customer builds, a plain shared-kernel container is not a sufficient boundary — reach for gVisor or a micro-VM. If you're shipping your own ten microservices to your own cluster, plain containers are right and the sandbox overhead isn't worth it. The reliability and blast-radius reasoning around *which* workloads need which isolation is an SRE topic — see [designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) for thinking about blast radius and trust boundaries.

## 8. The security caveat: a container is not a strong boundary by default

This is the part teams skip and regret, so I'll be blunt: **a default container is not a security boundary you should bet on against hostile code.** It is a convenience and an isolation *of accident* — it stops your services from stepping on each other's ports and files. It does *not*, by itself, stop a determined attacker who has code execution inside the container from attacking the host.

The reason is the thesis of this whole post: one shared kernel. A namespace is a *restricted view*, enforced by the same kernel the attacker's code is calling into. If there is a kernel vulnerability — and the kernel is a huge, evolving attack surface — a process inside a container can sometimes exploit it to break out into the host or into other containers. A VM's separate guest kernel is a much harder wall. With containers, "isolation" leans heavily on the kernel having no exploitable bug, plus the specific hardening you applied.

The single sharpest issue is **root in the container.** By default, the process inside a container runs as **UID 0 — root** — and, unless you've enabled user namespaces (off by default in plain Docker), that is the *host's* root, just with a restricted view. So a container compromise of a root process is a much shorter hop to host compromise. Concretely, default-root containers plus a careless `-v /var/run/docker.sock:/var/run/docker.sock` or `--privileged` flag is a near-instant host takeover, because the attacker can drive the Docker daemon (which is root) to mount the host filesystem.

The hardening that turns the convenience into something closer to a boundary — all of which belong in your delivery pipeline as defaults, not afterthoughts:

```dockerfile
# In the Dockerfile: never ship root. Create and switch to a non-root user.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER 10001:10001
```

```yaml
# In the Kubernetes manifest: enforce non-root + drop privileges at the runtime.
securityContext:
  runAsNonRoot: true            # refuse to start if the image runs as root
  runAsUser: 10001
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true  # the rootfs is read-only; writes go to mounted volumes
  capabilities:
    drop: ["ALL"]               # drop all Linux capabilities, add back only what's needed
  seccompProfile:
    type: RuntimeDefault        # filter the syscall surface the container can reach
```

The principles, each a direct consequence of the mechanism:

- **Run as non-root.** Drop root inside the container so a compromise lands on an unprivileged user, not host root. Build it into the image (`USER`) *and* enforce it at the runtime (`runAsNonRoot`).
- **Drop capabilities.** Linux splits root's power into ~40 capabilities; a typical service needs almost none. Drop `ALL`, add back the one or two you truly need.
- **Read-only root filesystem.** The image is immutable anyway; make the running rootfs read-only and route the few writable paths to explicit `tmpfs`/volume mounts. An attacker can't drop a binary into a read-only filesystem.
- **`seccomp` and the syscall surface.** A seccomp profile filters which syscalls the container may make, shrinking the kernel attack surface — fewer syscalls reachable, fewer kernel bugs reachable.
- **Minimal base image.** A distroless or scratch base has no shell, no `curl`, no package manager — so an attacker who lands inside has no tools to pivot with, and there are far fewer packages to carry CVEs. This is also where **image scanning** earns its place in the pipeline: the image is a *supply-chain entry point*, and you want a Trivy/Grype gate failing the build on a critical CVE in a base layer. That's the subject of the planned [image security scanning and a minimal attack surface](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface) post — scan the image, minimize the base, run non-root, and the container becomes a meaningfully harder target.
- **For genuinely hostile code, use a real sandbox** (section 7) — don't pretend a plain container is a VM.

The honest framing for your delivery platform: a container's default isolation is a *fence*, not a *vault*. It keeps honest neighbors from wandering in. For hostile inputs you add hardening (non-root, dropped caps, read-only, seccomp, scanning) and, past a threshold of distrust, a sandboxed runtime. Treating the default container as a security boundary is the mistake; treating it as a convenience you harden deliberately is the practice.

## 9. The runtime stack, briefly: containerd, runc, and the CRI

You don't need to know the runtime internals to ship, but a one-paragraph map keeps you from being mystified, and it draws the line to where orchestration takes over (which this post deliberately does *not* re-derive — that's the [Kubernetes-for-delivery](/blog/software-development/ci-cd/kubernetes-for-delivery-the-objects-that-matter) post).

When something asks for a container, the request flows down a small stack of layers, each doing less and standing closer to the kernel:

- **runc** is the **low-level runtime**: a small binary that does exactly what section 6 described — set up namespaces and cgroups, mount the overlay rootfs, and `exec` the process. It implements the **OCI runtime spec**. It is the thing that actually turns a bundle (config + rootfs) into a running, namespaced, cgrouped process. gVisor's `runsc` and Kata's runtime are *drop-in replacements* for runc at exactly this layer — same interface, stronger isolation.
- **containerd** is the **high-level runtime / daemon**: it manages images (pull, store, unpack layers), manages container lifecycle, handles networking hooks and storage, and calls runc to do the actual create/start. Docker uses containerd internally; so does Kubernetes on most clusters.
- The **CRI (Container Runtime Interface)** is the gRPC contract Kubernetes uses to talk to a container runtime. The kubelet on each node speaks CRI to containerd (via the CRI plugin) or to CRI-O. This is the standard seam that lets Kubernetes be runtime-agnostic — and the reason "Kubernetes dropped Docker support" was a non-event for delivery: Kubernetes always talked to containerd under Docker, so it just talks to containerd directly now.

The clean mental picture: **Kubernetes/kubelet → (CRI) → containerd → runc → namespaced + cgrouped process on the kernel.** Everything above containerd is orchestration — scheduling, desired-state reconciliation, the objects that matter — and that's a different post and a different layer of this series. Everything from containerd down is "make this image into that restricted process," which is what this post is about. Knowing where the seam is means that when a pod won't start, you can reason about *which* layer is the problem: scheduling (orchestration), image pull (containerd), or container create (runc + kernel — a bad cgroup limit, a missing namespace permission, a seccomp denial).

## 10. Stress-testing the model: edge cases that prove you understand it

A model is only useful if it survives the awkward cases. Let's pose the ones that come up in real delivery and answer each from first principles, because being able to *predict* these from the mechanism is the difference between knowing containers and reciting commands.

**"My container shows the host's CPU count / RAM and sizes thread pools or heaps wrong."** Predicted directly from section 3: cgroups *limit and account*, but historically didn't fully *namespace the view* of `/proc`. So a runtime that auto-sizes from "available CPUs" or "available RAM" may read the host. Fix: make the runtime cgroup-aware (`-XX:MaxRAMPercentage`, `GOMEMLIMIT`, set worker counts explicitly) or run with tools that present cgroup-correct values. This is the generalization of the JVM OOM story to every auto-sizing runtime.

**"Writes inside my container vanished on restart."** Predicted from section 4: the writable layer is copy-on-write and *ephemeral* — discarded when the container is removed. State that must survive goes in a volume, explicitly outside the image. If you're surprised by this, you were treating the container like a VM with a persistent disk.

**"My container takes 10 seconds to stop and then gets force-killed."** Predicted from section 2: your app is PID 1 in its PID namespace, and PID 1 has special signal duties. If your entrypoint is a shell script that doesn't forward `SIGTERM`, or your app ignores it, the runtime waits out the grace period and then `SIGKILL`s. Fix: handle `SIGTERM`, or use a tiny init (`tini`) as PID 1 to forward signals and reap zombies.

**"Two containers both bind 8080 — why no conflict? But two on one host port *do* conflict."** Predicted from section 2: each container has its own net namespace, so port 8080 *inside* each is independent. The conflict only appears when you publish both to the *same host* port — that's the single shared host net namespace, not the containers'.

**"A kernel CVE dropped — am I exposed across all my containers?"** Predicted from the thesis: one shared kernel means a kernel vulnerability is a shared blast radius across every container on the node. This is precisely the case where, for untrusted workloads, you'd have wanted gVisor or a micro-VM (section 7), and where, for trusted workloads, you patch the *host* kernel and roll nodes. A VM's separate guest kernel would have contained it.

**"My image is huge and I can't tell why."** Predicted from sections 4–6: a fat base image, build tools that shipped into the runtime layer (no multi-stage), and cache-busting layer order. `docker history <image>` shows the per-layer sizes and the command that produced each; the giant layer is usually an `apt install` of build tools that should have lived in a discarded builder stage, or a `COPY . .` that dragged in `.git` and `node_modules` for lack of a `.dockerignore`.

Each of these is *derivable* from "namespaced + cgrouped + overlay-rooted process on a shared kernel, packaged as an immutable content-addressed image." That's the test of the model: you don't memorize the symptoms, you predict them.

## 11. War story: the container that thought it owned the box

A composite of incidents I've seen and that are well documented in the wild, because the failure mode is so common it's practically a genre.

A data team ran a batch job in a container on a shared Kubernetes node. The job processed files and, under a large input, allocated aggressively. They'd set no memory limit on the pod — "it's our own node, let it use what it needs." On a node packed with other teams' pods, the batch container's allocation climbed past the *node's* available memory. With no per-container limit, the kernel's *node-level* OOM killer woke up and started killing processes to reclaim memory — and its victim-selection didn't politely pick the batch job. It killed pods belonging to *other* teams, including a latency-sensitive API, which crash-looped and paged its on-call. One unbounded container took down neighbors it had never heard of.

The first-principles diagnosis: a container without a memory cgroup limit is *not* contained on the memory axis — it can consume up to the whole node, and when the node runs out, the kernel OOM killer operates at node scope and picks victims by heuristic, not by "whose fault was it." The namespaces gave the batch job its own *view*; the missing cgroup meant it had no *limit*. Isolation of view without limitation of use is not isolation in the way that matters under contention.

The fix was three lines of policy, all cgroup-backed:

```yaml
# Every pod gets a memory limit (arms the per-container OOM killer, protects neighbors)
# and a request (so the scheduler reserves room and doesn't overpack the node).
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"     # this container OOM-kills itself at 1Gi, before it can starve the node
```

…plus a `LimitRange` in the namespace so a pod that *forgets* to set limits gets a default instead of running unbounded, and a `ResourceQuota` so a team can't over-commit the cluster. The deeper lesson is the one the JVM story made the first time: **the cgroup limit is a contract that protects the whole node, and "it's our own box" is exactly when you most need it**, because the blast radius of an unbounded container is every neighbor sharing the kernel. The reliability framing of bounding blast radius and protecting neighbors is SRE territory — see [designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) — but the *mechanism* that lets you bound it is the humble cgroup.

A second, broader war story worth naming because it's about the *image* as a delivery liability: the era of the "everything image." Teams would build one giant base image — full Ubuntu, every language runtime, every internal tool, `git`, `curl`, `vim`, build chains — and base everything on it "for convenience." The result: multi-gigabyte images, slow pulls that made deploys and autoscaling sluggish, and a sprawling attack surface where a CVE in *any* of the dozens of bundled packages flagged *every* service. The fix was the lesson of section 6: a minimal, distroless base; multi-stage builds so build tools never reach the runtime layer; one purpose per image. Image size and image contents are *delivery* and *security* properties, not cosmetic ones — the bloated image is slow to ship, slow to start, and broad to attack.

## 12. How to reach for this (and when not to)

Containers are the default unit of delivery for good reasons, but "default" is not "always," and a principal engineer's job is to know the edges.

**Reach for containers when:**

- You're shipping your own services to your own infrastructure and want one immutable artifact promoted across environments — the build-once-promote-everywhere sweet spot.
- You need fast start and dense packing (autoscaling, rolling deploys, many small services).
- You want a standard interface so your toolchain (CI, registry, runtime, orchestrator) is built from interchangeable parts.
- You want to kill "works on my machine" by shipping dependencies inside the artifact.

**Be cautious / reach for more than a plain container when:**

- **You're running untrusted code.** A plain container is not a sufficient boundary against hostile input. Use gVisor, Kata, or Firecracker — or full VMs — when you run code you don't control. Don't learn this from an incident.
- **The workload is fundamentally stateful and demanding** (a primary database with strict latency and durability needs). You *can* run it in a container, but the operational nuances (storage, failover) are real; see [running stateful systems reliably](/blog/software-development/site-reliability-engineering/running-stateful-systems-reliably). The container doesn't make stateful operations easy by itself.
- **You need a different guest OS or kernel-level features** the host kernel can't provide. That's a VM, not a container.

**Don't over-engineer when:**

- **You're a tiny team shipping one app.** You may not need to hand-craft Dockerfiles, multi-stage builds, and a registry strategy on day one — a PaaS (Fly, Render, Cloud Run, a Heroku-like) often builds the container *for* you from a buildpack and handles the rest. Adopt the container mechanics deliberately as your scale and team grow. The thesis of *this* post is to understand the mechanism so the PaaS isn't magic; it's not a mandate to operate the whole stack yourself before you need to.
- **You're chasing the last 10 MB of image size at the cost of debuggability.** Past a sane minimal base, the returns flatten. A slightly larger base that you can reason about (and reach with `nsenter` from the host) beats a heroic scratch image you can't operate. Optimize image size when it actually hurts deploy latency or attack surface — measure first.
- **You haven't fixed your build order yet.** If your rebuilds are slow, the first lever is Dockerfile layer ordering and the build cache, not exotic tooling. Cache before you shard. (Covered in the [build stage](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) post.)

The meta-rule: containers are a *means* — the unit that makes build-once-promote-everywhere and everything-as-code practical. Adopt the depth you need, understand the mechanism so nothing downstream is magic, and don't pay for isolation or optimization you don't have a workload to justify.

## 13. Key takeaways

- **A container is a normal Linux process with a restricted view of the world, on the shared host kernel** — not a tiny VM. Every property follows from that one fact.
- **Three kernel features do the work:** namespaces (what it can *see* — PID, net, mount, UTS, IPC, user), cgroups (what it can *use* — CPU, memory, I/O, and the OOM kill at the limit), and the overlay filesystem (layered, copy-on-write, shareable, cacheable).
- **Set memory limits and make your runtime cgroup-aware.** The classic crash-loop is a runtime that reads the host's RAM instead of its cgroup limit and gets OOM-killed at exit 137; fix it with `MaxRAMPercentage`, `GOMEMLIMIT`, or explicit caps.
- **The image is the unit of delivery — promote it by digest.** An OCI image is a manifest + config + ordered content-addressed layers; the `sha256` digest makes it immutable, portable, and cacheable, which is what makes build-once-promote-everywhere enforceable.
- **Image vs container is template vs instance:** one immutable image spawns many disposable, namespaced, cgrouped containers, each with its own throwaway writable layer.
- **Containers won delivery on five mechanism-backed properties:** dependency isolation, the immutable artifact, fast start, density, and the standard OCI interface.
- **A container is not a strong security boundary by default.** One shared kernel, root-in-container by default. Run non-root, drop capabilities, use a read-only rootfs and seccomp, scan the image, minimize the base — and use a sandbox (gVisor/Kata/Firecracker) for genuinely untrusted code.
- **Small, minimal, multi-stage images are a delivery and security property,** not cosmetics: a 75 MB distroless image deploys faster, packs denser, and presents a smaller attack surface than a 1.9 GB tarball-of-a-VM.
- **Know where the seam is:** Kubernetes → CRI → containerd → runc → namespaced process. From containerd down is "make this image a restricted process"; above it is orchestration — a different layer, and a different post.

## 14. Further reading

- **The intro map for this series** — [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model): the commit→build→test→package→deploy→operate spine, the two principles, and the four DORA metrics this post plugs into.
- **The sibling on Dockerfile craft** — [Writing a production Dockerfile](/blog/software-development/ci-cd/writing-a-production-dockerfile): multi-stage builds, layer-cache ordering, distroless/minimal bases, non-root, and the `.dockerignore` that kills image bloat.
- **The sibling on the image as a supply-chain entry point** — [Image security scanning and a minimal attack surface](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface): Trivy/Grype as a build gate, base-image patching, and shrinking what an attacker can reach.
- **Where orchestration takes over** — [Kubernetes for delivery: the objects that matter](/blog/software-development/ci-cd/kubernetes-for-delivery-the-objects-that-matter) and the microservices view in [Kubernetes for microservices: the essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials).
- **The reliability lens on isolation and blast radius** — [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) and [running stateful systems reliably](/blog/software-development/site-reliability-engineering/running-stateful-systems-reliably).
- **The OCI specs** — the Open Container Initiative image-spec and runtime-spec: the authoritative definition of the manifest, config, layers, and what runc actually does.
- **The Twelve-Factor App** — for the build/release/run separation and "store config in the environment," which the immutable image embodies.
- **The Linux man pages for `namespaces(7)`, `cgroups(7)`, `unshare(1)`, and `nsenter(1)`** — the primary sources for the kernel features this post is built on; read them once and the mechanism is no longer abstract.

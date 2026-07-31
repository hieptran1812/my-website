---
title: "AgentENV: Building a Distributed Sandbox Runtime for Long-Running AI Agents"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "A systems-level analysis of AgentENV's Firecracker sandboxes, snapshot lifecycle, copy-on-write storage, multi-node control plane, and the security gaps teams must close before production."
tags: ["ai-agents", "agentenv", "sandboxing", "firecracker", "microvm", "distributed-systems", "agent-infrastructure", "security", "machine-learning"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

An agent that can only call three typed APIs is an application feature. An agent that can write files, install packages, run shell commands, open network connections, and keep state across a long task is a small operating-system workload.

That distinction is where most agent platforms become uncomfortable. The model is not the hard part anymore. The hard part is creating thousands of isolated Linux environments quickly, releasing them when they go idle, preserving the exact state that a task needs, and still knowing which machine owns each environment when the cluster grows.

AgentENV is an open-source attempt to make that substrate explicit. It runs agent environments inside Firecracker microVMs, exposes an E2B-compatible API, uses overlaybd and ublk for layered block storage, and treats snapshots as the primitive from which templates, pauses, resumes, and forks are built. The repository says the system targets resume and boot in under 50 ms, pause in under 100 ms, and incremental snapshotting in under 100 ms. Those are ambitious numbers, but the more interesting contribution is architectural: AgentENV makes the execution environment a resumable distributed-system object instead of a disposable container.

![AgentENV system architecture: client requests pass through the API and orchestrator into Firecracker, while snapshots and layered storage supply the runtime state](/imgs/blogs/agentenv-distributed-sandbox-runtime-1.webp)

The diagram above is the mental model: a request crosses a control boundary, an isolation boundary, a guest-execution boundary, and a storage boundary before an agent command runs. Once we see those boundaries, the repository stops looking like a collection of Rust modules and starts looking like a coherent answer to one operational question:

> How do we make arbitrary agent execution feel as cheap and repeatable as a function call without pretending that arbitrary code is a function call?

This article is an analysis of that answer. It follows the implementation and documentation in the [AgentENV repository](https://github.com/kvcache-ai/AgentENV) and the [official documentation](https://kvcache-ai.github.io/AgentENV/). Where the project documents a guarantee or a current limitation, I call it out as such. Where I propose an operational interpretation, I label it as an engineering inference rather than a feature claim.

## 1. Why ordinary containers are not enough for agent workloads

The first design decision is not about Rust, Firecracker, or a faster image loader. It is about the threat model.

When a service runs a fixed binary, the service author knows which syscalls, files, libraries, and network destinations matter. A container is often a reasonable boundary for that workload. The process can be restricted with namespaces, cgroups, seccomp, read-only mounts, and a narrow set of capabilities.

An agent running generated code is different. The program is selected at runtime. It may create child processes, compile native extensions, inspect the filesystem, spawn a local server, or interpret text from an untrusted document as an instruction. The action space is not a small API schema; it is whatever the guest operating system allows.

That does not make containers useless. It means that “containerized” is not the same statement as “isolated from every other tenant.” A container normally shares the host kernel. A kernel vulnerability, a privileged capability, a dangerous device mount, or a bad runtime configuration can turn a process-level boundary into a host-level incident.

![Containers versus AgentENV microVMs: the microVM adds an independent kernel boundary but requires stronger host prerequisites](/imgs/blogs/agentenv-distributed-sandbox-runtime-2.webp)

The useful comparison is not “containers are insecure and microVMs are secure.” It is that the two boundaries fail differently.

| Dimension | Container-oriented execution | AgentENV-style microVM execution |
| --- | --- | --- |
| Kernel | Shared with the host | Independent guest kernel per sandbox |
| Startup | Usually very fast | Designed for fast restore, with more host setup |
| Filesystem | Namespaces and mounts | Guest root filesystem plus attached block devices |
| Network | Network namespace | Guest network stack and namespace |
| Host dependency | Container runtime | Linux 6.8+, `/dev/kvm`, Firecracker support |
| Primary risk | Misconfiguration or kernel escape | API exposure, guest escape, host privilege, secrets |
| Best fit | Trusted application processes | Arbitrary code and tool execution |

The microVM is valuable because it puts a kernel boundary between the agent and the host. It is not free. The quick start requires Linux kernel 6.8 or later and access to `/dev/kvm`; the Docker deployment is privileged and mounts `/dev` into the server container. That is a meaningful operational tradeoff. A platform team must be able to secure the host before it can benefit from securing the guest.

The right question is therefore: what does the workload need to be allowed to do? If the answer is “call a handful of HTTP APIs,” a microVM may be needless complexity. If the answer is “run a model-generated repository repair script with arbitrary dependencies and a potentially hostile test suite,” the stronger boundary is easier to justify.

### The agent changes the unit of isolation

In a conventional deployment, the unit of isolation is often the service instance. In an agent platform, it is the task environment. Each task may require a different base image, a different set of packages, a different network policy, and a different persistence lifetime.

That creates three scaling pressures at once:

1. **Environment diversity.** Different tasks need different OCI images and tools.
2. **Environment churn.** Tasks start, pause, resume, fork, and disappear unpredictably.
3. **Environment state.** A useful session contains files, processes, installed packages, and intermediate results that should not be reconstructed after every request.

AgentENV addresses all three with on-demand image loading, snapshot-backed lifecycle operations, and copy-on-write storage. The rest of the design is an elaboration of those choices.

## 2. The AgentENV mental model

The central mistake when reading a sandbox runtime is to begin with its API endpoints. APIs show nouns and verbs, but not ownership. AgentENV becomes easier to reason about if we start with the path of a single command.

A client sends an HTTP request to the API server. The API validates the request and hands lifecycle work to the orchestrator. The orchestrator creates or restores a Firecracker VM, attaches block devices, configures networking, and starts the guest-side `envd` daemon. The daemon executes commands and reports health. A reverse proxy can route HTTP and WebSocket traffic to services inside the guest. The storage stack resolves the root filesystem and memory snapshot from local or remote layers.

The API is therefore not the runtime. It is the front door to a lifecycle controller whose job is to make a VM and its artifacts agree about state.

| Component | Responsibility | Failure if misunderstood |
| --- | --- | --- |
| API server | Exposes sandbox, template, snapshot, and proxy operations | Clients depend on an interface that does not describe durability |
| Orchestrator | Owns lifecycle transitions and resource setup | Two actors race to pause, resume, or delete the same sandbox |
| Firecracker | Provides the guest kernel and VM boundary | The host assumes process isolation is enough for arbitrary code |
| `envd` | Executes commands and manages guest-side processes | A running VM exists but cannot perform useful agent work |
| overlaybd + ublk | Presents layered storage as block devices | Image diversity consumes all node-local disk |
| Snapshot manager | Resolves and commits memory/filesystem state | Resume restores only part of the environment |
| Reverse proxy | Routes HTTP and WebSocket traffic into guests | Services run correctly but are unreachable to the agent client |

This separation also explains why the project contains both a Rust runtime and Go gateway/scheduler services. The runtime node is close to the kernel, storage, and Firecracker. The control plane is responsible for placement and ownership across nodes. They have different failure domains and different useful abstractions.

### A control plane with a stateful data plane

The control plane creates intent: start this template, pause that sandbox, route this sandbox ID to its owner. The data plane carries the consequences: VM processes, block-device reads, guest network traffic, and command output.

This is a familiar distributed-systems pattern, but agents make the state especially valuable. A stateless web request can be retried at another node. A live sandbox cannot be moved by simply replaying `POST /sandboxes/{id}/exec`. It contains processes, open files, filesystem mutations, and possibly an in-progress compilation.

The design consequence is important:

> Placement is not just where a request runs; placement is part of the identity of a live sandbox.

That is why the multi-node documentation routes existing sandbox IDs through a scheduler lookup instead of treating every runtime node as interchangeable.

## 3. Sandbox lifecycle as a state machine

Agents do not have a clean request boundary. A coding task can run tests, pause while waiting for a review, resume with its process tree intact, and fork a second attempt after a failed patch. If the runtime models all of that as “container exists” or “container does not exist,” operators eventually lose track of what resources are safe to reclaim.

![Sandbox lifecycle: creating, running, pausing, resuming, snapshotting, forking, and killing represent explicit operational states](/imgs/blogs/agentenv-distributed-sandbox-runtime-3.webp)

The documented lifecycle includes these states:

| State | Meaning | Resource implication |
| --- | --- | --- |
| Creating | VM boot, devices, and networking are being prepared | CPU and setup resources are active |
| Running | Commands, proxy traffic, and TTL are active | Guest consumes runtime resources |
| Pausing | Memory and disk state are being captured | Resource release is not yet complete |
| Paused | VM is stopped and state is persisted | Active CPU and memory can be reclaimed |
| Resuming | Snapshot artifacts are restored into a runnable VM | Storage and startup work are active |
| Snapshotting | A persistent checkpoint is being created | Source sandbox remains part of the operation |
| Forking | Child sandboxes are being created from a running source | Temporary pressure can rise sharply |
| Killing | VM is torn down and resources are released | State disappears unless separately persisted |

The difference between `Pausing` and `Paused` is not cosmetic. It is the difference between “the system intends to release resources” and “the system has a durable artifact it can use to recover.” A scheduler that counts both states as free will overcommit. A user interface that treats both as resumable will promise more than the persistence layer can deliver.

### Pause versus snapshot

AgentENV distinguishes pausing a sandbox from creating a persistent snapshot. A pause is about stopping and later resuming the same sandbox. A snapshot is a reusable checkpoint that can become a template or a source for new sandboxes.

The distinction mirrors database terminology:

- A paused sandbox is a suspended execution instance.
- A snapshot is a durable versioned artifact.
- A template is a user-facing alias for a committed snapshot.

The source code and docs also describe failure behavior during snapshotting. Recoverable failures should leave the sandbox running and return an error. Terminal failures, where the runtime has been mutated beyond safe resume, can require teardown. That is the kind of detail that makes a state machine operationally meaningful: each transition has a recovery contract, not just a label.

### TTL is a resource policy, not a timeout convenience

Every sandbox has a time-to-live. The default behavior is to pause when the TTL expires, while an API option can request deletion instead. That policy changes the cost and safety profile of the platform.

Auto-pause is appropriate when the environment contains valuable state and the task may return. Auto-kill is appropriate when the state is disposable, sensitive, or too expensive to retain. The wrong default can cause either data loss or a slow accumulation of paused artifacts.

For an agent platform, TTL should be visible in the application-level task model. An agent run that owns a sandbox must know whether the environment will be paused, deleted, or extended while it is thinking. Otherwise, the runtime will make a correct infrastructure decision that looks like a mysterious model failure.

## 4. Firecracker as the isolation boundary

Firecracker is the component that makes AgentENV a microVM runtime rather than a container wrapper. Each sandbox receives a Linux kernel, filesystem, and network stack. That means a guest process can behave like a normal Linux process without sharing the host kernel's process namespace.

The boundary has two sides. Inside the guest, the agent gets a familiar execution environment. Outside the guest, the host must provide a carefully controlled VM runtime. The host still owns the Firecracker process, block devices, network plumbing, snapshots, and credentials needed to fetch images. Isolation reduces the blast radius; it does not eliminate host responsibility.

### What Firecracker solves

For arbitrary agent code, a separate kernel changes the escape problem. A malicious or compromised guest process no longer begins with direct access to the host process table, host filesystem mounts, or host network namespace. The guest's kernel mediates its own syscalls and devices.

This is a stronger default than running `python generated_code.py` inside the same container as the API service. It also makes the environment more portable as a unit: a snapshot can represent a running Linux workload rather than only a filesystem tree.

### What Firecracker does not solve

It does not solve API authorization. The current README explicitly warns that AgentENV does not support authorization and must not be exposed to the public network. A user who can call the API may be able to create sandboxes, inspect state, run commands, or consume host resources even if every sandbox is perfectly isolated from every other sandbox.

It does not solve secrets management. A guest with a cloud credential can use that credential from inside an isolated VM. Isolation limits where the code runs; it does not make the credential least-privilege.

It does not solve cost control. An agent can create many sandboxes, fork children, consume network bandwidth, or keep enough state hot to pressure the host. Resource policies still belong at the API and scheduler layers.

It does not solve prompt injection. If a model reads hostile text and decides to run a command, the microVM contains the command but does not make the decision correct.

### Host prerequisites are part of the product

The documented prerequisites—Ubuntu 24.04 for the install script, Linux kernel 6.8 or newer, and `/dev/kvm`—are not an installation footnote. They define the operating envelope of the runtime.

An infrastructure team evaluating AgentENV should test these paths before performance tuning:

```bash
uname -r
test -e /dev/kvm
ls -l /dev/kvm
systemctl status aenv
curl --fail http://127.0.0.1:8000/health

# Verify the server is reachable only on the intended interface.
ss -lntp | rg ':8000|:8080|:9090'

# Inspect the service account and device access.
systemctl show aenv -p User -p Group -p ExecStart
getfacl /dev/kvm

# Confirm that no public route exists before exposing the endpoint.
ip route
sudo nft list ruleset
```

The last two commands are not AgentENV commands; they are deployment checks. The point is to make the host network and device boundary observable before a generated program enters the system.

## 5. Storage architecture: overlaybd, ublk, and copy-on-write layers

A sandbox platform that supports many images cannot afford to copy every complete root filesystem onto every node. It needs to share what is immutable, isolate what is mutable, and fetch cold data only when a workload needs it.

![AgentENV storage path: OCI layers feed overlaybd and ublk, while a per-sandbox writable upper layer ends at Firecracker drives](/imgs/blogs/agentenv-distributed-sandbox-runtime-4.webp)

AgentENV's storage path uses overlaybd for layered block images and ublk for userspace block devices. The result is a block-device interface that Firecracker can attach while the underlying data remains layered and cacheable.

### The three storage questions

When debugging a slow or full node, ask three separate questions:

1. **Where is the immutable base data?** It may live in an OCI registry, a committed overlaybd layer, or a node-local image cache.
2. **Where are this sandbox's writes?** They belong to a writable upper layer or a snapshot delta, not to the shared base.
3. **Where is the runtime configuration?** A node-local `image.json` or derived path may be rebuildable and should not be confused with durable snapshot metadata.

The repository documentation repeatedly makes this distinction. Runtime `image.json` files under the snapshot local cache are derived launch inputs. The committed snapshot repository contains logical metadata, Firecracker manifests, VM state, and durable managed layers. The image cache is disposable even when regenerating it is expensive.

### Why block-level layering matters

A filesystem-level copy-on-write layer is not always enough for a VM. Firecracker wants block devices. Overlaybd lets the runtime expose image layers as block-oriented data, while ublk provides a userspace path for serving those devices.

That design creates a useful sharing pattern:

- Many sandboxes read the same base layer.
- Each sandbox writes to its own upper layer.
- Snapshotting seals new data into content-addressed layers.
- A node-local cache keeps hot blocks close to the runtime.
- Remote storage remains the durable source when local data is evicted.

The tradeoff is that a cache miss is now part of the latency path. A “fast sandbox start” claim is meaningful only when it says what happens on a warm cache, a cold cache, a remote read, and a snapshot restore.

### Page cache sharing is an architectural lever

The README highlights shared host page-cache behavior between storage and memory-snapshot data. That is more than a micro-optimization. If the same host pages can serve both the memory snapshot and storage reads, the runtime avoids treating every artifact as a completely separate resident allocation.

This is one reason to resist simplistic capacity planning such as “number of sandboxes × configured memory.” A host may benefit from shared immutable pages, while divergent writable layers and restored memory state consume private capacity. The actual density depends on access patterns, not just declared limits.

### Second-order failure: cache eviction and snapshot roots

The dangerous cache bug is not merely “the cache gets full.” It is deleting a layer that appears cold but is still reachable through a running sandbox, a pinned runtime lease, or a committed snapshot.

AgentENV addresses this with separate ownership rules and runtime leases. Operators should preserve that separation in monitoring and garbage collection. A cleanup job should never infer durability from directory names alone. It should understand which committed records, active sandboxes, and runtime leases reference each content-addressed artifact.

## 6. Snapshots are the real primitive

The most important idea in AgentENV is not the API compatibility layer. It is the decision to make snapshots the durable runtime primitive.

Templates are stored as snapshots. Sandboxes launch by resuming snapshots. Running sandboxes can produce snapshots for later reuse or branching. Paused sandboxes persist enough state to resume the same execution instance. Once this is understood, the apparent feature list—templates, pause, resume, fork—collapses into variations on one state-management mechanism.

![AgentENV artifact ownership: committed snapshots, paused sandboxes, runtime caches, and optional P2P blobs have different durability and rebuild rules](/imgs/blogs/agentenv-distributed-sandbox-runtime-5.webp)

The artifact ownership table is the operational heart of the design:

| Artifact domain | Example contents | Durable source | Rebuildable? |
| --- | --- | --- | --- |
| Committed snapshot repository | Snapshot records, aliases, VM state, Firecracker manifest, managed layers | Repository backend | Usually no |
| Paused sandbox store | Paused record database and artifact generation | Sandbox persister | No while it is the only copy |
| Node-local runtime cache | Resolved `image.json`, downloaded VM state, local paths | Committed metadata plus backend | Yes |
| Image cache | Converted OCI layers, remote block cache, premerged indexes | Source registry or snapshot layers | Yes, at a cost |
| P2P transport store | Iroh blobs and artifact catalog | Snapshot repository remains source of truth | Yes |

### Three layers of snapshot state

The documentation describes three layers that should not be conflated.

**Builder staging** is temporary. A template build needs a workspace while it executes installation steps and captures a running sandbox. It may contain local rootfs layers, memory images, and temporary upper data. It is an implementation detail of the build.

**The committed snapshot repository** is durable. It contains the user-visible snapshot record, alias bindings, Firecracker manifest, VM state, and managed content-addressed layers. It is the source of truth wrapped by the template and snapshot APIs.

**The node-local runtime cache** is derived. Before a sandbox launches, the runtime resolves logical layer references into paths and configurations that the current node can open. Those files can be evicted and regenerated while the committed record remains valid.

This is the same separation that mature databases make between a logically durable record and a local buffer pool. Losing the buffer pool is expensive. Losing the durable record is a correctness failure.

### Memory state and filesystem state are not the same artifact

A snapshot may contain filesystem layers and memory state, but those pieces have different semantics. The root filesystem can often be published as an image or reused as a base. Memory state includes the exact running VM state, open processes, and execution position.

This distinction matters when publishing snapshots back to an OCI registry. The docs state that published images contain the root filesystem only; memory state and `vm_state.bin` remain in the snapshot repository. A team that assumes an image tag alone can restore the complete paused process is making a dangerous category error.

### Snapshot consistency is a protocol

Creating a snapshot is not copying a directory. The runtime has to coordinate:

1. Guest memory state.
2. Writable block-device state.
3. Firecracker launch metadata.
4. Logical snapshot record and alias.
5. Durable backend publication.
6. Optional image publication and P2P advertisement.

Partial success must be handled explicitly. The repository documents best-effort P2P publication after the repository commit: a failed P2P publication does not roll back a successful snapshot publish. That is a sensible consistency boundary. The committed repository is authoritative; P2P is an acceleration path.

The same logic should guide application behavior. If snapshot creation returns an error, the caller should not assume that no state changed. It should query the snapshot catalog and sandbox lifecycle before retrying. Idempotent aliases and content-addressed layers help, but they do not eliminate the need to understand the commit protocol.

## 7. Templates: turning environment setup into a reusable artifact

The best sandbox startup optimization is often to stop doing setup during startup.

Installing Python packages, compiling a tool, downloading a browser, and writing configuration files are all reasonable things for a template build to do. They are poor things to repeat for every agent task. AgentENV's template abstraction turns those steps into a committed snapshot that can be restored many times.

![From image to reusable sandbox: an OCI base becomes build steps, a committed snapshot, a template alias, and many parallel restores](/imgs/blogs/agentenv-distributed-sandbox-runtime-6.webp)

The concrete CLI workflow is:

```bash
# Pull a base image into the AgentENV template store.
aenv pull ubuntu:22.04 --name coding-base

# Start an interactive build environment.
aenv start coding-base --detach

# Use the returned sandbox ID for repeatable setup steps.
aenv exec <sandbox-id> bash -lc '
  apt-get update &&
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl build-essential python3-venv
'

# Capture the prepared state as a reusable snapshot.
aenv snapshot create <sandbox-id> --name coding-template-v1

# Start many task environments from the same prepared state.
aenv start coding-template-v1 --detach
aenv start coding-template-v1 --detach
aenv start coding-template-v1 --detach
```

The exact build interface can evolve, but the engineering shape is stable: expensive preparation is amortized, while mutable task work lives in per-sandbox layers.

### A template is an executable contract

A template is more than an image tag. It may encode environment variables, working directory, user, exposed ports, volumes, labels, startup configuration, rootfs layers, attached drives, and memory layers. That makes it powerful and dangerous.

The template author should document:

| Contract field | Why the agent harness cares |
| --- | --- |
| Working directory | Relative paths and patch commands depend on it |
| User and permissions | Package installation and file ownership can fail unexpectedly |
| Exposed ports | Reverse proxy and health checks need a stable target |
| Environment variables | Missing variables create silent tool failures |
| Network policy | Dependency downloads may be needed during build but forbidden during tasks |
| Snapshot version | Reproducibility requires an immutable version, not only a moving alias |
| Secrets | Build-time credentials must not accidentally become runtime state |

The last item deserves emphasis. If a package manager writes credentials into a configuration file during template construction, every future sandbox may inherit them. A reusable snapshot makes accidental persistence easier, not harder.

### Cold start versus warm start

AgentENV documents both warm starts from templates and cold starts from OCI images. A warm start restores known state. A cold start converts or loads an image at runtime and creates a writable root filesystem.

That distinction should appear in service-level objectives. A single “sandbox startup latency” metric hides the most important operational variable: whether the environment has already paid the image and snapshot preparation cost.

At minimum, record these labels:

- `start_mode=template|cold`
- `image_cache=hit|miss`
- `snapshot_cache=hit|miss`
- `node_id`
- `rootfs_source=registry|local|managed-layer`
- `restore_memory=true|false`

Without those labels, a p95 regression may be impossible to attribute. A new image distribution pattern can make cold starts slower while warm starts remain excellent, or a cache eviction policy can create the reverse pattern.

### Second-order optimization: version the alias, not just the bytes

Aliases are convenient for humans: `coding-template-v1` is easier to pass than a UUID. They also create a deployment problem if an alias is moved while tasks are running. A task started from a template should record the resolved snapshot ID in its run metadata. That gives us a stable answer to “which environment did this agent actually use?” even after an alias is updated.

## 8. Forking and parallel agent workflows

Forking is where snapshots become a coordination primitive for agents rather than merely a faster boot path.

Suppose an agent has spent two minutes installing dependencies, indexing a repository, and building a test harness. It now wants to explore four possible fixes. Rebuilding four environments repeats the expensive work. Forking produces child sandboxes with the same prepared state and lets each branch diverge through copy-on-write.

![Forking for parallel agent workflows: one prepared parent branches into independent agents while lower layers remain shared](/imgs/blogs/agentenv-distributed-sandbox-runtime-7.webp)

The repository documents forking a running sandbox into up to 16 children on the same node. That is a useful upper bound, but it is not a promise that sixteen children are free. The shared lower layers reduce repeated storage, while memory state, writable data, process activity, and output artifacts still create pressure.

### Forking is speculative execution for environments

The analogy to speculative decoding is helpful, but the cost model is different. In speculative decoding, rejected tokens are cheap relative to a full model forward pass. In environment forking, rejected branches may have already consumed CPU, memory, disk writes, network bandwidth, and model tokens.

A harness should attach a budget to each fork tree:

```python
from dataclasses import dataclass


@dataclass
class ForkBudget:
    max_children: int = 4
    max_total_seconds: int = 180
    max_write_mb: int = 512
    max_output_bytes: int = 2_000_000


def should_fork(budget: ForkBudget, children: int, elapsed: int,
                written_mb: int) -> bool:
    if children >= budget.max_children:
        return False
    if elapsed >= budget.max_total_seconds:
        return False
    if written_mb >= budget.max_write_mb:
        return False
    return True


budget = ForkBudget()
if should_fork(budget, children=2, elapsed=48, written_mb=96):
    print("fork another sandbox")
```

This code is an application-level policy, not an AgentENV API implementation. Its purpose is to make the hidden costs explicit before calling a fork endpoint.

### Independent state is the point

Forking is useful only if child sandboxes have a clear independence contract. The agent harness must decide:

- Which files are shared by ancestry and which are branch-local.
- Whether processes are inherited or restarted.
- Whether network identity is unique per child.
- How child results are collected.
- When a failed child can be deleted.
- Which parent snapshot remains the canonical baseline.

If two agents edit the same logical workspace and the harness later merges their filesystem changes without conflict detection, the microVM boundary has not solved the collaboration problem. It has only hidden the conflict until result collection.

### Fork fan-out and placement

The documentation says forked children are created on the same node. That is a strong clue about the implementation: fork is optimized for locality, shared layers, and access to the source's runtime state. It is not a general cluster-wide clone operation.

For a scheduler, that means fork capacity should be tracked separately from ordinary sandbox capacity. A node with enough free CPU to start a fresh environment may not have enough memory headroom to fork a large running VM. A node with sufficient disk may still be unable to restore all child memory states quickly.

## 9. Pausing, resuming, TTLs, and density

Raw boot latency gets attention because it is easy to quote. Pause/resume is often more important because it determines whether an idle environment continues to consume resources.

AgentENV's intended operating model is not “keep every sandbox running.” It is “keep useful state durable, release active resources when the task is idle, and restore the task when work returns.” This is much closer to hibernating a process than restarting a container.

### The density equation is qualitative before it is numerical

A host's usable sandbox density depends on more than the configured memory limit. A useful capacity model is:

$$
\text{usable density} \approx \frac{\text{host memory} - \text{kernel and runtime reserve}}{\text{private guest state} + \text{writable storage pressure} + \text{working-set overhead}}
$$

This is an explanatory capacity heuristic, not an equation stated by the AgentENV project. It captures why shared lower layers and ballooning help, while divergent memory snapshots and write-heavy workloads hurt.

The project describes memory ballooning as a way to return reclaimable guest memory to the host and sustain high overcommit as environments run longer and diverge. That is a density optimization, not a correctness guarantee. If an agent's working set grows again during resume, the host must have enough headroom to satisfy it.

### Auto-pause versus auto-kill

The choice between pausing and killing should be attached to data value and task semantics:

| Policy | Preserves | Costs | Appropriate for |
| --- | --- | --- | --- |
| Pause | Memory, filesystem, processes, open execution state | Snapshot storage and restore work | Long-running coding or research tasks |
| Kill | Nothing unless separately snapshotted | Lowest retention cost | Disposable evaluation trials |
| Snapshot then kill | Reusable checkpoint | Extra commit latency and storage | Branching workflows and durable templates |

An application should not infer “paused” from an HTTP success alone. It should poll or query the lifecycle state until the transition is complete, then persist the resulting sandbox ID and snapshot generation.

### Memory pressure is a feedback loop

Memory pressure can cause a chain reaction:

1. The host reclaims memory from idle or balloonable guests.
2. A burst of resumes requests the same pages.
3. Storage and memory artifacts compete for page-cache space.
4. Remote layer reads increase.
5. Resume latency rises.
6. More tasks remain active while waiting, increasing pressure again.

This is why p50 startup numbers are not enough. Operators should measure resume latency under a controlled burst and distinguish “warm resume with local artifacts” from “resume after eviction.” The latter is closer to production reality during a fleet-wide traffic spike.

## 10. Multi-node control plane

One AgentENV node can be useful for development and small deployments. At scale, the runtime needs a front door that can decide where new sandboxes live and where existing sandboxes already live.

![Multi-node control plane: the gateway receives client traffic, the scheduler selects runtime nodes, and each node runs Firecracker sandboxes](/imgs/blogs/agentenv-distributed-sandbox-runtime-8.webp)

The documented multi-node design adds a Go gateway and scheduler in front of AgentENV runtime nodes. The gateway is the client-facing HTTP and WebSocket entry point. The scheduler provides gRPC placement, heartbeats, and sandbox binding. Runtime nodes host the actual Firecracker environments.

The request paths are different for new and existing sandboxes:

| Request | Scheduler question | Gateway action |
| --- | --- | --- |
| Create sandbox | Which node should receive this new workload? | Forward to selected runtime node |
| List sandboxes | Which nodes should be queried? | Aggregate or route according to deployment semantics |
| Execute in sandbox | Which node owns this sandbox ID? | Forward to owner |
| Proxy service | Which node owns the data plane? | Preserve routing and WebSocket semantics |
| Resume sandbox | Is the persisted state local or fetchable here? | Route according to ownership and artifact availability |

### Static discovery is intentionally simple

The static multi-node deployment uses a configured node list. Each runtime node has an ID and endpoint. The scheduler does not automatically register unknown nodes from heartbeats; changing membership requires updating configuration and restarting the scheduler.

That is not a weakness by itself. Static membership is often the right first control plane when the cluster is small and node identity changes infrequently. It reduces the number of moving parts and makes network policy easier to audit.

It does mean that operations must treat configuration as part of cluster state. A node ID mismatch, an unreachable endpoint, or a stale scheduler process can produce routing failures that look like sandbox failures.

The documented static deployment uses a shape like this:

```json
{
  "scheduler": {
    "grpc_listen_addr": "0.0.0.0:9090",
    "strategy": "round_robin",
    "report_ttl": "30s",
    "binding_ttl": "30s",
    "discovery": { "mode": "static" },
    "nodes": [
      { "id": "node-a", "endpoint": "http://10.0.0.21:8000" },
      { "id": "node-b", "endpoint": "http://10.0.0.22:8000" }
    ]
  },
  "gateway": {
    "http_listen_addr": "0.0.0.0:8080",
    "scheduler_addr": "10.0.0.10:9090",
    "request_timeout": "90s"
  }
}
```

The values above come from the project's static multi-node documentation. A production deployment should add its own authentication boundary, network encryption, metrics exposure policy, and failure-testing rules.

### Ownership is more important than round robin

Round-robin placement is fine for an initial scheduler. It is not sufficient as a mental model for stateful workloads. After a sandbox exists, the important data is not “the next node”; it is “the node that owns this sandbox and can reach its artifacts.”

The gateway and scheduler should therefore expose ownership signals to operators:

- Sandbox ID to node ID binding.
- Binding age and last heartbeat.
- Whether the node has the required template layers locally.
- Whether the sandbox is running, paused, or resuming.
- Whether the request is control-plane or data-plane traffic.

### Control-plane failure modes

The obvious failure is scheduler unavailability. The less obvious failures are stale truth and split ownership.

If the scheduler restarts and loses in-memory bindings, it must repopulate them from runtime heartbeats or persistent records. If a node is alive but its heartbeat is delayed, routing can oscillate between “node unavailable” and “sandbox exists.” If a gateway caches an old endpoint longer than the binding TTL, it can send valid requests to the wrong place.

These are standard distributed-systems problems, but the recovery action is not always retry. Retrying a command against a different sandbox is wrong. Retrying a read-only health check may be fine. A client library should classify operations by idempotency and sandbox ownership before it chooses a retry target.

## 11. E2B compatibility as an adoption strategy

Compatibility is often more valuable than novelty in infrastructure. AgentENV exposes an E2B-compatible HTTP API, so existing E2B Python and TypeScript SDK code can point at an AgentENV server by changing environment variables.

The benefit is not only convenience. It creates a migration seam. An application can start with a managed E2B environment, move to a trusted single-node AgentENV deployment for development or cost control, and later put AgentENV behind a gateway without rewriting the agent harness.

The compatibility boundary also clarifies what must not be assumed. API shape compatibility does not guarantee identical behavior for:

- Authentication and authorization.
- Template availability.
- Snapshot durability.
- Network egress semantics.
- Proxy domains.
- Startup latency under cold cache.
- Limits, quotas, and error codes.

A minimal Python integration looks like this:

```python
import os

from e2b import Sandbox, SandboxQuery, SandboxState


os.environ["E2B_API_URL"] = "http://127.0.0.1:8000"
os.environ["E2B_SANDBOX_URL"] = "http://127.0.0.1:8000"
os.environ["E2B_API_KEY"] = "dummy"
os.environ["E2B_ACCESS_TOKEN"] = "dummy"


sandbox = Sandbox.create("coding-template-v1")
result = sandbox.commands.run("pytest -q", timeout=120)
print(result.stdout)

running = Sandbox.list(
    limit=20,
    query=SandboxQuery(state=[SandboxState.RUNNING]),
)
print(running.next_items())

sandbox.beta_pause()
```

The SDK call structure is compatible, but the deployment still needs to define what “dummy” means, where the server is reachable, and how a paused sandbox is recovered after a node restart. Compatibility lowers application migration cost; it does not outsource infrastructure semantics.

### Compatibility as a testing strategy

The same harness can test against two backends:

1. A managed provider for baseline API behavior.
2. AgentENV for self-hosted isolation, snapshot, and storage behavior.

That creates a useful contract test suite. For each backend, test sandbox creation, command streaming, file transfer, proxy routing, pause/resume, timeout behavior, and deletion. Record not only success but state transitions and latency distributions.

## 12. Security is a stack, not a single boundary

AgentENV's strongest security lesson is the warning in its own quick start: the API currently does not provide authorization and should not be exposed publicly.

![Security is a stack: untrusted input, API authorization, network policy, Firecracker isolation, and guest processes protect different failure surfaces](/imgs/blogs/agentenv-distributed-sandbox-runtime-9.webp)

That warning should dominate the deployment design. Firecracker protects the host from many guest-level failures, but it does not decide who can call the API. Network policy limits guest egress, but it does not remove secrets already mounted into the guest. A trusted network reduces exposure, but it does not authenticate every caller inside that network.

### Five distinct boundaries

**Input boundary.** User prompts, repositories, documents, and tool results are untrusted data. They may contain instructions that the model repeats or converts into code.

**API boundary.** The caller should be authenticated and authorized before it can create or control sandboxes. AgentENV's current README says this layer is not yet provided by the project, so it must be supplied by a trusted network or an authorization proxy.

**Network boundary.** A sandbox may need package downloads or access to a test service. It should not automatically reach every internal address. AgentENV documents `allowOut` and `denyOut` controls and allows network rules to be updated on a running sandbox.

**VM boundary.** Firecracker provides an independent guest kernel and device model. This is the boundary that makes arbitrary Linux execution materially safer than an unconfined process.

**Guest-process boundary.** The guest still needs least privilege. A process can misuse files, credentials, packages, and local services even without escaping the VM.

### Egress policy is a capability system

The API supports disabling internet access or passing an allow/deny network object. The documented precedence is that allowed entries take precedence over denied entries.

An application should construct the policy from task requirements rather than expose a generic “internet on” switch:

```bash
curl -X POST \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: task-proxy' \
  -d '{
    "templateID": "coding-template-v1",
    "allowInternetAccess": false,
    "network": {
      "allowOut": ["pypi.org", "files.pythonhosted.org"],
      "denyOut": ["0.0.0.0/0"]
    }
  }' \
  http://127.0.0.1:8000/sandboxes

curl -X PUT \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: task-proxy' \
  -d '{
    "allowOut": ["8.8.8.8/32"],
    "denyOut": ["0.0.0.0/0"]
  }' \
  http://127.0.0.1:8000/sandboxes/<sandbox-id>/network
```

The example is illustrative of the documented HTTP shape. A real deployment should put an authenticated proxy in front of these calls and avoid treating an API key in a shell example as a security boundary.

### Secrets and snapshots

Snapshots create a subtle persistence risk. A running process may hold a token in memory. A configuration file may hold a package registry credential. A shell history may contain an access token. A snapshot that preserves memory and filesystem state can preserve those values too.

The safe pattern is:

- Inject short-lived credentials at task start.
- Scope them to the minimum API actions.
- Avoid baking credentials into templates.
- Scrub shell history and build logs.
- Treat memory snapshots as sensitive artifacts.
- Delete or expire snapshots with the same seriousness as secrets.

The microVM boundary does not change this. An attacker who legitimately controls the guest can use every secret the guest can read.

### Security review checklist

Before a production deployment, answer these questions with evidence:

| Question | Evidence to require |
| --- | --- |
| Who may create a sandbox? | Proxy policy and audit logs |
| Who may execute commands in an existing sandbox? | Sandbox ownership and authorization tests |
| Which destinations can a guest reach? | Network policy tests from inside the guest |
| How are snapshots encrypted and retained? | Backend configuration and deletion verification |
| What is mounted from the host? | Firecracker and service unit review |
| What happens when `/dev/kvm` or storage fails? | Failure-injection results |
| Can an agent exhaust the node? | Quotas, TTLs, fork limits, and alerts |
| Can operators reconstruct a run? | Lifecycle events, command logs, and trace IDs |

## 13. Observability and failure handling

Agent environments create two traces at once. There is the agent trace—model calls, tool calls, retries, and reasoning steps—and there is the infrastructure trace—sandbox creation, image resolution, VM boot, command execution, snapshotting, and proxy forwarding.

If those traces are not correlated by a task ID and sandbox ID, the system will produce the most frustrating class of incident: the model says the tool timed out while the infrastructure says the request completed, or the infrastructure says the VM is healthy while the gateway routed the request to the wrong node.

### Metrics that explain the runtime

At minimum, collect:

| Metric | Useful dimensions | What it diagnoses |
| --- | --- | --- |
| Sandbox create latency | template, cold/warm, node | Image or scheduler delay |
| Resume latency | snapshot, cache hit, node | Artifact cache and memory restore pressure |
| Pause latency | dirty bytes, memory size | Snapshot write path and guest quiescing |
| Snapshot failure count | phase, backend, sandbox | Persistence correctness |
| Image cache hit rate | image digest, node | On-demand loading effectiveness |
| Node disk usage | cache, snapshot, runtime | Incorrect eviction or capacity planning |
| Guest command latency | command class, sandbox | `envd`, guest process, or VM issue |
| Proxy latency | route, protocol, node | Gateway and data-plane bottlenecks |
| Heartbeat age | node ID | Stale placement truth |
| Fork count and child survival | parent, node | Fan-out pressure and branch quality |

The exact metric names are an implementation choice. The important property is that each metric maps to a subsystem boundary.

### Errors should preserve lifecycle context

An error such as `connection refused` is not enough. The runtime should make it possible to answer:

- Which sandbox ID was involved?
- Which lifecycle state was expected?
- Which node owned it?
- Was this a control-plane or data-plane request?
- Which snapshot generation was being restored?
- Was the artifact local, remote, or missing?
- Was the operation safe to retry?

This is closely related to [agent observability and tracing](/blog/machine-learning/ai-agent/agent-observability-and-tracing): the model-level trace needs infrastructure spans underneath it. It also connects to [tool error recovery](/blog/machine-learning/ai-agent/tool-error-recovery): the agent should receive a structured distinction between a transient proxy failure, a sandbox expiration, a permanent permission error, and an unavailable snapshot.

### Retry policy must understand side effects

The runtime exposes operations with very different retry semantics:

| Operation | Safe default |
| --- | --- |
| Health check | Retry with backoff |
| List sandboxes | Retry if the caller can tolerate stale results |
| Execute command | Retry only with an idempotency key or command classification |
| Create sandbox | Retry with a client request ID to avoid duplicates |
| Pause | Query state before retrying |
| Snapshot create | Query catalog and sandbox state before retrying |
| Delete | Retry after confirming the target identity |
| Fork | Never blindly retry without checking whether children already exist |

The most expensive bugs here are duplicate side effects. If the command inside a sandbox sends an email, charges a card, or pushes a commit, a gateway retry can execute it twice even though the sandbox API itself is healthy.

## 14. Operational case studies

The repository does not present these as historical production incidents. They are concrete failure scenarios derived from AgentENV's documented lifecycle, storage, and deployment model. That distinction matters: the purpose is to show how an operator should reason about the system, not to invent an outage history for the project.

### 1. The cold-start illusion

The team benchmarks `aenv start coding-template-v1` on a node that has already run the template ten times. The p95 is excellent. A production burst arrives with twenty new images, and latency triples.

The first hypothesis is that Firecracker is slow under concurrency. The actual root cause is image and artifact locality. The warm benchmark measured restored runtime configs and hot page-cache pages. The burst caused OCI resolution, remote block reads, overlaybd setup, and local cache writes to happen together.

The fix is not to make every node pre-pull every image. That defeats AgentENV's diversity goal. The fix is to separate warm and cold startup SLOs, expose cache-hit dimensions, and size the remote registry and node-local cache for burst behavior. A scheduler can also use locality signals when assigning a template with known hot layers.

The lesson is that a fast restore path and a fast image materialization path are different products. Benchmark both.

### 2. The paused sandbox that will not resume

An agent pauses successfully, receives a durable-looking response, and later fails to resume after the server restarts. The operator finds a paused record, but the expected artifact directory is incomplete.

The first hypothesis is a Firecracker regression. The more likely class of failure is a persistence protocol mismatch: the record was written before all artifacts were durable, or cleanup removed an orphaned generation that the record still referenced.

The fix is to make pause completion conditional on artifact verification, write the record only after the generation is complete, and test restart recovery with interrupted writes. The persister should use generation IDs so a failed new pause cannot destroy the last known-good artifact.

The lesson is that a state record and its artifacts form one logical object. Persisting a JSON row is not the same as persisting a resumable sandbox.

### 3. The full disk caused by a bounded cache

The node-local image cache is configured with a size limit, yet disk usage continues to climb. The team sees many old overlaybd files and assumes the cache garbage collector is broken.

The actual problem is ownership. Some files are referenced by active runtime leases or committed snapshot roots. Others are paused sandbox artifacts, not image-cache entries. Deleting everything that looks old would trade a capacity problem for a correctness failure.

The fix is to report disk usage by artifact domain, then make garbage collection reference-aware. Eviction should remove only unpinned derived artifacts. Committed snapshot layers require repository-level lifecycle decisions, not local cache heuristics.

The lesson is that bounded working sets still need a root graph. “Not recently accessed” is not enough to prove “safe to delete.”

### 4. The fork storm

An agent decides that eight candidate patches are better than one. Four parents each fork four children. CPU looks acceptable, but resume latency and host memory pressure become erratic.

The first hypothesis is scheduler unfairness. The root cause is that fork shares immutable ancestry but not every active working-set page. Children begin running tests, diverge in memory, write logs, and create separate build outputs. The copy-on-write advantage shrinks as the branches become independent.

The fix is to limit fork depth and total children per task, charge branches to a shared resource budget, and terminate low-value branches early. The harness should collect partial results so it does not keep every child alive until the slowest one finishes.

The lesson is that fork is a latency optimization for common ancestry, not a license to multiply workloads without accounting.

### 5. The scheduler with stale ownership

A gateway receives a request for an existing sandbox. The scheduler says node A owns it, but node A returns not found. A retry to node B finds a different sandbox with a similar template but not the requested state.

The first hypothesis is data loss. A more precise diagnosis is stale binding state: the runtime node restarted, the scheduler lost or aged out its binding, and the gateway used an old route or incomplete recovery data.

The fix is to make sandbox identity and node ownership observable, reconcile runtime heartbeats with persisted sandbox records, and distinguish “not found on this node” from “sandbox does not exist.” A client must never silently substitute another environment for the requested ID.

The lesson is that stateful routing needs reconciliation, not just load balancing.

### 6. The public API exposure

An operator deploys the Docker image with port 8000 published on a cloud interface. They assume Firecracker is the security boundary and plan to add authentication later.

The first incident is not a VM escape. It is unauthorized sandbox creation and command execution through an API that the project itself warns is not authorized. An attacker can consume resources or use the runtime as a network foothold without ever breaking out of a guest.

The fix is to bind the service to a trusted interface, put an authenticated authorization proxy in front, restrict firewall paths, and log every control-plane call. The proxy must protect WebSocket and reverse-proxy routes as well as ordinary HTTP endpoints.

The lesson is direct: isolation is not authorization. A perfectly isolated public API can still be an exposed service.

### 7. The template that bakes in secrets

A build step configures a private package registry and installs dependencies. The template works. Weeks later, a task running from the same template can read the registry credential from a config file and from process environment inherited by a startup script.

The first hypothesis is prompt injection. The underlying problem is snapshot hygiene. Templates preserve the environment that was built, including files and metadata engineers forgot to remove.

The fix is to use short-lived build credentials, clean package-manager configuration before snapshotting, scan the root filesystem and environment, and separate build templates from task templates. Treat memory and filesystem snapshot artifacts as sensitive until proven otherwise.

The lesson is that reproducibility preserves mistakes with the same efficiency as it preserves good setup.

### 8. The proxy that becomes the bottleneck

Commands execute quickly inside the sandbox, but an agent serving a local HTTP application experiences long tail latency. Direct health checks from the node are fine. The gateway path is not.

The first hypothesis is guest networking. The actual bottleneck may be the reverse proxy: connection tracking, WebSocket forwarding, request size limits, timeout configuration, or an ownership lookup on every request.

The fix is to instrument proxy spans separately from sandbox command spans, test long-lived WebSocket connections, and verify that routing headers or sandbox proxy domains preserve identity. A control-plane success should not be mistaken for a healthy data plane.

The lesson is that “the sandbox is running” and “the service is reachable” are separate health signals.

## 15. When to reach for AgentENV—and when not to

AgentENV is compelling when the environment itself is part of the agent's work. It is less compelling when the environment is incidental.

### Reach for AgentENV when

- Agents need arbitrary code, shell commands, compilers, or package installation.
- Each task needs a fresh or isolated Linux environment.
- A long-running task must pause and resume without losing process state.
- Many tasks share a prepared base but diverge in their writes.
- Forking can accelerate parallel search, testing, or review workflows.
- Your team can operate Linux hosts with `/dev/kvm`, privileged setup, storage backends, and an authorization proxy.
- E2B API compatibility gives you a practical migration path.

### Skip AgentENV when

- The agent only needs fixed, typed API tools.
- A managed sandbox service already meets your latency, residency, and cost requirements.
- You cannot safely operate the privileged host and storage dependencies.
- The deployment requires public multi-tenancy before an authorization layer exists.
- Workloads are short, stateless, and do not benefit from snapshots or forkable state.
- Your team cannot build the lifecycle, quota, observability, and incident-response layer around the runtime.

### The decision is about operational appetite

The architecture is technically interesting because it combines several hard systems problems: VM isolation, remote layered storage, stateful snapshots, copy-on-write branching, and distributed placement. That combination is also the reason not to adopt it casually.

AgentENV can reduce the cost of agent environments by sharing immutable layers, pausing idle sandboxes, and restoring prepared state. It can increase operational complexity through kernel prerequisites, privileged device access, cache correctness, snapshot retention, scheduler ownership, and security boundaries that the project currently leaves to the deployment.

The senior engineering decision is not “does AgentENV have a fast startup demo?” It is:

> Can our platform make sandbox identity, artifact durability, resource policy, and authorization as observable and boring as the agent API we want developers to use?

If the answer is yes, AgentENV is a promising substrate for long-running and code-executing agents. If the answer is no, the runtime will faithfully expose every missing piece of the platform around it.

### A practical evaluation and rollout sequence

Adoption should happen in stages. Start with a single trusted host and a workload that already has a clear cleanup policy. Do not begin with public multi-tenancy, a large image catalog, or a workload that can spend money through external APIs. The first goal is to verify that the runtime can preserve the state your application actually needs.

The first experiment should be a boring command session:

1. Pull one small OCI image.
2. Start one sandbox.
3. Create a file and a long-running process.
4. Pause the sandbox.
5. Restart the server.
6. Resume the sandbox.
7. Verify the file, process state, and command output.
8. Delete the sandbox and verify artifact cleanup.

This test is more valuable than a benchmark that only measures an empty boot. It exercises the boundary between the guest, the filesystem, the memory snapshot, and the persisted sandbox record.

The second experiment should vary storage locality. Run the same template after a warm cache, after clearing only derived runtime configuration, and after evicting the node-local image cache. Record each phase separately. A useful report has rows for image resolution, block-device setup, VM boot or restore, guest health, and first command completion. A single stopwatch around `aenv start` does not tell an operator which resource to add.

The third experiment should vary task lifetime. Run a short task that is killed, an idle task that is auto-paused, a task that is manually paused and resumed, and a task that creates a persistent snapshot. Compare retained disk, retained memory, restore latency, and cleanup time. The results will reveal whether the default TTL matches the application rather than merely proving that the command exists.

The fourth experiment should vary write behavior. A read-heavy task should share a large portion of its lower layers. A package installation or compiler workload should create a large writable delta. A task that repeatedly rewrites a large file may stress the upper layer more than a task that creates many small files. These cases determine whether copy-on-write is helping the workload you care about.

### A benchmark record worth keeping

Store benchmark results with the repository revision, host kernel, CPU count, memory size, storage device, registry endpoint, image digest, template snapshot ID, and configuration file. The same template alias is not sufficient because an alias can move.

```ini
revision=main@<commit>
kernel=6.8.x
host_memory_gib=256
storage=local-nvme
image_digest=sha256:<digest>
template_snapshot=<snapshot-id>
start_mode=warm-template
concurrency=32

create_p50_ms=<value>
create_p95_ms=<value>
resume_p50_ms=<value>
resume_p95_ms=<value>
pause_p95_ms=<value>
first_command_p95_ms=<value>
image_cache_hit_rate=<value>
node_disk_peak_gib=<value>
```

The placeholders are intentional. AgentENV's documentation reports target latencies, not a universal benchmark that applies to every host, image, and workload. A team should measure its own values rather than copy a number into a service-level objective.

The benchmark should also include a concurrency sweep. Start with one sandbox, then repeat at 2, 4, 8, 16, and 32 concurrent starts while keeping the image and template fixed. For each point, capture the median, tail latency, host CPU, host memory, disk read rate, disk write rate, and remote registry traffic. The shape of the curve is often more informative than the absolute number. A sharp knee usually means one shared resource has become the limiting queue.

Run the same sweep for pause and resume. Pause is not only a shutdown operation; it is a write workload whose cost depends on dirty memory and filesystem changes. Resume is not only a VM boot; it is a read workload whose cost depends on artifact locality and page-cache pressure. Measuring one without the other makes it easy to optimize the wrong half of the transition.

For agent workloads, include a command-level measure after resume. A VM can report healthy before the first useful command has completed. The application cares about “sandbox ready for `git status`” or “sandbox ready for the test runner,” not only about a Firecracker process existing. Keep both numbers: infrastructure readiness and task readiness.

Finally, retain failure samples, not just successful timings. A slow resume caused by a remote read and a failed resume caused by a missing artifact may share the same p99 bucket but require entirely different fixes. A benchmark report that discards errors is a capacity report without a reliability report.

### Capacity planning by workload class

A single quota for all agent tasks usually produces poor utilization. Separate at least three classes:

| Workload class | Typical behavior | Useful policy |
| --- | --- | --- |
| Short evaluation | Small writes, no durable state, many trials | Short TTL, auto-kill, strict output cap |
| Coding session | Large repository, tests, pauses, valuable state | Auto-pause, longer TTL, persistent snapshot option |
| Parallel search | One prepared parent, many child branches | Fork budget, aggregate CPU and memory quota |
| Service-backed tool | Long-lived HTTP process and WebSockets | Proxy quota, stable port contract, heartbeat checks |

The scheduler can place these classes differently even before it has sophisticated bin packing. Short trials can use nodes with warm common images. Coding sessions can prefer nodes with snapshot locality. Parallel search can be constrained to nodes with enough memory headroom for branch divergence.

### Failure injection before production traffic

The runtime has several failure points that should be tested deliberately:

- Kill the server during sandbox creation.
- Interrupt a pause while artifacts are being written.
- Restart the node after a pause but before the client receives the response.
- Remove a derived runtime cache and verify it is regenerated.
- Make the registry unavailable during a cold start.
- Fill the node's image cache and confirm that active layers remain reachable.
- Delay scheduler heartbeats and observe gateway routing.
- Drop gateway-to-node traffic while a guest process continues running.
- Terminate a fork child during a large write.
- Expire a sandbox during a long command and verify the client receives a state-aware error.

For each test, record the final state rather than only the HTTP status. A request can return an error while the sandbox remains safely running, or return success while the next operation still needs to wait for a transition. The recovery runbook should tell an operator when to query, when to retry, and when to stop touching the target.

### Rollout gates

Use explicit gates instead of a vague “ready for production” label:

**Gate one: single-node correctness.** Every lifecycle transition survives process restart. Snapshot records match their artifacts. Deletion does not remove state still referenced by a committed snapshot.

**Gate two: isolation.** A guest cannot read host paths, reach forbidden network destinations, or access another sandbox's writable state. The test suite runs from inside the guest and records both allowed and denied results.

**Gate three: resource control.** A task cannot exceed its CPU, memory, disk, process, network, command-output, or fork budget without a visible terminal state. Host pressure produces backpressure before the node becomes unresponsive.

**Gate four: ownership recovery.** Runtime restarts, scheduler restarts, and gateway restarts produce a consistent answer for every existing sandbox ID. The system does not silently route a request to a different environment.

**Gate five: observability.** An engineer can start at an agent run ID and find the sandbox ID, snapshot ID, node ID, lifecycle transitions, command spans, proxy spans, and cleanup result. Logs alone are not enough if they cannot be joined.

**Gate six: authorization.** The API is reachable only through an authenticated and authorized path. This gate must be complete before public or cross-tenant exposure; the repository's current no-authorization warning is an explicit reason not to skip it.

### The handoff to an on-call engineer

The first production run should ship with a runbook that starts from identifiers rather than symptoms. Given an agent run ID, the engineer should be able to find the sandbox ID, resolved snapshot ID, node ID, current lifecycle state, last transition timestamp, and the latest command or proxy span. Given a sandbox ID, the engineer should be able to determine whether it is active, paused, resuming, or terminal; whether its artifacts are local or remote; and whether another operation currently holds a lifecycle lease.

The runbook should define safe actions in order. First query state. Then inspect the node health and scheduler binding. Then inspect artifact availability. Only after those checks should the operator retry, pause, resume, or delete. “Restart the service” is not a diagnosis and can make an in-progress snapshot harder to recover.

A useful incident page includes four small tables:

| Symptom | First query | Do not do first |
| --- | --- | --- |
| Command timeout | Guest command span and sandbox state | Retry against another sandbox |
| Resume failure | Snapshot generation and artifact presence | Delete the paused record |
| Gateway 404 | Sandbox-to-node binding | Create a replacement sandbox silently |
| Disk pressure | Artifact ownership and active leases | Delete old-looking layers manually |

| State | Safe operator action | User-facing meaning |
| --- | --- | --- |
| Creating | Wait, inspect setup spans | Environment is not ready |
| Running | Execute or inspect guest health | Work may be active |
| Pausing | Wait for transition completion | Resource release is in progress |
| Paused | Resume or delete by policy | State is retained but inactive |
| Resuming | Wait, inspect local and remote artifacts | Work is being restored |
| Killing | Confirm terminal cleanup | State is being removed |

This operational detail is where a sandbox runtime becomes a platform. Developers should see a small, stable API. Operators need the full state machine, artifact graph, ownership record, and resource history underneath it.

The rollout should also include an explicit rollback path. If a new runtime revision changes snapshot format, image resolution, network setup, or gateway routing, keep the previous binary and configuration available until existing sandboxes have either completed or been migrated. Do not assume that a blue-green deployment for stateless HTTP handlers automatically works for live VM state. A new control-plane process may be compatible with old records, while a new runtime node may not be able to resume every older artifact.

That compatibility check belongs in continuous integration, not only in the launch checklist.

For that reason, record the runtime revision that created each committed snapshot and paused generation. Test a mixed-version cluster before making a new version the default. If mixed-version support is not available, drain live sandboxes, create a durable checkpoint where appropriate, and perform a deliberate migration rather than allowing the scheduler to discover incompatibility through user traffic.

### What success looks like

Success is not a fleet of VMs with a low median startup time. Success is a task whose environment behaves predictably across the moments that matter:

- The requested template resolves to a known snapshot.
- The sandbox ID remains stable across pause and resume.
- A command runs in the intended guest and nowhere else.
- A failed request can be classified without guessing whether work happened.
- An idle task releases active resources without losing valuable state.
- A resumed task can explain which artifacts were restored.
- A forked branch has a budget and a result owner.
- A node failure produces a recoverable state or an explicit terminal error.
- An operator can prove who controlled the sandbox and which network policy applied.

That is the standard AgentENV should be evaluated against. The project supplies important primitives: Firecracker isolation, layered storage, snapshots, fork, TTLs, an E2B-compatible API, and an initial multi-node control plane. The surrounding platform must supply the policy, identity, quotas, encryption, and operational evidence that turn those primitives into a safe service.

## Further reading

- [AgentENV repository](https://github.com/kvcache-ai/AgentENV)
- [AgentENV official documentation](https://kvcache-ai.github.io/AgentENV/)
- [Agent Sandboxing Strategies](/blog/machine-learning/ai-agent/agent-sandboxing-strategies)
- [Code Execution as a Tool](/blog/machine-learning/ai-agent/code-execution-as-a-tool)
- [Stateful Agent Deployment](/blog/machine-learning/ai-agent/stateful-agent-deployment)
- [Scaling Managed Agents: Decoupling the Brain from the Hands](/blog/machine-learning/ai-agent/scaling-managed-agents-decoupling-brain-from-hands)

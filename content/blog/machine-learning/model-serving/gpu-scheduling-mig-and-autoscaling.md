---
title: "GPU scheduling, MIG, and autoscaling: squeezing more serving out of a fixed fleet"
date: "2026-07-03"
publishDate: "2026-07-03"
description: "How to get the most serving capacity out of an expensive, indivisible GPU fleet — partition each card with MIG, share it with time-slicing or MPS, schedule it on Kubernetes, and autoscale model servers on the metrics that actually matter."
tags:
  [
    "model-serving",
    "inference",
    "mig",
    "gpu-scheduling",
    "autoscaling",
    "kubernetes",
    "keda",
    "hpa",
    "cost-optimization",
    "ml-infrastructure",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/gpu-scheduling-mig-and-autoscaling-1.webp"
---

Your finance team sends you the cloud bill and asks one question: "We are paying for forty A100s. Why is the average GPU utilization on the dashboard 22%?" You already know the answer, because you built the fleet. Half of those GPUs run small classifiers and embedding models that each use a sliver of the card. A third run an LLM chatbot that is busy at 2 PM and idle at 2 AM but stays powered on all night because a cold start takes three minutes and nobody wants the first customer of the morning to wait. A handful sit reserved for a batch job that runs twice a week. Every one of those GPUs is billed at the same rate whether it is at 5% or 95%, and a GPU is the single most expensive line item you have.

This is the central tension of GPU serving infrastructure: **a GPU is expensive and, by default, indivisible.** You rent the whole card. If your model uses one-seventh of it, you have thrown away six-sevenths of the most expensive compute in the datacenter. And if traffic drops to zero overnight, you keep paying — unless you can turn the card off fast enough to turn it back on before anyone notices. Every technique in this post is an attack on one of those two problems: **make one GPU serve more work** (partitioning and sharing) and **make the fleet track the load** (autoscaling, including scaling to zero).

The figure below frames the whole thing. On the left, one small model on a full A100 80GB: 14 GB of weights on an 80 GB card, roughly 12% streaming-multiprocessor (SM) utilization, the rest of the silicon burning power for nothing. On the right, that same card carved by Multi-Instance GPU (MIG) into seven hardware-isolated slices, each running its own tenant, aggregate utilization near 85%, and a 7x reduction in card count for a many-small-model fleet. This post is about how to get from the left picture to the right one, safely, on Kubernetes, and about when doing so is a mistake.

![Before and after diagram contrasting one small model wasting a full A100 against seven models packed into MIG slices on one card](/imgs/blogs/gpu-scheduling-mig-and-autoscaling-1.webp)

By the end you will be able to: partition an A100 or H100 into MIG slices from the command line and request them from a pod; decide between MIG, time-slicing, and CUDA MPS by their isolation and QoS properties rather than by which one a blog told you to use; schedule GPU workloads on Kubernetes with the NVIDIA device plugin, bin-packing, and gang scheduling for multi-GPU replicas; autoscale model servers on queue depth and KV-cache utilization instead of the useless CPU metric; wire up KEDA to scale a spiky service to zero; and budget the cold-start latency so scale-to-zero does not blow your SLA. This is Track D of the serving series — it assumes you know what serving is and how to reason about an SLO. If you do not, start with [what model serving is](/blog/machine-learning/model-serving/what-is-model-serving) and [serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics), then come back.

Throughout, keep the series' spine in mind: every technique is a trade on the **latency, throughput, cost** triangle. Partitioning buys cost (more tenants per card) at the price of per-tenant peak throughput. Scale-to-zero buys cost at the price of tail latency on the first request. There is no free capacity — only trades you choose deliberately.

## The packing problem: why a GPU is expensive and indivisible

Start with the economics, because they drive everything. A datacenter A100 80GB rents, at representative on-demand cloud rates, for somewhere around \$3.50 to \$5.00 per GPU-hour; an H100 80GB runs higher, often \$4 to \$12 per GPU-hour depending on provider and commitment. Reserved and spot pricing cut that substantially, but the ratio to a CPU core does not change: a single GPU costs as much per hour as dozens of CPU cores, and you cannot buy a fraction of one from the hypervisor. When you schedule a pod that requests `nvidia.com/gpu: 1`, Kubernetes gives that pod the entire physical device. There is no `nvidia.com/gpu: 0.15`.

Now put a small model on it. A 3-billion-parameter model in FP16 is about 6 GB of weights; in INT8, about 3 GB. Add the KV cache (the per-request attention key/value memory that grows with sequence length and batch size — the [reason LLM serving is memory-bound](/blog/machine-learning/model-serving/what-is-model-serving)), a CUDA context, and activation scratch, and you might touch 10-15 GB on an 80 GB card. You are using one-sixth of the memory. Worse, at low query rates the compute is nearly idle: a single request at a time barely occupies the SMs, so GPU compute utilization sits in the low teens. You are renting a Ferrari to drive to the mailbox.

Put real numbers on that waste before naming its parts. The 3B INT8 classifier holds about 3 GB of weights, and even under a steady trickle of requests its working set — weights, a modest KV cache, CUDA context, activation scratch — rarely crosses 10 GB. On an 80 GB A100 that is a memory-fill fraction of 10/80, roughly 12%; the other 70 GB are allocated to nobody. The compute axis is worse. A short classification request launches a burst of kernels that occupy the SMs for a few milliseconds, then the card idles until the next request arrives. At 30 requests per second with a 5 ms compute footprint each, the SMs are busy for 150 ms out of every second — a 15% duty cycle. The two losses are independent and they multiply: the card delivers on the order of a tenth of what you rent. And the waste does not shrink if you buy a bigger card — it grows, because a fixed-size model fills an even smaller fraction of more silicon.

The waste has two independent axes, and it is important to keep them separate:

- **Memory waste.** The model plus its working set does not fill HBM. This is a *packing* problem: you could fit several such models side by side if only the scheduler would let you.
- **Compute waste.** Even the memory you do use is not kept busy — the SMs idle between requests. This is a *utilization* problem, and it is what batching (covered in [batching fundamentals](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff)) attacks by keeping more work in flight.

The techniques in this post attack the packing problem directly. If one model cannot fill a GPU, put several models on it. The question is *how* you share the card, and the answer determines whether your tenants are safely isolated or free to destroy each other's latency.

There are three mechanisms, and they sit at very different points on the isolation-versus-flexibility spectrum:

1. **MIG (Multi-Instance GPU)** — hardware partitioning. The GPU is physically divided into isolated instances with dedicated SMs, dedicated slices of HBM, dedicated memory bandwidth, and dedicated L2 cache. A tenant in one MIG instance literally cannot see or affect a tenant in another. This is the strongest isolation and the least flexibility.
2. **Time-slicing** — software oversubscription. The GPU context-switches between processes in round-robin fashion. Every process sees the full GPU, one at a time. There is *no* memory isolation and *no* compute quality-of-service (QoS): any process can allocate all of HBM or hog the SMs. Maximum flexibility, zero safety.
3. **MPS (Multi-Process Service)** — spatial sharing of one context. CUDA MPS lets multiple processes submit kernels into a single shared GPU context so their kernels run *concurrently* rather than time-sliced. You can cap each client's SM fraction and set a soft memory limit, but they share one context, so a fatal GPU error takes them all down. It sits between the other two.

Let us take each in turn, then compare them on the axes that matter and derive the utilization math that tells you when partitioning actually pays.

## MIG: hardware-partitioned GPU slices

MIG is available on NVIDIA's datacenter Ampere, Hopper, and Blackwell parts — A30, A100, H100, H200, and B200. It is **not** available on the Ada Lovelace inference cards (L4, L40S), on Turing (T4), on Volta (V100), or on any consumer GPU. If your fleet is L40S, MIG is off the table and you are choosing between dedicated, time-slicing, and MPS. Check the silicon before you design around MIG.

On an A100 or H100, the GPU is internally divided into **seven compute slices** (groups of SMs) and **eight memory slices** (each a fraction of HBM plus its share of memory-controller bandwidth). A MIG instance is built from some number of compute slices and memory slices, and NVIDIA names the profiles by that composition: `<compute>g.<memory>gb`. A `1g.10gb` instance is one compute slice and 10 GB of memory; a `3g.40gb` is three compute slices and 40 GB; a `7g.80gb` is the whole card as a single instance. Because there are only seven compute slices, the profiles form a ladder where slice count and per-slice power move in opposite directions — you get seven tiny instances *or* one big one, never both.

The figure shows the ladder for an A100 80GB. Read it as a menu: pick a profile, and the "per GPU" number tells you how many of that instance fit on one card. Seven `1g.10gb` slices, or three `2g.20gb`, or two `3g.40gb`, or one `4g.40gb` (leaving three compute slices that can only host smaller profiles), or the whole `7g.80gb`. You can also mix — a common production layout is one `3g.40gb` for a medium model plus four `1g.10gb` for small ones, as long as the compute-slice budget of seven and the memory budget are not exceeded.

![Grid diagram of the MIG profile ladder for an A100 showing slice counts and per-slice compute fractions from 1g.10gb to 7g.80gb](/imgs/blogs/gpu-scheduling-mig-and-autoscaling-2.webp)

The critical property is that this partitioning is enforced in hardware. Each instance has its own memory address space, its own path to its slice of HBM, and its own compute slices. A runaway allocation in one instance cannot OOM another. A compute-heavy prefill in one instance cannot steal cycles from a latency-sensitive decode in another. This is **guaranteed QoS**: the p99 latency of a tenant on a `1g.10gb` slice is a function of that slice's fixed resources and its own load, not of what the noisy neighbor is doing. For multi-tenant serving — internal platform teams renting slices to product teams, or a SaaS serving many customers' fine-tuned models — this isolation is the entire value proposition.

The cost of that isolation is rigidity. MIG geometry is set at the device level and changing it requires draining the GPU: you cannot resize a `1g.10gb` into a `3g.40gb` while a pod is using it. Enabling MIG mode itself typically requires that no CUDA process is running and often a GPU reset. And each instance is capped at its slice: a `1g.10gb` instance has one-seventh of the compute and roughly one-eighth of the memory bandwidth of the full card, so a single workload that needs the whole GPU's bandwidth (a large model doing high-throughput decode) will be *slower* on any MIG slice than on the unpartitioned card. MIG multiplies the number of isolated workloads; it does not speed up any one of them.

### What each slice actually owns

It is worth being precise about what a MIG instance owns, because the naming can mislead. A `1g.10gb` slice is one of seven compute slices and one memory slice, and those two budgets are quantized separately. The compute slice gives you one-seventh of the SMs — on an A100, roughly 14 of the 108 SMs. The memory slice gives you a fixed 10 GB partition of HBM *and* a dedicated share of the memory-controller channels and L2 cache that feed it, which is why a MIG slice has bounded, predictable bandwidth rather than a contended fraction of the full card's roughly 2 TB/s. That predictability is the point: a slice's throughput is a fixed portion of the whole, known in advance, unaffected by neighbors. A model that is memory-bandwidth-bound — most LLM decode is — runs at close to one-seventh of the whole card's decode throughput on a `1g` slice, and no faster, no matter how idle the rest of the card sits.

The mismatch between seven compute slices and eight memory slices is the detail that surprises people. MIG also reserves memory for instance bookkeeping, so you cannot build seven `1g` instances that each own a full eighth of HBM — the practical profile on an 80 GB A100 is `1g.10gb` (seven of them reach 70 GB, not 80), and NVIDIA also exposes a `1g.20gb` variant that spends a compute slot to double a slice's memory for models that are memory-heavy but compute-light. Always read the profile list on your exact card and driver with `nvidia-smi mig -lgip` before planning a layout: the numbers here are the A100 80GB shape, and they differ on the 40GB A100, on H100 (same seven-slice geometry, HBM3, different per-slice memory), and on H200.

#### Worked example: a mixed layout for a real workload mix

A platform team has three tenants of different sizes to place on one H100 80GB: a 13B chat model that profiles well on three compute slices, and two 3B classifiers that each fit a single slice. MIG lets you carve exactly that shape — one `3g.40gb` plus two `1g.10gb` — as long as both budgets hold. Compute: $3 + 1 + 1 = 5$ of the seven slices used, leaving two idle. Memory: $40 + 10 + 10 = 60$ GB of the roughly 80 GB partitionable, comfortably inside budget.

The two leftover compute slices are the real cost of a mixed layout. They cannot be lent to the `3g.40gb` tenant, whose geometry is fixed at creation, and they sit stranded unless your catalog has another small model to fill them. If it does, add a `2g.20gb` or two more `1g.10gb` and drive the card to full; if it does not, you are running a 5/7-packed card and paying for two idle slices. The lesson is that MIG geometry is a bin-packing problem in two dimensions at once — compute slices and memory — and whatever you fail to pack is stranded until you drain and repartition. Plan the layout around the workload mix you actually run, not the one that looks tidy on a slide.

### Partitioning a GPU from the command line

Here is the actual sequence to enable MIG on GPU 0 and carve it into seven `1g.10gb` slices. This runs on the node, as root, with no CUDA clients active.

```bash
# 1. Enable MIG mode on GPU 0. This may require a GPU reset or a reboot;
#    it fails if any process holds a CUDA context on the device.
sudo nvidia-smi -i 0 -mig 1

# 2. (If prompted that a reset is required.)
sudo nvidia-smi --gpu-reset -i 0

# 3. List the GPU-instance profiles this card supports, with their IDs and
#    how many of each are still placeable given current partitioning.
nvidia-smi mig -lgip

# 4. Create seven 1g.10gb GPU instances and, with -C, a default compute
#    instance inside each. You can pass profile names or the numeric IDs
#    from -lgip (IDs vary by GPU model and driver, so -lgip first).
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb -C

# 5. Verify: seven MIG devices, each with its own UUID and 1/7 of the SMs.
nvidia-smi mig -lgi
nvidia-smi -L   # lists MIG device UUIDs like MIG-xxxx used by CUDA_VISIBLE_DEVICES

# To undo: destroy compute instances, then GPU instances, then disable MIG.
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
sudo nvidia-smi -i 0 -mig 0
```

Once created, each MIG instance appears to CUDA as a separate device with its own UUID. A process pinned to one instance (via `CUDA_VISIBLE_DEVICES=MIG-...`) sees only that slice. In Kubernetes you almost never run these commands by hand — the GPU Operator's MIG Manager applies a declarative geometry from a node label — but you need to understand the primitive to debug it when a node comes up with the wrong partitioning.

### Requesting a MIG slice from a pod

With the NVIDIA device plugin configured for the **mixed** MIG strategy (heterogeneous profiles exposed as distinct resources), a pod asks for a specific slice by name:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: classifier-a
spec:
  replicas: 1
  selector:
    matchLabels: { app: classifier-a }
  template:
    metadata:
      labels: { app: classifier-a }
    spec:
      # Land only on A100 nodes that have been partitioned into 1g.10gb slices.
      nodeSelector:
        nvidia.com/gpu.product: A100-SXM4-80GB
        nvidia.com/mig.strategy: mixed
      containers:
        - name: server
          image: myregistry/classifier:1.4.2
          resources:
            limits:
              # One 1g.10gb MIG instance — a hardware-isolated seventh of the card.
              nvidia.com/mig-1g.10gb: 1
```

The device plugin advertises `nvidia.com/mig-1g.10gb: 7` on a node whose single A100 is carved into seven such slices, and the scheduler treats each slice as an allocatable unit. Seven of these deployments land on one physical GPU, each hardware-isolated. If you had instead chosen the **single** MIG strategy — every MIG device on the node identical, exposed simply as `nvidia.com/gpu` — the pod spec would just say `nvidia.com/gpu: 1` and the plugin would hand out one MIG instance per "GPU". Single strategy is simpler and works well when a node's whole job is to serve one profile; mixed strategy is what you want when a node hosts several profiles at once.

## Time-slicing and MPS: software sharing without isolation

MIG is a hardware feature you may not have. Time-slicing and MPS are software mechanisms that work on *any* CUDA GPU, including the T4s and L40S where MIG is unavailable. They buy you higher utilization by oversubscription, and they do it by giving up isolation.

**Time-slicing** is the bluntest tool. You tell the NVIDIA device plugin to advertise more copies of each physical GPU than exist, and the GPU's hardware scheduler round-robins between the processes that land on it. Each process believes it has a whole GPU; in reality they take turns, context-switching at a coarse granularity. There is no memory isolation — all processes allocate from the same HBM pool, so two 10 GB models on a 16 GB card will OOM, and nothing stops one process from grabbing all the memory. There is no compute QoS — a process running a heavy kernel simply holds the GPU longer, inflating everyone else's latency. Time-slicing is a way to *pack more processes onto a GPU*, not a way to *guarantee any of them anything*.

Configure it with a device-plugin ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  # A named config; nodes select it via a label so different node pools can
  # use different replica counts.
  four-way: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        # Rename the shared resource so users must opt in explicitly and can't
        # confuse a time-sliced "GPU" with an exclusive one.
        renameByDefault: true
        resources:
          - name: nvidia.com/gpu
            replicas: 4          # advertise 4 replicas per physical GPU
```

With `renameByDefault: true`, a node's single physical GPU now advertises `nvidia.com/gpu.shared: 4`. Four pods each requesting `nvidia.com/gpu.shared: 1` land on the one card and take turns. Use this for development clusters, notebooks, CI, or a set of low-traffic models you *trust* not to misbehave — never for a production tenant boundary, because there is no boundary.

**CUDA MPS** is more sophisticated. The MPS control daemon lets multiple processes submit work into a *single, shared* GPU context, so their kernels can execute concurrently on different SMs instead of serially time-slicing. You can bound each client with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (cap the fraction of SMs a client may use) and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` (a soft per-client memory cap). That gives you *soft* compute partitioning and *soft* memory limits — much better than time-slicing for concurrent small-model serving, because you avoid the context-switch overhead and get real parallelism. But there is a sharp edge: because all clients share one CUDA context, a fatal GPU error (an XID, an illegal memory access) in one client can poison the context and take down every client on that GPU. MPS gives you flexibility and concurrency; it does not give you fault isolation. The GPU Operator's device plugin supports MPS sharing, configured much like time-slicing but with `sharing.mps` instead of `sharing.timeSlicing`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mps-config
  namespace: gpu-operator
data:
  mps-4: |-
    version: v1
    sharing:
      mps:
        resources:
          - name: nvidia.com/gpu
            replicas: 4          # 4 concurrent MPS clients per physical GPU
```

The device plugin brings up an MPS control daemon per GPU and advertises the shared resource as `nvidia.com/gpu.shared: 4`. Each of the four clients is then bounded by `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (roughly one-quarter of the SMs) and a soft memory cap, and — the property time-slicing cannot offer — their kernels run *concurrently* on different SMs instead of taking turns. The practical difference matters most for many-small-model serving: four low-traffic models under MPS overlap their work and each sees low added latency, whereas under time-slicing the same four serialize and each pays the others' time-slices. What MPS still does not give you is a fault boundary. Because the four share one CUDA context, an illegal-memory-access XID raised by any one client can poison the context and take the other three down with it. That is the line you cannot cross for a tenant boundary: MPS buys concurrency and soft limits, never isolation.

## Choosing a sharing mode: isolation vs utilization vs flexibility

Now put the four options side by side. The matrix below is the decision surface: sharing mode on the rows, the properties that matter on the columns. Read down the "Memory isolation" and "Fault isolation" columns and one fact jumps out — **only MIG gives hardware isolation on both.** Everything else trades a form of isolation for flexibility or a higher replica count.

![Matrix comparing dedicated, MIG, time-slicing, and MPS across memory isolation, compute QoS, fault isolation, utilization gain, and best-fit workload](/imgs/blogs/gpu-scheduling-mig-and-autoscaling-3.webp)

The same trade-offs in more operational detail:

| Property | Dedicated (1 pod/GPU) | MIG | Time-slicing | MPS |
|---|---|---|---|---|
| Memory isolation | Full (own card) | Hard, hardware | None (shared HBM) | Soft (per-client limit) |
| Compute QoS | Full | Hard, dedicated SMs | None (round-robin) | Soft (thread %) |
| Fault isolation | Full | Hard | None | None (shared context) |
| Hardware support | Any GPU | A30/A100/H100/H200/B200 | Any CUDA GPU | Any CUDA GPU |
| Replicas per card | 1 | Up to 7 (fixed profiles) | Any count you set | Any count you set |
| Reconfigure cost | n/a | Drain + repartition | Change ConfigMap, restart plugin | Restart MPS daemon |
| K8s resource | `nvidia.com/gpu` | `nvidia.com/mig-<profile>` | `nvidia.com/gpu.shared` | `nvidia.com/gpu.shared` |
| Best for | One full-GPU model | Multi-tenant, guaranteed QoS | Dev/test, trusted low-traffic | Many processes, single trust domain |

### The mechanics: effective capacity and cost per request

Let us make "MIG pays for many small models" a provable statement rather than a slogan. Define **effective capacity** as the number of independent workloads a single GPU can serve at acceptable QoS.

For a **dedicated** GPU, effective capacity is 1 by definition, no matter how little of the card the workload uses. If the workload occupies a fraction $f$ of the GPU, utilization is $f$ and you waste $1-f$. For our 3B model, $f \approx 0.12$ and you waste 88%.

For **MIG** with a profile of $p$ compute slices, the number of instances is $n_p = \lfloor 7/p \rfloor$, subject also to the memory budget. If each workload fits in one instance, effective capacity is $n_p$ and aggregate utilization is $n_p$ times the per-slice utilization. Seven `1g.10gb` slices each 80% busy gives aggregate $7 \times \tfrac{1}{7} \times 0.8 = 0.8$, or 80%. The card does the same *total* work it would if one workload used it fully — MIG did not create compute — but it now does that work for **seven tenants with guaranteed isolation** instead of one.

For **time-slicing** with $r$ replicas, effective concurrency is $r$, but with no isolation the binding constraint is memory: $r$ workloads each needing $M$ of HBM require $r \cdot M \le \text{HBM}$ or something OOMs, and there is no mechanism to enforce fairness — one greedy process can starve the rest. So time-sliced effective capacity is $\min\!\big(r,\ \text{HBM}/M\big)$ *with no QoS guarantee attached to any of it.*

Now the money. Let $C$ be the GPU's hourly cost and $\lambda$ a workload's request rate. On a **dedicated** card serving one workload, the amortized cost per request is

$$
c_{\text{dedicated}} = \frac{C}{3600\,\lambda}.
$$

Pack seven identical workloads onto one card with MIG and the same card now serves $7\lambda$ requests per second for the same $C$:

$$
c_{\text{MIG}} = \frac{C}{3600 \cdot 7\lambda} = \frac{1}{7}\,c_{\text{dedicated}}.
$$

The cost per request drops by exactly the packing factor — *provided* each workload still meets its SLA on one-seventh of the compute. That proviso is the whole game. If your model needs more than one-seventh of the card to hit p99, you step up to `2g.20gb` (three per card, 3x savings) or `3g.40gb` (two per card, 2x savings). The savings equal the number of tenants you can pack, and that number is set by the smallest slice your model can tolerate.

#### Worked example: packing seven small models — MIG vs seven GPUs

A platform team hosts seven fine-tuned 3B classifiers for seven product teams. Each is quantized to INT8 (about 3 GB of weights), each peaks at roughly 30 requests per second on short inputs, and each must isolate its tenant — team A's traffic spike must never inflate team B's p99. Weights plus KV cache plus context fit comfortably in 10 GB, and profiling shows a single `1g.10gb` slice sustains ~35 req/s at the target p99 for these short-sequence classification calls.

**Option A — dedicated.** Seven A100 80GB cards, one model each. At a representative \$3.67 per GPU-hour, that is $7 \times \$3.67 = \$25.69$ per hour, and each card runs at roughly 12% utilization. Cost per request, using $c_{\text{dedicated}}$ with $\lambda = 30$: $\$3.67 / (3600 \times 30) \approx \$3.4 \times 10^{-5}$ per request.

**Option B — MIG.** One A100 80GB carved into seven `1g.10gb` slices, one model per slice. Cost is $1 \times \$3.67 = \$3.67$ per hour; aggregate utilization ~85%; each tenant still hardware-isolated. Cost per request: $\$3.67 / (3600 \times 210) \approx \$4.9 \times 10^{-6}$ — seven times cheaper.

The fleet shrinks from seven cards to one, saving about \$22 per hour, or roughly \$16,000 per month for this one workload group, with **no loss of isolation** because MIG's boundary is hardware. The catch, and it is a real one: this only holds because the 3B INT8 model fits its SLA on a `1g.10gb` slice. Re-run the profiling on a larger model and you might find it needs `2g.20gb`, at which point the fleet is one card of three tenants and the saving is 3x, not 7x. Always profile the model *on the slice* before you promise the finance team a 7x.

#### Worked example: the same fleet on time-slicing, and why it is not the same

Take the same seven 3B classifiers and, instead of MIG, put them on one A100 with the device plugin advertising `nvidia.com/gpu.shared: 7`. On paper the packing is identical — seven models, one card, one-seventh of the bill each — and the arithmetic $c_{\text{shared}} = \tfrac{1}{7}\,c_{\text{dedicated}}$ matches MIG's exactly. Three differences decide whether you can actually run it.

First, memory. The seven models need $7 \times 10 = 70$ GB, which fits an 80 GB card — barely. Add one more tenant, or let any tenant's KV cache grow under a long-context request, and the shared HBM pool overflows. Because there is no per-tenant memory partition, the allocation that happens to trip the limit fails, and it may not be the greedy tenant's — it is whoever asks for memory next. MIG refuses the eighth tenant at schedule time; time-slicing accepts it and fails at runtime, unpredictably.

Second, latency under contention. The seven processes round-robin on the hardware scheduler at a coarse quantum. When tenant A fires a burst, it holds the GPU for its full slice and the other six wait; their p99 is now a function of A's behavior, not their own. On MIG each tenant owns its 14 SMs outright and A's burst is invisible to the rest. For a shared internal fleet where a blown p99 is an annoyance, that is tolerable; for a tenant boundary you bill against or promise an SLA on, it is not.

Third, no admission control on compute. Time-slicing has no equivalent of MIG's fixed slice — you set the replica count and hope the models behave. If one classifier is quietly heavier than profiling suggested, it eats more than its seventh of the wall-clock and everyone else silently degrades, with nothing in the metrics naming the culprit until you correlate latencies by hand.

The bottom line: time-slicing reaches the same utilization figure as MIG on paper and the same cost per request, and on a trusted, well-behaved, headroom-to-spare fleet it delivers. It buys that figure by removing every guardrail MIG puts in hardware. Choose it when the tenants trust each other; choose MIG when they must not have to.

## Scheduling GPUs on Kubernetes

Partitioning tells you how a card is divided. Scheduling is how Kubernetes decides which pod lands on which slice of which node. The machinery is the NVIDIA device-plugin ecosystem, and the figure shows the stack a GPU pod depends on. Read it top to bottom: a model-server pod requests a GPU resource; the kube-scheduler places it; but the pod only schedules because the NVIDIA device plugin advertised that resource, and the plugin only runs because the GPU Operator installed it, discovered the hardware, and labeled the node.

![Stack diagram of the Kubernetes GPU serving stack from model-server pods down through scheduler, device plugin, GPU Operator, feature discovery, to the GPU node](/imgs/blogs/gpu-scheduling-mig-and-autoscaling-4.webp)

You do not assemble this by hand. The **NVIDIA GPU Operator** deploys the whole stack as a set of DaemonSets: the driver (or validates a preinstalled one), the container toolkit that makes GPUs visible inside containers, the **device plugin** that advertises `nvidia.com/gpu` and `nvidia.com/mig-*` resources, **DCGM** and its exporter for GPU metrics, **GPU Feature Discovery (GFD)** which labels nodes with facts like `nvidia.com/gpu.product` and `nvidia.com/mig.strategy`, and the **MIG Manager** which applies a declarative MIG geometry when you set a node label. Installing the operator and labeling a node `nvidia.com/mig.config=all-1g.10gb` is enough to bring the node up partitioned into seven slices and advertising them — the operator drains, repartitions, and re-labels for you.

### Bin-packing versus spread, and why cost wants bin-packing

The scheduler's default scoring spreads pods across nodes for resilience (`LeastAllocated` in the `NodeResourcesFit` plugin). For GPUs, that default costs you money. If you have four GPU nodes and eight GPU pods, spreading puts two pods on each of four nodes — and now the cluster autoscaler can never remove a node, because every node has work on it. Bin-packing instead fills one node before opening the next, so idle nodes end up completely empty and can be scaled down. You enable it with a scheduler configuration:

```yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
  - schedulerName: default-scheduler
    pluginConfig:
      - name: NodeResourcesFit
        args:
          scoringStrategy:
            type: MostAllocated        # bin-pack: prefer the fullest node
            resources:
              - name: nvidia.com/gpu
                weight: 100             # weight GPU packing above CPU/memory
```

`MostAllocated` with a heavy weight on `nvidia.com/gpu` tells the scheduler to prefer the node that is already the most GPU-loaded, consolidating work so empty nodes can be reclaimed. Karpenter, if you use it instead of the Cluster Autoscaler, does this consolidation natively — it continuously repacks pods onto fewer nodes and terminates the emptied ones. Bin-packing is the scheduling half of cost control; scale-to-zero, later, is the other half.

Bin-packing also fights *fragmentation*, the GPU-scheduling failure that quietly strands capacity. Suppose eight GPUs are free but spread one-per-node across eight nodes, and a replica needs two GPUs on one node for NVLink-connected tensor parallelism. There are eight free GPUs and the replica still cannot schedule — the free capacity is fragmented into unusable single-GPU holes. `MostAllocated` scoring reduces this by keeping partially filled nodes together, but the durable fix on an autoscaled cluster is Karpenter's consolidation, which actively repacks. A GPU NodePool that consolidates underused nodes looks like this:

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-serving
spec:
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 5m        # repack and drain underused GPU nodes
  template:
    spec:
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["p4d.24xlarge", "p5.48xlarge"]   # 8-GPU nodes
      taints:
        - key: nvidia.com/gpu
          effect: NoSchedule      # only GPU pods (with the toleration) land here
```

The taint keeps non-GPU pods off expensive GPU nodes, so a stray CPU workload never pins a node open and blocks it from draining; `consolidateAfter` gives Karpenter permission to move a lonely pod onto another node and terminate the emptied one. Fragmentation is worst exactly where GPUs are most expensive — multi-GPU nodes with NVLink — so consolidation there is not a nicety, it is how you keep the fleet schedulable at all.

### Gang scheduling for multi-GPU replicas

Everything above assumes a replica fits on one GPU. The moment a single model replica needs *several* GPUs — tensor parallelism across eight GPUs on a node, or a replica spanning nodes (covered in [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/choosing-your-serving-stack)) — the default scheduler becomes a liability. It places pods one at a time. If a replica needs eight GPUs and only five are free, the scheduler may bind five pods, leave three Pending, and *deadlock*: the five hold GPUs waiting for the three, the three wait for GPUs the five are holding, and nothing progresses. Multiply that across several multi-GPU replicas and the cluster gridlocks with everything half-placed.

The fix is **gang scheduling**: place all pods of a group atomically, or none of them. You get it from a batch-aware scheduler — Volcano, Kueue, or the Kubernetes scheduler-plugins coscheduling package. With Volcano you declare a `PodGroup` with a `minMember`:

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: llm-70b-tp8
spec:
  minMember: 8              # all 8 GPU workers, or none — no partial placement
  queue: serving
  minResources:
    nvidia.com/gpu: 8
```

The scheduler holds the group in a Pending "gang" until eight GPUs can be committed at once, then binds all eight together. No partial placement, no deadlock. For multi-node LLM replicas you typically combine gang scheduling with a workload API like LeaderWorkerSet so the leader and workers of one replica are scheduled and restarted as a unit. The pairing looks like this — a LeaderWorkerSet defines one leader and $N-1$ workers as a single replicated unit, and it schedules through Volcano so the whole group places atomically:

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: llm-70b
spec:
  replicas: 2                   # 2 independent serving replicas
  leaderWorkerTemplate:
    size: 8                     # 1 leader + 7 workers = 8 GPUs per replica
    workerTemplate:
      spec:
        schedulerName: volcano  # gang-schedule the whole group atomically
        containers:
          - name: worker
            resources:
              limits: { nvidia.com/gpu: 1 }
```

Two more constraints bite at this scale. Placement must be *topology-aware*: the eight workers of a tensor-parallel replica should land on GPUs joined by NVLink or NVSwitch, not scattered across nodes where the interconnect is slow PCIe or Ethernet — a tensor-parallel all-reduce runs on every token and is punishingly sensitive to link bandwidth. You express that with node affinity on an NVLink-domain label, or by pinning each replica to a single 8-GPU node. And restart semantics must be all-or-nothing too: if one worker crashes, the whole replica is dead until it rejoins, so the workload API must restart the group as a unit rather than leaving seven workers waiting on a missing eighth. Gang scheduling gets the replica *placed*; topology awareness and group restart keep it *fast* and *alive*.

## Autoscaling model servers: the right signals

Partitioning packs the fleet; autoscaling makes the fleet track load. And here is where most teams make their first serious mistake: **they scale on CPU utilization.** For a GPU inference server, CPU is nearly idle — the CPU marshals requests and copies tensors while the GPU does the work — so CPU-based autoscaling either never triggers or triggers on noise. The Horizontal Pod Autoscaler (HPA) default target of CPU is exactly wrong for GPU serving.

Scale on a signal that reflects *how loaded the GPU actually is.* The good signals, in rough order of usefulness for LLM serving:

| Signal | Reflects load? | Why / caveat |
|---|---|---|
| CPU utilization | No | GPU-bound server; CPU stays low. The default trap. |
| GPU utilization (DCGM) | Partially | Saturates at 100% well before the server is actually overloaded; a batched server can be at 100% GPU and still have headroom, or at 100% and badly backed up. Coarse. |
| Requests in flight (running) | Yes | Directly measures concurrency the server is handling; scales with real demand. |
| Queue depth (requests waiting) | Yes | The clearest overload signal — requests are waiting because capacity is exhausted. |
| KV-cache utilization | Yes (LLM) | For LLMs, KV memory is the binding resource; near 100% means the scheduler is about to preempt or reject. Excellent early-warning signal. |
| TTFT / p99 latency | Yes, but lagging | Scaling on the SLO metric itself reacts *after* the SLO is already degrading. Use as a guardrail, not the primary trigger. |

vLLM exports exactly the metrics you want on its Prometheus endpoint: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`, and `vllm:time_to_first_token_seconds`. Those are your autoscaling inputs.

The figure shows the decision flow of a GPU-aware autoscaler. Metrics — queue depth, KV utilization, request rate — feed the autoscaler, which branches: scale up by adding pods when load rises, and if there is no free GPU for the new pods, trigger the cluster autoscaler to add a node; or scale to zero when the service has been idle past a cooldown, accepting that the next request pays a cold start. Each branch has a distinct latency cost written on it, and those costs are the subject of the next two sections.

![Graph diagram of the autoscaler decision flow branching from metrics through KEDA or HPA to scale-up, add-node, and scale-to-zero paths](/imgs/blogs/gpu-scheduling-mig-and-autoscaling-5.webp)

#### Worked example: why GPU utilization lies, and what to target instead

DCGM reports `DCGM_FI_DEV_GPU_UTIL`, the fraction of the last sample window in which at least one kernel was resident. It is the metric every dashboard shows and the wrong one to autoscale on, for a specific reason: on a continuously batched LLM server the GPU is running *some* kernel essentially all the time even at trivial load, so `GPU_UTIL` pins near 100% whether the server is handling two requests or two hundred. It cannot distinguish "busy" from "saturated," and it is the two hundred queued requests, not the 100%, that blow your TTFT.

Watch what each signal reports as one vLLM replica on an A100 fills up. At 8 concurrent requests the GPU shows 96% utilization, the running-requests gauge reads 8, the waiting queue is 0, and KV-cache usage is 35% — plenty of headroom, but `GPU_UTIL` already looks maxed. At 40 concurrent the utilization is still 96%, running reads 40, waiting is still 0, KV usage is 82%. At 64 concurrent the utilization is *unchanged at 96%*, but now KV cache sits at 99%, the scheduler starts refusing to admit new sequences, and `num_requests_waiting` climbs past 20. The only signals that moved monotonically with real load were running-requests, waiting, and KV usage; `GPU_UTIL` was flat and useless across the entire range.

So the target you set on the HPA is a concurrency number read off that curve, not a utilization percentage. Profile the replica to find the concurrency at which p99 first touches your SLO — say 48 in-flight — then set the HPA target below it with margin: `averageValue: 24` runs each replica at half the danger point, leaving room for the roughly 60 seconds a new replica takes to come up. Set the target too high and every scale-up starts from an already-overloaded replica; set it too low and you pay for idle capacity. The number is a knob on the cost-versus-tail-latency trade, and it belongs to profiling, not to a default.

### HPA on a custom metric

To scale on requests-in-flight, you expose that metric to Kubernetes through the **prometheus-adapter**, which serves Prometheus queries on the `custom.metrics.k8s.io` API the HPA reads. First the adapter rule that turns a vLLM metric into a per-pod custom metric:

```yaml
# prometheus-adapter values: expose vllm:num_requests_running as a pod metric.
rules:
  - seriesQuery: 'vllm:num_requests_running{namespace!="",pod!=""}'
    resources:
      overrides:
        namespace: { resource: namespace }
        pod: { resource: pod }
    name:
      matches: "vllm:num_requests_running"
      as: "vllm_requests_running"
    metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

Then the HPA that targets an average of 24 in-flight requests per pod, with scale-up allowed to react instantly and scale-down damped by a five-minute stabilization window so a brief dip does not thrash the fleet:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm
  minReplicas: 2
  maxReplicas: 16
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_requests_running     # from prometheus-adapter
        target:
          type: AverageValue
          averageValue: "24"              # target ~24 concurrent req/pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0       # add capacity immediately
      policies:
        - type: Pods
          value: 4                        # up to +4 pods per 30s
          periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300     # wait 5 min of low load before removing
      policies:
        - type: Pods
          value: 1
          periodSeconds: 60
```

The HPA computes desired replicas from a simple ratio. Letting $N_{\text{cur}}$ be current replicas, $m_{\text{cur}}$ the current average metric, and $m_{\text{target}}$ the target:

$$
N_{\text{desired}} = \Big\lceil N_{\text{cur}} \cdot \frac{m_{\text{cur}}}{m_{\text{target}}} \Big\rceil.
$$

Ten pods averaging 30 in-flight against a target of 24 gives $\lceil 10 \times 30/24 \rceil = 13$ replicas. The asymmetric `behavior` — instant scale-up, slow scale-down — is deliberate: under-provisioning hurts users immediately (queued requests, blown TTFT), while over-provisioning only costs money, so you react fast to load and retreat slowly.

## The cold-start problem

Scaling up sounds instant on a slide. It is not. When the autoscaler decides to add a replica, the new pod must be scheduled, possibly onto a brand-new GPU node, pull a multi-gigabyte container image, load multi-gigabyte weights into HBM, and warm up CUDA before it can serve a single token. The figure lays this out as a timeline — each stage adds tens of seconds, and they run in series.

![Timeline diagram of the cold-start sequence from scale event through node boot, image pull, weight load, and CUDA warmup to first token near T plus 110 seconds](/imgs/blogs/gpu-scheduling-mig-and-autoscaling-6.webp)

### The mechanics: a cold-start latency budget

Model total cold-start time as the sum of its stages, because each is independently attackable:

$$
T_{\text{cold}} = T_{\text{detect}} + T_{\text{node}} + T_{\text{pull}} + T_{\text{load}} + T_{\text{warm}}.
$$

- $T_{\text{detect}}$ — time for the autoscaler to notice load and decide. Bounded by the polling interval (KEDA polls every 30 s by default; HPA syncs every 15 s). Mitigation: shorter intervals, or push metrics.
- $T_{\text{node}}$ — time to provision a GPU node if none is free. This is the killer: a fresh cloud GPU node must boot, install or load the NVIDIA driver, and join the cluster — 60 to 300 seconds. Mitigation: keep warm nodes (over-provision headroom) so this term is zero.
- $T_{\text{pull}}$ — time to pull the container image. An image is $\text{size} / \text{bandwidth}$; an 8 GB vLLM image at 500 MB/s is 16 s. Mitigation: pre-pull the image to nodes with a DaemonSet, use a node-local image cache, or use image streaming (lazy layer pull) so the container starts before the whole image lands.
- $T_{\text{load}}$ — time to read weights from storage into HBM. 14 GB of FP16 weights (a 7B model) from local NVMe at 3 GB/s is under 5 s; from a remote volume at 500 MB/s it is 28 s. Mitigation: cache weights on node-local NVMe, use a fast weight streamer, keep the OS page cache warm.
- $T_{\text{warm}}$ — CUDA context init plus warmup: capturing CUDA graphs, compiling kernels. vLLM's CUDA-graph capture is tens of seconds; a `torch.compile` warmup can run minutes. Mitigation: keep replicas warm, snapshot a warmed process, or skip compilation on the cold path.

The reason this matters is that the terms are *additive and serial*, so a cold start with nothing pre-warmed easily runs into minutes, while a well-warmed one is tens of seconds. Which regime you are in decides whether scale-to-zero is viable for a given SLA.

#### Worked example: cold-start budget for a spiky 7B chatbot

A 7B FP16 model (14 GB weights) served with vLLM (8 GB image). Two regimes:

**Cold, nothing pre-warmed.** $T_{\text{detect}} = 15\text{ s}$ (KEDA poll) $+\ T_{\text{node}} = 90\text{ s}$ (new GPU node boots and loads the driver) $+\ T_{\text{pull}} = 16\text{ s}$ (8 GB at 500 MB/s) $+\ T_{\text{load}} = 28\text{ s}$ (14 GB from a remote PVC at 500 MB/s) $+\ T_{\text{warm}} = 25\text{ s}$ (CUDA-graph capture) $=\ 174\text{ s}$, roughly **2.9 minutes** to first token.

**Warm, everything pre-staged.** A warm GPU node already in the pool ($T_{\text{node}} = 0$), the image already cached ($T_{\text{pull}} \approx 1\text{ s}$), weights on local NVMe (14 GB at 3 GB/s $= 4.7\text{ s}$), CUDA graphs pre-captured on a kept-warm process ($T_{\text{warm}} \approx 5\text{ s}$), and metrics pushed rather than polled ($T_{\text{detect}} \approx 5\text{ s}$) $\approx\ \textbf{16 s}$.

If the product promises "first token within 30 seconds of a cold request," only the warm path qualifies; the cold path violates it by nearly 6x. That single number decides your architecture: to offer scale-to-zero on this service you *must* keep a warm node and cached image, or you must front the service with a request buffer (below) and accept a multi-minute wait on the very first request after idle.

### Mitigating cold starts

The mitigation table maps each stage of the budget to the technique that zeroes it:

| Technique | Attacks | How |
|---|---|---|
| Warm pool / headroom | $T_{\text{node}}$ | Keep N idle-but-ready GPU nodes (or a spare replica) so a scale-up schedules onto existing capacity instantly. |
| Over-provisioning with pause pods | $T_{\text{node}}$ | Low-priority placeholder pods reserve node capacity; real pods preempt them and schedule at once while the autoscaler backfills the node in the background. |
| Image pre-pull / streaming | $T_{\text{pull}}$ | DaemonSet pulls the image to every GPU node ahead of time; or lazy image streaming starts the container before the full image is local. |
| Local weight cache | $T_{\text{load}}$ | Weights on node-local NVMe or a shared read-cache instead of pulled from object storage per start. |
| Fast weight streamer | $T_{\text{load}}$ | Stream weights storage-to-HBM in parallel with high bandwidth rather than a serial file read. |
| Warm process / snapshot | $T_{\text{warm}}$ | Keep a warmed process (weights loaded, CUDA graphs captured); or snapshot a warmed process and restore it, skipping init. |

Two of these deserve their own code. The warm pool via over-provisioning uses a negative-priority PriorityClass and a placeholder Deployment that does nothing but hold GPU capacity:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: overprovisioning
value: -10                 # lower than any real workload; preemptible
globalDefault: false
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-headroom
spec:
  replicas: 2              # reserve 2 GPUs of warm, ready capacity
  selector: { matchLabels: { app: gpu-headroom } }
  template:
    metadata:
      labels: { app: gpu-headroom }
    spec:
      priorityClassName: overprovisioning
      terminationGracePeriodSeconds: 0    # evict instantly when preempted
      containers:
        - name: pause
          image: registry.k8s.io/pause:3.9
          resources:
            limits:
              nvidia.com/gpu: 1
```

When a real vLLM pod needs a GPU, the scheduler preempts a `pause` pod and the real pod schedules immediately onto the already-warm node; the cluster autoscaler then adds a node to restore the headroom, in the background, off the critical path. And the local weight cache: mount a node-local hostPath (or a fast read-only shared volume) and have an init container populate it once so subsequent starts read weights locally:

```yaml
      initContainers:
        - name: fetch-weights
          image: myregistry/weight-fetcher:latest
          command: ["/bin/sh", "-c"]
          args:
            - |
              # Only download if not already cached on this node's NVMe.
              if [ ! -f /cache/model/config.json ]; then
                aws s3 sync s3://models/llama-7b /cache/model --no-progress
              fi
          volumeMounts:
            - { name: weight-cache, mountPath: /cache/model }
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args: ["--model", "/cache/model", "--enable-prefix-caching"]
          volumeMounts:
            - { name: weight-cache, mountPath: /cache/model }
          resources:
            limits: { nvidia.com/gpu: 1 }
      volumes:
        - name: weight-cache
          hostPath: { path: /mnt/nvme/model-cache, type: DirectoryOrCreate }
```

The first pod on a node pays the download; every subsequent pod on that node reads 14 GB from local NVMe in seconds. Combined with a warm node pool and a pre-pulled image, this is what turns the 2.9-minute cold start into the 16-second warm start from the worked example.

One subtlety ties the cold-start budget back to the autoscaler. The HPA `scaleUp` policy earlier allowed up to four pods every 30 seconds — but if each pod takes 90 seconds to become ready from cold, the policy's period is shorter than the pods' readiness time, and the autoscaler keeps issuing scale-up decisions against pods that have not finished starting. Without a readiness gate this over-shoots: you ask for four, they are slow, load is still high at the next tick, you ask for four more, and now twelve pods are booting for a burst that needed six. The guard is to make a pod report `Ready` only when it can actually serve a token — a readiness probe that hits the model endpoint, not just the process liveness — so the HPA counts warming pods as not-yet-capacity and does not double-order. Cold start is not only a latency tax on the first request; it is a stability property of the whole control loop.

The frontier mitigation for $T_{\text{warm}}$ is process snapshotting. Tooling that checkpoints a fully warmed server — weights resident in HBM, CUDA context built, CUDA graphs captured, kernels compiled — and restores that image on wake can collapse the tens-of-seconds warmup into a second or two, because restore skips every initialization step and simply re-maps memory. The catch is that a GPU-process snapshot must capture device memory and CUDA state, which is fragile across driver versions and not yet turnkey on most stacks, so today it is used mainly by teams with the scale to justify the engineering. For everyone else the reliable win remains the boring stack from the mitigation table: a warm node, a pre-pulled image, a local weight cache, and a kept-warm replica. Those four are available to any team, and they are what turn the 2.9-minute cold start into 16 seconds.

## Scaling to zero and the cost of idle

The last lever is the most aggressive: scale a service all the way to **zero** replicas when it is idle, and pay nothing for its GPUs until the next request. For spiky or long-tail workloads — an internal tool used during business hours, a rarely-hit model in a large catalog, a demo environment — this is the difference between a large bill and almost none. The figure contrasts the two regimes.

![Before and after diagram contrasting an always-on 8-GPU fleet at 30 percent utilization against a scale-to-zero deployment with one warm spare](/imgs/blogs/gpu-scheduling-mig-and-autoscaling-7.webp)

On the left, a static fleet of eight GPUs running 24x7 at 30% average utilization: nights and weekends idle, but every hour billed, so roughly 70% of the spend is burned on idle capacity. On the right, KEDA scaling the deployment between zero and eight replicas with a single warm spare kept for cold-start protection: capacity tracks load, utilization rises toward 80%, and total GPU spend drops by more than half. The trade you are making is explicit — you exchange a chunk of the bill for a cold-start penalty on the first request after an idle period, mitigated (but not eliminated) by the warm spare.

### KEDA and scale-to-zero

The built-in HPA cannot scale to zero without an alpha feature gate, and even then it needs custom metrics. **KEDA** (Kubernetes Event-Driven Autoscaling) is built for this: it manages the HPA for the 1-to-N range and handles the 0-to-1 activation itself, watching a trigger metric and spinning up the first replica when the metric crosses an activation threshold. A ScaledObject for a vLLM deployment that scales to zero after five idle minutes:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm                    # the Deployment to scale
  minReplicaCount: 0              # scale all the way to zero when idle
  maxReplicaCount: 8
  cooldownPeriod: 300             # wait 5 idle minutes before scaling to 0
  pollingInterval: 15            # check the trigger every 15s
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: vllm_pending
        # Scale on queued + running requests; any waiting work keeps it alive.
        query: |
          sum(vllm:num_requests_waiting{app="vllm"})
            + sum(vllm:num_requests_running{app="vllm"})
        threshold: "24"           # ~24 in-flight per replica → add a replica
        activationThreshold: "1"  # any request at all → scale 0 → 1
```

The distinction between `threshold` and `activationThreshold` is the crux of scale-to-zero. `threshold` drives the normal 1-to-N scaling (24 in-flight per replica, matching the HPA target above). `activationThreshold` is the "wake up from zero" trigger: the moment even one request is waiting, KEDA scales from 0 to 1. But that first request still has to *survive* the cold start — KEDA does not buffer it. If the client's timeout is shorter than $T_{\text{cold}}$, the request fails. This is why scale-to-zero for latency-sensitive services is usually paired with a request buffer: KServe's serverless mode runs on Knative, whose **activator** holds the incoming request while the pod cold-starts and then proxies it, so the client waits rather than fails. You still pay the cold-start latency; you just do not drop the request.

#### Worked example: when does scale-to-zero actually pay?

Scale-to-zero is not free even when idle — it costs a cold start every time the service wakes, and if the service wakes often, you pay that penalty repeatedly for little idle saving. Put it on a scale. An internal tool backed by one A100 at \$3.67 per hour serves traffic only during business hours — nine active hours on weekdays, idle the other 15 hours plus weekends. Always-on, that is a full 168-hour week billed, about \$616 per week. Scaled to zero outside the nine-to-five, it bills roughly $9 \times 5 = 45$ hours, about \$165 per week — a \$451 weekly saving, more than 70%, for a service nobody uses at night.

Now the cost side of the ledger: cold starts. If the tool wakes once each morning, that is five cold starts a week, each a one-time tens-of-seconds wait for the first user — trivially worth \$451. Contrast a *different* service that goes idle for five minutes at a stretch all day and wakes forty times a day: it saves only the scattered idle minutes (little, because it is rarely idle for long) while inflicting forty cold-start waits a day on real users. The break-even is a ratio: scale-to-zero pays when the idle stretches are long relative to the cold-start penalty and infrequent relative to traffic. A rough rule that holds up in practice: scale to zero when the expected idle period between requests is at least an order of magnitude longer than $T_{\text{cold}}$, *and* a cold-start wait on the first post-idle request stays inside the SLA. The morning-only tool clears both bars by a mile; the every-five-minutes service clears neither and should keep a warm floor of one replica instead.

### The mechanics: HPA headroom for an SLO

Scaling reactively is not enough on its own, because scaling takes time and traffic does not wait. You need to reason about how much *standing headroom* to keep so a burst does not blow your SLO during the seconds it takes to scale. Model each replica as a queue with service capacity $\mu$ requests per second, and keep per-replica utilization $\rho = \lambda_{\text{per}} / \mu$ below a ceiling $\rho_{\max}$ so queueing stays bounded (as request rate approaches capacity, waiting time blows up — the same [Little's Law](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) reasoning behind batching). For total arrival rate $\lambda$ over $N$ replicas, $\lambda_{\text{per}} = \lambda/N$, so the replica count you need is

$$
N \ge \frac{\lambda}{\rho_{\max}\,\mu}.
$$

#### Worked example: headroom for a 200 ms p99 SLO

A vLLM replica on one A100 sustains $\mu = 40$ short chat completions per second while keeping p99 under 200 ms, and you hold $\rho_{\max} = 0.7$ to keep queueing tame. Incoming traffic is $\lambda = 300$ req/s.

Steady-state replicas: $N \ge 300 / (0.7 \times 40) = 300/28 = 10.7 \to \textbf{11 replicas}$. That is what the HPA converges to under steady load.

Now a burst: traffic jumps by $\Delta\lambda = 150$ req/s. The autoscaler needs about 60 seconds to react and bring warm replicas up (15 s metric scrape + 15 s HPA sync + ~30 s warm pod start). During those 60 s the extra load has nowhere to go, so the queue accumulates $\Delta\lambda \times T_{\text{scale}} = 150 \times 60 = 9{,}000$ requests — a catastrophic backlog that will blow p99 long before the new replicas arrive. Two ways to absorb it, both costing money:

1. **Standing spare replicas.** Keep enough warm replicas to serve the burst instantly: $\Delta\lambda / \mu = 150/40 \approx 4$ spare replicas. They cost four idle GPUs but the burst is absorbed with zero queue growth.
2. **Lower the target utilization.** Run at $\rho_{\max} = 0.5$ instead of 0.7, which raises the steady count to $N = 300/(0.5 \times 40) = 15$ replicas — four extra, functioning as distributed headroom.

Either way, the lesson is the same: **reactive autoscaling alone cannot meet a tight SLO under bursty load; you must carry headroom.** How much is a direct trade of cost against burst tolerance, and it is a business decision, not a technical one. The finance team's "22% utilization" complaint and the product team's "never make a customer wait" requirement meet exactly here, and the right answer is a deliberately chosen $\rho_{\max}$ and spare count, not an accident.

### The cost picture

Put the two cost levers together. Bin-packing plus partitioning raises the *numerator* of utilization — more useful work per GPU-hour. Scale-to-zero plus tight headroom lowers the *denominator* — fewer GPU-hours billed for idle. A useful way to hold the whole thing in your head:

$$
\text{monthly cost} \approx C \times \text{(GPU-hours billed)} = C \times \sum_{\text{replicas}} \text{hours each replica is alive}.
$$

MIG and time-slicing cut the number of physical GPUs you need for a given set of workloads. Scale-to-zero cuts the hours each replica is alive. Headroom and warm pools add hours back, deliberately, to protect latency. The art is spending the fewest GPU-hours that still meets the SLO — and that is why you cannot separate the cost conversation from the latency conversation. They are two ends of the same lever.

#### Worked example: both levers on one platform bill

Bring the two halves together on one fleet. A platform runs 40 A100s at \$3.67 per hour — the fleet from the opening complaint, about \$107,000 per month if every card is billed 24x7. Audit it and the 40 split into three groups. Twenty cards host many small models at 12% utilization; seven small models fit per card under MIG, so twenty cards of small models collapse to roughly three cards' worth of slices — call it four cards with headroom. Twelve cards run a spiky daytime LLM service at 30% average utilization; scale-to-zero with a one-replica warm floor and bin-packing tracks that load down to an average of maybe five cards billed instead of twelve. Eight cards run a genuinely full-GPU model at high utilization and stay exactly as they are — the discipline is knowing not to touch them.

The fleet goes from 40 always-on cards to about $4 + 5 + 8 = 17$ average billed, a bit over \$45,000 per month, without dropping a single SLO — because none of the moves touched a workload that needed its whole card, and each move was chosen by the same question: is this waste *memory* (pack it with MIG), *idle time* (scale it toward zero), or *neither* (leave it alone)? The 22% utilization the finance team complained about was never one problem. It was three, and each has its own lever.

## Case studies

**NVIDIA MIG for multi-tenant inference.** NVIDIA's MIG documentation and Triton Inference Server guidance describe the canonical pattern this post is built on: an A100 or H100 partitioned into up to seven instances, each running an independent Triton (or other) model server, each with hardware-guaranteed memory and compute isolation. NVIDIA's own materials emphasize that MIG delivers *predictable* QoS — a tenant's latency depends only on its own slice — which is precisely why cloud providers expose MIG-backed offerings and why it is the default recommendation for platform teams renting GPU slices internally. The numbers to take away are structural, not benchmark-specific: seven isolated tenants per card, each capped at one-seventh of the compute, with a hardware boundary between them.

**Managed MIG on the hyperscalers.** Google Kubernetes Engine supports GPU partitioning directly: you request a partition size (for example `cloud.google.com/gke-gpu-partition-size: 1g.10gb`) on an A100 or H100 node pool, and GKE brings the nodes up partitioned and schedulable as MIG resources. Amazon EKS and Azure AKS support MIG through the NVIDIA GPU Operator in the same way this post describes. The practical lesson from these managed offerings is that MIG geometry is a *node-pool-level* decision — you generally dedicate a node pool to a profile rather than repartitioning live — which reinforces that MIG is for stable, known workload mixes, not for moment-to-moment flexibility.

**KEDA and Knative scale-to-zero for LLMs.** The KServe project's serverless deployment mode runs inference services on Knative, which provides scale-to-zero out of the box: when a service is idle, Knative scales it to zero pods, and its **activator** component buffers the next incoming request while a pod cold-starts, then forwards it. This is the reference implementation of "scale to zero without dropping the first request," and it is widely used for long-tail model catalogs where most models are rarely hit. KEDA offers the same scale-to-zero on plain Deployments with richer trigger sources (any Prometheus query, message-queue depth, and dozens of other scalers). The honest caveat both projects document: scale-to-zero trades the idle bill for cold-start latency, and it is only appropriate where the SLA tolerates a wait — which is exactly the decision the cold-start budget above is meant to make quantitative.

**Cold-start mitigation in practice.** Teams running scale-to-zero LLM services converge on the same stack of mitigations covered above: pre-pulled or streamed container images to eliminate the pull, node-local weight caches (or dedicated fast weight streamers) to cut the multi-gigabyte load, and warm node pools with over-provisioning pause pods to remove node-provisioning from the critical path. Some also exploit process-level tricks — keeping a warmed server process resident with weights loaded and CUDA graphs captured, so waking it is milliseconds rather than a full re-initialization. The consistent reported result is the one the budget predicts: without these, cold starts run into minutes; with all of them, first token lands in the low tens of seconds. There is no single trick — the wins are additive across the stages of $T_{\text{cold}}$.

**Bin-packing and consolidation for fleet cost.** The scheduling half of the story shows up in every large GPU platform's cost work: the move from spread scheduling to bin-packing plus autoscaler consolidation is what lets idle nodes empty out and be reclaimed. Teams that adopt Karpenter or the Cluster Autoscaler with `MostAllocated` scoring report the same shape of result — average fleet utilization rises and node count falls at equal served load — because the scheduler stops smearing pods thinly across nodes that can then never drain. The structural takeaway matches the math above: utilization is a numerator-and-denominator problem, and bin-packing attacks the denominator by making idle capacity *collectible* rather than stranded.

**Autoscaling on the right signal, in production.** The consistent report from teams running vLLM behind KEDA or a custom-metric HPA is that switching the trigger from CPU or raw GPU utilization to queue depth and KV-cache usage is the single change that makes autoscaling track real load. The vLLM project exposes `num_requests_waiting` and `gpu_cache_usage_perc` precisely because those are the signals operators asked for, and the LLM control planes built on vLLM — KServe, AIBrix, and the others in [this series](/blog/machine-learning/model-serving/llm-control-planes-aibrix-kserve) — wire them into their autoscalers by default. The lesson is not a specific number but a direction: the metric has to reflect the binding resource, and for LLM decode that resource is KV-cache memory, not a compute-time percentage.

## When to use this (and when not to)

The decision tree below routes a workload to the right strategy with three questions. Follow it top to bottom.

![Decision tree diagram routing workloads by three questions to dedicated GPUs, MIG slices, time-slicing or MPS, and KEDA scale-to-zero](/imgs/blogs/gpu-scheduling-mig-and-autoscaling-8.webp)

**Use MIG when** you have many small-to-medium models that each fit in a slice and you need guaranteed isolation between tenants — a multi-tenant platform, a SaaS serving customers' fine-tuned models, or any case where one tenant's spike must never touch another's p99. MIG is the only option that gives you a *hardware* tenant boundary.

**Do not MIG a model that needs a full GPU.** This is the most common mistake. A `1g.10gb` slice has one-seventh of the compute and one-eighth of the memory bandwidth; a large model doing high-throughput decode will be dramatically *slower* on any slice than on the whole card. MIG multiplies isolated tenants; it never accelerates one workload. If your model wants the whole GPU, give it the whole GPU (and use tensor parallelism across several whole GPUs if it needs more than one).

**Use time-slicing or MPS when** MIG is unavailable (L4, L40S, T4, consumer GPUs) or when the co-tenants are within a single trust domain that you control — a dev cluster, CI, notebooks, or a bundle of your own low-traffic models where a noisy neighbor is an annoyance, not an SLA breach. MPS over time-slicing when you want real concurrency and can cap SM fractions; time-slicing when you just need to pack more processes and do not care about interference.

**Never put a production tenant boundary on time-slicing or MPS.** Neither gives memory isolation, and MPS shares a CUDA context so one client's fatal error takes down the rest. If two workloads must be isolated for correctness, cost-accounting, or SLA reasons, that boundary belongs in hardware (MIG) or on separate GPUs.

**Use scale-to-zero when** the workload is genuinely spiky or long-tail and the SLA tolerates a cold start on the first request after idle — internal tools, rarely-hit catalog models, demo and staging environments. Pair it with a request buffer (Knative activator) so the first request waits rather than fails, and with the cold-start mitigations so the wait is tens of seconds, not minutes.

**Do not scale to zero a latency-critical, always-on service.** A user-facing chatbot with a strict TTFT SLO cannot afford a cold start on the first request of the morning. For those, keep a warm floor (`minReplicas ≥ 1`, or a warm spare), autoscale on queue depth and KV utilization for the peaks, and carry deliberate headroom for bursts. Scale-*down* aggressively at night if you must, but not to zero — keep enough warm capacity to serve the first request within the SLO.

**Always bin-pack GPU pods and always scale on GPU-load signals, not CPU.** These two are close to universal. Bin-packing lets idle nodes empty out so the autoscaler can reclaim them; scaling on queue depth, requests-in-flight, and KV-cache utilization is the only way the autoscaler sees the load that actually matters. CPU-based autoscaling of a GPU server is simply a bug.

## Key takeaways

- **A GPU is expensive and indivisible by default.** A small model on a full card wastes most of the silicon. Partitioning and sharing exist to reclaim that waste; autoscaling exists to stop paying for idle.
- **MIG is the only sharing mode with hardware isolation.** Dedicated SMs, dedicated HBM slices, dedicated bandwidth, guaranteed QoS — but fixed profiles, a reconfiguration that requires a drain, and no acceleration of any single workload. It is for many isolated small-to-medium tenants on A30/A100/H100/H200/B200.
- **Time-slicing and MPS trade isolation for flexibility.** They work on any GPU and pack any number of processes, but neither isolates memory and MPS shares a fault domain. Use them only within a trust boundary you control — never as a production tenant boundary.
- **The MIG payoff equals the packing factor.** Seven tenants on one card is a 7x cost reduction *if* each meets its SLA on one-seventh of the compute. Profile the model on the slice before you promise the saving; a model that needs `2g.20gb` gives 3x, not 7x.
- **Scale on GPU-load signals, not CPU.** Requests-in-flight, queue depth, and KV-cache utilization reflect real load; CPU stays idle on a GPU server. Use prometheus-adapter to feed the HPA, or KEDA for richer triggers and scale-to-zero.
- **Bin-pack, and gang-schedule multi-GPU replicas.** `MostAllocated` scoring consolidates work so idle nodes can be reclaimed. Multi-GPU replicas need all-or-nothing placement (Volcano/Kueue) or they deadlock the cluster half-placed.
- **Cold start is a serial sum of stages.** Detect, node-provision, image-pull, weight-load, warmup — each additive, easily minutes if nothing is pre-warmed, tens of seconds if everything is. Attack each term: warm pools, image pre-pull, local weight caches, warmed processes.
- **Scale-to-zero trades the idle bill for cold-start latency.** Great for spiky and long-tail workloads with a tolerant SLA and a request buffer; wrong for latency-critical always-on services, which need a warm floor.
- **Reactive autoscaling cannot meet a tight SLO alone.** A burst accumulates a queue during the seconds it takes to scale. Carry standing headroom — spare replicas or a lower target utilization — sized to the burst you must absorb.

## Further reading

- **NVIDIA Multi-Instance GPU User Guide** — the authoritative reference for MIG profiles, geometry constraints, and the `nvidia-smi mig` commands. Read the profile-placement rules before designing a mixed layout.
- **NVIDIA GPU Operator documentation** — how the operator deploys the driver, device plugin, DCGM, GPU Feature Discovery, and MIG Manager, and how to set MIG geometry and time-slicing/MPS sharing via config.
- **Kubernetes Horizontal Pod Autoscaler documentation** — the HPA v2 API, custom and external metrics, and the `behavior` field for scale-up/down policies and stabilization windows.
- **KEDA documentation (keda.sh)** — ScaledObjects, the Prometheus scaler, `activationThreshold` versus `threshold`, `cooldownPeriod`, and scale-to-zero mechanics.
- **Knative Serving documentation** — scale-to-zero, the activator's request buffering during cold start, and the KPA (Knative Pod Autoscaler), the reference design for buffered scale-to-zero.
- **vLLM production metrics documentation** — the `vllm:` Prometheus metrics (`num_requests_running`, `num_requests_waiting`, `gpu_cache_usage_perc`, `time_to_first_token_seconds`) that are the right autoscaling inputs.
- Within this series: [what model serving is](/blog/machine-learning/model-serving/what-is-model-serving), [serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics), [LLM control planes: AIBrix and KServe](/blog/machine-learning/model-serving/llm-control-planes-aibrix-kserve), [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management), and [choosing your serving stack](/blog/machine-learning/model-serving/choosing-your-serving-stack).

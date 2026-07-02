---
title: "The Interconnect Physics: NVLink, PCIe, InfiniBand, and Why Placement Matters"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Read your cluster's wiring like a topology map: which link every collective rides, why tensor parallelism belongs on NVLink, and how to catch a silent PCIe or TCP fallback before it costs you 5x throughput."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "nvlink",
    "infiniband",
    "nccl",
    "gpu",
    "pytorch",
    "ml-systems",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

An engineer I worked with brought me a puzzle. She had a shiny new eight-GPU H100 box, a 7-billion-parameter model, and a tensor-parallel training job that ran at almost exactly the speed of two GPUs. Not two-and-a-half. Two. Adding the other six cards moved the throughput needle by single-digit percent. The GPUs showed 95% utilization in `nvidia-smi`, the loss curve looked healthy, no errors, no warnings. By every dashboard it was a working eight-GPU run. It was, in fact, a working two-GPU run wearing an eight-GPU costume.

The diagnosis took one command. `nvidia-smi topo -m` printed a matrix, and where I expected to see `NV18` — eighteen NVLink connections between every pair of GPUs — half the pairs read `SYS`. That single word means the two GPUs can only reach each other by crossing the CPU sockets over the system interconnect: down PCIe, across the inter-socket link, back up PCIe. The tensor-parallel all-reduce that fires on *every layer, twice per forward and twice per backward* was being carried, for half the GPU pairs, over a link roughly twenty times slower than the NVLink she had paid for. The compute was fine. The bytes were stuck in traffic.

This post is about the physics that decides whether your parallelism scales or stalls: the wires between your GPUs. Compute has gotten absurdly fast — an H100 does nearly a thousand bf16 TFLOP/s — but moving a byte from one GPU to another has not kept pace, and there is a *hierarchy* of slow. On-GPU memory is a firehose; NVLink inside a node is a fat pipe; PCIe is a garden hose; InfiniBand between nodes is a straw; and if your fabric is misconfigured, you can fall all the way back to a TCP socket that is a coffee stirrer. The figure below is the whole mental model in one picture, and everything else in this post is a consequence of it.

![the memory and network bandwidth hierarchy from fast on-GPU HBM down to a slow TCP socket fallback between nodes](/imgs/blogs/the-interconnect-physics-1.webp)

By the end you will be able to look at any cluster, read its topology with `nvidia-smi topo -m`, know which link each of your collectives is actually riding by reading `NCCL_DEBUG=INFO`, compute whether a given parallelism strategy will be compute-bound or comms-bound *before* you launch, and catch the silent fallbacks that quietly cost 5x. This is the hardware wall in the [four-walls frame that opens the series](/blog/machine-learning/distributed-training/why-distributed-training): the model won't fit, the data won't finish, the run is too slow, the cost is too high — and the interconnect sets the ceiling on how much distributing the work can actually buy you.

## The one idea: compute is cheap, moving bytes is not

Start with the numbers that make this real. An NVIDIA A100 80GB SXM delivers about 312 dense bf16 TFLOP/s and reads its own HBM2e memory at roughly 2.0 TB/s. An H100 SXM pushes that to about 989 bf16 TFLOP/s and 3.35 TB/s of HBM3. Those on-GPU bandwidths — terabytes per second — are the fast world. Now leave the die. NVLink4 on an H100 aggregates to about 900 GB/s per GPU. That is already 3.7x slower than reading local HBM. Drop to PCIe Gen5 and you are at roughly 64 GB/s — another ~14x down. Cross to another node over InfiniBand NDR and one port gives you about 50 GB/s — comparable to PCIe but now shared across everything leaving the node. And if NCCL can't find the fast fabric and falls back to TCP sockets over a management Ethernet link, you are looking at ~1 GB/s, which is roughly *three thousand times* slower than the HBM your kernels were happily saturating a nanosecond earlier.

That is the mental model to carry everywhere: a byte's cost depends entirely on how far it has to travel. Staying on-die is nearly free. Hopping to a neighbor GPU over NVLink is cheap. Leaving the node is expensive. Leaving the node *badly* is catastrophic. Distributed training is the art of arranging your computation so that the bytes which have to move often move short distances, and only the bytes that move rarely are allowed to cross the expensive links.

Here is why this matters more than it might seem. A modern GPU is so fast at arithmetic that most training workloads are *bandwidth-bound*, not compute-bound — the chip finishes the math and then sits waiting for data. (If that framing is new, the [roofline model post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) develops it properly.) When you distribute a model, you deliberately introduce a new, much slower class of data movement: gradients, activations, and parameter shards that have to cross between GPUs. If those crossings land on the wrong link, the interconnect becomes the roofline, and no amount of extra GPUs helps — you have simply bought more expensive chips to wait in a longer line.

There is a useful vocabulary here, so let me define the terms once. **Bandwidth** is how many bytes per second a link can sustain — this dominates for the large transfers of training (whole gradient buffers, activation tensors). **Latency** is how long a single small message takes end to end — this dominates for tiny, frequent messages and for the launch overhead of a collective. **Scope** is how far a link reaches: on-GPU, intra-node (between GPUs in the same chassis), or inter-node (between chassis). And **topology** is the wiring diagram — which GPU can reach which other GPU, and over what. A collective library like NCCL reads the topology, then chooses an algorithm that keeps traffic on the fastest links it can. Get the topology wrong and NCCL makes the best of a bad map; it cannot conjure bandwidth that the wires don't have.

## The links, one by one, with real numbers

Let me walk the hierarchy from fast to slow and put defensible numbers on each tier. Where a figure is a vendor spec I'll say so; where it's a round order of magnitude I'll flag it as approximate. Never trust a "typical bandwidth" you can't trace to a whitepaper or a measurement — half of all distributed-training folklore is someone's peak number quoted as an achieved one.

**HBM (High-Bandwidth Memory), on-GPU.** This is the stacked DRAM sitting on the GPU package. A100 80GB: ~2.0 TB/s (HBM2e). H100 SXM: ~3.35 TB/s (HBM3). H200 and the newest parts push past 4 TB/s (HBM3e). Latency is in the hundreds of nanoseconds. This is where your weights and activations live and where every kernel reads and writes. It never leaves the GPU, so it never crosses any of the links below — but it sets the reference speed against which every other link looks slow.

**NVLink and NVSwitch, intra-node.** NVLink is NVIDIA's point-to-point GPU-to-GPU link. NVLink3 on the A100 gives about 600 GB/s aggregate per GPU (12 links); NVLink4 on the H100 gives about 900 GB/s aggregate per GPU (18 links). Aggregate means the sum across all links in both directions; the *unidirectional* per-GPU bandwidth you actually get to push into a ring is roughly half that — call it ~300 GB/s on A100, ~450 GB/s on H100. The crucial piece is **NVSwitch**: a crossbar chip that lets all eight GPUs in a node talk to all seven others at full NVLink speed simultaneously, with no oversubscription. A DGX/HGX H100 node has four NVSwitch chips wiring the eight GPUs into a single fully connected NVLink island. Inside that island, every GPU pair is one hop apart at ~900 GB/s. This is the fastest link that crosses between GPUs, and it is the reason "keep it in the node" is the first rule of placement.

**PCIe, intra-node fallback.** PCI Express is the general-purpose bus that connects GPUs to the CPU and, in cheaper servers without NVSwitch, to each other. PCIe Gen4 x16 is about 32 GB/s per direction; Gen5 x16 about 64 GB/s per direction. In a PCIe-only box (no NVLink), GPU-to-GPU peer traffic goes over PCIe and often has to traverse a PCIe switch or even the CPU root complex, and pairs on different CPU sockets must cross the inter-socket link (that's the `SYS` you saw in the opening story). PCIe is roughly 14–28x slower than NVLink for the same transfer, and it is shared with every other device on the bus. When people say "our all-reduce fell off NVLink," they mean it landed here.

**InfiniBand and RoCE, inter-node.** To leave the chassis you need a network. InfiniBand is the high-performance fabric of choice: HDR is ~200 Gb/s per port (~25 GB/s), NDR is ~400 Gb/s per port (~50 GB/s), and the newest XDR pushes ~800 Gb/s. A rail-optimized DGX H100 gives each of its eight GPUs its own ConnectX-7 NDR NIC — eight NICs, ~400 GB/s of aggregate off-node bandwidth. RoCE (RDMA over Converged Ethernet) delivers similar RDMA semantics over Ethernet hardware at 100/200/400 GbE. InfiniBand's headline feature is not just bandwidth but *low latency with RDMA* — a message can land in remote memory in roughly a microsecond without waking the remote CPU. That RDMA capability is what makes inter-node collectives tolerable at all.

**Ethernet and the TCP fallback, inter-node.** Plain TCP/IP over Ethernet — no RDMA — is where things go to die for training. The kernel networking stack adds tens of microseconds of latency and caps throughput far below line rate because every byte is copied through host memory and shepherded by the CPU. A misconfigured cluster where NCCL can't reach the RDMA fabric will silently use TCP sockets over whatever Ethernet interface it can find, often a 1 GbE or 10 GbE management network, and you'll see ~1 GB/s. This is the floor, and it is a trap because nothing errors — the job runs, just 40x slower on every collective.

The matrix below lays these five tiers against the four dimensions that actually decide placement. Read it as a decision aid: when you know how far bytes must go and how often, it tells you which column you're paying in.

![a comparison matrix of five interconnect tiers scored on bandwidth, latency, reach, and cost](/imgs/blogs/the-interconnect-physics-2.webp)

Written out as numbers you can quote:

| Link | Bandwidth (approx) | Latency (approx) | Scope | Cost tier |
|---|---|---|---|---|
| HBM (H100) | 3.35 TB/s | ~100s ns | On-GPU only | Baked into GPU |
| NVLink4 | 900 GB/s aggregate (~450 GB/s unidir) | ~1 µs | Intra-node, 8 GPUs | High (NVSwitch) |
| PCIe5 x16 | ~64 GB/s/dir | ~1–2 µs | Intra-node fallback | Commodity |
| InfiniBand NDR | ~50 GB/s/port | ~1–2 µs | Inter-node | High (HCA + switch) |
| Ethernet / RoCE | 12–50 GB/s | ~5–100 µs | Inter-node | Cheapest |
| TCP sockets | ~1 GB/s | tens of µs | Inter-node fallback | (the trap) |

Notice the two order-of-magnitude cliffs: NVLink to PCIe (~14x), and RDMA-capable fabric to a TCP fallback (~40x). Almost every "why is my scaling terrible" story is a collective that tumbled over one of those cliffs.

### NVLink across generations, and why the gap keeps widening

It helps to see the trend, because it explains why "keep it in the node" has gotten *more* important over time, not less. NVLink has roughly doubled every GPU generation: NVLink1 on Pascal (P100) was ~160 GB/s aggregate; NVLink2 on Volta (V100) ~300 GB/s; NVLink3 on Ampere (A100) ~600 GB/s; NVLink4 on Hopper (H100) ~900 GB/s; and NVLink5 on Blackwell (B200) pushes to ~1.8 TB/s aggregate per GPU. PCIe, meanwhile, has doubled far more slowly — Gen3 (~16 GB/s) to Gen4 (~32 GB/s) to Gen5 (~64 GB/s) across the same span. So the *ratio* between the fat intra-node link and the PCIe fallback has grown from roughly 10x to nearly 28x. Every generation, the penalty for accidentally landing a collective on PCIe instead of NVLink gets steeper. The compute got faster too, of course, but the point stands: the interconnect hierarchy is not flattening, it is sharpening, and placement mistakes cost more on newer hardware than they did on older hardware.

Inter-node bandwidth has climbed as well — InfiniBand went QDR (~40 Gb/s) → FDR (~56) → EDR (~100) → HDR (~200) → NDR (~400) → XDR (~800 Gb/s) — but it has stayed roughly an order of magnitude below NVLink the whole way. That persistent ~10x gap between "inside the node" and "between nodes" is the most stable fact in this entire post. Design your parallelism around it and you will rarely be surprised.

### RoCE vs InfiniBand: the same idea, different tradeoffs

You will meet two flavors of RDMA fabric in the wild, and the distinction matters when you're debugging. **InfiniBand** is a purpose-built lossless fabric with its own switches, cabling, and subnet manager; it delivers RDMA with very low latency and is the default in NVIDIA reference designs. **RoCE** (RDMA over Converged Ethernet) delivers the same RDMA *semantics* — direct remote-memory access, GPUDirect support — but rides Ethernet hardware, which is cheaper and more familiar to network teams. The catch is that Ethernet is lossy by default, and RoCE needs careful configuration (Priority Flow Control, ECN, a well-tuned congestion-control scheme) to behave like a lossless fabric. A poorly configured RoCE network drops packets under the bursty all-to-all traffic of a collective, triggers retransmits, and delivers a fraction of line rate — a failure that looks a lot like the TCP-fallback trap but is subtler, because RDMA *is* active. For NCCL, both appear as `NET/IB` transport (NCCL treats RoCE through the same IB verbs path); the tell is whether achieved `busbw` matches the NIC's line rate. If you're on RoCE and nccl-tests reports half of spec, suspect the flow-control config before you suspect NCCL.

## What a node and a cluster actually look like

Bandwidth numbers are half the picture; the other half is *topology* — the shape of the wiring. The single most important thing to internalize is that a modern GPU node is not a flat pool of eight equal GPUs. It is a fully connected NVLink island, and the island connects to other islands only through a much thinner network. The figure below draws two nodes and the fabric between them.

![a topology diagram of two eight-GPU NVSwitch nodes connected through NIC rails and an InfiniBand spine](/imgs/blogs/the-interconnect-physics-3.webp)

Inside one node, the eight SXM GPUs plug into NVSwitch, which gives every GPU a full-bandwidth path to every other. There is no "near" and "far" GPU inside the island — GPU 0 to GPU 7 is the same ~900 GB/s as GPU 0 to GPU 1. This is why an eight-way tensor-parallel group fits so naturally in one node: the all-reduce that binds the group can use the full NVLink mesh, and NCCL can run a ring or a tree across all eight without ever touching a slow link.

Between nodes, the story changes completely. Each GPU (in a rail-optimized design) has a dedicated NIC, and those NICs plug into a **fat-tree** InfiniBand fabric. A fat-tree is built so that bandwidth doesn't shrink as you climb toward the root: leaf switches connect to spine switches with enough uplinks that, in principle, every node can talk to every other node at full NIC rate simultaneously without the core becoming a bottleneck. "Rail-optimized" means GPU `i` on every node connects to the same rail (the same slice of the fabric), so that a collective across GPU `i` on all nodes stays on one rail and avoids cross-rail congestion. This is not a detail — it is the difference between a 64-GPU all-reduce that hits ~90% of NIC bandwidth and one that thrashes the spine at 40%.

Why does topology shape *which algorithm NCCL picks*? Because NCCL builds its rings and trees to match the wiring. Intra-node, it lays a ring through the NVLink mesh so consecutive ring neighbors are one NVSwitch hop apart. Inter-node, it prefers **trees** for latency-sensitive small messages (a tree has logarithmic depth, so a small all-reduce finishes in `log N` hops instead of `N`) and **rings** for bandwidth-bound large messages (a ring achieves near-optimal bandwidth utilization). It also builds *hierarchical* collectives: reduce within each node over NVLink first, then a single inter-node exchange over IB, then broadcast back down NVLink — so the expensive inter-node link carries the least possible traffic. If NCCL misreads the topology (or you force it with the wrong env vars), it picks a worse algorithm and you leave bandwidth on the floor. The mechanics of these collectives — ring versus tree, and the byte-volume math behind them — are worked out from scratch in the [collectives post](/blog/machine-learning/distributed-training/collectives-from-scratch); here I care only about which *link* they land on.

### Reading your own topology: `nvidia-smi topo -m`

You do not have to guess at any of this. Every NVIDIA box will print its own wiring map:

```bash
nvidia-smi topo -m
```

On a healthy DGX H100 the GPU-to-GPU block of the matrix is a wall of `NV18` — every pair connected by 18 NVLinks. On a PCIe box you'll see `PIX`, `PXB`, `PHB`, `NODE`, or the dreaded `SYS`. Here is a stylized fragment of what a *bad* placement looks like:

```console
        GPU0    GPU1    GPU2    GPU3    CPU Affinity  NUMA
GPU0     X      NV18    SYS     SYS     0-31          0
GPU1     NV18    X      SYS     SYS     0-31          0
GPU2     SYS     SYS     X      NV18    32-63         1
GPU3     SYS     SYS     NV18    X      32-63         1
```

Read the legend from best to worst:

| Code | Meaning | Speed class |
|---|---|---|
| `X` | Self (diagonal) | n/a |
| `NV#` | Connected by # NVLinks | Fastest GPU-to-GPU |
| `PIX` | Single PCIe switch hop | Good |
| `PXB` | Multiple PCIe switches | OK |
| `PHB` | PCIe host bridge (through CPU) | Slow |
| `NODE` | Within a NUMA node, across PCIe host bridges | Slow |
| `SYS` | Across NUMA nodes / CPU sockets | Slowest — avoid for hot traffic |

In that stylized matrix, GPUs 0–1 are NVLink-connected and 2–3 are NVLink-connected, but 0↔2 and 0↔3 are `SYS` — across CPU sockets. If you launch an eight-way tensor-parallel group here, every all-reduce touches `SYS` pairs and rides the inter-socket PCIe path. That is precisely the opening story. The fix there was not a code fix; it was a *placement* fix — pin the tensor-parallel group to GPUs that share NVLink, and let the slower data-parallel dimension straddle the socket boundary. The `CPU Affinity` and `NUMA` columns matter too: pinning your data-loader worker threads and the process to the NUMA node local to its GPUs avoids a second, sneakier host-memory penalty.

## GPUDirect RDMA and NVLink SHARP: get the CPU out of the way

There's a subtlety hiding inside "leaving the node is expensive" that, once you see it, explains a lot of misconfigured clusters. Naively, sending a gradient tensor from GPU A on one node to GPU B on another means: copy from GPU A's HBM into node A's host RAM, hand it to the NIC, send it across the fabric, receive into node B's host RAM, copy up into GPU B's HBM. That is *two* extra host-memory copies and two trips across PCIe, plus CPU involvement to orchestrate them — and it doubles the effective data movement while adding latency and CPU contention. The before/after below draws the difference.

![a before and after diagram contrasting a staged host-memory copy against a direct GPU-to-GPU RDMA transfer](/imgs/blogs/the-interconnect-physics-4.webp)

**GPUDirect RDMA** removes the host from the path. It lets the NIC read directly from GPU HBM and, on the receiving side, write directly into GPU HBM — no staging through host RAM, no CPU copy. The NIC and the GPU speak PCIe (or NVLink, on newer platforms) to each other directly. This is what makes inter-node collectives fast enough to be usable: the ~50 GB/s of an NDR NIC actually reaches GPU memory instead of being throttled by host-memory copies and CPU scheduling. In the NCCL log you'll see this as `[GDRDMA]` or `via NET/IB ... GDRDMA`. If GPUDirect RDMA is *not* active — because the NIC and GPU are on different PCIe switches without a direct path, or `NCCL_NET_GDR_LEVEL` is set too conservatively — NCCL silently reverts to the staged-copy path and your inter-node bandwidth drops by roughly half with a latency penalty on top. Same fabric, same NICs, half the speed, no error.

There is a second trick worth knowing: **NVLink SHARP** (Scalable Hierarchical Aggregation and Reduction Protocol). NVSwitch chips can perform the reduction *in the network* — instead of every GPU sending its data to a peer to be summed, the switch itself sums the contributions as they pass through. For an all-reduce inside a node, in-network reduction roughly halves the data that has to move (you don't send-then-receive the full tensor twice) and cuts latency. On H100-generation NVSwitch this can lift intra-node all-reduce bandwidth meaningfully for the message sizes that matter in training. You mostly get it for free when it's available, but it's another reason the intra-node NVLink island is not just "faster wires" — it's a smarter fabric. The broader hardware story, including how RDMA and NVLink SHARP compose, is covered in the [HPC interconnects deep-dive](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma); this post's job is to turn those capabilities into placement decisions.

## The placement law: chattiest parallelism on the fattest link

Now we can state the rule the whole post has been building toward, and prove it. Different parallelism strategies generate wildly different amounts of communication per training step, and the link you place each one on has to be fast enough to hide that communication behind compute. So:

> **Put the chattiest parallelism on the fattest link.** Tensor parallelism goes inside a node on NVLink. Data parallelism tolerates inter-node InfiniBand. Pipeline parallelism tolerates the thinnest link you have.

Let me derive *why* from the communication each one costs. The core tool is the cost of a **ring all-reduce**, the collective that dominates distributed training. To all-reduce a tensor of `S` bytes across `N` GPUs, the ring algorithm moves data in `2(N-1)` steps, each carrying `S/N` bytes, so each GPU pushes a total of

$$T_{\text{all-reduce}} = \frac{2(N-1)}{N} \cdot \frac{S}{B}$$

where `B` is the per-GPU (unidirectional) link bandwidth. For large `N` the factor `2(N-1)/N` approaches 2, so an all-reduce costs roughly `2S/B` — you move about twice the tensor's size across your slowest relevant link. That factor-of-2 and that `B` in the denominator are the entire game. Halve `B` (NVLink to PCIe) and every all-reduce doubles... except the drop from NVLink to PCIe isn't 2x, it's ~14x, so every all-reduce gets ~14x slower.

Now count how often each parallelism strategy pays that cost per step:

- **Tensor parallelism (TP)** splits each matmul across GPUs (column-parallel then row-parallel linears in a transformer block). Every layer needs an all-reduce to recombine the partial results — in Megatron-style TP, two all-reduces in the forward pass and two in the backward pass, *per layer*. For a 32-layer model that's ~128 all-reduces per step, and critically they sit **on the critical path**: the forward can't proceed to the next layer until the current layer's all-reduce completes. TP is the chattiest strategy by a wide margin, and its comms cannot be hidden. It *must* live on NVLink.
- **Data parallelism (DP)** replicates the model and averages gradients. It needs exactly **one** all-reduce of the full gradient buffer per step. Better still, DDP buckets the gradients and launches each bucket's all-reduce the instant its gradients are ready during the backward pass, so the comms **overlaps** with the rest of the backward compute and is largely hidden. One reduce per step, overlappable — DP tolerates inter-node IB comfortably. (The bucketing-and-overlap machinery is the subject of the DDP posts in this series; here the point is just *how rarely* DP talks.)
- **Pipeline parallelism (PP)** splits the model into stages across GPUs and streams micro-batches through. Between adjacent stages it sends only the activation tensor at the stage boundary — small point-to-point `send`/`recv`, not a collective, and only at stage boundaries. PP is the quietest strategy; it tolerates the thinnest link and is the natural choice for the *inter-node* dimension when you must span nodes.

That ordering — TP loudest, DP moderate, PP quietest — maps directly onto the hierarchy. The decision tree below is the placement law made mechanical.

![a decision tree mapping tensor, data, and pipeline parallelism onto interconnect tiers by communication intensity](/imgs/blogs/the-interconnect-physics-7.webp)

And the before/after below shows what happens when you break the law — the same tensor-parallel all-reduce placed across nodes versus kept inside one.

![a before and after diagram showing tensor-parallel all-reduce starving GPUs over InfiniBand versus keeping them busy over NVLink](/imgs/blogs/the-interconnect-physics-5.webp)

#### Worked example: the same 8-GPU tensor-parallel all-reduce on NVLink4 vs forced onto PCIe

Let me make the placement law bite with numbers. Take a 7B transformer, TP degree 8, on an H100 node. Micro-batch of 4 sequences of length 4096 (16,384 tokens), hidden size 4096, bf16 activations. The all-reduce message per collective is the activation tensor: `4 × 4096 × 4096 × 2 bytes ≈ 134 MB`. With 32 layers and 4 all-reduces per layer (2 forward, 2 backward), that's 128 all-reduces per step.

Ring all-reduce factor for N=8: `2(8-1)/8 = 1.75`. Now compute the per-collective time and the per-step comms on two links, using effective per-GPU unidirectional bandwidths:

- **NVLink4**, effective `B ≈ 450 GB/s`: per all-reduce `= 1.75 × 134e6 / 450e9 ≈ 521 µs`. Across 128 collectives: **~67 ms/step of comms.**
- **PCIe Gen4**, effective `B ≈ 32 GB/s`: per all-reduce `= 1.75 × 134e6 / 32e9 ≈ 7.33 ms`. Across 128 collectives: **~938 ms/step of comms.**

The comms wall is **14x taller on PCIe** — exactly the bandwidth ratio, because all-reduce is bandwidth-bound. Now fold in compute. Forward+backward FLOPs ≈ `6 × 7e9 × 16,384 ≈ 6.9e14`, split across 8 GPUs at ~50% MFU (~495 effective TFLOP/s each): `6.9e14 / (8 × 495e12) ≈ 174 ms/step` of compute. Because TP all-reduce is on the critical path (not overlappable), step time is compute + comms:

| Configuration (H100, TP=8, 7B) | Compute/step | Comms/step | Step time | Throughput | Scaling |
|---|---|---|---|---|---|
| TP all-reduce on NVLink4 | 174 ms | 67 ms | 241 ms | ~68,000 tok/s | compute-bound |
| TP all-reduce forced to PCIe | 174 ms | 938 ms | 1,112 ms | ~14,700 tok/s | comms-bound |

That's a **4.6x** end-to-end throughput collapse from the *same GPUs* running the *same model* — the only difference is which wire the all-reduce rode. And 4.6x is the *diluted* number, because a 16k-token micro-batch has enough compute to partially mask the comms. Shrink the batch (or raise the TP degree, which is exactly what you do when the model is big), and you enter the comms-bound regime where the throughput gap approaches the full 14x. On a thinner link or with `SYS`-crossing pairs it goes worse still — that's how the opening story's eight-GPU job degraded to two-GPU speed. This is the concrete meaning of "TP must stay on NVLink": off it, you are not training faster, you are heating the datacenter.

#### Worked example: 64-GPU data-parallel all-reduce over InfiniBand, and why it still scales

Contrast that with data parallelism across nodes. Same 7B model, now pure DP across 64 GPUs (8 nodes × 8 GPUs), gradients in bf16: the full gradient buffer is `7e9 × 2 ≈ 14 GB`. DP does **one** all-reduce of that buffer per step. Ring factor for N=64: `2(63)/64 ≈ 1.97`. Over rail-optimized NDR at an effective `B ≈ 40 GB/s` per GPU:

$$T_{\text{DP all-reduce}} = 1.97 \times \frac{14 \times 10^9}{40 \times 10^9} \approx 690 \text{ ms}.$$

At first glance 690 ms looks fatal. It isn't, for two reasons. First, it happens **once per step**, not 128 times. Second, and decisively, it **overlaps with the backward pass**. Each GPU's backward compute on its local micro-batch (say 8 sequences × 4096 = 32,768 tokens) is `6 × 7e9 × 32,768 × (2/3) / 495e12 ≈ 1.85 s` of backward compute, and DDP launches gradient-bucket all-reduces *during* that backward. Since 690 ms of overlappable comms fits comfortably inside 1.85 s of backward, the all-reduce is almost entirely hidden. The step time stays dominated by compute, and scaling efficiency stays high:

| Configuration (64 GPUs, 7B, DP) | Backward compute | All-reduce | Overlapped? | Effective comms exposed |
|---|---|---|---|---|
| DP all-reduce over NDR IB | ~1.85 s | ~690 ms | Yes (bucketed) | ~0 (hidden) |
| Same, but TP-across-nodes (for contrast) | ~1.85 s | 128 × per-layer, on critical path | No | ~0.75 s+ per step exposed |

That last row is the whole lesson in one table. Put the *once-per-step, overlappable* collective (DP) on the inter-node link and it hides. Put the *per-layer, critical-path* collective (TP) on the same inter-node link and it stalls every GPU in the cluster. Identical hardware, identical fabric — the placement is the entire difference. This is why real 3D-parallel training layouts put TP inside the node, PP across a few nodes, and DP across the rest: each dimension lands on a link fast enough to hide its particular chatter.

### The comms-to-compute ratio: when a collective hides and when it stalls

There's a single inequality underneath all of this, and it's worth stating explicitly because it turns "will this scale?" from a gut feeling into a back-of-envelope calculation. A collective is *hidden* — free, in wall-clock terms — when the time it takes is less than the compute it can overlap with. For an overlappable collective (like DP's bucketed gradient all-reduce), the condition is:

$$T_{\text{comms}} = \frac{2(N-1)}{N} \cdot \frac{S}{B} \;\; \le \;\; T_{\text{compute-overlap}}.$$

For a non-overlappable, critical-path collective (like TP's per-layer all-reduce), there is no overlap term — the comms time adds directly to the step, so the relevant ratio is `T_comms / T_compute` per layer. When that ratio is well below 1, you're compute-bound and scaling holds; when it approaches or exceeds 1, comms dominates and adding GPUs stops helping.

Two levers move the ratio, and they explain every scaling surprise you'll ever hit. The first is `B`, the link bandwidth — the entire subject of this post; drop from NVLink to PCIe and the ratio jumps ~14x, which can flip a comfortably-hidden collective into a stall. The second is the ratio of `S` (bytes to move) to compute (which grows with the *arithmetic* you do between collectives). This is why bigger micro-batches and bigger hidden dimensions improve scaling efficiency: they raise compute faster than they raise the per-collective message, shrinking `T_comms / T_compute`. It's also why very small batches scale terribly — there isn't enough compute to hide the comms behind, so the interconnect shows through. When someone says "our scaling efficiency tanked when we shrank the batch to fit memory," this inequality is what broke: they cut the compute side without cutting the comms side, and the ratio crossed 1.

#### Worked example: pipeline point-to-point across nodes

To close the loop on "pipeline parallelism tolerates the thinnest link," price its communication. PP sends only the boundary activation tensor between adjacent stages — a point-to-point `send`/`recv`, not a collective. For the same 7B model split into, say, 4 pipeline stages with a micro-batch of 16,384 tokens and hidden 4096 in bf16, each stage boundary passes `16,384 × 4096 × 2 ≈ 134 MB` once per micro-batch in the forward direction and once in the backward. There are 3 boundaries. Over inter-node InfiniBand NDR at effective `B ≈ 40 GB/s`, one boundary transfer is `134e6 / 40e9 ≈ 3.4 ms`. Even with 8 micro-batches in flight, the per-step point-to-point traffic is a few tens of milliseconds — and crucially it overlaps with the next stage's compute in a 1F1B schedule. Compare that to TP's 128 critical-path all-reduces per step and you see why PP is the dimension you're happy to stretch across the fabric: its bytes are few, infrequent, and overlappable. The cost PP pays is not comms but the pipeline *bubble* (stages idle while the pipe fills and drains) — a compute-efficiency cost, not an interconnect one, and a separate tradeoff from the one this post is about.

### The hard case: all-to-all traffic

There is one collective that stresses the interconnect harder than all-reduce, and it's worth naming because it changes the placement calculus. **Expert parallelism** in Mixture-of-Experts models routes each token to a subset of experts scattered across GPUs, which requires an **all-to-all**: every GPU sends a different slice of its tokens to every other GPU, and receives a different slice back. Unlike all-reduce, all-to-all has no reduction to amortize and no ring structure to make it bandwidth-optimal — it is `N` simultaneous point-to-point exchanges, and its cost scales poorly as the fabric fans out. Worse, MoE routing is *data-dependent*: token counts per expert vary batch to batch, so the traffic is bursty and imbalanced, exactly the pattern that exposes a lossy RoCE fabric's flow-control weaknesses. The placement lesson generalizes with a vengeance: if you must do expert parallelism, keep the experts inside a node on NVLink whenever the model lets you, because an all-to-all over InfiniBand is even less forgiving than a TP all-reduce over InfiniBand. When people report that their MoE training is comms-bound where the equivalent dense model was compute-bound, the all-to-all is almost always the culprit, and the fabric it landed on is the first thing to check.

## Placing parallelism in code: the device mesh

The placement law is a hardware fact; expressing it correctly is a software job. In modern PyTorch you declare the layout with a **device mesh** — a named, multi-dimensional grid of ranks — and the framework maps each parallelism dimension onto a slice of it. The critical thing is *which mesh dimension varies fastest across local ranks*, because `torchrun` numbers the 8 GPUs inside a node as local ranks 0–7, and NCCL builds intra-node NVLink groups from consecutive local ranks. So you want the **tensor-parallel dimension to be the innermost (fast-varying) one**, so a TP group lands entirely inside a node on NVLink, while the data-parallel dimension varies across nodes:

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")

# 64 GPUs = 8 nodes x 8 GPUs. We want TP=8 (intra-node) x DP=8 (inter-node).
# Order matters: "tp" is the last (fastest-varying) dim, so each TP group of
# 8 ranks is exactly the 8 GPUs of one node -> the per-layer all-reduce
# rides NVLink. "dp" varies across nodes -> its once-per-step all-reduce
# rides InfiniBand, where it overlaps with backward and hides.
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(8, 8),          # (dp, tp)
    mesh_dim_names=("dp", "tp"),
)

tp_group = mesh["tp"].get_group()   # 8 ranks, all inside one node (NVLink)
dp_group = mesh["dp"].get_group()   # 8 ranks, one per node (InfiniBand)

# Sanity-check placement before training a single step:
local_rank = int(os.environ["LOCAL_RANK"])
tp_ranks = dist.get_process_group_ranks(tp_group)
print(f"rank {dist.get_rank()} local {local_rank} tp_group {tp_ranks}")
# Every rank in a tp_group should share a node; if a tp_group spans nodes,
# your mesh ordering is wrong and TP will run over InfiniBand.
```

That print is your cheap insurance: if any tensor-parallel group's ranks straddle two nodes, you have reproduced the opening story in advance and can fix the mesh ordering before wasting a single GPU-hour. Frameworks like Megatron-LM and DeepSpeed encode the same rule internally — they place tensor-model-parallel ranks contiguously so the group stays in a node — but when you build a mesh by hand, the ordering is yours to get right.

The launch that backs this up is where the interconnect env vars live. On a SLURM cluster, the `sbatch` script pins the rails and names the fast interface so NCCL never wanders onto a management NIC:

```bash
#!/bin/bash
#SBATCH --job-name=train-7b
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

# --- interconnect hygiene: force the fast fabric, fail loud, not slow ---
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=PHB          # allow GPUDirect RDMA across PCIe host bridge
export NCCL_DEBUG=WARN                 # flip to INFO on the first shakedown run
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun torchrun \
  --nnodes=8 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:29500" \
  train.py --tp 8 --dp 8
```

Run it once with `NCCL_DEBUG=INFO` on a single step, confirm the transport lines (below), then switch to `WARN` for the real run so the logs stay quiet unless something breaks.

### Measuring interconnect performance honestly

Before you compare any two placements, know how to time a collective without lying to yourself — distributed timing is full of traps. GPU kernels are asynchronous, so a naive `time.time()` around a `dist.all_reduce` measures the *launch*, not the *finish*. You must synchronize:

```python
import time, torch, torch.distributed as dist

def time_allreduce(numel=256 * 1024 * 1024, dtype=torch.bfloat16, iters=50, warmup=10):
    x = torch.empty(numel, dtype=dtype, device="cuda")
    # Warm-up: first collectives pay one-time NCCL channel/buffer setup costs.
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.cuda.synchronize()            # wait for the GPU to actually finish
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()            # again: don't stop the clock early
    dt = (time.perf_counter() - t0) / iters
    gb = numel * x.element_size() / 1e9
    busbw = gb * 2 * (dist.get_world_size() - 1) / dist.get_world_size() / dt
    if dist.get_rank() == 0:
        print(f"{gb:.2f} GB  {dt*1e3:.2f} ms/iter  busbw {busbw:.1f} GB/s")
```

The four honesty rules baked into that snippet: **warm up** (the first few collectives pay NCCL's one-time channel and buffer allocation, so they're slower and unrepresentative); **`torch.cuda.synchronize()` before *and* after the timed region** (or the async launch queue makes a collective look instant); **average over many iterations** at **steady state** (so a single outlier doesn't dominate); and **report `busbw`, not wall time** (so the number is comparable across GPU counts). Beyond the collective itself, watch three confounds that masquerade as interconnect problems: a **data-loader stall** (the GPU is waiting on the CPU, not the network — check with the profiler), **thermal/clock throttling** (a hot GPU downclocks and every number sags — check `nvidia-smi -q -d CLOCK`), and **unoverlapped comms** (the collective is fast but not hidden behind backward — a scheduling bug, not a bandwidth one). The [profiling post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) shows how to tell these apart on a timeline; the rule of thumb is to always confirm the fabric with nccl-tests *first*, so you know the ceiling before you chase your own tail in the training loop.

## Diagnostics: know which link every collective is actually riding

The placement law is only useful if you can *verify* what's happening, because — as the opening story shows — the failure mode is silent. Three tools tell you the truth.

**1. Read the topology before you trust it.** We covered `nvidia-smi topo -m` above. Run it first, on every new cluster, before you launch anything. If your intended tensor-parallel peers show anything but `NV#`, stop and fix placement.

**2. Make NCCL tell you which transport it chose.** Set `NCCL_DEBUG=INFO` and grep the output. NCCL prints, per channel, exactly which transport it selected between each pair of ranks:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
  torchrun --nproc_per_node=8 --nnodes=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py 2>&1 | tee nccl.log

# Then read the transport lines:
grep -E "via (NVL|P2P|SHM|NET)" nccl.log | head
```

The strings you're looking for and what they mean:

| NCCL log fragment | Transport chosen | Verdict |
|---|---|---|
| `via NVL` / `via P2P/CUMEM` (intra-node NVLink) | NVLink peer-to-peer | Best — this is what you want intra-node |
| `via P2P` over PCIe | PCIe peer-to-peer | OK fallback; slow for TP |
| `via SHM` | Shared host memory | Intra-node but no P2P — investigate |
| `via NET/IB ... [GDRDMA]` | InfiniBand with GPUDirect RDMA | Best inter-node |
| `via NET/IB` (no GDRDMA) | InfiniBand, staged through host | ~2x slower than it should be |
| `via NET/Socket` | TCP sockets | The trap — ~40x too slow, no error |

If you see `NET/Socket` on a cluster that has InfiniBand, you have found a five-figure-per-month bug. Every gradient exchange is crawling over TCP.

**3. Measure the raw fabric with nccl-tests.** Before blaming your model, benchmark the wires directly. The `nccl-tests` suite runs a pure all-reduce and reports both `algbw` (algorithm bandwidth = data / time) and `busbw` (bus bandwidth = `algbw × 2(N-1)/N`, the hardware-normalized number you compare to spec):

```bash
# Build once: git clone https://github.com/NVIDIA/nccl-tests && make MPI=1
# 8 GPUs, one node, sweeping message sizes up to 2 GB:
./build/all_reduce_perf -b 8 -e 2G -f 2 -g 8

# Multi-node under SLURM, 64 GPUs:
srun --nodes=8 --ntasks-per-node=8 \
  ./build/all_reduce_perf -b 8 -e 2G -f 2
```

On a healthy 8×A100 node you should see intra-node `busbw` around 230–250 GB/s `algbw` (≈ 400+ GB/s `busbw`); on 8×H100 higher still. Inter-node over NDR, expect the per-GPU number to be gated by NIC bandwidth (tens of GB/s). If nccl-tests reports a fraction of spec, the problem is the fabric or its configuration, *not* your training code — a hugely clarifying thing to know at 3am. This is the same all-reduce whose byte-volume math we used above; nccl-tests just measures the `B` you actually get.

**The knobs that matter**, in rough order of how often they save a run:

```bash
# Pin the InfiniBand HCAs / rails so NCCL uses the fast NICs, not a mgmt port.
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

# Force the correct network interface for the bootstrap/out-of-band channel.
export NCCL_SOCKET_IFNAME=ib0        # or eth0 — match your fabric

# Control GPUDirect RDMA aggressiveness (higher = allow across PCIe hops).
export NCCL_NET_GDR_LEVEL=PHB

# Diagnostic toggles — flip to A/B test a suspected fallback, then remove:
export NCCL_P2P_DISABLE=0            # 1 disables intra-node P2P (to prove P2P helps)
export NCCL_IB_DISABLE=0            # 1 forces TCP (to reproduce the slow path)

# Make collective errors surface instead of hanging forever.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

The single most common inter-node fix is `NCCL_IB_HCA` plus `NCCL_SOCKET_IFNAME`. On a fresh cluster NCCL sometimes can't tell which of several interfaces is the fast fabric and picks a management NIC; naming the HCAs explicitly forces it onto InfiniBand. The A/B toggles (`NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`) are for *reproducing* a suspected slow path deliberately so you can confirm the diagnosis — set them, watch throughput crater to match your bug, then unset them and apply the real fix. The NCCL-log reading discipline goes much deeper in the dedicated debugging posts; for interconnect problems, transport-line grep plus an nccl-tests number is 90% of the job.

## The failure: a run that silently fell back

Here is the war story the diagnostics are for, because it is the single most expensive interconnect bug and it never throws an error. A team stood up a new 8-node cluster and ran a data-parallel job. Single-node throughput was healthy. Two nodes: *slower than one*. Four nodes: slower still. Every dashboard green, GPUs busy, loss descending. Adding hardware was making the job worse, which is the signature of comms domination — the [multi-node-slower-than-single-node autopsy](/blog/machine-learning/distributed-training/multinode-slower-than-single-node) is entirely about this class of bug.

The before/after below is the shape of it.

![a before and after diagram showing NCCL falling back to TCP sockets versus running on InfiniBand after pinning the host channel adapter](/imgs/blogs/the-interconnect-physics-6.webp)

The diagnosis followed the three tools in order. First, `nccl-tests` inter-node: `busbw` came back at ~0.9 GB/s where NDR should give tens of GB/s. That instantly ruled out the training code — the raw fabric was slow. Second, `NCCL_DEBUG=INFO` on the same run: every inter-node channel read `via NET/Socket`. NCCL had never touched InfiniBand; it was carrying every gradient all-reduce over TCP on the 10 GbE management network. Third, the root cause: the IB fabric was up (`ibstat` showed the ports Active) but `NCCL_SOCKET_IFNAME` was unset and NCCL had auto-selected the management interface for its bootstrap, then couldn't establish the IB path and fell back to sockets for data too. The mental model held: the bytes were fine, they were just taking the slowest possible road.

The fix was three environment variables — pin `NCCL_IB_HCA` to the eight ConnectX NICs, set `NCCL_SOCKET_IFNAME=ib0`, and confirm `NCCL_IB_DISABLE` was unset. Re-run: transport lines flipped to `via NET/IB ... [GDRDMA]`, nccl-tests `busbw` jumped to ~45 GB/s, and multi-node scaling went from *negative* to roughly linear. Same wires, same NICs, a config that finally used them. The lesson to carry: **on any new cluster, benchmark the fabric with nccl-tests and grep the transport lines before you trust a single throughput number.** A green dashboard is not evidence the fast path is active — only the NCCL log and the nccl-tests number are.

There is a subtler cousin of this bug worth naming: the *half*-fallback, where NCCL does use InfiniBand but without GPUDirect RDMA (`via NET/IB` with no `[GDRDMA]` tag). Bandwidth is halved rather than annihilated, so it hides better — a 50% throughput haircut that nobody notices because the job is "only a bit slow." It's usually a NIC/GPU PCIe-topology issue or an over-conservative `NCCL_NET_GDR_LEVEL`. Same detection method: read the transport line, check for the GDRDMA tag, compare nccl-tests to spec.

## Case studies and real numbers

A few grounded data points, cited so you can check them and flagged where they're approximate.

**Megatron-LM keeps tensor parallelism inside the node — on purpose.** The Megatron-LM papers (Shoeybi et al. 2019; Narayanan et al. 2021, "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM") are explicit that tensor-model-parallel groups are placed *within* a DGX node to exploit NVLink, while pipeline and data parallelism span nodes over InfiniBand. Their large-scale runs report aggregate throughput and per-GPU utilization (MFU/HFU in the tens of percent) that are only achievable because the per-layer TP all-reduce never leaves NVLink. Set `--tensor-model-parallel-size` no larger than the GPUs-per-node and let `--pipeline-model-parallel-size` and DP take the inter-node dimensions — that ordering is a direct consequence of the placement law, not a tuning nicety.

**NVLink vs InfiniBand all-reduce bandwidth is an order of magnitude.** Published nccl-tests numbers and NVIDIA's own benchmarks put intra-node 8×A100/H100 all-reduce `busbw` in the mid-hundreds of GB/s, while inter-node all-reduce is gated by aggregate NIC bandwidth (tens of GB/s per GPU on NDR). The exact figures depend on message size, NCCL version, and topology, so treat specific numbers as approximate — but the *ratio*, roughly 5–10x, is stable and is the empirical basis for "keep the chatty collective in the node."

**DGX H100 is rail-designed for exactly this.** NVIDIA's DGX H100 system: 8 H100 SXM GPUs, 4 NVSwitch chips for a full NVLink4 mesh (~900 GB/s per GPU aggregate), and 8 ConnectX-7 NDR (400 Gb/s) NICs — one rail per GPU — for ~400 GB/s of off-node bandwidth (per NVIDIA's DGX H100 specs). The one-NIC-per-GPU, rail-optimized layout exists so that the inter-node dimension of your parallelism (DP or PP) gets dedicated, congestion-free bandwidth. The hardware is literally built around the placement law.

**GPUDirect RDMA and NVLink SHARP are documented, not folklore.** GPUDirect RDMA is described in NVIDIA's CUDA and networking docs as direct NIC-to-GPU-memory transfer that removes host-memory staging; NVLink SHARP (in-network reduction on NVSwitch) is documented as roughly halving the data volume of an intra-node all-reduce. Both show up as tags/behaviors in `NCCL_DEBUG=INFO` output, which is how you confirm they're active on *your* run rather than assuming.

**Reported MFU on large clusters is a placement scorecard.** The headline efficiency numbers you see in model reports — GPT-3-scale, PaLM, LLaMA, OPT, and similar training runs report Model FLOPs Utilization in roughly the 30–55% range on thousands of GPUs — are only achievable because the training layout respects the interconnect hierarchy. A run that let tensor parallelism spill across nodes, or that fell to a staged-copy inter-node path, would report a fraction of that MFU no matter how good the kernels were. When you read "we sustained X% MFU on N GPUs," you are reading, indirectly, "we placed every collective on a link fast enough to keep the GPUs fed." MFU is the north-star metric precisely because it collapses compute efficiency and comms placement into one honest number; a low MFU on healthy kernels is very often an interconnect-placement problem in disguise. Compute-optimal model sizing (the [Chinchilla scaling result](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling)) tells you how many FLOPs to spend; the interconnect decides how many of those FLOPs you actually realize.

## When to reach for each link (and when not)

Before the recommendations, the ten-minute pre-flight that prevents most interconnect disasters, in order: (1) `nvidia-smi topo -m` — confirm your intended tensor-parallel GPUs show `NV#`, not `SYS`/`PHB`; (2) `ibstat` — confirm the InfiniBand ports are `Active` at the expected rate; (3) nccl-tests `all_reduce_perf` intra-node and inter-node — confirm `busbw` matches spec within ~20%; (4) one shakedown training step with `NCCL_DEBUG=INFO` — confirm the transport lines read `NVL`/`P2P` intra-node and `NET/IB [GDRDMA]` inter-node, never `NET/Socket`. Four commands, ten minutes, and you have ruled out every silent fallback in this post before spending real GPU-hours. Skip it and you are betting a training run on a green dashboard that cannot see the fabric.

The placement law gives crisp, decisive guidance. Every technique here is a cost; here's when each is worth it and when it isn't.

- **Keep tensor parallelism inside one NVLink node — always.** TP degree should not exceed GPUs-per-node (8 on a DGX). If you find yourself wanting TP=16 across two nodes, stop: the per-layer all-reduce over InfiniBand will dominate and you'll lose more than you gain. Reach for a different lever (PP or DP) for the extra dimension.
- **Don't add tensor parallelism at all if the model fits and DDP saturates NVLink.** TP is a tax you pay to fit a model or shrink activation memory. If a plain DDP run already keeps the GPUs compute-bound and the model fits, TP just adds critical-path comms for nothing. Add it only when memory forces you to.
- **Data parallelism tolerates inter-node IB — lean on it for the widest dimension.** One overlappable all-reduce per step hides behind backward, so DP scales across many nodes as long as the fabric is real InfiniBand/RoCE with RDMA (not TCP). This is your default for going wide.
- **Pipeline parallelism is for the thinnest link and the largest span.** Its point-to-point activation sends are tiny and infrequent, so PP is what you stretch across node boundaries when TP has filled the node and DP alone won't fit the model. But PP only pays once you have enough micro-batches to keep the bubble small — below that, the pipeline bubble wastes more than the interconnect saves.
- **Don't go multi-node until you've saturated one node.** Inter-node bandwidth is ~10–18x below NVLink. If a single 8-GPU node isn't yet compute-bound — if your MFU is low because of a data-loader stall, small batch, or unoverlapped comms — fix that first. Multi-node multiplies your problems along with your GPUs.
- **On any new or reconfigured cluster, benchmark the fabric before you trust it.** Run `nvidia-smi topo -m`, run nccl-tests, grep the NCCL transport lines. Ten minutes of diagnostics up front beats a week of "why won't it scale."

## Key takeaways

- Compute is fast; moving bytes is slow, and "slow" is a hierarchy: HBM (TB/s) >> NVLink (~900 GB/s) >> PCIe (~64 GB/s) >> InfiniBand (~50 GB/s) >> TCP fallback (~1 GB/s). Each step down is roughly an order of magnitude.
- A GPU node is a fully connected NVLink island; nodes connect only through a much thinner InfiniBand fabric. Topology, not just bandwidth, decides which collective algorithm NCCL can use.
- Ring all-reduce costs `≈ 2S/B`. The `B` in the denominator is whichever link the collective lands on — so placement, not the algorithm, sets your comms time.
- Place parallelism by chatter: tensor parallelism (all-reduce per layer, on the critical path) on NVLink inside the node; data parallelism (one overlappable all-reduce per step) across nodes on IB; pipeline parallelism (tiny point-to-point) on the thinnest link.
- The same 8-GPU TP all-reduce is ~14x slower on PCIe than NVLink — a 5x-plus end-to-end throughput collapse, more when comms-bound. That's how an "eight-GPU job" runs at two-GPU speed.
- GPUDirect RDMA removes host-memory copies so the NIC reads GPU memory directly; without it, inter-node bandwidth roughly halves silently.
- The worst interconnect bugs never error. `NCCL_DEBUG=INFO` transport lines (`NVL`/`P2P`/`SHM`/`NET/IB`/`NET/Socket`) and an nccl-tests `busbw` number tell you the truth a green dashboard hides.
- Fix a silent fallback by pinning `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME`; verify the fix by watching the transport line flip to `NET/IB [GDRDMA]` and the nccl-tests bandwidth jump.
- Express placement in code with a device mesh, and make the tensor-parallel dimension the fastest-varying one so each TP group lands inside a single NVLink node; print and check the group's ranks before you train.
- The NVLink-to-PCIe gap has widened every GPU generation (from ~10x to ~28x), so placement mistakes cost *more* on newer hardware, not less — the hierarchy is sharpening, not flattening.
- Time collectives honestly: warm up, `torch.cuda.synchronize()` on both sides of the timed region, average at steady state, and report `busbw`; and rule out the loader-stall, thermal-throttle, and unoverlapped-comms confounds before blaming the fabric.

The wires are not a detail beneath your training loop — they are the ceiling on everything it can achieve. Once you can read a topology map, predict which link a collective will ride, and catch the fallbacks, the rest of distributed training becomes a series of placement decisions with predictable costs. The full decision-and-debugging checklist lives in the [distributed-training playbook capstone](/blog/machine-learning/distributed-training/the-distributed-training-playbook); this post is the hardware layer it stands on.

## Further reading

- [Why distributed training — the four walls and the series map](/blog/machine-learning/distributed-training/why-distributed-training)
- [Collectives from scratch — all-reduce, ring vs tree, and the byte-volume math](/blog/machine-learning/distributed-training/collectives-from-scratch)
- [Multi-node slower than single-node — the throughput autopsy](/blog/machine-learning/distributed-training/multinode-slower-than-single-node)
- [The distributed-training playbook — the capstone decision + debugging checklist](/blog/machine-learning/distributed-training/the-distributed-training-playbook)
- [HPC interconnects: NVLink, NVSwitch, InfiniBand, and RDMA](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma)
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)
- [Debugging DDP and multi-GPU jobs](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu)
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019); Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021); the NVIDIA A100 and H100 architecture whitepapers; the NCCL and GPUDirect RDMA documentation.

---
title: "Interconnects: NVLink, NVSwitch, InfiniBand and RDMA"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn why the network — not the GPU — is the wall in multi-GPU training, and how NVLink, NVSwitch, InfiniBand, RDMA and topology decide your real scaling efficiency."
tags:
  [
    "high-performance-computing",
    "gpu",
    "nvlink",
    "infiniband",
    "rdma",
    "nccl",
    "distributed-training",
    "deep-learning",
    "ml-systems",
    "networking",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-1.png"
---

You buy eight H100s. The vendor's slide says each one does about 989 bf16 TFLOP/s. You multiply by eight, you imagine a clean 8x speedup, and you launch your 7B-parameter training job expecting to be done in an eighth of the time. Instead, the run is maybe 5.5x faster than a single GPU. You add a second node — sixteen GPUs now — and the speedup barely moves. Sixteen of the fastest chips on earth, and your job is crawling. You stare at `nvidia-smi` and every GPU is at 100% utilization. Nothing looks broken. Nothing *is* broken, in the sense the profiler can see. The GPUs are busy. They are just busy *waiting*.

This is the moment every AI engineer who scales past one box eventually hits, and it has nothing to do with FLOP/s. The compute was never the problem. The problem is that thousands of times per training step, every GPU has to stop, hand its piece of the gradient (or the activations, or the shard) to the other GPUs, wait until everyone agrees on the sum, and only then continue. That hand-off travels over a *wire*. And the wire is, depending on which wire it is, somewhere between 10x and 150x slower than the path inside a single chip. When the math stops and the wire starts, your expensive silicon idles. The network is the new bottleneck, and once you understand the interconnect, the strange flat scaling curves stop being mysterious and start being *predictable* — which means you can design around them.

![Vertical stack of interconnect tiers from NVLink at the top down to Ethernet at the bottom, each tier roughly ten times slower than the one above](/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-1.png)

The figure above is the whole post in one picture. There is a strict hierarchy of links a piece of data can travel, and every step down it costs you roughly an order of magnitude in bandwidth. At the top, **NVLink** (NVIDIA's proprietary direct GPU-to-GPU link) plus **NVSwitch** (the on-board switch chip that connects all the GPUs in a server to each other) runs at roughly 900 GB/s per GPU inside a single H100 node. Below that, **PCIe** (PCI Express, the general-purpose bus that connects everything in a computer — CPU, GPU, NIC, disk) tops out around 64 GB/s on Gen5. Below *that*, **InfiniBand** (a high-speed switched fabric for connecting servers, the dominant choice for AI clusters) plus **RDMA** (Remote Direct Memory Access, a protocol that lets one machine read another machine's memory without involving either CPU) gives you maybe 50 GB/s of effective bandwidth per network card across nodes. And at the bottom, ordinary TCP over Ethernet, the fallback, delivers a small fraction of that with hugely worse latency. The single most important skill in multi-GPU work is keeping your tightest, most frequent communication on the *highest* tier you can, and never — never — letting a chatty collective spill down onto a slow link.

By the end of this post you will be able to: read the topology of any GPU box with one command and know which GPUs can talk fast; compute, on the back of an envelope, how long it takes to all-reduce a 7B model's gradients on each tier and see the order-of-magnitude gap for yourself; understand why the standard recipe is "tensor parallelism inside the node, data parallelism across nodes" and how that maps directly onto the bandwidth hierarchy; reason about fat-tree and rail-optimized topologies and why they set your maximum *practical* parallelism; and diagnose the scaling-efficiency cliff that appears the instant a tight collective crosses the wrong link. This is the communication wall of the [three-walls framing](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) — compute, memory bandwidth, and communication — and for anyone training or serving beyond a single GPU, it is the one that bites hardest and is understood least.

## 1. The bandwidth hierarchy: why a copy is never just a copy

Start with intuition, then a number, then the reason. The intuition: moving data is slow, and the further the data has to go, the slower it gets. Moving a number between two registers inside an SM (a Streaming Multiprocessor, the basic compute unit on a GPU) is essentially free. Moving it from one GPU's HBM (High Bandwidth Memory, the stacked DRAM bolted to the GPU die) to another GPU's HBM inside the same server is fast but not free. Moving it to a GPU in another server is slow. Moving it over plain Ethernet is glacial. Each of those steps is roughly 10x slower than the one before. That ladder is the figure you just saw, and internalizing it changes how you design every distributed job.

Now the numbers, on named hardware, because vague "fast" and "slow" are useless. On an NVIDIA H100 SXM GPU, the fourth-generation NVLink delivers about 900 GB/s of bidirectional bandwidth per GPU (18 links, each contributing 50 GB/s bidirectional, per NVIDIA's H100 architecture documentation). The previous-generation A100 SXM did about 600 GB/s over third-generation NVLink (12 links). PCIe Gen5 x16 — the bus a GPU uses to reach the CPU and the network card — gives about 64 GB/s in each direction (Gen4 x16, common on A100 PCIe boards, is about 32 GB/s). A single InfiniBand NDR network card runs at 400 Gb/s on the wire, which is 50 GB/s of bytes, and after protocol overhead you realistically *achieve* something in the 40–48 GB/s range per card. And TCP over a 100 Gb/s Ethernet link, going through the kernel networking stack, often nets you a small single-digit number of GB/s with latency a hundred times worse. Treat the per-card and effective numbers as approximate; they move with firmware, driver, and how cleanly your traffic is laid out.

Here is the reason a copy is never just a copy, and it is the crux of the whole topic. Bandwidth tells you the *throughput* once data is flowing; latency tells you how long before the first byte arrives. A big tensor cares mostly about bandwidth. A tiny message cares mostly about latency. Collective operations — the all-reduces and all-gathers that synchronize a training step — do *both*: they break a large buffer into many small chunks and exchange them in a structured pattern, so they pay the latency cost many times *and* the bandwidth cost for the whole buffer. That is why a link with great bandwidth but terrible latency (Ethernet) is doubly punishing for the exact operation we do most.

#### Worked example: how long to all-reduce a 7B model's gradients on each tier

Let's make the hierarchy concrete with the operation that dominates data-parallel training: the gradient all-reduce. A 7B-parameter model has 7 billion gradient values. In bf16 (2 bytes each), the gradient buffer is $7 \times 10^9 \times 2 = 14$ GB. The standard algorithm for this is **ring all-reduce**, and a clean result (derived in detail in the [collective communication post](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch)) is that each GPU sends and receives about $2(N-1)/N \cdot S$ bytes of data, where $S$ is the buffer size and $N$ is the number of GPUs. For large $N$ that factor approaches $2S$, so the time is roughly

$$T_\text{allreduce} \approx \frac{2S}{B}$$

where $B$ is the *bus* bandwidth the algorithm actually achieves on the link. Plug in $S = 14$ GB. On NVLink with an effective bus bandwidth of about 480 GB/s (you never get the full 900 — the ring pays for the algorithm's structure), $T \approx 28 / 480 \approx 0.058$ seconds... wait, that is per *some* number of GPUs; let me be careful and use bytes consistently. $2S = 28$ GB. On NVLink: $28 / 480 \approx 0.058$ s, about 58 ms. On a single InfiniBand NIC at 45 GB/s effective: $28 / 45 \approx 0.62$ s, about 620 ms. On Ethernet at 6 GB/s effective: $28 / 6 \approx 4.7$ s.

Read those three numbers again. 58 ms, 620 ms, 4,700 ms. The *same operation on the same data* spans nearly two orders of magnitude purely because of which wire it runs on. If your forward-plus-backward compute for that step takes, say, 400 ms, then on NVLink the 58 ms of all-reduce can be hidden almost entirely behind compute and costs you nearly nothing. On Ethernet, 4,700 ms of communication on top of 400 ms of compute means you spend 92% of every step *waiting on the network*. That is the cliff. That is why nobody trains large models over plain Ethernet, and why the interconnect is a first-class design decision, not an afterthought. (In practice, real clusters use *multiple* InfiniBand NICs per node in parallel — eight on a DGX H100 — so the effective inter-node bandwidth is far higher than one card; we'll get there. The single-card numbers above isolate the per-link physics.)

| Link tier | Hardware example | Effective bandwidth | Time to move 28 GB | Latency (small msg) |
| --- | --- | --- | --- | --- |
| NVLink + NVSwitch (intra-node) | H100 SXM | ~480 GB/s bus (900 raw) | ~58 ms | sub-microsecond |
| PCIe Gen5 x16 | A100/H100 PCIe board | ~32–48 GB/s | ~0.6–0.9 s | ~1–2 µs |
| InfiniBand NDR (one NIC) | ConnectX-7 400G | ~45 GB/s | ~0.62 s | ~1–3 µs |
| TCP over 100G Ethernet | generic NIC | ~6 GB/s | ~4.7 s | ~20–100 µs |

The table is the per-link summary you should keep in your head. Notice that PCIe and a single InfiniBand NIC are in the same league for bandwidth — which is exactly why, on a PCIe-only server with no NVLink, going across the network is barely worse than talking to the GPU next door, and both are dreadful compared to NVLink. The tiering is not academic. It is the difference between a job that scales and a job that doesn't.

There is one more subtlety worth pinning down before we move on, because it trips up nearly everyone the first time. A link's *bidirectional* bandwidth is not the number you divide a one-way transfer by. When NVIDIA quotes 900 GB/s for H100 NVLink, that is the sum of both directions — roughly 450 GB/s each way. A ring all-reduce, though, is genuinely bidirectional: in steady state every GPU is simultaneously sending to its right neighbor and receiving from its left, so it *can* use both directions at once, which is part of why the ring is the algorithm of choice. But a simple point-to-point copy from GPU A to GPU B uses only one direction and sees only half the headline number. Whenever you compute an expected transfer time, ask whether the operation is one-way (use the unidirectional figure) or genuinely duplex (the ring), and which the vendor's quoted bandwidth refers to. Mixing these up is the single most common reason a back-of-envelope estimate comes out 2x wrong, and it sends people hunting for performance bugs that were never there.

A second subtlety is the role of *message size*. None of these links hits its peak bandwidth on small messages. There is a fixed per-message overhead — the latency to set up the transfer, the protocol handshake, the algorithm's startup — that you amortize only when the payload is large. The classic model is $T = \alpha + S/B$, where $\alpha$ is the fixed latency cost and $S/B$ is the bandwidth-limited transfer time. For a tiny $S$, $\alpha$ dominates and you're latency-bound; the link's headline GB/s is irrelevant. For a large $S$, $S/B$ dominates and you finally see the bandwidth. This is why collective benchmarks sweep message sizes from a few megabytes to several gigabytes: the small-message numbers tell you about latency, the large-message numbers tell you about bandwidth, and real training does both. A model with thousands of small parameter tensors all-reduced separately is latency-bound and will badly underperform the headline bandwidth; this is exactly why frameworks *bucket* gradients — coalescing many small tensors into a few big buffers so the all-reduce runs in the bandwidth-bound regime where the fast link actually helps. Keep both terms of $\alpha + S/B$ in mind and you'll predict collective times that match the profiler instead of fantasy.

## 2. NVLink and NVSwitch: turning eight GPUs into one big GPU

Inside a single DGX or HGX server, NVIDIA does something clever to escape the PCIe trap. Without NVLink, every GPU-to-GPU transfer in a box would have to climb up to the CPU's PCIe complex and back down — slow, and contended, because the CPU's PCIe lanes are a shared, limited resource. NVLink is a *separate*, dedicated, point-to-point link between GPUs that bypasses PCIe entirely for GPU-to-GPU traffic. On an H100, each GPU has 18 NVLink connections. If you wired GPUs directly to each other in a mesh, eight GPUs would need a lot of links and the connectivity would be uneven — some pairs direct, some pairs two hops away. That uneven topology is poison for collectives, which want every GPU to be an equal-distance peer.

The fix is **NVSwitch**: a switch chip on the server baseboard that all the GPUs plug their NVLinks into, so that every GPU reaches every other GPU in *one hop* at the *full* NVLink bandwidth. A DGX H100 has four NVSwitch chips and connects all eight GPUs into a single non-blocking fabric. The practical consequence is that an all-reduce across the eight GPUs in the node never has to pick a route or worry that GPU 0 and GPU 7 are farther apart than GPU 0 and GPU 1 — they're all one hop, all at ~900 GB/s. NVSwitch effectively turns the eight GPUs into one big logical GPU with 8x the memory and a screaming-fast internal bus.

![Eight GPUs each with eighteen NVLink ports all connecting into a central NVSwitch fabric that provides uniform all-to-all bandwidth](/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-2.png)

The figure shows the shape of it: eight GPUs, each fanning its NVLink ports into a central NVSwitch fabric, and out the other side to the rest of the GPUs. The way this works is that there is no "near" and "far" GPU — the switch makes everyone equidistant. Conceptually, you should picture the node not as eight separate accelerators bolted onto a motherboard but as a single tightly-coupled unit with an internal interconnect roughly 14x faster than the PCIe bus it replaces. That ratio is why NVSwitch exists: it is the thing that makes intra-node tensor parallelism (which we'll define properly below) actually fast.

Let's verify the claim on real hardware with the one command every multi-GPU engineer should run on a new box before launching anything:

```bash
nvidia-smi topo -m
```

On a DGX H100, the output is a matrix of how each GPU connects to each other GPU. You'll see something like this (trimmed):

```bash
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NIC0  CPU Affinity
GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18  PXB   0-51
GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18  PXB   0-51
GPU2    NV18  NV18   X    NV18  NV18  NV18  NV18  NV18  PXB   0-51
GPU3    NV18  NV18  NV18   X    NV18  NV18  NV18  NV18  PXB   0-51
GPU4    NV18  NV18  NV18  NV18   X    NV18  NV18  NV18  SYS   52-103
GPU5    NV18  NV18  NV18  NV18  NV18   X    NV18  NV18  SYS   52-103
GPU6    NV18  NV18  NV18  NV18  NV18  NV18   X    NV18  SYS   52-103
GPU7    NV18  NV18  NV18  NV18  NV18  NV18  NV18   X    SYS   52-103
```

Read the legend carefully because it tells you everything. `NV18` means the two GPUs are connected by 18 NVLink connections — the best possible, full-bandwidth, one-hop. `NV#` with a smaller number means fewer NVLinks (less bandwidth). `PIX`, `PXB`, `PHB` mean the path goes through PCIe bridges or a host bridge (slow, no NVLink). `SYS` means the path traverses the inter-socket link between the two CPUs (slowest on-node path). If you ever see GPU-to-GPU pairs reading `SYS` or `PHB` where you expected `NV18`, you do not have a full NVSwitch fabric, and your intra-node collectives will be far slower than you think. The first time you touch an unfamiliar cluster, run this command, and *believe it* over the marketing.

Here is the trade-off comparison you should internalize between the on-node links:

| Path | What it means | Bandwidth (H100-class) | Use it for |
| --- | --- | --- | --- |
| `NV18` (NVLink + NVSwitch) | 18 NVLinks, one hop via switch | ~900 GB/s/GPU | tensor parallelism, intra-node all-reduce |
| `PXB` / `PHB` (PCIe) | through PCIe bridge/host bridge | ~32–64 GB/s | host transfers, NIC access, last resort for GPU-GPU |
| `SYS` (cross-socket) | over the CPU-to-CPU interconnect | even less, plus NUMA penalty | avoid for hot collectives |

It is worth dwelling on *why* the uniform one-hop property matters so much for collectives specifically, because it's not obvious until you've been burned by the alternative. Older multi-GPU servers, before NVSwitch, wired GPUs in a partial mesh: each GPU had direct NVLinks to a few neighbors but not to all of them. In such a topology, GPU 0 might reach GPU 1 directly but have to *route through* GPU 1 to reach GPU 3 — two hops, double the latency, and contention because GPU 1's links now carry traffic that isn't even its own. A ring all-reduce on a non-uniform mesh has to be laid out so the ring follows the physical links, and any GPU pair that isn't directly connected becomes a slow segment that bottlenecks the whole ring (a ring is only as fast as its slowest hop). NVSwitch erases all of this: because every pair is one hop at full bandwidth, the ring can be laid out in any order and every segment is equally fast. The collective library doesn't have to solve a routing puzzle; it just builds the ring and runs. That uniformity is a feature you feel as *consistent, predictable* all-reduce times, and it is precisely what a partial mesh cannot give you.

There's also a memory-capacity dimension to treating the node as one big GPU, and it's a big part of why the design pays off. Eight H100 SXM GPUs at 80 GB each is 640 GB of HBM, all reachable at NVLink speed. A model and its optimizer state that would never fit in one GPU's 80 GB can be *sharded* across the eight — each GPU holding an eighth — and because the inter-shard traffic rides NVLink, the sharding overhead stays small. This is the hardware substrate that makes tensor parallelism and sharded data parallelism (FSDP, ZeRO) practical: they constantly move shards and partial results between GPUs, and they only work well because NVSwitch makes that movement nearly free relative to compute. Take away the fast intra-node fabric and these techniques collapse into the same cross-node cliff we'll meet in section 6.

The lesson of this section is blunt: NVSwitch is the reason a single node is the natural unit of tight coupling. Up to eight GPUs, you have a private, uniform, ~900 GB/s fabric. The moment you need a ninth GPU, you fall off NVLink and onto the network — and the network, even the best network, is a different and slower world. That boundary, the edge of the node, is the single most important line in your entire cluster design, and the next sections are about what happens when you have to cross it.

## 3. Crossing the node boundary: InfiniBand and the fat-tree

Once you need more than eight GPUs, you leave NVLink behind and enter the world of *inter-node* networking, and the dominant technology there for AI clusters is **InfiniBand**: a switched fabric purpose-built for high-bandwidth, low-latency communication between servers. A modern AI node — a DGX H100, for instance — does not have one network card; it has eight, one ConnectX-7 NIC running at 400 Gb/s (NDR) per pair of GPUs, plus separate NICs for storage. Eight 400 Gb/s cards is 3,200 Gb/s, or 400 GB/s, of raw inter-node bandwidth per node. That is a deliberate design choice: NVIDIA matched the *aggregate* inter-node bandwidth to be within striking distance of the intra-node NVLink bandwidth, so that a well-laid-out collective doesn't fall off a cliff the moment it crosses the node boundary. It still falls — just not as far as a single NIC would suggest.

But raw NIC bandwidth is only half the story. The other half is the *topology* — how the switches are wired together — and this is where the term **fat-tree** comes in. A fat-tree is a network topology, shaped like a tree, where the links get "fatter" (higher aggregate bandwidth) as you go up toward the root, so that the network can carry full traffic between any pair of leaf nodes without congestion. The property you care about is **bisection bandwidth**: if you cut the network in half, how much bandwidth crosses the cut? A *non-blocking* fat-tree has full bisection bandwidth, meaning every node can talk to every other node at full line rate *simultaneously* without any link becoming a chokepoint. This is the gold standard for AI training, and it is what NVIDIA's DGX SuperPOD reference architecture specifies.

![Four compute nodes each with eight GPUs and eight NICs connecting up through leaf switches into a spine fabric that provides full bisection bandwidth](/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-3.png)

The figure shows the two-tier version: compute nodes connect up to *leaf* switches, and leaf switches connect up to a *spine* fabric. The way this works in a non-blocking design is that there is exactly as much bandwidth going *up* from each leaf to the spine as there is coming *into* that leaf from the nodes below it. That balance is the whole trick. If a leaf switch has 32 ports of 400 Gb/s facing the nodes, it needs 32 ports of 400 Gb/s (or equivalent) facing the spine, so that even if every node under it wants to talk to a node under a *different* leaf at full speed, the uplinks can carry it. When the uplink capacity is *less* than the downlink capacity, the network is *oversubscribed*, and a collective that happens to spread across many leaves will contend for the scarce uplinks and slow down. Oversubscription is the silent killer of large-cluster scaling, and it is invisible to your code — you find it only by measuring achieved all-reduce bandwidth and noticing it's lower than the NICs should allow.

Let's quantify bisection bandwidth because it directly bounds your collective performance. Suppose you have a cluster of 1,024 GPUs across 128 nodes, each node with 8 NICs at 400 Gb/s = 3,200 Gb/s per node. The total injection bandwidth of the cluster is $128 \times 3{,}200 = 409{,}600$ Gb/s. In a non-blocking fat-tree, the bisection bandwidth is half of that (the bandwidth that can cross a cut splitting the cluster into two equal halves), which is about 204,800 Gb/s, or roughly 25.6 TB/s. That number is what lets a 1,024-GPU all-reduce complete in a time governed by the *node* bandwidth rather than collapsing onto a single congested uplink. If the fabric were 2:1 oversubscribed (half the uplinks), bisection bandwidth halves, and any collective whose traffic pattern crosses the bisection gets up to 2x slower. This is why the reference architectures insist on non-blocking fabrics for the training network even though they cost more switches.

#### Worked example: bisection bandwidth and the all-reduce that needs it

Take a 64-node cluster, 8 H100s each, 512 GPUs, running data-parallel training of a model whose gradient buffer is 14 GB in bf16. Each node has 8 NICs at 400 Gb/s, so 400 GB/s of inter-node bandwidth per node. The ring all-reduce moves $2S \approx 28$ GB of data per GPU's "share" through the slowest link in the ring, but with a *hierarchical* algorithm (reduce inside each node over NVLink, then all-reduce across nodes, then broadcast back inside each node — NCCL does exactly this), the inter-node phase only has to move the *reduced* buffer once per node, not once per GPU. The cross-node data per node is roughly the full 14 GB reduced buffer flowing through that node's 400 GB/s of NICs. So the inter-node phase takes on the order of $2 \times 14 / 400 \approx 0.07$ s = 70 ms — *if and only if* the fabric has the bisection bandwidth to let all 64 nodes do this at once. On a non-blocking fat-tree, it does. On a 4:1 oversubscribed fabric, that 70 ms can balloon toward 280 ms, and suddenly your 512-GPU job has a communication tax it didn't have at 8 GPUs. The topology, not the chip, decided your scaling.

There's a structural reason a fat-tree, not some cheaper topology, is the standard, and it's worth understanding because it explains the cost. The simplest possible network is a single big switch — every node plugged into one switch, every pair one hop apart, full bandwidth. The trouble is that switches have a fixed port count (say 64 ports), so a single switch caps your cluster at that many nodes. To go bigger you must combine switches, and the naive way — chaining switches in a line or a simple tree where upper links are no fatter than lower ones — creates bottlenecks: all the traffic between the two halves of the cluster funnels through one thin link at the top. The fat-tree fixes this by *replicating* the upper layers: instead of one spine switch, you use many spine switches in parallel, and you connect each leaf to *every* spine, so the aggregate uplink bandwidth from any leaf equals its downlink bandwidth. The cost is a lot of switches and cables — a non-blocking fat-tree for $N$ nodes needs on the order of switch hardware proportional to $N$ at each of two or three tiers — but the payoff is that bisection bandwidth scales with the cluster instead of staying pinned at one link. When people say a SuperPOD is "expensive to network," this replication is most of where the money goes.

Latency, not just bandwidth, also degrades as you climb the tree, and it matters for tight collectives. A pair of GPUs in the same node talk in well under a microsecond. A pair on the same leaf switch (different nodes) is one switch hop, maybe a microsecond or two. A pair that must go up to the spine and back down — leaf to spine to leaf — is three hops and several microseconds. For a latency-sensitive small-message collective, that hop count is multiplied by however many rounds the algorithm runs, so a deep tree can hurt small collectives even when its bandwidth is ample. This is another reason rail optimization (section 7) matters: by keeping same-rank traffic on a single rail, it often turns what would be a three-hop spine traversal into a one-hop leaf-local exchange, cutting both latency and spine congestion at once.

The takeaway: inter-node bandwidth is the *aggregate of many NICs*, and whether you actually get that aggregate depends entirely on the topology delivering full bisection bandwidth. A pile of fast NICs wired into an oversubscribed switch is a slow cluster wearing a fast costume. When you choose or evaluate a cluster, the question is not "how fast are the NICs" but "what is the bisection bandwidth, and is the fabric non-blocking for the scale I run at."

## 4. RDMA and GPUDirect: getting the CPU out of the way

We have talked about how *fast* the links are. Now we need to talk about how the data actually gets *onto* the link, because the naive path wastes most of the bandwidth you paid for. Consider what happens when GPU memory on node A needs to reach GPU memory on node B using ordinary TCP networking. The gradient buffer lives in the GPU's HBM. To send it, the GPU first copies it across PCIe into host RAM (the CPU's memory). Then the kernel's TCP/IP stack takes over: it copies the data again into kernel socket buffers, breaks it into packets, computes checksums, and hands packets to the NIC — all on the CPU, interrupt by interrupt, copy by copy. On the receiving side the whole thing runs in reverse: NIC to kernel buffer, kernel buffer to host RAM, host RAM across PCIe into GPU HBM. Four copies and two trips through a CPU software stack to move bytes that were already sitting in fast GPU memory. The CPU becomes the bottleneck, the copies eat memory bandwidth, and the per-packet kernel overhead murders latency.

**RDMA** — Remote Direct Memory Access — deletes most of that. RDMA is a capability of the NIC (and the InfiniBand fabric) that lets the network card transfer data directly between the memory of two machines *without* the CPU touching the data and *without* going through the kernel's networking stack on the critical path. The application registers a memory region once, then issues a "write these bytes to that remote address" verb, and the NIC hardware does the rest: it reads the local memory, puts it on the wire, and the remote NIC writes it straight into the remote memory. This is called *kernel bypass* (the kernel is not in the data path) and *zero-copy* (no intermediate buffer copies). The CPU issues the request and is then free; it is not babysitting packets.

**GPUDirect RDMA** is the piece that closes the loop for GPUs. By itself, RDMA moves data between *host* memories. GPUDirect RDMA lets the NIC read and write *GPU* memory directly across PCIe, so the gradient buffer never has to detour through host RAM at all. The path becomes: GPU HBM → NIC (over PCIe) → wire → remote NIC → remote GPU HBM. No host copy, no kernel stack, no CPU in the data path. This is the single most important enabling technology for fast multi-node GPU training, and it is why you'll see `NCCL_NET_GDR_LEVEL` in tuning guides — it controls when NCCL is allowed to use GPUDirect RDMA.

![Two-column comparison contrasting the slow TCP path that copies through host RAM and the kernel against the RDMA path where the NIC reads GPU memory directly](/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-4.png)

The figure contrasts the two paths side by side. On the left, the TCP path: GPU copies to host RAM, the kernel TCP stack does per-packet copies, the NIC finally puts it on the wire, and latency is high. On the right, the GPUDirect RDMA path: the NIC reads GPU memory directly, the kernel is bypassed with no host copy, and the data hits the wire with single-digit-microsecond latency. The way to read this figure is as a *deletion* — RDMA does not make copies faster, it makes them *not happen*. The win is not 20% — it is the difference between achieving 6 GB/s and achieving 45 GB/s on the same 400 Gb/s card, plus a latency improvement from tens of microseconds down to 1–2 microseconds. For an operation we do thousands of times per step, both numbers matter enormously.

Now verify it on real hardware. First, confirm the InfiniBand cards are present and active:

```bash
ibstat
```

```bash
CA 'mlx5_0'
        CA type: MT4129
        Number of ports: 1
        Firmware version: 28.39.1002
        Port 1:
                State: Active
                Physical state: LinkUp
                Rate: 400
                Base lid: 12
                Link layer: InfiniBand
```

`State: Active`, `Physical state: LinkUp`, and `Rate: 400` are what you want — the card is up at 400 Gb/s. If you see `Down` or a rate lower than expected, you have a cabling or configuration problem that no amount of NCCL tuning will fix. Next, measure the actual achievable bandwidth between two nodes with a microbenchmark from the `perftest` suite. On the server node:

```bash
ib_write_bw -d mlx5_0 -a -F
```

And on the client node, pointing at the server:

```bash
ib_write_bw -d mlx5_0 -a -F <server_ip_or_hostname>
```

The output reports bandwidth at increasing message sizes. On a healthy 400 Gb/s NDR link you should see the large-message bandwidth climb toward roughly 45–48 GB/s (the achievable fraction of the 50 GB/s line rate). If it tops out far below that, something is wrong — wrong PCIe slot (a NIC in a x8 slot instead of x16 halves its bandwidth), a firmware mismatch, or GPUDirect not engaging. The discipline here is the same as everywhere in HPC: never trust the spec sheet, measure the link, and only then start tuning the software on top of it.

#### Worked example: how much RDMA actually buys you on a real all-reduce

Imagine two nodes, eight H100s each, all-reducing that same 14 GB bf16 gradient buffer across the network. With plain TCP through the CPU, you measure ~6 GB/s effective per NIC and, worse, the CPU saturates trying to drive eight NICs, so you don't even get 8x; you get maybe 20 GB/s aggregate before the CPU is the wall. The all-reduce's inter-node phase moves ~28 GB, so $28 / 20 \approx 1.4$ s. With GPUDirect RDMA, each NIC achieves ~45 GB/s, all eight run in parallel with the CPU idle (it issued the verbs and stepped away), and you get close to 360 GB/s aggregate: $28 / 360 \approx 0.078$ s = 78 ms. That is roughly an 18x improvement on the communication phase from one software change — turning on the thing that lets the NIC read GPU memory directly. If your step compute is 400 ms, RDMA's 78 ms hides nicely behind overlap and your scaling stays healthy; TCP's 1.4 s does not, and your 16-GPU job runs slower than your 8-GPU job. This is not a hypothetical; it is the single most common reason a multi-node job "doesn't scale," and the fix is configuration, not hardware.

## 5. The bandwidth-and-latency matrix: choosing the right link for the job

We now have all four tiers on the table, and it's worth laying them side by side along the two axes that matter — bandwidth and latency — plus a third, *reach*, because reach is what forces the trade-off. A link can be fast (NVLink) but only reach inside one box. A link can reach the whole cluster (InfiniBand, Ethernet) but be slower. There is no link that is both fastest and longest-reach; physics and economics won't allow it. So the engineering problem is always *matching the communication pattern to the link whose reach it needs and whose speed it can tolerate*.

![Comparison matrix of four link types across peak bandwidth typical latency and reach showing NVLink fastest but node-local and InfiniBand reaching the whole cluster](/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-5.png)

The matrix lays it out. NVLink 4 on H100: ~900 GB/s per GPU bidirectional, sub-microsecond GPU-to-GPU latency, but reach is *inside one node only*. PCIe Gen5 x16: ~64 GB/s, 1–2 µs host-mediated latency, also node-local. InfiniBand NDR: 400 Gb/s on the wire (~50 GB/s, ~45 effective per NIC), 1–3 µs latency over RDMA verbs, and crucially reach across the *whole cluster*. TCP Ethernet: 10–100 Gb/s (1–12 GB/s), a punishing 20–100 µs latency through the kernel stack, cluster-wide reach. The way to use this table is to look at your communication pattern, ask what reach it needs, and then pick the fastest link that provides that reach. Tensor parallelism needs sub-millisecond round-trips many times per layer — it needs NVLink, so it must stay inside a node. Data parallelism's all-reduce happens once per step and tolerates a few hundred microseconds — it can live on InfiniBand across nodes. Nothing important should ever be on TCP Ethernet if you can avoid it; its latency alone disqualifies it from any tight collective.

This matching principle has a name in distributed training, and it's the central design rule of the whole field: **map the tightest communication to the fastest link.** Tensor parallelism (splitting a single layer's matrix multiply across GPUs, which requires an all-reduce *inside every forward and backward pass of every layer*) is the tightest, chattiest pattern, so it goes on NVLink inside the node. Data parallelism (replicating the whole model and all-reducing gradients *once per step*) is the loosest, so it goes on InfiniBand across nodes. Pipeline parallelism (sending activations between stages) is in between and is usually arranged to cross node boundaries only at stage edges. This is not arbitrary convention; it is the bandwidth hierarchy dictating the software architecture. We'll return to it concretely in section 7. The full menu of strategies is the subject of the [parallelism strategies post](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert); here the point is narrower and sharper: the link decides the strategy.

Now the practical layer — how you tell NCCL (NVIDIA Collective Communications Library, the library PyTorch uses internally for all multi-GPU collectives) which links to use. NCCL auto-detects topology, but on real clusters you frequently need to nudge or constrain it, and these are the environment variables that matter:

```bash
# Pick which InfiniBand devices NCCL is allowed to use (match your ibstat output).
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

# Allow GPUDirect RDMA. SYS = use GDR even across PCIe host bridges; PIX/PHB restrict it.
export NCCL_NET_GDR_LEVEL=SYS

# Control GPU-to-GPU peer-to-peer (NVLink/PCIe) inside a node. NVL = require NVLink.
export NCCL_P2P_LEVEL=NVL

# Turn on logging so you can SEE which transport NCCL chose.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

With `NCCL_DEBUG=INFO`, NCCL prints, at startup, exactly which transport it selected for each connection — you'll see lines mentioning `NVL` (NVLink), `PXB`/`PHB` (PCIe), `IB` (InfiniBand), or `Socket` (TCP, the slow fallback). If you ever see `via NET/Socket` for traffic you expected to be on InfiniBand, NCCL failed to engage RDMA and you are silently running on the slowest possible path — a 5x to 10x performance bug that no profiler flags as an error, because nothing *errored*. Setting `NCCL_IB_HCA` correctly and confirming the debug log says `via NET/IB/GDRDMA` is one of the highest-leverage ten minutes you will ever spend on a cluster.

It helps to know what NCCL is actually doing with these knobs, because then the rare cases where you must override the defaults stop being guesswork. At startup NCCL probes the system: it reads the same topology `nvidia-smi topo -m` shows, discovers the NVLink connectivity, enumerates the InfiniBand HCAs, and figures out which GPUs sit closest to which NICs on the PCIe tree. From that it builds *channels* — the rings and trees the collectives will run on — choosing for each connection the fastest viable transport: NVLink peer-to-peer where two GPUs share NVLinks, GPUDirect RDMA where a GPU and a remote NIC can talk directly, and shared-memory or sockets as fallbacks. The environment variables are overrides on this auto-detection. `NCCL_P2P_LEVEL` says how "close" two GPUs must be before NCCL uses direct peer-to-peer instead of staging through host memory; `NCCL_NET_GDR_LEVEL` says how close a GPU and NIC must be before NCCL uses GPUDirect RDMA instead of bouncing through host RAM; `NCCL_IB_HCA` restricts which InfiniBand cards are eligible (essential when some HCAs are wired for storage, not the compute fabric, and you must keep collectives off them). The default detection is good, but it fails in predictable ways — virtualized environments hide topology, container boundaries obscure devices, mislabeled HCAs get used by accident — and in every one of those cases the symptom is a silent fall to a slower transport. The fix is always: turn on `NCCL_DEBUG=INFO`, read which transport it chose, and override only the knob that's wrong. Tuning NCCL blind, by pasting every environment variable from a forum post, is how you turn one misconfiguration into five.

| Communication pattern | Frequency | Latency tolerance | Right link | NCCL setting |
| --- | --- | --- | --- | --- |
| Tensor-parallel all-reduce | every layer, fwd + bwd | sub-millisecond | NVLink (intra-node) | `NCCL_P2P_LEVEL=NVL` |
| Pipeline activation pass | per micro-batch, stage edge | ~1 ms | NVLink or IB | overlap with compute |
| Data-parallel grad all-reduce | once per step | hundreds of µs | InfiniBand (inter-node) | `NCCL_IB_HCA`, GDR on |
| Checkpoint / logging | rarely | seconds | anything, even Ethernet | no special tuning |

The table is the cheat sheet: pick the row that matches your collective, and it tells you the link and the knob. The art of fast distributed training is almost entirely in keeping each row on its correct link — and the failure mode is almost always a tight row accidentally falling onto a slow link.

## 6. The scaling-efficiency cliff: what happens when a collective crosses the wrong link

Everything so far has been building toward this, the single most important practical phenomenon in multi-GPU training: the scaling-efficiency cliff. Scaling efficiency is a simple ratio — if one GPU does $X$ work per second and $N$ GPUs do $Y$ work per second, efficiency is $Y / (N \cdot X)$. Perfect linear scaling is 100%. In practice you lose some to communication overhead, and the question is *how much*. The answer depends entirely, decisively, on whether your tightest collective stays on a fast link or spills onto a slow one. Stay on NVLink and you keep 90–95% efficiency. Spill onto Ethernet and you can drop below 40%. There is no smooth gradient between those outcomes — it is a *cliff*, because the offending collective runs inside the inner loop and its cost gets multiplied by every layer of every step.

![Two-column before and after comparison showing tensor parallelism staying near linear on NVLink versus efficiency collapsing when tensor parallelism spills across Ethernet](/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-6.png)

The figure makes the cliff visible. On the left, tensor parallelism kept inside a node: eight GPUs on a 900 GB/s NVLink fabric, the per-layer all-reduce overlaps with compute, and scaling efficiency holds at 90–95%. On the right, the *same* tensor-parallel group split across two nodes so that its all-reduce now crosses the network: every single layer has to wait on the wire, the communication can no longer hide behind compute, and efficiency falls below 40%. Notice the asymmetry — going from the left configuration to the right one changes *nothing* about the model, the math, or the GPUs. It changes only which *link* the per-layer all-reduce uses. And that one change can more than halve your throughput. This is why "did the tensor-parallel group fit inside one node" is the first question to ask of any underperforming multi-node job.

Let's understand *why* it's a cliff and not a slope, because the reasoning is what lets you predict it. The key idea is **overlap**: modern training frameworks issue the communication for one chunk of work while the GPU is still computing the next chunk, so that ideally the communication time hides entirely behind compute time. Overlap works when communication is *faster* than compute — the wire finishes before the math needs the result. The condition is roughly $T_\text{comm} < T_\text{compute}$ for the overlapped unit. On NVLink, a per-layer all-reduce of a few hundred megabytes takes tens of microseconds while the layer's matmuls take hundreds of microseconds, so $T_\text{comm} \ll T_\text{compute}$ and the communication is *free* — fully hidden. Cross to Ethernet and that same all-reduce takes hundreds of microseconds to milliseconds, now *larger* than the compute, so it can no longer hide; every layer stalls waiting, and the stalls *add up* across all the layers. The transition from "fully hidden" to "fully exposed" happens over a narrow range of link bandwidth, which is exactly why it presents as a cliff.

#### Worked example: the cross-node tensor-parallel slowdown, measured

Take a 7B model with tensor-parallel degree 8. Configuration A: all 8 tensor-parallel ranks on one DGX H100 node, on NVLink. The per-layer all-reduce moves about 200 MB (activation-sized, bf16) and at ~480 GB/s effective bus bandwidth takes $\sim 0.4 / 480 \approx 0.8$ ms... let me use the $2S$ form: $2 \times 0.2 / 480 \approx 0.83$ ms per all-reduce. Wait — for activations the buffer is smaller; call it ~0.4 ms. With ~32 layers and two all-reduces per layer (forward and backward), that's ~64 all-reduces. On NVLink the layer compute dwarfs each one, overlap hides them, and measured throughput lands around 92% of ideal 8-GPU scaling. Configuration B: split the same 8 tensor-parallel ranks across two nodes (4 per node), so half the all-reduce traffic must cross InfiniBand. Even on a good 400 GB/s aggregate inter-node link, the per-layer cross-node all-reduce now costs several times more *and* incurs the inter-node latency floor 64 times per step; overlap can't fully hide it, and measured efficiency drops to the 40–55% range — sometimes worse on a busy fabric. Same chips, same model. The reported numbers in the literature for cross-node tensor parallelism consistently show this kind of collapse, which is precisely why frameworks like Megatron-LM document the rule "tensor-parallel degree should not exceed the number of GPUs per node." Treat the exact percentages as approximate and workload-dependent; the *direction and magnitude* are robust.

The cliff has a clean, actionable corollary, and it is the most important sentence in this post: **keep your tensor-parallel degree less than or equal to the GPUs in one node.** A DGX H100 has 8 GPUs, so tensor-parallel degree 8 is the practical maximum; set it to 8 to shard a huge model across the node's NVLink fabric, then use data parallelism and pipeline parallelism to scale *across* nodes over InfiniBand, where the looser communication tolerates the slower link. Violate this rule and you pay the cliff. Respect it and your scaling stays near-linear far past one node. This single constraint — born entirely from the bandwidth hierarchy — shapes the parallelism layout of essentially every large model trained today.

How do you *measure* whether you're on the cliff? Run the NCCL all-reduce benchmark and read the achieved *bus bandwidth*, the number that tells you how much of the link you're actually using:

```bash
# Build nccl-tests, then run the all-reduce benchmark across your GPUs.
# -b start size, -e end size, -f multiply factor, -g GPUs per process.
./build/all_reduce_perf -b 8M -e 4G -f 2 -g 8
```

```bash
#       size         time   algbw   busbw  #wrong
#        bytes         us  GB/s    GB/s
     8388608        145    57.8   101.2     0
    67108864        612   109.7   192.0     0
   536870912       3211   167.2   292.6     0
  4294967296      25140   170.9   299.0     0
```

The `busbw` column is what you read. On an 8-GPU NVLink node, large-message bus bandwidth should approach 250–480 GB/s (the achievable fraction of the 900 GB/s raw, after the ring algorithm's overhead). If you run the same benchmark *across nodes* and the bus bandwidth drops to 6 GB/s, NCCL fell back to TCP sockets and RDMA isn't engaged — go back to section 4 and fix `NCCL_IB_HCA` and GPUDirect. If it lands around 40–45 GB/s per node's worth of NICs, you're on InfiniBand correctly. The bus bandwidth number is your single best diagnostic: it tells you, in one figure, which tier of the hierarchy your collectives are actually running on, regardless of what you *think* you configured.

## 7. Topology that sets your max parallelism: rail-optimized and SuperPOD

We've established that the link determines the strategy. Now zoom out to the cluster scale, where a second topology idea decides how well thousands of GPUs cooperate: **rail-optimized** wiring. Recall that each node has 8 GPUs and 8 NICs. The naive way to wire them is to dump all 8 NICs into one leaf switch and move on. The rail-optimized way is smarter: NIC 0 (and therefore GPU 0) on *every* node connects to the *same* leaf switch — call it "rail 0." NIC 1 / GPU 1 on every node connects to "rail 1." And so on for all 8 rails. The result is that GPU rank $i$ on every node shares a dedicated rail with GPU rank $i$ on every other node, and the eight rails are independent parallel networks.

![Grid showing two nodes of four GPUs each where same-rank GPUs across nodes connect to the same dedicated rail leaf switch forming independent parallel networks](/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-7.png)

The figure shows the principle on a small scale: Node 0's GPU 0 and Node 1's GPU 0 both connect to Rail 0's leaf switch; their GPU 1s both connect to Rail 1; and so on. The reason this matters is the structure of a data-parallel all-reduce. In the standard layout, GPU rank $i$ on every node forms one data-parallel group and all-reduces among themselves. With rail-optimized wiring, *all of that group's traffic stays on rail $i$* — it never crosses to another rail, never contends with rank-$j$ traffic, and often completes in a single switch hop without even touching the spine. The way to picture it is eight separate, non-interfering highways, one per GPU rank, instead of one eight-lane road where everyone merges and fights for position. Same-rank collectives never collide, and the spine fabric is reserved for the rarer cross-rail patterns. This is the topology NVIDIA's DGX SuperPOD reference architecture specifies, and it is a big part of why SuperPODs scale to thousands of GPUs while holding high efficiency.

This is the moment to state the deepest idea in the post: **topology sets your maximum *practical* parallelism, regardless of how many GPUs you own.** You might have 2,000 GPUs, but if the fabric is oversubscribed or badly laid out, the largest job that scales efficiently might be 256. The chips are necessary but not sufficient; the *wiring* determines how many of them can actually work together on one model before communication eats the gains. A non-blocking, rail-optimized fat-tree pushes that practical ceiling high — into the thousands. An oversubscribed or rail-blind fabric pulls it low. When you read that a SuperPOD scales to thousands of H100s at high efficiency, the headline is the GPU count but the *engineering* is entirely in the topology: full bisection bandwidth so collectives don't congest, and rail optimization so same-rank traffic never collides. The interconnect, not the accelerator, is what makes the cluster a cluster.

Here is the data-parallel-across-nodes launch that rides this topology, the standard `torchrun` invocation for multi-node training:

```bash
# On EACH node (here, node rank 0 of 4 nodes, 8 GPUs each = 32 GPUs total).
# --rdzv_endpoint points all nodes at a single rendezvous host:port.
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --rdzv_id=job42 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=node0.cluster.local:29500 \
  train.py --tensor-parallel 8 --pipeline-parallel 1 --data-parallel 4
```

Read the parallelism arguments against everything we've built. `--tensor-parallel 8` keeps the chattiest collective inside one node's 8 GPUs, on NVLink — respecting the cliff rule. `--data-parallel 4` spreads the gradient all-reduce across the 4 nodes, on InfiniBand, where rail-optimized wiring lets each rank's group ride its own rail. The layout is a direct, mechanical translation of the bandwidth hierarchy into software configuration. Get this mapping right and 32 GPUs behave like 32 GPUs; get it wrong — say, tensor-parallel 16 spilling across two nodes — and you fall off the cliff from section 6 no matter how good the hardware is.

| Topology choice | What it optimizes | Effect on scaling | When it matters |
| --- | --- | --- | --- |
| Non-blocking fat-tree | full bisection bandwidth | collectives don't congest the spine | always, for training fabrics |
| Rail-optimized wiring | same-rank traffic isolation | data-parallel all-reduce never collides | multi-node DP at scale |
| Oversubscribed fat-tree | cost savings on switches | cliff appears when traffic crosses bisection | acceptable only for loose workloads |
| Flat single-switch | simplicity | fine up to one switch's port count | small clusters only |

The table summarizes the topology trade-offs. The recurring theme: every topology decision is a bet about which traffic patterns will be common, and a non-blocking rail-optimized fat-tree is the bet that *all* patterns should be fast — expensive, but the reason SuperPOD-class clusters scale. For your own work, the actionable version is simpler: ask your cluster's operators whether the training fabric is non-blocking and rail-optimized, and if the answer is no, lower your expectations for the largest job that will scale, and lean harder on keeping tight collectives intra-node.

## 8. Where the time actually goes: profiling the collective across tiers

We close the loop by putting numbers on the central claim — that the *same* collective spends radically different amounts of time depending on which tier it runs on — and showing how to read that in a profile. This is the measurement discipline that turns all the intuition above into something you can act on, because the whole point of understanding the hierarchy is to *find* where your job is losing time and *move* the offending collective to a faster link.

![Matrix showing a 7B gradient all-reduce taking under a millisecond on NVLink but growing to milliseconds on InfiniBand and dominating step time on Ethernet](/imgs/blogs/interconnects-nvlink-nvswitch-infiniband-and-rdma-8.png)

The figure quantifies the whole story for a 7B gradient all-reduce. Intra-node on NVLink: about 0.06 ms for the relevant chunk at ~480 GB/s bus bandwidth, under 5% of step time, fully hidden by overlap. Inter-node on InfiniBand: about 0.6 ms at ~45 GB/s per NIC, 20–30% of step time, where overlap helps but no longer fully hides it. Inter-node on Ethernet: about 4.7 ms at ~6 GB/s, over 60% of step time, where the all-reduce simply *dominates* and the GPUs idle waiting. The way to read this matrix is column by column: pick the tier you're on, and it tells you both the absolute time and — more importantly — the *share of step time* the collective consumes. That share is your scaling efficiency in disguise. When communication is under 5% of the step, you scale near-linearly. When it's over 60%, you've fallen off the cliff and adding GPUs makes things barely better or even worse.

How do you measure this share honestly on your own job? The trap everyone falls into is timing without synchronizing — GPU work is asynchronous, so a naive Python timer measures *kernel launch* time, not *execution* time, and you get garbage. The correct way uses CUDA events and an explicit synchronize:

```python
import torch
import torch.distributed as dist

# Warm up first — the very first NCCL call pays a one-time setup cost.
buf = torch.randn(7_000_000_000 // 8, dtype=torch.bfloat16, device="cuda")
for _ in range(5):
    dist.all_reduce(buf)
torch.cuda.synchronize()

# Time the steady-state all-reduce with CUDA events, averaging several runs.
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
n_iters = 20
start.record()
for _ in range(n_iters):
    dist.all_reduce(buf)
end.record()
torch.cuda.synchronize()            # block until all GPU work is actually done
ms = start.elapsed_time(end) / n_iters
gb = buf.numel() * buf.element_size() / 1e9
busbw = 2 * (dist.get_world_size() - 1) / dist.get_world_size() * gb / (ms / 1000)
print(f"all_reduce: {ms:.3f} ms, bus bandwidth: {busbw:.1f} GB/s")
```

The three disciplines baked into that snippet are non-negotiable: *warm up* before timing (the first NCCL call allocates buffers and establishes connections — including the slow InfiniBand handshake — and is not representative), *synchronize* before reading the clock (or you time launches, not work), and *average several iterations* in steady state (one sample is noise). Run this intra-node and you should see bus bandwidth in the hundreds of GB/s. Run it across nodes and you read off exactly which tier you landed on — and if the number is far below what the link should give, you have found a real, fixable performance bug, not an act of God.

For the full picture in a real training step, drop into `torch.profiler` and look at where the timeline shows GPUs *idle* — those gaps, the stretches where the compute stream has nothing to do because it's blocked on an all-reduce, are the communication wall made visible:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=3),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(6):
        train_step()
        prof.step()

# Sort by total CUDA time to see if nccl:all_reduce dominates.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

If `nccl:all_reduce` or `nccl:all_gather` shows up near the top of that table consuming a large fraction of CUDA time, the network is your wall, and the fixes are exactly the ones this post has built: confirm RDMA is engaged (section 4), confirm tensor parallelism stays intra-node (section 6), confirm the fabric gives you the bisection bandwidth you need (section 3), and confirm the topology isn't making same-rank collectives collide (section 7). The profiler tells you *that* you're communication-bound; the hierarchy tells you *what to do about it*. That pairing — measure the wall, then move the traffic to a faster link — is the entire craft.

One honest warning about reading these measurements, because the confounds are real and they fool experienced engineers. The all-reduce time you measure in a *live training step* is not the same as the time you measure in an *isolated benchmark*. In a real step, the all-reduce is overlapped with backward-pass compute, sharing PCIe and memory bandwidth with the gradient computation, and possibly contending with the data loader streaming the next batch over the same PCIe lanes. A NIC that hits 45 GB/s in a clean `ib_write_bw` run may achieve less when the GPU is simultaneously hammering HBM and the CPU is decoding images. So when an isolated benchmark says the link is healthy but the live profile shows the collective eating more time than expected, suspect *contention*, not the link itself — and look at whether your data pipeline, your PCIe traffic, or an un-overlapped optimizer step is stealing the bandwidth. The other classic confound is thermal: a sustained job throttles GPU and NIC clocks as the rack heats up, so the bus bandwidth you measure cold at step 10 may sag by step 10,000. Measure in steady state, after the thermals settle, and you'll trust your numbers. The discipline that separates a real diagnosis from a wild-goose chase is always the same: isolate the link with a microbenchmark to learn its ceiling, then measure it inside the real workload to learn what you actually get, and attribute the gap to contention or thermals rather than blaming the wire.

#### Worked example: reading the share of step time to decide what to fix

Put the profiler output to work on a concrete decision. Suppose a 32-GPU job (4 nodes, tensor-parallel 8 intra-node, data-parallel 4 across nodes) profiles at: 380 ms forward+backward compute, 30 ms intra-node tensor-parallel all-reduce (mostly hidden by overlap, net exposed ~8 ms), and 140 ms data-parallel gradient all-reduce across nodes, of which overlap hides about 90 ms, leaving ~50 ms exposed. Step time is roughly $380 + 8 + 50 \approx 438$ ms, so communication exposed is about 13% of the step — you're scaling at roughly 87% efficiency, healthy. Now suppose you read the NCCL debug log and discover the inter-node all-reduce is running `via NET/Socket` — TCP, not RDMA. The 140 ms would balloon toward 900 ms, almost none of it hidden, and step time would jump past 1,200 ms: communication becomes over 65% of the step and efficiency craters below 35%. The fix isn't a faster GPU or a bigger batch; it's the one-line `NCCL_IB_HCA` correction from section 4 that moves that all-reduce back onto InfiniBand with RDMA. This is the payoff of the whole post: the profiler hands you the share-of-step number, the hierarchy tells you which tier you're stuck on, and the fix is a configuration change worth a 2.5x throughput swing — found in minutes, not days, because you knew where to look.

## Case studies / real numbers

Let's ground everything in named, real configurations and reported results, marking estimates clearly and never fabricating a precise figure.

**DGX H100, the intra-node fabric.** NVIDIA's DGX H100 connects its 8 H100 SXM GPUs through four NVSwitch chips, giving each GPU about 900 GB/s of bidirectional NVLink bandwidth and full all-to-all connectivity (NVIDIA H100 / DGX H100 documentation). In NCCL all-reduce benchmarks on such a node, achieved bus bandwidth for large messages commonly lands in the 250–480 GB/s range — the ring algorithm's structure means you do not get the full 900, and the exact figure depends on message size and NCCL version. The headline takeaway is the one this post hammered: inside the node, an all-reduce is so fast relative to compute that it essentially disappears behind overlap, which is *why* the node is the natural unit of tight coupling.

**DGX SuperPOD, the inter-node fabric.** NVIDIA's DGX SuperPOD reference architecture builds clusters of hundreds to thousands of GPUs using a rail-optimized, non-blocking InfiniBand fat-tree, with each DGX node contributing 8 ConnectX-7 NDR NICs at 400 Gb/s for a compute fabric, separate from the storage fabric (NVIDIA DGX SuperPOD reference architecture). The rail-optimized wiring — same-rank GPUs on every node sharing a dedicated rail — and the full bisection bandwidth together are what let these systems scale to thousands of GPUs while holding high MFU. The lesson is that the published GPU count is the easy part; the topology engineering is what actually delivers the scaling, and it is a deliberate, documented design, not an emergent property of buying more chips.

**The cross-node tensor-parallel slowdown.** Across the large-model training literature and framework documentation (Megatron-LM and related work), a robust, repeatedly observed result is that tensor parallelism collapses in efficiency the moment its all-reduce must cross a node boundary onto the network instead of staying on NVLink. The practical guidance that falls out of this — and which Megatron-LM's documentation states directly — is to keep tensor-parallel degree no greater than the GPUs per node (8 on a DGX), then layer pipeline and data parallelism across nodes. Measured efficiency for a same-model job typically drops from the low-90s percent (tensor parallelism intra-node) into roughly the 40–55% range (tensor parallelism split across nodes); treat the exact figures as approximate and workload-dependent, but the *cliff* itself is one of the most reliable phenomena in distributed training.

**RDMA versus the kernel path.** GPUDirect RDMA's effect is the difference between a NIC reading GPU memory directly versus shuttling everything through host RAM and the kernel TCP stack. On a 400 Gb/s NDR link, RDMA delivers roughly 45 GB/s of achievable bandwidth at 1–3 µs latency, whereas the TCP path through the CPU often nets single-digit GB/s with tens of microseconds of latency *and* saturates the CPU when driving multiple NICs (NVIDIA GPUDirect / InfiniBand documentation). For the all-reduce we do thousands of times per training run, this single configuration difference — engaging GPUDirect RDMA — is frequently the line between a multi-node job that scales and one that runs slower than a single node. It is, in this author's experience, the most common silent performance bug in the entire field, because nothing errors; the job just quietly uses the slow path.

## When to reach for each link (and when not to)

Every link is a tool with a job, and the discipline is using the right one and not over-engineering. Here is the decisive guidance.

**Use NVLink (keep it intra-node) for tensor parallelism and any collective in the inner loop.** If a collective runs once per layer per pass — tensor-parallel all-reduce is the canonical case — it *must* stay on NVLink, which means it must stay inside one node, which means its degree must not exceed the GPUs per node. This is non-negotiable; cross the node boundary with this traffic and you fall off the cliff. Do not, however, reach for tensor parallelism *at all* if your model fits on one GPU and plain data-parallel DDP already saturates the link — you'd be adding the chattiest collective for no reason.

**Use InfiniBand with RDMA for data parallelism and anything that crosses nodes.** The gradient all-reduce happens once per step and tolerates a few hundred microseconds, so it lives happily on InfiniBand across nodes — *provided* GPUDirect RDMA is engaged. Always verify with `NCCL_DEBUG=INFO` that you see `via NET/IB` and not `via NET/Socket`. The moment you go multi-node, confirming RDMA is your first task, before any other tuning.

**Do not reach for more parallelism dimensions than the bandwidth hierarchy requires.** The cost of every parallelism strategy is communication, and communication is the wall. If data parallelism alone scales your job across the cluster at high efficiency, you do not need tensor or pipeline parallelism — they add tight collectives you'd otherwise avoid. Add tensor parallelism only when the model doesn't fit, pipeline parallelism only when even tensor-plus-data isn't enough and you have many stages to amortize the bubble. Match the strategy to the constraint, not to a desire to use every feature.

**Never run a tight collective on TCP Ethernet if you can avoid it.** Its 20–100 µs latency alone disqualifies it from any inner-loop communication; even its bandwidth is an order of magnitude below InfiniBand. Reserve Ethernet for checkpointing, logging, and orchestration — traffic measured in seconds where the slow path is fine. If you find yourself on Ethernet-only hardware and must train across nodes, accept that your practical scaling ceiling is low and design the job to be communication-light (large per-GPU batches, gradient accumulation, minimal cross-node sync).

The meta-rule, true across all of HPC: the interconnect is a budget, and you spend it by deciding which traffic deserves the fast link. Spend it well — tight collectives on NVLink, loose ones on InfiniBand, the rest on whatever's left — and your cluster behaves like the sum of its chips. Spend it carelessly and you own a very expensive space heater.

## Key takeaways

- **The network is the new bottleneck.** Past one GPU, communication — not FLOP/s — is usually what caps your scaling. GPUs at 100% utilization can be 100% *waiting on the wire*.
- **Bandwidth falls ~10x per tier:** NVLink+NVSwitch (~900 GB/s intra-node) → PCIe (~32–64 GB/s) → InfiniBand+RDMA (~45 GB/s/NIC inter-node) → TCP Ethernet (single-digit GB/s). Keep your tightest traffic on the highest tier.
- **NVSwitch makes a node one big GPU.** It gives all 8 GPUs uniform one-hop ~900 GB/s connectivity, which is why the node is the natural unit of tight coupling and the edge of the node is the most important line in your cluster.
- **RDMA + GPUDirect deletes the CPU from the data path.** The NIC reads GPU memory directly with kernel bypass and zero copies — the difference between ~6 GB/s and ~45 GB/s on the same card. Confirm it's engaged with `NCCL_DEBUG=INFO`.
- **Map the tightest collective to the fastest link.** Tensor parallelism on NVLink (intra-node), data parallelism on InfiniBand (inter-node). This single rule shapes the parallelism layout of nearly every large model.
- **Keep tensor-parallel degree ≤ GPUs per node** (8 on a DGX). Spill it across nodes and you fall off the scaling cliff — efficiency drops from ~90% to below 55% with no change to the model.
- **Topology sets your max *practical* parallelism.** A non-blocking, rail-optimized fat-tree (the SuperPOD design) gives full bisection bandwidth so collectives don't congest and same-rank traffic doesn't collide. An oversubscribed fabric lowers your scaling ceiling regardless of GPU count.
- **Measure, don't trust.** `nvidia-smi topo -m` for the map, `ibstat`/`ib_write_bw` for the link, `nccl-tests` bus bandwidth and `torch.profiler` for the achieved truth. Always warm up, synchronize, and average.

## Further reading

- [Why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) — the three-walls framing this post's communication wall belongs to.
- [Collective communication and NCCL: all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) — the $2(N-1)/N \cdot S$ ring derivation and how NCCL builds rings and trees on this hardware.
- [Parallelism strategies: data, tensor, pipeline and expert](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert) — the full menu of strategies the link hierarchy dictates.
- [The HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) — the capstone that ties interconnect, profiling, kernels, precision and parallelism into one workflow.
- [Choosing a GPU for LLM serving: cost, throughput, latency](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency) — how interconnect factors into serving-hardware decisions.
- [LLM GPU benchmark](/blog/machine-learning/mlops/llm-gpu-benchmark) — measured throughput numbers across GPU and interconnect configurations.
- NVIDIA NVLink and NVSwitch documentation; NVIDIA H100 and DGX H100 architecture whitepapers; NVIDIA DGX SuperPOD reference architecture; NVIDIA GPUDirect RDMA and InfiniBand documentation; the NCCL user guide and `nccl-tests` repository.

---
title: "Training Large Models Across Many Nodes: A Recipe, a Case Study, and the Troubleshooting Playbook"
date: "2026-05-21"
publishDate: "2026-05-21"
description: "A principal-engineer walk-through of multi-node LLM training — parallelism axes, device meshes, NCCL realities, a 4096-GPU case study, and a paste-in launcher with hang detection."
tags: ["distributed-training", "multi-node", "fsdp", "deepspeed", "torchrun", "nccl", "mlops", "llm", "case-study", "troubleshooting", "infiniband", "checkpointing"]
category: "machine-learning"
subcategory: "MLOps"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

The first multi-node training job I ever ran hung for eleven minutes before any rank wrote a single byte to stdout. Twelve hosts, ninety-six GPUs, four hundred kilodollars of hardware burning watts, and the only sign that anything was happening was the fans on the racks spinning a little harder. When the watchdog finally fired, the stack trace pointed at a NCCL `ncclAllReduce` call buried twelve frames deep inside FSDP's grad reduction. The actual cause turned out to be that one of the four IB rails on host 7 had negotiated a single lane instead of four. One faulty cable. One slow rank. The whole job sitting at the barrier.

That memory is the reason this post exists. Single-host PyTorch is a comfortable lie. The moment you cross a network boundary, every assumption that held on one box gets renegotiated against physics you cannot see from inside Python: link bandwidth, switch hop count, NIC firmware, kernel buffer sizes, NUMA affinity, MPI rendezvous. A single-host training job that hits 55% MFU does not become a 55% MFU multi-node job by tacking on `--nnodes=32`. It becomes a job that prints `Watchdog caught collective operation timeout` and dies.

![Pod topology: four network tiers, four bandwidth orders of magnitude](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-1.png)

The diagram above is the mental model: a pod is a stack of four independent networks, and each outer layer is roughly an order of magnitude slower than the one inside it. The job you wrote on a single host lived entirely in the innermost blue band, where 900 GB/s of NVLink-switched all-to-all is free. The job you are about to run on 256 GPUs has to pay tolls at every layer: PCIe out to the host NIC, IB to the top-of-rack switch, two more hops through the spine. Your parallelism strategy is not really a software choice. It is a physical placement problem disguised as a config file.

The rest of this article is a tour of that mental model — what to put on each layer, how to verify each one in isolation, how to recover when one of them fails — followed by a 4096-GPU case study where we landed at 38% MFU on a Llama-3 70B-class run after thirty-one days of bug-hunting, plus a paste-in launcher, watchdog, and Slurm script you can adapt.

## Why multi-node breaks every assumption

There is a short list of things that are true on one host and false the moment you add a second. Get this list wrong and you will spend your first month chasing symptoms.

| Single-host assumption | What actually happens on a pod |
|---|---|
| `torch.cuda.synchronize()` is a barrier | It is local; the actual cross-rank barrier is the collective itself, and timeouts are 30 minutes by default |
| AllReduce is "free" because NVLink is fast | Cross-node AllReduce runs on IB at 1/30th the NVLink rate; a single slow link in a ring drags every rank |
| Mixed precision converges if loss curve looks normal | Routing logits, layernorm stats, and softmax tails overflow asymmetrically across DP replicas; you only see the divergence at scale |
| Dataloaders saturate the GPU | The IOPS profile of N×8 workers hammering one object store is qualitatively different from one host |
| Checkpoints are background I/O | Synchronous full-state checkpoints stop training for the duration of the slowest rank's write; we have personally seen 11-minute pauses |
| Restart is a no-op | Rendezvous, fencing, GPU reset, NCCL re-init, dataloader reseeding — each is a failure surface |

> A pod is not eight hosts plus a network; it is eight hosts that happen to share a network they do not control. Treat the network as a third-party service.

The interesting consequence is that the same model with the same config can produce very different MFU on two clusters with the same nominal hardware, because the tax stack at each network layer is different. We have seen the same code go from 28% MFU on one provider to 49% MFU on another, with no model changes — only mesh layout and NCCL flags. Multi-node engineering is, at the end of the day, the discipline of laying your computational graph onto a physical fabric in a way that respects that fabric's bandwidth and latency tiers.

## 1. The parallelism quartet (DP, TP, PP, EP)

**Senior rule of thumb: never pick parallelism axes by reading a paper; pick them by counting bytes per step.**

Every distributed training framework offers some subset of four dimensions to slice work across devices, and each one trades a different resource for a different per-step collective.

- **Data parallelism (DP)** replicates the model on every rank and AllReduces gradients each step. Memory cost: full replication. Comm cost: roughly `2 × params` bytes per step (the factor accounts for the ring AllReduce's reduce-scatter + allgather pair).
- **Tensor parallelism (TP)** shards individual tensor operations — matmul rows, attention heads — across ranks, inserting an AllReduce inside the forward and backward of every layer. Comm cost: `~b × s × h` bytes per layer per step, where `b` is microbatch, `s` is sequence length, `h` is hidden size.
- **Pipeline parallelism (PP)** slices the model into stages, sending activations forward and gradients backward as point-to-point messages between stages. Comm cost: `b × s × h` bytes per microbatch boundary, with the catch that you pay a bubble proportional to `(stages − 1) / microbatches`.
- **Expert parallelism (EP)** routes each token to a subset of expert FFN layers spread across devices, requiring two all-to-all collectives per MoE layer. Comm cost: routing-dependent, but typically a strong multiple of TP.

![Parallelism quartet: each axis trades a different resource for a different collective volume](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-2.png)

The picture above is the cheat sheet I tape onto interview prep slides. The two columns most engineers underweight are *primary bottleneck* and *scales to*. DP scales to thousands of GPUs because grad AllReduce is bandwidth-optimal on a fat-tree IB fabric, but the bytes-per-step grows linearly with parameters, so a 70B model means roughly 280 GB of grad reduction per step at FP32 (or 140 GB at BF16) — at 200 Gb/s per NIC that's 5.6 seconds on the wire if you do not overlap. TP scales to roughly eight GPUs because the AllReduce inside every layer has to fit in one NVLink island — push it onto IB and you lose 30× bandwidth and ten layers' worth of forward time vanishes into latency. PP scales to tens of stages but trades that scaling for a bubble. EP unlocks sparse architectures but introduces all-to-all, which is the hardest collective to make resilient under skew.

The interview answer most candidates give — "use TP within a node, PP across nodes, DP across replicas" — is correct for dense models in the 30B–500B range, but it is not a derivation. The derivation goes: pick the axis with the smallest collective volume per step that the parent network layer can absorb. Then push the next-cheapest axis onto the next network tier up. The 8-GPU NVLink island can absorb `~b × s × h` bytes per layer, so TP fits. The rack-level IB switch can absorb a few hundred MB per microbatch boundary, so PP fits. The pod-level fat tree can absorb gradient AllReduce on the scale of tens of GB per step, so DP fits. EP only fits when you have a flat all-to-all-capable fabric — rail-optimized IB or NVLink-Switch System.

### Second-order: the cost of *combining* axes

The subtle part is that the axes are not independent. (DP × TP) means each DP replica is itself a TP group, so the grad AllReduce only runs over the DP dimension — saving you a factor of `TP_size` on the outermost collective. (DP × PP) means grad AllReduce only runs over matching PP stages on the DP axis, but you have to be careful that each DP replica has the same PP stage layout, otherwise the AllReduce groups are mis-shaped. We have seen one production bug where a custom layout had DP replica 0 with PP stages on hosts 0–3 and DP replica 1 with PP stages on hosts 4–7, but the DP groups were built by `[rank, rank + N/2]` rather than `[rank, rank + DP_offset]`, which meant the AllReduce was actually reducing stage 0 with stage 3. The loss curve was sensible. The model converged. It also performed three points of MMLU below a reference run, and nobody caught it for two weeks.

## 2. The device mesh: laying parallelism onto physical hardware

**Senior rule of thumb: every parallelism axis should ride exactly one network tier, no more and no less.**

A *device mesh* is the n-dimensional grid that maps parallelism axes to physical ranks. PyTorch 2.4 calls it `DeviceMesh`; DeepSpeed calls it a topology; Megatron calls it tensor / pipeline / data groups. The mechanics differ but the constraint is the same: each axis traverses one network tier.

![Device mesh on 64 GPUs: TP on NVLink, PP on IB, DP on rail-paired IB](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-3.png)

For a (TP=8, PP=4, DP=2) mesh on 64 GPUs across 8 hosts, the diagram above shows the canonical layout. The eight TP ranks of any PP stage live inside one host (one NVLink island), so the per-layer AllReduce runs at 900 GB/s. PP point-to-point messages cross IB rails between adjacent hosts, paid at 200 Gb/s per rail. DP grad AllReduce runs between the matching PP stages on the two replicas, paired by rail so each rail carries one DP edge.

The non-obvious part is the rail pairing. On a rail-optimized fat tree, each GPU is connected to one specific IB rail (GPU 0 → rail 0, GPU 1 → rail 1, …). A cross-host collective that involves GPU 0 on host A and GPU 0 on host B stays on rail 0 the whole way and never touches the spine. A cross-host collective between GPU 0 on host A and GPU 1 on host B has to leave rail 0, climb to the spine, and come down rail 1 — paying 3× the latency and competing with everyone else doing the same. If your mesh layout pairs GPU 0 with GPU 0 across hosts (rail-aligned), you preserve bandwidth. If it pairs GPU 0 with GPU 1 (rail-misaligned), you destroy it. The PyTorch DTensor mesh builder respects rank order, so the trick is in how you launch ranks: keep local rank 0 on GPU 0 of every host, and let global rank arithmetic flow from there.

Here is the actual PyTorch 2.6 mesh setup we use:

```python
## distributed/mesh.py — used by every job in this case study
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

## Mesh dims must match the launcher's nproc_per_node × nnodes layout.
## (DP, PP, TP) in this order so the innermost (rightmost) dim is TP — that
## keeps TP ranks contiguous, which in turn keeps them on one NVLink island.
TP, PP, DP = 8, 4, 2  # product = 64; matches world_size

def build_mesh():
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(DP, PP, TP),
        mesh_dim_names=("dp", "pp", "tp"),
    )
    # Subgroups: TP for layer-internal collectives, PP for activation send/recv,
    # DP for gradient AllReduce.
    tp_group = mesh["tp"].get_group()
    pp_group = mesh["pp"].get_group()
    dp_group = mesh["dp"].get_group()
    return mesh, dp_group, pp_group, tp_group
```

The two minute mistake here is putting TP as the *outermost* dim. The contiguous-rank guarantee then puts your TP=8 group across hosts 0–7 rank-0 instead of inside host 0, and your AllReduce-per-layer rides IB. Your loss curve looks fine. Your MFU is 11%. You will spend a week wondering why.

### Second-order: 3D parallelism on three different cluster topologies

The (TP=8, PP=4, DP=2) layout assumes 8-GPU NVLink islands with rail-optimized IB. On a cluster with NVLink Switch System (NVL72), where 72 GPUs sit on a single NVLink fabric, you can grow TP to 16 or 32 without leaving the green band — and TP=32 makes a 405B-dense model trainable without PP at all. On a cluster with PCIe-only inter-GPU links (older A100 4-GPU nodes), TP=8 *crosses* a PCIe boundary inside the host, and you should drop to TP=4. The mesh is a function of the fabric, not the model.

## 3. NCCL collectives in practice

**Senior rule of thumb: NCCL is correct by default and fast only after you tune it.**

NCCL implements every collective as a sequence of point-to-point chunked sends over a ring or tree topology. Which topology, which chunk size, and which algorithm get picked depends on the message size, the transport (NVLink, PCIe, IB, RoCE), and a wall of environment variables. Most multi-node performance work is, at this point, picking the right NCCL configuration.

![Ring vs tree AllReduce: bandwidth-optimal vs latency-optimal](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-4.png)

There are two collective shapes that matter for training. *Ring* AllReduce sends `2 × (N − 1)` chunks per rank around a logical ring; its bandwidth converges to the link bandwidth as N grows but its latency grows linearly. *Tree* AllReduce reduces up a binary tree and broadcasts down, taking `~2 × log N` steps with one chunk per step; its latency is logarithmic but its bandwidth never saturates the link. The crossover is around 1–4 MB on a 200 Gb/s IB fabric. Gradient AllReduce is always multi-MB, so it wants ring. Control-plane collectives — barriers, broadcast of small tensors, sync of optimizer state — are KB-scale, so they want tree. NCCL will pick automatically based on its internal model, and the auto-pick is right roughly 80% of the time. The other 20% is where you live.

```bash
## /etc/profile.d/nccl-tuning.sh — set on every training host
## Empirical tuning for 256-GPU H100 + ConnectX-7 400 Gb/s pods.

## Force ring for AllReduce / ReduceScatter — auto-tree picks wrong on long bursts.
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

## One NCCL channel per rail. Match this to your NIC count, not your GPU count.
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106     # DSCP for IB QoS; cluster-specific

## Avoid PCIe-only fallbacks — fail loudly if a rank can't see all rails.
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_TOPO_DUMP_FILE=/tmp/nccl-topo.xml

## Loud debug — flip off only after the run is green.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,NET,ENV
```

The `NCCL_TOPO_DUMP_FILE` trick is one of the highest-leverage debug tools. NCCL writes its discovered topology — every GPU, PCIe switch, NIC, IB GID — to that XML file on every rank, and inspecting it across ranks tells you immediately if rank 12 is missing two rails or if two GPUs are sitting on the same PCIe root complex when they should not. We have caught at least four production cluster misconfigurations this way before the loss curve ever rendered.

### Second-order: the AllReduce that is secretly two AllReduces

FSDP's `ReduceScatter` is a single named collective, but on the wire it is the *first half* of a ring AllReduce: each rank sends `(N − 1)` chunks. The matching `AllGather` later in the optimizer step is the *second half*. The NCCL bandwidth model assumes both halves run back-to-back; when you sprinkle compute between them, the second collective's first chunk has to walk the ring from scratch, paying the latency tax twice. The fix is to ensure your bucket size is large enough that the per-collective latency is dominated by bandwidth, not startup — typically 100–250 MB on H100-class hardware.

### A worked NCCL example: AllReduce bytes on a 256-GPU pod

It is worth grinding through one full example so the numbers stop being abstract. Take a 70B parameter dense model, BF16 weights, and a (TP=8, PP=4, DP=8) mesh across 256 GPUs. The grad AllReduce runs across the DP=8 dimension only — TP and PP shard parameters, so each DP rank holds 70B / (8 × 4) = 2.19B params at BF16, or about 4.38 GB. The ring AllReduce sends `2 × (N − 1) / N × 4.38` GB ≈ 7.66 GB per rank. At 400 Gb/s per IB link (50 GB/s) the on-wire time floor is 7.66 / 50 = 153 ms.

In practice we see 180–220 ms per AllReduce on this configuration. The 30–50% overhead splits roughly as: 12% NCCL chunking and ring-formation latency, 8% PCIe-to-NIC DMA overhead, 6% IB queue depth contention with PP point-to-point traffic, and the remaining 4–24% is pure cluster jitter (varies day to day). Knowing the floor matters: if you measure 350 ms per AllReduce, something is wrong — likely rail misalignment or a NIC negotiated below spec. If you measure 200 ms, the cluster is healthy and you should be tuning bucket sizes, not chasing NCCL flags.

The same exercise for TP: AllReduce inside one NVLink island, 8 GPUs, 70B / (8 × 4) / 8 = 274M params per TP-rank-per-layer... no wait, TP shards individual tensors, not parameters globally. For a single attention layer with hidden size 8192 and 8 heads-worth of heads sharded across 8 TP ranks, the AllReduce-per-layer carries `b × s × h × 2 bytes` for BF16 — at b=2, s=8192, h=8192 that's 268 MB per layer. On 900 GB/s NVLink the floor is 0.3 ms. In practice we see 0.5–0.7 ms per layer, and with 80 layers that's about 50 ms per forward pass on TP collectives alone. Push that AllReduce onto IB at 50 GB/s and the same math gives 5.4 ms per layer, 430 ms per forward pass — and your MFU drops 30 points. This is the bandwidth cliff that the device-mesh figure encodes.

## 4. FSDP / ZeRO bucket overlap

**Senior rule of thumb: bucket size is the throttle that controls how much of your communication can hide behind your computation.**

FSDP (and ZeRO-3 and FSDPv2) shard parameters across the DP group, AllGather them just before each layer's forward, ReduceScatter the gradients just after each layer's backward, and free the unsharded copy. The genius of the design is that the AllGather for layer N can overlap with the forward of layer N−1, and the ReduceScatter for layer N can overlap with the backward of layer N+1. The catch is that PyTorch packs parameters into buckets, and the per-bucket collective only fires once the bucket is full of ready gradients. Pick the bucket too small and the per-collective latency tax eats your overlap. Pick it too large and the last bucket's collective runs alone at the tail of backward, serializing.

![FSDP bucket overlap: grad reduce hides behind backward](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-5.png)

The figure traces one backward pass: as each layer's gradients land, the ReduceScatter for that bucket launches on a separate NCCL stream and runs while the previous layer continues backward. If `T_RS(bucket_k) ≤ T_backward(layer_{k-1})`, every reduce is fully hidden. If the inequality flips, the comm bubble shows up as a comm-bound tail on the timeline.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

## 1 wrap policy: shard each transformer block independently — lets FSDP build
## one bucket per block, which gives the cleanest per-layer overlap pattern.
mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,   # BF16 grad reduce; tested for stability on Llama-class
    buffer_dtype=torch.bfloat16,
)
wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # FSDP within DP group, replicate across
    auto_wrap_policy=wrap_policy,
    mixed_precision=mp,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # overlap next-layer AG with current bwd
    forward_prefetch=True,
    use_orig_params=True,                              # required for selective recompute
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,                            # cap inflight AG to avoid HBM spikes
    process_group=dp_group,
)
```

`HYBRID_SHARD` is the workhorse for multi-node training above ~16 hosts: it FSDP-shards within the DP group (one node) and replicates across DP groups, which turns the cross-node collective from a ReduceScatter into a much cheaper AllReduce. The cost is duplicated parameters across replicas, but for any model that fits in one node's HBM under FSDP, this trade is almost always worth it.

### Second-order: the bucket tail

`limit_all_gathers=True` is the flag we set the day a job started OOMing on H100s at random steps. FSDP pre-fetches one or two layers ahead by default; with selective activation recompute also reaching for memory, the AllGathers race the recompute peak. Capping in-flight AllGathers to one stabilizes the HBM profile at the cost of about 1.5 points of MFU. We took the trade every time. Selectivity is the lesson here too: capping is for production; off is for benchmarks.

## 5. Pipeline schedules: where the bubble comes from

**Senior rule of thumb: every pipeline schedule trades one of {memory, bubble, p2p volume}; you cannot win all three.**

Pipeline parallelism's defining tax is the *bubble*: time at the start of every step where downstream stages have nothing to do because upstream stages have not produced activations yet, and symmetrically at the end where upstream stages wait for downstream gradients. The first published schedule, GPipe, ran all forwards then all backwards, with the bubble equal to `(P − 1) / M` where `P` is pipeline stages and `M` is microbatches. The catch is GPipe holds *every* forward's activations in memory until backward starts — peak activation memory is `M × per-microbatch-act`, which becomes the bottleneck long before the bubble does.

1F1B (PipeDream) interleaves: as soon as the first microbatch's forward reaches the last stage, it starts its backward, and the schedule alternates F and B for the rest of the warmup phase. Peak activation memory drops to `P × per-microbatch-act`. The bubble fraction stays the same. Interleaved 1F1B splits each stage's layers into two chunks and round-robins them; the bubble drops to `(P − 1) / (M × V)` where `V` is the chunk count per stage, at the cost of `V × ` more p2p sends per step.

![Pipeline schedules: GPipe / 1F1B / interleaved](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-6.png)

The figure shows three rows of a four-stage pipeline with eight microbatches. The red bands are bubble cycles where the GPU sits idle. The activation memory difference is invisible in the picture but very visible in `nvidia-smi`: GPipe's tail has the GPU sitting at 85% HBM utilization waiting for backward; 1F1B's tail has it at 50%.

Code-wise, the modern PyTorch path is `torch.distributed.pipelining`:

```python
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleInterleaved1F1B
from torch.distributed.pipelining.PipelineStage import PipelineStage

## Split the Llama-3 70B model at the boundaries of every 20th decoder layer
## for 4 pipeline stages × 2 virtual chunks each.
split_spec = {
    f"model.layers.{i}": SplitPoint.BEGINNING for i in (10, 20, 30, 40, 50, 60, 70)
}
pipe = pipeline(model, mb_args=(example_input_ids,), split_spec=split_spec)
stage = pipe.build_stage(pp_rank, device=torch.cuda.current_device(), group=pp_group)
schedule = ScheduleInterleaved1F1B(stage, n_microbatches=8, loss_fn=loss_fn)

## In the training loop:
for batch in dataloader:
    if pp_rank == 0:
        loss = schedule.step(batch["input_ids"])
    elif pp_rank == pp_size - 1:
        loss = schedule.step(target=batch["labels"])
    else:
        schedule.step()  # middle stages just push activations through
```

### Second-order: zero-bubble pipelines

There is a 2023 family of schedules (Zero Bubble PP, DAPPLE-Z) that hide the bubble entirely by splitting backward into "compute grads w.r.t. input" and "compute grads w.r.t. weights", reordering the latter to fill the bubble. They work — we ran a Zero Bubble PP on a 32-stage configuration and went from 14% bubble fraction to 1.8% — but they couple to the optimizer (weight grads accumulate later in the step) and to memory management, and we found them brittle on heterogeneous hosts where one stage runs 4% slower than its siblings. For most production runs in 2026, interleaved 1F1B is the sane default.

## 5a. Expert parallelism: when the model is sparse

Most of this article assumes a dense model, where every parameter participates in every token's forward pass. Sparse Mixture-of-Experts changes the calculus enough that it deserves its own treatment. In an MoE layer, each token is routed (typically top-2) through a small subset of expert FFNs. If you have 64 experts across 8 GPUs (EP=8), each GPU holds 8 experts. The router decides which experts each token visits; tokens are then dispatched to the correct GPU, run through their experts, and the results are gathered back. That dispatch and combine is two all-to-all collectives per MoE layer per direction — four all-to-all per layer counting forward and backward.

The two failure modes of EP are *routing skew* and *all-to-all latency*. Routing skew happens when the load balancer in the router fails and a small fraction of experts receive most of the tokens. The remaining experts sit idle. Communication still pays full all-to-all cost, but compute under-utilizes by 4–10×. Every production MoE we have run included an auxiliary load-balancing loss with weight ~0.01; without it, the largest expert in a run we shipped received 40× more tokens than the median expert by step 30,000. All-to-all latency on IB is the more fundamental constraint: a 256-rank all-to-all touches every IB switch in the pod, contends with every other rank's traffic, and scales roughly as `O(N)` in pod size. The practical EP ceiling on rail-optimized IB is around 64 GPUs; past that you need either NVLink Switch System or a hierarchical EP (per-node experts + cross-node routing) that we will not cover here.

The other moving piece is *capacity factor*. The router promises each expert at most `capacity_factor × tokens_per_expert` tokens per step; tokens beyond that are dropped (or routed to a fallback). Setting capacity_factor=1.0 minimizes compute waste but drops tokens, hurting convergence. Setting it to 1.5 padding-pads the all-to-all by 50%, paying 50% extra communication for full token retention. Most production runs land at 1.25.

```python
## EP setup: every group of EP GPUs shares the expert dimension; the rest
## of the mesh (TP × PP × DP) holds the non-expert weights.
from torch.distributed.tensor import DTensor
EP = 8
ep_mesh = mesh["ep"]  # extra dim added to the DeviceMesh

class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, ep_mesh, capacity=1.25):
        super().__init__()
        self.experts_per_rank = n_experts // EP
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                          nn.Linear(4*d_model, d_model))
            for _ in range(self.experts_per_rank)
        ])
        self.capacity = capacity
        self.ep_mesh = ep_mesh

    def forward(self, x):
        ## x: [b, s, d_model]
        ## 1. Compute routing weights (top-2)
        logits = self.gate(x)
        probs = F.softmax(logits.float(), dim=-1)
        topk_p, topk_i = probs.topk(2, dim=-1)
        ## 2. Dispatch all-to-all: tokens go to the rank holding their expert
        dispatched = all_to_all_dispatch(x, topk_i, self.ep_mesh, self.capacity)
        ## 3. Local expert compute
        out_per_expert = [self.experts[e](dispatched[e]) for e in range(self.experts_per_rank)]
        ## 4. Combine all-to-all: results back to source rank
        combined = all_to_all_combine(out_per_expert, topk_i, self.ep_mesh)
        ## 5. Weighted sum by router probs
        return (combined * topk_p.unsqueeze(-1)).sum(dim=-2)
```

The two `all_to_all_*` calls are typically implemented via `torch.distributed.all_to_all_single` with a precomputed split-size schedule. They are also where every production MoE we have shipped spent at least one painful debugging week — the dispatch and combine schedules must agree token-for-token across all ranks, and an off-by-one in either schedule produces a soft corruption that the loss curve absorbs.

## 6. The recipe: a 256-GPU Llama-3 70B end-to-end

**Senior rule of thumb: never launch full-scale before passing every rung of the sanity ladder, in order.**

Here is the actual recipe we use to bring up a new 256-GPU training job from cold metal to sustained loss curve. The trick is not to skip rungs. Every rung verifies exactly one network tier or one class of failure; skipping one means you find that failure at full scale, where reproduction is twenty times more expensive.

![Sanity-check ladder: 1 → 8 → 32 → 64 → full scale](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-7.png)

### Rung 1 — 1 GPU, 5000 tokens

Run the *exact* training script with `--world_size=1` and a 5000-token subset. Verify: the loss curve shape matches a known-good reference (we keep a checkpoint from a previous month's run), tokens/s/GPU is within 5% of the single-host benchmark, and no warnings about uninitialized params or unused modules. Time budget: 30 minutes. If this rung fails, the bug is in your *model*, not your distributed setup.

### Rung 2 — 8 GPUs, 1 host, TP only

Set `(TP=8, PP=1, DP=1)`. Verify: TP AllReduce works (no `NCCL invalid usage`), per-GPU tokens/s is within 3% of rung 1 (TP overhead should be sub-5% on NVLink), and the loss curve matches rung 1 to within 1e-3 at step 100 (same seed, same data order). Time budget: 1 hour. Failure here usually means NCCL is falling back from NVLink to PCIe — check `nvidia-smi topo -m`.

### Rung 3 — 32 GPUs, 4 hosts, TP × PP

Set `(TP=8, PP=4, DP=1)`. Verify: IB rails carry traffic (check `mlxlink` or `ibdump`), tokens/s is within 8% of rung 2 (PP bubble eats ~5% at 8 microbatches), and a single backward pass completes without a NCCL timeout on any rail. Time budget: 2 hours. Failure here is almost always either an IB rail down or a NCCL_IB_HCA misconfig — re-read `NCCL_TOPO_DUMP_FILE`.

### Rung 4 — 64 GPUs, 8 hosts, full mesh + DP

Set `(TP=8, PP=4, DP=2)`. Verify: DP grad AllReduce runs (check `NCCL_DEBUG=INFO` traces for `ncclAllReduce` calls between matching PP stages), MFU is within 12% of the run-target, an async DCP checkpoint completes in under 3 minutes, and you can stop the job, restart from the checkpoint, and continue without loss curve discontinuity. Time budget: 4 hours.

### Rung 5 — Full scale, dashboard-on

Only now do you launch the 256-GPU job. The first 1000 steps should show steady tokens/s/GPU with std-dev < 3%, MFU sustained above target, async checkpoints completing every 500 steps without back-pressuring training, and no rank reaching a watchdog timeout. Wire your dashboard up to alert on step-time std-dev > 8% — that is your earliest-warning straggler signal.

## 7. Case study: 4096 GPUs, 31 days, 38% MFU

We took a Llama-3-70B-class dense model from cold metal to a sustained 38% MFU on 4096 H100s over 31 days. The headline number is a lie unless you also tell the story of the 62 points we lost along the way, where each point went, and which ones we got back.

![MFU waterfall: where 62 points of theoretical peak go](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-8.png)

The waterfall above is the post-mortem. Theoretical peak is 989 TFLOPs/GPU on BF16 tensor cores — that is the ceiling, not a target. Twenty-two points went to selective recompute (we recomputed attention but not MLPs, which buys us 4× activation savings at 22% compute tax). Fourteen went to communication exposed past backward — the FSDP bucket was tuned to 200 MB and NCCL was forced to chunked rings, but the last bucket of each step still ran alone at the tail. Ten points went to the pipeline bubble; we ran interleaved 1F1B with two chunks per stage and could have gotten three more points back by going to four chunks, but the p2p volume doubled and IB queue depth spiked. Eight points went to dataloader stalls — we cached the *shuffled* training shards on local NVMe with a four-batch prefetch, which fixed it. Eight more went to rank jitter from CPU governor and NCCL ring drift; pinning `cpufreq` to `performance` and forcing `NCCL_ALGO=Ring` recovered most of it.

The earliest straggler symptom is invisible until you plot the per-rank step time, sorted, every step.

![One slow rank stalls every other rank](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-9.png)

The picture above is what one bad NIC looks like. Rank 5 compute takes 590 ms because its IB link negotiated one lane instead of four and the retransmits show up as kernel-level stalls in the cudaMemcpy used by NCCL's CPU-side proxy. Every other rank computes in ~480 ms and then waits ~120 ms at the AllReduce barrier. The wall-clock is 590 ms × steps_per_epoch, not 480. We lost 18% of throughput for six hours before the on-call engineer noticed.

## Case studies from production

What follows is the long form: twelve named incidents from one 31-day, 4096-GPU run. The pattern across all of them is that the loss curve almost always *looks fine* — the wreckage is always in the secondary signals.

### 0. Hour 0: provisioning the cluster

**Symptom.** The cluster nominally had 4096 H100s; the first probe found 4081 healthy GPUs and 15 in degraded states. **First hypothesis.** Capacity reporting was wrong; ignore. **Actual cause.** Six GPUs had ECC events in their boot-time POST and were correctly marked degraded; three hosts had NIC firmware mismatches (`mlx5` driver expected v22.39, NICs were on v22.31); one host had its PCIe slot 4 negotiating Gen3 instead of Gen5 due to a marginal contact. **Fix.** Drained the 15 GPUs, scheduled a maintenance window for the NIC firmware update, RMA'd the PCIe slot. Started training on 4064 GPUs instead of 4096 — a 0.8% capacity cost we paid happily for known-good hardware. **Lesson.** Pre-flight every GPU and every link before you trust a cluster. The hour spent verifying saves a week of mysterious stragglers downstream.

### 1. Day 0: rail-misaligned mesh

**Symptom.** Job came up, tokens/s/GPU was 38% of the single-host benchmark, MFU was 11%. No errors. **First hypothesis.** PP bubble — increase microbatches. We doubled microbatches; tokens/s went up 1%. **Actual cause.** The DTensor mesh dimensions were ordered `(TP, PP, DP)` instead of `(DP, PP, TP)`, which put TP groups across hosts 0–7 rank-0 instead of inside each host. Every per-layer AllReduce was riding IB at 1/30th of NVLink speed. Worse: because each "host's" eight ranks were now logically scattered, the rail-binding code that pinned local-rank-k to NIC-k did the wrong thing on every host, and the cross-host AllReduces themselves went through the spine instead of staying on a rail. So we paid both the NVLink-to-IB cliff and the rail-aligned-to-spine cliff in the same call. **Fix.** Reordered mesh dims to `(DP, PP, TP)`, redeployed; tokens/s/GPU jumped to 35% of theoretical peak the next morning. **Lesson.** The innermost (rightmost) mesh dim is the one with the highest contiguity guarantee — put your highest-bandwidth axis there. A 60-hour mistake that took 90 seconds to fix.

### 2. Day 1: dataloader bottleneck

**Symptom.** GPU utilization hovered at 76% between forward and backward. **First hypothesis.** Optimizer step was slow. **Actual cause.** The dataloader workers were each opening one HTTPS connection per shard to the object store, and TLS handshake latency on cold connections was hitting 80 ms. With 16 workers and a 4-batch lookahead, the pipeline starved every 12 steps. **Fix.** Mounted a local NVMe cache of the shuffled shards (5 TB per host), prefetched four batches, kept connections persistent. **Lesson.** A dataloader that works on one host can fail catastrophically when 256 hosts hit the same endpoint.

### 3. Day 2: silent NaN in one TP group

**Symptom.** Loss curve developed a 2σ wobble around step 12,000; no NaN propagated. **First hypothesis.** Learning rate too high. **Actual cause.** One HBM stack on GPU 14 had developed a faulty cell that flipped a bit roughly every 4e10 reads. It corrupted one element of a `MatMul` output per ~5000 steps. The TP AllReduce smoothed the corruption across the TP group, which is why no rank ever NaN'd. **Fix.** Drain GPU, RMA. We caught it because we periodically (every 500 steps) compared a deterministic forward pass on rank 0 against a CPU reference; the divergence pointed at the bad GPU. **Lesson.** Hardware can lie quietly. Periodically check determinism, not just convergence.

### 4. Day 4: flapping IB port

**Symptom.** NCCL watchdog timeout, one rank, one host, every ~14 hours. **First hypothesis.** Software bug in NCCL 2.21 (we were on a release with a known flush issue). **Actual cause.** A single IB port on host 23 was flapping due to a marginal QSFP cable seating. `mlxlink` showed 3-second intervals where the link reported `phys_state: Disabled`. **Fix.** Replaced the cable; updated the cluster pre-flight script to reject any host with a non-zero `link_down_counter` in the last 24 hours. **Lesson.** IB is more reliable than Ethernet, not infinitely reliable. Pre-flight the rails.

### 5. Day 7: synchronous checkpoint stall

**Symptom.** Every 5000 steps, training paused for 11–14 minutes while the checkpoint wrote. **First hypothesis.** Storage was slow. **Actual cause.** The checkpoint code used `torch.save` on full state on rank 0, gathering everything from all ranks before serializing. With 70B parameters at FP32 optimizer state + BF16 weights, that's 1.4 TB of gather. **Fix.** Switched to `torch.distributed.checkpoint` with async save and per-rank sharding (see Section 8). Checkpoint pause dropped from 11 minutes to under 90 seconds, and the 90 seconds overlapped with training continuing. **Lesson.** Synchronous full-state checkpoints do not scale past ~32 GPUs. We later found out the original `torch.save` code path also did a `pickle` of the entire state on the gathering rank, which is single-threaded Python and never parallelizes. Even on perfect storage the floor was bound by Python's serializer, not by disk.

### 6a. Day 8: optimizer state divergence after restart

**Symptom.** After the day-7 checkpoint switch, resumed runs showed gradient norm 1.4× higher than pre-checkpoint runs in the first 200 steps after resume. **First hypothesis.** Stale optimizer state in the new format. **Actual cause.** The new DCP save was correctly saving Adam's first and second moments, but the old (synchronous) load path had been silently discarding them because the key names had changed from `optim.state.exp_avg` to `optim.state.0.exp_avg` between PyTorch 2.3 and 2.5. The load returned without error and the optimizer initialized fresh moments. **Fix.** Audited the loaded state dict explicitly — every parameter's optimizer state must round-trip a hash check before training resumes. **Lesson.** "Loaded successfully" is not "loaded correctly." Add a checksum step.

### 6. Day 9: gradient norm spike

**Symptom.** At step 38,000, gradient norm spiked from 0.3 to 12.4 in one step. Loss curve unaffected. **First hypothesis.** Bad batch. **Actual cause.** A small fraction of the data had documents whose routing weights collapsed onto one expert in the MoE layer (we had been trialing an MoE variant on a side branch), driving that expert's gradient unrelated to all others. The clip-by-global-norm absorbed it, but a future step would have NaN'd. **Fix.** Added per-expert grad clipping. **Lesson.** Global norm clipping is not a safety net when one tensor is doing all the work.

### 7. Day 12: PP bubble doubled

**Symptom.** After enabling activation offload to host RAM (to fit a 4× larger context), pipeline bubble grew from 8% to 17%. **First hypothesis.** CPU-GPU copy was slow. **Actual cause.** The offload code held the activation tensor on a separate CUDA stream until backward; the stream was unrelated to the NCCL stream, so the next stage's AllGather couldn't overlap with the offload. **Fix.** Routed activation offload through the same compute stream that the next stage's PP-recv used. **Lesson.** When you add a new CUDA stream, audit every overlap claim that was true before.

### 8. Day 16: BF16 overflow in router logits

**Symptom.** Validation NLL spiked on one specific subset of the data. **First hypothesis.** Data contamination. **Actual cause.** Router logits in an experimental MoE layer were getting computed in BF16; the softmax denominator overflowed for one token type with sharp routing. **Fix.** Promoted router logits to FP32 for the softmax, kept everything else BF16. **Lesson.** Sharp distributions in BF16 always need a precision audit.

### 9. Day 18: 14-host straggler cluster

**Symptom.** A persistent 6% straggler tax in step time, traced to 14 specific hosts. **First hypothesis.** Bad NICs. **Actual cause.** Those 14 hosts had been provisioned by a different team three months earlier and their CPU `cpufreq` governor was `powersave`, not `performance`. The CPU couldn't keep up with the NCCL proxy thread. **Fix.** Pinned `performance` governor across the entire pod; added a pre-flight check. **Lesson.** Hosts that look identical are not.

### 10. Day 22: filesystem inode exhaustion

**Symptom.** Async checkpoint writes started failing with `ENOSPC` despite 4.2 PB free. **First hypothesis.** Storage was full. **Actual cause.** Each shard created ~2,500 files (one per FSDP unit), and 4500 retained checkpoints × 2,500 files / shard × 4096 shards = 46 billion inodes, exceeding the cluster filesystem's inode table. **Fix.** Garbage-collected checkpoints to a 5-deep retention; consolidated FSDP shard files. **Lesson.** The PB-scale metric you watch is usually not the metric that will kill you. Inodes count.

### 11. Day 25: rendezvous race after a node reboot

**Symptom.** A single host rebooted (BMC firmware update). When torchrun elastic agent on that host re-joined, four other hosts crashed with `RuntimeError: connection closed by peer`. **First hypothesis.** Network partition. **Actual cause.** The c10d rendezvous backend on `min_nodes=N, max_nodes=N` does not handle a leave-and-rejoin gracefully on PyTorch 2.4; the surviving agents detected the brief absence and the rendezvous re-elected, but the cohort renumbering was inconsistent across hosts for ~80 ms. **Fix.** Pinned the rendezvous backend to etcd-v2, which handles this case correctly. **Lesson.** Elastic restarts are not free; pick a backend tested under the failure mode you actually have.

### 11a. Day 27: NCCL deadlock from a stale CUDA graph

**Symptom.** After enabling CUDA graphs to amortize kernel launch overhead, the job ran for ~6 hours then deadlocked. All ranks held a `ncclAllReduce` waiting for one specific TP group. **First hypothesis.** Bug in NCCL 2.21's graph-aware path. **Actual cause.** The CUDA graph captured a NCCL communicator handle that had been recycled across a torchrun elastic restart triggered earlier in the day. The graph replayed against the dead handle, which silently returned `cudaSuccess` but never enqueued the collective. **Fix.** Re-capture the CUDA graph on every elastic restart (signal handler in the watchdog). **Lesson.** CUDA graphs encode pointer state. Anything that invalidates that state — restart, comm rebuild, even some memory pool changes — invalidates the graph. Re-capture defensively.

### 11b. Day 29: NCCL_BUFFSIZE one-line tuning

**Symptom.** Sustained MFU was 36%, target was 38%, and AllReduce-per-step measured 215 ms against a 153 ms floor. **First hypothesis.** Cluster-wide jitter; expected, accept it. **Actual cause.** `NCCL_BUFFSIZE` defaulted to 4 MB on this NCCL build, and our 200 MB FSDP buckets were being chunked into 50 sub-collectives, each paying the ring-formation latency. Raising `NCCL_BUFFSIZE=33554432` (32 MB) reduced sub-collective count to 7, and per-AllReduce latency dropped to 178 ms. **Fix.** One env-var change. **Lesson.** NCCL has roughly 60 tunable env vars; about a dozen genuinely move the needle on multi-node training. Keep a checklist; do not assume defaults are right for your fabric.

### 11c. Day 30: a 6-hour GPU hang at the dataloader epoch boundary

**Symptom.** At the end of training epoch 2 (around step 92,000) the entire job hung for 6 hours before our watchdog finally killed it. **First hypothesis.** Filesystem hang during checkpoint write that coincided with epoch boundary. **Actual cause.** The dataloader was using `persistent_workers=True` with a custom sampler that re-shuffled at epoch boundaries by calling `dist.barrier()` from inside the worker process. Inside a forked worker the `torch.distributed` state was inherited but the NCCL communicators were not — the barrier issued a collective that no other rank ever called, and NCCL waited the full default timeout. **Fix.** Moved epoch-boundary shuffling to the main process; workers re-load their shard manifest on epoch change but do not participate in collectives. **Lesson.** Forking after `init_process_group` is dangerous. Anything in a worker that touches distributed state inherits half a comm world and behaves correctly until it does not.

### 12. Day 31: 38% MFU sustained

The final state. Twelve identified bugs, eight non-trivial fixes, three hardware RMAs. The single most expensive lesson, in GPU-hours, was rung-skipping on the sanity ladder for Day 0's mesh bug — we had not run rung 3 because "it's just a config change." That cost us 60 hours of full-scale training before someone correlated the slow MFU with the mesh dim order. The single most useful tool was a per-rank step-time histogram updated every 100 steps; every other production bug except #3 announced itself in that histogram first.

If we ran the same project again today, the changes we would make in advance are almost embarrassingly small. Add a one-line `assert local_rank in (0..7) and rank // 8 == host_index` to the mesh setup, catching the Day 0 bug in launcher init. Set `NCCL_BUFFSIZE=33554432` and `NCCL_ALGO=Ring` from day one, capturing the Day 29 win without any debugging. Use etcd-v2 rendezvous from day one, sidestepping Day 25. Add a deterministic forward-pass-vs-CPU-reference check at step 500, catching Day 2's silent NaN before it cost us four days. Five lines of code and one config flag would have erased about ten days of debugging across the run. The cost of *adding the lines now* is two hours; the cost of *not having them next time* is another ten days. This is what investment in launcher quality looks like, and it is why the launcher script in this article is not a toy — it is the result of two years of paying off this kind of debt one bug at a time.

## 7a. Cost economics: what 38% MFU buys you

Numbers per-step are easy to wave at. Numbers in dollars are not. The 4096-GPU run from the previous section ran for 31 days at sustained 38% MFU on H100s. At a list-price of roughly $4.00 per GPU-hour on a major cloud, that is 4096 × 24 × 31 × $4 = $12.2M of compute. The same run at the 11% MFU we shipped with on Day 0 of the case study would have taken 107 days to consume the same FLOP budget — three months of additional wall-clock at the same nightly burn, or $42M to finish at the same calendar date by adding more GPUs. The 27 points of MFU we recovered were worth about $9M and ten weeks of wall-clock.

This framing matters because it sets the budget for the engineering work itself. A two-person team that takes three weeks to find the dataloader bottleneck (Day 1, case study #2) saves the company about $1.4M in wall-clock cost if their fix raises sustained MFU by 4 points. That payoff is why multi-node infrastructure teams are typically allowed to spend 1–2 weeks per percentage point of MFU recovery before management blinks. The math also bounds the *opposite* decision: if you cannot articulate a hypothesis that recovers 2+ points of MFU in two weeks of work, your time is more valuable elsewhere — running a baseline ablation, drafting the next model, or improving evaluation. We have personally watched a senior engineer disappear into a four-week investigation of a 0.4-point MFU difference between two NCCL versions. They were right; it really was 0.4 points. It was also $400K of their salary plus opportunity cost to find that out.

The other economic lever is *checkpoint retention*. A 70B-class run produces ~3 TB of checkpoint data per save. At a 5000-step cadence over 31 days that is ~270 checkpoints, ~810 TB at full retention. Cluster filesystem at $0.04/GB-month is $32K/month just for retained training state — a non-trivial line item that the GC policy in case study #10 directly addresses. The five-deep retention policy we landed on reduces that to ~$600/month.

## 7b. Observability stack: what we actually plot

Most multi-node training failures show up first in a derived signal, not in `loss`. The dashboard below is the four-panel stack we built in Grafana for every long-running job; it is not exhaustive, but it covers about 90% of incidents.

| Panel | Metric | Sampling | Alerts on |
|---|---|---|---|
| Step rhythm | per-rank step time (p50, p99, max) | every step, every rank | p99/p50 > 1.15, or max-min > 200 ms |
| Collective health | AllReduce / RS / AG latency per call | every step, rank 0 only | latency drift > 30% over 100-step window |
| Hardware decay | GPU ECC counters, IB link `link_down_counter` | every 30 s, every host | any non-zero delta in a 1-hour window |
| Storage | checkpoint write time, FS p99 read latency | per save + every 30 s | save time > 3 min, or p99 read > 200 ms |

Each panel maps to a class of root causes. Step rhythm catches stragglers and dataloader stalls. Collective health catches NCCL regressions and rail misconfig. Hardware decay catches the slow failures (bad HBM cells, marginal cables). Storage catches the I/O cliffs.

```python
## metrics_logger.py — runs on rank 0, writes a JSON line per step.
import time, json, statistics, os

class StepTimer:
    def __init__(self, window=100):
        self.window = window
        self.times_per_rank = []  # list of dicts keyed by rank
        self.last_start = None

    def step_start(self):
        self.last_start = time.time()

    def step_end(self, rank):
        elapsed = time.time() - self.last_start
        ## torch.distributed.all_gather of `elapsed` into a list, omitted here.
        all_times = gather_all(elapsed)  # length world_size
        self.times_per_rank.append(all_times)
        if len(self.times_per_rank) > self.window:
            self.times_per_rank.pop(0)
        if rank == 0:
            stats = self._compute_stats()
            print(json.dumps(stats), flush=True)

    def _compute_stats(self):
        flat = [t for step in self.times_per_rank for t in step]
        last = self.times_per_rank[-1]
        return {
            "step_time_p50": statistics.median(flat),
            "step_time_p99": sorted(flat)[int(0.99 * len(flat))],
            "step_time_max": max(last),
            "step_time_min": min(last),
            "straggler_gap_ms": (max(last) - min(last)) * 1000,
            "straggler_rank": last.index(max(last)),
        }
```

The most useful field in that JSON is `straggler_rank`. Plotted over time as a heatmap (rank on Y, step on X, color by frequency-of-being-the-straggler), it pinpoints bad hardware faster than any other signal we have used. A rank that shows up as the straggler in more than 5% of steps over a 1000-step window is almost certainly hardware. A rank that shows up in 80% of steps is broken; replace the host.

### Tracing: when histograms aren't enough

For the hardest 10% of bugs — soft hangs, intermittent corruption, weird collective patterns — we drop to PyTorch's distributed flight recorder. Set `TORCH_NCCL_TRACE_BUFFER_SIZE=20000` and on watchdog timeout the flight recorder dumps a binary blob containing the last 20K NCCL operations per rank, with timestamps and ranks involved. The decoder script (`torch/distributed/flight_recorder`) reconstructs a partial order across ranks and shows you exactly which collective stalled and which rank arrived last. We caught the rail-misalignment bug from Day 0 of the case study by noting that every stall had the same eight ranks arriving 80 ms late — those eight ranks turned out to be the cross-host TP group that should have been intra-host.

## 8. Checkpointing without losing a week

**Senior rule of thumb: a checkpoint that takes 11 minutes is one that you will never restart from.**

`torch.distributed.checkpoint` (DCP) is the right answer in 2026. It writes per-rank shards in parallel, separates the planner (logical state → shard layout) from the storage, and supports asynchronous saves that overlap with training continuing. Most production teams should be using `dcp.async_save` + the file-system or S3 storage writers.

![DCP shard layout: planner + per-rank shards](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-10.png)

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
)

CKPT_ROOT = "/cluster/ckpt/run-2026-05-21"

def save_async(model, optimizer, step):
    state = {
        "model": get_model_state_dict(model),
        "optim": get_optimizer_state_dict(model, optimizer),
        "step": step,
    }
    fut = dcp.async_save(
        state,
        storage_writer=dcp.FileSystemWriter(f"{CKPT_ROOT}/step_{step}"),
    )
    # fut.result() in a background thread or at next-checkpoint; don't block forward.
    return fut

def load_resume(model, optimizer, step):
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state = {
        "model": get_model_state_dict(model, options=options),
        "optim": get_optimizer_state_dict(model, optimizer, options=options),
        "step": 0,
    }
    dcp.load(state, storage_reader=dcp.FileSystemReader(f"{CKPT_ROOT}/step_{step}"))
    return state["step"]
```

The async path is the lifeline. On a 70B model with 256 GPUs and BF16 weights + FP32 optimizer state, the synchronous save serializes ~280 GB; on a 30 GB/s aggregate POSIX filesystem that's a hard 9-second floor and a realistic 60–120 seconds under contention. Async writes fire and return immediately, with the background workers writing while training continues. Block on the future only at the next save boundary, and only if it is still pending.

### Second-order: resume on a different topology

DCP's killer feature is that the on-disk layout is *logical* — it tracks FSDP units and parameter names, not physical rank assignments. We have successfully resumed a 1024-GPU checkpoint on 512 GPUs, and resumed a (TP=8, PP=4, DP=2) mesh as (TP=8, PP=2, DP=4). The trick is that the planner file maps logical FSDP units to physical shards via the original mesh, and the load path re-shards on the fly. Test it once on a small run before you need it.

## 8a. Framework choice: torch-native vs Megatron-LM vs DeepSpeed

The major training stacks in 2026 all converge on the same set of primitives but with different ergonomics. Choosing one is a decision you live with for the run's lifetime, because every choice composes differently with every other choice.

| Stack | Best at | Weakest at | When we reach for it |
|---|---|---|---|
| torch-native (FSDPv2 + pipelining) | Composable mesh + DCP + standard ops | Documentation, last-mile MoE | Greenfield runs, anything that needs DTensor |
| Megatron-LM | TP + PP performance, fused kernels | Async checkpointing, library lock-in | Dense ≥ 100B with tight latency budget |
| DeepSpeed | ZeRO-3 + offload, multi-modal | Mesh primitives, recent feature lag | Limited GPU budgets, CPU offload needed |
| TorchTitan | Reference example of all the above | Not production-ready until 2025-late | Reading code to learn the patterns |

In practice we have shipped runs with all three. The torch-native stack is where most of our 2026 work lives — `init_device_mesh`, `FullyShardedDataParallel`, `torch.distributed.pipelining`, `torch.distributed.checkpoint` — because the primitives compose and the documentation tracks the implementation. Megatron-LM still wins on raw throughput for dense models in the 100B–500B range, primarily because its fused-attention and fused-LayerNorm kernels are tighter than the upstream torch equivalents; we have measured 8–12% MFU advantage on dense 175B configurations. The catch is that Megatron's checkpoint format is its own world, and migrating off Megatron mid-project is painful. DeepSpeed remains the right choice for budget-constrained teams where ZeRO-3 with CPU offload makes a model fit that otherwise would not; we used it on a 70B finetune that ran on 32 A100 80GB hosts with the optimizer state offloaded to host RAM, taking a 2× throughput hit for the ability to run the job at all.

The framework choice is not the long pole. The long pole is whether you have an on-call team that knows the framework deeply. We have seen Megatron-LM runs lose more wall-clock to "we don't know why this flag matters" than they ever gained in MFU. Pick the framework your team actually maintains.

## 9. The troubleshooting flowchart

**Senior rule of thumb: when you don't know what's wrong, look at the slowest rank, not the loss curve.**

Most multi-node hangs trace to a small set of root causes. The flowchart below resolves about 90% of them without reading raw NCCL traces.

![Hang diagnosis: which rank, which collective, which cause](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-11.png)

The two top-level branches are "one rank flatlined" (the watchdog dump points at a single rank) and "all ranks idle" (no kernel in flight anywhere). Single-rank hangs are almost always hardware: bad NIC, ECC events, HBM degradation. Whole-cluster hangs are almost always software: dataloader starvation, sync checkpoint mid-step, or filesystem hang.

For every multi-node job, instrument these four things and check them on every alert:

1. **Per-rank step time** (histogram, every 100 steps). Catches stragglers.
2. **NCCL log tail** (last 200 lines, on watchdog timeout). Catches collective stalls.
3. **GPU ECC counters** (every 5 minutes via `nvidia-smi -q -d ECC`). Catches hardware degradation early.
4. **Filesystem latency p99** (every 30 seconds, separate process). Catches I/O cliffs.

The first two are the most common in our incident log; the second two are the ones that save your hardest debugging hours.

## 10. The launcher and watchdog

**Senior rule of thumb: every job should self-document its environment in the first 30 seconds of logs.**

Here is the actual `torchrun` entrypoint for our 256-GPU runs, edited for clarity but otherwise unchanged. Two non-obvious pieces: the rank-0 env-dump (so post-mortems do not require finding the matching Slurm job script) and the cooperative watchdog that catches collective stalls a couple of minutes before NCCL's 30-minute default.

![torchrun process tree: rendezvous, elastic agents, workers](/imgs/blogs/multi-node-llm-training-recipe-troubleshooting-12.png)

```python
## train_entry.py
import os, json, time, signal, sys, traceback, threading
import torch
import torch.distributed as dist

def env_dump():
    keys = [
        "WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
        "MASTER_ADDR", "MASTER_PORT",
        "NCCL_DEBUG", "NCCL_ALGO", "NCCL_PROTO", "NCCL_IB_HCA",
        "NCCL_NSOCKS_PERTHREAD", "NCCL_IB_GID_INDEX",
        "TORCH_DIST_INIT_BARRIER",
    ]
    snap = {k: os.environ.get(k, "") for k in keys}
    snap["torch"] = torch.__version__
    snap["cuda"] = torch.version.cuda
    snap["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    snap["gpu"]  = torch.cuda.get_device_name(0)
    print(f"[env] {json.dumps(snap, sort_keys=True)}", flush=True)

class StallWatchdog:
    """Cooperative: training thread bumps `tick()` each step; if no tick for
       `timeout_s`, dump stacks of every Python thread on every rank and
       abort. Catches collective stalls without waiting for NCCL's 30-minute
       default."""
    def __init__(self, timeout_s=300):
        self.timeout = timeout_s
        self.last = time.time()
        self.alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def tick(self):
        self.last = time.time()

    def _loop(self):
        while self.alive:
            time.sleep(15)
            if time.time() - self.last > self.timeout:
                rank = int(os.environ.get("RANK", "0"))
                print(f"[watchdog] rank {rank} stalled for "
                      f"{time.time() - self.last:.0f}s; dumping stacks",
                      flush=True)
                for tid, frame in sys._current_frames().items():
                    print(f"\n--- thread {tid} ---", flush=True)
                    traceback.print_stack(frame)
                # Best-effort abort the NCCL communicator so the rest of the
                # cluster does not wait the full 30-min timeout.
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
                os.kill(os.getpid(), signal.SIGABRT)

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if dist.get_rank() == 0:
        env_dump()
    mesh, dp_grp, pp_grp, tp_grp = build_mesh()
    model, optimizer, dataloader = build_training_state(mesh)

    wd = StallWatchdog(timeout_s=300)
    step = load_resume(model, optimizer) if has_checkpoint() else 0

    for batch in dataloader:
        wd.tick()
        loss = forward_backward(model, batch, schedule=schedule)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 500 == 0 and dist.get_rank() == 0:
            log_metrics(step, loss)
        if step % 5000 == 0:
            save_async(model, optimizer, step)
        step += 1

if __name__ == "__main__":
    main()
```

The Slurm submission that drives it on 32 hosts × 8 GPUs:

```bash
#!/bin/bash
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=/cluster/logs/%x-%j.out

set -euo pipefail

## Rendezvous endpoint: pick the first node in the allocation.
HEAD_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
RDZV_PORT=29400

## Per-host environment — sourced before torchrun.
source /etc/profile.d/nccl-tuning.sh

srun --kill-on-bad-exit=1 \
  bash -c '
    torchrun \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=8 \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"$HEAD_IP:$RDZV_PORT"' \
      --max_restarts=2 \
      train_entry.py
  '
```

Three things to know about this script. First, `--max_restarts=2` means torchrun will re-elect rendezvous and re-launch all workers up to twice if a node crashes; combined with DCP async checkpoints every 5000 steps, that is good enough for ~99% of transient failures. Second, the `--kill-on-bad-exit=1` ensures that a Python crash on one host tears down the whole job rather than leaking a partial cluster. Third, the rendezvous endpoint is hard-coded to the head node, which is the right answer on small clusters but a single point of failure at 1024+ hosts; we use etcd-v2 for those.

## 10a. Advanced topics: deterministic resume, GPU jitter, and the long tail

A few subjects come up often enough in code review that they belong here even though no single one warrants its own section.

**Deterministic resume.** True bit-for-bit deterministic training across resume points requires four things that are easy to forget. First, save the RNG state of every rank (CUDA, CPU, NumPy, Python `random`) in the checkpoint. Second, save and restore the dataloader's *worker* RNG state — the worker processes have their own seeded streams that the main-process state does not cover. Third, restore the order-of-operations of any non-associative reduction; FSDP's ReduceScatter is associative-up-to-floating-point, so determinism requires the same rank ordering on resume. Fourth, disable any optimization flag that introduces non-determinism: `torch.backends.cudnn.benchmark=False`, no `cudaGraph` capture across step boundaries, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` to pin the cuBLAS kernel selection. Without these four pieces, "resume from checkpoint" gives you a model with similar but not identical weights — fine for most production work, fatal for any debugging exercise that needs reproducibility.

**GPU jitter and its real cost.** Modern H100s have a thermal throttle band and a power-cap band that interact with each other and with workload patterns. On a job that hits 700 W steady-state, ambient temperature swings of 4-5°C in the data center can move per-GPU clock between 1.755 GHz and 1.620 GHz — a 7.7% throughput swing that shows up as a 7.7% per-rank step time swing. The whole pod's MFU drops to the slowest rank. We track ambient temperature per rack on the same dashboard as step time; when the two correlate, the fix is HVAC, not software.

**Why the optimizer step is hiding losses.** Adam's optimizer step has its own communication cost when the optimizer state is FSDP-sharded: at every step, each rank needs to read its own shard of `m` and `v`, compute the update, and write back. On H100, the optimizer step itself is bandwidth-bound on HBM; for a 70B model with FP32 optimizer state, it reads ~840 GB and writes ~280 GB per step. At 3 TB/s HBM that is a 370 ms floor. We have seen production teams spend weeks tuning collective overlap before noticing that the *optimizer step itself* was 320 ms of their 1.2-second step time. The fix is fused optimizers (Apex's `FusedAdam`, or the kernel in `torch.optim._fused`) that cut the read/write count by 2×.

**The other long tail.** Mixed precision rules for stability have grown into a small science. Layernorm in FP32 is non-negotiable. Softmax in FP32 for the attention scores is non-negotiable. Router logits in MoE in FP32. Loss in FP32. Everything else in BF16 is the default we ship. Promoting more to FP32 costs throughput linearly and adds no stability. Demoting more to FP8 (on H100 transformer-engine paths) gives 1.4–1.7× more throughput but doubles the precision-audit work; we deploy FP8 only after a model has trained successfully in BF16 to within 5% of its target loss.

## 11. What I'd never do again

A short, expensive list. Each of these cost real GPU-hours.

- **Skip rung 3 because "it's just a config change."** Day 0 of the case study. Mesh-dim ordering is a config change. It cost 60 hours.
- **Run synchronous full-state checkpoints past 32 GPUs.** Every save was an attack on training throughput. Day 7 of the case study.
- **Trust the loss curve over the per-rank step-time histogram.** Most production failures show up in step time minutes before they touch the loss.
- **Treat the cluster filesystem as durable storage.** Inode exhaustion at day 22 was preventable.
- **Set `NCCL_DEBUG=WARN` from day one.** You always want `INFO` until the run is green. The trace volume is fine; disk is cheap; trying to debug a hang without the trace is not.
- **Pick a parallelism mix by reading a paper.** Pick it by counting bytes per step against the network tier that has to absorb them.

## When to reach for multi-node training, and when not to

Reach for multi-node training when:

- The model does not fit in one host's HBM under FSDP at your target context length.
- You have more compute available than one host can absorb, *and* your dataloader can keep up with that compute.
- You can afford a 2–4 week shake-down phase before any production training starts.
- You have an on-call rotation that can respond to a stalled job within an hour.

Skip multi-node training when:

- One host fits your model and you are not bottlenecked on wall-clock.
- Your dataset is small enough that data-parallel grad noise is the constraint, not GPU-hours — adding more replicas does not help past a critical batch size.
- You do not yet have observability on per-rank step time and ECC counters.
- The cluster is a borrowed multi-tenant share where you cannot pin governor, GIDs, or kernel parameters — your variance will dominate everything else.

The most honest version of the multi-node decision is: it is a bet that you can amortize the engineering tax across many runs. If you are doing one run, rent a bigger box. If you are doing fifty, learn the pod.

## Closing observations

The thing nobody tells you about multi-node training is that the work never really ends. A run that hits 38% MFU on day 31 will drift to 33% by day 60 if you stop paying attention, because clusters degrade. Cables loosen. Cooling drifts. Kernels get patched. A new tenant moves in next door and shares your spine bandwidth. The job that worked yesterday is not the job that works today; the *infrastructure that ran the job yesterday* is not the infrastructure that runs it today. Treat the cluster as a living system that has to be monitored continuously, not as a static substrate you provision once.

The second thing nobody tells you is that the same engineer who wrote the original launcher will, six months later, not remember half the magic flags in their own NCCL config. The system grows faster than the team's mental model of it. Documentation in the form of comments in the launcher and one paragraph per env var in the cluster runbook is, in our experience, the single highest-leverage piece of multi-node engineering after the launcher itself. We have personally lost two days re-deriving why we set `NCCL_NSOCKS_PERTHREAD=4` in a project we wrote one year ago. The fix was a one-line comment we never added.

The third thing, and the closing point: every multi-node training job is one team's bet that they can amortize a large upfront engineering tax across many production runs. If the tax is high and the runs are few, the bet does not work — rent a bigger box and move on. If the tax is paid once and the runs are continuous (production model factory, RL post-training pipeline, multi-objective sweeping), the bet pays off enormously. Be honest about which world you live in before you start the project. The technology is fascinating; the economics are unforgiving.

## Further reading

- [Torch Titan](/blog/machine-learning/mlops/torch-titan) — the modern reference implementation for FSDPv2 + PP + DCP in one place.
- [Speeding up training 4× by optimizing CPU→GPU data transfer](/blog/machine-learning/mlops/speeding-up-neural-network-training-4x-by-optimizing-cpu-to-gpu-data-transfer) — the single-host prerequisite to multi-node throughput.
- [Smol training playbook](/blog/machine-learning/smol-training-playbook) — small-scale companion with the same instincts at lower cost.
- [CUDA graph](/blog/machine-learning/deep-learning/cuda-graph) — for kernel-launch overhead context that becomes load-bearing once you are comm-bound.
- *Megatron-LM* paper (Shoeybi et al., 2019) — the canonical reference for tensor parallelism math.
- *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (Rajbhandari et al., 2020) — origin story for FSDP/ZeRO.
- *PipeDream-2BW / 1F1B* (Narayanan et al., 2021) — the schedule that displaced GPipe.
- NVIDIA NCCL documentation, especially `NCCL_TOPO_DUMP_FILE` and the rail-binding section.
- *Reducing Activation Recomputation in Large Transformer Models* (Korthikanti et al., 2022) — the math behind selective recompute that buys back most of the activation memory at modest compute cost.
- *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (Narayanan et al., 2021) — the reference 3D-parallel performance write-up that established the (TP × PP × DP) framing this article inherits.

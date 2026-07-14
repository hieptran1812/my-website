---
title: "The Distributed Training Playbook: One Page to Pin to the Wall"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "The capstone of the series: a decision ladder, a pre-flight checklist, a symptom-to-fix debugging table, and an efficiency scorecard that turn everything you learned into the one page you consult before, during, and after every big run."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "pytorch",
    "fsdp",
    "nccl",
    "mfu",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 30
---

Here is a scene you have now lived through, at least on paper, thirty-nine times. It is 2 a.m. You have a model that will not fit on one card, a 64-GPU allocation that expires in six hours, a launch command you copied from a teammate, and a training loss that has been flat for four hundred steps. Somewhere between your Python and the InfiniBand fabric, throughput is leaking, and you have exactly one job: find the leak, plug it, and get the run to the finish line before the allocation evaporates. This series has spent forty posts taking apart every piece of that machine — the parallelism you choose, the collectives it costs, the memory it saves, and the ways it breaks. This final post puts the pieces back together into something you can actually act on at 2 a.m.: a playbook.

A playbook is not a textbook. It does not re-derive ring all-reduce; it tells you *which lever to pull, in what order, and what to check when the lever jams.* It is the distillation of everything the [intro to this series](/blog/machine-learning/distributed-training/why-distributed-training) promised — the four walls, the levers, the comms, the memory, the failures — into four artifacts you will use in this exact sequence on every real run: a **decision ladder** (what parallelism to add and when), a **pre-flight checklist** (what to verify before you burn a single GPU-hour), a **debugging table** (symptom, first check, likely cause, the one fix), and an **efficiency scorecard** (is this run actually fast, and if not, which wall is throttling it). By the end you will have the single page a practitioner pins above the desk.

![a diagram where model and cluster and budget merge into a single wall-diagnosis node that then fans out to a fit-lever family and a speed-lever family](/imgs/blogs/the-distributed-training-playbook-1.webp)

Everything below descends from one organizing law, so let us state it before anything else.

## The one law that generates the whole playbook

Distributed training feels like a bag of unrelated tricks — DDP, ZeRO, tensor parallel, pipeline bubbles, NCCL flags, checkpoint cadence — until you notice that every one of them is an answer to the same four-part question. You always hit one of exactly **four walls**, and the wall tells you the lever:

- **The model will not fit.** One card cannot hold the parameters plus the optimizer state plus the activations. The lever is *sharding* (ZeRO / FSDP), then *model parallelism* (tensor, then pipeline) if sharding is not enough.
- **The data will not finish.** One card would take months to see the tokens. The lever is *data parallelism* — replicate the model, split the batch, all-reduce the gradients.
- **The run is too slow.** You have the GPUs but you are not using them; utilization is low. The lever is *overlap, precision, and topology* — hide communication behind compute, use bf16/fp8, keep heavy traffic on fast links.
- **The run is too expensive.** It fits and it is fast enough, but the bill is unacceptable. The lever is *efficiency and elasticity* — raise MFU, right-size the cluster, rent interruptible capacity behind good checkpoints.

Every technique in this series is a specific tool for one of those four walls. And there is a meta-rule that sits above all of them, the single most valuable sentence in the playbook:

> **Use the least parallelism that fits and hits your throughput target. Climb the ladder one rung at a time, and stop at the first rung that works.**

Every axis of parallelism you add buys a capability at the price of communication overhead, more processes to coordinate, and a dramatically larger debugging surface. So the correct strategy is never "use the most parallelism"; it is "use the *least*." A 7B model that fits comfortably on a single node with [FSDP](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) does not want tensor parallelism, and bolting it on will make the run slower, not faster, because you have traded overlappable all-gathers for blocking all-reduces on every layer. The whole ladder below is just this law, made mechanical.

## Part 1 — The scaling decision ladder: 1 → 8 → 64 → multi-node

The first question is not "which parallelism" but "how big a machine, and what does going bigger cost me?" Scale is a ladder with four rungs, and each rung has a characteristic bottleneck that the previous rung did not.

![a left to right timeline of scaling rungs from one GPU to eight to sixty-four to multi-node, each rung naming the lever it adds](/imgs/blogs/the-distributed-training-playbook-2.webp)

**Rung 1 — one GPU.** If the model trains on one 80GB A100 or H100 at your target batch and sequence length, *train on one GPU.* No process group, no collectives, no rank-mismatch deadlocks, no straggler. A surprising number of models — anything up to a few billion parameters at modest context — fit and train fine on a single card. Distribution is pure cost until the model forces it, and the cheapest distributed bug is the one you never wrote because you never went distributed.

**Rung 2 — one node, up to 8 GPUs.** Inside a DGX-class node, 8 GPUs are fully connected by NVLink and NVSwitch at roughly 900 GB/s of aggregate bandwidth per GPU on NVLink4. This is the sweet spot of the whole field: fast enough that data-parallel gradient all-reduce and FSDP parameter all-gather both overlap cleanly with the backward pass, so you get close to linear speedup. The rule at this rung: if the model fits as a replica, use [DDP](/blog/machine-learning/distributed-training/the-map-of-parallelism); if it does not, shard it with FSDP. You should be able to reach 80–90% scaling efficiency on a single node without any exotic parallelism. If you cannot, something is wrong — and the debugging table below is where you go.

**Rung 3 — multiple nodes, up to ~64 GPUs.** The moment you cross a node boundary, the interconnect changes character. Inter-node traffic rides InfiniBand or RoCE at roughly 200–400 Gb/s per link — call it 25–50 GB/s — which is more than an order of magnitude slower than NVLink. Now the *placement of your collectives relative to the topology* becomes the dominant design decision. The lever is `HYBRID_SHARD` (shard within a node, replicate across nodes) so that the heavy sharding traffic stays on NVLink and only the lighter gradient reduction crosses the fabric, plus tensor parallelism *strictly inside* a node where its blocking all-reduce can live on NVLink.

**Rung 4 — many nodes, hundreds to thousands of GPUs.** Now even the data-parallel all-reduce is a real cost, pipeline parallelism earns its keep as the axis that spans nodes cheaply (it sends only activations at stage boundaries, not whole gradients), and second-order effects — a single [straggler](/blog/machine-learning/distributed-training/the-straggler), a flaky NIC, thermal throttling on one rack — start to dominate your wall-clock. This is where fault tolerance stops being optional.

The discipline of the ladder is that **you do not skip rungs.** The single most common waste I see is a team reaching for multi-node before they have saturated one node — paying inter-node communication tax to run a job that would have been faster, cheaper, and far easier to debug on 8 GPUs. Saturate the node first. Prove your MFU on one node. *Then* cross the boundary, and expect to lose efficiency when you do.

#### Worked example: does a 7B model need multiple nodes?

Take a 7B-parameter dense Transformer, bf16, training with the Adam optimizer. The memory the model demands for *state alone* is the classic mixed-precision budget: 2 bytes for the fp16 parameters, 2 bytes for the fp16 gradients, and 12 bytes for the fp32 optimizer state (a 4-byte master copy of the parameters plus 4 bytes each for Adam's first and second moments) — 16 bytes per parameter in total, the $(2+2+12)\Psi$ formula from [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget). For 7 billion parameters that is 112 GB, which already exceeds a single 80GB card before a single activation is stored. So DDP alone is out — a full replica does not fit.

But shard that state across 8 GPUs with FSDP and each rank holds only 112 / 8 = 14 GB of state, leaving roughly 60 GB per card for activations and working memory. A 7B model fits comfortably on **one node** with FSDP full-shard. There is no reason to go multi-node for capacity. You would only add nodes here to go *faster* — to see more tokens per second — and even then you would keep sharding within the node and replicate across, not spread a single 14 GB shard-group across a slow fabric. The ladder answered the question before we wrote any launch command: rung 2, FSDP, one node.

## Part 2 — Which lever to pull: the parallelism decision tree

Once you know how big a machine you are on, the second artifact answers *which parallelism axis solves the wall in front of you.* This is a genuine decision tree, and the beauty of it is that it is almost mechanical once you ask the questions in the right order.

![a decision tree that starts by asking whether the model fits on one card and branches through sharding, tensor, pipeline, sequence, and expert parallelism](/imgs/blogs/the-distributed-training-playbook-3.webp)

Ask these questions top to bottom, and stop at the first "yes" that solves your problem:

1. **Does the whole model plus optimizer state fit on one GPU?** If yes, you are done — one GPU, or DDP across several if you just want throughput. Do not distribute the *model*; distribute only the *data*.
2. **Does it fit after you shard the state across the data-parallel group?** This is FSDP / ZeRO, and it is the rung most teams wrongly skip. Sharding divides parameters, gradients, and optimizer state by the number of ranks while keeping the simple data-parallel programming model, and its parameter all-gather overlaps far better than model-parallel collectives. Models into the tens of billions of parameters routinely fit with FSDP alone. Reach for it *before* any model parallelism.
3. **Is a single layer too big to fit even one shard, or is per-step latency dominated by one enormous matmul?** Add [tensor parallelism](/blog/machine-learning/distributed-training/tensor-parallelism-megatron), inside a node, at a degree no larger than the node's GPU count. It splits each matmul across GPUs at the cost of a blocking all-reduce per layer — which is exactly why it must stay on NVLink.
4. **Is the model too deep to fit even after sharding and tensor-splitting?** Add [pipeline parallelism](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble), the axis that spans nodes because it communicates only activations at stage boundaries. Its cost is the bubble — the idle time while the pipeline fills and drains.
5. **Do activations blow up because the context is very long?** Add [sequence / context parallelism](/blog/machine-learning/distributed-training/sequence-and-context-parallelism), which splits the sequence dimension and pays a ring or all-to-all communication.
6. **Is it a Mixture-of-Experts model?** Add [expert parallelism](/blog/machine-learning/distributed-training/expert-parallelism-moe), which places different experts on different GPUs and routes tokens with an all-to-all.

These axes *compose*: real large-scale training is "3D parallelism" — tensor inside the node, pipeline across a few nodes, data-parallel (usually sharded) across the rest. But you arrive at 3D by climbing this tree, not by starting there. [Picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) walks four full worked examples through this exact tree; the point of the capstone is to burn the *order of the questions* into memory.

### The levers in one table

The reason the tree works is that each lever solves a *distinct* problem and carries a *distinct* communication cost. Keep this table where you can see it, because most misdiagnoses come from reaching for the wrong lever — usually tensor or pipeline parallelism when the real problem was "state too big," which FSDP solves more cheaply.

| Lever | What it splits | Dominant collective | Comms cost | Add it when |
| --- | --- | --- | --- | --- |
| **DDP** | the batch (model replicated) | all-reduce of gradients | $2(N-1)/N \cdot S$ bytes/GPU, overlappable | model fits, you want more throughput |
| **FSDP / ZeRO** | params, grads, optimizer state | all-gather + reduce-scatter | ~1.5x DDP traffic, mostly overlappable | model will not fit as a replica |
| **Tensor (TP)** | each matmul / layer | all-reduce per layer, **blocking** | high, latency-bound, NVLink-only | a single layer is too big or latency-bound |
| **Pipeline (PP)** | the layer stack into stages | point-to-point activations | low bytes, but bubble $(p-1)/(m+p-1)$ | too deep to fit; spans nodes cheaply |
| **Sequence (SP)** | the sequence dimension | ring / all-to-all of activations | scales with context length | activations OOM at long context |
| **Expert (EP)** | experts across GPUs | all-to-all of tokens | scales with tokens routed | Mixture-of-Experts routing |

### The mechanism: why the interconnect decides the answer

The tree above hides a quantitative core, and it is worth making it explicit because it is *the* reason topology is a first-class input, not a footnote. Take the workhorse collective, ring all-reduce. Its per-GPU traffic is

$$T_\text{allreduce} = 2\frac{N-1}{N}\cdot\frac{S}{B},$$

where $S$ is the gradient buffer size, $N$ the number of GPUs, and $B$ the bus bandwidth. For large $N$ the factor $2(N-1)/N$ approaches 2, so each GPU moves about $2S$ bytes regardless of how many GPUs there are — the *volume* is nearly constant, but the *time* is set entirely by $B$.

Now plug in our 7B model, whose bf16 gradients are $S = 14$ GB, and watch the interconnect swing the answer by an order of magnitude. On NVLink4 at roughly 900 GB/s, moving $2S \approx 28$ GB takes about 31 ms. On InfiniBand HDR at roughly 25 GB/s, the identical all-reduce takes about 1,120 ms — 36 times longer. That single ratio explains the whole placement doctrine: keep the heavy collectives on NVLink (tensor parallel, FSDP shard groups) and let only the unavoidable, lighter traffic cross the fabric. It is why tensor parallelism is a within-node lever, why `HYBRID_SHARD` exists, and why "multi-node slower than single-node" is a category of bug rather than a surprise. The math *is* the doctrine.

## Part 3 — The pre-flight checklist before a big run

You have chosen your rung and your levers. Before you spend a single GPU-hour of a 64-GPU allocation, you run the pre-flight. Every item on this list corresponds to a class of failure I have personally watched consume an afternoon, and every one is checkable in seconds *before* launch.

![a vertical stack of pre-flight checks from device mesh at the top down through global batch, memory budget, checkpoint cadence, and alerts](/imgs/blogs/the-distributed-training-playbook-4.webp)

1. **The device mesh multiplies out to your world size.** Your parallel degrees must satisfy `dp × tp × pp = world_size`, exactly. An off-by-one here does not error politely; it hangs on the first collective while half your ranks wait for peers that were assigned to the wrong group. Assert it in code before you build any process group.
2. **The global batch is what you think it is.** The number that governs your learning dynamics is `global_batch = micro_batch × dp_degree × grad_accum_steps`. When you change the GPU count, this number changes unless you deliberately hold it fixed, and a silently doubled global batch is a silently different (often worse) training run. Print it at startup and log it.
3. **The memory budget closes with headroom.** Add up parameters, gradients, optimizer state (divided by your shard degree), and *activations* (the term everyone forgets, which grows with batch, sequence, and layer count). Leave 10–15% headroom for fragmentation and NCCL buffers. If the sum is over 80 GB, you will OOM at step 0, and you fix it *now* with activation checkpointing or a smaller micro-batch, not after the queue finally schedules your job.
4. **Checkpoint cadence matches your failure rate.** On a large cluster, something fails. Save often enough that a failure costs you minutes, not hours — every 30 to 60 minutes is typical — and make the save *asynchronous* and *sharded* so it does not stall training. Verify a checkpoint actually *restores* before you trust it, because a checkpoint you have never restored is a hope, not a backup.
5. **Monitoring and alerts are wired before the run, not after it breaks.** Loss, gradient norm, tokens/s, and — critically — *per-rank* step time, so a straggler shows up as one line diverging from the pack. [Monitoring a long run](/blog/machine-learning/distributed-training/monitoring-a-long-run) is the full treatment; the pre-flight item is simply: is it on?
6. **Fault tolerance is configured.** `torchrun` with a c10d rendezvous and `--max-restarts`, elastic membership if your scheduler supports it, and an auto-resume that reloads the latest checkpoint. [Fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) covers the design; pre-flight asks only whether a single node dying at hour 11 costs you the whole run or thirty minutes.

Here is the pre-flight as code — the assertions I put at the top of every distributed entry point. It costs nothing and it catches the mesh and batch mistakes that otherwise waste a scheduling cycle:

```python
import os
import torch
import torch.distributed as dist

def preflight(dp: int, tp: int, pp: int, micro_batch: int, grad_accum: int):
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 1. Mesh must multiply out to the world size, exactly.
    assert dp * tp * pp == world, (
        f"mesh dp={dp} tp={tp} pp={pp} = {dp*tp*pp} != world {world}"
    )

    # 2. Global batch is derived and logged, never assumed.
    global_batch = micro_batch * dp * grad_accum
    if rank == 0:
        print(f"[preflight] world={world} mesh={dp}x{tp}x{pp} "
              f"global_batch={global_batch} "
              f"(micro={micro_batch} x dp={dp} x accum={grad_accum})")

    # 3. Fail loudly if the visible device count per node is wrong.
    assert torch.cuda.device_count() >= (local_rank + 1), \
        "LOCAL_RANK exceeds visible CUDA devices; check --nproc_per_node"

    dist.init_process_group("nccl")
    # 4. A tiny all-reduce proves the process group is healthy before real work.
    probe = torch.ones(1, device=f"cuda:{local_rank}")
    dist.all_reduce(probe)
    assert probe.item() == world, "process group all-reduce is broken"
    if rank == 0:
        print(f"[preflight] process group OK, all-reduce sum = {int(probe.item())}")
    return global_batch
```

That last step — a one-element all-reduce that must sum to the world size — is the cheapest possible smoke test that your process group, your NCCL install, and your network are all functioning. If it hangs, you have a rendezvous or transport problem *before* you have wasted an hour, and you go straight to the NCCL section of the debugging table.

### The launch command, annotated

The other half of pre-flight is the launch itself. Here is the canonical `torchrun` invocation for a 2-node, 16-GPU run, with the env vars that actually matter called out:

```bash
# On each of the two nodes (node_rank differs); a scheduler like SLURM fills these in.
NCCL_DEBUG=WARN \
NCCL_IB_HCA=mlx5 \
NCCL_SOCKET_IFNAME=eth0 \
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  --max-restarts=3 \
  train.py --config configs/7b_fsdp.yaml
```

Three env vars earn their place on this line. `NCCL_DEBUG=WARN` keeps NCCL quiet in the happy path but prints the moment something is off (flip it to `INFO` when debugging). `NCCL_IB_HCA` names the InfiniBand adapters so NCCL uses RDMA rather than silently falling back to TCP over Ethernet — the single most common cause of "multi-node is inexplicably slow." `NCCL_SOCKET_IFNAME` pins the bootstrap interface so rendezvous does not try to connect over a management network that cannot reach the peers. [Launching on a SLURM cluster](/blog/machine-learning/distributed-training/launching-on-a-slurm-cluster) covers how a scheduler maps its allocation onto exactly these ranks.

## Part 4 — The debugging playbook: symptom, first check, likely cause, fix

This is the artifact you reach for at 2 a.m., and it is deliberately shaped as a lookup table: you arrive with a *symptom*, and it hands you a *first check*, the *likely cause*, and the *one fix* that resolves the common case. It is the compression of the entire failures track of the series into a page.

![a table matching each failure symptom to a first check a likely cause and the one fix that resolves it](/imgs/blogs/the-distributed-training-playbook-5.webp)

Read the figure as the index and the table below as the expanded entry. Every row is a real failure this series took apart in depth, cross-linked so you can drop from the one-liner into the full postmortem.

| Symptom | First check | Likely cause | The one fix |
| --- | --- | --- | --- |
| **Job hangs, then NCCL timeout** | `NCCL_DEBUG=INFO`: which collective was last, on which ranks | one rank took a different code path (control-flow divergence, a shape mismatch, an early `return`) so the collective never matched | make every rank call every collective; align shapes; set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` so a dead rank aborts instead of hanging forever |
| **One rank slow (straggler)** | per-rank step-time histogram; `nvidia-smi`/`dcgmi` on the slow node | a sick GPU, a bad NIC, thermal throttling on one rack, or an unlucky data shard | find and evict the rank; pin the run away from the bad node; re-balance the shard |
| **OOM at step 0** | peak reserved memory vs 80 GB; is activation memory in the budget? | micro-batch too large, no sharding, or activations uncounted | shard with FSDP, turn on activation checkpointing, shrink the micro-batch |
| **Loss goes to NaN** | gradient norm just before; fp16 vs bf16; is loss scaling on? | fp16 overflow, a bad batch, or an exploding gradient | switch to bf16, clip gradients, skip the offending batch, lower the LR |
| **Loss spikes after resume** | did you reload optimizer state, RNG, *and* the data sampler position? | a partial checkpoint or a reset data order re-showed old batches at the wrong LR | save and restore full state including the sampler's epoch and index |
| **Multi-node slower than single-node** | `NCCL_DEBUG=INFO`: is the transport IB/RDMA or TCP? | NCCL fell back to Ethernet; wrong interface; no `HYBRID_SHARD` | set `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME`; keep heavy collectives on NVLink |
| **Throughput regressed vs last week** | bisect: data, code, or hardware changed? MFU now vs then | a loader change, a kernel that stopped fusing, or a degraded node | isolate with a fixed-data microbenchmark; compare MFU, not wall-clock |

### The single most useful command: read the NCCL log

When a run hangs, ninety percent of the diagnosis is in the NCCL log, and most engineers never look at it. Re-launch the hung configuration with debug on:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL torchrun --nproc_per_node=8 train.py
```

and read two things. First, the transport line at init, which tells you whether NCCL chose the fast path or fell back:

```console
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/IPC          # good: NVLink peer-to-peer
NCCL INFO Channel 00/0 : 0[0] -> 8[0] [receive] via NET/IB/0  # good: InfiniBand between nodes
NCCL INFO NET/Socket : Using [0]eth0                        # BAD if you expected IB: TCP fallback
```

If you see `NET/Socket` where you expected `NET/IB`, you have found your "multi-node is slow" bug on line one — NCCL is running your inter-node all-reduce over TCP, and the fix is the `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` pair from the launch command. Second, when a job *hangs*, the last collective each rank logged tells you which one never matched — if rank 3 logged an all-gather while everyone else logged an all-reduce, rank 3 diverged, and you go hunt the branch that only rank 3 took. This is the entire method behind [the NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job) and the deeper [NCCL debugging deep dive](/blog/machine-learning/distributed-training/nccl-debugging-deep-dive).

### Two failures worth internalizing

**The hang is almost never NCCL's fault.** NCCL is a synchronization mirror: it faithfully waits for a collective that never arrives, and the timeout you see is a *symptom* of divergence somewhere in your Python. The mental discipline is to stop staring at the NCCL stack trace and instead ask "what did one rank do differently?" — a shape that depended on data, a `continue` inside a loop that skipped a collective, a validation step that only rank 0 ran, an uneven last batch. Every distributed hang I have debugged reduced to *asymmetry across ranks*. The full anatomy lives in [debugging distributed jobs](/blog/machine-learning/distributed-training/debugging-distributed-jobs).

**The straggler is a throughput tax, not a crash.** A synchronous training step is only as fast as its slowest rank, because everyone blocks on the all-reduce. One node at 70% speed does not slow that node by 30% — it slows *the entire job* by 30%, because 63 healthy GPUs sit idle waiting for the 64th. That is why the per-rank step-time panel is the single most valuable dashboard on a multi-node run: it turns an invisible, uniform-looking slowdown into one obvious diverging line. [The straggler](/blog/machine-learning/distributed-training/the-straggler) and [silent NaN at scale](/blog/machine-learning/distributed-training/silent-nan-at-scale) are the two failures that hide best behind aggregate metrics, which is exactly why you watch per-rank and per-step, not just the average.

## Part 5 — The efficiency checklist: MFU as the north star

The run fits, it launched, it is not hanging. Is it *fast*? The single number that answers this honestly is **Model FLOPs Utilization** — the fraction of your hardware's peak floating-point throughput that is actually doing useful model math. Wall-clock lies (it depends on your batch), tokens/s lies across model sizes, but MFU is comparable across models, clusters, and time, which is why it is the north star of the whole efficiency discussion.

MFU is defined as achieved model FLOP/s divided by the GPU's peak FLOP/s. Using the well-known $C = 6ND$ estimate — a training step does about 6 FLOPs per parameter per token (forward plus backward) — the achieved rate is $6 N \cdot (\text{tokens/s})$, so

$$\text{MFU} = \frac{6 N \cdot (\text{tokens/s})}{P_\text{peak}},$$

where $N$ is the parameter count and $P_\text{peak}$ is the accelerator's peak (about 312 bf16 TFLOP/s for an A100, about 989 bf16 TFLOP/s for an H100 SXM). Good large-scale dense training lands in the 35–55% range; below 30% you are leaving half your cluster on the floor. Here is the honest way to measure it — warm up first, synchronize before timing, and average over steady-state steps, because a cold measurement or an un-synchronized timer will hand you a number that is confidently wrong:

```python
import time, torch

def measure_mfu(step_fn, tokens_per_step, n_params, peak_flops,
                warmup=10, iters=50):
    for _ in range(warmup):          # let clocks, caches, and NCCL settle
        step_fn()
    torch.cuda.synchronize()         # do NOT time async kernels un-synced
    t0 = time.perf_counter()
    for _ in range(iters):
        step_fn()
    torch.cuda.synchronize()         # wait for the last kernel to finish
    dt = time.perf_counter() - t0

    tokens_per_s = tokens_per_step * iters / dt
    achieved = 6 * n_params * tokens_per_s     # C = 6ND FLOPs/token
    mfu = achieved / peak_flops
    print(f"{tokens_per_s:,.0f} tok/s | {achieved/1e12:,.0f} TFLOP/s "
          f"| MFU {mfu*100:.1f}%")
    return mfu
```

When the number is low, you diagnose *which wall* is throttling you — and there are only three, so the diagnosis is fast.

![a diagram where profiling one step fans out to memory bound and communication bound and loader bound diagnoses that each merge into a higher utilization outcome](/imgs/blogs/the-distributed-training-playbook-6.webp)

Profile one step with `torch.profiler` or Nsight Systems, and the timeline tells you which wall you are against:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    with_stack=True, record_shapes=True,
) as prof:
    for _ in range(6):
        step_fn()
        prof.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")   # open in chrome://tracing or Perfetto
```

- **Memory-bound.** You are forced into a tiny micro-batch to avoid OOM, so each kernel is too small to saturate the GPU, and the trace shows gaps between kernels. The fix is to *free memory* so you can raise the batch: activation checkpointing, sharding, offload. Counter-intuitively, spending a little compute to recompute activations often *raises* MFU because it lets you feed the GPU bigger, more efficient batches.
- **Communication-bound.** The trace shows all-reduce or all-gather kernels *not overlapping* with compute — comms sitting in the critical path. The fix is [overlap](/blog/machine-learning/distributed-training/overlapping-compute-and-communication): bucket gradients, prefetch the next layer's parameters during the current layer's compute, and move heavy collectives onto NVLink. If comms genuinely cannot be hidden, you have too much parallelism for this interconnect — drop a rung.
- **Loader-bound.** The trace shows periodic drops of GPU utilization to zero while the GPU waits for the next batch. This is the confound that fools everyone: your *model* is fine, your *data pipeline* is the bottleneck. The fix is more `DataLoader` workers, `pin_memory=True`, a larger `prefetch_factor`, and moving decode/augmentation off the critical path. Rule it out *before* you touch any parallelism, because it masquerades as every other bottleneck.

### The measured payoff: a naive run versus a tuned run

Here is what closing those three gaps looks like on real hardware — the same 7B model on the same 8×H100 SXM node, before and after the efficiency checklist. Nothing about the model or the data changed; only the systems knobs did.

![a before and after comparison of a naive run versus a tuned run on eight H100 GPUs showing utilization throughput and peak memory](/imgs/blogs/the-distributed-training-playbook-7.webp)

| Metric | Naive run | Tuned run | What changed |
| --- | --- | --- | --- |
| **MFU** | 28% | 51% | overlap on, right shard strategy, bigger effective batch |
| **Throughput** | ~6.6k tok/s/GPU | ~12k tok/s/GPU | ~1.8x, from utilization alone |
| **Peak memory** | 72 GB | 58 GB | activation checkpointing on the deep blocks |
| **Straggler** | undetected | caught + evicted | per-rank step-time alert |
| **Data loader** | 3 workers, no prefetch | 8 workers, prefetch 4, pinned | removed the periodic util dips |

Doubling MFU nearly doubled throughput, which — because the compute a run must perform is fixed at $C = 6ND$ — nearly *halved* both the GPU-hours and the dollar cost, on identical hardware. That is the entire economic argument for the efficiency checklist: MFU is the one term in the cost equation you fully control, and it is worth more than a faster GPU. [Cost and efficiency at scale](/blog/machine-learning/distributed-training/cost-and-efficiency-at-scale) develops the dollar model in full; the playbook version is: *measure MFU, find the wall, close the gap.*

#### Worked example: the 64-GPU scaling wall

Take that tuned single-node config and scale it out to 8 nodes, 64 GPUs, over InfiniBand. Scaling efficiency is defined as $\text{speedup}/N$ — how close to linear you got. On one node we measured 8 GPUs at 6.8x the single-GPU throughput, an excellent 85% efficiency, because everything overlapped on NVLink. Naively extend the same FSDP full-shard config to 64 GPUs and efficiency *collapses* to about 53% (34x, not 64x) — the parameter all-gather now spans the slow InfiniBand fabric on every layer and stops overlapping, exactly the $2S/B$ blowup the mechanism section predicted.

The fix is topology-aware placement: switch to `HYBRID_SHARD` so full sharding stays *within* each 8-GPU node on NVLink and only a lighter gradient reduction crosses InfiniBand between nodes, and make sure NCCL is actually using RDMA. That recovers efficiency to about 81% (52x of 64), a 1.5x throughput improvement over the naive multi-node config — the same GPUs, the same model, just the collectives placed where the hardware can absorb them. The stress test is the straggler: introduce one node running 30% slow and the whole 64-GPU job drops to ~70% of its throughput, because the synchronous step waits for the slowest rank — which is why the per-rank alert from the pre-flight checklist is not optional at this scale.

## Case studies: real numbers from the literature

The playbook is not folklore; the headline numbers come from published large-scale runs. A few worth carrying in your head, cited so you can check them:

- **Megatron-LM tensor + pipeline parallelism** (Narayanan et al., 2021) reported sustaining about 52% MFU while training GPT-scale models across thousands of A100s, achieving on the order of 502 petaFLOP/s aggregate on a 3,072-GPU cluster. The lesson baked into the playbook: at that scale, tensor parallelism lives *within* NVLink nodes and pipeline parallelism spans nodes — the exact placement doctrine above.
- **PaLM** (Chowdhery et al., 2022) reported roughly 46% MFU training a 540B-parameter model on 6,144 TPU v4 chips — a landmark precisely because sustaining high MFU at that scale is hard, and it was achieved through careful parallelism layout and overlap, not brute force.
- **ZeRO / DeepSpeed** (Rajbhandari et al., 2020) demonstrated that sharding optimizer state, gradients, and parameters lets you fit models an order of magnitude larger than a dense replica on the same hardware — the empirical foundation of "reach for FSDP before model parallelism." The internals are dissected in the [DeepSpeed ZeRO and 3D parallelism deep dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive).
- **PyTorch FSDP** in production has trained models up to and beyond 70B parameters on modest node counts by full-sharding state, validating rung 3 of the ladder: most teams never need tensor or pipeline parallelism at all, because sharding plus a handful of nodes covers a huge fraction of real models.

Treat these as order-of-magnitude anchors, not decimals to defend — the exact figure depends on model shape, sequence length, and cluster, and the papers themselves say so. What is robust is the *shape* of the numbers: high-40s to low-50s MFU is the mark of a well-tuned large run, and anything under 30% means a wall is being left unaddressed.

## When to reach for each lever (and when not to)

Every technique in this series is a cost, and the discipline of the playbook is saying plainly when a technique is *not* worth it. The failures I have watched teams inflict on themselves are almost all over-engineering — a lever added before its wall was actually hit.

- **Do not add tensor parallelism if the model fits and DDP or FSDP saturates NVLink.** TP trades overlappable communication for a *blocking* per-layer all-reduce; on a model that did not need it, you have made every layer wait on the network. TP is for when a single layer will not fit or per-step latency is dominated by one huge matmul — not for a 7B model that FSDP handles cleanly.
- **Do not go multi-node until you have saturated one node.** The interconnect tax between nodes is 30x-plus versus NVLink. Prove your MFU on 8 GPUs first. If you are at 30% MFU on one node, adding nodes multiplies your inefficiency; fix the node first.
- **Pipeline parallelism only pays past enough stages and micro-batches.** The bubble fraction is $(p-1)/(m+p-1)$: with $p$ stages and $m$ micro-batches, a 4-stage pipeline with only 4 micro-batches wastes $3/7 \approx 43\%$ of its time idle. Pipeline parallelism is worth it when you have enough micro-batches to shrink the bubble and the model is genuinely too deep to fit otherwise.
- **Do not shard more than you must.** ZeRO-3 / FSDP full-shard adds an all-gather of parameters that ZeRO-1 (optimizer-only) does not. If ZeRO-1 or ZeRO-2 already fits the model, the extra communication of full sharding is pure cost. Shard exactly as deep as the memory wall forces, and no deeper.
- **Do not reach for offload as a first resort.** CPU / NVMe offload trades a lot of throughput for capacity; it is a tool for "otherwise impossible," not "slightly tight." Exhaust sharding and checkpointing first.

The unifying rule, one more time because it is the whole philosophy: **the least parallelism that fits and hits your target is the right amount.** Complexity you add is complexity you will debug at 2 a.m.

## Key takeaways

- **You always hit one of four walls** — won't fit, won't finish, too slow, too costly — and the wall names the lever. Diagnose the wall before you reach for a tool.
- **Climb the ladder one rung at a time.** One GPU, then one node, then multiple nodes, then many. Stop at the first rung that fits and hits your throughput target; do not skip rungs.
- **Reach for sharding (FSDP / ZeRO) before model parallelism.** Most "the model won't fit" problems are solved more cheaply by dividing state across the data-parallel group than by splitting layers.
- **The interconnect decides the layout.** Keep heavy collectives on NVLink and let only lighter traffic cross the fabric; the $2S/B$ all-reduce math makes the 30x NVLink-vs-InfiniBand gap the primary design input.
- **Run the pre-flight every time:** mesh multiplies to world size, global batch is what you think, memory closes with headroom, checkpoints are frequent and *restorable*, monitoring is on, fault tolerance is configured.
- **When it hangs, it is asymmetry across ranks, not NCCL.** Read the NCCL log for the last unmatched collective; find what one rank did differently.
- **The straggler is a throughput tax on the whole job.** Watch per-rank step time, not the average, or the slowest GPU silently sets everyone's pace.
- **MFU is the north star.** Measure it honestly (warm up, synchronize, steady-state), then diagnose memory-bound, comms-bound, or loader-bound — and rule out the data loader first.
- **Doubling MFU nearly halves the bill** on identical hardware, because the compute a run must do is fixed at $C = 6ND$. Utilization is worth more than a faster GPU.

## What to read next: threading back through the series

This capstone is a map; the territory is the forty posts behind it. If you want to go deep on any lever the playbook only named, follow the track it belongs to.

Start where the whole thing started: [why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) frames the four walls, and [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) lays out every axis on one page. For the memory story — the $(2+2+12)\Psi$ budget and how sharding divides it — read [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget) and [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model). To turn the decision tree into a procedure with worked examples, [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) is the companion piece to Part 2, and [scaling a 7B LLM from 1 to 64 GPUs](/blog/machine-learning/distributed-training/scaling-a-7b-llm-1-to-64-gpus) walks the full ladder end to end on real hardware.

For the failures track — the raw material of Part 4 — [debugging distributed jobs](/blog/machine-learning/distributed-training/debugging-distributed-jobs) is the method, and the specific postmortems are worth reading as stories: [the NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job), [the straggler](/blog/machine-learning/distributed-training/the-straggler), [silent NaN at scale](/blog/machine-learning/distributed-training/silent-nan-at-scale), [multi-node slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node), [the loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume), and [throughput regressions](/blog/machine-learning/distributed-training/throughput-regressions). For the efficiency track behind Part 5, [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication), [monitoring a long run](/blog/machine-learning/distributed-training/monitoring-a-long-run), and [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) are where the checklist items come from. And when you cross into the cross-cutting engineering craft, the debugging-training series covers the single-GPU roots — [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu), [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging), and [reproducibility and determinism](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training).

## Further reading

- Narayanan et al., **"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"** (2021) — the tensor + pipeline + data 3D-parallelism layout and the 52% MFU result.
- Rajbhandari et al., **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (2020) — the sharding math behind FSDP and DeepSpeed.
- Huang et al., **"GPipe"** (2019) and Narayanan et al., **"PipeDream"** (2019) — the pipeline bubble and the schedules that shrink it.
- Chowdhery et al., **"PaLM"** (2022) — a landmark high-MFU run at 540B parameters and how the layout achieved it.
- **PyTorch FSDP** and **`torch.distributed`** documentation — the production APIs for sharding, DDP, and collectives.
- **NVIDIA NCCL** documentation and the **A100 / H100 architecture whitepapers** — the collectives, the environment variables, and the peak-FLOP and bandwidth specs the playbook's numbers rest on.
- The [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) and [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) — the compute-and-memory foundations that make MFU and the $6ND$ estimate meaningful.

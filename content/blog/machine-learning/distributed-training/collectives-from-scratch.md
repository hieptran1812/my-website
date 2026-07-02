---
title: "Collectives From Scratch: How GPUs Actually Talk"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Build every distributed-training collective from first principles, derive the ring all-reduce byte law by hand, and learn to read the one operation that decides whether eight GPUs run eight times faster or hang forever."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "nccl",
    "all-reduce",
    "collective-communication",
    "pytorch",
    "ddp",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

You add a second GPU to your training script, wrap the model in `DistributedDataParallel`, launch with `torchrun`, and watch the throughput number. It goes up by 1.6x. You add six more GPUs — eight total — and it goes up by 3.9x. Not 8x. Not even 6x. You are burning eight GPUs of electricity and getting less than four GPUs of work. Somewhere in that step, more than half of your hardware is sitting idle, waiting.

Waiting for what? For the other GPUs. Every training step, the eight GPUs have to agree on a single averaged gradient before they can take an optimizer step, and to agree they have to move bytes between each other over whatever wire connects them. That byte-movement is invisible. It does not show up in your Python. It has no line in your model definition. But it is the single thing standing between you and linear scaling, and if you cannot see it, you cannot fix it. This post makes it visible.

The unit of GPU-to-GPU communication is the **collective** — an operation that a whole *group* of processes execute together, each contributing data and each walking away with a result. `broadcast`, `reduce`, `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `barrier`: these seven cover essentially every distributed training system ever built. Data parallelism is one `all_reduce` per step. ZeRO and FSDP are a `reduce_scatter` plus an `all_gather`. Tensor parallelism is an `all_reduce` inside every transformer block. Mixture-of-experts is an `all_to_all`. Learn the seven primitives and their byte costs and you can predict, on the back of an envelope, whether a given parallelism strategy will scale on a given cluster — before you write a line of code.

![a data parallel training step where forward and backward run locally on each rank and a single gradient all-reduce is the only communication before the optimizer step](/imgs/blogs/collectives-from-scratch-1.webp)

By the end you will be able to: name what goes in, what comes out, and who holds the result for every collective; derive the famous ring all-reduce law that it moves ${2(N-1)/N \cdot S}$ bytes per GPU and is bandwidth-optimal; explain why ZeRO and FSDP can shard state at all (they cut the all-reduce in half and keep only the middle); write and launch `torch.distributed` collectives on four ranks and read the output; measure your interconnect's *achieved* bandwidth in GB/s and compare it to the spec; compute how long it takes to all-reduce a 7B model's gradients on NVLink versus PCIe versus InfiniBand; and recognize the number-one distributed bug — the collective that hangs the whole job because one rank showed up with the wrong shape. This is the third post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series; it sits right underneath everything that comes later.

## A training step is secretly a sequence of collectives

Before the primitives, three words you will see on every line of every distributed log. A **rank** is one process in the job — usually one process per GPU, so "rank 3" means "the process driving GPU 3." The **world size** is the total number of ranks, so eight GPUs on one node is a world size of eight. **Local rank** is the rank's index *within its own node*, which is what you use to pick which physical GPU to bind to. When you launch with `torchrun --nproc_per_node=8`, torchrun spawns eight processes, hands each one a distinct `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` through environment variables, and your code reads them to figure out who it is. A collective is something all `WORLD_SIZE` ranks call together.

Now look at what a single step of ordinary data-parallel training actually does. Each GPU has its own full copy of the model. Each GPU pulls a different slice of the batch — rank 0 gets examples 0 to 31, rank 1 gets 32 to 63, and so on — and runs the forward pass entirely locally. No communication. Each GPU runs the backward pass entirely locally and ends up with a full gradient for every parameter. Still no communication. But now every rank holds a *different* gradient, because every rank saw different data. If they each stepped their optimizer right now, the eight model copies would drift apart and you would no longer be training one model — you would be training eight models badly.

So before the optimizer step, all eight ranks must agree on one gradient: the average of the eight local gradients. Every rank needs that same averaged gradient, because every rank has to apply the identical update to stay in sync. "Every rank contributes its gradient, sums them, and every rank gets the sum back" is the exact definition of an **all-reduce**. That one collective, sitting between backward and the optimizer step, is the *entire* communication cost of data parallelism. The figure above draws it: the two local phases branch out per rank, the gradients merge into a single all-reduce, and the averaged result flows into a synchronized optimizer step. Everything expensive about scaling data parallelism lives in that one caution-colored box.

This is the frame for the whole series. You pick a parallelism strategy to knock down one of the four walls — the model won't fit, the data won't finish, the run is too slow, the cost is too high — and every strategy pays for itself in collectives over the interconnect. Data parallelism costs one all-reduce of the gradients. Tensor parallelism costs an all-reduce inside every layer. FSDP trades that one big all-reduce for a reduce-scatter of gradients plus all-gathers of parameters. If you can price the collective, you can price the strategy. So let us build all seven.

## The seven primitives, built from first principles

Here is the trick to never confusing these again: define each one by three questions. What does each rank put *in*? What does each rank get *out*? And how many bytes must leave each GPU to make that happen? Everything else — the name, the use case, the NCCL log line — follows from those three answers. Throughout, let `N` be the world size and `S` be the size in bytes of the buffer each rank contributes.

![a comparison table of the seven collective primitives listing input, output, which ranks hold the result, and the bytes moved per GPU for each](/imgs/blogs/collectives-from-scratch-2.webp)

**Broadcast.** One rank (the root) has a buffer; everyone else has garbage. After the broadcast, every rank has a copy of the root's buffer. Input: one buffer on the root. Output: the same buffer everywhere. This is how you distribute the initial model weights at the start of training — rank 0 loads the checkpoint, broadcasts it, and now all ranks start from identical weights. Bytes per GPU: order `S`, since the data has to reach `N-1` other ranks but a good tree does it in log-N rounds without any single link carrying more than about `S`.

**Reduce.** The mirror image of broadcast. Every rank has a buffer; you combine them with an operator (sum, max, min, product) and the *root* ends up with the single combined result. Input: `N` buffers. Output: one reduced buffer, on the root only. Everyone else keeps nothing. You reach for this when only one process needs the answer — say, summing a scalar loss to rank 0 for logging. Bytes: order `S`.

**All-reduce.** Reduce, but *everyone* gets the result, not just the root. Every rank contributes a buffer, they are summed, and every rank walks away with the identical sum. Input: `N` buffers. Output: the same reduced buffer on all `N` ranks. This is the workhorse — the gradient averaging in DDP, the activation sums in tensor parallelism. It is the most important collective in this entire series, and its byte cost, which we will derive in full, is ${2(N-1)/N \cdot S}$ per GPU.

**All-gather.** No reduction at all — pure concatenation. Each rank holds a *shard*: rank 0 has shard 0, rank 1 has shard 1, and so on, each of size `S/N`. After the all-gather, every rank holds the full concatenation of all `N` shards, size `S`. Input: `N` shards of `S/N` each. Output: the full `S`-byte buffer on all ranks. This is how FSDP reconstructs a full parameter tensor from its shards right before it needs to compute with it. Bytes per GPU: `(N-1)/N · S` — each rank has to receive the `N-1` shards it did not already own.

**Reduce-scatter.** The strangest-sounding one, and the most important to internalize, because it is half of an all-reduce. Every rank contributes a full `S`-byte buffer. The result is summed *element-wise* across ranks, like a reduce — but then the sum is *scattered*: rank 0 keeps only the first `S/N` chunk of the total, rank 1 keeps the second chunk, and so on. Input: `N` full buffers. Output: one `S/N` summed shard per rank, a different shard on each. Nobody holds the whole sum. Bytes per GPU: `(N-1)/N · S`, the same as all-gather. This is exactly the collective that lets FSDP average gradients without ever materializing the full averaged gradient on any single GPU.

**All-to-all.** A transpose across ranks. Each rank starts with `N` blocks, one destined for each rank; after the all-to-all, each rank holds the block that every other rank sent to *it*. If you lay the data out as an `N` by `N` grid where cell (i, j) is "the block rank i wants to send to rank j," all-to-all transposes it so rank j ends up with column j. Input: `N` blocks per rank. Output: the transposed blocks. This is the routing step in mixture-of-experts training: each token has to travel to whichever GPU holds its chosen expert, and every GPU is simultaneously sending and receiving tokens. Bytes per GPU: `(N-1)/N · S`.

**Barrier.** The degenerate collective — it moves no user data at all. Every rank calls `barrier()`, and no rank returns from the call until *all* of them have arrived. Input: nothing. Output: a synchronization point. You use it to make sure every rank has finished writing a checkpoint before rank 0 uploads it, or to line up ranks before a timed benchmark. Bytes: essentially zero, but — and this matters — a barrier is still a rendezvous, so a rank that never calls it hangs everyone, exactly like every other collective.

Read down the "bytes per GPU" column and a pattern jumps out. Everything that touches all the data and leaves everyone with a full copy costs around `S` or `2S`. Everything that leaves each rank holding only a `1/N` slice costs `(N-1)/N · S` — which, as `N` grows, approaches a flat `S`. That is not a coincidence. It is the whole reason sharding works.

### Which parallelism uses which collective

Before we exploit that pattern, here is the map that makes the rest of the series legible. Every parallelism strategy you will meet is a particular pattern of these seven primitives, and knowing which one it leans on tells you immediately where its communication cost lives and on what part of the cluster it must run.

| Strategy | Primary collective(s) | How often | Where it must live |
|---|---|---|---|
| Data parallel (DDP) | all-reduce of gradients | once per step | fast fabric; overlaps backward |
| ZeRO / FSDP | reduce-scatter grads + all-gather params | per layer, on demand | fast fabric; extra all-gather |
| Tensor parallel (Megatron) | all-reduce of activations | twice per layer | inside one NVLink node |
| Pipeline parallel | point-to-point send/recv | per micro-batch boundary | any link; small messages |
| Expert parallel (MoE) | all-to-all of tokens | twice per MoE layer | fast fabric; latency-sensitive |
| Sequence / context parallel | all-gather or ring send/recv | per attention block | fast fabric; large messages |
| Startup / checkpoint load | broadcast of weights | once per job | any link; one-time |
| Synchronized logging / eval | reduce or barrier | occasional | any link; tiny messages |

Two rows deserve a second look. Tensor parallelism does an all-reduce *twice per layer*, which for a deep model is dozens of all-reduces per step — orders of magnitude more collective traffic than data parallelism's single one. That is why it is confined to a single NVLink node. Pipeline parallelism, by contrast, only sends activations across the boundary between two stages, a comparatively small point-to-point transfer, which is why it tolerates slower links and is often the dimension you stretch across nodes. The collective *is* the strategy's cost, and the byte counts we just tabulated are how you predict it. All of this rests on one small algebraic fact about the all-reduce, so let us prove it.

## The identity that makes sharding possible

Here is the single most useful equation in distributed training, and it has no exponents in it:

$$\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}$$

Read it slowly. An all-reduce leaves every rank with the full summed buffer. A reduce-scatter leaves every rank with one summed *shard*. An all-gather takes shards and reconstructs the full buffer on everyone. So if you reduce-scatter and *then* all-gather, you get: sum the buffers and give everyone one shard of the sum, then have everyone exchange shards so they all reconstruct the whole sum. The end state is identical to an all-reduce — every rank holds the full sum — and you got there in two steps.

![a dataflow graph showing an all-reduce decomposed into a reduce-scatter that leaves each rank with one summed shard and an all-gather that reconstructs the full buffer, with a branch where sharding frameworks stop at the midpoint](/imgs/blogs/collectives-from-scratch-3.webp)

Why does anyone care about splitting one collective into two? Because of the *midpoint*. Look at the figure: after the reduce-scatter, before the all-gather, each rank is holding exactly one `S/N` shard of the summed result — and nothing else. That intermediate state is a fully-averaged, fully-sharded buffer. If what you are averaging is *gradients*, then at the midpoint each rank has the final averaged gradient for its own slice of the parameters, and only that slice. It never needs the rest.

That is the entire idea behind ZeRO and FSDP. Standard DDP does a full gradient all-reduce and every rank keeps a full copy of the averaged gradient, the full optimizer state, and the full parameters — so memory does not go down at all when you add GPUs, only throughput goes up. FSDP instead does the reduce-scatter half, stops at the midpoint, and each rank keeps only its `1/N` shard of the gradient and the optimizer state built on it. When it later needs the full parameters to compute a layer, it does the all-gather half, on demand, one layer at a time, and throws the gathered copy away afterward. The all-gather that FSDP "adds" compared to DDP is not extra work invented from nowhere — it is the *second half of the all-reduce that DDP was doing anyway*. FSDP just refuses to keep the full result around. We will spend a whole post on that in [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/ddp-from-first-principles); the point here is that it is only *possible* because of this one identity, and the identity is only obvious once you have drawn the midpoint.

Notice the byte accounting lines up perfectly, too. Reduce-scatter costs `(N-1)/N · S` and all-gather costs `(N-1)/N · S`; add them and you get `2(N-1)/N · S` — exactly the all-reduce cost we claimed earlier. The all-reduce is not cheaper or more expensive than doing its two halves; it *is* its two halves. Which is a strong hint about how it is actually implemented. Let us derive that.

## Deriving ring all-reduce: the 2(N-1)/N law

There is a naive way to do an all-reduce and a smart way, and the gap between them is the difference between a job that scales and a job that does not.

The naive way: pick rank 0 as a coordinator. Everyone sends their `S`-byte buffer to rank 0. Rank 0 sums all `N` of them and broadcasts the result back. Count the bytes crossing rank 0's single network link: it *receives* `(N-1)·S` bytes on the way in and *sends* `(N-1)·S` bytes on the way out. That link carries traffic proportional to `N`. Double your GPU count and rank 0's link has to move twice as much data in the same step. This is the textbook definition of not scaling — you have built a hot spot, and it gets hotter with every GPU you add. On eight GPUs averaging a 14 GB gradient, rank 0 alone would shovel about 196 GB. No interconnect saves you from an algorithm like that.

The smart way is the **ring all-reduce**, and its defining property is that *no link ever carries more than about `S` bytes regardless of `N`*. Arrange the `N` GPUs in a logical ring: 0 sends to 1, 1 sends to 2, ..., `N-1` sends back to 0. Every GPU has exactly one neighbor it sends to and one it receives from, and — crucially — all `N` links are used *at the same time*, in parallel. Now chop each rank's `S`-byte buffer into `N` equal chunks of `S/N` bytes.

The algorithm runs in two phases, and the first phase is a reduce-scatter.

**Phase one — reduce-scatter, `N-1` steps.** On each step, every rank simultaneously sends one chunk to its right neighbor and receives one chunk from its left neighbor, and *adds* the received chunk into its own copy of that chunk. The chunks are chosen so that, after `N-1` steps, each rank has accumulated the complete sum of exactly one chunk. Concretely, on step 1 rank `i` sends chunk `i` and receives chunk `i-1`; on step 2 it sends the chunk it just completed (`i-1`) onward and receives chunk `i-2`; and so on, a "reduction wave" circulating the ring. After `N-1` steps the sum for chunk `k` has visited every rank and come to rest on rank `k`.

![a rank by chunk grid showing that after the reduce-scatter phase of a ring the fully summed data sits on the diagonal with each GPU owning one chunk](/imgs/blogs/collectives-from-scratch-4.webp)

The figure shows the end state on three GPUs: the completed sums sit on the diagonal — GPU 0 owns the total for chunk 0, GPU 1 owns chunk 1, GPU 2 owns chunk 2 — and every off-diagonal chunk has been handed onward and forgotten. This is precisely the reduce-scatter midpoint from the identity above, now shown as a physical layout. Count the traffic in this phase: each rank did `N-1` steps, and on each step it *sent* one chunk of `S/N` bytes. So each rank sent `(N-1) · S/N = (N-1)/N · S` bytes. That is the reduce-scatter cost, derived by counting.

**Phase two — all-gather, `N-1` steps.** Now every rank owns one complete chunk, and everyone needs all the chunks. Same ring, same one-chunk-per-step motion, but this time you *overwrite* instead of add: on each step every rank sends the complete chunk it currently holds to its right neighbor, which stores it. The completed chunks circulate the ring, and after `N-1` steps every rank has seen and stored all `N` chunks — the full summed buffer. Traffic: again `N-1` steps of `S/N` bytes sent per rank, another `(N-1)/N · S` bytes.

![a timeline of a ring all-reduce showing N minus one reduce-scatter steps followed by N minus one all-gather steps, each step moving one over N of the buffer](/imgs/blogs/collectives-from-scratch-5.webp)

Add the two phases and you have the law, derived not asserted:

$$T_\text{bytes per GPU} = \underbrace{\frac{N-1}{N} S}_\text{reduce-scatter} + \underbrace{\frac{N-1}{N} S}_\text{all-gather} = \frac{2(N-1)}{N} S$$

Three things are worth sitting with. First, as `N` grows, `2(N-1)/N` approaches 2, so each GPU moves at most about `2S` bytes *no matter how many GPUs you have*. Going from 8 GPUs to 64 GPUs to 512 GPUs barely changes the per-GPU byte count — that is what "bandwidth-optimal" means, and it is why data parallelism can scale to enormous clusters as long as the bandwidth holds up. Second, the *time*, not just the bytes, follows: if your interconnect delivers `B` bytes per second per GPU, the ring all-reduce takes about `2(N-1)/N · S / B` seconds, which is why the interconnect's `B` is the number that decides everything. Third — and this is the sharp edge — the ring takes `2(N-1)` sequential steps, so its *latency* grows linearly with `N`. For a big gradient buffer that does not matter, because each step moves a large chunk and bandwidth dominates. For a tiny buffer, those `2(N-1)` round-trips of pure latency are the whole cost, and the ring becomes a bad choice. Which is the entire reason NCCL has more than one algorithm.

#### Worked example: a ring all-reduce on four GPUs, step by step

Put numbers on it. Four GPUs (`N = 4`), each holding a 400 MB gradient buffer (`S = 400` MB). Chunk it into `N = 4` pieces of 100 MB each. The reduce-scatter phase runs `N-1 = 3` steps. On step 1, every GPU sends one 100 MB chunk right and receives one from the left, adding it in; on step 2 it forwards the chunk it just extended and receives the next; on step 3 the third addition lands, and each GPU now owns the complete sum of exactly one chunk — GPU 0 owns chunk 0, GPU 1 owns chunk 1, and so on, the diagonal from the grid figure. Bytes sent per GPU so far: 3 steps times 100 MB equals 300 MB, which is `(N-1)/N · S = 3/4 · 400 = 300` MB. Check.

The all-gather phase runs another `N-1 = 3` steps, circulating the completed chunks so everyone ends with all four. Another 300 MB sent per GPU. Total per GPU: 600 MB, which is `2(N-1)/N · S = 2 · 3/4 · 400 = 1.5 · 400 = 600` MB. Now price the time on a 200 GB/s link: `600e6 / 200e9 = 0.003` seconds, 3 milliseconds. Notice what the naive star-to-rank-0 algorithm would have done instead: rank 0 alone receives `(N-1)·S = 3 · 400 = 1200` MB and sends 1200 MB back — 2.4 GB across one link versus 600 MB across every link in parallel. The ring is 4x lighter on the busiest link at `N = 4`, and the gap widens with every GPU you add. That is the entire argument for the ring in one worked example.

## Ring versus tree: bandwidth-optimal versus latency-optimal

NCCL — NVIDIA's Collective Communications Library, the thing PyTorch actually calls when you run `dist.all_reduce` on GPUs — does not have one all-reduce implementation. It has several, and it picks one at runtime based on the message size, the number of ranks, and the physical topology it discovered at startup. The two you must know are the ring and the tree.

The **ring** we just derived: bandwidth-optimal, `2(N-1)` steps of latency. The **tree** trades that around. In a tree all-reduce, ranks are arranged as a binary tree; data is reduced up the tree to the root in about `log2(N)` steps and broadcast back down in another `log2(N)` steps. Latency is now logarithmic in `N` instead of linear — on 256 ranks that is roughly 8 levels versus 255 ring steps, a night-and-day difference for a small message. The cost is bandwidth: in a plain tree, interior links carry more traffic and some links sit idle, so you do not get the ring's clean `2(N-1)/N · S` optimum. NCCL's production tree is a *double binary tree* that uses two complementary trees at once so that every link does useful work in both directions, recovering most of the bandwidth while keeping the logarithmic latency — which is why it is the default for large multi-node all-reduces.

![a comparison table of ring, binary tree, and double binary tree algorithms across latency, bandwidth, best message size, and when NCCL selects each](/imgs/blogs/collectives-from-scratch-7.webp)

The practical upshot is a crossover. For **small** messages — a handful of kilobytes, where the cost is dominated by the number of sequential hops — the tree wins because `log N` beats `N`. For **large** messages — megabytes of gradient, where the cost is dominated by bytes moved — the ring (or a bandwidth-preserving tree) wins because it is bandwidth-optimal. NCCL estimates the cost of each algorithm using a latency-plus-bandwidth model and picks the cheaper one for the message in front of it. You can see the decision and override it:

```bash
# Watch NCCL announce which algorithm and protocol it selected, per collective.
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING torchrun \
  --nproc_per_node=8 train.py

# Force a specific algorithm to A/B test a suspected bad autotune choice.
NCCL_ALGO=Ring  torchrun --nproc_per_node=8 bench.py   # force ring
NCCL_ALGO=Tree  torchrun --nproc_per_node=8 bench.py   # force double-tree
```

You almost never need to set `NCCL_ALGO` by hand — the autotuner is good. But when a multi-node run is mysteriously slow and the messages are large, forcing `Ring` and comparing is a five-minute experiment that tells you whether the tuner mis-guessed. The [collective communication and NCCL all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) post goes deeper into NCCL's protocols (Simple, LL, LL128) and how it builds its rings and trees across a real topology; here we stay at the level of "which shape moves the bytes, and how many."

| Algorithm | Latency (steps) | Bandwidth per GPU | Sweet spot | Typical use |
|---|---|---|---|---|
| Naive (star to rank 0) | 2 | grows with `N` (hot spot) | never | a lesson in what not to do |
| Ring | `2(N-1)` | `2(N-1)/N · S`, optimal | large messages, one node | intra-node gradient all-reduce |
| Binary tree | `~2·log2(N)` | sub-optimal, idle links | small messages | small scalars, many nodes |
| Double binary tree | `~2·log2(N)` | near-optimal | medium to large, multi-node | NCCL's multi-node default |

## The code: torch.distributed collectives you can run

Enough theory — here is the machinery you can copy, launch on four GPUs, and watch. Every collective in `torch.distributed` follows the same shape: initialize the process group so the ranks can find each other, build a tensor, call the collective, and every rank ends up with the operation's defined result. Save this as `collectives.py`.

```python
import os
import torch
import torch.distributed as dist


def main():
    # torchrun sets these env vars; init_process_group reads them.
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)          # bind this process to one GPU
    dev = torch.device("cuda", local_rank)

    # --- all_reduce: everyone contributes, everyone gets the sum ---
    x = torch.tensor([rank + 1.0], device=dev)          # rank r holds r+1
    dist.all_reduce(x, op=dist.ReduceOp.SUM)            # in place
    # On 4 ranks: 1 + 2 + 3 + 4 = 10 on EVERY rank
    print(f"[rank {rank}] all_reduce sum -> {x.item():.0f}")

    # --- all_gather: collect one tensor from each rank into a list ---
    src = torch.tensor([rank], device=dev)
    gathered = [torch.zeros_like(src) for _ in range(world)]
    dist.all_gather(gathered, src)
    vals = [t.item() for t in gathered]
    print(f"[rank {rank}] all_gather -> {vals}")        # [0,1,2,3] on every rank

    # --- reduce_scatter: sum full buffers, keep only your shard ---
    # Each rank contributes a length-`world` buffer of its rank id.
    full = [torch.tensor([float(rank)], device=dev) for _ in range(world)]
    shard = torch.zeros(1, device=dev)
    dist.reduce_scatter(shard, full, op=dist.ReduceOp.SUM)
    # rank r keeps the r-th summed element: sum over ranks of r' = 0+1+2+3 = 6
    print(f"[rank {rank}] reduce_scatter shard -> {shard.item():.0f}")

    # --- all_to_all: rank r sends value (r*world + j) to rank j ---
    send = torch.tensor([rank * world + j for j in range(world)],
                        dtype=torch.float32, device=dev)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    print(f"[rank {rank}] all_to_all recv -> {recv.tolist()}")

    dist.barrier()                 # line everyone up before exit
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Launch it on a four-GPU box:

```bash
torchrun --standalone --nproc_per_node=4 collectives.py
```

Here is the output you should see. Rank ordering in the terminal is nondeterministic because four processes print concurrently, but the *values* are fixed by the math:

```console
[rank 0] all_reduce sum -> 10
[rank 1] all_reduce sum -> 10
[rank 2] all_reduce sum -> 10
[rank 3] all_reduce sum -> 10
[rank 0] all_gather -> [0, 1, 2, 3]
[rank 2] all_gather -> [0, 1, 2, 3]
[rank 1] all_gather -> [0, 1, 2, 3]
[rank 3] all_gather -> [0, 1, 2, 3]
[rank 0] reduce_scatter shard -> 6
[rank 1] reduce_scatter shard -> 6
[rank 2] reduce_scatter shard -> 6
[rank 3] reduce_scatter shard -> 6
[rank 0] all_to_all recv -> [0.0, 4.0, 8.0, 12.0]
[rank 1] all_to_all recv -> [1.0, 5.0, 9.0, 13.0]
[rank 2] all_to_all recv -> [2.0, 6.0, 10.0, 14.0]
[rank 3] all_to_all recv -> [3.0, 7.0, 11.0, 15.0]
```

Trace one line to make sure the model in your head matches the machine. The `all_reduce` sums `1+2+3+4=10` and every rank agrees — that is data-parallel gradient averaging in miniature (real DDP divides by `world` afterward to get a mean). The `all_gather` gives everyone the list `[0,1,2,3]`. The `reduce_scatter` sums the per-rank buffers element-wise — every element is `0+1+2+3=6` — and hands each rank one element, so all four print 6. And the `all_to_all` transposes: rank 0 keeps the "column 0" that every rank addressed to it, namely the values `0, 4, 8, 12` that ranks 0, 1, 2, 3 each sent to rank 0. If your run reproduces those numbers, your process group is healthy and your NCCL is talking.

### Broadcast and barrier: the primitives you use at the edges

The two quietest collectives, broadcast and barrier, are the ones you reach for at the *start* and *end* of things rather than in the hot loop. Every distributed job begins with a correctness problem: each rank builds its own model, and if they build from different random seeds or load a checkpoint independently, the ranks start from *different* weights and the first all-reduce silently averages garbage. The fix is a broadcast — rank 0 is the source of truth, everyone copies it — followed by a barrier so no rank races ahead before the weights have landed.

```python
import torch, torch.distributed as dist

def sync_initial_weights(model, src=0):
    # Make every rank's parameters identical to the source rank's.
    for p in model.parameters():
        dist.broadcast(p.data, src=src)          # rank `src` sends, others receive
    dist.barrier()                               # nobody proceeds until all synced

# Guard I/O so only rank 0 touches disk, then broadcast the loaded state.
if dist.get_rank() == 0:
    state = torch.load("checkpoint.pt", map_location="cpu")
    model.load_state_dict(state)
sync_initial_weights(model)                       # ranks 1..N-1 now match rank 0
```

This is exactly what `DistributedDataParallel` does for you under its constructor — it broadcasts rank 0's parameters and buffers so you never have to think about it — but the moment you hand-roll anything (a custom optimizer state, an EMA copy, a manually loaded adapter), you own this invariant again. Forget the broadcast and you get the most confusing bug in distributed training: a run that *looks* fine, loss decreasing, but converges to nonsense because the ranks were never training the same model. The barrier is the cheap insurance that the broadcast finished before anyone read the weights.

### Measuring achieved bandwidth honestly

Knowing the algorithmic byte count is half the story; the other half is how fast your wire actually moves them. The metric that matters is **bus bandwidth** — the effective GB/s the ring sustains — and you compute it from the algorithmic bytes and the measured time. There are three ways to get this wrong, so the benchmark below does all three fixes: it *warms up* (the first NCCL call pays a one-time setup and channel-establishment cost you must not time), it calls `torch.cuda.synchronize()` before reading the clock (CUDA is asynchronous — without a sync you are timing kernel *launches*, not kernel *completions*), and it averages many iterations to reach steady state.

```python
import os, time, torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

numel = 256 * 1024 * 1024                 # 256M floats
x = torch.ones(numel, dtype=torch.float32, device=dev)
S = x.element_size() * x.numel()          # bytes in the buffer = 1 GiB

for _ in range(10):                       # WARMUP — never timed
    dist.all_reduce(x)
torch.cuda.synchronize()

iters = 50
dist.barrier()
t0 = time.perf_counter()
for _ in range(iters):
    dist.all_reduce(x)
torch.cuda.synchronize()                  # wait for completion before stopping clock
t = (time.perf_counter() - t0) / iters

# Algorithmic bytes moved per GPU by ring all-reduce, and the bus bandwidth.
algo_bytes = 2 * (world - 1) / world * S
busbw = algo_bytes / t / 1e9              # GB/s
if rank == 0:
    print(f"world={world}  S={S/1e9:.2f} GB  time={t*1e3:.2f} ms  "
          f"busbw={busbw:.1f} GB/s")

dist.destroy_process_group()
```

The number this prints is the honest one: the bandwidth the ring *achieved*, accounting for the `2(N-1)/N` factor. You will find it is meaningfully below the interconnect's advertised peak — 60 to 80 percent of spec is a healthy result on NVLink, because the peak assumes perfect, one-directional streaming and the ring is doing bidirectional sends with reduction kernels in between. If you see 10 percent of spec on what you thought was NVLink, you are almost certainly falling back to PCIe or the network, and that is a topology bug to chase in [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics). NVIDIA's own `nccl-tests` (`all_reduce_perf`) report exactly this `busbw` column; matching your PyTorch number against `nccl-tests` is a good sanity check that your framework is not adding overhead.

Two more confounds turn a "measurement" into a fiction if you ignore them. The first is the **data-loader confound**: if you time a real training step rather than an isolated collective, and your `DataLoader` is starving the GPU, you will measure a step time inflated by input-pipeline stalls and wrongly blame the all-reduce. Isolate the collective (as the benchmark above does) before you attribute slowness to communication, and separately profile the loader with the GPU idle to see whether it can keep up. The second is **thermal and clock throttling**: a GPU under sustained load drops its clocks as it heats up, so a benchmark that runs for two seconds reports a higher bandwidth than the same kernel does in hour three of a real run. For an honest steady-state number, run long enough that clocks settle, and watch `nvidia-smi -q -d CLOCK,TEMPERATURE` (or `dcgmi dmon`) to confirm you are not measuring a boosted transient. The rule that ties all of this together: measure the *one* thing you are reasoning about, in the *steady state* the real job will actually live in, and never trust the first iteration of anything on a GPU.

## Worked example: all-reducing a 7B model's gradients

Now the payoff — pricing a real all-reduce three ways, using nothing but the law we derived. Take a 7-billion-parameter transformer trained in bf16. Each parameter has one gradient, and bf16 is 2 bytes, so the gradient buffer is about `7e9 · 2 = 14e9` bytes, or 14 GB. That is our `S`. Put it on eight GPUs, so `N = 8`.

First, the bytes each GPU moves, from the ring law: `2(N-1)/N · S = 2 · 7/8 · 14 GB = 1.75 · 14 = 24.5 GB` per GPU, per step. This number does not change with the interconnect — it is fixed by the algorithm and the model. What changes is how fast you can move it.

#### Worked example: the same all-reduce on three interconnects

**On NVLink4 (one DGX-style node, all eight GPUs on NVSwitch).** NVLink4 delivers roughly 900 GB/s of aggregate bandwidth per GPU. Time to move 24.5 GB: `24.5 / 900 ≈ 0.027 s`, about 27 milliseconds. For a 7B model, a forward-plus-backward step on eight H100s takes on the order of a few hundred milliseconds, so a 27 ms all-reduce is maybe 10 percent of the step — and because DDP overlaps the all-reduce with the backward pass (a trick we dissect in [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles)), most of it hides entirely behind compute. This is why single-node DGX data parallelism scales so well: the wire is faster than the math needs.

**On PCIe4 (eight GPUs, no NVLink, talking over the PCIe bus).** PCIe4 x16 gives roughly 32 GB/s per direction. Time to move 24.5 GB: `24.5 / 32 ≈ 0.77 s`, about 770 milliseconds. Now the all-reduce is *longer than the compute step*. Overlap cannot save you — there is not enough backward pass to hide 770 ms behind. Your eight GPUs spend more than half their wall-clock waiting on gradients, and you get the 3.9x-on-8-GPUs disappointment from the top of this post. Same bytes, 28x slower wire, and data parallelism stops scaling.

**Across nodes on InfiniBand HDR (eight GPUs spread over multiple nodes).** InfiniBand HDR runs at 200 Gb/s, which is 25 GB/s per direction. Time to move 24.5 GB: `24.5 / 25 ≈ 0.98 s`, about 1 second — and that is the *optimistic* figure assuming the ring is perfectly network-bound and GPUDirect RDMA is working so the data never detours through host memory. Cross-node all-reduce of a full 14 GB gradient every step is brutal, and it is exactly why large runs shard (FSDP) or reduce gradient traffic (gradient accumulation, bf16 comms, hierarchical all-reduce) rather than doing a naive full all-reduce over the network.

![a before and after comparison showing the same 24.5 GB per GPU gradient all-reduce taking about 0.77 seconds on PCIe versus about 27 milliseconds on NVLink](/imgs/blogs/collectives-from-scratch-6.webp)

The figure lines the two extremes up side by side: identical bytes, a 28-fold difference in time, and the same job flips from comms-bound to compute-bound purely because of the wire. This is the single most important intuition in data-parallel scaling. The interconnect, not the GPU, decides whether DDP scales.

| Interconnect | Bandwidth (per GPU/dir) | Bytes moved per GPU | All-reduce time | Verdict for 7B DDP |
|---|---|---|---|---|
| NVLink4 (NVSwitch) | ~900 GB/s | 24.5 GB | ~27 ms | scales; comms hide behind backward |
| PCIe4 x16 | ~32 GB/s | 24.5 GB | ~770 ms | comms-bound; DDP stalls |
| InfiniBand HDR | ~25 GB/s | 24.5 GB | ~980 ms | shard or accumulate, don't full-AR |

A quick stress test of the model. What if you go to 64 GPUs on a single fast fabric? The per-GPU bytes barely move — `2·63/64·14 ≈ 27.6 GB`, only 13 percent more than at 8 GPUs — so on NVLink/NVSwitch the all-reduce stays cheap and DDP keeps scaling, which is the good case. What if the batch per GPU is tiny? Then the backward pass is short, there is little compute to hide the all-reduce behind, and even a fast wire leaves the comms exposed — the fix is a bigger per-GPU batch or gradient accumulation, not a faster network. What if one node is a straggler? The all-reduce is a rendezvous, so the whole collective runs at the speed of the slowest rank, and every other GPU idles waiting for it — a failure mode we devote an entire post to later. And what if the optimizer state will not fit? Then you have left DDP's regime entirely and it is time to shard with FSDP, spending an extra all-gather to buy back memory.

### The comms-to-compute ratio, in one inequality

The NVLink-versus-PCIe result generalizes into the single inequality that governs whether *any* data-parallel run scales. Overlapping the all-reduce with the backward pass — the trick DDP uses to hide communication — works only if the communication finishes before the compute it hides behind does. Write the backward-pass time as roughly the model's backward FLOPs divided by the GPU's achieved throughput, call it `T_compute`, and the all-reduce time as `T_comms = 2(N-1)/N · S / B`, where `B` is the achieved interconnect bandwidth per GPU. Data parallelism scales when

$$T_\text{comms} < T_\text{compute} \quad\Longleftrightarrow\quad \frac{2(N-1)}{N}\cdot\frac{S}{B} < \frac{\text{backward FLOPs}}{\text{throughput}}$$

Everything in this series is a campaign to keep that inequality true. A faster interconnect raises `B`. A bigger per-GPU batch raises `T_compute` (more FLOPs to hide behind) without changing `S`, which is why tiny batches expose comms and large batches hide them. Gradient accumulation multiplies `T_compute` by the number of micro-steps while doing the all-reduce only once, which is the cheapest way to restore the inequality on a slow link. Bf16 gradient communication halves `S`. And sharding (FSDP) changes the shape of the problem entirely by spreading the same byte volume across on-demand per-layer collectives that each overlap their own layer's compute. When someone tells you "we added a second node and throughput barely moved," they are telling you `B` fell off a cliff and the inequality flipped — and now you know the four levers to flip it back.

## The number-one distributed bug: a collective is a rendezvous

Everything above assumed the collective *completes*. Here is what happens when it does not, and why it is the single most common way distributed jobs die.

A collective is a **rendezvous**: it does not return on any rank until *every* rank in the group has called it with compatible arguments. There is no timeout by default on the semantics — NCCL will wait. So if rank 3 hits a code path that skips the all-reduce, or calls it with a differently-shaped tensor, or crashes right before it, the other seven ranks call `all_reduce`, block, and *wait forever*. No error. No traceback. No progress. Your `nvidia-smi` shows eight GPUs pinned at 100 percent utilization — because they are busy-waiting inside NCCL — and the loss stops printing. To a newcomer it looks like the job is working hard. It is doing nothing.

The classic trigger is a **shape mismatch**. Suppose your batching logic occasionally hands rank 3 a tensor of a different length — a ragged last batch, a variable sequence length, a conditional that fires on some ranks and not others. Every rank calls `all_reduce`, but rank 3's buffer is a different size, so NCCL cannot pair up the sends and receives around the ring. Depending on versions you get a hang, or a cryptic `NCCL error: invalid usage`, or — worst of all — a silent partial reduction. Here is the shape of the bug, drawn as what each rank called:

| Rank | Called | Tensor shape | Outcome |
|---|---|---|---|
| 0 | `all_reduce(g)` | `[7_000_000_000]` | blocks, waiting for the ring |
| 1 | `all_reduce(g)` | `[7_000_000_000]` | blocks, waiting for the ring |
| 2 | `all_reduce(g)` | `[7_000_000_000]` | blocks, waiting for the ring |
| 3 | `all_reduce(g)` | `[6_999_999_488]` | shapes cannot pair — whole group hangs |

Three symptoms tell you it is a collective mismatch and not something else. The job hangs at a *consistent* step number (wherever the divergent data or code path first fires), not randomly. GPU utilization is pinned high while throughput is zero. And it is often *rank-dependent* — it hangs only when a particular rank sees particular data. The first-response toolkit is short: set `NCCL_DEBUG=INFO` and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` so a divergent collective raises after a timeout instead of hanging silently; set `TORCH_NCCL_TRACE_BUFFER_SIZE` to capture a flight recorder of the last collectives each rank issued; and audit for the three usual suspects — a conditional that runs a collective on some ranks only, a variable-shape tensor entering a collective, and a per-rank early `return`/`continue` that skips one. Shape is the loudest mismatch, but it is not the only one: the *dtype* must match (one rank in fp32 and the rest in bf16 desyncs the ring), the *reduce op* must match (a stray `MAX` where others use `SUM`), and — subtlest of all — the *order* must match, because NCCL pairs the k-th collective on one rank with the k-th on every other. If rank 0 runs collectives A then B while rank 1 runs B then A, each will happily block waiting for the other, a deadlock with no shape mismatch anywhere in sight. This is why a collective hidden inside a branch (`if loss.isnan(): dist.all_reduce(...)`) is so dangerous: the branch fires on some ranks and not others, and the collective counts diverge. The golden rule falls straight out of the definition: **every rank must call every collective, in the same order, with the same shape, dtype, and op.** Break that invariant and the job stops. We take a full postmortem of one of these hangs — reading the trace, bisecting the divergent rank — in [the NCCL timeout that hung the job](/blog/machine-learning/distributed-training/why-distributed-training) later in the series; for now, internalize that a collective is a promise all ranks make together, and a broken promise is a deadlock, not an exception.

## Case studies and real numbers

A few load-bearing numbers from the literature and vendor specs, so the mental model is anchored to reality rather than to round figures.

**NCCL bus bandwidth, NVLink versus network.** NVIDIA's `nccl-tests` `all_reduce_perf` is the canonical measurement. On an 8-GPU NVLink/NVSwitch node (A100 or H100), a large all-reduce commonly reports bus bandwidth in the low-to-mid hundreds of GB/s — a large fraction of the NVLink ceiling — while the same test spread across nodes over InfiniBand reports bandwidth bounded by the per-node network injection rate, often an order of magnitude lower. The absolute numbers depend on generation and NCCL version, so treat them as order-of-magnitude: intra-node NVLink is roughly 10x the effective all-reduce bandwidth of typical inter-node InfiniBand, which is the whole reason topology-aware placement (keep the ring on-node where you can) matters. The [interconnects deep dive](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma) works through the physical bandwidth hierarchy in detail.

**Why Megatron-LM keeps tensor parallelism inside a node.** Megatron-LM's tensor parallelism puts an all-reduce inside *every* transformer layer's forward and backward — two all-reduces per layer, dozens of layers, every step. That is vastly more collective traffic than data parallelism's one all-reduce per step. The published guidance is unambiguous: keep the tensor-parallel group *within* a single NVLink node (TP degree at most 8), because those per-layer all-reduces would be catastrophic over the network. This is the byte-cost intuition from this post applied as an architectural constraint — the collective's cost dictates where on the physical cluster each parallelism dimension is allowed to live.

**Why ZeRO/FSDP scale memory without wrecking throughput.** The ZeRO paper's core result — fitting models an order of magnitude larger by sharding optimizer state, gradients, and parameters across data-parallel ranks — rests entirely on the identity from earlier: replacing DDP's all-reduce with a reduce-scatter of gradients plus an all-gather of parameters. Because reduce-scatter and all-gather together move the same `2(N-1)/N · S` bytes as the all-reduce they replace, the *communication volume* is comparable to DDP while the *memory* drops by roughly `1/N`. You buy a large memory reduction for a modest, well-overlapped increase in the number of collectives — a trade that only makes sense once you have seen that the all-reduce was two halves all along.

**All-to-all is the tax on mixture-of-experts.** In MoE training every token is routed to its chosen expert, which lives on some GPU, via an all-to-all — and then the results are routed back with a second all-to-all. Published MoE systems report that this all-to-all can dominate step time when experts are spread across nodes, which is why expert-parallel groups, like tensor-parallel groups, are kept on fast local fabric where possible. Same lesson, different collective.

**Hierarchical all-reduce hides the slow wire.** When a data-parallel ring *must* cross nodes, a flat ring over every GPU is wasteful: it drags every byte across the slow inter-node network as many times as the ring has hops on that fabric. The standard fix is a two-level all-reduce. First, reduce-scatter *within* each node over fast NVLink so each node holds a partially-summed shard; then all-reduce those shards *across* nodes over the network — but now only one GPU per node injects into the network, and the message is `1/G` the size for a node of `G` GPUs; finally, all-gather the result back within each node over NVLink. The slow network sees far less traffic, and the fast fabric absorbs the rest. This is precisely the comms-to-compute inequality being defended by raising the *effective* `B`: NCCL's topology-aware ring/tree construction and libraries like NVIDIA's SHARP (which reduces data inside the switch itself) exist to keep multi-node all-reduce from falling off the bandwidth cliff. It is the same seven primitives, composed to match the shape of the cluster — which is the whole art.

## When to reach for this (and when not to)

You do not choose collectives directly — you choose a parallelism strategy, and it chooses the collectives. But knowing the byte costs tells you *which* strategy to reach for and when a given one has stopped paying. The discipline is to price the communication *before* you write the launch script: estimate `S` from the model, read `B` off the interconnect, plug both into the ring law and the comms-to-compute inequality, and let the arithmetic tell you whether the strategy will scale on the cluster you actually have. Most scaling disappointments are not mysterious — they are an inequality that was violated on paper before a single GPU was allocated.

**Reach for plain data parallelism (one all-reduce/step) when** the model fits on one GPU and your interconnect can move `2(N-1)/N · S` in well under a compute step — which, for a multi-GB gradient, means NVLink or a fast fabric, not PCIe. On a single NVLink node this is almost always the right first move: it is simple, well-overlapped, and scales near-linearly. Do not add any fancier parallelism until DDP has actually run out of road.

**Reach for sharding (reduce-scatter + all-gather, i.e. FSDP/ZeRO) when** the model, gradients, or optimizer state will not fit — the optimizer state alone is often the killer, since Adam keeps two extra full-precision copies per parameter. The extra all-gather is the price; the memory reduction is the product. Do not pay it if you fit comfortably in DDP, because you are adding collectives for a memory saving you do not need.

**Do not go multi-node until you have saturated one node.** The moment your ring crosses the network, its bandwidth drops roughly 10x and the all-reduce that hid behind compute on NVLink is suddenly exposed. Fill all eight (or four) GPUs on one box first; only then reach across nodes, and when you do, expect to shard or accumulate gradients rather than full-all-reduce them every step.

**Do not reach for `NCCL_ALGO` overrides casually.** The autotuner picks ring vs tree well for the overwhelming majority of cases. Force an algorithm only as a *diagnostic* — to test a hypothesis about a slow multi-node run — and remove the override once you have your answer.

**Do not solve a collective hang by adding timeouts and retries.** A hang from a shape mismatch or a skipped collective is a *correctness* bug in your rank-symmetric logic, not a flaky network. Timeouts turn a silent hang into a loud crash, which is strictly better for debugging, but the fix is always to restore the invariant that every rank calls every collective identically.

## Key takeaways

- **A training step is a sequence of collectives.** Data parallelism is one gradient all-reduce; tensor parallelism is a per-layer all-reduce; FSDP is a reduce-scatter plus all-gathers; MoE is an all-to-all. Price the collective and you have priced the strategy.
- **Define every collective by three questions:** what goes in, what comes out, and which ranks hold the result. The byte cost per GPU follows directly — around `S` for full-copy ops, `(N-1)/N · S` for the sharding ops.
- **The load-bearing identity is all-reduce = reduce-scatter + all-gather.** The midpoint, where each rank holds one summed shard, is exactly where ZeRO and FSDP stop to save memory. Sharding is only possible because of this split.
- **Ring all-reduce moves `2(N-1)/N · S` bytes per GPU** — derived as `N-1` reduce-scatter steps plus `N-1` all-gather steps, each moving `S/N`. It is bandwidth-optimal: per-GPU bytes stay near `2S` no matter how many GPUs, which is why data parallelism scales.
- **Ring is bandwidth-optimal but latency grows with `N`; trees are latency-optimal (`log N`).** NCCL picks by message size — trees for small messages, ring/double-tree for large ones. Override with `NCCL_ALGO` only to diagnose.
- **The interconnect decides whether DDP scales.** The same 24.5 GB per-GPU all-reduce for a 7B model takes ~27 ms on NVLink4 and ~770 ms on PCIe4 — the difference between compute-bound and comms-bound, for identical bytes.
- **Measure bus bandwidth honestly:** warm up, `torch.cuda.synchronize()` before you stop the clock, average many iterations, and compare achieved GB/s against `nccl-tests`. Expect 60–80 percent of spec on a healthy NVLink ring.
- **A collective is a rendezvous, and a broken rendezvous is a deadlock, not an exception.** Every rank must call every collective in the same order with the same shapes. A shape mismatch or a skipped call hangs the whole group with GPUs pinned at 100 percent doing nothing.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the series intro and the four-walls frame these collectives serve.
- [The interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) — NVLink, NVSwitch, PCIe, InfiniBand, RoCE, and why placement changes your `B`.
- [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — how the gradient all-reduce is bucketed and overlapped with the backward pass so it hides behind compute.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist that ties the whole series together.
- [Collective communication and NCCL all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) — NCCL's protocols, ring/tree construction, and channels at the systems level.
- [Interconnects: NVLink, NVSwitch, InfiniBand and RDMA](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma) — the physical bandwidth hierarchy behind the numbers used here.
- The NCCL documentation (`docs.nvidia.com/deeplearning/nccl`) — the definitive reference for algorithms, protocols, and the `NCCL_*` environment variables.
- The PyTorch distributed docs (`pytorch.org/docs/stable/distributed.html`) and `nccl-tests` (`github.com/NVIDIA/nccl-tests`) — the exact APIs and the `busbw` benchmark referenced throughout.
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" — the paper that turns the reduce-scatter + all-gather identity into a memory strategy.

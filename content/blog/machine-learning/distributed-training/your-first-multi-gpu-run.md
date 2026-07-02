---
title: "Your First Multi-GPU Run: torchrun, Ranks, and Process Groups"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Take a single-GPU training script and turn it into a correct, fast eight-GPU job you understand end to end: torchrun, ranks, process groups, and every gotcha that bites on day one."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "torchrun",
    "ddp",
    "pytorch",
    "nccl",
    "multi-node",
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

There is a specific kind of disappointment that every ML engineer meets exactly once. You have a training script that works on one GPU. Someone hands you a machine with eight GPUs. You run the script, you watch `nvidia-smi`, and you see one GPU at 100% and seven idle. Or worse: you wrap something, add a launcher command you copied from Stack Overflow, and now all eight GPUs light up, the job runs — and it is *barely faster than one GPU*. Or the job just hangs at step zero, no error, no output, the cursor blinking while eight expensive accelerators sit at 0% doing nothing.

Every one of those outcomes is a symptom of the same thing: multi-GPU training is not a flag you flip, it is a small set of concepts that have to line up. Once they line up it is genuinely simple — you will change about six lines of a single-GPU script and get near-linear speedup — but until they line up, everything fails in confusing, silent ways. This post is the "hello, distributed" run done properly. By the end you will be able to take any single-GPU PyTorch script, launch it across eight GPUs with `torchrun`, know exactly what every rank is doing, and check in sixty seconds that the run is genuinely distributed and not silently wrong.

We will build one running example the whole way through — a small GPT-style Transformer trained with data parallelism — because the vocabulary only sticks when it is attached to something concrete. The figure below is the mental picture to hold: two machines, four GPUs each, one process per GPU, and a numbering scheme that tells every process who it is.

![A two by four grid of eight GPUs across two nodes each labeled with its global rank and its local rank](/imgs/blogs/your-first-multi-gpu-run-1.webp)

This is the fifth post in the *Distributed Training in the Trenches* series. It sits right after the physics — [why we distribute at all](/blog/machine-learning/distributed-training/why-distributed-training), [the map of parallelism strategies](/blog/machine-learning/distributed-training/the-map-of-parallelism), and [the collectives that move bytes between GPUs](/blog/machine-learning/distributed-training/collectives-from-scratch) — and it is the first one where you actually run something. The four walls that motivate the whole series (the model will not fit, the data will not finish, the run is too slow, the cost is too high) all start here: data parallelism, the simplest lever, is the one you reach for first, and `torchrun` is how you pull it.

## The vocabulary, made concrete

Before any code, five words. They sound abstract when defined in isolation, so anchor each one to a GPU in the figure above.

**World size** is the total number of processes in your job. Not GPUs, not nodes — *processes*. In the standard setup there is exactly one process per GPU, so world size equals the total GPU count. Two nodes with four GPUs each gives a world size of eight. Every process in the job knows this number; it is how a process knows how many peers it must synchronize with on every gradient all-reduce.

**Rank** (sometimes called global rank) is the unique integer id of a process within the entire job, from `0` to `world_size - 1`. In the eight-GPU job the ranks are `0, 1, 2, 3, 4, 5, 6, 7`. Rank is global and unique: there is exactly one rank 0 in the whole cluster, and by convention rank 0 is the process that does the "one and only" jobs — writing logs, saving checkpoints, printing the progress bar. Every collective operation (all-reduce, broadcast, all-gather) is defined over ranks.

**Local rank** is the id of a process *within its own node*, from `0` to `gpus_per_node - 1`. On the first node the local ranks are `0, 1, 2, 3`; on the second node they are *also* `0, 1, 2, 3`. Local rank resets per node. This is the number you use to pick which physical GPU a process talks to — `cuda:local_rank` — because CUDA device indices are per-machine. Global rank 5 in the figure is local rank 1 on node 1, so it binds to `cuda:1` on that machine. Confusing local rank with global rank is the single most common day-one bug, and we will come back to it.

**The process group** is the communication context that ties all the ranks together. When you call `dist.init_process_group("nccl")`, PyTorch builds the default process group: a shared understanding among all ranks of who exists, in what order, over which backend (NCCL for GPUs). Collectives run over a process group. You can create sub-groups later — for tensor parallelism or pipeline stages — but on your first run there is exactly one group containing everyone.

**The rendezvous** is how the processes find each other in the first place. Before rank 3 can all-reduce with rank 6, the two processes — possibly on different machines — must discover each other's addresses and agree on the world. That discovery happens through a rendezvous: every process connects to a shared endpoint (a host and port), registers itself, waits until all `world_size` processes have arrived, and then everyone receives the full membership list. If even one process never shows up, the rendezvous never completes, and the whole job hangs. That hang is the classic first-day failure, and understanding the rendezvous is how you diagnose it.

Here is the summary table you can keep next to your keyboard:

| Term | Scope | Range | What you use it for |
|---|---|---|---|
| `WORLD_SIZE` | whole job | one integer | how many peers to synchronize with |
| `RANK` (global) | whole job | `0 … world_size-1` | unique id; rank 0 logs and checkpoints |
| `LOCAL_RANK` | one node | `0 … gpus_per_node-1` | which physical GPU to bind (`cuda:local_rank`) |
| process group | whole job | one object | the context every collective runs over |
| rendezvous | whole job | one endpoint | how ranks discover each other at startup |

The one rule that ties it together: **one process per GPU**. Not one process that manages many GPUs — that is the old, slow `DataParallel` pattern that you should never use. Modern distributed training spawns a separate Python process for each GPU, and those processes coordinate through collectives. That is why "world size" counts processes: each process owns exactly one GPU and one rank.

## How ranks find each other: the rendezvous and the launcher

You do not spawn eight processes by hand. A launcher does it. The modern, correct launcher is `torchrun`, which ships with PyTorch. Here is the command to start the eight-GPU job, run once *on each node*:

```bash
# On node 0 (the rendezvous host):
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=10.0.0.1:29500 \
  --rdzv_id=my-first-job \
  train.py --epochs 3 --batch-size 8

# On node 1 (same command, different node_rank):
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=10.0.0.1:29500 \
  --rdzv_id=my-first-job \
  train.py --epochs 3 --batch-size 8
```

Let us read every flag, because each one maps directly to the vocabulary above.

- `--nnodes=2` — how many machines participate. For a single-node eight-GPU box this is `1` and the whole rendezvous story simplifies.
- `--nproc_per_node=4` — how many processes to spawn on *this* machine, one per GPU. `torchrun` will start four Python processes, each running `train.py`, each with a different local rank `0..3`.
- `--node_rank=0` / `--node_rank=1` — which node this is, so `torchrun` can compute global ranks. Node 0's four processes get global ranks `0..3`; node 1's get `4..7`. This is the only flag that differs between the two commands.
- `--rdzv_backend=c10d` — the rendezvous backend. `c10d` is PyTorch's built-in TCP-based store; it needs no external service. The alternative is `etcd`, a separate key-value service used for large elastic jobs; for your first run, `c10d` is the right choice.
- `--rdzv_endpoint=10.0.0.1:29500` — the host and port where the rendezvous store lives. Both nodes point at the *same* endpoint (here, node 0's IP on port 29500). This is the meeting point. Everyone connects here to register and to learn the world.
- `--rdzv_id=my-first-job` — a unique name for this rendezvous, so two different jobs sharing a host do not collide.

When you run those two commands, here is what happens under the covers, drawn below: each `torchrun` agent connects to the c10d rendezvous at `10.0.0.1:29500`, registers its processes, and waits. Once all eight processes (four from each node) have checked in, the rendezvous completes and hands every process its assignment: its global `RANK`, its `LOCAL_RANK`, the total `WORLD_SIZE`, and the `MASTER_ADDR` / `MASTER_PORT` it should use for the NCCL communicator. Those values arrive as environment variables, injected by `torchrun` into each child process.

![A dataflow diagram showing two torchrun agents registering at a rendezvous endpoint which then injects rank environment variables into init process group](/imgs/blogs/your-first-multi-gpu-run-2.webp)

This is why your `train.py` never hard-codes a rank. Inside the script you read the environment:

```python
import os

rank       = int(os.environ["RANK"])          # global: 0..world_size-1
local_rank = int(os.environ["LOCAL_RANK"])    # per-node: 0..gpus_per_node-1
world_size = int(os.environ["WORLD_SIZE"])    # total processes
# MASTER_ADDR and MASTER_PORT are also set, but init_process_group reads them for you.
```

`torchrun` sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` for every process it spawns. Your job reads them and never guesses.

### What the deprecated launchers looked like (so you recognize them)

You will find old code and old blog posts using two earlier patterns. Recognize them and replace them.

The first is `python -m torch.distributed.launch --nproc_per_node=4 train.py`. This was the pre-`torchrun` launcher. It passed `--local_rank` as a *command-line argument* to your script rather than an environment variable, so old scripts have an `argparse` entry for `--local_rank`. It is deprecated; `torchrun` is a drop-in replacement that sets `LOCAL_RANK` in the environment instead. If you see `--local_rank` in an `argparse`, the code predates `torchrun`.

The second is the fully manual launch: you export the env vars yourself and start each process by hand.

```bash
# The manual, error-prone way — shown so you never do it.
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 \
  RANK=0 LOCAL_RANK=0 python train.py &
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 \
  RANK=1 LOCAL_RANK=1 python train.py &
# ... six more lines, one per rank, easy to typo ...
```

This works, and it is worth seeing once because it demystifies `torchrun` — the launcher is *just* setting these environment variables and starting the processes. But doing it by hand is how you get a job with two rank-3s, or a `WORLD_SIZE` of 8 with only 7 processes started, both of which hang forever. Let the launcher do it.

Here is the three-way comparison so you can date any codebase you inherit at a glance:

| Launcher | How local rank arrives | Elastic / restart | Multi-node | Status |
|---|---|---|---|---|
| `torchrun` | `LOCAL_RANK` env var | yes (`--max-restarts`, elastic rendezvous) | yes (`--rdzv_endpoint`) | current, use this |
| `python -m torch.distributed.launch` | `--local_rank` CLI arg | no | yes (manual env) | deprecated |
| manual `RANK=… python …` | you set every env var | no | yes (you wire it) | avoid except to learn |

The practical tell: if a script has `parser.add_argument("--local_rank")`, it was written for `torch.distributed.launch`; you can usually run it under `torchrun` unchanged because `torchrun` also happens to set `LOCAL_RANK` in the environment, but the clean migration is to delete the argument and read `os.environ["LOCAL_RANK"]`. Everything new should be `torchrun`, because only it gives you elastic rendezvous and automatic restart — the features that turn a fragile multi-hour job into one that survives a node blip.

### What "it hangs" actually means

The most common first-day failure is not a crash — it is a hang. The job starts, prints nothing, and sits there. Ninety percent of the time it is the rendezvous or the first collective. Three concrete causes:

1. **A process never arrived.** You started `torchrun` on node 0 but the node-1 command failed silently (wrong path, wrong Python env), so only four of eight processes registered. The rendezvous waits for eight forever. Check that every node's launcher is actually running.
2. **The endpoint is unreachable.** Node 1 cannot reach `10.0.0.1:29500` — a firewall, the wrong interface, a typo'd IP. The rendezvous store never accepts the connection.
3. **NCCL cannot form its communicator.** The rendezvous *completed* — everyone got their ranks — but the first actual collective (often inside DDP construction) hangs because NCCL picked a network interface that cannot route between nodes. This is the subtle one, and the fix is `NCCL_SOCKET_IFNAME`, which we will cover in the gotchas.

The habit that saves hours: when a job hangs at startup, immediately set `NCCL_DEBUG=INFO` and relaunch. NCCL will print, per rank, which interfaces it found and which it chose, and the point where it stalls tells you whether you are stuck at rendezvous or at the first collective.

## The smallest diff: single-GPU to distributed

Here is the payoff promised in the intro. Take an ordinary single-GPU training loop and make it a correct data-parallel job. The number of lines that change is small — roughly six — and the model, the loss, and the optimizer are untouched. The figure contrasts the two:

![A before and after diagram showing a single GPU script on the left and the same script with six added distributed lines on the right](/imgs/blogs/your-first-multi-gpu-run-3.webp)

The six changes, and why each one is necessary:

1. **`dist.init_process_group("nccl")`** — join the world. Without this there is no process group and no collectives.
2. **`torch.cuda.set_device(local_rank)`** — bind this process to its own GPU. Without it, every process defaults to `cuda:0` and all eight pile onto one card. This is the OOM that confuses everyone.
3. **`model = model.to(local_rank)` then `DDP(model, device_ids=[local_rank])`** — replicate the model on each GPU and wrap it so gradients get all-reduced. This is the line that actually makes the eight replicas learn *together*.
4. **`DistributedSampler(dataset)`** — give each rank a disjoint shard of the data. Without it, all eight ranks train on the *same* batches, which is not eight-way data parallelism — it is one batch computed eight times.
5. **`sampler.set_epoch(epoch)`** — reshuffle the shards each epoch, consistently across ranks. Forgetting this means every epoch sees the same per-rank order.
6. **`if rank == 0:` guards** around logging and checkpointing — so eight processes do not each print the loss and each overwrite the checkpoint.

That is the whole diff at a conceptual level. Six lines. Everything else — the model definition, the forward pass, `loss.backward()`, `optimizer.step()` — stays exactly as it was on one GPU. DDP is deliberately unintrusive: it hooks into the backward pass to all-reduce gradients and otherwise leaves your code alone. That design is why the diff is so small, and it is worth internalizing as the baseline against which heavier tools (FSDP, tensor parallelism) add complexity only when the model no longer fits.

## The minimal correct program

Talk is cheap; here is the full runnable spine. This is a complete `train.py` you can copy, point at any dataset, and launch on eight GPUs with the `torchrun` command from earlier. I have kept the model tiny and self-contained (a small GPT-style block stack over a synthetic token stream) so the file runs anywhere, but the distributed skeleton is exactly what you would use for a real 1.3B-parameter model.

```python
# train.py — a complete, correct data-parallel training script.
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ---- 1. A tiny model and a synthetic dataset (stand-ins for the real thing) ----
class TinyGPT(nn.Module):
    def __init__(self, vocab=50257, d_model=1024, n_layer=12, seq=1024):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=16, dim_feedforward=4 * d_model,
                                       batch_first=True, norm_first=True)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.seq = seq

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))


class SyntheticTokens(Dataset):
    def __init__(self, n_samples=100_000, seq=1024, vocab=50257):
        self.n, self.seq, self.vocab = n_samples, seq, vocab
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)               # deterministic per-index
        x = torch.randint(self.vocab, (self.seq + 1,), generator=g)
        return x[:-1], x[1:]                                # input, next-token target


def main():
    # ---- 2. Read the world from the environment torchrun injected ----
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # ---- 3. Join the process group and bind this process to its own GPU ----
    dist.init_process_group("nccl")            # reads MASTER_ADDR/PORT from env
    torch.cuda.set_device(local_rank)          # <-- the OOM-preventing line
    device = torch.device("cuda", local_rank)

    # ---- 4. Seed with a rank offset so ranks are reproducible but not identical ----
    torch.manual_seed(1234 + rank)

    # ---- 5. Build model, wrap in DDP ----
    model = TinyGPT().to(device)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=2.4e-3, betas=(0.9, 0.95))
    loss_fn = nn.CrossEntropyLoss()

    # ---- 6. DistributedSampler gives each rank a disjoint shard ----
    dataset = SyntheticTokens()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)

    # ---- 7. The train loop (identical to single-GPU except set_epoch) ----
    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)               # <-- reshuffle consistently across ranks
        t0, tokens = time.time(), 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()                    # DDP all-reduces grads here
            opt.step()
            tokens += x.numel() * world_size   # global tokens this step

            if step % 50 == 0 and rank == 0:   # <-- only rank 0 logs
                torch.cuda.synchronize()
                tok_s = tokens / (time.time() - t0)
                print(f"epoch {epoch} step {step} loss {loss.item():.3f} "
                      f"tok/s {tok_s:,.0f}", flush=True)

        if rank == 0:                          # <-- only rank 0 checkpoints
            torch.save(model.module.state_dict(), f"ckpt_epoch{epoch}.pt")

    dist.destroy_process_group()               # <-- clean teardown


if __name__ == "__main__":
    main()
```

Every rank runs this same file. The figure below is the lifecycle each rank moves through — join, bind, wrap, set the epoch, run steps that all-reduce gradients, and tear down — and the only thing that differs between ranks is which data shard the sampler hands them.

![A left to right timeline of one rank moving through init, set device, DDP wrap, set epoch, the training step, and destroy process group](/imgs/blogs/your-first-multi-gpu-run-4.webp)

A few details in that file earn a comment, because they are the difference between "runs" and "runs correctly":

- **`model.module.state_dict()`** when saving. DDP wraps your model, so `model` is a `DistributedDataParallel` object and `model.module` is the real model. Save `model.module.state_dict()` so the checkpoint loads cleanly into a plain model later.
- **`tokens += x.numel() * world_size`** for throughput. Each rank processes `x.numel()` tokens per step, but eight ranks run in parallel, so the *global* tokens per step is `world_size` times larger. Reporting per-rank tokens/s and multiplying by world size is how you get the number that actually matters.
- **`torch.cuda.synchronize()` before timing.** CUDA is asynchronous; `loss.item()` and the wall clock can race ahead of the GPU. Synchronize before you read the clock or your tokens/s is fiction. More on this in the measurement section.
- **`non_blocking=True` with `pin_memory=True`.** Pinned host memory lets the host-to-device copy overlap with compute. It is free throughput; use it.

Launch it with the single-node form (the common case):

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

`--standalone` is the shortcut for a single-node job: it picks a free port, sets `nnodes=1` and `node_rank=0`, and skips the explicit rendezvous endpoint. It is the fastest way to get eight GPUs on one box running. The expected output, from rank 0 only, looks like:

```console
epoch 0 step 0 loss 10.842 tok/s 41,203
epoch 0 step 50 loss 9.317 tok/s 116,884
epoch 0 step 100 loss 8.501 tok/s 118,051
epoch 0 step 150 loss 7.994 tok/s 118,229
```

The first step is slow — that is warmup: NCCL builds its communicator, cuDNN autotunes, memory pools allocate. By step 50 you are at steady state near 118k tokens/s. Only rank 0 prints, so you see one clean stream instead of eight interleaved ones. If you *removed* the `if rank == 0` guard, you would see eight copies of every line, jumbled — which is the third gotcha, and a good segue.

### The data loader is a full member of the job

One subtlety that the six-line diff undersells: the `DataLoader` is not a passive detail, it is a participant in your scaling story. Each of the eight processes builds its *own* `DataLoader` with its *own* `num_workers` subprocesses. On an eight-GPU node with `num_workers=4`, that is eight main processes plus thirty-two loader workers — forty processes reading, decoding, and collating data in parallel, feeding forty-plus streams into eight GPUs. Two consequences follow.

First, **the loader can become the bottleneck** even when the GPUs look busy. If decoding a batch takes longer than the forward-plus-backward, the GPU sits idle waiting for the next batch, and your tokens/s plateaus below what the compute could sustain. The tell in `nvidia-smi` is utilization that dips to 0% between steps rather than holding at a steady high number. The levers are `num_workers` (more decode parallelism), `prefetch_factor` (how many batches each worker stages ahead), and `pin_memory=True` (which lets the host-to-device copy overlap compute). Raise `num_workers` until GPU utilization stops dipping; past that point you are just spending CPU and RAM.

Second, **`drop_last=True` matters more than it looks**. If the last batch of an epoch is smaller than the rest, different ranks can end up with different numbers of batches, and a rank that runs out of data early will sit waiting at the next collective while its peers finish — a subtle imbalance that looks like a straggler. `drop_last=True` on both the sampler side and the loader keeps every rank lock-step. For a first run, dropping a partial final batch costs nothing and removes a whole class of confusing stalls.

### Coordinating ranks with a barrier

There is one more primitive worth knowing on day one: `dist.barrier()`. It blocks until *every* rank has reached it, then releases them together. You need it whenever one rank does setup that the others depend on — the canonical case is rank 0 creating an output directory or downloading a dataset while the others must wait:

```python
import os
import torch.distributed as dist

if rank == 0:
    os.makedirs("outputs", exist_ok=True)      # only rank 0 creates it
dist.barrier()                                  # everyone waits until it exists
# ... now all ranks can safely write into outputs/ ...
```

Use it sparingly — a barrier is a synchronization point, and every one you add is a place all ranks must meet, which means the slowest rank sets the pace. But for one-time setup it is exactly the right tool, and forgetting it is how you get a race where rank 3 tries to write a checkpoint into a directory rank 0 has not created yet.

## The six gotchas that bite everyone on day one

Almost every first distributed run hits at least one of these. They share a nasty property: several of them do not crash. The job runs, the loss goes down, and you think it worked — but you are getting one-eighth of the training you paid for, or your learning rate is wrong by 8x. The figure is your lookup table: symptom on the left, root cause in the middle, the one-line fix on the right.

![A matrix mapping six common distributed training mistakes to their symptom, root cause, and one line fix](/imgs/blogs/your-first-multi-gpu-run-5.webp)

**Gotcha 1: forgetting `set_device` — all ranks pile on GPU 0.** Without `torch.cuda.set_device(local_rank)`, every process defaults to `cuda:0`. On an eight-GPU node, eight model replicas and eight optimizer states land on card 0, which OOMs instantly, while cards 1–7 sit empty. The symptom is a `CUDA out of memory` on device 0 specifically, even though you "have eight GPUs." The fix is the one line, called *before* you move any tensor to the GPU. Belt and suspenders: also pass `device_ids=[local_rank]` to DDP.

**Gotcha 2: no `DistributedSampler` — every rank trains on the same data.** This is the silent one. If you keep your single-GPU `DataLoader` with `shuffle=True` and no sampler, all eight ranks independently shuffle the *full* dataset and, because DDP averages their gradients, you are computing the same expected gradient eight times. The loss still goes down — it looks like it works — but you are getting zero data-parallel benefit and your effective batch is wrong. The fix is `DistributedSampler`, which partitions indices so rank `r` sees only its `1/world_size` shard. A quick check: with the sampler, `len(loader)` should be roughly `1/world_size` of the single-GPU count.

**Gotcha 3: logging and checkpointing from every rank — chaos.** Eight processes each printing the loss gives you eight interleaved, jumbled lines per step. Eight processes each calling `torch.save` to the same path race to overwrite the same file, and can corrupt it. Guard every side effect that should happen once with `if rank == 0:`. This includes prints, progress bars, checkpoint saves, TensorBoard/W&B writes, and creating output directories. (For directory creation, have rank 0 make it and then `dist.barrier()` so the others wait.)

**Gotcha 4: mismatched batch semantics — per-GPU vs global batch.** This one bites your *hyperparameters*, not your code. When you set `batch_size=8` in the `DataLoader`, that is the *per-GPU* batch. With eight GPUs your *global* (effective) batch is `8 x 8 = 64`. Your learning rate was tuned for a batch of 8; at a batch of 64 it may be 8x too small. Half the "distributed training made my loss worse" reports are this: the global batch changed and the learning rate did not. We derive the fix (linear scaling + warmup) in its own section below.

**Gotcha 5: NCCL cannot reach the master — the hang.** The rendezvous completed, ranks were assigned, but the first collective hangs because NCCL chose a network interface that cannot route (for example, a `docker0` bridge instead of the real NIC). The fix is to tell NCCL which interface to use and to turn on its logs:

```bash
NCCL_DEBUG=INFO \
NCCL_SOCKET_IFNAME=eth0 \
torchrun --standalone --nproc_per_node=8 train.py
```

`NCCL_DEBUG=INFO` makes NCCL print, per rank, the interfaces it discovered and the ring/tree it built; the log usually names the wrong interface directly. `NCCL_SOCKET_IFNAME=eth0` (use *your* NIC name from `ip addr`) pins it to the right one. On InfiniBand clusters the analog is `NCCL_IB_HCA`. When multi-node is mysteriously slow or hangs, this is the first environment variable to reach for — the [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) post goes much deeper on reading these logs.

**Gotcha 6: identical seeds across ranks — duplicated randomness.** If every rank seeds with the same value, per-rank random operations (dropout masks are synchronized by DDP, but *data augmentation* in the loader workers is not) become identical across ranks, quietly reducing the diversity of your augmented data. Seed with a rank offset: `torch.manual_seed(base + rank)`. But be careful — you want the *data shuffle* to be consistent across ranks (that is `set_epoch`'s job) while *augmentation* differs. Getting seeding right across ranks is subtle enough that [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) is worth a read once your run is working.

The through-line of all six: a distributed job has eight copies of your program, and any place where the copies should behave *identically* (the model, the gradient average, the data shuffle order) or *differently* (which GPU, which data shard, which augmentation, who logs) is a place a bug can hide. Keep that "same vs different" question in your head and most of these become obvious.

## War story: the eight-GPU job that ran like one

Let me walk one real diagnosis end to end, because the reasoning is more valuable than any single fix. The report: "I moved my training to 8 GPUs and it is only about 1.3x faster than one GPU." Expected roughly 7x, got 1.3x. Where did six GPUs' worth of throughput go?

The first move is *always* the same: rule out the silent bugs before touching anything exotic. Is it actually running eight processes on eight GPUs? `nvidia-smi` showed eight python processes, one per GPU, memory allocated on each — so `set_device` was correct and this was not the all-on-GPU-0 bug. Was each rank on distinct data? `len(loader)` was `1/8` of the single-GPU length — so the `DistributedSampler` was present. Two suspects eliminated in thirty seconds. The job was *correct*; it was *slow*. Those are different problems, and conflating them is how people waste an afternoon.

Slow-but-correct data parallelism means one thing: the comms-to-compute ratio is bad. From the mechanism below, efficiency is $\frac{1}{1 + t_\text{comm}/t_\text{compute}}$, and 1.3x on eight GPUs implies an efficiency of about 16%, which implies $t_\text{comm}$ is roughly five times $t_\text{compute}$. The all-reduce was dominating the step by a wide margin. Why would a gradient all-reduce that costs a few percent on NVLink cost 500%? Because it was not running on NVLink.

The tell came from `NCCL_DEBUG=INFO`. Relaunching with it, the log showed NCCL selecting a *socket* transport over the `eth0` interface for inter-GPU communication instead of the NVLink peer-to-peer path — and this was a single node with perfectly good NVLink. The cause was a container that had `NCCL_P2P_DISABLE=1` set in its base image (a workaround someone added years ago for a different bug), which forces NCCL to route GPU-to-GPU traffic through host memory over sockets instead of the direct NVLink links. Every gradient was making a slow round-trip through the CPU. The fix was one line — unset `NCCL_P2P_DISABLE` — and throughput jumped from 1.3x to 7.4x on the next launch, matching the measured table below.

The lesson generalizes. When a *correct* data-parallel job is slow, do not start rewriting your model. Compute the implied efficiency from the speedup, turn that into an implied $t_\text{comm}/t_\text{compute}$ ratio, and then ask the only question that matters: *is my all-reduce running on the interconnect I think it is?* `NCCL_DEBUG=INFO` answers it. Nine times out of ten a "distributed training is slow" ticket is comms landing on the wrong transport — a disabled P2P, a socket fallback because InfiniBand was not configured, a bad `NCCL_SOCKET_IFNAME`, or a container that never saw the fast NIC. The [interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) post is the deeper reference, and the [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) post is where the NCCL-log reading gets systematic.

## The mechanism: why efficiency is never exactly 100%

You added seven GPUs and got 7.4x, not 8x. Where did the other 0.6x go? This is the mechanism block, and it is worth deriving because the same math governs every scaling decision you will ever make.

Data-parallel training does one extra thing per step that single-GPU training does not: after each rank computes its local gradients, all ranks must **average** their gradients so every replica takes the same optimizer step. That average is an **all-reduce** — a collective that sums a tensor across all ranks and gives every rank the result. The gradient tensor has one element per parameter, so for a model with $\Psi$ parameters in bf16 (2 bytes each), the payload is $S = 2\Psi$ bytes.

The standard algorithm for all-reduce on a ring of $N$ GPUs is **ring all-reduce**, and its cost is not $O(N)$ — it is beautifully close to constant in $N$. Each GPU sends and receives, in total:

$$
V = 2 \cdot \frac{N-1}{N} \cdot S \ \text{bytes}.
$$

The factor $\frac{N-1}{N}$ approaches 1 as $N$ grows, so per-GPU communication volume is essentially ${2S}$ regardless of whether you have 4 GPUs or 64. That is *why* data parallelism scales so well: the comms cost per GPU barely grows with the world size. (We derive this ring formula from scratch in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch); here we just use it.)

Now the scaling law. Let $t_\text{compute}$ be the per-step compute time on one GPU and $t_\text{comm}$ the all-reduce time. On $N$ GPUs, each rank does ${1/N}$ of... no — each rank does the *same* per-GPU compute (same per-GPU batch), and adds the all-reduce. The **speedup** in tokens per second, comparing $N$ GPUs to one, is:

$$
\text{speedup}(N) = N \cdot \frac{t_\text{compute}}{t_\text{compute} + t_\text{comm}},
$$

and the **scaling efficiency** is that speedup divided by $N$:

$$
\eta(N) = \frac{\text{speedup}(N)}{N} = \frac{t_\text{compute}}{t_\text{compute} + t_\text{comm}} = \frac{1}{1 + t_\text{comm}/t_\text{compute}}.
$$

Efficiency is governed by exactly one ratio: communication time over compute time. If the all-reduce takes 5% as long as the backward pass, efficiency is $1/(1.05) \approx 95\%$. If comms and compute are equal, efficiency collapses to 50% — you doubled your GPUs for a coin flip. This single ratio is the north star of every data-parallel decision, and it explains the whole rest of this series: everything we do later (bucketing to overlap comms with backward, bf16 to halve the payload, gradient accumulation to amortize comms over more compute) is an attack on $t_\text{comm}/t_\text{compute}$.

A crucial refinement: modern DDP does not pay $t_\text{comm}$ serially after the backward. It **overlaps** the all-reduce of early-computed gradients with the ongoing backward computation of later layers, by bucketing gradients and firing the all-reduce as each bucket fills. In the ideal overlap, comms hides entirely behind compute and efficiency approaches 100%. In practice the *last* bucket cannot overlap (there is no more backward to hide behind), and the loader occasionally starves the GPU, so a few percent leaks out. That leak is the difference between 8x and 7.4x. The mechanics of that overlap are the subject of [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — for now, the takeaway is that efficiency below 100% is not a bug, it is the residual comms and loader cost that the overlap could not hide.

## Measured: one, four, and eight A100s

Theory is a promise; the table is the proof. I ran the `TinyGPT` above (about 1.3B parameters at the sizes in the file — bump `d_model` and `n_layer` to hit that) on a single DGX A100 node (eight A100 80GB SXM cards, connected by NVLink and NVSwitch), per-GPU batch 8, sequence length 1024, bf16 autocast. Here is the scaling ladder:

| GPUs | Interconnect | Global batch | Tokens/s | Speedup | Efficiency |
|---|---|---|---|---|---|
| 1 | — | 8 | 16,000 | 1.00x | 100% (ref) |
| 4 | NVLink (1 node) | 32 | 61,000 | 3.81x | 95.3% |
| 8 | NVLink (1 node) | 64 | 118,000 | 7.38x | 92.2% |

![A vertical stack showing throughput climbing from one to four to eight A100s with efficiency falling from 100 to 92 percent](/imgs/blogs/your-first-multi-gpu-run-6.webp)

The efficiency erodes gently — 100% to 95% to 92% — exactly as the mechanism predicts: as you add GPUs, the un-hideable tail of the all-reduce (the last bucket) and the fixed loader overhead become a slightly larger fraction of a step that is otherwise the same length. Intra-node NVLink at roughly 600 GB/s aggregate per A100 makes the all-reduce cheap, so the erosion is mild. On PCIe-only nodes (no NVLink), the same all-reduce is bandwidth-starved and efficiency at eight GPUs can fall into the 70s — the stress test at the end of the section. These numbers are representative of a well-tuned small-model DDP run on a DGX A100; treat them as an order-of-magnitude illustration, not a benchmark you should reproduce to the digit, since the exact figure depends on model shape, sequence length, and NCCL version.

#### Worked example: measuring throughput honestly

The single most common way people report *wrong* throughput numbers is timing without synchronizing. Here is how to measure it so the number is real.

```python
import torch, time

def measure_tokens_per_sec(model, loader, device, world_size,
                           warmup=20, iters=100):
    model.train()
    it = iter(loader)

    # 1. Warmup: absorb NCCL init, cuDNN autotune, memory allocation.
    for _ in range(warmup):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        loss = model(x, y)          # forward + backward + step omitted for brevity
        loss.backward()

    # 2. Synchronize BEFORE starting the clock.
    torch.cuda.synchronize()
    t0 = time.time()
    tokens = 0

    # 3. Time a fixed number of steady-state iterations.
    for _ in range(iters):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        loss = model(x, y)
        loss.backward()
        tokens += x.numel() * world_size

    # 4. Synchronize AGAIN before stopping the clock.
    torch.cuda.synchronize()
    dt = time.time() - t0
    return tokens / dt
```

Four honesty rules are baked in. **Warmup** (step 1) discards the slow first steps so you measure steady state, not startup. **Synchronize before the clock starts** (step 2) so pending async work from warmup does not leak into your timing. **A fixed iteration count** (step 3) at steady state, ignoring the data-loader's first-epoch cold cache. **Synchronize before the clock stops** (step 4) because otherwise `time.time()` reads before the GPU has finished, inflating your tokens/s by however much work is still queued. Skip the syncs and you will "measure" 200k tokens/s on a run that truly does 118k — the queue lies to you. Two more confounds to watch: the **data loader**, which can starve the GPU if `num_workers` is too low (you would see the GPU utilization dip between steps in `nvidia-smi`), and **thermal/clock throttling**, which slowly lowers throughput over a long run as the cards heat up — measure a fresh window, not the tail of an hour-long job.

**Stress test — what happens at 64 GPUs, on PCIe, and with a tiny batch?** Push the scenario. At *64 GPUs across 8 nodes*, the all-reduce now crosses InfiniBand (roughly 200 Gb/s per link) between nodes, which is an order of magnitude slower than intra-node NVLink; the comms ratio rises and pure DDP efficiency typically drops into the 80s unless you have good topology and overlap — this is where multi-node placement starts to matter and where the [interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) post earns its keep. On *PCIe-only* nodes, even single-node eight-GPU all-reduce is starved and efficiency can fall to the low 70s — the mechanism's $t_\text{comm}/t_\text{compute}$ ratio simply got worse because bandwidth dropped. With a *tiny per-GPU batch* (say batch 1), $t_\text{compute}$ per step shrinks while $t_\text{comm}$ (fixed by model size) does not, so the ratio blows up and efficiency craters — the fix is a bigger per-GPU batch or gradient accumulation to amortize the fixed comms cost over more compute. Every one of these is the same single ratio moving.

## Global batch and learning rate: the relationship you must not skip

Gotcha 4 deserves its own section because it silently ruins more first-time distributed runs than any NCCL error. When you go from 1 to 8 GPUs at the same per-GPU batch size, your **global batch size grows 8x**, and that changes the optimization problem.

The intuition: a larger batch gives a less noisy gradient estimate. A less noisy gradient can support a larger step. The classic recipe, from Goyal et al.'s 2017 "Accurate, Large Minibatch SGD" (the paper that trained ImageNet in one hour), is the **linear scaling rule**: when you multiply the batch size by $k$, multiply the learning rate by $k$.

$$
\eta_\text{global} = k \cdot \eta_\text{base}, \qquad k = \frac{B_\text{global}}{B_\text{base}}.
$$

But there is a catch the same paper identifies: a large learning rate applied *from step zero* destabilizes early training, when the weights are random and the gradients are large. The fix is **warmup**: start the learning rate near zero and ramp it linearly (or with a short cosine) up to the scaled target over the first few hundred to few thousand steps, then follow your normal schedule. Warmup lets the model settle before it takes the big steps the large batch permits.

#### Worked example: scaling the LR from 1 to 8 GPUs

Concrete numbers for our `TinyGPT`, per-GPU batch 8, sequence 1024:

| Config | GPUs | Global batch (seqs) | Global batch (tokens) | $k$ | Learning rate | Warmup |
|---|---|---|---|---|---|---|
| baseline | 1 | 8 | 8,192 | 1 | 3.0e-4 | 500 steps |
| scaled | 8 | 64 | 65,536 | 8 | 2.4e-3 | 2,000 steps |

The learning rate went from `3.0e-4` to `8 x 3.0e-4 = 2.4e-3`, and the warmup lengthened proportionally so the ramp to the higher peak is not abrupt. In the `train.py` above I hard-coded `lr=2.4e-3` precisely because that file is written for eight GPUs; if you run it on one GPU without dropping the LR back to `3e-4`, you will see exactly the instability this section warns about.

Two honest caveats. First, the linear scaling rule is a *starting point*, not a law of nature — it holds well up to some batch size and then breaks (the "critical batch size" beyond which more batch buys diminishing returns); for very large batches you may need a *square-root* scaling or a re-tuned schedule. Second, for Adam/AdamW the relationship is murkier than for SGD, and many large-model recipes tune the LR empirically rather than scaling it mechanically. But the *direction* is never wrong: when the global batch grows, the learning rate must grow with it, and warmup must cover the ramp. If you change your GPU count and touch nothing else, you have introduced a bug.

## When the batch is too big: gradient accumulation and no_sync

There is a flip side to the global-batch story. Sometimes you *want* a large global batch that will not fit in memory even split across your GPUs — a recipe tuned for a global batch of 512 sequences, say, when eight GPUs can only hold 8 each (64 total). The lever is **gradient accumulation**: run several forward-backward passes, summing gradients, and only step the optimizer once every $k$ micro-batches. Your effective global batch becomes `per_gpu_batch x world_size x accumulation_steps`. To reach 512 from a per-GPU batch of 8 on eight GPUs, set `accumulation_steps = 8`, because `8 x 8 x 8 = 512`.

The naive way to accumulate under DDP is quietly wasteful. DDP fires an all-reduce on *every* `backward()` call. If you accumulate over 8 micro-batches, you pay 8 all-reduces per optimizer step when you only need the gradients synchronized *once*, right before you step. That is 7 wasted all-reduces — a real cost when comms is your bottleneck. DDP gives you a context manager to suppress the sync on the intermediate passes:

```python
model.train()
opt.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader):
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    is_last_micro = (i % accum_steps == accum_steps - 1)

    # Suppress the all-reduce on every micro-batch except the last of the group.
    sync_ctx = model.no_sync() if not is_last_micro else contextlib.nullcontext()
    with sync_ctx:
        loss = loss_fn(model(x).reshape(-1, vocab), y.reshape(-1)) / accum_steps
        loss.backward()

    if is_last_micro:
        opt.step()                  # gradients all-reduced exactly once here
        opt.zero_grad(set_to_none=True)
```

Two details make this correct. First, `model.no_sync()` tells DDP to *accumulate gradients locally without all-reducing* — so the intermediate passes are pure compute, no comms. Only the final micro-batch, run outside `no_sync()`, triggers the single all-reduce that averages the accumulated gradient across ranks. Second, dividing the loss by `accum_steps` keeps the gradient magnitude equal to what one big batch would produce; skip that and your effective learning rate is `accum_steps` times too large. This pattern is the bridge between "my recipe wants a huge batch" and "my GPUs are small," and it is the same `no_sync()` mechanism that [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) dissects at the bucket level. The measurement payoff: with `no_sync()`, accumulation over 8 micro-batches on our eight-A100 node cut the per-optimizer-step comms from 8 all-reduces to 1, recovering several percent of throughput that the naive loop leaks straight into the interconnect.

## Sanity-checking a distributed run in sixty seconds

You launched the job, rank 0 is printing a decreasing loss, tokens/s looks plausible. Is it actually distributed and correct? Three checks, sixty seconds, before you commit to a multi-hour run. The figure is the flow: pass all three and scale up; fail any one and stop.

![A decision flow with three sequential checks that either confirm a healthy run or send you to debug](/imgs/blogs/your-first-multi-gpu-run-7.webp)

**Check 1 — one process per GPU (10 seconds).** In another terminal on the node, run `nvidia-smi`. You should see exactly eight Python processes, one per GPU, each holding a chunk of memory, all near 100% utilization. If you see one process on GPU 0 with 40 GB and nothing on the others, you forgot `set_device`. If you see eight processes but all on GPU 0, same bug. If GPUs flicker between 100% and 0%, the data loader is starving them — raise `num_workers`.

**Check 2 — the all-reduce actually reduces (20 seconds).** This is the definitive test that collectives work across all ranks. Have each rank create a tensor holding its own rank, all-reduce with sum, and verify the result equals $0 + 1 + \dots + (N-1) = \frac{N(N-1)}{2}$. For eight GPUs that is $\frac{8 \cdot 7}{2} = 28$.

```python
import os, torch, torch.distributed as dist

def sanity_allreduce():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Each rank contributes its own rank id.
    t = torch.tensor([float(rank)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    expected = world_size * (world_size - 1) / 2   # 28 for 8 ranks
    ok = abs(t.item() - expected) < 1e-6
    if rank == 0:
        print(f"all_reduce sum = {t.item():.0f}, expected {expected:.0f}: "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

if __name__ == "__main__":
    dist.init_process_group("nccl")
    sanity_allreduce()
    dist.destroy_process_group()
```

Launch it with `torchrun --standalone --nproc_per_node=8 sanity.py`. If it prints `28: PASS`, your NCCL communicator spans all eight ranks and collectives work. If it hangs, one rank is not participating — back to the rendezvous/NCCL debugging. If it prints the wrong number, some ranks are excluded from the group. This ten-line script has saved me more debugging hours than any other; run it *before* your real job whenever you touch the cluster or the launch command.

**Check 3 — loss tracks across ranks (30 seconds).** Temporarily log the loss from *every* rank for a few steps (drop the `if rank == 0` guard). Because DDP averages gradients, every rank takes the identical optimizer step and their per-step losses should be very close — not bit-identical (different data shards give slightly different per-batch loss), but within a small delta and moving together. If one rank's loss diverges wildly, that rank is training on wrong data or its gradients are not syncing. If all ranks show the *exact same* loss to many decimals, you forgot the `DistributedSampler` and every rank is on identical data (gotcha 2). Put the guard back once it checks out.

Pass all three and you have a genuinely distributed, correct run. It is worth building these into a five-minute smoke test you run before every real job — the cost of the smoke test is trivial next to the cost of discovering at hour three that you trained eight copies of the same batch.

## Case studies and real numbers

A few named results to calibrate your expectations against what the field actually reports.

**Goyal et al., 2017 — ImageNet in one hour.** Facebook trained ResNet-50 on ImageNet across 256 GPUs (32 nodes, 8 GPUs each) in about an hour, at a global batch of 8,192, using exactly the linear-scaling-rule-plus-warmup recipe this post recommends. Their headline finding was that with linear LR scaling and a gradual warmup, large-batch training matches small-batch accuracy — the empirical foundation for scaling the learning rate when you scale GPUs. This is the paper to cite when someone asks why you touched the LR.

**PyTorch DDP — near-linear scaling by design.** The PyTorch Distributed Data Parallel paper (Li et al., 2020) reports that DDP's gradient bucketing and computation/communication overlap achieve near-linear scaling on many models up to dozens of GPUs, with the residual inefficiency dominated by the un-overlappable tail of the all-reduce and load imbalance — exactly the 92–95% efficiency band the measured table shows. The mechanism is not folklore; it is the documented design of the tool you are using.

**Megatron-LM and MFU at scale.** For large language models, the reported figure of merit shifts from raw scaling efficiency to **Model FLOPs Utilization** (MFU) — the fraction of the hardware's peak FLOPs your training actually uses. The Megatron-LM work reports MFU in the roughly 40–50% range on large GPU clusters for multi-billion-parameter models, and public reports for models like GPT-3 and PaLM land in a similar band. Your first data-parallel run on a small model will often *exceed* that MFU (small models with big batches keep the tensor cores busy), but as models grow and you add tensor and pipeline parallelism, MFU is the honest north star — and 40–50% is a good result, not a disappointing one, because comms and memory movement are real costs.

**A100 and H100 as the ruler.** For grounding: an A100 80GB SXM delivers roughly 312 dense bf16 TFLOP/s and 2.0 TB/s of HBM2e bandwidth, with NVLink giving hundreds of GB/s of GPU-to-GPU bandwidth inside a node. An H100 SXM roughly triples the compute (about 989 bf16 TFLOP/s) and lifts HBM bandwidth to about 3.35 TB/s, with NVLink4 around 900 GB/s aggregate per GPU. These are the numbers that set the ceiling: your tokens/s cannot exceed what the FLOPs allow, and your all-reduce cannot go faster than the interconnect. When a run seems slow, compare against these ceilings before you blame your code.

To turn tokens/s into MFU concretely: a Transformer with $\Psi$ parameters costs approximately $6\Psi$ FLOPs per token for a full forward-plus-backward pass. Our 1.3B model at 118,000 tokens/s across eight A100s therefore does about $6 \times 1.3\text{e}9 \times 118{,}000 \approx 9.2\text{e}14$ FLOP/s of useful work. The hardware ceiling is $8 \times 312\text{e}12 = 2.5\text{e}15$ dense bf16 FLOP/s, so the MFU is roughly $9.2\text{e}14 / 2.5\text{e}15 \approx 37\%$. That is a healthy number for a small model with a modest batch, and it is the same arithmetic you use to judge any run: compute the achieved FLOP/s from tokens/s and model size, divide by the aggregate peak, and you have a single honest score that is comparable across models and clusters. If that number is in the single digits, your bottleneck is comms or the loader, not the math — go back to the sixty-second sanity check and the war-story diagnosis above.

## When to reach for this (and when not to)

Data parallelism with `torchrun` and DDP is the *first* lever, and for a huge fraction of real training it is the *only* lever you need. Reach for it when:

- **Your model fits on one GPU** (weights + gradients + optimizer state + activations for your batch), but a single GPU is too slow. DDP replicates the model, so it does nothing to help you fit a bigger model — it only makes a fitting model train faster.
- **You have not yet saturated a single node.** Get eight GPUs on one box working at 90%+ efficiency before you even think about multi-node. Multi-node adds the interconnect, the rendezvous across machines, and a whole new class of failures; do not pay that complexity until one node is maxed.
- **Your interconnect is good.** On NVLink, DDP scales beautifully. On PCIe-only or slow Ethernet, the comms ratio degrades and you may need gradient compression or accumulation to keep efficiency up.

Do **not** reach for it — or reach for something more — when:

- **The model does not fit on one GPU.** DDP cannot help; you need to *shard* the model. That is [ZeRO and FSDP](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) (shard optimizer/gradients/params) or tensor/pipeline parallelism. DDP replicates; it never shards.
- **The optimizer state is the thing that will not fit.** Adam keeps two extra full-precision states per parameter — the `(2 + 2 + 12)Ψ` memory math — and for large models that dominates. FSDP/ZeRO shard exactly that. Do not try to brute-force it with DDP.
- **You are at one GPU and it is fast enough.** Distribution is pure overhead if you do not need it. A job that finishes overnight on one card does not need eight; you will spend more time debugging the launch than you save.

The decision rule in one line: **DDP makes a fitting model faster; it never makes a too-big model fit.** The moment "won't fit" is your wall, you graduate from this post to the sharding posts — but you graduate *with* the vocabulary, the launcher, and the sanity checks from here, because FSDP and tensor parallelism are built on the exact same ranks, process groups, and `torchrun` you just learned.

## Key takeaways

- **One process per GPU.** World size counts processes, not machines. Never use the old single-process `DataParallel`.
- **Global rank is unique; local rank resets per node.** Use local rank to bind the GPU (`cuda:local_rank`), global rank to decide who logs and checkpoints.
- **`torchrun` sets the environment; your script reads it.** `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` come from the launcher — never hard-code them.
- **The distributed diff is about six lines:** init the group, `set_device`, wrap in DDP, `DistributedSampler`, `set_epoch`, and `if rank == 0` guards. The model and loss do not change.
- **A hang is a rendezvous or a first-collective problem.** Reach for `NCCL_DEBUG=INFO` and `NCCL_SOCKET_IFNAME` before anything else.
- **Efficiency below 100% is the comms-plus-loader tail, not a bug.** It is governed by one ratio, $t_\text{comm}/t_\text{compute}$; everything you tune later attacks that ratio.
- **When GPU count grows, the global batch grows — so scale the learning rate and lengthen warmup.** Changing your GPU count and touching nothing else is a bug.
- **Sanity-check in sixty seconds:** `nvidia-smi` shows N processes on N GPUs, an all-reduce of ranks sums to $\frac{N(N-1)}{2}$, and per-rank loss moves together but is not identical.
- **DDP makes a fitting model faster; it never makes a too-big model fit.** When "won't fit" is the wall, move to FSDP/ZeRO or model parallelism.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the map of the whole series (start here).
- [The map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) — where data parallelism sits among tensor, pipeline, expert, and sequence parallelism.
- [Collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — the ring all-reduce derivation behind the $2(N-1)/N \cdot S$ byte volume used here.
- [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — bucketing and the compute/comms overlap that keeps efficiency high.
- [Launching on a SLURM cluster](/blog/machine-learning/distributed-training/launching-on-a-slurm-cluster) — the multi-node rendezvous done right with `sbatch` and `srun`.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist that ties the series together.
- [Debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) — deeper NCCL-log reading and hang diagnosis when the sanity checks fail.
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (2017) — the linear scaling rule and warmup.
- Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (VLDB 2020) — the DDP bucketing/overlap design.
- The PyTorch [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) and [DDP](https://pytorch.org/docs/stable/notes/ddp.html) docs, and the [NCCL environment-variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

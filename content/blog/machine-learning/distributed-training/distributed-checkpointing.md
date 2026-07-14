---
title: "Distributed Checkpointing: Saving 500GB of State Without Stalling 512 GPUs"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "How real distributed checkpointing works: sharded parallel saves, resharding onto a different world size, asynchronous writes that hide the disk behind training, and the atomicity that keeps a crash from corrupting your run."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "checkpointing",
    "fsdp",
    "pytorch",
    "fault-tolerance",
    "deep-learning",
    "ml-systems",
    "multi-node",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

It is 3:14 a.m. and your 512-GPU run has been training a 40-billion-parameter model for nine days. The loss curve is beautiful. Then a single H100 on node 23 throws an Xid error, NCCL times out, and the whole job dies. You reach for the last checkpoint. It was written 47 minutes ago, because that is how long a full checkpoint stalls the job, so you only take one every few thousand steps. Forty-seven minutes of 512 GPUs — about 400 GPU-hours, roughly \$800 at \$2 per GPU-hour — evaporates. Worse, when you resume, the loss jumps from 2.1 to 2.9 and takes 600 steps to recover, because the optimizer state came back wrong.

Every one of those failures is a checkpointing failure, and none of them is `torch.save`. At one GPU, a checkpoint is a function call. At 512 GPUs, it is a distributed-systems problem: the state you must persist is hundreds of gigabytes to terabytes, it is scattered across hundreds of ranks as shards, you have to write it every N steps *without stalling the whole job*, and you have to be able to read it back **correctly** — possibly onto a different number of GPUs than you saved it on. Get any of that wrong and you either lose hours of compute to stalls, lose hours of progress to crashes, or silently corrupt the run.

This post is how distributed checkpointing actually works in the trenches. We will start from the naive approaches everyone tries first and watch them break at scale. Then we will build up the real machinery: **sharded (distributed) checkpoints** where every rank writes only its slice in parallel; **resharding** so you can resume a 512-GPU run on 256 GPUs; **asynchronous** saves that hide the disk write behind the next training steps so the stall drops from 90 seconds to 3; and the **atomicity and correctness** discipline — atomic rename, retention, versioning, and saving *all* of the state — that keeps a crash from corrupting your last good checkpoint. Figure 1 is the thing we are actually trying to persist, and it is bigger and weirder than most people expect.

By the end you will be able to write a `torch.distributed.checkpoint` save/load path that shards across ranks, resumes onto a different world size, overlaps the disk with training, and never leaves a half-written checkpoint on disk. This sits in the reliability corner of the [distributed training series](/blog/machine-learning/distributed-training/why-distributed-training): it is what makes a long run *survivable*.

## 1. What actually lives in a checkpoint

Before you can save state efficiently, you have to know how much of it there is and what it is. The instinct is "the model weights." At scale, the weights are the *small* part.

![Stacked breakdown of checkpoint contents showing model weights dwarfed by the fp32 master copy and Adam's two moment tensors, plus tiny scheduler and RNG state](/imgs/blogs/distributed-checkpointing-1.webp)

Take a 40B-parameter model trained with Adam in mixed precision — the standard recipe covered in [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model). Per parameter, the persistent state is:

- **bf16 weights** — 2 bytes/param. These are what you actually compute forward/backward with. For 40B params: 80 GB.
- **fp32 master weights** — 4 bytes/param. The high-precision copy the optimizer updates so that tiny gradient steps don't vanish into bf16 rounding. 160 GB.
- **Adam first moment `m`** — 4 bytes/param (fp32). 160 GB.
- **Adam second moment `v`** — 4 bytes/param (fp32). 160 GB.

That is the famous `(2 + 2 + 12)Ψ` memory model, minus the transient 2 bytes of gradients you don't checkpoint. The optimizer state — the fp32 master plus the two moments, `12Ψ` in bytes — is **six times** the size of the bf16 weights you were worried about. Add it up: roughly 480 GB of optimizer-and-master state plus 80 GB of weights, and you are at ~500 GB for a *40B* model. A 70B run pushes past 800 GB; a 175B run is well into the terabytes.

And that is still not everything. A checkpoint that lets you *resume the exact run* — bit-for-bit continuing, not "start a new run from these weights" — also has to carry the small-but-load-bearing state:

| What | Typical size | Why it must be saved | What breaks if you skip it |
|---|---|---|---|
| bf16 / fp32 weights | 100s of GB | The model itself | Nothing to resume |
| Optimizer state (`m`, `v`, master) | 100s of GB to TB | Adam's momentum and variance | Loss spikes, effective LR wrong for ~hundreds of steps |
| LR scheduler state | bytes | Where you are on the warmup/decay curve | Wrong learning rate on resume |
| Gradient scaler state (fp16) | bytes | The dynamic loss scale | fp16 overflow/underflow storm on resume |
| Data sampler / iterator position | bytes to KB | Which examples come next | You re-see or skip data; epoch boundaries drift |
| RNG states (torch, cuda, numpy, python) | KB | Dropout masks, augmentation, sampling | Non-deterministic divergence; can't reproduce |
| Global step / token count | bytes | Where the run is | Scheduler, logging, and stopping conditions all wrong |

The tiny fields at the bottom are the ones that cause the 3 a.m. "why did the loss spike after resume" mystery. We have a whole war story on exactly that — [the loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume) — and the punchline there is almost always "the optimizer state or the RNG or the data order came back wrong." So the checkpointing problem is really two problems stapled together: **move 500 GB efficiently**, and **move the last few kilobytes correctly**. Miss either and the run is worse off than if you hadn't resumed at all.

The rest of this post is about doing both at 512 GPUs.

## 2. The two naive approaches and why they collapse

There are exactly two ways to checkpoint that a single-GPU intuition suggests, and both are catastrophic at scale. It is worth watching precisely *how* they break, because the failure modes tell you what the real design has to avoid.

![Before and after comparison of gathering all state to rank zero versus each rank writing its own slice in parallel](/imgs/blogs/distributed-checkpointing-2.webp)

**Naive approach 1: rank-0 gather-and-save.** Everyone writes this first. Under FSDP, each rank only holds a *shard* of every parameter and optimizer tensor. So to call the familiar `torch.save(model.state_dict(), path)`, you first have to reassemble the full, unsharded state on one rank. PyTorch even gives you the knob:

```python
# THE NAIVE PATH — do not do this at scale.
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

# Gather the FULL, unsharded state_dict onto rank 0's CPU.
cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    full_sd = model.state_dict()          # rank 0 now holds ALL 80 GB of weights
    if dist.get_rank() == 0:
        torch.save(full_sd, "ckpt/model.pt")   # single-threaded, one host, one disk stream
```

Trace what this actually does at 512 GPUs. To build `full_sd`, every rank has to `all_gather` its shard of every tensor so rank 0 can hold the whole thing. That is 500 GB of network traffic *funneled into a single host*, serializing what should be a parallel operation through one node's NIC. Then rank 0's CPU RAM has to actually hold hundreds of GB (host OOM is a real risk — a DGX node has 1-2 TB of RAM, but you are now spending most of it on one dict). Then `torch.save` writes that hundreds-of-GB blob to disk **single-threaded, from one process, over one filesystem mount**. At a realistic single-stream write of 1-2 GB/s, 500 GB alone is 4-8 minutes of pure disk time — before you count the gather. And the *optimizer* state is 6x bigger than the weights, so a full optimizer checkpoint by this path is a 30-plus-minute stall in which 511 GPUs sit idle burning money. This is the 47-minute stall from the intro.

**Naive approach 2: every rank saves everything.** "Fine," you say, "skip the gather bottleneck — let every rank save the full state independently." Now you have removed the single-host funnel, but you have created write amplification: 512 ranks each writing a full 500 GB copy is **256 TB** of writes to your shared filesystem for a *single* checkpoint. You will saturate the storage system, blow your storage budget, and every rank still has to `all_gather` the full state first. This is strictly worse. Nobody runs it, but it is the reductio that shows the real answer: you want each rank to write *only what it already holds*, and you want the union of those writes to be *exactly one copy* of the state.

That is the entire design brief for sharded checkpointing. Formally, if the total state is $S$ bytes across $N$ ranks, the naive gather writes $S$ bytes through one node with an $O(S)$ gather first; every-rank-saves-all writes $N \cdot S$ bytes; and the target we want is each rank writing $S/N$ bytes **in parallel**, with total on-disk footprint $S$. For $S$ = 500 GB and $N$ = 512, that is each rank writing under 1 GB, concurrently, to a shared filesystem that can absorb hundreds of GB/s aggregate. Seconds, not minutes.

## 3. Sharded checkpoints: the core mechanism

The fix is to stop thinking of a checkpoint as "one file" and start thinking of it as "a set of shard files plus a metadata index that describes the global layout." This is what `torch.distributed.checkpoint` — universally abbreviated **DCP** — implements, and it is the single most important tool in this post.

The core move: **each rank saves only its own shard, in parallel, to a shared filesystem, and no gather ever happens.** Under FSDP each rank already holds a contiguous slice of every flattened parameter and optimizer tensor. DCP asks each rank to write that slice to its own file, and writes a single `.metadata` file that records, for every *logical* tensor in the model, its global shape and which byte ranges of which shard files hold which parts of it. The checkpoint on disk looks like this:

```console
$ ls -la ckpt/step_9000/
-rw-r--r--  __0_0.distcp     1.0G   # rank 0's shard of params + optim
-rw-r--r--  __1_0.distcp     1.0G   # rank 1's shard
-rw-r--r--  __2_0.distcp     1.0G   # ...
...
-rw-r--r--  __511_0.distcp   1.0G   # rank 511's shard
-rw-r--r--  .metadata        4.2M   # global layout: tensor -> (file, offset, shape)
```

Every `.distcp` file was written *concurrently* by its owning rank. There is no reassembly, no single-host funnel, no `all_gather`. The metadata is what makes the pile of shard files a coherent checkpoint rather than 512 opaque blobs.

The API separates two concerns cleanly: getting a *sharding-aware* state dict, and writing it. Since PyTorch 2.2, `torch.distributed.checkpoint.state_dict` gives you helpers that return the model and optimizer state as `DTensor`/`ShardedTensor` objects that carry their placement — which is exactly the information DCP needs to write correct metadata.

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

def save_sharded(model, optimizer, step, base_dir="ckpt"):
    # Sharding-aware state dicts: each rank's local shards, tagged with placement.
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state_dict = {"model": model_sd, "optim": optim_sd}

    path = f"{base_dir}/step_{step}"
    # Every rank writes its own shard file(s) in parallel; one .metadata is written once.
    dcp.save(
        state_dict,
        storage_writer=dcp.FileSystemWriter(path, thread_count=4),
    )
    # Collective internally — all ranks participate; returns when the write is durable.
```

Two details that matter in production. First, `thread_count` lets each rank use multiple writer threads, so a rank with a 1 GB shard can saturate the filesystem bandwidth it's been allocated rather than dribbling out one stream. Second, `dcp.save` is a **collective** — every rank must call it, or you deadlock exactly like a mismatched `all_reduce` (see [debugging distributed jobs](/blog/machine-learning/distributed-training/why-distributed-training) for the general hang pattern). Never guard it with `if rank == 0`.

The `get_state_dict` / `set_state_dict` pair is doing quiet but essential work. It reconciles the FSDP flat-parameter view with the logical module structure, pulls the optimizer's `m` and `v` tensors into the same sharded representation, and — critically — makes the optimizer state *reshardable*, which we are about to exploit. If you have wrestled with FSDP state-dict types before, this is the modern replacement for manually juggling the `StateDictType.SHARDED_STATE_DICT` context manager; the [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) post covers the underlying sharding strategies these helpers wrap.

It is worth being concrete about what the `.metadata` file holds, because it is the whole reason resharding works later. For every logical tensor in the model — say `layers.12.attention.wq.weight`, globally shaped `[8192, 8192]` — the metadata records the global shape and a list of *chunks*: each chunk is `(offset, size, which shard file, byte range in that file)`. A tensor that FSDP split across 512 ranks appears in the metadata as 512 chunks, each pointing at a different `.distcp` file and a different global offset. The shard files themselves are just opaque byte payloads; all the *meaning* lives in the metadata. This is exactly why every rank must call `dcp.save` — the write is a collective that agrees on the global metadata — and why one truncated `.metadata` poisons the entire checkpoint even if all 512 shard files are perfect.

Under the hood, on FSDP you can reach the same sharding-aware tensors through the older explicit context manager, which is useful to know when you're reading legacy code:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

# The mechanism get_state_dict() wraps: SHARDED_STATE_DICT yields each rank's
# local shards as ShardedTensor/DTensor, ready for dcp.save (NO gather).
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    sharded_sd = model.state_dict()          # local shards, layout-tagged
    dcp.save({"model": sharded_sd},
             storage_writer=dcp.FileSystemWriter("ckpt/step_9000"))
```

Two `FileSystemWriter` knobs matter in production. `single_file_per_rank=True` (the default) keeps one file per rank so 512 ranks don't create 512x-more small files and hammer the filesystem's metadata server; flip it off only if your filesystem prefers many small files to a few large ones. And `thread_count` sets the writer threads *per rank*, so a rank with several GB to write can drive its allotted bandwidth with parallel streams instead of one. On a busy parallel filesystem, tuning these two is often the difference between a 3-second and a 20-second save.

#### Worked example: a 70B run, full-gather vs sharded save

Concrete numbers make the gap visceral. Take a 70B model, Adam mixed precision, on 64 nodes of 8x H100 (512 GPUs), writing to a parallel filesystem rated ~200 GB/s aggregate. Full state to persist: ~840 GB of optimizer+master state plus ~140 GB of bf16 weights ≈ **980 GB**, call it ~1 TB.

| Metric | Rank-0 gather | Sharded DCP |
|---|---|---|
| Gather traffic | ~1 TB funneled to 1 host | none |
| Rank-0 host RAM needed | ~1 TB (host OOM risk) | ~2 GB (its shard) |
| Bytes written per rank | ~1 TB (rank 0) | ~2 GB (every rank) |
| Write parallelism | 1 stream | 512 streams |
| Effective write path | ~1.5 GB/s single stream | ~200 GB/s aggregate |
| Wall-clock stall | 11+ min write, plus gather | ~5 s |
| GPU-hours wasted / ckpt | ~95 GPU-hours (512 x 11 min) | ~0.7 GPU-hours |

At \$2 per GPU-hour, the rank-0 path burns roughly \$190 *per checkpoint* in pure stall; the sharded path burns about \$1.40. If you checkpoint every 30 minutes over a two-week run, that is the difference between ~\$130,000 and ~\$950 spent stalling. The sharded save is not an optimization; it is the difference between checkpointing being feasible and not.

## 4. Resharding: load onto a different world size

Here is the feature that separates a real distributed checkpoint from a pile of shard files, and the one most people don't realize they have until they need it. Because DCP stored each *logical* tensor with its global shape and per-shard offsets — not "rank 3's opaque 1 GB blob" — you can **load the checkpoint onto a different world size than you saved it on.**

![Branching diagram of one layout-independent checkpoint being loaded onto 256, 512, and 1024 GPUs and a changed device mesh](/imgs/blogs/distributed-checkpointing-3.webp)

Why would you ever need this? All the time, it turns out:

- You saved on 512 GPUs, but 128 of them are down for maintenance, and you want to resume *now* on 384 rather than wait.
- You are debugging a loss spike and want to reproduce a step on a single node (8 GPUs) using the exact checkpointed state.
- You changed your parallelism plan — say you moved from tensor-parallel degree 2 to degree 4 — and the optimizer sharding is now different.
- You want to fine-tune a pretrained checkpoint on a smaller cluster than it was trained on.

With a naive per-rank save (rank `i` writes its opaque shard `i`), *none* of this works: shard file `i` only means anything to a run with the exact same world size and sharding, because it is just "the bytes rank `i` happened to hold." You are welded to 512 GPUs forever, or you have to do a full gather-and-reshard offline.

DCP makes resharding automatic because the load is *coordinate-based*, not *file-based*. On load, each rank looks at the metadata, computes the global coordinates of the slice *its current sharding* needs, intersects that with the saved shards' coordinate ranges, and reads exactly those byte ranges — potentially stitching a slice together from two or three different shard files. Load onto half as many GPUs and each new rank's slice is the union of two old shards; load onto twice as many and each new rank reads half of one old shard. The reader does the arithmetic; you don't.

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

def load_sharded(model, optimizer, path):
    # get_state_dict on the FRESH model/optimizer gives correctly-shaped, sharded
    # PLACEHOLDERS for THIS world size. DCP fills them in place, resharding as needed.
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state_dict = {"model": model_sd, "optim": optim_sd}

    dcp.load(
        state_dict,
        storage_reader=dcp.FileSystemReader(path),
    )  # each rank reads only the byte ranges its shard needs, from any layout

    # Write the loaded tensors back into the live model + optimizer.
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict["optim"],
    )
```

The mental trick that makes this click: you never "load the checkpoint and then reshard it." You construct *empty placeholders shaped for the world size you have right now*, hand them to `dcp.load`, and DCP reads from disk directly into those placeholders. The load target defines the sharding; the checkpoint is layout-agnostic. That is why the same 512-GPU checkpoint feeds a 256-GPU resume and an 8-GPU debug session with zero conversion step.

This is also the correctness fix for a whole class of resume bugs. If you save an optimizer state that is implicitly tied to a specific FSDP flat-parameter layout and then load it under a different world size *without* a layout-aware system, Adam's `m` and `v` land against the wrong parameters — the optimizer thinks a weight has momentum that actually belongs to a different weight. The result is precisely the loss spike in [the loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume) and in the checkpoint-and-resume debugging guide over in [debugging AI training](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume). DCP's coordinate-based load is what keeps `m` and `v` glued to the right parameters across a world-size change.

#### Worked example: resuming 512 → 384 after a node failure

You are 9 days into a run on 64 nodes. Two nodes die. You have 62 nodes = 496 GPUs, but your parallelism plan wants a multiple of 8, so you resume on 60 nodes = 480 GPUs while the two bad nodes are swapped. With a layout-locked checkpoint, you are stuck: you must either wait for replacement hardware (hours of idle cluster) or run an offline gather-reshard job (another big stall). With DCP, you point `dcp.load` at the same `step_9000/` directory from a freshly-initialized 480-GPU job. Each of the 480 ranks now owns a slightly larger slice than before; on load, most ranks stitch their slice from parts of two old shard files. No offline step, no gather, no conversion. The run is back within the time it takes `torchrun` to rendezvous. This composes directly with elastic restarts — see [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) for the rendezvous side of the same story.

## 5. Asynchronous checkpointing: killing the stall

Sharded save already dropped the stall from minutes to seconds. But "seconds, every few hundred steps" still adds up, and it is *pure* stall — the GPUs are idle while the disk writes. The last big win is to notice that the save has two very different phases with very different speeds, and to overlap the slow one with training.

![Timeline of an asynchronous checkpoint showing a short device-to-host copy stall, training resuming, and a background thread writing to disk in parallel](/imgs/blogs/distributed-checkpointing-4.webp)

When a rank saves its ~1 GB shard, two things happen in sequence:

1. **Device-to-host copy.** The shard tensors live in GPU HBM. They get copied out to CPU RAM. Over PCIe Gen4 at ~25 GB/s (or faster on Grace-Hopper's NVLink-C2C), a 1 GB shard copies in ~40 ms; a few GB with overhead is on the order of 1-3 seconds. This step *must* stall training, because if training keeps stepping the optimizer, it mutates the very tensors you're trying to snapshot.

2. **Disk write.** The CPU-side copy gets written to the shared filesystem. This is the slow part — seconds to tens of seconds depending on shard size and filesystem contention — and, crucially, it touches *nothing on the GPU.*

The insight: only step 1 needs to block training. Step 2 can run in a **background thread** while the GPUs go right back to computing step N+1, because it operates on a CPU-side copy that training will never touch. So the recipe is: synchronously copy each shard into a **pinned CPU staging buffer** (pinned so the D2H copy is fast and async-capable), then hand that buffer to a background writer thread and immediately resume training. The GPU stall collapses to just the D2H copy — a couple of seconds instead of ninety.

DCP ships this as `dcp.async_save`:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

_ckpt_future = None  # module-level handle to the in-flight write

def async_save_sharded(model, optimizer, step, base_dir="ckpt"):
    global _ckpt_future

    # If a previous checkpoint is still writing, wait for it now. This is the
    # double-buffer guard: never start staging a new snapshot until the old
    # staging buffer has been flushed to disk.
    if _ckpt_future is not None:
        _ckpt_future.result()

    model_sd, optim_sd = get_state_dict(model, optimizer)
    state_dict = {"model": model_sd, "optim": optim_sd}

    # Returns almost immediately: it performs the (blocking) D2H copy into a
    # pinned staging buffer, then kicks the disk write onto a background thread
    # and hands you back a Future. Training can proceed after this returns.
    _ckpt_future = dcp.async_save(
        state_dict,
        storage_writer=dcp.FileSystemWriter(f"{base_dir}/step_{step}"),
    )

# --- in the training loop ---
for step, batch in enumerate(loader):
    loss = train_step(model, optimizer, batch)
    if step % CKPT_EVERY == 0 and step > 0:
        async_save_sharded(model, optimizer, step)   # ~2 s stall, not ~90 s
```

The **double buffer** is the subtle part, and the reason `_ckpt_future` is module-level. The staging buffer holds the snapshot that the background thread is writing. If you started a second `async_save` before the first finished, you would either overwrite the buffer mid-write (corruption) or need a second buffer's worth of pinned RAM. The clean discipline is: keep one Future, and *block on it at the start of the next save*. In steady state the write finishes long before the next checkpoint interval, so `_ckpt_future.result()` returns instantly and you never actually wait; but if the disk is slow that step, you correctly fall back to waiting rather than corrupting. (You should also call `.result()` in your shutdown path so a clean exit doesn't drop the last in-flight checkpoint.)

Two resource facts govern whether async is safe on your node. First, the **staging RAM budget**: the pinned CPU buffer has to hold one rank's snapshot — its shard of weights plus optimizer state — for the whole duration of the background write. That is a couple of GB per rank for our 70B example, times 8 ranks per node, so ~16 GB of pinned host RAM per node dedicated to in-flight checkpoints. On a node with 1-2 TB of RAM this is a rounding error, but pinned memory is a limited resource and over-allocating it can starve your data loader's pinned buffers, so budget it deliberately. Second, **backpressure**: if the disk write consistently takes longer than your checkpoint interval, the `_ckpt_future.result()` guard at the top of the next save will start actually blocking — async silently degrades back toward synchronous. That is the system telling you your interval is too tight for your filesystem; either lengthen the interval or fix the storage, because you have run out of overlap to hide behind.

There is one real hazard to name: between the D2H copy and the background write completing, the checkpoint is **not yet durable.** If the *process* crashes during those tens of seconds, that checkpoint is lost — you fall back to the previous one. That is fine (you lose at most one interval), but it means "async_save returned" is *not* "the checkpoint is safe." Only the Future resolving — and, as we will see next, the atomic rename — means safe.

#### Worked example: 90-second stall → 3-second stall

Same 70B run, 512 GPUs. Each rank writes ~2 GB. Suppose the shared filesystem, under real contention from 512 concurrent writers, gives each rank an effective ~25 MB/s of *sustained* write for its stream after metadata and coordination overhead — so the disk write for a 2 GB shard takes ~80 s, and with barrier/metadata overhead the synchronous save blocks for ~90 s. The D2H copy of 2 GB over PCIe Gen4 is ~80 ms of transfer, ~1-3 s with pinning, allocation, and the collective barrier.

| | Synchronous DCP | Async DCP |
|---|---|---|
| GPU-blocking time / ckpt | ~90 s | ~3 s |
| Disk-write time | ~80 s (blocking) | ~80 s (background, overlapped) |
| GPUs idle / ckpt | ~90 s x 512 = ~12.8 GPU-hr | ~3 s x 512 = ~0.43 GPU-hr |
| Overhead if ckpt every 500 steps @ 1.2 s/step (600 s) | 90/600 = **15%** | 3/600 = **0.5%** |

A 15% throughput tax versus a 0.5% tax, from the same checkpoint, same interval, same disk. On a two-week, \$500k run, 15% overhead is ~\$75k of stall; 0.5% is ~\$2.5k. And — the part we will make rigorous in section 8 — because async makes checkpoints cheap, you can afford to take them *more often*, which means you lose *less* work when a node dies. Async doesn't just cut the stall; it changes the whole frequency calculus.

## 6. Correctness and atomicity: never ship a half-written checkpoint

A fast checkpoint that is occasionally corrupt is worse than a slow one that is always correct, because the corrupt one you *trust* — you delete the good checkpoint behind it and only discover the corruption when you try to resume from it after a crash. Distributed checkpointing at scale has three correctness hazards, and each has a standard defense.

![Branching flow of writing shards to a temp directory then either a crash leaving an ignored partial or an atomic rename producing a good checkpoint and pruning old ones](/imgs/blogs/distributed-checkpointing-6.webp)

**Hazard 1: a crash mid-write.** If the job dies while writing `step_9000/`, you now have a directory with some shard files, maybe a truncated `.metadata`, and no way to know it's incomplete. If your resume logic just globs for the highest step number, it will happily try to load a half-written checkpoint. The defense is the classic **write-to-temp-then-atomic-rename**: every rank writes into `step_9000.tmp/`, everyone fsyncs, a barrier confirms all ranks finished, and *then* one rank atomically renames `step_9000.tmp/` → `step_9000/`. On a POSIX shared filesystem a directory rename is atomic, so a resume only ever sees a *complete* `step_9000/` or nothing — never a partial. A crash before the rename leaves an ignorable `.tmp` directory that your cleanup sweeps away.

**Hazard 2: deleting your only good checkpoint.** Retention is a trap. You want to keep the last K checkpoints (disk isn't free — 500 GB each adds up), so you prune the oldest after each save. The bug is pruning *before* the new one is confirmed good. The rule: **verify the new checkpoint is complete and loadable, then prune.** "Complete" at minimum means the atomic rename succeeded and the `.metadata` plus all expected shard files exist with the right sizes; the paranoid version actually does a metadata-only `dcp.load` dry run. Only after that do you delete the `step_(9000 - K*interval)` directory. And you never let retention drop below 2, so there is always a fallback if the newest turns out bad.

**Hazard 3: saving the weights but not the *run*.** Section 1's table listed the small state — scheduler, scaler, sampler, RNG, step — that resume correctness depends on. DCP will happily checkpoint whatever you put in the state dict, so the discipline is to define one object that owns *all* of it. The clean pattern is a small `Stateful`-style app-state class:

```python
import torch, random, numpy as np
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict,
)

class AppState:
    """The COMPLETE run state. Everything needed to resume bit-for-bit."""
    def __init__(self, model, optimizer, scheduler, scaler, sampler):
        self.model, self.optimizer = model, optimizer
        self.scheduler, self.scaler, self.sampler = scheduler, scaler, sampler
        self.step = 0

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_sd,
            "optim": optim_sd,                       # sharded, reshardable
            "sched": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sampler": self.sampler.state_dict(),    # data iterator position
            "step": self.step,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }

    def load_state_dict(self, sd):
        set_state_dict(
            self.model, self.optimizer,
            model_state_dict=sd["model"], optim_state_dict=sd["optim"],
        )
        self.scheduler.load_state_dict(sd["sched"])
        self.scaler.load_state_dict(sd["scaler"])
        self.sampler.load_state_dict(sd["sampler"])
        self.step = sd["step"]
        torch.set_rng_state(sd["rng_torch"])
        torch.cuda.set_rng_state(sd["rng_cuda"])
        np.random.set_state(sd["rng_numpy"])
        random.setstate(sd["rng_python"])
```

Now put atomicity, retention, and the app state together into a save that is safe against a crash at any instant:

```python
import os, glob, shutil
import torch.distributed as dist

def save_checkpoint(app_state, step, base_dir="ckpt", keep=3):
    final = f"{base_dir}/step_{step}"
    tmp   = f"{final}.tmp"

    # 1. Every rank writes its shard into the TEMP dir (collective).
    dcp.save({"app": app_state.state_dict()},
             storage_writer=dcp.FileSystemWriter(tmp, thread_count=4))

    # 2. Barrier: no rank proceeds until ALL shard files + metadata are on disk.
    dist.barrier()

    # 3. Rank 0 atomically publishes the checkpoint, updates the pointer, prunes.
    if dist.get_rank() == 0:
        os.replace(tmp, final)                       # atomic dir rename
        with open(f"{base_dir}/latest.tmp", "w") as f:
            f.write(f"step_{step}\n")
        os.replace(f"{base_dir}/latest.tmp",
                   f"{base_dir}/latest")             # atomic pointer update

        # Prune only AFTER the new checkpoint is safely published.
        steps = sorted(int(p.split("_")[-1])
                       for p in glob.glob(f"{base_dir}/step_*")
                       if ".tmp" not in p)
        for old in steps[:-keep]:
            shutil.rmtree(f"{base_dir}/step_{old}", ignore_errors=True)
    dist.barrier()
```

The `latest` pointer file is the resume entry point: it is updated *after* the checkpoint directory is complete and via its own atomic rename, so a resume reads `latest`, gets a step number, and is guaranteed that `step_{n}/` is a fully-written checkpoint. Order matters — publish the data, *then* publish the pointer, *then* prune. Any crash in that sequence leaves you with a valid previous checkpoint and, at worst, a stray `.tmp` directory.

One more correctness note specific to *async* save: the atomic rename and pointer update have to happen when the background *write* completes, not when `async_save` returns. In practice you attach a completion callback to the Future (or do the rename in the same background machinery), so "published" still means "fully durable." DCP's `FileSystemWriter` handles the temp/finalize dance internally for the shard write; the retention-and-pointer logic above is the piece you own.

## 7. Comparing the strategies at a glance

We have now built up four distinct strategies. It is worth laying them side by side, because the point of the whole post is that only the last one is actually viable at 512 GPUs.

![Matrix comparing four checkpoint strategies across per-rank write size, GPU stall, resharding support, and total storage](/imgs/blogs/distributed-checkpointing-5.webp)

| Strategy | Per-rank write | GPU stall | Reshardable? | Total storage | Verdict |
|---|---|---|---|---|---|
| Rank-0 gather | Full state (rank 0) | 10-20+ min | Yes (unsharded) | 1x | Impossible at scale: gather + single-stream write |
| Every rank saves all | Full state x N | Minutes + all-gather | Yes (full copy) | N x | Absurd: 256 TB per checkpoint at N=512 |
| Sharded DCP (sync) | ~state/N, parallel | Seconds (blocking) | Yes (offsets) | 1x | Good; the stall is small but pure |
| Sharded DCP + async | ~state/N, parallel | 1-3 s (D2H only) | Yes (offsets) | 1x | The answer: parallel, one copy, reshardable, hidden |

The table encodes the design pressure of the whole post: you want the **per-rank write small** (rules out both naive approaches), the **total storage at one copy** (rules out every-rank-saves-all), **resharding supported** (rules out opaque per-rank blobs), and the **stall hidden** (rules out synchronous save at high frequency). Only sharded DCP with async save is a `success` on all four axes. Everything else fails at least one.

A caution on the "reshardable" column: rank-0 gather and every-rank-saves-all are technically reshardable only because they store the *full unsharded* state — you can slice a full tensor any way you like. But they pay for that with the gather or the N-x storage. Sharded DCP is the only option that is reshardable *and* cheap, because resharding is a property of the *metadata format* (global offsets), not of storing a redundant full copy.

## 8. How often should you checkpoint? The MTBF trade

Cheap checkpoints change the strategic question from "can we afford to checkpoint at all?" to "how *often* should we?" This is not a vibes decision — there is a clean optimum, and it is one of the most useful formulas in large-scale training.

![Decision tree for choosing a checkpoint interval based on whether async save is available and whether the cluster mean-time-between-failures is short](/imgs/blogs/distributed-checkpointing-7.webp)

Set up the trade. Let $\delta$ be the cost (in time) to take one checkpoint — the *blocking* part, so $\delta \approx 90$ s for synchronous save or $\approx 3$ s for async. Let $\tau$ be the interval between checkpoints, and let $M$ be the cluster's mean time between failures. Over a long run, two things cost you time:

- **Checkpoint overhead:** you pay $\delta$ every $\tau$ seconds, a fraction $\delta / \tau$ of your time.
- **Lost work on failure:** when a crash hits, you lose the work since the last checkpoint. On average that is half an interval, $\tau / 2$, and crashes come every $M$, so the expected fraction lost is $\tau / (2M)$.

Total wasted fraction is the sum:

$$W(\tau) = \frac{\delta}{\tau} + \frac{\tau}{2M}$$

Checkpoint too often (small $\tau$) and the first term dominates — you're always saving. Checkpoint too rarely (large $\tau$) and the second dominates — you lose a lot each crash. Minimize by setting the derivative to zero: $-\delta/\tau^2 + 1/(2M) = 0$, which gives the classic **Young-Daly optimal interval**:

$$\tau_\text{opt} = \sqrt{2\,\delta\,M}$$

The optimum scales as the square root of *both* the checkpoint cost and the MTBF. Two consequences drive every real decision:

1. **Cheaper checkpoints → checkpoint more often.** Drop $\delta$ from 90 s to 3 s and $\tau_\text{opt}$ drops by $\sqrt{30} \approx 5.5\times$. This is *why* async matters beyond the per-checkpoint stall: it lets you checkpoint ~5x more frequently at the same overhead, so you lose ~5x less work per crash.
2. **Bigger cluster → shorter MTBF → checkpoint more often.** Failures are roughly independent per node, so a cluster's failure rate scales with node count: $M_\text{cluster} \approx M_\text{node} / (\text{num nodes})$. Double the cluster, halve the MTBF, and $\tau_\text{opt}$ shrinks by $\sqrt{2}$. Frontier runs on thousands of GPUs fail *often*, which is exactly why they checkpoint aggressively.

#### Worked example: the interval and the overhead, sync vs async

Take a 64-node cluster where each node fails on average once every ~27 days of uptime, so $M_\text{node} \approx 640$ node-hours, and $M_\text{cluster} \approx 640 / 64 = 10$ hours $= 36{,}000$ s. Step time 1.2 s.

- **Synchronous save**, $\delta = 90$ s: $\tau_\text{opt} = \sqrt{2 \cdot 90 \cdot 36000} \approx 2545$ s $\approx 42$ min $\approx 2100$ steps. Overhead at the optimum: $W = 90/2545 + 2545/72000 \approx 3.5\% + 3.5\% = 7.1\%$. And on each crash you lose ~21 minutes of work on average.
- **Async save**, $\delta = 3$ s: $\tau_\text{opt} = \sqrt{2 \cdot 3 \cdot 36000} \approx 465$ s $\approx 7.75$ min $\approx 390$ steps. Overhead: $W = 3/465 + 465/72000 \approx 0.65\% + 0.65\% = 1.3\%$. On each crash you lose only ~4 minutes of work on average.

Async is not just less stall per checkpoint — it moves you to a *better operating point entirely*: 1.3% total overhead and 4 minutes lost per crash, versus 7.1% overhead and 21 minutes lost. Note the elegant symmetry the formula guarantees: at the optimum the two terms are always equal, so half your waste is checkpointing and half is rework. If you find yourself spending 15% of the run checkpointing, you are past the optimum — check whether you've gone async, and whether your interval matches $\sqrt{2\delta M}$. The failure side of this — detecting the crash and restarting fast — is the subject of [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training); checkpointing and fault tolerance are two halves of the same reliability story.

## 9. Measuring checkpoint stall honestly

If you are going to tune $\tau$ and choose sync vs async, you have to *measure* $\delta$ correctly, and there are three ways to fool yourself. First, the checkpoint write is asynchronous with respect to the CPU in ways that make naive wall-clock timing lie. Second, the OS page cache makes the first few checkpoints look fast because they haven't actually hit disk yet. Third, for async you have to distinguish the *blocking* time (what stalls the GPU, the number that goes into $\delta$) from the *total* time to durability.

```python
import time, torch, torch.distributed as dist

def measure_ckpt(app_state, step, base_dir, mode="sync"):
    # 1. Make sure the GPUs are actually idle at a known point, so we time the
    #    checkpoint and not the tail of the previous training step.
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    if mode == "sync":
        save_checkpoint(app_state, step, base_dir)   # blocks until durable
        torch.cuda.synchronize()
        blocking = time.perf_counter() - t0
        durable = blocking                           # same thing for sync
    else:
        fut = dcp.async_save({"app": app_state.state_dict()},
                             storage_writer=dcp.FileSystemWriter(f"{base_dir}/step_{step}"))
        torch.cuda.synchronize()
        blocking = time.perf_counter() - t0          # <-- this is delta: the GPU stall
        fut.result()                                 # now wait for the disk
        durable = time.perf_counter() - t0

    # 2. Only rank 0 reports, and report BOTH numbers for async.
    if dist.get_rank() == 0:
        print(f"[ckpt step {step}] blocking={blocking:.2f}s durable={durable:.2f}s")
    return blocking, durable
```

Three honesty rules baked into that harness. **Warm up and drop the cache:** discard the first two or three checkpoint measurements, because the OS page cache and filesystem client will absorb early writes and report times that steady state won't reproduce; measure the median of several steady-state saves, not the minimum. **`torch.cuda.synchronize()` before you start the clock and before you stop it**, or you will time CPU-side launch overhead instead of the actual copy, and you will attribute the previous training step's tail to the checkpoint. **For async, `blocking` is what you put into the Young-Daly $\delta$** — it is the GPU stall — while `durable` tells you whether the write is finishing inside your interval; if `durable` creeps up toward $\tau$, the background writer is falling behind and the double-buffer guard will start actually blocking you.

One more confound worth naming, the same one that haunts throughput measurement generally (see [throughput regressions](/blog/machine-learning/distributed-training/why-distributed-training) for the full treatment): filesystem contention is *time-varying*. If another job on the shared cluster starts hammering the same parallel filesystem, your `durable` time will spike even though nothing in your code changed. Always log checkpoint timing as a time series alongside loss and throughput, so a checkpoint that suddenly takes 3x longer shows up as a graph, not a surprise at resume time.

## 10. Case studies and real numbers

The design above is not theoretical; it is what the frontier labs learned the hard way and wrote down. A few anchors, stated as approximate because published details vary:

**Llama 3 (Meta, 2024).** The Llama 3 team reported that during a 54-day snapshot of pretraining on a 16,384-GPU H100 cluster, they hit on the order of 400-plus *unexpected* interruptions — roughly one every three hours (54 days is ~1,300 hours; ~419 interruptions is one per ~3.1 hours). At that failure rate, checkpointing is not a nicety; it is the difference between a run that finishes and one that never gets more than a few hours ahead of the last crash. Their answer was exactly the machinery in this post: frequent, low-overhead, sharded checkpoints so that each interruption costs minutes, not hours. This is the empirical face of the $M_\text{cluster} \approx M_\text{node}/\text{num nodes}$ scaling — thousands of GPUs means failures every few hours.

**OPT-175B (Meta, 2022).** The OPT-175B chronicles (the training logbook released with the model) documented a run on 992 80GB A100s that was interrupted dozens of times by hardware failures, ECC errors, and NCCL issues, requiring frequent manual restarts from checkpoints across roughly two months. It is the canonical public example of "at scale, the run *is* a sequence of resumes," and of how much operational pain you inherit if resume isn't fast and correct.

**PyTorch DCP async checkpointing.** When the PyTorch and IBM teams introduced asynchronous DCP, they reported that moving the disk write off the critical path cut the *blocking* checkpoint time by roughly an order of magnitude for large models — the same 90-seconds-to-a-few-seconds shape as our worked example — by staging to pinned CPU memory and writing in the background. The exact factor depends on model size, shard size, and filesystem, but the structure is universal: the D2H copy is milliseconds-to-seconds, the disk write is seconds-to-minutes, and only the former needs to block.

**The reshard-on-resume win.** Teams running large FSDP jobs routinely rely on DCP's layout-independence to resume on whatever hardware is available after a failure, and to convert between training-time sharding and a consolidated inference checkpoint offline. The operational value is exactly the 512→480 scenario: you are never blocked waiting for the *same* number of GPUs to come back, because the checkpoint doesn't care how many you have.

**In-memory and tiered checkpointing (the research frontier).** Async-to-shared-disk is the production baseline, but the systems literature pushes further. *CheckFreq* (Mohan et al., FAST 2021) formalized pipelining the snapshot and the flush and auto-tuning the frequency — essentially the Young-Daly idea made adaptive. *Gemini* (Wang et al., SOSP 2023) checkpoints to *peer GPU/CPU memory across the cluster* first, so recovery from the common case (a transient failure where most nodes survive) reads the checkpoint back from RAM in seconds, with async writes to durable storage only as a slower backstop. Microsoft's *just-in-time checkpointing* work goes further still, capturing state only when a failure is detected. The common thread is a **storage hierarchy**: local NVMe or peer RAM is fast but dies with the node; shared filesystem is durable but slow; so you stage to the fast tier synchronously and drain to the durable tier asynchronously — the same D2H-then-background-write pattern, generalized to more tiers. You do not need this on day one, but it is where the field is heading, and it explains why "checkpoint to local NVMe" is not a complete answer on its own.

The through-line of all five: at scale, the checkpoint system is load-bearing infrastructure, and the four properties we derived — parallel, one-copy, reshardable, non-blocking — are the ones the people running the biggest jobs converged on independently.

## 11. Stress-testing the design

A design you haven't tried to break is a design you don't understand yet. Let's push the sharded-async-atomic checkpoint into the corners where it bends, because those corners are exactly where a real run finds you.

**Stress test 1: 2048 GPUs.** Scale from 512 to 2048 and two things get worse. First, the *file-creation storm*: 2048 ranks each creating shard files at once can overwhelm a filesystem's metadata server even if the raw bandwidth is fine — this is why `single_file_per_rank=True` matters, and why some sites go further and have ranks write in *groups*, so 2048 ranks produce, say, 256 files instead of 2048. Second, the *collective cost*: the `dist.barrier()` and the metadata all-gather inside `dcp.save` now coordinate 2048 ranks, and a single straggler rank — a node whose disk is momentarily slow — makes everyone wait, because the save only finishes when the *slowest* writer finishes. The fix is the same as everywhere else in distributed training: the barrier is a synchronization point, so your checkpoint time is set by your worst rank, and you monitor the *distribution* of per-rank write times, not the mean. If one rank is consistently 3x slower, you've found a bad disk before it fails.

**Stress test 2: object storage instead of a POSIX filesystem.** Our whole atomicity story rested on one primitive: `os.replace(tmp, final)`, an atomic directory rename. On S3, GCS, or any object store, **there is no atomic directory rename** — "renaming" a prefix is a copy-then-delete of every object, which is neither atomic nor cheap. So the temp-then-rename trick doesn't translate. The standard replacement is a **completion sentinel**: write all the shard objects and the metadata under the final prefix directly, and *only after everything is uploaded*, write a tiny `_SUCCESS` (or `.complete`) marker object. Resume logic then treats a checkpoint as valid only if its marker exists:

```python
import os

def is_complete(path, storage="posix"):
    """A checkpoint is trustworthy only if its completion marker exists."""
    if storage == "posix":
        return os.path.isdir(path)                 # published via atomic rename
    else:  # object store: the marker is written LAST, after all shards + metadata
        return object_exists(f"{path}/_SUCCESS")

def latest_complete(base_dir, storage="posix"):
    candidates = sorted(list_checkpoints(base_dir), key=step_of, reverse=True)
    for path in candidates:
        if is_complete(path, storage):
            return path                            # newest one that finished
    return None                                    # cold start
```

The marker is doing the job the atomic rename did on POSIX: it is the single bit that flips a checkpoint from "maybe half-uploaded" to "safe to load." A crash before the marker leaves objects behind (clean them up on a schedule), but resume never trusts them. DCP can write to object stores via fsspec-backed writers, but *you* own the sentinel discipline.

**Stress test 3: a rank dies in the middle of a save.** This is the scariest one, because it sits on the seam between checkpointing and fault tolerance. If rank 137's GPU faults *during* `dcp.save`, the other 511 ranks reach the internal barrier and *wait for a rank that will never arrive.* Without a timeout, the whole job hangs silently — the worst failure mode, because it burns GPU-hours doing nothing until a human notices. The defense is to give every collective a deadline:

```python
import datetime, torch.distributed as dist

dist.init_process_group(
    "nccl",
    timeout=datetime.timedelta(minutes=10),   # collectives abort instead of hanging forever
)
# with TORCH_NCCL_ASYNC_ERROR_HANDLING=1, a stuck collective raises and tears
# down the process group, so the launcher can restart the job from the last
# COMPLETE checkpoint — the in-flight save is simply discarded.
```

With a timeout set, the hung save aborts, the process group tears down, and your launcher (`torchrun` with restarts, or SLURM requeue) brings the job back and loads the last *complete* checkpoint — which, thanks to atomic publish, is never the half-written one. This is the exact handoff to [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training): checkpointing guarantees a *correct* restore point, and fault tolerance guarantees you *reach* it quickly. Neither is sufficient alone — a perfect checkpoint you hang before loading is worthless, and a fast restart to a corrupt checkpoint is worse.

**Stress test 4: local NVMe is 10x faster — why not checkpoint there?** Local NVMe on the node can hit several GB/s per drive, far faster than a contended shared filesystem, so it is tempting to checkpoint locally and skip the network. The catch is durability: when a node *dies* — the single most common reason you needed the checkpoint — its local disk goes with it. A checkpoint that only exists on the failed node is no checkpoint at all. This is why the tiered approach from the case studies is the real answer: stage to fast-local (or peer RAM) synchronously for the common transient-failure case, and drain to durable shared storage asynchronously for the node-loss case. Local-only checkpointing is a trap that works in testing and fails exactly when you need it.

The pattern across all four stress tests: the checkpoint is only as good as its *weakest guarantee under failure* — the slowest rank, the missing atomic primitive, the hung collective, the disk that dies with the node. Design for the failure, not the happy path, because at 2048 GPUs the failure *is* the common case.

## 12. When to reach for this (and when not to)

Distributed checkpointing is not free complexity to add on day one, so be honest about when it earns its keep.

**Reach for sharded DCP when** your model + optimizer state is sharded across ranks at all — i.e. you are on FSDP, ZeRO-2/3, or any tensor/pipeline-parallel setup where no single rank holds the full state. The moment you're sharded, rank-0 gather is a gather you don't need and a stall you can't afford, and DCP is the native fit. If you're already on FSDP2's `fully_shard`, the `get_state_dict`/`set_state_dict` helpers were built for exactly this.

**Reach for async save when** checkpoint overhead is a measurable fraction of your run *and* you have the pinned host RAM to stage a snapshot (you need roughly one shard's worth of extra pinned CPU memory per rank; on a node with 1-2 TB RAM this is trivial). Below a few percent overhead the added complexity of the double-buffer and the Future lifecycle may not be worth it — but at scale you are almost always above that.

**Reach for resharding-aware checkpoints (DCP's default)** essentially always, because it costs nothing extra over sharded save and buys you the freedom to resume on different hardware. There is rarely a reason to save an opaque, world-size-locked checkpoint.

**Don't bother when** you are on a single GPU or a small DDP job where every rank holds the full model — plain `torch.save` on rank 0 is fine, the state is small, and the stall is a fraction of a second. DDP replicates rather than shards, so there is no sharded state to distribute; the sophistication here is a response to *sharding*, and without sharding it is over-engineering. Also skip async if your checkpoints are already sub-second: you would be adding a background thread and a staging buffer to hide a stall that doesn't exist.

**Don't confuse a training checkpoint with an inference checkpoint.** The sharded, optimizer-carrying, resume-exact checkpoint we built is for *continuing the run*. When you're done, you consolidate to a single unsharded weights-only file (weights, no optimizer, no RNG) for serving — that is a one-time offline gather where the minutes-long stall doesn't matter because training is over. Don't ship the 1 TB training checkpoint to your inference cluster.

## 13. Key takeaways

- **A checkpoint at scale is not the weights; it is mostly optimizer state.** For Adam mixed precision the persistent state is `(2 + 12)Ψ` bytes — the fp32 master and two moments are 6x the bf16 weights. Budget for hundreds of GB to terabytes, not for the model size.
- **Rank-0 gather is impossible at scale** — it serializes all state through one host, risks host OOM, and writes single-threaded — and every-rank-saves-all is absurd (N-x storage). The target is each rank writing `state/N` in parallel, one copy total.
- **Sharded distributed checkpointing (DCP) is the answer:** each rank writes only its shard concurrently, plus one `.metadata` file describing the global layout. O(state/N) per rank, fully parallel, seconds not minutes.
- **Resharding is the killer feature.** Because DCP stores tensors with global offsets, you can load a 512-GPU checkpoint onto 256 GPUs, or a changed device mesh, with no gather and no conversion — construct placeholders for the world size you have and let the load fill them.
- **Async save hides the disk behind training.** Only the device-to-host copy into a pinned buffer needs to block (1-3 s); the disk write runs in a background thread. Keep one Future and block on it before the next save (the double-buffer guard). This turns a 90 s stall into ~3 s.
- **Atomicity is non-negotiable:** write to a temp directory, barrier, atomic-rename to publish, update the `latest` pointer, and only *then* prune old checkpoints — never below 2 kept. A crash before the rename leaves an ignorable partial, never a trusted corrupt checkpoint.
- **Save the whole run, not just the model:** optimizer, scheduler, grad scaler, data sampler position, all four RNG states, and the step. Missing any is a loss spike or a non-reproducible resume.
- **Checkpoint frequency has a clean optimum:** $\tau_\text{opt} = \sqrt{2\,\delta\,M}$. Cheap (async) checkpoints and short-MTBF (large) clusters both push you to checkpoint more often; at the optimum, half your waste is saving and half is rework.

For the whole reliability picture — how the crash gets detected and the job restarted around these checkpoints — continue to [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training), and see how it all ties together in [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook).

## Further reading

- **PyTorch Distributed Checkpoint (DCP)** — the official `torch.distributed.checkpoint` docs: `save`, `load`, `async_save`, `FileSystemWriter/Reader`, and the `state_dict` helpers (`get_state_dict`, `set_state_dict`, `StateDictOptions`).
- **PyTorch FSDP docs and the "Getting Started with Distributed Checkpoint" tutorial** — the sharded state-dict types and the resharding contract.
- **"Asynchronous Checkpointing" (PyTorch/IBM engineering blog)** — the pinned-staging-buffer-plus-background-write design and its measured stall reduction.
- **Young (1974) and Daly (2006)** — the derivations behind the optimal checkpoint interval $\tau = \sqrt{2\delta M}$ for HPC systems.
- **"The Llama 3 Herd of Models" (Meta, 2024)** — the infrastructure section on interruptions and checkpointing at 16k GPUs.
- **"OPT: Open Pre-trained Transformer Language Models" and the OPT-175B logbook (Meta, 2022)** — the canonical public account of restarts and checkpoint-driven recovery at scale.
- Within this series: [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model), [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice), [the loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume); and out to [checkpoint-and-resume debugging](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume).

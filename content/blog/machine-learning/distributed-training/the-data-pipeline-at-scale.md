---
title: "The Data Pipeline at Scale: Don't Let the Loader Starve Your GPUs"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "You can perfectly parallelize a model and still run at half speed because the DataLoader cannot feed the GPUs fast enough. This is the throughput budget that decides it, the sharding that keeps ranks from reading the same data twice, and the streaming and pre-tokenization tricks that push a 55 percent MFU run back to 96 percent."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "dataloader",
    "data-pipeline",
    "streaming",
    "pytorch",
    "webdataset",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 42
---

You spent three weeks getting the parallelism right. The 7B-parameter Transformer is wrapped in FSDP, the shards are balanced, the all-reduce overlaps cleanly with the backward pass, and a single-GPU microbenchmark says each H100 should be doing about 9,000 tokens per second at 45 percent MFU. You launch on 64 GPUs, open the dashboard, and the achieved MFU reads 55 percent. Not 44, not 30 — a maddening 55, high enough that everything *looks* fine and low enough that you are lighting roughly \$4,000 an hour on fire and only getting \$2,200 of training out of it.

You do the reflex thing: profile the model. The kernels are efficient. The communication is overlapped. The memory is fine. There is no bubble, no straggler, no NaN. And then you look at the raw GPU utilization trace instead of the averaged number, and you see it — a sawtooth. Every step, `nvidia-smi` shows the GPUs pinned at 100 percent for a stretch and then *dropping to zero* for a beat before the next step begins. The GPUs are not slow. They are **waiting**. Waiting for the one part of the system that has nothing to do with the model, the parallelism, or the interconnect: the data loader that is supposed to have the next batch ready and does not.

![a producer and consumer pipeline where storage feeds decode workers that fill a prefetch queue copied to the GPU](/imgs/blogs/the-data-pipeline-at-scale-1.webp)

This is the wall that surprises people, because it is the one that feels like it should be trivial. Reading files is easy; the hard part was the distributed model. But at scale, feeding thousands of GPUs a steady stream of tokens is itself a first-class distributed-systems problem, and a data loader that keeps up on one GPU can collapse on sixty-four. This is the twenty-first post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and it is squarely about the third of the four walls — *the run is too slow* — except the slowness is coming from a direction nobody profiles first. By the end you will be able to: compute the exact tokens-per-second your GPUs *demand* and what your loader can *supply*, and know which one is losing; tune `num_workers`, `prefetch_factor`, `pin_memory`, and `persistent_workers` from first principles rather than superstition; shard a dataset across ranks *and* workers so no example is read twice; stream a multi-terabyte corpus from object storage without random access; and restore the exact data position after a checkpoint so a resumed run does not silently re-train on data it already saw. The GPU is the expensive part. The whole job of the pipeline is to never make it wait.

## The data loader is a distributed system

Before any tuning, hold the right picture in your head. A training data loader is a classic **producer-consumer pipeline**. The consumer is the GPU: once per step it needs exactly one batch of tensors sitting in device memory, ready to go. The producers are a pool of worker processes, each one pulling raw bytes from storage, decoding and tokenizing them, and collating the results into a batch. Between the two sits a queue: a small buffer of batches that have been prepared but not yet consumed. Figure 1 above is the whole thing. Storage on the left, a fan of workers, the prefetch queue, the host-to-device copy, the GPU on the right.

The single most important property of this pipeline is that its throughput is set by its **slowest stage**, exactly like an assembly line. It does not matter that your GPU can consume a batch every second if the workers can only produce a batch every two seconds — the line runs at the slower rate, and the GPU spends half its time standing at an empty conveyor belt. The whole art of the data pipeline is arranging for the producer side to be *comfortably* faster than the consumer side, so the queue is never empty when the GPU turns around and asks for the next batch.

Here is why this matters in dollars in a way it never did on your laptop. On one GPU, if the loader is slow, your training is slow, and you shrug and go get coffee. On a 64-GPU H100 cluster at roughly \$3 to \$4 per GPU-hour, every percentage point of GPU idle time is money spent on silicon that is doing nothing. A run that sits at 55 percent utilization instead of 95 is not "a bit slow" — it is paying for 64 GPUs and getting the training throughput of 37. Over a three-week pretraining run that gap is real six-figure money, and it is invisible in every metric except the one nobody looks at first: the wall-clock time between when a step ends and the next one begins.

The tell is precisely what the intro described: **GPU utilization dipping to zero between steps**. If you profile a healthy run — and profiling a distributed run to find exactly this kind of stall is its own craft, which we cover in [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) — the GPU trace is a solid block of kernels. If the loader is starving you, the trace has gaps, and the gaps line up exactly with the moments your training loop is blocked inside `next(iter(dataloader))` waiting for a batch that is not ready. Once you learn to see that gap, you cannot unsee it, and it is the first thing to check whenever measured MFU is below what your model microbenchmark promised.

The pipeline has real stages, and any of them can be the slow one:

- **Storage read.** Pulling raw bytes off local NVMe, a network filesystem, or object storage. Bottleneck when files are many and small (per-request latency dominates) or the corpus lives on slow or remote storage.
- **Decode and tokenize.** Decompressing, parsing JSON or Parquet, running the tokenizer, packing sequences. This is CPU work, and it is the stage that most often starves large text runs, because on-the-fly tokenization of long documents is not free.
- **Collate.** Stacking per-sample tensors into a batch, padding, building attention masks. Usually cheap, occasionally a hidden cost if done in slow Python.
- **Host-to-device copy.** Moving the assembled batch from pinned host memory across PCIe or NVLink into the GPU. Small for token-id tensors, larger for images or audio.

The rest of this post is a tour of how to keep every one of those stages ahead of the GPU, starting with the knobs PyTorch gives you and ending with the streaming and sharding machinery you need when the corpus is measured in terabytes.

## When the loader wins: a 55-to-96 percent MFU story

Let me make the abstract concrete with the exact failure from the intro, because the fix is instructive and the numbers recur throughout the post. The setup: a 1.5B-parameter model on a single 8×A100 80GB node, bf16, micro-batch of 8 sequences at 2,048 tokens each, so 16,384 tokens per step per GPU. The A100 does roughly 312 bf16 TFLOP/s of dense matmul (the SXM spec; treat it as an approximate peak). Using the standard estimate of about 6N FLOPs per token for a forward-plus-backward pass on an N-parameter model, one step costs about $6 \times 1.5\times10^9 \times 16384 \approx 1.5\times10^{14}$ FLOPs, and at a realistic 45 percent MFU the GPU chews through that in roughly 1.1 seconds. So the GPU **demands** one fresh batch about every 1.1 seconds — call it 0.9 batches per second.

The original loader read gzipped JSONL shards from S3, decompressed each shard in the worker, ran `json.loads`, tokenized each document with a Python-level loop, packed the tokens into 2,048-length sequences, and collated. Measured, a single worker took about **4 seconds** of CPU time to produce one batch. With `num_workers=2` and no prefetch overlap, the loader delivered roughly one batch every two seconds — about 0.5 batches per second against a demand of 0.9. The GPU asked for a batch, waited, computed for 1.1 seconds, asked again, waited again. Utilization sat at 55 percent and the trace showed the sawtooth.

![a starved loader running at fifty five percent utilization beside a fed loader running at ninety six percent](/imgs/blogs/the-data-pipeline-at-scale-2.webp)

Two changes fixed it, and figure 2 shows the before and after. First, **pre-tokenize offline**. There is no reason to re-parse JSON and re-run the tokenizer on the same corpus every single epoch. Run the tokenizer once, write the token IDs to disk as packed `uint16` arrays (a 32K-or-smaller vocabulary fits in two bytes per token), and now the worker's job is to memory-map a file and slice out a 2,048-token window — pure sequential I/O, about 2 milliseconds per batch instead of 4 seconds. Second, **raise `num_workers`** from 2 to 8 and turn on `pin_memory`, `prefetch_factor`, and `persistent_workers` so the workers stay alive across epochs and stage batches ahead. Post-fix, a single worker produces batches roughly 2,000 times faster, the loader supplies hundreds of batches per second against a demand of 0.9, the queue is never empty, and MFU climbs to 96 percent. Same model, same GPUs, same parallelism — a 1.75x throughput improvement from the data pipeline alone.

Here is the loader configuration that gets you to the "after" state. Every argument in it is load-bearing, and we will justify each one in the next section.

```python
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,   # one shard per rank
    rank=rank,
    shuffle=True,
    drop_last=True,            # keep every rank's batch count identical
)

loader = DataLoader(
    dataset,
    batch_size=8,
    sampler=sampler,           # NOT shuffle=True — the sampler shuffles
    num_workers=8,             # 8 subprocesses decode in parallel
    prefetch_factor=4,         # each worker stages 4 batches ahead
    pin_memory=True,           # page-locked host buffers for fast H2D
    persistent_workers=True,   # do not respawn workers every epoch
    drop_last=True,            # drop the ragged final batch
)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)   # reshuffle differently each epoch
    for batch in loader:
        x = batch["input_ids"].to(device, non_blocking=True)
        # ... forward / backward / step
```

Two lines in there are the ones people get wrong. `sampler.set_epoch(epoch)` is easy to forget and, if you do, every epoch sees the data in the *same* order — we will come back to it. And `non_blocking=True` on the `.to(device)` only actually overlaps the copy with compute when the source tensor is in pinned memory, which is exactly what `pin_memory=True` arranges. The two work as a pair; one without the other is a no-op.

## Anatomy of the DataLoader: four knobs, four stages

The PyTorch `DataLoader` exposes exactly four performance knobs that matter, and each one governs a different stage of the pipeline. Understanding which stage each knob touches is the difference between tuning by principle and tuning by flailing. Figure 3 maps them onto the pipeline.

![four DataLoader levers mapped onto the stages of the pipeline from storage read down to the host to device copy](/imgs/blogs/the-data-pipeline-at-scale-3.webp)

**`num_workers`** is the big one. Set it to zero and the main process does all the fetching and decoding *between* GPU steps — the GPU is guaranteed to wait, because nothing can prepare batch k+1 while the GPU computes batch k. Set it to a positive number and the DataLoader forks that many subprocesses, each running the dataset's `__getitem__` (or the iterable's `__iter__`) independently and in parallel, so decode work happens *while* the GPU is busy. This is the knob that parallelizes the CPU-bound decode stage, and it is the first thing to raise when you are starved.

**`prefetch_factor`** controls how many batches each worker prepares ahead of demand. The default is 2, meaning at steady state each worker keeps roughly 2 batches queued. It exists to absorb *variance*: real decode times are not constant — one shard is bigger, one document is longer, one S3 request is slow — and a deeper prefetch queue rides over those spikes without ever letting the GPU-facing queue run dry. Raise it when your per-batch decode time is bursty; the cost is memory, which we will bound shortly.

**`pin_memory`** allocates the worker output in page-locked ("pinned") host memory. Ordinary pageable memory cannot be copied to the GPU by DMA directly; CUDA has to stage it through a pinned bounce buffer first, serializing the copy. Pinned memory can be DMA'd straight across, and — critically — the copy can be issued as `non_blocking` and overlapped with GPU compute. For a token-ID batch this saves a little; for image or audio batches, where the tensors are large, it is a substantial win.

**`persistent_workers`** keeps the worker subprocesses alive across epoch boundaries instead of tearing them down and respawning them at the start of each epoch. Respawning is expensive: every worker re-imports your modules, re-opens file handles, re-builds any in-memory index. On a dataset with many short epochs, or one where the workers hold an expensive cached index, respawning at every epoch boundary shows up as a stall right where the loss curve has a little flat spot. Set it to `True` and the stall disappears.

Here is the decision table, with the failure mode each knob addresses and the cost of overusing it.

| Knob | Governs | Raise it when | Cost of too much |
|---|---|---|---|
| `num_workers` | Parallel decode | GPU idle, decode is CPU-bound | CPU oversubscription, RAM, too many open FDs |
| `prefetch_factor` | Queue depth / variance | Per-batch time is bursty | Memory grows as workers × prefetch × batch |
| `pin_memory` | H2D copy speed | Batches are large (images/audio) | A little extra pinned RAM; harmless for text |
| `persistent_workers` | Worker lifetime | Many epochs or expensive worker init | Workers hold memory between epochs |

The one that bites hardest at scale is `num_workers`, and not for the obvious reason. On a single GPU, more workers is almost always better up to the core count. But in a distributed run, **every rank has its own DataLoader with its own workers**. On an 8-GPU node with `num_workers=8`, that is 8 ranks times 8 workers equals *64 worker processes* on one node, all competing for the same CPU cores, the same memory bandwidth, and the same NIC. Set `num_workers` too high and you oversubscribe the CPU so badly that each worker slows down, per-batch decode time *rises*, and the "fix" of adding workers makes starvation worse. The right way to think about it is a budget, which is the mechanism we derive next.

### The collate step and sequence packing

The collate function is where per-sample tensors become a batch, and it is usually cheap — but two things can quietly make it the slow stage. The first is **padding waste**. If you batch variable-length sequences by padding every one up to the longest in the batch, a single long outlier inflates the whole batch, wasting both compute and loader memory on padding tokens the model then has to mask out. Length-bucketed sampling — grouping similar-length sequences into the same batch — cuts the padding sharply, at the cost of a little shuffling randomness, and is worth it whenever sequence lengths vary a lot.

The second is **sequence packing**, and it is the better answer for pretraining. Instead of padding, you concatenate multiple short documents into one dense fixed-length sequence with document-boundary separators (and a mask so attention does not cross documents), so every token in the batch is real training signal and there is zero padding. The pre-tokenization script later in this post does exactly this: it fills 2,048-token windows across document boundaries with an end-of-sequence separator. The key move is *where* the packing happens. Packing offline, during pre-tokenization, is free at train time — the online collate stays a trivial stack-and-copy. Packing on the fly in the collate function puts that concatenation logic squarely on the critical path, in Python, in the worker. As with tokenization, the rule is the same: do the expensive shaping once, offline, and keep the online collate a stack-and-copy.

## The throughput budget: supply versus demand

This is the section that turns loader tuning from folklore into arithmetic. There is a clean law, and once you have it you can predict starvation before it happens.

Let $t_c$ be the GPU's **compute time per step** — the wall-clock time the GPU spends actually computing one batch, which you measure with a pure-compute microbenchmark. Let $t_d$ be the **data time**: the wall-clock time for the loader to have the next batch sitting in device memory. With no prefetch overlap, the two stack, so the real step time is $t_c + t_d$ and utilization is $t_c / (t_c + t_d)$. That is the naive, slow world.

The whole point of workers and prefetch is **overlap**: while the GPU computes batch k, the workers are already preparing batch k+1. When overlap is working, the step time is not the sum but the *maximum* of the two stages — the GPU and the loader run concurrently and the step takes as long as whichever is slower:

$$\text{step time} = \max(t_c, t_d), \qquad \text{util} = \frac{t_c}{\max(t_c, t_d)} = \min\!\left(1, \frac{t_c}{t_d}\right).$$

Read that second form carefully because it is the whole game. If $t_d \le t_c$ — the loader can prepare a batch at least as fast as the GPU consumes one — utilization is 100 percent (in the limit) and the loader is invisible. The instant $t_d > t_c$, utilization drops to exactly $t_c / t_d$ and every bit of the excess is pure waste. There is no partial credit and no graceful degradation: the loader is either fast enough to be free, or it is the bottleneck.

Now turn $t_d$ into something you can control. If one worker takes $t_w$ seconds to produce a batch, and you have $W$ workers producing in parallel with deep enough prefetch, the effective data time is $t_d \approx t_w / W$ (ignoring I/O contention for a moment). Substitute into the requirement $t_d \le t_c$:

$$\frac{t_w}{W} \le t_c \quad\Longleftrightarrow\quad W \ge \frac{t_w}{t_c}.$$

**That is the number-of-workers law.** You need at least $t_w / t_c$ workers, rounded up, to keep the GPU fed. It is exact enough to plan with. In the 55-percent story, the slow path had $t_w = 4$ seconds and $t_c = 1.1$ seconds, so the requirement was $W \ge 4/1.1 \approx 3.6$, meaning at least 4 workers — and we were running 2. Starved, exactly as observed. Bump to 8 and we clear the bar with margin. Pre-tokenize and $t_w$ collapses to 0.01 seconds, so a single worker satisfies the law and the extra workers are just insurance against variance. Figure 4 lays the three regimes side by side.

![a supply versus demand table showing two slow workers starving the GPU while eight workers or pre-tokenized shards over-supply it](/imgs/blogs/the-data-pipeline-at-scale-4.webp)

You can also read the same law in tokens per second, which is often how the dashboards report it. The loader **supplies** $W \times (\text{tokens per batch} / t_w)$ tokens per second; the GPU **demands** $\text{tokens per batch} / t_c$ tokens per second. Supply must exceed demand, which reduces to the same $W \ge t_w / t_c$. Whichever units you prefer, the ritual is the same: measure $t_c$ from a compute-only run, measure $t_w$ by timing one worker, and check the inequality before you launch on the big cluster.

#### Worked example: the 64-GPU CPU budget

Now scale it, because the distributed twist changes the arithmetic. We have 8 nodes, 8 A100s each, 64 GPUs total. Every rank runs its own DataLoader, so the per-rank requirement is unchanged: each rank needs $W \ge t_w / t_c$ workers. With pre-tokenized shards, $t_w \approx 0.01$ s and $t_c \approx 1.1$ s, so even one worker per rank technically suffices — but you want a few for variance, say $W = 8$.

Here is the trap. Each node has 8 ranks. If every rank runs 8 workers, that is 64 worker processes per node. A typical DGX-class node has on the order of 96 to 128 physical CPU cores. Sixty-four workers plus 8 main training processes plus NCCL's own threads is fine on 128 cores — roughly one core per worker with headroom. But push `num_workers` to 16 "to be safe" and you now have 128 workers plus overhead on 128 cores: **oversubscription**. The workers time-slice, context-switch, and thrash the cache, and the measured $t_w$ *increases* — the exact opposite of what you wanted. The budget is:

$$W_{\text{per rank}} \le \frac{C_{\text{node}} - C_{\text{overhead}}}{G_{\text{node}}}$$

where $C_{\text{node}}$ is the node's physical cores, $C_{\text{overhead}}$ is what you reserve for the main processes and communication threads (leave a dozen), and $G_{\text{node}}$ is the GPUs per node. For a 96-core, 8-GPU node: $(96 - 16)/8 = 10$ workers per rank, comfortably. Set `num_workers` to 8 or 10, not 16.

There is a companion gotcha: each worker process, by default, may spin up its own pool of intra-op threads (OpenMP, MKL, the tokenizer's own threads). Eight ranks times ten workers times, say, four OMP threads each is 320 threads fighting over 96 cores. Pin it down in your launch script:

```bash
# Give each rank's workers a fair slice; stop libraries from
# spawning a thread pool inside every one of the 64+ worker processes.
export OMP_NUM_THREADS=1          # tokenizers/BLAS: one thread per worker
export TOKENIZERS_PARALLELISM=false

torchrun \
  --nnodes=8 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
  train.py --num-workers 8 --prefetch-factor 4
```

Setting `OMP_NUM_THREADS=1` inside data workers is one of those changes that looks like it should slow things down and instead speeds the whole node up, because the workers stop fighting each other for cores. Measure it on your hardware, but on a busy multi-rank node it is almost always the right call.

### Gradient accumulation does not relax the requirement

Large global batches are usually reached with gradient accumulation: the optimizer steps once every K micro-batches, so the effective batch is K times the micro-batch. It is tempting to think this buys the loader breathing room — the optimizer step is now K times longer, so surely the loader has more time. It does not, and the reasoning is worth being precise about. The GPU still runs one forward-plus-backward per micro-batch, and it still needs a fresh micro-batch for each one. Over a single optimizer step the GPU computes K micro-batches in about $K \cdot t_c$ seconds, and the loader must deliver K micro-batches in that same window — which is exactly $t_d \le t_c$ per micro-batch, unchanged from before. Accumulation moves nothing in the supply-versus-demand budget.

What it does change is where the stall shows up. With accumulation on, the loader has to keep at least K micro-batches queued to cover one optimizer step without a gap, so raise `prefetch_factor` until `num_workers × prefetch_factor` comfortably exceeds K. And because a shortfall now surfaces as one longer stall per optimizer step rather than a small stall per forward, an averaged dashboard smooths it away — count wait time per micro-batch, not per optimizer step, or you undercount the starvation by a factor of K.

### Bounding the memory the loader eats

The prefetch buffers are not free, and at high worker counts they can OOM the *host* (not the GPU). The peak host memory the loader holds is approximately:

$$M \approx \text{num\_workers} \times \text{prefetch\_factor} \times \text{batch\_size} \times \text{sample\_bytes}.$$

For token-ID batches this is tiny — 8 workers × 4 prefetch × 8 sequences × 2,048 tokens × 2 bytes is about 2 MB, nothing. But swap in a vision loader with decoded 512×512×3 uint8 images at batch 32: 8 × 4 × 32 × 786,432 bytes ≈ 800 MB *per rank*, times 8 ranks is 6.4 GB of host RAM just in flight buffers, plus pinned copies. Crank `prefetch_factor` to 16 on that loader and you will get an out-of-memory kill on the host that looks, infuriatingly, like a random worker dying. Which is a good segue, because that is one of several gotchas that only show up at scale.

## When decode is heavy: images, audio, and GPU-side decoding

For text with pre-tokenized shards, $t_w$ is milliseconds and the loader is essentially never the bottleneck. That calculus flips for vision, audio, and video, where the per-sample decode is genuinely expensive and $t_w$ can dwarf $t_c$. A JPEG decode plus resize plus normalize for a 512×512 image is a few milliseconds of CPU each; a batch of 256 is most of a second of pure CPU work, and now the number-of-workers law bites hard because $t_w$ is large. Audio is worse: decoding a compressed clip, resampling to the model's rate, and computing a mel spectrogram is tens of milliseconds per sample. These are the workloads where the loader routinely starves modern accelerators, and the reason is structural — GPUs got several times faster across the last few hardware generations while single-thread CPU decode barely moved, so the ratio $t_w / t_c$ has been creeping up for years.

There are three levers specific to heavy decode, and they are the same three ideas from the text case aimed at a bigger $t_w$:

- **Move decode onto the GPU.** NVIDIA DALI and the `nvJPEG` library decode JPEGs directly on the GPU, and `torchaudio`/`torchvision` have GPU codepaths for resampling and resizing. This trades scarce CPU cycles for abundant GPU cycles and can eliminate the CPU bottleneck outright — at the cost of stealing a little GPU time from the model, so measure the net effect rather than assuming it wins.
- **Pre-resize and pre-resample offline.** Exactly the pre-tokenization logic applied to pixels and samples. If every epoch resizes the same images to the same resolution, do it once and store the resized tensors (a WebDataset of already-decoded arrays, or a fixed-resolution format), so the online worker only reads and copies.
- **Pin memory and use `non_blocking` for real.** Image and audio batches are megabytes, not kilobytes, so the host-to-device copy is a real cost here, and pinned memory with an overlapped copy actually matters — unlike the text case, where it is a rounding error.

The framing does not change — supply versus demand, $W \ge t_w / t_c$ — but $t_w$ is now the term to attack, and the highest-leverage move is almost always to push the expensive decode off the training critical path, whether offline (pre-resize) or onto the GPU (DALI). A vision run stuck at 60 percent utilization with the CPUs pinned at 100 percent is not a model problem and not a bandwidth problem; it is a decode problem, and it is fixed on the CPU side of the pipeline, not the GPU side.

## Streaming a terabyte-scale corpus

Everything so far assumed the dataset is a map-style dataset — a thing with a length and random access by index, which `DistributedSampler` shards cleanly. That model breaks the moment the corpus is too big to fit on local disk. A serious pretraining corpus is multiple terabytes of raw text; even pre-tokenized to `uint16` it can be hundreds of gigabytes to a terabyte, and you cannot assume every node in a 64-GPU cluster has that much fast local scratch. So you **stream**: keep the data in object storage (S3, GCS, Azure Blob) as a sequence of shards, and pull them over the network as you go. Figure 5 sketches the streaming pipeline.

![a streaming pipeline pulling tar shards from object storage through a network prefetch stage into a shuffle buffer before decoding](/imgs/blogs/the-data-pipeline-at-scale-5.webp)

Streaming forces three design decisions that random-access datasets never had to make.

**Shard the data as files, not as indices.** The corpus is split into shards — WebDataset uses `.tar` archives of samples, Mosaic's StreamingDataset uses its own `.mds` format, Megatron and nanoGPT use flat `.bin` files with an index. A shard is typically a few hundred megabytes to a couple of gigabytes: big enough that per-file overhead is amortized, small enough that you can prefetch one while decoding another. Each worker is assigned a disjoint set of *whole shards* and reads them sequentially. Sequential reads from object storage are fast; the enemy is per-object latency, so you want few large reads, not millions of tiny ones.

**Recover randomness with a shuffle buffer.** Random access gave you a globally shuffled epoch for free. Streaming cannot seek to a random sample without killing throughput, so you approximate shuffling in two layers: shuffle the *order of shards* each epoch (cheap — it is a list of a few thousand filenames), and maintain an in-memory **shuffle buffer** of, say, 10,000 samples from which you draw randomly, refilling it as you consume. The buffer decorrelates samples that arrived together in the same shard. It is not a perfect global shuffle, but with a large enough buffer and shard shuffling it is close enough that models train fine — this is exactly the tradeoff `tf.data` popularized and every streaming loader copies.

**Hide network latency with prefetch.** The first read of a cold shard from S3 has real latency — tens to hundreds of milliseconds before the first byte. If you fetch a shard only when you need it, the GPU stalls on every shard boundary. So you prefetch: while the workers decode shard k, a background fetch is already pulling shard k+1 into a local cache. Done right, the network latency is completely hidden behind decode and compute, and the cold-start stall happens exactly once, at the very beginning of the run.

Here is the shape of a WebDataset streaming loader reading `.tar` shards straight from S3, with both levels of sharding and a shuffle buffer. This is close to production-usable; the important lines are the two `nodesplitter`/`workersplitter` arguments that make the sharding correct across ranks and workers.

```python
import webdataset as wds
from torch.utils.data import DataLoader

# Shard URLs live in object storage; brace-expansion enumerates them.
shard_urls = "pipe:aws s3 cp s3://corpus/tokenized/shard-{000000..004095}.tar -"

dataset = (
    wds.WebDataset(
        shard_urls,
        nodesplitter=wds.split_by_node,     # disjoint shards per RANK
        workersplitter=wds.split_by_worker, # disjoint shards per WORKER
        shardshuffle=True,                  # shuffle the shard order
        resampled=False,                    # True = infinite, with replacement
    )
    .shuffle(10_000)                        # in-memory shuffle buffer
    .decode()
    .to_tuple("input_ids.npy")
    .batched(8)
)

loader = DataLoader(
    dataset,
    batch_size=None,        # dataset already batches
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
)
```

The `split_by_node` and `split_by_worker` pair is the entire correctness story of distributed streaming, and getting it wrong is the single most common data bug at scale. It deserves its own section.

Before that, the other half of the streaming decision is *what format the shards are in*, because it decides whether your workers are I/O-bound or CPU-bound. This table is the one to internalize.

| Strategy | Read pattern | Per-batch cost | Best when |
|---|---|---|---|
| Raw text on the fly | Parse + tokenize every epoch | Seconds (CPU-bound) | Never, for repeated epochs |
| Pre-tokenized `.bin` + memmap | Sequential slice, local | ~ms (I/O-bound) | Corpus fits on local NVMe |
| Streaming tar/mds shards | Sequential, over network | ~ms + hidden latency | Corpus too big for local disk |
| Pre-tokenized + cached shards | Stream once, reuse locally | ~ms after warmup | Multi-epoch on big corpus |

The pattern that wins for large multi-epoch runs is the last row: pre-tokenize offline so decode is trivial, stream the shards from object storage so you do not need the whole corpus on local disk, and cache shards on local NVMe after first fetch so subsequent epochs read locally. Here is the offline pre-tokenization step and the memory-mapped dataset that reads it — the piece that turns 4-second batches into 2-millisecond batches.

```python
import numpy as np
from transformers import AutoTokenizer

# --- Offline: run ONCE, write packed uint16 token shards ---
def pretokenize(docs, out_path, tokenizer, seq_len=2048):
    tok = AutoTokenizer.from_pretrained(tokenizer)
    buf = []
    with open(out_path, "wb") as f:
        for doc in docs:
            ids = tok(doc, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            buf.append(tok.eos_token_id)         # document separator
            while len(buf) >= seq_len:
                chunk = np.array(buf[:seq_len], dtype=np.uint16)  # vocab < 65536
                f.write(chunk.tobytes())
                buf = buf[seq_len:]

# --- Online: memory-map the shard, slice a window, near-zero CPU ---
class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len=2048):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n = len(self.data) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        x = torch.from_numpy(self.data[s : s + self.seq_len].astype(np.int64))
        return {"input_ids": x}
```

Memory-mapping means the operating system pages the file in on demand and shares those pages across all the worker processes on the node, so eight workers reading the same `.bin` do not each hold a private copy. The `__getitem__` does no parsing and no tokenization — it slices bytes the kernel has likely already cached. That is why $t_w$ falls to milliseconds. Where the raw corpus comes from and how it was assembled is a whole discipline of its own, covered in [sourcing and collecting training data](/blog/machine-learning/training-data/sourcing-and-collecting-training-data); here we assume it exists and focus on feeding it to the GPUs.

#### Worked example: sizing the shuffle buffer

The shuffle buffer trades memory for shuffle quality, and you can reason about how big it needs to be rather than guessing. Say a shard holds 10,000 sequential samples that are correlated — consecutive documents scraped from the same source, so adjacent samples are similar. If your shuffle buffer holds B samples and you draw uniformly at random, two samples that were adjacent inside a shard end up on average about B positions apart in the training stream. To break the within-shard correlation you want B at least a shard's worth, so a full shard is always in flight and being mixed. With B equal to 10,000 samples at 2,048 `uint16` tokens each, the buffer costs about 10,000 × 4,096 bytes ≈ 40 MB per rank — trivial. Push B to 100,000 for stronger mixing and it is 400 MB per rank, still fine on any training node.

The reason you cannot simply set B enormous is that the buffer must *fill* before the first batch can come out, so a huge buffer adds a cold-start delay at the very beginning of the run and again after every resume. A buffer of one to a few shards is the sweet spot: large enough to decorrelate samples within a shard, small enough that the fill latency is a one-time cost measured in seconds, not minutes. Combined with shuffling the *order* of shards each epoch, this two-level scheme gets you within striking distance of a true global shuffle at a fraction of the memory and none of the random-access cost — which is exactly the tradeoff every production streaming loader has converged on.

## Sharding across ranks and workers: the duplicate bug

This is the bug that does not crash, does not slow you down, and quietly corrupts your training. It is worth slowing down for.

For a **map-style** dataset — one with `__len__` and index access — `DistributedSampler` handles sharding across ranks correctly and automatically. It partitions the index range into `world_size` disjoint subsets, hands rank r its subset, and within a rank the DataLoader round-robins those indices across the workers. So the two levels of sharding — across ranks, then across workers — are both taken care of, and you get exactly one pass over the data per epoch. The only thing you must remember is `sampler.set_epoch(epoch)` so the shuffle differs each epoch (its role in seeding is covered in [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas)).

For an **iterable-style** dataset — a stream with no length and no random access, which is what you use for streaming — *nothing* is automatic. This is the trap. When the DataLoader forks workers, **each worker gets its own complete copy of the dataset object**, and by default each copy iterates the *entire* stream. So on a single GPU with `num_workers=8`, if you do nothing, all 8 workers yield all the data — you train on every example 8 times per "epoch" and call it one epoch. Now add distribution: all 8 ranks also each iterate the entire stream. The total duplication is `world_size × num_workers`. On our 64-GPU cluster with 8 workers each, **every example is fed to the model 512 times per nominal epoch**, your effective dataset is 1/512th the size you think, and the model overfits a tiny slice of the corpus while the loss curve looks perfectly healthy. Nothing errors. You find out when the eval numbers are inexplicably bad and you have burned a week of cluster time.

The fix is to shard the stream by a **global worker id** that is unique across the whole job. There are `world_size × num_workers` streams total; assign each a distinct id and have each stream take every `(world_size × num_workers)`-th shard. Figure 6 shows the correct assignment for a small case — 2 ranks, 3 workers — where all six streams read disjoint stripes of shards.

![a grid showing two ranks by three workers where every cell reads a disjoint stride of shard indices with no overlap](/imgs/blogs/the-data-pipeline-at-scale-6.webp)

```python
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

class ShardedStream(IterableDataset):
    def __init__(self, shard_paths):
        self.shard_paths = shard_paths   # e.g. ["shard-000.bin", ...]

    def __iter__(self):
        # --- rank: which GPU process am I, out of how many? ---
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1

        # --- worker: which subprocess am I within this rank? ---
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1

        # --- global id and total number of streams ---
        global_id = rank * nworkers + wid
        total_streams = world * nworkers

        # Each stream takes a DISJOINT stride of whole shards.
        my_shards = self.shard_paths[global_id::total_streams]
        for path in my_shards:
            yield from self.read_shard(path)
```

The two lines that matter are `global_id = rank * nworkers + wid` and the strided slice `self.shard_paths[global_id::total_streams]`. Together they guarantee every shard is owned by exactly one (rank, worker) pair. Miss the `rank *` term and workers collide across ranks; miss the worker split and workers collide within a rank. Either half alone is a silent duplicate bug.

#### Worked example: catching the double-sharding bug

Suppose you sharded by rank but forgot the worker split — a very common half-fix, because "shard by rank" is the part everyone remembers. You have 8 ranks, `num_workers=8`, and 4,096 shards. Each rank correctly takes 512 shards (`shards[rank::8]`), but within a rank all 8 workers iterate those same 512 shards. Result: 8x duplication within each rank, so the model sees every example 8 times per epoch, your "300B token" run is really a 37.5B-token run replayed 8 times, and your data-scaling assumptions are off by nearly an order of magnitude.

How would you catch it before wasting the run? Two cheap checks. First, a **coverage assertion**: at startup, have every (rank, worker) print the count and a hash of its assigned shard ids, gather them, and assert the union equals the full shard set and the pairwise intersections are empty. Second, a **cardinality check**: count the total samples the loader yields in one epoch across all ranks and compare it to the known corpus size — if it comes out 8x too large, you have duplication; if it comes out short, you are dropping data. Both take minutes and save days.

## Resumability: restoring the exact data position

Long runs get interrupted — a node fails, a spot instance is preempted, you hit the scheduler's wall-clock limit. You checkpoint the model and optimizer and restart from the last checkpoint. But the *data stream* has state too, and if you restore the weights without restoring the data position, you resume from the beginning of the data — re-training on examples the model already saw and skipping the ones it was about to see. On a shuffled multi-epoch run this quietly distorts the data distribution the model trains on. Figure 7 shows the two resume paths.

![a timeline where a naive resume replays data from the first sample while a stateful resume seeks back to the saved sample offset](/imgs/blogs/the-data-pipeline-at-scale-7.webp)

A correct resume has to checkpoint three pieces of data-pipeline state alongside the model: the **shard index** (which shards are consumed), the **sample offset** within the current position, and the **RNG state** that drives shuffling. Restore all three and the resumed stream is bit-identical to the one that would have run without the interruption. This is the same determinism discipline that makes a run reproducible in the first place, and it is covered in depth from the checkpoint side in [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume).

PyTorch's newer `StatefulDataLoader` (from `torchdata`) makes this a first-class API — it exposes `state_dict()` and `load_state_dict()` on the loader itself, so you save the loader's state in the same checkpoint as the model. If you are rolling your own iterable dataset, you thread the position through explicitly:

```python
# --- Save alongside the model checkpoint ---
ckpt = {
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
    "loader": {
        "epoch": epoch,
        "shard_idx": dataset.shard_idx,       # which shard we are on
        "sample_offset": dataset.offset,      # position within it
        "rng_state": torch.get_rng_state(),   # shuffle RNG
    },
}
torch.save(ckpt, path)

# --- On resume: restore, then fast-forward the stream ---
state = ckpt["loader"]
torch.set_rng_state(state["rng_state"])
dataset.seek(state["shard_idx"], state["sample_offset"])
```

The subtlety in a distributed run is that each rank has its own stream position, so each rank must save and restore *its own* loader state — you cannot restore rank 0's position onto rank 3. With sharded checkpointing each rank writes its own slice, which lines up naturally with per-rank loader state. The failure to watch for is a resume that "works" — loss is fine, no crash — but where the data order silently diverges from the original schedule; that is exactly the kind of bug that produces a small, unexplained quality regression, and the cross-linked checkpoint post walks through diagnosing it.

## The gotchas that only bite at scale

A collection of the failure modes that a single-GPU loader never shows you, each with the fix.

**Workers dying silently.** A worker process that hits an out-of-memory kill, an uncaught decode exception on one corrupt sample, or a segfault in a native library does not always take the main process down with it. Sometimes the DataLoader just hangs waiting for a batch that will never come, and the whole job stalls with no error. Set a `timeout` on the DataLoader so a dead worker raises instead of hanging, wrap `__getitem__`/`__iter__` in try/except to skip-and-log corrupt samples rather than crashing, and remember the host-RAM budget from the throughput section — the most common cause of a "randomly dying worker" is the prefetch buffers OOMing the host.

**The ragged last batch.** If your dataset length is not divisible by `world_size × batch_size`, the final batch is smaller — or worse, some ranks get a final batch and others do not. In a synchronous data-parallel step, a rank with no batch never reaches the all-reduce, and the ranks that do reach it wait forever: a **deadlock at the last step of the epoch**. `drop_last=True` on both the sampler and the loader avoids it by discarding the ragged tail so every rank runs exactly the same number of steps. You lose a fraction of a batch of data per epoch; you gain not deadlocking. For pretraining this is always the right trade.

**Worker seeding.** Each worker forks with the same base RNG state, so if your `__getitem__` uses randomness — augmentation, random cropping, any stochastic transform — all workers produce *identical* "random" augmentations, and worse, they repeat the same sequence every epoch. You need each worker to have a distinct but reproducible seed. Use `worker_init_fn` to reseed from a base seed combined with the worker id *and* the rank, so the randomness is decorrelated across all 64 workers but still reproducible from the run's master seed. This is the same two-part seeding problem — identical model init, decorrelated data randomness — that [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas) covers for the model side.

```python
import numpy as np, random

def worker_init_fn(worker_id):
    rank = dist.get_rank() if dist.is_initialized() else 0
    # Distinct seed per (rank, worker); reproducible from base_seed.
    seed = base_seed + rank * 1000 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

loader = DataLoader(dataset, num_workers=8, worker_init_fn=worker_init_fn, ...)
```

**On-the-fly tokenization as the hidden bottleneck.** Worth repeating because it is the number one cause of text-run starvation: tokenizing during training re-does the same CPU work every epoch and puts the tokenizer squarely on the critical path. Pre-tokenize once, offline, and the problem evaporates. If you *must* tokenize online (rapidly changing preprocessing, or a corpus too big to materialize tokenized), use the fast Rust-backed tokenizer with batched encoding, and set `TOKENIZERS_PARALLELISM=false` so it does not spawn threads inside every worker.

## Measuring data-wait honestly

You cannot fix what you refuse to measure correctly, and the data loader is where naive measurement lies to you the most.

The trap is `torch.cuda.synchronize()`. If you time your step by synchronizing before and after the compute, you measure only the GPU's busy time — and you *hide* the data wait entirely, because the wait happens on the CPU while the GPU is idle, outside your timed region. A microbenchmark that syncs around compute will happily report 45 percent MFU while the real end-to-end run limps at 55. To catch starvation you must time **wall-clock, end to end**, including the `next(loader)` call, and compare it to the pure-compute step time. The gap between them *is* your data wait.

The honest measurement recipe: warm up for a few dozen steps (the first steps include worker spawn, cache cold-start, and CUDA context setup — throw them out), then measure steady-state wall-clock per step and pure-compute per step separately. If wall-clock exceeds compute, the loader is your bottleneck and the ratio tells you by how much. `torch.profiler` makes the wait visible directly — the data wait shows up as a gap on the GPU timeline and as time spent in the DataLoader's `__next__` on the CPU timeline.

```python
from torch.profiler import profile, ProfilerActivity, schedule

prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=5, warmup=5, active=10),  # skip cold start
    record_shapes=False,
)
with prof:
    for step, batch in enumerate(loader):
        x = batch["input_ids"].to(device, non_blocking=True)
        loss = model(x).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        prof.step()
        if step >= 20: break

# Look for a gap on the CUDA stream before each step's first kernel,
# and time attributed to "enumerate(DataLoader)" on the CPU thread.
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
```

The confounds to control, beyond the loader, are the usual suspects for any throughput measurement: thermal and clock throttling (a hot GPU quietly downclocks and your tokens/s sag with no code change), background jobs sharing the node's CPU or NIC, and a cold filesystem cache on the first epoch. Measure steady state, on a quiet node, after warmup — the same discipline the whole series applies to any before-and-after number.

## Case studies and real numbers

A few reference points from real systems, so the design choices here are not just my assertions.

**Megatron-LM's indexed dataset.** Megatron-LM (Shoeybi et al., 2019, and the ongoing Megatron-LM codebase) trains on pre-tokenized data via its `MMapIndexedDataset`: the corpus is tokenized once into a binary blob plus an index of document boundaries, memory-mapped at load, and sliced into training sequences with near-zero per-sample CPU. This is the reference implementation of "pre-tokenize offline, memmap online," and it is why Megatron-scale runs are compute-bound rather than loader-bound. The `.bin` + `.idx` format the whole ecosystem copied comes from here.

**MosaicML's StreamingDataset.** MosaicML built `StreamingDataset` specifically for the terabyte-scale, multi-node streaming case: shards in object storage, deterministic shard-level shuffling with a configurable shuffle buffer, local caching after first fetch, and — notably — elastic-friendly resumption that restores the exact sample position across a different number of nodes than you saved on. Their reported motivation is exactly the problem in this post: at cluster scale the loader must sustain the aggregate token demand of hundreds of GPUs without any node needing the full corpus on local disk.

**WebDataset and the tar-shard pattern.** WebDataset (Aizman et al., the format used across many large vision and multimodal runs) established the `.tar`-shard convention with `split_by_node` and `split_by_worker` as the canonical way to shard an iterable stream correctly across ranks and workers. The reason it exists is precisely the double-sharding duplicate bug: making the two-level split explicit and standard so people stop getting it wrong.

**nanoGPT and the packed `.bin`.** On the small end, Karpathy's nanoGPT reduces the entire data pipeline to two `np.memmap`'d `uint16` files (train and val) and a random-offset slice — no workers, no decode, no tokenization at train time. It is the minimal proof that once data is pre-tokenized and memory-mapped, the loader stops being a bottleneck entirely, even with the simplest possible implementation. The lesson scales: the fanciest streaming stack in the world is just this idea plus the machinery to handle "the file does not fit locally."

**Data echoing, when the loader genuinely cannot keep up.** Google's "Faster Neural Network Training with Data Echoing" (Choi et al., 2019) tackles the case where the input pipeline is the hard bottleneck and you cannot make it faster — for example, decode is expensive and the storage is fixed. Their trick is to *reuse* each batch a few times downstream in the pipeline (echo it) so the GPU is not idle waiting for the next fresh batch, trading a small amount of statistical efficiency per example for a large gain in wall-clock throughput. It is not a first-choice tool — fixing the pipeline is almost always better — but it is a useful reminder that the supply-versus-demand framing has a second escape hatch: if you cannot raise supply, you can sometimes lower the demand for *fresh* data. Reach for it only after pre-tokenizing, adding workers, and moving decode off the critical path have all been exhausted.

## When to reach for each lever (and when not to)

Not every run needs a streaming stack, and over-engineering the pipeline wastes your time instead of the GPUs'. The decision, roughly:

- **If the corpus fits on local NVMe: pre-tokenize and memmap. Stop there.** This is nanoGPT's setup and it is bulletproof. Do not build a streaming pipeline you do not need — object storage adds latency, failure modes, and complexity you only want when you are forced to.
- **If the corpus does not fit locally: stream from object storage** with shard-level shuffling and local caching. This is the MosaicML/WebDataset regime. It is the right answer for multi-terabyte pretraining and the wrong answer for a 50 GB fine-tune.
- **Only raise `num_workers` until supply comfortably exceeds demand, then stop.** The law says $W \ge t_w / t_c$; going far past that just burns host RAM and oversubscribes CPU. More workers is not a virtue, it is a cost you pay to meet a target.
- **Do not tokenize on the fly for repeated-epoch training.** Ever. It is pure recomputation on the critical path. The only defensible online tokenization is a single-pass run over a corpus too large to materialize, and even then, cache aggressively.
- **Do not obsess over `pin_memory` for text.** Token-ID batches are kilobytes; the pinned-copy win is negligible. Save the attention for image/audio/video loaders where the batches are large and the H2D copy is real.
- **If your model is huge and your step is long, the loader is probably not your problem.** A 70B model with a 10-second step gives the loader an eternity to prepare a batch; starvation bites hardest on *small* models, *short* steps, *high* MFU, and *heavy per-sample decode* (vision, audio). Check the tell — utilization dipping to zero — before you spend a day tuning a loader that was never the bottleneck.

The meta-point: the data pipeline is a means to an end, and the end is keeping the GPU busy. Measure the demand, measure the supply, fix the pipeline only as much as the arithmetic says you must, and put the rest of your energy back into the model.

## Key takeaways

- **A waiting GPU is wasted money, and the loader is the most-overlooked reason it waits.** The tell is GPU utilization dipping to zero between steps — a sawtooth on the trace, not a low average.
- **The throughput budget is the whole game:** with overlap, $\text{util} = \min(1, t_c/t_d)$. The loader is free if $t_d \le t_c$ and a pure bottleneck the instant it is not.
- **The number-of-workers law is $W \ge t_w / t_c$** — the single-worker per-batch time divided by the compute step time, rounded up. Measure both and check the inequality before launching.
- **Every rank has its own loader,** so worker counts multiply by GPUs-per-node. Budget `num_workers ≤ (cores − overhead) / GPUs_per_node` and set `OMP_NUM_THREADS=1` to stop workers fighting for cores.
- **Pre-tokenize offline and memory-map.** It turns 4-second batches into 2-millisecond batches and is the highest-leverage change you can make to a text pipeline.
- **Stream from object storage only when the corpus does not fit locally,** with shard-level shuffling, a shuffle buffer for randomness, and shard prefetch to hide network latency.
- **Iterable datasets do not shard themselves.** Shard by `global_id = rank * num_workers + worker_id` out of `world_size * num_workers`, or you feed every example to the model dozens of times with no error to warn you.
- **Use `drop_last=True`** so every rank runs the same number of steps and the last batch never deadlocks the all-reduce.
- **Checkpoint the loader state** (shard index, sample offset, RNG) alongside the model, per rank, or a resumed run silently re-trains on data it already saw.
- **Measure wall-clock end to end, not synchronized compute.** `cuda.synchronize()` around compute hides the exact stall you are hunting.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the series map; the data pipeline is the "run too slow" wall in disguise.
- [Overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) — the profiler signature of an idle GPU and the overlap discipline the loader must respect.
- [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas) — `DistributedSampler`, `set_epoch`, and the two-part seeding problem for the model side.
- [Debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume) — restoring exact state so a resumed run is bit-identical, from the checkpoint side.
- [Sourcing and collecting training data](/blog/machine-learning/training-data/sourcing-and-collecting-training-data) — where the terabytes of raw corpus come from before they ever hit the loader.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist that ties the pipeline back to the whole scaling and debugging frame.
- Megatron-LM (Shoeybi et al., 2019) and the Megatron-LM `MMapIndexedDataset`; MosaicML StreamingDataset docs; the WebDataset format and its `split_by_node`/`split_by_worker` sharding; the PyTorch `DataLoader` and `torchdata` `StatefulDataLoader` documentation.

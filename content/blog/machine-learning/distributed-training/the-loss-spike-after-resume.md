---
title: "The Loss Spike After Resume: When a Checkpoint Restore Isn't Actually a Resume"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Your job crashed at 3am, you restored from the last checkpoint, and the loss jumped from 2.1 to 6.8 before crawling back. Here is exactly which piece of state you forgot to save, why cold Adam kicks the weights off the basin, and how to test that a resume equals an uninterrupted run."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "checkpointing",
    "fsdp",
    "pytorch",
    "optimizer",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 34
---

The page came in at 3:11am. A 7-billion-parameter model, 40,000 steps into a 100,000-step run on a 64-GPU cluster, had died — one node dropped off the InfiniBand fabric and NCCL took the whole job down with it. Annoying, but this is what checkpoints are for. I pulled the latest checkpoint, relaunched the job, watched the ranks rendezvous, saw the first training log scroll by, and went back to bed.

At 3:40am I looked again, because something felt wrong. The loss before the crash had been sitting at **2.1** and dropping slowly. The first logged loss after the resume was **6.8**. Not a NaN, not an error — just a loss more than three times higher than where we left off, as if the model had forgotten a week of training. Over the next few hundred steps it slid back down: 6.8, 5.2, 3.9, 2.9, 2.3, and finally back to 2.1 around step 40,400. Then it kept going as normal, as if nothing had happened.

Nothing crashed. Nothing errored. The run "recovered." But it had just burned roughly 400 steps of 64-GPU compute — call it a couple of GPU-hours and a real slice of the training budget — re-learning something it already knew. And on some runs a spike like that doesn't fully recover: it leaves a permanent dent in the final loss. That is the bug in this post. A checkpoint restore is not automatically a resume. A **resume** means the run continues as if the interruption never happened — ideally bit-for-bit. A **restore** that only reloads the model weights is a different, worse thing wearing the same clothes.

![Two loss curves side by side showing a weights-only restore spiking from 2.1 to 6.8 and recovering over 400 steps versus a full-state restore that continues smoothly with no spike](/imgs/blogs/the-loss-spike-after-resume-1.webp)

The figure above is the whole post in one picture. The left curve is what I saw at 3:40am. The right curve is what a correct resume looks like: the loss at step 40,001 is 2.09, indistinguishable from an uninterrupted run, and there is no spike to recover from. By the end of this post you will know exactly which pieces of state separate those two curves, why the single most common omission — the optimizer state — produces a spike of *precisely* this shape, how to catch the bug with a five-line test instead of a 3am page, and how the whole problem changes (and gets worse) once your checkpoint is sharded across ranks with FSDP.

This sits squarely on the [four walls](/blog/machine-learning/distributed-training/why-distributed-training) frame that runs through this whole series: your run is too slow and too expensive to throw away 400 steps every time a node hiccups, and at 64 GPUs nodes hiccup *often*. Getting resume right is not a nicety — it is a load-bearing part of finishing a large run at all. If you have not read the intro, [why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) sets up the vocabulary (rank, world size, shard) I lean on below, and the [capstone playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) collects the checklist this post feeds into.

## 1. The scene: a resume that wasn't a resume

Let me reconstruct the crime scene, because the shape of the evidence tells you almost everything.

A training step is not just a set of weights. It is a set of weights plus a large amount of *momentum* — in the physics sense and in the literal Adam sense. The optimizer has spent 40,000 steps building up a per-parameter picture of "which direction is this weight consistently moving, and how noisy is its gradient?" The learning-rate scheduler is 40% of the way down a cosine decay. The data loader is somewhere in the middle of epoch 3, having shuffled the dataset with a particular seed and consumed exactly 5.1 million samples. Dropout masks and data-augmentation choices are being drawn from a particular random-number stream. In fp16 runs, the gradient scaler has settled on a loss-scale value of 4096 after a long series of overflow-driven halvings and doublings.

All of that is *state*. It is every bit as much a part of "where the run is" as the weights are. When you save a checkpoint that contains only `model.state_dict()`, you have saved maybe a fifth of the state and thrown away the rest. The restore reloads the weights perfectly — bit-for-bit, no complaints — and then re-creates everything else *from scratch*, as if step 40,000 were step zero of a brand new run that happens to start from unusually good weights.

That mismatch is the entire bug. The model is at step 40,000. The optimizer thinks it is at step 1. Those two facts are incompatible, and the loss spike is the sound of them colliding.

Here is the tell that lets you diagnose it in thirty seconds, before you understand any of the mechanism: **the shape of the recovery.** A spike from a corrupted weight tensor would not recover — it would diverge or stay broken. A spike from a bad data batch would be a single-step blip. But a spike that jumps high on step one and then decays smoothly over a few hundred steps back to *exactly* the pre-crash loss is the fingerprint of a cold optimizer re-warming its statistics. Once you have seen this shape once, you recognize it instantly. Most of the rest of this post is earning the right to say that sentence with confidence.

There is a second, sneakier failure mode hiding in the same scene, and it does *not* announce itself with a spike. If your data loader restarts from the beginning of its shard list instead of resuming at sample 5.1 million, the loss might look *fine* — even suspiciously good — because the model is being re-fed data it has already memorized. No spike, no error, just a quiet bias toward the early part of your dataset that shows up only as a slightly worse final model. That one is genuinely dangerous precisely because it is invisible. We will come back to it.

## 2. What actually lives in a training step's state

Before we derive anything, let's catalog what a *complete* checkpoint has to contain. This is the list I now paste into every code review, because every single item on it is something I have personally watched someone forget.

![A vertical stack showing the seven layers of state in a complete checkpoint from model weights on top down through optimizer state scheduler sampler RNG grad scaler and shard metadata](/imgs/blogs/the-loss-spike-after-resume-2.webp)

The stack, from the piece everyone remembers down to the piece almost everyone forgets:

1. **Model weights** — `model.state_dict()`. For a 7B model in bf16 this is about 14 GB. Everyone saves this. It is necessary and radically insufficient.
2. **Optimizer state** — `optimizer.state_dict()`. For Adam/AdamW this holds the first moment `m` (momentum) and second moment `v` (variance), both in fp32, per parameter. That is roughly `2 × 4 bytes × 7B ≈ 56 GB` — *four times larger than the weights themselves*. This is the single most important thing to restore and the single most commonly dropped. The size is exactly why people drop it: it makes checkpoints huge, so someone "optimizes" the checkpoint by saving weights only, and the loss spike is the invoice.
3. **LR scheduler state** — the step counter and any internal state of your `LRScheduler`. A cosine or warmup-then-decay schedule is a pure function of the step count; if the step count resets to zero, you restart warmup and slam a converged model with the peak learning rate.
4. **Data sampler / epoch / position** — the epoch number, the shuffle seed, and *how far into the current epoch you are*. For a `DistributedSampler` this is the epoch you passed to `set_epoch`; for a streaming dataset it is a resumable position token.
5. **RNG state** — the states of the PyTorch CPU generator, every CUDA device generator, and NumPy (and Python's `random` if you use it). This controls dropout masks, data augmentation, and any stochastic layer. Skip it and your regularization is drawn from a different stream after the resume.
6. **Gradient scaler state** — for fp16 (not bf16) training, `torch.cuda.amp.GradScaler` carries the current loss scale and the growth counter. Reset it and you re-enter the overflow/halving dance, which can produce a burst of skipped steps or even a NaN cascade right after resume.
7. **Sharding metadata** — for FSDP/DeepSpeed sharded checkpoints, the world size, the per-shard offsets, dtypes, and the device mesh. Without this, the bytes on disk are just an undifferentiated pile that only makes sense if you reload onto the *exact* same cluster shape.

Notice the ordering by how-often-forgotten. Weights: never forgotten. Optimizer: forgotten constantly, catastrophic. Scheduler: forgotten often, catastrophic. Sampler and RNG: forgotten almost always, and the damage is subtle rather than loud. Scaler: forgotten by everyone not using fp16, harmless for them, a NaN factory for fp16 users. Metadata: only relevant once you shard, and then it is the whole ballgame.

Let me lay out the same seven items as a failure matrix, because the point I most want to land is that these are **distinct bugs with distinct symptoms** — not one fuzzy "resume is flaky" problem.

![A matrix with one row per state object showing what happens if you skip it the symptom on resume and where that state lives in the code](/imgs/blogs/the-loss-spike-after-resume-3.webp)

The matrix is your triage table. If the loss *spikes*, suspect the optimizer or the scheduler — the two rows whose "if you skip it" column produces a large, immediate perturbation. If the loss is *quietly worse* with no spike, suspect the sampler or the RNG — the rows whose damage is statistical, not dynamical. If you get a NaN *burst* seconds after resume on an fp16 run, suspect the grad scaler. If you get a load-time *error* or a shard-shaped garbage tensor, suspect the sharding metadata. Four different symptoms, four different columns to look at. Now let's earn each one, starting with the loud one.

## 3. The number-one culprit: cold Adam and the mechanism of the spike

Here is the claim I need to make precise: **restoring weights but not the optimizer state makes Adam take steps that are far too large for the first few hundred steps, which kicks the weights off the loss basin, and the loss spikes.** People say this all the time, and it is *true*, but the naive version of the argument is wrong, and if you believe the wrong version you will draw the wrong conclusions. So let's do it carefully.

Recall the AdamW update. At step $t$, with gradient $g_t$ for a given parameter, decay rates $\beta_1, \beta_2$, learning rate $\eta$, and epsilon $\epsilon$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The quantity that matters is the **effective step size** per parameter: $\eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$. In steady state, after tens of thousands of steps, $\hat{v}_t$ has converged to an exponential moving average of $g^2$ — roughly $\mathbb{E}[g^2]$ for that parameter — and $\hat{m}_t$ to roughly $\mathbb{E}[g]$. For a parameter whose gradient is consistent and modest, $\hat{m}/\sqrt{\hat{v}} \approx 1$ and the step is about $\eta$. Fine.

Now here is the part the naive argument gets wrong. If you re-initialize the optimizer, $t$ resets to 1, and *because of bias correction* the very first step is **not** obviously enormous. Plug in $t=1$: $m_1 = (1-\beta_1)g_1$, so $\hat{m}_1 = m_1/(1-\beta_1) = g_1$; and $v_1 = (1-\beta_2)g_1^2$, so $\hat{v}_1 = v_1/(1-\beta_2) = g_1^2$. Then $\hat{m}_1/\sqrt{\hat{v}_1} = g_1/|g_1| = \text{sign}(g_1)$, and the step is $\eta \cdot \text{sign}(g_1)$ — magnitude $\eta$, same as steady state. So if the first step is bounded by $\eta$, where does a spike from 2.1 to 6.8 come from?

The answer is that **$\eta$ is the wrong baseline.** The right baseline is *the step this particular parameter should be taking*, and for a large fraction of parameters that is far smaller than $\eta$.

Think about what Adam's second moment actually buys you. For a parameter with a **large, noisy gradient** — a high-variance weight, common in embeddings, LayerNorm gains, and the last layer — the warm $\hat{v}$ is large, so the effective step $\eta \cdot \hat{m}/\sqrt{\hat{v}}$ is *much smaller* than $\eta$. Adam has learned to damp that parameter, to tiptoe. When you cold-restart, $\hat{v}$ collapses to a fresh estimate built from a single minibatch, the damping vanishes, and that parameter's step jumps *back up toward the full $\eta$* — which for it might be 10x or 100x too large. Multiply that across the millions of parameters Adam had learned to slow down, all of them suddenly moving at full speed in a poorly-coordinated direction, and the weight vector gets displaced a long way off the low-loss manifold it was sitting on. The loss spikes.

![A left-to-right timeline showing the second moment starting at zero the effective step becoming too large the weights getting kicked off the basin the loss peaking and then the variance re-accumulating so the loss returns to 2.1](/imgs/blogs/the-loss-spike-after-resume-4.webp)

The timeline traces the full arc. And crucially, it tells you the *recovery timescale*, which is the other half of the fingerprint. The second moment $v$ is an exponential moving average with decay $\beta_2$; it forgets its initialization and converges to the true $\mathbb{E}[g^2]$ over roughly $1/(1-\beta_2)$ steps. For the LLM-typical $\beta_2 = 0.95$ that is about 20 steps; for the older default $\beta_2 = 0.999$ it is about 2,000 steps. Real recoveries land in between — a few tens to a few hundred steps — because the loss also has to physically climb back down the basin after being kicked, not just wait for $\hat{v}$ to converge. That is exactly the 400-step recovery I watched at 3:40am. The spike magnitude is set by how far the un-damped step kicks the weights; the recovery *length* is set by $\beta_2$. Both match. That is why I can say "cold optimizer" from the curve shape alone.

One more subtlety worth stating because it changes your mitigation. Some Adam implementations apply the update *without* full bias correction, or with a fused kernel that behaves slightly differently in the first steps. In the uncorrected form, at $t=1$ with $v=0$ the raw ratio is $(1-\beta_1)/\sqrt{1-\beta_2} \cdot \text{sign}(g) \approx 0.1/0.0316 \approx 3.16$ for $\beta_1=0.9, \beta_2=0.999$ — so the first step is ~3x oversized even before the un-damping argument. Either way the conclusion is the same: **a cold optimizer takes mis-scaled steps, and the fix is to never let it go cold.** Restore the optimizer state.

#### Worked example: the price of a weights-only resume

Let me put numbers on the invoice. Take the run from the intro: a 7B model, 64 × A100 80GB, bf16, roughly 3,000 tokens/s/GPU at ~40% MFU, so about 192,000 tokens/s cluster-wide. Say a step is 2 million tokens (global batch), so ~10.4 seconds per step.

- **Weights-only restore.** Loss spikes 2.1 → 6.8, recovers over ~400 steps. Those 400 steps at 10.4 s/step ≈ **69 minutes** of wall clock on 64 A100s ≈ **74 GPU-hours**. At a representative cloud rate of \$1.50 per A100-hour that is roughly **\$110 of pure waste per crash** — and large runs crash *many* times. Ten crashes over a run is \$1,100 and about 12 hours of schedule slip, for a bug whose fix is three extra lines in the save function. Worse, if the spike leaves a residual dent, you may pay again in final quality.
- **Full-state restore.** Loss at 40,001 is 2.09. Zero recovery steps. Zero wasted GPU-hours. The resume is invisible in the loss curve.

That is the entire economic argument for this post in one example: the fix costs three lines and pays back tens to hundreds of dollars *every time a node sneezes*, which at 64 GPUs is often. If you cross-reference the [cost-and-efficiency](/blog/machine-learning/distributed-training/the-distributed-training-playbook) thinking from the capstone, a weights-only checkpoint is one of the highest-ROI-to-fix bugs in the entire distributed stack.

## 4. The other five ways a resume silently diverges

The optimizer is the loud one. The rest of the stack fails more quietly, and quiet failures are the ones that survive to production. Let's go through them.

### 4.1 The LR scheduler and step count

This one is nearly as bad as the optimizer and often travels with it. Your learning rate is a function of the step: warmup for the first few thousand steps, then cosine or linear decay. If you restore weights but let the scheduler start at step 0, you re-enter warmup — the learning rate climbs from near-zero back up to the *peak*, and then that peak LR is applied to a model that has already converged 40% of the way. A big LR on a well-fit model is a recipe for a spike that looks a lot like the cold-optimizer spike but has a different cause.

The fix is to restore `scheduler.state_dict()`, or — if your scheduler is stateless and purely a function of a global step counter you track yourself — to restore that counter and call `scheduler.step()` the right number of times (or construct the scheduler with `last_epoch=global_step`). The failure signature: check whether your logged learning rate right after resume matches the LR right before the crash. If it jumped back up, you found it.

### 4.2 The data order, sampler position, and streaming state

Now the invisible one. In an ideal resume, the model at step 40,001 sees the *next* batch it would have seen — sample 5,100,001 onward — not a batch it has already trained on. Getting this right depends on your data plumbing.

For a **map-style dataset** with a `DistributedSampler`, the order within an epoch is determined by the epoch number and the shuffle seed. If you never call `set_epoch(epoch)` on resume, or you call it with the wrong epoch, you replay a shuffle you have already consumed. Worse, most training loops don't track the *within-epoch* index at all: they resume at the *start* of the current epoch, silently re-feeding up to a full epoch of already-seen data.

For a **streaming / iterable dataset** (WebDataset, Mosaic Streaming, a token-packed shard reader), there is no random access; you consume shards in order. If the reader restarts at shard 0, the model re-trains on the earliest shards. On a big corpus this biases the model toward whatever happens to sort first — often a particular data source — and you will never see it in the loss curve. You will see it, faintly, in a worse eval weeks later.

#### Worked example: the data-order bug that has no spike

Here is the scenario that scares me more than the loss spike, because nothing pages you. A team trains a 3B model on a 1.2-trillion-token corpus streamed from 12,000 shards, ordered so that shards 0–2,000 are high-quality web text and the tail is a long, more diverse mix. The job crashes and restarts roughly every ~6 hours (spot preemption). Each restart, the streaming reader restarts at shard 0. The loss curve is *beautiful* — arguably better than a clean run, because the model keeps getting re-fed the cleanest, easiest early shards. But over 40 restarts the model has seen shards 0–2,000 perhaps 40 times and the tail shards once or not at all. The final model is quietly overfit to the head of the corpus and underexposed to the tail; downstream, it is worse on exactly the diverse tasks the tail was meant to cover. Nobody notices for a month. The fix is one field in the checkpoint — the reader's resumable position token — and one line to seek to it on resume. This is the bug I most want you to remember, because it is the one that costs the most and screams the least. It composes directly with [determinism across ranks](/blog/machine-learning/distributed-training/determinism-across-ranks): a resume that is not deterministic in its data order is not reproducible, full stop.

### 4.3 The RNG state: dropout and augmentation

Dropout draws a fresh mask every forward pass from a PRNG stream. Data augmentation (crops, flips, mixup) draws from one too. If you don't restore the PyTorch CPU generator, every CUDA device generator, and NumPy's generator, then after resume you are drawing masks and augmentations from a *different* stream than an uninterrupted run would. This will not spike the loss — it is not a big perturbation — but it *does* mean your resumed run is a different random experiment than the one that would have happened. It breaks bit-for-bit reproducibility, and it very mildly perturbs regularization. For most runs this is a small effect; for a paper claiming exact reproducibility, or for a run where you are bisecting a NaN and need determinism, it is the difference between a debuggable job and an un-debuggable one. See [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) for how deep this rabbit hole goes.

### 4.4 The gradient scaler (fp16 only)

If you train in true fp16 (not bf16), you use dynamic loss scaling: a `GradScaler` multiplies the loss by a large factor before backward so small gradients don't flush to zero, then unscales before the optimizer step, halving the scale on overflow and slowly doubling it otherwise. After tens of thousands of steps the scale has settled at some stable value (say 4096). If you reset the scaler to its default (often 65536) on resume, you re-enter the overflow dance: the first few steps overflow, get skipped, the scale halves repeatedly, and in the worst case you get a burst of skipped steps or a NaN cascade right after resume. Restore `scaler.state_dict()`. Note this is a non-issue for bf16, which has the dynamic range to skip loss scaling entirely — one of several reasons [mixed-precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale) prefers bf16 when the hardware supports it.

### 4.5 Sharded and distributed checkpoint mismatch

This one only shows up once your state is *sharded* across ranks, but then it dominates everything else, so it gets its own full section below. The one-line version: an FSDP checkpoint that stored one shard per rank on an 8-rank job cannot be blindly reloaded onto a 4-rank job. Hold that thought for section 7.

## 5. The correct recipe: save and restore everything, atomically

Enough diagnosis. Here is the complete, copy-and-adapt save/load that captures the whole stack. This is the single most valuable code block in the post, so it is worth reading line by line.

```python
import os
import torch
import numpy as np
import random

def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    sampler, epoch, global_step, samples_seen):
    """Write a complete, resumable checkpoint atomically."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),          # Adam m, v  <-- the big one
        "scheduler": scheduler.state_dict(),          # step count / warmup position
        "scaler": scaler.state_dict() if scaler is not None else None,  # fp16 loss scale
        "epoch": epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,                 # data-loader position
        "sampler_seed": getattr(sampler, "seed", None),
        # RNG streams: CPU, all CUDA devices, NumPy, Python
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        # metadata so a human (and a loader) can sanity-check the shape
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "format_version": 3,
    }
    # ATOMIC WRITE: never overwrite the good checkpoint in place. A crash
    # mid-write must not corrupt the last known-good file.
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)   # atomic on POSIX: rename is all-or-nothing

def load_checkpoint(path, model, optimizer, scheduler, scaler, map_location):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])      # restore m, v -> no cold start
    scheduler.load_state_dict(ckpt["scheduler"])      # restore LR position
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    torch.set_rng_state(ckpt["rng_cpu"])
    torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_python"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["samples_seen"]
```

Three details in that block are load-bearing and routinely botched:

**Atomicity.** Note the `torch.save(tmp)` then `os.replace(tmp, path)`. If the job dies *while writing the checkpoint* — which happens more than you would think, because checkpoint writes are I/O-heavy moments when the cluster is under stress — an in-place `torch.save(path)` leaves you with a half-written, corrupt file *and* you have already overwritten the last good one. Writing to a temp file and atomically renaming means a crash mid-write leaves the previous good checkpoint intact. This is [distributed checkpointing](/blog/machine-learning/distributed-training/distributed-checkpointing) hygiene 101, and it has saved runs I care about.

**The optimizer really is in there.** `optimizer.state_dict()` for a fresh optimizer that has never stepped is *empty* — the state dict only populates after the first `.step()`. So if you accidentally save the checkpoint before the first step, or if you construct a new optimizer and never verify its state is non-empty, you can "save the optimizer" and still resume cold. We will assert against exactly this in the next section.

**The data position is data-format-specific.** Saving `samples_seen` is necessary but not sufficient — you have to actually *use* it on resume. For a map-style sampler:

```python
# On resume: fast-forward the epoch's shuffle to the right position.
sampler.set_epoch(epoch)                 # reproduce THIS epoch's shuffle order
loader = DataLoader(dataset, sampler=sampler, batch_size=bs,
                    num_workers=8, pin_memory=True)
it = iter(loader)
skip_batches = samples_seen % len(sampler) // bs   # within-epoch offset
for _ in range(skip_batches):            # fast-forward past seen data
    next(it)
# training continues from the correct batch
```

Fast-forwarding by iterating is wasteful (you pay to decode-and-drop the skipped batches), which is one reason **streaming datasets with native resumability** are worth the switch at scale. A well-designed streaming reader stores a position token and seeks directly:

```python
# Streaming / iterable dataset with a resumable position token.
# (WebDataset/Mosaic-style API; the exact call differs per library.)
stream = StreamingDataset(shards="s3://corpus/{00000..11999}.tar",
                          shuffle=True, seed=42)
if resuming:
    stream.load_state_dict(ckpt["stream_state"])   # seeks to the exact token
# and on save:
ckpt["stream_state"] = stream.state_dict()         # {shard_idx, sample_idx, epoch, seed}
```

The principle across both cases: **the data pipeline must be able to answer "where was I?" and seek there.** If it can't, you have the invisible data-order bug from section 4.2 baked in, and no amount of optimizer-state care will save you from it. This is the resume side of [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale).

## 6. Proving it: resume-equals-continuation as a test

Here is the part that separates people who *hope* their resume works from people who *know* it does. You do not need a 3am page to discover a broken resume. You can catch it in a two-minute unit test, and you should run that test in CI on every change to your checkpoint code.

The idea is a direct operationalization of the definition. A resume is correct if and only if the resumed run equals the uninterrupted run. So: take one step from a checkpoint two ways and compare.

![A branching and merging graph where one checkpoint at step N forks into an uninterrupted path and a restore path each producing a next-step loss that merge into an assertion that the two match](/imgs/blogs/the-loss-spike-after-resume-5.webp)

The figure is the test. In code:

```python
def test_resume_equals_continuation(build_trainer, tol=1e-4):
    """Save at step N, then assert the restored next-step loss matches
    the uninterrupted run's next-step loss."""
    torch.manual_seed(0)
    trainer = build_trainer()

    # Run to step N and snapshot.
    for _ in range(N):
        trainer.train_one_step()
    save_checkpoint("ck.pt", trainer.model, trainer.optimizer,
                    trainer.scheduler, trainer.scaler, trainer.sampler,
                    trainer.epoch, trainer.global_step, trainer.samples_seen)

    # Path A: keep running one more step, uninterrupted.
    loss_A = trainer.train_one_step()

    # Path B: fresh trainer, restore, run one step.
    trainer_B = build_trainer()
    load_checkpoint("ck.pt", trainer_B.model, trainer_B.optimizer,
                    trainer_B.scheduler, trainer_B.scaler,
                    map_location="cuda")
    loss_B = trainer_B.train_one_step()

    assert abs(loss_A - loss_B) < tol, (
        f"resume diverged: A={loss_A:.6f} B={loss_B:.6f} "
        f"delta={abs(loss_A - loss_B):.2e}")
```

If your checkpoint is correct, `loss_A` and `loss_B` agree to within floating-point noise (with full determinism enabled, bit-for-bit; in practice a tiny `tol` absorbs non-deterministic kernel reductions). If you forgot the optimizer state, `loss_B` will be dramatically higher — you will catch the exact 3am bug in a green-vs-red test. This is the same discipline as [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume) in the debugging series, made concrete for the distributed case.

Two cheaper checks belong in the same test file, because they catch the two most common omissions before you even run a step:

```python
# 1) The optimizer state must be non-empty. A fresh optimizer that has
#    never stepped has an EMPTY state dict -- saving it resumes cold.
ck = torch.load("ck.pt", weights_only=False)
assert len(ck["optimizer"]["state"]) > 0, "optimizer state is empty!"

# 2) The scheduler must resume at the right step, not zero.
assert ck["scheduler"]["last_epoch"] == ck["global_step"], \
    "scheduler step count does not match global step -- warmup will restart"
```

When a resume misbehaves in the wild, this is the decision tree I walk. It routes the symptom to exactly one object to inspect.

![A decision tree routing a post-resume loss spike into either state that was never saved or state loaded into the wrong shape with the specific object to check on each leaf](/imgs/blogs/the-loss-spike-after-resume-6.webp)

Read it top-down. The spike is either **missing** state (something is empty in the checkpoint — cold Adam, scheduler at step 0, RNG never captured) or **mismatched** state (something is present but restored into the wrong shape or position — sampler at index 0, world size changed). The two branches are diagnosable in seconds: `print(len(ck["optimizer"]["state"]))`, `print(ck["scheduler"]["last_epoch"])`, `print(ck["samples_seen"])`, `print(ck["world_size"])`. Whichever comes back wrong is your bug. You almost never need to reason harder than that once you have the tree.

## 7. Sharded checkpoints and the world-size trap

Everything above assumed a single, complete `state_dict` that lives in one place. The moment you shard — FSDP, DeepSpeed ZeRO — that assumption breaks, and a whole new failure class opens up. This is where a lot of otherwise-careful teams get bitten, because their single-GPU checkpoint code was correct and they assume it scales. It does not.

Under FSDP with `ShardingStrategy.FULL_SHARD`, each rank physically holds only *its slice* of the flattened parameter buffer and its slice of the optimizer state. There is no rank that has the whole model. So "save the state dict" has two very different meanings, and choosing wrong is the bug.

![A device mesh grid with one cell per rank across two nodes each cell holding one eighth of the parameter and optimizer buffer connected across the InfiniBand link between nodes](/imgs/blogs/the-loss-spike-after-resume-7.webp)

The grid shows the layout: an 8-GPU, 2-node FSDP job where each rank owns one-eighth of the flat parameter-plus-optimizer buffer. There are two ways to checkpoint this, and you must pick deliberately:

**Full (consolidated) state dict.** You gather all shards onto rank 0 (or every rank) and save one unsharded checkpoint that looks exactly like a single-GPU checkpoint. This is portable — you can reload it onto *any* world size, or onto one GPU for eval — but the gather is expensive in memory (rank 0 must briefly hold the whole model) and time, and it does not scale to models too big to fit on one rank.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig)

# Gather everything onto rank 0 as one unsharded checkpoint (portable, pricey).
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
opt_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                          save_policy, opt_policy):
    model_sd = model.state_dict()
    optim_sd = FSDP.optim_state_dict(model, optimizer)  # reshapes to full, unflattened
if dist.get_rank() == 0:
    torch.save({"model": model_sd, "optimizer": optim_sd, ...}, path)
```

**Sharded (distributed) checkpoint.** Each rank writes its own shard, and the checkpoint records per-shard metadata — offsets, shapes, dtypes, and the mesh. This is fast (every rank writes in parallel, no gather) and scales to any model size, but the naive version bakes in the world size. The modern tool is `torch.distributed.checkpoint` (DCP), which stores enough metadata to **reshard on load** — you can save on 8 ranks and load on 4, and DCP re-slices for you:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict)

# SAVE: every rank writes its shard in parallel, with resharding metadata.
model_sd, optim_sd = get_state_dict(model, optimizer)
dcp.save({"model": model_sd, "optimizer": optim_sd},
         checkpoint_id="ckpt/step_40000")

# LOAD onto a possibly-different world size: DCP re-slices per the new mesh.
model_sd, optim_sd = get_state_dict(model, optimizer)  # empty, correct shapes
dcp.load({"model": model_sd, "optimizer": optim_sd},
         checkpoint_id="ckpt/step_40000")
set_state_dict(model, optimizer, model_state_dict=model_sd,
               optim_state_dict=optim_sd)
```

The trap, stated plainly: **a raw per-rank `torch.save(model.state_dict())` under `LOCAL_STATE_DICT` is only reloadable onto the exact same world size, with the exact same wrapping/auto-wrap policy, and the exact same sharding strategy.** Change your job from 8 GPUs to 4 after a cluster reconfiguration, reload that checkpoint, and you get either a hard shape-mismatch error (if you are lucky) or — the nightmare — a silent load where shard boundaries no longer line up, the optimizer moments attach to the wrong parameters, and the loss spikes for reasons that look like the cold-Adam bug but are not. You will chase the optimizer for hours. The actual cause is that shard 3 of an 8-way split is not the same slice of the flat buffer as shard 1 of a 4-way split.

The rule that keeps you out of trouble: **use a distributed checkpoint format that stores resharding metadata (DCP), or consolidate to a full state dict, and never rely on raw local shards surviving a world-size change.** If you must reload sharded, assert the world size matches — the `world_size` field we saved in section 5 exists exactly so this check is one line. This is the deep end of [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice), and it composes with [distributed checkpointing](/blog/machine-learning/distributed-training/distributed-checkpointing) in the reliability track.

#### Worked example: 8-GPU save, 4-GPU restore

A team saves an FSDP sharded checkpoint from an 8-GPU job using per-rank local shards (no resharding metadata). A week later a node is down for maintenance, so they relaunch on 4 GPUs to keep training. The checkpoint has 8 shard files; the 4-rank job's FSDP wrapping expects 4 shards. With DCP this just works — the loader reshards 8→4 transparently. With raw local shards, the best case is a loud `RuntimeError` about mismatched tensor sizes at load, which costs an afternoon but no compute. The worst case, seen when the flatten boundaries happen to be compatible in size but not in *content*, is a silent load where optimizer moments are permuted relative to their parameters: the run resumes, the loss spikes 2.1 → 5-ish, and someone spends a day re-checking Adam before realizing the checkpoint format, not the optimizer, is the culprit. Prevention is one decision made once: save with `torch.distributed.checkpoint`, not with per-rank `torch.save`.

## 8. Measuring it honestly: how you confirm the fix worked

You do not get to *believe* the resume is fixed; you get to *measure* it. Two measurements, one micro and one macro.

The **micro** measurement is the resume-equals-continuation test from section 6, run in CI. It is fast (a few steps on a tiny model), deterministic if you enable deterministic kernels, and it fails loudly the instant someone drops a state object. Treat a red test as a broken build.

The **macro** measurement is the loss curve across a real, deliberately-induced resume. Save a checkpoint at step N on a real run, kill the job, resume, and overlay the resumed loss on the pre-crash trajectory. The correct picture is *no visible discontinuity*. Be careful to measure honestly here — the same discipline the rest of this series harps on:

- **Warm up before you trust the number.** The first couple of steps after any (re)start pay one-time costs — CUDA caching-allocator warmup, cuDNN autotuner, NCCL channel setup. A tiny bump on step one that vanishes by step three is *not* the resume bug; it is warmup. The cold-Adam spike is hundreds of steps long and monotonically decaying, not a two-step blip.
- **Synchronize before timing.** If you are also measuring throughput to confirm the resume didn't wreck your data pipeline, call `torch.cuda.synchronize()` before reading the clock, and compare *steady-state* tokens/s, not the loader-starved first few steps.
- **Watch for the loader confound.** A resumed run whose data loader is refilling worker prefetch buffers can look slower for a few steps. Distinguish "slow because warming up" from "slow because the fix broke prefetching."

Here is the honest before→after, on named hardware, for the intro's run.

| Metric | Weights-only restore | Full-state restore (DCP) |
| --- | --- | --- |
| Hardware | 64 × A100 80GB SXM | 64 × A100 80GB SXM |
| Loss at step 40,001 | 6.8 | 2.09 |
| Steps to re-reach 2.1 | ~400 | 0 |
| Wasted wall-clock per crash | ~69 min | ~0 min |
| Wasted GPU-hours per crash | ~74 | ~0 |
| Wasted \$ per crash (@ \$1.50/GPU-hr) | ~\$110 | ~\$0 |
| Checkpoint size (7B) | ~14 GB (weights only) | ~70 GB (weights + Adam) |
| Save time (sharded, parallel) | n/a | seconds, all ranks write |
| Final eval quality risk | residual dent possible | none from resume |

The one honest cost of doing it right is checkpoint *size*: a full-state checkpoint is ~5x bigger because the Adam moments dominate. That is real disk and real write bandwidth. But it is the correct trade — you are spending gigabytes of disk to save tens of GPU-hours per crash, and disk is thousands of times cheaper than A100 time. If checkpoint size genuinely hurts, the answer is sharded parallel writes (DCP) and less-frequent checkpoints, *not* dropping the optimizer state.

## 9. Case studies: loss spikes in real large-model logbooks

This is not a boutique problem. The published training logbooks of frontier models are full of it, and reading them is the fastest way to calibrate what "normal" looks like at scale.

**OPT-175B (Meta, 2022).** Meta released the actual chronicle of training OPT-175B, and it is a litany of exactly this class of problem: dozens of restarts from checkpoints due to hardware failures and loss divergences, hand-management of the optimizer and learning rate across restarts, and lowered learning rates after spikes to regain stability. The document is worth reading in full precisely because it shows that even a well-resourced team treats checkpoint-and-resume as a first-class, hands-on part of training, not an afterthought. Restarts were frequent enough that the *quality of the resume* directly shaped the final model.

**PaLM (Google, 2022).** The PaLM paper reports roughly 20 loss spikes during training and describes a mitigation that is a direct cousin of this post's data-order point: they restarted from a checkpoint taken about 100 steps *before* each spike and **skipped the data batches** implicated in it. Two things fall out of that. First, it confirms that *which data you see after a resume matters* — they deliberately changed the post-resume data order to avoid re-triggering a spike, which only works if your data pipeline can be positioned deterministically. Second, it shows the standard operational move: keep enough checkpoint history to rewind *past* a spike, not just to the latest step.

**Megatron / BLOOM-scale runs.** Public reports from large Megatron-DeepSpeed runs (including the BLOOM 176B training) describe both instability spikes and the operational machinery around sharded checkpointing across large world sizes — the exact regime where the section-7 world-size trap is a live risk and distributed checkpoint formats earn their keep. The recurring lesson across all of these is not any single trick; it is that at scale, *resume is part of the training algorithm*, and teams that treat it casually pay for it in wasted compute and degraded models.

I have quoted these numbers as *approximate and as reported* — the exact spike counts and step offsets vary by source and I would rather you trust the shape of the story than a specific digit. The shape is: everybody hits this, the good teams instrument and test their way out of it, and the mechanism is always some piece of state that didn't survive the restore.

## When to reach for this (and when not to)

Every safeguard is a cost, so here is the decisive version of when each piece is worth it.

- **Always restore the optimizer and scheduler state.** There is no run, at any scale, where restarting Adam cold and re-running warmup is acceptable. This is not a trade-off; it is a correctness requirement. If your checkpoint saves weights only, it is broken, full stop.
- **Always save data position; how hard you work at it scales with corpus heterogeneity.** On a small, homogeneous, fully-shuffled dataset that you re-shuffle every epoch, restarting at an epoch boundary is a minor sin. On a large, *ordered or heterogeneous* streaming corpus, exact position resume is mandatory — the data-order bias is invisible and permanent. Match the effort to how much your data ordering matters.
- **Restore RNG state when you need reproducibility or are debugging; skip the fuss otherwise.** For a production run where you only care about final quality, mildly different dropout masks after a resume are in the noise. For a paper claiming bit-for-bit reproducibility, or a run where you are bisecting a NaN across ranks, RNG restore is non-negotiable.
- **Use a distributed checkpoint format (DCP) the moment you shard.** Below the sharding threshold — a model that fits with plain DDP — a single consolidated state dict is simpler and fine. The instant you are on FSDP or ZeRO across many ranks, switch to a resharding-capable format. Do *not* hand-roll per-rank `torch.save` and hope; that is the world-size trap waiting to spring.
- **Don't over-engineer a run that never gets interrupted.** If you train small models to completion on one node in an hour and never resume, an elaborate checkpoint stack is wasted effort — save weights, ship, move on. The whole apparatus in this post earns its cost precisely in proportion to *how often you actually resume*, which is why it is a distributed-training concern: at 64 GPUs across many nodes, you resume constantly, and every one of those resumes is a chance to spike.

## Key takeaways

1. **A restore is not a resume.** A resume means the run continues as if the interruption never happened — ideally bit-for-bit. Reloading only the weights is a strictly worse thing wearing the same name.
2. **The optimizer state is the big one.** For Adam it is ~4x the size of the weights, it is the most-commonly-dropped item, and dropping it un-damps the parameters Adam had learned to slow, kicking the weights off the basin. That is the loss spike.
3. **The recovery shape is a fingerprint.** A jump on step one that decays smoothly over a few hundred steps back to *exactly* the pre-crash loss is cold Adam re-warming its second moment over ~$1/(1-\beta_2)$ steps. Diagnose it from the curve alone.
4. **Quiet failures are the dangerous ones.** A dropped optimizer spikes loudly and recovers; a dropped data position causes no spike, silently re-feeds early data, and biases the final model in a way you find weeks later, if ever.
5. **Save the whole stack, atomically.** Weights, optimizer, scheduler, scaler (fp16), data position, and all RNG streams — written to a temp file and atomically renamed so a crash mid-write never corrupts your last good checkpoint.
6. **Test resume-equals-continuation in CI.** Save at step N, take one uninterrupted step and one restored step, assert the losses match to tolerance. Add cheap asserts that the optimizer state is non-empty and the scheduler step matches the global step.
7. **Sharding changes everything.** Once state is sharded across ranks, use `torch.distributed.checkpoint` (which stores resharding metadata) or consolidate to a full state dict. Never rely on raw per-rank shards surviving a world-size change — that is a silent-corruption trap.
8. **This is an economic decision, not a nicety.** At 64 GPUs a weights-only resume can waste ~74 GPU-hours (~\$110) *per crash*, and large runs crash often. The fix is a few lines and pays back every single time.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four-walls frame and the vocabulary (rank, world size, shard) this post builds on.
- [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — wrapping policy, sharding strategy, and where sharded checkpoints come from.
- [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas) — the sibling data-parallel path where the sampler and seeding traps first appear.
- [Distributed checkpointing](/blog/machine-learning/distributed-training/distributed-checkpointing) — async saves, sharded formats, and correctness under sharding, in depth.
- [Determinism across ranks](/blog/machine-learning/distributed-training/determinism-across-ranks) — why a non-deterministic data order breaks reproducible resumes.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist this post feeds into.
- [Debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume) and [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) — the broader debugging-series treatments.
- Zhang et al., *OPT: Open Pre-trained Transformer Language Models* (2022) and its released training logbook — restarts, spikes, and hands-on optimizer management at 175B scale.
- Chowdhery et al., *PaLM* (2022) — the ~20 loss spikes and the rewind-and-skip-batches mitigation.
- The PyTorch `torch.distributed.checkpoint` (DCP) documentation — the resharding-capable distributed checkpoint format.

---
title: "Debugging Checkpoint and Resume: The Loss Jump That Shouldn't Happen"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Localize and fix the resume bugs that detonate your loss — the fresh optimizer that loses Adam's moments, the scheduler that restarts warmup, the RNG and dataloader drift — and build the resume-equivalence test that proves a resume is truly continuous."
tags:
  [
    "debugging",
    "model-training",
    "checkpointing",
    "optimizer-state",
    "reproducibility",
    "pytorch",
    "finetuning",
    "deep-learning",
    "llm",
    "transformers",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/debugging-checkpoint-and-resume-1.png"
---

You trained a model for eleven hours. The loss came down beautifully from 4.1 to 0.31, the curve was smooth, and you saved a checkpoint at step 5,000 feeling good. Then your spot instance got reclaimed at step 5,200. You relaunched, pointed the trainer at `checkpoint-5000`, and watched the loss come back on the very first logged step as **1.9** — six times higher than where you saved it. The model didn't start from scratch (the eval was still reasonable, so the weights clearly loaded), but something jumped, and over the next thirty-five steps the loss clawed its way back down to 0.32 before continuing as if nothing had happened. No crash. No error message. Just a visible scar in the loss curve every time you resume, and a quiet feeling that your training is not as continuous as you think.

That scar is the single most diagnostic symptom in this entire post, and it has a precise cause. A loss jump on resume means **something in the training state was not restored**. The weights round-tripped fine — that's why the eval still worked — but a checkpoint is not just the weights. It is the *entire* state of an in-progress optimization: the optimizer's accumulated moment estimates, the learning-rate scheduler's position, the random number generators, the dataloader's place in the epoch, the gradient scaler's loss scale, the step and epoch counters, and possibly an EMA shadow and a gradient-accumulation counter. Drop any one of them and the resume diverges from the run that never stopped. The most common single culprit — the canonical resume bug — is restoring only `model.state_dict()` and constructing a *fresh* optimizer, which throws away Adam's first and second moments and forces the optimizer to spend tens of steps rebuilding running averages it already had. That is the spike, and then the recovery.

This post is about every way a resume can lie to you and how to make it bit-continuous. We will start with the **science**: why Adam's moments are state and not derivable from the weights, why a fresh optimizer mis-scales its first steps (the math of the bias correction predicts the transient spike), why the scheduler's step counter must be restored or warmup restarts, and why exact resume is a *determinism* problem at heart. Then we go bug by bug — optimizer, scheduler, RNG, dataloader, scaler/EMA/accumulation, step counters, format/version mismatch, and best-vs-last — each with a mechanism, runnable diagnostic code, and before→after evidence. Figure 1 shows the symptom that ties them all together: the weights-only resume that spikes versus the full-state resume that holds the loss flat across the boundary.

![Side by side comparison of a weights-only resume spiking the loss from 0.31 to 1.9 against a full-state resume that stays continuous at 0.31](/imgs/blogs/debugging-checkpoint-and-resume-1.png)

The spine of this whole series holds here too: **a bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and you bisect to the right one before touching code.** Resume bugs straddle **optimization** (the optimizer/scheduler state) and **systems** (RNG, dataloader, checkpoint format), and the two master tools cut straight to them: *make-it-fail-small* (run a 200-step resume test, not a 5,000-step one) and *read the instruments* (log the loss the instant you resume — a jump localizes the problem to a piece of dropped state). The definitive test, which we will build and which appears in [the FSDP and sharding-bugs post](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs) in its distributed form, is brutally simple: **a run that trains 200 steps straight must equal a run that saves at 100, resumes, and trains 100 more — loss for loss.** If it doesn't, you have a resume bug, and this post tells you which one.

By the end you will be able to take any run whose loss jumps on resume and localize the dropped state in minutes, write a checkpoint that saves *everything*, and prove with one short script that your resume is genuinely a continuation and not a soft restart. For the master decision tree this slots into, start with [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); for the field-wide checklist, [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) is the capstone.

## 1. The science: why a checkpoint is more than weights

Before any debugging, we need a precise account of what a training step depends on, because every resume bug is a piece of that dependency that was not saved. The mistake everyone makes early is to believe that the model *is* the weights, so saving the weights saves the model. That is true for *inference* — a forward pass depends only on the parameters and the input. It is false for *optimization*, because the next gradient step depends on far more than the current weights.

### 1.1 The state of an in-progress optimization

Write the update rule of a generic optimizer. At step $t$, the new parameters are a function of the old parameters, the current gradient, and some **internal optimizer state** $S_t$:

$$
\theta_{t+1} = \theta_t - \eta_t \cdot f(g_t, S_t), \qquad S_{t+1} = u(g_t, S_t).
$$

For plain SGD with no momentum, $S_t$ is empty: $f(g_t) = g_t$, and the only thing you need to resume is $\theta_t$ and the learning rate $\eta_t$. SGD is the special case where "weights are enough" is almost true — almost, because $\eta_t$ comes from a scheduler that itself has state. But the moment you add momentum, $S_t$ becomes a velocity vector $v_t$ that is *just as large as the parameters* and *cannot be recovered from $\theta_t$ alone*. With Adam, $S_t$ is two vectors the size of the parameters — the first moment $m_t$ (a running mean of gradients) and the second moment $v_t$ (a running mean of squared gradients) — plus the step count $t$ used for bias correction.

This is the core insight: **the optimizer carries memory that the weights do not encode.** You cannot look at a parameter tensor and reconstruct the running average of its recent gradients. That information lives only in the optimizer's `state_dict`, and if you do not save and restore it, it is gone forever. The model "looks" the same, but the optimizer has amnesia.

It helps to count exactly how much state we are talking about, because the *size* of what you're forgetting is what makes the bug expensive. For a model with $P$ parameters trained with Adam, the optimizer holds $2P$ floats of moment state ($m$ and $v$), plus a per-parameter step counter. In fp32 that is $8P$ bytes of optimizer state — *twice the size of the model itself*. For a 7B-parameter model that is 56 GB of optimizer state that a weights-only checkpoint silently throws away. People skip saving the optimizer partly to make checkpoints smaller, not realizing that the thing they're omitting is the thing that makes the resume continuous. The disk you save is paid back, with interest, in wasted recovery steps and slightly-worse models. A correct mental accounting is: a *resumable* checkpoint of a 7B model in fp32-Adam is on the order of $4P + 8P = 12P$ bytes (weights + moments) ≈ 84 GB, while a *shippable* weights-only export is just $2P$–$4P$ bytes. They are different sizes because they are different objects with different jobs.

There's a second, quieter consequence. The optimizer state is not just *large* — it is *per-parameter and order-sensitive*. PyTorch stores optimizer state in a dict keyed by the parameter's position in its parameter group, not by a name. So restoring it correctly depends on the optimizer being reconstructed with the *exact same parameter groups in the exact same order* as the original. Reorder your modules, add a parameter, freeze a layer between save and load, and the moments land on the wrong tensors — a far nastier bug than simply forgetting them, because now the optimizer is *confidently wrong*. We return to this in §3 and §9, but flag it here: optimizer state is positional, and position is fragile across refactors.

### 1.2 Why Adam's moments matter — and why losing them spikes the loss

Adam's full update, including bias correction, is:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
$$

With the defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, the second moment $v$ is a *very* slow-moving average — its effective window is about $1/(1-\beta_2) = 1000$ steps. The whole point of dividing by $\sqrt{\hat{v}_t}$ is per-parameter adaptive scaling: a parameter whose gradients have been large for a long time gets a *small* effective step, and a parameter with small gradients gets a *large* one. After 5,000 steps of training, $v$ encodes a thousand-step history of every parameter's gradient magnitude. That history is what keeps the steps well-scaled.

Now resume with a fresh optimizer. At the first step after resume, $m_0 = 0$ and $v_0 = 0$. Look at what happens to the update. The bias correction at $t=1$ gives $\hat{m}_1 = m_1 / (1-\beta_1) = g_1$ and $\hat{v}_1 = v_1 / (1-\beta_2) = g_1^2$, so the very first update is

$$
\theta_1 = \theta_0 - \eta \frac{g_1}{\sqrt{g_1^2} + \epsilon} \approx \theta_0 - \eta \cdot \text{sign}(g_1).
$$

That is a step of magnitude $\eta$ in *every* coordinate, regardless of how large or small that parameter's gradients have been historically. A parameter that the warm optimizer was nudging by $10^{-7}$ per step (because its $\sqrt{\hat{v}}$ was large) suddenly gets nudged by the full $\eta$, perhaps $2\times10^{-5}$ — a step **two orders of magnitude too big**. Multiply that across millions of well-tuned parameters and the model lands in a worse place than it left. That is the spike. Then, as $v$ re-accumulates the gradient-magnitude history over the next tens of steps, the steps re-shrink to their proper scale and the loss recovers. The recovery time is governed by how fast $v$ catches up, which is on the order of $1/(1-\beta_2)$ but in practice much shorter for the loss to *visually* recover — roughly 20–50 steps — because the largest mis-scalings self-correct fastest. Figure 3 walks through this rebuild step by step.

So the loss jump is not random noise; it is the *predictable* signature of an optimizer that lost its second moment. The size of the jump scales with how adapted the optimizer had become — early in training a fresh optimizer barely hurts (the moments were small anyway), but deep into a finetune where the adaptive scaling has done a lot of work, a fresh optimizer can be catastrophic. This is exactly why the resume bug is worse the longer you have trained, and why people who only ever resume near the start never notice it.

### 1.3 The scheduler is state too

The learning rate $\eta_t$ in those equations is not a constant — it comes from a scheduler that is *itself* a stateful object. A cosine schedule with warmup computes $\eta_t$ as a function of the current step $t$ and the total steps $T$. PyTorch schedulers track their position with `last_epoch` (which, despite the name, is usually a step counter when you call `scheduler.step()` per optimizer step). If you build a fresh scheduler on resume, `last_epoch` is $-1$, so the first `scheduler.step()` puts you at step 0 — the *bottom* of the warmup ramp. The LR you resume with is not the LR you saved with; it is the tiny warmup-start LR. Figure 6 shows this exact failure: a finetune resumed at step 800 with a fresh scheduler restarts a 500-step warmup, dropping the LR from $1.4\times10^{-5}$ to $4\times10^{-7}$ and stalling.

The scheduler is sneakier than the optimizer because it is *small* — it has almost no parameters, so people never think to save it. But its effect is global: it sets the step size for every parameter. A wrong LR after resume produces a different *shape* of jump than a fresh optimizer: instead of a spike-then-recover, you get a *stall* (LR too small, loss flat) or a *re-spike* (LR jumps back up to the warmup peak and overshoots). Reading which shape you got is half the diagnosis.

### 1.4 Exact resume is a determinism problem

There is a deeper requirement hiding under all of this. Even if you save and restore the model, optimizer, *and* scheduler perfectly, the resumed run can still diverge from the continuous one — because the two runs will see *different data in a different order*, *apply different dropout masks*, and *use different augmentations* unless the random number generators are in the same state. The continuous run's RNG advanced through 5,000 steps of sampling; the resumed run starts with a fresh RNG (or a re-seeded one) and produces a different sequence. From step 5,001 onward the two runs are training on different mini-batches, so of course their losses differ. This is not a "jump" so much as a slow, permanent *drift*, and it is the hardest resume bug to see because there is no dramatic spike — just two curves that should be identical quietly separating.

Making a resume bit-continuous therefore requires the same machinery as making a run reproducible at all: deterministic algorithms, restored RNG state for Python, NumPy, PyTorch CPU and CUDA, and a dataloader that picks up where it left off. We cover the full determinism story in [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training); here the point is narrower but sharp: **you cannot verify a resume without determinism, because without it you can never tell whether a difference came from a dropped state or from inherent run-to-run noise.** Determinism is the measuring instrument; without it the resume-equivalence test is meaningless.

### 1.5 A taxonomy of resume divergence

It is worth fixing, up front, the three *qualitatively different shapes* a broken resume produces, because the shape is the fastest diagnostic you have and the rest of the post is organized around it.

First, the **discontinuous jump at the boundary**: the loss at the first post-resume step is sharply different from the last pre-save step, then it recovers (or, in the worst case, doesn't). This is an *optimization-state* failure — a dropped optimizer or scheduler. The mechanism is immediate: the very first update after resume uses wrong moments or a wrong LR, so the very first measured loss is wrong. Boundary failures are the loud, obvious ones.

Second, the **continuous-but-divergent drift**: the loss is smooth *across* the boundary — no spike — but the resumed trajectory slowly peels away from the continuous one over tens to hundreds of steps. This is a *data-stream* failure — un-restored RNG or a non-stateful dataloader. The mechanism is delayed: the first post-resume step is fine because the weights and optimizer are right, but the *data* seen from that step onward differs, so the divergence accumulates. Drift failures are the quiet, dangerous ones, because they pass a visual loss-curve inspection.

Third, the **near-total reset**: the loss comes back close to its *initial* (untrained) value, not a recoverable spike. This is a *weight-loading* failure — `strict=False` silently dropping keys, or a sharded/full state-dict mismatch — where the weights themselves didn't load. Reset failures look superficially like a giant boundary jump but are categorically worse, because the model is genuinely back near random and will not self-heal.

Knowing which of the three you have collapses the search space immediately: a *boundary jump* sends you to §3–§4 (optimizer/scheduler), a *drift* to §5–§6 (RNG/dataloader), and a *reset* to §9 (load-time corruption). The rest is confirming which specific field within that class is missing.

## 2. The complete checkpoint: everything you must save

Let's enumerate the full state. Figure 2 stacks it: model, optimizer, scheduler, scaler, RNG, step/epoch counters, dataloader position, and EMA/accumulation. Each layer below is a thing people routinely forget, ordered roughly by how often forgetting it bites.

![Vertical stack of the eight kinds of state a resumable checkpoint must save from model weights down through optimizer moments, scheduler, scaler, RNG, counters, dataloader position, and EMA](/imgs/blogs/debugging-checkpoint-and-resume-2.png)

Here is the canonical save function. Notice that *everything* with a `state_dict()` gets one, and the things without (RNG, counters) get saved explicitly.

```python
import torch
import numpy as np
import random

def save_checkpoint(path, *, model, optimizer, scheduler, scaler,
                    step, epoch, ema=None, extra=None):
    ckpt = {
        # --- the things with state_dict() ---
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),     # Adam m, v, step per param
        "scheduler": scheduler.state_dict(),     # last_epoch, base_lrs, etc.
        "scaler": scaler.state_dict() if scaler is not None else None,
        # --- counters ---
        "step": step,
        "epoch": epoch,
        # --- RNG state for every generator that touches the run ---
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),                  # CPU
            "cuda": torch.cuda.get_rng_state_all(),          # per device
        },
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()                       # shadow weights
    if extra is not None:
        ckpt["extra"] = extra                                # accum counter, etc.
    # atomic write: write to a temp file, then rename, so an interrupted
    # save never leaves a half-written checkpoint you later try to load
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    import os
    os.replace(tmp, path)
```

And the matching load. The critical detail is *order* and *strictness*: load weights with `strict=True` (so a missing or extra key throws, instead of silently doing nothing), then load the optimizer, scheduler, scaler, RNG, and counters.

```python
def load_checkpoint(path, *, model, optimizer, scheduler, scaler,
                    ema=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    # strict=True raises on mismatch; if you must use strict=False,
    # ASSERT the lists are empty so the drop is never silent:
    # assert not missing and not unexpected, (missing, unexpected)

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    rng = ckpt["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])

    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])

    return ckpt["step"], ckpt["epoch"], ckpt.get("extra")
```

Two subtleties that bite people. First, **`optimizer.load_state_dict` must run after the optimizer is constructed with the same parameter groups as the original** — the optimizer state is keyed by parameter *index within a group*, so if you changed the param groups (added LoRA adapters, froze a layer, reordered modules) between save and load, the moments land on the wrong parameters or fail to load. Second, **the optimizer state's tensors must end up on the same device as the parameters.** `torch.load(..., map_location="cpu")` loads everything to CPU; when you then move the model to GPU, PyTorch's `optimizer.load_state_dict` followed by the first `.step()` usually relocates the state, but in some setups you must move the optimizer state tensors yourself. A device mismatch shows up as a cryptic "expected device cuda but got cpu" at the first step, not a loss jump — a different, louder failure mode.

A third subtlety, easy to miss and increasingly common: **the order of construct-then-load matters, and so does where you call `.to(device)`.** The safe sequence is (1) build the model, (2) move it to the target device, (3) build the optimizer from the *now-on-device* parameters, (4) load the model state dict, (5) load the optimizer state dict. If you build the optimizer *before* moving the model to GPU, the optimizer's internal references can point at the CPU parameters, and the loaded state may not relocate cleanly. Frameworks like `accelerate` handle this ordering for you with `accelerator.prepare(...)` and `accelerator.save_state`/`load_state`, which is one good reason to use them rather than hand-rolling the dance. But when you *do* hand-roll it — and you will, in research code — getting the order wrong is a real source of "the resume threw an inscrutable device error" reports.

One more thing about the *atomic write* in `save_checkpoint`. Writing to a temp file and renaming is not pedantry — it is the difference between a survivable crash and a corrupt checkpoint that *itself* causes a resume failure. If your job is preempted *while writing the checkpoint*, a non-atomic `torch.save` leaves a half-written file. On the next resume, `torch.load` either throws (best case — you notice) or, worse, loads a truncated state dict that silently drops the tensors written after the truncation point. `os.replace` is atomic on POSIX filesystems, so the checkpoint either fully exists or doesn't — there is no half state. For the same reason, keep the *previous* checkpoint until the new one is fully written; deleting the old one first and crashing mid-write leaves you with *no* valid checkpoint at all. `save_total_limit` in HF Trainer does this correctly; if you roll your own rotation, rotate *after* the atomic rename, never before.

The following table is the diagnostic core of this post: each symptom shape maps to one dropped piece of state, with the confirming test and the fix. Figure 4 renders the same mapping.

| Resume symptom | Likely cause | Confirming test | Fix |
|---|---|---|---|
| Spike then recovers in ~20–50 steps | Optimizer `m, v` lost (fresh optimizer) | Resume-equivalence: 200 vs 100+100 | Save and load `optimizer.state_dict()` |
| LR wrong; warmup restarts; loss stalls | Scheduler `last_epoch` reset to −1 | Log LR at the resume step | Save and load `scheduler.state_dict()` |
| Slow drift; never matches continuous run | RNG state not restored (data/dropout differ) | Diff the first 5 batches' ids | Save and restore all four RNG states |
| Re-sees early data; plateau or overfit | Dataloader starts at epoch 0 | Log first batch indices at resume | Skip to step or use a stateful sampler |
| Loss off by a constant factor | GradScaler / accum counter lost | Log loss scale and accum step | Save scaler and accumulation counter |
| Schedule ends early or logs wrong x-axis | `step` / `epoch` counter wrong | Print resumed step vs expected | Save and restore the global step |

![Matrix mapping four resume symptoms to their cause, a confirming test, and the fix that removes the loss jump](/imgs/blogs/debugging-checkpoint-and-resume-4.png)

## 3. Bug 1 — the fresh optimizer (the canonical resume bug)

This is the one you will hit most. Someone resumes by loading `model.state_dict()` and then writing `optimizer = AdamW(model.parameters(), lr=...)` — a *brand new* optimizer. The weights are right; the moments are zero.

### 3.1 The mechanism, made concrete

We derived in §1.2 why a fresh optimizer's first step is roughly $\eta \cdot \text{sign}(g)$ in every coordinate. Let's put numbers on it. Suppose at step 5,000 a particular weight had $\sqrt{\hat{v}} = 200$, so its effective step was $\eta / 200 = 2\times10^{-5}/200 = 1\times10^{-7}$. After a fresh optimizer, its first effective step is $\eta / \sqrt{g^2} \approx \eta = 2\times10^{-5}$ — a **200× larger** step. The weight overshoots; the loss it contributes to rises. Across the network, the well-adapted parameters (the ones with large accumulated $v$) take the biggest erroneous jumps, which is precisely the set of parameters the model had carefully tuned. Hence a *large* jump deep in training and a *small* one early.

A useful corollary: **SGD with momentum has the same bug, milder.** Plain SGD (no momentum) is the *only* optimizer where a weights-only resume is genuinely lossless, because its state is empty. The instant you add momentum, you carry a velocity vector $v_t = \mu v_{t-1} + g_t$ that is the size of the parameters, and a fresh optimizer resets $v_0 = 0$. The damage is smaller than Adam's because momentum doesn't do per-parameter *scaling* (so there's no 200× mis-scaling), but it does lose accumulated direction — the update right after resume is just $g_1$ instead of the accumulated $\mu v + g_1$, so you lose the "inertia" the optimizer had built up, and the loss takes a small step backward. So the rule generalizes: *any stateful optimizer* must have its state saved, and the size of the resume jump scales with how much state-dependent adaptation that optimizer does. Adam (per-parameter adaptive scaling) is the worst; SGD+momentum is mild; plain SGD is immune. If you ever want to *prove* the optimizer is your problem and you can afford it, temporarily switch to plain SGD: if the resume jump vanishes, the dropped state was the optimizer's.

### 3.2 The diagnostic: a post-resume loss-jump detector

You don't need the full equivalence test to *catch* this in production. Just log the loss for a window of steps before saving and after resuming, and compare. Here is a drop-in detector.

```python
import collections

class ResumeJumpDetector:
    """Compare the loss right after resume to the loss right before save."""
    def __init__(self, window=20, jump_ratio=1.3):
        self.pre_save_losses = collections.deque(maxlen=window)
        self.post_resume_losses = []
        self.window = window
        self.jump_ratio = jump_ratio
        self.resumed = False

    def on_step(self, loss):
        if self.resumed:
            self.post_resume_losses.append(loss)
            if len(self.post_resume_losses) == 1:
                pre = (sum(self.pre_save_losses) / len(self.pre_save_losses)
                       if self.pre_save_losses else float("nan"))
                post = loss
                if post > self.jump_ratio * pre:
                    print(f"[RESUME-JUMP] loss {pre:.3f} -> {post:.3f} "
                          f"({post/pre:.2f}x). State likely NOT restored. "
                          f"Check optimizer/scheduler/RNG.")
                else:
                    print(f"[RESUME-OK] loss {pre:.3f} -> {post:.3f} "
                          f"({post/pre:.2f}x), within {self.jump_ratio}x.")
        else:
            self.pre_save_losses.append(loss)

    def mark_resumed(self):
        self.resumed = True
```

In a real loop you would persist `pre_save_losses` *into the checkpoint* (it is cheap) so the comparison survives the restart, then call `detector.mark_resumed()` right after loading. A `[RESUME-JUMP]` line in your logs is an instant, unambiguous signal that a resume dropped state — far better than squinting at a curve days later.

### 3.3 Before → after evidence

The fix is one line — save and restore `optimizer.state_dict()` — and the effect is total. Below is a representative measurement from a transformer finetune (AdamW, $\eta = 2\times10^{-5}$, resumed at step 5,000).

| Metric | Weights-only resume | Full optimizer resume |
|---|---|---|
| Loss at last pre-save step | 0.31 | 0.31 |
| Loss at first post-resume step | 1.9 | 0.31 |
| Steps to return below 0.35 | ~35 | 0 |
| Resume-equivalence diff at step 1 | 1.6 | < 1e-6 |
| Wasted GPU time per resume | ~35 steps | 0 |

The jump goes from a 6× spike to nothing, and the resume becomes indistinguishable from a run that never stopped. Figure 8 renders this side by side with the scheduler and RNG fixes folded in.

#### Worked example: the cost of forgetting the optimizer

Suppose you train a 7B-parameter model on 8 × A100 spot instances that get reclaimed on average every 6 hours, and your job runs for 5 days. That's about 20 reclaim-and-resume cycles. With a weights-only resume, each resume wastes ~35 steps recovering, and at this scale each step costs roughly \$0.45 in GPU time (8 GPUs × ~\$1.80/GPU-hour × the per-step wall time). Thirty-five wasted steps per resume × 20 resumes × \$0.45 ≈ **\$315 of pure waste** — and that ignores the subtler damage: every spike nudges the model toward a slightly worse basin, and across 20 spikes the final model can measurably underperform a continuously-trained one. Saving the optimizer state costs you a few extra gigabytes per checkpoint on disk and *zero* training time. It is the highest-return one-line fix in this entire post.

## 4. Bug 2 — the scheduler that resets to step 0

You saved the optimizer. The spike is gone. But the loss *stalls* after resume — it doesn't spike, it just goes flat and barely moves for a few hundred steps. That is the scheduler.

### 4.1 The mechanism

A fresh scheduler has `last_epoch = -1`. The first `scheduler.step()` advances it to 0, which on a warmup schedule is the *minimum* LR. If your warmup is 500 steps and you resumed at step 800, you have just thrown away the warmup you already completed and restarted it — the LR plummets to its warmup-start value, the steps become tiny, and the loss flatlines until the warmup ramps back up 500 steps later. Worse, when the warmup *does* finish the second time, the schedule thinks you are at step 500 when you are really at step 1,300, so the *entire* remaining schedule is shifted — your cosine decay ends 800 steps too late, or your LR never reaches the small value it should have at the end of training.

Figure 6 shows this exactly: resume at step 800 with a fresh scheduler drops the LR from $1.4\times10^{-5}$ to $4\times10^{-7}$, restarting a 500-step warmup and stalling the loss; the restored scheduler keeps the LR on its decay and the loss continues.

![Before and after of a scheduler resume where a fresh scheduler restarts warmup and drops the learning rate to a tiny value versus a restored scheduler that keeps the learning rate on its decay](/imgs/blogs/debugging-checkpoint-and-resume-6.png)

### 4.2 The diagnostic: log the LR at the resume boundary

The confirming test is trivial and you should have it always-on: log `scheduler.get_last_lr()` (or `optimizer.param_groups[0]["lr"]`) every step. At the resume boundary, the LR should be *continuous*. If it jumps — especially if it drops to near zero — your scheduler reset.

```python
# always-on: log the LR; at resume it must be continuous
lr = optimizer.param_groups[0]["lr"]
logger.log({"lr": lr, "step": global_step})

# explicit assertion at the resume boundary
if just_resumed:
    expected_lr = scheduler.get_last_lr()[0]
    print(f"[RESUME] step={global_step} lr={expected_lr:.3e}")
    # sanity: if you saved the pre-shutdown lr, compare directly
    if saved_lr is not None:
        ratio = expected_lr / saved_lr
        assert 0.9 < ratio < 1.1, (
            f"LR discontinuity on resume: saved {saved_lr:.3e}, "
            f"now {expected_lr:.3e} ({ratio:.2f}x). Scheduler not restored?")
```

### 4.3 The fix and a finetuning caveat

The fix is `scheduler.load_state_dict(ckpt["scheduler"])` — but there is a subtlety that *only* shows up on resume. `LambdaLR` and friends store the *function* implicitly and only the `last_epoch` in the state dict, so as long as you construct the scheduler with the same `T_max`/`num_warmup_steps`/`num_training_steps` and then load the state dict, you are fine. But if you change the *total* training steps on resume — say you decide to train longer — a cosine scheduler restored to `last_epoch=800` will recompute its curve against the *new* `T_max`, which silently changes the LR at every future step. That is sometimes what you want and sometimes a bug; the rule is: **changing the schedule on resume is a deliberate decision, and you should log the full LR curve to confirm it does what you intend.**

There's a finetuning-specific trap here worth calling out. Hugging Face `Trainer` ties the scheduler to `max_steps` / `num_train_epochs` *and* to the dataset length. If you resume with `resume_from_checkpoint` but pass a different dataset or a different `num_train_epochs`, the scheduler the Trainer builds will not match the one you saved, and even though `Trainer` *does* restore scheduler state, the recomputed `max_steps` can shift the decay. The defensive move is to keep every training argument that affects the schedule *identical* across resume. We return to `Trainer`'s resume contract in §10.

#### Worked example: the warmup-restart that cost 500 steps and 1.2 points of perplexity

Concretize the scheduler bug with numbers. A language-model finetune uses a linear-warmup-then-cosine schedule: warmup over the first 500 steps to a peak LR of $2\times10^{-5}$, then cosine decay over the remaining 4,500 steps. At step 800, the run is 300 steps into the decay; the scheduled LR is about $1.96\times10^{-5}$ (just past the peak). The job is preempted and resumed. The optimizer state was saved correctly, so there's no fresh-optimizer spike — but the scheduler was rebuilt fresh with `last_epoch=-1`. The first `scheduler.step()` after resume sets the step to 0, so the LR is the *warmup-start* value: with linear warmup from 0, step 1 of 500 gives $2\times10^{-5} \times (1/500) = 4\times10^{-8}$ — effectively zero. For the next ~500 steps the LR crawls back up the warmup ramp, during which the model barely moves: the loss is flat, not spiking. Logging the LR makes it instantly obvious — a continuous LR at the boundary would read ~$1.96\times10^{-5}$, but you see $4\times10^{-8}$, a 500× drop.

The cost is twofold. First, ~500 steps of near-zero learning are wasted — at the spot-instance rate from §3.3, that's real money, but more importantly the run effectively pauses. Second, and more insidious, the *total* schedule is now shifted: the cosine decay that was supposed to end at step 5,000 now thinks step 0 is at the resume point, so it ends at step 5,800. The model's final LR — the small, careful tail of the cosine that does the last bit of refinement — never reaches the value it should have, and the final checkpoint lands at a measurably higher loss. In one such run the team measured a **1.2-point perplexity regression** versus a continuous baseline, entirely attributable to the shifted schedule tail. The fix — `scheduler.load_state_dict(...)` — is again one line, and it closed the gap. The takeaway: the scheduler is small and forgettable, but its blast radius is the entire LR trajectory, and a wrong LR is, as the [learning-rate post](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) argues at length, almost always the most expensive thing you can get wrong.

## 5. Bug 3 — RNG state and the silent drift

This is the bug with no spike — the one you only catch with the equivalence test. You saved model, optimizer, and scheduler. The loss is continuous at the boundary. And yet a careful comparison shows the resumed run slowly diverging from the continuous one over hundreds of steps. The cause is randomness.

### 5.1 The mechanism

Three things in training consume random numbers every step: **data order** (the shuffle in your sampler), **dropout masks**, and **stochastic augmentation** (random crops, flips, mixup, SpecAugment). The continuous run's RNGs advanced through 5,000 steps of these draws. The resumed run, with fresh or re-seeded RNGs, produces a *different* sequence of draws — a different shuffle, different dropout, different augmentations — from step 5,001 onward. The two runs are now optimizing on different data with different stochastic regularization, so their losses differ. Not by a spike; by an accumulating drift, because each step's tiny difference compounds.

The reason this is so insidious is that the drift is *within the noise band* you'd expect from a run anyway, so it doesn't look like a bug — it looks like "training is a bit noisy." The only way to know the resume isn't continuous is to compare against a run that *never stopped*, which requires determinism so the continuous run is itself reproducible.

### 5.2 The diagnostic: diff the data order across resume

The fastest confirming test is to log the indices of the first few batches after resume and compare them to the indices the continuous run sees at the same global step. If they differ, the RNG (or the dataloader, §6) wasn't restored.

```python
# log the first global indices each batch yields, around the boundary
def log_batch_ids(batch, global_step, tag):
    # assumes your dataset returns an "idx" field; if not, wrap it to
    ids = batch["idx"][:8].tolist()
    print(f"[{tag}] step={global_step} first_ids={ids}")

# Run A (continuous): record ids at steps 100..105
# Run B (resumed at 100): record ids at steps 100..105
# They MUST match index-for-index. If not, RNG/dataloader drifted.
```

A more direct check: hash the RNG state at save and at the moment after load.

```python
import hashlib

def rng_fingerprint():
    parts = [
        repr(random.getstate()).encode(),
        np.random.get_state()[1].tobytes(),       # the MT19937 key array
        torch.get_rng_state().numpy().tobytes(),
    ]
    if torch.cuda.is_available():
        for s in torch.cuda.get_rng_state_all():
            parts.append(s.numpy().tobytes())
    return hashlib.sha1(b"".join(parts)).hexdigest()[:12]

print("RNG fingerprint:", rng_fingerprint())
# save this fingerprint in the checkpoint; after load, recompute and assert equal
```

### 5.3 The fix and the determinism requirement

Saving and restoring all four RNG states (Python, NumPy, Torch CPU, Torch CUDA — the `save_checkpoint` in §2 does this) gets the *generator* state right. But for the resumed run to actually reproduce the continuous one, the operations themselves must be deterministic, or two identical RNG states can still produce different results on the GPU. The standard setup:

```python
import torch, os

def make_deterministic(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)        # raises on nondeterministic ops
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # needed for cuBLAS determinism
```

There is a real cost: `use_deterministic_algorithms(True)` can be 10–30% slower and forbids a few fast kernels. The pragmatic stance is to turn determinism *on* while you debug a resume (so the equivalence test is meaningful) and decide afterward whether to keep it. The full trade-off — including dataloader-worker seeding via `worker_init_fn` and the `generator` argument — is in [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training). The key cross-cutting point: a resume can only be *proven* continuous under determinism, so determinism is not optional for *debugging* resume even if you don't run production that way.

#### Worked example: the resume that "looked fine" but quietly lost a point of accuracy

A team finetunes a classifier with strong augmentation (RandAugment + mixup). They resume from spot reclaims about a dozen times over a run. The loss curve looks continuous at every boundary — no spikes, because they correctly save the optimizer and scheduler. But the final validation accuracy lands at **77.8%**, while a teammate who ran the same config straight through on a reserved instance gets **78.9%**. A full point, reproducibly. The cause: they never saved RNG state, so each resume restarted the augmentation and shuffle sequence. Across a dozen resumes, the model saw a subtly different (and effectively *less diverse*, because re-correlated) augmentation stream than the continuous run, and the regularization was weaker. Restoring RNG state closed the gap to within run-to-run noise (78.8% vs 78.9%). No spike ever flagged this; only the equivalence mindset — "a resume must equal a continuous run" — caught it.

## 6. Bug 4 — the dataloader that starts over

Closely related to RNG, but distinct: even with RNG restored, if you resume *mid-epoch* and your dataloader starts iterating from the beginning of the dataset, you re-see data you already trained on this epoch and skip data you hadn't reached. The epoch's data distribution is now wrong.

### 6.1 The mechanism and why it's subtle

Standard PyTorch `DataLoader` is *not* resumable out of the box — it has no notion of "I was at batch 4,000 of 10,000." When you create a fresh `DataLoader` and `enumerate` it, it starts at index 0. If you resumed at global step 5,200 and your epoch is 10,000 steps long, you are at batch 200 of epoch 1, but the dataloader hands you batch 0. You re-train on the first 200 batches (mild over-emphasis) and, more importantly, your shuffle for the epoch is regenerated, so the *order* is different even if RNG is restored — because the order depends on which `set_epoch` was called and when the sampler was constructed.

The damage is usually small for short epochs (you re-see a little data) but large for long-epoch / single-epoch LLM pretraining, where "the epoch" is the whole dataset and resuming from the start means re-training on data you've seen and never reaching the tail. For a single-epoch run over a trillion tokens, a naive resume can mean you train twice on the first chunk and zero times on the last chunk you never reached.

### 6.2 The fix: skip-to-step, or a stateful sampler

There are two correct fixes. The simple one for map-style datasets is to **set the sampler's epoch and skip the already-consumed batches**:

```python
from torch.utils.data import DataLoader

# DistributedSampler / your sampler must be told the epoch so the shuffle
# matches what the continuous run used for this epoch
sampler.set_epoch(resume_epoch)
loader = DataLoader(dataset, batch_size=bs, sampler=sampler,
                    num_workers=4, generator=g)

# skip the batches already consumed this epoch
batches_done_this_epoch = resume_step % steps_per_epoch
it = iter(loader)
for _ in range(batches_done_this_epoch):
    next(it)            # discard; advances the sampler to the resume point
# now train from `it`
```

Skipping is correct but wasteful for huge datasets (you pay to load and discard). The robust fix is a **stateful dataloader** that checkpoints its own position. Hugging Face `datasets` with `IterableDataset` supports `state_dict()`/`load_state_dict()` for exactly this, and `torchdata`'s `StatefulDataLoader` does too:

```python
from torchdata.stateful_dataloader import StatefulDataLoader

loader = StatefulDataLoader(dataset, batch_size=bs, num_workers=4)

# save the loader's position INTO your checkpoint
ckpt["dataloader"] = loader.state_dict()
# on resume, restore it so iteration continues from the exact batch
loader.load_state_dict(ckpt["dataloader"])
```

### 6.3 The diagnostic

Log the first batch's global indices right after resume and compare to what the continuous run saw at that global step (the §5.2 snippet). With a correct stateful loader they match; with a fresh loader they are `[0, 1, 2, ...]` when they should be `[some shuffled ids from mid-epoch]`. That `[0, 1, 2, ...]` pattern at a non-zero resume step is the unmistakable fingerprint of a dataloader that started over.

### 6.4 The worker-seeding trap inside the dataloader

There's a subtle interaction between dataloaders and RNG that bites specifically on resume with `num_workers > 0`. Each dataloader worker is a separate process with its *own* RNG, seeded from a base seed plus the worker id at iteration start. If your dataset does any randomness *inside* `__getitem__` (random augmentation is the common case), that randomness lives in the *worker's* RNG, not the main process's — so restoring `torch.get_rng_state()` in the main process does *nothing* for the per-worker augmentation streams. The fix is a deterministic `worker_init_fn` that seeds each worker from a value derived from the *restored* epoch/step and the worker id, plus passing an explicit `generator` to the `DataLoader` for the sampler's shuffle:

```python
import torch, numpy as np, random

def seed_worker(worker_id):
    # called in each worker process at iteration start;
    # base seed is derived from torch's main-process seed + worker id
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(base_seed + resume_epoch)     # shuffle RNG, tied to epoch

loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                    num_workers=4, worker_init_fn=seed_worker, generator=g)
```

The deeper point, which we develop in [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training): there is not *one* RNG in a training run, there are *several* — the main process's, each worker's, and the CUDA generator — and a correct resume must account for all of them. The §2 checkpoint saves the main-process and CUDA generators; the worker generators are reconstructed deterministically from the restored epoch via `worker_init_fn` and `set_epoch`, because you generally cannot snapshot a worker process mid-iteration. If your augmentation stream drifts on resume despite restoring the main RNG, the workers are almost always the culprit.

## 7. Bug 5 — scaler, EMA, and the accumulation counter

Three smaller state objects that, when forgotten, produce their own characteristic failures. Figure 7 will catalog the load-time failures; here we cover the *state* you forgot to save.

### 7.1 GradScaler (AMP loss scaling)

With fp16 automatic mixed precision, `torch.amp.GradScaler` maintains a dynamic **loss scale** — a multiplier that keeps small gradients from underflowing fp16's representable floor (~$6\times10^{-5}$). The scaler adapts this factor over time: it grows the scale when no overflow occurs and halves it when it sees an `inf`. After thousands of steps the scale has settled to a value well-tuned to your gradients. If you resume with a fresh scaler, it starts at the default (often $2^{16} = 65536$) and has to re-discover the right scale, which means a burst of *skipped* steps at resume (every overflow makes the scaler skip the update and halve the scale) until it settles. The signature is a short run of steps where the loss doesn't move at all (updates skipped) right after resume. We go deeper on loss scaling in [mixed-precision debugging](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) — but the resume fix is just `scaler.load_state_dict(...)`. Note bf16 needs no scaler, so this bug is fp16-specific.

```python
# fp16 AMP step with a scaler whose state you saved/restored
with torch.amp.autocast("cuda", dtype=torch.float16):
    loss = model(batch).loss
scaler.scale(loss).backward()
scaler.step(optimizer)     # skips the step if grads overflowed
scaler.update()            # adjusts the loss scale
# the scale lives in scaler.state_dict() — save it, or pay skipped steps on resume
```

### 7.2 EMA (exponential moving average of weights)

Many recipes keep an EMA copy of the weights — a slowly-updated shadow $\theta^{\text{ema}}_t = \alpha \theta^{\text{ema}}_{t-1} + (1-\alpha)\theta_t$ — and *evaluate* and *ship* the EMA weights, not the raw ones, because the average generalizes better. The EMA is a second full set of parameters and is **pure state** — it cannot be reconstructed from the current weights. Forget to save it and your resumed run rebuilds the EMA from scratch starting at the current weights, which means for the next $\sim 1/(1-\alpha)$ steps your "EMA" is just a lagged copy of the current weights with none of the averaging benefit. If you evaluate during that window, the EMA eval is *worse* than it should be, and you might wrongly conclude the model regressed. Save `ema.state_dict()` with everything else.

### 7.3 Gradient-accumulation counter

If you accumulate gradients over $K$ micro-steps before an optimizer step, you have a counter tracking where you are in the accumulation cycle. Resume mid-cycle without restoring it and you either do a partial optimizer step on a fraction of the intended batch, or you double-count. The cleanest defensive policy is to **only checkpoint on accumulation boundaries** (when the counter is 0), so a resume always starts a fresh accumulation cycle and the counter is implicitly 0. If you must checkpoint mid-cycle, save the counter and the accumulated gradients — but the boundary discipline is far simpler and is what most frameworks do. We cover the accumulation-equals-bigger-batch invariant in [gradient accumulation and effective-batch bugs](/blog/machine-learning/debugging-training/gradient-accumulation-and-effective-batch-bugs); for resume, the rule is just "checkpoint on a boundary."

## 8. Bug 6 — the wrong step or epoch counter

A subtle one with outsized downstream effects: you save the weights, optimizer, scheduler, and RNG, but you reset `global_step` to 0 (or to the wrong value) on resume.

### 8.1 What breaks

The step counter feeds three things: the **scheduler** (if you call `scheduler.step()` a wrong number of times, or if your schedule is computed from `global_step` directly, the LR is wrong), the **logging x-axis** (your resumed curve overwrites or misaligns with the pre-resume curve in W&B/TensorBoard), and the **stopping condition** (if you train "until `global_step == max_steps`" and the counter is wrong, you train too few or too many steps). The most damaging variant: if `max_steps` is checked against a reset counter, your run trains for `max_steps` *again* after every resume — a 5-day run that resumes 10 times might run for 50 days, or, if the counter is too high, stop almost immediately.

### 8.2 The fix and a logging note

Restore `global_step` and `epoch` from the checkpoint (the §2 loader returns them) and use them everywhere: as the scheduler's reference, as the logging step, and in the stopping condition. For W&B, pass `step=global_step` explicitly to `wandb.log` so the resumed run's points land at the correct x-positions and continue the curve rather than restarting it. A continuous-looking curve in your dashboard is itself a weak check that the counter is right; a curve that restarts at x=0 after resume is a dead giveaway it isn't.

## 9. Bug 7 — load-time corruption: strict=False, format, dtype, device

So far the bugs were *missing* state. This class is *present but mis-loaded* state — the load path silently drops or mangles data even though everything was saved. Figure 7 catalogs them.

![Tree of resume failures at load time branching into silent key drop from lenient loading, format mismatch between sharded and full state dicts, and dtype or device placement skew](/imgs/blogs/debugging-checkpoint-and-resume-7.png)

### 9.1 strict=False hides missing keys

`model.load_state_dict(sd, strict=False)` returns the lists of missing and unexpected keys instead of raising. People reach for `strict=False` to load a checkpoint into a slightly-changed model (a renamed layer, an added head). The danger is that it *silently does nothing* for any key that doesn't match — so if a refactor renamed `encoder.layers` to `encoder.blocks`, `strict=False` loads zero of those weights and your "resumed" model has a *randomly initialized* encoder. The eval is garbage, the loss is back near its initial value, and there is no error. **Always inspect the returned lists**, or assert them empty:

```python
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing or unexpected:
    print(f"MISSING (not loaded, left at init): {missing}")
    print(f"UNEXPECTED (in ckpt, ignored): {unexpected}")
    # for a true resume, both MUST be empty:
    assert not missing and not unexpected, "Silent weight drop on resume!"
```

A resume that loads only *some* weights looks like a fresh-optimizer spike but is far worse — it doesn't recover, because the dropped layers are genuinely random. If the loss jumps and *never* comes back, suspect a silent key drop before the optimizer.

### 9.2 Sharded vs full state dict (FSDP)

Under Fully Sharded Data Parallel, each rank holds only `1/N` of every tensor, so a checkpoint can be saved as *sharded* (per-rank slices) or *consolidated full* (gathered). Loading a full state dict with a sharded `StateDictType`, or vice versa, mismatches the shapes and either throws or — with the wrong settings — loads garbage. This is the most common cause of the "resume explodes the loss" report in distributed training, and it gets its own full treatment in [FSDP and sharding bugs](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs). The one-line rule: **save and load with the same `StateDictType`** (e.g., both `SHARDED_STATE_DICT`, via `torch.distributed.checkpoint`), and never hand a single-GPU full checkpoint to an FSDP load path without consolidation.

### 9.3 Dtype and device mismatch

Two quieter ones. **Dtype:** if you keep an fp32 master copy of params in the optimizer but save the model's bf16 compute copy, loading the bf16 weights into an fp32 master without the conversion loses precision; and loading optimizer moments saved in fp32 into a setup expecting bf16 (or the reverse) can throw or silently truncate. **Device:** `torch.load` without `map_location` tries to restore tensors to the device they were saved from; if you saved from `cuda:3` and load on a box where that device doesn't exist, you get an error — always pass `map_location="cpu"` (or a device map) and move to the target device deliberately. These produce loud errors more often than silent ones, which is mercifully easy, but the dtype one can silently degrade a resume, so verify the optimizer state's dtypes after load.

#### Worked example: the refactor that silently dropped half the encoder

A team renames a module during a cleanup: `self.encoder.layers` becomes `self.encoder.blocks` in the model code. They resume a 3-day finetune from `checkpoint-8000` with `model.load_state_dict(sd, strict=False)` — `strict=False` because an unrelated classification head was also being reshaped, and they wanted to "be lenient." The resume runs with no error. The loss comes back at **3.8** (the run had been at 0.4) and *never recovers* — it descends from 3.8 as if training from a partial init. Three hours of confused debugging later, someone prints the return value of `load_state_dict`: `missing=['encoder.blocks.0.attn.q_proj.weight', ... (288 keys)]`, `unexpected=['encoder.layers.0.attn.q_proj.weight', ... (288 keys)]`. The entire encoder was left at random initialization because the saved keys (`encoder.layers.*`) didn't match the renamed model keys (`encoder.blocks.*`), and `strict=False` swallowed it. The "resume" had silently reinitialized 60% of the network. The distinguishing symptom from a fresh-optimizer spike: the loss came back near the *fresh-init* value (3.8, close to the initial 4.1), not a recoverable spike near 0.4, and it *never* recovered — random layers don't self-heal the way a fresh optimizer's mis-scaled steps do. The fix was a one-time key-remapping dict to translate `layers`→`blocks` at load, then `strict=True` forever after. The lesson: `strict=False` is a *loaded gun* on resume; always inspect or assert-empty the returned lists.

## 10. The definitive test: resume-equivalence

Everything above is a *symptom* check. The *definitive* check — the one that proves a resume is correct rather than just plausible — is the **resume-equivalence test**: a run that trains $2N$ steps straight must produce, loss for loss, the same trajectory as a run that trains $N$ steps, saves, resumes, and trains $N$ more. If the two match to within floating-point tolerance at every step, every piece of state was restored. If they diverge at step $N+1$, you dropped something, and *where* they diverge tells you *what*. Figure 5 shows the structure.

![Graph of the resume-equivalence test where a fixed seed feeds both a 200-step straight run and a split 100-plus-100 resumed run whose step-by-step losses are compared for a match under one part per million or a divergence at step 101](/imgs/blogs/debugging-checkpoint-and-resume-5.png)

Here is the full, runnable test. It is small on purpose — make-it-fail-small — so it runs in seconds and you can put it in CI.

```python
import torch, copy

def build():
    """Construct model+optimizer+scheduler+scaler deterministically."""
    make_deterministic(0)                      # from §5.3
    model = make_model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    scaler = torch.amp.GradScaler("cuda")
    return model, opt, sched, scaler

def train_steps(model, opt, sched, scaler, batches, log):
    for batch in batches:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = model(batch).loss
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        log.append(loss.item())

def resume_equivalence_test(N=100):
    data = fixed_batches(2 * N)                 # SAME data both runs

    # ---- Run A: 2N steps straight ----
    a = build(); log_a = []
    train_steps(*a, data, log_a)

    # ---- Run B: N steps, full save, resume, N more ----
    b = build(); log_b = []
    model, opt, sched, scaler = b
    train_steps(model, opt, sched, scaler, data[:N], log_b)
    save_checkpoint("eq.pt", model=model, optimizer=opt, scheduler=sched,
                    scaler=scaler, step=N, epoch=0)
    # fresh objects, then load EVERYTHING
    model2, opt2, sched2, scaler2 = build()
    load_checkpoint("eq.pt", model=model2, optimizer=opt2,
                    scheduler=sched2, scaler=scaler2)
    train_steps(model2, opt2, sched2, scaler2, data[N:], log_b)

    # ---- compare ----
    diffs = [abs(x - y) for x, y in zip(log_a, log_b)]
    max_diff = max(diffs)
    first_bad = next((i for i, d in enumerate(diffs) if d > 1e-4), None)
    print(f"max |loss_A - loss_B| = {max_diff:.2e}")
    if first_bad is None:
        print("PASS: resume is bit-continuous.")
    else:
        print(f"FAIL: diverges at step {first_bad} "
              f"(diff {diffs[first_bad]:.2e}). State dropped there.")
    return max_diff, first_bad
```

The output is the whole diagnosis. `PASS` with `max_diff < 1e-6` means a perfect resume. A divergence *exactly at step N+1* with a *large* diff means the optimizer or scheduler was dropped (the first post-resume step is immediately wrong). A divergence that *starts small at N+1 and grows* means RNG/dataloader drift (the data slowly diverges). A divergence with the *first* run's values being random-init-like means a silent key drop. The test doesn't just tell you *that* the resume is broken; the *shape* of the failure tells you *which* bug.

#### Worked example: bisecting a real resume failure with the equivalence test

A run resumes with a visible spike. You suspect the optimizer, but instead of guessing you run the equivalence test and add state back *one piece at a time* — a bisection over the checkpoint fields. First run: save only `model` → `FAIL: diverges at step 101, diff 1.6e+0`. Add `optimizer` → `FAIL: diverges at step 101, diff 4.0e-2` (spike shrank 40×, so the optimizer was the big one, but something remains). Add `scheduler` → `FAIL: diverges at step 130, diff 8.0e-3` (now it matches at the boundary but drifts later — that's a *data/RNG* tell, since the divergence moved *downstream*). Add `rng` → `FAIL: diverges at step 145, diff 5e-3` (still drifting — the dataloader isn't stateful). Switch to a `StatefulDataLoader` and save its state → `PASS: max_diff 7e-7`. Five runs, each one isolating one field, and the *location* of the divergence (boundary vs downstream) told you at each step whether you were missing optimizer-class state (boundary spike) or data-class state (downstream drift). That is bisection applied to a checkpoint, and it beats reading code.

## 11. The Hugging Face Trainer resume contract

Most people don't hand-roll the loop — they use `transformers.Trainer` or `trl.SFTTrainer`, and resume via `trainer.train(resume_from_checkpoint=...)`. The good news: `Trainer` saves and restores model, optimizer, scheduler, scaler, RNG, and the global step *for you*. The checkpoint directory it writes contains `optimizer.pt`, `scheduler.pt`, `scaler.pt`, `rng_state.pth`, and `trainer_state.json` — exactly the state we enumerated. The resume bugs with `Trainer` are therefore not "it forgot the optimizer" but "you broke the contract that lets it restore correctly."

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="out",
    save_steps=500,
    save_total_limit=3,          # keep last 3; see best-vs-last below
    # these MUST be identical across the original run and the resume:
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    num_train_epochs=3,
    seed=42,
    data_seed=42,                # controls data shuffling RNG specifically
)
trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=eval_ds)

# resume: point at the checkpoint dir; Trainer restores ALL state
trainer.train(resume_from_checkpoint="out/checkpoint-5000")
```

The contract you must not break:

- **Same dataset, same order.** `Trainer` restores the RNG and *skips* the already-seen batches (it actually iterates and discards to reach the resume step, which is slow but correct). If you change the dataset or its size, the skip lands on the wrong data. `data_seed` pins the shuffle.
- **Same schedule-affecting args.** `num_train_epochs`, `warmup_steps`, `lr_scheduler_type`, `max_steps`, batch size, and accumulation steps all feed the scheduler's `max_steps`. Change any and the restored scheduler recomputes a different LR curve.
- **`ignore_data_skip=False`** (the default). If you set `ignore_data_skip=True` to avoid the slow batch-skipping, you *re-see* early data — the §6 bug — so only use it knowingly.

A quick way to confirm `Trainer` resumed correctly: check `trainer_state.json` after resume shows the right `global_step`, and log the LR for the first few steps — it should continue the pre-resume schedule, not restart warmup. If the resumed loss spikes with `Trainer`, the usual cause is a *changed training argument* that desynced the scheduler or data, not a missing state file.

## 12. Best vs last: which checkpoint do you resume?

A policy question that quietly causes bugs. You typically save two kinds of checkpoint: the **last** (most recent, for resuming after a crash) and the **best** (lowest val loss, for shipping). These are different objects with different purposes, and confusing them breaks resume.

The rule: **resume from `last`, ship from `best`.** Resuming from `best` rather than `last` after a crash means you discard all the training done *since* the best checkpoint and re-train it — wasted compute, and if `best` is much older you can lose hours. Worse, if your `best` checkpoint only saved the *model* (because it was meant for shipping, not resuming) and you try to resume from it, you hit every missing-state bug in this post at once. Keep the two concerns separate: `best` can be a weights-only export (smaller, no optimizer needed for inference); `last` must be a *full* checkpoint with all the state. `save_total_limit` should keep enough recent `last` checkpoints to survive a reclaim, and `best` is tracked separately.

| Checkpoint | Contents | Purpose | Resume from it? |
|---|---|---|---|
| `last` (latest) | Full state: model+opt+sched+scaler+RNG+step | Crash recovery / spot reclaim | Yes — this is the resume target |
| `best` (lowest val) | Model weights (often weights-only) | Shipping / inference | No — missing optimizer state |
| Periodic archive | Full state, kept every N steps | Rewind past a loss spike | Yes — pick the pre-spike one |

That last row is worth a note: periodic *full* checkpoints are also your tool for recovering from a loss spike (covered in [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence)) — you rewind to a checkpoint from *before* the spike and resume from there, sometimes with a lower LR or after skipping the bad batch. That recovery only works if the pre-spike checkpoint is a *full* checkpoint, which is one more reason `last`-style checkpoints must carry all the state. And knowing *when* to kill a run versus rewind it is its own decision, covered in [monitoring a run and when to kill it](/blog/machine-learning/debugging-training/monitoring-a-run-and-when-to-kill-it).

## 13. Why the spike recovers — the rebuild timeline in detail

Let's close the science loop on the most common case by tracing the optimizer rebuild step by step, because understanding the *shape* of the recovery is what lets you distinguish a fresh-optimizer spike (recovers) from a silent key drop (doesn't) from RNG drift (no spike, just drift). Figure 3 lays out the timeline.

![Timeline of an Adam optimizer rebuilding its lost moments from step zero with a huge mis-scaled step through to step fifty where the loss has fully recovered](/imgs/blogs/debugging-checkpoint-and-resume-3.png)

At **step 0** after a fresh-optimizer resume, $m=0$ and $v=0$. At **step 1**, as derived, the update is roughly $\eta \cdot \text{sign}(g)$ everywhere — uniformly mis-scaled, biggest harm to the well-adapted parameters. The loss spikes here. Through **steps 2–5**, the bias-correction terms $1/(1-\beta_1^t)$ and $1/(1-\beta_2^t)$ are still large (because $t$ is small), so the steps are still mis-scaled, but $v$ is beginning to accumulate the squared-gradient history. By **step ~15**, $v$ has accumulated enough recent gradient magnitude that the per-parameter scaling is roughly right for the parameters whose gradients are currently large; the loss is coming down. By **step ~30**, the effective window of $v$ has enough samples that the adaptive scaling is close to where the warm optimizer was, and by **step ~50** the loss has rejoined the trajectory it would have had. The full second-moment window is ~1000 steps, but the *loss* recovers in tens of steps because the parameters doing the most damage are the high-gradient ones, and those re-stabilize fastest.

This timeline is diagnostic: a spike that recovers in **tens of steps** is a fresh optimizer; a spike that **never** recovers is a silent key drop (random layers don't self-heal); a "spike" that is actually a **flat stall** is the scheduler; and **no spike but a slow drift** is RNG/data. The recovery shape is a fingerprint.

## 14. Case studies and known signatures

Real resume bugs leave recognizable marks. Here are patterns worth committing to memory.

**The spot-instance loss scar.** The most reported pattern in large-scale training on preemptible instances: a clean loss curve with a small, repeating spike at every resume boundary, each recovering in 20–50 steps. The textbook fresh-optimizer signature. The fix — save the optimizer state — is standard in every serious framework now precisely because this bug was so common in early large-model training. The cumulative cost was not just the recovery steps but the slightly-worse final models from repeated perturbation.

**The warmup-restart stall.** Reported across many finetuning setups: a resume where the loss doesn't spike but goes *flat* for a few hundred steps, then resumes descending. Logging the LR shows it cratered at the resume boundary — the classic fresh-scheduler-restarts-warmup signature. Especially common when people hand-roll the loop and remember the optimizer but forget the scheduler, since the scheduler is small and easy to overlook.

**The FSDP resume explosion.** In sharded training, a resume where the loss comes back near its *initial* value (not a 6× spike — a near-total reset) and then NaNs within tens of steps. This is the state-dict-type mismatch — loading a full state dict through a sharded path or vice versa — and it is severe because the *weights* themselves load wrong, not just the optimizer. The full diagnosis and fix is in [FSDP and sharding bugs](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs); the tell that distinguishes it from a fresh optimizer is the magnitude (near-total reset, not a recoverable spike) and the subsequent divergence.

**The augmentation-drift accuracy gap.** The quiet one from §5's worked example: no visible loss anomaly at any boundary, but a reproducible final-accuracy gap (often a fraction of a point to a point) between a resumed run and a continuous one, traced to un-restored RNG changing the augmentation stream. The lesson that generalizes: **the absence of a visible spike does not prove a correct resume.** Only the equivalence test does.

**The double-counting epoch.** A run configured to stop at `max_steps` that, after a resume that reset `global_step`, trains `max_steps` *again* — discovered only when the run took twice as long as budgeted and the LR schedule's cosine decay ran twice. The step-counter bug from §8, with a budget-busting blast radius.

**The corrupt-checkpoint preemption.** A team on aggressive spot instances saved checkpoints non-atomically (`torch.save` directly to the final path). One preemption landed *during* a save, leaving a truncated `checkpoint-12000/optimizer.pt`. The next resume loaded the model fine (it had been written first) but the optimizer file was truncated — `torch.load` raised an `UnpicklingError` on resume, and because the *previous* checkpoint had already been pruned by an over-eager `save_total_limit=1`, there was no fallback. The run had to restart from a much older archive. Two lessons converge here: write checkpoints atomically (temp-then-rename, §2), and keep enough recent checkpoints that a single corrupt one is survivable (`save_total_limit ≥ 2`, and a separate slower cadence of *archive* checkpoints you never prune).

**The optimizer-on-wrong-params refactor.** The nastiest signature in this list because the resume *appears* to work. A team added a per-layer learning-rate scheme (discriminative finetuning) on resume — splitting parameters into new param groups — without realizing the saved optimizer state was keyed by the *old* group layout. `optimizer.load_state_dict` either errored on a group-count mismatch (if they were lucky) or, when the counts happened to match, loaded the moments onto the *wrong* parameters. The loss didn't spike dramatically, but the run trained subtly worse for hundreds of steps because every parameter's adaptive scaling was borrowed from a *different* parameter's gradient history. The tell was that the equivalence test failed even though no single state field was "missing" — the optimizer state was present but *mis-keyed*. The fix: never change param-group structure across a resume, or rebuild the optimizer state explicitly with a remapping.

## 15. When this is (and isn't) your bug

A decisive section, because misattributing a resume symptom wastes more time than the bug itself.

**It IS a resume/checkpoint bug when:** the loss is continuous *until* a resume boundary and discontinuous *at* it; the discontinuity is *reproducible* across resumes; and the magnitude scales with how long you'd trained before saving (deeper = bigger jump for the fresh-optimizer case). The single cleanest confirmation is the resume-equivalence test failing — that is dispositive.

**It is NOT a resume bug — look elsewhere — when:**

- The loss spikes *mid-run*, away from any resume boundary. That is a loss spike, not a resume bug; see [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence) (a bad batch, an LR too high, a numeric overflow).
- The loss is *high from step 0* on a fresh run with no checkpoint involved. That is initialization, data, or LR — not resume.
- The resume "jump" is *within the run-to-run noise band* and the equivalence test *passes*. Then there is no bug; you are looking at normal stochasticity.
- The model loads but *inference* is wrong while *training* loss is fine. That is a train/eval mismatch or a serving-format issue, not a resume bug — different post.
- The resume errors *loudly* with a shape/key/device exception. That is a load-time failure (§9), adjacent but distinct from the silent loss-jump bugs — and easier, because it tells you exactly what's wrong.

The discriminating question is always: **does the symptom appear at a resume boundary and reproduce, and does the equivalence test fail?** If yes, it's here. If the equivalence test passes, stop blaming the checkpoint and bisect to data, optimization, or numerics using [the taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).

Figure 8 summarizes the whole before→after: the four-line table of weights-only versus full-state resume that you can hold in your head.

![Matrix contrasting weights-only and full-state resume across loss at resume, steps to recover, learning rate at resume, and the resume-equivalence match showing the full-state column passing every check](/imgs/blogs/debugging-checkpoint-and-resume-8.png)

## 16. Key takeaways

- **A checkpoint is the whole optimization state, not the weights.** Model, optimizer moments, scheduler, scaler, RNG, step/epoch, dataloader position, and EMA — save all of them or the resume diverges from a continuous run.
- **A loss spike on resume means the optimizer state was dropped.** Adam's lost moments make the first step ~$\eta\cdot\text{sign}(g)$, two orders of magnitude too big for well-adapted params; it spikes, then recovers in 20–50 steps as $v$ rebuilds.
- **A loss stall on resume means the scheduler reset.** A fresh scheduler restarts warmup, cratering the LR; log `optimizer.param_groups[0]["lr"]` and assert it is continuous at the boundary.
- **No spike but a slow drift means RNG or dataloader drift.** Restore Python/NumPy/Torch/CUDA RNG and use a stateful dataloader; the absence of a spike does not prove a correct resume.
- **The resume-equivalence test is the definitive check.** Train $2N$ straight versus $N$+resume+$N$; equal loss-for-loss (under determinism) proves every state was restored. Where it diverges tells you what's missing: boundary = optimizer/scheduler, downstream = data/RNG.
- **`strict=False` hides missing keys — never resume with it silently.** Assert the missing/unexpected lists are empty, or a refactor silently loads a randomly-initialized layer and the loss never recovers.
- **Resume from `last` (full state); ship from `best` (weights).** They are different objects; resuming from a weights-only `best` hits every missing-state bug at once.
- **Checkpoint on accumulation boundaries** so the gradient-accumulation counter is implicitly zero, and **write checkpoints atomically** (temp file then rename) so an interrupted save never poisons a resume.
- **You cannot verify a resume without determinism.** Turn on `use_deterministic_algorithms(True)` while debugging so the equivalence test is meaningful; the run-to-run noise otherwise masks the bug.

## 17. Further reading

- **PyTorch documentation — "Saving and Loading a General Checkpoint for Inference and/or Resuming Training"** and the `torch.optim` / `torch.optim.lr_scheduler` `state_dict`/`load_state_dict` references. The canonical API for everything in §2.
- **Kingma and Ba, "Adam: A Method for Stochastic Optimization" (2015).** The moment-estimation and bias-correction math behind why a fresh optimizer spikes (§1.2, §13).
- **Micikevicius et al., "Mixed Precision Training" (2018).** Dynamic loss scaling and the `GradScaler` state you must restore on an fp16 resume (§7.1).
- **PyTorch `torchdata` / `StatefulDataLoader` and Hugging Face `datasets` `IterableDataset.state_dict()`.** Resumable dataloaders for the §6 dataloader-position bug.
- **Hugging Face `transformers` Trainer docs — `resume_from_checkpoint`, `save_total_limit`, `data_seed`, `ignore_data_skip`.** The resume contract in §11.
- **PyTorch reproducibility notes — `torch.use_deterministic_algorithms`, cuDNN/cuBLAS determinism.** The determinism prerequisite for the equivalence test (§5.3).
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the master decision tree), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone checklist), [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) (the determinism prerequisite), [FSDP and sharding bugs](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs) (the sharded resume explosion), [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence) (rewind-and-resume recovery), and [monitoring a run and when to kill it](/blog/machine-learning/debugging-training/monitoring-a-run-and-when-to-kill-it) (kill vs rewind).

---
title: "Reading the Loss Curve as a Diagnostic: From Curve Shape to Root Cause"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn to read a training loss curve like a field guide, mapping each canonical shape to its root cause so you can localize a stalled, spiking, or secretly-overfit run in minutes."
tags:
  [
    "debugging",
    "model-training",
    "loss-curve",
    "diagnostics",
    "finetuning",
    "deep-learning",
    "pytorch",
    "optimization",
    "overfitting",
    "monitoring",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-1.png"
---

A loss curve is the cheapest, richest instrument you will ever attach to a training run, and most engineers read maybe ten percent of what it is telling them. You glance at it, see that the line is going down, and move on. Then three days later you discover the "converged" model predicts the majority class on everything, or the run you babysat all weekend was silently spiking on a recurring corrupt batch that your smoothed dashboard averaged into a gentle decline. The curve was screaming. You were reading the wrong channel.

This post is a field guide to reading that one instrument well. The thesis is simple: **the shape of a loss curve is a diagnostic signature**, and a small catalogue of canonical shapes — flat-at-chance, slow-start, the spike, divergence to NaN, sawtooth, the train-val gap, the suspiciously-smooth descent — maps almost one-to-one onto root causes that live in one of the six places a training bug can hide: data, optimization, model code, numerics, systems, or evaluation. If you can name the shape, you have already bisected your search to one or two suspects, and you can confirm the right one with a single cheap test before you touch any code. That is the entire game of this series — *symptom to suspect to confirming test to fix* — and the loss curve is where the symptom shows up first.

![A workflow graph showing how a raw and smoothed loss curve, plus grad-norm and a train-val split, route a training symptom to one of six suspects](/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-1.png)

We will do three things at once for every shape, because this series is technical, practical, and scientific by mandate. The **science**: why does a never-learning curve sit at exactly $\ln(C)$ nats for $C$ classes? Why does too-high a learning rate produce a spike and then a NaN rather than a smooth slowdown? Why does an exponential-moving-average smoother provably hide a spike, and by how much? The **practice**: the exact logging and plotting setup that makes these shapes visible at all — per-step not per-epoch, raw alongside smoothed, the running max, grad-norm overlaid, separate train and val axes, log-scale y — with copy-and-run code for Weights & Biases and TensorBoard and a PyTorch hook. And the **proof**: a real before-and-after where a curve that was "going down fine" on the smoothed plot was spiking on the raw max, and clipping plus a skip-batch guard took the per-step max from 9.2 back to 1.5. By the end you should be able to take a screenshot of any stalled, spiking, or secretly-overfit run and read it to a root cause without re-running anything.

A word on what a loss curve is *not*. It is not the metric you ship on. A falling cross-entropy with flat accuracy is its own diagnostic shape, and we will spend real time on the gap between loss and metric, because the day you confuse them is the day you celebrate a leak. Keep that distinction in your pocket; we return to it.

## 1. The instrument and how to read it at all

Before any shape means anything, you have to be plotting the thing that carries the signal. The single most common reason engineers cannot read their loss curve is that they are looking at a smoothed, per-epoch, linear-y, train-only line, which is the one configuration guaranteed to hide every interesting failure.

Start from what a loss value *is*. At step $t$, your model produces a loss $\ell_t$ on the current mini-batch — a single scalar, the mean per-example loss over that batch. The sequence $\ell_1, \ell_2, \ldots$ is a noisy time series. The noise comes from mini-batch sampling: each batch is a different random draw, so even a perfectly-trained model has batch-to-batch variance in its loss. The *signal* is the trend, the spikes, the periodicity, and the gap to a held-out set. Your job is to plot in a way that exposes signal and tames noise without erasing the spikes that *are* signal.

Five layers of logging get you there, and each one reveals a shape the layer above it hides.

![A vertical stack of five logging layers, from per-step loss at the top through running max, grad-norm, train-val split, to log-scale axis at the base](/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-4.png)

**Log per step, not per epoch.** If you log one loss value per epoch, a spike that lasts fifty steps inside a ten-thousand-step epoch is a single averaged point — invisible. Per-step logging (or per-N-steps for large runs) is non-negotiable for seeing the spike, the sawtooth, and the exact step where a NaN appears. The cost is log volume; log every step early, then throttle to every 10 or 50 steps once the run is stable.

**Log the raw curve and a smoothed curve together.** The raw curve is jagged and the eye struggles with the trend; a smoothed line makes the trend obvious. But — and this is the trap that costs people weekends — *the smoothed line can hide a spike entirely*. We prove exactly how badly in the science block below. Plot both, on the same axes.

**Log a running max alongside the mean.** Over a sliding window, track $\max_t \ell_t$. The mean and the smoothed curve pull spikes down; the max pulls them up. A run whose smoothed loss glides from 1.4 to 1.1 while its windowed max touches 9.2 has a bad-batch problem the average is laundering. The running max is the cheapest spike detector you can add.

**Log grad-norm beside the loss.** The gradient norm $\lVert g_t \rVert$ — the L2 norm of the full flattened gradient — is the single best companion signal to disambiguate a loss spike. A spike with a simultaneous grad-norm spike is an optimization or numerics event; a spike with a quiet grad-norm is more likely a single weird batch with a large per-example loss but bounded gradient. We use this overlay constantly.

**Separate train and val onto their own series, and offer a log-scale y.** The generalization gap only exists if you can see train and val on comparable axes. And a log-scale y-axis turns a flat-looking plateau near zero into a readable slope, and makes the early warmup region legible instead of a vertical cliff.

Here is the logging setup I reach for first. It is framework-agnostic in spirit; this version uses Weights & Biases, and the TensorBoard variant is a one-line swap.

```python
import torch
import wandb
from collections import deque

wandb.init(project="loss-curve-diagnostics", config={"lr": 3e-4})

# Sliding window for the running max — the spike detector the EMA hides.
window = deque(maxlen=50)
ema = None
EMA_BETA = 0.98  # smoothing factor; higher = smoother = hides more

def grad_global_norm(model):
    # The L2 norm of the full flattened gradient: sqrt(sum of all grad^2).
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5

for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()

    gnorm = grad_global_norm(model)                 # BEFORE clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    raw = loss.item()
    window.append(raw)
    ema = raw if ema is None else EMA_BETA * ema + (1 - EMA_BETA) * raw

    wandb.log({
        "loss/raw": raw,                            # see the spike
        "loss/ema": ema,                            # see the trend
        "loss/window_max": max(window),             # the spike detector
        "grad_norm": gnorm,                         # disambiguate spikes
        "lr": optimizer.param_groups[0]["lr"],      # explain warmup/plateau
    }, step=step)
```

Two details earn their keep. We compute the grad-norm *before* clipping, because the clipped norm is capped at `max_norm` by construction and tells you nothing about how bad the spike was; the pre-clip norm is the instrument. And we log `loss/window_max`, the single channel that would have caught the bad-batch bug we open with. The TensorBoard version replaces `wandb.log({...}, step=step)` with a handful of `writer.add_scalar("loss/raw", raw, step)` calls; everything else is identical.

With those five channels live, the shapes below become legible. Without them, you are reading a blurred photograph and guessing.

### The noise floor: how much jitter is normal

Reading a curve well means knowing how much per-step wobble is *expected* before you call anything a spike. The mini-batch loss is an average of $B$ per-example losses, so by the standard error of the mean its batch-to-batch standard deviation is roughly $\sigma_\ell / \sqrt{B}$, where $\sigma_\ell$ is the standard deviation of the per-example loss across the dataset. Two consequences fall out directly. First, smaller batches are noisier: halving the batch size raises the expected jitter by $\sqrt{2}$, so a batch-size-8 run legitimately rattles around far more than a batch-size-512 run, and excursions that would alarm you on the large-batch curve are just sampling noise on the small-batch one. Second, the noise floor *shrinks as the model improves* on easy examples but stays high while a hard subset still produces large losses — so a curve whose jitter refuses to shrink late in training is telling you a stubborn hard or mislabeled subset remains. The practical rule: estimate the noise band as a few times $\sigma_\ell/\sqrt{B}$, draw it mentally around the smoothed line, and only treat an excursion as a real spike when it punches well outside that band. A "spike" inside the noise band is not a bug; it is statistics.

This is also why the smoothing window is a genuine trade-off, not a free knob. A wider EMA (higher $\beta$) suppresses noise — good for seeing the trend — but, as we prove in the worked bisection below, it suppresses *real* spikes by the same factor $(1-\beta)$, so the smoother that makes your trend pretty is the one that hides your bad batch. There is no single window that both reveals the trend and preserves the spike, which is precisely why you log raw, smoothed, and running-max as three separate channels rather than picking one.

### Why loss is not your metric, and why that matters here

One more framing before the catalogue, because it changes how you read every curve. Cross-entropy loss and the accuracy (or AUC, or WER, or BLEU) you actually care about are *correlated but not identical*. Loss is a smooth, per-example, probability-calibrated quantity; the metric is usually a thresholded, discrete count. You can drive loss down by making correct predictions more confident without changing a single decision, so loss falls while accuracy sits flat. Conversely, near a decision boundary a tiny loss change can flip many predictions, so accuracy can jump while loss barely moves.

The practical consequence: **a falling loss with a flat metric is a diagnostic shape in its own right**, usually meaning "the model is getting more confident about what it already knew, but not learning new distinctions" — common in late training, or when the data lacks the signal for the next distinction, or when your metric is measuring something the loss does not optimize. Always plot at least one real metric beside the loss. When they disagree, the disagreement is information, not noise.

## 2. Flat at chance: the run that never learns

The most demoralizing curve is the flat line. The loss starts at some value and stays there, batch after batch, epoch after epoch. The good news is that this shape is the easiest to read, because it has a precise, computable signature: **a model that has learned nothing sits at the chance-level loss**, and you can calculate that number in advance.

### The science: chance loss is $\ln(C)$

For multi-class classification with $C$ classes and cross-entropy loss, the loss of a model that outputs the uniform distribution — assigning probability $1/C$ to every class — is

$$\ell_{\text{chance}} = -\sum_{c=1}^{C} \frac{1}{C}\log\frac{1}{C} \cdot \mathbb{1}[c = y] = -\log\frac{1}{C} = \ln(C).$$

More carefully: cross-entropy on a single example with true class $y$ is $-\log p_y$ where $p_y$ is the predicted probability of the true class. If the model predicts uniform, $p_y = 1/C$ for every example regardless of $y$, so the per-example loss is $-\log(1/C) = \ln C$, and the mean is the same. This is the loss of *no information*. A model that has learned nothing — because no gradient is flowing, the learning rate is zero, the parameters are frozen, or the labels have been shuffled to noise — will sit here.

The numbers are worth memorizing for the objectives you train most:

- 10 classes (MNIST, CIFAR-10): $\ln(10) \approx 2.303$ nats.
- 1000 classes (ImageNet): $\ln(1000) \approx 6.908$ nats.
- Binary cross-entropy, balanced: $\ln(2) \approx 0.693$ nats.
- Language-model token cross-entropy over a 50,000-token vocabulary: $\ln(50000) \approx 10.82$ nats — though in practice unigram frequency makes the *real* chance baseline lower, since predicting the unigram distribution beats uniform; the uniform value is the absolute ceiling on "learned nothing."
- Mean-squared error on a target standardized to unit variance: predicting the mean gives MSE $= \mathrm{Var}(y) = 1.0$.

![A matrix mapping five training objectives to their chance-loss formula, numeric chance value, and what sitting at that value means](/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-7.png)

That table is a lookup you should keep nearby. The moment you see a flat curve, compare its value to the chance loss for your objective. If your 10-class classifier is pinned at 2.30, it is not "training slowly" — it is not training at all. If it is pinned at 4.5, something is partially working but stuck. The exact number tells you which story you are in.

There is a subtlety the careful reader should hold: a curve can be flat *above* chance, which is a different and more interesting failure. A 10-class run stuck at 2.30 is learning nothing. A run stuck at, say, 2.31 after a warmup might be at chance because the final layer's bias absorbed the class prior but no features are flowing — which still points at a no-grad-flow cause. A run flat at 5.0 on a 10-class problem (well above $\ln 10$) usually means the loss is mis-scaled or the labels are wrong, because you cannot do *worse* than uniform by much unless you are confidently wrong, which itself is a clue.

### The diagnostic: is gradient actually flowing?

A flat-at-chance curve has a short list of causes, all of which share one observable signature: **the gradient is not reaching the parameters that need to change.** So the confirming test is to print the gradient norm per parameter group and look for zeros.

```python
def report_grad_flow(model, loss):
    """Run one backward pass and print which params have no gradient."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    dead, alive = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            dead.append((name, "requires_grad=False"))
        elif p.grad is None:
            dead.append((name, "grad is None (not in graph)"))
        elif p.grad.detach().abs().sum().item() == 0.0:
            dead.append((name, "grad all zeros"))
        else:
            alive.append((name, p.grad.norm().item()))
    print(f"ALIVE: {len(alive)} params  DEAD: {len(dead)} params")
    for name, why in dead[:20]:
        print(f"  DEAD  {name:40s} {why}")
    for name, g in alive[:5]:
        print(f"  alive {name:40s} grad_norm={g:.3e}")
```

This one function distinguishes the four canonical flat-at-chance causes:

- **Learning rate is zero (or absurdly tiny).** Gradients flow — the params have non-zero `grad` — but the update $\Delta\theta = -\eta g$ is zero or negligible. `report_grad_flow` shows everything alive; the tell is that the parameters do not *change* between steps. Add a check: log `param_norm` and watch it stay constant. The fix is to set a real learning rate; we have a whole sibling post on why the learning rate is almost always the problem.
- **No gradient flow.** Some parameter has `requires_grad=False` by accident, or a `.detach()` / `torch.no_grad()` block severed the graph, or a non-differentiable op (argmax, a hard threshold) sits between loss and the frozen submodule. `report_grad_flow` prints those names as DEAD. The fix is to find the break in the graph.
- **Frozen parameters.** Common in finetuning: you froze the backbone to train a head, but froze the head too, or never unfroze anything. The DEAD list is your whole model.
- **Shuffled / corrupted labels.** Gradients flow and parameters change, but the labels carry no learnable signal — they have been randomly permuted, or the label column is misaligned by one row, so the loss cannot drop below chance because there is nothing to learn. `report_grad_flow` shows everything alive and the params *do* move, but the loss stays at $\ln C$. This is the sneaky one: the optimization machinery is healthy, the data is poison. The confirming test is the **overfit-one-batch test** — take two batches, turn off shuffling and augmentation, and try to drive the loss to zero. If a model with healthy gradients cannot memorize four examples, the labels are not learnable. (That test is so central it has its own post in this series.)

#### Worked example: a BERT finetune stuck at 2.30

A team finetunes a BERT classifier on a 10-class intent-classification dataset. The loss starts at 2.31 and after 2,000 steps is at 2.29. Accuracy on the dev set is 10.4% — chance. They suspect the learning rate.

We run `report_grad_flow` on one batch. Output: `ALIVE: 2 params  DEAD: 199 params`, and every DEAD line reads `requires_grad=False`. The two alive params are the classifier head's weight and bias. Someone had called `model.bert.requires_grad_(False)` to "save memory" and then *also* forgot that with a frozen backbone and a randomly-initialized head, the head alone over a frozen `[CLS]` representation can in principle learn — but here the frozen `[CLS]` came from a backbone that had never seen this domain, so the linear head over those features genuinely could not separate the classes much above chance, and the tiny head was learning at a crawl.

The number told the story before the fix did. Chance loss for 10 classes is $\ln 10 = 2.303$; the run sat at 2.29, a hair below chance, exactly where a barely-useful linear probe lands. We unfroze the top four transformer layers, set the backbone learning rate to $2\times10^{-5}$ and the head to $1\times10^{-4}$, and the loss fell from 2.30 to 0.41 over the first 400 steps, dev accuracy 88%. Before: flat at 2.30, 199 dead params. After: descending, 56 alive params, accuracy 88%. The confirming instrument was the grad-flow print; the curve's *value* (2.30, not 1.5) was what told us "nothing is learning" rather than "learning slowly."

## 3. Slow start then a drop: warmup, init, and a too-cold learning rate

Not every flat region is fatal. A very common and *healthy* shape is a slow start — the loss barely moves for a few hundred steps — followed by a clean drop. The diagnostic question is whether the flat region is a warmup you intended or a learning rate so low the run is crawling toward a cliff it will fall off in an hour you do not have.

### The science: why warmup exists

In the first steps of training, the parameters are at their random initialization, the adaptive-optimizer statistics (Adam's first and second moment estimates) are uninitialized and biased toward zero, and the gradients can be large and poorly-conditioned. A learning-rate warmup — linearly ramping $\eta$ from near zero up to its target over, say, the first 1–10% of steps — lets the optimizer's running statistics stabilize and keeps the early, noisy updates from knocking the parameters into a bad region. During warmup, the loss legitimately moves slowly because the effective step size is small by design.

So a slow-then-drop curve where the drop coincides with the end of warmup is *correct*. You read it by overlaying the learning-rate schedule on the loss plot — which is exactly why the logging snippet above logs `lr` every step. If the loss elbow lines up with the warmup peak, your run is healthy and you can stop worrying.

The failure version is a learning rate that is simply too low for the whole run, warmup or not. Here the loss does eventually drop, but glacially, and it will take ten times longer than it should — burning GPU-hours and your patience. The tell is that the slope, even after warmup, is far shallower than a healthy run on a similar problem. The confirming test is the **learning-rate range test**: run a few hundred steps while exponentially increasing the learning rate, and plot loss against LR. The loss falls, bottoms out, then explodes; the steepest-descent LR is roughly an order of magnitude below the explosion point. If your chosen LR sits far to the left of the steep region, it is too cold.

```python
# LR range test: ramp LR exponentially, watch where loss is steepest, then explodes.
import math

lrs, losses = [], []
lr = 1e-7
mult = (1e1 / 1e-7) ** (1 / 200)   # span 1e-7 -> 1e1 over 200 steps
for step, batch in zip(range(200), loader):
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    lrs.append(lr)
    losses.append(loss.item())
    if loss.item() > 4 * losses[0]:   # loss blew up; stop
        break
    lr *= mult
# Plot losses vs lrs on a log-x axis. Pick LR ~ 1 decade below the minimum.
```

The practical rule: warmup explains a slow start that ends in a clean elbow at the warmup peak; a globally-too-low LR explains a slow start that never gets steep. Read the LR overlay first — it answers the question for free.

#### Worked example: a finetune crawling at a tenth of the right LR

A team finetunes a 7B model on eight A100s and reports that the loss "is going down, just slowly." Over 6 hours it crept from 1.95 to 1.78 — real motion, no spikes, no NaN, so nobody suspected a bug. But the slope after warmup was suspiciously shallow: a healthy finetune on this data should reach 1.4 in well under an hour. We ran the LR-range test on a few hundred steps and plotted loss against LR on a log-x axis. The steep-descent region sat around $2\times10^{-5}$ and the loss exploded just past $3\times10^{-4}$. The configured learning rate was $1\times10^{-6}$ — fully a decade and a half to the *left* of the steep region, deep in the flat "barely moving" zone. Someone had copied a from-scratch pretraining LR schedule into a finetune config and never re-tuned it.

The cost framing makes the point land. At roughly \$2 per A100 GPU-hour, eight GPUs for 6 hours of crawling burned about \$96 producing a loss of 1.78 that a correctly-tuned run reaches in 25 minutes. We set the LR to $2\times10^{-5}$ with a 50-step warmup and reran: the loss fell from 1.95 to 1.36 in 22 minutes (about \$6 of compute), then continued to 1.05 by the end of the first epoch. Before: 6 hours, \$96, loss 1.78, slope a flat crawl. After: 22 minutes to beat the old number, slope steep right after warmup. The curve shape — a slow start that *never got steep* — was the entire diagnosis; the LR-range plot confirmed it in three hundred steps. A crawling-but-descending curve is not a happy curve; it is a too-cold LR quietly spending your budget.

## 4. The spike: transient bump or the first crack of divergence

Now the dramatic shape. The loss is descending nicely, and then — a vertical jump. 1.4 to 9.2 in one step. This is the spike, and it is the single most important shape to read correctly, because it splits into two completely different stories: a *transient* spike that the run shrugs off, and a *terminal* spike that is the first crack of a divergence to NaN. Mistaking one for the other wastes either a good run (you kill a recoverable one) or a bad one (you babysit a run that was never coming back).

![A timeline of one training run walking through warmup, a fast drop, a spike at step 800, then forking into recovery or divergence to NaN](/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-6.png)

### The science: where spikes come from

A spike is a sudden, large increase in loss, which means the last update moved the parameters into a much worse region. There are three common mechanisms.

**Learning rate too high for the local curvature.** Gradient descent assumes the loss is approximately locally quadratic with curvature bounded by some $L$ (the Lipschitz constant of the gradient). The classical stability condition for plain gradient descent is $\eta < 2/L$. When the loss landscape passes through a sharp region — high curvature, large $L$ — a learning rate that was fine in a flat region now violates the bound, and a single step overshoots the valley and lands higher than it started. With momentum and Adam the exact bound is messier, but the intuition holds: a spike is an overshoot in a locally-sharp region, and large models hit sharp regions stochastically, which is why loss spikes are a famous nuisance in large-language-model pretraining.

**A bad batch.** One mini-batch contains pathological data — a corrupted example, an extreme outlier, a mislabeled cluster, a sequence of mostly-padding tokens that should have been masked from the loss. That batch produces a huge per-example loss and a huge gradient, and the update it induces spikes the loss. If the bad batch is recurring (the same corrupt file every epoch, or a dataloader that re-serves it), you get a periodic spike at the same point in each epoch — which is also a sawtooth story we cover below.

**A numerics event.** An overflow in fp16, a $\log 0$ from a probability that rounded to exactly zero, a division by a near-zero variance — these produce an `inf` or a huge finite value that, depending on where it lands, spikes the loss or goes straight to NaN.

### The diagnostic: overlay grad-norm and watch the recovery

The confirming instrument is the grad-norm overlay plus the recovery behavior, and the decision is mechanical.

![A decision graph triaging a loss spike by reading grad-norm at the spike and whether the curve recovers, splitting transient from terminal](/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-8.png)

- **Read the grad-norm at the spike step.** If the loss spiked but the pre-clip grad-norm stayed near its normal range, the spike is probably a single batch with a large *loss* but a bounded *gradient* — a per-example outlier, often recoverable. If the grad-norm *also* spiked (say from a normal 2.0 to 8,000), the optimizer took a giant step, which is the overshoot-or-numerics story.
- **Watch whether the loss recovers.** A transient spike comes back down within a handful of steps — the run was knocked sideways and the next few updates pull it back. A terminal spike does *not* recover; the loss stays elevated or climbs, and within tens to hundreds of steps you see NaN. Recovery within ~50 steps is the strongest single signal that a spike was transient.

So: **grad-norm bounded plus quick recovery means transient; grad-norm exploding plus no recovery means terminal.** That single rule resolves the majority of spikes without any deeper investigation.

The fix differs accordingly. A transient spike from a bad batch is handled with **gradient clipping** (cap the global grad-norm at, say, 1.0, so no single batch can take a giant step) and optionally a **skip-batch guard** that detects an anomalous loss and skips the optimizer step for that batch. A terminal spike from too-high an LR is handled by lowering the LR (and adding warmup), and the safest recovery is to **rewind to the last good checkpoint** and resume with the lower LR, because once a run has spiked terminally its optimizer state may be poisoned.

```python
# Skip-batch guard: detect an anomalous loss/grad and skip the step, log it.
SPIKE_FACTOR = 4.0          # loss more than 4x the running median = suspicious
running = deque(maxlen=100)

for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    raw = loss.item()

    median = sorted(running)[len(running)//2] if len(running) >= 20 else raw
    if running and raw > SPIKE_FACTOR * median:
        # Anomalous batch: log it, do NOT step the optimizer, do NOT poison Adam state.
        wandb.log({"skipped_batch_loss": raw, "skipped_at_step": step}, step=step)
        running.append(median)   # keep the median honest
        continue

    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    running.append(raw)
    wandb.log({"loss/raw": raw, "grad_norm": float(gnorm)}, step=step)
```

This guard is conservative by design — it only skips truly anomalous batches, and it logs every skip so you can go inspect what was in them (often the inspection finds a data bug you then fix at the source). For the deep mechanics of spikes in large models, including optimizer-state effects and rewind strategies, see the sibling post on loss spikes and divergence.

## 5. Divergence to NaN: when the curve leaves the building

A spike that does not recover ends in divergence: the loss climbs, then reports `NaN` or `inf`, and from that step on every number in the model is poison (any arithmetic with NaN yields NaN, so the corruption spreads through the next forward pass instantly). The curve's signature is a climb-then-flatline-at-NaN, or sometimes a clean descent that hits a NaN cliff with no warning at all — and *that* second variant is the tell that it is numerics, not optimization.

### The science: where NaN and Inf are born

NaN and Inf come from a short list of arithmetic sins, and knowing the list lets you predict which one your curve is hitting:

- **$\log 0$ (or $\log$ of a negative).** Cross-entropy computes $-\log p$; if a probability underflows to exactly 0.0, you get $-\log 0 = +\infty$. This is why you compute cross-entropy from *logits* with a numerically-stable `log_softmax`, never from probabilities you softmaxed yourself.
- **Division by zero or near-zero.** A variance that collapses to zero in a normalization layer, a denominator in an attention or loss term that was not floored with an `eps`.
- **`exp` overflow.** $\exp(x)$ for large $x$ overflows the float range; in fp16 the max representable value is 65,504, so $\exp(x)$ overflows for $x > \ln(65504) \approx 11.1$. An un-stabilized softmax over large logits hits this.
- **`sqrt` of a negative**, from a variance that went slightly negative due to floating-point error.
- **fp16 overflow in the forward or backward pass.** The representable range of fp16 tops out at 65,504 and its smallest normal positive value is about $6.1\times10^{-5}$; an activation or gradient outside that range becomes `inf` or flushes to 0. This is the heart of the mixed-precision story.

There is a reason a NaN, once born, ends the run instantly rather than passing as a transient: NaN is *absorbing* under arithmetic. Every operation that touches a NaN returns NaN — $\text{NaN} + x = \text{NaN}$, $\text{NaN} \times 0 = \text{NaN}$, even comparisons return false — so a single non-finite value in one weight propagates through the next forward pass to every output it can reach, and the gradient that flows back writes NaN into the optimizer's moment estimates, poisoning parameters that were healthy. This is why the curve does not "recover" from a true NaN the way it recovers from a spike: a spike is a large but finite excursion the next few good updates can pull back, while a NaN corrupts the state irreversibly within one step. The operational consequence is that you must catch a NaN at the step it is *born*, not three thousand steps later when the whole model is poison — which is exactly what the per-step guard below is for.

The "clean descent then sudden NaN cliff" signature is diagnostic: a loss that was falling smoothly and then NaNs *without a preceding spike* is almost always a numerics event (a $\log 0$ on a particular input, an fp16 overflow on a particular activation), not an optimization instability — because an optimization blowup announces itself with a grad-norm climb first. When you see the cliff with no preceding grad-norm climb, stop blaming the learning rate and go hunt the numeric op.

### The diagnostic: anomaly detection and bisection by step

PyTorch ships the right instrument. `torch.autograd.set_detect_anomaly(True)` makes the autograd engine check every backward op for NaN/Inf and raise at the *exact* operation that produced it, with a stack trace pointing at the forward line that created the offending tensor. It is slow — it roughly doubles backward time — so you enable it only when hunting, not in production.

```python
import torch

# 1) Catch the exact op that produces NaN in the backward pass.
torch.autograd.set_detect_anomaly(True)   # SLOW: hunting only

# 2) Or guard the forward pass directly and dump the offending batch.
def forward_with_guards(model, batch, step):
    loss = model(batch).loss
    if not torch.isfinite(loss):
        torch.save(batch, f"bad_batch_step{step}.pt")   # capture for inspection
        raise FloatingPointError(f"Non-finite loss {loss.item()} at step {step}")
    return loss

# 3) Bisect-by-step: the NaN appears at a deterministic step? Set a seed,
#    run to step N-1, then single-step with anomaly detection on.
```

The bisection method for a NaN is the same disciplined search the whole series preaches. *Is it deterministic by step?* Set the seed and rerun; if the NaN reappears at the same step, you have a reproducible bug and you can capture the exact batch (snippet above saves it) and inspect it for a corrupt label, an empty sequence, an out-of-range target index. *Is it precision-dependent?* Rerun the same step in fp32; if the NaN vanishes in fp32 but appears in fp16, it is an fp16 range problem — switch to bf16 (which has the same exponent range as fp32, so it does not overflow where fp16 does, at the cost of mantissa precision) or add loss scaling. *Is it layer-localized?* A forward hook that checks `torch.isfinite` on every module's output pinpoints the first layer to go non-finite. The dedicated post on hunting NaNs and Infs walks the full layer-bisection; here the point is that the *curve shape* — clean descent into a cliff, versus a spike-then-climb — already tells you whether to reach for the anomaly detector (numerics) or the LR knob (optimization).

#### Worked example: an fp16 finetune that NaNs at step 412

A vision-language finetune in fp16 AMP runs clean for a while, then the loss prints `nan` at step 412 — same step on every rerun with a fixed seed. The smoothed curve showed a perfectly healthy descent right up to the cliff; *no* preceding spike. That "clean descent into a cliff" shape is the fingerprint of numerics, so we did not touch the learning rate.

We enabled `set_detect_anomaly(True)` and reran to step 412. The trace pointed at a `log` inside the contrastive loss: a similarity logit had grown large enough that, after the fp16 softmax, one denominator term overflowed to `inf`, and the subsequent `log` of the normalized probability produced `nan`. The pre-clip grad-norm in the steps before 412 was flat at about 1.8 — no optimization blowup, confirming numerics. We reran step 412 in fp32 and the NaN disappeared, the second confirming test.

The fix was two lines: switch the autocast dtype from fp16 to bf16 (bf16's 8-bit exponent matches fp32's range, so the same logits no longer overflow), and add a logit clamp before the softmax as a belt-and-suspenders guard. Before: NaN at step 412, every run. After: 3 full epochs, no non-finite step, final loss 0.19, and the grad-norm stayed bounded at ~2.0 throughout. The whole hunt took twenty minutes because the curve shape pre-selected the suspect: a cliff with no preceding spike means go to the numeric op, not the optimizer.

## 6. Sawtooth and periodic bumps: the dataloader is talking

A subtler shape: the loss descends overall, but rides a periodic ripple — a sawtooth, or a recurring bump that lands at the same place in every epoch. This is one of the most under-recognized signatures, and it almost always means the *order* of your data carries structure it should not, or a recurring batch is pathological.

### The science: why order leaves a periodic signature

If your data is not shuffled — or is shuffled once and then iterated in a fixed order every epoch, or sorted by some field (length, class, time) — then each epoch presents the model with the same sequence of distributions. Easy regions of the data produce low loss; hard regions produce high loss; and because the sequence repeats every epoch, the loss curve inherits a period equal to the number of steps per epoch. A run with 1,000 steps per epoch that shows a bump every 1,000 steps is telling you the data order is periodic and the model is responding to *where in the epoch it is*, not just to learning.

The most damaging version is **sorted-but-not-shuffled** data. If your dataset is sorted by class (all the 0s, then all the 1s), each mini-batch is nearly single-class, the gradient pulls hard toward predicting that one class, and the next batch yanks it the other way. The loss sawtooths violently and the model never settles, because batch-mean gradients with low class diversity are high-variance and biased within each batch. Shuffling restores i.i.d.-like batches and the sawtooth vanishes.

A second version is a **single recurring bad batch**: one corrupted file, served at the same index every epoch, spikes the loss at the same phase each time. The periodic spike at a fixed epoch-position is the giveaway, and it ties back to the spike section — the difference is that here the spike is *periodic*, which immediately implicates data order rather than a stochastic optimization event.

### The diagnostic: plot against epoch-phase and inspect the offending index

The confirming test is to plot the loss against *position within the epoch* (step modulo steps-per-epoch) and see whether the bumps align. If they do, the order is the cause.

```python
steps_per_epoch = len(loader)

for step, batch in enumerate(loader):
    loss = train_step(batch)
    phase = step % steps_per_epoch          # position within the epoch
    wandb.log({
        "loss/raw": loss,
        "loss/vs_phase": loss,              # plot this against `phase`, not `step`
        "epoch_phase": phase,
    }, step=step)
    # If a spike recurs at the SAME phase every epoch, capture that batch:
    if loss > 4.0:
        wandb.log({"spike_phase": phase}, step=step)
```

If the bumps align by phase, the fixes are direct: turn on shuffling (`DataLoader(..., shuffle=True)` and verify the sampler is actually a `RandomSampler`, not a `SequentialSampler` you set somewhere), set a per-epoch shuffle seed so each epoch reorders, and if a single recurring index is the culprit, capture and inspect that example for corruption. A classic gotcha: with multiple dataloader workers, an incorrectly-set `worker_init_fn` can make every worker draw the *same* sequence, reintroducing periodicity even with shuffling nominally on — so verify the actual order, do not trust the flag.

#### Worked example: a violent sawtooth on a sorted CSV

A tabular classifier sawtooths between loss 0.2 and 1.8 every few steps and never converges; the smoothed curve is a flat blur because the oscillation dominates. We plot loss against epoch-phase: the sawtooth period is exactly the batch size relationship to the class blocks. Inspecting the raw CSV, the rows are sorted by label — all class 0, then all class 1. Each batch is single-class, the model is whipsawed, and the batch gradient variance is enormous because every batch is a biased sample of one class. We set `shuffle=True` and add a fixed shuffle seed for reproducibility. Before: loss oscillating 0.2 to 1.8, no convergence, dev AUC 0.61. After: a smooth descent to 0.18, dev AUC 0.93. The shape — a fast, regular sawtooth riding the descent — pointed straight at data order, and plotting against phase confirmed it in one chart.

## 7. The train-val gap: four shapes in one plot

Everything so far reads a single curve. The richest diagnostics come from reading *two* curves together — training loss and held-out validation loss — because their *relative* motion separates four very different bugs. This is the generalization gap, and it deserves its own catalogue.

![A decision tree splitting the relative motion of the train and validation curves into overfitting, underfitting, leakage, and eval noise](/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-5.png)

### The science: the generalization gap

The training loss measures fit to the data the model is updating on; the validation loss measures fit to data the model has never updated on. The gap between them — $\ell_{\text{val}} - \ell_{\text{train}}$ — is an estimate of how much the model has fit *noise specific to the training set* rather than *signal that generalizes*. A model with enough capacity can drive training loss arbitrarily low by memorizing, but memorized training noise does not transfer, so validation loss stops following and eventually rises. The shape of the two curves together tells you which regime you are in.

**Train down, val up — overfitting.** The classic shape: both curves descend together, then the validation curve bottoms out and turns up while training keeps falling. The gap grows. The model has moved from learning signal to memorizing noise. The minimum of the validation curve is the right early-stopping point, and the size of the final gap is a measure of how much you over-trained. The fixes are the regularization toolbox — early stopping, weight decay, dropout, data augmentation, more data — and the diagnostic is simply tracking the gap and stopping at the val minimum.

**Both plateau high — underfitting or capacity/LR limits.** Both curves descend a little and then flatten *together* at a high value, with little gap. The model is not memorizing — there is no gap — but it is also not learning enough. This points away from overfitting and toward insufficient capacity, a learning rate too low to reach a good basin, too few steps, or a representation that cannot express the target function. The fix is more capacity, a better LR, longer training, or better features — not regularization, which would make underfitting worse.

**Val below train — regularization, dropout, or a leak.** Counterintuitively, validation loss can sit *below* training loss. There are two benign explanations and one alarming one. Benign: dropout and other stochastic regularization are active during training (raising training loss) but disabled at eval (lowering validation loss), so a modest val-below-train gap is *expected* with heavy dropout; and the training loss is often a running average over a whole epoch during which the model was improving, while validation is measured at the end with the better weights, which can also put val below train. Alarming: a **data leak** — validation examples that are duplicates or near-duplicates of training examples, or a feature that encodes the target — makes the validation set artificially easy, so val loss is suspiciously low. The test that separates these is to *audit the split for duplicates and check whether disabling dropout removes the inversion*; if val is below train even with dropout off and no duplicates, suspect a leak.

**Val noisy — small val set or an eval bug.** The validation curve jumps around wildly from one evaluation to the next while training is smooth. Usually this means the validation set is too small (so its loss has high variance — a 200-example val set on a hard task has a large standard error), or the evaluation has a bug (different preprocessing than training, dropout left on at eval, a `model.eval()` you forgot to call). The fix is a bigger val set or fixing the eval path; the test is to evaluate the *same* val set twice and check the numbers match (they must, if eval is deterministic), which catches a non-deterministic eval bug immediately.

#### Worked example: a 0.4-nat gap that was honest, and one that was a leak

Two finetunes, two gaps, two completely different stories — and the shape distinguishes them.

Run A: a language-model finetune where training loss falls to 0.8 and validation loss bottoms at 1.2 (gap +0.4 nats) around epoch 2, then validation turns up while training keeps falling toward 0.5. Classic overfitting. The fix was to early-stop at the epoch-2 validation minimum and add a touch of weight decay; the deployed checkpoint was the epoch-2 one with val loss 1.2, not the epoch-5 one with the lower *training* loss. The gap was real and the shape — val turning up while train falls — named it.

Run B: a tabular model where validation loss sat *below* training loss by 0.3 the entire run, and validation AUC was a gorgeous 0.97. The val-below-train inversion with no dropout in the model was the red flag. We audited the split and found that a customer-ID-derived feature leaked the target through a near-deterministic mapping, and 6% of validation rows were exact duplicates of training rows. We removed the leaked feature and de-duplicated across the split. Honest validation AUC came back at 0.78 — a 0.19-AUC drop that was entirely leak. Before: val AUC 0.97, val loss below train. After: val AUC 0.78, val loss correctly above train. The shape (val *below* train with no regularization to explain it) was the tell; the duplicate audit was the confirming test. This is exactly the kind of silent killer the data-leakage post in this series exists to catch.

### Two more shapes worth naming: the staircase and the resume jump

Two shapes routinely scare engineers who have not catalogued them, and both have benign and malign versions you can tell apart by overlaying one other signal.

The **staircase**: the loss descends, flattens, then suddenly *steps down* again to a lower plateau, and repeats. If you overlay the learning-rate schedule and the steps line up with scheduled LR drops (a step or cosine schedule that cuts the LR by, say, 10x at fixed milestones), the staircase is *healthy* — a lower LR lets the optimizer settle into a tighter basin, and the loss drops to a new floor. The malign version is a staircase with *no* corresponding LR change, which usually means a phase change in the data (a new data source mixed in at a milestone, or a curriculum that switched) or a periodic evaluation artifact. So: staircase plus LR drops at the same steps means it is the schedule working; staircase with a flat LR means go look at what changed in the data pipeline at those steps.

The **resume jump**: you stop a run, resume from a checkpoint, and the loss is suddenly *higher* than where you left it, then re-descends. A small jump that recovers within a few steps is benign — the dataloader RNG state restarted, so the first few post-resume batches are a different draw, or the running normalization statistics needed a moment to re-warm. A *large* jump that does not recover is a real bug: the optimizer state (Adam's moments), the LR-scheduler position, the gradient-scaler state, or the RNG was not restored, so the resume is effectively a fresh, badly-initialized optimizer hitting a trained model. The confirming test is to check whether the resumed loss at step $N$ matches the pre-stop loss at step $N$ within the noise band; if it jumps by more than a couple of standard errors and stays elevated, your checkpoint is not restoring everything. The dedicated checkpoint-and-resume post in this series covers the full state-restoration checklist; the curve signature is your first warning.

## 8. Too smooth, too good: the curve that lies by being perfect

The last shape is the most dangerous because it does not look like a problem. The loss descends *too* cleanly — no batch-to-batch noise, an implausibly smooth glide to a near-zero value — or the validation metric is suspiciously, deliriously good (99.8% accuracy on a task where the state of the art is 94%). A curve that looks too good is, far more often than not, a measurement error or a leak, not a triumph.

### The science: why "too smooth" is suspicious

Real mini-batch training is noisy. Each batch is a random draw, so the per-step loss *must* have batch-to-batch variance proportional to the within-batch loss variance over the square root of the batch size. A loss curve with *no* visible per-step noise is physically suspicious: either you are accidentally plotting a heavily-smoothed series and calling it raw, or every batch is producing nearly the same loss, which happens when every batch contains the same easy signal — a leak. The absence of expected noise is itself a diagnostic.

Likewise, a validation metric far above the plausible ceiling for the task is evidence of a leak or an eval bug, by the same logic as the gap section: if the held-out set is contaminated with training examples, or a feature encodes the target, or the eval is accidentally scoring against the training labels, the number is too good because it is not measuring generalization. The science here is information-theoretic: a model cannot extract more signal than the data contains, so a result that beats the information available is measuring something other than what you think.

### The diagnostic: the shuffle-label test and the holdout audit

Two cheap tests catch most "too good" curves.

The **shuffle-label test**: randomly permute the labels so there is *no* learnable relationship between inputs and targets, and retrain. A correct pipeline should now be unable to beat chance — training loss should hover near $\ln C$ and validation accuracy near $1/C$. If your model *still* gets good validation accuracy on shuffled labels, you have a leak: information is reaching the model through a path other than the (now-random) labels, almost always a feature that encodes the target or a train-val contamination.

```python
import numpy as np
from sklearn.model_selection import cross_val_score

# Shuffle-label test: with random labels, honest CV should collapse to chance.
y_shuffled = np.random.permutation(y)
scores = cross_val_score(pipeline, X, y_shuffled, cv=5, scoring="roc_auc")
print(f"CV AUC on SHUFFLED labels: {scores.mean():.3f} +/- {scores.std():.3f}")
# Expect ~0.50. If it's 0.80, a feature is leaking the target.
```

The **holdout audit**: check for duplicate and near-duplicate examples across the train and validation splits (exact hash match for tabular/text, perceptual hash or embedding cosine similarity for images), and verify the split was made *before* any preprocessing that could leak (a scaler or imputer fit on the full dataset before splitting leaks distribution information from val into train). For time-series, verify no future data leaked into the past via a non-temporal split. The number of cross-split duplicates is a direct measure of contamination.

For a vision or text classifier, the fastest "too good" sanity check is to *look at the examples the model gets right with highest confidence* — if they are all trivially easy or you spot the same image in train and val, you have found the leak by eye in minutes.

## 9. A complete worked bisection: the run that was going down fine

Let me put the whole field guide together on one realistic run, because reading a single shape is easy and reading a *real* run — where the smoothed dashboard is lying to you — is the skill that matters.

The setup: a transformer finetune, fp16 AMP, that everyone agreed was "going down fine." The shared dashboard showed an EMA-smoothed training loss gliding from 1.4 to 1.1 over 1,000 steps. No alarms. But the final model was subtly worse than a previous run, and nobody could say why.

![A two-column before-and-after contrasting an EMA-smoothed curve that hid a spike against raw, running-max, and grad-norm channels that exposed it](/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-3.png)

**Step 1 — read the right channel.** The dashboard showed only the smoothed loss. We turned on `loss/raw`, `loss/window_max`, and `grad_norm`. Immediately the story changed: the raw loss was spiking to 9.2 at step 800, and — this is the part that explains the whole mystery — *it recurred at the same phase in every subsequent epoch*. The EMA with $\beta = 0.99$ had buried it.

Here is the science of *why* the EMA hid it, made precise. An exponential moving average with smoothing factor $\beta$ updates as $\hat{\ell}_t = \beta \hat{\ell}_{t-1} + (1-\beta)\ell_t$. A single spike of size $S$ above the baseline contributes only $(1-\beta)S$ to the EMA on the spike step, and decays geometrically as $\beta^k$ thereafter. With $\beta = 0.99$, a spike from a baseline of 1.4 up to 9.2 — an excess of 7.8 — moves the EMA by just $(1-0.99)\times 7.8 = 0.078$, lifting the smoothed curve from 1.40 to about 1.48 for a single step before it decays back. On a y-axis spanning 1.0 to 1.5, an 0.078 blip is invisible. The smoother did exactly what smoothers do — it attenuated a high-frequency event by a factor of $(1-\beta)$ — and an 0.078 bump on a smoothed curve reads as "going down fine." This is the precise, quantitative reason you must *also* plot the raw curve and the running max: the max channel showed 9.2, undecayed, exactly where the EMA showed 1.48.

**Step 2 — bisect to the suspect.** A spike that recurs at a fixed epoch-phase is, per the sawtooth section, a *data-order* signature: the same batch is spiking every epoch. We overlaid the grad-norm: at step 800 (and its per-epoch recurrences) the pre-clip grad-norm jumped from a normal ~2.0 to ~8,000. So this was a bad-batch spike with a real gradient explosion, recurring because the dataloader served the same pathological batch at the same index each epoch. That is two suspects collapsed into one: data (a bad batch) manifesting as an optimization event (a grad-norm spike), recurring because of order.

**Step 3 — confirm with a test.** We added the skip-batch guard from Section 4, which logs the offending batch. The captured batch (`bad_batch_step800.pt`) contained a handful of examples whose target sequence was longer than the truncated input could support after tokenization, producing an enormous per-example loss and the gradient spike. Confirming test passed: the spike was a specific, capturable, recurring bad batch.

**Step 4 — fix and measure.** Two changes: gradient clipping at `max_norm=1.0` (so even if a bad batch slips through, no single step can spike the parameters) and the skip-batch guard (so the known-pathological batch is skipped and logged rather than applied). We also fixed the root cause in the data pipeline — the truncation bug that produced the over-long targets. The before-and-after on the instruments:

| Instrument | Before (smoothed only) | After (clip + skip + data fix) |
| --- | --- | --- |
| Smoothed loss @ step 1000 | 1.10 (looked fine) | 1.06 |
| Raw window-max | 9.2 @ step 800, recurring | 1.5, no recurrence |
| Pre-clip grad-norm | ~8,000 at spikes | bounded ~2.0 throughout |
| Skipped batches | n/a (none detected) | 1 per epoch, logged + later fixed |
| Final dev metric | baseline minus 1.3 points | baseline plus 0.4 points |

The point of the whole exercise: the smoothed curve was not *wrong*, it was *incomplete*, and reading only it cost the team a degraded model and days of confusion. The shapes were all there — a recurring spike (bad batch + order), a grad-norm explosion (optimization signature of a data event) — but only on the channels they were not plotting. Adding three log lines turned an invisible bug into a five-minute fix.

## 10. The diagnostic table: shape to suspect to test to fix

Everything above compresses into one lookup you can pin to your monitor. When you see a shape, this is the bisection in a row.

![A matrix mapping seven canonical loss-curve shapes to their leading suspect, the confirming test, and the likely fix](/imgs/blogs/reading-the-loss-curve-as-a-diagnostic-2.png)

| Curve shape | Leading suspect | Where it hides | Confirming test | Likely fix |
| --- | --- | --- | --- | --- |
| Flat at chance ($\ln C$) | No learning: LR=0, frozen, no grad flow, shuffled labels | Optimization / model code / data | `report_grad_flow`; overfit-one-batch | Unfreeze, set LR, fix graph break, fix labels |
| Slow start then clean drop at warmup peak | Warmup (healthy) or init | Optimization | Overlay LR schedule; does the elbow match warmup end | Nothing — it is correct; or shorten warmup |
| Slow start that never gets steep | LR too low globally | Optimization | LR-range test | Raise LR by ~3–10x |
| Spike then recovery (<50 steps), grad-norm bounded | Transient bad batch / outlier | Data / numerics | Capture the batch; read grad-norm at spike | Clip 1.0, skip-batch guard, fix the batch |
| Spike then climb, grad-norm explodes, no recovery | LR too high / terminal instability | Optimization | Grad-norm at spike; does loss recover | Lower LR, add warmup, rewind to checkpoint |
| Clean descent into a NaN cliff (no preceding spike) | Numerics: log0, fp16 overflow, /0 | Numerics | `detect_anomaly`; rerun step in fp32 | bf16, eps/clamp, loss scaling |
| Sawtooth / periodic bump at epoch phase | Data not shuffled / recurring bad batch | Data | Plot loss vs epoch-phase | `shuffle=True`, fix sampler/worker_init_fn |
| Train down, val up; gap grows | Overfitting | Evaluation / capacity | Track val minimum | Early stop, weight decay, dropout, more data |
| Both plateau high, no gap | Underfit / capacity / LR | Optimization / model code | LR-range test; add capacity | More capacity, higher LR, longer, better features |
| Val below train (no dropout to explain) | Leak or duplicate split | Evaluation / data | Duplicate audit; shuffle-label test | Fix split, remove leaked feature, de-dup |
| Val noisy, train smooth | Small val set / eval bug | Evaluation | Eval same set twice; check determinism | Bigger val set, fix eval path, call `model.eval()` |
| Too smooth / too good | Leak or eval bug | Data / evaluation | Shuffle-label test; holdout audit | Find the leak, fix the eval, re-measure |

And the companion table — which instrument catches which class — because the *curve* is only the first instrument and the disambiguation usually needs a second:

| Bug class | Curve symptom | Second instrument that confirms |
| --- | --- | --- |
| No grad flow | Flat at chance | Per-param grad-norm (zeros) |
| LR too high | Spike, then climb | Pre-clip grad-norm (explodes to 1e3–1e4) |
| LR too low | Crawling descent | LR-range test plot |
| Numerics | Clean descent into NaN | `detect_anomaly` stack trace; fp32 rerun |
| Bad batch | Spike (transient or periodic) | Captured batch contents; loss-vs-phase |
| Overfitting | Train↓ val↑ | The val-minimum step |
| Leak | Too-good / val-below-train | Shuffle-label test; duplicate audit |
| Loss≠metric | Loss↓, metric flat | The metric plotted beside the loss |

## 11. Case studies and real signatures

A few well-known, real signatures to calibrate your eye, cited honestly.

**fp16 loss scaling, from "Mixed Precision Training" (Micikevicius et al., 2018).** The paper that made fp16 training practical documented exactly the divergence-to-NaN shape we covered: in fp16, gradients whose magnitude falls below the smallest representable normal value (about $6.1\times10^{-5}$) flush to zero, and large activations overflow past 65,504 to `inf`. Their fix, *loss scaling* — multiplying the loss by a large factor before the backward pass to shift the gradient distribution up into fp16's representable range, then unscaling before the optimizer step — is the standard answer to the fp16 NaN cliff. The shape (clean descent then a NaN cliff with no grad-norm warning) and the fix (bf16 or loss scaling) trace directly to this work. This is the deep mechanism behind the mixed-precision debugging sibling post.

**Loss spikes in large-model pretraining.** Multiple large-language-model technical reports (the PaLM and OPT logbooks among the most candid) document loss spikes during pretraining and the now-standard recovery: when a spike is terminal, *rewind to a checkpoint before the spike, skip the batches that triggered it, and resume*. The signature they describe is exactly ours — a loss jump with a grad-norm spike, sometimes recoverable and sometimes terminal — and the operational fix (rewind + skip) is why production training loops checkpoint frequently and log the running max. The exact recovery recipe is the subject of the loss-spikes-and-divergence post.

**Confident learning and label noise in benchmark test sets (Northcutt et al., 2021).** The work behind `cleanlab` found pervasive label errors even in the *test* sets of canonical benchmarks — on the order of a few percent of labels mislabeled across datasets including ImageNet and others — which caps achievable accuracy and shows up as a loss that plateaus above where a clean dataset would allow. The signature is a stubborn loss floor and a model "confidently wrong" on a consistent subset; the diagnostic is loss-ranking or confident-learning to surface the mislabeled rows. When your loss plateaus higher than the task should allow and the hardest examples are confidently-wrong, suspect label noise, not capacity. (The numbers here are approximate and dataset-dependent; the cited paper has the exact per-dataset figures.)

**The Kaggle leakage post-mortem genre.** A recurring competition story: a model with a gorgeous cross-validation AUC (often 0.95+) collapses on the private leaderboard because a feature leaked the target or the CV split leaked across groups/time. The curve signature is the too-good / val-below-train shape, and the fix is `GroupKFold` or `TimeSeriesSplit` plus removing the leaked feature — the same diagnosis as our Run B worked example. The exact AUC drops vary, but a leak commonly costs 0.10–0.20 AUC when corrected, which is precisely the gap that should make a too-good curve suspicious.

## 12. When this is (and isn't) your bug

Reading the curve well also means knowing when the curve is *not* the right instrument and when a shape points elsewhere than your instinct.

- **A clean descent into a NaN cliff is numerics, not your learning rate.** If the loss fell smoothly with a bounded grad-norm and then NaN'd with no preceding spike, lowering the LR will not save you — go find the `log`, the `exp`, or the fp16 overflow. The absence of a grad-norm climb before the NaN is the discriminator.
- **A flat-at-chance curve with healthy gradients is data, not optimization.** If `report_grad_flow` shows everything alive and the params *move* but the loss stays at $\ln C$, stop tuning the LR — your labels are not learnable (shuffled, misaligned, or noise). Run the overfit-one-batch test: a model that cannot memorize four examples has a data problem.
- **If overfit-one-batch passes, stop blaming the model.** If your model *can* drive loss to ~0 on one or two batches, the forward pass, the loss, and the gradient path are correct. A full-dataset failure after that is data or optimization or evaluation — not model code. This single test rules an entire suspect out.
- **A too-good validation number is a leak until proven otherwise.** Do not ship a 0.97 AUC you did not earn. The shuffle-label test and the duplicate audit are cheap; run them before you celebrate.
- **A loss that falls while the metric is flat is not necessarily a bug.** Late in training, loss often keeps falling (the model grows more confident) while accuracy plateaus (no new decisions flip). Confirm by plotting the metric; if the metric is genuinely flat *and* you are early in training, then suspect that the loss is optimizing the wrong thing (wrong reduction, wrong target alignment) — but late-training confidence-sharpening is benign.
- **A noisy curve on a tiny batch size is expected, not a spike.** With batch size 4, per-step loss variance is large by construction (the variance scales with $1/\sqrt{B}$); do not chase "spikes" that are just sampling noise. Smooth, look at the running max, and only worry about excursions far above the noise band.

The meta-rule: the loss curve narrows the suspect list, but the *confirming test* is what closes the case. Never fix on the strength of the shape alone — read the shape, predict the suspect, run the one cheap test, then fix. That is the bisection discipline the whole series is built on, and the master decision tree in the taxonomy post is the map that ties every shape here to its branch.

## 13. Key takeaways

- **A flat curve sits at $\ln(C)$ when nothing is learning.** Memorize the chance loss for your objective (10 classes → 2.30, 1000 → 6.91, binary → 0.69, unit-variance MSE → 1.0). A curve pinned there is not slow, it is dead — confirm with a per-param grad-norm print.
- **Always log per-step, raw, smoothed, and the running max — together.** An EMA with $\beta=0.99$ attenuates a spike by $(1-\beta)$, turning a jump to 9.2 into an invisible 0.08 bump on the smoothed line. The running max is your cheapest spike detector.
- **Overlay grad-norm to triage a spike.** Bounded grad-norm plus recovery within ~50 steps means transient (clip and continue); exploding grad-norm with no recovery means terminal (rewind to a checkpoint, lower the LR).
- **A clean descent into a NaN cliff with no preceding spike is numerics.** Reach for `detect_anomaly` and an fp32 rerun, not the learning-rate knob. fp16's range floor is $\approx 6.1\times10^{-5}$ and ceiling 65,504; bf16 or loss scaling fixes the cliff.
- **A periodic bump at a fixed epoch-phase is a data-order bug.** Plot loss against step-modulo-epoch; if the bumps align, turn on shuffling and verify the sampler and `worker_init_fn` actually randomize.
- **Read train and val together — their relative motion names the bug.** Train↓ val↑ is overfitting (stop at the val minimum); both flat-high is underfitting; val-below-train with no dropout is a leak; noisy val is a small set or an eval bug.
- **A too-good or too-smooth curve is a leak or an eval bug until proven otherwise.** The shuffle-label test (should collapse to chance) and the duplicate-across-split audit catch most of them in minutes.
- **Loss is not your metric.** Plot at least one real metric beside the loss; a falling loss with a flat metric is its own diagnostic, and a metric far above the task's plausible ceiling is evidence of a leak.
- **The shape narrows the suspect; the confirming test closes the case.** Never fix on shape alone — read shape, predict suspect, run the one cheap test, then fix.

## Further reading

- **"Mixed Precision Training"**, Micikevicius et al., 2018 (arXiv:1710.03740) — the fp16 representable range, gradient underflow, and loss scaling; the mechanism behind the NaN-cliff shape.
- **"A Recipe for Training Neural Networks"**, Andrej Karpathy, 2019 — the canonical practitioner's guide to overfitting one batch, looking at your data, and reading curves; this post is in its lineage.
- **"Pervasive Label Errors in Test Sets..."**, Northcutt, Athalye, Mueller, 2021 — confident learning and the label-noise floor that shows up as a stubborn loss plateau; the basis of `cleanlab`.
- **PyTorch autograd anomaly detection docs** — `torch.autograd.set_detect_anomaly`, for pinning the exact op that produces a NaN in the backward pass.
- **Weights & Biases and TensorBoard logging docs** — for the per-step, raw-plus-smoothed, multi-axis dashboards this post depends on.
- Within this series: [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the master symptom→suspect→test→fix decision tree), [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone), [Loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence), [Hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs), [The learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem), and [Instrumenting a training run: what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log).

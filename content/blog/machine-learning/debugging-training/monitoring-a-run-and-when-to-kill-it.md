---
title: "Monitoring a Run and When to Kill It: Early Stopping, Dashboards, and the Dead Run"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn to read a live training run like an instrument panel, catch a dead run in the first hundred steps, and apply a disciplined kill-or-keep decision rule that saves GPU-weeks."
tags:
  [
    "debugging",
    "model-training",
    "early-stopping",
    "monitoring",
    "finetuning",
    "deep-learning",
    "pytorch",
    "wandb",
    "tensorboard",
    "optimization",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/monitoring-a-run-and-when-to-kill-it-1.png"
---

It is 11pm and you launch a finetune that will take three days and roughly forty dollars of GPU time. You glance at the dashboard once — the loss line is going down, which is what loss lines do — and you go to bed. Three days later you discover the run plateaued at step 300, sat at exactly the chance-level loss for the remaining forty-nine thousand seven hundred steps, and produced a model that predicts the majority class on everything. The instruments were screaming the whole time. You were asleep, and more to the point, you were not asking the run the one question that mattered: are you actually learning, or just spending money?

The opposite failure is just as expensive and far more common among people who have been burned once. You launch the same run, watch it like a hawk, and at step 1,200 the validation loss ticks up by 0.02. You panic, kill it, change the learning rate, and relaunch. You do this six times. Every one of those runs was a slow-starter that would have converged beautifully if you had let it cook, and you have now spent more compute thrashing than the original run would have cost to finish. You were too impatient, and impatience on a noisy signal is its own bug.

Both of these are monitoring failures, and they are not minor. The arithmetic is brutal: a dead run caught at step 80 instead of step 50,000 saves you essentially the entire run, and across a team running dozens of experiments a month, the difference between catching doomed runs early and catching them late is measured in GPU-weeks and real money. This post is about the discipline that sits between "launch and pray" and "babysit and thrash" — how to read a live run, how to tell a slow-starter from a dead one, and how to encode a kill-or-keep decision into a rule and then into automation so the run watches itself.

![A workflow graph showing how loss slope, gradient norm, and a chance-level reference route a live training run into healthy, dying, or dead, and then into keep, wait, or kill](/imgs/blogs/monitoring-a-run-and-when-to-kill-it-1.png)

We will do three things at once for every idea, because this series is technical, practical, and scientific by mandate. The **science**: why a never-learning run sits at exactly $\ln(C)$ nats for $C$ classes, why validation noise sets a floor on how patient your early stopping must be, and why early stopping is a form of regularization rather than just a budget cap. The **practice**: runnable code — a correct `EarlyStopping` callback that monitors the right metric and restores the best weights, a kill-on-NaN-or-stall callback, a Hugging Face `Trainer` and PyTorch Lightning wiring, and a first-100-steps preflight you run before you trust any long run. And the **proof**: a concrete before-and-after where a preflight plus an auto-kill guard caught a doomed run at step 80 instead of step 50,000, and an early-stopping fix that recovered a validation AUC of 0.78 the naive version had thrown away at 0.71.

This is one post in a series whose recurring frame is that **a training bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and a disciplined debugger bisects to the right one before touching code**. Monitoring is where you read the symptom that starts that bisection, and the kill decision is what you do when the bisection says the run is not worth finishing. By the end you should be able to look at any live run and answer, with a rule and not a vibe, the only question that matters at 11pm: let it cook, wait and watch, or kill it now.

## 1. The only question: are you learning, or just spending?

Every monitoring decision reduces to one question asked repeatedly: is this run making progress toward the goal, fast enough to be worth its remaining cost? Everything else — the dashboards, the early-stopping patience, the kill rules — is machinery for answering that question reliably on a noisy signal under time pressure.

The reason it is hard is that the loss curve is a *noisy* time series, and the human eye is terrible at distinguishing "noisy descent that is genuinely working" from "noisy flatness that is going nowhere." Mini-batch sampling injects variance: each batch is a different random draw, so even a perfectly healthy run jitters from step to step. Layer an exponential-moving-average smoother on top, as most dashboards do by default, and you have a line that always looks like it is gently going somewhere whether or not it is. The smoother is a confidence trick your dashboard plays on you, and the antidote is to read the *right* signals, not the prettiest one.

There are three verdicts a live run can earn, and they map to three actions. A **healthy** run is one whose loss is falling below its chance-level reference and whose gradient norm is in a sane range — let it cook, the right action is patience. A **dying** run is one showing a transient pathology — a loss spike, a brief grad-norm excursion — that may or may not recover; the right action is to wait and watch, possibly with a guardrail like gradient clipping. A **dead** run is one stuck at chance with a gradient norm at or near zero, learning nothing and provably going to keep learning nothing; the right action is to kill it and fix the bug before relaunching. The whole art is assigning the right verdict quickly and cheaply, ideally in the first hundred steps where the evidence is already conclusive and the spend is still trivial.

Notice what monitoring is *not*. It is not a substitute for the pre-flight checks that prevent a dead run from ever launching — the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) is the single best of these, and we lean on it hard below. And it is not a substitute for the [instrumentation](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) that makes the signals readable in the first place; you cannot monitor what you did not log. Monitoring is the live act of reading instruments you already set up, against thresholds you decided in advance, to make a decision under uncertainty. The rest of this post builds those thresholds.

## 2. The chance-level reference: knowing what "not learning" looks like

You cannot tell whether a run is learning unless you know the number it would sit at if it learned nothing. That number is not zero, and it is not a vibe — for most losses it is a closed-form constant you can compute before you ever launch. This is the single most useful piece of arithmetic in monitoring, and most engineers never do it.

Consider $C$-class classification with cross-entropy loss. A model that has learned nothing outputs a uniform distribution over the $C$ classes — probability $1/C$ on each. The cross-entropy of the true label under a uniform prediction is

$$\ell_{\text{chance}} = -\log\left(\frac{1}{C}\right) = \log C.$$

In nats (natural log, which is what PyTorch's `nn.CrossEntropyLoss` reports) this is $\ln C$. For binary classification, $C = 2$, so $\ell_{\text{chance}} = \ln 2 \approx 0.693$. For a 10-class problem like CIFAR-10, $\ell_{\text{chance}} = \ln 10 \approx 2.303$. For a language model over a vocabulary of $V = 50{,}257$ tokens, an untrained model sits near $\ln(50{,}257) \approx 10.82$ nats, which is why fresh LM training starts at a loss around 10–11 and the first job of training is to claw down to the unigram entropy and below.

This gives you a hard reference line to draw on your loss dashboard. If your 10-class run is sitting at 2.30 after a thousand steps, it is at chance — it has learned exactly nothing, no matter how the smoothed line wiggles. If your LM is stuck at 10.8, same story. The chance line converts "is it learning?" from a judgment call into a comparison: are we meaningfully below $\ln C$, and is the gap growing?

There is a subtlety worth internalizing. The chance level for a class-imbalanced problem is *not* $\ln C$ — it is the entropy of the marginal label distribution, because a model can trivially learn the class priors and beat uniform. For labels with prior probabilities $p_1, \ldots, p_C$, the best constant-prediction loss is the marginal entropy

$$H(Y) = -\sum_{c=1}^{C} p_c \log p_c,$$

which is strictly less than $\ln C$ whenever the classes are imbalanced. So for a 99%/1% binary problem, the "learned nothing useful" floor is $H(Y) \approx -(0.99\ln 0.99 + 0.01\ln 0.01) \approx 0.056$ nats, not $\ln 2 = 0.693$. A run that drops from 0.693 to 0.056 and stops has learned only the prior; it looks like progress against the $\ln C$ line but is dead against the right reference. This is exactly the trap covered in [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies) — pick the reference that matches your label distribution.

For regression with mean-squared-error loss, the chance reference is the variance of the target: a model predicting the constant mean achieves MSE equal to $\text{Var}(y)$. Standardize your targets to unit variance and the chance MSE is 1.0, which is a clean line to watch against.

#### Worked example: reading three runs against their chance lines

Three runs land on your dashboard, all "going down" on the smoothed plot. Run A is a 10-class image classifier at loss 2.28 after 2,000 steps. Run B is the same task at 1.40. Run C is a binary fraud detector (1% positive) at loss 0.052 after 5,000 steps.

Compute the references. Run A's chance line is $\ln 10 = 2.303$; it is at 2.28, which is 0.02 below chance — within batch noise of "learned nothing." It is dead. Run B is at 1.40 against the same 2.303 line, a gap of 0.90 nats and clearly learning. It is healthy. Run C looks great at 0.052 against the $\ln 2 = 0.693$ line, a huge gap — except the right reference for a 1% problem is the marginal entropy $\approx 0.056$. Run C is at 0.052, four-thousandths of a nat below the prior-only floor. It has learned the prior and essentially nothing else. Against the wrong line it looks like a triumph; against the right line it is barely alive. Three identical-looking smoothed descents, three completely different verdicts, all decided by one constant you compute before launch.

Here is the helper I keep in every project so the reference line is never a guess:

```python
import math
import torch

def chance_ce_loss(num_classes: int, label_priors: torch.Tensor | None = None) -> float:
    """Chance-level cross-entropy in nats.

    Uniform-prediction floor is ln(C). For imbalanced labels the real
    floor is the marginal label entropy H(Y), which is lower.
    """
    if label_priors is None:
        return math.log(num_classes)
    p = label_priors / label_priors.sum()
    # add tiny eps so a zero-count class does not produce nan
    return float(-(p * (p + 1e-12).log()).sum())

# 10-class balanced: 2.303 nats
print(chance_ce_loss(10))
# binary, 1% positive: marginal entropy ~0.056 nats, NOT ln(2)=0.693
priors = torch.tensor([0.99, 0.01])
print(chance_ce_loss(2, priors))
```

Draw that number as a horizontal line on your loss panel. A run that hugs it is not learning, full stop, and the smoothed curve's wiggle is noise around a dead level.

## 3. The first 100 steps: the cheapest place to catch a doomed run

The economics of monitoring are dominated by one fact: the evidence that a run is doomed is almost always present in the first hundred steps, but the *cost* of the run is spread across all of them. Catch the doom early and you pay almost nothing; catch it late and you have paid for the whole thing. So the highest-leverage monitoring discipline is a structured pre-flight and first-100-steps checklist that you run on every long run before you walk away from it.

![A timeline of the first hundred steps showing five ordered checks from overfitting a single batch through loss movement, gradient sanity, throughput, to a cheap keep-or-kill verdict](/imgs/blogs/monitoring-a-run-and-when-to-kill-it-2.png)

The checklist is ordered so that the cheapest, most decisive checks come first. Each one rules a class of failure in or out before the next.

**Step 0 — overfit one batch.** Before the real run, take a single batch and train on it alone for a few hundred steps. A correctly-wired model with a working loss and optimizer will drive the loss to near zero on one batch — it has enough capacity to memorize a handful of examples. If it *cannot* drive a single batch to near zero, your model, loss, or data pipeline is broken, and no amount of full-dataset training will fix it. This is the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test), and it is the pre-flight that prevents you from ever launching a dead run. It costs seconds and catches a startling fraction of doomed runs before they start.

**Steps 1–20 — is the loss moving off chance?** Once the real run starts, the very first thing to read is whether the loss is moving *down and away from $\ln C$*. Not converged — just moving. A run that starts at 2.30 and is at 2.28 after twenty steps with a flat trend is suspicious; a run that is at 2.10 and falling is alive. Warmup complicates this (the loss may be flat-ish during a short LR warmup), which is exactly why we separate "flat because warmup" from "flat because dead" in the next section.

**Steps 1–20 — is the gradient norm sane?** Log the global gradient norm $\lVert g \rVert_2$ every step. A healthy run has a grad norm that is non-zero and not exploding — typically somewhere in the 0.1 to 10 range depending on the model and loss scale, settling over the first dozens of steps. A grad norm pinned at exactly zero means no gradient is flowing (frozen parameters, LR of zero, a detached graph) and the run is dead on arrival. A grad norm of 1e4 and climbing means you are about to spike to NaN. Both verdicts are available in twenty steps.

**Steps 20–50 — is throughput acceptable?** Read tokens-per-second or samples-per-second and, if you can, model FLOPs utilization (MFU — the fraction of the GPU's peak FLOPs you are actually using). A run that is functionally correct but running at 30% GPU utilization because of a dataloader bottleneck is not a *kill*, but it is a "fix before you commit three days," covered in [the GPU is idle: throughput debugging](/blog/machine-learning/debugging-training/the-gpu-is-idle-throughput-debugging). Catch it now, not on day three when you realize the run will take a week instead of three days.

**Step 100 — the cheap verdict.** By a hundred steps you have a loss trend, a gradient-norm trend, a throughput number, and a passing or failing one-batch overfit test. That is enough to decide keep-or-kill with high confidence, at a cost of roughly a hundred steps out of fifty thousand — two-tenths of one percent of the run. This is the single best moment to make the decision, and the entire point of the checklist is to *force* you to make it then rather than three days later.

Here is the first-100-steps preflight as a runnable callback you can drop into a PyTorch loop. It checks each gate and refuses to let a run continue silently past a hundred steps if it is obviously dead.

```python
import math
import torch

class FirstHundredSteps:
    """Pre-flight gate: catch a dead run in the first ~100 steps."""

    def __init__(self, num_classes, throughput_floor_tok_s=None,
                 grad_lo=1e-4, grad_hi=1e3, check_until=100):
        self.chance = math.log(num_classes)
        self.throughput_floor = throughput_floor_tok_s
        self.grad_lo, self.grad_hi = grad_lo, grad_hi
        self.check_until = check_until
        self.start_loss = None

    def __call__(self, step, loss, grad_norm, tokens_per_s=None):
        if step == 0:
            self.start_loss = loss
        if step > self.check_until:
            return  # gate only governs the cheap window

        # 1. gradient sanity
        if grad_norm == 0.0 or math.isnan(grad_norm):
            raise RuntimeError(
                f"step {step}: grad norm {grad_norm} -> no gradient flow; "
                "check requires_grad, LR, detached graph")
        if grad_norm > self.grad_hi:
            print(f"WARN step {step}: grad norm {grad_norm:.1f} very high; "
                  "expect a spike, add clip_grad_norm_")

        # 2. loss moving off chance by step 50
        if step >= 50:
            moved = self.start_loss - loss
            if loss > self.chance - 0.05 and moved < 0.05:
                raise RuntimeError(
                    f"step {step}: loss {loss:.3f} ~ chance {self.chance:.3f}, "
                    f"moved only {moved:.3f} -> likely DEAD; kill and bisect")

        # 3. throughput acceptable
        if (self.throughput_floor and tokens_per_s
                and tokens_per_s < self.throughput_floor):
            print(f"WARN step {step}: {tokens_per_s:.0f} tok/s < floor "
                  f"{self.throughput_floor} -> dataloader/GPU stall, profile it")
```

The point of raising rather than warning on the dead-run and zero-grad cases is that those are *unrecoverable without a code change* — there is no reason to keep paying. The throughput case is a warning because the run is correct, just slow.

## 4. Slow-starter versus dead: the distinction that decides everything

The single hardest live judgment is telling a **slow-starter** — a run that is genuinely learning but has not visibly moved yet — from a **dead run** that will sit at chance forever. Get this wrong in one direction and you kill a run that would have worked; get it wrong in the other and you babysit a corpse for three days. The good news is that there is a reliable instrument-based distinction, and it does not require waiting.

A slow-starter is flat *for a reason that resolves itself*. The two common reasons are **warmup** and a genuine **plateau before a phase transition**. During LR warmup — the first few hundred steps where the learning rate ramps from near-zero to its target — the loss can be nearly flat simply because the effective step size is tiny; this is by design and the loss will move once the LR reaches its plateau. Some runs also sit at a near-constant loss for a stretch and then drop sharply as the model finds a useful direction; large-model and some RL runs show this. In both cases the run is *alive*, and you can tell because the instruments are healthy underneath the flat loss.

A dead run is flat because *nothing is propagating*. The signature is unambiguous if you read the right channels:

- **Loss is at the chance reference** ($\ln C$ or the marginal entropy), not just flat at some arbitrary value.
- **Gradient norm is at or near zero**, or it is non-zero but the *parameter update norm* (the size of the actual weight change per step) is negligible — meaning the LR is effectively zero or the optimizer state is broken.
- **The parameters are not moving.** A direct check: snapshot a parameter, take a step, and measure $\lVert \theta_{t+1} - \theta_t \rVert$. If it is essentially zero, the model is not updating, period.

The discriminator is the gradient norm and the update norm, not the loss. A slow-starter in warmup has a *healthy* grad norm — gradients are flowing, the loss just is not moving much yet because the LR is small. A dead run has a grad norm of zero (no flow) or a non-zero grad norm paired with a negligible update (LR or optimizer broken). Read those two channels and the ambiguity collapses.

#### Worked example: warmup plateau versus a dead BERT finetune

You launch two BERT finetunes. Both show a flat loss at 0.69 (it is binary, so $\ln 2 = 0.693$) for the first 300 steps. Identical on the loss panel. Are they both dead?

Run X has a gradient norm averaging 3.2 over those 300 steps, an LR ramping from 0 to 2e-5 over a 500-step warmup, and a parameter update norm of about 6e-4 per step. It is a slow-starter: gradients flow, the LR is small because warmup is not done, and at step 350 the loss starts dropping and hits 0.41 by step 1,000. Keep it.

Run Y has a gradient norm of 0.0 — exactly zero — for all 300 steps, and a parameter update norm of 0.0. Investigation finds that the classification head was created after the optimizer was constructed, so its parameters are not in any optimizer group; the encoder was frozen with `requires_grad=False` for a transfer-learning experiment and never thawed. Nothing is being optimized. The loss is at chance because the model is exactly its initialization, forever. This is a [model that isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think). Kill it at step 20 — the grad norm told you everything by step 5.

Two runs, same loss panel, opposite verdicts, disambiguated entirely by the gradient and update norms. This is why the dashboard that matters is not the loss alone.

Here is the parameter-and-gradient probe that makes the call:

```python
import torch

@torch.no_grad()
def liveness_probe(model, optimizer):
    """Is the run actually updating? Returns grad norm and update norm."""
    grad_sq = 0.0
    n_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq += p.grad.float().pow(2).sum().item()
            n_with_grad += 1
    grad_norm = grad_sq ** 0.5

    # snapshot, step, measure how far the weights actually moved
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    upd_sq = sum((p.detach() - b).float().pow(2).sum().item()
                 for p, b in zip(model.parameters(), before))
    update_norm = upd_sq ** 0.5

    return {
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "params_with_grad": n_with_grad,
        "params_total": sum(1 for _ in model.parameters()),
    }

# Dead-run signature: grad_norm ~ 0  OR  update_norm ~ 0  OR
# params_with_grad << params_total (most params have no gradient).
```

If `params_with_grad` is much smaller than `params_total`, you have found the dead run's cause before you even look at the loss: most of your model is not in the graph.

## 5. The dashboards that matter, in priority order

A good dashboard is not a wall of every metric you can compute; it is a small, ordered set of panels each of which rules out a class of failure. If you only have screen space for five panels, these are the five, in priority order.

![A vertical stack of five dashboard panels in priority order, from train and validation loss at the top through gradient norm, learning rate, throughput, to sample generations at the base](/imgs/blogs/monitoring-a-run-and-when-to-kill-it-3.png)

**1. Loss — train and validation, on comparable axes.** The headline instrument, but with two non-negotiable refinements. First, plot *both* train and validation, because the gap between them is itself a diagnostic (overfitting, underfitting, leakage). Second — and this is the rule people break constantly — **monitor and decide on the validation metric, not the training loss.** Training loss almost always keeps falling; it tells you the optimizer is working, not that the model is generalizing. The decision-relevant signal is the held-out metric. We return to this in the early-stopping section because monitoring the wrong metric is the most common early-stopping bug.

**2. Gradient norm.** The disambiguator. As established above, the grad norm separates a slow-starter (healthy norm, flat loss) from a dead run (zero norm), and separates a transient loss spike (grad-norm spike that recovers) from terminal divergence (grad-norm spike that runs away to NaN). It is the single most informative companion to the loss, and a dashboard without it is half-blind. Full treatment in [gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing).

**3. Learning rate.** Cheap to log, essential for interpretation. A flat loss during a warmup ramp is benign; the LR panel tells you whether you are in warmup. A loss that turns up exactly when the LR schedule does something is a schedule bug, not a model bug. The LR panel is how you avoid blaming the model for the scheduler's behavior.

**4. Throughput and utilization.** Tokens-per-second or samples-per-second, plus GPU utilization or MFU if you can get it. This panel does not tell you if the run is *learning*, but it tells you if the run is *affordable* — a correct run at 30% utilization will take three times as long and cost three times as much as it should. It also catches the dataloader-stall sawtooth, where throughput oscillates because the GPU is starving between batches.

**5. Sample generations or predictions.** A handful — five or ten — of actual model outputs on fixed held-out inputs, logged every few hundred steps. For a language model, the generations; for a classifier, the predicted-vs-true on a fixed set of hard examples; for detection, the boxes drawn on a few images. This is the panel that catches the failure no scalar metric will: the loss is fine but the outputs are garbage, repetitive, or all-one-class. It is qualitative and slower to read, which is why it is last, but it is the ground truth the scalars are proxies for.

Here is a single logging block, using Weights & Biases, that populates all five panels. The TensorBoard variant is a near-mechanical swap of `wandb.log` for `writer.add_scalar`.

```python
import torch
import wandb

def log_dashboard(step, model, optimizer, train_loss, val_loss,
                  tokens_per_s, sample_fn=None, sample_every=500):
    # 1. loss (train + val)
    row = {"train/loss": train_loss}
    if val_loss is not None:
        row["val/loss"] = val_loss

    # 2. gradient norm (global L2 over all params with a grad)
    grad_sq = sum(p.grad.float().pow(2).sum().item()
                  for p in model.parameters() if p.grad is not None)
    row["train/grad_norm"] = grad_sq ** 0.5

    # 3. learning rate (first param group)
    row["train/lr"] = optimizer.param_groups[0]["lr"]

    # 4. throughput
    row["sys/tokens_per_s"] = tokens_per_s
    if torch.cuda.is_available():
        row["sys/mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # 5. sample generations / predictions, less often (they are expensive)
    if sample_fn is not None and step % sample_every == 0:
        row["samples/text"] = wandb.Html("<br>".join(sample_fn(model)))

    wandb.log(row, step=step)
```

A healthy run on this dashboard looks like: loss below chance and falling, train and val tracking with a modest gap, grad norm steady in a sane band, LR following its schedule, throughput flat and high, samples improving from gibberish to plausible. A dying run looks like: loss spiking with a simultaneous grad-norm spike, or a train-val gap widening fast. A dead run looks like: loss pinned at chance, grad norm at zero, samples that never change because the weights never change. You learn to read the *pattern across panels*, not any single line.

## 6. Early stopping, done right: the science of patience

Early stopping is the most-used and most-misimplemented monitoring tool. The naive version — "stop when the validation loss goes up" — is wrong in at least three independent ways, each of which costs you either a good model or a lot of compute. To fix it you have to understand what early stopping actually *is*, which is not a budget cap but a form of regularization.

**Early stopping as regularization.** Here is the science. Gradient descent, started from a small initialization, explores the space of models in a roughly increasing order of complexity: early in training the weights are small and the function is simple, and as training proceeds the weights grow and the function fits finer and finer structure in the training data, including its noise. The validation loss therefore typically falls (the model is learning real structure), reaches a minimum (it has captured the signal), and then rises (it is now fitting noise — overfitting). Stopping at the validation minimum returns the model at the point where it has captured signal but not yet noise. For a linear model trained with gradient descent, this is provably close to $L_2$ regularization: the number of training steps plays the role of the inverse regularization strength, and stopping early is equivalent to choosing a particular ridge penalty. So early stopping is not "give up when it gets worse" — it is "select the model complexity that generalizes best, using the validation curve as the selector." That reframing is why the *details* matter: you are choosing a model, so you had better choose the right one and keep it.

**The three ways the naive version fails:**

First, **monitoring the wrong metric.** If you early-stop on the *training* loss, you will essentially never stop, because training loss keeps falling. You must monitor a held-out metric. And it should be the metric you actually care about — for an imbalanced problem, validation loss and validation accuracy and validation PR-AUC can move in different directions, and stopping on the wrong one ships the wrong model. Monitor the metric whose improvement means the model is getting better at the real task.

Second, **no patience, so noise kills the run.** The validation metric is *noisy* — it is computed on a finite held-out set, so it has sampling variance. A patience of zero means the first time the metric ticks up by any amount, even pure noise, you stop. On a noisy validation curve that often happens long before the true minimum. You need *patience*: keep going for $k$ more evaluations after the best, and only stop if none of them beats it. Patience absorbs the noise.

Third, **not restoring the best weights.** When you do stop after patience $k$, the *current* weights are $k$ evaluations past the best — they are slightly worse, having started to overfit. If you ship the last weights, you ship a worse model than you found. You must *restore the best checkpoint* — the weights at the validation minimum, not the weights where you happened to stop. Skipping this throws away the entire point of early stopping.

![A two-column comparison of early stopping done wrong, monitoring train loss with no patience and no restore, versus done right, monitoring validation loss with patience and restoring the best checkpoint](/imgs/blogs/monitoring-a-run-and-when-to-kill-it-4.png)

**How much patience? The noise sets the floor.** Patience is not a magic number; it is set by the *variance of your validation metric*. If your validation set is small and the metric jitters by $\pm 0.01$ between evaluations purely from noise, then a `min_delta` (the minimum improvement that counts as real) smaller than that noise is meaningless, and a patience too small will trip on noise. The principle: **`min_delta` should be larger than the validation-metric noise, and patience should be large enough that a run of $k$ noisy non-improvements is unlikely if the true trend is still down.** Concretely, if your validation metric has standard deviation $\sigma$ across re-evaluations, set `min_delta` to roughly $\sigma$ to $2\sigma$, and set patience so that the probability of $k$ consecutive noise-driven non-improvements while the true trend is still improving is low. A larger validation set (smaller $\sigma$) lets you use smaller patience and stop sooner; a tiny noisy validation set forces large patience. This is the patience-versus-noise tradeoff, and it is why "just use patience 10" is bad advice — patience should track your validation noise.

**The too-impatient versus too-patient tradeoff.** Too little patience kills slow-starters and trips on noise — you stop a run that would have improved, wasting the work done so far and tempting you into a thrash loop of relaunches. Too much patience wastes compute past the point of no return — you keep evaluating long after the model has clearly plateaued or started overfitting, burning GPU-hours for a model you will not use. The sweet spot is set by the validation noise on one side (patience must exceed the noise) and the cost of extra epochs on the other (patience should not vastly exceed what the noise requires). For a cheap-to-evaluate, low-noise validation set, patience of a few epochs is plenty; for an expensive, noisy one, you may need more, but you should also consider evaluating more often on a fixed subset to reduce the noise rather than padding patience.

Here is a correct `EarlyStopping` implementation — monitors a configurable metric, respects `min_delta`, has patience, handles minimize-vs-maximize, and (the part everyone forgets) restores the best weights.

```python
import copy
import math

class EarlyStopping:
    """Patience-based early stopping that restores the best weights.

    monitor:  name of the metric you pass to step()
    mode:     'min' for loss, 'max' for accuracy/AUC
    patience: evaluations to wait after the best before stopping
    min_delta: smallest change that counts as an improvement
               (set this >= your validation-metric noise)
    """

    def __init__(self, mode="min", patience=5, min_delta=1e-3,
                 restore_best=True):
        assert mode in ("min", "max")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best = math.inf if mode == "min" else -math.inf
        self.best_state = None
        self.best_step = None
        self.num_bad = 0
        self.should_stop = False

    def _is_better(self, value):
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value, model, step):
        if self._is_better(value):
            self.best = value
            self.best_step = step
            self.num_bad = 0
            if self.restore_best:
                # store on CPU so we do not pin extra GPU memory
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"restored best weights from step {self.best_step} "
                  f"({self.mode} {self.best:.4f})")
```

The `best_state` is kept on CPU so it does not consume GPU memory, and it is restored into the model at the end of training. That `restore` call is the difference between shipping the validation-minimum model and shipping a model that has already started to overfit.

#### Worked example: the early-stopping fix that recovered 7 AUC points

A tabular fraud model is trained for 50 epochs with naive early stopping: monitor validation loss, patience 0, ship the last weights. The validation AUC peaks at 0.78 around epoch 18, then the validation loss starts to rise slightly, the run trips on the first uptick at epoch 19, and — because patience is 0 and there is no restore — it ships the epoch-19 weights at AUC 0.77. Close, but already past the peak. Worse, a re-run with a different seed trips at epoch 11 on a noise blip (the validation loss wobbled up by 0.002, smaller than the metric's own $\pm 0.004$ noise), shipping AUC 0.71. Same model, same data, 7 AUC points lost to noise plus no-restore.

The fix is three lines of config: monitor validation AUC in `max` mode (the metric you actually ship on), set `min_delta` to 0.005 (just above the measured $\pm 0.004$ noise), set patience to 5 epochs, and turn on `restore_best`. Now the run rides past the epoch-11 noise blip without stopping, continues to the real AUC peak of 0.78 at epoch 18, waits 5 more epochs to confirm it is the peak, stops at epoch 23, and *restores the epoch-18 weights*. Shipped AUC: 0.78, stable across seeds. The fix recovered 7 AUC points on the bad-seed run and removed the seed-dependence entirely, at the cost of 5 extra epochs of evaluation — a trade you take every time. Watch out for the related trap of [overfitting to the validation set](/blog/machine-learning/debugging-training/overfitting-to-the-validation-set) when you tune `min_delta` and patience against the same val set repeatedly.

## 7. The math of patience: deriving `min_delta` and the patience floor

"Set patience to ride out the noise" is the right instinct, but you can make it precise, and doing so is what separates a principled early-stopping config from a superstition. The two knobs — `min_delta` and `patience` — are both set by one quantity: the *standard error of your validation metric*. Derive that and both knobs follow.

Start with where the validation noise comes from. Your validation metric is computed on a finite held-out set of $n$ examples. Treat the per-example contribution to the metric (the per-example loss, or the per-example correctness for accuracy) as a random variable with variance $\sigma_{\text{ex}}^2$. The metric is the mean over $n$ examples, so by the standard result for the variance of a sample mean, the metric's standard error is

$$\text{SE} = \frac{\sigma_{\text{ex}}}{\sqrt{n}}.$$

This is the dominant source of the wobble you see between consecutive evaluations of a model that is not actually changing much. The $\sqrt{n}$ in the denominator is the lever: quadrupling the validation set halves the noise. That single fact drives the whole patience tradeoff — a small validation set has large $\text{SE}$ and forces large patience; a large one has small $\text{SE}$ and lets you stop sooner.

For accuracy specifically, the per-example correctness is a Bernoulli variable with variance $p(1-p)$ where $p$ is the accuracy, so the standard error of validation accuracy is

$$\text{SE}_{\text{acc}} = \sqrt{\frac{p(1-p)}{n}}.$$

Plug in numbers: a validation set of $n = 1{,}000$ at $p = 0.9$ accuracy has $\text{SE}_{\text{acc}} = \sqrt{0.9 \cdot 0.1 / 1000} \approx 0.0095$, about one accuracy point. So a model whose true accuracy is unchanged will show validation accuracy bouncing around by roughly $\pm 1$ point between checks purely from sampling. A validation set of $n = 100$ at the same accuracy has $\text{SE}_{\text{acc}} \approx 0.030$ — three points of pure noise. This is why a tiny validation set is so dangerous for early stopping: a three-point swing means nothing, but a patience-0 stopper reads it as a regression and pulls the plug.

**Setting `min_delta`.** An improvement smaller than the noise is not detectable. So `min_delta` must be at least the standard error, and to be safe against a single noisy evaluation you want it around $1$ to $2 \times \text{SE}$. For the $n = 1{,}000$ accuracy example, $\text{SE} \approx 0.0095$, so `min_delta` of about $0.01$ to $0.02$ is principled; a `min_delta` of $0.001$ is below the noise floor and will treat noise as signal. The rule: **`min_delta` $\gtrsim \text{SE}$ of the validation metric.**

**Setting the patience floor.** Patience must be large enough that a *run of non-improvements caused purely by noise, while the true trend is still down*, is unlikely to trip the stopper. Model the situation near the true minimum, where the true metric is roughly flat: each evaluation independently fails to beat the running best with some probability $q$ (for a truly flat true-metric, $q$ is high — most draws will not exceed a best that itself was a lucky high draw). The probability of $k$ consecutive non-improvements by chance is roughly $q^k$, which falls off geometrically in $k$. So even modest patience — say $k = 3$ to $5$ — drives the false-stop probability low when the metric is genuinely flat, and the whole point is that when the true trend is still *improving*, the probability of $k$ consecutive failures to beat the best by `min_delta` is far lower still, because the trend keeps pushing new bests. The rule: **patience large enough that $q^k$ is small, which for typical noisy-but-trending curves is a handful of evaluations, not one and not fifty.**

The cheaper alternative to large patience is to *reduce the noise*: evaluate on a larger fixed validation subset (bigger $n$, smaller $\text{SE}$), or average the metric over the last few evaluations before comparing. Cutting $\text{SE}$ in half lets you halve `min_delta` and shrink patience, stopping sooner with the same false-stop risk. This is why "evaluate more often" can backfire if it means evaluating on a *smaller* subset each time — you trade temporal resolution for more noise per check, and the noisier metric forces more patience, net-net a wash or worse.

#### Worked example: sizing patience for a small versus a large val set

Two image classifiers, both at 90% validation accuracy, both early-stopped on validation accuracy in `max` mode. Model P has a 100-example validation set; Model Q has a 5,000-example one.

For Model P: $\text{SE}_{\text{acc}} = \sqrt{0.9 \cdot 0.1 / 100} \approx 0.030$, three points of noise per check. A principled config sets `min_delta` $\approx 0.03$ and patience $\approx 8$ — large patience because each check is so noisy that short patience would trip constantly. Even so, the early-stop decision is mushy: the metric swings three points around the truth, so the "best" checkpoint is partly luck. The honest fix is not more patience but a bigger validation set.

For Model Q: $\text{SE}_{\text{acc}} = \sqrt{0.9 \cdot 0.1 / 5000} \approx 0.0042$, under half a point of noise. Now `min_delta` $\approx 0.005$ and patience $\approx 3$ suffice — the metric is stable enough that a three-check run of non-improvement is strong evidence of a real plateau, and the restored "best" checkpoint is genuinely the best. Same accuracy, same task, but the $50\times$ larger validation set lets Q stop sooner *and* trust the result more. The patience numbers are not arbitrary; they fall directly out of the $\sqrt{n}$ in the standard error. When someone asks "what patience should I use?", the honest answer is "compute your validation metric's standard error first."

## 8. Early stopping in the real toolchain: HF Trainer and Lightning

You rarely write the early-stopping loop from scratch in production; you use the framework's callback. But the frameworks have sharp edges — defaults that monitor the wrong thing, or that do not restore the best weights — so you need to configure them deliberately.

**Hugging Face `Trainer`.** The `EarlyStoppingCallback` works, but you must set up three things correctly in `TrainingArguments`: tell it to evaluate (`eval_strategy="steps"` or `"epoch"`), tell it which metric is "best" (`metric_for_best_model`) and whether bigger is better (`greater_is_better`), and — critically — set `load_best_model_at_end=True` so it restores the best checkpoint. The callback itself takes `early_stopping_patience` and `early_stopping_threshold` (the `min_delta` analog).

```python
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

args = TrainingArguments(
    output_dir="out",
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",        # must match eval to keep the right ckpts
    save_steps=200,
    # decide on the metric you SHIP on, not training loss:
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    load_best_model_at_end=True,  # restores best weights -- do NOT skip
    save_total_limit=3,           # keep the run from filling the disk
    logging_steps=10,
)

def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds, average="macro")}

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=5,       # evaluations, i.e. 5 * eval_steps
        early_stopping_threshold=0.005,  # min_delta on eval_f1
    )],
)
trainer.train()
```

Two traps. First, `save_strategy` must match `eval_strategy` (and `save_steps` must equal `eval_steps`) or `load_best_model_at_end` cannot find the best checkpoint and will raise. Second, `early_stopping_patience` is counted in *evaluations*, not steps — patience 5 with `eval_steps=200` means 1,000 steps of no improvement, not 5. People set patience 5 expecting 5 steps and wonder why the run goes forever.

**PyTorch Lightning.** Lightning's `EarlyStopping` callback is cleaner. You log the monitored metric in your `validation_step`, then configure the callback and a `ModelCheckpoint` that saves the best.

```python
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor="val_loss",   # the key you self.log() in validation_step
    mode="min",
    patience=5,
    min_delta=1e-3,
    verbose=True,
)
# Lightning does NOT auto-restore best on early stop; the checkpoint does:
checkpoint = ModelCheckpoint(
    monitor="val_loss", mode="min", save_top_k=1, filename="best",
)

trainer = L.Trainer(
    max_epochs=50,
    callbacks=[early_stop, checkpoint],
    val_check_interval=0.5,   # evaluate twice per epoch -> less noise per check
)
trainer.fit(model, datamodule=dm)
# load the best weights for inference/export:
best = MyModel.load_from_checkpoint(checkpoint.best_model_path)
```

The Lightning trap is the dual of the HF one: the `EarlyStopping` callback decides *when to stop* but does *not* by itself restore the best weights — you need the `ModelCheckpoint` to have saved them and you must explicitly load `checkpoint.best_model_path` afterward. Stop without that load and you again ship the past-the-peak weights.

A note on monitoring frequency: `val_check_interval` controls how often you evaluate. Evaluating more often gives you a finer early-stopping decision but costs compute and — if your validation set is small — produces a *noisier* per-check metric. A good middle path for a noisy small val set is to evaluate often but on a *fixed* validation subset so the noise is consistent, and to set `min_delta` above that subset's noise.

## 9. The kill decision: rules, not vibes

So far we have built the instruments. Now the decision: given a live run, do you keep it, wait on it, or kill it? The answer should be a *rule* you decided before the run started, not a gut call you make while staring at a wiggling line at midnight. Here is the rule set, organized by symptom.

The reason to write the rule down in advance is that the decision is made under two distortions that pull in opposite directions. The first is *sunk cost*: a run that has already burned twelve hours feels too expensive to kill, so you let a clearly-dead run keep billing because killing it feels like admitting the twelve hours were wasted — but they are already wasted, and every further hour is new waste, not recovered waste. The second is *loss aversion to action*: killing a run that might have recovered feels worse than letting a run die slowly, so on a recovered spike you panic-kill a perfectly healthy run rather than tolerate the discomfort of watching it wobble. A rule decided in the calm of the afternoon — *grad-zero-at-chance means kill, recovered-spike means keep, val-plateau-past-patience means stop* — neutralizes both distortions, because it removes the in-the-moment judgment that sunk cost and loss aversion corrupt. The rule is not bureaucracy; it is a precommitment device against your own predictable midnight biases.

![A decision tree rooted at whether the loss is falling below chance, branching through gradient-norm and validation-plateau checks into four outcomes: kill, wait, stop and restore, or keep](/imgs/blogs/monitoring-a-run-and-when-to-kill-it-6.png)

**Stuck at chance with flat grad norm = dead. Kill.** If the loss is at the chance reference ($\ln C$ or marginal entropy) after a reasonable number of steps *and* the gradient norm is at or near zero, the run is dead — it is not updating and will not start. There is no recovery without a code change. Kill it immediately and bisect to the cause (frozen params, LR zero, broken graph, data all-one-label). Do not wait "to be sure"; the grad norm already told you.

**Diverged to NaN or Inf = kill, or auto-restart from checkpoint.** A loss that has become NaN or Inf will never recover on its own — every subsequent step propagates the NaN through the weights, corrupting the model permanently. The moment you see NaN, the run is dead from that step forward. Either kill it and fix the numerics ([hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs)), or, if you have a guard, auto-restart from the last clean checkpoint with a guardrail (lower LR, gradient clipping, skip the offending batch). What you must *not* do is let it keep running — every step past the first NaN is pure waste.

**Validation plateaued past patience = stop (not kill).** If the run is healthy (learning, sane grad norm) but the *validation* metric has not improved for `patience` evaluations beyond `min_delta`, the model has converged or begun to overfit. This is a graceful *stop* — restore the best weights and you have your model. This is early stopping, and it is a success, not a failure. The distinction from a kill matters: a kill means "throw this away and fix something"; a stop means "you are done, here is the model."

**Loss spike that recovered = keep.** A transient loss spike — even a dramatic one, 1.4 jumping to 9.2 — that recovers within a few dozen steps is a normal event in many runs, especially large ones, and is usually a single bad batch or a momentary LR-induced excursion. The signature of *transient* is that the grad-norm spike accompanies it and both come back down. The signature of *terminal* is the grad norm running away to ever-larger values and the loss heading to NaN. If it recovered, keep the run; consider adding gradient clipping to bound future spikes. Killing a run over a recovered spike is the too-impatient failure. See [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence) for telling transient from terminal.

**Throughput too low = fix, do not kill.** A correct but slow run is not a kill — the model is learning fine, it is just expensive. Fix the dataloader or the GPU stall and let it continue. The mistake here is the inverse of impatience: killing a perfectly good run because it is slow, when the fix is a workers count.

Here is the full kill-or-keep map as a diagnostic table, which is the artifact I keep pinned next to the dashboard:

| Live symptom | Leading suspect | Confirming signal (one glance) | Verdict |
|---|---|---|---|
| Loss flat at chance, many steps | Dead run, no grad flow | Grad norm ~0; loss still at $\ln C$ | **Kill** — fix (frozen/LR/graph), relaunch |
| Loss became NaN / Inf | Numerics (fp16, log0, /0) | NaN guard fires at step N | **Kill or auto-resume** from last clean ckpt |
| Loss spike, then recovered | Bad batch or LR excursion | Grad-norm spike that came back down | **Keep** — add clip 1.0, ride it out |
| Validation flat past patience | Converged or overfitting | Val metric flat $k$ evals, train still falling | **Stop** — restore best checkpoint |
| Throughput 30%, util low | Dataloader / GPU stall | Sawtooth throughput, profiler stalls | **Fix** — more workers; not a kill |
| Train great, val garbage | Overfit / leak / eval bug | Train-val gap blows up; bad samples | **Stop + investigate** eval and split |

![A decision matrix mapping five live symptoms to a suspect, a one-glance confirming signal, and a keep, wait, or kill verdict](/imgs/blogs/monitoring-a-run-and-when-to-kill-it-5.png)

The discipline is to *decide the rule before launch* and then *follow it*. The reason to pre-commit is that midnight-you, watching money burn, is biased toward action — either premature killing (impatience) or denial (letting a dead run run because killing feels like admitting failure). A rule decided by calm-you in the afternoon removes the emotion: stuck-at-chance-with-zero-grad means kill, recovered-spike means keep, no argument.

## 10. Automated guards: making the run watch itself

The best monitoring is the monitoring you do not have to do. Once you have the kill rules, encode them as automated guards so the run enforces them itself — kills on NaN, kills on stall, stops on no-improvement, alerts you, and where appropriate auto-resumes. This is what turns "watch the first hundred steps and then go to bed" from a gamble into a safe default: the run will catch its own death and tell you.

![A dataflow graph showing live loss and gradient feeding three guards, kill-on-NaN, stall detection, and no-improvement, which fan out to alerting, auto-resume from checkpoint, and stop-with-restore](/imgs/blogs/monitoring-a-run-and-when-to-kill-it-8.png)

There are three guards worth automating, each enforcing one of the kill rules:

**Kill-on-NaN.** Check the loss for NaN/Inf every step; on detection, either stop immediately or trigger an auto-resume from the last clean checkpoint with a guardrail. This is the cheapest, highest-value guard — a single `math.isnan` per step that saves you from a run that has been propagating NaN for hours.

**Stall guard (kill-on-no-progress).** Track the loss over a sliding window; if it has been flat (within a small epsilon) for $N$ steps *and* sitting near the chance reference, the run is stalled — fire the kill. This is the dead-run rule automated. The window and epsilon are set so a genuine slow-starter (which has a healthy grad norm) is not killed; pairing the loss-flatness check with a grad-norm check makes it precise.

**No-improvement guard.** This is just early stopping — the `EarlyStopping` callback from section 6, which stops gracefully and restores the best weights when the validation metric plateaus past patience.

And two pieces of plumbing that make the guards actionable:

**Alerting.** When a guard fires, *tell someone* — a Slack message, a page to on-call, an email. A guard that kills a run at 3am and leaves no trace means you discover the dead run at 9am and have lost six hours of queue time. The alert turns a silent kill into a prompt to relaunch.

**Auto-resume from checkpoint.** For transient, recoverable failures (a NaN from a single bad batch, a node preemption), the most efficient response is not to kill-and-wait-for-a-human but to auto-resume from the last checkpoint, optionally with a guardrail applied. This requires that checkpointing and resume are correct — a botched resume that jumps the loss is its own bug, covered in [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume) — but when it works, it converts a class of failures from "human-in-the-loop kill" to "self-healing."

Here is a combined kill-on-NaN-and-stall callback, framework-agnostic, that you call once per step. It enforces the NaN-kill and the stall-kill rules and emits an alert hook on fire.

```python
import math
from collections import deque

class RunGuard:
    """Auto-enforce the kill rules: NaN and stall. Call every step."""

    def __init__(self, chance_loss, stall_window=500, stall_eps=0.02,
                 grad_dead_thresh=1e-5, alert_fn=None):
        self.chance = chance_loss
        self.window = deque(maxlen=stall_window)
        self.stall_window = stall_window
        self.stall_eps = stall_eps
        self.grad_dead = grad_dead_thresh
        self.alert = alert_fn or (lambda msg: print("ALERT:", msg))

    def __call__(self, step, loss, grad_norm):
        # 1. NaN / Inf guard -- never recovers, kill now
        if not math.isfinite(loss):
            self.alert(f"step {step}: loss={loss} (NaN/Inf) -> KILL/resume")
            return "kill_nan"

        self.window.append(loss)

        # 2. stall guard: flat AND near chance AND no gradient = dead
        if len(self.window) == self.stall_window:
            spread = max(self.window) - min(self.window)
            near_chance = (sum(self.window) / len(self.window)
                           > self.chance - 0.05)
            no_grad = grad_norm < self.grad_dead
            if spread < self.stall_eps and near_chance and no_grad:
                self.alert(
                    f"step {step}: loss flat {spread:.3f} near chance "
                    f"{self.chance:.2f}, grad {grad_norm:.1e} -> DEAD, KILL")
                return "kill_stall"
        return None  # healthy, keep going
```

And the Slack alert hook, so a fired guard reaches a human:

```python
import os, json, urllib.request

def slack_alert(msg):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        print("ALERT (no webhook):", msg)
        return
    body = json.dumps({"text": f":rotating_light: training run: {msg}"})
    req = urllib.request.Request(
        url, data=body.encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5)
```

Wire `RunGuard(chance_loss=math.log(num_classes), alert_fn=slack_alert)` into your loop and the run now kills itself on NaN or a dead stall, pages you, and you sleep without gambling. The stall guard's three-way condition — flat *and* near chance *and* grad near zero — is what keeps it from murdering a healthy slow-starter, because a slow-starter fails the grad-near-zero test.

A word on the auto-resume path, because it is the one guard people get wrong in a way that quietly corrupts results. Auto-resume is only safe if your checkpoint-and-resume is *truly continuous* — the optimizer state (momentum buffers, Adam's first and second moments), the learning-rate scheduler position, the RNG state, the gradient-scaler state under mixed precision, and any EMA shadow weights all have to be saved and restored, not just the model weights. If you restore only the weights and reset the optimizer, the resumed run takes a few hundred steps of garbage updates while Adam's moment estimates re-warm, and you see the telltale *loss jump on resume*: the curve discontinuously bumps up at the resume step, then settles. That jump is not the bad batch coming back; it is your resume being lossy, and it can erase exactly the progress the checkpoint was meant to preserve. Verify a resume is continuous before you trust auto-resume to run unattended — the full procedure is in [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume). Until you have verified it, prefer kill-and-alert over auto-resume, because a clean kill that pages you loses queue time, but a botched auto-resume loses correctness.

There is a cost framing worth stating plainly, because it is the whole justification for building this machinery. The expensive resource is not your time writing a callback; it is GPU-hours, and the distribution of waste is heavily skewed toward *late detection*. A doomed run caught at step 100 wastes 100 steps; the same run caught at the end wastes the entire budget, and across a team running, say, thirty experiments a month with a five-percent dead-run rate, the difference between catching those one-or-two doomed runs early versus late is roughly one-or-two full runs of GPU time saved every month — GPU-weeks over a year. The callbacks in this post are an afternoon of work that pays for itself the first time the stall guard kills a typo'd run at 3am instead of letting it bill until you wake up. That asymmetry — cheap to build, occasionally enormous to save — is why a babysitting checklist for the first hundred steps and a self-killing guard for the rest of the run are not optional polish; they are the highest-return-per-line code in your training stack.

## 11. Comparing runs without eyeballing

A subtle monitoring failure is comparing runs badly. You run experiment A and experiment B, glance at their loss curves, decide B "looks better," and commit to it. But the curves differ partly because of the *change* you made and partly because of the *seed* — random initialization, data shuffling order, augmentation randomness, dropout masks. If the seed-to-seed variance is comparable to the effect of your change, your eyeball comparison is reading noise.

The discipline has three parts. **First, control the seed.** Run the comparison with the same seed for both A and B so the only difference is your change; better, run each with three or more seeds so you can see the variance. A change that improves the mean by less than the seed-to-seed standard deviation is not a real improvement. This is the same determinism discipline covered in [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) — you need control over randomness to compare anything.

**Second, compare on the dashboard, not by eye.** Overlay the runs on the same axes, on the validation metric you ship on (not training loss), with comparable x-axes (steps or wall-clock, decided in advance). Weights & Biases and TensorBoard both make this trivial; the point is to put the lines on the *same* plot at the *same* scale rather than flipping between two screenshots.

**Third, compare at a fixed budget.** A run that reaches a lower loss after 100k steps is not "better" than one that reaches a slightly higher loss after 30k if you can only afford 30k. Decide the budget — a step count or a wall-clock or a dollar amount — and compare the runs *at that budget*. Comparing a finished long run to an unfinished short one is a category error.

The statistical core of all three is the same standard-error argument from the patience section, applied across seeds instead of across examples. If a configuration's final metric has a seed-to-seed standard deviation $\sigma_{\text{seed}}$, then the standard error of the *mean over $s$ seeds* is $\sigma_{\text{seed}} / \sqrt{s}$, and a difference between two configurations is only credible if it exceeds roughly two of those standard errors — the same two-sigma intuition you use for any noisy measurement. With one seed each ($s = 1$) you have no estimate of $\sigma_{\text{seed}}$ at all, so you cannot say anything; with three seeds you have a rough estimate and can rule out differences smaller than the spread; with five or more you can start to trust small effects. The practical rule is blunt: **never promote a configuration on a single-seed comparison, and treat any improvement smaller than the seed spread as zero.** Most "this trick helped" claims that fail to replicate are single-seed noise promoted to a finding.

#### Worked example: B "looked better" but was within seed noise

You change the learning-rate schedule (A: cosine, B: linear) and B's validation loss curve sits visibly below A's — 1.31 versus 1.35 at step 20k. You are about to standardize on B. Before you do, you run both with three seeds. A's three runs land at 1.35, 1.33, 1.37 (mean 1.35, std 0.02); B's at 1.31, 1.36, 1.34 (mean 1.337, std 0.025). The means differ by 0.013, well inside the $\pm 0.02$–0.025 seed noise of either. The "B is better" signal was a single lucky seed. The honest conclusion is "no detectable difference," and you keep the simpler choice. Without seed control you would have shipped a non-improvement and possibly chased phantom follow-on effects. Eyeballing two single-seed curves is reading noise as signal; three seeds and a same-axes overlay told the truth in one extra afternoon of compute.

## 12. Case studies and real signatures

A few named patterns make the abstract rules concrete. These are the signatures that, once you have seen them, you recognize instantly.

**The dead BERT finetune (the dead-run signature).** A classic: a Hugging Face finetune where the model sits at exactly $\ln 2 = 0.693$ for binary classification, forever. The common causes are a frozen encoder that was never thawed, a classification head created after the optimizer (so its params are in no optimizer group), or a learning rate left at the default for the wrong scale. The signature is unmistakable: loss pinned at chance, gradient norm zero, `params_with_grad` far below `params_total`. The preflight catches it in twenty steps; without the preflight it can run for the full budget producing a model that predicts one class. Caught at step 20, you fix the freeze and relaunch; the relaunched run drops below chance by step 50.

**Early stopping as regularization (the science, confirmed empirically).** The view that early stopping approximates $L_2$ regularization is not just folklore — for linear regression with gradient descent it is provable, and the connection between training time and effective regularization strength is well established in the optimization literature (the "early stopping as implicit regularization" line of work). The practical upshot is the one we built the section around: the validation curve has a minimum, and the model at that minimum generalizes best, which is *why* restoring the best weights matters. Empirically, on overparameterized models, you routinely see validation loss bottom out and then rise while training loss keeps falling — the textbook overfitting U-curve — and stopping-and-restoring at the bottom recovers points of accuracy that the last-weights model throws away. Our worked example of 7 recovered AUC points is a typical magnitude, not an extreme one.

**The loss spike that recovered (large-model training).** In large-model training it is well documented (in published training reports for large language models) that loss *spikes* occur during otherwise-healthy runs — the loss jumps sharply and then recovers within tens to hundreds of steps. The naive response, killing the run on the spike, throws away a healthy run. The disciplined response, recognizing the spike as transient (grad-norm spike that recovers) and either riding it out or skipping the offending batch and continuing, saves the run. The signature that distinguishes transient from terminal is whether the grad norm comes back down or runs away — the same disambiguation we use everywhere in this post.

**The preflight that caught a doomed run at step 80.** This is the headline before-after of the post. A team launches an LLM finetune; a chat-template formatting bug means the loss is computed over a mis-tokenized sequence and the model cannot learn the task — it sits at chance. Without a preflight, the run executes its full 50,000-step schedule, burning roughly 18 GPU-hours (about \$45 at a typical \$2.50/GPU-hour rate) to produce a useless checkpoint, discovered three days later. With the first-100-steps preflight plus the stall guard, the overfit-one-batch test fails at the start (the model cannot even memorize one batch through the broken loss), and even if that were skipped, the stall guard fires at step 80 when the loss is flat at chance with a low grad-to-update ratio. Cost of the doomed run: about 0.03 GPU-hour, a few cents. The fix (correct the chat template) takes ten minutes, and the relaunched run learns. Same bug, same fix, two-thousand-fold difference in wasted compute — entirely from *when* the run was killed.

**The thrash loop (too-impatient, the inverse failure).** Less dramatic than a dead run but more insidious because it masquerades as diligence: an engineer launches a finetune, watches it like a hawk, and kills it the first time the validation loss ticks up, relaunching with a tweaked learning rate. The run was a slow-starter every time — the validation wobble was noise on a small held-out set, and each killed run was on track to converge. Over an afternoon the engineer runs six aborted launches, each killed before step 2,000, and never finishes a single run. The total compute spent thrashing exceeds what one patient run to completion would have cost, and there is no model to show for it. The signature is a fleet of runs that all die young with healthy grad norms and a validation curve killed on its first noisy uptick. The fix is not a code change but the discipline of this post: compute the validation standard error, set `min_delta` and patience above it, and *let the run cook* unless an instrument — not a single noisy validation point — says it is dead. This is the failure that "watch it carefully" advice actively causes, and it is why monitoring well means monitoring against pre-set thresholds, not monitoring more anxiously.

| Scenario | Without monitoring | With preflight + guards | Saving |
|---|---|---|---|
| Dead LLM finetune (template bug) | 50,000 steps, ~18 GPU-hr, ~\$45 | killed @ step 80, ~0.03 GPU-hr | ~600× compute |
| Naive early stop, no restore | ship AUC 0.71–0.77 (seed-dependent) | restore best, AUC 0.78 stable | +7 AUC pts |
| Killed a recovered spike | relaunch loop, 3–6× wasted runs | keep + clip, one finished run | 3–6× compute |
| Eyeballed seed-noise as a win | ship a non-improvement | 3-seed overlay, keep simpler | avoid regression |

The pattern across all four is the same: monitoring discipline converts a late, expensive discovery into an early, cheap one, or a noisy guess into a confident decision.

## 13. When this is (and isn't) a monitoring problem

Monitoring is a powerful lens, but not every training failure is best attacked by watching the run, and using the wrong lens wastes time. Here is when the symptom points at monitoring and when it points elsewhere.

**It is a monitoring problem when** the run is *executing* but you cannot tell if it is *succeeding* — you need better instruments or a clearer decision rule. A run you babysat for three days that turned out to be dead, a thrash loop of premature kills, an early stop that shipped the wrong weights, a run whose health you genuinely cannot read from the dashboard — these are monitoring problems, fixed by the chance reference, the grad-norm panel, correct early stopping, and the guards.

**It is a data problem, not monitoring, when** the run is healthy by every instrument but the model is still wrong — the loss drops, generalizes, the grad norm is sane, and yet the predictions are bad on real inputs. That is usually a [data leakage](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), label-noise, or distribution-shift story, and no amount of better monitoring of the *training* run will surface it; you have to look at the data and the eval set. Monitoring tells you the run is *learning what you taught it*; it cannot tell you that what you taught it was wrong.

**It is an optimization or numerics problem, not monitoring, when** the symptom is a spike-then-NaN or a divergence. Monitoring *detects* it (the grad-norm spike, the NaN guard), but the *fix* is in the optimizer or numerics — the learning rate, the clipping, fp16-versus-bf16. The kill rule says "kill on NaN," but the next move is the numerics debug, not a monitoring change.

**It is a systems problem, not monitoring, when** the symptom is low throughput, a memory leak that grows each step, or a multi-GPU desync. The dashboard's throughput panel *flags* these, but the fix is in the systems layer — the dataloader, the memory budget, the DDP setup — covered in their own posts.

**The clean tell:** if the run's instruments (loss, grad norm, val metric) are all *healthy* and you still have a bad model, stop staring at the training dashboard — the bug is in the data or the eval, not in how the run is going. And conversely, if the instruments are *unhealthy* (dead, NaN, spiking), monitoring's job is to detect it fast and trigger the right downstream debug; the kill rule is the handoff, not the fix. Monitoring is the bisection's *symptom-reader*, and it hands off to the six-place [taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) once it has localized the run as dead, dying, or healthy. The [capstone playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) assembles the whole symptom-to-suspect-to-test-to-fix flow that monitoring kicks off.

![A before-and-after comparison showing a doomed run burning fifty thousand steps without a preflight versus the same run killed at step eighty by a preflight and stall guard](/imgs/blogs/monitoring-a-run-and-when-to-kill-it-7.png)

## 14. Key takeaways

The rules below are the ones worth memorizing — the symptom-to-action lines that turn midnight panic into a decision.

- **Compute the chance reference before you launch.** $\ln C$ for balanced classification, the marginal entropy $H(Y)$ for imbalanced, $\text{Var}(y)$ for standardized MSE. A run hugging that line is not learning, no matter how the smoothed curve wiggles.
- **The grad norm, not the loss, distinguishes slow-starter from dead.** A flat loss with a healthy grad norm is warmup or a plateau — keep it. A flat loss with a zero grad norm is dead — kill it. The loss alone is ambiguous; the grad norm is decisive.
- **Spend your monitoring budget in the first 100 steps.** The evidence of doom is there, the cost is not. Run the overfit-one-batch preflight, check loss-is-moving and grad-is-sane, and make a cheap keep-or-kill verdict at step 100 — two-tenths of a percent of the run.
- **Early-stop on the metric you ship on, not training loss, with patience and restore.** Monitor validation (the right metric), set `min_delta` above the validation noise, set patience to ride out that noise, and *restore the best weights*. Skipping restore ships a past-the-peak model.
- **Patience is set by validation noise, not by habit.** Too little patience kills slow-starters and trips on noise; too much wastes compute. A small noisy val set needs more patience (or a fixed eval subset to cut the noise); a large clean one needs little.
- **Kill on stuck-at-chance-with-flat-grad and on NaN; stop on val-plateau; keep on a recovered spike.** Pre-commit these rules in the afternoon so midnight-you follows them instead of either panic-killing a healthy spike or denial-running a corpse.
- **Automate the guards so the run watches itself.** Kill-on-NaN, a stall guard (flat *and* near chance *and* grad near zero), early stopping, an alert hook, and auto-resume from checkpoint turn "launch and pray" into "launch and sleep."
- **Compare runs with seed control, on the same axes, at a fixed budget.** A difference smaller than the seed-to-seed standard deviation is not a difference. Eyeballing two single-seed curves is reading noise as signal.
- **If every instrument is healthy and the model is still wrong, it is not a monitoring bug.** Monitoring confirms the run learned what you taught it; it cannot tell you the data or the eval was wrong. Hand off to the data and evaluation tracks.

## 15. Further reading

- **PyTorch documentation** — `torch.nn.utils.clip_grad_norm_`, AMP (`torch.amp.autocast`, `GradScaler`), and `torch.cuda.max_memory_allocated`, for the instruments and guardrails this post wires together.
- **Hugging Face `transformers` documentation** — `Trainer`, `TrainingArguments` (`load_best_model_at_end`, `metric_for_best_model`, `eval_strategy`), and `EarlyStoppingCallback`, the production early-stopping path.
- **PyTorch Lightning documentation** — the `EarlyStopping` and `ModelCheckpoint` callbacks, and why stopping does not by itself restore best weights.
- **Weights & Biases and TensorBoard documentation** — run dashboards, metric overlays for run comparison, and alerting integrations.
- **Prechelt, "Early Stopping — But When?" (1998)** — the classic treatment of early-stopping criteria, patience, and the tradeoff between stopping too early and too late.
- **"Mixed Precision Training," Micikevicius et al. (2018)** — loss scaling and the fp16 range, background for the NaN/divergence kill rules and the numerics handoff.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the six-place bisection frame monitoring feeds), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone symptom-to-fix flow), [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic) (the curve-shape field guide), [instrumenting a training run: what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) (the signals behind the dashboards), [the overfit a single batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) (the preflight that prevents a dead run), and [overfitting to the validation set](/blog/machine-learning/debugging-training/overfitting-to-the-validation-set) (the trap behind tuning patience and `min_delta`).

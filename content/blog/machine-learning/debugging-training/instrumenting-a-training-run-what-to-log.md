---
title: "Instrumenting a Training Run: What to Log and What Each Signal Rules Out"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Log the six signals that turn a silent stall into a named bug, and learn which class of failure each one rules in or out."
tags:
  [
    "debugging",
    "model-training",
    "instrumentation",
    "gradient-norm",
    "pytorch",
    "finetuning",
    "deep-learning",
    "logging",
    "wandb",
    "tensorboard",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/instrumenting-a-training-run-what-to-log-1.png"
---

You cannot debug what you cannot see. That sentence sounds like a platitude until you have spent four days staring at a loss curve that does exactly one thing: it plateaus. Loss flat at 2.31 nats, step after step, batch after batch. You change the learning rate. Nothing. You change the optimizer. Nothing. You shuffle the data, you swap the initialization, you add warmup, you remove warmup. The curve sits at 2.31 like a stone. By Thursday you are convinced the framework is broken, or your GPU is haunted, or you have personally angered the gradient gods.

The actual bug, in the run I am describing, was that one residual block out of twelve received a gradient of magnitude $10^{-7}$ — effectively zero — because of a normalization placement mistake that left its sub-path detached from the loss in all but name. The block never learned. The other eleven blocks learned around it as best they could and the loss settled at the floor that eleven-twelfths of a model can reach. The loss curve **could not** show me this. A scalar loss is a single number summarizing the behavior of hundreds of millions of parameters; by construction it throws away every detail about *where* the problem is. The fix took ninety seconds once I could see the per-layer gradient norms. The four days were spent flying blind.

This post is about the instruments. Not the philosophy of debugging — that lives in the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), the master decision tree this whole series fills in — but the concrete, runnable answer to the question *what do I actually log, and what does each thing tell me?* For every signal I will give you four things: what it is, how to compute it (real PyTorch, copy-and-run), the healthy range, and — the part that makes a signal worth a log line — which bugs it rules **in** and which it rules **out**. A signal that does not change your beliefs when you read it is decoration. Every signal below changes your beliefs.

![A vertical stack of six training signals from loss down through grad norm, update-to-weight ratio, activation histogram, learning rate, and throughput, each labeled with a healthy value, showing that a legible run requires all six layered together rather than loss alone.](/imgs/blogs/instrumenting-a-training-run-what-to-log-1.png)

By the end you will have a reusable instrumentation module — PyTorch hooks that capture per-layer gradient and activation statistics, a function that computes the global gradient norm and the update-to-weight ratio, and a logging cadence that does not slow your run down. And you will have a diagnostic table that maps each signal to the bugs it catches, so that the next time a run lies to you, you read three numbers and know which of the six places — data, optimization, model code, numerics, systems, evaluation — to go dig in. That is the whole game: stop guessing, start reading.

## 1. The spine: loss, but logged honestly

Loss is the signal you already log. I am putting it first not to belabor it but because most people log it wrong, and a badly-logged loss curve is worse than none — it actively misleads.

The three mistakes are: logging only the per-step loss (which is so noisy on small batches that you cannot see the trend), logging only a smoothed loss (which hides spikes that are the single most informative event in a run), and logging the loss as a Python float pulled off the GPU every single step (which, as we will see in section 9, inserts a synchronization stall into your hot loop). The fix for the first two is to log **three** views of the same number:

```python
import collections

class LossTracker:
    """Per-step, exponentially-smoothed, and running-max loss views."""
    def __init__(self, beta=0.98):
        self.beta = beta
        self.ema = None
        self.running_max = float("-inf")
        self.window = collections.deque(maxlen=50)

    def update(self, loss_value: float) -> dict:
        self.window.append(loss_value)
        self.ema = (loss_value if self.ema is None
                    else self.beta * self.ema + (1 - self.beta) * loss_value)
        self.running_max = max(self.running_max, loss_value)
        # debiased EMA so the early steps are not pulled toward zero
        return {
            "loss/step": loss_value,
            "loss/ema": self.ema,
            "loss/window_max": max(self.window),
            "loss/running_max": self.running_max,
        }
```

The **per-step** loss shows the texture: a sawtooth period that matches your epoch length is a dataloader-ordering bug; a single isolated spike is a bad batch or a numerical event; a step function down is a learning-rate-schedule boundary. The **smoothed (EMA)** loss shows the trend you actually care about for "is it learning." The **running max** and a short **window max** catch the spike that the EMA erases — a loss that EMAs to 2.0 but spiked to 9.4 on step 412 is a run that is one bad batch away from `NaN`, and you want to know that before it happens. Reading the loss curve as a diagnostic — what each *shape* implies — is a whole topic of its own, covered in [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic); here the point is narrower. Loss is the spine. It tells you *that* something is wrong. It almost never tells you *where*. For "where," you need the next five signals.

A word on the EMA, because the bias matters in the first few hundred steps where you most want a clean read. An exponential moving average with decay $\beta$ updates as $\text{ema}_t = \beta\,\text{ema}_{t-1} + (1-\beta)\,x_t$. If you initialize $\text{ema}_0 = 0$, the average is biased toward zero for roughly the first $1/(1-\beta)$ steps — at $\beta = 0.98$ that is about 50 steps where the smoothed loss reads artificially low, which can make a run look like it started learning when it has not. The two fixes are: initialize the EMA to the first observed loss (what the `LossTracker` above does, by setting `self.ema = loss_value` when it is `None`), or apply the standard bias correction $\hat{\text{ema}}_t = \text{ema}_t / (1 - \beta^t)$. Either keeps the early curve honest. The choice of $\beta$ is a real trade-off: $\beta = 0.9$ (window of $\approx 10$ steps) is responsive but still noisy on small batches; $\beta = 0.99$ (window of $\approx 100$) is smooth but lags real changes by dozens of steps, which can hide a divergence that has already started. I log two EMAs — a fast one and a slow one — when a run is misbehaving, because the *gap* between them is itself a signal: when the fast EMA pulls above the slow one, the loss is rising before the smooth curve admits it.

The texture of the per-step loss deserves more than a sentence, because each shape is a different bug. A **clean exponential-looking descent with thin noise** is the healthy case. A **sawtooth whose period equals your steps-per-epoch** means the data is not being reshuffled between epochs — the model memorizes the fixed order and the loss drops within an epoch then jumps at the epoch boundary when it re-sees the same sequence; the fix is to confirm `shuffle=True` and that your sampler reseeds each epoch. A **single sharp spike that recovers** is a bad batch (a corrupt sample, a label that is out of range, an image of all zeros) and is usually benign if it recovers, but worth logging the batch index for. A **spike that does not recover and slides into `NaN`** is the numerical event from section 2, and the grad norm will have warned you. A **staircase** — flat then a sudden drop then flat again — is often a learning-rate-schedule boundary (a step scheduler dropping the LR) or, less happily, the model finally escaping a plateau. And the **too-smooth curve**, a loss that descends with implausibly little noise, is the eeriest signature of all: it often means you are evaluating on data the model has memorized, a leak that makes the loss artificially clean. The loss-curve field guide covers these shapes exhaustively; the instrumentation lesson is that you only get to read the texture if you log the raw per-step value, not just the EMA.

What loss rules in: divergence (loss climbing), a dead run (loss perfectly flat from step 0), gross overfitting (train falls, val rises). What loss rules **out**: essentially nothing about location. Two completely different bugs — a vanishing gradient in one block and a learning rate that is 100× too small — produce the identical symptom, a flat loss. That ambiguity is the reason the rest of this post exists.

## 2. Gradient norm: the single most valuable extra signal

If you may add exactly one signal beyond loss, add the **gradient norm**. It is cheap, it is universal, and it disambiguates more bugs per byte logged than anything else.

### What it is, precisely

After `loss.backward()`, every parameter $p$ that requires a gradient has a `.grad` tensor of the same shape. The **global gradient norm** is the L2 norm of all of those gradients concatenated into one long vector:

$$
\|g\|_2 = \sqrt{\sum_{p} \sum_{i} \left(\frac{\partial \mathcal{L}}{\partial p_i}\right)^2}.
$$

This is exactly the quantity `torch.nn.utils.clip_grad_norm_` computes before it decides whether to clip, so if you already clip gradients you are already computing it — you just are not logging it. It is one scalar that summarizes "how big is the step the optimizer is about to take, before the learning rate scales it." Its units are gradient-of-loss-per-unit-parameter, and for a well-behaved transformer or ResNet finetune it typically lives in the range of roughly $0.1$ to $10$, settling toward $1$–$2$ as training stabilizes. Those are order-of-magnitude figures, not laws — they depend on your loss reduction (sum vs mean), batch size, and architecture — but the *dynamics* are what you read, not the absolute value.

### Why the magnitude tells you so much

The science here is short and worth internalizing. Gradient descent updates a parameter by $\Delta p = -\eta \cdot g$, where $\eta$ is the learning rate and $g$ the gradient. The size of the actual change to your weights is therefore $\eta \|g\|$. If $\|g\|$ explodes to $10^4$ and your learning rate is $10^{-3}$, your weights move by an effective magnitude of $10$ in a single step — which, for parameters initialized near unit scale, is a catastrophic jump that throws the network into a region where activations overflow and the very next forward pass produces `inf`, and the backward pass produces `NaN`. That is the mechanism behind "smooth loss, then a spike, then `NaN`": a single step where $\|g\|$ blew up. Conversely if $\|g\|$ collapses to $10^{-7}$, then $\eta\|g\| \approx 10^{-10}$, and the weights do not move at all in any number of steps a human will wait for. The loss is flat not because the model converged but because nothing is changing. Same flat loss, opposite cause, and the gradient norm tells them apart instantly.

There is a subtlety in the absolute magnitude that trips people up, and it is worth getting right so you do not chase a phantom. The gradient norm depends on how you **reduce** the loss over the batch. If your loss is a *mean* over $B$ examples, the gradient is an average and its norm is roughly batch-size-independent. If your loss is a *sum*, the gradient scales with $B$, so a sum-reduced loss on batch 256 has a gradient norm $256\times$ larger than the per-example gradient — and if you then compare it against a "healthy range" you read for a mean-reduced setup, you will think you are exploding when you are fine. The same caveat applies to gradient accumulation: if you accumulate $K$ micro-batches by summing their losses without dividing by $K$, the accumulated gradient is $K\times$ too large, which is a real and common bug (covered in the gradient-accumulation post) but also a reason the *absolute* grad norm is less informative than its *dynamics*. This is why I keep saying "read the dynamics, not the value": a norm that holds steady at $40$ because you sum-reduce a large batch is perfectly healthy; a norm that doubles every step regardless of its starting value is exploding. Log the number, but interpret the *trajectory*.

There is also a clean way to think about *why* gradients vanish or explode with depth, which makes the per-layer signal of the next section feel inevitable. In a deep network, the gradient at layer $\ell$ is a product of Jacobians of every layer above it: $\frac{\partial \mathcal{L}}{\partial h_\ell} = \frac{\partial \mathcal{L}}{\partial h_L}\prod_{k=\ell+1}^{L} \frac{\partial h_k}{\partial h_{k-1}}$. If the typical singular value of each layer's Jacobian is $\sigma$, the gradient magnitude at depth $\ell$ scales like $\sigma^{L-\ell}$. For $\sigma = 0.8$ and 30 layers between $\ell$ and the loss, that factor is $0.8^{30} \approx 1.2\times10^{-3}$; for $\sigma = 1.2$ it is $1.2^{30} \approx 240$. A few percent of systematic shrink or growth per layer, compounded over depth, is the entire vanishing/exploding story — and it is exactly why a global norm cannot see it (the global norm is dominated by the healthy upper layers) and a per-layer norm can (it reads the geometric decay directly). Residual connections and normalization exist to keep that per-layer $\sigma$ near $1$ so the product does not blow up or collapse; when a bug breaks them for one block, that block's gradient falls off the cliff while its neighbors stay healthy.

```python
import torch

def global_grad_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """L2 norm over all parameter gradients. Call after backward(), before step()."""
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(norm_type) ** norm_type
    return (total ** (1.0 / norm_type)).item()
```

Three healthy/unhealthy readings to memorize. **Norm $\approx 1$–$5$, slowly shrinking:** healthy, leave it alone. **Norm climbing past $10^2$ and accelerating:** an explosion in progress; you have a few steps before `NaN`, and the fix is gradient clipping plus, usually, a lower learning rate (this is the territory of [gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing)). **Norm pinned at $10^{-6}$ or below:** vanishing — the model is receiving no learning signal, and the cause is upstream (dead activations, a detached sub-graph, a frozen-by-accident layer, fp16 underflow). **Norm exactly $0.0$:** something is not in the graph at all — a parameter with `requires_grad=False` you did not intend, an output that never reaches the loss, or a `target_modules` mistake in a LoRA config that left the adapter dangling.

#### Worked example: the spike that became a NaN

A 350M-parameter decoder finetune ran clean for 1,100 steps: loss EMA gliding from 3.1 down to 1.9, global grad norm hovering at $1.4$. At step 1,147 the per-step loss logged 11.2 and the global grad norm logged $8.6\times10^3$. At step 1,148 the loss was `NaN` and stayed there. Without the grad-norm log, this looks like a mysterious sudden death. With it, the story is unambiguous: a single batch produced a gradient $6000\times$ larger than normal, the optimizer took a step of effective size $\eta \cdot 8.6\times10^3 = 2\times10^{-5}\cdot 8.6\times10^3 \approx 0.17$ in parameter space — enough, on top of an already-large pre-norm activation, to push the next forward pass into fp16 overflow. The confirming test: re-run with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` and watch the grad-norm log show the clip engaging at step 1,147 (raw norm $8.6\times10^3$, clipped to $1.0$) and the loss continue down instead of exploding. After the fix: 5,000 steps, no `NaN`, final loss 0.81. The grad-norm log is what let me say "a bad batch at 1,147," not "the run died, unclear why."

## 3. Per-layer gradient norm: localizing the failure

The global gradient norm tells you the *size* of the problem. The **per-layer** gradient norm tells you the *location*, and location is what saves the four days from the intro.

![A backward dataflow graph from the loss through the head and layers 5, 3, and 1, where head and layer 5 carry healthy gradient norms near one while layers 3 and 1 collapse to 1e-7 and 1e-8, yet the global norm of 2.0 still looks acceptable.](/imgs/blogs/instrumenting-a-training-run-what-to-log-3.png)

Here is the trap the global norm sets. Suppose your network has twelve blocks. Eleven of them have healthy gradients of norm $\approx 1$, and one — block 3 — has a gradient of norm $10^{-7}$. The global norm is dominated by the eleven healthy blocks: $\sqrt{11 \cdot 1^2 + (10^{-7})^2} \approx 3.3$, a perfectly healthy-looking number. The global instrument reads green while one-twelfth of your model is dead. **You cannot see a localized vanishing gradient in the global norm.** You can only see it per-layer, and once you log it per-layer it is impossible to miss: eleven lines clustered around $1$ and one line flat-lining at $10^{-7}$.

```python
def per_layer_grad_norms(model: torch.nn.Module) -> dict:
    """One L2 grad norm per named parameter. Reveals which block is dead."""
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[f"grad_norm/{name}"] = p.grad.detach().norm(2).item()
        else:
            norms[f"grad_norm/{name}"] = 0.0   # None grad == not in the graph
    return norms
```

Logging one line per parameter is too much for a large model — a 32-layer transformer has hundreds of parameter tensors. The practical move is to log per **parameter group** (per block, or per module type) by aggregating, and to log the full per-parameter detail only as a histogram or only when the global norm leaves its healthy band. A clean grouping:

```python
import re
import collections

def grouped_grad_norms(model: torch.nn.Module) -> dict:
    """Aggregate grad norms by transformer block index (layers.<i>...)."""
    sq_by_group = collections.defaultdict(float)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        m = re.search(r"layers?\.(\d+)", name)
        group = f"block_{m.group(1)}" if m else "other"
        sq_by_group[group] += p.grad.detach().norm(2).item() ** 2
    return {f"grad_norm_group/{g}": v ** 0.5 for g, v in sq_by_group.items()}
```

What per-layer grad norm rules in: a localized vanishing or exploding gradient, a frozen-by-accident block (its norm is exactly $0.0$), a detached sub-graph (norm $0.0$ on everything downstream of the detach), a LoRA adapter that is not training (the base weights have grad $0.0$ as intended, but so do the adapter weights — that is the no-op signature). What it rules **out**: once every block shows a healthy, comparable grad norm, the bug is *not* a gradient-flow problem and you should stop looking at the model graph and go look at data, learning rate, or evaluation. That negative is as valuable as the positive — it ends a whole line of fruitless investigation.

#### Worked example: vanishing gradient at layer 3

This is the run from the introduction, now with instruments. A 12-block encoder finetune, loss flat at 2.31 nats for 600 steps. Global grad norm: $2.0$ — looks fine, which is exactly why the global signal sent me chasing the learning rate and the data for four days. I added the `grouped_grad_norms` function above and re-ran for fifty steps. The log:

| Block | Grad norm | Reading |
| --- | --- | --- |
| block_0 (embed) | 1.7 | healthy |
| block_1 | 1.2 | healthy |
| block_2 | 0.9 | healthy |
| block_3 | 1.1e-7 | **dead** |
| block_4 | 4e-7 | starved (downstream of 3) |
| block_5–11 | 0.6–1.4 | healthy |

Block 3's gradient was seven orders of magnitude below its neighbors, and blocks downstream of it were starved. The cause turned out to be a normalization placement that, combined with a residual connection, let the main path bypass block 3's transformation almost entirely — the gradient that should have flowed *through* block 3 instead flowed *around* it. The fix was a one-line change to where the LayerNorm sat (the mechanism is in [initialization and normalization bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)). After the fix, block 3's grad norm rose to $0.8$ within ten steps and the loss broke off its plateau: 2.31 → 1.4 → 0.74 over the next 400 steps.

![A two-column before-and-after figure where the left column shows a loss-only view stuck at a 2.31 plateau with a healthy-looking global norm and three competing suspects, and the right column shows per-layer logging flagging layer 3 at grad norm 1e-7 and the loss falling to 0.74 after a localized fix.](/imgs/blogs/instrumenting-a-training-run-what-to-log-4.png)

The four days were not a skill problem. They were an instrument problem. The information needed to solve it in ninety seconds was always present in the gradients; it just was not being logged.

## 4. Parameter norm, update norm, and the update-to-weight ratio

The gradient norm tells you the size of the *proposed* step. But the optimizer scales it (by the learning rate, by Adam's per-parameter adaptive scaling, by momentum), so the gradient norm is not the size of the *actual* step your weights take. For that you want the **update norm**, and — the genuinely diagnostic quantity — the **update-to-weight ratio**.

### The ~1e-3 heuristic and why it exists

Andrej Karpathy's well-known "recipe for training neural networks" recommends logging the ratio of the magnitude of the weight update to the magnitude of the weights themselves, and notes that a healthy value sits around $10^{-3}$. The quantity is, per parameter tensor:

$$
r = \frac{\|\Delta p\|_2}{\|p\|_2} = \frac{\|\,p_{t+1} - p_t\,\|_2}{\|p_t\|_2},
$$

the relative size of one step. Why $10^{-3}$? The intuition is a signal-to-noise argument. Your parameters encode the function the network has learned so far. Each step should *refine* that function, not *overwrite* it. A ratio of $10^{-3}$ means each step moves each weight by about one part in a thousand — small enough that the accumulated knowledge is preserved across steps, large enough that over thousands of steps the weights can travel a meaningful distance. If $r \approx 10^{-1}$, you are rewriting 10% of every weight every step; the network has no stable representation, the loss thrashes, and you are effectively re-initializing constantly — that is too high a learning rate, full stop. If $r \approx 10^{-5}$, you are moving one part in a hundred thousand per step; at typical step counts the network barely moves from its initialization — that is too low a learning rate, and the "flat loss" is just slowness. The ratio converts the abstract question "is my learning rate right?" into a number you read against a target.

This is the second instrument that disambiguates a flat loss. The grad norm catches vanishing/exploding; the ratio catches LR-too-low vs LR-too-high *when the gradient itself is healthy*. A flat loss with a healthy grad norm of $2.0$ and a ratio of $10^{-5}$ is not a model bug at all — it is a learning rate that is roughly 100× too small. The mechanics of LR symptoms get the full treatment in the optimization track; here the point is that the ratio is the instrument that reads them.

![A two-column before-and-after figure contrasting an update-to-weight ratio of 1e-1 that produces a loss spike then NaN and calls for cutting the learning rate tenfold, against a ratio of 1e-5 that produces a flat crawling loss over three hundred steps and calls for raising the learning rate toward the 1e-3 target.](/imgs/blogs/instrumenting-a-training-run-what-to-log-5.png)

### Computing it correctly

The subtlety: the update is `p_after_step - p_before_step`, so you must snapshot the parameters *before* `optimizer.step()` and diff *after*. Do not approximate the update as $-\eta g$ — that is wrong for Adam, momentum, and weight decay, which all reshape the step. Snapshot and diff:

```python
import torch

class UpdateRatioTracker:
    """Logs ||delta p|| / ||p|| per parameter group. Snapshot before step()."""
    def __init__(self, model):
        self.model = model
        self._snapshot = None

    def before_step(self):
        self._snapshot = {n: p.detach().clone()
                          for n, p in self.model.named_parameters()
                          if p.requires_grad}

    def after_step(self) -> dict:
        out = {}
        for n, p in self.model.named_parameters():
            if n not in self._snapshot:
                continue
            update = (p.detach() - self._snapshot[n]).norm(2)
            weight = self._snapshot[n].norm(2).clamp_min(1e-12)
            out[f"update_ratio/{n}"] = (update / weight).item()
        return out
```

Snapshotting every parameter every step doubles your parameter memory transiently and is not free, so in practice you run this tracker every $N$ steps (say every 50) and on a *subset* of parameter groups — the first, middle, and last block are usually enough to see the LR story. What the ratio rules in: LR too high (ratio $\gg 10^{-3}$), LR too low (ratio $\ll 10^{-3}$), a layer whose LR is effectively wrong because of a per-layer LR schedule or a frozen group (ratio $\approx 0$ where you expected motion). What it rules **out**: when the ratio sits at $10^{-3}$ across groups and the loss is *still* flat, the learning rate is not your problem — go look at data and the gradient signal instead. The **parameter norm** $\|p\|$ on its own is a slower instrument but worth a log line too: a parameter norm that grows without bound across training signals missing or too-weak weight decay (the weights are drifting to large scale), and a parameter norm that collapses toward zero signals over-aggressive weight decay or a regularization bug.

One more reason to log the ratio rather than reason about $\eta$ in your head: with **Adam**, the effective step size is *not* $\eta\|g\|$. Adam normalizes each parameter's gradient by a running estimate of its own magnitude, so the per-parameter update is approximately $\eta \cdot \text{sign}(g)$ in the limit — the raw gradient magnitude largely cancels out. This is wonderful for robustness but it means the gradient norm and the update size are only loosely coupled under Adam, and the only honest way to know how big your steps actually are is to measure them. A run can have a perfectly healthy grad norm and still take steps that are 100× too large because the learning rate is wrong; Adam's normalization hides that in the grad norm and exposes it in the ratio. SGD-with-momentum is closer to the $\eta\|g\|$ intuition but momentum still smears the step across time. Measure, do not derive.

#### Worked example: the finetune that "wouldn't learn"

A team finetuning a 7B base model on an instruction dataset reported a loss that "barely moved" — it drifted from 1.62 to 1.58 over 200 steps and then stalled. Grad norm: a healthy $1.1$, steady. Activations: alive, no dead units. Every correctness instrument read green, which is precisely why the team had spent a day suspecting the data and the chat template. The update-to-weight ratio, logged on the first, middle, and last transformer block, read $7\times10^{-6}$, $9\times10^{-6}$, and $1.1\times10^{-5}$ — uniformly about 100× below the $10^{-3}$ target. The diagnosis was immediate and embarrassing: they had copied a from-scratch learning rate of $2\times10^{-7}$ (intended for a different, much larger pretraining run) into a finetune that wanted roughly $2\times10^{-5}$. The weights were moving one part in a hundred thousand per step; at 200 steps the model had traveled essentially nowhere from its initialization. The fix was a single config line, raising the LR by 100×. After: the ratio climbed to $\approx 1.4\times10^{-3}$, the loss fell from 1.58 to 0.91 over the next 300 steps, and the model began following instructions. The cost of *not* logging the ratio was a full day of chasing the wrong suspect; the cost of logging it was one number that named "optimization, LR too low" on step 20. This is the single most common finetuning mistake — the right finetuning LR is often 100× smaller than a from-scratch LR, and people overcorrect in both directions — and the ratio is the instrument that ends the argument.

## 5. Activation and weight histograms: why a scalar is not enough

The signals so far are scalars or per-group scalars. Activations are different: the thing you need to know about an activation is not its mean, it is its **distribution**, and a scalar mean can be healthy while the distribution is pathological. This is the case where a histogram earns its extra cost.

![A two-row by three-column grid reading an activation histogram, where ninety percent of the mass piles in the equals-zero bin marking dead ReLUs while the mean of 0.04 and standard deviation of 0.11 in the second row both look acceptable, demonstrating that a scalar summary hides the dead units.](/imgs/blogs/instrumenting-a-training-run-what-to-log-6.png)

### The dead-ReLU case, quantitatively

Consider a ReLU layer where 90% of the units output exactly zero for every input in the batch — dead units, stuck in the regime where their pre-activation is always negative so the ReLU clamps them to zero and, crucially, passes zero gradient back (the derivative of ReLU at a negative input is zero). The 10% that are alive output values in, say, $[0, 1]$. What is the mean activation? Roughly $0.1 \times 0.5 = 0.05$ — a small but perfectly innocuous-looking positive number. The standard deviation is similarly unremarkable. **If you log only the mean and std, this layer looks fine.** It is not fine: 90% of its capacity is gone, and because dead ReLUs pass zero gradient, those units cannot recover on their own — they are dead forever unless you intervene. The histogram makes this impossible to miss: a giant spike of mass in the zero bin and a thin tail of live values. The full diagnosis of dead and saturated units — how they arise, how to revive them — is the subject of [dead neurons and saturated activations](/blog/machine-learning/debugging-training/dead-neurons-and-saturated-activations); the instrumentation point is that you must log the *shape*, not the moment.

Why do units die in the first place, and why does the *initialization* show up in the histogram? A ReLU unit dies when its pre-activation $z = w^\top x + b$ is negative for essentially every input in the data distribution — once that happens, the gradient through that unit is zero, so the optimizer can never push $z$ back into the positive regime, and the unit is stuck. Two things drive units negative: a too-large learning rate that knocks a unit's weights into a bad region in one big step (a single update of effective size 10, as in section 2, can kill a large fraction of a layer's units at once), and a bad initialization that starts too many units with negative bias or with weights at the wrong variance. This is exactly why He initialization exists: for a ReLU layer with $n$ inputs, He init draws weights with variance $2/n$ so that the *variance of the pre-activations is preserved* across the layer (the factor of 2 compensates for ReLU zeroing half the distribution). Get that variance wrong — use Glorot/Xavier init (variance $1/n$, designed for tanh) on a deep ReLU network — and the pre-activation variance halves at every layer, the distribution narrows toward zero, and a growing fraction of units fall permanently into the dead zone with depth. The activation histogram reads this directly: a healthy ReLU layer shows roughly half its mass at exactly zero (the negative pre-activations, correctly clamped) and a smooth tail of positive values; a layer with bad init shows the zero-spike growing past 80–90% and the positive tail collapsing toward zero. You are, in effect, watching the init variance through the histogram.

The same logic applies to saturation in sigmoid/tanh networks (mass piling up at the $\pm 1$ extremes where the gradient is near zero) and to scale drift (the histogram of a layer's activations slowly widening or shifting across training, which precedes overflow). Saturation is the mirror image of death: where a dead ReLU sits at zero gradient because its input is too negative, a saturated sigmoid sits at near-zero gradient because its input is too *large* in magnitude — the sigmoid's derivative $\sigma(z)(1-\sigma(z))$ is maximal at $z=0$ ($=0.25$) and falls toward zero as $|z|$ grows, so a unit whose pre-activation has drifted to $\pm 8$ has a gradient of roughly $0.00033$ and learns 750× slower than a centered unit. Both failure modes are a *distributional* statement about pre-activations, and both are visible only in the histogram. Three cheap distributional checks worth logging per layer: the **fraction of exact zeros** (dead-ReLU detector), the **fraction in the saturated tail** (for bounded activations), and the **max absolute value** (overflow early-warning). All three are one-line reductions you can compute without ever copying the tensor to CPU, which is what makes them cheap enough to log frequently while the full histogram stays on the slow cadence.

### Logging histograms cheaply with hooks

You do not want to manually instrument every layer. PyTorch forward hooks let you attach a stats collector to every module of a given type in a few lines, and you control the cost by only running the hook every $N$ steps and by computing cheap summaries instead of full histograms when you can.

```python
import torch

class ActivationStatsHook:
    """Attach to modules; capture cheap distributional stats on the forward output."""
    def __init__(self):
        self.handles, self.stats = [], {}

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            x = output.detach().float()
            self.stats[name] = {
                f"act/{name}/mean": x.mean().item(),
                f"act/{name}/std": x.std().item(),
                f"act/{name}/frac_zero": (x == 0).float().mean().item(),
                f"act/{name}/max_abs": x.abs().max().item(),
            }
        return hook

    def attach(self, model, types=(torch.nn.ReLU, torch.nn.GELU)):
        for name, m in model.named_modules():
            if isinstance(m, types):
                self.handles.append(m.register_forward_hook(self._make_hook(name)))
        return self

    def collect(self) -> dict:
        flat = {}
        for d in self.stats.values():
            flat.update(d)
        return flat

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
```

For a genuine histogram (the visual that makes dead units obvious at a glance), W&B and TensorBoard both accept a tensor and bucket it for you — `wandb.log({"act/block3": wandb.Histogram(x.cpu())})` or `writer.add_histogram("act/block3", x, global_step)`. Run those every few hundred steps, not every step — a histogram of a large activation tensor is the most expensive thing in this post (it copies the tensor to CPU and buckets it), and at every-step cadence it will dominate your step time. The **weight** histogram is the same tool pointed at `p.data` instead of an activation: a weight histogram that develops a heavy tail of large-magnitude weights signals an exploding-weights problem upstream of any loss spike, and one that collapses toward a delta at zero signals dead capacity. What histograms rule in: dead units, saturation, scale drift, a layer initialized at the wrong variance. What they rule **out**: a healthy bell-shaped activation histogram with no zero-spike means the layer is alive and you should stop suspecting dead neurons there.

## 6. Learning rate, throughput, and data stats: the signals everyone skips

The remaining three signals are the ones people leave out because they feel like overhead — and each one has saved a run for me that the fancier signals could not.

### Log the actual scheduled learning rate

Not the learning rate you configured — the learning rate the scheduler is *actually applying this step*. These differ more often than you would believe. Warmup means your effective LR is near zero for the first hundreds of steps; a cosine or linear decay means it is dropping every step; a bug in how you constructed the scheduler (wrong total-steps, a `step()` called in the wrong place, a resume that reset the schedule) can leave the LR pinned at zero or stuck at the peak. The symptom of a scheduler bug — a loss that never moves because the effective LR is zero, or a loss that diverges because warmup never engaged — is indistinguishable from other bugs *unless you log the actual LR*. It is one line:

```python
lr_log = {"lr": optimizer.param_groups[0]["lr"]}
# if you use per-group LRs (discriminative finetuning), log each:
lr_log = {f"lr/group{i}": g["lr"] for i, g in enumerate(optimizer.param_groups)}
```

The number of debugging sessions I have ended by glancing at the LR log and seeing it flat at $0.0$ for 500 steps (a misconfigured warmup that never ramped) is embarrassing. Log the actual LR. What it rules in: scheduler bugs, warmup that never engaged, a resume that reset the schedule, per-group LRs that are wrong. What it rules out: when the LR log matches the schedule you intended, the scheduler is not your bug.

### Throughput: catch a dataloader bottleneck early

Throughput — samples per second, tokens per second, and the breakdown of where each step's wall-clock time goes — is not a *correctness* signal, it is an *efficiency* signal, but it belongs in your instrument panel because a 30%-utilized GPU is a run that costs you 3× the money and time it should, and you will not notice from the loss curve. The loss converges either way; it just converges slowly and expensively.

![A left-to-right timeline of one training step from a twelve-millisecond forward pass and twenty-four-millisecond backward pass through a three-millisecond loss.item sync stall, a two-millisecond histogram, the optimizer step, and a fast asynchronous log, showing that most logging cost is a hidden synchronization on the hot path.](/imgs/blogs/instrumenting-a-training-run-what-to-log-7.png)

The cheap version is to log GPU utilization and step time and watch for the signature of a dataloader bottleneck: high step time, low GPU util (say 31%), and a *sawtooth* GPU-util trace where the GPU spikes to 100% during compute and drops to 0% while it waits for the next batch. That pattern means your GPU is starving — it finishes the math and sits idle waiting for the CPU dataloader to produce the next batch. The fix is more dataloader workers, prefetching, or moving augmentation off the critical path.

A caution about "GPU utilization" as `nvidia-smi` reports it: that number is the fraction of *time* at least one kernel was running, not the fraction of the GPU's compute *capacity* you used. A run can show 100% utilization while doing tiny, inefficient kernels that use 5% of the hardware's FLOP/s. The honest efficiency metric is **model FLOPs utilization (MFU)**: the ratio of the floating-point operations your model actually needs per step to the operations the GPU could have done in that wall-clock time at its peak rate. If a forward+backward pass requires $6ND$ FLOPs for a model of $N$ parameters processing $D$ tokens (the standard approximation: roughly 6 FLOPs per parameter per token, counting forward and backward), and your GPU peaks at, say, $312$ TFLOP/s in bf16, then MFU is $\frac{6ND/\text{step-time}}{312\times10^{12}}$. A well-tuned large-model training run lands around 40–55% MFU; if you measure 12%, you are leaving most of the hardware on the table, and the cause is usually one of the throughput bugs — a dataloader stall, small batches that under-fill the kernels, or a memory-bound bottleneck. The deep version uses the PyTorch profiler to attribute time across dataloading, host-to-device copy, forward, backward, and optimizer step; the full methodology — and the roofline reasoning behind *why* a kernel is memory-bound vs compute-bound — lives in [the GPU is idle: throughput debugging](/blog/machine-learning/debugging-training/the-gpu-is-idle-throughput-debugging) and, for the hardware-level "where is my bottleneck" picture, in [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives). A minimal step-time breakdown:

```python
import time, torch

class StepTimer:
    """Wall-clock breakdown of a step. Use torch.cuda.synchronize for honest numbers."""
    def __init__(self):
        self.t = {}

    def mark(self, name):
        torch.cuda.synchronize()          # honest timing needs a sync HERE, not in the hot path
        self.t[name] = time.perf_counter()

    def report(self) -> dict:
        keys = list(self.t)
        out = {}
        for a, b in zip(keys, keys[1:]):
            out[f"time/{b}"] = (self.t[b] - self.t[a]) * 1000  # ms
        return out
```

Note the deliberate tension: honest timing *needs* a `cuda.synchronize()`, which is exactly the kind of sync you want to avoid in your hot loop (section 9). The resolution is to time only every $N$ steps, accept that the timed steps are slightly slower, and never call `synchronize` on the un-timed steps. What throughput rules in: a dataloader bottleneck, a host-to-device transfer stall, an under-fed GPU, a CPU-bound augmentation. What it rules out: if util is 90%+ and step time is stable, your slowness is real compute, not a pipeline bug, and you should be looking at the model size or precision, not the dataloader.

#### Worked example: the 31%-utilized run that cost triple

A computer-vision finetune on a single high-end GPU ran at 480 ms/step and was projected to take 14 hours for the planned 100k steps. The loss was descending normally, so nobody questioned the speed — the run was "working." The throughput log told a different story: GPU utilization averaged 31% with a clean sawtooth, and the step-time breakdown read forward 70 ms, backward 130 ms, optimizer 20 ms, and a 260-ms gap before the next forward where the GPU sat idle. That 260 ms was the dataloader: heavy image decoding and augmentation running on a single worker process, unable to prepare the next batch while the GPU computed. The fix was three config changes — `num_workers=8`, `pin_memory=True`, and `prefetch_factor=4` — which let eight CPU processes decode batches ahead of the GPU. After: util 91%, step time 165 ms, the idle gap gone. The run finished in 4.6 hours instead of 14. At roughly \$2.50 per GPU-hour that is a saving of about \$23 on a single run, and across the dozen runs of a hyperparameter sweep, real money and a day of wall-clock turnaround. The loss curve would have let this run finish, slowly and expensively, with no complaint. Only the utilization and step-time logs revealed that two-thirds of the rented GPU was being paid for and not used.

### Data stats: the bad shard detector

Finally, log per-batch **data statistics**: the input mean and standard deviation, and the label distribution. This is the cheapest possible defense against two silent killers. First, **normalization drift**: if your input mean drifts from the $\approx 0$, std $\approx 1$ you normalized to — say a batch comes in with mean $0.45$ — a shard of your data was not normalized, or was normalized with the wrong statistics, and the model is seeing inputs from a different distribution than the rest. Second, **a bad shard**: if your per-batch label distribution suddenly goes from balanced to 100% one class for a stretch of steps, a data shard is corrupt or your shuffle is broken, and the loss will spike or the model will collapse to predicting the majority. Both are invisible in the loss until they have already done damage; both are obvious in a two-line data-stats log.

```python
def batch_stats(x: torch.Tensor, y: torch.Tensor, num_classes: int) -> dict:
    """Cheap per-batch input and label stats. Catches normalization drift and bad shards."""
    stats = {
        "data/input_mean": x.float().mean().item(),
        "data/input_std": x.float().std().item(),
    }
    if y.dtype in (torch.long, torch.int64):
        counts = torch.bincount(y.flatten(), minlength=num_classes).float()
        frac = counts / counts.sum().clamp_min(1)
        stats["data/label_entropy"] = (-(frac * (frac + 1e-12).log()).sum()).item()
        stats["data/max_class_frac"] = frac.max().item()
    return stats
```

A `max_class_frac` that jumps to $1.0$, or a `label_entropy` that drops toward zero, is a bad shard caught in the act. An `input_mean` that drifts off zero is a normalization bug. The discipline of looking at your data *before* you train — and *while* you train, via these logs — is one of the highest-return habits in machine learning; the broader case for it is in the data-debugging track.

## 7. Putting it together: one instrumented training step

The pieces above are deliberately separate so you can read each one's purpose, but in a real run they live in one module that you wire into your training loop once and forget. Here is the assembled version — a thin `Instruments` class that owns the trackers, a `step` method that respects the cadence policy, and a training loop that calls it. The design principle is that instrumentation should be *additive*: you should be able to drop this in around an existing loop without touching the loss, the model, or the optimizer.

```python
import torch

class Instruments:
    """One object that owns every signal tracker and a cadence policy."""
    def __init__(self, model, optimizer, num_classes, log_fn):
        self.model = model
        self.opt = optimizer
        self.num_classes = num_classes
        self.log = log_fn                       # e.g. wandb.log
        self.loss_tracker = LossTracker()
        self.update_tracker = UpdateRatioTracker(model)
        self.act_hook = ActivationStatsHook().attach(model)
        self._running_loss = None

    def on_loss(self, loss: torch.Tensor):
        # accumulate on-device; NO .item() here (avoids the per-step sync)
        d = loss.detach()
        self._running_loss = d if self._running_loss is None else self._running_loss + d

    def before_step(self, step: int):
        if should_log(step, "per_layer"):
            self.update_tracker.before_step()

    def after_backward(self, step: int) -> dict:
        out = {}
        if should_log(step, "scalar"):
            out["grad_norm/global"] = global_grad_norm(self.model)
        if should_log(step, "per_layer"):
            out.update(grouped_grad_norms(self.model))
        return out

    def after_step(self, step: int, x, y) -> dict:
        out = {"lr": self.opt.param_groups[0]["lr"]}
        if should_log(step, "per_layer"):
            out.update(self.update_tracker.after_step())
        if should_log(step, "data"):
            out.update(batch_stats(x, y, self.num_classes))
        if should_log(step, "histogram"):
            out.update(self.act_hook.collect())
        if should_log(step, "scalar") and self._running_loss is not None:
            interval = max(1, step % 1 or 1)    # de-synced loss read
            out.update(self.loss_tracker.update((self._running_loss).item()))
            self._running_loss = None
        if out:
            self.log(out, step=step)
        return out
```

And the loop, with the instrument calls marked so you can see exactly where each signal is captured — note that the `before_step` snapshot must happen *before* the optimizer moves the weights, the grad-norm read must happen *after* `backward()` and *before* `zero_grad()`, and the update-ratio read must happen *after* `step()`:

```python
instruments = Instruments(model, optimizer, num_classes=1000, log_fn=wandb.log)

for step, (x, y) in enumerate(loader):
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    instruments.before_step(step)               # snapshot params (cadence-gated)

    logits = model(x)
    loss = criterion(logits, y)
    instruments.on_loss(loss)                    # accumulate on device, no sync

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_logs = instruments.after_backward(step) # grad norms BEFORE zero_grad next iter

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    instruments.after_step(step, x, y)           # update ratio, lr, data, histograms
```

The ordering is the whole subtlety. `clip_grad_norm_` returns the *pre-clip* global norm as its return value, so if you already clip you can capture the global grad norm for free from that return rather than recomputing it — `total_norm = torch.nn.utils.clip_grad_norm_(...)` and log `total_norm.item()`. Capture the per-layer norms *before* the optimizer step but *after* backward (the grads exist in that window). Capture the update ratio *after* the step (you need both the before-snapshot and the moved weights). And keep the histogram hooks gated to the slow cadence so the CPU copies do not bleed into every step. Wired this way, a complete panel is six method calls around a loop you already have, and you can disable the whole thing by passing a no-op `log_fn`.

A note on what *not* to instrument. It is tempting, the first time you build this, to log everything — every parameter's norm, every activation's full histogram, every step. Resist it. A dashboard with 400 lines is as blind as one with one line, because no human reads 400 traces. The art is choosing the *handful* of signals that disambiguate the bugs you actually hit: global grad norm, per-block grad norm, the ratio on three blocks, the LR, the throughput, and the data stats. Add the activation histogram on a few suspect layers only when a run misbehaves. The goal is a panel you can read in five seconds and trust, not a telemetry firehose.

## 8. The diagnostic table: signal → bug

Now assemble the panel. The whole reason to log six signals instead of one is that *combinations* of signals fingerprint a bug uniquely. Here is the master table I keep pinned, mapping a symptom (read across the signals) to the suspect place and the confirming test.

![A six-row matrix mapping each logged signal to what it catches, what it misses, and its tell, from loss alone through global and per-layer grad norm, the update-to-weight ratio, the activation histogram, and throughput, showing that reading two or three signals together pins the bug to one place.](/imgs/blogs/instrumenting-a-training-run-what-to-log-2.png)

| Symptom (what the signals read) | Suspect place | Confirming test |
| --- | --- | --- |
| Loss flat; global grad norm $\approx 2$; one block's grad norm $10^{-7}$ | model code (gradient flow) | per-layer grad log; fix init/norm at that block |
| Loss flat; grad norm $\approx 2$; update/weight ratio $10^{-5}$ everywhere | optimization (LR too low) | raise LR until ratio hits $10^{-3}$ |
| Loss spikes then `NaN`; grad norm jumped to $10^4$ at the spike step | numerics / optimization | clip grad norm to 1.0; lower LR; check that batch |
| Loss flat; grad norm $\approx 2$; ratio $10^{-1}$; loss thrashing | optimization (LR too high) | cut LR 10×; watch ratio fall to $10^{-3}$ |
| Loss decent; activation histogram shows 90% zeros in a layer | model code (dead units) | revive via init/LR/normalization at that layer |
| Loss converges slowly; GPU util 31%, sawtooth | systems (dataloader) | profiler; add workers / prefetch |
| Loss spikes for a stretch; `max_class_frac` $= 1.0$ those steps | data (bad shard) | inspect that shard; fix shuffle/sharding |
| Loss flat from step 0; LR log reads $0.0$ for 500 steps | optimization (scheduler) | fix warmup/total-steps; LR should ramp |
| Loss flat; some block's grad norm exactly $0.0$ | model code (frozen / detached / LoRA no-op) | check `requires_grad`, the graph, `target_modules` |

Read the table as a lookup: you observe a row's signal pattern, it names the place, you run the test. That is the bisection the whole series teaches, made mechanical by instrumentation. The same six-signal panel, arranged as the decision tree it implies, routes any stalled run from symptom to suspect in three reads.

![A decision tree rooting at a flat loss of 2.31 nats, branching on the gradient norm to either a vanishing 1e-7 reading that names the model as the suspect or a healthy 2.0 reading that then checks the update ratio of 1e-5 and names optimization, showing three signals route a stall to one place.](/imgs/blogs/instrumenting-a-training-run-what-to-log-8.png)

### The grad-to-param ratio and update cosine (optional, for the curious)

Two more signals worth knowing about, both cheap to add once you have the machinery above. The **grad-to-param ratio per layer**, $\|g_\ell\| / \|p_\ell\|$, is a scale-free version of the gradient norm that lets you compare layers of very different sizes on one axis — a layer whose grad-to-param ratio is two orders of magnitude below its neighbors is starved relative to its scale even if its raw grad norm looks fine. The **cosine of consecutive updates**, $\cos(\Delta p_t, \Delta p_{t-1})$, measures whether successive steps point in a consistent direction (cosine near $+1$, smooth descent) or fight each other (cosine near $-1$ or oscillating, a sign of an LR that is too high and bouncing across a valley). You will not need these on most runs, but when a run is misbehaving in a way the core six cannot explain, the update cosine in particular is a sharp instrument for "is the optimizer making consistent progress or thrashing?"

## 9. The cost of logging, and how to log without slowing the run

Instrumentation that halves your throughput is a tax you will eventually stop paying — and then you are flying blind again. So the engineering question is as important as the science: how do you log all of this *cheaply*? Three costs dominate, and each has a clean mitigation.

### The `.item()` sync stall

This is the one that surprises people. PyTorch CUDA operations are **asynchronous**: when you call `loss.backward()`, the kernels are *enqueued* on the GPU and Python returns immediately, before the math is done. This is the whole reason the GPU stays busy while Python prepares the next step. But the moment you call `loss.item()` — to log the loss as a float — Python *must* wait for the GPU to finish computing that value, because it needs the actual number. That wait is a **synchronization point**: it stalls the CPU until the GPU drains its queue, and it can cost a few milliseconds per call. Do it once per step and on a step that takes 40 ms you have added 3 ms — about 8% — for a single scalar. Do it for twenty logged scalars per step and you have throttled your run badly.

The mechanism, stated precisely: every `.item()`, every `.cpu()`, every `print(tensor)`, every `tensor.tolist()` forces a device-to-host sync. The mitigation is to **accumulate logged tensors on the device and sync rarely**. Keep your running loss as a GPU tensor, add to it each step without `.item()`, and call `.item()` once every $N$ steps when you actually log:

```python
import torch

# WRONG: a sync every single step
for batch in loader:
    loss = step(batch)
    running += loss.item()          # <-- device->host sync, ~3 ms, every step

# RIGHT: accumulate on device, sync once per logging interval
running = torch.zeros((), device="cuda")
for i, batch in enumerate(loader):
    loss = step(batch)
    running += loss.detach()        # stays on GPU, no sync
    if (i + 1) % LOG_EVERY == 0:
        avg = (running / LOG_EVERY).item()   # one sync per LOG_EVERY steps
        log({"loss/ema": avg})
        running.zero_()
```

#### Worked example: the 8% instrumentation tax

A vision run, batch 256, measured at 41 ms/step bare. I added naive logging: `loss.item()` every step, plus a per-layer grad-norm dict that called `.item()` on each of 48 parameter norms every step, plus an activation histogram (CPU copy + bucket) every step. New step time: 67 ms — a 63% slowdown, turning a 6-hour run into a 10-hour one and, at roughly \$2.50 per GPU-hour on the instance I was using, adding about \$10 of pure waste to a single run (and far more across a sweep). The fixes, in order of impact: move the activation histogram to every 200 steps (recovers most of it — the CPU copy was the dominant cost); accumulate the loss on-device and sync every 50 steps; compute grad norms with `torch._foreach_norm` over the whole parameter list in one fused call and sync once instead of 48 times. After: 43 ms/step — a 5% overhead for the full instrument panel, which is a price I will gladly pay forever. The principle: **the information is cheap; the synchronization is expensive.** Decouple them.

### Cadence: log every N steps, and stagger the expensive ones

Not every signal needs every-step cadence. Loss EMA, global grad norm, and LR are cheap enough for every step (once you have removed the sync). Per-layer grad norms, the update-ratio snapshot, and data stats are fine every 10–50 steps. Activation and weight histograms — the expensive ones — belong at every 200–500 steps, and you can stagger them so no single step pays for all of them at once. A clean cadence policy:

```python
def should_log(step: int, kind: str) -> bool:
    cadence = {
        "scalar": 1,        # loss, global grad norm, lr  (cheap after de-sync)
        "per_layer": 25,    # grad norm per group, update ratio
        "data": 25,         # input mean/std, label dist
        "histogram": 250,   # activation/weight histograms (expensive: CPU copy)
    }
    return step % cadence.get(kind, 1) == 0
```

### Async and fused computation

Two more cheap wins. First, compute grad norms with the **fused foreach** kernels so the whole model's per-parameter norms come back in one launch instead of one per parameter:

```python
grads = [p.grad for p in model.parameters() if p.grad is not None]
norms = torch._foreach_norm(grads)            # one fused kernel, not N launches
total = torch.stack(norms).norm(2)            # global norm; still one sync to read
```

Second, the logging *backend* call itself (the network round-trip to W&B, the disk write to TensorBoard) should not block your training step. W&B's `wandb.log` is already non-blocking by default — it buffers and flushes on a background thread — and TensorBoard's `SummaryWriter` buffers to disk; the thing that blocks is not the log call, it is the `.item()`/`.cpu()` that *produces the numbers you pass to it*. So the rule is simple and bears repeating one last time: **de-sync the measurement, batch the cadence, fuse the computation, and let the backend flush async.** Do that and a complete six-signal instrument panel costs you single-digit percent — cheap insurance against the four-day blind flight.

## When this is (and isn't) an instrumentation story

Instruments are necessary, not sufficient, and it is worth saying plainly where they stop helping. If your **overfit-one-batch test fails** — the model cannot drive loss to near zero on a single batch it sees over and over — no amount of full-run instrumentation will save you, because the bug is in the basic forward/backward/optimize loop and you should be debugging *that* in isolation first (this is exactly what the overfit-a-single-batch test, the highest-leverage sanity check in the series, is for). Instruments tell you about a run that is *mostly working but subtly wrong*; they are not a substitute for the make-it-fail-small discipline that confirms the loop works at all. The two tools are complementary and ordered: first make-it-fail-small to confirm the loop is correct in miniature, then read-the-instruments to find the subtle bug in the full run. Reaching for the instrument panel when the basic loop is broken is like checking the altimeter when the engine will not start.

A second boundary: a **smooth loss that suddenly goes to `NaN` with no grad-norm warning** is almost always a numerics story (a `log(0)`, a `sqrt` of a negative, an fp16 overflow on a single activation), not something the grad norm will pre-warn you about, because the bad value can appear in the *forward* pass before it ever reaches the gradients. For that you reach for `torch.autograd.set_detect_anomaly(True)` and the NaN-hunting bisection, not the grad-norm log. And a third: instruments cannot see **a leak in your evaluation set**. A run with a perfect six-signal panel — healthy grad norms, ratio at $10^{-3}$, alive activations, 92% util — can still be reporting a validation accuracy that is a data leak, because the leak is in the *data*, not in the *optimization dynamics* the instruments measure. When the run looks healthy by every instrument and the *result* is still too good or too bad to be true, stop reading the instruments and go audit the data and the metric. The instruments answer "is the optimization healthy?" — a real and important question, but not the only one.

The honest summary: the six-signal panel turns the large class of "my run is silently misbehaving" bugs from days into minutes, and it does so by replacing one ambiguous number with six disambiguating ones. It does not replace the small-scale sanity tests that come before it, and it does not see bugs that live entirely in the data or the metric. Know which question you are asking.

The deeper point — and the reason to build this *before* you have a bug rather than after — is that instrumentation is a property of the *system*, not of any one debugging session. The four days I lost in the introduction were not lost because I lacked skill; they were lost because the run was not instrumented, and there was no way to read the gradients after the fact. A run you cannot see into is a run you cannot debug, no matter how good you are. So the practice that pays off is to wire the panel into your training harness on day one — make per-layer grad norm, the update ratio, the LR, the throughput, and the data stats *always-on* (gated to a cheap cadence) so that when a run misbehaves, the evidence is already there waiting for you. The teams that ship models fast are not the ones that debug fastest; they are the ones whose runs were legible from the first step. Build the instruments before you need them, because the moment you need them is the moment it is too late to add them to a run that has already wasted a day. The capstone of this series turns that principle into a concrete checklist for a training system you can debug from day one.

## Case studies and known signatures

A few real and well-known patterns where instrumentation was the difference between a quick fix and a long hunt.

**The vanishing-gradient localization (this post's spine).** The pattern — a healthy global grad norm masking one block flat-lined at $10^{-7}$ — is the canonical case for per-layer logging, and it generalizes far beyond my one encoder. Deep networks without residual connections or normalization were historically *unable* to train for exactly this reason: gradients shrank geometrically with depth, and the early layers received nothing. The reason ResNets and Transformers train at all is that residual connections give the gradient a short path back to every layer. When a normalization or residual placement bug breaks that short path for one block, the per-layer grad norm is the only instrument that sees it. The signature is unmistakable once you log it: a cliff in the grad-norm-vs-layer profile.

**fp16 gradient underflow and loss scaling.** The Mixed Precision Training paper (Micikevicius et al., 2018) documents the mechanism precisely: fp16's smallest normal positive number is about $6.1\times10^{-5}$, so a gradient of magnitude $3\times10^{-6}$ — entirely normal in the deeper layers of a real network — rounds to exactly zero in fp16, and that layer silently stops learning. The instrument that catches this is the **gradient histogram in fp16**: you see mass piling up at zero that should be in the $10^{-6}$–$10^{-5}$ range. The fix, loss scaling, multiplies the loss (and hence all gradients) by a large factor before the backward pass to shift those small gradients up into fp16's representable range, then divides back out before the optimizer step. You only know whether your loss scale is right by *reading the gradient histogram*. This is squarely the territory of mixed-precision debugging, but it is an instrumentation story at heart: the bug is invisible to the loss and obvious to the histogram.

**The Karpathy update-ratio heuristic in the wild.** The $\approx 10^{-3}$ update-to-weight ratio target is folklore that holds up remarkably well across architectures because the underlying signal-to-noise argument is architecture-independent. Practitioners who log it report the same experience: a finetune that "won't learn" shows a ratio of $10^{-5}$ (LR 100× too low — the classic mistake of using a from-scratch LR of $10^{-3}$ on a finetune that wants $10^{-5}$), and a finetune that "destroys the pretrained features" shows a ratio of $10^{-1}$ (LR far too high, rewriting the very weights you wanted to preserve). The ratio converts the LR-for-finetuning question — which trips up nearly everyone the first time, since the right finetuning LR is often 100× smaller than a from-scratch LR — into a number you can read against $10^{-3}$ on step 20 instead of discovering it after a wasted run.

**The dataloader-bound run that nobody profiled.** A widely-repeated pattern: a team scales from 1 GPU to 8 and sees almost no speedup ("8× GPUs, same wall-clock"). The loss curve is identical, so nothing *looks* wrong. The throughput log tells the story immediately: GPU util at 31% with a sawtooth, meaning every GPU is starving on a single-process dataloader that cannot feed eight devices. The fix (more workers, sharded loading, prefetch) takes util to 90%+ and the run finally goes 7× faster. No correctness signal would ever have flagged this — only the throughput instrument.

**The LoRA adapter that never entered the graph.** This is the finetuning instrumentation story I see most often, and it is invisible to every signal except per-layer grad norm. A team configures a LoRA finetune, the loss goes down a little, the run finishes, and the merged model is identical to the base model — the adapter trained, but on the wrong modules, or it was never wired into the forward pass at all. The cause is almost always a `target_modules` list that does not match the actual module names in the checkpoint (e.g. specifying `["q_proj", "v_proj"]` when the model names them `["query", "value"]`), so PEFT silently attaches adapters to nothing. The loss still moves a little because the few correctly-targeted modules (or the bias/embedding terms) train. The instrument that catches it: log the per-parameter grad norm and confirm that the LoRA `A` and `B` matrices have *nonzero* gradients. If `lora_A.default.weight` shows a grad norm of exactly $0.0$, the adapter is a no-op. A one-line confirmation before you spend an hour training is `model.print_trainable_parameters()` from PEFT, which prints how many parameters actually require grad — if that number is implausibly small (or zero), stop. The grad-norm log turns "the merge did nothing, why?" into "the adapter weights have grad $0.0$, the targeting is wrong" in one read. This is the headline diagnostic of the debugging-LoRA work and a perfect illustration of why per-layer grad norm earns its place: the failure is a *location* problem, and only a *located* instrument can see it.

## Key takeaways

- **Loss tells you *that*; the other five signals tell you *where*.** A flat loss has at least four distinct causes (vanishing gradient, LR too low, dead activations, scheduler stuck at LR 0) that are identical in the loss curve and obvious the moment you read grad norm, update ratio, activation histogram, and the actual LR.
- **If you add one signal, add the gradient norm.** Global norm catches explosion (climbing past $10^2$, `NaN` imminent) and gross vanishing; it costs almost nothing if you already clip.
- **Per-layer grad norm localizes the bug.** The global norm can read a healthy $2.0$ while one block sits at $10^{-7}$. Log per parameter group; a block at exactly $0.0$ is frozen, detached, or a LoRA no-op.
- **The update-to-weight ratio reads your learning rate.** Target $\approx 10^{-3}$. Much higher means LR too high (thrashing, then `NaN`); much lower means LR too low (the "flat loss" that is just slowness). Snapshot params before `step()` and diff after — do not approximate the update as $-\eta g$.
- **Histograms beat scalars for activations.** A 90%-dead ReLU layer has an innocuous mean; only the zero-bin spike in the histogram reveals it. Log fraction-of-zeros, saturation-tail fraction, and max-abs cheaply via forward hooks.
- **Log the *actual scheduled* LR, the throughput, and per-batch data stats.** Each catches a class of bug — scheduler misconfiguration, dataloader starvation, a bad shard or normalization drift — that no other signal sees.
- **The cost of logging is synchronization, not computation.** `.item()`/`.cpu()` force a device-to-host sync (~3 ms each). Accumulate on-device, sync every $N$ steps, fuse with `torch._foreach_norm`, stagger histograms to every 200–500 steps. A full panel should cost single-digit percent.
- **Instruments are necessary, not sufficient.** They do not replace the overfit-one-batch test, they do not pre-warn a forward-pass `NaN`, and they cannot see a data or metric leak. Know which question each instrument answers.

## Further reading

- Andrej Karpathy, "A Recipe for Training Neural Networks" (2019) — the origin of the update-to-weight ratio heuristic and the broader "become one with the data, then instrument" methodology.
- Micikevicius et al., "Mixed Precision Training" (ICLR 2018) — the fp16 representable-range argument, gradient underflow, and loss scaling; the canonical case for reading the gradient histogram.
- PyTorch documentation — `torch.nn.utils.clip_grad_norm_`, `register_forward_hook` / `register_full_backward_hook`, `torch.autograd.set_detect_anomaly`, and the PyTorch Profiler tutorial.
- Weights & Biases and TensorBoard documentation — `wandb.Histogram` / `SummaryWriter.add_histogram` and the logging-cadence guidance for distributional logging.
- [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the master symptom → suspect → test → fix decision tree this instrument panel feeds.
- [Reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic) — the companion field guide for what each loss-curve *shape* implies.
- [Gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing) and [dead neurons and saturated activations](/blog/machine-learning/debugging-training/dead-neurons-and-saturated-activations) — the deep dives on the two failure modes the grad-norm and activation-histogram signals catch.
- [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone that assembles every instrument and test into a printable, build-it-debuggable-from-day-one checklist.

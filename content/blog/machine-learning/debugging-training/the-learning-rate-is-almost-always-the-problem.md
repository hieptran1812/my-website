---
title: "The Learning Rate Is Almost Always the Problem"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn to read the loss-landscape physics behind every learning-rate failure, run the LR-range test to pick a rate that works, and stop a too-high finetune from destroying a pretrained model."
tags:
  [
    "debugging",
    "model-training",
    "learning-rate",
    "optimization",
    "finetuning",
    "deep-learning",
    "pytorch",
    "llm",
    "mixed-precision",
    "hyperparameters",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-learning-rate-is-almost-always-the-problem-1.png"
---

A junior engineer pings you on a Friday afternoon. Their BERT finetune is "broken." The loss started at 2.3, dropped to 1.8 in the first dozen steps — looking great — and then, around step 60, leapt to 9.4 and printed `nan` by step 140. They have spent the afternoon rewriting the data collator, swapping the tokenizer, and adding three `assert` statements to the model's forward pass. None of it helped, because none of it was the bug. You ask one question — "what learning rate?" — and they say `1e-2`, because that is the number that was in the tutorial they copied. You tell them to set it to `3e-4` and re-run. The loss drops smoothly to 0.7 over six hundred steps and the model ships. The afternoon is over.

This happens constantly. The learning rate is the single most important hyperparameter in deep learning, the one that most directly controls whether optimization converges at all, and it is also the one engineers most often set by superstition — copied from a blog post, inherited from a config, or left at a framework default that was tuned for a completely different model and batch size. When a training run misbehaves in a way that *feels* mysterious — a sudden spike, a flat line that never learns, a finetune that produces garbage, a run that diverges only sometimes — the learning rate is the first dial you should suspect, and roughly half the time it is the whole story. That is not a rhetorical exaggeration. In the six-places framing this series is built on — a bug hides in **data, optimization, model code, numerics, systems, or evaluation** — the learning rate sits squarely in *optimization*, and it is the highest-prior single cause inside that bucket.

![A workflow graph showing how a loss spike, a slow crawl, and a finetune that forgot route back through an LR-range test to a single well-chosen learning rate](/imgs/blogs/the-learning-rate-is-almost-always-the-problem-1.png)

This post does three things at once, because that is the contract of this series. The **science**: I will derive, from a two-line argument about a quadratic bowl, exactly why a too-high learning rate diverges — the bound $\eta < 2/L$ where $L$ is the curvature — and why a too-low one crawls, and how the learning rate, the batch size, and warmup all fit into the same picture. The **practice**: the LR-range test from Leslie Smith's work, with runnable code you can paste into any PyTorch loop to *find* a good learning rate in a few hundred steps instead of guessing, plus the tooling (`torch_lr_finder`, PyTorch Lightning's `Tuner`) that wraps it. And the **proof**: concrete before-and-after evidence — the run that went from a `nan` at step 140 to a clean descent, and the finetune that went from catastrophic forgetting back to a model that gained on its task while keeping its base skills, by changing one number. By the end you should be able to look at a misbehaving curve, name the learning-rate failure mode, run one cheap test, and set a rate that works — across vision, language, tabular, and speech models, because the physics is the same everywhere.

A note before we start, because it is the most expensive single fact in this post: **a finetuning learning rate is often ten to one hundred times smaller than a from-scratch one.** Training a ResNet or a transformer from random initialization, you might use `1e-3`. Finetuning a pretrained 7B language model, you want something like `1e-5` to `2e-5`. Using the from-scratch rate on a pretrained checkpoint is one of the most common and most destructive mistakes in applied ML, and it has a precise signature — a loss spike in the first handful of steps followed by garbage output — that we will learn to recognize on sight.

## 1. The symptom catalogue: what each mis-set rate looks like

Before any theory, let us fix the three signatures in your eye, because most of the value of this post is being able to recognize them in two seconds from a screenshot. The learning rate has essentially three failure modes and one healthy mode, and they look completely different on a loss curve.

**Too high: the spike.** The loss drops for a few steps — sometimes encouragingly fast — and then jumps to a large value, often an order of magnitude above where it started, and either oscillates wildly or shoots to `inf`/`nan`. The grad-norm, if you are logging it (and you should be, per the [instrumentation post](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log)), spikes simultaneously to something like `1e3` or `1e4` right at the loss spike. This is the most dramatic and the most common acute failure. The key tell is *the spike is sudden and the grad-norm spikes with it*. A loss that drifts up slowly is a different bug; a loss that leaps is an optimization step that overshot.

**Too low: the crawl.** The loss decreases, but agonizingly slowly — a few thousandths of a nat per hundred steps — and flattens into a plateau well above where it should. There is no drama, no spike, no `nan`. The run looks "fine," which is what makes this insidious: people let a too-low-LR run go for days, see the loss inch down, and conclude the model "just needs more data" or "isn't powerful enough." The tell is that the curve is *monotone, smooth, and far too slow* — and crucially, the model underfits even the training set, which rules out overfitting and points straight at optimization.

**Right at the edge: the oscillation.** Just below the divergence threshold, the loss does not blow up but it does not settle either; it bounces — a sawtooth that does not converge to a clean minimum, with a noticeably higher floor than a well-tuned rate reaches. This is the rate sitting near $2/L$ but not over it: stable enough not to explode, too large to settle into the bowl. It is the most often-missed mode because the run does not crash; it just plateaus a little high and people accept it.

**Finetuning a pretrained model: the forgetting spike.** A special, important case. You load a checkpoint that already has low loss on its pretraining distribution, attach a small dataset, and start at a from-scratch learning rate. The very first few steps take huge strides through a parameter space the model was already sitting near the bottom of, and the loss *spikes upward* — from, say, 0.4 to 6.0 in one step — as the optimizer walks the weights away from the pretrained solution. The model "forgets" what it knew. The downstream symptom is catastrophic: garbage generations, a collapse on the base capabilities, a validation number that craters. We give this its own section, because it is the most expensive learning-rate bug in the LLM era and the easiest to fix once you see it.

**Just right: the smooth fast drop.** For contrast, what you want: the loss falls quickly and smoothly in the early phase, the grad-norm stays bounded (single digits, maybe up to low tens during warmup), and the curve bends into a healthy decelerating descent without spikes. That is the target. Everything in this post is in service of getting there.

Here is the symptom-to-suspect table I keep in my head. It is the spine of the whole post; each row gets its own section below.

| Symptom on the loss curve | Most likely cause | One-line confirming test | Fix |
| --- | --- | --- | --- |
| Spike then `nan`, grad-norm spikes with it | LR above the stability bound | Lower LR 10–30×; does the spike vanish? | Lower LR, add warmup, clip grads |
| Slow monotone crawl, underfits train set | LR far too low | Run the LR-range test; is current LR in the flat region? | Raise LR toward the descent knee |
| Sawtooth, plateaus a little high | LR near the $2/L$ edge | Halve LR; does the floor drop? | Halve LR or add a decay schedule |
| First-step loss spike on a finetune, then garbage | From-scratch LR on a pretrained model | Eval the base task before step 1; is it good, then ruined? | Drop LR to `1e-5`–`2e-5`, short warmup |
| Loss diverges only after batch-size change | LR not rescaled for new batch | Apply linear/sqrt rule; does it stabilize? | Rescale LR with batch size |

Notice that four of the five fixes touch the learning rate directly. That is the thesis in a table: when a run misbehaves, the LR is the cheapest, highest-prior thing to check, and you can check it without rewriting a single line of model code.

## 2. The science: why a too-high learning rate provably diverges

Everything above is empirical pattern-matching. Now we earn it. I am going to derive the stability bound $\eta < 2/L$ from first principles on the simplest possible model — a quadratic bowl — because that derivation is the load-bearing insight of this entire post. Once you have it, every learning-rate symptom becomes a prediction rather than a surprise.

### The quadratic bowl

Near a minimum, almost any smooth loss function looks like a bowl. Take the simplest one-dimensional case: a loss

$$
\mathcal{L}(x) = \tfrac{1}{2} L\, x^2,
$$

where $x$ is the (scalar) parameter measured as a displacement from the minimum at $x=0$, and $L > 0$ is the curvature — the second derivative $\mathcal{L}''(x) = L$, constant for a quadratic. The gradient is

$$
\nabla \mathcal{L}(x) = L\, x.
$$

Gradient descent with learning rate $\eta$ updates the parameter by stepping against the gradient:

$$
x_{t+1} = x_t - \eta\, \nabla \mathcal{L}(x_t) = x_t - \eta L\, x_t = (1 - \eta L)\, x_t.
$$

This is the whole derivation in one line. Each step multiplies the *distance to the minimum* by the constant factor $(1 - \eta L)$. So after $t$ steps,

$$
x_t = (1 - \eta L)^t\, x_0.
$$

The behavior of the run is entirely controlled by the magnitude of that multiplier $r = |1 - \eta L|$:

- If $|1 - \eta L| < 1$, the distance shrinks every step and the run **converges**.
- If $|1 - \eta L| = 1$, the distance is constant — the run **oscillates forever** without converging.
- If $|1 - \eta L| > 1$, the distance *grows* every step and the run **diverges**, with the loss, which is $\tfrac12 L x_t^2$, blowing up geometrically.

![A vertical stack deriving the stability bound from the quadratic bowl, from the curvature L through the per-step error multiplier to the lr below two over L condition and the divergence above it](/imgs/blogs/the-learning-rate-is-almost-always-the-problem-3.png)

Now solve $|1 - \eta L| < 1$ for $\eta$. The expression $1 - \eta L$ is below 1 for any positive $\eta$ (since $\eta L > 0$), so the binding constraint is the lower side: we need $1 - \eta L > -1$, i.e.

$$
\eta L < 2 \quad\Longleftrightarrow\quad \boxed{\;\eta < \frac{2}{L}\;}.
$$

There it is. **A learning rate above $2/L$, where $L$ is the curvature, makes gradient descent diverge on a quadratic bowl.** This is not a heuristic; it is exact for the quadratic case, and the quadratic is the leading-order approximation to any smooth loss near a minimum. It tells you precisely why the loss *spikes* rather than drifting: above the threshold, the error grows by a fixed multiplicative factor each step, so the loss grows geometrically — a spike, then `inf`, then `nan` — which is exactly the signature from §1.

### What this predicts, in numbers

A few consequences fall straight out, and they are all things you can see in a real run.

The **optimal** learning rate for a single quadratic direction is $\eta = 1/L$, which makes the multiplier $1 - \eta L = 0$: you jump to the minimum in one step. You never get this lucky in practice (the loss is not a perfect quadratic and has many directions), but it anchors the scale: the useful range of learning rates is roughly $(0, 2/L)$, with the sweet spot near $1/L$, and the "divergence knee" exactly at $2/L$.

For a **multivariate** loss, $L$ generalizes to the largest eigenvalue $\lambda_{\max}$ of the Hessian $H$ — the curvature in the *steepest* direction. The same single learning rate $\eta$ applies to every direction, so the constraint that binds is the most curved one:

$$
\eta < \frac{2}{\lambda_{\max}(H)}.
$$

This is why a model with even one very high-curvature direction (a sharp ravine) forces a small learning rate on *everything*, and why the ratio $\lambda_{\max}/\lambda_{\min}$ — the *condition number* — controls how slowly the low-curvature directions converge. A badly conditioned loss is the deep reason a single global learning rate is a compromise, and the reason adaptive optimizers (Adam, which scales the step per-coordinate by a running estimate of the gradient magnitude) exist at all. Adam does not repeal the $2/L$ bound; it effectively gives each coordinate its own $L$, which is why it is more forgiving of a guessed learning rate but still diverges if you push it far enough.

The bound also explains why **the same learning rate can be fine for one model and explosive for another.** $L$ depends on the architecture, the initialization, the normalization, and even the data scale. A model whose activations are large (because you forgot to normalize the inputs, say) has a larger effective $L$, so the same $\eta$ that was stable becomes too high. This is the mechanism behind "it worked yesterday and diverges today after I changed the preprocessing." We will use it.

#### Worked example: computing the divergence threshold

Suppose you have a simple linear layer and you can estimate the largest curvature. For a least-squares loss $\mathcal{L}(w) = \tfrac{1}{2}\lVert Xw - y\rVert^2$ over $n$ examples, the Hessian is exactly $H = X^\top X$, and its largest eigenvalue is $\sigma_{\max}(X)^2$, the squared largest singular value of the data matrix. Say your features are standardized and you have $n = 4096$ examples with $d = 100$ features; a typical $\sigma_{\max}(X)^2$ for standardized data scales with $n$, landing around, say, $L \approx 8{,}000$ (this is an order-of-magnitude figure — measure it for your own data). Then the stability bound is

$$
\eta < \frac{2}{L} = \frac{2}{8000} = 2.5 \times 10^{-4}.
$$

A learning rate of `1e-2` is forty times over the threshold — guaranteed to diverge. A rate of `2.5e-4` sits right at the edge; you want something below it, say `1e-4`, for a smooth descent. **This is why `3e-4` is such a famous "default" learning rate** (Andrej Karpathy once half-jokingly called it "the best learning rate for Adam, hands down"): for a lot of normalized, well-conditioned setups, the largest curvature lands in a range where a few times $10^{-4}$ is comfortably under $2/L$. It is not magic; it is the curvature of typical normalized problems. And it is exactly why the fix in the opening story — `1e-2` to `3e-4` — worked: it moved the rate from forty times over the bound to safely under it.

The practical upshot: you almost never compute $L$ directly (the Hessian is enormous), but you *measure its consequences* — the grad-norm, the loss spike — and you *find* the right $\eta$ empirically with the LR-range test in §4. The theory tells you the test will work and why the curve will have the shape it does.

### The condition number: why one global rate is always a compromise

There is a second, equally important consequence of going multivariate, and it explains why deep-network training is fundamentally harder than the clean $1/L$ story suggests. A real loss has *many* directions, each with its own curvature — the eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$ of the Hessian. A single global learning rate $\eta$ has to serve all of them at once. The constraint from §2 says $\eta < 2/\lambda_{\max}$, so the *steepest* direction caps the rate. But the *flattest* useful direction, with curvature $\lambda_{\min}$, converges with multiplier $|1 - \eta \lambda_{\min}|$, and since $\eta$ is forced small by $\lambda_{\max}$, that multiplier is very close to 1 — the flat direction barely moves.

Quantify it. The number of steps to make progress along the flattest direction, divided by the number along the steepest, scales with the **condition number** $\kappa = \lambda_{\max} / \lambda_{\min}$. If you set $\eta$ near the stability limit $2/\lambda_{\max}$ (as large as you safely can), the flat direction needs on the order of $\kappa$ times more steps than the steep one. A loss with $\kappa = 1000$ — entirely normal for an under-normalized deep network — means the flat directions converge a thousand times slower than the sharp ones at any single safe rate. This is the deep reason that:

- **Normalization helps so much.** BatchNorm, LayerNorm, and good initialization all *reduce the condition number* — they make the curvature more uniform across directions, which shrinks $\kappa$, which lets a single rate serve all directions better. A normalized network can use a larger, more uniform rate; an un-normalized one is stuck choosing between "too slow on flat directions" and "diverges on sharp ones." This is also why a preprocessing bug that un-normalizes your inputs raises the effective $\lambda_{\max}$ and forces your old rate over the bound, the "it worked yesterday" failure.
- **Adam exists.** Adam (and RMSProp before it) divides each coordinate's step by a running estimate of that coordinate's gradient magnitude, which approximately *equalizes* the effective curvature across coordinates — it gives each direction its own $\eta$, dramatically improving the conditioning the optimizer *sees*. That is why Adam tolerates a guessed rate that SGD would diverge on, and why the steep band in an LR-range test is wider for Adam than for SGD. Adam does not break $2/L$; it makes $L$ look more uniform.

The takeaway for debugging: if your loss crawls on the flat directions *despite* a rate that is at the stability edge on the sharp ones, the problem is conditioning, not the rate per se — and the fix is normalization or an adaptive optimizer, not a bigger global $\eta$ (which would just diverge). Knowing the difference saves you from the classic mistake of cranking the rate to fight a conditioning problem and getting a spike for your trouble.

## 3. The other side: why a too-low learning rate crawls (and underfits)

The diverging case gets all the attention because it is loud. The too-low case is quieter and, in aggregate, wastes far more compute, because runs that crawl get *babysat* for days before anyone admits the rate is the problem.

Go back to the multiplier $r = |1 - \eta L|$. If $\eta$ is tiny, then $\eta L \ll 1$, so $r \approx 1 - \eta L$, a number just barely below 1. The distance to the minimum shrinks by a factor of $(1 - \eta L)$ per step, which for small $\eta L$ is very close to "not at all." Concretely, the number of steps to halve the distance is

$$
t_{1/2} = \frac{\ln(1/2)}{\ln(1 - \eta L)} \approx \frac{0.693}{\eta L} \quad \text{(for small } \eta L\text{)}.
$$

So **halving the learning rate roughly doubles the number of steps to make the same progress.** If $\eta$ is ten times too small, you need ten times the steps — and that is the optimistic case where the loss is locally quadratic. In a deep network the low-curvature directions, which carry a lot of the useful signal, converge slowest of all, so a too-low global rate can leave the model genuinely underfit: it has not had enough effective steps to move the flat directions at all.

This is the diagnostic key. **A too-low learning rate underfits the *training* set.** That distinguishes it cleanly from overfitting (where train loss is low and val loss is high) and from a capacity problem (where even a tiny dataset cannot be fit). If your model cannot drive the loss down on the training data with enough steps, and the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) is also slow, the learning rate is the prime suspect before you reach for "a bigger model" or "more data."

#### Worked example: the crawl that wasn't a capacity problem

A team trains a 3-layer MLP on a tabular fraud dataset, 200k rows. Loss starts at 0.69 (binary cross-entropy at chance, which is $\ln 2 \approx 0.693$), and after 5,000 steps it is at 0.61. Smooth, monotone, no spikes. They conclude the features are weak and spend a week on feature engineering. The actual problem: someone set `lr=1e-5` "to be safe." Re-run with `lr=3e-3` (300× higher) and the loss reaches 0.41 in 800 steps and keeps dropping. The feature engineering was solving the wrong problem. The signature was right there: a *monotone, far-too-slow descent that underfits the training set* is the crawl, and the crawl is a too-low learning rate until the LR-range test says otherwise.

The honesty caveat: a slow start is *not always* a too-low rate. A correctly-warmed-up large model has a deliberately slow early phase. A slow descent that *accelerates* after warmup and then trains fine is healthy. The crawl that is a bug is the one that *stays* slow — that plateaus high and underfits — and the LR-range test is how you tell them apart in five minutes instead of five days.

## 4. The diagnostic: the LR-range test

Here is the single most useful learning-rate tool that almost nobody runs: the **LR-range test**, introduced by Leslie Smith in "Cyclical Learning Rates for Training Neural Networks" (2017) and popularized as the "LR finder." The idea is beautifully cheap. Instead of guessing a rate or grid-searching whole training runs, you do *one* short run of a few hundred steps in which you **exponentially increase the learning rate from very small to very large**, recording the loss at each step. Then you plot loss against learning rate (log-x), and the shape of that curve tells you the usable range directly.

The curve has three regions, and reading them is the whole skill.

![A decision tree that reads the loss-versus-learning-rate sweep into a flat floor, a steep descent, and a divergence knee, with the chosen rate sitting one notch below the knee](/imgs/blogs/the-learning-rate-is-almost-always-the-problem-6.png)

**The flat region (left).** At very small learning rates, each step barely moves the weights, so the loss barely changes — a roughly flat line. Any rate in this region is "too low": it works, but it crawls. You do not want to be here.

**The steep descent (middle).** As the rate grows, you reach a band where the loss falls *fastest* per step. This is the productive range — the rates that make real progress. The steepest point is where the loss is dropping most quickly as you raise the rate.

**The divergence knee (right).** Push the rate higher and you cross $2/L$: the loss stops improving and shoots upward. The point where the curve turns up sharply is the knee, and it marks the empirical stability bound for *this* model on *this* data. Any rate to the right of the knee diverges.

**The rule for picking:** choose a learning rate **one notch below the knee** — specifically, the rate roughly where the loss is still descending steeply, often about an order of magnitude below the divergence point, or at the point of steepest descent. The common heuristic is "the steepest-descent point" or "knee divided by ten." Either gives you a rate that makes fast progress with comfortable margin to the stability bound.

Here is a complete, runnable LR-range test as a manual loop. It is deliberately framework-light so you can see every moving part; you can drop it into any PyTorch training script.

```python
import math
import copy
import torch

def lr_range_test(model, optimizer, loss_fn, loader,
                  lr_start=1e-7, lr_end=1.0, num_steps=300,
                  diverge_factor=4.0):
    """Smith's LR-range test: exponentially ramp the LR over a few hundred
    steps, record (lr, loss), and stop when the loss blows up. Returns the
    swept lists so you can plot loss vs lr and read off the knee."""
    # Save state so the test does not corrupt your real training run.
    model_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(optimizer.state_dict())

    # Geometric LR schedule from lr_start to lr_end over num_steps.
    mult = (lr_end / lr_start) ** (1.0 / num_steps)
    lr = lr_start
    for g in optimizer.param_groups:
        g["lr"] = lr

    lrs, losses = [], []
    best = float("inf")
    smoothed = None
    beta = 0.9  # smooth the loss a little so the knee is readable

    data_iter = iter(loader)
    model.train()
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        raw = loss.item()
        # Exponential moving average so noise does not hide the knee.
        smoothed = raw if smoothed is None else beta * smoothed + (1 - beta) * raw
        debiased = smoothed / (1 - beta ** (step + 1))

        lrs.append(lr)
        losses.append(debiased)
        best = min(best, debiased)

        # Stop early once the loss has clearly diverged past the knee.
        if debiased > diverge_factor * best or math.isnan(raw):
            print(f"diverged at step {step}, lr={lr:.2e}, loss={raw:.3f}")
            break

        # Ramp the LR for the next step.
        lr *= mult
        for g in optimizer.param_groups:
            g["lr"] = lr

    # Restore the original weights/optimizer — the test was a probe, not training.
    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
    return lrs, losses
```

To use it and read the result:

```python
lrs, losses = lr_range_test(model, optimizer, loss_fn, train_loader)

import matplotlib.pyplot as plt
plt.plot(lrs, losses)
plt.xscale("log")
plt.xlabel("learning rate")
plt.ylabel("loss (smoothed)")
plt.savefig("lr_range_test.png")

# A simple automatic pick: the LR of steepest descent (most negative
# gradient of loss w.r.t. log-lr), which sits in the productive band.
import numpy as np
log_lrs = np.log10(np.array(lrs))
loss_arr = np.array(losses)
gradients = np.gradient(loss_arr, log_lrs)
steepest_idx = int(np.argmin(gradients))      # most negative slope
suggested = lrs[steepest_idx]
print(f"suggested lr (steepest descent): {suggested:.2e}")
```

Two things make this trustworthy. First, it **saves and restores model and optimizer state**, so running it does not perturb your real run — it is a pure probe. Second, it **smooths the loss** (a debiased EMA) so the knee is readable through batch noise, and it **stops early** once the loss has diverged past a multiple of its best value, so you never waste steps in the blown-up region.

The library versions wrap exactly this. `torch_lr_finder` gives you `LRFinder(model, optimizer, criterion).range_test(loader, end_lr=1, num_iter=300)` and a `.plot()`. PyTorch Lightning bakes it into the `Tuner`: `Tuner(trainer).lr_find(model)` returns an object with `.suggestion()` and `.plot()`. Use them in production; understand the manual loop so you know what they are doing and can debug them when the suggested rate looks wrong (usually because the loss was too noisy and the auto-pick latched onto a noise dip — smooth harder or run more steps).

#### Worked example: reading a real LR-range curve

You run the test on a ResNet-18 on CIFAR-10, 300 steps, `lr_start=1e-7`, `lr_end=1.0`. The printed curve: loss is flat at ~2.30 (which is $\ln 10$, chance for 10 classes) from `1e-7` up to about `1e-3`; then it descends steeply, reaching ~1.1 around `1e-1`; then at about `3e-1` it turns sharply upward and by `1.0` it is back above 5 and climbing. The knee is at roughly `3e-1`. Steepest descent is around `1e-1`. So you pick `1e-1` for SGD with momentum — a notch below the knee, in the steep band. (For Adam you would land much lower, around `1e-3`, because Adam's per-coordinate scaling changes the effective curvature.) You did this in 300 steps — about thirty seconds on a GPU — and you now have a defensible rate instead of a guess.

## 5. The before-and-after: from `nan` at step 140 to a clean descent

Theory and a diagnostic are nice; the evidence is what makes it real. Here is the full arc of the opening story, instrumented.

The buggy run: a BERT-base finetune for text classification, `lr=1e-2` (copied from a from-scratch tutorial), no warmup, no gradient clipping, fp32. The loss curve and grad-norm:

![A before-and-after figure contrasting a diverging run at learning rate one-hundredth that spikes to a NaN with a swept run at three-ten-thousandths that descends smoothly](/imgs/blogs/the-learning-rate-is-almost-always-the-problem-2.png)

```bash
# Buggy run: lr=1e-2, no warmup, no clip
step   10  loss 2.31  grad_norm 14.2
step   20  loss 2.05  grad_norm 31.7
step   40  loss 1.83  grad_norm 88.4     # encouraging... but grad-norm climbing
step   60  loss 9.41  grad_norm 8123.6   # the spike: lr overshot the bowl
step  100  loss 47.2  grad_norm 6.1e5
step  140  loss nan   grad_norm nan      # terminal
```

The signature is textbook §2: the loss drops for a few steps (because even a too-high rate makes progress on the easy, low-curvature directions first), the grad-norm climbs steadily as the optimizer starts overshooting the high-curvature directions, then at step 60 a step lands on the far wall of a sharp direction, the gradient there is enormous, the next step overshoots even more, and the multiplier $|1 - \eta L| > 1$ kicks in: geometric blow-up to `nan`.

It helps to see the run as a sequence of stations, because the failure is not instantaneous — it walks a recognizable path, and each station is a place you could have caught it.

![A timeline of a too-high learning-rate run walking from a flat warmup through a fast drop to a grad-norm spike that forks into recovery or a terminal NaN](/imgs/blogs/the-learning-rate-is-almost-always-the-problem-4.png)

Read the timeline left to right. The early flat phase (steps 0–50) looks innocent — the loss sits near chance because the model has not moved much yet. The fast drop (50–55) is the seductive part: the loss falls quickly and a tired engineer reads "it's learning" and walks away. The spike at step 60 is the actual event: the optimizer takes a step that overshoots a high-curvature direction, the loss leaps, and the grad-norm leaps with it to `8e3`. Then the *fork*: either a guardrail (gradient clipping, a lower steady-state rate) catches it and the run recovers, or the multiplier compounds and the run walks off the cliff to `nan` by step 140. The crucial insight is that the grad-norm spike at the fork is *visible before the loss is `nan`* — if you are logging it, you get a warning at step 60, not a corpse at step 140. The whole life of the run is a story about one number being too big.

The diagnostic confirmation took one LR-range test, which showed a knee around `1e-3` for this Adam-based finetune and a steep band around `2e-4`–`5e-4`. We picked `3e-4`, added a short linear warmup (more on why in §6), and added `clip_grad_norm_(..., 1.0)` as a guardrail. The fixed run:

```bash
# Fixed run: lr=3e-4, 200-step warmup, clip 1.0
step   10  loss 2.28  grad_norm 1.8      # warmup: small steps, bounded grads
step  100  loss 1.42  grad_norm 2.1
step  300  loss 0.71  grad_norm 1.9
step  600  loss 0.34  grad_norm 1.6      # smooth, no spike
final      val_acc 0.912                 # ships
```

| Metric | Buggy (`lr=1e-2`) | Fixed (`lr=3e-4` + warmup + clip) |
| --- | --- | --- |
| Loss at step 60 | 9.41 (spiking) | ~1.6 (descending) |
| Grad-norm at step 60 | 8,124 | ~2.0 |
| Outcome | `nan` at step 140 | converges, val acc 0.912 |
| Wasted GPU-hours | the whole run, repeatedly | none |

Three changes, but the learning rate is the one that mattered — warmup and clipping are guardrails that make a *correct* rate robust, not substitutes for a correct rate. Set `lr=1e-2` with warmup and clipping and you still diverge once warmup ends; the clip just delays it. **The fix is the rate.** This is the pattern to internalize: when you see the spike-grad-norm-then-`nan` signature, the first action is *lower the learning rate*, and the LR-range test tells you how far.

## 6. Warmup, schedules, and the LR–batch-size laws

A constant learning rate, even a well-chosen one, leaves performance on the table. Three additional ideas — warmup, schedules, and batch-size scaling — all follow from the same $2/L$ physics, and together they are how practitioners actually set the rate over a run.

### Why warmup stabilizes the early steps

Warmup means starting the learning rate near zero and linearly (or otherwise) ramping it up to the target over the first few hundred or few thousand steps. It looks like a hack; it is not. The reason it works is a direct consequence of the curvature picture.

At initialization, two things make the early steps dangerous. First, the loss landscape near a random init is often *sharper* — the curvature $L$ is larger — than it is later in training, so the stability bound $2/L$ is *smaller* early on. A rate that is safe at step 5,000 can be over the bound at step 5. Second, the gradient estimates are *high-variance* early: the model's predictions are near-random, the per-batch gradients are large and noisy, and adaptive optimizers like Adam have not yet built up reliable running estimates of the gradient scale (Adam's second-moment estimate $v_t$ starts at zero and is biased toward small values early, which *inflates* the effective step size precisely when the gradients are largest). Warmup keeps $\eta$ small while $L$ is large and the variance is high, then raises it as the landscape flattens and the gradient estimates stabilize. It is, literally, respecting $\eta < 2/L$ when $L$ is at its largest.

This is why warmup is essentially mandatory for transformers (the original Transformer used it, and every large-model recipe since does) and why "remove warmup and the loss spikes in the first 50 steps" is a recognizable bug. If your run spikes only at the very start and is fine afterward, you probably need warmup, not a lower steady-state rate.

```python
# Linear warmup then cosine decay — the workhorse LLM finetuning schedule.
from torch.optim.lr_scheduler import LambdaLR
import math

def warmup_cosine(optimizer, warmup_steps, total_steps, min_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)          # ramp 0 -> 1
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return min_ratio + (1 - min_ratio) * cos          # floor at min_ratio
    return LambdaLR(optimizer, lr_lambda)

# Hugging Face Trainer exposes this directly, no hand-rolling needed:
#   TrainingArguments(warmup_ratio=0.03, lr_scheduler_type="cosine", learning_rate=2e-5)
```

### What schedules buy you

A schedule decays the learning rate over training. The intuition, again from the bowl: early on you are far from the minimum and want big steps; late in training you are near the bottom of a noisy bowl and a large step just bounces you around the minimum (the oscillation mode from §1), raising the loss floor. Decaying $\eta$ toward zero lets the late steps settle into the minimum, lowering the final loss. The common choices:

- **Cosine decay** ramps down smoothly following a half-cosine from the peak to a small floor. It is the default for most modern training and finetuning because it spends a lot of steps at a moderately-high rate (good progress) and then eases in gently.
- **Linear decay** ramps straight down to zero (or a floor). Simpler, very common for LLM finetuning, near-equivalent to cosine in practice.
- **One-cycle** (Smith again) ramps *up* to a high peak then back down, pairing a rising LR with a falling momentum. The high mid-run rate acts as regularization (it keeps the optimizer out of sharp minima), and it can train faster — "super-convergence." It is excellent for from-scratch vision training with SGD; less common for LLM finetuning.

The schedule does not replace getting the *peak* rate right — it modulates around it. The LR-range test finds the peak; the schedule decides how to spend it.

Here is the comparison I use to pick a schedule, with the regime each one fits and the failure it prevents.

| Schedule | Shape | Best for | What it prevents |
| --- | --- | --- | --- |
| Constant | Flat at peak | Quick experiments, debugging | Nothing — leaves final loss on the table |
| Linear warmup + linear decay | Up then straight down | LLM finetuning (HF default) | Early spike (warmup) + high final floor (decay) |
| Linear warmup + cosine decay | Up then half-cosine down | Pretraining, most finetuning | Same, with a gentler late phase |
| One-cycle (Smith) | Up to a high peak, then down | From-scratch vision with SGD | Sharp-minimum overfit; enables fast training |
| Inverse-sqrt (Transformer) | Up then $1/\sqrt{t}$ decay | Transformer pretraining | Early instability at high model width |

The single most common schedule bug is *forgetting the schedule entirely* and training at a constant peak rate: the run trains fine but plateaus a little high because the late steps keep bouncing in the minimum. The second most common is *mismatching the schedule length to the run* — a cosine schedule configured for 10,000 steps but stopped at 3,000 never reaches its low-rate phase, so you get none of the late-training settling. Always set `total_steps` (or `num_training_steps` in Hugging Face) to the actual number of steps you will run, or the decay is a lie.

#### Worked example: warmup turns a first-step spike into a clean start

A team finetunes a 1.3B model with AdamW at `lr=2e-5`, no warmup. The loss starts at 1.9, spikes to 4.1 at step 3, and recovers by step 30 — a small "scar" at the very start. They shrug it off, but the scar correlates with a measurably worse final eval (about 1.5 points). The cause is the §6 mechanism: at step 0, Adam's second-moment estimate $v_t$ is near zero, so the bias-corrected denominator is small and the *effective* step is much larger than `2e-5` for the first few steps — large enough to overshoot the still-sharp early landscape. Adding a 3% linear warmup (`warmup_ratio=0.03`, here about 60 steps) keeps the rate near zero while $v_t$ fills in, the first-step spike vanishes entirely, the loss descends monotonically from step 1, and the final eval recovers the 1.5 points. The fix was not a smaller peak rate — `2e-5` was correct — it was respecting the bound while the curvature was largest. This is why "remove warmup and the first few steps spike" is a recognizable, self-inflicted bug rather than a mystery.

### The LR–batch-size scaling laws

The last piece: when you change the batch size, you must change the learning rate, and getting this wrong is a sneaky source of divergence ("it diverged after I went to 8 GPUs"). The reason is variance. A larger batch gives a *lower-variance* gradient estimate — the average of more examples is closer to the true gradient — so you can take a *larger* step safely. Two rules of thumb formalize this:

- **Linear scaling rule**: multiply the learning rate by the same factor you multiply the batch size. Batch $256 \to 1024$ (4×) means $\eta \to 4\eta$. This comes from a simple argument (Goyal et al., "Accurate, Large Minibatch SGD," 2017): $k$ steps of small-batch SGD make approximately the same weight update as one step of $k$×-larger-batch SGD *if* the large-batch rate is $k$ times larger. It holds well up to a point and breaks at very large batches, which is why Goyal et al. *also* needed warmup to make linear scaling work at batch 8192.
- **Square-root scaling rule**: multiply the learning rate by $\sqrt{k}$. This comes from keeping the *variance of the update* constant: the gradient noise scales as $1/\sqrt{\text{batch}}$, so a $\sqrt{k}$ rate increase keeps the noise-to-signal ratio of the step fixed. It is the more conservative rule and often more stable for adaptive optimizers.

![A grid contrasting the linear and square-root learning-rate scaling rules across a base batch and a four-times-larger batch, with the rescaled target rates in each cell](/imgs/blogs/the-learning-rate-is-almost-always-the-problem-8.png)

In practice: when you change the batch size, **rescale the LR with the linear rule as your aggressive target and the sqrt rule as your safe target, then verify with a short run or an LR-range test.** Do not keep the old rate — it will be too small for a bigger batch (crawl) or, if you also raised it carelessly, too big (spike). For the deeper theory of how the optimal rate and batch size co-vary, the scaling-laws literature is the place to go; see the [scaling-laws posts](/blog/machine-learning/scaling-laws) for the LR–batch-size relationship at scale.

#### Worked example: the batch-size change that diverged

A team scales a training run from 1 GPU (batch 256, `lr=3e-4`, stable) to 4 GPUs with DDP (effective batch 1024) and keeps `lr=3e-4`. The run trains, but slowly — the loss curve is the crawl from §3, because the now-larger batch makes the *same* rate effectively too small relative to the lower-variance gradient. They "fix" it by jumping to `lr=3e-3` (10×) and it diverges in 40 steps (the spike from §1). The right move was the scaling rules: linear gives $3e\text{-}4 \times 4 = 1.2e\text{-}3$ (aggressive), sqrt gives $3e\text{-}4 \times 2 = 6e\text{-}4$ (safe). They tested `6e-4` with warmup, it was stable and fast, and a short LR-range test confirmed the new knee sat around `2e-3`. The lesson: a batch-size change is a *learning-rate change in disguise*, and ignoring that is one of the most common multi-GPU "regressions."

## 7. The critical finetuning case: a too-high rate destroys pretrained features

This is the most important section for anyone working with pretrained models, which in 2026 is almost everyone. The single most expensive learning-rate mistake of the LLM era is using a from-scratch learning rate to finetune a pretrained model.

### Why pretrained models need a tiny rate

A from-scratch model starts at a random point in parameter space, far from any minimum, and *wants* big steps to get anywhere. A pretrained model starts at a point that is *already near the bottom of a good minimum* on its pretraining distribution — that is what "pretrained" means. Its weights encode features (syntax, world knowledge, visual primitives) that took enormous compute to learn. When you finetune, you want to *nudge* those weights to specialize them, not catapult them across the landscape.

Now apply the bowl picture. The model sits near $x = 0$ (the pretrained minimum). A from-scratch rate like `1e-3` is sized for a model that is *far* from the minimum, where big steps are productive. Applied to a model that is *already at* the minimum, that same big step throws it a huge distance away — to a region where the pretrained features no longer apply. The loss, which was low at the pretrained point, *spikes upward* on the first step, because you have just walked off the good solution. This is **catastrophic forgetting**, and on a loss curve it has an unmistakable signature: the loss starts low (the pretrained model is already decent), spikes up in the first handful of steps, and the model's *base capabilities collapse* — a chat model starts producing garbage, a vision backbone forgets its features, a classifier regresses to chance on held-out base classes.

The fix is to make the steps small enough to nudge rather than catapult: a finetuning learning rate **ten to one hundred times smaller** than from-scratch. For LLMs, the consensus range is `1e-5` to `2e-5` for full finetuning (and somewhat higher, `1e-4`–`3e-4`, for LoRA, because LoRA's low-rank update has a different effective scale). For finetuning a vision backbone, `1e-4` to `1e-3` with the backbone often at a *lower* rate than the new head (discriminative/per-layer learning rates). The principle is identical across modalities: **the closer you start to a good minimum, the smaller the step must be.**

### Seeing the forgetting before it costs you a run

![A before-and-after figure contrasting a finetune at learning rate one-thousandth that spikes and forgets against the same finetune at two-hundred-thousandths that adapts cleanly and keeps base skills](/imgs/blogs/the-learning-rate-is-almost-always-the-problem-7.png)

The diagnostic is to **evaluate the base task before step 1 and watch the early loss.** If the pretrained model is good on a base benchmark, you start finetuning, and the loss spikes while the base benchmark craters, you have a too-high rate. Here is a minimal guard you can put on any Hugging Face finetune to catch it at step 10 instead of after a wasted run:

```python
from transformers import TrainerCallback

class ForgettingGuard(TrainerCallback):
    """Catch a too-high finetune LR early: if the very first training losses
    spike above the pretrained baseline, the rate is destroying features."""
    def __init__(self, baseline_loss, spike_factor=2.0, watch_steps=20):
        self.baseline = baseline_loss      # eval loss of the pretrained model
        self.spike_factor = spike_factor
        self.watch_steps = watch_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        if state.global_step <= self.watch_steps:
            if logs["loss"] > self.spike_factor * self.baseline:
                raise RuntimeError(
                    f"Loss {logs['loss']:.3f} spiked above "
                    f"{self.spike_factor}x baseline {self.baseline:.3f} "
                    f"at step {state.global_step}. LR likely too high — "
                    f"try lr={args.learning_rate / 10:.1e}."
                )

# Measure the pretrained baseline first, then guard the finetune:
#   baseline = trainer.evaluate()["eval_loss"]
#   trainer.add_callback(ForgettingGuard(baseline_loss=baseline))
#   trainer.args.learning_rate = 2e-5   # not 1e-3
```

The before-and-after on a real-ish finetune of a 7B chat model on a small instruction dataset:

| Metric | `lr=1e-3` (from-scratch rate) | `lr=2e-5` (50× smaller) |
| --- | --- | --- |
| Loss at step 1 | 0.41 → 6.0 (spike) | 0.41 → 0.39 (smooth) |
| Base benchmark (MMLU-style) after 1 epoch | dropped 41 points | held within 1 point |
| Task metric | garbage / unusable | +6 points on the target task |
| Generations | repetitive / incoherent | coherent, on-task |

The numbers above are illustrative of the *pattern* (the precise drop depends on the model and data), but the direction and order of magnitude are exactly what you see: a 50× rate cut turns a destroyed model into a working one. The mechanism is the bowl: small steps near a good minimum *adapt*; big steps *forget*. For the full finetuning recipe — epochs (1–3, not 10), packing vs padding, the eval that catches regression — see [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it). This post's job is just the rate, and the rate is most of the battle.

## 8. Per-layer rates, optimizers, and the rate across modalities

One global learning rate is the default, but it is rarely the optimal choice, and the refinements all follow from the same curvature picture. This section is the practitioner layer: how the rate interacts with the optimizer you chose, why different layers want different rates, and how the whole story lands the same way in vision, language, tabular, and speech.

### Discriminative (per-layer) learning rates

When you finetune a pretrained model, the layers are not equal. The early layers learned general features (edges and textures in vision, syntax in language) that transfer almost unchanged to your task and should barely move. The later layers learned task-specific features that need to change more, and any *new* head you bolt on (a fresh classifier) starts from random init and needs a *from-scratch*-sized rate to learn at all. Applying one rate to all of them is a compromise that is either too high for the early layers (you wreck transferable features) or too low for the new head (it never learns).

The fix is **discriminative learning rates**: a smaller rate for early layers, a larger rate for the head, often geometrically spaced. This is standard practice in transfer learning (the ULMFiT recipe popularized it for NLP, and it is routine for vision-backbone finetuning). In PyTorch you express it through optimizer parameter groups:

```python
import torch

def layerwise_lr_groups(model, base_lr=1e-4, head_lr=1e-3, decay=0.9):
    """Geometrically decaying LR from the head down to the early layers.
    Early (transferable) layers get a tiny rate; the new head gets a big one."""
    groups = []
    # Assume model.backbone is a list/Sequential of blocks, shallow -> deep.
    blocks = list(model.backbone)
    n = len(blocks)
    for i, block in enumerate(blocks):
        # Deeper blocks get a larger rate; shallowest gets base_lr * decay**(n-1).
        lr = base_lr * (decay ** (n - 1 - i))
        groups.append({"params": block.parameters(), "lr": lr})
    # The freshly-initialized head learns from scratch: the largest rate.
    groups.append({"params": model.head.parameters(), "lr": head_lr})
    return groups

optimizer = torch.optim.AdamW(layerwise_lr_groups(model), weight_decay=0.01)
```

The same idea, in the extreme, is *freezing*: the early layers get a rate of exactly zero for the first few epochs (a "freeze schedule"), then unfreeze. Freezing is just a per-layer rate of zero with a schedule, and it is the safest way to avoid wrecking transferable features while the head warms up. The destroy-the-features failure from §7 is, at the layer level, "you applied the head's rate to the backbone."

### The rate is optimizer-relative

A learning rate is meaningless without naming the optimizer, because each optimizer rescales the gradient differently before the rate touches it. The same numerical rate that is correct for one is wildly wrong for another.

| Optimizer | What it does to the step | Typical good rate (transformer finetune) | Sensitivity to the rate |
| --- | --- | --- | --- |
| SGD | Raw gradient times rate | `1e-1` to `1e-2` (with momentum) | High — the $2/L$ bound bites directly |
| SGD + momentum | Smoothed gradient, can overshoot more | slightly below plain SGD | High, plus momentum can amplify a spike |
| Adam / AdamW | Per-coordinate normalized step | `1e-3` to `2e-5` | Lower — adaptive scaling forgives more |
| Adafactor / Lion | Memory-light adaptive variants | model-specific, often `~1e-4` | Varies; re-tune, do not port Adam's rate |

The two practical rules: (1) **never port a learning rate across optimizers** — a rate tuned for SGD will be enormously too large for Adam relative to Adam's effective step, and vice versa; re-run the LR-range test when you switch. (2) **AdamW's weight decay interacts with the rate** — AdamW decouples weight decay from the gradient step (unlike Adam's L2), so changing the rate does not change the decay strength, which is why AdamW is the finetuning default; with plain Adam, retuning the rate silently retunes regularization, a subtle source of "it was better last week."

### The same physics, every modality

The reason this post claims to span vision, language, tabular, and speech is that the $2/L$ bound does not know what data you are training on. The *symptoms* and *good rates* differ only because the curvature and the pretraining situation differ:

- **Vision.** From-scratch CNNs with SGD live around `1e-1` (one-cycle shines here); finetuning a pretrained backbone wants `1e-4`–`1e-3` with discriminative rates and often a freeze schedule. The forgetting signature on a finetune is identical to the LLM one: spike the backbone rate and the model loses its features.
- **Language.** Adam/AdamW dominates; from-scratch pretraining uses warmup + a peak around `1e-3`–`6e-4` with cosine decay; finetuning uses `1e-5`–`2e-5` (full) or `1e-4`–`3e-4` (LoRA). Warmup is mandatory.
- **Tabular neural nets.** Adam around `1e-3`; the most common bug is the §3 crawl from an over-cautious `1e-5`, because tabular practitioners often inherit conservative defaults from gradient-boosting habits.
- **Speech.** Whisper/wav2vec2 finetuning uses tiny rates (`1e-5`–`1e-4`) for the same pretrained-minimum reason as LLMs; the extra wrinkle is that the feature front-end can have very different curvature from the transformer body, which is a per-layer-rate situation in disguise.

The unifying claim is worth stating plainly: across every one of these, the workflow is the same — *recognize the symptom, run the LR-range test, pick below the knee, finetune at 10–100× smaller, warm up, and confirm with the 10× test.* The numbers move; the method does not.

## 9. The bisection: is it really the learning rate?

The discipline of this series is to *bisect to a suspect and confirm with a test* before you touch code. The learning rate has the highest prior inside the optimization bucket, but you should still confirm rather than assume. Here is the decision flow.

**Step 1 — Look at the curve shape.** Spike + grad-norm spike → too high. Slow monotone crawl that underfits → too low. First-step spike on a finetune → from-scratch rate on a pretrained model. Sawtooth, plateaus high → near the edge. This already names the suspect (§1).

**Step 2 — The cheapest confirming test: change the rate by 10× and re-run a few hundred steps.** If the symptom is too-high, lowering the LR 10× should make the spike disappear or move much later. If it is too-low, raising it 10× should accelerate the descent visibly. This is a two-minute test and it confirms or clears the learning rate decisively. If 10× in either direction does *nothing* to the symptom, the learning rate is probably *not* your bug — go look elsewhere.

**Step 3 — The LR-range test (§4)** if step 2 is ambiguous or you want a principled rate rather than a binary lower/raise. It gives you the knee and the steep band in 300 steps.

**Step 4 — Overlay the grad-norm** if the curve alone is ambiguous. A loss spike *with* a grad-norm spike is an optimization/numerics event (lower the rate, clip). A loss spike *without* a grad-norm spike is more likely a single bad batch with a large per-example loss but a bounded gradient — a *data* problem, not a learning-rate one. This single overlay disambiguates the two most-confused signatures, and it is why the [instrumentation post](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) insists on logging grad-norm beside loss.

The stress tests — the "what if it's not the LR" questions — are how you avoid tunnel vision:

- **What if it's data, not optimization?** A spike with a *quiet* grad-norm, or a spike that recurs at a fixed step *period* (every N steps, where N is batches-per-epoch), is a recurring bad batch, not the learning rate. Lowering the LR will not fix it; finding and fixing the batch will. See [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic) for the periodicity tell.
- **What at fp16?** Under fp16 mixed precision, gradients can *underflow to zero* (the representable floor is about $6 \times 10^{-5}$), which makes a too-low *effective* rate even when the nominal rate is fine, and overflow can produce `inf` gradients that look like a too-high-rate spike. Loss scaling (GradScaler) addresses underflow. A run that diverges only at fp16 and is fine at bf16 or fp32 is a *numerics* bug wearing a learning-rate costume; bf16's wider range usually fixes it. This is its own deep topic; the LR is not the cause there.
- **What if the batch is tiny?** A very small batch has a high-variance gradient, which raises the *effective* curvature the optimizer sees and forces a smaller stable rate — the same nominal LR that is fine at batch 256 can spike at batch 8. This is the batch-size law from §6 running in reverse.
- **What if it only diverges on multi-GPU?** Usually the effective batch grew (so the rate is now mis-scaled, §6) or gradient synchronization is broken (a *systems* bug). Check the effective batch and the scaling rule first; if the rate is correctly scaled and it still diverges only on multi-GPU, the bug has moved to systems, not optimization.

The point of the stress tests is calibration: the learning rate is the *highest-prior* cause of these symptoms, not the *only* one, and a disciplined debugger confirms with the 10× test before committing. When the 10× test moves the symptom, you have your bug. When it doesn't, you have just saved yourself from "fixing" the rate for three days while the real bug sits in the data loader.

## 10. The full failure-mode map

Pulling §1–§8 together, here is the complete map of learning-rate failure modes with their mechanism, signature, confirming test, and fix — the table to screenshot and keep.

![A matrix mapping four learning-rate symptoms to their cause, a confirming test, and a fix, from spike-then-NaN through finetune forgetting](/imgs/blogs/the-learning-rate-is-almost-always-the-problem-5.png)

| Failure mode | Mechanism | Loss signature | Confirm | Fix |
| --- | --- | --- | --- | --- |
| Spike → `nan` | $\eta > 2/L$, error grows geometrically | Sharp upward spike, grad-norm spikes with it, then `nan` | Lower LR 10×; spike vanishes | Lower LR to below the knee, add warmup + clip |
| Slow crawl | $\eta \ll 1/L$, multiplier ≈ 1, underfits | Monotone, far-too-slow descent, plateaus high on *train* | Raise LR 10×; descent accelerates | Raise LR toward the steep band |
| Oscillation | $\eta$ near $2/L$, bounces the bowl | Sawtooth, no convergence, higher floor | Halve LR; floor drops | Halve LR or add decay schedule |
| Finetune forgetting | From-scratch rate near a pretrained minimum | First-step spike up, base capability collapses | Eval base task before/after step 1 | Drop LR 10–100× (`1e-5`–`2e-5` for LLMs) |
| Batch-scaling divergence | LR not rescaled for new (effective) batch | Spike after batch/GPU change, or crawl | Apply linear/sqrt rule; stability returns | Rescale LR with batch size, re-warmup |
| fp16-only divergence | Grad over/underflow, not the rate itself | Diverges at fp16, clean at bf16/fp32 | Switch to bf16; spike gone | Loss scaling or bf16, *not* a rate change |

The last row is the trap that keeps people honest: not every learning-rate-*shaped* symptom is a learning-rate *bug*. The whole point of the confirming tests is to separate "the rate is wrong" from "something else looks like the rate is wrong."

Notice the structure of the whole table: the *mechanism* column is always a statement about the multiplier $|1 - \eta L|$ and the bound $\eta < 2/L$, the *signature* column is always something you can read off a loss curve plus a grad-norm overlay, the *confirm* column is almost always a single cheap re-run (10× the rate, halve it, apply a scaling rule, switch precision), and the *fix* column is almost always a change to one number plus a guardrail. That uniformity is not an accident of presentation — it is the reason the learning rate is the highest-leverage thing to check first. A class of bug whose entire diagnosis fits in a five-column row, whose confirming test costs a few hundred steps, and whose fix is one hyperparameter is, by definition, the cheapest possible win in a debugging session. Run the LR check first not because it is always the bug, but because when it *is* the bug it is the fastest one in the whole six-places taxonomy to localize and kill — and when it is *not*, you have cleared the most likely suspect in two minutes and earned the right to look elsewhere with a clear conscience.

## Case studies and real signatures

These are well-known patterns and results; where I give a number I cite it or flag it as approximate.

**Transformer warmup is not optional (Vaswani et al., 2017).** The original "Attention Is All You Need" used a learning-rate schedule with a warmup phase (the rate rises for `warmup_steps`, then decays as the inverse square root of the step). Subsequent work showed that *removing* warmup makes large transformers diverge early or land in worse minima — a direct demonstration of the §6 argument that the early-training curvature is large and warmup respects the $2/L$ bound when $L$ is largest. The practical takeaway has held for nearly a decade: warmup is part of the recipe, not a tunable luxury, for attention-based models. If your transformer finetune spikes only in the first 50 steps, the first thing to add is warmup.

**Linear scaling needs warmup at large batch (Goyal et al., 2017).** "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" trained ResNet-50 on ImageNet at a batch size of 8192 by applying the linear scaling rule — raising the learning rate by the same factor as the batch — *and* adding a gradual warmup, because the linear rule alone made the very large early steps unstable. This is the §6 batch-size law and the §6 warmup argument working together, validated at scale: they matched small-batch accuracy with an 8192-batch run, but only with both ingredients. It is the canonical evidence that a batch-size change is a learning-rate change in disguise.

**The LR-range test / one-cycle (Smith, 2017 and 2018).** Leslie Smith's "Cyclical Learning Rates for Training Neural Networks" introduced the LR-range test as a cheap way to find the usable rate band, and the follow-up "super-convergence" work showed that one-cycle schedules — ramping the rate up to a high peak and back down — can train some networks dramatically faster than a fixed rate, because the high mid-run rate acts as regularization. The range test is the most underused diagnostic in this whole post; the fact that you can *find* a good rate in 300 steps instead of grid-searching whole runs is the practical heart of it.

**The `3e-4` folklore.** The half-joke that `3e-4` is "the best learning rate for Adam" is folklore, but §2 explains why it is *useful* folklore: for a lot of normalized, well-conditioned problems the largest curvature lands where a few times $10^{-4}$ sits comfortably under $2/L$ for Adam's effective per-coordinate steps. It is a fine *starting guess* to then refine with the range test — not a law, but not a coincidence either.

**The finetuning-rate collapse.** Across countless LLM finetuning post-mortems, the same story repeats: someone uses `1e-3` or `5e-4` on a pretrained model, the loss spikes in the first epoch, the model's base evals crater, and the output is garbage. Dropping to `1e-5`–`2e-5` fixes it. The Hugging Face and `trl` defaults reflect this hard-won consensus (`SFTTrainer` and most recipes default to `~2e-5` for full finetuning). The mechanism is §7: a from-scratch rate catapults a pretrained model off its minimum. This is, in my experience, the single most common learning-rate bug in applied ML today.

## When this is (and isn't) your bug

Be decisive about when to stop blaming the learning rate.

**It IS the learning rate when:** the loss spikes sharply *and* the grad-norm spikes with it (too high); the loss crawls monotonically and underfits the *training* set (too low); a finetune's loss spikes in the first few steps and the base capability collapses (from-scratch rate on a pretrained model); the symptom appeared right after a batch-size or GPU-count change (mis-scaled rate); and — the decisive confirmation — *changing the rate by 10× moves the symptom.*

**It is NOT the learning rate when:** the loss spike has a *quiet* grad-norm (a bad batch — data, not optimization); the spike recurs at a fixed step *period* matching batches-per-epoch (a recurring corrupt batch — data); the run diverges only at fp16 and is clean at bf16/fp32 (numerics — loss scaling or bf16, not the rate); the loss is flat at exactly chance with a *zero* grad-norm (no gradient is flowing — frozen params or `requires_grad=False`, a model-code bug, not the rate); or the train loss is fine and only *val* is bad (overfitting or a leak — evaluation, not optimization). And the cleanest negative test of all: **if changing the learning rate 10× in either direction does nothing to the symptom, the learning rate is not your bug.** Stop tuning it and bisect elsewhere — start from the [taxonomy and decision tree](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).

The grad-norm overlay is your single best tie-breaker. Loss-spike-*with*-grad-spike is optimization (lower the rate). Loss-spike-*without*-grad-spike is data (find the batch). Flat-loss-with-*zero*-grad is model code (find the frozen layer). One instrument, three different bugs, decisively separated — which is exactly why the series tells you to log it.

## Key takeaways

- **The learning rate is the highest-prior single cause of a misbehaving run.** When optimization looks wrong, check the rate first; it is the cheapest dial and roughly half the time it is the whole bug.
- **Too high spikes; too low crawls; the edge oscillates.** A sharp loss spike with a simultaneous grad-norm spike is too high. A monotone, far-too-slow descent that underfits the *training* set is too low. A sawtooth that plateaus high sits near the edge.
- **The stability bound is $\eta < 2/L$.** On a quadratic bowl with curvature $L$ (the largest Hessian eigenvalue), gradient descent multiplies the distance to the minimum by $|1 - \eta L|$ each step; above $2/L$ that factor exceeds one and the loss blows up geometrically — a spike, then `nan`.
- **Run the LR-range test, don't guess.** Ramp the rate exponentially over ~300 steps, plot loss vs LR, and pick a rate one notch below the divergence knee. Thirty seconds buys you a defensible rate instead of a superstition.
- **Warmup respects $2/L$ when $L$ is largest.** Early training has higher curvature and noisier gradients, so the stable rate is smaller; warmup keeps the rate small while the landscape is sharp, then raises it. For transformers it is mandatory, not optional.
- **A batch-size change is a learning-rate change in disguise.** Bigger batch → lower-variance gradient → you can (must) raise the rate. Use the linear rule as the aggressive target and the sqrt rule as the safe one, then verify.
- **Finetuning LR is 10–100× smaller than from-scratch.** A pretrained model sits near a good minimum; a from-scratch rate catapults it off and the loss spikes upward as it forgets. Use `1e-5`–`2e-5` for full LLM finetuning, not `1e-3`.
- **Confirm with the 10× test before committing.** Change the rate by 10× and re-run a few hundred steps. If the symptom moves, it is the rate. If it doesn't, it isn't — go bisect elsewhere.
- **Grad-norm is the tie-breaker.** Loss-spike-with-grad-spike is optimization (the rate); loss-spike-without-grad-spike is a bad batch (data); flat-loss-with-zero-grad is a frozen layer (model code). Log it beside the loss.

## Further reading

- **Leslie N. Smith, "Cyclical Learning Rates for Training Neural Networks" (2017)** — the LR-range test and cyclical schedules; the source of the LR finder. Also Smith & Topin, "Super-Convergence" (2018), for one-cycle.
- **Priya Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)** — the linear scaling rule and why large-batch training needs gradual warmup.
- **Ashish Vaswani et al., "Attention Is All You Need" (2017)** — the transformer warmup-then-inverse-sqrt schedule; why attention models need warmup.
- **Diederik P. Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization" (2014)** — the adaptive optimizer whose per-coordinate scaling makes it more forgiving of a guessed rate, and the bias-correction that interacts with early-step instability.
- **PyTorch docs** — `torch.optim.lr_scheduler` (warmup, cosine, one-cycle), `torch.nn.utils.clip_grad_norm_`, and the `torch_lr_finder` library and PyTorch Lightning `Tuner.lr_find` for the automated range test.
- **Within this series** — start from [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the symptom→suspect→test→fix frame; pair this with [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic) and [instrumenting a training run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log); go deeper on the adjacent failure modes in [gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing) and [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence); apply the finetuning rate in [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it); and when you have localized the bug, return to the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

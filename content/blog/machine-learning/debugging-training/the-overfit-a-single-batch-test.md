---
title: "The Overfit-a-Single-Batch Test: The Highest-Leverage Sanity Check in ML"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Before you burn GPU-hours on a full run, prove your model can memorize a handful of examples — the one test that rules out a whole class of silent training bugs in 30 seconds."
tags:
  [
    "debugging",
    "model-training",
    "pytorch",
    "gradient-flow",
    "finetuning",
    "deep-learning",
    "sanity-checks",
    "optimization",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/the-overfit-a-single-batch-test-1.png"
---

A team I worked with once kicked off a four-day, eight-GPU finetune of an image classifier. The dashboard looked healthy: loss ticked down from 2.31 to 2.28 over the first thousand steps, the GPUs were pinned at 95% utilization, throughput was textbook. Three days in, validation accuracy was still hovering around 10% on a ten-class problem — exactly chance. Someone had, two weeks earlier, run a feature-freezing experiment and set `requires_grad = False` on the classifier head, then never turned it back on. The backbone was learning slightly better features; the head that turns those features into class scores was a brick. The run was never going to work, and we had paid for roughly 750 GPU-hours to discover it.

There is a thirty-second test that would have caught this before the run started. Take one fixed batch — 8 to 16 examples — turn off everything that adds noise (shuffling, augmentation, dropout, weight decay), and train on that single batch over and over. If your model, loss, optimizer, and data path are all wired correctly, the loss should crater toward zero within a couple hundred steps. The model has ten million parameters and you are asking it to memorize sixteen examples; this should be trivial. If it *cannot* do that — if the loss sits stubbornly at 2.30 no matter how long you train — then something is fundamentally broken, and no amount of data, compute, or patience will save the full run. In our case the loss would have sat flat at exactly $\ln(10) \approx 2.3026$, the cross-entropy of a uniform guess, screaming that the head was dead. The figure below shows the five things that all have to work for that test to pass, which is exactly why a failure is so diagnostic.

![Five stacked layers — data path, model capacity, gradient flow, optimizer step, and loss wired to labels — that must all work for one batch to overfit](/imgs/blogs/the-overfit-a-single-batch-test-1.png)

This is the **overfit-a-single-batch test**, and I will argue it is the single highest-leverage sanity check in machine learning. It costs seconds, requires no extra infrastructure, and a pass rules out an entire category of bugs at once: insufficient capacity, dead gradients, a frozen model, a loss that is not connected to the labels, an optimizer that does not step, a learning rate of zero. A fail, conversely, localizes you to a tiny set of mechanical faults you can check one by one. In the language of this series, it is the purest form of **make-it-fail-small**: shrink the problem until either it works (and you have learned the model is fine) or it breaks in a way you can read. By the end of this post you will have a clean, reusable `overfit_one_batch()` loop you can paste into any project, you will know exactly how to read a pass and a fail, you will be able to climb the full debugging ladder (one example, one class, one feature, one GPU, one step), and — critically — you will know the false-pass traps that make a broken run *look* like it overfit when it did not.

This sits at the very top of the [training-debugging taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs): before you ask *which of the six places* (data, optimization, model code, numerics, systems, evaluation) hides your bug, you run this test, because it cleaves that space in half on the first try.

## 1. The symptom: a run that looks fine and learns nothing

Let us be precise about the failure mode this test is built to catch, because it is the most demoralizing one in the field: the run that *looks* healthy and *is* dead.

A genuinely healthy run gives you obvious feedback. The loss falls, the metric climbs, you ship. A genuinely crashed run also gives you feedback — a stack trace, a NaN, an out-of-memory error at step 3. Those are annoying but honest. The dangerous case is the run that does neither: it executes, it consumes resources, the numbers move *just enough* to look alive, and yet the model is not learning the task. The classifier head is frozen, or the loss is computed against the wrong tensor, or the learning rate is 0, or the labels are all the same class because of a collation bug. The optimizer dutifully takes steps that accomplish nothing, the loss drifts on noise, and you do not find out for hours or days.

What ties all of these together is that they would all *fail the same simple test*. None of them can drive the loss to zero on sixteen fixed examples, because in every case the path from "model output" to "the right answer" is broken somewhere. The overfit test does not care *which* of those is broken — it just refuses to converge, and a refusal to memorize sixteen points is a signal you can act on immediately.

Here is the running example we will debug throughout this post. We have a ResNet-18 with a ten-class head, finetuning on a small image dataset. The dashboard reads:

| Instrument | Reading | What it seems to say |
| --- | --- | --- |
| Train loss @ step 1000 | 2.28 (from 2.31) | "It's learning, slowly" |
| GPU utilization | 95% | "Compute is healthy" |
| Throughput | 1,100 img/s | "Data pipeline is fine" |
| Val accuracy @ epoch 3 | 10.4% | "...still at chance?" |
| Gradient norm (global) | 1.1e-3 | "Small but nonzero" |

Everything looks like a slow-but-working run *except* the validation accuracy, which is pinned at chance. A junior engineer reads this table and concludes "the learning rate is too low" or "we need more epochs." A disciplined debugger reads it and concludes "before I touch anything, can this model even memorize one batch?" — because if it cannot, the LR and the epoch count are irrelevant. We will run that test in Section 4 and watch it crack the case open.

## 2. The science: why a model *should* trivially memorize a tiny batch

The reason this test is so powerful is that it rests on a hard mathematical fact: a sufficiently overparameterized model can fit *any* labeling of a small dataset, including a random one. If your model cannot fit sixteen examples, it is not because the task is hard — it is because something is preventing optimization from working at all.

### 2.1 The capacity / interpolation argument

Consider a model with $N$ trainable parameters trying to fit $K$ labeled examples. When $N \gg K$ — and for a modern network, $N$ is millions to billions while $K$ for this test is 8 to 16 — the model is wildly *overparameterized* relative to the data. In that regime the loss surface generically contains a continuum of parameter settings that drive the training loss to its global minimum, a phenomenon called **interpolation**: the model has so many free knobs that it can thread a function exactly through every training point.

The cleanest demonstration of this is the classic *Understanding deep learning requires rethinking generalization* result (Zhang et al., 2017): standard CNNs can fit ImageNet with the labels *randomly shuffled*, reaching essentially zero training error. If a network can memorize a million images with meaningless labels, memorizing sixteen real ones is not in question. The only thing that can stop it is a broken optimization path, which is exactly what this test probes.

A back-of-the-envelope version makes the point concrete. A ResNet-18 has about $1.1 \times 10^7$ parameters. The final linear layer alone maps a 512-dimensional feature to 10 logits, which is $512 \times 10 + 10 = 5{,}130$ parameters — and a single linear layer with 5,130 free parameters can perfectly separate 16 points in 512-dimensional feature space with room to spare, because 16 points in a space of dimension 512 are almost surely linearly separable for *any* labeling. (In general, $d+1$ points in general position in $\mathbb{R}^d$ can be shattered by a linear classifier; here $16 \ll 513$.) So even if the entire backbone were frozen and only the head trained, the head *should* still be able to fit the batch — unless the head itself is the thing that is frozen.

It is worth making the linear-separability claim rigorous, because it is the load-bearing piece of the argument. A linear classifier with weight matrix $W$ and bias $b$ produces logits $z_i = W \phi(x_i) + b$ for the feature $\phi(x_i)$ of example $i$. Cross-entropy is minimized (driven toward zero) when, for each example, the logit of the true class exceeds every other logit by an ever-growing margin, i.e., $z_{i,y_i} - \max_{c \ne y_i} z_{i,c} \to \infty$. Whether that is achievable is purely a question of whether the points $\{\phi(x_i)\}$ are *linearly separable* by class. For $K$ points in $\mathbb{R}^d$ in general position, separability for an arbitrary labeling is guaranteed whenever $K \le d + 1$ — and with $K = 16$ and $d = 512$, we have enormous slack. The features need not even be in general position for this to almost surely hold; random high-dimensional features are separable with overwhelming probability. So the head alone, with a working gradient, *must* be able to drive the batch's cross-entropy arbitrarily low. If it does not, the gradient is not reaching it. This is not a heuristic — it is a counting argument, and it is why a failed overfit test is a *proof* that something mechanical is broken rather than a hint that the task is hard.

There is a second, deeper reason the optimization actually *finds* one of these zero-loss solutions rather than getting stuck. On a separable batch, gradient descent on cross-entropy does not have spurious local minima that trap it above zero loss — the loss is convex in the logits, and even through the nonconvex backbone, the overparameterized regime tends to produce loss landscapes where gradient descent reaches near-global minima for the training set. You do not need to trust the full theory here; you only need the empirical fact (Zhang et al., below) that it works in practice on far larger and harder memorization problems than sixteen examples. The takeaway for debugging: *the optimizer's job on this test is easy*, so if it fails, blame the wiring, not the optimizer's ability to find a solution.

### 2.2 What "loss to ~0" actually means, per loss type

"Drive the loss to zero" is shorthand, and the exact floor depends on the loss. Knowing the real floor matters, because the threshold you assert against has to be below it.

**Cross-entropy (classification).** For a single example with true class $y$, cross-entropy is $-\log p_y$, where $p_y$ is the predicted probability of the correct class. As the model becomes confident, $p_y \to 1$ and $-\log p_y \to 0$. There is no hard zero — softmax never outputs an exact 1 — but the loss decays roughly geometrically as logits grow. A few concrete values:

| $p_y$ (prob of correct class) | Cross-entropy $-\ln p_y$ |
| --- | --- |
| 0.10 (chance, 10 classes) | 2.3026 |
| 0.50 | 0.6931 |
| 0.90 | 0.1054 |
| 0.99 | 0.0101 |
| 0.999 | 0.0010 |
| 0.9999 | 0.0001 |

So when I say "loss to ~0" for a 10-class problem, I mean it should fall from $\ln(10) = 2.3026$ (the value of a uniform guess) down to something like $10^{-3}$ or $10^{-4}$. A reasonable assertion threshold is **loss < 0.01**, which corresponds to the model being ~99% confident on every example. If label smoothing is on, the floor is higher (smoothing deliberately prevents $p_y \to 1$), so either turn smoothing off for the test or raise the threshold accordingly.

**Mean squared error (regression).** MSE has a true zero: if the model can output the exact target, the loss is 0. For a tiny batch the floor should be machine-epsilon small, on the order of $10^{-6}$ or below in fp32. A regression overfit test that floors at, say, 0.4 is telling you the model literally cannot represent the targets — a scaling or output-head bug.

**CTC and sequence losses.** These have task-dependent floors and length constraints (a CTC loss returns `inf` if the input is shorter than the target, which is itself a bug signature). For the overfit test, the principle is unchanged: a tiny fixed batch should reach the loss's natural floor; if it plateaus far above that floor, the path is broken. (We cover CTC's specific traps in [debugging CTC and alignment](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) via the taxonomy.)

The key number to memorize: for a $C$-class cross-entropy problem, **a loss stuck at $\ln(C)$ is the model guessing uniformly**. $\ln(2) = 0.693$, $\ln(10) = 2.303$, $\ln(1000) = 6.908$. If your overfit test parks at exactly that value, the model is producing constant or random logits — gradients are not reaching the part of the network that assigns class scores.

### 2.3 How many steps and what learning rate the test needs

A subtlety that trips people up: the overfit test can "fail" simply because you did not give it enough steps, or because the learning rate was too small to make visible progress in the steps you allowed. Knowing roughly how fast convergence *should* be saves you from a false alarm.

The rule of thumb is that a clean overfit-one-batch run with a reasonable learning rate (Adam at $10^{-3}$ is a good default for this test) should show *obvious* loss decrease within a few dozen steps and reach the floor within a few hundred. If the loss is still glued to $\ln(C)$ after 50 steps with no downward trend whatsoever, that is not "needs more steps" — a working setup moves *visibly* by step 50. A loss that is slowly but steadily decreasing (2.30 → 2.27 → 2.24 …) is a different signal: the wiring works but the LR is too small; bump it up by 10× for the test. The distinction matters because "flat" and "slowly decreasing" route to completely different bugs — flat is a wiring fault, slow is just a too-small LR.

Why Adam and not SGD for the *test* specifically? Adam's per-parameter adaptive step sizes make it far more forgiving of a poorly-scaled problem, so it converges on a tiny batch with almost any reasonable LR. SGD can stall on the overfit test for benign reasons (the LR is mis-scaled for the loss magnitude), which would muddy your read. Use Adam for the diagnostic even if your real run uses SGD — you are testing the *wiring*, not reproducing the production optimizer. Once the Adam overfit test passes, you have proven the path works; you can then separately verify SGD with your production LR also overfits if you want to validate the optimizer config itself.

One more numeric anchor. With Adam at $10^{-3}$ on a 16-example, 10-class batch, a healthy run typically looks like: loss 2.30 at step 0, visibly under 2.0 by step 20–30, under 0.5 by step 80–100, and under 0.01 by step 150–250. If you see that shape, the test passed and you can read the exact threshold crossing off the trajectory. If after 300 steps the loss has not moved off 2.30 at all, stop the test and go to the grad-norm print — more steps will not help a dead gradient.

### 2.4 Why a *failure* to overfit is so diagnostic

Here is the logical structure that makes the test valuable. Overfitting one batch requires a conjunction of conditions, all true at once:

$$
\text{pass} \iff (\text{capacity}) \land (\text{gradients flow}) \land (\text{optimizer steps}) \land (\text{loss wired to labels}) \land (\text{data path delivers labels})
$$

Because it is a conjunction, a single false term flips the whole result to fail. That is what makes a failure informative: the *space of explanations is small and mechanical*. When the test fails, it is almost always exactly one of:

- **No gradient flow** — a module has `requires_grad = False`, the graph was detached somewhere (`.detach()`, `.item()`, `.numpy()`, an in-place op that breaks autograd), or the whole model is in a `torch.no_grad()` context.
- **Frozen parameters** — a freeze experiment was left on, a `param_group` was excluded from the optimizer, or you passed `model.head.parameters()` to the optimizer but train the whole model.
- **Learning rate is 0 or absurd** — LR scheduler warming up from 0 and the test is too short, LR literally set to 0, or LR so high every step diverges.
- **Wrong loss reduction or wiring** — `reduction='none'` so the "loss" is a vector you accidentally `.mean()` over the wrong axis, logits passed where probabilities are expected, `ignore_index` masking every token, or labels offset by one.
- **Label / logit mismatch** — the labels are all one class (collation bug), the labels and logits are misaligned (shifted, transposed), or the target tensor is the wrong dtype and silently casts to zeros.
- **Data all the same** — every example in the batch has the same label or the same input, so the "task" is degenerate (this one can also cause a *false pass*, covered later).

A pass, conversely, asserts the *negation* of all of these at once. That is the leverage: one cheap test, many bugs ruled out. The figure below makes the cleave explicit.

![A branching decision graph showing a pass rules out capacity and gradient and loss-wiring bugs while a fail points at five mechanical faults](/imgs/blogs/the-overfit-a-single-batch-test-2.png)

## 3. The diagnostic: a clean, reusable `overfit_one_batch()`

Now the practical core. Here is a self-contained PyTorch function you can drop into any project. It grabs one batch, disables every source of stochasticity, trains to convergence, prints the loss trajectory, and asserts the loss drops below a threshold. Read the comments — every line is doing diagnostic work.

```python
import torch

def overfit_one_batch(
    model,
    batch,                 # (inputs, targets) already on the right device
    loss_fn,
    lr=1e-3,
    steps=300,
    threshold=0.01,        # CE: ~99% confident; raise if label smoothing is on
    log_every=20,
):
    """Train on ONE fixed batch and prove the model can memorize it.

    A pass rules out: dead gradients, frozen params, LR=0, loss-label
    mismatch, insufficient capacity. A fail localizes one of those.
    """
    inputs, targets = batch

    # 1. Kill all stochasticity: dropout, BatchNorm updates, augmentation.
    #    We WANT a deterministic, memorizable target.
    model.train()
    for m in model.modules():
        # Put dropout/BN into eval so they don't add noise or adapt stats.
        if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d,
                          torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()

    # 2. Fresh optimizer, NO weight decay (decay fights memorization).
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    # 3. Sanity: assert SOMETHING is trainable before we start.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable > 0, "No trainable parameters! Everything is frozen."
    print(f"trainable params: {n_trainable:,}")

    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

        # 4. Grad-norm check on step 0: if it's ~0, gradients aren't flowing.
        if step == 0:
            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total += p.grad.detach().norm().item() ** 2
            gnorm = total ** 0.5
            print(f"step 0 grad norm: {gnorm:.3e}")
            assert gnorm > 0, "Zero gradient at step 0 — detached graph or frozen model."

        opt.step()
        losses.append(loss.item())
        if step % log_every == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    final = losses[-1]
    print(f"final loss: {final:.6f}  (threshold {threshold})")
    assert final < threshold, (
        f"FAILED to overfit one batch: loss stuck at {final:.4f}. "
        "Model/loss/optimizer/data path is broken — do NOT launch the full run."
    )
    print("PASSED: model can memorize one batch. Capacity & wiring OK.")
    return losses
```

Three design choices are doing the heavy lifting and deserve a word each.

First, **we freeze the batch outside the loop.** `inputs` and `targets` are captured once and reused every step. This is non-negotiable: if you call `next(iter(dataloader))` inside the loop, you get a *different* batch each step (with shuffling on), and a falling loss then proves nothing about memorization. We will see this exact trap turn into a false pass in Section 7.

Second, **we put BatchNorm and Dropout into eval mode** even though the *model* is in `train()`. Dropout randomly zeroes activations, which adds noise that fights memorization; BatchNorm in train mode adapts its running statistics to this tiny 16-example batch, which can let the model "cheat" by overfitting through the normalization layer rather than the weights you care about. For a clean test of the *weights*, you want both quiet. (There is nuance here — sometimes you *do* want to test the train-mode path — which we return to in the false-pass section.)

Third, **the grad-norm check at step 0 is the single most useful line in the function.** If the global gradient norm is zero before the first optimizer step, then no learning is possible, full stop — the optimizer is about to take a step of size zero. That one print converts "the loss isn't dropping, why?" into "the gradient is literally zero, so find where the graph is cut." We will extend this into a per-module grad-norm print in Section 5 to name the exact dead layer.

To use it on our running example:

```python
# Grab ONE real batch from the actual dataloader (not a synthetic tensor).
batch = next(iter(train_loader))
inputs, targets = batch[0].to(device), batch[1].to(device)

loss_fn = torch.nn.CrossEntropyLoss()
overfit_one_batch(model, (inputs, targets), loss_fn, lr=1e-3, steps=300)
```

### How to read the result

There are exactly three outcomes, and each tells you something:

1. **Loss drops below threshold (e.g., 2.30 → 0.0004).** The model has enough capacity, gradients flow, the optimizer steps, and the loss is wired to the labels. You have ruled out a large class of bugs. If your *full* run is still failing, the bug is in something the overfit test deliberately turns off: the data at scale, augmentation, the LR schedule, regularization, or generalization itself. Stop blaming the model.
2. **Loss is completely flat (stuck at ~$\ln C$).** No learning is happening at all. Gradients are not reaching the parameters that matter. Go straight to the grad-norm-per-module check (Section 5). This is the frozen-head, detached-graph, LR=0 family.
3. **Loss moves but never reaches the floor, or goes NaN.** The wiring is partly working but the numbers are wrong — bad loss reduction, label/logit mismatch, or a numerics problem (log of zero, fp16 overflow). The loss-and-label section of the taxonomy is your next stop.

The figure below is the field guide for outcome 2 versus 3 — it bisects a failed test into the no-signal branch and the wrong-numbers branch, each with a one-line confirming check.

![A decision tree that splits a flat-loss failure into a no-gradient branch and a wrong-numbers branch, each leading to specific suspects](/imgs/blogs/the-overfit-a-single-batch-test-5.png)

#### Worked example: catching the frozen head in 30 seconds

Back to our ResNet-18 run that was pinned at 10% val accuracy. We run `overfit_one_batch` on one real batch of 16 images. The output:

```bashtrainable params: 11,176,512
step 0 grad norm: 1.142e-03
step    0  loss 2.3041
step   20  loss 2.3038
step   40  loss 2.3036
step   60  loss 2.3034
step   80  loss 2.3033
step  100  loss 2.3031
...
step  280  loss 2.3027
final loss: 2.302612  (threshold 0.01)
AssertionError: FAILED to overfit one batch: loss stuck at 2.3026.
```

Three things jump out. The loss is parked at **2.3026 = $\ln(10)$** — the model is guessing uniformly. The trainable-param count is 11.1M, so it is *not* the case that everything is frozen (that would assert immediately). And the global grad norm is a tiny but nonzero `1.1e-3`. That combination is the tell: gradients are flowing *somewhere* (the backbone), but the part that turns features into class logits is not moving, so the logits stay constant and the loss cannot drop below chance. The next step is to find *which* module is dead — which is Section 5. The whole diagnosis took less than a minute and zero GPU-days, versus the three days the real run cost.

## 4. Why this test belongs *before* every serious run

Let me make the economic and methodological argument explicit, because the test's value is not just technical — it is about the order in which you spend your attention and your money.

A full training run is the most expensive possible way to discover a bug. It is slow (hours to days), it is costly (at, say, \$2.50 per GPU-hour, a four-day eight-GPU run is roughly \$1,900), and worst of all it produces *ambiguous* feedback. When a full run "fails," you do not get a clean error — you get a loss curve that did something disappointing, and now you have to reason backward through six possible bug locations *and* the confound of real data, real augmentation, real distribution shift, and real generalization, all at once. You cannot tell whether the model is broken or the data is hard.

The overfit test removes every one of those confounds by construction. By training on a fixed, tiny, augmentation-free batch, you eliminate:

- **Generalization** — there is no held-out set, so you cannot be confused by a train/val gap.
- **Data scale and diversity** — sixteen examples, fixed, so it is not a "the dataset is noisy" story.
- **Augmentation and regularization** — turned off, so they cannot be the variable.
- **Learning-rate scheduling** — a constant LR for a few hundred steps, so warmup and decay are out of the picture.

What remains is a clean question with a binary answer: *can the optimization machinery fit a function to these points?* If yes, the machine works and your remaining bugs live in the stuff you turned off. If no, the machine is broken and you fix it before spending another dollar. This is **bisection of the bug space**, and the overfit test is the first and cheapest cut. It is the reason this series puts it second, right after the mindset post: it is the move you reach for before almost any other.

There is a cultural point here too. Andrej Karpathy's widely-cited "Recipe for Training Neural Networks" lists "overfit a single batch" as one of the first concrete steps after setting up a skeleton, precisely because it catches so many bugs so cheaply. It is one of those practices that separates people who debug ML by superstition ("let me try a different LR and see") from people who debug it by elimination. The discipline is: never launch a long run you have not first proven *can* learn at all.

To make the cost asymmetry concrete, consider the decision tree a disciplined engineer faces at the moment of launch. The overfit test costs essentially nothing — call it 30 seconds of wall-clock and a rounding error of GPU time. The full run costs hours to days. The probability that *some* wiring bug is present in a fresh or recently-modified training script is, in my experience, uncomfortably high — easily 20–30% for a script that just had a refactor, a new model head, a changed loss, or a new finetuning config. So the expected value calculation is lopsided:

| Path | Cost if clean | Cost if buggy | When you find out |
| --- | --- | --- | --- |
| Skip the test, launch | ~0 extra | full run wasted (hours–days, hundreds of \$) | after the run, ambiguously |
| Run the test first | +30 seconds | +30 seconds, then fix in minutes | before the run, precisely |

The test adds a fixed, trivial cost and, in the buggy case, converts a multi-day ambiguous loss into a 30-second precise diagnosis. There is no scenario where running it is the wrong call. The only reason people skip it is impatience — the urge to "just launch and see" — which is exactly the instinct this series is trying to retrain. The overfit test is the cheapest insurance policy in machine learning, and it pays out constantly.

A related discipline is to make the test *automatic*. Wire `overfit_one_batch` into a smoke test that runs in CI or as the first thing your training entrypoint does on a tiny config, so that a wiring regression — someone freezes a module, changes the loss, breaks the collator — fails fast and loud instead of silently surviving into a launched run. A model that *was* able to overfit one batch last week and suddenly cannot is a precise signal that a recent change broke the wiring, and a smoke test turns that signal into a red build instead of a wasted weekend. This is the same philosophy as the bisection frame: hold everything fixed, change one thing, and let a cheap test tell you the moment the wiring breaks.

## 5. Per-module grad norms: naming the dead layer

When the overfit test fails with a flat loss and a tiny-but-nonzero global grad norm, you need to know *which* module is not receiving gradient. The fix is a loop over `named_parameters()` that prints the grad norm of each, aggregated by top-level module. Here is the diagnostic:

```python
import torch
from collections import defaultdict

def module_grad_norms(model, batch, loss_fn):
    """One forward+backward, then print grad norm per top-level module.
    A module with grad norm 0.0 is frozen or cut out of the graph."""
    inputs, targets = batch
    model.train()
    model.zero_grad(set_to_none=True)

    loss = loss_fn(model(inputs), targets)
    loss.backward()

    norms = defaultdict(float)
    has_grad = defaultdict(bool)
    for name, p in model.named_parameters():
        top = name.split(".")[0]          # group by first module name
        if p.grad is not None:
            norms[top] += p.grad.detach().norm().item() ** 2
            has_grad[top] = True
        elif not p.requires_grad:
            norms[top] += 0.0             # frozen: no grad by design or by bug

    print(f"loss = {loss.item():.4f}")
    for top in norms:
        g = norms[top] ** 0.5
        flag = "" if g > 0 else "   <-- ZERO GRAD (frozen or detached)"
        req = any(p.requires_grad for n, p in model.named_parameters()
                  if n.split(".")[0] == top)
        print(f"  {top:20s} grad_norm={g:.3e}  requires_grad={req}{flag}")
```

Running it on our frozen-head ResNet:

```bashloss = 2.3041
  conv1                grad_norm=8.41e-04  requires_grad=True
  bn1                  grad_norm=2.02e-04  requires_grad=True
  layer1               grad_norm=3.77e-04  requires_grad=True
  layer2               grad_norm=4.10e-04  requires_grad=True
  layer3               grad_norm=5.55e-04  requires_grad=True
  layer4               grad_norm=6.98e-04  requires_grad=True
  fc                   grad_norm=0.000e+00  requires_grad=False   <-- ZERO GRAD (frozen or detached)
```

There it is, named: `fc` (the classifier head) has `requires_grad=False` and a grad norm of exactly zero. The backbone (`conv1` through `layer4`) is receiving gradient and would slowly learn features, but the head that maps those features to the ten class logits is a brick — so the logits are constant and the loss cannot move below chance. The fix is one line:

```python
for p in model.fc.parameters():
    p.requires_grad = True
# and make sure the optimizer actually includes these params:
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

After the fix, re-running `overfit_one_batch` gives a completely different trajectory, which the before/after figure captures: the head's grad norm is now ~0.8 at step 1, the loss falls from 2.30 to 0.0003 in about 180 steps, and the assertion passes. *Now* the full run is worth launching.

![A before-and-after comparison of the same run, frozen head with loss stuck at 2.30 on the left versus re-enabled gradients converging to 0.0003 on the right](/imgs/blogs/the-overfit-a-single-batch-test-3.png)

### The grad-flow picture

The grad-norm-per-module print is really an X-ray of gradient flow. In a healthy backward pass, every trained module shows a nonzero norm; the dead module shows zero. The figure below renders that X-ray for our case — gradient enters at the loss, flows through the backbone and neck, and stops dead at the head, which immediately names the culprit.

![A dataflow graph showing gradients flowing through backbone and neck with nonzero norms but reaching the classifier head at zero, naming it the frozen module](/imgs/blogs/the-overfit-a-single-batch-test-8.png)

This same per-module grad-norm check is the workhorse of [why your model isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think), which goes deep on the full taxonomy of broken gradient flow — detached graphs, in-place ops that silently break autograd, params excluded from the optimizer, and the difference between "no grad" and "tiny grad." For our purposes the rule is simple: **a flat overfit loss plus a zero per-module grad norm equals a frozen or detached module, and the print tells you its name.**

#### Worked example: the detached graph that ate the loss

A second, sneakier version of the same failure. An engineer adds a custom metric inside the training step and writes:

```python
logits = model(inputs)
preds = logits.detach()                 # for the metric, fine
loss = loss_fn(preds, targets)          # BUG: loss computed on the DETACHED tensor
loss.backward()
```

The loss is computed on `preds`, which was detached from the graph, so `loss.backward()` has nothing to propagate into the model — *every* parameter gets zero gradient. The overfit test catches this instantly:

```bashtrainable params: 11,176,512
step 0 grad norm: 0.000e+00
AssertionError: Zero gradient at step 0 — detached graph or frozen model.
```

The global grad norm is exactly 0 (not the 1.1e-3 from the frozen-head case, because here *nothing* gets gradient, not even the backbone). The assertion fires at step 0 before the loss even has a chance to plateau. The fix is to compute the loss on the live `logits` and only detach the copy used for the metric:

```python
logits = model(inputs)
loss = loss_fn(logits, targets)         # live tensor: gradients flow
with torch.no_grad():
    preds = logits.argmax(dim=1)        # detached only for the metric
loss.backward()
```

Two different bugs — frozen head versus detached graph — and the overfit test plus grad-norm print distinguished them in seconds: one gives a tiny-but-nonzero global norm with one dead module, the other gives an exactly-zero global norm at step 0.

#### Worked example: the regression head that could not reach its targets

The same test works for regression, and a third bug class shows up there: an output that *cannot represent the targets at all*. A team was training a model to predict house prices (targets in the hundreds of thousands), with a final linear layer feeding into a sigmoid by accident — left over from a binary-classification template the code was copied from. They ran the overfit test on 16 examples with MSE loss:

```bashstep    0  loss 4.21e+10
step   40  loss 3.98e+10
step   80  loss 3.98e+10
step  120  loss 3.98e+10
final loss: 3.98e+10  (threshold 1e-04)
AssertionError: FAILED to overfit one batch: loss stuck at 39800000000.0000.
```

The loss is enormous and *flat after the first few steps* — a different signature from the classification cases. The grad-norm print showed nonzero gradients everywhere, so it was not a frozen module. The tell was the floor: MSE has a true zero, so a model that floors at $4 \times 10^{10}$ literally *cannot output the targets*. A sigmoid output is bounded in $(0, 1)$; no setting of the weights makes it emit 350,000. The model was asked to represent values it cannot produce, so the best it can do is saturate the sigmoid to ~1 and eat the gigantic residual. The fix was to remove the spurious sigmoid (a regression head should output an unbounded linear value, or you should normalize the targets to a comparable scale):

```python
# Before (bug): bounded output cannot reach large targets
self.head = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
# After: linear output, or normalize targets to ~unit scale
self.head = nn.Linear(d, 1)
```

After the fix, the overfit loss fell to $8 \times 10^{-6}$ in 200 steps. The lesson generalizes: when an overfit test floors *above* the loss's true minimum with healthy gradients, the model's output range cannot represent the targets — a scaling or output-activation bug, not a gradient-flow bug. The grad-norm print is what distinguishes "can't reach the targets" (gradients fine, output range wrong) from "no gradient at all" (frozen or detached).

## 6. The debugging ladder: smaller and smaller failures

The overfit-one-batch test is the most common rung of a more general principle: **make the failing thing as small as possible, in every dimension, until it either works or breaks legibly.** There is a whole ladder of "overfit X" tests, each isolating a different subsystem. Climb up when the batch test is ambiguous; climb down when you want to isolate something even more specific. The matrix below lays out the rungs and what each one proves or rules in.

![A matrix of debugging-ladder rungs from one example to one step, each row showing what passing proves and what failing rules in](/imgs/blogs/the-overfit-a-single-batch-test-4.png)

**Overfit one example.** The absolute floor: train on a *single* input-target pair. If the loss cannot reach zero on one point, the most basic wiring is broken — there is no "the batch is too diverse" excuse possible. This is the test to run when even the 16-example batch is behaving strangely; it removes all batch-level effects (batching, padding, the mean over examples) and tests the rawest forward-backward-step loop.

**Overfit one class.** Build a batch where every example has the *same* label. The model should drive the loss to zero by learning to always predict that class — a trivial constant function. If this fails, the suspect is in the loss-to-label wiring: a class-index off-by-one (your labels are 1-indexed but the loss expects 0-indexed), a logit/target misalignment, or an `ignore_index` that is masking the very class you are training on. This is a sharp probe for [loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs), because it makes any indexing error glaringly visible.

**Overfit one feature (tabular).** For tabular models, construct a batch where the target is a deterministic function of a *single* input column (e.g., `y = 1 if x_3 > 0 else 0`), and feed the model only the informative column or a batch where that column alone determines the label. If the model cannot fit it, the feature is not reaching the model — an encoding bug, a collation that drops the column, a NaN that silently zeros it, or a scaler that maps it to a constant. This rung is how you debug feature-pipeline plumbing in XGBoost/LightGBM and sklearn `Pipeline`s, where the bug is usually that a transform mangled the feature before the model ever saw it.

**Overfit on one GPU.** Before you go multi-GPU, prove the *single*-device run works. If the single-GPU overfit test passes but the multi-GPU run fails, the bug is in the distributed layer — `find_unused_parameters` errors from a module that does not receive gradient on some ranks, gradient-sync incorrectness, rank desync from per-rank seeding, or BatchNorm statistics that should be SyncBN. This cleanly separates "is my model broken?" from "is my DDP setup broken?" — two questions people routinely conflate, then waste a day debugging distributed code when the model itself never worked.

**Run one step.** The smallest rung: just do *one* forward and one backward pass and check it does not crash or produce NaN. This catches shape bugs, dtype mismatches, and label-range errors (a label of 12 in a 10-class problem indexes out of bounds) before you even start a loop. It is the "does it compile and run once" test, and it is worth a single line at the top of every training script.

The unifying logic is the bisection frame from the [bug taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs): each rung holds more of the system fixed, so a failure at a given rung points at the subsystem that rung *added* compared to the one below it. One step → shapes/dtypes. One example → forward-backward-step wiring. One batch → batching and capacity. One GPU vs many → the distributed layer. You climb the ladder until you find the rung where the failure appears, and that rung names the bug's neighborhood.

#### Worked example: the bug that only appears on 8 GPUs

A team's model passed `overfit_one_batch` perfectly on a single GPU: loss 2.30 → 0.0007 in 200 steps. They launched it on 8 GPUs with `DistributedDataParallel` and got an immediate runtime error:

```bashRuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. ... find_unused_parameters=True ...
```

Because the single-GPU overfit test had passed, they knew instantly the bug was *not* in the model, the loss, or the data path — it was in the distributed layer. The cause: the model had an auxiliary head that was only used on a fraction of batches, so on some ranks its parameters received no gradient, and DDP's gradient reducer hung waiting for them. The single-GPU pass had cleaved the bug space cleanly: "model fine, DDP broken." The fix lived entirely in the DDP config (setting `find_unused_parameters=True`, or better, restructuring so every parameter is always used), and they never wasted a minute suspecting the model. That is the ladder doing its job — the rung where the failure appeared (single-GPU pass, multi-GPU fail) named the neighborhood. Multi-GPU specifics live in the systems track of the taxonomy.

## 7. False passes: when "it overfit" is a lie

A pass is only worth something if you earned it honestly. There are four common ways a *broken* setup can drive the loss to zero on a tiny batch for the wrong reason, making you think the wiring is sound when it is not. These false passes are dangerous precisely because they are reassuring. Know all four. The matrix below pairs each trap with the tell that gives it away and the guard that exposes it.

![A matrix of four false-pass traps, each row showing how it fakes a passing test and the guard that exposes it](/imgs/blogs/the-overfit-a-single-batch-test-7.png)

**1. Label leak in the batch.** If one of your input features *is* the label (or a near-perfect proxy for it), the model fits the batch trivially by reading that feature — but it learned nothing about the real task, and the same leak will inflate your validation metric and then collapse in production. This is the tabular nightmare: a column like `account_closed_date` that is only populated for the positive class. The overfit test passes, the full run posts a 0.97 AUC, and production gets 0.71. The guard: when you suspect a leak, drop the suspect column and re-run; an honest setup still overfits the batch without it (the model has the capacity to memorize 16 points regardless), so the test still passes — but now you have confirmed the model is not *depending* on the leak to do it. Leakage gets its own deep treatment in the data track; the overfit test is not a leak detector, but a suspiciously *easy* and *fast* convergence on a real batch is a hint worth following.

**2. BatchNorm in train mode.** This is the subtle one. If you leave BatchNorm in `train()` mode during the overfit test, it normalizes each batch using *that batch's* statistics and adapts its running mean/variance to your 16 examples. That gives the model an extra, almost-free way to fit the batch: it can "memorize" through the normalization layer's per-batch shift and scale rather than through the weights you actually care about. The test passes, but it passed partly because of an effect that will not exist at inference time (when BN uses fixed running stats over a different distribution), so the train-mode pass overstates how healthy the weights are. This is why the `overfit_one_batch` function above forces BN into `eval()`. The guard: repeat the test with the model fully in `eval()` (or swap BN for GroupNorm/LayerNorm, which have no batch-dependent stats); if it still converges, the weights are genuinely learning. The train/eval distinction is a whole bug family on its own, covered in the model-code track.

**3. Synthetic batch too easy.** If you test on a handcrafted tensor — say `inputs = torch.zeros(16, 3, 224, 224)` with `targets = torch.arange(16) % 10` — you may be testing a degenerate, trivially separable problem that hides a broken *real* data path. The model overfits the synthetic batch because it is easy, you conclude the pipeline is fine, and then the real dataloader is delivering corrupted images or all-zero labels. The guard is in the function's contract: **always pull the fixed batch from the real `DataLoader`** (`next(iter(train_loader))`), not from a synthetic tensor. That way a pass also exercises the collate function, the transforms (which you then disable), the label encoding, and the device transfer — the actual plumbing.

**4. Augmentation silently still on.** If augmentation is applied inside the dataset's `__getitem__` and you grab the batch *once* but the transforms are stochastic and re-applied — or worse, if you accidentally re-fetch the batch each step — then the "fixed" batch is not fixed, and a falling loss does not prove memorization. The model is just tracking a moving target that happens to be easy on average. The guard: freeze the batch into a plain tensor *outside* the loop and assert it is byte-identical across steps. A blunt, effective check:

```python
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.to(device), targets.to(device)
# Freeze: snapshot the exact tensor and assert it never changes.
ref = inputs.clone()
for step in range(steps):
    assert torch.equal(inputs, ref), "Batch changed between steps — not fixed!"
    ...  # train on the SAME inputs/targets every step
```

If that assertion ever fires, your batch is not frozen and your "pass" is meaningless.

The meta-lesson: a pass means "the model *can* memorize this exact batch." For that to imply "the model and wiring are healthy," the batch must be *real*, *fixed*, and *not trivially solvable by a leak or a normalization shortcut*. Honor those three conditions and the pass is gold. Violate them and you have a comforting lie. When in doubt, the cheapest tiebreaker is the per-module grad-norm print from Section 5: if the weights you care about are receiving real gradient and the loss is dropping because of *them*, the pass is honest.

## 8. The test for LLM finetuning: where it earns its keep most

The overfit-one-batch test is useful for vision, but it is *transformative* for LLM finetuning, because LLM finetuning has more silent failure modes than almost any other workflow — and nearly all of them announce themselves on a one-batch test. If you take one practical habit from this post into your finetuning work, make it this: before you launch an SFT, LoRA, or DPO run, prove the model can drive the loss on one short fixed batch of your formatted examples to near zero.

The mechanics are the same, with one twist: for a language model the loss is a *token-level* cross-entropy, averaged over the unmasked target tokens, so "loss to ~0" means the model becomes nearly certain of every target token in the batch. For a model whose tokenizer has vocabulary size $V$, an untrained or broken run sits near $\ln(V)$ for an unconditioned guess (for a 32,000-token vocab that is $\ln(32000) \approx 10.4$), though in practice a pretrained model starts well below that because it already predicts language. A *finetuning* overfit test should drive the batch's average token loss from its pretrained starting value (often 1–3) down toward 0.05 or lower as the model memorizes the exact completions. If it cannot, one of the LLM-specific wiring bugs is present.

Here is the finetuning-specific version, built on HuggingFace `transformers` and `trl`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def overfit_one_batch_llm(model, tokenizer, texts, lr=1e-4, steps=200, threshold=0.05):
    """Prove an LLM can memorize a tiny fixed batch of formatted examples.
    Catches loss-masking, label-shift, frozen-adapter, and template bugs."""
    model.train()
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    # Standard causal-LM labels: copy of input_ids, pad positions masked to -100.
    labels = input_ids.clone()
    labels[attn == 0] = -100              # don't compute loss on pad tokens

    # Confirm SOMETHING trains (the classic frozen-LoRA check).
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train > 0, "No trainable params — LoRA adapter not in the graph?"
    print(f"trainable params: {n_train:,}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        loss.backward()
        if step == 0:
            g = sum(p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None) ** 0.5
            print(f"step 0 grad norm: {g:.3e}")
            assert g > 0, "Zero grad — detached, frozen, or adapter not attached."
        opt.step()
        if step % 20 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")
    assert loss.item() < threshold, (
        f"LLM failed to overfit: loss {loss.item():.4f}. Check loss masking, "
        "label shift, frozen adapter, or chat-template mismatch.")
    print("PASSED: LLM memorized the batch.")
```

What makes this so valuable is the specific bugs it catches, each of which would otherwise survive into a multi-hour run and produce a model that *looks* finetuned but is not:

- **The LoRA silent no-op.** If `target_modules` is wrong, or `get_peft_model` was never called, or the adapter dtype mismatches, the adapter contributes nothing and the trainable-param count is wrong (or the grad norm is zero). The first two lines — the trainable-param assert and the step-0 grad-norm assert — catch this immediately. The signature is a finetune that costs hours and produces a model byte-identical to the base.
- **The loss-masking bug.** If `labels` are not set to `-100` on the prompt tokens, the model trains to predict the *prompt* as well as the completion, which wastes capacity and produces a model that parrots prompts. Or, the inverse: if `-100` masks *everything* (a collation bug sets the whole label tensor to `-100`), the loss is computed over zero tokens and is `nan` or a constant — the overfit test cannot move and fails loudly. The loss-masking bug is so common it has its own post in this series.
- **The label-shift bug.** Causal LM loss requires labels shifted by one relative to logits (HuggingFace does the shift internally when you pass `labels=`, but custom training loops often get it wrong). A wrong shift means the model is trained to predict the *current* token from itself — trivially solvable in a degenerate way, which can produce a deceptive partial pass; or, if shifted the wrong direction, an unlearnable target. The overfit test surfaces both: a too-easy "pass" that does not generalize, or a stubborn failure.
- **The chat-template mismatch.** If you train on raw text but serve with a chat template (or vice versa), the formatted examples in your fixed batch will not match what the model expects, and the overfit loss may be unnecessarily high. Building the fixed batch from your *exact* formatted strings — the same `apply_chat_template` output you will use at inference — means a pass also validates the formatting.

#### Worked example: the LoRA adapter that never trained

A team finetuned a 7B model with LoRA for two hours, then evaluated and found the outputs identical to the base model — the finetune had done nothing. Running `overfit_one_batch_llm` on four formatted examples *before* relaunching would have caught it in under a minute:

```bashtrainable params: 0
AssertionError: No trainable params — LoRA adapter not in the graph?
```

Zero trainable parameters. The cause: `get_peft_model(model, lora_config)` was called, but the returned model was assigned to a *different* variable than the one passed to the `Trainer`, so the `Trainer` trained the frozen base model (whose params are all `requires_grad=False` under the peft wrapping) and the adapter never entered any optimizer. The base model's params were frozen by peft, the adapter was orphaned, and so *nothing* was trainable. The `print_trainable_parameters()` call from peft shows the same thing: `trainable params: 0 || all params: 6,738,415,616 || trainable%: 0.0`. Two hours and roughly \$10 of GPU time, versus a 40-second assert. After fixing the variable assignment, the trainable-param count read `trainable params: 4,194,304 || trainable%: 0.062`, the step-0 grad norm was nonzero, and the four-example loss fell from 2.1 to 0.03 in 120 steps — *now* the run was worth launching. This is the LoRA-specific instance of the exact same principle that found the frozen vision head: a finetune that runs but learns nothing, caught by asking whether it can memorize one batch.

The broader point: LLM finetuning multiplies the number of places to silently break the loss-to-label path — masking, shifting, padding, templates, adapters — and every one of those breaks shows up as a failure to overfit one batch. The overfit test is the single check that covers all of them at once, which is why it is the first thing to run before any finetune. The masking and template specifics get their own posts in the LLM track; the overfit test is how you *detect* that one of them is wrong without reading every line of your collator.

## 9. The full before→after, with instruments

Let us put the whole diagnosis on one page, the way you would write it in a post-mortem, with the instruments before and after. This is the evidence standard this series holds itself to: name the symptom, the confirming test, the fix, and what the instruments read afterward.

| Instrument | Before (broken) | After (fixed) | What it confirms |
| --- | --- | --- | --- |
| Overfit-one-batch final loss | 2.3026 (= $\ln 10$) | 0.0003 | Model can now memorize the batch |
| Loss trajectory | flat 200 steps | 2.30 → 0.0003 in ~180 steps | Optimization machinery works |
| Global grad norm @ step 0 | 1.1e-3 | 1.1e-3 (unchanged) | Backbone always flowed |
| `fc` (head) grad norm | 0.000 | 0.81 | Head now receives gradient |
| `fc.requires_grad` | False | True | The actual fix |
| Trainable params | 11,176,512 | 11,181,642 | Head's 5,130 params rejoined |
| Full-run val accuracy | 10.4% (chance) | 87.2% | The run finally learns |
| GPU-hours to detect | ~750 (3 days) | ~0.01 (30 s) | The test's whole point |

The honest way to *confirm* this fix is exactly the test that found it: re-run `overfit_one_batch` and watch the loss go to zero where it previously stalled. You do not need the full run to validate the fix — the overfit test is both the detector *and* the confirmation. That is unusually clean for a debugging tool. The figure below shows the healthy loss trajectory step by step, which is the shape you are confirming.

![A timeline of a healthy overfit run showing loss falling from 2.302 at init through a steep descent to 0.0004 by step 200](/imgs/blogs/the-overfit-a-single-batch-test-6.png)

Note the line in the table I want to highlight: **GPU-hours to detect, 750 versus 0.01.** That ratio — five orders of magnitude — is the entire economic argument for this practice. The bug was identical in both worlds; only the *order in which we looked* changed. The team that ran the overfit test first paid 30 seconds. The team that launched the full run first paid three days and \$1,900. Same bug, same fix, vastly different cost, purely because of debugging discipline. This is why instrumenting and sanity-checking before scaling is not bureaucratic caution — it is the highest-return habit in the field. The full menu of what to instrument and log is the subject of [instrumenting a training run: what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log); the overfit test is the one you run *before* you even start logging.

## 10. Case studies and real signatures

Three patterns from the wild, each a real-ish signature you will eventually meet, and each one the overfit test catches or clarifies.

**The frozen-by-default finetune.** When you finetune a pretrained backbone, many recipes freeze the backbone and train only a new head — and many *bugs* do the opposite by accident, freezing the head you just attached or forgetting to add it to the optimizer. The HuggingFace `Trainer` and `peft` flows are full of places to do this: `LoraConfig` with the wrong `target_modules` so the adapter is attached to layers that never see gradient, or a `freeze_backbone()` helper that froze everything. The signature is always the same: a finetune that runs but never beats the pretrained baseline, and an overfit-one-batch loss stuck at chance. The fix is to call `model.print_trainable_parameters()` (peft) or the grad-norm loop above and confirm the modules you *intend* to train have nonzero grad. This is the LoRA "silent no-op," and it is so common that "did you overfit one batch?" is the first question any experienced person asks when a LoRA run does nothing.

**The random-label control (Zhang et al., 2017).** The reason the overfit test is theoretically sound traces to *Understanding deep learning requires rethinking generalization*. The authors trained standard architectures on CIFAR-10 and ImageNet with the labels replaced by *random* noise and showed the networks still reached near-zero training error — they have enough capacity to memorize arbitrary labelings. The practical corollary for debugging: if your model *cannot* fit even the *true* labels of sixteen examples, the problem is categorically not "the task is too hard for this architecture." The architecture can memorize random noise at scale; something in your pipeline is stopping it from memorizing real labels at small scale. That is a wiring bug, not a capacity bug, and the overfit test is how you assert it.

**The confident-learning label-error finding (Northcutt et al., 2021).** The *Pervasive Label Errors in Test Sets* work used confident learning (the method behind `cleanlab`) to find that widely-used benchmark *test* sets contain on the order of a few percent mislabeled examples — roughly 3.4% averaged across ten datasets, including famous errors in the ImageNet and MNIST test sets. The connection to the overfit test is a subtle false-pass warning: if your tiny batch happens to contain a mislabeled example, a model that has correctly *learned the task* may be unable to drive that one example's loss to zero, because the "correct" answer in your batch is actually wrong. So a loss that floors at a small *nonzero* value (rather than near-machine-epsilon) on a handful of examples can be a hint that one of those examples is mislabeled — the model is right and the label is wrong. The data track covers finding these systematically; for now, treat a stubborn-but-small residual loss on one or two specific examples as a "look at that example" flag, not necessarily a bug in the model.

**The off-by-one label that made the loss unlearnable.** A common, infuriating bug: labels stored 1-indexed (classes 1 through 10) fed to `CrossEntropyLoss`, which expects 0-indexed class indices (0 through 9). The loss does not crash — it happily interprets label `10` as class index 10, which is *out of range* for a 10-logit output and either errors (if you are lucky) or, with certain configurations, silently produces garbage gradients. More subtly, if the labels are 0–10 (eleven distinct values) but the head has only 10 outputs, the highest class can never be predicted and its examples can never reach zero loss. The overfit-one-class variant from the ladder catches this with surgical precision: build a batch that is *all class 10*, and watch the loss refuse to drop while every other class works. That asymmetry — one class unlearnable, the rest fine — is the signature of an index off-by-one, and it is exactly the kind of thing that an overfit test on a *mixed* batch might partially hide (the other nine classes drag the average down) but a single-class overfit test exposes cleanly. The fix is a one-liner (`labels = labels - 1`), but finding it without the test can cost a day of staring at a loss that is "almost" converging.

These four together draw the boundary of the test cleanly. It is a *capacity-and-wiring* probe, grounded in the fact that overparameterized models memorize anything (Zhang). A failure means broken wiring (the frozen finetune, the off-by-one label). And its few honest limitations — a leak, a mislabeled example, a normalization shortcut — are exactly the false-pass traps from Section 7, made concrete by real findings (Northcutt).

## 11. When this is — and isn't — your bug

A decisive section, because the test's *negative* result is as informative as its positive one. Read these as routing rules.

**If overfit-one-batch PASSES, stop blaming the model, the loss, and the optimizer.** A pass is a strong statement: the model has the capacity, gradients flow, the optimizer steps, and the loss is wired to the labels. If your *full* run is still failing after a clean pass, the bug is in something the test deliberately turned off. The usual suspects, in rough order of likelihood:

- **Data at scale.** Label noise, leakage, distribution shift, a corrupted shard, class imbalance, an augmentation that destroys the label. The overfit test used 16 clean fixed examples; your real data has none of those guarantees.
- **The LR schedule and regularization.** Warmup too long, decay too aggressive, weight decay too high, dropout too strong. The test used a constant LR and no regularization; the full run uses neither.
- **Generalization itself.** The model overfits the training set (which the batch test *guaranteed* it can do) but does not generalize — that is a train/val gap story, a different post entirely. Overfitting one batch tells you nothing about generalization; it only tells you the machine *can* fit.

**If overfit-one-batch FAILS, the bug is in the model, loss, optimizer, or data path — and it is mechanical.** Do not go hunting for subtle distribution shift or a clever regularization fix. A model that cannot memorize sixteen examples has a hard, findable fault: a frozen module, a detached graph, LR zero, a loss-label mismatch, a NaN. Use the grad-norm print to localize it. This failure is never "the task is hard" — it is "the wiring is wrong."

**A smooth-then-NaN curve is numerics, not data.** If the overfit test trains fine for a while and then explodes to NaN, that is a numerics or LR signature (overflow, log of zero, too-high LR), not a data problem — the data was fine for the first hundred steps. Route to the NaN-hunting track, not the data track.

**A loss that floors at a small nonzero value (not chance, not machine-epsilon) on a tiny batch may be a label, not a model.** As the confident-learning case study showed, if one example in your batch is mislabeled, a correct model cannot zero its loss. Look at the specific examples whose loss stays high.

**If overfit-one-batch passes but your loss curve looks weird later, the test already did its job.** A sawtooth loss, a periodic spike every N steps, a plateau-then-drop — these are all *downstream* of "the model can learn," and a passing overfit test tells you to look at the dataloader, the schedule, or the data, not the model. The test is not a curve-shape diagnostic; it is a binary "is the machine wired correctly" gate. Once it passes, reading the loss curve's *shape* becomes the next instrument, and that is a separate skill.

Here is the routing in practice, as a worked decision rather than a rule list.

#### Worked example: routing a stalled finetune in three minutes

A finetune is stuck — the eval metric will not move past baseline after an epoch. Instead of randomly trying learning rates, you route:

1. **Run overfit-one-batch on four formatted examples.** Result: loss falls from 2.0 to 0.02 in 100 steps. *Pass.* This immediately rules out frozen adapter, detached graph, loss-masking-everything, LR zero, and wrong reduction. The model, loss, and optimizer path are sound. You have spent 40 seconds and eliminated half the bug space.
2. **Because it passed, you stop suspecting the model and look at what the test turned off.** The two big ones for a stalled finetune are the LR being too small for the *full* run (the test used a deliberately healthy LR) and the data not actually teaching anything new (the finetune set overlaps the pretraining distribution, so there is nothing to learn). You check the LR: it is `1e-6`, 10× too small for this model size. You also confirm the data is genuinely new.
3. **You raise the LR to `1e-5`, relaunch, and the eval metric moves.** Total debugging time: a few minutes, because the overfit test told you *where not to look* (the model is fine) and let you spend your attention on the schedule.

Compare that to the undisciplined path: try `2e-6`, wait an hour, no change; try a different optimizer, wait an hour, no change; suspect the data, spend a day auditing it; eventually stumble onto the LR. The overfit test does not solve the LR problem for you, but by cleanly clearing the model from suspicion it collapses a day of flailing into a focused few minutes on the right subsystem. That is the entire value proposition: not that the test fixes bugs, but that it tells you which half of the bug space to stop searching.

The point of all of this is to use the test to *route*, not just to pass or fail. It is the first branch of the bisection tree, and where it sends you is the whole value. The complete routing logic — every symptom, every suspect, every confirming test — is the [training-debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) that caps this series; the overfit test is the entry node.

## 12. Key takeaways

- **Before any serious run, prove the model can overfit one fixed batch.** If it cannot drive the loss toward zero on 8–16 examples, something is fundamentally broken and the full run is wasted. This is the highest-return 30 seconds in ML.
- **A loss stuck at $\ln(C)$ is the model guessing uniformly.** $\ln(10) = 2.303$, $\ln(2) = 0.693$. If your overfit test parks there, gradients are not reaching the part of the network that assigns class scores.
- **A pass rules out a whole class of bugs at once** — capacity, gradient flow, optimizer stepping, loss-label wiring — because overfitting is a conjunction and a pass asserts every term. Spend your remaining suspicion on data, schedule, regularization, and generalization.
- **A fail localizes to a few mechanical faults.** Frozen params, detached graph, LR zero, wrong reduction, label/logit mismatch. The per-module grad-norm print names the dead module in one backward pass.
- **The grad-norm-at-step-0 check is the single most useful line in the loop.** Zero global grad norm means no learning is possible; a tiny-but-nonzero norm with one dead module means a frozen or detached submodule.
- **Climb the ladder: one example, one batch, one class, one feature, one GPU, one step.** Each rung holds more of the system fixed, so a failure points at the subsystem that rung added. Single-GPU pass + multi-GPU fail = a distributed bug, not a model bug.
- **Beware the four false passes:** a label leak, BatchNorm in train mode, a too-easy synthetic batch, and augmentation silently still on. A pass is only honest if the batch is real, fixed, and not solvable by a leak or a normalization shortcut.
- **The test is both detector and confirmation.** You fix the bug and re-run the same loop to prove the fix — no full run required to validate it.
- **Use the result to route, not just to pass.** It is the first branch of the bisection tree: a pass sends you to data/schedule/generalization, a fail sends you to model/loss/optimizer.

## 13. Further reading

- **Andrej Karpathy, "A Recipe for Training Neural Networks" (2019)** — the canonical practitioner essay; lists "overfit one batch" among the first sanity checks after building a training skeleton, and articulates the make-it-fail-small philosophy this post is built on.
- **Zhang, Bengio, Hardt, Recht, Vinyals, "Understanding Deep Learning Requires Rethinking Generalization" (ICLR 2017)** — the random-labels experiment proving overparameterized networks can memorize arbitrary labelings; the theoretical foundation for why a tiny batch *should* be memorizable.
- **Northcutt, Athalye, Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks" (NeurIPS 2021)** and the `cleanlab` confident-learning library — why a stubborn small residual loss on a specific example may be a mislabel, not a model bug.
- **PyTorch Autograd documentation** — `tensor.register_hook`, `torch.autograd.set_detect_anomaly`, `requires_grad`, and how detaching cuts the graph; the API surface behind the grad-norm diagnostics here.
- **HuggingFace `peft` documentation, `print_trainable_parameters()`** — the one-call check for the LoRA "silent no-op," the most common modern frozen-module bug.
- **Within this series:** [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the master decision tree this test is the first branch of), [why your model isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think) (the full gradient-flow / frozen-params taxonomy), [loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs) (the wiring faults a failed overfit test localizes), [instrumenting a training run: what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) (what to watch once the model can learn at all), and the capstone [training-debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

---
title: "Train vs Eval Mode Bugs: The .eval() You Forgot to Call"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Diagnose the family of bugs where a single forgotten model.eval() or model.train() makes your validation number bounce, your finetune forget, and your inference quietly compute the wrong thing."
tags:
  [
    "debugging",
    "model-training",
    "batchnorm",
    "dropout",
    "finetuning",
    "deep-learning",
    "pytorch",
    "evaluation",
    "computer-vision",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/train-eval-mode-bugs-1.png"
---

The training loss curve was textbook. It fell from 2.30 to 0.04 over twenty epochs, smooth and confident. Then I ran the validation pass, and the accuracy came back 71%. I re-ran it: 66%. Again: 78%. Same checkpoint, same data, same code — three different numbers. For a model with frozen weights, that should be impossible. A network with fixed parameters, fed the exact same images, must produce the exact same predictions every time. It is a deterministic function. And yet my validation accuracy was rolling dice somewhere between 60% and 80% every time I called it.

I spent the better part of an afternoon hunting this in the data pipeline, convinced something was reshuffling the val set or leaking randomness through augmentation. It was none of that. The bug was one line I never wrote. I had never called `model.eval()` before the validation loop. So Dropout was still randomly zeroing half the activations of my classifier on every forward pass, and BatchNorm was still normalizing each batch with that batch's own statistics instead of the running averages it had accumulated over training. The "model" I was evaluating was not a fixed function at all — it was a different stochastic network on every call, and the metric was bouncing because the *computation* was bouncing. Figure 1 is the mechanism in one picture: a single boolean, `model.training`, silently rewires three families of layers, and forgetting to flip it changes what your network actually computes.

![Diagram showing how the model.training flag flips Dropout, BatchNorm, and other stateful layers between train-time and eval-time behavior, producing two different outputs for the same input](/imgs/blogs/train-eval-mode-bugs-1.png)

This post is about that entire family of bugs — the ones where the bug is not in your math, your data, or your optimizer, but in *which mode your layers think they are in*. They are some of the most common bugs in deep learning and some of the most embarrassing, because the fix is almost always a single method call. But they are worth a full deep-dive for two reasons. First, they fail *silently and non-deterministically*, which makes them maddening to chase if you do not know the signature — there is no exception, no NaN, just a number that will not sit still or a finetune that quietly refuses to learn. Second, the *why* is genuinely interesting numerics and statistics: understanding exactly what Dropout and BatchNorm do differently in each mode tells you precisely how the metric will be corrupted, in which direction, and by how much. Once you understand the mechanism you stop guessing and start asserting.

By the end you will be able to do five concrete things. First, derive *why* evaluating with Dropout on injects mean-zero noise into your metric, and *why* evaluating with BatchNorm in train mode makes the score depend on batch composition and leak information across the batch. Second, recognize all five distinct mode bugs from their instrument signatures — the forgotten `eval()`, the forgotten `train()` back, the `no_grad`-versus-`eval()` confusion, the cold-running-stats trap, and the freeze-BatchNorm-for-finetuning mistake. Third, write a mode-audit snippet that asserts `model.training` at the exact boundaries where it goes wrong, and a context manager that makes the eval path impossible to get wrong. Fourth, write a correct evaluation loop and the correct freeze-BatchNorm pattern, and prove each works with a before→after number rather than a vibe. Fifth, know precisely when a symptom is a mode bug and when it points elsewhere. These bugs live in the **model-code** and **evaluation** corners of the [six places a bug hides](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — data, optimization, model code, numerics, systems, evaluation — and the whole discipline is to read the signature before you start changing things.

A note on scope. This post is the *mode-mechanics* companion to the [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs) post, which covers the deeper statistics of BatchNorm — the small-batch variance cliff, the GroupNorm and SyncBN alternatives, LayerNorm placement. Here the focus is narrower and sharper: the two *modes* every stateful layer has, the boolean that selects between them, and the five ways that boolean gets set wrong. Where the science of *why a layer behaves differently* matters for the bug, we derive it; where it belongs to the normalization post, I will point there rather than re-derive it.

## 1. The science: what train mode and eval mode actually mean

Every PyTorch module carries a single boolean, `self.training`. It defaults to `True`. `model.train()` sets it to `True` on the module and, recursively, on every submodule; `model.eval()` sets it to `False` everywhere. That is the entire mechanism — one flag, propagated down the module tree. Most layers ignore it completely: a `nn.Linear`, a `nn.Conv2d`, a `nn.ReLU` computes exactly the same thing regardless of mode, because a matrix multiply or a pointwise nonlinearity has no notion of "training" versus "evaluation." The flag only matters for the small set of layers whose *forward computation is intentionally different at training and inference time*. There are essentially two such families that dominate real models — Dropout and BatchNorm — plus a handful of cousins (DropPath / stochastic depth, AlphaDropout, some RNN dropout). Understand those two precisely and you understand the whole class of bugs.

The reason these layers have two modes at all is not an implementation accident; it is a deliberate design where the train-time behavior is a *stochastic or batch-dependent approximation* of a clean inference-time behavior. The training-time version exists to regularize or to estimate statistics; the eval-time version exists to be deterministic and to use the accumulated knowledge. The bug is always the same shape: you run the train-time approximation when you wanted the clean inference behavior, or vice versa. To predict exactly how that corrupts your metric, we need the math of each.

### Dropout: the expectation argument

Dropout (Srivastava et al., 2014) regularizes by randomly zeroing each activation independently with probability $p$ during training. The intuition is that the network cannot rely on any single feature being present, so it learns redundant, robust representations. But there is a subtlety that creates the eval/train distinction: if you simply zero out a fraction $p$ of activations during training and then use *all* of them at test time, the expected magnitude of the signal entering the next layer is different in the two regimes — at test time it is roughly $1/(1-p)$ times larger than the network ever saw during training, which throws every downstream layer off its calibrated scale.

PyTorch (like every modern framework) solves this with **inverted dropout**, which puts the correction at train time. During training, each activation $a$ is replaced by

$$
\tilde{a} = \frac{m}{1-p}\, a, \qquad m \sim \text{Bernoulli}(1-p),
$$

where $m$ is $1$ with probability $1-p$ (the unit survives) and $0$ with probability $p$ (the unit is dropped). The survivors are scaled up by $1/(1-p)$ to compensate for the dropped fraction. Take the expectation over the random mask:

$$
\mathbb{E}[\tilde{a}] = \frac{a}{1-p}\,\mathbb{E}[m] = \frac{a}{1-p}\,(1-p) = a.
$$

So the *expected* train-time output equals the activation itself. That is the whole point: at eval time, Dropout becomes the identity — it passes every activation through unchanged, no mask, no scaling — and because of the $1/(1-p)$ scaling at train time, that identity pass equals the expectation of the train-time computation. The two modes agree in the mean. Figure 3 draws this: both modes flow into the same expectation, which is why a correctly written network has consistent scale across train and eval.

It is worth pausing on *why the scaling lives at train time* and not at test time, because that design choice is exactly what makes the forgotten-`eval()` bug behave the way it does. There are two arithmetically equivalent ways to keep the expectation matched. The original 2014 Dropout paper used **standard dropout**: drop with probability $p$ at train time with *no* scaling, then at test time multiply every activation by $(1-p)$ to shrink the larger test-time signal back down to the train-time scale. That puts an extra multiply on the test path. **Inverted dropout**, which every modern framework uses, moves the correction to the train path — scale survivors by $1/(1-p)$ during training — so that *the test path is a clean no-op*. The reason this matters for debugging: with inverted dropout, eval mode is literally "do nothing," so a correctly-configured network needs no special test-time arithmetic, and the only way to corrupt the eval is to accidentally run the *train* path. If frameworks had stuck with standard dropout, the forgotten-`eval()` bug would have produced a *systematic scale error* (a $1/(1-p)$ inflation of every activation) rather than mean-zero noise — a different and arguably easier-to-spot signature. The choice of inverted dropout is why the bug shows up as *variance*, not *bias*.

Now quantify the variance the bug injects, because "noise" is too vague to debug with. Consider a single activation $a$ entering an inverted-dropout layer at eval time, with the layer mistakenly in train mode. The output is $\tilde a = \frac{m}{1-p}a$ with $m \sim \text{Bernoulli}(1-p)$. Its variance is

$$
\text{Var}(\tilde a) = \frac{a^2}{(1-p)^2}\,\text{Var}(m) = \frac{a^2}{(1-p)^2}\,p(1-p) = a^2\,\frac{p}{1-p}.
$$

So the noise the bug injects into a single activation has standard deviation $|a|\sqrt{p/(1-p)}$. At $p=0.5$ that factor is $\sqrt{1}=1$ — the per-activation noise standard deviation *equals the activation magnitude itself*, which is enormous. At $p=0.1$ it is $\sqrt{0.111}\approx 0.33$, much milder. This is why the severity of the bouncing-metric symptom scales so steeply with the dropout rate: a dropout-0.5 layer corrupts each activation with noise as large as the signal, while a dropout-0.1 layer is a third of that. When you see a metric bouncing by several points, the dropout rate on the path to the output tells you roughly how big the bounce should be, and that is a useful consistency check — if you have only a 0.1 dropout and the metric is swinging 10 points, something *else* is also non-deterministic and dropout is not the whole story.

One more subtlety that trips people up: the noise does not simply average out across the validation set into a clean estimate of the true accuracy. You might hope that with 2,000 val examples, the per-example dropout noise washes out by the law of large numbers and the mean accuracy converges to the true number. It does *partly* — the standard error of the mean accuracy does shrink with the val-set size — but two effects keep it from being harmless. First, dropout noise is correlated *within* a forward pass because the same random mask is applied to a whole batch at once (PyTorch samples one mask per forward call per layer, shared across the batch dimension is *not* true — masks are per-element — but the point stands that a single eval pass is one draw from the noise distribution, not an average over many). Second, and more important, the noise is not a symmetric perturbation of the *accuracy*: pushing a confident correct prediction around rarely flips it, but pushing a borderline prediction flips it with significant probability, and those flips are not mean-zero in accuracy even though they are mean-zero in logits. The upshot is that the corrupted metric is both biased (usually downward, as the worked example shows) and high-variance, and the only clean fix is to remove the noise at the source by calling `eval()`.

![Diagram of dropout in train mode applying a Bernoulli mask scaled by one over one minus p, eval mode passing through identity, and both meeting at the same expected activation, with the eval-in-train-mode path branching off as a bug that adds variance](/imgs/blogs/train-eval-mode-bugs-3.png)

Now the bug. Suppose you evaluate with the model still in train mode. Dropout keeps sampling its Bernoulli mask. Your metric is no longer computed on $a$; it is computed on $\tilde{a}$, a random variable. Because $\mathbb{E}[\tilde{a}] = a$, the corruption is *mean-zero in the activations* — you are not introducing a systematic bias toward higher or lower scores, you are introducing **variance**. Each forward pass zeros a different random half of every dropout layer's activations, so the prediction for a given input is a noisy version of the true prediction, and the metric computed over the whole val set is itself a random variable with nonzero variance. That is *exactly* the signature I saw: a val accuracy that bounces around its true value without a consistent bias, 71% then 66% then 78%. The center of that cloud is roughly where the real eval-mode number is (66-78 straddling the true 84% imperfectly because a corrupted forward also shifts individual predictions across the decision boundary), but the spread is pure dropout noise that should not be there at all.

How big is the bounce? It scales with $p$ and with how many dropout layers sit on the path to the output. A single dropout-0.5 layer right before a softmax classifier can swing per-example logits enough to flip a few percent of borderline predictions, and across a small validation set (say 2,000 examples) the run-to-run standard deviation of accuracy can easily be a couple of points. With heavier dropout or several dropout layers stacked, it grows. The key qualitative fact for debugging: **a metric that bounces on a fixed checkpoint, with no consistent direction, and whose bounce shrinks toward zero the moment you call `model.eval()`, is dropout (or another stochastic layer) running at eval time.**

### BatchNorm: the statistics argument

BatchNorm (Ioffe and Szegedy, 2015) is the second and more dangerous half, because its two modes differ not just in *noise* but in *what statistics they use* — and one of those modes leaks information across the batch. During training, for a feature channel, BatchNorm computes the mean $\mu_\mathcal{B}$ and variance $\sigma_\mathcal{B}^2$ over the current mini-batch $\mathcal{B}$ and normalizes each activation with them:

$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta.
$$

Simultaneously it maintains an exponential moving average of these statistics — the **running mean** and **running variance** — updated each step as

$$
\mu_{\text{run}} \leftarrow (1-\eta)\,\mu_{\text{run}} + \eta\,\mu_\mathcal{B}, \qquad
\sigma^2_{\text{run}} \leftarrow (1-\eta)\,\sigma^2_{\text{run}} + \eta\,\sigma^2_\mathcal{B},
$$

for momentum $\eta$ (PyTorch default $\eta = 0.1$). At eval time, BatchNorm does *not* look at the batch at all. It normalizes every input with the frozen running statistics:

$$
\hat{x}_i = \frac{x_i - \mu_{\text{run}}}{\sqrt{\sigma_{\text{run}}^2 + \epsilon}}.
$$

The exponential moving average is itself worth understanding, because it explains both the warm-up bug (Bug 4) and how fast the stats forget. An EMA with momentum $\eta$ has an effective averaging window of roughly $1/\eta$ batches — with $\eta = 0.1$, about ten batches of "memory." More precisely, after $k$ updates the running mean is a weighted sum of the last $k$ batch means with geometrically decaying weights: the most recent batch gets weight $\eta$, the one before $\eta(1-\eta)$, and so on, with the initial value $\mu_{\text{run}}^{(0)} = 0$ retaining weight $(1-\eta)^k$. That initial-value weight is the warm-up story: after $k=10$ batches the init still carries weight $0.9^{10} \approx 0.35$, so a third of your running mean is still the zero prior; after $k=50$ batches it is $0.9^{50} \approx 0.005$, essentially gone. This is the precise reason you should not evaluate a model until a few dozen training batches have flowed through — the running statistics are a contaminated mixture of the data and the zero/one init until then. (PyTorch also offers `momentum=None`, which switches to a *cumulative* average over all batches seen, weighting every batch equally; that converges to the true statistics more slowly early on but is unbiased, and it is occasionally the right choice when you want a stable long-run estimate.)

This is a genuine difference in computation, and it has two consequences that produce two distinct bugs.

The first consequence is **batch dependence**. In train mode, the normalized value of example $i$ depends on the mean and variance of the *whole batch* — that is, on the other examples that happen to be in the batch with it. If you evaluate in train mode, your prediction for a single image depends on which other images are in its batch. Reorder the validation set, change the batch size, and the predictions change. This is the form of the bug for any BatchNorm network: the metric is not just noisy, it is **a function of batch composition**, which means it is non-reproducible across shuffles and, worse, it constitutes a subtle kind of test-time information leak — the model is "seeing" the statistics of the test batch it is being evaluated on. In a setting where test examples should be independent (and they almost always should be), this is a correctness violation, not just noise.

How large is this batch-dependence noise, quantitatively? The batch mean $\mu_\mathcal{B}$ over $m$ examples is an estimate of the true channel mean $\mu$ with standard error $\sigma/\sqrt{m}$, and the batch variance estimate has its own sampling error of order $\sqrt{2/(m-1)}$ in relative terms (the variance of a sample variance for roughly-Gaussian data). At batch size $m=64$ that variance estimate is off by about $\sqrt{2/63}\approx 18\%$ from the true value on any given batch, and at $m=16$ it is about $\sqrt{2/15}\approx 37\%$. So evaluating a BatchNorm network in train mode normalizes each example by a denominator that is randomly $\pm 18\%$ to $\pm 37\%$ wrong depending on the batch — and crucially, *which* wrong value depends on the batch composition, so reshuffling changes it. That is the mechanism behind a metric that bounces by several points across eval runs at small batch size. It is also why the bouncing gets *worse* as you shrink the eval batch size: the smaller the batch, the noisier the per-batch statistics, the more the predictions swing. A quick confirming test that is almost diagnostic on its own: run the (incorrectly-train-mode) eval at batch size 256 and again at batch size 8, and watch the bounce grow as the batch shrinks. A correctly-`eval()`-ed model, by contrast, gives *identical* numbers at every batch size, because it never touches batch statistics. That batch-size invariance is one of the cleanest tells that your eval mode is set correctly.

The "leak" framing deserves precision because it is the part that turns a noise problem into a correctness problem. When BatchNorm normalizes example $i$ with $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}$ computed over the batch, the output for $i$ literally contains information about the other examples in the batch — their feature values entered the mean and variance. If your evaluation is supposed to score each example independently (the standard assumption), this is information flowing between test examples that should be isolated. In benign cases it is just noise; in adversarial or near-adversarial cases it can be exploited (a model can behave differently when a known set of examples is batched together), and in benchmark settings it makes your reported number depend on an arbitrary batching choice. The clean statement: eval-in-train-mode breaks the independence of test predictions, and the only way to restore it is `model.eval()`, which makes each example's normalization depend solely on the frozen running statistics and not at all on its batch-mates.

The second consequence is the opposite-direction bug: if you forget to put the model *back* into train mode after an eval pass, BatchNorm stops updating its running statistics *and* stops using batch statistics for the forward pass during what is supposed to be training. The running stats freeze at whatever they were, the layers normalize training batches with stale eval-mode statistics, and learning degrades in a way that looks like a mysterious optimization stall. Figure 4 lays out this whole train-versus-eval-versus-bug grid layer by layer, which is the lookup table I keep in my head.

![Matrix figure mapping Dropout, BatchNorm, running statistics, and the no-grad flag across train-mode behavior, eval-mode behavior, and the wrong-mode bug each one produces](/imgs/blogs/train-eval-mode-bugs-4.png)

> The provable point: the eval/train distinction is not cosmetic. Dropout's two modes agree in *expectation* but differ in *variance*, so eval-in-train-mode injects mean-zero noise into your metric. BatchNorm's two modes differ in *which statistics they use*, so eval-in-train-mode makes the metric depend on batch composition and leak across the batch, while train-stuck-in-eval-mode freezes the running statistics and starves learning. Knowing which layer you have tells you exactly which signature to expect.

#### Worked example: the bouncing validation accuracy

Here is the run from the intro, made concrete. A ResNet-18 finetuned on a 10-class image dataset, batch size 64 for validation, a held-out val set of 2,000 images. The model has BatchNorm throughout (it is a ResNet) and a dropout-0.5 layer I added before the final linear head. After training, the train loss is 0.04 and I expect a val accuracy near 84%.

Evaluating with the model left in train mode, across five runs I get: 71.2%, 66.8%, 78.1%, 69.5%, 74.0% — mean about 71.9%, standard deviation about 4.0 points, with the variance coming from both sources. The dropout layer randomizes the head's predictions; the BatchNorm layers normalize each batch of 64 with that batch's own statistics, so the per-image prediction depends on its 63 batch-mates, and shuffling between runs changes the predictions. The number is not just noisy — it is consistently *pessimistic*, because the running statistics BatchNorm accumulated over thousands of training batches are a far better estimate of the true feature distribution than the statistics of any single 64-image batch, and the dropout noise only ever hurts a well-calibrated head. So eval-in-train-mode here is biased low *and* high-variance.

Add one line — `model.eval()` before the loop, and `model.train()` after — and the five runs become: 84.3%, 84.3%, 84.3%, 84.3%, 84.3%. Identical to the decimal, because the forward pass is now a deterministic function: dropout is the identity, BatchNorm uses the same frozen running stats every time, and batch composition is irrelevant. Same weights, same data, same code. The 12-point gap between the bouncing ~72% and the stable 84% is the entire cost of the missing method call. Figure 2 is that before→after.

![Two-column comparison of evaluating one checkpoint in train mode versus eval mode, showing dropout active versus identity, batch statistics versus running statistics, and a bouncing sixty-to-eighty percent accuracy versus a stable eighty-four percent](/imgs/blogs/train-eval-mode-bugs-2.png)

That direction — eval-in-train-mode being *pessimistic* — is the common case for a well-trained network, but it is not a law. If your batch statistics happen to be cleaner than your running statistics (for example, you trained with tiny noisy batches but evaluate with large clean ones), eval-in-train-mode can be *optimistic* instead, flattering the model. That is even more dangerous, because an optimistically-biased val number is the kind that ships a broken model. The only safe rule is: the eval number is whatever `model.eval()` produces, and anything else is corrupted in an unknown direction.

## 2. The diagnostic: assert the mode where it goes wrong

The fix for these bugs is trivial once you know which one you have; the skill is *catching them*, because they throw no exception. The single most effective diagnostic is to **assert `model.training` at the boundaries where it gets set wrong** — entering and leaving validation. This is the make-it-fail-small principle applied to mode: instead of debugging a corrupted metric, you turn a silent mode error into a loud assertion at the exact line responsible.

```python
import torch

def assert_eval(model):
    """Call at the top of every validation/inference function.

    Fails loudly if any submodule is still in training mode, which is the
    single most common cause of a bouncing or pessimistic val metric.
    """
    bad = [name for name, m in model.named_modules() if m.training]
    assert not model.training, "model is in TRAIN mode during eval"
    assert not bad, f"these submodules are still in train mode: {bad[:8]}"

def assert_train(model):
    """Call at the top of the training step, after re-entering the loop."""
    assert model.training, "model is in EVAL mode during training"
```

The per-module version matters because of a real failure mode: you can have a model whose top-level `.training` is `False` but a *sub-module* stuck in train mode, if some code path called `.train()` on a child after you called `.eval()` on the parent, or if a custom module overrode `train()` incorrectly. `model.named_modules()` walks the whole tree, so the assertion catches the case where one BatchNorm deep inside a frozen backbone is misbehaving. In practice I print the offending module names so I can see *exactly* which layer is wrong, not just that something is.

There is a complementary signal worth logging continuously: the *count* of modules in train mode. In a healthy training step that count equals the total module count; in a healthy eval it is zero; in the freeze-BatchNorm pattern (Bug 5) it is a small, *constant* nonzero number during training (exactly the frozen BatchNorm layers). The power of logging the count rather than just the top-level flag is that a *change* in the count is itself a bug signal: if your frozen-BN finetune suddenly shows a different count of train-mode modules between epochs, something re-flipped your BatchNorm layers and the freeze is broken. A one-line helper makes this loggable.

```python
def count_training_modules(model):
    """Return (n_train_mode, n_total). Log both; a healthy eval has 0 train,
    a healthy train has all, and a frozen-BN finetune has a small CONSTANT
    nonzero train count (exactly the frozen BatchNorms)."""
    mods = list(model.modules())
    n_train = sum(m.training for m in mods)
    return n_train, len(mods)

# In a frozen-BN finetune this should print the SAME nonzero number every epoch.
# A change means model.train() un-froze a BatchNorm you meant to keep frozen.
n_train, n_total = count_training_modules(model)
print(f"train-mode modules: {n_train}/{n_total}")
```

This count is also the cheapest possible regression test for the whole family: snapshot the expected count at the start of training and validation, assert it stays constant across epochs, and any mode bug that flips a layer at the wrong time fails the assertion immediately rather than silently corrupting a metric forty epochs later.

A second, even more robust pattern is to make the eval path impossible to get wrong with a context manager that sets eval mode, disables gradients, and *guarantees* the model is restored to its prior mode on exit — even if the eval loop raises an exception partway through. The naive "call `eval()` then `train()`" pattern has a sharp edge: if validation throws (an OOM, a bad batch), the `model.train()` line never runs, and your *next* training epoch silently runs in eval mode. The context manager closes that hole.

```python
import torch
from contextlib import contextmanager

@contextmanager
def evaluating(model):
    """Switch to eval + no_grad for the block, restore the prior mode on exit.

    Restores even if the block raises, so a crash mid-validation cannot leave
    the model stuck in eval mode for the next training epoch.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield model
    finally:
        model.train(was_training)  # restore exactly what it was before

# Usage — the eval path is now correct by construction:
with evaluating(model):
    acc = run_validation(model, val_loader)
```

This single helper eliminates three of the five bugs at once: it sets eval mode (kills the forgotten-`eval()` bug), it restores the prior mode in a `finally` block (kills the forgotten-`train()`-back bug), and it wraps `no_grad` (kills the eval-without-no_grad memory bug). I put this in every project's utilities file and never write a bare `model.eval()` in a training script again.

For continuous monitoring rather than point assertions, log the mode flag as a scalar to your tracker so a mode regression shows up on the dashboard the moment it happens:

```python
# Inside the training loop, log the mode as a 0/1 signal each step.
wandb.log({"is_training": int(model.training)}, step=global_step)
# Inside validation, the same — it should read 0 there.
# A sawtooth that should be 1-during-train / 0-during-val but reads 1 in val
# is the forgotten-eval() bug, visible at a glance.
```

The "metrics jump around between evals" tell deserves its own line, because it is the highest-signal symptom in the whole family. If you evaluate the *same checkpoint twice* and get two different numbers, you have a non-determinism source in your eval path, and by far the most common one is a stochastic or batch-dependent layer running in train mode. The diagnostic is one line: evaluate twice and compare.

```python
# The cheapest possible mode-bug test: a fixed checkpoint must score identically.
a = run_validation(model, val_loader)
b = run_validation(model, val_loader)
assert abs(a - b) < 1e-6, f"non-deterministic eval ({a} vs {b}) -> check .eval() / dropout / BN"
```

If that assertion fires, you have proven the eval path is non-deterministic before you have done any other work, and the prime suspect is mode. (The secondary suspects — non-deterministic CUDA kernels, a shuffling val loader — are covered in [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training); but mode is the first thing to rule out because it is the most common and the cheapest to check.)

## 3. The five distinct bugs and their signatures

There is not one mode bug; there are five, and they have *different* signatures, so the symptom alone routes you to the culprit. Figure 5 is the decision tree I run mentally; the rest of this section walks each branch.

![Tree diagram routing a mode-bug symptom to one of five distinct bugs, splitting first on whether the failure shows at evaluation or during training and then to the specific forgotten-eval, cold-stats, no-grad-confusion, forgotten-train, and freeze-batchnorm bugs](/imgs/blogs/train-eval-mode-bugs-5.png)

### Bug 1: forgot model.eval() at validation or inference

This is the headline bug. Symptom: validation or test metric is non-deterministic and/or worse than the training trajectory suggests, on a fixed checkpoint. Mechanism: dropout is sampling and/or BatchNorm is using batch statistics during eval. Confirming test: the evaluate-twice assertion above fires, or `assert_eval(model)` reports modules still in train mode. Fix: `model.eval()` before the eval loop (the context manager above is the robust version). This is the bug from the intro and the worked example: ~72% bouncing becomes 84% stable.

### Bug 2: forgot model.train() back after eval

The mirror image, and sneakier because it corrupts *training*, not eval — which means you do not notice it in the validation metric (that part is now fine), you notice it as a training stall. Symptom: training loss plateaus or descends much slower than it should, after the first validation pass, with no other change. Mechanism: after `model.eval()`, you never called `model.train()`, so for the rest of training Dropout is the identity (no regularization — usually harmless to loss, sometimes lets it overfit faster) and, far more damaging, BatchNorm stops updating its running statistics *and* normalizes training batches with the frozen stats instead of batch stats. The network is now training through a normalization layer that is not adapting, and the running statistics it will use at the *next* eval are frozen at an early, poorly-estimated value. Confirming test: `assert_train(model)` at the top of the training step fires after the first eval. Fix: `model.train()` at the start of each training epoch (or, again, the context manager that restores automatically). The signature — "training was fine until the first eval, then the loss curve bent" — is the giveaway, and it is precisely why the timeline in Figure 7 marks *both* boundaries of every epoch as bug sites.

The reason this degrades learning rather than just removing regularization is worth spelling out, because it is the part that surprises people who think "BatchNorm in eval mode during training is harmless — it just uses fixed stats." It is not harmless. When BatchNorm normalizes a *training* batch with stale running statistics instead of that batch's own statistics, two things break. First, the normalization is no longer mean-zero and unit-variance for the actual batch — the activations entering the next layer are off-center and off-scale, which is precisely the condition BatchNorm exists to prevent, so the optimization landscape degrades exactly as if you had removed BatchNorm. Second, and worse, the gradient that flows back through a frozen-stats BatchNorm is *different* from the gradient through a batch-stats BatchNorm: the batch-statistics computation is part of the graph in train mode and contributes to the gradient (the famous BatchNorm backward involves the batch mean and variance), but in eval mode the statistics are constants with no gradient path, so the layer's Jacobian changes. The network is now being trained with a different effective architecture than the one you will evaluate, which is a quiet train/eval skew on top of the optimization damage. The loss curve bending after the first eval is the visible symptom of both effects at once.

![Timeline figure marking the two boundaries of each epoch where mode is forgotten, entering validation without eval and returning to training without train, with assertions placed at each boundary](/imgs/blogs/train-eval-mode-bugs-7.png)

### Bug 3: torch.no_grad() versus eval() confusion

This is the conceptual bug, and it is worth dwelling on because the two operations are *orthogonal* and people routinely assume one implies the other. They control completely different things:

- `model.eval()` changes **layer behavior**. It flips the `training` flag so Dropout becomes identity and BatchNorm uses running stats. It does *nothing* to the autograd graph — gradients are still tracked, the graph is still built, memory for the backward pass is still allocated.
- `torch.no_grad()` changes **graph construction**. It tells autograd not to record operations, so no computation graph is built and no gradient buffers are allocated. It does *nothing* to layer behavior — dropout still drops, BatchNorm still uses batch stats, if you are in train mode.

Figure 6 makes the orthogonality concrete. Use only `no_grad()` at inference and you get a *correctness* bug: dropout and BatchNorm still run in train mode, so your metric is corrupted exactly as in Bug 1, you just saved some memory while computing the wrong thing. Use only `eval()` at inference and you get a *memory and speed* bug: the math is correct, but autograd is still building the full computation graph and holding every intermediate activation for a backward pass that never comes, so your inference uses roughly twice the memory it needs and can OOM on inputs that would otherwise fit. At inference you almost always want **both**: `eval()` for correct layer behavior and `no_grad()` (or the newer `torch.inference_mode()`, which is `no_grad` plus a few more optimizations) for memory and speed.

![Before-after figure contrasting using only no-grad or only eval at inference against using both, showing the correctness bug from dropout staying active and the memory bug from the graph still being built](/imgs/blogs/train-eval-mode-bugs-6.png)

```python
# WRONG: saves memory, but dropout/BN run in train mode -> corrupted metric
with torch.no_grad():
    logits = model(x)        # model.training is still True!

# WRONG: correct math, but full graph is built -> ~2x memory, OOM risk
model.eval()
logits = model(x)            # no no_grad, autograd still recording

# RIGHT: correct behavior AND no graph. inference_mode is the modern default.
model.eval()
with torch.inference_mode():
    logits = model(x)
```

The reason this confusion is so persistent is that in a *plain* network with no Dropout and no BatchNorm (a vanilla MLP or a CNN with only LayerNorm/GroupNorm, which are mode-independent), `eval()` is genuinely a no-op and `no_grad()` alone gives the correct answer — so people develop a habit on such models that breaks the moment they add a BatchNorm or a Dropout. The habit to build instead: at inference, always both, always.

#### Worked example: the two halves of Bug 3, in two instruments

Make the orthogonality concrete with a single transformer-classifier forward pass, batch of 32 sequences of length 512, a model with attention and residual dropout at $p=0.1$, served on a 24 GB GPU. Run the inference four ways and read two instruments — the metric (does it repeat?) and `torch.cuda.max_memory_allocated()` (how much memory did the pass hold?).

With *neither* `eval()` nor `no_grad()`: the metric bounces (dropout active) *and* the memory is high (graph built) — peak memory around 18 GB because every intermediate activation is retained for a backward that never comes, and a second run gives a different accuracy. With *only* `no_grad()`: memory drops to about 9 GB (no graph, no saved activations), but the metric *still bounces* run to run, because dropout is still sampling — this is the correctness half of the bug, and it is the dangerous one because the low memory makes you think you did it right. With *only* `eval()`: the metric is now stable and correct, but memory is back up at 18 GB, and on a longer input this is the run that OOMs — the memory half of the bug. With *both* (`eval()` plus `inference_mode()`): the metric is stable at the correct value and memory sits at about 9 GB. The diagnostic lesson is the punchline: the two halves of Bug 3 move *different* instruments. If your numbers are wrong but memory is fine, you forgot `eval()`. If your numbers are right but you OOM, you forgot `no_grad`. You never have to guess — read which instrument moved. The roughly 2x memory factor (18 GB versus 9 GB here) is approximate and depends on depth: it is the ratio of saved-activation memory to the rest, which grows with the number of layers, so a deeper model sees a bigger gap and a shallow one a smaller gap.

### Bug 4: BatchNorm running stats not warmed up

This one is a timing bug rather than a mode bug strictly, but it shares the family because it manifests as a bad eval despite correct `eval()` usage. Symptom: you evaluate a model right after initialization (or very early in training), with `model.eval()` correctly called, and the eval is garbage — much worse than even a randomly-initialized model should be, sometimes producing NaNs. Mechanism: at eval, BatchNorm uses its running statistics. At initialization those are *not the true data statistics* — PyTorch initializes `running_mean = 0` and `running_var = 1`, which is correct as a prior but is only refined toward the real statistics as training batches flow through in *train* mode. If you evaluate before enough training steps have updated the running stats, BatchNorm normalizes with a wrong mean and variance, and the output is badly miscalibrated. With momentum 0.1 it takes on the order of a few dozen batches for the running stats to converge to a usable estimate. Confirming test: print `bn.running_mean` and `bn.running_var` and check they have moved away from the 0/1 init; if they are still near 0/1, the stats are cold. Fix: do not evaluate before the running stats have warmed up (a handful of forward passes in train mode is enough), or, if you genuinely need to evaluate an untrained model, do a few warmup forward passes in train mode first, or use a normalization that does not depend on running stats (GroupNorm/LayerNorm — see the [normalization post](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs)). This is also the reason a `load_state_dict` that drops the BatchNorm buffers (they are buffers, not parameters) produces a model that trains fine but evaluates terribly — the running stats were not restored.

#### Worked example: a checkpoint that loads but evaluates as noise

A subtle and common version of Bug 4 has nothing to do with timing — it is a *serialization* bug that produces cold stats at load time. You train a ResNet-50, reach 91% val accuracy, save the model, ship it. A colleague loads it and gets 11% — almost exactly chance for ten classes. The weights are identical, the architecture matches, no exception is raised. The cause: the checkpoint was saved or loaded in a way that dropped the BatchNorm *buffers*. In PyTorch, $\gamma$ and $\beta$ are *parameters* (they appear in `model.parameters()`), but `running_mean` and `running_var` are *buffers* (registered with `register_buffer`, not parameters). If you saved only `model.parameters()` instead of `model.state_dict()`, or loaded a state dict with `strict=False` that silently skipped the buffer keys because of a name mismatch, the running statistics come back at their 0/1 init while every weight is correct. At eval, BatchNorm then normalizes with mean 0 and variance 1 — the prior, not the learned statistics — and the features are wildly miscalibrated, so the classifier reads noise. The accuracy of 11% is the tell: not zero, not high, but pinned near chance, with correct weights. Confirming test, two lines: after loading, print `model.layer1[0].bn1.running_var.mean()`; if it reads ~1.0 the buffers are cold, and you have your answer. The fix is to always serialize with `model.state_dict()` (which includes buffers) and load with `strict=True` so a missing buffer key raises instead of silently defaulting. The cost of getting this wrong is the entire model — a 91%-to-11% collapse — from a single buffer that did not make the trip, and it is invisible until you check the one statistic.

### Bug 5: freezing BatchNorm for finetuning — eval mode during training

This is the most subtle and the most important for finetuning, because here you *want* a BatchNorm layer in eval mode *during training*, and getting it wrong silently destroys a pretrained backbone. The setup: you are finetuning a pretrained network (a ResNet backbone, say) on a small dataset, and you want to freeze the backbone's BatchNorm layers so they keep the statistics they learned on the large pretraining dataset. If you let them update their running stats on your small, possibly non-representative finetuning batches, those carefully-learned statistics drift toward your tiny dataset's quirks, and the pretrained features degrade — sometimes catastrophically. Figure 8 is the correct pattern and the naive trap side by side.

![Graph figure showing that model.train() flips all BatchNorm layers back to train mode including frozen ones, that the naive requires-grad-false-only freeze leaves running stats drifting, and that re-applying bn.eval after train keeps pretrained statistics intact](/imgs/blogs/train-eval-mode-bugs-8.png)

The trap: people freeze a layer by setting `requires_grad=False` on its parameters, which stops the *affine* parameters $\gamma, \beta$ from being updated by the optimizer. But `requires_grad=False` does *nothing* to the running statistics — those are updated in the forward pass whenever the layer is in train mode, with no gradient involved. So a BatchNorm "frozen" by `requires_grad=False` alone is *still updating its running mean and variance* on every training batch, because `model.train()` (which you call to train the rest of the network) put it back in train mode. The running stats drift, the features rot, and your finetune underperforms for no visible reason. The fix has two parts: freeze the affine parameters with `requires_grad=False` *and* keep the layer in eval mode during training so it neither updates nor uses batch statistics. And critically, you must re-apply the eval mode *after every `model.train()` call*, because `model.train()` indiscriminately flips every submodule back to train mode.

```python
import torch.nn as nn

def freeze_batchnorm(model):
    """Put every BatchNorm in eval mode and freeze its affine params.

    Must be called AFTER model.train(), because model.train() flips every
    submodule (including these) back into train mode. Running stats stay
    pretrained; gamma/beta stop updating.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()                       # use + freeze running stats
            module.weight.requires_grad_(False) # freeze gamma
            module.bias.requires_grad_(False)   # freeze beta

# Correct finetuning loop:
for epoch in range(num_epochs):
    model.train()          # flips EVERYTHING to train, including the BNs
    freeze_batchnorm(model)  # ...so re-freeze the BNs right after, every epoch
    for batch in train_loader:
        ...
```

The "after every `model.train()`" detail is the part everyone misses. They call `freeze_batchnorm(model)` once, at setup, then call `model.train()` at the top of each epoch, which silently un-freezes every BatchNorm. The frozen-BN pattern only holds if the re-freeze happens *inside* the epoch loop, after `train()`. This is the same hazard as Bug 2 but in reverse: there, `train()` was forgotten; here, `train()` does too much and you have to undo part of it.

#### Worked example: the finetune that quietly forgot its backbone

Concrete numbers. Finetune a pretrained ResNet-50 (ImageNet) on a 5-class medical-imaging dataset of 1,200 images, batch size 16. I freeze the backbone with `requires_grad=False` on all backbone parameters, train only a new head, and expect the strong pretrained features to give me high accuracy fast.

With the naive freeze (params only), the BatchNorm layers stay in train mode because I call `model.train()` each epoch. On batches of 16 images from a narrow medical distribution, the running statistics drift away from the broad ImageNet statistics they were pretrained with. After 10 epochs the val accuracy is 73%, and — the tell — it is *unstable across epochs*, swinging several points, because the running stats are a moving target. I assumed the head just needed more epochs.

Adding `freeze_batchnorm(model)` after each `model.train()` — eval mode on every BN plus frozen affine params — keeps the ImageNet statistics intact. The same backbone, same head, same data: val accuracy climbs to 88% in the same 10 epochs and is stable epoch-to-epoch. The 15-point gain is entirely from *not* letting 1,200 medical images overwrite statistics estimated from 1.28 million ImageNet images. The diagnostic that would have caught it in one line: print `backbone.layer1[0].bn1.running_mean.mean()` at epoch 0 and epoch 10 and watch it drift in the naive version and sit still in the fixed version.

## 4. A correct evaluation loop, end to end

Putting the pieces together, here is the eval loop I actually use. It uses the context manager so the mode is correct by construction and restored on exit, asserts the mode for defense in depth, and is the version I would copy into a new project.

```python
import torch
from contextlib import contextmanager

@contextmanager
def evaluating(model):
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():   # eval behavior + no graph, in one block
            yield model
    finally:
        model.train(was_training)

@torch.no_grad()  # belt-and-suspenders; inference_mode below is the real guard
def validate(model, loader, device):
    with evaluating(model):
        # Defense in depth: prove we are actually in eval before scoring.
        assert not model.training, "validate() entered with model in train mode"
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += torch.nn.functional.cross_entropy(
                logits, y, reduction="sum"
            ).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    # On exit, the context manager restored the model to its prior mode.
    return loss_sum / total, correct / total

# Training driver — note model.train() at the top of every epoch.
for epoch in range(num_epochs):
    model.train()
    assert model.training
    for x, y in train_loader:
        ...  # forward / backward / step
    val_loss, val_acc = validate(model, val_loader, device)
    print(f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
```

Three properties make this correct and hard to break. The `evaluating` context manager sets eval mode and restores the prior mode in a `finally`, so a crash mid-validation cannot leave the model in eval mode for the next epoch. `torch.inference_mode()` gives both correct layer behavior (via `eval()`) and no graph building, so the metric is right *and* cheap. And the `assert not model.training` inside the loop turns any future regression — someone adding a stray `model.train()` somewhere — into a loud failure at the responsible line instead of a quietly wrong number. The `cross_entropy(..., reduction="sum")` then divide-by-total, rather than averaging per-batch means, is a small extra correctness detail so the last partial batch does not get over-weighted; that family of reduction bugs is its own topic, but it pairs naturally with getting the eval loop right.

### A mode-audit you can run on any checkpoint

When you inherit a model or suspect a mode bug in code you did not write, this audit dumps the mode and BatchNorm health of every relevant layer in one pass:

```python
import torch.nn as nn

def mode_audit(model):
    """One-shot report: which stateful layers are in which mode, and whether
    BatchNorm running stats look warmed up (moved away from the 0/1 init)."""
    print(f"top-level model.training = {model.training}")
    for name, m in model.named_modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.AlphaDropout)):
            print(f"  [dropout]  {name:40s} training={m.training} p={m.p}")
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            rm = m.running_mean.abs().mean().item() if m.running_mean is not None else None
            rv = m.running_var.mean().item() if m.running_var is not None else None
            cold = (rm is not None and rm < 1e-4 and abs((rv or 1) - 1) < 1e-4)
            warn = "  <-- COLD STATS?" if cold else ""
            print(f"  [BN]       {name:40s} training={m.training} "
                  f"|mean|~{rm:.4f} var~{rv:.4f}{warn}")

# mode_audit(model) right before validation tells you in one glance whether
# any layer is in the wrong mode and whether BN stats are warmed up.
```

This single function answers three of the five bugs directly: a dropout reading `training=True` during eval is Bug 1; a BatchNorm reading `training=True` during eval is Bug 1 (or, in a finetune, the *correct* Bug-5 freeze); a BatchNorm with `|mean|~0 var~1` flagged COLD is Bug 4. I run it once before the first validation of any new project and any time an eval number looks wrong.

## 5. The before→after evidence, consolidated

To make the cost of these bugs legible, here is the consolidated before→after across the failure modes, with the symptom value, the confirming test, the fix, and what the instruments read afterward. Every number traces to the worked examples above or to the mechanism; where a figure is approximate I say so.

| Bug | Symptom (before) | Confirming test | Fix | After |
| --- | --- | --- | --- | --- |
| Forgot `eval()` | val acc bounces 66-78% on a fixed ckpt | evaluate twice, numbers differ | `model.eval()` / `evaluating()` | stable 84.3% every run |
| Forgot `train()` back | train loss bends/plateaus after first eval | `assert model.training` fires in train step | `model.train()` per epoch | loss resumes normal descent |
| `no_grad` only | metric corrupted, low memory | `assert_eval(model)` lists train-mode layers | add `model.eval()` | correct, deterministic metric |
| `eval()` only | correct metric, ~2x memory, OOM | watch `max_memory_allocated` | wrap `inference_mode()` | memory drops ~2x |
| Cold BN stats | eval garbage right after init | `running_mean~0, var~1` | warm up / GroupNorm | stats move, eval sane |
| Freeze-BN wrong | finetune val ~73%, unstable | `running_mean` drifts across epochs | `bn.eval()` after each `train()` | stable 88% |

How to *measure* these honestly: for the determinism ones, the evaluate-twice test is exact — a fixed-weight model is a deterministic function, so any difference between two eval passes is a bug, full stop. For the memory one, `torch.cuda.max_memory_allocated()` before and after wrapping `inference_mode()` gives the real delta; the "~2x" is approximate and depends on model depth (it is the size of the saved activations relative to the parameters), so I report it as a range, not a constant. For the finetune one, the honest measurement is a paired comparison: same seed, same data order, only the freeze pattern changed, val accuracy at a fixed epoch — that isolates the BatchNorm freeze as the only variable, which is the make-it-fail-small discipline applied to a finetune.

## 6. Case studies and real signatures

A few well-known patterns, accurately attributed, to anchor the mechanism in the literature and in production folklore.

**The forgotten-`eval()` bug is the canonical PyTorch FAQ.** It is not in a paper because it is too routine to publish, but it is one of the most frequently reported issues on the PyTorch forums and is called out explicitly in the official documentation: `Module.eval()` "sets the module in evaluation mode," which "is equivalent to `self.train(False)`," and the docs state plainly that this affects `Dropout` and `BatchNorm`, and that `model.eval()` is *distinct from* `torch.no_grad()`. The recommended pattern in the PyTorch docs is exactly the one in Section 4: `model.eval()` plus a `no_grad`/`inference_mode` block for evaluation. The signature — a metric that is non-deterministic on a fixed checkpoint — is unmistakable once you know to look for it, and the "evaluate twice" test confirms it in seconds.

**The disharmony between Dropout and BatchNorm.** Li et al. (2019), "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift," analyzed what happens when Dropout and BatchNorm are stacked: Dropout changes the variance of its output between train and test (it is designed to preserve the mean, not the variance), and BatchNorm's running variance is estimated in train mode but applied in test mode, so the variance the BatchNorm sees at test does not match what it accumulated at train. The result is a "variance shift" that degrades test accuracy, and it is worst when a Dropout sits immediately before a BatchNorm. The practical lesson for this post: the train/eval distinction is not just about *forgetting* a mode — even when you call `eval()` correctly, the *interaction* of two stateful layers can leave a residual train/test gap, which is why modern architectures often avoid placing Dropout directly before BatchNorm. This is the rigorous backing for treating Dropout and BatchNorm as one coupled mode-sensitive system.

**Frozen BatchNorm in detection and finetuning.** It is standard practice in object-detection codebases (the Detectron / Mask R-CNN lineage) to *freeze* BatchNorm entirely when finetuning a backbone on a small detection dataset — both the affine parameters and the running statistics — precisely because detection batches are small (a handful of high-resolution images per GPU) and the per-batch statistics are too noisy and too unrepresentative to update the pretrained stats safely. Many of these codebases ship a dedicated `FrozenBatchNorm2d` module that hard-codes eval-mode behavior so it *cannot* accidentally update or use batch statistics regardless of the model's mode. That a major research lineage built a whole module to make this bug impossible tells you how often the naive `requires_grad=False`-only freeze burned people — it is the productionized version of the Section 3, Bug 5 fix.

**BatchNorm's batch-size sensitivity.** Wu and He (2018), "Group Normalization," quantified how BatchNorm's accuracy degrades as batch size shrinks, because the per-batch statistics become unreliable estimates of the true distribution. While that paper is primarily about the *small-batch* failure (covered in the [normalization post](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs)), it is directly relevant here: the reason evaluating in train mode is so corrupting is the same reason small-batch BatchNorm is fragile — both make the computation depend on an unreliable, batch-composition-dependent statistic instead of the stable running average. GroupNorm sidesteps the entire train/eval mode issue for normalization because it computes statistics per-image, identically at train and test, with no running stats and no mode dependence at all. If your model uses GroupNorm or LayerNorm throughout and has no Dropout, the forgotten-`eval()` bug literally cannot affect your metric — a useful fact when bisecting whether a non-deterministic eval is even a mode problem.

**The MC-Dropout exception that proves the rule.** There is exactly one well-known case where you *deliberately* keep Dropout active at inference: Monte-Carlo Dropout (Gal and Ghahramani, 2016), where you run the model in train mode (or selectively enable only the dropout layers) for many forward passes at test time and use the *spread* of the predictions as an estimate of the model's epistemic uncertainty. This is the same mechanism as the forgotten-`eval()` bug — dropout noise at inference — but turned into a feature: the bouncing predictions are not corruption, they are the signal. The reason this is worth knowing as a debugger is that it sharpens the boundary of what is a bug. The variance dropout injects at eval is *real information* about model uncertainty; the bug is not that the variance exists, it is that for a *point estimate* of accuracy you want the mean (which is what `eval()` gives, exactly, in one pass) and not a single noisy sample. So when you see someone running inference in train mode on purpose, ask whether they are doing MC-Dropout (legitimate, they want the variance) or whether they simply forgot `eval()` (a bug, they want the mean). The implementation tell: MC-Dropout enables *only* the dropout layers and keeps BatchNorm in eval mode (you do not want batch-statistics noise polluting an uncertainty estimate), whereas the forgotten-`eval()` bug leaves *everything* in train mode including BatchNorm. A correct MC-Dropout setup that accidentally leaves BatchNorm in train mode is itself a bug — the uncertainty estimate gets contaminated with batch-composition noise that has nothing to do with model uncertainty.

## 7. When this is (and isn't) your bug

Be decisive about ruling a mode bug in and out, because the symptoms overlap with data, optimization, and numerics bugs and you do not want to spend a day in the wrong corner.

It **is** a mode bug when: the **same checkpoint scores differently on repeated evaluation**. That is the fingerprint. A fixed-weight model is a deterministic function; if two eval passes over the same data disagree, something in the eval path is stochastic or batch-dependent, and the first suspect — before non-deterministic kernels or a shuffling loader — is a stochastic/batch-dependent layer running in train mode. Run the evaluate-twice test; if it fails, run `assert_eval(model)` to find the offending layers.

It **is** a mode bug when: **training was healthy until the first validation pass, then the loss curve bent.** That is Bug 2 — the forgotten `model.train()` back — and the confirming test is `assert model.training` at the top of the training step, which will fire on the step after eval. A loss curve that changes character *exactly* at the first eval boundary, and only there, is almost never a coincidence.

It **is** a mode bug when: a **finetune with a frozen backbone underperforms and its eval is unstable across epochs.** That is the freeze-BatchNorm trap (Bug 5): the running statistics are drifting because the BatchNorm layers are still in train mode despite `requires_grad=False`. Confirming test: print a BatchNorm `running_mean` at two epochs and watch it move.

It is **not** a mode bug — look elsewhere — when: the eval number is **stable but simply wrong**. A consistently-71% accuracy that repeats to the decimal on every eval run is *not* a mode bug, because mode bugs make metrics *unstable* (dropout) or *batch-dependent* (BatchNorm); a stable wrong number points to a data, label, or [loss-function](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) problem, or to a genuine generalization gap. Use stability as the discriminator: unstable-on-fixed-checkpoint → mode (or another eval-path non-determinism); stably-too-good → a [data leak](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); stably-too-bad with low train loss → overfitting or distribution shift; healthy-train-then-NaN → numerics, not mode.

It is **not** a mode bug when: the model has **no Dropout and no BatchNorm**. If your network uses only LayerNorm or GroupNorm (which are mode-independent) and no stochastic layers, then `model.eval()` is a genuine no-op for the forward pass, and a non-deterministic eval is coming from somewhere else — a shuffling val loader, a non-deterministic CUDA kernel, or test-time augmentation with randomness. Confirm by checking `mode_audit(model)` shows no mode-sensitive layers, then move to the [reproducibility and determinism](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) corner. This is an important discrimination because it stops you from staring at `model.eval()` when there is nothing for it to do.

There is one more discrimination worth making explicit, because it separates a *correctness* mode bug from a *memory* mode bug. If your inference is producing the *right* numbers but using twice the memory you expect, or OOMing on inputs that should fit, that is the `eval()`-without-`no_grad` half of Bug 3 — the math is correct (eval mode is set) but the autograd graph is being built for a backward pass that never runs. The instrument is `torch.cuda.max_memory_allocated()`: if it is roughly the train-time footprint during inference, you forgot `no_grad`/`inference_mode`. Conversely, if the memory is fine but the metric is corrupted, you forgot `eval()` and only have `no_grad`. The two halves of Bug 3 fail in opposite instruments — one in the metric, one in memory — and knowing which instrument moved tells you which half you have. When both eval and inference-mode are set and the symptom persists, the bug is not mode; return to the [taxonomy decision tree](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) and bisect to a different corner.

## 8. Mode under accumulation, DDP, and EMA

The mode flag interacts with three systems that change *when* and *over what* the train-time statistics are computed, and each interaction produces a mode-flavored bug that is easy to miss because the flag itself looks correct.

**Gradient accumulation.** When you accumulate gradients over several micro-batches to simulate a larger batch, every micro-batch is a separate forward pass, and BatchNorm — in train mode the whole time, correctly — computes its statistics over *each micro-batch independently*, not over the effective accumulated batch. So a run with micro-batch 8 and 4 accumulation steps gives BatchNorm an effective batch of 8 for its statistics, not 32, even though the optimizer sees an effective batch of 32. The mode is right; the *statistic* is computed over a smaller group than you think. The symptom is a model whose BatchNorm running stats are noisier than the nominal batch size would predict, which can surface as an unstable eval even with `eval()` called correctly, because the running average was built from noisy small-group statistics. This is not strictly a train/eval mode bug, but it lives in the same place — what statistics a layer uses — and the fix is the same family: use GroupNorm (no batch dependence at all) or accept that BatchNorm sees the micro-batch, not the effective batch. The confirming test is to compare the BatchNorm `running_var` between an accumulated run and a true-large-batch run; if the accumulated one is systematically noisier, you have found it.

**Distributed data parallel.** Under standard DDP, each GPU runs its own forward pass on its own shard of the batch, so each rank's BatchNorm computes statistics over only *its local micro-batch* — the per-GPU batch, not the global batch. Eight GPUs with local batch 16 give BatchNorm statistics over 16, not 128. The running stats each rank accumulates are local, and at eval each rank may even hold slightly different running stats unless they are synchronized. The mode flag is correct on every rank; the bug is that "batch statistics" silently means "this rank's slice." The fix is `torch.nn.SyncBatchNorm`, which computes statistics across all ranks so BatchNorm sees the true global batch. The signature of the missing SyncBN is a model that trains differently (often slightly worse, with noisier BatchNorm) on multi-GPU than on single-GPU at the same effective batch size — the "it behaves differently on 8 GPUs" tell that points at a systems-mode interaction. The deeper normalization statistics of all this belong to the [normalization post](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs); the mode-relevant point here is that the train-mode statistic is computed over a smaller, rank-local group than the effective batch, and no amount of `eval()`/`train()` discipline fixes that — it needs SyncBN.

**EMA of weights.** A separate but related trap: many training recipes keep an exponential-moving-average copy of the *weights* (not the BatchNorm stats — the actual parameters) and evaluate the EMA weights for a cleaner number. The mode bug here is subtle: the EMA model is a *separate* `nn.Module`, and it has its own `.training` flag and its own BatchNorm buffers. If you call `model.eval()` on the training model but forget to call `ema_model.eval()` on the EMA copy before evaluating it, you get the forgotten-`eval()` bug on the EMA model specifically — and because the EMA model is the one you report, your headline number is the corrupted one while the training model's number (which you do not report) is fine. The confirming test is the same evaluate-twice on the *EMA* model, and the fix is to route the EMA model through the same `evaluating()` context manager. The lesson generalizes: every `nn.Module` you ever call forward on — the training model, the EMA model, a teacher model in distillation, a reference model in DPO — has its own mode flag, and each one must be in eval mode when you use it for inference. A distillation run that leaves the *teacher* in train mode feeds dropout-noisy and batch-dependent targets to the student, quietly degrading the whole distillation; a DPO run that leaves the *reference* model in train mode computes a noisy KL term. The mode discipline is per-module, and "the model" is rarely just one module.

## 9. Mode bugs across modalities

The mechanism is the same everywhere a stateful layer lives, but the *prominence* of each bug shifts by domain, which is worth knowing because it tells you which one to check first.

In **computer vision**, BatchNorm dominates. ResNets, EfficientNets, and most convolutional backbones are BatchNorm-heavy, so the most common mode bug is the batch-dependent eval (forgot `eval()` → metric leaks batch composition) and, in finetuning, the freeze-BatchNorm trap. Vision is where the freeze-BN pattern matters most, because vision finetuning on small datasets is ubiquitous and the per-batch statistics on small high-resolution batches are especially unreliable. If you are debugging a vision model with a bouncing or unstable eval, check BatchNorm mode first.

In **transformers and LLMs**, Dropout dominates and BatchNorm is essentially absent — transformers use LayerNorm (or RMSNorm), which is mode-independent. So the forgotten-`train()`/`eval()` bug shows up through Dropout: attention dropout and residual dropout running at inference inject noise into generation, making outputs non-deterministic in a way that is *not* the intended sampling temperature. A transformer that produces different greedy-decoded outputs for the same prompt on repeated runs — with sampling disabled — has dropout active at inference. Because LayerNorm is mode-free, the *batch-dependence* class of mode bug does not occur in a standard transformer, which is a relief; the residual risk is purely the dropout-noise half. (The LLM-specific train/inference mismatches that *aren't* mode bugs — KV-cache discrepancies, left-versus-right padding, teacher-forcing versus autoregressive generation — are a different and deeper topic covered in train/inference mismatch for LLMs.)

In **speech and audio**, models are mixed: wav2vec2 and Whisper use LayerNorm-style normalization in their transformer bodies but often have BatchNorm or GroupNorm in their convolutional feature encoders, plus dropout and SpecAugment-style stochastic masking. The mode bug here is usually that SpecAugment or feature dropout stays active at eval, corrupting transcription, or that a convolutional-frontend BatchNorm is evaluated with cold or mismatched stats. The evaluate-twice test still works as the universal first check.

In **tabular and classical deep learning**, the small MLPs people build for tabular data frequently stack Dropout and BatchNorm, and because these models are small and fast to evaluate, the bouncing-metric symptom is often dismissed as "just noise" — when it is actually dropout running at eval. The fix is identical; the lesson is that the bug does not care how small your model is.

The universal first move, regardless of modality: **evaluate the same checkpoint twice and compare.** If the numbers differ, you have a mode bug (or another eval-path non-determinism) before you have written a single domain-specific diagnostic. It is the cheapest, most general test in this entire post, and it costs one extra forward pass.

## Key takeaways

- **`model.training` is one boolean that rewires three layer families.** It flips Dropout (train: zero + scale $1/(1-p)$; eval: identity) and BatchNorm (train: batch stats + update running stats; eval: frozen running stats). Forgetting to set it changes what your network *computes*, silently, with no exception.
- **A metric that bounces on a fixed checkpoint is a mode bug until proven otherwise.** Confirming test: evaluate twice and compare — a deterministic function must repeat. The usual cause is Dropout or BatchNorm running in train mode at eval. Fix: `model.eval()`.
- **Dropout's two modes agree in mean, differ in variance.** Inverted dropout scales survivors by $1/(1-p)$ so $\mathbb{E}[\tilde a]=a$; eval-in-train-mode therefore injects *mean-zero noise*, bouncing the metric without a fixed bias.
- **BatchNorm's two modes use different statistics.** Eval-in-train-mode makes each prediction depend on its batch-mates (non-reproducible, a test-time leak); train-stuck-in-eval-mode freezes running stats and starves learning. The first hurts eval, the second hurts training after the first eval.
- **`eval()` and `no_grad()` are orthogonal — you need both at inference.** `eval()` changes layer behavior; `no_grad()`/`inference_mode()` changes graph building. Only-`no_grad` gives a corrupted metric; only-`eval()` gives ~2x memory and OOM risk. The two halves fail in *different instruments* (metric vs memory).
- **Use a context manager for the eval path.** `evaluating(model)` sets eval, wraps `inference_mode`, and restores the prior mode in a `finally` — which kills the forgot-`eval()`, forgot-`train()`-back, and forgot-`no_grad` bugs at once and survives a crash mid-validation.
- **Freezing BatchNorm needs `bn.eval()`, not just `requires_grad=False`.** Params-only freeze leaves running stats drifting on small finetuning batches, rotting pretrained features. Re-apply `bn.eval()` *after every* `model.train()`, because `train()` un-freezes everything.
- **Cold BatchNorm running stats make a freshly-initialized model eval as garbage** even with `eval()` set correctly. The stats start at 0/1 and need a few dozen train-mode batches to warm up; a `load_state_dict` that drops BN buffers reproduces this.
- **Assert the mode at the boundaries.** `assert not model.training` at the top of validation and `assert model.training` at the top of the training step turn a silent mode error into a loud failure at the responsible line.

## Further reading

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." The original Dropout and the train/test scaling argument behind inverted dropout.
- Ioffe, S. and Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." The original BatchNorm, including the train-time batch statistics versus eval-time running statistics mechanism.
- Li, X., Chen, S., Hu, X., and Yang, J. (2019). "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift." Why stacking Dropout before BatchNorm leaves a train/test gap even with correct eval-mode usage.
- Wu, Y. and He, K. (2018). "Group Normalization." BatchNorm's batch-size sensitivity and a batch-independent, mode-free normalizer that sidesteps the train/eval distinction entirely.
- He, K., Gkioxari, G., Dollár, P., and Girshick, R. (2017). "Mask R-CNN," and the Detectron2 codebase, for the `FrozenBatchNorm2d` practice of hard-freezing BatchNorm when finetuning detection backbones on small batches.
- PyTorch documentation: `torch.nn.Module.train` / `.eval`, `torch.nn.Dropout`, `torch.nn.BatchNorm2d`, `torch.no_grad`, and `torch.inference_mode` — the authoritative semantics for the mode flag, the running-statistics buffers, and the orthogonality of mode and gradient tracking.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the symptom→suspect→test→fix decision tree this post instantiates), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone checklist), and the sibling tracks [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs) (the deeper BatchNorm statistics, GroupNorm/SyncBN, LayerNorm placement), [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) (the other eval-path non-determinism sources), and [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying) (the evaluation-corner bugs a corrected eval mode does not fix).

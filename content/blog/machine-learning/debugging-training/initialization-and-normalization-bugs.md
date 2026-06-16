---
title: "Initialization and Normalization Bugs: The BatchNorm Trap and Friends"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Diagnose and fix the silent killers that sink a run before step one: bad init that explodes or collapses the signal across depth, and the BatchNorm train/eval trap that makes your metrics lie."
tags:
  [
    "debugging",
    "model-training",
    "initialization",
    "normalization",
    "batchnorm",
    "finetuning",
    "deep-learning",
    "computer-vision",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/initialization-and-normalization-bugs-1.png"
---

There is a special kind of training bug that beats you before the optimizer ever takes a meaningful step. The loss prints, the GPUs spin, the progress bar advances — and the model learns nothing. Not "learns slowly." Nothing. The curve is a flat line at chance, glued there for ten thousand steps, and nothing about the code *looks* wrong. You did not get a NaN. You did not get a shape error. You got silence.

I have lost the better part of a week to exactly this, twice. The first time it was a 40-layer convolutional network that would not move off chance accuracy no matter how I tuned the learning rate — because the weights were initialized with a variance so small that the activation signal decayed to numerical dust by the time it reached the deep layers, and the gradient that flowed back was already $10^{-9}$ before the first weight update. The second time it was the opposite failure with a different mask: a perfectly good model whose **validation** accuracy bounced between 78% and 84% from run to run, for the dumbest possible reason — I had forgotten to call `model.eval()`, so BatchNorm was normalizing my test predictions with the statistics of whatever random batch they happened to land in.

Both of these are *initialization and normalization* bugs, and they share a cruel property: they live entirely in the **numerics** and **model-code** corners of the [six places a bug hides](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — data, optimization, model code, numerics, systems, evaluation — but they masquerade as optimization problems. You will tune the learning rate for a day. You will swap optimizers. You will add and remove warmup. None of it works, because the problem is set the instant the weights are sampled and the instant a normalization layer decides which statistics to use. Figure 1 shows the core mechanism we are going to make rigorous: how the *variance* of your initialization compounds across depth and either explodes the signal to $10^6$ or collapses it to $10^{-6}$ before you have computed a single gradient.

![Diagram showing how activation variance compounds through network depth, with He init staying near one while naive small or large init shrinks or grows geometrically until the signal explodes or collapses by layer forty](/imgs/blogs/initialization-and-normalization-bugs-1.png)

By the end of this post you will be able to do four concrete things. First, derive *why* Xavier/Glorot init uses variance $1/\text{fan}$ and He/Kaiming uses $2/\text{fan}$, and predict from the architecture alone whether a given init will blow up or die. Second, audit any model in three lines so you can *see* the activation scale across depth and catch a bad init before training. Third, recognize the entire family of BatchNorm bugs — the forgotten `eval()`, the small-batch failure, the frozen-running-stats trap, the train/serve skew, and the silent breakage under gradient accumulation and distributed data parallel — from their instrument signatures. Fourth, know exactly when to reach for GroupNorm, SyncBN, or LayerNorm instead, and prove the fix worked with a before→after number, not a vibe. We will use a single running example — a deep CNN that refuses to train, then a detection model whose eval bounces — and bisect it the same disciplined way every time.

A word on why these two bug families belong in one post. On the surface initialization (a one-time sampling of weights) and normalization (a layer that runs every forward pass) look unrelated. But they are two answers to the *same question*: how do you keep the scale of the signal under control as it flows through a deep network? Initialization controls the scale at step 0 by setting the weight variance; normalization controls it at every step by actively rescaling the activations. They are substitutes and complements — a correctly normalized network is far less sensitive to its init, which is one reason BatchNorm made deep nets so much easier to train; and a normalization-free network leans entirely on its init to stay in scale, which is why residual init scaling and Fixup matter so much. Debug them together and you cover the entire "is my signal at the right scale?" axis, which is one of the most common and most silent ways a run fails. Both also share the worst property a bug can have: they fail *quietly*, with no exception, so the only way to catch them is to read the instruments rather than wait for a crash.

## 1. The science of initialization: variance is a signal budget

Start with the cleanest possible model: a stack of linear layers with no nonlinearity, no bias, no normalization. Each layer computes $\mathbf{y} = W\mathbf{x}$, where $W$ is $n_{\text{out}} \times n_{\text{in}}$ and we sample every weight i.i.d. from a distribution with mean $0$ and variance $\sigma_w^2$. The input $\mathbf{x}$ has $n_{\text{in}}$ components, which we will also treat as i.i.d. with mean $0$ and variance $\sigma_x^2$. We want to know the variance of a single output component $y_i$, because *that variance is the scale of the signal*, and the whole game is keeping it from drifting as we stack layers.

Each output is a sum: $y_i = \sum_{j=1}^{n_{\text{in}}} W_{ij} x_j$. Because the weights and inputs are independent and zero-mean, the variance of a product is the product of variances, and the variance of a sum of independent terms is the sum of variances. So

$$
\text{Var}(y_i) = \sum_{j=1}^{n_{\text{in}}} \text{Var}(W_{ij} x_j) = n_{\text{in}} \, \sigma_w^2 \, \sigma_x^2.
$$

That single equation is the entire foundation. The output variance is the input variance multiplied by $n_{\text{in}} \sigma_w^2$. Call that multiplier the **layer gain** $g = n_{\text{in}} \sigma_w^2$. If $g > 1$ the signal grows; if $g < 1$ it shrinks; if $g = 1$ it is preserved. Now stack $L$ layers. The variance at the output is the input variance times $g^L$. This is the geometric compounding that Figure 1 draws. With $g = 1.4$ and $L = 40$, the gain is $1.4^{40} \approx 7 \times 10^5$. With $g = 0.7$, it is $0.7^{40} \approx 6 \times 10^{-7}$. Either way the network is destroyed before the first nonlinearity has a chance to matter.

So the design rule writes itself: **set $\sigma_w^2$ so that the layer gain is exactly $1$.** That gives $\sigma_w^2 = 1/n_{\text{in}}$. This is **Xavier/Glorot initialization** (Glorot and Bengio, 2010), and the only subtlety is that you also want the *backward* pass to preserve variance, which depends on $n_{\text{out}}$, so Glorot split the difference and used $\sigma_w^2 = 2/(n_{\text{in}} + n_{\text{out}})$ — the harmonic-ish average of the forward and backward constraints.

### Why ReLU needs a factor of two

Xavier assumes a roughly linear, symmetric activation. ReLU breaks that assumption: it zeroes out the negative half of its input. If the pre-activation $z$ is symmetric around zero, then on average **half** of its variance survives the ReLU, because the negative half is clamped to zero. More precisely, for $z$ symmetric with mean zero, $\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\text{Var}(z)$. That halving means the layer gain is effectively cut in half, so to keep the *post-activation* variance constant you need to *double* the weight variance:

$$
\sigma_w^2 = \frac{2}{n_{\text{in}}}.
$$

This is **He/Kaiming initialization** (He et al., 2015), the one you almost always want for ReLU and its relatives. The factor of two is not a hyperparameter someone tuned; it is the survival fraction of a ReLU falling out of the variance algebra. Get it wrong — use Xavier's $1/n$ with ReLU — and your gain is $0.5$ per layer, so after 40 layers the signal is down by $0.5^{40} \approx 9 \times 10^{-13}$. That is the original 40-layer-CNN bug, and it is why pre-residual very deep nets were nearly untrainable until He init (and later BatchNorm and residual connections) arrived.

### The backward pass has its own variance budget

Everything so far has been about the *forward* pass — keeping the activation variance constant as the signal flows from input to output. But training also runs a *backward* pass, and the gradient has its own variance that compounds across depth in exactly the same geometric way, with a gain that is generally *different* from the forward gain. This is the source of the `fan_in`-vs-`fan_out` choice that confuses people in `nn.init.kaiming_normal_`, so it is worth deriving.

The backward pass propagates the gradient $\partial \mathcal{L} / \partial \mathbf{y}$ back through $W$ to get $\partial \mathcal{L} / \partial \mathbf{x} = W^\top (\partial \mathcal{L} / \partial \mathbf{y})$. That is the same linear-map variance algebra as the forward pass, but now the sum runs over $n_{\text{out}}$ terms instead of $n_{\text{in}}$, because the transpose changes which dimension we contract over. So the *backward gain* is $g_{\text{back}} = n_{\text{out}} \sigma_w^2$, and to preserve gradient variance across depth you would want $\sigma_w^2 = 1/n_{\text{out}}$ (or $2/n_{\text{out}}$ for ReLU). You cannot satisfy both the forward constraint ($\sigma_w^2 = 1/n_{\text{in}}$) and the backward constraint ($\sigma_w^2 = 1/n_{\text{out}}$) at once unless $n_{\text{in}} = n_{\text{out}}$. That is precisely why Glorot averaged them into $2/(n_{\text{in}} + n_{\text{out}})$, and why PyTorch's Kaiming init gives you a `mode` argument: `fan_in` preserves the forward-pass variance (the default, and usually the right choice because forward-signal death is the more common killer), while `fan_out` preserves the backward-pass gradient variance. For a network whose layers are mostly square ($n_{\text{in}} \approx n_{\text{out}}$, like a uniform-width MLP or the body of a ResNet), the choice barely matters because the two gains are nearly equal. It matters most at the layers where the width changes sharply — a wide-to-narrow projection, or the first conv that expands a 3-channel image to 64 channels — where forward and backward gains diverge and you have to decide which signal you care more about.

### Gains for other nonlinearities, and the `gain` argument

The factor of two is specific to ReLU's "half the variance survives" property. Other activations have other survival fractions, captured by the **gain** in `nn.init.calculate_gain(nonlinearity)`. Linear/identity has gain $1$ (no correction). Tanh has gain $\approx 5/3$ because it compresses the signal near saturation. Leaky ReLU with negative slope $a$ has gain $\sqrt{2/(1+a^2)}$ — it recovers a little of the variance ReLU throws away, which is part of why leaky variants train slightly more robustly at the extremes. SELU has its own self-normalizing constant. The practical rule: when you call `kaiming_normal_`, *pass the actual nonlinearity* (`nonlinearity='relu'`, `'leaky_relu'`, etc.) so the gain matches your activation. The silent bug here is initializing for ReLU but using GELU or SiLU (both common in modern transformers), where the effective survival fraction is close to but not exactly ReLU's — usually close enough not to matter, but worth knowing when a deep custom block drifts.

```python
import torch.nn as nn

# Match the init to the actual nonlinearity. The gain is not optional decoration;
# it is the variance-survival correction for that activation.
def he_init_(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)   # bias starts at 0; it does not affect variance

# For tanh, use Xavier with the tanh gain instead:
def xavier_tanh_init_(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("tanh"))
```

### Residual connections change the rules: scale the branch, not just the weight

Modern deep nets do not stack bare layers — they use **residual connections**, $\mathbf{x}_{\text{out}} = \mathbf{x} + F(\mathbf{x})$, and that addition changes the variance bookkeeping in a way that creates its own init bug. If both the identity path and the residual branch $F(\mathbf{x})$ have unit variance and are roughly independent, their sum has variance $\approx 2$ — so the activation variance *grows by a factor of two at every residual block*, and a 50-block ResNet would see its variance multiplied by $2^{50}$ if nothing counteracted it. In practice BatchNorm inside the block clamps the variance back down, which is one of the reasons BatchNorm and residuals arrived together and work so well as a pair. But when you build a residual net *without* normalization (increasingly common in efficient or quantization-friendly architectures), you have to control the growth at init: the standard fix is to **initialize the last layer of each residual branch to zero (or near-zero)** so that $F(\mathbf{x}) \approx 0$ at the start and the block is an identity, letting the network "grow into" its depth as training proceeds. This is the idea behind Fixup, SkipInit, and the zero-init-residual flag in many ResNet implementations (`torchvision`'s `zero_init_residual=True`). The bug it prevents: a normalization-free residual net that explodes its activations across depth at step 0 and NaNs immediately, which looks like a numerics bug but is really a residual-init bug. The signature is identical to the exploding-init signature from Figure 1 — activation std climbing geometrically — but the fix is to zero the branch, not to rescale every weight.

> The provable point: initialization is not cosmetic. The activation scale at depth $L$ is the input scale times $g^L$, and $g$ is set entirely by your init variance, your nonlinearity, and your residual structure. A gain that is off by 30% is invisible at layer 2 and fatal at layer 40, and a residual sum that grows variance by $2\times$ per block is fatal even faster.

#### Worked example: the 40-layer net that would not move

Concretely: a 40-layer fully-connected ReLU net, width 256, initialized with `nn.init.normal_(w, std=0.01)`. That gives $\sigma_w^2 = 10^{-4}$, so the forward gain per layer is $g = n_{\text{in}} \sigma_w^2 = 256 \times 10^{-4} = 0.0256$. After 40 layers the activation variance is multiplied by $0.0256^{40}$, which is so far below the smallest fp32 normal number ($\approx 1.2 \times 10^{-38}$) that the deep-layer activations are effectively zero. The forward pass produces an all-zeros logit vector, the loss is $\ln(10) \approx 2.30$ for 10 classes (pure chance), and the backward gradient that reaches layer 1 has norm around $10^{-9}$. The loss curve is a flat line. Swap in `nn.init.kaiming_normal_(w, nonlinearity='relu')` and the gain becomes $g = 256 \times (2/256) = 1.0$ exactly, the activation std stays near $1.0$ at every depth, the gradient norm at layer 1 is $2.3$, and the loss falls from $2.30$ to $0.18$ in the first epoch. Same architecture, same data, same optimizer. One line of init. Figure 2 is that before→after.

![Two-column comparison of a forty-layer network under naive small init versus He init, showing activation standard deviation, gradient norm, and whether the loss falls below chance](/imgs/blogs/initialization-and-normalization-bugs-2.png)

This is the most important takeaway of the init half of this post: **the symptom of a bad init is a flat loss at chance, and the confirming test is to print the activation standard deviation per layer.** If it is decaying or exploding geometrically with depth, your init is wrong, and no amount of learning-rate tuning will save you because the gradient that depends on those activations is already destroyed.

## 2. The diagnostic: audit your activations right after init

Never train blind. Before the first optimizer step — literally after building the model and before the training loop — push one batch through and read the activation scale at every layer. This is the make-it-fail-small principle applied to init: a single forward pass tells you whether the signal budget is intact. Here is a copy-and-run audit using forward hooks.

```python
import torch
import torch.nn as nn

def audit_activation_scale(model, sample_input):
    """Print the output std of every module after one forward pass.

    A healthy net keeps activation std within a small band (roughly 0.3 to 3)
    across depth. Geometric decay or growth means the init variance is wrong.
    """
    stats = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                stats.append((name, out.detach().float().std().item()))
        return hook

    handles = []
    for name, module in model.named_modules():
        # Hook the leaf compute layers, not containers.
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(sample_input)

    for h in handles:
        h.remove()

    for name, std in stats:
        flag = ""
        if std < 1e-3:
            flag = "  <-- VANISHING"
        elif std > 1e3:
            flag = "  <-- EXPLODING"
        print(f"{name:40s} act std = {std:.3e}{flag}")

# usage
# audit_activation_scale(model, torch.randn(32, 3, 224, 224, device="cuda"))
```

Run this on the broken 40-layer net and you will see the std halving roughly every layer — `1.0e+00`, `5.1e-01`, `2.6e-01`, ... — until it underflows to `0.000e+00` somewhere around layer 25 and stays there. That is the unmistakable signature. The fix flips it to a flat sequence hovering around `1.0` at every depth. The same audit catches the *opposite* bug — a too-large init or a missing normalization in a transformer block, where the std climbs `1.0`, `1.4`, `2.0`, `2.8`, ... until it overflows to `inf`.

There is a backward-pass version of this audit that is just as important, because vanishing/exploding in the forward pass and in the backward pass are related but not identical (the forward gain and backward gain differ when $n_{\text{in}} \neq n_{\text{out}}$). Register a tensor hook on the gradient instead:

```python
import torch

def audit_grad_norms(model, loss):
    """After loss.backward(), print the grad norm of every parameter.

    Healthy: norms within a couple orders of magnitude of each other.
    Bug: norms decay geometrically toward the input layers (vanishing)
    or grow without bound toward them (exploding).
    """
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"{name:40s} grad = None  <-- NO GRADIENT")
            continue
        g = p.grad.norm().item()
        flag = ""
        if g < 1e-7:
            flag = "  <-- VANISHED"
        elif g > 1e3:
            flag = "  <-- EXPLODING"
        print(f"{name:40s} grad norm = {g:.3e}{flag}")
```

If you only adopt one habit from this post, make it this: **print the per-layer activation std after init and the per-layer grad norm after the first backward pass, every single time you stand up a new architecture.** It costs three seconds and one batch, and it converts a week-long mystery into a number you can read at a glance. For the full instrument panel — grad norm, update-to-param ratio, activation histograms — see [instrumenting a training run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log); here we care about the two readings that init and normalization bugs move first.

### Why warmup partially hides init sins

A note that trips people up: learning-rate warmup (ramping the LR from near-zero over the first few hundred or few thousand steps) can make a *marginally* bad init train anyway, which is why you sometimes "fix" a model by adding warmup without understanding what was wrong. Here is the mechanism. With a slightly-too-large init, the early gradients are large and noisy; a full learning rate would take a giant, destabilizing step on the first batch and blow up. Warmup keeps the first steps tiny, so the optimizer nudges the weights toward a better-conditioned region *before* it takes large steps, and the network's own training dynamics partially repair the variance mismatch. Warmup does *not* help a catastrophically bad init — if the forward signal has already underflowed to zero, there is no gradient to warm up *with*. So treat warmup as a band-aid for a near-miss init and a non-fix for a real one. The principled move is to set the init correctly and use warmup for the reason it actually exists (stabilizing large-batch / high-LR optimization), not to paper over a variance bug.

### The output-layer and embedding init bugs

Two specific layers deserve their own init attention because they fail in ways the per-layer variance audit does not flag, and both are common in classification and language models.

The **output (classifier) layer** controls the *initial loss*, and getting its init wrong wastes the first phase of training fighting a bad starting point. For a $K$-class softmax classifier at initialization, you want the logits to be near zero so the predicted distribution is roughly uniform — every class gets probability $\approx 1/K$ — which gives the expected initial loss of $\ln K$. If instead you initialize the final layer with a large variance, the logits at step 0 are large and random, so the model starts *confidently wrong*: the initial loss is much higher than $\ln K$, and the first hundred steps are spent un-learning the random confidence before any real learning begins. The fix is to initialize the final layer small (or zero the final bias and use a small weight std). This is also where a known detection/imbalance trick lives: in RetinaNet-style detectors, the final classification bias is deliberately initialized to a *negative* value so the model starts predicting "background" with high probability, which stabilizes training under extreme foreground/background imbalance (the "prior probability" init from the Focal Loss paper). The signature of getting this wrong is an initial loss far above the theoretical $\ln K$ and a slow, bumpy first epoch — easy to confirm by comparing your step-0 loss to $\ln K$.

```python
import math
import torch.nn as nn

# A 1000-class classifier should start near uniform: initial loss ~ ln(1000) = 6.9.
# If your step-0 loss is 15, the output layer is initialized too hot.
K = 1000
print(f"expected step-0 loss for {K} classes ~ {math.log(K):.2f}")

def init_classifier_head(linear, num_classes, prior=None):
    nn.init.normal_(linear.weight, std=0.01)   # small -> logits near 0 -> ~uniform
    if linear.bias is not None:
        if prior is not None:  # focal-loss-style bias for heavy imbalance
            linear.bias.data.fill_(-math.log((1 - prior) / prior))
        else:
            nn.init.zeros_(linear.bias)
```

The **embedding layer** in language models has the opposite concern: its init variance sets the *scale of the input* to the first transformer block, and many implementations scale the embeddings by $\sqrt{d_{\text{model}}}$ (as in the original Transformer) so that the embedding magnitude matches the positional encodings and the LayerNorm downstream behaves. Forgetting that scale, or double-applying it, shifts the input distribution to the whole network — a quiet bug that makes a from-scratch language model train slower or less stably than the reference, with no single layer's audit looking obviously wrong because the mismatch is a constant factor applied at the very input. The tell is a from-scratch run whose loss curve is shaped right but offset worse than a known-good reference; check the embedding scale against the reference implementation.

## 3. BatchNorm has two completely different modes, and that is the bug

Now the second half, and the one that has burned more engineers than init ever will: normalization layers, and BatchNorm above all. To debug BatchNorm you must hold one fact in your head at all times — **BatchNorm computes statistics two completely different ways depending on whether the module is in training mode or eval mode.** Almost every BatchNorm bug is a consequence of using the wrong mode, or of one mode's statistics being garbage.

In **training mode**, BatchNorm normalizes each activation using the mean and variance *of the current minibatch*. For a feature (channel) $c$, over the $m$ elements in the batch (and, for conv, the spatial positions), it computes the batch mean $\mu_B$ and batch variance $\sigma_B^2$, then normalizes:

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta.
$$

Simultaneously, it updates an exponential moving average of these statistics — the **running mean** and **running variance** — using momentum $\alpha$ (PyTorch default $0.1$): $\mu_{\text{run}} \leftarrow (1-\alpha)\mu_{\text{run}} + \alpha \mu_B$.

In **eval mode**, BatchNorm does *not* look at the batch at all. It normalizes using the frozen running mean and running variance it accumulated during training:

$$
\hat{x} = \frac{x - \mu_{\text{run}}}{\sqrt{\sigma_{\text{run}}^2 + \epsilon}}.
$$

This dual behavior is the whole point of BatchNorm — at training time you want the regularizing noise of batch statistics, and at inference time you want a deterministic, sample-independent transform. But it is also a loaded gun. Figure 3 lays the two modes side by side, because once you internalize that train mode reads the batch and eval mode reads the running stats, every BatchNorm bug becomes obvious.

![Two-column before-after diagram contrasting BatchNorm in training mode using noisy per-batch statistics against eval mode using stable running statistics, and the resulting validation accuracy stability](/imgs/blogs/initialization-and-normalization-bugs-3.png)

### Why BatchNorm helps at all (so you can predict how it breaks)

It is worth being precise about *why* BatchNorm accelerates training, because the mechanism is exactly what makes it fragile. The original paper (Ioffe and Szegedy, 2015) framed it as reducing "internal covariate shift" — the idea that as earlier layers update, the distribution of inputs to later layers keeps shifting, forcing them to chase a moving target, and that normalizing each layer's inputs to a fixed mean and variance stabilizes that. Later work (Santurkar et al., 2018, "How Does Batch Normalization Help Optimization?") argued the dominant effect is different and more useful to know: BatchNorm **smooths the loss landscape**, making the gradients more predictable (smaller Lipschitz constant), so larger learning rates are stable and optimization is faster. Either way, the common thread is that BatchNorm's benefit comes from *recentering and rescaling the activations using statistics computed from the data*. That is the source of both the speedup and every bug in this section: the moment those statistics are computed from too few samples (small batch), from the wrong samples (train batch at eval time), or from the wrong distribution (train/serve skew), the smoothing it was buying you turns into noise it is injecting. The benefit and the bug are the same mechanism viewed from two sides — which is why "is the statistic any good?" is the single question that resolves most BatchNorm mysteries.

The second-order effect to keep in mind is that BatchNorm makes the loss **depend on the whole batch**, not just the individual example. In a normal layer, example $i$'s output depends only on example $i$. Under BatchNorm in training mode, example $i$'s normalized activation depends on $\mu_B$ and $\sigma_B^2$, which depend on *every other example in the batch*. This coupling is what creates the regularizing noise (each example sees slightly different normalization depending on its batch-mates) and is also why batch *composition* matters so much: change which examples land together and you change the effective augmentation each one receives. It is a feature at large, well-mixed batches and a bug at small or skewed ones.

### Bug 1: forgetting `model.eval()`

The canonical BatchNorm bug. You finish a training step, run validation, and report a number. But you never called `model.eval()`, so BatchNorm is *still in training mode* during validation — it normalizes each validation prediction using the statistics of whatever validation batch it landed in, and it *also keeps updating the running stats* with validation data. Two things go wrong at once. The metric becomes **non-deterministic**: the same validation image gets a different prediction depending on its batch-mates, because $\mu_B$ depends on the whole batch. And the metric becomes **biased**, usually optimistically during training (the batch stats are a perfect fit for the batch) and then worse at serve time where you feed one example at a time.

The symptom is a validation number that **bounces from run to run, or from batch size to batch size**, with no real change in the model. You will see val accuracy of 78% on one evaluation and 84% on the next, with identical weights. That is not the optimizer being noisy; that is BatchNorm normalizing with different batch statistics each time. The confirming test is one line:

```python
# Before any evaluation, assert you are actually in eval mode.
def assert_eval_mode(model):
    training_modules = [name for name, m in model.named_modules() if m.training]
    assert not model.training, "model.training is True during eval!"
    bn_in_train = [
        name for name, m in model.named_modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training
    ]
    assert not bn_in_train, f"BatchNorm still in train mode: {bn_in_train}"
    print("OK: model and all BatchNorm layers are in eval mode")

# the fix is, of course:
model.eval()
with torch.no_grad():
    metrics = evaluate(model, val_loader)
model.train()  # remember to switch back before the next training step
```

This bug is so common that the discipline worth burning into muscle memory is: `model.eval()` before every evaluation, `model.train()` before resuming training, and a `torch.no_grad()` (or `torch.inference_mode()`) context around the eval forward passes. Note that `eval()` and `no_grad()` are *different* concerns — `eval()` switches BatchNorm and Dropout to inference behavior, while `no_grad()` disables the autograd graph to save memory. You need both for evaluation, and confusing them is its own [train/eval-mode bug](/blog/machine-learning/debugging-training/train-eval-mode-bugs). Calling only `no_grad()` without `eval()` leaves BatchNorm using batch stats — your gradients are off but your *statistics* are still wrong.

#### Worked example: the eval that bounced

A ResNet-50 image classifier, trained fine, reporting validation accuracy that drifted between 78.1% and 84.3% across evaluations on the *same checkpoint*. The team spent two days suspecting data ordering and a flaky augmentation. The actual cause: the eval loop never called `model.eval()`, and the validation `DataLoader` had `shuffle=True` with a `drop_last` that varied the final batch. So each eval normalized with different batch statistics. The confirming test was to run the eval loop twice with a fixed seed and watch the number move — a deterministic eval cannot do that. Adding `model.eval()` collapsed the variance: the number became a rock-steady 86.0% (higher *and* stable, because running stats are a better estimate than a 32-image batch). Symptom: val bounces 78→84. Confirming test: `assert not model.training`. Fix: `model.eval()`. After: stable 86.0%. That is the entire debugging loop, and it took ten minutes once we suspected the right corner.

## 4. The small-batch BatchNorm failure, derived

The forgotten-`eval()` bug is about *which* statistics. The small-batch bug is about whether the statistics are any *good*. BatchNorm estimates the channel mean and variance from the minibatch. Like any sample estimate, those estimates have **sampling error that grows as the batch shrinks**, and below a certain batch size the error is large enough to wreck training and, worse, to poison the running statistics that eval will later depend on.

Make it quantitative. Suppose the true per-channel activation distribution has variance $\sigma^2$. You estimate it from a batch of $m$ samples with the sample variance $s^2$. Under a roughly Gaussian assumption, the sample variance follows a scaled chi-squared distribution, and its relative sampling error is

$$
\frac{\text{Std}(s^2)}{\sigma^2} \approx \sqrt{\frac{2}{m-1}}.
$$

Plug in numbers. At $m = 256$, the relative error in your variance estimate is about $\sqrt{2/255} \approx 8.9\%$ — tolerable. At $m = 32$, it is $\sqrt{2/31} \approx 25\%$. At $m = 8$, it is $\sqrt{2/7} \approx 53\%$ — your normalizer's denominator is wandering by half its value batch to batch. At $m = 2$, it is $\sqrt{2/1} \approx 141\%$, and the estimate is meaningless. The mean estimate degrades the same way, with standard error $\sigma/\sqrt{m}$. So the normalization that is supposed to stabilize training is itself injecting enormous noise, and the running statistics it accumulates are a noisy average of noisy estimates. This is the rigorous reason BatchNorm degrades at small batch — it is not folklore, it is the sampling distribution of a variance estimate.

And the pathological endpoint: **batch size 1.** With $m = 1$, the batch variance is *identically zero* — a single sample has no spread around its own mean. So $\sigma_B^2 = 0$, the normalized output is $(x - x)/\sqrt{0 + \epsilon} = 0$ for every channel, and BatchNorm outputs the constant $\beta$ regardless of the input. The layer has destroyed all information. (For convolutional BatchNorm the spatial positions count toward $m$, so a single image still has $H \times W$ "samples" per channel and the variance is not literally zero — but for a BatchNorm over a non-spatial feature, or a $1\times1$ feature map, batch size 1 is genuinely undefined.) This is why you will see a fully-connected BatchNorm produce identical outputs for every input when the batch is 1, and why detection and segmentation models — which often run 1–4 images per GPU because the images are huge — cannot use vanilla BatchNorm.

Figure 4 is the decision matrix: which normalizer to use as a function of batch size and modality, and the signature failure of each.

![Matrix table mapping BatchNorm, LayerNorm, and GroupNorm to their best setting, what statistics they compute over, and their signature failure mode such as undefined variance at batch size one](/imgs/blogs/initialization-and-normalization-bugs-4.png)

### The fix: GroupNorm and friends remove the batch dependence

The clean fix for small-batch training is to use a normalizer that does **not** compute statistics over the batch dimension at all. **GroupNorm** (Wu and He, 2018) divides the channels into $G$ groups and normalizes within each group, over the channel-and-spatial dimensions *of a single sample*. Its statistics are computed per-example, so they are identical at batch size 1, 2, or 256, and identical in train and eval mode — which also means GroupNorm has **no train/eval discrepancy and no running statistics to forget**. **LayerNorm** is the special case $G = 1$ (normalize over all channels of one sample) and is the standard in transformers. **InstanceNorm** is $G = C$ (one group per channel). The trade-off is that GroupNorm gives up the cross-example regularization that makes BatchNorm so effective at large batch — at $m \geq 32$ BatchNorm usually still wins on accuracy — so it is a small-batch tool, not a universal replacement.

There is a second fix worth knowing when you are stuck with BatchNorm — for instance, a pretrained backbone whose architecture you do not want to change — called **precise BN** or **BN recalibration**. The idea: the running statistics accumulated by the EMA during training are a *biased, noisy* estimate, especially at small batch, because the EMA weights recent (small, noisy) batches and never sees a clean aggregate. After training, you can recompute the running mean and variance properly by doing a forward-only pass over a few hundred batches *in training mode but without optimizer steps*, accumulating the true average statistics, and then freezing those. This often recovers a fraction of a point that the noisy EMA estimate left on the table, and it directly addresses the heavy-augmentation mismatch from Section 7 if you recalibrate on un-augmented data. It does not fix the *fundamental* small-batch problem (the per-step training dynamics are still using noisy stats), but it cleans up the eval-time estimate, which is sometimes all you need.

```python
import torch

@torch.no_grad()
def recompute_bn_stats(model, loader, num_batches=200):
    """Precise BN: re-estimate running mean/var with a clean forward-only pass.
    Reset the running stats, run the model in train mode (so BN updates them)
    over many batches without stepping the optimizer, then the EMA converges
    to a far better estimate than the noisy one left by small-batch training."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average instead of EMA
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(x.cuda())
    model.eval()  # now eval uses the freshly recomputed, stable stats
    return model
```

A word on the `epsilon` in the BatchNorm denominator ($\sqrt{\sigma_B^2 + \epsilon}$): it is not just numerical hygiene. When the batch variance is genuinely tiny — a channel that is nearly constant across the batch, common early in training or for a dead channel — the $\epsilon$ (default $10^{-5}$) prevents a divide-by-near-zero that would blow the normalized activation up to a huge value and feed a spike into the next layer. If you ever see BatchNorm produce occasional huge activations at small batch, an underset $\epsilon$ interacting with a low-variance channel is a candidate; raising $\epsilon$ slightly is a legitimate stabilizer, and is one reason some small-batch recipes bump it to $10^{-3}$.

```python
import torch.nn as nn

# Vanilla BatchNorm — fine at large batch, broken at bs <= 8 and bs = 1.
norm_bn = nn.BatchNorm2d(num_channels)

# GroupNorm — batch-independent, identical in train and eval, no running stats.
# Rule: num_groups must divide num_channels. 32 is a common default;
# fall back to a divisor (e.g. gcd) for odd channel counts.
def make_groupnorm(num_channels, num_groups=32):
    g = num_groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)

norm_gn = make_groupnorm(num_channels)  # safe for any channel count
```

#### Worked example: a detector whose mAP would not settle

A two-stage object detector trained at **batch size 2 per GPU** (the images were $1333 \times 800$, so 2 was all that fit). The validation mAP swung between 31 and 38 across epochs that should have been monotonically improving, and the train/val gap made no sense. The instruments told the story: the BatchNorm running variances were jittering by 40–60% epoch to epoch, exactly the $\sqrt{2/(m-1)} \approx 141\%$-per-batch noise averaged down over many steps but never settling. Two days, again, suspecting the optimizer. The fix was to replace every `BatchNorm2d` with `GroupNorm(32, C)`. With statistics computed per-image, the batch-size-2 noise vanished, train and eval used the identical transform, and val mAP stabilized at 40.5 and climbed smoothly. The bisection that found it: overfit one batch (passed — the model code was fine), then read the running-stat instruments (the smoking gun), then confirm by swapping the normalizer (mAP stabilized). Figure 7, later, shows that before→after.

## 5. The running-stats traps: frozen BN, never-updating BN, and train/serve skew

BatchNorm's running statistics are a second source of bugs, separate from the train/eval mode switch. There are two failure modes, and they are opposites.

**Frozen BN that should not be frozen.** When you set `track_running_stats=False`, BatchNorm stops maintaining running statistics entirely — and crucially, it then uses **batch statistics even in eval mode**, because it has no running stats to fall back on. People reach for this flag thinking it "freezes" BatchNorm, but it does the opposite of what they want: it makes eval behave like train. If you actually want a frozen BatchNorm (common when finetuning a pretrained backbone on a small dataset, where you trust the pretrained statistics more than your tiny batch's), the correct move is to keep `track_running_stats=True`, load the pretrained running stats, and put just those BatchNorm modules into eval mode while the rest of the model trains:

```python
import torch.nn as nn

def freeze_batchnorm(model):
    """Freeze BatchNorm: use pretrained running stats, stop updating them,
    and stop learning the affine params. Correct for finetuning a backbone
    on a small dataset where your batch is too small for fresh BN stats.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()                       # use running stats, do not update them
            m.weight.requires_grad_(False) # freeze gamma
            m.bias.requires_grad_(False)   # freeze beta
    return model

# NOTE: a plain model.train() later will flip these BN layers back to train mode.
# Re-apply freeze_batchnorm() after any model.train(), or override .train()
# on the module to keep BN in eval. This re-flip is a classic silent regression.
```

That last comment is itself a bug magnet: calling `model.train()` at the top of your epoch loop silently un-freezes every BatchNorm you carefully froze, and now your tiny finetuning batch is overwriting good pretrained statistics with garbage. The signature is a finetune that starts strong (using the pretrained stats for the first eval) and then degrades over the first epoch as the running stats get polluted.

**Running stats that never update.** The opposite: you intend BatchNorm to learn fresh statistics, but they stay pinned at their initialization (`running_mean = 0`, `running_var = 1`) because the module is never in training mode during forward passes — for instance, you built the model, called `eval()` for a sanity check, and never called `train()` again before the loop; or you wrapped the whole forward in `torch.no_grad()` and a code path also forced eval. The confirming test is to print the running stats before and after a few training steps and check they actually move:

```python
import torch

def check_bn_running_stats_update(model, sample_batch, n_steps=5):
    """Confirm BatchNorm running stats are actually being updated in training.
    If running_mean/var do not change after several train-mode forwards,
    BN is stuck (model not in train mode, or track_running_stats=False)."""
    model.train()
    bns = [(n, m) for n, m in model.named_modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    before = {n: m.running_mean.clone() for n, m in bns}
    with torch.no_grad():
        for _ in range(n_steps):
            model(sample_batch)  # forward in train mode updates running stats
    for n, m in bns:
        delta = (m.running_mean - before[n]).abs().mean().item()
        status = "updating" if delta > 1e-6 else "FROZEN <-- bug?"
        print(f"{n:40s} running_mean delta = {delta:.2e}  {status}")
```

**Train/serve skew from batch composition.** Even with everything else correct, BatchNorm's running statistics encode the *distribution of your training batches*. If your serving distribution differs — you trained on balanced batches but serve a stream dominated by one class, or you trained on $224\times224$ crops but serve full-resolution images, or you trained with one normalization preprocessing and serve with another — the running mean and variance are mismatched to the serving inputs, and accuracy quietly drops at deployment even though offline eval (which used the matching val distribution) looked fine. This is a normalization-flavored [distribution shift](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world): the model "degraded" not because the weights are wrong but because the frozen statistics no longer match the inputs they are normalizing. The tell is an offline-vs-online gap that does *not* show up in your held-out val set because the val set shares the training batch distribution.

**The checkpoint-resume normalization signature.** Running statistics are *buffers*, not parameters, which means they live in the model's `state_dict` but are not touched by the optimizer — and that distinction is the root of a nasty resume bug. If your checkpoint-saving code saves only `model.parameters()` (the weights) and forgets the buffers, then on resume the BatchNorm running mean and variance reset to their initialization ($0$ and $1$), and the model's eval behavior changes discontinuously even though the weights are identical. The signature is a sharp *eval-metric jump on resume* — train loss continues smoothly from where it left off (the weights and optimizer state are fine) but the validation number drops a few points and then slowly recovers as the running stats re-accumulate. The confirming test is to compare a few BatchNorm `running_var` values before saving and after loading; if they snapped back to $1.0$, your save/load is dropping buffers. The fix is to always save and load the full `state_dict` (which includes buffers), not just the parameters. This is a normalization-flavored instance of the broader checkpoint-and-resume bug class, and it is easy to miss because the *training* curve looks perfectly continuous — only the eval curve betrays it.

### The BatchNorm-then-Dropout disharmony

A subtler interaction, documented by Li et al. (2019) in "Understanding the Disharmony between Dropout and Batch Normalization," explains a class of "I added Dropout and it got *worse*" bugs. Dropout, at training time, randomly zeros activations and scales the survivors so the *expected* activation is preserved — but it *increases the variance* of each activation (it is now sometimes zero, sometimes scaled up). At inference, Dropout is off, so the activation variance shifts. If a BatchNorm layer sits *after* a Dropout layer, its running variance is accumulated under the high-variance training distribution but applied to the low-variance inference distribution — a variance shift baked into the normalizer that hurts test accuracy. The clean rules that avoid this: put Dropout *after* the last BatchNorm in a block (so BN never sees the Dropout-inflated variance), or simply do not stack Dropout before BatchNorm. The signature is a model that trains fine but whose test accuracy is mysteriously a point or two worse than expected, improving when you remove or reorder the Dropout. It is the same family of bug as the augmentation interaction above: a training-time perturbation widens the distribution BatchNorm measures, and the eval-time mismatch costs you accuracy.

### Normalization in tabular and MLP models

Normalization bugs are not just a vision-and-transformers story; tabular deep nets hit them too, often in a more pernicious form because tabular features have wildly different natural scales. The most common tabular normalization bug is not in a `nn.BatchNorm1d` layer at all — it is in the *feature preprocessing*: fitting a `StandardScaler` (which subtracts the mean and divides by the std) on the full dataset before splitting, which leaks test-set statistics into training and inflates your validation score, a [data-leakage](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) cousin of the normalization-stats problem. But the in-model version bites too: `nn.BatchNorm1d` on a tabular MLP run at small batch (common with wide tables and limited memory) reproduces the small-batch variance failure from Section 4 exactly, and because tabular validation sets are often tiny, the bouncing-metric symptom is even louder. The fix is the same — GroupNorm or LayerNorm for batch-independence, or simply ensure the batch is large enough — but the *diagnosis* is muddier because tabular practitioners rarely instrument activation statistics. The discipline transfers directly: print the per-layer activation std after init, confirm the batch size against the $\sqrt{2/(m-1)}$ rule, and make sure your scaler is fit on the *training fold only*. A tabular MLP whose val AUC swings by 0.05 across folds is very often a small-batch BatchNorm or a leaked scaler, not an unstable model.

### A diagnostic decision tree for normalization bugs

Putting the BatchNorm family together, Figure 5 is the decision tree from symptom to suspect to confirming test. A metric that is *unstable* across evaluations points to eval-mode or small-batch BatchNorm; a metric that is *flat at chance* with a very deep net points to init; a *frozen* running stat points to the mode/flag traps.

![Decision tree starting from an unstable metric or train-eval gap, branching to BatchNorm mode checks and deep-net activation checks, ending in the targeted fix such as calling eval, switching to GroupNorm, or correcting init](/imgs/blogs/initialization-and-normalization-bugs-5.png)

| Symptom | Likely cause | Confirming test | Fix |
|---|---|---|---|
| Val metric bounces run-to-run on the same checkpoint | Forgot `model.eval()`, BN using batch stats | `assert not model.training` before eval | Call `model.eval()` + `torch.no_grad()` |
| Loss flat at chance, very deep net | Init variance wrong, signal vanished/exploded | Print activation std per layer after init | He/Xavier init, check gain $\approx 1$ |
| Eval far worse than train, small batch | BN batch stats too noisy at `bs ≤ 8` | Check batch size; print running-var jitter | GroupNorm (or SyncBN, or accumulate stats) |
| BN output constant for every input | Batch size 1, variance undefined | Print BN output variance across a batch | GroupNorm / LayerNorm, or raise batch size |
| Finetune starts strong then degrades | Pretrained BN stats overwritten by tiny batch | Print `running_mean` drift over first epoch | Freeze BN (eval mode + `requires_grad=False`) |
| `running_mean` stuck at 0, `running_var` at 1 | `track_running_stats=False` or never in train mode | Print stats before/after train steps | Set `track_running_stats=True`, ensure `train()` |
| Offline great, online drops, val never caught it | Train/serve batch-distribution skew | Compare serving input stats to BN running stats | Match preprocessing; consider GroupNorm |

## 6. LayerNorm: placement is the bug (pre-LN vs post-LN)

Transformers do not use BatchNorm — they use **LayerNorm**, which normalizes over the feature dimension of each token independently. Because LayerNorm's statistics are per-token, it has none of BatchNorm's batch-dependence problems: no train/eval mode discrepancy, no small-batch failure, no running statistics. But it has its own bug, and it is about **placement**, not statistics.

The original Transformer (Vaswani et al., 2017) used **post-LN**: the normalization is applied *after* the residual addition, $\mathbf{x}_{\text{out}} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$. Modern large models almost universally use **pre-LN**: the normalization is applied *inside* the residual branch, before the sublayer, and the residual path itself is left un-normalized: $\mathbf{x}_{\text{out}} = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$. The difference is not aesthetic — it changes the **gradient scale at the residual stream** and therefore the stability of very deep transformers.

Here is the mechanism, made rigorous enough to predict the bug. In **post-LN**, the residual stream passes *through* a LayerNorm at every block, so the gradient flowing backward is repeatedly rescaled by the LayerNorm Jacobian, and in deep stacks this compounds into gradients that are large near the output and require careful warmup and a low initial learning rate to avoid divergence — post-LN transformers are notoriously sensitive to learning rate and warmup, and a too-aggressive schedule produces an early loss spike. In **pre-LN**, the residual path is an identity highway with no normalization on it, so gradients flow back through the residual connections essentially un-rescaled (the identity has Jacobian 1), the effective gradient at every layer is well-behaved, and the model trains stably with much less warmup sensitivity. The cost is that pre-LN slightly reduces the effective depth (the residual stream can dominate the sublayers), which is why some recent work revisits post-LN with careful init scaling. The practical bug: if you build a deep transformer with post-LN and a standard high learning rate and short warmup, you get an **early loss spike or divergence** that looks like a learning-rate problem but is actually a normalization-placement problem. Swap to pre-LN (or extend warmup and drop the LR) and it trains. This is closely tied to [gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing) — placement is one of the levers that sets whether the residual-stream gradient is stable.

The diagnostic is the same per-layer grad-norm audit from Section 2, applied to the residual stream: in a misbehaving post-LN model you will see the grad norm grow toward the output layers; in a healthy pre-LN model it stays flat across depth. The other LayerNorm bugs are mundane but real: **missing normalization** (you forgot a LayerNorm in a custom block and the activations drift), **extra/double normalization** (two LayerNorms back to back, which over-constrains the signal), and **normalizing the wrong dimension** (LayerNorm over the sequence axis instead of the feature axis, a shape bug dressed as a normalization bug — print the `normalized_shape` and confirm it matches your feature dimension).

#### Worked example: a custom transformer block that exploded at depth

A team built a 24-layer decoder-only transformer from scratch with a custom block, and it NaN'd at step 30 every time. The instruments showed the residual-stream activation norm climbing block by block — `1.0`, `1.6`, `2.5`, `3.9`, ... — a clean geometric explosion. The learning-rate sweep did nothing, which is the tell that it is not an optimization bug. The root cause was two compounding mistakes: they had used **post-LN** placement *and* they had not scaled the residual branch, so each block added an unnormalized sublayer output to the residual stream and the variance grew roughly $1.5\times$ per block ($1.5^{24} \approx 16{,}800$, comfortably enough to overflow fp16 by the upper layers). The bisection: overfit one batch failed *the same way* (NaN at step 30), ruling out the data pipeline and pointing squarely at model code or numerics; the per-block activation audit showed the geometric climb, localizing it to the residual structure; switching to **pre-LN** placement and scaling the output projection of each block by $1/\sqrt{2L}$ (a standard deep-transformer init) made the activation norm flat across depth and the loss descended cleanly from $11.0$ to $3.2$. Symptom: NaN at step 30, residual norm climbing $1.5\times$/block. Confirming test: per-block activation-norm audit. Fix: pre-LN + residual-branch scaling. This is the LayerNorm-placement analog of the residual-init bug from Section 1 — same geometric-explosion signature, fix in the normalization placement and branch scale rather than the per-weight variance.

### RMSNorm and the normalization bugs of modern LLMs

The newest large language models (LLaMA, many Mistral-family models, and others) replace LayerNorm with **RMSNorm** (Zhang and Sennrich, 2019), which drops the mean-subtraction and the bias and normalizes only by the root-mean-square of the features: $\hat{x} = x / \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}$, then scaled by a learned $\gamma$. RMSNorm is cheaper (no mean to compute) and empirically just as stable, but it introduces a finetuning bug that bites people moving between codebases: the **`eps` placement and value** differs between implementations, and a mismatch shifts the normalization scale enough to degrade a finetune. Worse, some implementations compute RMSNorm in fp32 and cast back, while others stay in bf16; the numerical difference is small at inference but can change which checkpoint a finetune converges to, and a checkpoint trained with one convention loaded into code with the other shows a small but real quality regression. The diagnostic is to print the norm layer's `eps` and the dtype of its internal computation and match them to the checkpoint's training code. It is the LLM-era version of the BatchNorm running-stats mismatch: a normalization detail that is invisible until it silently moves your numbers.

The other modern-LLM normalization trap is **finetuning a model whose norm layers are frozen or wrongly trainable.** With parameter-efficient finetuning (LoRA/PEFT), the normalization layers' learned scales ($\gamma$, and $\beta$ for LayerNorm) are usually *not* adapted, which is correct — but some recipes deliberately unfreeze the norm layers (the "train the norms" trick that recovers a bit of quality), and if you unfreeze them at the wrong learning rate you can destabilize a finetune that would otherwise be fine. The signature is a LoRA finetune that trains smoothly until you add the norm layers to the trainable set, then develops a loss wobble. The fix is a separate, much smaller learning rate for the norm parameters, or leaving them frozen. The general principle holds across every modality: **a normalization layer's learned scale is a high-leverage parameter, so changing whether and how fast it trains changes stability.**

## 7. The interactions: BatchNorm under accumulation, DDP, and heavy augmentation

BatchNorm's batch-statistic dependence creates a class of bugs that only appear when something *changes the effective batch the layer sees* — and three common training tricks do exactly that, silently.

**Gradient accumulation.** To simulate a large batch on limited memory, you run several forward/backward passes with small micro-batches and only step the optimizer after accumulating their gradients. This correctly emulates a large batch *for the optimizer* — the summed gradient equals the gradient of the big batch (see [the gradient-accumulation rules](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) for the loss-normalization subtlety) — but it does **not** emulate a large batch *for BatchNorm*. Each micro-batch's BatchNorm sees only the micro-batch size, computes its statistics over that small sample, and updates the running stats with noisy small-batch estimates. So if you "accumulate 8 micro-batches of 8 to get an effective batch of 64," your BatchNorm is running at batch size 8, not 64, with all the small-batch noise that implies. The optimizer thinks the batch is 64; BatchNorm knows it is 8. The fix is to either use a normalizer without batch dependence (GroupNorm/LayerNorm), or recognize that accumulation does not help BatchNorm and size your micro-batch large enough on its own.

**Distributed Data Parallel.** Under DDP, each GPU (rank) processes its own shard of the batch and BatchNorm, by default, computes statistics **only over the local rank's samples** — not across ranks. So "8 GPUs × batch 8 = effective batch 64" again gives BatchNorm a batch of 8, per rank, independently. For large per-rank batches this is fine; for the small per-rank batches common in detection and segmentation it reproduces the small-batch failure on every rank. The fix is **SyncBN** (`torch.nn.SyncBatchNorm.convert_sync_batchnorm`), which all-reduces the statistics across ranks so BatchNorm sees the true global batch:

```python
import torch
import torch.nn as nn

# Convert every BatchNorm in the model to its cross-rank synchronized version.
# Do this BEFORE wrapping in DDP. Each BN now computes mean/var over the
# global batch (all ranks), not just the local shard.
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# Alternative for small per-rank batch: skip BN entirely and use GroupNorm,
# which is batch-independent and needs no cross-rank communication.
```

Figure 6 shows both paths converging on the fix: accumulation and DDP both shrink the batch BatchNorm sees, and both are repaired by either SyncBN (make BatchNorm see the global batch) or GroupNorm (remove the batch dependence).

![Dataflow graph showing an intended batch of sixty-four split by gradient accumulation and by distributed data parallel into noisy small per-step or per-rank statistics, then repaired by SyncBN gathering across ranks or by GroupNorm removing batch dependence](/imgs/blogs/initialization-and-normalization-bugs-6.png)

**Heavy augmentation.** A subtler interaction: very aggressive augmentation (strong color jitter, large-scale RandAugment, heavy mixup) widens the activation distribution that BatchNorm sees during training, so its running statistics are tuned to a *more dispersed* distribution than the clean images it normalizes at eval. The mismatch is usually small, but with extreme augmentation it can produce a measurable train/eval gap that — once again — looks like overfitting but is actually a normalization-statistics mismatch. The tell is that the gap shrinks if you recompute BatchNorm running stats on clean (un-augmented) data with a forward-only pass before eval (sometimes called "BN recalibration" or "precise BN"). It is rarely the dominant effect, but it belongs on the list because it is invisible unless you know normalization statistics depend on the input distribution.

#### Worked example: small-batch detection, the full bisection

Returning to the detector from Section 4, here is the disciplined bisection in full, because it shows the method as much as the fix. **Symptom:** val mAP bounces 31↔38, should be climbing smoothly. **Step 1 — overfit one batch.** Train on a single batch of 2 images for 200 steps; loss drives to near zero and mAP on that batch hits ~100. *Conclusion: the model code, loss, and data pipeline are correct — the bug is not there.* This rules out four of the six corners in one test. **Step 2 — read the instruments.** Log the BatchNorm running-variance for a few channels every epoch; they jitter by 40–60%. *Conclusion: the suspect is normalization statistics, and the batch size of 2 is the cause.* **Step 3 — confirm with a swap.** Replace `BatchNorm2d` with `GroupNorm(32, C)` and rerun. *Result: val mAP stops bouncing and settles at 40.5, climbing monotonically.* Before→after in one table:

| Instrument | Before (BatchNorm, bs=2) | After (GroupNorm) |
|---|---|---|
| Val mAP across last 5 epochs | 31, 38, 33, 37, 34 (bouncing) | 39.8, 40.1, 40.3, 40.5, 40.5 (rising) |
| BN/GN running-var jitter epoch-to-epoch | 40–60% | n/a (no running stats) |
| Train/eval transform | different (batch vs running stats) | identical |
| Time to root-cause once bisected | — | ~20 minutes |

Figure 7 is this before→after as a diagram: small-batch BatchNorm with its noisy statistics on the left, GroupNorm with its stable per-image statistics on the right.

![Two-column before-after diagram showing small-batch BatchNorm at batch size two with high variance error and bouncing detection mAP versus GroupNorm with no batch dependence and stable mAP](/imgs/blogs/initialization-and-normalization-bugs-7.png)

## 8. Putting it together: the init-and-norm preflight

Init and normalization bugs are the cheapest of all bugs to *prevent* and among the most expensive to *chase*, which is exactly the wrong ratio. The fix is a preflight you run before every new architecture or finetune. Figure 8 is the full symptom→test→fix matrix you can pin above your desk.

![Matrix table mapping four init and normalization bugs to their instrument symptom, the one-line confirming test, and the fix, from wrong init to forgotten eval to small-batch BatchNorm to frozen running stats](/imgs/blogs/initialization-and-normalization-bugs-8.png)

```python
import torch
import torch.nn as nn

def init_and_norm_preflight(model, sample_input):
    """Run BEFORE training. Catches the whole init/norm bug family in seconds."""
    # 1) Activation scale across depth (init audit).
    print("=== activation std per layer (want ~0.3 to 3 across depth) ===")
    audit_activation_scale(model, sample_input)  # from Section 2

    # 2) BatchNorm inventory: how many, and the small-batch risk.
    bns = [(n, m) for n, m in model.named_modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    bs = sample_input.shape[0]
    print(f"\n=== {len(bns)} BatchNorm layers, batch size = {bs} ===")
    if bns and bs <= 8:
        print(f"WARNING: batch size {bs} is small for BatchNorm "
              f"(var error ~{(2/(bs-1))**0.5:.0%}). Consider GroupNorm/SyncBN.")
    if bns and bs == 1:
        print("ERROR: batch size 1 with BatchNorm — variance undefined. "
              "Use GroupNorm/LayerNorm or raise batch size.")

    # 3) Confirm the eval/train mode discipline is wired (manual reminder).
    print("\nReminder: call model.eval() before every eval, model.train() after.")

# usage
# init_and_norm_preflight(model, torch.randn(32, 3, 224, 224, device="cuda"))
```

This is the make-it-fail-small philosophy applied to the two corners init and normalization occupy: one forward pass reads the activation scale (catches init), an inventory plus the batch size flags the small-batch BatchNorm risk, and a discipline reminder closes the eval-mode loop. Three seconds, and you have ruled out an entire bug family before the first gradient step.

### Reading the effective batch BatchNorm actually sees

When you combine the tricks of modern training — gradient accumulation, multi-GPU, and a per-GPU micro-batch — it is genuinely hard to keep track of what batch size BatchNorm experiences, and that confusion is where the silent bug lives. So make it explicit with one number. The *effective batch for the optimizer* is $b_{\text{opt}} = b_{\text{micro}} \times n_{\text{accum}} \times n_{\text{gpu}}$. The *effective batch for BatchNorm* (without SyncBN) is just $b_{\text{micro}}$ — the per-step, per-rank micro-batch — because each BatchNorm forward sees only its own micro-batch and there is no cross-rank reduction. Those two numbers can differ by an order of magnitude. A run with $b_{\text{micro}} = 4$, $n_{\text{accum}} = 8$, $n_{\text{gpu}} = 8$ has $b_{\text{opt}} = 256$ (the optimizer sees a healthy large batch) but $b_{\text{BN}} = 4$ (BatchNorm sees a tiny one, with $\sqrt{2/3} \approx 82\%$ variance-estimate error). The optimizer is happy; BatchNorm is drowning in noise. The fix is to compute both numbers in your config and, if $b_{\text{BN}} \leq 8$, either convert to SyncBN (raises $b_{\text{BN}}$ to $b_{\text{micro}} \times n_{\text{gpu}} = 32$, healthy) or switch to GroupNorm. Print these two numbers at the top of every run; the gap between them is the single most under-appreciated normalization diagnostic in distributed training.

| Configuration | $b_{\text{opt}}$ (optimizer) | $b_{\text{BN}}$ (BatchNorm, no SyncBN) | Verdict |
|---|---|---|---|
| micro 64, accum 1, 1 GPU | 64 | 64 | Healthy |
| micro 8, accum 8, 1 GPU | 64 | 8 | BN borderline noisy |
| micro 4, accum 8, 8 GPU | 256 | 4 | BN failing — SyncBN or GroupNorm |
| micro 4, accum 8, 8 GPU + SyncBN | 256 | 32 | Healthy (cross-rank stats) |

## Case studies and real signatures

A few well-known patterns and results, accurately attributed, to anchor the mechanisms above in the literature and in production folklore.

**He initialization unlocked very deep plain nets.** He et al. (2015), "Delving Deep into Rectifiers," showed that a 30-layer plain (non-residual) ReLU network *failed to converge* under Xavier initialization but trained fine under the $2/n$ He init — precisely the factor-of-two ReLU correction derived in Section 1. This is the original public demonstration that the init constant is the difference between training and not training a deep net, and it is the direct ancestor of the worked example in this post.

**BatchNorm's batch-size sensitivity, quantified.** Wu and He (2018), "Group Normalization," report that ResNet-50 on ImageNet with BatchNorm degrades sharply as batch size drops — the error climbs steeply below batch size 8 or so — while GroupNorm's error stays essentially flat across batch sizes from 32 down to 2. That flat-vs-cliff comparison is the empirical face of the $\sqrt{2/(m-1)}$ variance-error law: BatchNorm's accuracy falls off exactly where its statistics estimate becomes unreliable. This is the citation behind the detection worked example.

**The forgotten-`eval()` bug is a perennial.** It is not in a paper because it is too embarrassing to publish, but it is one of the most frequently reported issues on the PyTorch forums and in framework FAQs: validation/test metrics that are non-deterministic or worse than expected because Dropout and BatchNorm were left in training mode. The PyTorch documentation explicitly warns that `model.eval()` must be called to set BatchNorm and Dropout to evaluation behavior, and that `torch.no_grad()` is a separate concern. The signature — a metric that bounces on a fixed checkpoint — is unmistakable once you know to look for it.

**Pre-LN vs post-LN stability.** Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture," analyzed the gradient scale at initialization for both placements and showed that post-LN transformers have gradients that grow with depth near the output, requiring warmup to train, while pre-LN transformers have well-behaved gradients and can train without warmup. This is the rigorous backing for Section 6: placement, not the LayerNorm itself, sets deep-transformer stability.

**You can train very deep nets without normalization if you fix the init.** Zhang et al. (2019), "Fixup Initialization," showed that a residual network with *no* BatchNorm at all can match a normalized one if you scale the residual branches correctly at init (roughly down-scaling each branch by a depth-dependent factor and zeroing the last layer of each block). This is the empirical proof of the residual-variance argument from Section 1: the reason normalization-free residual nets explode is the $2\times$-per-block variance growth, and controlling it at init removes the need for the normalizer entirely. The practical lesson for debugging: if a normalization-free residual model NaNs at step 0, do not reach for "add BatchNorm" reflexively — reach for the residual-branch init scaling, because that is the actual mechanism.

**Internal covariate shift versus loss-smoothing.** The Santurkar et al. (2018) result mentioned in Section 3 is itself a useful case study in *not trusting the stated mechanism*: BatchNorm's original paper attributed its benefit to reducing covariate shift, but controlled experiments showed you can inject covariate shift back in and BatchNorm still helps, pointing to loss-landscape smoothing as the real driver. For a debugger this matters because it tells you what to measure: not "is the layer-input distribution shifting" but "are my gradients well-conditioned and is my learning rate stable" — which is exactly the per-layer grad-norm instrument from Section 2.

## When this is (and isn't) your bug

Be decisive about ruling init/normalization *in* and *out*, because the failure modes overlap with optimization bugs and you do not want to spend a day in the wrong corner.

It **is** an init bug when: the net is deep (say > 15–20 layers without residuals or normalization), the loss is *flat at chance from step 0* with no movement, and the per-layer activation std decays or grows geometrically with depth. The flatness-from-the-start is the key tell — a bad init does not "train slowly," it produces no gradient signal at all in the deep layers. If the loss moves at all, even slowly, your init is probably fine and you have an [optimization](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing) or learning-rate problem instead.

It **is** a normalization bug when: the **train metric looks fine but the eval/val metric is unstable, much worse, or non-deterministic** on a fixed checkpoint. That train/eval discrepancy is the fingerprint of a mode or statistics problem — BatchNorm using the wrong stats at eval, or small-batch stats that are too noisy. A bouncing val number on identical weights is *always* a statistics or mode issue, never the optimizer (the optimizer does not touch the weights during eval).

It is **not** your bug — look elsewhere — when: overfit-one-batch *passes* and the activation audit is clean. If the model can drive loss to zero on one batch and the activation std is flat across depth, init and normalization are doing their job; the bug is in data, the loss function, or evaluation. A *smooth-then-NaN* curve is numerics (overflow, a bad batch, too-high LR), not init — init bugs are NaN/flat from the start, not after a healthy descent. And a metric that is *stable but simply wrong* (consistently 71% when you expect 90%, with no bouncing) points to a data or label problem, not normalization — normalization bugs make metrics *unstable* or create a *train/eval gap*, they do not produce a steady wrong number. When init and normalization are clean and the symptom persists, return to the [taxonomy decision tree](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) and bisect to a different corner.

There is one more discrimination worth making explicit, because it separates an init bug from a normalization bug when both could plausibly produce a deep-net failure. An init bug is a *step-0* phenomenon: the signal is dead or exploded before any weight has moved, so the very first forward pass already shows the geometric activation profile, and the loss is pinned at chance (or NaN) from the first batch. A normalization bug, by contrast, usually shows up as a *train/eval discrepancy* or as a *gradual* degradation — the model trains, the train loss falls, but the eval bounces or the run slowly drifts as polluted running stats accumulate. So the timing of the symptom is itself diagnostic: dead-from-step-0 points to init; healthy-train-but-broken-eval points to normalization; healthy-then-degrades points to running-stats pollution or a frozen-then-unfrozen BN. Combine that with the two instruments — activation std after init, and eval-vs-train metric stability on a fixed checkpoint — and you can usually name the corner before writing a single line of fix code.

A final caution against over-attributing to normalization: not every train/eval gap is a normalization bug. A genuine *overfitting* gap (train loss low, val loss high, both stable and reproducible) is a regularization/data problem, not a BatchNorm problem — the tell is that overfitting is *stable and reproducible* across eval runs, whereas a BatchNorm mode bug makes the *same checkpoint* report different numbers on repeated evals. And a *data leak* makes the eval look implausibly good rather than unstable. Use the stability of the eval number as your discriminator: unstable → mode/stats bug; stably-too-good → leak; stably-too-bad-with-low-train → overfitting or distribution shift. Each points to a different one of the [six corners](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), and the whole discipline of this series is to read the signature before you start changing code.

## Key takeaways

- **Init sets a signal budget; depth compounds it.** The activation variance at layer $L$ is the input variance times $g^L$, where the per-layer gain $g = n_{\text{in}}\sigma_w^2$ (times the nonlinearity's survival fraction). Set $\sigma_w^2 = 2/n_{\text{in}}$ for ReLU (He) so $g \approx 1$; a 30% error in $g$ is invisible at layer 2 and fatal at layer 40.
- **A flat loss at chance from step 0, on a deep net, is an init bug** — not a learning-rate bug. Confirming test: print the activation std per layer; geometric decay/growth is the signature. Fix: He/Xavier init. Warmup hides a *near-miss* init and does nothing for a catastrophic one.
- **BatchNorm has two modes and that is the whole minefield.** Train mode normalizes with batch stats and updates running stats; eval mode uses frozen running stats. Forgetting `model.eval()` makes eval use noisy batch stats — the signature is a val metric that *bounces on a fixed checkpoint*. Fix: `model.eval()` + `torch.no_grad()`, and `model.train()` after.
- **Small batch breaks BatchNorm provably.** The variance-estimate error is $\sqrt{2/(m-1)}$ — about 25% at $m=32$, 53% at $m=8$, undefined at $m=1$. Below batch size 8, switch to **GroupNorm** (per-image, batch-independent, no train/eval gap, no running stats) or **SyncBN** under DDP.
- **Accumulation and DDP do not give BatchNorm the big batch you think.** Each micro-batch / each rank computes stats over its own small shard. The optimizer sees the effective batch; BatchNorm sees the micro-batch. Fix: SyncBN (global stats) or GroupNorm (no batch dependence).
- **Running-stats traps come in opposite flavors.** `track_running_stats=False` makes eval use batch stats (the opposite of "frozen"); a forgotten `model.train()` leaves running stats pinned at init. Freeze a backbone's BN correctly with eval mode + `requires_grad=False`, and re-apply it after every `model.train()`.
- **LayerNorm's bug is placement, not statistics.** Post-LN transformers have output-heavy gradients and need warmup; pre-LN has identity-highway residuals and trains stably. A deep post-LN model with a high LR spikes early — that is a placement bug masquerading as a learning-rate bug.
- **Run the preflight.** One forward pass for the activation audit, a BatchNorm inventory against the batch size, and the eval-mode discipline. Three seconds prevents the week-long chase.

## Further reading

- Glorot, X. and Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." The original variance-preservation argument and Xavier/Glorot init.
- He, K., Zhang, X., Ren, S., and Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." The He/Kaiming $2/n$ init and the factor-of-two ReLU correction.
- Ioffe, S. and Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." The original BatchNorm, including the train/eval running-statistics mechanism.
- Wu, Y. and He, K. (2018). "Group Normalization." The batch-size cliff of BatchNorm and the batch-independent GroupNorm fix.
- Ba, J., Kiros, J., and Hinton, G. (2016). "Layer Normalization." The per-token normalizer used in transformers.
- Zhang, B. and Sennrich, R. (2019). "Root Mean Square Layer Normalization." RMSNorm, the mean-free normalizer used in many modern LLMs.
- Xiong, R. et al. (2020). "On Layer Normalization in the Transformer Architecture." The pre-LN vs post-LN gradient-scale analysis behind warmup sensitivity.
- Santurkar, S., Tsipras, D., Ilyas, A., and Madry, A. (2018). "How Does Batch Normalization Help Optimization?" The loss-landscape-smoothing view of BatchNorm's benefit.
- Zhang, H., Dauphin, Y., and Ma, T. (2019). "Fixup Initialization: Residual Learning Without Normalization." Training deep residual nets with init scaling instead of BatchNorm.
- Li, X., Chen, S., Hu, X., and Yang, J. (2019). "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift." Why Dropout before BatchNorm hurts test accuracy.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). "Focal Loss for Dense Object Detection." The prior-probability bias init for the classifier head under heavy imbalance.
- PyTorch documentation: `torch.nn.BatchNorm2d`, `torch.nn.GroupNorm`, `torch.nn.SyncBatchNorm`, `torch.nn.init`, and the `Module.train()/eval()` semantics.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the symptom→suspect→test→fix decision tree this post instantiates), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone checklist), and the sibling tracks [gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing), [dead neurons and saturated activations](/blog/machine-learning/debugging-training/dead-neurons-and-saturated-activations), and [train/eval mode bugs](/blog/machine-learning/debugging-training/train-eval-mode-bugs).

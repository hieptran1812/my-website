---
title: "Mixed-precision quantization: which layers tolerate low bits, and how to decide"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A practical, math-grounded guide to per-layer sensitivity: how to measure which layers survive int4, which must stay fp16, and how to allocate bits to hit a size budget with the smallest accuracy loss."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "mixed-precision",
    "sensitivity-analysis",
    "hawq",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/mixed-precision-and-sensitivity-analysis-1.png"
---

The first model I shipped to a phone NPU lost two and a half points of top-1 accuracy the moment I converted it to uniform int8, and the product manager noticed before I did. The fix that "everyone knows" — quantization-aware training — would have cost me a week of retraining and a new data pipeline I did not have. So I did the lazy thing first: I quantized one layer at a time, kept the rest in float, and watched the validation loss. Within an afternoon a pattern fell out of the noise that I have now seen in every network I have ever quantized. A handful of layers were responsible for almost all of the damage. The other forty were essentially free to crush down to four bits. Once I let those forty layers go to int4 and kept the guilty handful in fp16, the model was *smaller* than the uniform int8 version and lost only half a point. That is the entire idea of mixed-precision quantization, and the hard part is not the quantizing — it is deciding which layers are which.

Uniform quantization treats every layer the same: pick a bit-width, apply it everywhere, hope for the best. It is the right first move because it is simple and it usually gets you most of the way. But "the same bit-width everywhere" is almost never the right answer, because layers are not the same. Some layers sit in a flat, forgiving region of the loss surface where rounding their weights to int4 barely moves the output. Others sit on a knife-edge where the smallest rounding error in their weights propagates into a visible accuracy drop. Uniform int8 leaves something on the table no matter which way you set the dial: set it conservatively (int8 or fp16 everywhere) and you waste bits — and therefore size, bandwidth, and energy — on layers that would have been fine at int4. Set it aggressively (int4 everywhere) and a few sensitive layers tank your accuracy and drag the whole model down with them.

Mixed precision is the resolution: assign each layer the *lowest* bit-width it can personally tolerate. Figure 1 is the thing you are trying to discover — a per-layer profile of how much accuracy each layer costs you at int8 versus int4 — and it is the map the rest of this post teaches you to draw and then act on. Notice the shape: the stem and the classifier are expensive, the middle of the network is nearly free. That shape is not a quirk of one model. It is a near-universal regularity, and there is a real reason for it that we will derive from the curvature of the loss surface.

![A matrix figure listing six layers of a network with their accuracy drop at int8 and int4 and a verdict for each, showing first and last layers most sensitive and middle layers tolerant](/imgs/blogs/mixed-precision-and-sensitivity-analysis-1.png)

By the end of this post you will be able to do three concrete things. First, *measure* per-layer sensitivity two different ways — a cheap label-free signal method and a more faithful loss-perturbation method — and understand the Hessian-based theory (HAWQ) that explains why curvature predicts the damage. Second, *allocate* bits across layers as a constrained optimization — minimize accuracy loss subject to a size or latency budget — using a greedy loop you can write in forty lines and read off a Pareto frontier. Third, *decide* whether mixed precision is even worth the complexity for your target, because on an int8-only NPU the answer is frequently "no, and here is what to do instead." This sits inside the series' four-lever frame — quantization is one lever, and mixed precision is how you tune that lever per-layer; see [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the map and [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) for the mechanics this post assumes.

## 1. The problem with one bit-width for the whole model

Let me make the waste concrete before we get scientific about it, because the intuition is what makes the math stick.

A neural network at inference is a stack of layers, each holding a tensor of weights. Quantization replaces each 32-bit float weight with a low-bit integer plus a per-tensor (or per-channel) scale, so that the integer times the scale approximates the original float. With $b$ bits you get $2^b$ representable levels. The error you introduce per weight is the rounding error — the gap between the true value and the nearest representable level. We will quantify that error precisely in the next section, but the headline you need now is that the error gets *worse fast* as $b$ shrinks: going from int8 (256 levels) to int4 (16 levels) makes each level sixteen times coarser, so the typical rounding error grows by roughly the same factor.

Here is the part people miss. That coarser error does not hurt every layer equally. Two layers can have identical weight distributions and identical quantization error, yet one of them shrugs it off and the other collapses, because *what matters is not the error in the weights, it is the error those weights cause in the loss*. A layer whose output the rest of the network barely depends on can be quantized to mush. A layer the network leans on heavily — the first layer that sees the raw input, the last layer that produces the logits, an attention projection that everything downstream conditions on — turns a small weight error into a large output error and a large loss increase.

So a single global bit-width forces a single compromise on a population of layers with wildly different tolerances. Concretely, on a ResNet-style classifier I will use as the running example, the spread in sensitivity across layers is about an order of magnitude — the most sensitive layer costs you roughly ten times the accuracy per bit removed compared to the most tolerant one. Faced with that spread, uniform quantization has only bad options:

- **Uniform int8.** Safe, simple, well-supported on every accelerator. But you are paying 8 bits for middle layers that would have been completely fine at 4, which means your model is roughly twice as big as it needs to be in those layers, and on a bandwidth-bound target that extra size is extra latency and extra energy you did not need to spend.
- **Uniform int4.** Half the size again, but now the two or three sensitive layers blow up. On the ResNet, uniform int4 (naive, post-training) typically drops 4 to 8 points of top-1 — unshippable — and *almost all* of that drop comes from the stem convolution and the final fully-connected layer.

The mixed-precision answer is to refuse the false choice. Keep the sensitive layers at int8 or fp16, push the tolerant ones to int4, and you land at a point that is *strictly better* than uniform int8: same or smaller size, much less accuracy loss than uniform int4. That is a free lunch — but only if you can correctly identify which layers are which, and only if your hardware can actually run more than one precision. Both of those caveats are load-bearing, and we will spend the rest of the post on them.

One framing to carry through: this is fundamentally a *budget allocation* problem. You have a fixed bit budget (set by your size or latency target) and a set of layers, each with a different "exchange rate" of bits-for-accuracy. You want to spend your bits where they buy the most accuracy and starve the layers that do not need them. That is an optimization problem with a clean objective, and naming it that way — instead of "vibes about which layers feel important" — is what turns mixed precision from folklore into engineering.

To make the exchange-rate idea concrete with arithmetic rather than adjectives: suppose two layers each have a million parameters. Layer A is the tolerant middle of the network; dropping it from int8 to int4 saves 4 bits per weight — half a megabyte — and costs 0.2 points of accuracy. Layer B is the classifier; the identical demotion saves the identical half a megabyte but costs 4.0 points. Same size saving, twenty times the accuracy cost. If you have to give back half a megabyte to hit a budget, the choice is obvious — take it from A, never from B — and that ratio of *size-saved per accuracy-lost* is the single number the whole allocation problem turns on. The job of sensitivity analysis is to estimate that ratio for every layer; the job of bit allocation is to spend your budget on the layers with the best ratios first. Everything in this post is an elaboration of those two sentences.

## 2. The science: why curvature predicts quantization sensitivity

This is the section that makes the rest non-magical. We want a *predictive* notion of sensitivity: a number $s_L$ for each layer $L$ such that "high $s_L$ means quantizing this layer hurts a lot." It turns out there is a principled one, and it comes straight out of a second-order Taylor expansion of the loss. This is the theory behind HAWQ (Dong et al., 2019) and its descendants, and the derivation is short enough to do in full.

### A second-order Taylor expansion of the loss

Let $W$ be the full set of trained weights and $\mathcal{L}(W)$ the loss on your evaluation data. Quantizing the weights perturbs them: $W \to W + \Delta W$, where $\Delta W$ is the quantization error — the difference between each weight and its nearest representable level. We want $\Delta \mathcal{L} = \mathcal{L}(W + \Delta W) - \mathcal{L}(W)$, the loss increase quantization causes. Expand the loss around the trained weights to second order:

$$
\mathcal{L}(W + \Delta W) \approx \mathcal{L}(W) + g^\top \Delta W + \tfrac{1}{2}\, \Delta W^\top H\, \Delta W
$$

where $g = \nabla_W \mathcal{L}$ is the gradient and $H = \nabla_W^2 \mathcal{L}$ is the Hessian, both evaluated at $W$. Now use a fact that is the whole reason this works: a *converged* model sits at (or very near) a minimum of the loss, so the gradient $g \approx 0$. The first-order term vanishes. We are left with

$$
\Delta \mathcal{L} \approx \tfrac{1}{2}\, \Delta W^\top H\, \Delta W .
$$

Read that equation slowly, because it is the entire theory. The loss increase from quantization is a *quadratic form in the Hessian*. The quantization error $\Delta W$ comes from rounding and is roughly the same magnitude in every layer for a given bit-width (it is set by the bit-width and the weight range, not by the layer's role). What differs between layers is $H$ — the curvature of the loss in that layer's weight directions. **A layer in a flat region (small Hessian) can absorb a large $\Delta W$ for almost no loss increase; a layer in a sharply curved region (large Hessian) turns the same $\Delta W$ into a large loss increase.** Curvature *is* sensitivity. That is not an analogy; it falls directly out of the Taylor expansion.

### From the full Hessian to a per-layer number

The full Hessian is enormous — it is $N \times N$ for $N$ parameters, so for even a small model it has trillions of entries and you will never form it. HAWQ's contribution is to make this tractable. Two key reductions:

First, treat the Hessian as **block-diagonal across layers** — ignore cross-layer curvature terms and consider each layer's own block $H_L$. This is an approximation, but a good one for ranking, and it gives you a per-layer quadratic form $\Delta \mathcal{L}_L \approx \tfrac{1}{2} \Delta W_L^\top H_L \Delta W_L$.

Second, bound that quadratic form by the **top eigenvalue** of the block. For any symmetric $H_L$ and any vector $\Delta W_L$,

$$
\tfrac{1}{2}\,\Delta W_L^\top H_L\, \Delta W_L \;\le\; \tfrac{1}{2}\,\lambda_{\max}(H_L)\, \lVert \Delta W_L \rVert^2 ,
$$

where $\lambda_{\max}(H_L)$ is the largest eigenvalue of layer $L$'s Hessian block. So the loss increase is bounded by the layer's top curvature times the squared norm of its quantization error. The first factor, $\lambda_{\max}(H_L)$, is a pure property of the layer's place on the loss surface — it is the **sensitivity score** HAWQ uses. The second factor, $\lVert \Delta W_L \rVert^2$, is the quantization error magnitude, which you control by choosing the bit-width. Figure 2 is this idea in picture form: the same rounding step in a flat basin versus a sharp basin, and the very different loss penalty each produces.

![A before-after figure contrasting a flat low-curvature loss basin where a quantization step causes a tiny loss rise against a sharp high-curvature basin where the same step causes a large loss rise](/imgs/blogs/mixed-precision-and-sensitivity-analysis-2.png)

There is a subtlety worth stating because it bit me once. $\lambda_{\max}$ alone ranks layers by *worst-case* sensitivity to an adversarially-aligned error. The actual quantization error is not adversarial — it is roughly random rounding noise — so a better expected-case score is the **Hessian trace** $\mathrm{Tr}(H_L)$ (the sum of all eigenvalues, i.e. the average curvature over all directions), which HAWQ-V2 (Dong et al., 2020) adopts precisely because random $\Delta W$ samples curvature in all directions, not just the worst one. In practice both give similar layer rankings; the trace is a little more stable. We will use the trace estimator in code.

### Computing curvature without forming the Hessian

You cannot form $H_L$, but you can get $\lambda_{\max}(H_L)$ and $\mathrm{Tr}(H_L)$ cheaply with two classic tricks, both of which only need *Hessian-vector products*, and a Hessian-vector product $Hv$ is computable by automatic differentiation (it is the gradient of $g^\top v$) without ever materializing $H$.

- **Top eigenvalue** via **power iteration**: repeatedly compute $v \leftarrow Hv / \lVert Hv \rVert$, and $\lambda_{\max} \approx v^\top H v$ converges in a handful of iterations.
- **Trace** via **Hutchinson's estimator**: $\mathrm{Tr}(H) = \mathbb{E}_z[z^\top H z]$ for a random Rademacher vector $z$ (each entry $\pm 1$ with equal probability), so average $z^\top (Hz)$ over a few random $z$.

Both need only Hessian-vector products and converge in maybe 10 to 50 iterations, which is a few hundred backward passes total — minutes on a GPU even for a large model. This is the practical magic that made Hessian-based sensitivity usable, and we will write the code in section 4.

### A second, cheaper proxy: signal-to-quantization-noise ratio

Curvature is the most principled signal, but it needs gradients and labels and a backward graph. There is a much cheaper proxy that needs *neither labels nor backprop*: the per-layer **signal-to-quantization-noise ratio (SQNR)**. The idea: quantize one layer's weights, dequantize them back to float, and measure how much the layer's *output* changed. A layer whose output barely moves when you quantize it is tolerant; a layer whose output is mangled is sensitive.

Formally, for a layer with float output $y$ and quantized-then-dequantized output $\hat{y}$,

$$
\text{SQNR}_L \;=\; 10 \log_{10} \frac{\lVert y \rVert^2}{\lVert y - \hat{y} \rVert^2} \quad [\text{dB}].
$$

High SQNR (say above 30 dB) means the quantization noise is tiny relative to the signal — the layer is tolerant. Low SQNR (below 20 dB) means the noise is a large fraction of the signal — the layer is sensitive. SQNR is the engineer's quick-and-dirty sensitivity meter: one forward pass over a handful of calibration batches, no labels, no gradients, no Hessian. It is less faithful than curvature (it measures local output distortion, not its propagation to the final loss), but it correlates well in practice and it is what I reach for first. It connects directly to the bit-width law we will derive next: each added bit buys you about 6 dB of SQNR.

### Why the first and last layers are almost always the guilty ones

The Taylor argument tells you *that* curvature predicts sensitivity, but it does not by itself tell you *where* the high-curvature layers will sit. Empirically they sit at the two ends of the network, and there is a clean reason for each end that is worth internalizing because it lets you predict the profile before you measure it.

The **first layer** is sensitive because of *fan-out*. Every feature the network ever computes is, transitively, a function of the first layer's output. An error introduced in the stem does not stay local — it propagates through every downstream layer, getting transformed but never averaged away, because there is no parallel path that did not pass through the stem. In the Hessian picture, the stem's weights influence the loss through an enormous number of downstream paths, and curvature accumulates along all of them. There is also an information-theoretic angle: the stem operates on the raw input, which has the highest information density in the whole forward pass (it has not yet been compressed into abstract features), and coarsely quantizing the operator that reads the richest signal throws away the most.

The **last layer** is sensitive because of *no downstream averaging*. The classifier produces the logits directly; its output is, up to a softmax, the loss. There is no deep stack of subsequent layers to smear out or partially cancel its errors — whatever distortion you put into the final projection lands more or less unfiltered on the loss. Deep middle layers get the opposite treatment: their errors pass through many subsequent nonlinearities and, critically, through *residual connections* that provide a clean bypass path, so a perturbed middle layer's damage is diluted by the untouched skip connection. The network has slack in the middle precisely because it is over-parameterized there and redundant features cover for each other.

This is why the rule "keep the stem and the classifier higher, let the middle go low" is so robust across architectures — it is not a coincidence of one model, it is a consequence of where information density is highest and where downstream averaging is weakest. When you see a CNN or a transformer profile that *violates* this (a tolerant first layer, a sensitive middle layer), treat it as a signal that something unusual is going on — often an activation outlier problem, which is the next subtlety.

### Weights are not the whole story: activation sensitivity

Everything so far quantized *weights*. But on most accelerators you also quantize *activations* (the intermediate tensors flowing between layers), and activation sensitivity follows different rules — it is driven by **outliers**, not curvature. A layer whose activations have a few extreme values forces a large quantization range, which makes every *ordinary* value coarse, and that wrecks the layer. This is the dominant failure mode in transformer quantization: a handful of channels in the attention and feed-forward activations carry values 10 to 100 times larger than the rest, and naively quantizing the activation tensor to int8 wastes almost all of the dynamic range on those outliers.

The practical consequence for mixed precision: a layer can be perfectly tolerant on the *weight* axis and intolerant on the *activation* axis (or vice versa), so you sometimes want to mix precision *within* a layer — int4 or int8 weights but fp16 activations for the outlier-heavy layers. When you run the SQNR estimator, run it on activations too (quantize the activation tensor, measure the output change), and treat the weight-SQNR and activation-SQNR as two separate sensitivity columns. A layer that is fine on weights but terrible on activations is telling you the bottleneck is dynamic range, and the fix is either keeping its activations in higher precision or a technique like SmoothQuant that migrates the outlier scale from activations into weights where it is easier to handle.

## 3. Bit allocation as constrained optimization

Once you have a sensitivity score per layer, the question becomes: how do I spend my bit budget? This is where the folklore ("keep the important layers in higher precision") becomes a real optimization problem with a real answer.

### The integer program

Let layer $L$ get bit-width $b_L$ chosen from a small menu, say $b_L \in \{4, 8, 16\}$ (the precisions your hardware actually supports — more on that constraint in section 6). Two quantities depend on the choice:

- **Cost**: the layer's size is $\text{params}_L \times b_L$ bits, and its latency contribution scales similarly on a bandwidth-bound target. Lower $b_L$ is cheaper.
- **Damage**: the accuracy loss the layer contributes. Using the second-order theory, the per-layer loss increase at bit-width $b_L$ is approximately $\Omega_L(b_L) \approx \tfrac{1}{2}\, \mathrm{Tr}(H_L)\, \lVert \Delta W_L(b_L) \rVert^2$, and $\lVert \Delta W_L(b_L) \rVert^2$ grows as you drop bits (quadrupling roughly each time you halve the number of levels, i.e. drop one... well, each bit you remove roughly quadruples the error variance — see the SQNR law).

We want to minimize total damage subject to a size budget:

$$
\min_{\{b_L\}} \sum_L \Omega_L(b_L) \quad \text{subject to} \quad \sum_L \text{params}_L \cdot b_L \;\le\; B,
$$

where $B$ is the bit budget implied by your target megabytes. This is an integer program — each $b_L$ is a discrete choice from a small set — and it is the clean mathematical statement of mixed-precision allocation. Figure 3 shows the structure: sensitivity scores and per-layer costs and a hard budget all feed a solver that emits one bit-width per layer.

![A graph figure showing per-layer sensitivity scores, per-layer cost, and a size budget all feeding into a solver that outputs a per-layer bit assignment which is checked for feasibility](/imgs/blogs/mixed-precision-and-sensitivity-analysis-3.png)

If the damage terms were independent and additive — which the block-diagonal Hessian approximation says they roughly are — this is a separable integer program, and it has a beautiful property: **the greedy solution is essentially optimal.** That is the next idea.

### The greedy / Pareto-frontier approach

You do not need an ILP solver for this in practice. The structure makes a greedy algorithm work. Think of it as a sequence of moves. Start with every layer at the highest precision (fp16) — the most accurate, biggest model. Now repeatedly ask: *of all the layers I could demote one step (fp16 to int8, or int8 to int4), which demotion gives me the most size saved per unit of accuracy lost?* Take that move. Repeat until you hit the budget.

The decision quantity for each candidate demotion is the ratio

$$
\text{value}_L \;=\; \frac{\text{bits saved by demoting } L}{\text{extra loss from demoting } L} \;=\; \frac{\Delta \text{size}_L}{\Delta \Omega_L},
$$

and greedily picking the highest-value move at each step traces out the **Pareto frontier** of accuracy versus size: every model on that frontier is one where you cannot save more size without losing more accuracy than some other model already on the frontier. The greedy order is exactly the order in which you would peel off bits to walk down that frontier. You stop at whatever point your budget dictates — and because you have the whole frontier, you can also answer the inverse question ("what is the smallest model that keeps accuracy within 0.5 points?") for free.

Why greedy works here and is not just a hack: when the objective is separable (each layer's damage depends only on its own bit-width) and the cost is linear, the problem is a multiple-choice knapsack, and the continuous relaxation of multiple-choice knapsack is solved exactly by this greedy "best marginal ratio first" rule. The integer solution it produces is provably within one item of optimal. For our purposes — where the sensitivity scores are themselves estimates with noise — greedy is not the approximation that matters. The estimates are.

### A note on the bit-width error law

I keep saying "error grows as you drop bits" — let me make it exact, because it sets the exchange rate the whole optimization runs on. For uniform quantization of a value over a range of width $R$ into $2^b$ levels, the step size is $\Delta = R / 2^b$. Rounding to the nearest level produces an error uniformly distributed on $[-\Delta/2, \Delta/2]$, which has variance

$$
\sigma_q^2 = \frac{\Delta^2}{12} = \frac{R^2}{12 \cdot 2^{2b}} .
$$

The error *variance* scales as $2^{-2b}$ — drop one bit and the variance quadruples. Turning this into a ratio of signal power to noise power and taking $10\log_{10}$ gives the famous law every signal-processing textbook has:

$$
\text{SQNR}(b) \approx 6.02\, b + 1.76 \ \text{[dB]} .
$$

**Each bit buys about 6 dB of SQNR.** This is the precise statement of "fewer bits, more error," and it is why the damage term $\Omega_L(b_L)$ in the integer program grows so steeply as you cut bits — and why a layer that is fine at int8 can fall off a cliff at int4. We use this same 6-dB-per-bit relationship when we read SQNR profiles in code.

## 4. Measuring sensitivity in code

Theory is worth nothing if you cannot run it. Here are the two estimators end to end in PyTorch, on the running ResNet example. Start with the cheap one.

### SQNR per layer (no labels, one pass)

The plan: for each layer, simulate int4 weight quantization (quantize then dequantize the weights), run a few calibration batches, and compare the layer's output before and after. We hook the layer outputs so we do not have to surgically rewire the model.

```python
import torch
import torch.nn as nn

@torch.no_grad()
def fake_quantize_per_channel(w: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Symmetric per-output-channel weight quantization, then dequantize back to float."""
    qmax = 2 ** (num_bits - 1) - 1            # int4 -> 7
    # per output channel (dim 0) scale from the max abs weight
    dims = tuple(range(1, w.dim()))
    scale = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q * scale                           # dequantized float, carries the rounding error

def measure_sqnr(model: nn.Module, loader, num_bits: int = 4, n_batches: int = 8):
    """For each Conv2d / Linear layer, report output SQNR if ONLY that layer is quantized."""
    layers = {name: m for name, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))}
    sqnr = {}

    for name, layer in layers.items():
        clean = []          # float outputs
        noisy = []          # outputs with this layer's weights quantized

        # capture the float output
        def grab(store):
            def hook(_m, _inp, out): store.append(out.detach())
            return hook

        h = layer.register_forward_hook(grab(clean))
        run_n_batches(model, loader, n_batches)
        h.remove()

        # swap in quantized weights, capture again, restore
        w_orig = layer.weight.data.clone()
        layer.weight.data = fake_quantize_per_channel(w_orig, num_bits)
        h = layer.register_forward_hook(grab(noisy))
        run_n_batches(model, loader, n_batches)
        h.remove()
        layer.weight.data = w_orig

        y = torch.cat([t.flatten() for t in clean])
        yq = torch.cat([t.flatten() for t in noisy])
        signal = y.pow(2).sum()
        noise = (y - yq).pow(2).sum().clamp(min=1e-12)
        sqnr[name] = float(10 * torch.log10(signal / noise))

    return dict(sorted(sqnr.items(), key=lambda kv: kv[1]))   # ascending = most sensitive first

@torch.no_grad()
def run_n_batches(model, loader, n):
    model.eval()
    for i, (x, _) in enumerate(loader):
        if i >= n: break
        model(x)
```

Run that and the lowest-SQNR entries are your sensitive layers. On the ResNet, `conv1` (the stem) and `fc` (the classifier) come out at the bottom — often 15 to 22 dB at int4, meaning the quantization noise is a serious fraction of the signal — while the middle bottleneck blocks sit at 30 to 40 dB, comfortably tolerant. That is Figure 1's profile, produced in a couple of minutes with no labels and no gradients.

### Loss-perturbation sensitivity (one layer at a time)

SQNR measures local distortion. The most *direct* sensitivity measure is the one I used on that first phone model: actually quantize one layer, measure the real validation metric, restore, repeat. It is more expensive (you need labels and a full eval pass per layer) but it answers the exact question — how much does the end-to-end loss rise?

```python
import copy

@torch.no_grad()
def loss_perturbation_sensitivity(model, val_loader, eval_fn, num_bits=4):
    """Quantize ONE layer at a time, measure the validation-loss increase it causes."""
    base = eval_fn(model, val_loader)        # e.g. returns (loss, top1)
    base_loss = base["loss"]
    deltas = {}

    for name, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        w_orig = layer.weight.data.clone()
        layer.weight.data = fake_quantize_per_channel(w_orig, num_bits)
        perturbed = eval_fn(model, val_loader)
        layer.weight.data = w_orig           # restore before next layer
        deltas[name] = perturbed["loss"] - base_loss      # loss increase from THIS layer

    return dict(sorted(deltas.items(), key=lambda kv: kv[1], reverse=True))  # biggest damage first
```

The two methods should largely agree on the *ranking*; if they disagree wildly on a layer, that layer is worth a closer look (often it is a layer whose output distortion does not propagate, or one whose distortion is amplified downstream by a residual connection). Use SQNR to triage cheaply, then confirm the top few with loss perturbation. Figure 5 lays out the estimators side by side so you can pick the right one for your scale.

![A matrix figure comparing four sensitivity estimators SQNR, loss perturbation, Hessian trace, and Hessian top eigenvalue across whether they need labels, their compute cost, and their fidelity](/imgs/blogs/mixed-precision-and-sensitivity-analysis-5.png)

### Hessian trace via Hutchinson (the principled one)

For completeness, here is the curvature estimator from section 2 — the trace of each layer's Hessian block via Hessian-vector products and Hutchinson's trick. This is what HAWQ-V2 uses, and for large models or where SQNR and perturbation disagree, it is the most trustworthy ranking.

```python
import torch

def hessian_trace_per_layer(model, loss_fn, x, y, n_samples: int = 16):
    """Estimate Tr(H_L) for each parameter tensor via Hutchinson + Hessian-vector products."""
    model.eval()
    params = [(n, p) for n, p in model.named_parameters()
              if p.requires_grad and p.dim() > 1]      # weight matrices only
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, [p for _, p in params], create_graph=True)

    traces = {n: 0.0 for n, _ in params}
    for _ in range(n_samples):
        # Rademacher probe vectors, one per param tensor
        zs = [torch.randint_like(p, high=2).float().mul_(2).sub_(1) for _, p in params]
        # Hessian-vector product Hz = grad( g . z )  (one backward over the graph)
        gz = sum((g * z).sum() for g, z in zip(grads, zs))
        Hzs = torch.autograd.grad(gz, [p for _, p in params], retain_graph=True)
        for (name, _), Hz, z in zip(params, Hzs, zs):
            traces[name] += float((Hz * z).sum()) / n_samples   # z^T H z, averaged

    return dict(sorted(traces.items(), key=lambda kv: kv[1], reverse=True))
```

Three honest caveats from having run this in anger. First, `create_graph=True` makes the first backward expensive in memory; do it on a single calibration batch, not the whole set. Second, the trace can be slightly negative for a not-perfectly-converged model (the $g \approx 0$ assumption is approximate); that is fine for ranking, just take magnitudes. Third, normalize by parameter count if you want a *per-weight* sensitivity rather than a per-layer total — a big layer can have a large trace simply because it has more weights, not because each weight is more sensitive.

## 5. The greedy bit-assignment loop

Now we put the score to work. Given a per-layer sensitivity score and a per-layer parameter count, here is the greedy allocator that walks the Pareto frontier down to your size budget. Figure 6 is the loop as a timeline — start at fp16, repeatedly demote the best-value layer, stop at the budget.

![A timeline figure showing the greedy bit-assignment loop starting with all layers at fp16, scoring layers, demoting the best candidate step by step, checking the budget, and finishing at a final plan](/imgs/blogs/mixed-precision-and-sensitivity-analysis-6.png)

```python
def greedy_bit_allocation(layer_params, sensitivity, budget_bits,
                          bit_menu=(16, 8, 4)):
    """
    layer_params: {name: num_params}
    sensitivity:  {name: score}   higher = more sensitive (e.g. Hessian trace or 1/SQNR)
    budget_bits:  total weight-storage budget in BITS
    Returns: {name: bits} minimizing summed damage under the budget.
    """
    # start everyone at the highest precision
    bits = {name: bit_menu[0] for name in layer_params}

    def total_bits(cfg):
        return sum(layer_params[n] * b for n, b in cfg.items())

    # damage from putting layer n at bit-width b: sensitivity * error-variance(b)
    # error variance ~ 2^(-2b); scale by sensitivity score
    def damage(name, b):
        return sensitivity[name] * (2.0 ** (-2 * b))

    while total_bits(bits) > budget_bits:
        best_move, best_value = None, -1.0
        for name in layer_params:
            cur = bits[name]
            lower = next((b for b in bit_menu if b < cur), None)
            if lower is None:
                continue                      # already at the floor
            bits_saved = layer_params[name] * (cur - lower)
            extra_damage = damage(name, lower) - damage(name, cur)
            extra_damage = max(extra_damage, 1e-12)
            value = bits_saved / extra_damage  # size saved per unit of damage
            if value > best_value:
                best_value, best_move = value, (name, lower)
        if best_move is None:
            break                              # every layer at the floor, cannot shrink more
        name, lower = best_move
        bits[name] = lower

    return bits
```

A few production notes. The `value` ratio is exactly the marginal-utility quantity from section 3; demoting the highest-value layer first is what traces the Pareto frontier. And crucially: **always re-measure end-to-end accuracy on the final plan.** The greedy loop uses an additive damage proxy; the real model has interactions (residual connections, batchnorm folding) that the proxy ignores, so the plan is a strong starting point, not the final word. Run one real eval; if you are over your accuracy budget, bump the worst offender up one precision step and re-eval. Two or three iterations of that converges.

### Size budget versus latency budget

The loop above optimizes for *size* (bits stored). But the budget you actually care about is often *latency*, and the two objectives produce different allocations. Size is simple — every layer's contribution is `params × bits`, full stop. Latency is messier: demoting a layer only speeds it up if that layer is memory-bound (fewer bits to fetch) and if the hardware has a faster kernel at the lower precision. A compute-bound layer can be demoted to int4 and run at *exactly the same speed*, because the bottleneck was multiply-accumulate throughput, not data movement. To optimize latency you swap `bits_saved` in the loop for a per-layer *latency* saving measured (not guessed) on the target — you profile each layer at each candidate precision once, build a lookup table of `latency[layer][bits]`, and let the greedy loop spend its accuracy budget on the demotions that actually move the wall clock. This is exactly what HAWQ-V3 and HAQ do, and it is why HAQ put a hardware simulator in its reward loop. The two objectives can disagree sharply: the layer with the most *parameters* (best for a size budget) is often a late, compute-bound conv (useless for a latency budget), so never assume a size-optimal plan is latency-optimal — pick the objective that matches your actual constraint.

Here is how the common allocation strategies stack up, so you can pick the one that fits your situation:

| Strategy | What it optimizes | Cost to run | When to use |
| --- | --- | --- | --- |
| Manual rule (stem/head high, rest int8) | nothing — heuristic | minutes | quick first cut; a few guilty layers |
| Greedy on SQNR | size, label-free proxy | ~minutes | no labels, fast triage |
| Greedy on Hessian trace | size, principled proxy | ~tens of minutes | need a trustworthy ranking |
| Latency-aware greedy | measured device latency | profile + minutes | latency budget, permissive HW |
| ILP (HAWQ-V3) | exact under linear constraints | solver + Hessians | the last few percent matter |
| RL search (HAQ) | hardware reward directly | hours of search | research-grade, big payoff justified |

For most edge work the second or third row is the sweet spot: a greedy allocator on a cheap-but-decent sensitivity score, confirmed by one real eval. The ILP and RL rows buy you the last percent and are worth it only when the budget is brutal and the model ships at huge scale.

### Advanced PTQ that buys you lower bits

The greedy loop decides *which* bit-width each layer gets. A separate question is *how well* you quantize at a given bit-width, and this is where the modern post-training-quantization literature earns its keep — because better PTQ shifts every layer's sensitivity curve, making more layers tolerant of int4 and shrinking your budget problem.

- **AdaRound** (Nagel et al., 2020) makes a per-weight decision to round *up or down* rather than always to nearest, optimizing those decisions to minimize the layer's output error. It turns the naive rounding in our `fake_quantize` into a learned, output-aware rounding, and it is the single biggest lever for getting int4 weights to behave. Typically recovers 1 to 3 points at int4 with no labels.
- **BRECQ** (Li et al., 2021) extends this to *block-wise* reconstruction: instead of optimizing one layer's output, it reconstructs a whole residual block's output, capturing the cross-layer interactions the block-diagonal approximation throws away. It is the current go-to for aggressive int4 PTQ on CNNs.
- **HAWQ-V3** (Yao et al., 2021) closes the loop end to end: it computes Hessian sensitivities, formulates the bit allocation as an integer linear program with *hardware-aware* latency constraints (not just size), and emits an integer-only deployable model. It is the fully-realized version of everything in this post.

The practical recipe most teams actually ship: use SQNR or Hessian to *rank*, greedy to *allocate*, and AdaRound or BRECQ to *fit* the low-bit layers so they hold up. The three are complementary — ranking tells you where to spend, allocation spends it, and reconstruction makes each int4 layer hurt less.

It is worth being precise about why reconstruction interacts with the allocation problem, because it changes the answer rather than just polishing it. Recall that the greedy loop's damage term for a layer is `sensitivity × error-variance(bits)`. AdaRound and BRECQ attack the *error-variance* factor directly: by choosing rounding directions that minimize the layer's output error instead of rounding to nearest, they shrink the effective $\lVert \Delta W_L \rVert^2$ at a fixed bit-width, sometimes by a factor of two or more. That means a layer the naive analysis flagged as "too sensitive for int4" may become perfectly tolerant *once you apply learned rounding*. So the correct order of operations is to run reconstruction first (or to estimate sensitivity *with* reconstruction applied), then allocate — otherwise you are pricing layers at their naive-rounding sensitivity and over-paying in precision for layers that BRECQ could have rescued. In practice this can move two or three borderline layers from the int8 column into the int4 column, which on a tight budget is the difference between hitting your target and missing it.

## 6. How hardware constrains the choice

Here is the reality that humbles the elegant theory: your accelerator decides which mixed-precision plans are even executable, and it is usually far more restrictive than "any bit-width per layer." This is the single most common reason a beautiful sensitivity-optimal plan never ships. Figure 7 maps the legal precision menu across common targets.

![A matrix figure showing which of int4, int8, and fp16 are natively supported, emulated, or unsupported across a mobile NPU, a Jetson GPU, a Cortex-M microcontroller, and a desktop GPU with TensorRT](/imgs/blogs/mixed-precision-and-sensitivity-analysis-7.png)

Walk the columns:

- **Mobile NPUs** (the Pixel Tensor NPU, Qualcomm Hexagon, Apple Neural Engine) are typically **int8-native and that is it**. Many do support fp16 as a fallback path, but it often runs on a different, slower engine (or the GPU), and int4 frequently has *no hardware path at all* — the runtime will silently emulate it by unpacking to int8, giving you the *size* saving but none of the *speed* saving, and sometimes a *slowdown* from the unpack overhead. On these targets, "mixed precision" realistically means **int8 everywhere plus a few layers kept in fp16** — and even that fp16 escape hatch can cost you a delegate switch (the runtime hands those layers to the GPU or CPU, and the round trip across the accelerator boundary can eat your savings).
- **Jetson / desktop NVIDIA GPUs** are the permissive end. TensorRT supports int8 and fp16 Tensor Cores natively, and newer architectures add int4 Tensor Core paths. Here genuine per-layer mixed precision is real: you can mark individual layers' precision in the builder and TensorRT will honor it (and even auto-tune the mix if you let it).
- **Microcontrollers (Cortex-M with CMSIS-NN)** are int8-or-bust. The optimized kernels are int8 SIMD MAC operations; fp16 falls back to slow software floating point (often no FPU at all on the smallest parts), and int4 is not supported by the standard kernels. On an MCU, mixed precision is essentially off the table — you ship int8 and you make int8 work (this is the MCUNet world).

The lesson is to **let the hardware define the bit menu before you optimize.** Do not solve the integer program over $\{4, 8, 16\}$ if your NPU only runs $\{8, 16\}$ — you will produce a plan you cannot deploy. Set `bit_menu` in the greedy loop to exactly the precisions your target accelerates, measure on the device, and ignore the precisions the runtime can only emulate. A plan that is sensitivity-optimal but hardware-illegal is worth nothing; a plan that is slightly suboptimal but runs natively is worth everything.

Two more device realities. First, fp16-on-int8-NPU often is not a free fallback — the layer can be evicted to a slower engine, so keeping a layer in fp16 to save accuracy might *cost* latency. Always measure the latency of your mixed plan, not just its size. Second, mixing precisions creates dtype conversion boundaries inside the graph (int4 output feeding an fp16 layer needs a requantize), and those conversions cost real cycles; a good compiler fuses them, a bad one does not. This is exactly the kind of thing the roofline and on-device measurement discipline catches.

### Setting precision per layer in TensorRT

When you *do* have permissive hardware, here is what marking a layer's precision actually looks like in the TensorRT builder — concrete, not pseudocode.

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()

# allow the engine to use both int8 and fp16 kernels
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)
# we will pin some layers ourselves, so stop TensorRT from overriding us
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

# ... parse your ONNX into `network` here ...

SENSITIVE = {"stem_conv", "classifier_gemm"}     # from our sensitivity profile
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if layer.name in SENSITIVE:
        layer.precision = trt.float16            # keep these in fp16
    else:
        layer.precision = trt.int8               # everything else int8

engine = builder.build_serialized_network(network, config)
```

`OBEY_PRECISION_CONSTRAINTS` is the flag that matters: without it, TensorRT treats your `layer.precision` as a hint and may ignore it if its kernel auto-tuner thinks another precision is faster. With it, your sensitivity-driven decisions are honored. On a phone NPU you would instead express this through the runtime's per-op precision annotations (or simply by which ops you leave in float in the converter), but the principle is identical: name the sensitive layers, pin them up, let the rest go to int8.

## 7. Stress-testing the decision

A plan that looks good on a clean validation set can fall apart under the conditions edge deployments actually meet. Before you ship a mixed-precision config, push on it deliberately. Here is the engineering reasoning for the four failure modes I check every time, because each one has bitten me or someone on my team.

**What happens at int4 — does the proxy still hold?** The greedy loop's damage term uses the second-order Taylor expansion, and that expansion assumes the perturbation is *small*. At int8 the rounding error is genuinely small and the quadratic approximation is excellent. At int4, the error is sixteen times coarser, and the second-order term is no longer the whole story — third- and higher-order terms start to matter, and the proxy systematically *under*-predicts the damage. The symptom: your greedy plan promises −0.4 points and the real eval comes back −1.2. The fix is not to abandon the method but to trust the *ranking* (which stays correct — sensitive layers stay sensitive) more than the *absolute predicted loss*, and to always reconcile against a real eval. When you are pushing layers to int4, also reach for AdaRound or BRECQ on those specific layers; learned rounding shrinks the actual error enough that the quadratic approximation becomes valid again, so the proxy and reality re-converge.

**When the calibration set is tiny.** Sensitivity scores, SQNR, and Hessian estimates all depend on the data you feed them. On the running ResNet, eight calibration batches (≈256 images) give a stable ranking, but I have seen a 32-image calibration set flip the verdict on a borderline layer purely from sampling noise — and worse, if the calibration set is not representative (all daytime images for a model that runs at night), the sensitivity profile is *confidently wrong*. The stress test: re-run the sensitivity estimation on two disjoint calibration subsets and confirm the layer ranking is stable. If the top few sensitive layers differ between subsets, you do not have enough or diverse enough calibration data, and any bit allocation built on that ranking is a guess. Spend the effort on a representative calibration set before you spend it on a clever allocator — garbage in, garbage out applies with full force here.

**When the NPU does not support an op and it falls back to CPU.** This is the cruelest one because it is invisible until you profile on-device. You pin a layer to fp16 on a mostly-int8 NPU to protect its accuracy, the converter accepts it, the model runs and is *correct* — and it is also three times slower than the uniform int8 version, because that one fp16 layer is not supported on the NPU's int8 engine, so the runtime evicts it to the CPU. Now every inference pays a round trip: int8 NPU → copy activations to CPU → fp16 on CPU → copy back to NPU. The data movement across the accelerator boundary dwarfs the compute you saved. The only defense is on-device latency measurement of the *actual mixed plan* (not the uniform baseline, not a desktop simulation), per the [metrics-that-matter discipline](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device). If a fp16 escape hatch triggers a CPU fallback, you either keep that layer at int8 and eat the accuracy (accepting a worse Pareto point) or you find a different layer to protect that *does* have a fast higher-precision path.

**When the model is memory-bound, not compute-bound.** Mixed precision saves bits, and on a memory-bound layer fewer bits means less data to move, which means real speed — but on a *compute-bound* layer, the bottleneck is the multiply-accumulate throughput, and the precision of the operands may not change that throughput much (an int8 MAC and an int4 MAC can take the same number of cycles on hardware whose int4 path just unpacks to int8). So the latency benefit of demoting a layer depends entirely on whether that layer is memory-bound or compute-bound, which is a roofline question. The trap: you demote a compute-bound layer to int4 expecting a speedup, lose accuracy, and gain *nothing* in latency because the layer was never bandwidth-limited. The discipline is to read each layer's arithmetic intensity off the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) and only spend accuracy on precision cuts that actually move latency — which, for the size-budget objective, you would do anyway, but for a *latency* objective it is the whole game.

The thread through all four: the sensitivity-and-allocation math gives you a strong *hypothesis*, and on-device measurement turns it into a *decision*. Never let the proxy be the last word.

## 8. Worked examples

Numbers make it real. Two scenarios, both on the running ResNet-50 (about 25.6M parameters, ~25 MB at int8 weights, baseline fp32 top-1 of 76.1% on ImageNet for this checkpoint). All accuracy figures below are illustrative of the *pattern* I have repeatedly measured; treat the exact decimals as representative rather than from a single canonical run.

#### Worked example: a ResNet sensitivity profile and the mixed config it implies

I run the SQNR estimator from section 4 at int4 over 8 calibration batches (no labels, about ninety seconds on one GPU). The profile sorts cleanly into three tiers:

| Layer group | int4 SQNR | int4 loss delta | Tier |
| --- | --- | --- | --- |
| `conv1` (stem, 3→64) | 17 dB | +3.8 pt | sensitive — keep fp16 |
| `layer1` blocks | 34 dB | +0.4 pt | tolerant — int4 |
| `layer2` blocks | 36 dB | +0.2 pt | tolerant — int4 |
| `layer3` blocks | 33 dB | +0.3 pt | tolerant — int4 |
| `layer4` blocks | 26 dB | +1.6 pt | borderline — int8 |
| `fc` (classifier, 2048→1000) | 16 dB | +4.1 pt | sensitive — keep fp16 |

The shape is the universal one: the **first layer** (it processes the raw, high-information input — every downstream feature is built on it, so error here propagates everywhere) and the **last layer** (it produces the logits directly, so its error lands straight on the loss with no downstream averaging to dampen it) are the most sensitive, the **middle is forgiving** (deep middle features are redundant and the network has slack), and the **late-but-not-last** layers are borderline. The two-line rule I now apply without even measuring on a new CNN: *keep the stem and the classifier in higher precision, and be suspicious of the last conv stage.*

Now the configs, all sized on weight storage:

| Config | Stem / classifier | Middle | Late stage | Size (MB) | Top-1 |
| --- | --- | --- | --- | --- | --- |
| fp32 baseline | fp32 | fp32 | fp32 | 102.0 | 76.1% |
| Uniform int8 | int8 | int8 | int8 | 25.5 | 75.7% |
| Uniform int4 | int4 | int4 | int4 | 12.8 | 70.3% |
| **Mixed (4/8/16)** | **fp16** | **int4** | **int8** | **12.6** | **75.5%** |

Read the bottom two rows together, because that is the whole point. Uniform int4 and the mixed plan are *the same size* (12.6 vs 12.8 MB) — but uniform int4 loses 5.8 points while the mixed plan loses 0.6. We took the bits we saved by crushing the tolerant middle to int4 and *spent them* on keeping the stem and classifier in fp16. Same budget, almost five points of accuracy recovered. Figure 4 is exactly this before-after. That is mixed precision earning its complexity.

![A before-after figure comparing uniform int8 against a mixed 4-8-16 plan at the same model size, showing the mixed plan recovers accuracy by keeping the end layers in fp16 and the middle in int4](/imgs/blogs/mixed-precision-and-sensitivity-analysis-4.png)

#### Worked example: hitting a hard 6 MB budget with minimal loss

Now the inverse problem, the one that actually shows up in product work: *the app store cares about download size and the team has decreed the model must be under 6 MB.* I cannot just pick a uniform precision — int8 is 25.5 MB (way over), int4 is 12.8 MB (still over), and below int4 there is no integer precision left. I have to allocate.

I feed the greedy loop the per-layer parameter counts and the Hessian-trace sensitivities, with the bit menu the hardware allows. Suppose this is a permissive target (a Jetson) so the menu is $\{4, 8, 16\}$, and a budget of $6 \text{ MB} = 6 \times 8 \times 10^6 = 4.8 \times 10^7$ bits. The loop starts at fp16 (51 MB) and demotes:

| Step | Layer demoted | New size | Cumulative top-1 delta |
| --- | --- | --- | --- |
| start | — (all fp16) | 51.2 MB | 0.0 pt |
| 1–18 | middle blocks fp16→int4 | 9.1 MB | −0.3 pt |
| 19–24 | late `layer4` fp16→int8 | 7.0 MB | −0.4 pt |
| 25 | `layer4` int8→int4 (lowest-value left) | 6.2 MB | −0.9 pt |
| 26 | one more middle int8→int4 | 5.9 MB | −1.0 pt |

The loop stops at 5.9 MB — under budget — at a predicted −1.0 point. I then run the one mandatory real eval: the actual measured drop is −1.2 points (the proxy was slightly optimistic, as proxies are), which is inside the team's −1.5 tolerance, so I ship it. Had it been over, the fix is mechanical: the last demotion (step 26) had the worst value ratio, so I undo it, costing 0.3 MB and buying back ~0.3 points — and I am at 6.2 MB, which violates the budget, so instead I would undo step 25 and find a different middle layer to demote, or reach for BRECQ to make the int4 layers hold up better. The point is that the greedy frontier hands you the *order of operations* for these trades; you are never guessing.

Compare to what uniform quantization could do: nothing. There is no uniform precision that lands at 6 MB with 1 point of loss — int4 is the floor and it costs 5.8 points. Mixed precision is the *only* tool that hits this budget at this accuracy, which is exactly when its complexity is justified.

#### Worked example: a 7B LLM on a laptop with k-quant mixed precision

The CNN examples are the cleanest place to *see* the idea, but the place mixed precision quietly does the most work today is LLMs on laptops, and it is worth one example because the vocabulary is different even though the principle is identical. Take a 7B-parameter model you want to run on an M2 MacBook (16 GB unified memory) with `llama.cpp`. At fp16 the weights are 14 GB — it technically fits but leaves almost nothing for the KV cache and the OS, so generation is slow and the machine swaps. Uniform int4 (a flat `Q4_0`) brings it to about 3.5 GB and runs fast, but it loses measurable perplexity — the model gets noticeably worse at exactly the hard tokens you care about.

The k-quant schemes (`Q4_K_M`, `Q5_K_M`, and friends) are mixed precision under a different name. They do not quantize every weight tensor to the same width. `Q4_K_M` keeps most weights at roughly 4 bits but *promotes specific, sensitivity-identified tensors to higher precision* — notably the attention output projection and the feed-forward down-projection, which the `llama.cpp` authors found (empirically, which is sensitivity analysis by another name) carry more of the model's quality than their parameter count suggests. The result, on a typical 7B model:

| Scheme | Bits/weight (avg) | Size | Perplexity delta vs fp16 | Tokens/s (M2) |
| --- | --- | --- | --- | --- |
| fp16 | 16.0 | 14.0 GB | 0.00 | ~8 (memory-bound) |
| Q4_0 (uniform) | 4.0 | 3.6 GB | +0.30 | ~22 |
| **Q4_K_M (mixed)** | **~4.5** | **4.1 GB** | **+0.12** | **~21** |
| Q5_K_M (mixed) | ~5.5 | 4.8 GB | +0.05 | ~19 |

Read the middle two rows. `Q4_K_M` is only half a bit per weight larger than uniform `Q4_0` (4.1 vs 3.6 GB) and barely slower (LLM decoding is memory-bound, so a slightly bigger model is slightly slower — the roofline again), but it cuts the perplexity penalty by more than half, because that extra half-bit is spent *exactly* on the sensitive projections instead of smeared uniformly. The command is a one-liner:

```bash
# convert an fp16 GGUF to the mixed-precision Q4_K_M scheme
./llama-quantize ./model-f16.gguf ./model-Q4_K_M.gguf Q4_K_M

# run it; -ngl offloads layers to the GPU/ANE, batch=1 is the on-device reality
./llama-cli -m ./model-Q4_K_M.gguf -ngl 99 -p "Explain mixed precision in one sentence."
```

The lesson transfers exactly from the CNN: the per-tensor sensitivity is real, it is concentrated in a few tensors, and spending a fractional extra bit precisely on those tensors beats spending it uniformly. You did not write the sensitivity analysis here — the `llama.cpp` authors did it for you and baked it into the scheme name — but it is the same analysis, and knowing that is why you should reach for `Q4_K_M` over `Q4_0` by default.

## 9. Case studies and real numbers from the literature

The pattern above is not just my ResNet. The mixed-precision literature is built on it, and the headline results are worth knowing.

**HAWQ (Dong et al., 2019)** introduced Hessian top-eigenvalue sensitivity and used it to mixed-precision-quantize ResNet and Inception models. On ResNet-50 they reached an average of under 3 bits per weight (a genuinely aggressive mix, with sensitive layers held higher) at accuracy competitive with the fp32 baseline — a regime uniform quantization cannot touch, because uniform 3-bit collapses. The core claim they validated is exactly section 2's: rank layers by curvature, give the high-curvature layers more bits, and you escape the uniform-quantization accuracy cliff.

**HAWQ-V2 (Dong et al., 2020)** swapped the top eigenvalue for the average Hessian trace (the expected-case sensitivity argument from section 2) and added an automatic bit-selection method, removing the manual layer-by-layer tuning. It pushed the same models to better accuracy-at-a-given-average-bit-width than V1, confirming that the *trace* is the more robust sensitivity signal for random quantization noise.

**HAQ (Wang et al., 2019)** took a different route to the same problem: instead of curvature, it used **reinforcement learning** to search the per-layer bit-width policy, with a hardware simulator in the loop so the reward was actual measured latency and energy on a target accelerator, not a proxy. The lesson HAQ drove home for the field is the one in section 6 — the *hardware* must be in the loop, because the sensitivity-optimal bit allocation and the latency-optimal bit allocation are different plans, and only the device knows which precisions are fast.

**AdaRound (Nagel et al., 2020)** and **BRECQ (Li et al., 2021)** attacked the orthogonal axis — making each low-bit layer hurt less. AdaRound's learned rounding and BRECQ's block reconstruction are the reason int4 PTQ went from "loses 5+ points" to "loses under 1 point" on many CNNs without any retraining. BRECQ in particular reported int4 (and even mixed 2/4-bit) post-training results on ImageNet CNNs within about 1 point of full precision — results that, a few years earlier, would have required full quantization-aware training.

**MobileNet on phones** is the cautionary counter-example, and it is as instructive as the wins. Depthwise-separable convolutions — the thing that makes MobileNet small — are *more* quantization-sensitive than ordinary convolutions, because a depthwise conv has very few weights per output channel, so each weight carries more of the signal and rounding it hurts more. The published MobileNetV2/V3 quantization results show that the depthwise layers are exactly the ones that need protection (higher precision or per-channel scales), and that uniform int8 on a naive MobileNet can drop several points where the same recipe on a ResNet barely moves. The lesson: efficient architectures and aggressive quantization can fight each other, and the sensitivity profile is how you find the truce — you protect the depthwise layers and let the cheap pointwise (1x1) layers go low. This is the same per-layer story, just with the guilty layers in a different place, which is precisely why you *measure* the profile rather than assuming the stem-and-head rule blindly.

A unifying read across all five points: HAWQ/HAWQ-V2 answer *where to spend bits* (sensitivity), HAQ answers *how to spend them given the hardware* (latency-aware search), AdaRound/BRECQ answer *how to make each bit go further* (reconstruction), and the MobileNet case warns that the architecture decides *which* layers are guilty. A modern mixed-precision pipeline uses all of these ideas — which is precisely the recipe in sections 4 through 6: measure the profile, allocate against a hardware-legal menu, and reconstruct the low-bit layers.

On the LLM side, the same per-layer-tolerance principle drives the k-quant schemes in `llama.cpp` (the `Q4_K_M` and friends keep certain weight groups — notably the attention output and the `feed-forward` down-projection — at higher precision than the rest) and methods like AWQ that protect the small fraction of "salient" weight channels that carry most of the activation magnitude. Different vocabulary, identical insight: not all weights are equally sensitive, so do not quantize them all the same. See [sub-8-bit networks: int4, ternary, and binary](/blog/machine-learning/edge-ai/sub-8-bit-int4-ternary-and-binary-networks) for how far this goes at the extreme low-bit end.

## 10. When mixed precision is worth it (and when it is not)

Mixed precision is real engineering complexity: a sensitivity-measurement pipeline, a bit-allocation step, per-layer precision plumbing in your converter, dtype-conversion boundaries in the graph, and a verification eval. Every one of those is a place for a bug or a deploy-time surprise. So be honest about when it pays. Figure 8 is the decision distilled to a tree.

![A decision tree figure routing whether to use mixed precision based on whether uniform int8 meets the target, whether the hardware runs more than one precision, and whether the gap is in size or accuracy](/imgs/blogs/mixed-precision-and-sensitivity-analysis-8.png)

**Reach for mixed precision when:**

- **Uniform int8 misses your target and uniform int4 is too lossy.** This is the sweet spot — you are stuck between two uniform options, one too big, one too inaccurate, and the only way through is per-layer allocation. The 6 MB worked example is exactly this.
- **Your hardware natively runs more than one precision.** Jetson/desktop GPUs (TensorRT int8 + fp16, increasingly int4), where per-layer precision is a real, fast deploy path. If you have the silicon, use it.
- **The budget is genuinely tight.** When you are squeezing the last megabyte or the last few milliseconds and a uniform precision overshoots or undershoots, the per-layer flexibility is the only lever with the resolution to land exactly on target.
- **A handful of layers are the entire problem.** If your sensitivity profile shows two or three guilty layers (the common case), the minimal mixed plan — "int8 everywhere, fp16 on these three" — is cheap to implement and recovers most of the accuracy. This is the 80/20 version and it is almost always worth it.

**Do not bother when:**

- **Uniform int8 already hits your target.** If int8 is small enough, fast enough, and accurate enough, you are done. Adding mixed precision here is pure complexity for zero benefit — the cardinal sin of premature optimization. Ship the int8 model.
- **Your hardware is int8-only.** On most mobile NPUs and all microcontrollers, "mixed" degenerates to "int8 plus a couple of fp16 layers that get evicted to a slower engine," and the eviction can cost more latency than the accuracy is worth. If int8-only int8 is too lossy on an int8-only target, the right levers are **quantization-aware training** (recover accuracy *at* int8) or **distillation** (a smaller model that survives int8) or better PTQ (AdaRound) — not mixed precision. Spend your effort where the hardware can cash it.
- **The accuracy gap is large.** Mixed precision recovers fractions to a couple of points. If uniform int8 is dropping 5+ points, your problem is bigger than bit allocation — you likely have a quantization-hostile architecture (lots of unbounded activations, problematic normalization) and need QAT or an architecture change. Mixed precision polishes; it does not rescue.
- **You cannot measure on the real device.** Mixed plans have hardware-dependent latency (dtype boundaries, delegate switches) that you *cannot* predict from a desktop. If you have no on-device measurement loop, a mixed plan is a guess, and a uniform int8 plan is a safe known quantity. Get measurement first.

The meta-rule, and the one this whole series keeps returning to: **a technique is a cost, and you only pay it for a win you actually need.** Mixed precision buys you a point on the accuracy-size Pareto frontier that no uniform precision can reach. If you do not need that specific point — if a uniform option already lands inside your budget — do not pay for it.

## 11. Key takeaways

- **Layers are not equally quantizable.** Sensitivity varies by roughly an order of magnitude within one network, so a single global bit-width is always a compromise — too wasteful or too lossy.
- **Curvature is sensitivity, and it is provable.** A second-order Taylor expansion at a converged minimum gives $\Delta \mathcal{L} \approx \tfrac{1}{2}\Delta W^\top H \Delta W$; the first-order term vanishes, so the loss increase from quantization is governed by the Hessian. High curvature means high sensitivity.
- **Each bit is worth about 6 dB.** The SQNR law $\text{SQNR}(b) \approx 6.02 b + 1.76$ dB sets the exchange rate; quantization error variance scales as $2^{-2b}$, which is why a layer can be fine at int8 and fall off a cliff at int4.
- **Measure cheap first, confirm faithful second.** Per-layer SQNR needs no labels and no gradients — run it to triage. Confirm the top suspects with loss-perturbation, and use the Hessian trace (Hutchinson + Hessian-vector products) when you need the principled ranking.
- **Bit allocation is a knapsack; greedy solves it.** Minimize summed damage under a size or latency budget by repeatedly demoting the layer with the best size-saved-per-damage ratio. That order traces the Pareto frontier and is provably near-optimal for the separable objective.
- **First and last layers are the usual culprits.** The stem sees the raw input and the classifier produces the logits, so both turn small weight errors into large loss increases. Keep them higher; let the redundant middle go low.
- **Hardware sets the legal menu, period.** Optimize only over the precisions your target accelerates natively. A sensitivity-optimal but hardware-illegal plan is worthless; int8-only NPUs make "mixed" mostly mean "int8 plus a few fp16 escape hatches" — and even those can cost latency.
- **Always run one real end-to-end eval on the final plan and measure latency on the device.** The additive damage proxy ignores residual and batchnorm interactions and dtype-conversion costs; the proxy picks the plan, the real measurement ratifies it.
- **Mixed precision polishes, it does not rescue.** It buys fractions to a couple of points at a tight budget. If you are losing 5+ points or your hardware is int8-only, reach for QAT, distillation, or better PTQ (AdaRound/BRECQ) instead.

## 12. Further reading

- **Dong et al., "HAWQ: Hessian Aware Quantization of Neural Networks with Mixed-Precision" (ICCV 2019)** — the Hessian top-eigenvalue sensitivity score and the original mixed-precision allocation.
- **Dong et al., "HAWQ-V2: Hessian Aware trace-Weighted Quantization" (NeurIPS 2020)** — the Hessian-trace sensitivity and automatic bit selection.
- **Yao et al., "HAWQ-V3: Dyadic Neural Network Quantization" (ICML 2021)** — the integer-linear-program, hardware-aware, integer-only end-to-end realization.
- **Wang et al., "HAQ: Hardware-Aware Automated Quantization with Mixed Precision" (CVPR 2019)** — reinforcement-learning bit-width search with a hardware simulator in the loop.
- **Nagel et al., "Up or Down? Adaptive Rounding for Post-Training Quantization" (AdaRound, ICML 2020)** — learned rounding that makes low-bit weights hold up without retraining.
- **Li et al., "BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction" (ICLR 2021)** — block-wise reconstruction for aggressive int4 PTQ.
- Official docs: the NVIDIA TensorRT Developer Guide (per-layer precision and `OBEY_PRECISION_CONSTRAINTS`), the PyTorch `torch.ao.quantization` reference, and the `llama.cpp` k-quants documentation for the LLM analogue of per-layer precision.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) for the mechanics this post builds on, [sub-8-bit networks: int4, ternary, and binary](/blog/machine-learning/edge-ai/sub-8-bit-int4-ternary-and-binary-networks) for the extreme low-bit end, [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for the latency side of the budget, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that stitches every lever together.

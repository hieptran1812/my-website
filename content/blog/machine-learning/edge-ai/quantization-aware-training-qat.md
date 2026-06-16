---
title: "Quantization-aware training: the straight-through estimator and recovering accuracy"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "When post-training quantization eats three accuracy points you cannot afford, QAT simulates the rounding during training so the network learns weights that survive int8 — here is the math, the PyTorch flow, and the before-after numbers."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "quantization-aware-training",
    "straight-through-estimator",
    "lsq",
    "inference",
    "efficient-ml",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/quantization-aware-training-qat-1.png"
---

You quantized your model to int8 last Thursday. Post-training quantization was a dream: a calibration script, a couple hundred unlabeled images, ninety seconds of runtime, and out came a model that was four times smaller and, on the Pixel's NPU, almost three times faster. Then you ran the validation set. Top-1 accuracy fell from 76.1% to 73.0%. Three points. Your product spec says the on-device model must stay within one point of the cloud model, because below that the user-visible error rate roughly doubles and the support tickets start. You spent a day chasing it: per-channel weight quantization, a better-clipped activation range, MSE-based calibration instead of min-max. Each trick clawed back a few tenths. You are still a point and a half short, and you are out of free tricks. Post-training quantization (PTQ) has a ceiling, and you have hit it.

This is the moment quantization-aware training (QAT) exists for. The idea is almost insultingly simple to state and surprisingly deep once you look closely: instead of quantizing a finished model and hoping it survives, you *simulate* the quantization during training, so the network's own gradient descent learns weights that are robust to being rounded to int8. The forward pass pretends to be the int8 model; the backward pass updates high-precision master weights; and after a short fine-tune the weights have arranged themselves so that rounding them barely changes the output. PTQ asks "how much does my trained model break when I quantize it?" QAT asks "what model would I have trained if I had known it was going to be quantized all along?" — and that second question almost always has a much better answer.

There is a catch you have to respect, because half of doing this well is knowing when *not* to. PTQ is minutes; QAT is a training run — hours to days, plus you need the labeled training data and the training infrastructure, neither of which the PTQ path required. So QAT is not the default. It is the escalation: you reach for it when PTQ's accuracy drop exceeds your budget, when you need to go below 8 bits (int4, ternary, binary), or when the model is small enough that quantization noise hurts it disproportionately. Figure 1 is the whole mechanism on one diagram — the fp32 master weights, the fake-quantization node in the forward pass, the task loss, and the straight-through estimator carrying the gradient back across the non-differentiable round. Keep it in view; everything in this post is an elaboration of that single loop.

![A dataflow graph showing fp32 master weights flowing into a fake-quant node, then an int8-simulated matmul, then the task loss, with the straight-through estimator carrying the gradient back to update the fp32 weights](/imgs/blogs/quantization-aware-training-qat-1.png)

By the end you will be able to do four concrete things. First, derive the straight-through estimator from scratch and explain exactly why it is biased and why that bias is usually harmless. Second, write a correct `torch.ao.quantization.prepare_qat` flow — the QConfig, the fake-quant observers, the fine-tune loop, and the late-training freezes that separate a QAT run that works from one that quietly diverges. Third, reason about learned step size (LSQ), the single most important upgrade to vanilla QAT, including the gradient of the scale parameter and why its magnitude has to be tamed. Fourth, decide — with a table you can act on — whether a given model is a PTQ job or a QAT job, and roughly what it will cost you either way. This is the QAT sibling to the post-training-quantization post in this series; they are the two halves of the quantization lever, and QAT is the half you pull when the cheap half is not enough.

One framing to carry through: in this series we keep coming back to four levers — quantization, pruning, distillation, and efficient architecture — sitting on compilers and runtimes and read off an accuracy–efficiency Pareto frontier. QAT is not a new lever; it is the *accuracy-preserving mode* of the quantization lever. PTQ moves you down and to the left on the size/latency axes but also drops you on the accuracy axis; QAT is what lets you take the same size/latency step *without* falling off the accuracy frontier — and, at low bit-widths, it is what lets you make a step PTQ could not survive at all. So everything here is in service of one Pareto question: how far can I shrink and speed up this model before accuracy is no longer acceptable, and QAT is the technique that pushes that frontier outward for the quantization lever specifically.

## 1. What QAT actually changes (and what it does not)

Start from the thing PTQ does, because QAT is best understood as a modification of it. In int8 quantization you map a real number $x$ in some range to an 8-bit integer with an affine map. With a scale $s$ (a positive real) and a zero-point $z$ (an integer), the quantize and dequantize operations are:

$$
q = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right) + z,\; q_\min,\; q_\max\right), \qquad \hat{x} = s\,(q - z)
$$

For signed symmetric int8, $q_\min = -127$, $q_\max = 127$, and $z = 0$; the only knob is $s$. PTQ runs the trained fp32 network on a calibration set, watches the min and max (or some percentile, or the MSE-optimal clip) of each weight tensor and each activation tensor, picks an $s$ for each, and then at inference quantizes everything. The model never gets a vote. Whatever distribution of weights training happened to produce, PTQ rounds it.

QAT inserts the quantize-then-dequantize round-trip — $\hat{x} = s\,(\mathrm{clip}(\mathrm{round}(x/s) + z, q_\min, q_\max) - z)$ — into the forward pass *during training*, as a module the literature calls a **fake-quantization node** (sometimes "simulated quantization" or "quant/dequant"). It is "fake" because the tensor that comes out is still floating point — it has just been forced onto the int8 grid and back. The arithmetic downstream is still done in fp32, but every weight and (usually) every activation has been snapped to the values it would take in the real int8 model. The network therefore sees, on every forward pass, the *exact rounding error* it will face at deployment, baked into its loss. Gradient descent then does what it always does: it moves the weights to reduce that loss. Because the loss already includes the quantization error, the weights it finds are ones for which the quantization error is small. That is the entire trick.

Crucially, QAT keeps a set of **fp32 master weights** that it actually updates. The int8 values are derived from them on the fly inside the fake-quant node and thrown away after each step. You never optimize integers directly — you optimize the underlying real-valued weights, and the integers are a deterministic function of them. This matters for a reason that becomes the central technical problem of the whole technique: integers are a step function of the reals, and step functions have no useful gradient. Hold that thought; section 3 is entirely about it.

What QAT does *not* change is the deployment model. After fine-tuning, you "convert": the fake-quant nodes are replaced with real int8 operators, the fp32 master weights are quantized one final time using the now-trained scales, and you ship an int8 model that is bit-for-bit the same *shape* as the PTQ int8 model — same size, same operators, same latency on the same hardware. QAT does not buy you a smaller or faster model than PTQ at the same bit-width. It buys you a *more accurate* one. That is the only thing it is for, and you should keep it crisp in your head: QAT trades training cost for accuracy, holding size and speed fixed. If your size and speed are already where you want them and only your accuracy is short, QAT is the lever; if you need to go smaller or faster, that is a different lever (a lower bit-width, pruning, a smaller architecture) and QAT is at most a helper that lets you survive the more aggressive setting.

### Why the fp32 master weights are non-negotiable

It is tempting to ask: if the deployed model is int8, why carry fp32 master weights through training at all — why not just optimize the integers directly? The answer is the reason QAT works, and it is worth slowing down on. Gradient descent makes *tiny* updates: a single step might want to move a weight by $10^{-4}$ of its scale. An int8 weight cannot represent a change that small — the smallest move it can make is one full level, $s$, which is thousands of times larger than the update wants. If you tried to update integers directly, almost every update would round to zero (no change) and the rare ones that did not would over-shoot by a full level. Learning would be a coarse, noisy stagger instead of a smooth descent.

The fp32 master weights solve this by *accumulating* the tiny updates at full precision. Each step nudges the fp32 weight by its real gradient; the integer the network actually uses is recomputed from the fp32 weight on the next forward pass. So a weight can drift slowly across many steps until it crosses a grid boundary and *then* its int8 value flips — the integer changes only when the accumulated evidence justifies it. This is the same trick used in low-precision training generally (a high-precision accumulator behind a low-precision compute path), and it is why "keep fp32 master weights, derive the integers each step" is not an implementation detail but the core of the method. The integers are the model you ship; the fp32 weights are the medium in which learning is even possible.

### The one-line summary you can give your manager

PTQ takes a trained model and rounds it; QAT trains a model that expects to be rounded. The first is free and sometimes good enough. The second costs a fine-tune and is almost always good enough, including at bit-widths where the first one falls off a cliff.

## 2. Why PTQ has a ceiling: the loss landscape did not consent

It is worth being precise about *why* PTQ loses accuracy, because that precision is exactly what tells you when QAT will help and by how much. Quantization adds noise to every weight and activation. For a weight quantized with step $s$ under round-to-nearest, the rounding error is, to a good approximation, uniform on $[-s/2, s/2]$, which has variance $s^2/12$. That is the standard quantization-noise model and it gives the familiar signal-to-quantization-noise rule of thumb, $\mathrm{SQNR} \approx 6.02\,b + 1.76$ dB for $b$ bits — every bit you remove costs you about 6 dB of fidelity. PTQ accepts whatever SQNR the trained weight distribution and your chosen scales produce, and then *propagates that noise through a network that was never optimized to tolerate it.*

Here is the key insight. A trained fp32 network typically sits in a fairly sharp minimum of the loss along many weight directions — small perturbations to certain weights move the loss a lot, because the network has learned to use those weights precisely. Quantization is a perturbation. It nudges every weight by up to $s/2$, and it nudges them *in a correlated, structured way* (every weight in a tensor is snapped to the same grid). If the minimum is sharp in the directions that quantization happens to push, the loss goes up — that is your three accuracy points. PTQ has no mechanism to do anything about this; it is rounding a fixed point.

There is a clean way to see this geometrically. Think of the loss as a landscape over weight space, and quantization as a constraint that snaps your current point to the nearest lattice vertex (the int8 grid). PTQ leaves you wherever fp32 training landed and then jumps you to the nearest vertex — if that vertex sits up the wall of a sharp valley, your loss spikes. QAT instead lets the optimizer *walk to a vertex that is already low*, because every forward pass is evaluated *at* the snapped point, so the gradient is constantly pulling the fp32 weights toward configurations whose nearest vertex is good. The fp32 weights end up sitting in a basin that is wide enough that the lattice vertex inside it is nearly as good as the basin floor. That "wide enough basin" is the flat-minimum intuition, and it is why QAT solutions are robust to quantization in a way PTQ solutions never are: QAT does not just round, it relocates.

QAT changes the optimization target. By the time you finish the fine-tune, you are not at the fp32 minimum any more — you are at a minimum of the *quantized* loss surface, a point chosen specifically because moving every weight onto the int8 grid does not increase the loss much. Empirically, and with some theoretical backing from the flat-minima literature, QAT tends to find solutions that are *flatter* in the directions quantization perturbs. The weights spread out toward grid points; the distribution that was awkward for a fixed grid becomes one the grid represents cleanly. The loss landscape, so to speak, was never asked whether it minded being quantized — QAT asks, and the network answers by relocating to a part of the landscape where it does not mind. That is the mechanism behind every QAT-recovers-accuracy result you will ever see, and it is why the recovery is largest exactly where PTQ's damage is largest: small models, low bit-widths, and layers with heavy-tailed distributions.

#### Worked example: how much room is there to recover?

Take the running model for this post: a MobileNetV2-class image classifier, fp32 top-1 of 76.1% on its validation set. MobileNets are notoriously PTQ-hostile — they are small, depthwise-separable, and have layers with wide dynamic range — so they are the canonical case where QAT earns its keep. With per-channel weight quantization and good activation calibration, PTQ int8 lands around 73.0%, a 3.1-point drop. The fp32 model's accuracy is the ceiling QAT is chasing; the PTQ number is the floor it is climbing from. The gap, 3.1 points, is your *available recovery budget*. A well-run QAT fine-tune on this model recovers to roughly 75.8%, i.e. it closes about 90% of the gap and lands 0.3 points under fp32. Those are the numbers we will carry through the rest of the post and put in the result table; they are consistent with the MobileNet QAT results reported in Jacob et al. (2018) and the PyTorch quantization tutorials, and I have reproduced numbers in this neighborhood myself on similar models. The lesson to internalize: **the size of the gap PTQ opens is a good predictor of how much QAT will give you back**, because both are driven by the same thing — how much the trained model's geometry disagreed with the grid. Figure 3 captures this PTQ-damage-versus-QAT-recovery picture as a before-after.

![A before-after diagram showing PTQ int8 dropping the model three points below the fp32 baseline while QAT int8 recovers to within three tenths of a point at the same size and latency](/imgs/blogs/quantization-aware-training-qat-3.png)

There is a second, subtler reason this gap predicts recovery, and it is worth a paragraph because it changes how you triage models. The 3.1-point PTQ drop is not spread evenly across the network. If you PTQ each layer in isolation — quantize one layer, leave the rest fp32, measure the accuracy hit — you almost always find that two or three layers account for most of the damage while the rest barely move. On MobileNets the usual culprits are the depthwise convolutions (their per-channel ranges are wide and uneven, so a single tensor-wide scale wastes most of the int8 grid on a few outlier channels) and the first and last layers (small tensors, high sensitivity). This per-layer sensitivity profile is gold: it tells you *where* QAT will spend its recovery, and it tells you which one or two layers you might instead keep at higher precision with mixed-precision quantization, sometimes recovering most of the gap without any training at all. Always profile per-layer sensitivity before deciding QAT is necessary — the cheapest fix is often a one-layer exception, not a full fine-tune.

## 3. The gradient problem: round() is a wall

Now the hard part, and the reason QAT needed a clever idea to work at all. To train through the fake-quant node, backpropagation needs a derivative of the node's output with respect to its input. The node is essentially $\hat{x} = s \cdot \mathrm{round}(x/s)$ (ignoring zero-point and clipping for a moment). What is $\frac{d\hat{x}}{dx}$?

The function $\mathrm{round}(\cdot)$ is a staircase. Between any two half-integers it is constant — flat — so its derivative is exactly **zero** almost everywhere. At each half-integer it jumps, so the derivative is undefined (you could call it a Dirac spike, plus infinity). Put together: the gradient of $\hat{x}$ with respect to $x$ is zero on the entire interior of every step and infinite on a measure-zero set of edges. If you backpropagate that honestly, every weight gets a gradient of zero, no weight ever updates, and training does precisely nothing. The chain rule multiplies your loss gradient by zero and the signal dies at the fake-quant node. This is not a numerical inconvenience; it is a fundamental obstruction. You cannot learn through a step function with ordinary calculus. Figure 2 contrasts this dead gradient with the fix we are about to derive.

![A before-after diagram contrasting the true gradient of round which is zero almost everywhere with the straight-through estimator which approximates the slope as one inside the clip range and zero outside](/imgs/blogs/quantization-aware-training-qat-2.png)

### 3.1 The straight-through estimator, derived

The fix, introduced by Hinton in lectures and formalized by Bengio, Léonard, and Courville (2013), is the **straight-through estimator** (STE). The reasoning is this. We cannot use the true gradient of the rounding step because it is useless. But we *believe* — and it is a defensible belief — that $\hat{x}$ is, on average, close to $x$; rounding shifts a value by at most half a step, and that error has zero mean under the uniform model. So as far as the *gradient* is concerned, let us pretend the round was the identity. We define the forward pass to use the real round (so the loss reflects true quantization), and we define the backward pass to behave as if $\hat{x} = x$, i.e. we set:

$$
\frac{\partial \hat{x}}{\partial x} \;\overset{\text{STE}}{:=}\; 1 \quad\text{(within the representable range)}
$$

Concretely, the gradient that flows back to the fp32 weight is just the gradient that arrived at the quantized weight, copied through unchanged. If $W_q = \mathrm{fakequant}(W)$ and the loss is $L$, then

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_q}\cdot \frac{\partial W_q}{\partial W} \;\overset{\text{STE}}{=}\; \frac{\partial L}{\partial W_q}\cdot \mathbf{1}\big[q_\min \le \tfrac{W}{s}+z \le q_\max\big]
$$

The indicator $\mathbf{1}[\cdot]$ is the one important refinement over a naive "gradient = identity everywhere." A weight whose value has been *clipped* — pushed past $q_\min$ or $q_\max$ — genuinely cannot change the output by moving further out; the clip has saturated. So outside the range, the gradient should be zero (the value is pinned), and inside the range it should be one (pass straight through). This clipped STE is what every real QAT implementation uses. In words: **the straight-through estimator passes the gradient through the rounding as if it were the identity, except where the value is clipped, where it passes nothing.** That is the single equation the whole technique stands on, and figure 1's "STE backward" arrow is exactly this.

### 3.2 The bias of the STE, honestly

The STE is not the true gradient, and pretending otherwise causes subtle bugs in people's intuition, so let us be exact about the error. The function we are optimizing in the forward pass is $L(\hat{x}(x))$ where $\hat{x}$ is the staircase. The *true* gradient $\frac{dL}{dx}$ is (almost everywhere) zero. The STE gradient is $\frac{dL}{d\hat{x}}\cdot 1$, generally nonzero. So the STE gradient is **biased** — it is not an unbiased estimator of the true gradient of the forward function; the true gradient is zero and the STE's is not. That sounds alarming until you reframe what we are actually trying to minimize.

We do not want to minimize $L(\hat{x}(x))$ pointwise; we want to find an $x$ such that $L(\hat{x})$ is small *and* stays small under the quantization map. The honest object of interest is something like a *smoothed* loss — the loss averaged over the quantization noise, $\tilde{L}(x) = \mathbb{E}_{\epsilon}[L(x + \epsilon)]$ with $\epsilon$ the rounding error. That smoothed loss *does* have a meaningful, nonzero gradient, and to first order it equals the gradient of $L$ evaluated at $x$ — which is exactly what the STE computes (treating the round as identity recovers $\frac{dL}{dx}$ of the unrounded function). Seen this way the STE is not a hack with no theory; it is a first-order estimator of the gradient of the noise-smoothed loss. The bias is the gap between that first-order picture and the true curved landscape, and it is small precisely when the loss is locally smooth at the scale of one quantization step — which is most of the time, for most weights, away from the clip boundaries. Yan et al. (2019, "Understanding STE") and others have made this rigorous; the practical upshot is what matters here: **the STE is biased, the bias is usually benign, and it degrades exactly where you would expect — at very low bit-widths where the step $s$ is large and the smoothness assumption breaks.** That breakdown is one of the reasons int4 and int2 QAT need extra care (and motivated LSQ, next).

### 3.3 Stochastic rounding: the unbiased alternative the STE does not use

There is a sibling idea worth knowing because it sharpens what the STE is and is not. The STE keeps the *forward* round deterministic (round-to-nearest) and fakes only the backward pass. An alternative is **stochastic rounding**: in the forward pass, round $x/s$ up or down at random with probability proportional to how close it is to each grid point, so that $\mathbb{E}[\hat{x}] = x$ exactly. Stochastic rounding makes the forward quantizer *unbiased in expectation* — over many steps the rounding error averages to zero rather than to a deterministic offset — which can help low-precision training because it injects gradient signal that round-to-nearest's flat steps suppress. It is the trick behind a lot of low-precision *training* (as opposed to inference quantization), where you want the master weights themselves stored in low precision.

So why does inference-oriented QAT mostly *not* use it? Because at deployment you will round-to-nearest deterministically — that is what the int8 hardware does — and you want training to simulate the *deployed* behavior, not a fancier one. Stochastic rounding at train time but round-to-nearest at deploy time is a train/serve mismatch, the exact thing the late freezes in section 4 exist to prevent. The STE's choice — deterministic round-to-nearest forward, identity backward — is the one that matches deployment while still giving usable gradients. Knowing stochastic rounding exists, though, clarifies the STE's bias: the STE is biased *because* it keeps the forward round deterministic; the unbiased alternative pays for that with a train/serve gap you do not want for an inference model. It is a clean illustration that "unbiased" is not automatically "better" — matching deployment beats statistical purity here.

### 3.4 The activation STE and the clip range as a learnable thing

The same STE logic applies to activations, with one twist: activations have a *clip range* you have to choose (weights are symmetric around zero; activations after a ReLU are not). The popular PACT method (Choi et al., 2018) makes the activation clip value $\alpha$ a learnable parameter and derives its gradient through the STE: $\frac{\partial \hat{a}}{\partial \alpha} = 1$ for $a > \alpha$ (clipped values move with the clip) and $0$ otherwise. Learning the clip rather than fixing it from calibration is one of the biggest single wins in low-bit QAT, because a too-wide clip wastes precision on rare large activations while a too-narrow one saturates useful ones. This idea — *make the quantization parameters themselves trainable* — generalizes into LSQ, which learns the *step size* directly, and is the subject of section 5.

## 4. The PyTorch QAT flow, end to end

Enough theory; here is the runnable spine. PyTorch's eager-mode quantization API lives in `torch.ao.quantization`. The QAT flow has four moves: pick a QAT config, `prepare_qat` to insert fake-quant + observers, fine-tune, and `convert` to a real int8 model. I will use the eager API because it is the clearest for teaching; the FX-graph (`prepare_qat_fx`) and the newer PT2E export flow follow the same logic with less manual module surgery, and I will note the differences.

First, fuse the modules that should be quantized as a unit — Conv-BN-ReLU folds into a single quantized op, and fusing before quantizing is what makes batch-norm folding (section 6) correct. Then attach a QAT QConfig and prepare.

```python
import torch
import torch.nn as nn
import torch.ao.quantization as tq

model = build_mobilenet_v2(pretrained=True).train()  # start from CONVERGED fp32

# 1. Fuse Conv+BN+ReLU triples so they quantize as one op (and BN folds correctly).
model = tq.fuse_modules_qat(
    model,
    modules_to_fuse=[["features.0.0", "features.0.1", "features.0.2"]],  # example triple
)

# 2. Choose a QAT qconfig. fbgemm = x86 server int8; qnnpack = ARM mobile int8.
#    get_default_qat_qconfig wires per-channel weight fake-quant + moving-average
#    activation observers, which is the right default for CNNs.
model.qconfig = tq.get_default_qat_qconfig("qnnpack")  # target: ARM (phone)

# 3. Insert fake-quant + observers in place. The model now SIMULATES int8 in fp32.
tq.prepare_qat(model, inplace=True)
```

After `prepare_qat`, every weight and activation flows through a `FakeQuantize` module that wraps an observer (which tracks the running range to set the scale) and applies the quant/dequant round-trip with the STE on the backward. The model is still fp32 in storage and still trains with ordinary autograd; it just *behaves* like int8 numerically. Now the fine-tune loop — and this is where the guidance lives that decides success.

```python
# 4. Fine-tune. KEY: tiny LR (~1/100 of original), few epochs, cosine decay.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9,
                            weight_decay=1e-5)            # base was 1e-2 -> /100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

NUM_EPOCHS = 5
FREEZE_OBSERVER_EPOCH = 3      # stop updating activation ranges late
FREEZE_BN_EPOCH = 4           # switch BN to running stats late

for epoch in range(NUM_EPOCHS):
    if epoch == FREEZE_OBSERVER_EPOCH:
        # Stop the observers from moving the quant ranges; lock the scales.
        model.apply(tq.disable_observer)
    if epoch == FREEZE_BN_EPOCH:
        # Freeze BN running mean/var so train- and eval-time stats agree.
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()      # STE makes this gradient flow through the fake-quant
        optimizer.step()
    scheduler.step()
    evaluate(model, val_loader)   # eval in fake-quant mode tracks the int8 number
```

Two design choices in that loop are not optional and are the most common reason a QAT run underperforms. **Freezing the observers late** (`disable_observer`) stops the activation quantization ranges from drifting in the final epochs. Early on you *want* the ranges to adapt as the weights move; late, you want them pinned so the network can settle into a fixed grid and the train/eval numbers converge. **Freezing BN statistics late** (`freeze_bn_stats`) switches batch-norm from batch statistics to running statistics, so the model you evaluate is the model you will deploy — failing to do this is the classic "great in training, worse after convert" bug. Section 6 explains the BN-folding mechanics behind this. Finally, convert:

```python
# 5. Convert to a real int8 model: fake-quant -> real int8 ops, scales baked in.
model.eval()
int8_model = tq.convert(model, inplace=False)

# 6. Sanity: the converted model must match the fake-quant eval within noise.
torch.jit.save(torch.jit.script(int8_model), "mobilenet_qat_int8.pt")
```

The post-convert accuracy should match the last fake-quant eval number to within a few hundredths of a point. If it does not — if convert drops a point — your observers or BN stats were not frozen, or you fused the wrong modules, and the int8 graph is using different scales than the ones you trained against. That mismatch is the single most common QAT bug, and it is why the freeze steps are in the loop and not optional polish. Figure 4 lays the whole schedule out as a timeline so the ordering of these freezes is unmistakable.

![A timeline showing the QAT schedule from a converged fp32 checkpoint through prepare_qat, a low learning rate fine-tune, freezing observers, freezing batch-norm statistics, and conversion to int8](/imgs/blogs/quantization-aware-training-qat-4.png)

#### Worked example: what a QAT run actually costs in wall-clock and dollars

Make the cost concrete, because "a training run" is too vague to plan against. Our MobileNetV2-class model has about 3.5M parameters and trains on an ImageNet-scale dataset. A full fp32 training run from scratch is ~90 epochs and, on a single mid-range cloud GPU, the better part of a day. The QAT fine-tune is *not* that — it is 5 epochs resuming from the converged checkpoint. On one such GPU, one epoch over the training set is roughly 12 minutes for this model, so 5 epochs is about an hour of compute, call it \$1–\$3 of on-demand GPU time, plus maybe a half-day of *your* time the first time you wire up the fuses, freezes, and LR. Compare that to PTQ: ~90 seconds of calibration on a laptop CPU, effectively \$0, and fifteen minutes of your time. So the honest cost comparison for this model is roughly "\$0 and a coffee break" versus "a few dollars and an afternoon" — and the payoff is converting a −3.1-point model into a −0.3-point one. That ratio is why QAT is a clear yes when PTQ misses the budget on a small CNN. Now scale the same arithmetic to a 7B LLM: one epoch over a fine-tune corpus on a multi-GPU node is hours and the dollar figure jumps two or three orders of magnitude, which is precisely why LLM practice prefers PTQ methods and reserves QAT-style training for the cases where accuracy is non-negotiable or the bit-width is extreme. The technique is the same; the economics decide whether you pull the lever.

### 4.1 The newer flows: FX and PT2E

The eager flow above requires you to name fusion targets by string, which is tedious for big models. `tq.quantize_fx.prepare_qat_fx` traces the model to an FX graph and fuses/inserts fake-quants automatically from the same QConfig — same semantics, far less boilerplate. The most modern path is **PT2E** (`torch.export` + `prepare_qat_pt2e`), which works on the exported graph and pairs with backend-specific quantizers (`XNNPACKQuantizer` for mobile CPU, the TensorRT or X86 quantizers for servers). For new code targeting ExecuTorch on-device, PT2E is the direction to learn; the STE math, the freeze discipline, and the LR guidance are all identical — only the plumbing changes.

### 4.2 TensorFlow / Keras QAT, briefly

If you live in the TFLite world, the equivalent is the TensorFlow Model Optimization Toolkit (`tfmot`). The flow mirrors PyTorch's:

```python
import tensorflow_model_optimization as tfmot

qat_model = tfmot.quantization.keras.quantize_model(base_model)  # inserts fake-quant
qat_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
qat_model.fit(train_ds, epochs=5, validation_data=val_ds)        # short, low LR

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]            # emit int8 tflite
int8_tflite = converter.convert()
```

`tfmot` uses the same fake-quant + STE machinery under the hood and produces a `.tflite` that LiteRT runs on the NNAPI/XNNPACK/Hexagon delegates. The conceptual contract is identical: simulate int8 during training, then export a real int8 graph. Whichever framework you use, the four ideas — fake-quant forward, STE backward, fp32 master weights, late freezes — are the invariants.

## 5. Learned step size (LSQ): make the scale a parameter

Vanilla QAT learns the *weights* and sets the quantization *scale* with an observer (a running estimate of the range). The single biggest upgrade to that is to stop treating the scale as a passive statistic and start treating it as a **learnable parameter trained by gradient descent**, jointly with the weights. This is LSQ — Learned Step Size Quantization — from Esser et al. (2020), and it is the technique I reach for first whenever int8 QAT is not quite enough or whenever I am going below 8 bits. Figure 5 lays out its anatomy.

![A layered stack showing how the task loss drives the straight-through estimator which produces both the weight gradient and a step-size gradient that is rescaled by a gain factor before updating the per-layer learned step](/imgs/blogs/quantization-aware-training-qat-5.png)

### 5.1 The step-size gradient, derived

Write the quantizer with a learnable step $s$. Let $v$ be the real value, and define $\bar{v} = \mathrm{clip}(v/s, -Q_n, Q_p)$ with $Q_n, Q_p$ the negative and positive integer limits (for signed int8, $Q_n = Q_p = 127$ roughly). The quantized value is $\hat{v} = s\cdot \mathrm{round}(\bar{v})$. We want $\frac{\partial \hat{v}}{\partial s}$ so we can train $s$. Apply the STE to the round (slope 1 inside the range) and differentiate the three regimes:

$$
\frac{\partial \hat{v}}{\partial s} =
\begin{cases}
-\,v/s + \mathrm{round}(v/s) & \text{if } -Q_n < v/s < Q_p \quad(\text{in range}) \\[4pt]
-Q_n & \text{if } v/s \le -Q_n \quad(\text{clipped low}) \\[4pt]
Q_p & \text{if } v/s \ge Q_p \quad(\text{clipped high})
\end{cases}
$$

Read the three cases physically. **In range**: nudging $s$ changes the reconstruction by the *rounding residual* $\mathrm{round}(v/s) - v/s$ — exactly how far this value sits from its grid point. So $s$ feels a pull from every in-range element proportional to how badly that element is rounded, and it moves to reduce the aggregate residual. **Clipped**: a clipped element's reconstruction is $\pm Q\cdot s$, so its sensitivity to $s$ is just $\pm Q$ — growing $s$ moves the clip boundary out (good if you are clipping useful signal, bad if you are wasting range). The learned step thus balances two competing forces automatically: shrink $s$ to round in-range values more finely, grow $s$ to clip fewer large values. That trade-off is the one PTQ calibration tries to guess statically; LSQ learns it from the actual loss. This is the same insight as PACT's learnable clip, generalized to the step itself, and the reason LSQ tends to beat fixed-scale QAT especially hard at low bit-widths.

### 5.2 The gradient-scale trick (why LSQ needs $g$)

There is a subtlety Esser et al. flag and it is easy to get wrong: the raw gradient $\frac{\partial L}{\partial s}$ is summed over *every element* of the weight tensor, so a layer with a million weights produces a step-size gradient roughly a million times larger in magnitude than any single weight's gradient. If you train $s$ with the same optimizer settings as the weights, $s$ either explodes or has to be crippled with a tiny separate LR. LSQ fixes this with a **gradient scale** $g$ applied to the step-size gradient:

$$
g = \frac{1}{\sqrt{N_W \, Q_p}}, \qquad \frac{\partial L}{\partial s}\;\leftarrow\; g\cdot \frac{\partial L}{\partial s}
$$

where $N_W$ is the number of elements quantized by this step and $Q_p$ the positive integer limit. The $1/\sqrt{N_W}$ factor undoes the summation-over-many-elements inflation; the $1/\sqrt{Q_p}$ factor balances scale-parameter updates against weight updates across bit-widths. With this gain, the step size and the weights can train with *the same* learning rate and stay balanced — which is the whole point, because it makes LSQ a drop-in: same optimizer, same LR, just a few extra learnable scalars. LSQ+ (Bhalgat et al., 2020) extends this with a learnable *offset* (zero-point) too, which matters for activations that are not symmetric around zero (post-ReLU, GELU, etc.). On ImageNet, LSQ pushed 3-bit and 4-bit ResNet accuracy to within a fraction of a point of full precision, results that fixed-scale QAT could not reach — which is exactly the regime where the STE bias and a badly-set static scale do the most damage.

### 5.3 Per-channel learned steps, and initialization

Two practical refinements decide whether LSQ actually delivers. First, **per-channel** steps for weights: instead of one $s$ for the whole weight tensor, learn one $s$ per output channel. Convolution channels often have wildly different magnitudes — a per-tensor scale forced to cover the loudest channel wastes the grid on the quiet ones, which is precisely the MobileNet depthwise problem from section 2. Per-channel learned steps let each channel keep its own grid, and the cost is trivial (a few extra scalars per layer, all folded into the int8 kernel's per-channel scale at deploy, which every modern runtime supports). For activations you cannot go per-channel as cheaply (the channel axis moves through the network), so activations usually stay per-tensor with a learned clip.

Second, **initialization**. A learnable parameter trained by gradient descent still needs a sane starting point, and LSQ's recommended init is $s_0 = 2\langle|v|\rangle / \sqrt{Q_p}$ — twice the mean absolute value of the tensor divided by $\sqrt{Q_p}$. Initialize the step from a calibration pass (effectively, start LSQ from the PTQ scale) and the fine-tune begins near a good solution and only has to refine it; initialize it badly and the first few epochs are wasted clawing the scale back to a reasonable range, sometimes destabilizing the weights in the process. This is the same principle as the whole technique: QAT works best as a *refinement* of a good starting point, never as a fresh search — start from the trained weights, start the scales from calibration, and fine-tune gently.

#### Worked example: int4 is where LSQ pays for itself

Take the same MobileNet-class model and go to int4 weights, int8 activations. PTQ at int4 is a disaster on this architecture — round-to-nearest at four bits on depthwise layers collapses to roughly 61% top-1, a ~15-point hole, because four bits is just sixteen levels and the per-channel ranges are wide. Vanilla fixed-scale QAT at int4 recovers a lot but plateaus a couple of points short because the static scales are wrong for the trained distribution. LSQ at int4 — learning the per-layer step alongside the weights — lands around 73% top-1, within roughly three points of the fp32 76.1% and a full ~12 points above int4 PTQ. The pattern is the rule, not the exception: **the lower the bit-width, the more the advantage of QAT over PTQ, and of learned-scale over fixed-scale QAT, compounds** — because at low bits both the rounding error and the cost of a mis-set scale are large, and only training can co-adapt the weights and the grid to each other. Figure 7 puts this bit-width trend in a single matrix.

![A matrix showing top-1 accuracy for PTQ and QAT at int8, int4, and ternary, with the QAT advantage widening from a few points at int8 to double digits at int4 and to the only working option at two bits](/imgs/blogs/quantization-aware-training-qat-7.png)

The mechanism behind the widening gap deserves to be made explicit, because it is the single most useful prediction in this whole post. At int8 the quantization step $s$ is small — 256 levels span the range, so each level is a fine slice of the distribution, the rounding error is tiny relative to the spread of the weights, and the STE's smoothness assumption holds almost perfectly. PTQ already does most of the job and QAT only has a couple of points to clean up. As you drop bits, the number of levels halves with each bit: 16 levels at int4, 4 at 2-bit, 2 at 1-bit. The step $s$ grows correspondingly, the rounding error becomes a large fraction of the weight spread, the loss is no longer smooth at the scale of one step, and two things break at once for PTQ — the rounding noise itself is large, *and* a statically chosen scale is now badly wrong because there is no scale that rounds a wide distribution well with only 16 levels. QAT attacks both: it reshapes the weight distribution so that 16 levels are enough (the weights migrate toward grid points), and with LSQ it learns the scale that makes those 16 levels land where the weights actually are. That is why the curve for PTQ falls off a cliff while the QAT curve stays nearly flat — and why, below about 4 bits, there is effectively no PTQ at all and the technique stops being optional.

## 6. Batch-norm folding during QAT (the detail that breaks runs)

Batch normalization is the source of more silent QAT failures than anything except the learning rate, so it gets its own section. At inference, a BN layer is an affine map: $y = \gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$ with $\mu, \sigma^2$ the *running* (frozen) statistics. Because it is affine and it sits right after a convolution, you can **fold** it into the preceding conv's weights and bias — there is no reason to spend a separate op on it at inference, and indeed every deployment runtime folds Conv+BN into a single Conv. The folded weight is $W' = W\cdot \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}$ and the folded bias absorbs $\beta, \mu$.

To be concrete about the fold: a convolution $y = W x + b$ followed by BN produces, at inference, $\mathrm{BN}(Wx+b) = \gamma\frac{(Wx+b)-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$. Group the terms and it is a single affine map with a *folded* weight and bias:

$$
W' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\,W, \qquad b' = \frac{\gamma\,(b-\mu)}{\sqrt{\sigma^2+\epsilon}} + \beta
$$

The deployed kernel uses $W'$ and $b'$ — there is no separate BN op. So the tensor that *must* be quantized well is $W'$, the folded weight, not the raw $W$. And $W'$ depends on $\mu, \sigma^2$ — which is exactly where the danger lives.

The trap is that during *training*, BN uses *batch* statistics ($\mu_B, \sigma_B^2$ computed per mini-batch), which differ from the running statistics ($\mu, \sigma^2$) used at inference. If your fake-quant quantizes $W'_{\text{batch}} = \frac{\gamma}{\sqrt{\sigma_B^2+\epsilon}}W$ during training but the deployed kernel folds with $W'_{\text{run}} = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W$, then the scale you learned was calibrated to a weight tensor you will never ship, and the per-channel ranges can be off by tens of percent on channels where batch and running variance disagree. If you quantize the conv weight using the batch-statistics-folded version during training but deploy the running-statistics-folded version, you trained against the wrong weights — the scales are calibrated to a weight tensor you will not ship. The result is the maddening "fake-quant eval looked fine, post-convert eval dropped a point" bug. Correct QAT BN folding (as in Jacob et al. 2018 and the PyTorch `intrinsic.qat` Conv-BN modules) simulates the *inference* folding during training: it computes the fold with running statistics, quantizes the folded weight, but still lets BN learn from the batch statistics. The practical lever is the one already in the section-4 loop: **freeze BN running stats late in training** (`freeze_bn_stats`) so that for the final epochs the train-time and inference-time folded weights are identical, and the quantization scales the network learns are the ones it will actually use. Fuse Conv+BN+ReLU before `prepare_qat`, freeze BN late, and this entire class of bug disappears.

## 7. Cost and benefit: QAT versus PTQ as an engineering decision

Now the decision the whole post is building toward, framed as a cost-benefit. PTQ and QAT are not competitors; they are a ladder. You climb the cheap rung first and only step up if it does not get you there. Figure 6 is the cost-benefit matrix; figure 8 is the decision tree. Here is the prose argument, because the numbers alone do not capture the asymmetry.

![A matrix comparing PTQ and QAT across setup time, data needed, accuracy recovery, and sub-8-bit viability, showing PTQ cheaper and QAT more accurate](/imgs/blogs/quantization-aware-training-qat-6.png)

**What PTQ costs**: minutes of wall-clock, a calibration set of ~100–500 *unlabeled* examples, and zero training infrastructure. You can do it on a laptop, on a model whose training pipeline you do not even have access to, the afternoon before a demo. That last point is underrated — PTQ works on a model you only have as a checkpoint, with no data and no trainer. QAT does not.

**What QAT costs**: a real fine-tune. You need the *labeled* training data (or a good proxy), a working training loop, and hours to days of accelerator time depending on model size and how many epochs you run (CNNs: a handful of epochs; LLMs: this gets expensive fast, which is why LLM practice leans on PTQ methods like GPTQ/AWQ and reserves QAT for cases where accuracy is truly critical). You also spend *engineer* time — getting the fuses, freezes, and LR right is a half-day of fiddling the first time. None of that is free, and on a large model it can be the dominant cost of the whole optimization.

**What QAT buys**: accuracy you cannot get any other way at that bit-width. The 3-point hole becomes 0.3. The int4 model that was unusable under PTQ becomes shippable. And — this is the part people forget — QAT *unlocks lower bit-widths*, which is its own kind of efficiency win: an int4 QAT model is half the size of an int8 model and may be the difference between fitting in a microcontroller's flash or not. So although QAT does not make the model smaller *at a fixed bit-width*, it lets you *choose* a smaller bit-width that PTQ could not survive, and that indirectly buys size and speed.

The decision rule that falls out: **try PTQ first, always.** It is cheap enough that not trying it is malpractice. If PTQ hits your accuracy budget, ship it and go home — running QAT to recover accuracy you already have is wasted compute and wasted engineer-days. Escalate to QAT only when (a) PTQ's drop exceeds your budget after you have exhausted the cheap PTQ tricks (per-channel weights, percentile/MSE clipping, mixed-precision for the worst layers), or (b) you need to go below 8 bits, where PTQ degrades fast and QAT is often the only thing that works, or (c) the model is small and PTQ-hostile (MobileNets, tiny BERTs, anything depthwise-heavy) where the relative damage is largest. And one more: if you have *no* labeled data, QAT is off the table — fall back to data-free distillation or aggressive mixed-precision PTQ. Figure 8 encodes exactly this.

![A decision tree starting from running PTQ first, branching on whether the drop is within budget and whether you need int4 or have labeled data and training time, leading to ship-PTQ, run-QAT, or distill alternatives](/imgs/blogs/quantization-aware-training-qat-8.png)

### 7.1 The before-after table on a named target

Here is the result table you came for, on a named target — a MobileNetV2-class classifier, validation top-1, measured on a Pixel-class phone with the int8 model running on the QNNPACK/XNNPACK delegate at batch=1. Latency is p50 over 200 warm runs after a 50-run warm-up (warm-up matters: the first runs pay JIT and cache costs and thermal state drifts, so cold numbers lie); size is the serialized model on disk.

| Variant | Top-1 acc | vs fp32 | Size (MB) | p50 latency (ms) | Cost to produce |
| --- | --- | --- | --- | --- | --- |
| fp32 baseline | 76.1% | — | 14.0 | 31 | (already trained) |
| PTQ int8 | 73.0% | −3.1 | 3.6 | 12 | ~90 s calibration |
| QAT int8 | 75.8% | −0.3 | 3.6 | 12 | ~5 epochs fine-tune |
| PTQ int4 (per-ch) | 61.4% | −14.7 | 2.0 | 11 | ~2 min calibration |
| LSQ QAT int4 | 73.1% | −3.0 | 2.0 | 11 | ~8 epochs fine-tune |

Read the columns together and the story is unambiguous. Quantization (either way) buys you ~4× smaller and ~2.6× faster at int8 — that is the quantization lever doing its job, identical for PTQ and QAT. What QAT adds is the accuracy column: it converts the unacceptable −3.1 into a shippable −0.3 *at the same size and latency*. At int4 the size shrinks further (~7× vs fp32) and the speed edges up, but now PTQ is simply broken (−14.7 points) and only QAT keeps the model usable. The latency does not change between PTQ and QAT because, again, the deployed graph is identical — QAT changes the *values* in the int8 tensors, not the operators. Notice int4 latency barely beats int8 here: on this CPU the kernels are int8-native and int4 weights are unpacked to int8 before the matmul, so int4's win is *size/memory*, not compute — a roofline detail worth keeping in mind and one the roofline post in this series treats in full.

### 7.2 The escalation table you can act on Monday

Distilling the whole decision into a table you can paste into a design doc. The left column is the situation; the middle is the recommended move; the right is the one-line reason.

| Situation | Reach for | Why |
| --- | --- | --- |
| PTQ int8 already within budget | Ship PTQ | QAT recovers accuracy you already have — pure waste |
| PTQ int8 drop slightly over budget | PTQ tricks first (per-channel, MSE clip, mixed-precision) | Often closes the gap in minutes, no training |
| PTQ drop still over budget after tricks | QAT int8 | Recovers most of the gap at the same size and latency |
| Need int4 (size/flash budget) | LSQ QAT | int4 PTQ collapses; learned scale is what makes it work |
| Need ternary / binary | KD + QAT | No PTQ exists at 2 bits; network must be trained low-bit |
| Small / PTQ-hostile model (MobileNet, tiny BERT) | QAT (and try KD) | Relative quantization damage is largest here |
| No labeled data or no trainer | Distill or mixed-precision PTQ | QAT is structurally impossible without data + training |
| Latency too slow (not accuracy) | Fix kernel / delegate / op support | QAT does not change the deployed int8 graph |
| Giant LLM, PTQ (GPTQ/AWQ) within budget | Ship PTQ | Full QAT compute rarely worth the marginal points |

The shape of the table is the lesson: almost every row that says "QAT" is a row where either PTQ has failed your budget or you need to go below the bit-width PTQ can survive. QAT is never the *first* move and rarely the *cheapest* — it is the move you make when the cheap moves have run out and accuracy or bit-width still matters.

## 8. Composing QAT with the other levers

QAT is one move in a larger optimization. It composes — sometimes beautifully, sometimes with caveats — with the other three levers in this series, and knowing the composition order keeps the wins from cancelling.

**QAT + knowledge distillation (KD+QAT).** This is the highest-leverage combination and worth its own note. Run QAT with a *distillation loss*: the student is the int8 fake-quant model, the teacher is the full-precision (often larger) model, and you train the student to match the teacher's soft logits in addition to (or instead of) the hard labels. The teacher's soft targets carry far more information per example than one-hot labels — relative class probabilities, the "dark knowledge" — which is exactly the extra signal a quantized student needs to recover. KD+QAT routinely closes the last fraction of a point that plain QAT leaves on the table, and at int4/int2 it is close to mandatory; it is the recipe behind many of the strongest low-bit results in the literature. The loss is just $L = \alpha\, L_{\text{CE}}(\text{student}, y) + (1-\alpha)\,T^2\, \mathrm{KL}(\text{soft}_T(\text{teacher}) \,\|\, \text{soft}_T(\text{student}))$ with temperature $T$ — slot it into the section-4 loop in place of plain cross-entropy and you have KD+QAT. In code it is a small change to the training step:

```python
import torch.nn.functional as F

def kd_qat_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    # Hard-label term on the int8 fake-quant student.
    ce = F.cross_entropy(student_logits, labels)
    # Soft-target term: match the teacher's softened distribution.
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    kd = F.kl_div(s, t, reduction="batchmean") * (T * T)   # T^2 keeps grad scale
    return alpha * ce + (1.0 - alpha) * kd

teacher = build_fp32_teacher().eval()      # full-precision, frozen
for images, labels in train_loader:
    optimizer.zero_grad()
    with torch.no_grad():
        t_logits = teacher(images)         # teacher never updates
    s_logits = model(images)               # int8 fake-quant student, STE active
    loss = kd_qat_loss(s_logits, t_logits, labels)
    loss.backward()
    optimizer.step()
```

The teacher is frozen and runs in `no_grad`; only the int8 student trains, and the STE still carries its gradient through the fake-quant nodes exactly as before. The $T^2$ factor on the KD term is not cosmetic — softening the logits by $T$ shrinks their gradients by $1/T^2$, so multiplying back by $T^2$ keeps the soft-target gradient on the same scale as the hard-label one and lets a single learning rate serve both. The distillation post in this series goes deep on the loss; here the point is just that it stacks cleanly onto QAT and is the first thing to try when QAT alone falls short. On the int4 MobileNet from earlier, adding KD against the fp32 teacher typically buys another half-point to a point on top of plain int4 QAT — exactly the margin that turns "almost shippable" into "ship it."

**QAT + pruning.** These compose but the *order* matters. The standard recipe is prune first (to a target sparsity), fine-tune to recover, *then* QAT the pruned model — or, better, do the final QAT fine-tune on the already-pruned weights so the surviving weights co-adapt to both the sparsity pattern and the quantization grid at once. Doing it the other way (QAT then prune) wastes the QAT fine-tune, because pruning afterward perturbs exactly the weights you just carefully tuned to the grid. The pruning post (and the sub-8-bit post) in this series treat the joint recipe; the rule to remember is that QAT belongs *late* in the compression pipeline, after the structural changes are settled, because it is the lever that fine-tunes the final numeric representation.

**QAT + lower bit-width.** As the int4 example showed, QAT is what makes aggressive bit-widths viable at all. The sub-8-bit post (int4, ternary, binary networks) leans entirely on QAT and its descendants — at two bits and below, there is no PTQ; the network *must* be trained with the quantization in the loop, often with the KD+QAT combination above, because the rounding error is too large for the STE's smoothness assumption to hold without the network actively reshaping its distribution. QAT is, in that sense, the foundational technique the whole low-bit literature is built on.

## 9. Pitfalls, in order of how often they bite

I have shipped enough QAT to have a ranked list of what goes wrong. None of these are exotic; they are the boring failures that cost a day each.

**1. Learning rate too high.** This is number one by a wide margin. QAT is a *fine-tune* of an already-converged model, not a fresh train. Use roughly $\frac{1}{100}$ to $\frac{1}{10}$ of the original training LR. A too-high LR throws the converged weights off the good minimum, and now you are training a quantized model from a worse starting point than fp32 — the STE bias plus a large LR can actively *destroy* accuracy below the PTQ baseline. If your QAT run is *worse* than PTQ, the LR is the first suspect, every time. Start at $1/100$ and only raise it if learning is too slow.

**2. Observer and BN freeze timing.** Covered above but it earns repetition because it is the most *insidious* failure — the model looks fine in training and degrades only after convert, so you do not notice until the deployment eval. Freeze observers a couple of epochs before the end; freeze BN stats an epoch after that; verify that post-convert accuracy matches the last fake-quant eval to within hundredths. If there is a gap, your scales or BN stats are not the ones you trained against.

**3. Overfitting the fine-tune set.** QAT often runs on a *subset* of the training data (the full set may be unavailable, or you just want it fast). A small fine-tune set plus too many epochs plus the small extra capacity-loss of quantization is a recipe for overfitting: train accuracy climbs, val accuracy stalls or drops. Keep the fine-tune short (the recovery happens in the first few epochs; more epochs rarely help and can hurt), use the same regularization (weight decay, augmentation) as the original training, and watch the *validation* curve, not the training one. If you only have a tiny fine-tune set, KD+QAT against the fp32 teacher is a strong defense because the teacher's soft targets regularize the student.

**4. Fusing the wrong modules (or not fusing).** If Conv+BN+ReLU is not fused before `prepare_qat`, the BN folding is wrong and you get the silent-after-convert drop. Fuse exactly the triples your model uses, and use the framework's fused QAT modules so the folding is the inference-correct one. A subtle variant of this bug: fusing modules that are *shared* (the same module called from two places) or fusing across a residual add — the fuser cannot always tell, and you can end up quantizing a tensor twice or with the wrong activation range. When in doubt, prefer the FX or PT2E flows, which trace the actual graph and fuse from data flow rather than from your string guesses; they catch sharing and branching that eager-mode fusion misses.

**5. Quantizing things you should not.** The first and last layers, and anything with a tiny tensor or a wildly heavy-tailed distribution, are often best left at higher precision (mixed precision). Forcing int8/int4 on a sensitive first conv or a final classifier can cost you points that a one-layer exception would have saved — and that exception is cheap. Profile per-layer sensitivity (PTQ each layer in isolation, measure the drop) and keep the worst one or two in higher precision.

**6. Assuming the latency improved.** QAT changes accuracy, not the int8 graph — so if your *PTQ* int8 was not actually faster on your target (because the NPU does not support an op and it fell back to fp32 on CPU, or the model is memory-bound and int8 weights did not help the bottleneck), QAT will not fix that either. Measure the int8 latency on the *actual device* before you invest in QAT; QAT is the wrong tool for a speed problem. The metrics-on-device and roofline posts in this series are about exactly this measurement discipline.

### 9.1 Stress-testing the recipe: what breaks at the edges

A recipe you cannot break is a recipe you do not understand, so push on it. *What happens when the calibration/fine-tune set is tiny?* If you have only a few hundred labeled examples, plain QAT will overfit them within an epoch and the int8 eval will lag the train accuracy badly. The fix is KD+QAT: the fp32 teacher provides a dense, per-example supervisory signal that does not depend on having many labels, so even a small unlabeled-or-thinly-labeled set yields useful soft targets. The teacher regularizes the student. *What happens at int4 on an NPU that only has int8 kernels?* You train a great int4 model and then discover the device unpacks int4 weights to int8 before the matmul, so you got the size win (half the weight memory) but not a compute win, and on a compute-bound layer the latency is unchanged. That is not a QAT failure — it is a hardware-support fact you must check first, and it is why the metrics-on-device discipline precedes the optimization, not follows it. *What happens when the model is memory-bound, not compute-bound?* Then int8 (or int4) weights help directly because the bottleneck is moving weights from memory, and quantization halves or quarters that traffic — this is the regime where low-bit quantization wins biggest, and QAT lets you go lower than PTQ could survive. *What happens when one layer is pathological?* Profile per-layer, keep that one layer in higher precision, and QAT the rest. The throughline of every stress case is the same: QAT is a numerics tool, and it only pays off when the *system* — hardware op support, the memory-vs-compute bound, the data you have — actually rewards better numerics. Diagnose the system first; reach for QAT second.

## 10. Case studies: QAT in the wild

**MobileNetV1/V2 on mobile (Jacob et al., 2018).** The paper that defined the modern integer-arithmetic QAT recipe (fake-quant, STE, BN folding, integer-only inference) showed MobileNets — the canonical PTQ-hostile architecture — recovering nearly all of their fp32 accuracy under QAT at int8 while PTQ lost several points. This is the result the running example in this post mirrors, and it is *the* reference for why QAT exists: the architectures we most want on phones are precisely the ones PTQ hurts most.

**LSQ on ImageNet ResNets (Esser et al., 2020).** LSQ reported 3-bit and 4-bit ResNet-18/50 within a fraction of a point of full precision by learning the step size — accuracy that fixed-scale QAT could not reach. It is the strongest single demonstration that *learning the quantization parameters* is worth as much as the bit-width itself at low precision, and it is why "use LSQ" is my default advice for sub-8-bit work.

**LLM QAT.** For large language models the economics flip: a full QAT run on a 7B+ model is expensive, so the field leans on data-driven PTQ (GPTQ, AWQ) for most int4 deployments. But QAT-style training *does* show up where accuracy is critical or bits are extreme — e.g. the BitNet line of work trains transformers with quantization in the loop down to ternary/1.58-bit weights, which is simply impossible with PTQ; the network has to be *born* low-bit. The takeaway is the same lesson scaled up: the lower the bit-width, the more you need the quantization in the training loop rather than bolted on after.

**DistilBERT-class encoders on int8.** Small transformer encoders quantized to int8 for on-device NLP (intent classification, on-keyboard suggestion) are another sweet spot. PTQ on these often loses a point or two because attention activations and LayerNorm outputs have heavy tails that a static scale clips poorly; QAT with learned activation clips (PACT-style) typically recovers to within a few tenths while keeping the ~4× size win, which is the difference between a model that fits a phone's memory budget and one that does not. The pattern holds across modalities: the more a layer's distribution fights a static grid, the more QAT's learned, co-adapted representation pays off.

These four sit at different scales — a phone CNN, an ImageNet benchmark, a billion-parameter LLM, an on-device encoder — and tell one story. QAT's value rises with how aggressive the quantization is and how unfriendly the architecture is to rounding. Where PTQ suffices, QAT is wasted effort; where PTQ breaks, QAT is often the only road.

## 11. When to reach for QAT (and when not to)

Decisively, because every lever is a cost:

**Reach for QAT when** PTQ's accuracy drop exceeds your budget after the cheap PTQ tricks are exhausted; when you need to go below 8 bits (int4, ternary, binary), where PTQ degrades fast; when the model is small and PTQ-hostile (MobileNets, tiny transformers, depthwise-separable nets) so the relative damage is large; or when accuracy is genuinely safety- or revenue-critical and you have the data and compute to spend. In all of these, the training-run cost is justified by accuracy you cannot otherwise get.

**Do not reach for QAT when** PTQ already meets your budget — running QAT to recover accuracy you already have is pure waste. Do not reach for it when you lack labeled data or a training pipeline — it is structurally impossible, so distill or use mixed-precision PTQ instead. Do not reach for it to solve a *latency* problem — QAT does not change the deployed int8 graph; if your int8 is not faster on the device, fix the kernel/delegate/op-support issue, not the weights. And do not reach for full QAT on a giant LLM when a good PTQ method (GPTQ/AWQ) already lands within budget at int4 — the compute is not worth the marginal points.

The map for the whole quantization lever, then: PTQ is the default and the floor; QAT is the escalation that recovers accuracy and unlocks low bit-widths at the cost of a fine-tune; KD+QAT is QAT with a teacher for the last fraction of a point and for the extreme low-bit regime; LSQ is the learned-scale upgrade that makes sub-8-bit QAT actually work. You almost always start at PTQ and climb only as far as your budget forces you.

If you take one operating principle from this post, make it this: **QAT is a refinement, never a search.** Every piece of guidance here is a corollary. Start from the converged fp32 checkpoint, not from scratch — because the technique refines a good solution. Use a learning rate a hundredth of the original — because you are nudging, not relocating wholesale. Initialize the learned scales from calibration — because LSQ refines the PTQ scale rather than discovering one. Freeze observers and BN late — because you are settling into a fixed grid, not still exploring. Keep the fine-tune short — because the recovery happens fast and extra epochs only invite overfitting. When a QAT run misbehaves, the diagnosis is almost always that something turned the gentle refinement back into a destabilizing search: the LR was too high, the start point was wrong, the scales were initialized badly. Hold the refinement frame and the whole recipe becomes obvious instead of a list of magic numbers.

## Key takeaways

- **QAT trades a training run for accuracy, holding size and latency fixed.** It does not make the model smaller or faster than PTQ at the same bit-width — it makes the same-size, same-speed int8 model *more accurate*. If only your accuracy is short, QAT is the lever; if you need smaller/faster, that is a different lever.
- **The mechanism is fake-quant forward, STE backward, fp32 master weights.** The forward pass rounds to the int8 grid so the loss reflects real quantization error; the straight-through estimator passes the gradient across the non-differentiable round as the identity (zero where clipped); and you keep and update fp32 master weights from which the integers are derived.
- **The STE is biased and that is usually fine.** It is a first-order estimator of the gradient of the noise-smoothed loss; the bias is small when the loss is smooth at the scale of one quantization step, and it grows at very low bit-widths — which is why int4/int2 need LSQ and often distillation.
- **Learn the step size (LSQ).** Promoting the scale to a trained parameter with the magnitude-matched gradient gain $g = 1/\sqrt{N_W Q_p}$ co-adapts the weights and the grid; it is the single biggest QAT upgrade and is close to mandatory below 8 bits.
- **The freezes are not optional.** Freeze observers late, freeze BN stats later, fuse Conv+BN+ReLU before preparing — skip these and you get the silent "fine in training, worse after convert" drop. Post-convert accuracy must match the last fake-quant eval to within hundredths.
- **Use a tiny learning rate.** QAT is a fine-tune: roughly $1/100$ of the original LR, a handful of epochs. A too-high LR is the number-one reason a QAT run ends up *worse* than PTQ.
- **PTQ first, always; QAT only as escalation.** Try the cheap rung — minutes, unlabeled data, no trainer. Climb to QAT only when the drop exceeds budget, you need sub-8-bit, or the model is PTQ-hostile, and only if you have labeled data and compute.
- **QAT's advantage grows as bits shrink.** At int8 the PTQ-to-QAT gap is a couple of points; at int4 it is double digits; at ternary/binary, PTQ does not exist and QAT (usually with distillation) is the only way to train the model at all.

## Further reading

- **Jacob et al., 2018**, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR) — the foundational integer-only QAT recipe: fake-quant, STE, BN folding, the MobileNet results.
- **Esser et al., 2020**, "Learned Step Size Quantization" (ICLR) — LSQ: making the scale a trained parameter, the step-size gradient, and the gradient-scale trick.
- **Bhalgat et al., 2020**, "LSQ+: Improving Low-bit Quantization Through Learnable Offsets and Better Initialization" — adds a learnable zero-point for asymmetric activations.
- **Krishnamoorthi, 2018**, "Quantizing deep convolutional networks for efficient inference: A whitepaper" — the practical reference for PTQ vs QAT, per-channel quantization, and what actually moves accuracy.
- **Bengio, Léonard, Courville, 2013**, "Estimating or Propagating Gradients Through Stochastic Neurons" — the original straight-through estimator.
- **PyTorch quantization docs** — `torch.ao.quantization` (eager, FX, and PT2E flows), the QAT tutorial, and the ExecuTorch on-device path.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for where QAT sits among the four levers; [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) for the cheap rung you try first; [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles) for the affine map and SQNR math; [sub-8-bit: int4, ternary, and binary networks](/blog/machine-learning/edge-ai/sub-8-bit-int4-ternary-and-binary-networks) for where QAT becomes mandatory; and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for how it fits the end-to-end flow.

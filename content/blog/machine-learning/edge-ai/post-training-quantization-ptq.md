---
title: "Post-training quantization: calibration, per-channel, and the cheapest win"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A hands-on guide to turning a trained model into int8 with no retraining — how calibration estimates activation ranges, why per-channel weights are nearly free, and exactly when this is enough and when you need QAT."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "post-training-quantization",
    "calibration",
    "int8",
    "inference",
    "efficient-ml",
    "onnxruntime",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/post-training-quantization-ptq-1.png"
---

Here is the promise that makes post-training quantization the first lever you should ever reach for: you take a model you already trained, run a few hundred unlabeled examples through it, type one conversion command, and ten minutes later you have a model that is four times smaller and two to four times faster, with — on a good day — an accuracy drop you would struggle to measure. No gradients. No retraining job. No GPU cluster booked for the weekend. The training is *done*; PTQ just changes how the finished weights and activations are stored and computed.

That is not a marketing line, it is the actual workflow, and I have shipped it more times than any other optimization. When a product manager asks on a Friday whether the new classifier can fit in the app's size budget and run under the latency SLA, PTQ is the answer you can have working before you leave for the day. It is the cheapest win in the whole compression toolbox precisely because it asks for almost nothing: a trained checkpoint, a representative slice of data, and a runtime that has int8 kernels.

It is worth being precise about *why* int8 is the sweet spot that makes this so cheap, because it is not arbitrary. A 32-bit float carries far more precision than a trained network's weights and activations actually need — most of those bits encode noise, not signal. Dropping to 8 bits keeps enough dynamic range and resolution that, for the overwhelming majority of layers, the model barely notices, while shrinking storage 4× and letting hardware do four int8 multiply-accumulates in the time of one fp32 one. The gap between "how much precision the math has" and "how much the model needs" is the free lunch PTQ collects. The whole technique is an exercise in collecting exactly that gap and no more — taking the precision the model was not using and handing it back as size and speed.

Here is the catch, stated up front so you are never surprised by it. *Sometimes it is not enough.* You convert, you measure, and the model has lost four points of top-1 accuracy, or your detector's mAP fell off a cliff, or the LLM started producing garbage on long contexts. When that happens, no amount of fiddling with the calibration set will fully recover it, and you graduate to quantization-aware training — which costs a real fine-tuning run but lets the model *learn* to be robust to the rounding. The whole skill of PTQ is two halves: squeezing every last drop of accuracy out of the free path, and recognizing — quickly, with evidence — the moment the free path runs out. We will do both. The companion post on [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) picks up exactly where this one's failure cases end.

Figure 1 is the entire workflow on one slide — load the trained model, insert observers, run the calibration loop, convert. Keep it in mind as the spine of everything below; every section is really just a detail of one of those four boxes.

![A vertical stack diagram showing the four post-training quantization steps from loading a trained model through inserting observers, calibrating, and converting to a deployed int8 model](/imgs/blogs/post-training-quantization-ptq-1.png)

This post assumes you already understand *what* int8 quantization is — the affine map from floats to integers, the scale and zero-point, the SQNR-versus-bit-width law. If those are fuzzy, read [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles) first; it derives the numerics this post takes as given. Here we are entirely concerned with the *post-training* recipe: how to estimate the ranges well, where per-channel and static-versus-dynamic decisions come from, how to do it in PyTorch and ONNX Runtime and TFLite, how many calibration samples you need, and how to debug an accuracy drop. PTQ is one specific way of pulling the quantization lever from the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) — the no-retraining corner of it — and it is the one you should master first because every harder technique is measured against "but did you try plain PTQ?"

## 1. The one thing calibration actually does

Quantizing a weight is easy, almost boring. The weights are sitting right there in the checkpoint as a fixed tensor of numbers. You look at them, find their range, pick a scale, and round. Nothing about the input changes that. Weights are *static*.

Activations are the hard part, and they are the entire reason calibration exists. An activation tensor — the output of a layer for a given input — does not exist until you run data through the model. Its range depends on what you feed in. To quantize it to int8 you need to know its range *ahead of time* so you can bake a fixed scale into the graph, and the only honest way to learn that range is to run representative data through the model and watch.

That watching is calibration. Concretely: you insert small recording devices, called **observers**, at every point where an activation will be quantized — typically the output of each layer. Then you push a few hundred representative inputs through the model in inference mode. Each observer accumulates statistics about the tensor that flows past it: at minimum the running minimum and maximum, often a full histogram. After the loop, each observer hands back an estimate of "this is the range of values this activation takes in practice," and from that range the converter computes the scale $s$ and zero-point $z$ that will be frozen into the int8 graph.

So calibration is, in one sentence, **estimating activation ranges from a representative sample of inputs.** Everything difficult about PTQ is downstream of that one estimation problem. How do you turn the observed values into a range — the raw min and max, or something cleverer? How many samples do you need before the estimate stops moving? What happens when your sample is not actually representative? Get the range estimation right and a good fraction of models quantize to int8 for free. Get it wrong and you leave accuracy on the table that QAT then has to claw back.

It helps to be precise about the affine quantization map calibration is feeding, because every range method is just a different way of choosing two numbers in it. For an asymmetric int8 quantizer over an observed range $[\beta_{\min}, \beta_{\max}]$, the scale and zero-point are

$$
s = \frac{\beta_{\max} - \beta_{\min}}{2^b - 1}, \qquad z = \mathrm{round}\!\left(-\frac{\beta_{\min}}{s}\right),
$$

with $b = 8$ bits, so $2^b - 1 = 255$ levels. The quantize and dequantize operations are

$$
q(x) = \mathrm{clip}\!\big(\mathrm{round}(x/s) + z,\; 0,\; 255\big), \qquad \hat{x} = s\,(q(x) - z).
$$

For symmetric quantization (the usual choice for weights) you drop the zero-point, set $\beta = \max(|x|)$, and use $s = \beta / (2^{b-1} - 1) = \beta/127$ over the signed range $[-127, 127]$. Calibration's whole job is to deliver $\beta_{\min}$ and $\beta_{\max}$ (or $\beta$) per activation. The methods differ only in how they read those two numbers off the observed data.

## 2. The science: clipping error versus quantization error

Here is the central tension of calibration, and once you see it, every range-estimation method becomes obvious as a different answer to the same question. When you choose the range $[\beta_{\min}, \beta_{\max}]$, you are making a trade-off you cannot escape.

If you set the range to the true full min and max of the data — never clip anything — then no value is ever distorted by being shoved outside the representable range. But your step size $s$ is determined by the most extreme value you ever saw, including a single freak outlier. One activation of magnitude 50 when 99.9% of the mass lives in $[-3, 3]$ will stretch $s$ enormously, and now every ordinary value in the bulk of the distribution is being rounded to a coarse grid. You have spent your precious 256 levels covering a range that is almost entirely empty. This is **quantization error** dominating: the step is too big for the values that matter.

If instead you clip the range tight — say, to the 99.9th percentile — then the bulk of the distribution gets a fine step size, because $s$ is small. But every value beyond the clip threshold is now slammed to the boundary; the outlier of magnitude 50 becomes 3. That distortion is **clipping error**: the tails are destroyed.

Figure 2 shows the two regimes as a before-and-after. The loose range protects the outliers but coarsens the bulk; the clipped range sharpens the bulk but sacrifices the tails. The art is finding the clip point that minimizes the *sum* of the two errors.

![A before-and-after diagram contrasting a loose full-range quantizer that coarsens the bulk with a clipped range that loses the tails but sharpens the common values](/imgs/blogs/post-training-quantization-ptq-2.png)

We can make this a genuine bias-variance-style trade-off, because that is exactly what it is. Model a quantized activation $\hat{x}$ as the original $x$ plus two error sources. Inside the kept range, the rounding error is well approximated as uniform noise on $[-s/2, s/2]$, which has variance

$$
\sigma_q^2 = \frac{s^2}{12} = \frac{(\beta_{\max} - \beta_{\min})^2}{12\,(2^b-1)^2}.
$$

This is the **quantization-error** term, and it *grows with the square of the range you keep*. Widen the range to admit an outlier and you pay for it quadratically across every single value.

The **clipping-error** term comes from the mass you threw away. If $p(x)$ is the activation density and we clip everything beyond a threshold $t$, the expected squared distortion contributed by the clipped tails is

$$
\sigma_c^2 = \int_{|x| > t} (|x| - t)^2\, p(x)\, dx.
$$

This term *shrinks* as you widen $t$ (you clip less mass), and it *grows* as you clip tighter. So the total expected MSE,

$$
\text{MSE}(t) = \underbrace{\frac{(2t)^2}{12\,(2^b-1)^2}}_{\text{grows with } t} \;+\; \underbrace{\int_{|x|>t}(|x|-t)^2 p(x)\,dx}_{\text{shrinks with } t},
$$

is a convex U with a clear minimum. Differentiate, set to zero, and you get the MSE-optimal clip threshold. For a Gaussian or Laplacian activation the optimum is a small multiple of the standard deviation — for int8 it lands around $3\sigma$ to $4\sigma$, *not* at the true max, which is the whole reason min/max calibration is so often suboptimal. This is precisely the analysis Banner et al. carried out in ACIQ (2019): they derived closed-form near-optimal clipping values for Gaussian and Laplacian activations and showed it beats plain min/max by a wide margin at low bit-widths. The lesson to carry: **the best range is almost never the true range.**

It is worth pausing on *why* the clipping term shrinks the way it does, because it explains why min/max gets dramatically worse as bit-width drops. The quantization-error term carries a factor of $1/(2^b-1)^2$ — halve the number of levels (drop one bit) and the quantization variance quadruples for the same range. So at int8 with 255 levels, the range you pick matters but a loose range is survivable; at int4 with 15 levels, that same loose range is catastrophic, because the step size is already 17× coarser before you even admit an outlier. This is the formal reason PTQ degrades gracefully at int8 and falls off a cliff at int4: the clipping/quantization trade-off has far less slack to work with when there are only fifteen levels to allocate. We will stress-test exactly this in a later section.

There is a connection here to the SQNR law that the [first-principles post](/blog/machine-learning/edge-ai/quantization-from-first-principles) derives. Signal-to-quantization-noise ratio for a uniform quantizer over a well-matched range is approximately $\text{SQNR} \approx 6.02\,b + 1.76$ dB — about 6 dB, or one bit of effective precision, per actual bit. But that clean law *assumes the range matches the signal*. A loose range from a single outlier effectively throws away bits: if your $\beta_{\max}$ is 16× larger than the bulk of the signal needs, you have wasted four bits, and your int8 quantizer behaves like an int4 one on the values that matter. Calibration, properly done, is what lets you actually *get* the $6.02\,b + 1.76$ dB the bit-width promises instead of a fraction of it. That reframing is useful: a bad calibration does not just lose accuracy in some vague way; it provably costs you effective bits.

### Symmetric versus asymmetric, and why it matters for hardware

One more knob hides inside the affine map: whether to use a zero-point at all. **Symmetric** quantization fixes $z = 0$ and uses a single magnitude $\beta = \max(|x|)$, so the integer grid is centered on zero. **Asymmetric** (affine) quantization lets the range be off-center with a nonzero $z$, so $[\beta_{\min}, \beta_{\max}]$ need not be symmetric about zero.

Asymmetric is strictly more expressive — it can fit a range like $[0, 6]$ (the output of a ReLU, which is never negative) without wasting half its levels on negative values that never occur. That sounds like a clear win, and for *activations after ReLU* it often is, which is why activation quantizers are commonly asymmetric (unsigned int8, `quint8`). But asymmetric quantization has a hardware cost in the matmul: the zero-point introduces cross-terms. When you multiply two asymmetric-quantized tensors, $\hat{x}\hat{w} = s_x s_w (q_x - z_x)(q_w - z_w)$, expanding the product gives a main term plus three correction terms involving the zero-points that the kernel must compute and add back. For *weights*, those cross-terms are annoying enough that the near-universal choice is **symmetric weights** ($z_w = 0$), which kills two of the three correction terms and keeps the integer matmul clean. So the standard, hardware-friendly recipe pairs **symmetric per-channel weights** with **asymmetric per-tensor activations** — symmetric where the cross-terms would hurt the kernel, asymmetric where the extra expressiveness is free to consume. When you see `qint8` weights and `quint8` activations in a `QConfig`, that is exactly this split.

#### Worked example: the cost of one outlier

Take a real activation distribution: 99.9% of values in $[-3, 3]$, one outlier at 50. Quantize to int8 (255 levels).

With **min/max** calibration, $\beta_{\max} = 50$, so $s = 100/255 \approx 0.392$. Every ordinary value in $[-3, 3]$ now rounds to a grid spaced 0.392 apart — only about $6/0.392 \approx 15$ distinct levels span the entire bulk of your data. You bought 255 levels and used 15 of them.

With **percentile** clipping at $t = 3.5$, $s = 7/255 \approx 0.0275$, giving roughly $6/0.0275 \approx 218$ levels across the bulk. The outlier at 50 gets clipped to 3.5 — one value distorted badly — but every other value is represented about $14\times$ more finely. The quantization-error variance dropped by a factor of $(50/3.5)^2 \approx 204$ on the bulk, in exchange for clipping a single sample. That is the trade in numbers, and it is why percentile and MSE clipping exist.

## 3. The range-estimation methods, ranked

Now the menu. Every PTQ toolchain ships several of these, and choosing among them is the single highest-leverage calibration decision you make. Figure 3 lays them out as a comparison matrix; I will walk each one.

![A comparison matrix scoring min-max, percentile, entropy KL, and MSE-optimal calibration on outlier robustness, compute cost, and calibration samples needed](/imgs/blogs/post-training-quantization-ptq-3.png)

**Min/max.** Track the running minimum and maximum across the calibration set; use them directly as the range. Trivially cheap, needs few samples, and for well-behaved weights it is often fine. For activations it is fragile: a single outlier in your calibration data sets the range for the entire deployment. Use it as a baseline and a sanity check, rarely as the final answer for activations. A common refinement is the **moving-average min/max** observer (PyTorch's `MovingAverageMinMaxObserver`), which exponentially averages the per-batch extremes so one freak batch cannot dominate — a cheap robustness upgrade over raw min/max.

**Percentile clipping.** Instead of the true max, take the 99.9th or 99.99th percentile as the clip threshold. This directly attacks the outlier problem: by construction you discard a fixed small fraction of the most extreme mass. Cheap to compute (a sort or a histogram), robust, and a great default. The only knob is the percentile, and 99.9–99.99 covers most cases. Push it tighter and you start clipping real signal; looser and you drift back toward min/max.

**Entropy / KL-divergence.** This is the method TensorRT's int8 calibrator made famous, introduced by Szymon Migacz in his 2017 GTC talk. The idea is elegant: treat the fp32 activation distribution as a reference, and choose the clip threshold that minimizes the Kullback-Leibler divergence between the fp32 distribution and the int8-quantized one. Concretely, the calibrator builds a fine histogram (e.g. 2048 bins) of each activation over the calibration set, then sweeps candidate clip thresholds; for each, it quantizes the reference histogram into 128 bins, expands it back, and measures the KL divergence

$$
D_{\mathrm{KL}}(P \,\|\, Q) = \sum_i P_i \log \frac{P_i}{Q_i}
$$

between the original distribution $P$ and the requantized one $Q$. The threshold with the lowest divergence is chosen — it is the range that preserves the *information* in the activation distribution best, rather than its raw extent. It needs more samples (to fill the histogram) and more compute (the sweep), but it is robust and principled, and it is the reason a TensorRT int8 engine built with a good calibration set so often "just works."

**MSE-optimal clipping.** Directly minimize the total mean-squared error from the previous section — sweep the clip threshold (often via a small grid search per tensor) and pick the one that minimizes the reconstruction MSE between fp32 and dequantized int8. This is what the bias-variance derivation pointed at, and on many models it is marginally the best of the four. PyTorch exposes it as the `MSEObserver` / `HistogramObserver` with an MSE criterion. Cost is moderate (the grid search), robustness is high.

| Method | What it optimizes | Outlier robust | Compute | Samples | When to use |
| --- | --- | --- | --- | --- | --- |
| Min/max | Raw extent | No | Trivial | ~100 | Weights, sanity baseline |
| Moving-avg min/max | Smoothed extent | Somewhat | Trivial | ~200 | Activations, quick first pass |
| Percentile 99.9–99.99 | Fixed tail fraction | Yes | Sort/bin | ~500 | Strong default for activations |
| Entropy / KL | Distribution fidelity | Yes | Histogram + sweep | ~1000 | TensorRT, robust production |
| MSE-optimal | Reconstruction error | Yes | Grid search | ~500 | Squeeze the last fraction of a point |

The honest default: **per-channel symmetric for weights, percentile or entropy for activations.** Start there, measure, and only escalate to MSE-optimal or QAT if the measurement says you must.

A practical note on how these methods actually behave when you run them, because the table flattens an important nuance. Min/max and moving-average min/max are *online* — they update a running statistic per batch and need only one pass. Percentile, entropy, and MSE are effectively *two-phase*: phase one accumulates a histogram of every value seen across the calibration set, phase two sweeps clip thresholds against that histogram to pick the best one. That two-phase structure is why histogram methods want more samples (a sparse histogram gives a noisy threshold) and why they cost more compute (the post-hoc sweep). It also explains a debugging gotcha: if you accidentally calibrate with too few samples on a histogram observer, it does not error — it silently picks a threshold from a half-empty histogram and hands you a quietly worse range. The symptom is "entropy calibration did worse than min/max," which should never happen with enough data and is a tell that your calibration set is too small for the histogram to fill.

There is also a quiet interaction between method and granularity worth flagging. The range-method debate (min/max vs percentile vs entropy vs MSE) is mostly an *activation* debate, because activations are per-tensor and a single bad scale hurts the whole tensor. Weights, being per-channel, are far more forgiving of the method — each channel gets its own tight range regardless, so min/max on per-channel weights is usually fine and the fancy methods buy little there. So in practice you spend your method-selection effort on the activation observer and leave weights on per-channel min/max. This is why the default `QConfig` in most toolchains pairs a histogram *activation* observer with a plain per-channel min/max *weight* observer — the asymmetry is deliberate.

## 4. Per-tensor versus per-channel: the nearly-free win

There is a second axis to the range question that is almost as important as the method, and it is the single best accuracy-per-effort move in all of PTQ: **the granularity of the scale.**

A *per-tensor* quantizer uses one scale (and one zero-point) for an entire tensor. A *per-channel* quantizer uses a separate scale for each channel — for a convolution weight of shape `[out_channels, in_channels, kh, kw]`, that means one scale per output channel, i.e. a small vector of `out_channels` scales instead of a single scalar.

Why does this matter so much? Because different output channels of a trained layer often have wildly different weight magnitudes. One channel might have weights in $[-0.02, 0.02]$ while another spans $[-0.8, 0.8]$ — a 40× difference. With a single per-tensor scale, the widest channel sets the step size for *all* of them, so the narrow channel's weights are quantized with a step 40× too coarse for their range, and most of their bits are wasted. Per-channel quantization gives each channel its own scale, so each one gets the full 256 levels across its actual range.

Figure 4 shows the contrast directly. The cost of per-channel is a vector of scales instead of a scalar — a few kilobytes — and on modern hardware it is fully supported for weights at essentially no runtime cost, because the per-channel scale folds into the same dequantize step. The accuracy recovery is large: on a depthwise-separable CNN like MobileNetV2, per-tensor int8 can lose several points of top-1, while per-channel int8 brings it back to within a fraction of a point of fp32. This is the headline result of Krishnamoorthi's 2018 PTQ study and of the Nagel et al. white paper: **for weights, per-channel is nearly free and you should almost always use it.**

![A before-and-after diagram showing per-tensor weights letting the widest channel coarsen all others versus per-channel weights giving each output channel its own scale at near-zero cost](/imgs/blogs/post-training-quantization-ptq-4.png)

The asymmetry is worth stating clearly. **Per-channel is standard for weights, but usually per-tensor for activations.** The reason is hardware: an activation tensor flows between layers, and a per-channel activation scale would force the matmul/conv kernel to apply a different scale per channel mid-computation, which most int8 kernels do not support efficiently (it breaks the clean integer accumulation). Weights are different — their per-output-channel scale folds into the output dequantization cleanly, after the integer accumulation is done. So the standard recipe is per-channel weights, per-tensor activations. (Per-channel or per-token *activation* quantization does exist and matters enormously for LLMs — SmoothQuant and per-token dynamic schemes are exactly about this — but for CNNs the per-channel-weights / per-tensor-activations split is the workhorse.)

#### Worked example: per-tensor versus per-channel on the same model

Quantize a MobileNetV2 (ImageNet, fp32 top-1 ≈ 71.9%) to int8 with entropy calibration, changing only the weight granularity.

- **Per-tensor weights:** top-1 ≈ 65.1%. That is a 6.8-point drop — unusable for most products. MobileNet's depthwise convolutions are notorious here: each depthwise filter is one channel, and their magnitudes vary enormously, so one shared scale is catastrophic.
- **Per-channel weights:** top-1 ≈ 71.1%. A 0.8-point drop — shippable. Same calibration, same activations, same everything; the *only* change was giving each output channel its own weight scale, a vector of ~1000 fp16 numbers adding under 2 KB to the model.

That is the entire argument for per-channel weights in one table. It is the cheapest accuracy you will ever buy, and forgetting to turn it on is the single most common reason a first PTQ attempt looks like a disaster. If your int8 model lost five points, check this *before* anything else.

## 5. Static, dynamic, and weight-only

The third axis is *what* you quantize and *when* you compute the activation ranges. There are three modes you will meet constantly, shown in Figure 5.

![A comparison matrix of weight-only, dynamic, and static int8 quantization across calibration needs, activation math, and best target hardware](/imgs/blogs/post-training-quantization-ptq-5.png)

**Weight-only quantization.** Quantize only the weights to int8 (or int4); keep activations in floating point. At inference, weights are dequantized on the fly (or the kernel does mixed int-weight/float-activation math). The huge advantage: **no calibration at all** — weights are static, so there is nothing to estimate from data. You get the 4× model-size reduction (weights dominate size in most models) and, when memory bandwidth is the bottleneck, a real speedup from moving fewer weight bytes. This is the dominant mode for LLMs — GPTQ, AWQ, bitsandbytes NF4, and llama.cpp's k-quants are all weight-only — because LLM decoding is memory-bound on the weights, so shrinking the weights *is* the speedup. The cost is that the compute stays in float, so on compute-bound workloads you get size but not much speed.

**Dynamic quantization.** Quantize weights offline, and quantize activations *at runtime*: just before each int8 op, compute the activation's min/max on the spot, derive a scale, quantize, do the int8 matmul, dequantize. Also needs **no calibration** (the range is computed live), which makes it the easiest possible PTQ. The downside is the per-inference overhead of computing activation ranges, and that the activation range is recomputed every forward pass. It shines on CPU NLP models — dynamically quantized BERT/Transformer encoders on CPU are a classic win — where the matmuls are big enough that the int8 speedup dwarfs the range-computation overhead. PyTorch's `quantize_dynamic` is a one-liner for exactly this.

**Static quantization.** Quantize both weights and activations to int8, with activation ranges *frozen offline* via the calibration loop. This is the mode that needs calibration data, and it is the one most of this post is about. The payoff: the entire forward pass runs in int8 with no per-inference range computation and no float fallbacks, so it is the fastest mode and the only one that fully exploits dedicated int8 hardware — NPUs, DSPs, the int8 path on a TensorRT engine, the Edge TPU. If your target is an accelerator with an int8 datapath, static is what unlocks it.

| Mode | Calibration | Activations | Speedup source | Best target |
| --- | --- | --- | --- | --- |
| Weight-only | None | Stay float | Less weight bandwidth | LLMs, memory-bound |
| Dynamic | None | int8 computed at runtime | int8 matmul, range overhead | CPU NLP, BERT encoders |
| Static | 100–1000 batches | int8, frozen offline | Full int8 datapath | NPU, DSP, TensorRT, Edge TPU |

The decision rule is short. **On an int8 accelerator, do static.** **On CPU for transformers, dynamic is the easy 2× with zero calibration.** **For LLMs where weights dominate, weight-only.** Pick the mode from the hardware and the bottleneck, not from habit.

## 6. The calibration loop in PyTorch

Enough theory. Here is the actual flow in `torch.ao.quantization`, the FX-graph PTQ path, end to end. This is the static-quantization recipe; it is the one that earns its keep on int8 hardware.

```python
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# 1. Start from a TRAINED, eval-mode fp32 model.
model_fp32 = build_model()
model_fp32.load_state_dict(torch.load("resnet50_fp32.pth"))
model_fp32.eval()

# 2. Choose the QConfig. "x86" gives per-channel symmetric int8 weights
#    + per-tensor activations with a HistogramObserver (entropy/MSE-style).
qconfig = get_default_qconfig("x86")
qconfig_mapping = {"": qconfig}  # apply to the whole model

# 3. prepare_fx inserts observers at every quantize site.
example_inputs = (torch.randn(1, 3, 224, 224),)
model_prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs)

# 4. THE CALIBRATION LOOP: run representative data through, no grad,
#    no labels needed. Observers silently record activation ranges.
model_prepared.eval()
with torch.inference_mode():
    for images, _ in calibration_loader:   # ~256-1000 representative images
        model_prepared(images)

# 5. convert_fx folds the observed ranges into scales/zero-points
#    and swaps float ops for their int8 implementations.
model_int8 = convert_fx(model_prepared)

torch.save(model_int8.state_dict(), "resnet50_int8.pth")
```

The four boxes from Figure 1 map exactly onto this code: load (steps 1–2), insert observers (step 3, `prepare_fx`), calibrate (step 4, the loop), convert (step 5, `convert_fx`). Everything subtle lives in the `QConfig` and in the loop.

The `QConfig` is where every decision from the last four sections gets encoded. It is a pair of observer factories — one for activations, one for weights:

```python
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import (
    HistogramObserver,          # entropy/MSE-style activation calibration
    PerChannelMinMaxObserver,   # per-output-channel weights
    MovingAverageMinMaxObserver # cheap activation alternative
)

# Per-channel symmetric int8 weights, per-tensor activations via histogram.
custom_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    ),
    weight=PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    ),
)
```

Read that config against the science: `per_channel_symmetric` weights is the nearly-free win from section 4; `HistogramObserver` for activations is the entropy/MSE clipping from section 3; `per_tensor_affine` activations is the hardware-friendly choice from section 4. The whole post is compressed into those eight lines.

A few things about the calibration loop that bite people:

- **No labels, no gradients, no optimizer.** Calibration is pure forward passes in `inference_mode`. You are not training; you are watching. If you find yourself computing a loss, you have wandered into QAT.
- **Use real, in-distribution data.** The calibration loader must draw from the same distribution your model will see in production. More on this in the next section — it is the most common silent failure.
- **`eval()` mode matters.** BatchNorm and dropout must be in inference behavior, or the activation statistics you record will be wrong. With FX PTQ, BatchNorm is typically already folded into the preceding conv during preparation, which is what you want.
- **Batch size during calibration is about throughput, not correctness** — the observers accumulate per-element statistics regardless of how you batch. Use whatever fits in memory.

For the easiest possible mode — dynamic quantization on a CPU transformer — the whole thing collapses to one call, no calibration loop at all:

```python
import torch

# Dynamic int8: weights quantized offline, activations at runtime.
# No calibration data required. Great for CPU BERT/Transformer encoders.
model_dynamic = torch.ao.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},   # which layer types to quantize
    dtype=torch.qint8,
)
```

That single call is often a 2–4× CPU speedup on the linear layers of an encoder, with no calibration set and a near-zero accuracy hit. It is the literal cheapest win in this post, and a perfectly good first thing to try before you invest in static calibration.

### What convert actually does

It is worth being concrete about the `convert` step (box four), because "fold the ranges into the graph" hides several distinct transformations that, when one of them goes wrong, produce confusing symptoms. Convert does four things. First, it computes the final scale and zero-point for every quantize site from the observer's recorded range, using the affine formulas from section 1. Second, it replaces each float op with its quantized implementation — `nn.Conv2d` becomes a `quantized.Conv2d` that takes int8 inputs, does int32 accumulation, and produces int8 output. Third, it inserts the explicit quantize/dequantize boundary ops: a `quantize_per_tensor` where float data enters the int8 region and a `dequantize` where it leaves, so the graph is unambiguous about where the int8 datapath begins and ends. Fourth — and this is the one people forget — it *folds* operations that can be absorbed: BatchNorm into the preceding conv, the requantization scale into the conv's output, and (with bias correction on) the estimated quantization bias into the bias term.

Two things go wrong here in practice. If your model has float ops *between* two int8 regions, convert will insert dequantize→float-op→quantize around them, and those boundary conversions are pure overhead — a sign you should either quantize that op too or accept it as a deliberate fp16 island (which is exactly the mixed-precision tool from the debugging section). And if your activation observer never saw data for some branch (a code path your calibration set never exercised), that branch's range is undefined and convert may fall back to a degenerate scale — another reason your calibration set must exercise every path the model takes in production.

### A profiling snippet for the deployed int8 graph

Once converted, you must measure on the real target, and you must measure the *steady state*, not the cold start. Here is the minimal honest benchmark for batch=1 edge latency:

```python
import time, torch

model_int8.eval()
x = torch.randn(1, 3, 224, 224)

# Warm up: discard the first runs (kernel compilation, clock ramp, cache fill).
with torch.inference_mode():
    for _ in range(20):
        model_int8(x)

# Time the steady state. Collect per-inference latencies for a tail percentile.
latencies = []
with torch.inference_mode():
    for _ in range(200):
        t0 = time.perf_counter()
        model_int8(x)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

latencies.sort()
print(f"p50: {latencies[len(latencies)//2]:.2f} ms")
print(f"p99: {latencies[int(len(latencies)*0.99)]:.2f} ms")
```

The warm-up loop is not optional — without it the first measurements include one-time costs that have nothing to do with your model's steady-state speed, and you will report a latency that is both wrong and unreproducible. Sort and read a percentile rather than averaging, because the tail is what your SLA is written against and a mean hides it.

## 7. The same flow in ONNX Runtime and TFLite

If your deployment target is not a PyTorch runtime — a mobile app, a cross-platform service, a web target — you will quantize through ONNX Runtime or TFLite instead. The *concepts* are identical (the four boxes never change); only the API surface differs. Knowing two toolchains is worth it because the one your hardware vendor supports is not always the one you trained in.

### ONNX Runtime static PTQ

ONNX Runtime's `quantize_static` wants the same things: a model, a calibration data source, and a few flags encoding the decisions above. The calibration data arrives through a `CalibrationDataReader` — an iterator that yields input feed dicts, which is just the calibration loop turned inside out into a generator.

```python
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader,
    QuantType, QuantFormat, CalibrationMethod
)
import numpy as np

class ImageCalibrationReader(CalibrationDataReader):
    """Yields representative input feed dicts for calibration."""
    def __init__(self, image_arrays, input_name="input"):
        self.input_name = input_name
        self.data = iter([{input_name: x} for x in image_arrays])

    def get_next(self):
        return next(self.data, None)   # None signals end of calibration

calib_reader = ImageCalibrationReader(load_calib_images(n=512))

quantize_static(
    model_input="resnet50_fp32.onnx",
    model_output="resnet50_int8.onnx",
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,            # insert QuantizeLinear/Dequantize ops
    activation_type=QuantType.QInt8,         # int8 activations (static)
    weight_type=QuantType.QInt8,             # int8 weights
    per_channel=True,                        # the nearly-free win, section 4
    calibrate_method=CalibrationMethod.Entropy,  # KL/entropy, section 3
)
```

The flags map one-to-one onto the science: `per_channel=True` is section 4; `calibrate_method=CalibrationMethod.Entropy` (the alternatives are `MinMax` and `Percentile`) is section 3; `QuantFormat.QDQ` inserts explicit QuantizeLinear/DequantizeLinear nodes so a downstream compiler like TensorRT can fuse them into true int8 kernels. Switch `calibrate_method` between `MinMax`, `Percentile`, and `Entropy` and you can reproduce the accuracy table from section 9 yourself in an afternoon — it is the cleanest way to *see* the range-method effect on your own model.

### TFLite representative dataset

TFLite's converter takes the same idea and wears it as a `representative_dataset` — a generator that yields representative inputs, which is the calibration loop expressed as a Python generator. `Optimize.DEFAULT` turns on quantization; specifying the int8 inference types forces full integer quantization (the static mode) rather than the default float-fallback.

```python
import tensorflow as tf

def representative_dataset():
    # Yield ~100-500 representative inputs, one at a time, as the
    # calibration set. Must be in-distribution.
    for image in calibration_images[:500]:
        yield [tf.expand_dims(tf.cast(image, tf.float32), 0)]

converter = tf.lite.TFLiteConverter.from_saved_model("mobilenet_fp32")
converter.optimizations = [tf.lite.Optimize.DEFAULT]      # turn on quantization
converter.representative_dataset = representative_dataset  # the calibration set
# Force FULL integer quantization (static int8 in + out), no float fallback:
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
open("mobilenet_int8.tflite", "wb").write(tflite_int8)
```

Three toolchains, one mental model. In every case you provide a trained model and a representative data source, and the tool inserts observers, runs your data, and folds the ranges into an int8 graph. If you internalize the four boxes, switching toolchains is just looking up the new names for "calibration data" and "per-channel."

Figure 6 shows what is happening inside that loop regardless of toolchain — data flowing through the observer-instrumented model, activation observers and weight observers each accumulating their statistics, the ranges merging into scales, and convert folding them into the int8 graph.

![A dataflow graph showing calibration data passing through an observer-instrumented model where activation and weight observers feed ranges and scales into the convert step](/imgs/blogs/post-training-quantization-ptq-6.png)

## 8. The calibration-set question

How many samples? What goes in them? This is where PTQ quietly succeeds or fails, and it gets far less attention than it deserves.

**How many.** The honest answer is "enough to estimate the ranges stably, and no more." In practice that is **100 to 1000 samples** for most vision and NLP models. The reason it is so few is that you are estimating a handful of scalars per tensor (a min and a max, or a histogram), not fitting millions of parameters. The range estimate converges fast: by a few hundred diverse samples the observed min/max or histogram has stopped moving meaningfully. Histogram-based methods (entropy, MSE) want the upper end of that range — closer to 1000 — because they need to fill the bins; min/max and percentile are stable at the lower end. A useful sanity check: re-run calibration on two disjoint subsets of your data and compare the resulting scales. If they differ by more than a few percent, you need more samples or your data is heterogeneous.

**What goes in them.** This matters more than the count. The calibration set must be **representative of production inputs** — same distribution, same preprocessing, same diversity. The ranges you bake in are only as good as the data that produced them. This is where things go subtly, expensively wrong.

What goes wrong with a bad calibration set, concretely:

- **Wrong preprocessing.** You calibrate on images normalized with one mean/std and deploy with another. The activation ranges are now systematically off and accuracy tanks. This is the single most common calibration bug, and it is invisible — the model runs, it just runs worse. Always calibrate through the *exact* preprocessing pipeline production uses.
- **Too narrow a distribution.** You calibrate a classifier only on daytime photos, then deploy it on night shots whose activations live in a range you never observed. The night activations get clipped hard because your frozen range never saw them. The fix is to make the calibration set span the production diversity — include the edge cases, the dim images, the long sentences, the rare classes.
- **Calibrating on the wrong split.** Calibrating on training data that was heavily augmented (random crops, color jitter) can give you wider, less representative ranges than the clean inputs you will actually see. Calibrate on data that looks like *inference-time* inputs, not training-time augmented ones.
- **Class imbalance in the calibration set.** If one class produces extreme activations and is over- or under-represented in your calibration sample, your ranges skew. A stratified sample across classes is a cheap insurance policy.

A pragmatic recipe: take 256–512 examples drawn the same way your validation set is drawn, pushed through the *production* preprocessing, stratified across classes or input types, including known edge cases. That covers almost every real model.

#### Worked example: a bad calibration set, debugged

A team quantizes a document-classification BERT to int8 static and sees top-1 drop from 91.2% to 84.7% — a 6.5-point disaster that "shouldn't happen for int8." The culprit turns out to be the calibration set: they sampled 512 *short* documents (the most common length) for calibration, but production sees a long tail of multi-page documents whose attention activations reach magnitudes the short documents never produced. The frozen activation ranges, calibrated only on short inputs, clip the long-document activations hard, and the long documents — about 15% of production traffic — are exactly where the model now fails.

The fix is one change to the calibration set, no change to the method or granularity: re-sample 512 documents *stratified by length*, including the long ones, so the activation observers see the full production range. After re-calibration, top-1 recovers to 90.8% — a 0.4-point drop, shippable. Same model, same per-channel weights, same entropy calibration. The entire 6-point swing was which 512 examples the observers saw. This is why "check the calibration set" sits so high on the debugging checklist: it is invisible (the model runs fine on the common case), expensive (it silently fails the tail), and trivially fixable once you suspect it.

### Two PTQ boosters: cross-layer equalization and bias correction

When plain PTQ leaves a gap but you would rather not jump straight to QAT, two data-free or lightly-data techniques from Nagel et al.'s 2019 "Data-Free Quantization Through Weight Equalization and Bias Correction" (DFQ) can recover a surprising amount, and most toolchains implement at least one of them.

**Cross-layer equalization (CLE).** Recall from section 4 that per-channel quantization fixes uneven *weight* channel magnitudes. But activations are quantized per-tensor, so uneven *activation* channel magnitudes still hurt — one fat channel forces a coarse activation scale. CLE exploits a scale-invariance of consecutive linear layers separated by a positively-homogeneous activation like ReLU: you can multiply the output channels of one layer by a per-channel factor $s_i$ and divide the corresponding input channels of the *next* layer by the same $s_i$, and the network computes the exact same function — but now the channel magnitudes are balanced. Formally, for two consecutive layers with ReLU between them, $\mathrm{ReLU}(s \cdot x) = s \cdot \mathrm{ReLU}(x)$ for $s > 0$, so rescaling by a diagonal matrix $S$ on one side and $S^{-1}$ on the other is a no-op on the function but a big improvement on quantizability. The DFQ paper picks the $s_i$ that equalize the per-channel ranges across the pair. It is *data-free* — pure weight algebra — and it brings per-tensor-activation models much closer to per-channel accuracy, which is exactly why MobileNet (all those depthwise convs) was the paper's flagship result.

**Bias correction.** Quantization introduces a *bias*, not just variance: the expected error $\mathbb{E}[\hat{x} - x]$ is not zero, because rounding and clipping are not symmetric around the true value for skewed distributions. That biased error propagates and shifts the output. Bias correction estimates the expected per-channel quantization error (from a little data, or analytically from the weight distribution) and subtracts it back into the layer's bias term — a free correction, since every conv/linear already has a bias to absorb it. It is cheap, it composes with everything, and it routinely recovers a few tenths of a point. If your toolchain offers it (TensorRT, several ONNX Runtime/optimum paths, and Qualcomm's AIMET all do), turn it on.

## 9. Results: fp32 to int8, measured

Now the part the whole series insists on — measured before→after on a named target, honestly. I will use ResNet-50 on ImageNet quantized for TensorRT int8 on a Jetson Orin Nano (the int8 datapath is exactly what static PTQ unlocks), reporting numbers in the range these consistently land in across NVIDIA's and the literature's published int8 results. Treat the specific decimals as representative, not as a single re-run of your exact build; the *shape* of the result is what is robust.

| Metric | fp32 baseline | int8 PTQ (static, per-channel, entropy) | Change |
| --- | --- | --- | --- |
| Model size | 98 MB | 25 MB | 3.9× smaller |
| Top-1 accuracy | 76.1% | 75.8% | −0.3 pts |
| Latency p50 (batch=1) | 6.8 ms | 2.4 ms | 2.8× faster |
| Latency p99 (batch=1) | 7.5 ms | 2.9 ms | 2.6× faster |
| Peak memory | ~310 MB | ~110 MB | 2.8× less |

That is the PTQ promise made concrete: ~4× smaller, ~2.8× faster, three-tenths of a point of accuracy, zero retraining. On a size–accuracy or latency–accuracy plot this is a strict Pareto improvement — you moved down and to the right with essentially no accuracy cost, which is the whole game from the [taxonomy's Pareto frame](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression).

How to measure this honestly, because a sloppy benchmark will lie to you in your favor:

- **Warm up.** The first dozen inferences include lazy kernel compilation, cache cold-start, and clock ramp. Discard them. Time the steady state, not the cold start.
- **Watch thermal throttling.** A Jetson or a phone will hit a thermal limit and downclock under sustained load. Report latency at a steady thermal state, and note whether your real workload is bursty (stays cool) or sustained (throttles).
- **Batch=1 is the edge reality.** Edge inference is almost always one input at a time, where you cannot amortize fixed overheads across a batch. Throughput at batch=64 is a server number; report batch=1 latency for an edge claim.
- **Report a tail, not just the mean.** p99 is what your SLA is written against. A great p50 with an ugly p99 fails users.
- **Measure the same accuracy, same way.** Use the identical eval pipeline for fp32 and int8 — same preprocessing, same test set — or the "drop" you report is noise.

And the calibration-method comparison, which is the second mandated table and the payoff of section 3 — same ResNet-50, same per-channel weights, varying only the activation calibration method. Figure 7 renders this as a matrix.

![A matrix of ResNet-50 int8 top-1 accuracy by calibration method showing min-max worst and entropy and MSE clipping nearly closing the fp32 gap](/imgs/blogs/post-training-quantization-ptq-7.png)

| Calibration method | Top-1 (%) | Drop vs fp32 | Notes |
| --- | --- | --- | --- |
| Min/max | 74.1 | −2.0 | One outlier set the scale |
| Percentile 99.99 | 75.6 | −0.5 | Robust, cheap default |
| Entropy / KL | 75.8 | −0.3 | TensorRT default, recommended |
| MSE-optimal | 75.9 | −0.2 | Marginally best, more compute |

Read that table next to the "cost of one outlier" worked example and the bias-variance derivation, and the whole arc closes: min/max overpays for the tails and loses two points; clipping methods recover almost all of it; the difference between a fragile int8 model and a shippable one was *which two numbers calibration chose for the activation range.* The method is not a footnote; it is most of the result.

It is worth translating the size column into the terms the edge actually cares about, because "3.9× smaller" is abstract until it crosses a hard budget. A 98 MB model that drops to 25 MB is the difference between a mobile app feature that ships and one that gets cut for bloating the download. On a microcontroller with 1 MB of flash, the same ratio is the difference between fitting and not fitting at all — there the question is binary, not gradual. And the peak-memory column matters as much as the size column on a constrained device: a model that *stores* in 25 MB but *peaks* at 110 MB of working memory during inference still OOMs a device with 64 MB of RAM. When you report PTQ results for an edge target, report both the static size (does it fit in storage?) and the peak runtime memory (does it run without OOM?), because they bind different budgets and a win on one is not a win on the other. The latency column, similarly, is only meaningful against the SLA: 2.4 ms p50 is a triumph if the budget is 5 ms and irrelevant if a downstream stage already eats 50 ms. Always quote the result against the budget it has to clear.

A note on cost, since the edge often runs on someone's electricity bill or battery. Energy per inference tracks data movement closely — moving a byte from DRAM costs orders of magnitude more energy than an arithmetic operation — so the same int8 conversion that cut bytes moved also cuts energy, often roughly in proportion to the memory-traffic reduction. On a battery device that can be the difference between a feature you can run continuously and one you ration. If you are paying for inference at scale, a 2.8× latency win is also close to a 2.8× reduction in the compute you rent per request; at, say, \$0.01 of compute per thousand inferences in fp32, the int8 build drops that toward \$0.0036, which compounds fast at production volume. The cheapest win is cheap on every axis at once.

## 10. Stress-testing PTQ: int4, tiny sets, and op fallback

A technique you only ever see succeed is a technique you do not understand. Here is where PTQ breaks, because the failure modes are exactly where the science from sections 2–5 predicts they should be.

**Stress 1: push to int4.** Everything above was int8. What happens at 4 bits? The quantization-error term carries $1/(2^b-1)^2$, so going from 8 bits (255 levels) to 4 bits (15 levels) makes the step size 17× coarser for the same range. On a typical CNN, naive per-tensor int4 PTQ collapses — double-digit accuracy losses are common. Per-channel int4 helps but still leaves several points on the table on most models. This is *exactly* the regime where the closed-form clipping of ACIQ and the second-order corrections of GPTQ were invented: at int4 the range choice and the rounding choice stop being refinements and become the difference between a working model and noise. The practical rule: **int8 PTQ is usually free; int4 PTQ usually needs either a sophisticated PTQ method (GPTQ/AWQ-style) or QAT.** For weight-only LLM quantization int4 is routine because the weights are well-behaved and the methods are clever; for activation quantization of CNNs at int4, expect to train.

#### Worked example: int8 versus int4 on the same model

Same ResNet-50, per-channel weights, entropy calibration, varying only bit-width.

- **int8:** top-1 ≈ 75.8% (−0.3 from fp32's 76.1%). Free.
- **int4, naive per-channel PTQ:** top-1 ≈ 71–72% (−4 to −5 points). Unusable for most products straight out of conversion.
- **int4 with GPTQ-style error correction:** top-1 ≈ 74–75% (−1 to −2 points). Much better, but now you are running a real second-order quantization pass, not a ten-minute convert.

The size win is tempting — int4 is 8× smaller than fp32, twice the int8 saving — but read the accuracy column before you reach for it. The marginal 2× size reduction from int8→int4 costs you, on a CNN, a multi-point accuracy hit that int8→fp32 never did. That is the bit-width cliff the variance law predicted.

**Stress 2: shrink the calibration set.** What if you only have 8 calibration samples? Or 1? With min/max, a tiny calibration set is dangerous in *both* directions: too few samples may *miss* the true peak activation (so your range is too tight and you clip real signal at deployment), or a single freak sample may *set* the peak (so your range is too loose and you coarsen everything). Histogram methods (entropy, MSE) degrade harder with tiny sets because the histogram is too sparse to estimate a good clip point — they genuinely need their few hundred samples to fill the bins. The robust behavior at small $n$ comes from percentile and moving-average min/max, which tolerate a handful of samples more gracefully. Below ~32 samples, prefer percentile or moving-average min/max over histogram methods; below ~8 samples, distrust the result and gather more data — calibration is cheap precisely because you need so few samples, so there is rarely a good reason to starve it.

**Stress 3: the NPU does not support the op.** You quantize to int8, deploy to an NPU, and it is *slower* than fp16. The cause is almost always an unsupported op forcing a fallback: the NPU runs your int8 convs fast, hits a layer it cannot do in int8 (an exotic activation, a custom op, a reshape pattern the compiler does not recognize), and falls back to the CPU — which means dequantizing int8→float, shipping the tensor off the accelerator, computing on the CPU, requantizing, and shipping it back. Each fallback is a round trip across the memory bus plus two conversions, and a few of them in the middle of your graph can erase the entire int8 speedup. The fix is to *profile the deployed graph* (every runtime has a layer-level profiler) and either (a) replace the unsupported op with a supported equivalent, (b) restructure so the unsupported ops cluster at the graph's edges rather than the middle, or (c) accept a partition where a contiguous int8 subgraph runs on the NPU and the rest on CPU. This is the moment the [roofline post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) earns its place: int8 only helps where you are compute-bound on supported ops; a fallback turns a compute-bound win into a memory-bound loss.

**Stress 4: the model is memory-bound, not compute-bound.** Here is the subtle one. If a layer is *memory-bound* — its runtime dominated by moving weights and activations, not by arithmetic — then quantizing the *compute* to int8 does nothing for its speed; the bottleneck was bandwidth, not math. But quantization *also* shrinks the data you move (int8 weights and activations are a quarter the bytes of fp32), so it can still help a memory-bound layer — just for a different reason than you think. The distinction matters because it tells you what to expect: on a compute-bound layer, int8 speeds up the math (up to the int8/fp32 throughput ratio of the hardware); on a memory-bound layer, int8 speeds up the data movement (up to the bandwidth saved). If your layer is memory-bound and you only quantized weights but kept activations in float (weight-only), you got the weight-bandwidth saving but not the activation one — which is exactly why weight-only is the right call for LLM decoding (weight-bandwidth-bound) and static is the right call for a compute-bound CNN on an int8 NPU. Match the mode to the bottleneck, and profile to know which bottleneck you actually have.

## 11. Debugging an accuracy drop

You ran PTQ, you measured, and the accuracy fell more than you can accept. Before you reach for QAT, work this checklist — in order, because the cheap fixes are also the common ones.

**1. Confirm per-channel weights are on.** This is the first thing to check, every time, because forgetting it is the most common cause of a big drop and the fix is a one-line config change. If you are per-tensor on weights, switch to per-channel and re-measure before anything else.

**2. Check the calibration set.** Wrong preprocessing, too-narrow distribution, augmented data — section 8's failure modes. Re-calibrate with a properly representative, production-preprocessed set. A surprising fraction of "PTQ doesn't work on my model" turns out to be a calibration-data bug, not a quantization-method limitation.

**3. Switch the calibration method.** If you were on min/max, move to entropy or percentile (section 9 shows the swing this alone produces). If you were already on entropy, try MSE-optimal. Free to try, sometimes the whole fix.

**4. Turn on the PTQ boosters.** Cross-layer equalization and bias correction (section 8) if your toolchain has them. Cheap, composable, often a few tenths of a point each.

**5. Do a per-layer sensitivity analysis.** This is the systematic move when the blunt fixes are exhausted. Quantize one layer at a time (or leave one layer at a time in fp32) and measure the accuracy impact of each. Some layers are far more sensitive than others, and a few sensitive layers usually account for most of the drop. The companion post on [per-layer sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) goes deep on the methodology; the practical upshot is that you can often recover most of the lost accuracy by leaving just two or three layers in fp16 while keeping the rest int8 — a *mixed-precision* model that is almost as fast and almost as small but much more accurate.

A minimal sensitivity sweep looks like this:

```python
# Leave-one-layer-in-fp32 sensitivity sweep.
# For each quantizable layer, keep it fp32 and quantize the rest,
# then measure how much accuracy that one layer's quantization was costing.
baseline_int8_acc = evaluate(model_int8_all)   # everything quantized

sensitivity = {}
for layer_name in quantizable_layers:
    qconfig_mapping = make_qconfig_excluding(layer_name)  # this layer stays fp32
    m = prepare_fx(model_fp32, qconfig_mapping, example_inputs)
    calibrate(m, calibration_loader)
    m = convert_fx(m)
    sensitivity[layer_name] = evaluate(m) - baseline_int8_acc

# Layers with the largest positive delta are the ones worth keeping in fp16.
for name, gain in sorted(sensitivity.items(), key=lambda kv: -kv[1])[:5]:
    print(f"{name}: +{gain:.2f} pts if left in higher precision")
```

**The first/last-layer rule.** Even before a full sweep, there is a heuristic that pays off constantly: **keep the first and last layers in higher precision.** The first layer sees the raw input, whose range is fixed by the data (pixel values, token embeddings) and is often wide and hard to quantize cleanly; the last layer produces the logits, whose relative spacing decides the argmax, so quantization noise there directly flips predictions. These two layers are also a tiny fraction of total compute (the first layer's input is small; the last layer is one matmul), so keeping them in fp16 costs almost no speed. Many production int8 recipes quantize everything *except* the first and last layer by default. It is the 80/20 of mixed-precision PTQ.

Figure 8 turns this into a decision: run the sensitivity sweep, keep the bulk in int8, and promote to fp16 exactly the layers the sweep flags — which, predictably, cluster around the first layer (raw input range), the last/classifier layer (logit spread), and any attention-softmax or outlier-heavy activation site.

![A decision tree rooted at a per-layer sensitivity sweep branching into layers kept in int8 and sensitive layers promoted to fp16 such as the first layer, the classifier, and attention activations](/imgs/blogs/post-training-quantization-ptq-8.png)

**When PTQ is genuinely not enough.** If you have done all of the above — per-channel, good calibration, best method, boosters, mixed-precision on the sensitive layers — and you are *still* below target, that is your signal. The remaining gap is not a calibration problem; it is that the model's weights and activations were never shaped to tolerate int8 rounding, and no amount of cleverness at conversion time can change weights that already exist. That is exactly the boundary where [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) earns its cost: by simulating the quantization during a fine-tuning pass (with the straight-through estimator carrying gradients past the round operation), QAT lets the weights *move* to positions that quantize cleanly. It costs you a training run, but it is the only thing that closes a stubborn gap, and it routinely recovers the last one to four points that PTQ cannot. The decision is binary and evidence-based: **PTQ until it stops being enough, then QAT — and not a moment sooner**, because QAT is strictly more expensive and you should never pay for it if the free path already hit the target.

## 12. Case studies: real numbers from the field

Four results from the literature and shipped products that ground everything above. I have kept the numbers to ones that are well-attested; where I am giving a representative figure rather than a single canonical one, I say so.

**MobileNetV2 on mobile (Krishnamoorthi 2018, and Nagel et al. DFQ 2019).** This is the canonical per-channel case study. Per-tensor int8 PTQ of MobileNetV2 loses several points of ImageNet top-1 (the depthwise convolutions, with their per-channel magnitude spread, are the culprit). Per-channel weight quantization recovers it to within roughly a point of fp32. DFQ's cross-layer equalization plus bias correction then closes most of the *remaining* gap *data-free*, getting per-tensor models to near per-channel accuracy without any calibration data at all — the result that made CLE famous. The takeaway: on depthwise-heavy mobile architectures, granularity and equalization are the whole ballgame.

**BERT dynamic quantization on CPU.** Dynamically quantizing the linear layers of a BERT-base encoder to int8 on CPU gives roughly a 2–4× speedup on those matmuls with a negligible accuracy change on most GLUE tasks — and it needs *zero* calibration data, just `quantize_dynamic`. This is the poster child for "the easiest PTQ is often enough": for CPU-served transformer encoders, dynamic int8 is frequently the entire optimization you ever need to do.

**TensorRT int8 with entropy calibration (Migacz 2017).** NVIDIA's int8 calibration — the KL-divergence method from section 3 — is the reason a TensorRT int8 engine built from a good calibration set so reliably lands within a fraction of a point of fp32 on ImageNet CNNs while delivering the 2–4× latency win on the int8 tensor cores. The 2017 GTC presentation that introduced it is still the clearest explanation of *why* entropy calibration beats min/max, and it is why "Entropy" is the default `calibrate_method` so many production pipelines use.

**LLMs and weight-only PTQ (GPTQ 2022, AWQ 2023).** For large language models the story flips to weight-only, because decoding is memory-bound on the weights. GPTQ quantizes a 175B-class model's weights to 3–4 bits with a clever second-order error-correction during the per-layer quantization, keeping perplexity close to fp16 — and it is still *post-training*, no full retraining, just a one-pass calibration over a few hundred sequences. AWQ refines this by protecting the salient weight channels (identified from activation statistics) so 4-bit weight-only quantization holds up even better. Both are PTQ in spirit — calibrate, quantize the finished weights, ship — and both are what make running a 7B–70B model on a laptop or a single consumer GPU practical. (The numerics of weight-only LLM quantization deserve their own treatment; this post's recipe is the CNN/encoder backbone they build on.)

## 13. When to reach for PTQ (and when not to)

PTQ is a cost like every technique, even though the cost is small — so here is the decisive guidance, including when *not* to bother.

**Reach for PTQ when:** you have a trained model and a representative data slice; your target has int8 (or has memory-bound weight-bandwidth that weight-only relieves); and you have not yet hit your size/latency budget. This is the default first move for *every* edge deployment. It is so cheap that "did you try plain PTQ?" should be the first question in any optimization review.

**Specifically, start with the cheapest variant that fits your hardware.** CPU transformer? `quantize_dynamic`, done in one line. LLM that is too big for memory? Weight-only (GPTQ/AWQ/GGUF k-quants). Dedicated int8 accelerator? Static PTQ with per-channel weights and entropy calibration. Climb to a more expensive variant only when the cheaper one misses the target.

**Do not reach for PTQ — or do not stop at it — when:**

- **PTQ already failed and you have target headroom only QAT can reach.** If you have exhausted section 10's checklist and you are still two-plus points short, more PTQ tuning is rearranging deck chairs. Go to QAT.
- **Your hardware has no int8 path and you are compute-bound.** Quantizing a model for a target whose kernels run int8 *slower* than fp16 (it happens — some GPUs, some unsupported ops that fall back to CPU) buys you size but *loses* you speed. Profile first; the [roofline post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) is how you know whether int8 can even help your bottleneck before you spend a day on it.
- **The model is already small enough and fast enough.** If you are inside budget in fp16, do not quantize for sport. Every quantization is a small accuracy risk and a maintenance burden; do not pay it for a win you do not need. This is the series' refrain — refuse to pay for wins you do not need.
- **An op you depend on has no int8 kernel on your target.** A single unsupported op can force a float fallback mid-graph, with expensive int8↔float conversions around it that erase the speedup. Check op coverage on your specific runtime before committing.

The series capstone, [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook), folds PTQ into the full decision flow across all four levers; treat this post as the deep dive on the corner of it that you will use most.

## 14. Key takeaways

- **PTQ is the cheapest win: no retraining, ~4× smaller, ~2–4× faster, minutes of work.** It is the correct first move for almost every edge deployment, and "did you try plain PTQ?" should precede every harder technique.
- **Calibration is one thing: estimating activation ranges from representative data.** Weights are static and easy; activations need a few hundred forward passes through observers to learn their ranges.
- **The range is a clipping-versus-quantization-error trade-off, and the best range is almost never the true max.** Min/max overpays for outliers; percentile, entropy/KL, and MSE-optimal clipping minimize total error and recover most of the gap.
- **Per-channel weights are nearly free and the single best accuracy-per-effort move.** Forgetting them is the most common cause of a big int8 drop; check this first when debugging.
- **Pick the mode from the hardware: static for int8 accelerators, dynamic for CPU transformers, weight-only for LLMs.** Static needs calibration; dynamic and weight-only do not.
- **A bad calibration set fails silently.** Match production preprocessing, span the production distribution, stratify, and include edge cases — 100–1000 samples is plenty.
- **Keep the first and last layers in higher precision, and use a sensitivity sweep for the rest.** A few fp16 layers in a mostly-int8 model recover most of a stubborn drop at almost no speed cost.
- **PTQ until it stops being enough, then QAT — never sooner.** When the section-10 checklist is exhausted and you are still short, the gap is in the weights themselves, and only training can move them.

## 15. Further reading

- Nagel, Fournarakis, Amjad, Bondarenko, van Baalen, Blankevoort — *A White Paper on Neural Network Quantization* (2021). The definitive practitioner reference for PTQ and QAT; read it cover to cover.
- Nagel, van Baalen, Blankevoort, Welling — *Data-Free Quantization Through Weight Equalization and Bias Correction* (ICCV 2019). The DFQ paper: cross-layer equalization and bias correction, the two PTQ boosters from section 8.
- Banner, Nahshan, Soudry — *Post-training 4-bit quantization of convolutional networks for rapid-deployment* (ACIQ, NeurIPS 2019). The closed-form optimal-clipping analysis behind section 2's bias-variance derivation.
- Migacz — *8-bit Inference with TensorRT* (NVIDIA GTC 2017). The original entropy/KL-divergence calibration method; still the clearest explanation of why it beats min/max.
- Krishnamoorthi — *Quantizing deep convolutional networks for efficient inference: A whitepaper* (2018). The foundational PTQ study, including the per-tensor-versus-per-channel result.
- Frantar, Ashkboos, Hoefler, Alistarh — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022), and Lin et al. — *AWQ: Activation-aware Weight Quantization* (2023). Weight-only PTQ for LLMs.
- Official docs: PyTorch `torch.ao.quantization`, ONNX Runtime quantization (`quantize_static`, `CalibrationDataReader`), TensorFlow Lite post-training quantization, and NVIDIA TensorRT int8 calibration.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles), [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat), and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

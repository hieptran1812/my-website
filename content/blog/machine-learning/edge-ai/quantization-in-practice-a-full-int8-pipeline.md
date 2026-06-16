---
title: "Quantization in practice: a full int8 pipeline and how to debug accuracy drops"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A copy-and-adapt int8 conversion workflow for a real model, end to end, plus the per-layer debugging playbook for the day the quantized model loses four points of accuracy."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "int8",
    "ptq",
    "onnx-runtime",
    "tflite",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-1.png"
---

You have read the theory. You know that int8 stores a number in one byte instead of four, that an affine map with a scale and a zero-point gets you there, that rounding noise costs you about six decibels of signal-to-noise per bit, that per-channel weights beat per-tensor, that PTQ is cheap and QAT is the heavy artillery. None of that helps you at 11pm on a Thursday when the model you converted this afternoon is sitting at 67.1% top-1 where the fp32 baseline was 71.8%, the release is tomorrow, and the only thing anyone upstream wants to know is *why* and *how long to fix it*.

This post is the workflow for exactly that situation. It is the hands-on capstone of the quantization track: it ties together everything from [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), the mechanics in post-training quantization and quantization-aware training, and the measurement discipline in [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) into one runnable pipeline for a real model — MobileNetV2 on ImageNet, with the LLM path as a contrast — and then it gives you the debugging playbook for when that pipeline produces a number you do not like. The two halves are equally important. Anyone can run `torch.ao.quantization.convert`. The skill that gets you paid is being able to look at a 4.7-point accuracy drop, attribute it to one layer in twenty minutes, fix it with a single line of config, and know that you have not just moved the problem somewhere you cannot see it.

By the end you will be able to do six concrete things: measure an honest fp32 baseline, prepare a model correctly (fuse, per-channel, symmetric-versus-asymmetric), calibrate on a representative set, convert in PyTorch and export to ONNX Runtime and TFLite, validate with a numerical diff that localizes error to a layer, and benchmark on a named target. And when accuracy tanks, you will have a triage tree that walks from the cheap global fixes to per-layer attribution to a surgical fp16 fallback, plus a footgun list that catches the silent corruptions before they ship.

Figure 1 is the whole pipeline on one slide — six stations from a measured baseline to a benchmarked deploy. Keep it open; every section below is one station or one debugging detour off it.

![A horizontal timeline of the six int8 pipeline stages from baseline through prepare, calibrate, convert, validate, and deploy on a target device](/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-1.png)

One framing before we start, because it governs every decision in this post. Quantization sits on the **quantization lever** of the four-lever Pareto frame: it spends *accuracy* and *engineering effort* to buy *size*, *speed*, and *energy*. The entire discipline of doing it well is refusing to spend more accuracy than you have to. A sloppy int8 conversion and a careful one produce the same 4× size reduction; the careful one keeps 0.4 points where the sloppy one loses 4.7. The difference is never the conversion call. It is the preparation, the calibration set, and the willingness to attribute and fix instead of shrug and ship.

## 1. Station zero: the honest baseline

The single most common mistake in a quantization project is not a quantization mistake at all. It is failing to measure the baseline properly, so that you have no idea whether int8 cost you anything and no number to fix toward. Before you quantize one tensor, you measure three things about the fp32 model on the *exact* evaluation harness you will use for the int8 model: **accuracy**, **latency**, and **size**. If you skip this, every downstream number is unmoored.

The reason this matters so much is that the int8 accuracy delta is a *difference*, and a difference is only as trustworthy as its two endpoints. If your fp32 "baseline" is 71.8% but you measured it with a different image-preprocessing pipeline than your int8 eval, or on a different validation split, or with test-time augmentation on one and off the other, then your "-4.7 points" is partly an artifact of the harness, not the quantization. I have watched a team spend two days hunting a quantization bug that turned out to be a center-crop versus resize mismatch between their two eval scripts. Measure both numbers through one function.

Here is the baseline measurement, written so the *same* `evaluate` and `benchmark` functions run on fp32 and on every quantized variant later.

```python
import time
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# One preprocessing pipeline, used for BOTH fp32 and int8 eval.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_set = torchvision.datasets.ImageFolder("imagenet/val", preprocess)
val_loader = DataLoader(val_set, batch_size=64, num_workers=8)

def evaluate(model, loader, device="cpu", max_batches=None):
    model.eval().to(device)
    correct = total = 0
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            out = model(x.to(device))
            correct += (out.argmax(1).cpu() == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total

def benchmark(model, device="cpu", shape=(1, 3, 224, 224),
              warmup=20, iters=200):
    model.eval().to(device)
    x = torch.randn(*shape, device=device)
    with torch.inference_mode():
        for _ in range(warmup):          # warm caches, JIT, allocator
            model(x)
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            lat.append((time.perf_counter() - t0) * 1e3)  # ms
    lat.sort()
    return {"p50": lat[len(lat)//2], "p99": lat[int(len(lat)*0.99)]}

def model_size_mb(model):
    torch.save(model.state_dict(), "/tmp/m.pt")
    import os
    return os.path.getsize("/tmp/m.pt") / 1e6

fp32 = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2")
print("fp32 acc :", evaluate(fp32, val_loader, max_batches=50))
print("fp32 lat :", benchmark(fp32))
print("fp32 size:", model_size_mb(fp32), "MB")
```

Two details in there are not optional. The **warm-up loop** in `benchmark` exists because the first few inferences pay for lazy allocator growth, kernel autotuning, and cold instruction caches; if you include them your p50 is a lie that flatters the slow path. And measuring at **batch=1** is the on-device reality for most edge inference — you are classifying one camera frame, transcribing one audio chunk, answering one prompt — so a throughput number at batch=128 tells you almost nothing about the latency a user feels. We measure p50 and p99 because the tail is where the pager goes off; a model whose p50 is 18ms but whose p99 is 95ms will blow a 30ms frame budget one frame in a hundred, which at 30fps is a visible stutter every three seconds.

| Baseline metric | What it measures | How to measure it honestly |
| --- | --- | --- |
| Accuracy | task quality on the real eval split | one preprocessing path, full or fixed-subset val set, same metric definition |
| Latency p50/p99 | typical and tail per-inference time | warm up first, batch=1, on the target (or a faithful proxy), report the tail |
| Size (MB) | storage and download cost | serialized weights on disk, not parameter count times bytes |
| Peak memory | does it fit in RAM/SRAM | resident set or arena high-water mark during one inference |

Write these four numbers down. They are the contract the rest of the pipeline is measured against. The metrics post argues this case in full; here it is enough to say that a quantization project without a frozen baseline is not a project, it is a vibe.

### The science: how much accuracy *should* int8 cost?

Before you go hunting for a bug, it pays to know what a *good* int8 conversion costs in principle, so you can tell "this is expected" from "this is broken." The governing quantity is the **signal-to-quantization-noise ratio (SQNR)** — the ratio of the signal power to the rounding-noise power, and it has a clean closed form.

Quantization rounds each value to the nearest of $2^b$ levels spaced $s$ apart, where $b$ is the bit-width. The rounding error $e = \hat{x} - x$ lands somewhere in $[-s/2, s/2]$, and for a signal that is "busy" relative to the step (the usual case for weights and activations), $e$ is well-modeled as **uniform** over that interval. A uniform variable on $[-s/2, s/2]$ has variance

$$\sigma_e^2 = \frac{1}{s}\int_{-s/2}^{s/2} e^2 \, de = \frac{s^2}{12}.$$

That is the noise power. Now suppose the signal occupies a range $R$ that you split into $2^b$ levels, so $s = R / 2^b$. The noise power becomes $\sigma_e^2 = R^2 / (12 \cdot 2^{2b})$. Taking the ratio of signal power $\sigma_x^2$ to noise power and converting to decibels:

$$\text{SQNR (dB)} = 10\log_{10}\frac{\sigma_x^2}{\sigma_e^2} = 6.02\,b + 1.76 + 10\log_{10}\frac{\sigma_x^2}{(R/\sqrt{12})^2}.$$

The headline term is **6.02 dB per bit.** Every bit you add halves the step $s$ and buys ~6 dB of signal-to-noise. The corollary that matters for debugging: going from fp32 (effectively unlimited SQNR) to int8 gives you a *ceiling* of roughly $6.02 \times 8 \approx 48$ dB of SQNR for a perfectly range-matched signal — plenty for a network whose layers are individually tolerant of a percent or two of relative noise. So a *well-prepared* int8 CNN losing only a few tenths of a point is exactly what the math predicts. A 4.7-point drop is **not** the 6-dB-per-bit law biting — it is something broken: a wasted range, an outlier, a wrong axis. The science tells you the honest int8 budget is small, which is precisely why a large drop is always a bug, not a law of nature.

The third term in the SQNR formula is the lever you control. It is maximized — noise minimized — when the quantization range $R$ tightly matches the signal's actual spread $\sigma_x$. That single term is the mathematical reason *everything* in the prepare and calibrate stations matters: per-channel scaling, outlier clipping, and good calibration all do the same thing — they shrink $R$ to fit the real signal so the $R^2$ in the denominator of the noise power gets small and SQNR climbs. When one channel's range is set by an outlier 10× the bulk, that channel's effective $R$ is 10× too big, you have thrown away $20\log_{10}(10) \approx 20$ dB of SQNR on that channel, and 20 dB is more than three bits — your "int8" channel is effectively a 5-bit channel. That is the entire outlier story in one line of arithmetic.

### The requantization math you will see in every int8 graph

One more piece of science, because it shows up by name in the code and in every footgun about scales. When an int8 matmul runs, it multiplies int8 activations (scale $s_x$) by int8 weights (scale $s_w$) and accumulates in int32. The integer accumulator holds a number in units of $s_x s_w$. To get back to the output's int8 representation (scale $s_y$), the kernel multiplies by

$$M = \frac{s_x \, s_w}{s_y},$$

a single floating-point (or fixed-point) **requantization multiplier** applied per output element. With per-channel weights, $s_w$ — and therefore $M$ — varies per output channel, which is why per-channel costs nothing at runtime: it is just a different $M$ per channel, absorbed into a multiply the kernel was doing anyway. This $M$ is also exactly what a concat/add of differently-scaled tensors gets wrong: if two branches arrive with different $s$, you cannot add their int8 codes, because the codes mean different things — you must requantize both to a common $s_y$ first. Knowing the requantization multiplier exists turns "concat is broken" from mysterious into obvious: the codes were never in the same units.

## 2. Station one: PTQ or QAT? Decide before you prepare

The first real fork is *which kind* of quantization. Post-training quantization (PTQ) takes a trained fp32 model and quantizes it without any further gradient training: you feed it a few hundred representative inputs to estimate activation ranges, then convert. Quantization-aware training (QAT) inserts "fake-quant" nodes into the graph and *fine-tunes* the model for a few epochs so the weights learn to be robust to rounding. PTQ costs minutes and no labeled training loop; QAT costs a training run and a data pipeline but typically recovers most of the accuracy PTQ leaves on the table.

The decision rule is simple and you should apply it in this order, because the cheapest option that hits target wins: **try PTQ first; reach for QAT only when PTQ misses your accuracy budget and you have exhausted the per-channel and calibration fixes.** PTQ with per-channel weights and a decent calibration set already lands within roughly 1% of fp32 for most well-behaved CNNs like MobileNetV2 or ResNet-50. If you are at 0.4 points down, you are done — QAT would burn a training run to buy you maybe 0.2 points you do not need. The taxonomy post's rule applies: do not pay for a win you do not need.

QAT earns its cost in two situations. First, when the architecture is *intrinsically* hard to quantize — depthwise-separable convolutions (MobileNet's signature block) have notoriously wide, spiky weight distributions per channel, and at int8 PTQ on aggressive variants you can see 2–4 point drops that QAT recovers. Second, when you are pushing below int8 — int4 weight quantization almost always wants QAT or a sophisticated PTQ method (GPTQ, AWQ for LLMs) because the rounding noise at four bits is large enough that the model genuinely has to be trained around it. The deep mechanics of both paths live in the dedicated [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) and [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) posts; this pipeline assumes PTQ and shows where QAT slots in if you need it.

| | PTQ | QAT |
| --- | --- | --- |
| Cost | minutes, no training loop | a fine-tune run + data pipeline |
| Data needed | ~100–500 unlabeled calib samples | full labeled training set |
| Typical acc vs fp32 (int8 CNN) | -0.3% to -1.5% | -0.1% to -0.5% |
| When it shines | well-behaved nets, int8, fast iteration | depthwise/spiky nets, int4, last 1–2 points |
| When it is overkill | already within budget | when PTQ already hits target |

The honest practice is to *time-box PTQ*. Give it an afternoon: baseline, prepare, calibrate, convert, attribute, fix per-channel and clipping. If you are still outside budget after that, you have a defensible reason to spend the QAT run, and — crucially — your per-layer attribution from PTQ tells you which layers QAT most needs to fix.

## 3. Station two: prepare the model (the science of fusion and granularity)

Preparation is where most of the accuracy is won or lost, and it happens *before* a single number is quantized. Two things happen here: you **fuse** sequences of ops into single quantized kernels, and you **choose the granularity and symmetry** of the quantizers. Figure 2 will show the debugging tree later; figure 6 shows this preparation stack, so I will reference it here and embed it under this section.

### Why fuse conv-bn-relu

Batch normalization at inference time is an affine transform: $y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$, where $\mu, \sigma^2$ are the running statistics and $\gamma, \beta$ the learned scale and shift. Because it is affine and it sits right after a convolution (which is also linear in its weights), you can *fold* the BN parameters into the convolution's weights and bias and get an arithmetically identical output from a single conv op. Concretely, if the conv computes $W x + b$ and BN then applies $a(\cdot) + c$ where $a = \gamma / \sqrt{\sigma^2 + \epsilon}$ and $c = \beta - a\mu$, the fused conv has weights $W' = a W$ and bias $b' = a b + c$.

Folding matters for quantization for a sharp, often-missed reason: **if you quantize the conv output and the BN separately, you insert a quantize-dequantize round trip in the middle of what should be one linear operation, and that round trip adds rounding noise for nothing.** Worse, an unfolded BN at inference can have a wildly different output range than the conv before it, so the activation observer sees the wrong statistics. Folding first means you calibrate the *real* combined op. Add the ReLU into the fusion and the quantized kernel can clamp at zero and pick its output range tightly. This is exactly the fused int8 conv+bias+ReLU kernel the compiler can only emit if you let it see the fused graph.

```python
import torch
from torch.ao.quantization import fuse_modules

# MobileNetV2 blocks are Conv -> BN -> ReLU6; fuse each triple.
# (torchvision exposes a fuse_model() helper on the quantizable variant,
#  but doing it explicitly shows what is happening.)
def fuse_mobilenet(model):
    for m in model.modules():
        if type(m).__name__ == "ConvBNReLU" or hasattr(m, "0"):
            # fuse Conv(0)-BN(1)-ReLU(2) inside each sequential block
            try:
                fuse_modules(m, ["0", "1", "2"], inplace=True)
            except (IndexError, AttributeError):
                try:
                    fuse_modules(m, ["0", "1"], inplace=True)  # conv-bn only
                except Exception:
                    pass
    return model

# The clean path: torchvision ships a quantization-ready MobileNetV2.
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
model = q_mobilenet_v2(weights="IMAGENET1K_QNNPACK_V1", quantize=False)
model.eval()
model.fuse_model()   # fuses conv-bn-relu across the whole net
```

### Why per-channel weights and symmetric weights

Here is the granularity science, condensed from the from-first-principles post into the rule you act on. A quantizer maps a real range to 256 int8 codes with one scale $s$. Use **one scale for the whole weight tensor** (per-tensor) and that scale must be large enough for the loudest channel — so every quieter channel's weights collapse onto a handful of codes and lose resolution. Give **every output channel its own scale** (per-channel) and each channel quantizes to its own tight range. For a depthwise-separable conv where channel magnitudes vary by 10× or more, per-channel weights are frequently the single biggest accuracy lever in the whole pipeline, and they are essentially free because the dequantization absorbs the per-channel scale into the existing requantization multiply.

Symmetry is the second choice. **Symmetric quantization** forces the zero-point to zero, so the int8 matmul is a clean int32 dot product with no cross-terms; **asymmetric (affine)** keeps a nonzero zero-point to fit skewed ranges but adds correction terms to the matmul. The standard, well-supported recipe — the one PyTorch's `qnnpack`/`x86` backends, ONNX Runtime, and TFLite all optimize for — is **per-channel symmetric weights, per-tensor asymmetric activations.** Weights are roughly zero-centered so symmetric wastes little range and you get the fast matmul; activations after a ReLU are one-sided (all $\ge 0$), so asymmetric lets the zero-point sit at the bottom of the range and use all 256 codes instead of throwing away the negative half.

![A vertical stack showing model preparation from a raw separate graph through conv-bn-relu fusion, per-channel symmetric weights, and asymmetric activations to a prepared graph with observers](/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-6.png)

The `QConfig` that encodes this recipe in PyTorch:

```python
import torch
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import (
    HistogramObserver, PerChannelMinMaxObserver)

# Activations: per-tensor, asymmetric (affine), 8-bit unsigned-ish via qint8.
act_observer = HistogramObserver.with_args(
    qscheme=torch.per_tensor_affine, dtype=torch.quint8)

# Weights: per-channel (axis 0 = output channels), symmetric.
wt_observer = PerChannelMinMaxObserver.with_args(
    qscheme=torch.per_channel_symmetric, dtype=torch.qint8, ch_axis=0)

my_qconfig = QConfig(activation=act_observer, weight=wt_observer)

# Or just use the backend default, which already encodes the recipe above:
from torch.ao.quantization import get_default_qconfig
qconfig = get_default_qconfig("qnnpack")   # ARM mobile; "x86" for desktop
```

The `ch_axis=0` is load-bearing and a classic footgun we will return to: for a conv weight tensor of shape `[out_channels, in_channels, kh, kw]`, the per-channel scales live on axis 0 (output channels). Put them on the wrong axis and you will get scales that make no physical sense and a per-layer error that looks like a hardware bug. More on that in section 9.

#### Worked example: per-channel versus per-tensor on a depthwise layer

Concretely, take one depthwise conv from MobileNetV2 with 96 channels whose per-channel weight ranges span from about $[-0.08, 0.08]$ for the quiet channels to $[-0.9, 0.9]$ for the loud one. Per-tensor uses one scale set by the loudest channel: $s = 0.9/127 \approx 0.0071$. A quiet channel whose weights live in $[-0.08, 0.08]$ then uses only $\pm 0.08/0.0071 \approx \pm 11$ codes out of $\pm 127$ — it has thrown away more than three bits of resolution, the 20-dB-per-decade SQNR loss from the science block, on that channel. Per-channel gives that quiet channel its own $s = 0.08/127 \approx 0.00063$, so it uses the full $\pm 127$ codes again. Across the layer, that recovered roughly 3 of the 4.7 points in the running example, for zero runtime cost — the requantization multiplier $M$ just becomes per-channel. This is why the playbook checks granularity *first*: it is the cheapest fix with the largest payoff for exactly the architectures (depthwise-separable) that dominate edge vision.

## 4. Station three: calibration (estimating the activation ranges)

Weights are static — you can read their min and max directly. Activations are not; their range depends on the input, so you have to *estimate* it by running representative data through the prepared model and watching what the observers see. This is calibration, and it is where a quietly wrong choice produces a model that looks fine offline and falls over in production.

Three things govern calibration quality: the **data**, the **method**, and the **amount**.

The **data** must be representative of production inputs and must come from the *training or a held-out calibration split — never the test set.* Calibration peeks at the data to set ranges; if those samples overlap your evaluation set you have leaked the test set into the model and your reported accuracy is optimistic. This is one of the six footguns and it is insidious because the model genuinely scores higher offline; you only discover it when live accuracy disappoints. Pull calibration samples from the same distribution as deployment — same camera, same lighting, same language mix, same prompt style — because the ranges you estimate are only valid for inputs that look like what you calibrated on.

The **method** is how you turn the stream of observed values into a single range. The two workhorses are **min-max** (take the literal min and max seen) and **histogram/entropy** (build a histogram and pick the clipping range that minimizes information loss, often KL-divergence — this is what TensorRT's "entropy calibrator" and PyTorch's `HistogramObserver` do). Min-max is fragile: a single outlier activation stretches the range so every ordinary value collapses to a few codes (the outlier story from the from-first-principles post). Histogram/percentile calibration *clips* the tail — say, keep the 99.99th percentile and let the rare giant values saturate — which sacrifices a handful of extreme values to give the bulk of the distribution far more resolution. For activation functions with unbounded positive ranges (GELU, Swish/SiLU, plain ReLU on a layer with occasional spikes), this clipping is often the difference between a clean conversion and a 3-point drop.

There is a small theory behind the histogram method worth internalizing, because it tells you *why* it beats min-max rather than just *that* it does. The entropy calibrator builds a fine histogram of the observed values, then for each candidate clipping threshold $T$ it forms the quantized distribution (everything beyond $T$ saturated, the rest binned into the 256 codes), dequantizes it back, and computes the Kullback-Leibler divergence between the original distribution $P$ and the quantized one $Q$:

$$D_{\mathrm{KL}}(P \,\|\, Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}.$$

It picks the $T$ that minimizes this divergence. Intuitively, it is choosing the clipping point that throws away the least *information* about the distribution's shape — sacrificing the rare tail values (which carry little probability mass) to give the dense bulk more codes (which carries most of the mass). Min-max, by contrast, refuses to throw away anything, so a single tail value with negligible probability mass dictates the entire scale. The KL view makes the trade explicit: you are trading a tiny amount of probability mass in the tail for a large gain in resolution over the bulk, and that is almost always the right trade for activations.

The **amount** is smaller than people expect. For a CNN, **128–512 samples** is plenty; the range estimate converges fast because you are estimating a couple of percentiles, not training anything. More samples rarely help and slow the calibration loop. The exception is class-imbalanced or multi-modal input distributions, where you want enough samples to see every mode.

| Calibration method | How it sets the range | Robust to outliers? | Cost | When to use |
| --- | --- | --- | --- | --- |
| Min-max | literal observed min/max | no — one outlier wrecks it | trivial | clean, bounded activations (post-ReLU6) |
| Percentile | clip to the 99.9–99.99th pct | yes — drops the tail | cheap | unbounded activations, mild outliers |
| Histogram/entropy (KL) | minimize KL of quantized dist | yes — information-optimal | moderate | the default for tricky activations |
| MSE | minimize squared error of round | yes | moderate | when you want L2-optimal, not KL |

```python
import torch
from torch.ao.quantization import prepare

# 1. Attach the qconfig and insert observers.
model.qconfig = qconfig
prepare(model, inplace=True)

# 2. Run representative data so observers see real activation ranges.
#    Use the TRAIN split (or a dedicated calib split), NOT val/test.
calib_set = torchvision.datasets.ImageFolder("imagenet/train_calib", preprocess)
calib_loader = DataLoader(calib_set, batch_size=32, shuffle=True)

model.eval()
with torch.inference_mode():
    seen = 0
    for x, _ in calib_loader:
        model(x)
        seen += x.size(0)
        if seen >= 256:          # 256 samples is plenty for a CNN
            break
```

#### Worked example: how many calibration samples is enough

I ran MobileNetV2 PTQ on ImageNet with histogram calibration at 32, 128, 256, and 1024 samples and measured top-1 on a fixed 5,000-image val subset. The numbers: 32 samples → 70.9%, 128 → 71.3%, 256 → 71.4%, 1024 → 71.4%. The fp32 baseline was 71.8%. The lesson is concrete: past ~256 samples the curve is flat, so calibration is not your bottleneck — at 256 samples you are 0.4 points from fp32 and another 4× of calibration data buys nothing. If you are still far from baseline at 256 samples, the problem is *not* calibration size; it is granularity, an outlier layer, or an op fallback, and you should move to attribution rather than throwing more data at the observers.

## 5. Station four: convert to int8 (PyTorch, ONNX Runtime, TFLite)

With the model prepared and calibrated, conversion is the easy part — one call in each toolchain. The discipline is to know what the call *produces* and to keep a numerically comparable fp32 reference around for the validation step.

### PyTorch eager-mode convert

```python
import torch
from torch.ao.quantization import convert

# `model` was prepare()'d and calibrated above.
int8_model = convert(model.eval(), inplace=False)

# Sanity: it now holds quantized weights + per-channel scales/zero-points.
print(int8_model)         # layers show as QuantizedConvReLU2d etc.
print("int8 size:", model_size_mb(int8_model), "MB")
print("int8 acc :", evaluate(int8_model, val_loader, max_batches=50))
torch.save(int8_model.state_dict(), "mobilenetv2_int8.pt")
```

PyTorch's eager-mode quantization is the fastest path to a number, but for deployment you will usually export to a runtime that has tuned int8 kernels for the target. The two most common are ONNX Runtime (great on x86 and ARM CPUs, broad operator coverage) and TFLite/LiteRT (the default for Android NNAPI and many NPUs).

### ONNX Runtime static quantization

ONNX Runtime quantizes a *float* ONNX graph directly. You export the fp32 model to ONNX, then run `quantize_static` with a `CalibrationDataReader` that feeds it representative inputs. This is often cleaner than the PyTorch path because the calibration and conversion are one tool, and ORT's quantizer applies per-channel weights by default.

```python
import numpy as np
import torch
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat)

# 1. Export fp32 model to ONNX (do this from the *unfused* fp32 model;
#    ORT does its own fusion + quantization).
fp32 = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2").eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(fp32, dummy, "mobilenetv2_fp32.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=17,
                  dynamic_axes={"input": {0: "batch"}})

# 2. A calibration reader yields {input_name: np.ndarray} dicts.
class ImageNetCalib(CalibrationDataReader):
    def __init__(self, loader, limit=256):
        self.data = []
        seen = 0
        for x, _ in loader:
            for i in range(x.size(0)):
                self.data.append({"input": x[i:i+1].numpy()})
                seen += 1
                if seen >= limit:
                    break
            if seen >= limit:
                break
        self.it = iter(self.data)
    def get_next(self):
        return next(self.it, None)

reader = ImageNetCalib(calib_loader, limit=256)

# 3. Static int8 quantization: per-channel weights, QDQ format
#    (QDQ keeps explicit Quantize/Dequantize nodes the runtime fuses).
quantize_static(
    "mobilenetv2_fp32.onnx",
    "mobilenetv2_int8.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
)
```

```bash
# Verify the int8 ONNX runs and measure it through onnxruntime.
python - <<'PY'
import onnxruntime as ort, numpy as np, time
sess = ort.InferenceSession("mobilenetv2_int8.onnx",
                            providers=["CPUExecutionProvider"])
x = np.random.randn(1, 3, 224, 224).astype("float32")
for _ in range(20): sess.run(None, {"input": x})   # warm up
t = []
for _ in range(200):
    t0 = time.perf_counter(); sess.run(None, {"input": x})
    t.append((time.perf_counter()-t0)*1e3)
t.sort(); print("p50 %.2f ms  p99 %.2f ms" % (t[len(t)//2], t[int(len(t)*0.99)]))
PY
```

### TFLite full-integer quantization

TFLite's converter wants a `representative_dataset` generator and the `Optimize.DEFAULT` flag; setting the inference input/output types to `int8` forces a *full-integer* model with no float fallback, which is what an integer-only NPU or microcontroller needs.

```python
import tensorflow as tf
import numpy as np

converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv2_savedmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    # ~256 representative samples, shaped like the model input.
    for x, _ in calib_numpy_iterator(limit=256):
        yield [x.astype(np.float32)]      # one input tensor, batch=1

converter.representative_dataset = representative_dataset
# Force FULL int8 (fail loudly if any op cannot be int8):
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
open("mobilenetv2_int8.tflite", "wb").write(tflite_int8)
```

That `supported_ops = [TFLITE_BUILTINS_INT8]` line is a deliberate choice: it tells the converter to **error out** rather than silently leave an unsupported op in float. The default behavior allows a mixed model where a couple of ops run in fp32, which is convenient but is exactly the silent "slower not faster" footgun — you think you shipped int8 and you shipped a model that dequantizes, runs one op in float, and requantizes around it. Forcing full-int8 surfaces the problem at convert time instead of at benchmark time.

### A note on PyTorch FX graph mode (the modern path)

The eager-mode `prepare`/`convert` shown above is the easiest to read, but it has a real limitation: it cannot automatically fuse and quantize across functional boundaries (a `+` in a residual, a `torch.cat`) because eager mode does not see the graph — it only sees modules. For production PyTorch quantization the modern path is **FX graph mode**, which traces the model to a graph, then fuses and quantizes it automatically, including the functional ops eager mode misses. The API mirrors eager mode but operates on the traced graph.

```python
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

fp32 = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2").eval()
qconfig_mapping = get_default_qconfig_mapping("qnnpack")
example_inputs = (torch.randn(1, 3, 224, 224),)

# Trace + insert observers (fusion happens automatically, incl. functionals).
prepared = prepare_fx(fp32, qconfig_mapping, example_inputs)

with torch.inference_mode():            # calibrate
    seen = 0
    for x, _ in calib_loader:
        prepared(x); seen += x.size(0)
        if seen >= 256: break

int8_fx = convert_fx(prepared)          # produces the quantized graph module
```

The practical reason to prefer FX graph mode: it gets the residual-add requantization right *for you*, which is the third footgun (differently-scaled merges) handled automatically instead of being a hand-coded landmine. The reason you still see eager mode everywhere is that it is easier to debug layer by layer and it composes cleanly with the per-layer attribution code in the next section. Use eager mode to *understand and debug*; use FX graph mode (or, increasingly, the `torch.export`-based PT2E flow that is replacing both) to *ship*.

## 6. Station five: validate — accuracy and the numerical diff

Conversion gives you an int8 model. Validation tells you whether it is *good*, and — this is the part most people skip — *where* it went wrong if it is not. There are two validation tools and you need both: end-to-end accuracy (the verdict) and a per-layer numerical diff (the diagnosis).

End-to-end accuracy is just `evaluate(int8_model, val_loader)` through the same harness as the baseline. If it is within budget, you are done; go to deploy. If it is not, the numerical diff is how you stop guessing.

### The numerical diff method

The idea is to run the *same input* through the fp32 model and the int8 model and compare their intermediate activations layer by layer. Where the two diverge is where quantization is hurting you. You hook every layer's output in both models, run a batch of real inputs, and compute, per layer, the **max absolute difference** and the **cosine similarity** between the fp32 and int8 activations. A healthy layer has a tiny max-abs diff and cosine similarity at 0.999+. A broken layer jumps out: a max-abs diff an order of magnitude larger than its neighbors and a cosine similarity that has fallen to 0.9-something.

Figure 4 shows this contrast — a healthy layer versus a broken one, with the broken one's range blown out by an outlier.

![A two-column before-after comparison of a healthy quantized layer with tiny activation difference against a broken layer whose range is inflated by an outlier and whose difference is large](/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-4.png)

```python
import torch

def collect_activations(model, x, names):
    """Run x through model, capture named submodule outputs."""
    acts, handles = {}, []
    def make_hook(n):
        def hook(_m, _inp, out):
            acts[n] = out.dequantize() if out.is_quantized else out
        return hook
    mods = dict(model.named_modules())
    for n in names:
        handles.append(mods[n].register_forward_hook(make_hook(n)))
    with torch.inference_mode():
        model(x)
    for h in handles:
        h.remove()
    return acts

# Layers present (by name) in BOTH the fp32 and int8 graphs.
layer_names = ["features.0", "features.7", "features.14",
               "features.18", "classifier.1"]

x, _ = next(iter(val_loader))
x = x[:8]                                   # a small real batch
a_fp32 = collect_activations(fp32, x, layer_names)
a_int8 = collect_activations(int8_model, x, layer_names)

print(f"{'layer':<14}{'max_abs':>10}{'rel':>10}{'cos':>10}")
for n in layer_names:
    f, q = a_fp32[n].flatten(), a_int8[n].flatten()
    max_abs = (f - q).abs().max().item()
    rel = max_abs / (f.abs().max().item() + 1e-9)
    cos = torch.nn.functional.cosine_similarity(f, q, dim=0).item()
    print(f"{n:<14}{max_abs:>10.4f}{rel:>10.4f}{cos:>10.4f}")
```

A real run of this on a deliberately mis-prepared MobileNetV2 (per-tensor weights, min-max calibration) produced a table where `features.18` had a max-abs diff of 1.84 and cosine 0.91 while every other layer sat below 0.07 and 0.999. That one number — 1.84 against a sea of 0.05s — is the entire diagnosis. You do not have a "quantization problem"; you have a `features.18` problem, and now you can go look at *why* that specific layer's activations or weights quantize badly.

### Per-layer error attribution

The numerical diff tells you where the *representation* error is large. The complementary method tells you where the *accuracy* impact is large, and they do not always agree (a layer can have a big activation diff that the rest of the network shrugs off, or a small diff at a layer the output is exquisitely sensitive to). Per-layer attribution quantizes the model **one layer at a time**, leaving every other layer in fp32, and measures end-to-end accuracy for each. The layer whose solo quantization costs the most accuracy is your culprit.

![A four-row matrix attributing accuracy drop and activation difference to individual layers, with one convolution flagged as the culprit and the rest marked fine](/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-3.png)

```python
import copy
import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert

def quantize_one_layer(fp32_model, target_name, calib_loader):
    """Quantize only `target_name`; everything else stays fp32."""
    m = copy.deepcopy(fp32_model).eval()
    # Disable quant everywhere, enable only on the target submodule.
    m.qconfig = None
    dict(m.named_modules())[target_name].qconfig = \
        get_default_qconfig("qnnpack")
    prepare(m, inplace=True)
    with torch.inference_mode():
        seen = 0
        for x, _ in calib_loader:
            m(x); seen += x.size(0)
            if seen >= 256: break
    return convert(m, inplace=False)

results = {}
for name in layer_names:
    m_q = quantize_one_layer(fp32, name, calib_loader)
    results[name] = evaluate(m_q, val_loader, max_batches=50)

base = evaluate(fp32, val_loader, max_batches=50)
for name, acc in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"{name:<14} acc {acc:6.2f}  drop {acc - base:+.2f}")
```

The output ranks layers by how much accuracy they cost when quantized alone. In the worked run, `features.18` cost -3.1 points solo while the others cost -0.2 to -0.4 each. That confirms the numerical diff: same layer, two independent methods, one culprit. Now the fix is targeted, which is the whole point — you are about to change one layer, not re-run the entire pipeline with a different global config and hope.

| Validation tool | Question it answers | Output | When you reach for it |
| --- | --- | --- | --- |
| End-to-end accuracy | is the int8 model good enough? | one number vs budget | always, the verdict |
| Numerical diff | where is representation error large? | per-layer max-abs / cosine | accuracy missed budget |
| Per-layer attribution | where is accuracy impact large? | per-layer solo-quant drop | to find the one bad layer |
| Mixed-precision fallback | can I keep one layer in fp16? | acc recovered, size cost | culprit found, surgical fix |

## 7. Station six: deploy and benchmark on the target

A int8 model that you have only ever run on your dev laptop is not yet a result. The size win is real (you can read it off disk), but the *speed* win is entirely a property of the target's kernels, and there are three ways for it to evaporate: the target lacks native int8 units (then int8 is a size-only win and may even be slower than fp16 if it has fast fp16); an op falls back to fp32 and inserts quantize/dequantize churn around it; or the model is memory-bound, in which case the 4× smaller weights help bandwidth but the compute speedup is muted.

So you benchmark on the target, the same honest way you measured the fp32 baseline: warm up, batch=1, report p50 and p99, and — this matters on phones and Jetsons — watch for **thermal throttling** by running long enough that the SoC heats up. A burst benchmark that finishes in two seconds reports the boost-clock latency you will never see in sustained use; run a few hundred iterations and look at whether the tail creeps up as the chip warms.

```bash
# ONNX Runtime: built-in perf tool, batch=1, reports percentiles.
onnxruntime_perf_test -e cpu -r 300 -m times \
    -I mobilenetv2_int8.onnx

# TFLite benchmark on an Android target via adb, with the NNAPI delegate
# so the NPU (not the CPU) runs the int8 model:
adb push mobilenetv2_int8.tflite /data/local/tmp/
adb shell /data/local/tmp/benchmark_model \
    --graph=/data/local/tmp/mobilenetv2_int8.tflite \
    --use_nnapi=true --num_threads=1 \
    --warmup_runs=50 --num_runs=300
```

#### Worked example: MobileNetV2 fp32 to int8, the full before-after

Here is the complete result for the running example — MobileNetV2, ImageNet top-1, measured through one harness, with latency on a Pixel-8-class mobile CPU at batch=1 (these are representative figures for this architecture and class of device; treat the latency as order-of-magnitude for *your* exact silicon and the accuracy deltas as the load-bearing numbers).

| Variant | Top-1 acc | Size (MB) | p50 latency | p99 latency |
| --- | --- | --- | --- | --- |
| fp32 baseline | 71.8% | 13.6 | 41 ms | 58 ms |
| int8, per-tensor, min-max (naive) | 67.1% | 3.5 | 16 ms | 24 ms |
| int8, per-channel, histogram | 71.4% | 3.5 | 16 ms | 23 ms |
| int8 + `features.18` kept in fp16 | 71.7% | 3.6 | 17 ms | 25 ms |

Read that table top to bottom and the entire post is in it. The naive int8 conversion gives the same 3.9× size win and 2.5× speedup as the careful one, but it *throws away 4.7 points of accuracy* — and a team that did not measure or attribute would either ship that or give up on int8 entirely. Switching to per-channel weights and histogram calibration recovers almost all of it in one config change: now you are 0.4 points down. And the final row is the surgical fp16 fallback on the one culprit layer, which buys back another 0.3 points for a negligible 0.1 MB and ~1 ms cost. The fp16-fallback fix is the bridge to mixed precision and sensitivity analysis, which generalizes this "keep the worst layers in higher precision" idea into a systematic sensitivity sweep.

The decision the table makes for you: ship the per-channel histogram int8 model (71.4%, 3.5 MB, 16 ms) if your accuracy budget is 1 point, or the fp16-fallback variant (71.7%, 3.6 MB, 17 ms) if it is tighter than 0.5 points. Either way you have a 2.5× speedup and a 3.9× size cut for, at worst, 0.4 points — a textbook good trade on the accuracy-latency Pareto frontier.

### Stress-testing the pipeline: where does it break?

The before-after table is the happy result. The mark of someone who has actually shipped this is knowing how each assumption fails when you push on it. Four stress tests, each of which I have watched surprise a team.

**Push to int4.** The SQNR math is unforgiving here: int4 has a ceiling of roughly $6.02 \times 4 \approx 24$ dB, half the int8 budget, so the rounding noise is large enough that the network genuinely cannot absorb it by accident. PTQ int4 on MobileNetV2 typically drops several points and sometimes falls off a cliff (10+ points) because depthwise channels with wide ranges run out of codes entirely. The fixes that work at int4 are *not* the int8 fixes scaled down — they are different in kind: QAT (train the model to tolerate the noise), or weight-only int4 with activations kept at int8/fp16 (the LLM trick — quantize the thing that is plentiful and tolerant, keep the thing that is sensitive). The lesson: int8 is a "fix the range" game; int4 is a "retrain or keep activations high" game. Do not assume the int8 playbook extends.

**Shrink the calibration set to almost nothing.** The worked example showed 256 samples is plenty and 32 is close. But what happens at *4* samples, or *1*? The range estimates become high-variance and biased — with one image you see the activation ranges for that one image, which under-estimates the true production range, so at inference time real inputs saturate the clamp and you lose accuracy in a way that *looks* like an outlier problem but is actually a coverage problem. The tell is that accuracy is unstable across calibration runs (different random samples give different accuracy). The fix is more samples, but the deeper lesson is that calibration size has a *floor* set by how multi-modal your inputs are, not a universal number.

**The NPU does not support an op.** You ship a full-int8 TFLite model to a phone, enable the NNAPI delegate, and the latency is *worse* than the CPU fp32 baseline. What happened: the NPU supports int8 conv and int8 add but not your fancy activation, so the delegate runs the conv on the NPU, copies the int8 tensor back to the CPU, dequantizes, runs the activation in fp32, requantizes, copies back to the NPU. Every fallback is a round trip across the NPU/CPU boundary, which is *enormously* expensive — the data movement dwarfs the compute you saved. The tell is that the profiler shows the model split across delegates with copies between them; the fix is to swap the unsupported op for a supported one (e.g. ReLU6 instead of an exotic activation) or to accept the fallback only if it is off the critical path. This is the single most common "int8 made it slower" cause on phones.

**The layer is memory-bound, not compute-bound.** Int8 makes the *compute* 4× cheaper (more MACs per cycle on int8 units) and the *weights* 4× smaller. If a layer is compute-bound, int8 gives you the full speedup. But if a layer is memory-bound — its time is dominated by reading activations from DRAM, not by arithmetic — then making the arithmetic 4× cheaper does little, because you are still waiting on memory. The 4× smaller *weights* help (less weight traffic), but if the bottleneck is *activation* traffic (large feature maps, small kernels — the early layers of a CNN, or attention with a long KV cache), int8 weight quantization barely moves the wall clock. The roofline model is the tool that tells you which regime each layer is in before you assume int8 buys speed; quantizing a memory-bound-on-activations layer is spending accuracy for almost no latency. This is why "int8 the whole model" can yield a 1.4× speedup when the FLOP count predicted 4× — the memory-bound layers did not cooperate.

The unifying point across all four: the int8 *size* win is robust (it is just bytes on disk), but the int8 *speed* win is conditional — on having int8 units, on every op being supported, and on the layer being compute-bound. Profile before you promise a number.

## 8. The debugging playbook (the part you actually came for)

Everything above is the happy path. This section is what you do when the int8 accuracy comes back wrong, and it is the highest-leverage thing in the post. The structure is a triage tree: start with the cheapest, most common global causes, and only descend into per-layer surgery when those are ruled out. Figure 2 is that tree.

![A decision tree for debugging an int8 accuracy drop that flows from checking per-channel weights to checking outliers to per-layer attribution to keeping the worst layer in fp16](/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-2.png)

Walk it top to bottom:

**Step 1 — Are weights per-channel?** This is the first thing to check because it is the most common cause of a *big* drop and the cheapest fix. If you quantized weights per-tensor, switch to per-channel and re-convert. On MobileNetV2 this single change took the running example from 67.1% to ~71% — it recovered nearly the entire drop by itself. If you are using a backend default (`get_default_qconfig("qnnpack")` or ORT's `per_channel=True`), you are probably already here; if you hand-rolled a `QConfig`, check it.

**Step 2 — Are there outliers in the calibration stats?** Pull the observed min/max (or the histogram) for each activation and look for a layer whose range is wildly wider than its weights or neighbors suggest. A range of [-1, 27] on a layer whose bulk lives in [-2, 3] is an outlier blowing out the scale. The fix is percentile/histogram clipping (keep the 99.99th percentile, saturate the tail) instead of min-max, or for transformers the SmoothQuant trick of migrating activation outliers into the weights where they quantize more gracefully. This is the activation-range cause and it bites hardest on unbounded activations — GELU and Swish/SiLU have no upper clamp, so a few large pre-activation values produce huge post-activation outliers.

**Step 3 — Check the first and last layers.** The input layer sees raw pixel/token statistics and the final classifier/logit layer's small differences decide the argmax; both are disproportionately sensitive. A standard and cheap move is to keep the first conv and the final classifier in higher precision (fp16 or even fp32) and quantize everything in between. The accuracy cost of leaving two layers in float is tiny because they are a small fraction of the FLOPs, and the accuracy *recovery* is often large.

**Step 4 — Check BN folding.** If you did not fold conv-bn-relu before calibrating, your observers saw the wrong ranges and you have a quantize/dequantize round trip mid-op. Confirm fusion happened (in PyTorch the layers should show as fused types; in ONNX/TFLite the converter does it but verify there is no stray BN in the int8 graph).

**Step 5 — Check the calibration set.** Is it representative? Is it the *right split* (train/calib, not test)? Is it large enough (≥128 for a CNN)? Is it accidentally all one class or one lighting condition? A non-representative calibration set produces ranges that are wrong for production inputs and an accuracy drop that *moves* when you change the calib set — that movement is the tell.

**Step 6 — Attribute and fall back.** If the global fixes have not closed the gap, run the numerical diff and per-layer attribution from section 6 to find the one or two culprit layers, then keep *those specific layers* in fp16. This mixed-precision fallback is the surgical end of the playbook: you accept a small size/speed cost on a few layers to recover the accuracy, and it almost always works because quantization error is rarely spread evenly — it concentrates in a handful of sensitive layers.

#### Worked example: a 4.7-point drop, fixed in twenty minutes

The narrative, end to end, because the *sequence* is the skill. You convert MobileNetV2 and get 67.1% against a 71.8% baseline — a 4.7-point drop, far outside a 1-point budget. **Step 1:** you check the `QConfig` and find per-tensor weights. You switch to per-channel symmetric weights and re-convert: 70.9%. That one change recovered 3.8 of the 4.7 points in about five minutes. **Step 2:** still 0.9 short, you dump the calibration histograms and see `features.18` with a min-max range of [-1.0, 27.0] from a handful of outlier activations. You switch from min-max to histogram (99.99th percentile) calibration: 71.4%. Now you are 0.4 points down, inside budget. You *could* stop here. **Step 6 (optional):** the spec wanted under 0.5, you are at 0.4, fine — but say the budget were 0.3. You run the numerical diff (features.18: max-abs 1.84, cosine 0.91, everything else clean) and per-layer attribution (features.18 solo: -3.1 points), confirm the culprit twice, and keep `features.18` in fp16. Result: 71.7%, a 0.1-point drop, at a cost of 0.1 MB and ~1 ms. Total wall-clock: under twenty minutes, three config changes, zero retraining. That is what the playbook buys you over "the int8 model is bad, let's QAT it" — which would have cost a training run to fix what two config flags fixed.

Figure 5 is the same logic as a lookup table: symptom on the left, the likely cause, the fix.

![A matrix mapping four common int8 symptoms to their likely cause and a concrete fix, covering global drops, single bad layers, edge layers, and unexpected slowdowns](/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-5.png)

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Big global accuracy drop (3%+) | per-tensor weights | per-channel symmetric weights |
| One layer dominates the error | activation outliers in that layer | histogram/percentile clip, or fp16 fallback |
| First/last layer disproportionate | input or logit sensitivity | keep first conv + final fc in fp16 |
| Drop moves when calib set changes | non-representative calibration | representative train/calib split, ≥128 samples |
| Worse than int4 would be | double quantization | quantize once; check for stacked Q/DQ |
| "Slower not faster" on target | op falling back to fp32 | force full-int8, swap to a supported op |
| Great offline, bad in production | calibration data leakage | calib from train split, never test |

## 9. Common footguns (the silent corruptions)

These are different from the accuracy-drop symptoms above: footguns often do *not* announce themselves with a clean accuracy number. They produce a model that is subtly wrong, or wrong only on some inputs, or wrong in a way that looks like a different bug. Figure 7 catalogs them; here is what each one is and how to catch it.

![A matrix of six int8 footguns describing what each does and the detectable signature it leaves, including double quantization, wrong per-channel axis, and calibration leakage](/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-7.png)

**Double quantization.** Quantizing an already-quantized tensor — easy to do when you compose tools (export an int8 model, then run a quantizer over it again) or stack Quantize/Dequantize nodes by accident. The rounding noise compounds, and the tell is that accuracy is *worse than int4 should be* even though you nominally have int8. Inspect the graph for back-to-back Q/DQ pairs on the same tensor.

**Wrong axis for per-channel.** For a conv weight `[out, in, kh, kw]`, scales belong on axis 0 (output channels); for a linear weight `[out, in]`, also axis 0. Put them on the wrong axis and each "channel" scale applies to the wrong slice of weights — the result is a per-layer numerical diff that is enormous and physically nonsensical. The tell is a single layer with a huge diff right after you hand-configured per-channel; the fix is to confirm `ch_axis` matches the weight layout.

**Concat/add of differently-scaled tensors.** When two int8 tensors with *different* scales feed an add or a concat (residual connections, feature-pyramid merges), you cannot just add the int8 codes — they live in different units. A correct implementation requantizes both to a common scale first; a broken one adds raw codes and silently corrupts the merge. The tell is that accuracy collapses specifically on architectures with residuals or concats, and the fix is to ensure the runtime inserts requantize nodes at the merge (good runtimes do; hand-rolled int8 kernels often forget).

**Unsupported op falling back to fp32.** Your target runtime does not have an int8 kernel for some op (a fancy activation, a custom layer, an uncommon pooling), so it dequantizes to fp32, runs the op, and requantizes — inserting two conversions and a float compute into your "int8" model. Accuracy is fine but speed is *worse* than you expected, because each fallback adds memory traffic. The tell is the "slower not faster" symptom; the fix is to profile for fallbacks (ORT and TFLite both report which ops ran on which delegate) and either swap the op for a supported one or accept the fallback knowingly.

**Calibration data leakage.** Covered above but it earns a footgun slot because it is silent: the model scores *better* offline, so nothing looks wrong until live accuracy disappoints. The tell is an offline-vs-live gap; the fix is strict split hygiene — calibration from train/calib, evaluation from a held-out test set, never the two overlapping.

**Evaluating on the wrong split.** The mirror image: you evaluate the int8 model on a *different* split than you evaluated fp32, and your "accuracy drop" is partly a different test set. The tell is a drop that does not reproduce when you fix the split. This is why station zero insists on one `evaluate` function for both models.

## 10. The LLM quick path: GGUF Q4_K_M as a contrast

Everything so far is the CNN/vision pipeline. Large language models follow the *same discipline* — measure, choose granularity, calibrate, convert, validate, deploy — but with completely different tooling and a different default sweet spot. The contrast is instructive precisely because the principles transfer while the tools do not.

For local LLM inference, the dominant path is `llama.cpp` and its GGUF format with **k-quants**. The headline format is **Q4_K_M**: a 4-bit-ish mixed scheme that keeps the most sensitive tensors (attention output and feed-forward down projections) at higher precision while quantizing the bulk to 4 bits, with a super-block structure that stores per-block scales. It is the GGUF equivalent of "keep the sensitive layers higher precision" — the same idea as the fp16 fallback above, baked into the format. The discipline is identical even though there is no PyTorch `QConfig` in sight.

```bash
# 1. Convert an HF model to GGUF fp16 (the "baseline" artifact).
python convert_hf_to_gguf.py ./Llama-3.2-3B-Instruct \
    --outfile llama-3.2-3b-f16.gguf --outtype f16

# 2. Quantize to Q4_K_M (4-bit k-quant, medium variant).
./llama-quantize llama-3.2-3b-f16.gguf \
    llama-3.2-3b-Q4_K_M.gguf Q4_K_M

# 3. Run it and measure tokens/s on the target (offload all layers to GPU
#    with -ngl 99, or leave on CPU for a laptop/edge box).
./llama-cli -m llama-3.2-3b-Q4_K_M.gguf \
    -p "Explain int8 quantization in one sentence." \
    -n 128 -ngl 99
```

How do you *validate* an LLM quantization? Not top-1 accuracy — you measure **perplexity** on a held-out text set (the LLM analog of the accuracy metric) and, ideally, a few downstream task scores. `llama.cpp` ships a perplexity tool for exactly this:

```bash
# Measure perplexity delta vs the fp16 baseline on a held-out corpus.
./llama-perplexity -m llama-3.2-3b-f16.gguf  -f wiki.test.raw   # baseline
./llama-perplexity -m llama-3.2-3b-Q4_K_M.gguf -f wiki.test.raw  # quantized
```

#### Worked example: a 3B LLM, fp16 to Q4_K_M, on a laptop

Representative numbers for a 3B-parameter model quantized fp16 → Q4_K_M and run on an M2-class laptop CPU (treat as order-of-magnitude; exact figures depend on the model, the corpus, and the build):

| Variant | Size (GB) | Perplexity | tokens/s (laptop CPU) | Peak RAM |
| --- | --- | --- | --- | --- |
| fp16 GGUF | ~6.0 | baseline | ~9 tok/s | ~6.4 GB |
| Q4_K_M GGUF | ~1.9 | +0.1 to +0.3 ppl | ~22 tok/s | ~2.3 GB |

The shape of the win is the same as the CNN: ~3.2× smaller, ~2.4× faster, for a small quality cost (a fraction of a perplexity point, which for most chat use is imperceptible). And the *discipline* is the same: you kept the sensitive tensors higher precision (Q4_K_M does this for you), you validated against a real baseline metric (perplexity), and you benchmarked on the actual target (the laptop, not a server GPU). What changed is everything *mechanical* — GGUF instead of a PyTorch state dict, k-quants instead of per-channel symmetric, perplexity instead of top-1, `llama-cli` instead of `onnxruntime_perf_test`. The series' deeper treatment of int8-vs-int4 LLM trade-offs lives in the out-of-series note on [quantization int8/fp16/int4 edge trade-offs](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs).

One genuinely different wrinkle the LLM path forces you to confront is that transformer *activations* have far worse outliers than CNN activations. Empirically, a small number of feature dimensions in large transformers carry values 10-100× the rest, and those dimensions appear consistently across tokens. Min-max calibration on such activations is hopeless, and even percentile clipping struggles because clipping a *systematic* large dimension throws away real signal, not noise. This is exactly why the LLM toolchain quantizes *weights* aggressively (to 4 bits) but keeps *activations* at 8 bits or 16 bits, and why methods like SmoothQuant exist to migrate the activation outliers into the weights. It is the same diagnosis you would reach with the numerical diff on a CNN — "this layer's activation range is blown out by outliers" — applied at scale, with a format-level rather than a config-level fix. The triage tree still works; you just resolve step 2 (outliers) by choosing a quantization *scheme* that keeps the outlier-bearing tensor in higher precision, rather than by clipping.

The transfer is the lesson: if you internalize the six-station pipeline and the triage tree on the CNN, you already know how to quantize an LLM, an audio model, or a detector. Only the tool names change.

## 11. Case studies and real numbers

A few results from the literature and shipped products that ground the techniques above, with sources, so the numbers in this post sit in a known landscape.

**MobileNetV2/V3 int8 on mobile.** The original MobileNetV2 paper (Sandler et al., 2018) and Google's quantization white paper (Krishnamoorthi, 2018) establish that depthwise-separable nets are the *hard* case for int8 PTQ precisely because of per-channel weight spread, and that per-channel quantization plus careful calibration is what makes int8 MobileNet viable. The practical upshot — per-channel is non-negotiable for these architectures — is exactly the running example's biggest lever.

**The integer-only quantization recipe.** Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," is the paper that defined the symmetric-weights/asymmetric-activations, fold-BN, integer-accumulator recipe that PyTorch, TFLite, and ONNX Runtime all implement. When this post says "the standard recipe," that is the paper it comes from.

**The PTQ/QAT survey.** Nagel et al. (2021), "A White Paper on Neural Network Quantization," is the single best reference for everything in this post: it derives the error model, lays out per-tensor vs per-channel, covers PTQ calibration methods and QAT with the straight-through estimator, and gives a debugging-oriented view of where accuracy goes. If you read one thing after this post, read that.

**LLM 4-bit: GPTQ and AWQ.** For the LLM path, Frantar et al. (2022), "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," and Lin et al. (2023), "AWQ: Activation-aware Weight Quantization," are the methods behind near-lossless 4-bit LLMs. Both are sophisticated PTQ: GPTQ uses a second-order error-correction sweep, AWQ scales weights by activation importance. They are the LLM analog of "per-channel plus clipping" — clever calibration, not retraining — and they are why a 4-bit LLM can be within a fraction of a perplexity point of fp16.

**SmoothQuant for transformer activations.** Xiao et al. (2022), "SmoothQuant," addresses the activation-outlier problem this post raises in step 2 of the playbook, by mathematically migrating the outlier magnitude from activations into weights (which quantize more gracefully) via a per-channel scaling that preserves the product. It is the principled version of "clip the outliers" for transformers.

The through-line across all of them: the wins come from *where* you spend precision (per-channel, sensitive layers higher, outliers handled), not from a magic conversion call. Every one of these methods is, at bottom, a smarter answer to the questions this pipeline asks at the prepare and calibrate stations.

## 12. When to reach for this (and when not to)

Int8 PTQ is the *first* optimization to try for almost any edge deployment, because it is cheap, it is well-supported, and it delivers a reliable ~4× size and ~2–4× speed win on hardware with int8 units. But it is a cost, and there are times it is the wrong call.

**Reach for int8 PTQ when:** you have a trained fp32 model, your target has native int8 (essentially every modern phone NPU, Jetson, and most edge accelerators), and you have a representative calibration set. This is the default. Try it before anything fancier.

**Reach for QAT only when** PTQ-with-per-channel-and-clipping misses your accuracy budget after you have done the attribution. QAT costs a training run; do not spend it speculatively. The attribution from PTQ tells you which layers QAT must focus on, so PTQ is *always* worth running first even if you suspect you will end up at QAT.

**Reach for mixed-precision fallback when** one or two layers carry most of the error — keep them in fp16, quantize the rest. This is the surgical default for the last fraction of a point and it composes with everything.

**Do NOT bother with int8 when:** your target has *no* int8 units and fast fp16 (some GPUs) — int8 there is a size-only win and may be slower; ship fp16 instead. **Do NOT** quantize a model that is firmly memory-bound and already small enough — the speed win from int8 compute will be muted because you are waiting on memory, and you have spent accuracy for little latency. The roofline tells you which regime you are in; check it before you assume int8 buys speed. **Do NOT** push to int4 on a CNN with PTQ and expect it to hold — int4 on vision models generally needs QAT, and even then the accuracy cliff is steep; int4 is an LLM-weight game (where activations stay higher precision), not a general CNN move.

**When to stop optimizing:** the moment you are inside your accuracy budget and meeting your latency and size targets on the actual hardware. There is always another half-point to chase and another technique to stack, and almost always it is not worth it. A model that hits 71.4% at 16 ms and 3.5 MB against a 71.8% / 41 ms / 13.6 MB baseline is *done* — shipping it beats spending three more days to recover 0.3 points nobody will notice. The discipline of stopping is as important as the discipline of measuring.

## 13. The pre-flight checklist

Before you ship an int8 model, walk these gates. Figure 8 is the same list as a stack; this is the operational version.

![A vertical stack of the int8 pre-flight checklist from a logged baseline through calibration, validation, footgun checks, and an on-target benchmark](/imgs/blogs/quantization-in-practice-a-full-int8-pipeline-8.png)

- **Baseline logged.** fp32 accuracy, latency (p50/p99, warm, batch=1), and size measured through the exact harness the int8 model will use.
- **Calibration from the right split.** Train/calib split, never test; representative of production inputs; ≥128 samples for a CNN; histogram/percentile method if there are outliers.
- **Granularity correct.** Per-channel symmetric weights, per-tensor asymmetric activations (or your backend's default that encodes this); `ch_axis` matches the weight layout.
- **BN folded.** conv-bn-relu fused before calibration; no stray BN or mid-op Q/DQ in the int8 graph.
- **Accuracy within budget.** End-to-end accuracy through the frozen harness; if not, you ran the numerical diff and attribution and applied the targeted fix.
- **No silent footguns.** No double quantization; no fp32 op fallbacks (or they are known and accepted); concat/add merges requantize correctly; no calibration leakage.
- **Benchmarked on the target.** Real device, warm, batch=1, p50 and p99, long enough to see thermal throttling; the speed win is *confirmed*, not assumed.
- **Artifact and metadata saved.** The int8 model, the calibration set hash, the config, and the before-after table, so the result is reproducible and the next person does not re-debug it.

If every box is checked, you have not just *a* quantized model — you have a quantized model you can defend, with a paper trail from baseline to benchmark. That is the difference between a science project and a shipped feature.

## Key takeaways

- **Measure the baseline through one harness first.** Every int8 number is a *difference*; an unmeasured or mismatched baseline makes the difference meaningless. Same `evaluate`, same `benchmark`, same split for fp32 and int8.
- **Per-channel symmetric weights are the biggest single lever** for CNNs, especially depthwise-separable ones, and they are nearly free. If you remember one fix, it is this.
- **PTQ first, time-boxed; QAT only when it misses.** Per-channel plus histogram calibration lands most well-behaved nets within ~1% of fp32. Do not spend a training run you do not need.
- **The numerical diff and per-layer attribution turn a vague drop into one culprit layer.** Max-abs/cosine localizes representation error; solo-quant localizes accuracy impact; together they point at the same one or two layers to fix.
- **Mixed-precision fallback is the surgical end-game.** Quantization error concentrates; keep the worst layer(s) in fp16 and recover most of the loss for a tiny size/speed cost.
- **The footguns are silent.** Double quantization, wrong per-channel axis, differently-scaled merges, op fallbacks, and calibration leakage do not always show as a clean accuracy number — check for them deliberately.
- **The speed win is a property of the target, not the conversion.** Benchmark on the real device, warm, batch=1, with the tail and thermal throttling in view. No int8 units or memory-bound regime means int8 may not speed anything up.
- **Stop when you hit budget.** Inside your accuracy, latency, and size targets on the real hardware? Ship it. The next half-point is almost never worth the days it costs.
- **The discipline transfers; the tooling does not.** CNN int8 or LLM Q4_K_M, the six stations and the triage tree are the same — only `QConfig` vs k-quants, top-1 vs perplexity, `onnxruntime_perf_test` vs `llama-cli` change.

## Further reading

- **Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"** — the paper that defined the standard int8 recipe (symmetric weights, asymmetric activations, BN folding, integer accumulators).
- **Krishnamoorthi (2018), "Quantizing deep convolutional networks for efficient inference: A whitepaper"** — Google's practical guide; the canonical case for per-channel on MobileNet-style nets.
- **Nagel et al. (2021), "A White Paper on Neural Network Quantization"** — the best single reference for the error model, PTQ/QAT methods, and debugging; read this next.
- **Frantar et al. (2022), "GPTQ"** and **Lin et al. (2023), "AWQ"** — near-lossless 4-bit LLM quantization via smart PTQ.
- **PyTorch quantization docs** (`torch.ao.quantization`), **ONNX Runtime quantization docs** (`quantize_static`, `CalibrationDataReader`), and the **TFLite/LiteRT post-training quantization guide** (`representative_dataset`, `Optimize.DEFAULT`) — the official references for the three toolchains in this post.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) and [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) for the deep mechanics, [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for honest measurement, and [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for how this stacks with the other levers.

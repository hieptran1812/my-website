---
title: "Case study: real-time vision on-device, from a fat model to a shipped one"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Take one concrete object detector from a 120 ms cloud model to a shipped 22 ms artifact on a phone and a Jetson Orin Nano, applying every lever in the series in order and measuring the before and after at each step."
tags:
  [
    "edge-ai",
    "model-optimization",
    "object-detection",
    "computer-vision",
    "quantization",
    "tensorrt",
    "tflite",
    "inference",
    "efficient-ml",
    "case-study",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/case-study-real-time-vision-on-device-1.png"
---

The demo looked great in the room. A detector trained on our internal furniture dataset, evaluated on a held-out set, hitting 37.4 mAP — comfortably ahead of the off-the-shelf baselines we had benchmarked against. The product team wanted it inside the camera tab of the app: point your phone at a room, get live bounding boxes around the couch, the lamp, the rug, so the AR feature could anchor virtual furniture next to the real thing. "It already works," someone said, gesturing at the laptop. "We just need to put it in the app."

Then we measured it on a phone. The same model, the same weights, ran at roughly **120 milliseconds per frame** on a mid-range Android device — about **8 frames per second**. The camera preview ran at 30. So every box the model drew lagged a quarter of a second behind the world, the preview hitched every time inference fired, and after a minute of pointing the phone around, the device was warm enough that the frame rate sagged further. The 98 MB float model also blew past the install-size budget the moment marketing saw the cellular-download warning. The feature, as built, was unshippable. Not "needs polish" unshippable — "does not function as a live experience" unshippable.

This post is the whole journey from that model to a shipped one: a single concrete detector, taken from the fat cloud artifact to something that runs at **45 FPS on the phone and clears a sustained-capture thermal test on a Jetson Orin Nano**, inside a 50 MB budget, with accuracy held within about a point of the original. The brief was specific and is worth stating as numbers, because numbers are what we will measure against at every step: **at least 20 FPS** (so **≤ 50 ms per frame**), **≤ 50 MB** model size, and **no thermal meltdown** under two minutes of continuous capture. The figure below is the entire trip on one timeline — baseline, architecture, distillation, pruning, quantization, compile, deploy — and every box on it is a place we measured before moving on.

![A seven-stage timeline showing a detector moving from an fp32 baseline at 120 ms through an efficient backbone, distillation, structured pruning, int8 quantization, compilation, and finally a shipped on-device build at 28 ms p99](/imgs/blogs/case-study-real-time-vision-on-device-1.png)

By the end you will be able to do this yourself: profile a vision model honestly on the target before changing anything, swap to a hardware-friendly backbone without throwing away accuracy, distill the original big model into the small one to win back what the swap cost, prune for a speedup the hardware can actually use, quantize to int8 with the care a detection head needs, compile and fuse for the real runtime on the real chip, and read the final result off the accuracy-latency frontier. We will keep the recurring spine of this series in view the whole way: the four levers (quantization, pruning, distillation, efficient architecture) sit on **compilers and runtimes**, validated by **profiling**, read off the **accuracy-efficiency Pareto frontier** — see [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the map and [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for the capstone that ties the whole series together. This post is where the map becomes one concrete shipped artifact.

A note before we start. Every number in this post is **representative** — drawn from what these levers typically deliver on these classes of model and these chips, anchored to public benchmarks where I can and labeled as approximate where I cannot. They tell a coherent story that matches reality, but your exact deltas will depend on your dataset, your backbone, and your driver stack. Treat them as the shape of the answer, not a guarantee of the digits.

## The running scenario, pinned down

Let me make the scenario concrete enough that the code later is unambiguous.

The model is a single-stage anchor-based object detector in the YOLO/SSD family: a convolutional **backbone** that turns the image into a feature pyramid, a small **neck** that fuses scales, and a **detection head** that, at each spatial location and anchor, predicts a class distribution and four box-regression offsets. The input is a 640×640 RGB frame. The training set is a 12-class indoor-furniture dataset; the metric is **mAP** (mean Average Precision at IoU 0.5:0.95, the COCO-style metric), where higher is better and a point or two is a meaningful amount of accuracy. The baseline backbone is a heavyweight one — think a ResNet-50-scale or CSPDarknet-scale feature extractor — chosen during research because accuracy was the only thing anyone was optimizing.

The targets are two real devices, because "the edge" is not one thing:

- A **mid-range Android phone** (a Snapdragon-class SoC with a Hexagon NPU and an Adreno GPU). This is the primary target — it is what most users will hold.
- A **Jetson Orin Nano** (8 GB, a small Ampere GPU with a couple of hundred dense TOPS of int8 and a handful of TFLOPS of fp16). This is the secondary target — a kiosk / robotics form factor where the same feature runs continuously, which makes thermals the dominant concern.

The brief, restated as a budget we will check at every step:

| Constraint | Target | Why it exists |
| --- | --- | --- |
| Latency per frame | ≤ 50 ms (≥ 20 FPS) | Live AR overlay must track the camera, not lag it |
| Model size on disk | ≤ 50 MB | Cellular-download warning threshold for app updates |
| Peak runtime memory | ≤ ~400 MB | Coexist with the camera pipeline and the rest of the app |
| Sustained thermals | No throttle in 2 min | Continuous capture is the actual use, not a single shot |
| Accuracy | mAP within ~2 points of 37.4 | Product will not accept a noticeably worse detector |

Notice the accuracy constraint is a *budget*, not "lose nothing." That framing is the whole game. We are allowed to spend up to two points of mAP to buy a 5× speedup and a 4× size cut — but not a point more, and we want to spend as little as we can. Every lever below is judged by how much accuracy it costs per millisecond it buys.

## Step 0: measure honestly before you touch anything

The first mistake in every optimization project is optimizing the wrong thing because you never measured. Before changing a single layer, we profile the baseline on the actual target, batch=1, warm-started, the way a user experiences it. This is not optional throat-clearing — it determines which lever to reach for first. (The full discipline lives in [profiling and benchmarking on device](/blog/machine-learning/edge-ai/profiling-and-benchmarking-on-device); here we apply it.)

Two things wreck naive measurements and both inflate or deflate your numbers in ways that matter.

**Warm-up.** The first inference after a model loads is not representative. The runtime is JIT-compiling kernels, allocating the workspace, paging in weights, and possibly selecting an execution provider. On a phone GPU delegate the first call can be 5–10× the steady-state latency. So you run 20–50 warm-up iterations, throw them away, and only then measure. If you report the cold number you scare yourself; if you only ever measure cold you ship a model whose first frame freezes the camera for 400 ms.

**Thermals.** A single inference, or a tight loop for two seconds, runs at boost clocks. Run the same loop for two minutes and the SoC throttles to stay under its thermal limit, and your p50 climbs 20–40%. Since our actual use is *continuous* capture, the steady-state-under-heat latency is the number that matters, not the cold-room sprint.

Here is the honest benchmark loop. It separates capture, preprocess, inference, and postprocess so we can see *where* the time goes, not just the total — which turns out to be the single most important measurement in this whole post.

```python
import time
import numpy as np

def bench_stage(fn, *args, warmup=30, iters=200):
    """Return (p50, p90, p99) milliseconds for one stage, warm-started."""
    for _ in range(warmup):
        fn(*args)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    samples = np.array(samples)
    return (np.percentile(samples, 50),
            np.percentile(samples, 90),
            np.percentile(samples, 99))

# Each fn is one stage of the pipeline, timed in isolation on the device.
frame = grab_frame()                      # camera -> np.uint8 HxWx3
print("capture   ", bench_stage(grab_frame))
print("preprocess", bench_stage(preprocess, frame))
x = preprocess(frame)
print("inference ", bench_stage(model_infer, x))
raw = model_infer(x)
print("nms+decode", bench_stage(postprocess, raw))
```

Running this on the baseline fp32 model on the phone gave the honest decomposition. Inference dominated — about **108 ms of the 120 ms** was the model forward pass, with preprocess, NMS, and copies eating the rest. The model was overwhelmingly **compute-bound**: the backbone's convolutions were saturating the available arithmetic units, and the device was nowhere near its memory-bandwidth ceiling. That single fact — compute-bound, backbone-dominated — is what tells us where to start. When the limiter is FLOPs in the backbone, the highest-leverage first move is **architecture**, not quantization. Quantizing a model whose problem is "too many multiply-accumulates in the wrong backbone" just gives you a smaller version of a model that is still doing too much work.

If you want the theory of *why* a model is compute-bound versus memory-bound and how to read it off arithmetic intensity, that is the roofline model — see [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives). The one-line version: a layer is compute-bound when it does many arithmetic operations per byte of memory traffic, and memory-bound when it does few. The baseline backbone is dense 3×3 convolutions on fat channel counts — high arithmetic intensity, classic compute-bound. We will watch that change as we strip work out, and by the end the bottleneck will have *moved*, which changes which levers still help.

Let me make "compute-bound versus memory-bound" quantitative for this specific model, because the rest of the post leans on it. The **arithmetic intensity** $I$ of a layer is the ratio of arithmetic operations it performs to the bytes it must move:

$$I = \frac{\text{FLOPs}}{\text{bytes moved}}\quad[\text{ops/byte}]$$

A device has a peak compute rate $P_{\text{compute}}$ (ops/s) and a peak memory bandwidth $B$ (bytes/s). The achievable rate is the lower of "what compute allows" and "what bandwidth allows," which is the roofline:

$$P_{\text{achievable}} = \min\!\big(P_{\text{compute}},\ I \cdot B\big)$$

A layer is **compute-bound** when $I > P_{\text{compute}} / B$ (the *ridge point*) and **memory-bound** below it. For our phone SoC, say $P_{\text{compute}} \approx 4$ TOPS int8 and $B \approx 25$ GB/s; the ridge is at $I = 4\times10^{12} / 25\times10^9 \approx 160$ ops/byte. A fat dense $3\times3$ convolution on 256→256 channels has an intensity in the *hundreds* of ops/byte — well above the ridge, firmly compute-bound, which is why the baseline backbone saturates the ALUs. A **depthwise** convolution, by contrast, does only $k^2 = 9$ multiply-adds per input element it reads, giving an intensity of *single digits* of ops/byte — far *below* the ridge, firmly memory-bound. This is the precise, quantitative reason the FLOP counter lies about depthwise nets: the efficient backbone removes arithmetic the model was bound by and replaces it with operators that are bound by *bandwidth instead*, so the realized speedup is the bandwidth-limited rate, not the FLOP-implied one. Keep $I = 160$ ops/byte in mind — every lever below either reduces FLOPs (sliding us left toward the memory roof) or reduces bytes moved (sliding us right, or raising the achievable rate under the same roof), and which one helps depends entirely on which side of the ridge we are on at that moment.

#### Worked example: the frame budget, written down

Before optimizing, write the budget as an equation so every later decision is checkable against it. The per-frame wall-clock time the user feels is the sum of the stages:

$$t_{\text{frame}} = t_{\text{capture}} + t_{\text{pre}} + t_{\text{infer}} + t_{\text{nms}} + t_{\text{render}}$$

and the frame rate the user sees is just its reciprocal, capped by the camera:

$$\text{FPS} = \min\!\left(\text{FPS}_{\text{camera}},\ \frac{1000}{t_{\text{frame}}\,[\text{ms}]}\right)$$

For the baseline, with $t_{\text{capture}} = 6$, $t_{\text{pre}} = 7$, $t_{\text{infer}} = 108$, $t_{\text{nms}} = 9$, $t_{\text{render}} = 2$ (all ms), we get $t_{\text{frame}} = 132$ ms in the worst case and a sustained $\approx 120$ ms when stages overlap on different cores, i.e. **about 8 FPS**. The budget says $t_{\text{frame}} \le 50$ ms. The gap is dominated by $t_{\text{infer}}$, so that is where the first ~70 ms has to come from. Equally important: **even if inference dropped to zero, we would still spend $6 + 7 + 9 + 2 = 24$ ms** on the non-model stages — which means there is a floor, NMS and preprocess are not free, and we will have to optimize them too before the end. The figure below is this budget after the journey, showing how the 50 ms is actually spent on the shipped pipeline.

![A vertical stack breaking the final 50 ms frame budget into model inference at 22 ms, NMS and decode at 9 ms, preprocess at 7 ms, camera capture at 6 ms, tensor copy at 4 ms, and overlay render at 2 ms](/imgs/blogs/case-study-real-time-vision-on-device-2.png)

Hold that stack in mind. By the end, inference is 22 ms — still the largest single slice but no longer the whole story — and the postprocess tail (NMS + copies) has become a real fraction of the budget, which is exactly the signature of a model that is no longer compute-bound.

## Step 1: architecture — swap the backbone for one the hardware likes

The baseline backbone was chosen in a world where the only metric was accuracy and the only hardware was a datacenter GPU. On a phone NPU it is the wrong shape entirely. The first and biggest lever is to replace it with an efficient backbone designed for mobile inference.

The science here is about what the hardware can do per millisecond, not what looks cheap on a FLOP counter. Two backbones with identical FLOP counts can differ 3× in real latency, because FLOPs ignore memory traffic, kernel launch overhead, and how well an operator maps to the accelerator. (The full treatment is [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) and the FLOPs-latency gap; the practitioner's lesson is below.) Depthwise-separable convolutions, the core trick of the MobileNet family, factor a standard convolution into a per-channel spatial filter (depthwise) followed by a 1×1 cross-channel mix (pointwise). For a $k\times k$ convolution mapping $C_{\text{in}}$ channels to $C_{\text{out}}$ over an $H\times W$ feature map, the standard cost is

$$\text{FLOPs}_{\text{std}} = H\,W\,C_{\text{in}}\,C_{\text{out}}\,k^2$$

while the depthwise-separable version costs

$$\text{FLOPs}_{\text{sep}} = H\,W\,C_{\text{in}}\,k^2 + H\,W\,C_{\text{in}}\,C_{\text{out}}$$

The ratio is roughly $\frac{1}{C_{\text{out}}} + \frac{1}{k^2}$, so for a typical $3\times 3$ conv to 256 channels that is about $\frac{1}{256} + \frac{1}{9} \approx 0.115$ — an **8–9× reduction in arithmetic** for that block. The catch, and it is the whole reason FLOPs lie, is that depthwise convolutions have *low arithmetic intensity*: they move a lot of activation bytes per multiply, so they are often **memory-bound** and run far below the chip's peak compute. That is why a naive "MobileNet is 9× fewer FLOPs so it'll be 9× faster" expectation lands at maybe 3–4× in practice. The FLOPs are a ceiling on speedup, never a promise.

The principled way to choose is not to read a FLOP table but to let the search target *measured latency on the actual device* — hardware-aware neural architecture search, where the cost in the search objective is the model's real on-device latency, not a proxy. (See [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas).) In practice for this project we did the pragmatic thing a team under a deadline does: we took a proven hardware-aware backbone (an EfficientNet-Lite / MobileNetV3-large-scale extractor whose latency on mobile NPUs is already well characterized) and attached our existing detection neck and head. We did not run a full NAS from scratch — that is weeks of compute — we *reused* the result of one.

The measured outcome of the swap, on the phone:

| | Backbone latency | Total infer | mAP |
| --- | --- | --- | --- |
| Baseline (heavy) | ~96 ms | 108 ms | 37.4 |
| Efficient backbone | ~50 ms | 62 ms | 34.3 |

Latency nearly halved — 108 ms down to 62 ms — which is the single biggest absolute win in the entire journey. But we **paid 3.1 points of mAP** (37.4 → 34.3). That is over the 2-point budget. A cheaper, faster model that is too inaccurate is not a win; it is a different failure. We are now *below* the accuracy frontier — there exists a faster-and-more-accurate config than what we have, namely "the same architecture but trained better." Which is exactly what the next lever is for.

## Step 2: distillation — teach the small model with the big one

We have a fast student that lost three points. We also still have the slow, accurate teacher sitting right there. Distillation is the lever that uses the teacher's *soft* outputs to train the student to recover accuracy it cannot reach from hard labels alone. (Fundamentals in [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals); detection has its own wrinkles, below.)

Why does this work at all? The intuition is that a one-hot label tells the student "this box is a couch" and nothing else, while the teacher's full output distribution says "this is a couch, but it is 12% chair-like and the box should be 4 pixels wider on the left." That extra structure — the "dark knowledge" in the relative probabilities and the regressed offsets — is a far richer training signal than a hard label, and it is exactly the signal the small model needs to find the good minimum its capacity can still reach. The classic classification distillation loss combines a temperature-softened cross-entropy against the teacher with the normal hard-label loss:

$$\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{CE}}(y,\,\sigma(z_s)) \;+\; \alpha\,T^2\,\mathcal{L}_{\text{KL}}\!\big(\sigma(z_t/T)\,\|\,\sigma(z_s/T)\big)$$

where $z_s, z_t$ are student and teacher logits, $T$ is the temperature that softens the distributions (higher $T$ exposes more of the inter-class structure), and the $T^2$ factor keeps the gradient magnitudes comparable as $T$ changes. For **detection** specifically you do not just distill the classification head — you distill three things, and which ones you pick is the subject of [what to distill: response, feature, relation](/blog/machine-learning/edge-ai/what-to-distill-response-feature-relation):

1. **Response / logit distillation** on the classification branch (the formula above), so the student matches the teacher's per-anchor class scores.
2. **Feature distillation** on the backbone/neck feature maps — match the student's intermediate features to the teacher's (with a 1×1 adapter to align channel counts), usually weighted toward foreground regions so the loss is not drowned by background.
3. **Box / localization distillation** — match the student's predicted offsets to the teacher's, which is where most of the *localization* accuracy comes back.

A pragmatic detection-distillation training step, foreground-weighted on the feature loss:

```python
import torch
import torch.nn.functional as F

def detection_distill_loss(student_out, teacher_out, targets,
                           alpha=0.5, T=2.0, beta=4e-4):
    # 1) response distillation on classification logits (per anchor)
    s_cls, t_cls = student_out["cls_logits"], teacher_out["cls_logits"]
    kd_cls = F.kl_div(
        F.log_softmax(s_cls / T, dim=-1),
        F.softmax(t_cls.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

    # 2) feature distillation on neck features, weighted toward foreground
    fg = targets["fg_mask"].unsqueeze(1).float()       # 1 where an object is
    s_feat = student_out["neck_feat"]
    t_feat = teacher_out["neck_feat"].detach()
    feat_l2 = ((s_feat - t_feat) ** 2 * fg).sum() / (fg.sum() + 1e-6)

    # 3) box / localization distillation on regression offsets
    box_kd = F.smooth_l1_loss(student_out["box_reg"],
                              teacher_out["box_reg"].detach())

    # hard-label detection loss the model was already training with
    hard = student_out["det_loss"]

    return (1 - alpha) * hard + alpha * kd_cls + beta * feat_l2 + 0.5 * box_kd
```

We trained the efficient-backbone student for the usual number of epochs with the teacher frozen, distilling all three signals. The result:

| | mAP | Δ vs baseline |
| --- | --- | --- |
| Efficient backbone, hard labels only | 34.3 | −3.1 |
| Efficient backbone, distilled | 36.7 | −0.7 |

Distillation bought back **2.4 of the 3.1 points** we lost in the swap, landing at 36.7 — comfortably inside the 2-point budget while keeping the 62 ms latency unchanged (distillation changes *training*, not the deployed graph; the student architecture is identical). We are now back *on* the frontier: as accurate as we can be at this speed, and much faster than the original. This is the cleanest illustration in the whole project of why levers compose — architecture bought the speed, distillation paid back the accuracy, and neither could have done both alone. The figure below reads the journey so far as accuracy against latency, showing exactly this dip-and-recover.

![A matrix comparing four steps across latency, mAP, and whether the point sits on the accuracy-latency frontier, showing the backbone swap dropping below the frontier and distillation pulling it back](/imgs/blogs/case-study-real-time-vision-on-device-3.png)

## Step 3: structured pruning — remove channels for a real speedup

We are at 62 ms. The budget is 50. We still have ~12 ms of inference to remove, and the model still has more capacity than it strictly needs for this 12-class task. Pruning is the lever for that — but only **structured** pruning, and the distinction is the entire point. (See [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up).)

Here is the trap, and I have watched a team fall into it. **Unstructured** pruning — zeroing individual weights — can remove 80–90% of parameters with tiny accuracy loss and produces a beautiful sparsity number for a slide. It also produces **zero speedup** on a phone NPU, because the hardware still does the same dense matrix multiply; a multiply-by-zero costs exactly as much as a multiply-by-anything on a dense MAC array. Sparse weights save *storage* (if you compress them) and nothing else on commodity edge hardware. To get a real speedup you must remove whole **structures** the hardware can skip — entire channels, filters, or blocks — so the resulting tensor is *smaller and still dense*, and the convolution genuinely does fewer operations.

So we prune **channels**. The procedure is the standard prune-then-finetune loop:

1. Score each channel by importance. A cheap, effective score is the L1 norm of its filter weights, or better, a Taylor-expansion estimate of how much removing it would change the loss: $\text{importance}(c) \approx \big|\,\frac{\partial \mathcal{L}}{\partial a_c}\, a_c \,\big|$ summed over a calibration batch, where $a_c$ is channel $c$'s activation.
2. Remove the lowest-importance channels up to a target ratio, *physically* shrinking the weight tensors and the matching dimensions of adjacent layers (this is the part that delivers speed — the tensors get smaller).
3. Fine-tune the pruned model to recover the accuracy the removal cost.

```python
import torch
import torch_pruning as tp

# Build a dependency graph so removing a channel also fixes the
# downstream layers that consume it -- this is what makes it STRUCTURED.
example = torch.randn(1, 3, 640, 640)
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)

# Taylor importance: how much does dropping this channel move the loss?
imp = tp.importance.TaylorImportance()
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs=example,
    importance=imp,
    pruning_ratio=0.25,           # remove 25% of channels overall
    ignored_layers=[model.head],  # never prune the detection head structurally
)

pruner.step()                     # physically removes channels + resizes tensors
# ... then fine-tune the smaller, still-dense model to recover mAP ...
```

Two things in that snippet matter. The **dependency graph** is what makes it structured-and-correct: removing channel 17 of a conv means the next layer must stop expecting channel 17 as input, and a residual add must stay consistent — the graph propagates the cut everywhere it needs to go. And `ignored_layers=[model.head]` is a decision I will return to: we do **not** structurally prune the detection head, because the head is small (so pruning it saves little) and fragile (so pruning it hurts a lot). Prune where the FLOPs and the slack are — the backbone — and leave the delicate parts alone.

The measured result, pruning 25% of backbone channels and fine-tuning:

| | Infer latency | Size | mAP |
| --- | --- | --- | --- |
| After distillation | 62 ms | 41 MB | 36.7 |
| 25% channel prune + finetune | 48 ms | 31 MB | 36.3 |

We crossed the 50 ms line — 48 ms — for a cost of **0.4 mAP**, and the model shrank to 31 MB along the way (fewer channels means fewer weights). The accuracy cost is small because the L1/Taylor scoring removed channels that were genuinely carrying little signal for our 12-class task, and the fine-tune healed the rest. We are now *technically* inside the latency budget on a single warm inference. But "48 ms in a benchmark loop" is not "ships" — we still have size headroom to spend, the model is still fp32, and we have not faced the phone under sustained heat. The next lever buys margin.

## Step 4: quantization — int8, but the head keeps fp16

Quantization is the lever that converts the model's fp32 weights and activations to int8, cutting size 4× and, on hardware with int8 MAC units (every modern phone NPU and the Jetson's tensor cores), running the math several times faster. This is also the step where a detector can fall off an accuracy cliff if you are not careful, so it earns the most science. (Foundations in [post-training quantization (PTQ)](/blog/machine-learning/edge-ai/post-training-quantization-ptq); the sensitivity reasoning in [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis).)

### The science: why int8 loses accuracy, and how much

Quantization maps a real value $x$ to an 8-bit integer with a scale $s$ and zero-point $z$: $\hat{x} = s\,(\text{round}(x/s) - z) \approx x$. The rounding introduces an error. If the values are spread over a range of width $R$ and we use $2^b$ levels (for int8, $b=8$, so 256 levels), the step size is $s = R / 2^b$ and, assuming the rounding error is uniform over $[-s/2, s/2]$, the error variance is

$$\sigma_q^2 = \frac{s^2}{12} = \frac{R^2}{12\cdot 2^{2b}}$$

The signal-to-quantization-noise ratio in decibels works out to the famous law

$$\text{SQNR} \approx 6.02\,b + 1.76\ \text{dB}$$

i.e. **each additional bit buys about 6 dB**. For int8 that is roughly 50 dB of headroom — plenty for most weights and activations, which is why int8 PTQ usually costs a fraction of a point. But the law assumes the range $R$ is set well. If a tensor has a few large outliers, you either clip them (losing those values) or stretch $R$ to include them (making $s$ huge, so every *typical* value gets a coarse, lossy step). This is the crux of detection quantization: **the box-regression outputs in the detection head have a wide, asymmetric dynamic range** — coordinate offsets can be large, and a 1-step rounding error in a coordinate moves a box edge by enough pixels to flip an IoU-thresholded match from hit to miss. Classification logits tolerate int8 fine; box coordinates do not.

Two mitigations follow directly from the math. First, **per-channel quantization**: give each output channel its own scale $s_c$ instead of one scale for the whole weight tensor. Channels with naturally small weights get a fine step; channels with large weights get a coarse one; nobody is held hostage to the tensor-wide max. For convolution weights this is almost free and recovers most of the loss. Second, **keep the sensitive layers in higher precision** — a mixed-precision policy where the backbone runs int8 (where the FLOPs and the tolerance both are) and the **detection head stays fp16** (where the dynamic range and the fragility both are).

It is worth seeing *why per-channel helps* in one line, because it is the difference between a usable int8 model and a broken one. With a single per-tensor scale $s = \max_c R_c / 256$, where $R_c$ is channel $c$'s range, the quantization error variance for a *small-range* channel is $\sigma_q^2 = s^2/12$ — but $s$ was set by the *largest* channel's range, so a channel whose own range is, say, 10× smaller is being quantized with a step 10× coarser than it needs, and its error variance is 100× larger than it should be. Per-channel quantization sets $s_c = R_c / 256$ independently, so each channel's error variance scales with *its own* range, $\sigma_{q,c}^2 = (R_c/256)^2 / 12$, and no channel pays for another's outliers. The total error injected into the layer's output drops by a factor equal to the variance of channel ranges across the tensor — large for real weights, which is exactly why per-channel is the default and per-tensor is a foot-gun for anything with heterogeneous channels. The detection head's regression branch is the pathological case where even per-channel is not enough, because the *outliers live within a single channel*, not across channels — which is why the head needs higher precision, not just finer per-channel scales.

### The practical flow: PTQ with a representative calibration set

PTQ needs a **representative dataset** — a few hundred real images that the calibrator runs through the model to observe the actual activation ranges, so it can pick good scales. "Representative" is load-bearing: calibrate on images that match deployment (lighting, object scales, clutter), not on a synthetic or out-of-distribution sample, or your ranges will be wrong and accuracy will crater. Here is the calibration loop in TensorFlow Lite / LiteRT, the path for the Android target:

```python
import tensorflow as tf

def representative_dataset():
    # ~300 real frames matching deployment conditions, preprocessed
    # EXACTLY as in training (same resize, same normalization).
    for img in calibration_frames[:300]:
        x = preprocess(img)[None].astype("float32")  # 1x640x640x3
        yield [x]

converter = tf.lite.TFLiteConverter.from_saved_model("detector_pruned")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Full int8 for the backbone path; allow float fallback for ops that
# would lose too much (this is where the head stays higher precision).
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,        # float fallback path
]
converter.inference_input_type = tf.uint8   # camera gives uint8 anyway
converter.inference_output_type = tf.float32 # decode boxes in float

tflite_int8 = converter.convert()
open("detector_int8.tflite", "wb").write(tflite_int8)
```

On the PyTorch / TensorRT path for the Jetson, the equivalent is per-channel int8 with an explicit calibrator and the head's layers marked to run in fp16:

```python
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

qmap = get_default_qconfig_mapping("x86")  # per-channel int8 weights
# Exclude the detection head from int8 -- it stays float.
qmap = qmap.set_module_name("head", None)

example = (torch.randn(1, 3, 640, 640),)
prepared = prepare_fx(model.eval(), qmap, example)

# Calibrate: run ~300 representative frames, observers learn the ranges.
with torch.no_grad():
    for img in calibration_frames[:300]:
        prepared(preprocess(img)[None])

quantized = convert_fx(prepared)   # backbone int8, head still float
```

The measured result of int8 PTQ with per-channel scales and the **fp16 head**:

| | Infer latency | Size | mAP |
| --- | --- | --- | --- |
| Pruned fp32 | 48 ms | 31 MB | 36.3 |
| int8 PTQ, full int8 head | 30 ms | 22 MB | **32.9** |
| int8 PTQ, **fp16 head** | 31 ms | 24 MB | **36.1** |

Read those last two rows carefully — they are the most important comparison in the post. Quantizing *everything* to int8 gave the best size and speed but **dropped mAP by 3.2 points** (36.3 → 32.9): that is the detection-head cliff, exactly as the dynamic-range argument predicted. Keeping the head in fp16 cost **1 ms and 2 MB** and **recovered 3.2 points** (back to 36.1). That is the trade of the year: a millisecond and two megabytes for three points of accuracy. The figure below is precisely this before/after — full int8 versus the mixed int8-backbone, fp16-head version.

![A before-after figure contrasting a full int8 model whose head clips box coordinates and loses accuracy against a mixed precision model with an int8 backbone and an fp16 head that recovers the lost accuracy](/imgs/blogs/case-study-real-time-vision-on-device-6.png)

#### Worked example: the one step that cost too much mAP, and how it was recovered

This is worth doing as numbers because it is the canonical edge-AI surprise. We applied int8 PTQ expecting the usual sub-point loss. Instead mAP fell from **36.3 to 32.9 — a 3.4 point cliff**, four times worse than the SQNR law predicts for int8. The law says int8 should cost a fraction of a point; something was violating its assumption.

We did the diagnosis the sensitivity-analysis way: quantize one block at a time, measure mAP after each, and find which block is responsible. The backbone blocks each cost **< 0.2 mAP** when quantized — exactly as expected. But quantizing the **box-regression branch of the head** alone dropped mAP by **3.1 points** on its own. The culprit was confirmed: the head's coordinate outputs span a range with rare large values, so the tensor-wide int8 scale was coarse, and a 1-LSB rounding error on a coordinate was enough to push many boxes just past the IoU-match threshold. The classification branch of the head, by contrast, quantized fine (logits are bounded and benign).

The fix followed from the diagnosis: **leave the head's regression branch in fp16**, keep the rest int8. Cost: +1 ms latency, +2 MB size. Benefit: +3.2 mAP, back to 36.1. The lesson generalizes far past this model — **profile sensitivity per layer, do not quantize uniformly, and expect the parts that emit unbounded numeric quantities (coordinates, depths, regression targets) to be the fragile ones.** Mixed precision is not a luxury for detectors; it is the difference between shipping and not.

## Step 5: compile and fuse — make the runtime earn its keep

We have a 31 ms, 24 MB, 36.1 mAP model in a quantized graph. The last big inference win is not changing the model at all — it is letting the compiler and runtime turn that graph into fused, hardware-tuned kernels. (See [inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared), [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization), and for the Jetson specifically [TensorRT and GPU edge inference on Jetson](/blog/machine-learning/edge-ai/tensorrt-and-gpu-edge-inference-on-jetson).)

The two mechanisms that matter:

**Operator fusion.** A naive graph executes conv, then bias-add, then batch-norm, then ReLU as four separate kernels, each reading its input from DRAM and writing its output back. Those intermediate round-trips to memory are pure overhead — the data is the same, it just gets shuffled to memory and back three times. Fusion collapses the chain into a single kernel that reads once, does all four operations in registers/cache, and writes once. For a model that, post-quantization, is starting to become **memory-bound**, killing those round-trips is the single biggest remaining lever. Conv-BN-ReLU fusion and the int8 dequant/requant fusion are the big ones.

**Hardware-specific kernel selection / autotuning.** TensorRT (and TVM, and the better mobile delegates) try multiple kernel implementations for each layer at build time and keep the fastest for *your specific* shapes and chip. This is why you build the engine on the target, not on your laptop — the optimal tiling for a Snapdragon GPU is not the optimal tiling for an Orin Nano.

The TensorRT path for the Jetson, building an int8 engine with fp16 fallback (which is also how the head ends up fp16) directly from the ONNX export:

```bash
# Export the quantized PyTorch model to ONNX first, then build the engine.
trtexec \
  --onnx=detector_int8.onnx \
  --saveEngine=detector.plan \
  --int8 --fp16 \                # int8 where it helps, fp16 fallback for the head
  --useCudaGraph \               # capture the launch sequence, cut CPU overhead
  --builderOptimizationLevel=5 \ # spend build time finding fast kernels
  --shapes=images:1x3x640x640    # fixed batch=1 shape unlocks better kernels
```

Two flags carry weight. `--useCudaGraph` captures the whole sequence of kernel launches once and replays it, which removes per-launch CPU overhead — meaningful when individual kernels are tens of microseconds and you have dozens of them. Fixing the shape with `--shapes` (batch=1, our reality) lets the autotuner specialize instead of hedging across dynamic shapes. On the Android side the equivalent is choosing the right delegate (GPU or the vendor NNAPI/QNN delegate) and accepting that the delegate fuses what it can and routes the rest to the CPU.

The measured effect of compile + fuse, on both targets:

| | Infer (phone) | Infer (Orin Nano) |
| --- | --- | --- |
| int8 graph, unfused | 31 ms | 14 ms |
| compiled + fused | **22 ms** | **9 ms** |

Fusion plus autotuning took the phone from 31 to **22 ms** — a 1.4× speedup with **zero accuracy change**, because fusion is numerically equivalent, it just stops wasting memory bandwidth. That puts us at 22 ms inference, **45 FPS** of model headroom, well inside the 50 ms total budget once you add back the other stages. On the Orin Nano the same model runs at 9 ms (it has real int8 tensor cores and bandwidth), which leaves enormous thermal headroom — the relevant question there shifts from "is it fast enough?" to "does it stay fast for two minutes?"

Here is the subtle, important thing this step revealed, and it is the science payoff of the whole journey. We profiled again after compiling, and **the bottleneck had moved**. At the baseline, 78% of inference was backbone convolutions — compute-bound, FLOPs-limited. After stripping the FLOPs out with an efficient backbone, pruning, and int8, the convolutions were so cheap that the model spent its time waiting on **memory traffic and the postprocess tail** — NMS and box decode now accounted for ~40% of the per-frame time (look back at the budget stack: 9 ms of NMS against 22 ms of inference). The model went from compute-bound to memory-and-NMS-bound. That is not a footnote; it changes the optimization strategy entirely. Once you are memory-bound, more FLOP cuts do almost nothing — fusion (which cuts memory traffic) and a faster NMS (which cuts the tail) are what move the needle. The figure below shows that shift.

![A before-after figure showing the bottleneck moving from a compute-bound regime where backbone convolutions dominate to a memory-and-NMS-bound regime where DRAM traffic and postprocessing dominate](/imgs/blogs/case-study-real-time-vision-on-device-5.png)

## Step 6: deploy — the part nobody profiles in a notebook

We have a 22 ms model. We do not have a shipped feature. The gap is everything that happens around the model on the device, and it is where I have watched more launches slip than at any modeling step. (The full discipline is [mobile deployment end to end](/blog/machine-learning/edge-ai/mobile-deployment-end-to-end); here are the three things that bit this project.)

### Warm-up so the first frame does not freeze the camera

The first inference triggers kernel compilation, workspace allocation, and (on a GPU delegate) shader compilation — that first call was ~280 ms on the phone, long enough to visibly hitch the camera preview on feature open. The fix is a **warm-up**: run a few inferences on a dummy frame the moment the model loads, on a background thread, *before* the user points the camera. By the time the first real frame arrives the kernels are compiled and cached.

```kotlin
// Android: warm up off the UI thread the moment the model is ready.
scope.launch(Dispatchers.Default) {
    val dummy = ByteBuffer.allocateDirect(640 * 640 * 3)
    repeat(5) { interpreter.run(dummy, outputBuffer) }  // discard results
    warmUpComplete.set(true)
}
```

### Preprocessing parity — the silent accuracy leak

There is an accuracy bug that never shows up in the model and never shows up in your offline eval, and it cost us most of an afternoon. The model was trained with a specific preprocessing pipeline: resize the 640×640 frame with bilinear interpolation, convert BGR→RGB, normalize each channel by the dataset mean and standard deviation in fp32, then quantize. The Android app, written by a different person, did its own preprocessing: it resized with the platform's default (which happened to be a slightly different interpolation), kept the camera's native color order, and normalized with hard-coded constants that did not quite match the training statistics. None of this crashed. The model ran, drew boxes, looked plausible. But mAP measured *on-device against ground truth* was **2.1 points below** the offline number — accuracy that had nothing to do with any compression lever and everything to do with the device feeding the model subtly different pixels than training did.

The fix is to make preprocessing **bit-for-bit identical** between training and the device, and to test that property directly rather than trusting it. The cheapest test is a golden-input check: run one fixed image through the training preprocessing and the device preprocessing, and assert the resulting tensors match within a tiny tolerance.

```python
# Golden-input parity check: device preprocessing must match training.
golden = load_image("parity_test.jpg")
train_tensor = training_preprocess(golden)          # the pipeline used in training
device_tensor = device_preprocess(golden)           # the on-device pipeline
max_abs_diff = float((train_tensor - device_tensor).abs().max())
assert max_abs_diff < 1e-3, f"preprocessing drift: {max_abs_diff}"
```

This belongs in CI, not in someone's memory. Pin the interpolation mode, the channel order, the normalization constants, and the dtype, and assert parity on every build. The reason this is so dangerous is that it is *invisible* — there is no error, the boxes look roughly right, and you only catch it if you measure mAP on-device against ground truth rather than assuming the on-device model behaves like the offline one. We did, eventually, and recovered the 2.1 points by aligning the resize and the normalization. **The model is only as accurate as the pixels it is fed, and the device feeds different pixels than your notebook unless you force it not to.**

### NMS and box decode on-device — and the CPU-fallback trap

Non-maximum suppression (NMS) takes the thousands of raw candidate boxes the head emits and collapses overlapping detections of the same object down to one per object, keeping the highest-confidence box and suppressing the rest. It is genuinely necessary and it is *not* a neural-network operation — it is a sort-and-loop over boxes. Two real problems showed up.

First, where it runs. If you fold NMS into the model graph (e.g. a `CombinedNonMaxSuppression` op or TensorRT's `EfficientNMS` plugin), it runs on the accelerator, fused and fast. If you leave it as a separate post-step, it runs on the CPU — and on the phone our hand-written CPU NMS was 9 ms, a meaningful chunk of the 50 ms budget. Worse: in our first attempt, the in-graph NMS op was **not supported by the phone's NPU delegate**, so the delegate silently routed that subgraph — *and everything topologically after it* — back to the CPU, quietly tripling the postprocess cost with no error message. This is the classic mobile failure mode: an unsupported op partitions the graph and you lose the accelerator for the whole tail.

The fix was to **keep NMS out of the accelerated graph deliberately**, run a tuned NMS on the CPU (with a confidence pre-threshold to cut the candidate count before the expensive sort, and a cap on boxes per class), and let the accelerator do only the pure-tensor backbone+head. Counterintuitively, *forcing* NMS onto the CPU was faster and more predictable than letting the delegate decide, because it stopped the silent fallback of the rest of the graph.

```python
def fast_nms(boxes, scores, iou_thr=0.5, score_thr=0.25, topk=200):
    # Pre-threshold BEFORE sorting -- most candidates are background.
    keep = scores > score_thr
    boxes, scores = boxes[keep], scores[keep]
    order = scores.argsort()[::-1][:topk]   # cap candidates, cheap sort
    boxes, scores = boxes[order], scores[order]
    # ... standard greedy IoU suppression over the surviving topk boxes ...
    return selected
```

The pre-threshold matters more than the suppression loop: the head emits ~8400 raw boxes, but after a 0.25 confidence cut perhaps 60 survive on a typical frame, and sorting 60 boxes instead of 8400 is the difference between 9 ms and 2 ms.

### Thermals under sustained capture

The Orin Nano (continuous-use target) ran the model at 9 ms cold. We ran a **two-minute sustained-capture stress test** — feed frames at 30 FPS continuously and watch p99 and SoC temperature. After about 70 seconds the device hit its thermal limit and clocks dropped; p99 inference climbed from 9 ms to **15 ms**, and total frame time crept toward the 50 ms line. On the phone the effect was milder but real: p50 went from 22 to ~27 ms after 90 seconds. Neither *broke* the budget, but the lesson is that **the cold number is a lie for continuous use** — you must measure under sustained load, and you must leave thermal margin. We had margin precisely because we did not stop at "48 ms passes" — the extra headroom from quantization and fusion is what absorbed the throttle. Had we shipped the 48 ms fp32-pruned model, the thermal throttle would have pushed it over budget within a minute.

#### Worked example: budgeting for the throttle

Put the throttle into the budget equation. Define the throttle factor $\tau$ as the ratio of sustained-load latency to cold latency. We measured $\tau \approx 1.25$ on the phone after 90 s. The budget must be met *under throttle*, so the requirement is not $t_{\text{frame, cold}} \le 50$ ms but

$$\tau \cdot t_{\text{infer, cold}} + t_{\text{other}} \le 50\ \text{ms}$$

With $t_{\text{infer, cold}} = 22$ ms, $t_{\text{other}} = 6 + 7 + 9 + 4 + 2 = 28$ ms, and $\tau = 1.25$: $1.25 \times 22 + 28 = 27.5 + 28 = 55.5$ ms — **which exceeds 50 ms under sustained heat.** This is the moment the worked example earns its place: the model passes cold (50 ms) and *fails* hot (55.5 ms). The fix was not more model optimization — it was cutting $t_{\text{other}}$. Moving the tensor copy to a zero-copy direct buffer (−4 ms) and the NMS pre-threshold work above (−7 ms on the tail) brought $t_{\text{other}}$ to 17 ms, giving $27.5 + 17 = 44.5$ ms under throttle. **The last 11 ms that made it ship came from the non-model stages, not the model** — exactly what the Step-0 budget warned us about when it noted a 24 ms floor that has nothing to do with the network.

## The cumulative ledger — the centerpiece

Here is the whole journey as one table. Each row is one lever applied on top of the previous, on the **phone** target (the primary device), batch=1, warm-started, p50 under the sustained-load condition where noted. This is the artifact I would put in front of the team — it tells the entire story in numbers, and every row traces to a step above.

| Step | Lever | mAP | Infer ms | FPS (model) | Size MB | Peak mem | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | fp32 baseline | 37.4 | 108 | 9 | 98 | ~520 MB | compute-bound, unshippable |
| 1 | Efficient backbone | 34.3 | 50 | 20 | 41 | ~310 MB | −3.1 mAP, over budget |
| 2 | + Distillation | 36.7 | 50 | 20 | 41 | ~310 MB | +2.4 mAP back, on frontier |
| 3 | + Structured prune 25% | 36.3 | 38 | 26 | 31 | ~260 MB | −0.4 mAP, smaller + faster |
| 4a | + int8 PTQ (full) | 32.9 | 20 | 50 | 22 | ~200 MB | **head cliff −3.4 mAP** |
| 4b | + int8 PTQ (fp16 head) | 36.1 | 21 | 48 | 24 | ~210 MB | +3.2 recovered |
| 5 | + Compile & fuse | 36.1 | 22 | 45 | 24 | ~190 MB | fusion, memory-bound now |
| 6 | + Deploy (warm-up, NMS, thermal) | 36.1 | 22 | 45 | 24 | ~190 MB | p99 28 ms under heat |

Read top to bottom, the story is: **120 ms → 22 ms inference (≈ 5.5×), 98 MB → 24 MB (≈ 4×), 37.4 → 36.1 mAP (−1.3, inside the 2-point budget)**, and a feature that went from 8 FPS and stuttering to 45 FPS of model headroom that holds up under sustained capture. The two rows worth staring at are 4a versus 4b — the head cliff and its recovery — and step 1 versus step 2 — the accuracy dip from the backbone swap and the distillation that paid it back. Those two before/after pairs are where the discipline lives. The figure below is this ledger as a cumulative matrix.

![A matrix laying out size, latency, and mAP for each lever applied cumulatively, reading as a running ledger from the fp32 baseline down to the compiled and fused shipped model](/imgs/blogs/case-study-real-time-vision-on-device-7.png)

And the headline before/after — the fp32 cloud model versus the shipped int8 engine — is the figure below, the one-slide summary for anyone who only wants the punchline.

![A before-after figure contrasting a 98 MB fp32 cloud model at 120 ms per frame with a 24 MB int8 shipped engine at 22 ms per frame that holds accuracy within about one point](/imgs/blogs/case-study-real-time-vision-on-device-4.png)

## Where each lever moved the bottleneck

Step back and look at the journey through the lens of the roofline, because this is the scientific spine that ties the levers together and explains *why we applied them in this order*.

At the start, the model was firmly **compute-bound**: dense convolutions on fat channels, high arithmetic intensity, ALUs saturated, 78% of time in the backbone. When you are compute-bound, the lever that helps is the one that removes arithmetic — and that is exactly what the first three levers do. The **efficient backbone** replaces dense convs with depthwise-separable ones (fewer FLOPs). **Pruning** removes whole channels (fewer FLOPs, still dense). **int8 quantization** doesn't reduce the *number* of operations but makes each one cheaper on int8 MAC units (more effective FLOPs per second). All three attack the compute ceiling, and they pay off precisely because that was the binding constraint.

But each FLOP you remove brings the model closer to the **memory ceiling** — the rate at which weights and activations can be streamed from DRAM. By the time we had an int8, pruned, efficient model, the convolutions were so cheap that the model spent its time moving bytes, not multiplying them. The bottleneck **moved from compute to memory** (and to the non-tensor NMS tail). That is why the levers we reached for *last* are different in kind: **fusion** (cuts memory round-trips between ops) and **NMS optimization** (cuts the postprocess tail). If we had tried to fuse first, before the model was memory-bound, it would have helped little. If we kept cutting FLOPs after the model went memory-bound, it would have helped little. **The order of the levers is not arbitrary — it follows the bottleneck, and the bottleneck is something you measure, not guess.** This is the deepest takeaway of the case study, and the reason Step 0 (profile first) and the re-profile after Step 5 are not bureaucracy but navigation.

Stated as a rule: *apply the lever that attacks your current bottleneck, re-measure, and only then pick the next lever — because the bottleneck will have moved.* Compute-bound? Cut FLOPs (architecture, prune, quantize). Memory-bound? Cut traffic (fuse, lower precision of activations, smaller intermediates). Tail-bound? Optimize the postprocess. The accuracy-latency frontier (see [the accuracy-latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier)) is the map of where you can go; the roofline tells you which direction is currently downhill.

## Reading the result off the frontier

The shipping decision is not "is the model good?" — it is "which point on the accuracy-latency frontier do we ship, given the constraints?" Plot every config we measured with latency on the x-axis and mAP on the y-axis. The fp32 baseline sits at (120 ms, 37.4) — most accurate, far too slow. The full-int8 model sits at (20 ms, 32.9) — fastest, but its accuracy has fallen off the frontier (the fp16-head version is faster-*and*-more-accurate, so full-int8 is dominated). The shipped point is (22 ms, 36.1): on the frontier, inside both the latency cap and the size cap, with thermal margin. It is the **knee** — the point where you stop buying meaningful speed without paying real accuracy.

Two configs we considered and rejected, to show the frontier doing its job. A 50%-pruned variant hit (18 ms, 34.8): faster, but it fell 1.3 points below the shipped point for 4 ms we did not need (we were already inside budget), so it was a dominated trade we declined — speed we couldn't use, accuracy we couldn't spare. And an int4 backbone variant hit (16 ms, 33.1): the sub-8-bit step (see the series' int4/ternary work) crossed the accuracy cliff for our task, so it was off the table despite being fastest. The decision rule is mechanical once the frontier is drawn: **filter to the configs that satisfy every hard constraint (the feasible region), then pick the knee of that feasible set.** The shipped model is the knee of the feasible region, which is the whole point of doing the work step by step and measuring each one.

## Stress-testing the decisions

A decision you have not pushed on is a decision you do not understand. Before shipping I deliberately broke each lever to learn where its edge was — and several of those experiments changed the final config.

**What happens at int4?** The obvious "go further" move is to push the backbone below 8 bits. We tried an int4 backbone (the sub-8-bit territory covered elsewhere in the series). The SQNR law is unforgiving here: dropping from 8 to 4 bits removes four bits, so $\text{SQNR}$ falls by $4 \times 6.02 \approx 24$ dB — a 250× increase in quantization-noise power. For a robust classification backbone that can sometimes be absorbed; for *our* detector feeding a coordinate-sensitive head, it was not. mAP fell to 33.1 even with the head kept fp16, because the backbone features feeding the box head were now too noisy to localize precisely. The latency win (16 ms vs 22 ms) was real but unusable — we were already inside the frame budget, so spending 3 points of mAP to shave 6 ms we did not need is a strictly dominated trade. **Lesson: int4 is a tool for when 8-bit cannot meet the budget, not a free extra speedup; below 8 bits, regression-sensitive models need QAT and often still cannot get there.**

**What happens when the calibration set is tiny?** PTQ's calibration set determines the activation ranges, so a bad set means bad scales. We re-ran int8 calibration with only **16 images** instead of 300. mAP dropped an extra 1.4 points versus the 300-image calibration, and the loss was concentrated in classes that happened to be under-represented in the 16. The reason is direct: with too few images the observer never sees the true tail of the activation distribution, so it picks a clipping range that is too tight, and real activations at inference saturate. We also tried the opposite failure — calibrating on *out-of-distribution* images (bright studio shots when deployment is dim indoor rooms) — which was worse than a small in-distribution set, because the ranges were systematically wrong rather than just noisy. **Lesson: a few hundred genuinely representative frames beats thousands of mismatched ones; calibration data quality is a first-class accuracy lever, not a checkbox.**

**What happens when the NPU does not support an op?** Already foreshadowed by the NMS fallback, but we stress-tested it on purpose by feeding the delegate a graph with a custom activation it did not support. The delegate partitioned the graph at the unsupported op, and inference latency more than tripled — not because that one op was slow, but because the partition forced an accelerator→CPU→accelerator round-trip, and each handoff copies the entire intermediate tensor across the memory boundary. The takeaway that survived into the shipped design: **prefer operators in the delegate's supported set even when a fancier op is slightly more accurate, because one unsupported op in the middle of the graph can cost more than the op itself many times over.** We replaced the custom activation with a hardware-supported one (a hard-swish the delegate fuses) and lost nothing measurable.

**What happens when the model is memory-bound and we keep cutting FLOPs?** After int8 + fusion, we tested whether *more* pruning helped. Pruning another 15% of channels (40% total) cut FLOPs by a further chunk but moved phone latency from 22 ms only to 21 ms — a 1 ms win — while costing 0.9 mAP. That non-result is the roofline made visible: once you are below the ridge point, the achievable rate is $I \cdot B$, set by bandwidth, and removing arithmetic does not raise it. The lever that *did* still help at that point was reducing activation **bytes** (smaller intermediate feature maps, fused dequant so activations stay int8 longer) — which raises effective intensity and slides us back up the bandwidth roof. **Lesson: the question "will more pruning help?" has a measurable answer — check which side of the ridge you are on first.**

## What broke, and how it was fixed

The clean ledger above hides three fights. Here they are, because the fights are where the real lessons live.

**The int8 detection-head cliff.** Already detailed in the Step-4 worked example: uniform int8 dropped mAP 3.4 points because the box-regression outputs have a wide, outlier-prone dynamic range that an int8 scale cannot represent finely. *Diagnosis:* per-layer sensitivity analysis isolated the head's regression branch as the culprit (backbone blocks each cost < 0.2 mAP; the head alone cost 3.1). *Fix:* keep the regression head in fp16, backbone int8 — recovered 3.2 points for +1 ms and +2 MB. *Lesson:* never quantize a detector uniformly; the layers that emit unbounded numeric quantities are the fragile ones, and mixed precision is mandatory, not optional.

**The silent CPU-fallback NMS.** The in-graph NMS op was unsupported by the phone NPU delegate, which partitioned the graph and routed NMS *and the entire tail after it* back to the CPU — silently, no error, ~3× the postprocess cost. *Diagnosis:* the per-stage benchmark (Step 0) showed postprocess at 27 ms when it should have been ~9, and the delegate's partition log confirmed the fallback. *Fix:* deliberately keep NMS out of the accelerated graph, run a tuned CPU NMS with a confidence pre-threshold and a top-k cap so it only sorts ~60 boxes instead of ~8400. *Lesson:* an unsupported op does not just slow itself down — it can hand the whole subgraph after it back to the CPU; design your graph partition deliberately rather than trusting the delegate to do the right thing.

**The thermal throttle at sustained 30 FPS.** The cold latency passed; the sustained-load latency did not (the throttle worked example: 44.5 ms under heat only after non-model fixes; 55.5 ms before them). *Diagnosis:* the two-minute stress test showed p99 climbing as the SoC throttled past 70 seconds. *Fix:* cut the non-model stages — zero-copy tensor buffers (−4 ms) and the NMS pre-threshold (−7 ms on the tail) — to create thermal margin, since the model itself was already as fast as the accuracy budget allowed. *Lesson:* measure under sustained load, budget for a throttle factor $\tau > 1$, and remember that the last few milliseconds often come from the pipeline around the model, not the model. The 24 ms non-model floor from Step 0 was a prophecy.

The "ship it?" decision that closes the project is a gate with two halves, and the build only ships if both pass. The figure below is that gate as a tree: a performance gate (does it hit the latency budget *and* survive thermals?) and a quality gate (is the mAP drop within budget *and* do the tail cases — small / occluded objects — still work?). Every check has a concrete pass/fail threshold, and a fail on any leaf sends you back to a specific lever.

![A decision tree showing a ship candidate splitting into a device performance gate and a quality gate, each with two concrete pass-or-fail checks covering speed, thermals, accuracy drop, and tail cases](/imgs/blogs/case-study-real-time-vision-on-device-8.png)

## Case studies and real numbers from the literature

The journey above is representative, but it rhymes with published, named results — worth grounding against so you trust the shape.

**MobileNetV3 on Pixel.** The MobileNetV3 paper (Howard et al., 2019) reports MobileNetV3-Large at roughly 75% ImageNet top-1 at about 51 ms on a Pixel-1-class CPU, versus MobileNetV2 needing more latency for less accuracy — the canonical demonstration that a hardware-aware architecture (their NAS used real on-device latency in the objective) moves the latency-accuracy frontier, not just the FLOP count. Our Step-1 backbone swap is the same move applied to a detector.

**DistilBERT.** Sanh et al. (2019) distilled BERT-base into a model with **40% fewer parameters that runs ~60% faster while retaining ~97% of GLUE performance** — the cleanest public proof that distillation recovers most of the accuracy a smaller architecture would otherwise lose. Our Step-2 detection distillation (recovering 2.4 of 3.1 lost points) is the detection analogue.

**int8 PTQ on detectors.** The TensorRT and TFLite documentation, and the original Jacob et al. (2018) integer-quantization work, consistently report that classification networks lose a fraction of a point to int8 PTQ with per-channel scales, while **detection and regression-heavy heads are markedly more sensitive** — the published guidance to keep sensitive layers in higher precision (mixed int8/fp16) is exactly the fix our Step-4 head-cliff required. NVIDIA's own detection examples ship with the NMS as a dedicated plugin (`EfficientNMS`) precisely because generic NMS is a known accelerator pain point — the Step-6 fallback trap is a documented, common failure.

**YOLO-family on Jetson.** Public benchmarks of small YOLO variants (YOLOv5n/v8n-scale) on Jetson Orin Nano with TensorRT int8 land in the high-single-digit to low-teens milliseconds per frame at 640×640 — consistent with our 9 ms Orin number after compile+fuse — and these reports also note thermal throttling under sustained inference on the Nano's power-constrained form factor, matching our Step-6 stress-test finding. Treat the exact milliseconds as version- and power-mode-dependent; the order of magnitude is robust.

## When to reach for this whole sequence (and when not to)

The full seven-step pipeline is the right answer when you have a real-time vision feature on a memory- and power-constrained device with a hard frame budget. It is overkill in several common situations, and saying so is part of the job.

- **If profiling shows you're already in budget, stop.** If the baseline ran at 40 ms on the phone, you would not touch the architecture at all — you'd quantize for size if needed and ship. Every lever costs accuracy and engineering time; spend only the ones the budget demands.
- **If the model is memory-bound from the start, don't lead with FLOP cuts.** A model dominated by a giant embedding table or huge activations won't speed up from a cheaper backbone. Lead with quantizing activations and fusing to cut traffic.
- **Don't do QAT if PTQ already hits target.** We used PTQ throughout because, with per-channel scales and an fp16 head, it held accuracy inside budget. Quantization-aware training (see [QAT](/blog/machine-learning/edge-ai/quantization-aware-training-qat)) is the next lever *only if* PTQ leaves you short — it costs a full retraining cycle to typically recover the last 0.5–1.5 points. We didn't need it; many projects do, especially below 8 bits.
- **Don't unstructured-prune for speed on commodity edge hardware.** It produces sparsity numbers and no speedup. Use structured pruning, or N:M sparsity *only* if your target has the sparse tensor cores to exploit it.
- **Don't distill if the small model already matches the teacher.** Distillation earns its keep when a capacity gap costs accuracy. If your efficient backbone trained from scratch already matches the teacher, skip it.
- **Don't skip the deploy step thinking the model work is done.** The last 11 ms in this project came from warm-up, zero-copy buffers, and NMS — not the model. The notebook number is never the shipped number.

## Key takeaways

- **Measure honestly on the target before touching anything.** Warm-up, batch=1, sustained-load, per-stage. The decomposition tells you which lever to reach for first; without it you optimize the wrong thing.
- **Apply levers in bottleneck order, re-measuring between each.** Compute-bound → cut FLOPs (architecture, prune, quantize). Memory/tail-bound → cut traffic and postprocess (fuse, NMS). The bottleneck moves as you work; follow it.
- **Architecture is the biggest single lever for a compute-bound vision model**, but it costs accuracy — pair it with distillation to win the accuracy back. The two compose; neither does both jobs alone.
- **Only structured pruning speeds up commodity edge hardware.** Channel/filter pruning shrinks dense tensors; unstructured sparsity is storage savings and a slide, not speed.
- **Never quantize a detector uniformly.** The box-regression head has a wide dynamic range that int8 cannot represent finely; keep it fp16. A millisecond and two megabytes for three points of mAP is the best trade in the project.
- **Fusion and compilation are free accuracy-preserving speed** once you're memory-bound — they cut memory round-trips, not arithmetic, so they cost nothing numerically.
- **An unsupported op can silently exile the whole graph tail to the CPU.** Design your accelerator/CPU partition deliberately; keep NMS out of the accelerated graph on purpose.
- **The cold number is a lie for continuous use.** Stress-test under sustained capture, budget a throttle factor, and leave thermal margin — which the extra speed from quantization and fusion provides.
- **Ship the knee of the feasible region.** Draw the frontier, filter by hard constraints, pick the knee. Faster-than-needed at an accuracy cost is a dominated trade — decline it.

## Further reading

- [A taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) — the map of all four levers this case study applies in sequence.
- [The edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) — the series capstone that turns this single journey into a repeatable process.
- [Profiling and benchmarking on device](/blog/machine-learning/edge-ai/profiling-and-benchmarking-on-device) and [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) — the measurement and the theory behind "follow the bottleneck."
- [The MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) and [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas) — why the backbone swap worked and how to choose one by measured latency.
- [Knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) and [what to distill: response, feature, relation](/blog/machine-learning/edge-ai/what-to-distill-response-feature-relation) — recovering the accuracy the swap cost.
- [Structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up) — why only structured pruning helps on edge hardware.
- [Post-training quantization (PTQ)](/blog/machine-learning/edge-ai/post-training-quantization-ptq) and [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) — the int8 flow and the per-layer reasoning behind the fp16 head.
- [Inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared), [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization), and [TensorRT and GPU edge inference on Jetson](/blog/machine-learning/edge-ai/tensorrt-and-gpu-edge-inference-on-jetson) — the compile and fuse step on both targets.
- [Mobile deployment end to end](/blog/machine-learning/edge-ai/mobile-deployment-end-to-end) and [the accuracy-latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier) — shipping, warm-up, thermals, and reading the final point off the frontier.
- Howard et al., *Searching for MobileNetV3* (2019); Sanh et al., *DistilBERT* (2019); Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (2018); NVIDIA TensorRT and Google LiteRT documentation.

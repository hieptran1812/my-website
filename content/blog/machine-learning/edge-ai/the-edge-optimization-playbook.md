---
title: "The edge optimization playbook: from a trained model to one that ships"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The single decision document that takes you from a trained model and a target device to a shipped one — the ordered procedure, the lever-ordering, the decision trees, the checklist, and the anti-patterns, distilling the whole series."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "pruning",
    "knowledge-distillation",
    "inference",
    "efficient-ml",
    "deployment",
    "pareto-frontier",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/the-edge-optimization-playbook-1.png"
---

You have a trained model. It works. The validation numbers are good, the demo got applause, and then someone hands you the actual target: a Pixel phone, a Jetson Orin Nano, a Raspberry Pi 5, a Cortex-M7 with 256 KB of SRAM, or a four-year-old laptop with no GPU. The model that sailed through training now has to live inside a battery, a thermal envelope, a flash partition, and a p99 latency budget that an angry product manager will quote back to you in a meeting. The gap between "the model is trained" and "the model ships" is where most edge projects die — not because the technique was missing, but because nobody had a procedure.

This is that procedure. Everything in this fifty-post series — quantization from first principles, the roofline model, structured pruning, distillation, NAS, compilers, runtimes, KV-cache tricks, MLOps — was building toward this one page. The rest of the series is the *why* and the *how* of each lever; this post is the *order*. It is the page you keep open when you sit down with a trained checkpoint and a device, and it tells you, in sequence, exactly what to do: know your target, measure the baseline, pull four levers in a specific order, compile, manage memory, deploy, operate, and finally read your config off the accuracy–latency frontier. By the end you will have a repeatable decision procedure, two decision trees you can route any project through, a reference cheat sheet, a pre-flight checklist, and a catalogue of the anti-patterns that quietly waste a quarter of everyone's time.

Figure 1 is the whole procedure on one slide — the seven stages from a trained model to a shipped one. Keep it open. The single most important thing it encodes is *ordering*: you measure before you optimize, you pull the architectural lever before the quantization lever, and you read the final config off a frontier rather than picking the "best" point by gut. Almost every expensive mistake in edge ML is an ordering mistake.

![A timeline figure showing the eight ordered stages of shipping a model from knowing the target through measuring, pulling levers, compiling, managing memory, deploying, and picking the final configuration](/imgs/blogs/the-edge-optimization-playbook-1.png)

A word on the spirit of this document before we start. Every optimization is a transaction. It buys you size, speed, or energy and it charges you in accuracy, engineering effort, or hardware portability. There is no free lunch — but there *is* a free *order* in which to buy your lunches, an order that makes the wins compound and the costs stay small. Most of this playbook is that order, and the discipline to measure whether each transaction actually closed at the price you expected. If you internalize one habit, make it this: never claim a win you did not measure on the target device, batch size one, after warm-up. Everything else is commentary.

## 1. The procedure, in one breath

Here is the entire procedure, compressed, so you have the skeleton before we hang flesh on it. Read it once now and it will make the next nine sections feel like elaboration rather than novelty.

> **(0)** Know the target hardware and write down the budget — which of latency-p99, peak memory, energy, binary size, or accuracy-floor actually *binds*. **(1)** Measure the baseline honestly on the device and find the bottleneck: compute-bound or memory-bound. **(2)** Pull the four levers *in order* — efficient architecture / SLM-by-design first (it changes the whole family), then distillation to recover or transfer capability, then structured pruning for real speedups, then quantization last because it is the cheapest and the most hardware-coupled. **(3)** Compile the result and choose the runtime that owns your hardware. **(4)** Manage memory — peak working set, and for LLMs the decode-time KV-cache tricks. **(5)** Deploy end to end and stand up the operations: telemetry, drift, rollback. **(6)** Pick the final configuration off the accuracy–latency Pareto frontier and validate it on the device against the budget from step 0.

That is the spine. Notice what is *not* in it: there is no "try quantization and see." There is no "prune until it breaks." Each stage has an entry condition (you measured the previous one) and an exit condition (you measured this one). The procedure is a series of gates, and the whole point of gates is that you cannot skip one and pretend you did not.

The reason this works — the reason ordering matters at all — is that the four levers are not independent. Each one changes the object that the next one operates on. If you quantize first and prune second, the pruning invalidates the quantization scales you so carefully calibrated, and you either re-calibrate (wasting the first pass) or ship stale scales (losing accuracy). If you prune first and quantize second, the quantization calibrates against the *already-pruned* weight distribution, which is exactly what you want. The order is not aesthetic. It falls out of the dependency graph between the levers, and we will derive it properly in section 5.

We map this whole series onto a single picture in Figure 2: the four levers, the substrate they sit on, and the discipline that referees them. This is the field map that the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) post develops in full; here it is the legend for the playbook.

![A tree figure mapping the series into four levers architecture distillation pruning and quantization resting on a compiler and runtime substrate and refereed by on-device profiling](/imgs/blogs/the-edge-optimization-playbook-2.png)

## 2. Step 0 — Know the target and define the constraints

You cannot optimize toward a target you have not named. The single most common reason an edge project flails is that "make it faster" was never turned into a number on a specific chip. So step zero produces two artifacts: a description of the hardware, and a budget where exactly one constraint is marked as *binding*.

**Name the hardware precisely.** "A phone" is not a target. "The Tensor G3 NPU on a Pixel 8, with a CPU fallback for unsupported ops" is a target. The difference matters because the hardware decides which levers even *can* help. An NPU that has fast int8 matmul units but no int4 path makes int4 quantization pointless (it would dequantize to int8 anyway, or fall back to CPU). A GPU with 2:4 structured-sparsity tensor cores makes a specific *kind* of pruning suddenly worth 2× while making unstructured pruning still worth nothing. A microcontroller with 256 KB of SRAM makes peak activation memory, not FLOPs, the thing that decides whether your model runs at all. Before anything else, read the deep-dive on [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) and write down: the compute units (CPU/GPU/NPU/DSP), the supported dtypes, the on-chip memory sizes, and the memory bandwidth. Those four facts constrain everything downstream.

**Write the budget and mark what binds.** A complete edge budget has five entries, and you must know which one is the wall you are about to hit:

- **Latency, p99 — not p50.** The tail is what users feel and what SLAs are written against. A model with a 12 ms median and a 60 ms p99 is a 60 ms model in production. If you are doing real-time vision at 30 FPS you have a 33 ms frame budget and your *p99* has to fit inside it, with headroom for the rest of the pipeline.
- **Peak memory.** Not the model file on disk — the *peak working set* during inference: weights, plus the largest activation tensor live at once, plus the runtime's own arena, plus (for LLMs) the KV cache. On a microcontroller this is the constraint that decides feasibility.
- **Energy per inference.** On battery devices the right unit is millijoules or mWh per inference, not just milliseconds. A faster kernel that runs the NPU at a higher clock can cost *more* energy per inference even though it finishes sooner.
- **Binary / model size.** Flash and app-download budgets are real. Shaving an LLM from 14 GB to 4 GB is the difference between "runs on the laptop" and "does not fit in RAM at all."
- **Accuracy floor.** The minimum quality below which the product is broken. This is the constraint every other lever is trading against, so it has to be a hard number measured on a representative eval set, not a vibe.

The deep-dive on [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) goes into how to define and measure each of these. The discipline here is to identify the *binding* constraint — the one you will hit first — because that determines which levers to reach for, which is the entire content of decision tree number one in section 6. If memory is the wall, you optimize differently than if compute is the wall, and you waste enormous effort if you optimize the wrong one.

Two of these five constraints deserve a word because they trip up people who only ever optimized for latency in the cloud. **Energy is not just latency in disguise.** Modern SoCs use dynamic voltage and frequency scaling, so a kernel that runs at a higher clock finishes sooner but draws disproportionately more power — energy scales roughly with $V^2 f$, and pushing $f$ up usually means pushing $V$ up too. The consequence is a "race-to-idle vs run-slow" trade-off: sometimes the lowest-energy choice is to run the model *slower* on a more efficient unit (a DSP instead of the big CPU cores) and let the chip return to its deep-sleep state faster. If your product is a wearable or an always-on sensor, energy per inference, measured in millijoules with the screen off, is the number that decides battery life, and it does not always move with latency. **Peak memory is a feasibility gate, not a performance dial.** Latency degrades gracefully — a slow model is annoying. Memory does not — a model that needs one kilobyte more than the arena holds simply does not run, and on a microcontroller there is no swap to bail you out. Treat the memory budget as a hard wall you size *before* you start, not a number you discover when the device reboots.

#### Worked example: writing the budget for a TinyML keyword spotter

Target: a Cortex-M7 at 480 MHz with 512 KB of flash and 256 KB of SRAM, always-on keyword spotting (wake-word detection) running off a coin cell. Budget: model + arena ≤ 200 KB of SRAM (the rest is the OS and audio buffers); flash ≤ 256 KB; energy ≤ 1 mJ per inference (it runs ~10× a second, so a coin cell has to last months); latency ≤ 20 ms (so a wake word is detected promptly); accuracy floor ≥ 94% on the keyword set. The binding constraint here is unambiguously **peak memory** — 256 KB of SRAM is the wall, and it decides feasibility before speed even enters the conversation. That one fact routes the whole plan into the size-bound branch: a tiny-by-design network, aggressive distillation from a larger model, int8 everything, and a fixed tensor arena with no dynamic allocation, all targeting [TFLite-Micro on a microcontroller](/blog/machine-learning/edge-ai/tinyml-on-microcontrollers). Quantization here is not optional polish; it is the difference between fitting and not fitting, because int8 weights are a quarter the size of fp32 and the activation buffers shrink proportionally. The deep-dive on [squeezing models into kilobytes](/blog/machine-learning/edge-ai/squeezing-models-into-kilobytes) is the full version of this exercise.

#### Worked example: writing the budget for a CV phone target

Target: Pixel 8, real-time semantic segmentation in a camera app, 30 FPS. Budget: p99 latency ≤ 25 ms on the NPU (leaving ~8 ms of the 33 ms frame for capture, pre/post-processing, and rendering); peak memory ≤ 200 MB (the camera pipeline already holds buffers); model size ≤ 25 MB (download size matters); accuracy floor mIoU ≥ 0.72 on the validation set (below this the masks look visibly wrong). The binding constraint here is almost always **latency** — segmentation is compute-heavy and the frame budget is brutal. We write that down and route accordingly. Energy matters but is secondary; size is comfortable. That one line — "latency binds" — will dictate that we reach for architectural FLOP reduction and pruning before we reach for quantization, because quantization mostly helps *memory-bound* ops and we are about to confirm this op is compute-bound.

## 3. Step 1 — Measure the baseline and find the bottleneck

You now have a budget. The next gate is the baseline: where does the unoptimized model actually stand on the target, and *why*. This is the step everyone is tempted to skip, and skipping it is how you spend three days quantizing a model only to find it was never compute-bound and quantization bought you nothing.

**Measure on the device, honestly.** Not on your workstation GPU, not in the cloud, not the FLOP count from `torchinfo`. On the device, batch size one (the edge reality), after warm-up, with the thermal state you will actually run in. The deep-dive on [profiling and benchmarking on device](/blog/machine-learning/edge-ai/profiling-and-benchmarking-on-device) covers the traps in detail; the short version is below.

```python
import time
import numpy as np
import torch

@torch.inference_mode()
def bench(model, example_input, warmup=20, iters=200, device="cpu"):
    model = model.to(device).eval()
    x = example_input.to(device)

    # Warm-up: the first runs pay for lazy kernel selection,
    # graph capture, allocator growth, and clock ramp-up.
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()  # CUDA is async; never time without this
        samples.append((time.perf_counter() - t0) * 1e3)  # ms

    samples = np.array(samples)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p99_ms": float(np.percentile(samples, 99)),  # the number that ships
        "mean_ms": float(samples.mean()),
        "std_ms": float(samples.std()),
    }
```

Three traps that silently corrupt edge measurements. **Cold start:** the first dozen inferences pay for kernel selection, graph capture, and clock ramp; if you time those you over-report latency by 2–5×. Always warm up. **Async execution:** on CUDA (Jetson) the Python call returns before the GPU finishes; if you do not `synchronize()` you are timing the launch, not the work, and your numbers are fiction. **Thermal throttling:** run a sustained load for a minute and watch the clocks; a model that hits 20 ms in the first second can settle to 35 ms once the SoC heats up and downclocks. Your p99 must be measured *in the thermal steady state you ship in*, not in the first cool second.

**Find the bottleneck: compute-bound or memory-bound.** This is the most important diagnostic in the whole playbook, because it tells you which lever can possibly help. The tool is the roofline model, covered in full in [the roofline model and where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives). The science, briefly, because it earns its place here.

Every operation has an **arithmetic intensity** $I$, defined as the number of floating-point operations it performs per byte of data it moves from memory:

$$ I = \frac{\text{FLOPs}}{\text{bytes moved}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right] $$

The hardware has two ceilings: peak compute $\pi$ (FLOP/s) and peak memory bandwidth $\beta$ (byte/s). The attainable performance for an op of intensity $I$ is

$$ P(I) = \min\left(\pi,\; \beta \cdot I\right). $$

The crossover happens at the **ridge point** $I^\* = \pi / \beta$. If your op's intensity $I < I^\*$, you are **memory-bound** — performance is $\beta I$, you are limited by data movement, and the only way to go faster is to *move fewer bytes*. If $I > I^\*$, you are **compute-bound** — performance is $\pi$, you are limited by raw arithmetic, and the only way to go faster is to *do fewer FLOPs*. This single inequality decides your entire lever strategy:

- **Memory-bound** ($I < I^\*$): quantization helps enormously, because halving the bytes-per-weight from fp16 to int8 directly halves the data moved and roughly doubles $\beta I$. KV-cache tricks help LLM decode. Pruning structure helps if it removes whole tensors you would otherwise stream. Cutting FLOPs alone does *nothing* — you were never compute-limited.
- **Compute-bound** ($I > I^\*$): cutting FLOPs helps — a cheaper architecture, structured pruning that removes real channels. Quantization helps only if the hardware actually runs the lower-precision *math* faster (int8 tensor cores), not merely stores it smaller; otherwise it dequantizes and you saved bandwidth you were not spending.

Here is the punchline that should reorganize how you think: **LLM decoding is overwhelmingly memory-bound** (each token reads the entire weight matrix to do one tiny matmul — intensity well below the ridge), which is *why* weight-only quantization is the single highest-leverage LLM optimization and why cutting FLOPs barely moves the needle. **CNN inference at batch one is usually compute-bound** in the convolution layers, which is why architectural FLOP reduction and pruning matter more there. Same two levers, opposite priorities, decided by one inequality. Measure $I$ relative to $I^\*$ and you know which lever to pull before you pull anything.

Make the LLM-decode claim concrete, because it is the most counterintuitive and the most consequential. A single decode step multiplies the hidden vector (one token, so a $1 \times d$ row) by each weight matrix. For a weight matrix of shape $d \times d$ that is roughly $2 d^2$ FLOPs, and it reads $d^2$ weights. In fp16, that is $2$ bytes per weight, so the intensity is

$$ I_{\text{decode}} = \frac{2 d^2 \ \text{FLOPs}}{2 d^2 \ \text{bytes}} = 1 \ \frac{\text{FLOP}}{\text{byte}}. $$

One FLOP per byte. Compare that to a typical accelerator ridge point of $I^\* = \pi / \beta$ in the range of 50–200 FLOP/byte. The decode step sits two orders of magnitude *below* the ridge — pinned to the bandwidth roof, nowhere near the compute roof. That single number explains the entire on-device-LLM playbook: you are bandwidth-starved, so every technique that moves *fewer bytes per token* (4-bit weights, int8 KV cache, fewer KV heads) is a near-linear speedup, while every technique that does *fewer FLOPs* (pruning, cheaper attention math) is almost wasted, because you were never paying for the FLOPs. The prefill phase, which processes the whole prompt at once, has high intensity and *is* compute-bound — which is why prefill and decode want different optimizations, and why "tokens per second" is two different numbers you must report separately.

Now stress-test the diagnosis, because the roofline lies if you read it lazily. **What if the op is a tiny matmul that does not even saturate the memory bus?** Then you are *launch-bound* — dominated by kernel-launch overhead and Python dispatch, not by compute or bandwidth — and the fix is operator fusion or CUDA graphs, not any of the four levers. **What if the NPU does not support your op and silently falls back to the CPU?** Then your beautifully quantized graph is running a slow fp32 path on the wrong unit, and your roofline (drawn for the NPU) is the wrong chart entirely; check the delegate's fallback log before you trust any number. **What if the model is compute-bound on the GPU but memory-bound on the CPU you actually ship to?** The bottleneck is a property of the (op, hardware) pair, not the op alone — which is the whole reason step 0 demands you name the exact target. A correct diagnosis on the wrong chip is still a wrong diagnosis.

#### Worked example: the baseline on the phone

We profile the segmentation baseline (an fp32 model, ~50 MB, ~90 ms p99 on the Pixel 8 NPU after warm-up — well over our 25 ms budget). The roofline shows the depthwise-separable conv stages sitting *near the ridge* and the dense convs sitting clearly **compute-bound**. Diagnosis: we need fewer FLOPs (architecture + pruning) *and* we will get a memory/bandwidth bonus from quantization, in that order. If the roofline had shown everything memory-bound, we would have gone quantization-first and skipped the expensive architecture surgery. Three days of profiling-driven thinking just saved three weeks of optimizing the wrong thing.

## 4. Step 2 — Pull the four levers in order

This is the heart of the playbook. You have a measured baseline and a known bottleneck. Now you pull the four levers — and you pull them in a specific order, biggest and coarsest first, quantization last. Figure 3 shows the order as a timeline; the rest of this section is the justification, lever by lever.

![A timeline figure showing the compound compression order architecture then distillation then pruning then quantization then compile then validate](/imgs/blogs/the-edge-optimization-playbook-3.png)

### Lever 1 — Architecture / SLM-by-design (biggest, changes the family)

The largest, cheapest win is the one you make before training is even finished: choose a model *family* that is efficient by construction. Replacing standard convolutions with depthwise-separable ones, full attention with an efficient variant, or a 13B LLM with a well-trained 3B one is not a 10% improvement — it is a 3–10× change in the FLOP and parameter budget, and it sets the ceiling on everything the other three levers can do afterward. You cannot quantize your way out of a fundamentally oversized model; you can only quantize your way *down a model that was already the right shape*.

The building blocks are the subject of [building blocks for efficient models](/blog/machine-learning/edge-ai/building-blocks-for-efficient-models), the automated search for good ones is [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas), and the LLM-specific version — picking or training a genuinely small model rather than crushing a large one — is [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design). The key idea binding all three: optimize for *latency on the target*, not for FLOPs in a spreadsheet. A depthwise-separable block has far fewer FLOPs than a dense conv but a much lower arithmetic intensity, so on a bandwidth-poor device it can be *slower* per FLOP. Hardware-aware NAS exists precisely because the FLOP-latency relationship is non-monotone and device-specific; you have to search against the real chip.

Why first? Because the architecture decision is the only one that changes the *family* of the model, and every later lever calibrates against whatever family you chose. Pruning a bloated architecture down to size is strictly worse than starting with the right architecture and pruning it lightly — you spend training compute fighting a shape you should not have picked. Architecture is the foundation; lay it first.

### Lever 2 — Distillation (recover and transfer capability)

Once you have a smaller architecture, it will, on its own, be less accurate than the big model it replaces. Distillation is how you close that gap: train the small "student" to mimic the large "teacher," transferring not just the right answers but the teacher's *soft* probability distribution, which carries far more information per example than a hard label. The fundamentals are in [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals).

The science, briefly: a hard one-hot label tells the student "this is a cat." The teacher's softened logits tell it "this is a cat, 70%, but it is 12% dog, 8% fox, and definitely not a truck" — the relative dark knowledge of which wrong answers are *plausible* is a much richer training signal. With temperature $T$, the student matches the softened teacher distribution

$$ p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}, $$

and the loss blends the distillation term (KL divergence to the teacher) with the ordinary task loss:

$$ \mathcal{L} = \alpha\, T^2 \cdot \mathrm{KL}\!\left(p^{\text{teacher}} \,\|\, p^{\text{student}}\right) + (1 - \alpha)\, \mathcal{L}_{\text{task}}. $$

The $T^2$ factor rescales the distillation gradient back to the same magnitude as the task gradient, since softening by $T$ shrinks the logits' gradients by $1/T^2$. The result, on a real example: DistilBERT recovered roughly 97% of BERT's GLUE score at 40% fewer parameters and ~60% faster inference — a near-pure win bought with a distillation run.

Why second, before pruning and quantization? Because distillation *adds capability back*, and you want to do that while the model is still in full precision and full density, where training is well-behaved and gradients are clean. Distilling into a model you have already pruned and quantized is fighting noisy gradients and a damaged loss surface. Recover the capability first, then start cutting.

There is a deeper reason distillation pairs so well with the other levers: it lets you *decouple* the architecture decision from the accuracy decision. Without it, choosing a 3× smaller backbone means accepting whatever accuracy that backbone reaches when trained from scratch on your labels — usually disappointing, because small models are hard to train well from hard labels alone. With distillation, the big model becomes a *teacher* that supplies a rich, smooth target, and the small student routinely reaches accuracy it could never hit on its own. This is why "pick a smaller architecture" is not the scary accuracy cliff it looks like: distillation is the safety net underneath the architectural lever, and the two are best thought of as one move — *shrink the family, then distill the capability back in*. For the LLM and reasoning-model variants of this, where you distill not just labels but chains of reasoning, see [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning).

### Lever 3 — Structured pruning (real speedups, not just sparsity)

Now you remove what the model is not using. The critical distinction — and the single most expensive misunderstanding in pruning — is **structured vs unstructured**. Unstructured pruning zeros out individual weights, producing a sparse matrix that is *smaller in theory* but runs at *exactly the same speed* on commodity hardware, because a CPU/GPU/NPU dense matmul kernel does not skip zeros; it multiplies by them just as fast as by anything else. You need either specialized sparse kernels (rare and often slower at moderate sparsity) or hardware that natively skips zeros (the 2:4 sparse tensor cores on recent NVIDIA GPUs) for unstructured sparsity to buy speed.

Structured pruning removes *whole* channels, filters, or attention heads. That shrinks the actual dimensions of the weight tensors, so the dense kernel simply has less work — a real, portable, no-special-hardware speedup. This is why the playbook reaches for structured pruning and treats unstructured sparsity as a special case requiring matching hardware. The full argument is in [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up).

```python
import torch
import torch.nn.utils.prune as prune

# Structured pruning: remove whole output channels (dim=0) of a conv,
# ranked by L2 norm. This SHRINKS the tensor shape -> real speedup
# after you physically remove the channels and re-fold the graph.
conv = model.backbone.layer3[0].conv2
prune.ln_structured(conv, name="weight", amount=0.3, n=2, dim=0)

# prune.* leaves a mask; you must make it permanent and then actually
# rebuild the smaller layer (e.g. via torch-pruning / FX) to get speed.
prune.remove(conv, "weight")
# ^ Without rebuilding to a genuinely smaller tensor, the zeros still
#   get multiplied and you measure ZERO speedup. This is the #1 trap.
```

After pruning you **fine-tune** to recover the accuracy the cut cost — often most of it comes back in a few epochs because the pruned channels were carrying little signal. The reason pruning comes *after* distillation and *before* quantization: pruning needs a full-precision model to fine-tune cleanly (so it follows distillation), and it changes the weight distribution that quantization must calibrate against (so it precedes quantization). Prune, fine-tune, *then* quantize against the final distribution.

### Lever 4 — Quantization (cheapest, most hardware-coupled, last)

Quantization is last for two reasons: it is the cheapest to apply (post-training quantization needs no retraining, just a calibration pass), and it is the most hardware-coupled (its speedup depends entirely on whether the target has fast low-precision math units). Doing it last means it calibrates against the *final* weights — already architected, distilled, and pruned — so the scales are correct for what actually ships.

The science you must understand to deploy it safely: quantization maps a real value $x$ to an integer $q$ with a scale $s$ and zero-point $z$,

$$ q = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right) + z,\; q_{\min},\; q_{\max}\right), \qquad \hat{x} = s\,(q - z). $$

Rounding introduces an error uniformly distributed on $[-s/2, s/2]$, with variance $s^2/12$. For a $b$-bit signed quantizer covering range $\pm A$, the step is $s = 2A / 2^b$, and the signal-to-quantization-noise ratio works out to the famous law

$$ \text{SQNR} \approx 6.02\,b + 1.76 \ \text{dB}. $$

Every bit you add buys about **6 dB** of headroom. That is *why* int8 (8 bits, ~50 dB) is usually safe and int4 (4 bits, ~26 dB) is dangerous — you have thrown away 24 dB of signal fidelity, and whether the model survives depends entirely on how much redundancy it had. It also tells you *where* quantization hurts: layers with large dynamic range (a few huge outlier activations stretch $A$, blowing up the step $s$ for everyone else) lose the most, which is exactly the failure mode SmoothQuant and AWQ exist to fix.

The series has a full quantization track. Start with the principles in [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles), the practical calibration-based flow in [post-training quantization (PTQ)](/blog/machine-learning/edge-ai/post-training-quantization-ptq), and the LLM-specific weight-only methods in [LLM quantization: weight-only GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq). The PTQ flow in PyTorch:

```python
import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

qconfig = get_default_qconfig("x86")          # or "qnnpack" for ARM/mobile
qmap = QConfigMapping().set_global(qconfig)

prepared = prepare_fx(model.eval(), qmap, example_inputs=(example_input,))

# Calibration: observers learn activation ranges from representative data.
# A few hundred samples is plenty; they MUST resemble production inputs.
with torch.inference_mode():
    for x in calibration_loader:        # ~200-500 real-distribution samples
        prepared(x)

quantized = convert_fx(prepared)        # fold scales, emit int8 kernels
```

For an LLM you would instead run GPTQ or AWQ to produce 4-bit weights and ship a `Q4_K_M` GGUF via `llama.cpp`. For activation quantization and the KV cache, see [LLM quantization: activations, SmoothQuant, KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache). If PTQ drops you below the accuracy floor, escalate to **quantization-aware training (QAT)**, which simulates quantization during fine-tuning so the model learns weights robust to it — covered in [quantization-aware training (QAT)](/blog/machine-learning/edge-ai/quantization-aware-training-qat). The rule: PTQ first (cheap), QAT only if PTQ misses the floor.

When PTQ falls just short, you rarely need to quantize *everything* aggressively — you need to find the few layers that are doing all the damage and spare them. This is **mixed-precision via sensitivity analysis**: quantize one layer at a time, measure the accuracy drop that layer alone causes, and rank. A small number of layers — almost always the first conv (it sees raw, wide-range input), the final classifier or LM head (its outputs feed a softmax where small errors flip predictions), and the attention output projections in transformers — account for most of the loss. Keep those in int8 or fp16 and quantize the robust middle to int4, and you recover most of the accuracy at most of the size win. The per-layer probe is cheap:

```python
import copy, torch

def layer_sensitivity(fp_model, quantize_layer, eval_fn, layer_names):
    """Quantize ONE layer at a time, measure the accuracy it costs.
    eval_fn(model) -> accuracy on a held-out set. Higher drop = keep
    that layer in higher precision (it is sensitive)."""
    base = eval_fn(fp_model)
    drops = {}
    for name in layer_names:
        probe = copy.deepcopy(fp_model)
        quantize_layer(probe, name)            # int8/int4 just this layer
        drops[name] = base - eval_fn(probe)    # accuracy points lost
    # Sort: the biggest droppers stay in higher precision.
    return dict(sorted(drops.items(), key=lambda kv: -kv[1]))
```

The full method, including the Hessian-trace heuristic that estimates sensitivity *without* re-evaluating every layer, is in [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis).

Stress-test the precision knob. **What happens at int4?** From the SQNR law you are at ~26 dB — fine for a highly redundant 7B LLM whose weights are deeply over-parameterized (this is why 4-bit LLMs work), but often catastrophic for a small, already-efficient CNN that has no redundancy left to spend. The rule is counterintuitive but reliable: *bigger, more redundant models tolerate more aggressive quantization*, because they had slack; the tiny edge models you most want to shrink are the ones that least tolerate it. **What happens when the calibration set is tiny or unrepresentative?** The observers learn the wrong activation ranges — typically too narrow, because the rare large activations never appeared in calibration — and at inference those outliers clip to $q_{\max}$, which is exactly the kind of error a softmax amplifies. A few hundred *representative* samples beats ten thousand from the wrong distribution. **What happens when activations have outliers?** A handful of huge activation values stretch $A$, inflating the step $s$ for every normal value and crushing their effective resolution — the precise pathology SmoothQuant fixes by migrating the difficulty from activations into weights, and AWQ fixes by protecting the salient weight channels. If your LLM loses several points at int8 activations, outliers are almost always the reason.

The whole compose-and-order rationale — which levers multiply, which conflict — is the subject of [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression); this section is the operational distillation of it.

## 5. Why this order, and the compound-compression recipe

Let me make the ordering rigorous, because "biggest first, quantize last" is a heuristic and you deserve the derivation. The recipe is **architecture → distill → prune → quantize → compile**, and the order falls out of three principles.

**Principle 1: coarse before fine.** Each lever operates at a granularity. Architecture changes the *family* (coarsest). Distillation changes the *weights wholesale* via training. Pruning changes *which structures exist*. Quantization changes *the precision of the numbers that remain* (finest). A coarse change invalidates the work of any fine change made before it: if you quantize and *then* swap the architecture, the quantization was wasted. Order coarse-to-fine and no earlier work is ever invalidated by later work.

**Principle 2: each later step calibrates against the earlier output.** Distillation transfers the teacher into *this* architecture. Pruning fine-tunes *this* distilled model. Quantization calibrates scales against *this* pruned weight distribution. Reverse any pair and the later step calibrates against a distribution that no longer exists. The clearest case: quantize-then-prune. You calibrate int8 scales against the dense weights; then you prune, which removes the largest-magnitude channels in some layers, shrinking the true dynamic range — and now your scales are *stale*, sized for a range the tensor no longer has, wasting precious quantization levels on empty space. Prune-then-quantize calibrates against the final, pruned range. Correct by construction.

**Principle 3: cheapest and most hardware-coupled last.** Quantization is the cheapest lever (PTQ is a calibration pass, no training) and the most hardware-specific (its payoff depends on the target's low-precision units). Putting it last means (a) you have not sunk training cost into it that an earlier change might invalidate, and (b) you quantize for the *actual deployment hardware* once everything else is fixed, rather than re-quantizing every time an upstream lever changes the weights. Cheap, late, hardware-final.

Put together: architecture sets the shape, distillation fills the shape with capability, pruning trims the shape, quantization sets the precision, and the compiler lowers the final result onto silicon. Each step's input is the previous step's measured output. That is the compound-compression recipe, and when you follow it the wins **multiply** instead of cancelling — a 2× from architecture, 1.5× from pruning, and 2× from quantization compose toward 6× rather than collapsing to the 2× you would get if a later step undid an earlier one.

### Why "multiply" is the right word — and when it stops being true

The multiplication is not a slogan; it follows from *what each lever measures*. Architecture and pruning both reduce **FLOPs and bytes** (fewer ops, fewer weights), and quantization reduces **bytes per remaining weight**. Total inference cost, to a first approximation, is

$$ \text{cost} \approx (\text{number of operations}) \times (\text{cost per operation}), $$

and the levers act on *different factors of that product*: architecture and pruning shrink the first factor, quantization shrinks the second. Factors multiply. If architecture and pruning together cut FLOPs by $3\times$ and quantization cuts the per-FLOP byte cost by $2\times$, and your op was memory-bound so cost tracks bytes, you get roughly $3 \times 2 = 6\times$. The wins compose *because the levers are not fighting for the same factor*.

The compounding breaks in exactly two situations, and both are predictable. **First, when two levers attack the same factor**, you get diminishing returns, not multiplication: pruning 50% of channels and *then* expecting another distillation pass to halve the model again is double-dipping on the parameter-count factor, and the second cut bites into signal the first one already thinned. **Second, when the bottleneck shifts mid-stack.** Say you start compute-bound and pruning cuts FLOPs by $3\times$; you might cross the ridge point and become *memory-bound*, at which point further FLOP cuts (more pruning) do nothing and the next real win has to come from quantization. This is why you **re-profile after every lever** (gate 4 of the checklist): the bottleneck you diagnosed in step 1 is only guaranteed true for the *baseline*; each lever can move it. The honest practitioner re-draws the roofline after each cut and lets the *current* bottleneck pick the *next* lever, rather than running the whole plan on a stale diagnosis. That single habit — re-diagnose, do not assume — is the difference between the spreadsheet's 6× and the device's actual 6×.

### The anti-patterns (read this before you write code)

The corollary of a correct order is a catalogue of wrong ones. Figure 7 collects the worst offenders; here is the reasoning behind each.

![A matrix figure cataloguing edge optimization anti-patterns each with its failure mode severity and the correct move to make instead](/imgs/blogs/the-edge-optimization-playbook-7.png)

- **Quantize-then-prune** → stale scales. As derived above: pruning after calibration changes the dynamic range your scales were sized for. Fix: prune then quantize.
- **Banking on unstructured sparsity for speed on commodity hardware** → zero speedup. A dense kernel multiplies by zeros at full speed. Fix: use structured pruning, or 2:4 sparsity *only if* your hardware has sparse tensor cores.
- **Optimizing FLOPs instead of latency** → no win on memory-bound ops, or even a regression. A lower-FLOP block with lower arithmetic intensity can be slower on a bandwidth-poor device. Fix: profile against the roofline; optimize the metric that binds.
- **Measuring wrong** → shipping a number that does not exist. Cold-start timing, un-synchronized CUDA, p50 instead of p99, no thermal soak. Fix: warm up, synchronize, report p99, measure in steady thermal state.
- **Double-counting stacked wins** → the spreadsheet says 16×, the device says 5×. Each lever's win was measured in isolation, against a baseline the *other* levers also changed. Fix: measure the full stack end to end, once, on the device.
- **Ignoring p99 / energy / peak memory** → SLA misses, dead batteries, OOM crashes. The mean looks fine; the tail kills you, the battery dies, or the peak activation tensor blows the SRAM arena. Fix: budget all five constraints in step 0 and validate against each.

## 6. The decision trees — routing your project

Two trees route any project to the right levers. The first asks *what constraint binds*; the second asks *what kind of model* you have. Together they get you from "I have a model and a device" to "here is my ordered lever plan" in two questions.

### Tree 1 — Which levers for my binding constraint?

Figure 4 is the tree. The logic, following directly from the roofline diagnosis in section 3:

![A tree figure routing each binding constraint memory-bound compute-bound size-bound and accuracy-critical to its recommended set of optimization levers](/imgs/blogs/the-edge-optimization-playbook-4.png)

- **Memory-bound** (the op moves more bytes than it computes; $I < I^\*$): reach for **quantization first** (it halves or quarters bytes-per-weight, directly raising attainable performance) and, for LLMs, the **KV-cache tricks** (KV quantization, paged attention, grouped-query attention) that shrink the per-token memory traffic that dominates decode. Cutting FLOPs is wasted effort here. This is the default for **LLM decoding**.
- **Compute-bound** ($I > I^\*$): reach for **fewer FLOPs** — an efficient **architecture** and **structured pruning** that physically removes channels. Quantization helps only if the hardware runs the low-precision *math* faster. This is the default for **batch-one CNN inference**.
- **Size-bound** (it has to *fit*, in flash or RAM, before speed even matters): stack **quantization + pruning + distillation** — quantization for bytes-per-weight, structured pruning for parameter count, distillation to fit capability into a genuinely smaller model. This is the **microcontroller / TinyML** default, where 256 KB is the wall.
- **Accuracy-critical** (the floor is tight and every cut is dangerous): be conservative — **QAT** instead of PTQ (the model learns to tolerate quantization), **distillation** to claw back capability, and **mixed precision** (keep the sensitive layers — first conv, last classifier, attention outputs — in higher precision and quantize only the robust middle), guided by per-layer sensitivity analysis. See [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) for the per-layer method.

The trees are not mutually exclusive — a real project is usually "compute-bound *and* size-bound with a tight floor," in which case you walk all the relevant branches and order the union of levers by the section-5 recipe. The tree tells you *which* levers; the recipe tells you *in what order*.

### Tree 2 — CNN / vision vs LLM vs MCU/TinyML

Figure 5 routes by model class to the right *track* and the right default toolchain. The model class is a strong prior on the bottleneck and therefore on the levers, which is why this is a useful first cut.

![A tree figure routing model classes vision LLM and microcontroller to their default tracks and starting toolchains](/imgs/blogs/the-edge-optimization-playbook-5.png)

- **CNN / vision** → usually compute-bound at batch one; optimize for *frame latency*. Default toolchain: **TFLite/LiteRT int8** for mobile NPUs, or **TensorRT** for Jetson. Track: efficient architecture → structured pruning → int8 PTQ → compile. The end-to-end version is [case study: real-time vision on device](/blog/machine-learning/edge-ai/case-study-real-time-vision-on-device).
- **LLM** → memory-bound at decode; optimize for *tokens per second* and *peak memory*. Default toolchain: **GPTQ/AWQ 4-bit** weights, **`llama.cpp` GGUF** (`Q4_K_M`) for CPU/laptop or **MLC-LLM** for mobile GPUs. Track: pick a small base model → weight-only 4-bit quant → KV-cache tricks → compile. See [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast) and the worked end-to-end in [case study: an LLM assistant on a laptop](/blog/machine-learning/edge-ai/case-study-an-llm-assistant-on-a-laptop).
- **MCU / TinyML** → memory-bound by *capacity*, not bandwidth; the KB budget decides feasibility. Default toolchain: **TFLite-Micro + CMSIS-NN**, int8 everything, no dynamic allocation (a fixed tensor arena). Track: tiny-by-design architecture → aggressive distillation → int8 → fit the arena. The constraint is so tight it gets its own discipline.

## 7. Step 3 — Compile and choose the runtime

You have a compressed model. Now lower it onto the silicon. This is the substrate step — the cheapest wins in the entire field, because a good compiler costs you *zero accuracy* and routinely buys 1.3–2× by fusing operators, picking memory layouts, and selecting hand-tuned kernels. The mechanism: a naive graph writes each layer's output to DRAM and reads it back for the next op; a fusing compiler keeps intermediates on-chip and writes once, moving far less data — which, recall from the roofline, is exactly what speeds up a memory-bound graph.

The artifact-export step (from PyTorch/TF to a portable graph) is [from model to deployable artifact](/blog/machine-learning/edge-ai/from-model-to-deployable-artifact); the runtime comparison is [inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared); the graph transforms themselves are [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization). The decision is mostly "which runtime owns my hardware":

- **NVIDIA Jetson / any NVIDIA GPU** → **TensorRT**. Build an engine with `trtexec`, enable int8 with a calibrator, let it fuse and pick kernels for your exact GPU.
- **Android / mobile NPU** → **TFLite / LiteRT** with the NNAPI or vendor delegate; or **ONNX Runtime** with the right execution provider.
- **Apple devices** → **Core ML** to reach the ANE; palettize weights with `coremltools`.
- **CPU / laptop LLM** → **`llama.cpp`** (GGUF, k-quants); for mobile GPU LLMs, **MLC-LLM**.

```bash
# TensorRT: build an int8 engine from an ONNX model on a Jetson.
# The compiler fuses ops and selects kernels for THIS GPU.
trtexec \
  --onnx=model.onnx \
  --int8 \
  --calib=calibration.cache \
  --saveEngine=model.int8.plan \
  --workspace=2048 \
  --useCudaGraph        # capture the graph to kill per-launch overhead
```

The non-obvious rule, and the reason compilation appears *after* the levers in the procedure: **re-compile after every lever.** Compression changes which kernels are optimal — after quantization the fast path is an *int8* fused conv, a different code path the compiler can only choose if you re-lower the model. Compile once on the baseline to get your honest starting number, then compile again on the final compressed model to get your honest shipping number. A surprising amount of "my quantized model is slower" turns out to be "I never re-lowered it, so it is running a dequantize-to-fp16 fallback."

## 8. Step 4 — Manage memory (and, for LLMs, decode)

Speed is not the only wall; for many edge targets, memory is *the* wall, and it is a different kind of problem because it decides feasibility, not just performance. The deep-dive is [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint). Three quantities to budget, all distinct from the on-disk model size:

- **Weights resident in memory.** After quantization this is roughly model-size; for a 4-bit 7B LLM, ~4 GB. This sets the floor on RAM.
- **Peak activation memory.** The largest set of activation tensors alive simultaneously during the forward pass. A runtime that reuses buffers (in-place ops, memory planning) can cut this dramatically; a naive one holds everything and OOMs. On a microcontroller this *is* the constraint: the tensor arena must hold the peak working set in SRAM, and if it does not, the model simply cannot run regardless of how fast it would be.
- **KV cache (LLMs).** This is the one that surprises people. The KV cache grows *linearly* with sequence length and dominates memory at long context. For a model with $L$ layers, $H$ KV heads, head dimension $d$, sequence length $T$, batch $B$, in $p$ bytes per element, the cache is

$$ \text{KV bytes} = 2 \cdot B \cdot L \cdot H \cdot d \cdot T \cdot p. $$

The factor of 2 is keys and values. For a 7B model at 8K context this is gigabytes — often *larger than the quantized weights*. This is why decode-time tricks are mandatory for on-device LLMs: **KV-cache quantization** (store K/V in int8 or int4, halving or quartering $p$), **grouped-query attention** (fewer KV heads $H$), and **paging** (allocate cache in blocks so you do not reserve worst-case contiguous memory). The full set is in [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast). The decode loop is memory-bound (section 3), so every byte you shave off the per-token KV traffic is a direct latency win as well as a feasibility win — a rare two-for-one.

#### Worked example: the LLM laptop target end to end

Target: a 7B-class assistant on an M2 MacBook Air (no discrete GPU, 16 GB unified memory), interactive chat, want ≥ 15 tokens/s and a model that leaves room for the rest of the laptop. Baseline: fp16 weights are ~14 GB — they barely fit and leave nothing for the KV cache or the OS, and decode crawls because every token streams 14 GB through memory (deeply memory-bound). Plan, by the trees: this is an **LLM**, so route to the LLM track; the binding constraint is **memory** (both capacity and bandwidth). Levers: pick a strong 7B base → **weight-only 4-bit quantization** (GPTQ or AWQ) bringing weights to ~4 GB → ship as a **`Q4_K_M` GGUF** under `llama.cpp` → **quantize the KV cache to int8** so long chats do not blow the budget → let `llama.cpp` use Metal for the GPU-able parts. Result, in the ballpark reported for this class of setup: ~3.5× smaller (14 GB → ~4 GB), comfortably resident, and decode jumps from single-digit to ~20–30 tokens/s because we are moving a quarter of the bytes per token, with perplexity degradation small enough to be imperceptible in chat. Notice the order: quantization did almost all the work *because the bottleneck was memory* — exactly what tree 1 predicted. The full build-and-measure walkthrough is the [case study: an LLM assistant on a laptop](/blog/machine-learning/edge-ai/case-study-an-llm-assistant-on-a-laptop).

## 9. Step 5 — Deploy, and step 6 — operate and pick the config

The model now fits, runs, and hits the budget on the bench. Two steps remain, and skipping them is how a working prototype becomes a 2 a.m. page.

**Deploy end to end.** Package the compiled artifact, wire up pre/post-processing on-device (a surprising amount of edge latency hides in resize, normalize, and tokenize), handle the warm-up so the first user request is not the cold-start outlier, and build the fallback path for ops the accelerator does not support. The full walkthrough is [mobile deployment end to end](/blog/machine-learning/edge-ai/mobile-deployment-end-to-end). The one rule people forget: **warm up at app start**, not on the user's first tap, or your p99 includes the cold-start cost you carefully excluded from your benchmark.

**Operate.** A shipped edge model is not done — it is *live*, on hardware you do not control, against an input distribution that drifts. Edge MLOps is the discipline of keeping it honest in the field: on-device telemetry (latency, fallback rate, battery impact), drift detection on the input distribution, A/B and canary rollout, and a rollback path for when v2 regresses on someone's three-year-old phone with a quirky NPU driver. The deep-dive is [edge MLOps](/blog/machine-learning/edge-ai/edge-mlops). The mindset shift: the lab number is a hypothesis; the field telemetry is the truth, and they will disagree.

The disagreement is usually largest on two axes, and knowing which lets you instrument for it. **Latency telemetry catches the op-fallback tail you cannot see on the bench.** Your test device supported every op on the NPU; a user's older chip silently falls back to CPU for one layer, and their p99 is 4× yours. You only learn this from a histogram of real latencies bucketed by device model — which is why fallback rate is a first-class metric, not an afterthought. **Accuracy telemetry catches drift.** The input distribution at ship time was your validation set; six months later it is a new camera sensor, a new lighting condition, a new slang the keyword spotter never trained on. You cannot measure accuracy directly in the field (you have no labels), so you proxy it — confidence-score distributions, abstention rates, user-correction rates — and alarm when the proxy moves. This is also where the **double-counting** anti-pattern finally gets caught: the lab said the stacked levers gave 6×, but the field histogram says 4.5×, and the gap is real-world inputs hitting the slow fallback path your clean benchmark never exercised. The honest closing of the loop is to measure the *full system* in the field and let that number, not the spreadsheet, be what you report up the chain. The lab earns you the right to ship; the field tells you whether you should have. The full lifecycle view — from artifact to field to the next iteration — is in [the edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle).

**Pick the final config off the Pareto frontier.** Throughout this procedure you have generated not one model but a *family* — fp32, int8, int8+pruned, int4, QAT-int8 — each a point in the (accuracy, latency) plane. The right way to choose is not "the fastest that passes" or "the most accurate that fits," but to read the **Pareto frontier**: the set of configs where you cannot improve one axis without sacrificing the other. Every config *not* on the frontier is strictly dominated — something is both faster and more accurate — so throw it out. Among the frontier points, pick the one that satisfies your binding constraint with the most slack on the others. This turns "best" from a vibe into a definition, and it is the subject of [the accuracy–latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier). Then **validate** that chosen config one final time on the device, in the thermal state you ship in, against every entry in the step-0 budget. If it passes all five, you ship. If it misses one, you go back to the tree for that constraint and pull one more lever.

Concretely, the family for the phone segmentation target might look like the table below — each row a config you actually built and measured on the Pixel 8 NPU, batch one, after warm-up, in steady thermal state. The budget from step 0 was p99 ≤ 25 ms, size ≤ 25 MB, mIoU ≥ 0.72. Figures here are illustrative of the *shape* of such a sweep, not measurements from a specific shipped product, but they are the order of magnitude you should expect.

| Config | mIoU | p99 latency | Size | On frontier? | Verdict |
| --- | --- | --- | --- | --- | --- |
| fp32 baseline | 0.781 | ~90 ms | ~50 MB | yes (most accurate) | misses latency badly |
| efficient backbone (fp32) | 0.762 | ~38 ms | ~24 MB | yes | still misses latency |
| backbone + structured prune 30% | 0.749 | ~27 ms | ~17 MB | yes | just misses p99 |
| backbone + prune + int8 PTQ | 0.731 | ~19 ms | ~6 MB | yes | passes — but thin margin |
| backbone + prune + QAT int8 | 0.744 | ~19 ms | ~6 MB | **yes (the pick)** | passes all five with slack |
| backbone + prune + int4 (naive) | 0.681 | ~17 ms | ~3.5 MB | no (dominated) | below floor — discard |

Read it the way the procedure demands. The naive int4 row is *dominated* — the QAT-int8 row is both faster-enough and far more accurate — so it is thrown out immediately; the extra MB it saves buys nothing because size was never the binding constraint. Among the rows that *pass* the latency wall, QAT-int8 sits on the frontier with the most accuracy slack above the 0.72 floor, so it is the pick over PTQ-int8 despite identical latency — the few points of mIoU are free margin against field drift. That is the whole discipline: generate the family, drop the dominated points, and choose the frontier config with the most slack on the constraints that do not bind. The QAT pass paid for itself precisely because PTQ landed too close to the floor for comfort.

## 10. The master decision function

Here is the whole playbook as a single function: given a target, a budget, and a model type, it returns an ordered lever plan. It is deliberately explicit — this is the procedure you would run in your head, written down so you cannot skip a branch.

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Target:
    accelerator: str        # "npu" | "gpu" | "cpu" | "mcu"
    dtypes: set             # e.g. {"int8", "fp16"}; {"int8"} only on many NPUs
    has_sparse_cores: bool  # 2:4 structured sparsity support (NVIDIA)
    ram_mb: int

@dataclass
class Budget:
    p99_ms: float
    peak_mem_mb: float
    model_mb: float
    energy_mj: float
    acc_floor: float        # hard minimum on a representative eval

def plan(target: Target, budget: Budget, model_type: str,
         bottleneck: str, acc_headroom: float) -> List[str]:
    """Return the ordered lever plan. bottleneck in {'memory','compute'};
    acc_headroom = how far current accuracy sits above the floor (points)."""
    steps = ["measure baseline on device (p99, peak mem, size)"]
    steps.append(f"roofline: confirm bottleneck == {bottleneck!r}")

    # --- Lever order is FIXED: arch -> distill -> prune -> quant -> compile.
    # We only decide WHICH levers to include and how aggressive to be. ---

    # 1. Architecture / SLM-by-design (always the foundation)
    if model_type == "mcu":
        steps.append("ARCH: pick tiny-by-design model (fit the SRAM arena)")
    elif model_type == "llm":
        steps.append("ARCH: choose smallest base model that clears the floor")
    else:  # cnn / vision
        steps.append("ARCH: efficient blocks; if compute-bound, NAS for latency")

    # 2. Distillation (recover capability; lean on it when headroom is thin)
    if acc_headroom < 2.0 or model_type in ("mcu", "llm"):
        steps.append("DISTILL: teacher -> student to recover/transfer skill")

    # 3. Structured pruning (only when compute-bound; unstructured only w/ HW)
    if bottleneck == "compute":
        steps.append("PRUNE: structured channel/head pruning + fine-tune")
    elif target.has_sparse_cores:
        steps.append("PRUNE: 2:4 sparsity (hardware can skip the zeros)")
    # else: skip pruning -- it would not speed up a memory-bound op here

    # 4. Quantization (LAST; PTQ first, QAT only if it misses the floor)
    if acc_headroom < 1.0:
        steps.append("QUANT: QAT int8 + mixed precision on sensitive layers")
    elif "int4" in target.dtypes and model_type == "llm":
        steps.append("QUANT: weight-only int4 (GPTQ/AWQ) + int8 KV cache")
    else:
        steps.append("QUANT: PTQ int8 (calibrate on ~300 real samples)")

    # 5. Compile / runtime (re-lower AFTER quant; pick the owner of the HW)
    runtime = {"gpu": "TensorRT", "npu": "TFLite/LiteRT delegate",
               "cpu": "llama.cpp/ORT", "mcu": "TFLite-Micro + CMSIS-NN"}
    steps.append(f"COMPILE: {runtime[target.accelerator]}; re-lower, fuse")

    # 6. Memory + validate + choose off the Pareto frontier
    if model_type == "llm":
        steps.append("MEMORY: KV-cache quant + paging for long context")
    steps.append("VALIDATE: p99 + peak mem + acc on device, thermal soak")
    steps.append("CHOOSE: pick the Pareto-frontier config meeting the budget")
    return steps
```

Run it on the phone segmentation target (compute-bound, thin headroom, NPU int8 only) and it returns: efficient architecture → distill → structured prune → QAT int8 with mixed precision → TFLite delegate → validate → choose. Run it on the laptop LLM (memory-bound, int4-capable, CPU) and it returns: smallest base model → distill → *skip prune* → int4 GPTQ + int8 KV cache → `llama.cpp` → KV paging → validate → choose. Same function, two correct and *different* plans, because the bottleneck and model type routed it differently. That is the playbook executing.

## 11. The pre-flight checklist and the reference tables

Before you ship, run the checklist in Figure 8 top to bottom. Each gate must pass before the next; the whole point is that you cannot ship on an unmeasured assumption.

![A stack figure showing the seven ordered pre-flight gates from a measured baseline through telemetry that each must pass before shipping](/imgs/blogs/the-edge-optimization-playbook-8.png)

The pre-flight checklist, in words:

1. **Baseline measured** on the device — p50/p99, peak memory, model size — after warm-up, in steady thermal state.
2. **Budget defined and binding constraint identified** — p99 ms, peak MB, energy mJ, model MB, accuracy floor; mark which one binds.
3. **Levers pulled in order** — architecture → distill → prune → quantize → compile; no reordering.
4. **Each step validated** — measure accuracy *and* latency after every lever, so you can attribute wins and catch regressions early.
5. **Compiled and re-lowered** after the last lever, on the runtime that owns your hardware.
6. **Peak memory fits** with margin, and **warm-up happens at app start** so the first user request is not the cold outlier.
7. **Telemetry live** — latency, fallback rate, battery, drift — so the field can disagree with the lab and you will know.

And the two reference tables that sit at the center of this document. First, the **lever cheat sheet** (also Figure 6) — the series-wide one-glance summary of what each lever cuts, its accuracy risk, hardware dependence, retraining need, and pipeline slot:

![A matrix figure summarizing each optimization lever by what it cuts its accuracy risk hardware coupling retraining need and slot in the pipeline](/imgs/blogs/the-edge-optimization-playbook-6.png)

| Lever | What it cuts | Accuracy risk | HW-dependence | Needs retraining? | Pipeline slot |
| --- | --- | --- | --- | --- | --- |
| Architecture / SLM | FLOPs + params | redesign cost | low (portable) | yes (train) | 1st (foundation) |
| Distillation | params (via smaller student) | *recovers* accuracy | none | yes (train) | 2nd (recover) |
| Structured pruning | FLOPs + size | medium | low (portable) | yes (fine-tune) | 3rd (trim) |
| Unstructured pruning | size only (speed needs HW) | low–medium | high (sparse cores) | yes (fine-tune) | 3rd (special case) |
| Quantization — PTQ | size + bandwidth | small | high (low-precision units) | no | 4th (last) |
| Quantization — QAT | size + bandwidth | tiny | high | yes (fine-tune) | 4th (if PTQ misses) |
| Compiler / runtime | latency (scheduling) | none | high (per-target) | no | after every lever |

Second, the **constraint → recommended plan** table — read your binding constraint, get your ordered plan:

| Binding constraint | Default model class | Recommended plan (in order) |
| --- | --- | --- |
| Latency, compute-bound | CNN / vision | efficient arch → distill → structured prune → int8 PTQ → TensorRT/TFLite → validate |
| Latency/throughput, memory-bound | LLM decode | small base → distill → (skip prune) → int4 GPTQ/AWQ + KV-cache tricks → `llama.cpp`/MLC → validate |
| Peak memory / capacity | MCU / TinyML | tiny-by-design → aggressive distill → int8 everything → TFLite-Micro → fit the arena |
| Binary / model size | any | distill into smaller → structured prune → int8/int4 quant → compile |
| Accuracy floor tight | any | distill → mixed precision (sensitivity-guided) → QAT int8 → minimal pruning → compile |
| Energy per inference | battery devices | fewer FLOPs (arch + prune) → int8 → tune clock/voltage → measure mJ, not just ms |

## 12. Case studies — the two ends of the procedure

The series carries two full end-to-end case studies that exercise this entire procedure on opposite kinds of model; here they are in capsule, as proof the playbook composes.

**Real-time vision on a phone.** A segmentation/detection CNN going from an fp32 baseline well over budget to a shipped int8 model inside a 30-FPS frame budget on a mobile NPU. The procedure ran exactly as written: compute-bound diagnosis → MobileNet-style efficient backbone → distillation from a heavy teacher → structured channel pruning with fine-tuning → int8 (escalating to QAT where PTQ dropped below the mIoU floor) → TFLite delegate compile → validate p99 in steady thermal state. The compounding of architecture (the biggest lever) and quantization (the bandwidth bonus) is what got it under the frame budget; either alone would have missed. Full numbers in [case study: real-time vision on device](/blog/machine-learning/edge-ai/case-study-real-time-vision-on-device).

**An LLM assistant on a laptop.** A 7B model going from "barely fits, crawls" to "interactive chat" on consumer hardware, as worked in section 8: memory-bound diagnosis → smallest strong base → weight-only int4 → `Q4_K_M` GGUF under `llama.cpp` → int8 KV cache and paging for long context → Metal compile → validate tokens/s and resident memory. Here quantization did almost all the work, *because the bottleneck was memory* — and pruning was correctly skipped because it would not have helped a memory-bound decode loop. Full walkthrough in [case study: an LLM assistant on a laptop](/blog/machine-learning/edge-ai/case-study-an-llm-assistant-on-a-laptop).

The contrast is the lesson. Same procedure, same checklist, same Pareto discipline — but the *plan* the procedure produced was different for each, because step 0 and step 1 (know the target, find the bottleneck) routed them down different branches of the trees. The procedure is fixed; the plan it yields is bespoke to your constraint. That is exactly what a playbook should be.

## 13. When to reach for this — and when not to

A decision document owes you the honest negative space. Here is when the full procedure is *not* worth running.

- **If the unoptimized model already hits the budget, ship it.** The cheapest optimization is the one you do not do. Run step 0 and step 1; if the baseline passes, stop. Every lever you skip is accuracy and engineering time you keep.
- **If PTQ already clears the floor, do not do QAT.** QAT is a training run; PTQ is a calibration pass. Escalate only on a measured miss.
- **Do not chase unstructured sparsity for speed on commodity hardware.** It buys you nothing without sparse-aware kernels or 2:4 tensor cores. If you only need size, fine; if you need speed, use structured pruning.
- **Do not optimize a model that is not on the critical path.** Profile first. If pre-processing or I/O dominates the frame, the model is not your problem and compressing it is wasted motion.
- **Do not pull a lever you cannot validate.** If you cannot measure accuracy on a representative eval after the cut, you are flying blind; build the eval before you build the pipeline.
- **Do not ship a single config without the frontier.** If you only ever produced one model, you do not know whether it is dominated. Generate a few points and check the frontier; the right answer is often one cheap step away from where you stopped.

## 14. Key takeaways

1. **Measure before you optimize.** Baseline on the device, batch one, warm-up, steady thermal state, p99 — or every later number is fiction.
2. **The roofline decides your levers.** Memory-bound → quantize and shed bytes; compute-bound → shed FLOPs (architecture, structured pruning). One inequality, $I$ vs $\pi/\beta$, reorganizes the whole plan.
3. **Pull the four levers in order: architecture → distill → prune → quantize.** Coarse before fine, each step calibrates on the last, quantization last because it is cheapest and most hardware-coupled.
4. **Quantization last, never quantize-then-prune.** Pruning after calibration leaves stale scales; prune-then-quantize calibrates against the final distribution.
5. **Structured, not unstructured, for speed on commodity hardware.** A dense kernel multiplies by zeros at full speed; only whole-channel removal (or 2:4 with the right hardware) buys real time.
6. **Re-compile after every lever.** A good runtime is a zero-accuracy win, and only a re-lowered graph picks the int8 fused kernels you just earned.
7. **Memory is feasibility, not just speed.** Budget peak working set and, for LLMs, the KV cache — which grows linearly with context and often exceeds the quantized weights.
8. **Pick the config off the Pareto frontier, then validate on device.** Discard every dominated point; choose the frontier config that meets the binding constraint with the most slack; validate against all five budget entries.
9. **Beware double-counting.** Isolated lever wins do not multiply on paper the way they compose on the device; measure the full stack end to end, once.
10. **Shipping is not done — operate.** Telemetry, drift, canary, rollback. The lab number is a hypothesis; the field is the truth.

## Further reading

- The field map this playbook executes: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) and [the accuracy–latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier).
- The diagnosis tools: [the roofline model and where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device), and [profiling and benchmarking on device](/blog/machine-learning/edge-ai/profiling-and-benchmarking-on-device).
- The four levers, in order: [building blocks for efficient models](/blog/machine-learning/edge-ai/building-blocks-for-efficient-models), [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals), [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up), and [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles).
- The substrate and deploy/operate steps: [from model to deployable artifact](/blog/machine-learning/edge-ai/from-model-to-deployable-artifact), [inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared), [mobile deployment end to end](/blog/machine-learning/edge-ai/mobile-deployment-end-to-end), and [edge MLOps](/blog/machine-learning/edge-ai/edge-mlops).
- Seminal papers: Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015); Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018); Frantar et al., "GPTQ" (2022); Lin et al., "AWQ" (2023); Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model" (2009).
- Official docs: TensorFlow Lite / LiteRT, ONNX Runtime, NVIDIA TensorRT, and `llama.cpp` — the runtimes you will actually compile against.

This is the end of the series and the beginning of your procedure. The whole arc was this: there are four levers — efficient architecture, distillation, pruning, quantization — sitting on a substrate of compilers and runtimes, refereed by profiling, and read off the accuracy–latency Pareto frontier. The mindset that ties it together is four words: **measure, order the levers, push the frontier, ship.** Measure so you optimize what binds. Order the levers so the wins compound instead of cancel. Push the frontier so "better" is a definition, not a vibe. And ship — then operate, because the field always has the last word. Keep this page open the next time someone hands you a checkpoint and a device; the rest of the series is the deep dive on every step it names.

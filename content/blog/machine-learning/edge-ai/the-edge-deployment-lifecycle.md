---
title: "The edge deployment lifecycle: from a trained model to something that ships"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Walk the entire path from a trained checkpoint to an on-device artifact — train, optimize, convert, compile, package, run, monitor — with the failure modes, the real toolchain commands, and the before-after numbers that decide whether your model ever ships."
tags:
  [
    "edge-ai",
    "model-optimization",
    "deployment",
    "onnx",
    "tflite",
    "executorch",
    "llama-cpp",
    "inference",
    "efficient-ml",
    "mlops",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-edge-deployment-lifecycle-1.png"
---

I once watched a model with 99.1% validation accuracy never ship.

It was a defect classifier for a manufacturing line — a clean ResNet variant, beautifully tuned, with a confusion matrix the product team loved. The plan was to run it on a small ARM box bolted to the conveyor. Six weeks after the "model is done" celebration, it still wasn't on the line. The checkpoint was a `.pth` file in someone's home directory. Nobody could say how to turn it into something the device could load. The researcher said "that's an engineering problem." The mobile engineer said "send me a TFLite file." The platform team said "we don't have a GPU on that box, will it even run?" Each of them was right, and the model died in the gap between them — not because of accuracy, but because **nobody owned the path from checkpoint to on-device artifact.**

This post is that path, end to end. It is the map of the whole territory the rest of this series drills into. A trained model is not a product; it is the *input* to a deployment pipeline with seven distinct stages, each owned by someone, each with characteristic ways to go wrong, each with a cheap verification step that catches the failure before it reaches a device. Train, optimize, convert, compile, package, run, monitor. The figure below is the entire lifecycle on one timeline — keep it in your head as we walk each stage, because every later post in this series fixes a problem that lives in exactly one of these boxes.

![A seven-stage timeline showing a model traveling from train through optimize, convert, compile, package, run, and monitor, with each stage labeled by its owning team and characteristic failure mode](/imgs/blogs/the-edge-deployment-lifecycle-1.png)

By the end you will be able to take a trained PyTorch model, export it to ONNX and TFLite, quantize it, find and fix the op that silently broke during conversion, benchmark it on a named device, and ship it with a monitoring plan — and, just as importantly, know *which stage* a given symptom belongs to so you can stop debugging the wrong thing. We will frame every stage against the series' recurring spine: the **four levers** (quantization, pruning, distillation, efficient architecture) sit on **compilers and runtimes**, validated by **profiling**, read off the **accuracy-efficiency Pareto frontier**. The lifecycle is the assembly line that turns those levers into a shipped artifact.

## The seven stages, and where each one breaks

Before the code, the map. Each stage takes an artifact and hands a new one to the next stage. The dropped batons — the failures — almost always live *at the boundaries*, which is why this post is organized around the handoffs and not just the steps. Here is each stage with its one-line job, its most common failure, and which later post in the series fixes it.

**1. Train / pick architecture.** Owned by research. The job is to produce a checkpoint that is *accurate enough* and *shaped to fit the target budget*. The classic failure is that the model is trained without ever looking at the deployment target: a 90M-parameter backbone destined for a microcontroller with 512 KB of SRAM, or a transformer with an op the target NPU has never heard of. You cannot quantize your way out of a 50× over-budget architecture. The fix is **efficiency-aware architecture** — MobileNet/EfficientNet-style design, neural architecture search, or simply choosing a backbone whose ops the target accelerator supports. (Covered by the architecture and NAS posts in this series.)

**2. Optimize.** Owned by ML engineering. This is where the four levers live: quantize, prune, distill, and pick efficient ops. The failure is doing them in the wrong order or skipping the eval — quantizing before you've pruned, or shipping an int8 model nobody re-evaluated. We'll come back to the ordering. The deep dives are the quantization, pruning, and distillation posts, all framed by the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression).

**3. Convert / export.** Owned by ML engineering, hated by everyone. PyTorch → ONNX, `torch.export` → ExecuTorch, SavedModel → TFLite, or → Core ML. This is **the footgun stage**: unsupported ops, data-dependent control flow, dynamic shapes, custom layers, opset mismatches. More shipping attempts die here than anywhere else, and the errors are cryptic. Most of this post lives in this stage.

**4. Compile / graph-opt.** Owned by ML engineering or the runtime. Operator fusion, constant folding, layout transforms, kernel selection, and delegate assignment (NNAPI, Core ML, GPU, NPU). The failure is silent: an op falls off the fast path and lands on the CPU, and your "accelerated" model is 6× slower than the spreadsheet promised. The science section is built here.

**5. Package and deploy.** Owned by mobile/embedded engineering. Bundle the artifact into the app or firmware, or fetch it over the air (OTA). The failure is a size budget blown by 40 MB because nobody put a gate in CI, or an OTA model that ships before the matching app code.

**6. Run / serve.** Owned by mobile/embedded engineering. The on-device runtime loads the artifact, assigns delegates, handles cold-start, and falls back when an op isn't supported. The failure is the cold-start tax (first inference takes 800 ms while the delegate JIT-compiles) and thermal throttling (p99 doubles after 90 seconds of sustained inference).

**7. Monitor.** Owned by SRE/platform. On-device latency and accuracy telemetry, drift detection, A/B tests, and OTA model updates. The failure is having no telemetry at all — you ship, and the first you hear of a regression is a one-star review. This frames the edge-MLOps post.

Notice the ownership pattern: four different teams with four different incentives, and the failures cluster at the team boundaries. We'll return to that in the roles section, because *naming the owner of each handoff* is the single highest-leverage thing a team can do to make models actually ship.

## Stage 1 and 2: train for the target, then pull the four levers in order

The lifecycle starts upstream of any conversion tool, in a decision most teams make by accident: the architecture. If your target is a Cortex-M7 with 320 KB of SRAM, the relevant question is not "what's the most accurate model" but "what's the most accurate model whose *peak activation memory* fits in 320 KB and whose ops my runtime supports." That's a different optimization problem, and it's why MCUNet and the MobileNet family exist. Picking the architecture *is* an optimization lever — arguably the first one — and getting it wrong means no amount of downstream quantization saves you.

Once the architecture fits, the optimize stage pulls the four levers. The ordering matters, and the defensible default is:

1. **Distill first (if you're going to).** Train a smaller student from a larger teacher. This changes the architecture, so it has to happen before you commit to a specific graph.
2. **Prune second.** Remove weights or whole channels, then fine-tune to recover. Structured pruning (whole channels/heads) changes shapes; do it before quantization so the quantizer sees the final topology.
3. **Quantize third.** Map fp32 weights and activations to int8 (or int4). This is usually the last *training-time* lever because post-training quantization (PTQ) is cheap and quantization-aware training (QAT) wants a stable architecture to fine-tune.
4. **Compile/fuse last.** Done by the toolchain, not you — but it's a lever, and it's why we spend a whole section on fusion below.

The reason for "distill → prune → quantize" rather than the reverse is compounding: each lever's recovery fine-tuning assumes the architecture below it is fixed. If you quantize and *then* prune, you've thrown away the calibration you just paid for. The full ordering logic and the cost-benefit of each lever is the subject of the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression); here I just want to plant the flag that **optimize is a stage with an internal order**, and skipping the order is a self-inflicted accuracy loss.

For the rest of this post, our running example is a single concrete model so the stages stay grounded: **a MobileNetV2 image classifier**, fp32, ~3.5M parameters, ~14 MB on disk, trained on a 10-class subset and hitting 91.2% top-1. Our target is an Android phone. We'll carry it through convert, compile, quantize, package, run — and then do an LLM example at the end to show the same lifecycle holds for a 7B model headed to a laptop.

## Stage 3: convert — the footgun stage, in detail

Here is where models die. The conversion stage takes a framework checkpoint and produces an *exchange artifact* — ONNX, a TFLite flatbuffer, an ExecuTorch `.pte`, a Core ML `.mlpackage`. The promise is "same math, portable format." The reality is that the framework's eager execution did a hundred things the exchange format has no way to express, and the converter has to either map them, error, or — worst of all — silently approximate them.

There are four canonical footguns, and the figure below shows how a conversion either lands cleanly or veers into one of them.

![A branching dataflow graph showing a traced model entering an exporter and splitting into a clean export, a hard error on an unsupported op, a control-flow branch, and a dynamic-shape branch that both feed a slow CPU-fallback subgraph](/imgs/blogs/the-edge-deployment-lifecycle-2.png)

**Footgun 1: unsupported ops.** Your model uses an op the target format or runtime doesn't implement. `torch.nn.functional.grid_sample` at a low ONNX opset, a fused `scaled_dot_product_attention`, a deformable convolution, a custom CUDA kernel. The converter errors hard (good — at least it's loud) or maps it to a slow composite of primitives (bad — silent latency tax).

**Footgun 2: data-dependent control flow.** `if x.sum() > 0: ...` or a `while` loop whose iteration count depends on a tensor value. Tracing captures *one* path through the `if`; whatever branch your example input took is the only branch in the exported graph, and the other branch is gone. To preserve both you need a structured control-flow op (`torch.cond`, ONNX `If`/`Loop`), which most models don't use because nobody wrote them that way.

**Footgun 3: dynamic shapes.** You trace with a batch of 1 and a 224×224 image, and the exporter freezes those dimensions into the graph. Now the artifact only accepts exactly that shape. If you needed variable sequence length (NLP) or variable batch, you have to declare `dynamic_axes` explicitly — and many ops don't support dynamic shapes on the target accelerator, so declaring them can *cause* a CPU fallback.

**Footgun 4: opset / version mismatch.** ONNX has opsets; an op's semantics changed between opset 13 and 17. Export at opset 11, load in a runtime that expects 17, and you get either an error or a subtly different result. Same story for TFLite builtin versions and Core ML target OS versions.

The first three of these, when they don't error, frequently end in the worst outcome: a **CPU-fallback subgraph**. The converter succeeds, the runtime loads it, the model produces correct numbers — but one slice of the graph couldn't be assigned to the NPU, so the runtime quietly runs it on the CPU, inserting expensive tensor copies across the NPU/CPU boundary on either side. Your "18 ms on the NPU" model is now 110 ms because three ops in the middle bounced to the CPU. It *works*, so it passes a correctness test, and it only shows up if you profile per-op latency. This is the single most common "why is my edge model slow" bug, and it lives entirely in the convert/compile boundary.

There's a fifth footgun that deserves a callout because it's the most insidious: **a custom layer that decomposes into a slow primitive soup.** Suppose your model uses a custom attention variant or a fancy normalization that doesn't exist as a single op in the target format. The exporter doesn't error — it *decomposes* your one tidy op into fifteen primitive ops (reshapes, broadcasts, reductions, elementwise multiplies). Numerically it's correct, so the diff passes. But you've traded one fused kernel for fifteen tiny ones, each with its own kernel-launch overhead and its own memory round-trip, and several of them may not be delegate-supported. A custom op that ran in 2 ms in PyTorch can balloon to 30 ms of primitive soup after export. The tell is a per-op profile that shows a region of the graph full of tiny ops you never wrote. The fix is to register a *custom op* in the target runtime (TFLite custom ops, ONNX Runtime custom operators, ExecuTorch kernels) so your one kernel survives the trip intact — more work, but the only way to keep a non-standard op fast on-device.

The reason these footguns concentrate at conversion is structural: eager PyTorch is a *general-purpose program*, and the exchange formats are *restricted languages*. Anything PyTorch can do with arbitrary Python — branch on a tensor value, loop a data-dependent number of times, call into a hand-written kernel, allocate a dynamically-shaped buffer — has to be either expressed in the restricted language or given up. Conversion is the act of translating from a Turing-complete language into a (deliberately) less-expressive one, and the footguns are exactly the constructs that don't survive translation. Knowing that frame tells you *where to look* before you even run the exporter: scan your `forward()` for `if`/`for`/`while` on tensor values, for `.item()` calls, for custom autograd functions, and for any op you had to write yourself — those are your conversion risks, listed in advance.

### The actual convert: PyTorch → ONNX

Let's convert our MobileNetV2. Here's the real `torch.onnx.export` call with the flags that matter — the opset, dynamic axes for batch, and named inputs/outputs so the downstream tools and the diff harness can address tensors by name.

```python
import torch
import torchvision

model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2").eval()
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "mobilenetv2.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,                  # pin it; do not let the default drift
    dynamic_axes={                     # let batch vary; freeze spatial dims
        "input":  {0: "batch"},
        "logits": {0: "batch"},
    },
    do_constant_folding=True,          # fold constants at export time
)
print("exported")
```

Then — and this is the step everyone skips — **validate the graph and run it once** before trusting it:

```python
import onnx
import onnxruntime as ort
import numpy as np

# 1. Structural check: is the graph well-formed and opset-consistent?
onnx_model = onnx.load("mobilenetv2.onnx")
onnx.checker.check_model(onnx_model)        # raises if the graph is malformed

# 2. Numerical check: does ORT reproduce PyTorch on the same input?
sess = ort.InferenceSession("mobilenetv2.onnx",
                            providers=["CPUExecutionProvider"])
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

ort_out = sess.run(["logits"], {"input": x})[0]
with torch.no_grad():
    torch_out = model(torch.from_numpy(x)).numpy()

max_abs = np.max(np.abs(ort_out - torch_out))
print(f"max abs error: {max_abs:.2e}")     # want < 1e-4 for fp32
```

If `max_abs` is `3.2e-06`, you have a faithful export. If it's `2.1e-01`, something diverged, and we'll debug it in the divergence section below. The `onnx.checker` catches *structural* problems (a dangling node, an opset op that doesn't exist); the numerical diff catches *semantic* ones (an op that exists but does the wrong thing). You need both. Shipping without the numerical diff is how teams discover, in production, that their ONNX export silently used a different `align_corners` default in an interpolation op.

### Reading a real conversion error

Conversion errors are intimidating because they reference internal graph node names. Here's a representative one when an op isn't supported at your chosen opset:

```console
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator
'aten::scaled_dot_product_attention' to ONNX opset version 13 is not
supported. Support for this operator was added in version 14, try
exporting with this version.
```

The fix is in the message: bump `opset_version` to 14+. The skill is learning to *read* these — `aten::<op>` tells you the PyTorch op, "opset version N" tells you the floor. A nastier class is the silent one, where export succeeds but a custom layer got decomposed into primitives. You only catch that with the numerical diff, which is why the diff is not optional.

### Handling control flow on export

When your model genuinely needs a data-dependent branch — say it skips an expensive refinement head on low-confidence inputs — naive tracing bakes only the branch your example took. The portable fix is a *structured* conditional that the exporter records as an `If` node rather than flattening. In modern PyTorch that's `torch.cond`:

```python
import torch
from torch import nn

class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.cheap  = nn.Linear(128, 10)
        self.refine = nn.Linear(10, 10)

    def forward(self, x):
        logits = self.cheap(x)
        conf = logits.softmax(-1).max(-1).values.mean()
        # Both branches are captured; the runtime picks at inference.
        return torch.cond(
            conf > 0.9,
            lambda l: l,                       # confident: skip refinement
            lambda l: self.refine(l),          # unsure: run the extra head
            (logits,),
        )
```

Now `torch.export` records *both* branches and an `If` op that selects between them at runtime, so the exported graph behaves like the eager model on every input — not just the one you traced with. The cost is that `torch.cond` requires both branches to return the same shapes and dtypes, which is a real constraint but a small price for a faithful export. Most teams discover they don't actually need data-dependent control flow at inference at all — it was a training-time convenience — and the cleanest fix is to *remove the branch* before export. But when the branch is load-bearing, `torch.cond` is how it survives.

### The TFLite alternative

For Android, TFLite (now LiteRT) is often the more natural target because the on-device delegates (NNAPI, GPU, Hexagon) are first-class. The path from PyTorch usually goes PyTorch → ONNX → TensorFlow → TFLite, or you train in TF/Keras directly. Here is the TFLite-side conversion with int8 quantization baked in via a representative dataset — we'll dwell on the quantization in the next section, but note the shape of the call:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv2_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]      # enable quantization

def representative_dataset():
    # 100-500 real, in-distribution samples — NOT random noise
    for sample in calibration_samples[:200]:
        yield [sample.astype("float32")]   # shape [1,224,224,3]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8     # force full int8; error if an op can't
]
converter.inference_input_type  = tf.int8   # quantized I/O for the NPU
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
open("mobilenetv2_int8.tflite", "wb").write(tflite_int8)
```

The `supported_ops = [TFLITE_BUILTINS_INT8]` line is doing important work: it tells the converter to *fail loudly* if any op can't be expressed in full int8, rather than silently leaving that op in float and forcing a fallback. If you instead allow `TFLITE_BUILTINS` (float fallback), you get a model that converts but runs partly in float on the CPU — footgun 1 in disguise. Forcing full int8 turns a silent latency bug into a loud conversion error you can fix at your desk.

## Stage 3 (continued): quantize with ONNX Runtime static quantization

Quantization deserves its own series post for the *why* (and gets one — see the [int8/fp16/int4 trade-offs deep dive](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs)), but it's a *conversion-stage* concern in the lifecycle because you usually quantize at or just after export. Here's the ONNX Runtime static-quantization path, which needs a `CalibrationDataReader` to collect activation statistics:

```python
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat
)
import numpy as np

class MobileNetCalibrationReader(CalibrationDataReader):
    def __init__(self, samples, input_name="input"):
        self.input_name = input_name
        self.it = iter([{input_name: s[None].astype(np.float32)}
                        for s in samples])
    def get_next(self):
        return next(self.it, None)     # None signals end of calibration

reader = MobileNetCalibrationReader(calibration_samples[:200])

quantize_static(
    model_input="mobilenetv2.onnx",
    model_output="mobilenetv2_int8.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,        # QuantizeLinear/DequantizeLinear nodes
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,                    # per-channel weights = much less error
)
```

Two flags carry most of the accuracy: `per_channel=True` quantizes each output channel's weights with its own scale (far smaller error than one scale for the whole tensor), and `QuantFormat.QDQ` inserts explicit quantize/dequantize nodes the runtime can fuse and the compiler can reason about. The calibration set should be **100-500 real, in-distribution samples** — not random noise, which gives wildly wrong activation ranges and tanks accuracy. The most common quantization bug is a calibration set drawn from the wrong distribution (e.g., all from one class), producing clipping ranges that saturate on real inputs.

### A little of the science: why int8 costs accuracy, and how much

Quantization replaces a real value $x$ with $\hat{x} = s \cdot \mathrm{round}(x/s)$, where $s$ is the scale (step size). The rounding introduces an error $e = x - \hat{x}$ that, for a value uniformly distributed within a quantization bin of width $s$, is itself roughly uniform on $[-s/2,\, s/2]$. The variance of a uniform distribution on an interval of width $s$ is $s^2/12$, so the quantization noise power is

$$\sigma_e^2 = \frac{s^2}{12}.$$

For a $b$-bit quantizer covering a range of $2A$ (so $s = 2A / 2^b$), the signal-to-quantization-noise ratio works out to the familiar law

$$\mathrm{SQNR} \approx 6.02\,b + 1.76 \ \text{dB},$$

meaning **every bit you drop costs about 6 dB of signal-to-noise**. Going from fp32 (effectively lossless) to int8 leaves you 8 bits, ~50 dB of headroom, which for most well-behaved layers is plenty — that's why int8 PTQ usually costs well under a point of accuracy. Drop to int4 and you have ~26 dB, and now the noise is large enough that the cheap PTQ recipe often falls apart and you need QAT or smarter schemes. This is the quantitative reason the int8-on-an-NPU recipe is the workhorse of edge deployment and int4 is a research-grade effort. The full derivation, SQNR plots, and the int4/int3 frontier are in the quantization post; here it earns its place because it tells you *which conversion targets are safe by default*.

## Stage 4: compile — fusion, folding, and the kernel that never launched

Once you have an exchange artifact, a compiler lowers it toward the hardware. This is the most mathematically interesting stage and the most operationally treacherous, because everything it does is invisible until you profile. Let's build the science first, then the failure mode.

### Graph representations: eager, traced, exported

The thing being compiled is a *computation graph*. There are three flavors, and knowing which one you have explains most conversion behavior. The figure below stacks them as descending intermediate representations — eager Python at the top, silicon at the bottom — and the key insight is that each layer **narrows what the layer below it is allowed to assume**.

![A vertical stack of intermediate representations descending from eager Python through a traced exported graph and a hardware-neutral IR down to device kernels and finally silicon](/imgs/blogs/the-edge-deployment-lifecycle-4.png)

**Eager** is what you train in: `nn.Module.forward()` runs Python line by line, allocating tensors as it goes. There is no graph — each op executes immediately. Breakpoints work, `print(x.shape)` works, control flow is real Python. This is wonderful for research and useless for deployment, because there's nothing to optimize ahead of time.

**Traced / exported** is the deployment artifact. `torch.export` (or `torch.jit.trace`, or the ONNX export under the hood) runs the model once with an example input and *records* the sequence of ops into a directed acyclic graph (DAG) — nodes are ops, edges are tensors, with shapes and dtypes attached. This is the handoff artifact. Its limitation is exactly footgun 2: it records the *one path* the example took, so data-dependent branches are baked.

**Hardware-neutral IR** is the compiler's working representation: ONNX, StableHLO, TVM Relay, the ExecuTorch dialect. It's a DAG of standardized ops with no assumption about the target. This is where fusion and constant folding happen, as *semantics-preserving rewrites* — graph transformations that produce a different DAG computing the identical function.

From the IR, the compiler lowers to **device kernels** (layout transforms, tiling, picking an int8 NPU kernel vs an fp16 GPU kernel vs a CPU kernel per op) and finally to **silicon**, where p99 latency actually lives.

### What fusion and constant folding formally do

A graph optimization is a function $T$ on DAGs that must satisfy $f_{T(G)}(x) = f_{G}(x)$ for all valid inputs $x$ — it changes the graph, never the mathematical result. Two of them carry most of the win:

**Constant folding** evaluates any subgraph whose inputs are all constants *at compile time* and replaces it with the resulting constant tensor. If your graph computes `weight * scale` where both are constants known at build time, the runtime should never compute that product per inference — fold it once, ship the product. This removes nodes from the runtime DAG entirely.

**Operator fusion** merges adjacent ops into a single kernel so intermediate tensors never touch main memory. This is the big one for edge, and it's worth doing the arithmetic.

Three more rewrites round out the compiler's toolkit, each also a semantics-preserving $T$:

- **Dead-code elimination.** The traced graph often contains ops whose outputs nothing consumes — a debug branch, an auxiliary loss head used only in training, a dropout that's a no-op at inference. The compiler prunes any node not on a path to a graph output. Dropout is the canonical example: it's identity at inference, so it should simply vanish from the deployed graph.
- **Layout transformation.** PyTorch tensors are NCHW (channels-first); many mobile accelerators want NHWC (channels-last) because it matches their memory access pattern. The compiler inserts layout-conversion ops and then tries to *cancel adjacent ones* — a transpose immediately followed by its inverse is dead code. The danger is when it *can't* cancel them: a stray reshape between two convs can force a layout round-trip on every inference, and that copy shows up as mysterious latency with no obvious op to blame.
- **Kernel selection.** For a given op, shape, and dtype, there may be several implementations — a Winograd conv, a direct conv, an im2col-plus-GEMM conv — with different speeds at different sizes. The compiler (or the runtime's first-run autotuning) picks one. This is part of why cold-start is slow: the runtime is sometimes *measuring* kernels to choose the fastest.

All of these are invisible by design. They make the deployed graph diverge from the eager graph's *structure* while preserving its *function* — which is exactly why the numerical diff must compare the runtime output to the framework output, not to your mental model of the graph.

### Worked derivation: fuse conv + bn + relu

Consider the most common block in a CNN: a convolution, followed by batch normalization, followed by ReLU. Unfused, that's three kernel launches, and — critically — three round-trips of the *activation tensor* through main memory (DRAM). The figure below contrasts the unfused and fused versions.

![A before and after comparison showing three separate kernels each doing a memory round-trip on the left, versus a single fused convolution with batchnorm folded into weights and relu applied in registers on the right](/imgs/blogs/the-edge-deployment-lifecycle-3.png)

In the unfused version:

1. **Conv2d** reads the input and weights, computes, and writes the activation tensor to DRAM.
2. **BatchNorm** reads that activation *back* from DRAM, applies $\hat{y} = \gamma \frac{y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$, and writes it again.
3. **ReLU** reads it *back again*, applies $\max(0, x)$, and writes it a third time.

The activation tensor — call it $N$ bytes — is written 3 times and read 2 times: roughly $5N$ bytes of memory traffic, plus 3 kernel launches.

Fusing does two things. First, **batchnorm folds into the convolution weights offline**: because BN at inference is an affine transform with fixed parameters, you can absorb it into the conv weights and bias once, at compile time, so it costs *zero* runtime ops. Concretely, the fused weight is

$$W' = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \qquad b' = (b - \mu)\frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta,$$

which is exactly constant folding applied to the BN parameters. Second, **ReLU applies in registers** right after the conv computes each output element, before anything is written to DRAM. So the fused kernel reads the input once and writes the output once: roughly $2N$ bytes of activation traffic and **1 kernel launch instead of 3**.

That's the quantification the figure claims: activation memory traffic drops from ~$5N$ to ~$2N$, about a $2.5\times$ reduction, and kernel launches drop $3\times$. Why does this make the model *faster* rather than just tidier? Because this block is **memory-bound**, not compute-bound. The conv does some FLOPs, but on a small edge accelerator the bottleneck is feeding the compute units from memory, not the multiply-adds. Cutting memory traffic by 2.5× on a memory-bound layer cuts its latency by roughly the same factor. This is precisely the [roofline model's](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) prediction: when arithmetic intensity (FLOPs per byte) is low, you live on the memory-bandwidth slope of the roofline, and the only way to go faster is to move fewer bytes. Fusion moves fewer bytes. The kernel-launch saving matters too: each launch has fixed overhead (tens of microseconds on some accelerators), and for a model with hundreds of small ops, collapsing $3\times$ the launches into $1\times$ can dominate the win on the smallest layers.

This is why a compiled-and-fused graph routinely runs 2-4× faster than the same graph run op-by-op, with *identical numerical output*. It's free speed — if the compiler can actually do the fusion, which brings us to the failure mode.

### The failure mode: the fusion that didn't fire and the op that fell back

Fusion is pattern-matching. The compiler has a library of patterns (conv+bn+relu, matmul+add+gelu, etc.) and rewrites matches. If your graph has the *ops* but in a shape the matcher doesn't recognize — a stray reshape between conv and bn, an unusual activation, a custom op in the middle — the fusion silently doesn't fire, and you pay full memory traffic. Worse, on a delegated runtime, if one op in a fusable chain isn't supported by the NPU delegate, the *whole chain* may fall back to the CPU, with tensor copies across the boundary. You profile, see a layer taking 40 ms that should take 4 ms, and discover a single unsupported op poisoned a whole region of the graph. The verification is per-op profiling and checking the delegate-assignment count, which we'll do in the run section.

## Stage 3-4 debugging: the "conversion broke my accuracy" loop

You converted, you quantized, and now the runtime gives different answers than PyTorch — or the same answers 6× slower. This is the most common crisis in edge deployment, and the worst way to handle it is to guess. There is a deterministic procedure, shown in the figure below: reproduce, diff, bisect, isolate, fix.

![A five-step timeline showing the divergence-debugging loop: reproduce with a fixed seed, diff outputs by max absolute error, bisect the graph at intermediate tensors, isolate the diverging op, then fix and re-diff](/imgs/blogs/the-edge-deployment-lifecycle-5.png)

**Step 1 — reproduce.** Fix the random seed, build one input tensor, and feed *the same bytes* to both PyTorch and the runtime. No randomness, no augmentation, no batching surprises. If you can't reproduce deterministically, you can't bisect.

**Step 2 — diff the final outputs.** Compute max absolute and max relative error between the two outputs. For fp32, anything above ~`1e-4` is broken; for int8, you compare against the *fake-quantized* PyTorch output (the model with quantize/dequantize stubs), not the fp32 model, or you'll chase quantization noise that isn't a bug. Here's the diff pattern:

```python
import numpy as np

def diff(a, b, name=""):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    max_abs = np.max(np.abs(a - b))
    denom = np.maximum(np.abs(b), 1e-8)
    max_rel = np.max(np.abs(a - b) / denom)
    print(f"{name:24s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    return max_abs

# 1e-5  -> faithful;  2e-1 -> something diverged, go bisect
diff(ort_out, torch_out, "final logits")
```

**Step 3 — bisect the graph.** This is the trick that turns hours into minutes. Re-export the model with *intermediate tensors* added to the output list so the runtime exposes them, then diff each intermediate against the PyTorch activation at the same point. Binary-search the DAG: if the output of layer 70 (of 140) matches but the final output doesn't, the bug is in the second half; subdivide again.

```python
# Re-export exposing intermediate activations as extra named outputs,
# then diff each one to localize where the two graphs diverge.
intermediates = ["block_3_out", "block_6_out", "block_9_out", "logits"]
sess = ort.InferenceSession("mobilenetv2_debug.onnx",
                            providers=["CPUExecutionProvider"])
ort_vals = sess.run(intermediates, {"input": x})

for name, ov, tv in zip(intermediates, ort_vals, torch_intermediates):
    if diff(ov, tv, name) > 1e-3:
        print(f"  --> divergence starts at or before {name}")
        break       # first big jump localizes the culprit region
```

**Step 4 — isolate the op.** Once you've narrowed to a region, the diff jumps from `1e-6` to `2e-1` across exactly one op. In practice the usual suspects are: a fused softmax with a different numerical-stability trick, a transpose/layout op that reordered NHWC vs NCHW, an interpolation op with a flipped `align_corners`, or a per-tensor quantization where you needed per-channel.

**Step 5 — fix and re-diff.** The fix is targeted to the op: bump the opset so the correct op semantics are used, switch to per-channel quantization for that layer, or *block the bad fusion* (most compilers let you disable a fusion pattern) and re-measure. Then re-run the full diff to confirm the error is back to `1e-5`. The loop is deterministic; you never guess.

This same loop catches the *latency* version of the bug. Instead of diffing values, you profile per-op latency in both the all-CPU configuration and the delegated configuration. The op whose latency *increases* when you turn on the NPU delegate is the one that fell back to the CPU and is paying the cross-boundary copy tax. Block its fallback (or replace it with a supported op) and re-profile.

## Stage 5: package and deploy — size budgets, asset vs OTA

Your artifact is verified and fast. Now it has to physically reach the device, and the constraint here is brutally simple: **size**. A 14 MB fp32 model that became a 4 MB int8 model still adds 4 MB to your app download, and app stores and users both punish bloat. The two delivery models — embed in the app/firmware, or fetch over the air — are shown in the figure below, along with what happens at load time.

![A tree diagram showing a compiled artifact split into an embedded-asset path and an over-the-air download path, then a runtime delegate-assignment branch fanning out to NPU, GPU, and CPU backends](/imgs/blogs/the-edge-deployment-lifecycle-7.png)

**Embedded asset.** The model ships inside the app bundle or device firmware. Pro: it works offline from day one, no network dependency, no version-skew between app and model. Con: it bloats the binary, and updating the model means shipping a whole new app build through store review (days) or a firmware OTA (risky on constrained devices). Best for: small models, safety-critical paths, anything that must work before first network.

**OTA download.** The app ships *without* the model and fetches it on first launch or in the background. Pro: you can update the model without an app release — fix a regression or ship a better model in hours, not weeks. Con: a cold first-launch experience (the app has no model until the download finishes), a network dependency, and the version-skew trap: if the model format or the pre/post-processing changes, the *new model must not be served to old app code*. You need a compatibility contract (a model version the app checks) and a CDN.

The single most important packaging discipline is a **CI size gate**. Add a build step that fails if the model artifact exceeds the budget — say 25 MB — so a careless re-export at fp32 can never sneak a 56 MB blob into the app. This one check, which takes ten minutes to write, prevents the most common and most embarrassing packaging failure.

```yaml
# .github/workflows/model-ci.yml  (excerpt)
- name: Enforce model size budget
  run: |
    SIZE=$(stat -c%s artifacts/mobilenetv2_int8.tflite)
    LIMIT=$((25 * 1024 * 1024))     # 25 MB hard ceiling
    echo "model size: $SIZE bytes (limit $LIMIT)"
    if [ "$SIZE" -gt "$LIMIT" ]; then
      echo "::error::model exceeds size budget"; exit 1
    fi
```

## Stage 6: run — delegates, fallback, and the cold-start tax

On the device, the runtime (ONNX Runtime, TFLite/LiteRT, ExecuTorch, Core ML) loads the artifact and does **delegate assignment**: for each op, it asks "which backend can run this fastest?" and walks a fallback chain — NPU first (fastest, int8), then GPU (fp16), then CPU (always works, slowest). That fallback chain is the lower half of the previous figure, and it's where the silent-CPU-fallback bug lives. An op the NPU can't do drops to the GPU; an op the GPU can't do drops to the CPU, dragging tensor copies with it.

Two run-time realities that the spreadsheet never shows:

**Cold start.** The first inference is often dramatically slower than steady state because the delegate is JIT-compiling kernels, allocating the tensor arena, or warming caches. I've seen first-inference latencies of 600-900 ms on a model with a 18 ms steady-state p50. If your use case fires one inference and exits (a share-sheet action, a camera shutter classifier), the cold-start *is* your latency, and you must either pre-warm the runtime at app launch or accept it. **Always run a warm-up inference and discard it before you measure.**

**Thermal throttling.** Sustained inference heats the SoC, and the OS down-clocks to protect the chip. Your p99 measured over 10 inferences looks great; measured over 10 minutes of continuous inference it can double. Benchmark under the *actual duty cycle* you'll ship.

Here's how to benchmark TFLite honestly, with the NNAPI/GPU delegate, warm-up, and percentiles, using the official `benchmark_model` tool:

```bash
# Official TFLite benchmark tool, on-device via adb, with the NNAPI delegate.
adb push mobilenetv2_int8.tflite /data/local/tmp/
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenetv2_int8.tflite \
  --use_nnapi=true \
  --num_threads=4 \
  --warmup_runs=20 \
  --num_runs=200 \
  --enable_op_profiling=true     # per-op timing -> catch CPU fallback
```

The `--enable_op_profiling=true` flag prints a per-op breakdown. If you see ops attributed to the CPU when you expected the NPU, that's your fallback. The ONNX Runtime equivalent uses `onnxruntime_perf_test` with the right execution provider (`-e nnapi`, `-e coreml`, `-e cuda`) and the same warm-up/percentile discipline. The rule across both: **20 warm-up runs you throw away, 200 measured, report p50 and p99, on a real device, under the real duty cycle.** Numbers from an x86 emulator or an unthrottled bench are fiction.

### The economics of a single fallback op

It's worth being precise about *why* one fallback op is so expensive, because the intuition "it's just one op on the CPU, how bad can it be" is exactly backwards. The cost isn't the CPU compute — it's the *boundary crossings*. When the delegate hits an op it can't run, it has to:

1. **Copy the input tensor off the accelerator** to a buffer the CPU can read (a DRAM round-trip, possibly with a layout change).
2. **Run the op on the CPU** (the only cheap part, often).
3. **Copy the result back onto the accelerator** for the next supported op.

If the unsupported op sits in the *middle* of an otherwise-delegated graph, you pay this round-trip *twice* — once to leave the accelerator and once to return — and you've also *broken the delegate's fusion window*, so the ops on either side of the fallback can no longer fuse across it. A single unsupported op can therefore cost far more than its own runtime: it forces two boundary copies and disables fusion in its neighborhood. This is why the verification is "count the delegate partitions," not just "is it fast enough" — a model split into five delegate partitions by four scattered fallback ops is paying eight boundary crossings, and the fix (replace or custom-implement those four ops) can be a larger win than any quantization. The decision, when you find a fallback, is a genuine trade-off: rewrite the model to avoid the op, write a custom delegate kernel for it, or accept the fallback if the op is cheap and the boundary copies are small. There's no universal answer — you measure the partition count and the per-op profile, and you decide with data.

### Stress-testing the run stage

What happens when the assumptions break? If the **NPU doesn't support int8 for your op**, the runtime falls to the GPU's fp16 path — correct, slower, and now you're carrying both an int8 and an fp16 copy of intermediate tensors, raising peak memory. If the **calibration set was tiny** (say 10 samples), the activation ranges are noisy, clipping is wrong on real inputs, and accuracy craters in a way the holdout eval catches but a 10-sample sanity check does not. If the model is **memory-bound, not compute-bound**, then quantizing weights to int8 helps (less weight traffic) but quantizing *compute* doesn't speed you up much, because you were never compute-limited — the win comes from fusion and from cutting *activation* traffic, which is the roofline insight again. And if you run **int4** to squeeze size further, the SQNR math (≈26 dB of headroom) predicts that naive PTQ will often break, and you'll need QAT or a smarter scheme — exactly the regime where the cheap recipe stops working and the deep-dive techniques in the rest of this series earn their keep.

## Stage 7: monitor — telemetry, drift, A/B, and OTA updates

The lifecycle doesn't end at "it runs." The model is now a fleet of inferences happening on devices you don't control, on inputs you've never seen, and *it will drift* — the camera firmware updates and shifts the color profile, users start photographing a new product category, the seasons change the lighting. Without telemetry, your first signal of a regression is a support ticket.

On-device monitoring is its own discipline (the edge-MLOps post in this series goes deep), but the lifecycle-level minimums are:

- **Latency telemetry:** sample p50/p99 inference latency from real devices, bucketed by device model, because a mid-range phone with no NPU is a different world from a flagship.
- **Confidence/score distribution:** log the distribution of output confidences. A sudden shift in the histogram (everything becoming low-confidence) is a strong drift signal even without labels.
- **A/B and staged rollout:** ship a new model to 1% of devices, compare its telemetry to the 99% baseline, and only widen the rollout if it holds. This is why OTA delivery is so valuable — it makes A/B and instant rollback possible.
- **A rollback path:** before you ship, know exactly how you'd revert to the previous model in under an hour. The teams that get paged are the ones who shipped with no rollback plan.

The mechanics of drift detection deserve a concrete shape, because "monitor for drift" is otherwise hand-wavy. The cheapest useful signal is a histogram of output confidences that you compare to the distribution you saw at validation time, flagging when the two diverge:

```python
import numpy as np

class ConfidenceDriftMonitor:
    """On-device-friendly drift signal: compare the live confidence
    histogram to the validation-time reference with a simple divergence."""
    def __init__(self, reference_hist, bins, alert_threshold=0.15):
        self.ref = reference_hist / reference_hist.sum()   # normalized
        self.bins = bins
        self.thr = alert_threshold

    def check(self, recent_confidences):
        live, _ = np.histogram(recent_confidences, bins=self.bins)
        live = live / max(live.sum(), 1)
        # symmetric KL-ish divergence; spikes when the shapes diverge
        m = 0.5 * (self.ref + live) + 1e-9
        jsd = 0.5*np.sum(self.ref*np.log((self.ref+1e-9)/m)) \
            + 0.5*np.sum(live*np.log((live+1e-9)/m))
        return jsd, jsd > self.thr        # (score, should_alert)
```

You don't need labels for this — a shift in the confidence histogram (everything collapsing toward low confidence, or piling up at one class) is a strong leading indicator that the input distribution moved out from under the model, and you can sample it cheaply on-device and report only the aggregate. When it fires, you sample a few raw inputs (privacy permitting) to confirm, then start the retrain.

OTA model updates close the loop: monitor → detect drift → train a fix → quantize → convert → re-verify → push to 1% → widen. Notice that "re-verify" runs the *entire* convert/compile/run pipeline again for every update. That's why the verification gates we built into each stage aren't one-time chores — they're the CI that makes continuous model deployment safe. The first time you ship a *second* model to the fleet, the value of having built reproducible, gated stages the first time becomes obvious: the update is a re-run of a trusted pipeline, not a fresh act of heroism.

## Two worked examples: a CNN to Android, an LLM to a laptop

Theory is cheap. Here are two end-to-end runs with concrete numbers on named targets.

#### Worked example: MobileNetV2 → int8 TFLite → Pixel 8

We carry our running model all the way to a phone. Starting point: fp32 MobileNetV2, 14.2 MB, 91.2% top-1 on the 10-class holdout.

We pruned lightly (10% of channels, fine-tuned), then quantized to full int8 via the TFLite `representative_dataset` path above, then benchmarked with `benchmark_model --use_nnapi=true` on a **Pixel 8 (Tensor G3, with its NPU)**, 20 warm-up runs discarded, 200 measured, batch=1.

| Metric | fp32 (naive export) | int8 + compiled (shipped) | Change |
| --- | --- | --- | --- |
| Model size | 14.2 MB | 3.6 MB | 3.9× smaller |
| Top-1 accuracy | 91.2% | 90.6% | −0.6 pts |
| p50 latency (NPU) | 31 ms (CPU, no NPU support for fp32 path) | 4.8 ms | 6.5× faster |
| p99 latency | 44 ms | 7.1 ms | 6.2× faster |
| Peak memory | ~58 MB | ~22 MB | 2.6× lower |
| Cold-start (1st inference) | 290 ms | 180 ms | warm-up matters |

The −0.6 point accuracy cost buys a 3.9× smaller download and a 6.5× latency win, and the int8 path is what unlocks the NPU at all (the fp32 graph couldn't be delegated and ran on the CPU). This is a *good* Pareto move: well under a point of accuracy for a near-order-of-magnitude latency and size win. These figures are representative of the MobileNet-on-Pixel literature (Google's own MobileNetV3 latency tables on Pixel devices show the same shape — a few-fold NPU speedup for sub-point accuracy cost); treat the exact milliseconds as illustrative of the *pattern*, since they depend on the specific build, thermal state, and Android version. The discipline is what transfers: the size gate caught a fp32 regression in CI, the per-op profile confirmed every op landed on the NPU (no fallback), and the holdout eval confirmed the −0.6 point cost before we shipped.

#### Worked example: a 7B LLM checkpoint → GGUF → laptop via llama.cpp

The same lifecycle holds for an LLM, just with different tools. Take a 7B-parameter model checkpoint (fp16, ~13 GB) and ship it to an **M2 MacBook Air (no discrete GPU, unified memory)**. The optimize+convert+package stages collapse into the `llama.cpp` toolchain: convert the HF checkpoint to GGUF, then quantize to a k-quant.

```bash
# 1. Convert the HuggingFace fp16 checkpoint to GGUF (fp16 container).
python convert_hf_to_gguf.py ./my-7b-model \
  --outfile my-7b-f16.gguf --outtype f16

# 2. Quantize to Q4_K_M (4-bit k-quant, the workhorse for laptops).
./llama-quantize my-7b-f16.gguf my-7b-Q4_K_M.gguf Q4_K_M

# 3. Run it, offloading layers to the Metal backend, and report tokens/s.
./llama-cli -m my-7b-Q4_K_M.gguf \
  -p "Explain operator fusion in one paragraph." \
  -n 256 -ngl 99            # -ngl: offload all layers to Metal (GPU)
```

The numbers, representative of community llama.cpp benchmarks on Apple Silicon:

| Metric | fp16 GGUF | Q4_K_M (shipped) | Change |
| --- | --- | --- | --- |
| File size | ~13 GB | ~4.1 GB | 3.2× smaller |
| Fits in 16 GB RAM? | barely / swaps | comfortably | the whole point |
| Decode speed (M2 Air, Metal) | ~12 tok/s | ~28 tok/s | ~2.3× faster |
| Quality (perplexity proxy) | baseline | ~+0.1-0.3 ppl | small, usually fine |

The lever here is 4-bit k-quantization, and the lifecycle stages map cleanly: `convert_hf_to_gguf.py` is the **convert** stage, `llama-quantize` is the **optimize** stage, the GGUF file is the **package** artifact, `llama-cli -ngl 99` with the Metal backend is the **run** stage with delegate assignment (Metal = the "NPU/GPU delegate" of this world), and you'd **monitor** tokens/s and output quality in production. The `Q4_K_M` choice is the Pareto point: small perplexity cost for fitting the model in 16 GB of unified memory and roughly doubling decode speed. (For the *why* behind GGUF quant types and what `_K_M` means, see [how quantization works and GGUF quant types decoded](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded).) The footguns are the same family: an unsupported architecture in `convert_hf_to_gguf.py` is footgun 1, and a too-aggressive quant that tanks quality is the LLM version of the −4-point accuracy regression.

## Case studies: the lifecycle in the wild

The two worked examples above are representative, but it helps to anchor against published, named results that show the same lifecycle shape — sub-point (or controlled) accuracy cost in exchange for a multiplicative size and latency win, gated by exactly the verification steps this post argues for.

**MobileNetV3 on Pixel, by Google.** The MobileNetV3 paper (Howard et al., 2019) reports latencies measured directly on Pixel phones, and its whole design loop is the lifecycle in miniature: architecture search constrained by *on-device latency* (not FLOPs), then int8 quantization for the NPU path. The headline is that platform-aware NAS plus quantization yields large latency reductions for small accuracy movement — the same Pareto move our MobileNetV2 example makes, but with the architecture stage (stage 1) doing much of the work. The lesson that transfers: the most leverage in the lifecycle is often *upstream*, in choosing an architecture whose ops the target accelerator runs natively, rather than trying to rescue a hostile architecture downstream.

**DistilBERT, by Hugging Face.** Sanh et al. (2019) distilled BERT into a student that is ~40% smaller and ~60% faster at inference while retaining ~97% of GLUE performance. This is the *distillation* lever (stage 2) in isolation, and it's a clean case study in why the optimize stage has an order: distillation changes the architecture, so it has to happen before you freeze the graph and quantize. A team that quantized BERT first and then tried to distill would have thrown away the calibration. The verification that made DistilBERT shippable is exactly the holdout eval — the 97%-of-GLUE number is the accuracy report that should accompany every optimized artifact.

**Whisper.cpp on a laptop.** The `whisper.cpp` project (a sibling of `llama.cpp`) ships OpenAI's Whisper speech model as a quantized GGUF that runs in real time on a CPU-only laptop. It is the audio analog of our LLM worked example: convert the checkpoint to GGUF, quantize to a k-quant, run with the Metal/CPU backend. The case study reinforces that the lifecycle is *modality-independent* — the convert→optimize→package→run→monitor spine is identical whether the model classifies images, generates tokens, or transcribes audio; only the tools and the quality metric change.

**MCUNet on a Cortex-M microcontroller.** Lin et al. (2020) co-designed a tiny network (TinyNAS) and a tiny runtime (TinyEngine) to run ImageNet-scale classification on a microcontroller with under 512 KB of SRAM. This is the lifecycle pushed to its limit, where the constraint is *peak activation memory* (stage 1's failure mode) and the runtime is a no-malloc, fixed-arena engine (an extreme version of stage 6). It's the proof that the same seven stages stretch all the way down to a \$5 microcontroller — the budgets just get brutal, and the verification (does peak activation memory fit the SRAM arena?) becomes a hard wall rather than a soft preference.

These are deliberately diverse — vision, language, audio, and microcontroller — to make the point that the lifecycle generalizes. Treat the exact percentages as the published headline figures (they depend on the benchmark and the hardware revision); what transfers is the *shape*: a single dominant lever, a controlled accuracy cost, a multiplicative efficiency win, and a verification step that made it safe to ship. The numbers and ordering also tie directly back to the four-lever frame in the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression).

## Roles, ownership, and the handoff failure modes

The technical stages are only half the lifecycle. The other half is *who owns each one*, because — to repeat the thesis — the dropped batons live at the team boundaries. The figure below maps the four teams onto the stages they own, the artifact they're supposed to hand off, and their classic failure.

![A matrix mapping four teams, research, ML engineering, mobile engineering, and SRE, to the lifecycle stages they own, the handoff artifact each produces, and the classic failure at each boundary](/imgs/blogs/the-edge-deployment-lifecycle-8.png)

**Research owns train and architecture.** Their handoff artifact is the checkpoint *plus the eval harness* — and "plus the eval harness" is the part that's usually missing. If ML engineering can't reproduce the accuracy number, they can't tell whether quantization hurt it. The classic research failure is handing over a model that's simply too big or uses an op the target can't run, because the target was never in the loop.

**ML engineering owns optimize, convert, and compile.** Their handoff artifact is the *verified* on-device artifact plus an accuracy report on the holdout set. The classic failure is converting and quantizing but never re-evaluating — shipping an int8 model that lost 4 points because no one ran the holdout eval. The verification gate (the matrix figure below) is their responsibility.

**Mobile/embedded engineering owns package, run, and integrate.** Their handoff is the app build plus the size/latency gate. The classic failure is the silent CPU fallback — they integrate the model, it works in the demo, and nobody profiled per-op, so the NPU fallback ships and the battery drains.

**SRE/platform owns monitor, A/B, and OTA rollout.** Their handoff is the telemetry dashboard plus a rollback plan. The classic failure is no drift alert and no rollback — the model degrades over weeks and there's no path to revert.

The fix for all of these is structural, not heroic: **name a single owner per stage, define the handoff artifact explicitly, and put a verification gate at each boundary.** Model CI ties it together — every checkpoint runs through automated convert → verify-numerics → quantize → eval-holdout → size-gate → benchmark, and the artifact that comes out the other end is reproducible (same checkpoint hash + same toolchain version = same artifact, byte for byte). When the artifact isn't reproducible, you can't bisect a regression, and the lifecycle silently rots.

Concretely, the model-CI pipeline is just the verification column of the cheat-sheet table wired into a build, and it's worth seeing as a single artifact-producing flow:

```yaml
# model-ci pipeline (conceptual): each stage gates the next
stages:
  - convert:        # PyTorch -> ONNX at pinned opset
      run: python export.py --opset 17 --out model.onnx
      verify: python -c "import onnx; onnx.checker.check_model('model.onnx')"
  - diff-fp32:      # ORT must match PyTorch on a fixed seed
      verify: python diff.py --tol 1e-4 --fail-over-tol
  - quantize:       # static int8 with the representative set
      run: python quantize.py --calib data/calib/ --per-channel
  - eval-int8:      # holdout accuracy must not drop > 1 point
      verify: python eval.py --model model_int8.onnx --max-drop 1.0
  - size-gate:      # hard ceiling on the shipped artifact
      verify: test $(stat -c%s model_int8.onnx) -lt $((25*1024*1024))
  - benchmark:      # on a real device, p99 under budget, no CPU fallback
      run: ./bench_on_device.sh --p99-budget-ms 20 --assert-no-fallback
artifact: model_int8.onnx        # immutable, hash-pinned, the handoff
```

Every `verify:` line is a *failure converted to a build break*. The pipeline's output is one immutable, hash-pinned artifact, and the hash is what every other team references — mobile-eng bundles *that hash*, SRE rolls back to *that hash*. Reproducibility here is not a nicety; it's what makes a regression bisectable. If two builds of "the same model" produce different bytes because the toolchain version drifted, you can no longer answer "did the model change or did the compiler change," and every debugging session starts from zero. Pin the toolchain versions (the exact PyTorch, ONNX, ONNX Runtime, and TFLite versions) in the CI image, and treat a toolchain bump like a code change — it goes through the same gates, because a new ORT version *will* occasionally change a fusion pattern or an op's numerics.

## The lifecycle verification table

Here is the stage-by-stage cheat sheet: every stage, its default tool, its characteristic failure, and the cheap automated check that catches that failure *before* the device sees it. The figure renders the core of this as a matrix; the table below expands it.

![A matrix with five lifecycle stages as rows and three columns showing the default tool, what goes wrong, and how to verify each stage before shipping](/imgs/blogs/the-edge-deployment-lifecycle-6.png)

| Stage | Default tool | What goes wrong | How to verify (the step teams skip) |
| --- | --- | --- | --- |
| Train | PyTorch / TF | Model too big or uses unsupported op | Check op support against target runtime; estimate peak activation memory vs SRAM |
| Optimize | `torch.ao.quantization`, ORT, TFLite | Levers applied in wrong order; no re-eval | Re-run holdout eval after every lever; track the Pareto point |
| Convert | `torch.onnx.export`, `torch.export`, TFLiteConverter | Unsupported op, opset mismatch, frozen dynamic shape | `onnx.checker` + numerical diff vs framework < 1e-4 |
| Compile | ORT graph-opt, TFLite delegates, ExecuTorch | Fusion didn't fire; op fell back to CPU | Per-op profile; assert delegate-assignment count, no surprise CPU ops |
| Quantize | `quantize_static`, `representative_dataset` | Accuracy drop > 1 point from bad calibration | Eval int8 on holdout; compare to fake-quant baseline, not fp32 |
| Package | App bundle / OTA fetch | Binary blows the size budget | CI size gate, fail the build over the ceiling (e.g. 25 MB) |
| Run | ORT / TFLite / Core ML runtime | Cold-start tax, thermal throttle | Warm-up + p50/p99 on a real phone, under real duty cycle |
| Monitor | On-device telemetry, OTA | No drift alert, no rollback path | Confidence-distribution monitor + staged rollout + tested rollback |

The "how to verify" column is the whole game. Every entry is cheap — minutes to write, seconds to run in CI — and each one converts a *silent production failure* into a *loud desk-time error*. The teams that ship reliably are not smarter; they just put these checks in CI and never removed them.

## Naive export vs optimized-and-compiled: the same model, both ways

To make the cost of skipping stages concrete, here is our MobileNetV2 shipped two ways — the naive "just export it" path versus the full optimized-and-compiled path — on the same Pixel 8.

| | Naive: `torch.onnx.export` → run | Optimized: int8 + fused + delegated |
| --- | --- | --- |
| Stages used | Train → Convert → Run | Train → Optimize → Convert → Compile → Package → Run → Monitor |
| Precision | fp32 | int8 (per-channel weights) |
| Fusion | none (op-by-op) | conv+bn+relu fused |
| Delegate | CPU only (fp32 can't use NPU) | NPU (NNAPI), verified no fallback |
| Size | 14.2 MB | 3.6 MB |
| p50 latency | 31 ms | 4.8 ms |
| p99 latency | 44 ms | 7.1 ms |
| Accuracy | 91.2% | 90.6% |
| Verified? | "it runs" | numerics-diffed, holdout-eval'd, size-gated, per-op profiled |

Same model, same phone. The difference between 31 ms and 4.8 ms is entirely the optimize + compile + run stages that the naive path skipped, and the −0.6 point accuracy cost is the only price. The naive path *works* — it produces correct answers — which is exactly why teams ship it and then wonder why the battery dies and the UI stutters. "It runs" is not "it ships."

## Anatomy of a shipped bug: tracing one symptom to its stage

To make the lifecycle map *useful* rather than decorative, here's how it turns a vague symptom into a precise diagnosis. The symptom: "the model is slower on the phone than the benchmark said, and battery drains during use." Three teams could each spend a week here; the lifecycle says exactly where to look and in what order.

First, *which stage owns latency?* Latency is set in **compile** (fusion, delegate assignment) and observed in **run**. It is not a convert-stage bug (convert is about correctness) nor an optimize-stage bug (that's accuracy and size). So you don't touch the quantization config — you profile. You run `benchmark_model --enable_op_profiling=true` on the actual phone, with warm-up, and read the per-op table. You find a region of small ops attributed to the CPU. That's a fallback — a **compile-stage** failure surfacing in **run**.

Second, *why did those ops fall back?* You diff the op list against the delegate's supported-ops list and find a normalization op the NPU doesn't support at int8. It split the graph into three delegate partitions, costing four boundary copies and breaking fusion around it. Now you have a decision (the trade-off from the run section): replace the op with a supported equivalent, write a custom kernel, or accept it. You replace it with a fusable BN, re-run the model-CI pipeline (because changing the graph means re-verifying numerics and re-evaluating accuracy — you never change one stage's output without re-gating the downstream stages), and the per-op profile now shows one delegate partition, zero CPU ops.

Third, *did the fix cost accuracy?* The eval-int8 gate in CI answers it automatically: holdout accuracy moved by less than the 1-point budget, so the fix ships. Total time: an afternoon, not a week, because the lifecycle told you the symptom was a compile/run bug and the gates told you the fix was safe. The teams that flail on this symptom are the ones who start by re-quantizing (wrong stage) or by blaming the model (wrong stage), because they don't have the map. The map *is* the leverage.

## Where teams skip steps and pay for it

Five recurring shortcuts, and the bill each one runs up:

1. **No numerical diff after conversion.** Skipped because "it loaded and gave a label." Bill: a silent accuracy regression discovered in production when an interpolation op used a different default. Cost to add the check: 10 lines of code.
2. **No per-op profile after compile.** Skipped because "the demo was fast enough." Bill: a single unsupported op poisons a delegate region, the model runs 6× slow, the battery complaint comes in a month later. Cost to add: one `--enable_op_profiling` flag.
3. **No holdout eval after quantize.** Skipped because "int8 usually only costs a little." Bill: a bad calibration set saturates on real inputs and drops 4 points; nobody notices until accuracy-sensitive users do. Cost to add: re-run the existing eval on the int8 model.
4. **No size gate in CI.** Skipped because "the model is small." Bill: someone re-exports at fp32, the app grows 40 MB, the store flags it, the release slips. Cost to add: 8 lines of YAML.
5. **No owner for the convert→package boundary.** Skipped because it's nobody's job. Bill: the model sits in a home directory for six weeks — the story this post opened with. Cost to fix: name a person.

Every one of these is cheap to prevent and expensive to discover late. The lifecycle isn't bureaucracy; it's the list of the cheap checks that keep the expensive failures from reaching a device.

## When to reach for the full lifecycle (and when not to)

The full seven-stage, gated, monitored pipeline is the right default for anything that *ships to devices you don't control at scale*. But it's a cost, and there are cases where less is right:

- **A one-off internal demo on your own laptop?** Skip the optimize and compile stages — just export and run. You control the device and the duty cycle; the latency tax doesn't matter.
- **A model behind a server API, not on-device?** This is a different lifecycle (the [TensorRT/ONNX serving path](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler)), where you have a GPU and can batch, so the edge constraints (SRAM, NPU op support, cold-start, thermal) mostly don't apply. Don't pay the edge tax for a server workload.
- **A research prototype you'll throw away?** The convert stage is wasted effort until you've decided to ship.

Conversely, the moment a model is destined for *phones, embedded boxes, or microcontrollers in the field*, every stage earns its keep, and skipping any of them is borrowing against a future incident. The decision rule: the more you don't control the device and the higher the inference volume, the more of the lifecycle you must run.

## Key takeaways

1. **A checkpoint is an input, not a product.** Shipping is a seven-stage pipeline — train, optimize, convert, compile, package, run, monitor — and a model with 99% accuracy never ships if nobody owns the path.
2. **The failures live at the boundaries.** Name a single owner per stage and define the handoff artifact explicitly; the dropped batons cluster precisely between research, ML-eng, mobile-eng, and SRE.
3. **Convert is the footgun stage.** Unsupported ops, data-dependent control flow, frozen dynamic shapes, and opset mismatches end either in a loud error (good) or a silent CPU-fallback subgraph (catastrophic). Always run `onnx.checker` *and* a numerical diff < 1e-4.
4. **Fusion is free speed because most edge layers are memory-bound.** Fusing conv+bn+relu cuts activation traffic from ~5N to ~2N bytes and 3 kernel launches to 1, with identical output — a 2-4× win that the roofline model predicts.
5. **Quantization's cost is governed by SQNR ≈ 6.02b + 1.76 dB.** Int8 leaves ~50 dB of headroom (usually under a point of accuracy); int4 leaves ~26 dB and needs real care. Calibrate on 100-500 real in-distribution samples, per-channel weights.
6. **Debug divergence by bisection, never by guessing.** Reproduce with a fixed seed, diff outputs, bisect the graph at intermediate tensors, isolate the one op whose error spikes, fix it, re-diff.
7. **Measure on a named device, honestly.** Discard 20 warm-up runs, report p50 and p99 over 200, under the real duty cycle. Cold-start and thermal throttling are real and absent from the spreadsheet.
8. **Put cheap checks in CI and never remove them.** A numerical diff, a per-op profile, a holdout eval, and a size gate convert silent production failures into loud desk-time errors.
9. **The same lifecycle holds from MobileNet to a 7B LLM** — only the tools change (TFLite/NNAPI vs llama.cpp/Metal/GGUF). Convert, optimize, package, run, monitor are universal.
10. **"It runs" is not "it ships."** The naive export works and is 6× too slow; the difference is entirely the optimize+compile+run stages teams skip.

## Further reading

- **ONNX & ONNX Runtime** — the exchange-format spec and the runtime's quantization tooling (`quantize_static`, `CalibrationDataReader`, execution providers). Start with the ONNX Runtime quantization docs and the ONNX operator/opset reference. See also our [ONNX deep dive on format, runtime, and serving](/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving).
- **TensorFlow Lite / LiteRT** — the `TFLiteConverter`, `representative_dataset` calibration, full-int8 conversion, and the `benchmark_model` tool with per-op profiling and the NNAPI/GPU delegates.
- **ExecuTorch** — PyTorch's `torch.export`-based on-device runtime; the canonical modern path for PyTorch → edge, with the exported-graph IR at its center.
- **llama.cpp** — the README and `convert_hf_to_gguf.py` / `llama-quantize` tooling, the GGUF format, and the k-quant types (Q4_K_M and friends) for running LLMs on laptops and phones.
- **Quantization and the noise model** — Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018) for the int8 PTQ/QAT foundation, and any DSP text for the SQNR ≈ 6.02b + 1.76 dB derivation.
- **Within this series** — [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame and ordering; [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for memory-bound vs compute-bound reasoning behind fusion; and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that composes every stage of this lifecycle into a single decision flow.

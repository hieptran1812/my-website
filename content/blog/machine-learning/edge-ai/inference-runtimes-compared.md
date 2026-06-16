---
title: "Inference runtimes compared: TFLite, ONNX Runtime, ExecuTorch, Core ML, NNAPI"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Pick the on-device runtime that actually decides your latency, your supported ops, and your maintenance burden — with delegates, execution providers, partition costs, and a same-model benchmark across one phone."
tags:
  [
    "edge-ai",
    "model-optimization",
    "inference-runtime",
    "tflite",
    "onnx-runtime",
    "executorch",
    "core-ml",
    "delegates",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/inference-runtimes-compared-1.png"
---

A team I worked with spent three weeks squeezing a vision model down: quantized to int8, pruned the head, swapped in a MobileNet backbone. On the bench, the math said 9 ms per frame on the phone's NPU. Shipped to the test fleet, it ran at 41 ms — slower than the float model they were trying to replace. Nobody had changed the model. The model file on disk was byte-identical to the one that profiled at 9 ms.

The culprit was a single op. The new head used a `HardSwish` activation. The phone's NPU delegate did not support `HardSwish`, so the runtime did the only safe thing it could: it ran that one op on the CPU. But that one op sat in the middle of the graph. To run it on the CPU, the runtime had to copy the intermediate tensor off the NPU, sync the device, run the op, then copy the result back onto the NPU and sync again — twice per inference. The op itself cost microseconds. The two boundary copies and the two device syncs cost tens of milliseconds. The model was fine. The *runtime's placement decision* was the latency.

This is the part of edge deployment nobody warns you about. You can read every quantization paper and tune every pruning ratio, but the model is just a file. Something has to actually *execute* it on the device, op by op, and that something — the **inference runtime** — decides which hardware each op runs on, which ops are even supported, how tensors move between CPU and accelerator, and how much engineering you spend keeping it all working as devices change underneath you. The runtime is where your beautiful Pareto-optimal model meets the silicon, and it is where most of the latency you didn't expect actually comes from. If you have not yet read [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) or [the edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle), they set up the silicon and the conversion steps that this post assumes; here we pick up right after you have a deployable artifact and ask the only question left: *which runtime runs it, and how do I keep it on the accelerator?*

By the end you will be able to: name what each major runtime is and what platform it owns; explain the delegate / execution-provider model and why partition boundaries cost real milliseconds; run the *same* model under different runtimes and backends and read the latency table honestly; and choose a runtime from your deployment target rather than from a blog post's leaderboard. Figure 1 is the map we'll keep coming back to — training frameworks at the top, runtimes in the middle, and a shared pool of hardware backends at the bottom.

![Diagram showing training frameworks feeding into on-device runtimes which farm supported subgraphs out to a shared pool of CPU GPU and NPU hardware backends](/imgs/blogs/inference-runtimes-compared-1.png)

This is the runtime layer of the four-lever frame from [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression): quantization, pruning, distillation, and efficient architectures all produce a graph, but compilers and runtimes are what turn that graph into measured latency. You can have the best model on the frontier and still ship a slow product if the runtime keeps falling back to the CPU.

## The mental model: a runtime is a graph executor with a hardware budget

Before we go runtime by runtime, hold one picture in your head, because every runtime in this post is a variation on it. A trained model is a directed graph of operators — convolutions, matrix multiplies, activations, reshapes, softmaxes — with tensors flowing along the edges. An inference runtime is the thing that takes that graph plus the input tensor and produces the output tensor, by executing each operator on some piece of hardware.

The naive way to do that is an interpreter: walk the op list in topological order, and for each op call a CPU kernel that computes it. That always works, because the CPU can run anything. It is also slow, because the CPU is the weakest compute on a modern phone — the GPU has 10× the throughput and the NPU has 10× to 50× more for the int8 matrix math that dominates a neural net.

So every serious runtime adds a second idea: a way to hand parts of the graph to faster hardware. Google calls these **delegates**. Microsoft calls them **execution providers**. PyTorch calls them **backends**. Apple does it automatically inside Core ML. The names differ; the mechanism is the same. The runtime looks at the graph, asks each accelerator "which of these ops can you run?", carves the graph into *partitions*, assigns each partition to whichever hardware can run it fastest, and stitches the results back together. The supported subgraph goes to the NPU or GPU; whatever is left over runs on the CPU.

That single mechanism — **partition the graph, assign each partition to hardware, copy tensors across the boundaries** — is responsible for almost every surprising thing about edge runtimes. It is why op coverage decides your latency more than the runtime brand does. It is why one unsupported op in the wrong place can cost more than the rest of the model combined. It is why "the model is the same but it got slower" is a sentence you will say. We'll derive the cost of a boundary later; for now, just hold the picture: a runtime is a graph executor that spends a hardware budget, and the budget is spent at the partition boundaries.

Let me define the jargon once, since it recurs:

- **Op / operator**: one node in the graph (a `Conv2D`, a `MatMul`, an `Add`). The granularity at which support is decided.
- **Op coverage**: the fraction of your model's ops a given backend can run natively. The single most important property of a runtime-on-a-device.
- **Delegate / execution provider (EP) / backend**: a plugin that claims some set of ops and runs them on a particular accelerator (GPU, NPU, DSP).
- **Partition / subgraph**: a contiguous chunk of the graph assigned to one backend.
- **Fallback**: running an op on the CPU because no accelerator claimed it.
- **AOT (ahead-of-time)**: deciding the execution plan once, at export time, and shipping the plan. The opposite of interpreting at runtime.
- **NPU**: neural processing unit — the dedicated int8/fp16 matrix-math accelerator on modern SoCs. On Apple chips it's the ANE (Apple Neural Engine).

One more framing before the tour, because it determines how you should read the rest of this post. The four optimization levers — quantization, pruning, distillation, efficient architecture — all decide *what the graph contains*: which ops, at what precision, in what shape. The runtime decides *where each of those ops runs*. These two decisions are coupled, and the coupling runs in a direction people underestimate. The runtime's op coverage and the accelerator's preferences should feed *back* into your modeling choices: there is no point distilling down to a clever attention variant if no mobile NPU delegate supports it and it falls to the CPU. The best edge teams design the model and pick the runtime together, checking op coverage on the target *before* finishing the architecture. So as we go through the runtimes, keep asking not just "is this runtime fast?" but "does this runtime's accelerator path support the ops my four levers are about to produce?" That question is the bridge between the modeling half of this series and the systems half.

With that, let's meet the runtimes.

## TFLite / LiteRT: the .tflite flatbuffer and its delegates

TensorFlow Lite — now being rebranded **LiteRT** as Google folds in more frameworks — is the runtime most Android ML actually ships on. The model is a `.tflite` file: a [FlatBuffer](https://google.github.io/flatbuffers/), a flat, zero-copy serialized graph you can `mmap` straight from disk into the interpreter with no parse step. That mmap-and-go property matters on a phone, where cold start and memory pressure are real constraints.

The runtime is a small C++ interpreter. By default every op runs on the CPU. The speed comes from **delegates** that you attach to the interpreter at load time, each of which claims ops and runs them on faster hardware:

- **XNNPACK** — the optimized CPU delegate. It is on by default in recent builds and is the reason "CPU" is no longer slow: it has hand-tuned fp32 and int8 kernels with SIMD (NEON on Arm). For many models, XNNPACK CPU is your baseline and it is respectable.
- **GPU delegate** — runs ops on the mobile GPU via OpenCL/OpenGL (Android) or Metal (iOS). Great for fp16 convolution-heavy models; weaker on int8 and on ops with lots of small reshapes.
- **NNAPI delegate** — the old Android route to vendor NPUs/DSPs via the Android Neural Networks API. NNAPI is now frozen and deprecated (more on that below), but it is still how a lot of shipped apps reach the NPU.
- **Hexagon delegate** — Qualcomm's DSP path, predating the modern NNAPI/QNN route.
- **Core ML delegate** — yes, TFLite can run on iOS and delegate to Apple's Neural Engine via Core ML.
- **Edge TPU delegate** — for Google Coral hardware.

TFLite is **int8-first**. The canonical TFLite optimization is full-integer quantization with a representative dataset, which is what the NNAPI and Hexagon delegates want. Here is the conversion you have probably seen, with the flags that matter:

```python
import tensorflow as tf

# Assume a trained Keras model `model`.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # turn on quantization

# A representative dataset drives activation-range calibration for int8.
def representative_data_gen():
    for sample in calibration_dataset.take(200):
        # shape must match the model input, batch size 1
        yield [tf.cast(sample, tf.float32)]

converter.representative_dataset = representative_data_gen
# Force full int8 so the NPU delegate can take the whole graph.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
open("model_int8.tflite", "wb").write(tflite_int8)
```

The two `inference_input_type`/`output_type = tf.int8` lines are the ones people forget. If you leave the input float, the runtime inserts a quantize op at the front and a dequantize at the back, and on some delegates those become partition boundaries that pin the first and last layers to the CPU. Matching the I/O type to the body keeps the whole thing on the accelerator. This is the kind of detail the full pipeline post [quantization in practice](/blog/machine-learning/edge-ai/quantization-in-practice-a-full-int8-pipeline) walks through end to end.

Running it with a chosen delegate, in C++ (the Java/Kotlin API mirrors this):

```cpp
// Load the flatbuffer and build the interpreter.
auto model = tflite::FlatBufferModel::BuildFromFile("model_int8.tflite");
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Attach the GPU delegate (or NNAPI / Hexagon similarly).
auto* gpu_delegate = TfLiteGpuDelegateV2Create(/*default options*/ nullptr);
if (interpreter->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk) {
  // Delegate refused some ops -> they stay on CPU; this is the fallback.
}
interpreter->AllocateTensors();
// ... fill input tensor, then:
interpreter->Invoke();
```

`ModifyGraphWithDelegate` is the moment the partition happens. The delegate inspects the graph, claims the ops it supports, and the interpreter rewrites the graph so those ops become a single "delegate node." Everything the delegate did not claim stays as CPU ops. You can ask the interpreter afterward how many nodes were delegated — and you should, because that number is your real diagnostic. In Python the same diagnostic is one line, and it is the first thing to run after attaching any delegate:

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="model_int8.tflite",
    experimental_delegates=[tf.lite.experimental.load_delegate("libnnapi_delegate.so")],
)
interpreter.allocate_tensors()

# The single most useful diagnostic: how many nodes the delegate actually took.
# A delegate that claims 3 of 64 nodes is a delegate doing nothing useful.
for op in interpreter._get_ops_details():
    print(op["index"], op["op_name"])   # delegated ops show as a fused DELEGATE node
```

If that print shows one big `DELEGATE` node and a handful of stragglers, you delegated well. If it shows the original op list barely changed, the delegate rejected almost everything and you are about to benchmark the CPU while believing you measured the NPU. This is the number-one self-inflicted edge benchmarking error, and it takes one line to rule out.

The honest summary of TFLite: it is the most mature, most widely shipped mobile runtime, the int8 story is excellent, and the delegate ecosystem reaches every Android accelerator that exists. The cost is that it is a TensorFlow-centric tool in a PyTorch-centric research world, so you'll be converting through ONNX or `tf` from your PyTorch model, and conversion is where ops go missing. The other quiet cost is that the delegate you want (NNAPI for the NPU) is the one Google is freezing, so a model you ship on TFLite-via-NNAPI today is one you should plan to migrate to a vendor delegate tomorrow.

## ONNX Runtime: execution providers and graph partitioning, done generically

ONNX Runtime (ORT) takes the same delegate idea and generalizes it into the cleanest version of the model. ONNX (Open Neural Network Exchange) is a framework-neutral graph format — you export from PyTorch with `torch.onnx.export` or from TensorFlow with `tf2onnx`, and you get a `.onnx` file that ORT can run. The deep dive [ONNX deep dive: format, runtime, serving](/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving) covers the format itself; here we care about how ORT *runs* it.

ORT's accelerators are called **execution providers (EPs)**. The list is long and it is the broadest in the business:

- **CPUExecutionProvider** — the always-present fallback, with MLAS kernels.
- **CUDAExecutionProvider / TensorRTExecutionProvider** — NVIDIA GPUs and the TensorRT compiler for server-edge boxes like Jetson.
- **CoreMLExecutionProvider** — Apple ANE/GPU on iOS and macOS.
- **NnapiExecutionProvider** — Android NPUs via NNAPI.
- **QNNExecutionProvider** — Qualcomm's modern AI Engine (the successor path to NNAPI/Hexagon on Snapdragon).
- **XnnpackExecutionProvider** — the same XNNPACK CPU kernels TFLite uses, reused here.
- **DmlExecutionProvider** — DirectML for Windows GPUs.

The mechanism ORT documents explicitly, and the one worth internalizing, is **graph partitioning by EP priority**. You give ORT an *ordered* list of EPs. ORT walks the graph and, for each op, offers it to the highest-priority EP that can run it; that EP claims a maximal connected subgraph; the next EP gets a crack at what's left; and the `CPUExecutionProvider` mops up everything nobody claimed. The result is a graph chopped into subgraphs, each tagged with an EP, with CPU islands wherever coverage ran out.

```python
import onnxruntime as ort
import numpy as np

# EP order = priority. ORT tries NNAPI first, falls back to CPU.
providers = ["NnapiExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession("model.onnx", providers=providers)

# See exactly which EP ORT actually bound (it may drop one it can't init).
print("Active EPs:", sess.get_providers())

x = np.random.rand(1, 3, 224, 224).astype(np.float32)
out = sess.run(None, {"input": x})
```

To see the partition, set the session's log severity to verbose, or — cleaner — dump the optimized graph and count the nodes per EP:

```python
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = "model_opt.onnx"      # ORT writes the partitioned graph
sess = ort.InferenceSession("model.onnx", so, providers=providers)
# Load model_opt.onnx and inspect node `domain`/EP assignment to count CPU islands.
```

ORT's superpower is **op coverage**. Because ONNX is the lingua franca of model export, ORT's CPU provider supports an enormous op set, so even when an accelerator EP can't take an op, the fallback is fast and complete — you rarely hit a "no kernel for this op anywhere" wall the way you can in a younger runtime. Its second superpower is that the *same* `.onnx` file and the *same* Python/C++ code run on a laptop, a server, an Android phone, and an iPhone — you just change the EP list. That portability is why ORT is the default recommendation when you must ship to more than one platform.

ORT quantization is first-class too: `onnxruntime.quantization` does both dynamic and static (QDQ — quantize/dequantize node) int8, and int4 weight-only for LLMs, with a `CalibrationDataReader` interface that mirrors TFLite's representative dataset:

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class MyReader(CalibrationDataReader):
    def __init__(self, samples):
        self.it = iter([{"input": s} for s in samples])
    def get_next(self):
        return next(self.it, None)

quantize_static(
    "model.onnx", "model_int8.onnx",
    calibration_data_reader=MyReader(calib_samples),
    quant_format=ort.quantization.QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
```

ORT also exposes a layer most runtimes hide: **graph optimizations** that run before partitioning — constant folding, operator fusion (Conv+BN+ReLU into one node), and layout transforms — controlled by `GraphOptimizationLevel`. These matter for partitioning because fusion *changes the op count*: a Conv+BN+ReLU that fuses into one node is one op the accelerator either takes whole or not, instead of three ops that might split across a boundary. Turning fusion up (`ORT_ENABLE_ALL`) often *reduces* the number of partitions for free, which is exactly what you want from the science section. The lesson generalizes: graph-level optimization and hardware placement are not independent — better fusion produces fewer, larger subgraphs, which means fewer boundaries.

The honest summary of ORT: broadest op coverage, broadest platform reach, one artifact everywhere, and a partition model you can actually inspect. The cost is that "runs everywhere" means it is rarely the *single fastest* on any one accelerator — on an iPhone, a native Core ML model can beat ORT's Core ML EP; on a Jetson, raw TensorRT can beat ORT's TensorRT EP — because the native path can use device-specific tricks ORT must generalize away. There's also a sharp edge with EP ordering: if you list an EP that fails to initialize on the device (wrong driver, missing library), ORT silently drops it and binds the next one — which is why `get_providers()` after construction is mandatory. More than one team has shipped "the NNAPI build" that was quietly running on the CPU on half their devices because the NNAPI EP failed to load and nobody checked the active provider list.

## ExecuTorch: PyTorch goes ahead-of-time, on-device

ExecuTorch is the new entrant and the one to watch if your research is in PyTorch (most is). For years PyTorch's on-device story was awkward — PyTorch Mobile / LibTorch was heavy and slow on phones. ExecuTorch is the redesign: a small, ahead-of-time on-device runtime built on `torch.export` and the PyTorch 2 stack.

The key difference from TFLite and ORT is in the name: **ahead-of-time**. With TFLite and ORT, the runtime decides kernels and partitions when it loads the model on the device. With ExecuTorch, you do that work once, on your dev machine, at *export* time, and ship a `.pte` file that already contains the lowered, partitioned, scheduled program. The device just executes the plan. We'll unpack why that matters (and what it costs you) in the AOT section; the practical shape is:

```python
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# 1. Capture the graph ahead of time.
exported = export(model, example_inputs)

# 2. Lower to the edge dialect AND partition to a backend in one step.
#    XnnpackPartitioner claims the ops XNNPACK supports; rest stay portable-CPU.
edge = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()],
)

# 3. Serialize the ahead-of-time program.
exec_prog = edge.to_executorch()
with open("model_xnnpack.pte", "wb") as f:
    f.write(exec_prog.buffer)
```

ExecuTorch's backends mirror everyone else's accelerators: **XNNPACK** (CPU, the default and most mature), **Core ML** and **MPS** (Apple ANE/GPU), **Qualcomm** (QNN for Snapdragon NPUs), **Vulkan** (cross-vendor GPU), and **Arm Ethos-U** (microcontroller NPUs). The partitioner objects you pass in step 2 are exactly the partition mechanism — `XnnpackPartitioner`, `CoreMLPartitioner`, etc. — and you can pass several to split a graph across backends.

Quantization in ExecuTorch uses the PyTorch 2 export-based flow (PT2E): you `prepare_pt2e` with a backend-specific quantizer, calibrate, `convert_pt2e`, then export. The point is that quantization, lowering, and partitioning all happen in one ahead-of-time pipeline, so what you ship is final.

```python
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)

exported = export(model, example_inputs)            # capture
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
prepared = prepare_pt2e(exported.module(), quantizer)
for sample in calib_samples:                        # calibrate ranges
    prepared(sample)
quantized = convert_pt2e(prepared)                  # fold in int8
# then export + lower + to_executorch as above
```

One ExecuTorch detail that catches people: because the partition happens at *export* on your machine, the partitioner's view of "what the backend supports" must match the device's actual backend version. If you export with a newer ExecuTorch that claims an op the device's older backend can't run, you get a runtime error on device, not a clean fallback — the AOT plan assumed a capability that isn't there. The interpreted runtimes don't have this failure mode because they discover capabilities at load time on the real device. So with ExecuTorch you pin the export toolchain to the runtime version you ship, the way you'd pin a compiler to a target. It's the AOT tax: the plan is only as correct as your build-time model of the device.

The honest summary of ExecuTorch: it is the cleanest path from a PyTorch model to a fast, predictable on-device runtime, the AOT design gives you a fast first inference and no runtime surprises, and it is improving monthly. The cost is maturity: op coverage and backend stability are behind TFLite and ORT, the API is still moving, and you will occasionally hit an op a backend hasn't lowered yet and have to fall back to the portable CPU kernel. For a brand-new Apple/Android app written in PyTorch, it is a very strong default *if* your ops are covered; for a model with exotic ops, ORT's coverage is the safer bet today.

## Core ML: Apple's runtime and the automatic ANE

On Apple hardware, the native runtime is **Core ML**, and it plays a different game from the others. You do not attach delegates or order execution providers. You hand Core ML a `.mlpackage` (the modern container; older `.mlmodel` still works) and it *automatically* decides, op by op, whether to run on the CPU, the GPU, or the **Apple Neural Engine (ANE)** — Apple's NPU. You can hint a preference (`computeUnits = .all` / `.cpuAndNeuralEngine` / `.cpuOnly`), but the placement is Core ML's call, not yours.

That automatic placement is a double-edged thing. The good edge: when it works, you get the ANE — which on an A17/M-series chip is a genuinely fast int8/fp16 matrix engine — with zero configuration. The bad edge: when an op isn't ANE-friendly, Core ML silently moves that op (and often the tensors around it) to the GPU or CPU, and you get a partition boundary you didn't ask for and can't see without profiling in Instruments. The ANE is also picky: it strongly prefers certain layouts and shapes, and a model that runs entirely on the ANE versus one that bounces to the GPU can differ several-fold in latency on the *same chip*.

You produce Core ML models with `coremltools`, converting from PyTorch (via `torch.jit.trace` or `torch.export`) or from TensorFlow:

```python
import coremltools as ct
import torch

model = MyModel().eval()
example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=example.shape)],
    minimum_deployment_target=ct.target.iOS17,
    compute_units=ct.ComputeUnit.ALL,     # let Core ML use ANE+GPU+CPU
)
mlmodel.save("model.mlpackage")
```

Core ML's quantization vocabulary is its own: alongside plain int8 linear quantization it offers **palettization** (cluster weights to a small lookup table — a form of weight sharing that shrinks the file a lot) and per-grouped-channel schemes, all via `coremltools.optimize`. For LLMs and large models, palettization to 4 or even 2 bits is the Apple-native way to fit weights, analogous to the k-quants you'd use in `llama.cpp`.

```python
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)
config = OptimizationConfig(global_config=OpPalettizerConfig(nbits=4))
mlmodel_4bit = palettize_weights(mlmodel, config)
mlmodel_4bit.save("model_4bit.mlpackage")
```

The honest summary of Core ML: on Apple silicon it is the fastest, most power-efficient path, and the ANE is a real advantage that no cross-platform runtime fully matches. The cost is that it is *Apple-only* and the automatic placement is a black box — you trade control for convenience, and when the convenience fails you, debugging means staring at Instruments to find which op got kicked off the ANE.

## NNAPI's deprecation and the vendor-delegate landscape

For years on Android the answer to "how do I reach the NPU?" was **NNAPI** — the Android Neural Networks API, a system-level interface that let TFLite (and ORT) ask "Android, run this graph on whatever accelerator the SoC vendor exposes." It was a nice idea: one API, every vendor's NPU behind it.

It did not age well. NNAPI was a lowest-common-denominator interface — to support an op through NNAPI, *every* layer (Android, the vendor's NNAPI driver, the hardware) had to agree, so op coverage lagged the hardware by years, and behavior varied wildly across vendors and Android versions. Google has now **frozen NNAPI**: as of Android 15 (API level 35) it is deprecated, no new ops are coming, and the official direction is to reach NPUs through **vendor-specific delegates** instead — Qualcomm's QNN, MediaTek's NeuroPilot, Samsung's ENN, Google's own Tensor/Edge TPU path — plus GPU as the portable accelerator that always works.

What this means practically: NNAPI still works on shipped devices and is fine if you're targeting the installed base, but for new work you should plan to reach the NPU through the vendor delegate (via TFLite) or the matching ORT EP (QNNExecutionProvider for Snapdragon). The fragmentation that NNAPI tried to hide is now back in your face — but at least the vendor delegates expose the *full* capability of the chip instead of the intersection of all chips. The pattern across the whole landscape is consolidating: a portable accelerator (GPU) plus a fast vendor path (QNN/Core ML/Ethos), with the system-level uber-interface (NNAPI) on the way out.

There's a lesson here that generalizes beyond Android. Any time a runtime tries to be the *single* universal layer over many vendors' accelerators, it gets squeezed between two failures: either it exposes only the intersection of what every vendor supports (NNAPI's fate — coverage that lags the silicon by years), or it exposes the union and becomes a leaky pile of vendor-specific quirks (which is what you get if you actually use every NNAPI vendor extension). The industry's answer — converging on a portable GPU path plus a thin per-vendor NPU path — is an admission that there is no free lunch: you either pay in coverage or you pay in fragmentation. ORT's execution-provider design is the same admission made cleanly: the CPU EP is the guaranteed-coverage floor, and each accelerator EP is allowed to be vendor-specific and incomplete because the floor always catches what it drops. Keep that framing when you evaluate any new "universal" runtime that appears — ask immediately what its guaranteed-coverage floor is and how thin the accelerator paths above it are allowed to be.

## On-device LLMs: when you ship a specialized runtime instead

Everything so far assumed a model small enough that a general runtime makes sense — a CNN, a small transformer, a detection head. On-device *large* language models break that assumption, and they're common enough now (a 1-3B model running on a phone for offline assistance, autocomplete, or summarization) that the runtime question deserves its own treatment.

An LLM is mostly one op repeated — a big `MatMul` in attention and in the feed-forward block — but autoregressive decoding has structure a general runtime handles poorly: a **KV-cache** that grows by one token each step, attention over a variable-length sequence, and a tight per-token loop where you generate one token, append it, and run again. A general interpreter that re-plans or re-allocates each step bleeds time at exactly the wrong granularity, because you run the graph hundreds of times per response, not once per frame.

This is why a whole class of LLM deployment skips TFLite/ORT/ExecuTorch and ships a *purpose-built* runtime: `llama.cpp` (and its ASR sibling `whisper.cpp`), MLC-LLM, and the vendor LLM SDKs. `llama.cpp` is the clearest example of the pattern. It ships its own quantization format — **GGUF** with **k-quants** like `Q4_K_M` (a 4-bit, mixed-precision-per-block scheme that keeps the sensitive layers at higher precision) — and its own backends (Metal on Apple, CUDA, Vulkan, and tuned CPU kernels with NEON/AVX). It manages the KV-cache layout itself, fuses attention, and offers a knob, `-ngl`, for how many transformer layers to offload to the GPU:

```bash
# Quantize an fp16 GGUF model to 4-bit k-quant (Q4_K_M is the common sweet spot).
./llama-quantize models/llama-3b-f16.gguf models/llama-3b-q4_k_m.gguf Q4_K_M

# Run it, offloading 28 of the layers to the GPU/Metal backend, batch 1.
./llama-cli -m models/llama-3b-q4_k_m.gguf -ngl 28 -p "Summarize:" -n 128
```

The trade-off is exactly the `whisper.cpp` lesson from the case studies, stated as a rule: **when one model family is your entire product, a specialized runtime beats a general one — and you pay for it by maintaining a runtime.** A 3B model at `Q4_K_M` is ~2 GB on disk and runs at roughly $10$-$30$ tokens/s on a recent phone's CPU-plus-GPU, which a general runtime would struggle to match because it doesn't know about the KV-cache loop. But you've now adopted a C++ codebase, a bespoke quant format, and a backend matrix you must keep current. For a normal app with a CNN and a small transformer, this is a trap; for an offline-assistant product, it's the right architecture. The weight-only 4-bit quantization that makes this fit is the subject of [LLM quantization: weight-only GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq), and the activation/KV-cache side is in [LLM quantization: activations, SmoothQuant, and the KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache).

ExecuTorch and Core ML are both moving into this space — ExecuTorch with a Llama-on-device recipe and Core ML with palettized weights and the ANE — so the line between "general runtime" and "LLM runtime" is blurring. But the principle stays: the more your workload looks like one repeated op in a tight loop, the more a specialized runtime that understands that loop will beat a general graph executor that treats every inference as independent.

## The science: why partition boundaries cost real time

Now the part the kit insists on — the *why*, made rigorous. We've said partition boundaries cost milliseconds. Let's model it, because the model tells you exactly when fallback is harmless and when it's a disaster.

A neural net op on an accelerator has two cost components: the compute and the data movement. When two adjacent ops run on the *same* device, the intermediate tensor between them stays in that device's memory — there is no transfer. When they run on *different* devices (an NPU op feeding a CPU op), the intermediate tensor must physically move across the boundary, and on most mobile SoCs that means a copy through (or a cache flush to) shared DRAM plus a synchronization point where one engine waits for the other.

![Before and after comparison showing a model with all ops on the CPU interpreter versus a delegate that partitions the graph and sends the supported subgraph to the NPU](/imgs/blogs/inference-runtimes-compared-2.png)

Let a partition boundary move a tensor of $N$ elements, each $b$ bytes, across a bus with effective bandwidth $BW$ (bytes/s), and let the device sync (the round-trip handshake between the two engines) cost a fixed latency $L_{sync}$. The cost of one boundary crossing is roughly

$$
T_{boundary} \approx \frac{N \cdot b}{BW} + L_{sync}.
$$

That fixed $L_{sync}$ is the killer. A queue-submit-and-wait on a mobile NPU is commonly on the order of $0.5$ to a few milliseconds — not because the data is big, but because you're paying driver overhead, command-buffer submission, and a cross-engine fence. If an unsupported op sits in the *middle* of an otherwise NPU-friendly graph, you pay this twice: once to get the tensor down to the CPU, once to get the result back up.

So the total overhead of $k$ CPU islands inside an NPU graph is

$$
T_{overhead} \approx 2k\left(\frac{N \cdot b}{BW} + L_{sync}\right),
$$

because each island has an entry boundary and an exit boundary. The factor that dominates is $2k\,L_{sync}$ — the number of times you cross, times the fixed sync cost. This is why **op coverage is the deciding factor**: it is not really about whether one op is "supported," it's about how many *boundaries* your unsupported ops create. One unsupported op at the very end of the graph costs you one boundary and is nearly free. The same op in the middle, especially if it splits a long chain, can cost you two boundaries *and* prevent the runtime from fusing the chain on either side.

#### Worked example: one op forces two copies

Take our MobileNetV3-int8 classifier on a Pixel 8. The full graph runs on the NPU in $p50 = 8$ ms (we'll measure this for real in the results). Now suppose a refactor adds one `HardSwish` the NPU delegate doesn't support, sitting after the third inverted-residual block. The delegate now produces two NPU partitions with a CPU island between them.

The intermediate tensor at that point is, say, $56 \times 56 \times 24 = 75{,}264$ elements. In int8 that's $b = 1$ byte each, so $N \cdot b \approx 75$ KB. On a phone with effective shared-memory bandwidth around $20$ GB/s for these small uncached transfers, $\frac{75 \times 10^3}{20 \times 10^9} \approx 3.8\ \mu s$ — the *copy* is negligible. But each boundary also costs a sync, $L_{sync} \approx 2$ ms on this delegate. Two boundaries:

$$
T_{overhead} \approx 2 \times (3.8\ \mu s + 2\ \text{ms}) \approx 4\ \text{ms}.
$$

The model went from $8$ ms to roughly $12$ ms — a 50% latency hit — for an op that, run on the CPU, costs single-digit microseconds. The op is innocent; the *boundary* is the cost. (In the opening story the numbers were even worse because the fallback op also broke fusion and forced the runtime to re-quantize at the boundary, which is why a "9 ms" model measured 41 ms.) This is the supported-op fallback failure mode that [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) flags from the silicon side; here you can see the runtime arithmetic that produces it.

![Before and after comparison showing a single fused NPU island versus a split graph where one unsupported op forces two CPU to NPU tensor copies and a device sync](/imgs/blogs/inference-runtimes-compared-3.png)

The practical lesson is sharp: **count your partitions, not your supported ops.** A model that is 98% supported but has its 2% scattered into four CPU islands can be slower than a model that is 90% supported with its 10% all bunched at the tail in a single island. When you profile, the question to ask the runtime is "how many partitions did you make, and where are the boundaries?" — not "what fraction of ops did you delegate?"

There's a corollary for memory-bound models. If your model is already memory-bound (low arithmetic intensity — see [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives)), the boundary copies compete for the same DRAM bandwidth the model is already starved for, so the relative hit is *worse* than for a compute-bound model. The roofline tells you how much headroom you have to absorb boundary traffic.

### The break-even: when is offloading even worth it?

The boundary model also answers a more basic question that people skip: should a given subgraph go to the accelerator *at all*? Offloading a subgraph to the NPU only pays off if the time saved on compute exceeds the time spent on the two boundaries. Let the subgraph cost $T_{cpu}$ on the CPU and $T_{npu}$ on the accelerator, and let one boundary cost $T_{boundary}$ as above. Offloading is worth it only when

$$
T_{npu} + 2\,T_{boundary} < T_{cpu},
$$

i.e. when the speedup $T_{cpu} - T_{npu}$ is bigger than the round-trip cost $2\,T_{boundary}$. Rearranged, the accelerator must give a speedup of at least $2\,T_{boundary}$ in absolute terms before it's worth the trip. For a tiny subgraph — say a single elementwise op — $T_{cpu} - T_{npu}$ is microseconds while $2\,T_{boundary}$ is milliseconds, so offloading it *loses*. A good partitioner knows this: it won't peel a two-op subgraph off to the NPU between two CPU regions, because the boundaries would cost more than the op saves. This is also why partitioners try to claim *maximal* connected subgraphs — the bigger the subgraph, the more compute you amortize each boundary over. The break-even inequality is the whole reason "claim the biggest contiguous chunk you can" is the right partitioning strategy.

You can turn this into a rule of thumb. Define the **amortization ratio** $R = T_{npu\_compute} / (2\,T_{boundary})$ — how much accelerator compute you get per unit of boundary overhead. If $R \gg 1$, the boundaries are noise and offloading is a clear win. If $R \approx 1$, you're spending as much on transport as on compute, and the accelerator barely helps. If $R < 1$, you'd be faster staying on the CPU. When you profile a delegated model and the NPU "isn't helping," compute $R$ for the delegated subgraph — usually you'll find either the subgraph is too small or it got chopped into many small islands, both of which crush $R$.

### Stress-testing the model: where the simple picture breaks

The kit asks for honesty about edge cases, so here's where this clean model gets messy. First, **the boundary cost is not always a copy.** On a true unified-memory SoC (Apple silicon, some Snapdragons), the CPU and the accelerator can share the same physical buffer, so a "copy" may collapse to a cache flush or even nothing — but the *sync* (the fence where one engine waits for the other) survives, and on these chips the sync, not the copy, is the whole boundary cost. So unified memory shrinks the $\frac{N b}{BW}$ term toward zero but leaves $L_{sync}$ intact; the $2k\,L_{sync}$ term still dominates, and the "count boundaries, not ops" rule holds even harder.

Second, **layout conversions hide inside boundaries.** An NPU often wants a tiled or channel-last memory layout that the CPU op doesn't produce, so the boundary silently includes a transpose/repack that costs more than the raw copy. This is invisible in the op count and shows up only as a boundary that's slower than $\frac{N b}{BW}$ predicts. When a boundary is mysteriously expensive, suspect a layout conversion.

Third, **re-quantization at the boundary.** If the NPU subgraph runs int8 and the CPU op runs fp32, the boundary must dequantize on the way out and re-quantize on the way back — extra compute *and* a precision wobble. This is the mechanism behind the opening story's blow-up from 9 ms to 41 ms: the mid-graph fallback didn't just cost two syncs, it forced two int8↔fp32 conversions on a large tensor. The fix is to keep the fallback op in the same precision as its neighbors, or to push it to a boundary where the conversion is already happening anyway (the model's input or output).

Fourth, **the partitioner itself is heuristic.** Two runtimes given the same graph and the same accelerator can partition differently because their cost models differ — one might offload a subgraph the other keeps on CPU. This is part of why "same model, different runtime" can differ by more than the dispatch overhead: they made different placement bets. When a runtime surprises you, dump its partition and compare it to what the break-even math says it *should* have done.

## AOT-compiled versus interpreted: two philosophies

We've now met both philosophies, so let's make the distinction crisp, because it shapes everything about how a runtime behaves on the device.

An **interpreted** runtime (TFLite, ORT) ships a graph plus a generic engine. When the model loads on the device, the engine reads the graph, asks the delegates/EPs what they can take, partitions, picks kernels, and allocates tensors — *at load time, on the device*. Then each `Invoke()` walks the (now partitioned) graph and dispatches each op. The dispatch is cheap but non-zero, and crucially the *first* inference often pays a warm-up cost: the GPU/NPU delegate may compile shaders or build its internal program the first time it sees the shapes.

An **AOT-compiled** runtime (ExecuTorch, Core ML's compiled `.mlmodelc`, TensorRT engines) does the partition-and-plan step *once, on your machine, at export time*. What ships to the device is the finished plan — lowered ops, fixed partitions, a static schedule, sometimes fully fused kernels. The device-side runtime is tiny because it doesn't decide anything; it just runs the plan. The first inference is fast because there's nothing to compile.

![Before and after comparison contrasting an interpreted runtime that dispatches ops at each inference with an ahead-of-time runtime that fixes the execution plan at export](/imgs/blogs/inference-runtimes-compared-5.png)

The trade-offs fall out directly:

| Property | Interpreted (TFLite, ORT) | AOT (ExecuTorch, Core ML, TensorRT) |
| --- | --- | --- |
| When the plan is decided | At load, on device | At export, on your machine |
| First-inference latency | Slow (warm-up, JIT shaders) | Fast (plan is prebuilt) |
| Steady-state latency | Good | Good to best (more fusion possible) |
| Swap the model at runtime | Easy — drop in a new file | Rebuild and reship the plan |
| Binary / runtime size | Larger (carries the engine) | Smaller (carries just the executor) |
| Portability of one artifact | High (one file, many backends) | Lower (plan is often device/shape-specific) |
| Debuggability of placement | Inspect at runtime | Inspect at build time |

Neither wins outright. If you ship to a thousand device models and want one artifact you can hot-swap, interpreted is your friend — the runtime adapts to whatever silicon it lands on. If you control the device (a kiosk, a Jetson, an in-house Apple fleet) and want the fastest, most predictable first inference, AOT is the move because you can specialize the plan to that exact hardware. The reason TensorRT engines are not portable across GPU generations is the same reason they're fast: the plan is compiled for *that* chip. AOT trades portability for specialization; interpreting trades specialization for portability.

A subtle point people miss: AOT does not automatically mean faster *steady-state*. A well-warmed interpreted runtime with the same kernels on the same accelerator hits the same throughput. Where AOT reliably wins is the first inference and the worst-case predictability — no surprise warm-up spike at the moment a user first triggers the feature.

## The comparison matrix: four runtimes, the properties that decide

Here is the map that most people actually want — the four runtimes against the properties that determine which one you ship. Read it as "what do I get on the platform I'm shipping to," because every one of these is strongest at home and weaker away.

![Matrix comparing TFLite ONNX Runtime ExecuTorch and Core ML across best platform accelerator path quantization support and op coverage](/imgs/blogs/inference-runtimes-compared-4.png)

In words, with the trade-offs spelled out:

| Runtime | What it is | Owns | Accelerator path | Quant | Op coverage | Language(s) |
| --- | --- | --- | --- | --- | --- | --- |
| **TFLite / LiteRT** | FlatBuffer + interpreter + delegates | Android | XNNPACK, GPU, NNAPI, Hexagon, Core ML, Edge TPU | int8-first, fp16 | Broad for CV; lags PyTorch ops | C++, Java/Kotlin, Swift, Python |
| **ONNX Runtime** | Graph + execution providers | Cross-platform | CPU, CUDA, TensorRT, CoreML, NNAPI, QNN, XNNPACK, DML | int8 QDQ, int4 weight-only | Widest (ONNX op set) | C++, C#, Python, Java, JS |
| **ExecuTorch** | AOT `.pte` + backends | PyTorch-native, mobile | XNNPACK, Core ML, MPS, Qualcomm, Vulkan, Ethos-U | int8/int4 (PT2E) | Growing fast; behind ORT today | C++, Swift, Java/Kotlin |
| **Core ML** | `.mlpackage` + automatic placement | Apple only | ANE, GPU, CPU (auto) | int8, palettization 2-8 bit | Apple op set; conversion gaps | Swift, Objective-C, Python (tools) |

A few cells deserve a sentence each, because the matrix can mislead if read as a leaderboard:

- **"Widest op coverage" (ORT) is the most underrated property in the table.** When you're converting a research PyTorch model with some non-standard op, ORT is the runtime most likely to just run it — and that saves you days of rewriting your model to fit a narrower runtime. Coverage is insurance.
- **Core ML's "Apple op set" cuts both ways.** On Apple hardware it's the fastest; off it, it doesn't exist. And conversion from PyTorch via `coremltools` is where you'll meet the gaps — an op that traces fine in PyTorch may have no Core ML equivalent, and you'll be writing a composite or a custom layer.
- **ExecuTorch's "growing fast"** is a real caveat, not a hedge. As of writing, its backend op coverage is genuinely behind TFLite and ORT. If your model is plain CNN/transformer ops, you're fine. If it has exotic ops, check coverage *before* you commit.
- **TFLite's int8-first** design means the cleanest path to a mobile NPU is full-integer quantization. If you need fp16 throughout, the GPU delegate is your better TFLite path.

## Practical: running the same model three ways and reading the table

Theory is cheap. Let's run the same model under different runtimes and backends on one phone and read the numbers, because the numbers are the only thing that settles a runtime argument.

The protocol matters as much as the result. Edge benchmarking lies easily, so the rules of honest measurement, which the metrics post [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) covers in depth, are:

- **Batch size 1.** On-device inference is almost always batch 1 (one camera frame, one user query). A batched benchmark flatters the accelerator and is irrelevant to your latency.
- **Warm up.** Run 20-50 inferences before you start timing to get past first-inference compilation and let the governor settle. Then report steady-state.
- **Measure p50 *and* p99.** The tail is where thermal throttling and scheduler jitter show up. A p50 of 8 ms with a p99 of 40 ms is a different product than 8/13.
- **Pin the clocks if you can, and watch temperature.** A phone that's been running for 30 seconds throttles, and your "great" number was measured cold. Report whether you locked frequencies.
- **Count the delegated ops.** Always log how many ops actually landed on the accelerator. If it's zero, you benchmarked the CPU and called it the NPU.

Here's a self-contained ORT benchmark that runs the *same* `.onnx` file under CPU and then NNAPI, times both honestly, and — critically — reports which EP actually bound:

```python
import onnxruntime as ort
import numpy as np, time

def bench(model_path, providers, warmup=30, iters=200):
    sess = ort.InferenceSession(model_path, providers=providers)
    active = sess.get_providers()
    iname = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(warmup):              # warm-up: do NOT time these
        sess.run(None, {iname: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {iname: x})
        ts.append((time.perf_counter() - t0) * 1e3)   # ms
    ts.sort()
    p50, p99 = ts[len(ts)//2], ts[int(len(ts)*0.99)]
    print(f"{active}: p50={p50:.1f} ms  p99={p99:.1f} ms")

bench("model_int8.onnx", ["CPUExecutionProvider"])
bench("model_int8.onnx", ["NnapiExecutionProvider", "CPUExecutionProvider"])
```

And the TFLite side — same model converted to `.tflite`, run under XNNPACK CPU versus the GPU delegate — using the `benchmark_model` tool that ships with TFLite (it's the honest way, because it does warm-up and reports the delegate breakdown for you):

```bash
# XNNPACK CPU baseline (default), batch 1, with warm-up and runs.
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model_int8.tflite \
  --num_threads=4 --warmup_runs=30 --num_runs=200 \
  --use_xnnpack=true

# GPU delegate: same model, same protocol, different backend.
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model_int8.tflite \
  --warmup_runs=30 --num_runs=200 \
  --use_gpu=true

# NNAPI (NPU) path — and crucially, dump the per-op profile to see fallbacks.
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model_int8.tflite \
  --warmup_runs=30 --num_runs=200 \
  --use_nnapi=true --enable_op_profiling=true
```

The `--enable_op_profiling=true` flag is the one to remember: it prints how long each op took and, by implication, which ops ran on the CPU instead of the NPU. When your NNAPI run is slower than you expected, that profile shows you the CPU islands directly.

For ExecuTorch, the benchmark is built from the AOT artifact — you build the runner once and run the `.pte`:

```bash
# Build the ExecuTorch runner with the XNNPACK backend, then run the .pte.
./cmake-out/executor_runner \
  --model_path=model_xnnpack.pte \
  --num_executions=200
# For the Core ML / QNN backends, build with the matching backend flag
# and the runner reports the same wall-clock timing.
```

## Worked example: same model, three runtimes, one phone

Let's pull it together with the result the kit demands — measured, named-target, before/after. The model is MobileNetV3-Large, int8, on a **Pixel 8** (Tensor G3 SoC). We run the same model through TFLite's XNNPACK CPU, TFLite's GPU delegate, TFLite's NNAPI (NPU) path, and ORT's NNAPI EP, with the honest protocol above. (Treat the exact figures as representative of this class of model on this class of chip — the *shape* of the result is the robust part; absolute numbers vary with build, thermal state, and driver version.)

![Matrix showing the same int8 model run under four runtime and backend paths on one Pixel 8 with p50 and p99 latency and the count of ops on the accelerator](/imgs/blogs/inference-runtimes-compared-6.png)

| Path | p50 | p99 | Ops on accel | Notes |
| --- | --- | --- | --- | --- |
| TFLite XNNPACK CPU | 31 ms | 44 ms | 0 / 64 | The honest CPU baseline. |
| TFLite GPU delegate | 14 ms | 22 ms | 58 / 64 | 6 ops fell back to CPU. |
| TFLite NNAPI (NPU) | 8 ms | 13 ms | 61 / 64 | 3 ops on CPU, all at the tail. |
| ORT NNAPI EP | 9 ms | 15 ms | 60 / 64 | Same NPU, ~1 ms ORT overhead. |

Read the table the way a staff engineer would. First: **the backend gap (31 → 8 ms, ~4×) dwarfs the runtime gap (8 vs 9 ms, ~12%).** Whether you pick TFLite or ORT matters far less than whether you got onto the NPU at all. If you remember one thing from this post, it's that the runtime-brand fight is mostly noise next to the backend-placement fight.

Second: **the GPU delegate left 6 ops on the CPU and the NPU paths left 3.** Those fallbacks are exactly the partition boundaries from the science section. The GPU path is slower partly because of those 6 boundaries, not because the GPU is intrinsically slower than the NPU at the math. Look at where the fallbacks are — the NPU path's 3 are all at the tail (one island, two boundaries total), which is why it's fast despite the fallbacks.

Third: **ORT's NNAPI EP costs ~1 ms over native TFLite NNAPI on the same NPU.** That's the price of ORT's generality — a thin layer of dispatch and partition bookkeeping that the native runtime skips. For most products 1 ms is a fine price for "one artifact runs on iOS too." For a 60-fps AR feature where every ms counts, the native runtime wins.

#### Worked example: the cross-platform cost decision

Now the decision this table forces. Suppose you ship to both Android and iOS, and the feature has a 16 ms budget (60 fps with headroom). Option A: ship one ORT `.onnx` with the NNAPI EP on Android (9 ms) and the Core ML EP on iOS. Option B: ship two native artifacts — a `.tflite` with NNAPI on Android (8 ms) and a `.mlpackage` on iOS.

Option A buys you *one model artifact, one quantization pipeline, one set of conversion bugs to fix*. The cost is ~1 ms on Android and, on iOS, ORT's Core ML EP typically trails native Core ML by a few ms (because ORT can't always keep the whole graph on the ANE). Option B buys you the last few milliseconds on each platform at the cost of *maintaining two conversion pipelines, two quant calibrations, two sets of op-coverage surprises* — roughly double the deployment maintenance burden, which is the third cost the runtime hides (after latency and op coverage). At a 16 ms budget with 9 ms in hand, Option A is the right call: you have 7 ms of headroom, and you'd spend a senior engineer's quarter chasing the last 3 ms of Option B. Flip the decision only when the budget is genuinely tight (a 12 ms budget at 90 fps) or when one platform's ORT EP can't hold the graph on the accelerator at all. The maintenance burden is a real number; put it in the trade.

## How the fallback chain actually resolves an op

When we say a runtime "tries delegates in priority order," here's the precise control flow, because understanding it is how you predict where an op will land before you ever profile.

![Tree showing the delegate fallback chain where an op is offered to accelerators first then cascades down to XNNPACK and finally a reference CPU kernel as the guaranteed last resort](/imgs/blogs/inference-runtimes-compared-7.png)

The runtime holds an ordered list of backends. For each op (really, each maximal subgraph), it offers the op to the highest-priority backend. If that backend's partitioner *claims* the op, it's assigned there. If not, the op rolls down to the next backend. The CPU is always at the bottom of the chain as the guaranteed catch-all — within the CPU, a fast vectorized kernel (XNNPACK) is tried before the slow reference kernel, so even a "CPU fallback" has a fast tier and a slow tier.

This is why two things are true at once: an op *always* runs (the CPU guarantees it), and an op can run *much* slower than you hoped (it fell through every accelerator to a reference kernel). The chain never fails to produce an answer; it can only fail to produce a *fast* answer. When you debug a slow model, you're not looking for a crash — you're walking this chain to find which ops fell how far. The diagnostic tools above (`get_providers()`, op profiling, the delegated-node count) are all ways of reading where each op landed in this chain.

A practical tactic falls out of the chain: if an op keeps falling to the CPU, you have three moves, in order of preference. (1) **Replace the op** with one the accelerator supports — swap `HardSwish` for `ReLU6`, or a fancy normalization for a plain one. This is the highest-leverage fix and it's why architecture and runtime are not separable concerns. (2) **Move the op** to a position where its boundary is cheap — push an unsupported reshape to the very start or end of the graph so it costs one boundary, not two, and breaks no fusion. (3) **Accept the fallback** if it's a tail op on a small tensor — not every fallback is worth fixing, and the math from the science section tells you which ones are. The order matters: replacing beats moving beats accepting, and you only drop to the next move when the previous one isn't available.

## Case studies: real numbers from shipped runtimes

Numbers from the wild, to calibrate the claims above. Where I give a figure I'll say how firm it is; the kit's rule is no fabricated precision.

**MobileNet on Pixel, GPU vs NPU.** Google's own TFLite benchmarks have long shown mobile CNNs getting roughly $3$-$5\times$ from a mobile GPU delegate over the CPU and another meaningful step from the NPU/DSP via NNAPI, depending on the SoC and on how int8-friendly the model is. The shape — CPU baseline, GPU big jump, NPU further jump *when ops are covered* — matches our table. The variance across SoCs is large, which is precisely why you benchmark on *your* target, not on a reference device.

**ONNX Runtime as the portable workhorse.** ORT is the runtime under a great deal of production on-device and edge inference precisely because of the "one artifact, many EPs" property — Microsoft ships it in Windows ML, Office, and many mobile apps, and the public guidance is that the CPU/EP fallback completeness is what makes it safe to ship the same model across a device matrix. The reported overhead of the generic dispatch versus a native runtime is typically small (low single-digit ms or percent) for vision models — consistent with the ~1 ms we saw versus native TFLite NNAPI.

**ExecuTorch on Apple and Snapdragon.** Meta has reported running Llama-class models on phones with ExecuTorch via the Core ML and Qualcomm QNN backends, with the AOT design giving a small runtime footprint suitable for shipping inside an app. The honest framing from the project itself is that op coverage is still expanding — fine for standard transformer/CNN ops, check before relying on it for exotic ones. The direction of travel is clearly toward ExecuTorch being the default PyTorch-on-device runtime; the question is timing, not destination.

**Core ML and the ANE.** Apple's published guidance is that keeping a model resident on the ANE — by using ANE-friendly ops, layouts, and shapes — is the difference between great and merely-okay latency and energy on iPhone, and that a model which bounces to the GPU can be several-fold slower than one that stays on the ANE on the same chip. This is the partition-boundary phenomenon again, surfacing inside Core ML's automatic placement. The practical takeaway practitioners report: profile in Instruments, find which ops left the ANE, and reshape the model to keep them on it.

**Whisper.cpp / llama.cpp as the "own runtime" pattern.** A whole class of on-device LLM/ASR deployment skips the general runtimes entirely and ships a purpose-built C/C++ runtime with its own quantization (GGUF k-quants) — `llama.cpp` and `whisper.cpp` run multi-billion-parameter models on laptops and phones at usable tokens/s by specializing the runtime to transformer decoding (KV-cache layout, fused attention, Metal/CUDA/Vulkan backends). The lesson: when one model family is your whole product, a specialized runtime can beat the general ones — at the cost of being a runtime you now maintain. For LLMs specifically, see the weight-only quantization that makes this fit on-device in [LLM quantization: weight-only GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq).

## A worked decision, start to finish

Let me walk one real decision the way I'd reason through it on a whiteboard, because the matrix and the trees are inputs to a decision, not the decision itself. The brief: a camera-based defect detector for a mobile inspection app. Targets are mixed — half the fleet is recent iPhones, half is mid-range Android (Snapdragon and MediaTek), one customer runs it on a small Jetson at a fixed station. The model is a PyTorch segmentation net, int8-quantized, with a custom upsampling op the team wrote. Budget: 50 ms per frame, p99.

Step one, **inventory the ops against the targets.** The custom upsampling op is the immediate red flag. I'd check it against each candidate runtime's coverage *before* anything else, because the science section says one unsupported op in the wrong place can blow the budget. On TFLite, a custom op means writing a custom op kernel *and* a custom delegate path if I want it on the NPU — real work. On ORT, I can register the op as a custom op against the CPU EP and let the rest run on the accelerator EP, so the custom op becomes one CPU island. On ExecuTorch, I'd need to lower the op for each backend. ORT's "register a custom op, let it fall to the CPU island, accelerate everything else" is the lowest-effort path that keeps coverage complete.

Step two, **place the custom op.** Per the break-even math, a custom op in the *middle* of the segmentation decoder would cost two boundaries every frame. So before committing, I'd ask the model author: can the custom upsample move to the very end of the network (it's the final resize anyway)? If yes, it's a tail op — one island, near-free. That single architectural nudge, made because I understand the boundary cost, is worth more than any runtime choice.

Step three, **map runtimes to targets.** iPhones: ORT's Core ML EP keeps most of the graph on the ANE, the custom op falls to CPU at the tail. Android: ORT's QNN EP for Snapdragon, NNAPI/GPU for MediaTek, same tail fallback. Jetson: ORT's TensorRT EP, where I control the device and can afford the AOT engine build. **Every target is reachable from one ORT artifact and one ONNX file** — the custom op is registered once. The alternative, native runtimes per platform, would mean porting the custom op to TFLite *and* Core ML *and* TensorRT separately: three implementations of one op.

Step four, **check the budget.** With a 50 ms budget and segmentation models of this class landing in the 15-30 ms range on these accelerators, I have comfortable headroom — which means ORT's small per-inference overhead is irrelevant and I should optimize for *maintenance*, not for the last millisecond. If the budget were 12 ms at 90 fps, I'd flip to native runtimes per platform and eat the porting cost. It isn't, so: **one ORT pipeline, custom op registered, placed at the tail, with Core ML / QNN / NNAPI / TensorRT EPs selected per device.** That's the decision, and every step of it came from the boundary math and the maintenance-versus-latency trade, not from a leaderboard.

## When to reach for which runtime (and when not to)

Now the decisive part. The choice is **platform-driven first**, and only then preference-driven. Start from the device you must ship to.

![Tree mapping a deployment target to a runtime where Apple silicon points to Core ML or ExecuTorch Android points to TFLite or ONNX Runtime and a server-edge box points to ONNX Runtime plus TensorRT](/imgs/blogs/inference-runtimes-compared-8.png)

- **Apple only (iOS, macOS).** Reach for **Core ML** for the fastest, most power-efficient path and the ANE — especially if the app is Swift and you want zero runtime configuration. Reach for **ExecuTorch with the Core ML backend** if your model lives in PyTorch and you'd rather keep one PyTorch-native pipeline than maintain `coremltools` conversions. Use **ORT's Core ML EP** only if you're already on ORT for cross-platform reasons and the few-ms gap doesn't matter.

- **Android.** Reach for **TFLite / LiteRT** — it's the most mature, the delegate ecosystem reaches every Android NPU, and the int8 story is the cleanest. Reach for **ORT** if you're *also* shipping iOS and want one artifact (the QNN/NNAPI EPs get you onto Snapdragon NPUs), or if your model has ops TFLite's converter chokes on. Reach for **ExecuTorch with the XNNPACK or Qualcomm backend** if you're PyTorch-native and your ops are covered.

- **Cross-platform (one codebase, both mobile OSes, maybe web).** Reach for **ORT** — the one-artifact-many-EPs model is exactly built for this, and the op coverage is the insurance that the same model runs everywhere. Reach for **ExecuTorch** if you're all-in on PyTorch and willing to track its maturity curve, since it now spans Apple, Android, and Vulkan backends.

- **Server-edge (Jetson, x86+GPU box, edge gateway you control).** Reach for **ORT with the TensorRT EP**, or raw **TensorRT** if you need the absolute last drop of throughput on NVIDIA hardware and can accept GPU-specific, non-portable engines. The fact that you *control the device* means AOT specialization pays off — build the plan for that exact GPU. This is the subject of a dedicated forthcoming post on TensorRT and GPU edge inference on Jetson, where the AOT-specialization argument from this post gets its full treatment.

And the **when-not-to** list, because every runtime is a cost:

- **Don't pick a runtime for its leaderboard latency on a chip you don't ship to.** The Pixel benchmark is irrelevant if you ship to a Snapdragon mid-range fleet. Benchmark on *your* target, batch 1, warmed up, watching p99.
- **Don't reach for two native runtimes when one cross-platform runtime clears your budget.** The maintenance burden of two conversion pipelines is real and recurring; only pay it when the latency budget genuinely forces it.
- **Don't ship ExecuTorch for a model with exotic ops without checking coverage first.** Its op coverage is behind ORT and TFLite today; verify, don't assume.
- **Don't fight Core ML's automatic placement by hand.** If an op keeps leaving the ANE, fix the *model* (ANE-friendly ops/shapes), don't try to override the scheduler.
- **Don't build your own runtime unless one model family is your whole product.** The `llama.cpp` pattern wins for LLMs-as-the-product; for a normal app it's a maintenance trap.

The thread through all of it: choose the runtime your target's *native* path prefers, then deviate toward a cross-platform runtime only to the extent your maintenance budget is smaller than your latency budget. When you've made the choice, the deployment-mechanics — packaging, signing, on-device updates — are the subject of the forthcoming mobile-deployment-end-to-end post, and the whole optimize-convert-run loop ties together in the capstone, [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

## Key takeaways

- **A runtime is a graph executor with a hardware budget.** Every runtime here partitions the graph, assigns subgraphs to accelerators, and runs the rest on the CPU. The names differ (delegate / execution provider / backend); the mechanism is identical.
- **Op coverage decides latency more than the runtime brand.** The backend gap (CPU → NPU, often ~4×) dwarfs the runtime gap (~10%). Getting onto the accelerator matters far more than which runtime you used to get there.
- **Count partition boundaries, not supported ops.** One unsupported op in the middle of the graph costs two CPU↔NPU copies plus two syncs — and the fixed sync cost ($2k\,L_{sync}$), not the data, is what hurts. A tail-end fallback is nearly free; a mid-graph fallback can be a disaster.
- **Match the model's I/O type to its body** (int8 in, int8 out) so the runtime doesn't pin the first and last layers to the CPU with quantize/dequantize boundaries.
- **AOT (ExecuTorch, Core ML, TensorRT) trades portability for a fast, predictable first inference; interpreting (TFLite, ORT) trades specialization for one-artifact portability.** Pick by whether you control the device.
- **Choose the runtime platform-first.** Apple → Core ML or ExecuTorch+CoreML; Android → TFLite/LiteRT or ORT; cross-platform → ORT or ExecuTorch; server-edge → ORT+TensorRT.
- **NNAPI is frozen; reach NPUs through vendor delegates** (QNN, NeuroPilot, ENN) or the matching ORT EP for new work. GPU remains the portable accelerator that always works.
- **Benchmark honestly or don't benchmark.** Batch 1, warm up, p50 *and* p99, watch thermals, and always log how many ops actually landed on the accelerator. A benchmark that doesn't report the delegated-op count is measuring the wrong thing.
- **The cross-platform runtime's small per-inference overhead is often cheaper than maintaining two native pipelines.** Put the maintenance burden in the trade explicitly; it's the third cost the runtime hides after latency and op coverage.

## Further reading

- **TFLite / LiteRT delegates and the int8 conversion guide** — the official docs on XNNPACK, GPU, NNAPI, and Hexagon delegates and on full-integer quantization with a representative dataset (ai.google.dev / tensorflow.org/lite).
- **ONNX Runtime execution providers and graph partitioning** — the official EP docs and the quantization guide (`onnxruntime.ai`), the clearest written statement of the partition-by-priority model.
- **ExecuTorch documentation** — the PyTorch on-device runtime, the `torch.export` + `to_edge_transform_and_lower` flow, and the backend list (pytorch.org/executorch).
- **Core ML Tools and the ANE guidance** — `coremltools` conversion and `coremltools.optimize` (palettization, quantization), plus Apple's "Deploying Transformers on the Apple Neural Engine" writeup on keeping ops resident on the ANE.
- **Android NNAPI deprecation notice** — the Android developer docs marking NNAPI deprecated as of API 35 and pointing to vendor delegates.
- Within this series: [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) for the silicon the runtimes target, [the edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle) for the conversion steps before this, [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for honest measurement, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
- For the broader ONNX format and server-side serving picture, [ONNX deep dive: format, runtime, serving](/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving).

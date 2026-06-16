---
title: "Mobile deployment end to end: shipping a model on Android and iOS"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Take an optimized, converted model and actually ship it inside a phone app on both Android and iOS — with delegate and compute-unit selection, warm-up, preprocessing parity, app-size budgets, and honest on-device latency you can measure and defend."
tags:
  [
    "edge-ai",
    "model-optimization",
    "mobile",
    "android",
    "ios",
    "tflite",
    "coreml",
    "executorch",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/mobile-deployment-end-to-end-1.png"
---

There is a specific kind of meeting I have come to dread. The model is done — quantized to int8, two points of accuracy traded for a 4× size cut, validated against the holdout set, converted cleanly to the target format. Someone presents a slide with a latency number measured in a notebook on a workstation: "8 milliseconds, well under our 33 ms budget for 30 fps." Everyone nods. The model is "ready." And then three weeks later the same model is stuttering at 90 ms on a mid-range Android phone, the app's install size has ballooned past the cellular-download warning threshold, the first inference after launch freezes the camera preview for almost half a second, and the battery drains noticeably during a two-minute session. Nothing about the *model* changed. Everything about the *deployment* did.

This post is about closing that gap — the distance between "runs in a notebook" and "runs in the App Store build that a real person installed over LTE on a four-year-old phone." It is the practical capstone of the runtime track of this series. We assume the upstream work is already done: you have chosen and applied your compression levers ([a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression)), you have converted the model and picked a runtime ([inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared)), and you understand the silicon you are targeting ([the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape)). What remains is the part nobody puts on a slide: putting the thing inside a phone app, on *both* platforms, without tanking battery, frame rate, or app size — and being able to say, with numbers you would defend under oath, how fast it actually is on the device in the user's hand.

By the end you will be able to take an optimized model and wire it into a real Android app with a delegate-selection-and-fallback ladder, into a real iOS app with the right compute-unit policy, hide the first-inference compile cost behind a warm-up, keep your on-device preprocessing bit-for-bit consistent with training, budget the app-size impact of the runtime and the model, decide between bundling the model and downloading it over the air, and measure latency and memory the way a skeptic would — warm-started, batch=1, with thermals in mind. The figure below is the whole journey on one timeline: the model leaves the notebook on the left and arrives, signed and instrumented, in a store build on the right, and almost every box between them is a place I have personally watched a launch slip.

![A six-stage timeline showing a model traveling from a notebook through convert and quantize, embed in app, pick delegate, warm up, and finally a signed build with telemetry](/imgs/blogs/mobile-deployment-end-to-end-1.png)

Keep that timeline in your head. The rest of this post is a walk through each stage with the actual code, the actual flags, the actual failure modes, and before-after numbers on named devices. We will frame it against the series' recurring spine — the four levers (quantization, pruning, distillation, efficient architecture) sit on **compilers and runtimes**, validated by **profiling**, read off the **accuracy-efficiency Pareto frontier**. Mobile deployment is where the runtime layer stops being a diagram and becomes a concrete `Interpreter` object or `MLModel` instance that either initializes on the user's specific SoC or does not.

## Why "it runs" is the wrong bar

Let me be precise about what changes between the notebook and the device, because the failure modes all descend from these differences and naming them is half the battle.

In a notebook you run on a server CPU or a datacenter GPU with effectively unlimited memory bandwidth, no thermal ceiling on the timescale of a single inference, and a Python process that owns the whole machine. On a phone you run on a heterogeneous SoC — a cluster of CPU cores (some "big," some "little"), a mobile GPU, and increasingly an NPU (Neural Processing Unit: a fixed-function accelerator for the multiply-accumulate-heavy operations in neural networks). That SoC is shared with the OS, the UI compositor, the camera pipeline, and every other app. Memory is a few gigabytes total, and memory *bandwidth* — the rate at which you can move weights and activations between DRAM and the compute units — is a small fraction of a datacenter part. And the device gets *hot*: after a sustained workload the SoC throttles its clocks to stay under a thermal limit, so the latency you measure in the first ten seconds is not the latency a user gets two minutes in.

The single most important consequence: the backend you tested on is almost never guaranteed to be the backend that runs in production. You test on your Pixel with a working GPU delegate, and the app ships to a phone whose vendor NPU driver silently rejects one of your operators and routes the entire subgraph back to the CPU — quietly, with no error, at 3× the latency. This is not a hypothetical; it is the default behavior of mobile inference runtimes, and designing *for* it rather than being surprised by it is the central skill of mobile deployment.

So the bar is not "it runs." The bar is: it runs **fast enough on the slowest device we support**, **falls back gracefully when the accelerator is unavailable**, **doesn't block the UI thread**, **produces bit-identical preprocessing to training**, **fits the install-size budget**, and **survives sustained use without thermal collapse** — and you have **telemetry** to confirm all of that in the field. The rest of this post is each of those clauses, made concrete.

## Android end to end

Android is the harder of the two platforms precisely because it is not one platform. "Android" spans a Pixel with a Google Tensor NPU, a flagship with a Qualcomm Hexagon, a mid-range MediaTek part, and a budget device with no usable NPU at all. Your code has to behave on all of them. The figure below shows the call path: your Kotlin app sits on top of an inference runtime, which sits on top of a *delegate*, which talks to a vendor driver, which finally reaches the SoC hardware. The layer you ship is rarely the layer you tested, because any one of those layers can route work back down to the CPU.

![A five-layer Android inference stack showing the Kotlin app on top, then the LiteRT runtime, then the delegate, then the vendor NN driver, then the SoC hardware at the bottom](/imgs/blogs/mobile-deployment-end-to-end-2.png)

### The runtime choice on Android

You have three serious options for running a neural network on Android, and they overlap.

**LiteRT** (the runtime formerly and still widely known as **TensorFlow Lite**, recently rebranded) is the default for most teams. You ship a `.tflite` flatbuffer model plus the LiteRT `.so` shared libraries, and you call it through a Java/Kotlin `Interpreter` or the C++ API. It has the broadest delegate ecosystem — XNNPACK for CPU, a GPU delegate, NNAPI (now deprecated, more on that below), and vendor delegates.

**ONNX Runtime Mobile** is the strong alternative if your model is already in ONNX. It ships a slimmed-down build (`onnxruntime-mobile`) with an ORT-format model and execution providers (EPs) that play the same role as LiteRT delegates: XNNPACK, NNAPI, QNN (Qualcomm), and others. If your team standardizes on ONNX across server and edge, ORT Mobile keeps one model format.

**ExecuTorch** is PyTorch's native on-device runtime. You `torch.export` the model, lower it to ExecuTorch with a backend partitioner (XNNPACK, Qualcomm, MediaTek, Vulkan), and ship a `.pte` file plus the ExecuTorch runtime. It is the natural choice if your training stack is PyTorch and you want to avoid the lossy round-trip through another format. It is younger than LiteRT but maturing fast and is the direction PyTorch is investing in for edge.

For this post I will use **LiteRT** for the running Android examples because it is the most common and its delegate model is the clearest teaching example, and I will note the ONNX Runtime and ExecuTorch equivalents where they differ.

### Delegates: where the speed (and the danger) lives

A *delegate* in LiteRT is a plugin that takes part of your model's computation graph and runs it on a specific backend. When you create an interpreter you hand it a list of delegates; the runtime walks your graph and, for each delegate in order, partitions out the subgraph of operators that delegate supports and assigns them to it. Whatever is left over runs on the CPU reference kernels.

This partitioning is the crux. If your model is, say, a MobileNetV3, almost every op (depthwise conv, pointwise conv, hardswish, global pool) is supported by the GPU and NNAPI delegates, so the whole thing gets delegated and you get the accelerator speed. But if your model contains one operator the delegate doesn't support — a custom op, an unusual `ResizeBilinear` configuration, a `5D` reshape — the graph gets *split* at that op. Now you have a CPU island in the middle of a GPU subgraph, and crossing that boundary means copying the activation tensor out of the accelerator's memory, running the lone op on the CPU, and copying back. Those copies can cost more than the op you "accelerated."

The standard delegate ladder, fastest to most portable:

1. **Vendor NPU / NNAPI / QNN** — the dedicated neural accelerator. Fastest and most power-efficient when it works, most likely to reject ops or be unavailable.
2. **GPU delegate** — the mobile GPU. Good middle ground, fp16-friendly, but pays a shader-compile cost on first use and a copy cost crossing CPU↔GPU.
3. **XNNPACK (CPU)** — the optimized CPU kernels. Always available, always works, slowest. This is your floor and your fallback.

XNNPACK is enabled by default in modern LiteRT for float models, which is why even a "CPU-only" path is reasonably fast. The whole point of the ladder is: *try the fast thing, and if it fails to initialize or isn't there, fall down to the next one, ending on a CPU path that cannot fail.*

### The NNAPI deprecation, and what replaced it

For years the recommended way to reach a vendor NPU on Android was **NNAPI** (the Neural Networks API), a system-level interface that let the runtime ask the OS to dispatch ops to whatever accelerator the device had. Google has **deprecated NNAPI as of Android 15** in favor of having vendors ship their own delegates that talk directly to their hardware. In practice this means: do not build new code around NNAPI as your NPU path. Use the GPU delegate as your portable accelerator, and use **vendor-specific delegates** (Qualcomm's QNN delegate, MediaTek's NeuroPilot/Litert delegate, Samsung's ENN) when you have the device coverage to justify them and want the last increment of NPU performance. The fallback logic in your code stays the same; only the top rung of the ladder changes from "NNAPI delegate" to "vendor delegate, if present." Everything I show below treats the top rung as pluggable for exactly this reason.

### The Kotlin call path with delegate selection and fallback

Here is a real, idiomatic LiteRT inference path in Kotlin. It loads the model from assets, tries the GPU delegate, falls back to a multithreaded XNNPACK CPU path, and never lets a delegate failure crash the app. This is the pattern I ship.

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.content.res.AssetManager

class Classifier(assets: AssetManager, modelPath: String) {

    private var gpuDelegate: GpuDelegate? = null
    private val interpreter: Interpreter
    val backend: String   // for telemetry: which rung of the ladder we landed on

    init {
        val model = loadModelFile(assets, modelPath)
        val (interp, name) = buildInterpreter(model)
        interpreter = interp
        backend = name
    }

    // Try the accelerator ladder: GPU -> CPU(XNNPACK). The vendor/NPU rung
    // would slot in above GPU as a third try with the same try/catch shape.
    private fun buildInterpreter(model: MappedByteBuffer): Pair<Interpreter, String> {
        val compat = CompatibilityList()
        if (compat.isDelegateSupportedOnThisDevice) {
            try {
                val opts = Interpreter.Options().apply {
                    gpuDelegate = GpuDelegate(compat.bestOptionsForThisDevice)
                    addDelegate(gpuDelegate)
                }
                return Interpreter(model, opts) to "gpu"
            } catch (e: Exception) {
                // Delegate init can throw on driver quirks. Clean up and fall down.
                gpuDelegate?.close(); gpuDelegate = null
            }
        }
        // CPU floor: XNNPACK is on by default; pin threads to the big cores.
        val cpuOpts = Interpreter.Options().apply {
            numThreads = 4
            setUseXNNPACK(true)
        }
        return Interpreter(model, cpuOpts) to "cpu-xnnpack"
    }

    fun run(input: ByteBuffer, output: Array<FloatArray>) {
        interpreter.run(input, output)
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    private fun loadModelFile(assets: AssetManager, path: String): MappedByteBuffer {
        val fd = assets.openFd(path)
        val input = java.io.FileInputStream(fd.fileDescriptor)
        return input.channel.map(
            FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
        )
    }
}
```

A few things in there are load-bearing and worth calling out. The `CompatibilityList().isDelegateSupportedOnThisDevice` check is how you avoid even *trying* the GPU delegate on a device whose driver is known-broken — Google maintains a denylist. The `try/catch` around delegate construction is not optional: GPU and vendor delegates can throw at construction time on specific driver versions, and an uncaught exception there is a crash on launch for that whole device population. We record which rung we landed on in `backend` so that field telemetry can tell us, across millions of devices, what fraction actually got the accelerator versus fell to CPU. And we `close()` everything — delegates hold native resources (GPU contexts, file descriptors) that leak if you forget.

The model is memory-mapped from the APK's assets directory, not copied into a byte array. Memory-mapping means the OS pages the flatbuffer in on demand and shares it across the process without doubling RAM, which matters when the model is tens of megabytes.

### The C++ / JNI path, and when you need it

The Kotlin API is enough for most apps. You drop to the C++ API (called over JNI from Kotlin) when you need to (a) share one inference implementation across Android and iOS and other platforms, (b) avoid the JNI marshaling cost on a very tight per-frame budget, or (c) use a runtime feature only exposed in C++. The shape is the same: build an `Interpreter`, add delegates, allocate tensors, copy input, `Invoke()`, read output. The trap people hit is doing the input copy *in Kotlin*, marshaling a big `ByteBuffer` across JNI per frame; if you are processing camera frames at 30 fps, do the preprocessing and the tensor copy on the C++ side from the raw frame buffer to avoid that overhead.

### The ONNX Runtime Mobile path, for comparison

If your model lives in ONNX rather than TFLite, the ONNX Runtime Mobile path is structurally identical — you build a session, register execution providers (the ORT name for delegates) in priority order, and run. The same fallback discipline applies: list the accelerator EPs first, and ORT will assign whatever subgraph each can run, leaving the rest on the CPU EP, which is always present. The Kotlin/Java binding looks like this:

```kotlin
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer

class OrtClassifier(modelBytes: ByteArray) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val backend: String

    init {
        val opts = OrtSession.SessionOptions()
        backend = try {
            // Top rung: the Qualcomm QNN EP reaches the Hexagon NPU directly
            // (the post-NNAPI way to use a vendor accelerator).
            opts.addQnn(mapOf("backend_path" to "libQnnHtp.so"))
            "qnn-npu"
        } catch (e: Exception) {
            try {
                opts.addXnnpack(mapOf("intra_op_num_threads" to "4"))  // CPU floor
                "xnnpack-cpu"
            } catch (e2: Exception) { "cpu-default" }
        }
        session = env.createSession(modelBytes, opts)
    }

    fun run(input: FloatArray, shape: LongArray): FloatArray {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { t ->
            session.run(mapOf("image" to t)).use { res ->
                @Suppress("UNCHECKED_CAST")
                return (res[0].value as Array<FloatArray>)[0]
            }
        }
    }
}
```

Notice the same shape as the LiteRT code: a try/catch ladder that records the rung it landed on, with the CPU EP as the floor that cannot fail. The two runtimes differ in model format and EP/delegate names, not in the deployment discipline. ExecuTorch's Android path is the same again — `Module.load(...)` on a `.pte`, with the backend chosen at *lowering* time (the partitioner you ran offline) rather than at session-creation time, which moves the fallback decision earlier but doesn't remove the need for it.

### Quantizing for the mobile target with TFLite

A note that ties the conversion work to deployment: the model you load above is presumably int8, and *how* you quantized it determines whether the NPU will accept it. Full-integer (int8) quantization with a representative dataset is what most mobile NPUs want — they are integer engines — and the LiteRT converter produces it like this:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for img in calibration_images(n=200):     # ~100-500 real samples
        yield [img.astype("float32")]          # shape (1, 224, 224, 3)

converter.representative_dataset = representative_dataset
# Force a fully-integer model so the NPU can take the whole graph.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

with open("classifier_int8.tflite", "wb") as f:
    f.write(converter.convert())
```

The `inference_input_type = tf.int8` line matters for deployment specifically: it makes the *input tensor* int8, so the app feeds quantized bytes and the NPU never has to do an fp→int cast at the boundary. If you leave the input float (a common default), the runtime inserts a quantize op at the very front of the graph that may itself fall back to the CPU and force an extra copy — exactly the boundary-crossing cost from the science section, paid on every single frame. This is the kind of detail where the quantization choice ([post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq)) and the deployment outcome are the same decision viewed from two ends.

### App-size impact on Android

Two things grow your app: the **runtime** and the **model**.

The LiteRT runtime is a set of `.so` files, one per ABI (`arm64-v8a` is the one that matters; `armeabi-v7a` only if you still support 32-bit). The core interpreter plus XNNPACK is roughly **1–3 MB per ABI**. The GPU delegate adds maybe **another 1–2 MB**. If you ship multiple ABIs in one APK you multiply that, which is why **Android App Bundles** (`.aab`) and per-ABI splits matter: Play delivers only the `arm64-v8a` slice to a 64-bit device instead of fattening every install with code it can't run. ONNX Runtime Mobile and ExecuTorch have similar footprints, with ORT's "mobile" build being deliberately stripped of unused ops to shrink the `.so`.

The model itself is whatever your quantized flatbuffer weighs — for a MobileNet-class classifier, single-digit MB after int8 quantization; for a small on-device LLM, hundreds of MB to a couple of GB, which is firmly OTA-download territory (next section). The headline: budget the *runtime* `.so` separately from the *model* asset, because the runtime cost is fixed per app while the model cost scales with how big a model you are trying to run.

## iOS end to end

iOS is the easier platform to ship on and the harder platform to control. Easier because the hardware is uniform — there are only a handful of current iPhone SoCs, all with an Apple Neural Engine (ANE), all running the same Core ML runtime. Harder because Core ML makes the scheduling decisions *for* you: you express a preference for which compute units it may use, and Core ML decides, per operator, whether each one runs on the CPU, the GPU, or the ANE. You do not get to pin an op to the ANE; you get to ask nicely. The figure below shows the stack: your Swift app calls Core ML, Core ML builds a per-op compute-unit plan, and most ops ideally land on the Neural Engine while unsupported ops fall back to CPU or GPU.

![A five-layer iOS Core ML stack showing the Swift app on top, then the Core ML runtime, then the compute-unit plan, then the Apple Neural Engine, with a CPU and GPU fallback layer for unsupported operators](/imgs/blogs/mobile-deployment-end-to-end-3.png)

### Converting to Core ML with coremltools

You ship a Core ML model as an `.mlpackage` (the modern container; the older `.mlmodel` still works but `.mlpackage` is what you want for anything recent). You produce it with `coremltools`, converting from PyTorch (via `torch.export` or a traced/scripted model) or from a TensorFlow/ONNX source. Here is a real conversion from a traced PyTorch image model, the path I use most:

```python
import torch
import coremltools as ct

# 1. Get a TorchScript trace of the eval model with a concrete input shape.
model = build_model().eval()                 # your nn.Module, weights loaded
example = torch.rand(1, 3, 224, 224)         # batch=1 is the on-device reality
traced = torch.jit.trace(model, example)

# 2. Convert to an .mlpackage. Declare the input as an image so Core ML can
#    accept a CVPixelBuffer directly and do the normalization on-device.
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image",
        shape=(1, 3, 224, 224),
        scale=1/255.0,                       # must match training preprocessing
        bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        color_layout=ct.colorlayout.RGB,
    )],
    outputs=[ct.TensorType(name="logits")],
    compute_units=ct.ComputeUnit.ALL,        # let Core ML use CPU+GPU+ANE
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,  # fp16 is the ANE's native precision
)

# 3. Optionally palettize / quantize weights to shrink the package further.
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights)
config = OptimizationConfig(
    global_config=OpPalettizerConfig(mode="kmeans", nbits=4))
mlmodel = palettize_weights(mlmodel, config)

mlmodel.save("Classifier.mlpackage")
```

The details that bite people: the `scale` and `bias` on the `ImageType` *are your preprocessing*, and they must exactly match the normalization you trained with — this is the iOS face of the preprocessing-parity problem we will hit again below. Declaring the input as an `ImageType` rather than a raw `TensorType` lets you feed Core ML a `CVPixelBuffer` straight from the camera, with the normalization baked into the model graph and (importantly) running on-device on the accelerator instead of in Swift on the CPU. `compute_precision=FLOAT16` matters because the ANE is a fp16 engine; leaving the model fp32 forces casts. And `palettize_weights` with 4-bit k-means is the Core ML analogue of weight quantization — it shrinks the package by clustering weights into a small codebook, which on a large model is the difference between bundling and OTA.

### Compute-unit selection in Swift

At runtime you load the compiled model and choose a compute-unit policy. The choice is not cosmetic: it changes which silicon runs your ops, and therefore both latency and power.

```swift
import CoreML
import Vision

final class Classifier {
    private let model: VNCoreMLModel
    let computeUnits: MLComputeUnits

    init() throws {
        let config = MLModelConfiguration()
        // .all lets Core ML use the ANE; .cpuAndNeuralEngine excludes the GPU
        // (often the lowest-power choice); .cpuOnly is the deterministic floor.
        config.computeUnits = .all
        self.computeUnits = config.computeUnits

        let coreMLModel = try Classifier_mlpackage(configuration: config).model
        self.model = try VNCoreMLModel(for: coreMLModel)
    }

    func classify(pixelBuffer: CVPixelBuffer,
                  completion: @escaping (String, Float) -> Void) {
        let request = VNCoreMLRequest(model: model) { req, _ in
            guard let best = (req.results as? [VNClassificationObservation])?.first
            else { return }
            completion(best.identifier, best.confidence)
        }
        request.imageCropAndScaleOption = .centerCrop   // resize policy = training
        // Run OFF the main thread; the handler does the heavy work.
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
            try? handler.perform([request])
        }
    }
}
```

The `MLComputeUnits` options are the iOS equivalent of the delegate ladder, except Core ML walks it internally:

- **`.all`** — Core ML may use CPU, GPU, and ANE, choosing per op. This is the default and usually the fastest.
- **`.cpuAndNeuralEngine`** — excludes the GPU. Often the *lowest power* choice for sustained workloads, because the GPU is power-hungry and contends with rendering; if your op set runs well on the ANE, this can beat `.all` on battery and on thermal headroom while matching it on latency.
- **`.cpuOnly`** — deterministic, slowest, what you use for reproducibility tests and as a guaranteed floor.

There is no "ANE only" option — the CPU is always in the set as the fallback for ops the ANE can't run. Which ops actually landed on the ANE is something you discover by profiling in Xcode's Core ML Instruments template, not something you declare.

### ExecuTorch with the Core ML backend

If your stack is PyTorch and you want one path across both mobile platforms, ExecuTorch can target Core ML as a *backend*: you `torch.export` once, then lower to ExecuTorch with the Core ML partitioner for iOS and the XNNPACK/Qualcomm partitioner for Android, shipping a `.pte` on each. This keeps a single source model and a single runtime API while still reaching the ANE on iOS and the NPU on Android. The trade-off is maturity and op coverage — Core ML's native `coremltools` path has the broadest ANE op support today — so I reach for ExecuTorch+CoreML when the PyTorch-everywhere consistency is worth more than squeezing the last op onto the ANE, and for plain `coremltools` when I want maximum ANE coverage on iOS specifically.

### App thinning on iOS

iOS handles per-device slicing for you via **app thinning**: the App Store delivers a variant with only the resources and slices that device needs, so you do not pay the full universal-binary cost on every install. For models specifically, the lever you control is the `.mlpackage` weight — fp16 plus palettization (shown above) is how you keep a bundled model small. For genuinely large models you use **on-demand resources** or your own OTA download, the same bundle-vs-download decision as Android.

## The science: backend selection, fallback, and the cost of crossing boundaries

Now the *why*. Mobile inference performance is dominated by two things people underestimate: the cost of *operator fallback* across a backend boundary, and the cost of the *first* inference. Both have a clean enough model that you can reason about them quantitatively instead of just measuring and shrugging.

### Why a fallback op can cost more than the op it "replaced"

When a delegate (or Core ML compute unit) does not support an operator, the runtime splits the graph at that op. Suppose your model is a chain and op $k$ is unsupported on the accelerator. The accelerator runs ops $1\ldots k-1$, then the activation tensor of size $S$ bytes must be copied from accelerator memory to CPU memory, op $k$ runs on the CPU, and (if more delegated ops follow) the result is copied back. The total cost of "running op $k$ on the CPU" is not the CPU compute of op $k$ alone; it is

$$
T_{\text{fallback}} = \underbrace{\frac{S}{B_{\text{copy}}}}_{\text{copy out}} + \underbrace{T_{\text{cpu}}(k)}_{\text{compute}} + \underbrace{\frac{S'}{B_{\text{copy}}}}_{\text{copy back}}
$$

where $B_{\text{copy}}$ is the achievable copy bandwidth across the boundary and $S, S'$ are the input and output activation sizes of op $k$. The compute term $T_{\text{cpu}}(k)$ might be tiny — a single `Softmax` or `Reshape` is almost free. But the two copy terms are governed by memory bandwidth, not compute, and they are paid *whether or not op $k$ is cheap*. For a large activation tensor — say an early conv feature map of a few megabytes — those copies can dwarf the op. This is why one unsupported op in the *middle* of an otherwise-delegated model is so much worse than the same op at the very end: in the middle you pay two copies and you break the accelerator's ability to keep the whole subgraph resident.

The practical corollary, which connects straight to the hardware reasoning in [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape): minimize the *number of partitions*, not just the number of unsupported ops. A model that is fully delegatable (one partition) beats a model that is 99% delegatable but cut into three islands. When you find a fallback, your first move is to replace or remove the offending op so the graph fuses back into one partition — not to micro-optimize the CPU kernel for that op.

### Why the first inference is slow: the compile tax

The second cost is **cold start**. The first time you run a model on the GPU or NPU, the runtime does work it never repeats: it compiles shaders or kernels for your specific op shapes, allocates and lays out tensor buffers on the accelerator, and warms instruction caches. Model this as a fixed one-time cost $C$ added to the first inference:

$$
T_1 = C + T_\infty, \qquad T_n = T_\infty \ \text{ for } n > 1
$$

where $T_\infty$ is the steady-state per-inference latency. On a mobile GPU, $C$ (shader compilation) is commonly tens of milliseconds; on an NPU or the ANE, $C$ (graph compilation plus weight loading) can be **hundreds of milliseconds**. If the user's *first* tap triggers the *first* inference, they feel that $C$ as a freeze — and first impressions of "this feature is slow" are made on exactly that tap.

The fix is structural, not a micro-optimization: **run a dummy inference at load time, off the UI thread, before the user can trigger a real one.** That dummy pays $C$ on a background thread while the user is still reading the screen; by the time they tap, the runtime is warm and they get $T_\infty$. We will quantify this with a worked example below, and the before-after figure later in the post is exactly this transformation.

It's worth seeing *why* the warm-up wins so decisively, in numbers. Without a warm-up, the average latency a user experiences over their first $n$ inferences is the amortized cost

$$
\bar{T}(n) = \frac{C + n\,T_\infty}{n} = T_\infty + \frac{C}{n}.
$$

For a short interaction — a user who taps once, sees the result, and leaves — $n$ is small, so the $C/n$ term dominates: at $n=1$ the user experiences the *full* $C + T_\infty$, which for the ANE example below is roughly $300 + 7 = 307$ ms instead of 7 ms. The compile tax is not a small constant overhead amortized away over a long session; for the most common interaction pattern (a single tap) it *is* the latency. That is precisely why moving $C$ off the user path matters more than any steady-state optimization: optimizing $T_\infty$ from 7 ms to 5 ms is invisible next to a 300 ms cold start, and the only way to make the 300 ms invisible is to have already paid it before the user arrives.

### Threading: never block the UI thread

This one is not subtle but it is the most common bug I see. A neural inference of even 10 ms is *forever* on the UI thread — at 60 fps you have 16.7 ms total to do everything including rendering, so a 10 ms synchronous inference on the main thread will drop frames and feel janky even though 10 ms "sounds fast." On Android, run inference on a background `Executor`/coroutine `Dispatchers.Default` and post results back to the main thread. On iOS, dispatch to a `userInitiated` global queue as in the Swift snippet above and hop back to the main queue for the UI update. For camera-driven inference, also make sure you are not queueing frames faster than you can process them — drop frames (process the latest, discard the backlog) rather than building an ever-growing queue that adds latency and memory pressure.

### Memory and thermals under sustained use

A single inference's peak memory is the model weights (resident, shared via mmap) plus the largest activation working set plus the runtime's scratch buffers. That is usually fine. The problem is *sustained* use: a camera feature running inference every frame for two minutes keeps the accelerator busy, the SoC heats up, and the OS throttles clocks. Throttled, your $T_\infty$ creeps up — I have measured 30–50% latency growth on a phone after a couple of minutes of continuous NPU use as it goes from cool to thermally limited. This is why your benchmark must include a *sustained* run, not just a warm-but-cool burst, and why `.cpuAndNeuralEngine` (avoiding the power-hungry GPU) is sometimes the right call for a long-running feature even when `.all` is marginally faster cold.

The backend fallback ladder, made into a decision the runtime walks at init time, looks like the tree below: try the fast accelerator, accept it if it initializes and keeps the graph whole, otherwise degrade down to a CPU path that cannot fail.

![A decision tree for backend selection showing an init request that tries the NPU or ANE first, then the GPU, accepting whichever initializes, and finally landing on a CPU XNNPACK path that always works](/imgs/blogs/mobile-deployment-end-to-end-4.png)

## Preprocessing parity: the silent accuracy killer

Here is a failure that does not show up in any latency number and is therefore the most dangerous: your on-device preprocessing does not match your training preprocessing, so the model sees inputs from a slightly different distribution than it was trained on, and accuracy quietly degrades — not catastrophically, just enough that the field metrics are a few points below the lab and nobody can explain why.

Preprocessing for an image model is typically: decode → resize to the model's input size → convert color space → normalize (subtract mean, divide by std). Every one of those steps has degrees of freedom that differ between PyTorch's `torchvision.transforms` on a server and the device's image pipeline:

- **Resize algorithm.** `torchvision` defaults to bilinear with a specific anti-aliasing behavior. Android's `Bitmap.createScaledBitmap` with `filter=true` is bilinear but *not* anti-aliased the same way; iOS's `VNImageRequestHandler` resize and `imageCropAndScaleOption` have their own behavior. Bilinear-with-antialias vs bilinear-without can shift pixel values enough to move predictions on borderline inputs.
- **Crop vs squash.** Did you train on center-crop-then-resize, or resize-the-whole-image (squash)? If training center-cropped and the device squashes, the aspect ratio is wrong.
- **Color order.** RGB vs BGR. A model trained on RGB fed BGR is not subtly wrong, it is badly wrong — but the bug looks like "the model is bad" rather than "the channels are swapped."
- **Normalization constants.** The mean/std must be the exact training values, applied in the exact order, on the exact value range (0–1 vs 0–255).

The fix has two parts. First, **push as much preprocessing as possible into the model graph** — that is exactly what the Core ML `ImageType` `scale`/`bias` does, and what a TFLite model with a normalization layer baked in does. If the normalization is *in the model*, it cannot drift between platforms. Second, for the resize and crop that must happen outside the model, write a **golden-image test**: take a handful of fixed input images, run the *exact* device preprocessing path, and assert the resulting tensors match the training preprocessing within a tight tolerance (a few least-significant bits, not "close enough"). Here is the shape of that test, run as part of CI before any build ships:

```python
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# The TRAINING preprocessing — the source of truth.
train_tf = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),                       # -> [0,1], CHW, RGB
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def golden_check(img_path, device_tensor_npy):
    """device_tensor_npy: the tensor produced by the on-device preproc path,
    exported from an instrumented build for the same input image."""
    ref = train_tf(Image.open(img_path).convert("RGB")).numpy()
    dev = np.load(device_tensor_npy)
    max_abs = np.abs(ref - dev).max()
    mean_abs = np.abs(ref - dev).mean()
    print(f"max |ref-dev| = {max_abs:.4f}   mean = {mean_abs:.5f}")
    # A correct path matches to a few LSBs; a wrong resize/colorspace blows up.
    assert max_abs < 0.02, "preprocessing parity FAILED — check resize/colorspace/norm"

for p in ["golden/cat.png", "golden/dog.png", "golden/car.png"]:
    golden_check(p, p.replace(".png", "_device.npy"))
```

When this test fails it tells you *which knob* is off: a uniform offset across all pixels is a normalization mismatch; a checkerboard of small differences is a resize-algorithm mismatch; a per-channel swap is RGB/BGR. I have caught all three with this test, and every one of them would otherwise have shipped as a mysterious accuracy regression. This is the kind of cheap verification step the series keeps coming back to — and it lives squarely in the deployment lifecycle ([the edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle)) at the convert/package boundary.

## Packaging and ops: bundle vs OTA, versioning, telemetry

You have a working model and a working call path. Now: how does the model get *onto* the device, and how do you operate it over time?

### Bundle the model, or download it?

The first decision is whether the model ships *inside* the app binary (a bundled asset) or is downloaded over the air (OTA) on first run or on demand. The figure below lays out the trade-off: bundling makes the app work instantly and offline but inflates the install; OTA keeps the install slim but pays a first-run download and requires you to handle the failure case where the download doesn't complete.

![A before-after comparison of bundling the model as an app asset versus downloading it over the air, showing bundling inflates install size but works offline while OTA keeps the install slim at the cost of a first-run download](/imgs/blogs/mobile-deployment-end-to-end-5.png)

The size budgets that force this decision are real and worth memorizing. On the **Play Store**, an Android App Bundle can serve up to a large compressed size, but the number that actually changes user behavior is the **cellular auto-download threshold** — historically around 200 MB, above which Play warns users and may defer the download to Wi-Fi — and the perception threshold, where every extra ~10 MB of install size measurably reduces install conversion. On the **App Store**, apps over the **cellular download limit** (Apple raised it over time; currently in the hundreds of MB) prompt the "download over Wi-Fi" dialog. The practical rule I use:

- **Model under ~30 MB after compression:** bundle it. The size hit is acceptable, you get instant offline functionality, and you avoid the entire OTA failure-handling burden.
- **Model 30–150 MB:** judgment call. If the feature is core to the app and used on first launch, lean bundle; if it's a secondary feature or you want to iterate the model independently of app releases, lean OTA.
- **Model over ~150 MB (e.g. an on-device LLM):** OTA, full stop. You cannot ship a 1.5 GB app, and you want to update the model without an app-store review cycle.

OTA buys you something beyond size: you can **update the model without shipping an app build**. An app release takes days (build, QA, store review, staged rollout) and reaches users on their own update schedule, which can be weeks. An OTA model update can reach all users in hours. That is a real operational advantage for a model you expect to iterate — but it comes with obligations: you must version the model, verify its integrity (checksum/signature) after download, handle the partial-download and offline cases gracefully, and make sure a bad model push can be rolled back fast. A bundled model can never be worse than the build it shipped in; an OTA model can take down your feature for everyone in one bad push, so OTA *needs* the telemetry and rollback machinery that bundling can skip.

### Versioning the model with the app

Whichever you choose, **the model and the app code that calls it are a single compatible unit** and must be versioned as such. The classic bug: you OTA-push a new model whose output tensor changed shape or whose label list grew, and the old app code that's still installed on most devices reads it wrong. Defend against this by stamping a **schema/contract version** into both the app and the model metadata, and refusing to load a model whose contract version the app doesn't understand (fall back to the bundled baseline model instead). For bundled models the contract is fixed at build time and this is automatic; for OTA it is a hard requirement.

### On-device A/B and telemetry

You cannot improve what you cannot see, and on-device you are blind by default. The minimum telemetry I ship with any on-device model:

- **Which backend each device landed on** (the `backend` string from the Kotlin snippet; the realized compute units on iOS). This tells you what fraction of your users actually got the accelerator versus fell to CPU — often a humbling number.
- **Latency distribution, p50 and p99**, bucketed by device model and backend. The p99 is where the thermal throttling and the cold-start outliers show up.
- **Cold-start time** (the first-inference $C$) separately from steady-state.
- **Crash/exception rate** at delegate init, broken down by device — this is how you discover that a specific vendor driver version is throwing and add it to your denylist.
- **For OTA models, the model version** every event is tagged with, so an A/B test or a rollback is a query, not a guess.

Concretely, the per-inference event I emit (sampled — you do not log every inference, you log a fraction to keep the data volume sane) is small and flat, something like this:

```json
{
  "event": "inference",
  "model_version": "classifier-v7",
  "backend": "gpu",
  "device_model": "Pixel 7a",
  "os": "android-14",
  "latency_ms": 14.2,
  "cold_start": false,
  "thermal_state": "nominal",
  "canary_ok": true
}
```

With a few million of those a week you can answer the questions that actually decide whether the feature is healthy: *what percentage of sessions got the accelerator?* (group by `backend`); *did the v7 model regress the tail on mid-range devices?* (p99 of `latency_ms` grouped by `model_version` and `device_model`); *is any device class throwing at init?* (a separate `init_failed` event with the exception class); *did a bad OTA push break correctness somewhere?* (`canary_ok == false` grouped by `device_model`). The `canary_ok` field is the on-device correctness check from the stress-test section, reported from the field — it's how you'd catch a silently-wrong accelerator on a device you never owned. Without this telemetry, every one of these questions is answered with a shrug and a hope; with it, a regression is a dashboard alert and a one-query rollback.

This is the on-ramp to a real edge MLOps practice — staged model rollouts, automatic rollback on a p99 or accuracy-proxy regression, shadow evaluation — which is a topic of its own and the natural next read after this one in the series.

## Measuring on-device latency and memory honestly

A latency number is worthless without its measurement protocol, and most reported mobile numbers are quietly dishonest in one of a few specific ways. Here is how to measure so the number means what you think it means, with code.

The non-negotiables:

1. **Warm up first.** Run several inferences and discard them before you start timing, so your measurement reflects $T_\infty$, not the cold-start $C$. (Report $C$ separately — it matters — but don't let it pollute your steady-state number.)
2. **Batch=1.** On-device inference is almost always one input at a time (one camera frame, one user query). Batch>1 throughput numbers are a server fiction here.
3. **Many iterations, report a distribution.** Report p50 *and* p99, not a mean. The mean hides the tail; the tail is what users feel.
4. **Measure on a named device, in a stated thermal state.** "Pixel 8, cool start" and "Pixel 8, after 2 min sustained" are different machines. Say which.
5. **Wall-clock the actual inference call**, not the wrapper, and pin to performance cores if your platform lets you (to reduce variance from the scheduler bouncing you onto a little core).

Here is the Kotlin benchmark I run on every device, wired to the `Classifier` above:

```kotlin
fun benchmark(clf: Classifier, makeInput: () -> ByteBuffer, iters: Int = 200): String {
    val out = Array(1) { FloatArray(1000) }

    // 1. Warm-up: pay the cold-start C here, OUTSIDE the timed region.
    repeat(20) { clf.run(makeInput(), out) }

    // 2. Timed steady-state runs, batch=1, recording each latency.
    val samples = LongArray(iters)
    for (i in 0 until iters) {
        val input = makeInput()
        val t0 = System.nanoTime()
        clf.run(input, out)
        samples[i] = System.nanoTime() - t0
    }
    samples.sort()
    fun pct(p: Double) = samples[(p * (iters - 1)).toInt()] / 1_000_000.0
    return "backend=${clf.backend}  p50=%.1fms  p90=%.1fms  p99=%.1fms"
        .format(pct(0.50), pct(0.90), pct(0.99))
}
```

For **memory**, do not trust a single `Runtime.getRuntime()` snapshot — it lies because of GC timing and because native (off-heap) allocations from the delegate don't show in the Java heap. Use Android Studio's Memory Profiler or `adb shell dumpsys meminfo <package>` to read the **PSS** (proportional set size) of the process before model load, after model load, and during inference; the deltas are your model-resident footprint and your inference working set. On iOS, use Xcode Instruments' Allocations and the Core ML template, which also tells you the per-op compute-unit assignment so you can see what actually ran on the ANE. The honest memory number is the *peak resident during a sustained inference loop*, not the idle figure.

There is a cold-start cost hiding in every one of those benchmarks, and the right way to handle it is to move it off the user's path with a warm-up at load time. The before-after figure makes the transformation concrete: without a warm-up the user's first tap pays the full compile tax as a visible stall; with a background warm-up at load, the first tap is already at steady-state latency.

![A before-after comparison showing that without warm-up the first user tap stalls for hundreds of milliseconds compiling the graph, while a background warm-up at load time makes the first tap run at steady-state latency](/imgs/blogs/mobile-deployment-end-to-end-6.png)

## Worked example: Android CPU vs GPU vs NPU on the same model

#### Worked example: one MobileNetV3-Large across three Android backends

Take MobileNetV3-Large (the running classifier from earlier in this series — see [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) for why it's a good edge baseline), quantized to int8, as a `.tflite` flatbuffer of about **6 MB**. Run the benchmark above on a flagship-class Android phone (a Pixel-8-class device with a Tensor-G3-class NPU), batch=1, warmed up, cool start. Approximate, defensible numbers for a model of this class:

| Backend | p50 latency | p99 latency | Cold-start C | Ops delegated | Notes |
|---|---|---|---|---|---|
| CPU (XNNPACK, 4 threads) | ~25 ms | ~34 ms | ~5 ms | all (CPU) | Always works; the floor. |
| GPU delegate (fp16) | ~14 ms | ~20 ms | ~80 ms | ~all | Pays shader compile once; copy cost across CPU↔GPU. |
| NPU / vendor delegate | ~9 ms | ~16 ms | ~380 ms | most; a few fell to CPU | Fastest, lowest power; one `hardswish` config and the final reshape fell back. |

Read the story in the numbers. The CPU path is the slowest but has a negligible cold-start and never fails — it is why you always keep it as the floor. The GPU nearly doubles throughput but pays an ~80 ms shader compile on the first run (hence: warm up) and a per-frame copy crossing the CPU↔GPU boundary that eats some of the win. The NPU is fastest at steady state and the most power-efficient, but it pays a **~380 ms** graph-compile cold start, and a couple of ops (a particular `hardswish` configuration, the trailing `reshape`) were not supported and fell back to CPU — those fallbacks are why its p99 gap to p50 is wider than you'd hope and why it isn't 3× faster than CPU despite the NPU's raw TOPS suggesting it should be.

The decision: ship the **NPU→GPU→CPU ladder** from the Kotlin snippet, warm up at load to bury the 380 ms, and — if telemetry shows the NPU op-fallback is costing too much — go fix the two offending ops in the model so the graph stays one partition, rather than accepting the fragmented version. The Pareto point here is real: for a 6 MB model you move p50 from 25 ms (CPU) to 9 ms (NPU), a 2.8× latency win at zero accuracy cost (same int8 weights), bounded only by getting the cold start off the user path and the op coverage clean.

## Worked example: iOS CPU vs ANE, and the cold-start tax

#### Worked example: the same classifier on iPhone, CPU vs Neural Engine

Convert that same int8/fp16 model to an `.mlpackage` with `coremltools` (the conversion command above), and run it on a recent iPhone (an A17-class SoC) with two compute-unit policies, warmed up, batch=1. Approximate numbers:

| Compute units | p50 latency | p99 latency | Cold-start C | Power | Notes |
|---|---|---|---|---|---|
| `.cpuOnly` | ~22 ms | ~30 ms | ~10 ms | high (cores busy) | Deterministic floor; reproducibility baseline. |
| `.all` (CPU+GPU+ANE) | ~7 ms | ~13 ms | ~300 ms | low at steady state | ANE runs most ops; GPU used for a few. |
| `.cpuAndNeuralEngine` | ~8 ms | ~13 ms | ~300 ms | lowest sustained | Matches `.all` on latency, beats it on battery/thermals. |

Three lessons fall straight out. First, the ANE is roughly **3× faster than the CPU** at steady state for this model and far more power-efficient — that's the whole reason it exists. Second, the cold-start tax is **~300 ms**, dominated by Core ML compiling the model for the ANE and loading weights, and it is paid on the *first* prediction; if a user's first tap triggers it, that's a 300 ms freeze, so you warm up at load exactly as on Android (run one prediction on a background queue when the view appears). Third — the non-obvious one — **`.cpuAndNeuralEngine` matched `.all` on latency while using less power**, because for this op set the GPU wasn't buying speed but was burning watts and contending with rendering. For a sustained camera feature I would ship `.cpuAndNeuralEngine` here; for a one-shot occasional inference I'd ship `.all` and not think about it. You only learn which by profiling in the Core ML Instruments template and reading the per-op compute-unit assignment — Core ML does not promise the ANE, it *chooses*, and the choice is visible only after the fact.

## Stress-testing the deployment: when it all goes wrong

The worked examples above are the happy path on flagship phones. The job is to know what happens at the edges of the device population, because that is where the field crashes and the one-star reviews come from. Let me reason through the failure modes deliberately, the way you would in a launch review.

**What happens on a device with no usable NPU?** A budget phone with no vendor delegate and a GPU whose driver is on the denylist falls straight to the CPU floor. With the ladder in place that's not a crash, it's a slower inference — ~25 ms instead of ~9 ms in our example. The decision this forces is upstream: *is the CPU number within budget?* If your frame budget is 33 ms (30 fps) and CPU is 25 ms, you ship and let those devices run on CPU. If CPU is 90 ms because the model is too big, then no amount of delegate cleverness saves you on those phones — the answer is a *smaller model* for the low-end tier (a distilled or more-aggressively-quantized variant, selected at download time by device class), not a faster runtime. The runtime can't make a model that doesn't fit the device fit it.

**What happens when the NPU accepts the model but silently runs it wrong?** This is the scary one. Some vendor NPUs have had bugs where a quantized op produces subtly wrong outputs — not a crash, not a fallback, just wrong numbers — on specific driver versions. You catch this only with an **on-device correctness check**: at init, run a fixed canary input through the model and compare the output against a precomputed golden output (computed once on the CPU reference path), within a tolerance. If the canary fails, *disable that backend for that device* and fall to the next rung. This canary check costs one extra inference at startup and is the only defense against a silently-wrong accelerator. I add it to the warm-up call so it's free — the warm-up inference *is* the canary.

**What happens when the model is memory-bound, not compute-bound?** For a small CNN you are compute-bound and the NPU's TOPS help. But for a decode step of an on-device LLM, or a model dominated by large embedding lookups, you are **memory-bound** — limited by how fast you can stream weights from DRAM, not by multiply-accumulate throughput (the roofline reasoning in [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives)). On a memory-bound op the NPU's compute units sit idle waiting for data, so the NPU may be *no faster than the CPU*, and the win comes instead from shrinking the weights (more aggressive quantization, so fewer bytes to stream) rather than from picking a faster compute backend. Knowing which regime you're in tells you whether to spend your effort on the backend or on the weights — and the only way to know is to profile, because the FLOP count alone won't tell you.

**What happens under sustained thermal load?** Run the benchmark for two minutes instead of two seconds and watch p50 drift up as the SoC throttles. If your feature is a one-shot tap, you never hit this. If it's a continuous camera feature, you do, and the mitigation is to *reduce duty cycle*: run inference every Nth frame instead of every frame, drop to `.cpuAndNeuralEngine` to avoid the hot GPU, or cap the inference rate so the SoC stays under its thermal ceiling. A model that's fast cold and unusable hot has not actually met its budget — and a benchmark that only measures cold will tell you it passed.

Each of these is a place where the naive "the model is optimized, just call it" assumption breaks, and each has a concrete defense that costs almost nothing if you design it in and is impossible to retrofit cleanly after a bad launch.

## Results: the platform-by-backend matrix

Pulling both platforms together, the shape of the trade-off is consistent and worth internalizing: **the fastest backend is also the least portable and pays the highest cold-start, while the most portable backend is the slowest but never fails.** You do not pick one row — you ship the ladder and let each device land where it can. The matrix below is the decision compressed to one frame.

![A matrix comparing CPU, GPU, and NPU or ANE backends across p50 latency, setup cost, and the main gotcha, showing latency improves down the rows while setup cost and fragility rise](/imgs/blogs/mobile-deployment-end-to-end-7.png)

| Platform × backend | p50 (6 MB classifier) | Setup / cold start | App-size cost | Main gotcha |
|---|---|---|---|---|
| Android CPU (XNNPACK) | ~25 ms | ~5 ms | runtime `.so` ~2 MB | thermal throttle under sustained load |
| Android GPU delegate | ~14 ms | ~80 ms shader compile | +~1–2 MB delegate `.so` | CPU↔GPU copy cost; per-device driver quirks |
| Android NPU / vendor | ~9 ms | ~380 ms graph compile | +vendor delegate `.so` | op fallback to CPU; NNAPI deprecated, use vendor delegates |
| iOS CPU only | ~22 ms | ~10 ms | runtime is in the OS | deterministic but slow; for baselines only |
| iOS ANE (`.all`) | ~7 ms | ~300 ms compile | `.mlpackage` weight only | Core ML chooses per op; ANE not guaranteed |
| iOS ANE (`.cpuAndNeuralEngine`) | ~8 ms | ~300 ms compile | `.mlpackage` weight only | best sustained power; profile to confirm op coverage |

### App-size and battery note

Two numbers to carry: the **runtime** costs you roughly 2–4 MB of `.so` on Android (LiteRT core + delegates, per ABI — use App Bundles to ship one ABI) and effectively *zero* on iOS, since Core ML is part of the OS — that asymmetry alone sometimes tips a cross-platform team toward Core ML on iOS even when they use LiteRT/ONNX everywhere. The **model** costs you its compressed weight, which is the whole bundle-vs-OTA decision above. On **battery**: the NPU/ANE is not just faster, it is *dramatically* more energy-efficient per inference than the CPU or GPU — often an order of magnitude fewer millijoules per inference — which is why for any always-on or high-frequency feature the accelerator path is about battery life as much as latency. Conversely, an inference-every-frame feature on the *CPU* will heat the phone and drain the battery fast; if your telemetry shows most devices falling to CPU, that's a battery problem, not just a latency one.

## Case studies: real shipped numbers

A few real, named results to calibrate expectations against published work and shipped products. I'll keep these to figures I can defend and flag anything approximate.

**MobileNetV3 on Pixel.** The MobileNetV3 paper (Howard et al., 2019) reported MobileNetV3-Large at roughly **~50–60 ms on a Pixel-class CPU** (single core, the paper's measurement protocol) — single-digit-millisecond numbers like my worked example come from quantization plus a GPU/NPU delegate plus multithreading, which is exactly the deployment work this post is about. The lesson: the architecture paper's latency is a *starting point*; the deployed latency is what your delegate-and-quantization pipeline makes of it. The architecture choice and the deployment pipeline are two different Pareto moves and you need both (see [EfficientNet, ShuffleNet, and the FLOPs–latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap) for why low FLOPs alone doesn't give you low latency).

**Core ML and the ANE for on-device transformers.** Apple's own engineering writeups on deploying transformers to the Neural Engine show large speedups and power reductions when the model is structured so its ops map cleanly to the ANE (specific tensor layouts, avoiding ops that force a CPU/GPU fallback). The headline isn't a single number; it's the principle that *ANE performance is contingent on op coverage* — the same fallback-cost story from the science section, on Apple silicon. Structuring the model for the accelerator is part of deployment, not a separate concern.

**On-device LLMs (Gemma / Phi class).** Small language models in the 1–4B parameter range now run on flagship phones at interactive token rates via runtimes like MLC-LLM, LiteRT's LLM stack, and ExecuTorch — typically reported in the **single-digit-to-low-tens of tokens per second** range on flagship SoCs after 4-bit quantization. These are firmly OTA-download models (hundreds of MB to low GB), they live or die on memory bandwidth (decode is memory-bound — see [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives)), and the deployment work is dominated by the bundle-vs-OTA, versioning, and warm-up machinery in this post at a much larger scale.

**DistilBERT-class encoders on mobile.** A distilled BERT (DistilBERT: ~40% smaller, ~60% faster than BERT-base while retaining ~97% of GLUE performance, per Sanh et al., 2019) quantized to int8 runs comfortably on-device for classification/embedding tasks at low-tens-of-milliseconds per short sequence. This is the well-trodden, low-risk end of on-device NLP and a good reminder that distillation and quantization compose — the deployment is the easy part once the model is small ([knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals)).

## When to bundle vs OTA, and when each backend is worth it

A decisive section, because every choice here is a cost and you should know when *not* to pay it.

**Bundle the model when** it's small (under ~30 MB), the feature is used on first launch, you need offline functionality, or you can't justify the OTA failure-handling and rollback machinery. Bundling is strictly simpler and a bundled model can never be worse than the build it shipped in. Default to bundling for small models.

**OTA the model when** it's large (over ~150 MB, certainly for on-device LLMs), you expect to iterate the model faster than your app-release cadence, or you want to run on-device A/B tests across model versions. Pay the cost — versioning, integrity checks, partial-download handling, fast rollback — only because you're getting the iteration speed and size relief in return.

**Use the NPU/ANE when** the feature is high-frequency or always-on (the battery efficiency dominates), or when you need the lowest latency and your op set maps cleanly to the accelerator. Don't *only* use it — always keep the CPU floor — and don't assume it; profile to confirm op coverage and watch the cold start.

**Use the GPU delegate when** you want a portable accelerator across many Android devices without per-vendor delegate integration, and your model is fp16-friendly. It's the pragmatic middle: most of the NPU's speed, far more of the device coverage, but watch the shader-compile cold start and the CPU↔GPU copy cost.

**Stay on CPU (XNNPACK) when** the model is small enough that CPU latency is already under budget, when you need bit-deterministic output, or as the universal fallback floor that every device gets. For a tiny model, the accelerator's cold-start and copy costs can make CPU the *better* choice end to end — measure before you reach for the NPU.

**Don't** build new code around NNAPI (deprecated as of Android 15) — use the GPU delegate as your portable accelerator and vendor delegates for the top-end NPU rung. **Don't** ship without a warm-up if your cold start is more than a few tens of milliseconds. **Don't** trust a notebook latency number for anything; the device is a different machine.

A useful way to make these decisions reproducible across a team is to write them down as a tiered policy keyed on model size and feature type, rather than re-litigating them per feature. For us that policy reads roughly: models under 30 MB bundle and run on the NPU→GPU→CPU ladder; models 30–150 MB bundle if first-launch-critical, else OTA; models over 150 MB always OTA with versioning, integrity, and rollback; always-on or high-frequency features prefer `.cpuAndNeuralEngine` on iOS and the vendor/NPU rung on Android for battery; low-end device tiers below a latency budget get a smaller distilled model variant selected at download time rather than the same model on a slower backend. The point of writing it down is that the *first* time you reason through bundle-vs-OTA or which-backend you do the full analysis above, and every time after you check the policy and only re-derive if the feature genuinely doesn't fit a tier. That is how an organization ships a tenth on-device model in a week instead of a quarter — the deployment reasoning becomes a checklist, not a research project, which is exactly the leverage the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) is built to give you across every lever in this series.

## The shippable pre-flight checklist

Before a build with an on-device model goes anywhere near a store, it passes a fixed set of independent checks. None of them is optional, and any one failing blocks the ship. The figure below is the gate: five checks all feed one decision, and the build ships only when every one is green.

![A graph showing five pre-flight checks — delegate fallback, warm-up, preprocessing parity, size budget, and telemetry — all converging into a single ship gate that passes only when all are green](/imgs/blogs/mobile-deployment-end-to-end-8.png)

The checklist, expanded:

1. **Delegate / compute-unit fallback works.** The NPU→GPU→CPU ladder degrades cleanly; force-disable the accelerator (simulate the denylist case) and confirm the app still runs on the CPU floor without crashing. Telemetry records which rung each device landed on.
2. **Warm-up is wired.** A dummy inference runs at load time on a background thread so the user's first real inference is at steady-state latency, not cold. Confirm the cold-start $C$ is paid off the UI thread.
3. **Preprocessing parity holds.** The golden-image test passes — on-device tensors match training preprocessing to a few LSBs — and as much normalization as possible is baked into the model graph. This guards accuracy, which no latency test will catch.
4. **Size budget met.** Runtime `.so` + model asset (or OTA contract) fits the install-size and cellular-download thresholds. App Bundle / per-ABI splits on Android, thinning on iOS, fp16 + palettization on the model.
5. **Threading is clean.** Inference runs off the UI thread; camera-driven inference drops backlog frames rather than queueing them. No synchronous inference on main.
6. **Thermal behavior measured.** A sustained run (minutes, not a cold burst) was benchmarked; the throttled p99 is within budget, or the feature is rate-limited / moved to `.cpuAndNeuralEngine` to stay cool.
7. **Telemetry is live.** Backend, p50/p99, cold-start, crash-at-init, and (for OTA) model version are all reported and queryable. You can answer "what fraction of users got the accelerator?" and "did the last model push regress p99?" from the field.
8. **Versioning is contract-checked.** The app refuses to load a model whose contract version it doesn't understand and falls back to the bundled baseline. Mandatory for OTA, automatic for bundled.

When all eight are green, you have not just a model that runs — you have a model that ships, survives the field, and tells you how it's doing. That is the actual deliverable, and it's the one the notebook latency slide never represents.

## Key takeaways

- **"Runs in a notebook" and "runs in the store build" are different problems.** The device is a shared, thermally limited, memory-bandwidth-starved heterogeneous SoC, and the backend you tested is rarely the backend that runs in the field. Design for that, don't be surprised by it.
- **Ship the fallback ladder, not one backend.** NPU/ANE → GPU → CPU, where the CPU floor cannot fail. Wrap delegate init in try/catch, record which rung each device landed on, and never let an accelerator failure crash launch.
- **The first inference is special.** A mobile GPU pays tens of ms and an NPU/ANE pays *hundreds* of ms of one-time compile on the first run. Warm up at load on a background thread so the user's first tap is at steady state.
- **One unsupported op in the middle of a graph can cost more than it "saves,"** because the copies across the backend boundary are memory-bandwidth-bound and paid regardless of how cheap the op is. Minimize *partitions*, not just unsupported ops.
- **Preprocessing parity is a silent accuracy killer.** Bake normalization into the model graph, golden-test the resize/crop/colorspace against training to a few LSBs, and you'll catch the mysterious field-vs-lab gap before it ships.
- **Bundle small models, OTA large ones.** Under ~30 MB: bundle (simpler, offline, never worse than its build). Over ~150 MB or fast-iterating: OTA (size relief and hours-not-weeks model updates), but pay for versioning, integrity, and rollback.
- **Measure honestly: warmed up, batch=1, p50 *and* p99, on a named device in a stated thermal state.** Report cold start separately. The mean lies; the tail is what users feel.
- **The accelerator is a battery decision as much as a latency one.** The NPU/ANE is often an order of magnitude more energy-efficient per inference; if telemetry shows most devices on CPU, that's a battery problem too. NNAPI is deprecated — use vendor delegates for the top NPU rung.

## Further reading

- **LiteRT / TensorFlow Lite documentation** — delegates (GPU, XNNPACK), the `Interpreter` API, model optimization, and the LiteRT rebrand notes. The canonical Android runtime reference.
- **Core ML Tools (`coremltools`) documentation** — conversion from PyTorch/TF, `ImageType` preprocessing, `ComputeUnit` selection, palettization and weight compression, the Core ML Instruments profiling template.
- **ONNX Runtime Mobile documentation** — the mobile/minimal build, ORT-format models, execution providers (XNNPACK, NNAPI, QNN) — the alternative if you standardize on ONNX.
- **ExecuTorch documentation** — `torch.export`, backend partitioners (XNNPACK, Core ML, Qualcomm, MediaTek), `.pte` packaging — PyTorch's native on-device runtime for one-source cross-platform deployment.
- Howard et al., **"Searching for MobileNetV3"** (2019) — the architecture and its reported on-device latencies; the baseline your deployment pipeline improves on.
- Sanh et al., **"DistilBERT, a distilled version of BERT"** (2019) — the 40%-smaller, 60%-faster, ~97%-of-performance distillation result behind on-device encoders.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame, [inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared) for choosing the runtime you ship here, [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) for the silicon under the delegates, [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for the measurement discipline, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that ties every lever together.

---
title: "From model to deployable artifact: graph capture, ONNX, and the conversion footguns"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn exactly how a PyTorch model becomes a frozen graph the device can run — eager versus traced versus exported capture, ONNX as the interchange hub, the conversion footguns that silently break accuracy, and the numeric-diff discipline that catches them before they ship."
tags:
  [
    "edge-ai",
    "model-optimization",
    "onnx",
    "torch-export",
    "graph-capture",
    "model-conversion",
    "onnxruntime",
    "inference",
    "efficient-ml",
    "deployment",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/from-model-to-deployable-artifact-1.png"
---

The bug report said the model was returning the wrong answer for "about a third" of requests, and only in production.

The model was a small audio front-end that decided, per frame, whether to run an expensive denoiser or pass the frame through cheaply. In Python it had a clean little branch: `if noise_energy.sum() > threshold: x = denoise(x)`. On the engineer's laptop, every unit test passed. The exported model — a `.onnx` file blessed by every checker we ran — also passed those same tests. We shipped it. Two days later, quiet rooms started getting the denoiser applied to silence and loud rooms started getting nothing, which was exactly backwards. Same weights, same architecture, same test suite. The training model was right and the deployed model was wrong, and nothing in our pipeline had said a word about it.

The cause was not a bug in our code, in PyTorch, in ONNX, or in the runtime. It was the conversion working *exactly as designed*. We had exported the model by **tracing** it — running it once on an example input and recording the operations that happened. The example input was a noisy frame, so the trace went down the `denoise` branch and recorded only that branch. The `if` was gone. The exported graph denoised *everything*, unconditionally, and the calibration-set inputs we tested with happened to be noisy too, so the test never caught it. The whole class of failure has a name in my head now: **the trip from Python to a frozen graph is where "it worked in training" goes to die.**

This post is about that trip. Your model lives in Python, in PyTorch, with the interpreter in the loop on every forward pass and your full debugger available. The device runs a frozen, serialized graph in C++ with no Python anywhere. Between those two worlds sits **graph capture and export**, and it is the single most underestimated stage in the whole edge pipeline — the place where data-dependent control flow vanishes, where shapes get hard-coded, where an op silently falls back or fails, and where an opset mismatch turns a Friday export into a Monday outage. By the end you will be able to take a PyTorch model, capture its graph correctly, export it to ONNX with the right opset and dynamic axes, run it under ONNX Runtime, *numerically verify* that the runtime agrees with the framework to a stated tolerance, read an export error and fix it, and decide when ONNX is the right hub versus when to go straight to TFLite, Core ML, or ExecuTorch. The figure below is the map: four ways to capture a graph and what each one keeps or loses.

![A matrix comparing eager, traced, scripted, and exported graph capture across graph shape, control flow handling, and whether the result is deployable](/imgs/blogs/from-model-to-deployable-artifact-1.png)

This is the post that opens the compiler-and-runtime track of the series. The [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) frames the four levers — quantization, pruning, distillation, efficient architecture — sitting on compilers and runtimes, validated by profiling. Every one of those levers eventually produces a model that has to *leave Python*. Quantization gives you a model with fake-quant nodes that must become real int8 ops in the runtime. Pruning gives you a sparse model that a runtime has to actually exploit. Distillation gives you a student you still have to export. None of it ships until the graph is captured and converted, and a botched conversion can erase every accuracy point you fought for. So before we optimize anything further in this series, we have to get the model *out* of the training framework intact. That is the job here.

## Eager, traced, scripted, exported: what a graph capture actually is

Start with the thing everyone skips: what is the difference between the model running in your notebook and "a graph"?

When you call `model(x)` in PyTorch, you are in **eager mode**. There is no graph. Each operation — a matmul, a ReLU, an add — executes immediately, in Python, the moment its line is reached. The "graph" exists only implicitly, as the sequence of calls the Python interpreter happens to make this time. This is define-by-run: the program *is* the model. It is wonderful for research because you can put a breakpoint anywhere, print any tensor, and write arbitrary Python — loops whose length depends on the data, branches on tensor values, recursion. The cost is that Python is in the loop on every single op, which is fine on a server with a fast CPU feeding a GPU and catastrophic on a microcontroller with no Python at all.

A **computation graph** is the explicit, reified version of that: a directed acyclic graph (DAG) whose nodes are operations and whose edges are tensors. "Directed" because data flows one way, from inputs toward outputs. "Acyclic" because — at the level the device cares about — there are no loops back; a recurrent network is *unrolled* into a finite chain of ops at export time. Once you have this DAG as data (not as Python code), you can serialize it to a file, ship it, load it in C++, optimize it (fuse a conv-bn-relu into one kernel), and run it with zero interpreter overhead. The entire reason edge inference is fast is that the graph is frozen and the runtime can plan everything ahead of time. The price you pay for that freezing is the subject of this post.

There are four ways PyTorch turns your eager model into something graph-shaped, and the differences between them are the whole ballgame.

**Eager** is the baseline — no capture at all. You ship Python. On the edge this is almost never an option, but it is the *reference*: whatever you capture has to match eager numerically.

**Tracing** (`torch.jit.trace`) runs the model once on an example input and records every tensor operation that actually executed. It is the easiest capture and the most dangerous, for one reason: it records *what happened*, not *what the code says*. An `if x.sum() > 0:` on a tensor value is a Python branch — the trace takes whichever side ran for the example and bakes that single path into the graph. A `for i in range(x.shape[0]):` loop gets unrolled to exactly the length it had for the example input. Anything data-dependent is silently flattened to the one path the example took. Tracing also can't see Python-side control flow at all; it only sees the tensor ops. This is exactly the bug that opened this post.

**Scripting** (`torch.jit.script`) takes the opposite approach: it parses your Python source with a compiler that understands a typed subset of Python (TorchScript) and produces a graph that *includes* the control flow as real graph constructs. An `if` becomes a graph-level conditional; a `for` loop becomes a graph loop. Scripting therefore preserves data-dependent control flow that tracing destroys. The catch is that the subset is restrictive — many idioms, libraries, and dynamic-typing tricks don't compile — and the resulting graphs can be brittle. TorchScript is essentially in maintenance mode now; it is the legacy path, and you should know it exists mostly so you recognize it in old codebases.

**Export** (`torch.export`, stabilized in recent PyTorch) is the modern answer and the one to reach for. It performs a specialized trace that produces a **sound, full-graph IR** — a single ATen-level graph — but unlike `jit.trace` it does *not* silently bake control flow. Instead it requires you to express data-dependent control flow through structured operators (`torch.cond`, `torch.while_loop`) that survive into the graph, and it inserts **guards**: recorded assumptions about input shapes and values, so that if you later feed an input that violates an assumption the export was specialized for, you get a loud error instead of a silent wrong answer. `torch.export` is what feeds ExecuTorch (PyTorch's own edge runtime) and is increasingly the front-end for the ONNX exporter too. The mental model to carry: tracing answers "what ops ran?", export answers "what is the *complete* graph, with its assumptions written down?"

The matrix at the top of this section lays these four against the three properties that decide whether you can ship: what shape of graph you get, whether control flow survives, and whether the result is deployable. The single most important row is the control-flow column, because that is where silent correctness bugs hide — and silent is the operative word. A crash is a gift; it tells you something is wrong. A traced model that quietly bakes one branch passes every test you wrote against the inputs that happened to take that branch.

### Why "frozen graph" is both the point and the problem

The reason the device wants a frozen graph is performance and portability, and both are real. A frozen DAG can be ahead-of-time optimized: constant-folded, operator-fused, memory-planned, scheduled. There is no interpreter, no Python GIL, no dynamic dispatch — just a fixed sequence of kernel calls over pre-allocated buffers. On a Cortex-M microcontroller there is *no Python runtime at all*; a frozen graph compiled to C is the only thing that can run. On a phone NPU the driver wants a static graph it can map to its fixed-function units. Freezing is not an optimization you opt into; it is the precondition for edge inference existing.

But freezing means committing. Every decision that was dynamic in Python — which branch, how many loop iterations, what shape — has to be resolved into something static, or explicitly marked as dynamic and carried into the graph as a symbolic dimension or a graph-level conditional. The footguns in this post are all the same shape: *something that was dynamic and correct in Python got frozen to one specific value during capture, and the one value was wrong for production.* Keep that sentence; it explains every failure mode in this article.

### The IR underneath: what export actually produces

It helps to be precise about what a "captured graph" *is* as a data structure, because the abstraction leaks in ways that matter. When `torch.export` runs, it produces an `ExportedProgram`: a `GraphModule` whose `graph` is a list of `Node` objects, each with an `op` (one of `placeholder` for an input, `call_function` for an operation, `get_attr` for a parameter, `output` for the result), a `target` (the actual function, e.g. `torch.ops.aten.convolution.default`), `args`, and `kwargs`. The operations are at the **ATen** level — PyTorch's core tensor operator set, the layer below the friendly `nn.Module` API. Your `nn.Conv2d` has already been decomposed into `aten.convolution`; your `nn.BatchNorm2d` in eval mode has been folded into the affine `aten.mul` and `aten.add` it actually is at inference. This decomposition is itself a place bugs hide: a module that does something clever in its Python `forward` (caches a value, mutates a buffer, reads `self.training`) decomposes to whatever its tensor ops were *at capture time*, and the cleverness is gone.

Alongside the graph, the `ExportedProgram` carries a `graph_signature` (which placeholders are user inputs versus parameters versus buffers), the `state_dict` (the actual weight tensors), and — critically — the **guards**, also called the symbolic-shape `ShapeEnv`. The guards are a set of boolean assertions on the symbolic shapes that the export specialized under: "input dim 0 is dynamic in [1, 64]", "dim 1 equals 3", "dim 2 equals dim 3". When you run the exported program, those guards are checked; a violation raises rather than silently recomputing. This is the mechanical difference from `jit.trace`, which records no such assertions at all and therefore cannot tell you when you've left the regime it was captured under. The whole `torch.export` value proposition reduces to: *the assumptions are first-class data in the artifact, and they are enforced.*

### Why a DAG and not arbitrary code

One more piece of the science, because it explains a constraint you'll keep bumping into: why must the deployable thing be a *directed acyclic* graph, and not just "whatever Python did"? The acyclic requirement comes from how runtimes schedule. A frozen graph is executed by topological order: compute a node only after all its inputs are ready. A topological order exists if and only if the graph is acyclic. A genuine cycle — a node that depends on its own output — has no valid execution order and cannot be scheduled ahead of time. Recurrence in a model (an RNN, a diffusion loop, an autoregressive decode) is therefore *not* a cycle in the deployed graph; it is either unrolled to a fixed number of steps at capture time (which is why a traced RNN bakes in the sequence length it saw) or expressed as a graph-level `while_loop` operator whose *body* is an acyclic subgraph executed repeatedly by the runtime. Either way, the thing on the device is acyclic. When you see "no cycles" as a rule for graph IRs, that is the reason: schedulability. It is also why a feedback loop in your architecture is a capture decision you must make explicitly — unroll it (fixed, fast, but fixed length) or use `while_loop` (dynamic length, preserved, slightly more runtime machinery) — rather than something the exporter can guess for you.

## ONNX: an interchange format and why interchange wins

You have a captured graph. Now you need it in a format the device's runtime understands. You could export directly to your target's native format — TFLite for Android, Core ML for Apple, a TensorRT engine for NVIDIA, ExecuTorch for PyTorch's own runtime — and sometimes you should (we'll get there). But the workhorse of cross-framework, cross-runtime deployment is **ONNX** (Open Neural Network Exchange), and understanding *why* an interchange format exists is more useful than memorizing its API.

The problem ONNX solves is combinatorial. Suppose there are $N$ training frameworks (PyTorch, TensorFlow, JAX, scikit-learn, XGBoost) and $M$ inference runtimes (ONNX Runtime, TensorRT, OpenVINO, a phone NPU SDK, a custom embedded engine). If every framework has to write a direct exporter for every runtime, you need $N \times M$ bridges, each one separately written, tested, and kept in sync as both ends evolve. That is the integration explosion. An **interchange format** breaks it: every framework writes *one* exporter to the common format, and every runtime writes *one* importer from it. Now you need $N + M$ adapters instead of $N \times M$. With $N = M = 3$ that is the difference between $9$ bridges and $6$; at realistic scale ($N, M \approx 6\text{--}10$) it is the difference between dozens of brittle one-off integrations and a maintainable hub. The figure below is that hub.

![A branching graph showing PyTorch, TensorFlow, and scikit-learn all exporting into a central ONNX graph, which then fans out to ONNX Runtime, TensorRT, and OpenVINO runtimes](/imgs/blogs/from-model-to-deployable-artifact-2.png)

So what is *in* a `.onnx` file? It is a serialized protocol buffer (protobuf) describing a graph. Concretely it holds: a list of **nodes**, each an operator invocation (an `op_type` like `Conv`, `Gemm`, `Relu`, `MatMul`, plus its attributes); a list of **initializers** (the weights, as tensors); the **inputs and outputs** of the graph with their names, types, and shapes (shapes can include symbolic dimensions like `batch`); and an **opset version**, the single most operationally important field in the file. Every operator in ONNX is defined by a versioned **operator set (opset)** — a specification that says, for opset 17, exactly what `Resize` means, what attributes it takes, and what its numerics are. The opset is a contract. When you export, you target an opset; when a runtime loads the file, it has to implement that opset's definitions. Mismatch here is the classic "worked Friday, broke Monday" outage: the exporter used opset 18, the serving fleet's runtime only implements up to opset 17, and the file refuses to load or — worse — an operator behaves subtly differently across opset versions.

This is the part of the ONNX promise people get burned by. "Train in PyTorch, export to ONNX, run anywhere" is true the way "this car can go 200 mph" is true: under specific conditions, with the right setup, not in your driveway. ONNX is not magic portability; it is a precise contract between a serialization format and a graph-compiling runtime, and the production incidents come from violating one half of the contract. The [ONNX deep dive in the MLOps track](/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving) goes deep on what the runtime does with the file once it loads — execution providers, graph partitioning, the real cost of portability — and is the natural companion to this post; here we stay focused on the *capture and export* half, the part that happens on your side before the file ever reaches a runtime.

### Opsets, operators, and the version contract in practice

A concrete feel for the opset contract. The `Resize` operator changed its coordinate-transform behavior between opsets; `Squeeze`/`Unsqueeze` moved the axes from an attribute to an input around opset 13; many quantization ops (`QLinearConv`, `DequantizeLinear`) only exist or only support certain dtypes at certain opsets. When you choose an opset to export to, you are choosing a *floor*: the runtime must implement at least that opset. The right discipline is to pin the export opset to the **minimum opset that supports every op your model needs**, and to verify that floor against the **actual runtime version running on the device** — not the one on your laptop. A surprising fraction of "it doesn't load" reduces to: the device's runtime is older than your export. Treat the opset like a deployment dependency, because it is one.

Why do operator semantics change across opsets at all? Because ONNX is a standard maintained by committee, and standards accrete corrections. An early `Resize` made an implicit assumption about how pixel coordinates map between input and output grids; that assumption produced visibly different results from the framework that originated the model, so a later opset added explicit `coordinate_transformation_mode` attributes to make the behavior unambiguous. The upshot for you: the *same* op name can compute slightly different numbers at opset 11 versus opset 18, and if your exporter and your runtime disagree on which opset's semantics to apply, you get a numeric diff that traces to exactly one op. This is not hypothetical; resize/interpolation mismatches are one of the most common per-op divergences people find when they bisect a drifting vision model. The defense is the same numeric-diff discipline we build below: if you pin the opset and *still* see drift on a `Resize` or `Upsample`, you have found an opset-semantics mismatch, and the fix is to align the exporter's emitted attributes with what the runtime expects.

### What the bytes actually look like

To make the file concrete rather than abstract: a `.onnx` file is a `ModelProto` protobuf. At the top sits metadata — the producer name and version, the IR version, and the `opset_import` list (domain plus version; the default domain `""` is the standard ops, and custom-op domains like `"com.microsoft"` or your own appear here too). Inside is one `GraphProto` with four lists that carry all the meaning: `node` (the ops, each a `NodeProto` with `input` names, `output` names, an `op_type`, and `attribute` entries), `initializer` (the weights, each a `TensorProto` with a dtype enum, dims, and raw bytes), `input` and `output` (each a `ValueInfoProto` with a name and a `TypeProto` whose shape can mix concrete `dim_value` integers and symbolic `dim_param` strings like `"batch"`), and `value_info` (optional type/shape annotations for intermediate tensors, which is exactly what the per-op bisection trick later exploits). Edges are implicit: a node's input names refer to other nodes' output names, to initializers, or to graph inputs — there is no separate edge list, the names *are* the wiring. When you `onnx.load` a file and print `model.graph.node`, you are reading this directly. Knowing the structure turns "the export is wrong somehow" into "let me list the nodes and find the one with the surprising `op_type` or the fp64 initializer," which is a tractable investigation instead of a mystery.

## The export, end to end: code that actually runs

Enough theory. Here is the canonical `torch.onnx.export` for a real model — a ResNet-18 image classifier, the running spine for this post — with the three things people forget: the opset, the dynamic axes, and named inputs/outputs.

```python
import torch
import torchvision

# 1. Build and FREEZE the model. eval() is not optional:
#    it switches dropout off and batch-norm to inference stats.
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.eval()

# 2. An example input. Its shape and dtype are what the exporter
#    specializes on, so make it representative (right channels,
#    right spatial size, fp32 not fp64).
example = torch.randn(1, 3, 224, 224, dtype=torch.float32)

# 3. Export. The flags that matter:
torch.onnx.export(
    model,
    example,
    "resnet18.onnx",
    input_names=["input"],            # name the graph IO so you can
    output_names=["logits"],          # bind to it by name at runtime
    opset_version=17,                 # pin to the runtime's floor
    dynamic_axes={                    # mark dim 0 (batch) as dynamic
        "input":  {0: "batch"},
        "logits": {0: "batch"},
    },
    do_constant_folding=True,         # fold constants at export time
)
print("exported resnet18.onnx")
```

Three things to internalize. `model.eval()` is the single most common forgotten line and it is a correctness bug, not a style nit: in `train()` mode dropout is active (your inference becomes stochastic) and batch-norm uses the *current batch's* statistics instead of the running averages it learned, so a model exported in train mode is numerically wrong in a way that may pass tests on large batches and fail on batch-1 edge inputs. `opset_version=17` pins the contract. And `dynamic_axes` is what saves you from the frozen-batch failure we'll dissect later: without it, the exporter hard-codes the example's batch size (here, 1) into every shape in the graph.

Now **always** validate the file structurally before you trust it:

```python
import onnx

m = onnx.load("resnet18.onnx")
onnx.checker.check_model(m)          # raises if the graph is malformed
print("opset:", m.opset_import[0].version)
print("inputs:", [i.name for i in m.graph.input])
# Inspect the symbolic shape we asked for:
in0 = m.graph.input[0].type.tensor_type.shape.dim
print("input dim0:", in0[0].dim_param or in0[0].dim_value)  # -> "batch"
```

`onnx.checker.check_model` validates that the graph is well-formed — every node's inputs exist, types are consistent, the opset is coherent. It does **not** check that the model is numerically correct. That is a separate, mandatory step we'll do in a moment. A model can pass the checker and still produce garbage; the checker only guarantees the file is a *valid* ONNX graph, not a *correct* one.

### Run it under ONNX Runtime and compare to PyTorch

The whole point of capturing a graph is to run it somewhere other than PyTorch. Here is the minimal ONNX Runtime inference, plus the comparison that turns "I exported it" into "I verified it":

```python
import numpy as np
import onnxruntime as ort

# A fresh input we'll run through BOTH engines.
x = torch.randn(4, 3, 224, 224)          # note: batch=4, not 1!

# Reference: PyTorch eager output.
with torch.no_grad():
    ref = model(x).numpy()

# Candidate: ONNX Runtime output.
sess = ort.InferenceSession("resnet18.onnx",
                            providers=["CPUExecutionProvider"])
ort_out = sess.run(["logits"], {"input": x.numpy()})[0]

# The discipline: a NUMERIC diff with a stated tolerance.
max_abs = np.max(np.abs(ref - ort_out))
max_rel = np.max(np.abs(ref - ort_out) / (np.abs(ref) + 1e-8))
print(f"max abs error: {max_abs:.2e}")
print(f"max rel error: {max_rel:.2e}")
assert np.allclose(ref, ort_out, atol=1e-4, rtol=1e-3), "EXPORT DRIFTED"
print("verified: ORT matches PyTorch")
```

Notice the input is **batch 4** while we exported with example batch 1. That is the dynamic-axes test: if dim 0 were frozen, this `sess.run` would throw a shape mismatch. Because we marked it dynamic, it runs and we get to compare. The `assert np.allclose` is not optional ceremony — it is the gate between "exported" and "deployable." Floating-point reassociation during graph optimization (a fused conv-bn computes in a slightly different order) means you will rarely get bit-exact agreement; `atol=1e-4` on fp32 logits is a sane default. If the error is $10^{-6}$, ship it. If it is $10^{-1}$, something is structurally wrong and the rest of this post is your debugging guide.

### The newer flow: torch.export

`torch.onnx.export` increasingly runs *on top of* `torch.export` under the hood (the `dynamo=True` path), but you can also use `torch.export` directly when your target is ExecuTorch or when you want the sound IR before lowering to ONNX. The shape:

```python
import torch
from torch.export import export, Dim

model.eval()
example = (torch.randn(1, 3, 224, 224),)

# Declare which dims are dynamic with explicit symbolic Dims.
batch = Dim("batch", min=1, max=64)
ep = export(
    model,
    example,
    dynamic_shapes={"x": {0: batch}},   # arg name -> {dim: Dim}
)

# ep is an ExportedProgram: a sound ATen graph + guards.
print(ep.graph_module.code[:400])       # human-readable graph
# Lower to ONNX from the exported program (dynamo exporter):
onnx_prog = torch.onnx.export(ep, dynamo=True)
onnx_prog.save("resnet18_dynamo.onnx")
```

The difference that matters: `Dim("batch", min=1, max=64)` is a *constraint*, not just a flag. The exported program records the assumption "batch is between 1 and 64." Feed it batch 128 later and you get a guard violation — a loud, specific error — instead of a silent miscompile. That is the whole philosophy of `torch.export`: assumptions are data in the graph, and violating them fails loudly. The conversion pipeline below ties the stages together, with a verification step bolted to every handoff.

![A six-stage timeline of the conversion pipeline running from freeze model, export graph, onnx checker, ORT inference, numeric diff, to deploy artifact](/imgs/blogs/from-model-to-deployable-artifact-3.png)

## The footguns, each one explained

Every conversion failure I have debugged fits into a small catalog. Name the category and the fix is usually one line. Here is the catalog, each with the mechanism, the symptom, and the fix.

### Footgun 1: data-dependent control flow

The headliner, and the bug that opened this post. A branch or loop whose condition depends on a *tensor value* — `if x.sum() > 0`, `while not converged`, `for i in range(x.shape[0])` — is dynamic. Tracing records only the path the example took. The symptom is the nastiest possible: no error, correct output for inputs that take the recorded path, wrong output for inputs that would have taken the other path. It survives unit tests if your tests happen to use inputs from the recorded branch. The fix is to either (a) use `torch.export` with `torch.cond`/`torch.while_loop` so the control flow becomes graph ops, or (b) refactor the control flow out of the model entirely (do the branch in the host code around the model, not inside the captured graph). We dissect this with code in the worked examples.

### Footgun 2: dynamic shapes

If your model must accept variable batch size, variable sequence length, or variable image resolution, every one of those dimensions has to be marked dynamic at export — `dynamic_axes` in `torch.onnx.export`, `Dim(...)` in `torch.export`. Forget it and the exporter helpfully hard-codes the example's shape into the graph. The symptom is a shape-mismatch error at the first production input that differs from the example — usually batch size, because you exported with batch 1 and production batches are larger. Less obviously, sequence models exported with a fixed length silently truncate or pad incorrectly. The fix is to enumerate every dimension that varies and mark it dynamic. The figure on static-versus-dynamic shapes later in this post shows exactly what the graph commits to.

### Footgun 3: unsupported and custom ops

ONNX's opset is large but finite. A custom CUDA kernel, an exotic op from a research repo, a brand-new layer that predates the opset that standardized it, or a third-party library op — any of these can be a hole in the conversion. The symptom is an explicit export error: "Exporting the operator `::my_op` to ONNX opset version 17 is not supported." This one at least fails loudly. The fixes, in order of preference: (a) rewrite the op in terms of supported primitives (often a fancy attention or activation decomposes into matmuls, softmaxes, and elementwise ops that all export fine); (b) register a **custom symbolic** function that tells the exporter how to emit the op as an ONNX subgraph; (c) export the op as a custom ONNX op and implement a matching custom operator in the runtime. We do (b) in the code below.

### Footgun 4: opset version mismatch

Covered above — the export targets an opset the runtime doesn't implement, or an op's semantics differ across opset versions. Symptom: load failure ("opset N is not supported") or, more insidiously, a numeric diff that traces to one op whose definition changed. Fix: pin the export opset to the runtime's floor and *verify against the deployed runtime version*, not your dev box.

### Footgun 5: Python-side preprocessing that doesn't get captured

This one is a category error that bites everyone once. Your model's `forward` is not the whole inference path. The resize, the normalization (subtract mean, divide by std), the tokenization, the `argmax` decode at the end — if any of that lives in Python *around* the model rather than inside the captured graph, it does not get exported. The symptom is a model that's numerically "correct" against your PyTorch test (because your test also runs the Python preprocessing) but produces nonsense in the C++ app (which only has the graph). The fix is to decide explicitly where the boundary is: either fold preprocessing into the model so it gets captured (PyTorch's `transforms` can often be made traceable, or you write the normalize as graph ops), or replicate the exact preprocessing in the deployment host code and *test the host code against the same reference*. The trap is assuming "the model" and "the inference" are the same thing. They are not.

### Footgun 6: fp64 constants

A subtle one. If a constant in your model is a Python float promoted to fp64 (double), or a `numpy` array created without an explicit dtype, the exporter may emit fp64 tensors into the graph. Many edge runtimes — phone NPUs especially — do not support fp64 at all, or fall back to CPU for fp64 ops, tanking performance. The symptom ranges from an unsupported-dtype error to a mysterious slowdown where one op runs on CPU. The fix is to ensure every constant is fp32: build tensors with `dtype=torch.float32`, cast literals, and audit any `numpy` arrays baked into the module.

### Footgun 7: training-only layers in train mode

The `model.eval()` footgun, promoted to its own entry because it is that common. Dropout, batch-norm, and any layer with train/eval behavior must be in eval mode at export. Forget `eval()` and dropout randomly zeros activations at inference (non-deterministic outputs) and batch-norm normalizes by the export batch's statistics instead of the learned running stats. Symptom: the exported model's output differs from a properly-eval'd PyTorch run, sometimes only at small batch sizes. Fix: `model.eval()` before export, every time, and assert it in your export script.

The matrix below is this catalog as a field guide — symptom to cause to fix — the thing to pin above your desk during a conversion sprint.

![A matrix mapping six conversion symptoms to their underlying cause and the one-line export-side fix for each](/imgs/blogs/from-model-to-deployable-artifact-6.png)

## Worked example: the if-on-a-tensor that baked one branch

Let's reproduce the opening bug in miniature and fix it, with numbers. Here is a model with exactly one data-dependent branch.

#### Worked example: trace bakes the branch, export preserves it

```python
import torch

class Gated(torch.nn.Module):
    """If the input's mean is positive, scale up; else scale down.
       The branch is DATA-DEPENDENT (depends on a tensor value)."""
    def forward(self, x):
        if x.mean() > 0:
            return x * 2.0          # branch A
        else:
            return x * 0.5          # branch B

m = Gated().eval()

# Trace it with a POSITIVE-mean example -> records branch A only.
pos = torch.ones(8)
traced = torch.jit.trace(m, pos)

# Now feed a NEGATIVE-mean input to both:
neg = -torch.ones(8)
print("eager  (neg):", m(neg)[0].item())       # -0.5  (branch B, correct)
print("traced (neg):", traced(neg)[0].item())  # -2.0  (branch A, WRONG)
```

The eager model returns `-0.5` for the negative input (branch B, correct). The traced model returns `-2.0` — it took branch A, the one path it recorded, even though the input clearly should hit branch B. There is no error, no warning. If your test set were all-positive inputs, this ships and breaks in production exactly like our audio front-end did. The numeric diff would have caught it: `max_abs = 1.5` on a model whose outputs are order-1, screaming that something is structurally wrong.

The fix with `torch.cond`, which `torch.export` captures as a real graph conditional:

```python
import torch
from torch.export import export

class GatedFixed(torch.nn.Module):
    def forward(self, x):
        return torch.cond(
            x.mean() > 0,           # predicate (a tensor bool)
            lambda x: x * 2.0,      # true branch
            lambda x: x * 0.5,      # false branch
            (x,),                   # operands passed to both
        )

m = GatedFixed().eval()
ep = export(m, (torch.ones(8),))

# Both branches are now in the graph. Verify on the hard input:
neg = -torch.ones(8)
out = ep.module()(neg)
print("exported (neg):", out[0].item())   # -0.5  (correct!)
assert torch.allclose(out, m(neg))
print("both branches preserved")
```

`torch.cond` turns the Python `if` into a graph operator with two subgraphs and a predicate. The exported program contains *both* branches and selects at runtime based on the actual predicate value — exactly what eager did. Now the negative input correctly returns `-0.5`. The before-and-after below is the mechanism: tracing collapses the diamond to a single edge; export keeps the diamond.

![A before-and-after figure contrasting a traced model that baked only one branch against an exported model that preserves both branches via torch.cond](/imgs/blogs/from-model-to-deployable-artifact-4.png)

The deeper lesson is about *where* to put control flow. Often the cleanest fix is not `torch.cond` at all — it is to **lift the branch out of the model**. If the audio front-end's "denoise or not" decision had lived in the host C++ loop ("compute noise energy, then choose which graph to run"), there would have been no data-dependent control flow inside the captured graph and no footgun. The captured graph should be the pure, branch-free tensor math; the host orchestrates. When you can't lift it out — when the branch is genuinely in the middle of the tensor pipeline — `torch.cond` is the tool. But reach for "move it to the host" first.

## Worked example: dynamic axes and the frozen-batch failure

The second classic. Export a model without telling the exporter that batch size varies, and watch it commit to the example's batch.

#### Worked example: frozen batch vs dynamic batch

```python
import torch, torchvision, onnxruntime as ort, numpy as np

model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
example = torch.randn(1, 3, 224, 224)     # batch 1 example

# --- WRONG: no dynamic_axes. Batch dim is frozen to 1. ---
torch.onnx.export(model, example, "static.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=17)

sess = ort.InferenceSession("static.onnx",
                            providers=["CPUExecutionProvider"])
try:
    sess.run(None, {"input": np.random.randn(8, 3, 224, 224)
                    .astype("float32")})
except Exception as e:
    print("STATIC export, batch 8:", type(e).__name__)
    # -> Fails: "Got invalid dimensions ... index 0 Got 8 Expected 1"
```

The static export hard-codes `[1, 3, 224, 224]` into the input shape and every downstream shape. Feed it batch 8 and ONNX Runtime refuses — the graph literally says the batch dimension is 1. In production, where you might batch requests for throughput, this is an immediate hard failure (at least it's loud, unlike the control-flow bug). The fix is one argument:

```python
# --- RIGHT: mark dim 0 dynamic. ---
torch.onnx.export(model, example, "dynamic.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=17,
                  dynamic_axes={"input":  {0: "batch"},
                                "logits": {0: "batch"}})

sess = ort.InferenceSession("dynamic.onnx",
                            providers=["CPUExecutionProvider"])
for b in (1, 8, 32):
    out = sess.run(None, {"input": np.random.randn(b, 3, 224, 224)
                          .astype("float32")})[0]
    print(f"DYNAMIC export, batch {b}: out shape {out.shape}")
    # -> works for every batch size
```

With `{0: "batch"}`, the graph's input shape becomes `[batch, 3, 224, 224]` with a *symbolic* first dimension, and the runtime accepts any batch. The same trick handles variable sequence length for transformers (`{1: "seq"}`), variable resolution for vision (`{2: "height", 3: "width"}`), and so on — you mark every axis that genuinely varies and leave fixed the ones that don't (marking too much dynamic can cost you graph-optimization opportunities, so don't reflexively mark everything). The figure makes the commitment concrete: a static export carves the example's numbers into the shapes; a dynamic export carries a symbol.

![A before-and-after figure showing a static export freezing the batch dimension versus a dynamic export carrying a symbolic batch dimension](/imgs/blogs/from-model-to-deployable-artifact-5.png)

There is a subtle stress test worth naming: dynamic axes are not free at every runtime. Some accelerators — certain NPU SDKs, TensorRT without explicit optimization profiles — prefer or require static shapes to plan memory and select kernels. The honest practice is to mark dynamic only what *must* vary, and when targeting a static-shape accelerator, export a small set of fixed-shape variants (e.g. batch 1, 8, 32) or use the runtime's dynamic-shape facility (TensorRT optimization profiles) explicitly. "Mark everything dynamic" is as wrong as "mark nothing"; the right answer is "mark exactly what varies, and know your runtime's appetite for it."

#### Worked example: the preprocessing the graph never saw

This is the footgun that's hardest to believe until it bites you, so let's make it concrete. A vision classifier's real inference path is: load image, resize to 224, convert to float, subtract the ImageNet mean and divide by the std, then run the network. In a research notebook, the resize and normalize live in a `torchvision.transforms` pipeline *outside* the model. You export "the model" — just the `forward` from the normalized tensor onward — and it numerically matches PyTorch perfectly, because your PyTorch test *also* runs the transforms before calling the model. Both sides do the preprocessing, so the diff is clean.

Then the C++ app loads the `.onnx` file and feeds it a raw `uint8` image tensor, because that's what "the model" takes, right? The graph receives unnormalized pixels in the range 0 to 255 instead of normalized values around zero, and the network — which learned on normalized inputs — produces confident nonsense. No error. The numeric diff against PyTorch would have caught it *if the diff had used the same raw input the app uses*, but the diff used the notebook's already-normalized tensor, so it passed. The trap is a definition mismatch: "the model" in the notebook starts after normalization; "the model" in the app starts before it.

There are two clean fixes, and the right one depends on the platform. Fold the preprocessing into the captured graph so it exports as ops:

```python
import torch, torchvision

class WithPreprocess(torch.nn.Module):
    """Bake normalization INTO the graph so it gets captured."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        # Constants become graph initializers (note: fp32!).
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):           # x: float image in [0, 1]
        x = (x - self.mean) / self.std    # now part of the graph
        return self.net(x)

model = WithPreprocess(
    torchvision.models.resnet18(weights="IMAGENET1K_V1")).eval()
torch.onnx.export(model, torch.rand(1, 3, 224, 224),
                  "resnet18_pp.onnx",
                  input_names=["image"], output_names=["logits"],
                  opset_version=17,
                  dynamic_axes={"image": {0: "batch"},
                                "logits": {0: "batch"}})
```

Now the normalization is `Sub` and `Div` nodes in the graph, the app feeds a `[0, 1]` float image, and there is one boundary instead of two. (Note the `register_buffer` constants are explicitly fp32 — building them as bare Python floats risks the fp64 footgun.) The alternative fix, preferred when preprocessing is genuinely platform-specific or expensive in graph ops, is to keep it in host code but **test the host code against the same reference**: write the C++ normalize, run a fixed image through both the C++ path and the Python reference path end to end, and diff *that*. The cardinal rule the example teaches: the unit of verification is the *whole inference*, not "the model." Whatever your boundary, the diff must span it.

#### Stress-testing the capture: where does it fall apart?

Good engineering is not just making the happy path work; it is knowing where the approach breaks. So let's stress the capture-and-export discipline and see where it strains.

*What happens when the model has a genuinely dynamic loop — an autoregressive decode whose length depends on an end-of-sequence token?* Tracing bakes the exact number of steps the example produced, which is almost never the right number for production. `torch.export` with `while_loop` preserves it, but the loop body must be a clean acyclic subgraph with a tensor-valued termination condition, and writing it that way can require restructuring code that was comfortable as a Python `while`. The honest answer is that fully dynamic generation loops are often *kept in the host*: the device runs one decode step as a graph, and the host C++ loop calls it repeatedly, checking the stop condition itself. This is exactly the [LLM serving](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) pattern — the per-step graph is captured, the generation loop is host code — and it sidesteps the hardest capture problem entirely.

*What happens when an op the NPU doesn't support sits in the middle of the graph?* The export succeeds (ONNX supports the op), but at runtime the execution provider can't run it, so the graph is *partitioned*: the unsupported op falls back to CPU, which means a tensor copy from NPU memory to CPU and back around that one node. The graph is correct but slow, and the slowness is invisible until you profile. This is where capture hands off to the [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization) and [inference-runtimes-compared](/blog/machine-learning/edge-ai/inference-runtimes-compared) posts — partitioning is a runtime concern — but the *export* decision that affects it is whether you used ops the target's provider actually claims. A model exported with a fancy op no NPU supports is a model that will run mostly on CPU.

*What happens when the calibration set for verification is tiny or unrepresentative?* The numeric diff is only as good as the inputs you diff on. If you verify on three positive-mean inputs, you will never catch the baked-branch bug that only fires on negative-mean inputs. The defense is to deliberately construct *adversarial* verification inputs for each footgun you know about: a negative input for control flow, an off-size batch for dynamic axes, an edge resolution for dynamic spatial dims, a raw-pixel image for the preprocessing boundary. A verification suite that only uses "normal" inputs is a verification suite that passes silently-broken models.

## Reading and fixing an export error: the unsupported op

The control-flow and shape footguns are silent or loud-but-simple. The unsupported-op footgun fails loudly with a specific message, and the fix is a genuinely useful skill. Here is a model using an op the exporter doesn't know, and the symbolic-function fix.

```python
import torch

# Pretend `torch.special.foo` is an op with no ONNX exporter.
# (In real life this is a custom CUDA op or a brand-new layer.)
class HasCustomOp(torch.nn.Module):
    def forward(self, x):
        # Stand-in for an unsupported op; in practice this raises
        # "Exporting the operator ... to ONNX is not supported".
        return torch.ops.aten.special_op(x)   # illustrative
```

When you export this, you get a precise error naming the operator and the opset. You have three moves.

**Move 1 — rewrite in primitives.** Most "exotic" ops are exotic only in name. A custom GELU variant, a fancy normalization, a novel attention — decompose it into matmul, softmax, mul, add, and the standard exporter handles all of those. This is the most portable fix because it produces a graph of standard ops every runtime supports. Read the op's math, write it with `torch` primitives, verify it matches numerically, export the rewritten version.

**Move 2 — register a custom symbolic.** If you can't or don't want to rewrite, you teach the exporter how to emit the op as an ONNX subgraph:

```python
import torch
from torch.onnx import symbolic_helper

# Tell the exporter: when you see my_op, emit this ONNX subgraph.
def my_op_symbolic(g, x):
    # g is the ONNX graph builder; emit standard ONNX ops.
    # Example: my_op(x) == Mul(Sigmoid(x), x)  (a silu/swish)
    sig = g.op("Sigmoid", x)
    return g.op("Mul", sig, x)

# Register for the namespaced operator at a given opset.
torch.onnx.register_custom_op_symbolic(
    "mylib::my_op", my_op_symbolic, opset_version=17)

# Now torch.onnx.export will emit Sigmoid+Mul wherever my_op appears.
```

The symbolic function receives the graph builder `g` and the op's inputs, and returns a subgraph of *standard* ONNX ops. This keeps the file portable (no custom runtime op needed) while supporting a source op the exporter didn't know.

**Move 3 — custom ONNX op plus runtime kernel.** If the op truly has no decomposition (a genuinely novel primitive), you export it as a custom ONNX op in your own domain and implement a matching custom operator in the runtime (ONNX Runtime supports registering custom ops). This is the heaviest path — you now own a kernel on the device — and the least portable, so use it last. The discipline across all three: after the fix, run the numeric diff against the original eager model. An unsupported-op rewrite that's *almost* right is its own footgun.

## The verify-the-conversion discipline

Everything above converges on one practice that, if you take only one thing from this post, take this: **a conversion is not done until you have numerically diffed the framework output against the runtime output, on the same input, to a stated tolerance.** Exporting is not verifying. The checker is not verifying. "It loaded" is not verifying. The only verification is: same input, both engines, compare numbers.

The method, in increasing depth:

**Level 1 — graph-level diff.** Run a representative input through PyTorch (eager, `eval()`, `no_grad`) and through the runtime, compute the max absolute and max relative error over the output, and assert it's under tolerance. This is the `np.allclose` we wrote earlier. Do it on *several* inputs, including adversarial ones for your known footguns: a negative-mean input (control flow), a different batch size (dynamic axes), an edge resolution (dynamic spatial dims). One input passing proves nothing about the others; the control-flow bug passes on positive inputs and fails on negative ones.

What tolerance? Here a little numeric science pays off, because "how close is close enough?" is the question that decides whether you ship. Floating-point addition is **not associative**: in fp32, $(a + b) + c$ and $a + (b + c)$ can differ in the last bits because each `+` rounds to the nearest representable value. Graph optimization *reorders* operations — a fused conv-bn-relu computes the bias add and the scale in a different order than the three separate ops, and a runtime may tile a matmul's reduction differently than PyTorch's kernel — so the exported graph computes the *same math* in a *different order* and lands on slightly different bits. The size of that drift is bounded by machine epsilon. For fp32, $\epsilon \approx 1.19 \times 10^{-7}$, and the relative error of a sum of $n$ terms accumulates roughly as $n\,\epsilon$ in the worst case. A ResNet layer reducing over a few thousand multiply-accumulates therefore drifts on the order of $10^{-5}$ to $10^{-4}$ relative — which is exactly the tolerance to set. So `rtol=1e-3, atol=1e-4` is not a guess; it's the floating-point reassociation envelope with headroom. If you see relative error around $10^{-5}$, that's clean reassociation: ship it. If you see $10^{-2}$ or larger, that is *not* rounding — no amount of reassociation produces percent-level error — and it is a structural mismatch (a baked branch, a wrong op, a train-mode layer) that you escalate to level 2. The threshold between "rounding" and "bug" is roughly three orders of magnitude above $\epsilon$; below it is physics, above it is a defect. For int8 quantized models the tolerance is necessarily looser and you compare *task metrics* (accuracy on a validation set) rather than raw logits, because the whole point of quantization is to trade tiny numeric drift for size and speed — that comparison belongs to the [quantization-in-practice](/blog/machine-learning/edge-ai/quantization-in-practice-a-full-int8-pipeline) post, not here.

**Level 2 — per-op bisection.** When the graph-level diff is too large and you don't know *which* op drifted, you bisect. ONNX lets you mark intermediate tensors as outputs, so you can run both engines and compare *at each layer*, walking forward until the first op whose outputs disagree. That op is your culprit — the unsupported op that got a wrong symbolic, the `Resize` whose opset semantics changed, the train-mode batch-norm. The technique:

```python
import onnx, onnxruntime as ort, numpy as np, torch

# Add every intermediate value_info as a graph output so ORT
# will return it (a "make all activations observable" trick).
m = onnx.load("resnet18.onnx")
value_names = [v.name for v in m.graph.value_info]
for name in value_names:
    m.graph.output.extend([onnx.ValueInfoProto(name=name)])
onnx.save(m, "resnet18_debug.onnx")

sess = ort.InferenceSession("resnet18_debug.onnx",
                            providers=["CPUExecutionProvider"])
x = torch.randn(1, 3, 224, 224)
ort_acts = dict(zip([o.name for o in sess.get_outputs()],
                    sess.run(None, {"input": x.numpy()})))

# Capture PyTorch's intermediate activations with forward hooks,
# then walk layers in order and report the FIRST that diverges.
torch_acts = {}
def hook(name):
    return lambda mod, inp, out: torch_acts.__setitem__(name, out)
for n, mod in model.named_modules():
    mod.register_forward_hook(hook(n))
with torch.no_grad():
    model(x)
# (Map module names to ONNX node outputs and diff in graph order;
#  the first op with max-abs error above tolerance is the bug.)
```

This forward-references the [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization) post, which covers how the runtime fuses and rewrites the graph — and why a fusion can introduce the drift you're bisecting toward. The bisection method is the same regardless: make activations observable, run both engines, find the first divergent op. The figure below is the loop you run every single time, not just when something breaks — because the cheapest moment to catch a conversion bug is the moment after it happens, not the day after it ships.

![A six-stage timeline of the verify-by-diff loop running fixed input, PyTorch reference, ONNX Runtime output, error computation, pass gate, and per-op bisection on failure](/imgs/blogs/from-model-to-deployable-artifact-7.png)

**Level 3 — task-metric diff.** For the cases where raw-output tolerance is the wrong question (quantized models, models with stochastic components, models where small logit shifts don't change the decision), verify the *task*: run your validation set through both engines and compare accuracy, F1, WER — whatever your metric is. A 0.1-point accuracy delta is usually fine; a 4-point drop is the [quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) accuracy cliff and a different investigation. The principle holds: define what "the same" means *numerically*, then prove it.

## Results: the footgun table and the export-path comparison

Two reference tables. First, the footgun catalog as a lookup — symptom, cause, fix, and how loud the failure is (the loud ones are mercies; the silent ones are the dangerous ones).

| Symptom | Cause | Fix | Failure mode |
|---|---|---|---|
| Wrong output on some inputs, no error | Data-dependent `if`/`while` baked by tracing | `torch.export` + `torch.cond`, or lift branch to host | Silent (worst) |
| Shape-mismatch error at a new batch/seq/res | No `dynamic_axes` / `Dim` for that axis | Mark the varying axis dynamic | Loud at runtime |
| "Operator X not supported" at export | Op outside the opset; custom/new op | Rewrite in primitives, or custom symbolic | Loud at export |
| File won't load on device | Export opset > runtime's supported opset | Pin opset to the deployed runtime's floor | Loud at load |
| Numerics drift, traces to one op | Op semantics changed across opset versions | Pin opset; verify per-op | Loud via diff |
| Nonsense only in the C++ app, fine in Python | Preprocessing lived in Python, not captured | Fold into graph or replicate + test host | Silent until prod |
| Unsupported-dtype error or one op on CPU | fp64 constant in the module | Cast constants to fp32 | Loud or slow |
| Output differs at small batch | `model.eval()` forgotten (dropout/BN) | `eval()` before export, assert it | Silent-ish |

Second, the export-path comparison — when each capture method earns its place. The numbers in the "drift risk" column are qualitative because they depend on your model, but the *ordering* is robust and the table is the decision you actually make.

| Path | What it does | Control flow | Shapes | Drift risk | Use when |
|---|---|---|---|---|---|
| `torch.jit.trace` | Records one execution | Baked (dangerous) | Fixed unless told | High if any data-dependent branch | Pure feed-forward, no data-dependent control flow, and you verify |
| `torch.jit.script` | Compiles TorchScript subset | Preserved | Supported | Medium (brittle subset) | Legacy code already on TorchScript |
| `torch.export` | Sound full-graph IR + guards | Preserved via `torch.cond` | Symbolic `Dim` constraints | Low (guards catch violations) | New code; ExecuTorch; the default to reach for |
| `torch.onnx.export` (dynamo) | `torch.export` then lower to ONNX | Preserved (export-backed) | `dynamic_axes` | Low | Targeting ONNX Runtime/TensorRT/OpenVINO |

The takeaway the two tables encode together: prefer `torch.export` (and its dynamo-backed ONNX path) because it converts the *silent* failures into *loud* ones. Tracing's whole problem is that it's quiet. Export's whole value is that it complains.

#### Worked example: a real export-path decision with size and latency

Concretely, for our ResNet-18 spine targeting an ONNX Runtime deployment on a Raspberry Pi 5 CPU. The PyTorch checkpoint is about 45 MB of fp32 weights. Exported to ONNX at opset 17 with dynamic batch, the `.onnx` file is the same ~45 MB (export doesn't compress; that's the [quantization](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) lever's job). The capture cost: `torch.jit.trace` would export in well under a second and *appear* to work — ResNet-18 is pure feed-forward, so tracing happens to be safe here, which is exactly the trap (it lulls you into tracing the *next* model that does have a branch). `torch.onnx.export` with the dynamo path takes a few seconds longer and gives you the guarded, shape-symbolic graph. The numeric diff: max relative error around $10^{-5}$ between PyTorch and ORT on fp32 — well within tolerance, ship it. The lesson in the numbers: capture method changes correctness and robustness, not file size; the file size lever is a different post. Choose capture for *correctness*, choose quantization for *size*, and never confuse the two stages.

## When ONNX is the right hub, and when to go direct

ONNX is a hub, and hubs have a cost: every hop is a chance for an op to map imperfectly. So when is the hub worth it, and when do you skip it and export straight to the target's native format? The decision tree below is the short answer; the reasoning follows.

![A decision tree for choosing an export path branching on whether you need many runtimes or one known target, leading to ONNX, TensorRT, ExecuTorch, TFLite, or Core ML](/imgs/blogs/from-model-to-deployable-artifact-8.png)

**Reach for ONNX when** you need portability across more than one runtime or you don't yet know the final target. If the same model has to run on a Windows desktop (ONNX Runtime CPU), a Linux server (ORT CUDA), and an Intel edge box (OpenVINO), ONNX is exactly the right hub — one export, three runtimes, and ORT's execution-provider system lets one file target many backends. ONNX is also the right hub when you want a stable, inspectable artifact you can checker-validate, version, and diff in CI independent of any one runtime.

**Go direct when** you have a single, known target whose native toolchain is better than the ONNX round-trip. For **Apple devices**, `coremltools` exports straight to Core ML and gets you the Apple Neural Engine and palettization that an ONNX round-trip may not. For **Android phones**, TFLite/LiteRT is the native path to the phone's NN delegate and its hardware quantization. For **PyTorch's own edge runtime**, `torch.export` to **ExecuTorch** keeps you on a single sound IR with no cross-format translation at all — the fewest hops, the least drift. For **NVIDIA**, you usually *do* go through ONNX (ONNX to TensorRT is the well-paved road), but TensorRT is then the runtime, not ONNX Runtime.

The honest framing: ONNX trades a little potential per-target performance and one extra translation hop for portability and a single inspectable artifact. When portability is the requirement, that trade is excellent. When you have exactly one target with a great native exporter, the hub is overhead and you should go direct. This is the same decision the [inference-runtimes-compared](/blog/machine-learning/edge-ai/inference-runtimes-compared) post drills into from the runtime side; here the relevant axis is the *export*: fewer hops means less drift, so direct export to a single known runtime is the lower-risk path when portability isn't needed, and the hub is the right call the moment you have two or more targets.

There's a second axis the tree hides: *who maintains the bridge you're standing on*. The PyTorch-to-ONNX exporter and the ONNX-to-TensorRT importer are both heavily exercised, well-funded paths — bugs get found and fixed because thousands of teams walk them daily. A niche framework's exporter to an exotic runtime might be a half-maintained contribution that breaks on the next op you add. So part of "is the hub worth it" is really "is *this specific edge* of the hub well-trodden?" PyTorch to ONNX to ONNX Runtime is about as paved as it gets; PyTorch to ONNX to a small embedded runtime might be more fragile than going direct with that runtime's own converter. Weigh the *maturity of the specific path*, not just the abstract topology. And whichever path you pick, the verification discipline does not change: the numeric diff against the framework reference is the gate regardless of how many hops the artifact took to get there, because the diff measures the *result*, and the result is the only thing the device actually runs.

A final practical note on artifact hygiene that crosses both paths. Whatever you export, version it: store the export script, the exact opset, the PyTorch and exporter versions, and the input/output names alongside the `.onnx` (or `.mlmodel`, or `.pte`) file, because six months later when the model needs a tweak, "how was this exported?" is a question you do not want to reverse-engineer from a binary. The export is a build step; treat its inputs as build inputs. A reproducible export — same script, same versions, same numeric diff — is the difference between a model you can maintain and a model that becomes load-bearing legacy nobody dares touch. This is the [edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle) discipline applied to the one stage this post owns: the conversion is code, and code that produces shipping artifacts gets pinned, versioned, and tested like any other code.

## Case studies and real numbers

A few grounded data points from the field, framed honestly as approximate where I'm not citing an exact figure.

**The control-flow bug is endemic, not exotic.** PyTorch's own export documentation leads with data-dependent control flow as the canonical reason `torch.export` exists and `torch.cond`/`torch.while_loop` were added — the framework team built guards and structured control-flow operators *specifically because* tracing's silent branch-baking was the most common production conversion failure. The `torch.export` design (proposed and stabilized across PyTorch 2.x) is essentially an answer to "make the silent failures loud." If the team that owns the exporter considers this the headline footgun, believe them.

**Opset mismatch is a real outage class.** The "exported with a newer opset than the serving runtime supports" failure is common enough that ONNX Runtime's documentation maintains an explicit opset-to-runtime-version compatibility matrix, and the standard production advice is to pin the export opset to the deployed runtime's floor. The fix is procedural, not clever: treat the opset as a versioned deployment dependency and gate it in CI.

**Preprocessing-not-captured bites speech and vision pipelines especially.** Whisper.cpp and similar C++ ports re-implement the audio front-end (mel-spectrogram computation) in C precisely because that preprocessing was never in the model graph — it lived in Python in the original. Any team that exports "the model" and forgets the front-end ships a graph that produces nonsense on raw audio. The discipline that catches it is testing the *host* code against the same reference, not just the model.

**The verify-by-diff gate catches what checkers miss.** The pattern of "passed `onnx.checker`, passed unit tests, still wrong" is exactly why the numeric-diff-against-eager step is treated as mandatory in mature deployment pipelines. A structurally valid graph can be a numerically wrong graph; only the diff against the framework reference distinguishes them. Teams that bolt this assertion to CI catch baked branches, train-mode layers, and bad symbolics before they reach a device; teams that don't, ship them.

**Dynamic shapes are a first-class TensorRT concern, which tells you it's a real problem.** TensorRT — the runtime most people reach ONNX for on NVIDIA hardware — has an entire **optimization profile** mechanism for dynamic shapes precisely because static-shape assumptions break in production and dynamic shapes need explicit min/opt/max bounds to plan kernels and memory. When a major runtime vendor builds a dedicated subsystem for the dynamic-shape problem, that is direct evidence the frozen-batch footgun is common enough to engineer around, not a corner case. The export-side discipline (mark exactly what varies) and the runtime-side discipline (declare the profile bounds) are two halves of the same concern.

**fp64 constants are a documented edge-runtime hazard.** Mobile and embedded runtimes commonly support fp32, fp16, and int8 but *not* fp64; the practitioner guidance for TFLite and Core ML conversions routinely warns to keep all constants fp32, and the typical symptom — one op forced to a slow CPU fallback, or an outright unsupported-dtype rejection — matches the mechanism exactly. It is a one-line bug (a Python float promoted to double) with an outsized cost, which is why it earns a row in every conversion checklist.

These six are the recurring incidents I'd put money on hitting any team's first serious edge deployment. None of them are bugs in the tools. All of them are the conversion working as designed, and all of them are caught by the same two disciplines: capture with `torch.export` (so silent failures get loud), and verify with a numeric diff (so the ones that slip through get caught at the gate).

#### Worked example: a conversion-incident postmortem, by the numbers

To make the disciplines concrete, here is the audio front-end from the opening, reconstructed as a postmortem with the numbers that mattered. The model: ~3 MB, a small per-frame gate plus a denoiser, targeting an ARM Cortex-A box running ONNX Runtime. The defect: the data-dependent `if noise_energy.sum() > threshold` was baked by `jit.trace` on a noisy example, so the deployed graph denoised unconditionally. The blast radius: roughly a third of frames (the quiet ones) got the wrong path, which in production audio is audible as the denoiser chewing on silence. Time-to-detect without a numeric diff: two days, via user reports. Time-to-detect *with* the diff we should have run: about thirty seconds — a single negative-energy (quiet) input through both engines would have shown `max_abs` on the order of the denoiser's output magnitude, an obvious percent-plus relative error, far above the $10^{-4}$ gate. The fix: lift the gate decision into the host loop (compute energy in C++, choose which of two small graphs to run), which removed the data-dependent control flow from the captured graph entirely and dropped the per-frame cost because the cheap path no longer ran the denoiser's ops at all. Same accuracy as eager on every input class, verified across noisy *and* quiet adversarial inputs before re-deploy. The whole incident is one sentence of this post made expensive: a dynamic, correct Python decision got frozen to one value at capture, and we didn't diff on an input that would have exposed it.

## When to reach for this (and when not to)

A decisive section, because "always use `torch.export`" is too glib.

- **Use `torch.export` (and the dynamo ONNX path) by default for new code.** It is the path that converts silent failures into loud ones, preserves control flow, and carries shape guards. There is little reason to start a new export on `jit.trace` in current PyTorch.
- **Tracing is fine for pure feed-forward models with no data-dependent control flow** — and you *verify* with a numeric diff. A plain CNN or MLP traces correctly because there's nothing dynamic to bake. The danger is habit: trace the safe model, then trace the unsafe one out of muscle memory. Make the diff non-negotiable and tracing is acceptable for the simple cases.
- **Skip ONNX and go direct when you have exactly one known target with a strong native exporter** — Core ML for Apple, TFLite for Android, ExecuTorch for PyTorch's runtime. The hub is overhead you don't need, and direct export means fewer hops and less drift.
- **Don't fold preprocessing into the graph reflexively.** Sometimes the clean design is a thin captured graph (pure tensor math) with preprocessing in well-tested host code, especially when the preprocessing is platform-specific (a phone's camera pipeline, an embedded ADC). The rule is not "always capture everything"; it's "decide the boundary explicitly and test both sides against the same reference."
- **Don't mark every axis dynamic.** Dynamic shapes can cost graph-optimization opportunities and some accelerators dislike them. Mark exactly what varies; for static-shape accelerators, export fixed-shape variants or use the runtime's profile mechanism.
- **Don't treat `onnx.checker` passing as verification.** It checks structure, not numerics. The numeric diff is the verification. Conflating them is how silently-wrong graphs ship.

## Key takeaways

1. **The trip from Python to a frozen graph is where "it worked in training" breaks.** Capture and export is a distinct, high-risk stage, not a formality.
2. **Tracing records what ran, not what the code says.** It silently bakes one branch of any data-dependent control flow. This is the most dangerous footgun because it fails *silently* and passes tests that only exercise the recorded path.
3. **Prefer `torch.export`.** It produces a sound full-graph IR, preserves control flow via `torch.cond`/`torch.while_loop`, and carries shape guards that turn silent miscompiles into loud errors.
4. **ONNX is a hub, not magic portability.** It turns an $N \times M$ integration problem into $N + M$, but only if you respect the opset contract — pin the export opset to the *deployed* runtime's floor.
5. **`model.eval()` before every export.** Train-mode dropout and batch-norm produce a numerically wrong graph, sometimes only at small batch sizes.
6. **Mark dynamic axes for every dimension that varies** — batch, sequence, resolution — and no more than that.
7. **`onnx.checker` validates structure, not correctness.** Passing the checker proves the file is well-formed, not that it computes the right thing.
8. **The only verification is a numeric diff against the framework reference**, on several inputs including adversarial ones for your known footguns, to a stated tolerance ($\approx 10^{-4}$ relative for fp32). When the diff is too large, bisect per-op to find the first divergent op.
9. **Go direct (Core ML, TFLite, ExecuTorch) for a single known target; use the ONNX hub for portability across runtimes.** Fewer hops means less drift.
10. **Choose capture for correctness, quantization for size.** Export doesn't shrink the model; don't confuse the two stages.

## Further reading

- **PyTorch `torch.export`** — the official tutorial and the `torch.export` design RFC, the authoritative source on the sound-IR model, guards, and `torch.cond`/`torch.while_loop` for control flow.
- **PyTorch ONNX export docs** — `torch.onnx.export` reference, the dynamo-based exporter, `dynamic_axes`, and the custom-symbolic registration API (`register_custom_op_symbolic`).
- **ONNX specification and operator docs** — the ONNX repository's operator (`Operators.md`) and opset-versioning documentation; the canonical definition of the opset contract.
- **ONNX Runtime documentation** — execution providers, the opset-to-runtime compatibility matrix, and graph optimizations; the runtime side of everything in this post.
- **Within this series:** [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame, [the edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle) for where capture sits in the full path, [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization) for what the runtime does to the graph after you hand it over, [inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared) for the runtime-side of the direct-versus-hub decision, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
- **MLOps companion:** [ONNX, from spec to serving](/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving) for the deep dive into what a `.onnx` file contains and how the runtime partitions and serves it.

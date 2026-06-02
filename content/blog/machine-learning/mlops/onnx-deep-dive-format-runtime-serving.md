---
title: "ONNX, From Spec to Serving: A Deep Dive Into the Format, the Runtime, and the Real Cost of Portability"
date: "2026-05-17"
publishDate: "2026-05-17"
description: "A practitioner's deep dive into what a .onnx file actually contains, how ONNX Runtime compiles and partitions it, why exports silently break, and where ONNX wins or loses against TensorRT and vLLM."
tags: ["onnx", "onnx-runtime", "inference", "model-deployment", "quantization", "execution-providers", "tensorrt", "pytorch", "mlops", "llm-serving"]
category: "machine-learning"
subcategory: "MLOps"
author: "Hiep Tran"
featured: true
readTime: 50
---

Everyone who has shipped a model to production has met the same promise: train in PyTorch, export to ONNX, run anywhere. One file, every backend, no framework lock-in. It is a clean story and it sells the format to managers in a single sentence.

It is also wrong in the ways that cost you a weekend.

I have personally watched a "validated" ONNX model pass every check on a laptop and then run 70% of its operators on the CPU of an expensive GPU box, because two layers used an op the GPU execution provider did not claim. I have watched an INT8 quantized model lose four points of accuracy with zero warnings. I have watched a model exported on a Friday refuse to load on Monday because the serving fleet ran an ONNX Runtime release one minor version behind the exporter's opset.

None of these are bugs. They are all *the format working as designed* — and the design is more subtle than "export once, run anywhere." ONNX is not a magic portability layer. It is a precise contract between a serialization format and a graph-compiling runtime, and almost every production incident comes from misunderstanding which half of that contract you violated.

This post is the deep dive I wish I had before my first ONNX deployment. We will take the format apart byte by byte, walk through how ONNX Runtime turns a graph into an execution plan, untangle execution providers and the silent CPU fallback, get exporting from PyTorch right, quantize without lying to ourselves, handle custom operators, and finish with the honest story of ONNX for large language models against vLLM and TensorRT-LLM. Throughout, the spine is one mental model, and every section closes with the second-order gotcha that the documentation tends to leave out.

## Why ONNX is not what most people think it is

Let us start by being precise about the gap between the marketing and the machine.

| Topic | Common assumption | Naive mental model | The reality |
|---|---|---|---|
| Portability | "Export once, run anywhere" | The `.onnx` file is self-contained and backend-agnostic | The file is portable; *correct execution* depends on opset version, runtime version, and which execution provider claims each op |
| The file | "ONNX is a model format like a `.pt` checkpoint" | A serialized weight blob | A protobuf-encoded **dataflow graph** plus weights plus a *version contract* (opset) |
| The runtime | "ONNX runs the model" | An interpreter that walks the graph | ONNX Runtime is a **graph compiler**: it fuses, folds, partitions, and plans before a single inference runs |
| GPU execution | "Set the CUDA provider and it runs on GPU" | The whole graph goes to the GPU | The graph is *partitioned*; unsupported ops silently fall back to CPU with host↔device copies between them |
| Quantization | "INT8 export is a one-liner" | `quantize_dynamic()` and ship | Dynamic vs static vs QDQ change accuracy *and* whether your EP can fuse the result; calibration data quality decides the outcome |
| LLMs | "Export the transformer and serve it" | `torch.onnx.export(model)` works | Generic export produces a graph with no KV cache; you need cache tensors as graph I/O, transformer fusions, and a decode loop outside the graph |

Every row of that table is a section of this post. The thread connecting them: **ONNX is two artifacts wearing one name.** There is the *format* — a static, serialized description of a computation. And there is the *runtime* — a piece of software that reads that description, compiles it, and dispatches it onto silicon. Confusing the two is the root cause of nearly every "but it worked on my machine" ONNX story.

> The `.onnx` file does not run anything. It is a blueprint. The runtime is the contractor, and contractors interpret blueprints differently depending on the tools they own.

The format being stable does not make execution stable, any more than a valid architectural drawing guarantees the building gets built the same way by two different crews. Once you internalize that split, the rest of ONNX stops being surprising.

## The mental model

![Diagram showing ONNX split into the .onnx format on the left producing a dataflow graph and initializers, and ONNX Runtime on the right doing optimization, fusion, partitioning and dispatching to hardware execution providers](/imgs/blogs/onnx-deep-dive-format-runtime-serving-1.png)

The diagram above is the mental model, and the entire article is a tour of it. On the left is the **format**: a `.onnx` file is a protobuf that encodes a dataflow graph (the operators and how tensors flow between them), tagged with an opset version, plus the initializers (the weight tensors). On the right is **ONNX Runtime**: it loads that file, runs optimization and fusion passes, partitions the graph across execution providers, and finally dispatches the partitioned work onto whatever hardware is available — CPU, a CUDA GPU, an NPU, an edge SoC.

Two consequences fall straight out of this picture, and they matter more than anything else in this post.

First, **the format and the runtime version independently.** A `.onnx` file declares an opset version — the version of the operator specifications it was written against. ONNX Runtime declares the *range* of opsets it supports. When those disagree, you get a load error or, worse, a subtly different numerical result. Most people version-pin their PyTorch and never think about ONNX Runtime; that is exactly backwards.

Second, **"runs on GPU" is a property of the partition, not the file.** Whether your model actually runs on the GPU depends on whether the CUDA or TensorRT execution provider can claim every operator in the graph. The file has no say in this. We will spend a whole section on it because it is the single most expensive misunderstanding in ONNX deployment.

Hold this two-sided picture in your head. Everything below is either *the left half* (what is in the file) or *the right half* (what the runtime does with it).

## 1. The IR — what is actually inside a `.onnx` file

**Senior rule of thumb: before you debug an ONNX model, look at it. The file is plain protobuf, fully introspectable, and ten minutes with `onnx.load` saves you a day of guessing.**

ONNX defines an *intermediate representation* (IR). The IR is specified as a set of [Protocol Buffer](https://protobuf.dev/) messages, which is why a `.onnx` file is a binary protobuf blob and not, say, JSON or a pickle. Protobuf gives ONNX three things it badly needs: a compact binary encoding, a strict schema with forward/backward-compatibility rules, and language-neutral parsing — you can read a `.onnx` file from C++, Python, Rust, or JavaScript without the producing framework installed.

![Diagram of the nested protobuf structure of a .onnx file: a ModelProto wrapping a GraphProto wrapping NodeProto nodes and TensorProto initializers](/imgs/blogs/onnx-deep-dive-format-runtime-serving-2.png)

The figure shows the nesting. The outermost message is the **`ModelProto`**. It carries metadata about the model as a whole: the `ir_version` (which version of the IR spec the file uses), the `producer_name` and `producer_version` (what exported it — `pytorch 2.4.0`, `tf2onnx 1.16`, and so on), the `opset_import` list (more on this in the next section, and it is the most important field in the file), and a single `GraphProto`.

The **`GraphProto`** is the model proper. It holds:

- **`input`** and **`output`** — lists of `ValueInfoProto`, each naming a tensor and declaring its element type and shape. Shapes may contain symbolic dimensions (a string like `batch` instead of an integer) — those are *dynamic axes*, and they are the difference between a model that accepts any batch size and one frozen to whatever you exported with.
- **`node`** — the list of operators, the actual computation.
- **`initializer`** — the weight tensors, stored as `TensorProto` messages. A trained ResNet-50 is mostly initializers by byte count.
- **`value_info`** — optional type/shape annotations for *intermediate* tensors. Often absent; shape inference fills it in.

Each entry in `node` is a **`NodeProto`** with an `op_type` (the operator name — `"Conv"`, `"MatMul"`, `"Add"`), a list of input tensor names, a list of output tensor names, and a list of `AttributeProto` attributes (the operator's static configuration — a `Conv`'s `strides`, `pads`, `dilations`). Critically, **nodes are connected by string names, not pointers.** Node A writes a tensor named `"hidden_42"`; node B reads `"hidden_42"`. That shared name *is* the edge. There is no explicit ordering — the graph is a pure dataflow graph, and execution order is anything consistent with the data dependencies.

![Graph diagram of a small ONNX dataflow graph: input X and weight initializer W feed a Conv, then BatchNorm, then Relu, then an Add that also takes a skip connection from X, producing output Y](/imgs/blogs/onnx-deep-dive-format-runtime-serving-3.png)

The figure makes this concrete with a tiny residual block. Notice three things that trip up newcomers. First, **initializers are first-class graph inputs.** The weight tensor `W` is not "inside" the `Conv` node — it is a named tensor the `Conv` consumes, exactly like the activation `X`. This is why constant folding (section 3) works so cleanly: the runtime can tell at a glance which inputs are constants. Second, **a tensor can fan out to multiple consumers.** `X` feeds both the `Conv` and the residual `Add` — that is the skip connection, and in the protobuf it is just the same string name appearing in two nodes' input lists. Third, **the graph carries no execution order.** The runtime is free to schedule `Conv` and the skip path however it likes, as long as `Add` runs after both its inputs are ready. That freedom is exactly what lets ONNX Runtime fuse, reorder, and parallelize without changing the result.

This dataflow-graph property is also why ONNX is *analyzable* in ways an eager PyTorch model is not. Because the whole computation is a static graph of named tensors, you can ask global questions — "is this tensor ever consumed?", "what is the longest dependency chain?", "which subgraph depends only on constants?" — and answer them with a graph traversal. Every optimization in section 3 is a graph rewrite enabled by this structure. A `torch.nn.Module`, by contrast, only reveals its computation by *running* it.

Here is what looking inside actually looks like:

```python
import onnx
from onnx import shape_inference

model = onnx.load("resnet50.onnx")

from collections import Counter

onnx.checker.check_model(model)          # structural integrity check
graph = model.graph

for imp in model.opset_import:           # which opsets does this file need?
    # domain "" is the core ONNX operator set; "com.microsoft" etc. are extensions
    print(f"domain={imp.domain or 'ai.onnx':<16} version={imp.version}")

print(f"producer: {model.producer_name} {model.producer_version}")
print(f"ir_version: {model.ir_version}")

for inp in graph.input:                  # which input dims are symbolic?
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    print(f"input {inp.name}: {dims}")

ops = Counter(node.op_type for node in graph.node)   # operator histogram
print(ops.most_common(10))

inferred = shape_inference.infer_shapes(model)       # populate tensor shapes
```

That operator histogram is worth its weight in gold. The first question to ask of any model you did not export yourself is "what ops does it contain?" — because the answer predicts whether a given execution provider will accept it. A model that is 95% `Conv` and `Gemm` will partition cleanly onto almost any backend. A model peppered with `NonMaxSuppression`, `RoiAlign`, `ScatterND`, or custom-domain ops will fragment.

### A note on weights and the 2 GB ceiling

`TensorProto` stores weight data inline by default. Protobuf has a hard 2 GB message-size limit, so any model whose weights exceed 2 GB — most modern LLMs — must use **external data**: the weights live in a sibling file and the `.onnx` file holds only references. When you copy a large ONNX model, copy the `.onnx_data` file with it. A model that loads on your machine and 404s on the serving box is almost always a missing external-data file.

```python
onnx.save_model(   # externalize weights into a sibling file
    model,
    "llm.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="llm.onnx_data",   # this file must travel with llm.onnx
    size_threshold=1024,        # tensors smaller than 1 KB stay inline
)
```

### Second-order optimization: shape inference is not free correctness

ONNX ships a *static shape inference* pass that propagates shapes through the graph. It is genuinely useful — many runtime optimizations and most execution providers need known shapes to plan memory and pick kernels. But shape inference is best-effort: it is implemented per-operator, it does not cover every op, and a single op with unknown output shape poisons everything downstream. If a downstream EP refuses to claim a subgraph "for no reason," the non-obvious cause is often an upstream op that broke the shape-inference chain, leaving the EP unable to plan. Run `shape_inference.infer_shapes` and check whether the tensors near the failure actually have shapes.

## 2. Opsets and operator versioning — the hardest portability problem

**Senior rule of thumb: an ONNX model is not "an ONNX model." It is a model written against a specific opset. The opset is the contract, and contracts have versions.**

This is the section that, if you skip it, you will rediscover at 2 a.m.

ONNX separates two things that frameworks usually conflate: the *IR version* (the version of the protobuf schema — changes rarely) and the **opset version** (the version of the operator *specifications* — changes with every ONNX release). The opset is what `opset_import` in the `ModelProto` declares. When you export with `opset_version=17`, you are saying "every operator in this graph behaves according to the version-17 spec of that operator."

Here is the subtlety that bites everyone: **operators are versioned individually, and their contracts change.** An operator's input count, attribute set, type support, and even numerical behavior can change between opsets. The opset number you import is really a *snapshot* — it pins every operator to its definition as of that opset.

![Timeline showing the Resize operator changing its contract across opset versions 10, 11, 13, and 18, and an ONNX Runtime version rejecting opset 21 as too new](/imgs/blogs/onnx-deep-dive-format-runtime-serving-4.png)

`Resize` is the canonical horror story, and the figure walks its history. At opset 10, `Resize` took only a `scales` input. At opset 11, it gained a `sizes` input and a region-of-interest input, and the attribute set grew — a model using opset-11 `Resize` is structurally a different node from an opset-10 one. Opset 13 broadened type support. Opset 18 added `antialias` and an `axes` attribute. A preprocessing graph that upsamples an image will produce *numerically different pixels* depending on which opset its `Resize` was frozen at. This is not a bug in ONNX; it is the spec evolving. But it means "convert the model to ONNX" is an underspecified instruction.

The failure modes split into two clean cases:

**Opset too new for the runtime.** You export with the latest PyTorch, which defaults to a high opset, and your serving fleet runs an ONNX Runtime release from eight months ago. The runtime supports opsets up to N; your file declares N+2. ONNX Runtime refuses to create the session — a hard, loud failure at load time. Annoying, but honest. The figure's rightmost event is exactly this: a 1.14-era runtime rejecting an opset-21 model.

**Opset too old, silently lossy.** Less obvious. You take an old model at opset 9 and run it on a current runtime. It loads fine. But a downstream tool — a quantizer, a specific EP, a fusion pass — only supports opset 13+. It silently skips the optimization, and you ship a model that is correct but slower than it should be, with no error anywhere.

The fix for both is the same: **pin the opset deliberately, and pin it to what your runtime supports, not to what your exporter defaults to.** Pick the opset from the deployment side of the pipeline.

When you inherit a model at the wrong opset, the `version_converter` can move it:

```python
import onnx
from onnx import version_converter

model = onnx.load("legacy_model.onnx")
current = model.opset_import[0].version
print(f"model is at opset {current}")

converted = version_converter.convert_version(model, target_version=17)
onnx.checker.check_model(converted)   # converter rewrote each op to opset 17
onnx.save(converted, "model_opset17.onnx")
```

The converter is genuinely useful but not omniscient: it covers the core operator set well and custom-domain ops not at all. Treat a successful conversion as "probably fine" and *always* re-run a numerical parity check (we build one in section 5) afterward — the conversion can succeed structurally and still shift outputs.

### Second-order optimization: the opset is a deployment decision, not an export afterthought

The right workflow is to choose a target opset *once*, at the org level, as a function of the oldest ONNX Runtime version in your fleet, and to make every exporter emit that opset. Bake it into CI: a test that loads every model artifact and asserts `opset_import` matches the org target. Opset drift is invisible until it is an outage; a three-line CI check makes it a red build instead.

## 3. ONNX Runtime internals — from graph to execution plan

**Senior rule of thumb: ONNX Runtime is a compiler, not an interpreter. The expensive thinking happens once, at `InferenceSession` creation. After that, inference is cheap and the plan is frozen.**

People treat `ort.InferenceSession("model.onnx")` as "load the file." It is much more than that. Session creation is a multi-stage compilation pipeline, and understanding it explains both why the *first* call is slow and why every call after is fast.

![Pipeline diagram of ONNX Runtime session creation: load the .onnx file, L1 constant folding, L2 node fusion, layout transform, partition to execution providers, allocate and plan, then run inference on a frozen plan](/imgs/blogs/onnx-deep-dive-format-runtime-serving-5.png)

The figure lays out the pipeline. Walk it left to right.

**Load.** The protobuf is parsed into an in-memory graph. Shape inference runs. The graph is checked.

**L1 — basic optimizations.** Semantics-preserving rewrites that need no hardware knowledge. The headline pass is **constant folding**: any subgraph whose inputs are all constants (initializers) is executed *at session-creation time* and replaced by its result. If your model computes `Reshape` targets or attention masks from constant shape arithmetic, that arithmetic vanishes from the inference path. L1 also includes redundant-node elimination (dropping no-op `Identity`, `Dropout` in inference mode, `Cast` to the same type) and trivial algebraic simplification.

**L2 — extended optimizations.** Pattern-based **node fusion**. This is where multi-node patterns collapse into single fused operators. The classics: `Conv` + `BatchNormalization` folds the BN's scale/shift into the conv weights (BN disappears entirely at inference); `MatMul` + `Add` becomes a `Gemm`; the GELU pattern (a specific arrangement of `Div`, `Erf`, `Mul`, `Add`) fuses into one `Gelu` op; the LayerNorm pattern fuses into `LayerNormalization`. Some L2 fusions are EP-specific — they only fire if the target EP has a kernel for the fused op.

**Layout transform.** Hardware wants specific memory layouts. CPUs with certain instruction sets and GPUs prefer different tensor layouts (`NCHW` vs `NHWC`); this pass inserts the necessary layout conversions and tries to cancel adjacent ones.

**Partition.** The graph is divided across execution providers. This is section 4 — it gets its own treatment.

**Allocate and plan.** ONNX Runtime computes a static memory plan: it figures out the lifetime of every intermediate tensor and *reuses buffers* across tensors whose lifetimes do not overlap, the same way a register allocator reuses registers. This is why ONNX Runtime's memory footprint is often dramatically smaller than a naive "one buffer per tensor" execution.

**Run.** Inference executes the frozen plan. No graph rebuilding, no re-optimization, no allocation. The plan is immutable for the session's life.

You control the optimization level explicitly:

```python
import onnxruntime as ort

opts = ort.SessionOptions()

opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.optimized_model_filepath = "model_optimized.onnx"  # dump optimized graph
opts.enable_profiling = True                # emits a chrome://tracing JSON
opts.intra_op_num_threads = 8               # threads within one operator
opts.inter_op_num_threads = 1               # parallel operators

sess = ort.InferenceSession(
    "model.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)
```

Setting `optimized_model_filepath` is the most underused debugging trick in ONNX Runtime. It dumps the post-optimization graph to disk. Load *that* in [Netron](https://netron.app/) and you see exactly what fused and what did not — the difference between the graph you exported and the graph that actually runs.

### Why fusion matters

![Before-and-after diagram: an unfused graph of separate MatMul, Add, and Gelu nodes each launching a kernel and writing to memory, versus a single fused GemmGelu kernel with one launch and one memory pass](/imgs/blogs/onnx-deep-dive-format-runtime-serving-6.png)

Fusion is not a micro-optimization; on modern hardware it is often *the* optimization. The figure contrasts the two cases. Run `MatMul`, `Add`, and `Gelu` as three separate nodes and you pay for three kernel launches and — this is the expensive part — three round-trips through memory. Each op reads its input from global memory (DRAM or GPU HBM) and writes its output back. The intermediate tensors between `MatMul` and `Add` and `Gelu` are large; shuttling them to memory and back is pure overhead.

Most deep-learning operators are **memory-bandwidth bound**, not compute bound. A `Gelu` does a trivial amount of arithmetic per element; its cost is almost entirely the read and the write. Fuse `MatMul` + `Add` + `Gelu` into one `FusedGemmGelu` kernel and the intermediates never leave fast on-chip memory (registers, shared memory, cache). One launch, one read of the inputs, one write of the final output. On a GPU the speedup for transformer-shaped workloads is routinely 1.3–1.6×, and it stacks across every layer.

This is also why opset and EP choice are not academic. A fusion only fires if the runtime recognizes the pattern *and* the target EP has a kernel for the fused result. Block the fusion — by a stray `Cast` node breaking the pattern, by an unsupported opset, by an EP without the kernel — and you silently pay the unfused price.

### Second-order optimization: the cold-start tax

Every optimization pass runs at session creation. For a large model with TensorRT in the mix (which compiles CUDA engines), session creation can take *minutes*. In a serving system this is the cold-start tax, and it has three mitigations: persist the optimized model and load that on subsequent starts; use the EP's engine/kernel cache (TensorRT writes compiled engines to disk — point it at a persistent volume); and never put session creation on the request path — build sessions at startup, behind a readiness probe, and keep them warm.

## 4. Execution providers — where "runs on GPU" is decided

**Senior rule of thumb: the execution provider list is an ordered preference, not an assignment. ONNX Runtime asks each provider, in order, "which of these nodes can you run?" — and whatever no one claims goes to the CPU.**

An **execution provider** (EP) is a backend: a library of operator kernels plus a capability description, targeting a particular piece of hardware. The important EPs:

| EP | Target | Strength | Watch out for |
|---|---|---|---|
| `CPUExecutionProvider` | Any CPU | Always available, supports every op, the fallback floor | Slowest for large models; the thing your GPU model lands on by accident |
| `CUDAExecutionProvider` | NVIDIA GPU | Broad op coverage, per-op GPU kernels | Per-op dispatch leaves cross-op fusion on the table |
| `TensorRTExecutionProvider` | NVIDIA GPU | Compiles subgraphs into fused TensorRT engines, fastest on NVIDIA | Long engine build (cold start); narrower op coverage → more fallback |
| `OpenVINOExecutionProvider` | Intel CPU/iGPU/NPU | Best on Intel silicon | Intel-only |
| `CoreMLExecutionProvider` | Apple Silicon | Uses the ANE/GPU on Mac and iOS | May downcast to FP16; partial op coverage |
| `DmlExecutionProvider` | Any DirectX 12 GPU | Vendor-neutral GPU on Windows | Windows-only |

You pass EPs as an ordered list. The order is a *priority*:

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    "model.onnx",
    providers=[
        # 1st choice: compile what you can into TensorRT engines.
        ("TensorRTExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "/var/cache/trt",  # persist across restarts
        }),
        # 2nd choice: anything TRT rejects, try plain CUDA kernels.
        ("CUDAExecutionProvider", {"device_id": 0}),
        # Floor: CPU claims whatever is left. Always implicitly last.
        "CPUExecutionProvider",
    ],
)

print("active providers:", sess.get_providers())  # did GPU EPs get the work?
```

### Partitioning and the silent CPU fallback

![Graph diagram of execution-provider partitioning: a full ONNX graph goes through an EP capability scan that claims a TensorRT subgraph and a CUDA subgraph for the GPU, rejects unsupported ops to a CPU fallback that incurs host-to-device copies, and merges all outputs](/imgs/blogs/onnx-deep-dive-format-runtime-serving-7.png)

Here is the mechanism, and the figure is the whole story. At partition time, ONNX Runtime walks the EP list in order. For each EP it asks: *which contiguous subgraphs of nodes can you take?* The EP returns the subgraphs it has kernels for. Those nodes are assigned to that EP and removed from consideration. The next EP gets asked about what remains. The `CPUExecutionProvider` is always implicitly last, and it supports every operator — so it is the floor that catches everything no one else claimed.

The result is a graph **partitioned into islands**. Some islands run on TensorRT, some on CUDA, some on CPU. And here is the cost the marketing never mentions: **at every boundary between a GPU island and a CPU island, the runtime inserts a host↔device memory copy.** Tensors move from GPU HBM across PCIe to host RAM and back.

This is why a model that is "on the GPU" can be slow. Suppose your model is 200 GPU-friendly ops with one `NonMaxSuppression` in the middle that the CUDA EP does not support. The graph partitions into: GPU island → copy to host → CPU runs one op → copy back to GPU → GPU island. You have inserted two PCIe transfers *and* serialized the GPU behind a CPU op. One unsupported operator in the wrong place can cost more than all the others combined.

The defenses are concrete:

1. **Always assert on `get_providers()` after creating the session.** If you asked for CUDA and only see `CPUExecutionProvider`, the GPU EP failed to load (usually a CUDA/cuDNN version mismatch) and you are running fully on CPU. Make this assertion a hard failure in your service's startup.
2. **Profile the partition.** Enable profiling and read which EP each node landed on. ONNX Runtime can log node placement; a handful of CPU nodes scattered through a GPU graph is a red flag.
3. **Eliminate the offending op at export time.** If one op forces fallback, the fix is upstream: replace it with EP-supported ops, move it out of the graph into pre/post-processing, or pick an opset where the GPU EP *does* support it.

### Reading the partition explicitly

The profile shows you *where* time went; to see the partition itself — which EP claimed which node — turn on verbose session logging and read it directly:

```python
opts = ort.SessionOptions()
opts.log_severity_level = 0   # 0 = VERBOSE: logs node-to-EP assignment

sess = ort.InferenceSession(
    "model.onnx", sess_options=opts,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

The verbose log now lists each node and the EP that claimed it — grep it for `Node placements` and for any node assigned to `CPUExecutionProvider`. Every such node is a partition boundary and a candidate host↔device copy.

A healthier and quieter habit is to settle the question at *build* time, before the model ever reaches a serving box. The number of nodes a given EP can claim is a deterministic function of the graph and the EP version — so make it a CI assertion. Load the model with only the GPU EP, count how many nodes fall through to the implicit CPU floor, and fail the build if that count exceeds a threshold you set deliberately (often zero for a model that *must* be fully on the GPU). This converts "the model is mysteriously slow in production" into "the build is red because three nodes do not have CUDA kernels" — a diagnosis delivered to the engineer who can still cheaply fix it, weeks before a customer notices. The partition is not a runtime surprise; it is a property you can test.

One more EP subtlety worth internalizing: **EPs are not interchangeable even when both "support" an op.** The CUDA EP and the TensorRT EP can both run a `Conv`, but TensorRT will fuse that `Conv` with its surrounding `BatchNorm` and activation into a single tactic-selected kernel, while the CUDA EP dispatches a standalone cuDNN call. Same op, same result, very different performance — which is why the EP *order* in your provider list is a real tuning knob, not boilerplate. Putting TensorRT first means "fuse aggressively, accept a long engine build"; putting CUDA first means "skip the build, accept per-op dispatch." Benchmark both on your model; the winner is not always the one the marketing predicts.

### Second-order optimization: I/O binding

By default, ONNX Runtime copies your input from host to device at the start of every inference and copies the output back at the end. If your inputs already live on the GPU (the output of a GPU preprocessing step, or the previous decode iteration of an LLM), those copies are pure waste. **I/O binding** lets you hand the runtime pre-placed device tensors and receive device tensors back:

```python
io_binding = sess.io_binding()
io_binding.bind_input(
    name="input", device_type="cuda", device_id=0,
    element_type=np.float16, shape=tuple(gpu_tensor.shape),
    buffer_ptr=gpu_tensor.data_ptr(),
)
io_binding.bind_output(name="logits", device_type="cuda", device_id=0)
sess.run_with_iobinding(io_binding)
```

For LLM decoding, where the KV cache is large and lives on the GPU across hundreds of steps, skipping those copies is not a micro-optimization — it is the difference between competitive and embarrassing latency.

## 5. Exporting from PyTorch without lying to yourself

**Senior rule of thumb: export is a lossy translation. Your job is not to "export the model" — it is to export it and then *prove* the ONNX graph computes the same function as the PyTorch one.**

Most ONNX pain originates here, at the export boundary, because PyTorch is an eager imperative framework and ONNX is a static graph. Bridging that gap necessarily loses information, and *which* information it loses depends on which exporter you use.

![Diagram of two export paths from a PyTorch nn.Module: the TorchScript tracer which runs one example and risks freezing control flow into a single branch, and the dynamo exporter which captures the real FX graph, both producing an ONNX graph that feeds a parity check](/imgs/blogs/onnx-deep-dive-format-runtime-serving-8.png)

The figure shows the two paths PyTorch gives you.

**The TorchScript tracer** (`torch.onnx.export` in its classic mode) runs your model once on an example input and records the operations that *actually executed*. It is robust and well-worn. Its fatal weakness is right there in the description: it records what executed *on that input*. Any Python `if` whose condition depends on tensor *values* is resolved to whichever branch the example took, and the other branch vanishes from the graph. A `for` loop over a dynamic length is unrolled to the length it saw. The tracer does not warn you — it produces a perfectly valid ONNX file that silently implements a *special case* of your model.

**The dynamo-based exporter** (`torch.onnx.export(..., dynamo=True)`, the strategic direction for modern PyTorch) uses `torch.export` / FX graph capture. It captures real control flow into ONNX control-flow operators (`If`, `Loop`) instead of flattening it, and it handles dynamic shapes more faithfully. It is the right default for new work; the tracer remains the fallback for models the new path cannot yet capture.

A correct export:

```python
import torch

model = MyModel().eval()  # eval() matters: BatchNorm/Dropout behave differently
example = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    (example,),
    "model.onnx",
    input_names=["input"],
    output_names=["logits"],
    # WITHOUT this, batch is frozen to 1 — the single most common export bug.
    dynamic_axes={
        "input":  {0: "batch"},
        "logits": {0: "batch"},
    },
    opset_version=17,          # pinned to the deployment fleet, not the default
    do_constant_folding=True,
    dynamo=True,               # prefer the FX-based exporter
)
```

The `dynamic_axes` argument is not optional polish. Leave it out and every shape in the graph is frozen to your example input's shape. You export with batch size 1, deploy, and discover the model rejects every batched request in production. Declare which axes are symbolic, by name, for every input and output.

### Proving parity — the step everyone skips

Exporting is half the job. The other half is proving the ONNX graph is the *same function*. This is non-negotiable, and it is fifteen lines:

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

max_abs_diff = 0.0
for _ in range(32):  # several random inputs, not one
    x = torch.randn(4, 3, 224, 224)  # batch=4 exercises the dynamic axis

    with torch.no_grad():
        torch_out = model(x).numpy()
    onnx_out = sess.run(None, {"input": x.numpy()})[0]

    max_abs_diff = max(max_abs_diff, np.abs(torch_out - onnx_out).max())

print(f"max abs diff over 32 inputs: {max_abs_diff:.2e}")
assert max_abs_diff < 1e-4, "export changed the function — investigate before shipping"
```

Run this in CI on every export. A few times `1e-6` is the floating-point noise of reordered operations and is fine. `1e-2` means something fused wrong, a branch got frozen, or an op's opset behavior differs — and you want to find that here, not from a metrics regression in production. Vary the batch size in the test so you actually exercise the dynamic axis; a model that is correct at batch 1 and broken at batch 4 is a classic frozen-shape bug.

### Second-order optimization: simplify before you ship

PyTorch exporters emit graphs with cruft — redundant `Cast`s, `Identity` chains, shape-arithmetic subgraphs that constant-fold away. The community tool [`onnx-simplifier`](https://github.com/daquexian/onnx-simplifier) (`onnxsim`) runs constant folding and cleanup *offline*, so the runtime does not redo it on every cold start, and — this matters — a cleaner graph exposes more fusion patterns. A stray `Cast` between `MatMul` and `Add` blocks the `Gemm` fusion; `onnxsim` removes it and the fusion fires. Simplify, then re-run the parity check.

## 6. Quantization on ONNX — INT8 without the accuracy cliff

**Senior rule of thumb: quantization is not a switch you flip. It is a numerical approximation, and whether it is free or catastrophic depends entirely on calibration and granularity.**

Quantization stores weights and activations in INT8 instead of FP32, cutting memory 4× and letting the hardware use integer arithmetic units that are far faster and more power-efficient. The catch: INT8 has 256 values. Mapping a continuous distribution onto 256 buckets loses information, and the *art* is losing it where the model does not care.

![Grid diagram comparing three ONNX quantization modes: dynamic with scales computed at runtime and no calibration set, static QDQ with scales frozen by calibration and Q/DQ pairs wrapping every tensor, and operator-oriented quantization using QLinearConv and QLinearMatMul](/imgs/blogs/onnx-deep-dive-format-runtime-serving-9.png)

ONNX Runtime gives you three modes, and the figure lays out how they differ — which comes down to *where the scale factors live and when they are computed*.

**Dynamic quantization.** Weights are quantized to INT8 offline. Activation scales are computed *at runtime*, per inference, from the actual tensor passing through. No calibration data needed — it is genuinely a one-liner. The per-inference scale computation costs a little, and it shines on models dominated by `MatMul` with small activations: transformer encoders, BERT-style models. For convolutional vision models it usually underperforms static.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx", "model.int8.onnx",
    weight_type=QuantType.QInt8,
)
```

**Static quantization.** Both weights *and* activations are quantized offline. Activation scales are frozen ahead of time by running the model on a **calibration dataset** — a few hundred representative inputs — and observing the activation ranges. No per-inference scale cost, so it is the fastest mode at inference. Its quality is entirely a function of calibration data quality: calibrate on data that does not match production and your frozen ranges clip real activations.

```python
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat,
)

class Reader(CalibrationDataReader):
    def __init__(self, samples):           # samples: representative inputs
        self.it = iter([{"input": s} for s in samples])
    def get_next(self):
        return next(self.it, None)

quantize_static(
    "model.onnx", "model.int8.onnx",
    calibration_data_reader=Reader(calibration_samples),
    quant_format=QuantFormat.QDQ,          # QDQ vs QOperator — see below
    weight_type=QuantType.QInt8,
    per_channel=True,                      # almost always the right call
)
```

The `quant_format` argument picks between the two *representations* of a statically quantized graph:

- **QDQ (Quantize-DeQuantize).** The graph keeps its original FP32 operators and inserts explicit `QuantizeLinear` / `DequantizeLinear` node pairs around tensors. The graph still "looks" FP32, with Q/DQ pairs marking quantization boundaries. The EP is responsible for recognizing `DQ → Op → Q` patterns and fusing them into real integer kernels. This is the portable, recommended format — TensorRT, the CPU EP, and others all understand QDQ — and it is the left-vs-middle distinction in the figure.
- **QOperator (operator-oriented).** Quantized operators are baked directly into the graph as dedicated ops: `QLinearConv`, `QLinearMatMul`, `QLinearAdd`. No Q/DQ pairs to fuse. Useful for EPs that lack QDQ-fusion logic, but less portable.

### The granularity decision: per-tensor vs per-channel

The single most important quantization flag is `per_channel`. **Per-tensor** quantization uses one scale factor for an entire weight tensor. **Per-channel** uses a separate scale per output channel. Weight distributions vary enormously across channels — one filter's weights might span ±0.1, another's ±2.0. Force them onto a shared scale and the small-range channels get crushed into a handful of INT8 buckets, destroying their precision. Per-channel costs almost nothing (a vector of scales instead of a scalar) and routinely recovers most of the accuracy lost to per-tensor. **Default to `per_channel=True`.** Per-tensor quantization is the cause of most "INT8 tanked my accuracy" stories.

It is worth working the arithmetic, because the failure mode is quantitative. A symmetric INT8 quantizer maps a float value $x$ to an integer via $q = \mathrm{round}(x / s)$, where the scale $s = \max(|x|) / 127$ — the largest magnitude in the tensor mapped to the largest INT8 value. Dequantization is $\hat{x} = q \cdot s$, and the worst-case error per element is $s/2$.

Now take a per-tensor scale over a weight tensor whose channels span very different ranges. Suppose channel A's weights live in $[-2.0, 2.0]$ and channel B's in $[-0.05, 0.05]$. Per-tensor, the shared scale is $s = 2.0 / 127 \approx 0.0157$. Channel A is fine — its values use the full INT8 range. But channel B's largest weight, $0.05$, quantizes to $\mathrm{round}(0.05 / 0.0157) = 3$. Channel B's *entire* weight distribution is represented by the integers $-3$ to $3$ — seven values out of 256. Six of every eight bits are wasted, and the rounding error of $s/2 \approx 0.008$ is 16% of channel B's own range. The channel is numerically destroyed.

Per-channel quantization gives channel B its own scale, $s_B = 0.05 / 127 \approx 0.00039$, and channel B again uses all 256 levels. The cost is a 128-element vector of scales instead of one scalar — negligible. This is the whole story of why `per_channel=True` is not an optimization but a correctness default for any model whose channels are not pre-normalized.

### Second-order optimization: quantization is an EP-coupled decision

A quantized graph is only fast if the target EP can *execute it as integers*. QDQ format on an EP that does not fuse `DQ → Op → Q` will dequantize back to FP32, run the FP32 kernel, and requantize — slower than the original FP32 model *and* less accurate. Always (1) pick the quant format your deployment EP actually fuses, (2) benchmark the quantized model on the real EP, not just on CPU, and (3) measure accuracy on a real eval set, not eyeballed outputs. INT8 that is correct on three test images and broken on the long tail is the most expensive kind of broken.

## 7. ONNX for LLMs — the honest comparison

**Senior rule of thumb: a transformer is not a vision model. The naive `torch.onnx.export(llm)` gives you a graph that technically runs and is useless for serving, because it has no concept of a KV cache.**

Everything so far applies cleanly to CNNs and encoder models. Decoder-only LLMs break the assumptions, and it is worth being precise about why.

An LLM generates autoregressively: one token at a time, each token attending to every previous token. The naive approach recomputes attention over the whole prefix at every step — quadratic and ruinous. The fix, universal in production, is the [KV cache](/blog/machine-learning/large-language-model/kv-cache): cache the key and value tensors of past tokens and, at each step, compute attention for just the *new* token against the cached keys and values.

A vanilla ONNX export of an LLM has no KV cache. It is a pure function `tokens → logits`. To serve it efficiently, the cache has to become part of the graph's *interface*.

![Graph diagram of an ONNX LLM decode loop: prompt tokens feed a prefill pass that builds the KV cache as past_kv graph input, a decode step consumes one token and past_kv to produce present_kv as a graph output and logits, present_kv is appended and fed as past_kv for the next step until an EOS or max-length stop](/imgs/blogs/onnx-deep-dive-format-runtime-serving-10.png)

The figure shows the shape of a real ONNX LLM. The exported graph takes `past_kv` tensors as **explicit graph inputs** and emits `present_kv` tensors as **explicit graph outputs** — one pair per attention layer. The autoregressive loop lives *outside* the graph, in your serving code: run the graph for one step, take `present_kv` from the outputs, feed it back as `past_kv` next step, sample a token from the logits, repeat until EOS or the length cap. Prefill is the same graph run once over the whole prompt with an empty cache, producing the initial KV state.

This is why you do not hand-export LLMs. The tooling does it for you. [Hugging Face Optimum](https://huggingface.co/docs/optimum/) (`optimum.onnxruntime`) exports transformers with the cache plumbed through correctly, applies transformer-specific fusions (attention fusion, fused LayerNorm, fused GELU), and handles dynamic sequence-length axes. For generation specifically, the [ONNX Runtime GenAI](https://onnxruntime.ai/docs/genai/) library wraps the decode loop, sampling, and KV-cache management so you call `generate()` instead of orchestrating the loop yourself.

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model = ORTModelForCausalLM.from_pretrained(   # KV cache as graph I/O + fusions
    "meta-llama/Llama-3.2-1B", export=True,
    provider="CUDAExecutionProvider",
)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

inputs = tok("ONNX Runtime for LLMs is", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

### Where ONNX Runtime wins and loses for LLMs

Now the honest part. Here is how ONNX Runtime stacks up against the dedicated LLM-serving stacks:

| Dimension | ONNX Runtime | vLLM | TensorRT-LLM |
|---|---|---|---|
| Datacenter GPU throughput | Good | **Best** | **Best** |
| Continuous batching | Limited | **Native** | **Native** |
| PagedAttention-class KV memory | No | **Yes** | Yes |
| CPU inference | **Excellent** | Poor | None |
| Edge / mobile / ARM | **Excellent** | None | None |
| Windows / DirectML GPUs | **Yes** | No | No |
| Non-NVIDIA accelerators | **Yes** (via EPs) | Limited | No |
| Single-stream latency | Good | Good | **Best** on NVIDIA |
| Packaging footprint | **Small, embeddable** | Large | Large |

The pattern is clear. For **high-throughput LLM serving on datacenter NVIDIA GPUs**, vLLM and TensorRT-LLM win, and it is not close. Their advantage is not kernels — it is *serving architecture*: [continuous batching](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) (swapping finished sequences out and new ones in mid-batch) and [PagedAttention-style KV management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) (non-contiguous KV memory that eliminates fragmentation). ONNX Runtime executes a graph extremely well; it does not, on its own, give you a continuous-batching scheduler.

![Grid diagram mapping deployment surfaces to the best runtime: datacenter NVIDIA GPU to vLLM or TensorRT-LLM for continuous batching and paged KV, CPU-only server to ONNX Runtime because vLLM does not run on CPU, edge and mobile and embedded apps to ONNX Runtime for small footprint and per-accelerator execution providers](/imgs/blogs/onnx-deep-dive-format-runtime-serving-11.png)

The figure above is the decision in one picture. ONNX Runtime wins decisively *off* the datacenter NVIDIA path. CPU-only inference: ONNX Runtime is excellent and vLLM is not in the conversation. Edge — a phone, a Jetson, an ARM SBC, an Intel NUC: ONNX Runtime runs there and the alternatives do not. Windows desktops via DirectML, non-NVIDIA accelerators via the EP interface: ONNX Runtime, alone. And its packaging footprint is small enough to embed inside a desktop or mobile application, which vLLM's is not.

So the LLM verdict is not "ONNX is worse." It is: **match the tool to the deployment surface.** Serving a 70B model to thousands of concurrent users on an H100 fleet — reach for vLLM or TensorRT-LLM. Running a 1–3B model on-device, on CPU, on an edge box, or inside a shipped application — ONNX Runtime is very likely the only thing that fits, and it is good at it.

## 8. Custom operators — when the standard set is not enough

**Senior rule of thumb: a custom operator is not a feature, it is a deployment liability. Every custom op you add is a shared library you must build, ship, and version-match on every machine that runs the model.**

The standard ONNX operator set is large but finite. Sooner or later you hit a model with an operation that has no standard equivalent: a fused attention variant, a domain-specific signal-processing kernel, a hand-written CUDA op from a research codebase. ONNX handles this through **domains**. The core operator set lives in the empty domain `""` (also written `ai.onnx`). Microsoft's ONNX Runtime ships an extended set under `com.microsoft` — the *contrib operators*, which include the fused transformer ops (`Attention`, `SkipLayerNormalization`, `FastGelu`) that the optimizers target. And you can define your own domain for genuinely custom ops.

The crucial property, and the one teams forget: **a `.onnx` file references operators; it does not contain their code.** A `NodeProto` with `op_type="MyCustomOp"` and `domain="com.example"` is a *name*. The implementation — the actual kernel that runs on CPU or GPU — lives in a shared library that the runtime must load separately. The file is portable; the kernel is a binary dependency.

Registering a custom-op library at session creation looks like this:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.register_custom_ops_library("/opt/myops/libcustom_ops.so")  # ship this .so

sess = ort.InferenceSession(
    "model_with_custom_op.onnx",
    sess_options=opts,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

If that `.so` is missing or built against a different ONNX Runtime ABI, session creation fails with `NOT_IMPLEMENTED: could not find an implementation for <domain>::<op>` — which is exactly the failure in case study 12. There are three sane strategies, in order of preference:

1. **Decompose into standard ops.** Before writing a custom kernel, ask whether the operation can be expressed as a subgraph of standard ops. It is slower than a fused kernel but it is *free portability* — no library to ship, every EP supports it. For anything not on the hot path, this is the right call.
2. **Use the `onnxruntime-extensions` package.** Many "custom" needs — tokenization, text and image pre/post-processing — are already implemented in the [`onnxruntime-extensions`](https://github.com/microsoft/onnxruntime-extensions) library. Registering it is one line and it is maintained for you.
3. **Write and ship a real custom op.** Only when the operation is genuinely novel *and* on the hot path. Then it is a build-system problem: the `.so` must be compiled per platform, version-matched to the runtime ABI, baked into the serving image, and covered by a CI test that loads the model on a clean machine.

The contrib domain deserves a specific warning. The `com.microsoft` operators are *not* part of the ONNX standard — they are an ONNX Runtime extension. A model that uses `com.microsoft::Attention` (because Optimum's transformer optimizer inserted it) will run beautifully on ONNX Runtime and **fail to load on any other ONNX consumer.** That is usually fine — you are deploying on ONNX Runtime anyway — but it means "this is a portable ONNX file" quietly becomes "this is an ONNX Runtime file." Know which one you have shipped.

### Second-order optimization: keep the portable graph as the source of truth

A clean pattern is to keep two artifacts: the *portable* graph (standard ops only, the thing you would hand to a different runtime or archive for the long term) and the *optimized* graph (contrib ops, EP-specific fusions, the thing you actually serve). The optimized graph is derived from the portable one by a reproducible pipeline. When ONNX Runtime ships a better fusion next year, you re-derive — you do not re-export from a PyTorch checkpoint that may no longer reproduce. The portable graph is the source of truth; the optimized graph is a build artifact.

## Cross-cutting concerns

### Benchmarking honestly

A benchmark that does not warm up is fiction. The first inference triggers lazy allocations, EP kernel selection, and cache population. Always discard warmup iterations, run enough timed iterations for a stable percentile, and report **p50 and p99**, not the mean — production cares about the tail.

```python
import time, numpy as np

x = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

for _ in range(20):           # warmup — never timed
    sess.run(None, x)

lat = []
for _ in range(500):
    t0 = time.perf_counter()
    sess.run(None, x)
    lat.append((time.perf_counter() - t0) * 1000)

lat = np.array(lat)
print(f"p50={np.percentile(lat,50):.2f}ms  "
      f"p99={np.percentile(lat,99):.2f}ms  "
      f"throughput={1000/lat.mean():.0f} infer/s")
```

Benchmark on the **real deployment hardware and the real EP**. CPU numbers do not predict GPU numbers; a `g4dn` does not predict an `H100`. And benchmark the model *as it will be served* — same opset, same EP list, same batch shape.

### Accuracy validation as a gate

Every transformation in this post — export, opset conversion, simplification, fusion, quantization — can shift outputs. Each must be gated by a numerical check against a reference, on a real eval set, in CI. The parity snippet from section 5 is the template. The discipline: a model artifact does not advance through the pipeline until it has proven it computes the same function as the stage before it. "It looked fine" is not a gate.

### Profiling the partition

When a model is slower than expected, the answer is almost always in the profile. Enable `enable_profiling`, run a few inferences, and load the emitted JSON in `chrome://tracing`. You see every kernel, its duration, and its EP. The two smells: long stretches of `CPUExecutionProvider` kernels in a model you believe is on the GPU (partition fallback), and `MemcpyToHost` / `MemcpyFromHost` nodes peppered through the timeline (EP-island boundaries). Both point straight at the fix.

### Versioning the whole pipeline, not just the model

The deepest cross-cutting lesson of this post is that an ONNX deployment has *four* version surfaces, and a mismatch on any of them is an incident:

| Surface | What it pins | Failure when it drifts |
|---|---|---|
| Opset version | The operator contracts the file was written against | Load error (too new) or skipped optimizations (too old) |
| ONNX Runtime version | The opsets supported, the fusion kernels, the EP behavior | Session-creation failure, or a numerically different fused kernel |
| EP / driver version | CUDA, cuDNN, TensorRT, the GPU driver | EP silently fails to load → full CPU fallback |
| Custom-op ABI | The ABI the custom `.so` was built against | `NOT_IMPLEMENTED` at session creation |

The reason ONNX deployments feel fragile is that teams version-control exactly *one* of these — the `.onnx` file — and treat the other three as ambient. The fix is to pin all four together as a unit. Concretely: the serving container image fixes the ONNX Runtime version, the CUDA/cuDNN/TensorRT stack, and any custom-op libraries; the model artifact records the opset it was exported at; and a CI job loads the model inside the exact serving image, on the exact target hardware, and runs the parity check. The model and the image are released *together*, as one versioned bundle. A model artifact is never "compatible with ONNX Runtime" in the abstract — it is compatible with one specific, tested image.

A useful discipline: emit a small JSON sidecar next to every model artifact recording the opset, the producer, the ONNX Runtime version it was validated against, the EP list, and the parity-check max-diff. When something breaks six months later, that sidecar is the difference between a five-minute diagnosis and a day of bisecting.

### Observability in production

Inference does not stop being a distributed-systems problem because it is a model. Three signals belong on every ONNX serving dashboard. **Provider placement**, logged once at startup: assert the active providers match what was requested and alert if they do not — this catches the silent CPU fallback before users do. **Per-request latency percentiles**, p50 and p99 separately: a rising p99 with a flat p50 is the signature of occasional CPU-fallback paths or cold TensorRT engines. **Output distribution drift**: log a cheap summary statistic of the output tensor (mean, or argmax entropy) and alert on a regime change — this is what would have caught the fused-LayerNorm `NaN` of case study 7 in minutes instead of a week. None of this is ONNX-specific, and that is the point: a model in production is a service, and it earns the same instrumentation as any other service.

## Case studies from production

### 1. The opset-21 model that crashed a 1.14 runtime

A team upgraded to the latest PyTorch, whose exporter defaulted to a higher opset. Local tests passed. On deploy, every pod crash-looped: the serving image pinned `onnxruntime==1.14`, which supported opsets only up to a lower number, and `InferenceSession` raised on creation. The first hypothesis was a corrupted artifact — the file was re-exported twice. The actual root cause was the opset gap, visible in one line: `model.opset_import[0].version` read higher than the runtime's max. The fix was to pin `opset_version` in the export call to the fleet's runtime support level. The lesson: the opset is dictated by the deployment side, never by the exporter's default. A CI check that asserts every artifact's opset against the org target turns this outage into a red build.

### 2. The "GPU" model that ran 70% on CPU

A detection model was deployed with `providers=["CUDAExecutionProvider"]` and GPU utilization sat near 15% while latency was triple the target. The team suspected a small batch size or a slow data loader. Profiling told the truth: the graph had partitioned into a GPU island, a CPU island, another GPU island. A `NonMaxSuppression` and a couple of `ScatterND` ops were not claimed by the CUDA EP, so they fell to CPU — and every island boundary inserted a host↔device copy. The model spent more time copying tensors across PCIe than computing. The fix moved NMS into post-processing outside the graph and re-exported. Lesson: a node-level profile is the first diagnostic for any "GPU model is slow," and `get_providers()` plus placement logging belong in every startup path.

### 3. Dynamic axes — the fixed batch baked in by the tracer

An embedding service exported a model and validated it with single inputs. In production, batched requests failed with a shape-mismatch error. The team assumed an ONNX Runtime bug and considered downgrading. The cause was the missing `dynamic_axes` argument: without it, the tracer froze the batch dimension to the example's value of 1, so the graph literally only accepted batch-1 input. The fix was a one-line `dynamic_axes={"input": {0: "batch"}, ...}` and a re-export. Lesson: every export must declare its symbolic axes, and the parity test must run at a batch size other than 1 — a model correct at batch 1 and broken at batch 4 passes a careless test.

### 4. INT8 quantization that tanked an ASR model

A speech model was statically quantized to INT8 with the default settings. Word error rate jumped several points; the team concluded "INT8 does not work for audio" and reverted. It did work — the default was `per_channel=False`. The model's convolutional weights had wildly different ranges across channels, and per-tensor quantization crushed the small-range channels into a few INT8 buckets. Re-quantizing with `per_channel=True`, plus a calibration set drawn from real production audio rather than the clean test clips, brought WER back to within noise of FP32. Lesson: `per_channel=True` is the default you want, and calibration data must look like production traffic.

### 5. Control flow lost by the tracer

A model had a Python `if` selecting between two decoding strategies based on an input flag. Exported with the classic tracer, the example happened to hit the first branch — so the ONNX graph contained *only* that branch. In production, requests needing the second strategy silently got the first, with no error: just quietly wrong outputs that a metrics dashboard caught a week later. The fix was the dynamo-based exporter, which captured the `If` as a real ONNX control-flow operator. Lesson: the tracer records what *ran*, not what your model *is*; any value-dependent control flow demands the dynamo exporter and a parity test that exercises *both* branches.

### 6. TensorRT EP engine-cache cold-start spike

A service using the TensorRT EP saw the first request after every deploy take over a minute and time out the readiness probe. The TensorRT EP compiles graph subgraphs into TensorRT engines, and that compilation happens at session creation. The team's container had no persistent cache, so every pod rebuilt every engine from scratch. The fix was two settings — `trt_engine_cache_enable: True` and `trt_engine_cache_path` pointed at a mounted persistent volume — plus moving session creation behind the readiness probe so traffic never hit a cold session. Lesson: compile-heavy EPs need a persistent engine cache and warmup before readiness; cold start is a configuration problem, not a latency law.

### 7. NaN from a fused LayerNorm on an old runtime build

A transformer encoder started emitting `NaN`s for a fraction of inputs after a model update, while the same `.onnx` file was clean on a developer laptop. The laptop and the serving fleet ran different ONNX Runtime patch versions, and a known bug in the older build's fused `LayerNormalization` kernel mishandled a numerical edge case the new weights happened to exercise. Setting `graph_optimization_level` to `ORT_ENABLE_BASIC` disabled the L2 fusion and made the symptom vanish — confirming the fused kernel as the culprit — and upgrading the runtime fixed it properly. Lesson: the runtime version is part of the model's behavior; pin it across dev and prod, and toggling the optimization level is a fast way to bisect a fusion bug.

### 8. CoreML EP silently downcasting to FP16

An iOS app embedded an ONNX model and used the CoreML EP to reach the Apple Neural Engine. Outputs differed subtly from the server's CPU reference — enough to fail a strict equality test. Nothing was broken: the CoreML EP, to use the ANE, runs much of the graph in FP16, and FP16 has less mantissa precision than FP32. The "fix" was to correct the *expectation*: the test moved from exact equality to a tolerance appropriate for FP16, and accuracy was validated against a task metric on-device rather than against bit-identical server outputs. Lesson: EPs make hardware-driven precision choices; validate quality with task metrics and per-EP tolerances, not cross-device bit equality.

### 9. The ONNX LLM that lost to vLLM — and the case where it still won

A team exported a 7B chat model to ONNX and benchmarked it against vLLM on the same A100, expecting portability with comparable speed. At high concurrency it was roughly 3× behind on aggregate throughput. The gap was architectural, not kernel-level: vLLM does continuous batching and paged KV memory, packing the GPU; the ONNX serving wrapper used static batching and left the device underfilled between requests. For datacenter serving they correctly moved to vLLM. But the same ONNX artifact, in the *next* project, was the winner: a 3B model running on a CPU-only on-prem appliance with no GPU, where vLLM simply does not run. Lesson: ONNX Runtime competes on portability and breadth of deployment surface, not on datacenter-GPU throughput — pick the runtime to match where the model has to live.

### 10. The missing external-data file

A 13B model was exported with `save_as_external_data=True`, validated locally, and pushed to the artifact registry. Every serving pod failed to start with a cryptic protobuf error about a missing file. The team suspected registry corruption and re-uploaded the `.onnx` three times. The actual cause: the `.onnx` file is only a few megabytes of graph structure when weights are externalized — the multi-gigabyte `.onnx_data` sibling holds the real tensors, and the deployment script copied only the `.onnx`. ONNX Runtime opened the graph fine and then failed the instant it tried to materialize an initializer. The fix was to treat the model as a *directory* of artifacts, not a single file, and to checksum the pair together in CI. Lesson: a large ONNX model is two files; whatever moves one must move both, and a load-time test on a clean machine catches the omission before production does.

### 11. The constant-fold that exploded session memory

A segmentation model's `InferenceSession` creation started consuming 30+ GB of RAM and OOM-killing the pod, even though the model on disk was under a gigabyte. The team blamed a memory leak in ONNX Runtime. Profiling session creation showed the spike was inside the L1 constant-folding pass. The exported graph contained a constant subgraph that built a very large intermediate tensor from a `Range` and a broadcast `Expand` — and constant folding dutifully *materialized* that tensor at session creation and baked it into the graph as an initializer. The model was correct but had a multi-gigabyte constant frozen into it. The fix was to restructure the export so the offending tensor was computed from a small input rather than a constant, keeping it out of the fold. Lesson: constant folding trades session-creation memory and disk size for inference speed; when a graph folds a large tensor, the cure is at export time, not a bigger pod.

### 12. The custom op that did not travel

A research team trained a model with a custom CUDA operator, registered it under their own domain, and exported a perfectly valid ONNX file — `onnx.checker` passed, the graph loaded. It ran on their training box and raised `NOT_IMPLEMENTED: could not find an implementation for <domain>::<op>` on every serving node. A `.onnx` file references operators by `(domain, op_type, version)`; it does not *contain* their implementations. The custom-op kernel lived in a shared library that was installed on the training box and absent everywhere else. The fix was to package the custom-op shared library with the serving image and register it via `SessionOptions.register_custom_ops_library`, plus a CI test that loads the model on a clean image. Lesson: the format carries operator *references*, not operator *code* — any non-standard op is a deployment dependency you must ship and test, exactly like a Python package.

## When to reach for ONNX / when not to

### Reach for ONNX Runtime when

- You need **one model artifact across heterogeneous hardware** — CPU, NVIDIA GPU, Intel, Apple Silicon, ARM edge — without maintaining a separate stack per target.
- You are deploying **on CPU**, where ONNX Runtime is genuinely excellent and the GPU-first serving stacks do not apply.
- You are shipping **on edge or mobile** — phones, Jetson, SBCs, NUCs — or **embedding inference inside a desktop or mobile application**, where a small, self-contained runtime is a hard requirement.
- You want **framework decoupling**: train in PyTorch today, keep the option to change training stacks without rewriting serving.
- Your workload is **CNNs, encoder models, or classical pipelines**, where export is faithful, fusion is effective, and there is no KV-cache complication.
- You need **Windows GPU inference** via DirectML, or a **non-NVIDIA accelerator** reachable through an execution provider.

### Skip ONNX (or pair it carefully) when

- You are running **high-throughput LLM serving on datacenter NVIDIA GPUs** — vLLM or TensorRT-LLM win on continuous batching and KV memory management, decisively.
- Your model leans hard on **bleeding-edge or highly custom operators** with no ONNX equivalent — you will fight op coverage and custom-op registration the whole way.
- You are still in **research iteration**, changing architecture daily — export friction will tax every experiment for portability you do not yet need.
- The team has **no capacity for the validation discipline** — opset pinning, parity tests, EP placement checks, quantization accuracy gates. ONNX Runtime is powerful *and* sharp; without the gates, it fails silently and expensively.
- You need **dynamic control flow that even the dynamo exporter cannot capture cleanly** — sometimes keeping the orchestration in Python around smaller exported sub-models beats forcing one monolithic graph.

If there is one habit to take from this post, it is that every ONNX artifact should arrive with evidence. Not "we exported it and it seemed fine," but a recorded opset, a recorded runtime version, a parity-check max-diff against the source model, a node-placement count, and a benchmark on the real hardware. Those five numbers fit in a sidecar JSON smaller than this paragraph, and they convert every failure mode in this article from a production mystery into a build-time assertion. The teams who ship ONNX successfully are not the ones with the cleverest models — they are the ones who treat the export-to-serving path as a pipeline of verifiable stages rather than a single hopeful command.

ONNX is not magic and it is not a trap. It is a precise, two-part system — a versioned format and a compiling runtime — that rewards engineers who respect the contract and punishes those who treat it as a black box. Look inside the file. Pin the opset. Verify the partition. Prove parity after every transformation. Do that, and "export once, run anywhere" stops being a marketing slogan and becomes something close to true.

## Further reading

- [ONNX specification and IR](https://github.com/onnx/onnx/blob/main/docs/IR.md) — the authoritative description of `ModelProto`, `GraphProto`, and the operator versioning rules.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — execution providers, graph optimizations, and the GenAI library.
- [Operator and opset reference](https://onnx.ai/onnx/operators/) — every operator, every version, every contract change.
- [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) — the precision-and-calibration story in depth, applicable directly to ONNX static quantization.
- [Optimizing LLM inference: a complete guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) — continuous batching and the serving-architecture wins ONNX Runtime alone does not provide.
- [INT8 / FP16 / INT4 edge tradeoffs](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) — choosing a precision for the edge targets where ONNX Runtime shines.
- [Deploying LLMs on Jetson AGX Orin](/blog/machine-learning/mlops/deploying-llms-on-jetson-agx-orin) — a concrete edge deployment where the ONNX Runtime story applies end to end.

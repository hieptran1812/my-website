---
title: "TorchTitan: A Production-Ready Framework for Distributed Training at Scale"
publishDate: "2026-03-15"
category: "machine-learning"
subcategory: "Training Techniques"
tags: ["distributed training", "Training techniques", "PyTorch"]
date: "2026-03-15"
author: "Hiep Tran"
featured: false
excerpt: "TorchTitan is a PyTorch-native framework for training large language models using multi-dimensional parallelism. This article breaks down how it unifies FSDP, Tensor Parallel, and Pipeline Parallel into one clean, composable system."
---

Training large language models (LLMs) with billions of parameters requires distributing computation across hundreds or thousands of GPUs. Historically, this involved stitching together multiple parallelism strategies using fragmented, framework-specific code — Megatron-LM for tensor parallelism, DeepSpeed for ZeRO sharding, custom scripts for pipeline scheduling. The result was brittle, hard-to-debug training pipelines with deep vendor lock-in.

**TorchTitan** changes that by providing a PyTorch-native, production-ready framework that composes multiple parallelism dimensions cleanly. It was developed by the PyTorch Distributed team at Meta and is the same system used internally for training Llama 3.1 models up to 405B parameters on 512 H100 GPUs.

In this article, we will cover:

- What TorchTitan is and the problem it solves
- Deep dives into each parallelism strategy
- How the 1D → 2D → 3D → 4D parallelism progression works
- Architecture internals: DeviceMesh, DTensor, and composable APIs
- Memory optimizations, checkpointing, and compilation
- Performance results and ablations
- Practical guide to getting started

## What Is TorchTitan?

TorchTitan is an open-source distributed training framework designed as a reference architecture for training LLMs at scale using **multi-dimensional parallelism** — combining Data Parallel (DP), Tensor Parallel (TP), Pipeline Parallel (PP), and Context Parallel (CP) in a single, composable system.

Unlike Megatron-LM or DeepSpeed that introduce their own abstractions and custom kernels, TorchTitan is built entirely on PyTorch-native APIs (`torch.distributed`, `DTensor`, `DeviceMesh`). This means:

- **No custom CUDA kernels** — everything runs through standard PyTorch ops
- **No forked model code** — models are plain `nn.Module` definitions
- **No vendor lock-in** — uses the same APIs available to every PyTorch user
- **Composable by design** — each parallelism is applied as an independent transformation

The codebase is intentionally minimal (~4K lines of core code) to serve as both a production tool and a learning resource.

## Why Multi-Dimensional Parallelism?

Before diving into TorchTitan's architecture, it is important to understand *why* we need multiple parallelism strategies and when each one becomes necessary.

### The Scaling Problem

Consider training Llama 3.1 405B. With BF16 precision, just storing the model parameters requires:

```
405B params × 2 bytes = 810 GB
```

Add optimizer states (Adam stores 2 extra copies), gradients, and activations, and you need roughly **3-4 TB of memory** — far beyond any single GPU (H100 has 80 GB). Even an 8B model with its optimizer states, gradients, and activations during training can push past the limits of a single GPU when using large batch sizes.

Each parallelism strategy addresses a different bottleneck:

| Strategy | What It Parallelizes | Bottleneck Addressed | Communication Pattern |
| --- | --- | --- | --- |
| **Data Parallel (FSDP)** | Replicates model, splits data batches | Throughput — process more data per step | AllGather + ReduceScatter |
| **Tensor Parallel (TP)** | Splits individual layers across GPUs | Memory — single layers too large for one GPU | AllReduce (intra-node NVLink) |
| **Pipeline Parallel (PP)** | Splits model stages across GPU groups | Communication — reduces cross-node traffic | Point-to-point (peer-to-peer) |
| **Context Parallel (CP)** | Splits long sequences across GPUs | Sequence length — attention grows O(n²) | Ring Attention |

For small models (< 10B), FSDP alone is often sufficient. But as model size grows, you hit walls that require adding dimensions:

- **1D (FSDP only)**: Works well up to ~128 GPUs. Beyond that, collective latency increases linearly with world size.
- **2D (FSDP + TP)**: TP handles intra-node parallelism over fast NVLink, FSDP handles inter-node. Necessary for 70B+ models.
- **3D (FSDP + TP + PP)**: PP transmits only activations between stages via peer-to-peer, drastically reducing cross-node bandwidth. Required for 405B+ models.
- **4D (FSDP + TP + PP + CP)**: CP shards the sequence dimension for long-context training (128K+ tokens).

### A Concrete Example

Training Llama 3.1 405B on 512 H100 GPUs with 3D parallelism:

```
Total GPUs: 512
├── Pipeline Parallel: 16 stages (16 GPUs per pipeline)
├── Tensor Parallel: 8-way (within each node, via NVLink)
└── Data Parallel (FSDP): 4 replicas (across pipeline groups)

512 = PP(16) × TP(8) × DP(4)
```

Each dimension operates on a different "axis" of the GPU cluster, and TorchTitan manages this mapping through its `DeviceMesh` abstraction.

## Architecture Deep Dive

TorchTitan's architecture revolves around three core PyTorch primitives that work together to enable composable parallelism.

### 1. DeviceMesh — The GPU Topology Map

`DeviceMesh` is a multi-dimensional abstraction that represents how GPUs are organized for different parallelism dimensions. Think of it as a coordinate system for your GPU cluster.

```python
from torch.distributed.device_mesh import init_device_mesh

# 2D mesh: 4-way DP × 2-way TP on 8 GPUs
# GPU layout:
#         TP-0    TP-1
# DP-0  [ GPU0,   GPU1 ]
# DP-1  [ GPU2,   GPU3 ]
# DP-2  [ GPU4,   GPU5 ]
# DP-3  [ GPU6,   GPU7 ]
mesh_2d = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))

# Extract sub-meshes for each parallelism dimension
dp_mesh = mesh_2d["dp"]  # Groups: [0,2,4,6], [1,3,5,7]
tp_mesh = mesh_2d["tp"]  # Groups: [0,1], [2,3], [4,5], [6,7]
```

For HSDP (Hierarchical Sharded Data Parallel), DeviceMesh creates a 2D mesh within the data parallel dimension itself — with replica groups on one axis and shard groups on the other. This is useful when you want some degree of replication for faster convergence alongside sharding for memory savings.

For a full 3D setup:

```python
# 3D mesh: 2-way DP × 4-way TP × 2-way PP on 16 GPUs
mesh_3d = init_device_mesh(
    "cuda", (2, 4, 2),
    mesh_dim_names=("dp", "tp", "pp")
)
```

The key insight is that TorchTitan always places TP on the **innermost** dimension (within a single node connected by NVLink) and PP/DP on **outer** dimensions (across nodes connected by slower InfiniBand/RoCE). This hardware-topology-aware placement is critical for performance.

### 2. DTensor — Distributed Tensor with Placement Semantics

`DTensor` (Distributed Tensor) is a tensor type that carries sharding metadata — specifically, *how* it is distributed across a `DeviceMesh`. This eliminates the need for manual `all-gather` and `reduce-scatter` calls.

```python
from torch.distributed.tensor import DTensor, Shard, Replicate

# A tensor sharded along dimension 0 across the TP mesh
# If the full tensor is [1024, 4096] and TP degree is 4,
# each GPU holds a [256, 4096] shard
dtensor = DTensor.from_local(
    local_tensor,
    device_mesh=tp_mesh,
    placements=[Shard(0)]
)

# A replicated tensor — full copy on each GPU
replicated = DTensor.from_local(
    local_tensor,
    device_mesh=dp_mesh,
    placements=[Replicate()]
)
```

**Why DTensor matters**: In FSDP2, each parameter is independently represented as a DTensor with its own sharding specification. This is a major improvement over FSDP1's `FlatParameter` approach, which concatenated all parameters into a single flat buffer. Per-parameter sharding means:

- More efficient memory management (7% lower per-GPU memory vs. FSDP1)
- Better compatibility with `torch.compile`
- Cleaner checkpointing via native state dict support

DTensor also encapsulates both global and local tensor information, preserving single-device semantics. When you perform operations on DTensors, the distributed communication is handled automatically based on the placement annotations.

### 3. Composable Parallelism APIs

The fundamental design principle is that **each parallelism is applied as an independent transformation on a standard `nn.Module`**:

```python
# 1. Start with a vanilla model (initialized on meta device for speed)
with torch.device("meta"):
    model = LlamaModel(config)

# 2. Apply Tensor Parallel — shard attention and MLP layers
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module
)

tp_plan = {
    "attention.wq": ColwiseParallel(),    # Shard output dim
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),    # Shard input dim
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
parallelize_module(model, tp_mesh, tp_plan)

# 3. Apply FSDP (Data Parallel with full sharding)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32  # FP32 for gradient reductions
)
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

# 4. Apply Pipeline Parallel (if needed)
from torch.distributed.pipelining import ScheduleInterleaved1F1B

pipeline_schedule = ScheduleInterleaved1F1B(
    model, pp_mesh, chunks=4
)
```

There is no monolithic "distributed model" wrapper. Each parallelism transformation is composable and independent, which makes it far easier to debug, iterate, and understand what is happening at each layer.

## Deep Dive: Each Parallelism Strategy

### FSDP2 (Fully Sharded Data Parallel)

FSDP is the baseline scaling strategy in TorchTitan. It shards model parameters, gradients, and optimizer states across GPUs, only gathering them when needed for computation.

**How it works during a training step**:

1. **Forward pass**: Each layer calls `AllGather` to reconstruct full parameters, computes the forward pass, then *reshards* parameters to free memory
2. **Backward pass**: Same AllGather-compute-reshard pattern, plus `ReduceScatter` to aggregate gradients
3. **Optimizer step**: Each GPU updates only its shard of the parameters

```
Forward:  [Shard] → AllGather → [Full] → Compute → Reshard → [Shard]
Backward: [Shard] → AllGather → [Full] → Compute → ReduceScatter → [Shard]
```

**A practical optimization**: TorchTitan does not reshard the final transformer block during the forward pass, since it will be immediately re-gathered at the start of the backward pass. This small trick avoids one unnecessary AllGather-Reshard round trip.

FSDP2 improvements over FSDP1:
- Per-parameter DTensor sharding (vs. FlatParameter concatenation)
- 7% lower per-GPU memory requirement
- Better torch.compile compatibility
- Cleaner mixed precision with `MixedPrecisionPolicy`

### Tensor Parallel (TP)

TP splits individual weight matrices across GPUs. For a Transformer's attention layer, the query/key/value projections are sharded column-wise, and the output projection is sharded row-wise. This way, each GPU computes a portion of the attention heads independently.

**Column-wise parallel** (`ColwiseParallel`): Splits the weight along the output dimension. Each GPU computes a slice of the output, which is then concatenated.

**Row-wise parallel** (`RowwiseParallel`): Splits the weight along the input dimension. Each GPU computes a partial result that must be summed via `AllReduce`.

```
ColwiseParallel (Q, K, V projections):
  Input: [batch, seq, hidden]  (replicated)
  Weight: [hidden, heads/TP]   (sharded)
  Output: [batch, seq, heads/TP] (sharded)

RowwiseParallel (Output projection):
  Input: [batch, seq, heads/TP] (sharded)
  Weight: [heads/TP, hidden]    (sharded)
  Output: [batch, seq, hidden]  (reduced via AllReduce)
```

**Why TP is always intra-node**: TP requires frequent AllReduce operations (once per attention layer, once per MLP). These are only fast over NVLink (900 GB/s on H100 NVSwitch) — doing them over InfiniBand (400 Gb/s) would be prohibitively slow.

#### Async Tensor Parallel

TorchTitan supports **AsyncTP**, which fractionalizes matrix multiplications into smaller chunks and overlaps communication with computation:

1. Split the matmul into N chunks
2. Compute chunk 1, start AllGather for chunk 2's input
3. While chunk 2's data transfers, compute chunk 1's reduction
4. Repeat — communication and computation run in parallel

This uses the `SymmetricMemory` abstraction, which allocates a shared memory buffer on each GPU accessible via NVSwitch. On the Llama 3.1 70B at 256 GPUs, AsyncTP provides a **12.59% speedup** over standard TP.

### Pipeline Parallel (PP)

PP splits the model vertically into stages. Each stage contains a subset of transformer layers, and different stages run on different groups of GPUs. The key advantage is that inter-stage communication only involves **point-to-point transfer of activations** — much less data than the all-to-all communication patterns of FSDP.

TorchTitan supports multiple pipeline schedules, each with different memory-throughput tradeoffs:

#### GPipe Schedule

The simplest approach: all microbatches execute forward passes sequentially, then all execute backward passes.

```
GPU0 (Stage 0): [F0][F1][F2][F3][          ][B3][B2][B1][B0]
GPU1 (Stage 1): [    ][F0][F1][F2][F3][B3][B2][B1][B0]

F = Forward, B = Backward
Blank spaces = Pipeline bubble (idle time)
```

**Problem**: Large pipeline bubbles — GPUs sit idle waiting for other stages.

#### 1F1B (One Forward One Backward)

Interleaves forward and backward passes to reduce the bubble:

```
GPU0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][B3][B4][B5]
GPU1: [    ][F0][F1][F2][B0][F3][B1][F4][B2][B3][B4][B5]
```

After the warmup phase, each GPU alternates between one forward and one backward microbatch, keeping both computation and memory utilization more balanced.

#### Interleaved 1F1B

The most efficient schedule in TorchTitan. Each GPU holds **multiple non-contiguous model chunks** (vertical stacking), which allows more overlap and smaller bubbles.

```
# With 2 chunks per GPU:
GPU0 holds: Layers [0-3] and [8-11]
GPU1 holds: Layers [4-7] and [12-15]
```

On Llama 3.1 405B at 512 GPUs, Interleaved 1F1B achieves a **30% throughput improvement** over standard 1F1B (130 vs 100 tokens/sec).

TorchTitan also supports experimental **ZeroBubble** and **Flexible-Interleaved-1F1B** schedules via a pipeline IR that expresses schedules as a list of compute actions, making it easy to experiment with new scheduling strategies.

### Context Parallel (CP)

CP addresses the quadratic memory cost of attention for long sequences. It shards the **sequence dimension** across GPUs, implementing Ring Attention:

```
Sequence length: 128K tokens, CP degree: 4
GPU0: tokens [0:32K]
GPU1: tokens [32K:64K]
GPU2: tokens [64K:96K]
GPU3: tokens [96K:128K]
```

The elegant part: CP is applied as a **Python context manager** during training and requires **no changes to model code**. It dynamically replaces calls to `scaled_dot_product_attention` with CP-aware operations that coordinate across GPUs.

```python
# CP is transparent to the model definition
with context_parallel(cp_mesh):
    output = model(input_ids)  # Attention is automatically distributed
```

CP also extends the DTensor dispatcher to handle causal attention load balancing — ensuring that the triangular mask doesn't create compute imbalance across GPUs.

**Scale demonstrated**: TorchTitan has trained with context lengths up to **262,144 tokens** on 8 H100 GPUs using CP.

## Memory Optimization Techniques

Training large models is often memory-bound before it is compute-bound. TorchTitan provides several techniques to reduce memory pressure.

### Activation Checkpointing

During the forward pass, intermediate activations are stored for the backward pass. For large models, these activations can consume more memory than the model parameters themselves.

TorchTitan supports three granularities:

```toml
[activation_checkpoint]
mode = "selective"  # Options: "none", "selective", "full"
```

- **Full**: Recompute all activations in the backward pass. Maximum memory savings, but ~33% more compute.
- **Selective (op-level)**: Only save results from computation-intensive operations (like matrix multiplications) and recompute cheap operations (like activations, norms). This is the best tradeoff — nearly the same memory savings as full, with much less recomputation overhead.
- **Layer-level selective**: Apply checkpointing to every N transformer blocks, giving you a knob to trade off memory vs. compute.

### Mixed Precision Training

TorchTitan uses a split precision strategy:

- **Forward/backward**: BF16 for parameters and activations (halves memory vs. FP32)
- **Gradient reductions**: FP32 to maintain numerical stability during AllReduce
- **Optimizer states**: FP32 for Adam momentum and variance

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32
)
```

### Float8 Training

On H100 GPUs with FP8 Tensor Cores, TorchTitan integrates with `torchao` to enable Float8 linear layers. This effectively doubles the compute throughput of matrix multiplications.

```toml
[float8]
enable_float8_linear = true
```

Three scaling strategies are available:
- **Dynamic**: Per-tensor scaling adjusted every iteration (most accurate)
- **Delayed**: Updates scaling factors every N iterations (more stable)
- **Static**: Fixed scaling factors (fastest, least flexible)

Float8 is composable with FSDP2, TP, and `torch.compile`, and includes a Float8 AllGather capability so that communication can happen in the compressed format.

**Impact**: On Llama 3.1 8B at 128 GPUs, Float8 combined with torch.compile achieves a **65.08% speedup** over the BF16 baseline.

### Meta Device Initialization

TorchTitan initializes models on a `meta` device — storing only metadata (shapes, dtypes) without allocating any GPU memory:

```python
with torch.device("meta"):
    model = LlamaModel(config)  # Ultra-fast, no memory allocated
```

After parallelism transformations are applied, parameters are materialized with correct sharding layouts via user-defined initialization functions. This means a 405B model can be "created" in milliseconds on a single CPU before being distributed across 512 GPUs.

## Checkpointing and Fault Tolerance

### Distributed Checkpoint (DCP)

TorchTitan uses PyTorch's Distributed Checkpoint system, which understands DTensor sharding layouts. This enables **elastic resharding** — you can save a checkpoint with one parallelism configuration and resume with a completely different one.

```python
# Save with 8-way TP, 4-way DP
save_checkpoint(model, "checkpoint_step_1000/")

# Resume with 4-way TP, 8-way DP — DCP handles the resharding automatically
load_checkpoint(model, "checkpoint_step_1000/")
```

DCP converts DTensor placement information into an internal storage format. On load, it matches stored shards with the current model's DTensor layout, performing any necessary redistribution.

### Asynchronous Checkpointing

Synchronous checkpointing blocks training while data is written to storage. TorchTitan's async mode offloads storage persistence to a separate thread, **reducing checkpointing overhead by 5-15x**.

```toml
[checkpoint]
enable_checkpoint = true
async_mode = "async"     # or "disabled" for synchronous
interval = 1000          # Save every 1000 steps
```

This is critical at scale — a synchronous checkpoint of a 405B model across 512 GPUs can take minutes, during which all GPUs sit idle.

### Flight Recorder

For debugging distributed training failures (the bane of every ML engineer), TorchTitan includes a Flight Recorder that logs start, end, and enqueue times for all collective and point-to-point operations. When an NCCL timeout occurs, you can trace exactly which GPU was stuck and in which communication pattern.

## torch.compile Integration

TorchTitan supports `torch.compile` for kernel fusion and optimization, providing an additional 10-20% speedup on top of parallelism gains.

```toml
[training]
compile = true
```

**Regional compilation**: Rather than compiling the entire model (which would be brittle with distributed ops), TorchTitan compiles individual `TransformerBlock` modules. The compiler detects that all blocks share the same structure and compiles the graph **only once**, then reuses it for every block.

**What the compiler does**:
- Fuses multiple small operations (layernorm + dropout + residual) into single Triton kernels
- Reorders computation and communication to maximize overlap
- Eliminates unnecessary memory allocations and copies

torch.compile is compatible with FSDP2, TP, and DTensor subclasses — one of the key advantages of building on native PyTorch primitives.

## Performance Results

Meta's benchmarks on Llama 3.1 models demonstrate the incremental value of each optimization.

### 1D Parallelism — Llama 3.1 8B on 128 H100 GPUs

| Configuration | Tokens/sec/GPU | Speedup | MFU |
| --- | --- | --- | --- |
| FSDP2 baseline (BF16) | ~5,645 | — | ~52% |
| + torch.compile | ~6,482 | +14.82% | ~56% |
| + Float8 | ~9,319 | **+65.08%** | ~68% |

### 2D Parallelism — Llama 3.1 70B on 256 H100 GPUs

| Configuration | Tokens/sec/GPU | Speedup |
| --- | --- | --- |
| FSDP2 + TP baseline | ~897 | — |
| + Async TP | ~1,010 | **+12.59%** |

### 3D Parallelism — Llama 3.1 405B on 512 H100 GPUs

| Configuration | Tokens/sec/GPU | Speedup |
| --- | --- | --- |
| FSDP + TP + PP (1F1B) | ~100 | — |
| FSDP + TP + PP (Interleaved 1F1B) | ~130 | **+30%** |

### 4D Long Context — Llama 3.1 405B on 512 H100 GPUs

| Context Length | CP Degree | Tokens/sec/GPU |
| --- | --- | --- |
| 32K | 1 | ~76 |
| 128K | 4 | ~32 |
| 262K | 8 | ~16 |

The progressive speedups show that each optimization layer contributes meaningfully, and they compose well together.

## Getting Started

### Installation

**Stable release** (recommended):
```bash
pip install torchtitan
```

**From source** (for latest features):
```bash
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
pip install -r requirements.txt
```

Requires PyTorch nightly or latest stable build with CUDA support.

### Download Tokenizer

```bash
# Requires Llama access from Meta
python scripts/download_hf_assets.py \
    --repo_id meta-llama/Llama-3.1-8B \
    --assets tokenizer \
    --hf_token=YOUR_TOKEN
```

### Running a Training Job

```bash
# Single node, 8 GPUs — trains Llama 3.1 8B with FSDP
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh

# Equivalent torchrun command with custom config
torchrun --nproc_per_node=8 train.py \
    --job.config_file ./torchtitan/models/llama3/train_configs/llama3_8b.toml

# Multi-node (4 nodes, 32 GPUs total)
torchrun --nnodes=4 --nproc_per_node=8 train.py \
    --job.config_file ./torchtitan/models/llama3/train_configs/llama3_70b.toml
```

### Minimal Configuration

A TOML config controls everything — no code changes needed to switch parallelism:

```toml
[job]
dump_folder = "./outputs"

[parallelism]
dp_degree = 4
tp_degree = 2
pp_degree = 1
cp_degree = 1

[model]
name = "llama3"
flavor = "8B"

[training]
batch_size = 8
seq_len = 8192
max_steps = 100000
compile = true

[optimizer]
name = "AdamW"
lr = 3e-4

[float8]
enable_float8_linear = false

[activation_checkpoint]
mode = "selective"

[checkpoint]
enable_checkpoint = true
async_mode = "async"
interval = 1000
```

Setting `dp_degree = -1` (the default) tells TorchTitan to automatically use all available GPUs for data parallelism after accounting for other dimensions.

### Adding a New Model

TorchTitan is model-agnostic. To add a new architecture:

1. **Define the model** as a standard `nn.Module` — no distributed code needed
2. **Create a parallelism plan** specifying TP sharding for each layer
3. **Add a TOML config** with model hyperparameters
4. **Register** in the model registry

```python
from torchtitan.models import model_registry

@model_registry.register("my_model")
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

Then define a parallelism plan:

```python
# parallelize.py
def get_tp_plan(model):
    return {
        "layers.*.attention.wq": ColwiseParallel(),
        "layers.*.attention.wk": ColwiseParallel(),
        "layers.*.attention.wv": ColwiseParallel(),
        "layers.*.attention.wo": RowwiseParallel(),
        "layers.*.ffn.w1": ColwiseParallel(),
        "layers.*.ffn.w2": RowwiseParallel(),
    }
```

### Monitoring

TorchTitan supports both TensorBoard and Weights & Biases for tracking loss curves, throughput, and GPU utilization:

```toml
[metrics]
enable_tensorboard = true
log_freq = 10
```

## When to Use TorchTitan

TorchTitan is a good fit when:

- You are **pre-training** models with 1B+ parameters or doing **full fine-tuning** at scale
- You need **multi-node, multi-GPU** training with composable parallelism
- You want to stay within the **PyTorch ecosystem** without external dependencies
- You need **elastic checkpointing** for changing cluster configurations between training runs
- You want a **clean reference implementation** to learn from or extend

It is **not** designed for:

- **Inference serving** — use vLLM, TGI, or TensorRT-LLM
- **Parameter-efficient fine-tuning** like LoRA — use Hugging Face PEFT or torchtune
- **Single-GPU training** — just use standard PyTorch
- **Non-Transformer architectures** — while technically model-agnostic, the parallelism plans are optimized for Transformer-style models

## TorchTitan vs Other Frameworks

| Feature | TorchTitan | Megatron-LM | DeepSpeed |
| --- | --- | --- | --- |
| PyTorch-native APIs | Yes | No (custom ops, custom kernels) | Partial (custom ZeRO engine) |
| Multi-D parallelism | DP + TP + PP + CP | DP + TP + PP | DP + PP + ZeRO |
| torch.compile | Yes | No | No |
| Float8 support | Yes (H100+) | No | No |
| Elastic checkpointing | Yes (DCP with resharding) | No | Partial |
| Model-agnostic | Yes | Transformer-focused | Yes |
| Async TP | Yes (SymmetricMemory) | No | No |
| Codebase complexity | ~4K lines (clean) | ~50K+ lines | ~100K+ lines |
| Learning curve | Low (standard PyTorch) | High (custom abstractions) | Medium |

**When to choose Megatron-LM**: If you need battle-tested training recipes for very specific model architectures and have teams experienced with its codebase. Megatron is heavily optimized for NVIDIA hardware and has extensive model-specific tuning.

**When to choose DeepSpeed**: If you need ZeRO-Infinity (offloading to CPU/NVMe) or MoE (Mixture of Experts) support, which TorchTitan does not yet provide.

**When to choose TorchTitan**: If you want clean, composable, PyTorch-native distributed training that is easy to understand, modify, and extend. Especially strong for teams that want to stay in the PyTorch ecosystem and benefit from ongoing improvements to `torch.compile`, DTensor, and FSDP2.

## Conclusion

TorchTitan represents a shift toward composable, PyTorch-native distributed training. By building on `DeviceMesh`, `DTensor`, and composable parallelism APIs, it avoids the complexity and lock-in of older frameworks while delivering production-grade performance.

The key design principle — **separation of model definition, parallelism application, and training loop** — means you can write a standard PyTorch model and progressively add parallelism dimensions as you scale. This composability is what makes TorchTitan both a production tool and the best way to learn modern distributed training techniques.

For teams training large models on GPU clusters, TorchTitan provides a clean, well-documented starting point that scales from 8 GPUs to thousands — without leaving the PyTorch ecosystem.

## References

- [TorchTitan GitHub Repository](https://github.com/pytorch/torchtitan)
- [TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training (Paper)](https://arxiv.org/abs/2410.06511)
- [PyTorch Distributed Training Documentation](https://pytorch.org/docs/stable/distributed.html)
- [DTensor Documentation](https://pytorch.org/docs/stable/distributed.tensor.html)
- [Pretrain Llama 3.1 8B with TorchTitan (AMD ROCm Tutorial)](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/torchtitan_llama3.html)
- [TorchTitan on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)
- [TorchTitan on SkyPilot](https://docs.skypilot.co/en/latest/examples/training/torchtitan.html)

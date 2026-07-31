---
title: "FlashKDA: How Kimi Delta Attention Becomes a Fast CUDA Kernel"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "A source-level tour of FlashKDA: why Kimi Delta Attention uses sixteen-token chunks, why the implementation splits into K1 and K2, and how precision and memory movement decide real GPU speed."
tags: ["flashkda", "kimi-delta-attention", "linear-attention", "cuda-kernels", "cutlass", "gpu-optimization", "kernel-fusion", "bf16", "pytorch", "inference"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 27
---

There is a seductive sentence in almost every discussion of linear attention: it is $O(n)$ instead of $O(n^2)$, therefore it should be faster. That sentence is useful for choosing a model family and almost useless for predicting the latency of a real GPU kernel.

The GPU does not execute asymptotic notation. It executes loads, stores, barriers, matrix instructions, register shuffles, and kernel launches. A recurrent attention variant can remove the quadratic score matrix and still lose to a carefully tiled quadratic kernel if its recurrence exposes too little parallelism or moves too much data through shared memory. The interesting engineering question is how the formula is arranged so NVIDIA hardware can consume it.

[FlashKDA](https://github.com/MoonshotAI/FlashKDA) is MoonshotAI's CUTLASS-based CUDA implementation of Flash Kimi Delta Attention. The repository exposes the decisions that usually disappear behind a Python API: a sixteen-token chunk, a two-kernel split, mixed-precision state handling, a Neumann-series inverse, shared-memory unions, and a register-file transpose in the recurrence path.

The diagram above is the mental model: FlashKDA is not a replacement model and not a new serving engine. It is a specialized implementation below `flash-linear-attention`'s `chunk_kda` call. Compatible calls dispatch into a preparation kernel and a recurrence kernel; incompatible calls can fall back to the Triton path.

![The FlashKDA execution path from PyTorch and flash-linear-attention through backend dispatch, K1 preparation, K2 recurrence, and output state.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-1.webp)

## Why FlashKDA is different

| Assumption | Naive view | FlashKDA reality |
|---|---|---|
| Linear attention is automatically fast | $O(n)$ settles the question | Parallelism, memory reuse, and instruction mapping decide the kernel |
| Larger chunks amortize overhead | `CHUNK=64` should be better | `CHUNK=16` keeps the gate range, inverse, and MMA path manageable |
| One fused kernel minimizes launches | Fewer kernels must win | K1 and K2 have different natural grids |
| Lower precision is one global switch | Pick bf16 or fp16 for everything | Each operation gets a deliberate dtype |
| A benchmark speedup is universal | One number transfers everywhere | Shape, GPU, and competitor change the result |

The repository requires SM90 or newer, CUDA 12.9 or newer, and PyTorch 2.4 or newer. That requirement is already a design signal: this is a hardware-specialized fast path, not a generic implementation. FlashKDA is valuable when its assumptions line up with deployment; it is not valuable merely because the name contains “Flash.”

## 1. The computation: Kimi Delta Attention as state evolution

Stop picturing a dense attention matrix. At each token, the layer reads query-like, key-like, value-like, and gating inputs, updates a recurrent state, and emits an output. The state is a matrix associated with a head. It carries information from previous tokens into the current computation.

For a sequence split into chunks, the state is continuous across chunks. The first chunk starts from an optional `initial_state` or zero state. Each chunk produces outputs and a new state. At the end, the caller may request `final_state`. This is why the API has state shapes such as `[B, H, V, K]` for fixed-length sequences and `[N, H, V, K]` for variable-length sequences.

The following is an explanatory abstraction, not a verbatim equation from the repository. It captures the dependency that matters for the kernel:

![Kimi Delta Attention processes tokens through a recurrent state: each chunk reads q, k, v, g, and beta, updates the state, and emits outputs.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-2.webp)

$$
S_{t+1} = \operatorname{update}(S_t, q_t, k_t, v_t, g_t, \beta_t),
\qquad
y_t = \operatorname{read}(S_t, q_t, k_t, v_t).
$$

The consequence is structural. The recurrence introduces a dependency along time, while the preparation work inside a chunk still contains substantial token-level parallelism. FlashKDA follows that split instead of pretending the entire computation has one uniform shape.

The reference implementation in `tests/torch_ref.py` makes the boundary concrete. It uses `CHUNK = 16`, pads a short final chunk, builds cumulative gate values, decays keys and queries, forms a lower-triangular matrix, approximates its inverse, and applies the state update. The CUDA implementation reorganizes those operations around TMA loads, shared-memory layouts, MMA instructions, and two grids.

### What the public API tells us

The `flash_kda.fwd` signature is close to the tensors used by the FLA operation:

~~~python
flash_kda.fwd(
    q, k, v, g, beta, scale, out,
    A_log, dt_bias, lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
)
~~~

The README documents `q`, `k`, `g`, and `out` as bf16 tensors, `beta` as `[B, T, H]`, `A_log` as fp32 `[H]`, and `dt_bias` as fp32 `[H, K]`. The current implementation requires `K = V = 128`. With `cu_seqlens`, `B` must be one, `T` is the concatenated length, and state tensors are indexed by sequence.

That last detail matters. Variable-length batching changes how K1 maps global tiles to sequences. FlashKDA builds a tile-prefix array and uses binary search to map a global tile index to a sequence and local chunk, avoiding an $O(N)$ scan per CTA.

## 2. Why `CHUNK = 16` is a systems decision

Flash Linear Attention uses `CHUNK = 64`; FlashKDA v1 uses `CHUNK = 16`. This is a three-way contract between numerical range, algebraic cost, and instruction-level portability.

![FlashKDA chooses CHUNK=16 by balancing bf16 gate range, 16 by 16 inversion cost, and an SM80 MMA-compatible math path.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-3.webp)

With `lower_bound = -5`, sixteen positions keep the range of $\exp(\operatorname{cumsum}(g))$ within the representable behavior expected by the bf16 path. Larger chunks need more elaborate intra-chunk rescaling. The smaller chunk keeps the numerical problem local.

The chunk computation also builds a lower-triangular matrix and an inverse-like transform. A $16 \times 16$ inverse has a different cost profile from a $64 \times 64$ inverse. FlashKDA uses a Neumann-series expansion. The following is an explanatory abstraction of that implementation:

$$
(I - L)^{-1} \approx I + L^2 + L^4 + L^8.
$$

Here $L$ denotes the chunk-local lower-triangular transform after gating. The code constructs successive powers and accumulates them in fp16. The fixed dimension makes the strategy viable and maps the work to matrix instructions.

The design document also calls out an SM80-only MMA path for the `CHUNK = 16` math. The chunk-level mapping remains simple across modern NVIDIA GPUs, even though the released package requires SM90 or newer for the complete implementation.

A sequence whose length is not divisible by sixteen gets a padded final chunk. Padding is part of correctness: masked positions must not update state as if they were real tokens. Irregular sequence lengths are therefore more informative than one perfectly divisible length.

## 3. Why one fused kernel became two

The first implementation instinct is to fuse everything. FlashKDA's authors tried that and found the more important constraint: the stages do not expose the same parallelism.

![A single fused FlashKDA kernel underutilizes the token-parallel preparation stage, while separate K1 and K2 kernels match each stage to its natural grid.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-4.webp)

| Stage | Natural parallelism | Main work |
|---|---|---|
| K1 | Token-parallel, grid `N × H × num_chunks` | Gate activation, L2 normalization, decay, `L`, `Mqk`, inverse |
| K2 | Head-parallel, grid `N × H` | Chunk recurrence, output projection, state accumulation |

K1 has one CTA for a sequence chunk and head. K2 has one CTA for a sequence and head, then walks through that sequence's chunks. If K1 is forced to share K2's execution shape, abundant token-parallel work becomes bottlenecked by the recurrence's lower parallelism, leaving SMs idle.

The source document reports that splitting the pipeline yielded at least a 15% end-to-end speedup in the authors' comparison. That number is evidence for the diagnosis, not a universal promise. The improvement came from exposing parallelism already present in the computation.

K1 loads `q`, `k`, `g`, `beta`, and `dt_bias`. It applies gate activation, normalization, decay, constructs chunk-local matrices, and stores workspace for K2. K2 loads values and workspace, reads an optional initial state, applies the recurrence, projects outputs, and writes the optional final state.

This is why independent tuning beats aesthetic fusion. K1 cares about occupancy and token tiles. K2 cares about state movement and recurrence throughput. Their best thread counts, shared-memory footprints, and staging policies need not be identical.

## 4. Precision is part of the algorithm

FlashKDA's precision strategy is not “run the model in bf16.” Different operations have different failure modes, so the implementation assigns different representations to different boundaries.

![FlashKDA uses bf16 for stored recurrent state, fp32 FMA for state updates, fp16 for the small inverse, and approximate fp32 math for gating.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-5.webp)

The recurrent state is stored in bf16. The repository describes two benefits: the shared-memory footprint is roughly halved, and an fp32-to-bf16 cast on the critical path of bf16 GEMMs is removed. The state update itself uses fp32 FMA instructions, so the accumulator is updated in a wider format and stored back in bf16 between updates. The source reports no measurable accuracy loss across its inference benchmarks for this choice.

The inverse path uses fp16 rather than bf16. The repository's rationale is that inverse elements are bounded within `[-1, 1]`, so fp16's narrower dynamic range is sufficient. fp16 also avoids a conversion before bf16 MMA and gives the Neumann expansion more mantissa headroom for this small matrix.

The reference test implements sigmoid using CUDA PTX `tanh.approx.f32`:

~~~cuda
float xh = input[idx] * 0.5f;
float th;
asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(xh));
output[idx] = th * 0.5f + 0.5f;
~~~

The identity is $\sigma(x) = \tfrac{1}{2}\tanh(\tfrac{x}{2}) + \tfrac{1}{2}$. The gating path needs a fast sigmoid with sufficient accuracy, not a general-purpose transcendental implementation at maximum precision.

In the `g_act` stage, the implementation rebases the exponent to base two and uses `ex2.approx.ftz.f32`. This removes a change-of-base FMA and uses a higher-throughput instruction. Approximation is not an afterthought here; it is part of the performance contract and must be represented in the correctness reference.

## 5. GPU mechanics: shared memory, registers, and occupancy

The source-level optimization is spread across layouts, unions, launch attributes, and staging code. None changes the public API. Together they determine whether the GPU spends time doing matrix math or moving temporary tensors.

![The K1 and K2 data path keeps hot intermediates in shared memory and registers, reducing unnecessary round trips to global memory.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-6.webp)

K1 reuses shared storage across non-overlapping lifetimes. During the load phase, shared memory holds `q`, `k`, and `g`. During compute, the same region can hold decayed keys, decayed queries, the inverse, `L`, and `Mqk`. The source comments estimate roughly 14 KB saved by this union.

This is manual liveness analysis. The engineer makes phase boundaries explicit and accepts responsibility for synchronization. Similar unions reuse storage for bf16 `g` load targets versus restored keys and fp32 `dt_bias` versus accumulated gate totals.

K1 uses `__launch_bounds__(256, 8)`. The documented optimization trades a small amount of register spilling for more thread blocks per SM. “Avoid all spilling” is not equivalent to “maximize throughput.” If a little register pressure permits enough resident blocks to hide latency, aggregate throughput can improve.

K2 uses `MOVM_T` to transpose operands directly in the register file. The source document describes this as eliminating intermediate shared-memory round trips and shrinking K2's shared-memory requirement. A register-level transpose can remove an entire synchronization and bandwidth event when the data is already close to the desired layout.

TMA is a boundary, not magic. K1 initializes a transaction barrier, expects a precise byte count, issues asynchronous loads, and waits before consuming the data. The value of the protocol comes from overlap and correct synchronization; otherwise it is simply more complicated memory movement.

## 6. Backend integration and dispatch

FlashKDA is designed to be adopted without rewriting every model implementation. After installation, it can be auto-dispatched from `flash-linear-attention`'s `chunk_kda` operation.

![The chunk_kda call branches into a FlashKDA fast path when requirements match, or a debuggable Triton fallback when they do not.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-7.webp)

~~~bash
git clone https://github.com/MoonshotAI/FlashKDA.git flash-kda
cd flash-kda
git submodule update --init --recursive
pip install -v --no-build-isolation .
~~~

For CI or wheel builds, compile all supported architectures:

~~~bash
FLASH_KDA_CUDA_ARCHS=all pip install -v --no-build-isolation .
~~~

The default `auto` mode detects the current CUDA device. A list such as `90a,100a` makes the target explicit. Compiling only for the build host can produce a wheel that installs successfully but contains no usable code for the deployment GPU.

The integration example runs under `torch.inference_mode()` and enables gate, normalization, sigmoid, safe-gate, state-layout, and varlen options:

~~~python
import logging
import torch
from fla.ops.kda import chunk_kda

logging.basicConfig(level=logging.INFO)

with torch.inference_mode():
    out, final_state = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens,
    )
~~~

Logging reports either `kda.chunk_kda -> flashkda` or a rejection reason. That message is operationally important: “the model is using FlashKDA” should be an observed dispatch decision, not an assumption based on package installation.

Setting `FLA_FLASH_KDA=0` forces the Triton path. A fallback switch is useful for correctness bisection, performance comparison, and deployments whose GPU or shapes do not meet the native assumptions.

## 7. Reading the H20 benchmark

The repository benchmark uses `warmup=30`, `iters=200`, and `repeats=5`. It measures `T=8192`, `D=128`, two head counts, and three sequence layouts.

![The H20 benchmark compares FlashKDA across fixed and variable-length layouts, head counts, and two competing FLA kernels.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-8.webp)

For `H=96`, the H20 table reports:

| Case | FlashKDA mean | FLA `chunk_kda` mean | Speedup |
|---|---:|---:|---:|
| Fixed, `T=8192` | 2.6220 ms | 4.8388 ms | 1.85× |
| Varlen `[1300, 547, 2048, 963, 271, 3063]` | 2.3449 ms | 4.8291 ms | 2.06× |
| Varlen `1024 × 8` | 2.0432 ms | 4.6723 ms | 2.29× |

For `H=64`, it reports 1.95× on fixed input, 1.91× on the irregular variable-length case, and 2.31× on eight equal sequences. These are means under the repository's exact configuration, not universal latency guarantees.

The `1024 × 8` case has the same total token count as the fixed workload but eight sequence boundaries. It tests sequence mapping, per-sequence state, and varlen handling. FlashKDA's reported speedup is higher here than in the fixed case for both head counts, which is evidence that the varlen path is not merely a correctness fallback for this shape.

The H20 table also includes `fla_chunk_gated_delta_rule`. At `H=96`, its fixed case is 3.1985 ms versus FlashKDA's 2.6220 ms. This is a different operation, not an interchangeable model baseline. Benchmark tables can contain useful comparisons without implying identical semantics.

The repository's GB200 table reports a different performance profile: at `H=96`, the fixed case is 1.0087 ms for FlashKDA versus 2.3271 ms for `fla_chunk_kda`, while `1024 × 8` is 0.7064 ms versus 2.3105 ms. Hardware labels belong in benchmark prose because the instruction mix, memory system, and CUTLASS path change with the GPU.

## 8. Correctness and numerical boundaries

The repository includes forward correctness tests against a Torch reference and `flash-linear-attention`. The reference mirrors the kernel's important choices: bf16 inputs, fp32 gate operations, approximate sigmoid, base-two exponentials, fp16 inverse math, and bf16 state storage.

![FlashKDA's adoption boundary combines correctness tests, dtype rules, shape constraints, and hardware requirements before the fast path is considered safe.](/imgs/blogs/flashkda-kimi-delta-attention-cuda-kernels-9.webp)

The shape constraints are part of the contract. The current implementation requires `K = V = 128`. In varlen mode, `B` must equal one and states use `[N, H, V, K]`. If initial and final states are both supplied, their dtypes must match. These assumptions are connected to compiled layouts and launch configuration.

Exact-match testing does not mean every intermediate is identical to an fp32 mathematical ideal. The kernel intentionally uses approximate special functions and low-precision storage. “Correct” means acceptably equivalent under the stated inference contract. If the kernel is repurposed for training or unusually sensitive recurrent dynamics, the existing validation is evidence, not blanket authorization.

## Case studies from the repository

### 1. Fixed-length H20 prefill

The fixed benchmark uses `T=8192`, `H=96`, and `D=128`. FlashKDA reports 2.6220 ms against 4.8388 ms for `fla_chunk_kda`, a 1.85× speedup. This is the clean baseline because every sequence has the same length.

The lesson is not simply “FlashKDA is 1.85× faster.” The speedup exists even when variable-length sequence mapping is removed. The two-kernel decomposition, chunk size, precision choices, and memory layout all contribute on the ordinary fixed path.

### 2. Irregular variable-length batching

The irregular case uses lengths `[1300, 547, 2048, 963, 271, 3063]`. The total is still 8192, but boundaries are uneven. At `H=96`, FlashKDA reports 2.3449 ms versus 4.8291 ms, or 2.06×.

This exercises tile-prefix mapping. K1 must map a global tile to the correct sequence, and K2 must maintain a separate state per sequence. A fast fixed batch can still mishandle the last chunk or assign a tile to the wrong state, so this case is a correctness test as much as a speed test.

### 3. Eight equal sequences

The `1024 × 8` case retains eight states while making boundaries predictable. At `H=96`, FlashKDA reports 2.0432 ms against 4.6723 ms, or 2.29×. At `H=64`, it reports 1.3951 ms against 3.2175 ms, or 2.31×.

This is a useful middle point between one long sequence and a fully irregular batch. It can reveal whether the implementation pays a large per-sequence overhead or distributes the work cleanly.

### 4. Head count changes the kernel

Changing `H` from 96 to 64 changes both absolute latency and the relative comparison. On H20, fixed FlashKDA changes from 2.6220 ms to 1.6217 ms, while `fla_chunk_kda` changes from 4.8388 ms to 3.1659 ms.

Head count is a kernel dimension: it affects the launch grid, available parallel work, and state traffic. A kernel tuned for one head count should not be assumed optimal for another.

### 5. H20 versus GB200

The GB200 fixed `H=96` case reports 1.0087 ms for FlashKDA and 2.3271 ms for `fla_chunk_kda`. The equal-length varlen case reports 0.7064 ms and 2.3105 ms. The relative advantage is larger than on H20.

This is why a deployment team should reproduce the benchmark on its actual GPU instead of importing a GB200 number into an H20 capacity plan.

### 6. KDA versus GDN is not a replacement claim

The benchmark includes `fla_chunk_gated_delta_rule` as a comparison. At `H=96`, the fixed H20 case reports 3.1985 ms for GDN versus 2.6220 ms for FlashKDA. This is useful context, but KDA and GDN are not interchangeable operations.

The correct conclusion is narrow: FlashKDA's implementation is competitive with the measured GDN kernel under the stated configuration. It is not that the two model mechanisms can be swapped without changing semantics.

## When to reach for FlashKDA

FlashKDA is a strong candidate when the deployment GPU satisfies the SM90-or-newer requirement, the toolchain satisfies CUDA 12.9 and PyTorch 2.4, the model already uses `chunk_kda`, the workload uses `K = V = 128`, and inference latency matters enough to justify a compiled extension.

The best adoption path is incremental: install the extension, enable logging, run correctness tests, benchmark against the fallback, and keep `FLA_FLASH_KDA=0` available for rollback. Treat the native kernel as a measurable implementation choice, not a permanent model-code fork.

## When not to reach for it

Keep the fallback when the GPU or toolchain is outside the documented requirements, the head dimensions violate `K = V = 128`, the workload needs training gradients, the build system cannot cache target architectures reliably, or the real sequence distribution has not been benchmarked.

The dangerous anti-pattern is benchmarking one friendly shape, seeing a large speedup, and declaring victory. FlashKDA's own benchmark makes the opposite point: fixed versus variable-length inputs, `H=64` versus `H=96`, H20 versus GB200, and KDA versus GDN all produce different stories.

A fast attention kernel is a negotiation between algebra and hardware. The algebra determines which dependencies can be exposed. The hardware determines which precision boundaries are safe, which intermediates should be materialized, and which memory movements can be eliminated. FlashKDA is compelling because those negotiations are visible in the source: sixteen-token chunks, two grids, three precision regimes, explicit liveness reuse, and a benchmark that names its shapes.


## A closer source walkthrough

The most reliable way to read a kernel repository is to follow one tensor from the public API to the launch site. In FlashKDA, the journey begins in the Python extension entry point, continues through the C++ binding, and ends in templated CUDA launch code. This is more informative than reading the README as a list of features because it reveals which options are compile-time assumptions and which are runtime choices.

The public `fwd` function receives an output pointer rather than allocating the output itself. That makes the ownership boundary explicit: the caller owns tensor allocation, while the extension owns the computation and optional state writes. It also means a benchmark can accidentally measure a different thing if it allocates tensors inside the timed function. The repository benchmark allocates tensors before timing and uses CUDA events around the repeated calls.

The launch code materializes TMA descriptors for query, key, beta, gate, bias, value, workspace, state, and output. It computes the dynamic shared-memory requirements for each stage, sets the maximum dynamic shared-memory attribute, and launches K1 followed by K2 on the supplied stream. K1 uses a grid with total tiles on the x-axis and heads on the y-axis. K2 uses sequences on x and heads on y.

That sequence is a useful debugging map. If the dispatch log says FlashKDA but the profiler shows no K1 or K2 kernels, investigate the extension, stream, and build before investigating model math. If K1 dominates, inspect tile count, TMA loads, occupancy, and shared-memory usage. If K2 dominates, inspect recurrence length, state mode, and output staging. A backend name is only the first breadcrumb; the actual kernel timeline is the evidence.

### Fixed length and variable length use different indexing

When `cu_seqlens` is absent, the implementation treats each batch element as an independent sequence. The source derives the sequence length from the total length and number of sequences, then computes a fixed number of tiles per sequence. This is the straightforward path.

When `cu_seqlens` is present, the total token buffer contains several sequences packed together. The kernel first builds cumulative tile counts. Given a global tile index, it performs a binary search to locate the sequence whose tile range contains that index. It then derives the beginning and end offsets and the local tile position.

The important distinction is not only complexity. A state is semantically owned by a sequence. If a packed buffer accidentally lets one sequence's last state flow into the next sequence, the output can remain numerically finite while being completely wrong. The explicit `N)-indexed state layout makes that ownership visible in the API and in the launch grid.

### The final partial chunk

A final chunk may contain fewer than sixteen real tokens. The reference allocates a full chunk and writes only the valid prefix. The CUDA implementation must preserve that boundary through its loads and stores. In a recurrent operation, padding errors are especially dangerous because a padded value can affect every later token through the state.

This is why a correctness suite should include lengths such as 1, 15, 16, 17, and a large length with a short remainder. The repository's benchmark uses larger workloads for performance, but those small boundary cases are the first place to look when a custom kernel disagrees with a reference.

## The cost model behind the two grids

A useful performance model for FlashKDA has four terms:

$$
T \approx T_{\mathrm{load}} + T_{\mathrm{prepare}} + T_{\mathrm{recur}} + T_{\mathrm{store}}.
$$

This is an explanatory performance decomposition, not a timing equation reported by the repository. It gives us a vocabulary for interpreting the profiler. K1 mainly changes the preparation term and the workspace traffic. K2 mainly changes the recurrence and output terms. The shared-memory unions reduce the effective load/store cost of intermediates, while the two-grid split reduces the cost of under-parallelized preparation.

For a fixed sequence, the number of chunks is approximately $\lceil T / 16 \rceil$. K1 gets work proportional to the number of chunks times heads. K2 gets one program per sequence and head and loops over chunks. As the number of sequences changes, the balance between available K1 work and K2 recurrence work changes too. This is why a benchmark needs both fixed and varlen cases.

The model also explains why the public `K = V = 128` restriction is consequential. The recurrent state has a matrix-shaped footprint per sequence and head. Increasing the feature dimensions changes shared-memory capacity, register pressure, matrix-instruction tiling, and state bandwidth together. Supporting arbitrary dimensions would not be a small shape-check change; it would require additional layout and launch specializations.

### What a profiler should show

A healthy trace should show K1 and K2 as distinct kernels with a clear dependency between them. K1 should have enough blocks to occupy the GPU, while K2 should not be mistaken for a token-parallel kernel: its grid is intentionally smaller and its work is organized along the recurrent sequence.

The trace should also show whether the CPU is introducing gaps between launches. Two kernels do not automatically mean two expensive synchronizations. They run on a stream and can be launched back to back, but host overhead, tensor preparation, or an unexpected synchronization can still create a visible gap. A benchmark that reports only device time may hide an integration bottleneck in the full model step.

For production inference, measure the complete layer or model context as well as isolated kernel time. A faster K1/K2 pair can fail to improve end-to-end latency if the surrounding framework copies state, inserts layout conversions, or dispatches through a Python path that is expensive at small batch sizes.

## Numerical reasoning without overclaiming

The source's precision decisions can be read as a sequence of risk controls.

First, the gate path is converted and activated in a way that keeps the special function fast. Second, the chunk-local matrix inverse is assigned fp16 because its values have a bounded range. Third, the recurrent state is stored in bf16 to reduce footprint. Fourth, state updates use fp32 FMA so the accumulation itself has a wider intermediate representation.

This layered policy is more defensible than a blanket claim that “bf16 is accurate enough.” Accuracy depends on where the rounding occurs. Rounding a bounded inverse is different from rounding a long-running accumulator. Updating in fp32 and storing in bf16 is different from accumulating in bf16. The implementation separates those cases.

The lower gate bound is also a numerical control. The README and benchmark configuration use `lower_bound=-5`. The deep-dive document connects that bound to the choice of sixteen-token chunks and the representable range of the cumulative exponential. If an integrator changes the gate range, the original numerical argument must be revisited rather than treated as an independent hyperparameter.

### What the tests can and cannot establish

The forward tests establish that the native implementation agrees with the Torch reference under the tested shapes, dtypes, and options. They also compare with `flash-linear-attention`, which tests integration semantics in addition to local arithmetic.

They do not establish training stability, gradient correctness, arbitrary head dimensions, or behavior under every possible gate distribution. Those would require additional tests. The responsible handoff is therefore to state the tested contract precisely and avoid widening it through implication.

A production validation matrix should include:

| Dimension | Minimum useful coverage |
|---|---|
| Sequence length | Short, exact multiple of 16, and partial final chunk |
| Packing | One fixed sequence, irregular varlen, equal-length varlen |
| State | No state, bf16 state, fp32 state where supported |
| Head count | At least the deployment head count and a neighboring supported shape |
| Backend | FlashKDA hit, explicit Triton fallback, rejection logging |
| Numerical check | Reference comparison at the actual model dtype |

## Benchmarking as an engineering experiment

The benchmark script is readable enough to serve as a template. It creates normalized bf16 `q` and `k`, bf16 `v`, `g`, and `beta`, fp32 `A_log` and `dt_bias`, then times FlashKDA with bf16 state, without state, and with fp32 state before timing the FLA paths. That is a better experiment than comparing only one call because state mode can change memory traffic and output behavior.

The script warms up, synchronizes, records CUDA events, repeats the timed loop, sorts samples, and reports mean, minimum, and maximum. These details are not boilerplate. CUDA launches are asynchronous, and first-use compilation or cache effects can dominate a short measurement. The benchmark's settings make the reported means more stable, though they still do not replace production traces.

A useful extension is to report percentiles and end-to-end layer time. The repository's generated markdown reports means because that is the chosen comparison format. A serving system should add p50, p95, and p99 latency, plus the batch and sequence distribution that produced each sample. Kernel means are excellent for diagnosing a kernel; tails are what users experience.

There is also a measurement distinction between throughput and latency. The benchmark times one operation repeatedly on a stream. A serving engine may interleave several requests, overlap host work, or use a different state lifetime. More concurrency can improve device utilization while increasing the latency of an individual request. A kernel that wins in an isolated benchmark should be evaluated again under the scheduler that will actually call it.

The state mode deserves its own experiment. No-state execution avoids writing a final recurrent state. bf16 state writes less data than fp32 state, but fp32 state may be required by a surrounding system. The benchmark script times these modes separately. That is a strong practice: do not collapse semantically different calls into one headline number simply because they share the same kernel name.

## Failure modes worth testing before production

The most obvious failure is a build failure, but runtime failures are often subtler. A binary can compile for the wrong architecture, dispatch can reject a shape and silently use Triton, or a packed sequence can produce plausible outputs with incorrect state boundaries. Each failure needs a different diagnostic.

For build failures, record the compiler, CUDA toolkit, PyTorch version, submodule revision, and architecture list. For dispatch failures, turn on backend logging and capture the rejection reason. For numerical failures, compare a small deterministic tensor against the Torch reference, beginning with one head and one short sequence. For performance failures, capture a profiler trace and check whether the expected K1 and K2 kernels are present.

Stateful kernels also need reset tests. Run two independent sequences separately, then run them packed with `cu_seqlens`. The outputs should agree for each sequence, and the state of the first sequence must not influence the second. This test targets the semantic boundary that variable-length batching introduces.

Partial chunks need the same treatment. Compare a length of 16 with a length of 17 and inspect only the valid outputs. Then compare a packed set containing lengths 15, 16, and 17. These tests exercise padding, tile-prefix indexing, and the transition from one sequence to the next.

Finally, test fallback equivalence at the model boundary. The native path and Triton path may differ in small floating-point details, but they should meet the model's accepted tolerance and preserve state semantics. A fallback is useful only if the team knows what equivalence means and has automated the check.

One more operational detail is easy to miss: the extension is tied to the layout expected by the caller. A tensor can have the right logical dimensions while using a layout that causes an implicit conversion before the kernel. Such a conversion may erase the measured kernel gain. Inspect strides, contiguity, and the surrounding model code when an isolated microbenchmark transfers poorly to production. The fastest kernel is not the fastest path if the caller must first rearrange every token.

## What FlashKDA teaches about kernel design

FlashKDA is a narrow project, but its design pattern generalizes. Begin with the dependency graph of the algorithm, not with the number of kernels you hope to launch. Identify which work is token-parallel, which work is head-parallel, and which work is inherently sequential. Then choose boundaries that expose parallelism without creating unacceptable intermediate traffic.

Choose a tile size by balancing more than arithmetic reuse. Numerical range, matrix dimensions, instruction shapes, shared-memory capacity, and portability all participate. A tile that is ideal on paper can be a poor choice if it requires a stabilization trick, an expensive inverse, or a specialized instruction path that excludes the deployment fleet.

Treat precision as a map over the computation. The right question is not “what dtype is the model?” but “where can rounding occur, and what numerical property protects that boundary?” FlashKDA uses bounded fp16 inverse values, bf16 stored state, fp32 FMA updates, and approximate gate functions because those choices correspond to different risks.

Make memory lifetimes visible. Shared-memory unions are not free: they introduce a correctness obligation around synchronization and reuse. But keeping every intermediate alive is also not free. The useful design is the one whose liveness plan matches the actual phases of the kernel.

Finally, publish the benchmark dimensions. A benchmark without shape, sequence layout, state mode, warmup policy, and competitor is an anecdote. FlashKDA's H20 and GB200 tables are valuable partly because they name the cases. The numbers can be challenged, reproduced, or replaced without guessing what was measured.

### Do not benchmark a fallback accidentally

The dispatch log is part of the benchmark configuration. If the call rejects FlashKDA because of a shape, an unsupported option, or an environment variable, the timing may still look perfectly plausible because the Triton implementation runs successfully. Every benchmark result should therefore record:

- GPU model and driver;
- CUDA and PyTorch versions;
- FlashKDA commit;
- tensor shapes and sequence lengths;
- state mode;
- dispatch result;
- warmup, iterations, and repeats.

Without those fields, a speedup number is difficult to reproduce and easy to misattribute.

## Operational deployment checklist

Before rolling a custom kernel into a model service, build the extension for the actual deployment architectures. The repository supports `auto`, `all`, and an explicit architecture list. CI should use an explicit policy rather than relying on whichever GPU happens to be attached to the build runner.

Run the repository tests in the same container image that will serve the model. A local developer machine can hide differences in CUDA minor versions, compiler behavior, driver compatibility, and submodule state. The installation command uses `--no-build-isolation`, so the environment's existing PyTorch and CUDA toolchain are part of the build contract.

At runtime, log the backend decision at a controlled verbosity. Keep the fallback switch documented and tested. If a rollout shows a numerical mismatch or unexpected latency regression, the fastest response is to disable the native path and compare the same model inputs through Triton.

Finally, profile the model in context. FlashKDA can reduce the time of one KDA operation while another layer, a state copy, or a layout conversion becomes the new bottleneck. Kernel optimization is complete only when the system-level metric improves.

## A practical decision tree

The adoption decision can be made in a short sequence:

1. Does the target GPU and software stack satisfy the repository requirements?
2. Does the model use the supported `chunk_kda` contract and `K = V = 128`?
3. Can the extension be compiled for the deployment architecture?
4. Do correctness tests pass for the model's fixed and packed sequence shapes?
5. Does the dispatch log confirm a FlashKDA hit?
6. Does an end-to-end benchmark improve the metric that matters?

A “no” at any step is useful information. It tells the team to keep the fallback, adjust the build, or avoid the specialized path. The decision tree is deliberately conservative because a fast kernel with an unverified dispatch or an unsupported state layout is not an optimization; it is an unmeasured behavior change.


## Further reading

- [FlashKDA repository](https://github.com/MoonshotAI/FlashKDA)
- [FlashKDA v1 design deep dive](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md)
- [FlashKDA H20 benchmark](https://github.com/MoonshotAI/FlashKDA/blob/master/BENCHMARK_H20.md)
- [FlashQLA: Inside Qwen's High-Performance Gated DeltaNet Kernels](/blog/machine-learning/open-source-library/flashqla-gated-deltanet-kernels)
- [The selective-scan and delta-rule decode kernel](/blog/machine-learning/inference-engineering/the-selective-scan-and-delta-rule-decode-kernel)

---
title: "Optimizing MoE training and inference: a practitioner's playbook from multi-GPU to multi-node"
date: "2026-05-08"
publishDate: "2026-05-08"
description: "An end-to-end engineering guide to making Mixture-of-Experts models fast: parallelism choices, all-to-all dispatch, GroupedGEMM, FP8, expert offloading, and a profiling-driven tuning loop, with case studies from DeepSeek-V3, Qwen3-235B, GPT-OSS, and Mixtral."
tags: ["mixture-of-experts", "moe", "training", "inference", "expert-parallelism", "all-to-all", "fp8", "cuda-graphs", "expert-offloading", "megatron-core", "deepseek", "qwen3"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

Mixture-of-Experts is the only credible answer the field has produced to the cost equation of frontier-scale pretraining: parameters going up faster than per-token FLOPs. The first time you stand up a 200B-active-out-of-1T MoE on a single rack, this stops being a charming paper and starts being a systems problem with no individual subsystem to blame. Memory looks fine. Compute looks fine. Throughput is half what the dense baseline gave you per dollar. Then you read the Nsight trace and see two big purple bands that are not in your dense profiles, and the rest of this article begins.

![MoE optimization is the joint minimization of three walls — memory, communication, and compute](/imgs/blogs/optimizing-moe-training-and-inference-1.png)

The diagram above is the mental model: MoE optimization is the joint minimization of three walls. Memory, communication, and compute are not independent — every lever you pull on one moves the others. Push expert parallelism (EP) to shard weights and you trade memory for an all-to-all on every layer. Fuse the GEMMs to recover compute, and now you constrain how the dispatcher can shape its sends. The reason MoE work feels so unfamiliar after years of dense-model intuition is that the dense bottleneck — tensor-core utilization — is rarely the binding constraint. Bandwidth is. So is launch overhead. So is the activation memory that nobody warned you a load-imbalanced step would generate.

This post is the playbook I wish I had on my first MoE bring-up. It walks the parallelism choices end-to-end, then the dispatch and compute optimizations that matter on H100 NVL8 islands and GB200 NVL72 domains, then the inference-side toolkit — expert offloading, on-GPU caches, batching effects — and closes with named case studies from DeepSeek-V3, Qwen3-235B-A22B, GPT-OSS, and Mixtral. The numbers come from the Megatron-Core MoE work ([arXiv:2603.07685](https://arxiv.org/abs/2603.07685)), the [NeMo Megatron-Bridge MoE optimization guide](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/moe-optimization.html), [Unsloth's faster-MoE notes](https://unsloth.ai/docs/basics/faster-moe), and the [APXML expert-offloading chapter](https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading). If you want the architectural backstory before the systems angle, start with [MoE LLM architecture, training, and fine-tuning](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) and [Modern LLM architectures: Qwen, Llama, Gemma, DeepSeek](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek).

## 1. Why MoE is a systems problem, not a modeling problem

MoE works because it decouples *capacity* from *per-token compute*. A dense 70B model touches every weight on every token. A 685B MoE with top-2 of 256 experts touches a couple of percent of the weights per token. The total parameter count — call it the "memory footprint" — is now decoupled from the FLOPs the GPU has to do, which is the whole point. But this only buys you something if the systems stack actually delivers the sparsity. If the network and the kernel layer cannot keep up, you have all the disadvantages of a 685B model — checkpoint size, cold start, RAM pressure — and none of the advantages.

![Dense vs MoE per-token compute: sparsity decouples FLOPs from parameters but couples the cost to bandwidth](/imgs/blogs/optimizing-moe-training-and-inference-2.png)

The mismatch most engineers feel on day one is that the dense playbook stops fitting. With dense models, you find the dominant kernel, you saturate the tensor cores, you fight for the next 5% of MFU. With MoE, the dominant kernel is rarely the bottleneck. Instead the bottleneck is the *layer envelope* — what the GPU is doing while the GEMM is not running. Two all-to-alls, a permute, an unpermute, a router, a gate, and a top-k. None of those are GEMMs. Each is a few hundred microseconds. Multiplied by 60+ layers and 8 micro-batches per pipeline bubble, that is your throughput number.

| Assumption from dense world | What MoE actually does | Consequence |
|---|---|---|
| Parameter count = compute cost | Capacity decoupled from per-token compute | FLOPs/GPU benchmarks need active params |
| TP scales with model size | TP shards GEMMs that are already small | Use EP, not TP, for FFN scaling |
| All-reduce is the comm pattern | A2A dispatch + combine is the comm pattern | Allreduce-tuned topologies underperform |
| Activation memory linear in batch | Load-imbalanced experts blow it up | Plan for 2x worst-case activation budget |
| Kernel launches are amortized | Per-expert GEMMs are launch-bound | Group them with GroupedGEMM |
| Mixed-precision = just FP16 | FP8/NVFP4 needs router in FP32 | Dual-precision recipe per layer-type |

Hold this table next to your dense profile. Every row is a place where the operational habits transfer poorly.

A useful mental shortcut: a dense model's per-step time is approximately `T_dense ≈ T_attn + T_ffn + T_comm_dp`. An MoE's per-step time is approximately `T_moe ≈ T_attn + T_ffn_active + T_router + T_permute + 2·T_a2a + T_combine + T_comm_dp`. Five of the eight terms are new. Three of them — router, permute, A2A — are *not* GEMMs and so do not benefit from the tensor-core wins that drove dense optimization for the last decade. The optimization problem genuinely is structurally different, and the systems mindset has to follow.

The corollary that catches teams off guard is that *MFU as a metric is misleading on MoE*. A dense 70B at 50% MFU is doing well; an MoE 685B at 50% MFU might be doing terribly because half of "MFU" is being spent on expert GEMMs that themselves are running at sub-saturation. The number to track instead is *active-parameter throughput*: tokens per second times the number of active parameters per token, divided by the GPU's peak FLOPs at the relevant precision. That ratio is comparable across dense and MoE, and it lines up with cost-per-token, which is what actually matters for production.

## 2. The three walls: memory, communication, compute

Anytime an MoE training run is slow or OOMing, the answer is one of three walls. The diagnostic loop is: profile, classify, apply the matching fix, re-profile. The walls are not symmetric — communication is by far the most common bottleneck on H100, while memory dominates on long-context Blackwell training, and compute only becomes the critical wall once you have done the comm and memory work. Knowing which wall you are standing in front of is more than half the optimization problem.

**Memory wall.** Activation memory grows with sequence length and is roughly the same shape as a dense model's, except that load imbalance can spike per-rank activations 1.5–2× over the average. Weight memory shrinks with EP and ETP — under EP=64 you carry 1/64 of the expert weights per rank — but the optimizer states for those weights are still 4–8× the weight bytes in a typical Adam recipe.

**Communication wall.** Dispatch and combine are the two all-to-alls that bracket every MoE layer. On a 60-layer model, that is 120 A2A collectives on the forward critical path of a single micro-batch, before you even count backward. If your fabric is 400 Gbps IB and your tokens-per-rank dispatch is 2 MB, that is roughly 40 µs per A2A on a perfect day; multiplied by 120 layers, ~5 ms is sitting in pure communication on every step.

**Compute wall.** GEMM FLOPs are typically *not* the bottleneck on MoE — the per-expert GEMMs are too small to saturate tensor cores without help. Help looks like GroupedGEMM, kernel fusion for the router and permutation, and FP8/NVFP4 on the expert GEMMs themselves. Once those are in place, you can occasionally push tensor cores into the math-bound regime; on a GB300, [DeepSeek-V3 685B was reported at 1,233 TFLOPS/GPU](https://arxiv.org/abs/2603.07685), which is real saturation territory.

> The mistake I keep watching teams make is to optimize compute first because that is the lever they know. Profile first. Then classify. Then fix the right wall.

There is a more subtle reason to lead with profiling. Walls *interact*. Reducing the comm wall by going from EP=8 to EP=4 lowers A2A latency, but it doubles the per-expert weight footprint, which can push you back into the memory wall and force optimizer offload, which itself adds a different kind of communication (host-device transfers per step). Reducing the memory wall by enabling activation recomputation cuts the memory cost of dense layers but does almost nothing for MoE expert activations because they are already streamed through GroupedGEMM, and it adds 25–30% extra compute that may move you into the compute wall. The interactions form a dependency graph; you cannot solve them by greedy local edits.

The way I keep this manageable is to model each step's wall as a vector `(M, C, K)` for memory, communication, and compute respectively, normalized to the device limit. A healthy MoE training run lives near `(0.7, 0.7, 0.7)` — all three resources roughly equally used. A run with `(0.95, 0.4, 0.6)` is memory-bound; the lever to pull is whichever one ships the most memory back: in 2026, that is usually optimizer-state offload to CPU, followed by memory-efficient permutation and increasing PP. A run with `(0.4, 0.95, 0.5)` is communication-bound; pull DeepEP/HybridEP, A2A overlap, shared-expert overlap, and reduce EP if the model architecture allows it. A run with `(0.5, 0.5, 0.95)` is the rarest case in MoE and almost always means you have already done the comm and memory work — at this point GroupedGEMM, fused kernels, FP8/NVFP4, and CUDA Graphs are your remaining levers.

## 3. Parallelism for MoE: TP, EP, DP, PP, CP, and Parallel Folding

There are five parallelism axes and one operational technique you must understand before you write a single training command. Each axis has a memory cost, a communication cost, and an interaction with the others. Get the combination wrong on a 1024-GPU job and you waste a week of cluster time.

![Parallel Folding decouples attention parallelism from MoE parallelism on a 256-GPU cluster](/imgs/blogs/optimizing-moe-training-and-inference-3.png)

**Tensor parallelism (TP).** Shards individual GEMMs along an output dimension. Communication is `allgather + reducescatter` per layer (or its fused equivalent), bounded by NVLink bandwidth. TP=8 on a single H100 NVL8 island is the sweet spot for dense LLMs because it stays inside NVLink. For MoE *expert* GEMMs, TP is usually wrong: the GEMMs are already small per token, and slicing them further makes the launch problem worse, not better.

**Expert parallelism (EP).** Shards experts across ranks. Each rank holds `N/EP` experts. Tokens routed to a non-local expert get sent over an A2A. Memory cost: 1/EP for expert weights and optimizer states. Communication cost: medium — a single A2A pair per layer that scales with `tokens × hidden × top_k`. EP is the *primary* parallelism axis for MoE, and the question of where to set it is the question that matters most.

**Data parallelism (DP).** Replicates the model across rank groups; allreduces gradients each step. The familiar pattern. With ZeRO-style sharding (FSDP, MegatronCoreParallelism), DP also shards optimizer states, which is the only thing that lets you fit a 685B-param optimizer on a single rack at all.

**Pipeline parallelism (PP).** Splits layers across ranks; passes activations between stages. Bubble dominates if you don't use micro-batching. Critical for very deep MoEs because PP is how you fit the activation memory of long contexts in a budget that EP and TP cannot reach.

**Context parallelism (CP).** Splits the sequence dimension across ranks; replaces the attention all-reduce with a ring-reduce of K and V. Linear in sequence length. Use when `seq_len > 8K`. The Megatron guide's [recommendation is `CP ≈ seq_len / 4096`](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/moe-optimization.html) — a 32K-context job wants CP=8 or so.

**Parallel folding** is the technique that makes the above tractable. The trick is recognizing that *attention* and *MoE* have different parallelism preferences. Attention loves TP and CP; MoE loves EP and dislikes TP. Old MoE codebases forced a single `(TP, EP, DP, PP)` tuple over the whole model, which meant you had to compromise between the two halves. Parallel folding decouples them: attention runs `TP=4, CP=2, DP=8, PP=4`, the MoE layers run `ETP=1, EP=64, EDP=1, PP=4`, on the same 256 physical GPUs. The constraint that EP must divide DP is gone. From the [Megatron-Core MoE paper](https://arxiv.org/abs/2603.07685), this typically wins 10–25% of throughput on 256+ GPU clusters.

| Axis | Memory shrink | Comm cost | Best for | Anti-pattern |
|---|---|---|---|---|
| TP | 1/TP weights, attn act | NVLink-bound | Dense GEMMs, attention | MoE expert GEMMs |
| EP | 1/EP expert weights | A2A per layer | MoE FFN | Tiny expert counts |
| DP | activations / DP (with ZeRO) | Allreduce gradients | Always | When you're memory-bound |
| PP | activations / PP | P2P, bubble cost | Deep models | Short contexts, low utilization |
| CP | activations / CP | Ring on KV | Long context | Short context |
| Parallel folding | — | — | EP × DP > 1 cases | Single-island runs |

The recipe I use: minimize model parallelism, maximize DP, keep `EP × TP` inside the fastest interconnect domain (NVL8 or NVL72), then turn on parallel folding once `EP × DP` would otherwise be constrained.

```bash
## 256-GPU H100 NVL8 cluster, DeepSeek-V3-style 685B MoE
## Attention: TP=4, CP=2, DP=8, PP=4  →  fills 256 ranks
## MoE:       ETP=1, EP=64, EDP=1, PP=4 (parallel folded)
megatron-core-train \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 4 \
  --context-parallel-size 2 \
  --expert-model-parallel-size 64 \
  --expert-tensor-parallel-size 1 \
  --num-experts 256 --moe-router-topk 8 \
  --moe-grouped-gemm \
  --moe-router-fusion \
  --moe-permute-fusion \
  --moe-token-dispatcher-type flex \
  --moe-flex-dispatcher-backend deepep \
  --overlap-moe-expert-parallel-comm \
  --use-parallel-folding \
  --pipeline-model-parallel-layout "Et*3|(tt|)*29m|L" \
  --fp8-format e4m3 --fp8-recipe blockwise \
  --offload-optimizer-states
```

### 3.1. Why EP is special

The reason EP gets singled out among the parallelism axes is that it changes the *kind* of GEMM the FFN is doing, not just where it runs. Under TP, an FFN GEMM `(B, d_model) × (d_model, d_ffn)` becomes `(B, d_model) × (d_model, d_ffn/TP)` on each rank — same shape class, smaller. Under EP, the FFN GEMM becomes a *segmented* GEMM where each rank's local experts each see a different number of tokens. The shape changes from "rectangular" to "ragged-rectangular," and that is what motivates GroupedGEMM. None of the other parallelism axes have this property: they preserve the GEMM shape class. EP is the parallelism that earned its own kernel.

Two derived facts follow. First, EP only helps when the per-expert GEMMs are large enough to amortize the overhead of *segmenting* them. With 8 experts and a small batch, the segmented overhead can outweigh the parallelism gain — you are better off replicating experts and using DP. The crossover is around 1024 tokens per expert per step on H100 and around 512 on B200. Second, EP composes with TP only when ETP is decoupled from TP. Megatron-Core treats ETP as an *independent* TP for expert layers, sharing the world ranks but using a different sharding pattern. Without that decoupling, every TP rank in an EP=N group has to share the same expert assignment, and you lose the EP benefit on the FFN.

### 3.2. Second-order: when EP doesn't divide DP

The classic constraint is "EP must divide DP" because each DP rank group needed its own copy of the experts. Parallel folding breaks this. The new constraint is `EP × ETP × EDP × PP = total ranks for MoE layers`, with `(TP × CP × DP × PP) = total ranks for attention layers`. The two products must equal the same physical rank count, but the factorization is independent. This is the *only* way to set EP=64 on a 256-GPU cluster without sacrificing attention parallelism.

## 4. Token dispatch: All-to-All, DeepEP, HybridEP

Dispatch is the hot path. Every MoE layer pays two all-to-alls per micro-batch, and every byte that moves over IB is a byte that the FFN GEMMs are waiting on. The dispatcher is the piece of code that decides how those bytes move.

![Each MoE layer pays two all-to-alls per micro-batch, and they sit on the critical path](/imgs/blogs/optimizing-moe-training-and-inference-4.png)

The lifecycle is: router produces top-k indices, permute groups tokens by destination expert, A2A-dispatch sends each group to its owning rank, the destination runs a GroupedGEMM over the local experts, A2A-combine returns the outputs, and unpermute scatters them back to the original token positions, weighted by the router probabilities.

Three implementations matter in 2026:

**Standard MoE.** Torch-native A2A using NCCL. Works everywhere; fast enough for EP ≤ 8; the baseline you measure against.

**DeepEP.** SM-based dispatch with GPU-side routing. The [dispatch happens inside CUDA kernels](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/moe-optimization.html), not on the host, which removes a couple of microseconds of CPU launch latency per A2A and — more importantly — keeps the dispatcher's metadata on-device so it can be fused with the permute step. DeepEP is the right answer on H100 NVL8 islands when EP crosses islands.

**HybridEP.** Fused intra-node NVLink + inter-node IB dispatch. On a GB200/GB300 NVL72 domain, all 72 GPUs are reachable over copper NVLink with no IB hop. HybridEP exploits this by treating the NVL72 domain as one big NVLink fabric and only invoking IB for *inter-domain* communication. EP=64 inside an NVL72 domain runs at sub-5 µs per dispatch; the same EP across H100 NVL8 islands is at least an order of magnitude slower because the IB hop dominates.

![Dispatcher choice should track the GPU domain interconnect topology](/imgs/blogs/optimizing-moe-training-and-inference-5.png)

```bash
## H100 NVL8, EP=64 across islands → DeepEP
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep

## GB200/GB300 NVL72, EP=64 inside domain → HybridEP
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep
```

### 4.1. The permute and unpermute kernels deserve their own attention

Behind every dispatch is a permutation: tokens arrive in source order, but the dispatcher needs them in *destination-expert order* so the A2A send buffer is laid out as a contiguous block per expert. The naive permute is a scatter on the GPU, and the naive unpermute is its inverse — both are bandwidth-bound on the activation tensor, which can be 1–4 GB for a 60-layer model at long context. If you let them run as separate kernels, you pay two extra GPU passes over the activations per layer per direction.

Megatron-Core's `--moe-permute-fusion` flag fuses the permute with the gate scoring on one side, and the unpermute with the routing-weight multiplication on the other. The fused kernel computes the destination index from the router's top-k output and writes directly into the destination buffer — one pass instead of three. On a H100 running a 60-layer model with hidden size 8192, this is a 50–100 µs saving per layer on the forward, ~3–6 ms per step. It is among the cheapest of the optimization wins to enable, and it should be on by default in 2026.

The unpermute side has an additional subtlety: when each token has top-k > 1, the unpermute has to *combine* the outputs of multiple experts back into a single token-aligned tensor, weighted by the router probabilities. The naive form does a scatter-add, which has nondeterministic ordering and therefore non-bitwise-reproducible numerics. If your training pipeline depends on bit-exact reproducibility (e.g., to debug a divergent run by replay), use the deterministic combine path; it is ~10% slower but reproducible. Most production runs accept the nondeterminism for the throughput.

### 4.2. Second-order: capacity factor and drop policy

A2A is a fixed-shape collective: every rank sends `tokens × hidden × top_k / world_size` bytes regardless of how many tokens actually want each expert. To support load imbalance, you set a *capacity factor*: every expert's input buffer is sized to `cf × (tokens / num_experts × top_k)` for some `cf > 1`. If routing is perfectly balanced, `cf = 1.0` is enough; in practice you want `cf ∈ [1.1, 1.5]` early in training and you can lower it once a load-balancing loss has done its work. Tokens that overflow are *dropped* in dropping mode, or *padded forward* in dropless mode. Dropless costs more memory but trains better; dropping costs more accuracy but trains faster. Most production runs in 2026 are dropless with aux-loss-free balancing, where each expert has a learnable bias on its routing logits that drifts to maintain even load — DeepSeek-V3 popularized this, Qwen3 inherited it.

## 5. Compute-side wins: GroupedGEMM, router and permute fusion, memory-efficient permutation

Once dispatch is healthy, the next budget to harvest is kernel-launch overhead and saved-tensor memory.

![GroupedGEMM consolidates per-expert GEMMs and lifts kernel occupancy from launch-bound to math-bound](/imgs/blogs/optimizing-moe-training-and-inference-6.png)

**GroupedGEMM.** A naive implementation runs one GEMM per expert: `for e in range(N): y_e = matmul(x_e, W_e)`. With 64 experts, that is 64 kernel launches per layer per direction, each with 50–100 µs of host-side overhead. GroupedGEMM (cuBLASLt segmented batched, or the cutlass GroupedGEMM kernel) consolidates them into a single launch with a ragged batch dimension. The router-produced offsets become the kernel's per-group token counts. Empirically this is a 2–5× speedup on the FFN forward, and it is the single biggest compute win in the Megatron-Core flag set.

```python
## Conceptual: naive vs GroupedGEMM
def naive_moe_ffn(x, expert_idx, W1, W2):
    out = torch.zeros_like(x)
    for e in range(num_experts):
        mask = expert_idx == e
        h = F.silu(x[mask] @ W1[e]) @ W2[e]
        out[mask] = h
    return out

def grouped_moe_ffn(x_perm, offsets, W1, W2):
    # x_perm sorted by expert; offsets[i]..offsets[i+1] are tokens for expert i
    h = F.silu(grouped_gemm(x_perm, W1, offsets))
    return grouped_gemm(h, W2, offsets)
```

The pure-Python loop is bandwidth-bound *and* launch-bound; the grouped form is math-bound on tensor cores once token counts per expert are above ~256.

**Router and permute fusion.** The router computes `softmax(top_k(W_r @ x))` and the permutation reorders tokens by expert. Both are tiny; on a 60-layer model running them as separate kernels costs you ~1 ms of launch overhead per layer. `--moe-router-fusion` and `--moe-permute-fusion` collapse them into one Triton or CUTLASS kernel each, recovering 60–100 ms per step on a 60-layer model.

**Memory-efficient permutation (routing-weight absorption).** The standard MoE forward is `y = Σ p_i · W2_i · φ(W1_i · x)`. The naive autograd implementation saves both `φ(W1_i · x)` and the routing probabilities `p_i` for the backward pass — that's an extra activation tensor per layer. The memory-efficient form is algebraically identical: `y = Σ W2_i · (p_i · φ(W1_i · x))`. The router weights are absorbed into the activations *before* the second FC, so only the merged tensor needs to be saved. Backward recovers the router gradient directly from that tensor. Net effect: half the saved-tensor memory in the MoE FFN, zero compute overhead, valid as long as the experts have no bias.

![Folding routing weights into activations before W2 removes a saved tensor at zero overhead](/imgs/blogs/optimizing-moe-training-and-inference-7.png)

### 5.1. The shape of GroupedGEMM in practice

The kernel that powers most production MoE in 2026 is the `cutlass::gemm::GroupedGemm` family, with cuBLASLt segmented batched as a fallback. The interface looks deceptively simple: pass an array of `(A_ptr, B_ptr, C_ptr, M, N, K)` tuples, get a single launch back. In practice, the bottleneck is the *prologue* — getting the per-group offsets onto the device in time for the kernel. If you compute them on the CPU, you have a ~5 µs PCIe transfer per launch, which is most of the saving you got from grouping. The trick is to keep the offset computation on the GPU: the router emits `top-k` indices, a fused scatter writes the per-expert token counts directly into the device-side offset array, and the GroupedGEMM kernel reads from that array without a host round-trip. This is what `device-initiated grouped GEMM` (mentioned in the CUDA Graphs section) refers to.

The other practical concern is *autotuning*. cutlass has many tile configurations; the right one depends on M (token count), N (hidden), and K (d_ffn) per expert. Unsloth's reported 2.5× over `torch._grouped_mm` comes mostly from autotuning: their stack runs a 2-minute warmup that benchmarks 30+ configurations and picks the best per shape class. On a long training run, that 2 minutes amortizes into a 35% throughput gain — pure profit. If you are not on Unsloth, copy the discipline: profile your shapes, pin the best tile config, save it across runs.

### 5.2. Second-order: the GEMM grain that wins on H100 vs B200

The math-bound regime for grouped GEMM on H100 starts at roughly 128 tokens per expert; on B200 it starts at 64. Below that grain you are still launch-bound regardless of GroupedGEMM. The implication: B200 lets you push toward fine-grained MoE — 256 experts of width 256 — that simply doesn't pay back on H100. DeepSeek-V3 uses 256 experts in part because it was designed for the sub-100-token-per-expert regime that H100/H200 reach with EP=64 and CP=8.

## 6. Low-precision training: FP8 on Hopper, MXFP8 / NVFP4 on Blackwell

Low-precision is no longer optional for frontier MoE. The expert GEMMs are the big arithmetic chunk; the router is small but extremely sensitive to quantization noise.

The recipe per platform:

| Platform | Default | Production | Maximum throughput |
|---|---|---|---|
| Hopper (H100/H200) | Per-tensor FP8 (E4M3) | Blockwise FP8 | — |
| Blackwell (GB200/GB300) | MXFP8 | MXFP8 | NVFP4 (with RHT + stochastic round) |

Two non-negotiable rules from the Megatron-Core MoE paper:

1. **Router stays in FP32.** A noisy top-k destroys load balancing because tokens flip experts on tiny logit perturbations.
2. **Expert GEMMs are the primary quantization target.** Quantize the FFN, not attention's QKV unless you have measured headroom.

NVFP4 specifically requires Random Hadamard Transforms applied to the activations and stochastic rounding on the GEMM accumulators. Without RHT, the FP4 outliers in the activations destroy GEMM accuracy after a few thousand steps. With RHT, NVFP4 reaches 95–98% of BF16 quality on the same training tokens — and it is roughly 2× the throughput of MXFP8 on B200.

```bash
## Hopper, production-grade FP8
--fp8-format e4m3 --fp8-recipe blockwise --fp8-amax-history-len 1024

## Blackwell, default
--fp8-format mxfp8

## Blackwell, max-throughput
--fp8-format nvfp4 --fp8-rht --fp8-stochastic-round
```

## 7. Pipeline layout for asymmetric MoE stacks

Pipelines live or die by stage balance. A naive split of a 60-layer MoE into 4 stages of 15 layers each is *unbalanced* — the embedding layer is cheap, the MTP head is expensive, and dense layers (if you have any) cost less than MoE layers. The pipeline bubble is set by the slowest stage.

![Pipeline stages must be balanced by wall-clock cost, not by raw layer count](/imgs/blogs/optimizing-moe-training-and-inference-8.png)

The Megatron-Bridge layout-string DSL lets you express asymmetric VPP partitions explicitly. The tokens are: `E` = embedding, `t` = transformer (dense), `m` = MTP, `L` = loss, `|` = stage boundary. The product means a repeated group. So `Et*3|(tt|)*29m|L` reads as: stage 0 holds the embedding plus 3 dense layers; stages 1–29 each hold 2 transformer layers; stage 30 holds the MTP head; stage 31 holds the loss. The result is roughly equal wall-clock per stage, which minimizes the bubble.

```bash
## DeepSeek-V3-685B-style asymmetric pipeline
--pipeline-model-parallel-layout "Et*3|(tt|)*29m|L"
--num-virtual-stages-per-pipeline-rank 4   # VPP for bubble reduction
```

### 7.1. Second-order: VPP bubble is a function of micro-batch count

Virtual pipeline parallelism (VPP) reduces the bubble by interleaving stage assignments — each rank holds multiple non-contiguous chunks of layers. The bubble shrinks linearly with the number of virtual chunks per rank, but only if the micro-batch count exceeds VPP × PP. With micro-batches=8, PP=4, VPP=4 means VPP×PP=16 > 8 — the bubble doesn't shrink at all. Practical rule: `num_microbatches ≥ 2 × VPP × PP` for VPP to actually pay back.

## 8. Multi-node: when EP crosses the rack

Inside an NVL8 island, A2A is fast: 8 GPUs, NVLink, ~5 µs per dispatch. The moment EP crosses islands — EP=16 means at least one IB hop — the latency picture changes by an order of magnitude. The IB fabric is 400 Gbps in the best modern clusters; that is fast for bandwidth but slow for *latency-bound* small messages, which is exactly what dispatch sends.

The two levers that matter in this regime are A2A overlap and the choice between "EP grows inside the island" vs "EP grows across islands."

![Splitting backward MLP into weight-grad and data-grad lets weight-grad overlap with the next forward pass](/imgs/blogs/optimizing-moe-training-and-inference-9.png)

**A2A overlap.** Backward MLP is two GEMMs: weight-grad (`dW = X.T @ dY`) and data-grad (`dX = dY @ W.T`). Only data-grad sits on the critical path — it produces the input to the previous layer. Weight-grad doesn't, because the weights aren't needed again until the optimizer step. So you can launch weight-grad in a side stream that runs *during the next forward pass*, and the A2A for the next layer's dispatch can overlap with it. This is what `--overlap-moe-expert-parallel-comm` does in Megatron-Core. The win is 8–15% throughput on multi-node MoE.

**EP-inside-island vs across.** On NVL8 islands, the heuristic is to keep EP ≤ 8 if your expert count permits, and use TP+ETP to combine experts within an island for higher EP-equivalent capacity. On NVL72 domains (GB200/GB300), EP=64 inside the domain is the right answer; HybridEP makes it work without per-layer IB traffic.

```bash
## Multi-node, H100 NVL8, EP=64 across 8 islands
--expert-model-parallel-size 64 \
--moe-flex-dispatcher-backend deepep \
--overlap-moe-expert-parallel-comm \
--moe-shared-expert-overlap   # hide shared-expert FFN behind dispatch

## Multi-node, GB200 NVL72, EP=64 inside one domain
--expert-model-parallel-size 64 \
--moe-flex-dispatcher-backend hybridep \
--overlap-moe-expert-parallel-comm
```

### 8.1. Topology-aware rank ordering

A subtle but throughput-critical detail: how you assign physical GPUs to logical ranks matters. NCCL builds its A2A topology from the rank list, and a poor assignment turns intra-island A2As into inter-island ones. The rule is to assign consecutive ranks to GPUs in the same NVL8 island, then move to the next island. Under EP=8, this puts every EP group entirely inside one island — every dispatch stays on NVLink. Under EP=64 across 8 islands, the topology is a fully-connected bipartite graph between islands, and rank ordering matters less; but the attention TP and DP assignments still benefit from island locality.

Megatron-Core has `--use-distributed-optimizer` and `--ddp-bucket-size` flags that respect rank ordering for the dense gradient allreduce. For MoE, the equivalent is to ensure that EP groups are contiguous in the rank list and that the `EXPERT_MODEL_PARALLEL_GROUP` is constructed before `DATA_PARALLEL_GROUP`. The default is sometimes wrong on multi-rack setups; check the actual NCCL topology with `NCCL_DEBUG=INFO` on the first run.

### 8.2. Multi-node failure modes I keep seeing

Three multi-node anti-patterns recur in real bring-ups. The first is *fabric saturation by metadata collectives*: a framework helpfully runs an allreduce after every dispatch to "verify" that no tokens were dropped. On 64 GPUs at 60 layers, that is 60 extra allreduces per step, each ~50 µs, for 3 ms of pure overhead. The fix is to disable the verifier in production runs. The second is *unaligned DP and EP groups*: when DP=8 and EP=8 are not aligned to the same physical rank assignment, every gradient allreduce crosses islands. The fix is to set `--data-parallel-group-aligned-with-expert-parallel`. The third is *the silent IB-vs-Ethernet fall-through*: on cloud clusters, the "IB" interface sometimes resolves to a TCP fallback that runs at 10 Gbps. Throughput is one-tenth what you expected and the trace looks normal. Always verify with `NCCL_NET=IB` and a sanity-check bandwidth test before a long run.

### 8.3. Second-order: shared experts and the `--moe-shared-expert-overlap` flag

DeepSeek-V3 uses *shared experts* — a small set of always-on experts that run on every token in addition to the routed experts. They contribute a fixed compute cost per token, but because they don't need dispatch, they can run on the source rank in parallel with the A2A dispatch of the routed tokens. `--moe-shared-expert-overlap` does exactly this: launches the shared-expert FFN on a separate stream during the dispatch A2A, hiding most of its cost.

## 9. CUDA Graphs for dropless MoE

CUDA Graphs amortize launch overhead by capturing a sequence of kernel invocations into a single replay-able object. For dense LLMs this is a routine 10–15% win. For dropless MoE it is harder, because the dispatcher's token counts per expert change every step.

![Partial capture wraps the static MoE layer body while the dynamic dispatcher and grouped GEMM stay outside](/imgs/blogs/optimizing-moe-training-and-inference-10.png)

Two modes:

**Full graph capture.** Works for *drop-and-pad* MoE, where every expert always processes a fixed token count (overflow gets dropped, underflow gets padded with zeros). Simple but wastes ~20% compute on padding.

**Partial / layer-wise capture.** The right answer for dropless. Capture the parts that *are* static: norms, attention, residuals, gate logits, permute prep. Leave the dispatcher, grouped GEMM, and combine outside the graph. The captured region usually accounts for 60–70% of the layer's launches, so the win is real.

For full-graph dropless, the Megatron-Core MoE work introduces three additional tricks: device-initiated grouped GEMM (the dispatcher writes the group counts into device memory and the GEMM kernel reads them), ECHO (Elastic Cloning for Hot Experts — replicate the most-loaded experts across multiple ranks to flatten the worst-case per-expert tokens), and paged stashing (stash activations in pages rather than per-layer slabs, so memory becomes O(actual) not O(layers × worst-case)).

```bash
--cuda-graph-impl transformer_engine
--cuda-graph-mode partial   # dropless MoE
--enable-echo               # hot-expert replication for full graph mode
```

## 10. Fine-tuning MoE: Unsloth's Split-LoRA + Triton GroupedGEMM

Pretraining is the giant problem; fine-tuning is the problem most teams actually face. The challenge is that LoRA on MoE is awkward: a naive implementation materializes a separate LoRA delta for every expert, which means `E × m × n` extra parameters touched per token even if only `k` experts are active. That makes LoRA on a 64-expert MoE about 32× more expensive than LoRA on the equivalent dense model.

[Unsloth's Split-LoRA approach](https://unsloth.ai/docs/basics/faster-moe) reorders the matrix operations using associativity to avoid materializing per-expert deltas:

```
## Naive: each expert merges its own LoRA before forward
y_e = (W_e + B_e @ A_e) @ x

## Split-LoRA: keep base weights and adapters separate; route once
y_e = W_e @ x + B_e @ (A_e @ x)   # per active expert only
```

Combined with a custom Triton GroupedGEMM kernel that is, per their numbers, 2.5× faster than `torch._grouped_mm` on A100, the reported wins are substantial: GPT-OSS 20B fine-tunes at 7× the throughput and 36% less VRAM at 16K context, Qwen3-30B-A3B is 1.8× faster on B200, GLM-4.7 Flash is 2.1× faster on RTX PRO 6000. The PRO 6000 number is the one I find most striking: it makes 30B-class MoE fine-tuning tractable on a single consumer-tier card with 48 GB VRAM, which was unthinkable on the dense equivalent.

```python
## Unsloth-style Split-LoRA call (sketch)
from unsloth.kernels import grouped_gemm_lora

## x: (T, d_model), expert_offsets: (E+1,) cumulative tokens per expert
## W_base: (E, d_model, d_ffn), A: (E, d_model, r), B: (E, r, d_ffn)
y = grouped_gemm_lora(x, W_base, A, B, expert_offsets, fused=True)
```

If you are not on Unsloth, the same idea works hand-rolled: keep base experts in 4-bit (QLoRA), put the LoRA adapters in BF16, run dispatch on the routed tokens only, and run the LoRA path as two small grouped GEMMs (down-projection then up-projection). The trick is *not* to merge the LoRA into the base weight at runtime — that is what makes the naive version expensive.

Two further patterns matter for MoE fine-tuning specifically. The first is *expert-selective LoRA*: instead of putting a LoRA adapter on every expert, put one only on the experts that the routing distribution shows as the most active for your task domain. On a 64-expert model fine-tuned on a 50K-example domain dataset, typically 10–15 experts handle 80% of routed tokens; LoRA-ing only those 15 experts gives you 95% of the quality at 25% of the trainable-parameter cost. The selection is automatic: run one validation pass with logits collected, sort experts by routed-token count, take the top-k.

The second is *router fine-tuning vs router freezing*. The router is a small parameter set but a load-bearing one: changing it during fine-tuning can drift the expert utilization distribution and silently degrade out-of-distribution behavior. The conservative recipe is to freeze the router during the first phase of fine-tuning (let the LoRA adapters specialize the existing experts to your domain) and then unfreeze it for a second phase (let the router shift toward the experts that the LoRAs have improved). This two-phase approach consistently outperforms joint training in my experience, especially on small fine-tuning datasets where the router lacks signal to adapt safely. See [effective LLM fine-tuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) for the broader fine-tuning context that this builds on.

## 11. Inference: expert offloading and on-GPU caches

Now to the inference half. The training optimizations don't all carry over: expert parallelism still works for serving, but the economics shift because batch size is smaller and the cost of a wasted IB hop is paid per request. The single most useful inference technique that has no training analogue is *expert offloading* — keeping cold experts on CPU RAM or NVMe and using VRAM as a cache.

![Expert offloading turns VRAM into a working-set cache while CPU and NVMe back the cold weights](/imgs/blogs/optimizing-moe-training-and-inference-11.png)

The setup: shared experts, the router, and the KV cache stay GPU-resident. The routed expert weights live on CPU RAM (or NVMe for very large models). A small LRU cache on the GPU holds the most-recently-used `c` experts. On every token, the router picks `top_k` experts; those that are cache-hit run immediately on the GPU; those that miss are fetched asynchronously over PCIe (or NVLink for Grace-Hopper / GB200, which is much faster), then pinned into the cache and used.

```python
class ExpertCache:
    def __init__(self, capacity, num_experts, expert_shape, device='cuda'):
        self.capacity = capacity
        self.cpu_experts = [None] * num_experts   # populated lazily from disk
        self.gpu_buf = torch.empty(capacity, *expert_shape, device=device)
        self.expert_to_slot = {}                   # expert_id -> slot in gpu_buf
        self.lru = []                              # MRU at end

    def get(self, expert_ids):
        # expert_ids: list of indices we need this step
        out_slots = []
        for eid in expert_ids:
            if eid in self.expert_to_slot:
                slot = self.expert_to_slot[eid]
                self.lru.remove(eid); self.lru.append(eid)
            else:
                if len(self.expert_to_slot) >= self.capacity:
                    evict = self.lru.pop(0)
                    slot = self.expert_to_slot.pop(evict)
                else:
                    slot = len(self.expert_to_slot)
                # async copy: cpu_experts[eid] is pinned host memory
                self.gpu_buf[slot].copy_(self.cpu_experts[eid], non_blocking=True)
                self.expert_to_slot[eid] = slot
                self.lru.append(eid)
            out_slots.append(slot)
        return self.gpu_buf, out_slots
```

The numbers from the [APXML expert-offloading chapter](https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading) and from production deployments are consistent: a cache that holds 30–40% of total experts captures 80–90% of accesses on most workloads, because token-level expert selection has substantial temporal locality (consecutive tokens in the same prompt route to similar experts). Cache miss latency is dominated by PCIe — 30–50 µs to pull a 50 MB expert across PCIe Gen5 — which is masked by overlapping the fetch with the compute on hit experts.

### 11.1. Cache-size sensitivity is non-monotonic

A surprising pattern from production deployments: hit rate as a function of cache size is *not* monotonically improving — it has a knee. Below the knee (cache holds < ~20% of experts), miss rate is high and dominated by the cold start of every prompt. At the knee (~30–40% of experts), hit rate jumps to 80–90% because the working set of any single conversation tends to fit. Past the knee, marginal hit-rate gains are small (90% → 93% → 95%) but they cost real VRAM that could be batch capacity. The right operating point is *just past the knee*, leaving the rest of VRAM for batched KV cache and longer contexts.

The implication for ops is that you should *measure* the knee for your workload — it depends on prompt diversity, context length, and the model's expert specialization patterns. A workload of code completion has tighter expert locality than a workload of open-ended chat. Code workloads can run a smaller cache; chat workloads need a bigger one. Single-tenant deployments cache better than multi-tenant ones because there is less prompt-distribution churn.

### 11.2. Second-order: NVMe is for batch, not latency

NVMe offload (NVMe → CPU → GPU, optionally via GPUDirect Storage) is *only* viable for throughput-optimized batch processing. The single-token latency of NVMe under realistic queue depths is in the millisecond range, which destroys interactive use. The right deployment pattern: cold experts on NVMe, prefetch the most likely candidates to CPU at the start of a batch based on the router's prefill output, then serve the batch from CPU + GPU cache.

### 11.3. Second-order: predictive prefetch

The router's logits at prefill time predict which experts the model will visit during decode. Sorting experts by predicted activation probability and preloading the top `2k` to GPU before decode starts cuts the cold-cache cost of the first dozen decoded tokens. This is the trick used by the open MoE inference stacks; combined with LRU it usually pushes hit rates above 95% on conversational workloads.

## 12. Inference: serving-time parallelism

Beyond offloading, inference has its own parallelism story. EP at serving time looks similar to EP at training time, but two things change.

**Decode-time EP is mostly bandwidth-bound on the dispatch/combine, not the FFN.** With small per-step batches (1–32), the GEMMs on each expert are tiny; the A2A latency is the dominant cost. On H100 serving, EP=8 (inside an NVL8 island) usually beats EP=16, because crossing islands adds an IB round-trip that is not amortized by larger GEMMs.

**Replicated experts beat sharded experts at low batch.** If you have spare VRAM, replicating each expert across multiple ranks and routing by hash to the least-loaded replica avoids the dispatch entirely on the replicated path. This is what high-throughput MoE servers do for the most-frequently-accessed experts; the rest go through standard dispatch. See [serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) for the broader serving picture.

**Continuous batching helps MoE *more* than dense.** Because expert utilization scales with batch size (more tokens = better expert load balance = better GEMM grain), MoE inference benefits from aggressive continuous batching. vLLM's PagedAttention plus a router-aware batcher can lift effective MoE throughput 3–4× over a naive request-per-batch server. This composes with the [KV cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) techniques from the dense world; nothing about MoE breaks them.

| Workload | Recommended config | Why |
|---|---|---|
| Single-user interactive, short context | Replicated experts, no offload | Latency floor matters; VRAM cheaper than PCIe |
| Single-user interactive, long context | EP=8 inside NVL8 island, partial offload | Activation memory limits; cache covers prompt |
| Batch / offline scoring | EP=8 + NVMe offload, prefetch by router | Throughput-bound; PCIe latency amortized |
| Multi-tenant API | EP=8 with HybridEP if NVL72, predictive prefetch | Locality is workload-dependent |

## 13. A profiling-driven tuning loop

The thing that ties all of this together is a tuning loop. There is no single "best" configuration for an MoE model; the optimum depends on hardware, sequence length, model size, and load. The right operational discipline is to measure first, hypothesize, change one thing, re-measure.

![MoE tuning is a triage loop that maps the dominant wall to the matching technique](/imgs/blogs/optimizing-moe-training-and-inference-12.png)

The Megatron-Bridge guide formalizes this as a three-phase workflow. I'll restate it in the form I use:

1. **Feasibility on a fake cluster.** Run with `--fake-init-process-group` and the full parallelism configuration but no real cluster. Confirms the model fits and the code path executes. Catches OOMs and shape errors before you burn cluster minutes.
2. **Single-step profiling on the real cluster.** Run with Nsight Systems for 2–3 steps. Look at the trace. Identify the dominant wall: is the longest gap between GEMMs on the GPU stream A2A (communication wall), is the GPU stream mostly idle but tensor cores not saturated when active (compute wall), or are you OOMing or recomputing aggressively (memory wall)?
3. **Apply the matching fix and re-profile.** Comm wall → DeepEP/HybridEP, A2A overlap, shared-expert overlap. Compute wall → GroupedGEMM, router/permute fusion, FP8/NVFP4. Memory wall → memory-efficient permutation, optimizer offload, increase PP, increase CP for long context.

```bash
## Phase 1: fake-cluster feasibility
megatron-core-train --fake-init-process-group ... # remaining flags

## Phase 2: real-cluster profiling
nsys profile -o moe-profile.nsys-rep --capture-range=cudaProfilerApi \
  megatron-core-train --train-iters 5 --profile-step-start 2 --profile-step-end 3 ...

## Phase 3: apply fix; re-run phase 2
```

Iterate. Each pass should flip the dominant wall to a different one — a sign that the fix worked. When you can't make any single wall the dominant one, you're near the configuration optimum.

### 13.1. The flag set I keep coming back to

For a generic 100B-class MoE on H100 NVL8, this is my starting point; the cluster-specific flags get added after profiling:

```bash
## Communication
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep
--overlap-moe-expert-parallel-comm
--moe-shared-expert-overlap

## Compute
--moe-grouped-gemm
--moe-router-fusion
--moe-permute-fusion
--cuda-graph-impl transformer_engine --cuda-graph-mode partial

## Memory
--moe-permute-fusion-memory-efficient
--offload-optimizer-states

## Numerics
--fp8-format e4m3 --fp8-recipe blockwise
## router stays FP32 by default in Megatron-Core; do not override
```

## 14. Checkpointing, upcycling, and resharding

A practical concern that tends to get under-discussed: how do you save and restore an MoE? The answer is harder than it sounds because the optimizer state, the expert weights, and the parallelism layout are all coupled. A naive `torch.save(state_dict)` on a 685B MoE will produce per-rank files that are bound to the *exact* parallelism configuration that wrote them. If you want to resume on a different cluster shape — different EP, different PP, different DP — you cannot.

The Megatron-Core *distributed checkpoint* format solves this by saving a parallelism-agnostic logical layout and reconstructing the per-rank shards at load time. Each expert's weight tensor is stored as a single logical tensor with metadata about its sharding pattern; on load, the framework reshards as needed. This is what lets you train a model with EP=64 on one cluster and resume it with EP=32 on another, or scale up EP after a hardware refresh.

*Upcycling* is the more interesting trick. You can take a dense pretrained checkpoint and convert it to MoE without retraining from scratch. The recipe is to duplicate the FFN N times (one copy per expert), randomly initialize the router, and continue pretraining for a small number of tokens (typically 5–10% of the original training budget). The duplicated experts will diverge under the routing loss, and the resulting MoE recovers most of the dense model's quality plus the additional capacity from the new experts. This is how several open MoE releases were trained — it is dramatically cheaper than starting fresh, and the quality at the same active-parameter count usually matches a from-scratch MoE within a few percent.

```bash
## Upcycle a dense Llama-3-70B into a 4×70B-A70B MoE
megatron-core-upcycle \
  --source-checkpoint /ckpts/llama-3-70b \
  --target-num-experts 4 \
  --target-topk 1 \
  --noise-init expert_diverge \
  --output /ckpts/llama-3-4x70b-moe
## Then continue pretraining for ~5% of original tokens
```

The third concept here is *resharding for inference*. Training EP is rarely the right inference EP. Training optimizes for amortized cost across many tokens; inference optimizes for low per-request latency. A model trained with EP=64 PP=4 typically serves best at EP=8 PP=1 with replicated experts on the most-frequently-routed quarter of the expert set. The reshard is a one-time operation: load the distributed checkpoint, write out a new layout in the inference framework's format. If you skip this step, your inference latency will be 2–3× worse than necessary because you will be paying training-time A2A costs at request time.

## 15. Numerics: what stays in FP32, what survives FP4

A short but load-bearing section. The numeric recipe for MoE is more nuanced than for dense models because the router is a discrete-ish operation that is sensitive to noise.

**Always FP32.** The router (gate logits, top-k selection, softmax). The aux-loss-free expert biases. Any layer-norm statistics. The optimizer's master weights and momentum. Together these make up <1% of the model parameters and FLOPs, so the FP32 cost is negligible.

**FP8 (E4M3) is fine.** The expert FFN GEMMs (W1 and W2). The attention QKV and output projection. Embedding lookups (static FP8 with per-token scale). KV cache for serving (E5M2 if you can tolerate slightly higher dynamic range loss; E4M3 if precision matters more than range).

**FP4 / NVFP4 with care.** The expert FFN GEMMs only. Requires Random Hadamard Transforms applied to the activations and stochastic rounding in the accumulators. If either is missing, the FP4 outliers will swamp the GEMM accuracy after a few thousand training steps and you will see validation loss drift up monotonically. The first time I watched this happen, the team's hypothesis was a learning-rate bug; the actual cause was an RHT implementation that silently fell back to identity on certain hidden sizes.

The practical heuristic: if FP4 is a research project, run a 24-hour A/B against MXFP8 and compare validation loss curves. If they diverge by more than 0.5% relative after 10K steps, your FP4 plumbing is broken — debug before scaling up. If they match, FP4 is roughly 2× faster on B200 and saves about 40% of expert weight memory; it is worth the engineering investment.

```bash
## Validation: train both for 10K steps from the same init
megatron-core-train --fp8-format mxfp8 ...     | tee mxfp8.log
megatron-core-train --fp8-format nvfp4 \
  --fp8-rht --fp8-stochastic-round ... | tee nvfp4.log
## Compare validation loss curves; reject if relative gap > 0.5%
```

## 16. Production serving: prefill-decode disaggregation for MoE

The serving optimization that has the largest single effect on cost-per-token in 2026 is *prefill-decode disaggregation* — running the prefill phase (long context, batch-1 in the worst case, FLOP-bound) on a different GPU pool than the decode phase (short steps, large batch, bandwidth-bound). For MoE this is more impactful than for dense, because prefill and decode have *different* dominant walls.

Prefill is FLOP-bound: it is processing thousands of tokens at once, the per-expert GEMMs are large, and tensor cores can saturate. EP doesn't help much because the dispatch is amortized over a large per-step token count anyway; replication and TP work fine.

Decode is bandwidth-bound: each step processes 1–8 tokens, the per-expert GEMMs are tiny, and the dominant cost is fetching the expert weights from HBM (or, with offloading, from CPU). EP and replicated experts both help; the right choice depends on the cache hit rate.

Disaggregation lets you tune each pool independently. Prefill nodes run with TP=4, EP=4 (small, since prefill batches are big), no offload, on H100 SXM. Decode nodes run with EP=8, replicated experts for the top-25% most-routed, expert offloading for the rest, on either H100 or PCIe-tier hardware. The transfer between them is the KV cache; with [LMCache](/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive)-style cross-node KV transfer, this is bandwidth-bounded by IB and contributes a few hundred microseconds of TTFT, which is fine for almost any workload.

```bash
## Prefill pool config
sglang serve \
  --model deepseek-v3 --tp 4 --ep 4 --topk 8 \
  --enable-prefill-disagg --kv-transfer lmcache://decode-pool

## Decode pool config
sglang serve \
  --model deepseek-v3 --tp 1 --ep 8 --topk 8 \
  --enable-decode-disagg \
  --replicate-top-experts 64 --offload-cold-experts cpu \
  --expert-cache-size 96
```

The numbers I have seen on this: 1.6–2.4× cost reduction at the same SLA, depending on workload. The win is bigger on long-context workloads (where prefill cost is large in absolute terms) and on multi-tenant deployments (where decode batches consolidate across tenants). See [SGLang's serving techniques](/blog/machine-learning/mlops/the-techniques-using-in-sglang) for the broader disaggregation pattern.

## 17. Case studies from production

### 1. DeepSeek-V3 685B on GB300

A reference number to anchor against: 1,233 TFLOPS/GPU on GB300, 1,048 on GB200, [reported by NVIDIA on Megatron-Core](https://arxiv.org/abs/2603.07685). Configuration: EP=64 inside the NVL72 domain, HybridEP dispatcher, parallel folding (attention TP=4 with DP across the full domain, MoE EP=64), aux-loss-free balancing, NVFP4 on the expert GEMMs with RHT and stochastic rounding, MTP head on its own pipeline stage. The thing that made this work was *parallel folding* — without it, EP=64 forced TP=1 on attention, which hurt the attention throughput by ~30%. With folding, attention runs TP=4 inside an NVL8 sub-region of the domain, MoE runs EP=64 across the full 64 GPUs, and both are happy.

### 2. Qwen3-235B-A22B

974 TFLOPS/GPU on GB300, 919 on GB200. The interesting number here is the 235B-total / 22B-active ratio — much sparser than DeepSeek's 685B/37B. Sparser MoE is harder to optimize because the per-expert GEMM grain is smaller (more experts, fewer tokens per expert). The Qwen3 recipe leans heavily on GroupedGEMM and on a 128-expert fine-grained design that pushes the math-bound regime into B200 territory; on H100 the same model runs ~25% slower per token because the GEMM grain falls below H100's 128-token math-bound threshold. Lesson: fine-grained MoE chases B200 and beyond.

### 3. GPT-OSS 20B fine-tune with Unsloth

7× faster fine-tuning, 36% VRAM reduction at 16K context, on a single H100. The full Split-LoRA + Triton GroupedGEMM stack. The key practical observation: when fine-tuning a smaller MoE on a single GPU, EP doesn't apply (one rank, no dispatch). The wins come entirely from the kernel layer — collapsing the per-expert LoRA materialization, custom GEMM, and absorbing the routing weights into the activations. This is a useful demonstration that MoE optimization is not just multi-node lore; it pays back at single-GPU scale too.

### 4. Mixtral 8×7B inference with expert offloading

The first really practical demonstration that a 47B-total / 13B-active MoE could run on a 24 GB consumer GPU using expert offloading. The setup: shared FFN and attention on GPU, all 8 experts × 32 layers on CPU RAM, LRU cache of ~3 experts per layer on GPU. Hit rates above 92% on conversational workloads, decode latency around 60% of fully-resident — slow, but tractable. This is the configuration that opened MoE to the hobbyist GPU world and that the [APXML offloading chapter](https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading) builds on.

### 5. DBRX 132B — the EP-only generation

DBRX (Databricks, 2024) trained with EP-only parallelism, no parallel folding, no HybridEP. At 132B / 36B-active it was state of the art for its time, but the EP × DP = world_size constraint meant the team had to pick: either large EP (good for the FFN, hurts attention) or small EP (good for attention, leaves FFN underutilized). They picked something in the middle and ate the throughput hit. Parallel folding was the change that unblocked the next generation; the architectural lessons from DBRX about expert count and granularity transferred forward, but the parallelism strategy did not.

### 6. GLM-4.7 Flash on RTX PRO 6000 with Unsloth

2.1× speedup on a single PRO 6000 (48 GB). What's notable is not the multiplier but the *floor* — an A22B-class model fine-tuning on a single workstation card. The PRO 6000 has neither NVLink nor IB, so EP and any multi-node parallelism is irrelevant. All wins come from the kernel layer: GroupedGEMM, Split-LoRA, FP8 expert weights with BF16 LoRA. The blast radius of MoE optimization is broad: the same techniques that buy you 1,233 TFLOPS/GPU on GB300 also buy you a tractable training run on a single workstation.

### 7. A failed run: EP=128 across 16 NVL8 islands

I have personally watched a team try EP=128 with `EP × TP × PP = 128 × 4 × 4 = 2048 GPUs` across 16 NVL8 islands and 5 racks. The throughput was less than half the EP=32 baseline. The trace told the story: every layer paid 8 IB hops on the dispatch, the IB fabric was saturated 80% of the time, and the GPUs were idle waiting on A2A. The fix was not a tuning flag; it was rethinking the topology assignment. Switching to EP=8 with 16-way EDP, parallel-folded so attention got TP=4 + DP=64, recovered 1.9× throughput. Lesson: the interconnect topology is part of your model architecture choice, not an afterthought.

### 8. Long-context 128K MoE — the recompute trap

A pretraining team OOMed on 128K context with their MoE and reflexively turned on full SDPA recompute. Throughput collapsed by 35%. The Megatron-Bridge guide is explicit on this: do not recompute SDPA at long context — the recompute cost is huge, the memory savings are modest because the dominant memory consumer is *expert activations*, not attention. The actual fix was to bump CP from 2 to 8 (`CP ≈ seq_len/4096 = 32`, but they capped at 8 due to NVLink topology), turn on memory-efficient permutation, and offload optimizer states to CPU. Memory came back, throughput stayed, no recompute needed.

### 9. The capacity-factor regression

Production training of a 100B MoE was healthy at `cf=1.25` for the first 50K steps, then drift in expert load balance (the aux-loss-free bias was not yet stable) caused 2% of tokens to overflow. Validation loss spiked. The team's first response was to bump `cf` to 1.5, which fixed the loss but cost 12% throughput on the dispatch (more memory, more bytes over A2A). The right fix turned out to be dropless mode with paged stashing: no more dropped tokens, no more padded buffers, paged stashing kept activation memory under control. Lesson: capacity factor is a knob with two costs (memory vs. accuracy); dropless is the better operating point in 2026.

### 10. The aux-loss-free balancing roll-out

A 100B-class team migrating from aux-loss balancing to aux-loss-free saw a brief expert-utilization collapse: 30% of experts received < 5% of tokens for the first 10K steps after the switch, before the learnable bias caught up. The fix was a *warmup* of the bias term: initialize it from the running token-count statistics of the previous aux-loss training, then let it adapt. This eliminated the collapse and the resulting validation-loss spike. The lesson generalizes: any MoE-balancing change should be staged with a warmup that smooths the transient, because a bad expert distribution for even a few thousand steps leaves a measurable scar on validation loss that takes much longer to recover.

### 11. EP=64 in NVL72: what HybridEP actually does

A team migrating from H100 NVL8 to GB200 NVL72 expected a 2× speedup on the same EP configuration; they got 3.4×. The extra factor came from HybridEP keeping the *full* dispatch inside the NVL72 domain — no IB hop, ever, for any inter-rank A2A within the 72-GPU group. The H100 reference was paying an IB round-trip on roughly 7/8 of its A2As (dispatching to ranks outside the local NVL8 island); GB200 paid zero because the entire EP group fit in one NVL72 domain. The lesson is that hardware platform changes can shift optimization regimes more than software flags do; reach for HybridEP whenever your cluster supports NVL72.

### 12. The all-reduce-was-actually-an-A2A debugging story

A new training framework went live on a 64-GPU cluster and throughput was 40% below the reference. The team spent two days assuming a TP regression. Profiling showed something stranger: the timeline had 60 small allreduces per step that didn't appear in the reference. The cause: a default `True` flag in the framework's MoE config that ran an extra allreduce to "validate" the dispatcher's output for debugging. Disabling it recovered the 40%. The lesson: every collective on the timeline is on the critical path; if you don't know what one is, you have an unfound regression.

### 13. The router-precision regression

A team that had been running stable for months added a "small" change: cast the router GEMM to FP8 along with the expert GEMMs. Validation loss began drifting upward over the next 5K steps. The cause was straightforward in hindsight: at FP8, two close gate logits could be represented identically, and the top-k tie-breaking by index biased the router toward low-indexed experts. The expert load distribution skewed; the aux-loss-free bias couldn't keep up with the systematic error; quality degraded. Reverting the router to FP32 fixed it instantly. The lesson is non-negotiable: the router operates on a discrete decision and is far more sensitive to quantization than the expert GEMMs that surround it. The 0.1% of FLOPs you save by FP8-ing the router is not worth the failure mode.

### 14. The dispatch-buffer leak

A long-running MoE inference deployment slowly degraded throughput over weeks, eventually OOMing once a month. Profiling showed that the dispatch buffer was being allocated fresh on every request rather than reused. The framework had a "convenience" mode for variable-batch dispatch that allocated, used, and freed a buffer per layer per step. Replacing it with a pre-allocated arena that grew to the workload's high-water mark eliminated the leak and recovered 8% throughput as a side effect (no more allocator overhead in the hot path). MoE inference frameworks vary widely in their memory hygiene; this is one of the first things to audit on a new framework adoption.

### 15. The capacity-cliff at 1.0

A research team trying to push capacity factor to exactly 1.0 (the "no slack" point that is theoretically optimal under perfect load balance) saw their training loss diverge after about 2K steps. The aux-loss-free bias was almost balanced but not quite, and 0.5–1% of tokens overflowed every step. With dropping enabled, those tokens were lost; with dropless, they ran on a cold expert with no cached activation. Either way, the gradient signal was systematically biased. The fix was to keep capacity factor at 1.1 throughout training and lean on the bias to drive the actual load distribution toward 1.0 — the "slack" pays for the occasional miscount without changing training semantics. The lesson: capacity factor is a *system-level* knob, not an architectural constant. Treat it as a hyperparameter with a recommended range, not a target to optimize.

## 18. When to reach for MoE optimization, when to skip it

### Reach for the full optimization stack when …

- The model is ≥ 30B total parameters with ≥ 8 experts, or ≥ 100B with fine-grained 64+ experts.
- You're training on ≥ 32 H100s or any Blackwell cluster — below this scale, the fixed costs of EP and parallel folding don't pay back.
- Sequence length exceeds 8K, where CP and memory-efficient permutation start mattering.
- The model is intended for production serving with throughput SLAs, not just research-eval throughput.
- You have a profiler set up. Without one, you'll tune blind and waste cluster time.

### Skip and use simpler defaults when …

- You're doing exploratory pretraining at < 10B total params. Dense is simpler and the MoE wins are marginal at this scale.
- You're fine-tuning a small MoE on a single GPU — Unsloth's stack handles it; you don't need to think about EP, dispatch, or parallel folding.
- Your fabric is consumer-tier (no NVLink across cards, no IB). MoE multi-GPU on PCIe-only is dominated by transfer cost; you want a single-card or NVLink-pair workflow.
- The use case is latency-sensitive single-user inference at low context. A dense 13B fits the latency budget; an MoE has to amortize router and dispatch overhead that a dense model doesn't pay.
- You're prototyping. The full MoE stack — parallel folding, GroupedGEMM autotuning, FP8 plumbing, pipeline layout strings — has a real ramp-up cost. Stand up a dense baseline first, validate the data and model code, then graduate to MoE once the rest of the system is debugged. The hardest MoE bring-up I have ever helped on was one where data-pipeline bugs and MoE optimization bugs were entangled; we wasted three days separating signal from noise that the team would have caught in an hour on a dense baseline.

The summary, if I had to pick a single sentence: MoE is a systems problem first and a modeling problem second, and the practitioner who treats it as the latter will spend most of their cluster budget paying for the difference.

## 19. Anti-patterns and traps

A consolidated list of the patterns I have watched cost real cluster time. Most of them sound obvious in hindsight; few of them are obvious before you've burned a week on them.

**Setting EP greater than the expert count.** Megatron-Core won't stop you from `EP=128, num_experts=64`; it will silently fall back to a sub-optimal partition where each rank shares ownership of a fractional expert. Throughput tanks. The constraint is `EP ≤ num_experts`, with equality being the highest-throughput configuration when the cluster permits it.

**Mixing dropping and aux-loss-free.** Aux-loss-free balancing assumes dropless mode — the bias term is calibrated against the actual token counts received. Running it with dropping mode causes the bias to chase a moving target (the dropped-token distribution), and balance never converges. Use dropping only with the classical aux loss; use aux-loss-free only with dropless.

**Forgetting that PP rotates which stages are MoE.** With pipelined MoE, different stages may have different parallelism configurations (e.g., dense layers on the first stage, MoE layers on later stages). The optimizer-state offload, the FP8 recipe, and the recompute policy may apply only to certain stages. Verify per-stage memory profiles, not just per-rank.

**Re-running the autotuner on every job.** Unsloth's grouped-GEMM autotuner takes 2 minutes; it is meant to run *once* per shape class and persist its picks. Some users were inadvertently re-running it on every restart, eating 2 minutes of wall-clock per launch. Cache the autotuner output to disk and pin it.

**Treating MoE inference frameworks as drop-in replacements for dense ones.** vLLM, SGLang, TensorRT-LLM, and TGI each have different MoE optimization maturity. As of mid-2026, SGLang has the most mature MoE serving stack (HybridEP, expert replication, prefill-decode disaggregation); vLLM is rapidly catching up. Audit before adopting; the framework-vs-framework gap on MoE is much wider than on dense.

**Reasoning about throughput from a 100-step run.** MoE training has a longer warmup than dense — caches need to fill, NCCL needs to settle, the autotuner needs to pick tile configs, the load distribution needs to balance under aux-loss-free. The first 500 steps are not representative. Always profile from step 1000+ for production tuning decisions.

**Conflating MoE quality regressions with optimization regressions.** When validation loss drifts up after an optimization change, the first hypothesis should be a numerics or balance regression (router precision, capacity factor, drop policy) before assuming the model is "broken." The reverse is also true: when quality is unexpectedly *good* after an optimization change, double-check that you did not silently disable an aux loss or load-balancing mechanism that is now letting tokens train on memorized experts.

**Assuming linear scaling.** MoE rarely scales linearly past 256 GPUs without effort. The scaling efficiency depends on EP × A2A latency; doubling the cluster size with the same EP roughly halves the per-rank A2A bandwidth requirement, but only if the topology cooperates. Plan for 0.85× scaling efficiency per doubling, not 1.0×, and treat anything above as a happy surprise.

## Further reading

- [Megatron-Bridge MoE optimization guide](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/moe-optimization.html) — official source for the flags discussed above.
- [Scalable Training of Mixture-of-Experts with Megatron Core (arXiv:2603.07685)](https://arxiv.org/abs/2603.07685) — the paper behind the GB300 numbers.
- [Unsloth: faster MoE](https://unsloth.ai/docs/basics/faster-moe) — Split-LoRA and Triton kernel details.
- [APXML expert-offloading chapter](https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading) — the practical inference-side companion to this article.
- Sibling posts on this blog: [MoE LLM architecture, training, and fine-tuning](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies), [Modern LLM architectures: Qwen, Llama, Gemma, DeepSeek](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek), [Optimizing LLM inference: a complete guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide), [TorchTitan: a tour of distributed training primitives](/blog/machine-learning/training-techniques/torch-titan), [Serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems).

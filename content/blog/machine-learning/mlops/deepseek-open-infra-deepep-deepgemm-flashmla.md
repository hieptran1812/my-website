---
title: "Reading DeepSeek's Open Infrastructure: DeepEP, DeepGEMM, FlashMLA, and DualPipe as the V3 Paper's Footnotes"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A systems and MLOps deep-dive that maps each claim in the DeepSeek-V3 technical report to the open-sourced kernel or library that implements it — FP8 GEMM to DeepGEMM, cross-node all-to-all to DeepEP, MLA decode to FlashMLA, pipeline overlap to DualPipe, and storage to 3FS."
tags: ["deepseek", "deepseek-v3", "mlops", "moe", "fp8", "deepep", "deepgemm", "flashmla", "dualpipe", "3fs", "gpu", "distributed-training"]
category: "machine-learning"
subcategory: "MLOps"
author: "Hiep Tran"
featured: true
readTime: 50
---

Most technical reports cite their footnotes. DeepSeek shipped theirs as Git repositories.

When the DeepSeek-V3 technical report (arXiv 2412.19437) landed, the headline was the **$5.6M training cost on 2,048 throttled H800 GPUs**. We covered the model-side of that story — FP8 training, Multi-Token Prediction, loss-free balancing — in the companion post on [how DeepSeek-V3 trained a 671B model for $5.6M](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing). That post read V3 as a *model*. This one reads it as a *deployed system*, and the deployment is what makes the cost number believable.

Here is the thing that almost nobody noticed at the time: every load-bearing systems claim in that report — "we overlap communication with computation," "our FP8 GEMM hits near-peak throughput," "our all-to-all is nearly free," "our MLA decode is memory-bandwidth-bound" — reads in the paper like an assertion you are asked to take on faith. Three months later, during **Open Infra Week (February 2025)**, DeepSeek open-sourced the actual code behind each assertion: **DeepEP** (the expert-parallel comms library), **DeepGEMM** (the FP8 matmul library), **FlashMLA** (the MLA decode kernel), **DualPipe** (the bidirectional pipeline schedule), and **3FS** (the file system), plus **EPLB** (an expert-parallel load balancer) and **Smallpond** (a data framework on 3FS).

> The open-infra repos *are* the V3 paper's footnotes. Each paper claim maps to shipping code, 1:1. You no longer have to trust the report — you can `git clone` it.

That 1:1 mapping is the spine of this post. We will walk the stack from the bottom up, and for each layer we will (a) quote the V3 report's claim, (b) point at the repo that implements it, (c) read enough of the code to understand the mechanism, and (d) extract the reusable engineering principle. The diagram below is the mental model — internalize it and the rest of the post is a guided tour of its layers.

![The DeepSeek open-infra stack: hardware at the base, then comms, compute kernels, schedule, and storage, each layer mapping to one open-sourced repo.](/imgs/blogs/deepseek-open-infra-deepep-deepgemm-flashmla-1.webp)

The stack reads bottom-to-top: throttled H800 hardware forces every layer above it to economize on data movement. The comms layer (DeepEP) rations the slow inter-node link. The compute kernels (DeepGEMM, FlashMLA) extract near-peak throughput from the silicon. The schedule (DualPipe) hides the comms behind the compute. And the storage layer (3FS) feeds the whole thing fast enough that data loading never becomes the bottleneck. Every layer exists to serve one constraint: *the interconnect is too slow, so move less and overlap everything.*

## Why this stack is different from the assumption

The default mental model for "how a frontier lab trains a giant MoE" is wrong in specific, instructive ways. Here is the assumption-versus-reality table that frames the entire post:

| What you'd assume | The naive approach | What DeepSeek actually did |
|---|---|---|
| Tensor parallelism splits the big matmuls | TP all-reduce after every layer | **No TP at all** — the all-reduce would crush the throttled link |
| FP8 needs a vendor library | Wait for cuBLAS FP8 | **DeepGEMM**, ~300 lines, JIT-compiled, 1350+ TFLOPS |
| All-to-all for MoE is unavoidable overhead | Eat the comms cost | **DeepEP** + node-limited routing makes it nearly free |
| Decode is compute-bound | Throw FLOPS at it | **FlashMLA** treats decode as memory-bandwidth-bound (3000 GB/s) |
| Pipeline parallelism has a big bubble | Accept ~1/PP idle | **DualPipe** feeds both ends, shrinks the bubble to near zero |
| Storage is "just NFS plus a cache" | Mount a shared filer | **3FS** at 6.6 TiB/s aggregate read, KV-cache at 40+ GiB/s/client |
| Cheap means worse hardware | Buy fewer, slower GPUs | A **two-layer Fat-Tree** topology that halves cluster cost at ~83% perf |

Notice the pattern in the right column: nearly every entry is a *systems* decision, not a model decision. DeepSeek did not win because their transformer was cleverer. They won because they treated the H800 cluster — its throttled NVLink, its single 200 Gbps NIC per node — as a fixed adversary and engineered the whole software stack to beat it. The export-control throttle that was supposed to slow them down instead forced a co-design discipline that most labs, sitting on un-throttled H100s, never bothered to acquire.

The rest of this post is structured as a tour of that right column. We will spend the most time on the three repos that did the most work — DeepEP, DeepGEMM, and DualPipe — and close with the cluster economics that make the whole thing affordable.

## The claim-to-repo map

**Rule of thumb: when a paper says "we are efficient" without showing you the kernel, treat it as marketing until proven otherwise. DeepSeek proved otherwise by shipping the kernels.**

Before we go deep on any single layer, it is worth seeing the full mapping laid out, because the discipline of the mapping is itself the lesson. Most published systems work gives you a paper *or* a code release, rarely both, and almost never with a clean correspondence between the two. DeepSeek's release is unusual in that you can take a sentence from the report, find the repo, and read the implementation in an afternoon.

![Paper claim to shipping repo, 1:1 — FP8 matmul maps to DeepGEMM, cross-node all-to-all to DeepEP, MLA decode to FlashMLA, pipeline overlap to DualPipe plus 3FS storage.](/imgs/blogs/deepseek-open-infra-deepep-deepgemm-flashmla-2.webp)

Read the grid left-to-right, row by row. The report claims **per-tile block-scaled FP8 with FP32 promotion** — that is DeepGEMM, ~300 lines of core logic, hitting 1350+ FP8 TFLOPS on Hopper. The report claims a **cross-node MoE all-to-all dispatch and combine** — that is DeepEP, doing NVLink intranode and RDMA internode with native FP8 dispatch on roughly 20 SMs. The report claims **MLA decode under KV-cache compression** — that is FlashMLA, 3000 GB/s memory-bound and 580 BF16 TFLOPS compute-bound. And the report's **pipeline overlap plus training data store** maps to DualPipe (the bidirectional schedule) plus 3FS (6.6 TiB/s aggregate read). Seven claims, seven repos. No hand-waving in between.

The significance of this release pattern is worth stating plainly, because it is rare. Most frontier labs publish a report that asserts efficiency and a model checkpoint you can run, but the *systems* claims — the kernels, the schedule, the collective — stay proprietary, and you are left to reverse-engineer them from latency measurements. DeepSeek inverted that: they kept some training details lighter than a reproducibility purist would want, but they open-sourced the exact systems machinery that the report's numbers depend on. The result is that a skeptical reader can do something better than trust or replicate from scratch — they can read the kernel that produced the number and check that the number is achievable on their own silicon. For a field where "we are efficient" is frequently a marketing claim, shipping the footnotes as Git repositories is a meaningfully higher standard of evidence, and it is the standard the rest of this post holds each claim to.

Here is the correspondence in table form, with the exact section of the report each repo grounds:

| V3 report claim | Open-infra repo | Headline number | What it proves |
|---|---|---|---|
| Fine-grained FP8 GEMM, FP32 promotion every 128 elements | **DeepGEMM** | 1350+ FP8 TFLOPS, ~300 lines | The FP8 throughput in the report is real and reproducible |
| Efficient cross-node all-to-all for MoE | **DeepEP** | ~20 SMs saturate IB+NVLink | The "nearly free" all-to-all is an actual library |
| MLA decode is memory-bandwidth-bound | **FlashMLA** | 3000 GB/s, 580 BF16 TFLOPS | The decode latency claims hold on H800 |
| Computation-communication overlap via pipeline | **DualPipe** | near-zero all-to-all bubble | The overlap is a concrete schedule, not aspiration |
| High-throughput data loading | **3FS** | 6.6 TiB/s read on 180 nodes | Storage never bottlenecks the training run |
| Expert load balancing in deployment | **EPLB** | rebalances expert replicas | The loss-free balancing has a serving-side twin |

We will not cover EPLB and Smallpond in their own sections — they are the supporting cast — but they belong in the map because they close the loop. EPLB is the deployment-time counterpart to the auxiliary-loss-free balancing we discussed in the [V3 training deep-dive](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing): the model is trained to spread tokens across experts, and at serving time EPLB places redundant expert replicas to keep every GPU busy. Smallpond is the data-prep framework that sits on 3FS. Both exist because a frontier run has no weak links: every layer of the stack has to keep pace with the GPUs, or the GPUs idle and the dollar-per-token math falls apart.

## 1. The hardware floor: why the throttle dictates everything

**Rule of thumb: design for your slowest link, not your fastest one. On an H800 cluster the slowest link is the inter-node InfiniBand, and it is roughly 3x slower than intra-node NVLink.**

Before any software, understand the silicon. The V3 cluster is 2,048 H800 GPUs. The H800 is the export-compliant H100: the SXM variant has its NVLink bandwidth cut roughly in half versus the H100, and the cluster's inter-node fabric is InfiniBand. The practical bandwidth hierarchy that the entire software stack is built around looks like this:

- **Intra-node NVLink**: roughly 160 GB/s of usable all-to-all bandwidth between GPUs in the same server.
- **Inter-node InfiniBand**: roughly 50 GB/s per direction across the network.

That is a ~3.2x ratio between the fast link and the slow link. Every architectural decision above the hardware exists to keep traffic on the fast link and to ration the slow one. When you read "no tensor parallelism" later, the reason is this ratio: a TP all-reduce runs after *every* layer and hammers the slow link in a way that cannot be hidden. When you read "node-limited routing," the reason is this ratio: capping the number of nodes a token visits caps the number of slow-link hops.

A quick worked example makes the stakes concrete. Suppose a single forward microbatch needs to move 2 GB of activations across GPUs for its MoE all-to-all. On NVLink at 160 GB/s, that is 12.5 ms. On InfiniBand at 50 GB/s, that is 40 ms. If your matmul for the same microbatch takes 30 ms of compute, then NVLink traffic *hides* under compute (12.5 ms < 30 ms) but InfiniBand traffic *exposes* a 10 ms stall (40 ms > 30 ms). Multiply that stall by tens of thousands of steps and you have lost weeks. The whole stack is an answer to "how do we keep the 40 ms number from ever appearing on the critical path."

The reason tensor parallelism is uniquely poisonous here, and not just expensive, is *where* its communication lands. Tensor parallelism splits a single matmul across GPUs and must all-reduce the partial results *before the next operation in the same layer can start* — the all-reduce sits squarely on the critical path of the forward pass, with no independent work to hide it behind. Expert parallelism's all-to-all, by contrast, lands at the dispatch/combine boundary, which the schedule can shadow with the attention or MLP of an adjacent microbatch. Same fabric, same bandwidth, but one collective is hideable and the other is not. That is the structural reason DeepSeek's parallelism strategy is "pipeline plus expert, never tensor": they kept only the communication primitives that the DualPipe schedule could overlap, and threw out the one it could not. The hardware throttle did not just make TP slow — it made TP categorically the wrong tool, and recognizing that is half the design.

You can confirm the bandwidth hierarchy on any Hopper cluster before you write a line of model code:

```bash
nvidia-smi topo -m                      # intra-node NVLink topology + link speeds
  # "NV#" entries between GPUs mean NVLink; "SYS"/"NODE" means a PCIe/IB hop.

nvidia-smi nvlink --status -i 0         # per-GPU NVLink status + bandwidth counters

  # Inter-node InfiniBand link rate (per port): expect ~200 Gb/s = 25 GB/s line
  # rate, ~50 GB/s aggregate with dual-port or bidirectional accounting.
ibstat | grep -A3 "Port 1"
```

If `nvidia-smi topo -m` shows `NV8` between GPUs on the same node and `SYS` to anything off-node, you have exactly the topology DeepSeek designed against: fast within the box, slow across the wire. Internalize that asymmetry and the rest of the design choices stop looking exotic and start looking inevitable.

## 2. DeepGEMM: the FP8 matmul, in ~300 lines

**Rule of thumb: a GEMM library you cannot read is a GEMM library you cannot trust at FP8. DeepGEMM's whole pitch is that the core logic fits in your head.**

The V3 report's most aggressive claim is that it trained in FP8 without losing accuracy, using **fine-grained quantization**: per-tile (1x128) scaling for activations and per-block (128x128) scaling for weights, with **FP32 accumulation promoted every 128 elements** to fight the limited dynamic range of FP8. We unpacked the *why* of that scheme in the [V3 training post](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing). DeepGEMM is the *how*: it is the FP8 GEMM library that implements exactly that scaling and promotion, for both dense layers and the grouped/MoE case.

The headline numbers: **up to 1350+ FP8 TFLOPS on Hopper**, with **core logic in roughly 300 lines**, **fully JIT-compiled** at runtime. The 300-line claim is the interesting one. cuBLAS and CUTLASS are tens of thousands of lines of template machinery. DeepGEMM deliberately rejects the template-heavy CUTLASS style in favor of a single, readable kernel that is compiled on demand for the exact shapes you hand it.

### Why JIT, and why it is not crazy

The standard objection to JIT-compiling a GEMM is "compilation latency will kill you." DeepGEMM's answer is that the kernel is small enough to compile in well under a second, the result is cached, and the payoff is that the compiler sees the *actual* matrix shapes as compile-time constants. When `M`, `N`, and `K` are constants, the compiler can fully unroll loops, pick the optimal tile and pipeline-stage counts, and eliminate the bounds checks that a shape-generic kernel must carry. You trade a one-time sub-second compile for a kernel specialized to your problem.

```python
import torch
import deep_gemm
  # DeepGEMM, dense FP8 GEMM. The library JIT-compiles a kernel specialized to
  # the (m, n, k) shapes the first time it sees them, then caches the result.

m, n, k = 4096, 7168, 2048
  # Activations: per-token (1x128) scaled FP8; weights: per-block (128x128) scaled.
lhs_fp8, lhs_scale = deep_gemm.per_token_cast_to_fp8(torch.randn(m, k, device="cuda"))
rhs_fp8, rhs_scale = deep_gemm.per_block_cast_to_fp8(torch.randn(n, k, device="cuda"))
out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

  # The matmul accumulates in FP32 and promotes every 128 elements along k,
  # exactly the scheme the V3 report describes. Output is bf16.
deep_gemm.gemm_fp8_fp8_bf16_nt((lhs_fp8, lhs_scale), (rhs_fp8, rhs_scale), out)
```

The `_nt` suffix is the operand layout (non-transposed lhs, transposed rhs), and `gemm_fp8_fp8_bf16` reads as "FP8 times FP8, accumulate, emit BF16." The two-tuple operands carry the quantized tensor *and* its scale factors, because in fine-grained FP8 the scale is not a single scalar — it is a tensor of per-tile scales that the kernel must apply as it accumulates.

### The MoE path: grouped GEMM with two layouts

V3 is a Mixture-of-Experts model, so the GEMM that matters most is not the dense one — it is the **grouped GEMM**, where a single launch computes one matmul per expert over a ragged set of tokens. DeepGEMM ships two grouped variants, and the distinction is exactly the training-versus-inference split:

```python
  # Contiguous-layout grouped GEMM for TRAINING / prefill. Tokens for each
  # expert are laid out contiguously; m_indices tags each token's group.
  # One launch computes all experts.
deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
    (lhs_fp8, lhs_scale), (rhs_fp8, rhs_scale), out, m_indices=m_indices)

  # Masked-layout grouped GEMM for DECODE. The per-expert token counts are not
  # known at launch time (they depend on routing the current batch), so a mask
  # selects the live rows. This is the CUDA-graph-friendly path.
deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
    (lhs_fp8, lhs_scale), (rhs_fp8, rhs_scale), out, masked_m=masked_m, expected_m=expected_m)
```

The contiguous version assumes you can sort tokens by expert ahead of time — true during training and prefill, where you have the whole batch. The masked version handles the decode case, where each step routes one token per sequence to its top-k experts and the per-expert counts are only known at runtime; the mask lets the kernel run inside a CUDA graph with a fixed launch shape. This is the kind of distinction that only shows up when you ship the code: the report says "grouped GEMM," the repo shows you that "grouped GEMM" is actually *two* kernels with different launch contracts.

### The Hopper-specific tricks that get to 1350 TFLOPS

It is worth being concrete about *where* the throughput comes from, because "1350 FP8 TFLOPS" is not free — it is the sum of several Hopper-specific moves that DeepGEMM stacks. First, **TMA (Tensor Memory Accelerator)** does the global-to-shared-memory copies asynchronously, so the load of the next tile overlaps the MMA of the current one; on Ampere you would burn warp cycles on the copy. Second, the kernel uses **warpgroup-level `wgmma` instructions**, which issue a matmul across a whole warpgroup (four warps) in one instruction rather than per-warp `mma`. Third — and this is the FP8-specific part — it does **two-level accumulation**: the tensor cores accumulate in their native low-precision register format for 128 elements along the contraction dimension, then *promote* that partial sum into an FP32 accumulator before continuing. That promotion every 128 elements is the exact mechanism the V3 report calls out, and it is what keeps FP8's narrow mantissa from losing bits across a long `k` reduction.

There is also a **persistent-kernel scheduling** trick: rather than launch one thread block per output tile, DeepGEMM launches a fixed number of persistent blocks that loop over tiles, which keeps the SMs saturated and hides the tail effect where the last few tiles would otherwise leave SMs idle. For the grouped/MoE case it adds a tile scheduler that assigns each persistent block a stream of (expert, tile) pairs so that experts with few tokens do not strand an SM. None of this is novel in isolation — CUTLASS does all of it — but DeepGEMM's contribution is doing it in ~300 readable lines you can actually modify, rather than behind a wall of C++ templates.

### Benchmarking it honestly

The right way to trust the 1350 number is to reproduce it on your shapes, not DeepSeek's. The library ships a test harness, and the pattern for a fair measurement is to warm up the JIT, then time with CUDA events across enough iterations to amortize launch overhead:

```python
import torch
import deep_gemm
from deep_gemm.utils import bench_kineto

m, n, k = 4096, 7168, 2048
lhs = deep_gemm.per_token_cast_to_fp8(torch.randn(m, k, device="cuda"))
rhs = deep_gemm.per_block_cast_to_fp8(torch.randn(n, k, device="cuda"))
out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

def run():
    deep_gemm.gemm_fp8_fp8_bf16_nt(lhs, rhs, out)

run()                                   # warm the JIT + cache the compiled kernel
t = bench_kineto(run, "fp8_gemm")       # median kernel time over many launches
tflops = 2 * m * n * k / t / 1e12       # 2*M*N*K flops per GEMM
print(f"{tflops:.0f} FP8 TFLOPS at ({m},{n},{k})")
```

The `2 * m * n * k` is the standard GEMM flop count (one multiply and one add per inner-product element). Run this across the shapes your model actually uses — the per-expert FFN dimensions, not a square benchmark matrix — and you will see the throughput is shape-dependent: tall-skinny grouped GEMMs with small per-expert token counts will land well below 1350, which is exactly why the masked decode path exists and why you should measure your shapes rather than quoting the headline.

### When DeepGEMM is the wrong choice

The second-order consequence worth flagging: DeepGEMM is Hopper-specific and FP8-specific. It uses Hopper's tensor memory accelerator (TMA) and the warpgroup-level MMA instructions. On an Ada or Ampere card it does not apply, and for BF16-only workloads you are better served by the vendor library. The 300-line readability is bought partly by *not* trying to be portable. If you need one GEMM library across five GPU generations and four dtypes, this is not it; if you are running FP8 on Hopper and want to actually understand and tune your matmul, it is close to ideal.

## 3. DeepEP: making the all-to-all nearly free

**Rule of thumb: in a large MoE, the all-to-all is the whole ball game. If you cannot hide it, your expensive GPUs spend their lives waiting on the network.**

This is the most important repo in the release, because it implements the single hardest claim in the report: that the cross-node MoE all-to-all is *nearly free*. **DeepEP is the first open-source expert-parallel communication library for MoE** — it is the productionized version of the dispatch/combine all-to-all described in the V3 report. It does intra-node communication over NVLink, inter-node over RDMA, and supports **native FP8 dispatch** so that the payload crossing the wire is half the size of a BF16 dispatch.

Let us first see why the naive approach fails, then how DeepEP and node-limited routing fix it.

![Tensor-parallel all-reduce versus DeepEP plus DualPipe: replacing per-layer all-reduce with node-limited all-to-all overlapped under the pipeline schedule unblocks the throttled link.](/imgs/blogs/deepseek-open-infra-deepep-deepgemm-flashmla-4.webp)

The left column is the textbook path most labs take. Tensor parallelism splits every matmul across GPUs, which is fine for compute but forces an **all-reduce after every layer** to recombine the partial sums. On an H800 cluster that all-reduce floods the 50 GB/s InfiniBand link, the comms block compute, the GPUs stall on the wire, and throughput is crushed by the throttled interconnect. The right column is DeepSeek's path: no tensor parallelism at all; experts are sharded and tokens are *routed* to them via a **node-limited all-to-all** where each token visits at most four nodes; and the DualPipe schedule overlaps the dispatch and combine behind compute so the all-to-all bubble approaches zero as the model scales.

### Node-limited routing: the key constraint

The report describes a routing constraint that is easy to skim past but does enormous work: **each token is dispatched to at most M=4 nodes**. Tokens select their top-k experts, but the experts a token can reach are constrained so that the token never touches more than four physical nodes. Why four? Because the number of *inter-node* hops a token makes is bounded by the number of distinct nodes it visits. Cap the node count and you cap the slow-link traffic per token. The intra-node fan-out to multiple experts on the same node is free-ish, because it rides NVLink.

This is the crucial co-design point: the routing algorithm in the *model* is shaped by the bandwidth ratio in the *hardware*. You cannot understand node-limited routing without the bandwidth hierarchy from Section 1, and you cannot understand DeepEP without node-limited routing.

![One token's hop budget under node-limited routing: the token crosses the slow 50 GB/s InfiniBand link once per node, then rides 160 GB/s NVLink to the local experts, exploiting the ~3.2x ratio.](/imgs/blogs/deepseek-open-infra-deepep-deepgemm-flashmla-6.webp)

The hop budget above shows exactly what the cap buys. A routed token (7 KB in FP8) is first capped to at most M=4 distinct nodes; for each of those nodes it pays exactly **one IB hop at the slow ~50 GB/s rate**. Once it lands on the target node's GPUs, intra-node **NVLink fans it out to the local experts at ~160 GB/s** — and because that NVLink forward overlaps the IB hop, the slow-link cost is hidden. The ~3.2x bandwidth ratio is exploited rather than wasted: the fast link does the bulk of the work, the slow link does one cheap hop per node. Without the cap, a token that scattered to experts on eight different nodes would pay eight IB hops, and the slow link would dominate the whole dispatch.

A worked example makes the cap's value quantitative. Suppose a token's hidden state is 7 KB (in FP8) and it activates 8 experts. If those 8 experts were scattered across 8 distinct nodes, the token crosses IB 8 times — 56 KB of inter-node traffic at 50 GB/s, ~1.12 microseconds of slow-link time per token. Now cap it at 4 nodes with 2 experts each: the token crosses IB only 4 times (28 KB at 50 GB/s, ~0.56 microseconds) and the second expert on each node is reached over NVLink for ~free. Halving the node count halves the slow-link traffic, and across millions of tokens per step that is the difference between a comms phase that hides under compute and one that does not. The cap is not a quality compromise; it is the lever that keeps the slow link off the critical path.

### Warp specialization: how DeepEP hits the bandwidth ceiling

DeepEP's implementation trick is **warp specialization**. Instead of one monolithic kernel that does everything, it assigns different warps to different stages of the all-to-all pipeline, communicating across roughly 10 channels. The result is that only **about 20 SMs** are needed to saturate both the InfiniBand and the NVLink fabrics — leaving the other ~110 SMs on the H800 free to do the matmuls that the all-to-all is feeding. That is the mechanism behind "nearly free": the comms barely use the GPU's compute resources.

![DeepEP warp specialization: warps split into IB-send, IB-to-NVLink forward, and NVLink-receive roles across about ten channels, so only ~20 SMs saturate both fabrics.](/imgs/blogs/deepseek-open-infra-deepep-deepgemm-flashmla-5.webp)

Trace the graph left-to-right. Routed tokens (capped at M=4 nodes) split into two concerns: **IB-send warps** push the payload over RDMA, and the **FP8 dispatch** halves that payload before it goes. At the target node, **IB-to-NVLink forward warps** receive the inter-node traffic and immediately forward it onto the intra-node NVLink fabric, organized into about ten warp-specialized channels. Finally, **NVLink-receive warps** fan the tokens out to the local experts. The punchline node on the right: only ~20 SMs are needed to saturate IB plus NVLink. The combine direction runs the same pipeline in reverse — NVLink-gather, NVLink-to-IB, IB-receive.

The reason warp specialization wins here is the same reason it wins in FlashAttention-style kernels: the IB send, the IB-to-NVLink forward, and the NVLink receive are *different* operations with *different* bottlenecks, and a warp dedicated to one of them keeps its pipeline full instead of stalling at a phase boundary. A monolithic kernel that did send-then-forward-then-receive in lockstep would idle the network half the time.

### Two kernel families: training-throughput vs decode-latency

DeepEP ships two distinct kernel sets, mirroring the training/inference split we saw in DeepGEMM:

- **High-throughput kernels** for training and prefill, where you have large token batches and want to maximize bytes-per-second.
- **Low-latency kernels** for decode, where each step moves a tiny payload and you care about the *latency* of a single dispatch/combine round-trip, not its bandwidth. These use a pure RDMA path and are designed to be hidden inside the decode loop.

Here is the shape of the training-side API. The combine is the transpose of the dispatch, and both return handles you use to overlap them with compute:

```python
import deep_ep

  # One Buffer per process group; it owns the NVLink + RDMA scratch.
buffer = deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)

  # DISPATCH: route each token's hidden state to its selected experts. topk_idx
  # came from the gating network; node-limited routing caps it to <= 4 nodes.
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens, handle, event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
    num_tokens_per_expert=num_tokens_per_expert)

  # ... experts compute on recv_x here (the grouped GEMM from Section 2) ...

  # COMBINE: scatter expert outputs back to the originating tokens. `handle`
  # carries the routing metadata so combine is the exact inverse of dispatch.
combined_x, _, event = buffer.combine(expert_out, handle)
```

The `event` handles are the bridge to DualPipe: you launch the dispatch, do unrelated compute, and only block on the event when you genuinely need the dispatched tokens. That deferred-wait pattern is what lets the schedule hide the comms.

### The low-latency decode path and the overlap budget

The decode kernels deserve their own look, because decode is where MoE serving usually falls apart. At decode time each step routes a *single* new token per sequence to its top-k experts, so the dispatch payload is tiny — a few KB — but you do it on *every* step, thousands of times per sequence. A throughput-optimized kernel that is great at moving gigabytes is the wrong tool: its setup cost dominates when the payload is 4 KB. DeepEP's low-latency kernels take a pure-RDMA path with minimal launch overhead, and crucially they support a **hook-based overlap** mode where the communication is issued and then a hook lets you run the attention of the *next* microbatch while the RDMA round-trip is in flight.

```python
  # Low-latency decode dispatch. The returned `hook`, when called, waits for
  # the RDMA round-trip; between the dispatch call and the hook you run other
  # compute (e.g. the next layer's attention) so the comms are fully hidden.
recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts)
  # ... run unrelated compute here; the dispatch RDMA is in flight ...
hook()                                  # now block until the tokens have arrived
```

Here is the overlap budget that makes this pay. Suppose a decode step's attention plus MLP compute takes 250 microseconds, and a single dispatch+combine RDMA round-trip takes 150 microseconds. Done serially that is 400 microseconds per step. With the hook pattern, the 150 microseconds of comms overlaps the 250 microseconds of compute, so the step is bounded by the 250-microsecond compute — a 1.6x decode speedup from overlap alone, before any kernel tuning. The catch is that the overlap only works if there is *independent* compute to run during the round-trip; on the very first layer of the first microbatch there is nothing to hide behind, which is why the schedule matters as much as the kernel.

### Second-order gotcha: FP8 dispatch is lossy, and that is fine

FP8 dispatch halves the wire payload, but it means the tokens that arrive at the experts are FP8-quantized. For training this is consistent with the FP8 GEMM that follows — the tokens were going to be quantized anyway. For the combine direction you typically want higher precision, because you are accumulating weighted expert outputs and the rounding compounds. DeepEP lets you choose precision per direction; the default is FP8 dispatch, BF16 combine. Getting this wrong — FP8 on the combine — shows up as a slow accuracy degradation that is maddening to debug because the kernel is "correct," just lossy in the wrong place.

### EPLB: the serving-side twin of loss-free balancing

DeepEP moves tokens efficiently, but it cannot help you if the tokens are unevenly distributed across experts — a GPU holding a hot expert becomes the straggler that the whole all-to-all waits on. This is where **EPLB (Expert-Parallel Load Balancer)** closes the loop. In the [V3 training deep-dive](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) we covered auxiliary-loss-free balancing, which uses a per-expert bias adjusted during training to keep token assignment roughly uniform. EPLB is the *deployment-time* counterpart: it watches the observed expert load and physically *replicates* the busiest experts onto more GPUs, then routes a share of their tokens to the replicas.

The mechanism is a "redundant experts" strategy. Suppose expert 7 is receiving 3x the average token load. EPLB places a second copy of expert 7's weights on another GPU and splits expert 7's tokens between the two copies, so neither GPU is the straggler. It runs in two modes: a **hierarchical** policy that keeps replicas of a node's experts within that node (so the replication does not add inter-node hops, preserving the node-limited routing invariant), and a **global** policy for when the expert-parallel group spans the whole cluster. The choice mirrors the bandwidth hierarchy yet again: replicate locally if you can, because a replica on a remote node would reintroduce the slow-link traffic that node-limited routing worked so hard to avoid.

> Training-time balancing keeps the *gradient* fair; serving-time balancing keeps the *GPUs* busy. You need both, because a model trained to balance perfectly can still hit a hot expert under a skewed production traffic distribution that the training data never saw.

The lesson EPLB drives home is that load balancing is not one problem solved once at training time — it is a control loop that has to run at serving time too, against a traffic distribution you do not control. A model that is beautifully balanced on the training mix can develop a hot expert the moment production traffic skews toward, say, code or a particular language. EPLB is the feedback controller that keeps that skew from turning into a straggler.

## 4. DualPipe: feeding the pipeline from both ends

**Rule of thumb: a pipeline bubble is wasted money. If your pipeline-parallel degree is 16, a naive schedule idles ~1/16 of your cluster — and that fraction grows with depth.**

We have the comms (DeepEP) and the compute (DeepGEMM). DualPipe is the *schedule* that interleaves them so the comms hide behind the compute. It is the implementation of the V3 report's "computation-communication overlap" claim.

Classic pipeline parallelism (GPipe, 1F1B) has a structural flaw: the **bubble**. With P pipeline stages, the first stage finishes its forward pass and then waits while the data flows down the pipe and the gradients flow back, before it can start the next microbatch. The bubble is the idle time at the head and tail of the pipeline, and it scales with P. DualPipe's insight is to run **two pipelines at once, in opposite directions** — feeding microbatches from *both ends* simultaneously so that the bubble of one direction is filled by the work of the other.

![DualPipe bidirectional schedule: forward microbatches enter from the head while backward microbatches enter from the tail, and the four chunk components overlap so the bubble approaches zero as the model scales.](/imgs/blogs/deepseek-open-infra-deepep-deepgemm-flashmla-3.webp)

The timeline reads left-to-right. A **forward batch enters from the head** and does its attention. The **dispatch all-to-all** for that batch overlaps the attention of the *next* batch. **MLP compute** then hides the dispatch latency. The **combine all-to-all** overlaps a **backward batch entering from the tail** — the mirror-image microbatches that fill the gaps the forward batches leave. The result, on the right: the **bubble approaches zero as the model scales**.

### The four-component chunk

The key structural decision is that DualPipe splits each pipeline chunk into **four components**: **attention**, **all-to-all dispatch**, **MLP**, and **all-to-all combine**. This four-way split is what makes fine-grained overlap possible. The dispatch and combine are the DeepEP all-to-alls; the attention and MLP are the DeepGEMM matmuls. By scheduling them as separate units, the runtime can slot a dispatch into the shadow of an MLP, and a combine into the shadow of the next attention. If you treated the whole chunk as one opaque block, you would have nothing to overlap it with.

Think of it as a kitchen line, since the timeline figure above makes the overlap concrete: the attention station preps the next order while the dispatch station plates the current one, and nobody stands idle waiting for the order in front of them to finish. The bidirectional feed means there is always a backward-pass order coming up the line to fill any gap a forward-pass order leaves.

```python
import torch
from dualpipe import DualPipe

  # DualPipe wraps the model's forward/backward as schedulable chunks. Each
  # chunk is split into attention / dispatch / MLP / combine so the scheduler
  # can overlap the all-to-alls (DeepEP) with the matmuls (DeepGEMM).
dualpipe = DualPipe(modules)   # modules = the pipeline-stage modules

  # Microbatches are fed from BOTH ends. The scheduler interleaves the forward
  # pass entering the head with the backward pass entering the tail, so each
  # direction's bubble is filled by the other direction's compute.
loss, outputs = dualpipe.step(
    *inputs,
    num_chunks=num_chunks,        # micro-batches per direction
    criterion=criterion,
    return_outputs=False)
```

### The cost: ~2x parameter storage

DualPipe is not free. To feed the pipeline from both ends, you keep a **copy of the model parameters at each end of the pipeline** — roughly **2x parameter storage** versus a one-directional schedule. For a 671B-parameter MoE this is a real memory cost, and it is the explicit tradeoff DeepSeek made: spend memory to buy back the bubble. On a cluster where the GPUs are the expensive resource and HBM is comparatively cheap to fill, trading memory for utilization is the right call. On a memory-starved setup it might not be.

It helps to make the bubble-versus-memory tradeoff numeric. The classic 1F1B bubble fraction is approximately `(P - 1) / (M + P - 1)`, where `P` is the pipeline depth and `M` is the number of microbatches in flight. With `P = 16` and `M = 16` that is `15 / 31`, about 48% — nearly half the pipeline idle. Push `M` up to 64 and the bubble falls to `15 / 79`, about 19%, but you pay for it in activation memory (more microbatches means more saved activations). DualPipe's bidirectional feed attacks the numerator instead: by overlapping the two directions, the *effective* bubble approaches zero even at modest `M`, which is why it is so valuable on a deep pipeline. The cost it pays — duplicated parameters at the ends — is paid in parameter memory, which for an MoE is dominated by experts that are sharded across expert-parallel ranks anyway, so the duplication is of the much smaller shared (non-expert) parameters. That is the quiet reason the 2x figure is tolerable: it is 2x of the *small* part of the model, not 2x of the whole 671B.

This is also why DeepSeek **avoids tensor parallelism entirely**. TP would add yet another all-reduce that DualPipe cannot easily hide — the all-reduce sits in the critical path of a single layer's forward, not between chunks where there is slack to overlap it. By dropping TP and relying on pipeline parallelism plus expert parallelism (the DeepEP all-to-all), every communication primitive in the system is one that DualPipe *can* overlap. The schedule and the parallelism strategy were chosen together. The DualPipe repo even ships **profile-data** — the actual overlap traces — so you can see the interleaving in a profiler rather than taking the schedule on faith.

### Second-order consequence: the schedule constrains the model

A subtle point: because the four-component split assumes attention-then-dispatch-then-MLP-then-combine, the model architecture has to *be* that shape. An MoE where the experts sit inside the attention block, or where there is no clean dispatch/combine boundary, would not slot into this schedule. DualPipe is co-designed with the MoE-FFN layer structure of V3. You cannot lift it onto an arbitrary architecture and expect the overlap to materialize; the model's dataflow has to expose the four components as separable units.

## 5. FlashMLA: decode is a bandwidth problem, not a FLOPS problem

**Rule of thumb: at decode time you process one token per step, so you are reading a giant KV cache to do a tiny amount of math. That makes decode memory-bandwidth-bound, and the kernel should be built for bandwidth, not FLOPS.**

FlashMLA is the MLA (Multi-head Latent Attention) decode kernel for Hopper. The V3 report's architectural innovation on the attention side is MLA: instead of caching full key and value tensors, it caches a *compressed latent* and reconstructs K and V on the fly, which shrinks the [KV cache](/blog/machine-learning/large-language-model/kv-cache) dramatically. We covered MLA's compression scheme in the [KV cache deep-dive](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management). FlashMLA is the kernel that makes MLA decode fast on real hardware.

The headline numbers tell you exactly what kind of kernel it is: **3000 GB/s in the memory-bound regime, 580 BF16 TFLOPS in the compute-bound regime, on H800**. Two numbers for two regimes, because decode and prefill stress different resources:

- **Decode** is memory-bound. Each step reads the whole KV cache to attend over it, but only computes attention for a single new token. The bottleneck is HBM bandwidth, and FlashMLA's 3000 GB/s is most of the H800's ~3.35 TB/s peak — it is reading the KV cache about as fast as the memory system allows.
- **Prefill** (and large-batch) is compute-bound. With many query tokens, the attention matmul is large enough to saturate the tensor cores, and FlashMLA's 580 BF16 TFLOPS is the relevant number.

A worked example clarifies the bandwidth regime. Suppose a sequence has a 64 KB compressed KV latent per layer and 61 layers, so ~3.9 MB of KV state. To decode one token you must read all of it: at 3000 GB/s that read takes ~1.3 microseconds per step from the KV alone. The actual arithmetic for one token is a handful of small matmuls — nanoseconds of tensor-core work. The ratio is wildly lopsided toward memory, which is why a decode kernel that is not bandwidth-optimal leaves most of the H800 idle.

```python
import torch
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

  # Variable-length decode: each sequence in the batch is a different length.
  # FlashMLA plans a tile schedule once, then reuses it across decode steps.
cache_seqlens = torch.tensor([1023, 887, 1500, 64], dtype=torch.int32, device="cuda")
tile_metadata, num_splits = get_mla_metadata(
    cache_seqlens, num_q_heads * q_seq_len // num_kv_heads, num_kv_heads)

  # One decode step: q is the single new token per sequence; the paged KV cache
  # holds the compressed MLA latent. The kernel is memory-bound here (3000 GB/s).
out, lse = flash_mla_with_kvcache(
    q, kv_cache, block_table, cache_seqlens, head_dim_v,
    tile_scheduler_metadata=tile_metadata, num_splits=num_splits, causal=True)
```

The `get_mla_metadata` / `flash_mla_with_kvcache` split is the production-grade move: planning the tile schedule once and reusing it across many decode steps amortizes the scheduling cost, and the variable-length support (`cache_seqlens`) means a batch of sequences at different lengths is handled in one kernel without padding to the max length. Padding to max length is the naive approach that wastes bandwidth reading zeros; FlashMLA's variable-length tiling reads only the live KV.

### The roofline, and why MLA changes it

To see why FlashMLA is built the way it is, do the roofline arithmetic. A kernel's **arithmetic intensity** is flops-per-byte: the number of floating-point operations it performs divided by the number of bytes it moves from HBM. The H800's roofline ridge point — where compute-bound meets memory-bound — sits at roughly its peak-flops divided by its peak-bandwidth, on the order of 200-300 flops per byte for BF16. Below that intensity you are memory-bound; above it, compute-bound.

For standard multi-head attention decode, every new token reads the full K and V tensors. With grouped-query attention you amortize across query heads, but you still read a sizeable per-token KV. MLA's compression caches a *single low-rank latent* instead of full K and V, so the bytes-read per token drop dramatically — which, perversely, makes the kernel *more* memory-bound, not less, because you have removed bytes but the arithmetic is still tiny. The arithmetic intensity stays far left of the ridge point. That is the whole design pressure on FlashMLA: since you cannot escape the memory-bound regime, the only thing that matters is reading the compressed latent at the absolute peak the HBM can deliver, which is the 3000 GB/s number against the ~3.35 TB/s hardware ceiling — about 90% of peak.

The corollary for capacity planning: because decode is memory-bound and MLA shrinks the per-sequence KV footprint, you can pack *more concurrent sequences* into the same HBM, and each one's decode step is bounded by the same near-peak read. So MLA plus FlashMLA improves both latency (fast reads) and throughput (more sequences per GPU) at once. The prefill side is the mirror image: with hundreds of query tokens the attention matmul finally has enough arithmetic to push past the ridge point into the compute-bound regime, which is why FlashMLA reports a separate 580 BF16 TFLOPS number for that path. One kernel, two regimes, and the metadata-planning split is what lets the same code serve both without a branch in the hot loop.

### Why this matters for the cost story

FlashMLA closes the loop with the cost argument. MLA shrinks the KV cache so you can fit more concurrent sequences in HBM; FlashMLA reads that compressed cache at near-peak bandwidth so decode is fast; and the combination is what lets DeepSeek serve at the prices they serve at. If you care about decode economics generally, the same memory-bound reasoning shows up in [speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding), where the goal is to do more useful work per expensive memory read.

## 6. The numbers, side by side

**Rule of thumb: a systems claim without a measured number against a baseline is a vibe. Every repo in this release ships the number and the baseline.**

It is worth pausing to put the headline numbers in one place, because the discipline of "claim, number, baseline" is the thing most worth copying from this release. Each repo does not just assert it is fast — it tells you the number and what it beats.

![Open-infra kernel and system numbers: DeepGEMM 1350+ FP8 TFLOPS, FlashMLA 3000 GB/s, 3FS 6.6 TiB/s read, HFReduce 6.3-8.1 GB/s versus NCCL 1.6-4.8 GB/s, each against its baseline.](/imgs/blogs/deepseek-open-infra-deepep-deepgemm-flashmla-7.webp)

Read the matrix row by row. DeepGEMM posts **1350+ FP8 TFLOPS in ~300 JIT-compiled lines** against a hand-tuned cuBLAS path. FlashMLA posts **3000 GB/s and 580 BF16 TFLOPS** against a generic attention kernel. 3FS posts **6.6 TiB/s aggregate read on 180 nodes** against being local-SSD-bound. And HFReduce — the CPU-based allreduce we will meet in Section 8 — posts **6.3 to 8.1 GB/s inter-node** against NCCL's **1.6 to 4.8 GB/s** on the same PCIe-A100 fabric.

The HFReduce-versus-NCCL row is the one that should make you raise an eyebrow, so let us be precise about it. This is not a claim that HFReduce beats NCCL everywhere — it is a claim that on the *specific* Fire-Flyer fabric (PCIe A100s with a single 200 Gbps NIC per node), a CPU-driven async allreduce over PCIe outperforms NCCL's GPU-driven collective, because NCCL's design assumes more NICs and faster inter-GPU links than this cluster has. The lesson is not "always use HFReduce." The lesson is "your collective should match your fabric, and the vendor default assumes a fabric you may not have."

| System | Metric | DeepSeek number | Baseline it beats |
|---|---|---|---|
| DeepGEMM | FP8 GEMM throughput | 1350+ TFLOPS | cuBLAS/CUTLASS hand-tune |
| FlashMLA | KV-cache read (decode) | 3000 GB/s | generic attention kernel |
| FlashMLA | BF16 compute (prefill) | 580 TFLOPS | generic attention kernel |
| 3FS | Aggregate read | 6.6 TiB/s (180 nodes) | local-SSD-bound loading |
| 3FS | GraySort | 3.66 TiB/min | — |
| 3FS | KV-cache lookup | 40+ GiB/s/client | — |
| HFReduce | Inter-node allreduce | 6.3-8.1 GB/s | NCCL 1.6-4.8 GB/s |
| DeepEP | SMs to saturate fabric | ~20 SMs | full-kernel comms |

## 7. 3FS: keeping the GPUs fed

**Rule of thumb: at frontier scale, the file system is a first-class part of the training system. If data loading cannot keep up with the GPUs, you have bought expensive idle silicon.**

3FS — the **Fire-Flyer File System** — is the storage layer. It is a parallel file system designed to saturate SSD and RDMA network bandwidth, and its numbers are the kind you usually associate with a national-lab supercomputer: **6.6 TiB/s aggregate read throughput on a 180-node cluster**, **3.66 TiB/min on the GraySort benchmark**, and — the one that matters for inference — **40+ GiB/s per client for KV-cache lookup**.

Why does a model lab build its own file system? Because the alternatives fail at this scale in predictable ways. A standard NFS filer becomes a metadata bottleneck when thousands of GPU processes open files simultaneously. An object store has the throughput but not the low-latency random-read profile you need for shuffled training data. 3FS is built around RDMA and a disaggregated architecture so that read bandwidth scales linearly with the number of storage nodes, and the **40+ GiB/s/client KV-cache lookup** number is the tell that it is designed to be a *serving-time* component too, not just a training data lake — it can back a disaggregated KV cache for inference.

### The disaggregated architecture

3FS splits responsibilities across four service roles, which is what lets read bandwidth scale with node count instead of hitting a single-server ceiling. A **cluster manager** tracks membership and handles failover. **Metadata services** sit on a transactional key-value store so that the namespace operations (open, stat, list) do not bottleneck on one node the way an NFS metadata server does. **Storage services** own the actual SSDs and serve data over RDMA, using a chain-replication scheme (called CRAQ in the design) that gives strong consistency without a single serialization point. And **clients** — the FUSE mount or the native API — read directly from the storage services over RDMA, bypassing the kernel page cache for the large sequential reads that training data loading is made of.

The reason this hits 6.6 TiB/s is that a client read fans out across many storage services in parallel: a single large file is striped across SSDs on many nodes, so reading it pulls from all of them at once. Add storage nodes and you add both capacity and aggregate bandwidth, linearly. This is the architectural opposite of a scale-up filer, where adding capacity does nothing for throughput.

The practical shape of using it — you mount the cluster, then point your data loader at the FUSE path, and for the bandwidth-critical paths you use the native USRBIO (user-space ring-based IO) API that skips FUSE entirely:

```bash
  # Format and start the storage targets on each storage node, then mount the
  # FUSE client on the GPU nodes. The data loader then reads the mount like any
  # POSIX path, but bandwidth-critical reads use the native USRBIO ring API.
hf3fs_admin create-targets --node-id $NODE_ID --disk /dev/nvme0n1
mount -t hf3fs none /3fs -o cluster=fireflyer,token=$TOKEN
  # Verify aggregate read with the bundled fio-style benchmark before training:
hf3fs_bench --op read --threads 16 --bs 4M --path /3fs/dataset/shard_00000
```

**Smallpond** sits on top of 3FS as a data-processing framework, the way Spark sits on HDFS, and it leans on DuckDB for the per-partition compute. It is how DeepSeek runs the dataset transformations — dedup, filtering, tokenization — at a scale where the data itself is measured in petabytes and the processing has to keep pace with the training run consuming it. The pattern is the same one you would use with any distributed dataframe, but the storage underneath is fast enough that the compute, not the IO, is the bottleneck:

```python
import smallpond
  # Smallpond reads partitioned data straight off 3FS and runs DuckDB SQL per
  # partition, so the petabyte-scale dedup/filter keeps pace with the GPUs.
sp = smallpond.init()
df = sp.read_parquet("/3fs/raw/corpus/*.parquet")
df = sp.partial_sql("SELECT * FROM {0} WHERE length(text) > 200", df)
df.write_parquet("/3fs/clean/corpus/")
```

The architectural lesson generalizes beyond DeepSeek: the [MoE training and inference optimization](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) story is incomplete if you only optimize the GPU kernels. A 6.6 TiB/s file system exists so that the carefully-tuned DeepGEMM and DeepEP kernels never sit idle waiting for the next batch to load. Every layer of the stack is balanced against every other; the slowest layer sets the pace, and a lab that spends a year tuning kernels and then runs them off a slow filer has optimized the wrong thing.

## 8. The cluster underneath: Fire-Flyer AI-HPC

**Rule of thumb: the cheapest FLOP is the one you provisioned correctly. Most of DeepSeek's cost advantage is not a kernel — it is a network topology decision made at procurement time.**

The software stack we have walked sits on top of a hardware design that predates V3: the **Fire-Flyer 2** cluster, documented in arXiv 2408.14158 (SC'24). This is where a surprising fraction of the cost story actually lives, and it is the part most labs cannot copy after the fact because it is baked into the purchase order.

Fire-Flyer 2 is **10,000 PCIe A100 GPUs** — roughly 1,250 nodes of 8 GPUs each — with **one CX6 200 Gbps InfiniBand NIC per node**. The critical decision was the network topology.

![Fire-Flyer Fat-Tree economics: a two-layer Fat-Tree on PCIe A100s uses 122 switches versus 1,320 for an equivalent DGX-A100 cluster, landing DGX-class throughput at roughly half the cluster cost.](/imgs/blogs/deepseek-open-infra-deepep-deepgemm-flashmla-8.webp)

The before/after is the whole economic argument. A reference **DGX-A100 cluster** uses a three-layer Fat-Tree fabric requiring **1,320 QM8700 switches**, draws **4200 W per node**, and is the 100% performance baseline. Fire-Flyer 2 uses a **two-layer Fat-Tree** with a single CX6 200G NIC per node, needs only **122 QM8700 switches**, draws **2500 W per node**, and lands at **~83% of DGX performance for 60% of the node price, 50% of the cluster cost, and 40% less energy**.

The switch-count difference — 122 versus 1,320 — is the single biggest cost lever in the entire DeepSeek story. Network switches at this scale are eye-wateringly expensive, and a three-layer Fat-Tree's top layer multiplies the switch count. By accepting a two-layer topology (which means slightly lower bisection bandwidth and is *exactly why* node-limited routing matters so much), Fire-Flyer cut the switch bill by an order of magnitude. The ~17% performance hit versus DGX is more than paid for by halving the cluster cost.

### HFReduce: the software that the cheap topology demands

A two-layer Fat-Tree with one NIC per node is a fabric NCCL was not designed for. NCCL assumes a richer inter-GPU fabric. So Fire-Flyer ships **HFReduce**, a **CPU-based asynchronous allreduce over PCIe** that hits **6.3 to 8.1 GB/s inter-node** versus NCCL's **1.6 to 4.8 GB/s** on the same fabric. The mechanism: instead of having the GPUs drive the collective over a fabric that bottlenecks them, HFReduce copies gradients to host memory, does the reduction on the CPU, and overlaps it asynchronously with the next forward pass. On a fabric with one NIC per node, the CPU-driven path wins because it does not contend for the scarce GPU-to-NIC bandwidth the way NCCL's GPU-driven collective does.

**HaiScale** is the training framework on top, and its parallel-efficiency number is the proof the whole thing works: **91% parallel efficiency scaling LLaMA-13B from 64 to 512 GPUs**. That number is what justifies the cheap topology — if the topology had crippled scaling, the cost savings would be illusory. 91% efficiency across an 8x GPU scale-out on a half-price cluster is the real headline of the Fire-Flyer paper.

## Case studies from production

The following are composite scenarios — drawn from the patterns these repos are built to handle — of how each design decision shows up as a real incident when you run this stack or one like it.

### 1. The tensor-parallel all-reduce that ate the cluster

A team ports a dense-model training setup to a large MoE on an H800-class cluster, keeping their tensor-parallel degree of 8 because "it worked before." Throughput is a third of what the FLOP count predicts. The wrong first hypothesis is that the MoE routing is unbalanced and some experts are starving. The actual root cause: the TP all-reduce after every layer is saturating the 50 GB/s InfiniBand link, and the GPUs spend two-thirds of every step blocked on the wire. The fix is the DeepSeek fix — drop tensor parallelism entirely, shard the experts, and move tokens with a DeepEP-style node-limited all-to-all that DualPipe can overlap. The lesson: TP is a tax on the slow link that grows with model depth, and on a throttled interconnect that tax is unaffordable.

### 2. The FP8 combine that quietly degraded accuracy

A team adopts FP8 dispatch for their MoE all-to-all and sees a beautiful throughput win. Three days into a run the eval metrics start drifting down, slowly, with no loss spike and no crash. The wrong hypothesis is a data-quality problem in the later shards. The actual root cause: they set FP8 precision on the *combine* direction as well as dispatch, and the weighted accumulation of expert outputs in FP8 compounds rounding error across the top-k sum. The fix: FP8 dispatch (the tokens were going to be quantized for the FP8 GEMM anyway), but BF16 combine. The lesson: quantization is safe where the next operation expects low precision and dangerous where you are accumulating — match the precision to what the downstream op needs.

### 3. The pipeline bubble nobody could find in the profiler

A 16-stage pipeline-parallel run shows 30% lower utilization than expected, but the per-kernel profiler shows every kernel running at full speed. The wrong hypothesis is a slow kernel hiding somewhere. The actual root cause: the pipeline bubble — the head and tail stages sitting idle while data flows through the pipe — which does not show up as a slow kernel because it shows up as *no kernel*, a gap between kernels. The fix is a DualPipe-style bidirectional schedule that feeds microbatches from both ends so the idle stages of one direction are filled by the other. The lesson: utilization gaps live *between* kernels, not inside them; a per-kernel profiler is blind to the bubble, and you need a timeline view to see it. This is exactly why DualPipe ships its profile-data traces.

### 4. The decode kernel that was compute-tuned for a memory-bound problem

A team optimizes their MLA decode by increasing tensor-core occupancy, and decode gets *slower*. The wrong hypothesis is that the new kernel has a launch-overhead regression. The actual root cause: decode is memory-bandwidth-bound — it reads the whole KV cache to decode one token — and "increasing tensor-core occupancy" added arithmetic the kernel did not need while doing nothing about the HBM reads that are the real bottleneck. The fix is a FlashMLA-style kernel built for bandwidth: read the compressed KV latent at near-peak HBM rate, with variable-length tiling so you never read padding. The lesson: profile to find the bound *first*; optimizing the wrong resource is worse than not optimizing.

### 5. The grouped GEMM that broke CUDA graphs

A team uses the contiguous grouped-GEMM path for both training and decode, and decode latency is erratic with periodic stalls. The wrong hypothesis is a memory-allocator hiccup. The actual root cause: the contiguous layout requires knowing the per-expert token counts at launch time, which during decode are only known after routing the current step — so the kernel cannot be captured in a CUDA graph, and every step re-incurs launch overhead and synchronization. The fix is the masked grouped-GEMM path (DeepGEMM's `_masked` variant), where a fixed-shape launch with a runtime mask is CUDA-graph-friendly. The lesson: the layout that is optimal for training (contiguous, sorted by expert) is wrong for decode (dynamic counts); ship two kernels, not one.

### 6. The NCCL collective that underperformed on the cheap fabric

A team builds a budget A100 cluster with one NIC per node to save money, runs NCCL allreduce, and gets a third of the inter-node bandwidth the NICs should deliver. The wrong hypothesis is a misconfigured NCCL topology file. The actual root cause: NCCL's GPU-driven collective assumes more GPU-to-NIC bandwidth than a single-NIC-per-node fabric provides, so the GPUs contend for the one scarce NIC. The fix is an HFReduce-style CPU-driven allreduce that copies to host memory and reduces on the CPU, overlapping asynchronously — 6.3 to 8.1 GB/s versus NCCL's 1.6 to 4.8. The lesson: the vendor collective is tuned for the vendor reference fabric; when you buy a cheaper fabric you have to bring your own collective.

### 7. The file system that throttled a perfectly-tuned kernel stack

A team spends a quarter tuning their GEMM and attention kernels to near-peak, then sees no end-to-end speedup. The wrong hypothesis is that the kernels are not actually as fast as the microbenchmarks claim. The actual root cause: the training data loader, backed by a shared NFS filer, cannot deliver shuffled batches fast enough, so the GPUs idle between steps waiting on I/O — the kernels are fast, but they are starving. The fix is a high-throughput parallel file system in the 3FS mold (RDMA-backed, bandwidth scaling with node count) plus a data framework like Smallpond to keep the pipeline full. The lesson: the slowest layer of the stack sets the pace, and a 6.6 TiB/s file system exists precisely so the kernel work above it never waits.

### 8. The node-limited routing constraint that looked like a bug

A team reviewing the V3 routing sees that each token is capped at four nodes and assumes it is a leftover debugging constraint that hurts model quality by limiting expert choice. They remove the cap. Inter-node traffic explodes, all-to-all time triples, and the run slows to a crawl. The wrong hypothesis is that removing the cap should have *helped* by giving tokens more expert options. The actual root cause: the cap is load-bearing — it bounds the number of slow-link hops per token, and without it tokens scatter across many nodes and saturate the InfiniBand fabric. The fix is to restore the M=4 cap and let intra-node NVLink handle the fan-out to multiple experts on the same node. The lesson: a routing constraint that looks like a model-quality limitation is often a comms-cost limitation in disguise — read it against the bandwidth hierarchy before you remove it.

### 9. The hot expert that production traffic created

A model trained with loss-free balancing serves beautifully on the eval set, then a customer routes a flood of code-completion traffic through it and p99 latency triples while average utilization stays flat. The wrong hypothesis is a memory leak or a noisy-neighbor problem on one node. The actual root cause: code traffic skews token routing toward a small set of experts that the training mix never stressed, so those experts' GPUs become stragglers that the all-to-all barrier waits on every step — average utilization looks fine because the *other* GPUs are idle waiting. The fix is EPLB-style redundant-expert replication: detect the hot experts from live load counters and place replicas (hierarchically, within the node, to preserve node-limited routing) so the hot expert's tokens split across copies. The lesson: training-time balance does not survive contact with a skewed production distribution; you need a serving-time load controller watching real traffic.

### 10. The JIT stall that looked like a deadlock

A team adopts DeepGEMM, and the very first training step hangs for several seconds before proceeding normally. The wrong hypothesis is a distributed deadlock — a missing collective somewhere. The actual root cause: DeepGEMM JIT-compiles a kernel specialized to each new GEMM shape the first time it sees it, and the first step touches dozens of new shapes, each triggering a sub-second compile that serializes on the first rank. After the kernel cache warms, every subsequent step is fast. The fix is to warm the JIT cache with a representative dummy step before the timed run (and to persist the compiled-kernel cache across job restarts so a requeue does not recompile). The lesson: JIT trades a one-time compile for a specialized kernel; budget for the warm-up and never measure the first step.

## When to reach for this stack — and when not to

**Reach for the DeepSeek open-infra playbook when:**

- You are training or serving a **large MoE** where the cross-node all-to-all dominates your step time. DeepEP plus node-limited routing is the single biggest lever here.
- You are on **Hopper GPUs running FP8** and want a matmul you can actually read, tune, and trust. DeepGEMM's 300-line JIT kernel is built for exactly this.
- Your interconnect is the bottleneck — **throttled NVLink, single-NIC nodes, or a deliberately cheap fabric**. The entire stack is co-designed around a slow link, so it pays off most when your link is slow.
- You run **pipeline parallelism deep enough that the bubble hurts** (PP degree of 8 or more) and you have the HBM headroom to spend ~2x parameter storage on a DualPipe-style bidirectional schedule.
- You are serving **MLA-style compressed-KV models** at decode and need a bandwidth-optimal kernel. FlashMLA's 3000 GB/s is the target.
- Your **data loading or KV-cache serving is I/O-bound** at a scale where a standard filer chokes — a 3FS-class parallel file system removes that ceiling.

**Skip it — or reach for something else — when:**

- You are running **dense models small enough to fit with simple data parallelism**. The whole stack is overhead you do not need; plain DDP and the vendor GEMM will serve you better.
- You are **not on Hopper, or not using FP8**. DeepGEMM and FlashMLA are Hopper-and-FP8-specific; on Ampere/Ada or in BF16-only setups they do not apply, and the vendor library is the right call.
- You have a **fast, uniform fabric** (un-throttled NVLink, multiple NICs per node, full-fat-tree). Then NCCL collectives and tensor parallelism are fine, and the co-design discipline buys you less because your slow link is not actually slow.
- You are **memory-starved**. DualPipe's ~2x parameter storage and 3FS's infrastructure footprint assume you can spend memory and nodes to buy utilization; if HBM is your binding constraint, a single-direction schedule may be the better tradeoff.
- You need **portability across GPU generations and dtypes**. This stack trades portability for peak performance on one specific platform; a portability requirement is a reason to use the more general libraries.

The meta-lesson is the one worth carrying out of this post: DeepSeek's advantage was not a single brilliant kernel. It was the *discipline of co-design* — every layer from the procurement-time network topology up through the routing algorithm was chosen against the same constraint, and then the code that implements each choice was published so you can verify the claims yourself. Treat the open-infra repos as the V3 report's footnotes, and read the paper and the code together. The paper tells you what they did; the code tells you it was true.

## Further reading

- [How DeepSeek-V3 Trained a 671B-Parameter Model for $5.6M](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — the model-side companion to this post: FP8 training, Multi-Token Prediction, and loss-free balancing.
- [Optimizing MoE Training and Inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) — the broader MoE systems context that DeepEP and EPLB fit into.
- [KV Cache Optimization and Management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) — why MLA's compressed KV cache matters and how FlashMLA reads it.
- [Speculative Decoding](/blog/machine-learning/large-language-model/speculative-decoding) — the other half of the decode-economics story, where memory-bound reasoning recurs.
- DeepSeek-V3 Technical Report (arXiv 2412.19437) and Fire-Flyer AI-HPC (arXiv 2408.14158) — the two papers these repos footnote.
- The `deepseek-ai/open-infra-index` repository — the canonical index of every Open Infra Week release: DeepEP, DeepGEMM, FlashMLA, DualPipe, 3FS, EPLB, and Smallpond.

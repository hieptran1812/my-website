---
title: "Serving Mixture-of-Experts Models at Scale: The Memory-Bound Giant"
date: "2026-07-03"
publishDate: "2026-07-03"
description: "Why a 671B MoE model that activates only 37B parameters still needs 16 GPUs, and how expert parallelism, all-to-all kernels, and load balancing make it serve fast in production."
tags:
  [
    "model-serving",
    "inference",
    "mixture-of-experts",
    "expert-parallelism",
    "deepseek",
    "mixtral",
    "vllm",
    "sglang",
    "ml-infrastructure",
    "llm",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/serving-moe-models-at-scale-1.webp"
---

You pull `deepseek-ai/DeepSeek-V3` onto a fresh 8×H100 node — 640 GB of HBM, the kind of box that serves a 70B dense model with room to spare — and vLLM dies before it emits a single token. The log says `CUDA out of memory`. You double-check the model card: 671B total parameters, but *only 37B active per token*. Thirty-seven billion. That should fit on two GPUs. Why did an eight-GPU node choke?

Because "active" is a statement about **compute**, not **memory**. Every one of those 671B parameters has to live somewhere the router can reach it in a few microseconds, and in FP8 that is 671 GB of weights. Your node has 640. The model that "only touches 37B params" won't even load, let alone serve. This is the single fact that reorganizes everything you know about serving from dense models: a Mixture-of-Experts (MoE) model is a **memory-capacity-bound giant** wearing the FLOP profile of a much smaller network. The GPU holds all the experts; the compute touches almost none of them.

Get this asymmetry right and MoE is the best deal in serving — DeepSeek-V3 answers with the quality of a frontier dense model while burning the FLOPs of a 37B one. Get it wrong and you pay for a fleet of GPUs whose arithmetic units sit 60% idle behind a hot expert, or you OOM on load, or your all-to-all network becomes the bottleneck nobody profiled for. By the end of this post you will be able to size an MoE deployment (how many GPUs, which parallelism, how much interconnect), launch it under vLLM or SGLang with expert parallelism, monitor per-expert load, and reason about when MoE is *not* worth it.

![Diagram of the MoE forward pass where a router selects top-8 of 256 experts per token and a weighted combine merges their outputs](/imgs/blogs/serving-moe-models-at-scale-1.webp)

The figure above is the whole model in one layer. A token's hidden vector hits a **router** (a small learned linear layer plus a top-k selection), which picks a handful of experts — 8 of 256 in DeepSeek-V3 — and routes the token only to those. Each selected expert is a full feed-forward network (FFN). Their outputs are combined with the router's gating weights, a shared expert (always active) is added, and the layer moves on. Everything difficult about serving MoE flows from that branch-and-merge: different tokens pick different experts, the experts live on different GPUs, and the router's choices are not uniform. Let us take these one at a time, and tie every technique back to the recurring tension of this series — the SLO triangle of **latency ↔ throughput ↔ cost**. If you are new to that framing, start with [what model serving actually is](/blog/machine-learning/model-serving/what-is-model-serving); if you have never felt the KV-cache memory wall, [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) is the prerequisite for the decode-path math below.

## The asymmetry that defines MoE serving

Start from the dense baseline. A dense 70B transformer serving in FP8 holds 70 GB of weights, and *every* token flows through *every* one of those parameters. Its arithmetic per token is roughly ${2 \times 70\text{B} = 140}$ GFLOP (two FLOPs — a multiply and an add — per parameter per token). Memory footprint and compute scale together: bigger model, more of both. Serving it is compute-bound on the GPU's tensor cores at large batch, memory-bandwidth-bound on weight reads at small batch, and you provision GPUs mostly for the FLOPs.

MoE breaks that coupling. It grows total parameters — and therefore *knowledge capacity* — without growing per-token compute, by only running a token through a few experts. The consequence for a serving engineer is stark, and it is worth stating as a table before we prove it.

![Comparison matrix of dense and MoE models across total parameters, active parameters per token, HBM footprint, active FLOPs, and interconnect needs](/imgs/blogs/serving-moe-models-at-scale-2.webp)

Read the DeepSeek-V3 row against the dense 70B row. DeepSeek-V3 activates only 37B parameters per token — *half* the active compute of the 70B dense model, at ${2 \times 37\text{B} = 74}$ GFLOP — yet it must hold 671 GB of weights in HBM, nearly ten times as much. Qwen3-235B-A22B tells the same story: 44 GFLOP of arithmetic per token sitting on top of 235 GB of resident weights. The active FLOPs say "small model"; the HBM footprint says "enormous model." You provision GPUs for the *bigger* of the two constraints, and for MoE that is almost always memory capacity.

### Why memory capacity, not bandwidth or FLOPs, is the binding constraint

There are three resources a GPU brings to inference: arithmetic (FLOP/s), memory bandwidth (HBM GB/s), and memory capacity (HBM GB). For dense LLM decode, the binding constraint is usually **bandwidth** — you re-read all the weights for every token and the arithmetic intensity is low. For MoE decode, the arithmetic per token is even lower (fewer active params), so bandwidth per token drops too. But the *capacity* requirement does not drop at all — you still store every expert. So the ratio of "memory you must buy" to "compute you actually use" blows up.

Quantify it. Define the **capacity-to-compute ratio** as total parameters divided by active parameters:

$$R = \frac{P_{\text{total}}}{P_{\text{active}}}$$

For a dense model ${R = 1}$. For Mixtral 8x7B, ${R = 46.7 / 12.9 \approx 3.6}$. For DeepSeek-V3, ${R = 671 / 37 \approx 18}$. That number is the multiplier on how much HBM you buy relative to the compute you burn. A dense 37B-active workload would fit on one H100; DeepSeek-V3, which does the same arithmetic per token, needs a *rack* because ${R \approx 18}$. MoE trades cheap FLOPs for expensive HBM and expensive interconnect. Whether that trade wins depends entirely on whether you can keep those many GPUs busy — which is the rest of this post.

#### Worked example: sizing DeepSeek-V3's HBM

DeepSeek-V3 is 671B parameters, trained and shipped in FP8 (1 byte per weight), so the weights alone are:

$$671 \times 10^{9} \text{ params} \times 1 \text{ byte} = 671 \text{ GB}$$

Now add what serving needs on top of weights:

- **KV cache.** DeepSeek-V3 uses Multi-head Latent Attention (MLA), which compresses the per-token KV state to a latent of 512 plus a 64-dim rotary component, times 61 layers, in BF16: roughly ${(512 + 64) \times 61 \times 2 \approx 70}$ KB per token. At 8,000 concurrent tokens of context that is only about 0.5 GB — MLA is why DeepSeek's KV cache is almost a rounding error. (Contrast a naive MHA model at this scale, which would spend many GB per thousand tokens; see [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) for the general story.)
- **Activation and communication buffers.** All-to-all staging buffers, the router logits, and intermediate FFN activations add tens of GB across the fleet.

So the dominant term is the 671 GB of weights. On H100 80GB cards, framework overhead leaves roughly 76 GB usable per GPU, so:

$$\left\lceil \frac{671}{76} \right\rceil = 9 \text{ GPUs, minimum, just to load weights}$$

Nine GPUs is the floor for the model to *exist* in memory. Add KV cache, activations, and enough spare capacity to run a useful batch and you land at 16 GPUs (two nodes) as a realistic minimum, and much more for high throughput. On the 640 GB single node from the intro — 8×H100 — the arithmetic is unforgiving: ${8 \times 80 = 640 < 671}$. It cannot hold the weights. That is not a tuning problem; it is a capacity wall. Your options are more GPUs, a higher-capacity card (H200 at 141 GB, so ${8 \times 141 = 1128}$ GB fits comfortably), or heavier quantization.

### The MoE family on one sheet

The ${R}$ ratio is easier to trust once you see the whole family laid out. Every row below is FP8 weights (1 byte per parameter) on H100-class hardware with roughly 76 GB usable per card; "min H100" is the ceiling of the weight footprint over usable HBM, before any KV cache or activation headroom.

| Model | Total params | Active / token | ${R}$ | FP8 weights | Active GFLOP | Min H100 (weights only) |
|---|---|---|---|---|---|---|
| Dense 70B (baseline) | 70B | 70B | 1.0 | 70 GB | 140 | 1 |
| Mixtral 8x7B | 46.7B | 12.9B | 3.6 | 47 GB | 26 | 1 |
| Mixtral 8x22B | 141B | 39B | 3.6 | 141 GB | 78 | 2 |
| Qwen3-30B-A3B | 30B | 3B | 10.0 | 30 GB | 6 | 1 |
| Qwen3-235B-A22B | 235B | 22B | 10.7 | 235 GB | 44 | 4 |
| DeepSeek-V3 | 671B | 37B | 18.1 | 671 GB | 74 | 9 |

Read the "Active GFLOP" column against the "FP8 weights" column and the whole thesis of MoE serving is one glance away: DeepSeek-V3 does *less* arithmetic per token than a dense 70B (74 versus 140 GFLOP) while carrying almost ten times the weights. Qwen3-30B-A3B is the extreme — 6 GFLOP of compute, the arithmetic of a 3B model, riding on 30 GB of resident weights. Active FLOPs decide how fast a single token computes; the weight footprint decides how many GPUs you buy. They point in opposite directions, and the size of the gap between them is exactly ${R}$.

#### Worked example: the roofline says memory-bound

Why does MoE decode stay memory-bound even at large batch, when a dense model of the same active size would go compute-bound? Roofline arithmetic. An H100 SXM delivers about 990 TFLOP/s of dense FP8 and about 3.3 TB/s of HBM bandwidth, so its **ridge point** — the arithmetic intensity above which a kernel becomes compute-bound — is:

$$\frac{990 \times 10^{12} \text{ FLOP/s}}{3.3 \times 10^{12} \text{ B/s}} \approx 300 \text{ FLOP/byte}$$

For a per-expert FFN GEMM in FP8, each weight byte is read once and drives two FLOPs per token that landed on that expert, so the arithmetic intensity is roughly ${2 \times (\text{tokens on that expert})}$ FLOP/byte. To clear the ridge you need about 150 tokens on a single expert. But top-8 routing over 256 experts sends the *average* expert only ${B \times 8 / 256 = B/32}$ tokens, so you would need a global decode batch of ${B \approx 150 \times 32 = 4800}$ tokens per step just to push the average expert over the line — and the cold experts never get there. The sparsity that saves FLOPs also starves each expert's GEMM of the batch it would need to be compute-bound. That is the roofline reason MoE decode is a memory system problem wearing a compute costume.

#### Worked example: the quantization ladder for DeepSeek-V3

Precision is the other lever on the capacity wall, and DeepSeek-V3 is the clean case because it ships in FP8 natively. The same 671B parameters cost very different amounts of HBM depending on how many bytes you spend per weight:

| Precision | Bytes / param | Weight footprint | Min H100 (80 GB) | Min H200 (141 GB) |
|---|---|---|---|---|
| BF16 | 2 | 1342 GB | 18 | 10 |
| FP8 (native) | 1 | 671 GB | 9 | 5 |
| INT4 (AWQ/GPTQ) | 0.5 | 336 GB | 5 | 3 |

BF16 would demand eighteen H100s for weights alone — which is why nobody serves DeepSeek-V3 in BF16 when the FP8 checkpoint is native. INT4 halves the footprint again to five cards, at an accuracy cost that has to be measured per task (see [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving)). The higher-capacity H200 (141 GB) collapses these counts: the FP8 model fits in five H200s where it needed nine H100s, because the binding constraint is capacity and the H200 brings 76% more of it per card. This is the rare case where buying a bigger card, not more cards, is the right move — capacity walls reward density.

#### Worked example: what the spare HBM buys you

Once the 671 GB of weights are placed, everything left over is your batch budget. On 16 H100s (2 nodes), weights take about 42 GB per GPU, leaving roughly 34 GB per GPU for KV cache and activations — call it ~540 GB of runtime state across the fleet. DeepSeek-V3's MLA KV cache is about 70 KB per token, so 540 GB / 70 KB is on the order of 7.7 million token-slots of KV, which at 8k context is roughly 950 concurrent sequences before eviction. Double the fleet to 32 H100s and the weight share per GPU halves — the 671 GB is a *fixed* cost spread over more cards — so free HBM per GPU climbs to about 55 GB and total runtime budget more than triples. That is the counterintuitive economics of MoE capacity: adding GPUs past the load floor buys *disproportionately* more batch, because the fixed weight cost amortizes. The first nine GPUs buy existence; nearly every GPU after that buys concurrency, which is exactly why large-scale EP deployments post such high per-node throughput.

## The router: top-k gating and why batching goes irregular

The router is a deceptively small piece of machinery with outsized consequences for serving. For each token, it computes a score for every expert, keeps the top-k, normalizes those into gating weights, and dispatches the token to exactly those k experts. Here is the mechanism in PyTorch, stripped to essentials:

```python
import torch
import torch.nn.functional as F

class MoERouter(torch.nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor):
        # x: [num_tokens, d_model]
        logits = self.gate(x)                       # [num_tokens, n_experts]
        # top-k expert selection per token
        topk_val, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        gates = F.softmax(topk_val, dim=-1)         # gating weights, sum to 1
        # per-expert token counts drive load balance and all-to-all sizing
        counts = torch.bincount(topk_idx.flatten(), minlength=self.n_experts)
        return topk_idx, gates, counts              # who, how much, how many
```

The routing decision (`topk_idx`) is *data-dependent*: two tokens in the same request, let alone the same batch, generally go to different experts. That single fact is what makes MoE batching irregular. In a dense model, a batch of 256 tokens is one clean matrix multiply against one weight matrix — the GPU loves it. In an MoE layer, that same batch of 256 tokens fragments into 256 routing decisions across (say) 256 experts, so each expert receives a *ragged* handful of tokens: maybe 12 for expert 7, zero for expert 41, 31 for expert 130. You can no longer do one big GEMM; you do many small, uneven ones — a **grouped GEMM** or a sequence of per-expert matmuls — and the token→expert scatter has to happen physically, over the interconnect, because the experts do not all live on the same GPU.

Three second-order effects follow, and each one is a serving problem:

1. **Small, uneven GEMMs are less efficient** than one big one. An expert that receives 12 tokens runs a skinny matmul that under-utilizes the tensor cores; the GPU spends more time launching kernels and moving data than computing. Grouped-GEMM kernels (in DeepEP, CUTLASS, and Triton) exist specifically to batch these ragged matmuls efficiently.
2. **Routing is not uniform.** The router is trained, and trained routers develop favorites. Some experts are hot; others are nearly dead. This is the load-imbalance problem we will spend a whole section on.
3. **The counts change every step.** The `counts` vector above is different for every batch and every decode step, so the all-to-all communication is variable-sized. Engines pre-allocate to a worst case or pay a synchronization cost to discover the sizes — either way it complicates the scheduler.

### Grouped GEMM: making the ragged matmuls efficient

The first of those effects deserves its own paragraph, because the fix is a specific kernel you should know by name. After dispatch, each GPU holds a variable number of tokens for each of its local experts — 12 for one, 300 for another, 0 for a third. The naive implementation loops over experts and launches one matmul per expert, which is death by kernel-launch overhead: dozens of tiny GEMMs, each too small to fill the tensor cores, each paying fixed launch latency. A **grouped GEMM** (also called a batched or segmented GEMM) fuses all of an expert group's matmuls into a *single* kernel launch that internally walks a list of ${(\text{offset}, \text{row count})}$ segments — one contiguous weight matrix per expert, one ragged block of tokens per expert, all processed in one launch with the tensor cores kept fed.

DeepSeek open-sourced **DeepGEMM** for exactly this: FP8 grouped GEMMs with fine-grained scaling, tuned for the token-count distributions MoE produces. vLLM and SGLang reach the same goal through CUTLASS grouped-GEMM kernels and Triton implementations. The operational point is that the grouped GEMM is *the* expert-compute kernel — its efficiency is what decides whether the "37B active" promise turns into 37B-active *speed* or into a swarm of underfilled matmuls. When someone reports that MoE inference is "slower than the active-parameter count suggests," the grouped GEMM (or its absence) is the first place to look, right after the all-to-all.

The `counts` vector is the most important line in that snippet for an operator. It is the raw material for load-balance monitoring, all-to-all buffer sizing, and the decision of whether to replicate a hot expert. We will emit it to Prometheus later.

### Auxiliary-loss-free balancing: keeping the router honest without hurting quality

There is a chicken-and-egg problem baked into the router. Left alone, it develops favorites and the load skews; the classic fix is an **auxiliary load-balancing loss** that penalizes imbalance during training, but that auxiliary gradient fights the language-modeling objective and measurably costs quality. DeepSeek-V3's contribution here — worth understanding because it changes what you monitor at serving time — is **auxiliary-loss-free load balancing**.

The mechanism is a per-expert **bias** added to the routing score for the *selection* decision only. Let ${s_i}$ be the affinity of a token for expert ${i}$ (the router logit). Instead of selecting the top-k by ${s_i}$ directly, the router selects by ${s_i + b_i}$, where ${b_i}$ is a bias maintained per expert. Crucially, the bias steers *which* experts are chosen but does **not** enter the gating weight that scales an expert's output — so it rebalances load without distorting the model's arithmetic. After each step, the bias is nudged: raise ${b_i}$ for underloaded experts, lower it for overloaded ones. The update, stripped to essentials:

```python
class AuxFreeBalancer:
    def __init__(self, n_experts: int, target_load: float, gamma: float = 1e-3):
        self.bias = torch.zeros(n_experts)   # b_i, added to scores for SELECTION only
        self.target = target_load            # desired tokens/expert per step
        self.gamma = gamma                   # bias update speed

    def route(self, scores: torch.Tensor, top_k: int):
        # scores: [num_tokens, n_experts] raw affinities s_i
        adjusted = scores + self.bias        # selection uses s_i + b_i
        topk_idx = torch.topk(adjusted, top_k, dim=-1).indices
        # gating weight uses the ORIGINAL scores, not the biased ones
        gate = torch.softmax(scores.gather(-1, topk_idx), dim=-1)
        return topk_idx, gate

    def update(self, counts: torch.Tensor):
        # push underloaded experts up, overloaded experts down
        err = self.target - counts.float()   # >0 means underloaded
        self.bias += self.gamma * torch.sign(err)
```

Two things follow for a serving engineer. First, the bias is a *trained* artifact shipped in the checkpoint — at inference you inherit a router already close to balanced on the training distribution, which is why a fresh DeepSeek-V3 deployment starts with a lower imbalance factor than a naively-trained MoE would. Second, the catch: your production traffic is **not** the training distribution. A workload skewed toward code, or toward one language, re-skews the routing no matter how balanced the pretrained bias was. That residual, distribution-shift-driven imbalance is precisely what EPLB and the monitor below exist to catch — the router's built-in balancing gets you close, and placement-level balancing closes the gap that your specific traffic opens.

## Expert parallelism: spreading experts across GPUs

If a single GPU cannot hold all the experts — and for DeepSeek-V3 it obviously cannot — you shard the experts across GPUs. **Expert parallelism (EP)** assigns each GPU a disjoint subset of experts and keeps the rest of the model (attention, embeddings) either replicated or sharded by another scheme. It is a distinct axis from tensor parallelism (which splits each weight matrix across GPUs) and pipeline parallelism (which splits layers across GPUs); for the training-side treatment of these axes see [expert parallelism for MoE](/blog/machine-learning/distributed-training/expert-parallelism-moe), and for the serving-side composition of TP and PP see [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving).

![Grid of eight GPUs each holding 32 experts under expert parallelism, with one GPU receiving far more tokens and becoming a straggler](/imgs/blogs/serving-moe-models-at-scale-3.webp)

The grid shows the layout for DeepSeek-V3's 256 routed experts across 8 GPUs: 32 experts per GPU, plus the shared expert replicated or parked on GPU7. Each GPU owns its experts' weights and only those. When a batch arrives, every token has already been routed by the router, so the system knows which GPU each token must visit. Now the physical dance begins, and it has three phases per layer:

- **Dispatch (all-to-all).** Each GPU sends every token to the GPU(s) that own the token's selected experts. Because routing is arbitrary, a token on GPU0 may need experts on GPU2, GPU5, and GPU6. This is a collective **all-to-all**: every GPU sends a (possibly different) chunk of data to every other GPU simultaneously.
- **Expert compute.** Each GPU runs its local experts over the tokens it just received — the ragged grouped GEMM.
- **Combine (all-to-all).** Each GPU sends the expert outputs back to the GPU that owns each token, where they are merged with the gating weights.

Two all-to-all collectives per MoE layer, every layer, every forward pass. That is the defining communication pattern of MoE serving, and it is why interconnect — NVLink within a node, InfiniBand or RoCE across nodes — is a first-class capacity dimension for MoE the way it never was for a single-GPU dense model.

### The all-to-all communication volume

All-to-all is the expensive collective. Unlike an all-reduce (whose volume is fixed by the tensor size regardless of world size), all-to-all moves *distinct* data between every pair of ranks, and the aggregate bytes on the wire grow with both the token count and the number of experts touched. Let us compute the per-token volume, because that number decides whether your fabric can keep up.

For dispatch, each token sends its hidden vector (dimension ${d}$) to each of its ${k}$ selected experts. For combine, each expert returns a ${d}$-dimensional result per token. So per token, per MoE layer:

$$V_{\text{token,layer}} = \underbrace{k \cdot d \cdot b_{\text{dispatch}}}_{\text{dispatch}} + \underbrace{k \cdot d \cdot b_{\text{combine}}}_{\text{combine}}$$

where ${b}$ is bytes per element for each direction.

#### Worked example: DeepSeek-V3 all-to-all volume

DeepSeek-V3 has ${d = 7168}$ and ${k = 8}$. Dispatch is done in FP8 (1 byte) to save bandwidth; combine is typically BF16 (2 bytes) because the summed result is precision-sensitive:

$$\text{dispatch} = 8 \times 7168 \times 1 = 57{,}344 \text{ bytes} \approx 56 \text{ KB}$$
$$\text{combine} = 8 \times 7168 \times 2 = 114{,}688 \text{ bytes} \approx 112 \text{ KB}$$
$$V_{\text{token,layer}} \approx 168 \text{ KB}$$

Generating one token means running all 58 MoE layers, so a single token's all-to-all footprint for one decode step is:

$$168 \text{ KB} \times 58 \approx 9.5 \text{ MB per generated token}$$

Scale that to a real decode batch. If a GPU is participating in a decode step over a batch of 256 sequences, the fabric moves on the order of ${256 \times 168\text{ KB} \approx 43}$ MB *per layer*. Across an internode RDMA link at an effective 50 GB/s, that is roughly 0.86 ms per layer if it were fully serialized and fully internode — which, multiplied by 58 layers, would be a catastrophic ~50 ms per token. That figure is exactly why three optimizations are non-optional at scale:

1. **Node-limited routing.** DeepSeek-V3 caps each token to at most 4 nodes, keeping most dispatch traffic on intra-node NVLink (~150+ GB/s) instead of the slower internode fabric.
2. **FP8 dispatch.** Halving the dispatch bytes (versus BF16) directly halves the dominant traffic.
3. **Compute-communication overlap.** The all-to-all of one micro-batch is overlapped with the expert GEMM of another, so the wire time is hidden behind arithmetic instead of added to it.

The takeaway: MoE decode is not merely weight-bandwidth-bound like dense decode; it is *also* all-to-all-bandwidth-bound. Provision interconnect accordingly, and never benchmark MoE throughput on a single node and extrapolate — the moment you cross a node boundary the communication regime changes.

#### Worked example: why Mixtral's all-to-all barely registers

Run the same volume math for Mixtral 8x7B and you see why small MoEs feel nothing. Mixtral has ${d = 4096}$ and ${k = 2}$. Dispatch in FP8, combine in BF16, per token per MoE layer:

$$\text{dispatch} = 2 \times 4096 \times 1 = 8192 \text{ bytes} \approx 8 \text{ KB}$$
$$\text{combine} = 2 \times 4096 \times 2 = 16384 \text{ bytes} \approx 16 \text{ KB}$$
$$V_{\text{token,layer}} \approx 24 \text{ KB}$$

That is one-seventh of DeepSeek-V3's 168 KB per layer, and Mixtral has 32 layers to DeepSeek's 58. More important, at EP2 or EP4 the entire all-to-all stays inside one node, riding NVLink at 150+ GB/s rather than crossing a 50 GB/s RDMA hop. A batch of 256 tokens moves ${256 \times 24\text{ KB} = 6}$ MB per layer over NVLink in roughly 40 µs — comfortably hidden behind the expert GEMM. This is the arithmetic behind a claim from the case studies: the hard all-to-all problems do not bite until you both grow the expert count *and* cross a node boundary. Mixtral does neither.

#### All-to-all versus all-reduce: why the collective matters

It is worth being precise about why all-to-all is the collective that scares MoE operators when dense tensor-parallel serving lives happily on all-reduce. A ring all-reduce moves a *fixed* volume — about ${2(N-1)/N}$ times the tensor size per rank — regardless of how the data is distributed, because every rank ends with the same reduced result. An all-to-all moves **distinct** data between every ordered pair of ranks: rank ${i}$ sends a private chunk to rank ${j}$ for every ${j}$, and the chunk sizes are *data-dependent*, set by how many of rank ${i}$'s tokens routed to experts on rank ${j}$. So all-to-all is worse on three axes at once: the aggregate volume grows with world size rather than staying fixed, the per-pair sizes are irregular (the straggler problem in transit), and the pattern is a full mesh that punishes any slow link because every rank waits on its slowest partner. That is why a naive `torch.distributed.all_to_all` is a decode-throughput trap and why DeepEP's kernels exist.

## Mixing EP with TP and DP: the real layer layout

Pure expert parallelism handles the FFN, but a transformer layer is more than its FFN. Attention has its own weights and its own KV cache, and those do not want the same parallelism as the experts. Production MoE serving therefore uses **mixed parallelism**: one scheme for attention, another for the experts, stitched together by the all-to-all.

![Layered stack of one MoE transformer layer showing data-parallel attention, expert-parallel FFN, and two all-to-all collectives](/imgs/blogs/serving-moe-models-at-scale-4.webp)

The stack shows the layout DeepSeek recommends for their model. Attention (MLA) runs **data-parallel (DP)**: each GPU holds a full copy of the attention weights and processes a different slice of the batch, keeping that slice's KV cache local. Because MLA's weights and compressed KV are small, replicating attention across 144 GPUs is cheap, and DP attention avoids the all-reduce that tensor-parallel attention would need every layer. The FFN runs **expert-parallel**: the 256 experts are sharded across those same GPUs, so the FFN weights are *not* replicated — they are the huge part, and sharding them is the whole point. The shared expert is replicated per GPU (it is small and always runs). Between attention and experts sits the dispatch all-to-all; between experts and the residual add sits the combine all-to-all.

This is a genuinely different mental model from dense serving, so it is worth naming why each choice is made:

- **Attention → DP (or TP for smaller models).** Attention weights are a small fraction of an MoE model, so replication is affordable and it keeps the KV cache local to a GPU (no KV sharding, no cross-GPU attention). DeepSeek uses DP attention specifically because MLA makes the per-GPU KV cheap. For an MoE where attention is a larger share, or where the KV cache is too big to replicate, tensor-parallel attention is the alternative.
- **FFN → EP.** The experts are where the parameters live, so they must be sharded, and EP shards them along the natural boundary (whole experts) rather than cutting into each expert's matrices (which would be TP-within-expert, adding more all-reduces).
- **The two all-to-alls are the tax.** They exist only because attention's data layout (batch-sharded) and the FFN's data layout (expert-sharded) disagree. Every technique in the next two sections is about making that tax smaller or hiding it.

#### The shared expert and the dense first layers

Two DeepSeek-V3 details clarify the "58 of 61 layers" and the "256 + 1" counts you keep seeing. First, the model's *first three* transformer layers use a plain dense FFN, not MoE — early layers learn general features that do not benefit from specialization, so keeping them dense avoids routing overhead where it buys nothing. That is why only 58 of the 61 layers pay the all-to-all tax. Second, alongside the 256 routed experts sits **one shared expert** that *every* token visits, always active, never routed. The shared expert absorbs the common computation that would otherwise be duplicated across many routed experts, freeing the router to specialize the rest; because it always runs, it is replicated on every GPU rather than sharded, and it never touches the all-to-all. When you size the FFN weights, remember it: DeepSeek-V3's per-token active compute is the top-8 routed experts *plus* the shared expert, and the shared expert's weights sit on every rank.

A quick contrast of the parallelism axes, since MoE serving mixes all of them:

| Axis | What it shards | Comm per layer | Best for in MoE serving |
|---|---|---|---|
| Tensor parallel (TP) | Each weight matrix, across GPUs | All-reduce (fixed size) | Attention when KV is large; experts on few GPUs |
| Pipeline parallel (PP) | Whole layers, across GPUs | Point-to-point (small) | Spanning many nodes cheaply; adds bubble latency |
| Data parallel (DP) | The batch (weights replicated) | None for attention | MLA attention with small KV |
| Expert parallel (EP) | Whole experts, across GPUs | All-to-all (variable) | The FFN — the parameter-heavy part |

Real deployments combine these. DeepSeek's decode setup is DP144 attention + EP144 experts on 18 nodes; a smaller Mixtral deployment might be TP2 attention + EP2 experts on a single 2-GPU box. The composition is the art; the constraint is always the same triangle.

## The all-to-all cost and why overlap is the whole game

We have established that each MoE layer pays two all-to-alls. Now look at where the time actually goes on the decode critical path, because that is what your p99 token latency (TPOT — time per output token) is made of.

![Timeline of one MoE decode layer showing gate, dispatch all-to-all, expert GEMM, combine all-to-all, and overlap](/imgs/blogs/serving-moe-models-at-scale-5.webp)

The timeline is a representative decode layer at moderate batch. The gate is nearly free (~5 µs). The expert GEMM is real work (~200 µs). But the two all-to-alls together (~60 µs each, ~120 µs) are *comparable to the compute*. At small decode batch the GEMM shrinks and the all-to-all does not, so communication can outweigh compute entirely. That is the crux: on the decode path, MoE communication is not a small correction — it is a co-equal cost, and if you run it serially after the compute you roughly double your per-layer latency.

The fix is **overlap**: while GPU tensor cores chew on one micro-batch's expert GEMM, the network moves the *next* micro-batch's dispatch (and the *previous* one's combine). Done well, the all-to-all time disappears behind the arithmetic and the per-layer latency approaches the GEMM time alone. This requires kernels that (a) are fast on their own and (b) release the GPU's compute streams so they can run concurrently with the network.

That is exactly what **DeepEP** provides. DeepEP is DeepSeek's open-source expert-parallel communication library, and it ships two families of kernels tuned for the two serving phases:

- **Normal (high-throughput) kernels** for prefill and training, optimized to saturate both NVLink (intra-node, reported ~150+ GB/s) and RDMA (inter-node, reported ~45–50 GB/s on 400 Gb/s InfiniBand). These move the most bytes per second.
- **Low-latency (pure-RDMA) kernels** for decode, where the batch is small and what matters is round-trip latency, not raw bandwidth. DeepEP reports dispatch latencies on the order of ~160 µs for decode-shaped workloads, and — crucially — a hook-based design that overlaps communication with computation *without consuming GPU SM resources*, so the expert GEMM runs at full speed alongside the transfer.

The difference between a naive `torch.distributed.all_to_all` and DeepEP's low-latency kernels on the decode path can be several times the effective decode throughput, because the naive path serializes communication and steals SMs. If you take one implementation lesson from this post: on the MoE decode path, the collective library is not an implementation detail, it is a top-three throughput lever.

### Two-batch overlap, in numbers

It helps to see the overlap as a software pipeline across micro-batches rather than one serialized layer. Split the decode batch into two micro-batches, A and B, and stagger them: while the expert GEMM for micro-batch A runs on the tensor cores, the dispatch all-to-all for micro-batch B runs on the network, and the combine all-to-all for the *previous* step drains at the same time. Three things happen at once on three different resources — SMs computing, RDMA NICs dispatching, NVLink combining — so the wall-clock per layer collapses toward ${\max(\text{GEMM}, \text{all-to-all})}$ instead of their sum. This is SGLang's **two-batch overlap**, and it is the reason DeepEP's low-latency kernels are designed to consume *no* SMs: if the communication kernel stole even 10% of the streaming multiprocessors, the overlapped GEMM would slow by that much and the overlap would partly cancel itself.

Put numbers on it. If a layer is ~200 µs of GEMM and ~120 µs of all-to-all, running them serially costs ~320 µs; perfect overlap costs ~200 µs — a 1.6× speedup on the decode critical path, compounding across all 58 MoE layers. You never get perfect overlap in practice (there is a tail where one micro-batch has no partner to hide behind), but recovering even two-thirds of that gap is the difference between a decode path that is communication-bound and one that is compute-bound. Overlap is not a knob you turn at the end; it is the shape of a competitive MoE decode loop.

#### Worked example: what interconnect the decode path demands

Turn the volume math into a hardware requirement. At EP across nodes, each GPU in a decode step must dispatch and combine its share of tokens every layer, and the per-layer wire time has to fit inside the layer's compute budget for overlap to hide it. Take a decode step where a GPU handles 128 tokens at DeepSeek-V3's 168 KB per token per layer: that is ${128 \times 168\text{ KB} \approx 21}$ MB of all-to-all traffic per layer per GPU. To hide it behind a ~200 µs expert GEMM, the GPU's network must move 21 MB in under 200 µs — about 105 GB/s of *effective* bidirectional bandwidth. A single 400 Gb/s InfiniBand NIC delivers only ~50 GB/s effective one-way, so you need the traffic to be dominated by intra-node NVLink (150+ GB/s) with a minority crossing the NIC — which is exactly what node-limited routing arranges. Flip it around and the provisioning rule falls out: if your effective internode bandwidth per GPU is below roughly the per-layer dispatch volume divided by the GEMM time, the all-to-all will not hide, and your decode path is communication-bound no matter how good your kernels are.

## Expert load imbalance and the straggler problem

Now the failure mode that eats MoE's FLOP savings if you ignore it. All-to-all is a **barrier**: the combine cannot start until every GPU has finished its local experts, because the layer output needs all of them. So the layer's wall-clock time is set by the *slowest* GPU — the one that got the most tokens. And routing, being learned and skewed, guarantees that one GPU gets the most tokens.

![Before and after comparison of a hot overloaded expert versus EPLB load balancing with redundant experts](/imgs/blogs/serving-moe-models-at-scale-6.webp)

Quantify the damage. In a batch of ${B}$ tokens with top-k routing over ${E}$ experts, the total expert assignments are ${B \cdot k}$, so the *mean* tokens per expert is:

$$\bar{L} = \frac{B \cdot k}{E}$$

Define the **load-imbalance factor** as the ratio of the hottest expert's load to the mean:

$$\rho = \frac{L_{\max}}{\bar{L}}$$

Because the all-to-all is a barrier and the FFN compute is proportional to token count, the layer's compute time scales with ${L_{\max} = \rho \cdot \bar{L}}$. The GPUs that received only the mean load finish in ${1/\rho}$ of the bottleneck GPU's time and then *idle* at the barrier. So the effective fleet utilization is roughly:

$$U \approx \frac{1}{\rho}$$

A realistic ${\rho = 2.4}$ (one expert 2.4× the mean, which unbalanced routing easily produces) means ~42% utilization: you bought the GPUs, and more than half their cycles evaporate waiting for one straggler. That is the FLOP savings of MoE handed straight back.

#### Worked example: capacity, drop, and overflow

Consider a decode batch of ${B = 8192}$ tokens, ${k = 8}$, ${E = 256}$:

$$\bar{L} = \frac{8192 \times 8}{256} = 256 \text{ tokens per expert on average}$$

Two ways to handle the fact that the hottest expert wants more than 256:

- **Capacity + drop.** Set a **capacity factor** ${C}$ and cap each expert at ${C \cdot \bar{L}}$ tokens. With ${C = 1.25}$ each expert holds at most 320 tokens; a token routed to an over-capacity expert is *dropped* (its FFN output is zeroed, only the residual passes). This bounds compute and keeps buffers fixed-size, but dropping tokens at inference degrades the response unpredictably per request — usually unacceptable for user-facing serving. Common in *training*, rare in serving.
- **No drop (ragged).** Process whatever each expert receives. Quality is preserved, but the hottest expert sets the step time — the straggler. This is what most inference engines do by default, which is why the straggler is the serving problem and dropping is the training problem.

The production answer is neither: **rebalance the placement** so no single GPU is hot. That is EPLB.

### EPLB and redundant experts

**EPLB** (Expert-Parallel Load Balancer) is DeepSeek's open-source algorithm for placing experts on GPUs to equalize load. Its lever is **redundant experts**: replicate the hottest experts onto multiple GPUs so their token load is split. If expert 66 is receiving 3.1× the mean, place three copies of it on three different GPUs and route a third of its tokens to each. The right panel of the figure above shows the result — the hot expert's load spreads across copies, per-GPU utilization climbs from ~40% to ~88%, and the layer's step time falls from 2.4× ideal toward ~1.15× ideal.

EPLB computes the placement from *estimated* expert loads (measured over recent traffic), and offers two policies: a **hierarchical** policy when the number of nodes divides the expert groups cleanly (balance within a node first, then across nodes, to keep replicas NVLink-local), and a **global** policy otherwise. Because routing distributions drift slowly, the placement can be recomputed periodically (minutes, not milliseconds) from live telemetry — which means you need that telemetry.

#### Worked example: how many redundant experts to add

Suppose the monitor reports that on a given layer the hottest expert takes ${\rho = 2.4}$ times the mean load, and you want to bring the effective imbalance down to ${\rho' = 1.2}$. Redundant experts split a hot expert's traffic across copies: place ${r}$ copies and each sees ${1/r}$ of the tokens. To pull a ${2.4\times}$ expert down to the ${1.2\times}$ target:

$$r = \left\lceil \frac{\rho}{\rho'} \right\rceil = \left\lceil \frac{2.4}{1.2} \right\rceil = 2 \text{ copies of that expert}$$

If three experts are hot at ${2.4\times}$, that is three extra physical experts, growing the physical count from 256 to 259 and the per-GPU weight footprint slightly. The trade is explicit: spend a little more HBM to reclaim a lot of utilization. A redundant copy costs one expert's worth of parameters — for DeepSeek-V3, roughly ${671\text{B}/256 \approx 2.6}$B params, about 2.6 GB in FP8 — and buys back tens of percent of fleet throughput, so the exchange rate is overwhelmingly favorable until you run out of HBM.

Here is a greedy placement that turns a measured load vector into a physical-expert layout — the shape of what EPLB does internally (the real algorithm adds node-locality constraints so replicas stay NVLink-reachable):

```python
import numpy as np

def place_experts(loads: np.ndarray, n_gpus: int, n_redundant: int):
    """loads[i] = measured tokens for logical expert i.
    Returns gpu_of[phys] and the resulting per-GPU imbalance."""
    # 1. Replicate the hottest experts, splitting their load across copies.
    phys = [{"logical": i, "load": loads[i]} for i in range(len(loads))]
    for _ in range(n_redundant):
        hot = max(range(len(phys)), key=lambda p: phys[p]["load"])
        logi = phys[hot]["logical"]
        copies = [p for p in phys if p["logical"] == logi]
        share = loads[logi] / (len(copies) + 1)   # split evenly over copies + 1
        for c in copies:
            c["load"] = share
        phys.append({"logical": logi, "load": share})
    # 2. Greedily bin-pack physical experts onto GPUs, heaviest first (LPT).
    order = sorted(range(len(phys)), key=lambda p: -phys[p]["load"])
    gpu_load = np.zeros(n_gpus)
    gpu_of = {}
    for p in order:
        g = int(np.argmin(gpu_load))     # least-loaded GPU takes the next expert
        gpu_of[p] = g
        gpu_load[g] += phys[p]["load"]
    imbalance = gpu_load.max() / gpu_load.mean()
    return gpu_of, imbalance
```

The longest-processing-time (LPT) bin-packing in step 2 is the same greedy heuristic used to spread any set of weighted jobs across machines, and it lands within a few percent of optimal for this shape of problem. Feed it the `counts` your monitor already exports, recompute on a slow cadence (minutes), and push the new placement to the fleet — that is the closed loop that keeps ${\rho}$ near 1.2 as your traffic distribution drifts.

### An expert-load monitor you can ship

You cannot balance what you cannot see. Here is a monitor that accumulates per-expert token counts from the router (the `counts` vector from earlier) and exports the imbalance factor and per-expert load to Prometheus, so a Grafana panel or an EPLB recompute job can act on it:

```python
from prometheus_client import Gauge, Counter, start_http_server
import torch

# One time series per expert, plus a fleet-level imbalance gauge.
EXPERT_TOKENS = Counter(
    "moe_expert_tokens_total", "Tokens routed to each expert", ["layer", "expert"]
)
IMBALANCE = Gauge(
    "moe_load_imbalance_factor", "L_max / L_mean across experts", ["layer"]
)
DROPPED = Counter("moe_tokens_dropped_total", "Tokens dropped at capacity", ["layer"])

class ExpertLoadMonitor:
    def __init__(self, n_experts: int, port: int = 9400):
        self.n_experts = n_experts
        start_http_server(port)  # scraped by Prometheus

    def observe(self, layer: int, counts: torch.Tensor, dropped: int = 0):
        # counts: [n_experts] tokens routed to each expert this step
        counts = counts.detach().float().cpu()
        for e in range(self.n_experts):
            EXPERT_TOKENS.labels(layer=str(layer), expert=str(e)).inc(counts[e].item())
        mean = counts.mean().clamp(min=1e-6)
        rho = (counts.max() / mean).item()
        IMBALANCE.labels(layer=str(layer)).set(rho)
        if dropped:
            DROPPED.labels(layer=str(layer)).inc(dropped)

# In the serving loop, after routing each MoE layer:
#   monitor.observe(layer_idx, counts, dropped=n_dropped)
```

The alert that matters is on `moe_load_imbalance_factor`. A Prometheus rule such as the following tells you when a hot expert is silently taxing the fleet, long before it shows up as a throughput regression:

```yaml
groups:
  - name: moe-serving
    rules:
      - alert: MoEExpertImbalanceHigh
        expr: max by (layer) (moe_load_imbalance_factor) > 1.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "MoE layer {{ $labels.layer }} imbalance {{ $value }} (>1.5)"
          description: "A hot expert is gating the all-to-all. Consider EPLB recompute or more redundant experts."
```

An imbalance factor drifting above ~1.5 sustained is the signal to recompute EPLB placement or add redundant copies. Below ~1.2 you are in good shape and further balancing buys little.

## Prefill and decode want different EP degrees

Here is a subtlety that trips up first-time MoE operators: the *ideal expert-parallel width is different for prefill and decode*, because the two phases stress different resources. Prefill (processing the prompt) is **compute-bound** — it runs the full prompt through the network in big, dense-ish matmuls, so it wants enough GPUs to hold the model and saturate compute, but a *narrow* EP group keeps the all-to-all cheap. Decode (generating tokens one at a time) is **latency- and memory-bound** — each step touches every expert with a tiny batch, so spreading experts across a *wide* EP group reduces the per-GPU expert count (and thus the grouped-GEMM cost and the weight-read pressure) even though it widens the all-to-all.

![Dataflow graph of prefill and decode running as separate clusters with EP32 prefill and EP144 decode bridged by a KV cache transfer](/imgs/blogs/serving-moe-models-at-scale-7.webp)

This is why large-scale MoE serving pairs naturally with **prefill/decode (PD) disaggregation**: run prefill on one pool of GPUs at a narrow EP degree, decode on a separate pool at a wide EP degree, and transfer the KV cache between them. DeepSeek's disclosed production topology does exactly this — prefill on EP32 (4 nodes) and decode on EP144 (18 nodes) — and MLA's tiny ~70 KB/token KV cache is what makes the cross-cluster transfer cheap enough to be worth it. The full PD story (compute-bound prefill vs memory-bound decode, the NCCL P2P transfer, the routing) is its own deep-dive in [prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation); the MoE-specific twist is that the two phases don't just want different *hardware*, they want different *parallelism widths*.

#### Worked example: why EP32 for prefill and EP144 for decode

Put the two phases side by side on DeepSeek-V3's 256 experts. Prefill runs the whole prompt through the network in one shot, so the per-GPU batch is *large* — every prompt token is present at once. Even at EP32 (256/32 = 8 experts per GPU) each expert's grouped GEMM sees hundreds of tokens and sits comfortably compute-bound; a wider EP would only add all-to-all cost for no compute benefit, because prefill already has the batch it needs. Decode is the opposite: one token per sequence per step, so the per-GPU batch is *tiny*, and the only way to keep each expert's weight-read amortized and its GEMM from starving is to shrink the per-GPU expert count. At EP144 (256/144 ≈ 1.8 experts per GPU) each GPU holds barely more than one expert, so the weights it re-reads per step are minimal and the tiny decode batch is spread across the most GPUs possible. The narrow-for-prefill, wide-for-decode split falls straight out of the batch asymmetry: prefill has batch to spare and guards the all-to-all, while decode is batch-starved and spends GPUs to buy back per-expert efficiency.

The practical implication: do not pick one EP degree for your whole deployment and call it done. Benchmark prefill and decode separately, and if you are at a scale where a dedicated pool for each is affordable, disaggregate. At small scale (a single node, one Mixtral instance) the phases share the same GPUs and the same EP degree, and that is fine — the asymmetry only pays off when you have enough traffic to keep two specialized pools busy.

## Serving MoE in practice: vLLM and SGLang

Enough mechanics. Here is how you actually launch these models. The two open-source engines that lead on MoE serving are **vLLM** and **SGLang**, and both expose expert parallelism as a first-class flag.

![Comparison matrix of vLLM, SGLang, DeepEP, and TensorRT-LLM across expert parallelism, prefill-decode disaggregation, FP8 dispatch, and low-latency decode](/imgs/blogs/serving-moe-models-at-scale-8.webp)

### vLLM with expert parallelism

In vLLM, expert parallelism is enabled with `--enable-expert-parallel`. When it is on, the MoE experts are sharded across the combined world (tensor-parallel × data-parallel ranks) rather than replicated. For a smaller MoE like Mixtral 8x7B on a single 2-GPU box:

```bash
# Mixtral 8x7B on 2x H100: TP shards attention, EP shards the 8 experts
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

For DeepSeek-V3, one node cannot hold the weights, so you need multiple nodes and the DeepSeek-recommended mix of **data-parallel attention + expert-parallel FFN**. On two 8-GPU nodes (16 GPUs), the clean configuration is data parallelism across nodes with EP spanning the full world:

```bash
# DeepSeek-V3 across 2 nodes (16 GPUs): DP attention + EP experts.
# EP degree = tensor_parallel_size * data_parallel_size = 8 * 2 = 16.
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --trust-remote-code \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92
```

The key idea: `--enable-expert-parallel` changes *how the FFN weights are placed* (sharded by expert) without changing the TP/DP layout of attention. The effective EP degree is the product of the parallel sizes, so scaling out nodes scales out EP. vLLM's V1 engine adds data-parallel attention and prefill/decode support that make this composition efficient for DeepSeek-class models.

For a mid-size MoE that spans more than one card but does not need DeepSeek-scale EP, you often want **TP within a node and EP across the world** — TP keeps each node's attention and per-expert matmuls on fast NVLink, while EP spreads whole experts across the ranks. Mixtral 8x22B on four GPUs is the compact version:

```bash
# Mixtral 8x22B on 4x H100: TP4 shards attention + each expert's matrices,
# and --enable-expert-parallel shards the 8 experts across the same 4 ranks.
vllm serve mistralai/Mixtral-8x22B-Instruct-v0.1 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90
```

The mental model for the EP degree is a single equation: with expert parallelism on, the experts are sharded across the **product** of the tensor-parallel and data-parallel sizes.

$$\text{EP degree} = \text{tensor-parallel size} \times \text{data-parallel size}$$

So TP8 × DP2 gives EP16, TP8 on a single node gives EP8, and TP4 gives EP4. This is why "scale out nodes" and "scale out EP" are the same action in vLLM — every rank you add to the TP×DP world is another shard of experts. The rule to internalize: pick TP high enough that each expert's matmul and the attention stay NVLink-local within a node, then let DP across nodes carry the EP degree the rest of the way.

### SGLang with large-scale EP and DeepEP

SGLang leans harder into large-scale EP and integrates DeepEP directly. Its knobs are `--ep-size` for the expert-parallel degree, `--enable-dp-attention` for data-parallel attention, and DeepEP mode for the kernels:

```bash
# DeepSeek-V3 on SGLang, single node (8 GPUs), DP attention + EP experts + DeepEP.
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --ep-size 8 \
  --enable-dp-attention \
  --dp-size 8 \
  --enable-deepep-moe \
  --deepep-mode auto \
  --trust-remote-code \
  --mem-fraction-static 0.90
```

For the multi-node, disaggregated deployment that reproduces DeepSeek's production numbers, SGLang adds PD disaggregation, EPLB, and two-batch overlap on top of this — the launch spans a prefill pool and a decode pool with different `--ep-size` values, exactly the asymmetry from the previous section. TensorRT-LLM offers a comparable feature set with native FP8 on the NVIDIA stack; the trade there is peak performance on NVIDIA hardware against the open-source flexibility of vLLM and SGLang.

### Benchmarking across EP degrees

Never trust a single configuration. The right EP degree depends on your model, your hardware, and your traffic shape, so sweep it. Here is a benchmark harness that launches an OpenAI-compatible endpoint at several EP degrees and measures decode throughput and TPOT:

```python
import time, requests, subprocess, statistics, concurrent.futures as cf

def bench(base_url: str, n_requests: int = 128, out_tokens: int = 256):
    prompt = "Explain expert parallelism in one paragraph."
    def one():
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/v1/completions", json={
            "model": "deepseek", "prompt": prompt,
            "max_tokens": out_tokens, "temperature": 0.0,
        }, timeout=120)
        dt = time.perf_counter() - t0
        toks = r.json()["usage"]["completion_tokens"]
        return toks, dt
    with cf.ThreadPoolExecutor(max_workers=n_requests) as ex:
        results = list(ex.map(lambda _: one(), range(n_requests)))
    total_toks = sum(t for t, _ in results)
    wall = max(dt for _, dt in results)
    tpot_ms = 1000 * statistics.mean(dt / t for t, dt in results if t)
    return {"throughput_tok_s": total_toks / wall, "tpot_ms": round(tpot_ms, 2)}

# Sweep: launch the server at each EP degree, warm up, then bench.
for ep in [8, 16, 32]:
    print(f"EP={ep}", bench("http://localhost:8000"))
    # (launch/teardown of the server per EP omitted; drive it from your
    #  orchestration so each run is a clean process on the right GPU count)
```

The pattern that almost always appears: aggregate throughput rises with EP degree (more GPUs, more experts spread out, bigger batches), while per-request TPOT can *worsen* past a point as the all-to-all widens across more nodes. You are looking for the knee — the EP degree that maximizes throughput per dollar while keeping TPOT under your SLA.

To make the knee concrete, here is the shape of a representative sweep for a DeepSeek-class model as EP widens across nodes (illustrative numbers, decode phase, moderate concurrency):

| EP degree | GPUs | Per-GPU experts | Aggregate decode tok/s | Median TPOT | Where the time goes |
|---|---|---|---|---|---|
| 8 | 8 (1 node) | 32 | 9k | 22 ms | GEMM-bound; batch cramped by per-GPU expert count |
| 16 | 16 (2 nodes) | 16 | 17k | 26 ms | balanced; first node crossing adds RDMA all-to-all |
| 32 | 32 (4 nodes) | 8 | 31k | 31 ms | throughput scales; all-to-all now a real share of TPOT |
| 64 | 64 (8 nodes) | 4 | 52k | 40 ms | knee — throughput still climbing, TPOT past SLA for many apps |

The pattern is the one to expect every time: aggregate throughput rises almost linearly with EP degree because you spread experts thinner (fewer per GPU, so bigger batches and lighter weight-read pressure), while median TPOT creeps up as the all-to-all widens across more node boundaries. If your SLA is "TPOT under 30 ms," this table says EP16 is your ceiling and EP32 already breaks it; if you serve a batch-throughput workload with a loose latency budget, EP64 leaves nothing on the table. There is no single right EP degree — there is only the right one for your latency budget, and you find it by sweeping, not by guessing.

## Benchmarks and measurement on named hardware

Numbers ground everything. Here is the before→after that motivates the whole apparatus — the naive attempt to serve DeepSeek-V3 versus the large-scale EP deployment — on H100/H800-class hardware. Treat the large-scale figures as reported production and reproduction results (DeepSeek's own inference-system disclosure and SGLang's reproduction), and the naive row as the arithmetic of the capacity wall.

| Deployment | Hardware | Fits? | Decode throughput | Notes |
|---|---|---|---|---|
| Naive single-node replication | 8×H100 (640 GB) | No — OOM on load | 0 | 671 GB FP8 weights > 640 GB HBM |
| Minimum viable EP | 16×H100 (2 nodes), EP16 | Yes, tight | Modest; small batch | Weights fit (~42 GB/GPU), little KV headroom |
| Prefill pool | 32×H800 (4 nodes), EP32 | Yes | ~73.7k input tok/s per node | DeepSeek disclosed prefill figure |
| Decode pool | 144×H800 (18 nodes), EP144 | Yes | ~14.8k output tok/s per node | DeepSeek disclosed decode figure |
| SGLang reproduction | 96×H100 (12 nodes), PD+EP | Yes | ~52.3k input / ~22.3k output tok/s per node | Public reproduction with DeepEP + EPLB |

Two things to read out of this table. First, the jump from "0 (OOM)" to a working deployment is a *capacity* step, not a performance-tuning step — you cross it by adding GPUs, full stop. Second, the per-node throughput at large EP is high precisely because the wide expert parallelism keeps per-GPU expert counts and weight-read pressure low while EPLB keeps utilization up; that is the payoff for all the all-to-all machinery.

#### Worked example: Mixtral active vs total, and its serving footprint

Mixtral 8x7B is the friendly end of the spectrum and a good sanity check on the ${R}$ ratio. It has 46.7B total parameters and 12.9B active per token (top-2 of 8 experts), so ${R \approx 3.6}$. Footprints:

- **BF16:** ${46.7 \times 2 = 93.4}$ GB — does not fit one 80 GB card; needs 2×A100 80GB or 2×H100, or a single H200 141GB.
- **FP8/INT8:** ~46.7 GB — fits one 80 GB card with room for KV.
- **INT4 (AWQ/GPTQ):** ~23 GB — fits comfortably on one card; see [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) for the accuracy trade.

Compute per token touches only 12.9B params (~25.8 GB in BF16), so even in the tightest single-card INT4 setup, you are provisioning ~23 GB of resident weights to run ~6.5 GB (INT4) of active compute per token — the ${R \approx 3.6}$ asymmetry in miniature. The lesson scales: Mixtral is servable on a single modern GPU, DeepSeek-V3 needs a rack, and the difference is entirely ${R}$ times the total-parameter count.

## Operational failure modes and their fixes

Everything above is the happy path. Here are the five ways MoE serving actually breaks in production, and the lever that fixes each — a runbook to keep next to the dashboard.

| Symptom | Root cause | Fix |
|---|---|---|
| `CUDA out of memory` on model load, before any request | Total-parameter footprint exceeds fleet HBM — the capacity wall, not a leak | Add GPUs to clear ${R \times P_{\text{active}}}$ bytes, move to higher-capacity cards (H200/B200), or drop precision (FP8 to INT4) |
| Decode throughput far below the GEMM roofline; GPUs show bursty low utilization | Hot-expert straggler — one GPU gates the all-to-all barrier every layer | Watch `moe_load_imbalance_factor`; recompute EPLB placement; add redundant copies of the hottest experts |
| p99 TPOT spikes when traffic crosses a node boundary | All-to-all saturating internode RDMA; overlap not hiding the transfer | Enable DeepEP low-latency kernels + two-batch overlap; cap node fan-out (node-limited routing); verify FP8 dispatch is on |
| KV-cache evictions or requests queuing despite spare compute | Weights ate the HBM; too little left for KV at the batch you want | Widen EP (more GPUs amortizes the fixed weight cost, frees per-GPU HBM); shorten max context; or disaggregate prefill and decode |
| Imbalance factor drifts up over hours with no code change | Traffic distribution shifted away from the pretrained router balance | Recompute EPLB from *recent* telemetry on a slow cadence; the pretrained bias balanced the training mix, not today's mix |

The unifying diagnosis is that every one of these is a *capacity or barrier* problem, not a compute problem — the through-line of this whole post. When an MoE deployment is slow, the answer is almost never "the GPUs are too slow"; it is "the GPUs are waiting" — on HBM they could not fit into, on a straggler at the barrier, or on a wire the overlap did not hide. Instrument those three and MoE serving becomes predictable.

## Case studies

### DeepSeek-V3 with DeepEP and EPLB

DeepSeek-V3 (671B total, 37B active, 256 routed experts + 1 shared, top-8, FP8, MLA) is the reference design for large-scale MoE serving, and DeepSeek disclosed the production system in their "DeepSeek-V3/R1 Inference System Overview." The salient choices: **PD disaggregation** with prefill on EP32 (4 nodes) and decode on EP144 (18 nodes); **DP attention** (MLA replicated per GPU, KV cache local) paired with **EP experts**; **DeepEP** for the dispatch/combine kernels with FP8 dispatch and compute-communication overlap; and **redundant experts** placed by EPLB-style load balancing to keep the hottest GPU from gating the barrier. The disclosed per-node throughput — roughly 73.7k input tokens/s during prefill and 14.8k output tokens/s during decode on H800 nodes — is what makes the economics work; DeepSeek reported a theoretical cost-profit margin well above 500% at their published token prices. The point for a serving engineer is not the exact numbers but the *shape*: every technique in this post appears in that one system, composed together, because at 671B none of them is optional.

Two details reward a second look. The **disaggregation ratio** is not arbitrary: DeepSeek runs roughly 4 nodes of prefill (EP32) feeding 18 nodes of decode (EP144), because decode is the long pole — a 1,000-token answer is 1,000 sequential decode steps against a single prefill pass, so the decode pool must be several times larger to keep the two sides from starving each other. And the **redundant experts** are not a handful: to keep the hottest GPU off the barrier at EP144, the balancer replicates a meaningful fraction of experts, so the *physical* expert count exceeds the 256 logical ones and per-GPU placement is recomputed from live load. The composed system is what makes the disclosed cost-profit margin plausible — not any single kernel, but the full stack of PD split, DP attention, EP experts, DeepEP overlap, and EPLB placement running together.

### Mixtral 8x7B and 8x22B

Mixtral (Mistral AI) is the MoE that most teams actually run, because it fits on hardware they already have. Mixtral 8x7B (46.7B total, 12.9B active) serves happily on 2×A100 or a single H200, and even MoE-naive engines handle it because at EP2 the all-to-all is intra-node NVLink and the imbalance across 8 experts is mild compared to 256. Mixtral 8x22B (141B total, 39B active, top-2 of 8) pushes into multi-GPU territory (2–4 cards) and starts to reward EP and load monitoring. Mixtral is the case study for *when MoE is easy*: few experts, top-2 routing, single-node interconnect. The hard problems in this post — internode all-to-all, EPLB, PD disaggregation — mostly do not bite until you scale expert count and cross node boundaries.

### Large-scale EP reproduction (SGLang)

The SGLang team publicly reproduced DeepSeek-scale serving on 96 H100 GPUs (12 nodes) using PD disaggregation, large-scale expert parallelism, DeepEP kernels, EPLB, and two-batch overlap, reporting roughly 52.3k input tokens/s and 22.3k output tokens/s per node — throughput in the same class as DeepSeek's own disclosed figures, at a cost on the order of \$0.20 per million output tokens. This is the most useful case study for practitioners because it is open source and reproducible: it demonstrates that the techniques are not a proprietary DeepSeek secret but a composable open-source stack, and it quantifies how much of the throughput comes from each piece (large-scale EP and overlap are the biggest levers; EPLB recovers the utilization that skew would otherwise waste).

### Qwen-MoE family

Qwen's MoE line spans the range: Qwen3-30B-A3B (30B total, 3B active) is a single-GPU MoE with an aggressive ${R = 10}$ ratio, while Qwen3-235B-A22B (235B total, 22B active, 128 experts, top-8) is a multi-node deployment much like DeepSeek but smaller. The Qwen family is a good reminder that "MoE" is not one workload — a 30B-A3B model is a laptop-adjacent inference target that happens to be memory-heavy, and a 235B-A22B model is a rack-scale EP deployment, and they share almost no operational characteristics beyond the router.

#### Worked example: sizing the two ends of the Qwen-MoE range

The Qwen family makes the "not one workload" point quantitative. Qwen3-30B-A3B in FP8 is 30 GB of weights — it loads on a *single* H100 with ~46 GB left for KV and batch, and its all-to-all never leaves the card because all 128 experts are local (EP1). It is a memory-heavy single-GPU model, full stop: no RDMA, no EPLB, no disaggregation. Qwen3-235B-A22B in FP8 is 235 GB — four H100s minimum for weights, realistically eight to leave batch headroom, and now every technique in this post switches on: EP across GPUs, all-to-all on the interconnect, a router over 128 experts that will skew under real traffic, and a payoff that only materializes at enough QPS to fill the wide batch. Same model family, same top-8 router, a 7.8× jump in weights — and the operational story crosses from "laptop-adjacent" to "rack-scale" somewhere in the middle. The lesson to carry out of the case studies: read an MoE's ${R}$ and its total-parameter count *first*, because together they tell you which of these two worlds you are in before you write a single launch flag.

## When to use this (and when not to)

MoE is not free, and the honest recommendation is that it wins in a specific regime and loses outside it. The costs are exactly the three we derived: you provision HBM for *all* the experts (the ${R}$ multiplier on memory), you provision interconnect for the all-to-all (a new capacity dimension), and you pay an operational tax for load balancing (EPLB, monitoring, redundant experts). Whether those costs are worth it depends on scale and traffic.

**Use MoE serving when:**

- **You want frontier quality at a fraction of the active FLOPs, and you have the HBM.** This is the whole reason MoE exists. If you can afford the memory footprint, DeepSeek-V3 gives you a frontier model at 37B-active compute cost.
- **Your traffic is high and steady enough to keep many GPUs busy.** Large-scale EP amortizes the all-to-all and fills the batch. At high QPS, the FLOP savings translate into real cost-per-token wins (the SGLang reproduction's ~\$0.20/1M output tokens is a serious number).
- **You can cross node boundaries with fast interconnect (InfiniBand/RoCE ≥ 400 Gb/s).** Without it, the all-to-all becomes the bottleneck and the FLOP savings vanish into wire time.

**Do not use MoE serving (or do not scale its EP) when:**

- **Your traffic is low.** At low QPS the batch is small, the grouped GEMMs are tiny, the all-to-all latency dominates, and utilization is poor. A dense model of comparable *active* size, or a well-quantized smaller dense model, will serve low-traffic workloads at better latency and lower complexity. The memory you tie up in idle experts is pure cost.
- **You lack the interconnect.** Running internode EP over slow Ethernet turns the all-to-all into a 50 ms-per-token disaster. If you cannot fit the model within a single NVLink domain and you do not have RDMA, do not run large-scale EP.
- **You cannot afford the HBM.** If ${R \times P_{\text{active}}}$ in your chosen precision exceeds your fleet's total HBM, MoE is a non-starter regardless of how few FLOPs it uses. Buy the memory or pick a dense model.
- **Operational simplicity matters more than peak efficiency.** MoE adds router monitoring, EPLB, redundant-expert management, and a communication library you must tune. A single-node dense model is dramatically simpler to run. For a small team serving modest traffic, that simplicity is often worth more than the FLOP savings.

The clean decision rule: MoE serving pays off at **high scale with fast interconnect and enough HBM**, and is a liability at **low scale, on slow networks, or under tight memory budgets**. The 30B-A3B end of the family blurs this — a small MoE on one GPU is easy — but the moment you need multi-node EP, the full cost structure applies.

## Key takeaways

- **MoE is memory-capacity-bound, not compute-bound.** The GPU must hold *every* expert (total params) while compute touches only top-k (active params). Size your fleet by the ${R = P_{\text{total}}/P_{\text{active}}}$ multiplier on HBM, not by active FLOPs. DeepSeek-V3's ${R \approx 18}$ is why a 37B-active model needs a rack.
- **The router makes batching irregular.** Data-dependent top-k routing fragments a clean batch into ragged per-expert GEMMs and forces a physical token scatter over the interconnect. This is the source of every downstream complication.
- **Expert parallelism means two all-to-alls per layer.** Dispatch and combine are the defining communication pattern. All-to-all volume is ${k \cdot d}$ bytes per token per direction per layer — provision interconnect for it, and never extrapolate single-node MoE benchmarks across a node boundary.
- **On the decode path, communication rivals compute.** The all-to-all is comparable to the expert GEMM at moderate batch and dominates at small batch. Overlap it, use FP8 dispatch, and use low-latency kernels (DeepEP) — the collective library is a top-three throughput lever, not a detail.
- **Load imbalance is the straggler tax.** All-to-all is a barrier, so the layer runs at the speed of the hottest GPU. Utilization is ~${1/\rho}$; a ${\rho = 2.4}$ hot expert wastes more than half your fleet. Monitor per-expert token counts and the imbalance factor.
- **EPLB and redundant experts fix imbalance.** Replicate hot experts across GPUs to drive ${\rho}$ toward ~1.15 and utilization toward ~88%. Recompute placement from live telemetry on a slow cadence.
- **Prefill and decode want different EP degrees.** Prefill is compute-bound (narrow EP); decode is latency/memory-bound (wide EP). Disaggregate the two pools at scale — DeepSeek runs EP32 prefill and EP144 decode.
- **Mix parallelism by component.** Attention runs DP (or TP) with local KV; the FFN runs EP. The two all-to-alls exist only because those layouts disagree.
- **Launch it with `--enable-expert-parallel` (vLLM) or `--ep-size` (SGLang).** EP degree is the product of the parallel sizes; sweep it and find the throughput/TPOT knee for your traffic.
- **MoE wins at high scale with fast interconnect and enough HBM — and loses at low scale, on slow networks, or under tight memory.** The FLOP savings are real only when you can keep the many GPUs busy.

## Further reading

- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) — the architecture: MLA, 256 experts + shared, top-8 routing, auxiliary-loss-free load balancing, FP8 training.
- DeepSeek, "DeepSeek-V3/R1 Inference System Overview" (2025) — the production serving disclosure: PD disaggregation, EP32 prefill / EP144 decode, per-node throughput and cost-profit figures.
- DeepSeek, "DeepEP" (open-source repository, 2025) — the expert-parallel all-to-all kernels: normal (high-throughput) and low-latency (pure-RDMA) modes, FP8 dispatch, hook-based overlap.
- DeepSeek, "EPLB: Expert Parallelism Load Balancer" (open-source repository, 2025) — redundant-expert placement, hierarchical and global balancing policies.
- Jiang et al. / Mistral AI, "Mixtral of Experts" (2024) — the sparse MoE that most teams serve: 8 experts, top-2 routing, 46.7B total / 12.9B active.
- SGLang team, "Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism" (2025) — open reproduction on 96 H100 GPUs with DeepEP, EPLB, and two-batch overlap.
- vLLM documentation — expert parallelism (`--enable-expert-parallel`), data-parallel attention, and DeepSeek serving guides.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving), [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different), [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving), and [prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation).

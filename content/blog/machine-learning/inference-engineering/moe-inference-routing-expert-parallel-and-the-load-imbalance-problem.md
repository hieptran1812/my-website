---
title: "MoE inference: routing, expert parallelism, and the load-imbalance problem"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Build the piece of nanoserve that makes Mixture-of-Experts run: a top-k router, an expert-parallel layer with all-to-all dispatch and combine, an EPLB replica for the hot expert, and the instrumentation that shows you why one straggler stalls the whole GPU cluster."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "mixture-of-experts",
    "expert-parallelism",
    "distributed-inference",
    "pytorch",
    "gpu",
    "ml-systems",
    "throughput",
    "latency",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 43
---

Someone hands you a Qwen3-30B-A3B checkpoint and says it is "a 3B model in disguise" — only three billion parameters fire per token, so it should decode about as fast as a small model while thinking like a big one. You try to load it on your RTX 4090 and it will not even fit: thirty billion weights in bf16 is sixty gigabytes, and your card has twenty-four. You reach for an A100 80GB, where it does fit, and at batch 1 it decodes at a speed that is nowhere near "3B fast." Then you put it under real load — dozens of concurrent requests — and suddenly it is fast, throughput climbs past what a dense model of the same active size would give you, and it keeps climbing as you add GPUs. Nothing about that story is intuitive until you understand what a Mixture-of-Experts (MoE) actually does to the inference math.

That is what this post is about. MoE changes the fundamental trade of inference. A dense model reads its entire weight matrix to produce every token — the whole thing, once per token, streamed through the arithmetic units. An MoE keeps *all* of its weights resident in memory but only reads and computes a small **active** slice per token: the attention layers, a shared expert if there is one, and the handful of "routed" experts a little gating network picks for that specific token. So the compute is small-model-sized while the memory footprint is huge-model-sized. MoE trades memory for FLOPs. On one GPU at batch 1 that trade can be a *pessimization* — you hold sixty gigabytes of weights to do three billion parameters' worth of work, and you are bandwidth-bound reading whichever experts happened to get routed. Across a cluster at high batch, the same trade becomes the reason DeepSeek-V3 serves at thousands of tokens per second per GPU.

![Side-by-side contrast of a dense model that reads and computes every weight per token against an MoE that keeps all weights resident but reads only the active slice](/imgs/blogs/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem-1.webp)

Between those two regimes sits the hardest engineering problem in this series so far: **routing tokens to experts that live on different GPUs**. When you shard the experts across a cluster — **expert parallelism** — a token routed to a remote expert has to be physically *sent* there, computed, and sent *back*. That is an all-to-all collective, twice per MoE layer, and it is where the microseconds go. Worse, routing is data-dependent: some experts are "hot" and draw a flood of tokens while others sit nearly idle, so the GPU holding the hot expert becomes a straggler that stalls everyone waiting at the combine barrier. By the end of this post you will have written `nanoserve/layers/moe.py`: a real top-k router, an expert-parallel MoE layer built on `torch.distributed.all_to_all_single`, a simple **EPLB** replica that spreads the hottest expert across two GPUs, and per-expert load instrumentation that makes the imbalance visible. You will be able to derive the all-to-all comm volume per token, the imbalance stall, and the exact reason "3B active" does not mean "3B fast."

One standing promise from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or an official post with a link, or framed as something you will reproduce yourself with a named script and an expected range. Results tables carry a `Source` column. The memory and comm arithmetic is division, so it is honest to derive; end-to-end throughput stays cited or reproduce-it-yourself. This post leans hard on the vLLM team's production write-ups on large-scale MoE serving — they are the benchmark target, and I will name and link every one.

---

## 1. The premise: active versus total parameters

Start with the one number that defines a MoE: the ratio of active to total parameters. The vLLM team, in their [Expert Parallelism at Scale post](https://vllm.ai/blog/2025-12-17-large-scale-serving) (2025-12-17), describe serving **DeepSeek-V3, which activates 37B of its 671B parameters per forward pass**. That is a 5.5% activation ratio: for every token, 94.5% of the model's weights do nothing at all. Qwen3-Next, per the vLLM [Qwen3-Next post](https://vllm.ai/blog/2025-09-11-qwen3-next) (2025-09-11), is even more extreme — an 80B-A3B model with an activation ratio the team quotes as "1:50," so only 3B of 80B fire per token. MiniMax-M1, from the vLLM [MiniMax-M1 post](https://vllm.ai/blog/2025-06-30-minimax-m1) (2025-06-30), is a 456B model with roughly 45.9B active, about 10%.

Now derive what that means for inference, because two different costs move in opposite directions.

**Compute (FLOPs) scales with active parameters.** A forward pass does a fixed amount of arithmetic per active weight. DeepSeek-V3 does 37B-worth of matmul per token, not 671B-worth. In FLOPs it is a 37B model. This is the whole selling point: you get the quality of a 671B parameter count for the compute bill of a 37B model.

**Memory scales with total parameters.** Every expert has to be resident in GPU memory, because you do not know until the router fires *which* experts this token will need — and the next token will need different ones. You cannot page a 671B model down to 37B of resident weights; you must hold all 671B. In bytes it is a 671B model. In fp8 that is 671 GB of weights; in bf16 it is 1.34 TB. No single GPU holds that. This is why big MoEs are inherently multi-GPU: the *memory* forces you to shard even when the *compute* would fit on far fewer devices.

That split — small compute, large memory — is the entire character of MoE inference. Hold onto it; every design decision below is a consequence.

### The effective weight bytes per token

At batch 1, decode time is set by bytes moved over bandwidth — the same law that governs every decode step, derived at length in [the skinny-matrix GEMM post](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem):

$$
t_{\text{decode}} \approx \frac{\text{bytes read per token}}{\text{HBM bandwidth}}
$$

For a dense model the bytes read per token is the whole model. For a MoE it is only the **active** slice: the non-expert weights (attention, embeddings, norms, and a shared expert if present) plus the top-k routed experts this token selected:

$$
b_{\text{token}} = b_{\text{shared}} + k \cdot b_{\text{expert}}
$$

where $b_{\text{expert}}$ is the byte size of one expert's feed-forward weights and $k$ is the number of experts the router activates. This is the quantity that decides your batch-1 decode speed, and it is far smaller than the resident footprint. The gap between $b_{\text{token}}$ and the resident bytes is exactly the memory you are "wasting" to buy the FLOP savings.

#### Worked example: why Qwen3-30B-A3B is slow at batch 1 and does not fit a 4090

Take a 30B-A3B-style model: 128 routed experts, top-8 routing, roughly 0.22B parameters per expert, and about 2B parameters of shared weights (attention plus a shared expert). In fp8 (one byte per weight) each expert is 0.22 GB and the shared part is 2.0 GB.

- **Batch-1 bytes read per token:** $b_{\text{token}} = 2.0 + 8 \times 0.22 = 3.76 \text{ GB}$.
- **On an A100 80GB SXM at 2039 GB/s** ([NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)): the floor is $3.76 / 2039 = 1.84$ ms, or about 540 tok/s ceiling. Overheads pull the real number well below that.
- **Resident footprint:** all 128 experts plus shared weights, $2.0 + 128 \times 0.22 = 30.2$ GB in fp8, or 60 GB in bf16.

Two things fall out. First, the batch-1 decode reads 3.76 GB but you are *holding* 30.2 GB in fp8 — you carry eight times the weights you touch. Second, 30 GB in fp8 (and 60 GB in bf16) does not fit a 24 GB RTX 4090 at all, which is why "3B active" is meaningless on consumer hardware: the number that decides whether it *runs* is the total, and the total is 30B. We build the full experiment around this fact in the Track H post on [running a MoE model on consumer hardware](/blog/machine-learning/inference-engineering/experiment-a-moe-model-on-consumer-hardware) later in the series; for now the point is that active-parameter marketing measures FLOPs while your VRAM budget measures total parameters.

| Quantity | Dense 30B | MoE 30B-A3B | Source |
| --- | --- | --- | --- |
| Total params | 30B | 30B | model definition |
| Active per token | 30B | ~3B | derived (2.0 GB + 8×0.22 GB fp8) |
| Resident bytes (fp8) | 30 GB | 30 GB | derived |
| Bytes read per token (fp8) | 30 GB | 3.76 GB | derived |
| Batch-1 ceiling on A100 | ~68 tok/s | ~540 tok/s | derived (bytes ÷ 2039 GB/s) |
| DeepSeek-V3 activation | — | 37B of 671B | cited: [vLLM large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving) |

The MoE's batch-1 ceiling is roughly eight times higher than the dense model of the same *total* size — that is the FLOP-and-bandwidth win. But it comes with a footprint identical to the dense model, and that footprint is what stops it from fitting a smaller GPU. Trading memory for FLOPs is only a win when you have the memory to spend.

---

## 2. Routing: top-k gating and why the load is unpredictable

Every MoE layer begins with a **router** (also called the gate): a tiny linear projection from the hidden dimension to the number of experts, followed by a softmax and a top-k selection. For each token it produces a score over all $E$ experts, keeps the $k$ highest, renormalizes those $k$ weights, and sends the token only to those experts. The router is the whole reason the load is data-dependent — it is a learned function of the token's hidden state, so which experts get picked depends on the *content*, and you cannot predict it before the forward pass runs.

![Dataflow of a single token through the router, branching to a local expert and a remote expert across the dispatch, then merging back through the combine into one weighted output](/imgs/blogs/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem-2.webp)

Here is the router as it lands in `nanoserve/layers/moe.py`. It is small, exact, and runs on CPU so you can play with it immediately:

```python
# nanoserve/layers/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """Top-k gate: hidden state -> which experts, and with what weight."""
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor):
        # x: [num_tokens, d_model]
        logits = self.gate(x)                      # [T, E]
        probs = F.softmax(logits, dim=-1)          # [T, E]
        weights, experts = torch.topk(probs, self.top_k, dim=-1)  # [T, k], [T, k]
        weights = weights / weights.sum(dim=-1, keepdim=True)      # renormalize
        return experts, weights                    # ids in [0, E), weights sum to 1
```

Run it on a batch and look at what comes out:

```python
torch.manual_seed(0)
T, d_model, E, k = 6, 512, 8, 2
router = Router(d_model, E, k)
x = torch.randn(T, d_model)
experts, weights = router(x)
print(experts)   # e.g. tensor([[3, 5], [3, 1], [6, 3], [3, 0], [2, 3], [3, 7]])
print(weights.sum(dim=-1))  # all 1.0
```

Two things are worth staring at in that `experts` tensor. First, expert 3 shows up in almost every row — with random weights and a random input, some experts are simply more likely to win, and a *trained* router concentrates far harder because it has learned that certain experts specialize. Second, there is no structure you can exploit ahead of time: token 0 goes to experts 3 and 5, token 1 to 3 and 1, and the next layer's router will scatter them completely differently. Routing is per-token, per-layer, and content-dependent. That unpredictability is the source of every hard problem in the rest of this post.

### The dense reference: gather, compute, scatter

Before parallelizing anything, write the single-GPU MoE so you have a correctness oracle. The trick is that you do *not* run every token through every expert — that would defeat the entire purpose and cost you the full dense FLOPs. Instead you **gather** the tokens routed to each expert, run that expert's feed-forward on just those tokens, and **scatter** the results back, weighted by the gate:

```python
class Expert(nn.Module):
    """One expert = a SwiGLU FFN."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class DenseMoE(nn.Module):
    """Single-GPU reference: gather per expert, compute, scatter back."""
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.num_experts = num_experts

    def forward(self, x):                         # x: [T, d_model]
        experts, weights = self.router(x)         # [T, k], [T, k]
        out = torch.zeros_like(x)
        # Flatten (token, slot) pairs so each routed assignment is one row.
        flat_tok = torch.arange(x.size(0)).repeat_interleave(experts.size(1))
        flat_exp = experts.reshape(-1)
        flat_w   = weights.reshape(-1)
        for e in range(self.num_experts):
            mask = flat_exp == e                  # which assignments hit expert e
            if not mask.any():
                continue
            rows = flat_tok[mask]                 # tokens routed to e
            y = self.experts[e](x[rows])          # run expert e once, on its tokens
            out.index_add_(0, rows, y * flat_w[mask].unsqueeze(-1))
        return out
```

The `for e in range(num_experts)` loop is the shape of every MoE kernel: it is a **grouped GEMM**, a batch of matmuls where each expert's matmul has a different, data-dependent number of rows. On one GPU the cost of this layer is proportional to the *total* tokens times $k$ (each token does $k$ experts' worth of FFN), not to the number of experts — that is the FLOP saving, made concrete. The [grouped-GEMM MoE kernel post](/blog/machine-learning/model-serving/vllm-deep-dive) in the model-serving series goes deeper on how a real kernel fuses that loop; here the loop is the reference we will check the parallel version against.

---

## 3. Batching helps a MoE differently than a dense model

For a dense model, batching is almost free on the memory side: whether you decode 1 token or 64, you read the weight matrix once and reuse it across the whole batch. The bytes-per-token fall like $1/B$ until you hit the compute roofline. A MoE breaks that clean story, because different tokens in the batch route to *different* experts. Add a token and you might pull in an expert nobody else in the batch touched — a fresh weight matrix off HBM.

The quantity that governs this is the number of **distinct experts touched** by a batch. If $B$ tokens each independently pick $k$ of $E$ experts uniformly, the expected number of distinct experts that receive at least one token is:

$$
E_{\text{touched}}(B) = E \left( 1 - \left( 1 - \tfrac{k}{E} \right)^{B} \right)
$$

This is the coupon-collector shape: at small $B$ you touch few experts, and as $B$ grows you saturate toward touching *all* of them. Real routers are skewed rather than uniform, so the true curve saturates even faster, but the uniform bound is the clean version to reason with. The bytes read per step is the touched experts' weights plus the shared weights; the bytes per *token* is that divided by $B$:

$$
b_{\text{token}}(B) = \frac{b_{\text{shared}} + E_{\text{touched}}(B) \cdot b_{\text{expert}}}{B}
$$

![Table of distinct experts touched, bytes per step, bytes per token, and regime at three batch sizes showing per-token bytes collapsing as batch grows](/imgs/blogs/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem-3.webp)

Plug the 30B-A3B numbers ($E=128$, $k=8$, $b_{\text{expert}}=0.22$ GB, $b_{\text{shared}}=2.0$ GB) into that formula at three batch sizes:

- **Batch 1:** $E_{\text{touched}} = 8$. Bytes/step $= 2.0 + 8 \times 0.22 = 3.8$ GB. Bytes/token $= 3.8$ GB.
- **Batch 32:** $E_{\text{touched}} = 128(1 - 0.9375^{32}) \approx 112$. Bytes/step $= 2.0 + 112 \times 0.22 = 26.6$ GB. Bytes/token $= 0.83$ GB.
- **Batch 256:** $E_{\text{touched}} \approx 128$ (all). Bytes/step $= 2.0 + 128 \times 0.22 = 30.2$ GB. Bytes/token $= 0.12$ GB.

The per-token bytes fall from 3.8 GB to 0.12 GB — a 32× drop — as the resident expert weights get *amortized* across more and more tokens. This is the deep reason MoE serving is a high-batch game: at low batch you pay to read a few experts to serve a few tokens; at high batch you read the whole model once and spread it across hundreds of tokens, and the memory you "wasted" holding all those experts finally earns its keep. It is also why the vLLM team's 2.2k tok/s per H200 figure (next section) is an *aggregate* number under heavy concurrency, not something you would ever see at batch 1.

Here is the same curve as a script you can run on CPU to build the intuition — no GPU, no claim about wall-clock, just the counting:

```python
# nanoserve/tools/expert_working_set.py — how many distinct experts a batch touches
import torch

def distinct_experts(B, E=128, k=8, trials=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = []
    for _ in range(trials):
        # each of B tokens picks k of E experts uniformly (no replacement per token)
        picks = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(B)])
        counts.append(picks.unique().numel())
    return sum(counts) / len(counts)

for B in (1, 8, 32, 128, 256):
    touched = distinct_experts(B)
    frac = touched / 128
    print(f"B={B:>3}  distinct experts ~ {touched:5.1f}  ({frac:4.0%} of the model resident-touched)")
# B=  1  distinct experts ~   8.0  (  6% ...)
# B= 32  distinct experts ~ 112.x  ( 88% ...)
# B=256  distinct experts ~ 128.0  (100% ...)
```

The formula and the simulation agree because they are counting the same thing. The lesson is a serving lesson: **a MoE that looks bandwidth-starved at batch 1 becomes bandwidth-efficient exactly when you give it enough concurrent traffic to touch every expert per step.** If your workload cannot supply that batch — a single interactive user, say — a MoE is the wrong tool, and a dense model of the *active* size will beat it on both latency and memory.

---

## 4. Expert parallelism: the all-to-all

Once the model is too big for one GPU — and any real MoE is — you have to shard the experts across devices. The natural way is **expert parallelism (EP)**: give each GPU a disjoint subset of experts. With 128 experts across 8 GPUs, each GPU holds 16 experts. The router still runs on every GPU for its local tokens, but now a token routed to an expert on *another* GPU has to physically travel there. That is the all-to-all.

Concretely, one MoE layer under EP is a five-step dance:

1. **Route.** Each GPU runs the router on its local tokens and gets, per token, the $k$ expert ids.
2. **Permute and dispatch.** Group each GPU's tokens by which GPU owns their target experts, then run an **all-to-all**: every GPU sends each other GPU the tokens destined for its experts. After this collective, each GPU holds exactly the tokens its local experts must process.
3. **Expert GEMM.** Each GPU runs its local experts (the grouped GEMM from Section 2) on the tokens it received.
4. **Combine.** A second **all-to-all** sends every result back to the GPU that owns the original token.
5. **Weighted sum.** Each GPU merges its tokens' $k$ expert outputs using the gate weights.

The two all-to-alls are the cost. Let me derive the communication volume, because it is what decides whether EP scales.

### The comm volume, derived

Each token's hidden vector has dimension $d_{\text{model}}$. In the dispatch, a token is sent to whichever GPUs hold its $k$ experts — worst case, $k$ distinct GPUs, so the token's activation is sent $k$ times. In the combine, $k$ expert outputs (each $d_{\text{model}}$ wide) come back. So per token, per MoE layer, the all-to-all moves:

$$
V_{\text{token}} = \underbrace{k \cdot d_{\text{model}} \cdot s}_{\text{dispatch}} + \underbrace{k \cdot d_{\text{model}} \cdot s}_{\text{combine}} = 2 k \, d_{\text{model}} \, s
$$

where $s$ is the bytes per activation element. This is the number that grows with the model and with $k$, and it is *per layer* — you pay it at every MoE layer in the stack.

![Stacked byte budget of one expert-parallel decode step showing active weights dominating HBM traffic and the per-layer all-to-all payload becoming the wall at high expert-parallel degree](/imgs/blogs/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem-4.webp)

#### Worked example: the all-to-all bill for DeepSeek-V3

DeepSeek-V3's configuration (per its [model card](https://huggingface.co/deepseek-ai/DeepSeek-V3)) uses a hidden size of 7168, top-8 routing, and 58 MoE layers. In bf16 activations ($s = 2$):

- **Per token, per layer:** $2 \times 8 \times 7168 \times 2 = 229{,}376$ bytes $= 224$ KiB.
- **Per token, across 58 MoE layers:** $58 \times 224 \text{ KiB} = 12.7$ MiB.

Nearly thirteen mebibytes of network traffic for a *single decode token*. On an InfiniBand fabric at, say, 50 GB/s of usable bandwidth, that is $12.7 \text{ MiB} / 50 \text{ GB/s} \approx 0.27$ ms of pure transfer per token — before you count the latency of dozens of small messages and the synchronization. This is why the vLLM team notes, in their [large-scale serving post](https://vllm.ai/blog/2025-12-17-large-scale-serving), that "higher EP degree increases inter-rank sync overhead": as you spread experts across more GPUs, the all-to-all touches more peers with smaller per-peer messages, and the collective gets latency-bound. The all-to-all, not HBM, becomes the wall.

Two production techniques attack this volume head-on:

**Faster all-to-all kernels.** The vLLM team's Wide-EP design (their term for expert parallelism combined with data parallelism, experts shared across ranks with token routing) uses specialized all-to-all backends — they name **DeepEP kernels**, **Perplexity MoE kernels**, and NCCL — rather than a naive collective. With these, plus prefill/decode disaggregation and DeepGEMM, they report **2.2k tok/s per H200 GPU for DeepSeek-V3 on a CoreWeave H200 cluster with ConnectX-7 InfiniBand, up from a roughly 1.5k baseline** ([vLLM large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving), 2025-12-17).

**Quantize the payload.** If the all-to-all volume is $2 k \, d_{\text{model}} \, s$, then shrinking $s$ shrinks the bill linearly. On Blackwell, the vLLM team quantizes activations to FP4 *before* the dispatch: their [DeepSeek-R1 on GB200 post](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) (2026-02-03) reports that **NVFP4 dispatch quantizes activations to FP4 before the all-to-all, giving 4× less communication versus FP16**. Drop $s$ from 2 bytes to 0.5 bytes and the 224 KiB/layer becomes 56 KiB/layer. The same post pairs this with GB200 NVL72's 8 TB/s of bandwidth (versus 4.8 TB/s on H200) and reports a generational 3–5× GPU-count reduction for the same DeepSeek-R1 deployment (H200 prefill 16→8 GPUs, decode 32→8 GPUs, at 2K in / 2K out).

### The expert-parallel layer in nanoserve

Here is the EP layer built on `torch.distributed.all_to_all_single`. It is real, runnable code — launch it with `torchrun --nproc_per_node=N`. The heart of it is computing the per-rank split sizes, because each rank sends a *different* number of tokens to each peer (that is the whole imbalance problem, encoded as unequal splits):

```python
# nanoserve/layers/moe_ep.py — expert-parallel MoE with all-to-all dispatch/combine
import torch
import torch.distributed as dist
import torch.nn.functional as F

def all_to_all_tokens(x, out_splits, in_splits, group):
    """Variable-length all-to-all of token rows."""
    out = x.new_empty(sum(out_splits), x.size(1))
    dist.all_to_all_single(out, x, out_splits, in_splits, group=group)
    return out

class ExpertParallelMoE(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, group):
        super().__init__()
        self.router = Router(d_model, num_experts, top_k)
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.E = num_experts
        self.per_rank = num_experts // self.world       # experts this rank owns
        # only this rank's slice of experts is instantiated here
        self.local_experts = torch.nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(self.per_rank)])
        self.group = group

    def forward(self, x):                                # x: local tokens [T, d]
        experts, weights = self.router(x)                # [T, k]
        T, k = experts.shape
        # Flatten to one row per (token, chosen-expert) assignment.
        tok_idx = torch.arange(T, device=x.device).repeat_interleave(k)
        exp_idx = experts.reshape(-1)                    # [T*k]
        w_flat  = weights.reshape(-1)
        dst_rank = exp_idx // self.per_rank              # which GPU owns each expert

        # Sort assignments by destination rank so each rank's slice is contiguous.
        order = torch.argsort(dst_rank)
        tok_idx, exp_idx, w_flat, dst_rank = (t[order] for t in
                                              (tok_idx, exp_idx, w_flat, dst_rank))
        send_counts = torch.bincount(dst_rank, minlength=self.world)

        # Exchange counts so every rank knows how many rows it will receive.
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # DISPATCH: send the token activations to the ranks that own their experts.
        send_rows = x[tok_idx]                            # gather [sum(send), d]
        recv_rows = all_to_all_tokens(send_rows, recv_splits, send_splits, self.group)
        recv_exp  = all_to_all_ids(exp_idx, recv_splits, send_splits, self.group)

        # EXPERT GEMM: run each local expert on the rows it received.
        local_exp = recv_exp - self.rank * self.per_rank  # global id -> local id
        y = torch.empty_like(recv_rows)
        for le in range(self.per_rank):
            m = local_exp == le
            if m.any():
                y[m] = self.local_experts[le](recv_rows[m])

        # COMBINE: send the results back to the originating ranks.
        back = all_to_all_tokens(y, send_splits, recv_splits, self.group)

        # WEIGHTED SUM: scatter the k outputs of each token and reduce with gate weights.
        out = torch.zeros(T, x.size(1), device=x.device, dtype=x.dtype)
        out.index_add_(0, tok_idx, back * w_flat.unsqueeze(-1))
        return out
```

The `all_to_all_ids` helper is the same pattern as `all_to_all_tokens` but for a 1-D int tensor of expert ids. The shape to internalize is that **every arrow in this code is either a gather/scatter on-device or an all-to-all across the fabric**, and the all-to-alls carry the $2 k \, d_{\text{model}} \, s$ volume we derived. Notice there is nothing here that *balances* the load: `send_splits` can be wildly uneven, and if every token in the batch routes to experts on rank 3, then rank 3 receives everything and the others receive nothing. That is the next problem.

---

## 5. Load imbalance: hot experts and stragglers

Routing is skewed. A trained router learns that some experts specialize in common patterns — punctuation, common function words, a dominant language — and those experts get a disproportionate share of tokens. Call them **hot experts**. The consequence in an EP deployment is brutal: the GPU holding a hot expert receives far more tokens in the dispatch, does far more expert-GEMM work, and finishes late. And because the combine is a collective — every rank must arrive before any rank proceeds — the whole layer runs at the speed of the slowest rank. One hot expert stalls the entire cluster.

![Grid of per-expert token counts across two GPUs where one expert draws over two hundred tokens while a sibling draws nine, saturating one GPU](/imgs/blogs/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem-5.webp)

Quantify it with an **imbalance factor**. If tokens were perfectly balanced, each expert would receive $\bar{n} = Tk/E$ tokens. Define:

$$
\rho = \frac{\max_e n_e}{\bar{n}}
$$

the ratio of the hottest expert's load to the mean. A perfectly balanced layer has $\rho = 1$. Because the combine barrier waits for the slowest rank, and the slowest rank's expert-GEMM time scales with its token count, the layer's effective slowdown versus a balanced layer is approximately $\rho$ (more precisely, the ratio of the busiest *rank's* tokens to the mean rank's tokens, since a rank holds several experts). A layer that "should" take 1 ms takes $\rho$ ms. With $\rho = 3.5$, you have thrown away 71% of your expert compute to idle waiting.

Watch the animation below: the router keeps firing tokens at the same hot expert, its backlog grows while its siblings sit idle, and the combine — which cannot start until every expert is done — is held hostage by the one that is buried.

<figure class="blog-anim">
<svg viewBox="0 0 780 330" role="img" aria-label="Tokens stream from a router and are dispatched across four experts on two GPUs; most tokens land on one hot expert whose backlog pile grows while the other experts sit nearly idle" style="width:100%;height:auto;max-width:900px">
<title>The router sends most tokens to one hot expert, whose backlog grows while sibling experts stay nearly idle; the all-to-all combine cannot finish until the hot expert drains.</title>
<style>
.moe-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.moe-hot{fill:var(--surface,#f3f4f6);stroke:#f97316;stroke-width:2}
.moe-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.moe-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.moe-dot{fill:var(--accent,#6366f1)}
.moe-pilebar{fill:#f97316;opacity:.85}
@keyframes moe-f0{0%{transform:translate(0,0);opacity:0}8%{opacity:1}86%{opacity:1}100%{transform:translate(360px,-109px);opacity:0}}
@keyframes moe-f1{0%{transform:translate(0,0);opacity:0}8%{opacity:1}86%{opacity:1}100%{transform:translate(360px,-39px);opacity:0}}
@keyframes moe-f2{0%{transform:translate(0,0);opacity:0}8%{opacity:1}86%{opacity:1}100%{transform:translate(360px,31px);opacity:0}}
@keyframes moe-f3{0%{transform:translate(0,0);opacity:0}8%{opacity:1}86%{opacity:1}100%{transform:translate(360px,101px);opacity:0}}
@keyframes moe-pile{0%{transform:scaleY(.06)}100%{transform:scaleY(1)}}
.moe-d0{animation:moe-f0 4s linear infinite}
.moe-d0b{animation:moe-f0 4s linear infinite;animation-delay:1s}
.moe-d0c{animation:moe-f0 4s linear infinite;animation-delay:2s}
.moe-d0d{animation:moe-f0 4s linear infinite;animation-delay:3s}
.moe-d1{animation:moe-f1 4s linear infinite;animation-delay:1.4s}
.moe-d2{animation:moe-f2 4s linear infinite;animation-delay:2.7s}
.moe-d3{animation:moe-f3 4s linear infinite;animation-delay:0.6s}
.moe-grow{animation:moe-pile 8s ease-in-out infinite alternate;transform-box:fill-box;transform-origin:top}
@media (prefers-reduced-motion:reduce){.moe-d0,.moe-d0b,.moe-d0c,.moe-d0d,.moe-d1,.moe-d2,.moe-d3{animation:none;opacity:1}.moe-grow{animation:none;transform:scaleY(1)}}
</style>
<rect class="moe-box" x="40" y="130" width="140" height="90" rx="10"/>
<text class="moe-lbl" x="110" y="170">router</text>
<text class="moe-sub" x="110" y="192">top-8 of 128</text>
<rect class="moe-hot" x="520" y="40" width="200" height="52" rx="8"/>
<rect class="moe-pilebar moe-grow" x="528" y="48" width="60" height="36" rx="4"/>
<text class="moe-lbl" x="640" y="62">E0 · GPU0</text>
<text class="moe-sub" x="640" y="82">hot · backing up</text>
<rect class="moe-box" x="520" y="110" width="200" height="52" rx="8"/>
<text class="moe-lbl" x="620" y="132">E1 · GPU0</text>
<text class="moe-sub" x="620" y="152">idle</text>
<rect class="moe-box" x="520" y="180" width="200" height="52" rx="8"/>
<text class="moe-lbl" x="620" y="202">E5 · GPU1</text>
<text class="moe-sub" x="620" y="222">idle</text>
<rect class="moe-box" x="520" y="250" width="200" height="52" rx="8"/>
<text class="moe-lbl" x="620" y="272">E7 · GPU1</text>
<text class="moe-sub" x="620" y="292">idle</text>
<circle class="moe-dot moe-d0" cx="196" cy="175" r="7"/>
<circle class="moe-dot moe-d0b" cx="196" cy="175" r="7"/>
<circle class="moe-dot moe-d0c" cx="196" cy="175" r="7"/>
<circle class="moe-dot moe-d0d" cx="196" cy="175" r="7"/>
<circle class="moe-dot moe-d1" cx="196" cy="175" r="7"/>
<circle class="moe-dot moe-d2" cx="196" cy="175" r="7"/>
<circle class="moe-dot moe-d3" cx="196" cy="175" r="7"/>
</svg>
<figcaption>The router sends most tokens to one hot expert; its backlog pile grows while the sibling experts sit idle, and the combine cannot finish until the hot expert drains.</figcaption>
</figure>

### Instrumenting the imbalance

You cannot fix what you cannot see, and the imbalance is invisible unless you count. Add per-expert token counters to the router, aggregate them per step, and compute $\rho$. This is the single most useful piece of MoE observability, and it costs almost nothing:

```python
# nanoserve/layers/moe_instrument.py — per-expert load, per step
import torch

class LoadMonitor:
    def __init__(self, num_experts, per_rank):
        self.E = num_experts
        self.per_rank = per_rank
        self.reset()

    def reset(self):
        self.counts = torch.zeros(self.E, dtype=torch.long)

    def observe(self, expert_ids):                # flat [T*k] tensor of chosen experts
        self.counts += torch.bincount(expert_ids.cpu(), minlength=self.E)

    def report(self):
        c = self.counts.float()
        mean = c.mean().clamp(min=1)
        rho = (c.max() / mean).item()             # hottest expert vs mean
        # per-GPU load = sum over the experts each rank owns
        per_gpu = c.view(-1, self.per_rank).sum(dim=1)
        gpu_rho = (per_gpu.max() / per_gpu.float().mean()).item()
        hot = int(c.argmax())
        return dict(rho=round(rho, 2), gpu_rho=round(gpu_rho, 2),
                    hot_expert=hot, hot_tokens=int(c.max()),
                    idle_experts=int((c == 0).sum()))
```

Run it against a deliberately skewed routing distribution — a Zipf draw, which is a realistic model for a trained router — and you see the problem in numbers:

```python
mon = LoadMonitor(num_experts=8, per_rank=4)     # 2 GPUs, 4 experts each
# Zipf-ish: expert 0 is hot, tail is cold
probs = torch.tensor([210, 44, 38, 51, 47, 9, 41, 40], dtype=torch.float)
draws = torch.multinomial(probs, 480, replacement=True)
mon.observe(draws)
print(mon.report())
# {'rho': 3.5, 'gpu_rho': 2.5, 'hot_expert': 0, 'hot_tokens': ~210,
#  'idle_experts': 0}
```

An $\rho$ of 3.5 and a per-GPU imbalance of 2.5 means GPU 0 does two-and-a-half times the work of GPU 1 and the layer runs at GPU 0's pace. In a 58-layer model, this tax compounds at every MoE layer. The [distributed inference post](/blog/machine-learning/inference-engineering/tensor-parallel-inference-by-hand) sibling covers the analogous all-reduce cost in tensor parallelism; the difference is that TP's collective is *symmetric* — every rank does the same work — while EP's all-to-all is inherently *asymmetric*, and asymmetry is what stalls.

![Timeline of one expert-parallel step where the combine barrier cannot complete until the rank holding the hot expert finishes, leaving the cold ranks idle until the straggler catches up](/imgs/blogs/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem-6.webp)

The all-to-all combine is a barrier: no rank proceeds until every rank has finished its experts and exchanged results. The rank holding the hot expert is the last to arrive, so the cold ranks sit idle from the moment they finish until the straggler catches up, and the step runs at the slowest rank's pace.

---

## 6. EPLB: replicate the hot expert, and overlap the comms

There are two independent levers against the straggler, and production MoE stacks pull both.

### Redundant experts (EPLB)

The fix for a hot expert is embarrassingly direct: make more than one copy of it. **Expert-Parallel Load Balancing (EPLB)** places *redundant* replicas of the hottest experts on additional GPUs and splits their token stream across the copies. Two copies of an expert that was drawing 210 tokens now draw 105 each; the per-GPU imbalance shrinks toward 1; the barrier stall shrinks with it. The vLLM team exposes this as **`--enable-eplb`** in their large-scale serving stack ([vLLM large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving)).

![Before-and-after of the hot expert: one saturated copy on a single GPU versus two EPLB replicas that halve the per-copy load and shrink the imbalance factor](/imgs/blogs/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem-7.webp)

The subtle part is that EPLB does *not* change the router — the model still routes to logical expert 0 — it changes the *mapping* from logical experts to physical GPUs, adding a level of indirection where a hot logical expert points to several physical replicas and the dispatcher round-robins among them. Here is a minimal version in `nanoserve` that replicates the single hottest expert onto a spare slot and splits its assignments:

```python
# nanoserve/layers/eplb.py — replicate the hottest expert, split its load
import torch

def build_placement(counts, world, per_rank, replicas=1):
    """Map logical experts -> list of physical (rank, local_slot) locations.
    The `replicas` hottest experts each get one extra physical copy."""
    E = counts.numel()
    placement = {e: [(e // per_rank, e % per_rank)] for e in range(E)}
    hot = counts.argsort(descending=True)[:replicas].tolist()
    spare_rank = 0                                   # in practice: least-loaded rank
    for e in hot:
        placement[e].append((spare_rank, per_rank))  # extra slot appended on a GPU
    return placement, set(hot)

def split_hot_assignments(dst_rank, exp_idx, placement, hot):
    """Round-robin the hot expert's tokens across its physical replicas."""
    for e in hot:
        rows = (exp_idx == e).nonzero(as_tuple=True)[0]
        half = rows[1::2]                            # send every other token to the replica
        replica_rank = placement[e][1][0]
        dst_rank[half] = replica_rank
    return dst_rank
```

Re-run the monitor after the split and the imbalance drops — the hot expert's 210 tokens become two streams of ~105, GPU 0's load falls, and $\rho$ moves from 3.5 toward 1.9. The cost is memory: each replica is a full extra copy of that expert's weights, so EPLB spends VRAM to buy latency. That trade is worth it precisely when a few experts are *persistently* hot — which, for a trained router on a stable workload, they usually are. vLLM's Elastic EP work (next section) even reshuffles the EPLB placement live as the traffic pattern drifts.

### Dual-batch overlap (DBO)

The second lever hides the all-to-all instead of shrinking it. The dispatch and combine are communication; the expert GEMM is computation; naively they run in series, so the GPU sits idle during every transfer. **Dual-batch overlap (DBO)** splits the batch into two halves and pipelines them: while half A is in the all-to-all, half B is in the expert GEMM, and vice versa, so comms and compute overlap and the transfer time is largely hidden behind useful work. The vLLM team exposes this as **`--enable-dbo`** ([vLLM large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving)). It is the same idea as double-buffering a memory transfer, applied at the batch level to a collective. EPLB attacks the *height* of the straggler; DBO attacks the *exposure* of the comms. In production you want both.

| Technique | What it fixes | vLLM flag | Cost | Source |
| --- | --- | --- | --- | --- |
| EPLB replicas | Hot-expert straggler ($\rho \gg 1$) | `--enable-eplb` | Extra VRAM per replica | cited: [large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving) |
| DBO overlap | Exposed all-to-all latency | `--enable-dbo` | Two-way batch split | cited: [large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving) |
| NVFP4 dispatch | All-to-all volume | (Blackwell path) | FP4 activation error | cited: [DeepSeek-R1 GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) |
| Faster kernels | Collective throughput | DeepEP / Perplexity | Backend complexity | cited: [large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving) |

---

## 7. Elastic Expert Parallelism: resizing the group without a restart

Traffic is not constant. A MoE deployment that is right-sized for the evening peak is over-provisioned at 4 a.m., and every idle GPU is money burned. The natural fix — change how many GPUs the expert-parallel group spans — normally means tearing down and rebuilding the whole engine, which drops in-flight requests and costs tens of seconds of cold start. The vLLM team's [Elastic Expert Parallelism post](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism) (2026-05-14) describes doing it *live*.

I cite this one as **mechanism only**: the post explicitly states there are **no performance numbers** (the demo runs DeepSeek-V2-Lite-Chat), so I will not attach any throughput or latency claim to it. What is citable is the shape of the operation. Changing the data-parallel size resizes the EP group through six stages, in the team's description:

1. **Drain** — stop admitting new requests and let in-flight ones finish (`VLLM_ELASTIC_EP_DRAIN_REQUESTS=1`).
2. **Init** — bring up the new engine-core processes for the resized world.
3. **Standby comms** — build the new communication groups on the side (a `StatelessGroupCoordinator`) without disturbing the running ones.
4. **Weight broadcast** — replicate the non-expert weights to any newly added ranks.
5. **The switch** — release the old CUDA graphs and promote the standby communication groups to active.
6. **EPLB reshuffle** — rebalance expert placement for the new group size.

The API is a single call, **`POST /scale_elastic_ep {"new_data_parallel_size": 8}`**, and the feature is gated behind **`--enable-elastic-ep --enable-expert-parallel --enable-eplb --data-parallel-backend ray`** with an all-to-all backend chosen via **`--all2all-backend {allgather_reducescatter | nixl_ep}`**. The NIXL EP backend is the interesting building block: it supports *incremental* rank join and leave (`connect_ranks` / `disconnect_ranks`) with a two-stage barrier that prevents the deadlock you would otherwise hit when some ranks think the group is one size and others think it is another. That incremental join/leave is also the foundation for fault tolerance — a dead rank can be dropped and a replacement added without a full restart.

The limitations the team names are the honest part to carry forward: **tensor_parallel_size must be 1, there is a single API server, the Ray data-parallel backend is required, and DBO is not supported** in this mode. So Elastic EP today is a mechanism for *scaling the expert-parallel group*, not a general elasticity story, and you would combine it with the tensor-parallel work from the [tensor-parallel-by-hand](/blog/machine-learning/inference-engineering/tensor-parallel-inference-by-hand) sibling only once those constraints lift. It is on this list because the *mechanism* — drain, build on the side, switch atomically, rebalance — is exactly how you would design live resizing yourself.

---

## 8. Expert offload: when the experts do not fit at all

Sometimes even the sharded footprint does not fit the GPUs you have. The MoE property that makes this survivable is the same one that made batch-1 slow: at any moment, only a few experts are active, so the *cold* experts can live in cheaper, larger memory and be streamed in on demand. **Expert offload** keeps cold experts in CPU/host DRAM (or NVMe) and copies the needed ones across PCIe into VRAM just before their GEMM. It is the same offload machinery we built for the KV cache in the [eviction and KV-swapping post](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) — a tiered memory hierarchy with GPU HBM as the fast tier and host DRAM as the slow, large tier — applied to weights instead of cache blocks.

Whether it pays is a bandwidth question, and the numbers are unforgiving. Derive the PCIe cost.

#### Worked example: the PCIe bill for streaming an expert

One expert's FFN in the 30B-A3B model is about 0.22B parameters — in fp8, 0.22 GB. PCIe 4.0 x16 delivers about 32 GB/s peak, and realistically 25–28 GB/s. Streaming one expert costs:

$$
t_{\text{stream}} = \frac{0.22 \text{ GB}}{28 \text{ GB/s}} \approx 7.9 \text{ ms}
$$

Compare that to the compute: running that expert on a handful of tokens is *microseconds*. So a single cold-expert fetch costs a thousand times its own compute. At batch 1, top-8 routing, if all 8 chosen experts are cold, you pay $8 \times 7.9 = 63$ ms of pure PCIe transfer to produce one token — catastrophic. Expert offload only pays when the **reuse count** is high: the streamed expert must serve many tokens before it is evicted. Define the break-even as the number of tokens $R$ over which a fetch amortizes below the HBM-resident cost. If the resident expert would contribute, say, 0.1 ms of HBM read per step, then the fetch amortizes once:

$$
\frac{t_{\text{stream}}}{R} \lt t_{\text{HBM per step}} \;\;\Rightarrow\;\; R \gt \frac{7.9 \text{ ms}}{0.1 \text{ ms}} = 79
$$

You need roughly 79 tokens to hit the same expert before its next eviction for offload to break even against keeping it resident — which is another way of saying offload wants **high batch and a stable hot set**, exactly the regime where a few experts are persistently hot. Offload the *cold tail*, keep the hot experts resident, and you win; try to stream the hot experts and you thrash. This is why the vLLM team's KV-offload work ([OffloadingConnector post](https://vllm.ai/blog/2026-01-08-kv-offloading-connector), 2026-01-08) leans so hard on *asynchrony* — overlapping the copy with compute so the transfer does not sit on the critical path — and the same discipline applies to expert offload: prefetch the likely-needed experts on a side stream while the current layer computes.

A minimal offload manager makes the tiering explicit. It is an LRU over experts with a resident budget:

```python
# nanoserve/layers/expert_offload.py — keep hot experts resident, stream the cold tail
from collections import OrderedDict

class ExpertCache:
    def __init__(self, all_experts_cpu, budget, device):
        self.cpu = all_experts_cpu          # list of state_dicts in pinned host DRAM
        self.budget = budget                # how many experts fit in VRAM
        self.device = device
        self.resident = OrderedDict()       # expert_id -> module on GPU (LRU)

    def get(self, e):
        if e in self.resident:
            self.resident.move_to_end(e)    # mark recently used
            return self.resident[e]
        if len(self.resident) >= self.budget:
            old, mod = self.resident.popitem(last=False)   # evict least-recent
            del mod                                          # free VRAM
        mod = load_expert(self.cpu[e]).to(self.device, non_blocking=True)  # PCIe copy
        self.resident[e] = mod
        return mod
```

The `non_blocking=True` copy from pinned memory is the only thing that lets you overlap the transfer with the previous layer's compute; without pinned host memory and a side stream, every fetch is a synchronous stall. Even then, the break-even math above is the governor: offload is a capacity technique, not a speed technique, and it is a net loss the moment your working set of experts exceeds what host bandwidth can feed.

---

## 9. The kernel: reading tokens through a routing index

Everything above treats the expert GEMM as a black box. In a real engine it is the hottest kernel in the layer, and it has a MoE-specific structure worth seeing, because it is where the last of the microseconds hide.

The naive grouped GEMM does a **gather** first — it physically reorders the token activations into per-expert contiguous blocks, writing a permuted copy to HBM, then runs a batched matmul over those blocks. That gather is pure overhead: it reads and writes the whole activation tensor just to rearrange it. The production trick is to *skip the standalone gather* by having the GEMM read its inputs through a **routing index** — an indirection table that tells each tile of the matmul which token rows it owns — so the reorder happens implicitly as the kernel loads, never as a separate HBM round-trip.

The vLLM team's [HPC-Ops post](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06), covering Tencent Hunyuan's attention and MoE backends, describes exactly this. Their fused FP8 MoE kernel has the **Gate-Up GEMM read tokens via a routing index, skipping the gather**, fuses the activation and FP8 quantization into the same kernel, and runs on a **persistent grid with PDL (Programmatic Dependent Launch) that erases the stage bubbles** between the grouped-GEMM stages. The measured result, on an H20 with 192 experts and top-8 routing at TP8/EP1 and batch 4, is **42.0 µs versus 56.4 µs for a Triton implementation and 74.5 µs for CUTLASS** — and end-to-end on 8×H20, **about 24% lower TTFT and 17% lower TPOT**. Their config is `--moe-backend hpc --kv-cache-dtype fp8_e4m3 --block-size 64`, and the acknowledged limits are that this MoE path is FP8-only and Hopper-only, best on H20.

That FP8 fusion connects directly to the [dequant-fused GEMM post](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) sibling: an expert's weights are just weights, so the same trick of unpacking a quantized weight in registers and feeding the tensor cores without ever writing the dequantized version back to HBM applies to expert GEMMs exactly as it does to a dense projection. In a MoE the payoff is even larger, because you are reading a *different* expert's weights for nearly every group and the bandwidth pressure is relentless. Quantize the expert weights, fuse the dequant into the grouped GEMM, and read the tokens through the routing index, and you have removed all three of the redundant HBM round-trips a naive MoE layer would pay.

| Kernel structure | Extra HBM traffic | Notes | Source |
| --- | --- | --- | --- |
| Gather → batched GEMM | Full activation read+write | The reorder is a standalone pass | derived |
| Routing-index GEMM | None (implicit reorder) | Skips the gather | cited: [HPC-Ops](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) |
| + dequant-fused weights | None (unpack in registers) | Expert weights never expand in HBM | cited: [dequant-fused GEMM](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) |
| HPC-Ops fused FP8 (H20) | Minimal | 42.0 µs vs 56.4 Triton / 74.5 CUTLASS | cited: [HPC-Ops](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) |

---

## 10. Stress tests: where MoE inference breaks

Take the pieces and push them until they fail. This is the section that turns a working layer into a robust one.

**The pathologically hot expert.** Feed a batch where routing is maximally skewed — every token's top choice is the same expert. Then $\rho = E$: one rank does *all* the expert work and the rest idle. EP has degenerated into serial execution on a single GPU, and adding GPUs makes it *worse* (more idle hardware, more all-to-all overhead for nothing). Your `LoadMonitor` catches it (`rho` near the expert count, `idle_experts` near $E-1$). The mitigations are EPLB replicas (spread the one hot expert across many GPUs) and, at training time, a load-balancing auxiliary loss — but at inference you cannot retrain the router, so replication is your only lever. This is the case that justifies a *capacity factor* (dropping tokens that overflow an expert's slot budget), which trades a little quality for a bounded straggler.

**Every token to the same expert, sustained.** The degenerate version of the above, held for a whole request — for instance a prompt entirely in a language one expert specializes in. Now the hot expert is hot for thousands of decode steps, not one, and even offload can help: the hot expert stays resident and warm, the cold tail streams. But if the *set* of hot experts drifts faster than PCIe can restage them, you thrash. Watch the resident-set churn, not just the instantaneous load.

**TP × EP combined topology.** Real deployments compose tensor parallelism (for the attention and dense layers) with expert parallelism (for the MoE layers), because the two collectives suit different parts of the model. The vLLM team's [distributed inference post](https://vllm.ai/blog/2025-02-17-distributed-inference) (2025-02-17) gives the placement rule verbatim: **"Use pipeline parallelism across nodes and tensor parallelism within nodes when interconnects are slow."** EP behaves like the node-crossing collective here — its all-to-all is heavy and latency-sensitive — so you keep the EP group on the fast intra-node fabric where you can. The failure mode is putting the all-to-all on a slow cross-node link: the $2 k \, d_{\text{model}} \, s$ per layer suddenly runs at InfiniBand speed instead of NVLink speed, and the comm term dominates everything. Recall too the super-linear memory effect the same post reports — going from TP=1 to TP=2 gave 13.9× more KV-cache blocks and 3.9× more token throughput because freeing weight memory freed cache room — which is the reason you shard at all.

**Offload thrash when the working set exceeds host bandwidth.** If your resident budget is 16 experts but the active set across a batch is 112 (from the Section 3 math), you evict and refetch constantly, each fetch a 7.9 ms PCIe stall. Throughput collapses to host bandwidth. The fix is not more clever eviction; it is admitting that offload only works when the *hot set* is small and stable, and sizing the resident budget to hold it. If you cannot, you need more GPU memory, not a smarter cache.

---

## 11. When MoE is a pessimization (be honest about it)

A MoE is not a free lunch, and there is a regime where it is strictly worse than a dense model. Say it plainly so you do not deploy one into the wrong workload.

**On one consumer GPU, at batch 1, a MoE is usually the wrong choice.** The whole model must be resident, so a 30B-A3B needs 30 GB in fp8 and does not fit a 24 GB RTX 4090 at all — the "3B active" number is irrelevant to whether it *runs*. Even where it fits (an A100 80GB), a single interactive user gives you batch 1, where the per-token bytes are the full active slice and none of the amortization from Section 3 has kicked in. A dense model of the *active* size (a real 3B model) would be smaller, fit anywhere, and decode just as fast at batch 1 — with none of the routing, all-to-all, or imbalance machinery. MoE's advantages are *quality per FLOP* and *throughput at scale*, and neither shows up for a single user on a single card. We put exact numbers on this in the Track H experiment on [MoE on consumer hardware](/blog/machine-learning/inference-engineering/experiment-a-moe-model-on-consumer-hardware).

**At the *other* extreme — very high batch — the MoE advantage also fades.** The vLLM team's [DeepSeek-R1 on GB200 post](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) reports that **throughput plateaus beyond 64K-token batches, and the MoE gains become negligible at high batch**. The reason is the coupon-collector saturation from Section 3: once the batch is large enough to touch every expert every step, you are reading the *entire* model each step anyway, and the "sparse" model is doing dense-model bandwidth. The sparsity bought you nothing once you are touching all the experts — you have the compute of a dense model of the *total* size, not the active size, plus the all-to-all tax on top. There is a sweet spot in the middle: batch high enough to amortize the resident weights, but not so high that you touch every expert every step and pay full dense bandwidth. Finding that spot for your model and fabric is the core MoE serving tuning problem.

| Regime | MoE verdict | Why | Source |
| --- | --- | --- | --- |
| Single user, one consumer GPU | Avoid | Does not fit; no amortization at batch 1 | derived |
| Moderate concurrency, fits in memory | Strong win | Resident weights amortize; small active FLOPs | derived + cited: [large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving) |
| Very high batch (64K+ tokens) | Gains fade | Every expert touched every step = dense bandwidth | cited: [DeepSeek-R1 GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) |
| Multi-node, slow interconnect | Risky | All-to-all runs at network speed, dominates | cited: [distributed inference](https://vllm.ai/blog/2025-02-17-distributed-inference) |

---

## Case studies: real numbers with provenance

**DeepSeek-V3 at 2.2k tok/s per H200.** The vLLM team's [Expert Parallelism at Scale post](https://vllm.ai/blog/2025-12-17-large-scale-serving) (2025-12-17) reports serving DeepSeek-V3 (37B of 671B active) at **2.2k tok/s per H200 GPU on a CoreWeave H200 cluster with ConnectX-7 InfiniBand, up from a ~1.5k baseline**. The ingredients they name: Wide-EP (expert parallelism plus data parallelism), all-to-all via DeepEP and Perplexity MoE kernels, DBO (`--enable-dbo`), EPLB (`--enable-eplb`), prefill/decode disaggregation, and DeepGEMM. The stated limit: higher EP degree increases inter-rank sync overhead. This is an aggregate throughput number under heavy concurrency — the exact opposite of the batch-1 ceiling — and it is the headline argument for the whole memory-for-FLOPs trade.

**DeepSeek-R1 on GB200 and the NVFP4 dispatch.** The [DeepSeek-R1 on GB200 post](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) (2026-02-03) reports, at 2K in / 2K out, **26.2K prefill tokens/GPU/s and 10.1K decode tokens/GPU/s**, with a deployment of 4 prefill servers (2 GB200 each) plus 1 decode server (8 GB200) — a 3–5× GPU-count reduction versus the H200 deployment (prefill 16→8, decode 32→8). The mechanism is **NVFP4 dispatch: activations quantized to FP4 before the all-to-all, giving 4× less communication than FP16**, on GB200 NVL72's 8 TB/s (versus 4.8 TB/s on H200). The caveat they state, and I repeated above, is the high-batch plateau.

**The HPC-Ops fused MoE kernel.** Tencent Hunyuan's backend, in the vLLM [HPC-Ops post](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06): a fused FP8 MoE kernel whose Gate-Up GEMM reads tokens through a routing index (no standalone gather), with PDL erasing stage bubbles. On H20, 192 experts top-8, TP8/EP1 batch 4: **42.0 µs versus 56.4 µs (Triton) and 74.5 µs (CUTLASS)**; end-to-end on 8×H20, **~24% lower TTFT and ~17% lower TPOT**. FP8-only, Hopper-only.

| Result | Setup | Number | Source |
| --- | --- | --- | --- |
| DeepSeek-V3 throughput | H200 cluster, Wide-EP, IB CX-7 | 2.2k tok/s/GPU (from ~1.5k) | cited: [large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving) |
| DeepSeek-R1 decode | GB200, NVFP4, 2K/2K | 10.1K decode tok/GPU/s | cited: [DeepSeek-R1 GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) |
| NVFP4 all-to-all | Blackwell, FP4 activations | 4× less comm vs FP16 | cited: [DeepSeek-R1 GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) |
| Fused MoE GEMM | H20, 192 experts, batch 4 | 42.0 µs vs 56.4 / 74.5 | cited: [HPC-Ops](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) |
| MoE all-to-all volume | DeepSeek-V3, bf16 activations | 12.7 MiB/token | derived (2·8·7168·2·58) |
| Batch-1 active bytes | 30B-A3B, fp8 | 3.76 GB/token | derived |

---

## When to reach for this (and when not)

Write the expert-parallel MoE layer yourself when you are *learning* how routing, all-to-all, and load balancing actually compose — that is what `nanoserve/layers/moe.py` is for, and by now you can trace a token from router to dispatch to expert GEMM to combine and back. You should also write the instrumentation for real: the `LoadMonitor` is a dozen lines and it is the difference between "the MoE is slow for some reason" and "expert 0 is drawing 3.5× the mean, put a replica on the idle GPU."

Do **not** ship your own all-to-all into production. The gap between the `torch.distributed.all_to_all_single` version above and a real deployment is enormous: DeepEP-class kernels, EPLB with live reshuffling, DBO overlap, NVFP4 dispatch quantization, prefix caching, prefill/decode disaggregation, and the fused routing-index GEMM. Use vLLM or SGLang, and read the [serving-MoE-at-scale post](/blog/machine-learning/model-serving/serving-moe-models-at-scale) and the [MoE architecture deep-dive](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) in the sibling series for the production picture. Reach for a MoE at all only when your workload has the concurrency to amortize the resident weights and the memory to hold them; for a single user on a single card, a dense model of the active size wins.

The overarching frame this series keeps returning to — weights → kernels → engine → decoding → API — puts MoE squarely in the "weights" and "kernels" layers, but its defining problem lives in the *engine*: routing, placement, and the scheduler-level decision of how much batch to gather before you fire a step. The [capstone playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) ties the whole scoreboard together, and MoE is the case where the memory line and the throughput line pull hardest in opposite directions.

---

## Key takeaways

- **MoE trades memory for FLOPs.** Compute scales with *active* parameters (fast); memory scales with *total* parameters (huge). DeepSeek-V3 is a 37B model in FLOPs and a 671B model in bytes.
- **At batch 1 a MoE is bandwidth-bound on its active slice** and holds many times the weights it reads. "3B active" does not mean "3B fast," and it says nothing about whether the model fits your GPU.
- **Batching amortizes the resident experts** via coupon-collector saturation: per-token bytes fall until the batch touches every expert, then flatten. MoE is a high-batch game.
- **Expert parallelism turns routing into an all-to-all** of volume $2 k \, d_{\text{model}} \, s$ per token per layer — for DeepSeek-V3, ~13 MiB per token — and at high EP degree the network, not HBM, is the wall.
- **Load imbalance is the central inference problem.** A hot expert makes its GPU a straggler, and the combine barrier runs the whole layer at the slowest rank's pace. Measure $\rho = \max_e n_e / \bar{n}$.
- **EPLB replicates hot experts; DBO overlaps the comms; NVFP4 shrinks the payload 4×.** Pull all three in production.
- **Elastic EP resizes the group live** (drain → standby comms → atomic switch → EPLB reshuffle) — mechanism only, no published perf, TP=1 and Ray-DP only.
- **Expert offload is a capacity trick, not a speed trick.** A cold-expert fetch costs ~8 ms over PCIe versus microseconds of compute; it only pays with a small, stable hot set and high reuse.
- **The kernel win is skipping the gather** (read tokens through a routing index) and fusing dequant — the same register-level trick as dense dequant-fused GEMM, applied per expert.
- **A MoE is a pessimization for a single user on one card, and its gains fade at very high batch** where every expert is touched every step and you pay full dense bandwidth.

---

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series intro and the honesty rule.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that ties the scoreboard together.
- [Tensor-parallel inference by hand](/blog/machine-learning/inference-engineering/tensor-parallel-inference-by-hand) — the symmetric collective (all-reduce) that EP's asymmetric all-to-all contrasts with.
- [Dequant-fused GEMM: int4 weights on the fly](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) — the register-level dequant trick that applies to expert weights.
- [Eviction, preemption, and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) — the tiered-memory offload machinery reused for cold experts.
- [Serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale) — the production-engine view of everything built here by hand.
- vLLM, [Expert Parallelism at Scale](https://vllm.ai/blog/2025-12-17-large-scale-serving) (2025-12-17) — Wide-EP, DeepEP, DBO, EPLB, 2.2k tok/s per H200.
- vLLM, [DeepSeek-R1 Wide-EP on GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) (2026-02-03) — NVFP4 dispatch and the high-batch plateau.
- vLLM, [Elastic Expert Parallelism](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism) (2026-05-14) — live EP-group resizing (mechanism only).
- vLLM, [HPC-Ops attention and MoE backends](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06) — the fused routing-index MoE kernel.

---
title: "Pruning in LLMs: A Practical Guide with Code and Case Studies"
publishDate: "2026-04-23"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["llm", "pruning", "sparsity", "model-compression", "inference", "deep-learning", "SparseGPT", "Wanda", "Llama", "Minitron"]
date: "2026-04-23"
author: "Hiep Tran"
featured: false
aiGenerated: true
image: "/imgs/blogs/pruning-in-llm-01-taxonomy.png"
excerpt: "A deep dive into pruning for large language models — the taxonomy, importance scoring math, prune-then-heal recipes, and case studies: SparseGPT, Wanda, LLM-Pruner, Sheared LLaMA, Minitron, and Llama 3.2."
---

## Why Pruning Matters

Frontier LLMs are dense and overparameterized. A 70B Llama stores roughly 140GB of weights in fp16 — most of them doing very little on any given input. Once a model is trained, a sizeable fraction of its parameters can be **zeroed out** or **physically removed** with minimal loss in quality, provided the removal is done carefully and (ideally) followed by a short recovery phase.

> Train a giant dense model. Then delete the parts that don't matter. Then fine-tune the rest.

Done well, pruning gives you **2–5× smaller models, 1.5–3× faster inference**, and — crucially — opens the door to running frontier capabilities on consumer hardware. It's behind SparseGPT, Wanda, LLM-Pruner, Sheared LLaMA, NVIDIA's Minitron series, and the structured pruning stage of Llama 3.2 1B/3B.

### The Three Reasons Pruning Works At All

If you've spent any time with neural networks, pruning feels wrong the first time you see it. You trained every weight for a reason. How can you throw half away?

Three results explain why pruning is viable — and why it keeps working better as models get larger:

**1. Overparameterization.** Training a modern LLM requires far more parameters than strictly necessary to represent the learned function. The excess parameters are a **training-time convenience**, not an inference-time requirement — they give the optimizer more degrees of freedom to find a good loss landscape, but once training is done, many are redundant. This is not a conjecture — it's been measured repeatedly. For LLaMA-7B, you can remove ~50% of weights with near-zero perplexity change and **no retraining**.

**2. The Lottery Ticket Hypothesis (Frankle & Carbin, 2019).** Inside every large trained network lives a smaller **subnetwork** — a "winning ticket" — that, when trained in isolation from its original initialization, matches the full network's quality. Pruning is how you find it. Modern LLM pruning empirically behaves like lottery-ticket extraction: the pruned subnet carries most of the capability; the pruned weights were scaffolding.

**3. Inference economy is non-linear.** Serving cost for an LLM scales with parameter count in two places: **memory bandwidth** (fetching weights during decode) and **compute** (matmul FLOPs during prefill). Cutting parameters by 2× does not save 2× — it also unlocks larger batch sizes, lower KV-cache spill, and cheaper hardware tiers. A 7B model serving 50K requests/day can drop infrastructure cost by 3–5× vs. a 70B model, not just 10×. That's why prune-then-deploy is an economic question, not just an engineering one.

### Pruning In The Compression Toolkit

Pruning is one of four major compression techniques. You rarely use just one:

| Technique | What it removes / changes | Typical compression | Quality cost | Compound with pruning? |
|---|---|---|---|---|
| **Pruning** | Parameters | 2–5× | Low with healing | (base) |
| **Quantization** | Bits per weight (fp16 → int4) | 2–4× | Low with GPTQ/AWQ | Yes — prune then quantize |
| **Distillation** | Full retrain into smaller arch | 5–20× | Moderate | Yes — distillation *is* a healing method |
| **Architecture tricks** | GQA, MQA, sliding window | 1.5–3× on KV cache | Minimal | Orthogonal |

In production, the dominant recipe today is **prune + distill + quantize**: structurally prune the dense teacher, distill into the pruned student, then int4-quantize the result for deployment. Llama 3.2 1B ships exactly this way.

### What This Guide Covers

This guide covers the full picture: the taxonomy, the math behind importance scoring, the code to run each variant, quality expectations and how to measure them, and concrete case studies with real numbers. It's written for engineers who will actually ship a pruned model to production — so it emphasizes the decisions senior practitioners have to make (when to use pruning vs. something else, how to detect silent failures, how to budget for healing) over the academic novelty of any particular method.

## Section 0 — When To Use Pruning: A Decision Framework

Before any code, answer four questions. Most pruning projects fail because they skip this step and end up in a regime where a different technique would have been a better fit.

### 0.1 Do I actually have a size problem?

Pruning is a **compression** tool. If the bottleneck is latency on a single request, the bigger win is usually elsewhere:

- **Prefill-bound** (long context, short output): cut context, use sliding-window attention, or optimize KV computation. Pruning helps but isn't the primary lever.
- **Decode-bound** (short context, long output): memory bandwidth is the bottleneck. Here pruning (and quantization) are direct wins.
- **Throughput-bound** (batch serving): bigger batches help more than a smaller model. Pruning helps indirectly by leaving more GPU memory for batches.

**Rule of thumb:** if the model fits in your target memory but is too slow, **quantize first** (int8 or int4). If the model doesn't fit or you need to halve parameters for a smaller GPU tier, **prune**.

### 0.2 Do I have the dense teacher's weights?

Almost every effective pruning recipe needs the **original dense weights** — both for calibration (scoring importance via activations) and for healing (distilling the quality back). If you only have API access to the teacher:

- Pruning is mostly off the table. You cannot run a forward pass on the teacher's internals, cannot score weights, cannot distill logit-for-logit.
- Fall back to **response distillation** into an already-small student (see the [distillation guide](/blog/machine-learning/large-language-model/distillation-in-llm)).

### 0.3 What is my deployment hardware?

Pruning's speedup depends critically on hardware:

| Hardware | Unstructured | 2:4 Semi-Struct | Structured Width/Depth |
|---|---|---|---|
| Consumer GPU (RTX 4090) | No speedup (memory only) | 1.5–1.7× | Linear in size |
| Datacenter GPU (A100/H100, Ampere+) | No speedup on tensor cores | 1.6–1.7× (hardware native) | Linear in size |
| CPU (Xeon / M-series) | Possible 2–3× with oneDNN / XNNPACK | Not native | Linear in size |
| Phone NPU | Often no sparse support | Rare | Linear in size |

**Rule of thumb:** for phones, laptops, and CPU servers, use **structured pruning** — it's the only variant that gives portable speedups. For H100 servers, **2:4 semi-structured** is the sweet spot. Unstructured is mainly for research or memory-constrained offline workloads.

### 0.4 How aggressive is my target size reduction?

- **< 30% reduction:** pruning alone with light LoRA heal is usually enough.
- **30–60% reduction:** pruning + short continued pretraining or distillation.
- **60–80% reduction:** iterative prune + distill from the dense teacher, multiple stages.
- **> 80% reduction:** **don't prune — distill or train small from scratch.** At that compression ratio, you're fighting too much of the model's inductive structure.

Beyond ~80%, the labour of keeping quality up exceeds the cost of just training a purpose-built small model. Pruning has a hard ceiling set by how much redundancy the big model actually contains.

### 0.5 The decision summary

```
         has teacher weights?
               │
         ┌─────┴──────┐
        yes           no
         │             └─→ response distillation (not pruning)
         ▼
   target size reduction?
         │
   ┌─────┼──────┬──────────┐
  <30%  30-60%  60-80%    >80%
   │      │       │         │
   │      │       │         └─→ train small from scratch or
   │      │       │             full distillation recipe
   │      │       │
   │      │       └─→ iterative structured prune + distill
   │      │
   │      └─→ one-shot structured prune + distill OR
   │          continued pretraining
   │
   └─→ one-shot unstructured (Wanda) +
       light LoRA heal + quantize
```

This framework is what a senior practitioner runs through before writing any pruning code. Everything below assumes you're in a regime where pruning is the right tool.

## Section 1 — The Taxonomy of Pruning

![Taxonomy of LLM pruning: unstructured, semi-structured (2:4), structured (width, depth), and MoE expert pruning](/imgs/blogs/pruning-in-llm-01-taxonomy.png)

Pruning methods split along two axes: **what you remove** (individual weights vs. whole structures) and **when you do it** (post-training one-shot vs. during/after fine-tuning).

### 1.1 Unstructured Pruning

Zero out individual weights — any weight, anywhere, independent of its neighbors.

- **Granularity:** single scalar.
- **Pros:** highest compression at a given quality level. Reaches 50–70% sparsity on LLMs with minimal accuracy loss.
- **Cons:** **no speedup on standard GPUs.** Sparse matrix multiplication on dense tensor cores is usually *slower* than dense. You need specialized sparse kernels (cuSPARSE, Sputnik) or accept the memory-only win.

**Why quality holds up well:** unstructured pruning has the maximum flexibility — for any sparsity budget, it can pick the exact worst-performing individual weights. It pays the smallest "constraint cost" among all methods.

**Why it rarely gives latency wins:** modern GPUs are heavily optimized for **dense** matmul through tensor cores. A 50%-sparse matrix, stored as a CSR sparse tensor, has overhead per element that exceeds the FLOP savings at typical GPU arithmetic intensity. You win on **model file size** (useful for serving N models on one machine) and **CPU inference** (where sparse BLAS libraries exist), not on per-request GPU latency.

**Use when:**
- You're storing many model variants and want to shrink the on-disk / on-RAM footprint.
- You're deploying to CPU where libraries like Intel oneDNN can exploit unstructured sparsity.
- You're doing research — it's the fairest apples-to-apples comparison across methods.

**Don't use when** your goal is real-world GPU serving latency. Use 2:4 or structured pruning instead.

### 1.2 Semi-Structured Pruning (N:M Sparsity)

A middle ground: within every group of M consecutive weights, zero out N of them. The canonical setup is **2:4 sparsity** — in every row of 4 weights, exactly 2 must be zero.

- **Granularity:** 4-weight blocks.
- **Pros:** NVIDIA Ampere+ tensor cores have *native* hardware support for 2:4 sparsity. You get a ~1.5–1.7× speedup on real workloads without custom kernels.
- **Cons:** the 2:4 constraint is strict — forcing it often costs more quality than free-form 50% unstructured.

Supported by PyTorch (`torch.sparse.to_sparse_semi_structured`) and by cuSPARSELt for matmul.

**Why the constraint costs quality:** unstructured 50% can pick any 50% of weights to prune; 2:4 must pick exactly 2 out of every 4. When a block of 4 has three "important" weights, 2:4 is forced to prune one of them. Empirically, 2:4 loses ~0.3–0.8 perplexity points vs. 50% unstructured on LLaMA-7B before healing. After healing, the gap shrinks to 0.1–0.3 — often within measurement noise.

**Why 2:4 won over 4:8, 8:16, etc.:** hardware. NVIDIA's sparse tensor core unit decodes 2:4 patterns in a single cycle. Other ratios require software emulation that kills the speedup. If your target is an NVIDIA GPU, pick 2:4. Other ratios are a research curiosity unless your hardware vendor supports them natively.

**Memory layout gotcha:** the pattern is per-row (along the input dimension), not per-column. This matters when you add LoRA adapters or quantization on top — the sparse pattern must be preserved at inference time, so your adapter merging logic has to respect the 2:4 block boundaries.

**Use when:** you're deploying on H100/A100/RTX Ampere+ and want a "free" 1.5× inference speedup with minimal engineering. **Don't use when** you're on older hardware (no speedup) or you can afford the full prune + distill recipe with structured pruning (better portability and higher speedup).

### 1.3 Structured Pruning — Width

Remove entire **columns of weight matrices**: attention heads, feed-forward hidden units, embedding dimensions. The model shrinks as a plain dense model of smaller size.

- **Granularity:** channel / head / neuron.
- **Pros:** **real speedup on any hardware** — it's still a dense matmul, just smaller.
- **Cons:** coarser granularity means bigger quality drop per parameter removed. Typically needs aggressive post-prune recovery.

**The three width axes and their trade-offs:**

| Axis | What you cut | Typical safe range | Quality sensitivity |
|---|---|---|---|
| **Attention heads** | Reduce `num_attention_heads` | 12.5–25% of heads | Medium — some heads are specialized (induction, copy) |
| **FFN hidden** | Reduce `intermediate_size` | 25–50% of FFN hidden | Low — FFN is the most redundant part |
| **Hidden dim** | Reduce `hidden_size` (affects everything) | 10–20% of `d_model` | High — touches residual stream; most destructive |
| **Vocab / embed** | Tied embedding compression | Rare | High — cannot usually be done without retraining |

FFN hidden is where you cut first: it's where most parameters live (usually 2/3 of FFN is the intermediate layer) and redundancy is highest. Cutting `d_model` is the last resort — it touches every layer's residual stream and every projection matrix.

**Coupling constraints:** structured width pruning has dependencies. If you prune a head in layer L's self-attention, you must also prune the corresponding columns in the Q/K/V projections *and* rows in the O projection. If you shrink FFN hidden, you must do it consistently across gate/up/down (for SwiGLU) projections. Missing a coupling produces a model that silently computes garbage — this is the #1 source of bugs in structured pruning code.

**Use when** you're deploying to a real GPU/CPU and need predictable latency. Structured width is the workhorse for phone-scale and edge models — Llama 3.2 and Minitron both go here first.

### 1.4 Structured Pruning — Depth

Remove entire **transformer blocks** — whole layers at a time.

- **Granularity:** layer.
- **Pros:** linear speedup in depth. Also reduces activation memory proportionally.
- **Cons:** deep layers often do task-specific work; dropping them can tank reasoning-heavy tasks even when perplexity looks OK.

Often combined with width pruning ("prune both depth and width"), as in Llama 3.2 and Minitron.

**Which layers are safe to drop?** Not all — this is the most common "it looks fine but breaks" trap in pruning. Empirical findings for decoder-only transformers:

- **Early layers (0–5):** low-level tokenization, copy heads, positional patterns. **Do not drop.** Removing even one often destroys tokenization robustness.
- **Middle layers (roughly 1/3 to 2/3):** the most redundant region. Safe to prune 10–30% here.
- **Late layers (last 3–5):** task-specific, high-impact. Dropping a late layer often crushes MMLU and reasoning benchmarks while leaving perplexity nearly unchanged. **Do not drop** without careful evaluation.

A good depth-pruning strategy scores layers by **output-change impact** (how much the final logit distribution shifts when layer L is replaced by identity) and never drops more than 25% of middle layers at a time.

**Depth vs. width trade-off:** at equal parameter reduction, depth pruning usually hurts more on reasoning, while width pruning hurts more on knowledge retrieval. Most production recipes cut ~15% depth + ~30% width rather than going deep in either axis.

**Use when** you need to halve latency quickly for a memory-fit model (depth prune is pure speedup) and you have budget for a thorough healing phase.

### 1.5 MoE Expert Pruning

For Mixture-of-Experts models (Mixtral, DeepSeek-MoE, Qwen-MoE), you can prune **entire experts** — whole `FFN` subnetworks — based on how often the router selects them.

- **Granularity:** expert.
- **Pros:** straightforward speedup; expert sharding is already the bottleneck in MoE inference.
- **Cons:** experts are often highly specialized — pruning the wrong one can destroy a niche capability.

**Why expert pruning is different:** MoE models already have sparsity built in — only `top-k` experts activate per token. But the **total memory** still holds every expert. Pruning experts cuts the memory footprint, which for MoE models is the dominant serving cost (typical Mixtral 8x7B = 47B total parameters even though only 12.9B activate per token).

**Scoring experts** — three signals that should agree before you prune:
1. **Router utilization** — fraction of tokens routed to the expert across a calibration set.
2. **Output contribution** — L2 norm of the expert's output contribution to the residual stream.
3. **Domain coverage** — does the expert fire on your target domain (code, math, chat)? Measure per-domain utilization separately.

Pruning by utilization alone is dangerous: some experts fire rarely but carry critical capabilities (coding in a general chat model, for example). Always bucket your calibration set by domain and check each domain's expert distribution before cutting.

**Use when** you're serving an MoE model under memory pressure and can accept a per-domain quality check. **Don't use when** you're serving a highly-specialized workload without enough domain coverage to measure expert specialization.

## Section 2 — Importance Scoring: Which Weights Deserve to Live?

The core question of any pruning method is: **given a weight or group of weights, how much does it matter?** Different answers give different methods.

![Importance scoring methods: magnitude, gradient-based (OBS/OBD), activation-based (Wanda), and loss-based (SparseGPT)](/imgs/blogs/pruning-in-llm-02-importance.png)

### 2.1 Magnitude Pruning

The simplest scoring function: **bigger absolute weight = more important**.

$$
s_{ij} \;=\; |W_{ij}|
$$

Remove the bottom-$k\%$ by magnitude. Set to zero. Done.

**Why it works at all:** for well-trained networks, small weights often reflect noise or near-redundancy. Zeroing them perturbs the output by a small amount.

**Why it's no longer state-of-the-art for LLMs:** it ignores how *sensitive* the loss is to each weight. A small weight in a heavily-used dimension can matter far more than a large weight in an unused one.

### 2.2 Gradient-Based Importance (OBD / OBS)

Optimal Brain Damage (LeCun, 1990) and Optimal Brain Surgeon (Hassibi, 1993) use a **second-order Taylor expansion** of the loss. Around a local minimum (so the gradient is zero), removing weight $w_i$ changes the loss by:

$$
\Delta \mathcal{L}_i \;\approx\; \tfrac{1}{2} w_i^2 \, H_{ii}
$$

where $H_{ii}$ is the $i$-th diagonal of the Hessian. The importance score becomes:

$$
s_i \;=\; \tfrac{1}{2} w_i^2 \, H_{ii}
$$

OBS additionally **compensates** the remaining weights after pruning:

$$
\delta w \;=\; -\, \frac{w_i}{H^{-1}_{ii}} \; H^{-1} e_i
$$

This reshuffles surviving weights to best approximate the pruned model's output on a calibration set. SparseGPT is a scalable approximation of OBS designed for billion-parameter models.

### 2.3 Activation-Based Importance (Wanda)

Wanda (Sun et al., 2023) makes a simple observation: in transformers, different input channels have **very different activation magnitudes** (outlier features). A small weight multiplied by a giant activation can matter more than a large weight multiplied by a tiny one. Score each weight by weight magnitude **times** the L2 norm of its input activations:

$$
s_{ij} \;=\; |W_{ij}| \cdot \lVert X_{:,j} \rVert_2
$$

where $X \in \mathbb{R}^{B \times d_\text{in}}$ is a batch of activations entering the layer. Prune lowest-$k\%$ per row.

- **No retraining. No gradient. No Hessian.** Just one forward pass over a small calibration set (128 sequences is enough).
- Matches or beats SparseGPT quality at 50% unstructured sparsity.
- **This is the default one-shot pruning method for LLMs in 2025.**

### 2.4 Loss-Aware Reconstruction (SparseGPT)

SparseGPT treats pruning as a **layer-wise least-squares problem**. For each linear layer, find the sparse weight matrix $\hat{W}$ that minimizes:

$$
\min_{\hat{W}} \; \lVert W X - \hat{W} X \rVert_F^2 \quad \text{s.t.} \quad \hat{W} \text{ has} \leq k\% \text{ non-zeros}
$$

where $X$ is activations from a calibration set. This is solved with an approximate OBS update using a running Hessian inverse, one column at a time. More expensive than Wanda but gives slightly better numbers at higher sparsities.

## Section 3 — Pruning Code, From Minimal to Full

### 3.1 Magnitude Pruning (unstructured, 50%)

```python
import torch
import torch.nn as nn

@torch.no_grad()
def magnitude_prune(module: nn.Linear, sparsity: float = 0.5):
    """Zero out the smallest |w| in the weight matrix."""
    W = module.weight.data
    k = int(sparsity * W.numel())
    # Flatten, find threshold
    threshold = W.abs().view(-1).kthvalue(k).values
    mask = W.abs() > threshold
    module.weight.data.mul_(mask)
    # Register the mask as a buffer so it can be re-applied after optimizer steps
    module.register_buffer("sparse_mask", mask)

def apply_masks(model):
    for m in model.modules():
        if hasattr(m, "sparse_mask"):
            m.weight.data.mul_(m.sparse_mask)

# Usage
for m in model.modules():
    if isinstance(m, nn.Linear):
        magnitude_prune(m, sparsity=0.5)
```

### 3.2 Wanda (activation-aware, one-shot)

```python
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

class WandaRow(nn.Module):
    """Collects L2 norm of inputs to a Linear, per input channel."""
    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.layer = layer
        self.scaler = torch.zeros(layer.in_features, device=layer.weight.device)
        self.n = 0

    def add_batch(self, x):
        # x: [..., in_features]
        x = x.reshape(-1, x.shape[-1])
        self.scaler += (x.pow(2).sum(dim=0))
        self.n += x.shape[0]

    def finalize(self):
        return (self.scaler / self.n).sqrt()    # per-channel L2 norm

@torch.no_grad()
def wanda_prune_layer(layer: nn.Linear, calib_inputs, sparsity=0.5):
    """One-shot Wanda on a single Linear using a list of calibration activations."""
    scorer = WandaRow(layer)
    for x in calib_inputs:
        scorer.add_batch(x)
    act_norm = scorer.finalize()                  # [in_features]

    W = layer.weight.data                          # [out, in]
    S = W.abs() * act_norm.unsqueeze(0)            # per-element importance

    # Per-row pruning: drop lowest sparsity% per output neuron
    k = int(sparsity * W.shape[1])
    _, idx = S.topk(k, dim=1, largest=False)
    mask = torch.ones_like(W, dtype=torch.bool)
    mask.scatter_(1, idx, False)

    layer.weight.data.mul_(mask)
    return mask

# Calibration: run 128 sequences through the model, capture inputs per layer
def collect_calibration(model, tokenizer, texts, layer_names):
    cache = {name: [] for name in layer_names}
    hooks = []
    def make_hook(name):
        def hook(_, inputs, __):
            cache[name].append(inputs[0].detach())
        return hook
    for name, mod in model.named_modules():
        if name in layer_names:
            hooks.append(mod.register_forward_hook(make_hook(name)))
    model.eval()
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model(ids)
    for h in hooks: h.remove()
    return cache
```

### 3.3 2:4 Semi-Structured Pruning

```python
import torch

@torch.no_grad()
def prune_2_4(W: torch.Tensor) -> torch.Tensor:
    """Within every 4-weight block along dim=1, keep the top 2 by |w|, zero the rest."""
    assert W.shape[1] % 4 == 0, "in_features must be divisible by 4"
    Wg = W.view(W.shape[0], -1, 4)                 # [out, groups, 4]
    _, idx = Wg.abs().topk(2, dim=-1)              # top-2 indices per block
    mask = torch.zeros_like(Wg, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return (Wg * mask).view_as(W), mask.view_as(W)

# Fast inference path using PyTorch native 2:4 support (Ampere+)
from torch.sparse import to_sparse_semi_structured
# (call after pruning and weight fusion)
# sparse_W = to_sparse_semi_structured(dense_W_pruned)
```

### 3.4 Layer Dropping (structured depth pruning)

```python
import torch

@torch.no_grad()
def block_importance(model, calib_loader):
    """Score each transformer block by the output change when it's skipped."""
    scores = []
    for i, block in enumerate(model.model.layers):
        total = 0.0
        original = block.forward
        def identity(x, *a, **kw): return (x,)
        block.forward = identity
        for batch in calib_loader:
            with torch.no_grad():
                out_skip = model(**batch).logits
            block.forward = original
            out_full = model(**batch).logits
            total += (out_full - out_skip).pow(2).mean().item()
            block.forward = identity
        block.forward = original
        scores.append((i, total))
    return sorted(scores, key=lambda x: x[1])      # lowest = most droppable

def drop_layers(model, drop_idx: list[int]):
    keep = [l for i, l in enumerate(model.model.layers) if i not in drop_idx]
    model.model.layers = torch.nn.ModuleList(keep)
    model.config.num_hidden_layers = len(keep)
```

### 3.5 Structured Width Pruning — Attention Heads

```python
import torch

@torch.no_grad()
def head_importance(model, calib_loader, num_heads: int, head_dim: int):
    """Per-head L1 importance using gradient-free activation norm."""
    scores = torch.zeros(len(model.model.layers), num_heads)
    for i, layer in enumerate(model.model.layers):
        def hook(module, inputs, output):
            # output: [B, L, H, D] after attention reshape
            o = output[0] if isinstance(output, tuple) else output
            o = o.view(o.size(0), o.size(1), num_heads, head_dim)
            scores[i] += o.abs().mean(dim=(0, 1, 3)).cpu()
        h = layer.self_attn.register_forward_hook(hook)
        for batch in calib_loader:
            model(**batch)
        h.remove()
    return scores

def prune_heads(model, heads_to_drop: dict[int, list[int]]):
    """Physically remove the lowest-scoring heads from each layer."""
    for layer_idx, head_idxs in heads_to_drop.items():
        attn = model.model.layers[layer_idx].self_attn
        # rebuild qkv projections without the dropped heads (omitted for brevity)
        ...
```

## Section 4 — Prune-Then-Heal: The Standard Recipe

![Prune-then-heal pipeline: calibrate -> score -> prune -> recover via distillation or continued pretraining -> evaluate](/imgs/blogs/pruning-in-llm-03-recipe.png)

Pruning a billion-parameter LLM **always** drops quality at the moment of the cut. The field has converged on a simple recipe:

1. **Calibrate.** Run 128–2048 sequences from a representative corpus through the dense model; capture per-layer activation statistics.
2. **Score & prune.** Apply your chosen method (Wanda / SparseGPT / head-prune / layer-drop) using the calibration stats.
3. **Heal.** Fine-tune or distill the pruned model to recover the quality drop. This is the step that separates a research result from a deployable model.
4. **Evaluate.** Measure on capability benchmarks, not just perplexity.

### 4.1 How to Heal

Three healing strategies, in order of increasing cost and effectiveness:

**(a) Short LoRA fine-tune** — 1–10k steps of LoRA on the pruned model using a small high-quality SFT set. Recovers 70–90% of the quality gap. Cheap: hours on a single GPU for 7B models.

**(b) Continued pretraining** — a few billion tokens of generic text through the pruned model with a low LR. Recovers near-full quality but is expensive; used by Sheared LLaMA (50B tokens) and Minitron (100–400B tokens).

**(c) Distillation from the dense teacher** — use the original dense model as teacher and the pruned model as student. Apply logit KD (Section 3 of the [distillation guide](/blog/machine-learning/large-language-model/distillation-in-llm)). This is the gold standard — Llama 3.2 1B/3B and Minitron both use it.

```python
# Minimal prune -> distill heal loop
def heal_via_distill(pruned_student, dense_teacher, loader, steps=5000,
                     lr=1e-5, T=1.0, alpha=0.1):
    opt = torch.optim.AdamW(pruned_student.parameters(), lr=lr)
    for step, batch in enumerate(loader):
        with torch.no_grad():
            t_logits = dense_teacher(**batch).logits
        s_logits = pruned_student(**batch).logits

        # Re-apply mask so optimizer can't revive pruned weights
        apply_masks(pruned_student)

        log_p_s = F.log_softmax(s_logits / T, dim=-1)
        log_p_t = F.log_softmax(t_logits / T, dim=-1)
        kl = F.kl_div(log_p_s, log_p_t, reduction="batchmean",
                      log_target=True) * (T * T)
        ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)),
                             batch["labels"].view(-1), ignore_index=-100)
        loss = alpha * ce + (1 - alpha) * kl

        opt.zero_grad(); loss.backward(); opt.step()
        if step >= steps: break
```

### 4.2 Iterative Pruning

Instead of cutting 50% in one shot, cut 10%, heal, cut another 10%, heal again. Slower but preserves more quality at aggressive sparsities (> 50%). Used by LLM-Pruner and the structured stages of Sheared LLaMA.

## Section 5 — Case Studies With Concrete Numbers

### 5.1 SparseGPT (2023)

- **Method:** layer-wise least-squares OBS approximation (Section 2.4).
- **Scale:** OPT-175B and BLOOM-176B pruned to 50% unstructured sparsity **in 4 hours on a single A100**.
- **Result:** perplexity increase of only ~0.2 on WikiText-2 vs. the dense model.
- **Takeaway:** demonstrated that 50% sparsity is essentially free on big LLMs, **without any retraining**.

### 5.2 Wanda (2023)

- **Method:** $\lvert W \rvert \cdot \lVert X \rVert_2$ (Section 2.3).
- **Setup:** 128 sequences of 2048 tokens for calibration; no gradient, no Hessian, no fine-tuning.
- **Result:** matches SparseGPT at 50% unstructured sparsity, with 10× less compute (~4 minutes vs. 4 hours on A100 for LLaMA-7B). Slightly worse at 2:4 sparsity.
- **Takeaway:** made one-shot pruning practical as an everyday operation — became the default baseline.

### 5.3 LLM-Pruner (2023)

- **Method:** structured pruning (width + coupled channels), gradient-based scoring via a small calibration set, followed by LoRA healing.
- **Result:** LLaMA-7B pruned by ~20% still retained 90%+ of MMLU/commonsense benchmarks after LoRA healing on 50K instruction samples.
- **Takeaway:** first widely-cited *structured* pruning recipe for LLMs — proved LoRA is enough for modest compression ratios.

### 5.4 Sheared LLaMA (2024)

- **Method:** joint optimization of layer / head / dimension masks via a learned gating mechanism, then **continued pretraining** on 50B tokens with dynamic data mixtures.
- **Scale:** LLaMA-2-7B pruned to 1.3B and 2.7B.
- **Results:** Sheared LLaMA 2.7B outperforms OpenLLaMA 3B on 9/11 downstream tasks, using ~3% of the compute required to train a 2.7B from scratch.
- **Takeaway:** prune-from-big is **much cheaper** than pretrain-small, even including the healing phase.

### 5.5 Minitron (NVIDIA, 2024)

- **Method:** iterative structured pruning of width (embedding, hidden, attention heads) + depth, each stage followed by distillation from the original dense teacher.
- **Scale:** Nemotron-15B pruned to 8B and 4B variants.
- **Results:** Minitron-8B reached quality on par with Nemotron-15B on most benchmarks, at ~40% the compute. Minitron-4B matched Llama-3.1-Minitron-4B at considerably lower training cost.
- **Takeaway:** prune + distill produces smaller models with 40× less training compute than from-scratch — now the industry-default recipe for model family compression.

### 5.6 Llama 3.2 1B/3B (Meta, 2024)

- **Recipe (Stage A):** structured pruning of Llama 3.1-8B — depth prune entire layers, width prune heads and hidden dim, guided by importance scores.
- **Recipe (Stage B):** distillation from Llama 3.1-8B and 3.1-70B as teachers on a large corpus, using logit KD.
- **Results:** Llama 3.2 1B runs at > 30 tokens/sec on a modern phone CPU, with benchmark scores a few points below Llama 3.1-8B.
- **Takeaway:** the combination of **structured prune + distill** is now the default edge-model recipe for production deployments.

## Section 5.5 — Quality After Pruning: What To Expect And How To Measure

This is the section most articles skip. Senior practitioners spend more time on **quality characterization** than on pruning itself. The reason: pruning rarely "fails loudly" — a pruned model looks 95% like the original while being dangerously worse in a narrow subset of behaviors.

### 5.5.1 The Canonical Quality Curve

For a well-calibrated dense LLM, quality as a function of sparsity (with healing) looks like:

```
Quality
  │
1.0├────────────────────────╮
   │         dense level    ╲ plateau ("free zone")
   │                         ╲
   │                           ╲_______  linear decay zone
   │                                   ╲
   │                                     ╲
   │                                       ╲  cliff
   │                                         ╲___________
   │                                                     ╲___
   └──────────────────────────────────────────────────────── →
   0%    10%   20%   30%   40%   50%   60%   70%   80%  sparsity
        └─ free ─┘  └─ linear decay ─┘    └── cliff ──┘
```

Three distinct regimes:

- **Free zone (0–30%):** quality drop is ~0 after healing. This is the region every production recipe targets. If pruning puts you here, measure sanity metrics, merge, ship.
- **Linear decay (30–60%):** each additional percent of pruning costs a roughly linear amount of quality. Healing recovers most but not all. This is where most research results sit.
- **Cliff (> 60–70%):** non-linear quality collapse. The remaining model can no longer represent the task — no amount of healing recovers it. Stop pruning, train a smaller model from scratch instead.

**The free zone is model-size dependent.** Larger models have larger free zones (70B can lose 40% almost free; 1B hits the cliff at 25–30%). Smaller models have less redundancy to give.

### 5.5.2 What Breaks First

Different capabilities degrade in different orders under pruning pressure. Approximate sensitivity ordering (most fragile → most robust):

1. **Long-context reasoning** (needle-in-haystack, multi-hop across 32K tokens) — depends on fine-grained attention structure.
2. **Code generation** (HumanEval, MBPP) — punishes small numerical errors.
3. **Math reasoning** (GSM8K, MATH) — sensitive to symbolic manipulation chains.
4. **Instruction following** (IFEval) — needs precise format tracking.
5. **World knowledge** (MMLU, TriviaQA) — moderately robust; knowledge is distributed.
6. **Language modeling perplexity** — the most robust. **Drops last.**

This ordering has a scary implication: **perplexity is a bad pruning canary.** A 2% perplexity increase may hide a 15% MMLU drop and a 40% HumanEval collapse. Every production pruning run needs at minimum:
- MMLU (general knowledge)
- GSM8K or MATH (reasoning)
- HumanEval or MBPP (code)
- IFEval (instruction following)
- A held-out internal eval matching your deployment distribution

Running only perplexity as the post-prune check is the single most common evaluation failure in the field.

### 5.5.3 The Calibration-Set Bias Trap

Wanda, SparseGPT, and any activation-based method score importance using a **calibration set**. Your choice of calibration set biases which weights survive:

- Calibrate on C4 (general web text) → weights helping code tend to get pruned.
- Calibrate on The Pile (more diverse) → balanced, but heavy on academic text.
- Calibrate on your deployment distribution → weights tuned to your task, but fragile outside it.

**Senior-level practice:** mix three calibration sources (generic web + deployment distribution + adversarial/edge cases) at roughly equal weights. This is the single biggest lever for pruning quality that beginners miss. SparseGPT papers report up to 5 perplexity points of difference depending purely on calibration choice — with the same pruning algorithm.

### 5.5.4 Statistical Significance in Pruning Evals

Pruning introduces **variance** in the final model. Two runs of Wanda with different calibration samples produce models with slightly different weights. A single evaluation score is not enough evidence to ship:

- Evaluate **3+ seeds** of the pruning run (different calibration samples).
- Report **mean and standard deviation**, not just best result.
- Use **pass@1 at temperature 0.6** for generative tasks (not greedy — underestimates).
- Use **bootstrap confidence intervals** for accuracy metrics.

A pruned model that beats dense by 0.5 points on one eval run but loses by 2 points on another is a **regression** masquerading as noise. Distinguish the two with multiple runs.

### 5.5.5 Silent Failures: Safety & Behavior Regressions

Pruning can change **behavior** without changing measured **capability**:

- **Safety:** refusal patterns often weaken after pruning. A pruned model may comply with prompts the dense one refused. Run your safety evals (ToxicChat, HarmBench, internal red team) on every candidate.
- **Calibration:** the pruned model's confidence scores often become miscalibrated — it says "very likely" for things that are wrong more often.
- **Stylistic drift:** tone, formatting, hedging change subtly. Users often notice before benchmarks do.
- **Language/script balance:** non-English capabilities are usually more fragile than English under pruning (less training signal, fewer redundant paths). Measure per-language if you ship multilingual.

**Senior-level practice:** include a 50–100-sample **human spot-check** pass before any pruned model ships, covering refusals, tone, non-English, and long-form coherence. This is the step that catches silent failures.

## Section 5.6 — Senior-Level Considerations: Things Only Experience Teaches

A non-exhaustive list of the considerations that distinguish a research prototype from a shippable pruned model.

### 5.6.1 Budget the Full Pipeline, Not Just Pruning

The pruning step itself is fast — often hours. The **healing** is the time sink, and it's systematically underestimated in planning:

| Approach | Typical healing cost (7B target) | Quality recovered |
|---|---|---|
| No healing (one-shot) | 0 | 60–80% |
| LoRA (5k steps, 1 GPU) | 2–6 hours | 80–92% |
| Continued pretraining (20B tokens) | 1–3 days × 8 GPUs | 95–98% |
| Distillation from dense (20B tokens, teacher forward too) | 3–7 days × 8 GPUs | 97–99% |

If your timeline is "ship in a week", plan for LoRA-heal quality, not distill-heal quality. Over-promising on pruning quality without budgeting healing is a classic project-killer.

### 5.6.2 Compounding: Prune + Quantize

Pruning and quantization **compound**. A 50% pruned model quantized to int4 has ~8× compression (not 4×+2×, but close). The order matters:

- **Prune first, then quantize.** Quantization is a local perturbation; pruning is a global one. Doing it in the wrong order can amplify errors.
- **Heal after each step.** Quality drops at each stage; heal at each stage if you want to hit top-line numbers.
- **GPTQ / AWQ are the typical choices** for post-prune quantization — they respect sparse weights and apply per-channel scaling that plays nicely with pruning masks.

A realistic production pipeline for 70B → phone:
```
70B dense → structured prune to 7B → distill from 70B → int4 quantize → ship
```
Each step is well-understood. Each step has its own heal. Skipping any one costs quality.

### 5.6.3 Reproducibility and Model Cards

Pruning introduces non-trivial reproducibility burden. The pruned model's behavior depends on:
- Random seed of the calibration sample draw.
- Calibration set content (exact tokens).
- Healing data mix.
- Optimizer state at each stage.

For shipped models, your **model card** should record:
- Pruning method + sparsity.
- Calibration set source and size.
- Healing recipe (data, LR, steps, method).
- Eval suite used for quality gate.
- Specific seeds for full reproducibility.

Without this, the pruned model is an opaque artifact that the team cannot rebuild if something goes wrong in production.

### 5.6.4 Know Your Exit Criteria Before You Start

Define your **shipping bar** before pruning — otherwise you will rationalize bad numbers. Example criteria for a consumer chat deployment:

- MMLU within 2 points of dense baseline
- IFEval within 3 points of dense baseline
- HumanEval pass@1 within 4 points of dense
- Safety refusal rate within 1% of dense
- Human spot-check: no major regressions on 50-prompt panel
- Latency ≥ 2× improvement on target hardware

If you hit all, ship. If you miss 1–2 by a small margin, iterate. If you miss 3+, the pruning recipe is wrong; stop tweaking and rethink.

### 5.6.5 Integration With The Training Story

A pruned model is not a standalone artifact — it interacts with fine-tuning pipelines, safety post-training, RLHF, etc. Several senior-level gotchas:

- **Sparse masks must be preserved through fine-tuning.** Standard optimizer steps will revive pruned weights (they receive non-zero gradients even from zero starting points). Either zero gradients on masked positions or re-apply the mask after every `optimizer.step()`. Missing this is the #2 bug in pruning pipelines (after coupling bugs in structured prune).
- **LoRA adapters need pruning-aware design.** If the base is sparse, LoRA rank-1 updates may inadvertently fill pruned positions. Enforce the mask on `W + AB^T` at merge time.
- **RLHF on a pruned model is riskier than on a dense one.** Less redundancy means KL constraints bite earlier; the policy can drift into a regime the pruned model no longer handles well. Reduce RLHF KL coefficient by 2–3× vs. dense.
- **Safety tuning should be **redone**, not skipped.** Pruning weakens safety in ways that transfer learning poorly. Run a fresh safety SFT pass on every pruned checkpoint.

### 5.6.6 Organizational Considerations

Pruning is a cross-functional project. It needs:

- **Infrastructure buy-in:** new serving path for the pruned variant; changes to CI model-quality gates.
- **Eval team sign-off:** expanded eval suite (see 5.5.2).
- **Safety team sign-off:** refusal/red-team panel must pass.
- **Product sign-off:** real users may notice differences; need UAT (user acceptance testing) before full rollout.
- **Rollback plan:** every pruned deployment needs a trivially-rollbackable path to the dense model. This is non-negotiable — silent capability regressions get caught *after* launch, not before.

Skipping any of these turns a successful technical project into an organizational disaster. Senior practitioners spend 30% of total project time on these concerns.

## Section 6 — Troubleshooting: What Goes Wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Perplexity is fine, benchmarks collapse | Pruned structure-specific capabilities (long-context, tool-use). | Evaluate on diverse tasks **before** and **after** pruning, not just WikiText. |
| Wanda score sensitive to calibration set | Activation norms are input-dependent. | Use ≥ 128 sequences covering deployment distribution. C4 is a safe generic default. |
| Optimizer "revives" pruned weights | Gradients are still flowing to zero'd positions. | Reapply the sparse mask after every `optimizer.step()`, or zero out gradients on pruned positions. |
| 2:4 sparsity tanks quality but 50% unstructured is fine | The 2:4 constraint is strict and rarely matches the natural importance pattern. | Heal with distillation (Section 4.1c), not plain fine-tuning. 2:4 almost always requires healing. |
| Pruned model is slower than expected | Unstructured sparsity with no sparse kernel. | Switch to 2:4 (hardware-accelerated) or structured (plain dense matmul, just smaller). |
| Layer drop hurts reasoning but helps perplexity | Removed layers are specialized for complex/deep reasoning. | Score layers on a reasoning-specific calibration set (math/code), not just plain text. |
| Healing loss goes down but generations degrade | Fine-tuning recovers perplexity but not behavior; student overfits healing set. | Use distillation (logit KD) instead of plain SFT; keep healing LR low (≤ 5e-6). |
| MoE pruning loses a capability | Dropped expert was doing specialized work (e.g., code). | Score experts separately per domain (code, math, chat); prune conservatively. |

## Section 7 — When To Use Pruning (And When Not To) — Revisited

Section 0 gave the upfront decision framework. With everything above in context, here is the richer senior-level view:

### Pruning is the right tool when…

- **You already have a strong dense model** and need a deployable smaller version. Pruning trades a known quantity (dense model quality) for a smaller cost; training a small model from scratch trades compute for unknown quality.
- **You have the dense teacher's weights.** Distillation-based healing requires them; without them, pruning loses half its power.
- **Compression target is < 60%.** Beyond that, the healing cost approaches the cost of training a small model from scratch.
- **Your hardware has sparse-matmul support** (H100/A100 for 2:4) or you're doing structured pruning (works everywhere).
- **Your eval suite is mature** — you can detect silent regressions. If you only have perplexity, don't prune; you'll ship a broken model without knowing it.
- **You have rollback capacity.** Every pruned deployment should be revertible to dense.

### Pruning is the wrong tool when…

- **You need extreme compression (> 80%).** Training a purpose-built small model from scratch — or full distillation — will usually win past this point.
- **The dense model is already small (< 1B).** There's not much redundancy to reclaim; quality degrades steeply.
- **You're CPU-only without a sparse kernel** — unstructured pruning is memory-only win, not latency.
- **You need strong behavioral guarantees** without thorough safety eval. Pruning silently shifts refusal, tone, calibration.
- **You're under a hard timeline and cannot budget the healing phase.** Pruning without healing is a research demo, not a product.
- **The big win is elsewhere** — latency often wants quantization or batching before pruning; prefill often wants context truncation or faster attention.
- **Your team has never pruned before and ships on Friday.** First pruning runs always hide bugs (coupling, mask persistence, calibration). Plan for 2–3 iterations.

### Pruning is overkill when…

- **Quantization alone suffices.** Int8 or int4 often gets you 2–4× compression and 1.5–3× speedup with less engineering.
- **A smaller model from the same family already exists.** If you can use Llama 3.2 1B out of the box instead of pruning Llama 3.1 8B → 1B yourself, use the shipped model. Meta has spent more engineering on it than you can afford.
- **Batch throughput, not per-request latency, is the goal.** Spin up more replicas and batch harder before pruning.

## Section 8 — The Big Picture

![Evolution of LLM pruning: from magnitude (1990) through OBS, SparseGPT, Wanda, to prune+distill in 2024-2025](/imgs/blogs/pruning-in-llm-04-evolution.png)

- **OBD / OBS (1990, 1993)** — the theoretical foundation: importance is a loss derivative.
- **Magnitude pruning (1990s–2020)** — simple, strong baseline that refuses to die.
- **SparseGPT (2023)** — made OBS-scale pruning tractable on billion-parameter models.
- **Wanda (2023)** — reduced one-shot pruning to a single forward pass; became the default.
- **LLM-Pruner (2023)** — proved structured pruning + LoRA heal works on LLMs.
- **Sheared LLaMA (2024)** — structured pruning + continued pretraining as a cheap way to create a new model family.
- **Minitron (2024)** — iterative structured prune + distill, the industry-default recipe today.
- **Llama 3.2 (2024)** — prune + distill shipped to production at scale on consumer devices.

The field has converged on one recipe: **structured prune + distill from the original dense teacher**. Unstructured and semi-structured variants are useful for memory-only wins and special hardware. For inference speed in 2026, structured + distill is what you should try first.

Pruning is no longer a research sideshow. Every major model family that ships a "small" variant now builds it by pruning the large one and distilling back the quality. If you deploy LLMs in production, you'll meet pruning — probably via a prune-then-distill pipeline that looks a lot like the one in Section 4.

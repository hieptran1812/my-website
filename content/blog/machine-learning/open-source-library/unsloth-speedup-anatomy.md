---
title: "The anatomy of an Unsloth speedup: where VRAM and time actually go"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "A precise mental model of where a single fine-tuning step spends VRAM and time, and the five exact rewrites Unsloth uses to attack each cost without losing a single bit of accuracy."
tags: ["unsloth", "llm-finetuning", "qlora", "lora", "triton", "gpu-memory", "optimization", "gradient-checkpointing", "cross-entropy", "bitsandbytes"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 28
---

You can take an 8-billion-parameter model, attach LoRA adapters, and run a full supervised fine-tune of it on a free Colab T4 — a single GPU with about 15 GB of usable VRAM. That sentence should sound impossible to anyone who has ever tried to do it the obvious way. Load Llama-3.1-8B in fp16 and you are already at 16 GB of weights, before a single gradient or optimizer state exists. The naive arithmetic says you need a machine with five or six times the memory the T4 has. And yet [Unsloth](https://github.com/unslothai/unsloth) does it, reproducibly, and finishes the run roughly twice as fast as a stock Hugging Face + PEFT setup would on the same card.

The interesting question is not "how do I run the notebook." It is: **what, exactly, did Unsloth change?** Because the headline — "2x faster, 70% less VRAM" — is a number, and a number is not a mental model. If you cannot say *which* tensor got smaller and *which* kernel got cheaper, you cannot reason about whether the trick still holds for your model, your sequence length, your hardware. This post is the hub of a ten-part series that takes Unsloth apart kernel by kernel. Before we open any of them, we need the map. (If you want the higher-level tour first, the [Unsloth overview](/blog/machine-learning/open-source-library/unsloth-lib) is the place to start.)

![How Unsloth attacks each cost: a single QLoRA step splits into a VRAM cost and a time cost, and each Unsloth technique is an exact rewrite aimed at one of them](/imgs/blogs/unsloth-speedup-anatomy-1.webp)

The diagram above is the mental model for the entire series. A single QLoRA training step incurs two kinds of cost — VRAM and time — and every Unsloth technique is an exact rewrite that attacks one named line in one of those two budgets. 4-bit NF4 attacks the weight bill; the offloaded checkpointer attacks the activation bill; the 8-bit/paged optimizer attacks Adam's state; fused cross-entropy attacks the logits bill; fused Triton kernels and hand-derived backprop attack wall-clock time. There is no box on that diagram labeled "approximate the math." That absence is the thesis. Everything Unsloth does is exact — the loss curve is the same, the final weights are the same — and the speed comes from spending memory and bandwidth more carefully, not from cutting corners on numerics.

The rest of this post builds that map precisely: what a training step actually produces, where the bytes live, where the microseconds go, and which of the nine sibling posts dissects each lever.

## 1. The anatomy of one training step

Before we can say where the cost goes, we have to be honest about what a step *is*. People say "the model trains" as if it were one event. It is three, and each one leaves a different artifact behind in GPU memory.

![What one training step produces: forward yields logits and saved activations, the loss runs cross-entropy over the vocab, backward consumes the activations into gradients, and the optimizer step rewrites the weights](/imgs/blogs/unsloth-speedup-anatomy-2.webp)

**Forward.** You push a batch of token IDs through the network. Each layer reads the previous hidden state, applies its weights (attention projections, MLP, norms), and writes a new hidden state. The final layer produces **logits** — one score per vocabulary entry, per token. Crucially, forward does not just produce the output; it also *stashes intermediate activations* so that backward can use them. That stash is the silent memory cost we will return to repeatedly.

**Loss and backward.** The loss function — cross-entropy over the vocabulary — turns the logits into a single scalar. Then backward walks the network in reverse, and at each layer it combines the incoming gradient with the saved activations to produce two things: the gradient with respect to the layer's *inputs* (passed further back) and the gradient with respect to the layer's *weights* (accumulated). The reason forward had to save activations is exactly this: the chain rule for a matmul $Y = XW$ needs $X$ to compute $\partial \mathcal{L}/\partial W = X^\top \, \partial \mathcal{L}/\partial Y$. No saved $X$, no weight gradient.

**Optimizer step.** Once gradients exist, the optimizer updates the weights. For Adam — the default everyone uses — the update is not just "subtract learning rate times gradient." Adam maintains two running statistics per parameter: a first moment (a smoothed gradient) and a second moment (a smoothed squared gradient). It reads those, updates them, and uses them to compute the actual step. Those two buffers are persistent — they survive across steps — and as we are about to see, they are the most expensive occupant of the entire ledger.

Each H2 from here on is about squeezing one of those phases. But you cannot squeeze a cost you have not measured, so let us build the ledger.

## 2. The VRAM ledger

There is a quiet lie in how most people estimate VRAM. They read "8B parameters," multiply by 2 bytes for fp16, get 16 GB, and budget for that. That number is real but it is only the *first* of four occupants, and for full fine-tuning it is the smallest of the big three.

The four occupants of GPU memory during training are:

1. **Weights** — the parameters themselves.
2. **Gradients** — one number per trainable parameter, same shape as the weights.
3. **Optimizer state** — Adam's two moment buffers, traditionally held in fp32.
4. **Activations** — everything forward stashed for backward, proportional to `batch × seq_len × layers × hidden`.

Here is a small pedagogical calculator — this is **my own illustration, not Unsloth API** — that prices each occupant for an 8B model under two regimes. It is deliberately simple; it ignores temporary buffers and the CUDA context, but it captures the structural story.

```python
# --- a memory ledger for an 8B model. Illustrative, not Unsloth code. ---
P = 8.0e9                      # 8 billion parameters
GiB = 1024 ** 3

def gib(bytes_):
    return bytes_ / GiB

def full_finetune_fp16(P, batch=2, seq=2048, layers=32, hidden=4096):
    weights   = P * 2          # fp16 weights:        2 bytes/param
    grads     = P * 2          # fp16 gradients:      2 bytes/param
    # Adam in mixed precision keeps fp32 master weights + fp32 m + fp32 v:
    optimizer = P * 4 * 3      # master + m + v:      12 bytes/param  <-- the hog
    # activations scale with the batch you actually push through:
    acts      = batch * seq * layers * hidden * 2    # rough fp16 estimate
    return {
        "weights":   gib(weights),
        "gradients": gib(grads),
        "optimizer": gib(optimizer),
        "activations": gib(acts),
    }

def qlora_4bit(P, r=16, layers=32, hidden=4096, batch=2, seq=2048):
    # base weights are frozen in NF4: ~0.5 bytes/param + a small absmax overhead
    weights   = P * 0.5
    # only the LoRA adapters are trainable. A rank-r adapter on a (hidden, hidden)
    # projection is 2 * r * hidden params; assume ~7 adapted matrices per layer.
    lora_params = layers * 7 * (2 * r * hidden)
    grads     = lora_params * 2                       # grads on adapters only
    optimizer = lora_params * 2 * 2                   # 8-bit Adam: ~2 bytes * (m, v)
    acts      = batch * seq * layers * hidden * 2     # checkpointing shrinks this
    return {
        "weights":   gib(weights),
        "gradients": gib(grads),
        "optimizer": gib(optimizer),
        "activations": gib(acts),
        "lora_params_millions": lora_params / 1e6,
    }

if __name__ == "__main__":
    print("full fp16:", {k: round(v, 1) for k, v in full_finetune_fp16(P).items()})
    print("qlora 4bit:", {k: round(v, 2) for k, v in qlora_4bit(P).items()})
```

Run it and the asymmetry is stark. The full-fine-tune optimizer line is `8e9 × 12 / GiB ≈ 89 GB` on its own — bigger than weights and gradients combined. This is the single most counterintuitive fact about training memory: **Adam's state, not the model, is usually the largest thing on the GPU.** Two moments at fp32 is 8 bytes per parameter; add the fp32 master copy mixed precision keeps and you are at 12 bytes per parameter, six times the 2-byte fp16 weight itself.

The QLoRA column tells the opposite story. The base weights drop from 16 GB to roughly 4-5 GB by living in 4-bit NF4 and staying *frozen* — they never get a gradient, never get an optimizer state. The only trainable tensors are the LoRA adapters, which for a rank-16 setup are a few tens of millions of parameters, three orders of magnitude smaller than the base. Gradients and optimizer state collapse from "the biggest bills" to a rounding error.

![The VRAM ledger for an 8B model: full fp16 fine-tuning pays ~16 GB weights, ~16 GB gradients, ~64 GB Adam state and ~48 GB master-plus-activations, while 4-bit QLoRA pays ~5 GB frozen weights, ~0.1 GB grads, ~0.2 GB optimizer, and checkpointed activations](/imgs/blogs/unsloth-speedup-anatomy-3.webp)

The figure above is the ledger drawn to scale (conceptually — the exact GB depend on batch and sequence length). The left column is fp16 full fine-tuning: four full-precision bills, dominated by the optimizer. The right column is 4-bit QLoRA: three of the four bills have been deleted or shrunk to near-zero, and the only survivor is activations — which is precisely why activation memory becomes the *next* thing worth attacking, and why the offloaded checkpointer exists.

This is the cleanest way to understand the "70% less VRAM" headline. It is not one trick. It is the compounding of: weights quantized to a quarter size, gradients and optimizer state restricted to the tiny adapter set, and activations offloaded. QLoRA itself (the technique from the [QLoRA paper](https://arxiv.org/abs/2305.14314)) is what removes the first three bills; Unsloth's contribution is making the kernels around them exact-but-cheap, and shrinking the fourth. The same NF4 numerics that make weight quantization safe are discussed in detail in the sibling [4-bit quantization deep-dive](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4), and the general problem of quantizing tensors without losing fidelity shows up again in the KV-cache context in [TurboQuant](/blog/machine-learning/open-source-library/turboquant-kv-cache-quantization-deep-dive).

| Occupant | fp16 full fine-tune (8B) | 4-bit QLoRA (8B) | What removed it |
|---|---|---|---|
| Weights | ~16 GB (fp16) | ~5 GB (NF4, frozen) | 4-bit quantization |
| Gradients | ~16 GB | ~0.1 GB (adapters only) | freeze base; train LoRA |
| Optimizer state | ~64-89 GB (Adam fp32) | ~0.2 GB (8-bit on adapters) | adapters + 8-bit/paged Adam |
| Activations | ~tens of GB | shrunk to ~1 block | offloaded checkpointing |
| **Total** | **well over 100 GB** | **fits in <15 GB** | the five levers, compounded |

## 3. Where the time goes

Memory is the bill that decides *whether* the run fits. Time is the bill that decides how long you wait. They are different problems with different bottlenecks, and conflating them is the most common reason people optimize the wrong thing.

The key distinction is **compute-bound vs memory-bound**. A GPU has two relevant resources: arithmetic throughput (how many FLOPs per second the tensor cores can do) and memory bandwidth (how many bytes per second it can read from and write to HBM). An operation is *compute-bound* if it does enough math per byte loaded that the tensor cores are the limit; it is *memory-bound* if it does so little math per byte that it spends its life waiting for HBM. The ratio of FLOPs to bytes is the operation's **arithmetic intensity**, and it is the single number that predicts which resource you are pinned on.

![Where the time goes: matmuls are FLOP-bound with high arithmetic intensity and best left to cuBLAS, attention is mixed and handled by Flash Attention, while RMSNorm, RoPE and SwiGLU are HBM-bandwidth-bound with low intensity and that is where fused Triton kernels pay off](/imgs/blogs/unsloth-speedup-anatomy-4.webp)

The matrix above classifies the operations in a transformer layer. The big matmuls — QKV projections, the MLP up/down/gate projections — have high arithmetic intensity. They reuse each loaded weight across many tokens, so they keep the tensor cores busy and they are already close to optimal under cuBLAS. There is almost nothing to win there, which is why Unsloth does **not** write a custom matmul kernel; it lets the vendor library do its job.

The losers are the small elementwise operations: RMSNorm, RoPE, the SwiGLU activation, residual adds. Each of these reads a big activation tensor from HBM, does a trivial amount of math per element (a multiply, a square root, a sigmoid), and writes the result back. Their arithmetic intensity is low, so they are pinned on bandwidth, and the *only* thing that makes them faster is reducing HBM traffic — which means **fusing** them so an intermediate result never round-trips to memory.

There is a second, subtler time cost that the matrix does not show: **kernel-launch overhead.** Every CUDA kernel you launch has a fixed cost — a few microseconds of CPU-side dispatch and GPU-side scheduling — that is independent of how much work it does. A naive RMSNorm in PyTorch is not one kernel; it is a square, a mean, a reciprocal-square-root, a broadcast multiply, and a scale — five or six launches, each round-tripping the activation through HBM, each paying its own launch tax. At small tensor sizes the launch tax alone can dominate. Fuse all six into one Triton kernel and you pay the launch cost once and touch HBM twice (read, write) instead of a dozen times. That is where the wall-clock win on the "boring" layers comes from. The mechanics of one such kernel are dissected in [Triton kernel fusion](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion), and the RoPE and attention path specifically in [RoPE and attention kernels](/blog/machine-learning/open-source-library/unsloth-rope-attention-kernels).

Why does fusion matter *more* for fine-tuning than for inference? Because each of these memory-bound ops appears twice per step — once in forward and once in backward — and a transformer with 32 layers has dozens of them per layer. A naive eager step might issue several thousand tiny kernel launches, and on a small GPU like the T4 the CPU dispatching them can become the actual bottleneck: the GPU finishes each tiny op and then sits idle waiting for the next launch to arrive. This is the "GPU is only 40% utilized" symptom that people misread as "my model is too small to saturate the card." It is usually not the model; it is launch latency. Fusion is the cure precisely because it cuts the launch count, and the fused kernels also avoid re-reading tensors that a chain of eager ops would each load independently. The compounding of fewer launches *and* fewer HBM round-trips is why the per-step time roughly halves even though the matmuls — the bulk of the FLOPs — were never touched.

```python
# What "fused" buys you. A naive RMSNorm in eager PyTorch:
def rms_norm_naive(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)   # launch 1-2: square + mean  -> read x, write var
    x = x * torch.rsqrt(var + eps)          # launch 3-4: rsqrt + mul     -> read x + var, write
    return x * weight                       # launch 5:   scale          -> read x, write
# ~5 kernel launches, x crosses the HBM boundary several times.
#
# Unsloth's fast_rms_layernorm does square -> mean -> rsqrt -> scale in ONE Triton
# kernel: x is read once, the output written once, and only a single float per row
# (the inverse RMS) is kept for the backward pass. Exact same arithmetic; one launch.
```

## 4. The five levers

Now the map has coordinates. Every Unsloth optimization lands on one of the costs we just named. Here are the five levers, each with its target, its one-line mechanism, and the sibling post that dissects it.

**Lever 1 — Fused Triton kernels (attacks: time on memory-bound ops).** RMSNorm, RoPE, and SwiGLU are rewritten as single Triton kernels that do the whole operation in one launch with minimal HBM round-trips. The forward kernel also saves only the *minimum* it needs for backward — for RMSNorm that is one float per row (the inverse RMS), not the full normalized tensor. Dissected in [Triton kernel fusion](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion).

**Lever 2 — Hand-derived backprop (attacks: time and memory in backward).** Instead of letting PyTorch autograd build a tape and store every intermediate, Unsloth writes custom `torch.autograd.Function`s whose backward is the *analytically derived* gradient. Because the math is closed-form, the backward kernel recomputes cheaply, reuses buffers in place, and saves far fewer tensors. Dissected in [manual backprop](/blog/machine-learning/open-source-library/unsloth-manual-backprop).

**Lever 3 — 4-bit QLoRA with frozen base weights (attacks: weight, gradient, and optimizer VRAM).** The base model lives in 4-bit NF4 and never trains. Only LoRA adapters carry gradients and optimizer state. This is what deletes three of the four ledger bills. Dissected in [4-bit NF4 quantization](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4).

**Lever 4 — Fused / chunked cross-entropy (attacks: the logits VRAM wall).** The loss is computed without ever materializing the full `(tokens × vocab)` softmax tensor; only a per-token logsumexp scalar is kept, and the gradient is written in place over the logits buffer. For large-vocab models a chunked path keeps even the logsumexp bounded. Dissected in [fused cross-entropy](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy).

**Lever 5 — Offloaded gradient checkpointing + 8-bit/paged optimizers (attacks: activation and optimizer VRAM).** Activations are recomputed in backward rather than stored, and Unsloth additionally streams the checkpoint inputs to system RAM during forward so VRAM holds roughly one block's worth at a time. The optimizer moments are held in 8-bit or paged to CPU. Dissected in [gradient checkpointing and offload](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload) and [8-bit and paged optimizers](/blog/machine-learning/open-source-library/unsloth-8bit-paged-optimizers).

Beyond the five, two more posts round out the series: [long-context training](/blog/machine-learning/open-source-library/unsloth-long-context-training), which shows how these levers combine to push sequence length far past what raw VRAM would allow, and the capstone [VRAM budget and export](/blog/machine-learning/open-source-library/unsloth-vram-budget-and-export), which adds up every byte and walks the merge/GGUF export path.

Let us look more closely at the two that are easiest to get wrong: the cross-entropy wall and the hand-derived backward.

### The cross-entropy wall

When people first hear "fused cross-entropy" they assume it is a minor kernel tweak. It is not — for large-vocabulary models it is often the difference between fitting and OOM. The reason is that a naive `F.cross_entropy` materializes a full softmax (or log-softmax) tensor of shape `(batch × seq, vocab)` in fp32.

![The cross-entropy memory wall: a naive loss materializes a full rows-by-vocab fp32 softmax tensor plus a separate gradient tensor, while the fused path keeps only one logsumexp float per row and writes the gradient in place on the logits buffer](/imgs/blogs/unsloth-speedup-anatomy-5.webp)

Do the arithmetic for a worst case. Gemma's vocabulary is 256K. At a sequence length of 8192 with a small batch, `tokens × vocab` is on the order of 2 billion entries; in fp32 that is roughly 8 GB for the softmax tensor alone — and a naive implementation needs a *second* tensor of the same shape for the gradient. That is 16 GB of transient memory for a loss function, on a GPU where the whole point was to fit in 15.

Unsloth's `Fast_CrossEntropyLoss` refuses to build either tensor. The forward kernel computes, per row, the numerically stable logsumexp $\text{lse} = c + \log \sum_i e^{x_i - c}$ where $c = \max_i x_i$, and the loss is just $\text{lse} - x_{\text{label}}$. It stores **one float per token** — the logsumexp — and nothing else. The backward kernel then recomputes the softmax on the fly from that saved scalar, because $\partial \mathcal{L}/\partial x_i = e^{x_i - \text{lse}} - \mathbb{1}[i = \text{label}]$, and writes the gradient **in place over the logits buffer** — no second tensor is allocated. For vocabularies that exceed the `MAX_FUSED_SIZE = 65536` chunk width, a chunked path computes a logsumexp per chunk and combines them with the identity $\text{logsumexp}(\text{concat}) = \text{logsumexp}([\text{lse}_1, \dots, \text{lse}_k])$. The 8 GB tensor becomes a vector of a few thousand floats.

### Hand-derived backward

The second easy-to-miss lever is what custom `autograd.Function`s buy you. Autograd is general: it records every operation onto a tape and, in backward, replays the chain rule, storing whatever intermediates it might need. That generality has a memory and launch cost. When you *know* the closed-form gradient of a fused block, you can do far better.

![Hand-derived backprop: a fused forward saves only adapters, the input X, and two SwiGLU intermediates; backward reuses them, dequantizes the frozen NF4 weight transiently then frees it, computes LoRA grads via addmm_, and writes dX in place into the X buffer](/imgs/blogs/unsloth-speedup-anatomy-6.webp)

The figure traces Unsloth's `LoRA_MLP` Function for a SwiGLU MLP with LoRA on the gate, up, and down projections. The forward runs `matmul_lora` (base matmul plus the low-rank adapter term) and the fused Triton SwiGLU, then calls `save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)` — the tiny adapters and two intermediates, not the big hidden states. Backward reads those, runs the analytically derived gradient (the SwiGLU derivative `df = sigmoid(e)*(1 - f) + f` is hand-coded, not retraced), and does three memory-frugal things at once:

1. It dequantizes the frozen 4-bit base weight *transiently* — `fast_dequantize(upW.t(), upW_quant)`, use it, then `del upW` — so the 16-bit copy of a base weight never persists in VRAM.
2. It accumulates the LoRA gradients with `addmm_(..., beta=0)`, which overwrites the destination instead of allocating a new tensor.
3. It computes the input gradient `dX` *in place* with `out = X` when `inplace=True`, reusing the input buffer rather than allocating a fresh one.

The result is a backward that touches almost no new memory and launches a handful of kernels instead of dozens. None of it changes the numerics — it is the same gradient PyTorch would compute, derived by hand and executed frugally.

## 5. The optimizer state, made small

Section 2 named Adam's fp32 state as the largest occupant for full fine-tuning. QLoRA already shrinks it by training only the adapters, but the *form* of the optimizer state still matters, and it is worth seeing the two regimes side by side.

<figure class="blog-anim">
<svg viewBox="0 0 640 320" role="img" aria-label="The optimizer state stack alternates between full fp32 Adam moments over all parameters and 8-bit paged moments over only the LoRA adapters" style="width:100%;height:auto;max-width:760px">
<style>
.a8-base{stroke:var(--border,#d1d5db);stroke-width:1.5}
.a8-full{fill:var(--accent,#6366f1);opacity:.85}
.a8-small{fill:var(--accent,#6366f1)}
.a8-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a8-num{font:600 14px ui-mono,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes a8-fadeA{0%,38%{opacity:1}52%,92%{opacity:0}100%{opacity:1}}
@keyframes a8-fadeB{0%,38%{opacity:0}52%,92%{opacity:1}100%{opacity:0}}
.a8-A{animation:a8-fadeA 9s ease-in-out infinite}
.a8-B{animation:a8-fadeB 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a8-A{animation:none;opacity:1}.a8-B{animation:none;opacity:0}}
</style>
<g class="a8-A">
<rect class="a8-full a8-base" x="120" y="60"  width="180" height="50"/>
<rect class="a8-full a8-base" x="120" y="115" width="180" height="50"/>
<rect class="a8-full a8-base" x="120" y="170" width="180" height="50"/>
<rect class="a8-full a8-base" x="120" y="225" width="180" height="50"/>
<text class="a8-lbl" x="210" y="90">m: fp32 over all params</text>
<text class="a8-lbl" x="210" y="145">v: fp32 over all params</text>
<text class="a8-lbl" x="210" y="200">fp32 master weights</text>
<text class="a8-lbl" x="210" y="255">gradients fp16</text>
<text class="a8-num" x="210" y="300">Adam = 2x params in fp32 ~ 64 GB</text>
</g>
<g class="a8-B">
<rect class="a8-small a8-base" x="380" y="240" width="180" height="35"/>
<rect class="a8-base" x="380" y="60" width="180" height="180" fill="var(--surface,#f3f4f6)"/>
<text class="a8-lbl" x="470" y="262">8-bit moments on LoRA</text>
<text class="a8-lbl" x="470" y="150">base frozen (no state)</text>
<text class="a8-num" x="470" y="300">paged 8-bit on adapters ~ 0.2 GB</text>
</g>
</svg>
<figcaption>The same optimizer role, two regimes: full fp32 Adam over every weight (the silent VRAM hog) versus paged 8-bit moments over only the LoRA adapters.</figcaption>
</figure>

The animation alternates between the two states the same role can take. On the left, full fp32 Adam over every parameter: two moments plus a master copy, the dominant bill. On the right, the QLoRA regime: the base is frozen and carries no optimizer state at all, and the moments that remain — only for the adapters — are held in 8-bit and can be *paged* to CPU memory when they are not in use. The 8-bit moments are not an approximation of the update direction in a way that hurts convergence; bitsandbytes' 8-bit Adam uses block-wise quantization tuned so the optimizer trajectory matches fp32 closely, which is the whole reason it is safe to use. The page table mechanics and the 8-bit numerics are the subject of [8-bit and paged optimizers](/blog/machine-learning/open-source-library/unsloth-8bit-paged-optimizers).

And once the optimizer is small, the last large occupant is activations — which the offloaded checkpointer collapses.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Activation memory grows layer by layer through the forward pass, then collapses to roughly one block under offloaded gradient checkpointing" style="width:100%;height:auto;max-width:820px">
<style>
.a7-axis{stroke:var(--border,#d1d5db);stroke-width:2}
.a7-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a7-cap{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a7-blk{fill:var(--accent,#6366f1);opacity:.85}
.a7-keep{fill:var(--accent,#6366f1)}
@keyframes a7-grow{0%{transform:scaleY(0)}40%{transform:scaleY(1)}70%{transform:scaleY(1)}82%{transform:scaleY(0)}100%{transform:scaleY(0)}}
@keyframes a7-keepgrow{0%{transform:scaleY(0)}40%{transform:scaleY(1)}70%{transform:scaleY(1)}82%{transform:scaleY(1)}100%{transform:scaleY(1)}}
@keyframes a7-modefade{0%,70%{opacity:0}84%,100%{opacity:1}}
@keyframes a7-fwdfade{0%,70%{opacity:1}84%,100%{opacity:0}}
.a7-g{transform-box:fill-box;transform-origin:bottom;animation:a7-grow 9s ease-in-out infinite}
.a7-d2{animation-delay:.5s}.a7-d3{animation-delay:1s}.a7-d4{animation-delay:1.5s}
.a7-d5{animation-delay:2s}.a7-d6{animation-delay:2.5s}.a7-d7{animation-delay:3s}
.a7-survivor{transform-box:fill-box;transform-origin:bottom;animation:a7-keepgrow 9s ease-in-out infinite}
.a7-fwd{animation:a7-fwdfade 9s ease-in-out infinite}
.a7-ckpt{animation:a7-modefade 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a7-g,.a7-survivor{animation:none;transform:scaleY(1)}.a7-fwd{animation:none;opacity:1}.a7-ckpt{animation:none;opacity:0}}
</style>
<line class="a7-axis" x1="60" y1="230" x2="690" y2="230"/>
<line class="a7-axis" x1="60" y1="40" x2="60" y2="230"/>
<text class="a7-lbl" x="32" y="135" transform="rotate(-90 32 135)">VRAM</text>
<rect class="a7-blk a7-g" x="90"  y="150" width="60" height="80"/>
<rect class="a7-blk a7-g a7-d2" x="170" y="150" width="60" height="80"/>
<rect class="a7-blk a7-g a7-d3" x="250" y="150" width="60" height="80"/>
<rect class="a7-blk a7-g a7-d4" x="330" y="150" width="60" height="80"/>
<rect class="a7-blk a7-g a7-d5" x="410" y="150" width="60" height="80"/>
<rect class="a7-blk a7-g a7-d6" x="490" y="150" width="60" height="80"/>
<rect class="a7-blk a7-g a7-d7" x="570" y="150" width="60" height="80"/>
<rect class="a7-keep a7-survivor" x="90" y="150" width="60" height="80"/>
<text class="a7-lbl" x="375" y="255">layer 1 . . . . . . . . . . . . . . . . . . . . . . . . . layer N</text>
<text class="a7-cap a7-fwd" x="375" y="30">forward: every layer keeps its activations resident</text>
<text class="a7-cap a7-ckpt" x="375" y="30">use_gradient_checkpointing = "unsloth": one block resident, rest offloaded to RAM</text>
</svg>
<figcaption>During the forward pass, activation memory grows layer by layer; offloaded checkpointing collapses it to roughly one block by streaming the rest to system RAM.</figcaption>
</figure>

Standard gradient checkpointing already trades compute for memory: instead of storing a block's inner activations, you store only its input and recompute the rest in backward. Unsloth's `Unsloth_Offloaded_Gradient_Checkpointer` goes one step further — during forward it copies the block's input hidden states to system RAM with a non-blocking async transfer (`hidden_states.to("cpu", non_blocking=True)`), and during backward it streams them back and recomputes. Because the copy is asynchronous it overlaps with compute, so the latency hit is small while the VRAM holding activations across the whole depth of the model collapses to roughly one block's worth. This is what `use_gradient_checkpointing="unsloth"` turns on, and it is the reason long-sequence fine-tunes that should OOM simply do not.

## 6. The zero-approximation principle

Here is the line that separates Unsloth from a long history of "go faster by being a little wrong." Every lever above is an **exact rewrite**. None of them changes what the model computes.

That claim deserves scrutiny, because it is the kind of thing libraries say loosely. Let us be precise about each lever:

- **Fused Triton kernels** compute the identical mathematical function as the eager PyTorch ops, just in one launch with fewer HBM round-trips. The forward RMSNorm produces the same normalized output bit-for-bit (modulo the floating-point reassociation any fusion incurs, which is the same order of perturbation as PyTorch's own kernels).
- **Hand-derived backprop** is the *analytic* gradient — literally the chain rule worked out on paper for the fused block. It is not a finite-difference approximation or a truncated series; it is exactly what autograd would compute, derived by hand so it can be executed with fewer saved tensors.
- **4-bit QLoRA** is the most likely to raise eyebrows, but the quantization is applied only to the *frozen* base weights, and the NF4 numerics match bitsandbytes' reference implementation. The trainable adapters are full precision. The loss landscape the optimizer sees is the QLoRA landscape — by design — and that landscape is what the QLoRA paper validated against full fine-tuning.
- **Fused cross-entropy** computes the exact same loss and gradient as `F.cross_entropy`; it just refuses to materialize the intermediate softmax tensor, recomputing it on the fly from the saved logsumexp.
- **Offloaded checkpointing** recomputes activations — the recomputation is the same forward pass, so the gradients are identical; only *when* and *where* the activations live changes.
- **8-bit/paged optimizers** are the one place a numerical choice is made, and it is bitsandbytes' block-wise 8-bit quantization of the Adam moments, which is well-studied to track fp32 Adam's trajectory closely. It is a memory format for the optimizer state, not a change to the update rule.

The practical consequence: a model fine-tuned with Unsloth should produce the same eval numbers as one fine-tuned with stock PyTorch + PEFT on the same data and seed (up to floating-point noise). The 2x speed and 70% memory savings are not bought with quality. That is the entire pitch, and it is why "no approximations" is repeated so often in Unsloth's own materials — it is the differentiator from the broad class of speedups that *do* approximate.

## 7. What Unsloth does NOT do

Knowing a library's boundaries is as important as knowing its tricks, because it tells you what you still have to think about.

**It does not write a custom attention kernel.** This surprises people, because attention is the famous bottleneck. But the famous bottleneck was *solved* — by [FlashAttention](https://arxiv.org/abs/2205.14135) and xformers, which already fuse the softmax and avoid materializing the `(seq × seq)` score matrix. Unsloth delegates attention to those libraries. Its win on the attention path comes from feeding them correctly-laid-out tensors and from the fused RoPE kernel that prepares Q and K, not from re-implementing the attention math. Trying to beat FlashAttention would be wasted effort; using it is the right engineering call.

**Its quantization is bitsandbytes NF4, not a bespoke scheme.** Unsloth binds bitsandbytes' CUDA dequantization functions directly (`cdequantize_blockwise_fp16_nf4` and friends) and wraps them in a fast dequantize-on-the-fly path. The 4-bit format, the double-quantization of the absmax scales, the block sizes — all of that is the QLoRA/bitsandbytes design. Unsloth's contribution is the *kernel that uses it efficiently inside the custom backward*, not a new number format.

**It supports multiple GPUs.** This is worth stating loudly because the old conventional wisdom — and some stale documentation, including older versions of the overview post — said Unsloth was single-GPU only. As of 2026 that is no longer true: multi-GPU training is available now, with a larger upgrade on the roadmap. If you read "Unsloth doesn't do multi-GPU" anywhere, it is out of date. Plan your scaling around current capability, not the 2024 limitation. For the broader landscape of multi-GPU training systems, the [DeepSpeed ZeRO and 3D parallelism deep-dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) covers the sharding strategies that complement what Unsloth does on a single device.

## 8. The real numbers

Headline numbers are only meaningful with the model and hardware they were measured on. Here are Unsloth's published 2026 figures, mapped to their context so you can find the row that matches your situation.

| Claim | Workload / model | Relative to | Notes |
|---|---|---|---|
| up to **2x faster**, up to **70% less VRAM** | general LoRA/QLoRA fine-tuning | Hugging Face + PEFT + FA2 | the flagship headline |
| **80% less VRAM** | GRPO (RL fine-tuning) | standard GRPO setup | RL keeps more state; the savings are even larger |
| **12x faster**, **35% less VRAM** | MoE LLM training | baseline MoE training | mixture-of-experts specific |
| **3x faster**, **30% less VRAM** | newer kernel + padding-free packing | prior Unsloth baseline | from the latest Triton kernels |
| gpt-oss-20B: **2x faster, 70% less** (SFT) | gpt-oss-20B supervised fine-tune | baseline | per-model published figure |
| gpt-oss-20B: **2x faster, 80% less** (GRPO) | gpt-oss-20B GRPO | baseline | RL again shows the bigger memory win |

Two things to read out of this table. First, the memory savings are *larger for RL* (GRPO) than for plain SFT — 80% vs 70% — because RL methods keep additional state (reference model logprobs, multiple generations) that the same levers also compress. Second, "2x faster" is the steady-state claim for ordinary fine-tuning; the dramatic "12x" is specific to the MoE path, where the routing and expert structure leave more on the table for a specialized implementation. Quote the row that matches your job, not the biggest number on the page.

## 9. How to read the rest of the series

This post was the map. Each lever has a dedicated deep-dive that opens the kernel, derives the math, and benchmarks the win in isolation. Read them in any order — but if you want the dependency-respecting path, follow the table top to bottom.

| Technique | What the post dissects | Slug |
|---|---|---|
| Fused Triton kernels | RMSNorm/SwiGLU fusion, one launch, minimal HBM | [unsloth-triton-kernel-fusion](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion) |
| Hand-derived backprop | custom `autograd.Function`, analytic gradients, in-place buffers | [unsloth-manual-backprop](/blog/machine-learning/open-source-library/unsloth-manual-backprop) |
| 4-bit NF4 quantization | NF4 format, double-quant, dequantize-on-the-fly | [unsloth-4bit-quantization-nf4](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4) |
| Fused cross-entropy | the softmax memory wall, logsumexp, chunked path | [unsloth-fused-cross-entropy](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy) |
| Gradient checkpointing + offload | recompute vs store, async CPU offload | [unsloth-gradient-checkpointing-offload](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload) |
| 8-bit / paged optimizers | block-wise 8-bit Adam, paging moments to CPU | [unsloth-8bit-paged-optimizers](/blog/machine-learning/open-source-library/unsloth-8bit-paged-optimizers) |
| RoPE & attention kernels | fused RoPE, delegating attention to FlashAttention | [unsloth-rope-attention-kernels](/blog/machine-learning/open-source-library/unsloth-rope-attention-kernels) |
| Long-context training | combining the levers to extend sequence length | [unsloth-long-context-training](/blog/machine-learning/open-source-library/unsloth-long-context-training) |
| VRAM budget & export | the full byte budget, merge/GGUF export | [unsloth-vram-budget-and-export](/blog/machine-learning/open-source-library/unsloth-vram-budget-and-export) |

### The call that ties it together

Every lever in this series is reachable from two functions. This is the real, current `FastLanguageModel` API — the kwargs and defaults match the source (`load_in_4bit=True` and `use_gradient_checkpointing="unsloth"` are the defaults that turn on levers 3 and 5 for free).

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048   # the from_pretrained default; raise it for long-context runs

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length  = max_seq_length,
    dtype           = None,            # None -> auto: bf16 on Ampere+, else fp16
    load_in_4bit    = True,            # lever 3: NF4 frozen base weights
    # use_gradient_checkpointing defaults to "unsloth" -> lever 5 (offloaded ckpt)
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,                # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,                # 0 is the optimized (and Unsloth-recommended) value
    bias           = "none",
    use_gradient_checkpointing = "unsloth",   # offloaded checkpointer
    random_state   = 3407,             # the Unsloth default seed
)

# From here, hand `model` to your usual TRL SFTTrainer / DPOTrainer.
# The fused kernels, manual backprop, and fused cross-entropy are patched in
# automatically the moment from_pretrained loaded the model.
```

Two lines, and every lever in this post is active. `from_pretrained` quantizes the base to NF4 (lever 3), wires the offloaded checkpointer (lever 5), and monkey-patches the fused kernels (lever 1), manual backprop (lever 2), and fused cross-entropy (lever 4) into the model's layers. `get_peft_model` attaches the LoRA adapters that make gradients and optimizer state tiny. From there, the model is a drop-in for [TRL](/blog/machine-learning/open-source-library/trl-lib)'s trainers — the speedup is invisible to your training loop, which is exactly the point. Note `lora_dropout=0`: Unsloth's fused path is optimized for the zero-dropout case, and it is also the value that best preserves the base model's behavior in the adapter, so it is the recommended default.

### When to reach for Unsloth — and when not to

**Reach for it** when you are fine-tuning a single model on one or a few GPUs and memory is the thing standing between you and the run: QLoRA on consumer or free-tier hardware, long-context SFT that OOMs under stock PEFT, RL fine-tuning where the extra state blows your budget, or any case where you want the 2x throughput without touching your training loop. The zero-approximation guarantee means you lose nothing in quality for the speed.

**Think twice** when your workload is dominated by something Unsloth deliberately does not own. If you are doing large-scale *pretraining* from scratch across dozens of nodes, the relevant machinery is 3D parallelism and sharded optimizers — that is DeepSpeed/Megatron/[TorchTitan](/blog/machine-learning/open-source-library/torchtitan-pytorch-native-pretraining-deep-dive) territory, and Unsloth's single-step kernels are not the bottleneck you are fighting. If your model architecture is exotic enough that the fused kernels do not have a patch for it, you fall back to eager PyTorch and lose the win. And if you are inference-bound rather than training-bound, the levers here (optimizer state, gradient memory, backward fusion) simply do not apply — you want a serving stack, not a fine-tuning stack.

The honest summary is narrow and strong: on the workload Unsloth targets — single-device parameter-efficient fine-tuning — it is close to a free lunch, because every gain traces to an exact rewrite of a cost you can name. That is the whole reason this series exists: not to admire the headline number, but to understand the ledger well enough that the number stops being magic.

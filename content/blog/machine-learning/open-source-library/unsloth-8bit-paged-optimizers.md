---
title: "8-bit and Paged Optimizers: Taming Adam's Memory in Unsloth"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "Adam keeps two fp32 state buffers per parameter — for full fine-tuning that exceeds the weights themselves and is the quietest VRAM hog of a step. Block-wise 8-bit optimizers quantize those buffers fourfold with no measurable accuracy loss, and paged optimizers spill state to CPU on a memory spike to survive OOM. Unsloth exposes both through one optim= string."
tags: ["unsloth", "optimizer", "adam", "8-bit-optimizer", "paged-optimizer", "bitsandbytes", "gpu-memory", "memory-optimization", "qlora", "quantization"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 28
---

There is a line item in every training step that nobody profiles until it kills a run, and it is not the weights, not the activations, and not the gradients. It is the optimizer state. The first time it bit me, I had carefully budgeted a full fine-tune: weights in fp16, gradients in fp16, activations checkpointed down to a sliver. The arithmetic said it would fit. It did not fit, by a wide margin, and the culprit was a pair of buffers I had mentally rounded to zero — Adam's first and second moments. Together they are eight bytes for every parameter in the model, and for a full fine-tune they are *larger than the model itself*.

This post is about the two moves that tame that hog, both of which Unsloth exposes through a single training-argument string. The first is the **8-bit optimizer**: quantize the moment buffers from fp32 to roughly one byte each, with no measurable accuracy loss. The second is the **paged optimizer**: when a memory spike threatens to OOM the card, spill optimizer state to CPU RAM over CUDA unified memory and page it back when the pressure clears. One fights the steady weight of optimizer state; the other fights the transient spikes that kill an otherwise-fitting run.

![What actually fills the card during full fine-tuning: Adam's two fp32 moment buffers (~64 GB for an 8B model) dwarf the weights, gradients, and checkpointed activations; the 8-bit optimizer shrinks that ~4x and the paged optimizer spills it to CPU on a spike.](/imgs/blogs/unsloth-8bit-paged-optimizers-1.webp)

The diagram above is the mental model for the whole post. On the left, the things we usually budget: fp16 weights, fp16 gradients, checkpointed activations. In the middle, the thing we don't: Adam's optimizer state, two fp32 buffers, at eight bytes per parameter — about 64 GB for an 8B model, four times the weights. On the right, the two escape hatches: quantize it to 8-bit (roughly 16 GB, a fourfold cut) or page it out to CPU on a spike.

One thing to be precise about up front, because it shapes how you reason about this post: **the 8-bit and paged optimizer machinery is bitsandbytes, not an Unsloth kernel.** Unsloth does not reimplement block-wise quantization or unified-memory paging; it *exposes* the bitsandbytes optimizers through the trainer's `optim=` string and wires them into the model it builds. So when this post explains the internals, it is explaining bitsandbytes (Dettmers et al., "8-bit Optimizers via Block-wise Quantization"); when it explains how you turn them on, it is explaining the Unsloth + TRL surface. Both halves matter, and conflating them is how people end up confused about what is actually doing the work.

This is part of the [Inside Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) series. The [speedup anatomy post](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) maps where every memory lever lands across a step; the [4-bit NF4 quantization post](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4) is the sibling that quantizes the *weights*, and this one quantizes the *optimizer state*; the [gradient-checkpointing and offload post](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload) is the sibling that attacks *activation* memory, with which this stacks cleanly. For the budget-level view of how it all adds up, see the [VRAM budget and export post](/blog/machine-learning/open-source-library/unsloth-vram-budget-and-export).

## 1. The quietest VRAM hog

Let us first say plainly what fills a GPU during training, because the optimizer's share is the one people consistently underestimate.

A training step holds, at minimum: the **weights** (the parameters being trained), the **gradients** (one per trainable parameter, produced by the backward pass), the **activations** (intermediate tensors saved during forward for the backward to consume), and the **optimizer state** (the running statistics a stateful optimizer keeps to compute its update). For mixed-precision training the weights and gradients are typically fp16 or bf16, two bytes each. The activations are the term everyone fixates on, because they scale with batch and sequence length, and they are the thing gradient checkpointing exists to shrink.

But the optimizer state is the term that scales with *the model*, persists for the *entire run*, and is held in *fp32*. For Adam — the default for essentially every LLM fine-tune — that state is two full-size buffers, the first and second moments, both fp32. That is eight bytes for every parameter, and unlike activations, it is never freed between steps: the moments are a running average, so they have to survive from one iteration to the next.

The consequence is stark. For an 8B-parameter model, full fine-tuning with fp32 Adam holds roughly:

| Component | Bytes / param | 8B model |
|---|---|---|
| Weights (fp16) | 2 | ~16 GB |
| Gradients (fp16) | 2 | ~16 GB |
| Optimizer state (Adam, fp32 m + v) | 8 | ~64 GB |
| Activations (checkpointed) | varies | a few GB |

The optimizer state alone is 64 GB — larger than the weights and gradients combined, and four times the model. It is the quietest hog because it does not move: it is allocated once, sits there for the whole run, and never appears in the per-step allocation churn that a profiler's timeline makes obvious. You only notice it when the initial allocation refuses to fit. That is exactly the failure that motivates both techniques in this post.

## 2. Why Adam costs eight bytes per parameter

To know what we are quantizing, recall what Adam actually stores. Adam (Kingma & Ba) is a stateful optimizer: it maintains, for each parameter $\theta_i$, two exponential moving averages computed from the gradient history.

The **first moment** $m$ is the EMA of the gradient — a smoothed estimate of the gradient's direction:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t.
$$

The **second moment** $v$ is the EMA of the squared gradient — a per-parameter estimate of the gradient's scale, used to adaptively normalize the step:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2.
$$

The update divides the bias-corrected first moment by the square root of the bias-corrected second moment:

$$
\theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}, \qquad \hat m_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat v_t = \frac{v_t}{1 - \beta_2^t}.
$$

![Why Adam costs eight bytes per parameter: each parameter carries a first-moment buffer m and a second-moment buffer v, both fp32 (4 bytes each), persistent across every step, feeding the adaptive update.](/imgs/blogs/unsloth-8bit-paged-optimizers-2.webp)

The figure above is the per-parameter picture. Each parameter $\theta$ (fp16, 2 bytes) is shadowed by two fp32 buffers — $m$ at 4 bytes and $v$ at 4 bytes — and both feed the update. That is the $4 + 4 = 8$ bytes per parameter, and it is fp32 for a reason: $v$ accumulates squared gradients, which span many orders of magnitude over training, and the division $\hat m / \sqrt{\hat v}$ is sensitive to precision in $v$ near zero. Naively halving the moments to fp16 destabilizes training; the second moment in particular underflows and the adaptive scaling goes haywire. So the textbook answer to "why is Adam so heavy" is: it keeps two persistent fp32 copies of the model, and you cannot trivially make them smaller without breaking the math.

The phrase "two persistent fp32 copies of the model" is the one to hold onto. The weights are one copy. The gradients are another. Adam adds *two more* — and at full precision. SGD-with-momentum keeps only the first moment, so it is half as heavy (4 bytes/param); plain SGD keeps none. But adaptive optimizers won the LLM era because they converge robustly without per-layer learning-rate tuning, and that robustness is bought with the second moment. The whole question this post answers is: can we keep Adam's behavior while paying less than fp32 for its state? The answer, from bitsandbytes, is yes — and it is not an approximation in the sense that matters.

## 3. What QLoRA already saves (and where it doesn't)

Before quantizing the optimizer, it is worth being honest about how much of this problem QLoRA already solves, because for many Unsloth users it solves most of it.

In a QLoRA fine-tune, the base model is frozen and quantized to 4-bit NF4 (the subject of the [4-bit quantization post](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4)). Only the LoRA adapters — small low-rank matrices injected into the attention and MLP projections — are trainable. The base weights carry **no gradient and no optimizer state**, because they are frozen. So the eight-bytes-per-parameter cost applies only to the adapter parameters, which for a rank-16 LoRA on an 8B model is on the order of tens of millions of parameters, not eight billion. The optimizer state for the adapters is a few hundred megabytes, not 64 GB.

This is why the speedup-anatomy post stresses that `get_peft_model` is what "makes gradients and optimizer state tiny." For the common case — QLoRA, modest rank, a single consumer card — the optimizer is genuinely not your bottleneck, and you can run 32-bit Adam on the adapters without a second thought.

So when does optimizer state still matter even with Unsloth?

- **Full fine-tuning** (`full_finetuning=True`): every parameter is trainable, so you are back to the full eight-bytes-per-param wall, 64 GB for an 8B model.
- **High LoRA rank**: rank scales the adapter parameter count linearly. A rank-256 adapter on every projection of a large model is no longer negligible, and its optimizer state grows with it.
- **Trainable embeddings and the LM head**: these are large matrices (vocab × hidden), and if you unfreeze them — common when adapting to a new vocabulary or domain — they dwarf the rest of the adapters. The embedding optimizer state alone can be gigabytes.
- **Spikes regardless of size**: even when the *steady* state fits, a long-sequence step or a large-gradient batch can momentarily push total allocation past capacity. That is the paged optimizer's domain, and it bites at any scale.

The table below previews how the two techniques interact with QLoRA — note that the absolute numbers shrink dramatically for adapters, but the *ratio* (8-bit being ~4× smaller) is the same.

| Config | Bytes/param | 8B full FT | QLoRA adapters |
|---|---|---|---|
| `adamw_32bit` | 8 | ~64 GB | ~0.3 GB |
| `adamw_8bit` | ~2 | ~16 GB | ~0.1 GB |
| `paged_adamw_8bit` | ~2 on GPU | ~16 GB + spill | ~0.1 GB |

## 4. 8-bit optimizers: block-wise quantization

Now the bitsandbytes machinery. An 8-bit optimizer stores the moment buffers as 8-bit values instead of 32-bit, a fourfold reduction, while preserving 32-bit optimization quality. The bitsandbytes implementation rests on three components: **block-wise quantization**, **dynamic quantization**, and a **stable embedding layer**.

### Block-wise quantization

You cannot quantize a whole tensor with a single scale and keep precision, because a single outlier sets the scale for everyone and crushes the small values to zero. Block-wise quantization sidesteps this by chunking the input tensor into small blocks of size $B = 2048$ and quantizing each block independently. For each block $b$, it computes a normalization constant — the block's absolute maximum:

$$
N_b = \max_{i \in b} |T_i|.
$$

Every value in the block is divided by $N_b$, mapping it into $[-1, 1]$, and then quantized. An outlier now only inflates the scale of *its own* 2048-element block, not the entire tensor, so error is isolated and distributed more evenly across all bits. And because blocks are independent, every block quantizes and dequantizes in parallel across cores — block-wise quantization is fast precisely because it has no cross-block dependency.

The storage cost of the per-block scale is trivial: one fp32 value per 2048 elements, which is $4 / 2048 \approx 0.002$ bytes per parameter — utterly negligible against the byte-per-value of the codes themselves.

![How bitsandbytes stores a moment buffer in 8 bits: the fp32 buffer splits into blocks of 2048, each block is normalized by its absmax then mapped to one of 256 dynamic-quantization codes, leaving 8-bit codes plus one fp32 scale per block.](/imgs/blogs/unsloth-8bit-paged-optimizers-3.webp)

The figure traces one block through the pipeline. The fp32 moment buffer is split into blocks of 2048; each block is normalized by its absmax $N_b$ into $[-1, 1]$; the normalized values are mapped to the nearest of 256 codes; what gets stored is the array of 8-bit codes (1 byte per value) plus the single fp32 block scale. That is the fourfold reduction made concrete.

Watching the same operation sweep across the buffer makes the block-independence vivid — each block is normalized and compressed on its own, so an outlier in one block never touches the codes of its neighbors:

<figure class="blog-anim">
<svg viewBox="0 0 720 240" role="img" aria-label="A sweep moves across blocks of an fp32 moment buffer, compressing each block into a small 8-bit code plus one block scale" style="width:100%;height:auto;max-width:820px">
<title>Each block of the moment buffer is quantized from fp32 to 8-bit codes plus one scale</title>
<style>
.q1-blk{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5;rx:6}
.q1-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.q1-big{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.q1-code{fill:var(--accent,#6366f1);rx:5}
.q1-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.q1-sweep{fill:var(--accent,#6366f1);opacity:.16;rx:8}
@keyframes q1-move{0%{transform:translateX(0)}100%{transform:translateX(450px)}}
@keyframes q1-shrink0{0%,8%{opacity:0;transform:scaleY(1)}20%,100%{opacity:1}}
@keyframes q1-shrink1{0%,28%{opacity:0}40%,100%{opacity:1}}
@keyframes q1-shrink2{0%,48%{opacity:0}60%,100%{opacity:1}}
@keyframes q1-shrink3{0%,68%{opacity:0}80%,100%{opacity:1}}
.q1-anim{animation:q1-move 8s steps(1,end) infinite}
.q1-c0{animation:q1-shrink0 8s ease-out infinite}
.q1-c1{animation:q1-shrink1 8s ease-out infinite}
.q1-c2{animation:q1-shrink2 8s ease-out infinite}
.q1-c3{animation:q1-shrink3 8s ease-out infinite}
@media (prefers-reduced-motion:reduce){.q1-anim{animation:none;opacity:.16}.q1-c0,.q1-c1,.q1-c2,.q1-c3{animation:none;opacity:1}}
</style>
<text class="q1-big" x="360" y="36">fp32 moment buffer, blocks of B = 2048</text>
<rect class="q1-blk" x="40"  y="60" width="140" height="70"/>
<rect class="q1-blk" x="200" y="60" width="140" height="70"/>
<rect class="q1-blk" x="360" y="60" width="140" height="70"/>
<rect class="q1-blk" x="520" y="60" width="140" height="70"/>
<text class="q1-lbl" x="110" y="100">fp32 x2048</text>
<text class="q1-lbl" x="270" y="100">fp32 x2048</text>
<text class="q1-lbl" x="430" y="100">fp32 x2048</text>
<text class="q1-lbl" x="590" y="100">fp32 x2048</text>
<rect class="q1-sweep q1-anim" x="40" y="56" width="140" height="78"/>
<rect class="q1-code q1-c0" x="60"  y="160" width="100" height="44"/>
<rect class="q1-code q1-c1" x="220" y="160" width="100" height="44"/>
<rect class="q1-code q1-c2" x="380" y="160" width="100" height="44"/>
<rect class="q1-code q1-c3" x="540" y="160" width="100" height="44"/>
<text class="q1-big" x="110" y="188" style="fill:#fff">8-bit + N_b</text>
<text class="q1-big" x="270" y="188" style="fill:#fff">8-bit + N_b</text>
<text class="q1-big" x="430" y="188" style="fill:#fff">8-bit + N_b</text>
<text class="q1-big" x="590" y="188" style="fill:#fff">8-bit + N_b</text>
<text class="q1-cap" x="360" y="228">normalize by block absmax, map to 8-bit codes: ~4x smaller, one scale per block</text>
</svg>
<figcaption>The sweep visits each 2048-value block in turn; each fp32 block collapses to an 8-bit code array plus a single block scale N_b. The moment buffer shrinks roughly fourfold with no per-block outlier bleeding into its neighbors.</figcaption>
</figure>

### Dynamic quantization

The 256 codes are not evenly spaced. Even spacing would waste precision: optimizer moments are concentrated near zero with a long tail of occasional large values, so a linear grid spends most of its codes on magnitudes that rarely occur. Bitsandbytes uses **dynamic tree quantization**, a data type with a *dynamic* split between exponent and fraction that varies per value — giving low relative error for both small and large magnitudes. This is what lets 8 bits stand in for 32: the codes are placed where the data actually lives, so the quantization error stays below the threshold where it would perturb the optimizer trajectory.

### The quantize-update-dequantize cycle

The elegant part is how the update is performed without ever materializing a full fp32 copy of the state. Each optimizer step:

1. **Dequantizes** the 8-bit state to fp32 — but element-by-element, **in registers**, not as a bulk copy to a temporary fp32 tensor in HBM.
2. **Applies** the Adam update in fp32 (the math from section 2, at full precision).
3. **Requantizes** the updated state back to 8-bit for storage.

Because the dequant→update→requant happens per element in registers, there is no slow round-trip to GPU memory and no large temporary allocation. This is why bitsandbytes 8-bit optimizers are not only smaller but often *faster* than 32-bit Adam on GPU: the state is a quarter the size, so there is a quarter of the memory traffic, and the arithmetic happens where it is cheap.

### The stable embedding layer

The third component is a stability aid rather than a memory trick. Word-embedding layers have notoriously non-uniform input distributions (a few tokens dominate), which produces extreme gradient variation that aggressive quantization can amplify into instability. The stable embedding layer uses Xavier-uniform initialization, layer-norm before adding positional embeddings, and — critically — keeps **32-bit optimizer states for the embeddings specifically**, while the rest of the model uses 8-bit. It is a targeted exception: spend full precision only where it buys stability.

### Why accuracy holds

The headline claim from the bitsandbytes paper is that 8-bit optimizers **match 32-bit performance with no changes to the original optimizer hyperparameters**, across a wide spread of tasks — 1.5B-parameter language modeling, GLUE fine-tuning, ImageNet classification, WMT'14 machine translation, MoCo v2, RoBERTa pretraining. The reported memory savings are real and large: 8.5 GB freed on the 1.5B language model, 2.0 GB on RoBERTa-Large, 1.7 GB on GPT3-Medium. The reason accuracy holds is that block-wise normalization plus dynamic quantization keeps the per-element quantization error small *relative to the value*, and Adam's update is robust to small per-step perturbations of its state — the moments are EMAs, so transient quantization noise averages out rather than accumulating. It is a drop-in replacement; the paper frames it as a two-line code change.

## 5. Paged optimizers: spill to CPU on a spike

The 8-bit optimizer fights the *steady* weight of optimizer state. Paged optimizers fight a different enemy: the *transient spike*. Even when your steady-state memory fits, a single unusually long sequence, or a batch with an unusually large gradient, can momentarily push total allocation past the card's capacity and OOM the run. A paged optimizer survives that spike by spilling optimizer state to CPU RAM exactly when — and only when — the GPU runs out.

Paged optimizers are built on **CUDA unified memory**. Unified memory provides a single address space that both the GPU and CPU can access; the CUDA driver migrates pages between the two on demand. PyTorch does not expose this, but bitsandbytes does, and it uses it to back the optimizer state with managed memory.

The mechanism works like operating-system paging. Optimizer state pages are allocated as managed memory, pre-mapped on the CPU. As long as everything fits on the GPU, the pages live in VRAM and there is **zero overhead** — the optimizer behaves exactly like a non-paged one. When a spike exhausts VRAM, the driver evicts pages of optimizer state to the pre-mapped CPU RAM, page by page, freeing room for the spike. The pages are not updated automatically; they move only when accessed or when a swap is launched. When the spike clears and the optimizer needs those pages again, they migrate back to the GPU on access.

![Paged optimizer: on a memory spike the CUDA driver evicts an optimizer page to pre-mapped CPU RAM, freeing VRAM; the page migrates back to the GPU when accessed after the spike clears.](/imgs/blogs/unsloth-8bit-paged-optimizers-4.webp)

The figure shows the spike-and-recover cycle. The GPU is full — an activation spike from a long-sequence step is consuming VRAM. Optimizer page C, which is not needed during the spike, is evicted across unified memory to its pre-mapped home in CPU RAM, making room. Pages A and B, still in use, stay on the GPU. When the spike clears and the optimizer step touches page C, it pages back. The whole dance is driven by the CUDA driver, not by your code.

<figure class="blog-anim">
<svg viewBox="0 0 720 260" role="img" aria-label="On a VRAM spike an optimizer page slides from GPU memory to CPU RAM, then pages back when the spike clears" style="width:100%;height:auto;max-width:820px">
<title>A paged optimizer evicts a page to CPU on a spike and pages it back afterward</title>
<style>
.pg-strip{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5;rx:8}
.pg-hdr{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.pg-lbl{font:600 13px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.pg-cap{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.pg-page{fill:var(--accent,#6366f1);rx:6}
.pg-spike{fill:var(--text-secondary,#9ca3af);opacity:.0;rx:6}
@keyframes pg-spike{0%,12%{opacity:0}24%,60%{opacity:.55}72%,100%{opacity:0}}
@keyframes pg-evict{0%,18%{transform:translate(0,0)}40%,66%{transform:translate(430px,0)}88%,100%{transform:translate(0,0)}}
.pg-spike-a{animation:pg-spike 9s ease-in-out infinite}
.pg-move{animation:pg-evict 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.pg-spike-a{animation:none;opacity:0}.pg-move{animation:none;transform:none}}
</style>
<text class="pg-hdr" x="160" y="34">GPU VRAM</text>
<text class="pg-hdr" x="560" y="34">CPU RAM (mapped)</text>
<rect class="pg-strip" x="30"  y="60" width="300" height="150"/>
<rect class="pg-strip" x="400" y="60" width="290" height="150"/>
<rect class="pg-spike pg-spike-a" x="48" y="78" width="264" height="60"/>
<text class="pg-cap" x="180" y="112">activation spike fills VRAM</text>
<rect class="pg-page pg-move" x="54" y="150" width="120" height="48"/>
<text class="pg-lbl pg-move" x="114" y="180">optim page C</text>
<text class="pg-cap" x="360" y="244">unified memory: page C evicts to CPU on the spike, returns when VRAM frees</text>
</svg>
<figcaption>When the activation spike fills VRAM, the CUDA driver evicts optimizer page C across unified memory to pre-mapped CPU RAM; once the spike clears the page returns to the GPU. Paging activates only on pressure, so steady-state steps pay nothing.</figcaption>
</figure>

The performance characteristics are worth understanding so you set expectations correctly. Unified-memory transfers are less efficient than explicit asynchronous copies — you typically get about half of PCIe bandwidth or worse. As a rough estimate from the bitsandbytes docs: evicting 1 GB per forward-backward-optimizer loop on PCIe 3.0 × 16 lanes (≈16 GB/s) at ~50% efficiency costs roughly $1 / (16 \times 0.5) = 125$ ms of overhead per step. That is real, but it is overhead you pay *only on the steps that spike*; steps that fit pay nothing.

This is the key contrast with **explicit CPU offloading** — the approach the [gradient-checkpointing and offload post](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload) describes for activations, where Unsloth deliberately moves a block's input to CPU every forward and streams it back every backward. Explicit offloading moves a *fixed* chunk every single iteration, whether or not you needed to. A paged optimizer has **zero overhead when everything fits** and pays only for what actually gets evicted under pressure. They are complementary tools: offload the activations you know you cannot afford every step; page the optimizer state you can usually afford but occasionally cannot. Use both and a spike that would have OOM'd instead degrades gracefully into a slightly slower step.

Now, the quantization in `paged_adamw_8bit` and the paging are orthogonal. The "8-bit" part shrinks the steady footprint fourfold; the "paged" part adds spike survival on top. You can have either alone (`adamw_8bit` is 8-bit without paging; `paged_adamw_32bit` is paging without quantization) or both together. They compose.

## 6. Turning it on

Here is where the Unsloth surface comes back in. You do not call bitsandbytes directly; you choose an optimizer string, and Unsloth plus TRL wire it through.

The optimizer is selected by the `optim=` argument of the trainer's config — a string that HuggingFace's `Trainer` / TRL's `SFTTrainer` maps to the corresponding bitsandbytes optimizer. The three strings relevant to this post:

- `optim="adamw_8bit"` — block-wise 8-bit AdamW, no paging.
- `optim="paged_adamw_8bit"` — 8-bit AdamW with paged (unified-memory) state.
- `optim="paged_adamw_32bit"` — full fp32 AdamW state, but paged for spike survival.

A complete Unsloth fine-tune that turns this on looks like the following. The model setup is the real, current `FastLanguageModel` API (the kwargs and defaults match the source); the optimizer choice is one line in the training config.

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# 1. Load the base model. load_in_4bit=True freezes + quantizes the base to NF4,
#    so only the adapters will carry optimizer state.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name        = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length    = 2048,
    load_in_4bit      = True,                 # QLoRA base
    use_gradient_checkpointing = "unsloth",   # offloaded checkpointing (stacks with paging)
)

# 2. Attach LoRA adapters — these are the trainable params whose optimizer state we shrink.
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,
    use_gradient_checkpointing = "unsloth",
    random_state   = 3407,
)

# 3. Choose the optimizer with one string. This is the entire lever.
trainer = SFTTrainer(
    model     = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps      = 500,
        learning_rate  = 2e-4,
        optim          = "paged_adamw_8bit",   # <-- 8-bit moments + spill on spike
        logging_steps  = 1,
        output_dir     = "outputs",
    ),
)
trainer.train()
```

Two lines of model setup, one string in the config, and the optimizer state is both quantized and spike-proof. Nothing else in your training loop changes — `SFTTrainer.train()` does not know or care that the moments are 8-bit and paged. This is the same patch-don't-fork philosophy the rest of the series describes: Unsloth builds the model, hands it to TRL's standard trainer, and the trainer routes the `optim=` string to bitsandbytes. The lever is reachable from the API you already use.

A practical note on stacking: `use_gradient_checkpointing="unsloth"` (the offloaded checkpointer, on by default) attacks activation memory, while `optim="paged_adamw_8bit"` attacks optimizer memory. They target different terms in the budget, so they add rather than conflict — and both being spill-to-CPU mechanisms, you want enough system RAM to back both. On a memory-tight single-GPU box, this combination is what lets a fine-tune that "should" need 40 GB run on 24.

## 7. Memory accounting

Let us make the savings concrete with the numbers you would actually compute when budgeting a run. The optimizer-state cost is simply (trainable parameter count) × (bytes per parameter), where bytes per parameter is 8 for fp32 Adam and roughly 2 for 8-bit Adam (1 byte per moment, plus the negligible per-block scale).

![Optimizer state across configurations: adamw_32bit at 8 bytes per parameter, adamw_8bit at ~2, and paged variants, shown for 8B full fine-tuning versus QLoRA adapters; 8-bit is ~4x smaller and QLoRA shrinks how many parameters carry state at all.](/imgs/blogs/unsloth-8bit-paged-optimizers-5.webp)

| Configuration | Bytes / param | 8B full FT | QLoRA adapters (r=16) |
|---|---|---|---|
| `adamw_32bit` | 8 | ~64 GB | ~0.3 GB |
| `adamw_8bit` | ~2 | ~16 GB | ~0.1 GB |
| `paged_adamw_8bit` | ~2 on GPU | ~16 GB + spill | ~0.1 GB |
| `paged_adamw_32bit` | 8 on GPU | ~64 GB + spill | ~0.3 GB |

Read the two columns as two different worlds. In **full fine-tuning** (left), the optimizer state is the dominant term and 8-bit's fourfold cut is enormous — 64 GB down to 16 GB frees half a high-end card. In **QLoRA** (right), the adapter parameter count is so small that even fp32 Adam is a few hundred megabytes; 8-bit still helps, but the absolute saving is small because QLoRA already removed the eight billion frozen parameters from the optimizer's books. The two techniques compound: QLoRA cuts *how many* parameters carry state, and 8-bit cuts *how many bytes* each one costs.

The paged rows look identical in steady-state footprint to their non-paged counterparts, because that is the point — paging adds no resident cost when everything fits. The "+ spill" is what happens only during a spike: pages temporarily live in CPU RAM instead of OOMing. So the right way to read `paged_adamw_8bit` is "the same 16 GB as `adamw_8bit`, but it survives the step where 16 GB momentarily wasn't available."

One more accounting subtlety: the 8-bit saving is on the *optimizer state only*. It does not touch the weights (that is NF4's job) or the gradients (which remain fp16/bf16) or the activations (checkpointing's job). So in a full budget, 8-bit Adam removes ~48 GB from the 64 GB optimizer line, leaving the other lines unchanged. Knowing which lever moves which line is the difference between a budget that fits and a guess that OOMs.

To see the compounding in numbers, work a concrete QLoRA example end to end. Take an 8B model, rank-16 LoRA on the seven projections (`q,k,v,o,gate,up,down`). The trainable adapter count is roughly $r \times (\text{in} + \text{out})$ summed over each adapted matrix; for a model of this size it lands around 40 million trainable parameters. The optimizer state is then:

- `adamw_32bit`: $40\text{M} \times 8\text{ B} = 320\text{ MB}$.
- `adamw_8bit`: $40\text{M} \times 2\text{ B} = 80\text{ MB}$.

Either fits trivially — which is exactly why, for steady-state QLoRA, the optimizer is not your bottleneck and the choice barely registers on the budget. Now flip to full fine-tuning of the same 8B model: the trainable count jumps from 40M to 8B, a 200× increase, and the same per-parameter bytes now read $8\text{B} \times 8\text{ B} = 64\text{ GB}$ versus $8\text{B} \times 2\text{ B} = 16\text{ GB}$. The *ratio* is identical (4×), but the *absolute* saving exploded from 240 MB to 48 GB because the parameter count it multiplies against grew 200-fold. This is the single most important intuition for budgeting: 8-bit's percentage saving is constant, but its dollar value tracks the trainable parameter count — which is precisely the quantity QLoRA collapses. Stack them and you get a small fraction of a small fraction; that is how an 8B fine-tune fits where the naive arithmetic says it cannot.

The gradient line deserves a footnote because people sometimes assume 8-bit optimizers shrink it too. They do not. Gradients are produced fresh each backward pass and consumed by the optimizer step; bitsandbytes quantizes the persistent *state* (the moments), not the transient gradient. The gradient stays at the model's compute dtype (fp16/bf16), so in the full-FT budget the 16 GB gradient line is untouched by `adamw_8bit`. If you need to attack that line too, the levers are smaller batch / gradient accumulation (which trades step count for peak gradient memory) — a different knob entirely.

## 8. When to use which

The decision is not "always use the smallest" — it is "match the tool to the failure you are fighting."

![Choosing the optim= string: if optimizer state is too big for VRAM, steady pressure points to adamw_8bit, occasional spikes to paged_adamw_8bit, and a need for exact fp32 state plus spike safety to paged_adamw_32bit.](/imgs/blogs/unsloth-8bit-paged-optimizers-6.webp)

The figure is the decision in one glance. The root question is whether optimizer state is too big for VRAM. If you face **steady pressure every step** — the state simply does not fit at fp32 — reach for `adamw_8bit`: a fourfold cut with no measurable accuracy loss and often a speedup, the right default for full fine-tuning or high-rank LoRA on a tight card. If you face **occasional spikes** — the steady state fits but a long-sequence or large-gradient step occasionally OOMs — reach for `paged_adamw_8bit`: it spills on the spike and pays nothing the rest of the time. If you need **exact fp32 optimizer state** for some numerically delicate reason *and* want spike safety, `paged_adamw_32bit` gives you paging without quantization, at the full eight bytes per parameter.

The interaction with gradient checkpointing is additive, as the bottom of the figure notes: checkpointing (`use_gradient_checkpointing="unsloth"`) cuts the *activation* term, the 8-bit/paged optimizers cut the *optimizer-state* term. They attack different lines of the budget, so you stack them. The general recipe for a memory-tight single-GPU fine-tune: QLoRA base (cuts weights), Unsloth offloaded checkpointing (cuts activations), and `paged_adamw_8bit` (cuts and spike-proofs optimizer state). Each lever owns a different term, and together they are why a model that looks too big for the card fits anyway.

A few cautions. The 8-bit optimizer's accuracy guarantee is empirical and broad, but it is "matches 32-bit," not "bit-identical to 32-bit" — there is a tiny per-step quantization noise that the EMA absorbs. For the overwhelming majority of fine-tunes this is invisible; if you are doing something numerically delicate (very small learning rates, very long runs where tiny biases could accumulate), the conservative choice is `paged_adamw_32bit`. Paging needs system RAM to spill into; if you are short on host RAM, paging cannot help and will itself fail. And paging's overhead, while only paid on spikes, is real — if *every* step spikes, you are effectively running a slow offloaded optimizer and should rethink the batch/sequence configuration rather than relying on paging to paper over it.

## 9. War stories

The shapes this problem takes in practice, and how the right `optim=` string changes the outcome.

**1. The full fine-tune that wouldn't allocate.** An engineer tried to full-fine-tune an 8B model on a single 80 GB card. Weights (16 GB) + gradients (16 GB) + activations fit comfortably, but the run failed at optimizer initialization with a 64 GB allocation request. fp32 Adam's two moment buffers were the entire problem. Switching `optim="adamw_32bit"` to `optim="adamw_8bit"` dropped the optimizer state to ~16 GB and the run allocated with room to spare. Nothing else changed; the loss curve was indistinguishable from a 32-bit baseline on a smaller model.

**2. The long-sequence spike that OOM'd one batch in fifty.** A QLoRA run with variable-length sequences trained fine for forty-odd steps, then died on a step that happened to pack the longest examples. Steady-state memory fit; the activation spike on that one step did not. The fix was `paged_adamw_8bit`: on the spike, optimizer pages evicted to CPU, the long step completed ~100 ms slower, and the run continued. The engineer's instinct had been to lower the max sequence length for *every* step, which would have hurt the model; paging let the rare long step survive without penalizing the common case.

**3. The "8-bit ruined my accuracy" false alarm.** Someone reported that `adamw_8bit` degraded their eval score versus `adamw_32bit`. The real cause was an unrelated change to the learning-rate schedule made in the same commit. When isolated, the 8-bit and 32-bit runs matched within noise — exactly as the bitsandbytes paper claims across GLUE, ImageNet, and translation. The lesson: 8-bit Adam is a drop-in with no hyperparameter changes; if accuracy moves, suspect the thing you changed alongside it.

**4. Trainable embeddings blowing the budget.** A team adapting a model to a new language unfroze the embedding and LM-head matrices (vocab × hidden — large). Their optimizer state ballooned because those big matrices now carried fp32 moments. `adamw_8bit` cut it fourfold, and the stable-embedding component of bitsandbytes kept the embedding training stable by holding *those* states at 32-bit while the rest of the model went 8-bit — a built-in targeted exception they did not have to configure.

**5. Paging with no RAM to page into.** An engineer enabled `paged_adamw_8bit` on a box with 16 GB of system RAM and a full-FT optimizer that needed to spill more than that on spikes. Paging cannot conjure RAM; the spill itself failed. Paging is a CPU-RAM-for-VRAM trade, and it requires the host side to actually have the room. The fix was either more system RAM or dropping to QLoRA so the optimizer state was small enough not to need a large spill.

**6. Choosing 32-bit-paged for a delicate run.** A researcher running a very long, very-low-learning-rate continued-pretraining job wanted maximum numerical fidelity but also occasionally spiked. `adamw_8bit` would almost certainly have been fine, but the conservative choice — and the one they made — was `paged_adamw_32bit`: exact fp32 moments, plus spike survival via paging. It paid the full eight bytes per parameter in steady state but never quantized the state, removing even the tiny EMA-absorbed noise from the equation.

**7. Stacking all three levers on a 24 GB card.** The canonical Unsloth single-GPU recipe: a QLoRA fine-tune of an 8B model on a 24 GB consumer card with NF4 weights, `use_gradient_checkpointing="unsloth"`, and `paged_adamw_8bit`. NF4 cut the weights to ~5 GB, offloaded checkpointing kept activations to roughly one block's worth, and the adapter optimizer state in 8-bit was a few hundred megabytes — with paging as the safety net for the occasional long batch. Each lever owned a different line of the budget; together they turned "needs an A100" into "runs on a 4090."

## Takeaways

The optimizer state is the quietest VRAM hog because it scales with the model, persists for the whole run, and hides in fp32 outside the per-step allocation churn. Adam's two moment buffers are eight bytes per parameter — for full fine-tuning, larger than the model.

Two bitsandbytes techniques tame it, and Unsloth exposes both through one `optim=` string. **8-bit optimizers** quantize the moment buffers block-wise (blocks of 2048, per-block absmax normalization, dynamic-tree codes), shrinking the state fourfold with no measurable accuracy loss and often a speedup, because the update is done element-by-element in registers with no HBM round-trip. **Paged optimizers** back the state with CUDA unified memory and spill it to CPU only on a spike, with zero overhead when everything fits — the opposite tradeoff from fixed CPU offloading.

Match the tool to the failure: `adamw_8bit` for steady pressure, `paged_adamw_8bit` for occasional spikes, `paged_adamw_32bit` when you need exact fp32 state plus spike safety. Stack them with QLoRA (cuts how many parameters carry state) and gradient checkpointing (cuts activations), and a model that looks too big for the card fits anyway. The throughline of the [Inside Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) series holds here too: know which lever moves which line of the budget, and the wall turns out to be optional.

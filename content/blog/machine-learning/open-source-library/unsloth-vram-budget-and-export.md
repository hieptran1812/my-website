---
title: "The Full VRAM Budget and Export: Assembling Everything Unsloth Saves"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "A byte-by-byte VRAM budget for one concrete 8B QLoRA run, showing how each Unsloth lever stacks up to fit a 15 GB T4 — then how to merge adapters and export to merged-16bit, merged-4bit, or GGUF for llama.cpp and Ollama."
tags: ["unsloth", "vram", "qlora", "gguf", "model-export", "quantization", "llama-cpp", "deployment", "lora", "ollama"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 27
---

The first post in this series, [the speedup anatomy](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy), opened a question and deliberately left it hanging: *where does the memory actually go, byte by byte, and how does each Unsloth lever buy back a piece of it?* We have spent the series taking those levers apart one at a time — the [4-bit NF4 weights](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4), the [fused Triton kernels](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion), the [hand-derived backward pass](/blog/machine-learning/open-source-library/unsloth-manual-backprop), the [fused cross-entropy](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy), the [offloaded checkpointing](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload), the [8-bit paged optimizer](/blog/machine-learning/open-source-library/unsloth-8bit-paged-optimizers). This capstone closes the loop. We assemble the complete budget for one real run, watch it fit under a card you can rent for pennies an hour, and then follow the model out the door — merging the adapter back into the weights and exporting to whatever your deployment target speaks.

![The same QLoRA run with Unsloth fits a T4, without it needs an A100 cluster: each lever shrinks one line of the budget so an 8B fine-tune drops from ~100 GB to ~8.8 GB.](/imgs/blogs/unsloth-vram-budget-and-export-1.webp)

The diagram above is the whole post in one frame. On the left, the naive fp16 full-finetune path: weights, optimizer state, gradients, and activations stacking past a hundred gigabytes — far over the 15 GB line a T4 gives you. On the right, the same model and the same fine-tune through Unsloth: five levers, each shrinking one line, summing to roughly 8.8 GB. The point of the figure — and of this post — is that the "70% less VRAM" headline is not one magic trick. It is five specific, separable reductions that stack, and once you can name each line of the budget you can predict whether *your* run will fit before you start it.

## 1. The payoff: a budget you can read

Most "how much VRAM do I need?" advice is a shrug and a rule of thumb. That is unsatisfying because the honest answer is a sum, and every term in the sum has a name and a lever attached. By the end of Section 3 you will be able to write that sum down for any model, and by the end of the post you will be able to take the trained result and ship it.

Here is the thesis stated plainly. A training step holds, simultaneously, six categories of bytes on the GPU: the model **weights**, the **LoRA adapters**, the **gradients**, the **optimizer state**, the **activations** saved for backward, and the **logits/loss** working set — plus a roughly fixed **CUDA context and framework overhead**. Generic full fine-tuning pays full freight on every one. Unsloth's bet, post by post, has been to attack each category with a technique that is *exact* — no approximation, identical numbers — but cheaper in memory. The budget is where those bets get cashed.

We close the loop opened in the speedup-anatomy post by making its abstract claim concrete: that post said the wins come from quantization, fusion, manual gradients, and offload; this post puts a gigabyte figure on each and adds them up under a hard ceiling. Then it does the thing that post explicitly deferred — what happens *after* the loss curve flattens.

## 2. The worked example

Let us pin everything to one concrete run so the numbers are real, not illustrative.

- **Model:** Llama-3.1-8B (8.03 billion parameters, 32 layers, hidden 4096, intermediate 14336, GQA with 8 KV heads, vocab 128256).
- **Method:** QLoRA — the 4-bit-quantized base with LoRA adapters — at rank `r = 16`, `lora_alpha = 16`, adapters on the seven projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Sequence length:** 2048. **Per-device batch size:** 1 (gradient accumulation handles effective batch).
- **Hardware:** a single NVIDIA T4 — 16 GB physical, about **15 GB usable** after the driver and CUDA context take their cut. This is the free-tier Colab card, which is exactly why "does it fit a T4?" is the question that matters.

The pipeline that produces this run is short, and every line is accurate to Unsloth's public API:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch

# 1. Load the base model already 4-bit quantized, with the offloaded checkpointer.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name        = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length    = 2048,
    dtype             = None,          # auto: bf16 on Ampere+, fp16 on T4
    load_in_4bit      = True,          # the NF4 base weights live here
)

# 2. Attach LoRA adapters — only these tensors will carry gradients.
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    lora_alpha     = 16,
    lora_dropout   = 0,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",   # offloaded checkpointing
    random_state   = 3407,
)

# 3. Train with TRL's SFTTrainer; the 8-bit paged optimizer keeps state tiny.
trainer = SFTTrainer(
    model     = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        max_steps         = 500,
        learning_rate     = 2e-4,
        optim             = "paged_adamw_8bit",   # 8-bit Adam, paged to host
        bf16              = False, fp16 = True,    # T4 has no bf16
        logging_steps     = 1,
    ),
)
trainer.train()
```

Note `use_gradient_checkpointing = "unsloth"` (the offloaded checkpointer, not the stock `True`) and `optim = "paged_adamw_8bit"`. Those two strings, plus `load_in_4bit = True`, are three of the five levers, set by configuration rather than code. The other two — fused kernels and the hand-derived backward — are automatic the moment you load through `FastLanguageModel`. Now let us account for what that run actually holds.

## 3. The budget, line by line

![The QLoRA VRAM ledger, line by line: each line item is shrunk by a specific Unsloth lever, and the right column names the lever that pays for it.](/imgs/blogs/unsloth-vram-budget-and-export-2.webp)

The ledger figure is the budget; this section justifies each row. I will give the naive cost, the Unsloth cost, and the lever in between.

### Base weights — ~5.0 GB (4-bit NF4)

The 8.03B parameters are the largest single line. In fp16 they would be `8.03e9 × 2 bytes ≈ 16.1 GB` — already over the T4 ceiling before a single activation exists. Unsloth loads them as 4-bit NF4: roughly `8.03e9 × 0.5 bytes ≈ 4.0 GB` of packed weights, plus the per-block `absmax` scales. With double quantization (the absmax scales themselves quantized to 8-bit), the scale overhead is small, and the embedding and LM head — which Unsloth can keep in higher precision — round the total to about **5.0 GB**. The mechanism, including how the NF4 codebook and the two-level block scales work, is the subject of the [4-bit NF4 post](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4). For the budget, the line is: 16 GB becomes 5 GB, and these weights are *frozen* — they carry no gradient and no optimizer state.

### LoRA adapters + their gradients — ~0.1 GB (fp16)

Only the adapters train. Each LoRA module adds an `A` of shape `(in, r)` and a `B` of shape `(r, out)`. At `r = 16` across the seven target modules and 32 layers, the adapter parameter count lands in the low tens of millions — on the order of 20–40M parameters depending on which exact modules you target — call it ~0.05 GB in fp16. Their gradients are the same shape, another ~0.05 GB. Together, **~0.1 GB**. This is the LoRA lever itself: by freezing the 16 GB of base weights and training only a sliver, the gradient line collapses from "as big as the model" to "as big as a rounding error." Contrast the naive full fine-tune, where gradients are `8.03e9 × 2 = 16 GB` — a line item that, on its own, blows the T4 budget.

### Optimizer state — ~0.05 GB (8-bit Adam, adapters only)

Adam keeps two moments (first and second) per trainable parameter. For a full fp32 Adam over all 8.03B parameters that is `8.03e9 × 2 moments × 4 bytes = 64.2 GB` — the single biggest line in the naive budget, larger than the weights themselves. This is the line that surprises people most, because it is invisible in a forward-only mental model: you think "16 GB of weights, that should fit a 24 GB card," and then Adam quietly demands four times the weight footprint for its moment estimates. Two levers crush it. First, because only the adapters train, the state covers tens of millions of parameters, not eight billion — the moments exist only for the `A` and `B` matrices. Second, the [8-bit paged optimizer](/blog/machine-learning/open-source-library/unsloth-8bit-paged-optimizers) stores each moment in 8 bits with block-wise scales and pages cold state to host RAM, so even that small footprint is further quartered and partly offloaded. The product: **~0.05 GB**, against 64 GB. If you remember one number from the naive column, remember 64 GB of Adam state — it is why full fine-tuning an 8B model is an A100-80GB job and QLoRA is a laptop job, and it is why "use LoRA" is the first thing anyone says when you mention a memory wall.

### Activations — ~1.5 GB (one block resident)

Activations are the bytes saved during the forward pass so the backward pass can compute gradients. Naively they scale with batch × seq-len × hidden × depth, all at once, and for a 32-layer model at seq 2048 they are several gigabytes and the term that grows fastest as you push context. Unsloth's [offloaded gradient checkpointing](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload) recomputes each block's internal activations in backward instead of storing them, *and* streams each block's input hidden-state to host RAM with a non-blocking copy, so the resident activation memory collapses to roughly one block's worth. At seq 2048, batch 1, that is on the order of **~1.5 GB**. This is the lever with the most dial-turning: it trades a little recompute (one extra forward per block in backward) for a large, depth-independent memory floor.

### Logits / loss — ~0.1 GB (fused cross-entropy)

The final linear projects the hidden state to a `(batch × seq, vocab)` logits tensor, and a naive `F.cross_entropy` materializes a full-precision softmax of the same shape. For Llama's 128K vocab at seq 2048 that is `2048 × 128256 × 4 bytes ≈ 1.05 GB` for the logits alone, doubled again by the softmax/log-softmax intermediate — a multi-gigabyte spike right at the end of the forward, exactly when memory is tightest. The [fused cross-entropy kernel](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy) stores only the per-row logsumexp (one float per token), recomputes the softmax on the fly in the backward kernel, and writes the gradient back over the logits buffer. The line drops to **~0.1 GB** of working set. On a 256K-vocab model like Gemma the naive spike is twice as large and this lever matters even more.

### CUDA context + overhead — ~2.0 GB (roughly fixed)

The CUDA context, cuBLAS/cuDNN workspaces, the framework's caching allocator fragmentation, and assorted working buffers take a roughly fixed **~2.0 GB** that no lever removes. It is the constant term in the budget; you plan around it, you do not optimize it away. It is also the line that makes "16 GB physical" mean "~15 GB usable" — the driver and context have already claimed their share before your first tensor allocates. Fragmentation is the wildcard here: PyTorch's caching allocator can hold freed-but-uncoalesced blocks, so the *effective* overhead drifts up over a long run, which is why a budget that fits with 200 MB of headroom sometimes OOMs at step 400. Leave a gigabyte of slack against this term, not a megabyte.

### The sum

Add the Unsloth column: `5.0 + 0.1 + 0.05 + 1.5 + 0.1 + 2.0 ≈ 8.75 GB`, call it **~8.8 GB**, comfortably under the 15 GB usable on a T4 — with a few gigabytes of headroom for a longer sequence or a larger batch. The same six lines in the naive fp16 column sum past **100 GB**. The budget *fills from the bottom up* as you add levers, and the satisfying part is watching it stop short of the ceiling.

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="The VRAM budget fills from the bottom up as each line item is added, staying below the 15 GB T4 ceiling line" style="width:100%;height:auto;max-width:760px">
<style>
.v7-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.v7-line{stroke:var(--text-primary,#1f2937);stroke-width:2;stroke-dasharray:6 5}
.v7-seg{fill:var(--accent,#6366f1)}
.v7-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:start}
.v7-tick{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:end}
.v7-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes v7-g1{0%,4%{transform:scaleY(0)}14%,100%{transform:scaleY(1)}}
@keyframes v7-g2{0%,20%{transform:scaleY(0)}30%,100%{transform:scaleY(1)}}
@keyframes v7-g3{0%,36%{transform:scaleY(0)}46%,100%{transform:scaleY(1)}}
@keyframes v7-g4{0%,52%{transform:scaleY(0)}62%,98%{transform:scaleY(1)}100%{transform:scaleY(0)}}
.v7-s1,.v7-s2,.v7-s3,.v7-s4{transform-box:fill-box;transform-origin:bottom}
.v7-s1{animation:v7-g1 11s ease-in-out infinite}
.v7-s2{animation:v7-g2 11s ease-in-out infinite}
.v7-s3{animation:v7-g3 11s ease-in-out infinite}
.v7-s4{animation:v7-g4 11s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.v7-s1,.v7-s2,.v7-s3,.v7-s4{animation:none;transform:scaleY(1)}}
</style>
<line class="v7-axis" x1="120" y1="40" x2="120" y2="270"/>
<line class="v7-axis" x1="120" y1="270" x2="420" y2="270"/>
<line class="v7-line" x1="120" y1="60" x2="640" y2="60"/>
<text class="v7-tick" x="112" y="64">15 GB T4 ceiling</text>
<text class="v7-tick" x="112" y="274">0</text>
<rect class="v7-seg v7-s1" x="180" y="180" width="180" height="90"/>
<rect class="v7-seg v7-s2" x="180" y="150" width="180" height="30"/>
<rect class="v7-seg v7-s3" x="180" y="115" width="180" height="35"/>
<rect class="v7-seg v7-s4" x="180" y="105" width="180" height="10"/>
<text class="v7-lbl v7-s1" x="380" y="230">+ 4-bit weights  ~5.0 GB</text>
<text class="v7-lbl v7-s2" x="380" y="170">+ activations (offloaded)  ~1.6 GB</text>
<text class="v7-lbl v7-s3" x="380" y="138">+ CUDA context  ~2.0 GB</text>
<text class="v7-lbl v7-s4" x="380" y="113">+ adapters/opt/logits  ~0.2 GB</text>
<text class="v7-cap" x="360" y="305">The stack tops out near 8.8 GB and never reaches the 15 GB line.</text>
</svg>
<figcaption>The budget fills from the bottom up: 4-bit weights, then offloaded activations, CUDA context, and the tiny adapter/optimizer/logits residue. The total settles around 8.8 GB, well under the 15 GB T4 ceiling.</figcaption>
</figure>

A budget calculator makes the arithmetic reusable. This is a pedagogical illustration — the constants encode the levers above — but it predicts the fit for any model you plug in:

```python
def qlora_vram_gb(n_params, n_layers, hidden, vocab, seq, batch,
                  lora_params, dtype_bytes=2):
    """Rough QLoRA VRAM budget in GB for an Unsloth-style run."""
    weights   = n_params * 0.5 / 1e9 + 0.7        # 4-bit NF4 + embed/head + scales
    adapters  = lora_params * dtype_bytes / 1e9   # fp16 LoRA A/B
    grads     = lora_params * dtype_bytes / 1e9   # adapter grads only
    optimizer = lora_params * 2 * 1 / 1e9         # 8-bit Adam, 2 moments, 1 byte
    activ     = batch * seq * hidden * dtype_bytes * 2 / 1e9   # ~1 block resident
    logits    = batch * seq * 4 / 1e9             # fused CE: per-row logsumexp
    overhead  = 2.0                               # CUDA ctx + framework
    total = weights + adapters + grads + optimizer + activ + logits + overhead
    return round(total, 2)

# Llama-3.1-8B, r=16 (~30M LoRA params), seq 2048, batch 1
print(qlora_vram_gb(8.03e9, 32, 4096, 128256, 2048, 1, 30e6))   # ~8.7
```

Plug in `8.03e9` params and you get roughly 8.7 GB — within a rounding step of the hand-summed 8.8. The calculator's value is not its precision; it is that it forces you to *name every term*. A surprise OOM is almost always a term you forgot — a batch size that scaled the activation line, a vocab that scaled the logits line, an optimizer string you didn't set.

## 4. The same run without Unsloth

The contrast is the argument. Take the identical model, dataset, and rank, and run a *full* fp16 fine-tune with a standard fp32 Adam — the default path you get if you do nothing special. The lines explode.

- **Weights, fp16:** `8.03e9 × 2 = 16.1 GB`. Already over the T4 by itself.
- **Adam state, fp32:** `8.03e9 × 2 × 4 = 64.2 GB`. The dominant term.
- **Gradients, fp16:** `8.03e9 × 2 = 16.1 GB`. Every parameter trains, so every parameter has a gradient.
- **Activations, no checkpointing:** all 32 layers' activations resident — several to tens of GB depending on batch and sequence.
- **Logits, full softmax:** the multi-GB spike from Section 3, un-fused.

The sum is north of **100 GB**, and that is before fragmentation. This run does not fit a T4. It does not fit a single A100-40GB. It needs an A100-80GB at minimum and realistically a multi-GPU setup with sharding — and even modern multi-GPU support (which, per the 2026 README, Unsloth now has) does not change the *per-step* arithmetic, only how you spread it. The crossfade below shows the two stacks against the same ceiling.

<figure class="blog-anim">
<svg viewBox="0 0 640 320" role="img" aria-label="The same run crossfades between a tall naive fp16 stack that overflows the 15 GB line and a short Unsloth stack that fits under it" style="width:100%;height:auto;max-width:720px">
<style>
.v8-line{stroke:var(--text-primary,#1f2937);stroke-width:2;stroke-dasharray:6 5}
.v8-over{fill:var(--border,#d1d5db)}
.v8-fit{fill:var(--accent,#6366f1)}
.v8-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.v8-tick{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:end}
.v8-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes v8-fadeNaive{0%,38%{opacity:1}52%,92%{opacity:0}100%{opacity:1}}
@keyframes v8-fadeUns{0%,38%{opacity:0}52%,92%{opacity:1}100%{opacity:0}}
.v8-N{animation:v8-fadeNaive 9s ease-in-out infinite}
.v8-U{animation:v8-fadeUns 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.v8-N{animation:none;opacity:0}.v8-U{animation:none;opacity:1}}
</style>
<line class="v8-line" x1="60" y1="90" x2="600" y2="90"/>
<text class="v8-tick" x="595" y="84">15 GB T4 ceiling</text>
<g class="v8-N">
<rect class="v8-over" x="240" y="20" width="160" height="250"/>
<text class="v8-lbl" x="320" y="150">naive fp16</text>
<text class="v8-lbl" x="320" y="175">~100+ GB</text>
<text class="v8-cap" x="320" y="300">Weights + Adam + grads overflow far past the ceiling.</text>
</g>
<g class="v8-U">
<rect class="v8-fit" x="240" y="180" width="160" height="90"/>
<text class="v8-lbl" x="320" y="225">Unsloth</text>
<text class="v8-lbl" x="320" y="248">~8.8 GB</text>
<text class="v8-cap" x="320" y="300">Five levers pull the same run under the line, onto a T4.</text>
</g>
<line class="v8-line" x1="240" y1="270" x2="400" y2="270" stroke-dasharray="0"/>
</svg>
<figcaption>The identical 8B QLoRA run, shown as two stacks against the same 15 GB ceiling: the naive fp16 path towers past it at ~100 GB, while Unsloth's five levers pull it down to ~8.8 GB and under the line.</figcaption>
</figure>

The contrast also clarifies *which* lever does what. The single largest saving is the optimizer line (64 GB to 0.05 GB), bought by LoRA's frozen base plus the 8-bit paged optimizer. The next is the weights (16 GB to 5 GB) via NF4. Together those two account for the bulk of the gap; activations and logits are the difference between "fits with headroom" and "fits barely." Knowing the ranking tells you where to spend your next gigabyte: if you are tight, a smaller batch or shorter sequence (the activation line) is the cheapest knob, because the big structural lines are already minimized.

## 5. Scaling the budget

The same arithmetic predicts the harder cases. Change one input at a time and watch which line dominates.

**At 70B parameters.** Weights in 4-bit NF4 are `70e9 × 0.5 ≈ 35 GB` plus scales — call it ~38 GB. That alone needs an A100-40GB or better, and with the ~2 GB context and ~2 GB of activations you are at ~42 GB, fitting an A100-80GB comfortably or an A100-40GB tightly with short sequences. The optimizer and gradient lines stay small because LoRA still freezes the base — the *weights* line is now the one that sets your GPU class. The lesson: above ~13B, the 4-bit weight footprint, not the optimizer, is the binding constraint.

**At 32K context.** The activation line scales linearly with sequence length, so seq 2048 to 32K is a 16× multiplier on the one term that depends on it: `~1.5 GB` becomes `~24 GB` of resident activations even with offloaded checkpointing, because each *block's* activation set is now 16× larger. This is where the [long-context training](/blog/machine-learning/open-source-library/unsloth-long-context-training) techniques earn their place — more aggressive offload, padding-free packing, and chunking the logits — without which the activation line alone would blow any single-GPU budget. At long context, the budget is dominated by the term the calculator computes as `batch × seq × hidden × dtype × 2`.

**At full fine-tuning.** Set `full_finetuning = True` and `load_in_4bit = False` and you opt back into the naive column on purpose: fp16 weights, full gradients, full optimizer state. The budget reverts to Section 4's ~100 GB+ for an 8B model. Full fine-tuning is sometimes the right call — when LoRA's low-rank update genuinely cannot express the adaptation you need — but you pay for it linearly in every line at once, and you should price the A100-80GB (or several) into the plan before you flip the flag.

The general rule the budget teaches: identify which single line dominates for your configuration (optimizer at small scale and full fine-tune, weights at large scale, activations at long context), and aim your remaining levers there. The budget is not just an accounting exercise; it is a profiler you can run in your head.

## 6. After training: saving the adapters

The loss curve has flattened. What is on the GPU is the frozen 4-bit base plus the trained fp16 adapters, and you have three things you might want to produce: the adapters alone, a merged full-precision model, or a quantized GGUF for local inference. Each is a different `save_method`, and each feeds a different runtime.

![Export fan-out: one trained model, four delivery formats — from the trained adapters, Unsloth's save methods produce LoRA-only, merged fp16, merged 4-bit, or GGUF artifacts for different runtimes.](/imgs/blogs/unsloth-vram-budget-and-export-4.webp)

The fan-out figure is the map for the next four sections: the trained model (4-bit base + adapters) branches into `save_method="lora"` (adapters only, headed for the Hub or an adapter-aware runtime), `save_method="merged_16bit"` (fp16 weights, headed for vLLM or TGI), and `save_pretrained_gguf` (quantized, headed for llama.cpp and Ollama). One trained artifact, four destinations, picked by where the model will run. Start with the lightest.

```python
# Save ONLY the LoRA adapters (a few tens of MB on disk).
model.save_pretrained("llama31-8b-lora", tokenizer,
                      save_method = "lora")
# Or, equivalently, the PEFT-native call:
model.save_pretrained("llama31-8b-lora")
tokenizer.save_pretrained("llama31-8b-lora")
```

`save_method = "lora"` writes the adapter weights and the PEFT config — typically tens of megabytes — and nothing else. The base model is referenced by name, not copied. This is the right artifact when you will serve with a runtime that loads base + adapter at inference time (vLLM's LoRA support, PEFT's `from_pretrained`), or when you want to keep several task-specific adapters cheaply against one shared base. It is also the smallest thing to push to a teammate or to the Hub. The cost is that the consumer must have the exact base model available and must pay the adapter-application overhead at load or at every forward.

There is a subtle correctness point worth flagging here. The adapter you save is exactly the `A` and `B` matrices that trained — bit-identical to what is on the GPU — and the base it references is the same NF4-quantized base you loaded. So `save_method="lora"` is the *most* faithful export: re-loading it reproduces your training-time numerics exactly, because nothing is merged or re-quantized. The moment you merge (next section) you fold the adapter into a dequantized weight and, for `merged_4bit`, re-quantize the sum, which is numerically excellent but no longer bit-identical. If your downstream eval is sensitive to that last bit — say you are comparing fine-tunes at the margin — keep the adapter separate and apply it at inference, so the only quantization in the pipeline is the one you trained against.

## 7. Merging: folding the adapter into the weights

When you would rather ship one self-contained model — no adapter to apply, no base dependency to resolve — you merge. Unsloth's `save_pretrained_merged` does the dequantize-and-fold step for you.

```python
# Merge LoRA into fp16 weights and save a standard HF model.
model.save_pretrained_merged(
    "llama31-8b-merged-16bit", tokenizer,
    save_method = "merged_16bit",          # default
)

# Or merge and re-quantize to 4-bit for a smaller on-disk model.
model.save_pretrained_merged(
    "llama31-8b-merged-4bit", tokenizer,
    save_method = "merged_4bit",
)
```

The full signature (from `unsloth/save.py`) is `unsloth_save_pretrained_merged(self, save_directory, tokenizer=None, save_method="merged_16bit", push_to_hub=False, token=None, ..., max_shard_size="5GB", safe_serialization=True, ..., maximum_memory_usage=0.75)`. The three `save_method` values are exactly `"lora"`, `"merged_16bit"`, and `"merged_4bit"`.

![Merging folds the adapter back into a dequantized base weight: save_pretrained_merged dequantizes each frozen 4-bit base weight to fp16, adds the scaled A@B adapter product, and writes one standard fp16 weight per layer.](/imgs/blogs/unsloth-vram-budget-and-export-3.webp)

The mechanism is the figure. For each adapted projection, the frozen 4-bit base weight `W` is dequantized to fp16 via the same `fast_dequantize` that the [manual backward](/blog/machine-learning/open-source-library/unsloth-manual-backprop) uses inside training, the scaled adapter product is added, and the result is written as a single standard fp16 tensor:

$$
W_{\text{merged}} = \text{dequant}(W) + S \cdot (A \cdot B)
$$

where `S` is the LoRA scaling (`lora_alpha / r`). For `merged_16bit` that fp16 tensor is what lands on disk; for `merged_4bit` it is re-quantized to NF4 after the fold, giving a small on-disk model that a 4-bit runtime can load directly. The `maximum_memory_usage = 0.75` default bounds how much GPU memory the merge step may use at once, and `temporary_location` names a scratch directory, because merging a 70B model can momentarily need more than the trained model did — it has to hold a dequantized fp16 weight alongside the 4-bit original for each layer it folds.

**When to merge versus keep adapters.** Merge to `merged_16bit` when you want maximum quality and will serve on a GPU runtime (vLLM, TGI) that wants standard fp16 weights — the adapter is gone, there is no application overhead, and the model is a drop-in. Merge to `merged_4bit` when on-disk size matters and your runtime speaks 4-bit. Keep `lora` when you are iterating, juggling multiple adapters on one base, or shipping the smallest possible artifact. The one trap: `merged_4bit` re-quantizes the *merged* weight, so it is a fresh NF4 quantization of `W + S·AB`, not the original base plus an exact adapter — numerically excellent but not bit-identical to applying the adapter at runtime. If you need exact reproducibility of training-time numerics, keep the adapter separate.

## 8. GGUF export for llama.cpp and Ollama

The other major destination is **GGUF**, the format [llama.cpp](/blog/machine-learning/open-source-library/unsloth-lib) and everything built on it (Ollama, LM Studio, Jan) consume. Unsloth merges, converts, and quantizes to GGUF in one call.

```python
# Export to GGUF with a chosen inference quantization.
model.save_pretrained_gguf(
    "llama31-8b-gguf", tokenizer,
    quantization_method = "q4_k_m",        # common default for local use
)

# You can pass a list to produce several quants in one conversion:
model.save_pretrained_gguf(
    "llama31-8b-gguf", tokenizer,
    quantization_method = ["q4_k_m", "q8_0", "f16"],
)
```

The signature is `unsloth_save_pretrained_gguf(self, save_directory, tokenizer=None, quantization_method="fast_quantized", first_conversion=None, push_to_hub=False, ..., maximum_memory_usage=0.85)`. Note the default `quantization_method` is `"fast_quantized"`, an alias bucket — for production local inference you almost always name an explicit type. The choice is a bits-for-size trade, and the figure lays out the common ones.

![GGUF inference quantization trades bits for file size: the quantization_method picks how many bits each tensor keeps, with q4_k_m the common default for local deployment.](/imgs/blogs/unsloth-vram-budget-and-export-5.webp)

The names follow llama.cpp's convention. `f16` keeps full 16-bit precision (~16 GB for an 8B, 100% accuracy, big and slow to load). `q8_0` is 8-bit (~8.5 GB, high quality, high resource use). The "K-quant" methods are mixed-precision: `q5_k_m` uses Q6_K for half the attention and feed-forward tensors and Q5_K elsewhere (~5.7 GB); **`q4_k_m`** does the same with Q4_K elsewhere (~4.9 GB) and is the de-facto default for running an 8B model on consumer hardware — the quality loss versus fp16 is small and the model fits in well under 6 GB. The `_m` suffix means "medium"; `_s` ("small") quantizes more tensors uniformly for a smaller file at slightly lower quality. Unsloth's own table describes `q4_k_m` precisely as "uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K" — the K-quants deliberately keep the most quality-sensitive tensors at higher precision.

Once you have the `.gguf` file, llama.cpp runs it directly (`llama-cli -m llama31-8b-gguf/...Q4_K_M.gguf -p "..."`), and Ollama wraps it with a `Modelfile`:

```dockerfile
# Modelfile for Ollama
FROM ./llama31-8b-gguf/unsloth.Q4_K_M.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.7
```

`ollama create my-llama31 -f Modelfile` and the model you fine-tuned on a free T4 is now a local chat endpoint. That is the whole arc: a budget that fits a rented card on one end, a quantized file running on your laptop on the other.

## 9. Pushing to the Hub

Each save method has a `push_to_hub_*` twin that does the same work and uploads in one step, so you skip the local round-trip.

```python
# Push merged fp16 weights straight to the Hub.
model.push_to_hub_merged(
    "your-username/llama31-8b-merged", tokenizer,
    save_method = "merged_16bit",
    token = "hf_...",
)

# Push GGUF quants straight to the Hub (great for sharing local-runnable models).
model.push_to_hub_gguf(
    "your-username/llama31-8b-gguf", tokenizer,
    quantization_method = ["q4_k_m", "q8_0"],
    token = "hf_...",
)
```

The signatures mirror the local savers: `unsloth_push_to_hub_merged(self, repo_id, tokenizer=None, save_method="merged_16bit", ..., private=None, token=None, max_shard_size="5GB", create_pr=False, commit_message="Trained with Unsloth", ...)` and the GGUF twin with `quantization_method="fast_quantized"` and the same `commit_description="Upload model trained with Unsloth 2x faster"` default. `max_shard_size="5GB"` keeps each safetensors shard under the Hub's friendly size; `create_pr=True` opens a pull request instead of committing to main, useful when pushing to a repo you do not own outright; `private=True` keeps the repo private. For the adapter-only case, the plain `model.push_to_hub("user/repo", token=...)` uploads the PEFT adapter — the lightest possible share, since the consumer pulls your base model by reference.

## 10. Case studies: reading the budget in the wild

The budget is only useful if it survives contact with real runs. Here are scenarios I have hit (or watched others hit), each diagnosed through the line-by-line ledger.

**1. "It OOMs at the very last step of the forward."** A fine-tune that trains fine for hundreds of steps and then OOMs right before the loss prints is almost always the logits line. The forward fills the budget, the final linear projects to `(batch × seq, vocab)`, and a non-fused cross-entropy materializes the full softmax exactly when memory is tightest. The fix is to confirm you are on Unsloth's fused path (you are, if you loaded through `FastLanguageModel`) and, if you wrote a custom head, to route the loss through the [fused cross-entropy kernel](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy). The tell is that the OOM size matches `seq × vocab × 4` — for Llama at seq 2048 that is the ~1 GB spike, for Gemma's 256K vocab it is ~2 GB.

**2. "It fits at seq 2048 but OOMs at seq 8192."** This is the activation line scaling linearly. Four-times the sequence is four-times the resident activation memory, because each block's activation set grew four-fold even though offloaded checkpointing keeps only one block resident. The budget predicts it: the `batch × seq × hidden × 2 × 2` term went from ~1.5 GB to ~6 GB, and added to the ~7 GB of fixed lines you cross the T4 ceiling. The fix is the [long-context techniques](/blog/machine-learning/open-source-library/unsloth-long-context-training) — more aggressive offload and chunked logits — or simply a card with more headroom. The diagnostic is that the OOM scales with sequence length and nothing else.

**3. "I switched to full fine-tuning and now nothing fits."** Setting `full_finetuning = True` opts every line back into the naive column at once: fp16 weights (16 GB), full gradients (16 GB), and full fp32 Adam (64 GB). The budget jumps from ~8.8 GB to ~100 GB+ in a single flag. This is working as intended — full fine-tuning genuinely needs that memory — but the person who flips the flag without re-reading the budget is the person who is surprised. The ledger is the warning label: full fine-tuning is an A100-80GB job for an 8B model, not a T4 job.

**4. "The merge step OOMs even though training didn't."** Merging a large model momentarily holds a dequantized fp16 weight *alongside* the 4-bit original for each layer it folds, so the peak can exceed the trained model's footprint. Unsloth bounds this with `maximum_memory_usage = 0.75` and a `temporary_location` scratch directory, paging intermediates to disk. If you still OOM, merge on a bigger card or export to `merged_4bit` (which re-quantizes layer by layer and never holds the whole fp16 model). The tell is an OOM during `save_pretrained_merged`, not during `train`.

**5. "q4_k_m made the model dumber than I expected."** GGUF inference quantization is a *lossy* step applied to the merged weights — unlike the training-time NF4, which Unsloth keeps exact. Dropping from fp16 to `q4_k_m` is usually a small quality hit because the K-quants keep the sensitive attention and feed-forward tensors at Q6_K, but on a model that is already small or already heavily fine-tuned the degradation can show. The fix is to step up: try `q5_k_m` (~5.7 GB) or `q8_0` (~8.5 GB) and measure. The budget tells you the file-size cost of each step; only an eval tells you the quality cost.

**6. "I have ten task adapters and don't want ten copies of the base."** This is the canonical reason to keep `save_method="lora"` and not merge. Ten merged 16-bit models are ten 16 GB files; ten adapters against one shared base are ten ~50 MB files plus one base. Serve them with a runtime that applies adapters at load (vLLM's multi-LoRA, or PEFT), and you pay the base footprint once. The budget here is about *disk and serving*, not training VRAM — but it is the same discipline of naming what each artifact costs.

**7. "It fits on my A100 but I want it on a T4 for cost."** Run the calculator with the T4's 15 GB ceiling before you migrate. For an 8B model at modest sequence the answer is yes with room to spare; for a 13B model it is yes but tight (4-bit weights ~7 GB plus the fixed lines); for anything above ~20B it is no on a single T4 regardless of levers, because the 4-bit weight line alone exceeds the budget. The migration decision is a budget lookup, not a try-it-and-see.

### When to merge, keep, or quantize

The export decision reduces to three questions. *Where does it run?* A GPU serving stack (vLLM, TGI) wants `merged_16bit`; a CPU or consumer-GPU runtime (llama.cpp, Ollama) wants GGUF; an adapter-aware runtime can take `lora`. *How many variants?* One model, merge it; many adapters on one base, keep them separate. *How tight is disk or memory at the destination?* Tight means GGUF `q4_k_m` or `merged_4bit`; loose and quality-first means `merged_16bit` or GGUF `q8_0`. Answer those three and the `save_method` writes itself.

## 11. Series recap: each saving maps to a post

The capstone earns its name by tying the budget back to the techniques. Every line of the ledger was a post; here is the map.

![The Inside Unsloth arc: five levers feed one budget feed one deployment, where each series post is one lever that together assembles the VRAM budget enabling fit, train, and export.](/imgs/blogs/unsloth-vram-budget-and-export-6.webp)

| Budget line (naive → Unsloth) | Lever | Series post |
| --- | --- | --- |
| Weights 16 GB → 5 GB | 4-bit NF4 + double quantization | [4-bit NF4 quantization](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4) |
| Optimizer state 64 GB → 0.05 GB | LoRA frozen base + 8-bit paged Adam | [8-bit paged optimizers](/blog/machine-learning/open-source-library/unsloth-8bit-paged-optimizers) |
| Gradients 16 GB → 0.05 GB | LoRA: only A,B carry gradients | [manual backprop](/blog/machine-learning/open-source-library/unsloth-manual-backprop) |
| Activations several GB → 1.5 GB | Offloaded gradient checkpointing | [checkpointing & offload](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload) |
| Logits multi-GB → 0.1 GB | Fused cross-entropy (logsumexp only) | [fused cross-entropy](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy) |
| Kernel launches & HBM traffic | Fused Triton kernels, in-place writes | [Triton kernel fusion](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion), [RoPE & attention kernels](/blog/machine-learning/open-source-library/unsloth-rope-attention-kernels) |
| Long-context activation blow-up | Aggressive offload + packing + chunking | [long-context training](/blog/machine-learning/open-source-library/unsloth-long-context-training) |

The recap figure draws the arc: five levers fan into the assembled budget, the budget sits at ~8.8 GB under the 15 GB T4 line, and from there the model flows to export (merged or GGUF) and on to deployment (vLLM, llama.cpp, Ollama). Read top to bottom, it is the story of the whole series: each technique is a single, exact, separable reduction, and the value of understanding them individually is that you can now read a budget, predict a fit, and pick the right export — for any model, on any card.

Two threads worth pulling next. First, the [speedup-anatomy](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) post is the time-domain twin of this memory-domain post — same levers, measured in milliseconds instead of gigabytes — and reading them together gives the full picture of why Unsloth is both faster and smaller. Second, the export choices here interact with serving: a `merged_16bit` model and a vLLM deployment is a different latency/throughput regime than a `q4_k_m` GGUF on llama.cpp, and which you pick should follow from where the model runs, not from which `save_method` you typed first. The budget got you to a trained model on a cheap card; the export got it out the door; the deployment is the next budget to read.

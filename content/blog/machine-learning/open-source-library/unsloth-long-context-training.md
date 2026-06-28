---
title: "Long-Context Training on Tiny VRAM: How Unsloth Fits 32K on a Single GPU"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "Training memory scales with sequence length on three different axes at once. Unsloth's four memory levers each flatten a different seq-dependent term, and together they let a 32K-context model fine-tune on a single consumer GPU where naive training OOMs."
tags: ["unsloth", "long-context", "rope-scaling", "sequence-length", "activation-memory", "flash-attention", "gpu-memory", "qlora", "fine-tuning", "gradient-checkpointing"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 29
---

The pitch is almost too good to believe the first time you read it: take a 7B model, fine-tune it on sequences of 32,000 tokens, and do the whole thing on a single 24 GB consumer card — the kind of GPU that, with a naive training loop, runs out of memory somewhere around sequence length 8K. Unsloth's published benchmarks make the claim concrete: on an RTX 4090, a Mistral 7B QLoRA fine-tune fits 56,420 tokens of context where Hugging Face plus Flash Attention 2 tops out at 14,099 — roughly four times the window, on identical hardware.

That 4× is not a single trick. It is the product of four memory optimizations that each attack a *different* term in the training memory budget, and the reason they compound is that long-context training has not one memory wall but three, all of which grow with sequence length simultaneously. If you only knock down one, the other two OOM you anyway. Unsloth's design is to knock down all of them at once, and the result is a near-flat memory curve where the naive curve is exponential-looking.

![Training VRAM at 2K, 8K, 32K: naive memory grows past a 24 GB card while the Unsloth stack stays near flat](/imgs/blogs/unsloth-long-context-training-1.webp)

The diagram above is the mental model. Same 7B model, same batch size of 1, three sequence lengths. The naive bars — with only 4-bit weights, no other optimization — start comfortable at 2K, are sweating at 8K, and have blown clear past the 24 GB ceiling by 32K. The Unsloth bars, with all four levers engaged, barely move: 4 GB at 2K, 5 GB at 8K, around 7 GB at 32K. The weights did not change. The math did not change — Unsloth's kernels are [exact rewrites](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion) with no approximation. What changed is *which terms are allowed to grow with sequence length*, and the answer Unsloth engineered is: almost none of them.

This post is the anatomy of that flat curve. We will pin down exactly where sequence length hits memory, walk each of the four levers and the seq-dependent term it kills, cover RoPE scaling for extending the usable context window, see Flash Attention's role, and then put it all together in a real `from_pretrained` + `SFTTrainer` call. If you have read the [speedup anatomy](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) post, this is the memory-side companion: the same lever-map philosophy, aimed at VRAM instead of throughput.

## 1. The promise, and why it is not obvious

Start with the naive intuition, because it is wrong in an instructive way. People reach for "use a smaller model" or "quantize the weights" when they OOM on long context, and both are reasonable instincts for *inference*. But for *training* a long-context model, the weights are almost never the problem. A 7B model in 4-bit NF4 is about 3.5 GB, and — this is the key fact — that number does not change when you go from 2K context to 32K context. The weights are constant in sequence length. You can quantize them to the floor and it buys you nothing as the context grows.

What grows is everything *else*: the activations the backward pass needs, the logit tensor the loss is computed over, and (naively) the attention score matrix. Each of those scales with sequence length, and they scale on different exponents, which is why a single fix is never enough. The honest version of the pitch is therefore: *long-context training is a memory-allocation problem with three independent seq-dependent terms, and fitting 32K on a small card means flattening all three plus the constant terms.* The reason Unsloth's numbers look magical is that it is the rare stack that addresses every term at once instead of one.

There is a second, subtler promise hiding in the title: the context window itself. A model pretrained on 8K tokens does not automatically "work" at 32K just because you found the memory to feed it — its rotary position embeddings have never seen positions past 8K. So there are really two problems bundled together. **Memory**: can the hardware hold a 32K training step? **Window**: can the model *use* positions out to 32K at all? Unsloth answers the first with its four memory levers and the second with RoPE scaling. We will treat them separately because they are separate, and conflating them is a common source of confusion.

A note on honesty about the numbers, up front: the published figures throughout this post are Unsloth's own benchmarks (Mistral 7B, 4-bit QLoRA, batch 1), and Unsloth itself flags that they are extrapolated and that you should set your context window roughly 10% below the table value to leave room for VRAM fragmentation. They are real, repeatable, and impressive — but they are best-case single-batch numbers, not a guarantee for every model and config. We will keep that caveat visible.

## 2. Where sequence length hits memory

To flatten the curve you first have to know the shape of every term. Here is the full training memory budget, broken into the pieces that matter and tagged by how each scales with the sequence length $s$.

![The four training-memory terms: activations, logits, and attention all grow with sequence length, while the 4-bit weights stay constant](/imgs/blogs/unsloth-long-context-training-2.webp)

**Weights — constant in $s$.** The frozen base weights, in 4-bit NF4, are about 3.5 GB for a 7B model. They do not depend on the sequence at all. This is the one term you do *not* fight with the long-context levers; you just quantize it once with [QLoRA's NF4](/blog/machine-learning/open-source-library/unsloth-manual-backprop) and forget about it.

**Activations — $\propto$ batch $\times s \times$ layers $\times$ hidden.** This is the big one, and it is the heart of the long-context problem. To compute the weight gradient $\partial C/\partial W = X^\top dY$ for any linear layer, the backward pass needs the layer's forward input $X$. So autograd pins $X$ in memory from the forward call until the backward call fires. Every linear projection, every attention input, every norm input keeps an activation alive across essentially the whole step. The total scales with depth ($L$ layers) *and* sequence length simultaneously: double $s$ and you double the retained activation in every one of the $L$ blocks at once. A concrete anchor from the [gradient-checkpointing deep-dive](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload): the residual-stream tensor alone at batch 1, $s = 8192$, hidden 4096, bf16 is $1 \times 8192 \times 4096 \times 2 = 64$ MB per block boundary, ~2 GB across 32 boundaries — and the inner block activations are several times larger again. This term is what actually OOMs a 24 GB card at long context.

**Logits / cross-entropy — $\propto s \times \text{vocab}$.** This is the term people forget about until it bites. The language-model head produces a logit for every token over the entire vocabulary, and a naive `F.cross_entropy` then materializes a full $(s \times \text{vocab})$ float32 softmax (or log-softmax) tensor to compute the loss and its gradient. For a 256K-vocab model (Gemma) at $s = 8192$ that is on the order of 8 billion floats — tens of gigabytes for the loss alone, dwarfing the model. The logit tensor grows linearly with sequence length *and* with vocab, and at long context it becomes a wall every bit as real as the activations.

**Attention — $\propto s^2$ naively, $\propto s$ with Flash Attention.** The attention score matrix $QK^\top$ is $(s \times s)$ per head. Materialize it in HBM and your attention memory is quadratic in sequence length — catastrophic at 32K, where $s^2 \approx 10^9$ entries per head. Flash Attention never materializes that matrix; it tiles the computation and keeps the scores on-chip, so the only thing that scales with $s$ is the output, which is linear. We will come back to this in section 5, but the headline is: with Flash Attention, attention stops being the dominant seq term. Without it, it dominates everything.

It is worth doing the budget arithmetic once concretely, because the orders of magnitude are what make the design obvious. Take a 7B model — hidden 4096, 32 layers — at batch 1, $s = 32768$, training in bf16. The frozen weights in NF4 are a flat 3.5 GB. The residual-stream activation at one block boundary is $1 \times 32768 \times 4096 \times 2 = 256$ MB; across 32 boundaries that is 8 GB *just for the boundaries*, and the inner-block activations (the attention projections, the SwiGLU intermediate at ~4× hidden width) multiply that several-fold if you store everything — comfortably 30–50 GB of activations alone at this length. The naive logit tensor for a 32K-vocab head is $32768 \times 32000 \times 4 \approx 4$ GB in float32; for a 256K-vocab Gemma it is 33 GB. And the per-head attention score matrix, if materialized, is $32768^2 \times 2 \approx 2$ GB *per head*. Sum those and a single 32K step "wants" well over 100 GB on a card that has 24 — which is precisely why naive long-context training is a non-starter and why every term has to be attacked, not just the biggest one.

So three of the four terms grow with $s$, on exponents $s$, $s$, and (without Flash) $s^2$. Naive long-context training fights all three at once and loses. The next section is how Unsloth wins all three.

## 3. The compounding stack: four levers, four terms

Here is the design that makes the flat curve. Four memory optimizations, each aimed at a different seq-dependent term. None of them is sufficient alone; the point is that they *stack*, because each one removes a different term from the budget.

![Each Unsloth lever kills a different seq-dependent term: 4-bit weights, offloaded checkpointing, 8-bit optimizer, fused cross-entropy](/imgs/blogs/unsloth-long-context-training-3.webp)

**Lever 1 — 4-bit NF4 weights, kills the weight term.** QLoRA stores the frozen base weights in 4-bit NormalFloat: a 7B model goes from ~14 GB in fp16 to ~3.5 GB in NF4, a 4× reduction. This is constant in $s$, so it is not really a "long-context" lever — it is the baseline that frees the budget for everything else. It is also the foundation the other levers assume: LoRA adapters on top of frozen 4-bit weights mean the only thing carrying a gradient is the tiny adapter, which is what makes levers 2 and 3 cheap. (The NF4 dequant mechanics — double quantization, the per-block `absmax` scale — are covered in the [manual-backprop post](/blog/machine-learning/open-source-library/unsloth-manual-backprop).)

**Lever 2 — offloaded gradient checkpointing, kills the activation term.** This is the lever that actually defeats the dominant long-context cost. Standard gradient checkpointing already trades compute for memory: it refuses to store a block's inner activations and recomputes them in the backward pass, paying roughly $O(\sqrt{L})$ in stored activations instead of $O(L)$. Unsloth's `use_gradient_checkpointing="unsloth"` goes one step further — it *also* copies each checkpointed block's input `hidden_states` off the GPU to system RAM during the forward pass, with an asynchronous `non_blocking` transfer that overlaps with compute. The on-GPU activation footprint across the whole network collapses to roughly one block's worth. Critically, this changes the activation term from $O(s \times L)$ on the GPU to $O(s \times 1)$ on the GPU (the rest lives on the host) — which is exactly why it flattens the steepest curve. The full mechanism is the subject of the [offloaded-checkpointing deep-dive](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload); here, the relevant fact is that it is the dominant lever for long context.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="As sequence length grows from 2K to 32K, the naive VRAM bar climbs past the card ceiling while the Unsloth VRAM bar stays nearly flat" style="width:100%;height:auto;max-width:820px">
<title>Sequence length grows; Unsloth VRAM stays flat while naive VRAM blows up</title>
<style>
.lc-axis{stroke:var(--border,#d1d5db);stroke-width:2}
.lc-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.lc-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.lc-ceil{stroke:var(--text-primary,#1f2937);stroke-width:1.5;stroke-dasharray:6 5}
.lc-naive{fill:#e8669e}
.lc-uns{fill:var(--accent,#6366f1)}
.lc-grow{transform-box:fill-box;transform-origin:bottom}
@keyframes lc-seq{0%{transform:scaleX(0.1)}100%{transform:scaleX(1)}}
@keyframes lc-tall{0%{transform:scaleY(0.12)}100%{transform:scaleY(1)}}
@keyframes lc-low{0%{transform:scaleY(0.45)}100%{transform:scaleY(1)}}
.lc-seqbar{transform-box:fill-box;transform-origin:left;animation:lc-seq 9s ease-in-out infinite alternate}
.lc-nv{animation:lc-tall 9s ease-in-out infinite alternate}
.lc-uv{animation:lc-low 9s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.lc-seqbar,.lc-nv,.lc-uv{animation:none}}
</style>
<text class="lc-lbl" x="310" y="24">sequence length: 2K -> 32K</text>
<rect class="lc-seqbar" x="60" y="40" width="600" height="16" rx="6" fill="var(--text-secondary,#6b7280)"/>
<line class="lc-axis" x1="60" y1="260" x2="660" y2="260"/>
<line class="lc-ceil" x1="60" y1="84" x2="660" y2="84"/>
<text class="lc-sub" x="540" y="78">24 GB card ceiling</text>
<rect class="lc-naive lc-grow lc-nv" x="380" y="60" width="90" height="200" rx="6"/>
<text class="lc-lbl" x="425" y="282">naive</text>
<rect class="lc-uns lc-grow lc-uv" x="560" y="204" width="90" height="56" rx="6"/>
<text class="lc-lbl" x="605" y="282">Unsloth</text>
</svg>
<figcaption>As the sequence length sweeps from 2K to 32K, naive VRAM climbs past the 24 GB ceiling, while the Unsloth stack's VRAM barely rises.</figcaption>
</figure>

**Lever 3 — 8-bit optimizer state, kills the optimizer term.** Adam keeps two state tensors per trainable parameter — the first and second moments $m$ and $v$. In fp32 that is 8 bytes of optimizer state per parameter, often more memory than the parameters themselves. Two things shrink it here: first, because the base weights are frozen, the *only* trainable parameters are the LoRA adapters (a few million, not billions), so the optimizer state is tiny to begin with; second, Unsloth uses 8-bit optimizers (via bitsandbytes) so even that small state is quarter-size. The optimizer term is constant in $s$ — it depends on parameter count, not sequence length — but it is worth keeping small so it does not eat into the budget the activation lever just freed.

**Lever 4 — fused cross-entropy, kills the logit term.** This is the lever that knocks down the second wall. Instead of materializing the full $(s \times \text{vocab})$ float32 softmax, Unsloth's fused cross-entropy kernel computes the loss in a single pass and stores only the per-row `logsumexp` — *one float per token* — then recomputes the softmax on the fly in the backward kernel and writes the gradient back in place over the logits buffer. The $(s \times \text{vocab})$ tensor never exists in HBM. The chunked variant (`MAX_FUSED_SIZE = 65536`) handles vocabularies larger than 65,536 (Gemma's 256K) by reducing per-chunk logsumexps. The mechanism and the chunking identity are the subject of the [fused cross-entropy deep-dive](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy); here, the relevant fact is that it converts the logit term from $O(s \times \text{vocab})$ stored to $O(s)$ stored — which at a 256K vocab is the difference between OOM and fitting.

Now you can see why the curve is flat. Of the three growing terms — activations, logits, attention — lever 2 reduces the on-GPU activations to one block, lever 4 reduces the logits to one float per token, and Flash Attention (section 5) keeps attention linear. The only term left growing with $s$ is the residual-stream activations of a single block plus the linear attention output, both small. Each lever was necessary; only together do they make 32K fit in ~7 GB.

The compounding is multiplicative, not additive, and that is the part worth internalizing. If you only quantized the weights, the 30–50 GB of activations still OOM the card. If you only added offloaded checkpointing but left the logits naive, a 256K-vocab model's 33 GB loss tensor still OOMs it. If you fixed both of those but ran eager attention, the 2-GB-per-head score matrices still OOM it. There is no single dominant term you can knock down to win; the budget is a sum of large numbers and you have to shrink *every* large addend below the card's capacity. This is why "Unsloth is just QLoRA" or "Unsloth is just gradient checkpointing" both miss the point — it is the *stack*, assembled so that no remaining term scales badly with $s$, that produces the flat curve. Take any one lever away and the curve bends back up to OOM. That property — that each lever is load-bearing — is also why the speedup is so consistent across GPU sizes in the benchmark table later: the levers remove terms with the same scaling regardless of how much total VRAM the card has, so the *ratio* of Unsloth's max context to the baseline's stays roughly constant.

| Lever | Term it kills | Scaling in $s$ before | Scaling in $s$ after | Config |
| --- | --- | --- | --- | --- |
| 4-bit NF4 weights | weights | constant (but 4× too big) | constant, ¼ size | `load_in_4bit=True` |
| Offloaded checkpointing | activations | $O(s \times L)$ on GPU | $O(s \times 1)$ on GPU | `use_gradient_checkpointing="unsloth"` |
| 8-bit optimizer state | optimizer | constant (LoRA-only) | constant, ¼ size | 8-bit optimizer |
| Fused cross-entropy | logits | $O(s \times \text{vocab})$ | $O(s)$ | automatic in Unsloth |

## 4. RoPE scaling: extending the usable window

Memory gets the step onto the GPU. It does not make a model trained on 8K tokens *work* at 32K. That is the job of RoPE scaling, and it is a distinct mechanism worth understanding on its own.

Rotary Position Embeddings (RoPE) encode token position by rotating the query and key vectors by an angle proportional to the position index. The model learns to interpret those rotations only over the range of positions it saw in pretraining — if it was pretrained on 8,192 tokens, it has never seen the rotation angle for position 20,000, and feeding it one produces garbage attention. The window is a property of the position encoding, not the memory budget.

![RoPE linear scaling divides each position index by the factor, remapping positions 0..32K into the pretrained 0..8K rotary range](/imgs/blogs/unsloth-long-context-training-4.webp)

The simplest fix is **linear scaling**: divide every position index by a `factor` before computing the rotation. With `factor = 4`, position 32,000 is encoded as if it were position 8,000 — back inside the range the model was pretrained on. The positions are "squeezed" into the trained window. You then fine-tune briefly so the model adapts to the finer-grained rotations, and it can now attend across the longer sequence using angles it already understands.

<figure class="blog-anim">
<svg viewBox="0 0 720 240" role="img" aria-label="RoPE linear scaling: position indices spread across 0 to 32K are divided by the factor and compressed into the pretrained 0 to 8K rotary range" style="width:100%;height:auto;max-width:820px">
<title>RoPE linear scaling compresses positions into the pretrained window</title>
<style>
.rp-track{fill:none;stroke:var(--border,#d1d5db);stroke-width:2}
.rp-win{fill:var(--accent,#6366f1);opacity:0.14;stroke:var(--accent,#6366f1);stroke-width:1.5}
.rp-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.rp-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.rp-tok{fill:var(--accent,#6366f1)}
@keyframes rp-fadeWide{0%,38%{opacity:1}52%,92%{opacity:0}100%{opacity:1}}
@keyframes rp-fadeNarrow{0%,38%{opacity:0}52%,92%{opacity:1}100%{opacity:0}}
.rp-wide{animation:rp-fadeWide 9s ease-in-out infinite}
.rp-narrow{animation:rp-fadeNarrow 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.rp-wide{animation:none;opacity:0}.rp-narrow{animation:none;opacity:1}}
</style>
<rect class="rp-win" x="60" y="70" width="170" height="60" rx="8"/>
<text class="rp-sub" x="145" y="156">pretrained 0..8K</text>
<line class="rp-track" x1="60" y1="100" x2="660" y2="100"/>
<g class="rp-wide">
<circle class="rp-tok" cx="60"  cy="100" r="9"/>
<circle class="rp-tok" cx="210" cy="100" r="9"/>
<circle class="rp-tok" cx="360" cy="100" r="9"/>
<circle class="rp-tok" cx="510" cy="100" r="9"/>
<circle class="rp-tok" cx="660" cy="100" r="9"/>
<text class="rp-sub" x="360" y="44">raw positions 0 .. 32K (too wide for the window)</text>
<text class="rp-sub" x="60"  y="190">0</text>
<text class="rp-sub" x="510" y="190">24K</text>
<text class="rp-sub" x="660" y="190">32K</text>
</g>
<g class="rp-narrow">
<circle class="rp-tok" cx="60"  cy="100" r="9"/>
<circle class="rp-tok" cx="98"  cy="100" r="9"/>
<circle class="rp-tok" cx="135" cy="100" r="9"/>
<circle class="rp-tok" cx="172" cy="100" r="9"/>
<circle class="rp-tok" cx="210" cy="100" r="9"/>
<text class="rp-sub" x="360" y="44">scaled positions pos / 4 (now inside 0..8K)</text>
<text class="rp-sub" x="430" y="100">pos' = pos / factor (factor = 4)</text>
</g>
</svg>
<figcaption>Linear RoPE scaling divides every position index by the factor, packing the 0..32K positions into the 0..8K rotary range the model was pretrained on.</figcaption>
</figure>

Linear scaling has a known cost: squeezing positions uniformly compresses the high-frequency rotations the model uses for fine local distinctions, which can slightly blur short-range attention. **Dynamic NTK scaling** addresses this by scaling the RoPE base frequency non-uniformly — it leaves high-frequency (local) components nearly untouched and stretches only the low-frequency (long-range) components, and it can apply the scaling dynamically as the sequence grows rather than at a fixed factor. In Unsloth this is exposed through the same `rope_scaling` argument:

```python
# Linear scaling: uniform pos / factor
rope_scaling = {"type": "linear", "factor": 4.0}

# Dynamic NTK scaling: non-uniform, often better short-range retention
rope_scaling = {"type": "dynamic", "factor": 4.0}
```

The companion knob is `max_seq_length`, which tells Unsloth the maximum sequence you intend to train on so it can size buffers, configure the RoPE tables, and (with `rope_scaling`) extend the window. Setting `max_seq_length = 32768` with `rope_scaling = {"type": "linear", "factor": 4.0}` on an 8K-pretrained model says: I want to train at 32K, and I will reach it by squeezing positions 4× into the trained range. The RoPE kernels that actually apply the rotation — the in-place, group-of-4-heads Triton kernels with the cos/sin tables cached for backward — are covered in the [RoPE & attention kernels post](/blog/machine-learning/open-source-library/unsloth-rope-attention-kernels); RoPE scaling is the configuration that decides *which* angles those kernels compute.

## 5. Flash Attention's role

We flagged attention as the $s^2$ term. Flash Attention is what turns it back into an $s$ term, and without it none of the memory levers above would matter, because the score matrix alone would OOM you.

![Naive attention materializes a quadratic score matrix in HBM; Flash Attention tiles it on-chip and only the linear output leaves](/imgs/blogs/unsloth-long-context-training-5.webp)

The naive attention computation builds the full $(s \times s)$ score matrix $QK^\top$ in HBM, applies softmax over it, and multiplies by $V$. At $s = 32{,}768$ that score matrix is over a billion entries per head — gigabytes per head, per layer, materialized in global memory. It is the single most expensive tensor in long-context attention if you store it.

Flash Attention never stores it. It tiles $Q$, $K$, and $V$ into blocks, loads a tile of each into SRAM, computes that tile's partial scores on-chip, runs an online (running) softmax that accumulates the output without ever holding the full row of scores, and then discards the tile. Only the attention output $O$, shape $(s \times \text{head\_dim})$, ever lands in HBM — and that is linear in $s$. The quadratic intermediate lives and dies in SRAM, tile by tile. This is the same on-chip-intermediate principle behind Unsloth's [fused Triton kernels](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion), applied to attention.

Unsloth does *not* write its own attention kernel — and this is a deliberate, correct design choice. Flash Attention (and xformers) are already state-of-the-art at keeping the softmax on-chip; there is nothing to gain by reimplementing them. What Unsloth contributes around attention is making sure the tensors fed to Flash Attention are laid out correctly and that the rotary embedding is applied with its own fused RoPE kernel before attention runs. The division of labor is clean: Flash Attention owns the $s^2 \to s$ memory win for the scores; Unsloth owns the fused RoPE, the correct layout, and every *other* memory term. Knowing what not to rewrite is part of the engineering.

The mechanism that makes the tiling correct is the *online softmax*, and it is worth a sentence because it is the non-obvious part. A normal softmax needs the maximum and the sum-of-exponentials over the *entire* row of scores before it can normalize — which seems to require the whole row in memory. The online softmax sidesteps this by carrying a running maximum and a running normalizer as it streams tiles: when a new tile produces a larger maximum, it rescales the partial output accumulated so far by the correction factor $e^{m_\text{old} - m_\text{new}}$ and continues. The result is bit-for-bit the same softmax as the materialized version, computed in a single streaming pass over the tiles with only $O(\text{tile})$ memory live at once. That is the trick that lets attention be exact and yet never store the $(s \times s)$ matrix — the same "recompute/stream instead of store" philosophy that runs through every Unsloth lever.

The practical consequence for this post: Flash Attention is a *prerequisite* for the flat curve, not one of Unsloth's four levers. If you somehow ran long-context training with eager attention, the score matrix would dominate and the four levers would be rearranging deck chairs. Unsloth assumes Flash Attention is in play, which on supported hardware it is by default — and on hardware where a fused Flash kernel is unavailable, the long-context memory profile degrades sharply, which is one reason the published benchmarks specify the GPU: the numbers assume a card that runs Flash Attention well.

## 6. Putting it together

Here is what the full long-context configuration actually looks like in code, accurate to Unsloth's `FastLanguageModel.from_pretrained` signature. This is the real API, not pseudocode.

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 32768  # the long context we want to train on

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "unsloth/mistral-7b-v0.3",
    max_seq_length  = max_seq_length,
    dtype           = None,            # None -> auto: bf16 on Ampere+, else fp16
    load_in_4bit    = True,            # lever 1: 4-bit NF4 frozen weights
    use_gradient_checkpointing = "unsloth",  # lever 2: offloaded checkpointing
    rope_scaling    = {"type": "linear", "factor": 4.0},  # 8K -> 32K window
    random_state    = 3407,
    max_lora_rank   = 64,
)
```

A few things to read carefully. `load_in_4bit=True` and `use_gradient_checkpointing="unsloth"` are actually the *defaults* in Unsloth's loader — you are not turning on something exotic, you are using the stack as shipped. `dtype=None` auto-selects bf16 on Ampere and newer, fp16 otherwise. `rope_scaling` is what extends the window from the model's pretrained 8K to the 32K we asked for via `max_seq_length`; drop it and you would have the *memory* to feed 32K but the model would not *understand* positions past 8K. `random_state=3407` is Unsloth's documented default seed. The fused cross-entropy (lever 4) and the 8-bit optimizer (lever 3) are wired in by Unsloth automatically and through the trainer config, respectively — there is no separate flag to materialize the logit-wall fix; it is simply how Unsloth's loss is computed.

Next, attach the LoRA adapters. The frozen 4-bit base plus small trainable adapters is what keeps the optimizer state tiny:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r              = 64,               # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 64,
    lora_dropout   = 0,               # 0 is fastest and Unsloth-optimized
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # offload here too
    random_state   = 3407,
    use_rslora     = False,
)
```

Then the trainer. The two flags that matter for long-context efficiency are `packing=True` and `gradient_accumulation_steps`:

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    args = SFTConfig(
        max_seq_length              = max_seq_length,
        packing                     = True,    # concatenate samples to fill the window
        per_device_train_batch_size = 1,       # one long sequence per step
        gradient_accumulation_steps = 8,       # effective batch = 8
        warmup_steps                = 10,
        max_steps                   = 120,
        learning_rate               = 2e-4,
        bf16                        = torch.cuda.is_bf16_supported(),
        optim                       = "adamw_8bit",   # lever 3: 8-bit optimizer
        logging_steps               = 1,
        seed                        = 3407,
    ),
)
trainer.train()
```

`optim="adamw_8bit"` is the explicit handle for lever 3. `per_device_train_batch_size=1` is normal for long context — at 32K, a single sequence is already a large amount of work, and you scale the *effective* batch with gradient accumulation instead of physical batch, which we cover next. This whole config is the [TRL](/blog/machine-learning/open-source-library/trl-lib) `SFTTrainer` that Unsloth integrates with; Unsloth swaps in its fused kernels and offloaded checkpointer underneath without changing the trainer API.

## 7. Gradient accumulation and dataset packing

Two efficiency techniques deserve their own section because they are easy to get subtly wrong at long context.

**Gradient accumulation** decouples the *effective* batch size from the *physical* batch size you can fit in VRAM. At 32K context you can typically fit only one sequence per step (`per_device_train_batch_size=1`). But training dynamics often want a larger effective batch for stable gradients. The fix: run several forward/backward passes, accumulate their gradients without stepping the optimizer, and only step (and zero) after `gradient_accumulation_steps` micro-batches. Effective batch = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`. With batch 1 and 8 accumulation steps you train as if the batch were 8, while never holding more than one sequence's activations in memory at a time. It trades wall-clock (more forward/backward passes per optimizer step) for the ability to keep the long sequence on the card — exactly the right trade for long context, where memory is the binding constraint.

One historical caveat worth knowing: there was a period where naive gradient accumulation produced a subtly wrong loss versus a true large batch, because the per-micro-batch loss normalization did not account for varying numbers of non-padding tokens across micro-batches. Unsloth was among the projects that fixed this so accumulated training matches full-batch training; if you are reading older tutorials, be aware the math was corrected and modern trainers handle the token-count normalization correctly.

**Dataset packing** (`packing=True`) addresses a different waste. If your training samples are shorter than `max_seq_length`, padding each one out to 32K wastes enormous compute and memory on pad tokens that contribute nothing. Packing concatenates multiple samples end-to-end (with separators) to fill the window densely, so every token in the 32K sequence is a real training token. At long context this is a large efficiency win — without it, a dataset of 2K-token samples padded to 32K would be ~94% wasted positions. Unsloth's "padding-free packing" is part of the same family of optimizations that earns its "3x faster" headline: the kernels are not spending bandwidth on pad tokens. The thing to watch is that packing mixes samples within a sequence, so for tasks where cross-sample attention contamination matters you may want attention masking that respects sample boundaries; for most SFT it is a clean win.

There is a memory subtlety in how packing interacts with the rest of the stack that is easy to miss. Packing makes every sequence *exactly* `max_seq_length`, which means your memory profile is no longer dominated by the rare long sample — it is uniform and predictable at the full window. That is actually good for the allocator (no spiky allocations from occasional long sequences) but it also means you are paying the full 32K memory cost on every single step, so the headroom math from the benchmark table applies to *every* step, not just the worst case. The flip side is throughput: a packed 32K sequence does the work of, say, sixteen 2K samples in one forward/backward with one set of launch overheads and one set of activation round-trips, so the per-token efficiency is much higher than padding-and-batching short samples. Packing and the four memory levers are designed to be used together — packing keeps the expensive long sequence full of useful tokens, and the levers keep that full sequence affordable.

The interaction with gradient accumulation is worth stating explicitly too, since both are about effective batch size from different angles. Packing increases the number of real training tokens *per sequence*; gradient accumulation increases the number of *sequences* per optimizer step. At long context you typically max out the first (pack to the full window) and then dial the second (accumulation steps) to reach whatever effective batch your training dynamics want. Because the corrected accumulation normalizes by the true non-padding token count, packing — which removes padding — also makes the accumulated loss cleaner, since there is far less padding for the normalization to account for in the first place.

## 8. The numbers

Now the published figures, presented as Unsloth's own benchmarks. These are Mistral 7B, 4-bit QLoRA, batch size 1 — the maximum context length that fits, Unsloth vs Hugging Face plus Flash Attention 2, across four GPUs.

![Published maximum context length: Unsloth fits roughly 4x the tokens of Hugging Face plus Flash Attention 2 on the same card](/imgs/blogs/unsloth-long-context-training-6.webp)

| GPU | VRAM | Hugging Face + FA2 | Unsloth | Improvement |
| --- | --- | --- | --- | --- |
| RTX 4090 | 24 GB | 14,099 tokens | 56,420 tokens | ~4× longer |
| A100 | 40 GB | 26,502 tokens | 105,500 tokens | ~4× longer |
| A6000 | 48 GB | 32,704 tokens | 130,040 tokens | ~4× longer |
| H100 | 80 GB | 57,510 tokens | 228,199 tokens | ~4× longer |

Read these correctly. The ~4× ratio is remarkably consistent across cards, which tells you the win is structural (it comes from removing seq-dependent terms, which scale the same way regardless of total VRAM) rather than an artifact of one GPU's memory size. The 24 GB row is the headline for the title: 56,420 tokens on a consumer 4090 comfortably covers a 32K training run, which is why "32K on a single GPU" is a real claim and not marketing.

Two honesty notes that Unsloth itself attaches. First, these numbers are *extrapolated* from experiments, and Unsloth recommends setting your actual context window roughly 10% below the table value to leave headroom for VRAM fragmentation — so plan for ~50K on the 4090, not a hard 56,420. Second, the offloaded gradient checkpointer that makes much of this possible is documented as buying "30% further" VRAM reduction at the cost of "+1.9% extra time overhead" — the memory win is nearly free in throughput, but not exactly free, and that small overhead is the price of streaming activations to host RAM. These are best-case single-batch figures; your mileage on a specific model, vocab size, and dataset will vary, but the *shape* — roughly 4× the context on the same card, near-flat memory growth — holds.

## 9. Caveats and case studies

The flat curve is real, but long-context training has sharp edges. Here are the ones that bite in practice.

**Case 1 — RoPE scaling degrades short-context quality if you overdo it.** Linear scaling with a large factor compresses the high-frequency rotations the model uses for fine local distinctions, and a model fine-tuned at `factor=8` can get measurably worse at short-context tasks it used to handle well. The mitigation is to use the smallest factor that covers your target length, prefer dynamic NTK scaling when short-range retention matters, and fine-tune on a mix that includes short sequences so the model does not forget them. Do not set `factor=8` "to be safe" if you only need 16K.

**Case 2 — fitting the step is not the same as a good gradient.** The memory levers get a 32K sequence onto the card, but a batch size of 1 (even with accumulation) and a heavily packed dataset can produce noisier gradients than a short-context run with large physical batches. Long-context fine-tuning often needs more careful learning-rate and warmup tuning, and the "it fits!" moment is the start of the work, not the end.

**Case 3 — the throughput tax is small but not zero.** Offloaded checkpointing streams activations to host RAM over PCIe. Unsloth masks the copy behind compute with `non_blocking` transfers, and the documented overhead is about +1.9% time — but on a system with a slow PCIe link or contended host RAM bandwidth, the copy may not fully hide, and you can see a larger slowdown. Recomputation (standard checkpointing) also costs an extra forward pass per block. The memory you save is paid for in a little compute; at long context that trade is almost always worth it, but it is a trade.

**Case 4 — the fragmentation 10% is real.** The published token counts are extrapolated and assume near-perfect allocation. In practice, PyTorch's caching allocator fragments, and at the very edge of VRAM a run that "should" fit by the arithmetic will OOM on a large activation allocation that cannot find a contiguous block. Heed Unsloth's own advice and leave ~10% headroom; setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can also help the allocator at the margin.

**Case 5 — vocab size moves the logit wall.** The fused cross-entropy lever matters more the larger the vocabulary. For a 32K-vocab Llama the naive logit tensor is already large but survivable; for a 256K-vocab Gemma it is the dominant term, and the fused/chunked CE is the difference between fitting and not. If you are training a large-vocab model at long context, the cross-entropy lever is doing most of the heavy lifting, and it is worth confirming your stack is using Unsloth's fused loss and not a vanilla `F.cross_entropy` that silently materializes the wall.

**Case 6 — single-GPU is no longer the only option.** The older framing of Unsloth as strictly single-GPU is out of date: multi-GPU is supported now, with a major upgrade flagged as on the way. For genuinely enormous context or larger models, sharding across GPUs ([DeepSpeed ZeRO](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive)-style) is complementary to the per-GPU memory levers here — the levers shrink each GPU's footprint, and sharding distributes what remains. The single-GPU 32K story is impressive precisely because it does *not* require sharding, but it is a floor, not a ceiling.

**Case 7 — quality at length needs evaluation, not faith.** A model that *trains* at 32K is not guaranteed to *use* 32K well. RoPE-scaled models can attend across the long window mechanically while still showing "lost in the middle" behavior — strong recall at the ends of the context, weak in the middle. The memory and window machinery in this post is necessary but not sufficient for genuine long-context capability; you still have to evaluate retrieval and reasoning across the full window, ideally with needle-in-a-haystack-style probes, before trusting the model at length.

### When to reach for this stack

Reach for Unsloth's long-context configuration when you need to fine-tune a model on sequences longer than its pretrained window on limited VRAM — a single consumer or workstation GPU, a 7B–13B model, a context target several times the pretrained length. Turn on all four levers (they are mostly defaults), set `rope_scaling` to the smallest factor that covers your target, use `packing=True` and gradient accumulation, and leave ~10% VRAM headroom. This is the sweet spot the published 4× numbers come from.

Be more cautious when short-context quality is critical (RoPE scaling has a cost), when you need the absolute maximum throughput (the checkpointing offload tax, small as it is, is real), or when you are already at multi-GPU scale where sharding may be the bigger lever. And remember the two-problem decomposition that this whole post rests on: memory and window are separate. The four levers solve memory; RoPE scaling solves the window; Flash Attention is the prerequisite that keeps attention from dominating. Get all three right and a 32K fine-tune on a 24 GB card stops being surprising and starts being routine — which, given where the field was two years ago, is the genuinely remarkable part.

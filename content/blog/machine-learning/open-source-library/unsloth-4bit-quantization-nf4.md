---
title: "4-bit NF4 Quantization: How Unsloth Shrinks the Frozen Base Model"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "The base model's weights are the first VRAM wall you hit. QLoRA freezes them in 4-bit NormalFloat, double-quantizes the scales, and dequantizes back to fp16 only transiently when a matmul needs them — collapsing an 8B weight bill from 16 GB to about 5 GB with no measurable accuracy loss. Here is exactly how Unsloth does it."
tags: ["unsloth", "quantization", "nf4", "qlora", "bitsandbytes", "double-quantization", "gpu-memory", "lora", "memory-optimization"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 28
---

The first wall you hit when you try to fine-tune an 8-billion-parameter model on a single 24 GB GPU is not the optimizer, not the activations, and not the gradient. It is the weights. Eight billion parameters in fp16 is `8e9 × 2 = 16 GB` before you have allocated a single activation buffer, before the optimizer reserves its moment estimates, before the first token of your dataset is tokenized. You load the model, you watch `nvidia-smi`, and two-thirds of your VRAM is already gone — spent on weights that you have frozen and will never update.

That is the absurdity QLoRA attacks. If the base weights are frozen, why are we paying full fp16 price to store them? We do not update them; we only read them during the forward pass and during the gradient-through-the-frozen-layer in the backward pass. A frozen, read-only tensor is exactly the kind of thing you should be willing to compress aggressively, as long as you can decompress it cheaply at the moment you need it. QLoRA's answer is to store the frozen base in **4-bit NormalFloat (NF4)** with per-block scales, to **double-quantize** those scales, and to **dequantize back to fp16/bf16 only transiently** when a matmul demands it. Unsloth ships exactly this machinery, wired into `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`, and it is the single largest line item in Unsloth's "up to 70% less VRAM" headline.

![The weight bill: an fp16 8B base costs about 16 GB and OOMs a 24 GB GPU before activations, while the NF4 base costs about 5 GB and leaves room to train.](/imgs/blogs/unsloth-4bit-quantization-nf4-1.webp)

The diagram above is the mental model for this entire post. On the left, the fp16 base: 16 GB of weights, a tiny stack of trainable LoRA adapters on top, and an out-of-memory error the moment you add activations on a 24 GB card. On the right, the NF4 base: the same 8 billion parameters stored at roughly half a byte each — about 5 GB — with the same fp16 LoRA adapters still trainable on top, and enough headroom left to actually run a fine-tune. Same model, same adapters, a fraction of the weight memory. The trick is that the base weights are frozen and read-only, so we can store them in a lossy-looking-but-actually-faithful 4-bit format and pay the decompression cost only on the hot path.

This is the fifth post in the *Inside Unsloth* series. It assumes you have read the [overview of what Unsloth is](/blog/machine-learning/open-source-library/unsloth-lib) and the [anatomy of where its speedups come from](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy); it leans directly on the dequantize-then-free pattern from the [hand-derived backpropagation post](/blog/machine-learning/open-source-library/unsloth-manual-backprop). If you want a sibling treatment of quantization on the inference side, the [KV-cache quantization deep-dive](/blog/machine-learning/open-source-library/turboquant-kv-cache-quantization-deep-dive) is the companion piece.

## 1. Why 4 bits — the memory math, and the catch

Start with the arithmetic, because the arithmetic is the whole motivation. A parameter stored in:

| Format | Bytes / parameter | 8B model weights | Notes |
| --- | --- | --- | --- |
| fp32 | 4.0 | 32.0 GB | full precision, almost never used for the frozen base |
| fp16 / bf16 | 2.0 | 16.0 GB | the default; the "16 GB wall" |
| int8 | 1.0 | 8.0 GB | LLM.int8(), halves it again |
| NF4 (single) | ~0.5 | ~5.0 GB | 4-bit codes plus per-block fp scales |
| NF4 + double quant | ~0.53 | ~4.3 GB | the scales are themselves quantized |

The jump from fp16 to NF4 is a 4× reduction in the weight bill. For an 8B model that is the difference between 16 GB and roughly 5 GB — the difference between "OOM on a 4090" and "fits with room to train." This is not a marginal optimization; it is the thing that makes single-GPU fine-tuning of an 8B model possible at all.

But there is a catch, and it is the catch that organizes the rest of this post. **Compute does not happen in 4 bits.** There is no hardware GEMM that multiplies a 4-bit weight matrix by an fp16 activation and gives you an fp16 result with full accuracy. The tensor cores want fp16, bf16, or fp8 operands. So somewhere between "the weight is stored in 4 bits" and "the matmul runs in fp16," the weight has to be expanded back to 16 bits. The entire design problem of 4-bit quantization for training is: *store in 4 bits, compute in 16 bits, and make the round trip from one to the other cheap enough that it does not eat the memory you just saved.*

If you naively dequantized the whole weight matrix to fp16, held it, ran the matmul, and held it again for the backward pass, you would be back to 16 GB of fp16 weights resident in VRAM — you would have saved nothing. The discipline that makes QLoRA actually work is that the fp16 weight is **transient**: it is rebuilt for exactly one matmul and freed immediately, so the peak fp16-weight footprint is one layer's worth, not the whole model's. We will trace that discipline through `fast_dequantize` in section 5.

Everything else — NF4's specific level placement, double quantization, the decode-time GEMV — is in service of two goals: make the stored form as small as possible, and make the transient expansion as fast and as faithful as possible.

## 2. What NF4 actually is

NF4 stands for **4-bit NormalFloat**. Four bits gives you 16 distinct codes. The question NF4 answers is: *where should those 16 levels sit?*

The obvious answer is uniform spacing: place the 16 levels evenly across `[-1, 1]`, like a standard int4 quantizer. That is the right choice if the values you are quantizing are uniformly distributed. But pretrained neural network weights are not uniformly distributed — they are approximately **normally distributed**, tightly clustered around zero with thin tails. If you space your 16 levels uniformly, you waste most of them on the tails where almost no weights live, and you starve the dense region near zero where most of the weights actually are. You spend your precious 16 codes in the wrong place.

NF4's insight, from the QLoRA paper, is to place the 16 levels at the **quantiles of a standard normal distribution**. Each of the 16 codes is positioned so that, if your weights really are normally distributed, each code "owns" roughly the same number of weights. That is information-theoretically the right thing to do: a code that is hit by very few weights is a wasted code. By placing the levels at quantiles, NF4 makes every code carry roughly equal probability mass.

![NF4 places its 16 levels at the quantiles of a normal: bars are narrow near the mean and wide at the tails, so each code owns roughly equal probability mass.](/imgs/blogs/unsloth-4bit-quantization-nf4-2.webp)

The figure above draws the 16 NF4 levels as bars positioned along the `[-1, 1]` axis. The bar *heights* trace the bell curve — most weights live near zero. The bar *widths* are the quantile cells each code owns. Notice the asymmetry: near the mean, the cells are only about 80 pixels wide because the levels are packed tightly (the codes around `-0.09`, `0.00`, `+0.08`, `+0.16` are crowded together); at the tails, the cells balloon to two or three times that width because the levels are spread out (from `+0.72` it is a long jump to `+1.0`). The levels are dense where the weights are dense and sparse where the weights are sparse. That is what "quantiles of a normal" buys you, and it is why NF4 beats uniform int4 on the same number of bits with no extra storage.

There is a subtlety worth stating precisely: the NF4 codebook is **not symmetric in the naive sense**. It is built to be symmetric around zero with an exact zero level (so that a genuine zero weight maps to a genuine zero code with no error), but the 16 levels include a 0 and 15 non-zero quantile points. Unsloth does not reimplement this codebook; it delegates to bitsandbytes, where the NF4 codes are baked into the CUDA kernels. What Unsloth owns is the *plumbing* — turning the 4-bit codes back into fp16 on the hot path — which is what the rest of this post is about.

### Per-block absmax scaling

A single global scale for the entire weight matrix would be a disaster: one large outlier weight would stretch the scale and crush the resolution for every other weight. So NF4, like all serious quantization schemes, uses **per-block scaling**. The weight tensor is chopped into contiguous blocks of `blocksize = 64` weights. Each block gets its own scale, computed as the **absmax** — the maximum absolute value in that block. Every weight in the block is divided by its block's absmax before being mapped to the nearest NF4 quantile, and multiplied back by the absmax on the way out.

The block size is a tradeoff. Smaller blocks mean more scales (more memory overhead) but tighter per-block dynamic range (better fidelity, because an outlier only contaminates its own 64-weight block). QLoRA's choice of 64 is the empirically good middle. With `blocksize = 64`, you store one scale per 64 weights — and that scale is where the next problem appears.

## 3. Double quantization: quantizing the scales

Here is the overhead nobody mentions when they say "4 bits per weight." If each block of 64 weights carries one fp32 absmax scale, then the scales alone cost `32 bits / 64 weights = 0.5 bits per weight`. On top of the 4 bits of weight, that is a 12.5% overhead — your "4-bit" model is really a 4.5-bit model. For an 8B model, that 0.5 bit/param is about 0.5 GB of pure scale metadata.

QLoRA's second trick is **double quantization**: quantize the scales themselves. The absmax array is just another array of numbers with its own distribution, so you can quantize it the same way you quantized the weights. QLoRA chops the absmax array into blocks of `blocksize2 = 256`, computes a second-level scale per block (`absmax2`), and stores each absmax value in 8 bits against an 8-bit codebook (`code2`), with a mean offset subtracted out first. The fp32 scales drop to 8-bit codes plus a sparse set of fp32 second-level scales.

The math: instead of `0.5 bits/param` of scale overhead, double quantization brings it down to roughly `8 bits / 64 weights + 32 bits / (64 × 256) weights ≈ 0.127 bits/param`. You save about `0.5 − 0.13 ≈ 0.37` bits per parameter. On an 8B model that is roughly 0.4 GB recovered — small in absolute terms, but it is free fidelity-neutral memory, and it is what takes the effective bytes-per-param from ~0.56 down to ~0.53.

<figure class="blog-anim">
<svg viewBox="0 0 700 300" role="img" aria-label="Double quantization nests three levels: 4-bit NF4 weights, an 8-bit absmax scale per block, and those absmax values themselves quantized inside state2" style="width:100%;height:auto;max-width:800px">
<title>Double-quantization nesting</title>
<style>
.dn-box{stroke:var(--border,#d1d5db);stroke-width:1.5}
.dn-l1{fill:var(--surface,#f3f4f6)}
.dn-l2{fill:var(--surface,#f3f4f6)}
.dn-l3{fill:var(--surface,#f3f4f6)}
.dn-hl{fill:var(--accent,#6366f1);opacity:0}
.dn-t{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.dn-s{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.dn-arr{stroke:var(--text-secondary,#6b7280);stroke-width:2;fill:none;marker-end:url(#dn-a)}
@keyframes dn-1{0%,4%{opacity:0}10%,30%{opacity:.22}36%,100%{opacity:0}}
@keyframes dn-2{30%,34%{opacity:0}40%,62%{opacity:.22}68%,100%{opacity:0}}
@keyframes dn-3{62%,66%{opacity:0}72%,94%{opacity:.22}100%{opacity:0}}
.dn-h1{animation:dn-1 14s ease-in-out infinite}
.dn-h2{animation:dn-2 14s ease-in-out infinite}
.dn-h3{animation:dn-3 14s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.dn-h1,.dn-h2,.dn-h3{animation:none;opacity:.22}}
</style>
<defs><marker id="dn-a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="var(--text-secondary,#6b7280)"/></marker></defs>
<rect class="dn-box dn-l1" x="30"  y="60" width="180" height="180" rx="10"/>
<rect class="dn-hl dn-h1" x="30"  y="60" width="180" height="180" rx="10"/>
<text class="dn-t" x="56" y="92">NF4 weights</text>
<text class="dn-s" x="56" y="116">4-bit codes</text>
<text class="dn-s" x="56" y="136">blocksize = 64</text>
<text class="dn-s" x="56" y="166">~0.5 byte / param</text>
<rect class="dn-box dn-l2" x="260" y="60" width="180" height="180" rx="10"/>
<rect class="dn-hl dn-h2" x="260" y="60" width="180" height="180" rx="10"/>
<text class="dn-t" x="286" y="92">absmax</text>
<text class="dn-s" x="286" y="116">1 scale / block</text>
<text class="dn-s" x="286" y="136">stored 8-bit</text>
<text class="dn-s" x="286" y="166">+offset (mean)</text>
<rect class="dn-box dn-l3" x="490" y="60" width="180" height="180" rx="10"/>
<rect class="dn-hl dn-h3" x="490" y="60" width="180" height="180" rx="10"/>
<text class="dn-t" x="516" y="92">state2</text>
<text class="dn-s" x="516" y="116">absmax2 (fp32)</text>
<text class="dn-s" x="516" y="136">code2 (8-bit)</text>
<text class="dn-s" x="516" y="166">blocksize2 = 256</text>
<path class="dn-arr" d="M214,150 L256,150"/>
<path class="dn-arr" d="M444,150 L486,150"/>
<text class="dn-s" x="216" y="262">weights carry a scale per block; those scales are themselves quantized in state2</text>
</svg>
<figcaption>Double quantization recurses: the 8-bit absmax array that scales the NF4 blocks is itself quantized, with its fp32 scales held in state2 (saving about 0.4 bits per parameter).</figcaption>
</figure>

### The quant_state structure

All of this metadata lives in a `quant_state` object attached to the weight tensor. In Unsloth, you get at it through a one-liner in `unsloth/kernels/utils.py`:

```python
def QUANT_STATE(W):
    return getattr(W, "quant_state", None)
```

When bitsandbytes quantizes a weight to NF4, it stuffs the original tensor's `.weight` attribute with the 4-bit codes and hangs a `quant_state` off it carrying everything needed to invert the quantization. The fields, all of which `fast_dequantize` reads, are:

- `absmax` — the per-block scales, themselves 8-bit quantized under double quant.
- `shape` — the original fp16 shape, so the dequantized tensor comes back the right shape.
- `dtype` — `torch.float16` or `torch.bfloat16`, which picks the dequant kernel.
- `blocksize` — 64, the first-level block size.
- `offset` — the mean that was subtracted from the absmax values before they were quantized.
- `state2` — the nested sub-state for the double-quantized absmax, which itself carries `absmax2` (the fp32 second-level scales), `code2` (the 8-bit codebook for the quantized absmax), and `blocksize2` (256).

![The quant_state structure holds NF4 codes (4-bit, blocksize 64), an 8-bit absmax array, and a nested state2 with absmax2, code2, and blocksize2.](/imgs/blogs/unsloth-4bit-quantization-nf4-3.webp)

The figure makes the nesting explicit: the top tier is the NF4 weight codes; the middle tier is the 8-bit absmax array, one scale per 64-weight block; and the bottom tier is `state2`, the sub-state that holds the fp32 scales (`absmax2`), the 8-bit codebook (`code2`), and the second block size (`blocksize2 = 256`) needed to reconstruct the absmax array. Reading it is a two-level decompression: first you decompress the scales, then you use the scales to decompress the weights. That is precisely what `fast_dequantize` does, in that order.

## 4. Dequant on the fly: `fast_dequantize`

Here is the function at the center of QLoRA's memory story, from `unsloth/kernels/utils.py`. It is the bridge from "stored in 4 bits" to "usable in fp16," and it runs on the hot path of every forward and backward pass that touches a frozen layer.

```python
# bitsandbytes C functions bound directly:
cdequantize_blockwise_fp32      = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4  = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4  = bnb.functional.lib.cdequantize_blockwise_bf16_nf4

@torch.inference_mode
def fast_dequantize(W, quant_state = None, out = None, use_global_buffer = False):
    if quant_state is None: return W
    # unpack the QLoRA double-quant state
    absmax = quant_state.absmax; shape = quant_state.shape; dtype = quant_state.dtype
    blocksize = quant_state.blocksize; offset = quant_state.offset
    state2 = quant_state.state2                      # the quantized-absmax sub-state
    absmax2 = state2.absmax; code2 = state2.code; blocksize2 = state2.blocksize
    # ...
    # Step 1: dequantize the absmax values themselves (they were 8-bit quantized) -> fp32
    cdequantize_blockwise_fp32(get_ptr(code2), get_ptr(absmax), get_ptr(absmax2),
        ptr_out_absmax, ctypes_c_int(blocksize2), ctypes_c_int(n_elements_absmax), CUDA_STREAM)
    out_absmax += offset
    # Step 2: dequantize the NF4 weights using the recovered absmax -> fp16/bf16
    fx = cdequantize_blockwise_fp16_nf4 if dtype == torch_float16 else cdequantize_blockwise_bf16_nf4
    fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
       ctypes_c_int(blocksize), ctypes_c_int(out.numel()), CUDA_STREAM)
    return out.t() if is_transposed else out
```

There are two CUDA calls, in a deliberate order that mirrors the double-quant nesting:

**Step 1 — reconstruct the absmax.** The scales were stored as 8-bit codes (`absmax`) against an 8-bit codebook (`code2`), with their own fp32 second-level scales (`absmax2`). `cdequantize_blockwise_fp32` runs a blockwise dequantization with `blocksize2` to turn those 8-bit codes back into fp32 scales, written into `out_absmax`. Then `out_absmax += offset` adds back the mean that was subtracted before the absmax values were quantized. After this step we have the genuine per-block fp32 scales the second tier of `quant_state` was hiding.

**Step 2 — reconstruct the NF4 weights.** Now we have real scales, so we can invert the first-level quantization. The function picks `cdequantize_blockwise_fp16_nf4` or `cdequantize_blockwise_bf16_nf4` depending on `quant_state.dtype`, and runs it with the recovered `out_absmax` and the first-level `blocksize` of 64. This expands the 4-bit NF4 codes into a full fp16/bf16 tensor — the transient weight the matmul will actually use.

Note the `@torch.inference_mode` decorator: dequantization is a read-only operation on a frozen tensor, so it never needs to participate in autograd. The gradient that flows *through* a frozen layer is computed against the dequantized weight, but the weight itself carries no gradient.

The word "transient" is doing enormous work here. The crucial detail — and the reason QLoRA's memory savings survive contact with reality — is that the fp16 tensor produced by `fast_dequantize` is not stored. It is consumed by exactly one matmul and then freed. This is the same dequantize-then-`del` pattern we walked through in the [manual-backprop post](/blog/machine-learning/open-source-library/unsloth-manual-backprop), where `LoRA_MLP.backward` does literally this:

```python
# dX: dequantize the frozen 4-bit base weight on the fly, matmul, free immediately
upW = fast_dequantize(upW.t(), upW_quant)
dX = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)  # reuse X buffer
del upW                                                            # free the fp16 weight NOW
dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)
gateW = fast_dequantize(gateW.t(), gateW_quant)
dX.addmm_(de, gateW.t()); del gateW                                # and again for gate
```

Each frozen base weight is dequantized to fp16 right before it is needed, multiplied, and `del`'d on the very next line. At any instant, the fp16 expansion of at most one weight matrix is resident. That is the whole game: the 4-bit codes persist for the full training run, but the fp16 form of any given layer exists for microseconds.

<figure class="blog-anim">
<svg viewBox="0 0 720 280" role="img" aria-label="Frozen NF4 weight blocks are dequantized to fp16 one at a time, consumed by a matmul, then freed, so only one transient fp16 block exists at any moment" style="width:100%;height:auto;max-width:820px">
<title>Block-by-block dequantize-use-free</title>
<style>
.dq-frozen{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.dq-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.dq-hd{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.dq-sweep{fill:var(--accent,#6366f1);opacity:.16}
.dq-live{fill:var(--accent,#6366f1)}
.dq-livelbl{font:700 13px ui-sans-serif,system-ui;fill:var(--background,#fff);text-anchor:middle}
@keyframes dq-move{0%{transform:translateX(0)}100%{transform:translateX(540px)}}
@keyframes dq-life{0%,8%{opacity:0}14%,20%{opacity:1}26%,100%{opacity:0}}
.dq-anim{animation:dq-move 12s steps(1,end) infinite}
.dq-b0{animation:dq-life 12s ease-in-out infinite}
.dq-b1{animation:dq-life 12s ease-in-out infinite;animation-delay:2s}
.dq-b2{animation:dq-life 12s ease-in-out infinite;animation-delay:4s}
.dq-b3{animation:dq-life 12s ease-in-out infinite;animation-delay:6s}
@media (prefers-reduced-motion:reduce){.dq-anim,.dq-b0,.dq-b1,.dq-b2,.dq-b3{animation:none}.dq-b1{opacity:1}}
</style>
<text class="dq-hd" x="360" y="32">frozen NF4 base (4-bit, persistent)</text>
<rect class="dq-frozen" x="40"  y="56" width="120" height="60" rx="8"/>
<rect class="dq-frozen" x="220" y="56" width="120" height="60" rx="8"/>
<rect class="dq-frozen" x="400" y="56" width="120" height="60" rx="8"/>
<rect class="dq-frozen" x="580" y="56" width="120" height="60" rx="8"/>
<text class="dq-lbl" x="100" y="92">W0 4-bit</text>
<text class="dq-lbl" x="280" y="92">W1 4-bit</text>
<text class="dq-lbl" x="460" y="92">W2 4-bit</text>
<text class="dq-lbl" x="640" y="92">W3 4-bit</text>
<rect class="dq-sweep dq-anim" x="40" y="56" width="120" height="60" rx="8"/>
<text class="dq-hd" x="360" y="180">transient fp16 (dequantize, use, then del)</text>
<rect class="dq-live dq-b0" x="40"  y="200" width="120" height="60" rx="8"/>
<rect class="dq-live dq-b1" x="220" y="200" width="120" height="60" rx="8"/>
<rect class="dq-live dq-b2" x="400" y="200" width="120" height="60" rx="8"/>
<rect class="dq-live dq-b3" x="580" y="200" width="120" height="60" rx="8"/>
<text class="dq-livelbl dq-b0" x="100" y="236">W0 fp16</text>
<text class="dq-livelbl dq-b1" x="280" y="236">W1 fp16</text>
<text class="dq-livelbl dq-b2" x="460" y="236">W2 fp16</text>
<text class="dq-livelbl dq-b3" x="640" y="236">W3 fp16</text>
</svg>
<figcaption>fast_dequantize rebuilds one fp16 block at a time; each is consumed by its matmul and freed before the next, so the full fp16 weight never coexists in VRAM.</figcaption>
</figure>

![fast_dequantize runs two bnb CUDA calls: step 1 dequantizes the 8-bit absmax to fp32, step 2 expands the NF4 codes into a transient fp16/bf16 weight that is freed after its matmul.](/imgs/blogs/unsloth-4bit-quantization-nf4-4.webp)

The static figure lays out the same flow as a pipeline you can read at rest: `quant_state` feeds step 1 (`cdequantize_blockwise_fp32` on the absmax), the `offset` is added back to recover the fp32 per-block scales, step 2 (`cdequantize_blockwise_fp16_nf4`) uses those scales to expand the NF4 codes, and the result is the transient fp16 `W` that the matmul consumes and then `del`s. The animation above shows the temporal version: a sweep walks across the frozen 4-bit blocks while, below, each block's fp16 expansion flickers into existence and vanishes — the visual proof that the full fp16 weight never coexists in VRAM.

There is one more knob in the signature worth flagging: `use_global_buffer`. For repeated dequantizations of similarly-shaped tensors, Unsloth can hand `fast_dequantize` a reusable scratch buffer rather than allocating a fresh fp16 tensor each call. That removes the allocator churn on the hot path — you are not asking CUDA's caching allocator for a new multi-megabyte block thousands of times per step — but it does not change the memory ceiling, because the buffer is still one layer's worth.

## 5. The decode-time GEMV: skipping the full expansion entirely

Section 4 covered the training and prefill path, where activations are a real matrix (`X` is `bsz × seq_len × hidden`) and a full GEMM against the dequantized weight is the right operation. But there is a special case where even the transient fp16 expansion is wasteful: **single-token decode**.

During autoregressive generation, once the prompt is processed, each step produces exactly one new token. The activation `X` is a single vector — `bsz == 1`, `q_len == 1`. The matmul is no longer a matrix-matrix product (GEMM); it is a matrix-vector product (GEMV). And for a GEMV, dequantizing the entire weight matrix to fp16 just to multiply it by one vector is enormously wasteful: you would expand millions of weights to fp16, read them once, multiply by a single column, and throw them away. The expansion dominates the cost.

Unsloth's answer, also in `unsloth/kernels/utils.py`, is `fast_gemv`:

```python
def fast_gemv(X, W, quant_state, out = None):
    # 4-bit matrix-vector for bsz==1, q_len==1 decode:
    # cgemm_4bit_inference_naive_fp16 / cgemm_4bit_inference_naive_bf16
    ...
```

`fast_gemv` calls bitsandbytes' `cgemm_4bit_inference_naive_fp16` (or the bf16 variant). This is a **fused 4-bit GEMV kernel**: it reads the 4-bit weight codes and the absmax scales, dequantizes each weight element *inside the inner loop* exactly when it is multiplied by the corresponding activation element, and accumulates the dot products — without ever writing a full fp16 weight matrix to memory. The dequantization happens in registers, on the fly, fused into the multiply-accumulate. The full fp16 weight is never materialized at all.

![One frozen 4-bit weight feeds two matmul paths: training/prefill dequantizes the full fp16 weight then frees it, while single-token decode runs a fused 4-bit GEMV that never materializes the full weight.](/imgs/blogs/unsloth-4bit-quantization-nf4-5.webp)

The figure shows the fork. The same frozen 4-bit weight `W` (with its `quant_state`) feeds two different paths depending on the request shape. The top path, taken during training and prefill, is `fast_dequantize` to rebuild the full fp16 `W`, then `torch.matmul` for the big GEMM, then `del W` to free the fp16 buffer. The bottom path, taken during single-token decode, is `fast_gemv` calling `cgemm_4bit_inference_naive_*`, which produces the output with no full fp16 `W` ever materialized. Same stored weight, two code paths, chosen by whether the activation is a matrix or a vector.

This matters for memory-bound decode. At `bsz == 1`, decode is bottlenecked by how fast you can move weights from HBM to the compute units, not by FLOPs. Reading 4-bit codes from HBM moves a quarter as many bytes as reading fp16, so the fused 4-bit GEMV is *bandwidth-friendly* in exactly the regime where bandwidth is the constraint. It is the same principle that makes 4-bit quantization attractive for pure inference, brought into Unsloth's training/inference dual-path engine.

## 6. Why there is no measurable accuracy loss

The instinct, when you hear "we threw away 75% of the bits," is to brace for a degraded model. That instinct is wrong here, for three independent reasons, and getting this right is central to Unsloth's value proposition: the whole *Inside Unsloth* series rests on the claim that Unsloth makes **no approximations** — its kernels are exact rewrites, and its QLoRA numerics match bitsandbytes bit-for-bit.

**First, NF4 is near information-optimal for the weight distribution.** As section 2 argued, placing the 16 levels at the quantiles of a normal means each code carries roughly equal probability mass. The quantization error is spread evenly across the codes rather than concentrated where the weights are dense. For a distribution that really is approximately normal — which pretrained weights, layer by layer, largely are — this is close to the best you can do with 16 levels. The per-block absmax scaling handles the layers and blocks whose dynamic range departs from the global norm, so even outlier-heavy blocks keep their resolution.

**Second, the dequantization is exact arithmetic, not a second approximation.** This is the subtle point that distinguishes Unsloth's contribution from the underlying quantization. The information loss happens *once*, at quantization time, when the original fp16 weight is rounded to the nearest NF4 quantile. From then on, `fast_dequantize` is a faithful inversion: it reconstructs exactly the fp16 value that the stored 4-bit code represents. The two-step CUDA dequant introduces no further error beyond fp32/fp16 rounding, which is identical to what bitsandbytes does. Unsloth does not approximate the dequantization to go faster; it does the same math bitsandbytes does, which is why the numerics match.

**Third, and most importantly, the fp16 LoRA adapters absorb the residual.** This is the architectural reason QLoRA works where naive 4-bit fine-tuning would not. The base weights are frozen in 4-bit, but the trainable LoRA adapters `A` and `B` are full fp16. During training, the optimizer is free to push the adapters in whatever direction compensates for the small, fixed quantization error in the frozen base. The model is not asked to *be accurate despite* the quantization; it is asked to *learn around* it, with full-precision degrees of freedom to do so. The frozen 4-bit base provides the bulk of the capability; the fp16 adapters provide the precision where the task needs it.

Put those three together and the empirical result — that QLoRA fine-tunes match full fp16 LoRA fine-tunes on downstream metrics — stops being surprising. The quantization is faithful, the format is matched to the data, and the trainable part stays at full precision. The 4× memory saving is genuinely free.

## 7. Turning it on: the public API

You do not call `fast_dequantize` or `fast_gemv` yourself. You flip one flag, and Unsloth wires the whole machine together. From `unsloth/models/loader.py`:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = 2048,
    dtype = None,             # None -> auto: bf16 on Ampere+, else fp16
    load_in_4bit = True,      # <-- the entire subject of this post
)
```

The signature of `from_pretrained` defaults `load_in_4bit = True`, so for most Unsloth workflows 4-bit is the path you are on whether or not you typed the flag. The relevant defaults from the loader are:

```python
@staticmethod
def from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    dtype = None,                       # None -> auto (bf16 on Ampere+, else fp16)
    load_in_4bit = True,
    load_in_8bit = False,
    load_in_16bit = False,
    full_finetuning = False,
    # ...
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    max_lora_rank = 64,
    # ...
):
```

Two things are worth knowing about how Unsloth loads the 4-bit base efficiently.

**The prequantized checkpoints.** Unsloth publishes `unsloth/<model>-bnb-4bit` checkpoints on the Hub — models already stored in NF4. When you load one of these, the weights arrive on disk already 4-bit, so you never download the full fp16 weights or quantize them locally. This matters: quantizing an 8B model at load time requires briefly holding the fp16 weights in memory, which is exactly the peak you were trying to avoid. The prequantized checkpoints sidestep that — you download ~5 GB instead of ~16 GB, and the model is 4-bit from the first byte. Under the hood it is bitsandbytes doing the storage and the dequant kernels; Unsloth supplies the prequantized artifacts and the fast plumbing around them.

**`get_lora_parameters` — how the kernels find the pieces.** When a fused Unsloth kernel needs to run a LoRA-augmented matmul against a frozen 4-bit base, it has to fetch five things: the 4-bit weight, its quant state, and the two adapter matrices plus their scaling. That is exactly what `get_lora_parameters` returns, from `unsloth/kernels/utils.py`:

```python
def get_lora_parameters(proj):
    base_layer = getattr(proj, "base_layer", proj)
    W = base_layer.weight
    W_quant = getattr(W, "quant_state", None)
    # ...
    adapter = ...  # the active adapter
    A = proj.lora_A[adapter].weight
    B = proj.lora_B[adapter].weight
    return W, W_quant, A, B, proj.scaling[adapter]
```

It returns the tuple `(W, W_quant, A, B, S)`: the frozen 4-bit base weight `W`, its `quant_state` `W_quant` (which is `None` for an unquantized layer, in which case `fast_dequantize` short-circuits and returns `W` directly), the LoRA down/up matrices `A` and `B`, and the scaling factor `S`. Every fused block — `LoRA_MLP`, `LoRA_QKV`, `LoRA_W` — opens by calling `get_lora_parameters` on each projection to get this tuple, then threads it into `matmul_lora`, which conceptually computes `out = X @ dequant(W) + (X @ A) @ B * S`. The `dequant(W)` in that expression is the `fast_dequantize` call we traced; the `(X @ A) @ B * S` is the LoRA path running entirely in fp16. The frozen 4-bit base and the trainable fp16 adapter meet exactly here.

## 8. The numbers

It helps to see the whole memory story in one table, with the case the post has been building toward made concrete for an 8B model.

![Bytes per parameter and 8B weight VRAM by format: fp16 is 2.0 bytes (16 GB), int8 is 1.0 (8 GB), single-quant NF4 is ~0.5 plus scale overhead (~5 GB), and NF4 with double quant is ~0.53 (~4.3 GB).](/imgs/blogs/unsloth-4bit-quantization-nf4-6.webp)

The matrix figure lays it out: the bytes-per-param column drops from 2.0 (fp16) through 1.0 (int8) to 0.5 (single-quant NF4), and the scale-overhead column shows where the "+0.5 b/param" naive scale cost gets clawed back to "+0.13 b/param" by double quantization, landing the effective rate at 0.53 bytes/param. The 8B-weights column is the punchline: 16.0 GB → 8.0 GB → ~5.0 GB → ~4.3 GB. The color gradient from danger-red at the top to success-green at the bottom is the whole point of the post in one image.

A few named case points to anchor the numbers:

- **Llama-3.1-8B on a 24 GB card (RTX 4090 / 3090).** In fp16 the weights alone are 16 GB; add activations and any optimizer state and you OOM. In NF4 the weights are ~4.3–5 GB, leaving ~19 GB for activations, gradient checkpointing buffers, and the 8-bit paged optimizer state. This is the canonical "QLoRA on a single consumer GPU" configuration, and it is feasible only because of the 4× weight reduction.
- **A 70B model on a single 80 GB A100/H100.** 70B in fp16 is 140 GB — it does not fit on one card at all. In NF4 it is ~37 GB, which fits on a single 80 GB card with room for a QLoRA fine-tune. The 4-bit base is the difference between "needs a multi-GPU node" and "runs on one card."
- **Decode bandwidth.** During generation, reading 4-bit codes moves a quarter of the bytes that reading fp16 weights would. For memory-bound single-token decode, that bandwidth reduction is the dominant lever, and `fast_gemv` is what captures it without the expansion penalty.

These numbers compose with the rest of Unsloth's stack: the [offloaded gradient checkpointer](/blog/machine-learning/open-source-library/unsloth-manual-backprop) shrinks the activation bill, the 8-bit paged optimizer shrinks the optimizer-state bill, and NF4 shrinks the weight bill. The weight bill is the largest of the three for a frozen base, which is why it is the headline.

## Case studies: where the 4-bit base decides the outcome

### 1. The OOM at model load, before training even starts

A common first-time QLoRA failure is an out-of-memory error during `from_pretrained`, not during the training loop. The cause is almost always loading a full fp16 checkpoint and quantizing it on-device: for a brief window, both the fp16 weights and the freshly-allocated 4-bit codes are resident, and the peak exceeds VRAM. The fix is to load one of Unsloth's prequantized `unsloth/<model>-bnb-4bit` checkpoints, which arrive 4-bit from disk and never construct the fp16 weights locally. If you must quantize a custom checkpoint, do it on a larger machine once and push the 4-bit artifact to the Hub. The lesson: the 16 GB wall is not only a steady-state cost, it is a transient load-time peak, and the prequantized checkpoints exist precisely to avoid it.

### 2. The "I set load_in_4bit but VRAM barely dropped" report

Occasionally someone reports that flipping `load_in_4bit=True` saved far less than 4×. The usual culprit is that the dominant memory consumer in their run was not the weights — it was activations (long sequences, large batch) or optimizer state (full fp16 Adam on a large adapter). NF4 only compresses the *weights*. If your weight bill was already a minority of the footprint, compressing it 4× moves the total by less than you expected. The diagnosis is to read the actual breakdown: weights, activations, optimizer, gradients. NF4 is the right tool for the weight line; gradient checkpointing and paged 8-bit optimizers are the tools for the other two lines, and they compose.

### 3. The transient-fp16 leak from holding the dequantized weight

A subtle bug in custom training code that wraps Unsloth: dequantizing a frozen weight to fp16 and *keeping a reference to it* — for logging, for a custom regularizer, for a hook — defeats the entire scheme. As long as a Python reference is live, the fp16 tensor cannot be freed, and if you hold several of them, you reconstruct the 16 GB fp16 model piecemeal in VRAM. The discipline `fast_dequantize` callers follow is to `del` the result on the next line. If you genuinely need the fp16 weight, dequantize it, use it, and let it go out of scope immediately; never stash it in a list or an attribute that outlives the matmul.

### 4. The bf16-vs-fp16 dtype mismatch in the dequant path

`fast_dequantize` picks `cdequantize_blockwise_bf16_nf4` versus the fp16 variant based on `quant_state.dtype`. A class of confusing errors comes from a checkpoint quantized assuming one compute dtype while the run uses the other — the dequantized weight comes back in a dtype the downstream matmul did not expect, and you get either a hard type error or, worse, a silent precision surprise. Unsloth's `dtype=None` auto-selection (bf16 on Ampere and newer, fp16 otherwise) is designed to keep the quantization dtype and the compute dtype consistent. Override `dtype` only if you know your hardware and your checkpoint agree.

### 5. The decode-path regression when batch shape changes

`fast_gemv` is gated on `bsz == 1, q_len == 1`. A serving setup that batches multiple decode requests together (`bsz > 1`) no longer hits the fused GEMV path; it falls back to dequantize-then-GEMM. That is correct — a batched decode is a GEMM, not a GEMV — but it means the per-token bandwidth advantage of the fused 4-bit kernel applies only to single-stream decode. If you are benchmarking decode throughput and see a step change when you turn on batching, this fork is why. The 4-bit *storage* saving is unconditional; the fused-GEMV *bandwidth* saving is specific to the single-token shape.

### 6. The 70B that "should" fit but doesn't

A 70B model in NF4 is ~37 GB of weights, comfortably under 80 GB — so it is tempting to assume a QLoRA fine-tune trivially fits on one A100. In practice the run can still OOM, because the weights are only one line item. The other lines — activations across the model's full depth, the LoRA optimizer state, gradient buffers, and the transient fp16 expansion of whichever layer is currently in flight — add up. The fix is to combine NF4 with `use_gradient_checkpointing="unsloth"` (the offloaded checkpointer that moves activations to host RAM) so the activation line collapses too. NF4 makes the 70B *loadable*; the offloaded checkpointer makes it *trainable*.

## When to reach for NF4 — and when not to

Reach for the 4-bit NF4 base when the base model's weights are your VRAM bottleneck and the base is frozen — which is the default state of any LoRA/QLoRA fine-tune. If you are fine-tuning an 8B model on a 24 GB card, or a 70B on a single 80 GB card, NF4 is not optional; it is the thing that makes the run fit. It composes cleanly with gradient checkpointing and paged optimizers, and on Unsloth's defaults you are already using it.

Reach for it on the inference side too, where single-stream decode is memory-bandwidth bound: the fused 4-bit GEMV moves a quarter of the bytes and is the dominant lever for decode latency on bandwidth-limited hardware.

Be more careful in three situations. First, **full fine-tuning** (`full_finetuning=True`) — if you are updating the base weights, you cannot freeze them in 4-bit; the trainable weights need their fp16/bf16 form and their optimizer state, and NF4's read-only-base assumption no longer holds. Second, **when the base is not your bottleneck** — if activations or optimizer state dominate your footprint, compressing the weights 4× moves the total by less than you hope; spend your effort on the line that actually dominates. Third, **when you need the absolute last fraction of accuracy on a distribution-shifted base** — NF4's near-optimality assumes the weights are roughly normal; a model with deliberately heavy-tailed or multi-modal weight distributions in some layers may quantize slightly worse there, though the fp16 LoRA adapters usually absorb it. For the overwhelming majority of fine-tuning workloads on standard pretrained transformers, none of these caveats bite, and the 4-bit frozen base is simply the right default — which is exactly why `load_in_4bit=True` is the default.

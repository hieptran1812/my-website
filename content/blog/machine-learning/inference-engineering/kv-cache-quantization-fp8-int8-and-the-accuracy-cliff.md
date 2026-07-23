---
title: "KV-cache quantization: fp8, int8, and the accuracy cliff"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Quantize the thing that grows with context, not the weights: derive the exact context-length gain from halving the cache, build an fp8 and an int8 KV path into nanoserve, and find the precise place where long-context accuracy falls off a cliff."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "quantization",
    "fp8",
    "int8",
    "long-context",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

The last two posts shrank the weights. We took Llama-3.1-8B from 15 GiB down to roughly 4 GiB with int4, [built the dequant-fused kernel](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) that makes that a real decode speedup instead of a memory trick, and moved on. That win is genuine, but it is a *fixed* win: you pay it once, per model, at load time, and it never changes no matter how the traffic behaves. This post pulls a completely different lever. It shrinks the one thing in your engine that grows without bound — the KV cache — the resource that costs you nothing at token zero and one gibibyte per request by token eight thousand.

Here is the shape of the problem, restated from the [memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache). On a 4090 you fit Llama-3.1-8B and about twenty-eight simultaneous 2,048-token conversations, and then you are out of KV budget. Not out of weights — those were paid for at startup. Out of the residual, the leftover VRAM the cache lives in, which fills at 128 KiB per token of context and empties only when a request finishes. Weight quantization does not help this directly; a smaller model frees budget, but every token still costs the same 128 KiB. To move the per-token number you have to store each cached key and value in fewer bytes. That is KV-cache quantization, and the entire question of this post is: how few bytes can you get away with before the model gets quietly, measurably dumber — and exactly *where* is that edge?

![A two-panel comparison contrasting a fixed weight-quantization cost against a cache-quantization cost that grows with context and concurrency](/imgs/blogs/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff-1.webp)

That edge is real and it is a cliff, not a slope. At 8-bit the model is essentially indistinguishable from full precision on almost every task. At some 4-bit configurations it is still fine. And then, one variant lower, or ten thousand tokens of context deeper, retrieval accuracy does not degrade gracefully — it collapses. The vLLM team has a cited, concrete example of exactly this: a needle-in-a-haystack score that holds at 91% and then falls to 13% when a numerical detail crosses a threshold. By the end of this post you will have written `nanoserve/kv_quant.py` — an fp8 KV path that quantizes K and V on write into the [paged blocks](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) and dequantizes on read inside the [attention kernel](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand), plus an int8 variant — and, more importantly, you will know why the cliff is where it is, which tasks fall off it first, and why `--kv-cache-dtype fp8` is nonetheless the right default for almost everyone.

Two promises up front, both carried from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is). First: **I have no GPU and have run none of this.** Every number below is either derived from arithmetic I show you, cited from a public source that I name and link, or framed as something you should reproduce yourself with a named script and an expected range. The results tables carry a `Source` column. Second: the accuracy numbers in this post are the crux, and *every one of them* is quoted with the exact model, GPU, and context length it was measured at — because a KV-quantization accuracy number without its context length is worse than no number at all.

---

## 1. The premise: quantize the thing that grows

Start from the law the whole series keeps returning to. The bytes a request holds in the KV cache, per token of context, are

$$B_{\text{tok}} \;=\; 2 \;\cdot\; L \;\cdot\; H_{kv} \;\cdot\; d_{\text{head}} \;\cdot\; b$$

where the leading 2 is for keys-and-values, $L$ is the layer count, $H_{kv}$ the number of key/value heads, $d_{\text{head}}$ the head dimension, and $b$ the bytes per stored element. For Llama-3.1-8B that is $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes, or 128 KiB per token in bf16. Everything about weight quantization touched the parameter count and left this formula alone. KV-cache quantization touches exactly one factor: $b$. And $b$ is a multiplier out front of the whole product, so cutting it cuts the entire cache linearly.

![A layered stack showing the same per-token key-value formula priced at bf16 fp8 int8 and four-bit storage widths](/imgs/blogs/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff-2.webp)

Store keys and values in fp8 or int8 instead of bf16 and $b$ goes from 2 to 1: the cache halves. Store them in 4 bits and $b$ goes to 0.5: the cache quarters. There is no cleverness here, no kernel trick, no scheduler policy — it is one number in a product, and it drags the whole cache down with it. What makes it worth a 10,000-word post is the second half of the trade: those cached numbers are what the model *reads* to compute attention, and reading them at lower precision is not free. But before the cost, let us be precise about the benefit, because the benefit is enormous and it is the reason anyone tolerates the risk.

### From byte-width to context length

The capacity a KV budget buys is just the budget divided by the per-token cost. From the [memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), a 4090 running Llama-3.1-8B in bf16 has about 7.04 GiB of VRAM left for the cache after the 14.96 GiB of weights, the activation working set, and the CUDA context have taken their cut. Divide:

$$N_{\text{tok}} \;=\; \frac{\text{KV budget}}{B_{\text{tok}}} \;=\; \frac{7.04 \text{ GiB}}{128 \text{ KiB/token}} \;=\; 57{,}671 \text{ tokens}$$

Now halve the byte-width. Nothing else about the deployment changes — same card, same weights, same budget — but $B_{\text{tok}}$ drops to 64 KiB and the same 7.04 GiB now holds 115,342 tokens. Quarter it to 4-bit and you get 230,684 tokens. The budget did not grow; the tokens got cheaper. Because concurrency is just token capacity divided by mean sequence length, the same multiplier lands directly on your user count: 28 simultaneous 2,048-token conversations in bf16 becomes 56 in fp8 and 112 in 4-bit. This is the win, and it is a pure geometric consequence of the formula.

#### Worked example: the context-length gain on a 4090

Fix the card (RTX 4090, 24 GiB), the model (Llama-3.1-8B), and the KV budget (7.04 GiB, from the memory-math derivation). Vary only the KV dtype.

| KV dtype | $b$ | $B_{\text{tok}}$ | Tokens in 7.04 GiB | Users at 2,048 tok | Source |
| --- | --- | --- | --- | --- | --- |
| bf16 | 2 | 128 KiB | 57,671 | 28 | derived |
| fp8 / int8 | 1 | 64 KiB | 115,342 | 56 | derived |
| 4-bit | 0.5 | 32 KiB | 230,684 | 112 | derived |

Every cell is arithmetic you can redo on a napkin: divide the budget in KiB (7.04 × 1024 × 1024 = 7,381,975) by the per-token cost in KiB, then divide by the sequence length for the user count. The point is not the exact figures — they move with the card and the budget assumptions — it is the *ratios*, which are exact: fp8 is always 2× and 4-bit is always 4×, on every card, for every model, forever. That invariance is what makes KV quantization such a reliable lever. You do not have to benchmark to know the capacity gain; the formula guarantees it. What you *do* have to benchmark is the accuracy cost, and that is where the rest of the post lives.

The A100 story is the same shape at a different scale. The memory-math post put an 80 GB A100's KV budget at 62.04 GiB and its bf16 capacity at 508,231 tokens; fp8 doubles that to just over a million tokens of aggregate context, and 4-bit doubles it again. On the big card the quantization question is less about "can I serve this model at all" and more about "how many long-context users share it" — but the arithmetic is identical, because the arithmetic does not know how big the card is.

---

## 2. Two ways to spend a byte: compute in FP8 vs dequant to BF16

Here is where the two mainstream approaches split, and the split matters because it decides your throughput, not just your memory. Both store the cache at reduced precision. They differ in what happens when the attention kernel reads it.

![A branching dataflow graph showing one quantized cache feeding an FP8 compute path and a low-bit dequantize path that merge at the attention output](/imgs/blogs/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff-3.webp)

**Approach one: FP8, and compute in FP8.** Store K and V as 8-bit floating point, and then run the attention math itself in FP8 — the query-times-key scores and the scores-times-value product both happen at 8-bit, on the tensor cores' FP8 path. You never widen the cache back to bf16 on the read; the low-precision numbers go straight into the matmul. This is what vLLM does. Per the vLLM team's post ["The State of FP8 KV-Cache and Attention Quantization in vLLM"](https://vllm.ai/blog/2026-04-22-fp8-kvcache) (2026-04-22), the KV cache uses the E4M3 format exclusively, and — the detail people miss — "the full attention math runs in FP8 (QK + ScoreV), not just storage." The cache is not merely a compressed archive that gets rehydrated; it is the operand.

**Approach two: low-bit integer, and dequantize to BF16 before you compute.** Store K and V in 3 or 4 bits, but when the kernel reads a block, expand it back to bf16 and do the attention math in bf16. The cache is a compression format for *storage and bandwidth*; the compute happens at full width. This is what the vLLM team's ["TurboQuant"](https://vllm.ai/blog/2026-05-11-turboquant) post (2026-05-11) describes: KV storage down to 3–4 bits, then "dequant back to BF16 for attention." The distinction from FP8 is explicit in that post — FP8 computes in FP8; TurboQuant dequantizes.

The difference has direct performance consequences, and they cut in opposite directions from what you might guess. Computing in FP8 means the attention matmul moves half the bytes *and* runs on a faster tensor-core path, so at long context it is a throughput win, not just a memory win. Dequantizing to bf16 means you save the storage and the HBM traffic of the narrow cache, but you pay to widen every block back out on every read, on every step — and that widening is extra work in the hottest loop your engine has. That is why, as we will see in the numbers section, FP8 shows *positive* throughput under load while the 3–4 bit dequant path shows *negative* throughput despite storing fewer bytes. Fewer bytes stored is not the same as faster.

There is a subtle third thing worth naming now because it changes how you reason about accuracy. When you compute in FP8, quantization error enters the *math*, not just the *storage*. The keys and values are rounded, yes, but so are the intermediate scores, because the QK product is itself an FP8 operation. When you dequantize to bf16, the only error is the rounding of K and V; the softmax and the value-weighting run at full precision. You might expect the dequant path to therefore be more accurate at equal bit-width — and at very low bit-widths, the fact that TurboQuant keeps the compute in bf16 is part of why some of its 4-bit variants hold up. But FP8's floating-point format has far more dynamic range per bit than a 4-bit integer, so at 8 bits FP8 wins the accuracy comparison outright. Bit-width and compute-precision are two separate axes, and you have to hold both in your head.

### Why E4M3 and not a plain 8-bit integer

FP8 comes in two standard layouts, described in ["FP8 Formats for Deep Learning"](https://arxiv.org/abs/2209.05433) (Micikevicius et al., 2022): E4M3 has four exponent bits and three mantissa bits with a dynamic range to about ±448, and E5M2 has five exponent bits and two mantissa bits with a wider range but coarser precision. The vLLM FP8 post is explicit that the KV cache uses E4M3 exclusively — there is no E5M2 KV path. The reason E4M3 beats an 8-bit *integer* for this job is dynamic range. An int8 tensor with a single scale spends all 256 codes evenly across the interval from minus-max to plus-max; if 99% of your keys sit near zero and 1% are ten times larger, those 256 codes are mostly wasted covering an empty middle while the small values that carry most of the signal get one or two codes each. E4M3's exponent bits give it a logarithmic spacing — dense codes near zero, sparse codes out at the tails — which matches the actual distribution of attention keys and values far better. That is the whole argument for a float format in the cache, and it is why the default KV-quant path in production engines is fp8 rather than int8, even though both are one byte.

---

## 3. Why keys and values want different scales

Quantization is never just "cast to fewer bits." Between the cast sits a *scale*: a floating-point number you divide by before rounding and multiply by after, so the small dynamic range of the low-precision format can cover the actual range of the data. The choice of *how many* scales, and *along which axis*, is where most of the accuracy in KV quantization is won or lost — and the answer is different for K and for V. This is the single most important mechanistic fact in the whole area, and it is not obvious.

![A two-by-three matrix contrasting keys and values across outlier pattern best scale axis and the axis that fails](/imgs/blogs/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff-4.webp)

The empirical finding, reported by the KV-quantization literature — the two canonical references are ["KVQuant"](https://arxiv.org/abs/2401.18079) (Hooper et al., 2024) and ["KIVI"](https://arxiv.org/abs/2402.02750) (Liu et al., 2024) — is an asymmetry: **key tensors carry large, persistent, channel-wise outliers, while value tensors are smooth.** A "channel" here is one coordinate of the head dimension — one of the 128 numbers in a key vector — held fixed as you sweep across tokens. In the key cache, a handful of those channels have magnitudes many times larger than the rest, and they are the *same* channels across nearly every token. In the value cache, no such structure exists; magnitudes are roughly uniform across both channels and tokens.

That asymmetry dictates the scale axis, and the derivation is short. A single per-tensor scale must be large enough to not clip the biggest outlier, so it is set by the largest channel. But then every *other* channel — the smooth 99% — is quantized against a scale that is far too big for it, and its values collapse into just a few codes near zero. You have spent your whole budget resolving the outlier and starved the signal. The fix for keys is a **per-channel scale**: give each of the 128 channels its own scale, so the outlier channel gets a big scale and the quiet channels get small ones, and everybody is resolved to full effect. For values, which have no channel structure, per-channel scales buy almost nothing, and a **per-token scale** — one scale per position, shared across the head dimension — is both cheaper and sufficient.

Now tie this back to the engine, because it is not an abstract preference — it constrains your memory layout. In [the KV-append kernel post](/blog/machine-learning/inference-engineering/the-kv-cache-append-and-gather-kernel) we chose separate physical layouts for K and V precisely so each could be written with a coalesced access pattern. That same separation is what lets you scale them on different axes: the key blocks carry a per-channel scale vector of length $d_{\text{head}}$, the value blocks carry a per-token scalar. If K and V shared one interleaved buffer you could not scale them independently without a gather, and the whole scheme would fall apart. The layout decision from four posts ago is what makes the accuracy decision in this post implementable. That is the kind of coupling that only shows up when you build the whole stack.

There is a caveat that keeps this honest: vLLM's *default* fp8 KV path does none of this per-channel cleverness. Per the FP8 post, the default is **per-tensor, uncalibrated, with the scale simply set to 1.0** — the crudest possible choice — and it supports per-head scales (via FlashAttention-3) and optional calibration as opt-ins. Calibration here means the same thing it does for weights and activations — running a small representative sample through the model to fit the scales to the real data distribution — and the [companion post on activation outliers, calibration, and measuring quality loss](/blog/machine-learning/inference-engineering/activation-outliers-calibration-and-measuring-quality-loss) goes deep on that machinery; the KV-cache version reuses it wholesale. The remarkable thing, which we will get to in the numbers, is that even the crude per-tensor-scale-1.0 default recovers most of the model's accuracy on most tasks, because E4M3's dynamic range is doing the heavy lifting that per-channel scales would otherwise do. Per-channel and per-token scales are the tool you reach for when the crude default is not enough — and at 4-bit, it is not.

---

## 4. The cliff: where accuracy falls off, and why long context hits it first

Everything so far has been benefit and mechanism. Now the cost, which is the reason this post has "the accuracy cliff" in its title. KV quantization does not degrade a model the way you might hope — a little bit dumber for a little less memory, all the way down. It holds nearly perfect for a long way, and then it falls off an edge. Finding that edge, and knowing which of your workloads walks toward it fastest, is the entire skill.

![A left-to-right timeline showing recovery holding through long context then collapsing at one-hundred-thousand-plus contraction length](/imgs/blogs/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff-5.webp)

Let me give you the cited shape of it first, then derive why it happens. The vLLM FP8 post reports accuracy as a *recovery* number — the quantized model's score as a fraction of the bf16 model's score on the same task — and it stays high across an enormous range. On Hopper, with the crude per-tensor uncalibrated default: Qwen3-30B-A3B-Thinking's *lowest* recovery across their suite was 97%; Llama-3.3-70B at 128k context held roughly 97–98% AUC on long-context retrieval; Qwen3-30B-A3B-Instruct at 256k context recovered 94% when the base model was bf16 and 98% when the base model was itself FP8; and Qwen3.5-27B at a full 1M-token context fully recovered. Those are the numbers that justify "fp8 is the default." Across models, across context lengths out to a million tokens, the crude default costs you a couple of points at most.

And then there is the cliff. The same post reports that on Hopper, the FlashAttention-3 FP8 path used an imprecise FP32 accumulation, and at "100k+ contraction" — a very long reduction inside the attention computation — a needle-in-a-haystack retrieval score collapsed from 91% to 13%. Not degraded. Collapsed. The model went from finding the fact almost every time to finding it almost never, and the trigger was a numerical detail in how partial sums were accumulated across a very long sequence.

<figure class="blog-anim">
<svg viewBox="0 0 700 300" role="img" aria-label="Six bars of needle retrieval recovery rise in sequence, the first four tall and the last two collapsing to a fraction of their height once contraction length crosses one hundred thousand" style="width:100%;height:auto;max-width:820px">
<style>
.cliff-bar{transform-box:fill-box;transform-origin:50% 100%;animation:cliff-grow 11s ease-out infinite}
.cliff-good{fill:var(--accent,#6366f1)}
.cliff-bad{fill:#ef4444}
.cliff-b2{animation-delay:.5s}.cliff-b3{animation-delay:1s}.cliff-b4{animation-delay:1.5s}
.cliff-b5{animation-delay:2s}.cliff-b6{animation-delay:2.5s}
.cliff-xlab{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.cliff-val{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.cliff-note{font:600 13px ui-sans-serif,system-ui;fill:#ef4444;text-anchor:middle}
.cliff-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.cliff-mark{stroke:#ef4444;stroke-width:2;stroke-dasharray:5 5}
.cliff-scan{fill:var(--accent,#6366f1);opacity:.14;animation:cliff-sweep 11s linear infinite}
@keyframes cliff-grow{0%{transform:scaleY(0)}12%{transform:scaleY(1)}88%{transform:scaleY(1)}100%{transform:scaleY(0)}}
@keyframes cliff-sweep{0%{transform:translateX(0)}100%{transform:translateX(600px)}}
@media (prefers-reduced-motion:reduce){.cliff-bar{animation:none;transform:none}.cliff-scan{animation:none;opacity:0}}
</style>
<line class="cliff-axis" x1="40" y1="240" x2="670" y2="240"/>
<rect class="cliff-scan" x="40" y="50" width="12" height="190"/>
<rect class="cliff-bar cliff-good" x="60" y="80" width="64" height="160"/>
<rect class="cliff-bar cliff-good cliff-b2" x="160" y="80" width="64" height="160"/>
<rect class="cliff-bar cliff-good cliff-b3" x="260" y="82" width="64" height="158"/>
<rect class="cliff-bar cliff-good cliff-b4" x="360" y="85" width="64" height="155"/>
<line class="cliff-mark" x1="452" y1="60" x2="452" y2="248"/>
<rect class="cliff-bar cliff-bad cliff-b5" x="480" y="217" width="64" height="23"/>
<rect class="cliff-bar cliff-bad cliff-b6" x="580" y="217" width="64" height="23"/>
<text class="cliff-val" x="92" y="72">91%</text>
<text class="cliff-val" x="192" y="72">91%</text>
<text class="cliff-val" x="292" y="74">90%</text>
<text class="cliff-val" x="392" y="77">88%</text>
<text class="cliff-val" x="512" y="209">13%</text>
<text class="cliff-val" x="612" y="209">13%</text>
<text class="cliff-xlab" x="92" y="262">8k</text>
<text class="cliff-xlab" x="192" y="262">32k</text>
<text class="cliff-xlab" x="292" y="262">64k</text>
<text class="cliff-xlab" x="392" y="262">100k</text>
<text class="cliff-xlab" x="512" y="262">128k</text>
<text class="cliff-xlab" x="612" y="262">160k</text>
<text class="cliff-note" x="452" y="286">the cliff: imprecise FP32 accumulation</text>
</svg>
<figcaption>Needle-in-a-haystack recovery holds near 90 percent as contraction length grows, then collapses to 13 percent once it crosses roughly 100k, per vLLM's FP8 KV-cache post; the drop is a numerical accumulation failure, not a gradual precision loss.</figcaption>
</figure>

### Why long context hits the cliff first

The animation above encodes the crucial intuition, so let me derive the mechanism it draws. Attention computes, for each query, a weighted sum over *every* cached key and value — the "contraction" is that reduction, and its length is the context length. When you sum a very long sequence of products, rounding error accumulates. In exact arithmetic the order of a sum does not matter; in floating point it does, and a naive running accumulator in FP32 loses low-order bits every time it adds a small term to a large partial sum. Over a few thousand terms this is invisible. Over a hundred thousand terms, the accumulated error can swamp the actual attention weight of the one key that matters — the needle. So the quantization error and the accumulation error compound *with sequence length*, which is exactly why long-context retrieval is the task that falls off the cliff first while short-context chat sails through unharmed.

This gives you the shape of the whole risk surface: **the cache is precisely what long context relies on, so quantizing the cache attacks long context precisely where it is weakest.** A summarization prompt of 500 tokens will never notice fp8 KV. A 200k-token retrieval-over-a-codebase prompt is walking straight at the edge. The vLLM FP8 post's own accuracy table is organized by context length for exactly this reason, and the recovery numbers, while high, are lowest at the longest contexts (94% at 256k for the bf16-base Qwen model versus full recovery at shorter lengths).

The good news is that this particular cliff has a fix. The vLLM post reports that the collapse was resolved with **two-level accumulation** — splitting the long reduction into blocks, accumulating within each block and then across blocks, so no single accumulator ever eats a hundred thousand additions. This restores the needle score, at the cost of some TTFT (time to first token, the latency before the first output token appears), because the more careful accumulation is more work during the prefill that builds the cache. That TTFT cost is a theme we will hit again in the stress tests: almost every fix for a KV-quant accuracy problem costs you something on the latency side.

### The 4-bit cliff is closer and less forgiving

FP8's cliff is a numerical-accumulation edge case that a fix retires. The 4-bit cliff is more fundamental — it is the bit-width itself running out of resolution — and TurboQuant maps it precisely. Its variants are named by how many bits go to K and V and whether scales are per-channel: k8v4 is 8-bit keys and 4-bit values, 4bit-nc is 4-bit with no per-channel scaling, k3v4-nc is 3-bit keys and 4-bit values without per-channel scaling, and 3bit-nc is 3-bit throughout. On the mrcr long-context benchmark up to 256k with Qwen3-30B-A3B-Instruct, the vLLM TurboQuant post reports these AUC scores: bf16 is 45.8%, fp8 is 43.1%, k8v4 is 43.0%, 4bit-nc is 42.3% — and then k3v4-nc drops to 33.5% and 3bit-nc to 31.2%. There is the cliff, in one row of a table: the step from 4bit-nc to k3v4-nc loses nearly nine points of AUC, more than the entire gap from bf16 all the way down to 4bit-nc. Everything down to a well-scaled 4-bit is tolerable; the move to 3-bit keys is the fall.

Notice what the naming tells you, tying straight back to section 3: the survivable 4-bit variant (4bit-nc) still loses only 3.5 points from bf16, but the moment you drop keys to 3 bits *without* per-channel scaling (k3v4-nc), you lose the outlier channels entirely and the score collapses. Keys are where the outliers live; take away both the bits and the per-channel scales that protect them, and there is nothing left to resolve the channels that carry the signal. The cliff is not arbitrary — it is the exact point at which the key outliers stop being representable.

#### Worked example: 4-bit KV on a 256k retrieval task

This is the stress test the whole post has been building toward: one very long request, on a card that cannot hold it at full precision, and the temptation to go past fp8 to buy the room. Price it out for Llama-3.1-8B at a 256k context. In bf16 that single request's cache is $256{,}000 \times 128 \text{ KiB} = 31.25$ GiB — for *one* request. On a 4090 whose entire KV budget is 7.04 GiB, that request does not fit at any precision short of 4-bit; on an A100 with 62 GiB of budget it fits, but it eats half the card, so you serve exactly one such user. Halve it with fp8 to 15.6 GiB and the A100 holds four of them; quarter it with 4-bit to 7.8 GiB and it holds eight. That eightfold difference in how many 256k users share one A100 is the entire reason 4-bit KV exists. It is a capacity story, told at the one context length where capacity is scarce.

Now the bill, cited. At exactly this regime — mrcr up to 256k on Qwen3-30B-A3B-Instruct — the vLLM TurboQuant post's AUC numbers say a well-scaled 4-bit variant survives: 4bit-nc holds 42.3 against bf16's 45.8, a 3.5-point loss for a 3.4× capacity gain, which for a capacity-desperate deployment is a defensible trade. But two guardrails clamp down hard. First, the throughput cost: TurboQuant runs at 66% of bf16 with TPOT 1.5–2.5× worse at burst, because every one of those eight users pays the dequant-to-bf16 tax on every decode step. You bought room and spent speed. Second, and non-negotiable, the cliff is one variant away: drop to k3v4-nc and the AUC falls to 33.5 — a 12-point loss that turns a working retrieval system into a coin flip. The honest reading of this worked example is that 4-bit KV at 256k is a real tool for a real corner (fits eight long-context users where fp8 fits four), but only in its well-scaled form, only when you have measured recovery at 256k yourself, and only when the throughput hit is a price you can pay. It is the deep end, and the k3v4-nc floor is right there.

| Precision | 256k cache / request | 256k users on an A100 | AUC (mrcr ≤256k) | Source |
| --- | --- | --- | --- | --- |
| bf16 | 31.25 GiB | 1 | 45.8 | derived; cited: TurboQuant |
| fp8 | 15.6 GiB | 4 | 43.1 | derived; cited: TurboQuant |
| 4bit-nc | 7.8 GiB | 8 | 42.3 | derived; cited: TurboQuant |
| k3v4-nc | ~7.0 GiB | 8+ | 33.5 (cliff) | derived; cited: TurboQuant |

---

## 5. Building the fp8 and int8 KV path in nanoserve

Time to write it. We are adding `nanoserve/kv_quant.py`, and it plugs into two components we already built: the [paged block store and block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) on the write side, and the [paged attention kernel](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) on the read side. The whole feature is: quantize K and V into the block on write, keep a scale next to the block, and dequantize (or compute directly) on read.

![A dataflow graph showing a per-head scale applied to compress key and value vectors into a paged block on write and reused to reconstruct them on read](/imgs/blogs/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff-6.webp)

### The quantization primitives

Start with the smallest honest pieces: cast to fp8 with a scale, and cast to int8 with a scale. PyTorch has a native `torch.float8_e4m3fn` dtype, so the fp8 path is a real cast, not an emulation.

```python
# nanoserve/kv_quant.py
import torch

FP8_E4M3 = torch.float8_e4m3fn
FP8_MAX = 448.0  # max representable magnitude of E4M3

def quantize_fp8(x: torch.Tensor, scale: torch.Tensor | float = 1.0):
    """Cast bf16/fp16 -> fp8 E4M3 with a scale. Returns (q_fp8, scale)."""
    # Divide by scale so the data lands inside +/-448, then cast (rounds to E4M3).
    q = (x.to(torch.float32) / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_E4M3)
    return q, scale

def dequantize_fp8(q: torch.Tensor, scale: torch.Tensor | float = 1.0):
    """Reconstruct bf16 from an fp8 block and its scale."""
    return q.to(torch.bfloat16) * scale
```

Two calibration choices sit on top of these primitives, and they are the difference between the crude vLLM default and the per-head refinement. The default is uncalibrated per-tensor with `scale = 1.0`: you just cast, clipping anything past ±448. It works because attention keys and values are usually already inside that range. The calibrated version computes a scale from the data so you use the format's precision fully:

```python
def per_tensor_scale(x: torch.Tensor) -> float:
    """Calibrated per-tensor scale: map the max magnitude to the format's max."""
    amax = x.abs().max().to(torch.float32)
    return (amax / FP8_MAX).clamp_min(1e-12).item()

def per_head_scale(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """One scale per KV head. x is [..., n_kv_heads, head_dim]."""
    amax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32)  # per head
    return (amax / FP8_MAX).clamp_min(1e-12)
```

The `per_head_scale` is what FlashAttention-3 enables in vLLM: instead of one scale for the whole tensor, one per KV head, so a head whose activations run hot does not force a coarse scale on a head whose activations are quiet. It is a strict superset of per-tensor in accuracy and costs you a tiny scale vector — length `n_kv_heads` — stored per block.

### The int8 primitives, on the right axis

The int8 path is where the K/V asymmetry from section 3 becomes code. Values get a per-token scale; keys get a per-channel scale. The rounding is symmetric — map the max magnitude to 127 and round-to-nearest.

```python
def quantize_int8_per_token(x: torch.Tensor):
    """Values: one scale per token position, shared across head_dim.
    x is [n_tokens, n_kv_heads, head_dim]."""
    amax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32)  # [T, H, 1]
    scale = (amax / 127.0).clamp_min(1e-12)
    q = (x.to(torch.float32) / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)

def quantize_int8_per_channel(x: torch.Tensor):
    """Keys: one scale per channel (head_dim coord), shared across tokens.
    x is [n_tokens, n_kv_heads, head_dim]."""
    amax = x.abs().amax(dim=0, keepdim=True).to(torch.float32)   # [1, H, D]
    scale = (amax / 127.0).clamp_min(1e-12)
    q = (x.to(torch.float32) / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)

def dequantize_int8(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.bfloat16) * scale
```

Read the two `amax` calls side by side — that one axis is the entire mechanism. Values reduce over `dim=-1` (the head dimension), producing a scale per token. Keys reduce over `dim=0` (the token dimension), producing a scale per channel. Same rounding, same clamp, opposite axis, because the outliers live on opposite axes. Swap them — quantize keys per-token — and you clip the outlier channels exactly as section 3 warned, and your long-context accuracy craters.

### Wiring it into the paged store

Now the store. In the paged-cache post, a physical block held bf16 K and V. The quantized store holds fp8 (or int8) K and V *plus* a scale sidecar per block. The scale must travel with the block, because the read path cannot reconstruct the numbers without it.

```python
# nanoserve/kv_quant.py  (continued)
from dataclasses import dataclass

@dataclass
class QuantKVConfig:
    dtype: str = "fp8"          # "fp8" | "int8" | "bf16"
    granularity: str = "per_head"  # "per_tensor" | "per_head" | "per_channel_k"
    calibrated: bool = True

class QuantizedKVStore:
    """Paged KV store that keeps blocks at reduced precision plus a scale sidecar."""
    def __init__(self, cfg, num_blocks, block_size, n_kv_heads, head_dim, device):
        self.cfg = cfg
        store_dtype = {"fp8": FP8_E4M3, "int8": torch.int8,
                       "bf16": torch.bfloat16}[cfg.dtype]
        shape = (num_blocks, block_size, n_kv_heads, head_dim)
        self.k = torch.empty(shape, dtype=store_dtype, device=device)
        self.v = torch.empty(shape, dtype=store_dtype, device=device)
        # Scale sidecar: per-head is [num_blocks, n_kv_heads]; tiny next to the block.
        self.k_scale = torch.ones(num_blocks, n_kv_heads, device=device,
                                  dtype=torch.bfloat16)
        self.v_scale = torch.ones(num_blocks, n_kv_heads, device=device,
                                  dtype=torch.bfloat16)
```

The sidecar is genuinely tiny. A per-head scale is one bf16 number per KV head per block; for Llama-3.1-8B with 8 KV heads and a 16-token block, that is 16 bytes of scale next to 16 tokens × 8 heads × 128 dims × 1 byte = 16 KiB of fp8 K in the block. The scale overhead is 0.1% — it does not dent the 2× capacity win. This is why per-head scaling is nearly free and per-*channel* scaling (a scale per head-dim coordinate) is the expensive one you reserve for keys under real pressure.

### The write path: quantize on append

In the append post, the write was one `index_copy_` of bf16 K and V into the block. The quantized write adds a scale computation and a cast before the copy. This is the `reshape_and_cache` analog, now precision-aware.

```python
# nanoserve/kv_quant.py  (continued)
def quant_write(store, block_ids, slot_offsets, k_new, v_new):
    """Quantize new K/V and scatter into paged blocks.
    k_new, v_new: [n_tokens, n_kv_heads, head_dim] in bf16.
    block_ids, slot_offsets: where each token lands in the paged store."""
    cfg = store.cfg
    if cfg.dtype == "bf16":
        qk, qv = k_new.to(torch.bfloat16), v_new.to(torch.bfloat16)
        ks = vs = None
    elif cfg.dtype == "fp8":
        # Per-head scale computed from this write's tokens (calibrated), or 1.0.
        ks = per_head_scale(k_new, k_new.shape[-1]) if cfg.calibrated else 1.0
        vs = per_head_scale(v_new, v_new.shape[-1]) if cfg.calibrated else 1.0
        qk, _ = quantize_fp8(k_new, ks)
        qv, _ = quantize_fp8(v_new, vs)
    elif cfg.dtype == "int8":
        qk, ks = quantize_int8_per_channel(k_new)   # keys: per-channel
        qv, vs = quantize_int8_per_token(v_new)      # values: per-token
    # Scatter the quantized tokens into their slots.
    store.k[block_ids, slot_offsets] = qk
    store.v[block_ids, slot_offsets] = qv
    if ks is not None:
        # Store a representative per-head scale for each touched block.
        store.k_scale[block_ids] = ks.reshape(-1, ks.shape[-2]).to(torch.bfloat16) \
            if torch.is_tensor(ks) else torch.as_tensor(ks)
        store.v_scale[block_ids] = vs.reshape(-1, vs.shape[-2]).to(torch.bfloat16) \
            if torch.is_tensor(vs) else torch.as_tensor(vs)
```

One honest caveat about this toy write, and it is the same caveat that makes production KV quantization harder than it looks: a *block* spans multiple tokens, but a per-head scale computed from one write's tokens is not necessarily right for tokens written to that block in a later step. Real engines handle this by fixing the scale per block once (from a calibration pass) or by using the uncalibrated `scale = 1.0` path that has no such coupling. The uncalibrated default is not just crude — it is *stateless*, and statelessness is a real engineering virtue when tokens dribble into a block across dozens of decode steps. This is a concrete reason the vLLM default is what it is.

### The read path: compute in FP8 vs dequant to BF16

Finally the read, where the two approaches from section 2 become two code paths in the attention kernel. The gather reads quantized blocks through the block table; then either we dequantize to bf16 and run attention in bf16 (the int8 / TurboQuant style), or we scale into fp8 and run the matmul in fp8 (the vLLM style).

```python
# nanoserve/kv_quant.py  (continued)
import torch.nn.functional as F

def paged_attention_dequant(q, store, block_table, seq_len):
    """Read path A: dequantize K/V to bf16, then standard attention.
    Mirrors the int8 / TurboQuant 'dequant back to BF16' approach."""
    # Gather the sequence's blocks (schematic; the real kernel loops blocks).
    k_q, v_q, k_s, v_s = gather_blocks(store, block_table, seq_len)
    k = dequantize_int8(k_q, k_s) if store.cfg.dtype == "int8" \
        else dequantize_fp8(k_q, k_s)
    v = dequantize_int8(v_q, v_s) if store.cfg.dtype == "int8" \
        else dequantize_fp8(v_q, v_s)
    return F.scaled_dot_product_attention(q.unsqueeze(2), k, v)  # bf16 math

def paged_attention_fp8(q, store, block_table, seq_len):
    """Read path B: keep K/V in fp8 and run QK + ScoreV in fp8.
    Mirrors vLLM's 'full attention math runs in FP8' approach."""
    k_q, v_q, k_s, v_s = gather_blocks(store, block_table, seq_len)
    # Scale the query into the same fp8 domain; scores come back scaled.
    q_fp8, q_s = quantize_fp8(q, per_tensor_scale(q))
    scores = torch.matmul(q_fp8.to(torch.float32), k_q.to(torch.float32).transpose(-1, -2))
    scores = scores * (q_s * k_s.mean())            # undo the input scales
    probs = torch.softmax(scores / (q.shape[-1] ** 0.5), dim=-1)
    # ScoreV also runs against the fp8 values, scaled back on the way out.
    out = torch.matmul(probs.to(torch.float32), v_q.to(torch.float32)) * v_s.mean()
    return out.to(torch.bfloat16)
```

The `.to(torch.float32)` calls in the fp8 path are standing in for what the tensor cores do in hardware — accumulate the fp8 products in higher precision — and this is exactly the accumulation that the vLLM cliff was about. In our toy we accumulate the whole contraction in one float32 matmul, which is fine at small scale; the production kernel splits it, and getting that split right is the difference between the 91% and the 13% needle score. The toy is correct in spirit; the last 10% of correctness is entirely in how the long reduction accumulates.

### Measuring recovery honestly

You do not trust a KV-quant path because the code type-checks. You trust it because you measured the model's output against the bf16 reference and it held. Here is the harness — and I am framing it as *reproduce it yourself*, because I have no GPU and cannot report a number.

```python
# tools/measure_kv_recovery.py  — run this, report YOUR numbers
import torch, torch.nn.functional as F

def recovery(q, k_bf16, v_bf16, quant_fn):
    """Cosine similarity between bf16 attention output and quantized output.
    Run on YOUR GPU; expect ~0.999+ for fp8, lower and axis-dependent for int8/4-bit."""
    ref = F.scaled_dot_product_attention(q, k_bf16, v_bf16)
    got = quant_fn(q, k_bf16, v_bf16)
    return F.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()

# Honest measurement discipline for the timing side:
#   - torch.cuda.synchronize() before and after; time with torch.cuda.Event.
#   - warm up 10+ iterations; measure steady state, not the first (compile/alloc) step.
#   - lock clocks (nvidia-smi -lgc) so boost does not add variance.
#   - measure recovery at MULTIPLE context lengths (2k, 32k, 128k) — the cliff is
#     invisible at 2k. A single short-context number tells you nothing about long context.
```

The comment on the last line is the whole lesson of this post compressed into a code comment. A recovery number at 2k tokens will read 0.9999 for fp8 and lull you into shipping it, and then a customer pastes a 180k-token document and retrieval falls off the cliff you never measured. **Measure recovery at the context lengths you actually serve, on the tasks that actually stress the cache — long-context retrieval, not perplexity on short text.** Perplexity is a notoriously forgiving metric for KV quantization precisely because it averages over easy short-range predictions; the needle test is unforgiving because it depends on one exact key surviving a hundred thousand tokens of contraction.

---

## 6. The numbers, with provenance

Here is the aggregate scoreboard, every row tagged with where it comes from. Read the `Source` column as carefully as the numbers; the derived rows are yours to reproduce with the formula, and the cited rows are vLLM's, each with its exact setup.

| Claim | Value | Setup / meaning | Source |
| --- | --- | --- | --- |
| KV cache halved | 2× capacity | fp8/int8, any model, any card | derived |
| KV cache quartered | 4× capacity | 4-bit, any model, any card | derived |
| Llama-3.1-8B, fp8 KV | 64 KiB/token | was 128 KiB in bf16 | derived |
| 4090 fp8 capacity | 115,342 tokens | 7.04 GiB budget ÷ 64 KiB | derived |
| ITL slope, fp8 vs bf16 | 54% of bf16 | Llama-3.1-8B, H100 | cited: vLLM FP8 KV post |
| fp8 break-even context | ~7k tokens | below it, bf16 is faster | cited: vLLM FP8 KV post |
| fp8 throughput under load | +14.9% | Llama-3.1-8B, H100 | cited: vLLM FP8 KV post |
| fp8 median ITL under load | −14.8% | Llama-3.1-8B, H100 | cited: vLLM FP8 KV post |
| fp8 gain, gpt-oss-20b | ~4.8% | much smaller than 8B dense | cited: vLLM FP8 KV post |
| fp8 recovery, worst-case | 97% | Qwen3-30B-A3B-Thinking, Hopper, per-tensor uncalibrated | cited: vLLM FP8 KV post |
| fp8 recovery @128k | ~97–98% AUC | Llama-3.3-70B, Hopper | cited: vLLM FP8 KV post |
| fp8 recovery @256k | 94% / 98% | Qwen3-30B-A3B-Instruct (bf16-base / fp8-base) | cited: vLLM FP8 KV post |
| Hopper needle collapse | 91% → 13% | 100k+ contraction, imprecise FP32 accum | cited: vLLM FP8 KV post |
| TurboQuant 4bit-nc AUC | 42.3% | mrcr ≤256k, Qwen3-30B-A3B-Instruct (bf16 45.8) | cited: vLLM TurboQuant post |
| TurboQuant cliff (k3v4-nc) | 33.5% AUC | same benchmark, one variant lower | cited: vLLM TurboQuant post |
| TurboQuant capacity | 2.4× (k8v4), 3.4× (4bit-nc) | vs bf16 | cited: vLLM TurboQuant post |
| TurboQuant throughput | 80% → 66% of bf16 | dequant cost, TPOT 1.5–2.5× at burst | cited: vLLM TurboQuant post |

A few things jump out of that table that are easy to miss in prose. First, the fp8 break-even at ~7k tokens: below that context length the vLLM post reports bf16 is actually *faster*, because the fixed cost of the fp8 quantize/dequantize is not yet amortized over enough cache reads. So fp8 KV is not a free win at all context lengths — it is a win that switches on around 7k tokens and grows from there. Second, the gpt-oss-20b gain of only ~4.8% versus the 8B dense model's larger gain: the benefit depends on how much of your bandwidth the cache actually is, which for a small-active-parameter MoE is a different fraction. Third, and most important, the TurboQuant throughput line: 4-bit stores fewer bytes yet runs *slower* (66% of bf16, TPOT 1.5–2.5× worse at burst), because the dequant-to-bf16 work in the hot loop outweighs the bandwidth saved. Fewer bytes is not faster. It buys capacity, not speed.

### How to measure this without fooling yourself

Two numbers in that table need very different measurement setups, and conflating them is how teams ship a KV-quant path that looked great on the bench and fell over in production. Recovery is a *quality* number: measure it offline, on the exact tasks and context lengths you serve, against the bf16 reference — and, as the harness in section 5 insisted, at multiple context lengths, because the cliff is invisible at 2k. Throughput is a *performance* number, and it obeys a completely different rule: the fp8 win only shows up under load. The vLLM post's +14.9% was measured under load, not at batch 1, and that is not an accident. At batch 1 a single decode step reads the whole cache once and the fp8 speedup is bounded by the ~7k-token break-even; the throughput win comes from fitting *more concurrent requests* into the freed capacity, which only matters when there are more requests to fit.

That is the deeper reason tok/s at batch 1 tells you almost nothing about a KV-quant deployment. Batch 1 measures latency of one request; KV quantization's payoff is a *capacity* payoff, and capacity only converts to throughput when the server is busy. So benchmark the way [the reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) argues: open-loop, with Poisson arrivals whose rate does not depend on how fast you complete requests, reporting TTFT, TPOT, p99, and goodput at the concurrency you actually run. A closed-loop test — a fixed pool of clients each firing the next request the instant the last returns — quietly caps the offered load at your own service rate and hides exactly the queueing that fp8's extra capacity is supposed to relieve. Measure fp8 KV closed-loop at batch 1 and you will conclude it does nothing; measure it open-loop at your real arrival rate and you will see the +14.9%. Same code, opposite verdicts, and only one of them is honest about what you deployed. And always synchronize before timing, warm up past the first allocation-and-compile step, and lock the clocks — the standard discipline, doubly important here because the quantize/dequantize kernels are new code paths that the autotuner has not seen before.

#### Worked example: composing int4 weights with fp8 KV

The two levers — weight quant and cache quant — are independent, and independent wins multiply. Take the 4090 again and stack the [int4 weights](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) from two posts ago with the fp8 KV from this one.

| Configuration | Weights | KV budget | KV dtype | Token capacity | Source |
| --- | --- | --- | --- | --- | --- |
| bf16 weights + bf16 KV | 14.96 GiB | 7.04 GiB | 128 KiB/tok | 57,671 | derived |
| int4 weights + bf16 KV | ~4.5 GiB | ~17.5 GiB | 128 KiB/tok | 143,360 | derived |
| int4 weights + fp8 KV | ~4.5 GiB | ~17.5 GiB | 64 KiB/tok | 286,720 | derived |

Walk the arithmetic. Int4 weights drop the model from 14.96 GiB to roughly 4.5 GiB (4-bit for 8B params plus fp16 group scales), which hands about 10.5 GiB back to the KV budget — from 7.04 GiB to about 17.5 GiB — a 2.49× budget increase from the weight lever alone. Then fp8 KV halves the per-token cost, a further 2× on capacity. Product: 2.49 × 2 ≈ 5×, taking a card that held 57,671 tokens of context to roughly 286,720. The two techniques did not interfere; the weight win freed the space and the cache win made each token cheaper to put in it. This is the composition the capstone leans on: no single technique gets you 5×, but three of them stacked — smaller weights, cheaper cache, and a scheduler that packs the freed room — do. And critically, the accuracy costs also compose independently: int4 weights degrade the model's *computation*, fp8 KV degrades its *memory*, and you can measure and tune each without the other.

---

## 7. Case studies: FP8 KV and TurboQuant in vLLM

Two public bodies of results anchor this whole post, and both are from the production engine we benchmark against rather than build. I am citing them, not reproducing them.

**FP8 KV cache in vLLM.** The ["State of FP8 KV-Cache"](https://vllm.ai/blog/2026-04-22-fp8-kvcache) post (2026-04-22) is the definitive account. Its headline mechanism is that the full attention math — QK and ScoreV — runs in FP8, using E4M3 exclusively, with a default of per-tensor uncalibrated scaling (scale set to 1.0) and per-head scaling plus optional calibration as opt-ins via FlashAttention-3. The hardware story matters for anyone deploying: Hopper is served through FlashAttention-3 and needed the two-level accumulation fix; Blackwell B200 is served through FlashInfer and was correct from the start; FlashMLA drifts on Kimi-K2.5; and Ada plus AMD are simply not covered by that post, so do not assume the numbers transfer there. On accuracy, the crude default recovered at least 97% in the worst case across their suite and fully recovered at 1M context on Qwen3.5-27B. On performance, the ITL slope dropped to 54% of bf16 on Llama-3.1-8B on an H100 with a break-even around 7k tokens, and under load throughput rose 14.9% while median ITL fell 14.8%. The verdict from that post is unambiguous and I will quote its spirit: fp8 KV is the recommended default. The HPC-Ops work from Tencent Hunyuan even bakes it into their recommended config — the vLLM ["HPC-Ops"](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) post (2026-07-06) runs with `--kv-cache-dtype fp8_e4m3 --block-size 64` — and DeepSeek's V4 long-context work, per the vLLM ["DeepSeek V4"](https://vllm.ai/blog/2026-04-24-deepseek-v4) post (2026-04-24), pairs `--kv-cache-dtype fp8` with a 256-token block for its million-token contexts. FP8 KV is not an experimental flag; it is the shipping default across the ecosystem's most demanding deployments.

**TurboQuant in vLLM.** The ["TurboQuant"](https://vllm.ai/blog/2026-05-11-turboquant) post (2026-05-11) is the honest counterweight — the account of what going *below* 8 bits actually costs. Its approach is the dequant-to-bf16 one: store K and V in 3–4 bits, expand to bf16 for the attention compute. The capacity is real and larger than fp8: k8v4 gives 2.4× and 4bit-nc up to 3.4× versus bf16's 1×, against fp8's 2×. But two things pull it back to a niche. The throughput is *negative* — 80% falling to 66% of bf16, with TPOT (time per output token) 1.5–2.5× worse at burst — because the dequant work in the hot loop dominates. And the accuracy has the cliff we drew: on mrcr up to 256k with Qwen3-30B-A3B-Instruct, the well-scaled variants hold (fp8 43.1, k8v4 43.0, 4bit-nc 42.3 against bf16's 45.8) but k3v4-nc falls to 33.5 and 3bit-nc to 31.2. The post's own verdict, which I will quote as its conclusion, is that `--kv-cache-dtype fp8` remains the best default, with 4-bit reserved for genuinely memory-constrained deployments. It also notes a scope limit worth respecting: TurboQuant targets standard attention and GQA only. When the production engine's own 4-bit paper tells you to use fp8 by default, that is the strongest possible signal about where the cliff sits.

---

## 8. When to reach for this, and when not to

Let me make this decisive, because the honest recommendation is narrow and clear.

![A four-by-four decision matrix comparing bf16 fp8 int8 and four-bit KV across capacity accuracy throughput and when to use each](/imgs/blogs/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff-7.webp)

**Reach for fp8 KV by default.** On Hopper or Blackwell, serving any standard-attention or GQA model at meaningful context length, `--kv-cache-dtype fp8` is the right call. It doubles your capacity, the accuracy loss is a couple of points at most on the tasks that matter (worst-case 97% recovery in vLLM's suite), and under load it is faster, not slower. There is almost no deployment where doubling KV capacity for a couple of accuracy points is a bad trade — that is why it is the ecosystem default. The one context-length caveat: below about 7k tokens, per the vLLM post, bf16 is actually faster, so if your entire workload is short chat turns and you are latency-bound rather than capacity-bound, you can leave it off. But the moment you serve long documents, RAG, or agents with growing histories, turn it on.

**Reach for int8 KV only when you lack fp8 hardware.** If you are on a GPU without an efficient fp8 path, int8 with the right axes — per-channel keys, per-token values, straight from section 3 — gets you most of fp8's capacity win at bf16-level accuracy, at the cost of the dequant work in the read loop. It is the fallback, not the goal.

**Reach for 4-bit KV only under hard memory pressure, and measure the cliff first.** If you genuinely cannot fit the context you need any other way — a very long context on a small card, and you have already quantized the weights — then a *well-scaled* 4-bit variant (per-channel keys, 4-bit throughout, the 4bit-nc or k8v4 shapes) can be worth it, because it holds accuracy to within a few points per TurboQuant. But you pay in throughput (down to 66% of bf16), and you must verify recovery at your actual context length before you ship, because one variant lower is the cliff. Never reach past a well-scaled 4-bit to 3-bit keys without measuring; that is exactly the k3v4-nc collapse.

**Do not quantize the cache on sliding-window or hybrid layers.** The vLLM FP8 post is explicit that you should skip sliding-window and hybrid layers. Sliding-window layers keep only a small local cache, so there is little to save and the quantization error is pure downside. And in [hybrid attention-SSM models](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), half the layers have *no per-token KV cache at all* — the SSM layers hold a fixed-size recurrent state — so there is literally nothing for KV quantization to compress in those layers, and applying it there is at best a no-op and at worst a correctness bug. Know your architecture before you set a global flag. This is the kind of detail the capstone's decision tree exists to catch, and it forward-links into the Track K hybrid-model posts where the two cache types have to coexist.

**And the always-true rule: when in doubt, use vLLM, not your own kernel.** Everything in `nanoserve/kv_quant.py` is here to teach you the mechanism — the scale axes, the compute-vs-dequant split, the accumulation cliff. In production you want the version that already survived the 91%-to-13% bug and shipped the two-level accumulation fix. Build it once to understand it; run the real one to serve it.

---

## Key takeaways

- **KV quantization shrinks the cost that grows.** Weight quantization is a fixed, one-time win; cache quantization scales with every token of context and every concurrent user. Different lever, different math, and they compose multiplicatively.
- **The capacity gain is exact and free of benchmarking.** Halving the byte-width halves the cache and doubles the context that fits, on every card and every model — 2× for fp8/int8, 4× for 4-bit. Only the accuracy cost needs measuring.
- **FP8 computes in FP8; low-bit dequantizes to BF16.** That distinction decides throughput: fp8 is faster under load, 4-bit is slower despite storing fewer bytes, because the dequant work dominates the hot loop.
- **Keys and values want different scales.** Keys carry channel-wise outliers (per-channel scales); values are smooth (per-token scales). Swap the axes and you clip the outliers and crater long-context accuracy. The separate K/V layout from earlier posts is what makes this implementable.
- **Accuracy is a cliff, not a slope.** It holds near-perfect for a long way, then collapses — vLLM's cited 91%→13% needle drop at 100k+ contraction, and TurboQuant's 42.3%→33.5% AUC drop from 4bit-nc to k3v4-nc are the two canonical edges.
- **Long context hits the cliff first.** The cache is exactly what long context relies on, and quantization error compounds with the length of the attention contraction. Measure recovery at the context lengths you actually serve, on retrieval tasks, not perplexity.
- **Almost every fix costs latency.** The two-level accumulation that retires the FP8 cliff costs TTFT. Budget for it.
- **`--kv-cache-dtype fp8` is the default; 4-bit is a niche.** Even vLLM's own 4-bit work concludes fp8 is the best default. Reach past it only under hard memory pressure, and never onto sliding-window or hybrid layers.

---

## Further reading

- vLLM, ["The State of FP8 KV-Cache and Attention Quantization in vLLM"](https://vllm.ai/blog/2026-04-22-fp8-kvcache) (2026-04-22) — the primary source for this post: E4M3-only, compute-in-FP8, per-head scales, the accuracy-by-context-length table, and the Hopper accumulation cliff.
- vLLM, ["TurboQuant"](https://vllm.ai/blog/2026-05-11-turboquant) (2026-05-11) — the 3–4 bit dequant-to-bf16 approach, the variant-by-variant AUC cliff, and the "fp8 remains the best default" verdict.
- Hooper et al., ["KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"](https://arxiv.org/abs/2401.18079) (2024) — the per-channel-key / per-token-value asymmetry, derived and measured.
- Liu et al., ["KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"](https://arxiv.org/abs/2402.02750) (2024) — the same asymmetry pushed to 2-bit, the mechanism from section 3.
- Micikevicius et al., ["FP8 Formats for Deep Learning"](https://arxiv.org/abs/2209.05433) (2022) — E4M3 versus E5M2, the dynamic-range argument for a float cache format.
- Within this series: [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) for the byte-per-token law this post quantizes; the [paged attention kernel](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) and [KV-append kernel](/blog/machine-learning/inference-engineering/the-kv-cache-append-and-gather-kernel) this path plugs into; [dequant-fused int4 weights](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) for the composing weight lever; the [series intro](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) for where this sits in the whole stack.
- Related model-serving deep dives: [fp8 and fp4 low-precision serving](/blog/machine-learning/model-serving/fp8-fp4-low-precision-serving-deep-dive) and [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

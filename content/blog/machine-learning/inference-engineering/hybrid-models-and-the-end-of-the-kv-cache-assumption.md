---
title: "Hybrid models and the end of the KV-cache assumption: what happens when most layers stop caching"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Derive the two-term memory law of a hybrid attention plus state-space model, find the exact context length where the fixed state starts paying for itself, and see why one architectural choice retires the single assumption your whole engine was built on."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "mamba",
    "state-space-models",
    "linear-attention",
    "hybrid-models",
    "transformers",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 47
---

Every allocator, scheduler, evictor and admission-control heuristic you have written in this series rests on one sentence that nobody ever bothered to write down: **a request's cache is proportional to its sequence length.** That is why a block table works. That is why sixteen tokens per block is a sensible unit. That is why preemption can drop blocks and recompute them, why prefix sharing can dedupe by block hash, why your admission math is `free_blocks >= ceil(prompt_len / 16)`. Take the proportionality away and every one of those follows it out the door.

A whole class of models has now taken it away. Load `nvidia/Nemotron-H-8B-Base-8K` and read its config: `num_hidden_layers` is 52, and `hybrid_override_pattern` is the string `M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-`. Count the characters: 24 `M` (Mamba-2), 24 `-` (FFN), and exactly four `*` (self-attention). Four layers out of fifty-two hold a KV cache. The other twenty-four sequence-mixing layers hold something else entirely — a fixed-size tensor that is *the same number of bytes at token 1 and at token 128,000*. Your engine has no name for that object, no allocator for it, no eviction policy for it, and no way to roll it back when a speculative draft gets rejected.

![Two columns comparing the per-request memory of an all-attention model against a hybrid, showing the hybrid replaces most of the per-token cache with a small constant term](/imgs/blogs/hybrid-models-and-the-end-of-the-kv-cache-assumption-1.webp)

The figure above is the whole post in one frame, and it is worth sitting with before we go anywhere. Left column: a 48-layer all-attention model whose per-request cache is 96 KiB per token and nothing else, so at 128k context one request holds 12.0 GiB. Right column: the same 48 layers, but only every fourth one is full attention and the rest are a linear-attention variant — 24 KiB per token *plus a flat 36 MiB that does not care how long the prompt is*, which lands at 3.0 GiB for the same 128k request. Both numbers are derived below from the published `config.json` of a real model, not asserted. The interesting part is not the 4× — it is the shape change. One column is a line through the origin. The other is a line with an intercept. Everything hard about serving hybrids comes from that intercept.

By the end of this post you will be able to read any model config, classify each layer as cache-bearing or state-bearing, derive the model's exact two-term memory curve, solve for the context length at which the hybrid overtakes the dense model it replaces, and state — with numbers — how many concurrent requests each will fit on a given GPU. You will have written `nanoserve/hybrid.py`: a shape model that parses both major hybrid config conventions, computes both cost terms, finds the crossover, and plans capacity. And you will have a precise list of the five things in your engine that a hybrid breaks, which is the agenda for the rest of this track.

Two promises inherited from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and they bind especially hard here because this is a landscape post full of model names. **First: I have no GPU and have run nothing.** Every number below is derived from arithmetic I show you, cited from a paper, model card or vendor post with a link, or framed as something you should reproduce yourself with an expected range. Results tables carry a `Source` column. **Second: I do not name a model, a ratio, or an architecture I could not verify against a primary source.** Several models that plausibly belong in this post are absent for exactly that reason, and I would rather have a shorter list than a wrong one.

---

## 1. The assumption your engine is built on

Recall the law from [the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache). A dense transformer's per-request cache is a single product:

$$M_{\text{full}}(S) \;=\; \underbrace{2 \cdot L \cdot H_{kv} \cdot d \cdot b}_{B_{\text{tok}} \;=\; \text{bytes per token}} \;\cdot\; S$$

The 2 is K and V; $L$ is the number of decoder layers; $H_{kv}$ is key/value heads after grouped-query sharing; $d$ is head dimension; $b$ is bytes per stored element; $S$ is the sequence length. For Llama-3.1-8B in bf16 that is $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes — 128 KiB per token, every token, forever.

The structural fact that made your engine possible is not the size of $B_{\text{tok}}$. It is that $M_{\text{full}}$ is **linear in $S$ with zero intercept**, and — critically — that the cache is **indexable by token position**. Those two properties are what every mechanism in the engine quietly assumes:

- **The block table works** because the cache decomposes into equal chunks along the token axis. vLLM's default block size is 16 tokens, per the vLLM team's [Inside vLLM anatomy post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05), and a block's byte size is `2 * block_size * num_kv_heads * head_size * dtype_bytes`. That formula only exists because the cache has a per-token dimension.
- **Preemption works** because you can throw away a suffix and recompute it. Drop the blocks for tokens 4000 through 8000, and rerunning prefill on those tokens reconstructs them exactly. The cache is a *pure function of the prefix*, position by position.
- **Prefix sharing works** because two requests with the same first 2000 tokens produce byte-identical cache for those 2000 tokens, so you hash blocks and share them. Same anatomy post: the block hash is a function of the previous block hash plus the current tokens.
- **Admission control works** because a request's future footprint is `B_tok × (prompt_len + max_new_tokens)` — one multiply, and you know whether it fits.
- **Chunked prefill works** because you can stop halfway through a prompt, keep the partial cache, and resume. There is nothing stateful to carry across the boundary beyond the cache itself.

Every one of those five is a consequence of "linear in $S$, indexed by position." Now hold that list, because a hybrid model breaks the second, third and fifth outright, changes the arithmetic of the fourth, and forces the first to become two mechanisms instead of one.

![Layer census of a fifty-two layer hybrid showing four self-attention layers, twenty-four Mamba-2 layers and twenty-four feed-forward layers, with the two different memory costs each produces](/imgs/blogs/hybrid-models-and-the-end-of-the-kv-cache-assumption-2.webp)

The census above is the Nemotron-H-8B layout, parsed straight from the `hybrid_override_pattern` string in [its model card config](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json). Read it as three populations. Twenty-four FFN layers, which hold no sequence state at all and never did — they are pointwise, they see one position at a time, and they have been free from a cache perspective since the original transformer. Four self-attention layers, which behave exactly as your engine expects: with `num_key_value_heads` 8 and `attention_head_dim` 128 in bf16, they contribute $2 \times 4 \times 8 \times 128 \times 2 = 16{,}384$ bytes = **16 KiB per token**. And twenty-four Mamba-2 layers, which contribute **49.4 MiB flat** — a number we derive in section 4, and which does not move when $S$ goes from 512 to 128,000.

Two things to notice immediately. First, restricting attention to the *sequence-mixing* layers only (ignore the FFNs, which mix nothing), the ratio is 4 attention to 24 Mamba-2 — a **1:6 interleave**. Second, that 16 KiB per token is exactly one eighth of Llama-3.1-8B's 128 KiB per token, because the head configuration is identical and only the layer count differs: 4 instead of 32. The hybrid did not invent a cheaper attention. It just has less of it.

If you want the architecture story — why Mamba-2 rather than an RNN, what selectivity buys, how the layer placement was chosen, how the thing was trained in FP8 — that is a different post and it already exists: [the Nemotron-H deep dive](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer) walks the report end to end. This post starts where that one stops. We take the architecture as given and ask the only question an engine builder cares about: **what does it cost to serve, and what does it break.**

---

## 2. One hidden vector, two ways to mix a sequence

Before the arithmetic, get the dataflow straight, because the memory consequence falls directly out of it.

A transformer layer does two jobs: **mix across positions** (attention) and **transform each position independently** (the FFN or MoE block). Only the first job needs history. A hybrid keeps the second job identical everywhere and swaps the first job's implementation on a per-layer basis. So at layer $\ell$, the same hidden vector $h_t$ arrives, and depending on which kind of layer it is, one of two very different things happens to the engine's memory.

![Dataflow where one hidden vector feeds both a gated DeltaNet branch and a gated attention branch, each writing to a different kind of memory before both merge back into the residual stream](/imgs/blogs/hybrid-models-and-the-end-of-the-kv-cache-assumption-3.webp)

The figure traces both branches on Qwen3-Next dimensions. Down the attention path: project $h_t$ to Q, K, V; **append** the new K and V to a buffer that already holds one entry per previous token; attend over all of them. The buffer grew. It will grow again next step. Nothing you appended is ever overwritten, because attention at step $t+1$ needs the key from step 3 exactly as much as it needed it at step 4.

Down the recurrent path: project $h_t$, compute input-dependent gates, and **overwrite** a fixed-shape tensor in place. The linear recurrence in its most general form is

$$h^{\text{state}}_t \;=\; A_t \, h^{\text{state}}_{t-1} \;+\; B_t \, x_t, \qquad y_t \;=\; C_t^{\top} h^{\text{state}}_t$$

where $A_t$, $B_t$ and $C_t$ are functions of the current token — that input dependence is the "selective" part that separates Mamba-class models from classic RNNs. The state $h^{\text{state}}$ has a shape fixed at model-definition time. Token 1 writes into it. Token 100,000 writes into the same bytes. There is no append, so there is nothing to index by position, so there is nothing to block, hash, share, or partially evict.

Both branches then add back into the residual stream and hand off to the FFN or MoE block, which is why a hybrid stack is still a stack and not a new model family. The whole difference lives in what got left behind.

This is also where the *compute* asymmetry starts, though it is a later post's subject. Attention at decode reads the entire cache: $O(S)$ bytes moved per step per layer. The recurrence reads and writes one state: $O(1)$ bytes per step per layer, regardless of how much context precedes it. During prefill the recurrence cannot simply be looped 4000 times — it is computed as a chunked parallel scan (Nemotron-H's config even exposes the `chunk_size` of 128 that the scan uses), which is a genuinely different kernel from the same layer's decode-time recurrence. One layer, two kernels, chosen by phase. A later post in this track builds both; here, only note that it exists, because it is the reason a hybrid engine has more kernel launch paths than a dense one.

---

## 3. The variant zoo, and what actually distinguishes them

"Linear attention", "state-space model", "gated DeltaNet", "lightning attention" — the names suggest a taxonomy of unrelated inventions. From the engine's seat they are one family with one shared property and a handful of differences that change the constant, not the shape.

The shared property: **the layer's per-request memory is a fixed tensor whose size is a function of the model's dimensions only, never of $S$.** That is the whole thing. Everything else is a question of what the update rule is and how big the constant turns out to be.

![Comparison table of five sequence mixers scored on per-layer state size, whether the state grows with context, recall fidelity, and the share of layers each typically occupies](/imgs/blogs/hybrid-models-and-the-end-of-the-kv-cache-assumption-4.webp)

Walk the rows of that table.

**Mamba-2 / state-space duality (SSD).** The reference point. Dao and Gu's [*Transformers are SSMs*](https://arxiv.org/abs/2405.21060) reformulated the selective state-space recurrence so it can be expressed with structured matrix operations, which is what lets it train as a parallel scan and run as a sequential recurrence at decode. Its state is a genuine three-axis tensor: one matrix of shape (head dim × state dim) per head. The Nemotron-H family and IBM's Granite 4.0 both use Mamba-2 layers.

**Linear attention and gated linear attention.** Start from the observation that softmax attention's cost comes from the softmax: drop it, use a kernel feature map, and by associativity you can accumulate $\sum_i \phi(k_i) v_i^{\top}$ into a running matrix instead of storing every $k_i$. That is [Katharopoulos et al.'s linear transformer](https://arxiv.org/abs/2006.16236) — a recurrent state that is exactly a $d_k \times d_v$ matrix. Pure linear attention has no forgetting mechanism, so the running sum saturates. Gated linear attention ([Yang et al., arXiv 2312.06635](https://arxiv.org/abs/2312.06635)) adds a data-dependent decay so old contributions fade, and contributes the hardware-efficient chunked training algorithm that made the family practical.

**DeltaNet and gated DeltaNet.** Instead of *adding* to the running matrix, the delta rule *edits* it: write the new value into the slot addressed by the current key, removing whatever was there. This traces back to [Schlag et al.'s fast-weight programmers](https://arxiv.org/abs/2102.11174) framing. [Gated DeltaNet](https://arxiv.org/abs/2412.06464) (Yang, Kautz and Hatamizadeh) combines the delta rule with Mamba-2's gating, and is the linear layer used in Qwen3-Next — whose model card describes the stack literally as `12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))`, i.e. 36 gated-DeltaNet layers and 12 gated-attention layers across 48 total.

**Kimi Delta Attention (KDA).** [Kimi Linear (arXiv 2510.26692)](https://arxiv.org/abs/2510.26692) extends gated DeltaNet with channel-wise rather than head-wise gating, on the argument that finer-grained forgetting uses a finite state better. The paper reports interleaving KDA with full attention at a uniform 3:1 ratio, up to 75% KV-cache reduction for long-sequence generation, and up to 6× higher decoding throughput at 1M context — all cited from the paper, not measured here.

**Sliding-window attention.** The odd one out, and worth keeping on the table precisely because it is *not* a fixed state. A window of $W$ caps the cache at $B_{\text{tok}} \cdot W$ bytes per request instead of $B_{\text{tok}} \cdot S$ — so it is $O(1)$ in $S$ *asymptotically*, but the constant is set by $W$, and it is still a per-token-indexed cache with a ring-buffer eviction rule. Your existing block machinery mostly survives sliding windows. It does not survive a recurrent state. That distinction is the reason this post exists and a sliding-window post would not need to.

Two more families that belong on the map by name:

**Parallel hybrids.** Everything above interleaves *by layer* — some layers are attention, others are recurrent. [Falcon-H1 (arXiv 2507.22448)](https://arxiv.org/abs/2507.22448) does it *within* a layer: attention heads and Mamba-2 heads run concurrently on the same input and their outputs are concatenated before the output projection, with the head counts of each tunable independently. For the memory model this changes nothing structurally — you still have a per-token term and a constant term — but it does mean you cannot classify a layer as one or the other, which matters when you build the descriptor tables in section 8.

**Lightning attention.** MiniMax-M1 combines a linear variant it calls lightning attention with softmax attention. Per the vLLM team's [MiniMax-M1 post](https://vllm.ai/blog/2025-06-30-minimax-m1) (2025-06-30), the linear component "reduces memory 83% and inference latency 67% for 100,000-token sequences" — quoted with its setup, and note that page does *not* state the interleave ratio, so I do not give one.

I will also name what I left out. Several models that would fit this section are omitted because I could not verify a specific claim about them against a primary source within this post's research budget, and inventing an interleave ratio is exactly the failure mode this series refuses. Where you see a ratio below, it came from a model card, a config file, or a paper, and the link is next to it.

---

## 4. Deriving the state bytes: what a recurrent layer actually holds

Now the arithmetic. This is the number that becomes the intercept of the whole memory curve, so it deserves a careful derivation rather than a hand-wave.

A Mamba-2 layer holds **two** per-request tensors, and it is easy to forget the second one.

**The SSM state.** One matrix per head, of shape (head dim × state dim). With $H_m$ Mamba heads, head dimension $P$, state dimension $N$, and $b_s$ bytes per element:

$$\sigma_{\text{ssm}} \;=\; H_m \cdot P \cdot N \cdot b_s$$

**The convolution state.** Mamba-2 applies a short depthwise causal convolution before the recurrence, which means it must retain the last $k-1$ inputs to that convolution, across all its channels. The convolved width covers the inner projection plus the group-shared $B$ and $C$ projections, so with $G$ groups:

$$\sigma_{\text{conv}} \;=\; (k-1)\,\bigl(d_{\text{inner}} + 2\,G\,N\bigr)\, b_c$$

Total per layer, $\sigma = \sigma_{\text{ssm}} + \sigma_{\text{conv}}$.

#### Worked example: one Nemotron-H-8B Mamba-2 layer

Take the numbers straight from [the published config](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json): `hidden_size` 4096, `expand` 2 (so $d_{\text{inner}} = 8192$), `mamba_num_heads` 128, `mamba_head_dim` 64 (check: $128 \times 64 = 8192$, consistent), `ssm_state_size` 128, `n_groups` 8, `conv_kernel` 4, `torch_dtype` bfloat16.

$$\sigma_{\text{ssm}} \;=\; 128 \times 64 \times 128 \times 2 \;=\; 2{,}097{,}152 \text{ bytes} \;=\; 2.00 \text{ MiB}$$

$$\sigma_{\text{conv}} \;=\; 3 \times \bigl(8192 + 2 \times 8 \times 128\bigr) \times 2 \;=\; 3 \times 10{,}240 \times 2 \;=\; 61{,}440 \text{ bytes} \;=\; 60 \text{ KiB}$$

$$\sigma \;=\; 2{,}158{,}592 \text{ bytes} \;=\; 2.06 \text{ MiB per layer}$$

Across 24 Mamba-2 layers: $24 \times 2{,}158{,}592 = 51{,}806{,}208$ bytes = **49.4 MiB per request, flat.** Source: derived from the model card config. That is the number in the census figure, and it is the entire intercept.

Two honest caveats on that figure. First, many implementations keep the SSM state in fp32 for numerical stability even when the model is bf16 — if yours does, the dominant term doubles and you get 97.4 MiB instead. Check what your runtime actually allocates rather than assuming; section 10 shows how. Second, this is per *request*, not per batch slot shared: 32 concurrent requests hold 32 copies, so 1.54 GiB of pure state before a single token of KV. That constant is small next to a long-context KV cache and large next to a short-prompt one, which is exactly the tension the next section quantifies.

#### Worked example: one Qwen3-Next gated-DeltaNet layer

The delta-rule state is one matrix per *value* head, of shape (key head dim × value head dim). [The config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) lists `linear_num_value_heads` 32, `linear_key_head_dim` 128, `linear_value_head_dim` 128:

$$\sigma_{\text{delta}} \;=\; 32 \times 128 \times 128 \times 2 \;=\; 1{,}048{,}576 \text{ bytes} \;=\; 1.00 \text{ MiB per layer}$$

plus a small depthwise-conv state (`linear_conv_kernel_dim` is 4, so three retained positions over the projected width — tens of KiB, which I round away here and note as an approximation). Across 36 gated-DeltaNet layers: **36 MiB per request, flat.** Source: derived from the model card config.

Now watch the two costs behave differently over a decode run.

<figure class="blog-anim">
<svg viewBox="0 0 660 315" role="img" aria-label="Eight decode steps in which the key-value cache row gains one filled cell per step while the recurrent state box keeps the same size and only rewrites its contents" style="width:100%;height:auto;max-width:820px">
<style>
.k1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.k1-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.k1-num{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.k1-slot{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:4 4}
.k1-cell{fill:var(--accent,#6366f1)}
.k1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.k1-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:bottom}
.k1-rule{stroke:var(--border,#d1d5db);stroke-width:1}
@keyframes k1-c1{0%,1%{opacity:.10}3%,100%{opacity:1}}
@keyframes k1-c2{0%,13%{opacity:.10}15%,100%{opacity:1}}
@keyframes k1-c3{0%,26%{opacity:.10}28%,100%{opacity:1}}
@keyframes k1-c4{0%,38%{opacity:.10}40%,100%{opacity:1}}
@keyframes k1-c5{0%,51%{opacity:.10}53%,100%{opacity:1}}
@keyframes k1-c6{0%,63%{opacity:.10}65%,100%{opacity:1}}
@keyframes k1-c7{0%,76%{opacity:.10}78%,100%{opacity:1}}
@keyframes k1-c8{0%,88%{opacity:.10}90%,100%{opacity:1}}
@keyframes k1-s1{0%{transform:scaleY(.35)}12%{transform:scaleY(.80)}25%{transform:scaleY(.50)}37%{transform:scaleY(.95)}50%{transform:scaleY(.42)}62%{transform:scaleY(.72)}75%{transform:scaleY(.58)}87%{transform:scaleY(.88)}100%{transform:scaleY(.35)}}
@keyframes k1-s2{0%{transform:scaleY(.85)}12%{transform:scaleY(.45)}25%{transform:scaleY(.92)}37%{transform:scaleY(.38)}50%{transform:scaleY(.78)}62%{transform:scaleY(.50)}75%{transform:scaleY(.90)}87%{transform:scaleY(.44)}100%{transform:scaleY(.85)}}
@keyframes k1-s3{0%{transform:scaleY(.55)}12%{transform:scaleY(.95)}25%{transform:scaleY(.40)}37%{transform:scaleY(.68)}50%{transform:scaleY(.90)}62%{transform:scaleY(.36)}75%{transform:scaleY(.74)}87%{transform:scaleY(.60)}100%{transform:scaleY(.55)}}
@keyframes k1-s4{0%{transform:scaleY(.70)}12%{transform:scaleY(.38)}25%{transform:scaleY(.82)}37%{transform:scaleY(.55)}50%{transform:scaleY(.34)}62%{transform:scaleY(.92)}75%{transform:scaleY(.46)}87%{transform:scaleY(.76)}100%{transform:scaleY(.70)}}
@keyframes k1-s5{0%{transform:scaleY(.44)}12%{transform:scaleY(.66)}25%{transform:scaleY(.88)}37%{transform:scaleY(.42)}50%{transform:scaleY(.60)}62%{transform:scaleY(.84)}75%{transform:scaleY(.38)}87%{transform:scaleY(.70)}100%{transform:scaleY(.44)}}
@keyframes k1-s6{0%{transform:scaleY(.92)}12%{transform:scaleY(.52)}25%{transform:scaleY(.64)}37%{transform:scaleY(.86)}50%{transform:scaleY(.48)}62%{transform:scaleY(.58)}75%{transform:scaleY(.94)}87%{transform:scaleY(.50)}100%{transform:scaleY(.92)}}
.k1-a1{animation:k1-c1 9s ease-in-out infinite}
.k1-a2{animation:k1-c2 9s ease-in-out infinite}
.k1-a3{animation:k1-c3 9s ease-in-out infinite}
.k1-a4{animation:k1-c4 9s ease-in-out infinite}
.k1-a5{animation:k1-c5 9s ease-in-out infinite}
.k1-a6{animation:k1-c6 9s ease-in-out infinite}
.k1-a7{animation:k1-c7 9s ease-in-out infinite}
.k1-a8{animation:k1-c8 9s ease-in-out infinite}
.k1-b1{animation:k1-s1 9s steps(1,end) infinite}
.k1-b2{animation:k1-s2 9s steps(1,end) infinite}
.k1-b3{animation:k1-s3 9s steps(1,end) infinite}
.k1-b4{animation:k1-s4 9s steps(1,end) infinite}
.k1-b5{animation:k1-s5 9s steps(1,end) infinite}
.k1-b6{animation:k1-s6 9s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.k1-a1,.k1-a2,.k1-a3,.k1-a4,.k1-a5,.k1-a6,.k1-a7,.k1-a8,.k1-b1,.k1-b2,.k1-b3,.k1-b4,.k1-b5,.k1-b6{animation:none}}
</style>
<text class="k1-lbl" x="30" y="22">full-attention layer</text>
<text class="k1-sub" x="30" y="42">appends 4 KiB every decode step, never overwrites</text>
<rect class="k1-slot" x="30" y="56" width="64" height="50" rx="6"/>
<rect class="k1-slot" x="106" y="56" width="64" height="50" rx="6"/>
<rect class="k1-slot" x="182" y="56" width="64" height="50" rx="6"/>
<rect class="k1-slot" x="258" y="56" width="64" height="50" rx="6"/>
<rect class="k1-slot" x="334" y="56" width="64" height="50" rx="6"/>
<rect class="k1-slot" x="410" y="56" width="64" height="50" rx="6"/>
<rect class="k1-slot" x="486" y="56" width="64" height="50" rx="6"/>
<rect class="k1-slot" x="562" y="56" width="64" height="50" rx="6"/>
<rect class="k1-cell k1-a1" x="30" y="56" width="64" height="50" rx="6"/>
<rect class="k1-cell k1-a2" x="106" y="56" width="64" height="50" rx="6"/>
<rect class="k1-cell k1-a3" x="182" y="56" width="64" height="50" rx="6"/>
<rect class="k1-cell k1-a4" x="258" y="56" width="64" height="50" rx="6"/>
<rect class="k1-cell k1-a5" x="334" y="56" width="64" height="50" rx="6"/>
<rect class="k1-cell k1-a6" x="410" y="56" width="64" height="50" rx="6"/>
<rect class="k1-cell k1-a7" x="486" y="56" width="64" height="50" rx="6"/>
<rect class="k1-cell k1-a8" x="562" y="56" width="64" height="50" rx="6"/>
<text class="k1-sub" x="30" y="128">token 1</text>
<text class="k1-sub" x="562" y="128">token 8</text>
<line class="k1-rule" x1="30" y1="150" x2="630" y2="150"/>
<text class="k1-lbl" x="30" y="180">recurrent layer</text>
<text class="k1-sub" x="30" y="200">rewrites the same 2.06 MiB in place, size never changes</text>
<rect class="k1-box" x="30" y="212" width="300" height="82" rx="8"/>
<rect class="k1-bar k1-b1" x="48" y="226" width="30" height="54" rx="3"/>
<rect class="k1-bar k1-b2" x="94" y="226" width="30" height="54" rx="3"/>
<rect class="k1-bar k1-b3" x="140" y="226" width="30" height="54" rx="3"/>
<rect class="k1-bar k1-b4" x="186" y="226" width="30" height="54" rx="3"/>
<rect class="k1-bar k1-b5" x="232" y="226" width="30" height="54" rx="3"/>
<rect class="k1-bar k1-b6" x="278" y="226" width="30" height="54" rx="3"/>
<text class="k1-num" x="380" y="234">after 8 decode steps</text>
<text class="k1-sub" x="380" y="258">KV cache grew by 32 KiB</text>
<text class="k1-sub" x="380" y="278">recurrent state grew by 0 bytes</text>
</svg>
<figcaption>Eight decode steps of one hybrid model. The attention layer's cache gains a cell per step and keeps every one of them; the recurrent layer's state changes value on every step but never changes size. The engine has to allocate for two completely different growth laws.</figcaption>
</figure>

That contrast is the entire engineering problem. Both layers are "using memory per request." One of them is using memory you can subdivide, index, evict and share; the other is using memory that is meaningless in pieces.

---

## 5. The two-term law and where the crossover actually falls

Write the hybrid's per-request memory with $L$ total sequence-mixing layers, of which a fraction $f$ are full attention and $(1-f)$ are recurrent with per-layer state $\sigma$:

$$M_{\text{hyb}}(S) \;=\; \underbrace{2 \cdot f L \cdot H_{kv} \cdot d \cdot b}_{\text{slope: grows with } S} \;\cdot\; S \;\;+\;\; \underbrace{(1-f)\,L \cdot \sigma}_{\text{intercept: constant}}$$

Compare it to the counterfactual dense model — the same stack with every mixing layer as attention, identical head configuration:

$$M_{\text{full}}(S) \;=\; 2 \cdot L \cdot H_{kv} \cdot d \cdot b \cdot S$$

Subtract:

$$\Delta(S) \;=\; M_{\text{full}}(S) - M_{\text{hyb}}(S) \;=\; (1-f)\,L\,\Bigl[\,2\,H_{kv}\,d\,b\,S \;-\; \sigma\,\Bigr]$$

Set $\Delta = 0$ and solve. The factor $(1-f)L$ divides out of both terms, and you are left with:

$$S^{*} \;=\; \frac{\sigma}{2\,H_{kv}\,d\,b}$$

Read that carefully, because it is the non-obvious result of this post. **The crossover length does not depend on the interleave ratio at all.** It depends only on one layer's state size divided by one layer's bytes-per-token. Choosing 1:3 versus 1:7 versus 1:15 changes *how steep the savings are* past the crossover; it does not move where the crossover is. The ratio is a slope knob, not a break-even knob.

The intuition behind the algebra: each layer you convert from attention to recurrence trades "$B_{\text{tok}}$ per token, per layer" for "$\sigma$ once, per layer". That trade is profitable for a layer exactly when $S$ exceeds $\sigma / B_{\text{tok}}^{\text{layer}}$, and converting more layers just repeats the same trade. Every converted layer flips profitable at the same length.

#### Worked example: Nemotron-H-8B, the exact break-even

Per-layer attention bytes per token: $2 \times 8 \times 128 \times 2 = 4096$ bytes. Per-layer Mamba-2 state: 2,158,592 bytes, from section 4.

$$S^{*} \;=\; \frac{2{,}158{,}592}{4096} \;=\; 527 \text{ tokens}$$

Five hundred twenty-seven tokens. Not fifty thousand — five hundred. Below that length, converting a layer from attention to Mamba-2 *costs* you memory; above it, it saves. A model whose whole selling point is long context turns memory-positive before the end of a moderately long system prompt. Source: derived.

That is the apples-to-apples number. In practice you are not comparing a model to its own counterfactual; you are choosing between two real checkpoints. Against actual Llama-3.1-8B (32 layers of attention, 128 KiB per token), the hybrid still carries its four attention layers at 16 KiB per token, so the slope difference is $131{,}072 - 16{,}384 = 114{,}688$ bytes per token and:

$$S^{*}_{\text{vs Llama}} \;=\; \frac{51{,}806{,}208}{114{,}688} \;=\; 452 \text{ tokens}$$

![Timeline of context lengths from two hundred fifty-six tokens to one hundred twenty-eight thousand, marking the break-even point and the widening memory advantage past it](/imgs/blogs/hybrid-models-and-the-end-of-the-kv-cache-assumption-5.webp)

The timeline above walks that comparison across the range you actually serve. The full arithmetic, all of it derived from the two config files:

| Context $S$ | Llama-3.1-8B cache | Nemotron-H-8B (state + KV) | Ratio | Source |
| --- | --- | --- | --- | --- |
| 256 | 32.0 MiB | 49.4 + 4.0 = 53.4 MiB | **0.60× (hybrid worse)** | derived |
| 452 | 56.5 MiB | 49.4 + 7.1 = 56.5 MiB | 1.00× (break-even) | derived |
| 1,024 | 128 MiB | 49.4 + 16.0 = 65.4 MiB | 1.96× | derived |
| 4,096 | 512 MiB | 49.4 + 64.0 = 113.4 MiB | 4.51× | derived |
| 32,768 | 4.00 GiB | 49.4 MiB + 512 MiB = 561 MiB | 7.30× | derived |
| 131,072 | 16.0 GiB | 49.4 MiB + 2.00 GiB = 2.05 GiB | 7.81× | derived |

Three things fall out of that table that you would not guess from the marketing.

**The advantage has a ceiling, and you reach it fast.** As $S \to \infty$ the ratio approaches ${128/16 = 8}$, the pure layer-count ratio. By 32k you are already at 7.3, which is 91% of the theoretical maximum. Going from 32k to 128k quadruples your context and buys 7% more relative advantage. If someone tells you hybrids get better and better at longer context, the memory curve says otherwise: they get *absolutely* better (you save 14 GiB instead of 3.4 GiB) but the *ratio* is nearly saturated by 32k.

**Short-context serving is a real regression.** At 256 tokens the hybrid holds 67% more memory per request than the dense model. If your workload is a classification endpoint with 200-token prompts and 5-token outputs, a hybrid is strictly worse on memory and you should say so out loud rather than adopting one because the architecture is fashionable.

**The intercept dominates the admission decision at small $S$.** At 1k context, 76% of the hybrid's per-request memory is the constant. Your scheduler's "how many more requests fit" arithmetic is now dominated by a term that has nothing to do with the request.

#### Worked example: concurrency on one H100

Concurrency is where this stops being an accounting curiosity. Take an H100 80GB SXM — NVIDIA's [H100 datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-datasheet) lists 80 GB of HBM3 at 3.35 TB/s. In binary units 80 GB is 74.5 GiB. An 8B model in bf16 needs about 16 GB = 14.9 GiB of weights. Reserve 4 GiB for activations, workspace and CUDA context. Free for per-request memory: about 55.6 GiB.

At 128k context per request:

- **Llama-3.1-8B**: 16.0 GiB each → $\lfloor 55.6 / 16.0 \rfloor = 3$ concurrent requests.
- **Nemotron-H-8B**: 2.05 GiB each → $\lfloor 55.6 / 2.05 \rfloor = 27$ concurrent requests.

Nine times the concurrency on identical hardware. At batch 3 versus batch 27 you are on a completely different part of the throughput curve — batch 3 decode on an 8B model is deeply memory-bandwidth bound and leaves the tensor cores idle, while batch 27 starts to amortize the weight read across requests. This is the real mechanism behind reported long-context throughput gains for hybrids: not a faster kernel, a bigger batch. Source: derived, with the 80 GB / 3.35 TB/s figures cited from the datasheet.

For contrast at 8k context: Llama fits 55 requests, Nemotron-H fits 320 — a 5.8× gap, and at that point some other limit (max sequence slots, compute, the scheduler's own bookkeeping) binds before memory does.

#### The bandwidth consequence

The same two terms show up in the decode-step roofline. Per decode step at batch $n$, the bytes that must cross HBM are approximately the weights $W$ once, plus every active request's cache:

$$t_{\text{step}} \;\approx\; \frac{W \;+\; n\,\bigl(B_{\text{tok}}\,S \;+\; \Sigma\bigr)}{\text{BW}}$$

where $\Sigma$ is the total fixed state per request. At batch 1 and 128k context on an H100 at 3.35 TB/s:

- **Llama-3.1-8B**: $(16.06 + 17.18)\text{ GB} / 3.35\text{ TB/s} \approx 9.9$ ms per token, a ceiling around 101 tok/s. The cache read is *more than half the step*.
- **Nemotron-H-8B**: $(16.4 + 2.20 + 0.05)\text{ GB} / 3.35\text{ TB/s} \approx 5.6$ ms per token, a ceiling around 180 tok/s. The cache read is 12% of the step.

Source: derived, using bandwidth cited from the H100 datasheet and weight sizes from parameter counts. These are *roofline ceilings*, not predictions — real kernels hit some fraction of peak bandwidth, and the recurrent layers have their own launch overhead that this model ignores entirely. Treat them as the best case each architecture could possibly achieve, which is exactly what a roofline is for.

---

## 6. Why interleave at all, instead of going pure linear

If the recurrent layer is $O(1)$ in memory and $O(1)$ per token in bandwidth, why keep any attention? Why is every serious hybrid a *hybrid* rather than a pure linear-attention model?

![Decision tree branching on whether a layer needs exact recall, a bounded local span, or only a compressed summary, with the mixer each answer selects](/imgs/blogs/hybrid-models-and-the-end-of-the-kv-cache-assumption-6.webp)

The tree above is the design question asked once per layer. The answer is a capability argument, and it is sharp enough to be stated precisely.

**A fixed state is a lossy compressor with a hard information budget.** A Nemotron-H Mamba-2 layer holds 1,048,576 numbers. That budget is fixed whether the context is 500 tokens or 500,000. At 500,000 tokens of context the layer is holding roughly two numbers per input token. It cannot possibly retain everything, so the gating decides what to keep — and the decision is made *at write time*, before the model knows what the eventual query will be. Attention makes the same decision at *read* time, when it knows the query, which is why it can retrieve anything.

That is not a hand-wave; it is the finding of a specific body of work. [Jelassi et al., *Repeat After Me* (arXiv 2402.01032)](https://arxiv.org/abs/2402.01032) isolates *copying* as the task where transformers beat state-space models, with the argument that a fixed-size state cannot store an arbitrarily long string to be reproduced later. [Arora et al.'s *Zoology* (arXiv 2312.04927)](https://arxiv.org/abs/2312.04927) isolates *multi-query associative recall* — being asked to retrieve several different key-value bindings from context — as the specific skill on which efficient gated-convolution architectures fall behind attention, and shows the gap accounts for much of the remaining perplexity difference.

Both findings point at the same operations, and they are exactly the operations that production traffic is made of: retrieve the function signature from the file you pasted at token 3,000; quote the clause from the contract; use the tool ID that appeared eleven turns ago; copy this identifier verbatim. Every one is exact retrieval of a specific token from an arbitrary earlier position. If your engine serves RAG, agentic tool loops or code assistance, this is not an edge case — it is the workload.

So the design becomes: use the cheap compressed mixer for the bulk of sequence mixing, and keep the expensive exact-recall mixer at enough layers that the model can always reach back. **The interleave ratio is the price of recall, expressed in memory.**

Here is what the industry has actually chosen, all with primary sources:

| Model | Interleave (mixing layers) | Recurrent variant | Source |
| --- | --- | --- | --- |
| Jamba | 1 attention : 7 Mamba, in blocks of 8 | Mamba | cited: [arXiv 2403.19887](https://arxiv.org/abs/2403.19887) |
| Nemotron-H-8B | 4 attention : 24 Mamba-2 (1:6) | Mamba-2 | cited: [config.json](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json) |
| Granite 4.0 H-Small | 4 attention : 36 Mamba-2 (1:9) | Mamba-2 | cited: [model card](https://huggingface.co/ibm-granite/granite-4.0-h-small) |
| Qwen3-Next-80B-A3B | 12 attention : 36 GDN (1:3) | gated DeltaNet | cited: [model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |
| Kimi Linear | 1 attention : 3 KDA (uniform 3:1) | Kimi delta attention | cited: [arXiv 2510.26692](https://arxiv.org/abs/2510.26692) |

The spread is real — 1:3 to 1:9 — and it is not noise. It reflects genuinely different bets about how much exact recall the target workload needs, and about how much recall capacity the particular recurrent variant recovers on its own. A delta-rule state that can *overwrite* a slot behaves differently on associative recall than a decaying sum, so it is coherent for a gated-DeltaNet model to need proportionally more attention than a Mamba-2 model *or* less, depending on scale and data. What you should not do is read a ratio off one model and assume it transfers.

There is a placement rule underneath the ratio too. The Nemotron-H report ([arXiv 2504.03624](https://arxiv.org/abs/2504.03624)) places its attention layers evenly dispersed through the depth rather than clustered, so exact recall is available at multiple stages of the representation's evolution rather than only once. From the engine's point of view dispersion is mildly annoying — your two memory kinds are interleaved rather than blocked, so any layer-wise pipeline split lands unevenly — and section 8 comes back to that.

---

## 7. Building the shape model in `nanoserve`

Enough theory. This section writes `nanoserve/hybrid.py`, the module that reads a config and tells you everything above. It is deliberately pure Python with no GPU dependency, because it is the thing you want to run *before* you allocate anything.

Start with the layer taxonomy and the two real config conventions you will meet in the wild.

```python
# nanoserve/hybrid.py
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class LayerKind(str, Enum):
    ATTENTION = "attention"   # holds a KV cache, grows with S
    RECURRENT = "recurrent"   # holds a fixed state, flat in S
    FFN = "ffn"               # holds nothing across positions


def layers_from_pattern(pattern: str) -> list[LayerKind]:
    """Nemotron-H style: one character per layer.

    'M' = Mamba-2, '*' = self-attention, '-' = FFN.
    """
    table = {
        "M": LayerKind.RECURRENT,
        "*": LayerKind.ATTENTION,
        "-": LayerKind.FFN,
    }
    return [table[c] for c in pattern]


def layers_from_interval(num_layers: int, full_attention_interval: int) -> list[LayerKind]:
    """Qwen3-Next style: every Nth block is full attention, the rest linear.

    The FFN/MoE sublayer is fused into every block here, so this returns only
    the sequence-mixing kind for each block.
    """
    return [
        LayerKind.ATTENTION if (i + 1) % full_attention_interval == 0 else LayerKind.RECURRENT
        for i in range(num_layers)
    ]
```

Those two functions cover both conventions in the table from section 6. Run them on the real strings:

```python
NEMOTRON_H_8B = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"

layers = layers_from_pattern(NEMOTRON_H_8B)
census = {k: sum(1 for x in layers if x is k) for k in LayerKind}
print(len(layers), census)

qwen = layers_from_interval(48, full_attention_interval=4)
print(len(qwen), {k: sum(1 for x in qwen if x is k) for k in LayerKind})
```

```console
52 {<LayerKind.ATTENTION: 'attention'>: 4, <LayerKind.RECURRENT: 'recurrent'>: 24, <LayerKind.FFN: 'ffn'>: 24}
48 {<LayerKind.ATTENTION: 'attention'>: 12, <LayerKind.RECURRENT: 'recurrent'>: 36, <LayerKind.FFN: 'ffn'>: 0}
```

Both match the published layouts: 4 of 52 for Nemotron-H, and 12 of 48 for Qwen3-Next's `12 * (3 * DeltaNet -> 1 * Attention)`. If your parser disagrees with the model card, fix the parser before you trust a single byte figure downstream.

Now the two cost functions, one per state kind. Keep them as separate frozen dataclasses so the asymmetry is visible in the type system rather than buried in a formula.

```python
@dataclass(frozen=True)
class AttentionShape:
    """A cache that grows one entry per token, per layer."""
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2          # bf16

    def bytes_per_token_per_layer(self) -> int:
        return 2 * self.kv_heads * self.head_dim * self.dtype_bytes


@dataclass(frozen=True)
class Mamba2Shape:
    """A state whose size is fixed at model-definition time."""
    d_inner: int
    d_state: int
    n_groups: int
    conv_kernel: int
    mamba_heads: int
    mamba_head_dim: int
    state_dtype_bytes: int = 2    # some runtimes keep this in fp32; check yours
    conv_dtype_bytes: int = 2

    def ssm_state_bytes(self) -> int:
        return self.mamba_heads * self.mamba_head_dim * self.d_state * self.state_dtype_bytes

    def conv_state_bytes(self) -> int:
        channels = self.d_inner + 2 * self.n_groups * self.d_state
        return (self.conv_kernel - 1) * channels * self.conv_dtype_bytes

    def bytes_per_layer(self) -> int:
        return self.ssm_state_bytes() + self.conv_state_bytes()


@dataclass(frozen=True)
class DeltaShape:
    """Gated DeltaNet / linear attention: one matrix per value head."""
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype_bytes: int = 2

    def bytes_per_layer(self) -> int:
        return self.value_heads * self.key_head_dim * self.value_head_dim * self.dtype_bytes
```

Then the model that combines them, and the crossover solver that falls out of the section-5 algebra in four lines.

```python
@dataclass
class HybridMemoryModel:
    name: str
    layers: list[LayerKind]
    attn: AttentionShape
    state_bytes_per_recurrent_layer: int = 0

    @property
    def n_attention(self) -> int:
        return sum(1 for k in self.layers if k is LayerKind.ATTENTION)

    @property
    def n_recurrent(self) -> int:
        return sum(1 for k in self.layers if k is LayerKind.RECURRENT)

    def kv_bytes_per_token(self) -> int:
        """The slope of the memory curve."""
        return self.n_attention * self.attn.bytes_per_token_per_layer()

    def fixed_state_bytes(self) -> int:
        """The intercept of the memory curve."""
        return self.n_recurrent * self.state_bytes_per_recurrent_layer

    def bytes_at(self, seq_len: int) -> int:
        return self.fixed_state_bytes() + self.kv_bytes_per_token() * seq_len

    def crossover_vs(self, other: "HybridMemoryModel") -> float:
        """Context length at which self becomes cheaper than other. inf if never."""
        slope_gap = other.kv_bytes_per_token() - self.kv_bytes_per_token()
        if slope_gap <= 0:
            return math.inf
        intercept_gap = self.fixed_state_bytes() - other.fixed_state_bytes()
        return intercept_gap / slope_gap

    def concurrency(self, free_bytes: int, seq_len: int) -> int:
        per_request = self.bytes_at(seq_len)
        return free_bytes // per_request if per_request else 0
```

Instantiate the three models from their published configs and reproduce every number in this post:

```python
llama31_8b = HybridMemoryModel(
    name="Llama-3.1-8B",
    layers=[LayerKind.ATTENTION] * 32,
    attn=AttentionShape(kv_heads=8, head_dim=128),
)

nemotron_h_8b = HybridMemoryModel(
    name="Nemotron-H-8B",
    layers=layers_from_pattern(NEMOTRON_H_8B),
    attn=AttentionShape(kv_heads=8, head_dim=128),
    state_bytes_per_recurrent_layer=Mamba2Shape(
        d_inner=8192, d_state=128, n_groups=8, conv_kernel=4,
        mamba_heads=128, mamba_head_dim=64,
    ).bytes_per_layer(),
)

qwen3_next = HybridMemoryModel(
    name="Qwen3-Next-80B-A3B",
    layers=layers_from_interval(48, 4),
    attn=AttentionShape(kv_heads=2, head_dim=256),
    state_bytes_per_recurrent_layer=DeltaShape(
        value_heads=32, key_head_dim=128, value_head_dim=128,
    ).bytes_per_layer(),
)

MIB = 1024 ** 2
for m in (llama31_8b, nemotron_h_8b, qwen3_next):
    print(f"{m.name:22s} slope={m.kv_bytes_per_token()/1024:8.1f} KiB/tok  "
          f"intercept={m.fixed_state_bytes()/MIB:7.2f} MiB")

print("crossover Nemotron-H vs Llama:",
      round(nemotron_h_8b.crossover_vs(llama31_8b), 1), "tokens")
```

```console
Llama-3.1-8B             slope=   128.0 KiB/tok  intercept=   0.00 MiB
Nemotron-H-8B            slope=    16.0 KiB/tok  intercept=  49.41 MiB
Qwen3-Next-80B-A3B       slope=    24.0 KiB/tok  intercept=  36.00 MiB
crossover Nemotron-H vs Llama: 451.7 tokens
```

Those are the numbers from sections 4 and 5, produced by a function rather than by me. That is the point of writing the shape model: it turns "the hybrid saves memory" into a claim you can check against a config file in under a second, on any model, before you ever start a server.

Add the capacity planner, which is the form a scheduler actually needs:

```python
def capacity_table(model: HybridMemoryModel, free_bytes: int, lengths: list[int]) -> None:
    GIB = 1024 ** 3
    print(f"{model.name}  (free {free_bytes/GIB:.1f} GiB)")
    for s in lengths:
        per_req = model.bytes_at(s)
        n = model.concurrency(free_bytes, s)
        print(f"  S={s:>7,}  per-request {per_req/GIB:7.3f} GiB  ->  {n:>4} concurrent")


FREE = int(55.6 * 1024 ** 3)
capacity_table(llama31_8b,    FREE, [1024, 8192, 32768, 131072])
capacity_table(nemotron_h_8b, FREE, [1024, 8192, 32768, 131072])
```

```console
Llama-3.1-8B  (free 55.6 GiB)
  S=  1,024  per-request   0.125 GiB  ->   444 concurrent
  S=  8,192  per-request   1.000 GiB  ->    55 concurrent
  S= 32,768  per-request   4.000 GiB  ->    13 concurrent
  S=131,072  per-request  16.000 GiB  ->     3 concurrent
Nemotron-H-8B  (free 55.6 GiB)
  S=  1,024  per-request   0.064 GiB  ->   870 concurrent
  S=  8,192  per-request   0.173 GiB  ->   320 concurrent
  S= 32,768  per-request   0.548 GiB  ->   101 concurrent
  S=131,072  per-request   2.048 GiB  ->    27 concurrent
```

Read the last column as the real deliverable of a hybrid architecture. At 128k it is 27 versus 3. Whether you can *use* 27 concurrent slots depends on compute, on your scheduler, and on whether the recurrent kernels are efficient enough to keep up — but the memory ceiling that used to cap you at 3 is gone.

Finally, make the "in place" claim concrete with a real decode step. This is a reference implementation, not a fast one — a later post in this track writes the kernel — but the shapes are correct and they carry the argument.

```python
import torch


@torch.inference_mode()
def ssd_decode_step(
    state: torch.Tensor,      # (H, P, N) MUTATED IN PLACE
    conv_state: torch.Tensor, # (C, k-1)  MUTATED IN PLACE
    x: torch.Tensor,          # (H, P) this token's per-head input
    dt: torch.Tensor,         # (H,)   input-dependent step size, post-softplus
    A_log: torch.Tensor,      # (H,)   learned decay, A = -exp(A_log)
    B: torch.Tensor,          # (G, N) group-shared input projection
    C: torch.Tensor,          # (G, N) group-shared output projection
    D: torch.Tensor,          # (H,)   skip connection
) -> torch.Tensor:
    """One token through a Mamba-2 style recurrence. Returns y of shape (H, P)."""
    n_heads, n_groups = state.shape[0], B.shape[0]
    rep = n_heads // n_groups
    Bh = B.repeat_interleave(rep, dim=0)               # (H, N)
    Ch = C.repeat_interleave(rep, dim=0)               # (H, N)

    dA = torch.exp(dt * -torch.exp(A_log))             # (H,)
    state.mul_(dA[:, None, None])                      # decay: in place
    state.addcmul_(x[..., None], (dt[:, None] * Bh)[:, None, :])  # write: in place

    y = torch.einsum("hpn,hn->hp", state, Ch)          # read out
    return y + D[:, None] * x
```

Drive it for a few steps and print what changed:

```python
H, P, N, G, K = 128, 64, 128, 8, 4
state = torch.zeros(H, P, N, dtype=torch.bfloat16)
conv_state = torch.zeros(8192 + 2 * G * N, K - 1, dtype=torch.bfloat16)

print("state bytes at step 0:", state.numel() * state.element_size())
for step in range(4):
    x = torch.randn(H, P, dtype=torch.bfloat16)
    dt = torch.rand(H, dtype=torch.bfloat16) * 0.1
    y = ssd_decode_step(
        state, conv_state, x, dt,
        A_log=torch.zeros(H), B=torch.randn(G, N, dtype=torch.bfloat16),
        C=torch.randn(G, N, dtype=torch.bfloat16), D=torch.ones(H),
    )
print("state bytes at step 4:", state.numel() * state.element_size())
print("state shape unchanged:", tuple(state.shape))
```

```console
state bytes at step 0: 2097152
state bytes at step 4: 2097152
state shape unchanged: (128, 64, 128)
```

Two megabytes at step 0. Two megabytes at step 4. Two megabytes at step 400,000. Contrast that with the equivalent attention step, which would have appended four entries to a buffer you now have to keep, index and eventually evict.

Notice what the signature makes obvious: `state` and `conv_state` are *mutated*. There is no version of this function that produces the state at step $t$ without having run every step from 1 to $t-1$. That single property is what section 8 is about.

---

## 8. What this actually does to your engine

Now collect the damage. Go back to the five properties from section 1 and check each one against a state that is mutated in place and has no per-token index.

![Two columns contrasting an engine with a single uniform block table against one that must manage a paged cache and a fixed state together](/imgs/blogs/hybrid-models-and-the-end-of-the-kv-cache-assumption-7.webp)

**1. The block table becomes two allocators.** The KV pool is unchanged — paged blocks of 16 tokens, a free list, a per-request block table. The state pool is a completely different object: a fixed number of fixed-size slots, one per in-flight request, allocated at admission and freed at completion, with no internal structure at all. Two allocators means two exhaustion conditions and two fragmentation stories, and admission has to satisfy both.

This is not hypothetical. The vLLM team's [Qwen3-Next post](https://vllm.ai/blog/2025-09-11-qwen3-next) (2025-09-11) describes their solution verbatim: "vLLM automatically tunes the 'logical' block size of the full attention layers to ensure that the state for the full attention layers and linear attention layers occupy the same amount of 'physical' GPU memory." That is a clever dodge — force both kinds into a common physical granularity so one allocator can serve both — and it is worth understanding as an existence proof that the two-allocator problem is solvable without two allocators, at the cost of a block size you no longer choose freely.

The same team's [hybrid SSM disaggregation post](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) states the underlying difficulty even more directly: full-attention layers use a uniform per-token KV layout while SSM layers hold a fixed conv state plus temporal state with no per-token dimension, so a single descriptor or index format cannot address both. Their answer is dual descriptor views over the same physical memory — one indexing attention blocks, another indexing SSM state — which lets a transfer happen without reshuffling. They also report decomposing the conv state into three sub-projections and laying the SSM state out as `(dim, state_len)` so each rank reads only its own tensor-parallel slice, saving roughly 50 MB per request of padding traffic on bf16. All of that is cited from their post; I have measured none of it.

**2. Preemption stops being free.** In a dense engine, preemption is cheap because eviction is reversible: drop blocks, recompute later, get identical bytes back. A recurrent state is not recoverable from a suffix. To restore the state at position 4000 you must replay the recurrence from position 0 — the full prefill, not a piece of it. So your preemption policy now has an asymmetric cost: evicting a hybrid request's KV blocks costs a partial recompute, but evicting its state costs a *total* one. In practice this pushes toward swapping the state to host memory rather than dropping it, since 2 MiB per layer over PCIe is far cheaper than re-prefilling 4000 tokens. Your evictor needs to know the difference.

**3. Prefix sharing does not transfer.** Two requests sharing a 2000-token prefix produce byte-identical KV for those tokens, which is why block-hash sharing works. They also produce an identical *state* after those 2000 tokens — but the state is a single object at one point in the sequence, not a sequence of shareable pieces. You cannot share "the first 1,500 tokens' worth" of it. The best you can do is checkpoint the state at chosen prefix boundaries and copy it wholesale to a new request, which is a different mechanism with a different hit-rate profile: it only helps on *exact* prefix boundaries you decided to snapshot in advance, and each snapshot costs the full per-request state in memory. vLLM's Qwen3-Next post lists prefix caching as a roadmap gap for hybrids rather than a solved problem, which tells you where the difficulty sits.

**4. Admission arithmetic gains a constant term.** `free_blocks >= ceil(prompt_len / 16)` becomes `free_blocks >= ceil(prompt_len / 16) AND free_state_slots >= 1`. The second condition is the one that will surprise you, because state slots are consumed by *request count* and not by *token count*. A burst of 200 tiny requests can exhaust your state pool while the KV pool is 95% free. If your admission controller only checks blocks, you will get an allocation failure at a point where every dashboard says you have memory.

**5. Speculative decoding needs a snapshot path.** Draft $k$ tokens, verify, accept $j \le k$, discard the rest. In a dense engine, rollback is trivial: move the cache write pointer back to position $\text{pos} + j$. The bytes past that are garbage nobody will read. In a hybrid, the recurrent state was overwritten $k$ times in place, and there is no pointer to move. Rolling back means having saved a copy before you started drafting, and restoring it. That is $\sigma \times n_{\text{recurrent}}$ bytes of copy per speculation round per request — 49.4 MiB for Nemotron-H-8B. At batch 27 that is 1.3 GiB of copies per round. The vLLM hybrid SSM post explicitly flags the interaction between speculative decoding and hybrid models as "not extensively validated", which is a candid way of saying this is live territory.

There is a sixth item that is not in the section-1 list because dense engines never had to think about it: **chunked prefill has to respect the scan's chunk boundary.** The recurrent layer's prefill is a chunked parallel scan with a chunk size baked into the model config (128 for Nemotron-H). Your scheduler's prefill chunking now interacts with the kernel's chunking, and the naive choice — cut wherever the token budget runs out — either forces a partial-chunk path in the kernel or wastes work. This is exactly the kind of coupling between scheduler and kernel that a dense engine got to ignore.

Finally, the honest status of engine support. The vLLM V1 architecture post ([2025-01-27](https://vllm.ai/blog/2025-01-27-v1-alpha-release)) listed Mamba and Jamba models as *not supported* at release. The anatomy post ([2025-09-05](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)) explicitly excludes hybrid KV handling from its write-up. The Model Runner V2 post ([2026-03-24](https://vllm.ai/blog/2026-03-24-mrv2)) lists linear-attention models — it names Qwen3.5 and Nemotron 3 Super — among the things the new runner does not yet cover. Three snapshots across fourteen months of the most actively developed open inference engine, and hybrid support is the recurring asterisk. That is not a criticism of vLLM; it is a measure of how much machinery the assumption in section 1 was actually holding up.

---

## 9. Case studies and public numbers

Four public results, each cited with its setup, and each carrying a caveat about what it does and does not tell you.

**Nemotron-H (NVIDIA).** [The technical report (arXiv 2504.03624)](https://arxiv.org/abs/2504.03624) reports that Nemotron-H-56B matches or exceeds Llama-3.1-70B on 16 of 17 benchmarks while delivering roughly 2.4× the long-context inference throughput, and that Nemotron-H-8B is about 3× faster than Llama-3.1-8B. What the number means: throughput at long context, where the dense model is memory-bound and batch-limited. What it does not mean: a 3× speedup on your short-prompt chat endpoint. Our concurrency derivation above explains the mechanism — the win is batch size, and batch size only helps where memory was the binding constraint.

**Kimi Linear (Moonshot AI).** [arXiv 2510.26692](https://arxiv.org/abs/2510.26692) reports KDA interleaved with full attention at a uniform 3:1 ratio, up to 75% KV-cache reduction for long-sequence generation, and up to 6× higher decoding throughput at 1M context. Note the sanity check against our law: a 3:1 interleave keeps one attention layer in four, so the KV cache is 25% of the counterfactual — a 75% reduction is exactly the arithmetic, which is a good sign the reported figure is the structural one rather than a benchmark artifact.

**MiniMax-M1.** Per the vLLM team's [post (2025-06-30)](https://vllm.ai/blog/2025-06-30-minimax-m1), the lightning-attention component reduces memory 83% and inference latency 67% for 100,000-token sequences, on a MoE with 456B total and about 45.9B active parameters. Caveat, and it is the one the series cares about: that page does not state the interleave ratio or describe how the linear state and the attention cache coexist in the allocator, so it is evidence about outcomes and not about mechanism.

**Hybrid disaggregation (vLLM).** The [hybrid SSM disaggregation post (2026-04-21)](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) reports a `Nemotron-3-Super-120B-A12B-FP8` deployment on 8×H200 with prefill and decode disaggregated (prefill TP4 plus decode TP4) that "Pareto-dominates the co-located baseline at higher batch sizes", and gives a worked configuration for `Nemotron-3-Nano-30B-A3B-FP8` at TP=2 with 52 layers alternating Mamba and full attention, five hybrid-memory-allocator groups, and six shared KV tensors. The value of this one for our purposes is not the throughput claim — it is the configuration detail, because it shows what the bookkeeping looks like once someone has actually built it. It also names its limits: Mamba1 unsupported, gated DeltaNet pending, mixed block sizes unsupported, speculative decoding not extensively validated.

Put together, the public record says something consistent: the memory result is structural and reliable (it is arithmetic, and it reproduces), while the throughput result is workload-shaped and every single report of it comes with a long-context qualifier.

---

## 10. Measuring it honestly

You should not take my derived numbers on faith, and you do not have to — the memory curve is one of the easiest things in this whole series to verify, because it is a straight line and you only need two points to check both of its parameters.

**The two-point fit.** Run a single request at two context lengths, record the steady-state allocation after prefill, and solve for slope and intercept:

```python
import torch

def measure_curve(run_prefill, s1: int, s2: int) -> tuple[float, float]:
    """Return (bytes_per_token, fixed_bytes) measured from two prefills."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()      # weights only

    run_prefill(s1)                            # warm-up + first point
    torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated() - base

    run_prefill(s2)                            # second point
    torch.cuda.synchronize()
    m2 = torch.cuda.memory_allocated() - base

    slope = (m2 - m1) / (s2 - s1)
    intercept = m1 - slope * s1
    return slope, intercept
```

Run it with `s1=2048` and `s2=32768` — far apart, so allocator rounding is a small fraction of the difference. Then compare against what `nanoserve/hybrid.py` predicted from the config:

```bash
python -m nanoserve.tools.measure_curve --model nvidia/Nemotron-H-8B-Base-8K --s1 2048 --s2 32768
```

**What you should see.** On a 24 GB card such as an RTX 4090, an 8B-class hybrid in bf16 should give you a slope within a few percent of 16 KiB per token and an intercept somewhere between 49 and 100 MiB — the low end if the runtime keeps SSM states in bf16, near the high end if it keeps them in fp32. If your measured intercept is roughly double the derived one, that is the fp32 state, not a bug. If the slope is off by an integer factor, count your attention layers again; the parser is the usual culprit. Frame it as a check, not a benchmark: you are validating a formula, and a formula either reproduces or it does not.

**Four ways this measurement lies to you**, in rough order of how often they bite:

- **Peak versus steady state.** `max_memory_allocated()` includes transient activation peaks during prefill, which scale with the chunk size, not with the cache. Use `memory_allocated()` at a quiescent point, and reset peak stats between runs if you want both.
- **Caching allocator rounding.** PyTorch's allocator rounds up and holds freed blocks. Two points far apart, plus `empty_cache()` before you start, keeps the rounding error inside the noise.
- **The runtime may preallocate.** A production engine grabs a large fraction of VRAM up front and manages it internally, so `torch.cuda.memory_allocated()` reports the pool, not your request. Read the engine's own cache-usage metric instead, and expect the two-point fit to need the engine's per-request accounting rather than torch's.
- **Batch size hides the intercept.** At batch $n$, the constant term is multiplied by $n$ while the weights are not. If you fit at batch 8 and apply the result at batch 1 you will attribute state bytes to weights. Fit at batch 1.

**And on timing, the usual discipline still applies**, because sooner or later you will want to know whether the recurrent kernels are actually fast: warm up until the numbers stop moving, `torch.cuda.synchronize()` before and after every timed region or use CUDA events, lock clocks if you can, and measure in steady state rather than on the first token. Then remember that batch-1 tok/s tells you almost nothing about a server — an open-loop load generator with Poisson arrivals reporting TTFT, TPOT and p99 tells you something. That machinery is [the reproducible-benchmark post's](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) subject and it does not change for hybrids. What *does* change is that you must report the context length alongside every throughput number, because for a hybrid the context length is the independent variable that decides whether the architecture helps at all.

| Claim | Value | Source |
| --- | --- | --- |
| Nemotron-H-8B Mamba-2 state per layer | 2.06 MiB | derived from config.json |
| Nemotron-H-8B total fixed state | 49.4 MiB per request | derived |
| Nemotron-H-8B KV | 16 KiB per token | derived from config.json |
| Crossover vs its own dense counterfactual | 527 tokens | derived |
| Crossover vs Llama-3.1-8B | 452 tokens | derived |
| Concurrency at 128k on one H100 | 3 vs 27 requests | derived (80 GB / 3.35 TB/s cited: NVIDIA H100 datasheet) |
| Nemotron-H long-context throughput | 2.4× (56B), 3× (8B) | cited: arXiv 2504.03624 |
| Kimi Linear KV reduction / 1M throughput | up to 75% / up to 6× | cited: arXiv 2510.26692 |
| MiniMax-M1 lightning attention at 100k | 83% memory, 67% latency reduction | cited: vLLM blog 2025-06-30 |
| Your engine's measured slope and intercept | within a few percent of derived | reproduce: `measure_curve` above |

---

## 11. When to reach for this (and when not to)

A decisive recommendation, because "it depends" is not advice.

**Serve a hybrid when your workload is long-context and memory-bound.** RAG with 16k-plus retrieved context. Agentic loops whose conversation grows for dozens of turns — the vLLM team's [Mooncake Store post](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06) reports agentic traces with a 131:1 input-to-output ratio, a median of 33 turns and roughly 80,000 tokens of median context by turn 30, which is squarely in the region where the ratio table says a hybrid is at 7×-plus. Document processing. Long-form summarization. In all of these the dense model's cache is what caps your batch size, and removing it is the single largest lever available.

**Do not reach for a hybrid when your contexts are short.** Below the crossover — a few hundred tokens — the hybrid is *worse* on memory, and you have taken on a pile of engine complexity for a regression. Classification endpoints, short-form chat, embedding-style workloads: stay dense.

**Do not reach for one when your engine cannot yet serve it well.** This is the recommendation people skip and regret. Check today, against the docs and not against this post, whether your engine supports: the specific recurrent variant in your checkpoint (Mamba-2 and gated DeltaNet have different support timelines and vLLM's own posts flag gaps in both), prefix caching for hybrids, speculative decoding with hybrids, and whatever tensor-parallel degree you need. If prefix caching is your biggest lever — high-prefix-overlap traffic such as a shared system prompt across thousands of requests — a dense model with excellent prefix caching may well beat a hybrid whose prefix caching is a roadmap item. That is a real trade and the memory arithmetic alone will mislead you about it.

**And do not write this part of the engine yourself unless you are learning.** The whole point of `nanoserve` is to understand the machinery, and the shape model in section 7 is genuinely worth having in your toolbox — it is a hundred lines and it answers capacity questions before you provision anything. But the two-allocator cache manager, the dual descriptor views, the state snapshot path for speculative decoding, the chunk-aligned prefill scheduler: that is a serious body of work that vLLM and its peers have been building for over a year and are still labeling as partially complete. Build the shape model. Read the engine.

**The one thing worth internalizing even if you never serve a hybrid**: the KV cache was never a law of nature. It is a consequence of one architectural choice, and once you have seen a model where 85% of layers do not have one, you stop treating "the cache" as a fixed cost and start treating it as a design variable — which is the frame [the capstone playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) uses to tie the whole series together.

---

## Key takeaways

1. **A hybrid's per-request memory is a line with an intercept, not a line through the origin.** Slope is $2 f L H_{kv} d b$ bytes per token; intercept is $(1-f) L \sigma$ bytes, constant in $S$. Every engine consequence follows from the intercept.
2. **The crossover length is $S^{*} = \sigma / (2 H_{kv} d b)$ — independent of the interleave ratio.** For Nemotron-H-8B that is 527 tokens against its own dense counterfactual, 452 against Llama-3.1-8B. The ratio sets the slope, not the break-even.
3. **The memory ratio saturates early.** Nemotron-H-8B versus Llama-3.1-8B reaches 7.3× at 32k and only 7.8× at 128k, against a theoretical ceiling of 8× set by the layer-count ratio. Long context makes the absolute saving bigger, not the ratio.
4. **The real product is concurrency, not per-token speed.** Derived on one H100 at 128k: 3 concurrent requests dense versus 27 hybrid. Reported throughput gains for hybrids are batch-size gains wearing a costume.
5. **Below the crossover a hybrid is a regression.** At 256 tokens it holds 67% more memory per request. Say this out loud before adopting one for a short-prompt workload.
6. **Interleaving exists because a fixed state cannot do exact recall.** Copying and multi-query associative recall are the documented failure modes, and they are what production traffic is made of. The ratio is the price of recall in bytes.
7. **Never assume an interleave ratio.** Published ones span 1:3 to 1:9 and reflect different bets about workload and variant. Read the model card.
8. **Five engine mechanisms break or change**: preemption (state is not recomputable from a suffix), prefix sharing (state has no shareable pieces), admission (a per-request constant, exhaustible independently of blocks), speculative rollback (needs a snapshot, not a pointer rewind), and chunked prefill (must respect the scan's chunk size).
9. **State slots exhaust separately from KV blocks.** A burst of tiny requests can fail admission with 95% of the KV pool free. Check both pools.
10. **Verify the curve, do not trust it.** Two prefills at widely separated lengths recover both slope and intercept; compare against the config-derived values before you plan any capacity on them.

---

## Further reading

- [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Dao and Gu, arXiv 2405.21060)](https://arxiv.org/abs/2405.21060) — the Mamba-2 / SSD formulation that makes the parallel-scan and recurrence duality practical.
- [*Gated Delta Networks: Improving Mamba2 with Delta Rule* (arXiv 2412.06464)](https://arxiv.org/abs/2412.06464) and [*Gated Linear Attention Transformers with Hardware-Efficient Training* (arXiv 2312.06635)](https://arxiv.org/abs/2312.06635) — the two update rules behind most current linear layers.
- [*Nemotron-H* technical report (arXiv 2504.03624)](https://arxiv.org/abs/2504.03624) and [*Jamba* (arXiv 2403.19887)](https://arxiv.org/abs/2403.19887) — the two clearest published accounts of interleave ratio and placement.
- [*Kimi Linear* (arXiv 2510.26692)](https://arxiv.org/abs/2510.26692) and [*Falcon-H1* (arXiv 2507.22448)](https://arxiv.org/abs/2507.22448) — a finer-gated delta rule, and the parallel rather than layer-wise hybrid.
- [*Repeat After Me* (arXiv 2402.01032)](https://arxiv.org/abs/2402.01032) and [*Zoology* (arXiv 2312.04927)](https://arxiv.org/abs/2312.04927) — the recall and copying limits that justify keeping any attention at all.
- vLLM on hybrids: [Qwen3-Next support](https://vllm.ai/blog/2025-09-11-qwen3-next) (the equal-physical-memory block-size trick) and [disaggregated serving for hybrid SSM models](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (dual descriptor views, and a candid list of what is still unsupported).
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), [MLA and attention variants at inference time](/blog/machine-learning/inference-engineering/mla-and-attention-variants-at-inference-time), and the architecture-side companion [Nemotron-H: how NVIDIA swaps most attention for Mamba-2](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer).

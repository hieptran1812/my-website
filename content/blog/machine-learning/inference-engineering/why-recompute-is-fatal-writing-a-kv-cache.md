---
title: "Why recompute is fatal: writing a KV cache"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Turn the quadratic decode loop into a linear one by writing the single data structure that makes autoregressive generation viable, then confront the memory bill it hands you."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "attention",
    "pytorch",
    "gpu",
    "latency",
    "throughput",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 53
---

The decode loop you wrote in [the previous post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) is correct and unusable. It generates 200 tokens after a 1,000-token prompt by pushing 219,900 token positions through a 32-layer transformer, when 1,199 would do. That is a factor of 183 in arithmetic, and it is not caused by a bad kernel, a bad dtype, or a bad GPU. It is caused by a missing array.

Here is the part that should make you uncomfortable. Almost all of that recomputed work is not merely redundant in the loose sense of "we could have been smarter." It is redundant in the strict sense: the model computes the *same numbers*, bit for bit, on every single step. Step 5 computes the key vector for prompt token 37. So does step 6. So does step 199. Nothing in the architecture makes them differ, and you can prove it in three lines. An engine that recomputes them is paying, over and over, for a value it already had in a register five milliseconds ago.

![A decode step branching into a large recomputed key and value region and one genuinely new column that merge into a single attention read](/imgs/blogs/why-recompute-is-fatal-writing-a-kv-cache-1.webp)

This post writes `nanoserve/cache.py` and rewires the forward pass from [the hand-built model](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch) to use it. By the end you will have a contiguous per-layer KV cache with the shape `[num_layers][2, batch, num_kv_heads, max_seq, head_dim]`, a prefill path and a decode path that are visibly different code, a parity test against the naive loop, and a derivation of exactly how much wall clock this buys you — which is emphatically *not* 183×, and understanding why is more valuable than the speedup itself.

Then the bad news, which is the real subject of Track B. You have not deleted the work; you have converted it into memory. The cache is a growing, per-request, gigabyte-scale object that your allocator did not know about ten minutes ago, and the naive way to hold it wastes roughly nine tenths of the space it claims. That is the problem [the memory math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) quantifies and the paging post solves.

The usual promise, restated from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic shown in the text, cited from a vendor spec or a public post with a link, or framed as a range you should expect when you run the script yourself. The results table carries a `Source` column for exactly that reason.

---

## 1. What a decode step actually computes, and which parts never change

Before writing a cache it is worth being precise about what is being cached, because "cache the attention" is the kind of vague statement that produces subtly wrong engines.

Take one transformer layer $\ell$ and a sequence of $t+1$ tokens. The layer receives a residual stream $h^{(\ell)} \in \mathbb{R}^{(t+1) \times d}$ and produces, per position $i$:

$$
q_i = W_Q\, \text{norm}(h^{(\ell)}_i), \qquad
k_i = W_K\, \text{norm}(h^{(\ell)}_i), \qquad
v_i = W_V\, \text{norm}(h^{(\ell)}_i),
$$

followed by rotary position embedding applied to $q_i$ and $k_i$, then

$$
o_i = \sum_{j \le i} \alpha_{ij}\, v_j, \qquad
\alpha_{ij} = \mathrm{softmax}_j\!\left( \frac{q_i \cdot k_j}{\sqrt{d_h}} \right).
$$

Three tensors per position: a query, a key, a value. The output at position $i$ reads *every* key and value at positions $j \le i$, but only its own query. That asymmetry is the whole game.

### The claim, and its proof

**Claim.** In a causal decoder, $k_j$ and $v_j$ at every layer are functions of tokens $0 \ldots j$ only. Appending a token at position $t+1$ cannot change $k_j$ or $v_j$ for any $j \le t$.

**Proof, by induction on layers.** At layer 0 the residual stream is the embedding lookup, and $h^{(0)}_j = E[x_j]$ depends only on token $x_j$. Suppose $h^{(\ell)}_j$ depends only on tokens $0 \ldots j$. Then $k_j^{(\ell)}$ and $v_j^{(\ell)}$, being pointwise linear maps of $\text{norm}(h^{(\ell)}_j)$, depend only on tokens $0 \ldots j$. The attention output $o_j^{(\ell)}$ sums over $i \le j$ because the causal mask sets every score at $i > j$ to $-\infty$, so it too depends only on tokens $0 \ldots j$. The MLP is pointwise. Therefore $h^{(\ell+1)}_j$ depends only on tokens $0 \ldots j$, which closes the induction. $\square$

That proof is short, and it is the entire justification for the KV cache. It is also worth reading for the assumptions it quietly uses, because every one of them is a place where a real architecture can break the cache:

- **The mask is strictly causal.** If a layer can see the future — a bidirectional encoder, a prefix-LM with a bidirectional prompt region — then earlier positions' activations *do* change when you append, and their K/V are not cacheable across the boundary.
- **Position encoding is a function of absolute index only.** RoPE rotates $q_i$ and $k_i$ by an angle that depends on $i$. That is fine: index 37 stays index 37 forever. It stops being fine the moment you shift positions, which is what StreamingLLM-style attention sinks and some long-context extrapolation tricks do. Then a cached key carries the wrong rotation for its new index and must be re-rotated or recomputed.
- **The layer is stateless across steps.** True for attention. Not true for a Mamba-style recurrent layer, whose state is updated in place and cannot be reconstructed from a slice — which is exactly why hybrid models need a second, differently shaped cache.
- **Nothing between steps mutates the weights.** LoRA adapter swaps mid-generation, quantization state changes, or a fused kernel with a different accumulation order all invalidate the "bit-identical" part of the claim even when they preserve the "semantically identical" part.

None of these apply to Llama-3.1-8B running a straight chat completion, which is why the cache works. All of them come back later in the series.

### What is not cacheable

The symmetric fact matters just as much. Three things are recomputed every step and cannot be avoided:

1. **The query.** $q_{t+1}$ depends on the token you just appended. There is nothing to reuse.
2. **The MLP activations.** Every layer's SwiGLU runs on the new position only. This is most of the FLOPs of a decode step and it is irreducible.
3. **The logits.** One row, at the last position. (The naive loop computed a full $T \times 128256$ matrix and used one row of it; that waste is separate, and deleting it does not require a cache — it was the free optimization in the previous post.)

So the cache does not make decode free. It makes decode's cost *independent of context length* for everything except the attention read itself. A decode step with a 200-token prefix and a decode step with a 100,000-token prefix run the same matmuls against the same weights; only the attention gather grows. Hold onto that, because it is the fact that makes long context viable at all and the fact that makes long context expensive in memory rather than in FLOPs.

<figure class="blog-anim">
<svg viewBox="0 0 700 280" role="img" aria-label="Two panels run the same four decode steps: the left one rereads a growing prefix while the right one appends a single column each step" style="width:100%;height:auto;max-width:860px">
<style>
.k1-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.k1-hot{fill:var(--accent,#6366f1);opacity:.85}
.k1-band{fill:var(--accent,#6366f1);opacity:.20;transform-box:fill-box;transform-origin:left center}
.k1-hdr{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.k1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.k1-cnt{font:600 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.k1-foot{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.k1-rule{stroke:var(--border,#d1d5db);stroke-width:1}
@keyframes k1-grow{0%,24%{transform:scaleX(.595)}25%,49%{transform:scaleX(.696)}50%,74%{transform:scaleX(.797)}75%,100%{transform:scaleX(.899)}}
@keyframes k1-p1{0%,24%{opacity:1}25%,100%{opacity:0}}
@keyframes k1-p2{0%,24%{opacity:0}25%,49%{opacity:1}50%,100%{opacity:0}}
@keyframes k1-p3{0%,49%{opacity:0}50%,74%{opacity:1}75%,100%{opacity:0}}
@keyframes k1-p4{0%,74%{opacity:0}75%,100%{opacity:1}}
@keyframes k1-g1{0%,24%{opacity:0}25%,100%{opacity:1}}
@keyframes k1-g2{0%,49%{opacity:0}50%,100%{opacity:1}}
@keyframes k1-g3{0%,74%{opacity:0}75%,100%{opacity:1}}
.k1-band{animation:k1-grow 10s steps(1,end) infinite}
.k1-a1{animation:k1-p1 10s steps(1,end) infinite}
.k1-a2{animation:k1-p2 10s steps(1,end) infinite}
.k1-a3{animation:k1-p3 10s steps(1,end) infinite}
.k1-a4{animation:k1-p4 10s steps(1,end) infinite}
.k1-s1{animation:k1-g1 10s steps(1,end) infinite}
.k1-s2{animation:k1-g2 10s steps(1,end) infinite}
.k1-s3{animation:k1-g3 10s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.k1-band{animation:none;transform:scaleX(.899)}.k1-a1,.k1-a2,.k1-a3{animation:none;opacity:0}.k1-a4{animation:none;opacity:1}.k1-s1,.k1-s2,.k1-s3{animation:none;opacity:1}}
</style>
<text class="k1-hdr" x="178" y="30">No cache</text>
<text class="k1-sub" x="178" y="50">every step rereads the whole prefix</text>
<text class="k1-hdr" x="514" y="30">With a KV cache</text>
<text class="k1-sub" x="514" y="50">every step writes one new column</text>
<line class="k1-rule" x1="350" y1="20" x2="350" y2="230"/>
<rect class="k1-band" x="20" y="76" width="316" height="46" rx="8"/>
<rect class="k1-cell" x="20"  y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="52"  y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="84"  y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="116" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="148" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="180" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="212" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="244" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="276" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="308" y="80" width="28" height="38" rx="5"/>
<text class="k1-sub" x="178" y="142">shaded = recomputed this step</text>
<text class="k1-cnt k1-a1" x="178" y="176">step 1 reads 6 columns</text>
<text class="k1-cnt k1-a2" x="178" y="176">step 2 reads 7 columns</text>
<text class="k1-cnt k1-a3" x="178" y="176">step 3 reads 8 columns</text>
<text class="k1-cnt k1-a4" x="178" y="176">step 4 reads 9 columns</text>
<text class="k1-sub" x="178" y="202">work per step keeps climbing</text>
<rect class="k1-cell" x="356" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="388" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="420" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="452" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="484" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell" x="516" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell k1-s1" x="548" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell k1-s2" x="580" y="80" width="28" height="38" rx="5"/>
<rect class="k1-cell k1-s3" x="612" y="80" width="28" height="38" rx="5"/>
<rect class="k1-hot k1-a1" x="548" y="80" width="28" height="38" rx="5"/>
<rect class="k1-hot k1-a2" x="580" y="80" width="28" height="38" rx="5"/>
<rect class="k1-hot k1-a3" x="612" y="80" width="28" height="38" rx="5"/>
<rect class="k1-hot k1-a4" x="644" y="80" width="28" height="38" rx="5"/>
<text class="k1-sub" x="514" y="142">solid = computed this step</text>
<text class="k1-cnt" x="514" y="176">every step writes 1 column</text>
<text class="k1-sub" x="514" y="202">work per step never changes</text>
<line class="k1-rule" x1="20" y1="230" x2="672" y2="230"/>
<text class="k1-foot" x="346" y="256">6-token prompt, 4 decode steps: 30 columns processed without a cache, 10 with one</text>
</svg>
<figcaption>The same four decode steps on both sides: the cacheless panel's shaded work region grows every step while the cached panel adds exactly one solid column, which is the difference between quadratic and linear total work.</figcaption>
</figure>

Ten cells and four steps is a toy. The point is the *shape* of the two curves: the left panel's shaded region has area that grows like $n^2/2$, the right panel's solid cells grow like $n$. At six prompt tokens and four steps that is 30 versus 10, a factor of 3. At a thousand prompt tokens and two hundred steps it is 183. At eight thousand and two thousand it is a thousand. The ratio has no ceiling.

---

## 2. The bill, in three different units

"183× more work" is true and slightly misleading, because *work* is ambiguous. Engineers who report a 183× algorithmic saving and then measure a 7× speedup usually conclude their measurement is broken. It is not. The three units disagree, and the disagreement is informative.

### Unit one: token positions

Let $p$ be the prompt length and $n$ the number of generated tokens. The naive loop makes $n$ forward passes, the $i$-th over $p + i$ positions:

$$
W_{\text{naive}}(p, n) = \sum_{i=0}^{n-1} (p+i) = n\,p + \frac{n(n-1)}{2},
\qquad
W_{\text{cached}}(p, n) = p + (n - 1).
$$

For $p = 1000$, $n = 200$: 219,900 against 1,199, a ratio of **183**. *(Source: derived.)*

### Unit two: model FLOPs

Positions convert to floating-point operations through the standard dense-forward estimate of ${2N}$ FLOPs per token, where $N$ is the parameter count. For Llama-3.1-8B, $N = 8.03 \times 10^{9}$, so one token position costs about 16.06 GFLOP.

$$
F_{\text{naive}} = 219{,}900 \times 16.06\,\text{GFLOP} \approx 3{,}532\ \text{TFLOP},
$$
$$
F_{\text{cached}} = 1{,}199 \times 16.06\,\text{GFLOP} \approx 19.3\ \text{TFLOP}.
$$

Same ratio, 183, because FLOPs are linear in positions for the dense part of the model. The attention score term is quadratic in sequence length rather than linear and makes the true ratio worse still; the previous post derives that term and lands at roughly 345× for the attention component alone at this shape. *(Source: derived.)*

### Unit three: seconds

This is where the story changes, and it changes because of the fact established in [the naive-loop post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) and in the [roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound): a batch-1 decode step is **memory-bound**, not compute-bound. Its floor is set by weight bytes over HBM bandwidth, not by FLOPs over peak throughput.

Take the A100 80GB SXM. [NVIDIA's A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) lists 2,039 GB/s of HBM2e bandwidth and 312 TFLOP/s of dense BF16 tensor throughput. Two floors per step:

$$
t_{\text{mem}} = \frac{16.06\ \text{GB}}{2.039\ \text{TB/s}} = 7.88\ \text{ms},
\qquad
t_{\text{compute}}(T) = \frac{T \times 16.06\ \text{GFLOP}}{312\ \text{TFLOP/s}} = T \times 0.0515\ \text{ms}.
$$

The two are equal at $T \approx 153$. **Any prefix longer than about 153 tokens makes a cacheless decode step compute-bound on an A100.** That is a striking inversion: the naive loop is not slow in the way decode is normally slow. It is slow in the way prefill is slow, on every single step, forever.

Now total the two loops for $p = 1000$, $n = 200$:

| Phase | Naive loop | Cached loop | Source |
| --- | --- | --- | --- |
| Prefill | none (folded into step 0) | 1,000 positions, compute-bound: 51.5 ms | derived |
| Decode | 200 steps, compute-bound, 3,532 TFLOP total: 11.32 s | 199 steps at the 7.88 ms memory floor: 1.57 s | derived |
| Weight traffic | 200 × 16.06 GB = 3.21 TB | 200 × 16.06 GB = 3.21 TB | derived |
| **Total floor** | **11.32 s** | **1.62 s** | derived |
| Ratio | — | **7.0×** | derived |

**183× less arithmetic buys 7× less wall clock.** *(Source: derived from the A100 datasheet figures above.)*

The reason is worth stating plainly because it is the single most useful mental correction in this post: the cached loop cannot spend the FLOPs it saved. It is bandwidth-bound, dragging all 16.06 GB of weights across HBM once per token no matter how little arithmetic it does. Deleting 99.5% of the FLOPs moves you from "compute-bound and terrible" to "bandwidth-bound and merely bad." Getting past *bandwidth-bound and merely bad* requires batching, which is Track C, and quantization, which is Track F. The cache is necessary and nowhere near sufficient.

Notice also the row that did not change: weight traffic is identical in both columns. Both loops read the whole model once per generated token. All the cache did was stop the *sequence-proportional* work; the *model-proportional* work is untouched. That is why a bigger prompt makes the naive loop dramatically worse and makes the cached loop barely worse at all.

#### Worked example: why the ratio moves with the workload

The 7× is specific to a shape. Redo the arithmetic for three shapes from the series prompt suite, all on the A100 floors above:

| Workload | $p$ / $n$ | Naive floor | Cached floor | Wall-clock ratio | Position ratio | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Chat | 128 / 512 | 196,352 pos → 10.11 s | 639 pos → 4.03 s | 2.5× | 307× | derived |
| RAG | 4,000 / 200 | 819,900 pos → 42.2 s | 4,199 pos → 1.77 s | 23.8× | 195× | derived |
| Agent turn | 8,000 / 2,000 | 17,999,000 pos → 926.5 s | 9,999 pos → 16.2 s | 57.3× | 1,800× | derived |

The chat case is the least impressive in wall clock, and it is instructive: with a 128-token prompt the naive loop stays memory-bound for its first 25 steps (the crossover is at a prefix of about 153 tokens), and the cached loop pays the same 7.88 ms floor per step, so the difference is only the sequence-proportional work the naive loop accumulates later. The RAG and agent cases are catastrophic without a cache and completely ordinary with one. **The longer the prompt, the more the cache is doing for you**, which is the opposite of the intuition most people carry about long contexts being where caches hurt.

Working the chat row end to end so you can check it: naive positions $512 \times 128 + \frac{512 \times 511}{2} = 65{,}536 + 130{,}816 = 196{,}352$, times 16.06 GFLOP each is 3,153 TFLOP, which at 312 TFLOP/s is 10.11 s; the memory floor over 512 steps is only $512 \times 7.88\ \text{ms} = 4.03$ s, so compute dominates. Cached: prefill $128 \times 0.0515\ \text{ms} = 6.6$ ms, then $511 \times 7.88\ \text{ms} = 4.03$ s. Ratio 10.11 / 4.03 = 2.5. Every other row in the table is the same three steps with different inputs — recompute one yourself before trusting the rest.

---

## 3. Designing the cache: the shape everything downstream inherits

The data structure is simple. The choices baked into it are not, because every later post in Track B and Track C is constrained by them.

The shape:

```python
# per layer, a single tensor:
#   [2, batch, num_kv_heads, max_seq, head_dim]
#    ^  K and V stacked on the leading axis
```

and `nanoserve` holds a list of `num_layers` of them. For Llama-3.1-8B that is `[2, B, 8, max_seq, 128]` per layer, 32 of them.

![A two row grid of key and value slots where the two middle cells are the ones a decode step writes and the trailing cells are reserved but unwritten](/imgs/blogs/why-recompute-is-fatal-writing-a-kv-cache-2.webp)

Six decisions are visible in that shape, and each is a fork you can take differently.

**Why `num_kv_heads` and not `num_attention_heads`.** Llama-3.1-8B has 32 query heads and 8 key/value heads — grouped-query attention, from [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245). The `repeat_kv` step in the forward pass expands 8 KV heads to 32 to match the queries. **You must cache the un-repeated version.** Caching post-repeat K/V would quadruple the memory for zero benefit, since the repeat is a pure broadcast. This is a real bug people ship, and it presents as "our cache is 4× bigger than the formula says." Store 8, repeat at read time, or better, pass `enable_gqa=True` to `scaled_dot_product_attention` and let the kernel do the indexing.

**Why K and V share a tensor.** They have identical shapes, they are written together and read together, and stacking them on a leading axis means one allocation, one pointer per layer, and one contiguous region per layer for a later memcpy or offload. The cost is that `cache[0]` and `cache[1]` are strided views half a tensor apart, which is fine for PyTorch and matters when you write a CUDA kernel. Production engines split this differently — some interleave K and V per block for locality, some keep separate tensors so the K path and V path can carry different quantization scales. That last one is a real reason to split: FP8 KV quantization often wants per-channel scales on K and per-token scales on V.

**Why per-layer tensors and not one giant tensor.** One `[L, 2, B, H, S, D]` tensor is a single allocation and looks tidier. It also forces all layers to be adjacent in memory, which is exactly wrong for the layer-by-layer transfer pattern that KV offloading and prefill/decode disaggregation use — you want to be able to ship layer $\ell$'s cache the moment layer $\ell$ is done, without waiting for layer 31. A list of per-layer tensors gives you that for free. It also lets a hybrid model give different layers different shapes, which matters the moment you serve a sliding-window or SSM architecture.

**Why `max_seq` up front.** Because a contiguous tensor cannot grow without reallocating and copying, and reallocating a multi-gigabyte tensor mid-generation is a stall you cannot afford. So you reserve for the worst case. This is the decision that costs you the most, and section 6 puts a number on it.

**Why store K after RoPE.** The rotation depends on absolute position, and position 37 stays position 37. Storing rotated keys means the decode read is a plain dot product with no per-step rotation of the entire cache — which would defeat the purpose. The exception is position-shifting schemes; if you ever implement StreamingLLM-style attention sinks, you must store pre-RoPE keys or re-rotate on eviction. Note this decision in a comment, because six months later it looks arbitrary.

**Why the sequence axis is second-to-last.** `[..., max_seq, head_dim]` puts `head_dim` innermost, so a single token's 128-dimensional key vector is contiguous. That makes the append a contiguous write and makes the attention gather read 256-byte runs. The alternative, `[..., head_dim, max_seq]`, makes the *time* axis contiguous, which is better for a kernel that streams one channel across many positions and worse for everything else. Every mainstream engine puts `head_dim` innermost. Do the same until you have a kernel that proves otherwise.

### The bytes, derived once

The per-token cost follows directly from the shape:

$$
B_{\text{token}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot s
$$

where the leading 2 is for K and V, $L$ is layers, $H_{kv}$ is KV heads, $d_h$ is head dimension, and $s$ is bytes per element. For Llama-3.1-8B in bf16:

$$
B_{\text{token}} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KiB}.
$$

Memorize that number for this model. It is the conversion rate between context and VRAM, and it is the axis on which every capacity decision in this series turns. [The memory math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) works it out across the whole model matrix; here we need only the one figure.

---

## 4. Implementing it in `nanoserve`

Now the code. Three files change: a new `nanoserve/cache.py`, a modified attention path in `nanoserve/model.py`, and a new loop in `nanoserve/generate.py`.

### The cache object

```python
# nanoserve/cache.py
from dataclasses import dataclass, field
import torch


@dataclass
class KVCache:
    """A contiguous per-layer KV cache: one [2, B, H_kv, S_max, D] tensor per layer.

    Layout notes that later posts depend on:
      * K and V are stacked on axis 0 so a layer is one allocation.
      * H_kv is the *un-repeated* KV head count (GQA repeat happens at read time).
      * Keys are stored AFTER rotary embedding; positions are absolute and final.
      * head_dim is innermost, so one token's key vector is contiguous.
    """

    num_layers: int
    batch: int
    num_kv_heads: int
    max_seq: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    layers: list[torch.Tensor] = field(default_factory=list)
    seq_len: int = 0                      # positions written so far

    def __post_init__(self):
        shape = (2, self.batch, self.num_kv_heads, self.max_seq, self.head_dim)
        self.layers = [
            torch.empty(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.layers)

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write q_len new columns at [seq_len : seq_len + q_len].

        k, v: [B, H_kv, q_len, D]. Returns views over the *live* prefix,
        each [B, H_kv, seq_len + q_len, D].
        """
        buf = self.layers[layer_idx]
        start, q_len = self.seq_len, k.shape[2]
        end = start + q_len
        if end > self.max_seq:
            raise RuntimeError(
                f"KV cache overflow: {end} > max_seq={self.max_seq}. "
                f"Reject the request or preempt one (see the eviction post)."
            )
        buf[0, :, :, start:end, :] = k
        buf[1, :, :, start:end, :] = v
        return buf[0, :, :, :end, :], buf[1, :, :, :end, :]

    def advance(self, q_len: int):
        """Commit the write. Separate from append() so all layers agree."""
        self.seq_len += q_len

    def reset(self):
        self.seq_len = 0                  # no reallocation, no zeroing
```

Six things in there earn their place.

**`torch.empty`, not `torch.zeros`.** Zeroing 1 GiB costs real time and buys nothing: every slot is written before it is read, because attention only ever reads `[:seq_len]`. If you find yourself needing zeros, you have an out-of-bounds read and zeroing will hide it rather than fix it. Keep `empty` and add an assertion instead.

**`append` returns views, not copies.** `buf[0, :, :, :end, :]` is a strided view over the same storage. No allocation, no copy. This matters: an implementation that returns `torch.cat([past_k, new_k], dim=2)` — which is what a great many tutorial implementations do — allocates and copies the *entire* cache on every decode step. At 1,000 tokens of context that is 128 MB of copy per step per layer set, which can cost more than the attention it was meant to accelerate. If your cached loop is somehow *slower* than your naive loop, this is why.

**`advance` is separate from `append`.** All 32 layers must write at the same `seq_len`. If `append` incremented the counter, layer 1 would write at position $t+1$ and layer 2 at position $t+2$. Separating commit from write makes the invariant explicit and makes the bug impossible. This is the single most common KV-cache bug, and its symptom is beautiful: output that is fluent for a few tokens and then degenerates into repetition, because the attention pattern is progressively skewed.

**`reset` does not free.** Reusing the allocation across requests is the whole reason to pre-allocate. Freeing and reallocating hands the memory back to the caching allocator and invites fragmentation.

**Overflow raises, loudly.** A silent wrap or a silent truncation produces wrong tokens with no error. In a real engine this is not an exception but a scheduler decision — reject, queue, or preempt somebody — which is the eviction post's subject. For now, crash.

**No `torch.compile` or CUDA-graph friendliness yet.** `seq_len` as a Python int means the shapes of the views change every step, which forces recompilation and breaks graph capture. Real engines keep the length on device and use fixed-shape buffers with masking. That is a Track F problem; noting it here so you know the design is deliberately naive in a second way.

### Threading it through the layer

The forward pass from the hand-written model post takes `(h, i, cos, sin, positions, mask)`. It now takes a cache and a position offset:

```python
# nanoserve/model.py  (attention sub-block, modified)
import torch
import torch.nn.functional as F


def attention_block(self, h, i, cache, past_len):
    cfg, w = self.cfg, self.w
    p = f"model.layers.{i}"
    b, q_len, _ = h.shape
    hq, hkv, dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    x = rms_norm(h, w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
    q = F.linear(x, w[f"{p}.self_attn.q_proj.weight"]).view(b, q_len, hq, dh)
    k = F.linear(x, w[f"{p}.self_attn.k_proj.weight"]).view(b, q_len, hkv, dh)
    v = F.linear(x, w[f"{p}.self_attn.v_proj.weight"]).view(b, q_len, hkv, dh)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))       # -> [B, H, q_len, D]

    # Absolute positions continue where the cache left off.
    positions = torch.arange(past_len, past_len + q_len, device=h.device)
    q, k = apply_rope(q, k, self.cos, self.sin, positions)

    if cache is not None:
        k, v = cache.append(i, k, v)                       # -> [B, H_kv, past+q, D]

    k = repeat_kv(k, cfg.n_rep)                            # -> [B, H_q, S, D]
    v = repeat_kv(v, cfg.n_rep)

    a = F.scaled_dot_product_attention(q, k, v, is_causal=(q_len > 1))
    a = a.transpose(1, 2).reshape(b, q_len, hq * dh)
    return h + F.linear(a, w[f"{p}.self_attn.o_proj.weight"])
```

The diff is four lines, and one of them is a trap.

### The `is_causal` trap, which the forward-pass post warned about

Look at `is_causal=(q_len > 1)`.

`torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` builds a causal mask **aligned to the top-left** of the $q_{\text{len}} \times k_{\text{len}}$ score matrix. During prefill, $q_{\text{len}} = k_{\text{len}}$, the matrix is square, and top-left alignment is exactly the causal mask you want. During decode, $q_{\text{len}} = 1$ and $k_{\text{len}} = S$. A top-left-aligned causal mask on a $1 \times S$ matrix allows only column 0. **Your single query would attend to the first token of the prompt and nothing else.**

That failure is silent. No shape error, no NaN, no warning. The model produces grammatical, confident, completely context-free text. It is one of the two or three most expensive hours you can lose in this codebase, and it is why the hand-written forward-pass post flagged it in advance.

The fix at decode is that there is *no mask at all*: a single query at position $S-1$ is allowed to attend to every cached position $0 \ldots S-1$, because they are all in its past by construction. So `is_causal=False`, which is what `q_len > 1` evaluates to.

The general case is chunked prefill, where $1 \lt q_{\text{len}} \lt k_{\text{len}}$ — you are feeding a chunk of a long prompt into a cache that already holds earlier chunks. Now you need a causal mask aligned to the **bottom-right**, and neither `is_causal=True` nor `is_causal=False` gives it to you. Build it explicitly:

```python
# nanoserve/mask.py
import torch


def bottom_right_causal(q_len: int, k_len: int, device, dtype) -> torch.Tensor:
    """Additive mask for q_len queries whose absolute positions are the LAST
    q_len of k_len. Query row r has absolute position (k_len - q_len + r), so it
    may attend to key columns 0 .. (k_len - q_len + r) inclusive.
    """
    q_pos = torch.arange(k_len - q_len, k_len, device=device).unsqueeze(1)  # [q,1]
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)                 # [1,k]
    allowed = k_pos <= q_pos
    return torch.zeros(q_len, k_len, device=device, dtype=dtype).masked_fill(
        ~allowed, float("-inf")
    )
```

Sanity-check it by hand for `q_len=1, k_len=4`: `q_pos = [3]`, so `allowed = [T, T, T, T]` — all four columns, no mask. Correct. For `q_len=4, k_len=4`: `q_pos = [0,1,2,3]`, giving the standard lower triangle. Correct. For `q_len=2, k_len=5`: rows are positions 3 and 4, allowing columns 0–3 and 0–4. Correct.

Keep this function even though `nanoserve` does not need it yet. Chunked prefill is post 13, and having the mask already right will save you the same silent bug a second time.

### The two entry points

The forward pass now splits by intent:

```python
# nanoserve/model.py  (continued)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, cache=None, last_only: bool = True):
        """input_ids: [B, q_len]. With a cache, q_len is the prompt length on the
        prefill call and 1 on every decode call."""
        cfg, w = self.cfg, self.w
        past_len = cache.seq_len if cache is not None else 0

        h = w["model.embed_tokens.weight"][input_ids]
        for i in range(cfg.num_hidden_layers):
            h = self.attention_block(h, i, cache, past_len)
            h = self.mlp_block(h, i)

        if cache is not None:
            cache.advance(input_ids.shape[1])       # commit AFTER every layer wrote

        if last_only:
            h = h[:, -1:, :]                        # [B, 1, d_model]
        h = rms_norm(h, w["model.norm.weight"], cfg.rms_norm_eps)
        head = (w["model.embed_tokens.weight"] if cfg.tie_word_embeddings
                else w["lm_head.weight"])
        return F.linear(h, head)                    # [B, 1, vocab]
```

`cache.advance` is called once, after the layer loop, which is the invariant made structural. `last_only` folds in the free optimization from the previous post: the LM head runs on one position instead of $q_{\text{len}}$, which during a 1,000-token prefill saves about 1.05 TFLOP and 256 MB of allocation.

### The loop

```python
# nanoserve/generate.py
import torch
from nanoserve.cache import KVCache


@torch.inference_mode()
def generate_cached(model, input_ids, max_new_tokens=200, eos_id=None,
                    max_seq=None, cache=None):
    """Prefill once, then decode one token at a time against a KV cache."""
    cfg = model.cfg
    b, p = input_ids.shape
    max_seq = max_seq or (p + max_new_tokens)

    if cache is None:
        cache = KVCache(
            num_layers=cfg.num_hidden_layers,
            batch=b,
            num_kv_heads=cfg.num_key_value_heads,
            max_seq=max_seq,
            head_dim=cfg.head_dim,
            dtype=torch.bfloat16,
            device=input_ids.device,
        )

    # ---- PREFILL: one wide pass over the whole prompt ------------------
    logits = model.forward(input_ids, cache=cache)        # [B, 1, vocab]
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    out = [int(next_id)]
    if eos_id is not None and out[0] == eos_id:
        return out, cache

    # ---- DECODE: one column per step ----------------------------------
    for _ in range(max_new_tokens - 1):
        logits = model.forward(next_id, cache=cache)      # q_len == 1
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token = int(next_id)
        if eos_id is not None and token == eos_id:
            break
        out.append(token)

    return out, cache
```

Compare this to the naive loop side by side and the structural difference is a single line. The naive loop reads `model(ids)` where `ids` is the whole growing sequence. This one reads `model.forward(next_id, cache=cache)` where `next_id` is one token. Everything else is bookkeeping.

Running it:

```python
# scripts/run_cached.py
import torch
from nanoserve.model import Llama
from nanoserve.tokenizer import Tokenizer
from nanoserve.generate import generate_cached

model = Llama.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                              dtype=torch.bfloat16, device="cuda").eval()
tok = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

ids = torch.tensor([tok.apply_chat_template(
    [{"role": "user", "content": "Explain the KV cache in two paragraphs."}],
    add_generation_prompt=True)], device="cuda")

out, cache = generate_cached(model, ids, max_new_tokens=200,
                             eos_id=tok.eos_id, max_seq=8192)
print(tok.decode(out))
print(f"cache: {cache.nbytes / 2**30:.2f} GiB reserved, "
      f"{cache.seq_len} of {cache.max_seq} slots used")
```

```console
The KV cache stores the key and value projections computed for every token ...
cache: 1.00 GiB reserved, 1179 of 8192 slots used
```

That last line is the entire second half of this post in one sentence: **one gigabyte reserved, 14% of it used.**

### The parity test that must exist

The naive loop is now a test fixture, exactly as promised. Greedy decoding is deterministic given deterministic kernels, so two implementations of the same model must emit the same token ids:

```python
# tests/test_cache_parity.py
import torch
from nanoserve.generate import generate_naive, generate_cached


def test_cache_matches_naive(model, tok):
    ids = torch.tensor([tok.encode("The capital of France is")], device="cuda")

    _, naive = generate_naive(model, ids, max_new_tokens=64)
    cached, _ = generate_cached(model, ids, max_new_tokens=64, max_seq=256)

    assert naive == cached, (
        f"diverged at token {next(i for i, (a, b) in enumerate(zip(naive, cached)) if a != b)}"
    )


def test_logit_parity_single_step(model, tok):
    """Stronger: compare logits, not tokens. Catches near-ties that argmax hides."""
    ids = torch.tensor([tok.encode("The capital of France is Paris, a city")],
                       device="cuda")
    ref = model.forward(ids, cache=None)[:, -1, :].float()

    from nanoserve.cache import KVCache
    cache = KVCache(model.cfg.num_hidden_layers, 1, model.cfg.num_key_value_heads,
                    64, model.cfg.head_dim, torch.bfloat16, ids.device)
    model.forward(ids[:, :-1], cache=cache)
    got = model.forward(ids[:, -1:], cache=cache)[:, -1, :].float()

    torch.testing.assert_close(ref, got, rtol=0, atol=2e-2)
```

The second test is the one that finds real bugs. Token-level parity is a coarse detector: two logit vectors can differ by a lot and still argmax to the same id, so a broken cache can pass `test_cache_matches_naive` for the first thirty tokens and then diverge. Comparing logits directly catches the problem on step one.

The tolerance deserves a note. `atol=2e-2` on bf16 logits sounds loose, and it is, deliberately. The prefill path computes attention as a $T \times T$ matmul; the cached path computes it as a $1 \times T$ matmul against a stored K. Those are different reduction shapes, so a good kernel picks different tile sizes and sums in a different order. Floating-point addition is not associative, so the results differ in the last bits and the difference compounds through 32 layers. A tolerance tight enough to fail on that is a tolerance that fails on correct code. The hand-written forward-pass post works through how to choose a tolerance you can defend; the same reasoning applies here.

Where a *bisect* helps: if parity fails, run the comparison layer by layer, feeding the same input to the cached and uncached attention block and diffing the outputs. A cache bug almost always shows a clean signature — the first layer diverges (an append or RoPE-offset bug), or every layer is fine but the final logits differ (a `last_only` slicing bug), or divergence appears only after position 1 (the `is_causal` trap).

---

## 5. Prefill and decode stop being an abstraction and become two code paths

Before the cache, prefill and decode were a conceptual distinction. Now they are two branches with different shapes, different kernels, and different bottlenecks — visible in the code you just wrote.

![A left to right sequence showing an empty cache then a wide prefill pass then decode steps each adding one column while reserved slots stay unused](/imgs/blogs/why-recompute-is-fatal-writing-a-kv-cache-3.webp)

**Prefill.** `model.forward(input_ids, cache=cache)` with `q_len = p`. Every projection is a matrix-matrix product with $M = p$. Attention is a $p \times p$ score matrix. `cache.append` writes $p$ columns in one strided copy. The GPU is doing the kind of work tensor cores exist for, and the step is compute-bound: 1,000 positions at 16.06 GFLOP each is 16.06 TFLOP, which at the A100's 312 TFLOP/s takes 51.5 ms against a 7.88 ms memory floor. Compute wins by 6.5×.

**Decode.** `model.forward(next_id, cache=cache)` with `q_len = 1`. Every projection is a matrix-*vector* product. Attention is a $1 \times S$ score row. `cache.append` writes exactly one column. The tensor cores are almost entirely idle: 16.06 GFLOP at 312 TFLOP/s is 0.051 ms of arithmetic inside a step whose floor is 7.88 ms of memory traffic. Arithmetic occupies **0.65% of the step**. *(Source: derived.)*

That contrast is the reason the rest of this series exists. Two workloads, one model, one process, opposite bottlenecks:

| Property | Prefill | Decode | Source |
| --- | --- | --- | --- |
| Tokens per pass | $p$ (1,000) | 1 | by construction |
| Matmul shape | GEMM, $M = p$ | GEMV, $M = 1$ | by construction |
| Arithmetic intensity | ≈ $p$ FLOP per weight byte | ≈ 1 FLOP per weight byte | derived |
| A100 floor, 8B bf16 | 51.5 ms for 1,000 tokens | 7.88 ms per token | derived |
| Bound by | compute | HBM bandwidth | derived |
| Cache operation | write $p$ columns | append 1 column | by construction |
| What makes it faster | a better GEMM | a bigger batch | derived |
| Latency metric it owns | TTFT | TPOT | definition |

The last two rows are the ones that change how you build a server. Nothing you do to the decode kernel makes decode meaningfully faster at batch 1, because you are bandwidth-bound and the bandwidth is spent on weights you must read regardless. The only lever is amortizing that weight read across more sequences — batching — which is why the next track is a scheduler and not a kernel. Conversely, batching does very little for prefill, which was already using the machine well; prefill's problem is that it *blocks* decode, which is what chunked prefill exists to fix.

#### Worked example: TTFT and TPOT for a RAG request on an A100

Take $p = 4000$, $n = 200$, Llama-3.1-8B in bf16 on an A100 80GB SXM.

Prefill FLOPs: $4000 \times 16.06\ \text{GFLOP} = 64.2$ TFLOP. At 312 TFLOP/s that is a **205 ms floor**. Real hardware does not hit peak; at a realistic 45–60% of peak for a well-shaped bf16 GEMM stack you should expect TTFT somewhere in the **340–460 ms** range before you add tokenization, scheduling, and the first sampling step. *(Source: derived floor; the utilization band is an assumption stated as such — measure yours.)*

Decode: 199 steps at the 7.88 ms floor is **1.57 s** at best, so TPOT ≥ 7.88 ms and per-request throughput ≤ 127 tok/s. In practice a Python decode loop with roughly 300 kernel launches per step and a host synchronization per token lands well below the floor — expect **60–100 tok/s** on this setup, which is 10–16 ms TPOT. *(Source: derived floor plus a reproduce-it-yourself range; run `bench_cache.py` from section 9.)*

End-to-end: $\text{TTFT} + 199 \times \text{TPOT} \approx 0.4 + 199 \times 0.013 \approx 3.0$ s.

Now the same request through the naive loop: 819,900 positions, 13,168 TFLOP, a 42.2-second floor. The user gets their first token at roughly the same time — prefill is prefill — and then waits three quarters of a minute for the rest. Same TTFT, catastrophically different TPOT. **The cache is a TPOT optimization, not a TTFT optimization**, and the two failure modes feel completely different to a user: a slow TTFT reads as "the site is loading", a slow TPOT reads as "the model is stupid and slow".

---

## 6. The cost of the win: you just allocated a second model

Nothing was deleted. The recomputation was converted into storage, and storage on a GPU is the scarcest thing you have.

![A layered budget showing model weights and activations consuming most of a 24 GiB card and the remaining headroom converted into a token budget and then into reserved request slots](/imgs/blogs/why-recompute-is-fatal-writing-a-kv-cache-4.webp)

Run the budget for the consumer baseline, an RTX 4090 with 24 GiB:

| Item | Bytes | Source |
| --- | --- | --- |
| Card capacity | 24 GiB (≈ 23.99 GiB usable) | cited: NVIDIA RTX 4090 specifications |
| Weights, 8.03B params in bf16 | 16.06 GB = 14.96 GiB | derived |
| CUDA context + activations + allocator slack | ≈ 1.6 GiB | estimate, order of magnitude |
| **KV cache headroom** | **≈ 8 GiB** | derived |
| Bytes per token | 128 KiB | derived from the shape |
| **Token budget** | **65,536 tokens, all users combined** | derived |

Sixty-five thousand tokens sounds like a lot until you divide it. One user with a 32k context consumes half the card. Eight users with 8k contexts consume all of it. And that is the *ideal* accounting, where every reserved slot is written.

The contiguous cache does not achieve the ideal accounting. It reserves `max_seq` per request slot at allocation time, before anyone knows how long the request will be:

$$
8{,}192 \text{ tokens} \times 128\ \text{KiB} = 1{,}073{,}741{,}824\ \text{bytes} = 1\ \text{GiB exactly}.
$$

Eight slots and the card is full — with almost nothing in them.

<figure class="blog-anim">
<svg viewBox="0 0 700 250" role="img" aria-label="Three request lanes each reserve a full context window while the written portion of each lane grows and stops far short of the reservation" style="width:100%;height:auto;max-width:840px">
<style>
.k2-res{fill:none;stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:6 5}
.k2-bg{fill:var(--surface,#f3f4f6)}
.k2-fill{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:left center}
.k2-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.k2-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.k2-hdr{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.k2-foot{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes k2-a{0%{transform:scaleX(0)}52%,93%{transform:scaleX(.0854)}100%{transform:scaleX(0)}}
@keyframes k2-b{0%{transform:scaleX(0)}38%,93%{transform:scaleX(.0305)}100%{transform:scaleX(0)}}
@keyframes k2-c{0%{transform:scaleX(0)}72%,93%{transform:scaleX(.2197)}100%{transform:scaleX(0)}}
.k2-fa{animation:k2-a 9s cubic-bezier(.35,0,.35,1) infinite}
.k2-fb{animation:k2-b 9s cubic-bezier(.35,0,.35,1) infinite}
.k2-fc{animation:k2-c 9s cubic-bezier(.35,0,.35,1) infinite}
@media (prefers-reduced-motion:reduce){.k2-fa{animation:none;transform:scaleX(.0854)}.k2-fb{animation:none;transform:scaleX(.0305)}.k2-fc{animation:none;transform:scaleX(.2197)}}
</style>
<text class="k2-hdr" x="400" y="28">Each slot reserves 8,192 token positions the moment the request is admitted</text>
<text class="k2-lbl" x="20" y="72">req A</text>
<rect class="k2-bg" x="90" y="56" width="520" height="24" rx="5"/>
<rect class="k2-fill k2-fa" x="90" y="56" width="520" height="24" rx="5"/>
<rect class="k2-res" x="90" y="56" width="520" height="24" rx="5"/>
<text class="k2-sub" x="622" y="73">700 written</text>
<text class="k2-lbl" x="20" y="122">req B</text>
<rect class="k2-bg" x="90" y="106" width="520" height="24" rx="5"/>
<rect class="k2-fill k2-fb" x="90" y="106" width="520" height="24" rx="5"/>
<rect class="k2-res" x="90" y="106" width="520" height="24" rx="5"/>
<text class="k2-sub" x="622" y="123">250 written</text>
<text class="k2-lbl" x="20" y="172">req C</text>
<rect class="k2-bg" x="90" y="156" width="520" height="24" rx="5"/>
<rect class="k2-fill k2-fc" x="90" y="156" width="520" height="24" rx="5"/>
<rect class="k2-res" x="90" y="156" width="520" height="24" rx="5"/>
<text class="k2-sub" x="622" y="173">1,800 written</text>
<text class="k2-sub" x="90" y="204">dashed outline = memory already charged to the request</text>
<text class="k2-foot" x="350" y="232">2,750 of 24,576 reserved positions ever written: 89 percent of the reservation is dead space</text>
</svg>
<figcaption>Three requests grow into a contiguous cache that reserved a full context window each; the fills stop far short of the dashed reservation, and none of the unused space is available to anyone else until the request ends.</figcaption>
</figure>

#### Worked example: the reservation tax on a 4090

Assume a chat workload where the average request ends up at 700 total tokens (prompt plus completion) but any request is *allowed* to reach 8,192, so you must reserve for 8,192.

- Slots that fit in 8 GiB of headroom: $8\ \text{GiB} / 1\ \text{GiB} = 8$.
- Tokens actually written across those 8 slots: $8 \times 700 = 5{,}600$.
- Tokens reserved: $8 \times 8{,}192 = 65{,}536$.
- Utilization: $5{,}600 / 65{,}536 = 8.5\%$. **About 92% of the cache is memory you paid for and never wrote.** *(Source: derived, given the 700-token average assumption.)*

Now the consequence that hurts. Your card can hold 65,536 tokens. Your workload needs 5,600. You are serving **8 concurrent users on a GPU whose memory could hold 93**. Concurrency is throughput, throughput is dollars, and you just left an order of magnitude on the floor — not to a slow kernel, but to an allocation policy.

Three flavors of waste are stacked here, and it is worth naming them separately because they get fixed by different things:

1. **Reservation waste (internal fragmentation).** Space reserved for a request that the request never uses. Fixed by allocating in small units on demand — paging.
2. **Padding waste.** In a batched contiguous cache, all sequences share one `max_seq` axis, so a batch containing one long sequence forces the short ones to sit in a buffer sized for the long one. Fixed by removing the shared batch axis — paging again.
3. **External fragmentation.** Free memory exists but not in a contiguous run large enough for the next request's slot. Fixed by decoupling logical contiguity from physical contiguity — paging, a third time.

The vLLM/PagedAttention paper, [Kwon et al. (2023)](https://arxiv.org/abs/2309.06180), measured these in real serving systems and reported that only a modest fraction of allocated KV memory in prior engines held actual token state, with the rest lost to reservation and fragmentation. Their fix — treat the cache as fixed-size blocks with a table mapping logical positions to physical blocks, exactly like virtual memory — is the subject of the paged-cache post. The number to carry forward is not the paper's exact percentage but the shape of the claim: **the dominant KV inefficiency in a naive engine is allocation policy, not kernel quality.**

You should also notice the temporal dimension the animation shows and a static figure cannot: the reservation is charged at admission and released only at completion. A request that generates 40 tokens and stops holds its full gigabyte for its entire life. There is no partial release, no shrink-to-fit, and no way for a queued request to use the space. That is the mechanism behind the "we have plenty of free VRAM and yet the queue is full" complaint, and it is why the scheduler and the allocator are the same problem.

---

## 7. Layout choices and what each one costs later

Every layout stores the same 128 KiB per token. They differ entirely in what they waste and how they behave under a batch of unequal lengths.

![A comparison table of four cache layouts scored on append cost and reservation waste and ragged batch behavior and where each one is used](/imgs/blogs/why-recompute-is-fatal-writing-a-kv-cache-5.webp)

**Batched contiguous** — what `nanoserve` has now. One tensor per layer with a batch axis and a shared `max_seq`. Append is a single slice write, which is as cheap as it gets. The price is that the batch axis forces every sequence to be padded to the same length, on top of the reservation waste. It is the right first implementation and the wrong second one.

**Per-request buffers** — one `[2, H_kv, max_seq, D]` tensor per layer per request. Removes padding waste entirely, since each request's buffer is its own. Keeps reservation waste. Attention now needs one call per request, or a ragged/varlen kernel, which is why this layout tends to come with a FlashAttention varlen entry point. Simple servers land here naturally.

**Paged blocks** — the cache is one big pool of fixed-size blocks (vLLM's default is 16 tokens), plus a per-request block table mapping logical block index to physical block id. Reservation waste drops to at most one partial block per request — under 16 tokens, or 2 MiB for this model, instead of a gigabyte. Padding waste disappears. The cost is an indirection on every read: the attention kernel must consult the block table, which is exactly why PagedAttention needed a custom kernel rather than a stock SDPA call.

**Sliding window** — the cache is a ring buffer capped at $w$ positions, and position $t$ overwrites slot $t \bmod w$. Memory becomes constant in context length, which is the only layout here that does. The cost is that tokens outside the window are gone, so the model's effective context is $w$, and any architecture using this must have been *trained* with the window. Gemma and Mistral-family models interleave windowed and full-attention layers precisely to get part of this saving without losing long-range recall, which means a real engine needs per-layer cache shapes, not one shape for the whole model.

### What the production engine does, and where the formula matches

vLLM's [Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) post (2025-09-05) documents the paged layout concretely. The vLLM team describes a default block size of 16 tokens and gives the per-block byte formula as `2 (key/value) * block_size * num_kv_heads * head_size * dtype_bytes`. Free blocks live in a `free_block_queue` pool — the post describes it as holding "hundreds of thousands" of blocks depending on available VRAM — and a `req_to_blocks` map takes a request id to its list of blocks.

Set that formula against ours. Their per-block bytes is exactly our per-token bytes with the layer factor moved out (they count per layer, we summed over layers) and multiplied by the block size. For Llama-3.1-8B, one 16-token block across all 32 layers is

$$
16 \times 128\ \text{KiB} = 2\ \text{MiB},
$$

so an 8 GiB pool holds 4,096 blocks, or 65,536 tokens — the identical token budget we derived, now handed out in 2 MiB units instead of 1 GiB units. **The paging win is not a memory-efficiency trick that finds new bytes. It is the same bytes, allocated at a granularity 512× finer.** That is the whole idea, and having derived both numbers yourself makes the vLLM design read as inevitable rather than clever.

The same post describes the scheduler as waiting and running queues plus a KV manager, and notes that continuous batching flattens all sequences into one concatenated "super sequence" with position and mask isolation. Hold that thought for Track C — it is what replaces our padded batch axis.

One more citation worth keeping in view because it shows how far the layout question travels: vLLM's [KV Cache Offloading to CPU](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) post (2026-01-08) reports that a change to the KV layout (their PR #27743) moved the physical block size for Llama-3.1-8B from 32 KB to 2 MB, and for Llama-3.1-70B from 8 KB to 1.25 MB — because transferring many tiny blocks over PCIe is bandwidth-hostile. The layout you pick for the attention kernel and the layout you want for host transfer are not the same layout, and production engines end up negotiating between them.

---

## 8. Stress test: what breaks when the batch is not one

Everything above assumed batch 1 and a single sequence length. Both assumptions are false in a server, and the ways they fail set up Track C.

![A decision tree from a batch of four requests through equal lengths and padding and per request buffers down to a shared block pool](/imgs/blogs/why-recompute-is-fatal-writing-a-kv-cache-6.webp)

### Ragged lengths in one batch

Our cache has a single `seq_len` for the whole batch. That is only correct if every sequence in the batch is the same length, which happens roughly never. Four requests arriving with prompts of 120, 900, 3,000 and 60 tokens produce a batch whose padded length is 3,000.

The padding tax: total real tokens 4,080, total padded slots $4 \times 3000 = 12{,}000$. **66% of the batch is padding.** *(Source: derived.)* Every one of those padded positions costs a full model forward pass in prefill and, if you are careless with the mask, corrupts the attention of the real positions.

Two sub-problems, and they are different:

**Left-pad or right-pad.** Decode always reads the *last* position of each sequence. If you right-pad, the last position of a short sequence is a pad token, and you must gather the real last index per row — doable but fiddly. If you left-pad, every sequence's last position is real and decode is a clean `[:, -1]`, but now the *absolute positions* differ per row, so RoPE must receive a per-row position vector rather than a scalar offset, and the causal mask must additionally block the leading pad region. Most batched implementations left-pad for exactly this reason and pay for it with position bookkeeping.

**A per-row `seq_len`.** Once lengths differ, `cache.seq_len` must become a tensor of shape `[B]`, `append` must scatter into per-row offsets, and `scaled_dot_product_attention` must receive an explicit mask that is different for each row. At that point you are three-quarters of the way to reimplementing a block table badly. Recognizing this moment — where the contiguous cache stops being simpler than the paged one — is the actual lesson.

### One long request in a batch of short ones

This is the pathology worth internalizing. Batch of 32: 31 chat requests around 600 tokens and one document-analysis request at 100,000 tokens.

With a shared `max_seq` axis you must size the cache for 100,000. Reserved bytes: $32 \times 100{,}000 \times 128\ \text{KiB} = 390$ GiB. That does not fit on any single GPU made. Even with per-request buffers, the one long request alone needs $100{,}000 \times 128\ \text{KiB} = 12.2$ GiB — more than the entire cache headroom of a 4090, and 15% of an A100 80GB.

Three consequences fall out immediately:

1. **Admission control is a memory problem, not a queueing problem.** You cannot admit a request without knowing its context length, because context length is a memory reservation. This is why real schedulers take `max_model_len` and `max_num_seqs` together and why the decision to admit is fundamentally "can I afford this request's cache".
2. **Head-of-line blocking becomes a memory phenomenon.** The long request holds its reservation for its entire life, blocking short requests that would have finished in 300 ms. This is the failure the queueing-collapse case study later in the series dissects.
3. **The padding tax and the reservation tax multiply.** A batch axis sized for the worst-case member, times a `max_seq` sized for the worst-case request, is a product of two worst cases.

Paging fixes all three by making the unit of allocation a 16-token block rather than a request-shaped rectangle. The long request holds 6,250 blocks; the short ones hold 38 each; nobody is padded to anybody.

![A side by side comparison of a contiguous cache reserving a full context window per request against a pool of small blocks handed out on demand](/imgs/blogs/why-recompute-is-fatal-writing-a-kv-cache-7.webp)

That comparison is the shape of the next post in this track, and it is worth stating as a design principle rather than a trick. The contiguous cache couples three things that have no business being coupled: **how long a request might get**, **how much memory it is charged**, and **where in physical memory its tokens live**. Reserving `max_seq` couples the first two. Requiring a contiguous run couples the second and third. Break both couplings — charge per block written, and let logical position $j$ live in any physical block — and every waste category in section 6 collapses at once. The price is one level of indirection on the hottest read in the engine, which is why the paged cache needs its own attention kernel and why that kernel gets its own post in Track E.

### Very long context

At 128k tokens the cache for a single Llama-3.1-8B request is

$$
131{,}072 \times 128\ \text{KiB} = 16\ \text{GiB},
$$

which is *slightly more than the model's own weights*. **At long context the cache is the model.** Three things change character at that scale:

- The attention read stops being negligible. A decode step must read the whole K and V for the sequence: at 128k that is 16 GiB of cache traffic per step on top of 16.06 GB of weight traffic, roughly doubling the memory-bound floor. Decode TPOT becomes context-dependent, which it was not at 1k.
- The prefill is quadratic in the attention term and enormous. This is where chunked prefill stops being an optimization and becomes a requirement, because a single unchunked 128k prefill both takes seconds and spikes activation memory.
- Cache precision starts to matter economically. Storing KV in FP8 rather than bf16 halves that 16 GiB. vLLM's [FP8 KV-cache post](https://vllm.ai/blog/2026-04-22-fp8-kvcache) (2026-04-22) reports, for Llama-3.1-8B on an H100, that the inter-token-latency slope falls to 54% of BF16 with a break-even around 7k tokens, and notes accuracy caveats including a Hopper accumulation issue at very long contraction lengths that they fixed with two-level accumulation. That is a Track F decision; the reason it exists is the arithmetic in this paragraph.

### Batch 1 versus batch 64, and the honest answer about the speedup

One more stress axis. Everything derived above is at batch 1. At batch 64 the weight read is amortized across 64 sequences, so the per-token memory cost of weights drops 64×, and the decode step becomes compute-bound rather than bandwidth-bound. The cache's *relative* value goes up, not down: without a cache, batch 64 means 64 growing prefixes recomputed every step, and you run out of both FLOPs and patience immediately.

But something subtler also happens. At batch 64 the cache read itself becomes a first-class cost: 64 sequences at 2,000 tokens each is 128,000 tokens of cache, 15.6 GiB, read once per decode step. On an A100 at 2,039 GB/s that alone is 8.2 ms — more than the entire weight read. **Past a certain batch-times-context product, your decode step is bound by cache traffic rather than weight traffic**, and the fixes are different (better attention kernels, FP8 KV, fewer KV heads) from the fixes for weight-bound decode (bigger batch, quantized weights). Knowing which regime you are in is what the roofline analysis in Track H is for.

---

## 9. Measuring the speedup without fooling yourself

You now have two implementations that produce identical tokens. Comparing them is the first real experiment in this series, and it is easy to do badly.

```python
# nanoserve/bench_cache.py
import time
import torch
from nanoserve.generate import generate_naive, generate_cached


def timed(fn, *args, warmup=2, iters=5, **kw):
    """Return (mean_ms, p50_ms) using CUDA events, after warmup and a sync."""
    for _ in range(warmup):
        fn(*args, **kw)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kw)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    return sum(samples) / len(samples), samples[len(samples) // 2]


def compare(model, tok, prompt_len=1000, n_new=200):
    ids = torch.randint(1000, 30000, (1, prompt_len), device="cuda")

    naive_ms, _ = timed(generate_naive, model, ids,
                        max_new_tokens=n_new, iters=2)
    cached_ms, _ = timed(generate_cached, model, ids,
                         max_new_tokens=n_new, max_seq=prompt_len + n_new)

    print(f"prompt={prompt_len} new={n_new}")
    print(f"  naive : {naive_ms / 1000:7.2f} s  "
          f"({n_new / (naive_ms / 1000):6.1f} tok/s)")
    print(f"  cached: {cached_ms / 1000:7.2f} s  "
          f"({n_new / (cached_ms / 1000):6.1f} tok/s)")
    print(f"  ratio : {naive_ms / cached_ms:6.2f}x")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
```

The habits baked into `timed` are the ones from [the benchmark-protocol thinking in the baseline post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline), and they are non-negotiable:

- **Warm up.** The first call allocates the cache, autotunes cuBLAS heuristics, JITs kernels, and loads the CUDA context. It can be several times slower than steady state and it is not what you are measuring.
- **`torch.cuda.synchronize()` before starting the clock and before reading it.** CUDA is asynchronous. A wall-clock timer around an unsynchronized launch measures Python, not the GPU.
- **CUDA events, not `time.perf_counter`.** Events are recorded in the stream and measure GPU time between them, which excludes host-side jitter you did not intend to include.
- **Fixed prompt tokens.** `torch.randint` gives a synthetic prompt of exactly the length you asked for; using real text means your prompt length varies with the tokenizer and your comparison drifts.
- **`max_memory_allocated`, reset between configurations.** Peak allocation is the number that determines how many users fit, and it is invisible to a latency timer.
- Two more that are not in the code and should be in your protocol: **lock the clocks** (`nvidia-smi -lgc`) so thermal drift does not become your p99, and **run the two implementations in separate processes** or at least reset the allocator between them, because a 1 GiB cache left resident changes the memory picture for whatever runs next.

### What you should expect to see

| Measurement | Naive loop | Cached loop | Source |
| --- | --- | --- | --- |
| Token positions, $p=1000$, $n=200$ | 219,900 | 1,199 | derived |
| Model FLOPs | 3,532 TFLOP | 19.3 TFLOP | derived |
| A100 end-to-end floor | 11.32 s | 1.62 s | derived from NVIDIA A100 datasheet |
| A100 decode rate, floor | 17.7 tok/s | 127 tok/s | derived |
| A100 decode rate, expect | 8–16 tok/s | 60–100 tok/s | reproduce: `bench_cache.py` |
| RTX 4090 decode rate, expect | 4–9 tok/s | 30–55 tok/s | reproduce: `bench_cache.py` |
| Peak VRAM added by the cache | 0 | +0.15 GiB at 1,199 tokens | derived |
| Peak VRAM added, 8,192 reservation | 0 | +1.00 GiB | derived |
| Wall-clock ratio you should see | — | 5–9× at this shape | derived floor, reproduce to confirm |

The RTX 4090 rows use the card's 1,008 GB/s of memory bandwidth from [NVIDIA's RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/), giving a 15.9 ms per-step floor and a 63 tok/s ceiling for this model; the cached band sits below that ceiling for the same Python-overhead reasons. The *naive* columns are the weaker numbers in this table and you should treat them as order-of-magnitude expectations rather than floors: a cacheless step is compute-bound past a ~153-token prefix, so its rate depends on the card's achieved bf16 tensor throughput, which varies enough by SKU and kernel that deriving it honestly needs the whitepaper for your exact card. Run the script; the ratio is what matters, not the absolute.

**If your measured ratio is far above 9×, be suspicious.** The most common cause is that you benchmarked the naive loop with the full-vocabulary LM head and the cached loop with `last_only=True`, which conflates two independent optimizations. Fix the naive baseline first — it should slice the last position too — and re-measure. The honest comparison isolates the cache.

**If your measured ratio is below 3×, look for a `torch.cat`.** An implementation that concatenates the cache each step instead of writing into a preallocated buffer spends its savings on memcpy. Check `max_memory_allocated`: a cat-based cache shows a peak roughly double the steady-state cache size, because the old and new tensors coexist during the copy.

**If your cached loop is slower at very short prompts, that is real.** At $p = 32$, $n = 16$, the naive loop's total extra work is small and the cache's fixed costs — a 1 GiB allocation, extra Python per layer, the append kernel launches — can dominate. Nobody serves that workload, but it explains why a microbenchmark on a toy prompt can mislead you.

### The metric that actually matters, and it is not this one

One caution to carry into Track C. Everything measured here is single-request latency at batch 1. That number tells you almost nothing about a server. A change that improves batch-1 tok/s by 20% and reduces the number of concurrent requests you can hold from 8 to 4 is a *large regression* in throughput and in cost per million tokens. The cache is such a change in the second dimension: it makes each request faster and each request more expensive in memory. Measuring only latency will make you feel great about a system that serves half as many people.

The metric that captures both is goodput — completed requests per second that met their latency SLO — and building the load generator that measures it honestly is what the experiment protocol post ships. Until then, always report peak VRAM next to tok/s. They are two halves of one number.

---

## Case studies and real numbers

**vLLM's paged block layout.** The [Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) post (2025-09-05) gives the production contrast to `nanoserve`'s contiguous cache: a default block size of 16 tokens, a per-block byte formula of `2 * block_size * num_kv_heads * head_size * dtype_bytes` per layer, a `free_block_queue` pool of free blocks, and a `req_to_blocks` map from request id to block list. Our derivation of 2 MiB per 16-token block and 4,096 blocks in 8 GiB comes straight from applying that formula to Llama-3.1-8B.

**PagedAttention's premise.** [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)](https://arxiv.org/abs/2309.06180) is the paper that named the problem this post's section 6 sets up: in pre-paging serving systems, a large fraction of allocated KV memory held no token state at all, lost to reservation and fragmentation. Their contribution is not a faster kernel but an allocator, which is why the fix generalizes to every engine.

**Prefix caching as the next step past this post.** vLLM's [V1 architecture post](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27) reports that their prefix-caching implementation is effectively free to leave on — the vLLM team describes under 1% throughput decrease even at a 0% cache hit rate, thanks to a cheap hash lookup plus LRU — and it is enabled by default. That result depends entirely on the block structure: you can only share a cache prefix if the cache is made of hashable, alignable units. Our contiguous per-request buffer cannot share anything with anyone. The prefix-sharing post picks this up.

**GQA as the reason any of this fits.** [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245) introduced grouped-query attention. For Llama-3.1-8B the KV head count drops from 32 to 8, and since $B_{\text{token}}$ is linear in $H_{kv}$, the cache drops from 512 KiB to 128 KiB per token. On our 4090 budget that is the difference between 16,384 tokens of headroom and 65,536. **The largest single KV-cache optimization in this series is an architecture choice made before the model was trained**, and no amount of engine work recovers it if the model you are handed uses full multi-head attention.

---

## When to reach for this (and when not to)

**Always write the cache.** There is no serving scenario where the naive loop is the right production choice. The only question is what shape of cache, not whether.

**Keep the contiguous cache when:** you are running batch 1, offline, on a fixed and known context length — a research script, an eval harness, a batch job over documents of similar size, or an embedded single-user deployment. The contiguous cache is a hundred lines you fully understand, and its reservation waste is irrelevant when there is exactly one request and the reservation is right-sized.

**Move to paged blocks when:** you serve more than one request concurrently, or the context length varies by more than about 2× across requests, or you want prefix sharing. The threshold is not throughput — it is *variance in request shape*. A workload of uniformly 4k-token requests wastes little in a contiguous cache; a workload mixing 200-token chats with 60k-token document reads wastes almost everything.

**Do not write your own engine when:** you need continuous batching, prefix caching, chunked prefill, speculative decoding, structured output, and multi-GPU — all at once, in production, next quarter. Use vLLM or SGLang. Everything in this series exists to make you a better *operator* of those systems and a competent debugger of them, not to convince you to ship a hand-rolled engine to paying users. The honest ledger of what `nanoserve` still loses to vLLM is the capstone's job; see [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).

**Do not reach for exotic cache compression before you have paging.** FP8 KV halves your cache; paging recovers roughly an order of magnitude from a naive contiguous layout. Do the allocator first. Quantizing a badly allocated cache is optimizing the wrong term, and it costs you accuracy for the privilege.

---

## Key takeaways

1. **Cache K and V because they are provably invariant.** In a causal decoder, $k_j$ and $v_j$ depend only on tokens $0 \ldots j$, so appending a token cannot change them. Every assumption in that proof — strict causality, absolute positions, stateless layers — is a place a real architecture can break the cache.
2. **The cache turns a quadratic loop into a linear one.** Positions drop from $np + n(n-1)/2$ to $p + n - 1$: 219,900 to 1,199 for a 1,000-token prompt and 200 generated tokens.
3. **183× less arithmetic is about 7× less wall clock**, because the cached decode step is bandwidth-bound and cannot spend the FLOPs it saved. Report the units you actually measured.
4. **The shape is the contract.** `[num_layers][2, batch, num_kv_heads, max_seq, head_dim]`: un-repeated KV heads, K stored post-RoPE, `head_dim` innermost, one tensor per layer.
5. **Never `torch.cat` the cache.** Write into a preallocated buffer and return views. A concatenating cache copies the whole thing every step and can be slower than no cache at all.
6. **`is_causal=True` is wrong at decode.** SDPA aligns its causal mask to the top-left, so a single query would attend only to position 0 — silently. Use no mask at decode, a top-left mask at prefill, and an explicit bottom-right mask for chunked prefill.
7. **Commit the sequence length once, after every layer has written.** Advancing inside the per-layer append skews positions across layers and produces fluent, degenerating text.
8. **Bytes per token is $2 \cdot L \cdot H_{kv} \cdot d_h \cdot s$** — 128 KiB for Llama-3.1-8B in bf16. That constant converts context length into VRAM and governs every capacity decision you will make.
9. **The contiguous cache wastes about 92% of what it reserves** on a realistic chat workload, because it charges a full context window per slot at admission and releases nothing until completion. That is an allocator problem, and paging is the allocator.
10. **Always report peak VRAM next to tok/s.** The cache makes each request faster and each request more expensive; a benchmark that measures only latency will applaud a change that halves your concurrency.

---

## Further reading

- [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)](https://arxiv.org/abs/2309.06180) — the paper that reframed KV memory as an allocator problem.
- [Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)](https://arxiv.org/abs/2305.13245) — why $H_{kv} = 8$ instead of 32, and what it did to cache size.
- [vLLM, *Inside vLLM: Anatomy of a High-Throughput Inference System*](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — block size, the per-block byte formula, the free-block pool, and the request-to-blocks map.
- [vLLM, *vLLM V1: A Major Upgrade to vLLM's Core Architecture*](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — the scheduler that erases the prefill/decode distinction, and near-free prefix caching.
- [PyTorch docs, `scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — backend selection, `is_causal` semantics, and `enable_gqa`.
- [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) — the 2,039 GB/s and 312 TFLOP/s figures every derivation here rests on.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) · [a forward pass by hand](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch) · [the naive decode loop and your first baseline](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) · [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) · [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook)
- Elsewhere on this blog: [the KV cache explained](/blog/machine-learning/large-language-model/kv-cache) · [KV cache optimization for serving](/blog/machine-learning/model-serving/kv-cache-optimization) · [the roofline model: compute-bound versus memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)

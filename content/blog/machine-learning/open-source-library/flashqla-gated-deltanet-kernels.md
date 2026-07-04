---
title: "FlashQLA: Inside Qwen's High-Performance Gated DeltaNet Kernels"
date: "2026-07-03"
publishDate: "2026-07-03"
description: "A deep dive into FlashQLA, Qwen's TileLang kernel library for Gated DeltaNet — the math of the chunked delta rule, the three systems tricks that buy 2–3× over the FLA Triton kernel, the forward/backward API, and production case studies."
tags: ["flashqla", "gated-deltanet", "linear-attention", "tilelang", "cuda-kernels", "qwen", "chunked-prefill", "hopper", "delta-rule", "gpu-optimization"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 43
---

There is a comfortable lie that senior engineers tell themselves about linear attention: because it is $O(n)$ in sequence length instead of $O(n^2)$, its kernels must be easy to make fast. You replace the softmax with a running state, the FLOP count collapses, and the GPU should be happy. Then you actually profile a Gated DeltaNet layer during prefill on an H200, watch the streaming multiprocessors sit two-thirds idle, and discover that your "cheap" linear-attention kernel is *slower per token* than the quadratic FlashAttention kernel it was supposed to beat.

[FlashQLA](https://github.com/QwenLM/FlashQLA) — short for **Qwen Linear Attention** — is the Qwen team's answer to that gap. It is a high-performance kernel library, built on [TileLang](https://github.com/tile-ai/tilelang), that optimizes the forward and backward passes of **Gated DeltaNet (GDN) chunked prefill**. On NVIDIA Hopper and Blackwell it delivers a reported **2–3× forward speedup and ~2× backward speedup over the FLA Triton kernel**, and roughly **2× over FlashInfer 0.6.9** on the same shapes. It powers the linear-attention layers of Qwen3.5 and Qwen3.6, both for pretraining and for edge-side agentic inference.

![Where FlashQLA sits in Qwen3.5: three Gated DeltaNet blocks per full-attention block, each GDN block's prefill driven by FlashQLA](/imgs/blogs/flashqla-gated-deltanet-kernels-1.webp)

The diagram above is the mental model for this entire post. Modern Qwen models are **hybrid**: roughly three of every four transformer blocks are Gated DeltaNet layers (linear attention with an $O(1)$-per-token state), and every fourth block is a full softmax-attention layer that keeps a growing KV cache for global retrieval. The full-attention blocks already have battle-tested kernels — FlashAttention, FlashInfer, FlashMLA. The GDN blocks did **not**, and they are 75% of the network. FlashQLA is the kernel behind every one of those GDN blocks. Get it wrong and three-quarters of your forward pass is leaving performance on the floor.

This is a deep dive in the literal sense. We will build the gated delta rule from its recurrence, derive why the chunked form turns it into matrix multiplies, then walk through the three ideas that make FlashQLA fast — gate-driven intra-card context parallelism, a hardware-friendly algebraic reformulation, and TileLang warp-specialized fused kernels. We will read the actual API, write runnable code against it, and close with seven production case studies and a decision guide.

## Why linear-attention kernels are secretly hard

Let us kill the comfortable lie first, because everything FlashQLA does is a reaction to it.

The appeal of linear attention is real: softmax attention materializes an $L \times L$ score matrix and grows a KV cache linearly with the number of tokens, so both compute and memory blow up on long context. Gated DeltaNet keeps a single fixed-size recurrent state $\mathbf{S}$ of shape $d_k \times d_v$ and updates it token by token. No score matrix, no growing cache. The asymptotics are strictly better. So where does the performance go?

| Assumption | The naive view | The reality on an H200 |
|---|---|---|
| "$O(n)$ FLOPs means fast" | Fewer FLOPs → less time | The chunked delta rule is *matmul-bound*, and its matmuls are small and awkwardly shaped |
| "Linear attention has no big matrices" | Nothing to feed the Tensor Cores | The WY/UT reformulation is deliberately matmul-heavy so it *can* use Tensor Cores |
| "One state vector → trivially parallel" | Embarrassingly parallel over heads | The inter-chunk scan is *sequential*; with small heads it starves the SMs |
| "The gate is just a scalar multiply" | Negligible cost | Gate decay lives on the SFU (exp) and CUDA cores, and can dominate if not overlapped |
| "Backward is symmetric to forward" | Same cost, run it twice | Backward recomputes the scan unless you save the right intermediates; naive bwd is 3–4× fwd |

The core tension is this. To use a modern GPU well you need large, regular matrix multiplies that keep the Tensor Cores saturated. But a *recurrence* is the opposite of that: each step depends on the last, and each step is a rank-1 update — a tiny outer product, not a big GEMM. The entire art of fast linear attention is reshaping that recurrence into big matmuls **without** changing what it computes. The chunked delta rule does exactly that, and it is where all the difficulty — and all the opportunity — lives.

> A linear-attention kernel is not fast because it does less arithmetic. It is fast because it does its arithmetic in a shape the Tensor Cores like, on data that is already in the right level of the memory hierarchy, with the scalar work hidden behind the matmuls.

Keep that sentence in mind. FlashQLA's three techniques each attack one clause of it: the *shape* (algebraic reformulation), the *occupancy* (gate-driven context parallelism), and the *overlap* (warp specialization).

## Gated DeltaNet in 90 seconds

Before we can talk about kernels we need the object they compute. Gated DeltaNet comes from the paper [*Gated Delta Networks: Improving Mamba2 with Delta Rule*](https://arxiv.org/abs/2412.06464), and it is the token-mixer that Qwen3.5 adopted for its linear layers. If you have read our take on [modern LLM architectures](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) or the [Qwen3-Next hybrid design](/blog/paper-reading/large-language-model/qwen3-next-hybrid-attention-ultra-sparse-moe), this is the layer doing the heavy lifting in three of every four blocks.

Start with the memory picture. Softmax attention remembers *everything* — every past key and value sits in the KV cache, and each new query attends over all of them. Gated DeltaNet remembers a *summary*: a single matrix $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ that is an associative memory mapping keys to values. That is the whole difference in one figure.

![Softmax attention grows an L×L score matrix and a KV cache; Gated DeltaNet keeps one fixed d_k×d_v state at every length](/imgs/blogs/flashqla-gated-deltanet-kernels-2.webp)

The left column is softmax attention: the $L \times L$ score matrix, the KV cache that grows with every token, $O(n^2)$ compute and $O(n)$ memory. The right column is Gated DeltaNet: one state $\mathbf{S}$ of shape $d_k \times d_v$, fixed in size no matter how long the sequence gets, $O(n)$ compute and $O(1)$ memory. That fixed-size state is exactly why linear attention is attractive for million-token context — and exactly why its kernels are hard, because updating that state is inherently sequential.

### The recurrence: decay, erase, write

The gated delta rule updates the state with a single, dense equation:

$$\mathbf{S}_t = \mathbf{S}_{t-1}\Big(\alpha_t\big(\mathbf{I} - \beta_t\,\mathbf{k}_t\mathbf{k}_t^\top\big)\Big) + \beta_t\,\mathbf{v}_t\mathbf{k}_t^\top$$

Every symbol earns its place. $\mathbf{S}_{t-1}$ is the previous state. $\alpha_t \in (0,1)$ is a data-dependent scalar **gate** — Mamba2's contribution — that decays the whole memory a little each step. $\beta_t \in (0,1)$ is a data-dependent scalar that controls **how strongly** the delta update writes. $\mathbf{k}_t$ is the key, $\mathbf{v}_t$ the value, $\mathbf{I}$ the identity. Read it as three operations applied in order:

![One gated-delta update: decay the state by alpha, erase the key's old value via I−βkkᵀ, then add the new value βvkᵀ](/imgs/blogs/flashqla-gated-deltanet-kernels-3.webp)

1. **Decay** ($\times\,\alpha_t$): shrink the entire state toward zero. Old associations fade; this is the "gated" part, and it is what lets the model forget stale context on long sequences.
2. **Erase** ($\times\,(\mathbf{I} - \beta_t \mathbf{k}_t\mathbf{k}_t^\top)$): this is the **delta rule**. The factor $\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top$ is (a scaled) Householder-like transform that removes the component of the state along the current key $\mathbf{k}_t$ — it deletes whatever value was previously associated with this key.
3. **Write** ($+\,\beta_t \mathbf{v}_t\mathbf{k}_t^\top$): add the new key→value association back in with strength $\beta_t$.

The intuition that makes DeltaNet beat vanilla linear attention: plain linear attention only ever *adds* $\mathbf{v}_t\mathbf{k}_t^\top$, so it accumulates interference and never corrects a mistaken association. The delta rule *first erases the old value for this key, then writes the new one* — it is an in-place update, like writing to a hash map, not appending to a log. Gated DeltaNet adds Mamba2's decay on top so the memory can also forget globally. Together they give the strong in-context-retrieval and length-extrapolation numbers that made Qwen adopt it.

At inference decode time — one token at a time — you literally run this recurrence, and it is cheap: a couple of rank-1 updates against a small matrix. The problem is **prefill** and **training**, where you have thousands of tokens at once and running them strictly sequentially wastes the entire GPU. That is what the chunked form fixes, and what FlashQLA accelerates.

### Why the gate matters for the kernel, not just the model

One detail that will pay off later. The decay $\alpha_t$ is a *scalar per (head, token)*, and the cumulative decay over a span is a product of these scalars:

$$\gamma_{[t]}^{j} = \prod_{i=tC+1}^{tC+j}\alpha_i$$

Because each $\alpha_i \in (0,1)$, the cumulative product $\gamma$ **shrinks exponentially** with distance. A token 4,000 positions back contributes to the current state with weight $\gamma \approx \prod \alpha \ll 1$. This exponential decay is a modeling choice, but FlashQLA turns it into a *systems* lever: if far-away chunks barely affect the current state, the sequential scan across chunks is *weakly coupled*, which is precisely what lets FlashQLA parallelize it across SMs. Hold that thought for the context-parallelism section.

## From recurrence to chunks

A recurrence is death on a GPU. The fix, shared by DeltaNet, Mamba2, and every serious linear-attention implementation, is **chunkwise parallel scan**: split the length-$L$ sequence into chunks of size $C$ (typically 64–128), compute *within* each chunk in parallel using matrix multiplies, and pass a single state between chunks sequentially. You get the exact same result as the recurrence, but the expensive part becomes GEMMs.

<figure class="blog-anim">
<svg viewBox="0 0 760 260" role="img" aria-label="Chunked delta scan: each chunk computes its outputs as one dense GEMM while the recurrent state is carried from one chunk to the next" style="width:100%;height:auto;max-width:880px">
<style>
.c1-card{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.c1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.c1-sub{font:500 12px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.c1-hi{fill:var(--accent,#6366f1);opacity:.16}
.c1-axis{stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:4 4}
.c1-state{fill:var(--accent,#6366f1)}
.c1-stxt{font:600 12px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
@keyframes c1-sweep{0%{transform:translateX(0);opacity:0}4%,20%{transform:translateX(0);opacity:.18}25%,45%{transform:translateX(180px);opacity:.18}50%,70%{transform:translateX(360px);opacity:.18}75%,94%{transform:translateX(540px);opacity:.18}100%{transform:translateX(540px);opacity:0}}
@keyframes c1-carry{0%{transform:translateX(0);opacity:0}4%,20%{transform:translateX(0);opacity:1}25%,45%{transform:translateX(180px);opacity:1}50%,70%{transform:translateX(360px);opacity:1}75%,94%{transform:translateX(540px);opacity:1}100%{transform:translateX(540px);opacity:0}}
.c1-sweepA{animation:c1-sweep 8s ease-in-out infinite}
.c1-carryA{animation:c1-carry 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.c1-sweepA,.c1-carryA{animation:none}}
</style>
<text class="c1-lbl" x="380" y="28">chunked delta scan (chunk size C = 128)</text>
<line class="c1-axis" x1="40" y1="72" x2="720" y2="72"/>
<rect class="c1-card" x="40"  y="120" width="140" height="96" rx="10"/>
<rect class="c1-card" x="220" y="120" width="140" height="96" rx="10"/>
<rect class="c1-card" x="400" y="120" width="140" height="96" rx="10"/>
<rect class="c1-card" x="580" y="120" width="140" height="96" rx="10"/>
<rect class="c1-hi c1-sweepA" x="40" y="120" width="140" height="96" rx="10"/>
<text class="c1-lbl" x="110" y="162">chunk 0</text>
<text class="c1-sub" x="110" y="186">intra: GEMM</text>
<text class="c1-lbl" x="290" y="162">chunk 1</text>
<text class="c1-sub" x="290" y="186">intra: GEMM</text>
<text class="c1-lbl" x="470" y="162">chunk 2</text>
<text class="c1-sub" x="470" y="186">intra: GEMM</text>
<text class="c1-lbl" x="650" y="162">chunk 3</text>
<text class="c1-sub" x="650" y="186">intra: GEMM</text>
<g class="c1-carryA">
<rect class="c1-state" x="80" y="54" width="60" height="30" rx="15"/>
<text class="c1-stxt" x="110" y="74">state S</text>
</g>
</svg>
<figcaption>Intra-chunk math runs as one dense GEMM (the sweeping highlight); the recurrent state S is carried chunk to chunk along the top (the sequential inter-chunk dependency).</figcaption>
</figure>

The animation captures the two-level structure. **Intra-chunk**, all $C$ tokens are processed together as dense matrix multiplies — this is the part the Tensor Cores love. **Inter-chunk**, one state matrix $\mathbf{S}_{[t]}$ is handed from chunk $t$ to chunk $t+1$; this part is sequential, and it is the bottleneck when there are too few chunks running in parallel to fill the GPU.

### The chunk output, in matrices

Inside a chunk, the output for every position is a sum of two contributions: what the state carried *in* from previous chunks, plus the intra-chunk interactions. Written for the whole chunk at once (capital letters are the stacked per-chunk matrices):

$$\mathbf{O}_{[t]} = \operatorname{Diag}(\gamma_{[t]})\,\mathbf{Q}_{[t]}\mathbf{S}_{[t]}^\top + \big(\mathbf{Q}_{[t]}\mathbf{K}_{[t]}^\top \odot \Gamma_{[t]}\big)\Big(\mathbf{U}_{[t]} - \operatorname{Diag}(\gamma_{[t]})\mathbf{W}_{[t]}\mathbf{S}_{[t]}^\top\Big)$$

The first term, $\operatorname{Diag}(\gamma_{[t]})\mathbf{Q}_{[t]}\mathbf{S}_{[t]}^\top$, is the **inter-chunk** contribution: query against the incoming state, decayed by the cumulative gate $\gamma$. The second term is the **intra-chunk** contribution: a causal, gate-masked $\mathbf{Q}\mathbf{K}^\top$ (the $\odot\,\Gamma$ applies the decay mask) times the value-side quantities $\mathbf{U}$ and $\mathbf{W}$. Every operation here is a matmul or an elementwise mask — exactly the shape a GPU wants. The state itself is propagated between chunks with:

$$\mathbf{S}_{[t+1]} = \gamma_{[t]}^{C}\mathbf{S}_{[t]} + \Big(\mathbf{U}_{[t]} - \operatorname{Diag}(\gamma_{[t]})\mathbf{W}_{[t]}\mathbf{S}_{[t]}^\top\Big)^\top \operatorname{Diag}\!\Big(\tfrac{\gamma_{[t]}^{C}}{\gamma_{[t]}}\Big)\mathbf{K}_{[t]}$$

You do not need to memorize these. The point is structural: **the recurrence became a handful of GEMMs plus diagonal scalings.** The only remaining sequential dependency is the single matrix $\mathbf{S}_{[t]}$ threaded across chunks.

### The UT transform: the trick that makes it matmul-heavy

There is a catch hiding in $\mathbf{U}$ and $\mathbf{W}$. Those quantities come from the *product* of the per-token $(\mathbf{I} - \beta_i\mathbf{k}_i\mathbf{k}_i^\top)$ factors inside a chunk — a product of $C$ rank-1 corrections, each depending on the last. Computed naively that is a serial scan of tiny operations, and it is death on a GPU for the same reason the original recurrence was.

The escape is the **WY representation** from numerical linear algebra (Bischof & Van Loan, 1985) — the same idea that makes blocked Householder QR fast in LAPACK — packaged here as the **UT transform**. The product of $C$ rank-1 factors is rewritten as a single triangular matrix inverse:

$$\mathbf{W}_{[t]} = \mathbf{A}^{W}_{[t]}\operatorname{Diag}(\beta_{[t]})\mathbf{K}_{[t]}, \qquad \mathbf{A}^{W}_{[t]} = \big(\mathbf{I} - \operatorname{lower}(\operatorname{Diag}(\beta_{[t]})\mathbf{K}_{[t]}\mathbf{K}_{[t]}^\top)\big)^{-1}$$

where $\operatorname{lower}(\cdot) = \operatorname{tril}(\cdot, -1)$ is the strictly lower-triangular part. Because $\mathbf{A}^{W}$ is unit lower-triangular, its inverse is a **forward substitution** — a small, dense, $C \times C$ triangular solve that maps beautifully onto Tensor Cores, not a serial scan of GEMVs.

![Naive delta scan runs C dependent rank-1 updates in order; the UT transform replaces them with one triangular solve plus batched Tensor-Core GEMMs](/imgs/blogs/flashqla-gated-deltanet-kernels-4.webp)

That before/after is the whole game. On the left, $C$ rank-1 updates in strict order — serial, tiny GEMVs, Tensor Cores idle. On the right, one triangular solve $\mathbf{T} = (\mathbf{I} - \operatorname{tril}(\mathbf{A}))^{-1}$ followed by batched matmuls — dense, parallel, Tensor-Core-bound. The gate $\alpha$ folds in cleanly because it only ever multiplies things elementwise (via $\operatorname{Diag}(\gamma)$), so it does not disturb the matmul structure. This is why the chunked *gated* delta rule stays hardware-friendly even after adding Mamba-style decay.

FlashQLA does not invent the UT transform — FLA's Triton kernels use it too. What FlashQLA does is *reshape the specific sequence of these matmuls, triangular solves, and gate scalings* so that fewer of them hit the slow paths of a Hopper SM, and so that the ones that remain overlap. That is the next three sections.

## The FlashQLA API

Before the optimizations, the interface — because the shape of the API tells you what the kernel is responsible for. FlashQLA exposes two layers.

### High-level: `chunk_gated_delta_rule`

This is the drop-in you call from a model. It is deliberately shape-compatible with FLA's function of the same name, so migrating an existing Gated DeltaNet layer is mostly an import change.

```python
import torch
from flash_qla import chunk_gated_delta_rule

# B = batch, T = sequence length, H_q/H_v = query/value heads, K/V = head dims
B, T, H_q, H_v, K, V = 1, 8192, 32, 32, 128, 128

q     = torch.randn(B, T, H_q, K, dtype=torch.bfloat16, device="cuda")
k     = torch.randn(B, T, H_q, K, dtype=torch.bfloat16, device="cuda")
v     = torch.randn(B, T, H_v, V, dtype=torch.bfloat16, device="cuda")
g     = torch.randn(B, T, H_v, dtype=torch.float32, device="cuda")  # log-decay (gate)
beta  = torch.rand(B, T, H_v, dtype=torch.float32, device="cuda")   # delta write strength
scale = K ** -0.5

o, final_state = chunk_gated_delta_rule(
    q=q, k=k, v=v,
    g=g,                       # gate: alpha_t = exp(g), typically stored as log-decay
    beta=beta,
    scale=scale,
    initial_state=None,        # optional carried-in state, e.g. from a previous window
    output_final_state=True,   # return S at the end of the sequence for streaming
    cu_seqlens=None,           # optional cumulative seqlens for variable-length packing
)
# o:           [B, T, H_v, V]  — the attention output
# final_state: [B, H_v, K, V]  — the recurrent state after the last token
```

Three details in that signature matter enormously in practice:

- **`g` is the gate**, carried as a log-decay so that `alpha_t = exp(g)` and cumulative decays are sums-then-exp (numerically stable). It has shape `[B, T, H_v]` — one scalar per value head per token.
- **`initial_state` and `output_final_state`** are what make *streaming* prefill possible: process a window, get `final_state`, feed it back as `initial_state` for the next window. This is how you prefill a million-token context without materializing it all at once.
- **`cu_seqlens`** is the variable-length path. Instead of padding a batch to a rectangle, you pack sequences end to end and pass their cumulative lengths, exactly like FlashAttention's varlen API. The kernel resets the state at each sequence boundary.

### Low-level: explicit forward and backward

Under the high-level wrapper are two functions that expose the saved-for-backward tensors directly. You reach for these when you are writing a custom autograd `Function`, fusing GDN into a larger kernel, or debugging gradients.

```python
from flash_qla import chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd

# Forward returns everything the backward pass will need to avoid recompute
g, A, o, h, final_state = chunk_gated_delta_rule_fwd(
    q, k, v, g, beta, scale=scale, initial_state=h0, cu_seqlens=cu_seqlens
)
#   A            — the per-chunk UT transform matrix (I - tril(...))^{-1}
#   h            — the chunk-boundary states from the inter-chunk scan
#   o            — the output
#   final_state  — S after the last token

# Backward consumes A and h (no scan recompute) plus the incoming gradients
dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
    q, k, v, g, beta, A, do, dht=dht, scale=scale, initial_state=h0, cu_seqlens=cu_seqlens
)
#   dq, dk, dv   — grads wrt query, key, value
#   db           — grad wrt beta (delta write strength)
#   dg           — grad wrt the gate
#   dh0          — grad wrt the initial state (for streaming / windowed backprop)
```

The reason the forward returns `A` and `h` explicitly is the single most important performance decision in any recurrent kernel's training path.

![Forward emits A and the chunk states h; backward reuses them with do and dht, skipping a full re-run of the chunked scan](/imgs/blogs/flashqla-gated-deltanet-kernels-5.webp)

A naive backward would re-run the entire chunked scan to reconstruct the intermediate states, then run the gradient scan on top — call it 3–4× the cost of the forward. By **saving the UT matrix `A` and the chunk-boundary states `h`** during forward, FlashQLA's backward reuses them directly: the gradient computation is another set of GEMMs against already-materialized tensors, not a recomputation of the forward. This is the same activation-vs-recompute tradeoff we discuss for [training memory](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload), applied at the kernel level. The cost is a little extra memory to hold `A` and `h`; the payoff is why the backward is ~2× the FLA Triton kernel instead of merely on par.

| API surface | You call it when… | Returns | Backward cost |
|---|---|---|---|
| `chunk_gated_delta_rule` | Standard GDN layer, forward + autograd | `o, final_state` | Managed automatically |
| `chunk_gated_delta_rule_fwd` | Custom autograd, kernel fusion, streaming | `g, A, o, h, final_state` | You own the saved tensors |
| `chunk_gated_delta_rule_bwd` | Paired with `_fwd`; explicit gradient step | `dq, dk, dv, db, dg, dh0` | Reuses `A`, `h` — no scan recompute |

## Technique 1 — Gate-driven intra-card context parallelism

Now the first of the three speed ideas, and the one that is genuinely novel. Recall the problem the chunked scan leaves us with: the state $\mathbf{S}$ is threaded *sequentially* across chunks. If you have a long sequence but few heads — which is exactly what happens under high tensor parallelism, where each GPU owns only a slice of the heads — you can end up with a small number of independent scan streams. On an H200 with 132 SMs, a handful of sequential scans leaves most of the chip idle.

<figure class="blog-anim">
<svg viewBox="0 0 560 290" role="img" aria-label="Intra-card SM occupancy alternates between a serial chunk scan that leaves most streaming multiprocessors idle and gate-driven context parallelism that fills them" style="width:100%;height:auto;max-width:660px">
<style>
.o1-base{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.o1-on{fill:var(--accent,#6366f1)}
.o1-ttl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.o1-cap{font:600 15px ui-sans-serif,system-ui;text-anchor:middle}
.o1-a{fill:var(--text-secondary,#6b7280)}
.o1-b{fill:var(--accent,#6366f1)}
@keyframes o1-fadeA{0%,38%{opacity:1}50%,92%{opacity:0}100%{opacity:1}}
@keyframes o1-fadeB{0%,38%{opacity:0}50%,92%{opacity:1}100%{opacity:0}}
.o1-gA{animation:o1-fadeA 10s ease-in-out infinite}
.o1-gB{animation:o1-fadeB 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.o1-gA{animation:none;opacity:0}.o1-gB{animation:none;opacity:1}}
</style>
<text class="o1-ttl" x="280" y="42">intra-card SM occupancy</text>
<rect class="o1-base" x="70"  y="90" width="48" height="32" rx="5"/>
<rect class="o1-base" x="140" y="90" width="48" height="32" rx="5"/>
<rect class="o1-base" x="210" y="90" width="48" height="32" rx="5"/>
<rect class="o1-base" x="280" y="90" width="48" height="32" rx="5"/>
<rect class="o1-base" x="350" y="90" width="48" height="32" rx="5"/>
<rect class="o1-base" x="420" y="90" width="48" height="32" rx="5"/>
<rect class="o1-base" x="70"  y="136" width="48" height="32" rx="5"/>
<rect class="o1-base" x="140" y="136" width="48" height="32" rx="5"/>
<rect class="o1-base" x="210" y="136" width="48" height="32" rx="5"/>
<rect class="o1-base" x="280" y="136" width="48" height="32" rx="5"/>
<rect class="o1-base" x="350" y="136" width="48" height="32" rx="5"/>
<rect class="o1-base" x="420" y="136" width="48" height="32" rx="5"/>
<rect class="o1-base" x="70"  y="182" width="48" height="32" rx="5"/>
<rect class="o1-base" x="140" y="182" width="48" height="32" rx="5"/>
<rect class="o1-base" x="210" y="182" width="48" height="32" rx="5"/>
<rect class="o1-base" x="280" y="182" width="48" height="32" rx="5"/>
<rect class="o1-base" x="350" y="182" width="48" height="32" rx="5"/>
<rect class="o1-base" x="420" y="182" width="48" height="32" rx="5"/>
<g class="o1-gB">
<rect class="o1-on" x="70"  y="90" width="48" height="32" rx="5"/>
<rect class="o1-on" x="140" y="90" width="48" height="32" rx="5"/>
<rect class="o1-on" x="210" y="90" width="48" height="32" rx="5"/>
<rect class="o1-on" x="280" y="90" width="48" height="32" rx="5"/>
<rect class="o1-on" x="350" y="90" width="48" height="32" rx="5"/>
<rect class="o1-on" x="70"  y="136" width="48" height="32" rx="5"/>
<rect class="o1-on" x="140" y="136" width="48" height="32" rx="5"/>
<rect class="o1-on" x="210" y="136" width="48" height="32" rx="5"/>
<rect class="o1-on" x="280" y="136" width="48" height="32" rx="5"/>
<rect class="o1-on" x="350" y="136" width="48" height="32" rx="5"/>
<rect class="o1-on" x="70"  y="182" width="48" height="32" rx="5"/>
<rect class="o1-on" x="140" y="182" width="48" height="32" rx="5"/>
<rect class="o1-on" x="210" y="182" width="48" height="32" rx="5"/>
<rect class="o1-on" x="280" y="182" width="48" height="32" rx="5"/>
<rect class="o1-on" x="350" y="182" width="48" height="32" rx="5"/>
<text class="o1-cap o1-b" x="280" y="256">gate-driven CP  -  15 / 18 SMs busy</text>
</g>
<g class="o1-gA">
<rect class="o1-on" x="70"  y="90" width="48" height="32" rx="5"/>
<rect class="o1-on" x="70"  y="136" width="48" height="32" rx="5"/>
<rect class="o1-on" x="70"  y="182" width="48" height="32" rx="5"/>
<text class="o1-cap o1-a" x="280" y="256">serial chunk scan  -  3 / 18 SMs busy</text>
</g>
</svg>
<figcaption>A serial chunk scan keeps only a few streaming multiprocessors busy; the gate's exponential decay lets FlashQLA split the sequence intra-card and fill the grid.</figcaption>
</figure>

FlashQLA's move is **context parallelism inside a single card** — splitting one sequence's scan across multiple SMs — and it earns the right to do so *from the gate*. Here is the reasoning, which is the elegant part.

Normally you cannot parallelize a scan naively: chunk $t+1$ needs chunk $t$'s exact output state. But the gated delta rule's state carry between chunks is $\mathbf{S}_{[t+1]} = \gamma_{[t]}^{C}\mathbf{S}_{[t]} + (\text{local update})$, and $\gamma_{[t]}^{C}$ is a product of $C$ decay gates, each $<1$ — so the coupling from a distant chunk decays **exponentially**. Split the sequence into segments; each segment computes its own *local* scan in parallel (its own partial state), and then the segments are combined with the appropriate cumulative-decay weights. Because the cross-segment influence is gated down, this is a numerically well-conditioned parallel-prefix — a Blelloch-style scan whose combine step is a decay-weighted add.

The result: instead of a few sequential scan streams, you get many independent local scans that saturate the SM grid, then a cheap logarithmic-depth combine. And crucially, FlashQLA does this **automatically** — it detects the under-occupied regimes (high TP, long sequence, small head count) and enables intra-card CP without the user configuring anything.

| Regime | Heads per GPU | Scan streams | Without CP | With gate-driven CP |
|---|---|---|---|---|
| TP1, many heads | 64 | plenty | SMs already full | ~no change |
| TP4, medium heads | 32 | moderate | ~60% occupancy | fills to near-peak |
| TP8, tiny heads | 8–16 | few | ~25% occupancy, memory-latency-bound | biggest win (approaches the 3× headline) |
| Very long sequence, batch 1 | any | 1 stream | badly serialized | split into many segments |

This is the technique that makes the **3× forward** number appear specifically at TP8 with small heads — the regime that matters for serving a 397B-parameter hybrid model, where you shard across eight GPUs and each one is left holding very few Gated DeltaNet heads. Without CP, that is the worst case for a linear-attention kernel. With it, it becomes the best case.

### Second-order optimization: the decay is a knob, not a constant

A non-obvious consequence: the *amount* of parallelism CP can safely extract depends on how fast the gate decays. Aggressively decaying heads (small $\alpha$) decouple across chunks faster, so they tolerate more segments with less error; near-1 gates couple longer and want fewer, longer segments. FlashQLA's automatic policy is conservative here — it splits based on shape (TP degree, sequence length, head count) rather than on runtime gate statistics — which is the right call for a kernel that must be numerically bit-reproducible for training. If you are doing research on the decay schedule itself, this is the interaction to keep in mind.

## Technique 2 — Hardware-friendly algebraic reformulation

The second technique is the least flashy and, in some ways, the most important, because it is the one that touches numerical precision. A modern GPU SM has three very different compute resources, and they run at wildly different throughputs:

- **Tensor Cores** — enormous matmul throughput (the reason the H200 exists), but only for matmuls in supported shapes/dtypes.
- **CUDA Cores** — general FP32/FP16 arithmetic; an order of magnitude less throughput than Tensor Cores for the same FLOPs.
- **SFU (Special Function Unit)** — transcendentals like `exp`, `log`; scarce, and used heavily by the gate ($\alpha = \exp(g)$, cumulative decays).

A textbook implementation of the chunked gated delta rule lands a lot of work on the CUDA Cores and SFU: the elementwise gate scalings $\operatorname{Diag}(\gamma)$, the decay masks $\Gamma$, the `exp`/`log` for the cumulative products, the triangular-solve fixups. Every one of those is arithmetic that, if you are not careful, *serializes against* the Tensor Core matmuls instead of overlapping with them — and the SFU in particular becomes a hidden bottleneck because `exp` is expensive and there is one per gate per token.

FlashQLA **restructures the forward and backward computation flows to reduce Tensor Core, CUDA Core, and SFU overhead without sacrificing numerical precision.** Concretely, the family of reformulations does things like:

- **Fuse the gate scaling into the GEMM epilogue** instead of a separate elementwise pass, so $\operatorname{Diag}(\gamma)\mathbf{Q}\mathbf{S}^\top$ costs one matmul, not a matmul plus a CUDA-core multiply.
- **Hoist and share cumulative-decay computations** ($\gamma$, $\gamma^C/\gamma$) so the SFU computes each `exp`/`log` once per chunk and reuses it, rather than recomputing per matmul.
- **Reassociate the intra-chunk products** so more of the total FLOPs land on Tensor Cores and fewer on CUDA Cores — the WY/UT triangular solve is arranged to be a batched Tensor-Core operation rather than a scalar forward-substitution loop.
- **Keep the state accumulation in FP32** while feeding BF16 operands to the Tensor Cores — the reformulation is careful to accumulate $\mathbf{S}$ in high precision so that long sequences do not drift, which is what "without sacrificing numerical precision" buys you.

| Work item | Naive placement | FlashQLA placement | Why it matters |
|---|---|---|---|
| Gate scaling $\operatorname{Diag}(\gamma)\cdot$ | Separate CUDA-core pass | Fused into GEMM epilogue | Removes a full-tensor read/write + serialization |
| Cumulative decay $\gamma$ (`exp`/`log`) | Recomputed per use | Computed once per chunk, reused | SFU is scarce; `exp` is not free |
| UT triangular solve | Scalar forward-substitution | Batched Tensor-Core solve | Moves FLOPs off CUDA cores onto Tensor Cores |
| State accumulation | BF16 in-place | FP32 accumulator, BF16 operands | Long-sequence numerical stability |
| Decay mask $\odot\,\Gamma$ | Materialized mask matrix | Folded into the masked-GEMM | Avoids an extra $C\times C$ tensor |

The reason this must not "sacrifice numerical precision" is that FlashQLA is a **training** kernel first. If the reformulation changed the answer even slightly, gradients would diverge from a reference and the model would train differently — or worse, subtly wrongly. The bar is that the reassociated math is *algebraically identical* (up to floating-point accumulation order, which is controlled by keeping the state in FP32). This is the difference between a serving-only kernel, which can take precision liberties, and one that Qwen uses to pretrain a frontier model. If you have felt the pain of [debugging training numerics](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — loss spikes that trace back to an accumulation-order change in a fused kernel — you understand why this clause is load-bearing.

## Technique 3 — TileLang fused warp-specialized kernels

The third technique is where all of this becomes an actual `.cu` — or rather, a TileLang program. The two obvious ways to implement the chunked gated delta rule are both wrong for peak performance:

1. **Decompose into many small kernels** (one for the UT solve, one for the intra-chunk GEMM, one for the state update, …). Clean, but every kernel boundary is a round-trip to HBM and a launch, and the Tensor Cores stall between them.
2. **One giant monolithic kernel** with everything inline. Fewer round-trips, but a single warp doing load-then-compute-then-store serializes the three engines from Technique 2 — while the Tensor Cores run, the load units and CUDA cores wait, and vice versa.

FlashQLA takes the third path: **fused kernels with manual warpgroup specialization.** On Hopper, warps are organized into warpgroups, and you can dedicate different warpgroups to different roles — a **producer** warpgroup that does nothing but issue TMA (Tensor Memory Accelerator) loads to stream the next chunk's data into shared memory, and **consumer** warpgroups that run the Tensor-Core matmuls and the CUDA-core/SFU gate math. With a shared-memory pipeline between them, the producer is fetching chunk $k{+}1$ while the consumers compute on chunk $k$.

![Warp-specialized kernel: a TMA/load lane, a Tensor-Core lane, and a CUDA-core+SFU lane all run in the same cycle instead of serializing](/imgs/blogs/flashqla-gated-deltanet-kernels-6.webp)

Read the three lanes as three engines that would otherwise take turns. The **TMA/load** lane streams chunks into SMEM. The **Tensor Core** lane runs the UT solve, the $\mathbf{Q}\mathbf{K}^\top$, the state GEMM, the output GEMM. The **CUDA core + SFU** lane computes the `exp` gates, scales, accumulates the FP32 state, and casts the output back to BF16. Because they are different warpgroups on the same SM, all three are busy *in the same cycle* — the whole point of warp specialization, and where the memory-movement/compute overlap that Technique 2 set up actually gets realized. TileLang is what makes this expressible without hand-writing PTX: you describe the tiles, the pipeline stages, and the warpgroup roles, and it generates the specialized kernel.

Why TileLang and not Triton or raw CUDA? A few reasons that matter in practice:

| Dimension | Raw CUDA / CUTLASS | Triton (what FLA uses) | TileLang (what FlashQLA uses) |
|---|---|---|---|
| Warpgroup specialization | Full control, but you hand-write everything | Limited; scheduler-driven | First-class, explicit producer/consumer |
| TMA + async pipelines | Manual, verbose | Partial | Expressed as pipeline stages |
| Iteration speed | Slow | Fast | Fast, with lower-level control than Triton |
| Hopper/Blackwell features | Everything, painfully | Lags new hardware | Targets SM90/SM100 directly |

This is also why the hardware requirements are strict: **SM90 (Hopper: H100/H200) or SM100 (Blackwell), CUDA 12.8+, PyTorch 2.8+.** Warp specialization and TMA are Hopper-and-later features; there is no meaningful port to Ampere because the hardware mechanisms the kernel is built around do not exist there. FlashQLA also runs on **SM121 / GB10** systems like DGX Spark, which is the bridge to the edge-inference use case we will hit in the case studies. If you want the broader context on why custom kernels like this exist at all, our post on [custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference) covers the fusion-and-overlap playbook that FlashQLA is a state-of-the-art instance of.

## Reading the benchmarks

The headline numbers are **2–3× forward and ~2× backward over the FLA Triton kernel**, and roughly **2× over FlashInfer 0.6.9**, measured on Qwen3.5/3.6 head configurations across tensor-parallel settings TP1–TP8 (value-head counts $h_{k,v} \in \{64, 48, 32, 24, 16, 8\}$) on NVIDIA H200. But the *shape* of the speedup is more instructive than any single number.

![FlashQLA's speedup over the FLA Triton kernel grows as head count shrinks toward TP8 — the regime where linear attention most under-occupies the GPU](/imgs/blogs/flashqla-gated-deltanet-kernels-7.webp)

The figure sketches the reported envelope (the exact per-cell values depend on batch and sequence length; treat them as representative of the 2–3× / 2× bands, not a published lookup table). The trend it encodes is the real, documented design story: **the forward advantage grows as heads shrink from TP1 toward TP8.** That is not a coincidence — it is Technique 1 paying off. TP1 with 64 heads already fills the SMs, so there is less to win; TP8 with $\le 16$ heads is the under-occupied regime where gate-driven context parallelism turns a starving kernel into a saturated one, and that is where the 3× lives.

A few honest caveats on reading these:

- **Forward benchmarks measure single-kernel latency** across varying batch lengths; **backward benchmarks measure total tokens per batch in a single update step.** They are microbenchmarks of the GDN op, not end-to-end model throughput. In a full hybrid model the GDN layers are ~75% of the blocks, so a 2–3× kernel speedup on those translates to a meaningful — but not 2–3× — end-to-end win, because the full-attention layers and MLPs are unaffected.
- **Backward tops out around 2×** because the save-`A`-and-`h` strategy removes the recompute, but the gradient GEMMs themselves are already reasonably shaped in FLA. The forward has more headroom precisely because occupancy (not arithmetic) was its bottleneck.
- **The FlashInfer comparison (~2×)** is against a different, well-optimized baseline, which is a stronger claim than beating a reference Triton kernel.

| Metric | FLA Triton (baseline) | FlashInfer 0.6.9 | FlashQLA |
|---|---|---|---|
| Forward, TP1 (h=64) | 1.0× | ~similar | ~2.0× |
| Forward, TP8 (small heads) | 1.0× | — | up to ~3.0× |
| Backward | 1.0× | — | ~2.0× |
| Hardware | Ampere+ | Hopper+ | **Hopper/Blackwell only (SM90/SM100)** |
| Primary use | General | Serving | **Pretraining + edge agentic inference** |

> The benchmark to trust is not the biggest multiplier on a slide; it is the one measured in your regime. For a TP8-sharded 397B hybrid model during long-context prefill, FlashQLA's regime and yours are the same — which is exactly why Qwen built it.

### A worked throughput example

Kernel multipliers are seductive; end-to-end wins are what pay for the migration. Let us do the arithmetic that keeps you honest. Take a Qwen3.5-style hybrid where 75% of the token-mixing blocks are Gated DeltaNet and 25% are full attention, and suppose that during a long-context prefill the token-mixing layers are ~40% of total step time (the rest being MLP/MoE, norms, projections). Of that 40%, say the GDN kernels are ~30 points and the attention kernels ~10 points, because there are three GDN layers per attention layer.

Now apply an Amdahl bound. FlashQLA speeds up only the GDN portion. At a 2.5× kernel speedup on that 30% slice:

$$\text{step time} \rightarrow 0.70 + \underbrace{0.10}_{\text{attn}} + \underbrace{\frac{0.30}{2.5}}_{\text{GDN}} = 0.70 + 0.10 + 0.12 = 0.92$$

That is a ~9% reduction in step time — a **1.09× end-to-end** from a 2.5× kernel speedup, because the GDN kernel was 30% of the budget. Underwhelming? Only if you stop reading. Two things change the picture. First, at TP8 with small heads the *un-optimized* GDN slice is far larger than 30% (it is the pathological occupancy case), so the same kernel speedup removes a much bigger slice — the end-to-end number climbs toward 1.3–1.5×. Second, at the fleet scale Qwen trains at, 9% of a multi-week pretraining run is days of H200 time. **Lesson:** compute the Amdahl bound for *your* regime before you get excited or disappointed — the kernel multiplier and the end-to-end multiplier are different numbers, and the gap between them is just how big the slice was.

| Your regime | GDN slice of step time | Kernel speedup | Amdahl end-to-end |
|---|---|---|---|
| TP1, many heads, short seq | ~15% | ~2.0× | ~1.08× |
| TP4, medium heads, long seq | ~30% | ~2.5× | ~1.20× |
| TP8, small heads, long-context prefill | ~50% | ~3.0× | ~1.50× |

## How FlashQLA compares to other linear-attention kernels

FlashQLA is not the only fast recurrent-attention kernel, and understanding the neighbors clarifies what it is and is not. The linear-attention kernel landscape splits by *which recurrence* it computes and *which hardware trick* it leans on.

| Kernel / family | Recurrence | Key trick | Hardware | Niche |
|---|---|---|---|---|
| **FlashQLA** | Gated delta rule | Gate-driven CP + warp-specialized TileLang | SM90/SM100 only | Qwen GDN training + edge inference |
| **FLA (Triton)** | Many (incl. gated delta rule) | Chunked scan + UT transform in Triton | Ampere+ | Reference, broad coverage |
| **Mamba2 selective scan** | SSM (scalar-gated, no delta) | Hardware-aware parallel scan | Ampere+ | Pure SSM models |
| **Lightning Attention** | Linear attention (decay, no delta) | Block-wise linear + IO-aware | Ampere+ | MiniMax-01/M1 |
| **FlashAttention / FlashInfer** | Softmax attention | Tiling + online softmax | Ampere+ | Full-attention layers |

Three distinctions matter. First, **FlashQLA computes the *gated delta* rule specifically** — the erase-then-write update — which is strictly more than Mamba2's scalar-gated SSM (no delta correction) or lightning attention (decay but no delta). More expressive memory, harder kernel. Second, **FlashQLA is Hopper/Blackwell-only by design**, trading portability for the warp-specialization and TMA that FLA's portable Triton cannot fully exploit; FLA remains the right choice on Ampere or for architectures FlashQLA does not implement. Third, FlashQLA and FlashAttention are *complements, not competitors* in a hybrid model — one drives the 75% of linear blocks, the other the 25% of full-attention blocks. If you are weighing linear vs. full attention at the architecture level, our notes on [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) and [MiniMax's lightning-attention hybrids](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe) map the design space FlashQLA sits inside.

The takeaway: FlashQLA is the *most specialized* kernel in this table — one recurrence, two GPU generations, two use cases — and that specialization is exactly what buys the 2–3×. FLA is the generalist you fall back to; FlashQLA is the sharp instrument for the shape Qwen actually ships.

## Putting it to work

Installation is a one-liner, with a from-source path for pinned or bleeding-edge builds:

```bash
# From PyPI (requires a Hopper/Blackwell GPU, CUDA 12.8+, PyTorch 2.8+)
pip install flash-qla

# Or build from source (for a pinned commit or local patches)
git clone https://github.com/QwenLM/FlashQLA.git
cd FlashQLA
pip install -v .

# Sanity-check the kernel against its reference
python -m pytest tests/test_gdr_unit.py -v

# Profile against the FLA Triton kernel (installs the baseline)
pip install flash_linear_attention==0.5.0
python profile/profile_gdr.py --set develop
```

Because the high-level function mirrors FLA's signature, migrating an existing Gated DeltaNet layer is usually a two-line diff. Here is the shape of that swap inside a model's token-mixer:

```python
# Before: FLA's Triton kernel
# from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# After: FlashQLA's TileLang kernel — same call, different backend
from flash_qla import chunk_gated_delta_rule


class GatedDeltaNet(torch.nn.Module):
    def __init__(self, d_model, n_heads, head_dim):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q_proj = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.b_proj = torch.nn.Linear(d_model, n_heads, bias=False)   # beta
        self.g_proj = torch.nn.Linear(d_model, n_heads, bias=False)   # gate (log-decay)
        self.o_proj = torch.nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x, initial_state=None, cu_seqlens=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        beta = self.b_proj(x).sigmoid()                # beta in (0, 1)
        g = torch.nn.functional.logsigmoid(self.g_proj(x))  # log-decay <= 0

        o, final_state = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=self.head_dim ** -0.5,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        return self.o_proj(o.reshape(B, T, -1)), final_state
```

The only thing to verify after the swap is that your `g` convention matches — FlashQLA expects the gate as a log-decay so that `alpha = exp(g)`. If your existing layer stored `alpha` directly, take its log before passing it in. Run `test_gdr_unit.py` shapes against your own, then diff a forward pass against the FLA reference on a fixed seed; they should match to BF16 tolerance.

### Streaming a long prefill

The `initial_state` / `final_state` pair is what turns FlashQLA into a windowed prefill engine. To prefill a very long context without holding it all at once:

```python
state = None
outputs = []
for window in x.split(WINDOW, dim=1):          # e.g. WINDOW = 32768
    o, state = chunk_gated_delta_rule(
        q=Q(window), k=K(window), v=V(window),
        g=G(window), beta=B(window), scale=scale,
        initial_state=state,                   # carry the state across windows
        output_final_state=True,
    )
    outputs.append(o)
out = torch.cat(outputs, dim=1)
```

Each window's cost is bounded, the state is $O(1)$, and you have decoupled peak activation memory from context length — the property that makes million-token prefill practical on a hybrid model.

### Debugging a FlashQLA integration

When a GDN layer misbehaves after wiring in FlashQLA, the failure almost always lands in one of a handful of buckets. This is the table to keep next to the profiler:

| Symptom | Most likely cause | Where to look |
|---|---|---|
| Output diverges from FLA reference | Gate convention: passing `alpha` where `g = log(alpha)` is expected | Confirm `g` is a log-decay ($\le 0$); `alpha = exp(g)` |
| Loss NaNs after a few hundred steps | Gate not clamped; `exp(g)` overflow, or `beta` outside $(0,1)$ | Clamp `g` from above; `sigmoid` the `beta` head |
| State leaks across documents | Missing `cu_seqlens` on packed batches | Pass cumulative seqlens; verify boundary resets |
| Import / launch error | Ampere GPU, or CUDA < 12.8 / PyTorch < 2.8 | Requires SM90/SM100; check `torch.cuda.get_device_capability()` |
| Backward far slower than expected | Rebuilding `A`/`h` instead of reusing them | Use `_fwd`/`_bwd` and thread the saved tensors |
| Slower than FLA at TP1, many heads | Already SM-saturated; CP has nothing to recover | Expected — the win is in the small-head regime |
| Wrong output only at chunk boundaries | Mismatched chunk size vs. reference, or state-carry sign | Diff a single chunk's intermediates against the recurrence |

The first row is the one that bites nearly every migration, and it is silent: an `alpha`-vs-`log(alpha)` mismatch does not crash, it just trains a subtly different (worse) model, or matches for a few steps then diverges. Always run the fixed-seed forward diff against the FLA reference before trusting a swap — it takes a minute and saves a day. This is the same discipline we preach for any [kernel-fusion change under torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile): the fused path is only a win if it is first proven equivalent.

## Case studies from production

Kernels earn their reputation in the specific, ugly situations where the microbenchmark's assumptions break. Here are nine — some are Qwen's documented use cases, some are the failure modes any team integrating FlashQLA will hit.

### 1. The TP8 small-head cliff

**Symptom.** A team shards a 397B hybrid model across eight H200s (TP8) and finds that Gated DeltaNet prefill is *slower per token* than the full-attention layers, even though GDN is supposed to be the cheap one. The profiler shows GDN kernels at ~25% SM occupancy while the full-attention FlashAttention kernels sit near 80%.

**Wrong first hypothesis.** "The GDN math must be more expensive than we thought — let us reduce the state dimension." That is chasing the wrong bottleneck; the kernel is not compute-bound, it is *occupancy-bound*.

**Root cause.** At TP8, each GPU holds only 8–16 value heads. The chunked scan produces one sequential stream per (batch, head), so a batch of 1 with 8 heads is 8 sequential scans on a 132-SM chip. The Tensor Cores are idle most of the time, waiting on the memory-latency-bound state carry.

**Fix.** FlashQLA's gate-driven intra-card context parallelism, enabled automatically in exactly this regime, splits each long scan into parallel segments and fills the SMs. This is where the 3× headline number comes from — it is not a uniform speedup, it is the recovery of a pathological under-occupancy case. **Lesson:** for linear attention, *occupancy is the first thing to profile, not FLOPs*. The asymptotics lie about wall-clock when the chip is starving.

### 2. Million-token prefill on a hybrid model

**Symptom.** Prefilling a 1M-token context on a Qwen3.5-class model, the full-attention layers' KV cache is the memory hog everyone expects — but the wall-clock is dominated by the GDN layers, because there are three of them for every attention layer and each runs a long sequential scan.

**Wrong first hypothesis.** "The attention layers must be the bottleneck at 1M tokens; optimize the KV cache." True for memory, false for the GDN compute time.

**Root cause.** A 1M-token scan with a modest number of heads is the long-sequence, batch-1 regime — very few parallel streams, a very long serial dependency. Even with chunking, the inter-chunk carry is a million/128 ≈ 8,000-step sequential chain per head.

**Fix.** Two FlashQLA properties compound here. Context parallelism splits that 8,000-step chain into parallel segments (the gate's exponential decay makes the split numerically safe), and windowed streaming via `initial_state`/`final_state` bounds activation memory. Together they turn a serialized megasecond-scale scan into a saturated one. **Lesson:** in a hybrid model, the linear layers are the *majority* of the network; their kernel is not a rounding error at long context, it is the main event.

### 3. Edge agentic inference on DGX Spark / GB10

**Symptom.** An agent running locally on a GB10 (DGX Spark, SM121) needs to prefill a long tool-use trajectory before each step. On this small, power-constrained part, a poorly-occupied linear-attention kernel means the agent stalls between actions.

**Wrong first hypothesis.** "Edge hardware is just slow; quantize harder." Quantization helps memory, but the stall is the prefill scan's latency, not bandwidth.

**Root cause.** GB10 has far fewer SMs than an H200, so the *balance* between the sequential state carry and the parallel intra-chunk work is different — and the SFU/CUDA-core gate math is a larger fraction of a smaller chip's budget.

**Fix.** FlashQLA explicitly targets SM121 and the "edge-side agentic inference" use case. The warp-specialized overlap (Technique 3) matters *more* on a small chip, because there is less slack to hide un-overlapped scalar work; keeping the SFU busy on gates while the Tensor Cores run the GEMMs is what keeps prefill latency low enough for interactive agents. **Lesson:** the same kernel that wins on an eight-GPU training cluster (occupancy) wins on a single edge chip (overlap) — for different reasons, from the same design.

### 4. The backward pass that was 4× the forward

**Symptom.** A team writes their own Gated DeltaNet autograd `Function` around a chunked forward, and training is dominated by the backward pass — profiled at nearly 4× the forward's time, wrecking tokens/sec.

**Wrong first hypothesis.** "Backward is inherently ~2× forward; this is just how it is." Not when you are silently recomputing.

**Root cause.** Their backward re-ran the entire chunked scan to reconstruct the per-chunk states, *then* ran the gradient scan — two full passes plus overhead. The forward's intermediate `A` (UT matrix) and `h` (chunk states) were thrown away and rebuilt.

**Fix.** Use `chunk_gated_delta_rule_fwd` and pass its returned `A` and `h` into `chunk_gated_delta_rule_bwd`. The backward becomes GEMMs against materialized tensors, no scan recompute — which is why FlashQLA's backward lands at ~2× the forward instead of 4×. The cost is holding `A` and `h` in memory; for GDN that is small relative to activations. **Lesson:** for any recurrent kernel, *what the forward saves determines what the backward costs*. Recompute-vs-store is the dominant knob, and it is exactly the kind of tradeoff we cover for [gradient checkpointing](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload).

### 5. Chunk size: 64 vs 128 vs 256

**Symptom.** Sweeping the chunk size $C$, a team finds 128 fastest on their shapes, but 64 wins for very small heads and 256 is never best — and they want to understand why before hard-coding a value.

**Wrong first hypothesis.** "Bigger chunks amortize overhead, so 256 should win." True for launch overhead, false for the UT solve.

**Root cause.** The chunk size trades two costs. The UT triangular solve is $O(C^2)$ per chunk in the intra-chunk term (a $C \times C$ matrix), so doubling $C$ roughly quadruples that work; but larger chunks mean fewer inter-chunk steps and better GEMM shapes. The sweet spot is where the $C \times C$ intra-chunk cost balances the number of sequential steps. For small heads, the GEMMs are already skinny, so the $C^2$ term dominates earlier and 64 can win; for larger heads, 128 hides the solve cost behind fat matmuls.

**Fix.** Trust FlashQLA's tuned default for the target shapes, and only sweep if you are in an unusual regime (tiny heads, unusual sequence lengths). **Lesson:** chunk size is a real Pareto knob, not a constant — and the $O(C^2)$ UT-solve term is the reason the curve is not monotone.

| Chunk size $C$ | UT-solve cost (per chunk) | Inter-chunk steps | Best for |
|---|---|---|---|
| 64 | low ($64^2$) | many | very small heads, tiny GEMMs |
| 128 | moderate | moderate | the common case (Qwen defaults) |
| 256 | high ($256^2$) | few | rarely optimal — solve dominates |

### 6. Variable-length packing with `cu_seqlens`

**Symptom.** A pretraining run packs many short documents into each sequence to avoid padding waste. Passing them as a padded rectangle, half the tokens are padding, and the GDN kernel dutifully scans the padding — burning ~2× the compute it should.

**Wrong first hypothesis.** "Padding is cheap, the kernel will mask it." The chunked scan does not free-lunch skip padded positions; they still occupy chunks.

**Root cause.** A padded `[B, T]` batch where `T` is the longest document means every shorter document carries dead tokens through the entire scan, and — worse — the state must not leak across the document boundary, which naive padding does not guarantee.

**Fix.** Pack the documents end-to-end into a single long sequence and pass `cu_seqlens` (cumulative sequence lengths), exactly like FlashAttention's varlen path. FlashQLA resets the recurrent state at each boundary and scans only real tokens. On a corpus of short docs this alone can double effective throughput and — critically — keeps each document's memory isolated. **Lesson:** for GDN, `cu_seqlens` is not just a throughput optimization, it is a *correctness* feature; without it, one document's state bleeds into the next.

### 7. Migrating a live training run off the FLA Triton kernel

**Symptom.** A team wants FlashQLA's speed mid-run but is terrified of perturbing a converging loss curve — a fused-kernel swap that changes numerics even slightly can spike the loss.

**Wrong first hypothesis.** "It is a drop-in; just change the import and go." Drop-in for *shapes*, yes; identical *numerics*, verify.

**Root cause.** Two kernels computing the "same" math can differ in floating-point accumulation order, and a training run is sensitive to that in the tails. The `g` convention (log-decay vs `alpha`) is also an easy silent mismatch.

**Fix.** Stage it: (1) swap the import behind a flag; (2) run `test_gdr_unit.py` and a fixed-seed forward diff against FLA to confirm BF16-tolerance agreement; (3) confirm the gate convention (`g = log(alpha)`); (4) check gradients match on a small batch via the explicit `_fwd`/`_bwd` path; (5) enable on a canary replica and watch the loss for a few hundred steps before rolling out. FlashQLA's Technique-2 promise — "without sacrificing numerical precision," with FP32 state accumulation — is what makes this migration safe, but you still verify. **Lesson:** "numerically identical" is a claim you *test*, not one you assume, when a live loss curve is on the line.

### 8. The gate-convention NaN that survived the unit tests

**Symptom.** A team's `test_gdr_unit.py` passes, a small forward diff against FLA matches, and yet a full pretraining run NaNs the loss around step 800 — reliably, on multiple seeds.

**Wrong first hypothesis.** "It is a learning-rate or data problem; the kernel passed its tests." The tests used random `g` in a benign range; production did not.

**Root cause.** The `g` head was a raw `Linear` with no activation, so early in training it could emit large positive values; `alpha = exp(g)` then blew past 1, the state grew without bound, and after enough steps a chunk overflowed to `inf` → `NaN`. The unit test never exercised `g > 0` because it sampled from a small range. The gate must encode a *decay*, i.e. $\alpha \in (0,1)$, so $g \le 0$.

**Fix.** Constrain the gate head — `logsigmoid` (as in the layer example above) or an explicit upper clamp on `g` — so `alpha` stays in $(0,1)$ by construction, and add a production-range case to the unit test. FlashQLA computes exactly what you feed it; it will not silently sanitize an out-of-range gate. **Lesson:** a kernel's unit test proves *the kernel*, not *your inputs*. The gate convention is a contract, and violating it fails slowly and catastrophically rather than fast and loudly.

### 9. Overlapping GDN prefill with the attention layers in a hybrid schedule

**Symptom.** A serving team pipelines a hybrid model's prefill and notices the GDN and full-attention layers each leave gaps — GDN is occupancy-bound early, attention is bandwidth-bound — and neither saturates the GPU on its own for short-to-medium contexts.

**Wrong first hypothesis.** "Just make each kernel faster in isolation." Each is already near its own ceiling; the waste is *between* them.

**Root cause.** In a 3:1 hybrid, the layers alternate GDN, GDN, GDN, attention, …; the two kernel types have different bottlenecks (compute/occupancy vs. memory bandwidth), so a strictly sequential schedule leaves whichever resource the current layer does not use idle.

**Fix.** This is where FlashQLA's high occupancy pays a second dividend: because gate-driven CP already fills the SMs *within* a GDN layer, there is less idle compute to try to overlap across layers, and the scheduler's job simplifies to hiding the attention layers' bandwidth stalls behind GDN compute. FlashQLA does not schedule across layers itself — that is the serving engine's job — but a kernel that saturates its own SMs is far easier to pipeline than one that starves them. **Lesson:** a well-occupied kernel is not just faster in isolation; it is a better *citizen* in a pipeline, because it leaves fewer holes for the scheduler to paper over. The interaction between per-kernel occupancy and cross-layer scheduling is the kind of thing our [request-scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) notes dig into.

## When to reach for FlashQLA — and when not to

FlashQLA is a sharp, specialized tool. It is the right one when your situation matches what it was built for, and overkill or unusable when it does not.

**Reach for FlashQLA when:**

- You are **training or prefilling a Gated DeltaNet / gated-delta-rule model** (Qwen3.5/3.6-style hybrids, or your own GDN variant) on **Hopper or Blackwell**.
- You shard with **high tensor parallelism** and each GPU is left with **few heads** — the TP4–TP8 regime where the gate-driven context parallelism delivers its biggest wins.
- You do **long-context prefill** where the linear layers' sequential scan, not the attention KV cache, dominates wall-clock.
- You need **edge agentic inference** on SM121/GB10 hardware and want the warp-specialized overlap to keep prefill latency interactive.
- You currently use the **FLA Triton kernel** and want a mostly-drop-in 2–3× on the forward and ~2× on the backward.

**Skip FlashQLA when:**

- Your model uses **standard softmax attention** (or MLA, sliding-window, NSA/DSA sparse attention) — FlashQLA computes the gated delta rule and nothing else; for those, see [FlashMLA and the DeepSeek open-infra kernels](/blog/machine-learning/open-source-library/deepseek-open-infra-deepep-deepgemm-flashmla) or [trainable sparse attention](/blog/machine-learning/large-language-model/trainable-sparse-attention-nsa-vs-dsa).
- You are on **Ampere or older** (SM80) — the warp-specialization and TMA the kernel is built on do not exist there; use the FLA Triton kernel instead.
- You are **decoding one token at a time** — the recurrence is already cheap at decode; FlashQLA optimizes the *chunked prefill/training* path, not single-step decode.
- You have **plenty of heads per GPU** (TP1, many heads) — you are already SM-saturated, so the context-parallelism win shrinks toward zero and the migration may not be worth it.
- You need **strict bit-for-bit reproducibility against an existing FLA-trained checkpoint** and cannot afford a validation pass — the numerics are close, but "close" is not "identical," and you must verify before committing a live run (see case study 7).

The deeper lesson FlashQLA teaches is the one it opens with: **linear attention is not automatically fast.** Its $O(n)$ asymptotics are necessary but nowhere near sufficient. Turning a recurrence into a fast kernel means reshaping it into Tensor-Core-friendly matmuls (the chunked/UT form), filling the SMs when the scan would otherwise starve them (gate-driven context parallelism), keeping the scalar gate work off the critical path (the algebraic reformulation), and overlapping load with compute (warp specialization). FlashQLA is what "we did all four, carefully, for the exact shapes a frontier hybrid model needs" looks like — and it is why three-quarters of Qwen3.5's blocks can be linear attention without paying for it at the kernel level.

## Further reading

- [FlashQLA on GitHub](https://github.com/QwenLM/FlashQLA) — source, benchmarks, and the reference unit tests.
- [*Gated Delta Networks: Improving Mamba2 with Delta Rule*](https://arxiv.org/abs/2412.06464) — the model FlashQLA computes.
- [*Parallelizing Linear Transformers with the Delta Rule over Sequence Length*](https://arxiv.org/abs/2406.06484) — the chunked/WY derivation the UT transform comes from.
- [Qwen3-Next: hybrid attention and ultra-sparse MoE](/blog/paper-reading/large-language-model/qwen3-next-hybrid-attention-ultra-sparse-moe) — the architecture FlashQLA serves.
- [Qwen3.5-397B flagship hybrid MoE](/blog/paper-reading/large-language-model/qwen3-5-397b-flagship-hybrid-moe) — the flagship that ships these kernels.
- [Custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference) — the fusion-and-overlap playbook FlashQLA is an instance of.

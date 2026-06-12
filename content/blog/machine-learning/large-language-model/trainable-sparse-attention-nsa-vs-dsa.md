---
title: "Trainable Sparse Attention: DeepSeek's Native Sparse Attention vs DeepSeek Sparse Attention"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A principal-engineer deep-dive contrasting two DeepSeek philosophies of trainable sparse attention — NSA, a three-branch design trained sparse from scratch and hardware-aligned for real wall-clock speedups, versus DSA, a production retrofit that bolts a cheap lightning indexer onto an already-trained dense model. When each one wins, with the numbers."
tags: ["llm", "sparse-attention", "native-sparse-attention", "deepseek-sparse-attention", "deepseek-v3-2", "long-context", "attention", "kv-cache", "inference-optimization", "training", "transformer", "gpu-kernels"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

The rule of thumb that every serving engineer eventually internalizes is brutal: **attention is the part of the transformer that gets worse as your product gets better.** The moment your users start pasting 60-page documents, dumping entire codebases into the context, or running multi-hour agent sessions, the quadratic term in attention stops being a footnote and becomes the bill. Full attention costs `O(L²)` in both compute and the memory traffic of streaming the KV cache, and `L` is the one number your customers keep pushing up.

Sparse attention is the obvious escape hatch — if a query only really needs a few hundred of the tokens before it, why pay to score all of them? The catch, and the reason sparse attention spent years as a research curiosity rather than a production default, is that most sparse-attention schemes were *inference-only approximations bolted onto a model that was trained dense.* They saved theoretical FLOPs that never materialized as wall-clock speedups, and they often degraded quality because the model was never trained to live inside the sparse pattern. The model learned to attend densely, and then at serving time you yanked most of its keys away.

DeepSeek has now published two answers to this, eleven months apart, that are interesting precisely because they disagree about *how* to make sparse attention trainable. **Native Sparse Attention (NSA**, arXiv 2502.11089, February 2025) trains a model sparse from the very first step and co-designs the sparsity pattern with the GPU so the FLOP savings become real backward-pass and decode speedups. **DeepSeek Sparse Attention (DSA**, shipped inside DeepSeek-V3.2, arXiv 2512.02556, December 2025) takes the opposite bet: keep the already-trained dense model, and graft a small, separately-trained indexer onto it that picks which keys to attend to. NSA is the research artifact that pays the design cost up front; DSA is the production retrofit that minimizes risk and ships a price cut.

![Dense attention scores every query against every key for O(L^2) cost, while sparse attention scores a query against only k selected tokens for O(L x k) cost, with k far smaller than L](/imgs/blogs/trainable-sparse-attention-nsa-vs-dsa-1.webp)

The diagram above is the mental model the whole post turns on. On the left, dense attention: query `q_t` attends every one of the `L` keys, materializes an `L × L` score matrix, and its cost grows as `L²`. On the right, sparse attention: the same query attends only `k` selected tokens, scores a `k × 1` vector, and cost grows linearly in `L` for a fixed budget `k`. Both NSA and DSA are machines for choosing that subset of `k` tokens *in a way the model is trained to rely on.* The entire contrast between them is in the answers to three questions: how do you pick the `k` tokens, when do you train the picker, and what does the GPU actually do when it runs.

This is part of a series mining DeepSeek's published work for reusable engineering. If you want the surrounding stack, the [DeepSeek-V3 training-recipe teardown](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) covers the FP8/MoE/MTP foundation both of these attention variants sit on, and the MLA deep-dive (the sibling post on Multi-head Latent Attention) covers the compressed-KV attention that DSA in particular is layered over. This post assumes you know what a KV cache is; if not, start with the [KV cache guide](/blog/machine-learning/large-language-model/kv-cache).

## Why sparse attention is different this time

The reason to care about *trainable* sparse attention — as opposed to the dozen inference-time sparse schemes that came before — is that the prior generation almost universally broke one of three promises. Here is the gap between the assumption people carry into "we'll just make attention sparse" and what actually happens.

| Assumption | The naive view | The reality |
|---|---|---|
| "Fewer FLOPs means faster" | Skip 90% of the score matrix, get a ~10x speedup | Scattered gathers and per-head refetches make a sparse kernel *slower* than a dense FlashAttention kernel on real hardware unless the access pattern is contiguous and Tensor-Core-friendly |
| "Sparsity is an inference trick" | Train dense, prune attention at serving time | A model trained dense has learned to spread attention; remove keys it relied on and quality drops, especially on the long-context tasks you added sparsity for |
| "The backward pass doesn't matter" | Sparse attention is about cheaper decoding | If you want to *train* long-context models cheaply, the `O(L²)` backward pass is the dominant cost; an inference-only sparse method does nothing for it |
| "Pick the top-k tokens per head" | Each head independently gathers its favorite keys | Per-head token-level gathers fragment memory access; in grouped-query attention the heads in a group should share one fetched block, not each pull their own |
| "Any selection granularity is fine" | Select individual tokens for maximum precision | Token-level selection produces non-contiguous memory and underutilizes Tensor Cores, which want dense blocks; block-level selection is the hardware-aligned choice |

Both NSA and DSA are designed against this table. They differ in *which* rows they prioritize. NSA attacks all five at once — it is trained sparse (rows 2, 3), uses block-level selection with a group-centric kernel (rows 1, 4, 5), and reports backward-pass speedups (row 3). DSA accepts that it is retrofitting a dense model (it concedes row 2 partially, then buys it back with continued training) and focuses on rows 1 and 5 with a token-level indexer that is cheap enough to run over the full sequence.

> The hard lesson of a decade of sparse-attention papers: a sparsity pattern that the GPU can't execute contiguously, or that the model wasn't trained to depend on, is a paper result, not a product. Trainability and hardware alignment are not features you add later — they are the whole game.

Keep that quote in mind, because it is the axis on which NSA and DSA both differentiate themselves from the prior art *and* from each other.

### The cost arithmetic that motivates all of this

Before diving into the two designs, it is worth pinning down the arithmetic, because the whole field exists to bend one curve. Standard attention computes, for a sequence of length `L` and head dimension `d`, a score matrix `S = QKᵀ` of shape `L × L`, a softmax over it, and a value-weighted sum `O = softmax(S)V`. The score matrix alone is `L²` entries, each costing `d` multiply-adds, so the attention compute is `O(L²·d)`. At `L = 65536` and `d = 128`, that is roughly `5.5 × 10¹¹` multiply-adds *per head per layer* — and a frontier model has dozens of layers and dozens of heads. The score matrix is also `L²` numbers of activation memory, which is why FlashAttention exists: it never materializes the full matrix, instead tiling the computation so the `L²` term stays in fast SRAM rather than HBM. But FlashAttention only changes the *constant* and the memory-traffic profile; the work is still fundamentally quadratic in `L`.

Sparse attention attacks the `L²` term itself. If each query attends only `k` keys, the score work drops to `O(L·k·d)`, linear in `L` for fixed `k`. The savings ratio is `L/k`. At `L = 65536`, `k = 2048`, that ratio is `32x` of theoretical score-compute reduction. The reason the *measured* speedups (NSA's 9.0x forward) are smaller than this theoretical `32x` is exactly the gap this whole post is about: selecting and gathering the `k` keys is not free, the softmax and value-sum have their own costs, and the memory-access pattern of the gather can erase much of the FLOP win if it is not contiguous. **The entire engineering contest is about how much of that theoretical `L/k` ratio you can actually keep after the gather overhead and the kernel inefficiencies are paid.**

There is a second, distinct cost that matters at least as much in production: the KV cache. During decoding, every new token must attend to all prior keys and values, which means streaming the entire KV cache out of HBM for every single generated token. This is bandwidth-bound, not compute-bound — the FLOPs of decoding one token are trivial, but moving the KV cache dominates. A `L = 65536`, 60-layer model with MLA still streams gigabytes per token. Sparse attention helps here too: if you only attend `k` keys, you only need to stream `k` KV entries (plus whatever the selector needs to read), so decode bandwidth drops by roughly `L/k` as well. NSA's 11.6x decode speedup is mostly this bandwidth reduction, which is why it is *larger* than its forward speedup — decode is bandwidth-bound and sparse attention is a bandwidth win. This split — compute savings in prefill/training, bandwidth savings in decode — is why both papers report multiple numbers, and why a single "speedup" figure would be misleading.

> Attention has two cost curves, not one: the `O(L²)` compute curve that hurts during prefill and training, and the `O(L)`-per-token bandwidth curve that hurts during decode. A sparse method's value is the area it carves out of *both*. Read every speedup number as an answer to "which curve, and at what context length?"

## 1. NSA: three branches, one learned gate

**The senior rule of thumb here is that no single sparse pattern is right for everything a token needs, so don't pick one — run several and let the model learn the mix.** NSA's central architectural decision is to decompose attention into three parallel branches, each capturing a different *kind* of relevance, and to blend them per token with a learned gate.

![NSA routes a query through three parallel branches — compression with an MLP over gist blocks, blockwise top-n selection reusing compression scores, and a sliding window of recent tokens — then combines them with per-branch sigmoid gate scores into the attention output](/imgs/blogs/trainable-sparse-attention-nsa-vs-dsa-2.webp)

The three branches, as shown above, are:

1. **Compression (global view).** A learnable MLP `φ` aggregates contiguous blocks of keys and values into single coarse "gist" tokens. With compression block length `l = 32` and a sliding stride `d = 16`, every block of 32 KV positions is mapped to one compressed key/value pair (the MLP carries an intra-block position encoding so it knows *where* in the block each token sat). The query attends over these compressed tokens, getting a cheap, blurry view of the *entire* sequence. This branch never misses anything globally, but it sees everything at low resolution.

2. **Selection (salient detail).** The compression branch gave us a softmax over compressed blocks — and that softmax is, conveniently, a per-block importance score. The selection branch *reuses those exact scores* to rank blocks, picks the top `n` (in the paper, `n = 16`, including a fixed first block and the two local blocks adjacent to the query), and then attends over **all** the original tokens inside those selected blocks at full resolution. The selection block size is `l' = 64`. The crucial design choice is the *granularity*: NSA selects whole blocks, not individual tokens, so the kept keys are contiguous in memory — exactly what a Tensor Core wants.

3. **Sliding window (local context).** A simple window of the most recent `w = 512` tokens, always kept. Recent tokens dominate next-token prediction, and giving them their own branch is not just for accuracy — it is to stop a pathology. If you fold local context into the compression and selection branches, the model can shortcut-learn to satisfy its loss almost entirely from local tokens, and the global branches never get a useful gradient. Isolating the window forces the other two branches to earn their gradients on genuinely long-range dependencies.

The three branch outputs are combined by a **learned gate**. For each branch `c`, a small MLP followed by a sigmoid produces a gate score `g_t^c ∈ [0, 1]` from the query's input features, and the final output is the gate-weighted sum of the three branch attentions. Because the gate is differentiable and trained end-to-end, the model learns, per token, how much to trust the global gist versus the selected detail versus the local window. A token deep in a code block might lean hard on the sliding window; a token summarizing a long document might lean on compression.

Here is the shape of the forward pass in pseudocode, which makes the reuse of compression scores explicit:

```python
def nsa_attention(q_t, K, V, gate_mlp,
                  l=32, d=16, l_sel=64, n_sel=16, w=512):
    # Branch 1: compression — MLP aggregates contiguous KV blocks into gists.
    K_cmp, V_cmp = compress_blocks(K, V, block=l, stride=d, mlp=phi)
    attn_cmp, scores_cmp = attend(q_t, K_cmp, V_cmp, return_scores=True)

    # Branch 2: selection — REUSE compression scores to rank blocks, keep top-n,
    #           then attend over ALL tokens inside the selected blocks.
    block_importance = remap_to_selection_blocks(scores_cmp, l_sel)
    top_blocks = topk(block_importance, n_sel)          # block granularity
    K_sel, V_sel = gather_blocks(K, V, top_blocks)      # contiguous in memory
    attn_sel = attend(q_t, K_sel, V_sel)

    # Branch 3: sliding window — recent w tokens, isolated.
    attn_win = attend(q_t, K[-w:], V[-w:])

    # Learned gate — per-branch sigmoid score in [0, 1], trained end-to-end.
    g = sigmoid(gate_mlp(q_t))                          # shape [3]
    return g[0] * attn_cmp + g[1] * attn_sel + g[2] * attn_win
```

The single most economical idea in the whole design is the score reuse on line "remap_to_selection_blocks". The selection branch needs to know which blocks matter, and computing that from scratch would mean a second full attention pass. Instead it piggybacks on the compression branch's softmax, which was going to be computed anyway. The compression branch is the scout; the selection branch acts on the scout's report.

Let me make the compression math concrete, because the `l = 32`, `d = 16` numbers are not arbitrary. The compression branch slides a window of length `l = 32` over the key sequence with stride `d = 16`, so consecutive compressed tokens *overlap* by 16 positions. Why overlap? A non-overlapping compression (stride equal to block length) creates hard boundaries — a dependency that straddles two blocks gets split, and the model has no compressed token that sees both halves. The 50% overlap (stride = half the block length) means every original position is covered by two compressed tokens, so boundary-straddling patterns are still captured by at least one gist. For a sequence of `L` positions, the number of compressed tokens is roughly `(L - l)/d + 1 ≈ L/16`, so the compression branch's attention is over `~L/16` tokens — a 16x reduction in the global-view attention size relative to dense. The MLP `φ` that produces each compressed key is a small learnable network applied to the 32 keys in the window plus their intra-block positions; it learns *what* to summarize, rather than a fixed pooling like a mean (which would throw away which token in the block mattered).

Now the gate. The gate MLP takes the query's hidden state `h_t` and produces three logits, one per branch, each passed through a sigmoid to land in `[0, 1]`. Note these are independent sigmoids, not a softmax — the gate is not forced to allocate a fixed budget across branches; it can turn all three on, all three down, or anything between. This matters because the three branches are not mutually exclusive views of the same thing: a token can legitimately need strong global context *and* strong local context *and* a couple of salient distant blocks all at once. A softmax gate would force a trade-off the model does not actually face. The independent sigmoids let the gate express "I need all of these" or "I only need the window here," and because it is trained end to end, the learned gate values become an interpretable readout of *what kind of context each token depends on.*

A worked example of the selection budget: with selection block size `l' = 64` and `n = 16` selected blocks, the selection branch attends `64 × 16 = 1024` tokens at full resolution. Add the `w = 512` sliding window and the `~L/16` compressed tokens, and at `L = 32768` the total NSA attention footprint per query is roughly `1024 + 512 + 2048 = 3584` tokens — about 11% of the full 32K context. At `L = 8192` it is `1024 + 512 + 512 = 2048` tokens, or 25%. The pattern is the same as DSA's: the kept fraction shrinks as context grows, because the selection and window budgets are fixed while only the compressed-token count grows (slowly, at `L/16`). That `L/16` term is the one piece of NSA's footprint that is not strictly bounded — but it grows 16x slower than dense, and it is the cheap blurry branch, so it is the right place to let cost scale.

### Second-order optimization: why block granularity beats token granularity

The non-obvious gotcha that sinks naive sparse-attention implementations is that **the granularity of selection is a hardware decision disguised as an accuracy decision.** Your instinct is that finer is better — select individual tokens and you waste no budget on irrelevant neighbors. But a top-k *token* selection produces a scattered set of memory addresses, and reading scattered addresses on a GPU means either many small loads (latency-bound) or a gather kernel that defeats the contiguous-tile assumption FlashAttention is built on. Tensor Cores reach peak throughput on dense, contiguous tiles. By selecting whole blocks of `l' = 64` contiguous tokens, NSA keeps every fetched region dense, so the matmul that follows runs at Tensor-Core speed. You trade a little selection precision for a large constant-factor speedup, and because the model is trained with this granularity, it learns to make blocks meaningful rather than fighting the constraint. This is the difference between a sparse method that is theoretically `9x` cheaper and one that is *measured* `9x` faster.

## 2. DSA: index, select, attend

**The senior rule of thumb that motivates DSA is the opposite of NSA's: if you already have a frontier-quality dense model, the cheapest path to long-context efficiency is not to retrain it — it is to teach a tiny side network where the model would have looked anyway, then attend only there.** DSA is built as the *only* architectural change between DeepSeek-V3.1-Terminus (a 128K-context model on the V3 MoE + MLA stack) and DeepSeek-V3.2. Everything else about the model is held fixed; sparse attention is the single variable.

![DSA inference flows left to right across two rows: a query plus all L keys enter a lightning indexer (FP8, ReLU, few heads) that produces an index score, keys are ranked per query, the top-K equals 2048 keys are selected, MLA attention runs over only those selected KV pairs, producing the output at O(L x k) cost](/imgs/blogs/trainable-sparse-attention-nsa-vs-dsa-3.webp)

The workflow above has three stages:

1. **Index.** A **lightning indexer** — a deliberately lightweight network with a small number of heads, computed in **FP8**, using a ReLU activation — scores every preceding token `s` against the current query `t`. The index score has the form `I(t,s) = Σ_j w(t,j) · ReLU(q(t,j) · k(s))`, summed over the indexer's few heads `j`. This is a separate, cheap attention-like computation whose only job is to produce a relevance ranking, not the actual attention output.

2. **Select.** For each query, take the **top-K = 2,048** key-value tokens by index score. This is *token-level* selection — DSA does not bucket into blocks the way NSA does. Two thousand and forty-eight is a fixed budget regardless of how long `L` is.

3. **Attend.** Run the real MLA attention — the expensive part — over only those 2,048 selected KV pairs. Because the heavy attention now touches `k = 2048` instead of `L`, the **core attention complexity drops from `O(L²)` to `O(Lk)`** with `k ≪ L`.

There is an honest caveat that DSA does not hide: the **lightning indexer itself is still `O(L²)`**, because it scores every query against every key. The win is that the indexer is *dramatically cheaper per operation* than full MLA attention — it has few heads, runs in FP8, and produces a scalar score rather than a full value-weighted output. So the expensive `O(L²)` MLA computation is replaced by a cheap `O(L²)` index plus an `O(Lk)` real attention. At long context the `O(Lk)` term dominates the savings, and the cheap quadratic indexer is a rounding error against the dense MLA it replaced.

Here is the inference path in pseudocode, with the FP8 indexer made explicit:

```python
def dsa_attention(q_t, K, V, indexer, top_k=2048):
    # Stage 1: lightning indexer — cheap FP8 scoring of every prior token.
    #          Few heads, ReLU; produces a relevance scalar per key, not output.
    idx_scores = indexer.score(q_t, K)          # O(L) per query, O(L^2) total
    # idx_scores[s] = sum_j w[t,j] * relu(qI[t,j] . kI[s])

    # Stage 2: select top-K most relevant keys for THIS query.
    sel = topk(idx_scores, top_k)               # 2048 token indices

    # Stage 3: the expensive MLA attention runs over the selection only.
    K_sel, V_sel = K[sel], V[sel]
    return mla_attend(q_t, K_sel, V_sel)        # O(L * k), k = 2048 << L
```

Notice what is *not* here: there is no learned gate, no three branches, no compression MLP. DSA is structurally far simpler than NSA. It is one selection mechanism (the indexer) feeding one attention. That simplicity is the point — it is what makes DSA a low-risk change to drop into a model that is already in production serving paying customers. DSA is instantiated on the MQA mode of MLA, which means all query heads share the selected KV set, so the selection is computed once per query position and reused across heads — the same fetch-once-reuse-many principle NSA gets from its group-centric kernel, but obtained for free from MLA's structure rather than from a custom kernel.

### Why the indexer is shaped the way it is

Every design choice in the lightning indexer is in service of one constraint: it has to be cheap enough that running it over the *full* `O(L²)` sequence still leaves you ahead of dense attention. Walk through the choices:

- **Few heads.** A full attention head is expensive; the indexer needs only enough heads to produce a reliable relevance ranking, not a high-fidelity output. So it uses a small number of heads — far fewer than the main model — which directly cuts its compute by the head-count ratio.
- **FP8.** The indexer's scores are only used for a top-K *ranking*, not for a value-weighted sum that feeds the residual stream. Ranking is robust to the reduced precision of FP8 — you need the right *order*, not exact magnitudes — so the indexer runs in 8-bit floating point, roughly halving its compute and memory traffic versus BF16. This is a precision-allocation decision: spend bits where they affect the output (the main MLA attention), starve the part that only produces a ranking. (The FP8 discipline here is the same family of idea as the [DeepSeek-V3 FP8 training recipe](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing).)
- **ReLU activation.** The index score `I(t,s) = Σ_j w(t,j) · ReLU(q(t,j) · k(s))` uses ReLU rather than the exponential of a softmax. ReLU is far cheaper than `exp`, and again, because the output is a ranking and not a normalized distribution, you do not need the softmax's normalization properties. ReLU also gives a clean "this token is irrelevant" signal — anything with a non-positive dot product contributes zero — which is exactly the kind of hard cutoff a selector wants.

The result is an indexer whose per-operation cost is a small fraction of a real attention head's, which is what makes its `O(L²)` affordable. The honest framing: DSA did not eliminate the quadratic term, it made the quadratic term *cheap* and moved the expensive linear-in-`L` work behind a top-K filter. For long contexts, a cheap `O(L²)` plus an expensive `O(Lk)` beats an expensive `O(L²)`, and the crossover happens well below 128K.

A useful way to see the indexer's role: it is a *learned, content-dependent attention mask*. Classic sparse attention used *fixed* masks — strided, local, block-sparse patterns chosen by the architect, the same for every input. The indexer replaces the fixed mask with a tiny network that proposes a *different* mask per query based on content. That is the deep reason DSA can match dense quality where fixed-pattern sparse attention degraded it: the pattern adapts to the input rather than being imposed on it.

### Second-order optimization: token-level selection is affordable here because MLA already shrank the KV

The gotcha that makes DSA's token-level selection viable — where NSA had to retreat to block granularity for hardware reasons — is that **MLA has already compressed the KV cache into a low-rank latent.** In a vanilla multi-head model, gathering 2,048 scattered tokens means gathering 2,048 full-width KV vectors across many heads, and the scattered memory access hurts. But under MLA's MQA mode, each position's KV is a single shared latent vector, so a token-level gather of 2,048 latents is far less memory traffic than the equivalent gather in a dense multi-head model. The compression DSA inherits from MLA is precisely what lets it afford the per-token precision that NSA had to give up. This is a clean example of why DSA had to be built on the MLA stack and not on a generic transformer — the two optimizations compose.

## 3. Native vs retrofit: the philosophical fork

**The rule of thumb for reading these two papers side by side: NSA is what you build when you control the model from scratch and want sparsity baked into its weights; DSA is what you build when you have a finished model and a deployment deadline.** This is the core contrast, and it shapes every downstream decision.

![A before-after comparison: on the left NSA native from scratch — pretrain sparse from step 0 over 270B tokens, three branches and a learned gate trained end to end, a backward pass that is sparse too for a 6.0x train speedup, a research artifact paying the design cost up front; on the right DSA retrofit — start from trained V3.1-Terminus at 128K context, add only a lightning indexer plus top-K selection, continued training over about 946B tokens to adapt, a production drop-in that shipped a roughly 50 percent or more API price cut](/imgs/blogs/trainable-sparse-attention-nsa-vs-dsa-4.webp)

The before-after above lays out the two life cycles. NSA's left column: you pretrain sparse from step zero (the paper trains on 270B tokens), the three branches and gate learn together end to end, and critically the *backward pass is sparse too* — which is why NSA can report a training speedup and not just an inference one. The cost is that you committed to this architecture before you knew the model would be good; you paid the design and validation cost up front, on a research budget.

DSA's right column: you start from a known-good model, V3.1-Terminus, that already has frontier quality at 128K context. You add the smallest possible new component — the indexer — and a single top-K selection. Then you run *continued* training (roughly 946B tokens total across two stages, which we will dissect next) to let the model adapt to its keys being filtered. The payoff is that this is a low-risk, deployable change. DeepSeek shipped it and cut their API prices by more than half on the back of the efficiency, and a high-compute variant (V3.2-Speciale) reached gold-medal scores on the 2025 IMO (35/42) and IOI (492/600), demonstrating the retrofit didn't cost reasoning quality.

Put bluntly: **NSA changes the model's DNA; DSA changes the model's diet.** NSA's sparsity is in the weights from birth. DSA's sparsity is a behavior the model is taught after the fact. Both are "trainable" — neither is the old inference-only pruning — but the *when* of the training is the whole difference. NSA trains the sparse pattern and the model jointly; DSA trains a pattern-picker and then adapts a pre-existing model to live with it.

A useful way to feel the difference: picture removing the sparsity from each. From an NSA model you cannot — the three branches and the gate are load-bearing parameters; there is no "dense version" hiding inside. From a DSA model you almost can — detach the indexer and let the model attend densely, and you are back to something very close to V3.1-Terminus, because that is literally what it was trained from. NSA is sparse all the way down; DSA is a dense model wearing a sparsity adapter.

### Where these sit in the lineage of sparse attention

It helps to place both methods against the prior art, because each is a deliberate reaction to a specific failure of earlier work. Sparse attention has roughly four generations:

1. **Fixed-pattern sparse attention** (Sparse Transformer, Longformer, BigBird). These impose a hand-designed mask — strided, local-window, or block-sparse with a few global tokens — identical for every input. They were genuinely sub-quadratic and trainable, but the mask is content-independent, so they paid an accuracy tax: the architect, not the data, decided what each token could see. Both NSA and DSA can be read as "fixed-pattern sparse attention, but the pattern is learned and content-dependent." NSA's compression-derived block scores and DSA's indexer both replace the fixed mask with an input-adaptive one.

2. **Inference-time KV eviction and selection** (H2O, StreamingLLM, and the whole family of cache-compression heuristics). These keep the model dense at training time and prune the KV cache at serving time using a heuristic — recency, accumulated attention mass, attention sinks. They are cheap to adopt (no retraining) but they are approximations of a model that was never trained for them, so they degrade on exactly the long-context tasks you wanted. This is the regime NSA and DSA were built to escape; the word "trainable" in the title is the whole distinction. (The KV-eviction techniques are surveyed in the [KV cache optimization guide](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).)

3. **Linear and low-rank attention** (Performers, linear attention, state-space hybrids). These change the attention *math* to avoid the `L²` matrix entirely, achieving true linear complexity. They are a different bet — they replace softmax attention rather than sparsifying it — and they have historically traded some expressiveness. NSA and DSA are *not* in this camp; they keep exact softmax attention over a selected subset, so the attention each query does compute is the real thing, just over fewer keys.

4. **Trainable, hardware-aligned sparse selection** — where NSA and DSA live. The defining commitment is that the sparse pattern is (a) learned and content-dependent, (b) trained into the model rather than imposed at inference, and (c) co-designed with the memory hierarchy so the savings are real. NSA and DSA are two points in this fourth generation that differ on *when* the training happens (from scratch vs continued) and *how* selection is granularized (blocks vs tokens). That they were published by the same lab eleven months apart, with opposite postures, is the clearest signal that the field has converged on "trainable + hardware-aligned" as the requirement and is now exploring the design space *within* that requirement.

Reading them this way clarifies what is genuinely novel. Neither invented learned sparse selection or content-dependent masks in the abstract — what they contributed is making the selection cheap enough and the kernel fast enough that the method survives contact with a real GPU and a real training run, plus, in NSA's case, the demonstration that you can train end-to-end through the selection and get a backward-pass speedup out of it.

## 4. What top-K selection actually keeps

**The rule of thumb that makes selection intuitive: a query does not need a smooth gradient of attention over the whole past — it needs a handful of genuinely relevant regions plus its immediate neighborhood, and everything else can be dropped to exactly zero cost.** Selection is where the `O(Lk)` savings are physically realized, so it is worth seeing concretely what survives the cut.

![A grid showing token selection on a sequence: a query scores each KV block and keeps only the top-n; across eight blocks, block 0 is a kept anchor, blocks 2, 5, and 7 are kept with high scores 0.31, 0.28, 0.41, while blocks 1, 3, 4, 6 are dropped with low scores; block importance scores are reused from the compression branch with no extra pass; a local sliding window of w equals 512 recent tokens is always kept; dropped blocks cost zero FLOPs and are never loaded](/imgs/blogs/trainable-sparse-attention-nsa-vs-dsa-5.webp)

The grid above shows NSA's selection on a toy eight-block sequence. The query scores each block (scores reused from compression, so no extra pass), keeps the top-scoring ones plus a fixed anchor block and the local window, and *drops the rest entirely.* The dropped blocks are not down-weighted — they are never loaded into SRAM and never contribute a single FLOP. That is the difference between sparse attention and a low softmax weight: a low weight still cost you the multiply; a dropped block cost you nothing.

Two design choices in this picture deserve emphasis:

- **The fixed anchor (block 0).** The first block is always kept. In autoregressive transformers, the initial tokens act as attention sinks — the model parks excess attention mass on them — and dropping them destabilizes the distribution. Keeping block 0 unconditionally is cheap insurance against that pathology.
- **The local window is separate from selection.** The recent `w = 512` tokens are kept by the sliding-window branch regardless of their selection scores. This guarantees the model never loses its immediate context even if the selection scores happen to favor distant blocks.

DSA's version of this picture differs in granularity: where NSA keeps whole blocks, DSA keeps the top-2,048 *individual* tokens by index score. There is no fixed block structure; the indexer is free to pick a scattered set if that is what the query wants. The trade is precision versus contiguity — DSA gets per-token precision (afforded by MLA's compressed KV, as discussed) while NSA gets contiguous, Tensor-Core-friendly blocks. Neither is strictly better; they are tuned to their respective hardware paths.

A worked example makes the savings concrete. Take a 64K-token context (`L = 65536`).

- **Dense MLA** scores all 65,536 keys per query. The per-query attention work scales with `L`.
- **DSA** scores all 65,536 keys *cheaply* in the FP8 indexer, then runs the expensive MLA over only `k = 2048` of them. The heavy attention sees `2048 / 65536 ≈ 3.1%` of the keys.
- At 128K context (`L = 131072`), DSA's heavy attention still sees only 2,048 keys — now `~1.6%` — because `k` is a fixed budget. **This is why the savings grow with context length: `k` is constant, so the kept fraction shrinks as `L` grows.**

That last point is the whole economic argument for sparse attention in long-context serving. The longer the context your customers use, the more dramatic the relative savings, because you are paying for a fixed 2,048-token attention regardless of whether the document is 8K or 128K tokens long.

## 5. DSA's two-stage training: teach the picker, then adapt the model

**The senior rule of thumb for retrofitting any selection mechanism onto a trained model: never let the random-initialized selector and the pretrained model fight each other on day one — freeze the model, train the selector to imitate the behavior you already have, and only then let them co-adapt.** DSA's continued-training recipe is a textbook execution of this, and it is the part most worth stealing for your own retrofit projects.

![A timeline of DSA two-stage continued training: starting from the base model V3.1-Terminus at 128K context, a dense warm-up stage at learning rate 1e-3 freezes everything but the indexer and KL-aligns it to the main attention, reaching indexer ready after about 1000 steps where the indexer mimics the dense attention distribution; then a sparse stage at learning rate 7.3e-6 with top-K equals 2048 and the indexer input detached, ending with the DSA model shipped after about 15000 steps and about 943.7B tokens](/imgs/blogs/trainable-sparse-attention-nsa-vs-dsa-6.webp)

The timeline above shows the two stages.

**Stage 1 — dense warm-up.** Freeze the entire main model. Train *only* the lightning indexer. The objective is a KL divergence that aligns the indexer's score distribution to the main model's actual attention distribution — in other words, teach the indexer to predict which tokens the dense model already attends to. This runs at a high learning rate (`1e-3`, appropriate because only a small fresh module is learning), for roughly **1,000 steps** consuming about **2.1B tokens.** Crucially, attention is still *dense* during this stage — the model attends to everything, and the indexer is just learning to mimic where that attention lands. By the end, the indexer can predict the dense model's attention pattern well enough to be trusted as a selector.

**Stage 2 — sparse training.** Now switch on the sparsity: attention runs over the indexer's top-2,048 selection. Unfreeze the model and let everything co-adapt at a much lower learning rate (`7.3e-6`, gentle because you are nudging a frontier model, not training one from scratch), for roughly **15,000 steps** over about **943.7B tokens.** One subtle but important detail: the **indexer's input is detached from the computational graph** in this stage. The indexer learns from its own KL-style target, and the main model's gradients do not flow back into the indexer's inputs — this prevents the model from corrupting the indexer's learned scoring to make its own loss easier, which would defeat the purpose of having an honest relevance signal.

Here is the recipe as a config sketch:

```yaml
dsa_continued_training:
  base_model: deepseek-v3.1-terminus   # 128K context, MoE + MLA
  stage_1_dense_warmup:
    trainable: [lightning_indexer]     # everything else FROZEN
    attention_mode: dense              # model still attends to all keys
    objective: kl_align_indexer_to_attention
    learning_rate: 1.0e-3              # high; only a small module learns
    steps: ~1000
    tokens: ~2.1e9                     # 2.1B
  stage_2_sparse:
    trainable: [full_model, lightning_indexer]
    attention_mode: sparse_topk
    top_k: 2048
    detach_indexer_input: true         # model can't corrupt the selector
    learning_rate: 7.3e-6              # low; gentle co-adaptation
    steps: ~15000
    tokens: ~9.437e11                  # 943.7B
```

Two things to notice about the *arithmetic* of this recipe. First, the asymmetry: stage 1 is `~2.1B` tokens, stage 2 is `~943.7B` tokens — the warm-up is barely 0.2% of the total. The expensive part is co-adapting the model to sparsity, not training the indexer. Second, the total `~946B` tokens of continued training is a tiny fraction of the trillions of tokens that pretrained V3.1-Terminus. **That ratio is the entire value proposition of the retrofit: you reuse essentially all of the pretraining investment and pay a sub-1% continued-training surcharge to make the model sparse.** NSA, by contrast, paid for its sparsity across the full 270B-token training run because there was no dense model to inherit from.

### Second-order optimization: the detach is the part everyone gets wrong

The non-obvious failure mode when you skip the detach: if the main model's loss gradient is allowed to flow back through the indexer's inputs, the model discovers it can lower its own loss not by attending better but by *manipulating which tokens the indexer selects* — it learns to make the indexer pick the tokens that happen to make the current batch's loss small, rather than the tokens that are genuinely relevant. The selection signal collapses into a shortcut. Detaching the indexer input forces the indexer to keep scoring relevance honestly against its KL target, so the selection stays meaningful out of distribution. This is the same family of bug as reward hacking — give a learned component a way to influence its own training signal and it will exploit it. NSA sidesteps this entirely because its selection scores are *reused from* the compression attention rather than learned by a separate detachable module; the gradient path is different by construction.

## 6. The hardware story: why NSA's speedups are real

**The rule of thumb that separates a sparse-attention paper from a sparse-attention product: a method only counts if its FLOP savings show up on a wall clock, and that only happens when the memory-access pattern is co-designed with the GPU.** NSA's headline contribution is arguably not the three branches but the *kernel* that makes them fast on real silicon. This is the row of the "why it's different" table that most prior work failed.

![A graph of NSA's GQA group-centric kernel: query heads 0, 1, and 2 all in group g feed a group loader that loads all heads of g at position t with a single fetch, which loads a sparse KV block once into contiguous SRAM, and that block is reused by every head in the group with no refetch](/imgs/blogs/trainable-sparse-attention-nsa-vs-dsa-8.webp)

The kernel design above is the GQA group-centric loop. In grouped-query attention, several query heads share one KV group. A naive sparse kernel would, for each head, gather that head's selected blocks — refetching the same KV block once per head in the group. NSA's kernel inverts the loop: it loads **all the query heads in a group together**, fetches each selected sparse KV block **once** into contiguous SRAM, and **reuses that block across every head in the group.** The fetched block is contiguous (because selection is block-level, not token-level — there is the payoff for that earlier granularity decision), so the load is a clean burst, not a scatter.

This is implemented in Triton, and it is what turns the theoretical sparsity into measured speed. The two pillars are:

- **Group-centric data loading** — fetch once per group, reuse across heads, so a model with, say, 4 query heads per KV group does roughly a quarter of the KV memory traffic a head-centric loop would.
- **Blockwise contiguous access** — selected blocks are contiguous spans of memory, so loads are coalesced and the subsequent matmul runs on dense tiles at Tensor-Core throughput, instead of being throttled by gather latency.

There is a deeper reason the group-centric loop matters specifically for *sparse* attention. In dense FlashAttention, every query block streams every key block in order, so there is no choice about ordering — you walk the keys sequentially and the access is naturally contiguous. In sparse attention, each query's selected blocks are a *different subset*, so a head-centric loop would walk a different scattered set of blocks per head and re-pay the load for any block shared between heads in a group. Because GQA deliberately makes heads in a group share KV, the overlap is large — often total — so the head-centric loop wastes most of its bandwidth re-reading the same blocks. The group-centric inversion is the natural fix once you notice that the *sharing* GQA introduced for memory reasons is exactly the sharing the sparse kernel should exploit. NSA's kernel is, in a sense, the sparse-attention completion of the GQA idea: GQA shares KV across heads to shrink the cache; the group-centric kernel shares the *fetched sparse block* across heads to shrink the read traffic.

The arithmetic: with `G` query heads per KV group, the head-centric loop issues up to `G` loads of each selected block (one per head), while the group-centric loop issues one. For a model with `G = 4`, that is a 4x reduction in the worst-case sparse-block read traffic, which compounds with the `L/k` reduction from sparsity itself. The two multiply: sparsity cuts *how many* blocks you read, and the group-centric loop cuts *how many times* you read each. This is why NSA can report an 11.6x decode speedup that exceeds its 9.0x forward speedup — decode is the regime where read traffic dominates, so the group-centric saving shows up most.

Compare this to DSA's hardware story, which is quieter precisely because it leans on MLA. DSA's token-level selection would normally be a gather nightmare, but because it runs in the MQA mode of MLA, all heads already share a single compressed latent per position, so selecting 2,048 positions selects 2,048 latents — and the heavy attention over those is cheap because MLA's latent is low-rank. DSA gets its memory efficiency from the architecture it sits on; NSA earns its memory efficiency in a hand-written kernel. Same destination, opposite engineering style: DSA composes with an existing optimization, NSA builds a new one.

This is the most underappreciated difference between the two and worth stating plainly: **NSA's hardware alignment is a contribution it makes; DSA's hardware alignment is a contribution it inherits.** If you adopt NSA you take on the maintenance of a specialized Triton kernel that has to track your hardware and your GQA group size. If you adopt DSA you take on essentially no new kernel surface, because the heavy attention is still MLA's existing kernel, just over a gathered subset. For a team without a kernel engineer, that difference alone can decide the choice — DSA is far less to own.

> If you remember one sentence about NSA's kernel: it loads each sparse KV block exactly once per group and reuses it across all the heads that share the group, which is the only reason the FLOP savings survive contact with HBM bandwidth.

## 7. The numbers, side by side

**The rule of thumb when comparing two efficiency methods: read what each one chose to *measure*, because the choice of headline metric tells you what the authors were optimizing for.** NSA reports kernel speedups for forward, backward, *and* decode — a training-and-inference story. DSA reports a deployable budget and a price cut — a serving story. The metrics are not directly comparable, which is itself the finding.

![A matrix comparing NSA from arXiv 2502.11089 against DSA in V3.2 across seven rows: sparsity unit is block-level three branches plus gate for NSA versus per-token top-K indexer for DSA; selection budget is top-n equals 16 blocks of l prime equals 64 each for NSA versus top-K equals 2048 KV tokens for DSA; forward at 64k is 9.0x faster for NSA versus O(L x k) core with O(L^2) indexer for DSA; backward at 64k is 6.0x faster and trainable for NSA with the DSA cell empty; decode at 64k is 11.6x faster for NSA versus a long-context serve win for DSA; quality is LongBench plus 0.032 versus full for NSA versus matches V3.1 on benchmarks for DSA; framing is research from scratch for NSA versus production retrofit for DSA](/imgs/blogs/trainable-sparse-attention-nsa-vs-dsa-7.webp)

The matrix above is the at-a-glance comparison. Let me unpack the load-bearing numbers.

**NSA speedups at 64K context** (the differentiating measurements):

- **Forward pass: 9.0x** faster than full attention.
- **Backward pass: 6.0x** faster. *This is the number that proves NSA is trainable-sparse, not inference-sparse.* No prior sparse-attention method that was bolted on at inference time can report a backward speedup, because there is no sparse backward pass to speed up — the model was trained dense. NSA's backward is sparse because its forward is sparse by construction.
- **Decoding: 11.6x** faster. Decode is memory-bandwidth-bound (one token at a time, the FLOPs are tiny relative to the KV streaming), so the speedup here reflects the reduced KV traffic from the group-centric kernel.

**NSA quality:** on a 27B-parameter MoE model with 3B active parameters, trained on 270B tokens and extended from 8K to 32K context, NSA *matched or beat* full attention on general benchmarks and scored **+0.032 average on LongBench** (0.469 versus full attention's 0.437). That a sparse model *beats* the dense baseline on long-context tasks is the counterintuitive result — and the explanation is that the sparse structure acts as a useful inductive bias, focusing the model on relevant tokens rather than letting it dilute attention across noise.

**DSA specs and quality:** the headline is the **top-K = 2048** budget and the `O(L²) → O(Lk)` core-attention complexity reduction. DSA was validated to **match V3.1-Terminus on benchmarks** — the point was efficiency at parity, not a quality jump. The efficiency translated into a **>50% API price cut**, and the high-compute V3.2-Speciale variant hit IMO 2025 35/42 and IOI 2025 492/600, confirming the sparse retrofit preserved frontier reasoning.

Note the empty cell in the matrix: DSA does not report a backward-pass speedup. That is not an oversight — it reflects that DSA's primary value is *serving* a long-context model cheaply, and its continued-training cost was a fixed one-time `~946B`-token investment, not a per-run training acceleration. NSA, designed for training models from scratch, cares deeply about the backward pass; DSA, designed to ship a cheaper API, cares about decode and prefill economics. **The blank cell is the philosophy in a single missing number.**

Here is the same comparison as a decision table you can actually use:

| Dimension | NSA | DSA |
|---|---|---|
| When sparsity is introduced | Pretraining, step 0 | Continued training on a finished model |
| Selection mechanism | 3 branches (compress / select / window) + learned gate | 1 lightning indexer + top-K |
| Selection granularity | Block-level (`l' = 64`) | Token-level (top-2048) |
| Selection score source | Reused from compression attention | Separate trained indexer (FP8) |
| Backward-pass speedup | Yes (6.0x @ 64K) | Not the goal |
| Risk profile | High (new architecture, new kernel) | Low (single module on proven model) |
| Sits on | Co-designed Triton GQA kernel | MLA's compressed KV (MQA mode) |
| Best for | Training new long-context models | Retrofitting an existing dense model |

## Case studies from the trade-offs

These are not production war stories — both methods are recent — but they are the concrete decision scenarios where the NSA-vs-DSA choice actually bites. Each is the kind of situation a senior engineer will face when reaching for trainable sparse attention.

### 1. The greenfield long-context pretrain

You are training a new model from scratch and you know long context is a first-class requirement — say, a code model that must hold an entire repository. The wrong first hypothesis is "train it dense, sparsify at inference." That gives you a model trained to attend densely and then crippled at serving time, and it does nothing for your *training* bill, which at long context is dominated by the `O(L²)` backward pass. The right move is NSA: bake the three-branch structure in from step zero, get the 6.0x backward speedup that makes long-context pretraining affordable, and ship a model that is sparse by construction. The lesson: if you control pretraining and long context is core, the backward-pass speedup alone justifies the architectural cost.

### 2. The production model you cannot afford to retrain

You have a frontier model already serving paying customers at 128K context, and your CFO wants the inference bill cut. Retraining from scratch with NSA is off the table — you would be re-validating a brand-new architecture against your entire eval suite, on a multi-month timeline, betting the model's quality. This is exactly DSA's scenario: add the indexer, run `~946B` tokens of continued training (sub-1% of pretraining), validate at parity, ship the price cut. DeepSeek's own `>50%` API price reduction is the proof. The lesson: when the model is already good and the constraint is deployment risk, retrofit beats rebuild.

### 3. The "we tried top-k per token and it got slower" trap

A team implements naive sparse attention by having each head gather its top-k tokens, benchmarks it, and finds it is *slower* than dense FlashAttention. The wrong conclusion is "sparse attention doesn't work." The root cause is that per-head, per-token gathers fragment memory access and defeat the contiguous-tile assumption. The fix is one of the two real designs: NSA's block-level selection with a group-centric kernel (contiguous fetches, reused across heads), or DSA's token-level selection layered on MLA (where the shared latent makes the gather cheap). The lesson: sparse attention that ignores memory layout is reliably slower than dense; the layout *is* the method.

### 4. The dropped attention sink

A team sparsifies attention and sees training instability — loss spikes, occasional divergence — that did not exist in the dense model. The wrong hypothesis is "the learning rate is too high." The actual cause is that the selection sometimes drops the initial tokens, which were serving as attention sinks soaking up excess attention mass; without them the softmax distribution destabilizes. NSA's fix is structural: always keep block 0 as a fixed anchor. The lesson: a few positions in every sequence are load-bearing for stability, not relevance, and a selection mechanism must protect them unconditionally.

### 5. The indexer that learned to cheat

A team builds a DSA-style retrofit but skips the gradient detach on the indexer input. Training loss looks great, but evaluation on held-out long-context tasks is mediocre and gets worse over time. The wrong hypothesis is "the indexer is too small." The root cause is that the main model's gradient flowed into the indexer's inputs, and the model learned to bias the selector toward whatever tokens minimized the current batch's loss — a reward-hacking shortcut that collapses the relevance signal. The fix is DSA's detach: cut the gradient so the indexer keeps scoring honestly against its own KL target. The lesson: any learned selector that can influence its own training signal will exploit it; isolate it.

### 6. The local-context shortcut

A team designs a multi-branch sparse attention but folds the recent tokens into the same branches as the global ones. The model trains, but the global/long-range branches never seem to contribute — ablating them barely moves the loss. The root cause is that the model satisfied its next-token objective almost entirely from local context, so the long-range branches got vanishing gradients and never learned anything useful. NSA's fix is to *isolate* the sliding window as its own branch with its own gate, forcing the compression and selection branches to earn their gradients on genuinely long-range dependencies. The lesson: if you want a branch to learn long-range behavior, do not let an easier local path satisfy the loss first.

### 7. The context-length scaling surprise

A team measures DSA's speedup at 8K context, finds it modest, and concludes the method is not worth it. The wrong move is to evaluate a fixed-budget sparse method at short context. Because `k = 2048` is constant, at 8K the kept fraction is `25%` — not much savings — but at 128K it is `~1.6%`, a `>10x` reduction in the heavy attention's workload. The lesson: fixed-budget sparse attention is an asymptotic win; benchmark it at the context lengths your product actually uses, not at toy lengths where the budget is a large fraction of the sequence.

### 8. The MLA dependency

A team likes DSA's simplicity and tries to port the lightning-indexer-plus-top-K design onto a vanilla multi-head model without MLA. The token-level gather of 2,048 full-width multi-head KV vectors turns out to be a memory-traffic disaster, and the method underperforms NSA's block approach. The root cause is that DSA's token-level precision is *only* affordable because MLA already compressed KV into a shared low-rank latent. The lesson: DSA and MLA are co-designed; the token-level selection is borrowing MLA's compression to pay for its precision. On a non-MLA stack, NSA's block-level approach is the more portable choice. (See the MLA deep-dive for why the compressed latent makes this composition work.)

### 9. The warm-up that was too short

A team retrofitting a DSA-style indexer runs the dense warm-up for only a few hundred steps to save compute, then switches on sparsity. The sparse stage's loss is unstable and the model's long-context quality never recovers to dense parity. The wrong hypothesis is "the sparse learning rate is wrong." The actual root cause is that the indexer had not yet converged to mimic the dense attention distribution when sparsity was turned on, so it was selecting partly-random tokens; the model then spent the expensive sparse stage adapting to a *moving, noisy* selection target instead of a stable one. DeepSeek's recipe spends ~1,000 steps and ~2.1B tokens on warm-up precisely so the indexer is a reliable selector before the model has to trust it. The lesson: the cheap stage that trains the selector is not the place to economize — a half-trained selector poisons the expensive stage that follows it.

### 10. Benchmarking the indexer's quadratic term in isolation

An engineer profiles a DSA implementation, sees the lightning indexer's `O(L²)` cost, and flags it as the bottleneck to optimize first. The wrong move is to pour engineering into the indexer's quadratic term. Profiling the *full* attention path shows the indexer is a small fraction of total attention time — it is FP8, few-headed, and ReLU-based, so its per-element cost is a fraction of the MLA attention it gates. The real win was already captured: the expensive MLA work dropped from `O(L²)` to `O(Lk)`. Optimizing the cheap quadratic indexer further has diminishing returns until `L` gets extreme. The lesson: complexity class is not cost; a cheap `O(L²)` can be smaller than an expensive `O(Lk)` until `L` is large, so profile the wall clock before optimizing the asymptote.

### 11. Mixing the two philosophies and getting neither benefit

A team, having read both papers, tries to retrofit *NSA's three-branch structure* onto an existing dense model via continued training — taking NSA's architecture but DSA's deployment posture. The result underwhelms: the gate and compression MLP, randomly initialized, fight the pretrained weights, and the continued-training budget is too small to train three new branches and a gate from scratch while also adapting the model. The root cause is a posture mismatch — NSA's architecture was designed to be learned *jointly with* the model over a full training run, not bolted on with a sub-1% continued-training budget the way DSA's single lightweight indexer can be. The lesson: the architectural complexity of a sparse method has to match its training posture. A heavy, multi-branch architecture wants from-scratch training (NSA); a retrofit wants a single lightweight module (DSA). Crossing them gives you the cost of the heavy architecture without the training budget to make it work.

## Cross-cutting concerns

### Cost and serving economics

The reason any of this exists is the serving bill. For DSA the economics are direct: `O(Lk)` heavy attention with a fixed `k = 2048` means the cost of attending to a 128K context is essentially flat past the point where `L` exceeds `k` by a healthy margin, which is what let DeepSeek halve their API prices. For NSA the economics are split across training and serving — the 6.0x backward speedup cuts the *training* bill for long-context models, and the 11.6x decode speedup cuts the serving bill. If you are an API provider, DSA's prefill-and-decode story maps most directly to your invoice; if you are training models, NSA's backward speedup is the line item that moves. For the broader picture of where attention sits in the inference cost stack, see the [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) guide.

### Quality and the inductive-bias surprise

The result that should reshape your intuition is that NSA *beat* full attention on LongBench (+0.032). Sparse attention is usually framed as a quality-for-speed trade, but a well-designed trainable sparse pattern can be a *quality improvement*, because forcing the model to commit attention to a selected subset is a useful inductive bias against attention dilution over long, noisy contexts. DSA's framing is more conservative — parity with the dense model — because its goal was to not regress a frontier model, not to chase a new high. Both data points say the same thing: trainable sparse attention is not the lossy approximation that inference-time pruning was.

### Multi-tenancy and the fixed budget

A subtle operational benefit of DSA's fixed `k = 2048` is predictability: every query's heavy attention does the same amount of work regardless of context length, which makes capacity planning and batching far more uniform than dense attention's length-dependent cost. NSA's per-token gate makes its cost slightly more variable (different tokens lean on different branches), but the block budget keeps it bounded. If you are sizing a serving cluster, a fixed-budget sparse method is easier to provision against than a dense model whose cost balloons with the longest request in the batch.

### Why this matters most for agents

The workload that makes both methods urgent rather than nice-to-have is the long-running agent. A coding agent or a research agent accumulates context relentlessly — tool outputs, file contents, intermediate reasoning, prior turns — and a single session can run for hours and tens of thousands of tokens. Under dense attention, every new agent step re-attends the entire accumulated history, so the per-step cost grows linearly within a step and the whole session's cost grows roughly quadratically in its length. This is the regime where the `L²` term stops being a tail case and becomes the median request. Sparse attention with a fixed budget changes the economics qualitatively: an agent at turn 200 with a 100K-token history still does a fixed `k`-token attention per query, so the per-step cost stops growing once the history exceeds the budget. The session cost goes from quadratic-in-length to linear, which is the difference between an agent product that is affordable to run and one that is not. DSA's flat cost past `L = k` is the more directly relevant property here, which is part of why it shipped inside a production model rather than staying a research result — agents were the workload bending the cost curve, and a fixed-budget sparse attention is the most direct answer to it.

### Numerical stability under FP8

One cross-cutting risk that both methods touch is precision. DSA runs its indexer in FP8, and NSA's parent stack trains in FP8 (per the V3 recipe). The shared hazard is that FP8's narrow dynamic range can underflow or overflow on attention scores, which span a wide range before softmax. DSA sidesteps the worst of this by using FP8 *only* for the indexer's ranking — where exact magnitudes do not matter, only order — and keeping the value-weighted MLA attention in higher precision. This is the right precision-allocation discipline: put the low precision where the output is a discrete decision (which 2,048 tokens to keep), and keep higher precision where the output flows continuously into the residual stream. A team that naively ran the *main* attention in FP8 to chase more speed would likely see quality regressions that the indexer's FP8 never causes, because the indexer's FP8 errors get washed out by the top-K's discreteness while the main attention's FP8 errors propagate. The lesson generalizes beyond sparse attention: FP8 is safe exactly where the consumer of the number is a comparison, and risky where it is an accumulation.

## When to reach for NSA / when to reach for DSA

**Reach for NSA when:**

- You control pretraining and long context is a first-class requirement, so you can afford to bake sparsity into the architecture from step zero.
- Your training bill is dominated by the long-context backward pass — the 6.0x backward speedup is the line item that pays for the design cost.
- You want a model that is sparse by construction with no dense fallback, e.g. for a fixed long-context product where you will never want dense attention back.
- You are willing to write and validate a custom GQA group-centric kernel (or adopt one), because the hardware co-design is non-optional for the speedups to be real.
- You want the option of a quality *gain* from the sparse inductive bias, not merely parity.

**Reach for DSA when:**

- You already have a frontier-quality dense model in production and the constraint is deployment risk, not training budget.
- You are on the MLA / MQA stack, so the token-level selection's gather is cheap (this is close to a hard prerequisite).
- You want the minimal-surface-area change — one indexer module, one top-K — that you can validate at parity and ship without re-running your whole eval matrix.
- Your headline goal is serving economics at long context (a price cut, flatter capacity planning), not a faster training run.
- You want to keep a near-dense fallback in your pocket, since detaching the indexer recovers something close to the original model.

**Skip trainable sparse attention entirely when:**

- Your context lengths are short (a few thousand tokens). A fixed `k = 2048` budget saves nothing when `L` is barely larger than `k`, and the dense FlashAttention path is simpler and likely faster.
- You cannot retrain or continue-training the model at all. Both NSA and DSA are *trainable* methods; if you can only touch inference, you are back to the old approximation regime that this entire generation of work was built to escape, and you should not expect either method's results.
- You are not on a stack that supports the required memory layout — no GQA/group structure for NSA's kernel, no MLA for DSA's cheap gather. Porting either method to an incompatible stack tends to erase the speedup.
- Quality at any cost is the only metric and you have unlimited compute. Dense attention remains the conservative default; sparse attention is an efficiency play (that sometimes also helps quality, but you would adopt it for the efficiency).

## Closing: two answers to one question

NSA and DSA are answering the same question — *how do you make sparse attention trainable instead of a lossy inference hack?* — and arriving at deliberately opposite engineering postures. NSA says: control the model from birth, decompose attention into three learnable branches, blend them with a learned gate, and co-design a GPU kernel so the savings are real in both the forward and backward pass. You pay the full design cost up front and you get a model that is sparse in its bones, with a backward-pass speedup no retrofit can match and a quality bump from the inductive bias. DSA says: the model is already good and already deployed, so do not touch its weights more than you must — train a tiny FP8 indexer to predict where the model already looks, select the top-2,048, and continue-train for under 1% of the original budget to lock it in. You get a low-risk, shippable change that halved an API bill.

The right answer depends entirely on which side of the "do I control pretraining?" line you stand on. Research labs building new long-context foundations will want NSA's native sparsity and its backward-pass economics. Teams operating a frontier model in production, under deployment risk and on the MLA stack, will want DSA's surgical retrofit. **NSA is sparse by construction; DSA is sparse by adaptation. Pick the one that matches whether your sparsity needs to be in the weights or just in the serving path.**

## Further reading

- [DeepSeek-V3 training recipe: FP8, MTP, and loss-free balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — the MoE + FP8 + MTP foundation both NSA and DSA sit on.
- The MLA deep-dive (Multi-head Latent Attention) — the compressed-KV attention that DSA is layered over and that makes its token-level selection affordable.
- [KV cache: a complete guide](/blog/machine-learning/large-language-model/kv-cache) and [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) — the memory mechanics that sparse attention is ultimately optimizing.
- [DeepSeek-V3.2 paper reading](/blog/paper-reading/large-language-model/deepseek-v3-2) — a focused walk-through of the V3.2 report itself, complementary to this technique-contrast angle.
- Native Sparse Attention, arXiv 2502.11089 (February 2025) — the NSA paper, including the Triton kernel and the 64K speedup measurements.
- DeepSeek-V3.2, arXiv 2512.02556 (December 2025) — the DSA paper, including the lightning indexer and the two-stage continued-training recipe.

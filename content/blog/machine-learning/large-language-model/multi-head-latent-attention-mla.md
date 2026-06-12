---
title: "Multi-head Latent Attention: How DeepSeek Made the KV Cache 14x Smaller Without Losing Quality"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A principal-engineer deep-dive on Multi-head Latent Attention (MLA) from DeepSeek-V2 and V3: low-rank joint KV compression to a 512-dim latent, the matrix-absorption trick that lets you attend without materializing K/V, decoupled RoPE, and why the bottleneck improves quality instead of merely saving memory."
tags: ["llm", "deepseek-v2", "deepseek-v3", "multi-head-latent-attention", "mla", "kv-cache", "attention", "rope", "inference-optimization", "low-rank", "transformer-architecture", "long-context"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

The dominant cost of serving a long-context language model is not the matrix multiplies. It is the **KV cache** — the per-token key and value tensors you keep around so that each new token can attend to every token before it. For a model with 128 attention heads and a 128-dimensional head, storing both K and V for a single token costs `2 x 128 x 128 = 32768` numbers. Multiply by a 128K context, by every layer, by every concurrent request, and the cache is what fills your HBM, throttles your batch size, and caps your tokens-per-second. Everyone who has tried to serve a large model at long context knows this wall.

Multi-head Latent Attention (MLA), introduced in the **DeepSeek-V2** technical report (arXiv 2405.04434) and carried unchanged into **DeepSeek-V3** (arXiv 2412.19437), is the cleverest answer the field has produced. It cuts the KV cache by **93.3%** versus DeepSeek's own dense 67B model, lifts maximum generation throughput **5.76x**, and — this is the part that should make you sit up — does so while *improving* downstream quality relative to plain Multi-Head Attention. The compression is not a lossy concession you pay for speed. It is a low-rank inductive constraint that happens to be good for the model.

This post is a principal-engineer teardown of MLA: not "here is the equation from the paper," but "here is the reusable idea, here is why each detail is load-bearing, and here is where you would reach for it or avoid it." If you have read [the V3 training deep-dive](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing), this is the architectural companion — that post covered FP8, Multi-Token Prediction, and loss-free balancing; this one covers the attention block those models are built on. It pairs naturally with the [KV cache primer](/blog/machine-learning/large-language-model/kv-cache) and the broader survey of [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

![MLA mental model: a token down-projects into a 512-dim latent that is cached, plus a 64-dim decoupled-RoPE key, and per-head K and V are reconstructed on the fly.](/imgs/blogs/multi-head-latent-attention-mla-1.webp)

The diagram above is the mental model for the whole article, and the rest is a tour of it. Read it left to right: a token's hidden vector `h` (dimension 7168 in V3) splits into two paths. The main path runs through a **down-projection** into a single low-rank latent `c_KV` of dimension 512 — and *this 512-dim vector is the only KV state you cache*. The second, tiny path produces a 64-dim **decoupled RoPE key** that carries position information. At attention time you **up-project** the cached latent back into per-head keys and values on the fly, concatenate the decoupled RoPE key, and attend. The whole trick is that the up-projection can be folded — *absorbed* — into the query and output weights, so in practice you never even materialize the per-head K and V. You attend directly against the 512-dim latent.

## Why the KV cache is the real bottleneck

> Senior rule of thumb: at long context the model is not compute-bound during decode — it is memory-bandwidth-bound on the KV cache. Shrink the bytes you read per token and you move the whole throughput frontier.

Take a transformer decoder at inference. Prefill (processing the prompt) is compute-heavy: it is a big batched matmul over the whole prompt, and it is bounded by FLOPs. Decode (generating tokens one at a time) is different. Each step you produce exactly one new token, and to compute its attention you must read the keys and values of *every previous token* out of HBM. The arithmetic intensity of that read is terrible: you stream a large tensor through the chip and do a tiny amount of math per byte. Decode is therefore **memory-bandwidth-bound**, and the tensor you are streaming is the KV cache.

The size of that cache is the product of four numbers: the number of layers, the number of tokens in context, the bytes per element, and — the one MLA attacks — the **per-token, per-layer footprint of K and V**. In standard Multi-Head Attention (MHA) that footprint is `2 x n_h x d_h` elements, where `n_h` is the number of heads and `d_h` is the per-head dimension. For DeepSeek-V2/V3's `n_h = 128`, `d_h = 128`, that is `2 x 128 x 128 = 32768` elements per token per layer. Across 60–61 layers and a long context, the cache dwarfs the model weights you have to keep resident for a single request.

| Common assumption | The naive view | The reality with MLA |
|---|---|---|
| "Bigger context just needs more RAM" | Buy more HBM, scale linearly | The cache read per token is the throughput limiter; you must shrink it, not just store more |
| "KV compression always loses quality" | Fewer K/V dims = worse attention | A low-rank latent bottleneck *improves* quality vs full MHA at equal active params |
| "GQA already solved this" | Share K/V across head groups | GQA shares heads; MLA compresses to a latent — denser at equal cache, and absorbable |
| "Compression means decompressing every step" | Up-project K/V on every decode token | Matrix absorption means you never materialize K/V — you attend on the 512-d latent directly |
| "RoPE is free to add anywhere" | Rotate the keys, done | RoPE's position-dependence breaks absorption; MLA isolates it in a 64-d decoupled key |
| "MLA is a memory hack bolted onto MHA" | A serving-time trick | It is a *training-time architecture*: the model learns the compression and never sees full per-head K/V |

The two prior art answers were Multi-Query Attention (MQA) and Grouped-Query Attention (GQA). MQA shares a single K and V across all heads — maximal cache savings, but a real quality hit because every head is forced to read the same key/value subspace. GQA is the compromise that won: partition the `n_h` query heads into `g` groups, each group sharing one K/V head. With `g = 8` you cut the cache `128/8 = 16x` and lose only a little quality. Most frontier models (Llama, Qwen, Gemma) ship GQA. MLA asks a sharper question: *instead of sharing whole heads, what if we compress the information all heads need into a shared low-rank latent, and reconstruct each head's view from it?*

## 1. Low-rank joint KV compression: the core idea

> Senior rule of thumb: if a tensor you are forced to cache has far more dimensions than the information it actually carries, cache the information, not the tensor. A low-rank bottleneck is a learned compressor you get for free at training time.

The starting observation is that the per-head keys and values across 128 heads are not 128 independent things. They are 128 linear views of the same underlying hidden state, and those views are highly redundant — the rank of the stacked K (or V) across heads is far below `n_h x d_h`. MLA makes that redundancy explicit and learnable.

Concretely, for each token with hidden state `h ∈ R^d` (with `d = 7168` in V3), MLA computes a single **compressed latent**:

```python
import torch
import torch.nn as nn

class JointKVCompression(nn.Module):
    """Down-project the hidden state into one shared low-rank latent c_KV.

    d     : model hidden size (7168 in DeepSeek-V3)
    d_c   : KV compression dim, the ONLY thing we cache (512 = 4 * d_h)
    """
    def __init__(self, d=7168, d_c=512):
        super().__init__()
        self.W_DKV = nn.Linear(d, d_c, bias=False)   # down-projection

    def forward(self, h):                            # h: [batch, seq, d]
        c_KV = self.W_DKV(h)                          # [batch, seq, 512]
        return c_KV                                  # cache THIS, nothing else
```

That `c_KV` of dimension `d_c = 512` is the entire KV state for the token. Note the chosen size: `512 = 4 x d_h`, where `d_h = 128` is the head dimension. So the latent is exactly four head-widths wide. At attention time you up-project it back into the per-head keys and values:

```python
class KVUpProjection(nn.Module):
    """Reconstruct per-head K and V from the cached latent.

    n_h : number of heads (128)
    d_h : per-head dim (128)
    """
    def __init__(self, d_c=512, n_h=128, d_h=128):
        super().__init__()
        self.W_UK = nn.Linear(d_c, n_h * d_h, bias=False)  # latent -> keys
        self.W_UV = nn.Linear(d_c, n_h * d_h, bias=False)  # latent -> values

    def forward(self, c_KV):                       # c_KV: [b, s, 512]
        b, s, _ = c_KV.shape
        k = self.W_UK(c_KV).view(b, s, 128, 128)   # per-head keys
        v = self.W_UV(c_KV).view(b, s, 128, 128)   # per-head values
        return k, v
```

This is a textbook low-rank factorization of the K and V projections. In plain MHA you would have `W_K, W_V ∈ R^{d x (n_h·d_h)}` mapping the hidden state straight to all the heads' keys/values. MLA replaces each with a product: `W_DKV` (down to 512) followed by `W_UK` / `W_UV` (up to `n_h·d_h`). The rank of the composed map is capped at 512. That is the constraint — and it is the point.

The compression ratio is stark. Per token, MHA caches `2 x 128 x 128 = 32768` elements. MLA caches one 512-dim latent (`W_UK` and `W_UV` are *weights*, not cached state — they are shared across all tokens and live in the model, not the cache). That is a 64x reduction from the latent alone, before we add back the small decoupled-RoPE key. The "~14x" in the title comes from comparing total realized cache against DeepSeek's prior dense 67B GQA-style baseline, and the paper's headline number is the **93.3%** reduction versus DeepSeek-67B.

### Why a bottleneck can *help* quality

Here is the counterintuitive bit. You would expect forcing K and V through a rank-512 bottleneck to *cost* quality — you are throwing away representational capacity. But the V2 ablations show MLA matching or beating full MHA at equal activated parameters. Two effects are at work.

First, **the bottleneck is a regularizer.** Full per-head K/V projections are heavily over-parameterized; the true information the attention needs is low-rank, and a model with `n_h x d_h` free dimensions per token will happily fit noise in the extra capacity. Constraining to rank 512 is a structural prior that says "the keys and values live on a low-dimensional manifold," which is true, and which steers the model toward the signal.

Second, **the saved memory buys back quality elsewhere.** Because the cache is 14x smaller, you can afford a wider model, a longer context, or a bigger batch at the same hardware budget. DeepSeek spent that headroom on a genuinely large MoE. The comparison "MLA vs MHA at equal *cache budget*" is wildly in MLA's favor, and the comparison "MLA vs MHA at equal *parameters*" is roughly a wash or slightly better. Either way you win.

> The best compression scheme is one the model learns to live inside. MLA does not compress the cache after the fact — it trains the model to never need the uncompressed form.

### The SVD intuition: why rank 512 is not arbitrary

It helps to ground the bottleneck dimension in a concrete linear-algebra picture. Stack the per-head key projections of an ordinary MHA layer into one big matrix `W_K ∈ R^{d x (n_h·d_h)}` — for V3 that is `7168 x 16384`. Take its singular value decomposition. In a trained MHA model, the singular values of `W_K` decay fast: a few hundred of them carry almost all the energy, and the long tail past a few hundred is near-noise. That spectral decay is precisely the statement "the keys all 128 heads need live on a low-dimensional manifold." If you truncated the SVD of a trained MHA model's `W_K` to its top 512 singular directions, you would lose very little.

MLA does not truncate a pretrained matrix; it *parameterizes the truncation up front* and trains inside it. `W_DKV` (the `d → 512` down-projection) and `W_UK` (the `512 → 16384` up-projection) are exactly a learned rank-512 factorization `W_K ≈ W_UK · W_DKV`. The model never explores the full-rank space, so it never wastes capacity on the near-noise tail. This is why the bottleneck is a regularizer and not a handicap: it bans a region of parameter space that a well-trained model would have left empty anyway. The choice of 512 is the engineer's bet on where the spectrum's "knee" sits — wide enough to capture the signal across 128 heads, narrow enough that the cache is tiny. Four head-widths turned out to be the right bet.

There is a sharper way to say this. In MHA the *effective* per-token KV information is bounded by `rank(W_K) ≤ min(d, n_h·d_h)`, but it is also bounded by the rank of the activations passing through — and that rank is empirically low. MHA pays to cache `n_h·d_h` numbers to represent information that occupies far fewer effective dimensions. MLA stops paying for the gap. The "joint" in *joint KV compression* is the other half: K and V are compressed through the *same* latent `c_KV`, exploiting the fact that the keys and values a token contributes are themselves correlated — they are two views of the same content. Sharing one latent for both is a second compression on top of the low-rank one.

### What the model actually learns in the latent

A natural worry: if all 128 heads reconstruct their keys from one shared 512-dim latent, do the heads collapse into copies of each other? They do not, and the reason is `W_UK`. Each head gets its own `128 x 512` slice of the up-projection, so each head reads a *different linear view* of the same latent. The latent is a shared substrate; the per-head up-projection is the lens. Two heads can extract orthogonal information from the same `c_KV` because their slices of `W_UK` point in different directions. This is the crucial difference from GQA, where heads in a group literally share the same key vector. In MLA the cache is shared but the per-head views are not — you get head diversity at shared-cache cost, which is the whole reason quality holds.

## 2. The decoupled-RoPE problem and its fix

![Before/after: putting RoPE on the compressed key injects a position-dependent rotation that blocks matrix absorption, so MLA routes position through a separate 64-dim key instead.](/imgs/blogs/multi-head-latent-attention-mla-5.webp)

> Senior rule of thumb: any position-dependent transform that sits between two weight matrices you wanted to multiply together will block you from pre-multiplying them. Isolate position into its own narrow channel and the rest of the path stays algebraically collapsible.

Everything above is clean until you remember that real models use **Rotary Position Embeddings (RoPE)**. RoPE injects position by rotating the query and key vectors by an angle that depends on the token's absolute position before computing the dot product. The rotation is what gives attention its relative-position sense. And it is exactly what breaks the low-rank story.

To see why, look ahead to the absorption trick (next section). The whole efficiency win of MLA at decode is that you can fold the up-projection `W_UK` into the query weight `W_Q`, so the attention score is computed directly against the cached latent `c_KV` without ever building the per-head key `k = W_UK c_KV`. That fold is a matrix product `W_UK^T W_Q`, computed once, offline. It works because both are *constant* linear maps.

Now insert RoPE. The actual key the query must dot against is `RoPE_t(W_UK c_KV)` — the up-projected key, rotated by the position-`t` rotation matrix. That rotation depends on `t`, the token's position, which changes every step and differs per cached token. You cannot pre-multiply `W_UK^T W_Q` through a rotation that is not fixed at compile time. The absorption collapses. You would be forced to rebuild the full per-head key for every cached token on every decode step — which destroys the entire memory and bandwidth win. RoPE on the compressed key turns MLA back into something no better than MHA.

DeepSeek's fix is **decoupled RoPE**, and it is the single most elegant detail in the design. Split the key (and query) into two parts:

- A **compressed part** `k^C = W_UK c_KV`, which carries content and stays *rotation-free*, so `W_UK` remains absorbable.
- A **decoupled RoPE part** `k^R`, a separate small key of dimension `d_h^R = 64`, produced by its own projection, which *does* carry RoPE.

The attention score is then the sum of two terms: the content score `q^C · k^C` (computed on the absorbable latent path) plus the position score `q^R · k^R` (computed on the tiny 64-dim rotated path). You concatenate, effectively, and the softmax sees both. Crucially, the RoPE-bearing `k^R` is *shared across heads* like a single extra key channel, so the extra cache it adds is only 64 elements per token — half a head-width — not `n_h x 64`.

```python
import torch, torch.nn as nn

def rope(x, pos):
    # apply rotary embedding to the last dim of x at absolute positions pos
    # (standard RoPE; elided for brevity — rotates pairs of dims by angle pos*theta)
    ...
    return x

class DecoupledRoPEKey(nn.Module):
    """Produce the small, position-bearing key that rides alongside the latent.

    d_h_R : decoupled RoPE dim (64). The ONLY part that carries position.
    """
    def __init__(self, d=7168, d_h_R=64):
        super().__init__()
        self.W_KR = nn.Linear(d, d_h_R, bias=False)   # shared across heads

    def forward(self, h, pos):                        # h: [b, s, d]
        k_R = self.W_KR(h)                            # [b, s, 64]
        k_R = rope(k_R, pos)                          # rotate -> carries position
        return k_R                                    # cache alongside c_KV
```

So the full MLA cache per token is `512 (latent) + 64 (decoupled RoPE key) = 576 elements`. The latent path is absorbable; the 64-dim path is the only thing that ever carries a position-dependent rotation, and it is small enough that rebuilding it costs almost nothing. The decoupling confines the un-absorbable cost to half a head-width. That is the whole trick.

### Second-order optimization: why position lives in 64 dims, not 128

A natural question: why `d_h^R = 64` and not the full `d_h = 128`? Because the decoupled key's only job is to carry *position*, not content. Content lives in the rank-512 latent. Sixty-four dimensions of rotary signal is empirically enough to encode relative position across a 128K window — and every dimension you add here is a dimension that must be cached uncompressed *and* recomputed un-absorbed. So you want it as small as it can be while still resolving position. Sixty-four is the sweet spot DeepSeek found. This is the same instinct as choosing the latent dim: spend dimensions where they buy expressiveness, starve the parts that are pure overhead.

### The RoPE algebra, made explicit

It is worth writing out exactly why the rotation does not commute with the absorption, because the failure is subtle and the intuition "RoPE is just another linear map" is the trap. RoPE applies a block-diagonal rotation matrix `R_t` that depends on the token position `t`. For two tokens at positions `m` (query) and `n` (key), the rotated dot product is:

```
(R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
```

The magic of RoPE is that `R_m^T R_n = R_{n-m}` depends only on the *relative* offset `n - m` — that is how RoPE gives attention relative-position awareness without any learned positional table. But notice what this does to absorption. To fold `W_UK` into `W_Q` you needed the score to be `q^T (constant matrix) k`. With RoPE on the key the score is `q^T R_{n-m} (W_UK c_KV)`, and `R_{n-m}` is *not constant* — it changes with the offset between the query token and each cached key token, which differs for every pair. There is no single matrix to pre-fold. The rotation has to be applied per-pair, at runtime, after the up-projection — which forces the up-projection to happen at runtime, which is exactly the cost MLA exists to avoid.

The decoupling sidesteps this by keeping the rotation entirely off the content path. The content score `q^C^T k^C` uses *no* rotation, so its `W_UK` folds cleanly. The position score `q^R^T R_{n-m} k^R` carries all the rotation, but it operates on tiny 64-dim vectors that are never up-projected from a latent in the first place — `k^R` is produced directly by `W_KR` and cached as-is. So the per-pair rotation cost is `O(64)` per cached token, not `O(n_h · d_h) = O(16384)`. The full attention logit is the sum: `q^C^T k^C + q^R^T R_{n-m} k^R`. One term is absorbable and content-rich; the other is tiny and position-bearing. That clean separation is the entire payoff of the decoupling.

A subtle implementation consequence: because the content and position scores are summed *before* the softmax, the model is free to weight position vs content per head, per query, implicitly through the magnitudes it learns for `q^C` and `q^R`. A head that wants to be strongly positional learns large `q^R`; a head that wants to be content-driven learns large `q^C`. The decoupling does not just preserve efficiency — it gives the model a clean knob to trade position against content head-by-head.

## 3. Matrix absorption: attending without materializing K/V

![Graph: because attention is bilinear, W_UK folds into the query weight and W_UV folds into the output weight, so per-head K and V are never built at inference.](/imgs/blogs/multi-head-latent-attention-mla-3.webp)

> Senior rule of thumb: when two linear maps are separated only by other linear maps, multiply them once at load time and never again at runtime. Bilinear forms like attention scores are full of such collapsible products waiting to be found.

This is where MLA stops being "just a low-rank cache" and becomes genuinely fast. The naive reading of section 1 is alarming: if you compress to 512 and then up-project back to `128 x 128` on every decode step, have you not just *added* a matmul to every step? For a moment, yes — but only if you build the per-head K and V explicitly. You should not. The attention math lets you skip materialization entirely.

Look at the content score for a single query head against a single cached token (ignoring RoPE, which lives on the separate path):

```
score = (q^C)^T · k^C
      = (W_Q h_query)^T · (W_UK c_KV)
      = h_query^T · W_Q^T · W_UK · c_KV
      = h_query^T · (W_Q^T W_UK) · c_KV
```

The product `W_Q^T W_UK` is a constant matrix — both factors are fixed weights. Pre-multiply them once, offline, into an **absorbed query weight**. At decode time you compute the query, then dot it directly against the 512-dim `c_KV` you read from the cache. You never build the `128 x 128` per-head key. The up-projection `W_UK` has been *absorbed into the query path*.

The same trick works on the value side. The attention output is `out = W_O · (sum_t a_t · v_t)`, where `a_t` are the softmax weights and `v_t = W_UV c_KV(t)` are the up-projected per-head values. Substitute:

```
out = W_O · sum_t a_t · (W_UV c_KV(t))
    = W_O · W_UV · sum_t a_t · c_KV(t)
    = (W_O W_UV) · sum_t a_t · c_KV(t)
```

Again, `W_O W_UV` is a constant product you fold once. You attend over the *latents* directly — taking the softmax-weighted sum of the 512-dim `c_KV` vectors — and apply the single absorbed output map at the end. The per-head value `v_t` is never materialized either. `W_UV` has been *absorbed into the output path*.

The net effect is profound: at inference, MLA touches the cache only as 512-dim latents, runs the attention dot products and value-weighting in that 512-dim space, and applies two absorbed weight matrices that were precomputed at load. There is no 128x128-per-head reconstruction in the hot loop. The 14x smaller cache is also a 14x smaller thing-to-read, and the absorption ensures you do not pay it back in extra FLOPs.

```python
class AbsorbedMLAInference(nn.Module):
    """Inference-time MLA: weights pre-folded, attention runs on the 512-d latent.

    Built ONCE at load from the trained W_Q, W_UK, W_O, W_UV.
    """
    def __init__(self, W_Q, W_UK, W_O, W_UV, n_h=128, d_h=128):
        super().__init__()
        # absorb W_UK into the query weight (per head); shapes elided
        self.WQ_absorbed = (W_Q.transpose(-1, -2) @ W_UK)   # query -> latent space
        # absorb W_UV into the output weight
        self.WO_absorbed = (W_O @ W_UV)                     # latent -> output
        self.n_h, self.d_h = n_h, d_h

    def forward(self, h_query, c_KV_cache, k_R_cache, q_R):
        # content scores: query mapped into latent space, dotted on the cache
        q_lat   = h_query @ self.WQ_absorbed                # no per-head K built
        s_cont  = q_lat @ c_KV_cache.transpose(-1, -2)      # over 512-d latents
        # position scores: tiny 64-d decoupled-RoPE path
        s_pos   = q_R @ k_R_cache.transpose(-1, -2)         # over 64-d keys
        attn    = torch.softmax(s_cont + s_pos, dim=-1)
        ctx_lat = attn @ c_KV_cache                         # weighted latents
        return ctx_lat @ self.WO_absorbed.transpose(-1, -2) # one absorbed output map
```

### Second-order optimization: absorption is a training/inference asymmetry

A subtlety worth internalizing: you do **not** absorb during training. At training time you want the explicit `W_UK`, `W_UV` factors as separate parameters so gradients flow through both the down- and up-projections and the model actually learns the low-rank structure. The absorption is purely an *inference-time refactor* of frozen weights — algebraically identical, numerically reorganized for speed. This train-explicit, serve-absorbed asymmetry is a pattern you will see again (it is the same shape as folding BatchNorm into the preceding conv at inference): keep the factors separate where you need gradients, collapse them where you need throughput.

## 4. The footprint, quantified: MHA vs GQA vs MLA

![Before/after stacks: full MHA caches K and V for all 128 heads per token; MLA caches one 512-dim latent plus a 64-dim RoPE key.](/imgs/blogs/multi-head-latent-attention-mla-2.webp)

> Senior rule of thumb: always reduce a "compression scheme" to elements-per-token-per-layer. That single number is what fills your HBM and sets your max batch size; everything else is commentary.

Let us put concrete numbers on the three schemes for the DeepSeek-V3 shape (`n_h = 128`, `d_h = 128`). Per token, per layer, in elements:

- **MHA:** caches K and V for all 128 heads. `2 x 128 x 128 = 32768` elements.
- **GQA (8 groups):** caches K and V for 8 shared K/V heads. `2 x 8 x 128 = 2048` elements.
- **MLA:** caches one 512-dim latent plus one 64-dim decoupled-RoPE key. `512 + 64 = 576` elements.

The before/after figure above shows the per-token arithmetic for MHA versus MLA directly: two stacks of `128 x 128` collapsing into a single 512-element latent plus a 64-element key. The reduction from MHA is `32768 / 576 ≈ 56.9x` in raw element count. The reduction DeepSeek headlines — **93.3%** versus DeepSeek-67B — is against a model that was not full MHA, which is why the realized factor in the title is ~14x against that baseline rather than ~57x. The exact multiple depends on what you compare against; the *shape* of the win does not.

![Matrix comparing cached K, cached V, elements per token, and equivalence for MHA, GQA, and MLA — MLA's 576 elements land near a 2.25-group GQA.](/imgs/blogs/multi-head-latent-attention-mla-4.webp)

The matrix figure makes the comparison precise and adds the most useful framing in the whole paper: **MLA's 576 elements per token are equivalent, in cache size, to a GQA with about 2.25 groups.** Work it out: a GQA group costs `2 x d_h = 256` elements per token, so `576 / 256 ≈ 2.25 groups`. But — and this is the headline — MLA delivers that 2.25-group *cache footprint* while matching or beating *full 128-head MHA quality*. GQA at 2.25 groups would be a brutal quality regression; almost no one ships fewer than 4–8 groups. MLA gets you sub-GQA cache with super-GQA quality, because it compresses information rather than throwing away heads.

| Scheme | Cached K (elems) | Cached V (elems) | Total / token | Cache vs MHA | Quality at this cache |
|---|---|---|---|---|---|
| MHA, 128 heads | 16384 | 16384 | 32768 | 1x (baseline) | full |
| GQA, 8 groups | 1024 | 1024 | 2048 | 16x smaller | slight loss |
| GQA, 2 groups | 256 | 256 | 512 | 64x smaller | large loss |
| MLA (512 latent + 64 RoPE) | 512 (joint) | (in latent) | 576 | ~57x smaller | ≥ full MHA |

The "(in latent)" cell is the conceptual key: in MLA the cached K and V are *not separate*. They are both reconstructed from the same shared `c_KV`, which is why one 512-dim vector replaces two 16384-element tensors. The joint compression is doing double duty.

### Second-order optimization: the cache is also your context-length budget

Because the per-token footprint is so small, the *number of tokens you can hold in cache at a fixed memory budget* explodes. This is the hidden reason MLA and long context are co-designed. If your cache is 57x smaller per token, you can hold 57x more tokens at the same HBM, or the same tokens across 57x more concurrent requests. The 128K context window in DeepSeek-V2/V3 is not an accident bolted on afterward — it is affordable *because* MLA made each token cheap to remember. We come back to long context in section 6.

### Where MLA sits among the KV-compression family

MLA is one point in a larger design space of KV-cache reduction techniques, and it is worth placing it among its neighbors so you can reason about combining them. The techniques attack different axes:

- **Head sharing (MQA/GQA):** reduce the *number* of K/V heads. Cheap, ubiquitous, but caps quality because heads lose distinct views. MLA dominates this axis by compressing rather than sharing.
- **Low-rank latent (MLA):** reduce the *dimension* of the cached state via a learned bottleneck. Quality-preserving, absorbable, but training-time and dense (less quantization slack).
- **Cross-layer sharing (e.g. YOCO, layer-wise KV reuse):** reduce the *number of layers* that cache by letting upper layers attend to a shared lower-layer cache. Orthogonal to MLA — you could in principle share latents across layers, though DeepSeek does not.
- **Quantization (KV in FP8/INT8/INT4):** reduce the *bytes per element*. Orthogonal to all of the above; stacks with MLA (FP8 latent is standard in the V3 stack).
- **Eviction / sparsity (H2O, sliding window, StreamingLLM):** reduce the *number of tokens* kept. Orthogonal; you can evict latents just as you would evict per-head K/V.

The key reading: MLA occupies the *dimension* axis, and dimension is the axis with the most headroom because the over-parameterization there is the largest. It composes with quantization, eviction, and cross-layer sharing — those attack different axes and multiply. It *substitutes for* head sharing — you would not run MLA and GQA together, because MLA already subsumes what GQA does and does it better. When you evaluate a KV-reduction proposal, the first question is always "which axis does it attack, and does it stack with what I already have?" MLA's answer is "dimension, and it stacks with everything except head-sharing, which it replaces."

| Axis attacked | Technique | Stacks with MLA? | Quality cost |
|---|---|---|---|
| Number of K/V heads | MQA / GQA | No — MLA replaces it | medium–high |
| Cached dimension | MLA (low-rank latent) | (this is MLA) | none / negative |
| Bytes per element | KV quantization | Yes (FP8 safe) | low if FP8 |
| Number of layers caching | cross-layer KV reuse | Yes (orthogonal) | medium |
| Number of tokens kept | eviction / sliding window | Yes (orthogonal) | task-dependent |

## 5. One decode step, end to end

![Pipeline of one MLA decode step: read the 512-dim latent cache, project the query, score on the latent, softmax-weight, output via the absorbed map, and append the new token's 576-element state.](/imgs/blogs/multi-head-latent-attention-mla-6.webp)

> Senior rule of thumb: trace one decode step byte-by-byte. If you can name exactly what is read from HBM, what is computed, and what is written back, you understand the cost model — and MLA's cost model is "read 576 elements per token, attend in 512-d, write 576 back."

Let us walk a single autoregressive decode step on the absorbed inference path, following the pipeline figure. Suppose the model has already generated `N` tokens and is producing token `N+1`.

1. **Read the cache.** Stream `N` latents of 512 dims plus `N` decoupled-RoPE keys of 64 dims out of HBM. That is `N x 576` elements per layer — the entire KV state. In MHA this read would be `N x 32768`. The read is the dominant cost of decode, and MLA has shrunk it ~57x.

2. **Project the query.** From the new token's hidden state, produce the content query `q^C` (via the query down-projection to the query latent of dim `d_c' = 1536`, then up-projection — the query is *also* low-rank compressed, though it is not cached because there is only one query per step) and the decoupled query `q^R` of dim 64 for RoPE. On the absorbed path, `q^C` is mapped straight into the 512-d latent space by the absorbed query weight.

3. **Score on the latent.** Dot the latent-space query against all `N` cached latents to get the content scores, and dot `q^R` against the `N` cached RoPE keys to get the position scores. Sum them. No per-head K is ever built.

4. **Softmax and weight.** Softmax over the `N` summed scores, then take the weighted sum of the `N` cached *latents* (not values — the values are never materialized). This produces a 512-dim context latent.

5. **Output via the absorbed map.** Apply the absorbed output weight `W_O W_UV` to the context latent to get the layer's attention output. No per-head V is ever built.

6. **Append the new token.** Compute the new token's own `c_KV` (512) and `k^R` (64), and append `576` elements to the cache. The cache grows by exactly 576 elements per layer per step — O(1) growth, the smallest of any attention variant that preserves full multi-head expressiveness.

The contrast with a materializing implementation is the whole game. A naive MLA serving stack that up-projects to full per-head K/V on every step would read 576 elements but then *expand* them to `2 x 128 x 128` in registers/SRAM and attend there — burning the compute and the SRAM bandwidth you were trying to save. The absorbed path keeps everything in 512-d. This is why FlashMLA exists.

### FlashMLA: the kernel that makes it real

A clean algebra does nothing if the kernel is slow. DeepSeek open-sourced **FlashMLA**, a decode kernel that serves the absorbed MLA path efficiently on Hopper GPUs. On an H800 it sustains roughly **3000 GB/s** in memory-bound configurations (near the card's HBM ceiling) and up to **580 BF16 TFLOPS** in compute-bound ones. The dual figure is the tell: MLA decode flips between memory-bound (when context is long and you are streaming a lot of latent) and compute-bound (when context is short and the query/output matmuls dominate), and a good kernel must saturate whichever wall it hits. FlashMLA is to MLA what FlashAttention was to MHA — the kernel that turns a paper into throughput. If you want the broader serving picture, the [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) post covers paging, eviction, and prefix reuse that sit on top of whatever attention variant you pick.

### Second-order optimization: MLA changes what speculative decoding costs

A non-obvious downstream effect: because the verification pass over a draft of `k` speculative tokens reads the same tiny cache, [speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) gets cheaper to layer on top of MLA. The bottleneck in spec-decode verification is often the KV read for the appended draft tokens; with MLA each draft token costs 576 elements instead of 32768 to verify against. DeepSeek's Multi-Token Prediction head (covered in the V3 training deep-dive) and MLA are complementary: MTP produces the draft, MLA makes verifying it bandwidth-cheap.

## 6. Long context: stretching only 64 dimensions

![Timeline: 4K pretraining with base RoPE, then YaRN applied only to the 64-dim decoupled-RoPE key, a ~1000-step 32K stage, reaching a 32x 128K context.](/imgs/blogs/multi-head-latent-attention-mla-8.webp)

> Senior rule of thumb: when you extend context, you only need to rescale the part of the model that encodes position. If position lives in a 64-dim subspace, your extension cost is confined to 64 dimensions — the content path never moves.

Context extension in RoPE models is usually done with **YaRN** (or NTK-aware scaling): you rescale the rotary frequencies so the same learned attention patterns generalize to positions far beyond the pretraining window. The cost and risk of YaRN is that it perturbs the position encoding everywhere it lives, and you then need continued training to let the model re-stabilize.

In MLA, position lives in exactly one place: the 64-dim decoupled-RoPE key/query. The content latent `c_KV` carries no position at all. So YaRN is applied **only to that 64-dim subspace.** The content path — the rank-512 latent that holds all the semantic information — is untouched by the extension. This is a beautiful consequence of decoupling: the same architectural choice that preserved matrix absorption *also* localizes the long-context surgery to half a head-width.

The DeepSeek recipe, per the timeline figure: pretrain at 4K context with base rotary keys, then run a short long-context stage — roughly **1000 steps at 32K sequence length with batch size 576** — applying YaRN to the decoupled-RoPE dimension, and the model reaches a **128K** window (a 32x extension over the 4K base). Because only 64 dims are being stretched, the extension is cheap and stable, and the model's content representations never have to relearn anything.

### Second-order optimization: extension is decoupled from the cache size

There is a clean separation of concerns here that is easy to miss. Context *length* is governed by the decoupled-RoPE path (you stretch RoPE to reach further). Context *cost* is governed by the latent path (576 elements per token regardless of how far RoPE reaches). The two are orthogonal. You can push the window to 128K by touching 64 dimensions, and the per-token memory cost does not change at all — it was 576 elements at 4K and it is 576 elements at 128K. Contrast a vanilla model where extending context both stretches RoPE *and* linearly grows an already-large cache. MLA gives you the reach without the proportional memory blow-up.

## 7. The dimension cheat-sheet

![Grid cheat-sheet of MLA dimensions: hidden d, heads, head dim, layers, KV latent 512, query latent 1536, RoPE dim 64, and the 93.3% / 5.76x wins.](/imgs/blogs/multi-head-latent-attention-mla-7.webp)

> Senior rule of thumb: keep the dimension table on a sticky note. Every MLA design decision is "how many head-widths wide should this path be," and the answers are small integer multiples of 128.

The grid figure is the reference card. Here is every number, with the reasoning for the size:

| Symbol | Name | DeepSeek-V2 | DeepSeek-V3 | Reasoning |
|---|---|---|---|---|
| `d` | hidden size | 5120 | 7168 | model width |
| `n_h` | number of heads | 128 | 128 | attention parallelism |
| `d_h` | per-head dim | 128 | 128 | the unit; everything else is a multiple of this |
| layers | depth | 60 | 61 | model depth |
| `d_c` | KV latent dim (cached) | 512 | 512 | `4 x d_h`; the only KV state cached |
| `d_c'` | query latent dim | 1536 | 1536 | `12 x d_h`; compresses Q, not cached (one Q per step) |
| `d_h^R` | decoupled RoPE dim | 64 | 64 | `0.5 x d_h`; carries position only |
| cache/token | per-layer footprint | 576 | 576 | `512 + 64` elements |

A few observations a careful reader should make. First, the KV latent (512) is *smaller* than the query latent (1536). That asymmetry is deliberate: the KV latent is cached per token and read on every step, so you want it tiny; the query latent is computed once per step and never cached, so you can afford it wider for expressiveness. Second, everything is a clean multiple of `d_h = 128`: the latent is 4 head-widths, the query latent is 12, the RoPE key is half a head-width. The head dimension is the quantum of the whole design. Third, the dimensions are *identical* between V2 and V3 — MLA was carried into V3 unchanged. Only the surrounding model (deeper, wider, bigger MoE) grew; the attention block was already right.

The compression ratio of the latent against full per-head K is `512 / (128 x 128) = 512 / 16384 = 1/32`. Combined with the 64-dim RoPE key, the realized total cache reduction against DeepSeek-67B is **93.3%**, and the realized max-generation-throughput gain is **5.76x**. Those two numbers are the bottom line of the entire technique.

## 8. Serving MLA in production: what changes downstream

> Senior rule of thumb: a new attention variant is not done when the math checks out — it is done when paging, tensor parallelism, and quantization all still work. MLA touches all three, mostly for the better, in ways that surprise people the first time.

Adopting MLA changes more than the attention kernel. It ripples through the whole serving stack, and a principal engineer evaluating it needs to know where.

### Paging the latent cache

Modern inference engines (vLLM, SGLang) manage the KV cache in fixed-size *blocks* (pages) rather than as one contiguous tensor per request, so that requests of varying length pack into HBM without fragmentation, and shared prefixes can be deduplicated. PagedAttention was designed around MHA/GQA, where a page holds `block_size x n_kv_heads x d_h` for both K and V. MLA fits this model cleanly but with a different unit: a page holds `block_size x 576` elements (the 512 latent plus the 64 RoPE key per token). Because the per-token footprint is ~57x smaller, a page of the same byte budget holds ~57x more tokens, so block bookkeeping overhead per token drops correspondingly. Prefix caching — reusing the cache of a shared system prompt across requests — works identically: you are sharing latents instead of per-head K/V, and the dedup logic does not care which.

One real subtlety: with MLA the K and V are *jointly* cached in one latent, so you cannot independently evict or quantize K versus V. In MHA stacks that occasionally do asymmetric things to K and V (different precisions, different eviction), MLA forecloses that — the latent is atomic. In practice this is fine; almost no one was doing asymmetric K/V management. But if your stack assumed separable K and V buffers, MLA will require a layout change.

### Tensor parallelism and the latent

Here is where MLA gets genuinely interesting under tensor parallelism (TP). In MHA, you shard the cache across TP ranks by *head*: rank 0 holds heads 0–15's K/V, rank 1 holds heads 16–31's, and so on. Each rank caches its slice. With MLA the cache is a single shared latent `c_KV` that *all heads* up-project from — it is not naturally head-sharded. So under TP, every rank needs the full 512-dim latent to reconstruct its heads' keys/values.

DeepSeek's answer is to **replicate the latent cache across TP ranks** rather than shard it. At first glance replication sounds wasteful — but recall the latent is 57x smaller than full K/V. Replicating a 576-element latent across 8 ranks costs `8 x 576 = 4608` elements per token; full MHA sharded across those same 8 ranks costs `32768` elements total (4096 per rank). MLA's *replicated* cache is still ~7x smaller in aggregate than MHA's *sharded* cache. So you pay a replication factor but win overall by a wide margin, and you gain the simplification that every rank is self-sufficient for the latent. The absorbed weights `W_Q^T W_UK` and `W_O W_UV` are sharded by head as usual.

### Quantizing the latent

The latent cache is a prime candidate for low-precision storage. Because it is the read-dominant tensor at decode, halving its bytes (FP8 or INT8 latent) roughly halves the decode bandwidth on top of MLA's structural win. The wrinkle is that the latent is *information-dense* — it is a rank-512 compression carrying all of K and V — so it is less tolerant of aggressive quantization than a redundant full-rank K/V tensor would be. The rule of thumb that emerged: FP8 the latent is generally safe (the V3 stack already runs FP8 broadly, as covered in the V3 training deep-dive); INT4 the latent risks visible quality loss because there is no redundancy to hide the error in. Compare a full MHA cache, which is so over-parameterized that INT4 barely dents quality — the redundancy MLA removed was also free error-correction. This is the one place where MLA's density works against you: a compressed tensor has less slack for further lossy compression.

| Serving concern | MHA / GQA behavior | MLA behavior |
|---|---|---|
| Paging unit | `block x n_kv_heads x d_h`, K and V separate | `block x 576`, K and V joint in one latent |
| Prefix caching | share per-head K/V | share latents — identical logic, smaller payload |
| Tensor parallelism | shard cache by head | replicate latent (still ~7x smaller in aggregate at TP=8) |
| KV quantization headroom | INT4 often fine (redundant) | FP8 safe, INT4 risky (latent is dense, no slack) |
| Asymmetric K/V handling | possible | foreclosed — latent is atomic |

### Second-order optimization: the prefill path is different from decode

One asymmetry worth flagging. During *prefill* you process the whole prompt at once, and there the absorbed path is not always the win. With many tokens to process in parallel, materializing per-head K/V and running a standard FlashAttention kernel can be more compute-efficient than the absorbed latent path, because prefill is compute-bound, not bandwidth-bound. The absorbed path shines at *decode*, where you read a long cache to produce one token. So a mature MLA serving stack often runs two code paths: materialize-and-FlashAttention for prefill, absorbed-on-latent for decode. FlashMLA targets the decode path specifically. This prefill/decode split is the same disaggregation logic that shows up across modern serving systems — see the [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) post for the broader pattern of specializing kernels per phase.

## 9. Putting it together: the full MLA block

> Senior rule of thumb: read the training-time forward pass once with all factors explicit, then read the inference-time forward pass once with everything absorbed. The gap between them is the whole optimization.

Here is a consolidated training-time forward pass that ties the pieces together. This is the explicit form — every factor separate so gradients flow — not the absorbed serving form.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def rope(x, pos):
    ...  # standard rotary embedding on the last dim
    return x

class MLA(nn.Module):
    """Training-time Multi-head Latent Attention (factors explicit).

    DeepSeek-V3 shape: d=7168, n_h=128, d_h=128, d_c=512, d_cq=1536, d_h_R=64.
    """
    def __init__(self, d=7168, n_h=128, d_h=128, d_c=512, d_cq=1536, d_h_R=64):
        super().__init__()
        self.n_h, self.d_h, self.d_h_R = n_h, d_h, d_h_R
        # KV compression: down to the cached latent, then up to per-head K/V
        self.W_DKV = nn.Linear(d, d_c, bias=False)
        self.W_UK  = nn.Linear(d_c, n_h * d_h, bias=False)
        self.W_UV  = nn.Linear(d_c, n_h * d_h, bias=False)
        # Query compression: down to a query latent, then up to per-head Q
        self.W_DQ  = nn.Linear(d, d_cq, bias=False)
        self.W_UQ  = nn.Linear(d_cq, n_h * d_h, bias=False)
        # Decoupled RoPE: tiny shared key and per-head query, position-bearing
        self.W_KR  = nn.Linear(d, d_h_R, bias=False)
        self.W_QR  = nn.Linear(d_cq, n_h * d_h_R, bias=False)
        self.W_O   = nn.Linear(n_h * d_h, d, bias=False)

    def forward(self, h, pos):                          # h: [b, s, d]
        b, s, _ = h.shape
        # ---- compressed (content) path ----
        c_KV = self.W_DKV(h)                            # [b, s, 512]  <-- CACHE
        k_C  = self.W_UK(c_KV).view(b, s, self.n_h, self.d_h)
        v    = self.W_UV(c_KV).view(b, s, self.n_h, self.d_h)
        c_Q  = self.W_DQ(h)                             # [b, s, 1536] (not cached)
        q_C  = self.W_UQ(c_Q).view(b, s, self.n_h, self.d_h)
        # ---- decoupled RoPE path (64-d, position only) ----
        k_R  = rope(self.W_KR(h), pos)                  # [b, s, 64]   <-- CACHE
        q_R  = rope(self.W_QR(c_Q).view(b, s, self.n_h, self.d_h_R), pos)
        # ---- attention: content score + position score ----
        s_C  = torch.einsum('bshd,bShd->bhsS', q_C, k_C)            # content
        s_R  = torch.einsum('bshr,bSr->bhsS', q_R, k_R)            # position (shared k_R)
        scale = (self.d_h + self.d_h_R) ** -0.5
        attn = F.softmax((s_C + s_R) * scale, dim=-1)
        ctx  = torch.einsum('bhsS,bShd->bshd', attn, v)
        return self.W_O(ctx.reshape(b, s, -1))
```

Two things to notice in this consolidated form. The query is also compressed (`W_DQ` to 1536, then `W_UQ`), which saves activation memory during training even though queries are never cached. And the decoupled RoPE query `q_R` is per-head (`n_h x 64`) while the decoupled key `k_R` is *shared across heads* (just 64) — that asymmetry is why the cache adds only 64 elements, not `128 x 64`. The score combines a content term (over the rank-512-derived keys) and a position term (over the shared 64-dim rotated key). At serving time, you would replace this with the absorbed path from section 3, which keeps everything in 512-d and never builds `k_C` or `v`.

## 10. MLA in the DeepSeek lineage, and what it enabled

> Senior rule of thumb: read an architecture choice by what it *unlocks*, not what it *is*. MLA's value is not "smaller cache" in the abstract — it is the specific other choices that became affordable once the cache stopped being the constraint.

MLA did not arrive in a vacuum. It is one move in a tightly co-designed sequence, and seeing the sequence is how you understand why it was worth the engineering.

**DeepSeek-V2 (May 2024)** introduced MLA alongside DeepSeekMoE (fine-grained experts with shared experts). The pairing is not coincidental. A large MoE has many parameters but activates only a fraction per token, so its *compute* per token is modest — which means its serving bottleneck shifts decisively toward memory and the KV cache. MLA is the attention design that matches a sparse-MoE compute profile: it attacks exactly the resource (cache bandwidth) that a MoE leaves as the limiter. V2 was a 236B-total / 21B-activated model, and the report's headline efficiency numbers — the 93.3% cache reduction and 5.76x throughput versus the dense 67B — are what made that model economical to serve.

**DeepSeek-V3 (December 2024)** scaled the same recipe to 671B total / 37B activated across 61 layers, and kept MLA *byte-for-byte identical*: same `d_c = 512`, same `d_h^R = 64`, same query latent 1536. That an attention block survives a 3x scale-up unchanged is itself a signal — it means the design was not tuned to a particular model size but to a structural truth (K/V are low-rank) that holds at any scale. The V3 report layered FP8 training, Multi-Token Prediction, and auxiliary-loss-free load balancing on top, all of which are covered in the V3 training deep-dive; MLA was the stable foundation underneath them. The throughput MLA bought was also what made V3's inference economics work at frontier scale.

What MLA *unlocked*, concretely:

- **128K context as a default, not a luxury.** Because each token costs 576 cached elements, a 128K window is ~2.2 GB per request (section on worked examples) rather than ~122 GB. Long context stopped being a memory-budget fight.
- **High concurrency on commodity HBM.** The ~70x batch-size headroom (versus MHA) is what lets a serving fleet amortize a huge MoE across many users — the per-user marginal cache cost is tiny.
- **Cheap speculative verification.** As noted, verifying drafted tokens reads the same small cache, so MTP-driven speculative decoding composes cheaply with MLA.
- **A clean long-context training stage.** YaRN on 64 dims is a low-risk, ~1000-step operation rather than a full-model perturbation.

The broader pattern is worth stating plainly. Each DeepSeek technique removes one resource as the binding constraint so the *next* technique can spend it: MLA frees memory bandwidth, MoE spends parameters cheaply, FP8 frees compute, MTP spends the freed compute on a richer training signal and a drafter. MLA sits at the base of that stack because attention cache is the constraint everything else runs into first. If you are studying these models to extract reusable engineering, MLA is the right place to start — it is the load-bearing decision.

### A note on uptraining: converting MHA/GQA models to MLA

Because MLA is a training-time architecture, the obvious question for teams with an existing MHA or GQA checkpoint is whether you can *convert* rather than retrain from scratch. This is an active area, and the recipe that works is a form of distillation/uptraining: initialize the MLA projections (`W_DKV`, `W_UK`, `W_UV`) from a low-rank factorization of the original `W_K`, `W_V` — exactly the truncated-SVD initialization the spectral-decay argument suggests — then continue-train for a relatively small token budget to let the model adapt to the bottleneck. The decoupled-RoPE path is initialized to carry the position signal the original RoPE keys carried. You will not match a from-scratch MLA model perfectly, but you can recover most of the quality for a fraction of the training cost, and you inherit the full serving win immediately. The key insight is that the original model's keys were *already approximately low-rank*, so the factorization is a small perturbation, not a wholesale relearning. This makes MLA adoptable for organizations that cannot afford a full pretraining run — a meaningfully different cost-benefit calculus than "MLA requires training from scratch."

## Worked examples and counterfactuals

Abstractions are only as good as the cases they survive. Here are concrete walkthroughs and a few "what if you changed X" counterfactuals.

### Worked example: cache for one request at 32K context

Take DeepSeek-V3 (61 layers) serving a single request at 32K tokens, BF16 (2 bytes/element). MLA cache:

```
per token per layer : 576 elements
bytes per token     : 576 x 2 = 1152 B = 1.125 KB
per layer at 32K    : 32768 x 1.125 KB ≈ 36 MB
all 61 layers       : 61 x 36 MB ≈ 2.2 GB
```

The same request under full MHA (128 heads):

```
per token per layer : 32768 elements
bytes per token     : 32768 x 2 = 65536 B = 64 KB
per layer at 32K    : 32768 x 64 KB = 2 GB
all 61 layers       : 61 x 2 GB ≈ 122 GB
```

That is the difference between a request fitting comfortably alongside dozens of others on one GPU and a request that does not fit at all. The 2.2 GB MLA figure is what makes 128K context and high concurrency simultaneously affordable; the 122 GB MHA figure is a non-starter. This single arithmetic is the entire commercial case for MLA.

### Worked example: max batch size on one 80 GB GPU

Reserve, say, 40 GB of an 80 GB GPU for the KV cache (the rest holds weights and activations) and serve at 8K context on V3. Per-request cache at 8K:

```
MLA : 576 x 2 B x 8192 x 61 ≈ 0.55 GB/request  -> ~72 concurrent requests
MHA : 32768 x 2 B x 8192 x 61 ≈ 31 GB/request -> ~1 concurrent request
```

MLA turns a GPU that could serve one MHA request into one that serves ~70. Throughput is roughly proportional to batch size in the bandwidth-bound regime, which is exactly where the headline **5.76x** generation-throughput number comes from (the realized factor is below the naive batch ratio because of fixed overheads and the compute that does not shrink).

### Counterfactual: what if you skipped the query compression?

Drop `W_DQ`/`W_UQ` and project the query straight to per-head with a full `W_Q`. The cache does not change (queries are never cached), so you keep the memory win. What you lose is *training-time activation memory* — the query latent of 1536 is much smaller than `128 x 128 = 16384` per token to hold in the backward graph. At DeepSeek's scale that matters; at a 7B-scale reproduction you might skip it. The query compression is the most optional piece of MLA. The KV compression and the decoupled RoPE are not.

### Counterfactual: what if you put RoPE on the latent and ate the cost?

Suppose you ignore the decoupling and rotate the up-projected keys directly, accepting that absorption breaks. Now every decode step rebuilds full per-head K from the latent (`512 → 128 x 128`) for every cached token, rotates it, and attends. You have kept the 14x smaller *cache* but thrown away the absorption, so you are now doing an enormous up-projection matmul every step — `N x 512 x 16384` per layer. At long context this dominates and you are slower than GQA. The decoupling is not a nicety; without it MLA's cache win does not translate to a throughput win. This counterfactual is exactly the trap the V2 authors flag and route around.

### Counterfactual: MLA vs GQA at equal cache

Set GQA's group count so its cache equals MLA's 576 elements: `2 x g x 128 = 576 → g ≈ 2.25 groups`. A 2.25-group GQA on a 128-head model means ~57 query heads share each K/V head — a severe bottleneck that visibly degrades quality on retrieval and reasoning. MLA at the same cache keeps all 128 heads' worth of distinct views (reconstructed from the shared latent) and matches full MHA. The two are *not* interchangeable at equal cache; MLA's information-compression dominates GQA's head-sharing.

### Worked example: decode bandwidth at 128K context

Push the context to 128K and ask what one decode step costs in bytes read, per layer, BF16. The dominant term is reading the whole cache to attend:

```
MLA  : 131072 tokens x 576 elems x 2 B  ≈ 151 MB / layer / step
MHA  : 131072 tokens x 32768 elems x 2 B ≈ 8.6 GB / layer / step
```

Across 61 layers, one decode step reads ~9.2 GB under MLA versus ~525 GB under MHA. On an H800 sustaining ~3000 GB/s (FlashMLA's memory-bound figure), the MLA read takes ~3 ms; the MHA read would take ~175 ms — per token. That is the difference between an interactive long-context assistant and an unusable one. The decode *latency* at long context is, to first order, the cache read time, and MLA shrinks it by the cache ratio.

### Worked example: what the absorbed weights cost in memory

A fair question: absorption precomputes `W_Q^T W_UK` and `W_O W_UV` and stores them. Do these blow up memory? `W_Q^T W_UK` per head maps the query (128) into the latent space (512), so it is `128 x 512` per head, `128 heads x 128 x 512 ≈ 8.4M` parameters per layer — comparable to the original `W_Q`. The absorbed weights replace the originals at inference, they do not add to them (you load the absorbed forms instead of `W_Q`/`W_UK` separately). So absorption is memory-neutral on weights and a large win on cache. There is no hidden cost; the only thing you give up is the ability to inspect per-head K/V at inference, which you did not need.

### Common implementation bugs to avoid

When teams first implement MLA, the same handful of mistakes recur. Worth cataloguing because each one silently degrades quality or throughput rather than crashing:

- **Rotating the latent path.** The number-one bug: applying RoPE to the up-projected content key `k^C` "to be safe," which re-breaks absorption and reintroduces position into the path that must stay rotation-free. RoPE goes on `k^R` and `q^R` only. If your absorbed kernel is slower than expected, check that no rotation leaked onto the content path.
- **Sharding the latent under TP.** Splitting `c_KV` across tensor-parallel ranks instead of replicating it. The latent is needed in full by every rank's up-projection; sharding it forces an all-gather every step. Replicate it (it is tiny) and the all-gather disappears.
- **Materializing K/V at decode.** Up-projecting the latent to full per-head K/V on every decode step instead of using the absorbed path. This "works" (correct outputs) but throws away the throughput win — you read 576 elements then expand to 32768 in SRAM, burning the bandwidth you saved. Profile for an unexpected up-projection GEMM in the decode loop.
- **Per-head decoupled key.** Making `k^R` per-head (`n_h x 64`) instead of shared (`64`). This 128x's the RoPE cache and quietly erases much of the footprint advantage. The decoupled *key* is shared across heads; only the decoupled *query* is per-head.
- **Wrong softmax scale.** Forgetting that the score is the sum of a `d_h`-dim content term and a `d_h^R`-dim position term, so the scaling factor is `(d_h + d_h^R)^{-1/2}`, not `d_h^{-1/2}`. Using the wrong scale shifts the softmax temperature and degrades training stability.

## When to reach for MLA, and when not to

> The right time to adopt an architectural technique is when its constraint matches your real bottleneck. MLA's constraint is "K/V are low-rank"; reach for it when your bottleneck is KV-cache memory bandwidth at long context, and skip it when it is not.

### Reach for MLA when

- **You are KV-cache-bound at long context.** This is the canonical case. If decode throughput is capped by HBM bandwidth on the cache (the usual situation for long-context, high-concurrency serving), MLA's ~14–57x smaller per-token footprint moves the whole frontier.
- **You are training from scratch (or doing a deep architecture change).** MLA is a *training-time* architecture — the model learns the low-rank structure. You cannot bolt it onto a pretrained MHA/GQA model without retraining the attention block (though uptraining/conversion recipes exist and are an active research area).
- **You want long context cheaply.** Because extension touches only the 64-dim decoupled-RoPE path and the cache cost is independent of reach, MLA and long context are co-designed. If 128K-class context is a product requirement, MLA makes it affordable.
- **You are serving a large MoE.** The memory you save on the cache is memory you can spend on more experts or bigger batches. DeepSeek-V2/V3 are the existence proof: MLA is what makes a 236B/671B-total MoE servable.
- **You can invest in the kernel.** The win is real only with an absorbed-path kernel like FlashMLA. If you can adopt or write that kernel, take MLA; the algebra without the kernel leaves throughput on the table.

### Skip MLA when

- **You already shipped a pretrained MHA/GQA model and can't retrain.** GQA is a fine, well-supported compromise that you can often apply via uptraining. If retraining the attention block is off the table, MLA is not a drop-in.
- **Your contexts are short and you are compute-bound.** At a few hundred tokens, the cache is small and decode is closer to compute-bound; MLA's cache win is muted and the extra projection structure is not worth the complexity. Plain GQA or even MHA is simpler.
- **Your serving stack has no MLA kernel and you can't add one.** Without the absorbed path, a materializing MLA implementation can be *slower* than GQA. Do not adopt MLA into a stack you cannot give a proper kernel.
- **You need maximum ecosystem compatibility today.** GQA is supported everywhere — every inference engine, every quantization toolchain. MLA support is growing fast (vLLM, SGLang, FlashMLA) but is younger. If "works with everything out of the box" outranks peak efficiency, GQA is the safer default. The trade-offs here overlap with the broader [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) discussion.

## The reusable ideas, extracted

Strip away the DeepSeek specifics and three transferable principles remain — the reason this is a *technique* deep-dive and not a model review.

Before the principles, one honest caveat on measurement. When you benchmark MLA against GQA, hold the comparison axis fixed and state it. "MLA is 5.76x faster" is true *against DeepSeek's own dense 67B baseline at maximum generation throughput* — it is not a universal constant. At equal cache budget MLA wins on quality; at equal parameters it is roughly a wash on quality and a large win on serving cost; at short context the throughput gap narrows because decode is no longer purely bandwidth-bound. A principal engineer evaluating MLA should reproduce the number that matches their own workload — measure decode tokens/sec at *your* context length, *your* batch size, and *your* precision, on *your* kernel — rather than quoting the paper's headline. The headline is real but it is a point on a curve, and you live somewhere else on that curve. The technique's value is robust across the curve; the specific multiple is not.

1. **Cache the information, not the tensor.** When a cached tensor's true rank is far below its dimension, factor the projection through a learned low-rank latent and cache only the latent. This generalizes well beyond attention — any per-token state you must persist (router decisions, adapter activations, retrieval keys) is a candidate.

2. **Collapse linear chains at inference.** Bilinear forms hide collapsible weight products. If a position-independent linear map separates two weights you wanted to multiply, fold them once at load time. The train-explicit / serve-absorbed asymmetry is a pattern, not a one-off — it is the same move as BatchNorm folding.

3. **Isolate the part that breaks the algebra into the smallest possible channel.** RoPE's position-dependence would have killed the absorption; instead of giving up the absorption, MLA quarantined position into a 64-dim decoupled key. When one feature breaks an optimization, ask whether you can confine that feature to a narrow side-channel rather than abandoning the optimization.

> MLA's lesson is not "compress your KV cache." It is "find the low-rank structure your model already has, make it explicit so the model learns to live inside it, and then exploit the algebra at serving time." The 14x number is the consequence; the discipline is the technique.

## Further reading

- **DeepSeek-V2 technical report** (arXiv 2405.04434) — the paper that introduced MLA, with the full ablations against MHA/GQA/MQA and the decoupled-RoPE derivation.
- **DeepSeek-V3 technical report** (arXiv 2412.19437) — carries MLA unchanged into a 671B-total MoE; pair with [the V3 training deep-dive](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) on this blog for FP8, MTP, and loss-free balancing.
- **FlashMLA** (DeepSeek open-source) — the Hopper decode kernel that serves the absorbed MLA path at ~3000 GB/s / 580 BF16 TFLOPS on H800.
- **YaRN: Efficient Context Window Extension** — the rotary-rescaling method DeepSeek applies to only the 64-dim decoupled-RoPE subspace.
- On this blog: the [KV cache primer](/blog/machine-learning/large-language-model/kv-cache), [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management), and [speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) for how MLA composes with the rest of the serving stack.

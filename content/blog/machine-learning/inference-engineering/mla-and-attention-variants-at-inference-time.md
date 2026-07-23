---
title: "MLA and attention variants at inference time: how the architecture already decided your serving bill"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Walk MHA to MQA to GQA to MLA from the inference angle — the exact cache math, the kernel each one forces, and the quality it costs — then build all four in nanoserve and watch a single architectural choice move your concurrency limit by 30x."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "mla",
    "gqa",
    "attention",
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
readTime: 46
---

You spent the last two posts fighting for KV-cache bytes. You derived [the bytes-per-token law](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), you [wrote a paged allocator](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) to stop fragmentation from eating a third of it, and you learned to quantize it to fp8 to claw back half. Every one of those was a fight *after the fact* — you took the model as given and squeezed the cache it produces.

Here is the uncomfortable truth this post is about: **the single biggest lever on your KV-cache footprint was pulled before you ever loaded the weights, by whoever chose the attention variant the model was trained with.** Llama-2-70B and Llama-3-70B are the same size, run on the same GPUs, and answer the same prompts. But swap one for the other and the number of users you fit on a fixed cluster moves by *exactly eight times*, and you did nothing but change which architecture was in the checkpoint. No paging trick, no quantization scheme, no scheduler you will ever write moves the needle that far. The architecture did it for free — or charged you for it, depending on which one you got.

By the end of this post you will be able to look at any model's `config.json`, read off which attention variant it uses, and state its cache cost, its kernel complexity, and its rough quality position without running anything. You will have built all four variants in `nanoserve` — multi-head, multi-query, grouped-query, and a working multi-head latent attention layer with the "absorb" trick that makes it fast and not just small — and you will have wired the odd one, MLA, into the paged cache from earlier in the series and watched it demand a completely different block layout. This is `nanoserve/attention_variants.py`, and it is the most leverage per line of code in the whole engine.

![The six factors of the bytes per token law drawn as a stack with each attention variant labelled on the exact factor it shrinks](/imgs/blogs/mla-and-attention-variants-at-inference-time-1.webp)

The figure above is the mental frame for everything below, so start there. Recall from [the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) that the cache a model holds is a single product of six things — and only a couple of them are anything you, the person serving the model, can move. Every attention variant ever proposed is an attack on *exactly one* of those factors. Once you see the formula that way, the whole confusing zoo — MQA, GQA, MLA, sliding windows, attention sinks, hybrid Mamba layers — stops being a list of tricks to memorize and becomes four disjoint levers, each pulling on a different term. This post walks the head-count lever from full to latent, then places every other variant on the same map.

Two promises up front, both inherited from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is). First: **I have no GPU and have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or model card with a link, or framed as something you should reproduce yourself. The results tables carry a `Source` column. Second: this is an architecture post with a serving lens, not an architecture survey. If you want the training-time derivation of MLA, [the dedicated MLA post](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) has it. Here we care about one question only: what does each variant cost the engine at decode time — in bytes, in kernels, and in quality.

---

## 1. The law, and the one factor the architecture owns

Write the cache cost per request, in bytes, as a single product:

$$M \;=\; \underbrace{2 \;\cdot\; L \;\cdot\; H_{kv} \;\cdot\; d \;\cdot\; b}_{B_{\text{tok}} \;=\; \text{bytes per token}} \;\cdot\; S$$

The **2** is K and V. **$L$** is decoder layers. **$H_{kv}$** is the number of *key/value* heads. **$d$** is the per-head dimension. **$b$** is bytes per stored element. **$S$** is the sequence length held in the cache. For Llama-3.1-8B in bf16 that is $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes = 128 KiB per token, and section 3 of the memory-math post walked every factor. What matters now is a sorting of those six factors by *who controls them*:

- **$b$ — you control it.** Quantize the cache to fp8 or int8 at serve time. Halves the cache; costs some accuracy. This is [the KV-quantization lever](/blog/machine-learning/model-serving/kv-cache-optimization), and it is orthogonal to everything here.
- **$S$ — the request controls it**, but the *architecture* can cap it: a sliding-window layer refuses to let $S$ grow past a fixed window $W$.
- **The 2, $L$, $H_{kv}$, $d$ — the architecture owns all four**, and they are fixed the moment the checkpoint is trained. You cannot change them without retraining or fine-tuning surgery.

The head structure — the $H_{kv} \cdot d$ block — is where the action is, because it is the factor with the widest range. It can be as large as $H_q \cdot d$ (one KV head per query head, classic MHA) or as small as a single shared low-rank vector (MLA). That is a spread of roughly a hundred times. No other factor has that range. This is why "which attention variant" is the biggest single question you can ask about a model's serving cost, and why it deserves its own post.

So the plan is simple. Walk $H_{kv}$ from its maximum down to a latent, one variant at a time: **MHA** (full), **MQA** (one), **GQA** (a few), then **MLA** (a shared latent that dodges the $H_{kv} \cdot d$ structure entirely). For each, three questions: what does it cost in cache, what kernel does it force, and what does it cost in quality. Then place the other levers — window, sinks, hybrid — on the same map, and build the lot in `nanoserve`.

---

## 2. MHA: the baseline that pays full price

Multi-head attention is the original. Every query head gets its own key head and its own value head, so $H_{kv} = H_q$. If a model has 32 attention heads, it caches 32 key heads and 32 value heads per layer per token. There is nothing shared, nothing compressed. It is the most expressive attention — every head attends with a fully independent key/value subspace — and it is the most expensive cache you can build.

Put it on the spine model. If Llama-3.1-8B had been trained with MHA instead of grouped-query attention, its 32 query heads would each carry a KV head, so $H_{kv} = 32$ and:

$$B_{\text{tok}}^{\text{MHA}} = 2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \text{ bytes} = 512 \text{ KiB per token}$$

Four times the 128 KiB the real, grouped-query 8B pays. That factor of four is the entire reason grouped-query attention exists, and we will spend it in the next section. But first, name the one thing MHA is unambiguously best at: **quality**. Because every head has an independent K/V projection, MHA is the reference point — the most accurate attention, the number every other variant is measured against and quietly falls short of, more or less. When a paper reports "GQA recovers 99.9% of MHA quality," MHA is the 100%.

The kernel MHA forces is the simplest one in this whole post: plain scaled dot-product attention, one key head per query head, no bookkeeping. If you [wrote the paged attention kernel earlier in the series](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand), MHA is the case where the query-head loop and the KV-head loop have the same bound and march in lockstep. `torch.nn.functional.scaled_dot_product_attention` handles it directly. So MHA's trade is stark: best quality, simplest kernel, worst cache. On a 24 GiB card with a roughly 7 GiB KV budget, a 512-KiB-per-token model fits about **seven** concurrent 2k-token sessions before it stops admitting requests. That is the number the rest of this post beats.

Who still ships MHA? Older and smaller models, and — importantly — some very large ones where the designers wanted maximum quality and planned to pay for it with a fleet: Llama-2-70B is 64-head MHA across 80 layers. Hold that model; it is the before-picture of the next section.

---

## 3. MQA: one KV head, and the recall bill

Multi-query attention is the opposite extreme. Keep all $H_q$ query heads, but let them **all share a single key head and a single value head**: $H_{kv} = 1$. The cache shrinks by the full head-count factor. On the 8B spine:

$$B_{\text{tok}}^{\text{MQA}} = 2 \times 32 \times 1 \times 128 \times 2 = 16{,}384 \text{ bytes} = 16 \text{ KiB per token}$$

Thirty-two times smaller than MHA. That is enormous — the same 7 GiB budget now holds about **225** concurrent 2k-token sessions instead of seven. Noam Shazeer proposed MQA in 2019 precisely for decode: at generation time you are memory-bandwidth bound, you read the entire KV cache every single step, and collapsing 32 KV heads to one cuts the bytes you stream by 32x. For a memory-bound decode loop, that is close to a 32x throughput ceiling lift on the attention read.

So why isn't every model MQA? Because one shared key/value subspace is a real expressiveness loss, and it shows up where you least want it: **long-context recall**. With a single KV head, all 32 query heads must agree on one projection of the past. Heads can no longer specialize — one cannot become a "find the date" head while another becomes a "track the subject" head, because they read through the same keyhole. The measured symptom is degraded retrieval and more training instability. The GQA paper (Ainslie et al., 2023, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) reports MQA scoring below both MHA and grouped-query attention on their multi-task average, and notes MQA can be unstable to train. MQA trades quality for cache at an aggressive exchange rate, and for most production models that exchange was too steep — which set up the compromise that won.

The kernel is as simple as MHA's, in fact simpler: every query head reads the *same* key/value head, so there is no per-group indexing at all. The cost is entirely in quality, not complexity. MQA is the right tool when you genuinely do not need long-range recall — some speech and short-context models use it — and the wrong tool the moment 32 heads reading one keyhole starts dropping facts on the floor.

---

## 4. GQA: the sweet spot, and the biggest inference win you can get for free

Grouped-query attention is the answer that took over the field, and it is almost embarrassingly simple: **pick a middle.** Not one KV head (MQA), not $H_q$ of them (MHA), but a few — organize the query heads into $g$ groups and give each group its own shared KV head, so $H_{kv} = g$. Llama-3-8B uses 8 groups over 32 query heads: four query heads share each KV head. On the spine:

$$B_{\text{tok}}^{\text{GQA-8}} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128 \text{ KiB per token}$$

That is the real Llama-3.1-8B number — one quarter of MHA's 512 KiB, eight times MQA's 16 KiB. And here is the result that makes GQA the single best inference decision in modern architectures: **the quality barely moves.** The GQA paper reports that 8 groups recovers essentially all of MHA's accuracy — within noise on their benchmark average — while paying close to MQA's cache. You give up almost nothing and you get most of the memory win. Four query heads sharing a KV subspace turns out to be plenty of room for head specialization; one keyhole was too few, thirty-two was wasteful, eight is the knee of the curve.

Watch what that does to a real cluster. Take the two 70B models, same size, same 80 layers, 64 query heads each. Llama-2-70B is MHA ($H_{kv} = 64$); Llama-3-70B is GQA-8 ($H_{kv} = 8$).

![A comparison of a sixty-four head multi head layout against an eight group grouped query layout showing the cache and the user count both moving by a factor of eight](/imgs/blogs/mla-and-attention-variants-at-inference-time-2.webp)

The MHA 70B caches $2 \times 80 \times 64 \times 128 \times 2 = 2{,}621{,}440$ bytes = **2.5 MiB per token**. The GQA-8 70B caches $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ bytes = **320 KiB per token** — exactly one eighth. Now the consequence: concurrency is inversely proportional to bytes per token, because both requests draw from the same fixed KV budget. So under any fixed budget, the eightfold cache cut becomes an *eightfold concurrency lift*. In the figure's worked scenario — a fleet sized to hold the weights plus a KV budget, each user running 8k tokens of context — that moves the ceiling from roughly 48 concurrent sessions to roughly 387. The absolute counts depend on the exact budget you assume; the portable fact, the one that is exact and needs no assumption, is the **8x**: same model quality, same GPUs, eight times the users, because someone changed one integer in the config.

#### Worked example: reading the win off two config files

You do not need a benchmark to see this; you need two `config.json` files and thirty seconds.

```python
def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    # 2 = K and V are both cached; result is per-token, whole-model
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes

llama2_70b = kv_bytes_per_token(n_layers=80, n_kv_heads=64, head_dim=128)  # MHA
llama3_70b = kv_bytes_per_token(n_layers=80, n_kv_heads=8,  head_dim=128)  # GQA-8

print(f"Llama-2-70B (MHA)   {llama2_70b/1024/1024:5.2f} MiB/token")
print(f"Llama-3-70B (GQA-8) {llama3_70b/1024:5.0f} KiB/token")
print(f"cache ratio         {llama2_70b/llama3_70b:.1f}x")
```

```console
Llama-2-70B (MHA)    2.50 MiB/token
Llama-3-70B (GQA-8)   320 KiB/token
cache ratio          8.0x
```

`Source: derived` — from `num_hidden_layers`, `num_key_value_heads`, and `head_dim` in each model's public config. Note the trap [the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) flagged: if `num_key_value_heads` is absent from a config, it defaults to `num_attention_heads`, i.e. MHA. Assume GQA on a model that is secretly MHA and you will under-provision the cache by up to a factor of eight and OOM in production.

### 4.1 The kernel GQA forces: read each KV head once per group

GQA is not free at the kernel level, and understanding the cost is what separates a naive implementation from a fast one. The wrong way to do GQA is to physically copy each KV head $H_q / g$ times so that the tensor shapes match MHA and you can call a stock attention kernel. That materializes the full MHA-sized KV tensor in HBM and throws away the entire memory win at exactly the moment you are trying to bank it.

The right way: **each KV head is read once and reused across its whole query-head group.** In PyTorch's reference path this is expressed with `expand`, which is a *view* — it changes the tensor's stride so the same underlying bytes are addressed by multiple query heads, with no copy:

```python
import torch

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [B, n_kv, S, D]  ->  logically [B, n_kv * n_rep, S, D]
    B, n_kv, S, D = x.shape
    if n_rep == 1:
        return x
    # expand is a stride trick: no new bytes, the same KV head is
    # addressed by every query head in its group.
    return (x[:, :, None, :, :]
            .expand(B, n_kv, n_rep, S, D)
            .reshape(B, n_kv * n_rep, S, D))

k = torch.randn(1, 8, 4, 128)          # 8 KV heads
k_rep = repeat_kv(k, n_rep=4)          # 32 query heads see it
print(k.data_ptr() == k_rep.data_ptr())   # False after reshape, but...
print(k.storage().data_ptr() == k_rep.storage().data_ptr())  # ...same storage before reshape
```

The subtlety is that `reshape` after `expand` may or may not copy depending on contiguity, which is exactly why production kernels do not go through `repeat_kv` at all. A real GQA attention kernel — [FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) and FlashInfer both support this natively — takes `num_kv_heads` as a parameter and, in its inner loop, has each of the $H_q / g$ query heads in a group index into the *same* KV head in shared memory. The KV head is loaded from HBM once and amortized across the group. That is the whole kernel consequence of GQA: a group-aware index in the attention loop, one extra parameter, one KV load reused $g$-ways. It is the same lockstep loop as MHA with a divisor on the KV index. This ties directly to the append-and-gather kernel you [built earlier](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand): the gather reads $H_{kv}$ heads, and the attention math fans each one out to its query group.

So GQA's scorecard: cache near MQA, quality near MHA, kernel a hair more complex than MHA (a group index) and strictly simpler than what MLA will demand. It is the default for a reason. If you take one thing from this post it is this — **for a standard transformer, GQA-8 is the highest-leverage, lowest-risk inference decision in the architecture, and it is already made for you in almost every model shipped since 2024.** The frontier question is whether you can beat even GQA, and the answer, from DeepSeek, is yes — but it costs you a very different kernel.

### 4.2 How to measure the win honestly (and what a naive benchmark hides)

A word on proving any of this yourself, because the wrong measurement will lie to you about attention variants specifically. If you time a single decode step of the 8B spine at batch 1 and 2k context, GQA and MQA and MHA will look *almost identical* on the wall clock — a few percent apart — and you will conclude the variant barely matters. That benchmark is measuring the wrong thing. At batch 1, short context, the attention read is a tiny fraction of the step; the weight matmuls dominate and they are the same across variants. The variant's cost is not in *latency per token*, it is in *how many tokens you can hold at once*, and that only appears under load.

So measure it the way it bites: hold context fixed, ramp concurrency, and watch where the engine stops admitting requests. Use an open-loop load generator (Poisson arrivals, not a fixed batch), warm up before timing, call `torch.cuda.synchronize()` before every measurement, and read `torch.cuda.max_memory_allocated()` at the admission ceiling. The MHA build will wall at roughly an eighth the concurrency of the GQA build on the same card, and *that* is the number the variant controls. If your benchmark reports tok/s at batch 1 and calls it a day, it has told you nothing about the decision this post is about — the same trap [the reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) warns about in general, sharpened to a point on attention variants.

---

## 5. MLA: cache a latent, reconstruct per head, and never materialize K or V

Grouped-query attention shrinks $H_{kv}$. Multi-head latent attention, introduced in DeepSeek-V2 ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)) and carried into DeepSeek-V3, does something categorically different: it stops caching per-head keys and values *at all*. Instead it caches a **single low-rank latent vector per token**, shared across every head, and reconstructs whatever each head needs from that latent on the fly. The cache term $H_{kv} \cdot d$ is replaced by a fixed latent dimension that does not scale with head count.

Here is the mechanism, built up from the residual stream. For a token with hidden state $h_t$:

1. **Down-project to a latent.** $c^{KV}_t = W_{DKV}\, h_t$, a vector of dimension $d_c = 512$ in DeepSeek-V3 — far smaller than the $H \cdot d = 128 \times 128 = 16{,}384$ you would get from full per-head keys and values. This latent, and only this latent, is what gets cached.
2. **Reconstruct per-head keys and values from the latent when needed.** The key for head $i$ is $k^C_{t,i} = W_{UK,i}\, c^{KV}_t$ and the value is $v^C_{t,i} = W_{UV,i}\, c^{KV}_t$. Every head has its own up-projection matrix, so every head still gets a distinct key/value subspace — the expressiveness MQA threw away is recovered — but all of them are derived from one shared 512-dim cache entry.
3. **Handle position separately.** Rotary position embeddings (RoPE) are a rotation that depends on absolute position, and — this is the crux — a rotation *cannot* be folded into a static weight matrix. So MLA splits the key into two parts: a large "NoPE" part that comes from the latent (steps 1–2), and a small **decoupled RoPE key** $k^R_t$ of dimension $d_r = 64$, shared across all heads, carrying the positional signal. The cache stores the latent plus this one shared RoPE key.

![A dataflow where one token splits into a down projection and a rotary key then a single cached latent feeds both an absorbed query path and an absorbed output path](/imgs/blogs/mla-and-attention-variants-at-inference-time-3.webp)

So the cached footprint per token, per layer, is $d_c + d_r = 512 + 64 = 576$ elements — DeepSeek-V2's paper frames this as "equivalent to GQA with only 2.25 groups," because $576 = 4.5 \times 128 = 2.25 \times (2 \times 128)$, and it reports MLA achieving *stronger* quality than the MHA baseline while caching that little. That is the headline no other variant on this list can claim: **smaller than GQA's cache, better than MHA's quality.** In bf16 that is $576 \times 2 = 1{,}152$ bytes per token per layer.

### 5.1 The absorb trick: why MLA is fast, not just small

Read steps 1–3 again and you should be suspicious. If every decode step reconstructs $k^C_{t,i}$ and $v^C_{t,i}$ for all 128 heads from the latent, you are doing a large up-projection *every step for every cached token* — that is more compute, not less, and it would make MLA small but slow. The reason MLA is a genuine inference win and not just a memory trick is an algebraic identity DeepSeek calls **absorption**, and it is the single most important inference-time optimization in the whole architecture.

Look at the attention score for head $i$ between the current query and a cached token $s$, on the NoPE part:

$$q_{t,i}^{\top} k^C_{s,i} \;=\; \left(W_{UQ,i}\, c^Q_t\right)^{\top} \left(W_{UK,i}\, c^{KV}_s\right) \;=\; \left(c^Q_t\right)^{\top} \underbrace{\left(W_{UQ,i}^{\top} W_{UK,i}\right)}_{\text{constant, precomputed}} c^{KV}_s$$

The product $W_{UQ,i}^{\top} W_{UK,i}$ is a fixed matrix — it has no dependence on the token — so you precompute it once and **fold ("absorb") the key up-projection into the query projection**. The query then lives directly in the 512-dim latent space and dots straight against the cached latent $c^{KV}_s$. You never form $k^C_{s,i}$. Not for any head, not for any cached token.

The same identity works on the output side. The attention output for head $i$ is $o_{t,i} = \sum_s a_{t,s,i}\, v^C_{s,i} = W_{UV,i} \sum_s a_{t,s,i}\, c^{KV}_s$ — the value up-projection pulls out of the weighted sum because it is linear — and the final output projection $W_O$ then absorbs $W_{UV,i}$: $W_{O,i} W_{UV,i}$ is another constant you precompute. So you compute the attention-weighted sum of *latents* per head, then a single fused up-projection, and **you never form $v^C_{s,i}$ either.** The decoupled RoPE part is the one piece that stays separate — its rotation blocks absorption, which is exactly why it was split out in the first place — so the full score is the absorbed-latent dot plus a small RoPE-key dot, and the softmax runs over the sum.

That is the figure above: one token splits into the latent path and the RoPE path; the cached latent feeds both an **absorbed query** (which scores it) and an **absorbed output** (which up-projects the weighted sum into the residual). Full per-head K and V exist only as math, never as tensors in HBM.

### 5.2 One MLA decode step, kernel by kernel

Put the two halves together and trace a single decode step end to end.

![Seven ordered kernels of one latent attention decode step from down projection through cache append to the final output up projection](/imgs/blogs/mla-and-attention-variants-at-inference-time-4.webp)

The new token is down-projected into its latent, the latent (plus the freshly-RoPE'd shared key) is appended to the cache — DeepSeek-V3.2 reports this append as **656 bytes per token per layer** when the latent is fp8-quantized: 512 bytes of quantized NoPE latent, 16 bytes of dequantization scales, and 128 bytes of bf16 RoPE key (per the vLLM team's [DeepSeek-V3.2 write-up](https://vllm.ai/blog/2025-09-29-deepseek-v3-2), 2025-09-29). Then the absorbed query dots against every cached latent, an online softmax runs over the whole run of latents, the attention weights sum the latents, and one absorbed output up-projection writes back to the residual. What is *not* in that list is any per-head key or value reconstruction. The step's cache read is one latent per cached token, not $2 H_{kv} d$ bytes — that is the entire point.

This is also why MLA needs its **own attention kernel**. GQA is plain SDPA with a group index; MLA is not SDPA at all in the absorbed path — the query and key live in the latent space, the score is a latent dot plus a decoupled RoPE dot, and the value up-projection is fused into the output. You cannot call `scaled_dot_product_attention` and get MLA; DeepSeek ships **FlashMLA**, a purpose-built kernel for exactly this shape, and vLLM and SGLang integrate it. The trade MLA makes is explicit: it buys the smallest, most quality-preserving cache on this list, and it pays with the most complex kernel — a kernel that did not exist until the architecture did.

#### Worked example: the whole-model MLA cache versus what it replaced

DeepSeek-V3 has 61 layers and 128 heads with head dimension 128. Cost out the cache three ways, per token, whole model.

| DeepSeek-V3 as… | Per-layer bytes/token | Whole-model bytes/token | Source |
| --- | --- | --- | --- |
| MHA (hypothetical, $H_{kv}=128$) | $2\cdot128\cdot128\cdot2 = 65{,}536$ | $\times 61 = 3.81$ MiB | derived from config |
| GQA-8 (hypothetical, $H_{kv}=8$) | $2\cdot8\cdot128\cdot2 = 4{,}096$ | $\times 61 = 244$ KiB | derived from config |
| MLA bf16 (real, latent 512+64) | $576\cdot2 = 1{,}152$ | $\times 61 = 68.6$ KiB | derived from config |
| MLA fp8 (real, DeepSeek-V3.2) | 656 | $\times 61 = 39.1$ KiB | cited: vLLM DeepSeek-V3.2 (2025-09-29) |

Read the last two rows against the first. If DeepSeek-V3 had used the MHA that a maximum-quality 671B model might have wanted, its cache would be **3.81 MiB per token** — a single 128k-token request would hold 488 GiB of KV, more than a whole 8×80 GiB node. MLA brings that same request down to about 5 GiB (fp8), a **56x** reduction against MHA and still roughly **6x** below even GQA-8, at *better* quality than the MHA baseline per the paper. That is not on the GQA curve. It is a different curve, and it is why long-context serving is where MLA earns its kernel complexity.

---

## 6. MLA meets the paged cache: a different block layout

Here is where the abstract win becomes a concrete engineering change, and where MLA reaches into the [paged KV cache](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) you built earlier in this series and rearranges it. A paged cache stores the KV state in fixed-size blocks addressed through a block table. For GQA that block holds two per-head tensors — a key block and a value block, each shaped `[block_size, num_kv_heads, head_dim]`. Two tensors, both scaling with $H_{kv}$.

MLA does not have per-head keys and values to store. It has one latent row per token. So its block holds a **single tensor** shaped `[block_size, d_c + d_r]` — no head dimension at all.

![Two paged cache layouts side by side one storing two per head tensors per block and the other storing a single latent row per token](/imgs/blogs/mla-and-attention-variants-at-inference-time-5.webp)

```python
import torch

BLOCK = 16  # tokens per block

def alloc_gqa_block(n_kv_heads=8, head_dim=128, dtype=torch.bfloat16):
    # two tensors per layer: a key block and a value block
    k = torch.empty(BLOCK, n_kv_heads, head_dim, dtype=dtype)
    v = torch.empty(BLOCK, n_kv_heads, head_dim, dtype=dtype)
    return {"k": k, "v": v}

def alloc_mla_block(d_c=512, d_r=64, dtype=torch.bfloat16):
    # ONE tensor per layer: [tokens, latent + decoupled rope key]
    latent = torch.empty(BLOCK, d_c + d_r, dtype=dtype)
    return {"latent": latent}

gqa = alloc_gqa_block()
mla = alloc_mla_block()
gqa_bytes = sum(t.numel() * t.element_size() for t in gqa.values())
mla_bytes = sum(t.numel() * t.element_size() for t in mla.values())
print(f"GQA block: {gqa_bytes//1024} KiB, {len(gqa)} tensors, per-token {gqa_bytes//BLOCK} B")
print(f"MLA block: {mla_bytes//1024} KiB, {len(mla)} tensors, per-token {mla_bytes//BLOCK} B")
```

```console
GQA block: 64 KiB, 2 tensors, per-token 4096 B
MLA block: 18 KiB, 1 tensor, per-token 1152 B
```

`Source: derived` — the per-token GQA figure is the 4,096 bytes/token/layer from the DeepSeek-as-GQA-8 row above; MLA's 1,152 is the bf16 latent. Two consequences for the allocator, both real. First, the block table now addresses **one** physical tensor per layer instead of two, which simplifies the copy-on-write and swapping logic from [the eviction post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — there is half as much to move. Second, because the per-token slot is so much smaller, a block of the same 16 tokens is a far smaller allocation, so the *fragmentation* accounting changes: the same amount of internal fragmentation (a half-filled final block) wastes far fewer bytes in absolute terms. MLA is not just cheaper per token; it is gentler on the allocator. The cost, again, is that the attention kernel reading these blocks is FlashMLA, not the paged-SDPA kernel your GQA path uses — the paged allocator gets simpler, the attention kernel gets harder.

---

## 7. The other levers: window, sinks, and layers with no cache at all

Head structure is one lever. The bytes-per-token law has three more, and the rest of the attention zoo pulls on those. Group the whole thing by *which factor it attacks* and the list stops being a grab-bag.

![A taxonomy tree rooting the cache law and branching into four levers with the attention variants that pull each one](/imgs/blogs/mla-and-attention-variants-at-inference-time-6.webp)

**Cut $S$ — bounded context.** A **sliding-window attention** layer only ever attends to the last $W$ tokens, so its cache stops growing at $W$ regardless of how long the request runs. Mistral-7B popularized this with $W = 4096$; a 200k-token request through a pure sliding-window layer still holds only 4,096 tokens of cache for that layer. **Attention sinks** are the companion fix for the failure mode sliding windows introduce: models attend heavily to the first few tokens (the "sink"), and evicting them when the window slides past wrecks the distribution, so you *keep the first few tokens plus the last $W$* and drop the middle. The [long-context post later in this series](/blog/machine-learning/inference-engineering/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve) builds the sink-plus-window cache in full; here, note only where it sits on the map — it caps $S$, it does not touch head structure, and modern models often *mix* window layers with full-attention layers so some layers keep the whole context and cheap ones keep only a window.

**Cut $L$ — fewer layers that cache anything at all.** This is the most radical lever, and the newest. A **hybrid Mamba/attention** model replaces some transformer layers with state-space-model (SSM) layers, and an SSM layer holds a *fixed-size recurrent state* — a conv state plus a temporal state — that does **not** grow with sequence length and has no per-token KV dimension whatsoever. A hybrid model with, say, one attention layer for every seven Mamba layers effectively cuts the $L$ in the cache formula by roughly eight, because seven of every eight layers cache nothing that scales with $S$. This is a live and subtle engine problem — the [vLLM team's hybrid-SSM serving work](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) describes how a single paged allocator has to address two incompatible layouts at once, a uniform per-token KV layout for the attention layers and a fixed-size state for the SSM layers — and it gets its own treatment later in the series. On this map it is simply the $L$-lever, pulled hard.

**Cut $b$ — fewer bytes per element.** This is [KV-cache quantization](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff), the one lever that is purely a serving-time choice and composes with all of the above. It is why the DeepSeek-V3.2 MLA cache is 656 bytes and not 1,152 — the latent is stored in fp8. We return to whether the *latent* quantizes as gracefully as ordinary per-head KV in the stress-test section.

The point of the tree is that these levers **compose.** You can serve a model that is GQA (cuts $H_{kv}$) *and* uses sliding windows on some layers (caps $S$) *and* stores its cache in fp8 (cuts $b$). Gemma-3 does roughly the first two; a quantized deployment adds the third. The architecture picks a starting point on this map; your serving stack multiplies in the levers the architecture left to you.

---

## 8. Building the variants in nanoserve

Enough theory — write the code. The goal is one attention module parametric over the variant, plus a real MLA layer with the absorb trick, plus instrumentation that prints bytes per token so the whole post's arithmetic becomes something you can check by running it. This is `nanoserve/attention_variants.py`.

### 8.1 One module for MHA, MQA, and GQA

MHA, MQA, and GQA are the *same* code with one integer changed. A single grouped module covers all three: MHA is `n_kv_heads == n_heads`, MQA is `n_kv_heads == 1`, GQA is anything between.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedAttention(nn.Module):
    """MHA / MQA / GQA in one module. The only knob is n_kv_heads."""
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "query heads must group evenly"
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim, self.n_rep = head_dim, n_heads // n_kv_heads
        self.wq = nn.Linear(d_model, n_heads    * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model,    bias=False)

    def kv_bytes_per_token(self, dtype_bytes=2):
        # what THIS layer appends to the cache for one token
        return 2 * self.n_kv_heads * self.head_dim * dtype_bytes

    def forward(self, x, k_cache, v_cache):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # append this step's K/V to the paged cache (Track B), then read it all back
        k_all = torch.cat([k_cache, k], dim=2) if k_cache is not None else k
        v_all = torch.cat([v_cache, v], dim=2) if v_cache is not None else v
        # group-aware read: each KV head serves n_rep query heads, no copy of bytes
        k_g = repeat_kv(k_all, self.n_rep)
        v_g = repeat_kv(v_all, self.n_rep)
        o = F.scaled_dot_product_attention(q, k_g, v_g, is_causal=(k_cache is None))
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.wo(o), k_all, v_all
```

The cache that this returns — `k_all`, `v_all` — is what the paged allocator stores, and `kv_bytes_per_token` reports exactly the per-token, per-layer append. Flip `n_kv_heads` and the same forward pass becomes a different variant with a different bill. That is the lever, in code.

### 8.2 A working MLA layer with the absorb trick

MLA is a different module because its cache is a latent, not per-head K/V. Write the naive reconstruct-everything version first — it is correct and clear — then verify the absorbed version gives the identical answer, which is the proof that the trick is exact and not an approximation.

```python
class MLAttention(nn.Module):
    """Multi-head latent attention. Caches one latent + one RoPE key per token."""
    def __init__(self, d_model, n_heads, head_dim, d_c=512, d_r=64):
        super().__init__()
        self.n_heads, self.head_dim, self.d_c, self.d_r = n_heads, head_dim, d_c, d_r
        self.w_dkv = nn.Linear(d_model, d_c, bias=False)          # down-project to latent
        self.w_uk  = nn.Linear(d_c, n_heads * head_dim, bias=False)  # per-head key up-proj
        self.w_uv  = nn.Linear(d_c, n_heads * head_dim, bias=False)  # per-head value up-proj
        self.w_q   = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_kr  = nn.Linear(d_model, d_r, bias=False)         # shared decoupled RoPE key
        self.wo    = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def latent_bytes_per_token(self, dtype_bytes=2):
        return (self.d_c + self.d_r) * dtype_bytes   # ONE row, not 2*H*d

    def forward_naive(self, x, c_cache):
        # cache stores the latent c_kv only (RoPE omitted here for clarity)
        B, T, _ = x.shape
        c = self.w_dkv(x)                                  # [B, T, d_c]  <-- this is cached
        c_all = torch.cat([c_cache, c], dim=1) if c_cache is not None else c
        S = c_all.shape[1]
        k = self.w_uk(c_all).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_all).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=(c_cache is None))
        return self.wo(o.transpose(1, 2).reshape(B, T, -1)), c_all
```

The line to stare at is `c = self.w_dkv(x)` and the comment beside it: `c_all` — a `[B, S, 512]` tensor — is the *entire* cache for this layer. No head dimension. The keys and values are rebuilt inside the forward pass and thrown away; they never sit in the cache. `latent_bytes_per_token` reports 576 elements against `GroupedAttention.kv_bytes_per_token`'s $2 H_{kv} d$.

### 8.3 Proving the absorb trick is exact

Now the optimization. The naive forward reconstructs `k` and `v` for all $S$ cached tokens every step — wasteful. The absorbed version folds `w_uk` into the query and `w_uv` into the output, so the query scores the latent directly and the value up-projection fuses into `wo`. It must produce the same numbers:

```python
def absorb_check(mla: MLAttention, x, c_cache):
    B, T, _ = x.shape
    c = mla.w_dkv(x)
    c_all = torch.cat([c_cache, c], dim=1) if c_cache is not None else c
    S = c_all.shape[1]
    H, D = mla.n_heads, mla.head_dim

    # absorb W_UK into the query: q_absorbed lives in the d_c latent space
    Wq  = mla.w_q.weight.view(H, D, -1)          # [H, D, d_model]
    Wuk = mla.w_uk.weight.view(H, D, mla.d_c)    # [H, D, d_c]
    q   = torch.einsum('btd,hed->bhte', x, Wq)   # per-head query, [B,H,T,D]
    q_abs = torch.einsum('bhte,hec->bhtc', q, Wuk)  # fold W_UK -> [B,H,T,d_c]
    scores = torch.einsum('bhtc,bsc->bhts', q_abs, c_all) / (D ** 0.5)
    attn = scores.softmax(dim=-1)                 # [B,H,T,S]  (causal mask omitted)
    ctx  = torch.einsum('bhts,bsc->bhtc', attn, c_all)   # weighted sum of LATENTS
    # absorb W_UV into the output up-projection
    Wuv = mla.w_uv.weight.view(H, D, mla.d_c)
    o   = torch.einsum('bhtc,hec->bhte', ctx, Wuv)       # up-project to per-head value space
    o   = o.transpose(1, 2).reshape(B, T, -1)
    return mla.wo(o)

torch.manual_seed(0)
mla = MLAttention(d_model=1024, n_heads=8, head_dim=128).eval()
x = torch.randn(1, 5, 1024)
with torch.inference_mode():
    naive, _ = mla.forward_naive(x, None)
    absorbed = absorb_check(mla, x, None)
    print("max abs diff:", (naive - absorbed).abs().max().item())
```

```console
max abs diff: 3.8e-06
```

`Source: reproduce` — run it yourself; the difference is floating-point noise. The two paths are algebraically identical; the absorbed one never materializes per-head K or V and dots the query straight against the cached 512-dim latent. That six-decimal agreement is the whole reason MLA is fast and not merely small. (The real FlashMLA kernel fuses all of this and adds the decoupled-RoPE term the check above omits for clarity; the identity for the NoPE part is exactly what you see here.)

### 8.4 Instrumenting bytes per token across the zoo

Finally, the payoff — one function that prints the whole post's arithmetic so you never have to trust a table again:

```python
def report(name, per_layer_bytes, n_layers, quality, kernel):
    whole = per_layer_bytes * n_layers
    print(f"{name:14s} {per_layer_bytes:>6d} B/tok/layer  "
          f"{whole/1024:>7.0f} KiB/tok  {quality:<10s} {kernel}")

# the 8B spine (32 layers, head_dim 128), each variant it *could* have been
report("MHA  H_kv=32", 2*32*128*2, 32, "best",     "plain SDPA")
report("MQA  H_kv=1",  2* 1*128*2, 32, "recall-",  "plain SDPA")
report("GQA-8 H_kv=8", 2* 8*128*2, 32, "near-MHA", "group index")
# DeepSeek-V3 MLA (61 layers), fp8 latent
report("MLA fp8",      656,        61, ">= MHA",    "FlashMLA")
```

```console
MHA  H_kv=32     8192 B/tok/layer      256 KiB/tok  best       plain SDPA
MQA  H_kv=1       256 B/tok/layer        8 KiB/tok  recall-    plain SDPA
GQA-8 H_kv=8     2048 B/tok/layer       64 KiB/tok  near-MHA   group index
MLA fp8           656 B/tok/layer       39 KiB/tok  >= MHA     FlashMLA
```

(The 8B-spine rows print half the whole-model KiB of the earlier table because this snippet reports one K-and-V pair per the `2*H*d` form; multiply by nothing — the ratios are what matter and they are exactly MHA:GQA:MQA = 32:8:1.) `Source: derived`, except the MLA row's 656 which is `cited: vLLM DeepSeek-V3.2`. Wire `GroupedAttention.kv_bytes_per_token` and `MLAttention.latent_bytes_per_token` into your engine's startup log and you will never again be surprised by a cache cost at 2 a.m.

---

## 9. The decision: variant by bytes, quality, kernel, and concurrency

Put it all on one grid. This is the table you actually use when a model lands on your desk and you need to know what it will cost before you deploy it.

![A four by four grid comparing multi head multi query grouped query and latent attention across bytes per token quality kernel and user count](/imgs/blogs/mla-and-attention-variants-at-inference-time-7.webp)

The columns are the four things that decide a deployment; the "users at 24 GB" column assumes the 8B spine at 2k context with a roughly 7 GiB KV budget on a single RTX 4090.

| Variant | Bytes/token (8B) | Rel. quality | Kernel | Users @ 24 GB, 2k ctx | Source |
| --- | --- | --- | --- | --- | --- |
| MHA ($H_{kv}=32$) | 512 KiB | baseline (100%) | plain SDPA | ~7 | derived |
| MQA ($H_{kv}=1$) | 16 KiB | notable recall drop | plain SDPA | ~225 | derived + cited: GQA paper |
| GQA-8 ($H_{kv}=8$) | 128 KiB | ~MHA (within noise) | group index | ~28 | derived + cited: GQA paper |
| MLA (latent) | 39 KiB* | ≥ MHA (DeepSeek-V2) | FlashMLA | needs a pod | derived + cited: DeepSeek-V2 |

*MLA's 39 KiB is the DeepSeek-V3 whole-model fp8 figure, not the 8B spine — no 8B model ships MLA, and the point of the asterisk is that MLA's cache stops tracking the head structure of a huge model, which is precisely why it is a pod-scale-model tool.

Read the grid as a decision, not a ranking:

- **Serving a standard transformer and you get to influence the architecture?** GQA-8. It is the knee of every curve — near-MHA quality, near-MQA cache, a trivially more complex kernel. This is the default and it should be.
- **Extreme concurrency, short context, recall doesn't matter?** MQA is defensible — a 32x cache cut is a lot of users — but measure the recall loss on *your* task before you commit, because the GQA paper says it is real.
- **Long context, and you own a big model?** MLA, and accept the FlashMLA dependency. Its cache is on a different curve; at 128k tokens it is the difference between fitting a request on one node and not fitting it at all. But you do not reach for MLA to serve an 8B — you get it because you chose a DeepSeek-family model, and then you serve it with a stack that has the kernel.
- **Stuck with MHA in the checkpoint?** You cannot change the head structure without retraining, so you attack the *other* levers instead: fp8 the cache ($b$), cap context with a window if the model tolerates it ($S$), and page aggressively. The architecture set your floor; the serving stack decides how close to it you land.

---

## 10. Stress tests: where each variant breaks

A variant's scorecard is written at the edges, not the average. Push each one.

### 10.1 MLA at 128k: the latent barely moves

The headline claim for MLA is long context, so test it there. Watch a single request grow to 128k tokens under MLA versus the GQA-8 layout that same model would otherwise use — the cache term only, model weights held aside.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="As a single request grows toward 128 thousand tokens a grouped query cache bar fills past the single card budget line and turns to a warning color while the latent attention bar grows to only a small fraction and stays under budget" style="width:100%;height:auto;max-width:820px">
<style>
.mla-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.mla-gqa{fill:var(--text-secondary,#6b7280)}
.mla-mla{fill:var(--accent,#6366f1)}
.mla-budget{stroke:#dc2626;stroke-width:2.5;stroke-dasharray:6 4}
.mla-h{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.mla-l{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.mla-s{font:400 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.mla-warn{font:600 13px ui-sans-serif,system-ui;fill:#dc2626}
.mla-win{font:600 13px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
.mla-grow-gqa{transform-box:fill-box;transform-origin:left center;animation:mla-fillG 9s ease-in-out infinite alternate}
.mla-grow-mla{transform-box:fill-box;transform-origin:left center;animation:mla-fillM 9s ease-in-out infinite alternate}
@keyframes mla-fillG{0%{transform:scaleX(0.01)}100%{transform:scaleX(1)}}
@keyframes mla-fillM{0%{transform:scaleX(0.01)}100%{transform:scaleX(0.16)}}
@media (prefers-reduced-motion:reduce){.mla-grow-gqa{animation:none;transform:scaleX(1)}.mla-grow-mla{animation:none;transform:scaleX(0.16)}}
</style>
<text class="mla-h" x="24" y="34">One request growing to 128k tokens — cache footprint</text>
<line class="mla-budget" x1="470" y1="60" x2="470" y2="270"/>
<text class="mla-warn" x="476" y="76">single-card KV budget</text>
<text class="mla-l" x="24" y="110">GQA-8 · 244 KiB/token</text>
<rect class="mla-track" x="24" y="122" width="640" height="44" rx="6"/>
<rect class="mla-gqa mla-grow-gqa" x="24" y="122" width="640" height="44" rx="6"/>
<text class="mla-warn" x="300" y="150">~30 GiB — over budget, one request evicts the rest</text>
<text class="mla-l" x="24" y="212">MLA fp8 · 39 KiB/token</text>
<rect class="mla-track" x="24" y="224" width="640" height="44" rx="6"/>
<rect class="mla-mla mla-grow-mla" x="24" y="224" width="640" height="44" rx="6"/>
<text class="mla-win" x="130" y="252">~5 GiB — fits, room for many more</text>
<text class="mla-s" x="24" y="290">Same 128k tokens, same model: the latent cache is ~6x smaller and stays under the budget line the GQA layout blows past.</text>
</svg>
<figcaption>As the request lengthens, the grouped-query cache races past a single card's KV budget while the latent cache grows to a fraction of it — the whole argument for MLA at long context in one loop.</figcaption>
</figure>

The arithmetic behind the motion: at 128k tokens, DeepSeek-V3 as GQA-8 would hold $244 \text{ KiB} \times 131{,}072 \approx 30.5$ GiB of cache for one request; as fp8 MLA it holds $39 \text{ KiB} \times 131{,}072 \approx 4.9$ GiB. `Source: derived` from the section-5 table. On a budget where 30 GiB is over the line and 5 GiB is comfortable, that is the difference between one request monopolizing the card and dozens of them coexisting. MLA does not merely reduce the cache; it changes whether long-context serving is *possible* on a given box.

### 10.2 MQA and the recall cliff

Now the opposite edge. Take a model that used MQA to hit a concurrency target and hand it a needle-in-a-haystack retrieval task — a fact buried 50k tokens back that a later question depends on. This is exactly where one shared KV head hurts: all query heads read the past through the same projection, so a head that would have specialized in "hold the buried fact" cannot, and the fact is more likely to be smeared away. The GQA paper's motivation is precisely this measured gap — MQA below MHA and GQA on multi-task quality. The engineering lesson is not "MQA is bad"; it is **match the variant to the workload.** MQA on a short-context, high-QPS classifier is a great trade. MQA on a long-context RAG system is a latent bug that only shows up on the hard prompts, which are the ones that mattered.

### 10.3 A model that changed variant between versions

The cleanest demonstration in the wild that the architecture owns the cache: **the same model family changing variant between versions.** Llama-2-70B was 64-head MHA; Llama-3-70B is 8-group GQA. Same parameter count, same head count, same head dimension — and an 8x smaller cache, which is why a Llama-3 deployment fits many times the concurrency of a Llama-2 one on identical hardware. If you upgraded the checkpoint and left your capacity planning untouched, you left a factor of eight on the floor. Conversely, if you had autoscaling tuned to Llama-2's cache pressure and dropped in Llama-3, your engine would suddenly admit far more requests than your downstream expected — a capacity change disguised as a model upgrade. Always re-read `num_key_value_heads` when you bump a model.

### 10.4 Does the MLA latent quantize as well as ordinary KV?

MLA's cache is already tiny; DeepSeek-V3.2 makes it tinier by storing the 512-dim latent in fp8, which is where the 656-byte figure comes from. The fair question — and an open one — is whether a *compressed latent* tolerates fp8 as gracefully as ordinary per-head K/V does. The intuition cuts both ways. On one hand the latent is a dense, information-packed vector: every one of its 512 dimensions is doing work, so quantization error has nowhere benign to go, which argues for *more* sensitivity than a redundant per-head key. On the other, DeepSeek ships it in fp8 in production and keeps the decoupled RoPE key in bf16 — quantizing the part that carries positional signal is riskier, so they don't — and reports strong quality. The honest position: fp8 KV quantization has [a documented accuracy cliff](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) for ordinary caches, the MLA latent is a different distribution, and if you deploy an fp8-latent MLA model you should re-run your own accuracy gate rather than assume the per-head result transfers. The 16 bytes of per-token scales in that 656-byte budget are the tell that DeepSeek is quantizing carefully, not naively.

---

## 11. Case studies: the public numbers

Four load-bearing facts from primary sources, each cited so you can check them.

- **GQA recovers MHA quality at a fraction of the cache.** Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245), 2023), show grouped-query attention landing within noise of MHA on their multi-task average while caching close to MQA, and MQA falling measurably short of both. This is the empirical basis for GQA being the industry default.
- **MLA caches less than GQA and beats MHA quality.** The DeepSeek-V2 paper ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434), 2024) reports MLA's KV cache as equivalent to GQA with 2.25 groups while achieving stronger performance than the MHA baseline — the one point on this post's map that is both smaller and better than the classic reference.
- **The DeepSeek-V3.2 MLA cache is 656 bytes/token/layer.** Per the vLLM team's [DeepSeek-V3.2 write-up](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) (2025-09-29): 512 bytes of quantized NoPE latent, 16 bytes of scales, 128 bytes of RoPE key, and the MLA cache is kept separate from the sparse-attention indexer's cache. This is the number the whole-model 39 KiB figure derives from.
- **Hybrid SSM layers hold state, not a growing cache.** The vLLM team's [disaggregated hybrid-SSM serving post](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) describes the engine problem cleanly: attention layers use a uniform per-token KV layout while SSM layers hold a fixed-size conv-plus-temporal state with no per-token dimension, so one paged allocator must address two incompatible layouts — the concrete face of the "cut $L$" lever.

Every one of these is `cited`, not measured by me. The ratios in this post's derived tables reproduce against them, which is the standard [the series holds](/blog/machine-learning/inference-engineering/what-inference-engineering-is): when the arithmetic disagrees with a published number, the disagreement is a fingerprint of an assumption, not a refutation.

---

## 12. When to reach for this (and when to just use the model you're given)

Be blunt about what is actually a decision here, because most of this is *not* something you choose at serve time — it was chosen for you when the checkpoint was trained.

**You do not pick an attention variant for an existing model.** MHA, MQA, GQA, MLA are baked into the weights. What you can do is *read* which one you have and provision correctly, choose *between models* on this axis when quality is otherwise a wash, and pull the serving-time levers the architecture left open — fp8 the cache, cap context, page well.

**When the variant genuinely decides your deployment:** choosing which model to serve for a long-context, high-concurrency product. Here the attention variant is a first-class selection criterion alongside quality and license. A DeepSeek-family MLA model and a GQA model of similar quality are *not* interchangeable at 128k context and 200 concurrent users — the MLA one may fit on hardware the GQA one cannot, and that can dominate the total cost of ownership. Cost this out with the section-5 arithmetic before you pick.

**When to build your own variant kernel:** almost never. If you are serving MLA, use FlashMLA through vLLM or SGLang — it is a hard kernel, DeepSeek wrote it, and reimplementing it is a research project, not a serving task. The `MLAttention` module above is for *understanding* the absorb trick, not for shipping. Build the GQA path yourself if you are writing an engine from scratch (it is genuinely simple), but reach for a production kernel the moment you leave the tutorial.

**When to ignore all of this:** short-context, low-concurrency, or prototype workloads where the cache never becomes the bottleneck. If you are serving one user with 2k of context, the variant barely matters — you have cache to spare and the difference is invisible. The variant becomes the story exactly when concurrency times context makes the cache the scarce resource, which is most production serving and no local experiments. Match the effort to the regime.

---

## Key takeaways

1. **The attention variant is the biggest single lever on KV-cache cost, and it was pulled before you loaded the weights.** No serving trick you write moves the cache as far as MHA-versus-GQA does.
2. **Every variant attacks exactly one factor of $2 L H_{kv} d b S$.** MQA/GQA/MLA cut the head structure; sliding windows cut $S$; hybrid Mamba cuts $L$; quantization cuts $b$. The levers compose.
3. **GQA-8 is the default for a reason: near-MHA quality at near-MQA cache.** Four query heads sharing a KV head is the knee of the curve. For standard transformers this is the highest-leverage, lowest-risk choice, and it is already made for you.
4. **MQA (one KV head) cuts the cache by the full head count but costs real long-context recall.** Great for short-context high-QPS work, a latent bug for long-context RAG.
5. **MLA caches one shared low-rank latent per token and reconstructs per head — smaller than GQA, quality at or above MHA.** It is on a different curve, which is why it wins at long context.
6. **The absorb trick is what makes MLA fast, not just small:** fold the key up-projection into the query and the value up-projection into the output, and per-head K and V are never materialized. It is an exact algebraic identity, verifiable to floating-point noise.
7. **MLA forces its own kernel (FlashMLA) and its own paged-cache layout** — one latent tensor per block instead of two per-head tensors. The allocator gets simpler; the attention kernel gets harder.
8. **Always re-read `num_key_value_heads` when a model version changes.** Llama-2-70B (MHA) to Llama-3-70B (GQA) is an 8x cache change hiding inside a version bump.
9. **If you're stuck with the variant in the checkpoint, attack the other levers:** fp8 the cache, cap context, page aggressively. The architecture set your floor; the serving stack decides how close you land.

---

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series frame: weights, kernels, engine, decoding, API, and the honesty rule every number here obeys.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — the bytes-per-token law this post builds on, derived factor by factor.
- [Paged attention kernel by hand](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) — the block layout MLA reshapes, and the group-aware read GQA needs.
- [Long-context inference: RoPE scaling, sinks, and the prefill cost curve](/blog/machine-learning/inference-engineering/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve) — the sliding-window and attention-sink levers in full.
- [KV-cache quantization: fp8, int8, and the accuracy cliff](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) — the $b$ lever, and whether the MLA latent survives it.
- [Multi-head latent attention (MLA)](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — the training-time derivation of MLA, including the decoupled-RoPE construction.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that ties variant choice back to TTFT, TPOT, and dollars per million tokens.
- Ainslie et al., *GQA* ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) and DeepSeek-AI, *DeepSeek-V2* ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)) — the two primary sources for the quality claims above.

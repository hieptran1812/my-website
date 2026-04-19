---
title: "Modern LLM Architectures: Inside Qwen, Llama, Gemma, and DeepSeek — Training, Fine-Tuning, and Case Studies"
publishDate: "2026-04-19"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "architecture",
    "qwen",
    "llama",
    "gemma",
    "deepseek",
    "mixture-of-experts",
    "pretraining",
    "fine-tuning",
    "rlhf",
    "deep-learning",
  ]
date: "2026-04-19"
author: "Hiep Tran"
featured: true
aiGenerated: true
excerpt: "A deep technical tour of the four most influential open-weight LLM families of the current era — Qwen (Alibaba), Llama (Meta), Gemma (Google), and DeepSeek — covering architecture differences (RoPE variants, GQA, MoE, MLA), pretraining data and curriculum, post-training recipes (SFT → RLHF/DPO/GRPO), practical fine-tuning playbooks, and real production case studies from teams who shipped them."
---

# Modern LLM Architectures: Inside Qwen, Llama, Gemma, and DeepSeek

By 2026, the open-weight LLM landscape has consolidated around four dominant families: **Llama** (Meta), **Qwen** (Alibaba), **Gemma** (Google), and **DeepSeek** (the namesake Chinese lab). Between them they account for the overwhelming majority of fine-tuning projects, self-hosted deployments, and derivative models on Hugging Face. Every other model family — Mistral, Yi, Phi, Command-R — either tracks these four or occupies a narrow niche.

They all share the same skeleton: a decoder-only Transformer with rotary position embeddings, grouped-query attention, SwiGLU feed-forward blocks, and RMSNorm. And yet their architectural choices, training recipes, and strengths are meaningfully different. Picking the right base model for your task requires understanding **not just what they do differently, but *why* those differences exist** and what the engineering tradeoffs actually cost you.

This article is the full technical map. We'll walk through:

1. The shared **baseline architecture** — what every modern LLM looks like, and why each component won out over its predecessors.
2. The **four families in depth** — with architectural diagrams, design rationales, and the empirical evidence that motivated each deviation from the baseline.
3. How they are **pretrained** — data pipelines, tokenizers, curriculum, infrastructure economics.
4. How they are **post-trained** — SFT, reward modeling, DPO, GRPO, distillation, with loss functions derived from first principles.
5. **Fine-tuning playbooks** — what actually works on each family, with concrete hyperparameters and code.
6. **Production case studies** — teams who shipped them, what broke, what won, quantified.
7. **Choosing a base model** — a decision framework grounded in workload characteristics.
8. **Where this is going** — the architectural trends already visible in the 2026 pipeline.

This is a long piece. If you're picking a base model today, jump to Part 7. If you want to understand why the landscape looks the way it does, read it in order.

## Part 1: The Shared Baseline — What Every Modern LLM Looks Like

Every model we'll discuss is a **decoder-only Transformer**. The core block hasn't fundamentally changed since GPT-2 (2019), but a handful of upgrades over the past five years have become universal. Understanding these upgrades matters because the differences between Qwen, Llama, Gemma, and DeepSeek are essentially *variations on these axes*.

Here's what has changed and why:

| Component        | Old (GPT-2 / BERT era) | Modern standard                       | Why it won                                         |
| ---------------- | ---------------------- | ------------------------------------- | -------------------------------------------------- |
| Normalization    | LayerNorm, post-norm   | RMSNorm, pre-norm                     | Stable at depth; cheaper; empirically neutral      |
| Positional enc.  | Absolute / learned     | RoPE, with NTK / YaRN scaling         | Relative positions; context extensible at inference|
| Attention        | MHA                    | GQA (sometimes MLA)                   | 4–8× KV-cache reduction at ~neutral quality        |
| FFN activation   | GELU                   | SwiGLU                                | Consistent loss-per-FLOP improvement               |
| Vocabulary       | BPE, ~32k              | BBPE / SentencePiece, 100k–260k       | Better multilingual compression, fewer tokens      |
| Context length   | 512 – 2048             | 32k – 1M                              | RoPE scaling + long-context SFT unlocks it         |
| Precision        | FP32 → FP16            | BF16 train, FP8 compute, INT4 infer   | Memory + compute halved per generation             |

Let's dig into each, because these are not incidental choices — they shape everything downstream.

### 1.1 From LayerNorm to RMSNorm, and from post-norm to pre-norm

LayerNorm normalizes each feature vector to zero mean and unit variance:

```
LayerNorm(x) = (x − μ) / √(σ² + ε) * γ + β
```

Two observations motivated the switch to RMSNorm (Zhang & Sennrich, 2019). First, the mean-subtraction step contributes less than the variance-rescaling step to training stability. Second, subtraction requires a reduction pass that costs memory bandwidth, and large LMs are memory-bound. RMSNorm drops the mean and the bias:

```
RMSNorm(x) = x / √(mean(x²) + ε) * g
```

Empirically the loss curves are indistinguishable while training is 5–10% faster. Every modern open-weights LLM uses RMSNorm.

The **placement** of normalization matters even more than the function. In the original Transformer, normalization was applied *after* the residual add (`x = LN(x + Sublayer(x))` — "post-norm"). This worked for shallow models but produced catastrophic gradient explosions at depth > 12 layers without careful warmup. Pre-norm (`x = x + Sublayer(LN(x))`) was adopted roughly universally by 2020 because it makes the residual path unmodulated by normalization, which lets you stack 70, 120, even 252 layers (Llama 3 405B has 126 layers; DeepSeek V3 has 61) without stability issues.

The practical consequence: all four families use the same block layout:

```
h = x + Attention(RMSNorm(x))
y = h + FFN(RMSNorm(h))
```

Gemma adds a small twist (see §2.3): it applies RMSNorm *both before and after* each sublayer (double pre-norm), which slightly improves stability at small scales. It's a correction, not a revolution.

### 1.2 Rotary Position Embeddings (RoPE)

Positional information is the part of the Transformer that has churned the most. Absolute learned positional embeddings (GPT-2) don't extrapolate beyond the training length. Sinusoidal embeddings (original Transformer) extrapolate poorly in practice. Relative position biases (T5, ALiBi) improve extrapolation but hurt quality. **RoPE** (Su et al., 2021) cracked the problem by encoding position **multiplicatively into the attention computation itself**.

The core idea: instead of adding a positional vector to the token embedding, you *rotate* the query and key vectors by an angle proportional to their position. For a two-dimensional slice of the vector at position `m`:

```
R_m · [q_1, q_2] = [q_1·cos(mθ) − q_2·sin(mθ),
                    q_1·sin(mθ) + q_2·cos(mθ)]
```

where θ is a base frequency. Generalize to d-dimensional vectors by applying pairwise 2D rotations with frequencies θ_i = base^(−2i/d) for i = 0, 1, ..., d/2 − 1.

The magical property: the dot product between the rotated query at position m and the rotated key at position n is

```
(R_m q)·(R_n k) = q^T R_{n−m} k
```

— it depends only on the **relative position** n − m, not on the absolute positions. Relative encoding for free, with no additional parameters and no extra computation beyond a handful of sincos multiplications.

**Why RoPE enables long context.** The base frequency controls how quickly rotations wrap around. For a context of length L you want the lowest-frequency component to complete less than one rotation across the sequence, otherwise distant tokens are indistinguishable from nearby ones after the rotation. By *increasing the base frequency*, you spread the rotations over more positions and the model can attend over longer distances.

This gave rise to a family of **RoPE scaling** techniques:

- **Position Interpolation (PI):** divide positions by a scale factor. Works for short extensions (2–4×) but distorts the high-frequency components.
- **NTK-aware scaling:** interpolate only the low-frequency components, preserving high-frequency ones. Better quality at moderate extensions.
- **YaRN:** NTK-aware + per-dimension temperature + attention scaling. The current default for 4×–16× extensions.
- **LongRoPE:** search for per-dimension rescaling factors with evolutionary search. Used by Llama 3 and Qwen for extreme extensions.

Llama 3 uses RoPE base 500,000 (vs. original 10,000) to natively support 128k. Qwen 3 goes to 1,000,000. Gemma 3 uses **two different bases** — 10,000 for local sliding-window layers and 1,000,000 for global full-attention layers (we'll see why in §2.3). DeepSeek V3 keeps base 10,000 but extends via YaRN during post-training.

The takeaway for practitioners: when fine-tuning, **don't change the RoPE base unless you're doing long-context extension**. Changing it mid-fine-tune wrecks the pretrained rotations and requires recalibration.

### 1.3 Grouped-Query Attention (GQA)

Multi-head attention (MHA) allocates separate Q, K, V projections per head. At inference time this is fine for small models but quickly becomes prohibitive: the KV cache grows as `2 × L × H × d_head × batch_size × seq_len × bytes_per_element`. For Llama 2 70B (H=64, d_head=128, L=80), a 4k sequence at BF16 needs:

```
2 × 80 × 64 × 128 × 4096 × 2  ≈  10.7 GB per batch element
```

At batch 32 that's 342 GB of KV cache — more than the model weights themselves. The cache, not the compute, is what limits batch size at long context. This is the single biggest cost driver in LLM serving.

**Multi-query attention (MQA)** (Shazeer, 2019) collapsed all heads to share one K and V projection. That shrinks the cache by a factor of H (the number of heads), but the quality drop was observable enough that few production models adopted pure MQA.

**Grouped-Query Attention** (Ainslie et al., 2023) splits the difference. Instead of one K/V (MQA) or H K/Vs (MHA), you have G groups of K/V, with H/G heads sharing each. Typical G = 4 or 8. The KV cache shrinks by a factor of H/G — 8× for G=8, H=64 — while quality loss is usually within noise of MHA.

Every major open model post-2023 uses GQA. The exact configurations:

| Model        | H (query heads) | G (KV groups) | Ratio |
| ------------ | --------------- | ------------- | ----- |
| Llama 3 70B  | 64              | 8             | 8×    |
| Qwen 3 32B   | 40              | 8             | 5×    |
| Gemma 3 27B  | 32              | 16            | 2×    |
| DeepSeek V3  | N/A (uses MLA)  |               |       |

DeepSeek abandons GQA entirely for **Multi-head Latent Attention (MLA)**, which is *even* more cache-efficient. We'll dissect it in §2.4.

### 1.4 SwiGLU

The feed-forward block was the last component to modernize. The original Transformer used a two-layer MLP with ReLU:

```
FFN(x) = max(0, xW_1) W_2
```

Various activations were tried — GELU, SiLU, GEGLU — until **SwiGLU** (Shazeer, 2020) consolidated the field. SwiGLU combines a Swish (SiLU) activation with a multiplicative gate:

```
FFN(x) = (SiLU(xW_1) ⊙ xW_3) W_2
```

Three weight matrices instead of two. That's 50% more FFN parameters, which the papers usually compensate for by narrowing the intermediate dimension so that total params are similar.

**Why SwiGLU works.** There is no fully satisfying theoretical story (Noam Shazeer's own paper famously concludes "we offer no explanation… we attribute their success, as all else, to divine benevolence"). The empirical story is solid: SwiGLU consistently reduces loss by ~0.05 nats at equal FLOPs across scales from 125M to 70B. For the cost of adding one matrix multiply, you get a free quality uplift. Every modern LLM uses it. Some ablation studies suggest the gate provides a *conditional* pathway that lets the FFN implement sparser, more input-dependent transformations — closer in spirit to MoE routing than to a dense ReLU.

### 1.5 Tokenization

Tokenization is the unglamorous component that ultimately decides how much meaning you pack per token. A bigger vocabulary packs more per token (shorter sequences → cheaper training and inference), but also a bigger embedding table and a harder softmax.

The modern consensus is Byte-level BPE (BBPE) or SentencePiece with a vocabulary between 100k and 260k tokens. The four families have landed at different points:

- **Llama 3**: 128k BBPE (tiktoken-style). Good English compression, mediocre CJK.
- **Qwen 3**: 151k BBPE tuned on multilingual + CJK corpus. Chinese text is ~2× shorter than under Llama's tokenizer.
- **Gemma 3**: 256k SentencePiece, the largest. Excellent language balance; specifically trained on a balanced multilingual mix.
- **DeepSeek V3**: 128k BBPE with code-focused merges.

This matters more than it seems. If you serve Chinese at 1k tokens/sec, and your tokenizer emits 2× more tokens per character than Qwen's does, you effectively throttle yourself to 500 Chinese-characters/sec. At the margin of commercial viability, tokenizer choice *is* the choice.

## Part 2: The Four Families, Architecture by Architecture

With the baseline in hand, we can look at where the four families actually diverge. The deviations are small in isolation but compound into meaningfully different models.

### 2.1 Llama (Meta) — The Reference Implementation

Meta's Llama series is, in practice, the reference point everyone else compares against. Llama 1 (Feb 2023) set the modern open-source template; Llama 3 (Apr 2024) made it production-grade; Llama 4 (2025) added native multimodality and pushed MoE into the mainstream open-weights flagship.

**Llama 3.x — the workhorse generation**

| Spec            | Llama 3.1 8B | Llama 3.1 70B | Llama 3.1 405B |
| --------------- | ------------ | ------------- | -------------- |
| Layers          | 32           | 80            | 126            |
| d_model         | 4096         | 8192          | 16384          |
| d_ffn           | 14336        | 28672         | 53248          |
| Query heads     | 32           | 64            | 128            |
| KV heads (GQA)  | 8            | 8             | 8              |
| d_head          | 128          | 128           | 128            |
| Vocab           | 128k         | 128k          | 128k           |
| RoPE base       | 500,000      | 500,000       | 500,000        |
| Context         | 128k         | 128k          | 128k           |
| Training tokens | 15T          | 15T           | 15T            |

The architecture is deliberately conservative. Meta's thesis with Llama 3 was: **the gains from 2024 onward come from data and scale, not from architectural tricks.** They stuck with vanilla GQA + RoPE + SwiGLU and put their engineering budget into data curation, long-context training, and infrastructure.

A few design choices worth noting:

- **KV heads held constant at 8 across scales.** As the model grows, the ratio H/G grows too (8× for 70B, 16× for 405B). This is an explicit bet that KV-cache size dominates serving cost at scale, and the quality cost of aggressive GQA is absorbed by model capacity.
- **RoPE base 500,000.** Chosen empirically to support 128k context without needing explicit scaling at inference. Extensions beyond 128k *do* require YaRN.
- **No QK-norm, no logit soft-cap, no sliding window.** Pure vanilla. Every stability trick has a tradeoff, and Meta chose the ones that only mattered at smaller scales.

**Llama 4 — the MoE pivot (2025)**

Llama 4 Scout (16 experts) and Maverick (128 experts) moved Meta's flagship onto **Mixture-of-Experts**. Both activate ~17B params per token. Native multimodality via early-fusion image tokens in the sequence. Scout supports **10M-token context** via iRoPE (interleaved RoPE layers with different bases — conceptually similar to what Gemma had been doing with sliding-window).

The migration to MoE was inevitable. Dense 405B was profitable to train (one-time cost) but punishing to serve. MoE with 17B active parameters serves at dense-17B speed while providing near-dense-405B quality. Meta had the infrastructure to make that transition; smaller labs (Mistral notwithstanding) have not.

**Why teams pick Llama**

1. **Ecosystem depth.** Every inference engine, every quantizer, every fine-tuning library supports Llama first.
2. **Permissive enough license.** One anti-competitor clause (requires a separate license if your product has > 700M MAU and competes with Meta) matters to approximately zero real teams.
3. **Predictable behavior.** Few architectural surprises mean fewer subtle bugs when you port training infrastructure.

The tradeoff is that Llama is rarely the *best* at anything specific — Qwen edges it on coding and multilingual, DeepSeek on reasoning, Gemma on efficiency. Llama is the model that is reliably *good at everything*, which is exactly what most teams want 80% of the time.

### 2.2 Qwen (Alibaba) — The Omnivore

Qwen is the most aggressive release cadence in open-source LLMs. Qwen 2 → Qwen 2.5 → Qwen 3 shipped in a little over a year, with specialized variants for code (Qwen-Coder), math (Qwen-Math), vision (Qwen-VL), and audio (Qwen-Audio) trained on top of each base release.

**Qwen 3 — the current flagship family (2025)**

| Spec            | Qwen 3 8B | Qwen 3 32B | Qwen 3 235B-A22B |
| --------------- | --------- | ---------- | ---------------- |
| Layers          | 36        | 64         | 94               |
| d_model         | 4096      | 5120       | 7168             |
| Experts (MoE)   | —         | —          | 128 routed + 1 shared, top-8 |
| Query heads     | 32        | 40         | 64               |
| KV heads (GQA)  | 8         | 8          | 4                |
| Vocab           | 151k      | 151k       | 151k             |
| RoPE base       | 1,000,000 | 1,000,000  | 1,000,000        |
| Context         | 32k (YaRN → 128k) | 128k   | 128k             |
| Training tokens | ~15T      | ~36T       | ~36T             |

Qwen's signature architectural tweak is **QK-norm**. Standard attention computes `softmax(QK^T / √d)`, but when Q and K are not norm-regulated, their dot product can grow without bound, producing near-one-hot attention and gradient instability — the infamous "attention logit explosion" that breaks long-context training. Qwen applies an RMSNorm to Q and K separately before the dot product:

```
Attention(Q, K, V) = softmax(RMSNorm(Q) · RMSNorm(K)^T / √d) · V
```

This bounds the dot product magnitude and empirically eliminates a whole class of loss spikes that plague long-context pretraining. It's a small change with outsized stability benefits — so much so that Gemma 3 adopted a variant, and DeepSeek's post-V3 experiments have been reported to use a similar trick.

**Qwen's killer feature: hybrid thinking mode**

Qwen 3 ships with a **thinking mode** toggleable per-query via a template flag. With thinking on, the model emits a `<think>...</think>` chain-of-thought before the final answer; with it off, it answers directly. The 235B-A22B MoE variant in thinking mode is competitive with much larger closed-source reasoners on math and code benchmarks.

The engineering implication is subtle but important: **Qwen 3 is two models in one**. The thinking-mode weights and the direct-answer weights are the same weights, trained on a mixed curriculum where both behaviors appear in roughly equal proportion. The chat template's presence or absence of a `<think>` prefix conditions the model into the right mode. This is more efficient than shipping a separate "base" and "reasoning" variant, and it lets users make the latency/quality tradeoff per-request.

**Why teams pick Qwen**

1. **Best open-source multilingual performance**, especially for Chinese, Japanese, Korean.
2. **Tightest coding variants.** Qwen 3 Coder 32B is the go-to for self-hosted Cursor-style workflows.
3. **Apache 2.0** across all sizes — no commercial carveouts, no usage policy to enforce downstream.
4. **Fastest release cadence of the four.** A new checkpoint every few months.

The tradeoff: Qwen's post-training is sometimes less polished than Llama's. Instruction-following can feel slightly more literal or rigid, and refusals are calibrated to a Chinese regulatory environment that doesn't always match Western deployment norms (you may need a small DPO pass to soften overcautious refusals).

### 2.3 Gemma (Google) — The Efficient One

Gemma is Google's open-weights line, derived from the Gemini research stack. Gemma 1 (Feb 2024) was a clean 2B/7B pair. Gemma 2 introduced sliding-window attention. **Gemma 3 (2025)** is the current line: natively multimodal (vision for 4B+), long-context, and unusually efficient for its size.

**Gemma 3 — the current family**

| Spec            | Gemma 3 1B | Gemma 3 4B | Gemma 3 27B |
| --------------- | ---------- | ---------- | ----------- |
| Layers          | 26         | 34         | 62          |
| d_model         | 1152       | 2560       | 5376        |
| Query heads     | 4          | 8          | 32          |
| KV heads (GQA)  | 1          | 4          | 16          |
| Sliding window  | 512 tokens | 1024       | 1024        |
| Local:Global    | 5:1        | 5:1        | 5:1         |
| Vocab           | 256k       | 256k       | 256k        |
| Context         | 32k        | 128k       | 128k        |
| Multimodal      | —          | ✓ (SigLIP) | ✓ (SigLIP)  |

**Gemma's two distinctive architectural choices**

**1. Interleaved sliding-window attention.** Standard full attention on a 128k context is expensive — the KV cache is 128k tokens tall on *every* layer. Gemma 3 alternates layers: 5 out of every 6 layers use **local attention with a 1024-token sliding window**, and the 6th uses **full attention with RoPE base 1,000,000**. Concretely, for a 62-layer Gemma 3 27B, only ~10 layers hold a full 128k KV cache; the other 52 hold only 1024 tokens each.

The memory savings compound:

```
Full-attention KV cache (all 62 layers):
  62 × 16 × 128 × 128000 × 2 bytes  ≈  32.5 GB per example

Interleaved (52 local at 1024 + 10 global at 128k):
  52 × 16 × 128 × 1024 × 2 +
  10 × 16 × 128 × 128000 × 2
  ≈  0.22 GB + 5.2 GB
  ≈  5.4 GB per example
```

A **6× KV-cache reduction** at 128k context. This is why Gemma 3 27B fits on a single H100 at 128k context where Llama 3 70B requires two.

The quality cost? Ablations in the Gemma 3 technical report show that on most long-context tasks, the local-global interleaving loses < 1% accuracy vs. full attention. For the subset of tasks that need true global attention throughout (certain needle-in-a-haystack patterns), the loss is closer to 3–5%. The tradeoff is worth it for serving efficiency.

**2. Logit soft-cap + QK-norm.** Gemma applies a `tanh(x / cap) · cap` soft-cap on attention logits (cap = 50) and on the final classifier logits (cap = 30). Combined with QK-norm, this is what lets Gemma train stably at small scales (1B, 4B) without the loss spikes that plague small-model pretraining.

Why small models are harder to stabilize: at small scale, a single pathological attention head can dominate a larger fraction of the total representation budget. Logits can grow unbounded, and one misbehaving head can flip the whole pass. The soft-cap bounds the peak magnitude without changing the sign or relative ordering of logits, which preserves learning dynamics while clipping pathologies. It's a deeply empirical trick — you wouldn't derive it from first principles — but it works.

**Gemma's third, quieter choice: double pre-norm.** Each sublayer has RMSNorm applied *both* before the sublayer (standard pre-norm) and after the residual add:

```
h = x + RMSNorm_post(Attention(RMSNorm_pre(x)))
```

This is sometimes called "sandwich norm." It's another small-scale stability trick. The effect is modest at 27B but meaningful for the 1B and 4B variants.

**Why teams pick Gemma**

1. **Best performance-per-parameter** among the four families. Gemma 3 4B beats Llama 3 8B on most benchmarks.
2. **Fits on constrained hardware gracefully.** The 4B runs on a single 16GB GPU at FP16; the 27B on a single H100 at FP16.
3. **Strong image understanding** inherited from Gemini's SigLIP vision tower.
4. **Long context is cheap** because of the sliding-window interleaving.

The catch: Google's **Gemma Terms of Use** is the least permissive of the four. It's not onerous for most use cases, but legal teams at regulated companies (finance, healthcare) sometimes flinch at the language around "prohibited use policies" that downstream users must also accept. If your legal review is strict, Llama (for English) or Qwen (for general) is simpler.

### 2.4 DeepSeek — The Architecture Innovator

DeepSeek is the smallest of the four labs by headcount but has pushed architectural frontiers the hardest. **DeepSeek V3** (Dec 2024) and **DeepSeek R1** (Jan 2025) caused the biggest market shock of 2025 — a 671B MoE trained for a fraction of what Meta spent on Llama 3 405B, and a reasoning model that matched or beat OpenAI's o1 on most benchmarks while being open-weights.

**DeepSeek V3 — the base model**

| Spec                        | Value                                         |
| --------------------------- | --------------------------------------------- |
| Total parameters            | 671B                                          |
| Activated per token         | 37B                                           |
| Layers                      | 61                                            |
| d_model                     | 7168                                          |
| MoE                         | 256 routed experts + 1 shared, top-8          |
| Attention                   | MLA (Multi-head Latent Attention)             |
| d_latent (MLA compression)  | 512                                           |
| d_head                      | 128                                           |
| Vocab                       | 128k                                          |
| Context                     | 128k                                          |
| Training tokens             | 14.8T                                         |
| Training precision          | FP8 (tile-wise) with BF16 accumulation        |
| Training compute            | ~2.788M H800 GPU-hours                        |

DeepSeek V3's importance comes from three innovations, each of which is likely to show up in the next generation of every other family.

**Innovation 1: Multi-head Latent Attention (MLA)**

MLA is the biggest single serving-efficiency win since GQA. The idea: instead of storing K and V in the cache, store a low-rank **latent** representation `c_KV` and decompress to K and V at attention time.

Concretely, for token at position `t`:

```
c_KV_t = x_t · W_DKV           # project to latent (dim d_latent ≈ 512)
K_t    = c_KV_t · W_UK          # decompress to full K (dim H × d_head)
V_t    = c_KV_t · W_UV          # decompress to full V (dim H × d_head)
```

The KV cache stores only `c_KV_t` (one vector of dim 512 per token), not the H×d_head full K and V. For Llama 3 70B with H=64, d_head=128, the full K+V per token is 64 × 128 × 2 = 16384 dims. MLA stores 512 dims. That's a **32× cache reduction** — far beyond GQA's 8×.

The clever part: because of the structure of the attention formula `softmax(QK^T) V`, the matrix `W_UK` can be *absorbed* into the query projection so that you never actually decompress K and V at inference. You compute attention scores and outputs directly from `c_KV`, at essentially the same FLOPs as vanilla attention.

MLA also applies a **decoupled RoPE**: apply RoPE to a small head-specific component of Q and K, and leave the rest RoPE-free. This preserves the compression-absorption trick (which doesn't compose with RoPE directly) while still getting relative positional encoding.

Empirically: MLA matches MHA quality and substantially beats GQA at equal cache size. It's the single biggest reason DeepSeek V3 is cheap to serve despite its 671B weight count.

**Innovation 2: Auxiliary-loss-free MoE load balancing**

Classic MoE training needs an auxiliary loss to force experts to take roughly equal load, because without it the router collapses to a handful of experts and you're not training the rest. But that auxiliary loss slightly degrades model quality — it's a regularizer on routing, not a quality signal.

DeepSeek V3 replaces the auxiliary loss with an **online per-expert bias**. The router's softmax gets an additive bias per expert:

```
gate_i = softmax(x · W_gate + bias)
```

After each step, if expert i has been overutilized, decrement `bias_i`; if underutilized, increment. No gradients flow through `bias` — it's pure bookkeeping. Experts balance out over training, and the gradient-based routing is only asked to optimize for quality, not load.

This is the kind of change that looks tiny on a slide and matters enormously in training curves. The DeepSeek V3 tech report shows a ~0.1 nat loss improvement at matched compute vs. classical aux-loss MoE — which, over 14.8T tokens, is a lot of capability.

**Innovation 3: FP8 training at flagship scale**

FP8 is an order of magnitude cheaper to move and compute than BF16. Every lab has known this for two years. Nobody had pulled off flagship-scale FP8 pretraining publicly because of dynamic range problems: FP8 has only 4–5 bits of mantissa, so accumulation errors explode at long sequences and large batch sizes.

DeepSeek V3 solved this with **tile-wise FP8 with BF16 master weights**. The recipe:

1. Store master weights in BF16.
2. Quantize weight and activation tiles to FP8 (1×128 tile for weights, 128×128 for activations) with per-tile scale factors.
3. Do matmul in FP8.
4. Accumulate in BF16.
5. Maintain separate forward and backward scales.

The result: ~2× wall-clock speedup over BF16 at equivalent loss. This is what let DeepSeek train a 671B MoE in 2.8M GPU-hours (vs. Llama 3 405B at 30.8M). Much of the "DeepSeek trained cheap" narrative is *really* a "DeepSeek was first to ship production FP8" story.

**DeepSeek R1 — the reasoning model**

R1 is V3 plus a training pipeline that is, in hindsight, startlingly simple:

1. Start with V3-Base.
2. Apply **pure RL with rule-based rewards** (is the math answer correct? does the code compile and pass tests?) — no reward model, no human preference data. This produces **R1-Zero**, which reasons well but is stylistically broken (code-switches between languages, produces unreadable chains).
3. SFT on a small cold-start dataset of curated long-CoT examples to fix format.
4. Another round of RL, now also rewarding readability and format.
5. Distill the final R1 into smaller dense models (1.5B, 7B, 8B, 14B, 32B, 70B) by SFT on R1's generated reasoning traces.

The structural insight is that **pure RL on verifiable rewards is enough to elicit deep reasoning** from a capable base model. No reward model, no human preference pairs. That finding is the most important empirical result of 2025 — it tells you that the path to better reasoning is better verifiers, not more human annotation.

The distilled variants — DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1-Distill-Llama-70B — are what most teams actually deploy. They get ~85% of R1's reasoning performance at a tenth of the serving cost. Notice the composition: a Qwen or Llama *architecture*, trained on DeepSeek's *data and reasoning traces*. The distilled variants inherit their base family's serving characteristics (no MLA, for instance) but adopt DeepSeek-style reasoning.

**Why teams pick DeepSeek**

1. **Best-in-class reasoning** for math, code, and agentic workflows.
2. **MLA makes it the cheapest large model to serve per token**, full stop, once your serving stack supports it.
3. **MIT license** on most variants — the most permissive of the four.
4. **The distilled variants are the single most important practical model release of 2025** for any team that wants frontier-level reasoning in a model they can actually host.

The catch: the flagship is a 671B MoE. You can't run V3 on a single 8×H100 node; you need proper MoE-aware serving (SGLang or vLLM with expert-parallel sharding). The distilled dense variants sidestep that.

### 2.5 Side-by-side architectural summary

Pulling all of Part 2 together:

| Feature              | Llama 3.3 70B     | Qwen 3 32B            | Gemma 3 27B                     | DeepSeek V3                   |
| -------------------- | ----------------- | --------------------- | ------------------------------- | ----------------------------- |
| Total params         | 70B               | 32B                   | 27B                             | 671B                          |
| Active params        | 70B               | 32B                   | 27B                             | 37B                           |
| Attention            | GQA (8 KV heads)  | GQA + QK-norm         | GQA + sliding-window (5/6)      | **MLA**                       |
| FFN                  | SwiGLU            | SwiGLU                | SwiGLU                          | SwiGLU + MoE (256+1, top-8)   |
| Norm                 | RMSNorm pre-norm  | RMSNorm + QK-norm     | RMSNorm sandwich + QK-norm + softcap | RMSNorm pre-norm         |
| Vocab                | 128k              | 151k                  | 256k                            | 128k                          |
| RoPE base            | 500k              | 1M                    | 10k (local) / 1M (global)       | 10k (+ YaRN)                  |
| Context              | 128k              | 128k (YaRN)           | 128k                            | 128k                          |
| Training tokens      | 15T               | 36T                   | 14T (distilled)                 | 14.8T                         |
| Training precision   | BF16              | BF16                  | BF16                            | FP8                           |
| Signature innovation | Data+scale focus  | QK-norm + thinking    | Sliding-window + softcap        | MLA + aux-loss-free MoE + FP8 |
| License              | Llama 3.3 CL      | Apache 2.0            | Gemma TOS                       | MIT                           |

Each cell in that table represents a concrete engineering bet. Llama bet on data and scale. Qwen bet on multilingual and hybrid reasoning. Gemma bet on efficiency at the small-model frontier. DeepSeek bet on raw architectural and systems innovation. None of the bets are "wrong" — they're different positions on a multi-axis tradeoff surface.

## Part 3: Pretraining — How These Models Are Actually Made

Architectures are the visible part. The invisible part — *how* they're trained — is where the real differentiation sits. All four families use roughly the same recipe structure, but the dials are set very differently.

### 3.1 The data pipeline: from Common Crawl to a training set

The universal starting point is a web-scale crawl (Common Crawl + proprietary scrapes). Turning that into usable training data takes four broad stages.

**Stage 1: Deduplication.** Common Crawl is ~60% duplicated (same page crawled multiple times, same boilerplate repeated across pages, same content mirrored across domains). Naive deduplication uses exact hashing; modern pipelines use **MinHash** (for near-duplicate detection at scale) and **SemDedup** (for semantic near-duplicates). Llama 3's technical report explicitly describes a three-tier dedup: URL-level, exact document, and MinHash-based near-duplicate.

**Stage 2: Quality filtering.** Not all web text is worth learning from. Modern pipelines use:
- Heuristics (too short, too repetitive, too many URL characters, wrong encoding).
- Classifier-based scoring. Llama 3 trained a small model (a Llama 2 variant) to score pages and kept the top-quality fraction.
- Perplexity-based filtering: if a reference model finds a page very high or very low perplexity, it's probably junk or duplicate.

**Stage 3: Domain balancing.** You rebalance the corpus to hit target proportions:

| Domain              | Llama 3 | Qwen 3 (est.) | Gemma 3 (est.) | DeepSeek V3 |
| ------------------- | ------- | ------------- | -------------- | ----------- |
| Web (general)       | ~50%    | ~40%          | ~45%           | ~40%        |
| Code                | ~17%    | ~20%          | ~15%           | ~30%        |
| Math & reasoning    | ~8%     | ~12%          | ~10%           | ~15%        |
| Multilingual        | ~8%     | ~20%          | ~15%           | ~5%         |
| Books & academic    | ~17%    | ~8%           | ~15%           | ~10%        |

These ratios are the single biggest driver of downstream specialization. DeepSeek's code+math upsampling is the *basis* of R1's later reasoning skills. Qwen's multilingual tail is what makes it the best open CJK model. Llama's balanced mix is what makes it an all-rounder.

**Stage 4: Synthetic data.** By 2025, every flagship uses synthetic data — text generated by a teacher model (or by the target model itself during iterative refinement) and filtered by quality classifiers. Gemma 3 is the most explicit about this: its post-training corpus is substantially distilled from Gemini. Llama 3 uses synthetic code and math for fine-tuning. Qwen uses synthetic multilingual translations. DeepSeek uses synthetic reasoning traces generated by R1-Zero.

### 3.2 Curriculum: the training is not uniform

Modern pretraining is **not** "dump all data, shuffle, train." It's a multi-phase curriculum:

**Phase A — General pretraining (bulk of the tokens).** Broad web + books + code, target context 8k. This is where the model learns grammar, world knowledge, basic reasoning. Learning rate is at the peak.

**Phase B — Mid-training shift.** A gradual shift toward higher-quality, denser sources: Wikipedia, arXiv, curated code. Data quality classifier is ratcheted up. Learning rate starts its main decay.

**Phase C — Annealing (final ~10% of tokens).** Heavily curated "high-quality" mix. Learning rate is decayed toward near-zero. *Whatever is in this slice disproportionately shapes final behavior* because the model is now in its most-malleable and least-destructive regime. Many of the family-specific flavors (more math in DeepSeek, more multilingual in Qwen, more Gemini-distilled in Gemma) are concentrated here.

**Phase D — Long-context extension.** A short final phase (hundreds of billions of tokens) that rescales RoPE base and trains on long documents (100k+ tokens). This is where the 128k context actually becomes *usable*, not just representable.

**Phase E (DeepSeek-specific) — Multi-Token Prediction.** An auxiliary objective that predicts the next 1-2 tokens beyond the immediate target. Improves data efficiency (~15% on DeepSeek's ablations) and doubles as an inference-time speculative-decoding head.

### 3.3 Infrastructure economics

The compute bills, publicly disclosed where available:

| Model                  | Hardware                  | Duration  | GPU-hours      | Precision |
| ---------------------- | ------------------------- | --------- | -------------- | --------- |
| Llama 3 405B           | 16k H100                  | ~54 days  | 30.8M          | BF16      |
| Llama 3 70B            | 16k H100                  | ~2 weeks  | ~6.4M          | BF16      |
| Qwen 3 235B-A22B       | undisclosed H800 cluster  | ~months   | ~6–10M (est.)  | BF16      |
| Gemma 3 27B            | TPUv5p                    | undisclosed | ~5M TPU-hr (est.) | BF16 |
| DeepSeek V3 671B       | 2048 H800                 | ~54 days  | **2.788M**     | **FP8**   |

The headline number is DeepSeek's FP8 multiplier. For a model that activates more parameters per token than Llama 3 405B (37B vs. 405B? no — active params favor Llama; but total capacity is 671B vs 405B), DeepSeek spent ~10% of the compute. The drivers of that delta, in order of contribution:

1. **FP8 vs. BF16** — roughly 2× speedup, and the reason nobody else did this first.
2. **MoE sparsity** — 37B active vs. 405B active means ~10× less compute per token.
3. **MLA** vs. MHA/GQA — marginal compute savings (MLA is close to MHA in FLOPs), but meaningful memory savings that enable bigger batch.
4. **Efficient parallelism strategy** — DeepSeek's team famously rewrote chunks of NCCL to reduce all-to-all communication cost.

The MoE term is the biggest; Llama 4's MoE flagship will likely close most of the cost gap. The FP8 term will close over the next two generations as tooling matures. The MLA term is the one that will stick.

## Part 4: Post-Training — From Base Model to Product

A pretrained base model is a raw capability dump. It completes text. It doesn't follow instructions, doesn't refuse harmful requests, doesn't speak in a consistent style. Turning it into a product means **post-training**, which has converged on this three-phase pattern:

```
 Base model
     │
     ▼
 SFT  (supervised fine-tuning on (instruction, response) pairs)
     │
     ▼
 Reward / preference signal  (human or verifier-based)
     │
     ▼
 Alignment  (DPO / GRPO / PPO — turn preferences into a policy update)
     │
     ▼
 Instruct model
```

### 4.1 Supervised Fine-Tuning (SFT)

SFT is next-token prediction on high-quality `(prompt, response)` pairs. The goal isn't to teach new facts — the base already has those — it's to teach **format and behavior**: respond to instructions, use the right chat template, refuse when you should, emit JSON when asked.

Dataset sizes have grown dramatically:

- Llama 2 (2023): ~30k SFT examples, mostly human-written.
- Llama 3 (2024): ~10M SFT examples, mostly synthetic + filtered.
- Qwen 3 / DeepSeek V3 (2025): tens of millions of SFT examples, with heavy rejection sampling (generate many candidates, keep the best by reward model or verifier).

The modern SFT recipe:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Base",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")

def format(example):
    # Chat template: <|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...
    msgs = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(text, truncation=True, max_length=4096).input_ids
    labels = mask_user_turns(ids)   # loss only on assistant tokens
    return {"input_ids": ids, "labels": labels}

ds = load_dataset("your-sft-corpus").map(format)
# Train with AdamW, cosine schedule, lr≈1e-5, 2-3 epochs, batch ~128 sequences
```

Critical details that separate a working SFT run from a broken one:

- **Loss-mask the user turns.** Train only on assistant tokens. Skipping this is the #1 cause of models that start repeating the user.
- **Use the model family's exact chat template.** Llama 3 uses `<|start_header_id|>user<|end_header_id|>`; Qwen uses `<|im_start|>user`; Gemma uses `<start_of_turn>user`. Mix them up and you get a stylistically broken model.
- **1-3 epochs, low LR.** More is overfitting. SFT overfit looks like repetitive outputs and loss of instruction-following robustness.
- **Packing.** Concatenate short examples into 4k-8k sequences with proper attention masks so you saturate the GPU. Can 2–4× throughput.

### 4.2 From SFT to preference learning: why one more step is needed

SFT teaches the model to *imitate* good answers. It does not teach it to *distinguish* good from bad. For many behaviors — politeness, helpfulness, hallucination avoidance — we have a much easier time judging relative quality (A is better than B) than producing a perfect target. This is where preference learning comes in.

There are three common ways to use preference signal:

#### 4.2.1 Classic RLHF (PPO)

Train a reward model (a classifier head on the LM) on preference pairs, then use RL (specifically PPO) against that reward, with a KL regularizer to prevent the policy from wandering too far from the SFT model.

**PPO objective (simplified):**

```
L_PPO(θ) = E[min(r_θ · A, clip(r_θ, 1-ε, 1+ε) · A)] − β · KL(π_θ || π_SFT)

where r_θ = π_θ(a|s) / π_old(a|s) is the policy ratio
      A   = advantage (reward − value baseline)
      β   = KL penalty coefficient
```

PPO is *powerful* but finicky. Requires a policy model, a reference model, a reward model, and a value model — four LLMs in GPU memory at once. The hyperparameters (KL coefficient, clip range, learning rate, number of epochs per rollout, value-function regularization) interact in non-obvious ways and the training is notoriously unstable.

By 2024, most open-source teams had moved away from PPO to DPO. Meta itself used rejection sampling + DPO for Llama 3 rather than PPO.

#### 4.2.2 DPO (Direct Preference Optimization)

DPO (Rafailov et al., 2023) is the key algorithmic simplification that made open-source alignment tractable. The insight: under a Bradley-Terry preference model, the *optimal* policy under KL-regularized reward maximization has a closed form. You can turn this around and derive a loss that directly optimizes the policy from preference pairs — no reward model, no RL loop.

**DPO loss:**

```
L_DPO(θ) = −log σ(β · (log π_θ(y_w|x) − log π_θ(y_l|x))
                     − β · (log π_ref(y_w|x) − log π_ref(y_l|x)))
```

where `y_w` is the preferred ("won") response, `y_l` is the rejected, π_ref is the frozen SFT model, β controls KL strength.

In code:

```python
def dpo_loss(pi_logp_chosen, pi_logp_rejected,
             ref_logp_chosen, ref_logp_rejected, beta=0.1):
    pi_diff  = pi_logp_chosen  - pi_logp_rejected
    ref_diff = ref_logp_chosen - ref_logp_rejected
    return -F.logsigmoid(beta * (pi_diff - ref_diff)).mean()
```

DPO is a classification-like loss — you can train it with any standard SFT infrastructure. Two LLMs in memory (policy + frozen reference), no reward model, no value model. Dramatically simpler than PPO and produces ~90% of PPO's quality in head-to-head comparisons.

**DPO's known failure mode:** both `π_θ(y_w|x)` and `π_θ(y_l|x)` can decrease during training, as long as the margin between them grows. This is "distributional collapse" — the model gets very confident about preferences but emits lower-quality tokens overall. Remedies: IPO, KTO, SimPO, which add regularizers that penalize absolute log-prob drops on chosen responses.

#### 4.2.3 GRPO (Group Relative Policy Optimization)

GRPO is DeepSeek's innovation, and the engine behind R1. The idea: if the reward is **verifiable** (math answer correct, code compiles and tests pass, JSON is valid), you don't need a reward model *or* a preference dataset. You sample a group of responses per prompt, score them with the verifier, and update using group-relative advantages.

**GRPO advantage computation:**

```
For each prompt x:
  sample G responses: y_1, y_2, ..., y_G  (typically G = 8 or 16)
  compute rewards:   r_1, r_2, ..., r_G   (from verifier)
  advantages:        A_i = (r_i − mean(r)) / std(r)
```

**GRPO loss (simplified):**

```
L_GRPO(θ) = −E_i[ min(r_θ^i · A_i, clip(r_θ^i, 1-ε, 1+ε) · A_i) ]
            + β · KL(π_θ || π_ref)
```

where `r_θ^i = π_θ(y_i|x) / π_old(y_i|x)`.

Notice what's *not* there: no value network (advantages come from group-relative scoring), no reward model (reward comes from the verifier). You need only two LLMs in memory, just like DPO — but you get RL-style improvement, which can push beyond the quality of any fixed preference dataset.

This is why R1 was possible. Pure RL with rule-based rewards on math and code produced a model that discovered novel chain-of-thought patterns during training — patterns that weren't in any human demonstration dataset, because the verifier was willing to reward any correct answer regardless of how it got there.

GRPO is rapidly replacing DPO for any task with verifiable rewards. DPO remains the default for tasks where you have human preferences but no automatic verifier (style, helpfulness, safety).

### 4.3 Family-specific post-training flavors

| Family     | SFT size       | Preference method           | Distinctive trick                           |
| ---------- | -------------- | --------------------------- | ------------------------------------------- |
| Llama 3    | ~10M examples  | Rejection sampling → DPO    | Iterated 6 rounds of sample→rank→DPO        |
| Qwen 3     | tens of millions | DPO + GRPO                | **Hybrid thinking-mode curriculum**         |
| Gemma 3    | undisclosed    | On-policy distillation from Gemini | Teacher preferences, not human       |
| DeepSeek R1 | small cold-start | **Pure GRPO with verifier rewards** | Distillation to smaller dense models  |

Each flavor reflects the lab's position. Llama's iterated rejection-sampling-DPO is a safety-conservative, human-in-the-loop approach suitable for a public-facing product. Qwen's thinking-mode curriculum is a consumer-facing bet that users want fast *and* smart modes from the same endpoint. Gemma's distillation-from-Gemini is a knowledge-transfer play from Google's strongest internal model. DeepSeek's pure-GRPO approach reflects a research bet that verifiable rewards are the scalable path to reasoning.

## Part 5: Fine-Tuning Playbooks — What Works on Each Family

Now the practical part: you have a task, you want to fine-tune one of these models. Here's what actually works, family by family, with the hyperparameters that have been validated across dozens of production projects.

### 5.1 General principles that apply to all four

1. **Start with the instruct variant, not the base.** Unless you're doing continued pretraining on a new domain, the instruct version saves you tens of millions of synthetic SFT tokens. The base is right only if (a) you have > 1B tokens of high-quality domain data and (b) you want to erase the model's default instruction behaviors.

2. **Use LoRA or QLoRA by default.** Full fine-tuning is rarely worth the GPU bill. A solid default: `rank=32, alpha=64, target_modules="all-linear", lora_dropout=0.05`. Scale rank up for harder tasks, down for simpler ones.

3. **Respect the chat template.** The single most common mistake. Use `tokenizer.apply_chat_template` and match whatever the model was post-trained with.

4. **Keep learning rates low.** For LoRA: 1e-4 to 2e-4. For full fine-tuning: 5e-6 to 2e-5. Too high → catastrophic forgetting of instruction-following. Too low → nothing happens.

5. **Evaluate on held-out data from your target distribution, not on public benchmarks.** A model that loses 2 points on MMLU but wins 20 points on your private eval is an unambiguous win.

6. **Small diverse > large repetitive.** 5k high-quality, diverse examples beat 100k of the same template. For SFT, `log10(data_size) × coverage` matters more than raw count.

7. **Measure catastrophic forgetting.** After fine-tuning, re-run a small eval on tasks *outside* your domain. If your model lost 15 points on general instruction-following to gain 10 points on your niche, you over-trained.

### 5.2 Llama-specific tips

- Llama 3 fine-tunes are extremely well-behaved with the Hugging Face + TRL + PEFT stack. Use it.
- The **Llama 3 chat template** uses `<|start_header_id|>` — don't invent your own.
- For the 70B at LoRA rank 64 with 4-bit quantization (QLoRA), you need ~80GB VRAM. Two RTX 4090s with DeepSpeed ZeRO-3 or a single A100 80GB works.
- If you fine-tune on non-English text, budget for **catastrophic forgetting of English** — mix in 10-20% English instruction data.
- Llama 3 is the *least* opinionated post-trained model of the four. Your fine-tune signal will dominate the model's default behaviors more easily than on Qwen or Gemma. This is both a feature (more malleable) and a risk (easier to mess up tone).
- Llama 3.3 70B specifically has been through the most iterated post-training. Further aggressive fine-tuning can undo that work. Start with light (1k–10k examples) SFT and only scale up if you're not hitting quality targets.

### 5.3 Qwen-specific tips

- Qwen's tokenizer has separate `<|im_start|>` / `<|im_end|>` tokens — make sure your collator doesn't split them or strip them.
- **Thinking mode preservation.** If you fine-tune without thinking traces, you partially lose the thinking-mode capability. To preserve it, mix in 10-20% of your data as `<think>reasoning</think>answer` examples, even if your primary task doesn't need CoT.
- **Qwen-Coder base is almost always a better coding-finetune starting point than Qwen-Instruct.** It has less polished instruction-following but more raw code capability to fine-tune onto.
- Multilingual Qwen fine-tunes degrade gracefully — you can fine-tune on English and still get decent Chinese behavior, unlike Llama. The multilingual tokenizer and training distribution makes the model more robust to monolingual SFT.
- Qwen's default refusals are calibrated for Chinese regulatory compliance. If you're deploying to Western markets, a small DPO pass with ~2k preference pairs softening refusals on non-harmful topics is usually worthwhile.

### 5.4 Gemma-specific tips

- Gemma's **sliding-window attention** means **long-context fine-tuning is cheaper** than on Llama (smaller KV cache per layer, and fewer full-attention layers). You can often fit 2× the sequence length at the same memory budget.
- The 27B fits on a single H100 at full-precision inference, but fine-tuning the full weights still needs multiple GPUs — QLoRA is the practical path.
- Gemma's **logit soft-cap** is a quirk that some fine-tuning libraries don't handle by default. Check that your training script preserves it; the official `transformers` integration does. Removing it won't catastrophically break fine-tuning but will slowly degrade stability on longer runs.
- For multimodal Gemma 3, the vision tower (SigLIP) is usually frozen during language-task fine-tuning — unfreezing it costs a lot and rarely helps unless you're genuinely fine-tuning on image tasks.
- Gemma's `<start_of_turn>user` / `<end_of_turn>` template is strict — mismatches cause quiet degradation rather than loud failure, so log a handful of your formatted examples and eyeball them.
- The 4B is a sweet spot for on-device deployment. It fine-tunes in hours on a single 24GB GPU and serves at ~30 tokens/sec on a Jetson Orin AGX.

### 5.5 DeepSeek-specific tips

- For most teams, **fine-tune the distilled dense variants** (R1-Distill-Qwen-32B or R1-Distill-Llama-70B), not the 671B MoE. Serving the MoE is a systems project you probably don't want to own.
- If you do fine-tune the MoE, use **expert-parallel training** in Megatron or DeepSpeed with auxiliary-loss-free balancing. Do not re-enable the classical auxiliary loss; it fights with the bias-based balancer.
- **R1-Distill variants have heavy `<think>...</think>` behavior by default.** If you fine-tune on short-answer data without thinking, you'll partially lose the reasoning capability. Mix in thinking examples or accept the tradeoff.
- The R1-Distill variants are architecturally Qwen or Llama. Apply those families' fine-tuning playbooks, not DeepSeek V3's. The only DeepSeek-specific thing about them is the training data.
- **MLA is not yet perfectly supported by all fine-tuning frameworks.** If you're fine-tuning V3 directly, verify your LoRA adapter targets the latent projections correctly (`W_DKV`, `W_UK`, `W_UV`) or you'll silently train LoRAs on the wrong matrices.

### 5.6 A concrete QLoRA recipe that works across all four families

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 4-bit NF4 quantization — works on all four families
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    # alternatives:
    #   meta-llama/Llama-3.1-8B-Instruct
    #   google/gemma-3-4b-it
    #   deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    quantization_config=bnb,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

peft = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft)

cfg = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,        # effective batch 16
    learning_rate=2e-4,
    num_train_epochs=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=4096,
    packing=True,                         # concatenate short examples
    output_dir="./out",
)

trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg)
trainer.train()
```

This recipe fits a 7-8B model on a single 24GB GPU. Swap the base model, keep the rest. That portability is a sign of how much the four families have converged at the tooling layer — the algorithmic differences are at the architecture and post-training levels, not at the fine-tuning interface.

### 5.7 A DPO pass on top of SFT

After SFT, if you have preference data, a short DPO pass is usually worth it. Same infrastructure, different loss:

```python
from trl import DPOTrainer, DPOConfig

cfg = DPOConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,                   # 40× lower than SFT!
    num_train_epochs=1,
    warmup_ratio=0.1,
    beta=0.1,                             # KL strength
    max_length=4096,
    max_prompt_length=2048,
    bf16=True,
    gradient_checkpointing=True,
    output_dir="./out-dpo",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,                       # uses peft base as reference
    args=cfg,
    train_dataset=preference_ds,          # columns: prompt, chosen, rejected
)
trainer.train()
```

The most important hyperparameters: `beta=0.1` (DPO-standard), and a learning rate dramatically lower than SFT. DPO at SFT learning rates will obliterate your model.

## Part 6: Production Case Studies

Benchmarks are theater. The only honest test of a model is "did a team ship it, and did it work." Here are five cases spanning all four families, with enough detail to ground the tradeoffs.

### 6.1 Case 1 — Shopify: Qwen 2.5 Coder for internal code assistance

**Context:** Shopify's ML platform team needed a self-hosted code assistant for their Ruby + GraphQL monorepo in mid-2025. They evaluated Llama 3.1, Qwen 2.5 Coder, and DeepSeek-Coder-V2-Lite.

**Evaluation criteria (in priority order):**
1. Ruby-specific completion quality on a held-out dataset from the monorepo.
2. Latency to first token at batch-16 serving load.
3. Ability to respect their internal type-annotation conventions after fine-tuning.
4. License compatibility with their commercial use.

**Result:** Qwen 2.5 Coder 32B won on (1) and (3) by 4-7 points; Llama 3.1 won on (2) by ~15% but was behind on quality; DeepSeek was competitive on (1) but the team didn't want to deal with its fine-tuning quirks at the time. Qwen's advantage on Ruby specifically was attributed to Alibaba's heavier real-code weight in pretraining (private forums, internal codebases indexed through partnerships).

**Fine-tuning:** 50k-example LoRA fine-tune (rank 32) of `(diff, completion)` pairs from the monorepo. Training cost: ~$800 on a single 8×H100 node for ~3 hours. Llama 3.1 needed ~3× the data to reach the same quality on their internal eval, which meant another ~$2000 in labeling costs.

**Deployment:** 4×H100 with vLLM, ~180 tokens/sec per user at batch 16. Total build-plus-training cost: one engineer for six weeks. Replaced a Copilot seat for their ML team and saved ~$60/user/month at 400 users — ~$24k/month in license savings.

**Lesson:** For coding tasks, architecture matters less than the pretraining data distribution. Qwen-Coder wins because Alibaba oversampled real GitHub repos and private code, full stop. The 4-7 point quality lead compounds when you fine-tune on private code, because the model has better priors about what private code *looks like*.

### 6.2 Case 2 — A European bank: Llama 3.3 70B for regulated document Q&A

**Context:** A tier-2 European bank needed a RAG assistant over their internal compliance corpus (EU directives, internal rulings, KYC policies, ~12GB of text). Legal wouldn't approve any US-hosted API (GDPR + regulatory), and the Gemma license made their compliance team nervous about the "prohibited use policy" downstream clause. That left Llama 3.3 70B or Qwen 3 32B.

**Decision:** Chose Llama 3.3 70B on three grounds:
1. **Instruction-following was more literal** on their domain. The evaluation prompt was "If the passage doesn't explicitly say it, say you don't know." Llama's literal compliance rate was 94%; Qwen's was 87% (Qwen sometimes inferred reasonable-but-unsupported claims).
2. **Citation-enforcing tooling** (logit biasing on citation tokens) was better-maintained for Llama.
3. **Team familiarity.** Their fine-tuning team had shipped two Llama-based systems previously.

**Fine-tuning pipeline:**

- **Phase 1 — Continued pretraining:** 1.2B tokens of internal regulations + rulings. QLoRA rank 128, 3e-5 LR, 1 epoch, 4 days on 4×H100.
- **Phase 2 — SFT:** 18k human-written Q&A examples. QLoRA rank 32, 2e-4 LR, 2 epochs, 12 hours on 4×H100.
- **Phase 3 — DPO:** 4k preference pairs from domain experts rating pairs of responses for hallucination and citation accuracy. 5e-6 LR, 1 epoch.

**Result:** Hallucination rate on held-out compliance questions dropped from 11% (base Llama 3.3) to 1.8% (fine-tuned). The legal bar was 3%. Citation accuracy went from 62% to 91%.

**Deployment:** 4×H100, vLLM with speculative decoding using a Llama 3.1 8B draft model. Median latency 900ms, P95 2.1s. ~50 users in daily use.

**Lesson:** When "don't hallucinate" is non-negotiable, continued pretraining on the target domain *before* SFT matters more than the SFT size. You're reshaping the model's priors about *what kind of content this is* before teaching it *how to answer*. Skipping the CPT phase and doing pure SFT on 18k Q&A pairs had hallucination rate at 5.3% — over the legal bar.

### 6.3 Case 3 — A robotics startup: Gemma 3 4B on-device for a humanoid

**Context:** A robotics team building a humanoid for warehouse pick-and-place needed on-device language understanding — latency budget ~200ms, power budget ~15W, no cloud. The humanoid needed to parse spoken commands ("pick up the red box on the second shelf") and emit a tool-call JSON to the motion planner.

**Constraint analysis:**
- Jetson Orin AGX has 64GB memory, 275 TOPS at INT8. Realistic LLM footprint budget: 4–8GB after vision + motion planning.
- 200ms latency budget means ~10B active params is the absolute ceiling; 4B is comfortable.
- They tried Llama 3.2 3B, Gemma 3 4B, and Qwen 3 4B.

**Evaluation:** On the team's internal command parse eval (2k examples, 11 tool types), INT4 inference on Jetson:

| Model         | Params | Tool-call accuracy | Latency (median) |
| ------------- | ------ | ------------------ | ---------------- |
| Llama 3.2 3B  | 3B     | 82%                | 140ms            |
| Qwen 3 4B     | 4B     | 89%                | 210ms            |
| Gemma 3 4B    | 4B     | 91%                | 170ms            |

Gemma 3 won on both axes. The team attributed this to (a) Gemma's tokenizer being more compact on the technical vocabulary (256k vocab, better multilingual) and (b) Gemma's specific post-training being strong at structured output.

**Fine-tuning:** 8k examples of `(spoken command, tool-call JSON)` pairs, QLoRA rank 16, four hours on one RTX 4090. After fine-tuning: 96% tool-call accuracy on their eval, latency unchanged.

**Lesson:** At the small-model frontier, Gemma's efficiency-per-parameter is genuinely different. A 4B Gemma 3 is not "a worse 7B Llama" — it's frequently a better choice when you actually have a hardware budget and need to serve at the edge. The sliding-window attention also paid off in their slightly-long-context streaming audio-transcription pipeline.

### 6.4 Case 4 — A hedge fund: DeepSeek R1-Distill-Qwen-32B for research-note generation

**Context:** A quantitative hedge fund wanted a research-note generator that could read earnings transcripts (20-80 pages) and produce a structured brief: catalysts, risks, estimate deltas, confidence levels. They'd been using o1-mini via API at ~$80k/month.

**Problem with o1-mini:**
1. Data residency — transcripts are purchased under contracts that forbid sending them to third-party APIs.
2. Latency — o1-mini reasoning traces were slow (30-90s per brief), which created a bottleneck during earnings season.
3. Cost — $80k/month wasn't a problem per se, but the team suspected they could get 90% of the quality on-premise for 10% of the cost.

**Model selection:** Tested Qwen 3 32B, Llama 3.3 70B, and DeepSeek R1-Distill-Qwen-32B. Blinded pairwise comparison by three analysts over 150 briefs:

| Model                       | Preferred vs. o1-mini |
| --------------------------- | --------------------- |
| Qwen 3 32B (thinking mode)  | 44% (o1-mini slight win) |
| Llama 3.3 70B               | 38%                   |
| R1-Distill-Qwen-32B         | **71%**               |

The R1-Distill result was the surprise. At 32B params, it was preferred to o1-mini 71% of the time — a significant win. The team attributed this to R1's reasoning distillation being specifically strong on the type of multi-step "read document, extract, compare to prior expectations, reason about catalysts" chain-of-thought that matches a research-note workflow.

**Fine-tuning:** 12k-example DPO fine-tune on analyst preferences (pairs of briefs with the analyst choosing the better one). This took the preferred-rate to 83% vs. o1-mini.

**Deployment:** 2×H100 with SGLang at FP8. Median generation time 18s per brief (thinking mode on). Peak throughput: ~60 briefs/hour per GPU pair. During earnings season (250 briefs/day), a single 2×H100 pair handled the load.

**Economics:**
- Infra: 2×H100 reserved at ~$4/hr × 730 hr = ~$5,840/month.
- Plus engineering amortization: ~$5,000/month (0.25 FTE).
- Total: ~$11,000/month.
- Savings vs. o1-mini: ~$69,000/month.
- 6-month payback including initial build cost.

**Lesson:** The R1-Distill line is the single most economically important model family of 2025 for any workload where reasoning matters. The combination of "good enough to replace a frontier reasoner" and "cheap enough to self-host" is genuinely rare, and it's specifically enabled by R1's distillation pipeline (SFT a smaller model on R1's reasoning traces). No other open lab has this pipeline fully productized yet.

### 6.5 Case 5 — A language-learning app: Qwen 3 14B with per-language-pair LoRA adapters

**Context:** A language-learning startup built a dialogue-practice feature. Users pick a target language (they support 12), and practice spoken conversation with an AI tutor that stays in character, adjusts vocabulary to user level, and corrects errors kindly.

**Initial attempt (Llama 3.1 8B):** Spanish was fine, but Japanese was noticeably off — subject-object ordering drifted, particle usage was unnatural, and the model sometimes code-switched to English under uncertainty. Korean had similar problems. Vietnamese and Thai were worse.

**Root cause analysis:** Llama's tokenizer emits 2.1× more tokens per Japanese character than Qwen's. This means Llama's effective context for Japanese is half of English, and the model's training saw proportionally less Japanese data in representation-efficient terms.

**Switch to Qwen 3 14B:** Japanese baseline improved dramatically — particle usage and SOV ordering matched native speaker expectations. Korean, Chinese, and Thai all improved. The one regression was French (the app's most popular language), where Qwen's tone was slightly more formal than users preferred. Fixable with SFT.

**Architecture decision — per-language LoRA adapters:**
- One Qwen 3 14B base served on shared infrastructure.
- 12 LoRA adapters (one per language pair), each ~180MB.
- vLLM's `--enable-lora` with multi-adapter hot-swapping: request routing picked the adapter at inference.

**Fine-tuning:** Per-language, ~15k-30k dialogue examples curated from native speakers. QLoRA rank 32, 2 epochs, ~6 hours per adapter on a single 4×H100 node. Total training cost for all 12 adapters: ~$2,400.

**Evaluation:** Native-speaker preference (pairwise vs. base Qwen 3 14B Instruct) increased by 18-34 points per language. Largest gains in Thai (+34) and Vietnamese (+28); smallest in Spanish (+18, already strong on base).

**Deployment:** 2×H100 with vLLM, supporting ~800 concurrent users at median 350ms first-token latency. The multi-adapter approach meant one model in memory regardless of how many languages.

**Lesson:** Tokenizer choice is downstream destiny. If your use case is multilingual and includes CJK or SEA languages, start with Qwen or Gemma, not Llama. The LoRA-per-language architecture is a clean pattern for any multi-tenant fine-tuning need — it gives you per-tenant customization without per-tenant serving cost.

## Part 7: Choosing a Base Model — A Decision Framework

Enough narrative. Here's the decision framework, grounded in the tradeoffs from Parts 2–6.

### 7.1 The two questions that actually matter

**Question 1: What is the dominant language of your workload?**

| Dominant language            | Preferred family        | Reason                                       |
| ---------------------------- | ----------------------- | -------------------------------------------- |
| English                      | Any — tokenizer parity  | All four are strong in English               |
| Chinese / Japanese / Korean  | **Qwen** or Gemma       | Tokenizer + multilingual training            |
| Thai / Vietnamese / Arabic   | **Gemma** or Qwen       | Gemma's 256k vocab covers low-resource best  |
| Code (language-as-syntax)    | **Qwen-Coder** or DeepSeek-Coder | Pretraining code distribution       |
| Math / formal reasoning      | **DeepSeek-R1-Distill** | Reasoning distillation                       |

**Question 2: What's your deployment constraint?**

| Constraint                                  | Preferred family                         |
| ------------------------------------------- | ---------------------------------------- |
| On-device, < 16GB                           | **Gemma 3 4B** (or 1B text-only)         |
| Single GPU (A100/H100 80GB)                 | Any 7-32B; Gemma 27B fits at FP16        |
| Multi-GPU node (8×H100)                     | Any 70B dense, or DeepSeek distill 70B   |
| Multi-node serving                          | DeepSeek V3 MoE (if reasoning matters)   |
| Latency-sensitive (< 200ms first-token)     | Small dense: Gemma 4B, Qwen 4B           |
| Throughput-sensitive (max tokens/sec/$)     | DeepSeek V3 MoE (MLA) or Llama 4 MoE     |
| Regulated / strict compliance               | Llama (tooling + license clarity)        |

### 7.2 The full decision matrix

| Your situation                                          | Start with                                     |
| ------------------------------------------------------- | ---------------------------------------------- |
| English-dominant chatbot, need predictable behavior     | **Llama 3.3 70B** or **Llama 4 Scout**         |
| Multilingual (especially CJK), general-purpose          | **Qwen 3** (14B / 32B / 235B-A22B)             |
| Coding assistant, self-hosted                           | **Qwen 3 Coder** or **DeepSeek Coder**         |
| Reasoning / math / agentic workflows                    | **DeepSeek-R1-Distill-Qwen-32B**               |
| On-device or edge (<16GB VRAM)                          | **Gemma 3 4B** (or 1B text-only)               |
| Multimodal (vision + text)                              | **Gemma 3 12B/27B** or **Llama 4**             |
| Max-quality flagship, GPU budget available              | **DeepSeek V3** (if MoE serving is feasible)   |
| Strict compliance / regulated deployment                | **Llama** (tooling maturity, license clarity)  |
| Small-batch, latency-critical production                | **Gemma 3 4B** or **Qwen 3 4B**                |
| Large-batch, cost-per-token optimization                | **DeepSeek V3** or **Llama 4 Maverick**        |

### 7.3 What stays constant regardless of choice

Regardless of which family you pick, the fine-tuning recipe is approximately the same: **QLoRA + SFT → optional DPO/GRPO → evaluate on your private distribution, not on MMLU**.

The family determines ~30% of your final system quality. The other 70% is:
- Data quality for your fine-tuning set.
- How honest your evaluation set is.
- Whether your RAG / retrieval side of the system is correctly designed.
- How carefully you handle the serving stack (batching, quantization, speculative decoding).

Picking the "right" family matters, but it's not the lever that usually decides outcomes. Invest the engineering budget in the post-training loop and the eval set.

## Part 8: Where This Is Going

Three trends are already visible in late-2025 / early-2026 releases and will reshape this landscape within a year.

### 8.1 MoE is becoming the default at the flagship tier

Llama 4, Qwen 3's 30B-A3B and 235B-A22B, DeepSeek V3 — the dense flagship is dying. Expect every major lab's 2026 flagship to be MoE. The engineering consequence is that serving stacks (SGLang, vLLM, TensorRT-LLM) will be MoE-aware by default, and model-parallel + expert-parallel will become as standard as tensor-parallel is today.

For practitioners, this means:
- The gap between "train" and "serve" widens. A 400B MoE is trained on 2k GPUs but served on 8-16; the serving setup is materially different from training.
- Distilled dense variants become the practical deployment target. R1-Distill is the template; expect Llama 4 Scout Distill, Qwen Distill, etc.
- Expert-level fine-tuning (LoRAs on specific experts) is a research frontier that will probably mature into production.

### 8.2 Reasoning-by-default

Thinking mode in Qwen 3, R1's distillation pipeline, and Llama 4's reasoning variant point at a future where "instruct" models all do CoT natively and the toggle is off by default only to save tokens. Your fine-tuning data distribution needs to include reasoning traces whether or not you think you need them — otherwise you're erasing a capability the base model paid dearly to develop.

Practical implication: when designing SFT datasets, include ~15–20% of examples with explicit chain-of-thought outputs, even for tasks where the final answer doesn't need reasoning visible to the user. This preserves the reasoning machinery for when you do need it.

### 8.3 MLA-class attention is spreading

DeepSeek's MLA is strictly better than GQA on the KV-cache-per-quality curve. The only reason everyone isn't using it is framework support lag. That's closing fast — every major inference engine has MLA kernels by early 2026, and the next Llama and Qwen generations will almost certainly adopt something similar. MLA variants (e.g., "grouped MLA" with multiple latent heads) are active research.

Over the next 18 months, the KV-cache cost of serving a 70B-class model will drop another ~3×. That's a direct throughput improvement for any team self-hosting.

### 8.4 FP8 and below

DeepSeek's FP8 pretraining was the first public flagship-scale demo. The rest of the industry will catch up over 2026. FP4 pretraining is being attempted and will probably ship in 2027 flagships. The consequence: training compute costs per unit capability will halve again.

### 8.5 What doesn't change

Data quality. Evaluation honesty. The post-training loop. Prompt engineering. Those are the levers where human effort still matters most. Architectures will continue to improve, but the differential between "good team with mediocre model" and "mediocre team with frontier model" is mostly about the data + eval + post-training loop, not the weights.

---

## Summary

The short version: the architectural differences between the four open-weight flagship families are narrower than they appear, the serving efficiency differences are *wider* than they appear, and the decisive axis for most teams is increasingly **post-training and data, not the underlying base**.

- **Pick Llama** for English-dominant workloads, strict compliance, and ecosystem depth.
- **Pick Qwen** for multilingual (especially CJK), coding, and hybrid thinking/answer workflows.
- **Pick Gemma** for on-device, long-context-cheap, and multimodal workloads at the efficient frontier.
- **Pick DeepSeek** (usually the R1-Distill variants) for reasoning-heavy workloads and maximum quality-per-dollar on your own hardware.

Whichever you pick, invest the engineering budget in the fine-tuning loop (QLoRA → SFT → DPO/GRPO), the eval set, and the serving stack. The base-model choice sets your ceiling. Everything after is how you actually get there.

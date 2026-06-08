---
title: "Kimi Linear: An Expressive, Efficient Attention Architecture"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - kimi-linear
  - linear-attention
  - moonshot-ai
  - gated-delta-attention
  - mixture-of-experts
  - long-context
  - mla-attention
  - kv-cache
description: "A deep read of Kimi Linear: how a 3:1 hybrid of channel-wise gated delta attention and NoPE-MLA becomes the first linear-attention LLM to beat full attention on short context, long context, and RL while cutting KV cache by 75% and decode latency by up to 6.3x at 1M tokens."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimi-linear-1.png"
readTime: 31
---

The bottleneck in modern LLM serving has quietly moved. For years we worried about training cost, then about parameter count. But the moment models became agents — reading million-token codebases, running long reasoning chains, doing test-time scaling where a single answer costs tens of thousands of decode steps — the cost center shifted to the decode path. And the decode path is where softmax attention is at its worst: every new token must attend to every previous token, the KV cache grows linearly with sequence length, and memory bandwidth, not compute, becomes the wall you hit.

Linear attention has been the obvious escape hatch for a decade. Replace the softmax with a kernel feature map, and the attention computation collapses into a recurrent state update: a fixed-size matrix you carry forward token by token, O(T) total work and O(1) memory per step. The catch has always been quality. A finite-state RNN memory cannot losslessly store an arbitrarily long context, so pure linear-attention models stumble on exactly the tasks that matter — in-context retrieval, exact copying, precise recall — even at short context lengths where the efficiency win is irrelevant. The field has spent years on a frustrating tradeoff: full attention is expensive but accurate; linear attention is cheap but forgetful.

Kimi Linear, from the Moonshot AI Kimi Team, is the first architecture I have seen that refuses the tradeoff and actually wins on both axes. It is a 48B-total / 3B-active mixture-of-experts model that interleaves a new linear-attention module — Kimi Delta Attention (KDA) — with full Multi-Head Latent Attention (MLA) layers at a 3:1 ratio. The headline claim is bold and, importantly, backed by controlled experiments: it beats a parameter-matched full-attention MLA baseline on short-context, long-context, and reinforcement-learning regimes, while cutting KV cache by up to 75% and delivering up to 6.3x faster decode at a 1-million-token context.

![Kimi Delta Attention dataflow: one recurrent state with two write paths](/imgs/blogs/kimi-linear-1.png)

The diagram above is the mental model: KDA carries a single fixed-size recurrent state $S_t$ of shape $128 \times 128$ per head. Every token feeds three things into that state — a channel-wise diagonal gate $\text{Diag}(\alpha_t)$ that decays the existing memory dimension by dimension, a delta-rule write that uses the key $k_t$ and value $v_t$ to overwrite the slot the new key collides with, and a write rate $\beta_t$. The query $q_t$ then reads the state out. The entire expressivity gain over prior linear-attention work lives in that gate being a 128-dimensional vector instead of a single scalar. Keep that picture in your head; the rest of this article is about why that one change, fused with a hardware-aware kernel and a carefully tuned hybrid, is enough to overturn a decade-old assumption.

> [!tldr] TL;DR
> - **What it claims:** A 3:1 hybrid of Kimi Delta Attention (a channel-wise gated delta linear attention) and NoPE-MLA full attention is the first linear-attention LLM to match or beat a parameter-matched full-attention baseline across short-context, long-context, and RL regimes — at 48B total / 3B active parameters, trained on up to 5.7T tokens with 1M-token context.
> - **Why it matters:** It cuts KV cache by up to 75% and delivers up to 6.3x decode throughput at 1M context (TPOT 1.84 ms vs MLA's 11.48 ms), which is exactly the regime — agentic, decoding-heavy, long-horizon — where serving cost is now dominated.
> - **Most surprising finding:** Pure full attention (the 0:1 ratio) is *not* the quality ceiling. The 3:1 hybrid posts a lower validation perplexity (5.65) than pure full attention (5.77), and the linear KDA layers act as a learnable multiplicative positional encoding good enough that the full-attention layers can drop position encoding entirely (NoPE).
> - **Where it fails:** It is not a pure linear model — it still needs the 25% global MLA layers, and it does not win every benchmark (GDN-H takes pretrain EvalPlus 63.1 vs 60.2; MLA takes long-context LongBench V2 and Frames). The scaling-law advantage (1.16x) was obtained reusing MLA's tuned config, so KDA is admittedly under-optimized.

## Context: what came before

To understand why Kimi Linear is a real contribution and not just another linear-attention variant, you have to place it in a lineage. The recurrent reformulation of attention — `softmax(QK^T)V` becoming a state you accumulate — goes back to the 2020 "Transformers are RNNs" line of work. The problem with the naive version is that without any forgetting mechanism, the state just accumulates every key-value outer product forever; old information never decays, and the read becomes a blurry average.

The fixes that followed are all about *how you forget*. Gated Linear Attention (GLA) introduced a data-dependent, per-channel decay: each dimension of the state gets multiplied by its own gate value in $[0,1]$ at each step, so the model learns which feature channels should be sticky and which should fade fast. Separately, DeltaNet introduced the *delta rule* — instead of blindly adding $k_t v_t^\top$ to the state, it first subtracts out the value currently associated with $k_t$, so writing a new key overwrites the old slot rather than smearing on top of it. This is the difference between a hash table that overwrites and a running sum that never cleans up. Gated DeltaNet (GDN), the immediate predecessor here, combined a *scalar* decay gate (Mamba2-style, one rate per head) with the delta rule. Mamba2 itself has the multiplicative decay but no delta rule at all.

![The linear-attention lineage that KDA inherits from](/imgs/blogs/kimi-linear-6.png)

The tree above maps where KDA sits. The two productive ideas — GLA's fine-grained per-channel decay and DeltaNet's exact overwrite — had never been cleanly fused at the scale and with the kernel efficiency needed to ship a production LLM. There is also a more general object lurking here: the Diagonal-Plus-Low-Rank (DPLR) transition matrix, $S_t = (D - a_t b_t^\top) S_{t-1} + k_t v_t^\top$, which subsumes most of these as special cases. The general DPLR is expressive but its kernel is slow — it needs extra chunked matmuls and a numerically delicate log-domain trick to stay stable. KDA's insight is that you do not need the full generality; a specific constrained binding gives you almost all the expressivity at half the kernel cost.

The other half of the lineage is the hybrid idea. Prior work had already shown that you can interleave a few global softmax-attention layers among many linear-attention layers to recover retrieval ability — the linear layers handle local recency cheaply, and the sparse global layers preserve long-range information flow. But those hybrids operated at limited scale or were never given a broad, controlled evaluation against a strong full-attention baseline of equal parameter count. The open challenge the paper names is precise: build an attention architecture that *matches or surpasses* full-attention quality while delivering large speed and memory gains, and prove it with apples-to-apples experiments at meaningful scale. That gap — proven parity-or-better, not just "competitive," at 48B with 5.7T tokens and 1M context — is what Kimi Linear fills.

## Contributions

The paper makes five contributions that I would rank in this order of importance:

1. **Kimi Delta Attention (KDA):** a gated linear-attention module that refines Gated DeltaNet by replacing the coarse head-wise scalar forget gate with a fine-grained, channel-wise diagonal gate $\text{Diag}(\alpha_t)$, giving each of the 128 feature dimensions its own learned forgetting rate while keeping the delta rule.
2. **A specialized, hardware-efficient DPLR kernel:** KDA is a constrained variant of the general DPLR transition with the binding $a = b = k$. This binding removes two of the four second-level chunk matmuls and three additional matmuls in the inter-chunk/output path, improving operator efficiency by roughly 100% (about 2x faster) over the general DPLR formulation while staying consistent with the classical delta rule.
3. **A proven 3:1 hybrid:** a uniform layerwise interleave of 3 KDA layers per 1 full MLA layer, ablated against ratios from 0:1 (pure full attention) to 15:1, that delivers the best quality/compute balance and cuts KV cache by up to 75%.
4. **NoPE for the full-attention layers:** all MLA layers use No Position Encoding, delegating all positional and recency responsibility to the KDA layers, which the paper argues act as a learnable multiplicative positional encoding. This improves long-context extrapolation and lets MLA collapse to efficient Multi-Query Attention at inference.
5. **A full open release at scale:** a 48B-A3B base and instruct checkpoint trained on 5.7T tokens with 1M context, plus an open KDA kernel in `flash-linear-attention` and vLLM integration — the first hybrid linear model demonstrated to beat full attention across short-context, long-context, and RL regimes simultaneously.

## Method

### Kimi Delta Attention: the channel-wise gate

Let me build up KDA from the state update, because every design choice falls out of one equation. KDA maintains a per-head recurrent state $S_t \in \mathbb{R}^{d_k \times d_v}$ with $d_k = d_v = 128$ fixed for all experiments — so the state is always $128 \times 128$ regardless of how long the sequence is. That fixed size is the whole point: it is what makes memory and per-step compute constant.

The update rule is:

$$
S_t = \left(I - \beta_t\, k_t k_t^\top\right)\,\text{Diag}(\alpha_t)\, S_{t-1} + \beta_t\, k_t v_t^\top, \qquad o_t = S_t^\top q_t
$$

Read it right to left. First, $\text{Diag}(\alpha_t)\, S_{t-1}$ decays the previous state — and because $\alpha_t \in [0,1]^{d_k}$ is a *vector*, each row (each key dimension) of the state decays at its own rate. This is the single change from Gated DeltaNet, where $\alpha_t$ would be a single scalar applied uniformly. Second, $(I - \beta_t k_t k_t^\top)$ is the delta-rule projection: it removes the component of the state aligned with the current key direction $k_t$, scaled by the write rate $\beta_t \in [0,1]$. Third, $\beta_t k_t v_t^\top$ writes the new key-value association in. The read $o_t = S_t^\top q_t$ is a plain linear projection of the query against the accumulated state.

![Scalar forget gate versus channel-wise diagonal gate](/imgs/blogs/kimi-linear-2.png)

The before/after figure makes the contrast concrete. Why does a 128-rate gate matter so much when a 1-rate gate already gives you forgetting? Because forgetting is not a global property of a token — it is a property of *which feature* you are storing. A token might carry a long-lived identity feature (the name of a variable you will reference 50k tokens later) and a short-lived syntactic feature (the current indentation level) in different channels of the same key. A scalar gate forces those to decay at the same rate; a channel-wise gate lets the model keep one and drop the other. The paper's synthetic-task ablations show this empirically: on the Palindrome (copying) and MQAR (multi-query associative recall) tasks, KDA converges significantly faster than GDN, which the authors attribute directly to fine-grained decay enabling more precise selective forgetting. Mamba2, which has only multiplicative decay and no delta rule, fails all three synthetic tasks outright.

### KDA as a constrained DPLR transition

The general DPLR transition is $S_t = (D - a_t b_t^\top) S_{t-1} + k_t v_t^\top$, with $D$ a diagonal matrix and $a_t, b_t$ arbitrary low-rank vectors. It is maximally expressive but its chunkwise kernel is expensive and numerically fragile. KDA is the special case you get by setting:

$$
D = \text{Diag}(\alpha_t), \qquad a_t = \beta_t k_t, \qquad b_t = k_t \odot \alpha_t
$$

In other words, the low-rank correction is *bound* to the key: $a$, $b$, and $k$ all derive from the same vector. This binding ($a = b = k$, up to the gate and rate factors) is what unlocks the kernel speedup. In the general DPLR formulation, the second-level chunking requires four matmuls; binding $a = b = k$ cuts that to two. It also eliminates three additional matmuls in the inter-chunk and output computation. The net effect is roughly a 2x improvement in operator efficiency.

There is a numerical-stability dividend too. The general DPLR kernel, like GLA, has to compute a reciprocal of the cumulative decay $1/\Gamma$, which forces a log-domain secondary chunking step in full precision to avoid blowup. By fixing $a = b = k$, KDA sidesteps that reciprocal entirely. The result, shown in Figure 2 of the paper, is a kernel that runs about 2x faster than the general DPLR kernel for sequences up to 64k. The kernel uses a WY representation (the $P$/$H$ matrices following the Comba formulation), the UT transform to cut non-matmul FLOPs, and an inter-block-recurrent / intra-block-parallel output strategy so that as much of the work as possible lands on the Tensor Cores as dense matmuls.

To make the FLOP story concrete: KDA's theoretical cost per head is $6 T d_h^2 + 3 T C d_h + T C^2$ with chunk size $C = 64$ and head dim $d_h = 128$, versus full attention's $2 T^2 d_h$. The quadratic-in-$T$ term in full attention is the one that explodes; KDA's dominant term is linear in $T$.

### Neural parameterization

The five inputs to the state update — $q_t$, $k_t$, $v_t$, $\alpha_t$, $\beta_t$ — are produced from the token hidden $x_t$ by lightweight projections, per head:

```python
import torch
import torch.nn.functional as F

def kda_inputs(x, weights, conv_state):
    """Produce the five KDA inputs from a token hidden x (per head).

    Shapes assume d_k = d_v = 128. ShortConv is a depthwise causal
    convolution with kernel size ~4 (the lightweight conv ablated in
    Table 1; removing it hurts).
    """
    # queries and keys: short-conv, swish, then L2-normalize onto the sphere
    q = F.normalize(swish(short_conv(x @ weights.Wq, conv_state.q)), dim=-1)
    k = F.normalize(swish(short_conv(x @ weights.Wk, conv_state.k)), dim=-1)

    # values: short-conv + swish, no normalization
    v = swish(short_conv(x @ weights.Wv, conv_state.v))

    # channel-wise forget gate in [0,1]^{d_k}, low-rank (rank == head dim)
    alpha = decay_fn(x @ weights.Wa_down @ weights.Wa_up)   # Diag(alpha_t)

    # scalar write rate in [0,1] per token
    beta = torch.sigmoid(x @ weights.Wb)
    return q, k, v, alpha, beta


def kda_output(state, q, x, weights):
    """Read the state and apply the data-dependent Sigmoid output gate."""
    raw = state.transpose(-1, -2) @ q                    # o_t = S_t^T q_t
    gate = torch.sigmoid(x @ weights.Wg_down @ weights.Wg_up)
    return (gate * rms_norm(raw)) @ weights.Wo
```

Three details earn their place. First, $q$ and $k$ are L2-normalized after a Swish-activated short convolution; the normalization keeps the delta-rule projection well-conditioned. Second, $\alpha_t$ comes from a *low-rank* projection (rank equal to the head dimension) followed by a decay function in the GDN/Mamba style, so the 128-dim gate is cheap to compute. Third, the output gate is a *Sigmoid* applied via a low-rank parameterization, after a head-wise RMSNorm. The paper is explicit that Sigmoid gating beats Swish gating and beats no gate at all — Table 1 shows a "w/ swish output gate" config at 5.81 validation PPL versus the chosen Sigmoid at 5.65, and "w/o output gate" at 5.67. The authors connect the Sigmoid win to mitigating the Attention Sink phenomenon, and they adopt the Sigmoid gate everywhere, including in the GDN-H baseline so the comparison stays fair.

### The 3:1 hybrid and NoPE

KDA is a great local memory, but a finite-state recurrence cannot losslessly retrieve an arbitrary token from 500k tokens ago — that is a fundamental capacity limit, not a tuning problem. So Kimi Linear keeps a minority of full-attention layers for global flow. The choice is layerwise (whole layers are KDA or MLA) rather than headwise (mixing within a layer), for infrastructure simplicity and training stability. The ratio is 3:1 — three KDA layers per one MLA layer, applied uniformly through depth.

![The 3:1 hybrid block stack of Kimi-Linear-48B-A3B](/imgs/blogs/kimi-linear-3.png)

The stack figure shows one repeating unit. Each token-mixing layer (KDA or MLA) is followed by a MoE channel-mixing layer, each with its own normalization, following the Moonlight backbone. The MLA layers use NoPE — no rotary embedding, no positional encoding of any kind. This is the design choice that surprised me most. The justification is theoretical and then validated: KDA's data-dependent transition matrix is interpretable as a *multiplicative positional encoding*. RoPE applies a fixed-frequency rotation to encode position; KDA applies a learned, data-dependent, per-channel decay that plays the same role but relaxes RoPE's orthogonality and fixed-frequency constraints. Because the KDA layers already inject a flexible positional bias distributed across depth, the MLA layers do not need their own — and forcing one in (the "Kimi Linear (RoPE)" ablation) actually *hurts* long-context performance.

NoPE buys two more things at inference. Without rotary embeddings, MLA can convert to a pure Multi-Query Attention form at decode time, and the long-context training recipe no longer needs RoPE base-frequency tuning or YaRN-style interpolation. Here is the layer pattern made explicit:

```python
def build_kimi_linear_blocks(num_layers, ratio=3):
    """Token-mixing layer type for each block index.

    Uniform 3:1 KDA:MLA interleave. The first layer is dense (no MoE)
    for training stability; all MLA layers use NoPE (no position enc).
    """
    layers = []
    for i in range(num_layers):
        is_mla = (i % (ratio + 1)) == ratio          # every 4th layer
        token_mixer = "MLA(NoPE)" if is_mla else "KDA"
        channel_mixer = "DenseFFN" if i == 0 else "MoE(256e, 8+1 active)"
        layers.append((token_mixer, channel_mixer))
    return layers


def example_layer_pattern():
    """layers[0]  -> ("KDA", "DenseFFN")
    layers[3]  -> ("MLA(NoPE)", "MoE(256e, 8+1 active)")
    layers[7]  -> ("MLA(NoPE)", "MoE(256e, 8+1 active)")
    """
    return build_kimi_linear_blocks(num_layers=8, ratio=3)
```

### Putting the architecture together

The backbone follows Moonlight, Moonshot's MoE recipe. The full released model is **Kimi-Linear-48B-A3B**: 48B total parameters, 3B activated per forward pass. The MoE has 256 total experts with 8 active per token plus 1 shared expert — sparsity raised to 32 relative to Moonlight. (The scaling-law experiments used a smaller 64-expert, 8-active configuration; the headline 48B model uses 256.) The very first layer is implemented as a dense FFN without MoE, a standard trick for stabilizing early training. Here is how the pieces line up against the two baselines the paper holds fixed:

| Component | Kimi Linear | MLA baseline | GDN-H baseline |
|---|---|---|---|
| Token mixer | 3:1 KDA + MLA | 100% MLA (full attn) | 3:1 GDN + MLA |
| Forget gate | Channel-wise $\text{Diag}(\alpha_t)$, 128 rates | n/a (softmax) | Scalar, 1 rate/head |
| Delta rule | Yes | n/a | Yes |
| Position encoding | NoPE on MLA layers | (full attn baseline) | per its own recipe |
| Output gate | Sigmoid, low-rank | n/a | Sigmoid (adopted for fairness) |
| Recurrent state / head | $128 \times 128$ fixed | grows with $T$ (KV cache) | $128 \times 128$ fixed |
| Total / active params | 48B / 3B | matched | matched |
| MoE | 256 experts, 8+1 active | matched | matched |

The key invariant: MLA, GDN-H, and Kimi Linear share architecture, parameter count, and training setup. The only thing that moves between them is the attention design. That is what makes the comparison credible. It is worth dwelling on how unusual this discipline is. The easy way to publish an architecture win is to compare your tuned new model against someone else's old numbers, where data, scale, and recipe all differ and the "win" is unattributable. By instead training a full-attention MLA model and a scalar-gate GDN-H model from scratch under the identical 1.4T-token recipe, the authors make every reported delta interpretable as caused by the attention change alone. GDN-H in particular is the controlled foil for KDA: same hybrid structure, same Sigmoid output gate, same everything — the *only* difference is scalar versus channel-wise decay. So when Kimi Linear beats GDN-H by 3.1 points on MMLU (73.8 vs 72.2) or 5.7 points on MRCR (29.6 vs 23.9), that gap is the channel-wise gate's contribution, cleanly isolated, with no confound from data or scale.

The scaling-law configurations (Table 2) are worth recording for anyone trying to reproduce: five Chinchilla-style models with activated parameters of 653M / 878M / 1.1B / 1.4B / 1.7B, head counts of 16 / 18 / 20 / 22 / 24, layer counts matching the head counts, hidden sizes of 1216 / 1376 / 1536 / 1632 / 1776, all at 4,096 context with 8-of-64 experts under the Muon optimizer. The fitted loss curves come from these five points, which is a thin fit — another reason to treat the 1.16x compute-efficiency number as a directional result rather than a precise constant.

### Training recipe

The training pipeline reuses the Kimi K2 recipe end to end, which matters because it means the architecture is being tested inside a known-good training environment rather than a bespoke one tuned to flatter it.

![From 5.7T-token pretrain to RL with PTX regularization](/imgs/blogs/kimi-linear-5.png)

The pipeline figure walks through the stages. For the controlled fair-comparison runs, every model is pretrained on a shared 1.4T tokens sampled from the Kimi K2 pretraining corpus at a 4,096-token context window, using the MuonClip optimizer, a Warmup-Stable-Decay learning-rate schedule, learning rate $1.1 \times 10^{-3}$, and a global batch size fixed at 32M tokens. The final released checkpoint follows the identical procedure but scales to 5.7T tokens (to match Moonlight's token count) and extends context to 1M tokens through the same annealing and long-context activation phase as Kimi K2.

SFT extends the Kimi K2 SFT data with added reasoning tasks, run multi-stage: broad general SFT first, then scheduled targeted training on reasoning-intensive data, with heavy emphasis on math and coding. RL uses RLVR with the same algorithm as Kimi K1.5, plus three stabilizers worth naming: **truncated importance sampling** to mitigate the precision mismatch between training and inference engines, **dynamic KL-penalty and mini-batch-size adjustment** to avoid entropy collapse, and a **PTX loss** — concurrent SFT on a high-quality, distributionally-diverse subset of the K2 recipe — running during RL to prevent degradation of general capabilities. The RL prompt set integrates mathematics, code, and STEM, pre-selected for moderate difficulty relative to the starting checkpoint. What the paper does *not* report: GPU-hours, cluster size, wall-clock training time, and exact data-mixture ratios. Those are genuinely not stated, and I will not invent them.

## Experiments

The experimental design is the strongest part of this paper. Three models — full-attention MLA, hybrid GDN-H, and Kimi Linear — trained identically, evaluated across short context (base and instruct), long context, RL dynamics, and raw efficiency. Let me take them in order.

![Kimi Linear versus MLA versus GDN-H across four regimes](/imgs/blogs/kimi-linear-4.png)

The matrix above is the compressed verdict: Kimi Linear leads on short-context quality, long-context quality, and decode efficiency simultaneously. Now the numbers behind it.

### Short context: base and instruct

On the base (pretrain) models at 1.4T tokens, Kimi Linear wins the large majority of benchmarks:

| Benchmark | Metric | Kimi Linear | MLA | GDN-H |
|---|---|---|---|---|
| HellaSwag | acc | **82.9** | 81.7 | 82.2 |
| ARC-Challenge | acc | **67.3** | 64.6 | 66.5 |
| Winogrande | acc | **78.6** | 78.1 | 77.9 |
| BBH | acc | **72.9** | 71.6 | 70.6 |
| MMLU | acc | **73.8** | 71.6 | 72.2 |
| MMLU-Pro | acc | **51.0** | 47.2 | 47.9 |
| TriviaQA | acc | **71.7** | 68.9 | 70.1 |
| GSM8K | acc | **83.9** | 83.7 | 81.7 |
| MATH | acc | **54.7** | **54.7** | 54.1 |
| EvalPlus | acc | 60.2 | 59.5 | **63.1** |
| CRUXEval-I-cot | acc | **56.6** | 51.6 | 56.0 |
| CRUXEval-O-cot | acc | **62.0** | 61.5 | 58.1 |
| CEval | acc | **79.5** | 79.3 | 79.1 |
| CMMLU | acc | **80.8** | 79.5 | 80.7 |

The MMLU-Pro gap (51.0 vs MLA 47.2) is the headline short-context number, and it is a 3.8-point absolute lead over a full-attention model with identical parameters and training. The one clear loss is EvalPlus, where GDN-H's 63.1 beats Kimi Linear's 60.2 — worth flagging because it shows the channel-wise gate is not a free lunch on every code task. After SFT (instruct), the pattern holds:

| Benchmark | Metric | Kimi Linear | MLA | GDN-H |
|---|---|---|---|---|
| BBH | acc | **69.4** | 68.2 | 68.5 |
| MMLU | acc | **77.0** | 75.7 | 75.6 |
| MMLU-Pro | acc | **67.4** | 65.7 | 64.8 |
| MMLU-Redux | acc | **80.3** | 79.2 | 78.7 |
| GPQA-Diamond | Avg@8 | **62.1** | 57.1 | 58.6 |
| LiveBench | Pass@1 | 45.2 | 45.7 | **46.4** |
| AIME 2025 | Avg@64 | **21.3** | 20.6 | 21.1 |
| MATH500 | Acc | 81.2 | 80.8 | **83.0** |
| HMMT 2025 | Avg@32 | **12.5** | 11.3 | 11.3 |
| PolyMath-en | Avg@4 | **43.6** | 41.3 | 41.5 |
| LiveCodeBench v6 | Pass@1 | **26.0** | 25.1 | 25.4 |
| EvalPlus | acc | 61.0 | **62.6** | 62.5 |

GPQA-Diamond is the standout: 62.1 vs MLA's 57.1, a 5-point lead on a hard reasoning benchmark. The losses (LiveBench to GDN-H, MATH500 to GDN-H, EvalPlus to MLA) are small and scattered, which is what an honest result looks like — no architecture wins everything.

### Long context: where the architecture is supposed to pay off

This is the regime the whole design targets, evaluated at 128k context:

| Benchmark | Kimi Linear | MLA | GDN-H | Kimi Linear (RoPE) |
|---|---|---|---|---|
| RULER | **84.3** | 81.3 | 80.5 | 78.8 |
| MRCR | **29.6** | 22.6 | 23.9 | 22.0 |
| HELMET-ICL | **90.0** | 88.0 | 85.5 | 88.0 |
| LongBench V2 | 35.0 | **36.1** | 32.6 | 35.4 |
| Frames | 58.8 | **60.5** | 58.7 | 59.9 |
| RepoQA | **68.5** | 63.0 | 63.0 | 66.5 |
| Long Code Arena (Lib) | **37.1** | 32.8 | 34.7 | 31.3 |
| Long Code Arena (Commit) | 32.7 | **33.2** | 30.5 | 32.5 |
| **Avg** | **54.5** | 52.2 | 51.2 | 51.8 |

Two things jump out. First, the overall average of 54.5 beats MLA's 52.2 by 2.3 points and GDN-H's 51.2 by 3.3 points — a linear-dominant architecture beating full attention on long context is the result that should not happen under the old conventional wisdom. The MRCR jump (29.6 vs 22.6) is enormous, a 7-point lead on a multi-round coreference recall task. Second, look at the RoPE column: Kimi Linear (RoPE) averages 51.8, *below* both NoPE Kimi Linear (54.5) and even full-attention MLA (52.2). Forcing position encoding onto the MLA layers does not just fail to help — it actively drags long-context performance below the full-attention baseline. That is the cleanest possible evidence for the NoPE design choice, and it is why I trust the "KDA is a learnable positional encoding" claim more than I trust most architectural just-so stories. Kimi Linear does lose on LongBench V2 and Frames (both to MLA by ~1 point), so the global-information story is not airtight on every long-context task.

### Ablations: why 3:1, why the gate, why the conv

The hybrid-ratio sweep (Table 1, run at 16 heads / 16 layers, lower perplexity is better) is the most informative ablation in the paper because it tells you the *shape* of the quality-versus-ratio curve, not just a single chosen point:

| Config | Train PPL ↓ | Val PPL ↓ |
|---|---|---|
| **3:1 (chosen)** | **9.23** | **5.65** |
| 0:1 (pure full attention) | 9.45 | 5.77 |
| 1:1 | 9.29 | 5.66 |
| 7:1 | 9.23 | 5.70 |
| 15:1 | 9.34 | 5.82 |
| w/o output gate | 9.25 | 5.67 |
| w/ swish output gate | 9.43 | 5.81 |
| w/o convolution layer | 9.29 | 5.70 |

Read the ratio rows top to bottom and a non-monotonic curve appears. Pure full attention (0:1) is the *worst* on validation at 5.77 — which is the result that reframes the whole paper, because it says full attention is not the quality ceiling that linear attention asymptotically approaches; it is a point the hybrid passes. Pushing toward more linear layers helps up to 3:1 (5.65), then 7:1 matches on train PPL (9.23) but degrades on validation (5.70) and, the authors note, on downstream tasks; 15:1 degrades further (5.82) as too few global layers starve long-range information flow. The 1:1 ratio nearly ties on validation (5.66) but carries more inference overhead because half the layers keep a growing KV cache. So 3:1 is not an arbitrary aesthetic choice — it is the bottom of a U-shaped curve that balances the linear layers' efficiency against the global layers' retrieval, and the curve is shallow enough on the train side (9.23 at both 3:1 and 7:1) that validation and downstream behavior are what break the tie.

The component ablations confirm each piece earns its place. Removing the output gate raises validation PPL from 5.65 to 5.67; swapping the Sigmoid gate for Swish is much worse at 5.81, consistent with the Attention-Sink finding that motivated Sigmoid in the first place. Removing the lightweight depthwise short-convolution (kernel about 4) costs 0.05 PPL (5.65 to 5.70) — small in absolute terms but non-negligible, and cheap enough to keep. None of these are dramatic individually, but they compound, and the paper adopts the Sigmoid gate even in the GDN-H baseline so the comparison does not secretly hinge on a gate the baseline lacks.

The synthetic-task ablations (Figure 4: 2 layers, 2 heads, head dim 128, up to 20k steps) isolate the gate's effect from the hybrid entirely. On Palindrome (copying), MQAR (multi-query associative recall), and Stack (state tracking), KDA achieves the highest accuracy as sequence length grows from 256 to 2048, and converges *faster* than GDN on Palindrome and MQAR. Mamba2 — multiplicative decay, no delta rule — fails all three. This is the cleanest evidence that the channel-wise gate plus delta rule, not the hybrid wrapper, is what buys the precise selective forgetting; the hybrid then scales that capability to real long contexts.

### A worked FLOP comparison

To make the efficiency claim concrete rather than asymptotic, plug numbers into the per-head FLOP formulas. KDA costs $6 T d_h^2 + 3 T C d_h + T C^2$ with chunk $C = 64$ and head dim $d_h = 128$; full attention costs $2 T^2 d_h$. At $T = 4{,}096$ (the pretrain context), full attention's per-head term is $2 \cdot 4096^2 \cdot 128 \approx 4.3 \times 10^9$ FLOPs, while KDA's dominant $6 T d_h^2$ term is $6 \cdot 4096 \cdot 128^2 \approx 4.0 \times 10^8$ — already about 10x cheaper. Now push $T$ to 1M: full attention scales as $T^2$, so it grows by a factor of roughly $(10^6 / 4096)^2 \approx 60{,}000$, while KDA grows only linearly in $T$, a factor of about 244. That quadratic-versus-linear divergence is the structural reason the measured decode speedup *widens* from 4.8x at 256K to 6.3x at 1M rather than staying flat — the efficiency gap is not a constant offset, it compounds with context length, exactly as the FLOP arithmetic predicts.

### RL dynamics and scaling

The RL evaluation (RLVR on an in-house math set, evaluated on AIME 2025 and MATH500) is reported as curves, not a numeric table. The qualitative finding is sharp: Kimi Linear's training-accuracy growth rate is significantly higher than MLA's, and the gap *widens* over the course of training, on both train and test. I would normally discount a curves-only result, but the consistency with the static benchmarks (and the fact that the gap grows rather than starting high and converging) makes it credible. The scaling-law study (five Chinchilla-style MoE models, 8-of-64 experts, Muon) fits loss curves of $\text{MLA} = 2.3092 \cdot C^{-0.0536}$ versus $\text{Kimi Linear} = 2.2879 \cdot C^{-0.0527}$, yielding roughly **1.16x compute efficiency** for Kimi Linear at compute-optimal training. The authors note this was obtained while reusing MLA's tuned config without per-KDA tuning, so they expect it to improve.

### Efficiency: the reason to care

![Decode speedup over MLA widens as context grows](/imgs/blogs/kimi-linear-7.png)

The timeline figure traces the efficiency story as context grows, and the structure of it is what matters: the advantage *compounds* with sequence length because the KV-cache gap compounds. The decode speedup over MLA is 4.8x at 256K, 5.7x at 512K, and 6.3x at 1M. At 1M tokens, the time-per-output-token is 1.84 ms for Kimi Linear versus 11.48 ms for MLA. Prefill speedups are 2.3x at 512K and 2.9x at 1M. The KV-cache reduction is up to 75%, a direct consequence of only 1 in 4 layers carrying a growing cache. Critically, against the GDN-H baseline, Kimi Linear adds *negligible* latency overhead despite its finer-grained gating — the prefill and decode curves are nearly indistinguishable from GDN-H. That is the kernel work paying off: the channel-wise gate would be a performance regression if the constrained-DPLR kernel were not nearly as fast as the scalar-gate one.

### What is load-bearing and what might not transfer

The load-bearing facts are the controlled-comparison design (identical params/data/training) and the kernel efficiency (the channel-wise gate is free at inference relative to GDN-H). Those are the two things that, if true, make the architecture worth adopting, and the evidence for both is direct.

What might not transfer: the 1.16x scaling advantage was measured with MLA's tuned hyperparameters, so it is a *lower bound* under one specific config — it could be larger with KDA-specific tuning, or it could shrink if MLA's config happens to suit KDA unusually well at this scale. The long-context wins are at 128k; the model supports 1M, but the long-context *quality* table tops out at 128k, so I would not extrapolate the quality lead to 1M without seeing the numbers. And the RL result is curves-only, so the magnitude of the RL advantage is qualitative.

## Critique

**What is strong.** The experimental hygiene is excellent. Holding architecture, parameter count, and training recipe fixed across three models, then sweeping the hybrid ratio from 0:1 to 15:1, is exactly how you isolate the contribution of an architectural change, and most papers do not bother. The NoPE result is the most convincing single ablation I have seen for a positional-encoding claim: the RoPE variant underperforming even full attention on long context is a falsifiable prediction that came out the right way. The kernel work is real engineering — the $a=b=k$ binding is not a hand-wave, it is a specific reduction in matmul count with a measured 2x kernel speedup and an open implementation you can read.

**What is weak or unfalsifiable.** The "KDA is a learnable multiplicative positional encoding" framing is suggestive but the paper leans on it as intuition more than proof; the NoPE ablation supports the *consequence* (you can drop RoPE) without fully establishing the *mechanism* (that KDA's decay is what plays RoPE's role). The scaling-law claim of 1.16x rests on a five-point fit with the acknowledged caveat that the config was not tuned for KDA — a 1.16x efficiency edge from $C^{-0.0527}$ vs $C^{-0.0536}$ is a small difference in exponents that I would want to see replicated at more scales before betting on it. And the RL curves-only reporting means I cannot independently judge the RL claim's magnitude.

**The missing ablation.** The biggest gap is a head-to-head against modern *sparse* attention — NSA, MoBA, DSA. The paper discusses them but reports no benchmark comparison. Sparse attention is the other major path to cheap long-context decode, and without a head-to-head, I cannot tell whether the hybrid-linear approach dominates sparse attention or merely sits beside it on the Pareto frontier. The second missing piece is the 1M-token *quality* numbers — efficiency is reported at 1M but the quality table stops at 128k. I would also want an ablation isolating the channel-wise gate alone (KDA vs GDN at fixed everything else) on the *real* long-context benchmarks, not just the synthetic Palindrome/MQAR/Stack tasks, to confirm the gate (not the hybrid) is what drives the long-context win.

**What would change my mind.** If a sparse-attention baseline (a well-tuned MoBA or NSA hybrid at the same 48B-A3B scale and 5.7T tokens) matched or beat Kimi Linear on the long-context average *and* the 1M-decode TPOT, I would downgrade the architecture from "first to beat full attention on all three axes" to "one of several competitive options." Conversely, if the 1M-token quality numbers (when released) hold the 128k lead, I would upgrade my confidence substantially.

## What I'd build with this

1. **A retrieval-augmented agent on the 48B-A3B instruct checkpoint.** The 75% KV-cache reduction and 6.3x decode speedup at 1M context are exactly what a long-horizon coding or research agent needs — you can keep an entire repository or research corpus resident and still decode cheaply. The economics of agentic loops change when each decode step costs 1.84 ms instead of 11.48 ms at 1M context.
2. **An ablation harness that ports KDA into a smaller open model** to test the central claim independently: take a 1-3B dense or small-MoE model, swap a 3:1 KDA+NoPE-MLA stack in for full attention, hold tokens fixed, and re-run RULER and MRCR. The open KDA kernel in `flash-linear-attention` makes this tractable, and it would tell me whether the long-context win survives outside the Kimi training environment.
3. **A KDA-vs-sparse-attention bake-off** — the comparison the paper skips. Wire the open KDA kernel and a MoBA/NSA kernel into the same serving stack and measure long-context quality and 1M-decode TPOT head to head, which is the single most decision-relevant experiment for anyone choosing a long-context architecture today.
4. **A per-channel-gate analysis tool.** Because $\alpha_t$ is a 128-dim learned decay, you can inspect which channels the model keeps sticky versus volatile across a long document. That is an interpretability handle on what the linear memory is actually storing, and it might reveal whether specific channels specialize in identity/recall versus local syntax.
5. **A NoPE long-context extrapolation study.** Train two otherwise-identical Kimi Linear models, one NoPE and one RoPE, then push both well past their training context and measure degradation. If the "KDA as positional encoding" claim is right, the NoPE model should extrapolate more gracefully — a clean test of the paper's central mechanistic story.

## When to reach for Kimi Linear (and when not to)

Reach for a Kimi-Linear-style architecture when your dominant cost is *decode at long context*: agentic loops, multi-turn tool use over large contexts, RL with long rollouts, or anything doing test-time scaling where you generate far more tokens than you read. In that regime the 75% KV-cache cut and up-to-6.3x decode speedup at 1M tokens are not marginal — they change what fits on a given GPU budget and whether real-time interactivity is possible at all. The fact that you get those wins *without* sacrificing short-context or long-context quality versus a full-attention baseline is the whole reason this is interesting; historically you paid for linear-attention efficiency with quality, and here you do not.

Do not reach for it if your workloads are short-context and prefill-dominated — at 4k tokens the efficiency advantage is small and the architectural complexity (a custom KDA kernel, a hybrid layer schedule, NoPE infrastructure) is not worth it over a well-understood full-attention stack. Be cautious if your task profile looks like the benchmarks Kimi Linear *loses*: heavy on certain code-completion patterns (EvalPlus went to GDN-H and MLA in different settings) or the specific long-context tasks where MLA still leads (LongBench V2, Frames). And remember it is not a pure linear model — you still pay for 25% global MLA layers, so if you were hoping to eliminate the growing KV cache entirely, this is not that. It is a hybrid that moves the Pareto frontier, not a free lunch that abolishes the tradeoff. For the agentic, decoding-heavy future the paper is aimed at, moving that frontier is exactly the right bet.

## References

- **Paper (arXiv abstract):** [Kimi Linear: An Expressive, Efficient Attention Architecture — arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
- **Code & checkpoints:** [MoonshotAI/Kimi-Linear (GitHub)](https://github.com/MoonshotAI/Kimi-Linear) — base/instruct 48B-A3B checkpoints, KDA kernel in `flash-linear-attention`, vLLM integration.
- [Muon is Scalable for LLM Training: Inside Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) — the MoE backbone and Muon/MuonClip optimizer that Kimi Linear builds on.
- [MoBA: Mixture of Block Attention for Long-Context LLMs](/blog/paper-reading/large-language-model/moba) — the sparse-attention alternative this paper discusses but does not benchmark against.
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5) — the RL algorithm reused in Kimi Linear's RLVR stage.
- [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2) — the pretraining corpus and SFT/RL recipe Kimi Linear inherits.

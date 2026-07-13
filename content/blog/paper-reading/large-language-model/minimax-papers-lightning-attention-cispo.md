---
title: "Reading the MiniMax Papers: Lightning Attention, CISPO, and a Retreat to Full Attention"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - minimax
  - lightning-attention
  - linear-attention
  - mixture-of-experts
  - cispo
  - reinforcement-learning
  - long-context
  - test-time-compute
  - flow-matching
  - text-to-speech
  - full-attention
  - agentic-llm
description: "A walkthrough of every public MiniMax model report — 01, M1, M2, and Speech — pulling out the architecture, RL, infrastructure, and finetuning techniques worth stealing, including why they bet on linear attention and then walked it back."
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/minimax-papers-lightning-attention-cispo-1.png"
readTime: 32
---

Most labs publish a model and a leaderboard screenshot. MiniMax, across eighteen months, published something rarer: a paper trail you can read as an argument with itself. They made a large, expensive bet on linear attention in **MiniMax-01**, doubled down on it to make test-time compute cheap in **MiniMax-M1**, and then — in the report for the agentic **MiniMax-M2** — wrote a public post titled, almost sheepishly, *"Why did M2 end up as a full attention model?"* The answer was: because the bet did not survive contact with a production inference stack. That arc is the most honest piece of engineering writing any frontier lab shipped in 2025, and it is the reason this post reads the MiniMax corpus end to end rather than one paper at a time.

The goal here is not a book report. It is extraction. Each of these reports contains a handful of techniques that transfer well beyond MiniMax's own models — a reinforcement-learning objective that fixes a real failure mode in GRPO, a one-line precision change that unblocks RL training on a hybrid architecture, a sequence-parallelism trick that makes million-token training tractable, a speaker encoder that quietly reframes zero-shot TTS. We will read the four reports in order, then collect the reusable parts into a single toolbox at the end.

![The MiniMax model lineage from 01 through M2 and Speech, showing the attention bet flip](/imgs/blogs/minimax-papers-lightning-attention-cispo-1.png)

The diagram above is the mental model for the whole post: a single lineage in which the core architectural bet flips. MiniMax-01 (January 2025) and MiniMax-M1 (June 2025) are built on a 7:1 hybrid of *lightning* (linear) attention and softmax attention; MiniMax-Speech (May 2025) is a different modality that nonetheless shares the engineering temperament; and MiniMax-M2 (October 2025) throws the linear-attention hybrid out entirely and returns to full softmax attention on every layer, while shrinking activated parameters from 45.9B to roughly 10B for agentic serving. Read left to right, it is a story about a hypothesis being tested at increasing scale until it broke.

> [!tldr] TL;DR
> - **What the corpus claims:** Lightning (linear) attention in a 7:1 hybrid with MoE gives near-flat long-context quality to 1M tokens (MiniMax-01); the same backbone makes long chains-of-thought cheap enough to scale test-time compute (M1, ~25% of DeepSeek-R1's FLOPs at 100K generation); a new RL objective, **CISPO**, trains the reasoning model ~2× faster than DAPO; and a learnable, transcript-free speaker encoder plus a Flow-VAE latent tops the TTS arenas (MiniMax-Speech).
> - **Why it matters:** These are concrete, transferable techniques — CISPO's stop-gradient importance weighting, an FP32 LM head to fix train/inference logprob drift, LASP+ sequence parallelism, staged context-and-RoPE warmup, a flow on the VAE latent — not just benchmark wins.
> - **Most surprising finding:** MiniMax *publicly reversed* its signature linear-attention bet for M2, arguing that on a real inference stack (prefix caching, speculative decoding, low-precision state) full attention is simply less fragile.
> - **Where it's soft:** Several load-bearing infra numbers are undisclosed (training MFU, exact $\varepsilon_{\text{high}}$, the GPU SKU for 01); the M2 RL recipe is unpublished; and the headline "intelligence index" for M2 was later re-baselined to a lower number under a new methodology.

## Context: the long-context race and the quadratic wall

Every technique in this post is a reaction to one cost curve. Standard softmax attention computes an $n \times n$ score matrix for a sequence of length $n$, which is $O(n^2 \cdot d)$ in time and, more painfully at inference, grows a KV cache linearly in $n$ that you must stream from HBM on every decode step. At 4K tokens nobody cares. At 1M tokens — the context length MiniMax-01 set out to train — the quadratic term dominates everything, and the KV cache alone can exceed the weights.

The academic escape hatch is **linear attention**. If you drop the softmax and write attention as $\phi(Q)\,(\phi(K)^\top V)$ for some feature map $\phi$, the associativity of matrix multiplication lets you compute $\phi(K)^\top V$ first — a $d \times d$ matrix — and reuse it across all queries. That collapses the cost to $O(n \cdot d^2)$, linear in sequence length, and replaces the growing KV cache with a fixed-size $d \times d$ recurrent state. The lineage MiniMax builds on runs through TransNormer and **Lightning Attention-2** (arXiv [2401.04658](https://arxiv.org/abs/2401.04658)), which made the linear-attention kernel actually fast on a GPU rather than merely cheap on paper.

The catch — and it is the catch that drives the entire MiniMax story — is that pure linear attention is bad at *exact recall*. Compressing all history into a fixed $d \times d$ state is lossy, and tasks like needle-in-a-haystack retrieval, where the model must pull one specific fact out of a 500K-token document, expose that loss immediately. The gap this corpus tries to fill is the obvious one: get linear attention's cost curve without giving up softmax attention's recall. MiniMax-01's answer is a hybrid; M2's answer, eventually, is "never mind, full attention." The DeepSeek-MoE line is the constant backdrop — the contrast class MiniMax keeps measuring itself against on both architecture and reinforcement learning.

## MiniMax-01: the Lightning Attention hybrid foundation

MiniMax-01 (arXiv [2501.08313](https://arxiv.org/abs/2501.08313)) is the foundation everything else sits on. It ships two models — **MiniMax-Text-01** and the vision-language **MiniMax-VL-01** — and a single, aggressive thesis: you can train a 456B-parameter mixture-of-experts model (45.9B activated per token) with a 1-million-token training context, extrapolating to 4M tokens at inference, if seven of every eight attention layers are linear.

### Lightning Attention: the kernel trick, made concrete

The cleanest way to understand lightning attention is to watch where the $n \times n$ matrix *doesn't* get formed. Think of it as splitting a single causal-attention computation into two cooperating halves that each stay cheap.

![Lightning attention split into an intra-block left product and an inter-block right product over a d-by-d state](/imgs/blogs/minimax-papers-lightning-attention-cispo-2.png)

The figure traces it. After a SiLU feature map produces $Q, K, V$, the sequence is tiled into blocks. Within a block, the **left product** computes $(Q K^\top) \odot M$ and multiplies by $V$, where $M_{ij} = \lambda^{i-j}$ is an exponential *decay mask* that enforces causality and recency. This block is small, lives in SRAM, and costs $O(\text{block}^2 \cdot d)$ — the only place a score matrix appears at all, and it is tiny. Across blocks, the **right product** carries a $d \times d$ running state $KV$: each block updates $KV \mathrel{+}= K_{\text{block}}^\top V_{\text{block}}$ (with decay) and reads out $Q_{\text{block}} \cdot KV$. History is summarized into that $d \times d$ matrix, never an $n \times n$ one. The way this works is precisely the kernel trick of the introduction, but tiled so the GPU stays busy — vanilla linear attention needs a sequential per-token prefix sum that kills parallelism, and Lightning Attention-2's block scheme is what removes it.

A simplified but faithful PyTorch sketch makes the two products explicit:

```python
import torch
import torch.nn.functional as F

def lightning_attention(q, k, v, decay, block=256):
    # q, k, v: [B, H, N, D];  decay: scalar lambda in (0, 1)
    B, H, N, D = q.shape
    q, k, v = F.silu(q), F.silu(k), F.silu(v)
    out = torch.zeros_like(v)
    kv = torch.zeros(B, H, D, D, device=q.device)        # the d x d running state

    idx = torch.arange(block, device=q.device)
    mask = (decay ** (idx[:, None] - idx[None, :])).tril()  # M_ij = lambda^(i-j), causal
    pos  = decay ** (idx + 1)                               # per-position decay within a block

    for s in range(0, N, block):
        qb, kb, vb = q[..., s:s+block, :], k[..., s:s+block, :], v[..., s:s+block, :]
        intra = ((qb @ kb.transpose(-1, -2)) * mask) @ vb              # LEFT product, SRAM-local
        inter = (qb * pos[:, None]) @ kv                              # RIGHT product, reads d x d state
        out[..., s:s+block, :] = intra + inter
        kv = decay**block * kv + (kb * pos.flip(0)[:, None]).transpose(-1, -2) @ vb  # carry state
    return out                                                        # no n x n matrix ever formed
```

The difference from FlashAttention is worth stating plainly because the two are often confused: FlashAttention is still $O(n^2 \cdot d)$ softmax attention that has been made *I/O-aware* — it tiles to avoid materializing the score matrix in HBM, but it still computes it. Lightning attention changes the *algorithmic complexity* to linear, borrowing the same tiling and SRAM-residency ideas to do it efficiently. One is a memory optimization of the quadratic; the other replaces the quadratic.

### The 7:1 hybrid and the recall tax

Pure linear attention would have been cheaper still, and MiniMax did not use it. The architecture interleaves one full softmax-attention block after every seven lightning blocks — a **7:1 ratio**, with softmax sitting at layers 8, 16, 24, and so on through the 80-layer stack.

![A vertical stack of eight transformer layers, seven lightning and one softmax, repeating to 80 layers](/imgs/blogs/minimax-papers-lightning-attention-cispo-3.png)

That single softmax layer per block is the *recall tax*. The consistent rationale across MiniMax's writing and the community analyses is that pure linear attention's compressed state cannot do precise long-range lookup, and one full-attention layer per eight restores associative recall — needle-in-a-haystack, exact retrieval — while keeping seven of eight layers at linear cost. It is a deliberately minimal tax: not a 1:1 interleave, not 1:3, but 1:7, betting that recall is sparse enough in the layer stack that you only need to pay for it occasionally. Hold onto this number. The entire M2 reversal is, in part, the discovery that "occasionally" was optimistic at larger scale.

### MoE: few fat experts, on purpose

MiniMax-01's mixture-of-experts design is a near-perfect inversion of the DeepSeek-V3 recipe, and the contrast is the most instructive way to read it.

![Matrix comparing MiniMax-01 and DeepSeek-V3 across expert count, routing, sharing, capacity, and balancing](/imgs/blogs/minimax-papers-lightning-attention-cispo-4.png)

| MoE design axis | MiniMax-01 | DeepSeek-V3 |
| --- | --- | --- |
| Total / activated params | 456B / 45.9B | 671B / 37B |
| Routed experts | 32 (fat) | 256 (thin) |
| Top-k routing | top-2 | top-8 |
| Shared expert | none | 1 shared |
| Capacity handling | capacity limit + token drop | dropless |
| Load balancing | GShard-style auxiliary loss | aux-loss-free bias |

DeepSeek bets on *many thin experts* (256 routed plus one always-on shared expert, top-8), with dropless routing and the now-famous auxiliary-loss-free bias term for balancing. MiniMax-01 bets on the opposite: **32 fat experts** (expert FFN hidden dimension 9216), top-2 routing, *no* shared expert, a hard per-expert capacity with token-dropping when an expert overflows, and a classic GShard-style auxiliary loss to keep the router balanced. Neither is obviously right; they are different points on a granularity–overhead curve. MiniMax's choice trades the routing flexibility of many experts for lower all-to-all communication volume and simpler kernels, and accepts token-drop as a training-efficiency lever rather than fighting to be dropless. If you are designing an MoE, this table is the decision you actually have to make, and MiniMax is the existence proof for the "fewer, fatter, drop-tolerant" corner.

### Training infrastructure for a million tokens

A 1M-token training context is an infrastructure problem before it is a modeling one. Two techniques carry the weight.

![Before-and-after of serial send-recv KV passing versus LASP+ AllGather sequence parallelism](/imgs/blogs/minimax-papers-lightning-attention-cispo-5.png)

The first is **LASP+** (Linear Attention Sequence Parallelism Plus). Naive sequence parallelism for a recurrent linear-attention state is serial: rank 0 computes its KV block, sends it to rank 1, which waits, computes, and passes on — a dependency chain that leaves most GPUs idle. LASP+ replaces the send-recv ring with an **AllGather** of the KV state across ranks, so every rank can compute its block in parallel instead of waiting its turn. The second is **varlen ring attention**: rather than padding every sequence in a packed batch to the longest length (enormous waste at million-token scale), ring attention is applied directly to the concatenated, variable-length sequence, so the padding tax disappears. On the MoE side, MiniMax decouples expert parallelism into separate process groups for expert weight sharding and expert data parallelism, and overlaps the all-to-all of one expert group with the compute of another — a token-grouping overlap that hides communication behind useful work.

One number deserves a caution flag, because it is easy to misquote. MiniMax reports **>75% MFU** — but that figure is for end-to-end *inference* on H20 hardware, not training. A training-time MFU is not disclosed, and neither is the precise GPU SKU (community analyses disagree between H800 and H100 for a dynamically sized cluster of roughly 1,500–2,500 GPUs). When you cite MiniMax-01's efficiency, cite the inference MFU as an inference number.

### Data, context warmup, and staged alignment

The recipe for *reaching* a million tokens is a staged warmup, and it is the part most worth copying for anyone extending context.

![Timeline of context length and RoPE base stepping together through pretraining, then short-to-long SFT and RL](/imgs/blogs/minimax-papers-lightning-attention-cispo-6.png)

The reported schedule steps sequence length and RoPE base *together*: main pretraining at 8K context with RoPE base 10K, then a phase at 128K with RoPE base bumped to 5M, then 512K and 1M phases with RoPE base at 10M. RoPE is applied to only half of each head's dimensions, and the base-frequency bumps are what let the model extrapolate cleanly — context extension here is as much a RoPE-base *schedule* as an architectural property. (These exact phase percentages and base values come from the community deep-dive rather than the paper body, so treat them as the reported recipe, not gospel.) Pretraining runs on roughly 12 trillion tokens with a WSD-like learning-rate schedule that decays only to 10% of peak, a batch-size warmup from 16M to 128M tokens, and a quality-scoring pass that up-weights knowledge-rich data and 4×-repeats the highest-quality sources.

The alignment ordering is the second transferable idea: **short-context SFT → long-context SFT → short-context RL → long-context RL** (offline DPO, then online GRPO). Long context is not something you bolt on at the end; it is sequenced as its own alignment stage so the model learns to use the window it was trained to hold.

### VL-01: bolting vision onto the backbone

MiniMax-VL-01 reuses Text-01 as the language backbone and adds a 303M-parameter ViT (24 layers, patch size 14, hidden 1024) through a randomly initialized two-layer MLP projector. Images are handled at dynamic resolution — resized on a grid from 336×336 up to 2016×2016 with a 336×336 thumbnail always kept — and the model is trained on 512B vision-language tokens across a four-stage process (ViT and adapter first, then the full pipeline). It is a conventional adapter-on-a-frozen-ish-backbone recipe; the interesting part is simply that the hybrid-attention backbone takes the vision graft without incident.

### What MiniMax-01 actually scores

The headline is long context, and the headline number is real: on RULER, MiniMax-Text-01 holds accuracy nearly flat as context grows, where the competition falls off a cliff.

| Benchmark | MiniMax-Text-01 | GPT-4o | Claude-3.5-Sonnet | DeepSeek-V3 |
| --- | --- | --- | --- | --- |
| MMLU | 88.5 | 85.7 | 88.3 | 88.5 |
| MMLU-Pro | 75.7 | 74.4 | 78.0 | 75.9 |
| RULER @ 64K | 0.943 | 0.884 | 0.952 | — |
| RULER @ 1M | 0.910 | n/a | n/a | n/a (Gemini-1.5-Pro: 0.850) |
| LongBench-v2 (CoT) | 56.5 | 51.4 | 46.7 | — |

The core-knowledge scores are competitive-but-not-leading (MMLU 88.5, in the pack with DeepSeek-V3 and Claude). The long-context scores are the marquee result: ~0.91 RULER accuracy at 1M tokens where Gemini-1.5-Pro drops to 0.85 and GPT-4o/Claude cannot run at all, plus a reported 100% on the 4M-token vanilla needle-in-a-haystack retrieval. The load-bearing claim of the whole architecture — the hybrid keeps recall while paying linear cost — holds up on these retrieval-shaped benchmarks. Whether it holds on *multi-hop reasoning* at scale is the question M2 will answer, uncomfortably.

## MiniMax-M1: cheap test-time compute and CISPO

MiniMax-M1 (arXiv [2506.13585](https://arxiv.org/abs/2506.13585)) is what you build when you realize the hybrid backbone has a second superpower beyond long input: cheap long *output*. It is the same 456B / 45.9B hybrid as Text-01, post-trained into a reasoning model with a 1M-token input context and "thinking budgets" of 40K and 80K tokens, where the 40K model is an intermediate checkpoint of the 80K run.

### Why linear attention makes long thinking cheap

A reasoning model's cost is dominated by generation length — a chain of thought that runs to 64K or 100K tokens is the expensive part. With quadratic attention, each additional generated token attends to a growing KV cache, so the cost of long thinking compounds. With the lightning hybrid, seven of eight layers carry a fixed $d \times d$ state, so the per-token cost barely grows with output length. MiniMax quantifies the win directly against DeepSeek-R1: M1 uses approximately **25% of the FLOPs at a 100K-token generation length**, and less than **50% at 64K**. That is the entire economic argument for the architecture in the reasoning era — large test-time compute budgets become affordable precisely because long generations are cheap.

### CISPO: clip the weight, not the token

The centerpiece contribution of M1 is a reinforcement-learning objective called **CISPO** — Clipped IS-weight Policy Optimization. To see why it exists, you have to look at what PPO and GRPO do to a specific, important kind of token.

![Before-and-after contrasting PPO/GRPO ratio clipping with CISPO's stop-gradient importance weight](/imgs/blogs/minimax-papers-lightning-attention-cispo-7.png)

PPO and GRPO clip the token-level importance-sampling ratio $r_{i,t}(\theta) = \pi_\theta(o_{i,t}) / \pi_{\theta_{\text{old}}}(o_{i,t})$ inside a `min(r·Â, clip(r)·Â)` objective. When the clip binds, the gradient for that token is *zeroed*. The paper points at the casualty by name: reflective tokens like "However", "Wait", "Aha", "Recheck" are rare and low-probability under the old policy, so after the first on-policy update their ratio jumps, they get clipped out, and they "stop contributing to subsequent off-policy gradient updates." You are silently muting exactly the tokens that drive reasoning behavior.

CISPO's fix is to never mask a token. Instead it clips the importance weight, applies a **stop-gradient** to it so it becomes a pure scalar coefficient, and multiplies it against the log-probability of every token:

$$
J_{\text{CISPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{\sum_i |o_i|} \sum_i \sum_t \text{sg}\big(\hat{r}_{i,t}(\theta)\big)\, \hat{A}_{i,t}\, \log \pi_\theta(o_{i,t} \mid q, o_{i,\lt t})\right]
$$

where $\text{sg}(\cdot)$ is stop-gradient, $\hat{r}_{i,t} = \text{clip}(r_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})$ is the clipped IS weight, and $\hat{A}_{i,t}$ is the GRPO-style group-normalized advantage. Because the clipped weight is stop-gradiented, the gradient *always* flows through $\log \pi_\theta$ for every token — nothing is masked. In a clean PyTorch shape:

```python
import torch

def cispo_loss(logp_new, logp_old, advantages, eps_high=0.2, eps_low=1e9):
    # logp_new, logp_old: [B, T] log-probs of sampled tokens under pi_theta and pi_old
    # advantages: [B, T] GRPO group-normalized advantage, broadcast over the tokens
    ratio = torch.exp(logp_new - logp_old)                 # IS ratio r_{i,t}
    # clip the IS weight, then DETACH it -> it is only a scalar coefficient.
    # eps_low is set huge, so there is no effective lower bound (paper tunes only eps_high).
    is_weight = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high).detach()
    per_token = is_weight * advantages * logp_new          # every token keeps its gradient
    return -per_token.sum() / logp_new.numel()             # token-level norm; no KL term
```

Three details are easy to miss and all matter. First, there is **no KL penalty** at all. Second, the loss is normalized at the **token level** (the $1/\sum_i|o_i|$ factor), not per sequence, which avoids length-weighting bias on very long chains. Third, the clip is asymmetric: MiniMax sets $\varepsilon_{\text{low}}$ to a large value so there is effectively no lower bound and only tunes $\varepsilon_{\text{high}}$ (whose exact value the paper does not disclose — do not quote a number). In a controlled zero-RL study on Qwen2.5-32B-base evaluated on AIME 2024, **CISPO matches DAPO's performance in half the training steps — a 2× speedup** — and beats both GRPO and DAPO at equal steps. This is the single most portable idea in the whole corpus: if your RL run is quietly killing rare-but-important tokens, stop masking and start scaling.

### The RL engineering that made it converge

CISPO is the headline, but M1 only trained because of a precision fix that reads like a debugging war story.

![Graph of the train versus inference logprob mismatch at the LM head, fixed by FP32 output precision](/imgs/blogs/minimax-papers-lightning-attention-cispo-8.png)

Under the hood of any large RL setup are two different engines: an inference engine generates rollouts, and a training engine computes gradients. They must agree on the log-probability of each sampled token, or the importance ratios are garbage. MiniMax found a "significant discrepancy" between the two, traced it to **high-magnitude activations in the LM output head** losing precision in bf16, and fixed it by computing the **LM head in FP32**. The correlation between training- and inference-mode probabilities went from roughly **0.90 to 0.99**, and reward growth — which had been stalled — resumed. The figure shows the fork: the bf16 path is a dead end where reward never grows; the FP32 path is what made the run trainable. If you take one operational lesson from M1, it is to *measure the train/inference logprob correlation* before blaming your reward model.

The rollout machinery has two more guards worth copying. A **repetition halt** stops generation whenever 3,000 consecutive tokens each have probability above 0.99 — a cheap, effective loop-breaker. And the 80K model is trained with a **progressive output-length curriculum**, expanding the generation budget through 40K → 48K → 56K → 64K → 72K → 80K rather than training at maximum length from the start, with the cheaper 40K checkpoint reused to filter and downsample data for the 80K run so long-output training does not destabilize.

### Reward design and the RL curriculum

The reward stack mixes two regimes. For verifiable domains — math, code, logical reasoning, software engineering — MiniMax uses **rule-based rewards** (final-answer correctness plus format rewards). For open-ended general tasks with no ground truth, it uses a **generative reward model (GenRM)**. The GenRM introduced a classic failure: it "preferred longer outputs over potentially superior concise alternatives," a length bias the team treated as a control-systems problem — continuous online monitoring of length bias during training, triggering GenRM recalibration, plus reward shaping, value clipping, and normalization. The curriculum starts with reasoning-intensive rule-rewarded tasks and gradually mixes in general-domain tasks: roughly 50K math samples, 53K logical-reasoning samples (41 task types via SynLogic), 30K competitive-programming samples, several thousand SWE samples, and 25K general-domain samples, after an SFT cold-start that is roughly 60% math and code.

The full RL run is famously cheap to state precisely: **512 H800 GPUs, three weeks, a rental cost of $534,700.** That number is doing a lot of rhetorical work — it is the "frontier reasoning RL on a startup budget" headline — and it is one of the few infra figures in the corpus disclosed exactly.

### M1 benchmarks

| Benchmark | M1-80K | M1-40K | DeepSeek-R1 (orig.) | Qwen3-235B |
| --- | --- | --- | --- | --- |
| AIME 2024 | 86.0 | 83.3 | 79.8 | 85.7 |
| MATH-500 | 96.8 | 96.0 | 97.3 | 96.2 |
| LiveCodeBench | 65.0 | 62.3 | 55.9 | 65.9 |
| SWE-bench Verified | 56.0 | 55.6 | 49.2 | 34.4 |
| OpenAI-MRCR @ 128K | 73.4 | 76.1 | 35.8 | 27.7 |
| LongBench-v2 | 61.5 | 61.0 | 58.3 | 50.1 |
| TAU-bench (airline) | 62.0 | 60.0 | — | 34.7 |

M1 beats the *original* DeepSeek-R1 on AIME, LiveCodeBench, and SWE-bench, and dominates on long-context and agentic tool-use (OpenAI-MRCR at 128K: 73.4 vs R1's 35.8; TAU-bench airline 62.0 vs 34.7 for Qwen3). It trails the newer R1-0528, Gemini 2.5 Pro, and o3 on most math and knowledge tasks — MiniMax calibrates its own claim to "comparable or superior to the original DeepSeek-R1." One genuinely odd result worth flagging: **M1-40K beats M1-80K on several long-context and retail-agent rows** (MRCR 76.1 vs 73.4), so a bigger thinking budget is not monotonically better.

## MiniMax-M2: the pragmatic retreat to full attention

If MiniMax-01 and M1 are the thesis, MiniMax-M2 (released October 2025, [minimax.io/news/minimax-m2](https://www.minimax.io/news/minimax-m2)) is the rebuttal — written by the same authors. It is a 230B-total / ~10B-activated MoE (precisely ~229.9B / ~9.8B), built for coding and agentic workflows, and it abandons the linear-attention hybrid that defined the lineage.

### Why they walked back linear attention

MiniMax published a dedicated post, *"Why did M2 end up as a full attention model?"*, and it is the most useful document in the corpus because it is a negative result stated honestly. Conceptually, the argument is that linear attention wins on paper FLOPs but loses across the *production* inference stack.

![Matrix of why M2 chose full attention over linear attention across production concerns](/imgs/blogs/minimax-papers-lightning-attention-cispo-9.png)

| Production concern | Lightning / linear attention | Full attention (M2) |
| --- | --- | --- |
| Compute | $O(n \cdot d^2)$, but memory-bound | $O(n^2 \cdot d)$ |
| Efficiency crossover | wins only past a few thousand tokens | fine at agent context lengths |
| Low-precision state | state storage is precision-fragile | robust |
| Prefix caching | breaks | works |
| Speculative decoding | breaks | works |
| Multi-hop reasoning at scale | deficits emerge | reliable |

The points, in MiniMax's own words: linear-attention kernels are "memory-bound — even during training," so the theoretical FLOP advantage does not translate to wall-clock; the efficiency crossover with full attention happens "at a few thousand tokens — which isn't particularly long"; low-precision state storage is fragile; and crucially, linear attention *breaks the two optimizations that matter most for agentic serving* — prefix caching and speculative decoding. On quality, the hybrid "looked just as good as pure full attention" on MMLU, BBH, MATH, and LongBench, but "the price paid became obvious at a larger scale: the model had clear deficits in complex, multi-hop reasoning tasks." A sliding-window variant they tried "performed extremely poorly on agent tasks." The config file settles any ambiguity: `attn_type_list` is all 1s across all 62 layers — full softmax attention everywhere. This is the recall tax of MiniMax-01 coming due with interest: one softmax layer per eight was enough for needle retrieval, not enough for multi-hop reasoning at production scale.

### The M2 architecture

M2 is co-designed around the agentic inference pattern rather than around raw quality, and the config reflects it:

| Property | MiniMax-M2 |
| --- | --- |
| Total / activated params | ~230B / ~10B |
| Routed experts / top-k | 256 / top-8 |
| Layers / hidden size | 62 / 3072 |
| Attention | full GQA, 48 query heads, 8 KV heads, head dim 128 |
| Normalization | per-layer QK-Norm, RMSNorm |
| RoPE base | 5,000,000 (rotary dim 64) |
| Context | 196,608 trained (~192K), 128K eval |
| Quantization / decoding | FP8 (e4m3) weights, Multi-Token Prediction (3 modules) |

Notice the reversal of the *MoE* bet too: M2 swings all the way to **256 thin experts with top-8 routing** — the DeepSeek granularity, not the 32-fat-expert MiniMax-01 granularity. Combined with full attention (prefix-cache and speculative-decode friendly), FP8 weights, and three MTP modules for speculative decoding, the architecture is a coherent argument that ~10B activated parameters served fast beats 45.9B served cleverly when the workload is long agent loops.

### Interleaved thinking and the serving contract

M2's most operationally important property is not in the weights — it is in how you must call it. M2 is an **interleaved thinking** model: it wraps reasoning in `<think>...</think>` blocks and alternates reasoning with tool calls across a multi-turn loop, and it requires that prior-turn thinking be passed back into the conversation **verbatim**.

![Before-and-after showing agentic scores collapsing when think blocks are dropped between turns](/imgs/blogs/minimax-papers-lightning-attention-cispo-10.png)

This is a hard contract, not a suggestion. MiniMax's docs are explicit: "Do not remove the `<think>...</think>` part, otherwise the model's performance will be negatively affected." The ablation is brutal — dropping prior-turn thinking versus keeping it: τ²-Bench **64 → 87** (+35.9%), BrowseComp **31.4 → 44.0** (+40.1%), GAIA 67.9 → 75.7, SWE-bench Verified 67.2 → 69.4. And there is a real-world trap baked in: the OpenAI Chat Completions API does not support passing reasoning content back in subsequent requests, so a naive OpenAI-style harness *silently strips* the thinking and quietly degrades M2. The fix is to preserve the assistant turn intact:

```python
def record_assistant_turn(messages, reasoning, visible_text, tool_calls):
    # WRONG: strips the model's prior reasoning, the way the OpenAI Chat
    # Completions schema does. M2 degrades hard (tau2-bench 87 -> 64).
    messages.append({"role": "assistant", "content": visible_text})

    # RIGHT: pass the full assistant turn back verbatim, <think> blocks included.
    messages.append({
        "role": "assistant",
        "content": f"<think>{reasoning}</think>{visible_text}",
        "tool_calls": tool_calls,        # M2 uses an XML-style tool-call format
    })
    # Do not drop the <think> span between turns or planning/self-correction collapses.
```

If you integrate exactly one M2-specific thing, it is this: audit your agent framework for where it discards reasoning between turns.

### Positioning, efficiency, and pricing

The ~10B activated count is the whole pitch for agentic serving: high throughput, low latency, high concurrency, low cost. MiniMax claims around 100 tokens/sec output (Artificial Analysis independently measured 111 tok/s) and prices it at **$0.30 per 1M input / $1.20 per 1M output tokens** — roughly 8% of Claude Sonnet 4.5's price. One honest caveat from Artificial Analysis: M2 is *verbose*, spending ~120M tokens to complete their full evaluation suite (tied for the highest), so the effective cost per task is higher than the per-token price suggests. On benchmarks:

| Benchmark | MiniMax-M2 | Claude Sonnet 4.5 | GPT-5 (thinking) | DeepSeek-V3.2 |
| --- | --- | --- | --- | --- |
| SWE-bench Verified | 69.4 | 77.2 | 74.9 | 67.8 |
| Terminal-Bench | 46.3 | 50.0 | 43.8 | 37.7 |
| BrowseComp | 44.0 | 19.6 | 54.9 | 40.1 |
| τ²-Bench | 77.2 | — | — | — |
| LiveCodeBench | 83.0 | 71.0 | 85.0 | 79.0 |
| GPQA-Diamond | 78.0 | 83.0 | 85.0 | 80.0 |

At launch, Artificial Analysis ranked M2 as the **#1 open-weights model** with an Intelligence Index of 61 (top-5 globally), which is the framing MiniMax cites. One honest correction for the record: AA later moved to a harder v4.0 methodology under which M2 shows 36 (rank ~#33) — a different, non-comparable index, not a regression. As for the post-training recipe: MiniMax's launch materials do *not* publish M2's RL details. The family's later writing confirms it "largely continued to rely on CISPO," but the specific reward-design and data details (the "Forge" framework, 140K augmented tasks) belong to the December 2025 **M2.1** writeup and should not be attributed to M2 itself.

## MiniMax-Speech: the same philosophy in a different modality

MiniMax-Speech (arXiv [2505.07916](https://arxiv.org/abs/2505.07916), the model behind the Speech-02-HD product) looks unrelated to the LLM line until you notice it makes the same kind of move: take a component everyone borrows off the shelf, make it *learnable and task-specific*, and win.

### The architecture

The system has three parts — a tokenizer, an autoregressive Transformer, and a latent flow-matching model.

![Graph of the MiniMax-Speech pipeline with a learnable speaker encoder conditioning the AR model](/imgs/blogs/minimax-papers-lightning-attention-cispo-11.png)

Text is BPE-tokenized; audio is tokenized by an Encoder-VQ-Decoder operating on mel-spectrograms at 25 tokens/second with CTC supervision. The AR Transformer generates discrete audio tokens from text, conditioned on a speaker embedding. Then — and this is the part worth stealing — a flow-matching model decodes those tokens into a **continuous VAE latent**, not a mel-spectrogram, which a Flow-VAE decoder turns into waveform. The two inputs to the AR model (text tokens and the speaker timbre vector) are what make this a branch-and-merge, not a straight pipeline.

### The learnable speaker encoder

The "intrinsic zero-shot" claim in the title is a precise one. Prior systems described as zero-shot — VALL-E, CosyVoice 2, Seed-TTS — actually require a *paired text-audio prompt* to clone a voice, which MiniMax argues makes them one-shot by a stricter definition. MiniMax-Speech's speaker encoder extracts timbre from an **untranscribed reference audio clip** — no transcript needed — into a fixed-size conditioning vector. The deeper difference is that the encoder is **learnable and jointly trained** with the AR model, rather than a fixed encoder pre-trained on a speaker-verification task. The argument: an SV encoder is optimized for a discriminative objective on different data, so jointly training the encoder for the *generation* objective tailors it to TTS and lets it cover every language in the training set. The ablation backs it (Chinese subset, zero-shot WER): the learnable speaker encoder beats a fixed pretrained SV embedding on word error rate (1.25 vs 1.40) while beating a prompt-only baseline on similarity. One easy-to-miss data rule makes it work: the reference audio fed to the encoder during training must *differ* from the target audio, or semantic content leaks and quality degrades.

### Flow-VAE: a normalizing flow on the latent

The second stolen-worthy idea is Flow-VAE. A plain VAE assumes a standard-normal latent prior, which is an expressive straitjacket.

![Before-and-after contrasting a plain VAE's standard-normal latent with Flow-VAE's invertible flow](/imgs/blogs/minimax-papers-lightning-attention-cispo-12.png)

Flow-VAE inserts an invertible normalizing flow $f$ on the encoder's output, so the latent is constrained to be *normal but not standard-normal* — a richer learned posterior — with a change-of-variables term (the Jacobian log-determinant) in the KL. Modeling the continuous VAE latent rather than the mel-spectrogram "raises the ceiling" of what the generator can represent and avoids the mel bottleneck. A compact sketch:

```python
import torch
import torch.nn as nn

class FlowVAELatent(nn.Module):
    """Normalizing flow on the VAE latent: map the posterior through an invertible f
    so the modeled distribution is 'normal, not standard-normal'."""
    def __init__(self, flow):          # flow: invertible RealNVP/Glow-style block
        super().__init__()
        self.flow = flow

    def kl_loss(self, mu, logvar):
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # encoder posterior q(z|x)
        u, log_det_jac = self.flow(z)                             # push z through f -> u
        log_qz = (-0.5 * (z - mu).pow(2) / logvar.exp() - 0.5 * logvar).sum(-1)
        log_pu = (-0.5 * u.pow(2)).sum(-1)                        # standard-normal base
        return (log_qz - log_pu - log_det_jac).mean()            # change-of-variables KL
```

The resynthesis ablation shows Flow-VAE beating a plain VAE on essentially every metric — wideband PESQ 4.30 vs 4.20, lower WER inside the full TTS system — and, more importantly per the paper, better *stability*.

### Extensions: emotion, design, and professional cloning

Because the speaker vector is disentangled and text-free, three extensions bolt on **without modifying the base model**:

| Extension | Mechanism | Why it works |
| --- | --- | --- |
| Emotion control | one LoRA per emotion, loaded at inference | base model untouched; trained on `<ref, text, emotive audio>` triples |
| Text-to-Voice (T2V) | natural-language description → 128-d PCA timbre | timbre space compressed via PCA; maps text + attributes to it |
| Professional Voice Cloning (PVC) | the speaker embedding is the *only* trainable parameter | one vector per speaker scales to thousands of voices cheaply |

The pattern — a disentangled conditioning vector as a reusable interface — is the transferable lesson, and it generalizes to any conditional generative-audio system.

### Speech benchmarks

| System | Mode | test-zh WER ↓ | test-zh SIM ↑ | test-en WER ↓ |
| --- | --- | --- | --- | --- |
| Ground truth | — | 1.25 | 0.750 | 2.14 |
| Seed-TTS | one-shot | 1.12 | 0.796 | 2.25 |
| CosyVoice 2 | one-shot | 1.45 | 0.748 | 2.57 |
| MiniMax-Speech | zero-shot | **0.83** | 0.783 | **1.65** |
| MiniMax-Speech | one-shot | 0.99 | **0.799** | 1.90 |

MiniMax-Speech posts the lowest word error rate in every column on Seed-TTS-eval — its zero-shot WER even beats its own one-shot and beats ground truth — and one-shot similarity surpasses Seed-TTS. It secured the #1 position on the Artificial Analysis Speech Arena (human-preference ELO), ahead of ElevenLabs and OpenAI. Note there is no RLHF or preference tuning for the model here — this is supervised generative modeling done carefully, not an RL story. There is also a real "zero-shot vs one-shot" trade-off worth internalizing: dropping the text-audio prompt *lowers* WER (more decoding freedom) while one-shot *raises* similarity, so exposing both modes is a feature, not a hedge.

## The technique toolbox: what to steal

Reading four reports in a row, the reusable ideas separate cleanly into four families.

![Grid of the MiniMax technique toolbox across architecture, RL, infrastructure, and finetuning](/imgs/blogs/minimax-papers-lightning-attention-cispo-13.png)

**Architecture.** Lightning attention's $d \times d$-state kernel trick is the cheapest way to get linear-cost long context, and the 7:1 hybrid is a clean knob for trading recall against cost. But the load-bearing lesson is the *retreat*: a hybrid that matches full attention on standard benchmarks can still carry multi-hop-reasoning deficits that only surface at scale, and efficient attention's wins evaporate once prefix caching and speculative decoding enter the picture. Treat "efficient attention matches full attention" as a claim to be verified on the full inference stack, not on MMLU.

**RL and post-training.** CISPO — clip a stop-gradient importance weight rather than masking tokens — is directly portable to any GRPO/PPO setup that might be muting rare reasoning tokens. The FP32 LM-head fix is a one-line change that should be the *first* thing you check when RL reward stalls on a model with high-magnitude output activations. Dropping the KL term, normalizing the loss at the token level, monitoring GenRM length bias online, and the 3,000-token repetition halt are all cheap, transferable guards.

**Training infrastructure.** LASP+ (AllGather instead of a serial send-recv ring) and varlen ring attention (no padding on packed sequences) are the two enablers of million-token training; decoupled expert parallelism with token-grouping all-to-all overlap is how the MoE scales. M2 adds the serving-side counterpart: FP8 weights plus Multi-Token Prediction co-designed for speculative decoding.

**Finetuning and data.** Stage context and RoPE base together when you extend the window, and sequence alignment short-context-before-long rather than bolting long context on at the end. And from Speech: make your conditioning encoder *learnable and task-specific* instead of borrowing a frozen one from a discriminative task, and put a flow on your VAE latent to escape the standard-normal straitjacket.

## Critique: what's strong, what's soft, and what would change my mind

What is strong is the intellectual honesty. The M2 reversal post is a genuine negative result published by the team that made the original bet, and that is worth more than a dozen leaderboard wins. CISPO is a real, well-motivated contribution with a clean ablation against DAPO, and the FP32-head story is the kind of operational detail most labs would never disclose. The Speech work's reframing of "zero-shot" is a sharp, defensible distinction backed by ablations.

What is soft is the infrastructure transparency, unevenly. MiniMax-01's training MFU is never given (only an inference number that is easy to misquote), the GPU SKU is ambiguous, and the context-warmup schedule rests on a community deep-dive rather than the paper body. CISPO's $\varepsilon_{\text{high}}$ — the one hyperparameter that defines the objective's behavior — is undisclosed, which makes exact reproduction impossible. M2's post-training recipe is essentially a black box at launch, and the headline "intelligence index" was quietly re-baselined downward under a new methodology, which is the kind of number that ages badly. The benchmark tables are also self-reported, with MiniMax computing several baseline scores itself.

The missing ablation that would most sharpen the story is a *clean, scale-matched* comparison of the 7:1 hybrid against full attention on multi-hop reasoning — the exact deficit M2's post asserts. The reversal post describes it qualitatively ("clear deficits at larger scale") but does not publish the curve. **What would change my mind** about the "full attention won" conclusion: a published experiment showing a hybrid at, say, a 3:1 or 1:1 ratio (rather than 7:1) closing the multi-hop-reasoning gap while keeping a meaningful fraction of the long-generation FLOP savings — and surviving prefix caching and speculative decoding in a real serving stack. Absent that, M2's retreat reads less as "linear attention is wrong" and more as "7:1 was too aggressive and the serving ecosystem is not ready," which is a narrower and more interesting claim.

## What I'd build with this

A few concrete extensions I would actually try:

1. **Drop CISPO into an existing GRPO pipeline** and measure the gradient contribution of reflective tokens before and after. The hypothesis — that ratio clipping silently mutes "Wait/However/Recheck" — is directly testable by logging per-token gradient norms bucketed by token identity.

2. **Add the train/inference logprob-correlation metric to every RL dashboard.** It is a one-line diagnostic that would have caught M1's bug immediately, and it generalizes to any setup with separate rollout and training engines (including quantized or hybrid models where output-head precision bites).

3. **Build the interleaved-thinking serving contract into an agent gateway.** A middleware that preserves `<think>` blocks across turns — and refuses to silently strip them through an OpenAI-shaped API — would turn M2's 35–40% agentic degradation into a non-issue, and the same pattern protects any reasoning model with persistent thinking state.

4. **Port Flow-VAE's "flow on the latent" to a non-speech generator.** Any predict-the-latent-then-decode stack (image, music, video) inherits the standard-normal straitjacket; inserting an invertible flow on the VAE latent is a small, self-contained change with a measurable resynthesis-quality target.

5. **Run the missing hybrid-ratio ablation.** Train small models at 1:1, 3:1, and 7:1 lightning:softmax ratios and measure multi-hop reasoning, prefix-cache hit rate, and speculative-decode acceptance — the experiment MiniMax's reversal post gestures at but never published.

## References

- MiniMax-01: *Scaling Foundation Models with Lightning Attention* — arXiv [2501.08313](https://arxiv.org/abs/2501.08313) · [GitHub](https://github.com/MiniMax-AI/MiniMax-01) · [Text-01 model card](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)
- MiniMax-M1: *Scaling Test-Time Compute Efficiently with Lightning Attention* — arXiv [2506.13585](https://arxiv.org/abs/2506.13585) · [GitHub](https://github.com/MiniMax-AI/MiniMax-M1)
- MiniMax-M2 — [announcement](https://www.minimax.io/news/minimax-m2) · [why full attention](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model) · [why interleaved thinking](https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2) · [model card](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- MiniMax-Speech: *Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder* — arXiv [2505.07916](https://arxiv.org/abs/2505.07916)
- Lightning Attention-2 — arXiv [2401.04658](https://arxiv.org/abs/2401.04658)
- Related on this blog: [Beyond GRPO: DAPO, Dr. GRPO, GSPO](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) · [Optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) · [Modern LLM architectures](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) · [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management)

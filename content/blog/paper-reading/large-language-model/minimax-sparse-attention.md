---
title: "MiniMax Sparse Attention: A Tiny Learned Selector for Million-Token Attention"
publishDate: "2026-06-16"
date: "2026-06-16"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - minimax
  - sparse-attention
  - long-context
  - grouped-query-attention
  - mixture-of-experts
  - attention
  - gpu-kernels
  - flashattention
  - kv-cache
  - paper-reading
description: "A close read of MiniMax Sparse Attention (MSA): a GQA-native block-sparse attention whose lightweight Index Branch selects k key-value blocks per group, trained through a detached KL loss, co-designed with exp-free top-k and KV-outer kernels to turn a 28.4x FLOP cut at 1M tokens into 14.2x prefill and 7.6x decode speedups on a 109B MoE."
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/minimax-sparse-attention-1.png"
readTime: 35
---

The expensive part of a long-context language model is not the weights — it is the attention. A 100-billion-parameter Mixture-of-Experts model with six billion active parameters per token is, by 2026 standards, a perfectly ordinary thing to serve. Ask that same model to read a one-million-token codebase, hold a few hundred interleaved reasoning-and-action steps in an agent loop, or keep a persistent memory across a long session, and the softmax attention — every query attending to every key — grows as the square of the sequence length. Double the context and you quadruple the attention compute. At 128K tokens this is uncomfortable. At 1M tokens it is the dominant term in your prefill budget, and the latency it imposes is the thing standing between you and the agentic workloads everyone now wants to deploy.

MiniMax Sparse Attention (MSA), from the MiniMax team (arXiv 2606.13392, June 2026), is a deliberate, production-first attack on that wall. The thesis is unusually disciplined: after extensive ablation, keep only the components that are *essential* for stable sparse training, and design everything to reuse the software and hardware infrastructure you already have. There is no exotic recurrence, no new model class, no parameter-free hand-coded window. There is a standard Grouped-Query Attention layer with a tiny selector bolted on, a clean training trick to make that selector learn, and a pair of GPU kernels engineered so that the theoretical sparsity actually shows up on the wall clock. The headline: on a 109B-parameter MoE trained from scratch on 3 trillion tokens with native multimodal data, MSA matches full GQA on downstream benchmarks while reducing per-token attention compute by **28.4x at 1M context**, and — paired with its co-designed kernel — delivers **14.2x prefill and 7.6x decode wall-clock speedups on an H800**.

![Two-branch MSA: hidden states fan into a lightweight Index Branch that scores and selects k key-value blocks per GQA group, and a Main Branch that runs exact softmax attention over only the selected blocks before emitting the layer output](/imgs/blogs/minimax-sparse-attention-1.png)

The diagram above is the mental model, and almost everything in this article is an elaboration of it. A query token does not attend to the whole context. A lightweight *Index Branch* — two small projection matrices on top of standard GQA — scores the key-value context at block granularity and selects, for each GQA group, the top-$k$ blocks worth reading. The *Main Branch* then runs ordinary exact softmax attention, but restricted to the tokens inside those selected blocks. The selector is the only new machinery, it operates per GQA group, and it is trained not by the language-modeling loss directly but by a side objective that teaches it to mimic where full attention would have looked. The rest is kernels.

> [!tldr] TL;DR
> - **What it claims:** A GQA-native block-sparse attention. A two-matrix *Index Branch* selects the top-$k$ key-value blocks per GQA group; the *Main Branch* attends only to those blocks. At a fixed budget of $kB_k = 16 \times 128 = 2{,}048$ key tokens per query, a 109B MoE matches its full-attention twin across text, code, math, image, video, and long-context retrieval.
> - **Why it matters:** It is engineered to be *deployed*. It rides the GQA backbone shared by most open frontier models, supports both training-from-scratch and near-lossless conversion of a pretrained dense checkpoint, and is co-designed with GPU kernels so the FLOP savings become real latency: 14.2x prefill and 7.6x decode at 1M tokens.
> - **Most surprising finding:** The selector is trained by a **detached KL loss** against the Main Branch — and the gradient is stopped at the indexer's input. Without that stop-gradient the language-modeling loss *diverges*; with it, training is as stable as full attention. The model also grows its own attention sinks and locality without being told to.
> - **Where it fails:** A residual long-context retrieval gap. The from-scratch model (MSA-PT) is strong on long-context; the converted model (MSA-CPT) trails full attention on RULER/HELMET reranking subtasks by a couple of points, and the runtime speedup, while large, lags the theoretical FLOP reduction because sparse access is less regular than dense.

## Context: what came before

The community has spent years building escape routes from the quadratic wall, and they fall into two broad families. The first family *replaces* softmax attention with something cheaper and fundamentally different: linear attention and its kernels, state-space models such as Mamba, gated recurrences. These genuinely break the quadratic, but they change the model class, and their behavior on hard reasoning and retrieval is still being mapped. MiniMax knows this terrain intimately — their own [MiniMax-01 used lightning linear attention in a 7:1 hybrid](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe), interleaving cheap linear layers with a minority of full-softmax layers. Hybrid stacks of this kind reduce the number of quadratic layers without removing them.

The second family keeps softmax attention exactly as it is, and instead *sparsifies* which keys each query reads. This is where MSA lives, and it splits again by a single question: **is the sparsity pattern fixed by a human, or learned by the model, and if learned, is it trained as part of the model or bolted on at inference time?**

Fixed-pattern methods — Longformer's local windows and global tokens, BigBird's random+window+global mix, StreamingLLM's attention sinks plus a sliding window — impose a predefined support. They are content-agnostic: the pattern does not depend on the input, so a needle buried 600K tokens back is invisible unless it happens to land in the window. Inference-time adaptive methods do better by making the support depend on the input, but they operate on a model that was *trained dense*: H2O and SnapKV prune the KV cache during decoding using accumulated attention statistics; Quest does page-level importance estimation per query; MInference and FlexPrefill dispatch per-head sparse kernels at prefill. All of these inherit the full cost of training dense attention, and at least one inference phase stays near full-attention speed.

The closest neighbors are the *natively trainable* sparse-attention methods — those that train the selector during pretraining so the model and its sparsity grow up together. NSA (Native Sparse Attention) uses three parallel branches: a compressed coarse-global branch, a fine-grained selected branch, and a sliding window. [MoBA applies MoE-style top-k routing to attention](/blog/paper-reading/large-language-model/moba) with a parameter-free block gate. InfLLM-V2 unifies parameter-free block selection with a local window for short-to-long switching. And [DeepSeek's DSA, shipped in DeepSeek-V3.2](/blog/paper-reading/large-language-model/deepseek-v3-2), sits on top of Multi-head Latent Attention: a ReLU-based lightning indexer scores tokens individually, all query heads share a single top-$k$ index, and selection is token-level.

MSA stakes out its own position along two axes taken together: **per-GQA-group top-$k$ sharing** combined with **block-level selection**. Per-group sharing means each key-value head's group of query heads gets its own selection — different groups can attend to different long-range stripes — which is more expressive than DSA's single shared global index. Block-level selection (rather than DSA's token-level) keeps KV reads contiguous, which is exactly what a GPU wants. That combination is the paper's architectural bet, and the rest of the design follows from making it train stably and run fast. For the broader landscape of these techniques, our [survey of KV-cache management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) and the [efficient-attention survey](/blog/paper-reading/large-language-model/efficient-attention-mechanisms-for-large-language-models-a-survey) are good companions.

## Contributions

The authors stake three claims:

1. **A minimal, scalable, accelerated block-sparse attention.** MSA adds two projection matrices to a standard GQA layer and selects $k$ key-value blocks per query and GQA group via max-pooling, always keeping the local block. It supports both training from scratch and near-lossless conversion from a pretrained GQA checkpoint.
2. **Co-designed training and inference kernels** that turn theoretical compute savings into real wall-clock speedups: an exp-free top-$k$ kernel specialized for the small-$k$ regime, a KV-outer sparse-attention forward, and a fused sparse-KL backward.
3. **Extensive ablations up to a 109B MoE with native multimodal training**, dissecting MSA's behavior across scales and modalities, and isolating which design choices are load-bearing.

## A two-stage view of sparse attention

Before the architecture, it helps to write attention as two stages, because that factoring is the whole game. Standard causal softmax attention for query position $t$ and head $h$ is

$$\mathbf{o}_t^{(h)} = \sum_{i \le t} \alpha_{t,i}^{(h)} \mathbf{v}_i^{(h)}, \qquad \alpha_{t,i}^{(h)} = \frac{\exp(\langle \mathbf{q}_t^{(h)}, \mathbf{k}_i^{(h)}\rangle / \sqrt{d_h})}{\sum_{j \le t} \exp(\langle \mathbf{q}_t^{(h)}, \mathbf{k}_j^{(h)}\rangle / \sqrt{d_h})}.$$

This costs $\Theta(2 H_q N^2 d_h)$ FLOPs, quadratic in sequence length $N$, where $H_q$ is the number of query heads and $d_h$ the head dimension. Grouped-Query Attention (GQA) reduces the *memory* of the KV cache by using $H_{kv}$ key-value heads and tying $G = H_q / H_{kv}$ adjacent query heads to one shared KV head — each KV head defines a *GQA group* — but it does nothing about the quadratic attention compute itself.

A sparse attention layer factors that computation into an *indexer* and a sparse attention over the keys the indexer selects. For each query position $i$:

$$\mathcal{I}_i = \text{Index}_\phi(\mathbf{q}_i, \mathbf{K}_{\le i}), \qquad \mathbf{o}_i = \text{Attn}(\mathbf{q}_i, \mathbf{K}[\mathcal{I}_i], \mathbf{V}[\mathcal{I}_i]).$$

The first stage — the *Index Branch* — produces a selected index set $\mathcal{I}_i \subseteq \{1, \dots, i\}$. The second stage — the *Main Branch* — is ordinary softmax attention restricted to that set. Everything MSA does is a concrete, GPU-friendly instantiation of these two stages: the indexer is made cheap and block-level, and the attention is made to run fast over the blocks it picks.

To get block-level granularity, partition the $N$ tokens into blocks of size $B_k$: block $b$ holds tokens $\{(b-1)B_k + 1, \dots, \min(bB_k, N)\}$, giving $B = \lceil N / B_k \rceil$ blocks. The indexer then selects a set of *block* indices $\mathcal{I}_i^{(r)} \subseteq \{1, \dots, B\}$ for query $i$ and GQA group $r$, and the Main Branch attends over the causally visible tokens inside those blocks. Selecting blocks instead of individual tokens reduces routing overhead and — critically — keeps the keys and values you read contiguous in memory.

## The architecture: two branches

MSA instantiates the two-stage formulation at GQA-group and block granularity. Let $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{model}}}$ be the input hidden states. For each query token, the Index Branch selects $k$ key blocks of size $B_k$ per GQA group, and the Main Branch attends to the tokens in the selected blocks, whose budget is at most $kB_k$ tokens. In the 109B experiment, $B_k = 128$ and $k = 16$, so each query reads at most $16 \times 128 = 2{,}048$ key tokens regardless of how long the context grows.

### The Index Branch: a cheap learned selector

The Index Branch introduces exactly two new matrices: one index query head per GQA group, and a single index key head shared across groups,

$$\mathbf{Q}^{\text{idx}} = \mathbf{X}\mathbf{W}_q^{\text{idx}} \in \mathbb{R}^{N \times H_{kv} \times d_{\text{idx}}}, \qquad \mathbf{K}^{\text{idx}} = \mathbf{X}\mathbf{W}_k^{\text{idx}} \in \mathbb{R}^{N \times 1 \times d_{\text{idx}}},$$

where $d_{\text{idx}}$ is a small index dimension. For query token $i$ and group $r$, the branch first scores visible key tokens, then aggregates those token scores up to the block level by a max-pool. Using the block partition $\mathcal{B}_1, \dots, \mathcal{B}_B$,

$$S_{i,j}^{\text{idx},(r)} = \frac{(\mathbf{Q}^{\text{idx}})_i^{(r)} (\mathbf{K}^{\text{idx}})_j^\top}{\sqrt{d_{\text{idx}}}}, \qquad M_{i,b}^{\text{idx},(r)} = \max_{\substack{j \in \mathcal{B}_b \\ j \le i}} S_{i,j}^{\text{idx},(r)}.$$

Causality is enforced by the $j \le i$ constraint, and any block with no visible token is scored $-\infty$. The branch then takes the top-$k$ block indices under that per-block max score:

$$\mathcal{I}_i^{(r)} = \text{TopK}_{b \in \{1, \dots, B\}}\left(M_{i,\cdot}^{\text{idx},(r)},\; k\right).$$

Two details carry weight. First, the block score is a **max** over the tokens in the block, not a mean: a block earns selection if it contains even one strongly relevant token, which is the right semantics for retrieval (a needle does not need a high *average* — it needs one high spike). Second, the local block containing position $i$ is **always** included regardless of its score, and the selection $\mathcal{I}_i^{(r)}$ is shared by all $G$ query heads in group $r$ while each head keeps its own query projection.

### The Main Branch: exact attention on a budget

Given the block index set from the indexer, the Main Branch attends only to the causally visible tokens in those blocks. For any query head $h$ in group $r$, it applies standard scaled dot-product attention restricted to those tokens, using the key-value head of group $r$:

$$\mathbf{O}_i^{(h)} = \text{softmax}\left(\frac{\mathbf{Q}_i^{(h)} (\mathbf{K}^{(r)}[\mathcal{I}_i^{(r)}])^\top}{\sqrt{d_h}}\right)\mathbf{V}^{(r)}[\mathcal{I}_i^{(r)}].$$

Here $\mathbf{K}^{(r)}[\mathcal{I}_i^{(r)}]$ gathers the keys from the selected blocks. Because the selected blocks contain at most $kB_k$ causally visible tokens, the per-query attention cost is reduced from $O(N)$ to $O(kB_k)$ — a constant that is *fixed as the sequence length increases*. That single sentence is the entire efficiency story; the rest of the architecture exists to make it train and run.

### Why per-group, block-level selection

![A causal block grid for one query: blocks b1 to b8 down the rows, two GQA groups across the columns; both groups keep the sink block b1 and the local block b8, but Group 1 selects b3 and b5 while Group 2 selects b4 and b6](/imgs/blogs/minimax-sparse-attention-2.png)

The figure makes the per-group story concrete. Picture one query and the key blocks visible to it, b1 through b8, with b1 the oldest (the sink) and b8 the query's own local block. The two GQA groups share the structural picks — both keep the sink and the local block — but their learned long-range selections *diverge*: Group 1 routes to b3 and b5, Group 2 routes to b4 and b6. This is what "per-GQA-group top-$k$ sharing" buys you over a single global index shared by every head: different groups specialize to different long-range stripes, while the kernel still only has to deal with $H_{kv}$ distinct selections rather than $H_q$.

The block-level choice is the other half of the bet. Token-level selection (as in DSA) is finer, but the tokens a query picks are scattered, and gathering scattered keys is a memory-access pattern GPUs hate. Selecting whole blocks of 128 contiguous tokens means each read is a clean, coalesced stripe. The cost is granularity — you cannot keep token 31 of a block without keeping all 128 — but the paper's ablations show this costs almost nothing in quality (more on that later), and it is exactly what makes the kernel fast.

## The cost math: where the speedup comes from

Under the same $H_q$, $H_{kv}$, $d_h$, and sequence length $N$, the causal attention FLOPs of GQA and MSA are:

$$F_{\text{GQA}}(N) = 2 H_q d_h N^2, \qquad F_{\text{MSA}}(N) = \underbrace{H_{kv} d_{\text{idx}} N^2}_{\text{Index Branch}} + \underbrace{4 H_q d_h N k B_k}_{\text{Main Branch}}.$$

This is worth reading slowly. GQA's only term is quadratic in $N$. MSA has two terms: the Main Branch is **linear** in $N$ — it is $4 H_q d_h N \cdot kB_k$, and $kB_k$ is a fixed constant (2,048) — while the Index Branch is quadratic but carries a tiny coefficient, $H_{kv} d_{\text{idx}}$ instead of $H_q d_h$. With $H_q = 64$, $H_{kv} = 4$, $d_h = 128$, the GQA coefficient is $2 \cdot 64 \cdot 128 = 16{,}384$, while the index coefficient is $4 \cdot d_{\text{idx}}$ for a small $d_{\text{idx}}$. The gap therefore *grows with $N$* as long as $kB_k \ll N$ and $H_{kv} d_{\text{idx}} \ll H_q d_h$.

<figure class="blog-anim">
<svg viewBox="0 0 720 380" role="img" aria-label="As context length grows from 32K to 1M tokens, dense GQA attention cost rises quadratically while MSA stays nearly flat, ending about 28 times cheaper" style="width:100%;height:auto;max-width:820px">
<style>
.ms1-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.ms1-grid{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:3 5;opacity:.5}
.ms1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ms1-lblp{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ms1-gqa{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}
.ms1-msa{fill:none;stroke:var(--accent,#6366f1);stroke-width:3.5;stroke-linecap:round;stroke-linejoin:round}
.ms1-dot{r:7}
.ms1-dg{fill:var(--text-secondary,#6b7280);offset-path:path('M90,318 L240,300 L375,252 L510,176 L590,112 L660,70');animation:ms1-ride 8s ease-in-out infinite alternate}
.ms1-dm{fill:var(--accent,#6366f1);offset-path:path('M90,316 L240,313 L375,309 L510,304 L590,301 L660,298');animation:ms1-ride 8s ease-in-out infinite alternate}
.ms1-scan{stroke:var(--accent,#6366f1);stroke-width:1.5;stroke-dasharray:4 4;opacity:.55;animation:ms1-sweep 8s ease-in-out infinite alternate}
.ms1-gap{stroke:var(--text-primary,#1f2937);stroke-width:1.5}
.ms1-gaplbl{font:800 16px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
@keyframes ms1-ride{from{offset-distance:0%}to{offset-distance:100%}}
@keyframes ms1-sweep{from{transform:translateX(0)}to{transform:translateX(570px)}}
@media (prefers-reduced-motion:reduce){.ms1-dg,.ms1-dm{animation:none;offset-distance:100%}.ms1-scan{animation:none;transform:translateX(570px)}}
</style>
<text class="ms1-lblp" x="60" y="34">Per-token attention cost vs. context length</text>
<line class="ms1-axis" x1="90" y1="60" x2="90" y2="330"/>
<line class="ms1-axis" x1="90" y1="330" x2="680" y2="330"/>
<line class="ms1-grid" x1="375" y1="60" x2="375" y2="330"/>
<line class="ms1-grid" x1="660" y1="60" x2="660" y2="330"/>
<path class="ms1-gqa" d="M90,318 L240,300 L375,252 L510,176 L590,112 L660,70"/>
<path class="ms1-msa" d="M90,316 L240,313 L375,309 L510,304 L590,301 L660,298"/>
<line class="ms1-scan" x1="90" y1="60" x2="90" y2="330"/>
<circle class="ms1-dot ms1-dg" cx="0" cy="0"/>
<circle class="ms1-dot ms1-dm" cx="0" cy="0"/>
<text class="ms1-lbl" x="500" y="58">GQA (dense, grows as N squared)</text>
<text class="ms1-lblp" x="430" y="292" style="fill:var(--accent,#6366f1)">MSA (fixed k.B_k budget)</text>
<line class="ms1-gap" x1="668" y1="70" x2="668" y2="298"/>
<line class="ms1-gap" x1="664" y1="70" x2="672" y2="70"/>
<line class="ms1-gap" x1="664" y1="298" x2="672" y2="298"/>
<text class="ms1-gaplbl" x="612" y="190">28.4x</text>
<text class="ms1-lbl" x="74" y="350">32K</text>
<text class="ms1-lbl" x="356" y="350">512K</text>
<text class="ms1-lbl" x="646" y="350">1M</text>
</svg>
<figcaption>As context grows to 1M tokens, dense GQA cost climbs quadratically while MSA's fixed selection budget stays almost flat — a 28.4x per-token FLOP reduction. The dots ride each curve as the context-length cursor sweeps right.</figcaption>
</figure>

A back-of-the-envelope at 1M tokens makes the 28.4x concrete. Per-token, GQA pays $2 H_q d_h N = 16{,}384 \times 10^6 \approx 1.6 \times 10^{10}$ FLOPs. MSA's Main Branch pays a fixed $4 H_q d_h k B_k = 4 \cdot 64 \cdot 128 \cdot 16 \cdot 128 \approx 6.7 \times 10^7$ FLOPs, and its Index Branch pays $H_{kv} d_{\text{idx}} N$ per token — small but linear in $N$, so at 1M tokens it becomes the larger of MSA's two terms. The two together come out to roughly a 28x reduction, matching the paper's reported 28.4x. The interesting consequence: at extreme context, MSA's cost is eventually dominated by its *index* quadratic term, not its main attention. That is the term to attack next if you want to push past 28x — and the paper says exactly that in its outlook.

## Training a non-differentiable selector

Here is the problem that sinks most learned-sparsity designs. The top-$k$ in $\mathcal{I}_i^{(r)} = \text{TopK}(M, k)$ is not differentiable. The selected block indices are used only as a discrete routing decision, so under a plain sparse forward pass the index projections $\mathbf{W}_q^{\text{idx}}, \mathbf{W}_k^{\text{idx}}$ receive *no useful gradient* from the language-modeling loss — the indexer never learns which blocks to select. MSA solves this with three mechanisms working together: a **KL alignment loss**, a **gradient detach**, and an **indexer warmup**, plus a forced local block. Each one earns its place; the ablations later show what breaks without them.

### KL alignment: teach the indexer where attention looks

![Two parallel branches from the hidden state: a detached student path (stop-gradient then index distribution P_idx) and a teacher path (main scores then a group-averaged detached teacher P), both feeding a KL loss that updates only the two index projection matrices](/imgs/blogs/minimax-sparse-attention-3.png)

The KL loss gives the Index Branch a direct learning signal by matching its scores to the Main Branch on the selected tokens. Writing $\mathcal{I}_{i,\text{tok}}^{(r)}$ for the causally visible tokens induced by the selected block indices, define the Index Branch distribution $P^{\text{idx}}$ and a Main Branch teacher $P$ over that token set:

$$P_{i,j}^{\text{idx},(r)} = \frac{\exp(S_{i,j}^{\text{idx},(r)})}{\sum_{u} \exp(S_{i,u}^{\text{idx},(r)})}, \qquad P_{i,j}^{(r)} = \frac{1}{G}\sum_{\ell \in \mathcal{H}_r} \frac{\exp(S_{i,j}^{(\ell)})}{\sum_{u}\exp(S_{i,u}^{(\ell)})},$$

where $S^{\text{idx}}$ is the token-level index score and $S^{(\ell)}$ is the Main Branch score for query head $\ell$ in group $r$. The teacher $P$ averages the per-head Main Branch attention distributions at the probability level. The indexer is then trained to match it:

$$\mathcal{L}_{\text{KL}} = \frac{1}{N H_{kv}}\sum_{i=1}^{N}\sum_{r=1}^{H_{kv}} D_{\text{KL}}\left(\text{stopgrad}(P_{i,\cdot}^{(r)}) \;\|\; P_{i,\cdot}^{\text{idx},(r)}\right).$$

The teacher is detached — the KL leaves the Main Branch projections untouched — so the indexer learns to put its probability mass where full attention actually concentrates. The selection then becomes semantically meaningful rather than random.

### Gradient detach: confine the signal to the indexer

This is the single most important training detail in the paper, and it is easy to get wrong. The natural way to wire the indexer is to compute $\mathbf{Q}^{\text{idx}}, \mathbf{K}^{\text{idx}}$ from the hidden state $\mathbf{X}$ and let autograd do its thing. But then the KL gradient flows from the index projections *back into the hidden state*, and from there into the backbone through the residual stream. The auxiliary KL becomes an extra objective on the whole model rather than a local signal for the selector. MSA stops the gradient at the indexer's input:

$$\mathbf{Q}^{\text{idx}} = \text{stopgrad}(\mathbf{X})\mathbf{W}_q^{\text{idx}}, \qquad \mathbf{K}^{\text{idx}} = \text{stopgrad}(\mathbf{X})\mathbf{W}_k^{\text{idx}}.$$

With the teacher also detached, $\mathcal{L}_{\text{KL}}$ now updates *only* $\mathbf{W}_q^{\text{idx}}$ and $\mathbf{W}_k^{\text{idx}}$. Why this matters is dramatic and measured: in the ablation, without the detach, occasional KL-gradient spikes propagate into the backbone, causing gradient-norm spikes and LM-loss divergence within a few hundred steps. Even at stable coefficients, standard short-context benchmarks slowly regress during training — the authors attribute this to a *self-distillation* effect, where the backbone learns to lower the KL loss by simplifying its own Main Branch attention distribution rather than by improving the indexer. The fix is the stop-gradient, and they use it everywhere.

### Indexer warmup: do not select while the target is moving

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="Training timeline: during warmup the model runs full attention while the indexer trains by KL loss; after T-warm it switches to sparse top-k selection, and the attended key blocks drop from all blocks to only a few" style="width:100%;height:auto;max-width:820px">
<style>
.ms2-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.ms2-fill{fill:var(--accent,#6366f1);opacity:.85;transform-box:fill-box;transform-origin:left center;animation:ms2-grow 9s ease-in-out infinite}
.ms2-warm{stroke:var(--text-primary,#1f2937);stroke-width:1.5;stroke-dasharray:5 5}
.ms2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.ms2-phase{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ms2-blk{stroke:var(--border,#d1d5db);stroke-width:1.5}
.ms2-on{fill:var(--accent,#6366f1)}
.ms2-off{fill:var(--surface,#f3f4f6)}
.ms2-dense{animation:ms2-fadeA 9s ease-in-out infinite}
.ms2-sparse{animation:ms2-fadeB 9s ease-in-out infinite}
@keyframes ms2-grow{0%{transform:scaleX(0)}85%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes ms2-fadeA{0%,28%{opacity:1}40%,100%{opacity:0}}
@keyframes ms2-fadeB{0%,28%{opacity:0}40%,95%{opacity:1}100%{opacity:1}}
@media (prefers-reduced-motion:reduce){.ms2-fill{animation:none;transform:scaleX(1)}.ms2-dense{animation:none;opacity:0}.ms2-sparse{animation:none;opacity:1}}
</style>
<text class="ms2-phase" x="150" y="44">Phase 1 — full attention + KL warmup</text>
<text class="ms2-phase" x="470" y="44">Phase 2 — sparse top-k selection</text>
<rect class="ms2-track" x="60" y="92" width="600" height="26" rx="13"/>
<rect class="ms2-fill" x="60" y="92" width="600" height="26" rx="13"/>
<line class="ms2-warm" x1="240" y1="70" x2="240" y2="250"/>
<text class="ms2-lbl" x="240" y="150">T_warm (~40B tokens)</text>
<text class="ms2-lbl" x="120" y="290">indexer entropy still settling</text>
<text class="ms2-lbl" x="470" y="290">selections now reliable</text>
<g class="ms2-dense">
<rect class="ms2-blk ms2-on" x="290" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="330" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="370" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="410" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="450" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="490" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="530" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="570" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="610" y="180" width="34" height="40" rx="5"/>
<text class="ms2-lbl" x="470" y="240">attend to every block</text>
</g>
<g class="ms2-sparse">
<rect class="ms2-blk ms2-on" x="290" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-off" x="330" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-off" x="370" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="410" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-off" x="450" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-off" x="490" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-off" x="530" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-off" x="570" y="180" width="34" height="40" rx="5"/>
<rect class="ms2-blk ms2-on" x="610" y="180" width="34" height="40" rx="5"/>
<text class="ms2-lbl" x="470" y="240">attend to only k blocks</text>
</g>
</svg>
<figcaption>The warmup runs full attention while the KL loss teaches the indexer; at T_warm the model switches to top-k sparse selection, collapsing the attended key blocks from "all" to "only k" — without the fragile early picks that destabilize sparse-from-scratch training.</figcaption>
</figure>

The Main Branch attention distribution changes fastest at the very start of training: attention entropy drops sharply from a smooth distribution to a much sharper one before settling into slower representation learning. If top-$k$ selection is on from step zero, the indexer is asked to track a rapidly moving target while its own selections are still nearly random — poor early picks route the Main Branch to uninformative tokens, which weakens both backbone learning and the KL supervision the indexer receives. It is a chicken-and-egg failure.

The warmup breaks it. For the first $T_{\text{warm}}$ steps (40B tokens in the full run), the Main Branch runs **full attention** while the Index Branch is trained by the KL loss against the full-sequence Main Branch distribution. The backbone passes through the early sharpening phase without sparse routing errors, and the indexer gets a meaningful initialization before it controls token selection. After warmup, the model switches to top-$k$ sparse selection and the KL loss is restricted to the selected support. The same schedule is used when converting a pretrained full-attention checkpoint — the warmup aligns the freshly added index projections before they control routing.

### The forced local block

The last mechanism is simplest: for each query position $i$ and group $r$, the local block containing $i$ is *always* part of $\mathcal{I}_i^{(r)}$, during both training and inference. This reserves one of the $k$ slots for the query's immediate neighborhood and leaves the other $k-1$ to the learned selector. It prevents degenerate selections that omit the local context — the one place attention almost always needs to look — and provides dense supervision near the diagonal early in training.

### The layer in one piece

Putting it together, one MSA layer's training forward and auxiliary loss read like this. The model loss is $\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda \sum_{\text{layers}} \mathcal{L}_{\text{KL}}$, assembled by the training loop.

```python
def msa_layer(X, Wq, Wk, Wv, Wo, Wq_idx, Wk_idx, B_k, k):
    # Main Branch projections: (N, Hq, dh), (N, Hkv, dh), (N, Hkv, dh)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv

    # Index Branch projections, detached from the backbone:
    #   Qi -> (N, Hkv, d_idx),  Ki -> (N, 1, d_idx)
    Qi = stopgrad(X) @ Wq_idx
    Ki = stopgrad(X) @ Wk_idx

    # Block-level causal scores via max-pool over B_k tokens -> (N, Hkv, B)
    M = block_max_pool(Qi, Ki, B_k)

    # Top-k block indices per query and GQA group; the local block is forced in
    I = topk(M, k)

    # Exact softmax attention restricted to the selected blocks -> (N, Hq, dh)
    O = topk_attention(Q, K, V, I)
    out = O @ Wo

    # Auxiliary KL over the tokens induced by I; teacher (Q, K) is detached
    L_kl = kl_div(Qi, Ki, stopgrad(Q), stopgrad(K), I)
    return out, L_kl
```

Note what is *not* here: no value projection on the Index Branch, no forced first-block sink, no large local window, no learnable sink logit. Those were all tried and dropped — the ablation section is where they go to die.

## Making sparsity actually fast: the kernel co-design

A theoretical FLOP reduction is a promise, not a speedup. Translating MSA's sparsity into end-to-end wall-clock gains required co-designing the GPU execution path, and this is where the paper earns its production credentials. Three pieces matter: an exp-free top-$k$, a KV-outer attention forward with query gather, and a scheduling trick for skewed block popularity.

### Exp-free top-k

The first realization is that you do not need softmax to pick the top-$k$ blocks. Softmax is monotonic — order-preserving — so $s_i \le s_j \iff \text{softmax}(s_i) \le \text{softmax}(s_j)$, and the top-$k$ indices of the raw scores are identical to the top-$k$ of the softmax. The forward pass therefore bypasses the max/exp/sum steps of softmax entirely and ranks the raw block scores directly.

```python
# softmax is order-preserving, so the top-k of softmax(s) equals the top-k of s.
# skip max/exp/sum and rank the raw block scores directly.
top_blocks = raw_block_scores.topk(k, dim=-1).indices   # no exp, no normalization
```

That saves real work, but the bigger win is a custom kernel for the *small-$k$* regime. The block size $B_k = 128$ and selection size $k = 16$ are co-designed with a per-thread register top-$k$ kernel: a large $B_k$ raises attention arithmetic intensity, and a small $k$ at that $B_k$ keeps both the per-row candidate block count $B$ and $k$ below the sweet spot of general-purpose top-$k$ kernels (which amortize multi-pass bucketing over large $B$, or scale as $O(B\log^2 B)$ for bitonic sort). Each of a warp's 32 lanes streams a $1/32$ stride of the input row and maintains a $k$-element min-heap in shared memory, with the heap root cached in a register and inserts done with deferred writes; a $k$-round shuffle merge combines the 32 local results. Against `torch.topk` and TileLang's radix-select, the specialized kernel wins in every tested setting:

| Seq len $N$ | Blocks $B$ | $k$ | `torch.topk` | TileLang | MSA | vs torch | vs TileLang |
|---|---|---|---|---|---|---|---|
| 128K | 1,024 | 16 | 3,970 µs | 2,864 | **779** | 5.1x | 3.7x |
| 128K | 2,048 | 32 | 5,378 | 1,991 | **1,991*** | 2.7x | 1.8x |
| 512K | 4,096 | 16 | 33,810 | 17,779 | **7,880** | 4.3x | 2.3x |
| 512K | 8,192 | 32 | 57,659 | 26,100 | **21,326** | 2.7x | 1.2x |

The largest gains land exactly at the deployed setting, $B_k = 128$, $k = 16$. All implementations produce identical index sets — the speedup is free.

### KV-outer iteration: chase arithmetic intensity

![Before and after of sparse-attention iteration order: Q-outer reloads K and V per query and yields FLOPs-over-IO near G, while KV-outer loops over KV blocks, gathers the queries that selected each shared tile, and yields FLOPs-over-IO near two-thirds of B_k](/imgs/blogs/minimax-sparse-attention-4.png)

The second piece is the loop order of the sparse-attention forward. With equal query and key lengths in sparse prefill, you can iterate either over queries or over key-value blocks, and the choice determines how busy the tensor cores are. Iterating *queries* on the outer loop — process one query row, read its selected K/V blocks — gives a FLOPs-to-IO ratio of roughly $G$ (the GQA group size, 16 in this model). That is a low arithmetic intensity: you reload K and V for every query, and the kernel spends its time waiting on memory.

Iterating *KV blocks* on the outer loop — for each block, gather the queries that selected it and run attention against the shared K/V — flips the math. The FLOPs-to-IO ratio becomes roughly $\tfrac{2}{3}B_k$. Since $\tfrac{2}{3}B_k \gg G$ in practice ($\tfrac{2}{3}\cdot 128 \approx 85$ versus 16), MSA chooses KV-outer iteration with query gather. The kernel runs as a persistent grid over (kv_block, kv_head) tiles; for each tile, a reverse sparse index from the top-$k$ selection identifies the relevant query positions, which are loaded into shared memory via TMA copies. A complementary trick — query concatenation — packs $\lceil 128/G \rceil$ query positions together with their $G$ associated query heads, all under the same KV head, into a $128\times128$ score MMA, so a single query position's mere $G$ query heads do not under-fill the matrix unit.

### Pre-scheduled tile chunking for skewed popularity

![A five-stage pipeline for a hot KV tile: a popular block selected by many queries is split by the scheduler along its query dimension into chunks, fanned out across many CTAs that compute partials in parallel, each writing to a preassigned slot without atomics, then merged by an LSE split-K combine into the final output](/imgs/blogs/minimax-sparse-attention-5.png)

The KV-outer order has a failure mode: block popularity is wildly skewed. A single early KV block — the attention sink — is selected by nearly *every* query, and the same hotspot pattern arises on any popular block. A naive one-CTA-per-tile mapping makes that one CTA do enormous work while others idle.

The figure walks the fix. A GPU scheduler kernel splits each KV tile along its query dimension into chunks of at most $\sim 2kB_k$ queries each, fanning the hot tiles across many CTAs that share the same K/V load. Because each query's $k$ partials are now produced by $k$ different CTAs, the scheduler preassigns each (query, chunk) pair a slot $s \in [0, k)$ in an output buffer $\mathbf{O}_{\text{buf}}$, packed with the query index — so the attention kernel writes its partial to the preassigned offset *without atomics*. A second combine kernel then reads each query's valid slots, computes the running max and log-sum-exp, forms the split-K softmax weights, and merges the partials into the final normalized output. The two kernels use Programmatic Dependent Launch to hide inter-kernel launch latency, and a persistent grid with a global atomic work-counter handles the order-of-magnitude variation in per-tile work under variable-length, data-dependent sparsity.

### Fusing the sparse-KL backward

One more kernel-level economy. The KL loss only affects the *backward* gradient, so MSA skips its forward pass entirely: during the main forward, the per-block log-sum-exp values needed for the KL are emitted directly to global memory, and during index-branch computation the per-block LSEs are saved and reduced over the top-$k$ blocks. The backward kernel loads these scalars directly into the softmax, eliminating the redundant forward KL computation. It is a small thing, but it is the kind of small thing that separates a paper kernel from a deployable one.

## Does it work? Experiments

The validation is two 109B-scale experiments on a native multimodal model trained on text and image/video data. **MSA-PT** trains MSA from scratch (3T tokens; 40B-token indexer warmup, then sparse for the rest). **MSA-CPT** starts from a full-attention GQA checkpoint trained on 2.6T tokens, replaces dense attention with MSA, and continues for 400B tokens (40B warmup, then sparse). Both share the same 41-layer MoE backbone: ~109B total / ~6B active parameters, the first three layers dense and the remaining 38 MoE (128 routed + 1 shared expert, top-4), 64 query heads, 4 KV heads ($G = 16$), head dimension 128, RoPE dimension 64, $d_{\text{model}} = 3072$, 200K vocabulary. Both MSA models use $B_k = 128$, $k = 16$.

The first thing to check is training stability. Over the full 3T-token run, the MSA-PT and full-attention LM-loss curves are nearly indistinguishable — the inset on the final 50B-token window shows them overlapping — and the gradient-norm curves stay in the same range. Native sparse pretraining is as stable as dense at this scale. For the conversion path, the indexer-warmup stage rapidly drops the KL loss before sparse attention is enabled; once sparse, block recall (the fraction of the Main Branch's true top-$k$ blocks that the indexer recovers) stays favorable and *score* recall — the share of Main Branch attention mass the retrieved blocks carry — is higher still, confirming the indexer recovers most of the attention mass.

The downstream comparison is the headline:

| Group | Benchmark | Full (GQA) | MSA-PT | MSA-CPT |
|---|---|---|---|---|
| General | MMLU | 67.0 | **67.2** | 66.8 |
| General | BBH | **67.7** | 66.6 | 66.1 |
| General | WinoGrande | 58.3 | 60.9 | **62.0** |
| Math | GSM8K | 76.2 | **77.7** | 73.7 |
| Math | MathVista | 43.8 | **46.8** | 44.5 |
| Code | HumanEval | 61.0 | **64.0** | 57.9 |
| Code | BigCodeBench | 44.8 | 44.0 | **45.7** |
| Retrieval | RULER-8K | 79.8 | **84.2** | 77.2 |
| Image | MMMU | **46.8** | 45.9 | 44.5 |
| Image | VisualWebBench | 55.6 | **68.4** | 59.4 |
| Video | VideoMME | 41.11 | **45.48** | 39.65 |
| Video | TemporalBench | 49.4 | **53.4** | 50.6 |

Two stories. **MSA-PT**, which learns the sparse pattern throughout pretraining, is broadly *competitive-to-ahead* — it posts the best result on many math, image, video, and long-context retrieval benchmarks, suggesting native sparse pretraining adapts the model representations to the sparse pattern rather than fighting it. **MSA-CPT** is more conservative: it preserves most of the full-attention checkpoint's behavior and stays close on text, code, and perplexity, which is the point — it is the practical conversion route when you already have a trained dense checkpoint and do not want to retrain from scratch. The remaining gaps are benchmark-dependent, not concentrated in one capability.

Long-context, scaled further, is where the honest caveat lives. After ~140B tokens of additional long-context training, MSA-CPT on HELMET-128K and RULER-128K stays close to full attention overall, but the deltas are revealing:

| Benchmark | Subset | Full | MSA-CPT | Δ |
|---|---|---|---|---|
| HELMET-128K | Overall | **46.53** | 45.93 | −0.60 |
| HELMET-128K | ICL | 70.40 | **72.80** | +2.40 |
| HELMET-128K | Rerank/RAG | **34.60** | 32.50 | −2.10 |
| RULER-128K | Overall | 72.00 | **72.12** | +0.12 |
| RULER-128K | MK/MQ/MV | 96.63 | **98.87** | +2.24 |
| RULER-128K | QA1/QA2 | **47.80** | 46.80 | −1.00 |

Each query and GQA group still attends to only $kB_k = 2{,}048$ tokens, so the fact that MSA-CPT tracks full attention at 128K at all is the strong result. But reranking/RAG and multi-hop QA — tasks that want to integrate evidence from *many* scattered positions — are exactly where a 2,048-token budget bites. That gap is the price of the budget, and it is the most useful number in the paper for deciding whether MSA fits your workload.

### Efficiency: the FLOP cut versus the wall clock

On the efficiency side, the 28.4x per-token FLOP reduction at 1M tokens (under the 64/4/128 head config, $B_k = 128$, $k = 16$) becomes 14.2x prefill and 7.6x decode speedups on an H800. The measured speedup lags the FLOP reduction — and the paper is candid about why. Sparse attention introduces index construction, top-$k$ selection, reverse-index materialization, query gathering, and load-balancing overheads, and its memory access is less regular than dense. The runtime gain still *grows* with context length, because the dense baseline keeps scaling with the full sequence while MSA holds its main-attention budget fixed. The decode speedup (7.6x) being smaller than prefill (14.2x) is expected: decoding is memory-bound, and a sparse method's irregular access pattern recovers less of a memory-bound budget than a compute-bound one.

## The ablations that shaped the design

![A kept-versus-dropped ledger: KL loss, gradient detach, indexer warmup, and the local block are KEEP rows in green; the index value head, the forced first-block sink, a large local window, and a learnable sink are DROP rows in red, each with its ablation finding](/imgs/blogs/minimax-sparse-attention-6.png)

The design philosophy is Occam's razor, and the ablations are the razor. The ledger above is the summary; here is the evidence behind each verdict.

**Gradient sources (KEEP the KL).** Two ways to give the indexer a signal were compared. An *Index Branch output head* attaches a value projection to the indexer and adds its attention output to the layer output, training the indexer through next-token prediction. The *KL loss* supervises the selection distribution directly. LM-loss-only preserves short-context ability but performs poorly on long-context retrieval (no objective on the selection itself). KL-only improves retrieval but reduces short-context ability (removing the extra output head shrinks attention capacity). The combination of both was best in the pilot — but the paper then shows that *once the indexer warmup is in place at full scale, the output head is no longer necessary*, so the final recipe keeps KL supervision and **drops the value head**. At inference, the top-$k$ indexer then only needs the block-wise max of $\mathbf{Q}^{\text{idx}}(\mathbf{K}^{\text{idx}})^\top$, avoiding value aggregation and exponentials entirely.

**Gradient detach (KEEP).** Covered above, and worth restating because it is the most consequential single choice: without the stop-gradient, the KL gradient reaches the backbone, LM loss diverges within a few hundred steps, and even when it does not diverge, short-context benchmarks regress via self-distillation. With the detach, the same KL coefficients that caused divergence are stable.

**Indexer warmup (KEEP).** Within the reported training range, the warmed-up run achieves better short-context performance *and* stronger long-context retrieval. A brief full-attention warmup is a strictly better initialization for sparse training, and it doubles as the clean conversion phase for dense checkpoints.

**Forced sink and local window (DROP the priors).** Early experiments hard-coded selection of the first block (the sink) and a fixed local window. Removing both has little effect on quality: reasoning, code, and perplexity are nearly unchanged, and long-context retrieval is comparable. The model *learns* both structures on its own — attention concentrates on the sequence prefix when useful, and nearby tokens stay frequently selected. Strikingly, even without forcing the first block, a visualization of the learned indexer shows every head directing a substantial fraction of its attention to the first token across all layers — the [attention sink emerges](/blog/paper-reading/large-language-model/gated-attention-for-large-language-models-non-linearity-sparsity-and-attention-sink-free) without being asked for. So the final recipe forces only the special incomplete self block, nothing else.

**Learnable attention sink (DROP).** Given that sink behavior emerges, the authors tested a GPT-OSS-style learnable sink logit per head. It absorbs sink mass in some heads but does not fully remove the first-token sink in others, and the downstream-perplexity comparison shows no consistent improvement. Extra parameters, extra complexity, no clear win — dropped.

**Block size (robust).** Varying $B_k$ across 32, 64, 128 while holding the total selected token budget constant has limited impact on quality: perplexity is nearly unchanged and RULER shows no clear degradation. This is the permission slip to use a large $B_k = 128$ for kernel efficiency without paying a retrieval tax — larger blocks raise arithmetic intensity, and the quality cost of coarser granularity is small.

## Critique: the senior-engineer lens

**What is strong.** The discipline. MSA is the rare efficiency paper that refuses to over-engineer: every component is justified by an ablation that shows the model falls over without it, and the components the model can learn on its own (sink, locality) are deliberately *removed*. The stop-gradient finding is the kind of hard-won detail that saves other teams a month of mysterious divergence. The kernel co-design is honest about the gap between FLOPs and milliseconds and reports both. And the architectural bet — per-group block selection on a GQA backbone — is a genuinely good fit for the hardware and for the models everyone already ships. The 1M-token training-stability curves overlapping full attention is a strong, falsifiable claim at a real scale.

**What is weak or unfalsifiable.** The fixed budget of 2,048 tokens per query is the design's load-bearing simplification, and the long-context numbers show its seams: reranking/RAG and multi-hop QA, the tasks that integrate many scattered positions, are precisely where MSA-CPT trails. The paper frames this as "benchmark-dependent," but it is structural — a budget that small cannot, by construction, attend to a thousand independently-relevant positions. There is no experiment that sweeps the budget $kB_k$ against retrieval quality at fixed compute, which is the single most decision-relevant ablation a deployer would want. The 28.4x FLOP number is reported under one head configuration; how it changes with $G$, $d_{\text{idx}}$, and $B_k$ is left to the formula. And "native multimodal" is asserted but the multimodal training mixture and tokenization are not detailed enough to reproduce.

**What ablation is missing.** Three. First, the budget-versus-quality sweep above. Second, a head-to-head against the closest neighbors — DSA, MoBA, NSA, InfLLM-V2 — *on the same backbone and token budget*; the paper compares only to full GQA, so we learn MSA is near-lossless but not whether per-group block selection beats token-level shared selection in practice. Third, a robustness probe of the indexer under adversarial or out-of-distribution long contexts, where a learned selector can fail in ways a fixed window cannot.

**What would change my mind.** If a budget sweep showed that pushing $kB_k$ from 2,048 to, say, 8,192 closed the reranking/RAG gap to under half a point at a still-favorable speedup, I would call the long-context caveat a tuning knob rather than a structural limit — and MSA would become my default recommendation for long-context conversion, not just a strong option. Conversely, if the gap stayed open as the budget grew, that would point to the indexer's recall — not the budget — as the bottleneck, and shift the design effort toward a richer scoring function.

## What I would build with this

1. **A budget-adaptive inference mode.** The training budget ($k = 16$) need not equal the inference budget. Because the indexer is decoupled from the attention, you could raise $k$ at inference time for retrieval-heavy requests and lower it for chat, trading latency for recall per request. The paper hints at "a larger selection budget at inference time" in its outlook; a router that sets $k$ from a cheap query classifier is the obvious product.

2. **Convert an open GQA checkpoint.** MSA-CPT is a recipe, not just a result: take a strong open GQA model, add two index matrices, run the 40B-token warmup, then a few hundred billion sparse tokens. The payoff is a drop-in long-context variant of a model you already trust, at a fraction of from-scratch cost. The per-group, block-level design composes with the GQA backbone most open frontier models share.

3. **A richer indexer score.** The current indexer is a single dot-product with a max-pool. DSA's lightning indexer uses ReLU and a multi-head score; NSA compresses a coarse global branch. A small MLP or a two-level (coarse-then-fine) indexer might lift block recall on the multi-hop tasks where MSA trails, attacking the retrieval gap from the selector side rather than the budget side.

4. **Push the index quadratic term.** At 1M tokens MSA's cost is eventually dominated by the Index Branch's $H_{kv} d_{\text{idx}} N^2$ term, not the main attention. A hierarchical indexer — score coarse super-blocks first, then refine only the promising ones — would make the selection itself sub-quadratic and push the speedup past 28x at the 10M-token scale that agentic workloads are heading toward.

MSA's quiet lesson is that the path to cheap million-token attention does not require a new model class. It requires a GQA layer, two small matrices, one carefully-placed stop-gradient, and kernels written by people who care about arithmetic intensity. That is a recipe most teams can actually follow.

## References

- **Paper:** MiniMax Sparse Attention. arXiv 2606.13392 (June 2026). [PDF](https://arxiv.org/pdf/2606.13392)
- **Code:** MSA inference kernel — [github.com/MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA)
- **Model:** MiniMax-M3, a production multimodal model powered by MSA — [huggingface.co/MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)
- **Sibling posts on this blog:**
  - [DeepSeek-V3.2: DeepSeek Sparse Attention on top of MLA](/blog/paper-reading/large-language-model/deepseek-v3-2) — the closest token-level prior work
  - [MoBA: Mixture of Block Attention for Long-Context LLMs](/blog/paper-reading/large-language-model/moba) — MoE-style top-k routing applied to attention
  - [MiniMax-01: Lightning Attention, the 7:1 Hybrid, and a Million-Token Context](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe) — MiniMax's own efficiency lineage
  - [A Survey on LLM Acceleration based on KV-Cache Management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) — the long-context cost backdrop

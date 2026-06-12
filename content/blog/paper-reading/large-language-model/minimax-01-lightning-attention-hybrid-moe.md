---
title: "MiniMax-01: Lightning Attention, the 7:1 Hybrid, and a Million-Token Context"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - minimax
  - lightning-attention
  - linear-attention
  - mixture-of-experts
  - long-context
  - rope
  - sequence-parallelism
  - vision-language-model
  - moe-routing
  - kv-cache
description: "A deep read of MiniMax-01: how lightning linear attention, a 7-to-1 softmax hybrid, a fat-expert MoE, and LASP+ sequence parallelism combine to train a 456B model at one-million-token context."
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-1.png"
readTime: 30
---

The headline number in the MiniMax-01 report is a million tokens of training context, extrapolating to four million at inference. The interesting number is seven. For every seven linear-attention layers in this 456-billion-parameter model, there is exactly one full softmax-attention layer, and that ratio — not the parameter count, not the context length — is the load-bearing design decision. Get it wrong in one direction and the model cannot recall a fact from a long document; get it wrong in the other and the quadratic cost you were trying to escape creeps back in. MiniMax-01 is the existence proof that 7:1 is enough, and this post is about why.

MiniMax-01 (arXiv [2501.08313](https://arxiv.org/abs/2501.08313), January 2025) ships two models — the text model **MiniMax-Text-01** and the vision-language **MiniMax-VL-01** — on one backbone. It is the foundation that the later [MiniMax-M1 reasoning model](/blog/paper-reading/large-language-model/minimax-m1-cispo-test-time-compute) builds on, and the architecture that [MiniMax-M2 eventually walked back](/blog/paper-reading/large-language-model/minimax-m2-full-attention-agentic). If you want the whole lineage in one place, the [combined MiniMax overview](/blog/paper-reading/large-language-model/minimax-papers-lightning-attention-cispo) is the hub; this post goes deep on the foundation.

![MiniMax-01 at a glance: hybrid lightning attention, a fat-expert MoE, and million-token context](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-1.png)

The diagram above is the mental model: three pillars. The **attention** column is a hybrid where linear "lightning" attention does the bulk of the work and softmax appears once every eight layers. The **MoE** column is a deliberately coarse mixture — 32 fat experts, top-2 routing, no shared expert, capacity-limited with token dropping. The **scale** column is what those two choices buy: 456B total parameters with only 45.9B activated per token, trained at one-million-token context. Everything below is an elaboration of these three pillars.

> [!tldr] TL;DR
> - **What it claims:** A 7:1 hybrid of lightning (linear) attention and softmax attention, on a 456B/45.9B MoE, gives near-flat long-context quality out to 1M training tokens (4M at inference) at roughly linear cost.
> - **Why it matters:** It is the cleanest published recipe for *escaping the quadratic-attention wall without losing recall* — the hybrid ratio, the LASP+ sequence-parallelism trick, and the joint context-and-RoPE warmup are all directly transferable.
> - **Most surprising result:** RULER accuracy stays near **0.91 at one million tokens** where Gemini-1.5-Pro falls to 0.85 and GPT-4o/Claude cannot run at all; needle-in-a-haystack hits 100% at 4M tokens.
> - **Where it's soft:** The training MFU is never disclosed (only an inference figure), the GPU SKU is ambiguous, and the context-warmup schedule rests on a community deep-dive rather than the paper body.

## Context: the quadratic wall and the linear-attention lineage

Every choice in this paper is a reaction to one cost curve. Standard softmax attention forms an $n \times n$ score matrix for a sequence of length $n$. That is $O(n^2 \cdot d)$ in compute and, worse for inference, it grows a KV cache linearly in $n$ that must be streamed from high-bandwidth memory on every decode step. At 4K tokens this is a rounding error. At one million tokens it is the entire budget — the KV cache alone can dwarf the model weights, and the quadratic compute term swamps everything else.

The escape hatch the field has chased for years is **linear attention**. Drop the softmax and write attention as $\phi(Q)\,(\phi(K)^\top V)$ for a feature map $\phi$, and the associativity of matrix multiplication lets you compute $\phi(K)^\top V$ first. That product is a $d \times d$ matrix, reusable across every query, which collapses the cost to $O(n \cdot d^2)$ — linear in sequence length — and replaces the growing KV cache with a fixed-size $d \times d$ recurrent state. MiniMax builds directly on the **Lightning Attention-2** line (arXiv [2401.04658](https://arxiv.org/abs/2401.04658)), which traces back through TransNormer and made the linear-attention kernel genuinely fast on a GPU rather than merely cheap in a complexity table.

The catch is the reason this paper is interesting at all: pure linear attention is bad at *exact recall*. Compressing all of history into a fixed $d \times d$ state is lossy, and retrieval-shaped tasks — pull one specific fact out of a 500K-token document — expose that loss immediately. So the field had two unsatisfying options: quadratic attention that recalls perfectly but does not scale, or linear attention that scales but forgets. MiniMax-01's contribution is to refuse the dichotomy and find the minimal amount of softmax attention that restores recall. The constant backdrop is the DeepSeek-MoE line, which MiniMax measures itself against on both architecture and, later, on its mixture-of-experts granularity.

MiniMax-01 is not the first model to mix attention types — the idea of interleaving a sub-quadratic mixer with occasional full attention is shared with Jamba (Mamba blocks plus a few attention layers), Samba (sliding-window attention plus Mamba), and a wave of 2024 hybrids. What sets MiniMax-01 apart is the *ratio it commits to at scale* and the *choice of linear primitive*. Jamba and Samba reach for state-space models; MiniMax stays with the attention family and uses lightning linear attention, which keeps the implementation close to a standard transformer block and lets the same kernels, the same RoPE, and the same MoE machinery slot in unchanged. The bet is that you do not need a fundamentally different sequence mixer (an SSM) to escape the quadratic — a well-engineered linear attention plus a thin layer of softmax does the job, and stays in a codebase your team already understands. That bet is what makes the report worth reading even if you never run a 456B model: it is a recipe for retrofitting linear attention into an architecture you already have.

A second framing worth holding: there are really two costs the quadratic imposes, and they are not the same. There is the *compute* cost of forming the $n \times n$ matrix during the forward and backward pass — the term that dominates training throughput — and there is the *memory* cost of the KV cache during autoregressive decoding, which dominates inference and caps how long a context you can serve. Lightning attention attacks both at once: linear compute in the forward pass, and a fixed $d \times d$ state instead of a growing cache at decode. The softmax layers in the hybrid reintroduce a little of each, which is exactly why the ratio matters — every softmax layer you add is a layer whose KV cache grows with context again.

## Contributions

Tightened from the report, the contributions are four:

1. **Lightning attention at scale**, the first deployment of an I/O-aware linear-attention kernel inside a frontier-scale (456B) model rather than a research-scale one.
2. **The 7:1 hybrid**, an empirically minimal interleave of one softmax layer per seven lightning layers that recovers exact recall while keeping seven of eight layers at linear cost.
3. **A long-context training stack** — LASP+ sequence parallelism, varlen ring attention, and decoupled expert parallelism — that makes one-million-token training tractable.
4. **MiniMax-VL-01**, a vision-language model that grafts a 303M ViT onto the same hybrid backbone, demonstrating the architecture takes a multimodal extension cleanly.

## Lightning attention: the mechanics

The clearest way to understand lightning attention is to watch where the $n \times n$ matrix never gets formed. Think of it as splitting one causal-attention computation into two cooperating halves, each of which stays cheap for a different reason.

![Lightning attention split into an intra-block left product and an inter-block right product over a d-by-d state](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-2.png)

After a SiLU feature map produces $Q, K, V$, the sequence is tiled into blocks, and the computation factors into two products. The **intra-block left product** handles attention *within* a block: it computes $(Q K^\top) \odot M$ and multiplies by $V$, where $M$ is a decay mask we will dissect in a moment. This block is small, lives in SRAM, and costs $O(B^2 \cdot d)$ for block size $B$ — the only place a score matrix appears, and it is tiny. The **inter-block right product** handles attention *across* blocks by carrying a $d \times d$ running state $KV$: each block updates the state and reads out $Q_{\text{block}} \cdot KV$. History is summarized into that $d \times d$ matrix, never an $n \times n$ one, which is exactly what the overlay annotation in the figure means by "history goes to $d \times d$, not $n \times n$."

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

    idx  = torch.arange(block, device=q.device)
    mask = (decay ** (idx[:, None] - idx[None, :])).tril()  # M_ij = lambda^(i-j), causal
    pos  = decay ** (idx + 1)                               # per-position decay within a block

    for s in range(0, N, block):
        qb, kb, vb = q[..., s:s+block, :], k[..., s:s+block, :], v[..., s:s+block, :]
        intra = ((qb @ kb.transpose(-1, -2)) * mask) @ vb              # LEFT product, SRAM-local
        inter = (qb * pos[:, None]) @ kv                              # RIGHT product, reads d x d
        out[..., s:s+block, :] = intra + inter
        kv = decay**block * kv + (kb * pos.flip(0)[:, None]).transpose(-1, -2) @ vb  # carry state
    return out                                                        # no n x n matrix is ever formed
```

The distinction from FlashAttention matters because the two are constantly confused. FlashAttention is still $O(n^2 \cdot d)$ softmax attention that has been made *I/O-aware* — it tiles to avoid writing the score matrix to HBM, but it still computes the quadratic. Lightning attention changes the *algorithmic complexity* to linear, borrowing the same tiling and SRAM-residency ideas to stay fast. One is a memory optimization of the quadratic; the other replaces it. And the reason the block scheme matters at all: a naive causal linear attention needs a sequential per-token prefix sum (a cumulative sum over the sequence) to build the running state, which destroys GPU parallelism — the block tiling is what restores it, because within a block you can compute the left product as a dense matmul, and across blocks you only carry one $d \times d$ state forward.

### The output gate

One line in the sketch is easy to skim past and is doing real work: `Y = RMSNorm(Y) * sig(X)`. After the attention output is computed, it is normalized with RMSNorm and then multiplied elementwise by a sigmoid of the input — an **output gate** in the SiLU/GLU family. This is not decoration. Linear attention without a gate tends to let the magnitude of the running state drift as the sequence grows, because every block keeps adding to the $d \times d$ accumulator; the sigmoid gate gives each token a learned, input-dependent valve on how much of the accumulated context actually flows into its output. In practice this is what keeps lightning attention numerically stable over hundreds of thousands of tokens — the gate damps the contributions that would otherwise let the state blow up, and it lets the model learn, per token, whether the long-range summary is relevant at all. The SiLU feature map on $Q, K, V$ plays a related role: keeping the features non-negative (or near it) makes the running state behave more like an accumulation of evidence than an unbounded sum of signed terms. These two pieces — the feature map and the output gate — are the difference between a linear-attention layer that trains cleanly at million-token scale and one that diverges, and they are the parts most often dropped from simplified explanations.

### The cost, in numbers

The complexity arguments are easy to wave at; the gap is easier to feel with concrete numbers. Take a single attention layer with head dimension $d = 128$ and sweep the sequence length. The softmax path scales as $n^2 \cdot d$; the lightning path scales as $n \cdot d^2$. The crossover where lightning becomes cheaper is at $n = d = 128$ tokens — past a couple of hundred tokens, the linear path wins, and the lead compounds quadratically.

| Sequence length $n$ | Softmax score-matrix entries ($n^2$) | Lightning state size ($d^2$) | Compute ratio (softmax / lightning) |
| --- | --- | --- | --- |
| 8K | 6.7e7 | 1.6e4 | ~64× |
| 128K | 1.6e10 | 1.6e4 | ~1,000× |
| 1M | 1.1e12 | 1.6e4 | ~8,000× |
| 4M | 1.6e13 | 1.6e4 | ~32,000× |

The right two columns are the real story. The lightning state never grows — it is a $128 \times 128$ matrix whether the context is eight thousand tokens or four million — while the softmax score matrix grows with the square of the length. At one million tokens, a single full-attention layer is forming a matrix with on the order of a trillion entries; a lightning layer is updating a sixteen-thousand-entry state. That is why the architecture can afford seven lightning layers per softmax layer and still come out far ahead: the seven cheap layers cost essentially nothing at long context, and the one expensive layer is the only place the quadratic term reappears. It is also why the KV cache story is so different — at decode time, the lightning layers contribute a constant-size state to the cache, and only the softmax layers contribute a cache that grows with the conversation.

### The decay mask

The mask $M$ in the left product is not a plain causal mask. It is an exponential **decay mask** $M_{ij} = \lambda^{i-j}$, and the way it works is worth a concrete picture.

![The decay mask as a triangular matrix: 1 on the diagonal, lambda powers below, zero above](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-3.png)

Reading the matrix (where $L$ denotes the decay $\lambda$): the diagonal is $1$ — a token attends to itself at full weight. Below the diagonal, each entry is $\lambda$ raised to the distance between query $i$ and key $j$, so nearby tokens get near-full weight and distant ones decay smoothly toward zero. Above the diagonal — the future — is hard-masked to $0$, which is what makes the attention causal. This single matrix is what lets the local left product behave like causal attention with recency bias, without ever materializing a softmax. The decay $\lambda$ is per-head, so different heads can carry different effective memory lengths.

## The 7:1 hybrid and the recall tax

Pure linear attention would have been cheaper still, and MiniMax did not use it, because the $d \times d$ state cannot do precise long-range lookup. The fix is to interleave one full softmax-attention layer after every seven lightning layers.

![A vertical stack of eight transformer layers, seven lightning and one softmax, repeating to 80 layers](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-4.png)

That single softmax layer per block is the **recall tax**. Conceptually, the seven lightning layers do cheap local and decayed-global mixing, and the eighth layer — full attention over the whole sequence — restores the exact, content-addressable lookup that retrieval needs. The bet is that recall is *sparse* in the depth of the network: you do not need every layer to do exact attention, you need a few, placed regularly. MiniMax found that one in eight was enough to keep needle-in-a-haystack and RULER strong while paying linear cost on the other seven.

Why not a different ratio? The trade is mechanical in both directions. Push toward more softmax — say 3:1 or 1:1 — and you reintroduce more of the quadratic compute and, more importantly, more KV cache that grows with context, eroding exactly the long-context economics the architecture exists to provide. Push toward less softmax — 15:1, or pure linear — and recall degrades, because the lossy $d \times d$ state cannot hold enough distinct facts to answer a precise lookup deep in a long document. The 7:1 point is where MiniMax's ablations said the long-context retrieval benchmarks were still saturated while the cost was as low as it could go. The placement matters as much as the count: the softmax layers are spread evenly through the 80-layer stack (layers 8, 16, 24, and so on), not clustered, so that exact lookups are available at multiple depths of abstraction rather than only near the input or the output.

There is a subtlety the report is honest about and the [M2 team later made central](/blog/paper-reading/large-language-model/minimax-m2-full-attention-agentic): the benchmarks that 7:1 saturates are *retrieval* benchmarks. Needle-in-a-haystack and RULER ask the model to find and copy a fact. They do not stress *multi-hop* reasoning, where the model must chain several retrieved facts through intermediate computation, and it is precisely there — at larger scale, on harder reasoning — that a hybrid's compressed state can fall behind full attention in ways a retrieval benchmark never reveals. Hold onto this ratio, because the entire M2 reversal hinges on the discovery that "enough for retrieval" was not the same as "enough for multi-hop reasoning at larger scale."

## The MoE: 32 fat experts, on purpose

MiniMax-01's mixture-of-experts is a near-perfect inversion of the DeepSeek-V3 recipe, and the contrast is the most instructive way to read it.

| MoE design axis | MiniMax-01 | DeepSeek-V3 |
| --- | --- | --- |
| Total / activated params | 456B / 45.9B | 671B / 37B |
| Routed experts | 32 (fat) | 256 (thin) |
| Top-k routing | top-2 | top-8 |
| Shared expert | none | 1 shared |
| Capacity handling | capacity limit + token drop | dropless |
| Load balancing | GShard-style auxiliary loss | aux-loss-free bias |
| Expert FFN hidden | 9216 | ~2048 |

DeepSeek bets on *many thin experts* with a shared always-on expert and dropless routing. MiniMax-01 bets on the opposite corner: **32 fat experts** (expert FFN hidden 9216), top-2 routing, no shared expert, and a hard per-expert capacity that drops overflow tokens, balanced with a classic GShard auxiliary loss. The figure below traces a single token through it.

![MoE routing: a token goes to the top-2 of 32 experts, with overflow tokens dropped](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-5.png)

A softmax router scores all 32 experts and sends the token to its two best; the chosen experts' outputs are combined by the gate weights. The danger branch in the figure is the part that distinguishes this recipe: each expert has a fixed capacity, and a token routed to an already-full expert is simply **dropped** — it skips the FFN entirely for that layer. Token-drop is a training-efficiency lever (it bounds the worst-case compute and communication per expert) that MiniMax accepts where DeepSeek fights to avoid it. A minimal sketch of the routing:

```python
import torch
import torch.nn.functional as F

def moe_route(h, gate_w, experts, capacity, k=2):
    # h: [T, d] tokens;  gate_w: [d, E] router;  experts: list of E callables
    logits = h @ gate_w                                   # [T, E]
    weights, idx = logits.softmax(-1).topk(k, dim=-1)     # top-2 experts per token
    out, load = torch.zeros_like(h), torch.zeros(len(experts))
    for slot in range(k):
        for e in range(len(experts)):
            sel = (idx[:, slot] == e).nonzero().squeeze(-1)
            sel = sel[: int(capacity - load[e])]          # CAPACITY LIMIT: overflow is dropped
            load[e] += sel.numel()
            out[sel] += weights[sel, slot, None] * experts[e](h[sel])
    aux = (logits.softmax(-1).mean(0) * (load / load.sum())).sum() * len(experts)  # GShard balance
    return out, aux
```

The token-drop choice deserves more than a shrug, because it is the part practitioners get nervous about. Dropping a token means that, for one layer, that token's representation passes through unchanged (a residual-only path) instead of being transformed by an expert. Intuitively that sounds like silent data loss, and at high drop rates it would be. The reason it is tolerable is twofold. First, the drop is *per layer*: a token dropped at layer 12 is very likely routed and processed at layers 20, 28, and so on, so over 80 layers the chance a token is starved everywhere is negligible. Second, the capacity factor is tuned so the drop rate is low in expectation — overflow is the exception, triggered by transient routing imbalance, not the rule. What you buy for that small risk is a hard ceiling on per-expert compute and communication: with a fixed capacity, every expert processes a bounded number of tokens, so the all-to-all payload is predictable and the slowest expert cannot stall the whole batch. DeepSeek's dropless approach removes the risk but pays for it with variable per-expert load and the engineering to handle it.

The auxiliary loss is the other half of keeping the router honest. Left alone, a softmax router collapses — it discovers a few experts that work and sends everything to them, leaving the rest untrained. The GShard-style auxiliary term, sketched in the `aux` line above, multiplies the fraction of tokens routed to each expert by that expert's mean gate probability and sums over experts; minimizing it pushes the router toward uniform utilization. It is added to the main loss with a small coefficient (undisclosed here), and it is the lever that prevents routing collapse without the bias-term machinery DeepSeek-V3 uses for its aux-loss-free balancing. The absence of a *shared* expert is the last distinctive choice: DeepSeek keeps one expert always-on to absorb common computation, freeing the routed experts to specialize; MiniMax declines, betting that 32 fat experts have enough capacity each that a dedicated common-knowledge expert is not worth the always-on FLOPs.

Neither granularity is obviously right; they are different points on a granularity-versus-overhead curve. MiniMax's choice trades the routing flexibility of many experts for lower all-to-all communication volume and simpler kernels. If you are designing an MoE, this table is the decision you actually have to make, and MiniMax-01 is the existence proof for the "fewer, fatter, drop-tolerant" corner. (For the mechanics of MoE communication and balancing in more depth, the blog's [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) post goes layer by layer.)

## Training infrastructure for a million tokens

A one-million-token training context is a systems problem before it is a modeling one — the activations alone are enormous, and the linear-attention state has a recurrence that does not parallelize for free.

![Before-and-after of serial send-recv KV passing versus LASP+ AllGather with varlen packing](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-6.png)

The first enabler is **LASP+** (Linear Attention Sequence Parallelism Plus). Naive sequence parallelism for a recurrent state is serial: rank 0 computes its KV block, sends it to rank 1, which must wait before it can compute, and so on down the chain — most GPUs idle most of the time. LASP+ replaces the send-recv ring with an **AllGather** of the KV state across ranks, so every rank computes its block in parallel instead of waiting its turn. The second enabler is **varlen ring attention**: rather than padding every sequence in a packed batch to the longest length — ruinous at million-token scale — ring attention runs directly on the concatenated variable-length sequence, so the padding tax disappears.

On the MoE side, the parallelism is deliberately decoupled. MiniMax runs three distinct process groups: **expert parallel** (EP) shards which experts live on which GPUs, **expert tensor parallel** (ETP) shards an individual expert's weight matrices across devices, and **expert data parallel** (EDP) replicates experts to spread the data batch. Keeping these as separate groups, rather than folding them into one monolithic mesh, lets MiniMax tune each axis independently — for example, widening ETP when an expert is too large to fit on one device, without disturbing how experts are distributed. The communication crux is the all-to-all that shuffles tokens to their assigned experts and back: it is overlapped with computation by grouping tokens so that while one expert group's tokens are in flight over the network, another group's tokens are being processed on-device. Hiding that all-to-all behind useful compute is the single most important systems trick for keeping an MoE from being communication-bound, and the decoupled groups are what make the overlap schedulable.

One number is easy to misquote, so state it carefully: MiniMax reports **greater than 75% MFU**, but that figure is for end-to-end *inference* on H20 hardware, not training. A training-time MFU is not disclosed, and neither is the precise GPU SKU — community analyses disagree between H800 and H100 for a dynamically sized cluster of roughly 1,500 to 2,500 GPUs. When you cite MiniMax-01's efficiency, cite the inference MFU as an inference number.

## Data, RoPE, and staged alignment

Reaching a million tokens is a staged warmup, and it is the part most worth copying for anyone extending context.

![Timeline of context length and RoPE base stepping together, then short-to-long SFT and RL](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-7.png)

The reported schedule steps sequence length and RoPE base *together*: main pretraining at 8K context with RoPE base 10K, then a phase at 128K with the base bumped to 5M, then 512K and 1M phases with the base at 10M. RoPE is applied to only half of each head's dimensions, and the base-frequency bumps are what let the model extrapolate cleanly — under the hood, context extension here is as much a RoPE-base *schedule* as an architectural property. (These exact phase values come from the community deep-dive rather than the paper body, so treat them as the reported recipe, not gospel.) Pretraining runs on roughly 12 trillion tokens with a WSD-like learning-rate schedule that decays only to 10% of peak, a batch-size warmup from 16M to 128M tokens, and a quality-scoring pass that up-weights knowledge-rich data and repeats the highest-quality sources four times.

Two pieces of that recipe are worth dwelling on because they generalize. The **WSD-like schedule** (warmup, stable, decay) keeps the learning rate at a high stable plateau for most of training and decays it only at the end — and MiniMax decays only to 10% of peak rather than to zero. The practical advantage is that a WSD run can be *extended*: because the bulk of training happens at a constant rate, you can append more tokens or more context-length phases without re-cooking a cosine schedule from scratch, which is exactly what a staged context warmup needs. The **data quality scoring** uses a previous-generation MoE model as a labeler that scores documents on axes like coherence, educational value, knowledge richness, and category, then up-weights the high-scoring data and repeats the best of it. The interesting choice is repeating high-quality data four times rather than diluting it with more mediocre tokens — a bet that, past a certain scale, *what* you train on matters more than raw token count, and that a few extra epochs on genuinely good data beats one epoch on a larger, noisier pile.

The partial-RoPE application — half the head dimensions get rotary, half stay untouched — is a small detail with outsized effect on extrapolation:

```python
import torch

def partial_rope(x, cos, sin, rotary_dim):
    # x: [B, H, T, D]; only the first `rotary_dim` channels get rotated, base e.g. 1e7
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    x_rot = x_rot * cos + rotated * sin                  # standard rotary on the rotary half
    return torch.cat([x_rot, x_pass], dim=-1)            # the pass-through half carries no position
```

The alignment ordering is the second transferable idea: **short-context SFT, then long-context SFT, then short-context RL, then long-context RL** (offline DPO, then online GRPO). Long context is not bolted on at the end; it is sequenced as its own alignment stage so the model learns to use the window it was trained to hold.

## MiniMax-VL-01: grafting on vision

The same backbone takes a vision graft without drama, which is itself the point.

![MiniMax-VL-01: a 303M ViT and a two-layer MLP projector feed visual tokens into the Text-01 backbone](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-8.png)

MiniMax-VL-01 reuses Text-01 as the language backbone and adds a **303M ViT** (24 layers, patch size 14, hidden 1024) through a randomly initialized **two-layer MLP projector**. Images are handled at dynamic resolution — resized on a grid from 336×336 up to 2016×2016, with a 336×336 thumbnail always kept — and the model is trained on **512B vision-language tokens** across a four-stage process: the new ViT and adapter get visual pretraining first, then the full pipeline is continued-pretrained and fine-tuned.

The dynamic-resolution scheme is the part worth copying. A naive vision model resizes every image to a fixed square, which destroys aspect ratio and throws away detail on large images (a dense document page) while wasting tokens on small ones (an icon). MiniMax-VL-01 instead picks a resolution from a preset grid based on the image, splits the resized image into non-overlapping patches that the ViT encodes, and *always* keeps a 336×336 thumbnail so the model has a global view alongside the high-resolution patches. The thumbnail-plus-patches construction is what lets the same model handle both a 4K screenshot and a small chart without a separate code path. The ViT itself was trained on 694M image-caption pairs, with refined captions sampled 50/50 against raw ones — a data trick that keeps the model fluent in both the terse alt-text distribution of the web and the richer descriptions a captioner produces. The staged schedule (vision-only first, then joint) is the standard guard against the randomly initialized projector destabilizing the pretrained backbone: you let the adapter and ViT find a sane alignment before you let gradients flow into the language model.

The noteworthy part is simply that the hybrid-attention backbone accepts the graft and posts competitive numbers against GPT-4o, Claude-3.5-Sonnet, and Qwen2-VL-72B:

| VL benchmark | MiniMax-VL-01 | GPT-4o | Claude-3.5-Sonnet | Qwen2-VL-72B |
| --- | --- | --- | --- | --- |
| MMMU | 68.5 | 63.5 | 72.0 | 64.5 |
| ChartQA | 91.7 | 88.1 | 90.8 | 91.2 |
| DocVQA | 96.4 | 91.1 | 94.2 | 97.1 |
| OCRBench | 865 | 806 | 790 | 856 |

That a vision model bolted onto a *linear-attention* backbone is competitive on document and chart understanding — tasks that lean heavily on dense, high-resolution detail — is a quiet vindication of the hybrid: the periodic softmax layers carry enough exact spatial lookup for OCR-style tasks to work.

## Experiments

The headline is long context, and the headline number is genuinely strong: on RULER, MiniMax-Text-01 holds accuracy nearly flat as context grows, where the competition falls off a cliff.

![RULER accuracy by context length: MiniMax stays near 0.91 to 1M tokens while Gemini drops to 0.85](/imgs/blogs/minimax-01-lightning-attention-hybrid-moe-9.png)

The matrix tells the marquee story: MiniMax-Text-01 sits near 0.91 at one million tokens where Gemini-1.5-Pro has decayed to 0.85, and GPT-4o and Claude cannot run at those lengths at all. On the 4M-token vanilla needle-in-a-haystack retrieval, MiniMax reports 100%. The core-knowledge scores are competitive-but-not-leading:

| Benchmark | MiniMax-Text-01 | GPT-4o | Claude-3.5-Sonnet | DeepSeek-V3 |
| --- | --- | --- | --- | --- |
| MMLU | 88.5 | 85.7 | 88.3 | 88.5 |
| MMLU-Pro | 75.7 | 74.4 | 78.0 | 75.9 |
| GPQA-Diamond | 54.4 | 46.0 | 65.0 | 59.1 |
| GSM8K | 94.8 | 95.6 | 96.9 | 96.7 |
| MATH | 77.4 | 76.6 | 74.1 | 84.6 |
| LongBench-v2 (CoT) | 56.5 | 51.4 | 46.7 | — |

The pattern is consistent: MiniMax-01 is in the pack on raw knowledge (MMLU 88.5, neck-and-neck with DeepSeek-V3) and out in front on anything that stresses the context window (LongBench-v2 56.5 vs Claude's 46.7). The load-bearing architectural claim — the hybrid keeps recall while paying linear cost — holds up on these retrieval-shaped benchmarks. Whether it holds on *multi-hop reasoning* at scale is the question the M2 report answers later, uncomfortably.

## How it compares: hybrids and the long-context field

MiniMax-01 lands in a crowded field of attempts to escape the quadratic, and it helps to place it against the main alternatives, because each makes a different bet about *which* part of the cost to attack and *what* to give up.

| Approach | Sequence mixer | Long-context cost | Exact recall | Notes |
| --- | --- | --- | --- | --- |
| Full softmax (GPT-4o, Claude) | softmax every layer | $O(n^2)$, growing KV cache | excellent | the recall baseline; does not reach 1M cheaply |
| Sliding-window attention | local softmax | linear, bounded cache | poor at long range | cheap, but global lookups fail past the window |
| Pure linear / SSM (Mamba) | linear / state-space | $O(n)$, fixed state | weak | fast, but loses precise retrieval |
| SSM hybrid (Jamba, Samba) | Mamba + a few attention | near-linear | good | escapes quadratic with a non-attention primitive |
| **MiniMax-01 (7:1 lightning hybrid)** | lightning linear + 1/8 softmax | near-linear, small cache | strong (retrieval) | stays in the attention family; recall via the 1/8 softmax |

The two rows worth contrasting directly are sliding-window attention and MiniMax-01, because they look similar — both make most layers cheap — but fail differently. Sliding-window attention keeps a bounded cache by only attending to a local window, which means a token at position one million simply cannot see a token at position one; global lookups past the window are impossible by construction. MiniMax-01's lightning layers, by contrast, carry a decayed *global* summary in the $d \times d$ state, and the periodic softmax layers can attend across the entire sequence, so a long-range lookup is degraded but not forbidden. That is the difference between "cheap and globally blind" and "cheap and globally fuzzy, with periodic sharp lookups" — and it is why MiniMax holds RULER at 1M where a sliding-window model would collapse. The SSM hybrids (Jamba, Samba) reach a similar destination by a different road, swapping the linear primitive for a state-space model; MiniMax's argument is that you can get there without leaving the attention family at all, which keeps the engineering surface small.

## Critique

What is strong is the clarity of the bet. The 7:1 hybrid is a falsifiable, minimal claim — "one softmax layer per eight is enough" — and the RULER curve is the kind of result that survives scrutiny because it degrades gracefully rather than cliff-edging. The LASP+ and varlen-ring infrastructure is genuinely reusable, and the joint context-and-RoPE warmup schedule is a recipe you can lift wholesale. The MoE granularity choice is a clean counterpoint to DeepSeek that makes the design space legible.

What is soft is the transparency around cost and the provenance of the recipe. The training MFU is never given — only the inference figure, which is easy to mistake for a training number — and the GPU SKU is left ambiguous. The most load-bearing parts of the long-context recipe, the staged RoPE-base schedule and the data quality-scoring pass, are sourced from a community deep-dive rather than the paper body, which means the exact phase boundaries should be treated as approximate. The MoE auxiliary-loss coefficient and the expert capacity factor — the two numbers that determine how much token-drop actually happens — are not disclosed, so you cannot reproduce the balance behavior.

**What would change my mind** about the "7:1 is the right ratio" conclusion: a published ablation, scale-matched, showing multi-hop reasoning (not just retrieval) as a function of the lightning-to-softmax ratio. The RULER curve proves the hybrid recalls; it does not prove the hybrid *reasons*, and the absence of that ablation is exactly the gap the M2 team later says they fell into. If a 3:1 or 1:1 hybrid closed a multi-hop gap that 7:1 leaves open while keeping most of the FLOP savings, the "7:1 is enough" framing would need revising — which is, in effect, what happened.

The redeeming feature on the transparency front is that the weights are open. MiniMax-Text-01 and VL-01 ship on Hugging Face with a config you can read, so the architecture facts — 80 layers, the 7:1 `attn_type_list`, 32 experts with FFN hidden 9216, RoPE on half the head dimensions at base 10M — are verifiable even where the *training* recipe is not. That is a meaningful split: the parts of the report you most want to copy (the architecture) are reproducible from the open weights, while the parts that are hardest to reproduce anyway (the exact data mix, the cluster, the MFU) are the parts left vague. For a practitioner, the open config plus the LASP+ and varlen-ring descriptions are enough to rebuild the inference path and most of the training path; what you cannot recover is the precise data curation that turned the architecture's *capacity* for long context into its measured long-context *quality*. That gap — architecture disclosed, data withheld — is the standard shape of a 2025 open-weights release, and MiniMax-01 is a fairly generous example of it.

## What I'd build with this

1. **A ratio sweep on a small model.** Train 1:1, 3:1, 5:1, and 7:1 lightning-to-softmax hybrids at a few hundred million parameters and measure both needle-in-a-haystack *and* a multi-hop reasoning suite. The paper gives you the 7:1 endpoint; the curve is the interesting object and nobody has published it cleanly.

2. **LASP+ as a drop-in for any recurrent-state layer.** The AllGather-instead-of-ring trick is not specific to lightning attention — any sequence-parallel layer with a carried state (Mamba-style SSMs included) inherits the same serial bottleneck, and the same fix.

3. **The decay mask as a tunable memory knob.** Since $\lambda$ is per-head, you can probe what each head's effective memory length converges to after training, and potentially initialize or regularize it to encode a deliberate spread of short- and long-memory heads.

4. **The staged RoPE-base schedule as a context-extension recipe.** Lift the 10K → 5M → 10M base schedule and the matching context steps directly onto an existing dense model to extend its window, decoupled from the rest of the architecture.

5. **A hybrid serving path that exploits the asymmetry.** Because only the softmax layers grow a KV cache with context, a serving stack could cache aggressively for those one-in-eight layers and treat the lightning layers as constant-state recurrences — a prefix-caching strategy specialized to the hybrid that a uniform full-attention model cannot use. This is, incidentally, the opposite of the conclusion [M2 reached for agentic serving](/blog/paper-reading/large-language-model/minimax-m2-full-attention-agentic), and testing where the crossover actually lies is the experiment that would settle the hybrid-versus-full debate.

One closing note on what is load-bearing in this setup that may not transfer. The long-context numbers rest on a *training* context of one million tokens, which only the LASP+ and varlen-ring infrastructure made affordable; a team without that infrastructure cannot simply adopt the 7:1 architecture and expect the same RULER curve, because the recall ability is learned during long-context training, not granted by the architecture alone. The architecture makes long context *cheap*; the staged training is what makes it *good*. Both halves are necessary, and the report's results are a property of the pair, not of the attention pattern in isolation.

## References

- MiniMax-01: *Scaling Foundation Models with Lightning Attention* — arXiv [2501.08313](https://arxiv.org/abs/2501.08313) · [GitHub](https://github.com/MiniMax-AI/MiniMax-01) · [Text-01 card](https://huggingface.co/MiniMaxAI/MiniMax-Text-01) · [VL-01 card](https://huggingface.co/MiniMaxAI/MiniMax-VL-01)
- Lightning Attention-2 — arXiv [2401.04658](https://arxiv.org/abs/2401.04658)
- Sibling MiniMax reads on this blog: [the combined overview](/blog/paper-reading/large-language-model/minimax-papers-lightning-attention-cispo) · [MiniMax-M1 and CISPO](/blog/paper-reading/large-language-model/minimax-m1-cispo-test-time-compute) · [MiniMax-M2's full-attention reversal](/blog/paper-reading/large-language-model/minimax-m2-full-attention-agentic)
- Related: [Optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) · [Modern LLM architectures](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) · [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management)

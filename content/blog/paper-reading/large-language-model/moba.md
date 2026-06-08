---
title: "MoBA: Mixture of Block Attention for Long-Context LLMs"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - moba
  - long-context
  - sparse-attention
  - mixture-of-experts
  - flashattention
  - kimi
  - moonshot-ai
  - attention
  - paper-reading
description: "A close read of MoBA, Moonshot AI's block-sparse attention that applies MoE-style top-k routing to attention, matching full-attention quality at up to 95.31% sparsity and 6.5x faster 1M-token prefill."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/moba-1.png"
readTime: 32
---

The expensive part of running a long-context language model is not the parameters; it is the attention. A model with eight billion weights will happily fit in a single high-memory GPU, but ask it to read a one-million-token document and the attention computation alone — every query attending to every key — grows as the square of the sequence length. Double the context and you quadruple the attention cost. At 128K tokens this is already painful. At 1M tokens it is the dominant term in the prefill budget, and at the 10M-token scale that some agentic and document-analysis workloads now demand, full attention is simply not a thing you can afford to run.

The field has spent years trying to escape that quadratic wall, and the escape routes have mostly looked like one of two things. The first is to hard-code where the model is allowed to attend: a sliding window that only looks at the last few thousand tokens, or an "attention sink" that always keeps the first few tokens around. These work, but they bake a human guess about relevance into the architecture, and that guess fails the moment the task wants a needle buried 600K tokens back. The second route is to rewrite attention into something cheaper and fundamentally different — linear attention, state-space models, recurrent approximations — which changes the model class entirely and whose behavior on hard reasoning tasks is, charitably, still being mapped out. Neither route gives you the thing you actually want: a drop-in replacement for full attention that keeps full-attention quality, costs sub-quadratic compute, and lets you flip between sparse and dense at will.

MoBA — Mixture of Block Attention, from the Moonshot AI / Kimi team (arXiv 2502.13189, February 2025) — is a serious attempt at exactly that. The core move is almost embarrassingly simple to state once you have seen Mixture-of-Experts: take the routing idea that lets an MoE feed-forward layer pick a few experts per token, and apply it to attention. Partition the context into blocks, and let each query token *learn* which blocks of keys and values it should attend to, picking only the top-k of them. No new parameters, no hand-coded window, no rewrite of the attention math — just a learned, parameter-free gate sitting in front of an otherwise standard FlashAttention call. MoBA is already deployed to serve Kimi's long-context requests, which is the strongest signal a paper of this kind can carry: it survived contact with production.

![How MoBA routes one query to its top-k blocks: parameter-free gating over mean-pooled key blocks, then FlashAttention on the selected union](/imgs/blogs/moba-1.png)

The diagram above is the mental model: a single query token does not attend to the whole context. It scores every block by the inner product between itself and that block's mean-pooled keys, keeps only the top-k blocks (plus, always, the block it currently lives in), and then runs ordinary FlashAttention over the union of just those selected blocks. The gate is parameter-free — it is one mean-pool and one dot product — so MoBA adds nothing to the weight matrix and can be switched on or off without retraining the model from scratch. Everything else in this article is an elaboration of that one picture: how the gate is made causally correct, how the kernel is made fast, how the model is trained so that sparsity does not cost quality, and how well it actually works.

> [!tldr] TL;DR
> - **What it claims:** Applying MoE-style top-k routing to attention — partition context into blocks, let each query learn which blocks to attend to — matches full-attention quality up to 1M-token context at sub-quadratic cost, with zero added parameters.
> - **Why it matters:** It is a parameter-free drop-in. Because MoBA is weight-compatible with full attention, you can train with it, switch back to full attention for the last 10% of tokens, and switch per-layer during SFT — no architecture fork, no loss spike.
> - **Most surprising finding:** At up to **95.31% sparsity** on a 1M-context Llama-8B, downstream benchmark scores track full attention within a few hundredths, and on several tasks (GSM8K, CEval, Loogle) MoBA is actually *ahead*. The validation-loss gap across the scaling-law family stays within **1e-3**.
> - **Where it fails:** It is **not** a zero-cost inference swap — you must continually train an existing model to adopt it. Pure MoBA shows slightly worse trailing-token loss and needs hybrid recipes (stage-wise and layer-wise full attention) to fully match dense attention; for best decode quality on the 8B model, generation still runs full attention.

## Context: what came before

To see why MoBA's "less structure" framing is the right one, it helps to lay out the design space it is reacting against. Long-context attention research has clustered into three families, and MoBA defines itself by rejecting the constraints of the first two while borrowing the routing idea from a fourth (MoE) that is not even about attention.

![Where MoBA sits in the long-context attention design space: it learns a block gate that subsumes the fixed sliding-window and attention-sink patterns](/imgs/blogs/moba-5.png)

The first family is **predefined structural sparsity**. Sliding-window attention (Beltagy et al., 2020) lets each token attend only to a fixed window of recent tokens; the attention-sink phenomenon (Xiao et al., 2023) observes that keeping the first few tokens of the sequence "anchored" stabilizes streaming generation, so those tokens are always attended. These methods are effective and cheap, but they hard-code *where* attention is allowed to go. That is a strong, task-specific bias. If your task needs a fact that lives outside the window and is not one of the sink tokens, the architecture has decided in advance that you cannot see it. The diagram above places these as special cases for a reason: MoBA's authors show that both sliding window and attention sink are recoverable as particular settings of MoBA's gate — a gate that always selects the local block reproduces sliding window; a gate that always selects the initial and most recent blocks reproduces attention sink. MoBA generalizes the family by making the gate *learned* instead of fixed.

The second family is **linear and radically modified attention** — linear attention, kernelized approximations, and the broader state-space lineage that replaces softmax attention with something whose cost is linear in sequence length. These are genuinely sub-quadratic and have produced strong results, but they change the model class. The paper's stance, and it is a defensible one, is that the performance of these radically modified mechanisms on complex reasoning tasks remains inadequately explored, and that adopting one is a bet on a different architecture rather than a refinement of the one you already trust. (If you want the most recent serious entry in this lineage, the same lab's [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) is a good companion read; MoBA and Kimi Linear are two different answers to the same long-context cost problem.)

The third family — where MoBA lives — is **learned sparse attention**: keep softmax attention exactly as it is, but let the model decide, per query, which subset of the context to actually compute attention over. The conceptual debt here is to Mixture-of-Experts. An MoE feed-forward layer does not run all experts for every token; a router scores the experts and a top-k gate sends each token to only a few. MoBA asks the obvious question nobody had productized at this scale: what if the "experts" are *blocks of the context*, and the router is the attention query itself? The design principle the authors name is "less structure" — rather than imposing predefined biases, let the model autonomously decide where to attend. The empirical license to do this comes from a property the paper leans on heavily: attention scores are *inherently sparse*. Sparsity arises both mathematically (softmax concentrates mass) and empirically (most query-key pairs contribute negligibly), so computing dense attention is largely wasted work. MoBA's bet is that a learned gate can find the blocks that matter and skip the rest without losing the ones that do.

The gap MoBA fills, then, is the intersection of three properties that prior work hit at most two of at a time: (a) match full-attention quality on hard tasks, (b) be sub-quadratic, and (c) be a parameter-compatible drop-in that can switch freely between sparse and full attention. Sliding window is (b) and (c) but not (a) on retrieval-heavy tasks. Linear attention is (b) but neither (a)-with-confidence nor (c). MoBA aims for all three, and the rest of this read is an evaluation of whether it gets there.

## Contributions

The paper makes four contributions that are worth separating, because they are at different levels of the stack and each could stand or fall independently.

1. **A parameter-free, MoE-style block gate for attention.** The central idea: partition context into blocks, compute a per-block affinity from the query and the block's mean-pooled keys, and attend only to the top-k blocks. This adds no parameters and changes no attention math — it is a pure substitute for the attention module. This is the load-bearing contribution; everything else supports it.

2. **A causally-correct formulation for autoregressive language modeling.** A naive top-k block gate would let a token attend to future blocks and leak information. MoBA's gating masks future blocks to $-\infty$, always forces selection of the *current* block (the one containing the query), and applies an intra-block causal mask. This makes the gate safe for left-to-right LM training while also nudging the model toward local attention.

3. **An efficient kernel (Algorithm 1) built on MoE routing plus FlashAttention.** The production `moba_efficient` path splits KV into blocks, computes gating with a causal mask, routes/permutes query tokens by their assigned block, runs `flash_attention_varlen` separately on the current block (causal) and the historical selected blocks (non-causal), and recombines partial outputs with online softmax. The README reports this kernel is roughly **40x** faster than a naive implementation at 32K.

4. **Hybrid training recipes that let sparse and full attention coexist.** Because MoBA is weight-compatible with full attention, the paper introduces a *stage-wise* hybrid for pretraining (90% of tokens with MoBA, final 10% with full attention) and a *layer-wise* hybrid for supervised fine-tuning (last few layers full attention, earlier layers MoBA). These recipes close the small quality gap that pure MoBA leaves on long-context-sensitive metrics.

Taken together, the contribution is not a new model — it is a new *attention module* plus the training discipline that makes it safe to deploy. That distinction matters for how you read the limitations later.

## Method

The method has four moving parts: block partitioning, the gate, the causal-correctness rules, and the efficient kernel. We will define each symbol on first use and build the picture from the gate outward.

### Block partitioning and the gate

Let the context have length $N$ and partition it into $n$ blocks of equal size $B = N/n$. The $i$-th block covers the token index range $I_i = [(i-1)B + 1,\ iB]$. So with $N = 32{,}768$ and $B = 512$ you get $n = 64$ blocks; with the 1M-context production model and $B = 4096$ you get roughly $244$ blocks.

For a single query token with vector $q$, MoBA computes an affinity score to each block $i$ as the inner product between the query and the *mean-pooled keys* of that block:

$$
s_i = \langle q,\ \text{mean\_pool}(K[I_i]) \rangle = \left\langle q,\ \frac{1}{B}\sum_{t \in I_i} k_t \right\rangle .
$$

The mean-pooled key $\bar{K}_i = \frac{1}{B}\sum_{t \in I_i} k_t$ is a single $d$-dimensional summary of the whole block. This is the crucial cheap trick: instead of scoring the query against every one of the $B$ keys in a block, you score it against *one* averaged key per block. The gate value is then a hard top-k indicator:

$$
g_i = \begin{cases} 1 & s_i \in \text{Topk}\big(\{s_j : j \in [n]\},\ k\big) \\ 0 & \text{otherwise} \end{cases}
$$

and the query attends only to the union of selected blocks. Writing $I = \bigcup_{i : g_i = 1} I_i$ for the set of selected token indices, the MoBA output for that query is plain softmax attention restricted to $I$:

$$
\text{MoBA}(q, K, V) = \text{Softmax}\big(q\, K[I]^\top\big)\, V[I].
$$

Notice what is and is not here. There is no learned router weight matrix — the "router" is the query vector you already have. There is no temperature, no soft mixing of blocks, no auxiliary load-balancing loss. The gate is a mean-pool, a dot product, and a top-k. That is the entire reason MoBA can claim to be parameter-free and a true drop-in.

### Why the current block is always selected, and other causal rules

A top-k gate as written above is not yet safe for autoregressive language modeling, because nothing stops a query at position $t$ from selecting a block that lies *ahead* of $t$ — which would leak future tokens into the prediction of the present one. MoBA fixes this with three rules, and they are worth stating precisely because they are where most of the subtlety lives.

First, **future blocks are masked out of the gate**. Any block $i$ whose token range lies entirely after the query's position has its score $s_i$ set to $-\infty$ before the top-k, so it can never be selected. Second, **the current block is always selected** — the block $I_i$ that *contains* the query position $t$ gets $g_i = 1$ unconditionally. This guarantees every token can always see its immediate local context, which is exactly the thing sliding-window attention hard-codes; MoBA gets it for free as a forced gate. Third, **within the current block, a causal mask is applied** so that a token at position $t$ attends only to positions $\le t$ inside that block, never to the not-yet-generated tail of its own block.

The combination is both safe and pedagogically tidy. The historical blocks (those entirely before the current one) need no causal mask, because every token in them precedes the query — so they can be computed with `causal=False`. The current block needs `causal=True`. This clean split between "current block, causal" and "history blocks, non-causal" is precisely what the efficient kernel exploits. With a top-k of, say, 3, each query ends up attending to at most two history blocks plus the current block — a tight, bounded read no matter how long the context is.

![Full attention pays for every key-value pair; MoBA reads only the top-k blocks, dropping over 95 percent of the work at 1M tokens](/imgs/blogs/moba-2.png)

The before/after above is the cost story in one frame. Full attention (left) has every query read all $N$ keys and values, costs $O(N^2)$, and skips no work. MoBA (right) has each query read only its top-k blocks — at the 1M-context setting that is 12 blocks of 4096 tokens out of ~244 blocks — which is sub-quadratic, delivers a 6.5x prefill speedup at 1M tokens, and runs at **95.31% sparsity** while still matching full-attention quality. The sparsity figure is not a hand-wave; it is exactly $1 - (4096 \times 12)/1{,}000{,}000 = 0.9531$, i.e. one minus the fraction of tokens any query actually attends to.

### A reference implementation of the gate

Here is the naive-but-correct version of the gate and attention — the `moba_naive` shape, written for clarity rather than speed. It makes the math above concrete and shows exactly where the causal rules land. (Comments are kept inside the function body so they never start a Markdown heading.)

```python
import torch
import torch.nn.functional as F

def moba_naive(q, k, v, block_size, topk):
    """Reference MoBA for one attention head.

    q, k, v : (S, d) tensors for a single head, S = sequence length.
    block_size : B, number of tokens per block.
    topk : k, number of blocks each query may attend to.
    Returns (S, d) attention output.
    """
    S, d = q.shape
    n_blocks = (S + block_size - 1) // block_size

    blk_id = torch.arange(S, device=q.device) // block_size   # block of each token
    mean_k = torch.zeros(n_blocks, d, device=q.device)
    for b in range(n_blocks):
        mean_k[b] = k[blk_id == b].mean(dim=0)                # K_bar_i, one key per block

    scores = q @ mean_k.t()                                   # (S, n_blocks) gate affinity

    q_blk = blk_id.view(S, 1)                                 # block index of each query
    cand = torch.arange(n_blocks, device=q.device).view(1, -1)
    future = cand > q_blk                                     # blocks strictly ahead -> illegal
    scores = scores.masked_fill(future, float('-inf'))        # rule 1: no future blocks

    gate = torch.zeros_like(scores, dtype=torch.bool)
    legal_topk = min(topk, n_blocks)
    sel = scores.topk(legal_topk, dim=-1).indices             # top-k legal blocks per query
    gate.scatter_(1, sel, True)
    gate.scatter_(1, q_blk, True)                             # rule 2: force current block

    out = torch.zeros_like(q)
    for i in range(S):
        keep = gate[i][blk_id]                                # token-level mask from block gate
        keep = keep & (torch.arange(S, device=q.device) <= i)  # rule 3: causal within reach
        kk, vv = k[keep], v[keep]
        a = (q[i:i+1] @ kk.t()) / (d ** 0.5)
        out[i] = (a.softmax(dim=-1) @ vv).squeeze(0)
    return out
```

This is $O(N^2)$ in the worst case because of the per-query loop and the token-level mask, which is exactly why it is only a reference. The production kernel never materializes a dense $S \times S$ mask; it routes tokens by their block assignment and calls FlashAttention on the resulting variable-length groups, which we turn to next.

### The efficient kernel (Algorithm 1)

The reference above wastes everything that makes block sparsity fast: it builds a dense mask and loops per query. The `moba_efficient` kernel does the opposite — it never looks at a pair of tokens that the gate did not select, and it expresses the surviving work as two batched FlashAttention calls.

![The moba_efficient kernel: five stages of Algorithm 1, recombined by online softmax](/imgs/blogs/moba-3.png)

The five stages above are the production path. (1) **Split KV into blocks** of size $B$. (2) **Compute gating scores** via the mean-pooled-key dot product (Eq. 6 in the paper), apply the causal mask to kill future blocks, and take top-k. (3) **Route queries** — permute the query tokens so that all queries assigned to a given KV block are contiguous, the same permutation trick that makes MoE expert dispatch efficient. (4) **Run two attention groups**: the current-block group with `causal=True` and the historical-selected-block group with `causal=False`, each as a `flash_attention_varlen` call over the packed, variable-length segments. (5) **Combine** the partial outputs from the two groups with **online softmax**, the same numerically-stable running-max-and-sum reduction FlashAttention already uses internally, so that a query attending to both its current block and several history blocks gets a single correctly-normalized output.

The reason this is fast is that FlashAttention is at its best on dense, contiguous blocks of work, and the routing step hands it exactly that. There is no token-level sparsity inside a selected block — every selected block is computed fully and densely — so you keep FlashAttention's memory and throughput advantages and pay only for the blocks you kept. In pseudocode the efficient path looks like this:

```python
def moba_efficient(q, k, v, block_size, topk):
    """Sketch of the production MoBA kernel (one head).

    Splits the work into a causal 'current block' group and a
    non-causal 'history blocks' group, each a varlen FlashAttention
    call, then merges them with an online-softmax reduction.
    """
    S, d = q.shape
    blk_id = torch.arange(S, device=q.device) // block_size
    mean_k = block_mean(k, block_size)                        # (n_blocks, d)

    gate_scores = q @ mean_k.t()                              # (S, n_blocks)
    gate_scores = mask_future_blocks(gate_scores, blk_id)     # set illegal blocks to -inf
    sel_blocks = topk_with_current(gate_scores, blk_id, topk) # include current block always

    cur_q, cur_kv = pack_current_block(q, k, v, blk_id)       # one block each, causal
    out_cur, lse_cur = flash_attention_varlen(
        cur_q, cur_kv.k, cur_kv.v, causal=True)

    hist_q, hist_kv = route_history(q, k, v, sel_blocks, blk_id)  # MoE-style permute
    out_hist, lse_hist = flash_attention_varlen(
        hist_q, hist_kv.k, hist_kv.v, causal=False)

    out = online_softmax_merge(                               # combine the two groups
        (out_cur, lse_cur), (out_hist, lse_hist))
    return unpermute(out)                                     # restore original token order
```

Two details are worth flagging for anyone who would implement this. First, `lse` (the log-sum-exp / softmax normalizer) must be carried out of each FlashAttention call, because `online_softmax_merge` needs the per-query running max and sum from *both* groups to renormalize correctly — you cannot just average the two outputs. Second, the routing in `route_history` is the part that actually saves compute: it produces variable-length segments (different queries select different numbers of history blocks, bounded by $k-1$), which is why the call is `flash_attention_varlen` and not the fixed-shape variant.

### Where MoBA collapses into known patterns

A useful sanity check on any "general" mechanism is whether it can reproduce the specific things it claims to subsume. MoBA passes this cleanly. If you force the gate to always select the local (current) block and nothing else, you have sliding-window attention with window equal to the block. If you force it to always select the first block plus the most recent block, you have an attention sink. The general MoBA gate is what you get when you let those forced selections become learned top-k selections instead. This is not just a rhetorical flourish — it is the reason MoBA can be a strict generalization rather than a competing alternative, and it is why the model can recover the inductive biases of the fixed-pattern family on the rare occasions those biases happen to be right.

Here is the architectural comparison in one table, which doubles as a summary of how MoBA differs from the families it is reacting against.

| Property | Full attention | Sliding window | Linear attention | MoBA |
|---|---|---|---|---|
| Added parameters | 0 | 0 | varies | **0** |
| Compute vs length | $O(N^2)$ | $O(N\cdot w)$ | $O(N)$ | **sub-quadratic** |
| Where to attend | everywhere | fixed window | implicit state | **learned top-k blocks** |
| Softmax attention kept | yes | yes | no | **yes** |
| Drop-in for pretrained model | n/a | partial | no | **yes (after continual training)** |
| Switch sparse to dense | n/a | no | no | **yes, no loss spike** |

## Training recipe and hybrid strategies

MoBA's weight-compatibility with full attention is not a footnote — it is what makes the training recipes possible, and the recipes are what close the last bit of quality gap. There are three things to understand: the continual-pretraining schedule for the headline 1M model, the stage-wise hybrid, and the layer-wise hybrid for SFT.

![Continual pretraining for Llama-8B-1M-MoBA: context stretched 128K to 1M, then MoBA runs for 100B tokens before decode reverts to full attention](/imgs/blogs/moba-6.png)

The grid above is the production recipe for **Llama-8B-1M-MoBA**, the paper's main evaluation model. It is continual-pretrained from a Llama 8B base (Llama 3.1 8B class: 32 layers, 32 heads, hidden size 4096). The context length is extended progressively — 128K, then 256K, then 512K, then 1M — using position interpolation (S. Chen et al., 2023), applied at the start of the 256K stage to keep RoPE well-behaved as the context stretches. Only after the model is comfortable at 1M context is MoBA *activated*, for **100B tokens** of training at block size 4096, top-k 12 (the 95.31%-sparsity configuration). A layer-wise hybrid is adopted so the model retains some full-attention capability, and at inference time the model uses MoBA for prefill and switches to full attention during generation for the best decode quality.

### Why hybrids exist: the trailing-token problem

Pure MoBA — every layer, every token, top-k blocks, no full-attention anywhere — is *almost* as good as full attention, but the paper is honest that there is a measurable gap on one specific metric: the loss on *trailing tokens* (the last tokens of a long sequence). On a position-wise loss curve, pure MoBA's loss for the trailing tokens sits slightly above full attention's. The intuition is that the very last tokens of a long context are exactly the ones whose prediction depends most on having the right history available, and a hard top-k gate occasionally drops a block that mattered.

The fix is the **stage-wise hybrid**: train with MoBA for the first 90% of tokens, then switch to full attention for the final 10%. Because MoBA and full attention share weights, this switch produces no loss spike. The ablation shows the 90/10 hybrid closes the trailing-token gap and matches full attention on the position-wise loss, while still getting the efficiency benefit of MoBA across 90% of training. This is a genuinely clever exploitation of the weight-compatibility property — you spend the cheap-but-slightly-lossy mechanism on the bulk of training and the expensive-but-exact one only at the margin where it matters.

### Layer-wise hybrid for supervised fine-tuning

SFT introduces a second wrinkle. During SFT, the loss is typically masked over the *prompt* tokens — you only train on the model's responses, not on the instruction it was given. That masking produces sparse gradient signals, and MoBA's own sparsity can compound the problem: if the gate has already dropped blocks and the loss has already masked tokens, the gradient reaching the early context can become too sparse to learn from well. The paper's fix is the **layer-wise hybrid**: make the last few layers full attention while earlier layers stay MoBA. The ablation shows that progressively adding full-attention layers monotonically reduces SFT loss. The reason the *last* layers are the ones to make dense is that they are closest to the loss and carry the densest gradient signal, so giving them full attention recovers the gradient flow that pure MoBA had thinned out.

### Scaling beyond 1M

The recipe for pushing past 1M to 10M context is worth noting because it reveals the right knob. Rather than scale top-k or the number of blocks, the authors **fix top-k and fix the number of MoBA blocks, and increase the block size proportionally** with the context length. This keeps the sparsity ratio constant as the sequence grows — if you always select the same number of blocks out of the same number of blocks, the fraction of context attended is invariant — which is exactly what you want for a mechanism whose whole pitch is sub-quadratic scaling. Combined with expanded tensor parallelism (Shoeybi et al., 2019), this is how the efficiency experiments reach 10M tokens.

## Experiments

The experimental story has three pillars: downstream benchmark parity at 1M context, scaling-law parity across a five-model family, and the efficiency numbers that justify the whole exercise. The dossier is explicit that several engineering details — optimizer, exact learning rates, batch sizes, GPU type and hours, and the pretraining dataset composition — are **not stated** in the report, so we will not invent them; the comparison is clean precisely because, between MoBA and full attention, the *only* thing that changes is the attention module, with learning rate and batch size held constant.

### Downstream benchmarks at 1M context

![Llama-8B-1M MoBA versus full attention on Table 2: MoBA tracks full attention within a few hundredths despite running 95 percent sparse](/imgs/blogs/moba-4.png)

The matrix above is the headline result, and the full Table 2 is below. The comparison is Llama-8B-1M-MoBA against an otherwise-identical Llama-8B-1M-Full. The pattern is striking: across fourteen general benchmarks plus two long-context ones, the two models are within a couple of hundredths of each other everywhere, and on several tasks MoBA is *ahead*.

| Benchmark (setting) | MoBA | Full | Delta |
|---|---|---|---|
| AGIEval [0-shot] | 0.5144 | 0.5146 | -0.0002 (tie) |
| BBH [3-shot] | 0.6573 | 0.6589 | -0.0016 (tie) |
| CEval [5-shot] | **0.6273** | 0.6165 | **+0.0108 MoBA** |
| GSM8K [5-shot] | **0.7278** | 0.7142 | **+0.0136 MoBA** |
| HellaSWAG [0-shot] | 0.8262 | 0.8279 | -0.0017 (tie) |
| Loogle [0-shot] | **0.4209** | 0.4016 | **+0.0193 MoBA** |
| Competition Math [0-shot] | 0.4254 | 0.4324 | -0.0070 Full |
| MBPP [3-shot] | **0.5380** | 0.5320 | **+0.0060 MoBA** |
| MBPP Sanitized [0-shot] | **0.6926** | 0.6615 | **+0.0311 MoBA** |
| MMLU [0-shot] | 0.4903 | 0.4904 | -0.0001 (tie) |
| MMLU Pro [5-shot][CoT] | 0.4295 | 0.4328 | -0.0033 Full |
| HumanEval [0-shot][pass@1] | 0.6951 | 0.7012 | -0.0061 Full |
| SimpleQA [0-shot] | 0.0465 | 0.0492 | -0.0027 Full |
| TriviaQA [0-shot] | 0.5673 | 0.5667 | +0.0006 (tie) |
| LongBench @32K [0-shot] | **0.4828** | 0.4821 | +0.0007 MoBA |
| RULER @128K [0-shot] | 0.7818 | 0.7849 | -0.0031 Full |

The honest read is that this is parity, not dominance. MoBA wins six, full attention wins five, and the rest are ties within noise. The wins and losses are small in both directions — the largest MoBA win is +0.0311 on MBPP Sanitized, the largest full-attention win is -0.0070 on Competition Math. What makes parity remarkable is the *conditions*: on RULER @128K, the most long-context-sensitive benchmark in the table, MoBA runs at up to 95.31% sparsity and lands at 0.7818 versus full attention's 0.7849 — a 0.0031 gap while reading less than 5% of the context. That is the whole thesis in one number.

### Scaling-law parity

The five-model scaling-law family (Table 1) is the cleaner scientific result, because it controls for everything except attention. The models are trained at 8K sequence length with block size 512 and top-k 3, sized from 568M to 2.1B parameters at Chinchilla-optimal token counts:

| Params | Heads | Layers | Hidden | Train tokens | Block | TopK |
|---|---|---|---|---|---|---|
| 568M | 14 | 14 | 1792 | 10.8B | 512 | 3 |
| 822M | 16 | 16 | 2048 | 15.3B | 512 | 3 |
| 1.1B | 18 | 18 | 2304 | 20.6B | 512 | 3 |
| 1.5B | 20 | 20 | 2560 | 27.4B | 512 | 3 |
| 2.1B | 22 | 22 | 2816 | 36.9B | 512 | 3 |

Across all five, the validation-loss difference between MoBA and full attention stays within **1e-3**. The fitted scaling laws are nearly indistinguishable — LM loss at 8K fits $2.625 \cdot C^{-0.063}$ for MoBA versus $2.622 \cdot C^{-0.063}$ for full attention, the *same* exponent. The one place a gap shows is the trailing-loss metric (loss on the last 2K tokens of a 32K sequence), where MoBA fits $1.546 \cdot C^{-0.108}$ versus full attention's $1.464 \cdot C^{-0.097}$ — MoBA marginally higher, but with an exponent that means the gap *shrinks* as compute scales. In the 12K-14K range reported in the appendix, the trailing fits are near-identical ($1.670 \cdot C^{-0.089}$ vs $1.645 \cdot C^{-0.088}$). The takeaway: whatever MoBA gives up on the hardest long-context positions, it gives up less the bigger you go, which is the right direction for a method targeting frontier-scale models.

| Metric | MoBA fit | Full fit | Read |
|---|---|---|---|
| LM loss (8K) | $2.625\,C^{-0.063}$ | $2.622\,C^{-0.063}$ | identical exponent, gap < 1e-3 |
| Trailing loss (32K, last 2K) | $1.546\,C^{-0.108}$ | $1.464\,C^{-0.097}$ | MoBA higher, gap shrinks with scale |
| Trailing loss (12K-14K) | $1.670\,C^{-0.089}$ | $1.645\,C^{-0.088}$ | near-identical |

### Efficiency

This is where MoBA earns its place in production. Measuring attention layers only (the FFN is identical between the two, so it would only dilute the comparison), the paper reports:

- **6.5x speedup** on 1M-token prefill versus FlashAttention.
- **16x reduction** in attention computation time at 10M tokens, holding sparsity fixed at 95.31% with 64 MoBA blocks, top-k 3, and proportionally larger block size.
- **40x** faster than the naive implementation at 32K (from the README), for the `moba_efficient` kernel.

The shape of these numbers matters more than the magnitudes: the speedup *grows* with sequence length (6.5x at 1M, 16x at 10M), which is the signature of genuinely sub-quadratic scaling. A method that gave you a constant 2x everywhere would be a nice constant-factor win; a method whose advantage compounds with length is the one you build a 10M-context product on.

### What is load-bearing, and what might not transfer

The cleanest load-bearing claim is the scaling-law parity, because it is the most controlled: same data, same optimizer, same LR and batch size, only the attention module changes, and the loss gap stays within 1e-3 across five model sizes. That is hard to argue with. The downstream benchmark parity is also strong but slightly softer — these are single-run numbers on a single 8B model with the layer-wise hybrid and full-attention decode, so they bundle MoBA-the-mechanism with MoBA-the-recipe. The thing most likely *not* to transfer cleanly is the specific sparsity sweet spot. The 95.31% figure is tied to block size 4096 and top-k 12 at 1M context; the comparable-quality sparsity at the 8K scaling regime is quoted around 75%, and the 8K experiments themselves run at 81.25% sparsity (top-k 3, block 512). So the right top-k and block size are functions of the context length and the model, not universal constants — anyone adopting MoBA should expect to tune them, and the ablation on block granularity (below) says the tuning is not free.

## Critique

**What is strong.** The conceptual move — MoE routing applied to attention blocks — is the kind of idea that, once stated, makes you wonder why it was not the default. It is parameter-free, which removes an entire class of "is the win just from extra capacity?" objections. The causal-correctness treatment (force the current block, mask the future, causal-mask within the current block) is careful and complete. The weight-compatibility is genuinely exploited, not just claimed: the stage-wise and layer-wise hybrids are direct payoffs of it, and the no-loss-spike switching is the property that makes those hybrids practical. And the result that sells the whole thing — within-1e-3 loss across a five-model scaling family — is exactly the controlled comparison you want, not a cherry-picked benchmark. Most importantly, it is deployed in Kimi, which means the efficiency claims survived a real serving stack.

**What is weak or unfalsifiable.** The benchmark wins are within noise in both directions, so "MoBA matches full attention" is well-supported but "MoBA is better" is not — the six MoBA wins on Table 2 are most honestly read as run-to-run variation rather than a real edge. The paper does not report variance or multiple seeds, so we cannot put error bars on the per-benchmark deltas, which means a +0.0108 on CEval and a -0.0061 on HumanEval are not cleanly distinguishable from zero. The trailing-token gap is real and the hybrid is the patch, but that patch slightly undercuts the "pure drop-in" story: the cleanest version of MoBA (every layer, all tokens) is *not* the version that matches full attention; the matching version needs the recipe. And the "less structure" framing, while appealing, is partly rhetorical — top-k with a forced current block and masked future blocks *is* structure, just less of it.

**What ablation is missing.** The block-granularity ablation (below) is good but partial. The paper shows finer-grained segmentation helps at fixed sparsity, sweeping (top-k / blocks) = 2/8, 4/16, 8/32, 16/64, 32/128 on a 1.5B model at 32K, with LM loss improving from ~2.245 to ~2.230 as granularity rises. But it does not isolate *how much* of the headline result depends on the hybrid recipes versus pure MoBA at scale — the 8B model bundles MoBA with the layer-wise hybrid and full-attention decode, so we never see pure-MoBA-at-8B numbers on the downstream table. We also do not see a sparsity-vs-quality frontier at 1M context: is 95.31% the knee of the curve, or could you push to 97% with a smaller top-k and lose little? The paper picks one operating point and validates it rather than charting the trade-off.

| Granularity (topk / blocks) | Sparsity | LM loss |
|---|---|---|
| 2 / 8 (coarsest) | fixed | ~2.245 (worst) |
| 4 / 16 | fixed | between |
| 8 / 32 | fixed | between |
| 16 / 64 | fixed | between |
| 32 / 128 (finest) | fixed | ~2.230 (best) |

**What would change my mind.** Two things. First, if a careful multi-seed re-run showed the trailing-token gap *widening* rather than shrinking at the 30B+ model scale — that would invalidate the central "the gap diminishes with scale" claim and turn MoBA into a small-model-only convenience. Second, if pure MoBA (no hybrid, no full-attention decode) at 8B turned out to be materially behind full attention on RULER-style retrieval, it would mean the mechanism's parity is carried by the recipe rather than the gate, which would change how I would deploy it.

## What I'd build with this

A few concrete extensions, in rough order of how much I would trust them to work.

1. **A dynamic top-k schedule keyed to query type.** Right now top-k is a fixed hyperparameter. But a retrieval query ("find the clause that mentions indemnification") and a local-coherence query ("finish this sentence") want very different block budgets. A cheap learned predictor that sets per-query top-k — small for local tokens, large for tokens whose gate scores are flat (a sign relevant context is spread out) — could push average sparsity higher without hurting the hard cases. The gate scores are already computed, so the signal is free.

2. **A KV-cache eviction policy derived from the gate.** MoBA already knows, per query, which blocks were never selected over a long generation. Blocks whose mean-pooled key never wins the top-k for many consecutive queries are strong candidates for offload or eviction from fast memory. Pairing MoBA's block gate with a [Mooncake](/blog/paper-reading/large-language-model/mooncake)-style disaggregated KV cache could turn the gate's selection statistics into an eviction signal, attacking the memory wall as well as the compute wall.

3. **MoBA on top of a linear-attention backbone for the unselected mass.** Instead of dropping unselected blocks entirely, route them through a cheap linear-attention summary so the model gets a coarse signal from the whole context plus exact softmax attention on the top-k blocks. This is a hybrid of the two families MoBA reacts against, and it might recover the last bit of trailing-token loss without the full-attention decode crutch. Kimi Linear's mechanism would be the natural partner here.

4. **Block-gate distillation for fast adoption.** The biggest practical wart is that MoBA needs continual training to adopt. A distillation objective that teaches a fresh MoBA gate to mimic the attention *mass distribution* of a frozen full-attention teacher — match where the dense model puts its softmax weight, block by block — might cut the 100B-token continual-training bill dramatically by giving the gate a supervised target instead of making it discover the right blocks from LM loss alone.

5. **Agentic long-context serving.** The 10M-context regime is where this matters most, and agent traces (long tool-call histories, accumulated scratchpads) are exactly the kind of context that is mostly irrelevant per step. A serving stack that runs MoBA prefill on the full trace and full-attention decode on the active window is the obvious shape; the work in [Kimi K2](/blog/paper-reading/large-language-model/kimi-k2) and [Kimi-Researcher](/blog/paper-reading/ai-agent/kimi-researcher) is where I would expect to see this land first.

## When to reach for MoBA (and when not to)

Reach for MoBA when you are training or continual-training your *own* model and the target is long context — say 128K and up, with 1M as the regime where the payoff is large. If you control the training run, MoBA is close to a free lunch: it adds no parameters, it slots into FlashAttention, the scaling-law evidence says it costs essentially nothing in loss, and the prefill speedup (6.5x at 1M, 16x at 10M) is real and grows with length. The hybrid recipes are simple enough to adopt — 90/10 stage-wise for pretraining, last-few-layers-dense for SFT — and they buy back the small trailing-token gap. If your workload is retrieval-heavy over long documents and you have been bitten by sliding-window's hard-coded blind spots, MoBA's learned gate is strictly the better structure.

Do *not* reach for MoBA expecting a free inference-time swap on a model you did not train. It is not weight-reusable without continual training — you cannot take an off-the-shelf full-attention checkpoint, flip on MoBA, and expect parity; the gate has to be trained into the model over a non-trivial token budget (100B for the 8B model). Do not reach for it if your contexts are short (a few thousand tokens), where the quadratic term is not your bottleneck and the block machinery is pure overhead. And be cautious if absolute peak quality on the very last tokens of a long generation is the thing you are graded on, because that is precisely the metric where pure MoBA is weakest and where the paper itself reverts to full attention during decode. For everything in between — frontier-scale, long-context, training-controlled — MoBA is one of the most convincing "keep softmax attention, just compute less of it" results to date, and the fact that it is already serving Kimi traffic is the part of the paper I trust most.

## References

- **MoBA: Mixture of Block Attention for Long-Context LLMs** — arXiv abstract: [arxiv.org/abs/2502.13189](https://arxiv.org/abs/2502.13189)
- **Code and tech report:** [github.com/MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA)
- Related reading on the same lab's long-context and efficiency work:
  - [Kimi Linear: An Expressive, Efficient Attention Architecture](/blog/paper-reading/large-language-model/kimi-linear) — the linear-attention answer to the same problem MoBA attacks with block sparsity.
  - [Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving](/blog/paper-reading/large-language-model/mooncake) — the serving-side counterpart to MoBA's compute-side savings.
  - [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2) — the agentic-scale model family where long-context serving matters most.
  - [Kimi-Researcher: End-to-End RL for Autonomous Research](/blog/paper-reading/ai-agent/kimi-researcher) — long tool-call traces are exactly the mostly-irrelevant context MoBA is built to skip.

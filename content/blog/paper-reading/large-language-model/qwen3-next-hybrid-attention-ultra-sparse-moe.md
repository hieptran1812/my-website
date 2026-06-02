---
title: "Qwen3-Next: Hybrid Attention and an 80B Model That Thinks With 3B"
date: "2026-05-17"
publishDate: "2026-05-17"
description: "A close read of Qwen3-Next-80B-A3B: how a 3:1 Gated DeltaNet / Gated Attention hybrid, a 512-expert ultra-sparse MoE, and multi-token prediction deliver flagship quality at a fraction of the compute."
tags: ["qwen3-next", "large-language-model", "mixture-of-experts", "linear-attention", "gated-deltanet", "hybrid-attention", "long-context", "multi-token-prediction", "paper-reading"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: false
readTime: 30
---

The Qwen3 flagship, Qwen3-235B-A22B, is a very good model that is also a very expensive object to keep alive. Two hundred and thirty-five billion parameters have to sit in GPU memory whether or not any given token routes through them, and at long context the attention cost climbs quadratically until the KV cache, not the weights, is what bounds your batch size. The model is excellent. The serving bill is not, and at long context the bill is the part that decides whether a deployment is viable at all — not whether the model is good enough, but whether you can afford to let it think over a million tokens at the throughput your users will tolerate.

Qwen3-Next-80B-A3B is the Qwen team's answer to a sharper question: *how much of that flagship quality can you keep if you are allowed to redesign the architecture purely for efficiency?* The answer the report gives is uncomfortable for anyone invested in dense transformers. Qwen3-Next activates **3 billion** parameters per token — out of 80 billion total — and lands within a point or two of the 235B flagship on most benchmarks, while training for roughly **10% of the cost** of the much smaller dense Qwen3-32B and serving long context at **10× the throughput**.

![Qwen3-Next: two efficiency levers, one model](/imgs/blogs/qwen3-next-1.png)

The diagram above is the mental model: two independent efficiency levers, pulled at the same time. The first is **hybrid attention** — replace most of the quadratic softmax-attention layers with linear-attention layers, keeping just enough full attention for exact recall. The second is an **ultra-sparse mixture of experts** — 512 experts, 10 active per token, an activation ratio so low it would have looked like a typo two years ago. Neither lever is novel on its own. The contribution is showing they compose cleanly at scale, and that the resulting model does not quietly fall apart on the tasks the efficiency tricks are supposed to threaten.

This post reads the Qwen3-Next release the way you would read a technical report: the architecture first, then why each choice is load-bearing, then the benchmark numbers, then the senior-engineer critique of what the headline figures are quietly assuming. If you have not read our companion piece on the [Qwen3 Technical Report](/blog/paper-reading/large-language-model/qwen3-technical-report), it is worth skimming first — Qwen3-Next inherits that model's dual-mode behavior and post-training pipeline, and this post focuses on what is *different*.

> [!tldr] TL;DR
> - **Hybrid attention.** 48 layers in a fixed 3:1 pattern — three Gated DeltaNet (linear, $O(n)$) layers for every one Gated Attention (full softmax, $O(n^2)$) layer. Linear attention carries the cost; full attention carries exact recall.
> - **Ultra-sparse MoE.** 512 experts, 10 routed + 1 shared active per token. 80B total parameters, ~3B activated — roughly a 3.75% activation ratio, versus ~9% for the Qwen3-235B flagship.
> - **Multi-token prediction (MTP).** A native MTP objective improves pre-training and doubles as a self-speculative decoding draft model at inference, with no separate draft network.
> - **The numbers.** ~10% of Qwen3-32B's training cost; ~10× its throughput at 32K+ context; native 262K context, extensible to ~1M with YaRN; 80.3% RULER accuracy at 1M tokens.
> - **Quality held.** Within ~1–2 points of Qwen3-235B-A22B on most knowledge, reasoning, and coding benchmarks despite activating roughly one-seventh the parameters.
> - **Where it's thin.** The 3:1 ratio is asserted, not swept in public; linear-attention recall failures are real and the benchmark suite under-probes them; the "10× throughput" figure is a long-context number that does not describe short-prompt chat.

## Context: what came before

For four years the recipe for a better language model was "the same transformer, but bigger." The transformer's self-attention is the part that scales badly: every token attends to every other token, so compute and memory grow as $O(n^2)$ in sequence length $n$. For a 2K-token prompt nobody cares. For a 200K-token codebase or a 1M-token document, the quadratic term is the whole bill, and it shows up twice — once as FLOPs during prefill, and once as a KV cache that grows linearly with every token you have already processed and must be kept resident for every future decode step. Our [KV cache deep-dive](/blog/machine-learning/large-language-model/kv-cache) walks through why that cache, not the parameter count, is what usually ends a long-context serving plan.

Two largely separate research threads spent those years attacking the two halves of the cost.

The **linear-attention** thread asked whether the $O(n^2)$ could be made $O(n)$. The idea: instead of materializing an $n \times n$ attention matrix, maintain a fixed-size recurrent *state* that summarizes everything seen so far, and update it token by token. Linear attention, state-space models (S4, Mamba), and the DeltaNet family all live here. They are genuinely linear and genuinely cheap — and they genuinely lose something. A fixed-size state cannot perfectly remember an arbitrary token from 100K positions ago the way full attention can; it compresses, and compression loses exactly the needle-in-a-haystack recall that retrieval and in-context learning lean on. Our survey post [Speed Always Wins](/blog/paper-reading/large-language-model/speed-always-wins-a-survey-on-efficient-architectures-for-large-language-models) catalogues this whole design space and the recall tax each point on it pays.

The **sparsity** thread asked whether every parameter needs to fire for every token. Mixture-of-experts says no: route each token to a small subset of expert FFNs, so total capacity and per-token compute become separate knobs. Qwen3 already shipped MoE at a 128-expert, 8-active configuration. The open question was how far the activation ratio could be pushed before routing instability or expert under-training broke the model — see [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) for why that ratio is the central tuning problem.

There is a reason these two threads stayed mostly separate. Each is, individually, a way to *lose* quality in exchange for speed, and stacking two quality-risking changes invites them to interact badly. A linear-attention layer that has compressed away a fact cannot route a token to an expert that would have recovered it; a mis-routed MoE token cannot be rescued by an attention layer that never saw the right context. When you change one axis you can attribute a regression to it. When you change two, a regression could be either, or the cross-term, and debugging a 80B training run by bisection is not a thing anyone wants to do. The conservative move — the move most of the field made — is to ship one architectural bet per model.

The gap Qwen3-Next claims to fill is the *combination*. Most efficient-architecture papers change one thing and hold the rest fixed, because changing two things at once makes attribution hard and instability likely. Qwen3-Next changes the attention mechanism *and* the MoE sparsity *and* adds a multi-token-prediction objective, at 80B scale, and reports that the result is stable and competitive. Whether you find that convincing depends on how much you trust a release that does not publicly ablate its own central ratios — a tension we return to in the critique. The Jet-Nemotron line of work (see our note on [Jet-Nemotron](/blog/paper-reading/large-language-model/jet-nemotron)) is the closest public neighbour: it also argues that a carefully chosen *minority* of full-attention layers, dropped into an otherwise-linear stack, recovers most of the quality. Qwen3-Next is the same thesis carried to production scale and bolted onto an aggressive MoE.

## Contributions

Tightened from the release materials and model card:

1. **A 3:1 hybrid-attention stack.** Forty-eight layers arranged as twelve repeats of a four-layer block: three Gated DeltaNet layers, then one Gated Attention layer, each followed by an MoE layer. Linear attention is the default; full attention is the exception.
2. **An ultra-sparse MoE.** 512 routed experts plus 1 shared expert, with 10 routed experts active per token. 80B total parameters, ~3B activated — an activation ratio under 4%.
3. **Native multi-token prediction.** An MTP objective during pre-training, reused at inference as a built-in self-speculative draft model, removing the need for a separate draft network.
4. **Training-stability engineering.** Zero-centered, weight-decayed layernorm and related fixes that the report credits for keeping a multi-change architecture trainable at scale.
5. **Demonstrated efficiency.** ~10% of Qwen3-32B's training cost; ~10× its long-context throughput; native 262K context extensible to ~1M; quality within ~1–2 points of the 235B flagship.

## Method

### Hybrid attention

Start with the layer stack, because the 3:1 pattern is the single most important number in the model.

![The repeating hybrid block](/imgs/blogs/qwen3-next-2.png)

Qwen3-Next has 48 layers. They are not homogeneous. They are twelve copies of a four-layer block: **Gated DeltaNet → Gated DeltaNet → Gated DeltaNet → Gated Attention**, with an MoE layer after each. Three quarters of the token-mixing layers — 36 of 48 — are linear-attention DeltaNet layers. Only twelve are full softmax attention.

The reasoning is a division of labour, and the table below is the argument for keeping both rather than committing to one.

![Gated DeltaNet vs Gated Attention](/imgs/blogs/qwen3-next-3.png)

**Gated DeltaNet** is the linear-attention workhorse. "DeltaNet" refers to the delta-rule update: rather than storing every key-value pair, the layer maintains a fixed-size state matrix $S$ and updates it with a correction proportional to the difference (the *delta*) between what the new value is and what the current state would have predicted. Schematically, for incoming key $k_t$ and value $v_t$:

$$
S_t = S_{t-1} + \beta_t \, \big(v_t - S_{t-1} k_t\big)\, k_t^\top
$$

where $\beta_t$ is a learned, input-dependent gate that controls how much the state is rewritten at step $t$. The "gated" part is exactly that $\beta_t$ — it lets the layer decide, per token, whether this is information worth overwriting old state for. The output for token $t$ is read out as $S_t q_t$ for query $q_t$. Crucially, $S$ has a *fixed* size (heads × head-dim × head-dim) regardless of sequence length. There is no $n \times n$ matrix and no per-token KV cache. Cost is $O(n)$ in sequence length, and the memory footprint is constant.

In Qwen3-Next the Gated DeltaNet layers use 32 value heads and 16 query/key heads at head dimension 128.

**Gated Attention** is standard scaled-dot-product softmax attention — the $O(n^2)$, $n \times n$ kind — with an output gate and partial rotary embeddings. It uses 16 query heads and 2 key/value heads (a wide grouped-query ratio to keep the cache small) at head dimension 256, with rotary position embedding applied to a 64-dimensional slice. The "gated" modification — an input-dependent gate on the attention output — is the same family of trick analyzed in [Gated Attention for LLMs](/blog/paper-reading/large-language-model/gated-attention-for-large-language-models-non-linearity-sparsity-and-attention-sink-free): it improves training stability and, the gated-attention literature argues, removes the pathological "attention sink" behavior where a disproportionate share of attention mass collapses onto the first token.

It is worth dwelling on what the delta rule is actually doing, because "linear attention" makes it sound like a lossy approximation of softmax attention when it is better understood as a different data structure entirely. Softmax attention is an *append-only log*: every key-value pair from every past token is kept verbatim, and a query searches the whole log. Gated DeltaNet is an *associative memory of fixed size*: the state $S$ is a key→value map, and the delta update is a write that overwrites whatever the old map predicted for key $k_t$ with the new value $v_t$, scaled by the gate $\beta_t$. When $\beta_t \approx 1$ the layer fully rewrites that slot; when $\beta_t \approx 0$ it leaves the old association intact. The model *learns a write policy*. That is a strictly more powerful object than the plain exponential-decay state of vanilla linear attention or an S4-style SSM, which can only fade old information uniformly — it cannot choose to surgically overwrite one association while preserving another. The cost of the power is the fixed size: the map has a bounded number of effective slots, and when the context contains more distinct associations than slots, the write policy has to start evicting.

Why keep any quadratic layers at all, if linear attention is so much cheaper? Because the fixed-size state of a linear layer is a *lossy summary*. It is excellent at carrying forward the gist of a long context and poor at the surgical "what exact token appeared at position 91,403" lookup. Full attention, with its complete $n \times n$ score matrix, can do that lookup precisely. The 3:1 ratio is a bet that you need that precise lookup only occasionally — that three layers of cheap state-mixing followed by one layer of exact attention recovers most of what a fully-quadratic stack would do, at a quarter of the attention cost. The twelve full-attention layers are the model's *random-access memory*; the thirty-six DeltaNet layers are its *streaming summary*.

The intuition for *why three and not zero* is worth making concrete. Imagine reading a long contract. Most of the time you are integrating — building a running sense of obligations, parties, dates — and a compressed mental summary is exactly right; re-reading every prior clause for every new word would be absurd. But occasionally you hit "as defined in Section 4.2(b)" and you need to jump to an exact location and read it verbatim. A pure-linear model is a reader with only the running summary and no ability to jump back; a pure-quadratic model re-reads the entire contract for every word. The 3:1 hybrid is a reader who integrates cheaply for three steps and is allowed one exact jump-back on the fourth. The bet is that the ratio of "integrate" to "look up exactly" in real language is heavily skewed toward integration — and for most text it plainly is.

Here is the shape of one hybrid block, written as PyTorch-flavored module code so the data flow is explicit:

```python
import torch.nn as nn

class HybridBlock(nn.Module):
    """One four-layer Qwen3-Next block: 3x DeltaNet + 1x GatedAttention,
    each token-mixer followed by an ultra-sparse MoE FFN."""
    def __init__(self, dim=2048):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(4):
            mixer = (GatedAttention(dim, q_heads=16, kv_heads=2, head_dim=256)
                     if i == 3 else
                     GatedDeltaNet(dim, v_heads=32, qk_heads=16, head_dim=128))
            self.layers.append(nn.ModuleList([
                nn.RMSNorm(dim), mixer,
                nn.RMSNorm(dim), SparseMoE(dim, n_experts=512, k=10, shared=1),
            ]))

    def forward(self, x, state):
        # `state` is the fixed-size recurrent state for the DeltaNet layers;
        # the GatedAttention layer ignores it and uses a KV cache instead.
        for norm1, mixer, norm2, moe in self.layers:
            x = x + mixer(norm1(x), state)   # pre-norm residual, as in Qwen3
            x = x + moe(norm2(x))
        return x

N_BLOCKS = 12   # full model: 12 blocks, 48 layers, 36 DeltaNet + 12 attention
```

The asymmetry to internalize: the DeltaNet layers carry a small fixed `state` from token to token; the attention layers carry a KV cache that grows with sequence length. At 1M tokens of context, the report puts the total KV footprint at roughly **25 GB** — about 4× smaller than a naive all-quadratic transformer of similar width would need — precisely because only a quarter of the layers contribute a growing cache at all.

Put the memory arithmetic side by side. In a conventional 48-layer transformer, *every* layer caches a key and a value vector for every token; KV memory is `layers × 2 × tokens × kv_heads × head_dim × dtype_bytes`, and it grows linearly with `tokens` without bound. In Qwen3-Next only the 12 attention layers cache anything at all — and those layers use a deliberately narrow 2-KV-head grouped-query configuration, so each cached token is cheap. The 36 DeltaNet layers contribute a *constant*: their state is `heads × head_dim × head_dim`, the same handful of megabytes whether the context is 4K or 1M tokens. So the model's memory curve is a steep dense line replaced by a shallow line (12 layers' worth of KV) plus a flat offset (36 layers' worth of fixed state). At short context the flat offset dominates and Qwen3-Next is not especially lean; at 1M tokens the shallow slope is the whole story, and that is the regime where the ~25 GB figure and the 4× saving live. This is the single most important fact for capacity planning: Qwen3-Next's memory advantage is *not* uniform — it is small at short context and enormous at long context, and your deployment math has to be done at the context length you actually serve.

### The ultra-sparse MoE

The feed-forward half of every layer is a mixture of experts, and Qwen3-Next pushes the sparsity well past where the Qwen3 flagship sat.

![MoE sparsity: Qwen3 vs Qwen3-Next](/imgs/blogs/qwen3-next-4.png)

The table makes the move concrete. Qwen3-235B-A22B used 128 experts with 8 active. Qwen3-Next uses **512 experts with 10 routed-active plus 1 shared**. The expert count quadrupled; the active count barely moved. The result is an activation ratio — activated parameters over total parameters — of roughly **3.75%**, against the flagship's ~9%.

Two design details matter here.

The **shared expert** is one expert that every token passes through, unconditionally, in addition to the 10 it is routed to. The intuition: some computation is useful for *every* token regardless of its content — basic syntactic processing, common-token handling — and forcing the router to rediscover that and allocate a routed slot to it for every token is wasteful. Pinning one always-on expert lets the routed experts specialize harder, because they no longer have to each carry a copy of the universal baseline. This is the same shared-expert idea DeepSeek-V3 popularized; note that the *Qwen3* flagship did not use it, and Qwen3-Next adding it back is a deliberate reversal.

The **512-expert count** is what makes the capacity-vs-compute decoupling extreme. Total parameters scale with the number of experts; activated parameters scale with the number active. Quadrupling experts while holding the active count near-constant means you can pack 80B of total capacity behind a per-token compute cost equivalent to a ~3B dense model. The catch — and it is the catch with all high-sparsity MoE — is that each of the 512 experts now sees only about 10/512 ≈ 2% of the token stream during training. An expert that rarely fires is an expert that rarely gets a gradient, and under-trained experts are dead capacity. This is why the report leans on **global-batch load balancing** (inherited from Qwen3): the balancing loss is computed across the entire global batch, not per-device, so the router is pushed toward genuinely spreading tokens across all 512 experts rather than collapsing onto a popular few. The deeper you go on sparsity, the more load balancing stops being a tuning detail and becomes the thing that decides whether the model trains at all — see [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference).

There is a second, subtler reason ultra-sparse MoE pairs well with hybrid attention specifically. Routing is a *content* decision — the router reads the token's hidden state and picks experts. The quality of that decision depends on how good the hidden state is, which depends on the token mixers feeding it. A linear-attention layer produces a hidden state that is a smooth, integrated summary; a full-attention layer produces one sharpened by exact lookups. Interleaving them means the router sometimes sees a "gist" representation and sometimes a "detail" representation, and the report's expert specialization presumably reflects that. The flip side is a failure mode worth naming: if a fact was compressed away by the DeltaNet layers before the routing decision, no expert can be selected to recover it, because the signal that would have triggered the right route is already gone. Sparsity cannot un-lose information that the token mixer dropped. This is why the recall question (below) is not separable from the MoE question — they share the same upstream representation.

A small PyTorch-shaped sketch of the routing step, to make the sparsity concrete:

```python
import torch, torch.nn.functional as F

def ultra_sparse_moe(x, router_w, experts, shared_expert, k=10):
    """x: (tokens, dim). 512 routed experts, 10 active, 1 shared."""
    logits = x @ router_w                      # (tokens, 512)
    topk_val, topk_idx = logits.topk(k, dim=-1)
    gates = F.softmax(topk_val, dim=-1)         # weights over the 10 chosen

    out = shared_expert(x)                      # always-on path, every token
    for slot in range(k):
        idx = topk_idx[:, slot]                 # which expert, per token
        g = gates[:, slot].unsqueeze(-1)
        for e in idx.unique():                  # grouped dispatch in practice
            mask = idx == e
            out[mask] += g[mask] * experts[e](x[mask])
    return out

PER_TOKEN = "~3B of 80B"   # 10 of 512 routed + 1 shared expert active
```

### Multi-token prediction

Standard language-model training predicts one token at a time: given tokens $1 \dots t$, predict token $t{+}1$. Multi-token prediction (MTP) trains the model to predict *several* future tokens — $t{+}1, t{+}2, t{+}3$ — from the same hidden state, using small additional prediction heads.

![Multi-token prediction and self-speculative decoding](/imgs/blogs/qwen3-next-5.png)

MTP pays off twice.

During **pre-training**, asking the model to predict $t{+}2$ and $t{+}3$ as well as $t{+}1$ is a denser, harder learning signal. The model cannot get $t{+}3$ right by leaning on the most recent token alone; it has to build a representation that anticipates structure several tokens out. The report credits the MTP objective with improving pre-training quality, consistent with what DeepSeek-V3 reported for the same technique.

During **inference**, the MTP heads become a free draft model for *self-speculative decoding*. Ordinary autoregressive decoding emits one token per forward pass — and each pass is memory-bandwidth-bound, dominated by streaming the weights, so the GPU's compute sits idle. Speculative decoding (covered in depth in [speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding)) hides that latency: a cheap draft model proposes several tokens, and the main model *verifies* all of them in a single parallel forward pass, accepting the longest prefix it agrees with. Normally the draft model is a separate, smaller network you have to train and serve. With MTP, the draft heads are *already inside the model* — the same weights, trained jointly. You get the speculative speedup without a second model in memory. That is what "self-speculative" means, and it is why MTP shows up in the throughput numbers, not just the quality ones.

The economics of speculative decoding are worth stating precisely, because they explain why MTP is more than a nice-to-have. A decode step's wall-clock cost is dominated by reading the model's parameters from HBM into the compute units; the actual matrix multiplies for a single token finish long before the weights have finished streaming. Verifying *four* candidate tokens in one pass reads those weights exactly once — the same memory traffic as decoding one token — so if the draft's first $a$ tokens are accepted, you have produced $a$ tokens for the price of one weight-streaming pass. The speedup is therefore the *mean accepted length*. If the MTP heads draft well enough that, on average, two-and-a-bit tokens are accepted per pass, you have roughly halved decode latency. If acceptance collapses to near-one — high-entropy text where the draft is usually wrong — you have gained nothing and spent a little extra compute on the rejected candidates. There is no quality risk either way: the main model verifies every token, so a bad draft costs latency, never correctness. That asymmetry — unbounded upside, bounded-and-small downside — is what makes self-speculative decoding a free option, and having the draft heads built in removes the one real cost (training and serving a separate draft network) that made the option non-free before.

### Training stability

A model that changes the attention mechanism, the MoE sparsity, and the training objective all at once is a model with three new ways to diverge. The report names a specific fix it considers load-bearing: **zero-centered, weight-decayed layernorm**. The pathology it targets is the slow drift of layernorm's learnable gain parameter during a long run — the gain creeps, activation magnitudes creep with it, and deep into training the loss develops an upward wobble or an outright spike. Re-centering the layernorm gain at zero and applying weight decay to it keeps the parameter from wandering. This is the same *category* of fix as the QK-Norm change in the Qwen3 flagship: it buys nothing on a benchmark and everything on whether a months-long run survives. With three architectural changes stacked, the stability budget is tight, and the report is explicit that this engineering is part of why the combination is trainable at all.

It helps to see why this model is unusually exposed to instability. Each of the three changes nudges the activation statistics. The DeltaNet layers introduce a recurrent state whose magnitude can compound across positions if the gates are not well-behaved. The ultra-sparse MoE means each expert is updated by a noisy, intermittent gradient — an expert that fires for 2% of tokens sees 2% of the smoothing that a dense FFN's parameters get — so expert weights are individually noisier and more prone to producing outlier activations. The MTP heads add auxiliary loss terms whose gradients flow back into the shared trunk. None of these is catastrophic alone; together they widen the distribution of activation magnitudes the optimizer has to keep bounded. Layernorm is the component that nominally bounds them, which is exactly why a *drifting* layernorm gain is so dangerous here and why the report singles the fix out. The general lesson for anyone stacking architectural bets: your stability budget is not additive, it is closer to multiplicative, and the normalization layers are where you spend it. The reason most labs ship one architectural change per model is not timidity — it is that the stability engineering for two simultaneous changes is genuinely harder than twice the engineering for one.

## The recall question

Every claim about Qwen3-Next eventually routes back to one worry, so it is worth isolating it. A hybrid model with 75% linear layers has a *structural* weakness — fixed-size state cannot losslessly hold an arbitrary amount of detail — and the entire design is a bet that the 25% full-attention layers patch it well enough. The honest way to read the model is to ask: where would that bet fail, and does the public evidence rule those cases out?

Three failure shapes are worth distinguishing.

**Single-needle retrieval.** "Find the one sentence in this 500K-token document that states the API key." This is what RULER mostly measures, and Qwen3-Next handles it well — 93.5% at 256K, 80.3% at 1M. A single distinctive needle survives compression because the write policy can give it a slot and the periodic full-attention layers can locate it. This case is, in practice, solved.

**Multi-needle retrieval under interference.** "There are forty similarly-worded configuration values scattered through this 800K-token log; report the one for `region=eu-west-2`." Now the fixed-size state has forty near-identical associations competing for slots, and the write policy must keep all forty distinct. This is the case a fixed-size memory is *designed* to struggle with, and it is exactly the case an aggregate RULER score blurs — RULER includes multi-key variants, but a single headline number averages the easy and hard sub-tasks together. The public evidence neither confirms nor rules out a weakness here. It is the first thing I would test.

**In-context learning with many exemplars.** A 200-shot prompt asks the model to hold 200 input-output pairs and generalize. Each exemplar is an association; 200 of them stress the state's capacity directly. Whether Qwen3-Next matches a full-attention model on many-shot ICL is, again, not something the released benchmarks isolate.

The reason this matters is not pedantic. Long context exists *for* these workloads — retrieval-augmented generation over large corpora, whole-repository code reasoning, many-shot prompting. If Qwen3-Next is excellent at long-context *integration* but merely good at long-context *interference-heavy recall*, then "flagship quality at a fraction of the cost" is true for summarization and false for the harder retrieval tasks, and you need to know which side of that line your traffic sits on. The 3:1 ratio is the knob that trades these off, and the absence of a public sweep over it is precisely why the question stays open.

## Experiments

The headline efficiency comparison is against the dense Qwen3-32B; the headline *quality* comparison is against the much larger Qwen3-235B-A22B. Numbers below are as reported in the Qwen3-Next release and model card — the authors' framing, not an independent reproduction.

![Efficiency: Qwen3-32B vs Qwen3-Next-80B-A3B](/imgs/blogs/qwen3-next-6.png)

**Quality, vs Qwen3-235B-A22B (thinking mode where applicable):**

| Benchmark | Qwen3-32B | Qwen3-235B-A22B | Qwen3-Next-80B-A3B |
|---|---|---|---|
| MMLU-Pro | 71.9 | 83.0 | 80.6 |
| MMLU-Redux | 85.7 | 93.1 | 90.9 |
| GPQA | 54.6 | 77.5 | 72.9 |
| AIME25 | 20.2 | 70.3 | 69.5 |
| HMMT25 | 9.8 | 55.4 | 54.1 |
| LiveCodeBench v6 | 29.1 | 51.8 | 56.6 |
| Arena-Hard v2 | 34.1 | 79.2 | 82.7 |
| BFCL-v3 (agentic) | 63.0 | 70.9 | 70.3 |

**Long context (RULER):**

| Context length | Qwen3-Next-80B-A3B accuracy |
|---|---|
| 4K | 98.5% |
| 256K | 93.5% |
| 512K | 86.9% |
| 1M | 80.3% |

The claims that carry weight:

- **The quality gap to the 235B flagship is small.** On reasoning (AIME25 69.5 vs 70.3) and knowledge (MMLU-Pro 80.6 vs 83.0) Qwen3-Next trails by roughly one to three points, while activating ~3B against the flagship's ~22B. On LiveCodeBench and Arena-Hard it actually *exceeds* the flagship. That is the load-bearing result: an architecture redesigned for efficiency did not have to surrender much quality to get it.
- **It dominates the dense Qwen3-32B outright.** Same family, more total capacity, far less activated compute, and it is not close — AIME25 69.5 vs 20.2 is the kind of gap that says the dense 32B and the sparse 80B-A3B are simply different weight classes despite comparable serving cost.
- **Long context degrades gracefully.** 98.5% at 4K to 80.3% at 1M is a real, monotonic decline — but a 1M-token model that still answers four in five RULER probes correctly is a usable long-context model, and the hybrid stack is what makes serving that length affordable.

What is load-bearing in the setup, and might not transfer:

1. **"10× throughput" is a long-context number.** It is measured at 32K+ context, where the quadratic attention term dominates and replacing 75% of it with linear layers pays off hugely. At short context — a 500-token chat turn — there is far less quadratic cost to remove, the MoE routing and the smaller activated-parameter count still help, but you should not expect a 10× wall-clock win on short prompts. Match the benchmark regime to your traffic before quoting the number.
2. **The benchmark suite under-probes linear-attention's weakness.** RULER is a long-context *retrieval* benchmark and 80.3% at 1M is genuinely informative. But the failure mode of a fixed-size state is specifically *precise multi-fact recall under interference* — many needles, similar distractors — and a single aggregate RULER score does not isolate it. The honest stress test is adversarial.
3. **The training-cost figure assumes the Qwen3 data pipeline already exists.** "10% of Qwen3-32B cost" is a marginal-compute comparison. The 15T-token corpus, the filtering, the verifier infrastructure — all inherited, none counted.

A worked latency example makes the regime-dependence concrete. Take two requests served on the same hardware. Request A is a 600-token chat turn generating a 200-token reply. Request B is a 400K-token document with a 2,000-token summary. For request A, prefill is trivial and decode is 200 memory-bandwidth-bound steps; Qwen3-Next's wins here come almost entirely from the small activated-parameter count (~3B streamed per step instead of a dense 32B's full 32B) and from MTP acceptance — real, but a factor of a few, not ten. For request B, prefill must process 400K tokens, and in a dense transformer that is 48 layers of quadratic attention; in Qwen3-Next it is 36 layers of linear DeltaNet plus 12 layers of (narrow-KV) attention. The quadratic term has been cut to a quarter of the layers, and the KV cache that must be held through all 2,000 decode steps is the ~25 GB hybrid figure instead of a multiple of it. *This* is where the 10× lives. The same model, the same weights, an order-of-magnitude difference in relative speedup — set entirely by the input length. Any capacity plan that quotes one number for both request types is wrong for at least one of them.

## Critique

**What is strong.** The composition is the achievement. Hybrid attention, ultra-sparse MoE, and MTP are each individually known; making all three work together at 80B scale, stably, with quality intact, is not a given, and the report is honest that the stability engineering (zero-centered layernorm and friends) is part of the price. The 3:1 division of labour is a genuinely good idea cleanly executed: it treats exact recall as a scarce resource to be spent deliberately rather than a property every layer must pay for. And reusing the MTP heads as a self-speculative draft model is the kind of detail that compounds — one objective that improves both training and serving, with no separate draft network to maintain.

The deeper thing the model demonstrates is an *economic* point, not just an architectural one. For most of the scaling era, "capacity" and "serving cost" were effectively the same variable — a better model was a more expensive model, full stop. Qwen3-Next is evidence that the two can be pried apart along three independent axes at once: total parameters (set by expert count), activated parameters (set by active-expert count), and attention cost (set by the linear/full ratio). A team that internalizes this stops asking "how big a model can we afford to serve" and starts asking three separate questions — how much capacity, how much per-token compute, how much context — each with its own knob. That reframing is worth more than any single benchmark number, because it changes the shape of the design space every future model in this family gets to explore. The Qwen3 flagship was the last word in the old framing; Qwen3-Next is the first word in the new one.

**What is weak or under-supported.**

- **The 3:1 ratio is asserted, not shown.** Why three DeltaNet layers per attention layer and not two, or five? The ratio is the model's central architectural hyperparameter and the public materials do not include the sweep. Maybe 3:1 is a sharp optimum; maybe the curve is flat between 2:1 and 4:1 and 3:1 was a round number. A reader cannot tell, and that is exactly the ablation that would settle whether the design is principled or tuned-by-vibes.
- **Linear-attention recall is reported in aggregate only.** RULER at 1M is good, but a hybrid model's whole risk is concentrated in *adversarial* recall — multi-needle, high-interference retrieval that a fixed-size state compresses away. One headline accuracy number is not a substitute for a recall-stress breakdown.
- **The throughput claim travels badly.** "10×" is true and impressive in its regime and quietly misleading out of it. A short-prompt chat deployment will see a real but much smaller speedup, and the release does not foreground that.
- **Three changes, one attribution problem.** Because hybrid attention, MoE sparsity, and MTP all changed at once, the report cannot cleanly say how much each contributed. That is a reasonable engineering choice for shipping a model and a frustrating one for anyone trying to learn *which* lever mattered most.

One more under-reported axis: the *thinking-mode* interaction. Qwen3-Next ships Instruct and Thinking variants, and the thinking variant emits long reasoning traces — which are themselves long context the model must attend over. A reasoning trace is exactly the integration-heavy, low-interference text the linear layers handle best, so the pairing is plausibly synergistic; but the release does not isolate whether hybrid attention costs anything on long-trace reasoning specifically. It is a benign-looking gap that happens to sit on the model's most valuable workload.

**What would change my mind.** If an independent long-context evaluation showed Qwen3-Next holding up on adversarial multi-needle retrieval — not just aggregate RULER — I would treat the 3:1 hybrid as a settled, general recipe rather than a Qwen-specific tuning. Conversely, if a recall-stress benchmark showed the linear layers dropping facts that a full-attention model of the same size keeps, the model would move from "flagship quality, fraction of the cost" to "flagship quality *except* on the workloads long context exists for" — a much narrower claim, and one the current benchmark suite is not designed to expose.

## What I'd build with this

1. **A long-context-first serving tier.** Qwen3-Next's economics specifically reward long inputs. Route your large-document, whole-repository, and long-transcript traffic to a Qwen3-Next endpoint and keep short chat on a cheaper dense model — the opposite of the usual "big model for hard things" split, because here the *context length*, not the difficulty, is what the architecture is tuned for.
2. **A hybrid-ratio probe for your own data.** Before trusting 3:1, build the ablation the report skips: evaluate the model on a recall-stress set you construct from your domain (many similar facts, adversarial distractors). If recall holds, adopt aggressively; if it cracks, you have found the workload to keep on full attention.
3. **MTP acceptance-rate monitoring.** Self-speculative decoding only speeds things up when the MTP draft is accepted often. Instrument the acceptance rate per traffic class — it will be high on predictable, templated output and low on high-entropy creative generation. That number tells you, per workload, whether the speculative path is actually earning its place.
4. **A KV-budget capacity planner.** Because only 12 of 48 layers grow a KV cache, the memory math for Qwen3-Next is genuinely different from a dense transformer. Build the capacity model explicitly — ~25 GB at 1M tokens — and you will find you can pack far more concurrent long-context requests onto a card than dense-transformer intuition predicts. The planner should model two separate terms: a constant per-request offset (the 36 DeltaNet states, fixed regardless of length) and a linear-in-tokens term (the 12 attention layers' KV). For a fleet serving mixed context lengths, that two-term model is what lets you safely oversubscribe — short requests cost almost only the offset, long requests cost mostly the slope, and the scheduler can pack them against a single memory budget instead of reserving worst-case KV for every slot.
5. **A hybrid-vs-dense A/B on your own evals.** The benchmark tables compare Qwen3-Next to other Qwen models; they do not tell you how it behaves on *your* task distribution. Run the model and your current dense model side by side on a frozen eval set, segmented by input length and by task type (summarize / retrieve / reason / generate). The expected shape is a near-tie or win on most segments and a possible dip on interference-heavy retrieval — and if that dip does not appear on your data, you have license to migrate the whole workload, not just the long-context slice.

## References

- **Qwen3-Next-80B-A3B-Instruct** — [model card on Hugging Face](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- **Qwen3-Next-80B-A3B-Thinking** — [model card on Hugging Face](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)
- **Qwen blog and code** — [github.com/QwenLM](https://github.com/QwenLM)
- Related on this blog:
  - [Qwen3 Technical Report: One Model, Two Minds](/blog/paper-reading/large-language-model/qwen3-technical-report)
  - [Gated Attention for LLMs: non-linearity, sparsity, attention-sink-free](/blog/paper-reading/large-language-model/gated-attention-for-large-language-models-non-linearity-sparsity-and-attention-sink-free)
  - [Speed Always Wins: a survey on efficient architectures for LLMs](/blog/paper-reading/large-language-model/speed-always-wins-a-survey-on-efficient-architectures-for-large-language-models)
  - [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache)
  - [Optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference)
  - [Speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding)

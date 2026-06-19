---
title: "IndexCache: How Cross-Layer Index Reuse Cuts the Cost of Sparse Attention"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A close read of IndexCache: it removes up to 75% of the lightning-indexer computations in DeepSeek Sparse Attention by reusing top-k indices across layers, buying 1.82x prefill and 1.48x decode speedups with no quality loss."
tags:
  - sparse-attention
  - deepseek-sparse-attention
  - long-context
  - lightning-indexer
  - kv-cache
  - inference-optimization
  - attention
  - cross-layer-reuse
  - glm
  - paper-reading
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 30
---

> [!tldr]
> - **The claim.** DeepSeek Sparse Attention (DSA) made *core attention* cheap, but its lightning **indexer** still runs at $O(L^2)$ in every layer. IndexCache keeps the indexer in only a quarter of the layers (the "Full" layers) and lets the rest (the "Shared" layers) reuse the nearest Full layer's cached top-k indices.
> - **Why it matters.** At 200K tokens the indexer is **81% of prefill latency** in a 30B DSA model. Removing 75% of indexer calls gives **1.82x prefill** and **1.48x per-request decode** speedups with essentially no accuracy change, on production-grade GLM models.
> - **The load-bearing observation.** Adjacent layers pick **70-100% of the same top-k tokens**. The indexer's *output* — not just full attention — is stable across depth, so you do not need a full-attention oracle to decide what to reuse.
> - **Two routes, one surprise.** A *training-free* greedy search (no weight updates) recovers baseline quality at 1/4 retention; a *training-aware* multi-layer distillation loss does even better — and it is provably equivalent to distilling each kept indexer toward the **centroid** of the attention distributions it serves.
> - **Where it fails.** Push to 1/8 retention and quality drops noticeably even with search; the early layers are genuinely irreplaceable because their errors propagate the furthest.

Every efficiency paper in long-context land eventually runs into the same humbling fact: the thing you optimized is no longer the bottleneck, and the cheap helper you bolted on to make it fast has quietly become the new one. IndexCache (Bai et al., 2026, Tsinghua University and Z.ai) is a clean instance of that story, and the fix is satisfying enough that I expect it to become a default in sparse-attention serving stacks within a release cycle or two.

The setup: DeepSeek Sparse Attention replaces the $O(L^2)$ attention score matrix with a lightweight **lightning indexer** that scores all preceding tokens and selects the top-$k$ to attend to, dropping core attention to $O(Lk)$. The problem: that indexer is itself $O(L^2)$, and it runs independently in every layer. Across $N$ layers that is $O(NL^2)$ of work that nobody removed — they just moved it. IndexCache removes most of it by noticing that consecutive layers almost always select the same tokens, so the index only needs to be computed a handful of times and reused everywhere else.

![IndexCache mental model: a few Full layers run the indexer and cache the top-k, while the majority of Shared layers skip the indexer and reuse the nearest cache.](/imgs/blogs/indexcache-cross-layer-index-reuse-1.webp)

The diagram above is the mental model: a Full layer (`F`) pays the full $O(L^2)$ indexer cost once and writes its top-$k$ index set into a tiny cache; the Shared layers (`S`) below it skip the indexer entirely and read that cache in $O(1)$. The cache holds exactly one index tensor and is overwritten at the next Full layer — so this costs no extra GPU memory beyond what DSA already allocates. Everything interesting in the paper is about *which* layers to keep Full, and whether you can do better by training for the sharing pattern instead of accepting it.

## Context: what came before

To see why IndexCache is the right shape of fix, you have to be precise about where the cost actually lives in a DSA layer. This is the part most summaries get wrong, so it is worth slowing down.

### Anatomy of a DSA layer

DeepSeek Sparse Attention decomposes each attention layer into two stages: **selection** and **computation**. Selection is the lightning indexer — a multi-head, ReLU-gated dot product between the current query and every preceding key — which produces a score vector $I_t^{(\ell)} \in \mathbb{R}^L$ for query position $t$ at layer $\ell$, where $L$ is the sequence length. From that score vector it keeps the top-$k$ positions, $\mathcal{T}_t^{(\ell)} = \text{Top-}k(I_t^{(\ell)})$, with $k = 2048$ throughout the paper. Computation is the main attention — Multi-head Latent Attention (MLA) — but evaluated only over those $k$ selected tokens.

![Anatomy of one DSA layer: the lightning indexer scores all L tokens at O(L-squared), and only the cheap top-k core attention runs at O(Lk).](/imgs/blogs/indexcache-cross-layer-index-reuse-2.webp)

That split is what makes DSA fast: $k = 2048 \ll L$, so core attention drops from $O(L^2)$ to $O(Lk)$. The indexer is deliberately engineered to be cheap *per FLOP* — few heads, low-rank projections, FP8 arithmetic — an order of magnitude lighter than the main MLA path. DSA is trained into an existing MLA model through two-stage continued pre-training: a short **dense warm-up** that trains only the indexer (via KL-divergence distillation against the aggregated full-attention distribution at each layer, with everything else frozen), then a longer **sparse training** phase that activates top-$k$ selection and jointly optimizes the whole model, with the indexer receiving its distillation gradients on a detached computational graph.

Here is the catch that everyone glosses. "Cheaper per FLOP" is not "cheap." The indexer still scores *all* $L$ preceding tokens — it operates at $O(L^2)$ — and it does so independently at every one of the $N$ layers. The total indexer cost is $O(NL^2)$, and because it grows quadratically with context while core attention grows only as $O(Lk)$, the indexer's *share* of the budget climbs as context gets longer.

### The indexer is the new bottleneck

The paper profiles a 30B DSA model and the numbers are stark. As a fraction of total latency, the indexer accounts for 27% of prefill at 10K tokens, 50% at 60K, 68% at 120K, and **81% at 200K**. Decode follows the same trend more gently (27% to 41% over the same range). The "cheap" helper is now the dominant line item.

![The indexer becomes the bottleneck at long context: its share of prefill latency grows from 27% at 10K tokens to 81% at 200K.](/imgs/blogs/indexcache-cross-layer-index-reuse-3.webp)

This is the kind of chart that should reorganize your priorities. If you are serving 200K-token agentic workloads — the paper's stated motivation, and increasingly everyone's — then four-fifths of your prefill time is being spent deciding *what* to attend to, not actually attending. Any speedup you want has to come from the selection stage, because the computation stage is already as sparse as it is going to get. Reducing indexer cost is the whole game for long-context DSA.

A back-of-the-envelope check makes the 81% feel less like a magic number. The indexer scores $L$ tokens for each of $L$ queries, so its work scales as $L^2$; core attention touches only $k$ tokens per query, so it scales as $Lk$. The ratio of the two is $L/k$. At a 2K prompt ($L = 2048 = k$) that ratio is 1 — the indexer and the attention it feeds do comparable amounts of token-level work. At $L = 200{,}000$ the ratio is $200000 / 2048 \approx 98$: the indexer now performs almost a hundred times more token-scoring than the sparse attention performs token-mixing. The indexer is roughly an order of magnitude cheaper *per FLOP* (low-rank projections, FP8, few heads), so the two effects roughly cancel at short context and the indexer pulls clear by about an order of magnitude at long context. That is precisely the 27%-to-81% climb the profiler reports — the asymptotics were always going to win, the only question was at what context length they would dominate, and the answer turns out to be "well inside the range people actually deploy."

### The redundancy nobody had exploited for sparse attention

The escape hatch comes from a property that the interpretability and efficiency literatures had already documented for *dense* models: the set of important tokens is remarkably stable across consecutive transformer layers. Deshmukh et al. (2025) and Gao et al. (2026) both show that adjacent layers share the vast majority of their top-$k$ attention mass, and methods like TidalDecode, OmniKV, and DELTA exploit this by designating a few *anchor* layers that compute full attention and letting intermediate layers reuse the anchor's top-$k$ indices.

But every one of those methods uses **full attention as the oracle** that identifies the important tokens. In DSA, full attention has been deleted — there is no $O(L^2)$ attention map to consult, only the lightweight indexer. So the open question the paper actually answers is narrower and more interesting: *does the indexer's own output exhibit the same cross-layer stability?* If it does, you can share indices without ever materializing a full-attention oracle.

It does. The authors compute the pairwise top-$k$ overlap ratio $|\mathcal{T}^{(i)} \cap \mathcal{T}^{(j)}| / k$ between all layer pairs of the 47-layer model, averaged over 768 samples of 200K length.

![Cross-layer top-k overlap heatmap: adjacent layers reuse 70 to 100% of each other's selected tokens, with block structure on the diagonal and divergence only between early and late layers.](/imgs/blogs/indexcache-cross-layer-index-reuse-4.webp)

The heatmap tells a layered story. Adjacent layers overlap 0.7 to 1.0 — they are selecting nearly the same tokens. There is clear **block structure**: clusters of layers (3-5, 6-8, 17-30, 31-36, and so on) with mutually high overlap, suggesting the model organizes into functional blocks where token selection is internally consistent. Overlap decays *unevenly* — faster across block boundaries than within them — and the early-to-late corners are dark (overlap below 0.4), confirming that early and late layers attend to fundamentally different token subsets. The takeaway: most per-layer indexers are computing something a neighbor already computed. That is the redundancy IndexCache harvests.

## What IndexCache contributes

Stripped to its claims, the paper offers four things:

1. **It identifies the indexer as the dominant long-context cost in DSA** and quantifies it (up to 81% of prefill at 200K), reframing "make sparse attention faster" as "make the *selector* run fewer times."
2. **It shows the indexer's top-$k$ output is cross-layer stable** even without a full-attention oracle, and turns that into a concrete mechanism: partition layers into Full (`F`) and Shared (`S`), where `S` layers inherit the nearest preceding `F` layer's indices through a single conditional branch and no extra memory.
3. **A training-free greedy layer-selection algorithm** that uses language-modeling loss on a small calibration set to decide *which* indexers to keep, retaining only 1/4 of them while matching the original DSA model's downstream quality.
4. **A training-aware multi-layer distillation loss** that trains each kept indexer to serve multiple layers at once — and a clean proof that this is exactly equivalent to distilling toward the centroid of the served layers' attention distributions, which lets even a naive uniform sharing pattern match the full-indexer baseline.

The rest of this post walks the method, the experiments, and where I think it bends.

## Method

### The Full / Shared partition

Formally, IndexCache encodes the layer roles as a binary **pattern string** $c = c_1 c_2 \cdots c_N$ with $c_\ell \in \{\texttt{F}, \texttt{S}\}$:

- A **Full** layer ($\texttt{F}$) retains its indexer, computes a fresh top-$k$ set $\mathcal{T}_t^{(\ell)}$ over all preceding tokens, and performs sparse core attention on that subset — identical to standard DSA.
- A **Shared** layer ($\texttt{S}$) has *no* indexer. It inherits the index set from the nearest preceding Full layer: $\mathcal{T}_t^{(\ell)} \leftarrow \mathcal{T}_t^{(f(\ell))}$ where $f(\ell) = \max\{j < \ell : c_j = \texttt{F}\}$, and applies sparse core attention directly on those inherited indices.

Layer 1 is always Full, to seed the initial indices. At inference, a Shared layer simply skips the indexer forward pass and reads the cached index tensor from its Full predecessor.

![IndexCache adds one conditional branch to the per-layer loop: a test on the pattern bit routes a layer to run the indexer and cache its top-k, or to reuse the cached top-k.](/imgs/blogs/indexcache-cross-layer-index-reuse-5.webp)

The implementation change is almost insultingly small — a single conditional branch in the per-layer loop:

```python
def dsa_layer_forward(x, layer):
    # Standard DSA: every layer runs its own O(L^2) indexer.
    scores = layer.lightning_indexer(x)            # [L, L], ReLU-gated, FP8
    idx = scores.topk(k=2048, dim=-1).indices      # O(L^2) selection
    return layer.sparse_attention(x, idx)          # O(Lk) computation

def indexcache_layer_forward(x, layer, cache, pattern_bit):
    # IndexCache: only Full layers pay for the indexer.
    if pattern_bit == "F":
        scores = layer.lightning_indexer(x)        # runs here, O(L^2)
        idx = scores.topk(k=2048, dim=-1).indices
        cache.top_k = idx                          # overwrite the single buffer
    else:                                          # "S": reuse, O(1)
        idx = cache.top_k                          # no indexer forward pass
    return layer.sparse_attention(x, idx)
```

`cache.top_k` is a temporary buffer holding only the current index tensor; it is overwritten at every Full layer and requires no additional GPU memory beyond what standard DSA already allocates. The core attention path — the $O(Lk)$ part — is untouched. So the only question that matters is how to choose the pattern $c$. The paper gives two answers.

### Training-free IndexCache: greedy layer selection

Suppose you have a pretrained DSA model and you are not allowed to touch its weights. You want a pattern $c$ that maximizes the number of Shared layers while minimizing the hit to quality. The obvious move is **uniform interleaving**: keep every $r$-th indexer and skip the rest (e.g., `FSSSFSSS...` for $r = 4$). The paper shows this is a trap.

The reason is that indexer importance varies sharply across layers. Early and transitional layers are far more sensitive to indexer removal than mid-block layers — recall the heatmap, where early layers diverge most from their neighbors. Uniform interleaving is blind to this: it will happily delete a critical early indexer while keeping a redundant mid-block one, and the quality drop is real (quantified below). So the authors let the model tell them which indexers are expendable, via a greedy search guided by language-modeling loss.

![Greedy search drains the expendable indexers first: starting from all-Full, each step flips the layer whose removal least raises LM loss, so critical layers survive longest.](/imgs/blogs/indexcache-cross-layer-index-reuse-6.webp)

The algorithm caches $B$ mini-batches from the training data and evaluates every candidate pattern on *exactly the same batches*, so loss differences reflect the pattern change and not data variance. Starting from the all-Full baseline, it runs for $K$ steps (e.g., $K = 3N/4$ to retain 1/4 of indexers); at each step it tentatively flips each currently-Full layer to Shared, measures the resulting LM loss, and commits the flip that yields the lowest loss:

```python
def greedy_select(model, calib_batches, K):
    # c[l] in {"F", "S"}; layer 0 is always Full (it seeds the indices).
    pattern = ["F"] * model.n_layers
    candidates = list(range(1, model.n_layers))          # layer 0 is excluded
    for step in range(K):                                # K = 3N/4 keeps 1/4
        best_l, best_loss = None, float("inf")
        for l in candidates:
            trial = pattern.copy()
            trial[l] = "S"
            loss = eval_lm_loss(model, calib_batches, trial)   # same batches every time
            if loss < best_loss:
                best_loss, best_l = loss, l
        pattern[best_l] = "S"                             # commit least-damaging flip
        candidates.remove(best_l)
    return pattern
```

A full search from all-Full to all-Shared costs $N(N-1)/2$ forward passes, which sounds expensive but is cheap relative to training. When the model is partitioned across $P$ pipeline-parallel stages, the search itself parallelizes: split the layers into $P$ blocks (each block's first layer fixed as Full), search blocks sequentially within each step, and commit the best flip per block before the next — placing up to $P$ layers per step and cutting total forward passes by roughly $P$x.

The authors report three properties of the greedy solution that I find convincing:

- It **outperforms uniform interleaving at the same retention ratio**, which is the whole point.
- The per-step validation-loss curve shows a clean separation between "easy" layers (the first ~20 flips barely move the loss) and "critical" layers (after ~35 flips the loss rises sharply) — a natural ordering of indexer importance.
- The ranking is **stable across different calibration sets**, indicating it is an intrinsic model property rather than a data artifact, and lower LM loss correlates with better downstream task performance — so LM loss is a valid cheap proxy.

### Training-aware IndexCache: distilling toward the centroid

Training-free IndexCache is limited by a subtle fact: each indexer was originally trained to serve *only its own layer*. If you are training a DSA model from scratch, or doing continued pre-training, you can do better — explicitly train each retained indexer to serve *multiple* layers at once.

In standard DSA training, each indexer at layer $\ell$ is distilled via KL divergence against its own layer's aggregated attention distribution $p_t^{(\ell)}$: the loss is $\mathcal{L}^{\text{I}} = \sum_t D_{\text{KL}}(p_t^{(\ell)} \,\|\, q_t^{(\ell)})$, where $q_t^{(\ell)} = \text{Softmax}(I_t^{(\ell)})$ is the indexer's output distribution. IndexCache generalizes this to a multi-layer objective. Let a retained Full layer $\ell$ serve the Shared layers $\ell+1, \ldots, \ell+m$ that reuse its index set. The multi-layer distillation loss is

$$
\mathcal{L}^{\text{I}}_{\text{multi}} = \sum_{j=0}^{m} \frac{1}{m+1} \sum_t D_{\text{KL}}\!\left(p_t^{(\ell+j)} \,\big\|\, q_t^{(\ell)}\right).
$$

Intuitively this asks the indexer to predict a top-$k$ set that is jointly useful for *all* the layers it serves, rather than overfitting to layer $\ell$ alone.

![Multi-layer distillation trains the indexer toward a centroid: summing the KL over every served layer is identical to one KL against their averaged attention distribution.](/imgs/blogs/indexcache-cross-layer-index-reuse-7.webp)

The elegant part is **Proposition 1**: this multi-layer loss produces *exactly the same gradient* as distilling against a single averaged target. Define the centroid $\bar{p}_t = \sum_{j=0}^{m} \frac{1}{m+1} p_t^{(\ell+j)}$ and the single-target loss $\mathcal{L}^{\text{I}}_{\text{avg}} = \sum_t D_{\text{KL}}(\bar{p}_t \,\|\, q_t^{(\ell)})$. Then $\nabla_\theta \mathcal{L}^{\text{I}}_{\text{multi}} = \nabla_\theta \mathcal{L}^{\text{I}}_{\text{avg}}$.

The proof is a one-liner once you see it. In $D_{\text{KL}}(p \,\|\, q^{(\ell)})$, only $q^{(\ell)}$ depends on the parameters $\theta$, so the entropy term of $p$ vanishes under differentiation and the gradient reduces to $-\nabla_\theta \sum_s p(s) \log q^{(\ell)}(s)$. Because that expression is *linear in $p$*, summing it over the served layers and dividing by $m+1$ is identical to evaluating it once at the average $\bar{p}_t$:

```python
import torch.nn.functional as F

def multilayer_distill_loss(indexer_logits, served_attn):
    # indexer_logits: [L, L] from a retained Full layer's indexer -> q
    # served_attn:    [m+1, L, L] aggregated attention p of every layer it serves
    log_q = F.log_softmax(indexer_logits, dim=-1)         # log q^(l)
    # Proposition 1: mean of per-layer KL == KL against the centroid p_bar,
    # because the cross-entropy term is linear in p.
    p_bar = served_attn.mean(dim=0)                       # centroid over served layers
    return F.kl_div(log_q, p_bar, reduction="batchmean")  # one KL, same gradient
```

So multi-layer distillation is not a heuristic regularizer — it is, provably, training the indexer toward the *centroid* of its served layers' attention distributions. The indexer learns a consensus top-$k$ that covers the important tokens across every layer it feeds, instead of specializing to one.

Two practical notes the authors flag. First, they adopt $\mathcal{L}^{\text{I}}_{\text{multi}}$ in practice rather than $\mathcal{L}^{\text{I}}_{\text{avg}}$, despite identical gradients, for memory reasons: when the subsequent layer is a Shared layer, it only needs the current layer's predicted $q^{(\ell)}$, whereas the averaged form requires materializing both $q^{(\ell)}$ and the per-layer $p^{(\ell)}$. Second, training follows the same two-stage DSA recipe: a warm-up phase trains the indexer in the Full layers with $\mathcal{L}^{\text{I}}_{\text{multi}}$ while other parameters stay fixed, then a sparse-training phase continues with $\mathcal{L}^{\text{I}}_{\text{multi}}$ (computed over the selected top-$k$ tokens) plus the LM loss for the remaining parameters.

### Why training erases the pattern sensitivity

There is a striking contrast between the two regimes, and it is the most instructive result in the paper. In the *training-free* setting, the greedy-searched pattern is *essential* — uniform interleaving degrades quality badly at aggressive ratios. In the *training-aware* setting, uniform interleaving works just as well as the searched pattern. What changed?

![Why uniform reuse needs training: frozen weights turn inherited indices into a distribution shift that cascades, while trained Shared layers adapt and the gap closes.](/imgs/blogs/indexcache-cross-layer-index-reuse-8.webp)

With frozen weights, a Shared layer is fed an index set chosen by a *different* layer's indexer. Even if the two layers select mostly the same tokens, the small set of mismatched tokens introduces a distributional shift in the hidden state, and because early layers sit at the start of the longest propagation path, that shift cascades and compounds through every downstream layer. Greedy search works precisely by *avoiding* these sensitive layers — keeping their indexers Full so no foreign index set ever reaches them.

When you retrain for sharing, the Shared layers learn to adapt their attention to the inherited indices, and the retained indexers simultaneously learn (via the centroid objective) to produce selections that generalize across the layers they serve. That joint adaptation eliminates the layer-specific sensitivity entirely — which is why a simple uniform pattern suddenly suffices. This is one of those rare cases where a training objective doesn't just improve a number, it removes the need for a separate search procedure.

## Experiments

The evaluation is on a 30B DSA model obtained by training GLM-4.7-Flash (a 30B-A3B MoE with MLA and 47 layers) into DSA, served with `dp_attention` (`dp_size=8`) in SGLang on an NVIDIA H100 node. The benchmark suite is broad: five long-context tasks (MRCR v2, GraphWalks, LongBench v2, RULER, AA-LCR) and four general and reasoning tasks (AIME 2025, GPQA-Diamond, LiveCodeBench v6, IFBench). All benchmarks use temperature 1.0, top-$p$ 0.95, top-$k$ 40, with a 200K context window (32K reserved for output) on long-context tasks.

### End-to-end speedups

This is the headline. IndexCache is compared against the DSA baseline at two retention ratios — 1/2 (half the indexers kept) and 1/4 (a quarter kept) — across context lengths from 10K to 200K, on three metrics: prefill latency, per-request decode throughput (single request per GPU), and full decode throughput (KV cache saturated, ~800K tokens per GPU).

![IndexCache speedups at 200K context on the 30B DSA model: removing 75% of indexers cuts prefill 1.82x and lifts decode throughput up to 1.51x with negligible quality loss.](/imgs/blogs/indexcache-cross-layer-index-reuse-9.webp)

The full table at the four context lengths:

| Metric (200K unless noted) | DSA | + IndexCache 1/2 | + IndexCache 1/4 |
| --- | --- | --- | --- |
| Prefill time, 10K (s) | 0.57 | 0.47 | **0.45** (1.27x) |
| Prefill time, 200K (s) | 19.5 | 13.7 | **10.7** (1.82x) |
| Decode / request, 200K (tok/s) | 58.0 | 73.0 | **86.0** (1.48x) |
| Decode full KV, 200K (tok/s) | 197 | 253 | **297** (1.51x) |

The pattern is exactly what the bottleneck analysis predicts. Prefill speedup *grows with context* — 1.27x at 10K rising to 1.82x at 200K — because the indexer's share of prefill grows with context, and IndexCache attacks precisely that share. Decode improves because DSA's decode involves a per-token indexer pass over the full context, which becomes the bottleneck for long sequences; removing three-quarters of those passes lifts per-request decode by 1.48x and full throughput by 1.51x at 200K. The authors note the same trends hold on the much larger 744B GLM-5 model, where IndexCache (1/4) delivers at least 1.3x improvement in both prefill and decode beyond 100K context, and roughly 1.2x end-to-end by removing 50% of indexer computations.

### Training-free quality: which layers, not how many

Table 2 reports training-free IndexCache at 1/2, 1/4, and 1/8 retention, comparing uniform interleaving against the greedy-searched pattern. I have condensed the long-context average (Long) and the general-and-reasoning average (G&R):

| Config | Long Avg | G&R Avg |
| --- | --- | --- |
| Original DSA | 50.2 | 74.6 |
| 1/2 uniform | 47.4 | 74.3 |
| 1/2 + searched | **50.3** | 74.4 |
| 1/4 uniform | 43.0 | 73.8 |
| 1/4 + searched | **49.9** | 74.9 |
| 1/8 uniform | 35.3 | 70.0 |
| 1/8 + searched | 46.1 | 73.7 |

Three things jump out. First, **uniform interleaving bleeds long-context quality** — 2.8 points at 1/2 and 7.2 points at 1/4 — while the searched pattern recovers almost all of it (50.3 and 49.9, both within a point of the 50.2 baseline). This confirms the thesis that *which* indexers you keep matters far more than how many. Second, the general-and-reasoning average barely moves across all configurations except 1/8 uniform; long chain-of-thought reasoning is preserved. Notably, the 1/4 searched pattern actually *improves* over DSA on AIME 2025 (92.6 vs 91.0) and GPQA-Diamond (78.6 vs 77.6) — removing redundant indexer computation appears to act as a mild regularizer at inference. Third, at the extreme 1/8 ratio even search cannot save you: Long Avg falls to 46.1. Beyond a point, the indexers really are load-bearing.

### Training-aware quality: uniform catches up

Table 3 shows training-aware IndexCache (using the multi-layer distillation loss) at 1/2 and 1/4 retention with uniform interleaving, plus two ablations:

| Config | Long Avg | G&R Avg |
| --- | --- | --- |
| Original DSA | 51.0 | 74.2 |
| 1/2 uniform IndexCache | **51.6** | 74.5 |
| &nbsp;&nbsp;w/ searched pattern | 50.6 | 73.6 |
| &nbsp;&nbsp;w/o cross-layer loss | 49.8 | 74.5 |
| 1/4 uniform IndexCache | 50.6 | 74.1 |

(The DSA baseline here is trained with the paper's shortened pipeline, so it differs slightly from Table 2.) Uniform IndexCache at 1/2 retention *surpasses* the DSA baseline on Long Avg (51.6 vs 51.0) and holds G&R within noise; at 1/4 both averages are within 0.4% of baseline. The pattern sensitivity from the training-free setting has vanished — uniform interleaving now performs on par with, and even slightly above, the greedy-searched pattern (51.6 vs 50.6). And the cross-layer distillation loss earns its keep: removing it drops Long Avg from 51.6 to 49.8, with AA-LCR collapsing from 49.8 to 44.0. Training each indexer toward the centroid of its served layers is doing real work, not decoration.

### Scaling to 744B (GLM-5)

The result that matters most for deployment is the one on the production-scale model. GLM-5 is a 744B-parameter (40B active) MoE that uses DSA by default, and the authors apply *training-free* IndexCache to it across five long-context benchmarks:

| Config | Long Avg | MRCR v2 | GraphWalks | LongBench v2 | RULER | AA-LCR |
| --- | --- | --- | --- | --- | --- | --- |
| Original DSA | 78.4 | 71.1 | **92.7** | 64.5 | **97.7** | 66.2 |
| 1/2 uniform | 78.1 | 72.8 | 90.2 | 65.1 | 97.6 | 64.6 |
| &nbsp;&nbsp;+ searched | **78.7** | 72.3 | 90.8 | **66.0** | 97.3 | 67.2 |
| 1/4 uniform | 72.7 | 65.8 | 74.9 | 62.2 | 96.2 | 64.6 |
| &nbsp;&nbsp;+ searched | 78.0 | 70.8 | 90.3 | 63.7 | 97.6 | **67.6** |

The trends mirror the 30B findings exactly: uniform interleaving degrades at the aggressive 1/4 ratio (Long Avg 72.7), and the searched pattern recovers it (78.0, within 0.4 of the 78.4 baseline). At 1/2 retention the searched pattern slightly exceeds baseline (78.7 vs 78.4). The authors are admirably careful here — they flag that 1/2 uniform happening to preserve Long Avg (78.1) is *likely coincidental*, a case where the fixed alternating pattern simply avoids skipping the most critical indexer layers by luck, whereas the searched pattern is consistently stable by construction. They also ran an all-round evaluation with 1/2 retention across the Artificial Analysis Index and report performance nearly identical to the original GLM-5 (Figure 1 in the paper). Training-aware adaptation of GLM-5 is left as future work — which is the right call, given a full DSA training pipeline from the base model is expensive, but it does mean the *strongest* form of IndexCache remains unproven at this scale.

### The negative result worth reading

I appreciate that the authors devote an appendix to a *failed* idea, and it is genuinely informative. Before the greedy LM-loss search, they tried the seemingly natural alternative: pick the sharing pattern by directly measuring how *similar* the attention outputs are when an indexer is reused across layers. They build an $N \times N$ similarity matrix $S_{i,j}$ — the cosine similarity between layer $i$'s core-attention output using its own index versus reusing layer $j$'s index — and solve for the optimal pattern via dynamic programming.

It does not work. The similarity-optimal patterns perform comparably to naive uniform interleaving, with the same quality degradation. The reason is sharp: per-layer output similarity is a *local* metric. Two layers can have nearly identical attention outputs ($S_{i,j} \approx 1$) yet differ in a small set of tokens whose importance only becomes apparent in later layers' reasoning. Those subtle mismatches accumulate across depth, producing final quality drops that no layer-local similarity score can predict. The greedy LM-loss search avoids this trap by optimizing a *global* metric — the end-to-end effect of each sharing decision on the model's output distribution. The lesson generalizes well beyond this paper: when a cheap local proxy and an expensive global metric disagree, the disagreement is usually the part you care about.

## How it relates to prior cross-layer sharing

Cross-layer reuse is not new — it is one of the busiest corners of the efficient-attention literature. What is new is *where* IndexCache applies it. It helps to lay the lineage side by side:

| Method | What it reuses | Oracle for selection | How the sharing is chosen |
| --- | --- | --- | --- |
| TidalDecode / OmniKV / DELTA | top-$k$ indices | full $O(L^2)$ attention | fixed or periodic anchor layers |
| Cascade | top-$k$ indices | full $O(L^2)$ attention | dynamic programming + head-aware remapping |
| Cross-layer KV (CLA-style) | key/value tensors | n/a (memory, not selection) | fixed layer groups |
| HySparse | indices **and** KV cache | full $O(L^2)$ attention | interleaved full + sparse layers |
| **IndexCache** | top-$k$ indices | the cheap **lightning indexer** | greedy LM-loss search / centroid distillation |

Two differences are load-bearing. First, **the oracle is fundamentally cheaper**. Every prior index-sharing method needs a few full-attention "anchor" layers to compute the exact top-$k$ that the sparse layers then copy — which means those anchors pay the full $O(L^2)$ attention cost the rest of the model is trying to avoid. IndexCache shares the output of DSA's lightweight indexer instead, so its "anchor" (a Full layer) is already an order of magnitude cheaper than a full-attention layer. There is no full-attention oracle anywhere in the model; the paper's contribution is showing you do not need one, because the indexer's own output is cross-layer stable.

Second, **the sharing configuration is optimized, not assumed**. The training-free greedy search and the training-aware centroid distillation are both systematic procedures for choosing and adapting the pattern, where prior work largely relied on fixed or heuristic anchor placement. And because IndexCache only assumes a *dynamic token-selection step* — not a particular attention kernel — the same principle drops onto block-level selectors like MoBA and NSA, where it has not yet been tried. Cross-layer sharing was a trick you applied where full attention was the oracle; IndexCache shows it survives the move to sparse attention, where the oracle is learned and cheap.

## Critique

### What's strong

The framing is honest and the mechanism is minimal. Reframing "speed up sparse attention" as "the selector is the cost now" is the kind of clarifying move that makes a paper worth reading even if you never deploy it. The single-conditional-branch implementation with zero extra memory is the right engineering — it composes with existing DSA serving stacks rather than replacing them. And Proposition 1 is a small gem: it turns a plausible-looking multi-layer loss into a provably principled centroid objective, which both explains *why* it works and tells you it has no hidden interaction terms to tune.

The empirical story is also unusually well-controlled for an efficiency paper. Evaluating every candidate pattern on identical calibration batches, reporting the failed similarity-based search, and showing the training-free vs training-aware contrast (which is itself a clean scientific result about where the sensitivity lives) all signal that the authors were trying to understand the phenomenon, not just post a speedup.

### What's weak or unproven

The 1/8 cliff is the honest limit, and the paper says so, but it bounds the upside: the dramatic 1.82x prefill number is at 1/4 retention, and you cannot simply crank the ratio further for more. The really aggressive regime is off the table without quality loss.

The training-aware results on the production-scale GLM-5 (744B) are explicitly *preliminary* — only the training-free variant is fully evaluated there, and the authors defer training-aware adaptation to future work. So the strongest version of the method (uniform pattern + multi-layer distillation matching baseline) is demonstrated only at 30B. That is a reasonable place to stop for a first paper, but "we expect it to scale" is a promissory note, not a result.

The greedy search is greedy: it offers no global-optimality guarantee, and while the authors observe nice properties (easy/critical separation, cross-calibration stability), there is no comparison against an exhaustive or near-optimal pattern at small $N$ to bound how much the greediness costs. And the calibration-set generality, though tested across a few sets, is all in-distribution training data — I would want to see whether a pattern searched on pre-training text transfers to, say, a code-heavy or multilingual serving distribution before trusting it in production.

### The missing ablation

The one experiment I most want and do not see: a sweep over **$k$**. The entire premise rests on top-$k$ overlap being high, and the paper fixes $k = 2048$ throughout. But overlap should depend on $k$ — for very small $k$ the selected sets are more brittle and may diverge faster across layers, while for large $k$ overlap is trivially high. Knowing how the safe retention ratio moves with $k$ would tell you whether IndexCache's headroom is a property of DSA or an artifact of this particular $k$. Relatedly, there is no head-level analysis: Cascade found head-aware remapping critical for cross-layer sharing, and IndexCache shares a single index set across all heads of a Shared layer — whether per-head sharing would extend the safe ratio is unexplored.

### What would change my mind

I would update toward "this is a transient trick, not a durable component" if either of two things turned out true: (1) the training-aware uniform-pattern result *fails to reproduce at 744B* — i.e., the pattern sensitivity that training erases at 30B comes back at scale, forcing you back into per-model greedy search; or (2) a sweep showed the safe retention ratio collapsing toward 1/2 as $k$ shrinks to the small values (256-512) that aggressive sparse-attention deployments actually want. Conversely, a clean training-aware GLM-5 result at 1/4 retention with intact quality would convince me this belongs in every DSA serving stack by default.

## What I'd build with this

1. **Port it to MoBA and NSA.** The authors point out that the principle extends to any sparse-attention method with a *dynamic* token-selection step — not just DSA. MoBA's block-level selection and NSA's block indices are both recomputed per layer and both plausibly cross-layer stable. A MoBA variant that shares block indices across layers via the same greedy search is a near-drop-in experiment, and MoBA's coarser block granularity might even be *more* shareable than DSA's token-level top-$k$. See the [MoBA paper read](/blog/paper-reading/large-language-model/moba) for where that selection step lives.

2. **An online / adaptive Full set.** The pattern is static here — chosen once and frozen. For an agentic workload whose context shifts regimes (retrieval dump, then code, then chat), an indexer that occasionally re-runs on a Shared layer and *checks* whether the cached top-$k$ still has high overlap — promoting a layer back to Full when overlap drops below a threshold — would adapt the sharing pattern to the actual input at a small monitoring cost.

3. **Stack it with cross-layer KV sharing.** IndexCache shares the *index* (which tokens); methods like HySparse share the *KV cache* (the tokens' values). They are orthogonal axes and the paper notes HySparse unifies both for full-attention oracles. A DSA model that shares both indices (IndexCache) and KV blocks across a layer group would compound the savings — index reuse cuts selection FLOPs, KV reuse cuts memory bandwidth. The [KV-cache management survey](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) maps that second axis.

4. **Per-head index sharing.** Give each Shared layer the option to inherit different Full layers' indices for different heads, with the head-to-source mapping itself chosen by the greedy search. This is the head-aware remapping idea from Cascade transplanted onto the indexer, and it might push the safe retention ratio past 1/4.

5. **A cheap deploy recipe for existing checkpoints.** Because the training-free variant needs no weight updates, you can ship it as a serving-time config: run the greedy search once on a few hundred calibration batches, save the pattern string, and load it as a per-model `c`. That is a weekend project for anyone already serving a DSA model, and it is where I would start. The [MiniMax Sparse Attention read](/blog/paper-reading/large-language-model/minimax-sparse-attention) covers a sibling learned-selector design whose Index Branch is the natural next target, and the [efficient-attention survey](/blog/paper-reading/large-language-model/efficient-attention-mechanisms-for-large-language-models-a-survey) places all of these on the same map.

The deeper point IndexCache makes is architectural, not just operational: the cross-layer sharing principle that the literature had only ever applied where full attention serves as the oracle extends naturally to sparse attention, where the oracle is a cheap learned indexer. As sparse attention becomes the default for frontier long-context models, reusing the selection across layers — not just the values — looks like it is going to be a standard line in the efficient-inference playbook.

## When to reach for IndexCache, and when not

Reach for it when you are **serving a DSA-style model on long contexts** and the indexer is measurably eating your latency. The whole value proposition scales with context length: at 10K tokens you get a modest 1.27x prefill, but at 200K you get 1.82x, and the gap only widens past 200K. If your traffic is dominated by long agentic sessions, retrieval-heavy prompts, or large codebases, this is close to free throughput. The training-free variant in particular is a serving-time config change — run the greedy search once, save the pattern string, ship it — so the cost of trying it is a few hundred calibration forward passes and an afternoon.

Reach for the **training-aware** variant if you are already training or continuing-pretraining a DSA model. The centroid distillation lets you drop to a uniform 1/4 pattern without a per-model search and even nudges quality up on reasoning benchmarks, so if you control the training run there is little reason to leave it on the table.

Do *not* reach for it when your context is short. Below ~10K tokens the indexer is a minority of the budget, so removing it buys little and you take on the (small) risk of a worse index. Do not push past 1/4 retention expecting free wins — the 1/8 cliff is real, and the early layers will not forgive you. And be cautious about trusting a pattern searched on one data distribution for a very different serving distribution until you have re-verified it; the ranking is stable in the paper's tests, but all of those tests are in-distribution. Finally, if you are not on a dynamic-selection sparse attention method at all — if you run dense attention, sliding-window, or a linear-attention variant — IndexCache simply does not apply; there is no per-layer index to cache.

## References

- **Paper:** Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li. *IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse.* arXiv:2603.12201 (2026). [PDF](https://arxiv.org/pdf/2603.12201)
- **DeepSeek Sparse Attention** (Liu et al., 2025) — the lightning-indexer mechanism IndexCache builds on; instantiated under Multi-head Latent Attention.
- **Cascade** (Deshmukh et al., 2025) and the cross-layer token-selection stability literature (Gao et al., 2026) — the full-attention-oracle precursors IndexCache generalizes to sparse attention.
- Related reads on this blog: [MiniMax Sparse Attention](/blog/paper-reading/large-language-model/minimax-sparse-attention), [MoBA: Mixture of Block Attention](/blog/paper-reading/large-language-model/moba), [a survey of efficient attention mechanisms](/blog/paper-reading/large-language-model/efficient-attention-mechanisms-for-large-language-models-a-survey), and [a survey of KV-cache management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management).

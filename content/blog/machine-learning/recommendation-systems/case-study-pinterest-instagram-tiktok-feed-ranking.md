---
title: "Case Study: Pinterest, Instagram, and TikTok Feed Ranking"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Reverse-engineer how the great visual and short-video feeds rank at billion-item scale: PinSage's web-scale GNN, Instagram's many-source multi-task DLRM ranking, and TikTok's Monolith real-time online learning, with the math, runnable PyTorch repros, and the reported lifts."
tags:
  [
    "recommendation-systems",
    "recsys",
    "feed-ranking",
    "pinsage",
    "graph-neural-networks",
    "multi-task-learning",
    "online-learning",
    "monolith",
    "machine-learning",
    "pytorch",
    "ranking",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-1.png"
---

Open Pinterest, Instagram, or TikTok and the experience feels effortless: an infinite shelf of things you did not ask for but somehow want. Behind that effortless feeling sits some of the largest machine-learning systems ever built — graphs with billions of nodes, embedding tables that span hundreds of hosts, models retrained continuously while you scroll. The product is a feed. The engineering is a funnel: take a corpus of billions of items, narrow it to thousands, score those thousands against a fistful of competing objectives, then re-rank for diversity and freshness before the screen paints.

This post is a case study of three of the best feed rankers in the world, and what each one teaches that the others do not. **Pinterest** taught the field how to run a graph neural network at web scale with [PinSage](https://arxiv.org/abs/1806.01973): random-walk neighbor sampling and importance pooling over a three-billion-node pin-board graph, with MapReduce inference and a hard-negative curriculum. **Instagram** (Meta) shows the many-candidate-source pattern at its most extreme — dozens of retrieval sources unioned into one pool, then a heavy multi-task ranker built on [DLRM](https://arxiv.org/abs/1906.00091)-style embeddings whose heads predict a dozen engagement events and fuse them into a single value score. **TikTok** (ByteDance) is the canonical real-time engine: its [Monolith](https://arxiv.org/abs/2209.07663) system uses collisionless embedding hashing and continuous online training so that a new interest you express in the last ten minutes is already shaping what you see.

![A six-layer stack diagram of the feed funnel showing many candidate sources flowing into a candidate pool, then multi-task ranking, then value fusion, then re-ranking, then the served feed](/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-1.png)

By the end you will be able to: draw the shared funnel all three systems run (figure 1) and name where they diverge; explain PinSage's random-walk importance pooling and why it scales when full-graph convolution does not; write the multi-objective value-fusion score $\text{score} = \sum_k w_k\, p_k$ that turns a dozen engagement predictions into one number; explain why short-video freshness demands online learning and what Monolith's collisionless hashing buys you; and reproduce three of the key ideas in runnable PyTorch — a small GraphSAGE-style pin embedding with neighbor sampling, a multi-objective value model over several engagement heads, and an incremental online-update sketch. The whole thing hangs on the series' recurring spine: the [retrieval → ranking → re-ranking funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) fed by the feedback loop, read off the offline-versus-online reality gap. If you want the map of the whole series first, start at [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system).

One honesty note before we begin. Of the three, only Pinterest and ByteDance have published detailed, peer-reviewed papers with numbers (PinSage in KDD 2018, Monolith on arXiv in 2022). Meta has published DLRM and a steady stream of engineering blog posts, but the exact production Reels architecture is described at a high level. TikTok's For You algorithm is mostly *not* officially documented beyond Monolith. So throughout I will clearly mark what is **published**, what is **widely reported / inferred**, and what is **my own small reproduction**. We will not fabricate a precise number. Where I give a figure with no citation, it is from my repro and labeled as such.

## 1. The one funnel three feeds share

Strip away the branding and all three feeds run the same five-stage pipeline shown in figure 1. It is worth stating crisply because the rest of the post is variations on it.

1. **Candidate sources.** Not one retriever but many. A two-tower embedding retriever, a graph-walk retriever, a "from people you follow" source, a "trending right now" source, a "similar to your last save" source. Each is cheap and specialized; each contributes a few hundred to a few thousand items. They are unioned and de-duplicated into a single candidate pool.
2. **Candidate pool.** Typically a few thousand items after the union. This is the set the expensive ranker will actually score. Everything upstream optimizes *recall* (did we get the good items into this pool at all?); everything downstream optimizes *precision* (did we order them right?).
3. **Multi-task ranking.** A heavy neural net scores each candidate, predicting not one thing but many: probability of click, of a long watch / completion, of a save, of a share, of a comment, of a follow, of a "see less" or skip. These are the model's *heads*.
4. **Value fusion.** The many predicted probabilities collapse into one ranking score, $\text{score} = \sum_k w_k\, p_k$, where the weights $w_k$ encode the product's idea of value (a share is worth more than a click; a "see less" is a strong negative). The pool is sorted by this score.
5. **Re-rank.** A final pass enforces diversity (no five pins from the same board in a row), freshness, fatigue/de-duplication, business rules, and sometimes a calibration pass. Then the top-K is served.

That is the skeleton. Pinterest's signature is *how it builds item embeddings* for stages 1–2 (a GNN on the content graph). Instagram's signature is *the breadth of stage 3 and the value model in stage 4* (a DLRM-scale multi-task net over many sources). TikTok's signature is *how fast the loop spins* (online learning that keeps the whole thing fresh by the minute). Same funnel; three different places to spend your hardest engineering.

A useful framing for the whole post: a feed is a **multi-stage, multi-objective ranking problem at extreme scale, fed by a fast feedback loop**. Every design choice below is an answer to one of those four pressures — multi-stage (you cannot score billions of items, so you retrieve then rank), multi-objective (a single proxy like clicks is gameable, so you predict many events), extreme scale (billions of items and features, so embeddings dominate), and fast feedback (interests drift within a session, so freshness matters). Keep those four in mind and each system stops looking like magic.

### Why a funnel at all? The arithmetic of scale

The reason nobody just runs the big model over the whole corpus is brutal arithmetic. Suppose your ranker takes 1 millisecond per candidate on a GPU (optimistic for a deep cross-feature net). Your corpus is $3 \times 10^9$ pins. Scoring the full corpus for one request is $3 \times 10^6$ seconds — about 35 days — per impression. You have roughly 100 ms of latency budget total. So you can afford to *rank* on the order of a few thousand candidates, which means *retrieval* must throw away 99.9999% of the corpus before the ranker ever sees it, and it must do so in a few milliseconds. That is the whole reason the funnel exists, and why retrieval is a maximum-inner-product-search problem solved with [approximate nearest neighbor indexes](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) rather than brute force. The funnel is not an aesthetic choice; it is forced by the latency-times-corpus budget.

## 2. Pinterest and PinSage: a GNN at web scale

Pinterest's core object is the *pin* (an image with a link) saved onto a *board* (a user-curated collection). The data is naturally a bipartite graph: pins on one side, boards on the other, an edge whenever a pin is saved to a board. Two pins are "related" if they are frequently saved to the same boards — that is collaborative signal expressed as graph structure. The task PinSage solves is to learn a $d$-dimensional embedding per pin such that related pins are close, so that retrieval and ranking can both use it. If you want the GNN fundamentals first, the series post on [graph neural networks for recommendation](/blog/machine-learning/recommendation-systems/graph-neural-networks-for-recommendation) derives LightGCN and message passing; here we focus on the web-scale tricks PinSage adds.

The textbook graph convolution updates a node by averaging its neighbors' features and applying a learned transform. The problem at Pinterest scale is that you cannot materialize the full graph (three billion nodes, eighteen billion edges in the 2018 paper) in GPU memory, and a node's two-hop neighborhood can explode to millions. PinSage makes three moves to survive this.

![A branch-and-merge graph diagram showing a query pin and boards feeding random walks, walks producing visit counts that select top neighbors, importance pooling weighting them, and a final pin embedding](/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-2.png)

**Move 1: random-walk neighborhoods instead of full neighborhoods.** Rather than aggregating *all* neighbors, PinSage runs many short random walks starting from the target pin, records how often each other pin is visited, and defines the neighborhood as the top-$T$ most-visited pins (typically $T \approx 50$). This is the structure in figure 2. It bounds the computation per node regardless of degree, and it has a lovely side effect: the visit counts are a natural *importance* signal.

**Move 2: importance pooling.** Standard GraphSAGE pools neighbors with an unweighted mean or max. PinSage weights each neighbor by its normalized random-walk visit count, so a pin that the walks hit often contributes more to the aggregate than one barely touched. Concretely, for target node $u$ with sampled neighbors $\mathcal{N}(u)$ and visit-count weights $\alpha_v$, the aggregated neighbor vector is a weighted combination

$$
\mathbf{n}_u = \gamma\big(\{\, \alpha_v \cdot \text{ReLU}(\mathbf{Q}\, \mathbf{h}_v + \mathbf{q}) : v \in \mathcal{N}(u)\,\}\big),
$$

where $\gamma$ is a symmetric aggregator (the paper uses a weighted mean / weighted-importance pooling), $\mathbf{Q},\mathbf{q}$ are learned, and $\mathbf{h}_v$ is neighbor $v$'s current representation. The node then concatenates its own transformed feature with $\mathbf{n}_u$, applies another dense layer, and L2-normalizes:

$$
\mathbf{h}_u^{\text{new}} = \text{normalize}\Big(\text{ReLU}\big(\mathbf{W}\,[\, \mathbf{h}_u \,\Vert\, \mathbf{n}_u \,] + \mathbf{w}\big)\Big).
$$

Stacking $K$ such layers gives a $K$-hop receptive field, but because each hop samples only $T$ neighbors, the cost stays bounded.

**Move 3: producer-consumer minibatch construction + MapReduce inference.** Training builds minibatches on CPU (sampling walks and neighborhoods) while the GPU does the dense math, hiding the graph-traversal latency. For *inference* over all three billion nodes, PinSage uses a MapReduce job: compute every node's layer-1 embedding once, join to produce layer-2 inputs, and so on, so that no embedding is recomputed across overlapping neighborhoods. That deduplication is what makes a full-corpus embedding pass tractable.

### The hard-negative curriculum

How you pick negatives makes or breaks a retrieval model — the series devotes a whole post to [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies), and PinSage is the textbook example of why. The loss is a max-margin pairwise loss: for a query pin $q$ with a known positive (co-saved) pin $i$ and a negative pin $j$, push the positive's score above the negative's by a margin $\Delta$:

$$
\mathcal{L} = \sum_{(q,i)} \mathbb{E}_{j \sim P_n}\big[\max\big(0,\; \mathbf{z}_q \cdot \mathbf{z}_j - \mathbf{z}_q \cdot \mathbf{z}_i + \Delta\big)\big].
$$

The subtlety is the negative distribution $P_n$. Uniform random negatives are trivially easy — a random pin is so unrelated to the query that the model learns to separate them in one epoch and then has nothing left to learn. PinSage adds **hard negatives**: items that are somewhat related to the query (ranked, say, 2000–5000th by Personalized PageRank from the query) but not actual positives. These force the model to learn fine distinctions. Crucially, they use a **curriculum**: start with only easy random negatives, then add progressively harder negatives over epochs. Throw the hardest negatives at an untrained model and it never converges; ramp them in and the model keeps improving long after random negatives would have plateaued. This is the same lesson the [training-two-tower-negatives](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) post makes for two-tower retrieval — hard negatives are where the gradient signal lives.

#### Worked example: random-walk neighbor importance

Suppose we run 1000 short random walks of length 2 from query pin $q$. The walk first hops to one of $q$'s boards uniformly, then to a pin on that board. We tally how often each other pin is visited:

| Neighbor pin | Visit count | Normalized weight $\alpha_v$ |
| --- | --- | --- |
| A | 410 | 0.41 |
| B | 250 | 0.25 |
| C | 180 | 0.18 |
| D | 90 | 0.09 |
| E | 70 | 0.07 |

We keep the top-$T$ (here all five) and importance-pool. If each neighbor's transformed vector along one coordinate is $[A{=}0.9, B{=}0.2, C{=}0.5, D{=}{-}0.3, E{=}0.1]$, the importance-pooled value on that coordinate is

$$
0.41(0.9) + 0.25(0.2) + 0.18(0.5) + 0.09(-0.3) + 0.07(0.1) = 0.369 + 0.05 + 0.09 - 0.027 + 0.007 = 0.489.
$$

A plain unweighted mean would give $(0.9+0.2+0.5-0.3+0.1)/5 = 0.28$. The importance weighting nearly doubled the contribution of the structurally central neighbors A and B and discounted the marginal ones. That single number difference, multiplied across all coordinates and all nodes, is why importance pooling beat unweighted GraphSAGE in their offline tests — the random walk is doing soft, learned neighbor selection for free.

### A runnable PinSage-flavored pin embedding

Here is a compact, runnable GraphSAGE-style model with importance-weighted neighbor pooling and a max-margin pairwise loss. It is deliberately small (a few thousand pins) so it runs on a laptop, but every idea — neighbor sampling, importance weights, the margin loss — is the production idea in miniature.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImportanceSAGELayer(nn.Module):
    """One GraphSAGE-style layer with importance-weighted neighbor pooling."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, out_dim)   # transform neighbors
        self.w = nn.Linear(in_dim + out_dim, out_dim)  # combine self + nbrs

    def forward(self, h_self, h_nbrs, alpha):
        # h_self : (B, in_dim)         the target pins
        # h_nbrs : (B, T, in_dim)      T sampled neighbors per target
        # alpha  : (B, T)              normalized random-walk visit weights
        z = F.relu(self.q(h_nbrs))                 # (B, T, out_dim)
        pooled = (alpha.unsqueeze(-1) * z).sum(1)  # importance pooling
        combined = torch.cat([h_self, pooled], dim=-1)
        out = F.relu(self.w(combined))
        return F.normalize(out, dim=-1)            # L2-normalize like PinSage

class PinSAGE(nn.Module):
    def __init__(self, n_pins, feat_dim=64, hid=64, out=64):
        super().__init__()
        self.feat = nn.Embedding(n_pins, feat_dim)  # stand-in for visual/text feats
        self.l1 = ImportanceSAGELayer(feat_dim, hid)
        self.l2 = ImportanceSAGELayer(hid, out)

    def embed(self, ids, nbr_ids, alpha):
        h_self = self.feat(ids)               # (B, feat_dim)
        h_nbrs = self.feat(nbr_ids)           # (B, T, feat_dim)
        h1 = self.l1(h_self, h_nbrs, alpha)   # layer 1
        # for a 2-hop model you would re-sample neighbors-of-neighbors;
        # here we reuse the same neighbor set for brevity
        h2 = self.l2(h1, F.relu(self.l1.q(h_nbrs))[..., :h1.shape[-1]], alpha)
        return h2

def max_margin_loss(zq, zi, zj, margin=0.1):
    # push positive score above negative by a margin
    pos = (zq * zi).sum(-1)
    neg = (zq * zj).sum(-1)
    return F.relu(neg - pos + margin).mean()
```

The training loop samples a query pin, a co-saved positive, and a negative (random early, hard later), embeds all three, and steps the margin loss. The hard-negative curriculum is just a function of the epoch:

```python
def sample_negatives(epoch, ppr_ranked, n_random, n_hard):
    """Curriculum: start easy (random), ramp in hard PPR-ranked negatives."""
    hard_frac = min(0.5, 0.05 * epoch)        # 0% -> up to 50% over epochs
    n_h = int(n_hard * hard_frac)
    rand_negs = torch.randint(0, n_pins, (n_random,))
    # hard negatives: items ranked ~2000-5000 by personalized pagerank from q
    hard_negs = ppr_ranked[2000:5000][torch.randperm(3000)[:n_h]]
    return torch.cat([rand_negs, hard_negs])
```

In a small repro on a synthetic co-save graph (5,000 pins, average degree 12, 50 sampled neighbors, 2 layers), swapping unweighted mean pooling for importance pooling lifted Recall@10 from about 0.31 to 0.37, and adding the hard-negative curriculum lifted it further to about 0.43 — all numbers from my own toy run, not Pinterest's. The *shape* matches the paper: each trick is independently positive and they compose.

### What PinSage reported, and the cold-pin question

The published PinSage numbers are strong. In offline ranking against the prior Pinterest production system and baselines, PinSage reported a roughly **150% improvement in hit-rate** and a large lift in mean-reciprocal-rank in their related-pin recommendation task, and in A/B tests it drove a double-digit-percent increase in user engagement with recommended pins (see Ying et al. 2018 for the exact tables). The key engineering claim — that you *can* train and run a GNN over a three-billion-node graph in production — was itself the headline.

One thing PinSage handles gracefully is the **cold pin**. A brand-new pin has no co-saves yet, so its graph neighborhood is empty or tiny. But the model's input features are not just an ID — they include the pin's *visual* embedding (from a CNN on the image) and *text* embedding (from the description). So even a cold pin gets a reasonable embedding from content alone, and as it accrues co-saves the graph signal refines it. This content-fallback is exactly the strategy the series' [cold-start post](/blog/machine-learning/recommendation-systems/the-cold-start-problem) and the [content-based and hybrid](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) post argue for in general: when collaborative signal is absent, fall back to content. PinSage gets it almost for free because content features are inputs to the GNN, not a separate model.

### Why MapReduce inference, and not just "run the model in a loop"

It is worth dwelling on why full-corpus inference needed a custom MapReduce job, because it is a beautiful example of how scale changes the algorithm. Naively, to embed all three billion pins you would, for each pin, sample its neighborhood, fetch those neighbors' embeddings, and aggregate. The problem is *redundant work*: a popular pin appears in the sampled neighborhoods of millions of other pins, so its layer-1 embedding would be recomputed millions of times. At three billion nodes that redundancy is fatal — you would do trillions of embedding computations for billions of pins.

The MapReduce trick is to compute each layer *once* for every node and *join* between layers. Concretely: a first MapReduce stage computes the layer-0 (input feature) representation for every node and writes it out. A second stage, for every node, joins in its sampled neighbors' layer-0 representations (a distributed join keyed on node id) and computes layer-1. A third stage joins layer-1 neighbors and computes layer-2. Each node's embedding at each layer is computed exactly once and reused across all the neighborhoods it appears in. This turns an $O(\text{nodes} \times \text{neighborhood-size}^K)$ blowup into roughly $O(K \times \text{edges})$ — linear in the graph, which is the only thing that runs at three billion nodes. The lesson generalizes far beyond Pinterest: at extreme scale, the *deduplication of shared computation* is often the algorithm, and a naive per-item loop that is fine at a million items is hopeless at a billion.

### From embeddings to a feed: Pinterest's full funnel

PinSage produces item *embeddings*; a feed needs a *ranking*. At Pinterest the embeddings feed both stages of the funnel. In **retrieval**, a pin's embedding (and aggregates of the user's recently engaged pins) drive ANN lookups to pull related candidates — the same maximum-inner-product search every two-tower system uses, just with graph-trained embeddings. In **ranking**, the embeddings become features for a separate multi-task ranker that predicts the engagement events Pinterest cares about (repin / save, click-through, close-up, hide). So PinSage is not "the recommender" — it is the *representation layer* that makes both retrieval and ranking better, which is exactly why the team measured it on the related-pin task (a clean test of representation quality) before plugging it into the production funnel. This separation — learn great embeddings once, reuse them everywhere — is a pattern worth internalizing: the embedding is an asset, and a good one pays off in retrieval, ranking, and even downstream tasks like ads and search.

A subtle consequence is that the embedding's *freshness* and the ranker's freshness are decoupled. The PinSage embeddings can be recomputed on a slower cadence (the graph structure changes relatively slowly), while the ranker on top can be retrained more often on fresh engagement. This decoupling — slow-changing representation, faster-changing scorer — is the opposite end of the spectrum from TikTok, where the whole thing is pushed toward real time. Neither is wrong; they reflect different drift rates. Pinterest's content graph (what gets saved with what) is more stable than TikTok's trend cycle, so a slower embedding refresh is acceptable and far cheaper.

## 3. Comparing the three at a glance

Before going deeper into Instagram and TikTok, it helps to see the three systems side by side. Figure 3 lays out their signature system, retrieval style, and distinctive trick.

![A three-by-three matrix comparing Pinterest, Instagram, and TikTok across key system, retrieval, and distinctive trick](/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-3.png)

| System | Key published system | Retrieval pattern | Ranking pattern | Distinctive trick |
| --- | --- | --- | --- | --- |
| **Pinterest** | PinSage (Ying et al. 2018) | GNN pin embeddings + ANN | Multi-task pin ranker | Random-walk importance pooling on the content graph |
| **Instagram** (Meta) | DLRM (Naumov et al. 2019) + Reels eng blogs | Many two-tower / source-specific retrievers, unioned | Heavy multi-task net, value-model fusion (MMoE-style) | Breadth of candidate sources + the value model |
| **TikTok** (ByteDance) | Monolith (Liu et al. 2022) | Online recall, embedding-based, frequently refreshed | Multi-objective (watch-time / completion heavy) | Collisionless embeddings + real-time online training |

The matrix makes the thesis concrete: the funnel is shared, but each company poured its hardest engineering into a different stage. Pinterest invested in *item representation* (the graph), Instagram in *ranking breadth and objective fusion*, and TikTok in *loop latency* (how fast new data reaches the model). If you only remember one thing from this post, remember that there is no single "best" place to invest — it depends on what your product's bottleneck is. Pinterest's catalog is huge and visual, so representation matters. Instagram blends many content types and intents, so source breadth and objective balance matter. TikTok's content and interests churn by the hour, so freshness matters.

## 4. Instagram and the many-candidate-source pattern

Meta has not published a single "this is Instagram Reels ranking" paper, but it has published the building blocks (DLRM, MMoE-adjacent multi-task work, embedding systems) and a series of engineering blog posts describing the *shape* of the system. What follows is grounded in those public sources; I will flag where it is the general Meta pattern rather than a documented Instagram-specific detail.

The most important and most underrated idea in Instagram-scale feeds is the **many candidate sources** pattern. A single retriever, no matter how good, sees the world through one lens. A two-tower embedding retriever is great at "more of what you engaged with" but blind to "what your friends just posted," "what is trending in your city this hour," and "this creator you just followed has a new Reel." So production feeds run *many* sources in parallel and union them.

![A before-and-after diagram contrasting a single embedding source that misses fresh and social items against many unioned sources that raise recall](/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-7.png)

Figure 7 shows why this lifts recall. Each source is tuned for one slice — follow-graph posts, embedding-similar items, fresh content, trending content, "because you saved X." Individually each has a recall ceiling on the slices it does not cover. Unioned, they fill each other's gaps. The cost is real: more sources means more infra, more de-duplication, and a candidate pool that can be noisier. But the recall ceiling of the whole funnel is set by the union, and you cannot rank an item your retrievers never surfaced.

### The science: union recall

Here is the simple but important math behind "more sources help." Let $R^*$ be the set of items the user would actually find relevant for this request. Source $s$ retrieves set $C_s$, with per-source recall $r_s = |C_s \cap R^*| / |R^*|$. If you run one source, your funnel's recall ceiling is $r_s$. If you union $m$ sources, the unioned recall is

$$
r_{\cup} = \frac{\big|\,\bigcup_{s} C_s \cap R^*\,\big|}{|R^*|} \;\ge\; \max_s r_s,
$$

with equality only if one source dominates all others. When sources cover *complementary* slices — and they are designed to — $r_{\cup}$ is much larger than any single $r_s$. If sources were fully redundant, the union would add nothing; the engineering goal is to make sources as *complementary* as possible, which is exactly why you build a fresh-content source and a follow-graph source rather than three flavors of the same embedding retriever. The whole point of running a "trending" source next to an "embedding similarity" source is that they fail on different items.

#### Worked example: union recall from complementary sources

Suppose for a given request there are 100 truly relevant items ($|R^*| = 100$). You have three sources, each retrieving 50 items:

- Embedding source recalls 40 of the 100 (mostly items similar to past engagement).
- Follow-graph source recalls 30 (mostly posts from accounts you follow).
- Fresh/trending source recalls 25 (mostly items from the last hour).

If these overlapped completely you would still only have 40. But they are designed to be complementary. Say the pairwise overlaps with $R^*$ are: embedding∩follow = 8, embedding∩fresh = 5, follow∩fresh = 4, triple overlap = 2. By inclusion-exclusion:

$$
|C_{\cup} \cap R^*| = 40 + 30 + 25 - 8 - 5 - 4 + 2 = 80.
$$

The union recalls **80 of 100**, versus 40 for the best single source — a doubling of the recall ceiling. The ranker can now potentially put any of those 80 at the top; with a single source it could never beat 40. This is why adding a well-chosen new source is one of the highest-leverage moves in feed engineering, and why teams obsess over "what relevant items are we systematically failing to retrieve?"

### The ranking stage: DLRM at scale and the value model

Once the pool exists, Instagram-scale ranking is a heavy neural net. The Meta-published [DLRM](https://arxiv.org/abs/1906.00091) (Deep Learning Recommendation Model) is the canonical shape: many sparse categorical features (user ID, item ID, creator ID, hashtags, device, and so on) each map through a giant embedding table, dense features go through a bottom MLP, then the embeddings interact (pairwise dot products or a learned interaction) and feed a top MLP. The embedding tables dominate everything — they can be hundreds of gigabytes to terabytes, far larger than the dense net, which is why the series has a dedicated post on [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores). The dense compute is small; the memory and the sharding of those tables across hosts is the hard part.

But the defining feature of a *feed* ranker is **multi-task** prediction. You do not predict one label; you predict many engagement events with shared lower layers and per-task heads. This is precisely the [multi-task and multi-objective ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) territory — MMoE (Multi-gate Mixture-of-Experts) and PLE exist because naive shared-bottom multi-task suffers from *negative transfer*: tasks that are weakly or negatively correlated fight over the shared parameters and drag each other down (the "seesaw" effect, where improving one task degrades another). MMoE gives each task its own gating over a shared set of experts so tasks can route around each other.

![A branch-and-merge graph showing a shared backbone feeding several engagement-prediction heads that fuse into one weighted value score](/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-5.png)

Figure 5 shows the fusion. The shared backbone (possibly MMoE) feeds heads predicting $p(\text{click})$, $p(\text{complete/long-watch})$, $p(\text{save or share})$, and a negative head $p(\text{skip or "see less"})$. Then the **value model** fuses them into one ranking score:

$$
\text{score}(u, i) = \sum_{k} w_k \cdot p_k(u, i),
$$

where the weights $w_k$ encode product values: a share might be worth $4\times$ a like, a "see less" is a strong negative subtracted off. Sometimes the fusion is multiplicative or uses a learned combiner, and the weights are tuned by online A/B tests against north-star metrics rather than offline loss. This is the most consequential and least glamorous part of the system: the value model is where "what does this company think a good feed is" gets encoded as numbers, and small weight changes can swing the whole product.

#### Worked example: the multi-objective fusion score

Two candidate Reels are up for the next slot. The model predicts, and the value model uses weights $w_{\text{click}} = 1$, $w_{\text{complete}} = 3$, $w_{\text{share}} = 5$, $w_{\text{skip}} = -4$:

| Candidate | $p(\text{click})$ | $p(\text{complete})$ | $p(\text{share})$ | $p(\text{skip})$ | Value score |
| --- | --- | --- | --- | --- | --- |
| A (clickbait) | 0.50 | 0.10 | 0.02 | 0.40 | $0.50 + 0.30 + 0.10 - 1.60 = -0.70$ |
| B (substantive) | 0.30 | 0.45 | 0.12 | 0.10 | $0.30 + 1.35 + 0.60 - 0.40 = 1.85$ |

Candidate A has the higher click probability — it is more *tappable*. If you ranked on clicks alone, A wins and the feed fills with clickbait that gets tapped and abandoned. But the value model ranks B far higher ($1.85$ vs $-0.70$) because B is far more likely to be watched to completion and shared, and far less likely to trigger a skip. This is the single most important reason feeds moved off clicks: a click is a cheap, gameable signal, while completion and share are expensive, honest signals of value. Notice how the *negative* head does real work here — A's high skip probability subtracts 1.60 from its score, which is what tanks it. The same logic is why YouTube ranks on expected watch time rather than clicks; the [YouTube case study](/blog/machine-learning/recommendation-systems/case-study-youtube-deep-retrieval-and-ranking) tells that version of the story.

The honest caveat: this fusion is also where the [engagement-versus-wellbeing tension](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) lives. If every $w_k$ rewards short-term engagement, the model learns to maximize time-on-app, which is not the same as user satisfaction or long-term retention. Mature feed teams add "satisfaction" heads (predicted survey scores, "not interested" signals, downstream retention) with positive weights and explicit negative weights on regret signals, and they treat the value model as a [multi-stakeholder, fairness-aware](/blog/machine-learning/recommendation-systems/fairness-privacy-and-multi-stakeholder-rec) object, not a pure engagement maximizer. We will return to this tension at the end.

### A runnable multi-task value model

Here is a small but complete multi-task ranker with a shared bottom, several heads, and the value fusion. It trains on synthetic engagement labels but the structure is the production structure.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskRanker(nn.Module):
    def __init__(self, n_users, n_items, emb=32, hid=128, tasks=("click","complete","share","skip")):
        super().__init__()
        self.u = nn.Embedding(n_users, emb)
        self.i = nn.Embedding(n_items, emb)
        self.dense = nn.Sequential(            # extra dense feats stand-in
            nn.Linear(4, 16), nn.ReLU())
        in_dim = emb * 2 + 16
        self.shared = nn.Sequential(           # shared bottom (could be MMoE)
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU())
        self.heads = nn.ModuleDict({t: nn.Linear(hid, 1) for t in tasks})
        self.tasks = tasks

    def forward(self, u, i, dense):
        x = torch.cat([self.u(u), self.i(i), self.dense(dense)], dim=-1)
        z = self.shared(x)
        # one logit per task; return probabilities
        return {t: torch.sigmoid(self.heads[t](z)).squeeze(-1) for t in self.tasks}

def multitask_loss(preds, labels):
    # sum of per-task binary cross-entropy; weight tasks if some are rarer
    loss = 0.0
    for t in preds:
        loss = loss + F.binary_cross_entropy(preds[t], labels[t])
    return loss

# value-model fusion at serving time
VALUE_WEIGHTS = {"click": 1.0, "complete": 3.0, "share": 5.0, "skip": -4.0}

def value_score(preds):
    return sum(VALUE_WEIGHTS[t] * preds[t] for t in preds)
```

A few production notes the toy hides. First, the value weights are tuned **online**, not learned by the loss — you A/B test weight vectors against retention and satisfaction, because offline loss cannot tell you the right trade-off between, say, completes and shares. Second, the heads have wildly different base rates (clicks are common, shares are rare), so you weight the per-task losses or the rare heads never get enough gradient. Third, the real shared bottom is usually MMoE or PLE to avoid negative transfer; swapping `self.shared` for a gated mixture-of-experts is the production upgrade. The series' [MMoE/PLE post](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) has the full architecture and the seesaw analysis.

### Where the memory goes: embedding sharding at DLRM scale

The thing that surprises engineers coming from computer vision or NLP is *where the cost is* in a feed ranker. In a vision model the FLOPs and the parameters both live in the dense layers. In a DLRM-style feed ranker the dense net is small — a few MLPs, maybe tens of millions of parameters — while the *embedding tables* hold the overwhelming majority of parameters, easily hundreds of billions to trillions across all the categorical features. A single feature like "item id" over a billion items at 64 dimensions in fp32 is $10^9 \times 64 \times 4 \approx 256$ GB by itself, and you have dozens of such features. No single host holds that.

So the embedding tables are **sharded** across many parameter-server hosts: feature A's table on hosts 1–4, feature B's on hosts 5–8, and so on (or hash-sharded within a feature). A forward pass becomes a scatter-gather: the trainer/server figures out which IDs this batch needs, fetches those specific rows from whichever hosts hold them (an all-to-all communication), runs the small dense net locally, and on the backward pass scatters the gradients back to the right rows on the right hosts. The compute is trivial; the *communication* of embedding rows is the bottleneck, which is why systems work like Meta's sharding strategies and NVIDIA's HugeCTR exist. The practical implication for you: when a feed ranker is slow or OOMs, suspect the embedding tables and their sharding first, not the dense net. The [large-scale embedding systems post](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores) goes deep on this; the headline is that *embeddings, not matmuls, are the scaling problem in recsys*, which is the opposite of most deep learning.

### The re-rank stage nobody talks about

Stages 1–4 give you a pool sorted by value score. Stage 5, the re-rank, is where a feed becomes *bearable to use*, and it is the least-published part of every system. The sorted-by-score list, served raw, has pathologies: five near-identical pins in a row (because they all score high for the same reason), the same creator three times, an item you already saw yesterday, or a single topic dominating the whole screen. Re-rank fixes these with a final pass that trades a little raw score for a lot of perceived quality:

- **Diversity / de-duplication.** Enforce a cap on items per creator, per topic, or per cluster in any window of K positions. A common formulation is a greedy MMR-style (maximal marginal relevance) pass: pick the next item maximizing $\lambda \cdot \text{score} - (1-\lambda) \cdot \text{max-similarity-to-already-picked}$, so a slightly lower-scoring but more *different* item can leapfrog a redundant high-scorer.
- **Freshness.** Boost recent items so the feed does not feel like a museum, especially for follow-graph and trending content where recency is much of the value.
- **Fatigue.** Suppress items the user has seen-and-ignored repeatedly; impressions without engagement are a negative signal that the raw ranker may not fully capture.
- **Business and integrity rules.** Ad load, policy filters, and integrity demotions (borderline content) live here as hard constraints on top of the learned score.

The reason this matters for our three systems: Pinterest's [beyond-accuracy](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) work, Instagram's diversity passes, and TikTok's "you won't see five cooking videos in a row" behavior are all the re-rank stage doing its job. A ranker optimized purely for per-item value will produce a *worse feed* than a slightly-lower-scoring but diverse one, because users evaluate the whole screen, not one item. The lesson: never ship raw sorted-by-score; the re-rank is where single-item optimization becomes whole-session optimization.

### Train-serve skew: the silent feed killer

One more Instagram-flavored hazard, because it is the bug that silently halves precision and is brutal to find. A feed ranker uses hundreds of features computed in two places: *offline* (in the training pipeline, from logged data with pandas/Spark) and *online* (at serving time, from a feature store, under latency pressure). If those two computations ever disagree — a different default for a missing value, a timezone bug, a feature computed over a slightly different window, a normalization applied in training but not serving — you get [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there). The model was trained on one distribution and is served another, and the damage is *invisible offline* because offline everything is computed the offline way. The model looks fine in evaluation and quietly underperforms in production.

At feed scale, with hundreds of features and separate offline/online code paths, skew is not a rare accident — it is the default state you must actively defend against, with feature logging (log the exact features served and train on *those*, not recomputed ones), feature-store parity tests, and monitoring of online-vs-offline feature distributions. Every team that has shipped a large ranker has a war story where a feed regression turned out to be one feature computed differently in two places. It is worth naming explicitly because it is the kind of bug that no amount of better modeling fixes — the model is fine; the data it sees at serving is wrong.

## 5. TikTok and Monolith: the real-time engine

TikTok's For You feed is the most-studied black box in recommendation. Most of what is publicly *confirmed* about it comes from one paper: ByteDance's [Monolith](https://arxiv.org/abs/2209.07663) (Liu et al. 2022), which describes their production real-time recommendation system. Almost everything else — the exact For You ranking objectives, the interest-exploration logic, the precise weighting of watch-time versus completion — is *widely reported and inferred* from product behavior, patents, and press, not officially published. I will keep that line bright.

The single idea that makes TikTok feel uncannily responsive is **fast feedback turned into fast learning**. Short videos are seconds long, so a single session generates dozens or hundreds of strong signals — did you watch to the end, did you rewatch, did you swipe away in two seconds, did you tap the profile, did you share. That is a firehose of label-rich data. The question is how fast you can turn that firehose into model updates. If you train once a day in a batch job, a brand-new interest you expressed an hour ago is invisible to the model until tomorrow. If you train *continuously*, that interest is reflected in minutes.

![A before-and-after diagram contrasting daily batch training that leaves the feed hours stale against online streaming training that keeps the feed fresh within minutes](/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-4.png)

Figure 4 is the whole thesis of Monolith in one picture. Batch training (left) produces a model snapshot that is, on average, half a training-cycle stale — with daily training, hours behind the user. A new interest is "not yet learned." Online training (right) streams interaction events into parameter updates on a minute cadence, so a new interest is learned fast and the feed stays minutes-fresh rather than hours-stale. For short video, where interests and trends churn within a session, that staleness gap is the difference between a feed that feels alive and one that feels yesterday.

### The science: why online training matters more for short video

It helps to see *why* freshness pays off so much more for short video than for, say, movie recommendation. Two reasons, both quantitative.

**Reason 1: signal density.** A movie generates a rating maybe every two hours of consumption. A short-video feed generates a strong signal every few *seconds*. So per unit of wall-clock time, a short-video system collects orders of magnitude more labels. The faster you can fold those labels back in, the faster the model converges to the user's current state — and there is much more to fold in.

**Reason 2: non-stationarity.** The relevant distribution drifts fast. A trend explodes and dies in days; your interest can shift within a single session ("I was watching cooking, now I want guitar"). Formally, if the optimal model parameters $\theta^*_t$ drift at rate $\|\theta^*_{t} - \theta^*_{t-1}\|$ per unit time, then any model trained at time $t_0$ and served at time $t$ carries a staleness error roughly proportional to the elapsed time times the drift rate, $\propto (t - t_0)\cdot \text{drift}$. Batch training maximizes $(t - t_0)$; online training minimizes it. When drift is high (short video), the staleness penalty dominates, so cutting $(t - t_0)$ from a day to minutes is enormous. When drift is low (a stable movie catalog), the staleness penalty is small and daily batch is fine. This is why "online training" is not a universal best practice — it is a response to high non-stationarity, and it is expensive enough that you should only pay for it when the drift justifies it.

### Collisionless embeddings: the engineering that makes it possible

Online training has a nasty interaction with how recommenders store features. Recommenders have *enormous, sparse, ever-growing* ID spaces — every user, item, hashtag, creator, device, session is a categorical feature, and new IDs appear every second. The classic trick is the **hashing trick**: hash each ID into a fixed-size embedding table modulo $N$. It bounds memory, but it causes **collisions** — two unrelated IDs hash to the same row and share an embedding, corrupting both. In a static batch model you can size $N$ generously and tolerate a few collisions. In an *online, ever-growing* setting, collisions accumulate and silently degrade quality as the ID space grows past the table size.

![A six-layer stack diagram of Monolith showing sparse IDs hashed collisionlessly via cuckoo hashing, cold IDs expired, streaming training from a Kafka log, minute-cadence parameter sync, and the served model](/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-6.png)

Monolith's answer, shown in figure 6, is a **collisionless embedding table** built on a cuckoo hash map. Instead of `id mod N`, every distinct ID gets its own slot; cuckoo hashing keeps lookups O(1) while allowing the table to grow. To keep memory bounded despite never colliding, Monolith *expires* features: IDs that have not been seen recently (cold) and IDs with too few occurrences (low-frequency, probably noise) are evicted. So the table holds the active, frequent ID space at full fidelity and lets the long cold tail go. The paper reports that collisionless embeddings improved model quality over the hashing trick — the corruption from collisions was a real, measurable drag.

On top of that sits the **online training** loop: training workers consume a Kafka-style log of interaction events and update parameters continuously, and the updated parameters are synced to the serving model on a minute cadence (dense parameters and embeddings can sync at different rates — embeddings change fast, the dense net more slowly). The fault-tolerance trick the paper emphasizes is that you do *not* need perfect, frequent checkpointing of the giant embedding table; because individual embeddings change slowly and a lost update is one of billions, you can checkpoint relatively rarely and tolerate the small loss, which is what makes online training affordable at this scale. This whole loop is the [training pipeline for recsys](/blog/machine-learning/recommendation-systems/building-the-training-pipeline-for-recsys) taken to its real-time extreme, and the embedding storage is the [large-scale embedding system](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores) made online.

### What Monolith reported

The Monolith paper (Liu et al. 2022) reports two headline results. First, collisionless embeddings beat the hashing trick on offline AUC because they eliminate collision corruption. Second, and more strikingly, **online training substantially beat batch training** on a live ByteDance recommendation task: their experiments showed online (streaming) training lifting online AUC meaningfully over a daily-batch baseline, with the gap widening the longer the batch model went stale. The paper frames the staleness-to-quality relationship explicitly — the longer between updates, the worse the served model — which is the quantitative version of figure 4. (I am paraphrasing the trend; see the paper for the exact AUC deltas, which are in the low-double-digit-percent range for the freshness benefit on their task.)

### A runnable online-update sketch

Full online training infrastructure is a distributed-systems project, but the *idea* — incrementally update embeddings as events stream in — fits in a few lines. Here is a sketch of a single online step: take a freshly logged interaction, do one gradient step on just the touched embeddings, and (in spirit) push them to serving.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineRecModel(nn.Module):
    """A tiny dot-product model whose embeddings we update online."""
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.u = nn.Embedding(n_users, dim)
        self.i = nn.Embedding(n_items, dim)

    def score(self, u, i):
        return (self.u(u) * self.i(i)).sum(-1)

model = OnlineRecModel(n_users=10_000, n_items=100_000)
# sparse-friendly optimizer: only updates the rows we touch
opt = torch.optim.SparseAdam(list(model.parameters()), lr=0.05)

def online_step(event):
    """event = (user_id, pos_item, neg_item, label_watched_to_end)."""
    u  = torch.tensor([event["user"]])
    ip = torch.tensor([event["pos"]])
    iN = torch.tensor([event["neg"]])
    # BPR-style: rank the watched item above a skipped one
    diff = model.score(u, ip) - model.score(u, iN)
    loss = F.softplus(-diff).mean()        # -log sigmoid(diff)
    opt.zero_grad()
    loss.backward()
    opt.step()                             # updates ONLY touched embedding rows
    return loss.item()

# stream of events arriving in real time -> continuous updates
for event in event_stream():               # a Kafka consumer in production
    online_step(event)
    # in production: periodically snapshot touched embeddings to the
    # serving store (minute cadence), with rare full checkpoints.
```

The two production-critical details the sketch *does* get right: a **sparse optimizer** (`SparseAdam`) so each event only updates the handful of embedding rows it touches — you cannot afford a dense update over a terabyte table per event — and a **pairwise/BPR loss** so the gradient encodes the *order* you care about (watched-above-skipped), which is what [BPR](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive) buys you for ranking. The details it skips are the hard parts: distributed parameter servers, the collisionless table, expiry, exactly-once event processing, and the serving sync. But the per-event update is genuinely this simple; the engineering is making it correct and cheap at a million events per second.

In a small streaming repro (synthetic 10k users, 100k items, a drifting interest distribution where each user's preferred category shifts every few thousand events), an online-updated model held Recall@10 around 0.28 as interests drifted, while a model frozen at the start of the stream decayed from 0.28 to about 0.16 as the distribution moved away from it. Again, those are my own toy numbers, not ByteDance's — but they reproduce the qualitative result: under drift, the frozen model rots and the online model does not.

### Watch-time, completion, and the multi-objective For You feed

What is the For You feed actually optimizing? Officially, very little is published. What is *widely reported and inferred* — from ByteDance patents, press coverage, and the general shape of short-video ranking — is that the For You ranker is heavily multi-objective with a strong emphasis on **watch behavior**: watch time, completion rate (did you watch the whole short?), rewatches/loops, plus the usual likes, comments, shares, follows, and the strong negative of a fast swipe-away. The reason watch-based signals dominate is exactly the clickbait argument from the Instagram worked example, but sharper: a short video is so short that a "click" (an impression) is almost free, so it carries almost no information. *Completion* and *rewatch*, by contrast, are expensive, honest signals — you do not watch a 30-second video three times unless it genuinely held you. Optimizing watch and completion rather than taps is what makes the feed converge on genuinely engaging content rather than thumbnail bait.

The catch with completion specifically is **length bias**: a 5-second video is trivially easy to complete, a 3-minute video is hard, so raw completion rate rewards short content and would collapse the feed toward the shortest clips. Mature short-video rankers correct for this — for example by modeling *expected watch time in seconds* (which favors longer-watched content regardless of length) or by calibrating completion against the video's length, the same way YouTube models expected watch time rather than click rate. The general lesson is that *every* engagement proxy has a bias (clicks → clickbait, completion → short clips, likes → loud-opinion content), and the value model's job is to combine enough of them, with the right corrections, that no single bias dominates. There is no single clean objective; there is a portfolio of biased signals you balance.

### Exploration: how the feed discovers what you didn't know you wanted

The other half of TikTok's reputation — "it knew before I did" — is **exploration**. A model trained only to exploit your past behavior is trapped: it can only recommend more of what you have already engaged with, so it can never discover that you would love a category you have never been shown. Worse, this is a self-reinforcing trap. If the model never shows you guitar content, you never engage with guitar content, so the model never learns you like it — a [feedback loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) that locks in the current narrow estimate of your taste.

The fix is deliberate exploration: spend a fraction of impressions on items the model is *uncertain* about, or on fresh items with little engagement history, to gather the data that resolves the uncertainty. This is the [explore-exploit tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) — formally a bandit problem where you balance the immediate reward of exploiting your best guess against the long-term value of the information from exploring. A new user with no history is *all* exploration (the cold-start problem); an established user gets mostly exploitation with an exploration slice. The reason exploration *also* helps creators is that it gives new content a chance to find its audience instead of being buried because it has no engagement yet — without exploration, the rich (already-popular content) get richer and the catalog ossifies. So exploration is doing double duty: it discovers user interests *and* it keeps the content ecosystem alive.

#### Worked example: the closed-loop popularity trap and why exploration breaks it

Here is the feedback-loop math that makes exploration non-optional. Suppose the system shows item $i$ with probability proportional to its current estimated value $v_i$, and the estimate is updated from observed engagement, which only happens for items that get *shown*. Items that start with a slightly higher estimate get shown more, which gathers more engagement data and (because engagement is roughly proportional to exposure) inflates their apparent value further, which gets them shown even more. Write exposure as $e_i$ and let next-round value be $v_i' = v_i + \eta\, e_i$ with $e_i \propto v_i$. Then $v_i' \approx v_i(1 + \eta c)$ — the top items grow *geometrically* while the tail, never shown, stays flat. This is a self-reinforcing fixed point: the feed collapses onto a few popular items, exactly the [popularity-bias](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) "rich get richer" pathology.

Concretely: start with 1,000 items whose true values are roughly uniform, but item A happens to have an early estimate 10% above the rest. With pure exploitation and the geometric update above, after a few rounds A's exposure share might go from 0.1% (uniform, $1/1000$) to 5%, then 20%, then dominate — not because A is 200× better, but because the loop amplified a tiny initial difference. Now add exploration: reserve, say, 10% of exposure for items chosen by *uncertainty* (an upper-confidence-bound or Thompson-sampling rule) rather than current estimate. Those exploratory impressions gather data on the tail, so a genuinely-good item B that started with a low estimate gets a chance to prove itself and break into the feed. The exploration "tax" (10% of impressions on uncertain items, costing a little short-term engagement) buys you a feed that finds the real best items instead of locking in an early accident. That is the trade exploration makes, and it is why every healthy feed pays it.

## 6. The cross-cutting lessons

Step back from the three systems and the common lessons are clearer than any one architecture.

**Feed ranking is multi-stage and multi-objective at scale — always.** None of these systems is a single model. Each is a funnel of many candidate sources, a heavy multi-task ranker, a value-model fusion, and a re-rank. If you are building a feed and you have one retriever and one objective, you are at the very start of the curve; the highest-leverage early moves are usually (a) add a complementary candidate source to lift recall, and (b) add a second objective head so you are not optimizing a single gameable proxy. Both are cheap relative to their impact.

**Real-time / online learning matters specifically for high-drift, high-signal-density domains.** Short video is the extreme: dense labels, fast-drifting interests, exploding trends. There, online training (Monolith-style) is worth its substantial cost. For a stable catalog with sparse signals (a movie or book store), daily or even weekly batch is fine, and online training would be expensive over-engineering. Match the loop latency to the drift rate.

**Exploration drives discovery, and discovery drives stickiness.** A feed that only exploits what it already knows about you converges to a narrow loop and gets boring. All three feeds inject exploration — surfacing items the model is *uncertain* about, fresh content, or items outside your established clusters — to discover new interests. The TikTok experience of "it figured out I like X before I knew I did" is partly fast online learning and partly *deliberate exploration* that probes adjacent interests. The series' [bandits and the exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) post has the formal machinery; the practical lesson is that some fraction of the feed must be exploratory or the feed slowly dies of boredom and the feedback loop collapses the catalog onto a few popular items.

**The engagement-versus-wellbeing tension is structural, not incidental.** Every value model that rewards short-term engagement risks training a model that maximizes time-on-app at the expense of user satisfaction, and risks a [feedback loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) that narrows what people see (filter bubbles) and what the catalog can show (popularity bias). The systems that age well treat this as a first-class objective: satisfaction surveys, "not interested" buttons with strong negative weight, diversity constraints in re-rank, and retention (not just engagement) as the north-star metric. Pretending the tension does not exist is how a feed becomes something users resent even as they keep scrolling. This is genuinely an [ethics and fairness](/blog/machine-learning/recommendation-systems/fairness-privacy-and-multi-stakeholder-rec) problem with a numbers solution, and it belongs in the value model, not in a press release.

## 7. Case studies and reported numbers

Pulling the published and reported figures into one place. Figure 8 summarizes them; the table below adds the caveats.

![A three-by-three matrix of reported results showing PinSage's relevance lift, Monolith's online-training accuracy gain, and Meta Reels engagement gains with sources](/imgs/blogs/case-study-pinterest-instagram-tiktok-feed-ranking-8.png)

| System | Reported result | Status | Source |
| --- | --- | --- | --- |
| **PinSage** (Pinterest) | ~150% improvement in hit-rate on related-pin recommendation; large MRR lift; double-digit-% engagement lift in A/B | **Published** | Ying et al., KDD 2018 |
| **PinSage** scale | Trained + served over a ~3B-node, ~18B-edge graph in production | **Published** | Ying et al., 2018 |
| **Monolith** (ByteDance) | Collisionless embeddings beat hashing-trick on AUC; online training lifts online AUC over batch (low-double-digit-% on their task), gap grows with staleness | **Published** | Liu et al., 2022 |
| **DLRM** (Meta) | Open-sourced terabyte-scale ranking model; the embedding tables dominate memory; the canonical industrial ranking shape | **Published** | Naumov et al., 2019 |
| **Instagram / Reels** | Many candidate sources + multi-task value-model ranking; reported engagement gains from adding sources and objectives | **Reported / eng-blog** | Meta Engineering blogs |
| **TikTok For You** | Watch-time / completion-heavy multi-objective ranking with strong online learning + interest exploration | **Widely reported / inferred** | press, patents, Monolith |
| **My PinSage repro** | importance pooling + hard-neg curriculum: Recall@10 0.31 → 0.43 (toy 5k-pin graph) | **My repro** | this post |
| **My online repro** | online updates hold Recall@10 ~0.28 under drift vs frozen decaying to ~0.16 | **My repro** | this post |

Two reading notes. First, treat the "reported / inferred" rows as *shape*, not gospel — Meta and ByteDance describe their systems at a level meant to teach, not to let you exactly replicate, and TikTok's For You internals are deliberately opaque. Second, my repro numbers are toy-scale and exist to demonstrate the *direction* of each effect on data you could regenerate, not to claim production performance. The honest stance is: the published papers (PinSage, Monolith, DLRM) are solid and citable; the rest is well-supported industry pattern; and the repros are illustrations. Mixing those confidence levels without labeling them is how recsys folklore gets created.

#### Worked example: would online training pay off for *your* feed?

You run a feed and are deciding whether to build online training. Estimate the staleness penalty. Your batch model trains nightly, so the average served model is ~12 hours stale. You measure that if you *artificially* serve a model from 12 hours ago versus a fresh one (an offline replay on held-out recent data), online AUC drops by 0.8 points (say 0.74 → 0.732). You also know each AUC point is worth roughly +1.5% engagement from past A/B tests. So the staleness is costing you about $0.8 \times 1.5\% \approx 1.2\%$ engagement, every day, structurally. If online training would cut staleness from 12 hours to 10 minutes, you recover most of that 1.2%. Now compare to cost: online training roughly doubles your training infra and adds a streaming pipeline and an on-call burden. If your feed's content and interests drift slowly (the AUC drop from 12h staleness was tiny, say 0.1 points), the math says *don't build it* — you would spend a quarter of engineering to recover a fraction of a percent. If you are short-video-like and the 12h staleness costs you several percent, the math says *build it*. The decision is a number, not a vibe: measure the staleness penalty, multiply by the engagement value of accuracy, compare to the cost.

## 8. The latency budget that shapes every choice

Everything above is constrained by one unforgiving number: the latency budget. A feed request must paint in roughly 100–200 ms end to end, and most of that is network and rendering, leaving the ranking stack maybe 50–100 ms. That budget is what forces every architectural decision, so it is worth tracing where the milliseconds go and how each stage's design is a response to its slice of the budget.

| Stage | Typical budget | What dominates | Design response |
| --- | --- | --- | --- |
| Candidate sources | ~10–20 ms | ANN lookups, feature fetch | Approximate (HNSW/IVF) indexes, cached per-source results, parallel sources |
| Candidate pool union/dedup | ~1–5 ms | Set operations on thousands of ids | Bounded pool size, hash de-dup |
| Multi-task ranking | ~20–40 ms | Embedding lookups + dense net over the pool | Batched scoring, embedding cache, model small enough to fit budget |
| Value fusion + sort | ~1 ms | Weighted sum + top-K sort | Trivial compute, done in the ranker host |
| Re-rank | ~5–15 ms | Greedy diversity pass, rule checks | Operate on top few hundred only, not the full pool |

Two things fall out of this table. First, the candidate pool size is a *latency knob*, not just a recall knob — every extra candidate costs ranker milliseconds, so the pool is sized to the budget, which caps how much recall you can carry into ranking. This is the central recall-versus-latency Pareto trade that runs through the whole series: a bigger pool means higher potential recall but a slower (or shallower) ranker. Second, retrieval *must* be approximate — exact nearest-neighbor over a billion vectors does not fit in 10 ms, so you accept a recall hit (an ANN index recalls maybe 95–99% of true neighbors) to make the budget. The [ANN serving post](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) quantifies exactly how much recall you trade for how much latency on each index type; the relevant fact here is that *approximation is not optional at feed scale* — it is the only way the funnel fits in the time you have.

#### Worked example: sizing the candidate pool against the budget

You have 60 ms for ranking and your multi-task net scores at 50 microseconds per candidate (batched on a GPU). That caps the pool at $60\text{ms} / 0.05\text{ms} = 1{,}200$ candidates. Suppose at 1,200 candidates your funnel recall ceiling (from the union of sources) is 0.78, and you measure that doubling the pool to 2,400 would raise recall to 0.84 — but at 2,400 candidates ranking takes 120 ms, blowing the budget. Your options are not "just rank more": you must either make the ranker 2× faster (distillation, a smaller net, quantization — see [distillation and compression for recsys](/blog/machine-learning/recommendation-systems/distillation-and-compression-for-recsys)), or improve *which* 1,200 candidates you carry (better, more complementary sources so the same pool size captures more relevant items). The second is usually the better lever: a smarter source mix raises recall *within* a fixed pool size, which costs no extra ranking time. This is the kind of decision the latency budget forces constantly — you cannot spend your way to recall, you have to engineer it within a fixed millisecond envelope. It is also why the candidate-source quality (figure 7) matters so much: it is free recall relative to the ranking budget.

## 9. When to reach for each idea (and when not to)

A decisive section, because every choice here is expensive and most teams should *not* build the most elaborate version.

**Reach for a GNN (PinSage-style) when** your items live in a rich graph (co-saves, co-purchases, shared boards/playlists) and that structure carries signal a two-tower cannot easily capture, *and* you have the scale to justify the infra. **Do not** reach for a GNN if a plain [two-tower retriever](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) on good features already hits your recall target — the GNN's serving and training complexity is large, and the [GNN-for-rec post](/blog/machine-learning/recommendation-systems/graph-neural-networks-for-recommendation) shows LightGCN's lift over MF can be modest on smaller graphs. The graph earns its keep at Pinterest scale; it may not at yours.

**Reach for many candidate sources when** your single retriever has a recall ceiling you can measure and a clear class of relevant items it systematically misses (fresh, social, trending). Adding a complementary source is one of the cheapest big wins available. **Do not** add sources that are redundant with existing ones — by the union-recall math, a redundant source adds infra and noise without raising the ceiling. Always justify a new source by the *complementary* slice it covers.

**Reach for multi-task + value-model fusion when** you have more than one objective that matters and a single proxy is gameable (clicks → clickbait). This is almost always true for a feed, so almost every feed should be multi-task. **Do not** ship a value model whose weights were tuned only on offline loss — the weights are a product decision and must be tuned online against retention and satisfaction, or you will faithfully optimize the wrong thing. And use MMoE/PLE, not naive shared-bottom, the moment your tasks show negative transfer.

**Reach for online learning when** your domain is high-drift and high-signal-density (short video, news, live commerce) and you have *measured* a real staleness penalty (the worked example above). **Do not** build online training for a stable catalog with sparse feedback — it is a large, ongoing operational cost (a streaming pipeline, more on-call, harder debugging because the model changes under you) that a nightly batch would render pointless. Match loop latency to drift.

**A general anti-pattern to avoid:** copying TikTok's architecture because TikTok is successful. TikTok's online-learning-heavy design is a response to *short video's* extreme drift and signal density. If you run a marketplace or a stable media catalog, that design is over-engineering — you would pay short-video infra costs for stable-catalog drift. The right lesson from these three systems is not "build what they built" but "find your bottleneck (representation, recall breadth, objective balance, or freshness) and invest there." The funnel is shared; where you pour the hard engineering is not.

## 10. The reproduction, end to end

Tying the three repros into one mental flow, here is how they compose into a miniature feed ranker you could actually run. (1) Build pin embeddings with the importance-pooled GraphSAGE model and the hard-negative curriculum — that is your item representation and the basis for an embedding retrieval source. (2) Add a second candidate source — say a "fresh" source that just returns recent items — and union the pools to lift recall, exactly the figure-7 pattern. (3) Score the unioned pool with the multi-task ranker, predicting click/complete/share/skip. (4) Fuse with the value model $\sum_k w_k p_k$ and sort. (5) For drift, wrap the embedding model in the online-update loop so it tracks shifting interests. Each piece is small; together they are the whole funnel of figure 1.

The honest measurement protocol for any of this, which the series hammers on in [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) and [offline vs online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys): use a **temporal split** (train on the past, evaluate on the future) so you do not leak; report **full-corpus** Recall@K and NDCG@K, not sampled metrics (sampled metrics are inconsistent, per Krichene & Rendle, KDD 2020); warm up before measuring latency; and remember that offline lift is necessary but not sufficient — the value-model weights and the freshness benefit ultimately have to be confirmed with an [A/B test](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) on the real feed, because the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) is exactly where these systems get surprising. PinSage's authors did this; so did Monolith's. So should you.

Here is the eval harness I used for the repro numbers above — full-corpus Recall@K and NDCG@K with a temporal split, the metrics every result in this post is measured with. Note it ranks each user's held-out positive against the *entire* item set, not a sampled subset:

```python
import numpy as np

def recall_at_k(ranked_ids, true_items, k):
    """Fraction of a user's held-out positives that appear in the top-k."""
    hits = sum(1 for t in true_items if t in ranked_ids[:k])
    return hits / max(1, len(true_items))

def ndcg_at_k(ranked_ids, true_items, k):
    """Normalized discounted cumulative gain over the full item set."""
    dcg = 0.0
    for rank, item in enumerate(ranked_ids[:k]):
        if item in true_items:
            dcg += 1.0 / np.log2(rank + 2)   # rank is 0-indexed
    # ideal DCG: all true items packed at the top
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(true_items))))
    return dcg / max(1e-9, ideal)

def evaluate_full(model, users, train_pos, test_pos, all_items, k=10):
    """Temporal split: train on past interactions, score the future."""
    rs, ns = [], []
    item_emb = model.i.weight.detach().cpu().numpy()      # (n_items, dim)
    for u in users:
        u_vec = model.u.weight[u].detach().cpu().numpy()
        scores = item_emb @ u_vec                          # full-corpus scores
        scores[list(train_pos[u])] = -np.inf               # mask seen items
        ranked = np.argsort(-scores)                       # rank ALL items
        rs.append(recall_at_k(ranked, test_pos[u], k))
        ns.append(ndcg_at_k(ranked, test_pos[u], k))
    return float(np.mean(rs)), float(np.mean(ns))
```

The two production-relevant choices baked in here: masking already-seen items so you do not get credit for re-recommending what the user already has, and ranking against `all_items` rather than a sampled candidate set — the Krichene-Rendle result is that sampling the negatives during *evaluation* can flip which model looks better, so for trustworthy offline numbers you rank the full corpus even though it is slower. Every Recall and NDCG figure quoted in this post came out of exactly this harness on the toy graphs described.

## What to steal from the feeds

If you are building or improving a feed, here is the concrete steal-list, ordered by leverage:

1. **Steal the many-sources pattern first.** It is the cheapest big recall win. Find the relevant items your single retriever misses, build a complementary source for them, union the pools. (Instagram)
2. **Steal multi-objective value fusion.** Stop ranking on one proxy. Predict several engagement events, fuse with tuned weights, and put a *negative* head (skip / "see less") in the sum so clickbait gets penalized. Tune the weights online. (Instagram, YouTube)
3. **Steal content-feature fallback for cold items.** Make content embeddings *inputs* to your model so a brand-new item gets a sane embedding before it has any interactions. (Pinterest)
4. **Steal the staleness measurement.** Replay a stale model against fresh data to put a number on what freshness is worth, then decide whether online learning pays. Do not build online training on faith. (ByteDance)
5. **Steal the hard-negative curriculum.** Start with easy negatives, ramp in hard ones. It keeps retrieval improving long after random negatives plateau. (Pinterest)
6. **Steal deliberate exploration.** Reserve a slice of the feed for uncertain or fresh items so the model can discover new interests and the catalog does not collapse. (TikTok)

## Key takeaways

- **One funnel, three investments.** All three feeds run many-sources → multi-task ranking → value fusion → re-rank. Pinterest invested in item representation (a graph GNN), Instagram in ranking breadth and objective fusion, TikTok in loop latency (online learning). Find *your* bottleneck.
- **PinSage made GNNs web-scale** with random-walk neighborhoods, importance pooling (weight neighbors by visit count), producer-consumer minibatching, MapReduce inference, and a hard-negative curriculum — reported ~150% hit-rate lift over the prior system.
- **Many candidate sources raise the recall ceiling** by the union-recall identity; the gain is large only when sources are *complementary*. This is the single cheapest high-leverage feed move.
- **The value model $\text{score} = \sum_k w_k p_k$ is where product values become numbers.** Predict several engagement events (including a negative skip head), fuse with online-tuned weights. It is why feeds rank on completion/share, not gameable clicks.
- **Online learning matters when drift and signal density are high** (short video). Monolith's collisionless embeddings (cuckoo hashing + feature expiry) plus streaming training keep the model minutes-fresh; the freshness benefit grows with staleness. For stable catalogs, batch is fine.
- **Match loop latency to drift rate.** Measure the staleness penalty (replay a stale model on fresh data) and only pay for online training when the number justifies it.
- **Exploration is not optional.** A pure-exploit feed converges to a boring loop and collapses the catalog; reserve a slice for discovery.
- **The engagement-vs-wellbeing tension is structural.** Put satisfaction and retention (not just engagement) in the value model, or you will optimize time-on-app at the cost of trust.
- **Confirm everything online.** Offline lift is necessary, not sufficient; temporal splits, full-corpus metrics, and A/B tests are how PinSage and Monolith proved their wins, and how you should prove yours.

## Further reading

- **PinSage** — Ying, He, Chen, Eksombatchai, Hamilton, Leskovec, *Graph Convolutional Neural Networks for Web-Scale Recommender Systems*, KDD 2018. The web-scale GNN, random-walk importance pooling, MapReduce inference, hard-negative curriculum. ([arxiv.org/abs/1806.01973](https://arxiv.org/abs/1806.01973))
- **Monolith** — Liu et al. (ByteDance), *Monolith: Real Time Recommendation System With Collisionless Embedding Table*, 2022. Cuckoo-hash collisionless embeddings, feature expiry, streaming online training, fault tolerance. ([arxiv.org/abs/2209.07663](https://arxiv.org/abs/2209.07663))
- **DLRM** — Naumov et al. (Meta), *Deep Learning Recommendation Model for Personalization and Recommendation Systems*, 2019. The canonical industrial ranking model whose embedding tables dominate memory. ([arxiv.org/abs/1906.00091](https://arxiv.org/abs/1906.00091))
- **MMoE** — Ma et al., *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*, KDD 2018, and PLE (Tang et al., RecSys 2020) — the multi-task backbones behind value-model ranking.
- **Sampled-metric warning** — Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020. Why sampled Recall/NDCG can be inconsistent — measure on the full corpus.
- Within this series: [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking), [graph neural networks for recommendation](/blog/machine-learning/recommendation-systems/graph-neural-networks-for-recommendation), [multi-task and multi-objective ranking (MMoE/PLE)](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple), [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores), [building the training pipeline for recsys](/blog/machine-learning/recommendation-systems/building-the-training-pipeline-for-recsys), the intro map [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

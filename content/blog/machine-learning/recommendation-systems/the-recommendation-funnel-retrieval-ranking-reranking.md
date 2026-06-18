---
title: "The Recommendation Funnel: Retrieval, Ranking, and Re-Ranking"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Derive the retrieval-ranking-reranking cascade from latency math, prove the recall ceiling, and build a runnable two-stage pipeline that beats single-stage scoring on both speed and NDCG."
tags:
  [
    "recommendation-systems",
    "recsys",
    "retrieval",
    "ranking",
    "re-ranking",
    "candidate-generation",
    "two-tower",
    "approximate-nearest-neighbor",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-1.png"
---

A product manager pings you on a Friday: "Why does our 'For You' feed take 800 milliseconds to load on mobile, and why is it still recommending the same five creators to everyone?" You open the trace. The feed service is doing exactly one thing — running the brand-new deep ranking model, the one the science team is proud of, over every video in the catalog and sorting the scores. There are ninety million videos. The model takes about a millisecond per video on a warm GPU batch. Ninety million milliseconds is a day. The only reason the page renders at all is that some engineer, eighteen months ago, quietly capped the candidate set to "the ten thousand most popular videos this week" so the request would return before the user gave up. That cap is why everyone sees the same five creators. The expensive model never gets a chance to be clever, because the cheap shortcut in front of it already decided the answer.

This is the single most important structural fact about production recommenders, and almost no introductory tutorial says it out loud: **you cannot score everything**. A recommender that serves billions of items to hundreds of millions of users in tens of milliseconds is not one model. It is a *cascade* of models, cheap to expensive, each one filtering the candidate set down by an order of magnitude before the next, more expensive model gets to look. We call it the funnel, and it has three canonical stages — retrieval, ranking, and re-ranking. The figure below is the whole shape on one page: a hundred million items at the top, ten items at the bottom, and a strict latency budget the sum of the stages has to fit inside.

![The three-stage recommendation funnel drawn as vertical layers from a 100M item catalog down through retrieval ranking and re-ranking to a top-10 page with candidate counts and latency labels on each layer](/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-1.png)

By the end of this post you will be able to derive the funnel from first principles (it falls directly out of a latency-versus-accuracy budget), prove the *recall ceiling* — the theorem that says no amount of ranking genius can recover an item retrieval threw away — build a runnable two-stage pipeline in PyTorch and faiss that beats single-stage scoring on both p99 latency *and* end-to-end NDCG, and reason about when a third stage earns its keep and when it does not. This is the spine of the entire series: retrieval feeds ranking feeds re-ranking, all of it riding on the feedback loop and read off the offline-versus-online reality gap. If you want the bird's-eye map first, start with [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system); if you want the end-state checklist, the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) is the bookend. This post is the load-bearing wall in between.

## 1. Why one model cannot serve a billion items in 50 milliseconds

Let me make the impossibility concrete, because the whole architecture is a response to it. The job of a recommender at request time is: given a user $u$ and a context $c$, score the catalog of $N$ items and return the best $K$ of them. The naive, mathematically-pure way to do that is to run your best scoring function $f(u, c, i)$ over every item $i$ and take the top $K$:

$$\text{Top-}K = \operatorname*{arg\,top\text{-}K}_{i \in \mathcal{I}} f(u, c, i)$$

This is a *brute-force scan*, and its cost is

$$T_{\text{scan}} = N \cdot t_{\text{item}}$$

where $t_{\text{item}}$ is the time to compute one score. The whole problem lives in those two numbers. $N$ is enormous and getting bigger: YouTube has billions of videos, Amazon hundreds of millions of products, a large social feed tens of millions of *fresh* candidates per day. And $t_{\text{item}}$ for a good ranker is *not* small. A modern click-through-rate (CTR) model — a deep network with embedding lookups, feature crosses, attention over the user's history, the works — costs somewhere between a hundred microseconds and a couple of milliseconds per item even with batching, because it touches dozens of features and runs a multi-layer network for each candidate.

Multiply them. Take $N = 10^8$ items and a deep ranker at $t_{\text{item}} = 1\text{ ms}$:

$$T_{\text{scan}} = 10^8 \times 1\text{ ms} = 10^8\text{ ms} = 100{,}000\text{ s} \approx 27.8\text{ hours}.$$

Twenty-eight hours to render one feed. Your latency budget is on the order of 50–100 milliseconds for the *entire* request, including network, feature fetch, and serialization. You are off by *six orders of magnitude*. There is no GPU, no quantization, no kernel fusion that closes a million-fold gap. You cannot brute-force scan a catalog of a hundred million items with a deep model. Full stop.

So you have exactly two levers, and the funnel pulls both:

1. **Shrink $N$** before the expensive model runs. If you can cheaply throw away 99.999% of the catalog and only hand the deep ranker a thousand survivors, the ranker's bill drops from $10^8 \cdot t_{\text{item}}$ to $10^3 \cdot t_{\text{item}}$ — a hundred-thousand-fold cut.
2. **Shrink $t_{\text{item}}$** for the part that *does* look at everything. The model that scores a hundred million items has to cost almost nothing per item — not a deep network but a single dot product, which on modern hardware is nanoseconds and, crucially, can be turned into a geometry problem (nearest-neighbor search) instead of a scoring loop.

Those two levers, applied in sequence, *are* the funnel. A cheap, recall-oriented stage looks at everything and is allowed to be sloppy because its only job is "don't throw away the good stuff." An expensive, precision-oriented stage looks at the survivors and is allowed to be slow per-item because there are only a thousand of them. The architecture is not a design choice somebody made for elegance. It is *forced* by the arithmetic. Any system that serves a large catalog under a tight SLA converges to this shape, whether it is YouTube, TikTok, Amazon, Spotify, or your e-commerce startup.

### The latency budget, made rigorous

Here is the governing inequality the whole system has to satisfy. Let the request be served by $S$ stages with per-stage wall-clock times $T_1, T_2, \ldots, T_S$, plus fixed overhead $T_{\text{fixed}}$ (network round trip, feature fetch, deserialization, logging). The service-level agreement says the *end-to-end* time must stay under a budget $B$ at the chosen percentile (almost always p99, the 99th-percentile latency — the slow tail, not the median):

$$T_{\text{fixed}} + \sum_{s=1}^{S} T_s \le B.$$

Each stage's time is its candidate count times its per-item cost:

$$T_s = n_s \cdot t_s,$$

where $n_s$ is how many candidates *enter* stage $s$ and $t_s$ is that stage's per-item cost. The design game is to choose the $n_s$ (how aggressively each stage prunes) and the $t_s$ (how expensive each stage's model is) so the sum fits under $B$ *and* the end-to-end quality is as high as possible. That is a constrained optimization, and the funnel is its solution: make $t_1$ tiny so you can afford a huge $n_1$, then let $n_s$ shrink fast enough that you can afford a growing $t_s$.

We will do the arithmetic with real numbers in the worked example below. For now, hold onto the shape: **cost per stage is (count in) times (cost each), the stages sum, and the sum has a hard ceiling.** Everything else in this post is a consequence.

#### Worked example: the funnel latency budget

Suppose your SLA is a p99 of 100 ms for the recommendation call, and fixed overhead (RPC, feature store fetch, serialization) eats 40 ms. That leaves **60 ms** for actual model work across all stages. Your catalog is $N = 10^8$. Let's allocate.

- **Retrieval.** You run approximate nearest-neighbor (ANN) search over $10^8$ item embeddings. ANN does not scan all $10^8$; a good HNSW or IVF index touches a few thousand items and returns the top 1000 in roughly **5 ms**. So $T_1 \approx 5$ ms, and it outputs $n_2 = 1000$ candidates.
- **Ranking.** A deep CTR model scores those 1000 candidates. Batched on a GPU it costs about $15\,\mu s$ per item, so $T_2 = 1000 \times 0.015\text{ ms} = 15$ ms. It keeps the top $n_3 = 50$.
- **Re-ranking.** A diversity and business-rules pass over 50 items, with a quadratic MMR step, costs maybe **5 ms**. It returns the top 10.

Total model time: $5 + 15 + 5 = 25$ ms. Add the 40 ms overhead and you land at **65 ms p99** — comfortably inside the 100 ms SLA, with 35 ms of headroom for the tail. Now ask the counterfactual: what if you skipped retrieval and ran the ranker on all $10^8$? $T = 10^8 \times 0.015\text{ ms} = 1.5 \times 10^6$ ms $= 25$ minutes. The funnel bought you a factor of about *23,000* in latency for the ranking stage alone, at the cost of a recall ceiling we will quantify shortly. That trade is the entire reason the architecture exists.

## 2. Stage one — retrieval: cheap, high-recall, billions to thousands

The first stage goes by two names — **retrieval** and **candidate generation** — and they mean the same thing: cut the catalog from $N$ down to a few hundred or a few thousand candidates, as cheaply as possible, while throwing away as few *relevant* items as possible. The defining word is **recall**. Retrieval does not care about getting the order exactly right; that is ranking's job. Retrieval cares about one thing: *did the good items make it into the pool at all?* If a movie the user would have loved never enters the candidate set, no downstream model can fix that. (That is the recall ceiling; we will prove it in Section 5.)

Because retrieval has to look at the whole catalog, its per-item cost $t_1$ must be microscopic. That rules out anything with per-candidate feature crosses or attention. In practice retrieval is built from a small number of *cheap* methods, usually run in parallel and merged:

**Two-tower embedding retrieval plus ANN.** This is the workhorse. You learn a user tower $g_\theta(u, c)$ and an item tower $h_\phi(i)$ that map users and items into the *same* $d$-dimensional vector space, trained so that a user's vector is close (in dot product) to the items they engaged with. At serving time you compute the user vector once, then find the items whose vectors have the highest dot product with it:

$$s(u, i) = g_\theta(u, c)^\top h_\phi(i).$$

This is a *maximum inner product search* (MIPS) over $N$ item vectors. You never run the item tower at request time — all item vectors are precomputed nightly and loaded into an ANN index (faiss, HNSW, ScaNN). The ANN index answers "give me the 1000 items with the highest dot product against this query vector" in a few milliseconds *without scanning all $N$*, by exploiting the geometry of the vector space. We build the two-tower model in [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) and the index itself in [approximate nearest neighbor serving with faiss, HNSW, and ScaNN](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann). For this post the key fact is: two-tower plus ANN turns "score everything" into "navigate a graph," and that is what makes $N = 10^8$ tractable.

**Item-item co-visitation.** "People who watched this also watched that." You precompute, for each item, a short list of the items most frequently co-engaged with it (from co-occurrence in sessions, often with a normalization like cosine over the co-visitation matrix). At serving time you take the user's recent items, look up each one's neighbors, and union them. It is a hash-map lookup — nanoseconds — and it captures local, session-level intent that a global two-tower model can blur. It is also a strong cold-start fallback for new users: as soon as they click one thing, you have neighbors.

**Popularity and freshness.** A small slate of globally or segment-popular items, and a slate of the freshest items (last few hours). Trivial to compute, and essential as a floor: it guarantees that even a brand-new user with no history, or a request where the personalized retrievers fail, returns *something* reasonable. It is also where you inject newly-uploaded content that has no interaction signal yet (the cold-start item problem).

**Rule- and graph-based sources.** Follows and subscriptions (a creator the user explicitly follows), geographic or language filters, editorial picks, and graph-walk retrievers (random walks on a user-item or item-item graph, the lineage of Pinterest's PinSage). Each is cheap and contributes a *different kind* of candidate.

The point of listing several is that no single retriever has good recall on its own. The two-tower model is great at "items like your long-run taste" but weak on sudden intent; co-visitation is great at session intent but blind to new items; popularity is a floor but unpersonalized. You run several and merge their outputs into one pool — that is **multi-source retrieval**, the subject of Section 6, and the graph below shows the shape.

![Multi-source retrieval graph where a two-tower ANN source an item-item co-visitation source and a popularity fallback source all fan into a merge and dedup node that feeds a heavy ranker and then a re-rank to ten step](/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-2.png)

A clarifying note on metric. Retrieval is graded on **Recall@K** — of all the items the user actually engaged with in the held-out window, what fraction appeared in the top-$K$ candidates the retriever returned? Not NDCG, not AUC. The order *inside* the candidate pool is almost irrelevant at this stage, because ranking is about to re-order all of them anyway. A retriever that returns the right 1000 items in a scrambled order is a *perfect* retriever. This is a constant source of confusion: teams optimize their retrieval model on NDCG, see it improve, and wonder why end-to-end metrics do not move. They are measuring the wrong thing. The retriever's only job is to *include* the good items; ranking decides the order.

### The science of why retrieval and ranking are trained differently

The metric difference is not cosmetic; it forces a *different loss function* at each stage, and understanding why is the cleanest illustration of the funnel's logic. The retriever has to discriminate the few hundred relevant items from a catalog of a hundred million — a needle-in-a-haystack contrastive problem — so it is trained with a **softmax over the whole catalog**, approximated by **sampled softmax** because the true normalizer over $10^8$ items is intractable. For a positive pair $(u, i^+)$ the loss is

$$\mathcal{L}_{\text{retr}} = -\log \frac{\exp\big(s(u, i^+)\big)}{\exp\big(s(u, i^+)\big) + \sum_{i^- \in \mathcal{N}} \exp\big(s(u, i^-) - \log Q(i^-)\big)},$$

where $\mathcal{N}$ is a set of sampled negatives and $Q(i^-)$ is the probability the sampler drew $i^-$. That $-\log Q(i^-)$ term — the **sampled-softmax correction** — is essential and routinely forgotten: the negatives are drawn from a non-uniform proposal (in-batch negatives are sampled in proportion to popularity, so popular items appear as negatives far more often), and without the correction the model systematically under-scores popular items. The correction de-biases the estimator back toward the true full-softmax gradient. The retriever's loss is a *global* contrastive objective because its job is global recall — pull relevant items toward the user across the entire catalog.

The ranker faces a different problem. Its candidate set is already small and already plausibly relevant; its job is *fine-grained ordering* within that pool. For top-$K$ ordering, a **pairwise** loss beats a pointwise one, and the reason is worth making rigorous. The Bayesian Personalized Ranking (BPR) objective maximizes the probability that a positive item $i^+$ outranks a sampled negative $i^-$ for the same user:

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i^+, i^-)} \log \sigma\big(s(u, i^+) - s(u, i^-)\big),$$

where $\sigma$ is the logistic function. Differentiate it: the gradient on the score gap is $\propto \big(1 - \sigma(s_{i^+} - s_{i^-})\big)$, which is *large exactly when the model has the order wrong* ($s_{i^-} \ge s_{i^+}$) and *vanishes when the order is already correct by a margin*. A pointwise logistic loss, by contrast, keeps pushing the absolute score of a positive toward 1 even after it already outranks every negative — wasting gradient on calibration the ranking metric does not reward, and under-weighting the pairs that are actually mis-ordered. **The pairwise gradient sees the order; the pointwise gradient sees the absolute label.** Since NDCG depends only on order, the pairwise loss optimizes a closer surrogate to the metric you actually report. (When you *also* need calibrated probabilities — for multi-task blending, Section 3 — you train pointwise heads with logloss and calibrate them separately, accepting the trade. The choice of loss follows the metric the stage is graded on, which follows the stage's job. Same chain every time.)

Here is the in-batch sampled-softmax loss for the retriever, with the $\log Q$ correction wired in. In-batch negatives are nearly free — every other positive in the minibatch serves as a negative for this user — which is why two-tower training scales, but it makes $Q$ popularity-skewed and the correction non-optional.

```python
import torch
import torch.nn.functional as F

def in_batch_sampled_softmax(user_vec, item_vec, log_q):
    """user_vec, item_vec: (B, d) aligned positive pairs.
    log_q: (B,) log sampling prob of each in-batch item (popularity-based).
    Every row i treats column i as the positive and all other columns as negatives."""
    logits = user_vec @ item_vec.T          # (B, B) all pairwise scores
    logits = logits - log_q[None, :]        # sampled-softmax / logQ correction
    labels = torch.arange(user_vec.size(0), device=user_vec.device)
    return F.cross_entropy(logits, labels)  # positive is on the diagonal
```

Drop the `- log_q` line and the model quietly learns to under-rank popular items, because they appear as in-batch negatives far more often than a uniform sampler would draw them; the correction restores the unbiased gradient. This single line is one of the most common silent bugs in two-tower training, and it is invisible offline until you notice the recall on popular items is mysteriously low.

## 3. Stage two — ranking: expensive, high-precision, thousands to tens

Now the candidate set is small — a thousand items, say — and you can afford to be expensive. This is the **ranking** stage, and its defining word is **precision**. Where retrieval asked "is this item plausibly relevant?", ranking asks "exactly how likely is this user to click, watch, like, purchase, and how long will they stay?" It assigns each of the thousand candidates a calibrated score and sorts them, keeping the top few dozen.

The ranker is where you spend your modeling budget. Because it only sees a thousand items per request, you can give it *everything*:

- **Rich features.** User features (long-term and short-term history, demographics, device), item features (content embeddings, age, creator, category), and crucially **cross features** and **context features** (time of day, the user's last few actions, the query if there is one). Retrieval mostly cannot use user-item cross features cheaply — that is precisely why it is cheap — so the ranker is the first place the model sees the *interaction* between a specific user and a specific item in full detail.
- **A deep architecture.** Wide-and-deep, DeepFM, DCN (deep-and-cross), DIN/DIEN with attention over the user's behavior sequence — whatever your problem needs. We cover the ranker in depth in [the ranking model and CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations); here it is a black box that takes one (user, item, context) triple and returns a score.
- **Multi-task heads.** Real systems almost never optimize a single objective. A feed ranker predicts $p(\text{click})$, $p(\text{like})$, $p(\text{share})$, $p(\text{long-watch})$, $p(\text{follow})$, and combines them into a final score with business-chosen weights. This is multi-task learning (often a shared-bottom or mixture-of-experts architecture), and it is where calibration matters most.

That last word — **calibration** — is the subtle, load-bearing property of a good ranker, and it is where the science of ranking diverges from plain classification. A score is *calibrated* if it means what it says: when the model outputs $p(\text{click}) = 0.3$ for a set of impressions, about 30% of them should actually be clicked. Why does calibration matter when you are just going to sort? For two reasons. First, the **final score is a combination** of several heads — $\text{score} = w_1 \, p(\text{click}) + w_2 \, p(\text{like}) + w_3 \, \mathbb{E}[\text{watch time}] + \ldots$ — and you cannot meaningfully add a click probability to a like probability to an expected watch-time unless each is on its true, calibrated scale. An uncalibrated click head that is systematically 2× too high will dominate the sum and silently turn your "engagement" ranker into a "clickbait" ranker. Second, when you **blend candidates from multiple retrieval sources** (Section 6) or compare against a price or a bid, the absolute score has to be comparable across sources. We measure calibration with the **expected calibration error** (ECE) and fix it with isotonic regression or Platt scaling. A staff-level instinct: *if your scores are going to be combined, compared, or thresholded — not just sorted within one pool — they have to be calibrated.*

Ranking is graded on **NDCG@K** (normalized discounted cumulative gain — does it put the most relevant items at the top, with a position discount?), **AUC** (does it order a relevant item above an irrelevant one?), and **logloss/ECE** (is it calibrated?). Note how different these are from retrieval's Recall@K. The same item set, graded on order rather than membership. The matrix below lays the three stages side by side on exactly these axes — input size, latency budget, model cost, objective, and metric — because internalizing *that the stages are graded on different things* is the single biggest conceptual unlock in this post.

![A matrix comparing retrieval ranking and re-ranking across five properties input size latency model cost objective and metric with colored cells showing retrieval as cheap and recall oriented ranking as expensive and precision oriented and re-ranking as set oriented](/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-3.png)

## 4. Stage three — re-ranking: shaping the set, not the items

You now have, say, the top 50 ranked candidates, each with a calibrated relevance score. The naive thing is to show the top 10 by score and call it done. The naive thing is usually wrong, and the reason is the central insight of re-ranking: **the value of a page is not the sum of the values of its items.**

Suppose the ranker's top five candidates are all clips from the same creator, on the same topic, posted the same day. Each one, scored independently, is highly relevant — the user loves that creator. But a *page* of five near-identical clips is a worse experience than a page of five varied-but-slightly-less-relevant items. The user is bored by item three and gone by item five. The ranker, scoring each item in isolation, has no way to see this. It does not know what *else* is on the page. Re-ranking is the stage that scores the **set**, not the item.

Re-ranking pursues several objectives that pure relevance cannot express, and they fall into two families — things that make the page better for the user, and things the platform requires. The tree below splits them out.

![A tree of re-ranking objectives branching into a user-experience family with diversity freshness and exploration and a platform-rules family with business rules and dedup and filtering](/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-6.png)

**Diversity** is the headline objective, and the canonical algorithm is **Maximal Marginal Relevance** (MMR). MMR builds the result list greedily: at each slot, it picks the candidate that maximizes a blend of relevance to the user and *dissimilarity* to what is already on the page. Formally, given items already selected $S$ and remaining candidates $R$, the next pick is

$$\text{next} = \operatorname*{arg\,max}_{i \in R}\;\Big[\, \lambda \cdot \text{rel}(i) - (1-\lambda) \max_{j \in S} \text{sim}(i, j) \,\Big],$$

where $\text{rel}(i)$ is the ranker's relevance score, $\text{sim}(i, j)$ is a similarity (cosine over item embeddings, or a shared-attribute indicator), and $\lambda \in [0, 1]$ trades relevance against diversity. At $\lambda = 1$ you get pure relevance (the ranker's order); at $\lambda = 0$ you maximize spread regardless of relevance. A typical production value is $\lambda \approx 0.7$ — mostly relevance, with a real penalty for redundancy. The subtraction term is the whole point: it makes the *marginal* value of an item depend on what is already chosen, which is exactly the set-awareness item-by-item scoring lacks.

MMR is small enough to write in a dozen lines, and seeing it makes the set-awareness concrete. It is a greedy loop over the ranked candidates, picking the next slot by the relevance-minus-redundancy criterion above. This runs over the 50 candidates the ranker handed up and returns the diversified top-10.

```python
import numpy as np

def mmr_rerank(cand_ids, rel_scores, item_vecs, k=10, lam=0.7):
    """Stage 3: greedy MMR. cand_ids/rel_scores are aligned 1D arrays;
    item_vecs are L2-normalized so dot product == cosine similarity."""
    selected, remaining = [], list(range(len(cand_ids)))
    # min-max normalize relevance so it is on the same [0,1] scale as cosine
    r = rel_scores
    r = (r - r.min()) / (r.ptp() + 1e-9)
    while remaining and len(selected) < k:
        best_idx, best_val = None, -np.inf
        for idx in remaining:
            if selected:
                sims = item_vecs[idx] @ item_vecs[[cand_ids[s] for s in selected]].T
                redundancy = float(sims.max())
            else:
                redundancy = 0.0
            val = lam * r[idx] - (1.0 - lam) * redundancy
            if val > best_val:
                best_val, best_idx = val, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [cand_ids[s] for s in selected]
```

Notice the cost: the inner loop is quadratic in the candidate count, $O(k \cdot |R|)$ similarity comparisons, which is fine for 50 candidates (a few thousand dot products, sub-millisecond) but explains why re-ranking runs on *tens* of items, not thousands. The funnel shrinks the set precisely so that an $O(n^2)$ set-aware pass is affordable at the bottom — yet another stage whose feasibility depends on the one before it having already pruned hard.

**Freshness and recency.** Boost newly-created items so the feed feels alive and so new content can accumulate the interaction signal it needs to ever rank organically (without a freshness boost, new items are invisible, never get clicks, and so stay invisible — a cold-start trap). **Exploration.** Reserve a slot or two for items the model is *uncertain* about, to gather signal and break out of the feedback loop that otherwise collapses the catalog onto a handful of safe bets. **Business rules.** Ad load and sponsored caps (no more than one promoted item every $k$ slots), creator fairness, regulatory filters, "don't show competitor brands next to ours." **Dedup and filtering.** Remove items the user has already seen, already dismissed, or that violate policy.

The contrast that makes this concrete is item-by-item ranking versus whole-page re-ranking, drawn below. On the left, independent scores stack five near-duplicates at the top — high NDCG, low satisfaction. On the right, a set-aware pass spreads topics across the page — a slight NDCG dip, measured engagement up. This is the canonical "offline metric down, online metric up" pattern, and re-ranking is the most common place it shows up.

![A before-after comparison showing item-by-item ranking producing a top set of near-duplicate items with high NDCG but low satisfaction versus whole-page re-ranking producing a topically spread page with a slight NDCG dip and higher engagement](/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-7.png)

The frontier of re-ranking is **listwise** and **whole-page** models that learn the set-scoring function directly rather than hand-coding MMR — sequence models (a transformer over the candidate list) that condition each item's final score on the others, trained to maximize a page-level reward. They are more powerful and more expensive; MMR is the robust default that gets you most of the gain for almost no cost. Re-ranking is graded not on NDCG (which, as we just saw, it intentionally sacrifices a little) but on **diversity** (intra-list dissimilarity), **catalog coverage** (what fraction of the catalog ever gets shown — the antidote to popularity collapse), and ultimately on **online engagement**, which is the only metric that integrates all of these.

## 5. The information-loss problem and the recall-ceiling theorem

Here is the most important — and most often ignored — property of the whole funnel. A cascade is a sequence of filters, and **a filter can only remove information, never add it.** Whatever retrieval drops, ranking never sees, and re-ranking never sees. The downstream stages can only re-order and prune what survived; they cannot resurrect a discarded item. This gives us a hard, provable upper bound on end-to-end quality.

Let me state it precisely. Define, for a given user, the set of *relevant* items $\mathcal{R}$ (the items they would have engaged with). Let $\mathcal{C}$ be the candidate set retrieval returns. The **retrieval recall** is

$$\text{Recall}_{\text{retr}} = \frac{|\mathcal{R} \cap \mathcal{C}|}{|\mathcal{R}|},$$

the fraction of relevant items that made it into the candidate pool. Now run a *perfect* ranker — one that orders the candidate set flawlessly. Even so, the end-to-end recall (relevant items in the final list) cannot exceed the number of relevant items that entered the pool. The relevant items *not* in $\mathcal{C}$ are simply gone. Therefore:

$$\boxed{\;\text{Recall}_{\text{end-to-end}} \le \text{Recall}_{\text{retr}}.\;}$$

This is the **recall ceiling**. It is not an empirical observation; it is a logical identity about cascaded filters. The end-to-end recall of the system — and, downstream of it, NDCG, engagement, GMV, everything — is *capped* by retrieval recall, regardless of how good ranking and re-ranking are. The figure below draws it: retrieval surfaces 70% of the relevant items and drops 30% forever; even a perfect ranker on the survivors is capped at 0.70.

![A graph showing the recall ceiling where a 100M catalog feeds a retrieval stage with recall at 1000 of 0.70 that surfaces 70 percent of relevant items to the ranker and drops 30 percent of items forever capping the end-to-end result at 0.70 regardless of ranker quality](/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-5.png)

The practical consequences are sharp and counterintuitive:

1. **Retrieval is the silent bottleneck.** Almost every team I have seen over-invests in ranking and under-invests in retrieval, because ranking is where the impressive deep-learning happens and where offline AUC is easy to push. But if retrieval recall is 0.6, the best possible end-to-end recall is 0.6, and a ranking improvement that lifts in-pool NDCG by 0.05 is rearranging deck chairs on the 60% of relevant items you managed to retrieve. The first question to ask of an underperforming recommender is almost never "is the ranker good?" — it is "what is our retrieval recall, and how much headroom is the ceiling leaving on the table?"
2. **A ranking metric is conditional.** When you report "NDCG@20 = 0.41," you are reporting NDCG *on the retrieved candidate set*, not on the catalog. If retrieval changes, that number changes even with the ranker frozen. Offline ranking experiments must hold the candidate set fixed, or you are measuring two things at once.
3. **Stages compound.** With multiple cascade stages, the ceiling is the *product* of survival probabilities. If retrieval recall is 0.7 and ranking's top-50 cut retains 0.9 of the relevant items it received, the recall reaching re-ranking is $0.7 \times 0.9 = 0.63$. Every cut you add is another multiplicative tax on recall. This is the cost side of adding stages, and it is why you do not add stages casually (Section 10).

#### Worked example: the recall ceiling caps NDCG

Make it numeric. A user has 10 relevant items in the held-out window. Retrieval returns 1000 candidates and, of the 10 relevant items, surfaces 7 — so $\text{Recall}_{\text{retr}@1000} = 0.7$. Now hand those 1000 to a *perfect* ranker that puts the 7 surfaced relevant items in positions 1 through 7. Compute NDCG@20. The discounted cumulative gain is $\sum_{p=1}^{7} \frac{1}{\log_2(p+1)} = 1.000 + 0.631 + 0.500 + 0.431 + 0.387 + 0.356 + 0.333 \approx 3.64$. The *ideal* DCG — if all 10 relevant items were retrievable and ranked first — is $\sum_{p=1}^{10} \frac{1}{\log_2(p+1)} \approx 4.54$. So even with a perfect ranker, NDCG@20 $\le 3.64 / 4.54 \approx \mathbf{0.80}$. The 3 dropped items cost you 0.20 of NDCG that *no ranker can ever recover*. If instead you improve retrieval to surface 9 of 10 relevant items, the ceiling rises to roughly $4.31 / 4.54 \approx 0.95$ — a +0.15 NDCG headroom unlock that the ranking team, working downstream of the old retriever, could never have reached. This is the arithmetic behind "fix retrieval first."

## 6. Multi-source retrieval and candidate blending

Section 2 listed several retrieval methods and claimed you run them together. Now the *why* and the *how*, because blending is where calibration and budget allocation get real.

The reason to blend is the recall ceiling. A single retriever has a recall ceiling; the *union* of several complementary retrievers has a higher one, because each source surfaces relevant items the others miss. Two-tower ANN surfaces long-run-taste items; co-visitation surfaces session-intent items; freshness surfaces brand-new items with no embedding signal; follows surface explicitly-chosen creators. Their union covers more of $\mathcal{R}$ than any one alone. Concretely, if two-tower has recall 0.55 and co-visitation has recall 0.45, and they are partly complementary, their union might hit 0.72 — a recall-ceiling lift you get for the cost of a second cheap lookup. *Raising the union recall is the highest-leverage thing you can do in the whole funnel*, because it raises the ceiling on everything downstream.

The hard part is **merging**. Each source returns candidates with its own scores, on its own scale. Two-tower returns dot products in $[-\infty, \infty]$; co-visitation returns a co-occurrence count or a cosine; popularity returns a view count. You cannot just concatenate and sort by raw score — the scales are incomparable, and whichever source happens to produce big numbers wins. There are three sane strategies:

1. **Quota-based merge (the robust default).** Give each source a fixed budget — "600 from two-tower, 300 from co-visit, 100 from popularity" — take that many from each, union, and dedup. You let *ranking* sort out the order across sources. This sidesteps the score-comparability problem entirely: you are not comparing scores across sources, you are reserving slots. It is what most large systems actually do, because it is robust and tunable (budgets are a few numbers you can A/B test).
2. **Calibrated-score merge.** Map every source's score onto a common, calibrated scale (e.g. an estimated $p(\text{engage})$ via isotonic regression per source) and then take the global top-$M$ by the calibrated score. This is more elegant and can be more efficient, but it lives or dies on calibration — an over-confident source floods the pool. *This is exactly why calibrated scores matter when blending*, the science point flagged in the kit: only calibrated scores are comparable across sources.
3. **Learned blender.** A lightweight model that takes per-source features (which sources retrieved this item, with what rank/score) and predicts a blend weight. More power, more to maintain; reach for it when quota tuning plateaus.

After merge you **dedup** (an item retrieved by three sources should appear once, ideally annotated with "retrieved by {2tower, covisit}" as a feature the ranker can use), then hand the deduped pool to ranking. A practical detail that bites people: the ranker should see *which sources retrieved each candidate* as a feature, because "this item came from your explicit follows" and "this item came from popularity" carry different priors even at the same ranker score.

**Budget allocation** is the meta-decision. You have a total candidate budget (say 1000 — the number the ranker can afford to score in its latency slice) and you split it across sources. Spend too much on popularity and you crowd out personalization; spend too much on a noisy source and you waste ranker budget scoring junk. The allocation is itself an A/B-tested hyperparameter, and the right split shifts by surface (a cold-start home feed leans on popularity; a logged-in power user's feed leans on two-tower and follows). The insight to carry: **the candidate budget is a scarce, allocatable resource, and how you split it across sources sets the recall ceiling.**

#### Worked example: how much union recall buys you

You run two retrievers. Two-tower has Recall@500 = 0.55; co-visitation has Recall@500 = 0.45. If they were perfectly redundant (surfaced the same relevant items), the union recall would still be 0.55 — adding the second source would buy nothing. If they were perfectly complementary (no overlap in the relevant items each surfaces), the union would be $0.55 + 0.45 = 1.00$. Reality is in between; suppose 40% of the relevant items two-tower finds are also found by co-visitation. Then co-visitation's *incremental* relevant items are the 60% of its 0.45 that two-tower misses: $0.45 \times 0.60 = 0.27$. Union recall $\approx 0.55 + 0.27 = 0.82$. From the Section-5 ceiling, lifting retrieval recall from 0.55 to 0.82 raises the maximum achievable end-to-end NDCG by a large margin — far more than any ranker tweak inside the old 0.55 pool could deliver. And the cost was one extra hash-map lookup at a few microseconds. This is the single highest-ROI move in most funnels, and the arithmetic is why: complementary sources are nearly additive in recall, and recall is the ceiling. The corollary: a *third* retriever that mostly overlaps the first two (high redundancy) buys almost nothing and just burns ranker budget — measure incremental recall before adding a source, not standalone recall.

A related failure mode the blend has to fight is **popularity collapse**, the feedback-loop fixed point this series keeps returning to. If your retrieval and ranking both favor popular items (which they will, because popular items have the most training signal), the system shows popular items, which generates more interactions on popular items, which makes the next model favor them even more — a self-reinforcing loop that, left alone, collapses the served catalog onto a tiny head of viral content (the "same five creators" from the intro). Multi-source retrieval fights it from the supply side (a freshness source injects new items; an exploration source injects uncertain ones) and re-ranking fights it from the demand side (a coverage objective penalizes over-showing the head). Neither stage alone breaks the loop; the funnel breaks it by giving non-popular items both a way *into* the candidate pool (retrieval diversity) and a way *onto* the page (re-rank coverage). Catalog coverage — the fraction of the catalog shown over a window — is the metric that detects the collapse before it becomes visible in engagement.

## 7. The single-stage-vs-cascade contrast, and a runnable two-stage pipeline

Let me put the architectural claim — cascade beats single-stage on *both* latency and quality — on a firm footing, first with the picture, then with code you can run.

![A before-after comparison contrasting single-stage full scoring that runs a deep ranker over 100M items taking 28 hours and missing the SLA by a million times against a cascaded funnel that retrieves down to 1000 then runs the deep ranker on only 1000 hitting a p99 of 35ms inside the SLA](/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-4.png)

The left column is single-stage full scoring: the deep ranker over the whole catalog, infeasible by six orders of magnitude. The right column is the cascade: cheap retrieval shrinks the set, the deep ranker runs on the survivors, p99 lands inside the SLA. The non-obvious part is that the cascade is not just *faster* — under a fixed latency budget it is also *more accurate*, because it lets you afford a much deeper ranker on the survivors than you could ever afford to run on the whole catalog. The funnel does not trade quality for speed; it trades *coverage* (the recall ceiling) for the ability to be both fast and deep on what survives. Now let's measure it.

### Building the pipeline

We will build a minimal two-stage recommender on MovieLens-style data: a dot-product retrieval over learned item embeddings (with a faiss top-200), feeding a small ranking MLP that re-scores the 200, and we will compute end-to-end Recall@20 and NDCG@20 against a single-stage baseline. Start with the item embeddings and the retrieval index.

```python
import numpy as np
import faiss

# Assume we have trained two-tower embeddings (see the two-tower post).
# user_emb: (num_users, d), item_emb: (num_items, d), L2-normalized.
d = 64
num_items = 100_000
rng = np.random.default_rng(0)
item_emb = rng.standard_normal((num_items, d)).astype("float32")
faiss.normalize_L2(item_emb)          # cosine == dot product on the unit sphere

# Build an exact inner-product index first (the ground-truth retriever),
# then an approximate IVF index to show the real serving path.
index_flat = faiss.IndexFlatIP(d)     # exact MIPS, O(N) per query but simple
index_flat.add(item_emb)

nlist = 256                            # number of Voronoi cells for IVF
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(item_emb)             # learn the cell centroids
index_ivf.add(item_emb)
index_ivf.nprobe = 16                 # search 16 of 256 cells -> recall/latency knob
print("flat ntotal:", index_flat.ntotal, "| ivf ntotal:", index_ivf.ntotal)
```

The `nprobe` flag is the recall-versus-latency knob for ANN: higher `nprobe` searches more cells, raising recall toward the exact index at the cost of latency. Now retrieve the top-200 for a batch of users — this is stage one.

```python
def retrieve(user_vecs, index, k=200):
    """Stage 1: ANN retrieval. Returns (scores, item_ids) per user."""
    faiss.normalize_L2(user_vecs)
    scores, ids = index.search(user_vecs, k)   # (B, k) each
    return scores, ids

num_users = 2_000
user_emb = rng.standard_normal((num_users, d)).astype("float32")
retr_scores, retr_ids = retrieve(user_emb.copy(), index_ivf, k=200)
print("retrieved candidate matrix:", retr_ids.shape)   # (2000, 200)
```

Stage two is the ranking MLP. In a real system it consumes rich (user, item, context) features; here we keep it honest but small — it takes the concatenated user and item embeddings plus their elementwise product (a cheap feature cross) and outputs a relevance logit. Crucially, the ranker only ever sees the 200 retrieved candidates per user, never the full catalog.

```python
import torch
import torch.nn as nn

class RankingMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),          # one logit per (user, item) pair
        )

    def forward(self, u, v):
        # u: (B, K, d) user vec broadcast over K candidates; v: (B, K, d) item vecs
        cross = u * v                  # elementwise interaction (a feature cross)
        x = torch.cat([u, v, cross], dim=-1)
        return self.net(x).squeeze(-1) # (B, K) ranker scores

ranker = RankingMLP(d).eval()
```

Now wire the two stages together and produce the final top-20. The ranker re-scores the 200 retrieved candidates; we take the top 20 of *its* ordering.

```python
@torch.no_grad()
def two_stage_topk(user_emb, item_emb, index, k_retr=200, k_final=20):
    scores, cand_ids = retrieve(user_emb.copy(), index, k=k_retr)  # stage 1
    u = torch.tensor(user_emb)[:, None, :].expand(-1, k_retr, -1)  # (B,K,d)
    v = torch.tensor(item_emb[cand_ids])                          # (B,K,d)
    rank_scores = ranker(u, v)                                    # stage 2
    order = rank_scores.argsort(dim=1, descending=True)[:, :k_final]
    final_ids = np.take_along_axis(cand_ids, order.numpy(), axis=1)
    return final_ids                                             # (B, k_final)

topk_ids = two_stage_topk(user_emb, item_emb, index_ivf)
print("final top-20 per user:", topk_ids.shape)
```

And the evaluation harness — Recall@20 and NDCG@20 against held-out relevant items, computed honestly with a temporal split (train on the past, evaluate on the future) so there is no leakage.

```python
def recall_at_k(pred_ids, relevant_sets, k=20):
    hits, total = 0, 0
    for preds, rel in zip(pred_ids, relevant_sets):
        if not rel:
            continue
        hits += len(set(preds[:k]) & rel)
        total += len(rel)
    return hits / max(total, 1)

def ndcg_at_k(pred_ids, relevant_sets, k=20):
    ndcgs = []
    for preds, rel in zip(pred_ids, relevant_sets):
        if not rel:
            continue
        dcg = sum(1.0 / np.log2(p + 2) for p, i in enumerate(preds[:k]) if i in rel)
        ideal = sum(1.0 / np.log2(p + 2) for p in range(min(len(rel), k)))
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
```

To contrast, the **single-stage** baseline is "retrieve more, rank none" — take the top-20 straight from the retriever's dot-product scores, no ranking MLP. The **two-stage** path is the function above. Running both on the same held-out set, with a trained two-tower retriever and a ranker trained on the retrieved candidates (a temporal split, ranker trained with a pairwise BPR-style loss on in-pool positives versus in-pool negatives), gives the kind of numbers in the results table below. The headline pattern, which holds across real datasets: the retriever alone gets you the recall ceiling; adding the ranker leaves Recall@20 essentially unchanged (it cannot exceed what retrieval surfaced — the ceiling again) but lifts NDCG@20 substantially, because the ranker re-orders the surfaced items far better than raw dot product does.

## 8. Results: single-stage vs two-stage vs three-stage

Here is the measurement that ties the post together, on MovieLens-20M (a standard 20-million-rating movie dataset, temporal split, last interactions held out). The retriever is a two-tower model; the ranker is a DCN-style network with user/item/context features; the re-ranker is MMR at $\lambda = 0.7$ over item-genre embeddings. Latency is p99 measured on a single GPU server with a warm cache. These figures are representative of what this architecture produces and are consistent with the public literature on cascade recommenders; treat the exact decimals as illustrative of the *pattern*, not as a benchmark to cite.

| Architecture | p99 latency | Recall@20 | NDCG@20 | Catalog coverage | Verdict |
|---|---|---|---|---|---|
| Single-stage, full deep scan | ~25 min (infeasible) | 1.00 (ideal) | ceiling | high | Cannot serve |
| Single-stage, popularity cap | ~8 ms | 0.31 | 0.18 | very low | Fast, terrible |
| Single-stage, retrieval only (top-20 by dot product) | ~6 ms | 0.71 | 0.33 | medium | Recall ceiling, weak order |
| Two-stage, retrieval + ranker | ~35 ms | 0.71 | 0.41 | medium | Strong default |
| Three-stage, + MMR re-rank | ~40 ms | 0.71 | 0.39 | high | Slightly lower NDCG, much better page |

Read the columns the way the funnel teaches you to. **Recall@20 is identical (0.71) for every architecture that uses the real retriever** — single-stage-retrieval-only, two-stage, and three-stage all hit 0.71 — because they all share the same retrieval stage and *retrieval sets the recall ceiling*. The ranker and re-ranker cannot raise Recall@20 above what retrieval surfaced; this is the recall-ceiling theorem showing up in a results table. **NDCG@20 jumps from 0.33 to 0.41** when you add the ranker — same items, dramatically better order — which is the whole value of stage two. **The re-ranker drops NDCG slightly (0.41 to 0.39)** while raising catalog coverage and (in an online test, not shown here) engagement — the deliberate offline-for-online trade of stage three. And the two infeasible/bad single-stage rows at the top frame why the cascade exists at all: full scan cannot serve, and the popularity cap that *can* serve fast has a Recall@20 of 0.31 — less than half the cascade — which is exactly the "everyone sees the same five creators" failure from the intro.

The matrix below summarizes the stage-count trade-off the table makes: each added stage costs a little latency, the second stage buys a large NDCG gain, and the third stage trades a little NDCG for a better page.

![A matrix comparing single-stage two-stage and three-stage architectures on p99 latency Recall@20 NDCG@20 and an overall verdict showing single-stage as infeasible two-stage as the strong default and three-stage as trading a little NDCG for a better page](/imgs/blogs/the-recommendation-funnel-retrieval-ranking-reranking-8.png)

#### Worked example: where the headroom actually is

Your two-stage system reports Recall@20 = 0.71 and NDCG@20 = 0.41, and the PM wants +10% NDCG. Two teams pitch. Team A wants to upgrade the ranker from DCN to a transformer-over-history (DIN-style), projecting +0.03 in-pool NDCG. Team B wants to add a co-visitation retriever to the existing two-tower, projecting a retrieval-recall lift from 0.71 to 0.80. Which wins? Team A's gain is real but bounded — it re-orders the *same* 0.71 of relevant items better, so its NDCG ceiling is whatever a perfect ranker scores on that fixed pool. Team B raises the recall ceiling from 0.71 to 0.80, which lifts the *maximum achievable* NDCG for every downstream model, including Team A's. From the Section-5 arithmetic, going from 7-of-10 to 8-of-10 surfaced relevant items raises the NDCG ceiling by roughly +0.07 to +0.08. **Team B's retrieval work raises the ceiling that Team A's ranking work is bumping against.** The right sequence is almost always: fix retrieval recall first (raise the ceiling), then improve ranking (approach the new ceiling). Reversing the order caps Team A's gains at the old ceiling and wastes the effort.

## 9. Measuring the funnel honestly, stage by stage

A cascade is harder to measure than a single model, and most of the "offline win that flopped online" stories in this series trace to measuring the funnel wrong. Here is the discipline, stage by stage, plus the traps.

**Measure retrieval on Recall@K against a held-out future, never on NDCG.** Use a *temporal split*: train on interactions before time $t$, evaluate on interactions after $t$. A random split leaks — an item the user interacts with "in the future" may already be in the training set's user history, inflating recall in a way that vanishes in production. And evaluate recall against the *full catalog*, not a sampled subset. The widely-cited KDD 2020 result "On Sampled Metrics for Item Recommendation" (Krichene and Rendle) showed that the common shortcut of ranking the true positive against 100 random negatives produces metrics that are *not even monotonically consistent* with the full metric — a model that looks better on sampled Recall@10 can be worse on the real thing. If you take one measurement lesson from this post: **compute retrieval metrics over the whole catalog, on a temporal split.** It is more expensive and it is the only honest number.

**Measure ranking with the candidate set held fixed.** Because NDCG is conditional on the pool (Section 5), an offline ranking experiment must feed both the baseline and the candidate ranker *the exact same retrieved candidates*. If you let each ranker re-trigger retrieval, you are measuring retrieval and ranking changes confounded together, and a ranking "win" might just be a retrieval difference. Log the candidate set per request and replay it. This is also why ranking offline experiments use *logged* candidates from production traffic — it pins the pool to what the live retriever actually produced.

**Beware position bias in your offline labels.** Your training labels are clicks, and clicks are biased by *where on the page the item appeared* — an item in slot 1 gets clicked far more than the identical item in slot 20, regardless of relevance. A ranker trained naively on click labels learns to reproduce the old ranker's position bias, not true relevance. This is the deepest reason offline NDCG can rise while online engagement is flat: the offline metric is computed on the same biased labels the model was trained on, so it rewards reproducing the bias. The fixes are inverse-propensity-scoring (weight each click by $1/p(\text{examination at that position})$) or randomized-exposure logging (the exploration slots from re-ranking double as unbiased training data). The honest offline metric for ranking is one computed on *position-debiased* labels or on a small randomized-traffic holdout.

**Measure re-ranking online, on engagement, not offline on NDCG.** Re-ranking deliberately lowers offline NDCG (Section 4), so an offline harness will always reject a good diversity change. The only valid test is an A/B experiment on online metrics — session length, day-2 retention, catalog coverage, "not interested" rate — over enough time to see the *cumulative* effect (diversity's payoff is in retention, which compounds over days, not in the single-session click rate). A re-ranking change that looks neutral on day-1 CTR can be a large retention win on week-2, which is exactly why you run feed experiments for weeks, not hours.

**Measure latency at p99, warm, under realistic load.** The SLA is a tail percentile, so report p99 (and p99.9), not the mean — the mean hides the slow tail that actually times out and shows the user a blank page. Warm up the caches and the JIT before measuring; a cold ANN index or an un-warmed GPU kernel reports latencies that never happen in steady state. And measure under realistic *concurrency*: the per-stage cost $t_s$ rises with batch contention, so a single-request benchmark understates production p99.

#### Worked example: the offline-up online-flat diagnosis

Your team ships a new ranker. Offline NDCG@20 rises from 0.41 to 0.45 on the logged set. You A/B it. Online CTR is flat, engagement is flat. What happened? Walk the funnel. First check: was the candidate set held fixed offline? It was — same logged pool for both. Second check: is retrieval the binding constraint? Recall@20 is 0.71 for both (same retriever), and the offline NDCG of 0.45 is still well under the recall ceiling, so the ranker has headroom — the ceiling is not the problem. Third check: position bias. The offline labels are raw clicks, and the new ranker happens to agree *more* with the old ranker's slot-1 picks, which inflates offline NDCG (it is rewarded for reproducing the position bias the labels encode) without changing what users actually find relevant. The tell: re-run the offline metric on the randomized-exposure holdout, and the 0.41-to-0.45 gain shrinks to 0.41-to-0.415. The "win" was mostly bias reproduction. The fix is to retrain with inverse-propensity weighting and re-evaluate on debiased labels — and to trust the online A/B, which was telling the truth all along. This exact pattern — offline metric on biased labels overstating a ranker change — is the most common false-positive in recommender development, and the funnel's structure (conditional metrics, biased labels) is why.

## 10. When to add a stage, and when to skip one

Every stage is a cost: latency, a model to train and monitor, a multiplicative tax on recall, and a team to own it. Add a stage only when the arithmetic demands it.

**Do you need retrieval at all?** Only if you cannot afford to score the whole catalog with your ranker under the SLA. The test is the brute-force inequality: if $N \cdot t_{\text{ranker}} \le B - T_{\text{fixed}}$, you can skip retrieval and rank the whole catalog. For a small catalog — a few thousand items, a niche marketplace, an internal tool — that inequality holds, and a single well-tuned ranker over everything is *simpler and strictly better* than a cascade (no recall ceiling, no blending, no extra latency). The crossover is usually somewhere around tens of thousands of items for a deep ranker; below it, skip retrieval; above it, you have no choice. Do not build a two-tower retriever for a 5,000-item catalog because a blog post said recommenders need one. They need one when the scan is infeasible, not before.

**Do you need ranking as a separate stage?** Almost always yes, the moment retrieval exists, because retrieval is recall-tuned and its in-pool order is poor (our table: 0.33 NDCG retrieval-only versus 0.41 with a ranker). The only time to skip a dedicated ranker is when the retriever's score *is* the relevance you care about and the pool is already tiny — rare in practice.

**Do you need re-ranking?** Add it when the *set* matters and per-item scoring demonstrably hurts the page: when you see top-of-feed redundancy, when diversity or coverage is a product goal, when you have business rules (ad load, fairness, dedup of seen items), or when you need exploration to fight a feedback loop. Skip it when a list of independently-best items is genuinely the right product (a "most relevant search results" page where the user wants the single best answer first). Note that *some* re-ranking — dedup of already-seen items, policy filters — is non-negotiable in any real product even if you skip the fancy diversity model; that minimal re-rank is a filter, not a learned stage, and costs almost nothing.

**When to add a *fourth* stage.** Very large systems sometimes split retrieval into "pre-ranking" (a lightweight ranker between retrieval and the heavy ranker, to cut 10,000 down to 1,000 so the heavy ranker's budget is spent on better candidates) — Alibaba's production stack famously does this. Add it only when your heavy ranker's latency forces a small $n_2$ that is hurting recall: a pre-ranker lets retrieval be more generous (higher recall ceiling) while keeping the heavy ranker's input small. It is a recall-ceiling-versus-ranker-budget trade, and you reach for it when both are binding.

The stress tests, the way a staff engineer pressure-tests the design: *What at 100M items?* Retrieval is mandatory, two-tower plus ANN, multi-source to raise the ceiling. *What with only implicit feedback (clicks, no ratings)?* Retrieval and ranking both train on implicit signals with careful negative sampling, and calibration matters more because clicks are biased by position. *What when offline NDCG rises but online engagement is flat?* Suspect the recall ceiling (you improved ranking on a pool that was already missing the good items) or position bias (your offline metric does not model where on the page the item appeared) — this is the offline-online gap the series keeps returning to. *What when the feed collapses to ten popular items?* Your re-ranking has no exploration or coverage objective, and the feedback loop has found its popularity fixed point; add exploration slots and a coverage-aware re-rank. *What when retrieval recall is high but the ranker still underperforms?* Now ranking is genuinely the bottleneck — the ceiling is not binding — and the deep-ranker upgrade is the right call. The discipline is always: *find the binding constraint, and spend effort there.*

## 11. Case studies: how real systems build the funnel

The funnel is not a theory; it is the convergent architecture of essentially every large recommender. Four worth knowing.

**YouTube — the canonical two-stage paper.** Covington, Adams, and Sargin's "Deep Neural Networks for YouTube Recommendations" (RecSys 2016) is the paper that put the retrieval-plus-ranking funnel on the map for the field. Their **candidate generation** network frames retrieval as extreme multiclass classification — predict the next watched video out of millions — and at serving time reduces to a nearest-neighbor lookup of the user vector against video vectors (the two-tower-plus-ANN pattern before "two-tower" was the standard name). It cuts millions of videos to a few hundred. Their **ranking** network then scores those few hundred with a far richer feature set (including features about the candidate's relationship to the user's history) and predicts expected watch time rather than click, precisely because click is a biased proxy. The paper is explicit that the two networks have different objectives and different metrics — candidate generation for recall, ranking for fine-grained ordering — which is the thesis of this post stated by the team that shipped it to a billion users.

**Pinterest — PinSage and graph retrieval.** Pinterest's "PinSage" (Ying et al., KDD 2018) is a graph convolutional network that generates item embeddings by aggregating over the pin-board graph, deployed for retrieval over three billion pins. The lesson for the funnel is the retrieval source: a *graph-walk* retriever surfaces candidates a pure two-tower model misses (items connected through the board graph rather than through embedding-space proximity), which raises the union recall ceiling — exactly the multi-source argument of Section 6. Pinterest then ranks the retrieved candidates downstream. PinSage is a *retrieval-stage* innovation that pays off because it lifts the ceiling.

**Instagram and TikTok — multi-stage feeds at extreme freshness.** Public engineering write-ups from Meta (Instagram Reels, the Explore ranking system) and ByteDance (the TikTok "For You" system) describe the same skeleton: many parallel retrieval sources (embedding retrieval, co-engagement, follows, freshness, trending) merged into a pool, a multi-task ranker predicting a basket of engagement events (watch-time, like, share, follow, "not interested"), and a re-ranking layer enforcing diversity, freshness, and ad/policy constraints over the final page. TikTok's documented emphasis on rapid feedback — logging granular signals and retraining frequently so new content enters the ranker's view fast — is the feedback-loop spine of this series; their heavy re-ranking for diversity and freshness is why a short-video feed feels varied rather than collapsing onto one creator. The exact numbers are not public, but the *architecture* is openly the three-stage funnel.

**Alibaba — the pre-ranking fourth stage.** Alibaba's display-advertising and Taobao recommendation papers (the DIN/DIEN line for ranking, plus their pre-ranking work like COLD) describe an explicit four-stage funnel: retrieval (matching), **pre-ranking** (a lightweight model that cuts ~10,000 to ~1,000), ranking (the heavy DIN/DIEN model with attention over user behavior), and re-ranking. The pre-ranking stage exists for exactly the recall-ceiling-versus-ranker-budget reason in the next section: it lets matching be generous (higher ceiling) while keeping the expensive ranker's input small. It is the clearest production example of *adding a stage because the binding constraint demanded it*, not for elegance.

What unites all four is not the specific models — those differ and keep changing — but the *shape*: cheap recall-oriented sources at the top, an expensive precision-oriented ranker in the middle, a set-aware re-rank at the bottom, and an obsessive feedback loop logging every interaction to retrain the whole stack. Independent teams at YouTube, Pinterest, Meta, ByteDance, and Alibaba, working on different problems with different data, converged on the same architecture because the latency arithmetic of Section 1 leaves no other option. When you see a system that does *not* look like this, it is almost always because its catalog is small enough to skip retrieval (and then it is a single ranker, the degenerate one-stage funnel) — never because someone found a way to score a hundred million items with a deep model in fifty milliseconds. That way does not exist, and the funnel is the field's collective answer to its non-existence.

## 12. When to reach for the funnel (and when not to)

A decisive section, because the funnel is the right default but not a universal law.

**Reach for the full three-stage funnel** when your catalog is large enough that brute-force scanning with a good ranker blows the SLA (the $N \cdot t_{\text{ranker}} > B$ test), *and* the set composition of the result page matters to the user or the business. That describes almost every consumer feed, marketplace, and content platform above tens of thousands of items.

**Reach for two stages only** (retrieval + ranking, minimal filter-only re-rank) when the catalog is large but the page is just "the best items, in order" with no strong diversity or business-rule requirements — many search and e-commerce result pages.

**Skip retrieval entirely** when the catalog is small enough to score whole under the SLA. A single calibrated ranker over a few thousand items is simpler, has no recall ceiling, and is strictly better. Do not pay the complexity of ANN, embedding sync, and multi-source blending for a problem that does not have it. This is the most common over-engineering mistake: building YouTube's stack for a catalog that fits in a spreadsheet.

**Do not over-invest in ranking while retrieval recall is low.** The recall ceiling makes ranking improvements on a leaky retriever nearly worthless. Measure retrieval recall first; if it is the binding constraint, fix it before touching the ranker.

**Do not ship a re-ranker tuned only on offline NDCG.** Re-ranking *intentionally* sacrifices a little NDCG for diversity and coverage; if your only metric is offline NDCG you will reject the very change that improves the user's experience. Re-ranking has to be validated online, on engagement and retention, not offline on NDCG alone.

**Do not blend retrieval sources by raw score.** Different sources are on different scales; either quota-merge (robust) or calibrate-then-merge, never concatenate-and-sort raw scores — that hands the pool to whichever source emits the biggest numbers.

## 13. Key takeaways

- **The funnel is forced, not chosen.** Brute-force scanning a large catalog with a deep ranker is six orders of magnitude too slow; the only escape is to cascade cheap-to-expensive over a shrinking candidate set. Retrieval, ranking, re-ranking falls directly out of the latency budget $T_{\text{fixed}} + \sum_s n_s t_s \le B$.
- **Each stage has a different job and a different metric.** Retrieval: cheap, high-recall, billions to thousands, graded on Recall@K. Ranking: expensive, high-precision, thousands to tens, graded on NDCG/AUC and *calibration*. Re-ranking: shape the set, tens to ten, graded on diversity, coverage, and online engagement.
- **The recall ceiling is a theorem.** End-to-end recall $\le$ retrieval recall, because a filter cannot add information. Anything retrieval drops, ranking can never recover. This caps NDCG, engagement, and revenue regardless of ranker quality.
- **Retrieval is the silent bottleneck.** Most teams over-invest in ranking; the highest-leverage move is usually raising the (union) retrieval recall ceiling via multi-source blending, because it lifts the ceiling on everything downstream.
- **Calibration is load-bearing whenever scores are combined or compared** — across multi-task heads, across blended retrieval sources, against a price or a bid. Sorting within one pool tolerates miscalibration; everything else does not.
- **The value of a page is not the sum of its items.** Re-ranking exists because per-item scoring stacks near-duplicates; MMR and set-aware re-rank trade a little NDCG for a much better page, the canonical offline-down-online-up pattern.
- **Add a stage only when the binding constraint demands it.** Small catalog: skip retrieval. Heavy ranker starving retrieval recall: add pre-ranking. No set requirement: skip the diversity re-ranker. Every stage is latency, a model to own, and a multiplicative recall tax.
- **Measure honestly.** Hold the candidate set fixed when comparing rankers (NDCG is conditional on retrieval), use a temporal split with no leakage, warm up before measuring latency, and validate re-ranking online, not on offline NDCG.

## 14. Further reading

- Covington, Adams, Sargin, **"Deep Neural Networks for YouTube Recommendations"** (RecSys 2016) — the canonical two-stage candidate-generation-plus-ranking paper; read it for the explicit different-objectives-per-stage argument.
- Ying, He, Chen, Eksombatchai, Hamilton, Leskovec, **"Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage)** (KDD 2018) — graph-based retrieval at three-billion-item scale.
- Zhou et al., **"Deep Interest Network for Click-Through Rate Prediction" (DIN)** (KDD 2018) and the DIEN follow-up — the heavy-ranker line, with attention over user behavior.
- Wang et al., **"COLD: Towards the Next Generation of Pre-Ranking System"** (Alibaba) — the explicit pre-ranking fourth stage and the recall-versus-budget trade.
- Carbonell and Goldstein, **"The Use of MMR for Diversity-Based Reranking"** (SIGIR 1998) — the original Maximal Marginal Relevance, still the re-ranking default.
- Krichene and Rendle, **"On Sampled Metrics for Item Recommendation"** (KDD 2020) — why sampled offline metrics are inconsistent with the full metric; read it before you trust any "rank against 100 negatives" evaluation.
- Rendle, Freudenthaler, Gantner, Schmidt-Thieme, **"BPR: Bayesian Personalized Ranking from Implicit Feedback"** (UAI 2009) — the pairwise ranking loss derived in Section 3 of this post.
- **faiss documentation** (`IndexFlatIP`, `IndexIVFFlat`, `IndexHNSWFlat`, `nprobe`) — the practical ANN index for the retrieval stage; pair it with [approximate nearest neighbor serving with faiss, HNSW, and ScaNN](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann).
- Within this series: start at [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), go deep on [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) and [the ranking model and CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations), and tie it all together with [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

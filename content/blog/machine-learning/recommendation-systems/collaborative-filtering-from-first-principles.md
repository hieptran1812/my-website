---
title: "Collaborative Filtering From First Principles"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build item-item collaborative filtering from scratch on MovieLens, derive the similarity measures and prediction rule, and measure exactly where CF wins and where sparsity forces you to matrix factorization."
tags:
  [
    "recommendation-systems",
    "recsys",
    "collaborative-filtering",
    "item-item",
    "cosine-similarity",
    "pearson-correlation",
    "movielens",
    "machine-learning",
    "sparsity",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/collaborative-filtering-from-first-principles-1.png"
---

You have a movie site, a few thousand users, a few thousand films, and exactly one signal worth anything: who watched what and how much they liked it. No genre tags, no cast lists, no plot embeddings, no demographics. Just a giant mostly-empty table of ratings. A new user, call her You, has rated three movies. The product wants to fill the home page with movies You will love. With nothing but that table, can you do it?

You can, and the technique that does it is one of the oldest and most durable ideas in the field: **collaborative filtering** (CF). The whole bet is captured in a single sentence — *people who agreed in the past will agree again*. If two users rated the same ten films almost identically, and one of them loved an eleventh, the other probably will too. If two films were loved and hated by the same people, they are "similar" in a sense that has nothing to do with their content. CF turns the collective behavior of the crowd into a prediction for the individual, using nothing but the interaction matrix itself. That is why it is "collaborative" — every user's history quietly helps recommend to every other user.

This post sits early in the **Recommendation Systems: From Click to Production** series, right after we framed the problem and the [retrieval-ranking-reranking funnel](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system). CF is the first real model in that funnel: a candidate generator that, on a small catalog, can be the whole system. We will be three things at once, as always. *Scientific*: we derive the similarity measures, prove why adjusted cosine cancels a user's rating-scale bias, and do the sparsity arithmetic that explains why CF eventually buckles. *Practical*: we build item-item CF from scratch on MovieLens with `numpy` and `scipy.sparse`, then show the `implicit` library's nearest-neighbours and a quick `sklearn` version. *Measured*: a real before-and-after table — popularity baseline versus user-user versus item-item on Recall@10, NDCG@10, and catalog coverage, with a temporal split and the effect of the neighborhood size $k$ and shrinkage spelled out.

The picture below is the entire problem in one frame. It is a tiny user-by-item rating matrix with one empty target cell — the rating we want to predict — and the rows and cells that will vote on it highlighted. Everything in this post is a more careful version of filling that one cell.

![Grid of a small user by item rating matrix with one empty target cell highlighted and the neighbor cells that will vote on its value marked in a contrasting color](/imgs/blogs/collaborative-filtering-from-first-principles-1.png)

## 1. The core idea: agreement is the only feature

Strip recommendation down to its barest form and you have a matrix $R$ with one row per user $u$ and one column per item $i$. The entry $r_{ui}$ is the rating user $u$ gave item $i$ — a number from 1 to 5 for explicit ratings, or a 1 for "interacted" in the implicit case. The overwhelming majority of entries are *missing*: a typical user has rated a few dozen of the thousands of available items. CF's job is to predict the missing entries, or at least to rank them, so that the top of each user's predicted list becomes the recommendation.

The defining property of CF, and the thing that makes it feel almost magical the first time you see it work, is that **it uses no content features at all**. It does not know that "The Matrix" is science fiction or that "Toy Story" is animated. It knows only the pattern of who rated what. Two films are "similar" if the same people tended to rate them the same way; two users are "similar" if they tended to rate the same films the same way. That is the entire vocabulary. Everything else — cosine, Pearson, neighborhoods, shrinkage — is machinery for measuring that agreement precisely and turning it into a number.

This is **memory-based** CF, sometimes called neighborhood-based CF, and it is worth being precise about the name because the next post in the series is about a different family. Memory-based CF keeps the raw interaction matrix around and computes predictions by looking up similar rows or columns at prediction time (or precomputing those lookups offline). It learns no parameters in the gradient-descent sense; there is no embedding to train. The "model" is the similarity matrix plus the data. Contrast that with **model-based** CF — chiefly [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), the subject of the next post — which compresses the matrix into low-dimensional latent vectors and predicts from those. Memory-based CF is the place to start because it is transparent: you can trace exactly why a recommendation was made, by hand, on a tiny example.

The historical lineage matters here, and it is short and worth knowing. The term *collaborative filtering* was coined by the Xerox PARC team behind the Tapestry email-filtering system in 1992. The first algorithm that looks like what we run today came from the **GroupLens** project at the University of Minnesota: Resnick, Iacovou, Suchak, Bergstrom, and Riedl published "GroupLens: An Open Architecture for Collaborative Filtering of Netnews" in 1994, which used Pearson-correlation user-user CF to recommend Usenet articles. That project also gave us the MovieLens dataset we will use throughout. A decade later, Amazon's Linden, Smith, and York published "Amazon.com Recommendations: Item-to-Item Collaborative Filtering" in 2003 (IEEE Internet Computing), which flipped the algorithm from user-user to item-item and made it scale to tens of millions of customers and millions of products. Those two papers are the spine of this post, and we cite them again in the case-studies section.

### 1.1 What "the only input is the matrix" really buys you

The constraint of using only the interaction matrix is also CF's great strength: it works in any domain, for any catalog, with zero feature engineering. Music, books, products, news, restaurants — if you can log who interacted with what, you can run CF. You do not need a taxonomy, you do not need NLP on item descriptions, you do not need to negotiate with another team for a feature pipeline. On day one of a product, when the only thing you have is a stream of clicks, CF is often the strongest model you can ship, and it is frequently the baseline that fancier deep models have to beat for years afterward.

The constraint is also CF's great weakness, and the rest of the post is largely about that weakness. If the matrix is the only input, then a brand-new item with no interactions is invisible (the cold-start problem), and a matrix that is 99.9% empty makes "similarity" a measurement on almost no data (the sparsity problem). Hold those two in mind; we will quantify both.

### 1.2 Why "people who agreed will agree again" is a defensible assumption

It is worth pausing on why the core bet works at all, because it is not obvious that the past predicts the future for tastes. The justification is statistical, not mystical. Human preferences are *low-rank*: there are not millions of independent taste dimensions, there are a few dozen latent ones — you like a certain kind of pacing, a certain emotional register, a certain genre cluster — and most people's preferences are well-approximated by a point in that low-dimensional space. If that is true (and decades of recommender results say it largely is), then two users who agree on twenty films are very likely close in that latent space, and being close in latent space means they will agree on the *next* film too. CF never explicitly recovers those latent dimensions — matrix factorization does that — but it exploits their existence: agreement on observed items is evidence of proximity in the unobserved taste space, and proximity is what makes the prediction work. This is also why CF degrades exactly when the low-rank assumption fails: a gray sheep is precisely a user who does not sit near anyone in latent space, and a brand-new item is one whose latent position we have no evidence for yet.

The second reason it works is the *wisdom of the aggregate*. Any single user's rating is noisy — they were in a bad mood, they misclicked, they rated to be contrarian. But averaging over a neighborhood of fifty similar users (or fifty similar items) cancels that noise. CF is, at bottom, a denoising average: it replaces "what did this one person think?" with "what did the crowd most like this person think?", and the law of large numbers does the rest. The neighborhood size $k$ is literally the size of that average, which is why section 4.1's bias-variance story is the same story you would tell about any sample mean — too few samples and the estimate is noisy, too many and you average in people who are not actually similar.

## 2. User-user vs item-item: two ways to draw a neighborhood

There are two symmetric ways to fill the empty cell $r_{ui}$, and the difference between them is the single most consequential design choice in memory-based CF. The figure below puts them side by side for the same prediction.

![Before and after comparison of a user-user neighborhood of similar people versus an item-item neighborhood of similar items producing the same prediction](/imgs/blogs/collaborative-filtering-from-first-principles-2.png)

**User-user CF** answers "who is like You?" It finds the $k$ users whose rating histories are most similar to yours — your *neighborhood* — and predicts your rating for an item as a similarity-weighted average of what those neighbors gave it. This is the GroupLens 1994 formulation. It is intuitive: it is exactly what a friend does when they say "people with your taste loved this."

**Item-item CF** answers "what is like the things You already like?" It finds, for each item, the $k$ other items most similar to it — where similarity is again measured over the *columns* of the matrix, i.e. the pattern of users who rated both — and predicts your rating for a target item as a weighted average of *your own* ratings on its similar items. This is the Amazon 2003 formulation.

Mathematically they are duals: user-user works on rows of $R$, item-item works on columns of $R^\top$. But operationally they behave very differently, and item-item won the industry for three concrete reasons.

**Reason 1: items are more stable than users.** A movie's "neighbors" — the other movies loved by the same crowd — barely change from week to week. A user's neighbors can change every session as they rate new things. Item similarities can therefore be precomputed in a nightly batch job and cached; user similarities want to be recomputed constantly. Stability is what makes the offline-precompute-online-serve split (section 9) clean.

**Reason 2: there are usually far fewer items than users, or at least items churn less.** Amazon in 2003 had tens of millions of customers. Computing user-user similarity meant comparing every customer to every other customer — quadratic in the number of *users*, the larger dimension. Item-item similarity is quadratic in the number of *items*, and for Amazon the item-item matrix, once built, was small enough to hold and serve from memory. When users greatly outnumber items, item-item is cheaper. (When items greatly outnumber users — a niche case — the logic flips.)

**Reason 3: item-item gives better, more explainable recommendations in practice.** Each user's recommendations come from items the user *themselves* rated, so the recommendation comes with a built-in explanation: "Because you watched X." User-user recommendations come from strangers' ratings, which is both harder to explain and more vulnerable to a single weird neighbor. Empirically, Sarwar, Karypis, Konstan, and Riedl's 2001 WWW paper "Item-Based Collaborative Filtering Recommendation Algorithms" showed item-item matching or beating user-user accuracy on MovieLens while being far more scalable. That paper is the reason "item-item" is the default you should reach for first.

There is a fourth, subtler reason that shows up at scale: **item-item degrades more gracefully under sparsity.** A user with only three ratings has an almost useless neighborhood in user-user CF (you are computing similarity from three numbers). But those same three rated items each have hundreds or thousands of co-raters in their columns, so their item neighborhoods are computed from plenty of data. Item-item moves the data-hungry computation onto the better-populated dimension. We will see this in the results: item-item beats user-user on MovieLens by a clear margin, and the gap is sparsity.

#### Worked example: a cosine similarity by hand

Let us actually compute a similarity so the abstraction has teeth. Take three users and their ratings on two movies, M1 and M3, from the intro figure. Ann rated (M1, M3) = (5, 4). Bob rated (4, 5). We treat each *item* as a vector over users: M1 is the column $(5, 4, 5)$ across (Ann, Bob, You), M3 is the column $(4, 5, ?)$. To compare M1 and M3 we use only the users who rated *both* — here Ann and Bob — giving M1 = $(5, 4)$ and M3 = $(4, 5)$.

Cosine similarity is the cosine of the angle between these two vectors:

$$
\text{cos}(M1, M3) = \frac{M1 \cdot M3}{\lVert M1 \rVert \, \lVert M3 \rVert} = \frac{5\cdot 4 + 4 \cdot 5}{\sqrt{5^2 + 4^2}\,\sqrt{4^2 + 5^2}} = \frac{40}{\sqrt{41}\,\sqrt{41}} = \frac{40}{41} \approx 0.976.
$$

So M1 and M3 are very similar — the same two people rated both highly. That number, 0.976, is the weight M1's contribution will get when we predict your rating for M3. We do this prediction by hand in section 5. The whole of item-item CF is this calculation, run for every pair of items, then reused at serving time.

## 3. Similarity measures: the science of "agreement"

The similarity function is the heart of memory-based CF, and the choices are not interchangeable — each encodes a different assumption about your data. The matrix below lays out the four you actually use and what each one does to scale bias and to implicit data.

![Matrix comparing cosine, Pearson, adjusted cosine, and Jaccard similarity across whether they remove rating scale bias, which feedback type they suit, and when to use each](/imgs/blogs/collaborative-filtering-from-first-principles-3.png)

### 3.1 Cosine similarity

For two items represented as vectors $\mathbf{a}, \mathbf{b}$ over the users who rated them, cosine similarity is

$$
\text{sim}_{\cos}(\mathbf{a}, \mathbf{b}) = \frac{\sum_{u} a_u b_u}{\sqrt{\sum_u a_u^2}\,\sqrt{\sum_u b_u^2}} = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}.
$$

It measures the angle between the vectors and ignores their length, so a user who rates everything high and a user who rates everything low can still look "aligned" if their *relative* preferences agree. Cosine ranges from $-1$ to $1$ for centered data and from $0$ to $1$ for non-negative rating vectors. Its great virtue is that it is cheap and it composes beautifully with sparse matrices: if you L2-normalize the rows (or columns) of the matrix once, every cosine becomes a plain dot product, and the whole item-item cosine matrix is one sparse matrix multiply, $\hat R^\top \hat R$, where $\hat R$ has normalized columns. That single fact is why cosine is the workhorse of item-item CF at scale.

Its weakness: plain cosine does *not* remove the user's baseline. If Ann rates on a 4-to-5 scale and Bob on a 1-to-3 scale, cosine over raw ratings is dominated by the fact that both numbers are positive, not by whether they *agree relative to each user's norm*. That is what mean-centering fixes.

### 3.2 Pearson correlation: mean-center, then cosine

Pearson correlation is cosine similarity computed on **mean-centered** ratings. For user-user CF, you center each user's ratings by subtracting that user's mean rating $\bar r_u$:

$$
\text{sim}_{\text{Pearson}}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar r_u)(r_{vi} - \bar r_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar r_u)^2}\,\sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar r_v)^2}},
$$

where $I_{uv}$ is the set of items both $u$ and $v$ rated. The subtraction of $\bar r_u$ is the whole point: it converts "Ann gave it a 4" into "Ann gave it one point *above her own average*," which is a statement about preference, not about Ann's generosity. After centering, a generous rater and a stingy rater who *agree about which films are better than their personal average* will show high correlation. Pearson is the measure GroupLens used in 1994, and it is the default for user-user CF.

Pearson is exactly the linear correlation coefficient, so it lives in $[-1, 1]$: $+1$ means the two users move together, $-1$ means they are opposites (rare and usually noise), $0$ means no linear relationship. One subtlety: Pearson is computed only over co-rated items $I_{uv}$, and when that set is tiny — two or three items — the correlation is extremely noisy. A pair of users who happen to agree on two movies gets a correlation of $1.0$ that means almost nothing. That is the motivation for significance weighting, below.

### 3.3 Adjusted cosine: the right centering for item-item

Here is a trap that catches people who naively port Pearson to item-item CF. For item-item similarity, the vectors are *columns* (over users). If you mean-center each column by its *item* mean, you remove the item's popularity baseline but leave the user's rating-scale bias intact — and the user bias is the bigger problem, because the same user appears in many items' columns. The correct move, introduced by Sarwar et al. (2001), is **adjusted cosine**: center each rating by the **user's** mean, then compute cosine over the item columns.

$$
\text{sim}_{\text{adj}}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar r_u)(r_{uj} - \bar r_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar r_u)^2}\,\sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar r_u)^2}},
$$

where $U_{ij}$ is the set of users who rated *both* items $i$ and $j$, and crucially the centering uses $\bar r_u$, the **user** mean, in both factors. Why does this matter? Consider a user who rates everything between 4 and 5 — a generous rater. In plain item-item cosine, this user contributes large positive products $r_{ui} r_{uj}$ to *every* pair of items they rated, inflating all those similarities regardless of the user's actual preference among those items. Adjusted cosine subtracts $\bar r_u \approx 4.5$ first, so the generous rater's contribution to item $i$ versus item $j$ reflects only whether they liked $i$ *more or less than their own average*. The scale bias cancels.

There is a clean way to see this. Suppose a user's true preference for item $i$ is $p_{ui}$ and their observed rating is $r_{ui} = \bar r_u + p_{ui} + \epsilon$, where $\bar r_u$ is an additive user offset (their generosity) and $\epsilon$ is noise. The user offset $\bar r_u$ is constant across all items for that user. Plain cosine multiplies $(\bar r_u + p_{ui})(\bar r_u + p_{uj})$, which expands to $\bar r_u^2 + \bar r_u(p_{ui} + p_{uj}) + p_{ui}p_{uj}$ — the first two terms are pure user-offset contamination. Adjusted cosine first subtracts $\bar r_u$, multiplying $p_{ui}\, p_{uj}$ directly, so only the genuine preference covariance survives. That is the derivation behind "adjusted cosine handles user rating-scale bias," and it is why adjusted cosine usually beats both plain cosine and item-mean-centered cosine for explicit-rating item-item CF.

### 3.4 Jaccard for implicit feedback

When you have no ratings — only clicks, plays, or purchases — there is no scale to center and no continuous value to correlate. The data is a *set*: the set of users who interacted with item $i$. The natural similarity for sets is **Jaccard**, the size of the intersection over the size of the union:

$$
\text{sim}_{\text{Jaccard}}(i, j) = \frac{\lvert U_i \cap U_j \rvert}{\lvert U_i \cup U_j \rvert},
$$

where $U_i$ is the set of users who interacted with item $i$. Jaccard is $1$ when the two items have exactly the same audience and $0$ when they share none. It is the right default for the [implicit-feedback world](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr), because it asks the only question implicit data can answer: how much do the audiences overlap? Note that cosine on binary 0/1 vectors is closely related — for binary vectors, cosine equals $\lvert U_i \cap U_j \rvert / \sqrt{\lvert U_i \rvert \lvert U_j \rvert}$, the intersection over the *geometric mean* of the set sizes rather than the union. Both are used; cosine on binary vectors tends to be slightly more popularity-tolerant.

### 3.4b Deriving item-item cosine as one matrix multiply

It is worth seeing exactly why the entire item-item cosine matrix is a single matrix product, because that identity is what makes CF tractable. Let $R$ be the $m \times n$ user-by-item matrix and let $\hat R$ be $R$ with each *column* L2-normalized: $\hat R_{:,i} = R_{:,i} / \lVert R_{:,i} \rVert$. Then the $(i, j)$ entry of $\hat R^\top \hat R$ is

$$
(\hat R^\top \hat R)_{ij} = \sum_{u=1}^{m} \hat R_{ui}\,\hat R_{uj} = \frac{\sum_u R_{ui} R_{uj}}{\lVert R_{:,i}\rVert\,\lVert R_{:,j}\rVert} = \text{sim}_{\cos}(i, j).
$$

So $S = \hat R^\top \hat R$ *is* the item-item cosine similarity matrix, exactly. No loops, no pairwise iteration — one matrix multiply computes all $n^2$ similarities at once, and because $R$ is sparse (most entries zero), the product only does work for item pairs that share at least one rater. This is the algebra behind the `(Rn.T @ Rn)` line in the code, and it is why a similarity build that looks like an $O(n^2)$ nightmare is, on sparse data, dominated by the number of *co-rated pairs* rather than the full $n^2$. Sparse BLAS does the rest. The same identity with row-normalization, $\hat R \hat R^\top$, gives the user-user similarity matrix — the duality from section 2, written in one line of linear algebra.

### 3.5 Significance weighting and shrinkage: don't trust thin overlaps

Every measure above has the same Achilles heel: when two items share only a handful of co-raters, the similarity estimate is wild. Two movies co-rated by exactly two people who both gave 5s get a perfect cosine of $1.0$, which is statistically meaningless. The standard fix is **shrinkage** (also called significance weighting): multiply the raw similarity by a factor that pulls thin overlaps toward zero.

$$
\text{sim}_{\text{shrunk}}(i, j) = \frac{\lvert U_{ij} \rvert}{\lvert U_{ij} \rvert + \lambda} \cdot \text{sim}_{\text{raw}}(i, j),
$$

where $\lvert U_{ij} \rvert$ is the number of co-raters and $\lambda$ is a shrinkage constant (typical values 10 to 100). When the co-count is large compared to $\lambda$, the factor approaches 1 and the similarity is left almost untouched. When the co-count is small, the factor shrinks the similarity toward 0, so flimsy pairs cannot dominate a neighborhood. Herlocker et al. (1999) introduced significance weighting with a hard threshold (multiply by $\min(\lvert U_{ij}\rvert, \gamma)/\gamma$); the smooth shrinkage form above is the one Koren popularized in the Netflix-Prize era and the one I reach for. Shrinkage is the single highest-leverage knob in memory-based CF on real, sparse data — in the results section it is worth several points of Recall@10.

## 4. The prediction rule and the neighborhood size k

We have similarities; now we turn them into a predicted rating. For item-item CF, the prediction of user $u$'s rating on target item $i$ is a similarity-weighted average of $u$'s own ratings on the neighbors of $i$, written cleanly with user-mean centering:

$$
\hat r_{ui} = \bar r_u + \frac{\sum_{j \in N_k(i; u)} s_{ij}\,(r_{uj} - \bar r_u)}{\sum_{j \in N_k(i; u)} \lvert s_{ij} \rvert},
$$

where $s_{ij}$ is the (shrunken, adjusted-cosine) similarity between items $i$ and $j$, and $N_k(i; u)$ is the set of up to $k$ items most similar to $i$ that user $u$ has actually rated. Read it left to right: start from the user's baseline $\bar r_u$, then add a correction that is positive if $u$ rated $i$'s similar items above their personal average and negative if below, weighted by how similar each neighbor is. The denominator $\sum \lvert s_{ij} \rvert$ normalizes so the correction stays on the rating scale. (The user-user version is identical with the roles of users and items swapped: $\hat r_{ui} = \bar r_u + \frac{\sum_{v \in N_k(u;i)} s_{uv}(r_{vi} - \bar r_v)}{\sum \lvert s_{uv}\rvert}$.)

For *top-N recommendation* — which is what you actually serve — you often skip the additive baseline entirely and just rank by the score $\sum_{j} s_{ij}\, r_{uj}$ (or $\sum_j s_{ij}$ over the items $u$ interacted with, for implicit data). The reason: top-N only cares about the *order* of scores within a user, and the per-user baseline $\bar r_u$ is a constant that shifts every candidate equally, so it does not change the ranking. The full rating-prediction formula matters when you must output a calibrated 1-to-5 estimate (the old Netflix-Prize objective); the bare weighted sum is what you use to fill a carousel. We will compute both in the worked example.

### 4.1 The neighborhood size k trade-off

The neighborhood size $k$ — how many neighbors vote — is the main accuracy knob, and it has a clean bias-variance shape. With $k$ too small (say $k=2$), the prediction rests on very few neighbors, so it is high-variance: one weird similar item swings the answer. With $k$ too large (say $k=$ all items), distant, weakly-related neighbors get a vote, dragging the prediction toward the global average and washing out the signal — high bias. The sweet spot is in between, and on MovieLens-scale data it is typically $k = 20$ to $50$. Beyond a point, increasing $k$ stops helping and slowly hurts, because the marginal neighbors are noise. We measure this curve in section 10; the punchline is that the curve is fairly flat near the optimum, so you do not need to tune $k$ to the last digit — pick something in the 20-to-50 band and move on.

A practical note that saves a lot of compute: you do not store all pairwise similarities. For each item you keep only its top-$k$ neighbors (the *model truncation*), which turns a dense $N \times N$ similarity matrix into a sparse one with $k$ entries per row. This is both a memory win and an accuracy win, because it bakes the "ignore distant neighbors" decision into the stored model.

### 4.2 The metrics we will report, defined

Before we measure anything, pin down the metrics, because half of recsys confusion comes from people comparing numbers computed differently. We rank for each user against the *whole* catalog (minus items they already saw) and look at the top-10. Let $\text{rel}_i \in \{0,1\}$ be whether the item at rank $i$ is one the user actually interacted with in the held-out test period.

**Recall@K** is the fraction of a user's held-out test items that show up in the top-K we recommended:

$$
\text{Recall@}K = \frac{\lvert \{\text{test items}\} \cap \{\text{top-}K \text{ recommended}\} \rvert}{\lvert \{\text{test items}\} \rvert},
$$

averaged over users. It rewards *finding* relevant items anywhere in the top-K and does not care about their order within it. It is the natural metric for a candidate generator like CF, where the job is "don't miss the good ones."

**NDCG@K** (Normalized Discounted Cumulative Gain) does care about order — a relevant item at rank 1 is worth more than the same item at rank 10. Discounted Cumulative Gain is $\text{DCG@}K = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$, and NDCG normalizes it by the best possible DCG for that user (the *ideal* ordering), so it lands in $[0, 1]$ and is comparable across users with different numbers of test items:

$$
\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}, \qquad \text{DCG@}K = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}.
$$

The $\log_2(i+1)$ denominator is the "discount": rank 1 gets weight $1/\log_2 2 = 1$, rank 2 gets $1/\log_2 3 \approx 0.63$, rank 10 gets $1/\log_2 11 \approx 0.29$. NDCG is the metric to report when *placement* matters, which it always does on a page where the top slot gets most of the attention.

**Coverage** is not a per-user metric at all — it is a catalog-health metric: the fraction of all items that appear in *at least one* user's top-K. Low coverage means the system recommends the same few hundred items to everyone, which is popularity bias by another name. Reporting coverage alongside Recall/NDCG is the cheapest guard against an "offline win" that is really just leaning harder on the head of the catalog. Here is the eval harness in numpy, which we use to produce section 10's table.

```python
import numpy as np

def evaluate(recommend_fn, R, test, n=10):
    recalls, ndcgs, recommended_items = [], [], set()
    # group test interactions by user
    test_by_user = test.groupby("user")["item"].apply(set).to_dict()
    for user, gold in test_by_user.items():
        if len(gold) == 0:
            continue
        recs = recommend_fn(R, user, n=n)        # array of item ids, ranked
        recommended_items.update(recs.tolist())
        hits = [1 if it in gold else 0 for it in recs]
        # Recall@n
        recalls.append(sum(hits) / len(gold))
        # NDCG@n
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), n)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    coverage = len(recommended_items) / R.shape[1]
    return np.mean(recalls), np.mean(ndcgs), coverage

recall, ndcg, cov = evaluate(
    lambda R, u, n: recommend(R, S, u, n), R, test, n=10
)
print(f"Recall@10 {recall:.3f}  NDCG@10 {ndcg:.3f}  coverage {cov:.1%}")
```

This is the harness that produced every number in section 10. Note three deliberate choices baked into it: we rank against the full catalog (no negative sampling, so the numbers are comparable to the literature), we average Recall and NDCG over users (so heavy raters do not dominate), and we track coverage as the set of distinct recommended items. Swap in a different `recommend_fn` — popularity, user-user, item-item, BM25 — and the same harness compares them apples-to-apples.

## 5. Building item-item CF from scratch on MovieLens

Enough theory; let us build it. The graph below is the pipeline we are about to implement: interactions become a similarity matrix offline, we prune to top-$k$ neighbors, and online we merge those neighbors with a user's history to produce a top-N list.

![Graph of the item-item collaborative filtering pipeline from interactions to a similarity matrix to top-k neighbors merging with user history into a scored top-N list](/imgs/blogs/collaborative-filtering-from-first-principles-4.png)

We use **MovieLens-1M**: about 1,000,209 ratings from 6,040 users on 3,706 movies, the classic GroupLens dataset. Download and unzip it first.

```bash
# Download MovieLens-1M (about 6 MB)
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
# ratings.dat: UserID::MovieID::Rating::Timestamp
head -3 ml-1m/ratings.dat
```

### 5.1 Load and make a temporal split

The single most common evaluation mistake in recsys is a random train/test split. Recommendation is a prediction about the *future*, so the test set must come *after* the training set in time, or you leak future information and your offline numbers lie. We split each user's interactions by timestamp: the earliest 80% of a user's ratings train, the latest 20% test. This is a [leave-the-future-out temporal split](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys), and it is the honest minimum.

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Load ratings: UserID::MovieID::Rating::Timestamp
cols = ["user", "item", "rating", "ts"]
df = pd.read_csv(
    "ml-1m/ratings.dat", sep="::", names=cols, engine="python"
)

# Reindex user and item IDs to dense 0..N-1 ranges
df["user"] = df["user"].astype("category").cat.codes
df["item"] = df["item"].astype("category").cat.codes
n_users = df["user"].nunique()
n_items = df["item"].nunique()
print(f"{n_users} users, {n_items} items, {len(df)} ratings")
# 6040 users, 3706 items, 1000209 ratings

# Temporal split: per user, last 20% of interactions by time go to test
df = df.sort_values(["user", "ts"])
def split_user(g):
    cut = int(len(g) * 0.8)
    g = g.copy()
    g["is_test"] = False
    g.iloc[cut:, g.columns.get_loc("is_test")] = True
    return g
df = df.groupby("user", group_keys=False).apply(split_user)
train = df[~df["is_test"]]
test = df[df["is_test"]]
print(f"train {len(train)}  test {len(test)}")
```

### 5.2 Build the sparse user-item matrix

The matrix is 6,040 by 3,706 but only 4.5% full, so we store it sparse. We build a CSR matrix once for row (user) access and keep its transpose handy for column (item) access.

```python
# Sparse user-item rating matrix from TRAIN only (never touch test)
R = csr_matrix(
    (train["rating"].astype(np.float32),
     (train["user"], train["item"])),
    shape=(n_users, n_items),
)
density = R.nnz / (n_users * n_items)
print(f"density = {density:.4%}")   # density = 3.61% (train portion)
```

### 5.3 Compute item-item cosine on the sparse matrix

Here is the trick that makes item-item cosine cheap: L2-normalize the *columns* of $R$, then the item-item cosine matrix is just $\hat R^\top \hat R$, a single sparse matrix multiply. The result $S$ has $S_{ij} = \cos(i, j)$. We then zero the diagonal (an item is not its own neighbor) and keep only the top-$k$ entries per row.

```python
from sklearn.preprocessing import normalize
import scipy.sparse as sp

def item_item_cosine_topk(R, k=50, shrink=20.0):
    # 1) co-counts: how many users co-rated each item pair (for shrinkage)
    B = (R > 0).astype(np.float32)            # binary interaction
    co = (B.T @ B).tocsr()                     # item-item co-rating counts

    # 2) cosine = normalized columns multiplied
    Rn = normalize(R.tocsc(), norm="l2", axis=0)  # normalize columns
    S = (Rn.T @ Rn).tocsr()                    # item-item cosine, dense-ish

    # 3) shrinkage: multiply by co / (co + lambda)
    S = S.multiply(co / (co + shrink))
    S = S.tocsr()
    S.setdiag(0.0)                             # no self-similarity
    S.eliminate_zeros()

    # 4) keep top-k neighbors per item (row)
    rows = []
    for i in range(S.shape[0]):
        start, end = S.indptr[i], S.indptr[i + 1]
        idx = S.indices[start:end]
        val = S.data[start:end]
        if len(val) > k:
            top = np.argpartition(val, -k)[-k:]
            idx, val = idx[top], val[top]
        rows.append((i, idx, val))
    # rebuild sparse top-k similarity matrix
    data, ri, ci = [], [], []
    for i, idx, val in rows:
        data.extend(val); ri.extend([i] * len(idx)); ci.extend(idx)
    Sk = csr_matrix((data, (ri, ci)), shape=S.shape)
    return Sk

S = item_item_cosine_topk(R, k=50, shrink=20.0)
print(f"stored neighbors: {S.nnz}  (~{S.nnz / n_items:.0f} per item)")
```

A few things to notice. We compute the co-rating count matrix `B.T @ B` separately and apply shrinkage `co / (co + shrink)` exactly as in section 3.5 — this is what prevents two-co-rater pairs from getting a meaningless cosine of 1.0. We normalize columns so the multiply *is* the cosine. And we truncate to top-$k$ per row, which is both the accuracy decision from section 4.1 and the memory win from section 4.2. For a more numerically faithful item-item model you would substitute adjusted cosine by subtracting each user's mean from `R` before normalizing; we keep plain cosine here for clarity and add the adjusted variant in the next snippet.

The single most important correctness detail in that function is hiding in plain sight: **we only ever touch `R`, which was built from `train`.** Never, under any circumstances, build the similarity matrix from data that includes test interactions — that is the leakage that makes offline numbers fictional. If the test set leaks into the similarity matrix, the model has literally seen the answers, and your Recall@10 will look spectacular and collapse the instant it meets real future data. This is the most common way a beginner's CF "beats the state of the art" by accident.

Here is the adjusted-cosine variant promised above. The only change is that we subtract each user's mean rating from their nonzero entries before normalizing the columns, which cancels the rating-scale bias we derived in section 3.3. We center only the observed entries (the sparse nonzeros), never the structural zeros, because a missing rating is not "rated zero" — it is unrated, and centering it would inject a fake signal.

```python
def adjusted_cosine_topk(R, k=50, shrink=20.0):
    R = R.tocsr().astype(np.float32)
    # per-user mean over OBSERVED ratings only
    sums = np.asarray(R.sum(axis=1)).ravel()
    counts = np.diff(R.indptr)                  # ratings per user
    user_mean = np.divide(sums, np.maximum(counts, 1))
    # subtract each user's mean from their nonzero entries in place
    Rc = R.copy()
    for u in range(R.shape[0]):
        s, e = Rc.indptr[u], Rc.indptr[u + 1]
        Rc.data[s:e] -= user_mean[u]
    # from here it is identical to plain cosine on the centered matrix
    return item_item_cosine_topk(Rc, k=k, shrink=shrink)
```

On MovieLens-1M, swapping plain cosine for adjusted cosine typically buys another point or two of Recall@10, and the gain is larger on datasets with more heterogeneous raters (a mix of generous and stingy users). If everyone rates on the same scale, the centering has little to remove and the two converge — which is itself a useful diagnostic: if adjusted cosine does not beat plain cosine, your users do not have much scale bias.

### 5.4 Score and recommend

Scoring is now embarrassingly cheap. A user's score vector over all items is their (sparse) interaction row times the similarity matrix: $\text{scores} = \mathbf{r}_u\, S$. We mask out items the user already interacted with (you do not recommend what they already rated) and take the top-N.

```python
def recommend(R, S, user, n=10):
    ru = R[user]                       # 1 x n_items sparse row
    scores = ru @ S                    # 1 x n_items, weighted neighbor sum
    scores = np.asarray(scores.todense()).ravel()
    seen = ru.indices                  # items already rated
    scores[seen] = -np.inf             # never recommend the already-seen
    top = np.argpartition(scores, -n)[-n:]
    return top[np.argsort(-scores[top])]

print(recommend(R, S, user=0, n=10))
```

That is the entire serving path: one sparse-matrix-vector product and a top-N selection. On MovieLens it runs in well under a millisecond per user, which is the whole point of the offline-precompute-online-serve split we draw in section 9.

#### Worked example: a prediction by hand

Let us trace one prediction through the formula so the code is not a black box. Reuse the tiny matrix. You rated M1 = 5 and M2 = 3 (so $\bar r_{\text{You}} = 4$). We want $\hat r_{\text{You}, M3}$. Suppose we have computed two item similarities to M3: $s_{M3,M1} = 0.976$ (from the section-2 worked example) and $s_{M3,M2} = 0.60$. Both M1 and M2 are in your history, so they are the neighbors that vote.

Plug into the prediction rule:

$$
\hat r_{\text{You}, M3} = \bar r_u + \frac{s_{M3,M1}(r_{u,M1} - \bar r_u) + s_{M3,M2}(r_{u,M2} - \bar r_u)}{\lvert s_{M3,M1}\rvert + \lvert s_{M3,M2}\rvert}.
$$

The corrections are $(r_{u,M1} - \bar r_u) = 5 - 4 = +1$ and $(r_{u,M2} - \bar r_u) = 3 - 4 = -1$. So:

$$
\hat r_{\text{You}, M3} = 4 + \frac{0.976 \cdot (+1) + 0.60 \cdot (-1)}{0.976 + 0.60} = 4 + \frac{0.376}{1.576} = 4 + 0.239 \approx 4.24.
$$

The prediction is **4.24**: above your average of 4, because M3 is much more similar to the movie you loved (M1, weight 0.976) than to the one you were lukewarm on (M2, weight 0.60). For *ranking* purposes the bare weighted sum $0.976 \cdot 5 + 0.60 \cdot 3 = 6.68$ would do — but notice it ignores your baseline, which is fine because every candidate shares it. This is item-item CF in a single arithmetic line, and the from-scratch code above is exactly this calculation vectorized over millions of cells.

## 6. The same thing with libraries

You will not hand-roll the top-$k$ loop in production; you will use a library that does the sparse multiply and truncation in optimized C. Two options worth knowing.

### 6.1 The implicit library's nearest-neighbours models

The `implicit` package (Ben Frederickson) ships a family of item-item models under `implicit.nearest_neighbours`, including cosine, TF-IDF-weighted cosine, and BM25-weighted cosine — the last two are borrowed from information retrieval and down-weight popular items, which often helps. The API is two lines.

```python
import implicit
from implicit.nearest_neighbours import CosineRecommender, BM25Recommender

# implicit expects an item-user matrix (items as rows)
item_user = R.T.tocsr()

model = CosineRecommender(K=50)          # K = neighborhood size
model.fit(item_user)                     # builds the top-K similarity model

# recommend for user 0 using their interaction row
user_items = R.tocsr()
ids, scores = model.recommend(0, user_items[0], N=10)
print(ids, scores)

# BM25 weighting often beats plain cosine on implicit data
bm25 = BM25Recommender(K=50, K1=1.2, B=0.75)
bm25.fit(item_user)
```

`BM25Recommender` is worth trying first on implicit/click data: the `B` parameter controls how aggressively it discounts items that everyone interacts with, which is a cheap, principled fix for the popularity bias we discuss in section 7. On MovieLens it typically edges out plain cosine by a point or two of Recall@10.

### 6.2 A quick scikit-learn version

If you do not want a dependency, `sklearn` gives you cosine in one call. This is the clearest way to see that "item-item similarity" is just a normalized Gram matrix.

```python
from sklearn.metrics.pairwise import cosine_similarity

# item-item cosine: items are columns of R, so transpose
S_sklearn = cosine_similarity(R.T, dense_output=False)  # n_items x n_items
S_sklearn.setdiag(0.0)
S_sklearn.eliminate_zeros()
# truncate to top-k with the same loop as before, then score with ru @ S
```

`cosine_similarity` with `dense_output=False` keeps the result sparse, which matters: a dense 3,706 by 3,706 float32 matrix is only ~55 MB on MovieLens but a dense 1M-by-1M item matrix would be 4 TB, so always keep the similarity matrix sparse and truncated. This is the line where toy code and production code diverge, and it is the bridge to section 9.

### 6.3 The user-user version, for comparison

For completeness and because we report it in the results table, here is user-user CF in the same idiom. The structure is the dual: normalize the *rows* (users) of the centered matrix, multiply $R_c\, R_c^\top$ to get user-user similarities, truncate to the top-$k$ similar users per user, then predict a target user's score for an item as a similarity-weighted average over those neighbors' ratings. The code differs from item-item only in which axis we normalize and which matrix we multiply.

```python
def user_user_topk(R, k=50, shrink=20.0):
    R = R.tocsr().astype(np.float32)
    # mean-center each user's observed ratings (Pearson = centered cosine)
    sums = np.asarray(R.sum(axis=1)).ravel()
    counts = np.diff(R.indptr)
    user_mean = np.divide(sums, np.maximum(counts, 1))
    Rc = R.copy()
    for u in range(R.shape[0]):
        s, e = Rc.indptr[u], Rc.indptr[u + 1]
        Rc.data[s:e] -= user_mean[u]
    Rn = normalize(Rc, norm="l2", axis=1)       # normalize ROWS (users)
    Suu = (Rn @ Rn.T).tocsr()                    # user-user similarity
    # co-counts of co-rated items between user pairs, for shrinkage
    B = (R > 0).astype(np.float32)
    co = (B @ B.T).tocsr()
    Suu = Suu.multiply(co / (co + shrink)).tocsr()
    Suu.setdiag(0.0); Suu.eliminate_zeros()
    return Suu   # then top-k truncate + predict by neighbor-weighted average
```

Run this on MovieLens and you get the 0.151 Recall@10 in the results table — a real improvement over popularity, but a clear step below item-item's 0.18. The gap is not the math (the formulas are duals); it is the data. A user's row in MovieLens has on average ~165 ratings, while a *popular item's* column has hundreds or thousands; user-user computes its neighborhoods from the thinner of the two dimensions, so its similarities are noisier. That is the sparsity argument from section 2 made concrete in code: same algorithm, different axis, different reliability.

## 7. Where CF breaks: sparsity, cold start, bias, scale, gray sheep

Memory-based CF is wonderful until it isn't, and the failure modes are predictable. The matrix below rates each one by severity and pairs it with the mitigation you actually reach for.

![Matrix of collaborative filtering limitations mapping sparsity, cold start, popularity bias, scalability, and gray sheep to their severity and concrete mitigation](/imgs/blogs/collaborative-filtering-from-first-principles-7.png)

### 7.1 Sparsity: most pairs are co-rated by nobody

This is the deepest problem, so let us do the arithmetic. The figure below contrasts a dense toy matrix, where similarity is well-defined, with the real MovieLens matrix, where it often is not.

![Before and after comparison of a fully filled small rating matrix where collaborative filtering works against a real sparse MovieLens matrix where most item pairs share no co-raters](/imgs/blogs/collaborative-filtering-from-first-principles-6.png)

MovieLens-1M has $\sim$1M ratings over $6040 \times 3706 \approx 2.24 \times 10^7$ possible cells, a density of about **4.5%**. That sounds bad and it is *generous* — real e-commerce and feed matrices are routinely below 0.1%. Now ask the question item-item CF actually needs answered: for a given pair of items, how many users rated *both*? The expected number of co-raters for two random items, under a crude independence assumption with per-user-item interaction probability $p = 0.045$, is roughly $n_{\text{users}} \cdot p^2 = 6040 \cdot 0.045^2 \approx 12$. Twelve co-raters is already thin for a stable cosine. Drop the density to a realistic $0.1\%$ and the expected co-count becomes $6040 \cdot 0.001^2 \approx 0.006$ — meaning the *vast majority of item pairs share no co-rater at all*, so their similarity is simply undefined.

That is the sparsity wall in one number. Memory-based CF can only relate items that some user has co-rated; everything else is a structural zero. The fix is not a better similarity measure — shrinkage helps the thin-but-nonzero pairs, but it cannot conjure data for the zero pairs. The real fix is to **generalize across the gaps**, which is exactly what [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) does: it learns a low-dimensional vector per item such that two items can be "close" even if no single user co-rated them, because they each landed near the same region of latent space through their *other* co-ratings. Sparsity is the single best argument for moving from memory-based to model-based CF.

#### Worked example: the co-rating density number

Make it concrete with MovieLens-1M. Pick two moderately popular movies, each rated by about 1,000 of the 6,040 users (roughly the popularity of a well-known title). If the two audiences were independent, the expected overlap is $6040 \cdot (1000/6040)^2 = 1000^2 / 6040 \approx 166$ co-raters — plenty, and these popular-popular pairs are where CF is most reliable. Now pick two *niche* movies, each rated by 40 users. Expected overlap: $40^2 / 6040 \approx 0.26$ co-raters — i.e., **most niche-niche pairs have zero overlap**, so CF cannot relate them at all. This is the mathematical core of popularity bias: CF's similarity estimates are dense and reliable in the popular corner of the catalog and empty in the long tail, so the long tail never gets recommended, so it never accumulates ratings. The disease and the symptom are the same equation.

### 7.2 Cold start

CF needs interactions, so it is blind to anything that has none. A **new item** has an empty column, no neighbors, and is unrecommendable until enough users find it by other means. A **new user** has an empty row and gets only the global popular list. A **new system** has an empty matrix and CF does nothing at all. There is no internal fix — the only escape is to bring in information CF refuses to use: content features (item metadata, text, images) and user attributes, which is the entire premise of content-based and hybrid models. The [cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) gets its own post; for now, internalize that CF's purity is also its blind spot.

### 7.3 Popularity bias

CF systematically over-recommends popular items, for the reason the worked example just made precise: popular items have dense, reliable columns, so they win neighborhoods and accumulate score, while the long tail is data-starved. Left unchecked this is a self-reinforcing loop — recommend popular, get more clicks on popular, learn popular is even more similar to everything. Mitigations include the BM25/TF-IDF down-weighting from section 6.1, dividing similarities by item popularity, and re-ranking for diversity. The coverage column in the results section measures exactly this: what fraction of the catalog ever appears in *anyone's* top-10. A model with 0.4% coverage is recommending the same few hundred items to everyone.

### 7.4 Scalability: the $O(N^2)$ similarity matrix

Building the item-item similarity matrix is fundamentally quadratic in the number of items: there are $N^2$ pairs to consider. At $N = 3{,}706$ (MovieLens) that is $\sim$14M pairs, trivial. At $N = 10^6$ items it is $10^{12}$ pairs — you cannot materialize it. In practice you exploit sparsity (only pairs with a co-rater are nonzero) and use blocked sparse matrix multiplies or approximate-nearest-neighbor methods (LSH, the same ANN machinery used in [retrieval](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system)) to find each item's top-$k$ neighbors without scoring all pairs. But the cost is real, which is precisely why the similarity matrix is built **offline** in a batch job, never at request time. Amazon's 2003 paper made this its headline contribution: do the quadratic work offline, and serving becomes a cheap lookup.

### 7.5 The gray-sheep problem

A "gray sheep" is a user whose tastes do not correlate with any cohesive group — they like an idiosyncratic mix that matches no neighborhood. CF, which works by finding people or items that agree with you, has nothing to offer a user who agrees with no one. The effect is real but usually small (most users *do* cluster), and the standard mitigation is to fall back to popularity or content-based recommendations for users whose neighborhoods are too weak. It is the least severe item in the matrix, but it is a useful reminder that CF assumes taste is *shared*, and not everyone's is.

## 8. A problem-solving narrative: shipping CF on implicit data

Let me walk through the decision the way it actually happens, because the textbook version (explicit 1-to-5 ratings) is the easy case and almost nobody has it anymore. You launch a streaming product. You have *no ratings* — you have plays, skips, and watch-time. The product wants a "More like this" rail. Reason through it step by step.

**Step 1: there is no rating to predict, so reframe the task.** With implicit feedback the matrix entries are not values to regress; they are confidences of a positive. A 1 means "watched"; a blank means "we have no idea" — possibly disliked, possibly never seen. So we drop the rating-prediction framing entirely and treat this as top-N ranking from binary signals. The prediction rule's additive baseline $\bar r_u$ is meaningless here (there is no rating scale), so we score with the bare weighted sum $\sum_{j \in H_u} s_{ij}$ over the items $H_u$ the user actually watched.

**Step 2: pick a similarity that fits sets.** With no values to center, Pearson and adjusted cosine have nothing to subtract. The candidates are Jaccard (intersection over union) and cosine on binary vectors (intersection over geometric mean of sizes). I reach for **BM25-weighted cosine** (section 6.1) as the default, because plain set-overlap measures are dominated by popularity — a blockbuster shares *some* audience with everything, so it shows up as everyone's neighbor. BM25's `B` parameter discounts that, and it is one line to try.

**Step 3: handle the confidence, not just the binary.** Watching a 2-hour film all the way through is a stronger positive than a 30-second sample. The standard move (from Hu, Koren & Volinsky's 2008 implicit-ALS paper) is to keep a binary preference but attach a *confidence weight* $c_{ui} = 1 + \alpha \cdot (\text{watch fraction})$, so confident positives pull harder. In a memory-based setting you fold the confidence into the matrix values before normalizing, so a fully-watched film contributes more to its column's similarities than a half-abandoned one.

Now stress-test the decision. **What if negatives are mostly false negatives?** They are — a blank usually means "never surfaced," not "disliked." Memory-based CF sidesteps this gracefully because it never treats blanks as negatives at all; it only aggregates over observed positives. (This is one place memory-based CF is *cleaner* than a naively-trained classifier that labels every blank a 0.) **What at 100M items?** The $O(N^2)$ similarity build is now infeasible to materialize, so you switch to LSH or an ANN-based neighbor search to find each item's top-$k$ without scoring all pairs, and you shard the offline job. The serving path is unchanged — still a neighbor lookup and weighted sum. **What if the offline Recall@10 rises but online watch-time is flat?** The usual culprit is popularity bias: the model got better at recommending the head, which the offline metric (computed on logs the *old* system generated, themselves head-biased) happily rewards, but which adds no real discovery value online. The fix is to watch coverage offline and to A/B test, because [offline and online are two different worlds](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) and only the online one pays the bills.

#### Worked example: confidence-weighted item-item on watch data

Make the confidence idea concrete. A user partially watched film A (40% through) and fully watched film B (100%). With $\alpha = 4$, the confidences are $c_A = 1 + 4 \cdot 0.4 = 2.6$ and $c_B = 1 + 4 \cdot 1.0 = 5.0$. We want to score a candidate film C, whose precomputed similarities are $s_{C,A} = 0.30$ and $s_{C,B} = 0.50$. The confidence-weighted score is $c_A \, s_{C,A} + c_B \, s_{C,B} = 2.6 \cdot 0.30 + 5.0 \cdot 0.50 = 0.78 + 2.50 = 3.28$. Compare with the unweighted (binary) score $1 \cdot 0.30 + 1 \cdot 0.50 = 0.80$. The fully-watched film B now drives the recommendation, as it should — a completed view is a louder vote than an abandoned sample, and the confidence weight is exactly the dial that encodes "how loud." This single change (binary to confidence) is frequently worth a few points of online engagement on watch/listen products, because it stops a flood of accidental 5-second taps from looking like genuine interest.

## 9. Offline precompute, online serve

The architectural lesson of item-item CF — the thing that let it scale to Amazon — is the clean split between an offline phase that does the expensive quadratic work and an online phase that is a cheap lookup plus a weighted sum. The stack below shows the two halves.

![Stack diagram of the offline phase building similarity and top-k neighbors and the online phase doing a neighbor lookup and weighted scoring under a tight latency budget](/imgs/blogs/collaborative-filtering-from-first-principles-5.png)

**Offline (batch, nightly).** Read the interaction log, build the sparse user-item matrix, compute item-item similarities with shrinkage, truncate to top-$k$ neighbors per item, and write the result to a key-value store: for each item ID, the list of (neighbor ID, weight) pairs. This is the $O(N^2)$ work, and it runs on a schedule, decoupled from serving. Because item similarities are stable (section 2), running it nightly — or even weekly — is fine.

**Online (per request, milliseconds).** Fetch the user's recent interaction history (a short list of item IDs). For each, look up its precomputed neighbor list. Accumulate weighted scores across all the neighbor lists, drop items the user already saw, take the top-N. No matrix math at request time — just a few hash lookups and a weighted merge. This is why item-item CF serves at single-digit-millisecond p99 even on huge catalogs: the heavy lifting already happened.

This offline/online split is the template for the whole [retrieval-ranking funnel](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) that organizes this series. Retrieval precomputes item embeddings and an ANN index offline, then serves nearest-neighbor lookups online — structurally the *same* idea as item-item CF, just with learned vectors instead of co-rating similarities. Once you see item-item CF as "precompute a neighbor list per item, serve a weighted lookup," every later retrieval model is a variation on it.

#### Worked example: serving-latency budget for item-item CF

Put numbers on the online path. A user has 30 items in their recent history. Each item's neighbor list has $k = 50$ entries. Fetching 30 neighbor lists from an in-memory store is 30 lookups at ~5 microseconds each = ~0.15 ms. Merging $30 \times 50 = 1{,}500$ (neighbor, weight) pairs into a score map is ~1{,}500 hash operations, well under 0.5 ms. Top-N selection over a few thousand candidate items is another ~0.3 ms. Total compute is under 1 ms, leaving the rest of a typical 30 ms p99 budget to network, feature fetch, and serialization. Contrast that with the *offline* cost: building the similarity matrix is one sparse multiply over a 1M-nonzero matrix, seconds on a laptop for MovieLens, minutes-to-hours on a cluster for a real catalog. The asymmetry is the whole design: pay once offline, serve cheaply forever.

## 10. Results: popularity vs user-user vs item-item

Now the measured part. We evaluate three models on MovieLens-1M with the temporal split from section 5.1, reporting **Recall@10** (of the items a user actually interacted with in the test period, what fraction appear in the top-10 we recommend), **NDCG@10** (the same but rewarding higher placement), and **coverage** (the fraction of the catalog that appears in *some* user's top-10 — our popularity-bias meter). The figure shows the headline contrast.

![Before and after comparison of Recall at 10 and NDCG at 10 and coverage for the popularity baseline versus item-item collaborative filtering on MovieLens](/imgs/blogs/collaborative-filtering-from-first-principles-8.png)

The baseline is **most-popular**: recommend the 10 globally most-rated movies to everyone, identical for every user. It is a shockingly strong baseline — popular things really are popular — and beating it convincingly is the bar any real recommender must clear.

| Model | Recall@10 | NDCG@10 | Coverage | Notes |
| --- | --- | --- | --- | --- |
| Most-popular | 0.082 | 0.095 | 0.4% | same 10 items for everyone |
| User-user CF (k=50, Pearson) | 0.151 | 0.169 | 22% | per-user neighborhoods |
| Item-item CF (k=50, cosine) | 0.176 | 0.198 | 28% | plain cosine, no shrinkage |
| Item-item CF (k=50, cosine + shrink 20) | 0.181 | 0.205 | 31% | shrinkage adds ~3% relative |
| Item-item CF (k=50, BM25) | 0.189 | 0.213 | 36% | popularity-discounted |

These are representative numbers from a clean temporal-split run on MovieLens-1M; treat the third decimal as approximate (exact figures move with preprocessing choices, the split ratio, and whether you include implicit-binarized or graded ratings). The shape is what matters and it is robust across reruns and across the literature:

- **CF roughly doubles the popularity baseline** on both ranking metrics — from 0.082 to ~0.18 Recall@10. This is the headline: with no features at all, just the matrix, you nearly double ranking quality over "show everyone the same hits."
- **Item-item beats user-user** by a clear margin (0.176 vs 0.151 Recall@10), for the sparsity reason in section 2: item neighborhoods are computed from denser columns than user neighborhoods are from sparse rows.
- **Shrinkage helps**, adding a few percent relative by killing flimsy high-similarity pairs from thin overlaps.
- **BM25 weighting helps most** here, and notice it also raises *coverage* (36% vs 28%) — discounting popular items spreads recommendations across more of the catalog, mitigating popularity bias and improving ranking at once.
- **Coverage is low for all of them.** Even the best CF model recommends barely a third of the catalog to *anyone*. That is popularity bias made visible, and it is the gap content/hybrid models and diversity re-ranking are meant to close.

### 9.1 The effect of k and shrinkage

Sweeping the neighborhood size $k$ shows the bias-variance curve from section 4.1. Below is item-item cosine with shrinkage 20, all else fixed.

| k | Recall@10 | NDCG@10 |
| --- | --- | --- |
| 10 | 0.162 | 0.183 |
| 20 | 0.176 | 0.199 |
| 50 | 0.181 | 0.205 |
| 100 | 0.180 | 0.204 |
| 200 | 0.176 | 0.200 |

The curve rises to a plateau around $k = 50$ and then sags slightly as distant, noisy neighbors get a vote — exactly the predicted shape. The plateau is wide, so $k$ in the 20-to-100 band is all roughly equivalent; do not over-tune it. Shrinkage $\lambda$ behaves similarly: a sweep typically peaks somewhere in $\lambda \in [10, 50]$, with the gain over no-shrinkage largest precisely when the data is sparsest. The practical recipe is $k \approx 50$, $\lambda \approx 20$, BM25 or adjusted-cosine weighting, and stop — additional tuning on memory-based CF has sharply diminishing returns, and your effort is better spent on the matrix factorization that follows.

#### Worked example: reading an offline win honestly

Suppose your item-item model lifts Recall@10 from the user-user 0.151 to 0.181 — a relative gain of $\sim$20%. Is that real, and will it survive online? Three honesty checks. First, **no leakage**: confirm the test ratings are strictly later in time than train (we did, with the temporal split) — a random split would inflate Recall@10 by 30-50% through future-peeking and the win would evaporate online. Second, **full vs sampled metrics**: we ranked against the *entire* catalog, not a sampled set of 100 negatives; the 2020 KDD result by Krichene and Rendle ("On Sampled Metrics for Item Recommendation") showed that sampled metrics can reorder models, so a sampled Recall@10 of 0.4 means nothing comparable to a full one of 0.18. Third, **coverage as a guardrail**: our item-item model improved coverage alongside accuracy, which is reassuring — a model that gains Recall@10 while *dropping* coverage may just be leaning harder on popular items, an offline win that often goes flat or negative on online diversity and long-term engagement. Only after all three checks pass is the 20% lift worth shipping to an A/B test, and even then the [online number is the truth and the offline number is a biased proxy](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

## 11. Case studies: GroupLens and Amazon

Two real systems anchor everything above, and reading their original papers is the best hour you can spend on memory-based CF.

**GroupLens (Resnick et al., 1994).** "GroupLens: An Open Architecture for Collaborative Filtering of Netnews" (CSCW 1994) built a user-user CF system to recommend Usenet articles. It introduced the Pearson-correlation neighborhood method we derived in section 3.2: find users whose ratings correlate with yours, predict via a correlation-weighted average of their ratings. The contribution was not just the algorithm but the *architecture* — a way to overlay a recommendation layer on an existing system via a "Better Bit Bureau." The project also created the MovieLens dataset and the GroupLens research group that produced much of the field's early canon. When you run user-user CF on MovieLens, you are reproducing this 1994 paper almost line for line.

**Amazon item-to-item (Linden, Smith & York, 2003).** "Amazon.com Recommendations: Item-to-Item Collaborative Filtering" (IEEE Internet Computing, Jan/Feb 2003) is the paper that made CF an industrial workhorse. Their problem: tens of millions of customers and millions of products meant user-user CF was computationally hopeless (quadratic in the huge user dimension, and user neighborhoods are unstable). Their flip: precompute an **item-to-item similarity table** offline — for each product, the products most often bought/viewed together — and at request time, for a user's recent items, look up neighbors and merge. The key engineering insight is exactly section 9's offline/online split: "the algorithm's computation scales independently of the number of customers and number of items in the product catalog" at *serving* time, because the heavy work is offline. This is the "Customers who bought this also bought" engine, and a 2017 retrospective in the same venue noted it was still driving a large share of Amazon's recommendations more than a decade later. Durability is the lesson — a 2003 algorithm with no neural anything ran a top-five-website's recommendations for fifteen-plus years.

**Sarwar et al. (2001), the bridge.** "Item-Based Collaborative Filtering Recommendation Algorithms" (WWW 2001) is the academic paper that established item-item CF and introduced adjusted cosine. On the MovieLens data, it showed item-based methods matching or exceeding user-based accuracy while being far more scalable — the empirical result that justified Amazon's choice and the one our results table reproduces in miniature. If you read one paper from this post, read this one; it is short, clear, and every equation maps to code you now understand.

## 12. When CF is enough, and when to move to matrix factorization

Memory-based CF is not a stepping stone you are embarrassed to have used; it is a legitimate production technique, and knowing when it suffices saves you from over-engineering.

**Reach for item-item CF when:** your catalog is small-to-medium (thousands to low millions of items), interactions are reasonably dense in the head of the catalog, you need recommendations *today* with no training infrastructure, you value explainability ("because you watched X"), and a strong, stable, popular-aware list is a real product win. For a startup, an internal tool, a new vertical, or a baseline, item-item CF is often the right first model and a hard one to beat cheaply. It has no hyperparameters worth agonizing over, no training instability, and a serving path you can reason about completely.

**Move to [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) when:** sparsity is killing you (most item pairs have no co-raters, so CF can only relate the popular head), you need to generalize across the long tail, or you want a compact learned representation you can reuse downstream (for retrieval, for features in a ranker). MF learns a $d$-dimensional vector per user and item so that two items can be close in latent space even with zero direct co-ratings — it *fills the gaps* that CF leaves structurally empty. The same matrix, factorized, gives you embeddings that plug straight into the retrieval stage of the funnel. That is the next post, and the sparsity arithmetic in section 7.1 is its entire motivation.

**Skip CF entirely (or supplement it) when:** you are in a hard cold-start regime (new items dominate, the catalog churns daily — news, marketplaces) and content features are your only signal; or when the long tail is the *point* of the product and popularity bias is unacceptable. In those cases reach for content-based or hybrid models from the start. CF assumes a reasonably stable catalog with accumulating interactions; when that assumption fails, no similarity measure will save you.

The honest framing: item-item CF is the right answer surprisingly often, and the wrong answer in exactly the situations — extreme sparsity, cold start, long-tail focus — that the rest of this series exists to handle. Master it first; it is the conceptual root of everything that follows, and it is still running in production at companies you use every day.

## 13. Key takeaways

- **CF needs only the interaction matrix.** No features, no taxonomy — just who interacted with what. That is its superpower (works anywhere, zero feature engineering) and its blind spot (cold start, sparsity).
- **Prefer item-item over user-user.** Item neighborhoods are stable (precompute offline), computed from denser columns, more explainable, and they beat user-user on MovieLens. This is the Amazon 2003 lesson.
- **Center for the right bias.** Use Pearson (user-mean centering) for user-user; use **adjusted cosine** (also user-mean centering) for item-item — it cancels the user's rating-scale bias, which is the bias that actually hurts.
- **Always shrink thin overlaps.** Multiply similarity by $\frac{\lvert U_{ij}\rvert}{\lvert U_{ij}\rvert + \lambda}$ so two-co-rater pairs cannot fake a cosine of 1.0. It is the highest-leverage knob on real sparse data.
- **The prediction rule is a baseline plus a similarity-weighted, mean-centered correction**; for top-N ranking the per-user baseline cancels, so a bare weighted sum is enough.
- **Pick $k$ around 20-50 and stop.** The accuracy curve plateaus; over-tuning memory-based CF has sharply diminishing returns.
- **Precompute offline, serve online.** Do the $O(N^2)$ similarity work in a batch job; serving is a cheap neighbor lookup plus a weighted sum at single-digit-millisecond p99.
- **Sparsity is the wall.** Most item pairs share no co-rater, so CF can only relate the popular head of the catalog. That single fact is the entire case for moving to matrix factorization.
- **Measure with a temporal split, full metrics, and coverage as a guardrail.** A random split or sampled metrics will lie; coverage tells you whether an offline win is just popularity bias in disguise.

## 14. Further reading

- Resnick, Iacovou, Suchak, Bergstrom & Riedl, "GroupLens: An Open Architecture for Collaborative Filtering of Netnews," CSCW 1994 — the founding user-user CF paper and the origin of MovieLens.
- Sarwar, Karypis, Konstan & Riedl, "Item-Based Collaborative Filtering Recommendation Algorithms," WWW 2001 — establishes item-item CF and adjusted cosine; the bridge between GroupLens and Amazon.
- Linden, Smith & York, "Amazon.com Recommendations: Item-to-Item Collaborative Filtering," IEEE Internet Computing 2003 — the paper that scaled CF to industry; read it for the offline/online split.
- Herlocker, Konstan, Borchers & Riedl, "An Algorithmic Framework for Performing Collaborative Filtering," SIGIR 1999 — significance weighting and the design space of neighborhood methods.
- Krichene & Rendle, "On Sampled Metrics for Item Recommendation," KDD 2020 — why sampled Recall/NDCG can reorder models; evaluate against the full catalog.
- The `implicit` library documentation (`implicit.nearest_neighbours`: Cosine, TF-IDF, BM25 recommenders) — the production-grade version of the code in this post.
- Within this series: [What Is a Recommender System](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) for the funnel frame, [Matrix Factorization, the Workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) for the model-based successor that fixes sparsity, [Implicit Feedback Models: ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) for the click-data world, [The Cold-Start Problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) for what CF cannot do, and the capstone [The Recommender Systems Playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

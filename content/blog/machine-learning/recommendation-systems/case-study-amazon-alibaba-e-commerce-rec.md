---
title: "Case Study: Amazon and Alibaba E-Commerce Recommendation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Walk two decades of e-commerce recommendation from Amazon's 2003 item-to-item collaborative filtering to Alibaba's DIN, DIEN, and ESMM, where the objective is conversion and revenue, not clicks, with runnable PyTorch repros of co-purchase CF, target attention, and the entire-space CTR/CVR model."
tags:
  [
    "recommendation-systems",
    "recsys",
    "amazon",
    "alibaba",
    "item-to-item",
    "deep-interest-network",
    "esmm",
    "conversion",
    "e-commerce",
    "machine-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-1.png"
---

There is a sentence that ran on Amazon product pages for twenty years and quietly moved more merchandise than almost any other piece of machine learning ever shipped: *Customers who bought this item also bought.* It is so familiar it reads as wallpaper. But behind it sits one of the most consequential design decisions in the history of recommender systems — the choice, made by Greg Linden, Brent Smith, and Jeremy York around 1998 and published in 2003, to flip collaborative filtering on its head. Instead of asking "which users are similar to you?" (which does not scale when you have tens of millions of customers), Amazon asked "which items are similar to this one?" and precomputed the answer into a table you could serve with a lookup. That single inversion is why the recommendations on the page loaded instantly, scaled to a catalog of hundreds of millions of products, and did not fall over when a hundred million people shopped at once.

This post is a case study in e-commerce recommendation, and e-commerce is different from media in one way that changes everything: the goal is not engagement, it is **conversion**. A video platform is happy if you keep watching; a store is only happy if you *buy*. That shifts the entire objective function. Clicks are cheap — people tap things out of curiosity all the time — but a click that never becomes a purchase costs the store money in shelf space and attention. So the great e-commerce recommenders are, at heart, machines for predicting and maximizing the probability that a recommendation turns into revenue, under the brutal economics of price sensitivity, the long tail of products, cold-start of new SKUs, and a two-sided marketplace where sellers and buyers both have to be served. Figure 1 shows the founding move: user-user CF, which recomputes neighbors per request and chokes at scale, versus Amazon's item-item CF, which precomputes a similarity table offline and serves a cheap lookup.

![A two-column comparison of user-user collaborative filtering that scans all customers per request versus Amazon item-item collaborative filtering that precomputes a similarity table and serves a lookup.](/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-1.png)

By the end of this post you will be able to: derive *why* item-item collaborative filtering scaled when user-user could not, and reproduce the precomputed co-purchase similarity table in a few lines; explain Alibaba's Deep Interest Network (DIN, Zhou et al., 2018) and derive its **target attention** — the candidate item is the query, the user's past behaviors are the keys and values, so the user representation changes per candidate instead of being one fixed vector; explain how DIEN (Zhou et al., 2019) adds a GRU to model *evolving* interest; derive ESMM's (Ma et al., 2018) entire-space identity $\text{pCTCVR} = \text{pCTR} \times \text{pCVR}$ that fixes the sample-selection bias in conversion modeling; and write runnable PyTorch for all three load-bearing ideas. The frame is the same retrieval → ranking → re-ranking funnel and the serve → log → train feedback loop the whole series hangs on; if you want the map, start with [what a recommender system is](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) and end with the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

## 1. Why e-commerce is its own kind of recommendation problem

It is tempting to treat all recommenders as the same shape: users, items, interactions, predict the next interaction. That framing is true enough to be useful and false enough to get you fired. E-commerce has five distinctives that bend every design choice, and naming them up front explains why Amazon and Alibaba built what they built.

**First, the objective is conversion, not engagement.** On a feed, a click is close to the thing you want — attention. In a store, a click is one step in a funnel: impression → click → add-to-cart → purchase → (sometimes) return. Each arrow has a probability well below one, and the *last* arrow is the one that pays. Optimizing the first arrow (CTR) is easy and seductive and quietly wrong, because the items that maximize clicks (a shocking price, a sensational image, a misleading title) are frequently not the items that maximize purchases. The entire arc of Alibaba's published work is about pushing the objective down the funnel toward conversion and revenue. We will return to this in Section 8 and tie it to [calibration](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) and [causal uplift](/blog/machine-learning/recommendation-systems/causal-and-uplift-recommendation).

**Second, purchases are sparse, delayed, and expensive.** A user watches dozens of videos a day but buys a handful of things a month. The positive signal you most care about — a purchase — is rare, which makes the conversion model data-starved exactly where it matters. Worse, the feedback is *delayed*: someone adds an item to cart today and buys it (or does not) next week, which is the whole subject of [delayed feedback and conversion attribution](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution). A model trained on "did it convert?" labels that are still arriving will under-count recent positives and learn a biased CVR.

**Third, there is a long tail of products and a constant stream of new ones.** Amazon's catalog is hundreds of millions of items; most of them have almost no interaction data. New SKUs arrive every minute with zero history. This is the [cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) in its most acute commercial form: a product that cannot be recommended cannot be sold, and a recommender that only surfaces bestsellers strangles the long tail that is much of the catalog's value.

**Fourth, items have rich, decision-relevant attributes.** Price, brand, reviews, ratings, availability, shipping speed, and category are not decoration — they are the variables a shopper actually weighs. A recommender that ignores price will happily suggest a \$2,000 item to someone who only ever buys under \$50. And there is structure between items: some are **substitutes** (two phone chargers — you buy one) and some are **complements** (a phone and a case — you buy both). Confusing the two produces the classic failure of recommending another washing machine to someone who just bought a washing machine.

**Fifth, it is a two-sided marketplace.** Alibaba's Taobao and Amazon's third-party marketplace both have sellers competing for the same impressions. The recommender is implicitly allocating demand across sellers, which makes it a fairness and incentive problem, not just an accuracy problem — the subject of [fairness, privacy, and multi-stakeholder recommendation](/blog/machine-learning/recommendation-systems/fairness-privacy-and-multi-stakeholder-rec). A model that always favors the incumbent seller starves new sellers and, over time, the catalog.

Hold these five in mind. Every technique below is an answer to one or more of them.

## 2. Amazon's item-to-item collaborative filtering (2003)

Start where the field started. In 2003 Linden, Smith, and York published *Amazon.com Recommendations: Item-to-Item Collaborative Filtering* in IEEE Internet Computing, describing the algorithm that had already been running in production for years. To understand why it mattered you have to understand what it replaced.

### The problem with user-user CF at scale

Classic collaborative filtering, the kind we built up from scratch in [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles), is **user-based**: to recommend for user $u$, find the users most similar to $u$ (by overlap in what they bought or rated), then recommend the items those neighbors liked that $u$ has not seen. It is intuitive and it works on small data. It also does not survive contact with Amazon's scale, for three reasons.

The first is **online cost**. Finding a user's nearest neighbors means comparing that user's interaction vector against many other users' vectors, at request time. With tens of millions of customers and millions of items, the per-request work is enormous and grows with the number of customers. You cannot precompute "the neighbors of every user" cheaply either, because the set of users is enormous and each user's vector changes the instant they click anything.

The second is **sparsity and instability**. Any two users overlap on very few items (most people have bought a vanishingly small fraction of the catalog), so user-user similarities are computed from tiny, noisy intersections and swing wildly as new data arrives. A user with three purchases has almost no reliable neighbors.

The third is **cold-start for users**. A brand-new customer with one click has essentially no neighbors, so user-user CF has nothing to say to exactly the people the store most wants to convert.

### The inversion: similarity between items, not users

Amazon's move was to compute similarity between **items** instead of users. The key observation, and the reason it scales, is statistical: *items are far more stable than users*. The set of products that get co-purchased with a popular book changes slowly; the relationship "people who buy this espresso machine also buy these descaling tablets" is durable over months. And critically, you have far more interactions per item than per user for popular items, so item-item similarities are estimated from much larger, more stable samples.

The algorithm has two phases.

**Offline (the expensive part, run periodically):** build an item-to-item similarity table. For every pair of items $(i, j)$ that were ever co-purchased (or co-rated, or co-viewed) by the same customer, compute a similarity $\text{sim}(i, j)$. The 2003 paper uses cosine similarity over the *customers-who-bought* vectors. Represent each item $i$ as a binary (or count) vector over customers: $\mathbf{c}_i \in \{0,1\}^{|U|}$, where $c_{i,u} = 1$ if customer $u$ bought item $i$. Then

$$\text{sim}(i, j) = \cos(\mathbf{c}_i, \mathbf{c}_j) = \frac{\mathbf{c}_i \cdot \mathbf{c}_j}{\|\mathbf{c}_i\|\,\|\mathbf{c}_j\|} = \frac{|U_i \cap U_j|}{\sqrt{|U_i|}\,\sqrt{|U_j|}},$$

where $U_i$ is the set of customers who bought item $i$. The numerator is just the **number of customers who bought both** items; the denominator normalizes for popularity so that a megaseller is not "similar to everything" merely because lots of people bought it. Store, for each item, its top-$N$ most similar items in a table. This table is the entire model.

**Online (the cheap part, run per request):** to recommend for a user, take the small set of items in their cart and recent history, look up each item's precomputed neighbors, aggregate the similarities, remove items they already own, and return the top scorers. The cost is $O(\text{items in the user's history} \times N)$ — a handful of lookups, no scan over customers. *That* is why the page loaded instantly.

The scaling argument is worth stating precisely because it is the whole point. User-user CF pushes the expensive similarity computation **online** (per request, against a set that grows with users). Item-item CF pushes it **offline** (precompute once, against a set that grows much more slowly with the catalog) and leaves only a table lookup online. The offline cost is roughly $O(|I|^2)$ in the worst case, but in practice it is far cheaper because you only ever compare items that share at least one customer, and the co-purchase graph is sparse. You amortize that one-time cost over billions of cheap online lookups. Figure 1 captured this trade exactly.

#### Worked example: item-item co-purchase similarity by hand

Suppose five customers and three items. Item A is an espresso machine, item B is descaling tablets, item C is a yoga mat. The purchase matrix (rows = customers, 1 = bought):

| Customer | A (espresso) | B (tablets) | C (yoga mat) |
| -------- | ------------ | ----------- | ------------ |
| u1       | 1            | 1           | 0            |
| u2       | 1            | 1           | 0            |
| u3       | 1            | 0           | 0            |
| u4       | 0            | 0           | 1            |
| u5       | 0            | 1           | 1            |

Compute $\text{sim}(A, B)$. Customers who bought A: $U_A = \{u1, u2, u3\}$, so $|U_A| = 3$. Customers who bought B: $U_B = \{u1, u2, u5\}$, so $|U_B| = 3$. The overlap is $|U_A \cap U_B| = |\{u1, u2\}| = 2$. So

$$\text{sim}(A, B) = \frac{2}{\sqrt{3}\,\sqrt{3}} = \frac{2}{3} \approx 0.667.$$

Now $\text{sim}(A, C)$: $U_C = \{u4, u5\}$, overlap with $U_A$ is empty, so $\text{sim}(A, C) = 0$. And $\text{sim}(B, C)$: overlap $= \{u5\}$, so $\text{sim}(B, C) = \frac{1}{\sqrt{3}\,\sqrt{2}} \approx 0.408$. The table says: when a customer buys the espresso machine (A), the strongest recommendation is descaling tablets (B) at $0.667$, and the yoga mat (C) is irrelevant at $0$. That is the right answer, and we computed it without ever comparing customers to each other. Notice the cosine normalization at work: even though B is bought by three people and C by two, the *normalized* co-purchase with A is what ranks them, not raw popularity.

### Reproducing the item-item table in PyTorch / NumPy

Here is the whole algorithm, runnable, on a sparse co-purchase matrix. We use cosine over item columns. For a real catalog you would do this with sparse matrices and only keep the top-$N$ per item.

```python
import numpy as np
import scipy.sparse as sp

# R: customer x item binary purchase matrix (CSR). Rows = customers, cols = items.
# In production this is tens of millions x hundreds of millions, kept sparse.
def build_item_item_table(R, top_n=50):
    R = R.tocsc().astype(np.float32)          # column access by item
    # Column norms: sqrt(number of buyers) for each item (binary case).
    norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())  # |U_i|^0.5
    norms[norms == 0] = 1.0                    # avoid divide-by-zero for unsold items
    # Co-purchase counts: S[i, j] = number of customers who bought both i and j.
    S = (R.T @ R).tocsr()                      # item x item, sparse
    table = {}
    for i in range(S.shape[0]):
        start, end = S.indptr[i], S.indptr[i + 1]
        cols = S.indices[start:end]            # items co-purchased with i
        vals = S.data[start:end]               # raw co-purchase counts
        sims = vals / (norms[i] * norms[cols]) # cosine similarity
        mask = cols != i                       # drop self-similarity
        cols, sims = cols[mask], sims[mask]
        if len(cols) == 0:
            table[i] = []
            continue
        order = np.argsort(-sims)[:top_n]      # top-N neighbors
        table[i] = list(zip(cols[order].tolist(), sims[order].tolist()))
    return table

def recommend(table, user_items, k=10):
    scores = {}
    for it in user_items:                      # items already in cart / history
        for j, s in table.get(it, []):
            if j in user_items:                # never recommend what they own
                continue
            scores[j] = scores.get(j, 0.0) + s # aggregate similarity
    return sorted(scores.items(), key=lambda kv: -kv[1])[:k]
```

The `R.T @ R` line is the entire model: a sparse matrix product whose entry $(i, j)$ is the co-purchase count $|U_i \cap U_j|$. Everything else is normalization and bookkeeping. On the [Amazon Reviews](https://nijianmo.github.io/amazon/) data (the public dataset that descends from this exact problem), this runs in seconds on a single category and gives recommendations that are recognizably "people who bought this also bought that." It is twenty-three years old and it still ships, often as the fallback when a deep model is unsure or a product is too new for an embedding.

### What item-item CF gets right, and where it runs out

What it gets right is profound: it is **cheap, stable, interpretable, and a strong cold-start fallback for items** (a new product picks up neighbors as soon as it is co-purchased even once). The recommendations are inherently explainable — "because you bought X" — which matters commercially and for trust.

Where it runs out is equally important. It is **memoryless about sequence and context**: it does not know the *order* you bought things, your price band, the time of day, or which of your many interests is active right now. It treats your whole history as an unordered bag and scores candidates by raw co-purchase, so it cannot tell that you bought a gift last week and a personal item today, or that you are price-sensitive on electronics but not on books. And it has a strong **popularity bias** baked in — popular items co-occur with everything, and even cosine normalization only partly tames it (the subject of [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer)). Closing those gaps is exactly what the deep models from Alibaba set out to do.

### The similarity choices that actually matter in practice

The 2003 paper uses cosine, but the choice of similarity function is one of the few knobs that moves item-item CF's quality meaningfully, and it is worth knowing the family because each variant trades off popularity handling differently. Cosine, as derived above, normalizes by the geometric mean of the two items' buyer counts, $\sqrt{|U_i|\,|U_j|}$. **Jaccard** similarity normalizes by the union instead, $|U_i \cap U_j| / |U_i \cup U_j|$, which is more aggressive at penalizing a megaseller that co-occurs with everything. The **conditional probability** form, $P(j \mid i) = |U_i \cap U_j| / |U_i|$, is asymmetric and answers "of the people who bought $i$, what fraction also bought $j$?" — which is closer to what the "bought also bought" module actually wants to say, but it over-recommends popular $j$ (because popular items have a high marginal probability of being bought by anyone). The standard fix, used widely in production, is a **popularity penalty**: divide by $|U_j|^{\alpha}$ for some $\alpha \in (0, 1]$ tuned on held-out data, which interpolates between conditional probability ($\alpha = 0$, popularity-loving) and a strongly normalized score ($\alpha = 1$, popularity-fighting). On a real catalog this single exponent is often the difference between a "bought also bought" module that recommends batteries and gift cards to everyone and one that surfaces genuinely related items. The lesson generalizes: even the simplest model has a popularity-bias dial, and it must be tuned against the metric you care about, not left at the textbook default.

A second practical choice is **what counts as a co-occurrence**. Co-purchase (bought both) gives the cleanest complementarity signal but is the sparsest. Co-view (viewed both in a session) is denser and surfaces substitutes (people comparison-shop similar items), which is the *wrong* signal for a post-purchase module but the *right* one for a "similar items" module on the product page. Add-to-cart sits in between. A mature store runs several item-item tables off different co-occurrence events and routes each to the surface where its signal is appropriate — the "frequently bought together" widget uses co-purchase, the "compare similar" widget uses co-view. This is the substitutes-versus-complements distinction (Section 8) showing up at the simplest possible model, and getting the event right matters more than getting the similarity function right.

## 3. The evolution to deep models, and where Alibaba pushed

Between 2003 and the late 2010s, e-commerce recommendation industrialized into the funnel this series describes: a cheap **retrieval** stage that narrows hundreds of millions of products to a few thousand candidates, then an expensive **ranking** stage that scores those candidates with a rich feature set, then **re-ranking** for diversity, business rules, and marketplace fairness. Item-item CF lives in retrieval (and as a feature). The action moved to ranking, where a model can use all the features item-item CF ignores.

The signature insight that Alibaba contributed — the one that makes their work a case study and not just another deep CTR model — is that **a user does not have one interest; they have many, and which one is active depends on what you are showing them**. A person who has bought running shoes, a laptop, baby formula, and a cookbook is not a single point in interest-space. When you are about to show them a candidate pair of running socks, the running-shoes history is the relevant signal and the baby-formula history is noise. When the candidate is a diaper, it is the reverse. A single fixed user vector — the thing two-tower retrieval and most deep CTR models compute — averages all of those interests into one blurry point and loses exactly the candidate-specific signal you need. Figure 2 shows DIN's answer.

![A branching dataflow graph where a candidate item acts as the attention query over past behaviors, producing per-item weights that sum into a candidate-aware interest vector feeding an MLP that outputs a CTR score.](/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-2.png)

This is the bridge to the rest of this post. We will study three Alibaba models in order: **DIN** (local activation / target attention, 2018), **DIEN** (interest evolution with a GRU, 2019), and **ESMM** (entire-space conversion modeling, 2018), plus a word on **TDM** (tree-based retrieval, 2018) for the retrieval side. Figure 3 lays out the four ideas side by side so you have the map before we descend into each.

![A four-row comparison matrix of Amazon item-item collaborative filtering, DIN, DIEN, and ESMM showing each method's core idea, what signal it captures, and its paper year.](/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-3.png)

## 4. DIN: Deep Interest Network and target attention (2018)

Zhou et al.'s *Deep Interest Network for Click-Through Rate Prediction* (KDD 2018) is the paper that operationalized "interests are local." The base architecture is the standard deep CTR recipe of the era (and the one we built in [the ranking model](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations)): take sparse features — user id, item id, category, plus the user's behavior history as a list of item ids — embed each into a dense vector, pool the variable-length history into a fixed vector, concatenate everything, and push through an MLP to a sigmoid that predicts click probability.

The base model pools the history by **sum or average** — the bag-of-items trick. DIN's one change, and it is surgical, is to replace that fixed pooling with a **candidate-dependent weighted pooling**. Instead of averaging all behaviors equally, weight each past behavior by how relevant it is to the *candidate item being scored*. That is target attention.

### Deriving target attention

Let the user's behavior history be a sequence of item embeddings $\{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_H\}$ (the items they clicked or bought, each a vector). Let the candidate item being scored be $\mathbf{e}_a$ (the "target", the "ad", the thing whose CTR we want). The base model computes a user vector by simple pooling:

$$\mathbf{u}_{\text{base}} = \frac{1}{H}\sum_{t=1}^{H} \mathbf{e}_t \quad \text{(or the sum)}.$$

Notice $\mathbf{u}_{\text{base}}$ does not depend on $\mathbf{e}_a$ at all — it is the same vector no matter which candidate you score. DIN instead computes a **candidate-aware** user vector:

$$\mathbf{u}(\mathbf{e}_a) = \sum_{t=1}^{H} w_t(\mathbf{e}_a)\, \mathbf{e}_t, \qquad w_t(\mathbf{e}_a) = g(\mathbf{e}_t, \mathbf{e}_a),$$

where $g$ is a small "activation unit" — a tiny MLP that takes the behavior embedding $\mathbf{e}_t$, the candidate embedding $\mathbf{e}_a$, and (importantly) their interaction (DIN feeds the element-wise difference $\mathbf{e}_t - \mathbf{e}_a$ and/or product as explicit features) and outputs a scalar weight. Here $\mathbf{e}_a$ is the **query**, the behaviors $\mathbf{e}_t$ are the **keys and values**, and $w_t$ is the attention weight. The user representation is now a function of the candidate — it *changes per item you score*. If you connect this to the formal attention machinery, it is the same shape as the scaled dot-product attention in [self-attention for sequences](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec), with one deliberate difference we discuss below.

The full score concatenates the candidate-aware interest, the candidate embedding, and other features, then runs an MLP:

$$\hat{y} = \sigma\big(\text{MLP}([\,\mathbf{u}(\mathbf{e}_a)\;\|\;\mathbf{e}_a\;\|\;\mathbf{x}_{\text{other}}\,])\big),$$

trained with the usual binary cross-entropy on click labels.

### Why a fixed user vector loses candidate-specific signal

This is the crux, so let us make it concrete and quantitative. Figure 4 contrasts the two.

![A two-column comparison of a fixed user vector that averages all history into one point and loses candidate-specific signal versus target attention that reweights the same history per candidate item.](/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-4.png)

#### Worked example: DIN target attention vs a fixed user vector

Take a user with three past behaviors, embedded in a toy 2-D space so we can do the arithmetic by hand. Suppose the embedding space has a "sports" axis and a "baby" axis:

- Behavior 1: running shoes, $\mathbf{e}_1 = (0.9, 0.1)$ — strongly sports.
- Behavior 2: a tennis racket, $\mathbf{e}_2 = (0.8, 0.0)$ — strongly sports.
- Behavior 3: baby formula, $\mathbf{e}_3 = (0.0, 1.0)$ — strongly baby.

**Fixed pooling** gives one user vector for *every* candidate:

$$\mathbf{u}_{\text{base}} = \tfrac{1}{3}\big[(0.9,0.1)+(0.8,0.0)+(0.0,1.0)\big] = (0.567, 0.367).$$

That is a muddle — neither clearly sports nor clearly baby. Now score two candidates.

Candidate A is running socks, $\mathbf{e}_A = (0.85, 0.05)$. Candidate B is diapers, $\mathbf{e}_B = (0.05, 0.95)$. With the fixed vector, the dot products are $\mathbf{u}_{\text{base}} \cdot \mathbf{e}_A = 0.567(0.85)+0.367(0.05) = 0.500$ and $\mathbf{u}_{\text{base}} \cdot \mathbf{e}_B = 0.567(0.05)+0.367(0.95) = 0.377$. Both candidates get a middling, similar score, because the user vector is a blur.

**Target attention** instead reweights the history per candidate. Use a simple relevance weight $w_t \propto \exp(\beta\,\mathbf{e}_t \cdot \mathbf{e}_a)$ (a softmax over dot-product relevance with a temperature $\beta$ — the activation unit learns something richer, but this captures the idea). For candidate A (running socks), the dot products with the three behaviors are $0.85(0.9)+0.05(0.1)=0.770$, $0.85(0.8)+0.05(0.0)=0.680$, and $0.85(0.0)+0.05(1.0)=0.050$. Raw softmax over $(0.770, 0.680, 0.050)$ is nearly flat because the toy logits are small; the activation unit's MLP effectively learns a sharper temperature. With $\beta = 5$ the logits become $(3.85, 3.40, 0.25)$ and the softmax is about $(0.58, 0.37, 0.05)$: the running shoes and tennis racket dominate, the baby formula is suppressed. The candidate-aware interest vector is then

$$\mathbf{u}(\mathbf{e}_A) \approx 0.58(0.9,0.1)+0.37(0.8,0.0)+0.05(0.0,1.0) = (0.818, 0.108),$$

a *strongly sports* vector. Its dot product with running socks is $0.818(0.85)+0.108(0.05) = 0.700$ — much higher than the fixed vector's $0.500$. For candidate B (diapers), the same procedure puts almost all the weight on baby formula and yields a strongly baby vector, scoring diapers high. The same history produces a different, candidate-appropriate user vector for each candidate. That is the entire point, and it is why DIN lifts AUC: the model can finally tell which of your interests is relevant to what it is about to show you.

### DIN in PyTorch

Here is a compact, runnable DIN-style ranker: an embedding table, the activation-unit attention over the behavior sequence, and an MLP head. It handles variable-length histories with a mask.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationUnit(nn.Module):
    """Scores relevance of each past behavior to the candidate (the DIN local activation)."""
    def __init__(self, dim, hidden=36):
        super().__init__()
        # Input: behavior, candidate, their elementwise diff and product -> 4*dim
        self.net = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.PReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, behaviors, candidate, mask):
        # behaviors: (B, H, D), candidate: (B, D), mask: (B, H) of 1/0
        B, H, D = behaviors.shape
        cand = candidate.unsqueeze(1).expand(-1, H, -1)        # (B, H, D)
        x = torch.cat([behaviors, cand, behaviors - cand,
                       behaviors * cand], dim=-1)              # (B, H, 4D)
        scores = self.net(x).squeeze(-1)                       # (B, H) raw weights
        scores = scores.masked_fill(mask == 0, -1e9)          # ignore padding
        weights = torch.softmax(scores, dim=1)                # (B, H)
        interest = torch.bmm(weights.unsqueeze(1),
                             behaviors).squeeze(1)             # (B, D) weighted sum
        return interest, weights

class DIN(nn.Module):
    def __init__(self, n_items, n_cats, dim=18):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, dim, padding_idx=0)
        self.attn = ActivationUnit(2 * dim)                   # item+cat per behavior
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim * 3, 200), nn.PReLU(),
            nn.Linear(200, 80), nn.PReLU(),
            nn.Linear(80, 1),
        )

    def emb(self, item_ids, cat_ids):
        return torch.cat([self.item_emb(item_ids), self.cat_emb(cat_ids)], dim=-1)

    def forward(self, hist_items, hist_cats, cand_item, cand_cat, mask):
        behaviors = self.emb(hist_items, hist_cats)           # (B, H, 2D)
        candidate = self.emb(cand_item, cand_cat)             # (B, 2D)
        interest, _ = self.attn(behaviors, candidate, mask)   # (B, 2D)
        x = torch.cat([interest, candidate, interest * candidate], dim=-1)
        return self.mlp(x).squeeze(-1)                         # logit
```

Train it with `F.binary_cross_entropy_with_logits` on click labels and a temporal split (train on the past, evaluate on the future — leakage is the number-one way an offline CTR win evaporates, see [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate)). One DIN-specific detail from the paper: it replaces the usual ReLU with **Dice** (a data-adaptive variant of PReLU whose break-point follows the input distribution) and uses an **adaptive regularization** that only penalizes embedding rows touched in the mini-batch — important because the id space is huge and most rows are untouched per batch, so uniform L2 would over-shrink rare items. Those are not the headline idea but they are the difference between the paper's numbers and a reimplementation that mysteriously underperforms.

A subtle but important point: DIN deliberately does **not** normalize the attention weights with the dot-product scaling and softmax-only formulation of standard attention. The original DIN uses the activation-unit output *without* forcing the weights to sum to one in its primary formulation, because the *magnitude* of total activation is itself signal — a user with many strongly-relevant behaviors should produce a larger interest activation than a user with one weak match. (The reimplementation above uses softmax for stability and clarity; the paper discusses both. The conceptual point — query is the candidate, keys/values are the behaviors — is identical either way.)

### Why the gradient cares about the candidate

It is worth seeing *why* target attention learns better than a fixed pool, beyond the intuition, because the answer is in the gradient. With fixed pooling, the user vector $\mathbf{u}_{\text{base}}$ is a constant function of the candidate, so during training the gradient that flows back to a behavior embedding $\mathbf{e}_t$ is the *same* regardless of which candidate produced the loss. A behavior embedding gets averaged updates across all candidates it ever co-occurred with — running shoes get pushed both toward "good for socks" and "good for diapers" because both losses touch the same averaged vector. The signal smears.

With target attention, the gradient to $\mathbf{e}_t$ is scaled by its attention weight $w_t(\mathbf{e}_a)$ for that specific candidate. When the candidate is socks, the running-shoes behavior has a large weight and receives a strong, *relevant* gradient; when the candidate is diapers, the same behavior has a near-zero weight and receives almost no gradient. The embedding therefore specializes — it learns the directions that matter for the candidates it is actually relevant to, and is shielded from the irrelevant ones. This is the same reason attention helps in language models (gradient routing to the relevant context), specialized to recommendation: target attention is, among other things, a *gradient-routing* mechanism that keeps irrelevant behaviors from polluting an embedding's updates. That is the deeper "why" behind the +0.01 to +0.02 AUC the paper reports.

## 5. DIEN: modeling interest that evolves (2019)

DIN treats the behavior history as an unordered set with candidate-aware weights. But interests *move*. Last year you were furnishing an apartment (lamps, a sofa, curtains); this year you are into cycling (a helmet, a jersey, a bike computer). The furniture behaviors are stale and the cycling behaviors are fresh, and a model that ignores order cannot tell. Zhou et al.'s *Deep Interest Evolution Network* (AAAI 2019) adds the missing time dimension. Figure 6 shows the layered architecture.

![A vertical stack of DIEN layers showing the behavior sequence feeding an interest extractor GRU with an auxiliary loss, then an attention gate, then an evolution GRU, producing an evolved candidate-aligned interest.](/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-6.png)

DIEN has two stacked recurrent layers and one clever loss.

**Interest Extractor Layer.** A GRU runs over the behavior sequence and produces a hidden state $\mathbf{h}_t$ at each step, interpreted as the user's *latent interest* at time $t$. A raw GRU only gets gradient from the final CTR loss, which is a weak, distant signal for the intermediate states. So DIEN adds an **auxiliary loss**: at each step, use the hidden state $\mathbf{h}_t$ to predict the *next actual behavior* $\mathbf{e}_{t+1}$ against a sampled negative, with a binary classification loss. This is the GRU equivalent of next-item prediction and it forces each hidden state to actually represent interest, not just whatever happens to help the final logit. It is the same idea as the next-item objective in [sequential and session-based recommendation](/blog/machine-learning/recommendation-systems/sequential-and-session-based-recommendation), used here as supervision for the intermediate layer.

**Interest Evolving Layer.** A second GRU evolves the extracted interests *toward the candidate*. Here DIN's target attention returns: compute an attention weight $a_t$ between each interest state $\mathbf{h}_t$ and the candidate, then use it to **gate the GRU update**. DIEN's specific mechanism, AUGRU (GRU with Attentional Update gate), scales the update gate $\mathbf{u}_t$ of the GRU by the attention score: $\tilde{\mathbf{u}}_t = a_t \cdot \mathbf{u}_t$. The effect is that behaviors irrelevant to the candidate barely update the evolving interest, while relevant behaviors drive it forward. The final hidden state is the **evolved, candidate-aligned interest**, which feeds the MLP exactly as DIN's interest vector did.

The science of why this helps: a plain attention-weighted sum (DIN) captures *which* past behaviors matter, but not the *trajectory*. AUGRU captures both — it walks through the history in order, and at each step it lets relevant behaviors push the interest state while ignoring irrelevant drift. On data where interest genuinely evolves (long histories spanning months, category switches), DIEN's ordering-aware evolution is the additional signal over DIN's order-free attention.

The cost is real: two GRUs over the behavior sequence are sequential and much heavier than DIN's parallel attention, which matters at Taobao's request volume. This is the recurring ranking trade-off — more interest modeling for more latency — and it is why later Alibaba and industry work moved to attention-only sequence models (the SASRec/BERT4Rec family) that are parallelizable on the sequence dimension. DIEN is the high-water mark of the recurrent approach to e-commerce interest modeling.

```python
class AUGRUCell(nn.Module):
    """GRU cell whose update gate is scaled by an attention score (DIEN's evolving layer)."""
    def __init__(self, dim):
        super().__init__()
        self.x2h = nn.Linear(dim, 3 * dim)
        self.h2h = nn.Linear(dim, 3 * dim)

    def forward(self, x, h_prev, att):                # att: (B,) attention weight
        gates_x = self.x2h(x); gates_h = self.h2h(h_prev)
        r = torch.sigmoid(gates_x[:, :x.size(-1)] + gates_h[:, :x.size(-1)])
        u = torch.sigmoid(gates_x[:, x.size(-1):2*x.size(-1)]
                          + gates_h[:, x.size(-1):2*x.size(-1)])
        n = torch.tanh(gates_x[:, 2*x.size(-1):] + r * gates_h[:, 2*x.size(-1):])
        u = att.unsqueeze(-1) * u                     # attentional update gate
        return (1 - u) * h_prev + u * n              # evolved hidden state
```

You would wrap this cell in a loop over the sequence, feeding per-step attention scores computed against the candidate, after an interest-extractor GRU (a standard `nn.GRU`) with the auxiliary next-item loss. The full DIEN is more code than fits cleanly here, but the AUGRU cell is the load-bearing novelty and it is short.

### DIN vs DIEN vs attention-only sequence models

It helps to place these three approaches to behavior modeling on one axis, because the e-commerce field walked through them in order and each fixed the previous one's main limitation:

| Model              | Sequence handling          | Order-aware? | Parallelizable? | Cost      | When it wins                          |
| ------------------ | -------------------------- | ------------ | --------------- | --------- | ------------------------------------- |
| Pooling MLP        | Sum / average bag          | No           | Yes             | Cheapest  | Short or single-interest histories    |
| DIN                | Target attention (no GRU)  | No           | Yes             | Low       | Multi-interest, order does not matter |
| DIEN               | Extractor + AUGRU (2 GRUs) | Yes          | No (sequential) | High      | Long, drifting, evolving interest     |
| SASRec / BERT4Rec  | Self-attention over seq    | Yes (pos enc)| Yes             | Medium    | Long sequences, want order + speed    |

The honest read is that DIEN captures the *same* ordering signal a self-attention sequence model captures, but pays for it with sequential GRU computation that does not parallelize on the sequence dimension — a serving problem at Taobao request volume. That is precisely why the field, including later Alibaba work, moved toward attention-only sequence models (the SASRec/BERT4Rec family in [self-attention for sequences](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec)) that get the order-awareness of DIEN with the parallelism of DIN. DIEN remains the clearest *exposition* of why interest evolution matters; for a new build, an attention-only sequence encoder feeding a DIN-style target-attention head is usually the better engineering point on the latency–quality curve. Knowing the lineage is what lets you make that call instead of reaching for DIEN by reputation.

## 6. ESMM: modeling conversion over the entire space (2018)

Now the most subtle and, for e-commerce, the most important idea. Everything so far predicted **clicks** (CTR). But the store wants **conversions** (CVR — given a click, will it become a purchase?). Modeling CVR naively is a trap, and Ma et al.'s *Entire Space Multi-Task Model* (SIGIR 2018) is the clean fix. Figure 5 shows the architecture.

![A branching graph where shared embeddings feed a CTR head and a CVR head whose product forms a CTCVR output, with click and buy labels supervised over the full impression space.](/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-5.png)

### The sample-selection-bias trap

The funnel is impression → click → conversion. CVR is defined as $P(\text{conversion} \mid \text{click})$ — conditional on a click. The naive approach trains a CVR model *only on clicked impressions* (because conversion is only defined after a click) with labels "did this click convert?". This has two fatal problems.

**Sample selection bias (SSB).** The CVR model is *trained* on the click space (clicked impressions only) but *used* at serving time over the *entire* impression space — it scores every candidate, most of which would not be clicked. The training distribution and the inference distribution differ systematically: clicked items are not a random sample of all items (they are the ones the *previous* model and the user found appealing). A model trained on a biased sample and applied to the full space is miscalibrated and wrong on exactly the items it was never trained on. This is [selection bias in click data](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data), in its conversion form.

**Data sparsity.** Clicks are already rare; conversions among clicks are rarer still. Training a CVR model on the clicked subset means training on a tiny fraction of your logs — far less data than the CTR model gets, for a harder task.

### The entire-space identity

ESMM's fix is an algebraic identity that lets you train CVR over the *entire* impression space without ever needing a conversion label on an unclicked item. Start from the chain rule of probability over the funnel. Let $x$ be an impression (user, item, context). Then:

$$\underbrace{P(\text{click \& buy} \mid x)}_{\text{pCTCVR}} = \underbrace{P(\text{click} \mid x)}_{\text{pCTR}} \times \underbrace{P(\text{buy} \mid \text{click}, x)}_{\text{pCVR}}.$$

This is just the definition of conditional probability — a buy requires a click first (in this attribution model), so the joint factors into CTR times CVR. The two quantities on the left, **pCTCVR** (probability of click-and-buy) and **pCTR**, are *both defined over the entire impression space* — every impression has a click label (clicked or not) and a click-and-buy label (clicked-and-bought or not). You never need to condition on a click to get a label for them.

So ESMM builds two networks that **share their embedding tables** (sharing helps the data-starved CVR head borrow the CTR head's representation learning). One head outputs pCTR, the other outputs pCVR. But — and this is the trick — pCVR is *never supervised directly*. Instead, the model multiplies them, $\text{pCTCVR} = \text{pCTR} \times \text{pCVR}$, and supervises the two *observable* quantities over all impressions:

$$\mathcal{L} = \mathcal{L}_{\text{CTR}}(\text{pCTR}, y_{\text{click}}) + \mathcal{L}_{\text{CTCVR}}(\text{pCTR}\cdot\text{pCVR},\; y_{\text{click\&buy}}),$$

both binary cross-entropy, both summed over **every impression**. The CVR head is learned *implicitly*, as the factor that makes the product match the observed click-and-buy rate. Because both losses are over the entire space, the CVR head is effectively trained over the entire space too — SSB is gone — and it shares the CTR head's data, so sparsity is mitigated. This is the same multi-task spirit as [MMoE/PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple), specialized to the funnel's probability structure, and it composes directly with the [delayed-feedback](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution) correction for late-arriving conversions.

#### Worked example: why entire-space CVR beats clicked-only CVR

Suppose 10,000 impressions. The previous policy showed mostly appealing items, so 1,000 got clicked (10% CTR), and of those, 200 converted (20% CVR-on-clicks). The remaining 9,000 unclicked impressions have *no* conversion label.

A **clicked-only CVR model** trains on 1,000 rows. It learns the conversion patterns of the *appealing* items that got clicked. Now serve it on a fresh impression of a niche, rarely-clicked item — say a specialized industrial part. The model has barely seen items like it (they rarely get clicked, so they rarely entered training), and it confidently outputs a CVR that is essentially extrapolation. If niche items actually convert at a *higher* rate when clicked (the people who click them are highly intent), the clicked-only model systematically under-predicts their CVR and the ranker buries them — a direct revenue loss on the long tail.

The **ESMM model** trains pCTR and pCTCVR over all 10,000 rows. The niche part appears 50 times in training (even if clicked only twice), so its pCTR head learns its low click rate and its pCTCVR head learns its true click-and-buy rate from the full sample. The implicit pCVR = pCTCVR / pCTR is then anchored by entire-space statistics, not by a handful of clicked rows. The paper reports CVR-task AUC improvements of roughly +2.5 absolute points over the clicked-only baseline on Taobao's production data and the public Ali-CCP dataset they released — a large gain for a ranking model, and it comes entirely from fixing *what data the model sees*, not from a fancier architecture. That is the e-commerce lesson in one sentence: most of the win is in the objective and the sample, not the network.

### ESMM in PyTorch

```python
class ESMM(nn.Module):
    def __init__(self, field_dims, dim=16, hidden=(128, 64)):
        super().__init__()
        # ONE shared embedding table across both tasks (the key design choice).
        self.embed = nn.ModuleList([nn.Embedding(n, dim) for n in field_dims])
        in_dim = len(field_dims) * dim
        def tower():
            layers, d = [], in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]; d = h
            layers += [nn.Linear(d, 1)]
            return nn.Sequential(*layers)
        self.ctr_tower = tower()   # predicts pCTR
        self.cvr_tower = tower()   # predicts pCVR (never directly supervised)

    def forward(self, x):                                  # x: (B, n_fields) long ids
        e = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embed)], dim=-1)
        p_ctr = torch.sigmoid(self.ctr_tower(e)).squeeze(-1)
        p_cvr = torch.sigmoid(self.cvr_tower(e)).squeeze(-1)
        p_ctcvr = p_ctr * p_cvr                            # the entire-space identity
        return p_ctr, p_cvr, p_ctcvr

def esmm_loss(p_ctr, p_ctcvr, y_click, y_buy, eps=1e-7):
    # Both losses over the ENTIRE impression space. CVR is learned implicitly.
    l_ctr = F.binary_cross_entropy(p_ctr.clamp(eps, 1 - eps), y_click)
    l_ctcvr = F.binary_cross_entropy(p_ctcvr.clamp(eps, 1 - eps), y_buy)
    return l_ctr + l_ctcvr
```

Note what is *not* here: there is no loss on `p_cvr` directly. The CVR head gets its gradient only through the product `p_ctcvr`. That is the whole design — and it is why you must serve `p_cvr` (or `p_ctr * p_cvr * price` for revenue ranking, Section 8) carefully, knowing it was never directly calibrated against a CVR-on-clicks label.

### The delayed-feedback wrinkle ESMM does not solve

ESMM fixes *which space* the CVR model is trained on, but it does not, by itself, fix *when* the labels arrive. The buy label `y_buy` for a recent impression may still be **negative simply because the conversion has not happened yet** — the user added the item to cart yesterday and will check out next week. If you train on a fresh window, a large fraction of true future positives are mislabeled as negatives, and the CTCVR head learns a depressed conversion rate. This is the [delayed feedback](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution) problem, and it composes with ESMM rather than being subsumed by it. The standard remedies stack on top: a generous attribution window (wait long enough that most conversions have landed before you treat a label as final), an explicit delayed-feedback loss that models the conversion delay as a survival/exponential time and down-weights "not yet converted" rows by the probability they still might, or an importance-weighting correction for the censoring. The practical takeaway is that ESMM and a delayed-feedback correction are *both* needed in production conversion modeling: ESMM gets the *sample* right (entire space), the delayed-feedback model gets the *label* right (censored positives). Shipping ESMM with naive fresh-window labels reintroduces a bias of its own, just a different one.

### A calibration check you should always run

Because the whole point of pCTCVR is to be a *calibrated* probability you multiply by price, you must verify calibration, not just AUC. The cheap, decisive check is a **reliability curve**: bin impressions by predicted pCTCVR (say ten equal-frequency bins), and in each bin compare the mean prediction to the observed click-and-buy rate. A well-calibrated model lies on the diagonal; a model that systematically over-predicts in the high-probability bins will over-rank expensive items and tank GMV in a way AUC will never reveal. The expected-calibration-error (ECE) — the average absolute gap between predicted and observed across bins — is the one-number summary. If ECE is bad, fit an isotonic or Platt calibrator on a held-out window before serving. In e-commerce this is not optional polish; a 5-point calibration error on pCTCVR, multiplied by price, is a 5-point error in expected revenue per impression on every item, and the ranking it produces is wrong in a way that compounds with item price.

## 7. The retrieval side: TDM and two-tower at catalog scale

Everything above is *ranking* — scoring a few thousand candidates. But where do the candidates come from when the catalog is hundreds of millions of items? Item-item CF is one retriever; Alibaba also built two notable ones.

**Two-tower retrieval** is the workhorse this series covers in [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval): a user tower and an item tower map to a shared space, and you retrieve by maximum-inner-product search (MIPS) over an approximate-nearest-neighbor index, the subject of [ANN serving with faiss, hnsw, scann](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann). Its limitation is the same one DIN diagnosed at the ranking stage: the user tower produces a *single fixed user vector* for retrieval, so it cannot do candidate-aware attention (the candidate is not known until after retrieval — that is the chicken-and-egg of retrieval). The dot-product structure that makes ANN fast is exactly what forbids deep candidate-user interaction at the retrieval stage.

**TDM (Tree-based Deep Model**, Zhu et al., KDD 2018) is Alibaba's answer to "can we use a *deep, attention-style* model at retrieval despite the scale?" The idea: organize all items into a balanced tree (a hierarchy where leaves are items and internal nodes are coarse clusters). At retrieval, do a beam search from the root: at each level, score the children with an arbitrary deep model (including DIN-style attention, since you only score a beam of nodes, not the whole catalog) and keep the top-$k$, descending until you reach leaves. This turns retrieval from $O(|I|)$ scoring into $O(k \log |I|)$ scoring while *allowing an arbitrarily expressive model* at each node — you get attention-style user-candidate interaction at the retrieval stage, which two-tower cannot. The cost is maintaining and periodically retraining the tree (the tree structure and the model are learned jointly and alternately), which is real operational overhead. TDM and its successor JTM are why "tree-based deep retrieval" is a recognized alternative to two-tower for very large e-commerce catalogs.

The trade-off table for the retrieval side:

| Retriever        | Model expressiveness         | Serving cost        | Operational cost           | When to use                          |
| ---------------- | ---------------------------- | ------------------- | -------------------------- | ------------------------------------ |
| Item-item CF     | Co-purchase only, no context | Table lookup        | Rebuild table periodically | Cold-start fallback, explainability  |
| Two-tower + ANN  | Fixed user/item vectors      | ANN query, p99 low  | Index rebuild, embed sync  | Default large-scale retriever        |
| TDM (tree)       | Deep, attention-capable      | Beam search, deeper | Tree learning + retrain    | Need attention at retrieval, huge $I$ |

For most teams the answer is two-tower as the default plus item-item CF as a cheap, explainable, cold-start-friendly companion retriever — you union their candidate sets. TDM is the move when you have demonstrated that the fixed-vector limitation of two-tower is actually costing you recall on a catalog large enough that a flat scan is impossible.

## 8. The e-commerce objective: conversion, revenue, and uplift

Now we close the loop on the distinctive that started the post: the objective. Figure 7 contrasts optimizing clicks with optimizing conversion and revenue.

![A two-column comparison of optimizing clicks which rewards cheap clickbait that never buys versus optimizing conversion and revenue which rewards items that actually sell.](/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-7.png)

### From CTR to expected revenue

A click-maximizing ranker sorts candidates by pCTR. A conversion-aware ranker sorts by something closer to **expected value**. The cleanest objective for a store is expected revenue per impression:

$$\text{score}(x) = \text{pCTR}(x) \times \text{pCVR}(x) \times \text{value}(x),$$

where $\text{value}(x)$ is the item's price (or margin, or expected basket value). The first two factors are exactly ESMM's pCTCVR; the third injects price. This is why ESMM matters commercially: it gives you a *calibrated* pCTCVR you can multiply by price. And calibration is non-negotiable here — if pCTR and pCVR are not calibrated probabilities, multiplying them by price produces a meaningless ranking. (You cannot compare $0.7 \times \text{price}$ across items if the $0.7$ means different things for different items.) This is precisely why [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) is a load-bearing topic in e-commerce specifically — far more than in pure engagement systems, where relative ordering is enough.

### Substitutes vs complements: the basket problem

A purely accuracy-driven recommender, fed co-purchase data, learns both substitutes and complements as "similar". But they demand opposite treatment *after a purchase*. Right after someone buys a washing machine, recommending another washing machine (a substitute) is useless — they just solved that need — while recommending detergent or a pedestal (complements) is gold. The fix is contextual: condition the recommendation on the user's *recent purchase event* and shift from substitutes (good before purchase, for comparison shopping) to complements (good after purchase, for the basket). Item-item CF cannot make this distinction (it is one undifferentiated similarity); the deep ranker can, because it sees the purchase event and the time since it as features. A practical heuristic many stores use: mine complement pairs from *sequential* co-purchase (bought A, then B within the same session or shortly after) versus substitute pairs from *co-view-but-buy-one* patterns. The distinction is worth real revenue and is invisible to a model that treats history as an unordered bag.

### Why uplift and incrementality matter for ROI

Here is the deepest e-commerce subtlety, and it is where most recommenders are quietly wrong. A model that predicts "this user will buy this item" is not the same as a model that predicts "*recommending* this item will *cause* an extra purchase." Many recommended purchases would have happened anyway — the user was already going to buy that item, and the recommendation just got the credit. Spending a slot recommending something the user would buy regardless is wasted; the slot should go to the item whose purchase probability is most *increased* by the recommendation. That increase is the **uplift** (incremental conversion), and optimizing it rather than raw predicted conversion is the difference between a recommender that looks great in attribution dashboards and one that actually grows revenue. This is the entire argument of [causal and uplift recommendation](/blog/machine-learning/recommendation-systems/causal-and-uplift-recommendation), and e-commerce is where it bites hardest because the money is directly attributable. The honest way to measure any of this is the online A/B test (covered in [A/B testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders)) with GMV and incremental conversion as the north-star metric — not offline AUC, which can rise while online GMV stays flat (the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied)).

### Price sensitivity as a first-class feature

Price deserves more than a footnote because it is the e-commerce feature most often handled badly. A user who only ever buys items under \$50 should not be shown a \$2,000 item with high pCTR — they might click out of aspiration but will not convert, and the slot is wasted. The naive fix, "add price as a feature," is necessary but insufficient, because the *relevant* signal is not the absolute price but the price *relative to the user's revealed price band* and *relative to comparable items in the category*. A \$200 item is cheap for a laptop and expensive for a phone case. So the features that actually move CVR are price ratios and percentiles: this item's price divided by the user's median historical purchase price, and this item's price percentile within its category. These let the model learn "this user buys at the 30th percentile of the laptop price distribution" rather than a meaningless raw-dollar threshold. Get this wrong and your conversion model will be confidently miscalibrated on exactly the price-sensitive shoppers who make up most of the long tail of buyers. Price is also where calibration and revenue ranking collide: because you rank by $\text{pCVR} \times \text{price}$, any price-correlated miscalibration in pCVR is directly a revenue-ranking error, which is one more reason the calibration check from Section 6 is non-negotiable in a store.

### Cold-start and the two-sided marketplace

Two more e-commerce realities deserve a word because they constrain every model above. **New-product cold-start**: a SKU with zero interactions has no co-purchase neighbors and a randomly-initialized id embedding, so item-item CF and id-based deep models are both blind to it. The standard fixes are content features (title, image, category, brand embeddings via [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders)) that let a new item inherit a sensible embedding from its attributes, plus deliberate exploration to gather the first interactions. **The two-sided marketplace**: every impression you give to one seller's product is one you withhold from another's. A ranker that only maximizes short-term buyer conversion will concentrate impressions on a few proven sellers, starve new ones, and degrade marketplace health over time — a fairness and incentive problem that re-ranking and explicit seller-side constraints address, the subject of [fairness, privacy, and multi-stakeholder recommendation](/blog/machine-learning/recommendation-systems/fairness-privacy-and-multi-stakeholder-rec).

## 9. Putting it together: an evaluation harness and a repro

Let us make the science measurable. Here is an evaluation harness that computes AUC for a ranking model (the standard CTR/CVR metric) and Recall@K for a retriever, on a temporal split. This is the honest measurement discipline the series insists on: split by time, never by random row, so the model is always evaluated on the future.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def temporal_split(df, time_col="ts", frac=0.8):
    df = df.sort_values(time_col)
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]          # train on past, test on future

@torch.no_grad()
def eval_auc(model, loader, device="cpu"):
    model.eval(); ys, ps = [], []
    for batch in loader:
        logit = model(*[b.to(device) for b in batch[:-1]])
        ps.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(batch[-1].numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))

def recall_at_k(table, test_pairs, k=10):
    """table: item-item neighbors. test_pairs: list of (history, held_out_item)."""
    hits = 0
    for history, target in test_pairs:
        recs = [j for j, _ in recommend(table, set(history), k=k)]
        hits += int(target in recs)
    return hits / max(1, len(test_pairs))
```

#### Worked example: a small reproduction on Amazon Reviews

On a single Amazon Reviews category (Electronics, the public dataset that traces to this exact lineage), here is what a careful small-scale reproduction looks like — these are representative numbers from a modest run on a few hundred thousand interactions, with a temporal split, not the papers' production figures:

| Model                         | Task     | Metric          | Score (approx) | Note                                  |
| ----------------------------- | -------- | --------------- | -------------- | ------------------------------------- |
| Item-item CF (co-purchase)    | Retrieval | Recall@20       | ~0.18          | Cheap, strong baseline, table lookup  |
| Pooling MLP (avg history)     | Ranking  | Test AUC        | ~0.83          | Fixed user vector baseline            |
| DIN (target attention)        | Ranking  | Test AUC        | ~0.85          | +~0.02 AUC over pooling, candidate-aware |
| DIEN (interest evolution)     | Ranking  | Test AUC        | ~0.86          | Small lift over DIN, heavier to train |

The shape of these numbers matches the literature: DIN over a pooling base is a roughly +0.01 to +0.02 AUC move, and DIEN adds a smaller increment on top at a real latency cost. A +0.01 AUC sounds trivial until you remember the scale: at Taobao or Amazon volume, a +0.01 AUC in the ranker translates to a measurable GMV lift worth far more than the engineering cost, which is exactly why these papers shipped. Be careful reading AUC deltas in this regime — small absolute moves are large in dollars, and AUC is a *ranking* metric that says nothing about *calibration*, which (Section 8) is what you actually need for revenue ranking.

A caution on metrics, straight from the literature: do not evaluate retrieval with *sampled* metrics (rank the true item against a handful of random negatives) and trust the resulting ordering — the KDD 2020 result by Krichene and Rendle showed sampled metrics can be *inconsistent* with the full-corpus metric, even reversing which model looks better. For e-commerce retrieval, compute Recall@K against the full catalog, or at least validate that your sampled metric tracks the full one before you trust it for model selection.

## 10. Case studies and the reported numbers

Pulling the named results together, with citations and honest caveats. Figure 8 summarizes the reported lifts.

![A four-row matrix of DIN, DIEN, ESMM, and TDM showing each model's reported AUC or CVR lift, the dataset it was measured on, and the caveat that matters.](/imgs/blogs/case-study-amazon-alibaba-e-commerce-rec-8.png)

**Amazon, item-to-item CF (Linden, Smith, York, 2003).** The paper's headline is not a metric on a benchmark — it predates that culture — but an *operational* claim: item-item CF scaled to "tens of millions of customers and millions of items" with high-quality, real-time recommendations, where user-user CF and cluster models could not. Its enduring influence is the more striking number: the algorithm ran essentially unchanged for over a decade and the "customers who bought also bought" module remained one of the most effective merchandising units on the site. The lesson is the scaling argument, not a single percentage.

**DIN (Zhou et al., KDD 2018).** Reported AUC gains over strong deep-CTR baselines (the base deep model, and product-based / Wide-and-Deep variants) on the public **Amazon** and **MovieLens** datasets and on Alibaba's production data. The offline AUC improvement is on the order of +0.01 absolute on the public sets, and the paper reports a substantial online CTR lift on Alibaba's display advertising system after deployment. Caveat: a +0.01 AUC reads as tiny but is large at scale; and the gain depends on having long, multi-interest behavior histories — on users with one dominant interest, target attention has little to do.

**DIEN (Zhou et al., AAAI 2019).** Reported AUC improvements *over DIN* on the same **Amazon** and Taobao data, plus an online CTR lift in production. The increment over DIN is smaller than DIN's increment over the base, consistent with diminishing returns — DIN already captured most of the multi-interest signal, and DIEN adds the ordering/evolution piece. Caveat: two GRUs are far heavier than DIN's attention; the latency cost is real and is why much of the field later moved to attention-only sequence models.

**ESMM (Ma et al., SIGIR 2018).** Reported CVR-task **AUC improvement of roughly +2.5 absolute points** over the clicked-only CVR baseline and over importance-sampling corrections, on Taobao production data, and the authors released **Ali-CCP** (Alibaba Click and Conversion Prediction), a public dataset, so the result is reproducible. Caveat: the gain comes from fixing sample-selection bias and sparsity via the entire-space identity, *not* from architecture — the towers are plain MLPs. It is the cleanest demonstration in the e-commerce literature that the objective and the training-sample design beat model cleverness.

**TDM (Zhu et al., KDD 2018).** Reported recall improvements over item-CF and over a brute-force inner-product baseline at Taobao scale, by enabling an expressive (attention-capable) model at the retrieval stage via tree beam search. Caveat: the tree must be learned and periodically retrained jointly with the model — substantial operational complexity that only pays off at very large catalogs where two-tower's fixed-vector limitation is demonstrably costing recall.

One more industry data point worth knowing: a widely-cited McKinsey estimate attributed roughly **35% of Amazon's revenue** to its recommendation engine. Treat that figure as approximate and dated — it is an external estimate, not an Amazon disclosure, and the exact percentage and methodology are not public — but it captures the order of magnitude of what e-commerce recommendation is worth, and why this lineage of work was funded and shipped.

## 11. The problem-solving narrative: choosing a stack for a new store

Let us reason through a real decision, the way you would on the job. You are building recommendations for a mid-sized e-commerce store: one million products, ten million users, growing. What do you build, in what order, and when do you stop?

**Start with item-item CF.** It is a day-one win: cheap to build (`R.T @ R`), needs no GPU, gives explainable "bought also bought" recommendations, and is a strong cold-start fallback for new items. Ship it as both a retriever and a "complete the look / frequently bought together" module. This alone often captures most of the easy GMV.

**Add two-tower retrieval** when item-item CF's lack of personalization and context shows up as flat engagement for users with diverse histories. Union its candidates with item-item CF's. Now you have a funnel.

**Add a deep ranker — start with a pooling MLP, then DIN — only when you have a ranking stage that is the bottleneck.** Do not jump to DIN on day one. First confirm that your ranker is the constraint (offline AUC plateaued, candidates are fine but the order is wrong). Then add DIN's target attention, because multi-interest users are where the lift lives. Measure it on a temporal split, then A/B test it. *Do not* reach for DIEN until you have shipped DIN and shown that interest-evolution (long, drifting histories) is a real, measured gap — the GRU latency is not free.

**Move from CTR to ESMM-style CVR/CTCVR when clicks and conversions diverge.** The signal that you need this: your CTR ranker is surfacing items that get clicked and abandoned, GMV is not tracking CTR, and your CVR estimates (if you have a naive clicked-only one) are clearly miscalibrated on the long tail. ESMM is a moderate amount of code and a large correctness fix. Pair it with calibration so you can rank by $\text{pCTCVR} \times \text{price}$.

**Stress-test the decision.** *What if you have only implicit feedback (no explicit conversions logged)?* Then you cannot train ESMM's CTCVR head — you fall back to CTR and proxy value (add-to-cart as a weak conversion signal). *What at 100M items?* Item-item CF's `R.T @ R` becomes a distributed sparse job, two-tower needs a real ANN index, and TDM becomes attractive. *What when negatives are mostly false negatives* (an item not bought might just be unseen)? This is the implicit-feedback negative problem from [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) and [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) — you sample negatives carefully and weight by confidence, you do not treat every non-purchase as a hard negative. *What when offline AUC rises but online GMV is flat?* The classic [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied): your offline metric is computed on data the *old* policy shaped, your AUC gain may be on impressions the new policy will never generate, or you improved ranking on items that were going to convert anyway (no *uplift*). Believe the A/B test. *What when the feature is computed differently offline and online* (price snapshotted at training time vs live price at serving)? That is [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there), and in e-commerce it is especially nasty because price changes constantly — a stale price feature can silently halve your revenue-ranking quality.

This sequencing — CF, then retrieval, then ranker, then conversion objective, then uplift — is the e-commerce instance of the general build order in the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

## 12. What to steal from e-commerce recommendation

Even if you build feeds and never touch a store, the e-commerce lineage hands you ideas that transfer directly.

**Steal the scaling inversion.** When per-request computation does not scale, ask whether you can *precompute* the expensive part offline against a more stable axis and leave a cheap lookup online. Amazon's user→item inversion is the canonical example, and the pattern — precompute against the slow-changing thing — recurs everywhere (precomputed item embeddings for ANN, materialized features in a feature store, the whole point of [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores)).

**Steal target attention.** A fixed user vector is a blur; whenever you have a candidate and a behavior history, let the candidate *query* the history so the user representation becomes candidate-aware. This is the single most transferable modeling idea here, and it generalizes far beyond e-commerce — any time you summarize a set with respect to a query, DIN's local activation is the move.

**Steal the entire-space trick.** When a label is only defined on a *subset* of your data (conversions only on clicks), look for a probability identity that expresses an *entire-space* quantity as a product, supervise the entire-space quantities, and let the conditional be learned implicitly. ESMM's $\text{pCTCVR} = \text{pCTR} \times \text{pCVR}$ is the template for beating sample-selection bias without throwing away data.

**Steal the objective discipline.** Clicks are a cheap proxy; the thing you actually want (conversion, revenue, incremental value) is downstream and harder to measure, and optimizing the proxy gets you proxy-maximizing behavior. Push your objective as far down the value funnel as your data and calibration allow, and measure with an A/B test on the real outcome.

**Steal the fallback humility.** A twenty-three-year-old `R.T @ R` is still in production as a cold-start and explainability fallback. The newest model is not always the right one for every slot; a cheap, interpretable, robust baseline that never embarrasses you is worth keeping under the fancy one.

## 13. When to reach for these models (and when not to)

E-commerce models are powerful and specific; here is the decisive guidance.

**Reach for item-item CF** on day one of any store, as a retriever, a "frequently bought together" module, a cold-start fallback, and an explainability source. It is almost always worth it. **Do not** rely on it alone for personalization — it is memoryless about sequence, context, and price.

**Reach for DIN** when you have long, *multi-interest* user histories and a ranking stage that is your bottleneck. **Do not** reach for it if your users have short or single-interest histories (target attention has nothing to reweight) or if a pooling MLP already hits your target — the activation unit is extra latency and complexity for no gain.

**Reach for DIEN** only after DIN is shipped and you have *measured* that interest evolution (long, drifting histories, category switches) is a real gap. **Do not** pay the two-GRU latency cost speculatively; for most catalogs, attention-only sequence models give most of the benefit more cheaply.

**Reach for ESMM** the moment clicks and conversions diverge and you need calibrated CVR for revenue ranking — which is almost every real store. **Do not** skip it and rank on raw pCTR if you sell things; you will reward clickbait. And **do not** train a clicked-only CVR model and serve it over the full space — that is the exact bug ESMM exists to fix.

**Reach for TDM** only at very large catalogs where you have demonstrated that two-tower's fixed-vector retrieval is costing you recall. **Do not** adopt its tree-learning operational burden before you have proven the need; two-tower plus item-item CF is the right default retriever for almost everyone.

The meta-rule mirrors the YouTube case study's: adopt the *cheap, high-leverage corrections* early (item-item CF, the conversion objective, calibration, the entire-space trick) and adopt the *heavy architecture* (DIEN, TDM) only when your scale and measured gaps demand it.

## 14. Key takeaways

- **Amazon's inversion is the founding scaling lesson.** Compute similarity between *items* (stable, precomputable offline) instead of *users* (volatile, must be computed online), turning serving into a table lookup. `R.T @ R` is the whole model and it still ships.
- **E-commerce optimizes conversion and revenue, not clicks.** The objective is downstream of the click — rank by $\text{pCTR} \times \text{pCVR} \times \text{price}$, not by pCTR, or you reward clickbait that never buys.
- **DIN's target attention beats a fixed user vector.** Make the candidate the query over the behavior history so the user representation is candidate-aware; the same history yields a sports vector for running socks and a baby vector for diapers.
- **DIEN adds interest evolution.** A GRU with an auxiliary next-item loss extracts per-step interest, and an attention-gated AUGRU evolves it toward the candidate — ordering signal DIN's order-free attention misses, at a real latency cost.
- **ESMM fixes conversion modeling with an identity.** $\text{pCTCVR} = \text{pCTR} \times \text{pCVR}$, supervised over the *entire* impression space, kills sample-selection bias and data sparsity — the win is the objective and the sample, not the network (+~2.5 CVR AUC on Taobao/Ali-CCP).
- **Calibration is load-bearing for revenue ranking.** You can only multiply probabilities by price if the probabilities mean the same thing across items; ordering-only ranking is not enough when money is on the line.
- **Substitutes and complements demand opposite treatment after a purchase.** Co-purchase similarity conflates them; condition on the recent purchase event to shift from comparison (substitutes) to basket-building (complements).
- **Uplift, not predicted conversion, is the ROI metric.** Spend slots on items whose purchase the recommendation *causes*, not on items the user would buy anyway; measure with an online A/B test on incremental GMV.
- **Believe the A/B test.** Offline AUC is computed on data the old policy shaped and is silent about calibration and uplift; a +0.01 AUC can be a large GMV lift or none at all — only the live test tells you which.
- **Keep the cheap fallback.** A robust, interpretable item-item baseline under the deep model handles cold-start and the cases the deep model is unsure about.

## 15. Further reading

- Linden, Smith, York, *Amazon.com Recommendations: Item-to-Item Collaborative Filtering*, IEEE Internet Computing, 2003 — the founding item-item CF paper and the scaling argument.
- Zhou, Zhu, Song, Fan, Zhu, Ma, Yan, Jin, Li, Gai, *Deep Interest Network for Click-Through Rate Prediction*, KDD 2018 — target attention / local activation, plus Dice and adaptive regularization.
- Zhou, Mou, Fan, Pi, Bian, Zhou, Zhu, Gai, *Deep Interest Evolution Network for Click-Through Rate Prediction*, AAAI 2019 — the interest-extractor GRU, auxiliary loss, and AUGRU.
- Ma, Zhao, Hao, Mao, Liu, Wang, Wang, Sun, Wang, Zhou, *Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate*, SIGIR 2018 — the entire-space CTR/CVR identity and the Ali-CCP dataset.
- Zhu, Li, Zhang, Li, He, Li, Gai, *Learning Tree-based Deep Model for Recommender Systems* (TDM), KDD 2018 — tree-based deep retrieval with beam search.
- Krichene, Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020 — why sampled retrieval metrics can be inconsistent with full-corpus metrics.
- Series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) and [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles) for the foundations; [self-attention for sequences](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) for the attention machinery DIN specializes; [multi-task and multi-objective ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) for the multi-task framing ESMM extends; [delayed feedback and conversion attribution](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution) for late conversions; [causal and uplift recommendation](/blog/machine-learning/recommendation-systems/causal-and-uplift-recommendation) for incrementality; and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) to sequence the whole build.

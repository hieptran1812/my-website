---
title: "What Is a Recommender System? From Click to Production"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A practitioner's first-principles tour of the recommendation problem, the retrieval-ranking-reranking funnel, the feedback loop, and a runnable MovieLens baseline you can reproduce by lunch."
tags:
  [
    "recommendation-systems",
    "recsys",
    "collaborative-filtering",
    "matrix-factorization",
    "ranking",
    "retrieval",
    "ndcg",
    "machine-learning",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/what-is-a-recommender-system-1.png"
---

A feed launches. Within three weeks the same ten viral clips dominate every user's home page, watch time per session is up but watch time per user is flat, and the catalog of two million videos is functionally a catalog of ten. Nobody wrote a bug. The recommender is doing exactly what it was trained to do: show what got clicked, log what got shown, and train the next model on those logs. That self-reinforcing loop is the single most important thing to understand before you write a line of recommender code, and it is invisible if you treat recommendation as "just another classifier."

This is the opening post of a long series, **Recommendation Systems: From Click to Production**. The goal across all 54 posts is to be three things at once: *scientific* (the math behind why a technique works), *practical* (runnable code in the real toolchain), and *measured* (before-and-after numbers on named datasets, with honest caveats). This first post sets the frame everything else hangs from. By the end you will be able to state precisely what makes the recommendation problem different from supervised classification, draw the production funnel that turns a billion items into the twenty a user actually sees, explain the feedback loop that is both the engine and the bias source, and run an end-to-end baseline on MovieLens that already beats showing everyone the same popular list.

We will keep one running example in view the whole way: a movie recommender trained on the MovieLens dataset, the `iris` of recsys. It is small enough to run on a laptop and rich enough to expose every real problem in miniature: positive-only feedback, a sparse user-item matrix, a temporal split, and the gap between "predicts ratings well" and "puts good movies at the top of the list." The picture below is the destination we are building toward, and the spine of the whole series.

![Diagram of the recommendation funnel narrowing a one billion item catalog through retrieval, ranking, and re-ranking down to about twenty shown items with a latency budget annotation](/imgs/blogs/what-is-a-recommender-system-1.png)

## 1. The recommendation problem, stated honestly

Strip away the product framing and a recommender answers one question: given a user $u$ and the moment they showed up, which handful of items from an enormous catalog should we show, in what order, so that they engage? That sounds like a ranking problem, and it is. But the way the problem is *posed* breaks four assumptions that ordinary supervised learning quietly depends on, and every one of those breaks is a research subfield.

**Break one: the catalog is enormous and the output is a ranking, not a class.** A web recommender might choose from $10^8$ to $10^9$ items. You are not predicting one of $K$ fixed labels; you are sorting a candidate set and returning the top of it. The natural objective is therefore a top-$K$ ranking quality, not point accuracy on a single prediction. We will make that distinction rigorous in section 4, because it is the most common conceptual mistake newcomers make: they minimize rating-prediction error, watch RMSE drop, and ship a model that ranks no better than before.

**Break two: the feedback is positive-only and implicit.** In a classifier you have labeled positives and labeled negatives. In a recommender you mostly have implicit signals: a click, a play, a purchase, a dwell time. A click is a noisy positive. The *absence* of a click is not a negative — the user may simply never have seen the item. This is the positive-unlabeled, or one-class, nature of implicit feedback, and it forces a whole machinery of negative sampling that has no analog in standard classification. Resnick and Varian's 1997 framing of "collaborative filtering" already assumed we learn from what people *did*, not from clean labels, and that has only become more true as explicit ratings dried up.

**Break three: the observed data is missing-not-at-random (MNAR).** Which user-item pairs you even *observe* is decided by the previous recommender. A movie that was never recommended has no impressions, so it has no clicks, so the model learns it is uninteresting, so it never gets recommended. The data is not a random sample of user preferences; it is a sample shaped by the system's own past decisions. In statistics this is selection bias; in recsys it shows up as **position bias** (items higher on the page get clicked more regardless of relevance) and **popularity bias** (popular items get shown more, so they accumulate more positives). We will return to this as a feedback-loop fixed point in section 5.

**Break four: the label space is a sparse high-cardinality ID space.** The core features are user IDs and item IDs, each a categorical variable with millions of values, each appearing a handful of times. You cannot one-hot encode them sensibly; you learn an embedding per ID. The interaction matrix is brutally sparse — MovieLens-1M has about 1 million ratings across roughly 6,000 users and 3,700 items, a density near 4.5%, and that is *dense* by industry standards. A real production matrix is often well below 0.1% dense. Sparsity is why pure memorization fails and why latent-factor models, which generalize across IDs through shared low-dimensional vectors, are the workhorse.

Hold those four breaks in mind. The figure below lines them up against vanilla supervised learning so you can see that recommendation is not a flavor of classification with a big output layer — it is a different problem with a different objective and a different evaluation.

![Matrix comparing recommendation systems against vanilla supervised machine learning across feedback signal, label space, objective, evaluation, and data distribution](/imgs/blogs/what-is-a-recommender-system-5.png)

The row that matters most is the last one. In ordinary supervised learning we assume the training and test data are drawn from the same distribution, more or less independently and identically. In recommendation, the training distribution is *generated by the model we are about to replace*. That single fact is why offline metrics and online metrics diverge, why A/B tests are non-negotiable, and why an entire track of this series is devoted to bias and the offline-online gap.

#### Worked example: the funnel arithmetic

Suppose the catalog has $N = 10^9$ items and you have a 100 ms total latency budget to produce a page. A strong deep ranking model costs roughly 1 ms of compute per scored item on a serving host (feature lookup, embedding gather, a few dense layers). Score the whole catalog with it and you spend $10^9 \times 1\,\text{ms} = 10^6\,\text{seconds}$, about 11.6 days, per request. That is not a tuning problem; it is off by eight orders of magnitude. Now stage it. A cheap retrieval model — a dot product between a user vector and item vectors, served by approximate nearest neighbor (ANN) search — returns the top 500 candidates in single-digit milliseconds. The expensive ranker scores only those 500: $500 \times 1\,\text{ms} = 500\,\text{ms}$... still over budget, so the ranker is built lean (sub-0.1 ms per item, batched on a GPU) to bring 500 items down to about 50 in well under the budget. A re-ranking pass then trims 50 to roughly 20 with diversity and business rules. The arithmetic *forces* a funnel: $10^9 \to 500 \to 50 \to 20$. No single model can be both cheap enough for the first arrow and accurate enough for the last.

## 2. Why anyone pays for this: the business case

Recommendation is not a vanity feature; it is, for many companies, the product. The commonly cited figures are worth stating with the right amount of skepticism. It has been widely reported, originating in industry talks and press around 2012-2013, that roughly **35% of what people buy on Amazon** and around **75-80% of what people watch on Netflix** come from recommendations rather than from search or direct navigation. Treat those as order-of-magnitude, dated, and self-reported — but the direction is not in doubt. When the default surface a user lands on is a recommended feed, the recommender is the funnel through which most engagement flows.

The mechanism is simple. Most catalogs are too large to browse. Search works only when the user already knows what they want. Recommendation captures the much larger volume of *latent* intent — the user who would have watched something good but would never have searched for it. The metrics that move are concrete:

- **Click-through rate (CTR)** on a feed or carousel: did the user engage with what we showed?
- **Watch time / dwell time / session length**: did the engagement have depth, not just a click?
- **Retention** (D1, D7, D30): did good recommendations bring the user back? This is the metric most correlated with long-term value and the hardest to move offline.
- **Gross merchandise value (GMV) / revenue per session** in commerce: did better ranking convert to dollars?

Here is the honest part, and it sets the tone for the series. **These online metrics are the only ones that truly count, and they are exactly the ones you cannot measure offline.** Your offline NDCG can rise while online watch time falls, because offline you are scoring a model on data the *old* model collected. We will spend a whole track on that gap. For now, internalize the hierarchy: online business metrics are the truth, offline ranking metrics are a cheap and biased proxy, and rating-prediction error is a proxy for the proxy.

#### Worked example: sizing a CTR lift

Take a feed serving 50 million daily active users, each seeing on average 40 recommended items per day, at a baseline CTR of 6%. That is $50{,}000{,}000 \times 40 = 2 \times 10^9$ impressions and $1.2 \times 10^8$ clicks per day. Suppose a new ranker lifts CTR by a *relative* 3%, from 6.00% to 6.18%. The extra clicks are $2 \times 10^9 \times (0.0618 - 0.0600) = 3.6 \times 10^6$ additional engaged actions per day. If each engaged session is worth, say, \$0.01 in downstream value (ads, conversion, retention-adjusted), that is \$36{,}000 per day, or roughly \$13.1M per year, from a change that barely registers as a number. This is why teams fight over the third decimal place of CTR and why a robust A/B testing culture, not a clever model, is what separates teams that ship from teams that don't.

## 3. The production funnel: one model cannot do it all

The arithmetic in section 1 already revealed the architecture. Production recommenders are **multi-stage funnels**, and this funnel is the organizing spine of the entire series. Each stage has a different job, a different model, a different latency budget, and a different definition of "good." Walk the stages.

**Stage 1, retrieval (also called candidate generation):** Reduce the catalog from $10^9$ to a few hundred candidates in a few milliseconds. The model here must be embarrassingly cheap to evaluate against the whole catalog. The dominant pattern is the **two-tower** model: a user tower produces a vector $\mathbf{u}$, an item tower produces a vector $\mathbf{v}_i$, and the score is the dot product $\mathbf{u}^\top \mathbf{v}_i$. Because item vectors do not depend on the user, you precompute all $10^9$ of them, build an ANN index, and at request time turn retrieval into a maximum-inner-product search (MIPS): find the items whose vectors point most in the direction of the user vector. ANN trades a little recall for a lot of speed. Retrieval optimizes for **recall** — it is fine to over-include; the next stage will sort it out. What is *not* fine is to miss a relevant item here, because nothing downstream can recover it.

**Stage 2, ranking:** Take the few hundred candidates and score them with a heavy, feature-rich model — a deep network that ingests user features, item features, cross features, and context (time of day, device, recent history). This is where the expensive modeling lives: DLRM, Wide and Deep, DeepFM, DCN, DIN, and their descendants, each a future post. Ranking optimizes for a calibrated, fine-grained ordering and often predicts *multiple* objectives at once (probability of click, probability of long watch, probability of purchase) and combines them. The budget here is tens of items times a sub-millisecond per-item cost.

**Stage 3, re-ranking:** Take the top tens from the ranker and apply the rules that a pure relevance score cannot express: diversity (don't show ten near-identical items), freshness, deduplication, business constraints (sponsored slots, fairness, "don't show the thing they just bought"), and list-level objectives where the value of an item depends on what else is on the page. This stage is small, cheap, and disproportionately important for the felt quality of the product.

![Acyclic chain diagram of the recommendation feedback loop from serving to impressions to clicks to log store to training the next model, with branches to position and popularity bias](/imgs/blogs/what-is-a-recommender-system-2.png)

The reason this split exists is a hard constraint, not an aesthetic preference. You have a latency budget — say 100 ms p99 for the whole page. You cannot spend it scoring a billion items with a heavy model. So you put the cheap, high-recall model where the candidate set is huge and the expensive, high-precision model where the candidate set is small. The funnel is the resolution of the tension between *coverage* (look at everything) and *cost* (be fast and accurate). When someone says "we built one big end-to-end model for recommendation," ask them how it scores a billion items in 50 ms. It doesn't; there is a retrieval stage hiding inside, even if they call it an index.

A subtle point that trips up newcomers: the stages do not share an objective, and that is by design. Retrieval is graded on recall, ranking on calibrated ordering, re-ranking on list-level quality. Optimizing each stage in isolation can be locally right and globally wrong — a retrieval model that maximizes recall against the *old* ranker's labels can starve the new ranker of the candidates it would have ranked highly. This stage-misalignment is a recurring failure mode, and we devote posts to retrieval-ranking consistency and to training retrieval on the right signal.

#### Worked example: budgeting latency across stages

Page budget: 100 ms p99 end to end. Allocate it. Network and serialization eat ~15 ms. Feature fetch from the online store (user features, recent history) eats ~20 ms. That leaves ~65 ms for model work. Retrieval against an ANN index over $10^9$ vectors at recall@500 returns in ~8 ms (a single HNSW or IVF query). Ranking 500 candidates through a deep net, batched on one GPU, at ~0.05 ms per item is ~25 ms. Re-ranking 50 items with diversity logic is ~5 ms. Total model time ~38 ms, comfortably inside the 65 ms. Notice the leverage: doubling the candidate set from 500 to 1000 adds ~25 ms to ranking (often the whole remaining budget) but might add only one or two points of recall. That recall-versus-latency Pareto curve is the single most-tuned dial in production recsys, and it is why we obsess over ANN index choice (`faiss` IVF-PQ versus HNSW versus flat) in the retrieval track.

## 4. The science: scoring, top-K, and why ranking is not RMSE

Now make the problem precise, because the math is what tells you *why* the funnel and the objective look the way they do.

A recommender is, at heart, a **scoring function** $s(u, i; \theta)$ that assigns a real-valued relevance score to a user-item pair, parameterized by $\theta$. Serving is the selection

$$
\hat{R}_K(u) = \operatorname*{arg\,top\text{-}K}_{i \in \mathcal{C}(u)} \, s(u, i; \theta),
$$

the $K$ highest-scoring items from a candidate set $\mathcal{C}(u)$ (the whole catalog at retrieval; the few hundred at ranking). Two things in that expression deserve attention. First, only the *order* of $s$ within the top region matters — any monotone transform of $s$ that preserves the top-$K$ ordering gives the identical served page. Second, the objective lives on a *list*, the top-$K$, not on any single score.

### 4.1 The implicit-feedback likelihood

With explicit ratings you would fit $s(u,i)$ to a numeric rating $r_{ui}$ and minimize squared error. With implicit feedback you have a binary-ish signal $y_{ui} \in \{0, 1\}$ where $1$ means "interacted" and $0$ means "did not observe an interaction" — and that $0$ is not a true negative. A common pointwise treatment models the probability of interaction with a logistic link,

$$
P(y_{ui} = 1 \mid \theta) = \sigma\big(s(u, i; \theta)\big), \qquad \sigma(x) = \frac{1}{1 + e^{-x}},
$$

and maximizes a weighted log-likelihood over observed positives and *sampled* negatives,

$$
\mathcal{L}_{\text{point}}(\theta) = \sum_{(u,i) \in \mathcal{D}^+} \log \sigma(s_{ui}) \; + \; \sum_{(u,j) \in \mathcal{D}^-} \log\big(1 - \sigma(s_{uj})\big),
$$

where $\mathcal{D}^-$ is drawn by negative sampling because we cannot enumerate all the unobserved pairs. This is the implicit-feedback analog of logistic regression, and it is the backbone of pointwise ranking models. The whole art of *which* negatives to sample — uniform, popularity-corrected, or hard negatives mined from the model — is a future post, because it changes results more than the model architecture does.

### 4.2 Why pairwise ranking beats pointwise for top-K

Here is the crux, and it is worth deriving rather than asserting. The pointwise loss tries to push every positive score toward 1 and every negative toward 0 *in absolute terms*. But serving only cares about the *relative* order. Bayesian Personalized Ranking (Rendle et al., 2009) optimizes the order directly. For a user $u$, an observed positive item $i$, and an unobserved item $j$, BPR maximizes the probability that the positive outscores the negative:

$$
\mathcal{L}_{\text{BPR}}(\theta) = \sum_{(u,i,j) \in \mathcal{D}_S} \log \sigma\big(s_{ui} - s_{uj}\big) - \lambda \lVert \theta \rVert^2 .
$$

Look at the gradient of one term with respect to a parameter $\theta$. Let $x_{uij} = s_{ui} - s_{uj}$. Then

$$
\frac{\partial}{\partial \theta} \log \sigma(x_{uij}) = \big(1 - \sigma(x_{uij})\big) \cdot \frac{\partial x_{uij}}{\partial \theta} = \underbrace{\frac{e^{-x_{uij}}}{1 + e^{-x_{uij}}}}_{\text{weight}} \cdot \frac{\partial (s_{ui} - s_{uj})}{\partial \theta}.
$$

Read what that weight does. When the order is already correct and confident ($x_{uij}$ large and positive), the weight $\to 0$: BPR stops pushing pairs it already ranks right. When the order is wrong ($x_{uij}$ negative), the weight $\to 1$: the gradient concentrates on the *misranked* pairs. The optimizer spends its effort on order errors, not on driving already-correct absolute scores closer to 0 or 1. A pointwise loss, by contrast, keeps spending gradient to make a positive's score 0.99 instead of 0.95 even when it already outranks every negative — effort that does nothing for the served page. That is the formal reason pairwise ranking typically wins on top-$K$ metrics: **the gradient sees the order, not the absolute value.**

### 4.3 Sampled softmax and the log Q correction (a preview with the math)

There is a third family of losses that dominates retrieval, and it is worth previewing the math now because it explains a flag you will trip over: the $\log Q$ correction. Candidate generation is naturally a multiclass problem — given a user, predict *which* of the $N$ items they will interact with — with a softmax over the whole catalog:

$$
P(i \mid u) = \frac{\exp\big(s(u,i)\big)}{\sum_{j=1}^{N} \exp\big(s(u,j)\big)}.
$$

The denominator sums over all $N \approx 10^9$ items, which is intractable per step. **Sampled softmax** replaces the full sum with the positive plus a small set of sampled negatives. But the negatives are not drawn uniformly — for efficiency and signal, they come from a proposal $Q(j)$ that is often proportional to item popularity (in-batch negatives, where the other positives in the minibatch serve as negatives, are implicitly drawn from the popularity distribution). Sampling negatives from a non-uniform $Q$ *biases* the estimated softmax toward whatever $Q$ over-samples. The fix is to correct each sampled logit by subtracting $\log Q(j)$:

$$
s'(u, j) = s(u, j) - \log Q(j).
$$

The intuition is importance weighting: a popular item that $Q$ proposes ten times more often must have its logit discounted so it does not dominate the partition function. Without this correction your retrieval model quietly becomes a popularity model — it will retrieve the head of the catalog and starve the tail, which then feeds the popularity bias of section 5. The correction is one line of code (`logits -= tf.math.log(candidate_sampling_prob)` or its PyTorch equivalent) and it is the difference between a retrieval tower that personalizes and one that re-derives popularity. We dedicate a full post to in-batch negatives, the $\log Q$ correction, and hard-negative mining in the loss track, because this single detail moves recall more than doubling the embedding dimension does.

### 4.4 Retrieval as maximum-inner-product search, and the recall-latency law

One more piece of science the funnel rests on. A two-tower model scores by dot product $s(u,i) = \mathbf{u}^\top \mathbf{v}_i$, so serving retrieval is: given the user vector $\mathbf{u}$, find the $K$ item vectors $\mathbf{v}_i$ with the largest inner product. That is **maximum-inner-product search** (MIPS). An exact scan over $N$ items costs $O(Nd)$ — fine for $N = 10^4$, impossible for $N = 10^9$ at single-digit milliseconds. Approximate nearest neighbor (ANN) indexes — IVF, HNSW, product quantization — trade exactness for speed: they probe only a fraction of the catalog and return *most* of the true top-$K$. The governing trade-off is a **recall-latency curve**. For an IVF index that partitions the space into $n_{\text{list}}$ cells and probes $n_{\text{probe}}$ of them at query time, latency scales roughly as $O\!\left(\frac{n_{\text{probe}}}{n_{\text{list}}} N d\right)$ while recall climbs monotonically (with diminishing returns) in $n_{\text{probe}}$. You buy recall with latency, point by point. Choosing $n_{\text{probe}}$ — or the HNSW `efSearch`, or the PQ code size — is choosing a point on that curve, and it is the most consequential serving decision in retrieval. We will benchmark `faiss` IndexIVFFlat, IndexHNSWFlat, and IndexIVFPQ on this exact curve in the deep-retrieval track.

### 4.5 NDCG, and a proof that RMSE can fall while ranking is unchanged

The metric that captures top-$K$ quality with position weighting is **Normalized Discounted Cumulative Gain**. For a ranked list, discounted cumulative gain at $K$ is

$$
\text{DCG@}K = \sum_{p=1}^{K} \frac{2^{\text{rel}_p} - 1}{\log_2(p + 1)},
$$

where $\text{rel}_p$ is the relevance of the item at position $p$ (for binary relevance, $1$ if the user engaged, $0$ otherwise). Dividing position $p$'s gain by $\log_2(p+1)$ encodes the truth that position 1 matters far more than position 10. NDCG@$K$ normalizes by the ideal DCG (the best possible ordering), giving a number in $[0,1]$. Recall@$K$ is simpler — the fraction of a user's held-out relevant items that landed in the top $K$ — and it ignores order within the top $K$.

Now the proof that point accuracy and ranking are different objectives. Consider one user and two items, A (truly relevant) and B (not), with true ratings $r_A = 5$, $r_B = 3$. Model 1 predicts $\hat{r}_A = 4.0, \hat{r}_B = 3.5$. Model 2 predicts $\hat{r}_A = 3.4, \hat{r}_B = 3.6$. Compute squared errors. Model 1: $(5-4.0)^2 + (3-3.5)^2 = 1.0 + 0.25 = 1.25$. Model 2: $(5-3.4)^2 + (3-3.6)^2 = 2.56 + 0.36 = 2.92$. Model 1 has lower error, so by RMSE it is "better." But check the *order*. Model 1 ranks A above B ($4.0 > 3.5$): correct. Model 2 ranks B above A ($3.6 > 3.4$): wrong — it puts the irrelevant item on top. Now flip it: I can construct a Model 3 with $\hat{r}_A = 4.9, \hat{r}_B = 4.85$ that has *tiny* RMSE yet, if a fourth item C with $\hat{r}_C = 4.88$ is truly irrelevant, ranks C between A and B and corrupts the top. The point is general: RMSE is invariant to where errors fall in the score range, but NDCG is dominated by errors near the top of the order. **Lowering RMSE provably does not guarantee — and can actively harm — top-$K$ ranking.** This is the single most expensive lesson a recsys newcomer learns the hard way, and it is why the Netflix-Prize-era focus on rating-prediction RMSE gave way to ranking objectives for production systems.

![Before and after comparison contrasting a point accuracy view that lowers RMSE with a ranking view that improves Recall at ten and NDCG at ten on the same predictions](/imgs/blogs/what-is-a-recommender-system-4.png)

## 5. The feedback loop: the engine and the bias source

The second spine of the series is the **feedback loop**, drawn above as a chain on purpose. A live recommender does not learn from a fixed dataset; it learns from its own logs. The cycle is: **serve** a page, **log** the impressions and the clicks, **train** the next model on those logs, **deploy** it, and serve again. This is what lets a recommender adapt to a non-stationary world — new items, new trends, new users — without a human relabeling anything. It is the engine.

It is also the bias source, and the mechanism is worth making rigorous. Define the long-run "exposure" $e_i$ of item $i$ as the probability the system shows it. Clicks accrue in proportion to exposure times relevance: $\text{clicks}_i \propto e_i \cdot \rho_i$, where $\rho_i$ is true relevance. The next model is trained to predict clicks, so it learns a score $\hat{s}_i$ that tracks $e_i \cdot \rho_i$, *not* $\rho_i$. The new exposure is then an increasing function of that learned score, $e_i' = g(\hat{s}_i)$. Iterate. An item that started with a small exposure edge gets more clicks, which raises its learned score, which raises its exposure, which raises its clicks. The map $e \mapsto e'$ has a stable fixed point at the *popular* items and an unstable one everywhere else: small differences in initial exposure get amplified until a few items absorb almost all the exposure. That is **popularity bias** as a dynamical-systems fixed point — not a modeling mistake, but an emergent property of optimizing for logged clicks inside a closed loop.

Make the amplification concrete with a toy that you could simulate in five lines. Start two items with *equal* true relevance $\rho = 1$ but a tiny exposure asymmetry: item A has exposure share 0.52, item B has 0.48. Suppose next-round exposure is proportional to this round's clicks, which are proportional to exposure times relevance, then renormalized: $e_i' = e_i \rho_i / \sum_j e_j \rho_j$. Round one leaves them where they started (relevance is equal), but add the smallest nudge — A is shown one position higher on average, so position bias multiplies its clicks by 1.1. Now A's effective click rate is $0.52 \times 1.1 = 0.572$ versus B's $0.48$, giving new shares $0.544$ and $0.456$. Iterate the same 1.1 nudge: $0.544 \times 1.1 = 0.598$ versus $0.456$, shares $0.567$ / $0.433$; then $0.594$ / $0.406$; the gap widens every round and converges toward A taking essentially all the exposure, *despite identical true relevance*. The only thing that differed was a starting position. That is the popularity-bias fixed point in numbers, and it is why a recommender left alone will quietly turn a rich catalog into a short list of winners.

Two more biases ride on the same loop. **Position bias**: items shown higher get clicked more *regardless of relevance*, so the click signal conflates "good" with "was on top." **Selection / MNAR bias**: you only observe feedback for items you chose to show, so the model never learns about items it never tried. The fix family — inverse-propensity scoring (IPS), exploration, position-debiasing towers, and unbiased learning-to-rank — is an entire track. The intuition to carry now: **the loop is why your offline metric, computed on logged data, systematically lies about online behavior.** The logged data is the equilibrium of the *old* policy; your new policy will visit states the logs never saw.

A blunt operational consequence: **you cannot fully trust offline evaluation, and you must A/B test.** The offline number is a necessary filter (it catches obvious regressions cheaply) but not a sufficient verdict. Every serious recsys org runs online experiments because the loop guarantees the offline-online gap. We have a whole post on why offline NDCG and online engagement diverge and how to make offline less of a liar (temporal splits, propensity weighting, counterfactual estimators).

## 6. Implicit vs explicit feedback, and the offline-online gap (previews)

Two ideas appear so often that it is worth planting them now, with full posts later.

**Explicit vs implicit feedback.** Explicit feedback is a deliberate rating: five stars, a thumbs up, a 1-to-10 score. It is high-signal but rare, biased toward extreme opinions (people rate things they loved or hated), and increasingly absent from modern products. Implicit feedback is a behavioral trace: a click, a play, a 90% completion, a re-watch, a purchase, a long dwell. It is abundant and unbiased about *intent to act*, but it is noisy (a misclick), ambiguous (a click is not satisfaction), and crucially **one-class** — you see positives and a sea of unlabeled. Almost all production systems run on implicit feedback, which is why negative sampling, confidence weighting (Hu, Koren, Volinsky 2008 weight a positive by how strong the signal is), and the implicit-feedback likelihood of section 4 are foundational rather than niche.

The confidence-weighting idea deserves a sentence more, because it is the bridge from "a click is a positive" to a trainable objective. Hu, Koren, and Volinsky treat the *strength* of an implicit signal as a confidence $c_{ui} = 1 + \alpha r_{ui}$, where $r_{ui}$ is, say, the number of times the user played the item, and they fit the latent factors to a preference indicator weighted by $c_{ui}$. The effect is that "watched it five times" pulls the embeddings harder than "clicked once," and the sea of unobserved pairs gets a small baseline confidence rather than being treated as hard negatives. This is the formalization that makes implicit-feedback ALS work in practice, and it is one call away in the `implicit` library.

**The offline-online gap.** Offline, you replay logged interactions on a temporal split and compute Recall@K / NDCG@K. Online, you run an A/B test and read CTR / watch-time / retention. These can — and routinely do — disagree, for three compounding reasons: distribution shift (the new model visits item-user pairs absent from the logs), missing-not-at-random data (the logs are the old policy's equilibrium), and position bias (logged clicks reward position, not just relevance). The mature stance is to treat offline metrics as a *cheap, biased pre-filter* and online experiments as the *truth*, and to invest in counterfactual / off-policy estimators that make the offline number a less biased predictor of the online one. This tension is the reason the series has a dedicated evaluation track sitting between the modeling tracks and the troubleshooting track.

It helps to keep the vocabulary of metrics straight from the start, because different stages of the funnel are graded on different ones. The table below is the working glossary the series uses; each metric gets a full post in the evaluation track, but you should recognize all of them now.

| Metric | What it measures | Where it is used | Order-sensitive? |
| --- | --- | --- | --- |
| Recall@K | fraction of relevant items in the top K | retrieval, top-K eval | no (within top K) |
| Precision@K | fraction of the top K that are relevant | ranking quality | no (within top K) |
| NDCG@K | position-weighted relevance, normalized | the headline ranking metric | yes |
| MAP | mean average precision across users | ranked-list quality | yes |
| MRR | reciprocal rank of the first hit | first-relevant-item tasks | yes |
| HitRate@K | did any relevant item land in top K | leave-one-out eval | no |
| AUC | probability a positive outscores a negative | pointwise ranking, calibration check | pairwise |
| LogLoss | calibration of predicted probabilities | the ranking stage, ad-style CTR | n/a |

The split to internalize: retrieval cares about Recall@K (don't miss the relevant items), the ranking stage cares about NDCG and calibration (order them well and predict honest probabilities), and the business cares about none of these directly — it cares about the online metrics these proxy for. Choosing the wrong metric for the stage is the section-4 mistake repeated at a different altitude.

## 7. The classic baselines, and a runnable MovieLens flow

Theory earns its keep only when it runs. Let us build the smallest honest end-to-end recommender: load MovieLens, do a proper temporal split, implement two non-personalized and one personalized baseline, and evaluate them with Recall@10 and NDCG@10. This is the practical core of the post; everything later in the series is a more sophisticated version of this loop.

### 7.1 Load the data and split by time

The cardinal sin in recsys evaluation is a *random* train/test split, which leaks the future into the past — you would be predicting a user's January clicks using their March clicks. Always split by time. We hold out each user's most recent interactions for the test set (a "leave-last-N-out" temporal split).

```python
import pandas as pd
import numpy as np

# MovieLens-1M: download from grouplens.org, columns separated by "::"
ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "item_id", "rating", "timestamp"],
)

# Treat any rating as an implicit positive (the user chose to watch and rate).
# Many studies threshold at rating >= 4; we keep all to maximize density here.
ratings = ratings.sort_values("timestamp")

# Temporal leave-last-N-out: last 10 interactions per user go to test.
def temporal_split(df, holdout=10):
    df = df.sort_values(["user_id", "timestamp"])
    test_idx = df.groupby("user_id").tail(holdout).index
    test = df.loc[test_idx]
    train = df.drop(test_idx)
    # Only evaluate users who still have history in train.
    valid_users = set(train["user_id"]).intersection(test["user_id"])
    test = test[test["user_id"].isin(valid_users)]
    return train, test

train, test = temporal_split(ratings, holdout=10)
print(f"train interactions: {len(train):,}  test interactions: {len(test):,}")
print(f"users: {ratings.user_id.nunique():,}  items: {ratings.item_id.nunique():,}")
```

### 7.2 Build the sparse interaction matrix

Recommenders live and die by sparse matrices. We map raw IDs to contiguous indices and build a user-by-item CSR matrix of ones.

```python
from scipy.sparse import csr_matrix

user_ids = np.sort(ratings.user_id.unique())
item_ids = np.sort(ratings.item_id.unique())
u_index = {u: k for k, u in enumerate(user_ids)}
i_index = {i: k for k, i in enumerate(item_ids)}
n_users, n_items = len(user_ids), len(item_ids)

def to_matrix(df):
    rows = df.user_id.map(u_index).values
    cols = df.item_id.map(i_index).values
    data = np.ones(len(df), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

train_mat = to_matrix(train)        # user x item, implicit ones
print(f"density: {train_mat.nnz / (n_users * n_items):.4%}")
```

### 7.3 The metrics: Recall@K and NDCG@K, computed honestly

We compute *full* metrics over the entire item catalog (no sampled negatives — sampled metrics are inconsistent, as Krichene and Rendle showed at KDD 2020), masking out items the user already saw in train.

```python
def recall_ndcg_at_k(scores_row, train_items, test_items, k=10):
    """scores_row: 1D array of scores over all items for one user."""
    scores = scores_row.copy()
    scores[list(train_items)] = -np.inf          # mask already-seen items
    topk = np.argpartition(-scores, k)[:k]
    topk = topk[np.argsort(-scores[topk])]       # order the top-k
    hits = [1 if it in test_items else 0 for it in topk]

    n_rel = len(test_items)
    recall = sum(hits) / min(n_rel, k) if n_rel > 0 else 0.0

    dcg = sum(h / np.log2(p + 2) for p, h in enumerate(hits))
    ideal_hits = min(n_rel, k)
    idcg = sum(1.0 / np.log2(p + 2) for p in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg

def evaluate(score_fn, k=10, max_users=2000):
    test_by_user = test.groupby("user_id")["item_id"].apply(
        lambda s: set(s.map(i_index))
    )
    train_by_user = train.groupby("user_id")["item_id"].apply(
        lambda s: set(s.map(i_index))
    )
    recalls, ndcgs = [], []
    for u in list(test_by_user.index)[:max_users]:
        ui = u_index[u]
        scores = score_fn(ui)                    # shape (n_items,)
        r, n = recall_ndcg_at_k(
            scores, train_by_user.get(u, set()), test_by_user[u], k
        )
        recalls.append(r); ndcgs.append(n)
    return float(np.mean(recalls)), float(np.mean(ndcgs))
```

### 7.4 Baseline 0 and 1: random and popularity

```python
rng = np.random.default_rng(0)

def random_scores(ui):
    return rng.random(n_items)

item_pop = np.asarray(train_mat.sum(axis=0)).ravel()   # column sums = popularity
def popularity_scores(ui):
    return item_pop                                    # same for every user

print("random   :", evaluate(random_scores))
print("popularity:", evaluate(popularity_scores))
```

Popularity is the baseline every personalized model must beat, and it is shockingly strong because engagement is heavy-tailed: a handful of items absorb most interactions, so "show the popular thing" is right surprisingly often. It is also the purest expression of popularity bias — a non-personalized recommender that, if shipped and looped, would collapse the catalog. It is the bar, not the goal.

### 7.5 Baseline 2: item-item collaborative filtering

Item-item CF (Sarwar et al. 2001; the engine behind Amazon's classic "customers who bought this also bought," Linden et al. 2003) computes item-item similarity from co-interaction and scores a user by aggregating similarities to the items they already liked. Cosine similarity on the (sparse) item columns, then a sparse matrix product, gives personalized scores cheaply.

```python
from sklearn.preprocessing import normalize

# Item-item cosine similarity: normalize item columns, then Gram matrix.
item_user = train_mat.T.tocsr()                 # item x user
item_user_norm = normalize(item_user, axis=1)   # L2-normalize each item row
sim = item_user_norm @ item_user_norm.T         # item x item cosine (sparse-ish)

def itemcf_scores(ui):
    user_row = train_mat[ui]                     # 1 x n_items of liked items
    return np.asarray((user_row @ sim).todense()).ravel()

print("item-item CF:", evaluate(itemcf_scores))
```

### 7.6 Baseline 3: matrix factorization with the `implicit` library

Matrix factorization (Koren, Bell, Volinsky 2009) learns a $d$-dimensional vector per user and per item so that the dot product approximates preference. For implicit data, Alternating Least Squares with confidence weighting (the Hu-Koren-Volinsky formulation) is the standard, and the `implicit` library ships a fast, multithreaded implementation. BPR (the pairwise loss of section 4) is also one call away.

```python
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

# implicit expects an item-by-user matrix for fit (recent versions: user-item).
als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20)
als.fit(train_mat)                              # confidence-weighted ALS

user_factors = als.user_factors                 # (n_users, 64)
item_factors = als.item_factors                 # (n_items, 64)

def mf_scores(ui):
    return item_factors @ user_factors[ui]      # dot product over all items

print("MF / ALS:", evaluate(mf_scores))

# Pairwise alternative, same interface:
bpr = BayesianPersonalizedRanking(factors=64, iterations=100)
bpr.fit(train_mat)
```

The point of laying all four baselines side by side is not the absolute numbers — it is the *gradient* of effort versus payoff. Random is the floor. Popularity is a strong, dangerous non-personalized bar. Item-item CF adds personalization with no learning, just linear algebra. MF learns latent structure and is the first model that generalizes across the sparse ID space. Every advanced model in this series — two-tower retrieval, deep rankers, sequential models, LLM4Rec — is, in spirit, a more expressive version of "learn a user vector and an item vector and score their interaction."

### 7.7 The same idea in PyTorch: a tiny BPR matrix factorization

To make the bridge to the deep-learning tracks concrete, here is matrix factorization written by hand in PyTorch with the BPR loss of section 4. It is exactly an embedding lookup for a user, a positive item, and a negative item, scored by dot product, trained to rank the positive above the negative. Every two-tower retrieval model in the series is this with deeper towers.

```python
import torch
import torch.nn as nn

class BPRMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users, items):
        u = self.user_emb(users)                 # (B, dim)
        v = self.item_emb(items)                 # (B, dim)
        b = self.item_bias(items).squeeze(-1)    # (B,)
        return (u * v).sum(dim=-1) + b           # dot product + item bias

    def bpr_loss(self, users, pos_items, neg_items, weight_decay=1e-5):
        s_pos = self.score(users, pos_items)
        s_neg = self.score(users, neg_items)
        # log sigmoid(s_pos - s_neg): push the positive above the negative
        loss = -torch.nn.functional.logsigmoid(s_pos - s_neg).mean()
        reg = weight_decay * (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        )
        return loss + reg
```

The negative sampler is the unglamorous component that decides whether this model works. A naive uniform sampler draws an item the user has not interacted with; it is fast but produces "easy" negatives the model already ranks correctly, so the gradient (recall the BPR weight) is near zero and training stalls. The training loop below uses uniform negatives for clarity; the loss track upgrades it to popularity-corrected and hard negatives.

```python
def sample_negatives(user_pos_sets, users, n_items, rng):
    """For each user in the batch, draw one item they have not interacted with."""
    negs = np.empty(len(users), dtype=np.int64)
    for k, u in enumerate(users):
        j = rng.integers(n_items)
        while j in user_pos_sets[u]:             # reject seen items
            j = rng.integers(n_items)
        negs[k] = j
    return negs

def train_bpr(model, pos_pairs, user_pos_sets, n_items,
              epochs=30, batch=4096, lr=0.05):
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    pos_pairs = np.asarray(pos_pairs)            # (n_obs, 2): [user, pos_item]
    for ep in range(epochs):
        rng.shuffle(pos_pairs)
        total = 0.0
        for start in range(0, len(pos_pairs), batch):
            chunk = pos_pairs[start:start + batch]
            users = torch.as_tensor(chunk[:, 0])
            pos = torch.as_tensor(chunk[:, 1])
            neg = torch.as_tensor(
                sample_negatives(user_pos_sets, chunk[:, 0], n_items, rng)
            )
            opt.zero_grad()
            loss = model.bpr_loss(users, pos, neg)
            loss.backward()
            opt.step()
            total += loss.item() * len(chunk)
        print(f"epoch {ep:02d}  bpr loss {total / len(pos_pairs):.4f}")
```

Serving this model at retrieval scale is where the funnel comes back: you precompute every item embedding once, build an ANN index over them, and at request time query with the user embedding. A few lines of `faiss` turn the dot-product score into a millisecond lookup.

```python
import faiss

item_vectors = model.item_emb.weight.detach().cpu().numpy()   # (n_items, dim)
dim = item_vectors.shape[1]

# Exact MIPS via inner-product flat index (fine up to ~1M items).
index = faiss.IndexFlatIP(dim)
index.add(item_vectors)

# At scale, swap to IVF for the recall-latency curve of section 4.4:
# quantizer = faiss.IndexFlatIP(dim)
# index = faiss.IndexIVFFlat(quantizer, dim, nlist=4096,
#                            faiss.METRIC_INNER_PRODUCT)
# index.train(item_vectors); index.add(item_vectors)
# index.nprobe = 32           # the latency-vs-recall dial

def retrieve(user_id, k=500):
    q = model.user_emb.weight[user_id].detach().cpu().numpy().reshape(1, -1)
    scores, ids = index.search(q, k)             # top-k items in ~ms
    return ids[0], scores[0]
```

That is the whole funnel in miniature: train a scoring model with a ranking loss, precompute item vectors, index them for fast MIPS, and serve the top of the list. Everything advanced in the series — two-tower towers with content features, deep rankers on top of the retrieved candidates, sequential and graph encoders, LLM-based scorers — swaps in richer pieces without changing this skeleton.

## 8. Results: representative numbers on MovieLens

Now the measured payoff. The table below gives **representative** Recall@10 and NDCG@10 on MovieLens-1M with a temporal leave-last-N-out split and full (un-sampled) metrics over the catalog. Exact numbers shift with preprocessing (rating threshold, holdout size, minimum interactions per user), so treat these as literature-consistent order-of-magnitude figures, not a leaderboard. They are here to show the *shape* of the result, which is the durable lesson.

| Model | Personalized? | Learns? | Recall@10 | NDCG@10 | Relative cost |
| --- | --- | --- | --- | --- | --- |
| Random | no | no | ~0.005 | ~0.006 | trivial |
| Popularity | no | no | ~0.13 | ~0.16 | trivial |
| Item-item CF | yes | no (just similarity) | ~0.22 | ~0.26 | one Gram matrix |
| MF / ALS (d=64) | yes | yes | ~0.27 | ~0.31 | minutes on CPU |
| MF / BPR (d=64) | yes | yes | ~0.28 | ~0.33 | minutes on CPU |

Three readings of this table set up the whole series. First, **personalization roughly doubles top-K hit rate** over the non-personalized popularity bar — that jump from ~0.13 to ~0.22-0.28 Recall@10 is the value of modeling *who* the user is. Second, **learning beats memorizing**: MF edges out item-item CF by generalizing across the sparse matrix rather than relying on direct co-occurrence, and the gap widens as data gets sparser. Third, **BPR's pairwise loss tends to nudge NDCG above ALS's pointwise fit** at the same dimensionality — exactly the section-4 prediction that optimizing order, not absolute score, helps top-K. Hold onto that last one; it recurs at every level of the stack.

![Before and after bar comparison showing non-personalized random and popularity baselines versus learned item-item collaborative filtering and matrix factorization on Recall at ten and NDCG at ten](/imgs/blogs/what-is-a-recommender-system-7.png)

#### Worked example: computing Recall@10 and NDCG@10 by hand

Take one user whose held-out (test) set is three movies, with internal item ids $\{42, 7, 113\}$, all equally relevant ($\text{rel}=1$). The MF model's top-10 for this user, in order, is

$$
[\,7,\ 200,\ 42,\ 991,\ 16,\ 113,\ 5,\ 88,\ 301,\ 9\,].
$$

The hit pattern (1 where the item is in the test set) is $[1, 0, 1, 0, 0, 1, 0, 0, 0, 0]$ at positions $1, 3, 6$. **Recall@10** is the fraction of the user's relevant items retrieved: all three are in the top 10, so $\text{Recall@}10 = 3/3 = 1.0$. **DCG@10** sums $1/\log_2(p+1)$ over hits: position 1 gives $1/\log_2 2 = 1.000$, position 3 gives $1/\log_2 4 = 0.500$, position 6 gives $1/\log_2 7 = 0.356$, totaling $1.856$. The **ideal DCG** puts all three hits at positions 1, 2, 3: $1/\log_2 2 + 1/\log_2 3 + 1/\log_2 4 = 1.000 + 0.631 + 0.500 = 2.131$. So $\text{NDCG@}10 = 1.856 / 2.131 = 0.871$. Notice that recall is a perfect 1.0 but NDCG is 0.87, because two of the three relevant items sit below the top — **NDCG penalizes burying relevant items even when recall is maxed out.** That sensitivity to *where* in the list things land is exactly why NDCG, not recall alone, is the headline ranking metric.

## 9. A problem-solving walkthrough, then stress tests

Numbers and code are not enough; the series is also about *judgment* — how a principal engineer reasons from a problem to a decision and then attacks that decision to find where it breaks. Let us do one full pass on the running MovieLens example, then stress-test it the way you would in a design review.

**The problem.** You shipped the MF baseline. Offline NDCG@10 went from 0.16 (popularity) to 0.31 (MF). The team is happy. Then the A/B test reads *flat* engagement, and a week later the home feed is visibly dominated by a handful of blockbuster titles. What happened, and what do you do?

**Step 1: separate the metric from the goal.** NDCG@10 rose because, on the *logged* test set, MF orders the held-out items well. But the logged test set is the equilibrium of the *old* policy — it contains mostly interactions with items the old system already showed, which skew popular. A model that scores popular items highly will score well on that test set *and* online look like a popularity model. The offline win was partly an artifact of evaluating on biased data. This is the offline-online gap of section 6, and the feedback loop of section 5, made operational.

**Step 2: check the negatives.** Recall the sampled-softmax math of section 4.3. If retrieval (or here, the implicit-feedback negatives) is drawn from the popularity distribution without the $\log Q$ correction, the model is *trained* to over-score popular items. The fix is mechanical: correct the sampling distribution, or mine harder negatives so the model must distinguish a relevant niche item from a popular-but-irrelevant one. This usually moves online tail engagement even when it barely moves offline NDCG — which is exactly the kind of disagreement that proves you cannot trust offline alone.

**Step 3: add a re-ranking diversity pass.** Even with a fixed model, the section-3 re-ranking stage can break the visible monoculture by penalizing redundancy and capping per-creator or per-genre slots. This is cheap, ships fast, and is often the first thing that moves the felt quality and the long-term retention number that the loop was eroding.

**Step 4: decide, and write down the cost.** The decision is: ship the diversity re-ranker now (cheap, fast win), correct the negative sampling next (moves tail recall), and instrument an exploration slot so the system collects feedback on items it would never have shown (breaks the MNAR loop at the source). Each is a cost — the diversity pass trades a hair of offline NDCG for online diversity, exploration trades a slice of short-term CTR for unbiased data. The principal-engineer move is to name those costs explicitly rather than pretend the change is free.

Now stress-test the conclusion. A robust design survives the "what if" interrogation:

- **What if you have only implicit feedback?** You already do — that is the default. The consequence is that every "0" in the matrix is ambiguous, so absolute calibration is unreliable and you must lean on ranking losses and confidence weighting, not rating regression.
- **What at 100M or 1B items?** The flat `faiss` index of section 7.7 stops fitting in memory and stops meeting latency. You move to IVF-PQ (compress vectors, probe a fraction of cells) and accept the recall-latency trade of section 4.4. The MF skeleton survives; the serving index changes.
- **What if the negatives are mostly false negatives?** With a huge sparse catalog, a "negative" you sampled may be an item the user *would* love but never saw. Uniform sampling makes this rare per pair but systematic in aggregate; it caps how high recall can go. The fix is debiasing (IPS) and exposure-aware sampling — a whole post.
- **What if offline rises but online is flat?** The case above. Trust the A/B test, suspect the loop, check the negatives, and add exploration. Do not "fix" it by tuning offline harder; that optimizes the proxy.
- **What if the feature is computed differently offline and online?** Train-serve skew silently halves precision. The user's "recent 20 items" computed in a batch job at midnight is not the same as the live feature at request time. This class of bug does not show up in offline metrics at all — it shows up only online, as a model that looks great in the notebook and mediocre in production. The defenses (a shared feature store, logging the features used at serving time, and offline-online feature parity tests) get a dedicated troubleshooting post.

This is the texture of real recsys work: the model is the easy part; the loop, the negatives, the evaluation, and the train-serve plumbing are where the engineering — and the series — actually lives.

#### Worked example: when offline and online disagree

Two candidate rankers, A and B. Offline on the logged temporal split, A scores NDCG@10 = 0.330 and B scores 0.318 — A wins by a clear 0.012. You ship A in an A/B test against the current production model. After two weeks: A is +0.4% CTR (inside the noise band, not significant) and *flat* on D7 retention; the B variant, run in parallel, is +1.9% CTR and +0.6% D7 retention, both significant. B *lost* offline and *won* online. Why? Offline, A wins because it ranks the logged-popular items slightly better, and the logged test set is popular-heavy. Online, B's harder negatives and diversity tilt surface relevant niche items the logs never recorded, lifting genuine engagement and the retention that the popularity monoculture was quietly eroding. The lesson in one line: **a 0.012 offline NDCG edge is worth less than a significant online retention lift, and the two can point in opposite directions.** Gate on offline to avoid shipping regressions; decide on online.

To make the model-choice trade-offs concrete, here is a compact comparison of the families this post touched, on the axes that actually drive a decision.

| Approach | Cold start | Personalized | Scales to 1B items | Top-K quality | When to reach for it |
| --- | --- | --- | --- | --- | --- |
| Popularity | great (no data needed) | no | trivially | weak | day-one baseline, fallback rail |
| Content-based | great (uses features) | weak | yes | moderate | cold-start items, niche catalogs |
| Item-item CF | poor (needs co-views) | yes | with offline precompute | good | mature catalog, "also bought" rails |
| Matrix factorization | poor | yes | yes (with ANN) | strong | the warm-data workhorse baseline |
| Two-tower + ANN | moderate (hybrid features) | yes | yes (the standard) | strong | production retrieval at scale |
| Deep ranker | n/a (ranks candidates) | yes | n/a (small candidate set) | strongest | the ranking stage of the funnel |

Read it as a decision aid, not a leaderboard: pick the lightest row that hits your target, and only climb when a measured Pareto improvement pays for the added cost.

## 10. The taxonomy: how to think about the model zoo

Before the series dives into specific models, it helps to have a map of the territory so each post slots into place. Recommenders divide first by *what signal they trust*.

![Tree diagram of a recommender taxonomy branching into collaborative, content-based, and hybrid deep families, each splitting into classic and deep methods such as item-item collaborative filtering, matrix factorization, two-tower retrieval, and deep rankers](/imgs/blogs/what-is-a-recommender-system-3.png)

**Collaborative filtering** trusts behavior: it recommends based on patterns of co-interaction across users, with no idea what an item *is*. Item-item CF and matrix factorization both live here. CF is powerful precisely because it needs no content features — it discovers that two movies are similar because the same people watch both, even if their genres differ. Its Achilles' heel is **cold start**: a brand-new item with zero interactions has no behavioral signal, so CF cannot place it.

**Content-based** methods trust features: they represent each item by its attributes (text, genre, image embeddings, price) and recommend items similar to what the user engaged with. This handles cold-start items gracefully (a new movie still has a genre and a description) but cannot capture the "people like you also liked" magic that makes CF feel personal, and it tends to over-specialize (recommend only more of the same).

**Hybrid and deep** models fuse both. A two-tower retrieval model can put behavioral IDs *and* content features into each tower. A deep ranker ingests user history, item content, and context together. This is where modern production recsys lives, and it is the bulk of the modeling tracks: two-tower and ANN for deep retrieval; DLRM, DIN, DCN, and friends for deep ranking; sequential models (SASRec, BERT4Rec) for session and history; graph neural networks (PinSage, LightGCN) where the interaction graph carries signal; and, most recently, LLM-based recommenders that treat recommendation as a generation or in-context task.

The serving stack below is the physical form this taxonomy takes in production: raw events become features, features feed retrieval, retrieval feeds ranking, ranking feeds re-ranking, and a response leaves the box — each layer a place where a specific model family and a specific class of bug live.

![Layered stack diagram of a recommendation serving pipeline from raw events to features to retrieval to ranking to re-ranking to the served response with a latency budget annotation](/imgs/blogs/what-is-a-recommender-system-6.png)

## 11. Case studies: how the giants actually do it

The funnel is not a textbook idealization; it is how shipped systems are built. Five classic references anchor the series.

**YouTube's two-stage deep recommender (Covington, Adams, Sargin, 2016).** The canonical modern paper. YouTube splits recommendation into exactly the funnel of section 3: a deep candidate-generation network reduces the corpus of millions of videos to hundreds, then a deep ranking network scores those hundreds with a far richer feature set (including the candidate's own features and fine-grained user-history signals). The candidate generator is trained as an extreme multiclass classification with sampled softmax over the video corpus — the ancestor of today's two-tower retrieval — and the ranker predicts expected watch time, not click, because clickbait optimizes clicks while degrading satisfaction. The paper is the clearest published statement of *why* one model cannot do it all and why each stage gets its own objective.

**Amazon item-to-item collaborative filtering (Linden, Smith, York, 2003).** Before deep learning, Amazon's recommendations ran on item-item CF — the exact algorithm of section 7.5 — for a sharp engineering reason: it moves the expensive computation (item-item similarities) offline and makes online recommendation a cheap lookup that scales to tens of millions of users and items with low latency. The "customers who bought this also bought" rail is item-item CF, and the paper's central insight — *precompute the heavy similarity offline, serve a cheap lookup online* — is the same instinct that makes two-tower retrieval precompute item embeddings into an ANN index today.

**Netflix.** Netflix's public writing (the Netflix Prize era and the later "Netflix Recommender System" overview by Gomez-Uribe and Hunt, 2015) tells two stories the series leans on. First, the Netflix Prize centered on rating-prediction RMSE — and Netflix later said much of the prize-winning machinery never shipped, partly because RMSE was the wrong objective for what the product needed (ranking, page construction, and the now-famous personalized artwork). That is the section-4 lesson, lived at scale: optimizing the wrong metric wins a contest and loses the product. Second, Netflix frames recommendation as the value driver, with the oft-cited claim that a large majority of viewing originates from recommendations — the business case of section 2.

**Matrix factorization for recommender systems (Koren, Bell, Volinsky, 2009).** The IEEE Computer article that crystallized MF as the workhorse after the Netflix Prize. It lays out latent-factor models, the bias terms (global, user, item), confidence weighting for implicit feedback, and temporal dynamics — the foundation that every embedding-based recommender, two-tower models included, builds on. When this series gets to two-tower retrieval, it is MF with deep towers and an ANN index; the lineage runs straight back to this paper.

**Pinterest's PinSage (Ying et al., 2018).** The case study that scaled graph neural networks to a real recommender — billions of pins and boards. PinSage runs graph convolutions over the pin-board interaction graph to learn item embeddings that fold in both content and graph structure, then serves them through the same retrieval-then-rank funnel. Its engineering contributions are about *making the graph tractable*: importance-based neighbor sampling instead of full neighborhoods, producer-consumer minibatching, and MapReduce inference over the whole graph. It is the clearest demonstration that when the interaction graph carries signal beyond a flat user-item matrix, a GNN can beat plain MF — and also that the funnel and the precompute-offline instinct survive the jump to graph models. We reproduce its ideas in the graph track, with the caveat that a GNN is worth it only when the graph structure genuinely adds information a two-tower cannot capture.

The shared theme across all five — and the reason they anchor the series — is that none of them is primarily a story about a clever loss function. Each is a story about *systems discipline*: stage the work, optimize the right objective per stage, precompute the expensive part offline, and serve a cheap lookup online under a hard latency budget. The model is the part that fits in a paragraph; the funnel, the index, the logging, and the evaluation are the part that fits in a career.

Common thread across all four: **stage the work, optimize the right objective per stage, and precompute the heavy part offline.** That is the engineering philosophy the funnel encodes.

## 12. When recommenders help, and when they don't

Recommendation is a cost — engineering, infra, and ongoing operational vigilance against the feedback loop. It is worth it when:

- **The catalog is large and browsing is infeasible.** Tens of thousands to billions of items, where search captures only known intent and most value is in latent intent.
- **Engagement is repeated and logged.** You need behavioral data to learn from; a one-shot transaction with no return visit gives the loop nothing to feed on.
- **The surface is a default feed or rail.** If the recommended surface is where users land, the recommender is the product and the investment compounds.

It is *not* worth a deep recommender when:

- **The catalog is small.** A few dozen items? A hand-curated list or a popularity sort with light personalization beats a trained model and is debuggable. Don't build a two-tower retrieval stack to choose among 40 products.
- **You have no interaction data yet (pure cold start).** At launch, with no logs, content-based heuristics and editorial curation outperform a model with nothing to learn from. The collaborative magic needs behavior first.
- **A simpler stage already hits target.** If a single ranking model over a small candidate set meets your latency and quality goals, you do not need a separate retrieval tower. Don't add a stage, a GNN, or an LLM because it is fashionable; add it because a measured Pareto improvement justifies its cost. The series will repeatedly say *don't reach for the heavy thing if the light thing hits target* — that is the principal-engineer instinct the whole series is trying to build.

A blunt rule for the offline-online gap: **never ship a ranker that was only validated on offline AUC or NDCG.** Offline metrics gate cheaply; online A/B tests decide. A model that improves offline and is flat or negative online is common, not exotic, and the loop is why.

## 13. The map of the series

Here is the territory ahead, organized as tracks. This intro is the frame; the capstone is the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) that ties it all back together. Each track is pinned to where it sits in the funnel and what skill it builds.

![Matrix mapping the series tracks from foundations through classic models, deep retrieval, deep ranking, loss and negatives, evaluation and bias, to LLM for recommendation and case studies against funnel stage and what you learn](/imgs/blogs/what-is-a-recommender-system-8.png)

- **Foundations** (this track): the funnel, the feedback loop, the offline-online gap, implicit feedback, and the metrics. The conceptual spine. See [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) for the deep dive on staging.
- **Classic models**: [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles), [matrix factorization, the workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), factorization machines, and the neighborhood methods. Where most of your intuition forms.
- **Deep retrieval**: two-tower models, in-batch negatives, sampled softmax with the $\log Q$ correction, and ANN serving with `faiss` / `hnswlib` / ScaNN — turning a billion items into a millisecond lookup.
- **Deep ranking**: DLRM, Wide and Deep, DeepFM, DCN, DIN/DIEN, multi-task and multi-objective ranking, and calibration — the expensive, feature-rich heart of the funnel.
- **Loss and negatives**: pointwise vs pairwise (BPR) vs listwise, InfoNCE / contrastive losses, hard-negative mining, and why the choice of negatives moves results more than the architecture.
- **Sequential and graph models**: SASRec, BERT4Rec, and session-based recommendation; LightGCN and PinSage where the interaction graph carries signal.
- **Finetuning and LLM4Rec**: treating recommendation as a language task, LoRA / PEFT finetuning of LLMs for ranking and generation, and generative retrieval (semantic IDs, TIGER). See the [large language model track](/blog/machine-learning/large-language-model/) for the transformer and finetuning background these posts assume.
- **Evaluation**: Recall@K, NDCG@K, MAP, MRR, AUC, calibration, the sampled-metrics-are-inconsistent result, and how to make offline evaluation a less-biased predictor of online.
- **Bias and troubleshooting**: position bias, popularity bias, MNAR data, IPS and off-policy estimation, the feedback loop as a fixed point, and the war stories — train-serve skew, embedding OOM, the offline win that flopped online.
- **Advanced and systems**: serving architecture, feature stores, embedding-table sharding, exploration, fairness, and cost.
- **Case studies**: end-to-end walkthroughs of shipped systems and reproductions of headline results.

If you read only two posts, read this one and the capstone. If you read the whole thing, you will be able to design, build, evaluate, and debug a production recommender — and, just as important, know when *not* to build one.

## 14. Key takeaways

- **Recommendation is not classification with a big output layer.** It breaks four assumptions: positive-only implicit feedback, a sparse high-cardinality ID space, a top-$K$ ranking objective, and missing-not-at-random data shaped by the system itself.
- **The funnel is forced by arithmetic, not taste.** You cannot score a billion items with a heavy model in 100 ms, so you stage retrieval (cheap, high recall) → ranking (expensive, high precision) → re-ranking (list-level rules). Each stage has its own model, budget, and objective.
- **Rank, don't regress.** Lowering RMSE provably does not improve — and can harm — top-$K$ ranking, because squared error is blind to where errors fall in the order. Pairwise losses like BPR win because their gradient concentrates on misranked pairs.
- **NDCG is the headline metric.** It weights position, so it penalizes burying relevant items even when recall is maxed out. Compute it with full (un-sampled) metrics on a temporal split.
- **The feedback loop is the engine and the bias.** Serve → log → train → deploy lets the system adapt, and it makes popularity bias a self-reinforcing fixed point and your logs a missing-not-at-random sample.
- **Offline metrics are a cheap, biased pre-filter; online A/B tests are the truth.** Never ship a ranker validated only on offline AUC or NDCG; the offline-online gap is a structural consequence of the loop.
- **Personalization roughly doubles top-$K$ hit rate over popularity, and learning beats memorizing.** But popularity is a strong, dangerous bar that, looped, collapses the catalog.
- **Precompute the heavy part offline, serve a cheap lookup online** — the instinct behind both Amazon's item-item CF and modern two-tower ANN retrieval.
- **Add complexity only for a measured Pareto win.** Don't build a deep recommender for 40 items, don't add a GNN if a two-tower hits target, and don't ship the fashionable thing because it is fashionable.

## 15. Further reading

- Resnick, P. and Varian, H. R. (1997). *Recommender Systems.* Communications of the ACM. The paper that named and framed collaborative filtering.
- Sarwar, B., Karypis, G., Konstan, J., Riedl, J. (2001). *Item-based collaborative filtering recommendation algorithms.* WWW. The item-item CF algorithm of section 7.5.
- Linden, G., Smith, B., York, J. (2003). *Amazon.com recommendations: Item-to-item collaborative filtering.* IEEE Internet Computing. Why precompute offline, serve cheap online.
- Hu, Y., Koren, Y., Volinsky, C. (2008). *Collaborative filtering for implicit feedback datasets.* ICDM. Confidence-weighted ALS for implicit data.
- Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from implicit feedback.* UAI. The pairwise loss of section 4.
- Koren, Y., Bell, R., Volinsky, C. (2009). *Matrix factorization techniques for recommender systems.* IEEE Computer. The workhorse, formalized.
- Covington, P., Adams, J., Sargin, E. (2016). *Deep neural networks for YouTube recommendations.* RecSys. The two-stage funnel, in production.
- Krichene, W. and Rendle, S. (2020). *On sampled metrics for item recommendation.* KDD. Why sampled Recall@K / NDCG@K can be inconsistent — compute full metrics.
- Official docs: `implicit` (benfred.github.io/implicit), `faiss` (github.com/facebookresearch/faiss), RecBole (recbole.io), `peft` (huggingface.co/docs/peft).
- Within series: [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) · [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles) · [matrix factorization, the workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) · [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

The next post takes the funnel apart stage by stage — retrieval, ranking, re-ranking — and shows how to budget latency, choose an ANN index, and keep the stages from fighting each other. From there, the classic models, then deep, then loss, then evaluation, then the bias and troubleshooting that pages you at 3 a.m. From click to production, one stage at a time.

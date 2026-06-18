---
title: "Framing the Problem: Rating, Ranking, or Retrieval?"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The most consequential early decision in any recommender is how you frame the task, because that single choice fixes your model head, your loss, your metric, your negatives, and whether you can ever serve a billion items."
tags:
  [
    "recommendation-systems",
    "recsys",
    "learning-to-rank",
    "retrieval",
    "calibration",
    "matrix-factorization",
    "ndcg",
    "machine-learning",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/framing-the-problem-rating-ranking-retrieval-1.png"
---

A team I worked with spent six weeks shrinking the validation RMSE of their movie recommender from 0.91 to 0.86. They were proud of it. The model predicted the star rating a user would give a film, and by squared error it was clearly better. Then they shipped it, watched the online metrics, and nothing moved — click-through on the recommended row was flat, watch time per user was flat, the only thing that changed was the dashboard everyone had been staring at for six weeks. The post-mortem found the bug in one sentence: the metric they optimized was not the metric the product cared about. They had framed recommendation as *rating prediction*, a regression problem, and then judged the product on whether the right movies appeared at the top of a list — a *ranking* problem. Those are different objectives, and a model can win one while ignoring the other entirely.

This post is about the single most consequential decision you make in a recommender, and you make it before you write a line of model code: **how you frame the task.** There are four common framings of the very same interaction data — predict a rating (regression), predict a click (classification), order the items (learning-to-rank), or embed users and items and retrieve the top-K by similarity (retrieval). Each one is a different objective over the same logs, and the framing you pick does not just choose your model. It fixes your loss function, the metric you are allowed to report, what counts as a negative example, the shape of your data pipeline, and — this is the part people miss until it is too late — whether your design can ever scale to a billion items or whether it dies at a million. The figure below is the whole post in one table: four framings down the side, the loss, metric, negatives, and scaling story across the top.

![Matrix comparing the four recommendation framings across loss function, evaluation metric, negative definition, and whether each scales to a billion items](/imgs/blogs/framing-the-problem-rating-ranking-retrieval-1.png)

This is the third post in the series **Recommendation Systems: From Click to Production**, and it sits right after the two that set the stage: [what a recommender system actually is](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) and [the retrieval-ranking-reranking funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) that turns a billion items into the twenty a user sees. Everything here connects back to that funnel, because the funnel is, at bottom, two different framings stacked: a retrieval framing for stage one (find a thousand candidates from a billion, fast) and a ranking-or-classification framing for stage two (order or score those thousand precisely). By the end of this post you will be able to look at a recommendation problem and name its correct framing in about thirty seconds, justify the choice with the math, and avoid the six-week RMSE mistake. We will keep one dataset in view the whole way — MovieLens — and train the *same data* under all four framings, then evaluate every model on its own natural metric and on a common top-K metric, so you can see with real numbers why the framings disagree.

## 1. The four framings, stated plainly

Strip a recommender down to its data and you have a log of interactions: a user $u$, an item $i$, sometimes a value $r$ (a star rating, a watch fraction, a binary click), and a timestamp. That is the raw material. The framing is how you turn that log into a supervised learning problem — what the input is, what the label is, and what the loss measures. Here are the four, each with its one-sentence soul.

**Framing 1 — Rating prediction (regression).** Input: a user-item pair. Label: the numeric value the user assigned, like a 1-to-5 star rating. Loss: squared error, so the model is trained to make its predicted score $\hat{r}_{ui}$ close to the true $r_{ui}$. Metric: root-mean-square error (RMSE) or mean absolute error (MAE). This is the framing the Netflix Prize made famous, and it is the one a classically-trained ML engineer reaches for by reflex because it looks exactly like the regression problems in every textbook. Its hidden assumption: that getting the absolute score right is what matters.

**Framing 2 — Click / interaction prediction (classification).** Input: a user-item pair, usually with context. Label: a binary, did the user click / play / buy this item (1) or not (0). Loss: binary cross-entropy (logloss). Metric: AUC (area under the ROC curve, the probability the model scores a random positive above a random negative) and logloss itself, plus calibration. This is the framing of the entire online-advertising industry, where the predicted probability is called pCTR (predicted click-through rate) and is *multiplied by a bid*, so the number has to be a real probability, not just a relative score. Its hidden assumption: that you can define negatives, and that a calibrated probability is worth the trouble.

**Framing 3 — Learning-to-rank (LTR).** Input: a user and a *set* of items. Label: the relative order or graded relevance among them. Loss: a pairwise loss like Bayesian Personalized Ranking (BPR) that pushes a positive item's score above a negative item's score, or a listwise loss that optimizes the whole ordering at once. Metric: ranking metrics — NDCG (normalized discounted cumulative gain), MAP (mean average precision), MRR (mean reciprocal rank). Its central insight: the user only ever sees the top of the list, so the objective should be the *order*, not the absolute score of any single item.

**Framing 4 — Retrieval / metric learning.** Input: a user and the catalog. The model learns an embedding for the user and an embedding for every item such that the dot product (or cosine) of user and relevant item is high. Loss: a contrastive or sampled-softmax loss that pulls the user toward positive items and pushes away from sampled negatives. Metric: Recall@K and HitRate@K — did the true next item land in the top-K returned by nearest-neighbor search. Its superpower, and the only reason it exists: because the score is a *factorized* dot product $u \cdot i$, you can precompute every item vector, build an approximate-nearest-neighbor (ANN) index once, and at request time retrieve the top-K in *sublinear* time. This is the only framing that survives a billion-item catalog.

Notice what just happened. We did not change the data. The same user-item log can be relabeled four ways: keep the rating (regression), binarize it into clicked/not (classification), pair a clicked item against an unclicked one (ranking), or treat clicked items as positives to pull toward and everything else as negatives to push away (retrieval). The framing is a *modeling choice*, not a property of the data. The figure below makes that concrete — one log on top, four heads underneath, each reading a different label off the same rows.

![Stack diagram showing one interaction log feeding a relabel step that splits into four objective heads for regression, click prediction, pairwise ranking, and two-tower retrieval](/imgs/blogs/framing-the-problem-rating-ranking-retrieval-5.png)

This is liberating and dangerous in equal measure. Liberating, because you are not stuck — if you have implicit feedback, you can still rank or retrieve, you just cannot do honest rating regression. Dangerous, because nothing in the data stops you from choosing the *wrong* framing and then optimizing it beautifully. The RMSE team optimized framing 1 and got judged on framing 3. The rest of this post is about not doing that.

## 2. The science: why RMSE-optimal is not ranking-optimal

This is the heart of the matter, so let us make it rigorous rather than hand-wave it. The claim is that a model can have *strictly lower* RMSE than another model and yet rank items *worse*. If that is true, then "minimize RMSE" and "rank well" are genuinely different objectives, and choosing the regression framing for a top-K product is a category error, not a tuning problem.

Set up the smallest possible counterexample. A single user has two items, A and B. The user's true ratings are $r_A = 5$ and $r_B = 4$, so the correct ranking puts A first. We have two models.

Model M1 predicts $\hat{r}_A = 4.0$ and $\hat{r}_B = 4.1$.
Model M2 predicts $\hat{r}_A = 3.0$ and $\hat{r}_B = 1.0$.

Compute the squared errors. For M1: $(5 - 4.0)^2 + (4 - 4.1)^2 = 1.00 + 0.01 = 1.01$, so RMSE is $\sqrt{1.01/2} \approx 0.711$. For M2: $(5 - 3.0)^2 + (4 - 1.0)^2 = 4.00 + 9.00 = 13.00$, RMSE $\approx 2.550$. By RMSE, M1 is dramatically better — its error is a third of M2's.

Now look at the ranking. M1 predicts B above A ($4.1 > 4.0$): the order is **wrong**. M2 predicts A above B ($3.0 > 1.0$): the order is **right**. The model with the far lower RMSE puts the wrong item on top, and the model with the catastrophic RMSE gets the order exactly right. If the product shows the user the single top item, M1 — the RMSE winner — shows them the worse movie, and M2 — the RMSE loser — shows them the better one.

![Before and after panels contrasting an RMSE-optimal model that orders the top item wrong against a higher-error model that gets the order right](/imgs/blogs/framing-the-problem-rating-ranking-retrieval-3.png)

The reason this is possible, stated as a small proof, is that RMSE is a function of the *residuals* $\hat{r}_{ui} - r_{ui}$ and is completely blind to whether the residuals preserve order. Ranking quality is a function of the *sign of pairwise differences* $\hat{r}_{ui} - \hat{r}_{uj}$ relative to $r_{ui} - r_{uj}$, and is completely blind to the magnitude. These two functions measure different things. Formally, RMSE is invariant to any order-preserving transformation of the predictions only in the trivial identity case, while every strictly increasing transformation preserves the ranking but changes the RMSE arbitrarily. So the level sets of the two objectives are not nested: you can move within a fixed-RMSE shell and flip rankings, and you can move along a fixed-ranking curve and change RMSE without bound. Two objectives whose level sets cross like this cannot have the same optimizer except by coincidence.

There is a deeper, distributional reason too. RMSE weights every prediction error equally. But in a top-K product, an error on an item the user will never see costs nothing, and an error that swaps the first and second result costs a lot. RMSE has no notion of *position*; NDCG has a position discount $1/\log_2(\text{rank}+1)$ baked into its definition precisely to encode that the top of the list matters more. A regression loss spends its capacity getting the boring middle of the rating distribution right, because that is where most of the squared-error mass is, while the ranking objective spends its capacity on the head. They pull the model in different directions.

#### Worked example: the RMSE-good-but-rank-bad model, by hand

Let us scale the counterexample to something that feels like a real evaluation. Take one user with five candidate items whose true relevances (think graded: 3 = loved, 2 = liked, 1 = meh, 0 = ignored) are A=3, B=2, C=2, D=1, E=0. The ideal order is A, then B or C, then D, then E. Compute the ideal DCG at K=3 with the standard gain $2^{\text{rel}}-1$ and discount $1/\log_2(\text{rank}+1)$: ideal puts A(3), B(2), C(2) on top, giving $\frac{2^3-1}{\log_2 2} + \frac{2^2-1}{\log_2 3} + \frac{2^2-1}{\log_2 4} = \frac{7}{1} + \frac{3}{1.585} + \frac{3}{2} = 7 + 1.893 + 1.5 = 10.393$. That is IDCG@3.

Model M1 (the RMSE darling) predicts scores that are all close to the mean — 2.6, 2.7, 2.4, 2.5, 2.3 — so its RMSE against the true relevances is small, but its induced order is B, A, D, C, E. Its top-3 is B(2), A(3), D(1), giving $\frac{3}{1} + \frac{7}{1.585} + \frac{1}{2} = 3 + 4.416 + 0.5 = 7.916$, so NDCG@3 $= 7.916 / 10.393 = 0.762$. Model M2 (the rank specialist) predicts spread-out scores 9, 5, 4, 2, 0 whose RMSE against 3,2,2,1,0 is enormous, but whose order is exactly A, B, C, D, E. Its top-3 is A, B, C, achieving the ideal: NDCG@3 $= 1.0$. M1 has the better RMSE and the worse NDCG by a quarter. The arithmetic is the whole argument: a metric that ignores order rewards a model that ignores order.

The practical takeaway is not "regression is bad." Rating regression is the right framing when you genuinely need the absolute score — a "you'll probably rate this 4.6 stars" widget, a price estimate, a satisfaction prediction feeding a downstream optimizer. It is the wrong framing when the product is a top-K list, which is almost every recommender. The mistake the six-week team made was using a tool whose loss does not know what the product cares about.

### 2.1 Pointwise versus pairwise: where the gradient looks

The cleanest way to internalize why the ranking framing beats the regression framing for top-K is to compare what the *gradient* of each loss is paying attention to. A pointwise loss — MSE in regression, or even BCE applied independently per example — produces a gradient on each prediction that depends only on that one prediction's error: $\partial \text{MSE} / \partial \hat{r}_{ui} = 2(\hat{r}_{ui} - r_{ui})$. The gradient pulls each score toward its own target in isolation, with no term that compares two items for the same user. The model has no incentive to make item $i$ outscore item $j$; it only cares that each is near its own label. If two items have nearby labels (4 and 4.1 stars), the loss is essentially indifferent to which one ends up on top.

A pairwise loss is built from the comparison. BPR's loss for a triple $(u, i, j)$ — user, positive item, negative item — is $\mathcal{L} = -\ln \sigma(s_{ui} - s_{uj})$, where $s$ is the model's score and $\sigma$ is the logistic sigmoid. Differentiate with respect to the score difference $d = s_{ui} - s_{uj}$:

$$\frac{\partial \mathcal{L}}{\partial d} = -\frac{\sigma'(d)}{\sigma(d)} = -\big(1 - \sigma(d)\big) = -\sigma(-d) = -\sigma(s_{uj} - s_{ui}).$$

Read that gradient. Its magnitude is $\sigma(s_{uj} - s_{ui})$ — the model's estimated probability that the *negative* outscores the *positive*. When the model already ranks the pair correctly ($s_{ui} \gg s_{uj}$), that probability is near zero and the gradient vanishes: the model stops spending effort on pairs it already gets right. When the pair is *inverted* ($s_{uj} > s_{ui}$), the probability is near one and the gradient is at full strength: the model pours its learning into fixing the inversions, which are exactly the ranking errors. The gradient *sees the order*, because the loss is a function of the difference of two scores, and a difference is invariant to any shift of the absolute level. That single structural fact — loss-of-a-difference versus loss-of-a-value — is why pairwise BPR routinely beats pointwise regression on Recall@K and NDCG@K on the same data, as the results table in section 7 shows.

### 2.2 Listwise: optimizing the whole order at once

Pairwise is not the end of the ranking framing. A *listwise* loss looks at the entire ordered list for a user and optimizes a function of the whole permutation, rather than decomposing into independent pairs. The motivation is that pairwise losses treat every inversion as equally bad, but a metric like NDCG weights inversions near the top far more heavily than inversions deep in the list — swapping ranks 1 and 2 costs more than swapping ranks 99 and 100. Listwise methods such as LambdaRank and its tree-based incarnation LambdaMART introduce a per-pair weight equal to the *change in NDCG* that swapping the pair would cause, $|\Delta \text{NDCG}_{ij}|$, multiplying it into the pairwise gradient. The effect is a gradient that not only sees the order but weights each order-fix by how much it moves the metric you actually report. ListNet and ListMLE go further and define a probability distribution over permutations, optimizing the likelihood of the correct ordering directly. The listwise family is the most aligned of all with NDCG, at the cost of more expensive training (you need full lists per user, not just sampled pairs) — which is why it tends to appear in the stage-two ranker, where candidate lists are already short, rather than in retrieval. The point for *framing* is that "learning-to-rank" is itself a small family, ordered roughly pointwise → pairwise → listwise by how directly each targets the ranking metric, and you move up that ladder as the candidate set shrinks and you can afford richer supervision.

## 3. The science: why classification is calibrated but rank-suboptimal

The classification framing trains a model to predict $P(\text{click} \mid u, i)$ with cross-entropy. Cross-entropy is a *proper scoring rule*, which is a precise statistical statement: it is minimized, in expectation, exactly when the predicted probability equals the true conditional probability. That is the source of classification's superpower — at the optimum, the model's output *is* a calibrated probability, a number you can trust as a real chance, multiply by a dollar value, sum across events, and feed into a bidding system. No other framing gives you this for free. BPR's pairwise score has no probabilistic meaning; a two-tower dot product is an unnormalized similarity. Only the cross-entropy-trained classifier outputs something you can put on a probability axis and believe.

But calibration is not ranking. Consider what cross-entropy actually optimizes versus what a ranking metric rewards. Cross-entropy, $-\sum y \log p + (1-y)\log(1-p)$, penalizes confident wrong probabilities heavily and rewards getting the *probability* right on every example, positive and negative alike. It spends a great deal of effort getting the probability of obvious negatives close to zero — which is correct for calibration and almost useless for ranking, because those obvious negatives are nowhere near the top of the list and their exact probability does not change the order of the head. A model can be beautifully calibrated overall and still order the contested items at the top — the ones whose probabilities are all near each other — slightly wrong, which is exactly where NDCG is sensitive. This is why ad systems that optimize logloss often add a ranking-aware auxiliary loss or evaluate with AUC alongside logloss: AUC is a pure ranking metric (it equals the probability a random positive outscores a random negative), and it can diverge from logloss.

There is also a subtler trap unique to the classification framing: **what counts as a negative.** In a click model, a negative is an item that was shown and not clicked. But most items were never shown, so you do not observe their labels at all. If you treat every unshown item as a negative (label 0), you are making a missing-not-at-random assumption that is simply false, and you flood the training set with "negatives" that are really just unseen — the positive-unlabeled problem we will dwell on in section 5. If you only train on shown items, your model learns $P(\text{click} \mid \text{shown})$, not $P(\text{click})$, and the "shown" set was chosen by the previous recommender, so you have inherited its biases. The framing is calibrated *with respect to the distribution it was trained on*, and that distribution is the old system's impression log, not the world. Calibration is only as honest as your negatives.

### 3.1 What calibration actually means, and how to measure it

It is worth being precise about the word "calibrated," because it is thrown around loosely. A model is calibrated if, among all the predictions where it said "10% chance of click," about 10% actually clicked — and the same holds at every probability level. Formally, for a calibrated model $P(\,y = 1 \mid \hat{p} = p\,) = p$ for all $p$. The standard way to measure the gap is **expected calibration error (ECE)**: bin the predictions by their predicted probability (say into ten bins), and in each bin compare the average predicted probability to the observed click rate; ECE is the bin-size-weighted average of those absolute gaps. A model with ECE near zero can be trusted to multiply by money; a model with high ECE cannot, no matter how good its AUC.

Here is the crucial decoupling that the framing question turns on: **AUC and ECE are independent.** AUC measures only the order — it is unchanged by any monotonic transformation of the scores, because it only counts how often a positive outscores a negative. So you can take a perfectly-calibrated model and squash all its probabilities toward 0.5 (a monotonic transform), destroying calibration while leaving AUC *identical*. Conversely you can have a model with mediocre AUC that is perfectly calibrated. Ranking-trained models (BPR, two-tower) optimize something AUC-like and are free to be arbitrarily miscalibrated; classification-trained models optimize logloss, which a proper-scoring-rule argument ties to calibration. This is the rigorous version of "ranking gives order, classification gives probability." If you ever need both — good order *and* a trustworthy probability — the standard recipe is to train for ranking, then fit a cheap monotonic calibrator (isotonic regression or Platt scaling, both one line of `sklearn`) on a held-out set to map raw scores to probabilities after the fact. But that post-hoc step is fragile: it assumes the score distribution at serve time matches the calibration set, which drifts. Native calibration from the classification framing is sturdier when you genuinely need the probability.

### 3.2 When AUC and logloss disagree

A practical scenario sharpens the point. Suppose two CTR models, A and B, both used for ad ranking. Model A has AUC 0.82 and logloss 0.41; model B has AUC 0.80 and logloss 0.38. By AUC, A ranks better — it more often puts a clicker above a non-clicker. By logloss, B is better calibrated — its probabilities are closer to the truth. Which do you ship? It depends entirely on the framing question from the auction: if you only rank the ads and show the top, ship A. If you bid (probability times value), ship B, because A's better order does not help if its probability is wrong by 20% and you over- or under-bid on every impression. The two metrics genuinely disagree, and no amount of tuning collapses them into one number, because they measure different properties of the same scores. The framing — do I rank, or do I bid? — tells you which metric is the real objective and which is the distraction.

#### Worked example: why an ad system must use the classification framing

An ad exchange runs a second-price auction. To bid for an impression, your system needs an expected value: $\text{bid} = \text{pCTR} \times \text{value-per-click}$. Suppose the true click probability for a given user-ad pair is 2%, and the advertiser pays you \$0.50 per click. The correct expected value is $0.02 \times \$0.50 = \$0.01$ per impression, so you should bid up to one cent. Now suppose you used a ranking model instead, which only produces a *relative* score — say it outputs 7.3 for this pair and 4.1 for another. Seven point three of what? You cannot multiply 7.3 by \$0.50; the number has no units of probability. You could try to map scores to probabilities post-hoc with isotonic regression, but a pairwise-trained model was never optimized to be calibrated, so the mapping is fragile and shifts as the score distribution drifts.

Concretely, if your model is systematically over-confident and reports 4% instead of the true 2%, you bid \$0.02 instead of \$0.01 — you double your bid, win impressions you should have lost, and burn budget at a loss. If it is under-confident and reports 1%, you bid \$0.005, lose auctions you should have won, and leave money on the table. A miscalibration of a factor of two translates directly into a factor-of-two error in spend efficiency. This is why ad-tech lives and dies on logloss and calibration error (ECE, expected calibration error), not on NDCG. The classification framing is not a stylistic preference there; it is the only framing whose output you can multiply by money. The figure below lines up four product types against whether they need a true probability, and only the money-multiplying ones force the classification framing.

![Matrix mapping four product types to whether they need a calibrated probability and which framing fits, with ad bidding and blended scores requiring classification](/imgs/blogs/framing-the-problem-rating-ranking-retrieval-8.png)

## 4. The science: why retrieval is the only framing that scales

Here is the constraint that quietly governs everything, and the one that newcomers underestimate by orders of magnitude. The regression and classification framings, as usually built, produce a score by feeding the user-item pair through a model: $s = f(u, i)$, where $f$ is a deep network that mixes user and item features together (cross features, attention, a DLRM-style interaction layer). That expressiveness is exactly what makes them accurate rankers in stage two. But it has a fatal property for stage one: because $f$ entangles $u$ and $i$, there is no way to find the best items for a user without *evaluating $f$ on every item*. To get the top-K from a catalog of $N$ items, you compute $N$ forward passes per request. That is $O(N)$ per request, and $N$ is a billion.

The retrieval framing makes one architectural sacrifice to escape this. It forces the score to *factorize*: $s = g(u) \cdot h(i)$, a dot product between a user vector $g(u)$ and an item vector $h(i)$, where the two towers $g$ and $h$ never see each other's inputs. This is the two-tower model, and the towers can be arbitrarily deep — the only rule is they meet only at the final dot product. The payoff is enormous. Because every item vector $h(i)$ depends only on the item, you compute all $N$ of them *offline*, once, and store them in an index. At request time you compute one user vector $g(u)$ and ask: which stored item vectors have the largest dot product with it? That is **maximum inner product search (MIPS)**, and approximate solutions to it — HNSW graphs, IVF-PQ, ScaNN — run in *sublinear* time, often $O(\log N)$ or close to it in practice. You go from a billion forward passes to one forward pass plus a logarithmic index lookup.

Make the gap concrete. Suppose a single scoring forward pass costs 1 millisecond on a serving host (feature lookup, embedding gather, a few dense layers — a realistic figure for a mid-sized ranker). Scoring a catalog of $N = 10^8$ items the full-scan way costs $10^8 \times 1\,\text{ms} = 10^5\,\text{seconds}$, about 28 hours, *per request*. That is not slow; it is impossible. The retrieval framing computes one user vector (1 ms) and an ANN lookup over precomputed item vectors that returns the top 500 in single-digit milliseconds. The whole request finishes in under 10 ms. The difference between 28 hours and 10 milliseconds is not an optimization — it is the difference between a system that exists and one that does not.

![Before and after panels contrasting full O of N scoring that takes 28 hours per request against retrieval that serves top-K in milliseconds via an approximate nearest neighbor index](/imgs/blogs/framing-the-problem-rating-ranking-retrieval-7.png)

The cost of the factorization is real and worth naming honestly: a dot product of two independent vectors cannot model fine-grained interactions between specific user and item features the way a cross-feature ranker can. The two-tower model is less expressive than a DLRM. That is *fine*, because retrieval's job is not to be the final judge — it is to find a few hundred plausible candidates from a billion without missing the good ones. The expressive ranker then judges those few hundred, where $O(N)$ is now $O(500)$ and entirely affordable. This is precisely the funnel from the [previous post](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking): retrieval framing for stage one because it is the only one that scales, ranking/classification framing for stage two because they are the most accurate when $N$ is small. The framing-to-funnel-stage fit is so central that it deserves its own picture.

![Matrix mapping the four framings to the three funnel stages, showing retrieval fits stage one retrieval while classification and ranking fit stage two ranking](/imgs/blogs/framing-the-problem-rating-ranking-retrieval-6.png)

We will not re-derive the MIPS-to-ANN recall-latency relation here; that is the subject of [the two-tower retrieval post](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval). The point for *framing* is just this: if your catalog is large and you need top-K, the retrieval framing is not one option among four — it is the only one that runs.

### 4.1 The science of the retrieval loss: sampled softmax and the logQ correction

The retrieval framing's loss deserves a careful look, because it is the one most often gotten wrong. The principled objective is a full softmax over the whole catalog: for a user $u$ with a true next item $i$, maximize $\log \frac{\exp(s_{ui})}{\sum_{j \in \text{catalog}} \exp(s_{uj})}$. That denominator sums over a billion items per training example — completely infeasible. So we *sample*: replace the full sum with a sum over a small set of sampled negatives, which gives a sampled softmax. The cheapest source of negatives is the other items already in the training batch — the in-batch negatives in the two-tower code above, where each user's positive item doubles as a negative for every other user in the batch. It is free, because those item vectors are already computed for the forward pass.

But there is a bias, and it is the single most common retrieval bug. The other items in a batch are not a uniform sample of the catalog — they are sampled in proportion to how often they appear as positives, which means *popular items appear as in-batch negatives far more often than rare ones*. Train naively and the model systematically pushes popular items down, because they keep showing up as negatives, which is exactly backwards from what you want. The fix is the **$\log Q$ correction** (from the sampled-softmax literature and made standard for recsys by Google's 2019 two-tower sampling-bias paper): subtract $\log Q_j$ from each negative's logit, where $Q_j$ is the probability item $j$ was sampled as a negative. Because in-batch sampling draws negatives in proportion to their frequency, $Q_j$ is roughly the item's empirical popularity, so the corrected logit is $s_{uj} - \log Q_j$. This exactly cancels the over-representation of popular items in expectation, recovering an unbiased estimate of the full softmax gradient. The corrected loss is the difference between a retrieval model that ranks well and one whose top-K is dominated by either over- or under-suppressed head items. The full derivation lives in the loss and retrieval posts; the framing-level lesson is that the retrieval framing's negatives are not "items the user did not click" — they are a *sampled proposal distribution that must be corrected*, a subtlety with no analog in the regression framing.

#### Worked example: the embedding-table memory of the retrieval framing

The retrieval framing's scaling win comes with a memory bill worth pricing, because it determines whether your item index even fits in RAM. Take a catalog of $N = 10^8$ items and a $d = 64$-dimensional item embedding in float32 (4 bytes). The item matrix alone is $10^8 \times 64 \times 4\,\text{bytes} = 2.56 \times 10^{10}$ bytes $\approx 25.6$ GB. That fits on a single large host's RAM, but barely, and the flat (brute-force) `IndexFlatIP` must scan all 25.6 GB per query — fast for a million items, too slow at a hundred million. Switch to product quantization (`IndexIVFPQ`), which compresses each 64-dim vector to, say, 16 bytes of codes: now the index is $10^8 \times 16\,\text{bytes} = 1.6$ GB, a 16x reduction, and IVF clustering means each query touches only a few percent of the items. The cost is a small recall hit (PQ is lossy), the classic recall-versus-memory-versus-latency Pareto. The regression and classification framings have *no equivalent option*, because there is no precomputable item vector to quantize — their score requires the user at compute time. The retrieval framing's factorization is what makes the item side cacheable, quantizable, and indexable, and that is the whole reason it scales. The arithmetic — 25.6 GB flat versus 1.6 GB quantized for the same hundred million items — is the engineering reality behind the "scales to 1B" checkmark in the table.

## 5. The decision: a tree you can run in thirty seconds

You now have the four framings and the science behind each. Choosing among them is not a vibe; it is a short decision procedure driven by four questions. Walk them in order.

**Question 1 — Do you have explicit ratings, or only implicit feedback?** If you have genuine numeric ratings (stars, scores) *and* you actually need the absolute value, regression is on the table. If you have only implicit signals — clicks, plays, purchases, dwell — regression is mostly off the table, because there is no "rating" to regress; binarizing a click into a 1 and treating non-clicks as 0 is really the classification framing in disguise. Most modern systems have implicit feedback, which is why explicit rating regression has faded since the Netflix Prize era. We covered the implicit-versus-explicit distinction in depth in [the data-you-have post](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have); the short version is that implicit feedback is positive-only, and positive-only data steers you toward ranking and retrieval.

**Question 2 — Do you need every item scored, or just the top-K?** If you need a score for one specific pair (a "will this user like this one item" question, a single-item satisfaction prediction) or a calibrated score for *every* item (an ad auction that may bid on any impression), you are in the all-scored regime, where regression and classification live. If you only ever surface the top handful from a large set — a feed, a "more like this" row, an email digest — you are in the top-K regime, where ranking and retrieval live.

**Question 3 — How many items?** Thousands, or billions? This is the question that splits ranking from retrieval *within* the top-K regime. If the candidate set is small (you already retrieved a few hundred, or your whole catalog is a few thousand), learning-to-rank is great: you can afford to score every candidate and optimize their order directly. If the catalog is millions or billions and you have to find the candidates in the first place, you need the retrieval framing, because only its factorized score supports ANN. This is exactly why the funnel exists: retrieval framing narrows billions to hundreds, then ranking framing orders the hundreds.

**Question 4 — Do you need a calibrated probability, or just the order?** If a downstream system multiplies your score by money (bidding), sums it across events (expected total engagement), or blends it with other models' scores (a weighted utility), you need a real probability and the classification framing is forced. If you only ever sort by the score and show the top, order is all you need and ranking or retrieval will do — and will usually rank better than a classifier, because they optimize order directly.

![Decision tree for choosing a recommendation framing that branches on whether you need all items scored or only top-K and then on calibration and catalog size](/imgs/blogs/framing-the-problem-rating-ranking-retrieval-2.png)

Run those four questions and you land on a framing almost every time. A feed recommender over a hundred-million-item catalog: top-K, billions, order only — retrieval for stage one, ranking for stage two. An ad system: all-scored, calibrated probability needed — classification. A "rate this for you" widget: all-scored, absolute value needed, explicit ratings present — regression. A "more like this" row over a few thousand SKUs: top-K, thousands, order only — learning-to-rank. The framing is downstream of the product, and the product answers the four questions for you.

The reason getting this right early matters so much is that the framing propagates into every other decision. It fixes the model head (a scalar output for regression, a sigmoid for classification, a pairwise margin for ranking, two towers for retrieval). It fixes the loss. It fixes what a negative is (none for regression, unclicked-shown for classification, sampled pairs for ranking, in-batch plus the $\log Q$ correction for retrieval). It fixes the data pipeline (points, pairs, or triples). And it fixes the serving shape (full scan or ANN index). Change the framing late and you are not tweaking a hyperparameter — you are rewriting the system. The figure below traces that propagation: one framing decision fanning out into the head, loss, negatives, metric, data pipeline, and serving shape.

![Graph showing a single framing decision propagating downstream into the model head, loss, negatives, metric, data pipeline, and serving shape](/imgs/blogs/framing-the-problem-rating-ranking-retrieval-4.png)

### 5.1 How the framing reshapes the data pipeline

The propagation into the *data pipeline* is the part teams discover late and painfully, so it is worth spelling out. Each framing wants its training examples in a different shape, and the shape determines how you join, sample, and batch your logs:

- **Regression wants points.** Each row is a single $(u, i, r)$ triple where $r$ is the numeric target. No negatives, no sampling — just the observed ratings, streamed in any order. The simplest pipeline of the four, which is part of why it is so tempting.
- **Classification wants labeled points with constructed negatives.** Each row is $(u, i, y)$ where $y$ is 0 or 1. The positives come from the click log, but you must *manufacture* negatives by sampling unshown or unclicked items, deciding a positive-to-negative ratio (commonly 1:4 to 1:10), and choosing a sampling distribution (uniform, popularity-weighted, or hard negatives). The negative sampler is now a first-class component you must build, monitor, and version.
- **Ranking wants pairs or triples.** BPR consumes $(u, i_{\text{pos}}, j_{\text{neg}})$ triples, so your pipeline must, for every positive, draw a negative the user did not interact with — which means maintaining a fast "has this user seen this item" lookup to avoid sampling false negatives, at the scale of your interaction set.
- **Retrieval wants batches that are themselves the negatives.** With in-batch negatives, the batch composition *is* the negative distribution, so batch size and shuffling are no longer just throughput knobs — they change what the model learns. Bigger batches give more negatives per positive (better), but also concentrate the $\log Q$ popularity correction's importance.

That is four genuinely different ETL jobs, four different sampler designs, four different monitoring dashboards. A team that frames the problem as classification and later wants to switch to retrieval is not just swapping a model file — it is rebuilding the data pipeline from points-with-negatives to batches-as-negatives. This is the deepest sense in which framing is upstream: it reaches all the way back to how the raw logs are joined into examples.

## 6. The practical flow: train one dataset under all four framings

Talk is cheap; let us train all four on the same data and watch them disagree. The dataset is MovieLens — the `iris` of recsys — using the ratings log. We treat ratings of 4 stars and up as positive interactions (the standard implicit-from-explicit convention) and do a temporal split: for each user, hold out their most recent positive interaction as the test item, train on the rest. Then we evaluate *every* model two ways: on its own natural metric, and on a common Recall@10 and NDCG@10 over a candidate set, so the comparison is apples to apples on what the product cares about.

First, the data preparation — shared by all four models, because the framing changes the *labels*, not the log.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Load MovieLens ratings: columns user_id, item_id, rating, timestamp
df = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python",
                 names=["user_id", "item_id", "rating", "ts"])

# Contiguous integer ids for embedding tables
u_codes = {u: k for k, u in enumerate(df.user_id.unique())}
i_codes = {i: k for k, i in enumerate(df.item_id.unique())}
df["u"] = df.user_id.map(u_codes)
df["i"] = df.item_id.map(i_codes)
n_users, n_items = len(u_codes), len(i_codes)

# Temporal split: each user's most recent interaction is the test item
df = df.sort_values("ts")
test = df.groupby("u").tail(1)
train = df.drop(test.index)

# Implicit positives: rating >= 4 is a "positive interaction"
train_pos = train[train.rating >= 4][["u", "i", "rating"]].reset_index(drop=True)
print(f"{n_users} users, {n_items} items, {len(train_pos)} positive interactions")
```

That single `train_pos` frame is the common substrate. Now four models read it four ways.

### Framing 1 — Regression matrix factorization (predict the rating, minimize RMSE)

The simplest latent-factor model: a user embedding and an item embedding whose dot product, plus biases, predicts the star rating. Trained with mean-squared error. Note that this model uses the *rating* column and never sees a negative — its world is only the observed ratings.

```python
class RegMF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.I = nn.Embedding(n_items, dim)
        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        self.mu = nn.Parameter(torch.tensor(3.5))   # global mean rating
        nn.init.normal_(self.U.weight, std=0.05)
        nn.init.normal_(self.I.weight, std=0.05)

    def forward(self, u, i):
        dot = (self.U(u) * self.I(i)).sum(-1)
        return self.mu + self.ub(u).squeeze(-1) + self.ib(i).squeeze(-1) + dot

model = RegMF(n_users, n_items)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
mse = nn.MSELoss()

u = torch.tensor(train.u.values)        # ALL ratings, not just positives
i = torch.tensor(train.i.values)
r = torch.tensor(train.rating.values, dtype=torch.float32)
for epoch in range(20):
    opt.zero_grad()
    pred = model(u, i)
    loss = mse(pred, r)                  # squared error on the rating
    loss.backward(); opt.step()
```

This model will get a low RMSE. To rank with it at eval time, we score the user against all items by the same dot-plus-bias formula and take the top-10 — but it was never trained to put the right items on top.

### Framing 2 — Binary CTR MLP (predict the click, minimize logloss)

Now binarize: a positive interaction is label 1, and we *sample* negatives (items the user did not interact with) as label 0. The model is a small multilayer perceptron over the concatenated user and item embeddings, with a sigmoid output, trained with binary cross-entropy. The negative sampling is the crux — it is what turns positive-only data into a classification problem.

```python
class CtrMLP(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.I = nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, u, i):
        x = torch.cat([self.U(u), self.I(i)], dim=-1)
        return self.mlp(x).squeeze(-1)   # logit

model = CtrMLP(n_users, n_items)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-6)
bce = nn.BCEWithLogitsLoss()

pos_u = train_pos.u.values; pos_i = train_pos.i.values
for epoch in range(20):
    # 4 sampled negatives per positive, drawn uniformly from the catalog
    neg_i = np.random.randint(0, n_items, size=4 * len(pos_i))
    uu = np.concatenate([pos_u, np.repeat(pos_u, 4)])
    ii = np.concatenate([pos_i, neg_i])
    yy = np.concatenate([np.ones(len(pos_i)), np.zeros(len(neg_i))])
    perm = np.random.permutation(len(yy))
    opt.zero_grad()
    logit = model(torch.tensor(uu[perm]), torch.tensor(ii[perm]))
    loss = bce(logit, torch.tensor(yy[perm], dtype=torch.float32))
    loss.backward(); opt.step()
```

Apply a sigmoid to the logit and you get a probability — and because BCE is a proper scoring rule, that probability is roughly calibrated *to the sampled negative distribution* (a caveat we return to). To rank, sort items by predicted probability.

### Framing 3 — BPR pairwise ranker (order the items, optimize pairwise)

BPR (Bayesian Personalized Ranking) trains directly on the *order*. For each positive interaction $(u, i)$ it samples a negative item $j$ the user did not interact with and pushes the score of $i$ above the score of $j$. The loss is $-\log \sigma(s_{ui} - s_{uj})$, the negative log-sigmoid of the score difference. This is the gradient that "sees the order, not the absolute score" — the central reason pairwise beats pointwise for top-K.

```python
class BprMF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.I = nn.Embedding(n_items, dim)
        nn.init.normal_(self.U.weight, std=0.05)
        nn.init.normal_(self.I.weight, std=0.05)

    def score(self, u, i):
        return (self.U(u) * self.I(i)).sum(-1)

    def bpr_loss(self, u, i_pos, i_neg):
        s_pos = self.score(u, i_pos)
        s_neg = self.score(u, i_neg)
        return -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()

model = BprMF(n_users, n_items)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
for epoch in range(20):
    j = np.random.randint(0, n_items, size=len(pos_i))   # sampled negative
    opt.zero_grad()
    loss = model.bpr_loss(torch.tensor(pos_u),
                          torch.tensor(pos_i),
                          torch.tensor(j))
    loss.backward(); opt.step()
```

The gradient of the BPR loss with respect to the score difference is $\sigma(s_{uj} - s_{ui})$ — it is largest exactly when a negative is *ranked above* a positive (the violation), and near zero when the pair is already correctly ordered. The model concentrates its learning on fixing inversions at the top, which is what NDCG rewards. We will derive this gradient fully in [the loss-function landscape post](/blog/machine-learning/recommendation-systems/the-loss-function-landscape-for-recsys); here it is enough to see that the loss is a function of the *difference* of two scores, so it can never reward a model for getting an absolute value right.

### Framing 4 — Two-tower retrieval (embed and retrieve, sampled softmax)

The two-tower model embeds the user and the item independently and scores by dot product, trained with an in-batch sampled softmax: within a training batch, each user's true item is the positive and all *other* users' items in the batch are negatives. This is the framing built for scale, because at serve time you precompute every item vector and use ANN.

```python
class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.I = nn.Embedding(n_items, dim)
        nn.init.normal_(self.U.weight, std=0.05)
        nn.init.normal_(self.I.weight, std=0.05)

    def forward(self, u, i):
        return self.U(u), self.I(i)      # user vec, item vec (independent towers)

model = TwoTower(n_users, n_items)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
ce = nn.CrossEntropyLoss()
B = 1024
for epoch in range(20):
    order = np.random.permutation(len(pos_i))
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        u_vec, i_vec = model(torch.tensor(pos_u[idx]), torch.tensor(pos_i[idx]))
        logits = u_vec @ i_vec.t()       # B x B: row r vs every item in batch
        target = torch.arange(len(idx))  # the diagonal is the true positive
        opt.zero_grad()
        loss = ce(logits, target)        # in-batch sampled softmax
        loss.backward(); opt.step()
```

The honest version of this loss applies a $\log Q$ correction to debias the in-batch negatives (popular items appear as negatives more often than uniform, so their logits are subtracted by $\log Q_i$ where $Q_i$ is the sampling probability). That correction, and the reason sampled softmax needs it, is the subject of the retrieval and loss posts; the skeleton above is the un-corrected baseline. To serve, you `model.I.weight` is your item matrix — push it into a `faiss.IndexFlatIP`, embed the user, and query the top-K.

```python
import faiss
item_vecs = model.I.weight.detach().numpy().astype("float32")
index = faiss.IndexFlatIP(item_vecs.shape[1])   # inner-product index
index.add(item_vecs)
# at serve time: top-10 for a user in sublinear time (with IVF/HNSW)
u_vec = model.U.weight.detach().numpy()[user_id:user_id + 1].astype("float32")
scores, top10 = index.search(u_vec, 10)
```

That `index.search` is the whole point of the retrieval framing: with a flat index it is a brute-force scan, but swap in `IndexHNSWFlat` or `IndexIVFPQ` and the same call is sublinear, which is what lets it serve a billion items. None of the other three framings can do this, because none of them factorizes the score into a user vector and an item vector.

## 7. Results: the same data, four framings, on a common metric

Now the measurement, which is where the argument stops being theory. We evaluate each model on its *own* natural metric and on a *common* Recall@10 and NDCG@10. The eval harness ranks the held-out test item against a candidate pool (the test positive plus a sample of items the user has not interacted with) and asks whether the true item lands in the top-10, with what discount. Recall@10 here is the hit rate — fraction of users whose held-out item is in their top-10 — and NDCG@10 weights by position.

```python
def evaluate_topk(score_user_items, test, n_items, K=10, n_neg=100, seed=0):
    rng = np.random.default_rng(seed)
    hits, ndcgs = [], []
    for u, true_i in zip(test.u.values, test.i.values):
        negs = rng.integers(0, n_items, size=n_neg)
        cand = np.concatenate([[true_i], negs])           # 1 positive + 100 negs
        scores = score_user_items(u, cand)                # framing-specific scorer
        order = cand[np.argsort(-scores)]
        topk = order[:K]
        if true_i in topk:
            rank = np.where(topk == true_i)[0][0] + 1
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 1))         # ideal DCG is 1 here
        else:
            hits.append(0.0); ndcgs.append(0.0)
    return np.mean(hits), np.mean(ndcgs)                   # Recall@10, NDCG@10
```

The numbers below are representative of what you get running this protocol on MovieLens-1M with a 64-dimensional model and 20 epochs each; treat them as illustrative order-of-magnitude figures from this kind of setup rather than a leaderboard, because exact values depend on seeds, the negative pool, and tuning. The pattern, however, is robust and reproducible: the framings trained to rank win the ranking metric, the framing trained to predict a rating loses it, and the framing trained to predict a probability is the only one whose output is calibrated.

| Framing | Model | Loss | Natural metric | Recall@10 | NDCG@10 | Scales to 1B? | Calibrated prob? |
|---|---|---|---|---|---|---|---|
| Regression | RegMF | MSE | RMSE 0.87 | 0.41 | 0.23 | No (O(N) scan) | No |
| Classification | CtrMLP | logloss | AUC 0.79 | 0.58 | 0.33 | No (O(N) scan) | Yes |
| Learning-to-rank | BprMF | BPR pairwise | — | 0.66 | 0.39 | Partly | No |
| Retrieval | TwoTower | sampled softmax | — | 0.68 | 0.40 | Yes (ANN) | No |

Read the table column by column and every claim in this post falls out of it. The **regression** model has a fine RMSE (0.87) and the worst top-K numbers by a wide margin — it nailed the absolute scores and ranked badly, exactly the counterexample of section 2 playing out on real data. The **classification** model ranks far better than regression because logloss at least cares about separating positives from negatives, and it is the only one with a calibrated probability — but it trails the dedicated rankers on NDCG because logloss spends capacity calibrating obvious negatives instead of sharpening the contested top. The **ranking** and **retrieval** framings win the top-K metrics because they optimize order directly, and the retrieval model alone earns the "scales to 1B" checkmark because its factorized score supports ANN. No single framing dominates every column. That is the entire point: the framing you pick *is* the column you are choosing to win.

#### Worked example: reading the deltas like an engineer

Suppose your product is a top-10 feed and you are deciding between the classification framing (your current CTR MLP at Recall@10 = 0.58) and switching to a BPR ranker (Recall@10 = 0.66). That is an absolute lift of 0.08, or a *relative* lift of $0.08 / 0.58 = 13.8\%$ in hit rate. If the feed serves 10 million sessions a day and a "hit" (the right item in the top-10) converts to engagement at some base rate, a 13.8% relative lift in hits is a large online swing — the kind that, in the public literature, has corresponded to single-digit-percent watch-time or GMV gains when it survives an A/B test. But notice the cost on the *other* axes: you lose the calibrated probability. If anything downstream multiplied that score by money, switching to BPR breaks it silently. The framing decision is a trade across columns, and the worked arithmetic forces you to price the trade instead of optimizing one number in isolation. The disciplined move is to keep the classification framing for the stage that needs calibration and use the ranking framing for the stage that only needs order — which is, again, the funnel.

## 8. Case studies: how real systems chose (and re-chose) their framing

The history of recommendation is, to a surprising degree, the history of teams discovering they framed the problem wrong and re-framing it. Four cases make the lesson vivid.

**The Netflix Prize: RMSE, and the pivot away from it.** The 2006–2009 Netflix Prize is the most famous recommender competition ever run, and it was framed entirely as *rating prediction*: predict the 1-to-5 stars a user would give a movie, minimize RMSE on a held-out set, and the team that cut RMSE by 10% over Netflix's Cinematch baseline wins a million dollars. The winning ensemble (BellKor's Pragmatic Chaos, 2009) was a heroic blend of matrix factorization, restricted Boltzmann machines, and temporal models. And then Netflix, by their own later accounts, **did not deploy the winning ensemble** — partly for engineering cost, but more tellingly because their product had moved on from a five-star widget toward ranked rows of recommendations, where RMSE was not the metric that mattered. Netflix's own engineering blog later described their shift toward *ranking* objectives and learning-to-rank for the homepage. The Prize is the canonical cautionary tale of optimizing the wrong framing: a million-dollar RMSE improvement that the product could not use because the product was a ranking problem all along. The matrix factorization techniques the Prize popularized are still everywhere — Koren, Bell, and Volinsky's 2009 "Matrix Factorization Techniques for Recommender Systems" is foundational — but the *framing* the Prize used has been largely abandoned.

**YouTube: the retrieval-plus-ranking split.** The 2016 paper "Deep Neural Networks for YouTube Recommendations" (Covington, Adams, Sargin) is the clearest public statement of the two-framing funnel. YouTube explicitly splits the system into a **candidate generation** model and a **ranking** model. Candidate generation is framed as *extreme multiclass classification / retrieval* — it predicts which video a user will watch next out of millions, trained with a sampled softmax over the catalog, and crucially produces user and video embeddings so that serving is a nearest-neighbor lookup, not a full scan. The ranking model is framed differently: it scores the few hundred candidates with a richer feature set, optimized for expected watch time. Two stages, two framings, chosen for exactly the reasons in this post — retrieval framing for the billion-item stage because it is the only one that scales, a finer-grained framing for the small-candidate stage because accuracy matters more than throughput there. The 2019 follow-up on multitask ranking (the "Recommending What Video to Watch Next" paper) deepened the ranking framing with multiple objectives, but kept the split.

**Ad tech: calibration is non-negotiable.** In computational advertising the framing is fixed by the auction, not by taste. The seminal works — Google's 2013 "Ad Click Prediction: a View from the Trenches" (McMahan et al.) and Facebook's 2014 "Practical Lessons from Predicting Clicks on Ads" (He et al.) — both frame the problem as *binary click prediction with calibrated probability*, optimizing logloss, and both spend significant ink on calibration because the predicted pCTR is multiplied by a bid. Facebook's paper is famous for the gradient-boosted-trees-plus-logistic-regression stack, but the framing lesson is the durable one: when your score becomes a bid, you cannot use a pure ranking framing, because a ranking score has no probabilistic units. The entire ad-tech industry runs on the classification framing for this single reason, and measures itself in logloss and calibration error, not NDCG.

**Sequential recommendation benchmarks: framing as next-item prediction.** A fourth, more recent case shows the framing question playing out in academic benchmarks. SASRec (Kang and McAuley, 2018) and BERT4Rec (Sun et al., 2019) both frame recommendation as *sequential next-item prediction* — given a user's history, predict the next item — which is a retrieval-style framing trained with a softmax (or sampled softmax) over the catalog, evaluated with Recall@K and NDCG@K. The interesting twist is what happened to their reported numbers. Because these papers evaluated against a *sampled* candidate set rather than the full catalog, later work (including the Krichene-Rendle KDD'20 result) showed the rankings between methods could shift when you switch to full evaluation. The framing of the *evaluation* — full versus sampled top-K — turned out to matter as much as the framing of the *model*. The lesson compounds the post's thesis: you frame the training objective, but you also frame the metric, and an honest comparison requires both framings to match the production reality. A model that wins on sampled NDCG@10 may not win on full NDCG@10, and only the full version reflects what serving actually does.

The throughline across all four: the framing was not a detail downstream of the model — it was the *first* decision, and getting it wrong (Netflix) or splitting it correctly (YouTube), being forced into it by the business (ad tech), or quietly distorting it in evaluation (sampled metrics) determined whether the system, and the comparison, told the truth.

## 9. Stress-testing the framing choice

A decision is only as good as its behavior at the edges, so push on each framing.

**What if you only have implicit feedback?** Regression is essentially out — there is no rating to regress, and binarizing-then-MSE is just a worse classifier. You are choosing among classification, ranking, and retrieval. This is the common case in 2026, and it is why ranking and retrieval framings dominate modern systems. The positive-only nature also means your negatives are *constructed*, not observed, which makes negative sampling a first-class design decision in three of the four framings (everything but regression).

**What happens at 100 million items?** Regression and classification, built as cross-feature models, must score all of them per request — that is the 28-hour calculation from section 4, and it is simply infeasible. Only the retrieval framing survives, because only its factorized dot-product score supports ANN. If someone proposes a "score everything with a deep ranker" design at 100M items, the framing is wrong before any code is written, and the fix is not a faster GPU — it is re-framing the first stage as retrieval.

**What if the negatives are mostly false negatives?** In implicit feedback, an "unclicked" item the user never saw is not a true negative — it is unlabeled. If you treat all of them as negatives (the naive classification framing), you train the model to suppress items that might be great, just unseen. This hurts every framing that uses sampled negatives, but it bites the classification framing hardest because its calibration is *with respect to* that negative distribution: a wrong negative distribution gives you a confidently-wrong probability. Ranking and retrieval are somewhat more robust because they only need the positive to outscore the *sampled* negative on average, but false negatives still inject noise. This is why hard-negative mining and importance weighting (the $\log Q$ correction in sampled softmax) exist.

**What if offline NDCG rises but online engagement is flat?** This is the offline-online gap, and it is not always a framing bug — it can be position bias, distribution shift, or missing-not-at-random data, which we cover in the evaluation and bias posts. But framing *contributes*: if you optimized a metric (say, NDCG over a sampled candidate set) that does not reflect the real serving distribution, the offline win can be an artifact. The 2020 KDD result by Krichene and Rendle, "On Sampled Metrics for Item Recommendation," showed that sampled top-K metrics can be *inconsistent* — they can even reverse the ranking of two models compared to full metrics — so an offline NDCG computed against a small sampled candidate pool is a treacherous foundation. The framing fix is to evaluate against the full item set when feasible, and to remember that the offline metric is a proxy, not the goal.

**What if the feature is computed differently offline and online?** Train-serve skew is orthogonal to framing in principle, but framing affects how badly it bites. A retrieval model precomputes item vectors offline; if the offline item-feature pipeline differs from the online user-feature pipeline, the dot products are computed in inconsistent spaces and recall collapses silently. The framing does not cause the skew, but the two-tower architecture makes it easy to introduce, because the two towers are often built and deployed by different code paths. The defensive move is a single shared feature-transformation library used by both training and serving — a theme we return to throughout the series.

**What happens on a brand-new item with no interactions?** Cold-start exposes a framing weakness that is easy to miss. A pure ID-based regression or BPR model has no embedding for an item it has never seen, so it cannot score it at all — the item is invisible until it accumulates interactions, and it cannot accumulate interactions until it is shown. The retrieval framing handles this more gracefully *if* the item tower consumes content features (text, image, category) rather than only an ID, because then a new item gets a reasonable embedding from its features on day one and can be retrieved before any clicks exist. The classification framing similarly can lean on content features in the MLP. So cold-start is not solved by the framing alone, but the framing constrains your options: an ID-only model of any framing is structurally helpless on new items, while a content-aware two-tower or CTR model degrades gracefully. When a product manager asks "will this work on items we launched this morning," the honest answer depends almost entirely on whether your chosen framing's towers consume features or only IDs — another way the framing decision reaches far past the loss function into the product's behavior at the edges.

## 10. When to reach for each framing (and when not to)

Be decisive. Every framing is a cost, and the honest engineering move is to say plainly when each is *not* worth it.

**Reach for regression** only when you genuinely need the absolute score — a satisfaction prediction, a price estimate, a "predicted rating" widget, or a regression target feeding a downstream optimizer. Do **not** reach for it for a top-K list, which is almost every recommender; you will optimize RMSE and ship a model that ranks no better, exactly the six-week mistake.

**Reach for classification** when something multiplies, sums, or blends your score — ad bidding, expected-value optimization, a weighted multi-objective utility, or any case where a calibrated probability earns its keep. Do **not** reach for it as your *only* framing in a pure top-K feed if you do not need calibration, because a dedicated ranker will usually rank better; and do **not** trust its calibration if your negatives are garbage, because calibration is only as honest as the negative distribution.

**Reach for learning-to-rank** when the candidate set is already small (hundreds to a few thousand) and you only need correct order — the classic stage-two ranker. Do **not** reach for it as your retrieval stage over a huge catalog; a pairwise or listwise model still has to score every candidate to rank it, so it cannot find the candidates in the first place at billion-item scale.

**Reach for retrieval** whenever the catalog is large and you need top-K — it is the only framing that serves a billion items, full stop, because only its factorized score supports ANN. Do **not** use it as your final, most-accurate scorer; the factorization sacrifices cross-feature expressiveness, so let it find candidates and hand them to a ranker. And do **not** forget the $\log Q$ correction on in-batch negatives, or popular items will be unfairly penalized.

The meta-rule: in any system over a large catalog, you will use *at least two* framings — retrieval for stage one, classification or ranking for stage two — because the funnel is two framings stacked. The four framings are not competitors fighting for one slot; they are specialists, each suited to a different stage and a different question. Choosing "one framing for the whole system" is itself the mistake.

One more piece of hard-won advice on *sequencing the work*. When you start a recommender from scratch, resist the urge to build the most sophisticated framing first. Begin with the framing that matches your immediate product surface and your immediate data — usually a classification or BPR model over a modest candidate set — and ship it. The retrieval framing earns its complexity only once your catalog outgrows what a full scan can handle, and the calibration machinery of the classification framing earns its keep only once a downstream system multiplies your score. Adding a framing because the architecture diagram looks more impressive, before the scale or the product demands it, is how you end up maintaining a two-tower retrieval stage and an ANN index to serve a catalog of ten thousand items that a single GPU could rank in full. The framing should track the constraint, and the constraint should be real and present, not anticipated three years out.

## 11. Key takeaways

- **Framing is the first and most consequential decision.** It fixes your model head, loss, metric, negatives, data pipeline, and serving shape all at once — change it late and you rewrite the system, not a hyperparameter.
- **The same log supports all four framings.** Regression, classification, ranking, and retrieval are relabelings of identical interaction data; the framing is a modeling choice, not a data constraint.
- **RMSE-optimal is not ranking-optimal.** A model with strictly lower RMSE can put the wrong item on top, because RMSE measures residual magnitude and ranking measures the sign of pairwise differences — different objectives with crossing level sets.
- **Classification is the only framing that gives a calibrated probability for free,** because cross-entropy is a proper scoring rule. If your score is multiplied by money, you are forced into it.
- **Retrieval is the only framing that scales to billions,** because it factorizes the score into a user vector times an item vector, which is exactly the maximum-inner-product-search problem that ANN solves in sublinear time.
- **Run the four-question decision tree:** explicit ratings or implicit? all items scored or top-K? thousands or billions? calibrated probability or just order? Those four answers pick your framing almost every time.
- **Calibration and ranking are different goals.** A model can be perfectly calibrated and rank the contested top slightly wrong; a model can rank perfectly and be wildly miscalibrated. Pick the framing that optimizes the goal you actually have.
- **Negatives are part of the framing.** Regression uses none, classification uses unclicked-shown, ranking uses sampled pairs, retrieval uses in-batch negatives with a $\log Q$ correction — and false negatives in implicit data hurt the calibrated framing most.
- **Real systems use at least two framings.** The funnel is retrieval (stage one) plus ranking or classification (stage two); choosing a single framing for the whole system is itself the error, as YouTube's split and Netflix's pivot both show.

## 12. Further reading

- Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," *IEEE Computer*, 2009 — the rating-prediction / regression framing that won the Netflix Prize era.
- S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback," *UAI*, 2009 — the canonical pairwise learning-to-rank framing.
- P. Covington, J. Adams, E. Sargin, "Deep Neural Networks for YouTube Recommendations," *RecSys*, 2016 — the explicit retrieval (candidate generation) plus ranking split.
- H. B. McMahan et al., "Ad Click Prediction: a View from the Trenches," *KDD*, 2013, and X. He et al., "Practical Lessons from Predicting Clicks on Ads at Facebook," *ADKDD*, 2014 — the calibrated-classification framing of ad tech.
- W. Krichene, S. Rendle, "On Sampled Metrics for Item Recommendation," *KDD*, 2020 — why sampled top-K metrics can be inconsistent, and why your offline framing of evaluation matters as much as your training framing.
- Official docs to go deeper on the toolchain: the `faiss` wiki for ANN indices, the `implicit` library for ALS and BPR, and RecBole for a config-driven model zoo that lets you swap framings by editing a YAML file.
- Within this series: start from [what a recommender system is](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) and [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking); go deeper on losses in [the loss-function landscape for recsys](/blog/machine-learning/recommendation-systems/the-loss-function-landscape-for-recsys), on metrics in [offline evaluation metrics](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr), and on the retrieval framing in [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval); and see how every framing decision fits the whole picture in [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

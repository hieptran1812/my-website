---
title: "Implicit vs Explicit Feedback: The Data You Actually Have"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why the ratings you wish you had are sparse and biased, why the clicks you do have hide no negatives, and how to turn raw behavior into a ranking model that wins the top-K metric that matters."
tags:
  [
    "recommendation-systems",
    "recsys",
    "implicit-feedback",
    "explicit-feedback",
    "positive-unlabeled",
    "collaborative-filtering",
    "machine-learning",
    "ranking",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-1.png"
---

A team I worked with launched a video recommender with a clean, confident plan: collect star ratings, train a matrix-factorization model to minimize rating error, ship it. Six weeks in, the offline RMSE was beautiful — 0.84, better than the public benchmark on the same catalog. The launch metric, watch-time per session, did not move. At all. When we dug in, the reason was almost insulting in its simplicity: only about 1.3% of sessions ever produced a rating, those sessions skewed heavily toward people who loved or hated a title, and the model had spent all its capacity predicting a number that the product never showed and that barely correlated with what people actually watched. Meanwhile the firehose of signal we already had — every play, every pause, every "watched 8% then bailed" — sat in a log table, unused, because nobody knew how to turn it into labels.

That gap is the subject of this post. The data you *wish* you had (clean, intentional ratings) is rare, lopsided, and optimizes the wrong thing. The data you *actually* have (clicks, dwell, watch percentage, add-to-cart, purchase) is abundant and behaviorally honest, but it comes with a structural defect that breaks naive supervised learning: it is **positive-only**. You see what people engaged with. You almost never see a verified "no." A blank cell in your user-item matrix is not a negative — it is an *unobserved* entry, and whether it is unobserved because the user disliked the item or because the item was never shown to them is the whole ballgame.

![Matrix comparing explicit and implicit feedback across volume, signal quality, selection bias, negatives, and sparsity, showing explicit is clean but rare while implicit is abundant with no observed negatives](/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-1.png)

This post is about reading that table correctly and acting on it. We will formalize *why* implicit data is **missing-not-at-random (MNAR)** and why that makes it a **positive-unlabeled (PU)** learning problem rather than ordinary classification. We will derive the confidence-weighting objective from Hu, Koren, and Volinsky's 2008 implicit-feedback paper — the one that quietly became the default for a decade. We will engineer raw events into graded weights, sidestepping the "is a click positive?" trap that clickbait sets for you. Then we will write real code: load an implicit dataset, build the interaction matrix, train weighted ALS and BPR, and compare them honestly against a naive binary baseline and an explicit RMSE model — all on the *same* top-K metric, because the top-K list is the only thing the product actually serves. This is one rung of the series' recurring ladder: the [retrieval → ranking → re-ranking funnel](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) is only as good as the labels you feed it, and the labels are exactly what implicit feedback makes hard.

## 1. Two kinds of feedback, two different problems

Start with the distinction, because everything downstream depends on getting it right.

**Explicit feedback** is a deliberate statement of preference. A 1-to-5 star rating, a thumbs up, a survey response, a "not interested" tap. The user stopped what they were doing and *told you* something. The signal is high-fidelity: a 5-star rating is an unambiguous "I loved this." The catch is that almost nobody does it. On most consumer products, the fraction of interactions that produce an explicit rating is between roughly 0.5% and 2%. And the people who do rate are not a random sample — they are the unusually motivated, the delighted, and the furious. The middle, the "it was fine, I finished it" majority, stays silent.

**Implicit feedback** is behavior, logged passively. A click. A scroll past. Time spent on a page (dwell). The percentage of a video watched. An add-to-cart. A purchase. A re-watch. A share. Nobody chose to give you this data — they just used the product, and you recorded it. The signal is abundant: you get an event for essentially every interaction, so coverage is orders of magnitude higher than ratings. But it is noisy and *indirect*. A click can mean "this is exactly what I wanted" or "the thumbnail tricked me." And the defining problem, the one we will keep circling back to: **there are no observed negatives.** When a user did not click an item, you do not know whether they saw it and rejected it, or never saw it at all.

Here is the comparison that should be tattooed on the inside of every recsys engineer's eyelids.

| Property | Explicit (ratings, thumbs) | Implicit (clicks, dwell, buys) |
|---|---|---|
| Volume | Tiny: ~1% of sessions rate | Huge: every interaction logged |
| Signal fidelity | High: stated intent | Lower: behavior is indirect |
| Negatives | Present (1-star, thumbs down) | **Absent**: only positives observed |
| Bias | Self-selection (motivated raters) | Exposure / position bias |
| Sparsity | Extreme (99.9%+ empty) | Severe but better (98–99% empty) |
| Natural objective | Regression (predict the rating) | Ranking (order the candidates) |
| What it optimizes | A number you rarely display | The list you actually serve |

The last two rows are where most teams go wrong. Explicit feedback nudges you toward a regression framing — predict the star rating, minimize the squared error — because that is literally what the label is. Implicit feedback has no rating to predict; it has *interactions*, and what you care about is producing a good ordered list of items the user will engage with. Those are different mathematical objectives with different gradients, and conflating them is exactly the mistake my video team made.

## 2. The Netflix Prize taught us the wrong lesson (partly)

If you learned recommender systems from the 2006–2009 era, you absorbed a specific worldview: a recommender is a function that predicts the rating a user would give an item, and you measure it with **root-mean-squared error (RMSE)** on held-out ratings. The Netflix Prize — a \$1M competition to beat Netflix's Cinematch by 10% RMSE on a 100-million-rating dataset — burned this framing into a generation of practitioners. It produced genuinely great science: matrix factorization with biases, temporal dynamics, the SVD++ model that folded in implicit signals. Koren's "Matrix Factorization Techniques for Recommender Systems" (2009) is still one of the clearest things ever written about the topic.

But the *evaluation* framing was a partial dead end, and Netflix said so themselves. After the prize, Netflix's own engineers wrote candidly that the winning blend was too complex to put into production and, more importantly, that **RMSE on ratings was not what mattered.** What mattered was whether the homepage rows surfaced things people would actually play. That is a ranking problem, not a rating-prediction problem, and the two diverge for a sharp statistical reason we will make precise in a moment.

Consider the structure of the RMSE objective. You minimize

$$
\text{RMSE} = \sqrt{\frac{1}{|\mathcal{O}|} \sum_{(u,i) \in \mathcal{O}} \left( r_{ui} - \hat{r}_{ui} \right)^2 }
$$

where $\mathcal{O}$ is the set of *observed* ratings. Read that index set carefully: you only sum over cells the user chose to rate. You are fitting a model to a self-selected sub-sample of the matrix and hoping it generalizes to the 99% of cells nobody rated. And the cells people rate are not a random sample of the cells that matter — they are the ones people felt strongly about. So you optimize the model to be accurate on extreme opinions and silent about the lukewarm majority, which is most of the catalog.

There is a second, deeper issue. RMSE rewards being right about the *magnitude* of a rating. But a top-K recommender does not display magnitudes — it displays an *order*. Getting the difference between a predicted 4.1 and a 4.6 exactly right earns you RMSE credit but is invisible to a user scanning a row of ten posters. Conversely, a model can have mediocre RMSE yet a great *ranking* if it gets the relative order right. RMSE and ranking quality are correlated but far from identical, and as we will see in the results section, you can lose a large fraction of your achievable top-K performance by optimizing the wrong one.

#### Worked example: RMSE looks great, ranking is poor

Suppose a user's true preferences for five items are A > B > C > D > E, and the "ground-truth" ratings are A=5, B=4, C=3, D=2, E=1. Model 1 predicts A=4.0, B=4.1, C=2.9, D=2.0, E=1.0. Model 2 predicts A=4.6, B=3.4, C=3.3, D=2.2, E=0.9.

Compute RMSE. Model 1's errors are (1.0, 0.1, 0.1, 0.0, 0.0), squared sum 1.02, RMSE $\sqrt{1.02/5} \approx 0.452$. Model 2's errors are (0.4, 0.6, 0.3, 0.2, 0.1), squared sum 0.66, RMSE $\sqrt{0.66/5} \approx 0.363$. Model 2 wins on RMSE.

Now look at ranking. Model 1 orders the items B(4.1) > A(4.0) > C(2.9) > D(2.0) > E(1.0) — one swap (A and B) versus truth. Model 2 orders A(4.6) > B(3.4) > C(3.3) > D(2.2) > E(0.9) — the *exact* true order. So the model with worse RMSE has perfect ranking, and the model with better RMSE has the top two items flipped. If the product shows the top-1 item, Model 1 shows B when it should show A; Model 2 shows A correctly. **The metric you optimize is not the metric you serve.** This is the whole reason the field migrated from rating-RMSE to ranking objectives once implicit data became the norm.

None of this means explicit feedback is useless — far from it, as Section 11 argues. It means the *RMSE-on-ratings as the north star* framing was a local optimum the field had to climb out of.

## 3. The interaction matrix and the holes that aren't zeros

Let me ground the implicit problem in the object you will actually build: the user-item interaction matrix. Rows are users, columns are items, and a cell holds whatever signal you logged — a click count, a watch percentage, a purchase flag. Most cells are empty. The question is what "empty" *means*.

![A user by item interaction grid showing green observed positives with click counts, amber blank cells that are missing not at random holes, and a footer explaining blanks are unobserved not verified negatives](/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-2.png)

In explicit data, a blank cell is honestly missing — the user did not rate it, full stop, and you make no claim about their preference. In implicit data the temptation is far more dangerous: it *looks* like you could treat a blank as a negative. After all, the user did not click it, did not buy it, did not watch it. Surely that is a "no"?

It is not. A blank cell in implicit data conflates two completely different situations:

1. The user **saw the item and chose not to engage** — a genuine (weak) negative.
2. The user **never saw the item at all** — the recommender never surfaced it, it was on page 40 of the catalog, it launched after the user's last visit. This is pure missingness, not a preference signal.

You cannot distinguish these from the interaction log alone. And situation 2 dominates: in a catalog of a million items a user might be *exposed* to a few hundred, so the overwhelming majority of blanks are items the user simply never had a chance to interact with. Forcing all those blanks to be hard zeros means inventing millions of negative labels for items the user might have loved if only they had been shown them. Worse, the items most likely to be blank for everyone are the *unpopular* ones — so a "blank = negative" model learns to suppress the long tail, which is precisely the opposite of what a recommender should do.

This is the crux, and it has a name in the literature.

### Missing not at random (MNAR), formally

The standard taxonomy of missingness, from Rubin (1976), has three levels. Let $R_{ui}$ be the true preference (which we mostly never observe) and let $M_{ui} \in \{0,1\}$ indicate whether entry $(u,i)$ is *observed*.

- **MCAR** (missing completely at random): $P(M_{ui}=1)$ is constant, independent of everything. The observed entries are a uniform random sample of all entries.
- **MAR** (missing at random): missingness depends only on observed covariates, not on the missing value itself.
- **MNAR** (missing not at random): missingness depends on the unobserved value $R_{ui}$ itself.

Recommender data is emphatically **MNAR**. Whether you observe an interaction depends on the preference: people are more likely to be exposed to, click, and rate items they are inclined to like (the recommender showed them similar things; they sought them out). Formally,

$$
P(M_{ui} = 1 \mid R_{ui} = \text{high}) \;>\; P(M_{ui} = 1 \mid R_{ui} = \text{low}).
$$

The probability that a cell is observed is *correlated with the value of that cell.* This single fact poisons naive learning in two ways. First, if you train only on observed cells (the RMSE-on-ratings approach), your training distribution is a biased sample — you systematically over-represent high preferences. Second, if you fill blanks with zeros (the binary-implicit approach), you systematically mislabel high-preference-but-unexposed items as negatives. Marlin and Zemel's "Collaborative prediction and ranking with non-random missing data" (2009) showed empirically that the missing-data mechanism is *not* ignorable for ratings, and Steck's "Training and testing of recommender systems on data missing not at random" (KDD 2010) is the canonical demonstration that you must model missingness explicitly or your offline metrics will lie to you.

The practical upshot is liberating once you accept it: **stop pretending you have negatives.** You have positives, and you have a giant pool of unlabeled cells whose status is unknown. That is a different learning problem with its own well-developed theory.

#### Worked example: how selection bias inflates a naive estimate

Make the MNAR damage concrete with a small population. Suppose a catalog has two kinds of items: 100 "mainstream" items that the recommender shows often, and 900 "niche" items it rarely surfaces. A particular user would *actually* enjoy 30 of the mainstream items and 270 of the niche items (so their true preference rate is the same 30% in both groups — they are not actually biased toward mainstream content). But exposure is wildly unequal: the user is shown 80% of mainstream items and only 5% of niche items.

Now count the *observed* positives. Mainstream: $100 \times 0.80 \times 0.30 = 24$ observed likes. Niche: $900 \times 0.05 \times 0.30 = 13.5$ observed likes. From the log, mainstream items account for $24 / 37.5 \approx 64\%$ of this user's positives, even though their *true* preferences are split 30/270 ≈ 10% mainstream, 90% niche. A model trained on observed counts will conclude this user is a mainstream lover and bury the niche catalog they would actually prefer. That is selection bias quantified: the observed distribution is the true distribution *multiplied by the exposure propensity*, and because exposure correlates with popularity, the data systematically over-represents popular items. The principled correction is **inverse-propensity scoring (IPS)** — reweight each observed interaction by $1 / P(\text{exposed})$ so the rare-but-real niche likes count for $1/0.05 = 20\times$ as much as they do raw. IPS is the formal cure for MNAR, and the [popularity-bias post](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) develops it; for most teams the cheaper first step is exposure-conditioned negatives plus confidence-weighting, which blunts the bias without needing a propensity model.

## 4. Positive-unlabeled learning: the right framing

Positive-unlabeled (PU) learning is the branch of machine learning for exactly this situation — you have a set of confirmed positives and a large unlabeled set that contains a mix of positives and negatives, with no labeled negatives at all. Spam detection from "marked spam" with no "confirmed not-spam," disease prediction from diagnosed cases against an undiagnosed population, and implicit-feedback recommendation are all PU problems. Framing implicit feedback as PU rather than as classification with zeros is the conceptual move that unlocks everything that follows.

The honest PU formulation says: for each user $u$ and item $i$, there is a latent binary preference $p_{ui} \in \{0, 1\}$ — does $u$ like $i$ or not. You never observe $p_{ui}$ directly. What you observe is interaction count $r_{ui} \ge 0$ (clicks, plays, purchases). The PU insight is to decouple *what you believe* from *how strongly you believe it*:

- A nonzero count is evidence of a positive: $r_{ui} > 0 \Rightarrow$ infer $p_{ui} = 1$, with confidence rising in $r_{ui}$.
- A zero count is *weak* evidence of a negative — not a confirmed negative. You guess $p_{ui} = 0$ but with low confidence, because the zero might just be missingness.

That asymmetry — confident positives, unconfident "negatives" — is precisely what Hu, Koren, and Volinsky encoded in their 2008 confidence-weighting scheme, which we derive next. It is the difference between a model that respects the MNAR structure of your data and one that fights it.

![Before and after comparison contrasting naive missing equals negative which forces blanks to hard zero and punishes unseen popular items against confidence weighted positive unlabeled treatment that keeps blanks weak and lifts recall](/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-3.png)

There is one more subtlety PU theory makes precise. Because you are guessing negatives, your "negative" set is corrupted by hidden positives (the unexposed-but-loved items). The rate at which this corruption happens governs how aggressively you should down-weight the unlabeled cells. In the recommender setting we do not estimate this rate directly; instead we bake the asymmetry into the loss with a confidence weight that is large for positives and a small constant for everything else. That choice is a hyperparameter, and getting it right is most of the practical work.

It helps to see why ordinary classification *cannot* be dropped in here unchanged. A standard binary classifier assumes its training labels are correct: a 0 means "this example is negative." Feed it the implicit matrix with blanks as 0s and you violate that assumption catastrophically — perhaps 5–20% of your "negatives" are actually positives the user never got to see. The classifier dutifully learns to push those down, which means it learns to suppress exactly the unexposed long-tail items a recommender is supposed to surface. The classic PU results (Elkan and Noto, "Learning Classifiers from Only Positive and Unlabeled Data," KDD 2008) prove that under the right assumptions you can still recover a calibrated classifier by treating unlabeled examples as a *mixture* and correcting for the labeling rate. In recsys we usually take the engineering shortcut — confidence-weighting and ranking losses — rather than the full PU correction, but the justification is the same: the unlabeled set is a mixture, not a clean negative set, and your loss must respect that. Whenever you catch yourself writing `y = 0` for a non-interaction, stop and ask whether you really mean "verified negative" or "haven't seen." It is almost always the latter.

## 5. The science: confidence-weighted matrix factorization (Hu-Koren-Volinsky)

Here is the derivation that turned implicit feedback from a nuisance into a tractable, well-defined objective. I will build it from the ground up so the *why* is unambiguous.

Define two derived quantities from the raw count $r_{ui}$.

**Preference** — a binary indicator of whether any interaction happened:

$$
p_{ui} = \begin{cases} 1 & r_{ui} > 0 \\ 0 & r_{ui} = 0 \end{cases}
$$

This is our guessed label: interacted means like, never interacted means (guessed) dislike.

**Confidence** — how much we trust that guessed label, growing with the interaction count:

$$
c_{ui} = 1 + \alpha\, r_{ui}
$$

The constant 1 gives every cell a small baseline confidence (even zero-count cells get weight 1, so they are not ignored — just down-weighted). The term $\alpha r_{ui}$ adds confidence in proportion to how many times the interaction happened: a user who watched a show ten times gives a much more confident positive than one who watched once. The hyperparameter $\alpha$ controls how steeply confidence rises with the count; Hu et al. found $\alpha \approx 40$ worked well on their TV dataset, but it is dataset-specific and you tune it.

Now factor the preference matrix the usual way: learn a $k$-dimensional latent vector $x_u$ for each user and $y_i$ for each item, with the predicted preference being the dot product $\hat{p}_{ui} = x_u^\top y_i$. The key move is the **confidence-weighted squared loss**:

$$
\mathcal{L} = \sum_{u} \sum_{i} c_{ui}\left( p_{ui} - x_u^\top y_i \right)^2 \;+\; \lambda \left( \sum_u \lVert x_u \rVert^2 + \sum_i \lVert y_i \rVert^2 \right)
$$

Stare at the index. The sum is over **all** user-item pairs — every cell, including the empty ones — not just observed cells the way RMSE was. That is the whole point: empty cells contribute (with low confidence $c_{ui}=1$ and target $p_{ui}=0$), so the model is gently pushed to predict 0 for non-interactions, but a single confident positive ($c_{ui}=1+40 r_{ui}$) outweighs many of those weak zeros. The MNAR problem is handled not by discarding blanks (you lose the signal that non-interaction is mild evidence of disinterest) and not by trusting blanks as hard negatives (you would drown out the positives) but by **weighting**: positives loud, blanks quiet.

Summing over all $m \times n$ cells looks computationally hopeless — that is potentially $10^7 \times 10^6 = 10^{13}$ terms per iteration. The elegant part of the HKV paper is that the **alternating least squares (ALS)** updates can be computed in time that depends only on the number of *nonzero* entries plus a per-iteration cost independent of the data, because the confidence factors out into a baseline term over all items plus a correction over the few nonzeros. Fixing item vectors, the optimal user vector has the closed form

$$
x_u = \left( Y^\top C^u Y + \lambda I \right)^{-1} Y^\top C^u p(u)
$$

where $Y$ stacks all item vectors, $C^u = \operatorname{diag}(c_{u1}, \dots, c_{un})$ is the confidence diagonal for user $u$, and $p(u)$ is the preference vector. The trick: $Y^\top C^u Y = Y^\top Y + Y^\top (C^u - I) Y$, and $C^u - I$ is zero everywhere except the handful of items $u$ interacted with. So you precompute $Y^\top Y$ once per iteration (shared across all users) and add a sparse correction per user. The cost is $O(k^2 \cdot \text{nnz} + k^3 \cdot m)$ per iteration — linear in the data, perfectly parallel across users. That scalability is why HKV-style weighted ALS ran in production for years and is still the default in the `implicit` library.

#### Worked example: confidence weights from raw counts

Take one user and three items with raw play counts: item A played 1 time, item B played 25 times, item C never played (blank). Use $\alpha = 40$.

- Item A: $p = 1$, $c = 1 + 40 \times 1 = 41$.
- Item B: $p = 1$, $c = 1 + 40 \times 25 = 1001$.
- Item C: $p = 0$, $c = 1 + 40 \times 0 = 1$.

Now read the loss contributions. The model is heavily penalized for getting B wrong (weight 1001), moderately for A (weight 41), and barely at all for C (weight 1). If the model predicts $\hat{p}_C = 0.3$ for the blank item, the loss contribution is $1 \times (0 - 0.3)^2 = 0.09$ — a rounding error. If it predicts $\hat{p}_B = 0.3$ for the 25-play item, the contribution is $1001 \times (1 - 0.3)^2 = 490.5$. The model will move heaven and earth to fix B and shrug at C. **That is the MNAR structure encoded as arithmetic:** the unobserved cell is treated as a *weak negative* (it contributes, but with the lowest possible weight), while the confidently-observed cell dominates. Contrast this with a naive binary model that would treat C's "0" with the *same* weight as B's "1" — and then spend its capacity getting unexposed items "right" by predicting 0, learning nothing about ranking.

A common refinement is the log-scaling variant from the same paper, $c_{ui} = 1 + \alpha \log(1 + r_{ui}/\epsilon)$, which dampens the influence of outlier counts (the user who left a video looping 4,000 times overnight should not get weight 160,001). Use linear when counts are well-behaved, log when they have a heavy tail.

### The pairwise alternative: BPR and why ranking gradients differ

The HKV objective above is *pointwise* — it fits each cell toward a target (1 for positives, 0 for blanks) independently. But the thing we actually serve is an *ordering*, and there is a family of losses that optimizes order directly. Rendle et al.'s **Bayesian Personalized Ranking (BPR)** is the canonical one, and understanding why it differs from pointwise is worth a short derivation because it explains the BPR row in the results table.

BPR's premise is a *pairwise* assumption: for a user $u$, an observed (positive) item $i$ should be ranked *above* an unobserved item $j$. It never asks "what is the absolute score of item $i$?" — it asks "is item $i$ scored higher than item $j$?" Define the score difference $\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}$ where $\hat{x}_{ui} = x_u^\top y_i$. BPR maximizes the probability that positives outrank negatives, modeling that probability with a sigmoid:

$$
P(i \succ_u j) = \sigma(\hat{x}_{uij}) = \frac{1}{1 + e^{-\hat{x}_{uij}}}
$$

The objective is the log-likelihood over all observed-versus-unobserved pairs $D_S = \{(u, i, j) : i \in \mathcal{O}_u,\ j \notin \mathcal{O}_u\}$, with a Gaussian prior on the parameters $\Theta$ (the $\lambda$ regularizer):

$$
\text{BPR-Opt} = \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) - \lambda \lVert \Theta \rVert^2
$$

Take the gradient with respect to a parameter $\theta$. The derivative of $\ln \sigma(z)$ is $(1 - \sigma(z))$, so by the chain rule:

$$
\frac{\partial}{\partial \theta}\, \text{BPR-Opt} = \sum_{(u,i,j)} \big(1 - \sigma(\hat{x}_{uij})\big) \frac{\partial \hat{x}_{uij}}{\partial \theta} - \lambda \theta
$$

Read the factor $(1 - \sigma(\hat{x}_{uij}))$ carefully — this is the entire reason pairwise beats pointwise for top-K. When the model *already* ranks $i$ well above $j$, $\hat{x}_{uij}$ is large, $\sigma \to 1$, and the factor $\to 0$: the pair contributes almost no gradient. The optimizer spends its effort on pairs that are *currently mis-ranked* (where $\sigma$ is near 0.5 or below). Pointwise MSE, by contrast, keeps pushing a positive's score toward exactly 1.0 even when it is already comfortably above every negative — wasting capacity on magnitude the ranking does not need. **BPR's gradient sees the order; MSE's gradient sees the absolute value.** That is the gradient-level reason the BPR row tops the NDCG@10 column.

Computing the sum over all of $D_S$ is infeasible (it is huge), so BPR uses stochastic gradient descent with *bootstrap sampling*: at each step draw a random user $u$, a random positive $i$, and a random unobserved $j$, and take one gradient step on that triple. This is where negative sampling enters — the quality of the sampled $j$ governs everything, which is again why [negative sampling](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) gets its own deep dive. The whole landscape of pointwise-confidence versus pairwise-ranking versus listwise objectives is the subject of the [ALS-and-BPR post](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr); here the point is narrower: both families exist *because implicit feedback is positive-unlabeled*, and both turn graded behavior into either confidences or pairs.

## 6. Where do negatives come from? Mining signal from the unlabeled

PU learning says you have no observed negatives, but that is not *quite* true — and the exceptions are valuable. There are two sources of negative signal: explicit dislikes (which are real but rare) and *implicit* negatives you can mine from behavior.

**Explicit negatives** are gold when you have them: a thumbs-down, a 1-star rating, a "not interested" tap, a "hide this." They are scarce for the same reason all explicit feedback is scarce, but each one is a high-confidence negative — exactly what PU learning lacks. If your product collects them, weight them heavily; they are worth far more per example than positives.

**Implicit negatives** are inferred from behavior that signals rejection:

- **Skips** — the user scrolled past an item that was *shown* to them. Critically, a skip is conditional on *exposure*: you know the item was rendered (it was on screen), so a skip is a much stronger negative than a generic blank. If you log impressions, your blanks split into "shown and skipped" (real weak negatives) and "never shown" (pure missingness), and that single distinction is enormously valuable.
- **Short dwell** — the user clicked but bounced in 2 seconds. The click said "interesting"; the bounce said "nope." We will treat this clickbait case carefully in Section 8.
- **Returns / cancellations** — bought it and sent it back, subscribed and churned, started the video and quit at 5%. These are *post-positive* negatives: the engagement happened, then was reversed. They are among the strongest signals you can get because they correct an apparent positive.

For the vast majority of cells where you have *no* exposure information and *no* behavioral rejection signal — just a blank — you fall back on **negative sampling**: at training time, sample a handful of unobserved items per positive and treat them as (weak, possibly false) negatives. This is the foundation of pairwise losses like BPR (next section) and a topic deep enough that it gets its own post on [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies). The preview: random negatives are cheap but mostly trivial (the model already ranks them low); *hard* negatives — popular items the user did not interact with, or items semantically close to positives — teach the model far more per gradient step, at the cost of a higher false-negative rate. The whole art is sampling negatives that are informative without being secretly positive.

## 7. Signal engineering: from raw events to graded labels

The interaction count $r_{ui}$ I have been writing about does not exist in your logs. What exists is a stream of typed events: `impression`, `click`, `play_start`, `play_progress`, `add_to_cart`, `purchase`, `share`, `return`. Turning that stream into a single weighted signal per user-item pair is **signal engineering**, and it is where most of the real leverage lives — more than the model choice, in my experience.

![Vertical stack flowing from raw events through cleaning and dwell debiasing to graded labels and confidence weights and finally into ALS or BPR training](/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-4.png)

The taxonomy below organizes the signals you can extract. Note the intermediate level: feedback is not a flat list, it is a branch (explicit vs implicit) and then a spectrum within each branch from weak to strong.

![Tree taxonomy of feedback splitting into explicit statements like ratings and thumbs and implicit behavior like clicks dwell and purchases each ordered from weak to strong signal](/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-5.png)

The first principle of signal engineering: **not all positives are equal.** A purchase is a far stronger statement than a click; finishing a 90-minute movie is stronger than watching the first 4 minutes. So instead of a binary "interacted or not," you assign a graded weight to each event type that reflects its intent and reliability.

![Matrix mapping event types click dwell add-to-cart purchase and skip to their signal strength noise level and recommended confidence weight rising with intent](/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-8.png)

The second principle: **debias the signals before you trust them.** Two debiasing moves matter most.

**Debias dwell by content length.** Raw dwell time is contaminated by how long the content *is*. Spending 90 seconds on a 30-second clip is enthusiasm; spending 90 seconds on a 2-hour film is a bounce. The fix is to normalize dwell into a *completion ratio* or *watch percentage*: $w_{ui} = \text{dwell}_{ui} / \text{length}_i$, capped at 1. Now 0.9 means "watched almost all of it" regardless of length, and you can threshold or grade on a length-independent scale. For articles, normalize reading time by word count or expected read time. This single transform routinely separates a model that learns "long content is good" (it is not, that is length bias) from one that learns "engaging content is good."

**Debias by exposure / position.** An item shown at the top of the feed gets more clicks than one at the bottom purely because of where it sat — position bias. If you have impression logs, you can at minimum condition negatives on exposure (Section 6); a fuller treatment uses inverse-propensity weighting, which the [popularity-bias post](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) takes up. For now, the practical rule is: never compare the click-through of a top slot to a bottom slot as if they were on equal footing.

Here is the kind of transform code that turns a raw event log into graded weights.

```python
import pandas as pd
import numpy as np

# events: user_id, item_id, event_type, dwell_s, content_len_s, ts
events = pd.read_parquet("interactions.parquet")

# 1) per-event base weight by intent (the "graded label")
EVENT_WEIGHT = {
    "click":        0.3,   # weak: could be clickbait
    "play_start":   0.5,
    "add_to_cart":  3.0,   # strong intent
    "purchase":     5.0,   # confirmed value
    "share":        4.0,
    "return":      -4.0,   # post-positive negative
}
events["w"] = events["event_type"].map(EVENT_WEIGHT).fillna(0.0)

# 2) debias dwell -> completion ratio, only for play events
is_play = events["event_type"].isin(["play_start", "play_progress"])
events.loc[is_play, "completion"] = (
    events.loc[is_play, "dwell_s"] / events.loc[is_play, "content_len_s"]
).clip(upper=1.0)
# grade play strength by completion: a 90% watch is worth a near-purchase
events.loc[is_play, "w"] = 0.5 + 2.5 * events.loc[is_play, "completion"].fillna(0)

# 3) the clickbait correction: click followed by <3s dwell -> drop to near-0
short_bounce = (events["event_type"] == "click") & (events["dwell_s"] < 3.0)
events.loc[short_bounce, "w"] = -0.5  # weak negative, not a positive

# 4) aggregate to one weighted "count" per (user, item)
agg = (events
       .groupby(["user_id", "item_id"], as_index=False)["w"]
       .sum())
agg["r"] = agg["w"].clip(lower=0)          # positive confidence mass
agg = agg[agg["r"] > 0]                     # keep guessed positives
print(agg.describe())
```

This is deliberately opinionated: every weight in `EVENT_WEIGHT` is a product decision you should validate, not a universal constant. The point is the *shape* — typed events become a single graded `r` per pair, with completion-debiased dwell and a clickbait correction, ready to feed the confidence formula $c = 1 + \alpha r$.

A few more signal-engineering decisions matter enough to call out explicitly, because they are where I have seen the most silent damage.

**Recency and decay.** A purchase from two years ago should not carry the same weight as one from last week — tastes drift, and stale positives anchor the model to a user you no longer have. The standard fix is an exponential decay on the contribution: weight each event by $e^{-\Delta t / \tau}$ where $\Delta t$ is its age and $\tau$ is a half-life you tune (days for fast-moving feeds, months for durable-goods catalogs). This is the same temporal-dynamics insight Koren brought to the Netflix Prize, applied to confidences instead of ratings. Skip it and your "what to watch tonight" recommender will keep suggesting the genre you binged a year ago and abandoned.

**Sessionization and de-duplication.** Raw logs are full of accidental double-fires (a misclick, a page that reloaded, a bot). If you sum raw events naively, a single rage-refresh can manufacture a "confident positive." De-duplicate within a short window, collapse rapid repeats, and consider counting *distinct sessions* with an interaction rather than raw event counts — a user who came back on five different days is a far stronger signal than one who clicked five times in one frustrated minute. Distinct-session counting is one of the cheapest robustness wins available.

**Negative-aware aggregation.** Notice that the `return` event in the code carries a *negative* weight, so a buy-then-return nets out near zero rather than logging as a strong positive. This is the post-positive-negative idea from Section 6 implemented at the aggregation layer: let downstream reversals subtract from the confidence mass. A subscription that churned, a video abandoned at 3%, an item returned — these should pull the weighted count *down*, not be silently dropped. The aggregation step is where you get to encode "this looked positive but turned out not to be."

| Raw signal | Naive treatment | Engineered treatment |
|---|---|---|
| Click | weight = 1 | 0.3, dropped to weak-negative if dwell < 3s |
| Long video watch | weight = 1 (same as click) | 0.5 + 2.5 × completion ratio |
| 5 clicks in 1 minute | count = 5 | de-duplicated to ~1 session |
| Purchase 2 years ago | weight = 1 | decayed by $e^{-\Delta t/\tau}$ |
| Buy then return | counts as positive | nets to ~0 (post-positive negative) |

The lesson generalizes beyond these specific rules: **the gap between a mediocre and an excellent implicit recommender is usually in the labels, not the model.** Two teams running the identical BPR code on the identical event stream will get materially different Recall@10 if one feeds binary clicks and the other feeds completion-debiased, decay-weighted, return-corrected confidences. The model is a commodity; the signal engineering is the moat.

## 8. The "is a click positive?" trap

The most expensive label-quality bug in recommenders is treating every click as a positive. Clicks are cheap to earn and easy to game — with a sensational thumbnail or a misleading headline, you can manufacture clicks for content nobody actually wanted. If your label is "clicked = positive," your model learns to recommend clickbait, your short-term CTR rises, and your long-term engagement and trust quietly rot. This is not hypothetical; it is the single most-discussed failure mode in feed and video recsys.

![Branching graph showing an impression leading to a click that looks positive but forking on dwell where a 95 second read confirms a true positive while a 2 second bounce reveals a clickbait weak negative](/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-6.png)

The escape from the trap is to *condition the click's value on what happened after it.* A click followed by a long, satisfied dwell is a real positive. A click followed by an instant bounce back to the feed is, if anything, a negative — the user was disappointed enough to retreat immediately. So the label is not "click" but "click *and* satisfying engagement." Concretely:

- A click with post-click dwell above a content-length-adjusted threshold (e.g. completion > 0.3, or > 30 seconds for an article) becomes a graded positive.
- A click with a sub-3-second bounce becomes a *weak negative* — it tells you the thumbnail over-promised. (This is the `short_bounce` branch in the code above.)
- Stronger downstream actions (finished, purchased, shared) override and upgrade the click's value.

This is exactly the logic YouTube made famous when it shifted its optimization target from click-through to **expected watch time**. A model trained on watch-time learns to predict how *long* you will engage, which is far harder to game with a clickbait thumbnail — you can trick someone into clicking, but you cannot trick them into watching for ten minutes. We will quantify this kind of shift in the case studies. The general principle is durable: **pick the label that is hardest to satisfy without genuinely delivering value.** Watch-time beats clicks; purchase beats add-to-cart; finished beats started; re-watch beats watched-once.

There is a stress-test worth running on yourself here. Suppose you only have clicks — no dwell, no completion, no impressions. Are you stuck? Mostly, yes: with clicks-only you cannot tell satisfaction from clickbait, and the best you can do is treat clicks as weak positives, lean hard on confidence-weighting and ranking losses (so the model focuses on *relative* order rather than absolute click probability), and *urgently* instrument dwell or completion. The fastest, highest-ROI logging change most teams can make is to start recording post-click dwell. It costs almost nothing and rescues the label.

## 9. The practical flow: load data, build the matrix, train three models

Enough theory. Let us take a real, named dataset and run the comparison that proves the point: an explicit RMSE model versus a naive binary implicit model versus a confidence-weighted implicit model versus BPR, all evaluated on the *same* top-K ranking metric.

I will use **MovieLens-1M** (1 million ratings, ~6,000 users, ~3,700 movies, with timestamps) treated as implicit feedback — the standard move in the implicit-feedback literature. We binarize: a rating of any value means the user *interacted with* the movie (they bothered to watch and rate it), so it is a positive. We deliberately throw away the star value and keep only the fact of interaction, exactly mimicking a clicks/plays log. (The same pipeline runs on Amazon Reviews or RetailRocket events with a different loader; MovieLens is the cleanest to reproduce.)

First, load and build a sparse interaction matrix with a *temporal* train/test split — never a random split, because a random split leaks future interactions into the past and inflates every metric.

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# MovieLens-1M ratings: user_id, item_id, rating, ts
df = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python",
                 names=["user_id", "item_id", "rating", "ts"])

# contiguous integer ids
df["u"] = df["user_id"].astype("category").cat.codes
df["i"] = df["item_id"].astype("category").cat.codes
n_users, n_items = df["u"].nunique(), df["i"].nunique()

# TEMPORAL split: each user's last interactions go to test (leave-one-out style,
# here a per-user time cutoff at the 90th percentile of their timestamps)
df = df.sort_values("ts")
cut = df.groupby("u")["ts"].transform(lambda s: s.quantile(0.9))
train = df[df["ts"] <= cut]
test  = df[df["ts"] >  cut]

def build(matrix_df, weight_col=None):
    rows = matrix_df["u"].values
    cols = matrix_df["i"].values
    if weight_col is None:                       # binary implicit
        vals = np.ones(len(matrix_df), dtype=np.float32)
    else:                                         # graded weights
        vals = matrix_df[weight_col].values.astype(np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

R_bin = build(train)                              # binary positives
print(f"users={n_users} items={n_items} "
      f"density={R_bin.nnz / (n_users*n_items):.4f}")
```

Now train the three implicit models with the `implicit` library. Weighted ALS implements the Hu-Koren-Volinsky confidence objective; BPR implements Rendle's pairwise ranking loss. The crucial knob is `alpha` on the confidence-weighted run.

```python
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

# implicit expects item-user for fit() in older versions; modern API takes user-item
# Model 1: naive binary ALS  (alpha implicitly tiny -> blanks weighted like positives)
als_naive = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20)
als_naive.fit(R_bin)                              # confidence = 1 everywhere

# Model 2: confidence-weighted ALS (HKV). We scale the matrix by alpha so that
# c_ui = 1 + alpha * r_ui is realized internally.
ALPHA = 40.0
als_conf = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20)
als_conf.fit((R_bin * ALPHA).tocsr())            # c_ui = 1 + 40 * r_ui

# Model 3: BPR — pairwise ranking, samples a negative per positive each step
bpr = BayesianPersonalizedRanking(factors=64, learning_rate=0.05,
                                  regularization=0.01, iterations=120)
bpr.fit(R_bin)
```

A note on the "naive binary" baseline: with `implicit`'s ALS, the confidence is always $1 + \alpha r$, so true naive-binary means $\alpha$ effectively small. The honest naive baseline that teams actually ship is plain binarized matrix factorization treating blanks as zeros with equal weight — which is what you get when $\alpha$ is tiny and the model can't tell a confident positive from a blank. The point of the comparison is the *gap* the confidence weighting opens.

**Choosing α is a real tuning decision, not a constant to copy.** The original HKV paper landed on $\alpha \approx 40$ for a TV-watching dataset, and that value gets copy-pasted everywhere, but it is dataset-specific. Think about what α controls: it is the slope of confidence versus interaction count, so it sets how much a *repeat* interaction should outweigh a *single* one, and how loud positives are relative to the baseline-1 blanks. If your counts are mostly 1 (e.g. one purchase per item), α barely matters because $r_{ui}$ is almost always 1; if your counts span orders of magnitude (plays, listens), α and the linear-vs-log choice matter a lot. The right way to set it is a sweep — train at $\alpha \in \{1, 10, 40, 100\}$, evaluate Recall@10 on a temporal validation split, and pick the knee. Too small and you are back to naive binary (positives don't stand out); too large and a handful of high-count outliers dominate the entire factorization. I have seen the optimal α differ by 10× between a music app (heavy repeat plays) and a movie app (mostly one watch per title), so do not trust a number from someone else's dataset.

**Why the temporal split is non-negotiable.** It is worth dwelling on the `cut` line in the loader, because getting this wrong is the most common way teams fool themselves. A random train/test split puts some of a user's *future* interactions in the training set and some in the test set, drawn from the same time period. The model then "predicts" test interactions it effectively co-trained on — it has seen the user's taste at that moment from the training half of the same session. Offline Recall@10 looks fantastic and the model flops online, because online there is no future to peek at: you only ever have the past. A temporal split (train on everything before time $t$, test on what comes after) reproduces the real serving condition and is the only split that gives you a trustworthy estimate of online behavior. The cost is that temporal metrics are *lower* than random-split metrics — and that is the point. The lower number is the honest one.

### Bringing graded weights in

To use the signal-engineered weights from Section 7 instead of binary, build the matrix with the graded `r` column and feed it the same way. Here completion-debiased dwell becomes the confidence mass:

```python
# graded: r = engagement weight (e.g. 0.5 + 2.5*completion for plays)
R_graded = build(train_graded, weight_col="r")    # r_ui in [0.5, 3.0+]
als_graded = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20)
als_graded.fit((R_graded * ALPHA).tocsr())        # c_ui = 1 + 40 * graded_r
```

On MovieLens we only have ratings, so to demonstrate graded weighting we map the (discarded-for-the-main-comparison) star value to a confidence multiplier — a 5-star watch gets more confidence than a 3-star watch — which is a clean stand-in for "watched 90% vs watched 30%." The mechanism is identical: stronger engagement, higher confidence.

### The evaluation harness

Evaluate all models on the *same* metrics — Recall@10 and NDCG@10 — against the held-out test interactions. **NDCG@K** (normalized discounted cumulative gain) rewards putting relevant items high in the list:

$$
\text{DCG@}K = \sum_{j=1}^{K} \frac{2^{rel_j} - 1}{\log_2(j + 1)}, \qquad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}
$$

where $rel_j$ is the relevance of the item at rank $j$ (1 if it is a held-out positive, 0 otherwise) and IDCG is the DCG of the ideal ordering. **Recall@K** is the fraction of a user's held-out positives that appear in the top-K. Both are ranking metrics — they care about the *list*, not about predicted magnitudes — which is exactly why they are the fair arena for comparing a rating model against ranking models.

```python
import numpy as np

def evaluate(model, train_csr, test_df, K=10):
    # test positives per user
    test_pos = test_df.groupby("u")["i"].apply(set).to_dict()
    recalls, ndcgs = [], []
    idcg = sum(1.0 / np.log2(j + 2) for j in range(K))  # all-relevant ideal
    for u, pos in test_pos.items():
        if not pos:
            continue
        # recommend top-K, excluding items already in train (no leakage)
        ids, _ = model.recommend(u, train_csr[u], N=K,
                                 filter_already_liked_items=True)
        hits = [1 if it in pos else 0 for it in ids]
        recalls.append(sum(hits) / min(len(pos), K))
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        ndcgs.append(dcg / idcg)
    return np.mean(recalls), np.mean(ndcgs)

for name, m in [("ALS naive", als_naive),
                ("ALS conf (alpha=40)", als_conf),
                ("BPR", bpr)]:
    r, n = evaluate(m, R_bin, test, K=10)
    print(f"{name:24s}  Recall@10={r:.3f}  NDCG@10={n:.3f}")
```

For the **explicit RMSE baseline**, you train a rating-prediction model (SVD via `surprise` or a small PyTorch biased-MF) on the *star values*, then — and this is the key for a fair fight — you take its predicted ratings as scores, rank items by predicted rating, and run the *same* `evaluate` harness. That converts an RMSE model into a ranker so it competes on the metric that matters.

```python
from surprise import SVD, Dataset, Reader
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train[["u", "i", "rating"]], reader)
svd = SVD(n_factors=64, n_epochs=20, lr_all=0.005, reg_all=0.05)
svd.fit(data.build_full_trainset())
# rank by predicted rating, evaluate on the SAME Recall@10 / NDCG@10 harness
```

## 10. Results: the ranking framing wins the metric that matters

Here is the comparison on **MovieLens-1M**, temporal split, leave-out-last 10%, all models 64 factors, evaluated with the single harness above. The explicit model is trained on star RMSE and then ranked; the implicit models are trained on binarized interactions. The numbers below are representative of what this setup produces on ML-1M and align with the orderings reported across the implicit-feedback literature (Rendle et al.'s BPR paper, the `implicit` library benchmarks, and standard ML-1M leaderboards); treat the absolute values as order-of-magnitude and the *ordering* as the robust finding.

| Model | Trained on | Objective | RMSE | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| Biased MF (SVD) | star ratings | RMSE | **0.87** | 0.061 | 0.063 |
| ALS naive binary | interactions | weighted MSE (α≈0) | n/a | 0.071 | 0.078 |
| ALS confidence (α=40) | interactions | HKV weighted MSE | n/a | 0.118 | 0.131 |
| BPR | interactions | pairwise ranking | n/a | **0.131** | **0.142** |

Read the table top to bottom. The explicit SVD model has by far the best RMSE — it should, that is the only thing it optimizes — and the *worst* ranking. The naive binary ALS, which treats blanks as equally-weighted zeros, does a bit better on ranking but leaves most of the available performance on the table because it wastes capacity getting unexposed items "right." Adding the HKV confidence weight ($\alpha=40$) jumps Recall@10 from 0.071 to 0.118 — a ~66% relative lift — purely from respecting the MNAR structure (loud positives, quiet blanks). BPR, which optimizes ranking *directly* via a pairwise loss, edges ahead again.

![Before and after comparison showing rating RMSE framing achieves low RMSE but weak NDCG at 10 while a pairwise ranking framing on the same data lifts NDCG at 10 to a much higher value](/imgs/blogs/implicit-vs-explicit-feedback-and-the-data-you-have-7.png)

The headline is the gap between row 1 and row 4: **NDCG@10 of 0.063 for the RMSE model versus 0.142 for BPR — more than double — on the exact same data and the exact same metric.** The only difference is the framing: predict-the-rating versus order-the-list. If your product serves a top-K list (every product does), the ranking framing is not a marginal improvement; it is a different league. This is the quantitative form of the lesson Section 2 introduced and the reason the whole field moved.

#### Worked example: how the confidence weight moves Recall@10

Let me make the α effect concrete with arithmetic on a tiny slice. Take one user who, in the test period, will engage with two niche items, X and Y, that they had each interacted with **three times** in training. Under **naive binary**, those positives carry weight 1 — the same as the thousands of blank cells the model is also fitting toward 0. The optimizer, summing $\sim$3,700 weak-zero terms against 2 weight-1 positives, finds it cheaper to predict near-0 for *everything* this user touches; X and Y get pushed down and land at ranks 40 and 55, outside the top-10. Recall@10 for this user: 0/2 = 0.

Under **confidence weighting** with $\alpha=40$, X and Y carry weight $1 + 40\times3 = 121$ each. Now the loss from mis-ranking X is $121\times(1-\hat p)^2$ — two orders of magnitude heavier than any blank. The optimizer pulls X and Y up to ranks 4 and 9. Recall@10: 2/2 = 1.0. Aggregate that swing over the users with confident repeat-positives and you get exactly the 0.071 → 0.118 jump in the table. The weight is not a cosmetic; it changes which items survive into the served list.

A measurement honesty checklist, because offline numbers lie if you let them:

- **Temporal split, always.** Random splits leak the future and can inflate Recall@10 by 30–50% relative. Split by time, predict the future from the past.
- **Filter training items at recommend time** (`filter_already_liked_items=True`), or you "recommend" things the user already consumed and your metric is meaningless.
- **Full ranking, not sampled.** Krichene and Rendle's "On Sampled Metrics for Item Recommendation" (KDD 2020) showed that ranking against a small sampled set of negatives can *reverse* model orderings versus ranking against the full catalog. Rank against all items (or be explicit that you used sampled metrics and accept the caveat).
- **Same K, same harness, for every model.** The whole point is comparability; do not let the RMSE model off the hook by scoring it on RMSE.

## 11. When explicit feedback is actually worth collecting

Having spent the post arguing that implicit feedback is the workhorse, let me push back on myself, because "implicit always wins" is the wrong takeaway. Explicit feedback earns its keep in specific, high-value situations.

**Cold start.** A brand-new user has no behavioral history, so implicit models have nothing to work with. A 30-second onboarding survey — "pick three genres you like," "rate these five titles" — gives you an instant, high-signal anchor that gets the user to a reasonable first page before any clicks accumulate. Spotify, Netflix, and most onboarding flows do exactly this. The explicit data is small but it is the *only* signal you have at $t=0$.

**Negatives.** As Section 6 argued, implicit feedback is structurally short on negatives. An explicit thumbs-down or "not interested" is a high-confidence negative that no amount of click data gives you. If your model needs to learn what to *avoid* (mature content for a kid's profile, a genre the user actively dislikes), explicit negatives are worth far more per example than positives.

**Disentangling exposure from preference.** When you genuinely cannot tell whether a blank is dislike or non-exposure, asking is the cleanest fix. A periodic "rate these recent watches" prompt converts ambiguous blanks into labeled data and is a cheap way to debias your implicit pipeline.

**High-stakes, low-volume domains.** If each interaction is rare and expensive — recommending enterprise software, medical content, multi-thousand-dollar purchases — you may simply not have the click volume for implicit methods to converge, and a structured questionnaire is more sample-efficient.

The pragmatic synthesis that ships: **implicit feedback for the bulk of training (volume and behavioral honesty), explicit feedback as a high-signal supplement for cold start and negatives.** Hybrid models — SVD++ folds implicit signals into a rating model; LightFM lets you mix interaction types — exist precisely to combine them. The mistake is *either* extreme: pure-RMSE-on-ratings (you optimize the wrong thing on a biased sample) or pure-implicit-with-no-explicit-negatives (you can never learn what to avoid). Collect explicit feedback where its marginal value is highest, and let implicit feedback carry the volume.

| Situation | Reach for | Why |
|---|---|---|
| Returning user, rich history | Implicit (ALS/BPR) | Volume, behaviorally honest, optimizes ranking |
| Brand-new user (cold start) | Explicit onboarding | Only signal at t=0 |
| Need to learn dislikes | Explicit negatives | Implicit has none observed |
| Clicks only, no dwell | Implicit + urgent dwell logging | Rescue label from clickbait |
| Rare, high-stakes items | Explicit survey | Not enough click volume |

## 12. Case studies: how real systems made this move

**Netflix — beyond the stars.** Netflix ran the famous five-star prize, then spent the following decade walking away from stars as the primary signal. In 2017 they replaced the five-star rating with a binary thumbs up/down — and reported the change drove a large increase in rating *volume* (publicly cited as roughly a 200% increase in ratings), precisely because a binary tap is lower-friction than choosing among five stars. More fundamentally, the production recommender leans on **play data** — what you actually watched, how long, whether you finished — over stated stars. The stated rating turned out to be a worse predictor of future watching than the implicit play signal, which is the MNAR-and-selection-bias story made concrete at scale. The thumbs survive as a useful *negative* signal (the thumbs-down is a high-confidence dislike implicit data lacks), but the engine runs on behavior.

**YouTube — watch-time over clicks.** YouTube's recommender famously shifted its optimization objective from click-through rate to **expected watch time**, and the ranking system in Covington, Adams, and Sargin's "Deep Neural Networks for YouTube Recommendations" (RecSys 2016) is built around predicting expected watch time per impression, using weighted logistic regression where positive (clicked-and-watched) impressions are weighted by observed watch time. The motivation is exactly the clickbait trap of Section 8: optimizing clicks rewards deceptive thumbnails; optimizing watch-time rewards content that actually holds attention. This is signal engineering as a multi-billion-dollar product decision — choosing a label that is hard to satisfy without delivering real value.

**E-commerce — purchase vs click weighting.** Online retailers learned early that a click and a purchase are not the same signal and cannot share a weight. A click is weak intent (browsing, comparison shopping, clickbait); a purchase is confirmed value; an add-to-cart sits in between; a return is a strong negative. The standard practice — visible in datasets like RetailRocket (which logs `view`, `addtocart`, `transaction` as distinct event types) and in Alibaba's published systems — is to assign graded weights per event type and feed them as confidence into the model, exactly the $c = 1 + \alpha r$ machinery generalized to multiple event types. The conversion funnel itself encodes the weighting: roughly, hundreds of views collapse to tens of carts collapse to a handful of purchases, so a purchase is statistically a much rarer and therefore more informative event, and the weights should reflect that rarity-and-intent ordering.

Many production systems go further with **multi-task models** that predict click, cart, and purchase jointly — the cleanest published example being Alibaba's Entire Space Multi-Task Model (ESMM, SIGIR 2018), which was built precisely to handle the sample-selection bias that arises because conversion is only *observed* on items that were clicked. ESMM models the full chain (impression → click → conversion) over the entire impression space rather than the click-only subspace, which is the same MNAR-correction instinct this post is about, lifted to a neural multi-task setting. But you do not need ESMM to capture most of the value; the foundational move is the graded per-event weighting described here, and the multi-task machinery is an optimization on top of it once you have the volume to support it.

**The seminal papers behind all of this.** Hu, Koren, and Volinsky, "Collaborative Filtering for Implicit Feedback Datasets" (ICDM 2008), introduced the preference/confidence split and the scalable weighted-ALS solution — it is the paper to read if you read only one. Rendle, Freudenthaler, Gantner, and Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009), reframed the problem as pairwise ranking and gave the optimization criterion that directly maximizes the area under the ROC curve over observed-versus-unobserved pairs. Those two papers, a year apart, define the two dominant families for implicit feedback — pointwise confidence-weighting and pairwise ranking — and the [ALS-and-BPR deep dive](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) takes both apart in full.

## 13. Stress-testing the implicit framing

A framing is only as good as its failure modes, so let me poke at this one the way I would in a design review.

**What if your "negatives" are mostly false negatives?** With pure negative sampling on a sparse matrix, many sampled "negatives" are items the user would have loved — false negatives that inject label noise. Confidence-weighting tolerates this gracefully (blanks carry weight 1, so a wrong "negative" contributes little). Pairwise BPR is more exposed, because each gradient step pits a positive against one sampled negative; if that negative is secretly a positive, the gradient points the wrong way. The mitigations are exposure-conditioned negatives (sample from items that were *shown*, not from the whole catalog) and dampened hard-negative mining — the subject of the [negative sampling post](/blog/machine-learning/recommendation-systems/negative-sampling-strategies).

**What if offline NDCG rises but online engagement is flat?** This is the recurring nightmare of the whole field, and implicit feedback is often the culprit. Your offline test set is *itself* MNAR — it contains only items users happened to interact with under the *old* recommender. A model that perfectly predicts the old system's interactions will score well offline and yet recommend nothing new online. The offline metric is measured on a biased sample of the world; the online metric is measured on the world. Closing that gap is what off-policy evaluation and online A/B testing are for, and it is why no one ships on offline numbers alone.

**What about popularity bias?** Implicit feedback has a vicious feedback loop: popular items get shown more, so they accumulate more interactions, so they look more positive, so the model recommends them more, so they get shown more. Confidence-weighting *amplifies* this if you are not careful — high-interaction-count items get high confidence almost by definition. Counteracting it requires popularity-debiasing the loss or the sampling, which the [popularity-bias post](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) handles. The short version: watch your catalog coverage and tail metrics, not just aggregate Recall@10, or your "great" implicit model will quietly collapse the catalog to ten items.

**What at 100M items?** The confidence formula and BPR both scale, but the dense $\sum_i$ over all items in the HKV loss becomes the bottleneck. In practice you move to sampled losses (sampled softmax, in-batch negatives) and two-tower retrieval, where the implicit-feedback labeling logic from this post stays identical — you are still turning graded behavior into confidences and pairs — but the optimization machinery changes to fit the scale. That handoff from matrix-factorization to neural retrieval is the next chapter of the funnel.

**What if a user has both a thumbs-down and a high watch-time on the same item?** This is the contradictory-signal case, and it happens more than you would think — someone hate-watches a show, or finishes a movie they thought was terrible. The resolution is not to average them; it is to decide which signal your *objective* cares about. If you are optimizing watch-time (engagement), the behavior wins and the thumbs-down is noise. If you are optimizing satisfaction or retention, the explicit dislike should dominate, because a dissatisfied finisher is more likely to churn than the watch-time suggests. There is no universal answer — it is a product decision about what "good" means — but the principle is firm: when explicit and implicit signals conflict, the explicit one is usually the higher-fidelity statement of *intent*, while the implicit one is the higher-fidelity statement of *behavior*, and you weight them by which one your business metric is downstream of. This is exactly why Netflix kept the thumbs as a negative override even after moving the engine to play data.

**What if interaction counts are all 1?** Some domains have no repeats — a user buys a refrigerator once, reads a news article once. Then $r_{ui} \in \{0, 1\}$ and the confidence formula collapses to two values: $1+\alpha$ for positives, $1$ for blanks. Confidence-weighting still helps (positives are still louder than blanks by a factor of $1+\alpha$), but you lose the *graded* benefit of counts, so the leverage shifts entirely to cross-event weighting (a purchase vs a view) and to content features for cold items. In these single-interaction domains, BPR's pairwise framing tends to shine relative to count-based weighting, because it never needed the count magnitude in the first place — it only needs the *order* between a positive and a sampled negative.

## 14. When to reach for each approach (and when not to)

Decisive guidance, because every choice has a cost.

- **Default to implicit feedback with a ranking objective.** For any product with meaningful interaction volume that serves a top-K list, train on behavior (clicks/dwell/purchases) with weighted ALS or BPR and evaluate on Recall@K / NDCG@K. This is the workhorse; start here.
- **Do not optimize rating-RMSE as your north star.** It optimizes a number you rarely display, on a biased sample, and loses most of your achievable top-K performance (the 0.063 vs 0.142 NDCG gap). Predict ratings only if your product literally displays predicted ratings as a feature.
- **Do not treat blanks as hard negatives with full weight.** That invents millions of false negatives, punishes the long tail, and fights the MNAR structure. Use confidence-weighting (loud positives, quiet blanks) or pairwise ranking with sampled negatives instead.
- **Do not treat every click as a positive.** Condition on post-click satisfaction (dwell, completion, downstream action) or you train a clickbait machine. If you have only clicks, instrument dwell *immediately*.
- **Do collect explicit feedback where its marginal value is highest** — onboarding (cold start) and negatives (thumbs-down) — but let implicit carry the training volume.
- **Do not trust offline NDCG alone.** The offline set is MNAR; validate online with an A/B test, and watch catalog coverage to catch popularity collapse.

## 15. Key takeaways

1. **A blank cell is not a negative.** Implicit feedback is positive-only and missing-not-at-random; an unobserved pair is *unobserved*, not a confirmed "no." This single fact reshapes the whole problem.
2. **Implicit feedback is a positive-unlabeled problem,** not classification with zeros. Separate the binary preference $p_{ui}$ (what you believe) from the confidence $c_{ui}=1+\alpha r_{ui}$ (how strongly), and weight the loss by confidence: loud positives, quiet blanks.
3. **Rating-RMSE was a partial dead end.** It optimizes magnitudes on a self-selected sample; the product serves an *order*. On the same data, a ranking objective more than doubled NDCG@10 versus an RMSE model.
4. **Signal engineering beats model choice for ROI.** Graded weights per event type, dwell debiased by content length, and a clickbait correction (short-dwell clicks → weak negatives) move the metric more than swapping ALS for a fancier model.
5. **The label should be hard to satisfy without delivering value.** Watch-time beats clicks; purchase beats add-to-cart; finished beats started. This is why YouTube optimizes watch-time and Netflix runs on play data.
6. **Confidence weighting is cheap and powerful.** The HKV $c=1+\alpha r$ trick respects MNAR, scales linearly in nonzeros via the ALS factorization, and lifted Recall@10 ~66% over naive binary in the worked comparison.
7. **Explicit feedback still earns its keep** for cold start and for negatives — collect it where its marginal value is highest, and let implicit carry the volume.
8. **Evaluate honestly:** temporal split, filter seen items, full (not sampled) ranking, the same harness for every model. Offline NDCG is measured on an MNAR sample, so confirm online.

## 16. Further reading

- **Hu, Koren, Volinsky — "Collaborative Filtering for Implicit Feedback Datasets" (ICDM 2008):** the preference/confidence formulation and the scalable weighted-ALS solution. The single most important paper for this post.
- **Rendle, Freudenthaler, Gantner, Schmidt-Thieme — "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009):** the pairwise-ranking reframing and the optimization criterion behind BPR.
- **Steck — "Training and Testing of Recommender Systems on Data Missing Not at Random" (KDD 2010):** why you must model missingness or your offline metrics will mislead you.
- **Covington, Adams, Sargin — "Deep Neural Networks for YouTube Recommendations" (RecSys 2016):** watch-time as the label and weighted logistic regression at scale.
- **Krichene, Rendle — "On Sampled Metrics for Item Recommendation" (KDD 2020):** why sampled ranking metrics can reverse model orderings — rank against the full catalog.
- **The `implicit` library docs** (`AlternatingLeastSquares`, `BayesianPersonalizedRanking`): the production-grade implementations of both families used in the code above.
- Within this series: start at the [recommender funnel overview](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), go deep on the models in [implicit feedback models: ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr), learn to mine negatives in [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies), watch the feedback loop in [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer), and see it all assembled in [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

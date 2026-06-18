---
title: "Implicit Feedback Models: Weighted ALS and BPR"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The two canonical recipes for the positive-only clicks you actually have: weighted ALS as confidence-weighted regression and BPR as pairwise ranking, derived from scratch, coded end to end, and measured on MovieLens-as-implicit."
tags:
  [
    "recommendation-systems",
    "recsys",
    "implicit-feedback",
    "alternating-least-squares",
    "bpr",
    "matrix-factorization",
    "learning-to-rank",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/implicit-feedback-models-als-and-bpr-1.png"
---

You have a log table with a billion rows. Each row says a user touched an item: played a track, opened a product page, added something to a cart. There are no stars, no thumbs, no "not interested" taps — just a firehose of things people did. And somewhere a product manager is asking you to turn that into a homepage row of ten items each user will actually want. The first instinct of every engineer who learned recommenders from the Netflix Prize is to reach for matrix factorization and minimize squared error on a rating. But there is no rating. There is no "no." The cell a user never touched is not a zero; it is a question mark. Treat it as a zero and you build a model that learns "predict not-clicked for almost everything," which is technically accurate and completely useless.

This post is about the two recipes the field settled on for exactly this situation — positive-only, missing-not-at-random implicit feedback — and why they look so different on paper yet so similar in the leaderboard. The first is **weighted ALS** (often written **iALS**), from Hu, Koren, and Volinsky's 2008 paper "Collaborative Filtering for Implicit Feedback Datasets." It keeps the regression framing but fixes it: every cell, observed and unobserved, gets a target and a *confidence weight*, and a clever precompute makes the whole thing scale to hundreds of millions of interactions. The second is **BPR** (Bayesian Personalized Ranking), from Rendle, Freudenthaler, Gantner, and Schmidt-Thieme in 2009, which throws out the regression idea entirely: instead of predicting how much a user likes an item, it learns to *order* items, optimizing the probability that a clicked item outranks an unclicked one. One is pointwise weighted regression solved in closed form; the other is pairwise ranking solved by stochastic gradient descent. The figure below is the whole post in one picture: the same clicks, two lenses.

![Side by side comparison of the weighted ALS pointwise regression view that scores all cells with a confidence weight versus the BPR pairwise ranking view that samples a triple and maximizes the score gap](/imgs/blogs/implicit-feedback-models-als-and-bpr-1.png)

By the end you will be able to derive both objectives and their updates from first principles, explain *why* BPR is approximately optimizing AUC while iALS is not, code all three of iALS, a library BPR, and a from-scratch PyTorch BPR with an in-loop negative sampler, run WARP from `lightfm` as a middle ground, and read a results table that tells you which to reach for. This sits one rung up from [implicit vs explicit feedback and the data you have](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have) — that post explained *why* your data is positive-only and MNAR; this one gives you the two models that turn it into a ranked list — and one rung down from the full [retrieval, ranking, re-ranking funnel](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system). Both iALS and BPR are workhorses of the *retrieval* stage: they learn user and item vectors whose dot product you then serve through nearest-neighbor search.

## 1. The implicit-feedback problem, in one paragraph

Let me restate the problem so this post stands on its own. You observe a set of (user, item) interactions. Each carries a quantity $r_{ui} \ge 0$: a play count, a watch fraction, a number of purchases, or just a 1 for "interacted." You do **not** observe any explicit negatives — there is no entry that means "user $u$ saw item $i$ and rejected it." Every cell of the user-item matrix is either a positive interaction or a blank, and a blank is *missing-not-at-random* (MNAR): the user might dislike the item, or might simply never have been shown it. This makes the task a **positive-unlabeled (PU)** learning problem, not ordinary classification, because your "negative" class is contaminated with unknown positives.

Two structural facts fall out of this and shape everything below. First, **you cannot just minimize error on the observed cells the way you would with ratings**, because the observed cells are all positive — a model that predicts "1 everywhere" has zero error on the data it can see and is garbage. You are forced to say *something* about the unobserved cells, and how you say it is the entire design decision. Second, **the thing the product serves is a ranked list**, so what you actually want to optimize is the *order* of items for each user, not the absolute predicted score. iALS and BPR are two answers to "what do we say about the unobserved cells, and how do we bias the model toward good ordering." iALS says: treat every unobserved cell as a weak negative with low confidence, and a strong interaction as a confident positive. BPR says: don't score cells at all — score *pairs*, and only require that a clicked item beats an unclicked one.

A quick notation contract, used throughout. Users are indexed by $u$, items by $i$ and $j$. The latent dimension is $k$ (the number of factors). User vectors are $x_u \in \mathbb{R}^k$, item vectors $y_i \in \mathbb{R}^k$, and the predicted affinity is the dot product $\hat{x}_{ui} = x_u^\top y_i$. The set of items user $u$ interacted with is $\mathcal{I}_u^+$. Everything is built on top of plain [matrix factorization, the workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse); the novelty here is entirely in the *objective* and the *optimizer*, not in the model class. Both iALS and BPR produce the same kind of artifact: two embedding tables.

It is worth pausing on *why* the field needed two recipes at all, because the answer is the spine of this whole post. There are really only two coherent ways to handle "I have positives and a sea of unlabeled cells." The first is to *commit* to labeling the unlabeled cells — say they are all weak negatives — and then run a normal supervised regression, but down-weight your commitment so a wrongly-labeled blank does not cost you much. That is iALS: it labels everything, but it puts almost no confidence behind the blank labels. The second way is to *refuse* to label the unlabeled cells at all, and instead only make claims about *comparisons*: "this clicked item should rank above that unclicked one." That is BPR: it never asserts a single absolute label, only relative orderings, which sidesteps the entire question of "is a blank a negative." Once you see the problem this way, the two methods stop looking like rival algorithms and start looking like the only two sensible answers to one question. Everything else — the closed-form solve, the sampler, the AUC connection — is downstream of that single fork.

## 2. Weighted ALS: keep the regression, fix the labels

Hu, Koren, and Volinsky (HKV) start from a refreshingly stubborn position: regression is fine, we just labeled the data wrong. Their move is to split the raw signal $r_{ui}$ into two separate quantities — a binary **preference** and a continuous **confidence** — and then run weighted least squares over the *entire* matrix.

The preference is the indicator of any interaction at all:

$$
p_{ui} = \begin{cases} 1 & r_{ui} > 0 \\ 0 & r_{ui} = 0 \end{cases}
$$

This says: if the user interacted, the preference is "yes" (1); if not, "no" (0). On its own this is the naive binary baseline, and it is bad, because it asserts a hard 0 on every blank cell — exactly the MNAR mistake. The fix is the second quantity, **confidence**, which says *how strongly we believe the preference label*:

$$
c_{ui} = 1 + \alpha \, r_{ui}
$$

Here $\alpha$ is a hyperparameter (HKV used $\alpha = 40$ as a default). Read the formula carefully. For an unobserved cell, $r_{ui} = 0$, so $c_{ui} = 1$ — a baseline minimal confidence. We *do* assert $p_{ui} = 0$ for that cell, but only weakly. For a cell where the user interacted heavily ($r_{ui}$ large), confidence grows linearly, so we assert $p_{ui} = 1$ strongly. The blanks all carry the same low weight; positives carry weight proportional to how much the user engaged. That single decoupling is the whole idea: **a watch-100% and a watch-2% both have preference 1, but very different confidence.**

The objective is weighted regularized least squares over every $(u, i)$ pair, observed and unobserved:

$$
\min_{x_\ast, y_\ast} \sum_{u, i} c_{ui}\left( p_{ui} - x_u^\top y_i \right)^2 + \lambda \left( \sum_u \lVert x_u \rVert^2 + \sum_i \lVert y_i \rVert^2 \right)
$$

The sum runs over **all** user-item pairs — for a 1M-user, 100K-item catalog that is $10^{11}$ terms. That should make you nervous about cost, and the resolution of that worry (Section 4) is the reason this model is famous. But first, the contrast with the naive approach, which is worth one figure because it is the single most common implicit-feedback mistake.

![Side by side contrast of naive all-unobserved-equal weighting where positives are swamped by blanks versus confidence-weighted ALS where the weight grows with interaction strength so strong signals dominate](/imgs/blogs/implicit-feedback-models-als-and-bpr-5.png)

The left side is what happens when you set all weights to 1: the 99% of cells that are blank zeros dominate the loss, and the handful of real positives barely move the fit. The model's cheapest way to reduce loss is to predict near-zero everywhere, and it does. The right side is the HKV fix: positives get weights of $1 + \alpha r_{ui}$, often in the tens or hundreds, so even though they are vastly outnumbered they collectively carry enough loss mass to shape the embeddings. The blanks still matter — they pull scores toward 0 and prevent the trivial "everyone likes everything" solution — but they no longer drown out the signal.

#### Worked example: how an unobserved pair contributes

Take $\alpha = 40$. User $u$ played track $i$ 20 times, so $r_{ui} = 20$, giving $c_{ui} = 1 + 40 \cdot 20 = 801$ and $p_{ui} = 1$. User $u$ never touched track $j$, so $r_{uj} = 0$, giving $c_{uj} = 1$ and $p_{uj} = 0$. Suppose the current model predicts $\hat{x}_{ui} = x_u^\top y_i = 0.3$ and $\hat{x}_{uj} = 0.05$.

The contribution of the *observed* cell to the loss is $c_{ui}(p_{ui} - \hat{x}_{ui})^2 = 801 \cdot (1 - 0.3)^2 = 801 \cdot 0.49 = 392.5$. The contribution of the *unobserved* cell is $c_{uj}(p_{uj} - \hat{x}_{uj})^2 = 1 \cdot (0 - 0.05)^2 = 0.0025$. The positive cell carries roughly **157,000 times** the loss of the blank cell. That ratio is exactly the point: a single heavy interaction outweighs a single blank by a colossal margin, so the optimizer spends its effort fitting the positives well, while the ocean of blanks acts as a gentle, collective regularizing pull toward zero. If you had set all weights to 1, the blank's contribution would still be 0.0025 but the positive's would be only $0.49$ — and with 99 blanks for every positive, the blanks would win the loss budget. Confidence weighting is what flips that.

## 3. Deriving the ALS solve

The objective above is non-convex in $x_\ast, y_\ast$ jointly, but it is *quadratic and convex in either block with the other fixed*. That is the whole reason for **alternating least squares**: hold the item vectors fixed, solve for all user vectors (each independently and in closed form), then hold the user vectors fixed and solve for all item vectors, and repeat. Each half-step is a set of independent ridge-regression problems, and ridge regression has an exact solution.

Fix the item matrix $Y \in \mathbb{R}^{n \times k}$ (rows are item vectors) and solve for a single user vector $x_u$. Drop the other users' terms (they don't involve $x_u$) and the item regularizer (constant w.r.t. $x_u$). The per-user objective is

$$
J(x_u) = \sum_i c_{ui}\left( p_{ui} - x_u^\top y_i \right)^2 + \lambda \lVert x_u \rVert^2.
$$

Take the gradient with respect to $x_u$ and set it to zero:

$$
\frac{\partial J}{\partial x_u} = -2 \sum_i c_{ui}\left( p_{ui} - x_u^\top y_i \right) y_i + 2\lambda x_u = 0.
$$

Let $C^u = \mathrm{diag}(c_{u1}, \dots, c_{un})$ be the diagonal matrix of this user's confidences and $p^u \in \mathbb{R}^n$ the vector of preferences. Rearranging in matrix form:

$$
\left( Y^\top C^u Y + \lambda I \right) x_u = Y^\top C^u p^u,
$$

so the closed-form solution for each user is

$$
x_u = \left( Y^\top C^u Y + \lambda I \right)^{-1} Y^\top C^u p^u.
$$

This is just ridge regression with sample weights $c_{ui}$. The item update is symmetric: swap roles, fix $X$, and $y_i = (X^\top C^i X + \lambda I)^{-1} X^\top C^i p^i$. ALS sweeps: solve all $x_u$, then all $y_i$, repeat for typically 15 to 30 iterations until the embeddings stop moving. Because each half-step solves its block *exactly*, ALS converges fast and monotonically decreases the objective — far fewer iterations than SGD on the same loss.

The catch is the cost of $Y^\top C^u Y$. Naively, $C^u$ is $n \times n$ and the product touches every one of the $n$ items, so each user solve is $O(n k^2 + k^3)$ — and you do this for every user, every sweep. With $n = 100{,}000$ items and $k = 64$, that is $\sim 4 \times 10^8$ operations *per user per sweep*, times a million users, times 20 sweeps. That is on the order of $10^{16}$ operations, which is a non-starter. The HKV trick rescues it, and it is elegant enough to deserve its own section.

## 4. The YᵀY precompute: why iALS scales

The expensive term is $Y^\top C^u Y$, and the insight is that $C^u$ differs from a baseline only on the user's *few* observed items. Write $C^u = I + \mathrm{diag}((c_{ui} - 1)_i)$. Note that $c_{ui} - 1 = \alpha r_{ui}$ is **zero for every unobserved item** — it is nonzero only on the items in $\mathcal{I}_u^+$, which for a typical user is dozens, not the full hundred thousand. Substitute:

$$
Y^\top C^u Y = Y^\top \left( I + (C^u - I) \right) Y = Y^\top Y + Y^\top (C^u - I) Y.
$$

The first term, $Y^\top Y$, **does not depend on $u$ at all.** Compute it *once* per sweep — a single $k \times k$ matrix, costing $O(n k^2)$ once rather than per user. The second term, $Y^\top (C^u - I) Y$, has a diagonal weight matrix that is nonzero only on $\mathcal{I}_u^+$, so the sum runs over just the user's observed items: $\sum_{i \in \mathcal{I}_u^+} \alpha r_{ui}\, y_i y_i^\top$. That costs $O(|\mathcal{I}_u^+| \, k^2)$. Similarly, the right-hand side $Y^\top C^u p^u$ simplifies because $p^u$ is nonzero only on observed items, so it too sums over $\mathcal{I}_u^+$ only.

Putting it together, each user solve becomes

$$
x_u = \left( \underbrace{Y^\top Y}_{\text{precomputed}} + \sum_{i \in \mathcal{I}_u^+} \alpha r_{ui}\, y_i y_i^\top + \lambda I \right)^{-1} \sum_{i \in \mathcal{I}_u^+} (1 + \alpha r_{ui})\, y_i.
$$

The per-user cost drops to $O(k^2 |\mathcal{I}_u^+| + k^3)$: $k^2 |\mathcal{I}_u^+|$ to build the rank-update and right-hand side over only the user's clicks, and $k^3$ to invert the $k \times k$ system. The total per sweep is $O(k^2 \, \mathrm{nnz} + (m + n) k^3)$ where $\mathrm{nnz}$ is the number of nonzero interactions and $m, n$ are user and item counts. **It is linear in the number of observed interactions, not in the size of the full matrix.** That is why iALS trains on industrial datasets where the full matrix has $10^{13}$ cells but only $10^9$ are nonzero — you only ever pay for the nonzeros.

![Layered stack showing the iALS precompute trick where the shared YtY Gram matrix is computed once per sweep then each user adds a small correction over only its clicked items giving a cost linear in nonzeros](/imgs/blogs/implicit-feedback-models-als-and-bpr-4.png)

The stack above is the trick as layers: the shared $Y^\top Y$ Gram matrix at the top, computed once; the per-user correction that touches only clicked items in the middle; the small $k \times k$ solve; and the resulting cost that is linear in nonzeros at the bottom. This is the difference between "interesting paper" and "ran in production at Spotify-scale for a decade."

There is one more practical subtlety the trick hides: the per-user solve still requires inverting a $k \times k$ matrix, which is the $k^3$ term. For $k$ in the typical range of 32 to 256, this is a tiny dense linear solve — a Cholesky factorization of a symmetric positive-definite matrix — that BLAS handles in microseconds, and it is the same matrix structure for every user (only the rank-1 corrections differ), so it vectorizes beautifully across users on a GPU. This is why GPU iALS implementations (including `implicit`'s CUDA path) can fit MovieLens-20M in seconds and Amazon-scale data in minutes: the per-user work is a batch of small, identically-shaped dense solves, which is exactly the workload modern accelerators are built for. The sparse part (gathering each user's clicked item vectors) is the only irregular step, and it touches only the nonzeros. Contrast this with BPR's SGD, where every step is a tiny irregular gather-scatter of three embedding rows and there is no batched dense linear algebra to exploit — BPR is *memory-bandwidth bound on embedding lookups*, while iALS is *compute-bound on dense solves*. That difference in computational character is a big part of why iALS often trains an order of magnitude faster in wall-clock time for the same factor count.

#### Worked example: counting the savings

Take $k = 64$, $n = 100{,}000$ items, and a user with $|\mathcal{I}_u^+| = 50$ interactions. The naive cost of $Y^\top C^u Y$ for this user is $\approx n k^2 = 100{,}000 \cdot 4096 \approx 4.1 \times 10^8$ flops. With the precompute, the per-user work is $\approx |\mathcal{I}_u^+| k^2 + k^3 = 50 \cdot 4096 + 262{,}144 \approx 4.7 \times 10^5$ flops, *after* paying $n k^2 \approx 4.1 \times 10^8$ once for $Y^\top Y$ shared across all users. For one million users, naive is $\approx 4.1 \times 10^{14}$ per sweep; the precompute version is $4.1 \times 10^8 + 10^6 \cdot 4.7 \times 10^5 \approx 4.7 \times 10^{11}$ per sweep. That is roughly an **870x speedup**, and the gap widens as the catalog grows because the naive cost scales with $n$ while the precompute cost scales with the average interactions per user. The precompute is the single line of math that makes iALS practical.

## 5. BPR: stop predicting scores, start ranking pairs

BPR begins from a different philosophy. Rendle and coauthors argue that fitting a number $\hat{x}_{ui}$ to a target is the wrong objective for a system that serves *ranked lists*. What we actually want is, for each user, the correct *ordering* of items. So instead of saying "item $i$ should score 1 and item $j$ should score 0," BPR says "for this user, item $i$ (which they clicked) should rank above item $j$ (which they did not)." It only ever asks about *relative* order, never absolute score.

Formally, BPR builds a training set of triples $(u, i, j)$ where $i \in \mathcal{I}_u^+$ is a positive (clicked) item and $j \notin \mathcal{I}_u^+$ is an item the user has *not* interacted with — a sampled negative. The assumption is that the user prefers $i$ over $j$, written $i >_u j$. Note this is a much weaker, safer assumption than the naive one: we never claim the user *dislikes* $j$; we only claim they like $i$ *more than* $j$, which is almost certainly true on average since they engaged with $i$ and not $j$.

This is worth dwelling on, because it is the philosophical heart of BPR and the reason it sidesteps the MNAR problem so cleanly. The naive binary model makes an *absolute* claim about every blank: "the score of $j$ should be 0." That claim is wrong whenever the user never saw $j$ but would have liked it — a false negative — and it is wrong for a large fraction of blanks. BPR makes only a *relative* claim: "$i$ should outscore $j$." When $j$ is a false negative (an item the user would actually like), the relative claim is still less damaging than the absolute one, because BPR only requires $j$ to score *below* the clicked item $i$, not below zero — and across many sampled triples, the same item $j$ will sometimes be the negative and, for users who did click it, the positive, so its vector gets pulled in both directions and lands somewhere sensible. The absolute model has no such self-correction; it always pushes $j$ toward 0 for users who did not click it, even if they would have. This is a concrete, mechanical reason the pairwise framing is more robust to the missing-not-at-random structure of implicit data, and it is why BPR-style losses survived into the deep-learning era essentially unchanged.

There is also a Bayesian story behind the name, which the paper develops carefully. BPR-OPT is the *maximum-a-posteriori* estimate of the parameters under a generative model: each user's preference is a total order over items, the observed triples are draws consistent with that order, and the parameters carry a zero-mean Gaussian prior. Maximizing the posterior gives exactly the log-sigmoid-likelihood-minus-L2 objective. You do not need the full derivation to use BPR, but the framing matters: it tells you the $L_2$ term is not an ad-hoc add-on but the negative log of a proper prior, and it tells you the method is estimating an *ordering*, which is why every diagnostic you run on it (AUC, rank correlation) should be order-based, never error-based.

The model defines a difference score for the triple:

$$
\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj} = x_u^\top y_i - x_u^\top y_j = x_u^\top (y_i - y_j).
$$

The probability that the user really does prefer $i$ over $j$ is modeled with the logistic sigmoid $\sigma(z) = 1/(1 + e^{-z})$:

$$
P(i >_u j \mid \Theta) = \sigma(\hat{x}_{uij}),
$$

where $\Theta = (x_\ast, y_\ast)$ are all parameters. The BPR objective is then the maximum-a-posteriori estimate: maximize the log-likelihood of all observed orderings under a Gaussian prior on the parameters (which gives an $L_2$ regularizer):

$$
\text{BPR-OPT} = \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) - \lambda \lVert \Theta \rVert^2,
$$

where $D_S$ is the set of all valid triples. Maximizing $\ln \sigma(\hat{x}_{uij})$ means pushing the score gap $\hat{x}_{uij}$ to be large and positive — making clicked items outscore unclicked ones by a wide margin. The figure below traces one step of this through the model.

![Branching dataflow of a single BPR update where a user a positive item and a sampled negative item feed into a score gap then a sigmoid loss then an SGD update that moves all three embedding vectors](/imgs/blogs/implicit-feedback-models-als-and-bpr-2.png)

The graph shows the three inputs to a step — the user vector, the positive item, and the sampled negative — merging into the score gap, flowing through the sigmoid loss, and producing an update that moves all three vectors at once. There are far too many triples to enumerate ($|D_S|$ can be in the trillions), so BPR is trained by stochastic gradient descent with **bootstrap sampling**: at each step, draw a user, draw one of their positives, draw a random negative, and take one gradient step on that single triple.

## 6. The BPR gradient and why it optimizes AUC

Let me derive the update, because the form of the gradient is what makes BPR work. We want to *maximize* BPR-OPT, equivalently minimize its negative. For a single triple, the loss is $\ell = -\ln \sigma(\hat{x}_{uij}) + \lambda(\dots)$. Use the identity $\frac{d}{dz} \ln \sigma(z) = 1 - \sigma(z) = \sigma(-z)$. Then for any parameter $\theta$,

$$
\frac{\partial \ell}{\partial \theta} = -\sigma(-\hat{x}_{uij}) \cdot \frac{\partial \hat{x}_{uij}}{\partial \theta} + \lambda \cdot 2\theta.
$$

The scalar prefactor $\sigma(-\hat{x}_{uij})$ is the *probability the model gets this pair wrong* — it is large when the gap is small or negative (the model is wrong or unsure) and near zero when the gap is large and positive (the model is confidently correct). This is the key property: **BPR self-weights toward the pairs it is currently getting wrong**, and essentially ignores pairs it has already ordered correctly. Contrast this with the squared-error gradient in iALS, where the update magnitude is proportional to the residual $(p_{ui} - \hat{x}_{ui})$ regardless of *order* — iALS keeps pushing a positive's score toward exactly 1 even after it already outranks every negative, wasting capacity on getting the magnitude right. BPR stops caring the moment the order is correct. That is the pointwise-vs-pairwise difference made mechanical: the iALS gradient chases a *target value*; the BPR gradient chases a *correct comparison*, and a comparison that is already correct produces almost no gradient. This is the single most important reason pairwise methods tend to win order-sensitive metrics — they spend their entire optimization budget on the orderings that are still wrong, and none of it on perfecting magnitudes the ranking will never display.

Now plug in the derivatives of $\hat{x}_{uij} = x_u^\top(y_i - y_j)$:

$$
\frac{\partial \hat{x}_{uij}}{\partial x_u} = y_i - y_j, \qquad
\frac{\partial \hat{x}_{uij}}{\partial y_i} = x_u, \qquad
\frac{\partial \hat{x}_{uij}}{\partial y_j} = -x_u.
$$

With learning rate $\eta$ and writing $g = \sigma(-\hat{x}_{uij})$ for the error gate, the gradient-*descent* updates (move against the gradient) are:

$$
x_u \mathrel{+}= \eta \left( g \cdot (y_i - y_j) - \lambda x_u \right),
$$
$$
y_i \mathrel{+}= \eta \left( g \cdot x_u - \lambda y_i \right),
$$
$$
y_j \mathrel{+}= \eta \left( g \cdot (-x_u) - \lambda y_j \right).
$$

Read these geometrically. The user vector moves toward $(y_i - y_j)$, i.e., toward the positive and away from the negative. The positive item vector moves toward the user. The negative item vector moves *away* from the user. All three moves are scaled by $g$, the probability of being wrong. The model literally pushes correctly-ordered apart and pulls the wrongly-ordered into line, harder when it is more wrong.

Now the headline claim: **BPR-OPT is a smooth surrogate for the AUC.** The area under the ROC curve, for a single user, is the probability that a randomly chosen positive item scores higher than a randomly chosen negative:

$$
\text{AUC}_u = \frac{1}{|\mathcal{I}_u^+| \cdot |\mathcal{I}_u^-|} \sum_{i \in \mathcal{I}_u^+} \sum_{j \in \mathcal{I}_u^-} \mathbb{1}\!\left[ \hat{x}_{ui} > \hat{x}_{uj} \right].
$$

That indicator $\mathbb{1}[\hat{x}_{ui} > \hat{x}_{uj}] = \mathbb{1}[\hat{x}_{uij} > 0]$ is exactly the Heaviside step, which is non-differentiable. BPR replaces it with the smooth $\sigma(\hat{x}_{uij})$ — a soft, differentiable version of "is this pair correctly ordered." Maximizing $\sum \ln \sigma(\hat{x}_{uij})$ is therefore maximizing a differentiable lower-bound-like surrogate for the count of correctly ordered pairs, which is AUC up to normalization. (The exact relationship is that BPR optimizes a softened version of AUC; the log-sigmoid is not literally a bound on AUC, but its maximizer drives the same quantity. The Rendle paper makes the surrogate connection explicit.) This is *why* BPR is a ranking method and iALS is not: BPR's loss only ever looks at *differences* of scores, so it is invariant to any monotone shift of all scores — it cares purely about order, which is precisely what AUC measures.

#### Worked example: one BPR update with numbers

Use $k = 3$, learning rate $\eta = 0.05$, regularization $\lambda = 0.01$. Current vectors:

- User: $x_u = (0.5,\ -0.2,\ 0.1)$
- Positive item $i$: $y_i = (0.4,\ 0.1,\ -0.3)$
- Sampled negative $j$: $y_j = (0.2,\ 0.5,\ 0.2)$

Compute the scores. $\hat{x}_{ui} = 0.5\cdot0.4 + (-0.2)\cdot0.1 + 0.1\cdot(-0.3) = 0.20 - 0.02 - 0.03 = 0.15$. And $\hat{x}_{uj} = 0.5\cdot0.2 + (-0.2)\cdot0.5 + 0.1\cdot0.2 = 0.10 - 0.10 + 0.02 = 0.02$. The gap is $\hat{x}_{uij} = 0.15 - 0.02 = 0.13$. The model already ranks $i$ above $j$, but only barely.

The error gate is $g = \sigma(-0.13) = 1/(1 + e^{0.13}) = 1/(1 + 1.1388) = 0.4675$. Because the gap is small, the gate is near 0.5 — the model is barely confident, so this update will be sizeable. Now the updates. First $y_i - y_j = (0.4 - 0.2,\ 0.1 - 0.5,\ -0.3 - 0.2) = (0.2,\ -0.4,\ -0.5)$.

User update: $x_u \mathrel{+}= \eta(g(y_i - y_j) - \lambda x_u) = 0.05 \cdot (0.4675 \cdot (0.2, -0.4, -0.5) - 0.01 \cdot (0.5, -0.2, 0.1))$. Inside: $0.4675 \cdot (0.2,-0.4,-0.5) = (0.0935, -0.1870, -0.2338)$; minus $(0.005, -0.002, 0.001) = (0.0885, -0.1850, -0.2348)$; times $0.05 = (0.00443, -0.00925, -0.01174)$. So $x_u \to (0.5044,\ -0.2093,\ 0.0883)$.

Positive update: $y_i \mathrel{+}= \eta(g\, x_u - \lambda y_i) = 0.05(0.4675(0.5,-0.2,0.1) - 0.01(0.4,0.1,-0.3))$. Inside: $(0.2338, -0.0935, 0.0468) - (0.004, 0.001, -0.003) = (0.2298, -0.0945, 0.0498)$; times $0.05 = (0.01149, -0.00472, 0.00249)$. So $y_i \to (0.4115,\ 0.0953,\ -0.2975)$.

Negative update: $y_j \mathrel{+}= \eta(-g\, x_u - \lambda y_j) = 0.05(-0.4675(0.5,-0.2,0.1) - 0.01(0.2,0.5,0.2))$. Inside: $(-0.2338, 0.0935, -0.0468) - (0.002, 0.005, 0.002) = (-0.2358, 0.0885, -0.0488)$; times $0.05 = (-0.01179, 0.00442, -0.00244)$. So $y_j \to (0.1882,\ 0.5044,\ 0.1976)$.

Verify the gap widened. New $\hat{x}_{ui} = 0.5044\cdot0.4115 + (-0.2093)\cdot0.0953 + 0.0883\cdot(-0.2975) = 0.2076 - 0.0199 - 0.0263 = 0.1614$. New $\hat{x}_{uj} = 0.5044\cdot0.1882 + (-0.2093)\cdot0.5044 + 0.0883\cdot0.1976 = 0.0949 - 0.1056 + 0.0174 = 0.0067$. New gap $= 0.1614 - 0.0067 = 0.1547$, up from 0.13. One step widened the margin by about 19%. That is BPR doing its one job: making clicked items outrank unclicked ones, harder when the margin is thin.

One more thing the gradient form reveals: BPR is **not** invariant to the choice of negative. Look again at the error gate $g = \sigma(-\hat{x}_{uij})$. If you sample an easy negative — an item the model already ranks far below the positive — then $\hat{x}_{uij}$ is large and positive, $g \approx 0$, and the update is essentially zero. The step was wasted. If you sample a *hard* negative — one the model wrongly ranks near or above the positive — then $\hat{x}_{uij}$ is near zero or negative, $g \approx 0.5$ or larger, and you get a big, informative update. So the *distribution you sample negatives from* directly controls how much signal each step carries. Uniform random sampling, the BPR default, draws mostly easy negatives once the model is even slightly trained, which is why vanilla BPR's loss curve flattens early: it is increasingly spending steps on pairs it already orders correctly. This observation is the seed of every advanced sampler — popularity-based, in-batch, dynamic hard-negative mining — and it is exactly what WARP automates by *searching* for a hard negative instead of hoping uniform sampling hands it one.

## 7. iALS vs BPR vs WARP, head to head

Now the comparison the whole post is building toward. The two models optimize genuinely different things, and a third — WARP — sits between them. The matrix below is the cheat sheet.

![Comparison matrix of iALS BPR and WARP across objective negatives optimizes and scales showing iALS uses weighted regression in closed form BPR uses random pairs for AUC and WARP searches for violating negatives to optimize top-K rank](/imgs/blogs/implicit-feedback-models-als-and-bpr-3.png)

Walk down the columns. **Objective**: iALS is pointwise weighted regression (fit a number per cell); BPR is pairwise log-sigmoid (order two items); WARP is also pairwise but *rank-weighted*. **Negatives**: iALS implicitly uses all unobserved cells as weak negatives (it never samples — they are all in the sum); BPR samples one random negative per step; WARP samples *repeatedly until it finds a negative the model currently ranks above the positive* — a "violating" negative — and weights the update by how many tries that took. **Optimizes**: iALS minimizes confidence-weighted squared error (correlated with but not equal to ranking); BPR optimizes AUC (the whole-list ordering); WARP approximates a *top-K* metric like Recall@K by focusing updates on negatives that intrude into the top of the list. **Scales**: iALS solves in closed form with the $Y^\top Y$ trick and parallelizes trivially across users; BPR is plain SGD and scales easily; WARP's per-step cost is higher because of the sampling search, especially late in training when violations are rare.

The crucial intuition about WARP — and the reason it usually wins offline top-K metrics — is the **rank-weighting**. AUC (what BPR optimizes) cares equally about an error anywhere in the list: getting a pair wrong at rank 5000 versus rank 5 counts the same. But the product only shows the top 10. WARP fixes this by sampling negatives until it finds one that violates the margin, then estimating the positive item's rank from how many samples that took (few samples to find a violator means the positive is ranked low, i.e., a big problem), and scaling the gradient by a function of that estimated rank. Errors near the top of the list get large updates; errors deep in the tail get small ones. This is the "listwise-ish" middle ground: it is still pair-based and cheap like BPR, but it bends the optimization toward the head of the list like a top-K metric wants. The taxonomy below places all three.

![Taxonomy tree of the implicit-model family branching into pointwise with iALS pairwise with BPR and rank-aware with WARP each a deeper approximation to optimizing the served top-K list](/imgs/blogs/implicit-feedback-models-als-and-bpr-6.png)

The tree organizes the family by *what the unit of learning is*: a single cell (pointwise: iALS), a pair (pairwise: BPR), or an approximate position in the ranked list (rank-aware: WARP). Reading left to right is reading "how directly does this optimize the thing the product serves." iALS is the least direct (it optimizes a weighted regression that happens to correlate with ranking) and the easiest to scale; WARP is the most direct (it chases top-K rank) and the most expensive per step; BPR is the pragmatic middle that, for years, was the default first thing everyone tried. None is strictly best — Section 15 makes the call concrete.

Let me make the WARP rank-estimation concrete, because it is the cleverest of the three and the least understood. For a positive item $i$ that the model scores $\hat{x}_{ui}$, WARP wants to estimate its *rank* among all items without computing all scores. It samples negatives $j$ one at a time and checks whether $\hat{x}_{uj} + \text{margin} > \hat{x}_{ui}$ — a violation. If it finds the first violation after $N$ samples out of a catalog of $n$ items, it estimates the positive's rank as $\lfloor (n-1)/N \rfloor$: finding a violator quickly (small $N$) means many items beat the positive, so its rank is high (bad); needing many samples (large $N$) means the positive is already near the top (good). WARP then scales the gradient by a rank-weighting function $L(\text{rank}) = \sum_{r=1}^{\text{rank}} 1/r$ (a harmonic-style penalty), which grows fast for low ranks and slowly for high ones. The effect: a positive buried at rank 5000 gets a large corrective push; a positive already at rank 3 gets almost none. This is precisely "optimize the head of the list," and it is why WARP tracks Recall@K and NDCG@K so much better than AUC-optimizing BPR. The cost is the sampling search: late in training, when the model is good, violations are rare, so $N$ grows large and each step gets slow — WARP usually caps the search at a maximum number of tries to bound this.

| Aspect | iALS (weighted ALS) | BPR | WARP |
|---|---|---|---|
| Loss family | Pointwise weighted MSE | Pairwise log-sigmoid | Pairwise rank-weighted |
| What it optimizes | Confidence-weighted error | AUC (full list) | Top-K rank (Recall@K) |
| Negatives | All unobserved, weighted | 1 random per step | Sampled until violation |
| Optimizer | Closed-form ALS sweeps | SGD bootstrap | SGD with rank search |
| Convergence | Fast, monotone, ~15–30 sweeps | Slower, many epochs | Slower, sampling cost grows |
| Parallelism | Embarrassingly (per user) | Easy (mini-batch / async) | Harder (search per step) |
| Confidence weighting | Native ($\alpha$) | No (use weighted variants) | No |
| Best when | Dense-ish, want speed, count signal | General default, AUC matters | Top-K is the metric, can afford it |
| Library | `implicit.AlternatingLeastSquares` | `implicit.BayesianPersonalizedRanking`, `lightfm` | `lightfm` (loss="warp") |

## 8. Coding it: iALS and BPR with the `implicit` library

Enough theory. Let me build the running example: MovieLens-as-implicit. We take MovieLens ratings, treat any rating as a positive interaction (a "1" or, if you want graded confidence, the rating value), drop the rating magnitude as a target, and learn to rank. The same recipe applies to RetailRocket (view/addtocart/transaction events) with only the data-loading changed. First, build the sparse interaction matrix.

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# MovieLens 1M as implicit: any rating -> an interaction.
# columns: userId, movieId, rating, timestamp
ratings = pd.read_csv(
    "ml-1m/ratings.dat", sep="::", engine="python",
    names=["userId", "movieId", "rating", "timestamp"],
)

# Temporal split: last 20% of each user's events go to test (no leakage).
ratings = ratings.sort_values("timestamp")
def temporal_split(df, frac=0.2):
    n_test = max(1, int(len(df) * frac))
    return df.iloc[:-n_test], df.iloc[-n_test:]

train_parts, test_parts = [], []
for _, g in ratings.groupby("userId"):
    if len(g) < 5:           # skip users too sparse to evaluate
        continue
    tr, te = temporal_split(g)
    train_parts.append(tr); test_parts.append(te)
train = pd.concat(train_parts); test = pd.concat(test_parts)

# Contiguous integer ids for the sparse matrix.
uids = {u: k for k, u in enumerate(train.userId.unique())}
iids = {i: k for k, i in enumerate(train.movieId.unique())}
def to_matrix(df):
    rows = df.userId.map(uids); cols = df.movieId.map(iids)
    keep = rows.notna() & cols.notna()
    rows, cols = rows[keep].astype(int), cols[keep].astype(int)
    # value = 1 for binary preference; use df.rating[keep] for graded r_ui
    vals = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(len(uids), len(iids)))

train_ui = to_matrix(train)   # users x items
print("train shape:", train_ui.shape, "nnz:", train_ui.nnz)
```

Now train iALS. The `implicit` library expects a user-by-item matrix and applies the confidence weighting internally; you pass `alpha` as a multiplier on the matrix values. Note that older `implicit` versions wanted an item-by-user matrix; modern versions (0.5+) take user-by-item and document it clearly — always check the version.

```python
from implicit.als import AlternatingLeastSquares

# Confidence c_ui = 1 + alpha * r_ui is applied by scaling the matrix by alpha.
als = AlternatingLeastSquares(
    factors=64,           # k
    regularization=0.05,  # lambda
    alpha=40.0,           # the HKV confidence multiplier
    iterations=20,        # ALS sweeps
    use_gpu=False,
    random_state=42,
)
als.fit(train_ui)         # learns als.user_factors, als.item_factors

# Recommend top-10 for user 0, filtering items already seen.
ids, scores = als.recommend(
    0, train_ui[0], N=10, filter_already_liked_items=True,
)
print("iALS top-10 item ids:", ids)
```

BPR from the same library is a one-line swap of the model class. Internally it runs SGD with bootstrap negative sampling, exactly the procedure derived above.

```python
from implicit.bpr import BayesianPersonalizedRanking

bpr = BayesianPersonalizedRanking(
    factors=64,
    learning_rate=0.05,
    regularization=0.01,
    iterations=100,       # BPR needs many more passes than ALS
    random_state=42,
)
bpr.fit(train_ui)
ids, scores = bpr.recommend(0, train_ui[0], N=10,
                            filter_already_liked_items=True)
print("BPR top-10 item ids:", ids)
```

Two things to notice. iALS converges in ~20 sweeps because each sweep solves blocks exactly; BPR needs ~100 epochs because each SGD step only sees one triple. And `alpha` is meaningful for iALS (it sets confidence sharpness) but absent for BPR (which never weights — it only orders).

## 9. Coding it: BPR from scratch in PyTorch with a negative sampler

To really understand BPR, write the negative sampler and training loop yourself. This is also the template you would extend with features, sequence models, or hard-negative mining later. The model is two embedding tables; the loss is the log-sigmoid of the score gap; the sampler draws a uniform random negative per positive in the batch.

```python
import torch
import torch.nn as nn

class BPRMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, k=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i, j):
        xu = self.user_emb(u)              # (B, k)
        yi = self.item_emb(i)             # positive
        yj = self.item_emb(j)             # sampled negative
        x_ui = (xu * yi).sum(dim=1)       # (B,)
        x_uj = (xu * yj).sum(dim=1)
        return x_ui - x_uj                # the score gap x_uij
```

The negative sampler is the part people get subtly wrong. A drawn negative should not accidentally be one of the user's positives, or you are penalizing the model for ranking a true positive highly. The clean version checks membership in the user's positive set; the fast version just resamples on collision (collisions are rare when items are many).

```python
import numpy as np

# Precompute each user's positive item set once.
user_pos = {u: set(train.userId.map(uids)
                   .where(train.userId.map(uids) == u).dropna().index)
            for u in range(len(uids))}
# Simpler: build positives from the sparse matrix rows.
user_pos = [set(train_ui[u].indices.tolist()) for u in range(train_ui.shape[0])]
n_items = train_ui.shape[1]

# Flat array of (user, positive_item) pairs to iterate over.
coo = train_ui.tocoo()
pairs = np.stack([coo.row, coo.col], axis=1)        # (nnz, 2)

def sample_negatives(user_ids):
    """Uniform random negative per user, avoiding their positives."""
    negs = np.random.randint(0, n_items, size=len(user_ids))
    for k, u in enumerate(user_ids):
        while negs[k] in user_pos[u]:      # resample on collision
            negs[k] = np.random.randint(0, n_items)
    return negs
```

Now the training loop. Bootstrap sampling means each epoch we shuffle all positive pairs, mini-batch them, and draw a fresh negative per pair every step. The loss is `-logsigmoid(gap)` plus L2 (handled by `weight_decay`).

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BPRMatrixFactorization(train_ui.shape[0], n_items, k=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
logsig = nn.LogSigmoid()

BATCH = 4096
for epoch in range(30):
    perm = np.random.permutation(len(pairs))
    total = 0.0
    for start in range(0, len(pairs), BATCH):
        idx = perm[start:start + BATCH]
        u = pairs[idx, 0]; i = pairs[idx, 1]
        j = sample_negatives(u)            # fresh negatives each step
        u_t = torch.as_tensor(u, device=device)
        i_t = torch.as_tensor(i, device=device)
        j_t = torch.as_tensor(j, device=device)
        gap = model(u_t, i_t, j_t)         # x_uij
        loss = -logsig(gap).mean()         # maximize ln sigma(gap)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(idx)
    print(f"epoch {epoch}: bpr loss {total / len(pairs):.4f}")
```

This is the exact objective from Section 5 and the exact gradient from Section 6 — autograd computes the $\sigma(-\hat{x}_{uij})$ error gate for you. The number of negative samples per positive is a knob: drawing several negatives per positive and averaging (or taking the max-violating one) is a cheap step toward WARP-like hard negatives, a topic the [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) post takes much further. The deeper mechanics of the pairwise loss itself — margins, the BPR-vs-hinge choice, multiple negatives — are in the [pairwise and BPR loss deep dive](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive).

## 10. WARP with LightFM, and the evaluation harness

LightFM gives you BPR and WARP behind one API, and it natively supports side features (so it is also your hybrid/cold-start tool later). The loss is a single argument.

```python
from lightfm import LightFM
from lightfm.data import Dataset

ds = Dataset()
ds.fit(users=train.userId.unique(), items=train.movieId.unique())
(interactions, weights) = ds.build_interactions(
    (row.userId, row.movieId) for row in train.itertuples()
)

# WARP: the rank-aware middle ground.
warp = LightFM(no_components=64, loss="warp",
               learning_rate=0.05, item_alpha=1e-6, user_alpha=1e-6)
warp.fit(interactions, epochs=30, num_threads=4)

# Swap loss="bpr" for the pairwise BPR variant in the same framework.
bpr_lf = LightFM(no_components=64, loss="bpr", learning_rate=0.05)
bpr_lf.fit(interactions, epochs=30, num_threads=4)
```

Now the honest evaluation harness. Three rules keep you from fooling yourself. **Temporal split** (already done above — no future leakage). **Full-catalog ranking** for the metric, or at least a clearly-stated sampled protocol — because sampled metrics can reorder models (the KDD'20 "On Sampled Metrics for Item Recommendation" result). And **filter seen items** from the candidate list. Here is a compact Recall@K / NDCG@K / MAP@K computer over learned user and item factors.

```python
def metrics_at_k(user_factors, item_factors, test_by_user,
                 seen_by_user, K=10):
    recalls, ndcgs, aps = [], [], []
    scores_all = user_factors @ item_factors.T          # (U, I) dense
    idcg_cache = {n: sum(1.0 / np.log2(t + 2) for t in range(n))
                  for n in range(1, K + 1)}
    for u, truth in test_by_user.items():
        if not truth:
            continue
        s = scores_all[u].copy()
        s[list(seen_by_user[u])] = -np.inf               # filter seen
        topk = np.argpartition(-s, K)[:K]
        topk = topk[np.argsort(-s[topk])]                # ordered top-K
        hits = [1 if it in truth else 0 for it in topk]
        n_rel = min(len(truth), K)
        recalls.append(sum(hits) / len(truth))
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        ndcgs.append(dcg / idcg_cache[n_rel] if n_rel else 0.0)
        # average precision @ K
        num_hits, ap = 0, 0.0
        for r, h in enumerate(hits):
            if h:
                num_hits += 1
                ap += num_hits / (r + 1)
        aps.append(ap / n_rel if n_rel else 0.0)
    return (float(np.mean(recalls)), float(np.mean(ndcgs)),
            float(np.mean(aps)))
```

The popularity baseline you must beat is one line: rank every item by its global interaction count and serve the same list to everyone (minus seen items). If your fancy model does not clearly beat popularity on a temporal split, something is broken — popularity is shockingly strong because of the head-heavy nature of catalogs, and beating it is the real bar, not beating random.

```python
# Popularity baseline: same ranking for every user.
item_pop = np.asarray(train_ui.sum(axis=0)).ravel()      # interactions/item
pop_order = np.argsort(-item_pop)
```

## 11. Results: popularity vs iALS vs BPR vs WARP

Here is a representative results table on **MovieLens-1M treated as implicit** with a per-user temporal split, full-catalog ranking, $k = 64$ factors, and each model tuned lightly. Numbers in this range are consistent with what these models score on ML-1M in the literature and in benchmark suites such as RecBole and Microsoft Recommenders; treat them as a calibrated illustration of the *relationships*, not a competition entry — your exact figures will shift with the split, the K, and whether you rank the full catalog or a sampled set.

| Model | Recall@10 | NDCG@10 | MAP@10 | Train time (rel.) | Notes |
|---|---|---|---|---|---|
| Popularity | 0.108 | 0.121 | 0.061 | trivial | non-personalized floor |
| iALS ($\alpha{=}40$, $k{=}64$) | 0.214 | 0.228 | 0.118 | 1.0x | fastest to fit |
| BPR (`implicit`) | 0.205 | 0.241 | 0.121 | ~4x | strong NDCG |
| WARP (`lightfm`) | 0.223 | 0.247 | 0.129 | ~6x | best top-K |

![Side by side result contrast of the popularity baseline at Recall@10 0.108 and NDCG@10 0.121 versus personalized iALS BPR and WARP that roughly double those numbers on the metrics the product serves](/imgs/blogs/implicit-feedback-models-als-and-bpr-8.png)

The shape of these results is the lesson, and it is stable across datasets. **Personalization roughly doubles the popularity floor** — that is the value of learning user vectors at all. **iALS leads on Recall@10** (it casts a wide net; the confidence-weighted regression tends to surface a slightly more diverse set of relevant items) but **trails BPR/WARP on NDCG@10** (it does not optimize order, so the *positions* within the top-10 are a touch worse). **BPR edges iALS on the order-sensitive metrics** because it optimizes ordering directly. **WARP wins the top-K metrics** because it bends the optimization toward the head of the list — at several times the training cost. And iALS is the cheapest to train by a wide margin, which at industrial scale often dominates the decision.

Now the hyperparameter sweeps that you should actually run. The matrix below shows the characteristic shape of each knob.

![Hyperparameter sweep matrix showing Recall@10 rising then plateauing across alpha for iALS factor count for both models and negative sample count for BPR with a clear sweet spot in each row](/imgs/blogs/implicit-feedback-models-als-and-bpr-7.png)

Reading the sweep: **$\alpha$ for iALS** has a clear peak. Too low ($\alpha=1$) and confidence weighting barely differs from flat, so positives don't dominate (Recall@10 $\approx$ 0.171); the sweet spot around $\alpha=40$ gives the best result (0.214); too high ($\alpha=200$) over-weights heavy users and slightly hurts (0.205). **Factor count $k$** improves sharply then plateaus — $k=16$ underfits (0.182), $k=64$ is near-optimal (0.214), and $k=256$ barely moves the needle (0.216) while quadrupling memory and the $k^3$ solve cost. **Negative samples for BPR** show diminishing returns: 1 negative per step works (0.196), 5 is meaningfully better (0.209), and 20 buys almost nothing (0.211) at much higher per-epoch cost. The practical reading: tune $\alpha$ and $k$ for iALS first (they are cheap to sweep because ALS is fast), and for BPR, a handful of negatives per positive is the efficient frontier.

#### Worked example: computing NDCG@10 by hand for two models

To make "BPR wins NDCG, iALS wins Recall" concrete, compute both for one user. The user has three held-out positives among the 10 served slots. Suppose iALS's top-10 contains hits at positions 1, 8, 9 (it surfaced all three relevant items but bunched two at the bottom), and BPR's top-10 contains hits at positions 1, 2, 5 (it found the same number of relevant items but ranked them higher).

Recall@10 is identical: both found 3 of 3 relevant items in the top-10, so Recall@10 = 3/3 = 1.0 for both. Recall does not care *where* in the top-10 the hits land. Now NDCG@10. The discounted cumulative gain is $\text{DCG} = \sum_r \text{hit}_r / \log_2(r+1)$ with positions $r=1,\dots,10$. For iALS, hits at 1, 8, 9: $\text{DCG} = 1/\log_2 2 + 1/\log_2 9 + 1/\log_2 10 = 1/1 + 1/3.170 + 1/3.322 = 1 + 0.3155 + 0.3010 = 1.6165$. For BPR, hits at 1, 2, 5: $\text{DCG} = 1/\log_2 2 + 1/\log_2 3 + 1/\log_2 6 = 1 + 1/1.585 + 1/2.585 = 1 + 0.6309 + 0.3869 = 2.0178$. The ideal DCG (all three hits at positions 1, 2, 3) is $1 + 0.6309 + 0.5 = 2.1309$. So iALS NDCG@10 $= 1.6165 / 2.1309 = 0.759$ and BPR NDCG@10 $= 2.0178 / 2.1309 = 0.947$. **Same Recall, very different NDCG** — BPR's order-optimizing objective put the hits where the discount is gentle, and that is the entire mechanical reason it leads on the position-sensitive metric while tying on Recall. This is not hand-waving; it falls straight out of what each model optimizes.

#### Worked example: reading the iALS Recall lift honestly

Suppose your popularity baseline serves Recall@10 = 0.108 and your tuned iALS hits 0.214. The naive headline is "we doubled recall." Before you write that in the launch doc, sanity-check three things. First, **the split** — did you use a temporal split, or a random one? A random split leaks future interactions into training and inflates every model, popularity included, but inflates the learned model more. Second, **full vs sampled metric** — if you computed Recall@10 against 100 sampled negatives instead of the full catalog, the absolute numbers are not comparable to the literature and, worse, the *ranking* of models can flip (KDD'20). Third, **the denominator** — Recall@10 with a per-user test set of size 1 is just HitRate@10; make sure your test users have enough held-out positives that the metric is stable. With a clean temporal split and full-catalog ranking, a jump from 0.108 to 0.214 is a real, defensible doubling — and on a deployed system you would expect it to translate to a single-digit-percent lift in your online engagement metric, not a doubling, because online has position bias, novelty effects, and the rest of the [offline-online reality gap](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

## 12. The role of α and the negative sampler

These are the two "soul" hyperparameters of the two methods, and they play analogous roles. In iALS, **$\alpha$ controls how sharply confidence rises with interaction strength.** At $\alpha=0$, every cell has confidence 1 and you are back to flat, broken regression. As $\alpha \to \infty$, only the heaviest interactions matter and the model becomes brittle, over-fitting your power users and ignoring the long tail of light-but-real signal. The sweet spot (HKV's $\alpha=40$ is a fine starting point for count data; for binary data $\alpha$ around 1 to 15 is more typical) is where heavy interactions dominate enough to shape the embeddings but light interactions still register. If your signal is graded (watch fraction, play count), $\alpha$ is doing real work; if it is binary, $\alpha$ mostly sets the global positive-to-blank weight ratio.

In BPR, **the negative sampler is the analog of $\alpha$ — it controls which non-interactions the model learns from.** The vanilla sampler draws uniform random negatives, and most random negatives are *easy*: items so unrelated to the user that the model already ranks them far below the positive, so the error gate $\sigma(-\hat{x}_{uij}) \approx 0$ and the update is tiny. As training proceeds, an ever-larger fraction of steps are wasted on already-correct easy pairs. This is the single biggest inefficiency of vanilla BPR and the reason WARP exists: WARP's sample-until-violation search is precisely a *hard-negative miner*, spending compute to find negatives the model is currently getting wrong. The trade-off is brutal late in training — when the model is good, violations are rare, so WARP samples many candidates per step before finding one, and per-step cost balloons. The general principle (covered in depth in [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies)) is that *harder negatives give a stronger gradient signal per step but cost more to find and risk sampling false negatives* — items the user would actually like but never happened to interact with. For implicit data, where blanks are MNAR, every "negative" is a guess, and the harder you mine, the more likely you are to mine a false negative. That tension never fully resolves; you tune it.

## 13. Convergence, initialization, and the things that actually break in practice

Beyond the headline objective, a handful of practical details decide whether your training run produces a usable model or a pile of NaNs. They are worth stating because they are exactly the things that page you.

**iALS convergence is monotone but the regularizer matters a lot.** Because each ALS half-step solves a convex sub-problem exactly, the objective decreases every sweep and you can watch it flatten — typically by sweep 15 to 20 on MovieLens-scale data. But the closed-form solve inverts $(Y^\top Y + \dots + \lambda I)$, and if $\lambda$ is too small relative to the confidence scale, that matrix can be ill-conditioned (near-singular), producing huge, unstable user vectors. The fix is not just "raise $\lambda$" — it is to scale $\lambda$ with $\alpha$, because higher confidence inflates $Y^\top C^u Y$ and demands proportionally more regularization. A common parameterization regularizes per-user and per-item by the *number of interactions* (so heavy users get more shrinkage), which `implicit` exposes. If your iALS recommendations look like noise, check the conditioning before you blame the model.

**BPR is far more fragile to learning rate and initialization.** Initialize the embeddings too large and the initial score gaps are huge, the sigmoid saturates, the gradient vanishes, and the model never moves — a silent failure that looks like "loss stuck at 0.69." Initialize too small (as in the code above, `std=0.01`) and you start near $\hat{x}_{uij} \approx 0$, $g \approx 0.5$, with healthy gradients everywhere. The learning rate interacts with this: too high and the updates overshoot and the loss oscillates; too low and 100 epochs is not enough to converge. Adam with a learning rate around $5\times10^{-3}$ and small embedding init is a robust starting point, but BPR genuinely needs a small learning-rate sweep where iALS needs only an $\alpha$/$\lambda$ sweep. This asymmetry — iALS "just works" given reasonable regularization, BPR needs babysitting — is an underrated reason iALS stays popular for first-pass production models.

**Both methods are blind to popularity unless you correct it.** Left alone, both iALS and BPR will over-recommend popular items, because popular items appear in more users' positive sets and so accumulate stronger, more confident gradients. For iALS this shows up as popular items having large-norm vectors that dominate dot products; for BPR it shows up as the negative sampler rarely drawing popular items (they are usually positives for the user) so their vectors get pushed up more than down. If diversity matters, you correct this downstream — popularity-debiased sampling for BPR, or post-hoc re-ranking — which connects to the broader [feedback-loop and popularity-bias](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) story the series keeps returning to. The model is not wrong; it is faithfully reproducing the popularity structure of your logs, which is exactly the self-reinforcing loop you must break on purpose.

## 14. Case studies and where these models actually run

**iALS at music-streaming scale.** Implicit matrix factorization with the confidence-weighting recipe became the canonical first-pass collaborative-filtering model at large music services; Spotify engineers have publicly described using ALS-style implicit MF (and built the widely-used `implicit` and earlier `annoy` libraries partly for this) to learn track and user vectors from play counts, then serving recommendations via approximate nearest-neighbor search over the item vectors. The reasons are exactly the ones this post derived: play counts are graded implicit signal (so confidence weighting is natural), the $Y^\top Y$ trick makes it scale to hundreds of millions of users and tens of millions of tracks, and ALS parallelizes trivially across users for distributed training. The learned item vectors double as a similarity space ("songs like this one"), which is a free second product.

**BPR's ubiquity as the baseline.** Since 2009, BPR has been the near-universal *baseline* in implicit-recommendation papers — if you read almost any retrieval or sequential-recommendation paper (SASRec, BERT4Rec, LightGCN, and hundreds more), BPR or a BPR-style pairwise loss is in the comparison table, and frequently it is the loss the proposed model itself is trained with. LightGCN, for instance, is a graph model trained with the BPR loss; the architecture changed but the objective did not. This longevity is the strongest evidence for the central claim: optimizing *order* rather than *score* is the right framing for implicit top-K recommendation, and BPR was the clean first articulation of it.

**WARP in production hybrid systems.** LightFM's WARP loss is a common choice for small-to-mid-scale recommenders and especially for *cold-start hybrids*, because LightFM blends collaborative signal with content features and WARP optimizes the top-K metric directly. Teams report WARP outperforming BPR on Recall@K and Precision@K on datasets like MovieLens and the LightFM authors' own benchmarks, at the cost of slower training — the same Pareto trade-off the results table showed.

**E-commerce graded confidence.** On e-commerce data (RetailRocket, Taobao), the natural move is graded confidence by event type: a view is a weak positive, an add-to-cart stronger, a purchase strongest. iALS handles this directly — set $r_{ui}$ to a weighted sum of event counts (e.g. $1 \cdot \text{views} + 3 \cdot \text{carts} + 10 \cdot \text{purchases}$) and let $c_{ui} = 1 + \alpha r_{ui}$ turn it into confidence. This is one of iALS's quiet advantages over vanilla BPR: graded confidence is built into the objective, whereas plain BPR treats every positive identically and needs a weighted or multi-level variant to express "a purchase is worth more than a view."

The seminal references, which you should read in full: Hu, Koren, and Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," ICDM 2008 (the iALS paper); Rendle, Freudenthaler, Gantner, and Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback," UAI 2009 (the BPR paper); and Weston, Bengio, and Usunier, "WSABIE: Scaling Up to Large Vocabulary Image Annotation," IJCAI 2011 (the origin of the WARP loss, which LightFM adapts to recommendation).

## 15. When to reach for iALS, BPR, or WARP

Here is the decision, stated plainly, because "it depends" is a cop-out.

**Reach for iALS when** you have graded implicit signal (play counts, watch fractions, multi-event e-commerce), you want a model that trains *fast* and scales to huge data via the $Y^\top Y$ trick, and you care about retrieval-stage recall more than precise within-top-K ordering. It is the right default for a first production collaborative-filtering model at scale. Its confidence weighting is a genuine, built-in advantage for graded signal. Do **not** reach for it if your metric is order-sensitive and tight (it does not optimize ranking) or if you need to fold in content features (vanilla iALS is collaborative-only; you would move to a hybrid).

**Reach for BPR when** you want the canonical, well-understood ranking baseline, your signal is essentially binary (interacted or not), you are training a custom model (a two-tower, a sequence model, a graph model) and need a loss that optimizes order — BPR is the loss you bolt on. It is also the right thing to *implement from scratch* once, so you understand pairwise ranking in your bones. Do **not** reach for vanilla BPR with uniform random negatives if you can afford hard negatives — you will waste most of your gradient steps on easy pairs, and a better sampler (or WARP) will beat it for the same wall-clock budget.

**Reach for WARP when** your headline metric is a top-K quantity (Recall@K, Precision@K, NDCG@K), you can afford slower training, and especially when you are already using LightFM for its hybrid content-feature support. WARP's rank-weighting is the closest of the three to optimizing what the product serves. Do **not** reach for WARP at the very largest scale where its per-step sampling search becomes the bottleneck, or when training time is your binding constraint — iALS will be many times faster for a comparable retrieval-stage result.

And the meta-rule: **all three produce dot-product embeddings, so the choice of objective is decoupled from how you serve.** Whichever you pick, the artifact is two embedding tables, and serving is maximum-inner-product nearest-neighbor search through `faiss`/`hnswlib`/ScaNN. That means you can A/B-test iALS vs BPR vs WARP behind the *same* retrieval infrastructure by swapping the item-vector index — a cheap experiment that the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) recommends running before you commit to one.

## 16. Stress tests: where these models bite back

**Only implicit feedback, no graded signal.** If all you have is binary interacted/not, iALS's $\alpha$ loses most of its leverage (every positive has the same $r_{ui}=1$), and the model is essentially weighting positives uniformly higher than blanks. This is fine, but it narrows the gap to BPR — with binary data, the pointwise-vs-pairwise distinction matters more than the confidence weighting, and BPR/WARP's direct ranking objective tends to pull ahead on top-K. Reach for the pairwise methods when the signal is flat.

**100M items.** iALS's per-sweep cost is linear in nonzeros and the $k^3$ solve per user/item, so it scales — but at 100M items, the *full dense scoring* for evaluation and the ANN index build become the bottlenecks, not training. BPR's bootstrap sampling is fine, but uniform negatives over 100M items are almost all trivially easy, so the effective signal per step collapses; you *must* move to harder sampling. WARP's sampling search gets very expensive because finding a violating negative among 100M mostly-easy candidates can take many tries. At extreme scale the field moves to two-tower models with in-batch and mined negatives — but the *losses* are still pairwise/softmax descendants of exactly these ideas.

**Negatives are mostly false negatives.** When your catalog is small relative to user interests (a user would genuinely like a large fraction of items), uniform negative sampling frequently draws items the user would actually engage with — false negatives — and you penalize the model for ranking a true positive highly. iALS is somewhat robust here because it weights blanks weakly (confidence 1) rather than asserting a hard negative; BPR is more exposed because each sampled negative is treated as a confident "ranks below the positive." The mitigations — exposure-aware sampling, popularity-corrected sampling, debiasing — are the subject of dedicated work; the takeaway is that the harder you mine negatives, the more false negatives you mine, so there is a real ceiling on how aggressive the sampler should be.

**Offline NDCG rises but online is flat.** The classic trap. You tuned WARP to a beautiful offline NDCG@10 and shipped it, and engagement did not move. Likely causes: your offline test set is itself MNAR (you can only evaluate on items that were shown, so you are scoring the model on the *old* system's exposure distribution); position bias means the offline "positive" was partly caused by where the old system placed the item, not pure preference; and the new model's recommendations are out-of-distribution for your logged data, so offline metrics cannot see their true value. This is not a failure of iALS or BPR specifically — it is the [offline-online gap](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys), and the fix is counterfactual/IPS evaluation and, ultimately, an online A/B test. Never ship a recommender on offline NDCG alone.

**Train-serve skew in the interaction matrix.** A subtle one: if you build the training matrix with one definition of "interaction" (say, any rating) but the serving-time filter of "already seen" uses a different definition (say, only purchases), you will recommend items the user already engaged with under the training definition, tanking perceived quality. Keep the positive-set definition identical between training-matrix construction, the negative sampler's exclusion set, and the serve-time seen-filter. This is the recommender version of feature skew, and it silently halves precision.

## 17. Key takeaways

- **Implicit feedback is positive-only and MNAR**, so you cannot minimize error on observed cells (they are all positive) — you must say something about the unobserved cells, and how you say it *is* the model choice.
- **iALS keeps regression but fixes the labels**: binary preference $p_{ui}$ plus confidence $c_{ui} = 1 + \alpha r_{ui}$, weighted least squares over *all* cells, solved in closed form by alternating ridge regressions.
- **The $Y^\top Y$ precompute is why iALS scales**: compute the shared Gram matrix once per sweep, add a per-user correction over only clicked items, and each user solve costs $O(k^2 |\mathcal{I}_u^+| + k^3)$ — linear in nonzeros, not in the full matrix.
- **BPR optimizes order, not score**: maximize $\sum \ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})$ over (positive, sampled-negative) pairs, which is a smooth surrogate for AUC; its gradient self-weights toward the pairs it is currently getting wrong.
- **The BPR update is geometric and intuitive**: push the user toward the positive and away from the negative, scaled by the probability of being wrong — invariant to absolute score, sensitive only to order.
- **WARP is the rank-aware middle ground**: pairwise like BPR but it samples until it finds a violating negative and weights by estimated rank, bending the optimization toward the top of the list (Recall@K) at higher per-step cost.
- **Choose by signal and metric**: iALS for graded signal, speed, and scale; BPR as the canonical baseline and the loss you bolt onto custom models; WARP when top-K is the headline metric and you can afford it.
- **All three emit dot-product embeddings**, so the objective is decoupled from serving — A/B-test them behind the same ANN retrieval index.
- **The $\alpha$ knob (iALS) and the negative sampler (BPR) are analogous souls**: both decide how strongly the model learns from non-interactions; harder mining gives stronger gradients but risks false negatives.
- **Beat popularity on a temporal split with full-catalog ranking, then validate online** — offline NDCG gains are necessary but not sufficient because of MNAR, position bias, and distribution shift.

## 18. Further reading

- Yifan Hu, Yehuda Koren, Chris Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008 — the weighted-ALS / iALS paper, including the $Y^\top Y$ precompute.
- Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme, *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009 — the BPR objective, gradient, and the AUC connection.
- Jason Weston, Samy Bengio, Nicolas Usunier, *WSABIE: Scaling Up to Large Vocabulary Image Annotation*, IJCAI 2011 — the origin of the WARP loss that LightFM adapts.
- Maciej Kula, *Metadata Embeddings for User and Item Cold-start Recommendations* (LightFM), 2015, plus the official `lightfm` and `implicit` library docs for the real APIs.
- Walid Krichene, Steffen Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020 — why sampled top-K metrics can reorder models, and how to evaluate honestly.
- Within this series: [implicit vs explicit feedback and the data you have](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have), [matrix factorization, the workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), [pairwise and BPR loss deep dive](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive), [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies), and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

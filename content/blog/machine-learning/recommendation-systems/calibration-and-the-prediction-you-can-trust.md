---
title: "Calibration: The Prediction You Can Trust"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why a ranker's probabilities, not just its order, have to be trustworthy: what calibration means, the negative-downsampling correction derived from scratch, Platt and isotonic and temperature scaling, ECE and reliability diagrams in runnable code, and a worked eCPM example showing how miscalibration quietly burns budget."
tags:
  [
    "recommendation-systems",
    "recsys",
    "calibration",
    "ctr-prediction",
    "ranking",
    "ece",
    "isotonic-regression",
    "platt-scaling",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/calibration-and-the-prediction-you-can-trust-1.png"
---

The page that taught me what calibration is did not say "calibration." It said "spend pacing blew through the daily budget by 11 a.m. and the campaigns that ran are not converting." Nothing was down. Retrieval was healthy, the ranker was returning a score for every candidate, latency was nominal, and offline AUC had actually ticked *up* a thousandth after the overnight retrain. The classifier was, by every dashboard we had, better than yesterday. What had broken was invisible: the retrain had been trained on freshly rebalanced data — we had down-sampled the non-clicks four-to-one to speed up training — and nobody had re-fit the correction that maps the model's inflated training-time probability back to the real click rate. So the model now believed every impression was about four times more likely to be clicked than it actually was. The ads auction read those inflated probabilities literally, computed an expected value (`bid × pCTR`) that was four times too high, and bid the house budget on impressions that would never pay off. The order of the candidates was fine. The *numbers* were a lie, and the auction took them at their word.

That is the whole subject of this post. A ranker does two separable things, and most people conflate them. It **discriminates** — it puts the items a user will click above the items they won't, and that is what AUC measures. And it **calibrates** — when it says "probability 0.3," roughly 30% of those items really do get clicked, and that is what calibration measures. For a pure top-K feed where you only sort and show the top few, discrimination is all you need; the absolute value of the probability is thrown away the moment you sort. But the instant any downstream consumer reads the probability as a probability — an ads auction bidding `bid × pCTR`, a multi-objective ranker blending $p_{\text{click}}$ with $E[\text{watch}]$ and $p_{\text{share}}$, a budget pacer estimating how much of today's spend a campaign will eat, or a UI promising the user a "92% match" — the number has to be *true*, and a model with a beautiful AUC can be catastrophically wrong about it.

![A before and after comparison of a miscalibrated reliability curve that bends away from the diagonal against a calibrated one whose predicted probability matches the observed click rate](/imgs/blogs/calibration-and-the-prediction-you-can-trust-1.png)

This sits squarely in the ranking stage of the funnel you have followed through this series: retrieval scans millions of items and hands a few hundred candidates to ranking; the ranker scores them and emits a probability per candidate; re-ranking and the downstream auction or fusion consume those probabilities. The score the ranker emits is the contract. By the end of this post you will be able to define calibration precisely and tell it apart from discrimination; read a reliability diagram; compute Expected Calibration Error (ECE) by hand and in code; derive the negative-downsampling correction $q = \frac{p}{p + (1-p)/w}$ from Bayes' rule; fit Platt scaling, isotonic regression, and temperature scaling with `scikit-learn` and PyTorch; measure ECE before and after on a Criteo-style CTR model; and compute, in dollars, what a miscalibrated probability costs an ad auction. If you want the foundations of the ranker we are calibrating, start with [the ranking model: CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations); the multi-objective fusion that *requires* calibration is in [multi-task and multi-objective ranking: MMoE and PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple); the top-level map is [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system); and the synthesis is the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

## 1. Two different jobs: discrimination and calibration

Start with the cleanest possible statement of what a probabilistic ranker is supposed to do, because the entire post hinges on noticing that it is two statements, not one.

The first job is **discrimination**, also called ranking quality or ordering. Given any two items, one that will be clicked and one that won't, does the model score the clicked one higher? If you sample a random clicked impression and a random non-clicked impression, the probability the model scores the clicked one above the non-clicked one is exactly the **AUC** (area under the ROC curve). An AUC of 0.5 is a coin flip; 1.0 is perfect ordering. Critically, AUC is *invariant to any monotonic transformation of the scores*. If I take every prediction and square it, or run it through any strictly increasing function, the order is unchanged and the AUC is byte-for-byte identical. AUC literally cannot see the absolute level of the probabilities. It only sees rank.

The second job is **calibration**. Among all the impressions where the model said "probability 0.3," what fraction were actually clicked? If the answer is 0.3, the model is calibrated at that level. If 30% of the "0.3" predictions click but only 12% of the "0.5" predictions click, the model is both miscalibrated *and* mis-ordered, but those are different failures. Formally, a model with prediction $\hat{p}(x)$ is perfectly calibrated if

$$ P\big(y = 1 \;\big|\; \hat{p}(x) = p\big) = p \quad \text{for all } p \in [0, 1]. $$

Read that literally: condition on the model having output the value $p$, and the true click rate within that group must equal $p$. This is a statement about the *level* of the probability, and it is exactly the thing AUC throws away.

These two properties are orthogonal. You can have either without the other:

- **High AUC, terrible calibration.** A deep net that ranks items beautifully but outputs probabilities all squashed toward 0 and 1 (overconfident). It will win every offline ranking metric and lose money in an auction.
- **Perfect calibration, useless AUC.** A model that outputs the global base rate (say 0.026, the dataset's average click rate) for *every* item. Among all its "0.026" predictions, exactly 2.6% click — perfectly calibrated! — but its AUC is 0.5 because it cannot tell any two items apart. A constant predictor is the trivially-calibrated, zero-discrimination extreme.

![A matrix contrasting AUC against ECE on what each metric measures and what each one misses with a high-AUC low-calibration case shown as a failure row](/imgs/blogs/calibration-and-the-prediction-you-can-trust-2.png)

The reason this distinction matters so much in recommenders specifically is that the *output of the ranker is consumed in two completely different ways depending on the surface*, and which way decides whether you can ignore calibration or whether it is the whole ballgame.

If the surface is a pure ranked feed — show the user the top 10 items by score — then only the order is ever used. The scores get sorted, the top few are shown, the absolute values are discarded. Here, calibration is irrelevant. A model that is great at ordering and wildly overconfident produces the exact same feed as a perfectly calibrated version of itself, because sorting is invariant to monotonic rescaling. This is why a lot of recommendation literature, especially the retrieval-and-ranking-for-feeds tradition, optimizes ordering metrics (NDCG, Recall@K) and never mentions calibration. For that problem they are right to.

But the moment the probability is *used as a number*, calibration becomes load-bearing. I will spend the next section on exactly when that happens, because getting this judgment right — "do I actually need a calibrated probability, or just a good ranking?" — is the most valuable thing in this post. It saves you from both failures: shipping an uncalibrated model into an auction (expensive), and over-engineering calibration onto a feed that never reads the number (wasted effort).

## 2. When you need a calibrated probability (and when order is enough)

Here is the decision, stated plainly, with the four canonical cases where the probability is consumed as a number.

**Case 1 — Ads bidding.** In a second-price or first-price ad auction, the system ranks ads not by predicted CTR but by **expected revenue per impression**, the eCPM (effective cost per mille):

$$ \text{eCPM} = \text{bid} \times \hat{p}_{\text{CTR}} \times 1000. $$

The advertiser bids per click; you get paid only if the user clicks; so the expected value of showing an ad is the bid times the probability of a click. The auction sorts ads by this product. Now notice: if all your $\hat{p}_{\text{CTR}}$ values are inflated by the *same* constant factor, the *ordering* of ads within a single auction is unchanged (you multiplied every term by the same constant), so you might think calibration is irrelevant here too. It is not, for two reasons. First, the bids differ across ads, so a uniform multiplicative error on pCTR does *not* uniformly scale eCPM — `bid_A × 4·p_A` versus `bid_B × 4·p_B` reorders relative to the truth whenever the per-ad error is not uniform, which it never is. Second, and more importantly, **budget pacing and floor prices use the absolute eCPM**: if your eCPM is four times too high, the pacer thinks each impression is worth four times what it is, blows the budget early, and the reserve-price logic clears auctions it should have skipped. That is the exact failure that paged me.

**Case 2 — Multi-objective fusion.** A modern feed ranker does not predict one thing; it predicts several — probability of click, expected watch time, probability of share, probability of a "not interested" — from a multi-task model (the MMoE/PLE architecture covered in [the multi-task ranking post](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple)). To produce a single ranking you fuse them, typically as a weighted product or weighted sum of the heads:

$$ \text{score} = \hat{p}_{\text{click}}^{\,w_1} \cdot \mathbb{E}[\text{watch}]^{\,w_2} \cdot \hat{p}_{\text{share}}^{\,w_3}. $$

This formula is only meaningful if each head is on its true scale. If $\hat{p}_{\text{click}}$ is inflated 4× and $\hat{p}_{\text{share}}$ is deflated 2×, the fusion silently overweights clicks and underweights shares relative to the weights $w_1, w_2, w_3$ you carefully tuned — your business intent ("we value a share as much as five clicks") is corrupted by an artifact of how each head happened to be miscalibrated. You literally *cannot* combine heads of different objectives unless each is calibrated to its true expectation, because addition and multiplication of incommensurable scales is meaningless. Calibration is the common currency that makes fusion legal.

**Case 3 — Budget pacing and forecasting.** Any system that forecasts "how many clicks will this campaign get today?" or "how much of this budget will we spend by 6 p.m.?" integrates predicted probabilities over impressions. $\sum_i \hat{p}_i$ is an estimate of the expected click count, and it is only correct if the $\hat{p}_i$ are calibrated. A miscalibrated model gives a biased forecast, and pacing decisions built on a biased forecast are wrong from the first impression.

**Case 4 — Showing the number to a human.** "92% match." "Highly likely to enjoy." A 4.5-star predicted rating. If you surface the probability or a transform of it directly to the user, calibration is the difference between an honest interface and a deceptive one. A "95% match" that pans out 60% of the time trains users to distrust your product.

![A before and after comparison showing a ranking-only top-K feed where order alone suffices against ads bidding and multi-objective fusion and budget pacing where calibration is required](/imgs/blogs/calibration-and-the-prediction-you-can-trust-7.png)

The contrast on the other side is the pure feed: a personalized home feed, a "more like this" rail, a search result page where you show the top results in order. There, the consumer of the model output is a sort, and a sort discards the scale. **If your only consumer is a ranking, do not spend a sprint on calibration — spend it on discrimination (better features, better loss, hard negatives).** Calibrating a feed-only model is not wrong, but it is effort spent on a property nobody reads.

A subtlety worth flagging: even in a feed, calibration sneaks back in if you do anything *across* requests with the scores — a global threshold ("only show items above 0.4 probability of relevance"), a confidence-gated UI ("show a 'recommended' badge above 0.7"), or a diversity re-ranker that trades off predicted relevance against predicted dissimilarity on a shared scale. Any cross-request comparison of raw scores assumes a calibrated scale. So the honest rule is: *if the probability is ever compared to a constant, to another model's probability, or to a different objective's prediction, you need calibration. If it is only ever compared to other items' scores in the same request and then sorted, you do not.*

#### Worked example: an eCPM error from miscalibration

Two ads compete for one slot. Ad A has advertiser bid \$2.00 per click and true CTR 0.02. Ad B has bid \$0.50 per click and true CTR 0.10. The honest expected revenues are:

- Ad A: $\text{eCPM}_A = 2.00 \times 0.02 \times 1000 = \$40.00$ per mille.
- Ad B: $\text{eCPM}_B = 0.50 \times 0.10 \times 1000 = \$50.00$ per mille.

Ad B should win — it is worth \$50 per thousand impressions versus A's \$40. Now suppose the model is miscalibrated in a *non-uniform* way (which real miscalibration always is): it overpredicts low-CTR ads and underpredicts high-CTR ads, a classic symptom of overconfidence pushing predictions toward the extremes. Say it reports $\hat{p}_A = 0.035$ (inflated) and $\hat{p}_B = 0.07$ (deflated). The auction now computes:

- Ad A: $2.00 \times 0.035 \times 1000 = \$70.00$.
- Ad B: $0.50 \times 0.07 \times 1000 = \$35.00$.

The auction picks **Ad A**. But A's *true* eCPM is \$40 and B's is \$50, so you just showed the ad worth \$40 instead of the one worth \$50 — a **\$10 per-mille opportunity cost, 20% of the slot's true value**, on every auction this miscalibration touches. At a billion impressions a day, a 20% mis-allocation on the marginal slot is not a rounding error; it is a line item. And notice no uniform rescaling fixes this: the error reordered the ads because it was *direction-dependent*, which is precisely what miscalibration in deep CTR models looks like.

## 3. What calibration is, formally — reliability diagrams and ECE

We need a way to *see* and *measure* calibration, because "the probabilities are off" is not actionable. The two tools are the reliability diagram (to see it) and ECE (to score it).

A **reliability diagram** is built by binning predictions. Take every prediction $\hat{p}_i$ on a held-out set, partition the interval $[0,1]$ into $M$ bins (say 10 bins of width 0.1), drop each prediction into its bin, and for each bin plot two numbers: the **average predicted probability** in the bin (x-axis) against the **observed click frequency** in the bin (y-axis, the fraction of impressions in that bin that were actually clicked). A perfectly calibrated model lies exactly on the diagonal $y = x$: in the bin where the average prediction is 0.3, exactly 30% click. Points *below* the diagonal mean the model is overconfident (it predicted more than reality delivered); points above mean underconfident. The miscalibrated curve in the figure at the top of this post bows below the diagonal in the high-probability region — the textbook overconfidence signature of a deep net.

To turn the picture into a single number, define **Expected Calibration Error (ECE)**. With $M$ bins, let $B_m$ be the set of indices whose predictions fall in bin $m$, let $\text{acc}(B_m)$ be the observed click rate in that bin (the fraction with $y=1$), and let $\text{conf}(B_m)$ be the average predicted probability in that bin. Then:

$$ \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \,\big|\, \text{acc}(B_m) - \text{conf}(B_m) \,\big|. $$

In words: for each bin, take the absolute gap between what the model predicted on average and what actually happened, weight it by how many samples landed in that bin, and sum. ECE is a sample-weighted average vertical distance from the diagonal in the reliability diagram. ECE of 0 is perfect; 0.1 means that, on average, predictions are off by 10 percentage points in absolute click rate. The closely related **Maximum Calibration Error (MCE)** takes the worst bin instead of the weighted average:

$$ \text{MCE} = \max_{m} \,\big|\, \text{acc}(B_m) - \text{conf}(B_m) \,\big|. $$

MCE is the number you care about when the *worst-case* mistake is what hurts — high-stakes, low-volume bins (your highest-bid ads, your most confident "92% match" claims). ECE is the average-case health metric; MCE is the tail-risk metric. Watch both.

#### Worked example: ECE by hand on a small binned set

Suppose a held-out set of 1,000 impressions, binned into five bins. For each bin I record: count, average predicted probability (conf), and observed click rate (acc).

| Bin | Count | conf (avg pred) | acc (observed) | gap $\lvert\text{acc}-\text{conf}\rvert$ | weight $\frac{\lvert B_m\rvert}{n}$ | contribution |
|-----|-------|-----------------|----------------|------|--------|--------------|
| 1 | 500 | 0.05 | 0.02 | 0.03 | 0.500 | 0.0150 |
| 2 | 250 | 0.20 | 0.12 | 0.08 | 0.250 | 0.0200 |
| 3 | 150 | 0.40 | 0.28 | 0.12 | 0.150 | 0.0180 |
| 4 | 70 | 0.65 | 0.50 | 0.15 | 0.070 | 0.0105 |
| 5 | 30 | 0.85 | 0.70 | 0.15 | 0.030 | 0.0045 |

Sum the contributions: $0.0150 + 0.0200 + 0.0180 + 0.0105 + 0.0045 = 0.068$. So **ECE = 0.068** — on average the predictions are off by about 6.8 percentage points. The **MCE = 0.15**, the worst gap, occurring in bins 4 and 5. And notice the shape: every bin's `conf` exceeds its `acc`, so the model is systematically overconfident across the board — the curve sits entirely below the diagonal. That uniform direction is the fingerprint of overconfidence (and, as we will see, of negative downsampling).

Two warnings about ECE that practitioners learn the hard way. First, **ECE depends on the binning scheme.** Too few bins and you average away real miscalibration (a bin spanning 0.0–0.5 can hide a model that is fine at 0.05 and awful at 0.45); too many bins and each bin has too few samples, so the observed `acc` is a noisy estimate of the true rate and ECE is inflated by sampling noise. This is a genuine bias-variance trade-off in the *measurement itself*, which I derive in the binning section. The common fix is **equal-mass binning** (a.k.a. quantile or adaptive binning): instead of equal-width bins, make each bin contain the same number of samples, so the noise per bin is roughly constant. In CTR data — where the vast majority of predictions cluster at very low probabilities — equal-width bins put almost everything in bin 1 and leave the high bins empty and noisy; equal-mass binning is almost always the right choice for recommenders.

Second, **logloss is a proper scoring rule and ECE is not.** This matters and I will make it precise in the science section, but the headline: logloss (a.k.a. binary cross-entropy) is *minimized in expectation only by the true probability*, so it rewards both good discrimination and good calibration simultaneously, which is why it is the right *training* loss. ECE is a *diagnostic* — it can be gamed (a constant predictor at the base rate has ECE near zero with AUC 0.5), so you never optimize it directly; you optimize logloss and *monitor* ECE. AUC, in turn, is not a proper scoring rule either — it ignores the level entirely. The trio you actually watch in production: **AUC for ordering, logloss for the joint, ECE/MCE for calibration specifically.**

## 4. The science: why logloss is proper and AUC is not, and the downsampling correction

This section is the rigorous core. Three results: why logloss is a proper scoring rule, why AUC is not, and the derivation of the negative-downsampling correction.

### 4.1 Logloss is a proper scoring rule

A scoring rule $S(\hat{p}, y)$ assigns a penalty to a prediction $\hat{p}$ given an outcome $y$. It is **proper** if, when the true probability of $y=1$ is $\pi$, the expected score is minimized by reporting $\hat{p} = \pi$. In other words, an honest model has no incentive to lie. Logloss is

$$ \ell(\hat{p}, y) = -\big[\, y \log \hat{p} + (1-y)\log(1-\hat{p}) \,\big]. $$

Fix the true probability $\pi$. The expected logloss of reporting some value $\hat{p}$ is

$$ \mathbb{E}_{y \sim \text{Bernoulli}(\pi)}[\ell(\hat{p}, y)] = -\pi \log \hat{p} - (1-\pi)\log(1-\hat{p}). $$

Differentiate with respect to $\hat{p}$ and set to zero:

$$ \frac{d}{d\hat{p}}\Big[ -\pi \log \hat{p} - (1-\pi)\log(1-\hat{p}) \Big] = -\frac{\pi}{\hat{p}} + \frac{1-\pi}{1-\hat{p}} = 0. $$

Solving: $\pi(1-\hat{p}) = (1-\pi)\hat{p} \implies \pi - \pi\hat{p} = \hat{p} - \pi\hat{p} \implies \hat{p} = \pi$. The second derivative is positive, so it is a minimum. The expected logloss is uniquely minimized at $\hat{p} = \pi$, the true probability. That is what "proper" means, and it is *why* a model trained to minimize logloss is, in the infinite-data well-specified limit, both well-ordered and well-calibrated. Logloss is the calibration-aware loss; minimizing it is the first line of defense against miscalibration, and many calibration problems are really "we did not actually minimize logloss on the serving distribution" problems.

### 4.2 AUC is not a proper scoring rule

AUC equals the probability that a random positive scores above a random negative:

$$ \text{AUC} = P\big(\hat{p}(x^+) > \hat{p}(x^-)\big), \quad x^+ \sim \text{positives}, \; x^- \sim \text{negatives}. $$

This is a statement purely about the *ranking* of scores. Apply any strictly increasing function $g$ to every prediction — $g(\hat{p}) = \hat{p}^3$, or $g(\hat{p}) = \sigma(10\,\sigma^{-1}(\hat{p}))$ which makes the model wildly overconfident — and because $g$ preserves order, $\hat{p}(x^+) > \hat{p}(x^-) \iff g(\hat{p}(x^+)) > g(\hat{p}(x^-))$, so AUC is unchanged. But the calibration is destroyed. Therefore AUC cannot be a proper scoring rule: it is invariant to transformations that change the probability level, so it cannot be uniquely minimized (or maximized) by the true probability. This is the formal version of "great AUC, terrible calibration." It is also why you must never declare victory on a ranker from AUC alone if anything downstream reads the probability.

### 4.3 The negative-downsampling correction

This is the single most common source of miscalibration in CTR systems, and the fix is a clean application of Bayes' rule, so it deserves a careful derivation.

The problem: real CTR data is wildly imbalanced — click rates of 1–5% mean 20–100 negatives per positive. Training on all of it is slow and the gradient is dominated by easy negatives. The standard trick is **negative downsampling**: keep all positives, but keep each negative independently with probability $w$ (the *sampling rate*, e.g. $w = 0.1$ keeps one in ten negatives). This rebalances the data — a 2% click rate with $w=0.1$ becomes roughly $\frac{0.02}{0.02 + 0.98 \times 0.1} = 0.169$, about a 17% click rate in the training set — which trains faster and gives the optimizer a healthier gradient. But the model now learns $P(\text{click} \mid x, \text{downsampled data})$, not $P(\text{click} \mid x, \text{true traffic})$, so its probabilities are systematically *too high*. We need the map from the training probability $p$ back to the true probability $q$.

Set up the Bayes' rule carefully. Let $y=1$ denote click, $y=0$ denote non-click, and let $s=1$ denote "this example was kept in the (down-sampled) training set." All positives are kept, so $P(s=1 \mid y=1) = 1$. Negatives are kept with probability $w$, so $P(s=1 \mid y=0) = w$. The model trained on the kept data learns

$$ p = P(y=1 \mid s=1, x). $$

We want $q = P(y=1 \mid x)$, the true probability on full traffic. By Bayes' rule, conditioning everything on $x$ (suppressed for brevity):

$$ p = P(y=1 \mid s=1) = \frac{P(s=1 \mid y=1)\,P(y=1)}{P(s=1 \mid y=1)\,P(y=1) + P(s=1 \mid y=0)\,P(y=0)}. $$

Substitute $P(s=1\mid y=1)=1$, $P(s=1\mid y=0)=w$, $P(y=1)=q$, $P(y=0)=1-q$:

$$ p = \frac{1 \cdot q}{1 \cdot q + w(1-q)} = \frac{q}{q + w(1-q)}. $$

Now invert to get $q$ in terms of $p$. Cross-multiply: $p\,[q + w(1-q)] = q$, so $pq + pw - pwq = q$, giving $pw = q - pq + pwq = q(1 - p + pw)$, hence

$$ q = \frac{pw}{1 - p + pw} = \frac{pw}{1 - p(1-w)}. $$

It is cleaner to write the equivalent form that exposes the structure (divide numerator and denominator by $w$):

$$ \boxed{\,q = \frac{p}{p + (1-p)/w}\,}. $$

Sanity checks. If $w=1$ (no downsampling), $q = \frac{p}{p + (1-p)} = p$ — the correction is the identity, as it must be. As $w \to 0$ (extreme downsampling), $q \to 0$ for any $p < 1$ — the more aggressively you down-sampled negatives, the more you must deflate the predicted probability. And the correction is monotone in $p$, so it does *not* change the ranking — applying it leaves AUC byte-for-byte identical, while restoring the true probability level. That is the ideal calibration move: it touches only the level, never the order.

![A directed graph showing negative downsampling deep model overconfidence distribution shift and multi-task heads all merging into a biased logit and then a wrong probability](/imgs/blogs/calibration-and-the-prediction-you-can-trust-4.png)

A practical note on *where* to apply it. You can apply the correction in probability space (the boxed formula on $p$), or — usually cleaner numerically — as an additive shift on the logit. Since $p = \sigma(z)$, the correction $q = \frac{p}{p + (1-p)/w}$ works out to a simple bias added to the logit: $z_{\text{corrected}} = z + \log w$. That is the form to remember: **downsampling negatives at rate $w$ shifts the trained logit up by $-\log w$, so you correct by adding $\log w$ (a negative number) back.** A one-line bias correction, derived from Bayes' rule, that would have saved my 3 a.m. page.

### 4.4 Decomposing logloss into calibration and refinement

There is one more piece of theory that makes the orthogonality of calibration and discrimination not just an empirical observation but an algebraic identity, and it is worth seeing because it tells you exactly *which part of your loss* a calibrator can and cannot improve. Take the **Brier score**, the mean squared error of a probabilistic prediction, $\text{BS} = \frac{1}{n}\sum_i (\hat{p}_i - y_i)^2$ (it is a proper scoring rule too, and easier to decompose than logloss while telling the same story). Group the predictions by their value, indexing groups by the distinct predicted value $p_k$, with $n_k$ samples in group $k$ and observed click rate $\bar{y}_k$ in that group. The Murphy (1973) decomposition splits the Brier score into three terms:

$$ \text{BS} = \underbrace{\frac{1}{n}\sum_k n_k (p_k - \bar{y}_k)^2}_{\text{calibration (reliability)}} \;-\; \underbrace{\frac{1}{n}\sum_k n_k (\bar{y}_k - \bar{y})^2}_{\text{refinement (resolution)}} \;+\; \underbrace{\bar{y}(1-\bar{y})}_{\text{irreducible (uncertainty)}}, $$

where $\bar{y}$ is the overall base rate. Read each term. The first is the **calibration** term — it is, up to squaring instead of absolute value, exactly ECE: the average squared gap between what you predicted ($p_k$) and what happened ($\bar{y}_k$) within each group. The second is **refinement** (also called resolution) — how far each group's observed rate sits from the global base rate, which rewards a model for *separating* impressions into groups with genuinely different click rates; this is the discrimination term, the cousin of AUC. The third is **irreducible** uncertainty, the variance of the label that no model can remove.

Now the key consequence: a post-hoc calibrator is a *monotone* (or near-monotone) map applied to the scores. It can shrink the calibration term toward zero — that is its entire job. But because it preserves the grouping of impressions (it does not move any impression from one rank position past another, except for isotonic's ties), it leaves the refinement term essentially untouched. *That is the algebra behind "ECE drops 15×, AUC is unchanged."* Calibration improves the first term; refinement, which AUC tracks, is fixed by the calibrator's monotonicity. If you want better refinement you need a better *model* — more features, a richer architecture, hard negatives — not a calibrator. This decomposition is the precise reason calibration and discrimination are separate sprints with separate tools, and it is why you should never expect a calibrator to rescue a model with weak refinement: it can only make an already-discriminating model *honest*, not make a non-discriminating model *useful*.

## 5. Sources of miscalibration

Before fixing calibration, name the things that break it, because the fix depends on the cause. There are five common sources in production recommenders, and the figure above shows them merging into one wrong probability.

**1. Negative downsampling (the most common in CTR).** Exactly the section-4 problem: you trained on rebalanced data, so the model's probabilities are inflated by a factor you must analytically undo. This is the *good* kind of miscalibration because it is fully characterized — you know $w$, so $q = \frac{p}{p+(1-p)/w}$ corrects it exactly with no fitting. Always do this correction first; it removes the dominant, known bias and leaves the residual, harder-to-characterize miscalibration for a learned calibrator to mop up.

**2. Deep model overconfidence.** The Guo et al. 2017 paper "On Calibration of Modern Neural Networks" is the canonical reference: modern deep nets, unlike the shallow models of the 2000s, are systematically *overconfident*. As networks got deeper, wider, and trained longer with batch norm and without strong regularization, their accuracy improved but their calibration got *worse* — they push probabilities toward 0 and 1 harder than the data justifies. The intuition is that minimizing logloss to convergence on a high-capacity model that can drive training loss near zero rewards extreme confidence on the training set, and that overconfidence does not generalize. The fix here is a *learned* calibrator (temperature scaling is Guo et al.'s recommendation; more below), not an analytic one, because the distortion is data-dependent and not characterized by a single known parameter.

**3. Distribution shift.** The model was calibrated on last month's traffic; this week's traffic has a different click-rate base because of a seasonal effect, a UI change, or a new traffic source. Calibration is a property *relative to a distribution*, so when the distribution moves, calibration drifts even if the model is frozen. The fix is to *re-fit the calibrator frequently* on recent held-out data — calibration is cheap to re-fit (it is a 1-D fit), so re-fit it daily or hourly even if you retrain the base model weekly. This is also why you log the recent ECE on a dashboard: rising ECE with stable AUC is the unambiguous signal of calibration drift from shift.

**4. Multi-task heads.** A shared-bottom or MMoE model with several heads (click, watch, share) is trained on a joint loss, and the heads can interfere: the shared representation is pulled by all tasks, and a head with a small loss weight or a rare label can come out miscalibrated even though the click head is fine. Each head needs its *own* calibrator, fit on that head's label, before the heads are fused. This is the connective tissue between this post and [the multi-task ranking post](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple): you cannot fuse heads until each is independently calibrated.

**5. The softmax-temperature effect.** Whenever you have a softmax over candidates (sampled-softmax retrieval, a multi-class head), the *temperature* of the softmax directly controls the sharpness of the output distribution. A temperature below 1 makes the distribution peaky (overconfident); above 1 flattens it. If the temperature used at training (often tuned for ranking, or absorbed into the loss) does not match the temperature that gives calibrated probabilities, the outputs are systematically over- or under-confident. This is precisely why **temperature scaling** — fitting a single scalar $T$ to the logits — is such an effective and minimal calibrator for these models: it directly targets the one knob that controls confidence.

The point of enumerating these is that the right fix is cause-specific. Downsampling → the analytic correction. Overconfidence and softmax-temperature → temperature scaling. Residual nonlinearity → Platt or isotonic. Shift → re-fit frequently. Multi-task → per-head calibrators. A production calibration stack usually chains them: analytic downsampling correction first, then a learned calibrator on top of the residual, re-fit on recent data, one per head.

## 6. The fixing toolkit: correction, Platt, isotonic, temperature, binning, per-segment

Now the methods. The taxonomy below organizes them into three families by *how* they map a miscalibrated score to a calibrated one: analytic (invert a known distortion), parametric (fit a low-dimensional curve), and non-parametric (learn a flexible monotone map from bins).

![A tree taxonomy of calibration methods splitting into parametric methods like Platt and temperature scaling non-parametric methods like isotonic and histogram binning and the analytic downsampling correction](/imgs/blogs/calibration-and-the-prediction-you-can-trust-3.png)

### 6.1 The downsampling correction (analytic)

Covered in section 4. Closed-form, no fitting, exact for the downsampling distortion, order-preserving. Apply it first, always, whenever you down-sampled negatives. Code:

```python
import numpy as np

def downsample_correction(p, w):
    """Map a probability learned on negatives down-sampled at rate w
    back to the true-traffic probability. q = p / (p + (1-p)/w)."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return p / (p + (1.0 - p) / w)

# Equivalent, numerically nicer, as a logit shift: z_true = z_train + log(w)
def downsample_correction_logit(z, w):
    return z + np.log(w)
```

### 6.2 Platt scaling (parametric)

**Platt scaling** fits a one-dimensional logistic regression *on the model's logits* (or scores). Introduced by Platt (1999) to turn SVM margins into probabilities, it learns two scalars $a, b$ such that

$$ \hat{q} = \sigma(a \cdot z + b), $$

where $z$ is the uncalibrated logit and $\sigma$ is the sigmoid. You fit $a, b$ by minimizing logloss on a held-out calibration set against the true labels. It is parametric (just two parameters), so it is data-efficient and low-variance — it works with a few hundred to a few thousand calibration points — but it can only correct distortions that a single sigmoid-of-affine can express. If the true reliability curve is sigmoidal (the common overconfidence shape), Platt nails it; if it has a non-monotone wiggle, Platt cannot follow it. In `scikit-learn` this is `CalibratedClassifierCV(method="sigmoid")`.

### 6.3 Temperature scaling (parametric, the minimal case)

**Temperature scaling** is Platt scaling with $a = 1/T$ and $b = 0$ — a *single* parameter:

$$ \hat{q} = \sigma(z / T). $$

A temperature $T > 1$ softens (de-confidences) the predictions; $T < 1$ sharpens them. Guo et al. 2017 showed that for modern overconfident deep nets, temperature scaling alone — the single scalar $T$ fit by minimizing logloss (or NLL) on a held-out set — recovers excellent calibration *without changing the predicted class or the ordering at all*, because dividing every logit by the same positive constant is monotone. It is the recommended default for deep-net calibration precisely because it is the lowest-variance possible fit (one parameter cannot overfit) and provably order-preserving. The catch: with one parameter it can only globally scale confidence; it cannot fix a curve whose miscalibration *direction* changes across the probability range. For CTR rankers where the residual after the downsampling correction is mostly a global over/under-confidence, temperature scaling is often enough.

### 6.4 Isotonic regression (non-parametric)

**Isotonic regression** fits the best *monotone non-decreasing* step function mapping uncalibrated score to calibrated probability. It makes no parametric assumption about the curve's shape — only that it is monotone (a reasonable assumption, since a higher score should mean a higher probability). The objective is a weighted least-squares fit subject to a monotonicity constraint:

$$ \min_{\hat{q}_1 \le \hat{q}_2 \le \cdots \le \hat{q}_n} \; \sum_{i=1}^{n} \big(\hat{q}_i - y_i\big)^2, $$

where the $y_i$ are the labels of the calibration examples sorted by uncalibrated score. This is solved exactly and in $O(n)$ by the **Pool Adjacent Violators Algorithm (PAVA)**: walk the sorted predictions, and wherever the running average would decrease (a "violation" of monotonicity), pool adjacent groups into a single block with their common mean, repeating until the whole sequence is non-decreasing. Because it is non-parametric and flexible, isotonic regression can fit *any* monotone reliability curve, so it almost always achieves the lowest ECE on a large calibration set. Its cost is variance: with little data it overfits the calibration set, producing a jagged step function that does not generalize, and it cannot extrapolate beyond the range of scores it saw. The rule of thumb from Niculescu-Mizil and Caruana (2005), who compared these head-to-head: **isotonic wins with enough calibration data (thousands+ of examples); Platt/sigmoid wins when data is scarce.** In recommenders you usually have millions of held-out impressions, so isotonic is the workhorse. In `scikit-learn`: `IsotonicRegression(out_of_bounds="clip")` or `CalibratedClassifierCV(method="isotonic")`.

### 6.5 Histogram / binning (non-parametric, simplest)

**Histogram binning** is the crudest non-parametric calibrator: partition predictions into bins, and replace every prediction in a bin with the *observed* click rate of that bin on the calibration set. It is dead simple and is exactly the inverse operation of the reliability diagram. Its weaknesses are that it produces a *step* output (all predictions in a bin collapse to the same value, destroying within-bin ordering — so it can *hurt* AUC) and that the bin edges are an arbitrary hyperparameter with the bias-variance trade-off discussed below. It is mostly useful as a baseline and as the conceptual bridge to isotonic (which is essentially adaptive binning that chooses the bin boundaries optimally under monotonicity).

### 6.6 Per-segment (per-position, per-slot) calibration

A single global calibrator assumes the miscalibration is the same everywhere. It is not. **Position bias** is the canonical example: an item shown in slot 1 gets clicked far more than the same item in slot 10, regardless of relevance, so a model that does not condition on position will be over-calibrated for low slots and under for high. The fix is **per-segment calibration**: fit a separate calibrator per slot (or per surface, per country, per device class), so each segment's distortion is corrected on its own. The trade-off is the familiar one — more segments means each calibrator sees less data and gets noisier — so segment only along axes where the miscalibration genuinely differs and you have enough data per segment. Position is almost always worth segmenting on in a feed; device class often is; per-user is almost never (no data per segment). The Facebook CTR paper (He et al. 2014) is explicit about per-position and per-context calibration as a production necessity.

It helps to see *why* position breaks a global calibrator, because it reveals the general principle. Suppose the true probability of a click factors into a relevance term and a position term: $P(\text{click} \mid \text{item}, \text{slot}) = P(\text{examine} \mid \text{slot}) \cdot P(\text{click} \mid \text{examine}, \text{item})$ — the user must first *look* at the slot (which depends only on position) and then decide to click (which depends on relevance). This is the standard position-based click model. A ranker trained on logged clicks without a position feature absorbs the average examination probability into its predictions, so it predicts something like the position-averaged click rate. When you then deploy it and the same item appears in slot 1 (high examination) versus slot 10 (low examination), the *single* predicted probability is simultaneously too low for slot 1 and too high for slot 10 — and a global calibrator, which sees only the prediction and not the slot, cannot pull it in two directions at once. Conditioning the calibrator on slot lets each slot's examination factor be absorbed into its own calibration curve. The general principle: **segment on any variable that shifts the true probability but is not (well) captured by the model's features.** Position is the textbook case; new-vs-returning user, cold-vs-warm item, and traffic source are common others. This also connects calibration to the broader bias problem in recommenders — position bias is the same beast that motivates inverse-propensity weighting in learning-to-rank — but here we are correcting its effect on the *probability level* rather than on the training gradient.

Here is the taxonomy as a decision table you can act on:

| Method | Family | Parameters | Order-preserving? | Best when | Risk |
|--------|--------|-----------|-------------------|-----------|------|
| Downsampling correction | Analytic | 0 (known $w$) | Yes | You down-sampled negatives | None if $w$ is right |
| Temperature scaling | Parametric | 1 ($T$) | Yes | Deep-net global overconfidence | Can't fix shape changes |
| Platt scaling | Parametric | 2 ($a, b$) | Yes | Sigmoidal curve, scarce data | Can't fit non-monotone wiggle |
| Isotonic regression | Non-parametric | many (steps) | Mostly (ties) | Large calibration set, any monotone curve | Overfits on small data; no extrapolation |
| Histogram binning | Non-parametric | bins | No (collapses) | Quick baseline | Hurts AUC; arbitrary bins |
| Per-segment | Wraps any above | × segments | Per-segment | Position/slot/device differences | Data-starved segments |

### 6.7 The bias-variance of binning, made precise

Why does the number of bins matter, for both measuring ECE and for histogram calibration? Consider estimating the true click rate within a bin that contains $n_m$ samples with true rate $\pi_m$. The observed rate $\hat{\pi}_m$ is a binomial mean, with variance $\frac{\pi_m(1-\pi_m)}{n_m}$. So the *variance* of the per-bin estimate shrinks as you put more samples in each bin — fewer, fatter bins are lower variance. But fewer bins means each bin spans a wider range of true probabilities, so collapsing them to one number introduces *bias* — the bin's average hides the variation of the true curve within the bin. More, thinner bins reduce that bias but raise the variance because each bin has fewer samples. That is a textbook bias-variance trade-off, living inside the calibration measurement and the histogram calibrator alike. Equal-mass binning is the pragmatic answer: it equalizes $n_m$ across bins so the variance is uniform, and it puts more (thinner) bins where the data is dense — exactly the low-probability region in CTR — so the bias is small where it matters. Isotonic regression sidesteps the choice entirely by letting PAVA find the bin boundaries that minimize squared error under monotonicity, which is why it dominates when you have the data.

## 7. The calibration pipeline: where it sits in the system

Calibration is a *post-hoc* layer. The base CTR model is trained to minimize logloss (which already pushes toward calibration); then a thin calibrator is fit on a *held-out* slice and inserted between the model's raw output and the decision that consumes the probability. The figure shows the flow.

![A vertical stack showing raw logit flowing into a sigmoid then a calibrator then a trustworthy probability and finally into an eCPM bid or fusion score](/imgs/blogs/calibration-and-the-prediction-you-can-trust-6.png)

Three engineering rules govern where the calibrator goes and how it is fit, and getting them wrong silently breaks everything downstream.

**Rule 1 — fit the calibrator on data the base model did not train on.** If you fit Platt or isotonic on the same examples the model trained on, the model has already memorized those labels, so the "uncalibrated" scores look artificially well-calibrated on that set and the calibrator learns nothing useful (or worse, learns to undo a distortion that only exists out-of-sample). Use a dedicated held-out calibration split, or fit on a temporally *later* slice than the training window — the latter is better for recommenders because it also captures a slice of the distribution shift the model will face at serving. This is exactly what `CalibratedClassifierCV` does with `cv="prefit"` (you pass an already-trained model and a fresh calibration set) or with internal cross-validation.

**Rule 2 — apply the calibrator at the right point in the chain.** The order is: model logit $z$ → downsampling correction (logit shift $+\log w$) → learned calibrator (temperature/Platt/isotonic on the corrected logit or probability) → calibrated probability → decision. Doing the learned calibrator *after* the analytic correction means it only has to clean up the residual, data-dependent distortion, which it does with far less data and far lower variance than if it had to learn the (large, known) downsampling bias too. Never make the learned calibrator do the analytic correction's job.

**Rule 3 — re-fit on a cadence, monitor ECE/MCE on a dashboard.** Because calibration drifts with distribution shift, the calibrator must be re-fit on recent data — far more often than you retrain the base model. A common pattern: retrain the base CTR model weekly (expensive), re-fit the 1-D calibrator daily or hourly (cheap). Put recent ECE and MCE on the same dashboard as AUC; the diagnostic signature of a calibration regression is **ECE up, AUC flat**, which no ranking-only metric would catch — and which is exactly the failure that pages you at 3 a.m.

## 8. The practical flow: training, breaking, and fixing calibration

Now the runnable spine. We will train a small CTR model with negative downsampling on Criteo-style data, show it is miscalibrated with a reliability diagram and ECE, then apply the three fixes — the analytic downsampling correction, Platt scaling, and isotonic regression — and re-measure. The pattern is the standard one for this series: name the dataset, measure honestly with a temporal split, report before→after.

### 8.1 The ECE and reliability-diagram code

First the measurement, because you cannot fix what you cannot see. This computes equal-mass-binned ECE and the points for a reliability diagram, with no dependency beyond numpy.

```python
import numpy as np

def reliability_curve(y_true, y_prob, n_bins=15, strategy="quantile"):
    """Return (conf, acc, weight) per bin for a reliability diagram + ECE.
    strategy='quantile' = equal-mass bins (recommended for CTR);
    'uniform' = equal-width bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if strategy == "quantile":
        # equal-mass: bin edges at quantiles of the predictions
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)            # collapse ties (common in CTR)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)

    conf, acc, weight = [], [], []
    n = len(y_true)
    for b in range(len(edges) - 1):
        mask = bin_id == b
        if mask.sum() == 0:
            continue
        conf.append(y_prob[mask].mean())   # average predicted prob in bin
        acc.append(y_true[mask].mean())    # observed click rate in bin
        weight.append(mask.mean())         # |B_m| / n
    return np.array(conf), np.array(acc), np.array(weight)

def expected_calibration_error(y_true, y_prob, n_bins=15, strategy="quantile"):
    conf, acc, weight = reliability_curve(y_true, y_prob, n_bins, strategy)
    return float(np.sum(weight * np.abs(acc - conf)))

def max_calibration_error(y_true, y_prob, n_bins=15, strategy="quantile"):
    conf, acc, _ = reliability_curve(y_true, y_prob, n_bins, strategy)
    return float(np.max(np.abs(acc - conf)))
```

To draw the reliability diagram, plot `conf` against `acc` and overlay the diagonal:

```python
import matplotlib.pyplot as plt

def plot_reliability(y_true, y_prob, label, ax):
    conf, acc, _ = reliability_curve(y_true, y_prob, n_bins=15)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(conf, acc, "o-", label=f"{label} (ECE={expected_calibration_error(y_true, y_prob):.3f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed click rate")
    ax.legend()
```

### 8.2 Train a CTR model with negative downsampling

We train a small logistic-regression-style CTR model (the foundations are in [the CTR prediction post](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations)) on a Criteo-style dataset, *with* negative downsampling at rate $w = 0.1$. The downsampling is the deliberate source of miscalibration we will then correct. For clarity the snippet uses `scikit-learn`'s `SGDClassifier` with log loss as a stand-in for the production logistic regression; the calibration story is identical for a PyTorch DNN — only the model object changes.

```python
import numpy as np
from sklearn.linear_model import SGDClassifier

rng = np.random.default_rng(0)

# --- Negative downsampling: keep all positives, keep each negative w.p. w ---
W = 0.1  # sampling rate for negatives
def downsample_negatives(X, y, w, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep_neg = neg[rng.random(len(neg)) < w]
    idx = rng.permutation(np.concatenate([pos, keep_neg]))
    return X[idx], y[idx]

# X_train, y_train: full hashed-feature CTR data with ~2% positive rate
X_ds, y_ds = downsample_negatives(X_train, y_train, W, rng)
print(f"base rate: full={y_train.mean():.4f}  downsampled={y_ds.mean():.4f}")
# e.g. full=0.0200  downsampled=0.1695  -> exactly the rebalancing we predicted

clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30)
clf.fit(X_ds, y_ds)

# Predict on a held-out set drawn from the TRUE (un-downsampled) distribution
p_raw = clf.predict_proba(X_calib)[:, 1]   # trained on rebalanced data -> inflated
```

When you evaluate `p_raw` against the true held-out labels, the reliability diagram bows hard below the diagonal: the model trained on a 17% base rate predicts a ~17% average probability into traffic whose true rate is ~2%. The ECE is large (around 0.14 in our runs), and AUC is whatever the features earn (around 0.78) — high AUC, terrible calibration, the canonical failure.

### 8.3 Fix 1 — the analytic downsampling correction

Apply the boxed formula. No fitting, no extra data.

```python
def downsample_correction(p, w):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return p / (p + (1.0 - p) / w)

p_corrected = downsample_correction(p_raw, W)
print("ECE raw      :", expected_calibration_error(y_calib, p_raw))
print("ECE corrected:", expected_calibration_error(y_calib, p_corrected))
# ECE raw      : 0.142
# ECE corrected: 0.038   <- the dominant bias is gone, AUC unchanged
```

The single most important line in this whole post is `p / (p + (1-p)/w)`. It removes the largest, fully-known source of miscalibration with zero fitting and zero risk to the ranking. Notice the residual ECE of 0.038 — that is the *data-dependent* overconfidence the analytic correction cannot touch, which the learned calibrators clean up next.

![A before and after comparison of an uncorrected inflated probability against a downsample-corrected probability that matches the true click rate](/imgs/blogs/calibration-and-the-prediction-you-can-trust-5.png)

### 8.4 Fix 2 — Platt scaling (logistic on the logits)

Fit a one-dimensional logistic regression on the *corrected* logits against the true labels, on the held-out calibration set.

```python
from sklearn.linear_model import LogisticRegression

# logits of the corrected probabilities; Platt fits sigma(a*z + b)
z_corrected = np.log(p_corrected / (1 - p_corrected)).reshape(-1, 1)
platt = LogisticRegression(C=1e6)          # near-unregularized 1-D fit
platt.fit(z_corrected, y_calib)
p_platt = platt.predict_proba(z_corrected)[:, 1]
print("ECE Platt:", expected_calibration_error(y_calib, p_platt))   # ~0.021
```

The cleaner production path is `CalibratedClassifierCV`, which handles the held-out split and the fit for you:

```python
from sklearn.calibration import CalibratedClassifierCV

# clf is already trained ("prefit"); fit the calibrator on a fresh split
platt_cv = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
platt_cv.fit(X_calib, y_calib)
p_platt_cv = platt_cv.predict_proba(X_test)[:, 1]
```

### 8.5 Fix 3 — isotonic regression (monotone, non-parametric)

With millions of held-out impressions, isotonic gets the lowest ECE. Fit it directly or via `CalibratedClassifierCV(method="isotonic")`.

```python
from sklearn.isotonic import IsotonicRegression

iso = IsotonicRegression(out_of_bounds="clip")  # clip beyond seen score range
iso.fit(p_corrected, y_calib)                    # fit on corrected probs
p_iso = iso.predict(p_corrected)
print("ECE isotonic:", expected_calibration_error(y_calib, p_iso))  # ~0.009

# Production form, same held-out discipline as Platt:
iso_cv = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
iso_cv.fit(X_calib, y_calib)
p_iso_cv = iso_cv.predict_proba(X_test)[:, 1]
```

### 8.6 Temperature scaling (for the deep-net case)

If the base model is a deep net (the Guo et al. setting), temperature scaling is the minimal, recommended fix. Fit a single scalar $T$ by minimizing NLL on held-out logits — a one-parameter optimization in a few lines of PyTorch.

```python
import torch
import torch.nn.functional as F

def fit_temperature(logits, labels, lr=0.01, steps=200):
    """Fit a single scalar T minimizing binary NLL on held-out logits."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    log_T = torch.zeros(1, requires_grad=True)        # optimize log T (T>0)
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad()
        T = log_T.exp()
        loss = F.binary_cross_entropy_with_logits(logits / T, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_T.exp())

T = fit_temperature(z_corrected.ravel(), y_calib)
p_temp = 1.0 / (1.0 + np.exp(-(z_corrected.ravel() / T)))
print("fitted T:", T, " ECE temp:", expected_calibration_error(y_calib, p_temp))
```

A fitted $T > 1$ confirms the deep net was overconfident (it softens the logits); $T \approx 1$ means the analytic correction already did the job and there is little global overconfidence left.

## 9. Results: before → after on Criteo-style CTR

Here is the measured before→after, on a held-out slice drawn from the *true* (un-downsampled) distribution, with a strict temporal split (train on earlier days, calibrate and evaluate on later days — no leakage). The numbers below are representative of what these methods deliver on a Criteo-style dataset with a ~2% base rate and $w = 0.1$ downsampling; treat them as the order of magnitude you should expect, not a benchmark claim.

| Method | ECE ↓ | MCE ↓ | AUC (order) | logloss ↓ | Notes |
|--------|-------|-------|-------------|-----------|-------|
| Raw (trained on downsampled data) | 0.142 | 0.31 | 0.781 | 0.612 | inflated probabilities, AUC fine |
| + Downsampling correction | 0.038 | 0.11 | 0.781 | 0.118 | analytic, no fitting, order unchanged |
| + Platt scaling | 0.021 | 0.07 | 0.781 | 0.114 | parametric, cleans residual |
| + Isotonic regression | 0.009 | 0.04 | 0.780 | 0.112 | non-parametric, best ECE, AUC −0.001 |

![A matrix of calibration methods against ECE and AUC showing ECE dropping sharply from raw to corrected to Platt to isotonic while AUC stays essentially fixed](/imgs/blogs/calibration-and-the-prediction-you-can-trust-8.png)

Read the table the way a practitioner should. The headline is the two columns moving in *opposite* ways: ECE collapses from 0.142 to 0.009 (a 15× improvement) while **AUC barely moves** — 0.781 to 0.780, a thousandth, well inside noise. That is the entire thesis made empirical: *calibration is a separate axis from discrimination, and you can fix one without touching the other.* The logloss column corroborates it — logloss crashes from 0.612 to 0.112 because logloss is the proper scoring rule that *sees* calibration, whereas AUC, which does not, is unmoved. Three more things to notice:

- **The analytic correction does most of the work.** It alone takes ECE from 0.142 to 0.038 — a 73% reduction — for free. The learned calibrators then squeeze the last bit. If you do only one thing, do the downsampling correction.
- **Isotonic edges out Platt on ECE** (0.009 vs 0.021) here *because we have a large calibration set*. With only a few hundred calibration points, isotonic would overfit and Platt would win — the Niculescu-Mizil/Caruana result. Always check your calibration-set size before choosing.
- **Isotonic's AUC dipped 0.001.** Because isotonic produces tied outputs (whole score ranges collapse to one calibrated value), it can break ties that the raw scores ordered, very slightly hurting AUC. It is negligible here, but in a tie-sensitive ranking it is a real (small) cost of isotonic over the strictly-monotone Platt/temperature.

#### Worked example: applying the downsampling correction with numbers

Concrete arithmetic for the correction, the way you would sanity-check it in a notebook. You down-sampled negatives at $w = 0.1$. The model, trained on the rebalanced data, outputs $p = 0.24$ for an impression. The true probability is

$$ q = \frac{p}{p + (1-p)/w} = \frac{0.24}{0.24 + 0.76 / 0.1} = \frac{0.24}{0.24 + 7.6} = \frac{0.24}{7.84} = 0.0306. $$

So the honest probability is about **3.1%, not 24%** — the model was off by roughly **8×** before correction. Check the logit form: $z = \log\frac{0.24}{0.76} = \log(0.3158) = -1.153$. The corrected logit is $z + \log(0.1) = -1.153 - 2.303 = -3.456$, and $\sigma(-3.456) = \frac{1}{1+e^{3.456}} = \frac{1}{1+31.7} = 0.0306$ — identical, confirming the logit-shift form. Now plug that into eCPM with a \$3 bid: the uncorrected eCPM would have been $3 \times 0.24 \times 1000 = \$720$ per mille; the true eCPM is $3 \times 0.031 \times 1000 = \$93$. Bidding on the uncorrected number means the pacer believes each thousand impressions is worth \$720 when it is worth \$93 — it will exhaust the daily budget at roughly one-eighth of the impressions it should buy, by mid-morning, on impressions that will not convert. That is the 3 a.m. page, in arithmetic.

## 10. Calibration in multi-objective fusion

The deepest reason calibration matters for *modern* recommenders is fusion, so it earns its own section. A production feed ranker is almost never single-objective. It is a multi-task model — MMoE, PLE, or a shared-bottom net (see [the multi-task ranking post](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple)) — that predicts several engagement signals per candidate: probability of click, expected watch time, probability of like, probability of share, probability of "see fewer like this." To rank with all of them you collapse them into one score, classically a weighted geometric mean:

$$ \text{score} = \hat{p}_{\text{click}}^{\,w_1} \cdot \big(\mathbb{E}[\text{watch}]\big)^{\,w_2} \cdot \hat{p}_{\text{like}}^{\,w_3} \cdot \hat{p}_{\text{share}}^{\,w_4}. $$

The weights $w_k$ encode business intent — "a share is worth as much as five clicks" — set by product, often via online experiments. Here is the trap: this formula is *only meaningful if each head outputs its true expectation*. Suppose the click head is calibrated but the share head (a rare label, small loss weight, hence poorly trained) is overconfident by 3×. Then $\hat{p}_{\text{share}}$ contributes a value 3× too large, the geometric mean over-weights shares relative to the $w_4$ you chose, and the feed tilts toward share-bait that the product team never intended to promote. The tuned weights are silently corrupted by a calibration artifact. You cannot debug this by looking at the weights — they are correct; the inputs to the formula are on the wrong scale.

The fix is structural: **calibrate every head independently before fusion.** Each head gets its own held-out calibrator (the regression head for watch time gets a regression calibrator — e.g. isotonic on predicted-vs-actual watch seconds; the binary heads get Platt/isotonic), fit on that head's own label. Only once each prediction is on its true scale does the weighted combination mean what the weights say it means. This is why the calibration post and the multi-task post are inseparable: multi-task ranking *generates* the calibration requirement, because fusing incommensurable scales is meaningless without it. A blunt way to say it: in a single-objective feed you can sometimes skip calibration; in a multi-objective feed, calibration is not optional, it is the precondition that makes the fusion arithmetic legal.

#### Worked example: a fusion error from one miscalibrated head

Two candidates, fused score $= \hat{p}_{\text{click}} \cdot \hat{p}_{\text{share}}^{2}$ (product team values a share heavily, so $w_{\text{share}} = 2$). True probabilities:

- Candidate A: true $p_{\text{click}} = 0.10$, true $p_{\text{share}} = 0.01$. True score $= 0.10 \times 0.01^2 = 1.0 \times 10^{-5}$.
- Candidate B: true $p_{\text{click}} = 0.04$, true $p_{\text{share}} = 0.03$. True score $= 0.04 \times 0.03^2 = 3.6 \times 10^{-5}$.

By the true probabilities, **B wins** (it is the better share-driver, which the weight rewards). Now suppose the share head is overconfident by 3× *for low values* (the rare-label distortion): it reports A's share as $0.01 \times 3 = 0.03$ and B's as $0.03 \times 1.5 = 0.045$ (a smaller multiplier at higher values — non-uniform, as always). The fused scores become:

- A: $0.10 \times 0.03^2 = 9.0 \times 10^{-5}$.
- B: $0.04 \times 0.045^2 = 8.1 \times 10^{-5}$.

The ranker now picks **A**. One miscalibrated head, on a single objective, flipped the ranking against the product team's tuned intent — and no amount of re-tuning $w_{\text{share}}$ fixes it, because the *input* to the formula is wrong, not the weight. Calibrate the share head first and the order returns to the truth.

## 11. Case studies / real numbers from the literature

Three named results anchor everything above in shipped systems and peer-reviewed work.

**Facebook CTR prediction (He et al., 2014), "Practical Lessons from Predicting Clicks on Ads at Facebook."** This is the paper that put CTR calibration on the industry map. Its headline modeling contribution is the GBDT-feature-transform-into-logistic-regression pipeline, but its operational lessons are about calibration and downsampling. The authors are explicit that for an ads system the *calibration* of the predicted CTR is as important as its ranking quality, because the auction consumes the probability. They use **negative downsampling** (they discuss subsampling negatives to make training tractable) and they apply the **re-calibration correction** to undo the sampling bias — the exact $q = \frac{p}{p+(1-p)/w}$ logic of section 4. They report tracking calibration (predicted-over-actual click ratio, the "normalized calibration" metric) as a first-class production health number, not just AUC. The paper also introduces the **Normalized Entropy** metric — logloss normalized by the entropy of the base click rate — precisely because raw logloss is hard to compare across datasets with different base rates, and because they wanted a calibration-aware metric rather than AUC alone. The takeaway the field absorbed: *in ads, you ship the calibration correction or you light money on fire.*

**Guo et al., 2017, "On Calibration of Modern Neural Networks."** The definitive study of deep-net miscalibration. The authors show that modern deep nets (ResNets and friends) are systematically and increasingly **overconfident** compared to the shallow nets of the early 2000s — depth, width, batch norm, and reduced regularization all push calibration in the wrong direction even as accuracy improves. They formalize ECE and reliability diagrams as the diagnostics, survey the calibration methods (histogram binning, isotonic, Platt, Bayesian binning, matrix/vector scaling), and find that **temperature scaling — a single scalar $T$ — is shockingly effective**, recovering near-perfect calibration on most benchmarks while being the lowest-variance possible fit and provably order-preserving (so it never hurts accuracy). This is the paper to cite for "deep CTR rankers need calibration" and "temperature scaling first." Its ECE-on-CIFAR/ImageNet numbers (ECE often dropping from several percent to well under one percent after temperature scaling) are the canonical demonstration that the level can be fixed without touching the order.

**Ad-tech bidding and the eCPM chain.** Across the programmatic-advertising industry, the eCPM identity $\text{eCPM} = \text{bid} \times \hat{p}_{\text{CTR}} \times 1000$ (and its conversion analogue with predicted CVR) is the spine of every bidder, which is why calibration is a deployment requirement, not a research nicety, in DSPs and ad networks. The standard production stack mirrors section 7 exactly: a base CTR/CVR model, a negative-downsampling correction, a learned (often isotonic, fit per-segment by placement and country) calibrator re-fit daily, and a calibration dashboard tracking predicted-over-actual by segment. The reported failure mode when calibration drifts is precisely budget mis-pacing and bid mis-allocation — the literature and practitioner write-ups converge on the same story: discrimination keeps you competitive, calibration keeps you solvent. (For the closely related problem where conversions arrive *late* and bias the labels you train on — a major calibration hazard for CVR models — see the conversion-attribution and delayed-feedback literature; it is its own deep topic that compounds with everything here.)

One honest caveat on numbers: where I have given precise ECE/AUC figures in the results table they are representative of these methods on Criteo-style data from runs in the literature and my own, not a single citable benchmark row; the *directions and magnitudes* (analytic correction does the bulk, learned calibrators clean the residual, AUC unchanged, 10–15× ECE reduction) are robust and are what you should reproduce. Never quote a specific ECE as if it were a universal constant — it depends on the dataset, the base rate, $w$, and the binning.

## 12. Stress-testing the calibration decision

Pose the engineering problem the way it actually arrives and reason to a decision, then break it.

**"Offline AUC went up after the retrain, ship it?"** Not until you check ECE. AUC rising tells you ordering improved; it says *nothing* about whether the probability level is still trustworthy. If anything downstream bids, fuses, or paces on the probability, you must confirm ECE/MCE did not regress and re-fit the calibrator on the new model's held-out outputs before shipping. The retrain that paged me had a *higher* AUC and a broken calibration — the AUC dashboard was the trap.

**"We down-sampled negatives 10×. Is the model wrong?"** Its *ranking* is fine (downsampling preserves order in expectation), but its *probabilities* are inflated by the known factor, so any probability consumer is wrong until you apply $q = \frac{p}{p+(1-p)/w}$. This is not a bug in the model; it is a missing post-processing step. The fix is one line and order-preserving.

**"ECE is great offline but the auction still mis-bids."** Almost always **distribution shift** or **per-segment miscalibration** hiding inside a good global ECE. A global ECE of 0.01 can coexist with a slot-1 ECE of 0.05 and a slot-10 ECE of −0.05 that cancel in the average. Segment the ECE by the dimensions the decision cares about (position, placement, country, device) and you will find the leak. Then fit per-segment calibrators on the offending axis.

**"The calibration set is tiny (a new market, a new surface)."** Isotonic will overfit — its non-parametric flexibility is variance you cannot afford with hundreds of points. Drop to Platt scaling (two parameters) or temperature scaling (one parameter); they are biased toward simple curves but they generalize from little data. Re-evaluate once the new surface has logged enough impressions to support isotonic.

**"The label is delayed (conversions land days after the click)."** Now your calibration set's labels are *censored* — recent impressions look like non-conversions only because the conversion has not arrived yet, biasing the observed rate downward and making the model look *under*-confident. Calibrating on censored labels bakes in a bias that flips as the window matures. The fix is delayed-feedback modeling (attribution windows, importance weighting for the censoring) before calibration — calibration assumes the labels it sees are the truth, and delayed feedback violates that assumption. This is a genuinely hard interaction and the reason CVR calibration is harder than CTR calibration.

**"A constant predictor has ECE ≈ 0 — is it calibrated?"** Yes, and uselessly so. This is the reminder that ECE alone can be gamed; it must be read *alongside* AUC. A model that is well-calibrated *and* well-discriminating is the goal; either alone is a half-built ranker. The constant predictor is the degenerate corner that proves ECE is a diagnostic, not an objective — you optimize logloss (which a constant predictor does *not* minimize unless the features are useless) and monitor ECE.

## 13. When to reach for calibration (and when not to)

A decisive recommendation, because every choice is a cost.

- **If the only consumer of the score is a single-objective top-K sort, do not invest in calibration.** Spend the sprint on discrimination — features, loss, hard negatives. Calibrating a feed nobody reads the number off of is wasted effort. (This is the most common over-engineering mistake: teams calibrate reflexively because a paper said to, on a surface that never uses the probability.)
- **If the score is bid on, fused, paced, thresholded, or shown to a user, calibration is mandatory.** Non-negotiable. An uncalibrated probability in an auction is a budget bug waiting to page you.
- **Always apply the downsampling correction if you down-sampled negatives.** It is one line, free, order-preserving, and removes the dominant bias. There is no argument against it.
- **Default to temperature scaling for deep nets, isotonic for everything with a large calibration set, Platt when data is scarce.** Don't reach for isotonic on hundreds of points (it overfits); don't reach for histogram binning in production (it hurts AUC and the bins are arbitrary).
- **Segment the calibrator only where the miscalibration genuinely differs and you have data.** Position is almost always worth segmenting; per-user almost never. More segments is more variance.
- **Re-fit the calibrator far more often than you retrain the model.** Calibration drifts with the distribution; the 1-D fit is cheap; daily or hourly re-fit on recent data is the standard.
- **Watch ECE and MCE next to AUC on the dashboard.** ECE-up-AUC-flat is the unique signature of a calibration regression and the failure no ranking metric catches.
- **Do not optimize ECE directly.** It is a diagnostic that a constant predictor games. Optimize logloss (the proper scoring rule) at training; monitor ECE.

## 14. Key takeaways

- **Discrimination and calibration are orthogonal jobs.** AUC measures only ordering and is invariant to any monotone rescaling; calibration measures whether $\hat{p} \approx P(\text{click})$. A model can ace one and fail the other.
- **You need a calibrated probability when the number is consumed as a number** — ads bidding (`eCPM = bid × pCTR`), multi-objective fusion, budget pacing, a "92% match" UI. A pure top-K sort needs only the order.
- **Negative downsampling is the most common CTR miscalibration, and it has an exact, free fix:** $q = \frac{p}{p+(1-p)/w}$, equivalently a logit shift of $+\log w$, derived from Bayes' rule. Apply it first, always.
- **Logloss is a proper scoring rule (uniquely minimized by the true probability) and AUC is not** — that is the formal reason logloss is the right training loss and AUC alone is a dangerous victory condition.
- **ECE is the calibration health metric, MCE the tail-risk one;** read both alongside AUC, never optimize ECE directly (a constant predictor games it), and use equal-mass binning for CTR's skewed predictions.
- **Pick the calibrator by data and shape:** temperature scaling (1 param, deep-net overconfidence, the Guo et al. default), Platt (2 params, scarce data), isotonic (non-parametric, large calibration set, lowest ECE), per-segment when position/placement differs.
- **Calibrate before fusing.** A multi-objective ranker can only combine $p_{\text{click}}$, $E[\text{watch}]$, $p_{\text{share}}$ if each head is independently calibrated; one miscalibrated head silently corrupts your tuned weights.
- **Calibration is a thin, frequently-re-fit post-hoc layer on a held-out split.** Fit on data the model did not train on; chain analytic-then-learned; re-fit daily even if you retrain weekly; and the regression to watch for is ECE up with AUC flat.

## 15. Further reading

- He, X. et al. (2014). *Practical Lessons from Predicting Clicks on Ads at Facebook.* ADKDD. The industry origin of CTR calibration, negative downsampling + correction, and Normalized Entropy as a calibration-aware metric.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). *On Calibration of Modern Neural Networks.* ICML. ECE, reliability diagrams, deep-net overconfidence, and temperature scaling.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.* The origin of Platt (sigmoid) scaling.
- Niculescu-Mizil, A., Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.* ICML. The Platt-vs-isotonic-by-data-size comparison and the overview of which calibrator to use when.
- Zadrozny, B., Elkan, C. (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates.* KDD. The isotonic-regression-for-calibration foundation.
- scikit-learn user guide: *Probability calibration* — `CalibratedClassifierCV`, `IsotonicRegression`, `calibration_curve` — the practical API for everything in section 8.
- Within this series: [the ranking model: CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations), [multi-task and multi-objective ranking: MMoE and PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple), the series intro [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), and the synthesis [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

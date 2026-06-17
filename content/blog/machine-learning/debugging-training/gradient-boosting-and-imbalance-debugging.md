---
title: "Gradient Boosting and Imbalance Debugging: XGBoost, LightGBM, and the Overfit You Didn't See"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why a gradient-boosted model can post 0.97 cross-validated AUC and crater to 0.71 in production, the boosting math that makes the overfit inevitable, and the early-stopping, regularization, metric, and calibration fixes that recover an honest model."
tags:
  [
    "debugging",
    "model-training",
    "gradient-boosting",
    "xgboost",
    "lightgbm",
    "class-imbalance",
    "tabular",
    "calibration",
    "data-leakage",
    "evaluation",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/gradient-boosting-and-imbalance-debugging-1.png"
---

A churn model lands on your desk with two numbers attached and one of them is a lie. The first number is **0.97 cross-validated ROC-AUC** — the kind of score that gets a model shipped, a deck written, a quarterly goal marked green. The second number arrives three weeks later, after the model has been scoring real customers in production: **0.71 AUC on the holdout the business actually tracks**. Twenty-six points of AUC evaporated between the notebook and the dashboard. Nobody touched the code. The training data did not change. The model that was "the best we've ever built" is now "the model that does barely better than guessing," and the only thing that changed is that reality showed up.

This is the most common way a gradient-boosted decision tree — XGBoost, LightGBM, CatBoost, the tabular workhorse that wins more Kaggle competitions and powers more production credit, fraud, and ranking systems than every deep net combined — lies to you. It is not one bug. It is a cluster of related bugs that share a single property: **they all inflate your validation number without improving the model.** A leaked feature that exists in your training table but not at scoring time. An early-stopping criterion that picks the number of trees off a validation fold the leak has contaminated. A thousand boosting rounds that drove training loss to near zero and validation loss back up the far side of its minimum. An eval metric — `logloss` — that you optimized while the thing you actually cared about was ranking, or recall at a fixed alert budget. A `scale_pos_weight` that recovered your rare-class recall but quietly destroyed the probabilities you ship downstream. The model did exactly what you told it. You told it the wrong thing, measured it with the wrong instrument, and stopped it at the wrong round.

![A two-panel before and after figure showing a gradient boosted model trusted as-is with 1500 rounds and a leaked validation set scoring 0.97 cross-validated and 0.71 in production, transformed by early stopping at round 240 a clean evaluation set and a dropped leaked feature into a debugged model scoring 0.84 cross-validated and 0.83 in production](/imgs/blogs/gradient-boosting-and-imbalance-debugging-1.png)

By the end of this post you will be able to take any gradient-boosted model that "won't generalize" or "was great in CV and terrible in prod" and, in about twenty minutes, bisect it to the right corner: a **data** leak, an **optimization** overfit, an **evaluation** metric mismatch, or a **numerics-and-calibration** distortion from imbalance handling. You will know why boosting fits residuals stage by stage and why that makes too many rounds overfit with mathematical certainty, why `scale_pos_weight` rescales the gradient in a way that lifts recall but provably inflates probabilities, what each regularization knob (`max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `lambda`, `alpha`) actually constrains, how the default `weight`-based feature importance can hide a leaked feature that `gain` and SHAP expose instantly, and how to read a train-versus-validation curve so the overfit announces itself. Then we go through the fixes — clean `eval_set` and early stopping, regularization, the right metric, probability calibration — with runnable `xgboost`, `lightgbm`, `scikit-learn`, and `shap` code for every one.

This post sits across the **data**, **optimization**, **evaluation**, and **numerics** corners of the six places a training bug hides — data, optimization, model code, numerics, systems, evaluation — which is exactly why a single misdiagnosis wastes weeks: the team treats a leak (data) as an overfit (optimization), or an imbalance-calibration problem (numerics) as a metric problem (evaluation), and tunes the wrong thing for a month. If you have not read the series' [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), it is the decision tree this post instantiates; the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) folds all of this into one checklist. Two siblings carry the cross-cutting threads in more depth: [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies) for the metric-and-loss story, and [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) for the leak story.

## 1. The symptom: a model that wins in the notebook and loses in production

Let us be precise about what makes gradient boosting *special* as a debugging surface, because the bugs are not the same as a deep net's. A GBDT has no NaN-at-step-4000, no fp16 underflow, no exploding gradients in the way a 100-layer network does. What it has instead is a different, sneakier failure mode: it is **almost too good at fitting**. A decision tree ensemble with enough rounds and enough depth can drive training loss to zero on essentially any dataset, including a dataset of pure noise. That capacity is a feature when your validation set is honest and your stopping criterion is sound. It is a loaded gun when either is not.

Here is the canonical symptom and the four root causes that produce it, because recognizing the symptom is the first bisection step:

| Symptom you observe | Likely corner | The deceptive instrument | The honest test |
| --- | --- | --- | --- |
| CV AUC 0.97, production AUC 0.71 | data (leak) | a validation set sharing rows or target-derived features with train | shuffle the target; drop suspect columns; `GroupKFold` |
| Train loss falls, valid loss rose long ago | optimization (overfit) | the final-round metric, read without the curve | plot train and valid loss over rounds |
| AUC is fine, the alerts are useless | evaluation (metric) | `logloss` optimized while you needed ranking or recall | score `aucpr` and recall at your budget |
| Recall recovered, downstream costs explode | numerics (calibration) | `predict_proba` after `scale_pos_weight` | a reliability curve and Brier score |

The reason this cluster is so reliably misdiagnosed is that **all four corners produce the same headline symptom** — a number that looks great until it meets reality — and three of them (leak, overfit, metric) inflate the *same instrument* (your validation score). The only way out is to stop trusting the headline number and run the confirming test for each corner. We will do all four. But first, the science, because the overfit and the imbalance distortion both fall straight out of how boosting works, and understanding the mechanism is what lets you predict the signature instead of pattern-matching it.

## 2. The science: how boosting fits residuals, and why too many rounds overfit

Gradient boosting builds an additive model one tree at a time. Start with a constant prediction $F_0(x)$ — for log-loss, the log-odds of the base rate. At each round $m$, you have the current ensemble $F_{m-1}(x)$, and you add one more tree $h_m$ scaled by a learning rate $\eta$ (the `eta` / `learning_rate` parameter):

$$F_m(x) = F_{m-1}(x) + \eta \, h_m(x).$$

The new tree $h_m$ is fit not to the labels but to the **negative gradient of the loss with respect to the current prediction** — the direction in which the ensemble's output should move to reduce loss. For squared error this negative gradient is literally the residual $y - F_{m-1}(x)$, which is why the intuition "each tree corrects the previous tree's mistakes" is exact for regression. For log-loss it is $y - p$ where $p = \sigma(F_{m-1}(x))$ is the current predicted probability, again a residual: the gap between the label and what the model currently believes.

XGBoost refines this with a second-order Taylor expansion. For each example $i$ at round $m$ it computes the gradient $g_i = \partial \ell / \partial F_{m-1}(x_i)$ and the Hessian $h_i = \partial^2 \ell / \partial F_{m-1}(x_i)^2$, then fits the tree to minimize the regularized objective

$$\mathcal{L}_m = \sum_i \left[ g_i \, h_m(x_i) + \tfrac{1}{2} h_i \, h_m(x_i)^2 \right] + \Omega(h_m),$$

where $\Omega(h_m) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2$ penalizes the number of leaves $T$ and the squared leaf weights $w_j$. The optimal weight for a leaf containing the example set $I_j$ is

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda},$$

and the gain of a candidate split is the drop in this objective it produces. That is the whole machine. Two facts from these equations drive every bug in this post, so hold onto them.

**Fact one: each round reduces training loss, monotonically, until it cannot.** Because every tree is fit to the current negative gradient and added with a positive step, the training loss is non-increasing in the number of rounds (modulo the learning rate being small enough, which it is by default). There is no early-stopping force *inside* the training objective. Left alone, the ensemble will keep carving the training data into finer and finer regions until each leaf is nearly pure — fitting the signal first, because the signal produces the biggest gain splits, then fitting the noise, because once the signal is exhausted the only gains left are the idiosyncratic quirks of the training sample. **This is why too many rounds overfit with certainty, not probability.** It is structural. The model has no reason to stop carving, so you must give it one.

**Fact two: the validation loss is U-shaped in rounds.** Early rounds reduce both training and validation loss, because they fit real structure that generalizes. At some round $m^*$, the marginal tree starts fitting training-sample noise that does not exist in the validation distribution; from there, validation loss rises while training loss keeps falling. That divergence point $m^*$ is the optimal number of trees, and the gap between the two curves *is the overfit*. You do not infer it, estimate it, or guess it — you read it off a plot.

It is worth dwelling on *why* the U-shape is inevitable rather than incidental, because that is what separates pattern-matching from understanding. A decision tree partitions the feature space into axis-aligned boxes, and a leaf's prediction is the (regularized) average gradient of the examples that fall in its box. Early in boosting, the high-gain splits are the ones that separate genuinely different populations — the split on `monthly_charges < 70` that separates a low-churn cohort from a high-churn one is high-gain *because the populations really are different*, and that difference reproduces in the validation set. As boosting proceeds and the big real differences are exhausted, the remaining gains come from splits that separate examples differing only by sampling noise — three unlucky training rows that happened to default sit together in a box, the tree carves them out, the training loss drops, and the validation loss does *not*, because the validation set has its own, different unlucky rows. The model is now learning the *seed of the random number generator that drew your training sample*, which is the textbook definition of overfitting. The transition from "fitting populations" to "fitting sampling noise" is gradual, which is why the validation curve has a smooth minimum rather than a sharp cliff — but it is monotone enough that early stopping with a patience window finds it reliably.

There is a subtle corollary that catches people: the *learning rate* moves where the U bottoms but not whether it exists. A smaller `eta` makes each tree's contribution smaller, so the ensemble approaches the same fit more slowly and needs more rounds to reach the validation minimum — but the minimum is still there, just further right and usually slightly *lower* (smaller steps overshoot the optimum by less, the classic shrinkage-improves-generalization result from Friedman's stochastic gradient boosting). This is why "more rounds at a smaller learning rate plus early stopping" is the standard recipe: you are trading compute for a lower, more precisely-located validation minimum. It is also why you must never tune the number of rounds and the learning rate independently — halve `eta` and your old `best_iteration` is now wildly too small. They move together along the same U.

![A timeline figure showing validation log loss falling from 0.69 at round 0 to a minimum of 0.33 near round 240 then rising to 0.46 by round 1500 while training log loss keeps falling from 0.38 toward 0.04 marking the round count where added trees only fit noise](/imgs/blogs/gradient-boosting-and-imbalance-debugging-2.png)

#### Worked example: where the validation curve turns

Take a binary classifier on 50,000 rows with a moderately learnable target. With `eta = 0.05`, `max_depth = 6`, and no regularization beyond defaults, you train 1,500 rounds and log both curves. The training log-loss falls smoothly: 0.69 at round 0, 0.38 by round 100, 0.18 by round 600, 0.04 by round 1,500 — it is heading for zero, and on a training set you can drive it arbitrarily close. The validation log-loss tells the real story: 0.69 at round 0, 0.41 at round 100, a **minimum of 0.33 near round 240**, then back up to 0.37 at round 600 and 0.46 by round 1,500. The model at round 1,500 has a *worse* validation loss (0.46) than the model at round 100 (0.41), and is far worse than the model at round 240 (0.33). Every one of the 1,260 trees added after round 240 actively hurt you. If you read only the final-round number — or worse, only the final training number, which looks fantastic at 0.04 — you would ship the round-1,500 model and never know that the round-240 model was 13 log-loss points better on data it had not seen. The fix is not a different model. It is a *stopping rule*: stop adding trees when the validation metric has not improved for a while. That is early stopping, and §4 shows how to do it without contaminating the very signal it relies on.

The same U-shape governs the other knobs. A deeper tree (`max_depth`) can carve finer regions per round, so it reaches the overfit zone in fewer rounds and overshoots harder. A smaller `min_child_weight` lets a leaf form on fewer examples (less total Hessian), so it can chase individual noisy points. Less subsampling means each tree sees the full, fixed training sample, so successive trees correlate and reinforce the same noise. Every regularization parameter is, mechanically, a way to flatten and right-shift that validation U so its minimum is lower and later — which is the entire content of §6.

## 3. The science: why scale_pos_weight rescales the gradient and distorts probabilities

Imbalance is the second half of this post, and the mechanism is just as clean. Suppose your positive class — the fraud, the churn, the defect — has prevalence $\pi$, often 1% or less. Under log-loss the per-example gradient is $g_i = p_i - y_i$. For a confidently-correct negative ($y=0$, $p \approx 0$) the gradient is near zero; for a missed positive ($y=1$, $p \approx 0$) the gradient is near $-1$. So far so symmetric. The problem is **counting**: with 99 negatives for every positive, the *sum* of negative gradients in any leaf or split dwarfs the sum of positive gradients, simply because there are 99 times as many terms. The optimal leaf weight $w_j^* = -\sum g_i / (\sum h_i + \lambda)$ is dominated by the majority, so the ensemble is pulled toward predicting "negative" everywhere — the same degenerate attractor we derived for cross-entropy in [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies), now expressed in boosting's gradient-and-Hessian language.

`scale_pos_weight = s` counteracts this by **multiplying the gradient and Hessian of every positive example by $s$.** Set $s$ to the negative-to-positive ratio $(1-\pi)/\pi$ and the total positive "weight" in each split matches the total negative weight; the majority no longer drowns the minority, the splits that separate positives become high-gain, and recall climbs. LightGBM exposes the same idea two ways: `scale_pos_weight` for an explicit factor, or `is_unbalance = True`, which sets the factor automatically to the class ratio. CatBoost has `auto_class_weights` and `class_weights`. They are all the same lever: reweight the gradient so the rare class is heard.

Here is the cost, and it is not optional — it follows directly from the math. By inflating the positive gradient by $s$, you have trained the model on a **reweighted distribution** in which positives are $s$ times more common than they really are. The model's `predict_proba` now estimates the probability under *that* reweighted world, not the real one. The effective base rate the model believes is

$$\pi_{\text{eff}} = \frac{s\,\pi}{s\,\pi + (1-\pi)},$$

which, for $s = (1-\pi)/\pi$, equals exactly $0.5$. So a model trained with `scale_pos_weight` set to the class ratio outputs probabilities centered on a 50/50 world, even though the real world is 1/99. **Its probabilities are systematically inflated.** A score of 0.30 from such a model does not mean "30% chance of fraud"; on real data the true frequency of fraud among examples scored 0.30 might be 5–8%. If anything downstream consumes the probability — an expected-loss calculation, a threshold tuned to a target precision, a calibrated alert budget — it is now wrong, and wrong in a direction that *over*-alerts. This is the calibration cost of `scale_pos_weight`, and it is invisible if you only ever look at AUC, because **AUC is rank-based and completely indifferent to calibration.** You can inflate every probability by an arbitrary monotone transform and AUC will not move a hair. The distortion hides in plain sight until something downstream consumes the absolute number.

![A matrix figure comparing four imbalance remedies scale_pos_weight is_unbalance auto random oversample and class threshold move across ranking AUC recall lift calibration and cost showing that scale_pos_weight and resampling lift recall but distort calibration while threshold moving preserves calibration for free](/imgs/blogs/gradient-boosting-and-imbalance-debugging-3.png)

The same distortion afflicts **resampling**. Random oversampling of the minority (or undersampling the majority) changes the *base rate the model sees*, so the trained model again estimates probabilities under a fabricated prevalence. Oversampling additionally slows training (more rows) and, with naive duplication, lets the model memorize specific minority rows — an overfit risk on top of the calibration shift. The one imbalance remedy that does *not* distort calibration is **threshold moving**: train an unweighted model that produces honest probabilities, then choose the decision threshold (not 0.5) that hits your target recall or precision on the probability the model honestly reports. You pay nothing in calibration because you never touched the gradient. The trade-off matrix above is the whole decision: if you ship *decisions*, `scale_pos_weight` plus threshold tuning is fine and recovers recall cheaply; if you ship *scores* that something multiplies by a dollar amount, either avoid the reweighting or recalibrate afterward. We will measure the distortion and fix it with calibration in §7.

It is worth distinguishing the gradient-reweighting approach from a genuinely different idea: **focal loss**. Where `scale_pos_weight` multiplies every positive's gradient by a constant $s$ regardless of how hard the example is, focal loss (Lin et al., 2017, originally for dense object detection) multiplies each example's gradient by $(1 - p_t)^\gamma$, where $p_t$ is the model's predicted probability of the *true* class and $\gamma$ is a focusing parameter. The effect is to down-weight *easy* examples (where $p_t$ is already near 1, so $(1-p_t)^\gamma \approx 0$) and keep the gradient on *hard* examples, of which the rare positives are disproportionately many. The crucial difference for debugging: a constant class weight rescales the whole class uniformly and so distorts calibration by a *predictable, invertible* amount (the base-rate shift derived above), whereas focal loss reweights *per example by difficulty*, which distorts calibration in a way that is harder to invert with a simple post-hoc map. Both can be implemented in XGBoost/LightGBM via a custom objective that returns the focal gradient and Hessian, but for tabular GBDTs the pragmatic ordering is almost always: try `scale_pos_weight` first (one knob, predictable distortion, fixable by isotonic), reach for focal loss only when the rare class is not just rare but genuinely *hard* (heavily overlapping with the majority in feature space), and budget for recalibration either way.

One more piece of the mechanism is worth making explicit because it is the source of a common confusion: **AUC measures discrimination, calibration measures honesty, and they are orthogonal.** Discrimination is "can the model rank a random positive above a random negative" — that is exactly ROC-AUC, and it is invariant to any monotone transform of the scores. Calibration is "does a score of 0.3 mean a 30% chance" — a property of the *values*, not the *order*. You can have perfect discrimination (AUC 1.0) and catastrophic calibration (every probability is 100× too large) at the same time, and `scale_pos_weight` produces almost exactly that profile: it improves the model's ability to *rank* the rare class up (better discrimination) while pushing the *values* off the diagonal (worse calibration). This is why a single number can never settle the question — you need a rank metric (AUC/PR-AUC) *and* a calibration instrument (reliability curve, Brier) to see both axes. A reviewer who signs off on AUC alone is signing off on half the model.

## 4. Diagnostic: early stopping on a clean eval_set (and the leak that ruins it)

Early stopping is the fix for the round-count overfit, and it is also where the most insidious leak in tabular ML hides. The mechanism is simple: hold out an evaluation set, after each boosting round score the model on it, and stop adding trees when the chosen metric has not improved for `early_stopping_rounds` consecutive rounds. The library then keeps the round at the best validation score — `best_iteration` — and discards the trees added afterward. Done right, it finds $m^*$ automatically.

Done wrong, it is a leak amplifier. If your evaluation set shares information with your training set — duplicate rows, near-duplicates, the same customer appearing in both, a feature computed using the future or the target — then the validation metric early stopping reads is optimistically biased, the curve bottoms *later* (or never visibly turns up), and early stopping selects too many trees. Worse, the `best_iteration` it reports is the round that best fit the *leak*, not the round that best fits the signal. You have automated overfitting and put a confidence interval on it. This is why "CV AUC 0.97, prod 0.71" is so often an early-stopping-on-leak story and not a pure round-count story: the leak inflates the very signal the stopping rule trusts.

Here is the correct pattern in XGBoost's scikit-learn API, with a genuinely held-out, leak-checked eval set:

```python
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# X, y are your full feature matrix and binary target.
# Split THREE ways: train / early-stopping-eval / final-test.
# The eval set drives early stopping; the test set is touched ONCE, at the end.
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=0)
X_eval, X_test, y_eval, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=0)

scale = (y_train == 0).sum() / (y_train == 1).sum()   # neg/pos ratio

clf = xgb.XGBClassifier(
    n_estimators=2000,            # an upper bound; early stopping picks the real count
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    scale_pos_weight=scale,       # imbalance handling (note the calibration cost, sec 7)
    eval_metric="aucpr",          # the RIGHT metric for imbalance (sec 5)
    early_stopping_rounds=50,
    random_state=0,
    n_jobs=8,
)

clf.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=50)

print("best_iteration:", clf.best_iteration)        # the m* it chose
print("trees actually used:", clf.best_iteration + 1)

# Final, honest numbers on the test set the model has never touched.
p_test = clf.predict_proba(X_test)[:, 1]
print("test ROC-AUC :", round(roc_auc_score(y_test, p_test), 4))
print("test PR-AUC  :", round(average_precision_score(y_test, p_test), 4))
```

Three things make this correct rather than just plausible. First, the **three-way split**: the eval set drives early stopping, which means early stopping is *tuning* on it, which means it is no longer an unbiased estimate of generalization — so a separate test set, touched exactly once, gives the honest number. People who report the early-stopping eval score as their final number are quoting a tuned-on set; it is optimistic by construction, a cousin of the bug in [cross-validation done wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong). Second, `early_stopping_rounds=50` with `n_estimators=2000`: the 2,000 is a ceiling, not a target — early stopping will halt long before it, typically in the low hundreds for a moderate problem. Third, `eval_metric="aucpr"`, which we justify next.

The leak check belongs *before* you trust any of this. The fastest leak detectors for tabular data are blunt and effective:

```python
import pandas as pd

# 1) Exact-duplicate rows across the train/eval boundary (a classic leak).
common = pd.merge(X_train.assign(_y=y_train.values),
                  X_eval.assign(_y=y_eval.values),
                  how="inner")
print("identical rows shared across train and eval:", len(common))

# 2) The "shuffle the target" sanity check: if a model still scores high
#    AUC after the target is randomly permuted, a feature is leaking the
#    label through the index, ordering, or a target-derived column.
import xgboost as xgb
from sklearn.metrics import roc_auc_score
y_shuf = np.random.RandomState(0).permutation(y_train)
probe = xgb.XGBClassifier(n_estimators=200, max_depth=4, eval_metric="auc")
probe.fit(X_train, y_shuf)
auc_shuf = roc_auc_score(y_eval, probe.predict_proba(X_eval)[:, 1])
print("AUC after shuffling target (should be ~0.50):", round(auc_shuf, 3))
```

If the duplicate count is non-zero you have a contamination leak; deduplicate before splitting or, better, split by a group key (customer id, session id) with `GroupKFold` so no entity straddles the boundary. If the shuffled-target AUC is meaningfully above 0.50, a feature is leaking the label through something other than honest signal — an id that correlates with the collection order, a column computed after the outcome, a `mean_target` encoding fit on the whole dataset. The shuffle test is one of the highest-leverage five-line scripts in tabular debugging, and it is the same discipline the [data leakage post](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) builds out in full.

#### Worked example: early stopping on a leaked vs clean eval set

A credit-risk model is trained with `early_stopping_rounds=50` against an eval set built by a random row split of a table that includes a `days_since_last_default` feature — which, for the positive (defaulted) class, is computed *after* the default event and so encodes the label. On the leaked eval set, validation AUC climbs to 0.971 and keeps climbing until round 880, where early stopping finally halts; the model ships with 880 trees and a reported 0.97. In production, where `days_since_last_default` is null for everyone (the default has not happened yet), AUC is 0.71. Now the fix: drop the leaked column, rebuild the splits with `GroupKFold` on customer id so no customer appears in both folds, and rerun. The clean eval AUC peaks at **0.843 around round 240**, early stopping halts at 290, and the once-touched test set reads 0.838. The model lost 13 points of *apparent* AUC and gained 13 points of *real* AUC — production now matches the offline number within noise. The leak did not just inflate the score; it pushed `best_iteration` from 240 to 880, so even the round count was a casualty. That is the signature of a leak laundered through early stopping: an inflated metric *and* an inflated tree count, both of which collapse together when the leak is removed.

## 5. Diagnostic: the wrong eval metric — optimizing logloss while you care about ranking

A gradient-boosted model optimizes a **training objective** (the loss its gradients come from) and is *stopped and selected* by an **eval metric**. These are different knobs and conflating them is a quiet, common bug. The objective for binary classification is almost always `binary:logistic` (log-loss); that is fine and rarely the problem. The bug is in the eval metric — the thing early stopping watches and the thing you report — when it does not match what you actually need from the model.

There are three things you might care about, and they want three different metrics:

| You actually need… | The metric that tracks it | The metric that misleads | Why |
| --- | --- | --- | --- |
| Well-calibrated probabilities | `logloss` (and a reliability curve) | `auc` | AUC ignores calibration entirely |
| Correct *ranking* of risk | `auc` | accuracy | accuracy collapses ranking to one threshold |
| Catching the rare class | `aucpr` (PR-AUC), recall@budget | `auc` | ROC-AUC is optimistic under heavy imbalance |
| A decision at a fixed alert budget | recall at top-k, precision@k | `logloss` | log-loss averages over all thresholds |

The most damaging mismatch under imbalance is **ROC-AUC versus PR-AUC**. ROC-AUC measures the trade-off between true-positive rate and false-*positive* rate. Under heavy imbalance the negative class is enormous, so the false-positive *rate* — false positives divided by the huge number of true negatives — stays tiny even when the *number* of false positives is large enough to drown your alert queue. ROC-AUC therefore looks reassuringly high (0.95+) while the model is operationally useless. PR-AUC (average precision) measures precision against recall, and precision *is* false-positives-relative-to-flagged, so it falls hard the moment the model floods the positive predictions with false alarms. On a 1%-prevalence problem a model can sit at ROC-AUC 0.96 and PR-AUC 0.34 simultaneously; the first number says "great," the second says "a third of your alerts are real if you want to catch most of the fraud." The second is the truth you ship on. This is the same lesson as [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies), applied to the metric early stopping watches.

The diagnostic is to compute all of them and let the disagreement teach you:

```python
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    precision_recall_curve, brier_score_loss)

def report(y_true, p):
    out = {
        "roc_auc":  roc_auc_score(y_true, p),
        "pr_auc":   average_precision_score(y_true, p),  # PR-AUC = avg precision
        "logloss":  log_loss(y_true, p),
        "brier":    brier_score_loss(y_true, p),         # calibration-sensitive
    }
    # Recall at a fixed alert budget: flag the top 1% of scores.
    k = max(1, int(0.01 * len(p)))
    top = np.argsort(p)[-k:]
    out["recall@top1pct"] = y_true[top].sum() / max(1, y_true.sum())
    return {m: round(v, 4) for m, v in out.items()}

print(report(y_test, p_test))
```

If you optimized and early-stopped on `logloss` but you actually operate at a fixed alert budget, the model selected the round that minimized average log-loss across *all* thresholds — which is not the round that maximizes recall in your top 1%. Switch `eval_metric` to `aucpr` (or evaluate recall@top-k with a custom metric) and early stopping selects a different, better round for your real objective. The model can be identical in architecture and worse-or-better depending only on *which round you kept*, and which round you kept depends on *which metric you watched*. The metric is not a passive measurement; under early stopping it is an active part of model selection.

When the built-in metrics do not match your real objective — and "recall in the top 0.25% of scores" is exactly the kind of objective they do not cover — XGBoost and LightGBM both accept a **custom eval metric**: a Python callable that receives the predictions and the labels and returns a name and a value, which early stopping then watches. This closes the last gap between "what the library can early-stop on" and "what the business measures." Here is a recall-at-budget metric wired into early stopping, which is the single most useful custom metric for fraud, abuse, and alert-queue problems:

```python
import numpy as np
import xgboost as xgb

def recall_at_budget(budget_frac=0.0025):
    # Returns an XGBoost-style eval metric: recall among the top budget_frac scores.
    def _metric(y_pred, dmatrix):
        y_true = dmatrix.get_label()
        k = max(1, int(budget_frac * len(y_pred)))
        top = np.argsort(y_pred)[-k:]
        recall = y_true[top].sum() / max(1.0, y_true.sum())
        return "recall@budget", recall      # higher is better
    return _metric

dtrain = xgb.DMatrix(X_train, label=y_train)
deval  = xgb.DMatrix(X_eval,  label=y_eval)
params = dict(objective="binary:logistic", eta=0.05, max_depth=6,
              min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, scale_pos_weight=scale, seed=0, nthread=8,
              disable_default_eval_metric=1)        # use ONLY our custom metric
bst = xgb.train(
    params, dtrain, num_boost_round=2000,
    evals=[(deval, "eval")], custom_metric=recall_at_budget(0.0025),
    maximize=True, early_stopping_rounds=50, verbose_eval=100)
print("best_iteration (by recall@budget):", bst.best_iteration)
```

The `maximize=True` flag matters — early stopping must know whether bigger is better, and for a recall metric it is (for a loss it is not, and getting this backwards silently selects the *worst* round). This is the deepest form of the metric-matching fix: instead of hoping a proxy metric correlates with your goal, you make the goal *itself* the early-stopping criterion, so the round the library keeps is provably the round that maximizes the thing you ship on. The cost is that recall@budget is a noisier, less-smooth metric than log-loss (it is a step function of the threshold), so you typically widen `early_stopping_rounds` to avoid stopping on a lucky bump.

The table below summarizes which metric to early-stop on as a function of what you actually deliver — the single decision that quietly determines which round the library keeps:

| What you deliver to production | Early-stop on | Report alongside | Do NOT early-stop on |
| --- | --- | --- | --- |
| A calibrated probability | `logloss` | reliability curve, Brier | `auc` (blind to calibration) |
| A risk ranking | `auc` | PR-AUC | accuracy (collapses ranking) |
| Alerts at a fixed budget | recall@budget (custom) | precision@k, PR-AUC | `logloss` (averages all thresholds) |
| A yes/no at a fixed cost ratio | `aucpr` | recall and precision at the threshold | `auc` (optimistic under imbalance) |

#### Worked example: AUC said fine, the alerts said no

A fraud team trains XGBoost with `eval_metric="auc"` and ships at ROC-AUC 0.962 — a number that survives every review. The analysts who work the alert queue have a budget of 500 reviews per day out of 200,000 transactions (the top 0.25%). At that operating point the model's precision is 0.11 and it catches 19% of the day's fraud. The team re-runs with `eval_metric="aucpr"`, which selects an earlier `best_iteration` and a model whose probabilities concentrate the true positives nearer the top of the ranking. ROC-AUC barely moves (0.958, within noise — *of course* it barely moves; ROC-AUC was never the problem). But PR-AUC rises from 0.31 to 0.39, and at the same 500-review budget precision rises to 0.17 and recall at top-0.25% rises from 19% to 28%. Nine more points of caught fraud per day, from changing one string in the config — because that string changed which round early stopping kept. The ROC-AUC that everyone trusted was structurally blind to the only operating point the business uses.

![A graph figure showing a suspiciously high cross-validated AUC of 0.97 forking into three confirming tests shuffle the target check the train versus valid gap and compare logloss against AUC each leading to a distinct fix dropping the leaked column adding early stopping with capped depth or setting the eval metric to aucpr](/imgs/blogs/gradient-boosting-and-imbalance-debugging-5.png)

## 6. Diagnostic: overfitting via depth, rounds, and missing regularization

When the train-versus-valid curve shows a clear, early minimum followed by a long rise — and you have ruled out a leak — you have a plain overfit, and the fix is regularization plus the early stopping from §4. The mistake here is treating regularization as one dial ("add more regularization") when it is six dials, each constraining a *different* overfit mechanism. Reading which dial moves the train-valid gap tells you what the trees were memorizing.

![A matrix figure showing four overfit knobs max_depth min_child_weight subsample and lambda against what each limits how it affects training fit how it narrows the validation gap and when to raise it explaining that depth and child weight cap per-tree capacity while subsample and lambda inject noise and shrink leaf weights](/imgs/blogs/gradient-boosting-and-imbalance-debugging-6.png)

The knobs, with the mechanism each attacks:

- **`max_depth` (XGBoost) / `num_leaves` (LightGBM)** — caps how finely one tree can carve the feature space. A depth-12 tree can isolate tiny noisy regions; a depth-6 tree cannot. This is the single highest-leverage anti-overfit knob. *Caveat for LightGBM:* it grows leaf-wise, not level-wise, so `num_leaves` is the real capacity control and `max_depth` is a secondary cap. A LightGBM with `num_leaves=255` is far deeper-fitting than its `max_depth` suggests, which is a frequent silent overfit when porting params from XGBoost.
- **`min_child_weight` (XGBoost) / `min_child_samples` and `min_sum_hessian_in_leaf` (LightGBM)** — the minimum total Hessian (roughly, weighted example count) required to form a leaf. Raise it and a leaf cannot form on a handful of noisy points; the tree is forced to find structure that holds across enough examples to be real.
- **`subsample`** — the fraction of *rows* each tree sees (stochastic gradient boosting). Below 1.0 it injects noise that decorrelates successive trees, so they stop reinforcing the same training-sample quirks. 0.8 is a sane default.
- **`colsample_bytree` / `colsample_bylevel`** — the fraction of *columns* each tree (or level) may split on. Like `subsample` but over features; it also blunts a single dominant (possibly leaked) feature by denying it to some trees.
- **`reg_lambda` (L2) and `reg_alpha` (L1)** — penalize leaf weights. Recall $w_j^* = -\sum g_i / (\sum h_i + \lambda)$: raising $\lambda$ shrinks every leaf weight toward zero, so individual trees make smaller, less confident corrections and the ensemble is smoother. `alpha` (L1) can zero out leaf weights entirely, a sparsity pressure.
- **`learning_rate` (`eta`)** — smaller steps mean each tree contributes less, so you need more rounds but each round overfits less; the classic recipe is small `eta` plus early stopping. It trades compute for generalization.

The diagnostic is to *plot the curves under different settings* rather than trust a single CV number, because the curve shows you the mechanism. XGBoost stores the per-round eval history; LightGBM does too. Here is the curve-reading code that turns the abstract "it overfits" into the concrete round and gap:

```python
import matplotlib.pyplot as plt
import xgboost as xgb

clf = xgb.XGBClassifier(
    n_estimators=1500, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, eval_metric=["logloss", "aucpr"],
    random_state=0)

clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_eval, y_eval)],
        verbose=False)

hist = clf.evals_result()                 # {'validation_0': train, 'validation_1': eval}
train_ll = hist["validation_0"]["logloss"]
eval_ll  = hist["validation_1"]["logloss"]

best = int(min(range(len(eval_ll)), key=lambda i: eval_ll[i]))
print(f"valid logloss bottoms at round {best}: {eval_ll[best]:.4f}")
print(f"train logloss at that round       : {train_ll[best]:.4f}")
print(f"final-round train/valid gap       : "
      f"{train_ll[-1]:.4f} / {eval_ll[-1]:.4f}")

plt.plot(train_ll, label="train")
plt.plot(eval_ll, label="valid")
plt.axvline(best, ls="--", color="k", label=f"best @ {best}")
plt.xlabel("boosting round"); plt.ylabel("log loss"); plt.legend()
plt.savefig("train_valid_curve.png", dpi=120, bbox_inches="tight")
```

The `eval_set=[(train), (eval)]` trick — passing the *training* set as an eval set too — is what gives you both curves, so you can see the gap, not just the valid minimum. A healthy run has the two curves close and both still gently falling at `best_iteration`. An overfit run has them diverging hard: valid bottoms early and rises while train keeps plunging toward zero. The *size of the gap at the best round* is a direct read on how much capacity the model has to burn — a big gap means you can regularize harder (raise `min_child_weight`, lower `max_depth`) and likely push the valid minimum lower still.

#### Worked example: closing a 0.30-log-loss train-valid gap

A LightGBM model is ported from an XGBoost config, but the porter set `num_leaves=255` (LightGBM's default) without realizing it corresponds to a far deeper effective tree than the old `max_depth=6`. The curves show it instantly: at `best_iteration` (round 410) the train log-loss is 0.06 and the valid is 0.36 — a **0.30 gap**, a chasm. The model is memorizing. The fix is three knobs, applied while watching the gap shrink: `num_leaves` 255 → 31 (matching depth-6 capacity), `min_child_samples` 20 → 100, `feature_fraction` (LightGBM's `colsample_bytree`) 1.0 → 0.8. Rerun: train log-loss at the new `best_iteration` (round 520) is 0.27, valid is 0.32 — a **0.05 gap**, and crucially the valid minimum itself dropped from 0.36 to 0.32. Tighter regularization did not just close the gap cosmetically; it found a genuinely better model, because the capacity it removed was capacity the model was spending on noise. The train number got "worse" (0.06 → 0.27) and that is exactly the point — a training log-loss heading for zero is not a trophy, it is the overfit warning light.

```python
import lightgbm as lgb

params = dict(
    objective="binary",
    metric="average_precision",   # LightGBM's PR-AUC; tracks the rare class
    num_leaves=31,                # was 255 — the silent overfit
    min_child_samples=100,        # was 20
    feature_fraction=0.8,         # colsample
    bagging_fraction=0.8,         # subsample
    bagging_freq=1,
    learning_rate=0.05,
    lambda_l2=1.0,
    is_unbalance=True,            # imbalance handling (note: distorts probabilities)
    seed=0,
    num_threads=8,
    deterministic=True,           # reproducibility (sec 9)
)
dtrain = lgb.Dataset(X_train, label=y_train)
deval  = lgb.Dataset(X_eval,  label=y_eval, reference=dtrain)
model = lgb.train(
    params, dtrain, num_boost_round=2000,
    valid_sets=[dtrain, deval], valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
print("best_iteration:", model.best_iteration)
```

## 7. Diagnostic: calibration — the cost of scale_pos_weight, and the fix

We proved in §3 that `scale_pos_weight` (and resampling) inflate predicted probabilities by training on a reweighted base rate. Now we measure it and fix it. The instrument is the **reliability curve**: bin the predicted probabilities, and for each bin plot the mean predicted probability against the actual fraction of positives in that bin. A perfectly calibrated model lies on the diagonal — a 0.30 bin contains 30% positives. A `scale_pos_weight` model sags far below: its 0.30 bin contains maybe 6% positives, because it believes the world is 50/50 when it is 1/99.

```python
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
for mp, fp in zip(mean_pred, frac_pos):
    print(f"predicted ~{mp:.2f}  ->  actual {fp:.2f}")
print("Brier score:", round(brier_score_loss(y_test, p_test), 4))
# A scale_pos_weight model shows predicted >> actual in every bin,
# and a Brier score inflated by the overconfidence.
```

The Brier score — mean squared error between predicted probability and outcome — is a single number that captures calibration *and* sharpness; a `scale_pos_weight` model has an inflated Brier even when its AUC is excellent, which is the numeric fingerprint of "great ranking, lying probabilities." If you only ship a *ranking* or a *threshold*, you can stop here: AUC and recall are fine, calibration is irrelevant, leave it. If anything downstream consumes the probability as a probability, recalibrate.

The fix is a held-out **calibration map** fit on a set the model did not train on — isotonic regression (a monotone step function, flexible, needs a few hundred positives) or Platt/sigmoid scaling (a logistic fit, lower variance, fine with less data). `scikit-learn`'s `CalibratedClassifierCV` wraps this, but the cleanest pattern for a pre-trained booster is `FrozenEstimator` so calibration does not retrain the model:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator   # sklearn >= 1.6
from sklearn.metrics import brier_score_loss

# clf is the already-fitted XGBoost model from sec 4 (trained with scale_pos_weight).
# Calibrate on the eval set (NOT the train set, NOT the final test set).
calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
calibrated.fit(X_eval, y_eval)

p_cal = calibrated.predict_proba(X_test)[:, 1]
print("Brier before:", round(brier_score_loss(y_test, p_test), 4))
print("Brier after :", round(brier_score_loss(y_test, p_cal), 4))
# Ranking metrics are UNCHANGED (isotonic is monotone, so AUC is identical),
# but the probabilities now land on the diagonal.
```

Because isotonic regression is monotone, **it cannot change the ranking** — your ROC-AUC and PR-AUC are bit-identical before and after — it only remaps the probability *values* so a 0.30 means 30%. You keep the recall `scale_pos_weight` bought you *and* recover honest probabilities. The only cost is one held-out set spent on the calibration fit. The alternative, if you have no probability consumer at all, is the calibration-free route: train *without* `scale_pos_weight` (honest probabilities out of the box) and recover recall by **threshold moving** — pick the operating threshold on the PR curve that hits your target recall, instead of the default 0.5. That route trades nothing and is the right default when you ship decisions, not scores; the matrix in §3 lays out exactly when each path wins.

![A two-panel before and after figure showing a raw scale_pos_weight model with recall 0.74 but a predicted 0.30 corresponding to a true 0.06 and a Brier score of 0.18 transformed by isotonic calibration into the same recall 0.74 with predicted 0.30 now matching true 0.29 and a Brier score of 0.09 sitting on the diagonal](/imgs/blogs/gradient-boosting-and-imbalance-debugging-4.png)

#### Worked example: recall kept, probabilities fixed

A claims-fraud model trained with `scale_pos_weight=60` (1.6% prevalence) reports ROC-AUC 0.91 and a recall of 0.74 at its chosen threshold — good. The actuary downstream multiplies its `predict_proba` by claim amount to set a reserve, and the reserves come out 4× too high. The reliability curve explains it: the model's 0.30 bin contains 6% true positives, its 0.50 bin contains 14%, everything is inflated roughly fivefold, and the Brier score is 0.18. Fitting isotonic calibration on the held-out eval set remaps the probabilities to the diagonal: the 0.30 bin now reads 0.29 true, the 0.50 bin reads 0.49, the Brier score drops to 0.09. ROC-AUC is still 0.91 to four decimals (isotonic is monotone, ranking is untouched), recall at the same *rank-equivalent* threshold is still 0.74, but the reserves the actuary computes are now correct. The bug was never the model's discrimination; it was that nobody asked whether the number labeled "probability" was one. AUC could not have told them — only the reliability curve and the Brier score could.

## 8. Diagnostic: feature importance misreads, and the leaked feature that dominates

When you suspect a leak but the shuffle test was ambiguous, feature importance can find the culprit — *if you read the right kind*. This is a trap because the **default importance type lies the most**. XGBoost's default `importance_type="weight"` counts how many times a feature is used to split, across all trees. A high-cardinality feature (an id, a timestamp, a near-continuous amount) gets used in *many* splits simply because it offers many cut points, so it scores high on `weight` regardless of how much each split actually helps. Conversely, a leaked feature that the model uses in *few but enormously valuable* splits can score *low* on `weight` and hide.

Two importance types tell the truth instead:

- **`gain`** — the total loss reduction the feature delivered, summed over its splits. This is "how much did this feature actually help," and a leaked feature that single-handedly separates the classes will dominate `gain` even with few splits.
- **`cover`** — the total Hessian (example coverage) of the splits using the feature. Useful context, less diagnostic than `gain`.

But model-internal importances of any kind have a deeper flaw: they are *split-based*, so they credit a feature for being *used*, not for being *causal* or even *available at inference*. The gold standard for "what is this model actually relying on" is **SHAP** (SHapley Additive exPlanations), which attributes each prediction to its features via a game-theoretic allocation and, summed over a dataset, gives a faithful global ranking. SHAP has fast exact algorithms for tree models (`TreeExplainer`), so it is cheap on GBDTs. A leaked feature lights up SHAP like a flare.

```python
import numpy as np
import xgboost as xgb
import shap

# Compare the three importance views. The default 'weight' can hide a leak.
booster = clf.get_booster()
for kind in ("weight", "gain", "cover"):
    imp = booster.get_score(importance_type=kind)
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:5]
    print(f"\nTop 5 by {kind}:")
    for f, v in top:
        print(f"  {f:30s} {v:,.1f}")

# SHAP: the faithful global ranking. A leaked feature dominates here.
explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(X_eval)             # (n_samples, n_features)
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(-mean_abs)
print("\nTop 5 by mean |SHAP|:")
for i in order[:5]:
    print(f"  {X_eval.columns[i]:30s} {mean_abs[i]:.4f}")
```

The tell of a leak is a feature that ranks **far higher by `gain` and SHAP than by `weight`**, *and* whose presence is implausibly predictive given the domain. An `account_id` that ranks 1st by gain is leaking record-collection order. A `last_payment_date` that dominates SHAP on a default-prediction task is leaking the future. The remedy is to drop the feature and re-measure honest performance — and watch the AUC fall to its true value, which is the *good* outcome (the inflated AUC was the bug, not the corrected one). Two cautions complete the picture. First, **importance is not causation**: a feature ranking high means the model used it, not that it drives the real-world outcome; never read a SHAP plot as a causal claim. Second, importance is *correlated-feature-blind*: if two features carry the same signal, tree importance splits the credit between them, so a low score does not prove a feature is useless. Use importance to *find suspects* and the leak/availability check to *convict* them. The deeper feature-engineering pitfalls — encodings, unseen categories, train-serve skew — get their own treatment in [categorical and feature bugs](/blog/machine-learning/debugging-training/categorical-and-feature-bugs).

![A two-panel before and after figure showing default weight based importance ranking a leaked timestamp feature 9th and hidden while gain and SHAP rank the same leaked timestamp 1st holding 0.21 of total gain and explaining the inflated AUC of 0.97 that drops to 0.84 once the feature is dropped](/imgs/blogs/gradient-boosting-and-imbalance-debugging-7.png)

#### Worked example: the timestamp that ranked 9th and explained everything

A subscription-churn model posts 0.96 CV AUC and 0.74 in production. The shuffle test is suggestive (shuffled-target AUC 0.61, above 0.50 but not damning), so you go to importance. By default `weight`, the top features are sensible: `monthly_charges`, `tenure_months`, `support_tickets` — and `last_event_ts` sits at rank 9, a low-split-count feature that draws no attention. By `gain`, `last_event_ts` is **rank 1**, carrying 0.21 of total gain — a single feature with a fifth of the model's entire predictive power. SHAP confirms it: `last_event_ts` has the largest mean absolute SHAP value by a wide margin. The domain check convicts it: `last_event_ts` is the timestamp of the customer's final recorded event, which for churned customers is *after* they churned and for active customers keeps updating — it is the label wearing a feature's coat. Drop it, retrain, and CV AUC falls to 0.84 while production rises to 0.83. The model that "lost" 12 points of AUC is the only one that was ever real. The `weight` importance that buried the leak at rank 9 was not malfunctioning — it was answering a different question ("how often is this used to split?") than the one you needed answered ("how much does this feature carry?"), and the gap between those two questions is exactly where the leak lived.

## 9. Categorical handling: native splits, one-hot, and the high-cardinality trap

The three major GBDT libraries disagree about categorical features, and the disagreement is a quiet source of both bugs and missed performance. Understanding it is part of the same scientific story: a categorical feature has no natural ordering, so a tree's "is feature less than threshold" split does not directly apply, and each library resolves that differently.

**One-hot encoding** is the lowest-common-denominator approach: turn a categorical with $C$ levels into $C$ binary columns. It always works and the resulting splits are interpretable, but it has two failure modes that bite under high cardinality. First, each one-hot column is sparse and individually weak — a split on `city == Hanoi` can only separate Hanoi from everyone else, never group cities that behave alike — so the tree needs many splits and many rounds to express "these twelve cities are high-risk," which it does inefficiently and is prone to overfit on the rare levels. Second, a category with thousands of levels (a zip code, a device id, a product sku) explodes into thousands of columns, blowing up memory and starving each column of examples; the rare levels each appear a handful of times and the tree memorizes them. This is the **high-cardinality trap**, and it is the most common categorical bug in tabular GBDTs: a model that scores well in CV (it memorized the rare-level-to-target mapping that happens to repeat across train and eval) and collapses on unseen levels in production.

**LightGBM native categorical** is fundamentally different and usually better. You tell LightGBM which columns are categorical (`categorical_feature=[...]` or a pandas `category` dtype) and it uses an algorithm from Fisher (1958) that, for a categorical split, sorts the categories by their accumulated gradient statistics and finds the optimal *partition of the category set into two groups* in near-linear time. So a single split can say "these twelve cities go left, the rest go right" — exactly the grouping one-hot cannot express in one split. This is more powerful and more compact, but it has its own traps: LightGBM caps the number of categories it will consider per split (`max_cat_threshold`) and applies smoothing (`cat_smooth`, `cat_l2`) precisely *because* high-cardinality categorical splits overfit easily, and if you leave these at defaults on a very high-cardinality feature you can still overfit hard. The silent bug here is forgetting to *mark* the column as categorical at all — pass a label-encoded integer column to LightGBM without declaring it categorical and LightGBM treats it as *ordinal*, splitting on "is category-id less than 7," which imposes a meaningless order (category 6 is not "less than" category 7) and quietly degrades the model.

**CatBoost** goes furthest, using ordered target statistics (a target-encoding that is computed in a way that provably avoids the target leakage naive target-encoding suffers from) as its default categorical handling. It is often the strongest out-of-the-box on heavily-categorical data for exactly this reason — but the lesson generalizes to a warning: **naive target/mean encoding is a leak generator.** If you compute "mean target for this category" over the *whole* dataset and feed it as a feature, you have leaked each row's own label into its features (its own label contributed to its category's mean), and the model's CV score inflates while production craters. Target encoding must be done inside the CV fold (fit the encoder on the training fold only) or with CatBoost's ordered scheme; doing it once on all the data is the canonical [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage) bug.

| Approach | Splits it can express | High-cardinality behavior | The trap |
| --- | --- | --- | --- |
| One-hot | one level vs rest, per column | columns explode, rare levels memorized | high-cardinality blowup, overfit |
| LightGBM native | optimal subset partition | smoothed, capped by `max_cat_threshold` | forgetting to mark the column categorical |
| Naive target encoding | continuous, dense | compact but leak-prone | self-leak if fit on all data |
| CatBoost ordered TS | leak-safe target stats | strong default | none major; just slower to train |

The diagnostic for a categorical bug is to check three things: that high-cardinality columns are *declared* categorical (or target-encoded inside-fold), that the set of category levels at *serving* time is a subset of training (unseen levels are a guaranteed train-serve skew — decide explicitly whether they map to a default bucket or `np.nan`), and that no target-derived encoding was fit on the full dataset. A model that uses a label-encoded `product_id` as an ordinal feature, with new product ids appearing daily in production, is carrying two bugs at once (meaningless order *and* unseen-level skew) and will degrade steadily as the catalogue turns over.

## 10. Reproducibility, missing values, and the sentinel trap

Two smaller-but-vicious classes of GBDT bug round out the surface, because they corrupt *everything above*: if your run is not reproducible you cannot trust any before/after comparison, and if your missing-value handling is wrong every split is computed on a lie.

**Reproducibility.** A GBDT is far more reproducible than a deep net, but it is not free. Setting `random_state` / `seed` controls the row and column subsampling, but two things still introduce nondeterminism. First, **`num_threads` / `n_jobs` greater than 1** with certain histogram-building or sketch methods can produce tiny floating-point differences from non-associative parallel summation, which then cascade through split selection into different trees. XGBoost's default `hist` tree method is deterministic across runs at fixed thread count but *may differ across different thread counts*; pin both the seed and the thread count to compare runs. Second, LightGBM needs `deterministic=True` *and* a fixed `num_threads` for bit-exact reproducibility; without the flag its feature bundling and histogram construction can vary. The rule for debugging: **fix `seed`, fix `num_threads`, and set the determinism flag**, then verify two runs produce identical predictions before you trust a before/after delta. This is the same discipline the series argues for in [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — you cannot debug what you cannot reproduce.

```python
# XGBoost: pin seed AND threads; verify determinism before trusting deltas.
import numpy as np, xgboost as xgb
def fit_once():
    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist",
                          random_state=0, n_jobs=4)   # pin n_jobs!
    m.fit(X_train, y_train)
    return m.predict_proba(X_eval)[:, 1]
p1, p2 = fit_once(), fit_once()
print("max abs diff between two runs:", np.max(np.abs(p1 - p2)))  # want 0.0
```

**Missing values and the sentinel trap.** XGBoost and LightGBM handle missing values *natively* and cleverly: at each split, examples whose split-feature is missing are sent to a **default direction** (left or right), and the library *learns* which direction minimizes loss. This is a real strength — you do not need to impute. But it creates a precise, devastating bug: if your pipeline has already replaced missing values with a **sentinel** like `-999`, `-1`, `0`, or `9999` before the data reaches the booster, then the booster sees those sentinels as *real numeric values*, not missing. It will happily split on "is `age` less than `-500`?" — a split that separates the sentinel-encoded missings from everyone else, treating "missing" as an extreme real value. Worse, if the sentinel is `0` and `0` is also a legitimate value (zero balance, zero clicks), the booster cannot tell a real zero from a missing one, and *both* meanings get fused into one. The fix is to let the booster see missing as missing: pass `np.nan` (XGBoost's default `missing=np.nan` then routes them) rather than pre-filling, or if you must use a sentinel, tell the library with `missing=-999` so it treats that value as the missing marker.

```python
import numpy as np, xgboost as xgb
# WRONG: missings pre-filled with -999 BEFORE the booster sees them.
X_bad = X.fillna(-999)
bad = xgb.XGBClassifier(n_estimators=300).fit(X_bad, y)   # -999 is now a "real" value

# RIGHT (option A): leave NaN; XGBoost routes missings to a learned direction.
X_good = X.copy()                                          # NaNs intact
good = xgb.XGBClassifier(n_estimators=300, missing=np.nan).fit(X_good, y)

# RIGHT (option B): if a sentinel is unavoidable, declare it.
sentinel = xgb.XGBClassifier(n_estimators=300, missing=-999).fit(X_bad, y)
```

The signature of the sentinel bug is subtle: the model often performs *fine* in CV (the sentinel is consistent across train and eval, so the model learns to use it) but degrades in production if the *serving* pipeline fills missings differently (a different sentinel, or real `np.nan`), producing a **train-serve skew** that no offline metric catches. The honest test is to confirm your training-time and serving-time missing-value handling are byte-for-byte identical — the same library version, the same `missing` argument, the same upstream fill. A GBDT that scores 0.88 offline and 0.74 in prod with no leak and no overfit is very often this: the serving layer hands the booster `np.nan` while training handed it `-999`, and the learned default directions no longer apply.

## 11. The full bisection: from "0.97 in CV, 0.71 in prod" to an honest model

Let us run the whole diagnostic as one narrative, because the order of operations is itself the skill. You have the canonical symptom: **0.97 cross-validated AUC, 0.71 in production.** Do not start tuning. Bisect.

**Step 1 — Reproduce and pin (systems/numerics).** Fix `seed`, `num_threads`, the determinism flag. Run twice; confirm identical predictions (`max abs diff == 0`). If they differ, you cannot trust any delta below — fix this first. (Cost: 2 minutes. Rules out: nothing yet, but makes everything else trustworthy.)

**Step 2 — Check for a leak (data).** Run the duplicate-rows check across the train/eval boundary and the shuffle-target probe. If duplicates exist or shuffled-target AUC is well above 0.50, you have a leak; go to importance to find it (`gain` and SHAP, not `weight`), drop the feature, rebuild splits with `GroupKFold`, and re-measure. **Most "0.97 → 0.71" cases die here.** (Cost: 10 minutes. Rules in/out: contamination and target leakage.)

**Step 3 — Plot the train-valid curve (optimization).** Pass `eval_set=[(train),(eval)]`, read both curves. A big gap at `best_iteration` with valid rising means overfit: lower `max_depth`/`num_leaves`, raise `min_child_weight`/`min_child_samples`, set `subsample`/`colsample` to 0.8, add `lambda`, and let early stopping pick the round on a *clean* eval set. (Cost: 15 minutes. Rules in/out: round-count and capacity overfit.)

**Step 4 — Check the metric (evaluation).** Compute ROC-AUC, PR-AUC, log-loss, Brier, and recall@budget together. If ROC-AUC is fine but PR-AUC or recall@budget is poor, switch `eval_metric` to `aucpr` so early stopping selects for your real objective. (Cost: 5 minutes. Rules in/out: metric mismatch.)

**Step 5 — Check calibration (numerics), only if you ship probabilities.** Plot the reliability curve and Brier score. If `scale_pos_weight`/resampling inflated the probabilities, fit isotonic calibration on the held-out eval set; AUC is unchanged, probabilities land on the diagonal. (Cost: 5 minutes. Rules in/out: imbalance-induced calibration distortion.)

**Step 6 — Confirm missing-value handling matches serving (systems).** Verify train-time and serve-time missing handling are identical (`np.nan` both sides, or the same declared `missing` sentinel). A no-leak, no-overfit prod gap is often this. (Cost: 5 minutes. Rules in/out: sentinel/skew.)

![A matrix figure mapping four gradient boosted model symptoms a widening train valid gap a high cross validated but low production score a flat AUC with good logloss and overconfident predictions each to a likely cause a one line confirming test and a fix from early stopping to a clean split to setting the eval metric to isotonic calibration](/imgs/blogs/gradient-boosting-and-imbalance-debugging-8.png)

The whole bisection is forty minutes and it almost never requires touching the model architecture. That is the deep lesson of GBDT debugging: the model is rarely the bug. The bug is in the *evaluation contract* around the model — the split, the stopping rule, the metric, the probability interpretation, the missing-value convention. XGBoost and LightGBM are extraordinarily good at fitting whatever you hand them, which is precisely why they will faithfully fit your leak, your noise, and your sentinel, and report a glowing number while doing it. Your job is not to make them fit better. It is to make sure the number they report is one you can ship.

## 12. Case studies and known signatures

A few documented, well-known patterns to calibrate your pattern-matching. Where I give a number I have tried to be accurate; where I am summarizing a class of result rather than one paper I say so.

**Confident learning finds label errors in benchmark *test* sets.** Northcutt, Athalye, and Mueller's 2021 work ("Pervasive Label Errors in Test Sets...") used confident learning (the `cleanlab` library) to estimate label-error rates across ten widely-used ML benchmarks and found an average of about **3.4%** errors, including on test sets people treat as ground truth. The relevance to GBDTs: when your model's residuals concentrate on a handful of high-loss training rows, the suspect is not always overfitting — it can be *label noise*, and loss-ranking those rows (the boosting analogue of confident learning) often surfaces genuine mislabels. The sibling [garbage in, finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise) builds the detector out; the headline is that even revered benchmarks carry several percent label noise, so your messy production table certainly does.

**Kaggle leakage post-mortems.** Competition write-ups are a catalogue of the leaks in this post. Recurring patterns: an `id` column whose ordering encoded the target (the test set's ids continued a sorted sequence), timestamps that revealed label timing, and "magic features" that turned out to be target-derived. The lesson that generalizes: a leaderboard score that jumps implausibly is a leak until proven otherwise, and the `gain`/SHAP-versus-`weight` discrepancy is the fastest way to find the culprit feature. The structural fix — split by group/time, never by random row when entities or time matter — is the through-line of [cross-validation done wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong).

**ROC-AUC's optimism under heavy imbalance.** This is a documented property, not folklore: Davis and Goadrich's 2006 ICML paper ("The Relationship Between Precision-Recall and ROC Curves") shows that a curve dominating in ROC space does not imply dominance in PR space, and that PR curves are the more informative view when the negative class is large. The practical upshot for fraud/defect GBDTs: report PR-AUC and recall-at-budget, and early-stop on `aucpr`; a high ROC-AUC on a 1%-prevalence problem is the single most over-trusted number in tabular ML.

**Scale_pos_weight and calibration.** The probability distortion from class reweighting is well understood in the calibration literature; isotonic regression (Zadrozny and Elkan, 2002) and Platt scaling (Platt, 1999) are the standard post-hoc fixes, both monotone so they preserve ranking. If you have ever seen a fraud model with excellent AUC whose expected-loss numbers were systematically too high, this is almost certainly why, and a held-out isotonic fit is the standard remedy.

**LightGBM's `num_leaves` versus XGBoost's `max_depth` — a porting hazard.** A genuinely common production incident, not from one paper but from the libraries' design: teams migrate a tuned XGBoost config to LightGBM (often for speed) by copying `max_depth=6` over and leaving `num_leaves` at its default of 31, or worse, raising it for "more capacity" to 255. Because LightGBM grows leaf-wise rather than level-wise, `num_leaves=255` corresponds to a far deeper effective tree than `max_depth=6` ever produced, so the "same" config silently overfits and the train-valid gap blows open. The fix is to set `num_leaves` no larger than roughly $2^{\text{max\_depth}}$ and treat it, not `max_depth`, as the real capacity knob in LightGBM. The signature is unmistakable once you plot the curve — a train loss collapsing to near zero with a valid loss that bottomed early and rose — but invisible if you trust the single CV number the migration reported.

**The "8% lift" that was a leak.** A recurring shape in post-mortems: a new feature is added, CV AUC jumps 0.08, the feature ships, and production does not move at all. The fast diagnosis is the `gain`-versus-`weight` discrepancy plus an availability check — the new feature is almost always either target-derived or computed from data not available at scoring time. The rule that prevents the whole class: every feature must pass a "could I have computed this at the moment of prediction, using only data that existed then" test before it enters the model. A feature that fails that test is a leak no matter how much it lifts CV.

## 13. When this is (and isn't) your bug

Bisection is as much about *ruling out* as ruling in. A few decisive calls:

- **If two seeded runs already differ, stop everything and fix reproducibility first.** Every before/after below is meaningless until `max abs diff == 0` across runs at a fixed thread count. A noisy delta is not evidence.
- **If the train-valid curves are close and both still falling at `best_iteration`, it is not an overfit — stop adding regularization.** Over-regularizing a model that was not overfitting only costs you accuracy. The gap, not the absolute valid loss, is the overfit signal.
- **If ROC-AUC is high but PR-AUC and recall@budget are fine too, the metric is not your bug** — the model genuinely discriminates. Do not switch metrics for its own sake; switch only when the metrics *disagree* in the direction of your real objective.
- **If you only ship a ranking or a threshold, calibration is irrelevant — leave the inflated probabilities alone.** Calibration matters only when a downstream consumer reads the number *as a probability*. Calibrating a pure ranker is wasted effort and a wasted held-out set.
- **A high-cardinality feature topping `weight` importance is usually not a leak** — it is just splittable. Convict on `gain`/SHAP *plus* a domain availability check, not on `weight` alone, or you will drop a useful feature.
- **A smooth, monotone training loss is not a success signal for a GBDT** — it is the default behavior of boosting and tells you nothing about generalization. Only the *validation* curve, and the gap to training, carry that information.
- **If overfit-one-feature passes (a model trivially memorizes a tiny subset) and the curves still diverge on full data, the bug is data or evaluation, not capacity.** The make-it-fail-small discipline from [the taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) applies to trees too: a model that *can* fit but *won't* generalize points at the contract around it, not the model.

## 14. Key takeaways

- **Boosting fits residuals stage by stage with no internal stop, so too many rounds overfit with certainty.** The validation loss is U-shaped in rounds; its minimum is the right tree count, and you read it off a train-versus-valid plot, never off the final number.
- **Early stopping needs a clean eval set, and a leaked one amplifies the leak** — it inflates both the metric *and* `best_iteration`. Split three ways (train/eval/test), check for duplicates and a shuffle-target signal, and prefer `GroupKFold`/time splits when entities or time matter.
- **`scale_pos_weight` (and resampling) rescale the gradient to recover rare-class recall, but provably inflate predicted probabilities** by training on a fabricated base rate. AUC is blind to this; the reliability curve and Brier score expose it; isotonic calibration fixes it without changing the ranking.
- **Match the eval metric to the goal.** Optimize/early-stop on `aucpr` (PR-AUC) or recall@budget under imbalance, not ROC-AUC, which stays optimistically high while the alert queue fills with false positives.
- **Regularize with the right knob for the mechanism**: `max_depth`/`num_leaves` cap per-tree capacity, `min_child_weight` blocks noise-leaves, `subsample`/`colsample` decorrelate trees, `lambda`/`alpha` shrink leaf weights. Read which one closes the train-valid gap.
- **Read `gain` and SHAP, not the default `weight` importance**, to find a leaked dominating feature — and remember importance is correlation, not causation or availability.
- **Let the booster see missing as missing.** A pre-filled sentinel (`-999`, `0`) becomes a real value the model splits on and a train-serve skew waiting to happen; pass `np.nan` or declare `missing=`.
- **Pin `seed` and `num_threads` and the determinism flag before trusting any before/after** — an unreproducible run cannot be debugged.

## 15. Further reading

- Chen and Guestrin, **"XGBoost: A Scalable Tree Boosting System"** (2016) — the second-order objective, the regularized leaf weight, and the sparsity-aware (missing-value) split-finding derived in §2 and §9.
- Ke et al., **"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"** (2017) — leaf-wise growth, histogram binning, and why `num_leaves` (not `max_depth`) is the real capacity control.
- Friedman, **"Greedy Function Approximation: A Gradient Boosting Machine"** (2001) and **"Stochastic Gradient Boosting"** (2002) — the residual-fitting view and `subsample` as variance reduction.
- Davis and Goadrich, **"The Relationship Between Precision-Recall and ROC Curves"** (ICML 2006) — why PR-AUC beats ROC-AUC under heavy imbalance (§5).
- Niculescu-Mizil and Caruana, **"Predicting Good Probabilities with Supervised Learning"** (ICML 2005), with Platt (1999) and Zadrozny and Elkan (2002) — calibration and the monotone post-hoc fixes used in §7.
- Lin et al., **"Focal Loss for Dense Object Detection"** (ICCV 2017) — the difficulty-based reweighting alternative to a constant class weight, discussed in §3 as a per-example versus per-class trade-off.
- Northcutt, Athalye, and Mueller, **"Pervasive Label Errors in Test Sets Destabilize ML Benchmarks"** (2021) and the `cleanlab` docs — confident learning and the ~3.4% benchmark label-noise figure (§11).
- The XGBoost and LightGBM parameter docs — the authoritative reference for `scale_pos_weight`, `is_unbalance`, `early_stopping_rounds`, `eval_metric`, `missing`, and the determinism flags.
- Within this series: the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the decision tree this instantiates), the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook), and the siblings [class imbalance and when accuracy lies](/blog/machine-learning/debugging-training/class-imbalance-and-when-accuracy-lies), [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), [cross-validation done wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong), [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying), and [categorical and feature bugs](/blog/machine-learning/debugging-training/categorical-and-feature-bugs).

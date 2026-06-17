---
title: "Tabular Data Leakage: The AUC That Drops 0.2 When You Fix It"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why your gradient-boosted model scores a glorious 0.97 AUC offline and a miserable 0.78 in production — the six leakage mechanisms specific to tabular pipelines, the math of how each one inflates the score, and the runnable scikit-learn Pipeline, out-of-fold encoder, and leak detector that recover the honest number."
tags:
  [
    "debugging",
    "model-training",
    "data-leakage",
    "tabular",
    "cross-validation",
    "scikit-learn",
    "xgboost",
    "evaluation",
    "feature-engineering",
    "machine-learning",
    "finetuning",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/tabular-data-leakage-1.png"
---

There is a particular kind of silence that should make any tabular modeler nervous, and it sounds like applause. You run five-fold cross-validation on your gradient-boosted classifier, the mean AUC comes back at 0.97, the standard deviation across folds is a tight 0.004, and the feature importances look reasonable enough that you ship it. Three weeks later the model is live, scoring real customers, and the dashboard says the production AUC is 0.76. Nobody changed the code. The model did not "drift." The training distribution did not move. The 0.97 was never real. It was a measurement artifact, and the artifact has a name: leakage.

Tabular data leakage is the single most common reason a model that looked brilliant offline collapses in production, and it is the most embarrassing because it is almost always self-inflicted. You did not get unlucky. Somewhere in the pipeline, information that the model could not possibly have at prediction time leaked into the features it trained on — a scaler fit on the whole dataset before the split, a categorical column encoded by its target mean over every row including its own, an ID that happens to be correlated with the label, a column populated only after the outcome was known, a duplicate row sitting on both sides of the split. The model dutifully learned to read that smuggled future, the validation set contained the same smuggled future, and so the score was a self-fulfilling prophecy. Production does not get to see the future, so production gets the truth.

![Vertical stack of the six tabular leakage mechanisms, each annotated with how it inflates the score, from fit-before-split through target encoding, proxy columns, time leakage, duplicates, and cross-split feature engineering](/imgs/blogs/tabular-data-leakage-1.png)

This post is the tabular-specific deep dive on leakage. There is a broader, modality-agnostic treatment of the topic in [Data Leakage: The Silent Killer of Your Validation Score](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) — read that for the general theory of how any leak inflates any metric. Here we get concrete and stay concrete: the exact `scikit-learn` `Pipeline` and `ColumnTransformer` patterns that prevent leakage, the out-of-fold target encoder you should be using instead of the naive one, the per-feature target-correlation scan and permutation-importance test that pinpoint the offending column, and the before-versus-after where moving preprocessing inside the cross-validation loop drops the AUC from 0.97 to 0.78. We will quantify the inflation, not just assert it. And we will tie it back to the spine of this whole series: a training bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and you **bisect** to the right one before touching code. Leakage lives at the seam between data and evaluation, which is exactly why it is so hard to see: both the data pipeline and the scoring pipeline agree on the lie. By the end you will be able to take any tabular model with a suspiciously good cross-validation score and, in minutes, decide whether it is real — and if it is not, find the exact column or pipeline step that is lying to you. For the master decision tree this all plugs into, see [A Taxonomy of Training and Finetuning Bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).

## 1. The symptom: a score too good to be true

Let me describe the failing run precisely, because precision is the whole game in debugging. You have a binary classification problem — say, predicting whether a loan will default — with 50,000 rows and 30 features, a mix of numeric columns (income, loan amount, credit utilization) and categorical columns (employment type, state, a few thousand distinct ZIP codes). The base rate of the positive class is about 18%. You build a reasonable pipeline: impute missing values, scale the numerics, encode the categoricals, fit an XGBoost classifier, and evaluate with five-fold cross-validation.

The number comes back at **0.97 AUC**. Now, 0.97 is not impossible. Some problems genuinely are that separable. But you have a working knowledge of loan default, and you know that the genuinely predictive signal — debt-to-income, prior delinquencies, utilization — gets you to maybe 0.80 on a good day, because human financial behavior is noisy and the future is genuinely uncertain. An AUC of 0.97 on default prediction is the model claiming it can rank a defaulter above a non-defaulter 97 times out of 100. That is not a credit model; that is a model that has seen the answer key.

The first instinct of an inexperienced modeler is to celebrate. The first instinct of someone who has been burned is to get suspicious, and suspicion is a skill you can systematize. Here is the smell-test checklist, in rough order of how strongly each one screams "leak":

| Smell | What it suggests | First confirming test |
| --- | --- | --- |
| AUC far above the domain's known ceiling | A feature is a proxy for the label | Per-feature target correlation scan |
| One feature dominates importance (>0.5 of total) | That feature is a leak or near-duplicate of `y` | Drop it, re-score; permutation importance |
| CV score tight and high, prod score much lower | Train/test contamination at fit time | Move preprocessing inside the CV fold |
| Score drops sharply when you shuffle row order | Temporal or index leakage | `TimeSeriesSplit`, check index–label correlation |
| Score drops when you deduplicate | Duplicate rows straddle the split | Hash rows, count cross-split collisions |

The discipline is to treat a great score as a hypothesis to be falsified, not a result to be banked. Every one of those tests is cheap — minutes, not hours — and the cost of skipping them is a model that fails in production where it is expensive and visible. As the spine of this series puts it: **read the instruments before you trust the number.** The AUC is an instrument, and a pegged instrument usually means a broken sensor, not a perfect engine.

It helps to have an internal model of how good a model *can* be on a given kind of problem, because the smell test is calibrated against that prior. Credit default, fraud, churn, click-through, conversion — these are noisy human-behavior problems where the irreducible uncertainty is large, and a well-built model lands somewhere in the 0.70–0.85 AUC range. Problems with a strong physical or deterministic structure — predicting whether a transaction will clear given the account balance, or whether a sensor exceeds a hard threshold — can legitimately hit 0.95+. The skill is knowing which kind of problem you have. When a noisy behavioral problem reports a deterministic-problem score, the gap between your prior and the result is the size of the lie, and that gap is your first quantitative clue about how much leakage you are hunting. A 0.97 on default prediction is not "a great model"; it is a 0.17-point anomaly that needs an explanation, and the only explanations are a genuine miracle feature (vanishingly rare) or a leak (overwhelmingly likely).

There is also a meta-signal in how the score *behaves* under small perturbations, and it is worth internalizing as part of the smell test. A genuine model degrades gracefully: drop a real feature and the AUC slips a little; add noise to the inputs and it slips a little; change the random seed and it wobbles within its fold-to-fold standard deviation. A leaked model is brittle in a telling way: drop the one leaked feature and the AUC craters from 0.97 to 0.79 in a single step. Real predictive power is distributed across many features, each contributing a slice; leakage concentrates power in one or two columns that carry the answer. So the perturbation profile — graceful versus cliff-edge — is itself a diagnostic you can run before any formal test, just by deleting columns and watching the number.

### Why the honest number matters more than the impressive one

There is a temptation to argue with reality here — to think that a 0.97 in cross-validation "must mean something," that maybe production is the anomaly. It almost never is. The cross-validation score is supposed to be an unbiased estimate of out-of-sample performance. When it is wildly higher than production, the estimate is biased, and a biased estimate is worse than no estimate, because it drives decisions: you set thresholds, allocate budget, and make promises based on 0.97 when the real operating point is 0.78. The honest 0.78 is not a disappointment; it is the number you needed all along. It tells you the true cost of a false positive, the true recall at your chosen threshold, and whether the project is even viable. Leakage does not just inflate a metric — it corrupts every downstream decision that depends on knowing how good the model really is.

### The science: how a leak biases AUC, formally

It is worth being precise about *why* a leak moves the number, because the mechanism explains both the direction and the magnitude. AUC is the probability that a randomly chosen positive row is scored above a randomly chosen negative row: $\text{AUC} = \Pr(s_+ > s_-)$, where $s_+$ and $s_-$ are the model's scores for a positive and a negative example. A model with no leak ranks pairs using only the genuine signal, so its AUC reflects the true separability of the classes — call it $A_{\text{true}}$.

Now introduce a leaked feature that is, say, an $\epsilon$-corrupted copy of the label: it equals the true label with probability $1 - \epsilon$ and is random otherwise. On the contaminated evaluation set, the model can read this feature, so for the fraction $1 - \epsilon$ of pairs where the leak is reliable, it ranks them correctly with probability near 1; only the remaining fraction $\epsilon$ falls back on genuine signal. To first order, the observed AUC becomes approximately

$$A_{\text{obs}} \approx (1 - \epsilon) \cdot 1 + \epsilon \cdot A_{\text{true}} = A_{\text{true}} + (1 - \epsilon)(1 - A_{\text{true}}).$$

The inflation $(1-\epsilon)(1 - A_{\text{true}})$ is largest exactly when the leak is most reliable (small $\epsilon$) and when the honest model is weakest (small $A_{\text{true}}$). That is why leakage is so seductive on hard problems: a noisy problem with $A_{\text{true}} = 0.78$ and a near-perfect leak ($\epsilon \approx 0.1$) lands at roughly $0.78 + 0.9 \times 0.22 \approx 0.98$ — precisely the kind of jump we are debugging. The formula also says leakage cannot help a problem that is already nearly perfect: if $A_{\text{true}} = 0.99$, the most a leak can add is $0.01$. Leaks shout loudest on the hard problems where you most want to believe a good number.

The second statistical signature is variance, not bias. A leaked feature is shared by every cross-validation fold (it is a column in the data, present everywhere), so it lifts every fold's score by nearly the same amount. That synchronization *shrinks* the fold-to-fold variance: an honest hard problem has visible spread because each fold draws a different sample of a noisy target, but a leak imposes a common, near-deterministic lift that all folds share. So a leak's fingerprint is two-fold — the mean goes up *and* the standard deviation goes down — and the implausibly tight standard deviation is often the first thing to catch your eye, before you have even questioned the mean.

## 2. Leak one: preprocessing fit before the split

This is the most common leak in all of tabular machine learning, and it is so subtle that most tutorials teach it as the correct way to do things. The pattern looks like this:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

df = pd.read_csv("loans.csv")
X = df.drop(columns=["defaulted"])
y = df["defaulted"]

# THE LEAK: fit the scaler on the entire dataset, then split.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes("number"))

model = XGBClassifier(n_estimators=400, max_depth=5, eval_metric="auc")
scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
print(f"CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
# CV AUC: 0.9712 +/- 0.0041   <-- inflated
```

The bug is on the line that calls `scaler.fit_transform(X)` before cross-validation. The scaler computes the mean and standard deviation of every numeric column **using all 50,000 rows** — including the rows that will later become each fold's test set. When fold 3 holds out 10,000 rows for testing, the model is evaluated on rows whose scaling was computed with knowledge of those very rows. The test set's statistics have leaked into the transform.

![Directed graph showing two forks from a raw data frame, one fitting the scaler on all rows producing a fake AUC and one splitting first and fitting a Pipeline per fold producing an honest AUC](/imgs/blogs/tabular-data-leakage-2.png)

### The science: how much does fit-before-split actually inflate the score?

It is worth being precise about the magnitude, because the answer surprises people in both directions. For a simple `StandardScaler` on a large dataset, the leak from fit-before-split is usually **small** — on the order of a hundredth of an AUC point — because the mean and variance of a column estimated from 40,000 training rows are nearly identical to the same statistics estimated from all 50,000. The leaked information is just the test rows' tiny contribution to the global mean and variance, which is diluted across tens of thousands of rows.

But the magnitude depends entirely on which preprocessing step you fit before the split. The leak scales with how much the fitted transform can memorize about individual rows:

- **`StandardScaler` / `MinMaxScaler`** on a large dataset: small leak (a global mean and std change negligibly when you add 20% more rows). Often +0.005 to +0.02 AUC.
- **`SimpleImputer(strategy="mean")`** on a column with many missing values: similar, small, unless missingness is correlated with the label.
- **Feature selection** (e.g., `SelectKBest` by mutual information with `y` on the full data): **large** leak, because you are choosing which features to keep using the test labels. The selected features are guaranteed to look predictive on the test fold.
- **`PCA` fit on the full data**: moderate leak — the principal axes are oriented partly by the test rows.
- **Target / mean encoding** fit on the full data: **catastrophic** leak (this is leak two, below), often +0.10 to +0.20 AUC.

So "preprocessing before the split" is not one bug with one magnitude; it is a family whose severity is governed by a simple rule: **a transform leaks in proportion to how much it can encode the label or memorize individual rows.** A scaler memorizes almost nothing. A target encoder memorizes the label directly. Quantify it before you panic — and quantify it before you dismiss it.

To make the rule precise, think about the *capacity* of the leaked statistic — how many bits about individual rows or about the label it can carry across the split boundary. A `StandardScaler` learns two numbers per column (a mean and a standard deviation), and those two numbers are estimated from tens of thousands of rows, so any single test row's influence on them is on the order of $1/N$. The leaked capacity is essentially two real numbers per feature, shared globally — almost nothing about any individual row. That is why the scaler leak is tiny. A feature-selector fit on the full data, by contrast, makes a discrete choice (keep or drop) per feature using the test labels, and that choice is exactly the kind of decision that overfits the held-out fold. A `PCA` learns a rotation oriented partly by the test rows. And a target encoder learns one number *per category*, where rare categories are estimated from a handful of rows — so the per-category statistic carries a large fraction of those rows' labels. The leaked capacity scales with the granularity of what the transform learns relative to the number of rows that inform it. Coarse, globally-pooled statistics leak little; fine-grained, per-category or per-feature, label-informed statistics leak a lot. This single principle predicts the magnitude of every fit-before-split leak in the list above without running a single experiment, and it tells you which transforms to worry about first.

There is a subtlety worth flagging for imputation specifically. A mean or median imputer fit on the full data is usually a small leak — until the *missingness itself* carries label signal. If rows tend to be missing a field precisely when they are about to default (an applicant who stops providing information is a risk signal), then the imputed value and the missingness pattern are both label-correlated, and fitting the imputer on the full data leaks more than the bland "two numbers per column" analysis suggests. The defensive move is to add an explicit missingness indicator (`MissingIndicator` in scikit-learn) as a feature and let the model use the *pattern* of missingness directly and honestly, fit inside the fold, rather than smuggling it through a globally-fit imputed value.

#### Worked example: measuring the scaler leak

Take a synthetic dataset where the truly honest AUC is 0.78. Fit a `StandardScaler` on all rows, then cross-validate: you get 0.792. Fit the scaler inside the fold (via a `Pipeline`): you get 0.781. The leak from scaling alone is **+0.011 AUC** — real, but not the thing that took you from 0.78 to 0.97. The 0.97 has a bigger culprit, and the discipline of measuring each leak separately is what tells you the scaler is not it. Now add a `SelectKBest(k=10)` fit on the full data: the CV AUC jumps to 0.88, because feature selection on test labels is a genuine peek at the answer key. Fold the selection inside the `Pipeline` and it drops back to 0.79. You have now attributed +0.011 to the scaler and +0.09 to feature selection, and you did it by toggling one step at a time. That is bisection applied to leakage.

### The fix: a Pipeline that fits inside every fold

The fix is mechanical and absolute: **never fit any data-dependent transform outside the cross-validation loop.** Wrap every preprocessing step and the model in a single `Pipeline`, and pass the Pipeline to `cross_val_score`. Now `scikit-learn` re-fits the entire chain on each fold's training rows only, and the test rows are genuinely unseen.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes("object").columns.tolist()

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
])

pre = ColumnTransformer([
    ("num", numeric, num_cols),
    ("cat", categorical, cat_cols),
])

clf = Pipeline([
    ("pre", pre),
    ("select", SelectKBest(mutual_info_classif, k=20)),
    ("model", XGBClassifier(n_estimators=400, max_depth=5, eval_metric="auc")),
])

# Every transform now fits ONLY on each fold's training rows.
scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
print(f"Leak-safe CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
# Leak-safe CV AUC: 0.7831 +/- 0.0089   <-- honest
```

Two details matter. First, the standard deviation across folds went **up** (from 0.004 to 0.009) when we removed the leak. That is expected and healthy: an honest estimate has more variance than a contaminated one, because the leak artificially synchronizes the folds (they all share the same global statistics). A suspiciously tight CV spread is itself a smell of contamination. Second, the `ColumnTransformer` keeps numeric and categorical handling separate, which prevents a whole class of secondary bugs (scaling a one-hot column, imputing a string with a median). The Pipeline is not just leak prevention; it is the correct factoring of a tabular preprocessing chain.

![Before and after comparison showing CV AUC 0.97 with one dominant feature collapsing to CV AUC 0.78 with even importances once preprocessing is fit inside each fold](/imgs/blogs/tabular-data-leakage-3.png)

## 3. Leak two: target encoding, the classic inflator

If fit-before-split with a scaler is a paper cut, target encoding done wrong is a severed artery. It is the single biggest source of catastrophic leakage in tabular competitions and real pipelines alike, and it hides inside a technique that is genuinely useful when done correctly.

Target encoding (also called mean encoding or likelihood encoding) replaces a categorical value with the mean of the target for that category. For a ZIP-code column, the value "94103" becomes the average default rate among all rows with ZIP 94103. This is a powerful way to handle high-cardinality categoricals — far better than one-hot encoding 3,000 ZIP codes — because it compresses each category to a single, label-informed number. The problem is the phrase "among all rows."

### The science: why naive target encoding leaks its own label

Consider a category that appears exactly once in the dataset — a ZIP code with a single loan, which defaulted. The naive target encoder computes the mean target for that category over all rows containing it. There is one such row. Its target is 1. So the encoded value is 1.0. The model now sees a feature that is **literally equal to that row's own label.** For singleton categories, naive target encoding copies the label into the feature. For categories with two or three members, it copies a heavily label-informed average. The rarer the category, the more directly the encoded feature reveals the row's own outcome.

Make this quantitative. Suppose a category has $n$ members, of which $k$ are positive. The naive encoded value for any member of that category is $\hat{p} = k/n$. For a row in that category whose own label is $y_i$, the encoded value includes its own contribution. The leave-one-out encoded value — what the row "should" see if it did not get to peek at itself — is

$$\hat{p}_{-i} = \frac{k - y_i}{n - 1}.$$

The difference between the naive value and the honest leave-one-out value is

$$\hat{p} - \hat{p}_{-i} = \frac{k}{n} - \frac{k - y_i}{n - 1} = \frac{y_i - \hat{p}_{-i}}{n}.$$

For a singleton ($n = 1$) this is undefined for the leave-one-out term, but the naive value is exactly $y_i$ — a perfect copy. For small $n$, the row's own label moves the encoded value by $\mathcal{O}(1/n)$, which for high-cardinality columns (where most categories are rare) is a large, label-correlated signal. The model is not learning that a ZIP code is risky; it is reading a slightly fuzzed copy of the answer. High-cardinality plus naive target encoding equals near-perfect leakage, and that is exactly the recipe that pushes a 0.78 model to 0.97.

#### Worked example: the singleton ZIP code

Your loan dataset has 3,200 distinct ZIP codes across 50,000 rows. About 1,400 of them appear only once or twice. For every one of those ~1,500 rows, the naive ZIP target encoding is within $1/n$ of the row's own default flag — effectively the label itself for singletons. XGBoost finds this column and splits on it relentlessly: the ZIP-encoding feature lands at importance 0.62 (62% of total gain), and the CV AUC is 0.97. When you switch to out-of-fold encoding, the ZIP feature's importance falls to 0.11, it ranks behind debt-to-income and utilization where it belongs, and the AUC settles at 0.79. The 0.18-point gap was almost entirely the self-label leak from rare categories.

### The fix: out-of-fold (K-fold) target encoding

The correct way to target-encode is to compute each row's category mean from data that **excludes that row** — ideally from a different fold entirely. The standard recipe is K-fold (out-of-fold, "OOF") target encoding: split the training data into K inner folds; for each fold, encode its rows using category means computed on the **other** K−1 folds; for the test set, use category means computed on the full training data. Add smoothing toward the global prior so rare categories regress toward the base rate instead of toward their own noisy mean.

![Before and after comparison of naive target encoding leaking each row's own label versus out-of-fold encoding computing category means from folds that exclude the row](/imgs/blogs/tabular-data-leakage-6.png)

Here is a runnable, self-contained out-of-fold target encoder that you can drop into a Pipeline. It implements smoothing and OOF encoding correctly, and crucially it is a proper `scikit-learn` transformer so it re-fits inside each outer CV fold:

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-fold target encoding with prior smoothing.

    fit():  learns the full-train category means (used at transform time
            for unseen / test rows) plus the global prior.
    transform() during training is handled via fit_transform's OOF logic;
            at inference it applies the learned full-train means.
    """
    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=0):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    def _smoothed_means(self, s, y):
        prior = y.mean()
        stats = y.groupby(s).agg(["mean", "count"])
        # blend category mean toward the global prior; rare = trust prior
        w = stats["count"] / (stats["count"] + self.smoothing)
        return prior, (w * stats["mean"] + (1 - w) * prior)

    def fit(self, X, y):
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        X = X.reset_index(drop=True)
        self.prior_ = y.mean()
        self.maps_ = {}
        for c in self.cols:
            _, m = self._smoothed_means(X[c], y)
            self.maps_[c] = m
        return self

    def fit_transform(self, X, y, **kw):
        # OOF encoding for TRAINING rows: each row encoded from other folds.
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        X = X.reset_index(drop=True)
        self.fit(X, y)
        out = X.copy()
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for c in self.cols:
            enc = pd.Series(np.full(len(X), self.prior_), index=X.index)
            for tr, va in kf.split(X):
                _, m = self._smoothed_means(X[c].iloc[tr], y.iloc[tr])
                enc.iloc[va] = X[c].iloc[va].map(m).fillna(self.prior_).values
            out[c] = enc.values
        return out[self.cols] if False else out.assign(**{c: out[c] for c in self.cols})

    def transform(self, X):
        # Inference / test rows: use full-train means, fall back to prior.
        out = X.copy()
        for c in self.cols:
            out[c] = X[c].map(self.maps_[c]).fillna(self.prior_).values
        return out
```

The two non-obvious correctness points:

1. **`fit_transform` and `transform` behave differently on purpose.** During training, each row must be encoded from folds that exclude it (OOF). At inference, there is no "fold that excludes this row" — the row is new — so you use the full-train category means. `scikit-learn` calls `fit_transform` on the training portion of each outer fold and `transform` on the test portion, which is exactly the asymmetry you want. This is the single most misunderstood part of target encoding, and it is why a hand-rolled `df["zip_enc"] = df.groupby("zip")["y"].transform("mean")` is always wrong: it uses the same (leaky) logic for train and test.

2. **Smoothing regularizes rare categories.** The weight $w = n / (n + \alpha)$ pulls a category with few members toward the global prior. With $\alpha = 20$, a category seen 5 times gets weight $5/25 = 0.2$ on its own mean and 0.8 on the prior, so a noisy singleton cannot dominate. Smoothing does not by itself prevent leakage — only the OOF split does that — but it prevents the encoder from manufacturing high-variance, overfit features even within a fold. The mathematical form is a shrinkage estimator: $\hat{p}_{\text{smooth}} = w\,\hat{p}_{\text{cat}} + (1-w)\,\hat{p}_{\text{prior}}$, an empirical-Bayes blend between the category mean and the population mean.

The `category_encoders` library ships a production-grade `TargetEncoder`, `MEstimateEncoder`, and `LeaveOneOutEncoder` that implement these ideas; in practice, prefer a battle-tested implementation. But you should understand the recipe well enough to know whether the one you grabbed actually does OOF on the training rows — many do not by default, and that default is a leak. As of recent versions, scikit-learn itself ships `sklearn.preprocessing.TargetEncoder`, which does interval-based cross-fitting internally when used inside cross-validation — a good default precisely because it is hard to misuse.

### Why leave-one-out encoding is still not enough

A tempting shortcut is leave-one-out (LOO) encoding: encode each training row with its category mean computed over all *other* rows in that category, $\hat{p}_{-i} = (k - y_i)/(n - 1)$. This removes the row's own label, so surely it is safe? Not quite, and the reason is instructive. Consider a binary category that is perfectly balanced within itself but whose members are split into positives and negatives. Under LOO, a positive row sees $(k-1)/(n-1)$ and a negative row sees $k/(n-1)$ — two *different* values, and the difference is exactly $1/(n-1)$, pointing in the direction of the row's own label. A sufficiently deep tree can learn the threshold that separates the "positive-row encoding" from the "negative-row encoding" within a category, recovering the label from the encoding's residual structure. LOO leaks less than naive encoding, but it leaks a thin, exploitable signal through the very mechanism that was supposed to remove the leak. K-fold (OOF) encoding is more robust because the fold boundary breaks the deterministic relationship between a row's label and its encoded value: the row's encoding comes from an entirely different subset, so there is no fixed $1/(n-1)$ offset to exploit. When in doubt, prefer K-fold over leave-one-out, and add noise or stronger smoothing if your categories are very small.

### Where to put the encoder: inside the Pipeline, always

A final practical point that trips people up: the OOF encoder must live *inside* the same Pipeline as the model, so that the outer cross-validation re-fits it per outer fold. A common mistake is to OOF-encode once, save the encoded dataframe, and then cross-validate a model on it. That is better than naive encoding, but it has a residual leak: the full-train category means computed in `fit` (used for the test rows) were computed once over all the data you then split, so the outer test fold's rows contributed to the means applied to the outer training fold. The clean construction nests the OOF encoder under the outer CV — encoder fits on outer-train, transforms outer-train via inner OOF, and transforms outer-test via outer-train means. The Pipeline handles this for you if you let it; do not pre-bake the encoding.

## 4. Leak three: ID columns and post-outcome proxies

Some leaks are not about how you process the data; they are about which columns are in the data at all. Two flavors dominate, and both produce the AUC-0.99 "model that has seen the answer key" signature.

### ID and index columns that correlate with the label

You would not deliberately feed a row's primary key to a classifier. But IDs sneak in. Rows are sometimes ordered by outcome (all the defaults appended at the end of the file), so the **row index** is correlated with the label. A customer ID issued sequentially over time correlates with anything that trends over time, including the label. An "account number" might encode the branch, the product, or the signup cohort, each of which carries label signal. The classic tell: a feature that is monotonic in the row order (an index, a timestamp-derived counter, a sequential ID) lands high in importance for no domain reason.

The science here is simple but worth stating: any feature $f$ with high mutual information $I(f; y)$ will be used by the model, and the model does not care whether that information is causal or an artifact of how the file was assembled. If the data was sorted by label before you got it and you kept the index, you have handed the model a perfect feature. This is also why **shuffling the rows changes the score**: if a leak rides on row order (an index feature, a cumulative count, a not-truly-stratified split), shuffling destroys it, and a score that moves when you shuffle is a score built on order, not signal.

### Post-outcome columns: the field set after the label is known

This is the deadliest proxy leak and the hardest to spot because the column looks legitimate. In a default-prediction dataset you might find a column like `days_since_last_payment`, `collections_flag`, `account_status`, or `charge_off_amount`. These are populated **after** the outcome — a charged-off account has a charge-off amount precisely because it defaulted. A `collections_flag` is set when an account goes to collections, which only happens after default. Feeding these to the model is showing it the consequence and asking it to predict the cause. In production, at the moment you score a new loan, these fields are all empty or zero, so the model that leaned on them has nothing to lean on.

The general rule is a temporal one: **a feature is a leak if its value at training time was determined after the moment you will make the prediction in production.** The discipline is to write down, for every feature, "when is this value known, relative to the prediction time?" Any feature known only after the label is a leak, full stop, regardless of how predictive it is.

| Column | Looks like | Actually is | Known at prediction time? |
| --- | --- | --- | --- |
| `credit_utilization` | A feature | A feature | Yes — keep |
| `days_until_event` | A countdown | Defined by the event | No — leak |
| `collections_flag` | A risk signal | Set after default | No — leak |
| `charge_off_amount` | A severity feature | Only nonzero if defaulted | No — leak |
| `row_index` | An identifier | Correlated with sort order | No — drop |
| `last_status_update_ts` | A timestamp | Updated after outcome | No — leak |

### The diagnostic: a per-feature target-correlation and single-feature-AUC scan

The fastest way to catch both ID leaks and post-outcome proxies is to score **each feature on its own** against the target. A legitimate feature has modest single-feature predictive power; a leak has near-perfect power. Here is a leak detector that ranks every column by how well it alone predicts the label:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def leak_scan(X: pd.DataFrame, y: np.ndarray, top: int = 15) -> pd.DataFrame:
    """Rank features by single-feature AUC and |correlation| with y.
    A single feature whose AUC is near 1.0 is almost certainly a leak."""
    y = np.asarray(y)
    rows = []
    for c in X.columns:
        col = X[c]
        if col.dtype.kind in "biufc":          # numeric
            v = col.fillna(col.median()).values.astype(float)
        else:                                    # categorical -> codes
            v = LabelEncoder().fit_transform(col.astype(str)).astype(float)
        # AUC is rank-based, so a monotone feature scores high either way
        auc = roc_auc_score(y, v)
        auc = max(auc, 1 - auc)                  # direction-agnostic
        corr = np.corrcoef(v, y)[0, 1]
        rows.append((c, auc, abs(corr)))
    out = (pd.DataFrame(rows, columns=["feature", "single_auc", "abs_corr"])
             .sort_values("single_auc", ascending=False)
             .reset_index(drop=True))
    return out.head(top)

scan = leak_scan(X, y)
print(scan.to_string(index=False))
# feature              single_auc  abs_corr
# charge_off_amount        0.992     0.71     <-- post-outcome leak
# collections_flag         0.961     0.66     <-- post-outcome leak
# zip_target_enc           0.943     0.55     <-- naive target-encoding leak
# debt_to_income           0.704     0.31     <-- legitimate
# credit_utilization       0.689     0.28     <-- legitimate
```

Any feature with a single-feature AUC above ~0.90 deserves an immediate audit: either it is a genuine, dominant, causal signal (rare in noisy domains) or it is a leak. The gap between the top leak (0.99) and the best legitimate feature (0.70) is the signature. Pair this with `permutation_importance` on the trained model — if removing one feature by shuffling it collapses the AUC from 0.97 to 0.79, that feature is doing all the work, and a single feature carrying a noisy domain should not be able to do that.

![Matrix mapping five tabular leak types to their mechanism and their confirming detector, from fit-before-split through duplicate rows](/imgs/blogs/tabular-data-leakage-4.png)

## 5. Leak four: time leakage and the future-predicts-the-past trap

Time leakage is leakage with a clock. It happens whenever your training rows contain information from the future relative to the prediction you are simulating, and it is endemic in any dataset with a temporal dimension: transactions, sensor readings, user events, prices, churn.

### The science: why random K-fold is wrong for time series

In random K-fold cross-validation, fold 3's test set is a random 20% of all rows, which means it contains rows from every time period — including time periods that come **before** some of the training rows. You are training on March and testing on February. In production you will never have March data when predicting February, because February comes first. The random split lets the model learn from the future to predict the past, which is impossible in deployment, so the score is optimistic.

Two distinct mechanisms inflate the score here. First, **direct temporal leakage**: a feature computed with a forward-looking window (a 7-day rolling average that includes future days, a "total spend over the account lifetime" that includes spend after the prediction date). Second, **distributional time leakage**: even with no forward-looking features, random K-fold leaks the fact that the world in the test fold's time period is similar to the training fold's time period, because they are interleaved. A model that has trained on every month will look better on a randomly held-out month than a model that, in production, has only ever seen the past.

Quantify it with an honest comparison. On a real churn dataset, random five-fold might report AUC 0.88, while a proper forward-chained `TimeSeriesSplit` reports 0.81. The 0.07 gap is the value of having seen the future — value you will not have in production. The forward-chained number is the one that will hold up.

### The fix: TimeSeriesSplit, an embargo, and forward-only features

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np

# Sort by time FIRST. The split assumes row order is chronological.
df = df.sort_values("application_date").reset_index(drop=True)
X = df.drop(columns=["defaulted", "application_date"])
y = df["defaulted"].values

# 5 expanding-window folds; each test fold is strictly AFTER its train rows.
tscv = TimeSeriesSplit(n_splits=5, gap=30)   # 30-row embargo between train/test

scores = cross_val_score(clf, X, y, cv=tscv, scoring="roc_auc")
print(f"Forward CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
# Forward CV AUC: 0.8093 +/- 0.0210   <-- honest, time-aware
```

The `gap` parameter inserts an **embargo**: a buffer of rows between the end of each training fold and the start of its test fold. The embargo matters when features have look-back windows. If a feature is a 30-day rolling average, then a training row dated June 1 "knows about" data up to June 1, and a test row dated June 2 whose 30-day window reaches back into late May overlaps the training window. Without a gap, the windows kiss at the boundary and leak. The embargo should be at least as long as your longest look-back window. This is the same idea as a purge in financial cross-validation, where you also remove training rows whose label window overlaps the test period.

There is a third, easy-to-miss form of time leakage that no splitter can save you from: **label-window overlap.** Suppose your label is "did this customer churn within the next 90 days?" Then the label for a row dated June 1 is determined by events through August 30. A training row dated July 15 — which a forward split happily puts in the training fold for a test row dated June 1 — has a label that was decided using the same August window. The two rows' label horizons overlap, so the training row's label carries information about the test row's future. The forward split looks correct (July 15 is "after" June 1), but the *labels* leak backward through their overlapping definition windows. The fix is a purge: remove from the training fold any row whose label-determination window overlaps the test fold's period, in addition to the embargo on feature windows. Whenever your label is defined over a future horizon, you must purge by that horizon, not just split by the prediction date.

| Time-leakage flavor | Mechanism | Fix |
| --- | --- | --- |
| Interleaved random folds | Train on the future, test on the past | `TimeSeriesSplit` (forward chaining) |
| Forward-looking feature window | A "rolling average" includes future days | Recompute features causally; add embargo |
| Label-window overlap | Train label decided over the test row's future | Purge by the label horizon |
| Distributional time drift | Random folds hide that prod is the future | Forward CV plus a time-based holdout |

For the deeper treatment of group and time-aware cross-validation — nested CV, purging, the optimism of tuning on the test fold — see the sibling post [Cross-Validation Done Wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong). The one-line rule to carry away: **if your data has a time order, your split must respect it, and your features must only look backward.**

![Tree decision diagram for choosing a tabular split and encoder based on whether the data has a time order, a shared entity, and high-cardinality categoricals](/imgs/blogs/tabular-data-leakage-7.png)

## 6. Leak five: duplicate and near-duplicate rows across the split

This one is mechanical and easy to miss. If the same row — or a near-copy of it — appears in both the training and the test set, the model can memorize it in training and "predict" it perfectly in test. Exact duplicates are common from joins that fan out, from re-ingested data, from logging the same event twice. Near-duplicates are subtler: two loan applications from the same person with one field changed, two sensor readings one millisecond apart, a record and its lightly-edited correction.

### The science: a duplicate is a memorization shortcut

A tree-based model with enough depth can memorize individual rows. If a test row has an exact twin in the training set, the model that memorized the twin gets the test row for free — not because it learned the pattern but because it learned that specific point. The inflation scales with the duplicate rate: if 10% of your rows are duplicated across the split and the model memorizes them perfectly, roughly 10% of your test predictions are memorized rather than generalized, which can lift AUC by several points depending on how hard the genuine problem is. On a base problem of AUC 0.78, a 10% cross-split duplicate rate might inflate to 0.83–0.85.

### The diagnostic: hash rows and count cross-split collisions

```python
import pandas as pd
from sklearn.model_selection import KFold

def count_cross_split_dups(X: pd.DataFrame, n_splits=5, subset=None):
    """Hash each row; count how many test rows have an exact twin in train,
    per fold. A nonzero count means duplicates straddle the split."""
    key = pd.util.hash_pandas_object(
        X[subset] if subset else X, index=False
    ).astype("int64")
    kf = KFold(n_splits, shuffle=True, random_state=0)
    total = 0
    for fold, (tr, va) in enumerate(kf.split(X)):
        train_keys = set(key.iloc[tr])
        collisions = key.iloc[va].isin(train_keys).sum()
        total += collisions
        print(f"fold {fold}: {collisions} test rows duplicated in train")
    print(f"TOTAL cross-split duplicate rows: {total}")
    return total

count_cross_split_dups(X)
# fold 0: 941 test rows duplicated in train
# fold 1: 977 ...
# TOTAL cross-split duplicate rows: 4863   <-- ~10% of the data
```

Once you have found duplicates, the fix is to deduplicate **before** splitting, or — if duplicates represent meaningful repeated entities (the same customer applying twice) — to treat the shared entity as a group and use `GroupKFold` so all of a customer's rows land on the same side of the split. That last point is the bridge to a closely related leak: group leakage, where rows are not identical but share a source (a patient with five images, a user with many sessions). For near-duplicates, hash a rounded or quantized version of the numeric columns, or use a similarity threshold on embeddings; exact hashing only catches exact twins.

Near-duplicate detection deserves its own treatment because exact hashing misses the most common real case: two records that are 99% identical with one field nudged. Three practical strategies, in increasing cost and recall:

1. **Quantized hashing.** Round every numeric column to a sensible precision (income to the nearest \$1,000, ratios to two decimals), bin continuous columns, then hash. Two records that differ only in the last digit of income now collide. This catches the "same application, re-keyed" duplicate cheaply.
2. **Key-subset hashing.** Hash only the columns that *identify* an entity — name, date of birth, address — and treat collisions as the same person even if their feature columns differ. This is the right tool when the duplicates are genuinely the same entity observed twice, and it feeds directly into `GroupKFold`.
3. **Embedding nearest-neighbors.** Embed each row (a simple concatenation of normalized features, or a learned representation) and flag pairs within a small cosine distance across the split. This catches semantic near-duplicates that no exact or quantized hash would, at the cost of an approximate nearest-neighbor search.

The honest way to report the impact is to deduplicate, re-run, and quote both numbers — "0.84 with cross-split duplicates, 0.79 after dedup" — so the reader knows exactly how much of the original score was memorization. Hiding the dedup step and quoting only 0.84 is, itself, a form of dishonesty about the model's true generalization.

#### Worked example: the dedup that costs you 0.05

You hash every row and find that 4,863 of 50,000 rows (9.7%) have an exact twin straddling the split — the data was assembled by appending a re-ingested batch. Your leaky CV AUC is 0.84. You deduplicate to 45,137 unique rows and re-run with a clean split: AUC 0.79. The duplicates were worth 0.05 points of pure memorization. Crucially, 0.79 is the number that survives to production, because production rows are genuinely new — they do not have twins in your training set. The 0.05 you "lost" was never yours.

## 7. Leak six: feature engineering computed across the split

The final mechanism is the most insidious because it can coexist with a perfectly correct split. You split the data correctly, you use a Pipeline, you respect time — and then, before any of that, you computed a feature using the entire dataset. The leak rides inside the feature itself.

Common culprits:

- **Global aggregations**: `df["zip_default_rate"] = df.groupby("zip")["defaulted"].transform("mean")` — this is naive target encoding by another name, computed over all rows including the test rows. Even if you later split correctly, the feature already contains test-set label information.
- **Frequency / count features over the full data**: `df["zip_count"] = df.groupby("zip")["zip"].transform("count")` leaks the test-set distribution, though more weakly (counts, not labels).
- **Cross-row normalizations**: ranking or percentile-izing a column over the whole dataset (`df["income_pct"] = df["income"].rank(pct=True)`) uses test rows to compute each train row's percentile.
- **Target-derived bins**: discretizing a feature by its relationship to the label over all data.

### The science: the feature is the leak vector

The split is correct, but the feature was manufactured with a function that took all 50,000 rows as input. The split happens downstream of the contamination, so it cannot help — the test rows' labels are already baked into the train rows' features. The rule generalizes the Pipeline rule: **any computation that touches the label or aggregates across rows must be fit inside the cross-validation fold, not before it.** If you cannot express a feature as a fitted transformer, compute it inside a custom transformer's `fit`/`transform` so it sees only training rows during `fit`.

The diagnostic is the same per-feature scan from leak three plus a structural audit: for each engineered feature, ask "does this value for row $i$ depend on rows other than $i$?" If yes, and especially if it depends on their labels, it must be computed out-of-fold.

#### Worked example: the count feature that quietly helped

You add `zip_count` (how many loans share each ZIP) as a feature, computed over the full dataset. It is not a label leak — counts do not see `y` — but it does leak the test set's distribution: a ZIP that is common in the test fold gets a count that reflects test rows. The inflation is tiny here, +0.004 AUC, because the count is only weakly related to the label. You confirm it is benign by computing the count inside the fold (over training rows only) and seeing the score barely move. Not every cross-split feature is catastrophic; the discipline is to measure each one, and the measurement is cheap.

![Two-by-three grid contrasting fitting preprocessing on all rows which taints every fold against fitting per fold which keeps test rows unseen and the score honest](/imgs/blogs/tabular-data-leakage-8.png)

## 8. Gradient boosting's own leakage trap: early stopping on a dirty validation set

Tabular work is dominated by gradient-boosted trees — XGBoost, LightGBM, CatBoost — and these libraries have a leakage trap that is specific to how they train: **early stopping.** Early stopping watches a validation set during training and halts when the validation metric stops improving, which is one of the most effective regularizers in all of tabular ML. But it makes the validation set part of the training procedure, and if that validation set is contaminated, the leak propagates in two compounding ways.

First, the obvious way: if your early-stopping validation set itself contains leaked features or duplicate rows from the training set, the model trains toward a contaminated target and reports a contaminated number. Second, and subtler: even with clean features, if you use the *same* held-out set both to early-stop and to report your final score, you have tuned the number of boosting rounds on that set, which makes it a training set, not a test set. The reported metric is optimistic because the round count was chosen to maximize it. This is the boosting-flavored version of "overfitting to the validation set," and it is everywhere.

### The science: early stopping is hyperparameter tuning in disguise

The number of boosting rounds is a hyperparameter, and early stopping selects it by reading the validation metric. Selecting any hyperparameter on a set and then reporting performance on the same set is biased upward by the multiple-comparisons problem: you tried hundreds of round counts (one per boosting iteration) and kept the best. With hundreds of "trials," even pure noise produces an inflated maximum. The inflation is usually modest for round count alone (a fraction of a point), but it stacks on top of every other leak, and it is pure measurement bias — the model is no better, the number is just wrong.

### The fix: three disjoint splits

The correct structure for boosting with early stopping uses three disjoint sets: a **training** set the trees fit on, a **validation** set for early stopping (and only early stopping), and a **test** set that nothing touches until the end. Inside cross-validation, this means each outer fold's training portion is itself split into an inner train/early-stop pair, and the outer fold's test portion is scored once.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def cv_with_clean_early_stopping(X, y, n_splits=5, seed=0):
    """Outer CV for the honest score; an inner split for early stopping.
    The outer test fold is NEVER used to choose the round count."""
    outer = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in outer.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        # inner split of the OUTER-TRAIN rows for early stopping only
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
        model = XGBClassifier(
            n_estimators=2000, max_depth=5, learning_rate=0.03,
            eval_metric="auc", early_stopping_rounds=50)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        # score the untouched outer-test fold ONCE
        p = model.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, p))
        print(f"  best_iteration={model.best_iteration}  fold AUC={aucs[-1]:.4f}")
    print(f"Honest CV AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    return np.mean(aucs)
```

The structural rule is simple: **the set that decides when to stop is not the set that reports how good you are.** If you only have one held-out set and you use it for early stopping, your reported number is optimistic; carve out a third set, or use nested cross-validation. This is also the place where the fit-before-split leak and the early-stopping leak conspire: if you target-encoded over all data *and* early-stop on a contaminated set, the two inflations multiply, and you can manufacture a 0.97 out of a 0.78 problem without a single post-outcome column. The deeper treatment of nested CV and tuning-set optimism lives in [Cross-Validation Done Wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong); the takeaway here is that gradient boosting's best feature, early stopping, is also a leakage surface you must wire up carefully.

#### Worked example: the early-stop inflation

You train XGBoost with `early_stopping_rounds=50` on a single 80/20 split and report the validation AUC at the best iteration: 0.811. Then you build the three-way structure above — fit on 64%, early-stop on 16%, report on a never-touched 20% — and the honest number is 0.792. The 0.019-point gap is the optimism of having chosen the round count (best_iteration was 740 out of a possible 2000) on the same set you reported. It is small, but it is exactly the kind of small, invisible bias that, stacked with three other small biases, turns a defensible 0.78 into an indefensible 0.97. Every layer of "I'll just reuse this split" adds a sliver of optimism, and the slivers compound.

## 9. The bisection: how to localize a leak in minutes

We have six mechanisms. The professional move is not to check them at random but to **bisect** — to run cheap tests in an order that rules out whole classes of leak before you pay for expensive ones. This is the master tool of the whole series applied to leakage: make-it-fail-small and read the instruments, in cost order.

![Left-to-right timeline of the five-step leak hunt in cost order, from the smell test through importance ranking, Pipeline-in-CV, out-of-fold re-encoding, to an honest refit](/imgs/blogs/tabular-data-leakage-5.png)

Here is the order, cheapest first:

1. **Smell test (seconds).** Is the AUC above the domain's known ceiling? Is the CV standard deviation suspiciously tiny? Did the score change when you shuffled rows? These cost nothing and tell you whether to even start.
2. **Single-feature AUC scan (one minute).** Run `leak_scan`. If one feature has single-feature AUC near 1.0, you have found a proxy or a naive-target-encoded column. This catches leaks three and the naive form of two immediately.
3. **Permutation importance (a few minutes).** Train once, then permute each feature and measure the AUC drop. If one feature's permutation drops AUC from 0.97 to 0.79, that feature is doing all the work. Combined with step 2 this localizes proxy leaks precisely.
4. **Move preprocessing into the Pipeline (minutes).** Re-run CV with everything inside a `Pipeline`. If the AUC falls, you had a fit-before-split leak (leak one). The size of the fall tells you which transform was the culprit — toggle them one at a time.
5. **Out-of-fold re-encode (minutes).** Replace naive target encoding with the OOF encoder. If the AUC falls, you had encoding leakage (leak two).
6. **Time and dedup checks (minutes).** Swap to `TimeSeriesSplit` if there is a clock; run `count_cross_split_dups`. Falling scores localize leaks four and five.

The discipline is that each step **changes exactly one thing** and you watch the AUC. A leak that survives step 4 but dies at step 5 is an encoding leak. A leak that dies at step 4 is a fit-before-split leak. A leak that only appears under random folds and vanishes under `TimeSeriesSplit` is temporal. You are bisecting the space of six mechanisms with five cheap tests, and you will usually have the answer before lunch.

### Adversarial validation: a second instrument

There is one more powerful, general detector worth its own mention: **adversarial validation.** Label every training row 0 and every test row 1, then train a classifier to distinguish train from test. If it cannot (AUC ≈ 0.5), train and test are drawn from the same distribution — good. If it can (AUC ≈ 0.9), some feature systematically differs between train and test, which is exactly the signature of a leak that rides on a distributional difference (a time-correlated ID, a column that exists in one split but not the other). The features the adversarial model finds most useful are your leak suspects.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

def adversarial_validation(X_train, X_test):
    """AUC near 0.5 = train/test indistinguishable (good).
    AUC near 1.0 = a feature separates the splits (leak suspect)."""
    combined = pd.concat([X_train, X_test], ignore_index=True)
    is_test = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    clf = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="auc")
    auc = cross_val_score(clf, combined, is_test, cv=5,
                          scoring="roc_auc").mean()
    clf.fit(combined, is_test)
    top = pd.Series(clf.feature_importances_, index=combined.columns)
    print(f"adversarial AUC: {auc:.3f}")
    print("most train/test-separating features:")
    print(top.sort_values(ascending=False).head(5).to_string())
    return auc

# adversarial AUC: 0.94
# most train/test-separating features:
# application_date_ordinal   0.41   <-- time separates the splits
# customer_id                0.22   <-- sequential ID leaks time
```

An adversarial AUC of 0.94 with `application_date` and `customer_id` at the top is a clear story: your splits differ by time, which means a random split was hiding a temporal structure, and the ID is a stand-in for time. The fix is `TimeSeriesSplit`, and adversarial validation found it without you guessing.

### A full bisection narrative

Let me run the bisection end to end on the loan model, narrating each instrument reading, because the *order* and the *one-thing-at-a-time* discipline are the whole skill.

You start at CV AUC 0.97 with a standard deviation of 0.003 across folds. **Smell test:** the domain ceiling for default prediction is ~0.82, the score is 0.15 above it, and the fold spread is implausibly tight for 50k noisy rows. Two smells, both pointing at a leak. You proceed.

**Single-feature scan (one minute).** You run `leak_scan` and the top of the table is `charge_off_amount` at single-feature AUC 0.99, `collections_flag` at 0.96, and `zip_target_enc` at 0.94, with the best legitimate feature (`debt_to_income`) trailing at 0.70. Three suspects, two of them obviously post-outcome fields by their names. You do not fix anything yet — you keep diagnosing so you learn the full picture.

**Permutation importance (a few minutes).** You train the model once and permute each feature. Permuting `charge_off_amount` drops the AUC from 0.97 to 0.92; permuting `zip_target_enc` drops it to 0.86; permuting both plus `collections_flag` drops it to 0.81. The three suspects together carry almost all the lift, which confirms the single-feature scan and tells you the genuine model under the leaks is around 0.80.

**Fix one thing — drop the post-outcome columns.** You remove `charge_off_amount` and `collections_flag` (they are not available at scoring time anyway) and re-run: CV AUC 0.931. The drop of 0.039 attributes that much to the proxy leak. The score is still too high, and `zip_target_enc` is still there.

**Fix the next thing — OOF-encode ZIP.** You swap naive target encoding for the OOF encoder inside the Pipeline: CV AUC 0.812. The 0.12 drop is the encoding leak, the biggest single contributor, exactly as the section-3 math predicted for a high-cardinality column with many rare categories.

**Fix the next thing — Pipeline the scaler and selector.** You move the imputer, scaler, and feature selector inside the Pipeline: 0.798. A 0.014 drop, consistent with the small fit-before-split leak.

**Stress-test the remaining mechanisms.** You hash rows and find 4,863 cross-split duplicates; dedup brings you to 0.785. You swap to `TimeSeriesSplit`; 0.781, a tiny additional drop because the duplicates and proxies were carrying most of the temporal signal already. Adversarial validation now reports 0.55 — train and test are finally indistinguishable. Every instrument reads clean.

The discipline that made this work: you changed exactly one thing at each step and watched a single number. At no point did you fix two leaks at once, because then you could not attribute the drop. The final 0.781 matches the production 0.78 you started with, which is the proof that the hunt is complete — not because you ran out of ideas, but because the offline number now equals the online number.

### Stress tests: what if it is not where you think?

Bisection also means knowing how the signatures shift under different conditions, so you do not chase the wrong leak. What if the dataset is *small* (a few thousand rows)? Then the naive-encoding leak is even worse, because more categories are rare, and the fit-before-split scaler leak grows too, because each test row is a larger fraction of the global statistics. What if the problem is *deep learning on tabular data* (a TabNet or an MLP) rather than trees? The leak mechanisms are identical — preprocessing-before-split, target encoding, proxies, time, duplicates — but the symptom can be muddier, because a neural net may not concentrate importance in one feature the way a tree does; lean harder on the single-feature scan and adversarial validation, which are model-agnostic. What if the leak only shows up at *scoring time* and not in CV? That is the train/serve skew variant — a feature computed differently in the serving pipeline than in training — which is a sibling failure covered in [Categorical and Feature Bugs](/blog/machine-learning/debugging-training/categorical-and-feature-bugs); the tell is that CV is honest but production still disappoints, which points to the feature computation, not the split. Each "what if" is just a reweighting of which instrument is most informative — the bisection itself does not change.

## 10. Before and after: the full recovery

Let me put the whole recovery in one table, because the before-versus-after is the evidence that makes the abstract concrete. This is a composite of the mechanisms above on the running loan dataset, where the genuinely honest AUC is 0.78.

| Step | Change made | CV AUC | What it revealed |
| --- | --- | --- | --- |
| 0 | Original leaky pipeline | 0.970 | The suspicious starting point |
| 1 | Drop `charge_off_amount`, `collections_flag` | 0.931 | Post-outcome proxies (leak 3) worth ~0.04 |
| 2 | OOF target-encode ZIP instead of naive | 0.812 | Naive encoding (leak 2) worth ~0.12 |
| 3 | Move scaler/imputer/selector into Pipeline | 0.798 | Fit-before-split (leak 1) worth ~0.014 |
| 4 | Deduplicate cross-split rows | 0.785 | Duplicates (leak 5) worth ~0.013 |
| 5 | Switch to TimeSeriesSplit | 0.781 | Temporal leak (leak 4) worth ~0.004 here |
| — | **Final honest pipeline** | **0.781** | Matches production AUC 0.78 |

Two things stand out. First, the leaks are **additive and stackable** — no single fix explained the whole 0.19-point gap; it was four real leaks plus two small ones, each found by changing one thing and watching the instrument. Second, the final 0.781 matches the production number that started this whole investigation. That match is the proof that you found all the leaks: an honest cross-validation score is, by definition, one that production reproduces. When CV and production agree, you can finally trust the number — and trust the threshold, the cost analysis, and every decision that depends on them.

### Measuring it honestly

How do you confirm in practice that the new number is the right one, before you ship? Three checks. First, a strict **time-based holdout**: hold out the most recent N% of data by date, train on everything before, score on the holdout. This simulates production exactly — train on the past, predict the future — and its number should match your forward CV. Second, a **fresh, never-touched test set** that no preprocessing, no feature selection, no hyperparameter search ever saw. If its score matches CV, you have no leak into model selection (the subtle "overfitting to the validation set" failure, covered in [Cross-Validation Done Wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong)). Third, the **production shadow**: log the model's scores on live traffic for a week before you act on them, and compare the realized AUC to your offline estimate. If they agree, the investigation is closed.

## 11. Case studies and known signatures

Leakage is not a hypothetical; it has a documented history of embarrassing very good teams, and the patterns repeat. Here are real, well-known signatures — described accurately, with the lesson each one teaches.

### The Kaggle leakage post-mortems

Kaggle has produced a long line of competitions where the leak, not the model, decided the leaderboard, and the community's post-mortems are some of the best leakage education available. The recurring pattern: a winning solution scores far above what the problem should allow, and the write-up reveals not a clever model but a discovered leak — a row ID correlated with the target, a file ordering that encoded the label, a feature that was a post-outcome field, or a metadata column (a timestamp, an index) that separated the public and private test sets. The organizers of several competitions have had to re-run scoring after the fact because a leak made the public leaderboard meaningless. The durable lesson, stated by many top competitors, is blunt: **the first thing you do on a new tabular dataset is hunt for leaks, because if there is one, it dominates everything else you could do.** A 0.001 model improvement is irrelevant next to a feature that is secretly the answer. The discipline of adversarial validation became popular precisely because Kaggle datasets so often had train/test distribution differences that signaled exploitable structure.

### Target encoding without out-of-fold: the importance signature

The most reproducible real signature is the one we derived in section 3. When a high-cardinality categorical is target-encoded naively (over all rows), it shows up with a dominating feature importance — often more than half the model's total gain — and a single-feature AUC near the model's full AUC. The fix (out-of-fold encoding) drops that feature's importance by 5–10× and the model's AUC by 0.10–0.20 depending on the cardinality. This is so consistent that "one categorical-encoding feature with runaway importance" is a near-certain diagnosis of naive target-encoding leakage, before you read a single line of the encoding code.

### Medical and scientific leakage: the duplicate-patient trap

In medical imaging and clinical tabular data, the canonical leak is group leakage: the same patient contributes multiple rows (multiple scans, multiple visits), and a random split puts some of a patient's rows in train and some in test. The model memorizes patient-specific quirks and "predicts" the held-out rows of patients it has already seen. Published audits of medical machine-learning papers have repeatedly found that a meaningful fraction of reported results used patient-leaking splits, and that fixing the split to `GroupKFold` by patient drops the reported metric substantially. The lesson generalizes far beyond medicine: **whenever a single real-world entity generates multiple rows, the entity, not the row, is the unit that must not straddle the split.** This is the tabular cousin of the duplicate-row leak in section 6, and it is why `GroupKFold` exists.

### The "too clean to be true" determinism tell

A final, subtler signature worth naming: when removing a leak, the cross-validation **variance** goes up, not just the mean going down. A contaminated CV is artificially consistent across folds because every fold shares the same leaked global statistics. If your CV standard deviation is implausibly small for your dataset size — say ±0.002 on 50,000 rows with a hard problem — treat the tightness itself as a leak smell, independent of the mean. Honest cross-validation on a noisy problem has visible fold-to-fold spread.

### The recommendation-system timestamp leak

One more documented pattern, common in recommendation and ranking systems: features built from a user's *full* interaction history, including interactions that happen after the event you are predicting. A "user's average rating" or "items the user has clicked" computed over the entire log leaks the future into every training row, because the average includes ratings the user gave after the row's timestamp. The fix is point-in-time correctness: every feature must be computed using only data available strictly before the prediction timestamp, which usually means a careful "as-of" join against a timestamped feature store rather than a naive `groupby().mean()` over the whole table. The signature is the same as classic time leakage — random CV looks great, a strict time-based holdout collapses — but the cause is hidden inside an aggregate feature rather than in the split, which is why it survives even a correct `TimeSeriesSplit` unless the features themselves are computed as-of. Point-in-time correctness is the single hardest piece of leakage discipline to get right in production, and it is why mature ML platforms invest heavily in feature stores that enforce it automatically.

## 12. When this is (and isn't) your bug

Bisection is as much about ruling leakage out as ruling it in. Here is when a bad-looking result is *not* leakage, so you stop hunting in the wrong place.

- **A modest, plausible CV score that drops a little in production is probably distribution shift, not leakage.** If CV says 0.80 and production says 0.77, that is a normal generalization gap and ordinary drift, not a leak. Leakage produces *large* gaps (0.10+) and a CV score above the domain's plausible ceiling. For the small-gap case, look at covariate shift and train-serving skew instead.
- **If the single-feature scan is flat (no feature above ~0.75) and the Pipeline-in-CV score does not move, you do not have a tabular leak — you have a hard problem.** Stop looking for a leak and start improving features or accepting the ceiling. Not every disappointing model is leaking; some problems are just hard, and 0.72 is the honest answer.
- **A high score that survives a strict time-based holdout and a fresh never-touched test set is probably real.** If you have done the discipline — Pipeline, OOF encoding, forward split, dedup — and a held-out-by-date test still says 0.90, then maybe your problem really is that separable. Believe it, but keep the production shadow check.
- **NaN-then-crash or a loss that diverges is numerics, not leakage.** Leakage never crashes; it silently inflates. If your training is unstable, that is a different chapter ([Hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) territory), not this one.
- **Class imbalance can make AUC look better than the model is useful, but that is not leakage.** A 0.95 AUC on a 1%-positive problem can still have terrible precision at any usable threshold. That is a metric-choice issue — measure PR-AUC and per-class recall — not a leak. The two failures look different: leakage inflates *all* metrics together (a leaked model has great precision *and* recall on the contaminated test set), while imbalance inflates AUC specifically while precision stays poor.

The decision rule: **a leak is large, silent, and shrinks dramatically when you move information across the split boundary correctly.** If your symptom is small, or crashes, or is specific to one metric, look elsewhere first.

## 13. Building a leak-resistant pipeline from day one

The best debugging is the kind you never have to do, and tabular leakage is highly preventable with a few structural habits. Adopt these and most of the six mechanisms become impossible by construction.

1. **Split first, always.** The very first operation after loading data is the split (or the CV-fold iterator). Nothing data-dependent happens before it. If you find yourself calling `.fit()` or `.transform()` or `.groupby(...)["y"].transform(...)` before the split, stop.
2. **Everything data-dependent lives in a Pipeline.** Imputation, scaling, encoding, feature selection, dimensionality reduction — all of it goes inside a `Pipeline`/`ColumnTransformer` that `cross_val_score` re-fits per fold. If a transform cannot be a fitted transformer, write a custom one so it sees only training rows in `fit`.
3. **Write the feature provenance down.** For every feature, record when its value is known relative to prediction time. Any feature known only after the label is deleted, no exceptions, regardless of importance. This single habit kills all post-outcome proxy leaks.
4. **Use the right splitter for the data's structure.** Time order → `TimeSeriesSplit` with an embargo. Shared entity → `GroupKFold`. Only truly independent rows → random `KFold`. The decision tree in figure 7 is the whole rule.
5. **OOF-encode every target-derived feature.** Never `groupby("cat")["y"].transform("mean")`. Use a proper out-of-fold encoder, or `category_encoders` with cross-fitting, inside the Pipeline.
6. **Keep a never-touched test set.** One slice of data that no preprocessing, no selection, no tuning, no eyeballing ever sees until the very end. Its score is the closest you get to production before production.
7. **Run the leak scan as a CI gate.** `leak_scan` and `adversarial_validation` are fast enough to run on every dataset version automatically. A feature crossing single-feature AUC 0.9, or an adversarial AUC above 0.7, fails the build until a human signs off. Make the discipline a test, not a hope.

These habits do not slow you down; they speed you up, because the time you spend setting up a Pipeline correctly is a fraction of the time you spend in production firefighting a model that scored 0.97 and delivered 0.78. The fastest path to a model that ships is the one that never tells you a number you cannot trust.

## 14. Key takeaways

- **An AUC above your domain's known ceiling is a hypothesis to falsify, not a result to bank.** Treat 0.97 on a noisy problem as a leak until proven otherwise.
- **Fit every data-dependent transform inside the cross-validation fold, never before the split.** A single `Pipeline` passed to `cross_val_score` makes most leakage structurally impossible; the leak's size equals how much the transform can encode the label.
- **Naive target/mean encoding leaks each row's own label** through rare categories; the inflation is $\mathcal{O}(1/n)$ per category and can be +0.10 to +0.20 AUC. Use out-of-fold encoding with prior smoothing.
- **A feature is a leak if its value was determined after the prediction moment.** Post-outcome columns (`charge_off_amount`, `collections_flag`) and IDs correlated with sort order are the deadliest proxies; the single-feature AUC scan catches them in a minute.
- **If your data has a clock, your split must respect it** (`TimeSeriesSplit` plus an embargo at least as long as your longest look-back window), and every feature must look only backward.
- **Hash your rows and count cross-split duplicates;** if the same entity generates multiple rows, group by the entity with `GroupKFold` so it never straddles the split.
- **Bisect in cost order:** smell test → single-feature AUC → permutation importance → Pipeline-in-CV → OOF re-encode → time/dedup. Change one thing, watch the AUC, localize the leak before lunch.
- **Adversarial validation finds distributional leaks for free:** if a classifier can tell train from test (AUC ≫ 0.5), the features it uses are your leak suspects.
- **An honest CV score has higher fold variance than a contaminated one** and matches production; suspiciously tight folds are themselves a leak smell.
- **The honest number is the win, not the loss.** A 0.78 that production reproduces is worth infinitely more than a 0.97 that production never sees.

## 15. Further reading

- **scikit-learn documentation** — [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) and the [Pipeline / ColumnTransformer guide](https://scikit-learn.org/stable/modules/compose.html). The canonical reference for fitting transforms inside CV folds.
- **scikit-learn cross-validation** — the [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) and [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) references; choose the splitter that matches your data's structure.
- **`category_encoders`** — the [TargetEncoder, MEstimateEncoder, and LeaveOneOutEncoder](https://contrib.scikit-learn.org/category_encoders/) implementations; read the docs on cross-fitting before trusting any target encoder.
- **Micci-Barreca (2001), "A Preprocessing Scheme for High-Cardinality Categorical Attributes"** — the original empirical-Bayes (smoothed) target-encoding paper that the OOF encoder formalizes.
- **Kaufman, Rosset, Perlich (2012), "Leakage in Data Mining: Formulation, Detection, and Avoidance"** — the foundational academic treatment of leakage taxonomy and detection.
- **Kapoor & Narayanan (2023), "Leakage and the Reproducibility Crisis in ML-based Science"** — a survey documenting how widespread leakage has corrupted published results across scientific fields.
- **Within this series** — [A Taxonomy of Training and Finetuning Bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the master symptom→suspect→test→fix decision tree; [Data Leakage: The Silent Killer of Your Validation Score](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) for the modality-agnostic theory; [Cross-Validation Done Wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong) for group/time CV and tuning-set optimism; [Categorical and Feature Bugs](/blog/machine-learning/debugging-training/categorical-and-feature-bugs) for encoding and train/serve skew; [Your Metric Is Lying](/blog/machine-learning/debugging-training/your-metric-is-lying) for metric-choice traps; and the capstone [The Training Debugging Playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the full bisection workflow.

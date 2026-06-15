---
title: "Tabular Data, Done Right: From Raw Rows to Calibrated Predictions"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "An end-to-end, opinionated field guide to building tabular ML systems: leakage control, splitting, encoding, normalization, GBDT-first modeling, calibration, and six production case studies."
tags: ["tabular-data", "gradient-boosting", "xgboost", "lightgbm", "catboost", "feature-engineering", "data-leakage", "cross-validation", "calibration", "machine-learning"]
category: "machine-learning"
subcategory: "Traditional Machine Learning"
author: "Hiep Tran"
featured: true
readTime: 51
---

Most people who "do machine learning" for a living do it on tables. Not on images, not on token streams — on rows and columns pulled from a warehouse, with a target column somebody in the business cares about. And almost everyone, at some point, ships a model that scored 0.93 in the notebook and 0.71 in production, then spends a frantic week discovering that the 0.93 was a lie the whole time.

The uncomfortable truth is that on tabular data **the model is the easy part.** Gradient-boosted trees are a near-solved technology: you can `pip install lightgbm`, accept the defaults, and land within a few percent of the best score a Kaggle grandmaster will ever extract from the same table. The edge — the difference between a model that survives contact with production and one that quietly rots — lives almost entirely in the boring 80% nobody tweets about: how you split, what you leak, how you encode, whether your probabilities mean anything, and how you notice when the world shifts underneath you.

![The tabular ML pipeline as a serpentine flow gated by the train/validation split](/imgs/blogs/tabular-data-done-right-1.png)

The diagram above is the mental model for this entire article. Read it as a single directed flow: raw table, a profiling and leakage audit, then **the split** — the one blue box, drawn that way on purpose because everything downstream of it is gated by it. Every fitted transform after the split (imputation, encoding, scaling, feature engineering) must learn its parameters from training rows only; the validation fold has to stay a stranger. Out the other end come a GBDT, a calibration step, a cost-aware threshold, and a drift monitor that decides when to retrain. We are going to walk that pipeline left to right, and the recurring theme — the thing I want burned into your retina by the end — is that the split is a membrane, and leakage is anything that crosses it backward.

This is a deep-dive, so it is long and it is opinionated. The throughline: **be paranoid about leakage, trust your cross-validation more than the leaderboard, reach for gradient boosting by default, and never ship a probability you have not calibrated.** If you want the internals of the models themselves, this post is the practical parent to three siblings on this blog — [gradient-boosted trees](/blog/machine-learning/traditional-machine-learning/gradient-boosted-trees), the [LightGBM deep-dive](/blog/machine-learning/traditional-machine-learning/lightgbm-deep-dive), and [CatBoost from the inside out](/blog/machine-learning/traditional-machine-learning/catboost-from-inside-out-ordered-boosting-oblivious-trees) — and it assumes you will follow those links when you want to know *why* a tree splits the way it does.

## Why tabular is its own world

Before any pipeline talk, we have to confront a question that shapes every decision: why does the deep-learning juggernaut, which flattened computer vision and natural language, keep losing to a 2016-era gradient-boosting library on tables? It is not for lack of trying — there is a decade of papers proposing neural architectures for tabular data, and benchmark after benchmark (Grinsztajn et al. 2022, the most-cited of them) finds tree ensembles still on top for the median real dataset.

The answer is about **priors**. A convolutional network bakes in the assumption that nearby pixels are related and that a cat is a cat wherever it sits in the frame — translation invariance and spatial locality. A Transformer bakes in that tokens form a sequence where order and proximity carry meaning. Those are *enormous* free lunches, and they are exactly the assumptions that make those architectures sample-efficient. Tabular data offers none of them.

![Matrix comparing locality, homogeneity, pretraining, size and default model across images, text and tabular data](/imgs/blogs/tabular-data-done-right-2.png)

The matrix above lays out the mismatch column by column. Columns in a table have **no locality**: `age` sitting next to `zip_code` next to `account_balance` is an accident of schema design, and permuting the columns changes nothing about the problem. There is **no homogeneity**: one column is a float in dollars, the next is an unordered category with 40,000 levels, the next is an ordinal rating from 1 to 5. There is **no useful pretraining**: a CLIP embedding transfers across image tasks, but there is no "ImageNet of tables" because every table's columns mean something different. And the **datasets are small** — a few thousand to a few million rows is typical, where language models train on trillions of tokens. Strip away locality, homogeneity, and pretraining, and you have stripped away everything a CNN or Transformer uses to win. What is left is a function-approximation problem on heterogeneous, unordered, medium-sized data — which is precisely the niche axis-aligned tree ensembles were born for.

Here is the same idea as the assumption-versus-reality table I promised, because it reframes how you should approach the whole project:

| What beginners assume | The naive view | The production reality |
|---|---|---|
| "The model choice is what matters most" | Spend the week tuning XGBoost vs a neural net | The split and leakage controls move the score 10x more than the model |
| "More features always help" | Engineer 500 features, let the model sort it out | Each feature is a new leakage surface; the best teams ship fewer, audited features |
| "Accuracy is the metric" | Optimize accuracy, report it proudly | On imbalanced data accuracy is a vanity metric; you need PR-AUC, calibration, and a cost-weighted threshold |
| "A high CV score means a good model" | Trust the number the notebook prints | A high CV score with the wrong split means a *confident* wrong model |
| "Deep learning is the modern choice" | Default to a neural net because it is 2026 | GBDT is the default; deep-tabular wins only in narrow, nameable cases |

There are two further reasons the gap persists, subtler but just as decisive. The first is that **real tabular target functions are irregular** — they are not smooth. The boundary between "will default" and "will not" jumps at hard thresholds: a debt-to-income ratio crossing 43%, an age crossing 25 for insurance pricing, a balance hitting zero. Neural networks are biased toward smooth functions, so they must spend enormous capacity to approximate a step that a tree represents with a single split. The second is that tabular data is full of **uninformative columns** that carry no signal at all, and trees handle them gracefully by simply never splitting on them, while a dense network mixes every input into every neuron and has to *learn* to ignore the noise — which it does imperfectly, and never for free. Irregular targets and noisy features are the rule on real tables, not the exception, and both favor the axis-aligned, feature-selective nature of trees.

Internalize that table and you already think like a senior practitioner. The rest of this article is the mechanics that make each row true.

## Profile before you touch a model

The first thing to do with a new table is **not** to train anything. It is to understand the data well enough that you could explain every column to a skeptic. A senior rule of thumb: *if you cannot describe how a column is populated in the source system, you cannot trust it as a feature.* That sentence is doing leakage-prevention work, and we will see why in a moment.

A profiling pass answers four questions for every column: what is its type and cardinality, how often is it missing, what does its distribution look like, and — the load-bearing one — **could it possibly contain information from the future relative to prediction time?** That last question is the leakage audit, and it is the highest-leverage 20 minutes in the whole project.

```python
import pandas as pd
import numpy as np

def profile(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_unique": s.nunique(dropna=True),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "card_ratio": round(s.nunique(dropna=True) / len(s), 4),  # ~1.0 == ID-like
            # Correlation of presence/value with the target is a leakage smell.
            "target_corr": (
                round(s.fillna(s.median()).corr(df[target]), 3)
                if pd.api.types.is_numeric_dtype(s) else np.nan
            ),
        })
    out = pd.DataFrame(rows).sort_values("target_corr", key=abs, ascending=False)
    return out

report = profile(df, target="is_fraud")
# Anything with target_corr above ~0.5 on a hard problem is guilty until proven innocent.
# Anything with card_ratio near 1.0 is an identifier, not a feature.
print(report.head(20))
```

Two patterns in that report should make your stomach drop. A column with a near-perfect correlation to the target on a genuinely hard problem is almost never a brilliant feature — it is usually a leak: a value populated *after* the outcome is known. The classic is `account_status = "closed"` predicting churn, when accounts are marked closed *because* they churned. The second pattern is a `card_ratio` near 1.0, meaning the column is essentially a row identifier; feed it to a high-capacity encoder and it will memorize the training set.

The discipline here is **prediction-time thinking**: for each feature, ask "would I actually have this value, populated this way, at the moment I need to score a new row in production?" If the answer is no, or "only after the label is known," the feature is poison no matter how good it looks offline. No model can fix a feature that will not exist at inference; it can only be fooled by it.

## The fit boundary: the one habit that separates pros

Now the central concept. When you scale a column, you compute a mean and standard deviation. When you impute missing values, you compute a fill value. When you target-encode a category, you compute a per-category average of the label. Every one of those is a *learned parameter*, and the question that decides whether your offline metrics are real is: **learned from which rows?**

![Before-after figure contrasting fitting transforms on all data versus on the training fold only](/imgs/blogs/tabular-data-done-right-3.png)

The before-after above is the whole ballgame. On the left, the seductive mistake: you call `scaler.fit(X_all)` on the full dataset, target-encode using every row's label, impute with the global mean — and *then* split into train and validation. The validation fold's information has already bled into the transform parameters. Your cross-validation score comes back at 0.94, you celebrate, you ship, and production hands you 0.78 because in production there is no "full dataset" to peek at. On the right, the disciplined version: split first, `fit` every transform on the training fold alone, then merely `transform` the validation fold. The CV score drops to 0.86 — and that 0.86 is *honest*, which means it will roughly hold in production.

> The single most expensive bug in applied machine learning is not a wrong model. It is a right model evaluated against a contaminated split. The score lies, and a lying score is worse than no score, because it buys conviction.

This is why frameworks matter. Scikit-learn's `Pipeline` and `ColumnTransformer` exist precisely to make the fit boundary impossible to cross by accident: when you call `pipeline.fit(X_train, y_train)`, every step's `fit` sees only the training rows, and `cross_val_score` re-fits the whole pipeline inside each fold. Hand-rolled preprocessing in loose pandas cells is where leakage breeds.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier

num_cols = ["age", "balance", "tenure_days"]
cat_cols = ["plan", "region", "device"]

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
])

# The model is in the SAME pipeline, so every fold re-fits preprocessing on train only.
clf = Pipeline([("pre", pre), ("gbm", LGBMClassifier(n_estimators=400, learning_rate=0.05))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
print(f"AUC {scores.mean():.4f} +/- {scores.std():.4f}")  # honest, because fit-on-train is enforced
```

Notice that the `StandardScaler` is technically pointless for the LightGBM at the end — trees do not care about scale, as we will belabor shortly — but it costs nothing and the pattern is what matters: **the preprocessing and the estimator live in one object that gets re-fit per fold.** That is the structural defense against the most common, most costly mistake in the field.

### A field guide to the four kinds of leakage

"Leakage" is one word for several distinct failure modes, and naming them makes them findable. Every one of them is "information from outside the training fold reaching the model," but the channel differs:

| Kind | The channel | The tell | The fix |
|---|---|---|---|
| Target leakage | A feature is populated *after* or *because of* the label | One feature with implausibly high target correlation | Prediction-time audit; drop the feature |
| Train-test contamination | A transform was fit on data that includes the validation rows | CV score far above production | Fit-on-train; `Pipeline` + `cross_val_score` |
| Temporal leakage | A feature or split lets the model see the future | Random split scores >> temporal split | Time-ordered split with a purge gap |
| Group leakage | One entity's rows span both train and validation | Random split scores >> `GroupKFold` | Group the split on the entity id |

The reason this taxonomy is worth memorizing is that the *symptom* is always the same — an offline score that does not survive production — but the *cure* is specific to the channel. When a model collapses on deployment, you walk this table top to bottom: is there a too-good feature (target)? Was a transform fit globally (contamination)? Does a temporal split kill the score (temporal)? Does grouping kill it (group)? One of the four is nearly always the answer, and "the model was just unlucky" essentially never is. I have debugged perhaps thirty of these collapses across teams, and the count of times the root cause was *not* on this table is zero.

A practical corollary: build the **temporal split and the grouped split as deliberate diagnostics**, even when you do not think you need them. If your random-split CV is 0.95 and your grouped-split CV is 0.79, you have just discovered a group leak you did not know you had — for free, before production discovered it for you.

## Split first: your cross-validation scheme is the model

If the fit boundary is the habit, the **split scheme** is the decision that determines whether the habit even helps. A perfect fit-on-train discipline applied to the wrong kind of split still leaks — just more subtly. The reason is that the entire point of a validation fold is to simulate "data the model has never seen," and what counts as *unseen* depends on the dependence structure of your rows.

![Decision tree mapping dependence structure to the correct cross-validation scheme](/imgs/blogs/tabular-data-done-right-4.png)

The decision tree above is the one I want you to run in your head every single time, before you type `train_test_split`. Three questions, in order of priority:

- **Is the data time-ordered?** If you are predicting anything about the future — fraud next week, churn next month, demand next quarter — then a random split lets the model train on March and test on February, which is impossible in production and wildly optimistic offline. You need a **temporal split**: train on the past, validate on the future, and for serious work a *purged walk-forward* scheme where you also drop a gap of rows around the boundary so that engineered lag features cannot straddle it.
- **Are there repeated entities?** If the same user, device, store, or patient appears in many rows, a random split will scatter one entity's rows across both train and validation. The model then "predicts" a user it already memorized. You need **GroupKFold**, which guarantees an entity's rows live entirely in one fold.
- **Are the rows genuinely independent?** Only then is the textbook **Stratified K-Fold** (which preserves the class ratio in each fold) the right tool. The stratification matters most when the positive class is rare; without it, some folds can come out nearly target-free.

The red branch is the default everyone reaches for and the default that quietly destroys most first attempts: a **plain random split** applied to data that is actually grouped or time-ordered. It does not error. It does not warn. It just hands you a number that is too good, and the gap only reveals itself in production weeks later.

```python
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, StratifiedKFold

# Grouped: no user_id appears in both train and validation.
gkf = GroupKFold(n_splits=5)
for tr, va in gkf.split(X, y, groups=df["user_id"]):
    assert set(df.loc[tr, "user_id"]) & set(df.loc[va, "user_id"]) == set()

# Temporal: every validation index is strictly later than its training indices.
tss = TimeSeriesSplit(n_splits=5, gap=7)  # 7-row gap purges boundary leakage
for tr, va in tss.split(X_sorted_by_time):
    assert X_sorted_by_time.index[tr].max() < X_sorted_by_time.index[va].min()
```

### Second-order optimization: adversarial validation

A subtle failure mode is that your train and test distributions differ in a way no single split scheme captures — the test set is simply *not drawn from the same distribution* as your training data (a Kaggle private leaderboard, a future month, a new market). The trick to detect this is **adversarial validation**: label every training row 0 and every test row 1, throw away the real target, and train a classifier to tell train from test. If it can — AUC meaningfully above 0.5 — your sets are distinguishable, and the features driving that classifier are your distribution-shift culprits. You then build a validation split that mimics the test distribution (for example, validate on the training rows the adversarial model thinks look most like test), so your local CV finally tracks the leaderboard.

### Second-order optimization: how many folds, and why repeat them

The number of folds is a bias-variance trade for your *estimate of the score*, not for the model. Few folds (3) means each training set is smaller, so the estimate is biased pessimistically; many folds (10) means training sets are nearly the full data, lower bias, but the fold estimates are correlated and the whole thing costs more compute. Five is the sane default for most tables; go to 10 only when data is precious. The more important move when you need a *stable* number — for example, to decide whether feature A genuinely beats feature B by 0.001 AUC — is **repeated cross-validation**: run 5-fold CV with several different random seeds and average. A single 5-fold split has enough variance that a 0.002 difference between two models is often noise; repeating with five seeds shrinks the standard error enough to tell a real improvement from a lucky fold. The discipline that follows: **never accept a model change on a single fold's improvement.** If a feature does not survive repeated CV across seeds, it did not help; it got lucky, and luck does not deploy.

A second habit worth the compute is **nested cross-validation** when you both tune hyperparameters and report a score. If you tune on the same folds you report, the reported score is optimistic — you have fit the hyperparameters to those folds. Nested CV wraps an inner tuning loop inside an outer scoring loop, so the number you report was never used to make a choice. It is expensive, and for most production work a single held-out test set you touch exactly once is enough; but for a published benchmark or a high-stakes model comparison, nesting is the honest way.

## Cleaning without lying to your model

With the split decided, we can finally touch values. Cleaning is where well-meaning defaults do the most quiet damage, because the default for a missing number — fill it with the mean — is frequently a lie about your data.

![Before-after figure showing mean-fill collapsing a bimodal column into a fake central spike](/imgs/blogs/tabular-data-done-right-5.png)

The before-after above shows the trap. Suppose a column has two real cohorts — say, light users clustered around 20 and heavy users clustered around 80 — and a chunk of missing values. Fill those NaNs with the global mean of 50 and you have invented a spike of rows at a value *no real user ever had*, sitting right in the valley between your two genuine modes. You did not "handle" the missingness; you fabricated a third fake population and smeared away the very structure the model needed. The fix on the right is twofold: first, **add an `is_missing` indicator column** so the model can learn that missingness itself carries signal (and it very often does — a blank income field correlates with all sorts of things). Second, impute *within* a sensible group, or with a model, so the two real modes survive.

The deeper principle is that **how you impute must match why the value is missing.** Statisticians name three mechanisms, and they demand different treatment:

| Mechanism | What it means | Example | Safe move |
|---|---|---|---|
| MCAR (missing completely at random) | Missingness independent of everything | A sensor drops a reading due to a network blip | Simple impute (median) is fine; indicator optional |
| MAR (missing at random) | Missingness depends on *other observed* columns | High earners skip the income field, but you observe their zip and job | Impute conditional on the observed columns; add indicator |
| MNAR (missing not at random) | Missingness depends on the *missing value itself* | People with very low balances refuse to report balance | Indicator is mandatory; the *fact* of missingness is the feature |

The practical upshot: for tree models, the single most robust default is to **add a missingness indicator for any column missing more than a percent or two, and let the model decide.** Modern GBDTs go further — LightGBM and XGBoost treat NaN as a first-class value and learn an optimal default direction at each split, which often beats any imputation you could hand-craft. That is a real reason to prefer them: they refuse to let you lie by omission.

Outliers deserve the same "why" question. A balance of negative two billion is a data-entry error and should be clipped or nulled; a genuine whale customer with a balance 100x the median is real signal you must not winsorize away. Trees are naturally robust to extreme values (a split at the 99th percentile isolates the tail without distortion), which is yet another reason they dominate messy real-world tables.

## Encoding categoricals: the highest-variance decision you will make

Numbers a model can eat raw. Categories it cannot, and **how you turn categories into numbers is the decision with the widest spread of outcomes** in tabular ML — the difference between a clean win and a silent overfit that costs you the project.

![Matrix comparing categorical encodings across cardinality, leakage risk, tree-friendliness and information kept](/imgs/blogs/tabular-data-done-right-6.png)

The matrix above scores the five encodings you will actually use against the four properties that matter. Read it as a decision aid:

- **One-hot** is the safe, dumb default: one binary column per level. It leaks nothing and preserves all information, but it **explodes dimensionality** — a 40,000-level merchant ID becomes 40,000 sparse columns, which is fine for linear models and miserable for trees (each split can only ask about one level at a time). Cap it to high-frequency levels (`min_frequency`) and one-hot is perfectly respectable for low-cardinality columns.
- **Ordinal / label encoding** maps each level to an integer. Compact and tree-friendly, but it **fabricates an order** — telling the model that `region=3` is "between" `region=2` and `region=4` when the regions have no natural ordering. Trees tolerate this better than linear models (they can carve the integer axis into arbitrary buckets) but it is still lossy.
- **Target / mean encoding** replaces each level with the mean of the target for that level. Compact, powerful, tree-friendly, and it directly injects the signal you care about — but it is **a leakage bomb** if done naively, because a level's encoding is computed from the very labels you are trying to predict. We dedicate the next subsection to defusing it.
- **Hashing** maps levels into a fixed number of buckets via a hash function. Bounded dimensionality regardless of cardinality, no leakage, but **collisions** blur distinct levels together — acceptable when cardinality is enormous and you cannot afford to store a mapping.
- **Native categorical** handling is the winner for tree models and the reason CatBoost and LightGBM exist in their current form: you simply declare the column categorical and the library splits on subsets of levels directly, with built-in leakage protection (CatBoost's ordered target statistics, covered in [its deep-dive](/blog/machine-learning/traditional-machine-learning/catboost-from-inside-out-ordered-boosting-oblivious-trees)). No exploded dimensions, no fake order, no manual leak risk.

Two encodings did not earn a column but earn a mention. **Frequency encoding** replaces a level with how often it appears — `merchant_id` becomes "this merchant accounts for 0.4% of transactions." It carries no target information so it cannot leak, it is compact, and on many problems frequency is itself predictive (rare merchants are riskier, common ones safer). It pairs beautifully with native categorical handling as a cheap extra column. **Weight of evidence (WoE)**, borrowed from credit scoring, encodes a level as the log-odds of the target within that level; it is target encoding's regulated cousin, prized in lending because it produces monotonic, auditable, business-explainable features — and it carries exactly the same leakage risk, so it demands the same out-of-fold treatment.

The senior rule of thumb for the whole section: **for a GBDT, default to native categorical handling, add frequency encoding for free, and reach for out-of-fold target encoding only on the highest-cardinality columns where it measurably helps.** For a linear model, default to one-hot with a frequency floor, and use WoE when you need an auditable, monotonic feature. Everything else is a special case.

### Target encoding done safely

Because target encoding is so powerful and so dangerous, it earns its own figure. The naive version computes, for each category level, the average target over *all* training rows with that level — including the row you are about to encode. That row's own label has leaked into its own feature. On a high-cardinality column where many levels appear once or twice, the encoding becomes a near-copy of the label, training AUC rockets to 0.99, and validation collapses.

![Grid showing out-of-fold target encoding where each fold is encoded from the other folds only](/imgs/blogs/tabular-data-done-right-7.png)

The fix, shown above, is **out-of-fold (OOF) encoding**: partition the training data into folds, and to encode the rows in fold 1, use target means computed *only* from folds 2 and 3; to encode fold 2, use folds 1 and 3; and so on. Each row's encoding is now computed from data that excludes it, so a row never sees its own label. Add **smoothing** toward the global mean for rare levels (a level seen twice should not be trusted as much as a level seen 2,000 times), and you have an encoding that is both powerful and honest.

```python
import numpy as np
from sklearn.model_selection import KFold

def oof_target_encode(train_cat, y, test_cat, n_splits=5, smoothing=20.0):
    """Out-of-fold mean target encoding with Bayesian smoothing toward the prior."""
    prior = y.mean()
    oof = np.full(len(train_cat), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fit_idx, enc_idx in kf.split(train_cat):
        stats = (pd.DataFrame({"c": train_cat.iloc[fit_idx], "y": y.iloc[fit_idx]})
                 .groupby("c")["y"].agg(["mean", "count"]))
        # Shrink small-count levels toward the global prior.
        smooth = (stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)
        oof[enc_idx] = train_cat.iloc[enc_idx].map(smooth).fillna(prior).values
    # Test uses the full-train mapping (test labels were never involved, so no leak).
    full = (pd.DataFrame({"c": train_cat, "y": y}).groupby("c")["y"].agg(["mean", "count"]))
    smooth_full = (full["mean"] * full["count"] + prior * smoothing) / (full["count"] + smoothing)
    return oof, test_cat.map(smooth_full).fillna(prior).values
```

If you take one habit from this section: **never compute a statistic from the target without folding.** The moment a label touches a feature, ask "did this row's own label help build this row's feature?" If yes, you have a leak, and out-of-fold is how you break it.

## Numerical handling and normalization

Here is a section where the right answer depends entirely on your model family, and getting the dependency backward wastes effort or breaks training.

![Before-after figure showing a quantile transform reshaping a skewed feature while trees stay unaffected](/imgs/blogs/tabular-data-done-right-8.png)

The before-after above captures the rule. On the left, a raw skewed feature like income: a long right tail where a handful of rows are 50x the median. For **distance- and gradient-based models** — k-nearest neighbors, SVMs with RBF kernels, logistic regression, and neural networks — this is a problem. kNN's distances are dominated by whichever feature has the largest raw scale; gradient descent zig-zags down an ill-conditioned loss surface when features span wildly different ranges. A **quantile or log transform** (right side) reshapes the feature toward Gaussian, balances the distance metric, and stabilizes optimization. This is not optional for those models; it is the difference between convergence and a model that silently underperforms.

And for **trees** — the box at the bottom of both columns — none of it matters. A decision tree splits on `feature > threshold`, and any monotonic transform (log, standardize, quantile) leaves the *ordering* of values unchanged, so it leaves every possible split point unchanged. Standardizing features before a GBDT is harmless busywork. This is genuinely liberating: when your model is a gradient-boosted tree, you can skip the entire scaling debate and spend that time on features and leakage instead.

| Transform | What it does | Helps which models | Trees care? |
|---|---|---|---|
| StandardScaler | Subtract mean, divide by std | Linear, SVM, NN | No |
| MinMaxScaler | Squash to [0, 1] | NN (bounded activations), distance | No |
| RobustScaler | Center on median, scale by IQR | Anything with outliers | No |
| QuantileTransformer | Map to uniform or Gaussian by rank | Linear, NN, kNN on skewed data | No |
| Log / Box-Cox | Compress multiplicative tails | Linear, NN on heavy-tailed features | No (monotonic) |

The column on the right is the punchline of the whole table: *No, no, no, no, no.* If you are GBDT-first — and on tabular data you should be — normalization is a concern you get to mostly delete. Keep it in your `ColumnTransformer` anyway (it is free and makes the pipeline portable to a linear baseline), but do not agonize over it.

## Feature engineering: where the real signal hides, and the real leaks

Models are commodities; **features are where domain knowledge enters the system**, and on most real problems a handful of well-constructed features beats any amount of model tuning. But feature engineering is also the single richest source of leakage, because the most powerful features are aggregations, and aggregations love to reach across time boundaries they should not.

![Graph of feature engineering dataflow from raw columns into a feature matrix with a leakage node](/imgs/blogs/tabular-data-done-right-9.png)

The dataflow graph above shows how features fan out from raw columns and merge into the model matrix. From a transaction's `amount`, `timestamp`, and `user_id`, you derive a rolling 7-day mean of spend, time features like hour-of-day and is-weekend, and an expanding count of the user's prior transactions. Each is a legitimate, powerful feature — *as long as it only looks backward.* The red node is the trap: a "future-window aggregation," computed over a window that includes data from after the prediction point. The textbook version is computing a user's average transaction amount over the *entire* history, including transactions that occur after the one you are scoring. Offline, in a random split, that future information is sitting right there and the feature looks brilliant. In production, the future has not happened yet, and the feature is garbage.

The defense is a rule you enforce mechanically: **every aggregation is parameterized by a cutoff time, and may only touch rows strictly before it.** Concretely, that means expanding or rolling windows with proper time alignment, never a plain `groupby().mean()` over the full table when the rows are temporal.

```python
# WRONG: leaks the future — the mean includes transactions after each row.
df["user_mean_amt"] = df.groupby("user_id")["amount"].transform("mean")

# RIGHT: expanding mean of *prior* transactions only (shifted so the current row is excluded).
df = df.sort_values(["user_id", "timestamp"])
df["user_mean_amt_sofar"] = (
    df.groupby("user_id")["amount"]
      .apply(lambda s: s.shift(1).expanding().mean())   # shift(1) drops the current row
      .reset_index(level=0, drop=True)
)

# RIGHT: rolling 7-day window of prior spend, time-indexed, current row excluded.
df["spend_7d"] = (
    df.set_index("timestamp")
      .groupby("user_id")["amount"]
      .rolling("7D", closed="left")   # closed="left" excludes the current timestamp
      .sum()
      .reset_index(level=0, drop=True)
)
```

The categories of features that pay off most on tabular problems, in rough order of return on effort: **aggregations** over entities and time windows (counts, means, recencies); **interactions and ratios** that encode domain relationships a tree would need many splits to approximate (debt-to-income, price-per-square-foot); **date-time decompositions** (hour, day-of-week, days-since-event, cyclical sine/cosine encodings of periodic time); and **frequency features** (how common is this category level, which often correlates with risk). Every one of them must pass the prediction-time test from the profiling section: would this value, computed this way, exist at the moment of inference?

### Fewer, audited features beat a feature dump

There is a folk belief that you should engineer hundreds of features and "let the model sort it out." For trees, this is half-true and half-dangerous. True: a GBDT is robust to irrelevant features and will mostly ignore noise columns. Dangerous: every feature is a new leakage surface, a new thing that can drift in production, a new column your serving pipeline must compute correctly under latency, and a new opportunity for a high-cardinality artifact to memorize. The best teams I have worked with ship *fewer* features than the median Kaggle notebook, and they can explain every one.

When you do need to prune, prefer methods that respect the model. **Permutation importance** — shuffle one feature's values and measure how much validation score drops — is model-agnostic and honest, far more trustworthy than a tree's built-in "importance," which is biased toward high-cardinality columns. **Null importance** is the sharper tool: train many models on *shuffled* targets to build a null distribution of each feature's importance, then keep only the features whose real importance clears that noise floor. It is the single most effective automated feature-selection method I know for tabular data, precisely because it directly answers "is this feature doing better than random?"

```python
from sklearn.inspection import permutation_importance

# Honest importance: how much does validation AUC fall when each feature is shuffled?
r = permutation_importance(fitted_pipeline, X_valid, y_valid,
                           scoring="roc_auc", n_repeats=10, random_state=0)
for i in r.importances_mean.argsort()[::-1][:15]:
    print(f"{X_valid.columns[i]:<24} {r.importances_mean[i]:+.4f} +/- {r.importances_std[i]:.4f}")
# Features whose mean importance is <= 0 are not pulling weight — candidates to cut.
```

The payoff is not just a smaller model. A leaner, audited feature set drifts less, serves faster, and — when something does break at 2 a.m. — is something you can actually reason about. Feature *restraint* is a senior signal.

## Models: GBDT first, deep as contrast

We have arrived at the part everyone wanted to start with, and the reason I made you wait is that everything before this moves the needle more. With a clean split, no leaks, and good features, the model choice is a tuning of the last few percent — and for that last few percent, the default is not in question.

**Reach for a gradient-boosted decision tree first.** XGBoost, LightGBM, or CatBoost — pick by the considerations below — will, on the median tabular problem, match or beat anything else with a fraction of the engineering. That is not nostalgia; it is the consistent finding of every large-scale benchmark on real tabular data through 2025.

![Grid contrasting how a GBDT splits raw features against how a deep-tabular model embeds then attends](/imgs/blogs/tabular-data-done-right-10.png)

The two-column figure above is *why* the gap persists, and it is worth understanding rather than memorizing. A **GBDT** (left) consumes a row by splitting on raw features, one axis at a time — `amount > 3.5?` — and builds an additive ensemble where each new tree fits the residual errors of the ones before it. The inductive bias is exactly right for tables: it carves the heterogeneous feature space into axis-aligned boxes, handles mixed types natively, is invariant to feature scale, and is robust to outliers and irrelevant columns. A **deep-tabular model** (right) cannot split on a raw feature; it must first *embed* every feature into a vector, then run self-attention across those feature-tokens, then push the result through an MLP head. That is a lot of machinery, a lot of parameters, and a lot of data required, to relearn from scratch the axis-aligned structure a tree gets for free.

So when *does* deep-tabular win? The honest, narrow answer:

![Matrix recommending a model by row count, categorical handling, accuracy and use case](/imgs/blogs/tabular-data-done-right-11.png)

The model-selection matrix above is the cheat sheet I actually use. Walk the rows:

- **Logistic / linear regression** — always build it first as a baseline. It is interpretable, trains in milliseconds, and tells you whether the problem is even learnable. If your GBDT cannot beat a well-regularized logistic regression by a meaningful margin, something is wrong (or the problem is genuinely linear).
- **Random forest** — a strong, low-tuning baseline that is harder to overfit than boosting. A great sanity check, rarely the final model.
- **XGBoost** — the tuned default. Battle-tested, superb regularization, the model that wins when someone actually tunes it carefully.
- **LightGBM** — my first reach for anything large: histogram-based, leaf-wise growth, native categorical support, and *fast*. See the [LightGBM deep-dive](/blog/machine-learning/traditional-machine-learning/lightgbm-deep-dive) for why it is often 5–10x faster than XGBoost at equal accuracy.
- **CatBoost** — the right default when you have many high-cardinality categoricals, because its ordered boosting and ordered target statistics handle them with built-in leakage protection and minimal tuning.
- **MLP + embeddings** — worth trying when you have very large data (10^5 rows and up), strong categorical structure, and a GPU. Often used as a *diversity* member in an ensemble rather than a standalone winner.
- **FT-Transformer** — the strongest of the attention-based tabular architectures; reach for it on very large datasets or research settings where the last fraction of a percent justifies the cost.
- **TabPFN v2** — the genuinely exciting recent development: a transformer *pre-trained on millions of synthetic tabular tasks* that does in-context learning, producing SOTA results on **small** datasets (up to ~10,000 rows, ~100 features) with **no tuning and no gradient steps** — you just feed it train and test together and read off predictions. It is the one place a neural method clearly beats GBDT today, and the constraint (small data) is exactly the opposite of where deep learning usually wins.

```python
import lightgbm as lgb

# Declare categoricals natively — no one-hot, no manual encoding, leakage handled internally.
cat_features = ["plan", "region", "device", "merchant_id"]
for c in cat_features:
    df[c] = df[c].astype("category")

dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_features)
dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=63,            # the real capacity knob for leaf-wise growth
    min_child_samples=100,    # higher == more regularization on noisy tables
    feature_fraction=0.8,     # column subsampling per tree
    bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0,
)
model = lgb.train(
    params, dtrain, num_boost_round=5000,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],  # stop when valid AUC plateaus
)
print("best iteration:", model.best_iteration)
```

The hyperparameters that actually matter, in order: the **learning rate** (lower is better, paid for with more trees and `early_stopping` to find the count), the **leaf/depth capacity** (`num_leaves` for LightGBM, `max_depth` for XGBoost), and the **regularization** trio of minimum samples per leaf, subsampling of rows and columns, and L2. Everything else is a rounding error. Do not run a 200-trial hyperparameter search before you have audited your split and features — you will be precisely optimizing a leak.

### Tuning that is worth the compute

When you do tune, tune with a budget and a sane search method. Grid search is a waste — it spends most of its trials varying parameters that do not matter. Use **Bayesian optimization** (Optuna is the standard) with the CV pipeline as the objective, so every trial respects the fit boundary, and let early stopping pick the tree count *inside* each trial so you tune the learning rate and capacity, not the count.

```python
import optuna, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def objective(trial):
    params = dict(
        objective="binary", metric="auc", verbosity=-1,
        learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_child_samples=trial.suggest_int("min_child", 20, 300),
        feature_fraction=trial.suggest_float("ff", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bf", 0.5, 1.0), bagging_freq=1,
        lambda_l2=trial.suggest_float("l2", 1e-3, 10.0, log=True),
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = []
    for tr, va in cv.split(X, y):
        d = lgb.Dataset(X.iloc[tr], y.iloc[tr], categorical_feature=cat_features)
        m = lgb.train(params, d, num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va], reference=d)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        scores.append(roc_auc_score(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60, show_progress_bar=True)  # 60 trials beats a 1000-cell grid
print(study.best_value, study.best_params)
```

Sixty Bayesian trials will routinely beat a thousand-cell grid, and on a clean problem they will move your score by a percent or two — meaningful, but a fraction of what the split and features already bought you. That ordering is the whole point: **tuning is the last, smallest lever, not the first.**

### Stacking and blending: where the last fraction lives

On a competition, or any setting where the last 0.5% of a metric is worth real money, you reach for **ensembling across model families**. The reliable recipe is **stacking**: train several diverse base models (a LightGBM, an XGBoost, a CatBoost, a logistic regression, perhaps a neural net), generate their *out-of-fold* predictions on the training data, and train a simple meta-model — usually a logistic regression — on those OOF predictions as features. The out-of-fold discipline is mandatory and for the same reason as target encoding: if the meta-model trains on base predictions that saw their own rows, the stack leaks and the gain evaporates in production.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# Each base model's OOF probabilities become a feature for the meta-model.
oof = np.column_stack([
    cross_val_predict(m, X, y, cv=5, method="predict_proba")[:, 1]
    for m in (lgbm, xgbm, catb, logit)
])
meta = LogisticRegression(C=1.0).fit(oof, y)   # learns how to weight the base models
```

The gain from stacking is real but small and shrinking as base models improve — typically a fraction of a percent over the best single model. Diversity is what pays: two tuned GBDTs blend to almost nothing, but a GBDT blended with a calibrated linear model or a TabPFN often clears a useful margin, because they make *different* mistakes. The senior judgment call is whether that fraction of a percent is worth a five-model serving footprint. In a competition, always. In production, almost never — one well-built, monitored LightGBM you can debug beats a five-model stack you cannot.

### Interpretability with SHAP

A model you cannot explain is a model you cannot debug, defend to a regulator, or trust when it disagrees with a domain expert. For tree ensembles the mature answer is **SHAP** (SHapley Additive exPlanations): TreeSHAP computes, exactly and fast, how much each feature pushed each individual prediction away from the base rate, with a sound game-theoretic foundation rather than a tree's biased split counts.

```python
import shap
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_valid)
shap.summary_plot(sv, X_valid)            # global: which features matter, and which direction
shap.plots.waterfall(explainer(X_valid.iloc[0]))  # local: why THIS row scored what it did
```

Two uses pay off immediately. Global summary plots are a **leakage detector of last resort**: if one feature dominates every prediction by a wide margin, that is the same flare the profiling step raised, now confirmed by the trained model. Local waterfall plots are how you answer "why was this customer declined?" — a question that, in lending and insurance, is not optional but a legal requirement. The same explanations, watched over time, also surface drift: when a feature's SHAP distribution shifts, the model is reacting to a changing world before the metric has had time to fall.

## Imbalance and cost-sensitivity

Fraud is 0.2% of transactions. Churn is 3% of users. Disease is 1% of screenings. On these problems, a model that predicts "negative" every time is 99.8% accurate and completely worthless, and the way you handle the imbalance determines whether you build something useful or something that merely scores well.

The instinct is to **resample** — SMOTE to synthesize minority examples, or undersample the majority. My strong opinion, earned the hard way: **resampling is usually the wrong first move, and almost never the right last one.** Oversampling the minority distorts the base rate the model learns, which wrecks calibration (your probabilities now reflect a 50/50 world that does not exist); SMOTE interpolates synthetic points that can fall in regions no real example occupies, especially in high dimensions. What works better, in order:

| Technique | What it does | When it helps | The catch |
|---|---|---|---|
| Class weights | Up-weight minority errors in the loss (`scale_pos_weight`) | Almost always the right first lever | None major; tune it as a hyperparameter |
| Threshold tuning | Keep probabilities honest, move the decision threshold | Always — this is where imbalance is *actually* handled | Requires calibrated probabilities to be meaningful |
| Undersampling majority | Drop majority rows to balance | Huge data where training cost dominates | Throws away real information; hurts calibration |
| SMOTE / oversampling | Synthesize minority points | Rarely; small data, simple models | Distorts base rate, breaks calibration, fragile in high-D |

The reframing that makes imbalance tractable: **you do not have an imbalance problem, you have a threshold-and-cost problem.** Train a well-regularized model with class weights, get its probabilities *calibrated* (next section), and then choose the operating threshold by minimizing expected business cost — because a false negative (missed fraud) and a false positive (a blocked legitimate customer) almost never cost the same, and the default 0.5 threshold implicitly and wrongly assumes they do.

There is a temptation, when accuracy disappoints on a rare-positive problem, to reach for an exotic loss — focal loss borrowed from object detection, custom asymmetric objectives, elaborate resampling schedules. Resist it until the basics are exhausted. In practice, a gradient-boosted model with a tuned `scale_pos_weight`, honest calibration, and a cost-chosen threshold beats almost every clever loss function on tabular data, and it does so with a fraction of the moving parts. The reason is that the rare class is rare in *quantity*, not in *learnability* — the model can usually rank the positives perfectly well; what is broken is the decision rule layered on top, and a decision rule is fixed with a threshold, not with a new loss. The operating point itself — where on the precision-recall curve you choose to sit — is not really a modeling decision; it is a business conversation about how many false alarms the operations team can absorb and how much a miss truly costs. The data scientist's job is to hand the business an honest, calibrated curve and let them choose the point, not to pick one in a vacuum and present it as math.

## Evaluation is the product

I will say this as plainly as I can: **the metric you optimize and the threshold you ship are not "evaluation," they are the product.** A recommender that surfaces the wrong items and a fraud model that blocks the wrong cards are both perfectly good *rankers* that were turned into bad *decisions* by the last mile. That last mile is metric choice, calibration, and thresholding.

Start with metric choice, because optimizing the wrong one poisons everything upstream:

| Metric | What it measures | Use when | Fails when |
|---|---|---|---|
| Accuracy | Fraction correct | Balanced classes, equal costs | Imbalanced — a vanity number |
| ROC-AUC | Ranking quality across all thresholds | Comparing rankers, moderate imbalance | Optimistic under extreme imbalance |
| PR-AUC (average precision) | Precision-recall trade-off on the positive class | Extreme imbalance, you care about the rare class | Harder to compare across datasets with different base rates |
| Log loss | Probability quality (proper scoring rule) | You will use the probabilities for decisions | Sensitive to a few confident mistakes |
| Brier score | Calibration + refinement combined | You want a single calibration-aware number | Less intuitive than a reliability plot |

For a rare-positive problem, **PR-AUC over ROC-AUC**, every time: ROC-AUC can look excellent while your precision in the regime you actually operate is dismal, because the vast true-negative count flatters the false-positive rate. PR-AUC stares directly at the trade-off you care about.

The same discipline extends past binary classification. For **regression**, the choice between RMSE and MAE is a statement about how much you fear large errors: RMSE punishes them quadratically and is the right loss when a single big miss is catastrophic, while MAE treats all errors linearly and is more robust to outlier targets; when the target spans orders of magnitude (prices, counts), optimize the metric in log space or use a relative error so a \$10 miss on a \$20 item is not dwarfed by a \$10 miss on a \$2,000 one. For **multi-class**, decide early whether you care about every class equally (macro-averaged F1, which weights a rare class as much as a common one) or proportionally to frequency (micro-averaged, which a dominant class will dominate) — the two can disagree wildly on imbalanced label sets, and reporting the wrong one hides exactly the failure you should be watching. In every case the rule is the same: **pick the metric that mirrors the real-world cost of a mistake, then optimize that and nothing else.** A model is only ever as good as the metric you held it to.

Then the step almost everyone skips:

![Before-after figure showing calibration pulling an over-confident reliability curve onto the diagonal](/imgs/blogs/tabular-data-done-right-12.png)

The before-after above is calibration, and it is the most under-appreciated step in applied tabular ML. A gradient-boosted model can be an excellent *ranker* — high AUC, orders cases correctly — while being a terrible *probability estimator*. GBMs in particular tend to push scores toward 0 and 1, so a predicted "0.9" might correspond to a real-world frequency of 70%. The ranking (AUC 0.88) is untouched by this; what breaks is every decision that uses the probability as a probability — expected-value calculations, cost-weighted thresholds, downstream pricing. **Calibration** (Platt scaling, which fits a logistic on the scores; or isotonic regression, which fits a monotonic step function) leaves the ranking exactly where it was and pulls the reliability curve onto the diagonal, so that "0.9" finally means 90%.

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import numpy as np

# Calibrate on a held-out fold the base model never saw (prefit) — or use cv= for nested CV.
calibrated = CalibratedClassifierCV(base_estimator=model, method="isotonic", cv="prefit")
calibrated.fit(X_calib, y_calib)
p = calibrated.predict_proba(X_test)[:, 1]
print("Brier:", round(brier_score_loss(y_test, p), 4))  # lower == better calibrated

# Choose the threshold that minimizes expected business cost, not the default 0.5.
COST_FN, COST_FP = 500.0, 20.0   # a missed fraud costs $500; a false alarm costs $20
thresholds = np.linspace(0.001, 0.5, 500)
costs = [((p >= t).astype(int) == 0) @ (y_test * COST_FN)              # false negatives
         + ((p >= t).astype(int) == 1) @ ((1 - y_test) * COST_FP)      # false positives
         for t in thresholds]
t_star = thresholds[int(np.argmin(costs))]
print(f"cost-optimal threshold = {t_star:.3f}")  # often far below 0.5 on imbalanced problems
```

That `t_star` is the actual product decision. It is set by the cost asymmetry and the calibrated probabilities, and it is frequently nowhere near 0.5 — on a rare-positive, high-miss-cost problem it can be 0.02. Shipping the default 0.5 threshold on such a problem is shipping a model tuned for a world where false negatives and false positives cost the same, which is a world that does not exist.

## The reference pipeline

Pulling the threads together, here is the shape of a tabular pipeline I would actually defend in a review. It enforces the fit boundary structurally, uses native categoricals, calibrates, and selects a cost threshold — every habit from above, in one object.

```python
import lightgbm as lgb
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def build_dataset(df):
    cats = ["plan", "region", "device", "merchant_id"]
    for c in cats:
        df[c] = df[c].astype("category")
    # Backward-only aggregations (see feature-engineering section).
    df = df.sort_values(["user_id", "timestamp"])
    df["spend_7d"] = (df.set_index("timestamp").groupby("user_id")["amount"]
                        .rolling("7D", closed="left").sum().reset_index(level=0, drop=True))
    df["txn_count_sofar"] = df.groupby("user_id").cumcount()  # prior count, current excluded
    for c in cats:                                            # frequency features
        df[c + "_freq"] = df[c].map(df[c].value_counts(normalize=True))
    return df, cats

def evaluate(df, target="is_fraud"):
    df, cats = build_dataset(df)
    feats = [c for c in df.columns if c not in {target, "user_id", "timestamp"}]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(df))
    for tr, va in cv.split(df[feats], df[target], groups=df["user_id"]):
        # Carve a calibration slice out of train so calibration never sees validation.
        cut = int(0.8 * len(tr)); fit_idx, cal_idx = tr[:cut], tr[cut:]
        dtr = lgb.Dataset(df.iloc[fit_idx][feats], df.iloc[fit_idx][target], categorical_feature=cats)
        dva = lgb.Dataset(df.iloc[cal_idx][feats], df.iloc[cal_idx][target], reference=dtr)
        base = lgb.train(dict(objective="binary", metric="auc", learning_rate=0.03,
                              num_leaves=63, min_child_samples=100, feature_fraction=0.8,
                              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                              scale_pos_weight=(df.iloc[fit_idx][target].eq(0).sum()
                                                / max(df.iloc[fit_idx][target].sum(), 1))),
                         dtr, num_boost_round=5000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        # Isotonic calibration on the held-out calibration slice.
        from sklearn.isotonic import IsotonicRegression
        raw_cal = base.predict(df.iloc[cal_idx][feats])
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, df.iloc[cal_idx][target])
        oof[va] = iso.predict(base.predict(df.iloc[va][feats]))
    y = df[target].values
    print(f"OOF  AUC={roc_auc_score(y, oof):.4f}  PR-AUC={average_precision_score(y, oof):.4f}"
          f"  Brier={brier_score_loss(y, oof):.4f}")
    return oof
```

It is more code than a notebook one-liner, and that is the point: every extra line is a leak that cannot happen, a probability that means what it says, a fold that an entity cannot straddle. This is what "done right" costs, and it is cheap insurance against the 0.93-to-0.71 collapse.

## Train-serve skew: the gap the gates cannot see

A model that passed every gate in this article can still fail in production for a reason none of them checks: the features it receives at serving time are computed by *different code* than the features it trained on. The training pipeline runs in a notebook over a warehouse snapshot; the serving pipeline runs in a request handler over a live database, under latency, frequently in a different language. Any divergence between the two — a different default fill value, a rounding rule, a timezone, a category spelled `"USA"` in one and `"US"` in the other, an aggregation window that is 7 days in training and "since midnight" in serving — is **train-serve skew**, and it is uniquely nasty because nothing errors. The model simply receives slightly wrong inputs and returns slightly wrong outputs, forever, and your offline metrics never see it.

The structural defenses, in increasing order of investment:

- **Share the transform code.** The single most effective move is to make training and serving call the *same* feature-computation function, not two implementations that are "supposed to match." A serialized scikit-learn `Pipeline` or a shared library imported by both paths eliminates an entire class of skew by construction.
- **Log features at serving time, replay them offline.** Persist the exact feature vector the model scored in production, then periodically re-score those logged vectors with the training-time pipeline and assert the predictions match. Any drift between them is skew, caught in a dashboard instead of in a quarterly loss review.
- **Adopt a feature store** when the same features feed many models, because it makes "compute this feature one way" an enforced contract rather than a hope. The store computes features once and serves the identical values to training and inference.
- **Pin the data contract.** Most skew originates upstream, where a team you do not control changes how a column is populated — exactly the silent schema change from the drift case study. A schema contract (column types, allowed ranges, allowed category sets, null rates) validated on every batch turns those silent changes into loud, early failures.

```python
# Log the exact scored vector so production features can be replayed offline.
def score_and_log(request_features: dict) -> float:
    x = pd.DataFrame([request_features])[FEATURE_ORDER]   # fixed column order, always
    p = float(calibrated_model.predict_proba(x)[:, 1])
    feature_log.append({"ts": now(), "features": request_features, "p": p})
    return p

# Nightly: re-score logged vectors with the training pipeline; any mismatch is skew.
replay = pd.DataFrame([r["features"] for r in feature_log])[FEATURE_ORDER]
assert np.allclose(calibrated_model.predict_proba(replay)[:, 1],
                   [r["p"] for r in feature_log], atol=1e-6), "train-serve skew detected"
```

The reproducibility habit that underpins all of this: **pin everything that touches the model** — the library versions, the random seeds, the exact training snapshot, the feature code commit. A model you cannot rebuild bit-for-bit is a model you cannot debug when it drifts, because you cannot tell whether the world changed or your pipeline did. Reproducibility is not bureaucracy; it is the difference between a one-hour incident and a one-week mystery.

## Case studies from production

Patterns are easier to remember as scars. Here are eight, drawn from the kinds of failures that recur across fraud, churn, credit, and competition work. Names and numbers are illustrative, but every mechanism is real and common.

### 1. The credit-risk model that was 99% accurate and useless

A team building a loan-default model reported 99.1% accuracy and wanted to ship. The default rate was 0.9%. The model had learned to predict "no default" for everyone — 99.1% accurate, zero defaults caught, zero business value. The wrong first hypothesis was "the model needs more features." The actual fix was a complete reframing: switch the metric to PR-AUC, train with `scale_pos_weight`, **calibrate the probabilities**, and choose a threshold by expected loss — a caught default saved roughly \$8,000 in principal, a wrongly-declined applicant cost roughly \$200 in lifetime margin, a 40:1 asymmetry. The cost-optimal threshold landed near 0.04, not 0.5. Same model, same features; the difference between worthless and profitable was entirely in evaluation and thresholding. The lesson: **on imbalanced problems, accuracy is a vanity metric and the threshold is the product.**

### 2. The churn model that leaked through the account table

A churn model hit 0.95 AUC in cross-validation and 0.74 in the first production month. Two leaks compounded. First, the team used a plain `KFold`, but the same customer appeared in multiple monthly snapshots, so a customer's August row trained the model that "predicted" their September row — a textbook group leak, fixed by `GroupKFold` on `customer_id`. Second, a feature called `account_status` included the value `"cancelled"`, which is set *when a customer churns* — the label in disguise, found only by the prediction-time audit ("would we know account_status='cancelled' before they churned? No — that *is* churning"). After grouping the split and dropping the leak, honest CV came back at 0.79, which is what production then delivered. The lesson: **a CV-to-production gap is almost always a split or a leak, never bad luck.**

### 3. The high-cardinality target-encoding blowup

An engineer target-encoded a `merchant_id` column with 200,000 levels — naive mean encoding over all training rows. Training AUC: 0.992. Validation AUC: 0.71. Most merchants appeared once or twice, so the encoding was effectively `mean of one label` — a near-perfect copy of the target, memorized. The fix was out-of-fold encoding with smoothing toward the global mean (so a merchant seen once contributes almost nothing beyond the prior), which dropped training AUC to 0.84 and *raised* validation AUC to 0.82. The lesson: **whenever a feature is built from the target, fold it and smooth it, or it will memorize.** The gap between train and validation AUC is your leakage detector — a 0.28 gap is a flare, not a model that needs more regularization.

### 4. The Kaggle competition where the leaderboard lied (in the good direction)

In a tabular competition, a competitor's local 5-fold CV said 0.881 but the public leaderboard said 0.847 — a persistent, stable gap. The instinct to "trust the leaderboard" and tune toward it would have been a mistake; the public leaderboard was 3% of the test data, high variance, and drawn from a slightly later time period. Adversarial validation confirmed train and test were distinguishable on a few drifting features. The competitor built a time-aware validation split that reproduced the 0.847 locally, tuned against *that*, and finished well above people who had overfit the public board. The lesson: **trust a well-constructed CV over a small leaderboard, and when they disagree, find out why before you chase the number.** A leaderboard is one more validation fold, and a tiny one.

### 5. The unscaled features that broke the linear baseline

Before reaching for boosting, a team built a logistic-regression baseline and got nonsense coefficients and an AUC barely above 0.5 — and concluded "the problem is not learnable, ship a constant." The real issue: they fed raw features, where `income` ranged to 10^6 and `n_products` to 10, so the L2 penalty effectively zeroed every small-scale coefficient and the optimizer crawled. A `StandardScaler` in front of the logistic regression took the baseline from 0.52 to 0.86 in one line. The same unscaled features fed to LightGBM scored 0.87 with or without scaling — because trees do not care. The lesson: **scaling is model-dependent; a broken linear baseline is often a scaling bug, not an unlearnable problem.** Always build the baseline, and build it correctly, before you conclude anything.

### 6. The fraud model that decayed silently in production

A fraud model deployed at 0.86 AUC and nobody looked at it again. Six weeks later, fraud analysts noticed more slipping through, but no alert had fired — the model's own confidence looked normal, because a miscalibrated model is confidently wrong.

![Timeline of silent model drift from deploy through a schema change to a retrain trigger](/imgs/blogs/tabular-data-done-right-13.png)

The timeline above is the autopsy. At week 0 the model deployed at 0.86. At week 3, an upstream team changed how a `device_type` field was populated — a silent schema change, no error, no notification. By week 6 the feature distribution had shifted enough to register a Population Stability Index of 0.18 against the training distribution, but nobody was computing PSI. By week 9 effective AUC had bled to 0.79 and fraud losses finally tripped a business alarm. The fix that should have existed: a **drift monitor** computing PSI and KS statistics on every feature nightly, with a retrain trigger at PSI > 0.2.

```python
import numpy as np

def psi(expected, actual, bins=10):
    """Population Stability Index: how far has a feature's distribution moved?"""
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected) + 1e-6
    a = np.histogram(actual,   cuts)[0] / len(actual)   + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))   # <0.1 stable, 0.1-0.2 watch, >0.2 retrain
```

The lesson, and the reason the pipeline figure ends in a monitor-and-retrain loop: **a tabular model is a perishable asset.** It is trained on a snapshot of a world that keeps moving, and without drift monitoring you will not learn it has gone stale until the business does — which is the most expensive possible way to find out.

### 7. The timestamp parsed as a string

A demand-forecasting model was inexplicably weak, and feature importance showed the `order_date` column doing almost nothing — strange, for a forecasting problem where time should be everything. The bug was upstream: `order_date` had been loaded as an *object* dtype (a raw string), so the model treated `"2026-01-15"` as a high-cardinality categorical with tens of thousands of unique levels, one per date. It could not extract month, day-of-week, or trend from a string; every date was just an opaque token. Parsing it to a real datetime and decomposing it into day-of-week, month, day-of-year, is-holiday, and a numeric days-since-epoch trend feature moved the model from barely-better-than-mean to genuinely useful in a single afternoon. The lesson: **check your dtypes before you check your model.** A datetime read as a string, a numeric id read as an integer feature, a category read as a number — silent type errors produce silently weak models, and no amount of tuning fixes a feature the model literally cannot read. A `df.dtypes` audit belongs in the same first-twenty-minutes pass as the leakage check.

### 8. The customer id that was secretly the label

A propensity model scored a suspicious 0.98 AUC on a random split and held up across folds, so the team nearly shipped it. The profiling audit caught it: `customer_id` had a 0.7 correlation with the target. The reason was procedural — the data had been *exported sorted by signup cohort*, and the most recent cohort (high ids) happened to contain almost all the positive labels because of a recent marketing push. The model had learned "high id, predict positive," which was perfectly true in the training snapshot and completely useless for scoring genuinely new customers, whose ids would be higher still and whose positive rate would be entirely different. Dropping the raw id and replacing it with legitimate behavioral features brought CV to an honest 0.81. The lesson: **a row identifier is never a feature, and any column with a near-monotonic relationship to collection order is a trap.** Ids encode *when* and *how* a row was collected, not anything causal about the entity — and "when it was collected" is exactly the kind of future-correlated signal that evaporates on the next batch.

## When to reach for GBDT, when for deep-tabular

Reach for **gradient-boosted trees** when:

- You have heterogeneous columns (mixed numeric, categorical, ordinal) — which is almost always.
- Your data is small-to-large but not enormous (thousands to tens of millions of rows).
- You need a strong result fast, with little tuning and no GPU.
- You have many categoricals — LightGBM or CatBoost with native handling.
- Interpretability matters — SHAP on a GBDT is mature, fast, and trusted.

Reach for **deep-tabular** only when one of these is genuinely true:

- **Small data, no tuning budget:** TabPFN v2 on a dataset under ~10,000 rows and ~100 features, where its in-context learning beats a tuned GBDT with zero gradient steps.
- **Very large data with strong categorical/sequential structure:** an MLP-with-embeddings or FT-Transformer when you have 10^6+ rows, a GPU, and the categorical embeddings can learn relationships worth the cost — frequently as an ensemble member that *diversifies* a GBDT rather than replacing it.
- **Multimodal tables:** rows that mix tabular columns with free text or images, where a neural model can fuse modalities and a tree cannot.

Skip deep-tabular when you are reaching for it because it feels modern, when your data is medium-sized and tabular-classic, when you have no GPU or no tuning time, or when you need calibrated probabilities and interpretability tomorrow. The boring answer is usually the correct one: **audit your split, kill your leaks, engineer a few honest features, train a LightGBM, calibrate it, threshold it by cost, and monitor it for drift.** Do those six things well and you will beat almost everyone who spent the same week tuning a neural network on a contaminated split.

## Further reading

- [Gradient-boosted trees](/blog/machine-learning/traditional-machine-learning/gradient-boosted-trees) — the mechanics of boosting residuals, the engine under every recommendation here.
- [LightGBM deep-dive](/blog/machine-learning/traditional-machine-learning/lightgbm-deep-dive) — histogram binning, leaf-wise growth, and why it is the fast default.
- [CatBoost from the inside out](/blog/machine-learning/traditional-machine-learning/catboost-from-inside-out-ordered-boosting-oblivious-trees) — ordered boosting and ordered target statistics, the principled fix for categorical leakage.
- [Tabular data understanding with LLMs: a survey](/blog/paper-reading/large-language-model/tabular-data-understanding-with-llms-a-survey-of-recent-advances-and-challenges) — where large language models do and do not fit into the tabular stack.
- Grinsztajn, Oyallon, Varoquaux (2022), *Why do tree-based models still outperform deep learning on tabular data?* — the benchmark that anchors the GBDT-first stance.
- Hollmann et al. (2025), *TabPFN v2* — the in-context tabular foundation model that rewrote the small-data corner of this map.

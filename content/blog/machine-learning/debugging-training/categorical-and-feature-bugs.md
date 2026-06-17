---
title: "Categorical and Feature Bugs: Encoders, Unseen Categories, and Train-Serve Skew"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why a tabular model that scored 0.86 offline crawls at 0.79 in production — a field guide to the feature-engineering bugs that never crash: label-encoding a nominal column, unseen categories at inference, train-serve skew, sentinel NaNs, and scaling that hurts trees, each with the math behind it and a runnable detector."
tags:
  [
    "debugging",
    "model-training",
    "tabular",
    "feature-engineering",
    "scikit-learn",
    "data-leakage",
    "machine-learning",
    "finetuning",
    "deep-learning",
    "pandas",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/categorical-and-feature-bugs-1.png"
---

Here is a run that looked finished and was not. A team builds a credit-risk model on a clean, well-understood tabular dataset — a few hundred thousand rows, thirty-odd columns, the usual mix of numbers and categories. Cross-validated accuracy comes back at **0.86**, AUC at **0.91**. The notebook is tidy, the feature list is sensible, nobody touched the labels. They wrap it in a service and ship. In the first week of production the model's real accuracy is **0.79** and falling, and the on-call engineer is staring at a stack trace that says `ValueError: y contains previously unseen labels: 'Hai Phong'`. Nothing about the model artifact changed. The training code is byte-identical to what ran in the notebook. And yet the offline number was a fiction, and the service is throwing exceptions on perfectly ordinary inputs.

The post-mortem finds not one bug but a small pile of them, and every single one is invisible at training time. The `city` column was passed through a `LabelEncoder`, which mapped each city to an integer — `Hanoi -> 0`, `Da Nang -> 1`, `Ho Chi Minh City -> 2` — and the linear part of the stacked model read those integers as a *quantity*, learning that "city 2 is greater than city 0" as if cities lay on a number line. A `-999` sentinel in the `age` column, meant to flag "missing," was fed to the model as a literal age of negative nine hundred and ninety-nine. The `income` column arrived as strings like `"1,200,000"` with thousands separators and was silently treated as a categorical with a quarter-million unique levels. And the encoder that mapped cities to integers had been fit on the training cities only, so the first time a customer from a city not in the training set hit the service, the transform crashed. The model did not have one disease. It had a feature pipeline that was lying in six different dialects at once.

This is the defining quality of feature bugs: **they do not crash at training time, they do not warn, and the metric you trust agrees the model is fine.** The pipeline runs end to end, the loss goes down, the validation number prints in green. The damage is done quietly — a spurious order a tree happily exploits but a linear model misreads, a feature computed one way in the notebook and a different way in the service, a category the encoder has never seen. In the six-places framework this series uses to localize any training bug — data, model code, optimization, numerics, systems, evaluation — feature bugs live almost entirely in the **data** place, but they have a nasty habit of being detected only in **evaluation**, and only when evaluation is honest. Figure 1 lays out the seven distinct mechanisms we will work through, because "feature bug" is not one bug; it is a family, and you debug each member differently.

![A vertical stack of the seven distinct categorical and feature bug classes from encoding bugs at the top through unseen categories, train-serve skew, NaN and sentinel handling, scaling that hurts trees, high-cardinality blowup, and dtype parsing at the bottom](/imgs/blogs/categorical-and-feature-bugs-1.png)

By the end of this post you will be able to take a tabular model that looks great offline and decide, in under an hour, whether its features are honest. You will know *why* label-encoding a nominal column injects a fake ordinal axis (and why trees can exploit it while linear models are wrecked by it); *why* feature scaling is mathematically irrelevant to a decision tree's splits but critical to any gradient-based or distance-based model; and *why* high cardinality trades variance for overfitting in a way you can quantify. You will have runnable detectors — a `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")`, an unseen-category audit on a holdout, a train-serve feature diff for a single entity, and a dtype-and-range audit — and you will end where every honest feature story ends: with the model that crawled at 0.79 in production back up to the 0.86 it always should have been, once the features stopped lying.

This post is one branch of the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the deepest part of the "data" branch — and it is a sibling of the [capstone debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook). If your symptom is "great offline, broken in production" or "crashes on a normal-looking input," this is your page.

## 1. The smell test for feature bugs: the offline-online gap

Start with the cheapest diagnostic, the one that costs zero compute: **does the offline number survive contact with a row the model has never seen?** Feature bugs have a single, reliable fingerprint — a gap between the offline metric and the online metric that opens up *even though the model weights never changed*. A leak inflates the offline number; a feature-pipeline bug usually does something subtler and meaner: it makes the offline number *honest for the offline data* and *wrong for production*, because the feature the model trained on is not the feature production computes.

That is worth saying slowly, because it separates feature bugs from leakage. A leak (covered in the sibling post [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage)) means information about the answer leaked into the features, so the offline number is *too good*. A feature bug means the *same feature name* carries *different values* in training and serving, or carries a value the model cannot interpret, so the offline number is fine but it measured a model that does not exist in production. Both produce an offline-online gap; the direction and the fix differ.

The professional habit is to treat the offline-online gap as a *measurement* of how much your feature pipeline lies, and to localize it before touching the model. There are exactly two ways the gap can arise from features:

- **The model is being asked to interpret a value it cannot.** A `-999` age, a string where a number belongs, a category integer read as a magnitude. The model produces a confident, wrong answer.
- **The model is being shown a different value than it trained on.** The bucket boundary moved, the default changed, the rounding differs, the timezone shifted by a day. The feature *name* is the same; the feature *content* is not.

Both are detectable offline if you look, and that is the entire game: you do not need production traffic to find a feature bug. You need a holdout that *behaves like production* (unseen categories, missing values, the full value range) and a discipline of comparing the train-time feature vector to the serve-time feature vector for the same entity. We will build both.

The reason this matters so much for tabular work specifically is that tabular pipelines have *more surface area for silent corruption* than image or text pipelines. A vision model takes a tensor of pixels — if the channels are swapped or the normalization is wrong, the values are still all floats in a plausible range, and the failure is usually visible the moment you look at a batch. A tabular pipeline takes thirty heterogeneous columns, each with its own type, units, missing-value convention, and category set, assembled from joins across several source tables, each maintained by a different team. Every join is a chance to duplicate a row, every type coercion is a chance to turn a number into a string, every category column is a chance to meet an unseen value, and every one of those failures keeps the pipeline running. The defining discipline of tabular debugging is therefore *auditing each column against what it is supposed to be* — its type, its range, its category set, its missing rate — because no single instrument downstream will tell you a column went wrong. The model will happily consume garbage and report a number.

#### Worked example: the gap that math says is a feature bug, not a model bug

Suppose your offline cross-validated accuracy is 0.86 and your first week of production accuracy is 0.79, a drop of 7 points. You retrain on more data — no change. You tune hyperparameters — no change. You swap the model from gradient-boosted trees to a random forest — the gap persists at 6–8 points. That invariance is the tell. If the gap were a *model* problem (overfitting, wrong capacity, bad regularization) it would move when you change the model. A gap that survives every model change while the *features* stay the same is, almost by definition, a feature problem: the model is faithfully learning a relationship that does not hold in production because the production features are not the training features. The fix is not in the model. It is in the pipeline. The rest of this post is about finding which of the seven mechanisms is responsible.

The smell test tells you to look; it does not tell you where. The next seven sections each take one mechanism, derive *why* it bites, give you the runnable detector, and show the before→after.

## 2. Encoding bugs: the fake order a LabelEncoder injects

The most common categorical bug is also the most seductively easy mistake: you have a column of strings, scikit-learn's `LabelEncoder` turns strings into integers, the integers feed the model, done. The problem is what those integers *mean* to the model. `LabelEncoder` assigns codes alphabetically or in order of appearance — `Da Nang -> 0`, `Hanoi -> 1`, `Hai Phong -> 2`, `Ho Chi Minh City -> 3` — and those codes are not labels in any geometric sense. But the model does not know that. To the model, column `city` is now a number, and a number has an order, a distance, a midpoint.

### Why the order is spurious, and why it matters differently per model

Here is the science. A *nominal* variable has categories with no inherent order: there is no sense in which Hanoi is "less than" Ho Chi Minh City. An *ordinal* variable does have an order: `small < medium < large`, `cold < warm < hot`. Label-encoding is correct for ordinal variables *if the codes respect the order*, and wrong for nominal variables *always*, because it manufactures an order that is not there.

What does a model do with that manufactured order? It depends entirely on the model family, and this is the crux:

- **A linear model** (logistic regression, linear SVM, the linear head of a neural net) computes a weighted sum $z = w \cdot x + b$. For the city feature it learns a single scalar weight $w_{\text{city}}$, and the contribution to the score is $w_{\text{city}} \cdot \text{code}$. This forces the relationship between city and target to be *monotone and linear in the code*: if $w_{\text{city}} > 0$, the model believes risk rises smoothly from Da Nang (0) to Ho Chi Minh City (3). But the true relationship between an arbitrary city ordering and risk is *not* monotone — it is a lookup table, not a line. The linear model literally cannot represent "Hanoi high, Da Nang low, Hai Phong high again" with a single weight on an integer code. It is being asked to fit a step function with a straight line, and it will fit it badly. This is why label-encoding a nominal column can cost a linear model 10–15 points of accuracy.

- **A tree model** (decision tree, random forest, gradient boosting) splits on thresholds: `city_code <= 1.5`. With enough splits a tree *can* carve the integer axis into the right buckets — `<= 0.5` isolates Da Nang, `> 2.5` isolates Ho Chi Minh City — so a deep tree can partially recover the lookup table. But it pays for it: every distinction between two non-adjacent codes requires multiple splits, the tree wastes depth on an axis that has no real geometry, and the splits it learns are an artifact of the *arbitrary code assignment*, not the data. Change the encoder's alphabetical order and the tree's splits change. Worse, the tree can "exploit" the fake order in a way that *helps on training data and hurts on new data*: it finds a threshold that happens to separate classes given this particular code assignment, which is pure overfitting to the encoding. So trees are not immune — they are *less catastrophically broken* than linear models, which is a different and dangerous thing, because the bug hides better.

The asymmetry is the whole lesson. **Label-encoding a nominal feature wrecks linear and distance-based models loudly and degrades tree models quietly.** If you label-encode and your linear baseline is terrible but your gradient-boosting model is "fine," you have not avoided the bug — you have hidden it inside a model robust enough to paper over it, which means it is still costing you accuracy you cannot see. Figure 2 shows the before→after for a linear model on a real-shaped problem.

![A two-column before and after figure showing label-encoding a nominal city column gives a linear model 0.71 accuracy because of a fake order, while one-hot encoding the same column lifts it to 0.86 accuracy with no order imposed](/imgs/blogs/categorical-and-feature-bugs-2.png)

### The fix: one-hot for nominal, ordinal-encode only true orders

The correct encoding for a nominal feature is **one-hot**: turn the single integer column into one binary column per category, so `city` becomes `city_Hanoi`, `city_DaNang`, `city_HCMC`, each 0 or 1. Now the linear model learns a *separate weight per city* — exactly the lookup table the data wants — with no order imposed. For a true ordinal feature, use `OrdinalEncoder` and *give it the order explicitly* via the `categories` argument, so `small -> 0, medium -> 1, large -> 2` respects the real ranking. And reserve `LabelEncoder` for what its name says: encoding the *target label* `y`, never an input feature. Here is the contrast in code, with the bug and the fix side by side:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.DataFrame({
    "city": ["Hanoi", "Da Nang", "HCMC", "Hai Phong", "Hanoi", "HCMC"] * 200,
    "size": ["small", "large", "medium", "small", "large", "medium"] * 200,
    "y":    [0, 1, 1, 0, 0, 1] * 200,
})

# BUG: LabelEncoder on a NOMINAL feature injects a fake order the linear model misreads.
le = LabelEncoder()
X_bug = np.c_[le.fit_transform(df["city"]),  # city as an ordered integer -> WRONG
              le.fit_transform(df["size"])]
acc_bug = cross_val_score(LogisticRegression(max_iter=500), X_bug, df["y"], cv=5).mean()

# FIX: one-hot the NOMINAL feature, ordinal-encode the TRUE ordinal feature with its real order.
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
city_oh = ohe.fit_transform(df[["city"]])
size_ord = OrdinalEncoder(categories=[["small", "medium", "large"]]).fit_transform(df[["size"]])
X_fix = np.c_[city_oh, size_ord]
acc_fix = cross_val_score(LogisticRegression(max_iter=500), X_fix, df["y"], cv=5).mean()

print(f"label-encoded (buggy):  {acc_bug:.3f}")
print(f"one-hot + ordinal (ok): {acc_fix:.3f}")
```

On a problem where the city→target relationship is genuinely non-monotone, the label-encoded linear model lands around chance-plus-a-little while the one-hot model recovers the real signal. The exact numbers depend on the data, but the *direction* is reliable and the *mechanism* is the point: the fake order is a representational lie, and one-hot removes it.

### Quantifying the damage: how much can a fake order cost?

It is tempting to treat the fake-order bug as a small inefficiency, but it can be large, and the size is predictable from the structure of the data. The damage scales with how *non-monotone* the true category→target relationship is under the arbitrary code assignment. Picture three regimes:

- **The codes happen to align with the target.** If, by luck, the alphabetical order of the categories also ranks them by target rate (rare, but possible for a 2- or 3-category column), the single linear weight fits the relationship and the bug costs almost nothing. This is the trap that makes label-encoding *seem* fine on small toy columns and lets the habit survive.
- **The codes are uncorrelated with the target.** For a nominal column where the code order is random with respect to risk, the best single linear weight on the integer code captures roughly the *average* slope, which for a balanced non-monotone relationship is near zero. The linear model effectively learns nothing from that feature — it has been handed the information and cannot read it. A feature that should contribute several points of accuracy contributes nearly zero.
- **The codes are anti-aligned in a structured way.** The worst case: a non-monotone relationship where low and high codes share a target value and the middle codes are opposite (a "valley" or "hill" in code space). A single linear weight cannot represent a valley, so the model is actively misled on part of the range. This is where you see the full 10–15 point hit on a feature that, one-hot encoded, would have been one of your strongest.

The general principle is exact: a linear model on a label-encoded categorical can only represent target relationships that are *monotone in the arbitrary code*, and almost no real categorical satisfies that. One-hot encoding gives the model a free parameter per category, so it can represent *any* mapping from category to target contribution — which is what a nominal feature actually requires. The cost of one-hot is the extra parameters (one weight per category instead of one for the whole column), which is only a problem when cardinality is high — and that is exactly the case section 7 handles with target encoding.

There is a third option worth naming because it appears the moment your model is a neural net: **learned embeddings.** Instead of one-hot (one binary column per category) or a single integer, you map each category to a small dense vector — say 8 or 16 dimensions — and *learn* those vectors during training. An embedding is strictly more expressive than one-hot for a downstream nonlinear model, it scales gracefully to high cardinality (the parameter count is `cardinality × embedding_dim`, not `cardinality` extra columns multiplied by every interaction), and it places similar categories near each other in the learned space. For tabular neural nets this is the standard treatment of categoricals, and the same unseen-category discipline applies: reserve an embedding row for "unknown" and route any category not seen in training to it, exactly as `handle_unknown="ignore"` reserves the all-zeros vector for one-hot.

#### Worked example: the linear baseline that "couldn't beat" the tree

A pricing team builds two models on the same 24-feature dataset: a logistic regression and a gradient-boosted tree. The tree scores 0.83 accuracy, the logistic regression 0.72, and the team concludes "this problem needs a nonlinear model" and ships the tree. The real story is in the preprocessing: a shared script `LabelEncoder`s all eight categorical columns, including a 60-level `region` column that is purely nominal. The tree routes around the fake order with extra splits (paying some depth but recovering most of the signal); the logistic regression, forced to read 60 regions as a single ordered axis, cannot. When the team re-runs the logistic regression with the categoricals one-hot encoded, it jumps from 0.72 to 0.81 — within two points of the tree, on a fraction of the inference cost. The lesson is not "always use linear"; it is that **an 11-point gap between a linear and a tree model on identical features is a feature-encoding signature, not evidence about model capacity.** The team was about to draw a permanent architectural conclusion from a one-line preprocessing bug.

## 3. Unseen categories: the crash and the silent default

The credit model's production crash — `ValueError: y contains previously unseen labels: 'Hai Phong'` — is the second mechanism, and it is the one most likely to take down a service. The setup is universal: you `fit` an encoder on the training categories, it learns a fixed mapping `{Hanoi: 0, Da Nang: 1, ...}`, you pickle it, you deploy it. Then a real request arrives with a category that was not in the training set — a new city, a new product SKU, a typo, a rare value that happened to fall entirely in the test split — and the encoder has no code for it. What happens next depends entirely on the encoder and its configuration, and the defaults are bad.

### The three things an unseen category can do

There are exactly three behaviors, and you must know which one your encoder has:

1. **Crash.** `LabelEncoder` and `OrdinalEncoder` (with default settings) raise a `ValueError` on an unseen category. In a batch job this fails the job; in a service it returns a 500 to the user. This is the *loud* failure, and ironically the safest, because at least you know.
2. **Map to a wrong/default code silently.** If you configure `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, the unseen category becomes `-1`. Now the model receives a code it never saw during training and produces a confident, meaningless prediction. This is the *silent* failure, and it is worse than the crash because nobody notices.
3. **Map to "all zeros" — the correct default.** `OneHotEncoder(handle_unknown="ignore")` represents an unseen category as the all-zeros vector across its one-hot columns, which is exactly "none of the known categories." The model treats it as a blank, which is the most defensible thing it can do with information it has never seen.

The discipline is **fit on train, transform on serve, and configure for the unknown explicitly.** Never re-fit an encoder at serving time (that would make the codes drift from what the model trained on — a guaranteed train-serve skew, which is the next section). Instead, freeze the encoder learned on the training split and make a deliberate choice about unseen values.

Choosing the right unknown behavior is itself a small decision with consequences. The all-zeros vector from `handle_unknown="ignore"` tells the model "none of the known categories," which is honest but throws away the fact that *this is a new category* — and newness can be predictive (a brand-new merchant is riskier; a brand-new product has no history). If you want the model to *learn* a "this is unknown" effect, you have two better options than plain ignore: reserve an explicit `"__unknown__"` category at training time and route rare/held-out categories to it (so the model trains on the unknown bucket and learns its risk), or, for trees, use `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` so unknowns land in their own splittable bucket. The wrong choice is the silent one: an `unknown_value` that collides with a real code (mapping unknowns to `0` when `0` is a legitimate category) hands the model a wrong, *confident* code instead of a flag. The rule of thumb: make "unknown" either a clearly out-of-range value the model can isolate, or an explicit category it can learn — never a real code in disguise. Figure 3 draws the fit-on-train / transform-on-serve discipline and the re-fit trap it prevents.

![A dataflow graph showing the training split fit-transforming an encoder into a frozen pickled artifact that the serving row transforms against, versus the buggy branch where re-fitting on the serving row causes the codes to drift apart](/imgs/blogs/categorical-and-feature-bugs-3.png)

### The detector: count unseen categories in a holdout before you ship

You do not need production to discover you will crash in production. Hold out a slice of data that *behaves like the future* — for time-ordered data, the most recent slice; for grouped data, held-out groups — and count, per categorical column, how many categories appear in the holdout that were absent from the training set. That count is a direct estimate of your unseen-category rate at serving time. Here is the audit:

```python
import pandas as pd

def unseen_category_audit(train_df, holdout_df, cat_cols):
    """For each categorical column, count categories in holdout absent from train."""
    rows = []
    for col in cat_cols:
        train_cats = set(train_df[col].dropna().unique())
        holdout_cats = set(holdout_df[col].dropna().unique())
        unseen = holdout_cats - train_cats
        n_unseen_rows = holdout_df[col].isin(unseen).sum()
        rows.append({
            "column": col,
            "train_cardinality": len(train_cats),
            "holdout_cardinality": len(holdout_cats),
            "n_unseen_categories": len(unseen),
            "n_unseen_rows": int(n_unseen_rows),
            "pct_unseen_rows": round(100 * n_unseen_rows / max(len(holdout_df), 1), 2),
            "examples": sorted(list(unseen))[:5],
        })
    return pd.DataFrame(rows).sort_values("pct_unseen_rows", ascending=False)

# audit = unseen_category_audit(train_df, holdout_df, ["city", "product_sku", "device"])
# print(audit)
```

A column with `pct_unseen_rows` of 3.1% is telling you that roughly one in thirty production requests will hit a category your encoder has never seen. If your encoder crashes on unknowns, that is a 3.1% error rate that is *purely a configuration bug*, not a modeling limitation. Fix it by switching to `handle_unknown="ignore"` (one-hot) or `handle_unknown="use_encoded_value"` with a deliberate `unknown_value` (ordinal), and re-run the audit to confirm the crash rate drops to zero.

### The math of how often you will meet a new category

The unseen-category rate is not random noise you can wish away; for many real columns it is governed by a heavy-tailed category distribution and is *predictable*. Category frequencies in the wild — cities, products, user agents, merchant names — typically follow a power law (Zipf-like): a few categories cover most of the mass, and a long tail of rare categories each appears a handful of times. Two consequences follow directly. First, **the unseen rate never goes to zero by adding training data**, because the tail keeps producing genuinely new categories: doubling your training set adds the next slice of the tail but the tail is infinite, so there is always a fresh-category rate at serving time. Second, **the unseen rate is higher than your intuition**, because the categories you *remember* are the frequent head, while the rare tail — each member individually negligible — sums to a non-trivial fraction of requests. A column that "feels" complete after a million training rows can still send 2–4% of production requests through a never-before-seen category.

This is why `handle_unknown` is not an edge-case nicety but a *load-bearing* configuration: for any real categorical, the unknown branch *will* be exercised, on a known and non-trivial fraction of traffic, forever. The only questions are whether it crashes, predicts garbage, or degrades gracefully — and you choose which when you configure the encoder, not when production picks it for you at 3 a.m. The audit above turns "feels complete" into a measured rate, and the measured rate is the SLA your unknown-handling has to meet.

#### Worked example: a 3% crash rate becomes 0 with one keyword

A recommendation service encodes `product_sku` (cardinality ~8,000 in train) with `OrdinalEncoder()` at its defaults. The holdout audit shows 2.8% of rows carry a SKU not present in training — new products launch every week. In production, 2.8% of requests throw `ValueError` and fall back to a non-personalized default, which the business measures as a 2.8% degradation in click-through on exactly the freshest, most interesting inventory. The fix is one argument: `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, paired with a model (gradient boosting) that can treat `-1` as its own bucket. The crash rate goes to 0%, the new SKUs get a sensible cold-start prediction, and the click-through recovers. The bug was never in the model. It was a default the team never questioned.

## 4. Train-serve feature skew: the same entity, two vectors

This is the most expensive feature bug and the hardest to see, because it is invisible in any single environment. Offline, your training and validation features are computed by the same code (the notebook), so they agree, and the offline metric is honest. Online, the features are computed by a *different* code path (the service), and if that path computes anything differently — a different default, a different rounding, a different bucket boundary, a feature read at a different time — then the model is being scored on inputs it never trained on. The model is fine. The training is fine. The serving features are skewed, and the model's behavior on them is undefined in the worst way: it produces confident predictions on a distribution it never saw.

### Why two code paths drift

The science here is organizational, not numerical, but it is rigorous: any time the *same feature* is computed by *two implementations*, the implementations will eventually disagree, because they are maintained separately and tested separately. The classic divergences:

- **Time of computation.** The offline pipeline computes `days_since_signup` as of the label date; the online pipeline computes it as of *now*. For a churned user the two can differ by months. Same feature name, different value.
- **Default and missing handling.** Offline, a missing income is imputed with the training median (say 45,000). Online, a missing income arrives as `null` and the service substitutes 0, or `-1`, or the *current* median. The model trained on 45,000-for-missing and now sees 0-for-missing.
- **Rounding and units.** Offline, a price is in dollars; online, it arrives in cents. Offline, a timestamp is parsed as UTC; online, as local time. Offline, a ratio is computed in float64; online, in float32 with a different rounding at the seventh digit. Small per-feature, large per-prediction when they compound.
- **Bucket boundaries.** Offline you bucket age into `[0, 25, 40, 60, 120]` using the training quantiles; online someone hard-codes slightly different cutpoints. A 41-year-old lands in bucket 2 offline and bucket 1 online.

Every one of these produces a model that "works" everywhere it is tested in isolation and fails only at the seam. Figure 6 shows the definitive test and its fix.

![A two-column before and after figure showing the same entity producing a high bucket offline and a low bucket online with production accuracy 0.79, versus a single shared transform that makes the offline and online vectors byte-equal and recovers production accuracy to 0.83](/imgs/blogs/categorical-and-feature-bugs-6.png)

### The definitive diagnostic: diff the feature vectors for one entity

There is exactly one test that settles train-serve skew, and it is the most important diagnostic in this entire post: **take one real entity, compute its feature vector offline and online, and diff them element by element.** If they differ, you have your bug, and the diff tells you *which feature* and *by how much*. This is the train-serve analog of overfit-a-single-batch — make it fail small, on one example, where you can read every number.

```python
import numpy as np
import pandas as pd

def feature_skew_diff(entity_id, offline_fn, online_fn, feature_names, atol=1e-6):
    """Compute the SAME entity's features via the offline and online paths; diff them."""
    off = np.asarray(offline_fn(entity_id), dtype=float)
    on  = np.asarray(online_fn(entity_id), dtype=float)
    if off.shape != on.shape:
        raise AssertionError(f"shape skew: offline {off.shape} vs online {on.shape}")
    abs_diff = np.abs(off - on)
    mism = abs_diff > atol
    report = pd.DataFrame({
        "feature": feature_names,
        "offline": off,
        "online": on,
        "abs_diff": abs_diff,
        "skewed": mism,
    })
    n_skew = int(mism.sum())
    print(f"entity {entity_id}: {n_skew}/{len(feature_names)} features skewed")
    return report.sort_values("abs_diff", ascending=False)

# report = feature_skew_diff(
#     entity_id="user_91823",
#     offline_fn=compute_features_offline,   # the notebook / training path
#     online_fn=compute_features_online,     # the service path
#     feature_names=FEATURE_NAMES,
# )
# print(report[report.skewed])
```

Run this on ten entities and the pattern jumps out: if `days_since_signup` is the top row of every diff, that is your skew. The fix is structural and non-negotiable: **share one implementation between training and serving.** The cleanest version is a single transform object — a fitted `ColumnTransformer` or a feature-store transformation — that both the offline training job and the online service call. If the *same code* computes the feature in both places, the two paths cannot drift, because there is only one path. When that is genuinely impossible (different languages, different latency budgets), the diff above becomes a *continuous test*: a CI job that logs a random sample of production entities, recomputes their features offline, and alerts if any feature's distribution drifts beyond a threshold.

### Why "log and join" beats "recompute" for point-in-time features

There is a deeper reason train-serve skew is so common for *time-dependent* features, and it is worth making rigorous because the naive fix makes it worse. A feature like `days_since_signup`, `count_purchases_last_30d`, or `current_account_balance` depends on *when* it is computed. At serving time the value is whatever it is *at the moment of the request*. At training time, to be honest, the feature must be computed *as of the moment the historical prediction would have been made* — its point-in-time value — not as of today. If your offline pipeline computes `count_purchases_last_30d` by looking at the last 30 days *from now* while building a training set of events from six months ago, every training label is paired with a feature from the wrong window. The feature has *future information* relative to the label (a leak) and *different information* than serving will have (skew) — both bugs at once.

This is why the industry-standard fix for time-dependent features is **log-and-join, not recompute**: at serving time, log the exact feature vector the model actually consumed, keyed by entity and timestamp; later, join those logged vectors to the realized labels to build the training set. Because the training feature *is literally the serving feature*, byte for byte, skew is impossible and point-in-time correctness is automatic. Recomputing features offline "to match production" is a trap — you are re-implementing the serving logic and re-introducing the very drift you are trying to kill. The feature-store pattern (Feast, Tecton, and the in-house equivalents) exists precisely to make log-and-join and point-in-time joins the default. Here is the lightweight, no-feature-store version: a serving-time logger plus an offline consistency check.

```python
import json, time
import numpy as np
import pandas as pd

# --- at serving time: log the exact vector the model consumed ---
def serve_and_log(entity_id, feature_vector, prediction, log_path="serve_log.jsonl"):
    record = {
        "entity_id": entity_id,
        "ts": time.time(),
        "features": [float(x) for x in feature_vector],  # exactly what predict() saw
        "prediction": float(prediction),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

# --- offline: sample logged rows, recompute via the OFFLINE path, alert on drift ---
def skew_monitor(log_path, offline_fn, feature_names, sample=500, atol=1e-6):
    logged = [json.loads(l) for l in open(log_path)][-sample:]
    n_rows_skewed, per_feature = 0, np.zeros(len(feature_names))
    for rec in logged:
        on  = np.asarray(rec["features"], dtype=float)
        off = np.asarray(offline_fn(rec["entity_id"]), dtype=float)
        mism = np.abs(off - on) > atol
        n_rows_skewed += int(mism.any())
        per_feature += mism.astype(float)
    rate = 100 * n_rows_skewed / max(len(logged), 1)
    top = pd.Series(per_feature, index=feature_names).sort_values(ascending=False)
    print(f"{rate:.1f}% of {len(logged)} logged rows show skew; worst features:")
    print(top.head())
    return rate, top
```

If `skew_monitor` reports 0.0% you have proven, on real traffic, that your offline and online paths agree. If it reports 4.0% with `days_since_signup` at the top of the list, you have localized the bug to one feature and one code path — the same conclusion as the single-entity diff, now continuous and quantified. This monitor is cheap insurance: it turns a class of bug that normally surfaces as a slow, unexplained production decline into a CI alert that names the feature.

## 5. NaN, inf, and the sentinel-as-number trap

The fourth mechanism is missing-value handling, and it has three sub-bugs that each turn a "missing" into a confident wrong number.

### Sub-bug A: the sentinel read as a magnitude

Legacy datasets encode "missing" as a sentinel value — `-999`, `-1`, `0`, `9999`, `NA` as a literal string. If that sentinel survives into the model as a number, the model reads it as a real measurement. A `-999` age is not "missing" to a linear model; it is an age of negative nine hundred ninety-nine, which dominates the weighted sum and poisons the prediction for every row that has it. A tree is a little safer — it can split off `age <= -500` into its own branch — but only if there are enough sentinel rows to justify a split, and the sentinel still distorts any feature interaction. The fix is to convert sentinels to true `NaN` *before* imputation:

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = df.replace({"age": {-999: np.nan, 0: np.nan},          # 0 was "unknown", not a real age
                 "income": {-1: np.nan}})
# Now NaN is honestly NaN, and the imputer can do its job.
```

### Sub-bug B: the imputer fit on the wrong data (leakage)

The imputer learns a fill value — the median, the mean, the most-frequent category. If you fit it on the *whole* dataset before splitting, the fill value for the training rows is contaminated by the validation/test rows, which is the same fit-before-split leakage covered in [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage). The fix is the same: fit the imputer inside the cross-validation fold, on training rows only, via a `Pipeline`. We build exactly that pipeline in section 9.

### Sub-bug C: the dropped missing-indicator and the inf from a ratio

Two finishing touches. First, *missingness is often informative* — the fact that income is missing may itself predict the target — so dropping the missing-indicator throws away signal. `SimpleImputer(add_indicator=True)` keeps a binary column flagging which rows were imputed, recovering that signal. Second, ratio features (`debt / income`, `clicks / impressions`) produce `inf` when the denominator is zero, and `inf` propagates into NaN gradients for any gradient-based model and breaks many tree implementations' split statistics. Guard ratios explicitly:

```python
import numpy as np

# inf-safe ratio: zero denominator -> NaN (then imputed), not inf
df["debt_to_income"] = np.where(df["income"] > 0, df["debt"] / df["income"], np.nan)
# catch any survivors before they reach the model
assert np.isfinite(df.select_dtypes("number").to_numpy()).all(), "inf/NaN leaked into features"
```

That final assert is the kind of guardrail this series preaches: a one-line check that catches the bug at the boundary, at step 1, instead of at step 4000 when the loss goes NaN and you have no idea why. (For the full NaN/inf hunt across deep nets, see the dedicated post on [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs); here the cause is almost always a sentinel or a zero-denominator ratio, both catchable at feature-build time.)

#### Worked example: the imputation that quietly halved a feature's signal

A lending model has an `employment_length` feature that is missing for 18% of applicants — and missingness is *informative*, because self-employed and gig applicants disproportionately leave it blank, and they default at a higher rate. The first pipeline imputes the median (5 years) and drops the missing-indicator. The result: the 18% of high-risk blanks are now indistinguishable from solid 5-year-tenured employees, and the model's recall on defaulters drops by several points because it can no longer separate "5 years" from "we don't know, and not-knowing is itself a red flag." The fix is two arguments: `SimpleImputer(strategy="median", add_indicator=True)`. The indicator column resurrects the missingness signal — the model learns a separate effect for "imputed" rows — and defaulter recall recovers. The mistake is not the imputation; medians are fine. The mistake was *throwing away the fact that the value was missing*, which for this feature was one of the strongest signals in the dataset. The general rule: when missingness might correlate with the target, keep the indicator; the cost is one binary column and the benefit can be points of recall.

A subtler version of the same bug is the imputer fit on the wrong split. If you call `SimpleImputer().fit_transform(X)` on the full dataset and *then* split, the median that fills your training rows was computed partly from your validation and test rows — a small but real leak that makes the offline number optimistic. On a dataset where the feature distribution is stable the effect is tiny; on a dataset where the test split is from a different time period (so its median differs), the leak can be worth a point or two of inflated validation accuracy that evaporates in production. The fix, again, is the `Pipeline`: fit the imputer inside the fold so it only ever sees training rows.

## 6. Scaling that hurts: why trees do not care and linear models do

The fifth mechanism is the inverse of a bug — it is *applying a transform that helps one model family and hurts another*. Standardization (subtract the mean, divide by the standard deviation) is reflexively applied to every numeric feature by many practitioners. For some models it is essential; for others it is irrelevant at best and harmful at worst. Knowing which is which is pure mechanism.

### The science: scale-invariance of axis-aligned splits

A decision tree splits on a threshold: `income <= 50000`. Now apply any *monotone* transform to `income` — standardize it, log it, multiply by a million. A monotone transform preserves order: if $a \le b$ then $f(a) \le f(b)$. The split `income <= 50000` becomes `income_scaled <= f(50000)`, the *same partition of the rows*, just with a different threshold value. The tree's chosen split, the resulting groups, the information gain — all identical. Formally, for any monotone $f$, the set of achievable partitions of the data by axis-aligned thresholds on $x$ equals the set achievable on $f(x)$, so the optimal tree is invariant. **Scaling a feature for a tree cannot change a single split.** It is computationally wasted work, and it actively hurts in two ways: it makes feature importances and split thresholds unreadable (your threshold is now `-0.34` instead of an interpretable `50000`), and if the scaler is fit on the full dataset it *introduces* leakage where there was none.

Contrast a gradient-based or distance-based model:

- **Gradient descent** on a linear/logistic model or a neural net moves all weights with one learning rate. If feature A ranges over $[0, 1]$ and feature B over $[0, 10^6]$, the loss surface is a long thin valley: the gradient is enormous along B and tiny along A, so a learning rate that is stable for B crawls for A, and a rate that moves A explodes B. Standardization makes the valley round, so one learning rate works for all features — this is why scaling is *critical* for gradient-based models and is the same conditioning argument behind the [learning-rate](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) and normalization posts.
- **Distance-based** models (k-NN, k-means, RBF-kernel SVM) compute $\|x_i - x_j\|$. An unscaled feature with a huge range dominates the distance, so the model effectively ignores every small-range feature. Scaling equalizes the contributions.

The rule is a one-liner: **scale for gradient-based and distance-based models; do not scale for trees.** And whichever you choose, fit the scaler on training data only, inside the pipeline. The table below makes the model-by-transform decision explicit:

#### Worked example: the unscaled feature the linear model ignored

A churn model has two strong features: `tenure_months` (range 0–120) and `monthly_charges` (range 20–120) — comparable scales — plus `total_charges` (range 0–9,000, because it is tenure times monthly). Trained with logistic regression and *no scaling*, the model's coefficient on `total_charges` is tiny relative to the others, and ablating `total_charges` barely changes accuracy: the model is effectively ignoring it. The reason is not that `total_charges` is uninformative; it is that gradient descent on the unscaled features sees a loss surface stretched 75× along the `total_charges` axis, so the optimizer makes almost no progress on that weight in the iterations allotted. After `StandardScaler`, all three features have unit variance, the optimizer moves every weight at a comparable rate, and `total_charges` earns a meaningful coefficient — accuracy rises from 0.79 to 0.82. Now run the *control*: feed the *same* unscaled features to a `HistGradientBoostingClassifier`. Accuracy is identical with and without the scaler, to three decimal places, because — as the scale-invariance argument proves — the tree's splits do not move under a monotone rescaling. Same data, same bug-or-not, opposite verdict depending on model family. That is the entire point of this section in one experiment.

The table below makes the model-by-transform decision explicit:

| Model family | Scale numerics? | One-hot nominal? | Why |
| --- | --- | --- | --- |
| Linear / logistic | Yes (standardize) | Yes | Weighted sum needs conditioning; one weight per category |
| Neural net (MLP) | Yes (standardize) | Yes (or embeddings) | Gradient conditioning; no spurious order |
| k-NN / k-means / RBF-SVM | Yes (essential) | Yes | Distance dominated by large-range features |
| Decision tree / RF / GBM | No (irrelevant, can hurt) | Optional | Splits are scale-invariant; trees handle integer codes |
| Naive Bayes (Gaussian) | Helps | Use categorical NB | Assumes per-feature Gaussian |

## 7. High-cardinality blowup and the target-encoding-leak tradeoff

The sixth mechanism is what happens when one-hot meets a column with thousands of categories. One-hot encoding a `zip_code` column with 40,000 distinct values creates 40,000 binary columns, almost all zero. Three things break: memory (a dense 40,000-wide matrix is enormous; even sparse it strains linear models), the curse of dimensionality (most columns are nonzero for a handful of rows, so the model has almost no data per parameter and overfits), and tree performance (a tree must consider 40,000 binary splits, slow and prone to isolating single categories).

### The science: variance versus the leakage of target encoding

The standard alternative is **target encoding** (a.k.a. mean encoding): replace each category with a statistic of the target for that category — for binary classification, the mean target rate. So `zip_code` becomes a single numeric column where each zip is its historical positive rate. This collapses 40,000 columns to 1 and gives trees and linear models a smooth, informative feature. It also has two failure modes you must respect:

- **The high-variance / rare-category problem.** A zip code seen 3 times in training has a target mean estimated from 3 samples — pure noise. Naively encoding it with that noisy mean overfits. The fix is *smoothing* (shrink the per-category mean toward the global mean, weighted by the category count): the smoothed encoding for a category with $n$ samples, category mean $\bar{y}_c$, global mean $\bar{y}$, and smoothing strength $m$ is $\hat{y}_c = \frac{n \bar{y}_c + m \bar{y}}{n + m}$. A category seen once leans almost entirely on the global mean; a category seen ten thousand times is essentially its own mean. This is a bias-variance knob: large $m$ is more bias, less variance.

- **The leakage problem (the dangerous one).** If you compute each category's target mean using *the row you are about to encode*, you have leaked the label into the feature — the encoding for row $i$ partly *is* row $i$'s answer. On training data this looks magical (the feature is almost perfectly predictive); in production it is worthless, and the offline-online gap is enormous. This is the same fit-before-split leakage as everything else, and the fix is the same discipline: compute the target encoding **inside cross-validation**, using only out-of-fold rows to estimate each category's mean, never the row being encoded. Libraries like `category_encoders` provide `TargetEncoder` with built-in cross-fitting; scikit-learn 1.3+ ships `sklearn.preprocessing.TargetEncoder` that does the cross-fitting for you.

Here is the cross-fitted, smoothed target encoder written out, so the two safeguards are concrete rather than abstract. The key is that the encoding for each fold's rows is computed from the *other* folds' targets, and within each computation the per-category mean is shrunk toward the global mean by the smoothing strength `m`:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def cross_fit_target_encode(series, y, n_splits=5, m=20.0, seed=0):
    """Leak-safe smoothed target encoding: each row encoded from OTHER folds only."""
    series, y = series.reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    global_mean = y.mean()
    encoded = pd.Series(np.full(len(series), np.nan), index=series.index)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in kf.split(series):
        # category statistics from the FIT folds only
        stats = y.iloc[fit_idx].groupby(series.iloc[fit_idx]).agg(["sum", "count"])
        # smoothed mean: (n*cat_mean + m*global) / (n + m)
        smooth = (stats["sum"] + m * global_mean) / (stats["count"] + m)
        encoded.iloc[enc_idx] = series.iloc[enc_idx].map(smooth).fillna(global_mean)
    return encoded, global_mean  # global_mean is the fallback for unseen at serve time
```

Two properties make this safe. The row being encoded never contributes to its own category statistic, so the encoding carries no label information about that row — the leak is gone. And a category seen `n` times is pulled toward the global mean with weight `m / (n + m)`, so a once-seen category leans ~95% on the global mean (at `m = 20`) while a 10,000-times category is essentially its own mean — the high-variance tail is tamed. At serve time you fit one final encoder on the full training set (still smoothed) and map unseen categories to `global_mean`, the most defensible value for "a category I have no statistics on." This is the structural twin of `handle_unknown="ignore"`: an explicit, sensible answer for the unknown.

So high-cardinality features force a real tradeoff: one-hot is leak-proof but blows up dimensionality; target encoding is compact and powerful but leaks unless you cross-fit and overfits unless you smooth. Figure 4 lays out the encoder-by-model-by-failure matrix so the choice is mechanical, not guessed.

![A matrix figure with rows for LabelEncoder, OneHotEncoder, OrdinalEncoder, and target encoding, and columns for what each is best for and its characteristic failure mode such as fake order, column blowup, or leakage if fit on all data](/imgs/blogs/categorical-and-feature-bugs-4.png)

And figure 5 turns the choice into a decision tree you can run in your head: ordinal or nominal first, then low or high cardinality.

![A decision tree starting from a categorical feature, branching on whether it carries a true order toward OrdinalEncoder, and otherwise branching on low versus high cardinality toward one-hot with handle_unknown or cross-fit target encoding for trees](/imgs/blogs/categorical-and-feature-bugs-5.png)

#### Worked example: 40k one-hot columns versus a smoothed target encoding

A click model has a `publisher_id` column with 38,000 distinct values. One-hot encoding produces a 38,000-wide sparse matrix; the gradient-boosted model trains slowly and the logistic baseline overfits, landing at validation AUC 0.74. Switching to smoothed, cross-fitted target encoding collapses the column to 1 numeric feature: the model trains 6× faster and validation AUC rises to 0.79 because the encoding carries the empirical click rate per publisher with rare publishers shrunk toward the global rate ($m = 20$). The trap to avoid: an earlier attempt computed the target mean on the full training set *including each row's own label*, which gave a glorious training AUC of 0.97 and a validation AUC of 0.71 — a textbook leak. Cross-fitting (compute each fold's encoding from the other folds) removed the leak and produced the honest 0.79. The compact feature was the right call; the *leaked* version of it was a disaster wearing the same name.

## 8. Datetime, timezone, units, and dtype: when "1,000" is a string

The seventh mechanism is the most pedestrian and the most common: the feature has the wrong *type* or *units*, and the model silently mishandles it. These never crash and never warn; they just feed garbage to the model.

- **Numbers parsed as strings.** A CSV column `income` arrives as `"1,200,000"` (thousands separators) or `"\$45.50"` or `"1.2e6"`. `pandas` reads it as `object` dtype, and if you pass it through a categorical encoder it becomes a high-cardinality categorical with one level per unique string — the model never sees the *quantity*. The fix is to parse it: `pd.to_numeric(df["income"].str.replace(",", ""), errors="coerce")`.
- **Datetime as a string or epoch-as-number.** A timestamp left as a string is useless; an epoch integer fed raw makes the model learn that "later is bigger" with no notion of hour-of-day or day-of-week. Parse to datetime and *extract the features that matter*: hour, day-of-week, month, is-weekend, days-since-event. The raw epoch is almost never the feature you want.
- **Timezone drift.** Offline timestamps in UTC, online in local time, is a silent skew that shifts every time feature by a fixed offset — and shifts day-of-week for late-night events. This is a special, common case of the train-serve skew from section 4, and the diff-one-entity test catches it.
- **Float precision and units.** A feature in dollars offline and cents online is off by 100×; a feature in float32 offline and float64 online differs in the last digits. The dtype-and-range audit below catches both.

The detector is a dtype-and-range audit that flags any column whose declared type, value range, or unique-value count is suspicious:

```python
import numpy as np
import pandas as pd

def dtype_range_audit(df, expected_numeric, expected_categorical):
    rows = []
    for col in df.columns:
        s = df[col]
        is_obj = s.dtype == object
        n_unique = s.nunique(dropna=True)
        flag = []
        if col in expected_numeric and is_obj:
            flag.append("numeric-stored-as-string")   # "1,000" bug
        if col in expected_categorical and n_unique > 1000:
            flag.append("suspiciously-high-cardinality")
        if not is_obj:
            arr = pd.to_numeric(s, errors="coerce")
            if np.isinf(arr).any():
                flag.append("contains-inf")
            lo, hi = arr.min(), arr.max()
            if lo is not None and (lo < -1e8 or hi > 1e9):
                flag.append(f"range-outlier[{lo:.3g},{hi:.3g}]")  # -999, 9999 sentinels
        rows.append({"column": col, "dtype": str(s.dtype),
                     "n_unique": int(n_unique), "flags": ",".join(flag) or "ok"})
    return pd.DataFrame(rows)

# print(dtype_range_audit(df, expected_numeric=["income","age","debt"],
#                         expected_categorical=["city","size","device"]))
```

Run this first, on every new dataset, before you train anything. It is thirty lines that surface the `"1,000"` string, the `-999` sentinel, the surprise 250,000-cardinality column, and the inf from a ratio — the four type/range bugs that account for an enormous fraction of "the model trains but is bad" reports. Figure 7 shows the audit as a grid: features down the rows, checks across the columns, each cell a pass or a fail, so the failing feature and the failing check are a single glance.

There is one more dtype trap that deserves a sentence, because it is uniquely nasty: the *silent dtype downcast on a join or a save-load round-trip*. A column that is `int64` in your notebook can become `float64` after a `merge` introduces a single NaN (pandas upcasts integer columns to float to hold the NaN), and a categorical saved to Parquet and reloaded can come back as `object` or as a `category` dtype with a *different category order* than you saved — which, if you are indexing categories by code, silently remaps every value. The defense is to assert your schema explicitly after every load and join: check `df.dtypes` against an expected mapping, and for categorical columns check that the category set and its order match what the encoder was fit on. A schema assert is three lines and it catches a class of bug that is otherwise diagnosable only by the offline-online gap it produces days later.

A useful way to internalize all of section 8 is that **a feature has a contract**: a name, a dtype, a unit, a valid range, and a category set, and the bug is always a violation of that contract that the runtime does not enforce. Python will not stop you from putting a string where a float belongs, a sentinel where a magnitude belongs, or cents where dollars belong — so you enforce the contract yourself, with the audit and the schema assert, at the boundary where the data enters. Every hour spent on that audit is repaid many times over, because the alternative is discovering the violation through a degraded production metric and a multi-day bisection.

![A three by three grid auditing the income, city, and age features against dtype, value range, and unseen-category checks, with cells flagging income stored as object, a -999 age sentinel, and twelve unseen city categories](/imgs/blogs/categorical-and-feature-bugs-7.png)

## 9. The leak-safe pipeline that fixes all seven at once

Everything above converges on one structural pattern: a single `ColumnTransformer` inside a `Pipeline`, fit on training data only, that handles numerics and categoricals correctly and is the *one shared transform* between training and serving. This is the single most important piece of code in the post, because it makes most of these bugs *structurally impossible*: the encoder is fit on train, the imputer is fit on train, the unknown handling is configured, and the exact same fitted object is pickled and called at serve time.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

numeric_cols     = ["age", "income", "debt_to_income"]
nominal_cols     = ["city", "device"]            # no inherent order -> one-hot
ordinal_cols     = ["size"]                       # true order -> ordinal-encode
ordinal_order    = [["small", "medium", "large"]]

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # keep missingness signal
    ("scale",  StandardScaler()),                                      # only matters for linear/NN
])
nominal_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # unseen -> all-zeros
])
ordinal_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ord",    OrdinalEncoder(categories=ordinal_order,
                              handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("nom", nominal_pipe, nominal_cols),
    ("ord", ordinal_pipe, ordinal_cols),
], remainder="drop")  # drop unlisted columns instead of silently leaking an id

# For a LINEAR model, scaling matters; for trees, the scaler is a no-op but harmless inside the pipe.
linear_model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])
tree_model   = Pipeline([("prep", preprocess), ("clf", HistGradientBoostingClassifier())])

# cross_val_score fits the WHOLE pipeline inside each fold -> imputer/encoder/scaler see train only.
print("linear CV acc:", cross_val_score(linear_model, X, y, cv=5).mean())
print("tree   CV acc:", cross_val_score(tree_model,   X, y, cv=5).mean())
```

Three things make this leak-safe and skew-safe, and they are worth naming explicitly:

1. **Every fitted step is fit inside the fold.** Because the whole `Pipeline` is passed to `cross_val_score`, the imputer median, the one-hot vocabulary, the scaler mean/std, and the ordinal order are all learned from the training rows of each fold and merely *applied* to the validation rows. No statistic leaks across the split. This is the structural fix for sub-bug B (imputer leakage) and target-encoding leakage alike.
2. **Unknown handling is configured, not defaulted.** `handle_unknown="ignore"` on the one-hot and `handle_unknown="use_encoded_value"` on the ordinal mean the service will *not crash* on an unseen category — it maps it to all-zeros or to `-1`, both of which the model can handle.
3. **The same object serves.** You `joblib.dump(tree_model.fit(X, y), "model.pkl")` and the service `joblib.load`s it and calls `.predict(serve_df)`. The features are computed by the exact same fitted transform offline and online, so train-serve skew on the *encoding* is impossible. (Skew can still enter *upstream* of the pipeline if the raw columns are built differently — which is exactly why the diff-one-entity test of section 4 stays in your toolkit.)

The comparison below is the diagnostic table this whole post builds toward — symptom, the mechanism behind it, the confirming test, and the fix:

| Symptom | Likely mechanism | Confirming test | Fix |
| --- | --- | --- | --- |
| Linear model bad, trees fine | Label-encoded nominal feature | One-hot it; linear acc jumps | `OneHotEncoder` for nominal |
| `ValueError: unseen labels` in prod | Encoder crashes on unknown | Unseen-category audit on holdout | `handle_unknown="ignore"`/`use_encoded_value` |
| Great offline, worse in prod | Train-serve feature skew | Diff one entity offline vs online | One shared fitted transform |
| Wild predictions on some rows | `-999`/0 sentinel as a number | Range audit flags the outlier | Replace sentinel with NaN, impute |
| Loss/grad NaN at step 1 | inf from a ratio feature | `assert np.isfinite(...)` | Zero-denominator guard on ratios |
| Train AUC ~0.97, val ~0.71 | Target encoding leaked the label | Cross-fit the encoder | `TargetEncoder` inside CV, smoothed |
| Model ignores a feature entirely | Unscaled feature in a distance/linear model | Compare scaled vs unscaled | `StandardScaler` (not for trees) |
| Quantity treated as 250k categories | Number parsed as string | Dtype audit flags `object` | `pd.to_numeric(...errors="coerce")` |

## 10. The bisection in action: debugging the credit model end to end

Let me walk the opening failure all the way to root cause, the way you would at a desk, using make-it-fail-small and read-the-instruments. The symptom: offline accuracy 0.86, production 0.79 and falling, plus intermittent `ValueError: unseen labels` 500s. Six places a bug can hide; we bisect.

The mindset is the one this whole series repeats: do not start by editing the model, start by *localizing* which of the six places holds the bug, using the cheapest discriminating test at each fork. For feature bugs the cheapest discriminator is almost always "does the gap move when I change the model," and the second cheapest is the dtype-and-range audit, which costs no training at all.

**Is it the model?** Swap gradient boosting for a random forest, then for logistic regression. The offline-online gap holds at 6–8 points across all three. A gap invariant to the model is not a model bug. Cross off optimization and model code. The gap is in data or evaluation.

**Is it evaluation (a metric or split bug)?** Re-run the offline metric with a *time-ordered* holdout instead of random CV, so the holdout behaves like the future. Offline accuracy on the time-ordered holdout drops to 0.81 — closer to production. That tells us part of the "0.86" was optimism from a random split that mixed future and past rows, but it does not explain the crashes or the rest of the gap. Evaluation is *part* of the story; keep going.

**Is it data — and which mechanism?** Run the dtype-and-range audit (section 8). It flags `income` as `object` dtype (the `"1,200,000"` string bug) and `age` with a range minimum of `-999` (the sentinel bug). Run the unseen-category audit (section 3) on the time-ordered holdout: `city` shows 12 categories absent from training, 3.1% of holdout rows — the source of the production crashes. Run the diff-one-entity test (section 4) on five production users: `days_since_signup` is skewed on every one (offline as-of-label-date, online as-of-now). Four confirmed feature bugs, each with a green-checkmark test.

**Fix and re-measure.** Parse `income` to numeric; replace `-999`/`0` age with NaN and impute the training median; route everything through the leak-safe `ColumnTransformer` from section 9 with `handle_unknown="ignore"`; and unify `days_since_signup` to a single shared transform computed as-of-prediction-time in both paths. Re-run on the time-ordered holdout: accuracy recovers to 0.86, and the crash rate on the holdout goes to 0. Deploy: production accuracy comes up to 0.85 within a week, and the 500s stop. The model never changed. The features stopped lying. Figure 8 captures the hunt as the cost-ordered sequence it actually was.

![A timeline of the feature-bug hunt in cost order from auditing dtypes through diffing one entity, counting unseen categories, checking the encoder handle_unknown setting, and refitting clean to recover accuracy from 0.79 to 0.86](/imgs/blogs/categorical-and-feature-bugs-8.png)

#### Worked example: the points each fix bought back

It is worth attributing the recovery, because "we fixed the features and it got better" is not a post-mortem. On the time-ordered holdout, the buggy pipeline scored 0.79. Parsing `income` from string to number bought +2.5 points (the model finally saw the quantity instead of 250k phantom categories). Converting the `-999` age sentinel to an imputed value bought +1.8 points (the poisoned rows stopped dominating). Switching `city` from a crashing `OrdinalEncoder` to `OneHotEncoder(handle_unknown="ignore")` bought +1.4 points *and* drove the crash rate to zero. Unifying `days_since_signup` across train and serve closed the last +1.3 points of the train-serve gap. Sum: roughly +7 points, 0.79 → 0.86, no change to the model class, no new data, no hyperparameter tuning. Every point came from making a feature honest. That attribution is the difference between a fix and a guess.

## 11. Case studies and real signatures

These patterns are not hypothetical; they recur across the industry and the literature, and recognizing the signature saves the bisection.

**The Titanic / categorical-as-ordinal beginner trap.** The single most common mistake in entry-level tabular tutorials is `LabelEncoder` on `Embarked` or `Sex` or `Pclass` feeding a logistic regression. `Pclass` is genuinely ordinal (1st > 2nd > 3rd class), so it survives; `Embarked` is nominal (Cherbourg, Queenstown, Southampton have no order), so label-encoding it injects a fake axis. The signature is a logistic-regression baseline that mysteriously underperforms a decision tree on the *same features* — the tell, as section 2 explains, that a nominal feature was label-encoded and the linear model is being wrecked by an order the tree can route around.

**The Kaggle "great LB, terrible private" target-encoding blowup.** A recurring competition post-mortem: a competitor target-encodes a high-cardinality categorical without cross-fitting, computing each category's mean target on the full training set including the row itself. The public-leaderboard and CV scores are spectacular; the private leaderboard collapses. The mechanism is exactly the leakage of section 7 — the encoding partly *is* the label — and the fix is exactly cross-fitting and smoothing. This pattern is common enough that experienced Kagglers treat any unexplained CV-to-LB gap on a dataset with high-cardinality categoricals as target-encoding leakage until proven otherwise.

**The production train-serve skew that Google's ML guidelines flag as a top failure.** Google's published machine-learning engineering guidance (the "Rules of ML" by Martin Zinkevich) names training-serving skew as one of the most damaging and common production failures, with the recommended fix being precisely to *reuse the same feature-computation code* between training and serving and to *log serving features and join them back for training*. The diff-one-entity test of section 4 is the lightweight version of that discipline; a feature store is the heavyweight version. The signature — a model that validates well and degrades in production with no model change — is the canonical feature bug, and the industry consensus fix is "one code path."

**The sentinel-as-number medical-data classic.** Older clinical and survey datasets routinely encode "not measured" as `-9`, `-99`, `-999`, or `9999`. A model trained without converting these to NaN learns that a missing blood-pressure reading is a blood pressure of negative nine, which not only poisons that feature but corrupts any interaction term involving it. The signature is a feature whose importance is high and whose partial-dependence plot has a bizarre spike at an impossible value — a dead giveaway that a sentinel is being read as a magnitude.

**The new-category cold-start that looks like model decay.** A marketplace launches a new product category every few weeks. The recommendation model's offline metrics are stable, but its production click-through sags for the first days after each launch and then recovers. The team chases "model decay" — retraining more often, adding capacity — and nothing helps, because the cause is structural: every new category is *unseen* by the encoder, gets the all-zeros (or `-1`) treatment, and therefore gets a cold-start prediction with no category-specific signal until the next retrain ingests it. The signature is a sawtooth in production performance synchronized with category launches, invisible offline because the offline holdout does not contain the future categories. The fix is not a bigger model; it is an encoding strategy designed for the tail (an explicit unknown bucket the model has trained on, plus faster ingestion of new categories), and the diagnostic is the unseen-category audit from section 3 run against the most recent slice of data.

## 12. When this is (and isn't) your bug

Be decisive about when a symptom points at features and when it points elsewhere, because chasing a feature bug that is really an optimization bug wastes the same week.

- **A gap that moves when you change the model is not a feature bug.** If swapping the model class closes or opens the offline-online gap, the problem is capacity/regularization/optimization, not features. Feature bugs produce a gap that is *invariant to the model*, because the features are the same regardless of what consumes them. This is the single most useful discriminator in the post.
- **A crash at training time is usually a shape or dtype bug, not a categorical bug.** `ValueError: unseen labels` at *serving* time is a categorical bug; a shape mismatch or a `could not convert string to float` at *training* time is the dtype/parse issue of section 8 or a [shape bug](/blog/machine-learning/debugging-training/shape-bugs-and-silent-broadcasting), surfaced before the model ever fits.
- **A too-good offline number is leakage, not a feature bug per se.** If the offline number is *unrealistically high* (the AUC-0.97-on-a-hard-problem signature), that is target leakage or fit-before-split leakage — read [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage). Feature bugs more often make the offline number *honest-but-wrong-for-prod*, not inflated. (Target encoding without cross-fitting straddles both worlds: it is a feature *technique* that *causes* leakage.)
- **A model that degrades over time with no code change is distribution shift, not a pipeline bug.** If your features are computed identically and the model still decays week over week, the world changed, not the pipeline — that is the [distribution-shift](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world) story (the categories' *meaning* drifted), not a skew bug. The discriminator: a skew bug is present from day one; drift grows over time.
- **If scaling changes a tree's accuracy, you have a bug, not a discovery.** Section 6 proves tree splits are scale-invariant. If standardizing a feature moves your gradient-boosting accuracy, something else changed (the scaler leaked, or it introduced NaNs, or it altered the feature order) — investigate that, do not conclude trees benefit from scaling.

## 13. Key takeaways

- **Label-encode the target, never a nominal input feature.** Label-encoding a nominal column injects a fake order: it wrecks linear and distance models loudly and degrades trees quietly. Use `OneHotEncoder` for nominal, `OrdinalEncoder(categories=...)` only for genuinely ordered features.
- **Always configure `handle_unknown`.** An encoder fit on train will meet unseen categories in production. `OneHotEncoder(handle_unknown="ignore")` maps them to all-zeros; `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` avoids the crash. Count unseen categories on a future-like holdout *before* you ship.
- **The diff-one-entity test settles train-serve skew.** Compute one real entity's feature vector offline and online and diff them element by element. A divergence is the bug; the fix is one shared fitted transform between training and serving.
- **Convert sentinels to NaN before imputing, and fit the imputer on train only.** A `-999` read as a magnitude poisons the feature; an imputer fit before the split leaks. Keep `add_indicator=True` so missingness stays a signal, and guard ratio features against `inf`.
- **Scale for gradient-based and distance-based models; never for trees.** Tree splits are invariant to any monotone transform, so scaling a tree is wasted work that hurts interpretability and risks leakage. Scaling is essential for linear/NN/k-NN conditioning.
- **High cardinality forces a real tradeoff.** One-hot is leak-proof but blows up to thousands of columns; target encoding is compact and strong but leaks unless cross-fitted and overfits unless smoothed. Use `sklearn`'s `TargetEncoder` (cross-fitted) for high-cardinality features feeding trees.
- **Audit dtypes and ranges first, on every dataset.** A number parsed as a string becomes a 250k-cardinality categorical; an epoch fed raw learns "later is bigger." The thirty-line dtype-and-range audit catches the four most common type bugs in one pass.
- **Put every fitted step inside a `Pipeline` and fit it inside cross-validation.** A `ColumnTransformer` in a `Pipeline` makes imputer/encoder/scaler leakage structurally impossible and gives you the one shared object that serves, killing most of these bugs at the architecture level.
- **A gap invariant to the model is a feature bug; a gap that moves with the model is not.** This single discriminator routes you to the right one of the six places before you touch any code.

## 14. Further reading

- **scikit-learn User Guide — Preprocessing data and `compose.ColumnTransformer`.** The authoritative reference for `OneHotEncoder` (`handle_unknown`), `OrdinalEncoder`, `SimpleImputer` (`add_indicator`), `StandardScaler`, `TargetEncoder` (cross-fitted, added in 1.3), and how `Pipeline` + `ColumnTransformer` keep transforms fit on train only.
- **Martin Zinkevich, "Rules of Machine Learning: Best Practices for ML Engineering" (Google).** The canonical treatment of training-serving skew and the discipline of reusing one feature-computation code path; rules on logging serving features and joining them back for training.
- **Daniele Micci-Barreca, "A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems" (2001).** The original smoothed target-encoding formulation behind the empirical-Bayes shrinkage in section 7.
- **`category_encoders` documentation.** Practical reference for target, leave-one-out, CatBoost, and James-Stein encoders, with the cross-fitting and smoothing options that make high-cardinality encoding leak-safe.
- **Within this series:** the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the symptom→suspect→test→fix decision tree this post instantiates) and the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone checklist). Siblings: [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage), [distribution shift: train vs. the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world), [gradient boosting and imbalance debugging](/blog/machine-learning/debugging-training/gradient-boosting-and-imbalance-debugging), and [look at your data before you train](/blog/machine-learning/debugging-training/look-at-your-data-before-you-train).

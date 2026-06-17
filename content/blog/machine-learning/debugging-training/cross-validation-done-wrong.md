---
title: "Cross-Validation Done Wrong: Group Leaks, Time Folds, and the Optimistic Estimate"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A field guide to the ways cross-validation quietly lies upward — group leaks, time folds, tuning on the test fold — and the runnable splitters and nested-CV audits that get you back to an honest number."
tags:
  [
    "debugging",
    "model-training",
    "cross-validation",
    "data-leakage",
    "tabular",
    "scikit-learn",
    "evaluation",
    "finetuning",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/cross-validation-done-wrong-1.png"
---

A hospital readmission model crossed my desk with a 5-fold cross-validated ROC-AUC of 0.94. The notebook was clean, the features were sensible, the pipeline used a real `sklearn` `Pipeline`, and the author had even held out a final test set that also scored 0.93. Everyone signed off. Three weeks after it went live, the on-call dashboard read 0.79 AUC and falling. Nobody had touched the model. The data hadn't drifted. The 0.94 was simply never real — it was an artifact of how the folds were drawn. The same patients had rows in both the train and validation side of every fold, so the model had quietly learned to recognize *patients*, not the *disease*, and cross-validation cheerfully rewarded it for doing so.

This is the most demoralizing class of bug in machine learning, because the instrument you trust to tell you the truth is the instrument that's lying. Cross-validation exists for exactly one purpose: to give you an honest, low-variance estimate of how a model will perform on data it has never seen. When it's done right, the CV score, a genuine forward holdout, and the production number all agree within noise. When it's done wrong, CV becomes an optimism-generator — it almost always lies *upward*, because nearly every CV mistake leaks information from the validation fold into training, and leaked information only ever helps the score. You do not accidentally make your CV pessimistic.

![A vertical stack showing the CV score, a forward holdout, the production number, and an honest CV all converging when leaks are removed](/imgs/blogs/cross-validation-done-wrong-1.png)

By the end of this post you will be able to take any suspiciously good CV number and localize *why* it's inflated in a few minutes: run a group-overlap audit to catch entities straddling folds, plot the temporal coverage of folds to catch training on the future, run nested CV and read the flat-versus-nested gap as a direct measurement of selection optimism, and run a shuffle-label sanity check that should collapse any honest CV to chance. We will use `scikit-learn`'s real splitters — `GroupKFold`, `StratifiedGroupKFold`, `TimeSeriesSplit`, `cross_val_score`, and nested `GridSearchCV` — and we'll back every claim with a before→after where switching the splitter drops a too-good 0.94 to an honest 0.79 that finally matches production.

This is one post in the *Debugging AI Training & Finetuning* series, and it sits squarely in the **evaluation** corner of the six places a bug can hide — data, optimization, model code, numerics, systems, and evaluation. The discipline is the same as everywhere else in the series: don't guess, *bisect*. A bad CV number has a small number of possible causes, each with a cheap confirming test. We'll go through them in the order you should actually run them.

## 1. What cross-validation is supposed to do (and the one assumption it makes)

Strip away the folklore and cross-validation is a variance-reduction trick for estimating generalization error. You have a dataset and you want to know the expected loss of your modeling *procedure* on a fresh sample from the same distribution. A single train/test split gives you one noisy estimate. $k$-fold CV gives you $k$ estimates from $k$ different train/validation partitions and averages them, which cuts the variance of the estimate roughly like averaging $k$ correlated samples.

Formally, $k$-fold CV partitions the $n$ rows into $k$ disjoint folds $F_1, \dots, F_k$. For each fold $i$, you train on the other $k-1$ folds and evaluate on $F_i$, producing a score $s_i$. The CV estimate is $\bar{s} = \frac{1}{k}\sum_i s_i$. The promise is that $\bar{s}$ is an approximately unbiased estimate of the true generalization performance of your training procedure on $n \cdot \frac{k-1}{k}$ rows.

That promise rests on **one load-bearing assumption**: the folds must be statistically independent and identically distributed samples, such that knowing everything in the training folds gives you *no unfair information* about the rows in the validation fold beyond what the real-world deployment scenario would also give you. Every cross-validation bug in this post is, at bottom, a violation of that single assumption. The information that leaks differs — the identity of an entity, the value of a future event, the result of a hyperparameter search — but the failure mode is identical: the validation fold is no longer a fair stand-in for "data the model has never seen," so the score it produces is no longer a fair estimate of generalization.

Here's the principle worth internalizing before we go further: **CV measures the generalization gap of a procedure, and any leak shrinks the measured gap without shrinking the real one.** The model still generalizes exactly as well as it always did. You've only corrupted the ruler. That's why the production number doesn't move when the leak is "fixed" in the metric — production was already telling you the truth.

It's worth being precise about what "the procedure" includes, because that phrasing does a lot of work. The procedure is *everything* you do to turn raw data into predictions: the preprocessing, the feature engineering, the hyperparameter search, the model fit. CV is honest only when each fold re-runs the *entire* procedure on that fold's training data and applies it to that fold's validation data, with the validation data contributing to *nothing* along the way. The instant any step — a scaler's mean, a selected feature set, a chosen hyperparameter — is computed using information the validation fold could see, that step has "looked ahead," and the fold's score is no longer a generalization estimate. Most of this post is really one idea applied to different steps: keep the validation fold *blind* to everything until the moment it's scored.

### The information-leak argument, made precise

Why does a leak *always* bias the estimate upward and never down? Consider the validation fold $F_i$. The model's score on $F_i$ depends on how much of $F_i$'s label-relevant information the model can reconstruct from its training data. Call the legitimate, generalizable signal $S$ and the leaked, non-generalizable signal $L$. At deployment, only $S$ is available, so true performance is a function of $S$ alone. During leaky CV, the model can exploit $S + L$. Because optimization is a maximization — the training procedure will use *any* signal that lowers validation loss — the score under $S + L$ is at least as high as the score under $S$, and strictly higher whenever $L$ carries label information. There is no symmetric mechanism that would make leaked information *hurt* the validation score. The asymmetry is the whole story: leaks help, so leaky CV is optimistic, full stop.

This is also why "but my holdout test set agreed with CV" is not the reassurance people think it is. If the *same* leak contaminates both the CV folds and the final holdout — the same patient appears across the CV split and also in the held-out set, or the same preprocessing was fit on everything once at the top of the notebook — then the holdout inherits the same optimism. A holdout only protects you against *one* specific failure (tuning on the test set), and only if it is drawn so the leak cannot cross into it. We'll see exactly how to draw it.

### The bias–variance trade-off in the choice of k

There's a second, more benign reason CV estimates can be off, and it's worth separating from leakage so you don't confuse them. The number of folds $k$ trades off bias against variance in the *estimate itself*, even when there's no leak at all. With small $k$ — say 2 or 3 — each training fold is a small fraction of your data (50% or 67%), so the model you cross-validate is systematically weaker than the model you'll ultimately ship on 100% of the data. That makes the CV estimate *pessimistic*: a rare case of CV lying *downward*. With large $k$ — leave-one-out at the extreme — each training fold is nearly all the data, so the bias of the estimate is tiny, but the $k$ models are almost identical to each other and their errors are highly correlated, which inflates the *variance* of the average. Leave-one-out also costs $n$ model fits, which is brutal on anything but tiny data.

The standard recommendation — 5 or 10 folds — is the empirical sweet spot identified in Kohavi's 1995 study of CV for accuracy estimation: low enough bias that the training folds resemble the full dataset, low enough variance that the average is stable, and a tractable number of fits. The crucial thing to understand is that **the $k$ trade-off is about the precision of an honest estimate; leakage is about the honesty of the estimate.** They are independent failure modes. You can have a perfectly-chosen $k=5$ that lies by 0.15 AUC because of a group leak, and a noisy $k=2$ that's honest. Fix the leak first — it dominates — then tune $k$ for precision.

### A vocabulary check before we go further

Because this post leans on a few terms, here are the working definitions so a reader coming from tutorial-land doesn't trip:

- **Fold:** one of the $k$ disjoint chunks the data is partitioned into; in each round one fold is the validation set and the rest are training.
- **Group / entity:** the real-world unit a set of rows belongs to (a patient, a user, a device). The thing the model must generalize *across*.
- **Stratification:** drawing folds so each preserves the global class proportions — essential when classes are imbalanced.
- **Forward chaining:** building time-respecting folds where the training window grows forward and validation always lies after training.
- **Purge / embargo:** for horizon labels, dropping training rows whose label window overlaps the validation block (purge) and a small post-validation gap (embargo) to kill residual serial correlation.
- **Nested CV:** two CV loops — an inner one that tunes hyperparameters and an outer one that scores the whole tune-then-fit procedure on folds the inner loop never saw.
- **Optimism:** the gap between a reported score and the honest one; in CV it's almost always positive (the score is too good).

## 2. Group leakage: the model learns the entity, not the pattern

This is the single most common way CV lies, and it is brutal precisely because the code looks correct. You call `KFold`, you shuffle, you get five clean folds, and every fold has the right class balance. Nothing crashes. What you've missed is that your *rows* are not your *units of generalization*.

Many real datasets have a grouping structure where multiple rows belong to the same underlying entity: several lab visits per patient, many clicks per user, dozens of photos from the same camera, repeated measurements from the same sensor, multiple sentences from the same document, several transactions per merchant. At deployment, the model will see *new entities* — a patient it has never met, a user who just signed up. So the honest question CV must answer is: *how well does the model do on a new entity?* But plain `KFold` splits at the row level, so the same patient's rows scatter across train and validation. The model sees patient #4471 in training, learns whatever idiosyncratic signal identifies that patient, and then is rewarded at validation time for recognizing patient #4471 again. That recognition does not transfer to a new patient, so it evaporates in production.

![A before and after comparison showing GroupKFold dropping a leaked CV AUC of 0.94 to an honest 0.79 that matches production](/imgs/blogs/cross-validation-done-wrong-2.png)

The fix is to split at the level of the entity, not the row. `scikit-learn` gives you `GroupKFold` (and `StratifiedGroupKFold` when you also need class balance), which guarantee that all rows for a given group land entirely in train or entirely in validation, never both.

What makes group leakage so easy to miss is that it requires *no obvious mistake*. There's no suspicious feature, no preprocessing slip, no off-by-one — just a `KFold` call on data whose rows happen to cluster into entities, which describes a huge fraction of real datasets. The grouping structure is often implicit: you might not even have a `patient_id` column because the rows were flattened from a per-patient source long ago, and the only trace of the grouping is a near-duplicate pattern across rows. This is why the *first* thing to ask of any new dataset is "what is the unit that repeats?" — and if rows repeat per entity, your default `KFold` is already wrong before you've written a line of modeling code.

![A three by three grid of patient rows where one patient straddles the train and validation side of a fold, leaking its identity](/imgs/blogs/cross-validation-done-wrong-3.png)

### The diagnostic: a group-overlap audit

Before you trust any CV number on grouped data, run this audit. It takes the splitter and the group labels and counts, for every fold, how many groups appear on *both* sides of the split. For an honest grouped split this number is exactly zero. For plain `KFold` on grouped data it is large.

```python
import numpy as np
from sklearn.model_selection import KFold, GroupKFold

def group_overlap_audit(splitter, X, y, groups):
    """Count groups that appear in BOTH train and val for each fold.
    An honest grouped split returns 0 straddling groups everywhere."""
    total_straddle = 0
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        g_tr = set(groups[tr])
        g_va = set(groups[va])
        straddle = g_tr & g_va
        total_straddle += len(straddle)
        print(f"fold {fold}: {len(straddle):4d} groups straddle "
              f"(train groups {len(g_tr)}, val groups {len(g_va)})")
    print(f"TOTAL straddling groups: {total_straddle}")
    return total_straddle

# n rows, but only 200 distinct patients -> ~25 rows/patient
rng = np.random.default_rng(0)
n = 5000
groups = rng.integers(0, 200, size=n)        # patient id per row
X = rng.normal(size=(n, 20))
y = rng.integers(0, 2, size=n)

print("Plain KFold (row-level shuffle):")
group_overlap_audit(KFold(5, shuffle=True, random_state=0), X, y, groups)

print("\nGroupKFold (entity-level):")
group_overlap_audit(GroupKFold(5), X, y, groups)
```

Plain `KFold` will report something like `~160 groups straddle` *per fold* — almost every patient is on both sides. `GroupKFold` reports `0` everywhere. That single number is the difference between a CV score you can trust and one you can't, and it costs you four lines to check.

#### Worked example: a readmission model deflates from 0.94 to 0.79

Here's the concrete readmission scenario with synthetic-but-realistic structure: a patient-level signal that the model can memorize, plus a weaker generalizable disease signal. We run plain `KFold` (the bug) and `GroupKFold` (the fix) and watch the AUC collapse to the honest number.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_score

rng = np.random.default_rng(7)
n_patients = 300
rows_per_patient = 20
n = n_patients * rows_per_patient

groups = np.repeat(np.arange(n_patients), rows_per_patient)

# A strong PATIENT-level latent (memorizable, does NOT generalize)
patient_latent = rng.normal(size=n_patients)[groups]
# A weaker DISEASE signal that DOES generalize
disease = rng.normal(size=n)
logit = 0.4 * disease + 1.6 * patient_latent
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

# Features expose the per-row disease signal + a noisy patient fingerprint
X = np.column_stack([
    disease + 0.3 * rng.normal(size=n),                  # generalizable
    patient_latent + 0.2 * rng.normal(size=n),           # patient fingerprint
    rng.normal(size=(n, 8)),                             # noise
])

clf = LogisticRegression(max_iter=1000)

auc_kfold = cross_val_score(
    clf, X, y, groups=groups, scoring="roc_auc",
    cv=KFold(5, shuffle=True, random_state=0)).mean()

auc_group = cross_val_score(
    clf, X, y, groups=groups, scoring="roc_auc",
    cv=GroupKFold(5)).mean()

print(f"Plain KFold  AUC: {auc_kfold:.3f}   <- leaks patient identity")
print(f"GroupKFold   AUC: {auc_group:.3f}   <- honest, new-patient performance")
```

The plain-`KFold` number lands around 0.92–0.95 because the model can lean on the patient fingerprint that's present on both sides. `GroupKFold` lands around 0.78–0.80 because now the validation patients are genuinely new, the fingerprint is useless, and only the weak disease signal survives. **The gap between those two numbers is the leak, measured in AUC points.** When the model shipped, production saw 0.79 — exactly the `GroupKFold` number, never the `KFold` number, because production patients are always new.

| Splitting strategy | Straddling groups/fold | CV ROC-AUC | Matches production? |
| --- | --- | --- | --- |
| `KFold(shuffle=True)` | ~160 | 0.94 | No (prod 0.79) |
| `StratifiedKFold` | ~160 | 0.94 | No (prod 0.79) |
| `GroupKFold` | 0 | 0.79 | Yes |
| `StratifiedGroupKFold` | 0 | 0.79 | Yes (+ balanced folds) |

### How much does grouping cost you, and why that's the point

A reasonable objection at this point: "If `GroupKFold` drops my AUC by 15 points, isn't it *hurting* my model?" No — and internalizing why is the heart of this section. The model is exactly as good as it always was. `GroupKFold` didn't make it worse; it stopped *measuring* a skill the model doesn't actually have. The 0.94 was the score for "recognize a patient I've seen before," a task that never occurs in production. The 0.79 is the score for "diagnose a patient I've never met," the only task that matters. Switching splitters changed the *question CV asks*, and the honest question has a lower answer. Reporting the harder, lower number isn't pessimism — it's the difference between a number you can defend to a regulator and a number that gets you paged at 2 a.m.

There's a quantitative way to see the size of the leak before you even retrain. The leak scales with two things: how many rows each entity contributes (more rows per entity → more fingerprint to memorize) and how strong the per-entity latent signal is relative to the generalizable signal. In the readmission worked example below, the patient latent had coefficient 1.6 in the logit while the disease signal had 0.4 — a 4:1 ratio — which is why the leak is so large. If each patient had exactly one row, `GroupKFold` and `KFold` would agree, because there'd be no entity to straddle a fold. The general rule: **the more your rows cluster into few, repeatedly-sampled entities, the more a row-level split lies.** A dataset of 5,000 rows from 200 patients has only 200 *effective* independent samples for the purpose of generalization, and a row-level CV that pretends it has 5,000 is overconfident by exactly that factor.

### When you also need stratification

If your grouped problem is imbalanced — say 8% positive — plain `GroupKFold` can hand you folds with wildly different positive rates because it balances on *group size*, not on *label*. That inflates the variance of your per-fold scores and can bias the mean for metrics that are sensitive to base rate. `StratifiedGroupKFold` is the right tool: it keeps every group intact *and* tries to balance the class distribution across folds. Use it whenever you have both grouping and imbalance, which in practice is most real-world grouped datasets (rare disease, fraud, churn).

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
    pos_rate = y[va].mean()
    straddle = len(set(groups[tr]) & set(groups[va]))
    print(f"fold {fold}: val pos-rate {pos_rate:.3f}, straddle {straddle}")
```

You should see `straddle 0` on every fold and val positive-rates clustered tightly around the global 8%. If you see positive-rates ranging from 3% to 14%, you were using plain `GroupKFold` and your fold variance was hiding a problem.

## 3. Time-series CV done wrong: training on the future to predict the past

The second great way to make CV lie is to apply ordinary $k$-fold to data that has a time order. The moment your rows are timestamped — daily sales, hourly sensor readings, tick-by-tick prices, user events — random shuffling commits a subtle but fatal sin: it puts rows from the *future* in the training set and rows from the *past* in the validation set. The model is then allowed to learn from events that, at the moment of the prediction it's being scored on, had not yet happened.

This matters because temporal data is autocorrelated. Tomorrow looks a lot like today. If "today" is in your training fold and "tomorrow" is in your validation fold, the model can essentially interpolate — it has seen both neighbors of the point it's predicting. In production it will only ever have the *past*, so it has to extrapolate, which is much harder. The shuffle-CV number measures interpolation skill; production measures extrapolation skill; they are not the same number, and the first is reliably higher.

![A timeline showing forward-chaining folds where the training window grows in time and each validation fold sits after it with an embargo gap](/imgs/blogs/cross-validation-done-wrong-5.png)

The fix is `TimeSeriesSplit`, which does **forward chaining**: fold 1 trains on the earliest block and validates on the next; fold 2 trains on the first two blocks and validates on the third; and so on. The training window only ever grows forward, and every validation fold is strictly *after* its training data — exactly the deployment scenario, where you train on history and predict the next period.

![A before and after comparison showing random KFold inflating a time-series R2 to 0.71 while a purged and embargoed split returns an honest 0.24 that matches live performance](/imgs/blogs/cross-validation-done-wrong-6.png)

### The science: autocorrelation is the leak channel

Make the mechanism concrete. Suppose your target follows a simple autoregressive process $y_t = \phi\, y_{t-1} + \varepsilon_t$ with $|\phi|$ close to 1 (high persistence, as in most financial and demand series). The correlation between $y_t$ and $y_{t+h}$ decays like $\phi^{|h|}$. Under random shuffling, the expected time-gap between a validation point and its *nearest* training neighbor is small — on average about one step — so the training set contains a point correlated with the validation target at level $\approx \phi$, which for $\phi = 0.95$ is enormous. The model exploits this neighbor and posts a high $R^2$. Under forward chaining, the nearest training neighbor of a validation point is at least one full fold away, the correlation has decayed by $\phi^{(\text{fold size})}$, and the realistic difficulty of the task is restored. **The leak is literally the autocorrelation the random split lets the model peek at.**

### The diagnostic: plot the temporal coverage of folds

The fastest way to *see* a time-fold bug is to plot, for every fold, the time span of its training rows and its validation rows. With `TimeSeriesSplit` you'll see a staircase: train spans grow, val spans march forward, and train always sits to the left of val. With plain `KFold` you'll see train and val intervals fully overlapping in time on every fold — the unmistakable signature of training on the future.

```python
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

def temporal_coverage(splitter, times, name):
    print(f"\n{name}:")
    for fold, (tr, va) in enumerate(splitter.split(times)):
        tr_lo, tr_hi = times[tr].min(), times[tr].max()
        va_lo, va_hi = times[va].min(), times[va].max()
        train_after_val = tr_hi > va_lo     # leak if True
        flag = "  <-- TRAIN OVERLAPS/AFTER VAL (leak)" if train_after_val else ""
        print(f" fold {fold}: train [{tr_lo:3d}..{tr_hi:3d}]  "
              f"val [{va_lo:3d}..{va_hi:3d}]{flag}")

times = np.arange(1000)          # a strict time index 0..999
temporal_coverage(KFold(5, shuffle=True, random_state=0), times, "KFold (shuffled)")
temporal_coverage(TimeSeriesSplit(5), times, "TimeSeriesSplit")
```

Every `KFold` fold gets flagged; no `TimeSeriesSplit` fold does. If you want one number instead of a plot, count the folds where `train max > val min` — it should be zero for any time-correct splitter.

### Purging and the embargo: the finance "purged K-fold" idea

Forward chaining fixes the gross leak, but there's a subtler one when your *labels* span time. In many problems a label at time $t$ is computed from a window — "did the patient get readmitted within 30 days," "did the price move up over the next hour," "was the loan repaid over 12 months." If the training fold ends at $t$ and the validation fold begins at $t+1$, but the *label* of the last training row depends on outcomes that extend into the validation period, then training and validation overlap in *information* even though they don't overlap in *index*. Marcos López de Prado named the fix in *Advances in Financial Machine Learning* (2018): **purged K-fold cross-validation**. You **purge** training rows whose label windows overlap the validation block, and you add an **embargo** — a small gap after the validation block — to kill residual autocorrelation from serial dependence that the model could otherwise exploit.

```python
import numpy as np

def purged_walk_forward(n, n_splits=5, label_horizon=24, embargo=5):
    """Forward-chaining folds with purge + embargo.
    Yields (train_idx, val_idx) where train rows whose label window
    (i .. i+label_horizon) overlaps the val block are PURGED, and an
    embargo gap after val is excluded from train."""
    fold_size = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        val_lo = k * fold_size
        val_hi = (k + 1) * fold_size
        val_idx = np.arange(val_lo, min(val_hi, n))

        train_idx = np.arange(0, val_lo)
        # PURGE: drop train rows whose label window reaches into val
        keep = (train_idx + label_horizon) < val_lo
        train_idx = train_idx[keep]
        # EMBARGO is enforced by never adding rows after val to THIS train
        yield train_idx, val_idx

for tr, va in purged_walk_forward(1000, n_splits=4, label_horizon=24, embargo=5):
    overlap = (tr.max() + 24) >= va.min() if len(tr) else False
    print(f"train n={len(tr):3d} (ends {tr.max() if len(tr) else '-'}), "
          f"val [{va.min()}..{va.max()}], label-window leak: {overlap}")
```

The `label-window leak` flag should be `False` on every fold. The first time you add purging to a financial or survival model, watch the CV score *drop* — that drop is the label-window leak you'd been scoring, and it's the difference between a backtest that looks tradeable and one that survives going live.

To make the embargo concrete, take a model that predicts a 5-day-ahead price move. The label for the last training day depends on prices over the *next* five days, which fall inside the validation block. Without purging, the model has effectively seen five days of validation outcomes baked into a single training label — a small leak per row, but compounded across the boundary it adds up, especially in serially-correlated series where those days are also predictive of the rows just after them. The embargo extends the buffer a little further to account for the fact that features themselves (rolling averages, momentum indicators) blend information across the boundary even when the raw labels don't. The right embargo length is roughly the longest lookback your features use; the right purge length is the label horizon. Get both from your feature and label definitions, not from a default — a purge of 0 on a 30-day-horizon label is the same as no purge at all, and a generous embargo on a short-horizon problem just wastes data. The discipline is to *derive* these lengths from how your data is constructed, then verify with the audit that no fold's training label window reaches into validation.

#### Worked example: a demand forecaster's R-squared falls from 0.71 to 0.24

A retail demand model trained on two years of daily sales reported a cross-validated $R^2$ of 0.71 with shuffled 5-fold. The team was thrilled — until the live $R^2$ over the next quarter came in at 0.22. The story is identical to the readmission case but with time instead of entities as the leak channel.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

rng = np.random.default_rng(3)
n = 730                                   # 2 years of daily sales
t = np.arange(n)
# Persistent, autocorrelated demand (phi ~ 0.95) + weak trend
y = np.zeros(n)
for i in range(1, n):
    y[i] = 0.95 * y[i - 1] + rng.normal()
y += 0.002 * t

# Features = recent lags (legit) but the SHUFFLE is what leaks
X = np.column_stack([np.roll(y, 1), np.roll(y, 7), t])
X[:7] = 0

model = Ridge(alpha=1.0)

r2_shuffle = cross_val_score(model, X, y, scoring="r2",
                             cv=KFold(5, shuffle=True, random_state=0)).mean()
r2_forward = cross_val_score(model, X, y, scoring="r2",
                             cv=TimeSeriesSplit(5)).mean()

print(f"Shuffled KFold R2:  {r2_shuffle:.3f}   <- trains on the future")
print(f"TimeSeriesSplit R2: {r2_forward:.3f}   <- honest, forward-looking")
```

Shuffled CV reports something near 0.70 because it can interpolate between adjacent days that are 95% correlated. `TimeSeriesSplit` reports near 0.25 because now it has to forecast forward from history only. The live number was 0.22 — squarely in the forward-CV range. The shuffle didn't make the model better; it made the *measurement* better, which is worse than useless.

### Two subtleties of TimeSeriesSplit that bite people

First, `TimeSeriesSplit` assumes your data is *already sorted by time*. It splits by row position, not by timestamp. If your dataframe arrived in some other order — sorted by entity, or shuffled by an earlier step — `TimeSeriesSplit` will happily forward-chain over a meaningless index and give you garbage that *looks* time-correct. Always `df.sort_values("timestamp")` before you split, and verify with the temporal-coverage audit above that the train spans actually precede the val spans in *time*, not just in row number.

Second, forward chaining produces *unequal training sizes*: fold 1 trains on a small early window, fold 5 on almost everything. The early folds are weak simply because they have little data, which drags down the CV mean and inflates its variance. This is usually *fine* — it honestly reflects that a freshly-deployed model with little history is weaker — but if you want a fixed-size rolling window instead of an expanding one (common in regimes where old data goes stale), use the `max_train_size` argument to cap the training window, or roll your own sliding-window splitter. The choice between expanding and rolling windows is a modeling decision about how much history stays relevant, and CV should mirror whichever one you'll actually deploy.

```python
from sklearn.model_selection import TimeSeriesSplit

# Expanding window (default): train grows each fold.
expanding = TimeSeriesSplit(n_splits=5)

# Rolling window: cap the training history at 180 rows, add a 7-row gap
# (embargo) between train and val to kill boundary autocorrelation.
rolling = TimeSeriesSplit(n_splits=5, max_train_size=180, gap=7)

for name, cv in [("expanding", expanding), ("rolling+gap", rolling)]:
    sizes = [(len(tr), len(va)) for tr, va in cv.split(range(1000))]
    print(f"{name:12s} (train,val) per fold: {sizes}")
```

The `gap` argument is `TimeSeriesSplit`'s built-in embargo: it leaves `gap` rows between the end of training and the start of validation, which is the lightweight version of the purge-and-embargo machinery for problems where labels don't span a long horizon but you still want a buffer against serial leakage at the fold boundary.

## 4. Tuning on the test fold: the optimism of model selection

You've fixed grouping and time. Your splitter is correct. And your CV number is *still* optimistic — by a smaller amount, but reliably. This third leak is the most intellectually slippery because no single row crosses any boundary. The leak is *selection*: you used the same cross-validation both to *choose* your hyperparameters and to *report* your score.

Here's the mechanism. Suppose you try $M$ hyperparameter configurations and pick the one with the best CV score. Each configuration's CV score is the true performance plus noise, $\hat{s}_m = \mu_m + \eta_m$. When you take the *maximum* over $M$ noisy estimates and report it, you are reporting an order statistic, and the expected maximum of $M$ noisy draws is *above* the mean even when every $\mu_m$ is identical. You've selected partly for genuinely-good configs and partly for *lucky noise*. The reported number includes that luck. This is the multiple-comparisons problem in disguise, and it's the same statistical sin as p-hacking: search enough and something will look good by chance.

How big is the bias? For $M$ independent configurations whose CV scores have standard deviation $\sigma$ across folds, the expected inflation of the selected score scales roughly with $\sigma \cdot \sqrt{2 \ln M}$ — it grows with the *log* of how many things you tried and *linearly* with how noisy your CV is. So small datasets (high $\sigma$) and big hyperparameter grids (large $M$) compound: a 200-row dataset with a 500-point random search can post a CV score one to several points above what the same procedure delivers on a fresh fold it never selected on. The fix is to never let the score you *report* come from a fold you *selected* on.

![A before and after comparison where flat cross-validation tunes and reports on the same folds while nested cross-validation scores on outer folds the search never saw](/imgs/blogs/cross-validation-done-wrong-7.png)

### Nested CV: the unbiased estimator

Nested cross-validation separates the two jobs into two loops. The **inner loop** runs over the training portion of each outer fold and does all the hyperparameter searching. The **outer loop** takes the winner from the inner loop, refits it, and scores it on the outer test fold — which the inner search never touched. Because the outer test fold played no part in selecting the hyperparameters, the outer score is an honest estimate of *the entire procedure, tuning included*. The expensive truth is that nested CV measures the performance of "search-then-fit," not of any single fixed model, and that's exactly what you want to report.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

X, y = make_classification(n_samples=400, n_features=20, n_informative=5,
                           class_sep=0.7, random_state=0)

param_grid = {"C": [0.01, 0.1, 1, 10, 100],
              "gamma": [1e-3, 1e-2, 1e-1, 1]}

inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)

# Inner loop: tuning. This object IS the "procedure" we score.
clf = GridSearchCV(SVC(), param_grid, cv=inner, scoring="roc_auc")

# FLAT (biased): tune on all data, then report the BEST inner score.
clf.fit(X, y)
flat_score = clf.best_score_

# NESTED (honest): outer loop scores the whole tune-then-fit procedure
nested_scores = cross_val_score(clf, X, y, cv=outer, scoring="roc_auc")

print(f"Flat CV (best tuned score):  {flat_score:.3f}   <- optimistic")
print(f"Nested CV (mean outer):      {nested_scores.mean():.3f} "
      f"+/- {nested_scores.std():.3f}")
print(f"Optimism (flat - nested):    {flat_score - nested_scores.mean():+.3f}")
```

The key trick to read in this snippet: `clf` is a `GridSearchCV`, and we hand that *whole object* to `cross_val_score`. Each outer fold re-runs the entire inner grid search on its own training portion. The `flat_score` is `best_score_` — the maximum over the grid, the order statistic, the optimistic number. The `nested_scores.mean()` is the honest one. **The difference between them is a direct, quantitative measurement of your selection optimism.** If that gap is 0.05, then any flat-CV number you've ever reported on this setup was about five points too high.

#### Worked example: a 0.84 flat CV is really a 0.79

On the configuration above, flat CV on a 400-row problem with a 20-point SVM grid typically reports about 0.84, while nested CV reports about 0.79 with a fold standard deviation around 0.03. The 0.05 gap is the optimism of having looked at the grid. It's not a bug in any one fold — every fold's splitter was stratified and clean — it's the unavoidable upward bias of reporting the best of many tries. Here's how the two numbers line up against what a genuine forward holdout would show:

| Reporting method | What it measures | ROC-AUC | Honest? |
| --- | --- | --- | --- |
| Flat CV `best_score_` | best of 20 tried configs (an order statistic) | 0.84 | No |
| Single tuned-then-holdout | one fixed model on one fresh split | 0.80 ± high var | Partly |
| Nested CV (outer mean) | the full tune-then-fit procedure | 0.79 ± 0.03 | Yes |
| Production | the deployed procedure on new data | 0.79 | Ground truth |

A subtle point worth stating plainly: nested CV does **not** give you a single "best" hyperparameter set to ship — it gives you an honest *estimate of the procedure*. Different outer folds may select different hyperparameters, and that's fine; it tells you how stable your selection is. To ship, you run the inner search one last time on *all* your data and deploy that, but the number you *promise stakeholders* is the nested-CV number, not the flat one.

### Why nested CV is unbiased, in one paragraph

The proof sketch is short and worth carrying around. The outer test fold $T$ is held out before the inner loop runs. The inner loop computes the best hyperparameters $\theta^*$ using *only* the outer-train rows; $T$ contributes nothing to that choice. Then the model is refit with $\theta^*$ on outer-train and scored on $T$. Because $\theta^*$ is a deterministic function of data that excludes $T$, the score on $T$ is an unbiased estimate of "run this exact selection-and-fit procedure on a training set of this size, then evaluate on fresh data." Averaging over outer folds reduces the variance of that estimate. There is no $\max$ over the outer scores anywhere — we report their *mean*, not their best — so the order-statistic inflation that poisons flat CV simply never enters. That single structural difference (mean of held-out scores vs. max of selected scores) is the whole reason nested CV is honest and flat CV is not.

### When you can skip nesting

Nested CV is expensive — it multiplies your fit count by the outer fold count — so it's fair to ask when you can skip it. Two cases. First, if your hyperparameter grid is *tiny* (a couple of values) and your dataset is *large*, the selection noise $\sigma$ is small and $M$ is small, so $\sigma\sqrt{2\ln M}$ is negligible; flat CV is then approximately honest and the nested loop isn't worth the cost. Second, if you have enough data to carve a genuine, never-touched forward holdout, you can tune freely with flat CV and report the *holdout* number — the holdout plays the role of the outer loop with a single fold. The danger in the second approach is discipline: the moment you peek at the holdout, tweak, and re-check, you've turned it into a tuning set and you're back to selection optimism, this time spread silently across your whole project. Nested CV is the bulletproof option precisely because it can't be peeked at.

## 5. Preprocessing fit outside the fold: the leak that hides in `fit`

This one is so common it deserves its own section even though it's a cousin of the leaks in the [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage) post. The mistake: you fit a transformer — a `StandardScaler`, a `SimpleImputer`, a target encoder, a PCA, a feature selector — on the *entire* dataset once, *before* you cross-validate. Now every fold's "training" was secretly informed by statistics computed from its own validation rows. The scaler's mean and standard deviation, the imputer's median, the PCA's components, the selected feature set — all of them saw the validation data. The leak is small for a `StandardScaler` (a couple hundredths of AUC) and catastrophic for a supervised step like a target encoder or a feature selector that uses the label (can be 0.1–0.3 AUC).

The mechanism is the same information-leak argument: the validation fold contributed to a quantity the model relies on, so the validation fold is no longer a fair stand-in for unseen data. The reason it's especially insidious is that the code reads as totally normal — `scaler.fit_transform(X)` at the top of the cell looks like good hygiene, not a bug.

### The fix: put everything in a `Pipeline` and let CV refit it per fold

`scikit-learn`'s `Pipeline` exists precisely to make this correct by construction. When you pass a `Pipeline` to `cross_val_score`, every transformer is *refit on each fold's training rows only* and then *applied* to that fold's validation rows. The validation fold never touches a `fit`. This is the single most important habit in `sklearn`: **fit nothing outside the cross-validation loop.**

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=2000, n_informative=5,
                           random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# WRONG: select 20 best features using the WHOLE dataset (label included),
# then cross-validate. The selector already saw every validation label.
sel = SelectKBest(f_classif, k=20).fit(X, y)
X_leaked = sel.transform(X)
wrong = cross_val_score(LogisticRegression(max_iter=1000),
                        X_leaked, y, cv=cv, scoring="roc_auc").mean()

# RIGHT: selection lives INSIDE the pipeline, refit per fold.
pipe = Pipeline([
    ("select", SelectKBest(f_classif, k=20)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
right = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

print(f"Feature selection OUTSIDE CV (leak): AUC {wrong:.3f}")
print(f"Feature selection INSIDE  CV (ok):   AUC {right:.3f}")
```

On 2000 noise-heavy features with only 5 informative ones, the leaked version can post an AUC around 0.90+ while the honest pipeline reports near 0.5–0.6 — because selecting "the 20 features most correlated with the label across all rows" on a 300-row dataset is mostly selecting *noise that happens to correlate with the validation labels too*. This is the famous leak that produced impossibly good gene-expression classifiers in early bioinformatics papers, where thousands of genes were screened against a few dozen samples before any cross-validation. The screen *was* the model, and it had memorized the labels.

### The leak severity ladder: not all out-of-fold fits are equal

It helps to rank preprocessing steps by how badly fitting them outside the fold leaks, because in a time crunch you fix the worst offenders first:

- **Catastrophic (0.1–0.4 AUC):** any *supervised* step that uses the label — feature selection by correlation/`f_classif`, target/mean encoding, supervised dimensionality reduction, SMOTE and other label-aware resampling. These directly memorize validation labels and should *never* exist outside the fold.
- **Serious (0.02–0.1 AUC):** unsupervised steps with strong global structure — PCA, learned embeddings, clustering used as features. They don't see labels, but they fit to the full feature distribution including validation rows, and on small data that's a real leak.
- **Mild (0.001–0.02 AUC):** simple per-feature statistics — `StandardScaler` mean/std, `SimpleImputer` median, min-max scaling. The leak is tiny because the statistic is stable and the validation rows barely move it, but it's still nonzero and there's no reason to incur it.

The practical rule collapses the ladder to one habit: put *everything* learned from data inside the `Pipeline`, no exceptions, so you never have to remember which tier a step is in. The cost of being disciplined is zero; the cost of guessing wrong is a model that fails in production. A particularly nasty member of the catastrophic tier is **resampling for imbalance applied before the split** — calling SMOTE or random oversampling on the whole dataset duplicates or synthesizes minority points that then land in *both* train and validation, so the model is validated on near-copies of its training data. Use `imbalanced-learn`'s `Pipeline` (which is fold-aware) and resample *inside* the fold, never before it.

```python
# Correct: resample INSIDE the fold using imbalanced-learn's Pipeline.
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

imb_pipe = ImbPipeline([
    ("smote", SMOTE(random_state=0)),          # refit per fold, only on train
    ("clf", LogisticRegression(max_iter=1000)),
])
# cross_val_score(imb_pipe, X, y, cv=cv) now resamples train-only each fold.
```

## 6. Missing stratification: high-variance folds and a biased mean

Stratification is the cheapest insurance in CV and the one most often skipped. For imbalanced or multiclass problems, plain `KFold` draws folds whose class proportions wander. With 5% positives and 5 folds on a small dataset, one fold might land at 2% positive and another at 9%. Two bad things follow. First, the *variance* of your per-fold scores explodes, so your CV mean has a fat confidence interval and you can't tell a real 1-point improvement from noise. Second, for metrics whose value depends on base rate — precision, F1, and to a lesser extent ROC-AUC at the extremes — the *mean* across imbalanced folds can be biased relative to the true population value, because the metric is a nonlinear function of the class mix.

`StratifiedKFold` fixes this by preserving the global class proportions in every fold. It is the correct default for any classification problem and especially for imbalanced ones. The diagnostic is trivial: print the per-fold positive rate and look at the spread.

```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

rng = np.random.default_rng(0)
n = 400
y = (rng.uniform(size=n) < 0.05).astype(int)   # 5% positive
X = rng.normal(size=(n, 10))

for name, cv in [("KFold", KFold(5, shuffle=True, random_state=0)),
                 ("StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=0))]:
    rates = [y[va].mean() for _, va in cv.split(X, y)]
    print(f"{name:16s} per-fold pos-rate: "
          f"{[f'{r:.3f}' for r in rates]}  spread {max(rates)-min(rates):.3f}")
```

You'll see plain `KFold` produce a spread of several percentage points and `StratifiedKFold` cluster tightly around 0.05. On a rare-event problem this is the difference between a usable estimate and a useless one. Note the composition rule: when you *also* have groups, you need `StratifiedGroupKFold` — stratification and grouping are orthogonal requirements and most real datasets need both. A related sin is **too few folds**: with 3 folds on a small dataset each training set is only 67% of the data, so the model is systematically weaker than the one you'll ship on 100%, which biases CV *pessimistic* (the rare downward bias) while also inflating variance. Five to ten folds is the standard sweet spot; leave-one-out has the lowest bias but punishing variance and cost.

The quantitative reason imbalance demands stratification is a small-counts argument. With a 5% positive rate and a 5-fold split of 400 rows, each validation fold has 80 rows and *expects* 4 positives — but the actual count follows a binomial with that small mean, so a fold landing with 1 or 7 positives is entirely ordinary. A metric like precision or recall computed on 1–7 positives is wildly noisy, and averaging five such noisy folds doesn't rescue you because the underlying counts are too small. Worse, for a base-rate-sensitive metric the *average over imbalanced folds* is not the same as the metric on the pooled population, because the metric is nonlinear in the class mix — so you get both high variance *and* a biased mean. Stratification fixes the counts so every fold sees the expected 4 positives, which is the most you can do short of getting more positive examples. When positives are *truly* scarce — single digits total — even stratified $k$-fold is shaky, and you should prefer repeated stratified CV (average several different stratified splits) or report a confidence interval rather than a point estimate.

## 7. The shuffle-label sanity check: does your CV know it's lying?

Here is a diagnostic that catches an entire family of leaks at once, including ones you didn't think to look for. The idea, formalized by Ojala and Garriga in their 2010 *permutation tests for studying classifier performance* work, is dead simple: **destroy the relationship between features and labels by shuffling the labels, then run your exact CV pipeline.** If your CV is honest, an honest model on random labels can do no better than chance — AUC 0.5, accuracy at the majority-class rate. If your "CV" reports meaningfully above chance on *shuffled labels*, you have a leak so severe the model is scoring on information that has nothing to do with the (now-random) target. The most common culprit it catches is a transform fit outside the fold, or an index/ID feature that aligns with the split.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

def shuffle_label_check(make_cv_score, y, n_repeats=20, seed=0):
    """Run the SAME cv-scoring callable on shuffled labels.
    Honest CV should collapse to ~chance (0.5 AUC)."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_repeats):
        y_perm = rng.permutation(y)
        scores.append(make_cv_score(y_perm))
    scores = np.array(scores)
    print(f"shuffled-label AUC: {scores.mean():.3f} +/- {scores.std():.3f} "
          f"(honest target ~0.500)")
    return scores

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 20))
y = (X[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(int)

cv = StratifiedKFold(5, shuffle=True, random_state=0)
pipe = Pipeline([("scale", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=1000))])

# Honest pipeline -> shuffled labels collapse to ~0.5
shuffle_label_check(
    lambda yy: cross_val_score(pipe, X, yy, cv=cv, scoring="roc_auc").mean(), y)
```

An honest pipeline returns about `0.50 ± 0.03`. If instead you scaled outside the loop, or you left an ID column in `X` that happens to correlate with the fold assignment, the shuffled-label AUC will sit at 0.6, 0.7, or higher — a screaming alarm that your CV is measuring leakage, not learning. Run this check once at the start of any new modeling effort. It costs a minute and it has saved me from shipping at least three "great" models that were memorizing leaks.

## 8. The CV-versus-holdout-versus-production gap, and how to close it

Step back and assemble the full diagnostic. You have three numbers available, in increasing order of honesty and cost:

1. **CV score** — cheap, low-variance, but only as honest as your splitter and your pipeline.
2. **Forward holdout** — a single block of data, drawn *the way deployment draws it* (later in time, new entities), that you touch *exactly once*, after all tuning is frozen. Higher variance than CV but immune to selection optimism if you truly touch it once.
3. **Production / live performance** — the ground truth, available only after you ship.

When all three agree within noise, your evaluation is trustworthy and you can iterate fast on the CV number. When they diverge, the *pattern* of divergence localizes the bug, and this is where bisection pays off:

![A matrix mapping each cross-validation error to the mechanism that biases the estimate upward, the detector that catches it, and the fix](/imgs/blogs/cross-validation-done-wrong-4.png)

- **CV high, holdout high, production low.** The holdout shares the leak. Almost always grouping or time: your holdout was drawn by random sampling, so the same entities/periods straddle it too. Redraw the holdout *the way production sees data* — split by entity, or split forward in time — and the holdout will drop to meet production.
- **CV high, holdout low.** Selection optimism. You tuned on the CV folds and the holdout is the first thing the search never saw. The gap is your flat-vs-nested optimism; confirm it with nested CV.
- **CV high, holdout high, production high, then production decays over weeks.** Not a CV bug at all — that's [distribution shift](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world). Don't go hunting leaks; go look at how the input distribution moved.

That last bullet is the discipline this whole series preaches: **the symptom pattern tells you which of the six places to look.** A CV-vs-prod gap that's there from day one is an evaluation leak. A gap that *opens up over time* is a data/distribution story. They feel identical on the dashboard and demand opposite fixes.

#### Worked example: bisecting a 0.94→0.79 mystery in four tests

Here's the readmission model from the intro, debugged the way you actually should. The symptom: CV 0.94, production 0.79, gap unexplained. Run the cheap discriminating tests in order, each one ruling a suspect in or out.

```python
# TEST 1 (1 min): shuffle-label sanity check on the current pipeline.
#   Result: shuffled AUC 0.50 -> pipeline itself isn't leaking via fit/ID.
#   Rules OUT preprocessing-outside-fold and an ID feature.

# TEST 2 (1 min): group-overlap audit with the patient_id column.
#   Result: ~160 patients straddle every fold.  <-- SMOKING GUN
#   Rules IN group leakage.

# TEST 3 (2 min): swap KFold -> GroupKFold, re-score.
#   Result: CV AUC 0.94 -> 0.79.  Matches production exactly.
#   CONFIRMS group leakage as THE cause.

# TEST 4 (2 min): nested CV on the GroupKFold setup to check residual
#   selection optimism.  flat 0.80 vs nested 0.79 -> 0.01 gap, negligible.
#   Rules OUT meaningful tuning optimism.
```

Total time: about six minutes to go from "the model degraded mysteriously" to "the model never degraded; our CV was leaking patient identity, here is the honest 0.79, and it matches production." No retraining marathon, no guessing. That is the entire point of the bisection method — you don't fix the most-likely cause first, you run the *cheapest discriminating test* first and let the result route you.

![A decision tree that routes a too-good cross-validation score through questions about entities, time, and tuning to the splitter that fixes it](/imgs/blogs/cross-validation-done-wrong-8.png)

## 9. Putting the splitters together: the decision procedure

You now have the full toolkit. The decision procedure for *which* CV to use is short and worth committing to muscle memory, because choosing the splitter is 90% of getting CV right:

- **Do rows belong to repeated entities** (patient, user, device, document, image source)? → Split by group. `GroupKFold`, or `StratifiedGroupKFold` if also imbalanced.
- **Are rows ordered in time** and will you predict the future? → Forward-chain. `TimeSeriesSplit`, plus purge+embargo if labels span a horizon.
- **Both grouped and temporal?** (e.g., per-user event streams) → Split by group *and* respect time within the split; combine a group-aware split with a time cutoff, or use a custom splitter that holds out whole groups from a later period.
- **Imbalanced or multiclass?** → Stratify. `StratifiedKFold` (or `StratifiedGroupKFold`).
- **Tuning hyperparameters and reporting a number?** → Nest. Inner loop tunes, outer loop scores.
- **Any preprocessing that learns from data** (scale, impute, encode, select, reduce)? → Put it in a `Pipeline` so it refits per fold.
- **None of the above** (plain i.i.d. tabular, no tuning)? → `StratifiedKFold` with 5–10 folds, shuffle on, fixed seed. The default that's actually correct.

Notice these compose: a churn model on per-customer monthly snapshots with class imbalance and a hyperparameter search needs *all* of grouped + temporal + stratified + nested, plus a `Pipeline`. Each requirement you skip is a separate channel for optimism. The reason CV-done-wrong is so pervasive is that the *default* `cross_val_score(model, X, y)` is correct for exactly one of these cases and silently wrong for the rest.

#### Worked example: a churn model that needs grouped *and* temporal splits

The hardest real case is when leaks compound. A subscription company predicts customer churn from monthly snapshots: each customer contributes one row per month, the target is "churned within 30 days," classes are imbalanced (~6% churn), and the team tuned a gradient-boosted model with a 40-point random search. They reported a cross-validated PR-AUC of 0.61 with stratified 5-fold. Live PR-AUC came in at 0.34. Three separate leaks stacked up.

The first leak is **group**: a customer who appears in month 3 (train) and month 4 (validation) lets the model recognize that specific customer's billing pattern — pure entity memorization. The second is **time**: random folds mix future months into training, so the model learns from churn events that hadn't happened yet at prediction time. The third is **selection**: the 0.61 was the best of 40 tried configs, an order statistic. Here's the honest splitter — held-out *future months* of *unseen customers*, evaluated with nested CV:

```python
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, cross_val_score

def grouped_forward_folds(months, customers, n_splits=4, embargo_months=1):
    """Honest churn CV: each fold trains on EARLIER months and validates on a
    LATER month, using only customers NOT seen in that fold's training window.
    Combines a time cutoff with entity disjointness."""
    uniq_months = np.sort(np.unique(months))
    for k in range(1, n_splits + 1):
        cutoff = uniq_months[k * (len(uniq_months) // (n_splits + 1))]
        val_month = cutoff + embargo_months
        train_mask = months < cutoff
        val_mask = months == val_month
        # enforce entity disjointness: drop val customers seen in train
        seen = set(customers[train_mask])
        val_mask &= ~np.isin(customers, list(seen))
        tr, va = np.where(train_mask)[0], np.where(val_mask)[0]
        if len(va):
            yield tr, va
```

When the team swapped to this splitter and wrapped the model in a nested `GridSearchCV`, the reported PR-AUC dropped from 0.61 to 0.35 — within noise of the live 0.34. The 0.27-point gap was the sum of three leaks: roughly 0.15 from grouping, 0.09 from time, and 0.03 from selection optimism, each measurable by toggling one fix at a time. That last technique — **turn the fixes on one at a time and watch the score fall in stages** — is the most convincing way to attribute a CV-vs-prod gap to its specific causes.

| Splitter (cumulative fixes) | PR-AUC | Drop attributed to |
| --- | --- | --- |
| Stratified KFold (the original) | 0.61 | — |
| + group-disjoint customers | 0.46 | group leak (−0.15) |
| + time-forward folds | 0.37 | time leak (−0.09) |
| + nested CV | 0.35 | selection optimism (−0.02) |
| Production | 0.34 | ground truth |

### The composition problem: grouped *and* temporal at once

The case that trips up even experienced practitioners is data that is *both* grouped and time-ordered — per-user event streams, per-device telemetry, per-patient longitudinal records. Neither `GroupKFold` nor `TimeSeriesSplit` alone is correct: `GroupKFold` ignores time (it can put a user's future in train and past in val), and `TimeSeriesSplit` ignores entities (it can put the same user on both sides at the fold boundary). You need *both* constraints simultaneously: validation entities must be (a) disjoint from training entities and (b) drawn from a later time period than training.

`scikit-learn` doesn't ship a single splitter for this, so you compose one — as in the churn worked example above — by choosing a time cutoff per fold and then dropping any validation entity that also appeared in the training window. The general recipe: pick the time axis as the primary split (forward chaining), then within each fold enforce entity disjointness by removing straddlers from validation. Always run *both* the group-overlap audit and the temporal-coverage audit on your custom splitter, because a hand-rolled splitter is exactly where a subtle bug hides. The two audits together are your proof that the composed splitter honors both constraints — if both return clean, your fold structure mirrors the only thing that matters: a new entity, in the future.

### A subtle trap: the same seed across the whole search

One more failure mode that masquerades as good practice. If you fix `random_state` on your splitter and then run a long hyperparameter search, every configuration is evaluated on the *exact same folds*. That's good for *comparing* configs fairly, but it means the noise $\eta_m$ in each config's score is correlated through the shared fold assignment, and your selected config is partly selected for fitting *this particular fold draw*. The honest accounting still comes from nested CV (the outer loop redraws), so this isn't a reason to randomize per config — it's a reason to never trust the flat best score and always confirm with nesting. Fix the seed for reproducibility, report the nested number for honesty.

## 10. Case studies and real signatures

These are well-documented patterns; where I cite a number it's from the source, and where I generalize I say so.

**Confident learning finds label errors in test sets.** Northcutt, Athalye, and Mueller (2021) audited the *test sets* of ten canonical benchmarks and estimated an average of about 3.4% label errors, with ImageNet's validation set around 5.8% and QuickDraw higher. The relevance to CV: if your *evaluation* fold has mislabeled rows, your CV score is measuring against a noisy target, and "improvements" can be fitting the label noise rather than the task. It's a reminder that even a perfectly-drawn fold can lie if the labels in it are wrong — the same `cleanlab` confident-learning machinery that finds training-set noise should audit your validation folds too. (See the sibling post on finding label noise for the mechanics.)

**The Kaggle leakage canon.** Competition post-mortems are a museum of CV-done-wrong. The recurring pattern: a leaderboard built on a random split that the private test set later breaks, because the public split shared an entity or a temporal block with training. Teams with airtight *grouped* or *time-based* local CV climb on the private leaderboard while teams who trusted the random public score collapse. The lesson the Kaggle community distilled — "trust your local CV, not the public leaderboard, *if and only if your local CV matches the test split's structure*" — is exactly this post's thesis in competition form. The splitter must mirror how the held-out data was actually carved.

**Financial backtests and purged K-fold.** López de Prado's *Advances in Financial Machine Learning* (2018) documents how standard CV systematically overstates strategy performance on serially-correlated financial data, and introduces purged-and-embargoed K-fold as the remedy. The signature is unmistakable: a backtest Sharpe that looks tradeable, a live Sharpe near zero, and the gap closing exactly when you purge the label-overlap and embargo the boundary. It's the time-leak section of this post, with money on the line.

**The bioinformatics feature-selection scandal.** Ambroise and McLachlan (2002) showed that selecting genes on the *full* dataset before cross-validating produced near-zero apparent error rates on datasets where the honest error was substantial — the feature screen had memorized the labels of a few dozen samples across thousands of genes. This is the "preprocessing fit outside the fold" leak (Section 5) at its most dramatic, and it's why "fit nothing outside the CV loop" is non-negotiable in any high-dimensional, low-sample regime. The terrifying part is how reasonable the leaky code looked: screen for predictive genes, then validate the classifier — a workflow that reads like good science and is in fact a label memorization machine.

**Medical imaging and the patient-level split.** A recurring finding in radiology and pathology ML is that models cross-validated at the *image* level dramatically overstate performance versus models validated at the *patient* level, because multiple slices or tiles from the same patient share anatomy, scanner artifacts, and staining characteristics that the model learns as a fingerprint. Studies that report image-level AUCs in the high 0.9s often collapse to the 0.7–0.8 range under patient-level (group) splits — exactly the readmission story, with pixels. This is now a standard reviewer check in medical-imaging venues: "did you split by patient?" If the answer is no, the headline number is presumed inflated until proven otherwise. It's the clearest institutional acknowledgment that group leakage is the default failure mode, not an edge case.

| Case | The leak channel | Symptom | The fix |
| --- | --- | --- | --- |
| Patient readmission | group (entity in 2 folds) | CV 0.94, prod 0.79 | `GroupKFold` |
| Demand / financial backtest | time (future in train) | CV R2 0.71, live 0.22 | `TimeSeriesSplit` + purge/embargo |
| Gene-expression classifier | preprocessing fit on all data | CV error ~0, true error high | selection inside `Pipeline` |
| Kaggle public-LB collapse | mismatched split structure | great public, bad private | local CV mirrors test split |
| Hyperparameter sweep | selection optimism | flat 0.84, holdout 0.79 | nested CV |

## 11. When this is (and isn't) your bug

A decisive section, because misdiagnosis wastes days.

**It IS a CV bug when:** the gap between CV and production is present from day one and stable; your data has obvious grouping (multiple rows per entity) or time order that your splitter ignores; the shuffle-label check returns above chance; or the flat-vs-nested gap is large. These all point to the evaluation corner, and the fix is a different splitter or a `Pipeline`, not a different model.

**It is NOT a CV bug when:** the model was fine for weeks and *then* degraded — that's [distribution shift](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world), a data story, and no splitter change will help. It's also not a CV bug when the model can't even *overfit a single batch* or your training loss won't descend — that's a model-code or optimization problem upstream of evaluation; a correct CV on a broken model just honestly reports that it's broken. And it's not *primarily* a CV bug when your *metric itself* is wrong (micro-vs-macro confusion, scoring on a base-rate-inflated number) — fix the metric first, then the splitter, because a leaking splitter and a lying metric can cancel out and mislead you twice. See [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying) for that adjacent failure mode.

The clean rule: **a stable CV-vs-prod gap is a leak; a widening one is drift; no gap with bad absolute numbers is the model.** Use the gap's *shape over time* to route yourself to the right corner before you touch the splitter. And if you've fixed every leak in this post and CV still beats a true forward holdout, suspect the most under-appreciated leak of all — [overfitting to the validation set](/blog/machine-learning/debugging-training/overfitting-to-the-validation-set) through dozens of iterations of "tweak and re-check CV," which is selection optimism spread across your whole project rather than one grid search.

One more boundary worth drawing, because it's a frequent misattribution: a *high-variance* CV (per-fold scores swinging by 0.1) is not the same problem as a *biased* CV (the mean is too high). Variance is a precision problem you fix with more folds, stratification, or repeated CV; bias is an honesty problem you fix with the right splitter. People often add folds hoping to "stabilize" a number that's actually leaking, and end up with a more *precise* estimate of the *wrong* quantity. Check honesty first (the audits), precision second (the fold count). If your shuffle-label check returns chance and your group/time audits are clean, *then* a wide fold spread is genuinely a precision issue and more folds or repeated CV is the right move. Diagnose in that order and you won't waste a day tightening the variance on a leaked mean.

## 12. Building CV that can't lie, from day one

The best debugging is the kind you never have to do. A few habits make leaky CV nearly impossible:

First, **define your unit of generalization before you write a single split.** Write it down: "the model must work on a *new patient*," or "a *future week*," or "an *unseen merchant*." That sentence dictates your splitter. If you can't say it, you're not ready to cross-validate.

Second, **make the splitter a parameter of your harness, not a buried default.** Pass `cv=` explicitly everywhere; never call bare `cross_val_score(model, X, y)`. A reviewer should be able to see, in one place, whether you grouped and stratified and respected time.

Third, **wire the four audits into CI**: a group-overlap audit (must return 0 straddling groups), a temporal-coverage check (train must precede val), a shuffle-label check (must collapse to chance), and a flat-vs-nested gap report (must be small or explained). Run them on every modeling PR. They're each a dozen lines and they catch the entire taxonomy of this post automatically.

Fourth, **keep a true forward holdout you touch once.** Carve it the way production carves data — later in time, disjoint entities — freeze it before any modeling, and look at it exactly once at the very end. When it agrees with your nested CV, you can trust the number you're about to promise.

Here's the four-audit harness condensed into one function you can drop into a test file. It takes your splitter, your data, and (optionally) groups and times, and it returns a verdict on whether your CV is trustworthy.

```python
import numpy as np
from sklearn.model_selection import cross_val_score

def audit_cv(splitter, pipe, X, y, groups=None, times=None, scoring="roc_auc"):
    """Four guardrails for an honest CV. Returns a dict of pass/fail."""
    report = {}

    # 1. Group-overlap audit (only if groups given): must be 0.
    if groups is not None:
        straddle = sum(len(set(groups[tr]) & set(groups[va]))
                       for tr, va in splitter.split(X, y, groups))
        report["group_straddle"] = straddle
        report["group_ok"] = (straddle == 0)

    # 2. Temporal coverage (only if times given): train must precede val.
    if times is not None:
        leaks = sum(times[tr].max() > times[va].min()
                    for tr, va in splitter.split(X, y, groups))
        report["time_leaks"] = leaks
        report["time_ok"] = (leaks == 0)

    # 3. Shuffle-label check: must collapse to ~chance.
    rng = np.random.default_rng(0)
    perm = [cross_val_score(pipe, X, rng.permutation(y), cv=splitter,
                            groups=groups, scoring=scoring).mean()
            for _ in range(10)]
    report["shuffled_score"] = float(np.mean(perm))
    report["shuffle_ok"] = abs(np.mean(perm) - 0.5) < 0.05

    return report
```

Wire this into CI so a pull request that introduces a leak — adds a feature that aligns with the split, switches to a row-level splitter on grouped data, drops the `Pipeline` wrapper — fails the build with a specific message instead of shipping a 0.94 that becomes a 0.79 incident. These habits cost an afternoon to set up and they convert "the model mysteriously degraded in production" — a multi-day fire drill — into "the CI shuffle-label check failed on the PR," a five-minute fix. That trade is always worth making.

Finally, treat the *honest CV number* as a contract, not a high score. When you write it in a model card or a launch doc, write next to it which splitter produced it and which audits passed: "nested PR-AUC 0.35, grouped by customer, forward-chained by month, shuffle-label check at 0.50." That one line tells the next engineer — possibly you in six months — exactly what the number means and what it doesn't. A bare "PR-AUC 0.61" is worse than useless because it invites trust it hasn't earned. The whole arc of getting CV right is moving from a number that *sounds* good to a number that *is* true, and the way you prove it's true is by stating the splitter and showing the audits. The optimistic estimate is seductive precisely because it's higher; the honest estimate is valuable precisely because it survives contact with production.

## Key takeaways

- **CV measures the generalization gap of a procedure, and every CV mistake shrinks the *measured* gap without shrinking the *real* one.** That's why leaks bias upward and why production never moves when you "fix" the metric.
- **Rows are not your unit of generalization.** If entities repeat, use `GroupKFold`/`StratifiedGroupKFold` and audit that zero groups straddle a fold. Plain `KFold` on grouped data learns the entity, not the pattern.
- **Random shuffling on time-ordered data trains on the future.** Use `TimeSeriesSplit` for forward chaining, and purge label-window overlap with an embargo when labels span a horizon.
- **The score you report must come from a fold you did not select on.** Nested CV (inner tunes, outer scores) is unbiased; the flat-vs-nested gap is a direct measurement of your selection optimism.
- **Fit nothing outside the CV loop.** Put every learned transform — scale, impute, encode, select, reduce — inside a `Pipeline` so it refits per fold. The worst leaks are supervised steps fit on all the data.
- **Stratify imbalanced and multiclass problems** to cut fold variance and de-bias base-rate-sensitive metrics; use 5–10 folds, not 3.
- **The shuffle-label check is your one-minute leak alarm:** honest CV collapses to chance on permuted labels; if it doesn't, you're scoring a leak.
- **A stable CV-vs-prod gap is a leak; a widening one is drift; bad absolute numbers with no gap is the model.** The shape of the gap over time tells you which of the six places to debug.

## Further reading

- Ojala, M. & Garriga, G. (2010). *Permutation Tests for Studying Classifier Performance.* Journal of Machine Learning Research — the formal basis for the shuffle-label sanity check.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Wiley — purged and embargoed K-fold cross-validation for serially-correlated data.
- Northcutt, C., Athalye, A. & Mueller, J. (2021). *Pervasive Label Errors in Test Sets Destabilize ML Benchmarks.* NeurIPS Datasets and Benchmarks — why even a well-drawn evaluation fold can lie if its labels are wrong.
- Ambroise, C. & McLachlan, G. (2002). *Selection bias in gene extraction on the basis of microarray gene-expression data.* PNAS — the canonical "feature selection outside the fold" leak.
- scikit-learn documentation, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — `GroupKFold`, `StratifiedGroupKFold`, `TimeSeriesSplit`, and nested CV with worked examples.
- Within this series: the master decision tree in [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), the row-level mechanics in [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage) and [data leakage the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), and the full checklist in [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

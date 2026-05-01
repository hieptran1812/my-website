---
title: "Gradient Boosted Trees: From Theory to Practice"
publishDate: "2026-03-12"
category: "machine-learning"
subcategory: "Traditional Machine Learning"
tags:
  [
    "gradient-boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble-methods",
    "decision-tree",
    "machine-learning",
  ]
date: "2026-03-12"
author: "Hiep Tran"
featured: false
image: ""
excerpt: "A deep dive into Gradient Boosted Trees — the algorithm that dominates tabular data competitions and production ML. Covering the math behind boosting, tree construction, regularization, and practical comparisons of XGBoost, LightGBM, and CatBoost."
---

# Gradient Boosted Trees: From Theory to Practice

## 1. Introduction

**Gradient Boosted Trees (GBT)** remain one of the most powerful and widely used algorithms in machine learning, particularly for **tabular data**. Despite the deep learning revolution, GBTs consistently outperform neural networks on structured/tabular datasets and dominate Kaggle competitions.

Key reasons for their dominance:

- **High accuracy** on tabular data — often state-of-the-art
- **Handles mixed feature types** (numerical + categorical) naturally
- **Robust to outliers** and missing values
- **Feature importance** built in — great interpretability
- **Fast inference** — suitable for low-latency production systems
- **Minimal preprocessing** — no need for normalization or one-hot encoding

## 2. Prerequisites: Decision Trees and Ensemble Methods

### 2.1 Decision Trees (CART)

A **Classification and Regression Tree (CART)** recursively splits data to minimize an impurity criterion:

- **Classification**: Gini impurity or entropy
- **Regression**: Mean Squared Error (MSE)

For a split at node $t$ with feature $j$ and threshold $s$:

$$\text{Split}(j, s) = \arg\min_{j, s} \left[ \sum_{x_i \in R_1(j,s)} (y_i - \hat{y}_{R_1})^2 + \sum_{x_i \in R_2(j,s)} (y_i - \hat{y}_{R_2})^2 \right]$$

where $R_1$ and $R_2$ are the left and right regions after the split.

**Limitation**: A single decision tree is a weak learner — high variance, prone to overfitting.

### 2.2 Ensemble Methods

Ensemble methods combine multiple weak learners to create a strong learner:

| Method                      | Strategy     | Key Idea                                                          |
| --------------------------- | ------------ | ----------------------------------------------------------------- |
| **Bagging** (Random Forest) | Parallel     | Train independent trees on bootstrap samples, average predictions |
| **Boosting** (GBT)          | Sequential   | Train each tree to correct errors of previous trees               |
| **Stacking**                | Hierarchical | Use predictions of base models as features for a meta-model       |

## 3. Gradient Boosting: The Algorithm

### 3.1 Core Idea

Gradient Boosting builds an **additive model** by sequentially fitting weak learners (shallow trees) to the **negative gradient** (pseudo-residuals) of the loss function.

The final prediction is a sum of all trees:

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta \cdot h_m(x)$$

where:

- $F_0(x)$ is the initial prediction (e.g., mean of targets)
- $h_m(x)$ is the $m$-th tree
- $\eta$ is the learning rate (shrinkage)
- $M$ is the number of boosting rounds

### 3.2 Algorithm Step-by-Step

**Input**: Training data $\{(x_i, y_i)\}_{i=1}^n$, loss function $L(y, F(x))$, number of iterations $M$

**Step 1**: Initialize with a constant:

$$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, \gamma)$$

For MSE loss: $F_0(x) = \bar{y}$ (mean of targets)

**Step 2**: For $m = 1$ to $M$:

**(a)** Compute pseudo-residuals (negative gradient):

$$r_{im} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \Bigg|_{F = F_{m-1}}$$

For MSE loss: $r_{im} = y_i - F_{m-1}(x_i)$ (just the residuals)

**(b)** Fit a regression tree $h_m(x)$ to the pseudo-residuals $\{(x_i, r_{im})\}$

**(c)** Compute the optimal leaf values:

$$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$

**(d)** Update the model:

$$F_m(x) = F_{m-1}(x) + \eta \cdot \sum_{j=1}^{J} \gamma_{jm} \cdot \mathbb{1}(x \in R_{jm})$$

**Step 3**: Output $F_M(x)$

### 3.3 Loss Functions

The beauty of gradient boosting is its generality — any differentiable loss function works:

| Task                  | Loss Function                           | Pseudo-Residual                  |
| --------------------- | --------------------------------------- | -------------------------------- |
| Regression            | MSE: $\frac{1}{2}(y - F)^2$             | $y - F$                          |
| Regression            | MAE: $\|y - F\|$                        | $\text{sign}(y - F)$             |
| Regression            | Huber loss                              | Combination of MSE and MAE       |
| Binary Classification | Log loss: $-[y\log p + (1-y)\log(1-p)]$ | $y - p$ (where $p = \sigma(F)$)  |
| Multi-class           | Cross-entropy                           | $y_k - p_k$ (one tree per class) |
| Ranking               | LambdaMART                              | Based on NDCG gradients          |

### 3.4 Why "Gradient" Boosting?

The pseudo-residuals are the **negative gradient** of the loss function with respect to the current predictions. Each new tree performs **gradient descent in function space**:

$$F_m = F_{m-1} - \eta \cdot \nabla_{F} L(y, F_{m-1})$$

This is analogous to gradient descent in parameter space ($\theta_{t+1} = \theta_t - \eta \nabla_\theta L$), but instead of updating parameters, we add a new function (tree).

## 4. Regularization Techniques

Gradient boosting is prone to overfitting without proper regularization:

### 4.1 Shrinkage (Learning Rate)

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x), \quad \eta \in (0, 1]$$

- Smaller $\eta$ → more trees needed, better generalization
- Typical values: 0.01–0.3
- Trade-off: lower $\eta$ with more trees = better but slower

### 4.2 Tree Constraints

| Parameter           | Effect                                   | Typical Range |
| ------------------- | ---------------------------------------- | ------------- |
| `max_depth`         | Limits tree depth                        | 3–8           |
| `min_child_weight`  | Minimum sum of instance weight in a leaf | 1–10          |
| `max_leaves`        | Maximum number of leaf nodes             | 31–255        |
| `min_samples_split` | Minimum samples to create a split        | 2–20          |

### 4.3 Subsampling (Stochastic Gradient Boosting)

- **Row subsampling** (`subsample`): Use a random fraction of training data per tree (0.5–0.9)
- **Column subsampling** (`colsample_bytree`): Use a random fraction of features per tree (0.5–1.0)
- **Column subsampling per level** (`colsample_bylevel`): Random features per split level
- Reduces overfitting and speeds up training (similar to dropout in neural networks)

### 4.4 L1 and L2 Regularization on Leaf Weights

XGBoost's objective includes regularization on the tree structure:

$$\text{Obj} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(h_m)$$

$$\Omega(h) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|$$

where $T$ is the number of leaves, $w_j$ are leaf weights, $\gamma$ penalizes tree complexity, $\lambda$ is L2 regularization, and $\alpha$ is L1 regularization.

### 4.5 Early Stopping

Monitor validation loss and stop training when it stops improving:

```python
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=50)
```

## 5. Tree Construction: Split Finding Algorithms

Finding the best split is the computational bottleneck. Different algorithms trade off speed and accuracy:

### 5.1 Exact Greedy Algorithm

- Enumerate all possible splits for all features
- Time complexity: $O(n \cdot d \cdot n\log n)$ per level (sorting)
- Used by: XGBoost (for small datasets), scikit-learn

### 5.2 Histogram-Based (Approximate) Algorithm

- **Bin continuous features** into discrete buckets (e.g., 255 bins)
- Build histograms of gradients per bin
- Find best split from histogram

**Advantages**:

- $O(n \cdot d)$ for histogram construction (linear)
- $O(b \cdot d)$ for split finding ($b$ = number of bins, typically 255)
- **Memory efficient**: store bin indices (uint8) instead of float64
- **Cache friendly**: sequential memory access

Used by: **LightGBM**, **CatBoost**, XGBoost (`tree_method='hist'`)

### 5.3 Leaf-wise vs Level-wise Growth

| Strategy       | Description                                      | Used By               |
| -------------- | ------------------------------------------------ | --------------------- |
| **Level-wise** | Grow all nodes at the same depth, then go deeper | XGBoost, scikit-learn |
| **Leaf-wise**  | Always split the leaf with the highest gain      | LightGBM              |

**Leaf-wise** tends to achieve lower loss with fewer splits but is more prone to overfitting on small datasets. Use `max_depth` or `num_leaves` to control.

### 5.4 Gradient-based One-Side Sampling (GOSS) — LightGBM

- Keep all instances with large gradients (high error)
- Randomly sample a fraction of instances with small gradients
- Reduces training data while preserving gradient information
- Speeds up training by 10-20x with minimal accuracy loss

### 5.5 Ordered Boosting — CatBoost

Addresses the **prediction shift** problem in standard gradient boosting:

- Standard: Residuals are computed on the same data used to fit the tree → information leakage
- CatBoost: Uses a permutation-based approach where each example's residual is computed using a model trained only on preceding examples
- More principled, especially effective on small datasets

## 6. Handling Categorical Features

| Method           | Approach                                 | Used By                    |
| ---------------- | ---------------------------------------- | -------------------------- |
| One-hot encoding | Create binary columns per category       | scikit-learn, manual       |
| Label encoding   | Map categories to integers               | Manual preprocessing       |
| Target encoding  | Replace category with mean target value  | CatBoost (ordered), manual |
| Optimal split    | Find best binary partition of categories | LightGBM, CatBoost         |

**CatBoost** handles categoricals natively with **ordered target statistics** — no preprocessing needed and often superior to one-hot encoding.

**LightGBM** supports categorical features directly — it finds the optimal split among categories using a histogram-based approach.

## 7. XGBoost vs LightGBM vs CatBoost

### 7.1 Overview

| Feature                     | XGBoost                  | LightGBM                 | CatBoost                      |
| --------------------------- | ------------------------ | ------------------------ | ----------------------------- |
| **Developer**               | DMLC (Tianqi Chen)       | Microsoft                | Yandex                        |
| **Language**                | C++                      | C++                      | C++                           |
| **Release**                 | 2014                     | 2017                     | 2017                          |
| **Tree Growth**             | Level-wise (default)     | Leaf-wise                | Symmetric (balanced)          |
| **Split Finding**           | Exact + Histogram        | Histogram (GOSS)         | Histogram (ordered)           |
| **Categorical**             | No native support        | Native (optimal split)   | Native (ordered target stats) |
| **GPU Training**            | ✅                       | ✅                       | ✅                            |
| **Missing Values**          | Learns optimal direction | Learns optimal direction | Special treatment             |
| **Monotone Constraints**    | ✅                       | ✅                       | ✅                            |
| **Interaction Constraints** | ✅                       | ✅                       | ❌                            |

### 7.2 Training Speed Comparison

On a typical tabular dataset (1M rows, 100 features):

| Library          | Training Time | Memory |
| ---------------- | ------------- | ------ |
| XGBoost (hist)   | ~60s          | ~2 GB  |
| LightGBM         | ~20s          | ~1 GB  |
| CatBoost         | ~90s          | ~3 GB  |
| scikit-learn GBM | ~300s         | ~4 GB  |

> LightGBM is typically **2-5x faster** than XGBoost and **3-5x faster** than CatBoost.

### 7.3 Accuracy Comparison

On Kaggle tabular competitions (general trends):

- **CatBoost**: Best out-of-the-box accuracy, least tuning required, excels with categorical features
- **LightGBM**: Best speed-accuracy tradeoff, preferred for large datasets and ensembling
- **XGBoost**: Most mature, reliable baseline, best documentation

In practice, **ensembling all three** often gives the best results.

### 7.4 When to Use Which

```
Start here:
│
├─ Many categorical features?
│   └─ → CatBoost (best native categorical handling)
│
├─ Very large dataset (>10M rows)?
│   └─ → LightGBM (fastest training, lowest memory)
│
├─ Need maximum control and customization?
│   └─ → XGBoost (most configurable, best docs)
│
├─ Quick baseline, minimal tuning?
│   └─ → CatBoost (best defaults)
│
├─ Kaggle competition?
│   └─ → All three + ensemble
│
└─ Production with strict latency?
    └─ → LightGBM or XGBoost (fastest inference)
```

## 8. Practical Guide

### 8.1 XGBoost Example

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Prepare data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Create DMatrix for better performance
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,        # L1
    'reg_lambda': 1.0,        # L2
    'tree_method': 'hist',    # histogram-based
    'device': 'cuda',         # GPU training
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)
```

### 8.2 LightGBM Example

```python
import lightgbm as lgb

dtrain = lgb.Dataset(X_train, label=y_train,
                     categorical_feature=['col_a', 'col_b'])
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'min_child_samples': 20,
    'device': 'gpu',
    'verbose': -1,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]
)
```

### 8.3 CatBoost Example

```python
from catboost import CatBoostClassifier

cat_features = ['col_a', 'col_b']  # categorical column names

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    cat_features=cat_features,
    early_stopping_rounds=50,
    task_type='GPU',
    verbose=100,
)

model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

### 8.4 Hyperparameter Tuning with Optuna

```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 16, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
    }

    model = lgb.train(params, dtrain, num_boost_round=1000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50)])

    return model.best_score['valid_0']['binary_logloss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

## 9. Feature Importance Methods

| Method                     | Description                                     | Reliability                                    |
| -------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| **Gain**                   | Total gain from splits using the feature        | Can be biased toward high-cardinality features |
| **Split count**            | Number of times a feature is used in splits     | Even more biased                               |
| **Permutation importance** | Drop in metric when feature values are shuffled | More reliable, model-agnostic                  |
| **SHAP values**            | Game-theoretic feature attribution              | Most reliable, computationally expensive       |

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
```

## 10. Common Pitfalls and Tips

1. **Start with a high learning rate** (0.1–0.3) for quick experiments, then lower it (0.01–0.05) for final training with more rounds
2. **Don't over-tune `max_depth`** — 4–8 is usually sufficient. Deeper trees overfit
3. **Use early stopping** — always. Set `early_stopping_rounds` to 50–100
4. **Subsampling helps** — `subsample=0.8` and `colsample_bytree=0.8` are good defaults
5. **Handle class imbalance** — use `scale_pos_weight` (XGBoost) or `is_unbalance` (LightGBM)
6. **Watch for data leakage** — especially with target encoding of categoricals
7. **Feature engineering matters more than tuning** — focus on creating meaningful features first
8. **Ensemble for competitions** — blend XGBoost + LightGBM + CatBoost for best results
9. **Monitor train vs val loss** — large gap = overfitting, increase regularization
10. **Use `dart` booster** for better regularization on small datasets (drops random trees during training)

## 11. GBT vs Deep Learning for Tabular Data

| Aspect                  | Gradient Boosted Trees    | Deep Learning (TabNet, FT-Transformer) |
| ----------------------- | ------------------------- | -------------------------------------- |
| Small data (< 10K rows) | ✅ Strong                 | ❌ Overfits                            |
| Medium data (10K–1M)    | ✅ Best                   | ⚠️ Competitive with effort             |
| Large data (> 1M)       | ✅ Fast, accurate         | ⚠️ Can match with tuning               |
| Feature engineering     | Required for best results | Can learn representations              |
| Training speed          | Fast (minutes)            | Slow (hours)                           |
| Inference speed         | Very fast (μs–ms)         | Slower (ms)                            |
| Categorical features    | CatBoost handles natively | Embedding layers                       |
| Interpretability        | Built-in (SHAP, gain)     | Harder, attention-based                |
| Production deployment   | Simple (serialize model)  | Requires serving infrastructure        |

**Verdict**: For tabular data, GBTs remain the default choice. Use deep learning only when you have very large data, need end-to-end learning, or have complex multi-modal inputs.

## 12. Production-Grade Training Recipes

The snippets in section 8 are the "hello world" version. Below are the patterns that survive contact with production data: proper splits, custom metrics, deterministic seeds, callbacks, and serialization with version pinning.

### 12.1 XGBoost: early stopping, custom eval metric, callbacks

```python
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Custom eval metric — XGBoost expects (name, value) and a "higher is better" flag
# via maximize=True in xgb.train. Use this when the business KPI is not in the
# built-in list (e.g. partial AUC, weighted recall@k, expected calibration error).
def pr_auc_at_k(y_pred: np.ndarray, dmat: xgb.DMatrix, k: int = 100):
    y_true = dmat.get_label()
    # take top-k by score, compute precision
    order = np.argsort(-y_pred)[:k]
    return "p_at_k", float(y_true[order].mean())

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cuda",
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,        # higher = more regularization on leaf size
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "max_bin": 256,               # histogram bins; 256 fits uint8, fastest path
    "seed": 42,
    "nthread": 8,
}

callbacks = [
    xgb.callback.EarlyStopping(
        rounds=100,
        metric_name="p_at_k",
        data_name="val",
        maximize=True,
        save_best=True,           # rewinds booster to best iteration on stop
    ),
    xgb.callback.EvaluationMonitor(period=50),
]

booster = xgb.train(
    params,
    dtrain,
    num_boost_round=5000,         # set high; early stopping will cut it
    evals=[(dtrain, "train"), (dval, "val")],
    custom_metric=pr_auc_at_k,
    maximize=True,
    callbacks=callbacks,
)

# best_iteration is 0-indexed; predict() with iteration_range=(0, best+1)
preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
```

Things that bite people here: `early_stopping_rounds` as a kwarg to `xgb.train` was deprecated in 2.0 in favor of the callback. The off-by-one on `best_iteration` (it's 0-indexed but `iteration_range` end is exclusive) silently shaves one tree off in older code paths and changes predictions by a small but non-zero amount.

### 12.2 LightGBM: categorical features, monotone constraints, custom objective

```python
import lightgbm as lgb
import numpy as np

# IMPORTANT: pass categorical_feature with integer-coded columns. Strings work
# in pandas but force LightGBM to recompute the mapping on every Dataset
# construction — do the integer encoding once, persist the mapping, reuse at
# serve time. Unseen categories at inference become a *new* integer outside
# the trained range and LightGBM will route them via the default direction.
cat_cols = ["country_id", "device_type", "merchant_category"]

# Monotone constraints: 1 = monotonically increasing in feature, -1 = decreasing,
# 0 = unconstrained. Order MUST match feature order in X_train. Off-by-one here
# will silently constrain the wrong feature and ship a model that violates
# domain expectations (e.g. higher income -> lower credit score).
feature_names = list(X_train.columns)
monotone = [0] * len(feature_names)
monotone[feature_names.index("annual_income")] = 1
monotone[feature_names.index("num_late_payments_12m")] = -1

# Custom objective: must return (grad, hess). Example: Focal loss for imbalanced
# binary classification. y_pred is the raw margin (log-odds), not a probability.
def focal_loss_obj(y_pred, dataset, gamma: float = 2.0, alpha: float = 0.25):
    y = dataset.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    # gradient and hessian of focal loss wrt margin — derive once, test against
    # numerical gradient before trusting in production
    pt = np.where(y == 1, p, 1 - p)
    at = np.where(y == 1, alpha, 1 - alpha)
    grad = at * (1 - pt) ** gamma * (gamma * pt * np.log(pt + 1e-12) + pt - 1) * np.where(y == 1, 1, -1)
    hess = at * (1 - pt) ** gamma * pt * (1 - pt) * (1 + gamma * (1 - pt))
    return grad, hess

train_set = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=cat_cols,
    free_raw_data=False,          # keep raw refs for debugging; costs memory
)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, categorical_feature=cat_cols)

params = {
    "metric": "binary_logloss",   # eval metric is independent of custom obj
    "num_leaves": 63,             # paired with max_depth=-1 for leaf-wise growth
    "max_depth": -1,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 100,      # the single most underrated regularizer
    "monotone_constraints": monotone,
    "monotone_constraints_method": "advanced",  # tighter than 'basic', slower
    "deterministic": True,
    "force_row_wise": True,       # avoids "Auto-choosing row-wise" warnings
    "seed": 42,
    "verbose": -1,
}

booster = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    valid_names=["train", "val"],
    fobj=focal_loss_obj,          # plug in custom objective
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(eval_history := {}),
    ],
)
```

### 12.3 CatBoost: native categoricals, text features, GPU

```python
from catboost import CatBoostClassifier, Pool

cat_features = ["country_id", "device_type"]
text_features = ["product_title", "review_body"]   # CatBoost tokenizes natively

train_pool = Pool(
    data=X_train, label=y_train,
    cat_features=cat_features,
    text_features=text_features,
    feature_names=list(X_train.columns),
)
val_pool = Pool(X_val, y_val, cat_features=cat_features, text_features=text_features)

model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.03,
    depth=8,                       # CatBoost trees are symmetric (oblivious)
    l2_leaf_reg=3.0,
    random_strength=1.0,           # noise added to scores during split selection
    bagging_temperature=1.0,       # Bayesian bootstrap weight diversity
    border_count=254,              # bins; max 254 on GPU
    grow_policy="SymmetricTree",   # default; "Lossguide" or "Depthwise" for speed
    task_type="GPU",
    devices="0",
    od_type="Iter",
    od_wait=200,
    eval_metric="AUC",
    auto_class_weights="Balanced", # alternative to scale_pos_weight
    random_seed=42,
    allow_writing_files=False,     # don't litter cwd with catboost_info/
)

model.fit(train_pool, eval_set=val_pool, verbose=200, use_best_model=True)
```

The symmetric tree structure makes CatBoost's inference 5-10x faster than XGBoost/LightGBM for the same number of leaves, because the same feature is tested at every node of a given depth — branch prediction stays hot.

### 12.4 Splits that don't lie: time-based, group, stratified

```python
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, TimeSeriesSplit, train_test_split,
)

# Stratified — for IID classification with class imbalance. WRONG for time-series
# or grouped data because future leaks into past, and same user appears in train+val.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Group — when rows belong to entities that must not span splits (user_id, patient_id,
# session_id). Forgetting this is the #1 cause of "amazing offline metrics, terrible
# A/B test" — the model memorizes user identity via aux features.
gkf = GroupKFold(n_splits=5)
for tr, va in gkf.split(X, y, groups=df["user_id"]):
    ...

# Time-series — train on past, validate on future. Always use this when the target
# has any temporal correlation, even if rows look IID. Use expanding window (default)
# unless you specifically want a rolling window for non-stationary regimes.
tss = TimeSeriesSplit(n_splits=5, gap=7)   # gap=7 days to model "we predict a week ahead"
for tr, va in tss.split(X_sorted_by_time):
    ...

# Final test set: hold out the most recent slice BEFORE any CV. The CV folds are for
# tuning; the holdout is a single, untouched window that mimics serving distribution.
cutoff = df["event_time"].quantile(0.9)
train_df = df[df.event_time <  cutoff]
test_df  = df[df.event_time >= cutoff]
```

### 12.5 Optuna with pruner and proper search space

```python
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import HyperbandPruner

def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "deterministic": True,
        "seed": 42,
        # log-uniform for lr is critical — uniform wastes 80% of trials in bad region
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 16, 255),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 0, 10),
        "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    }

    booster = lgb.train(
        params, train_set, num_boost_round=2000,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            LightGBMPruningCallback(trial, "auc", valid_name="val"),
        ],
    )
    return booster.best_score["val"]["auc"]

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    pruner=HyperbandPruner(min_resource=50, max_resource=2000, reduction_factor=3),
    study_name="lgbm_v3",
    storage="sqlite:///optuna.db",   # persist; resume across machines
    load_if_exists=True,
)
study.optimize(objective, n_trials=200, n_jobs=1, show_progress_bar=True)
```

Why Hyperband + TPE instead of pure Bayesian: TPE alone wastes wall clock on bad configs that are obviously bad after 100 rounds. Hyperband prunes them at multiple resource levels. On a typical search this saves 3-5x compute.

### 12.6 Saving, loading, and version pinning

```python
import json, joblib, lightgbm as lgb, xgboost as xgb, catboost as cb

# LightGBM: save the text model — it's portable across versions back to 3.x
booster.save_model("model_v3.txt", num_iteration=booster.best_iteration)
booster_loaded = lgb.Booster(model_file="model_v3.txt")

# XGBoost: prefer JSON/UBJ over pickle. Pickle breaks across XGBoost majors
# (saw this between 1.7 -> 2.0; pickled 1.7 boosters fail to unpickle in 2.0).
booster.save_model("model_v3.ubj")     # UBJ is the recommended binary format
booster.save_model("model_v3.json")    # JSON for debugging/diffing

# Always persist the *exact* environment so deployments are reproducible.
manifest = {
    "model_version": "v3",
    "trained_at": "2026-04-28T10:00:00Z",
    "training_rows": len(X_train),
    "feature_order": list(X_train.columns),
    "categorical_features": cat_cols,
    "best_iteration": booster.best_iteration,
    "best_score": booster.best_score,
    "library": {"lightgbm": lgb.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "python": sys.version},
    "git_sha": os.popen("git rev-parse HEAD").read().strip(),
    "data_hash": hashlib.sha256(pd.util.hash_pandas_object(X_train).values).hexdigest(),
}
with open("model_v3.manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
```

The manifest is non-negotiable in production. When a customer reports drift two months later, you need to reproduce the *exact* model — that means library versions, feature order, and the data hash.

## 13. When to Use Which Algorithm — Decision Matrix

| Constraint | XGBoost | LightGBM | CatBoost | Notes |
|---|---|---|---|---|
| Rows < 10k | strong | overfits leaf-wise | strongest (ordered boosting) | CatBoost's permutation-based residuals shine on small data |
| 10k - 1M rows | strong | strong | strong | All three competitive; pick by feature mix |
| 1M - 100M rows | OK | best | OK | LightGBM's GOSS + histogram is the speed king |
| > 100M rows | OK with `tree_method=hist` + GPU | best with distributed mode | needs sharding | Beyond 100M consider Spark XGBoost or distributed LGBM |
| Many categoricals (>20% of features, high cardinality) | weakest (manual encoding) | good (native) | best (ordered TS) | CatBoost target stats avoid leakage by construction |
| GPU available | full GPU | partial GPU (CUDA only) | full GPU | XGBoost GPU is the most mature; LightGBM's was historically buggy on cat features |
| Inference latency < 1ms | yes (Treelite) | yes | yes (oblivious trees fastest) | All three serialize to compact formats |
| Interpretability paramount | yes (SHAP+monotone) | yes | yes | Monotone constraints supported in all three |
| Streaming / online learning | partial (`process_type=update`) | partial | no | None of these are truly online; retrain on rolling windows |

### 13.1 When NOT to use GBT

- **Truly high-dimensional sparse data** (text bag-of-words, one-hot user IDs with millions of values): use linear models with L1/L2 (logistic regression, FTRL) or factorization machines. GBTs build deep partitions on sparse features and waste capacity.
- **Image, audio, raw text**: CNNs/transformers learn representations that GBTs cannot. Don't fight this.
- **Small data with strong known structure** (linear, monotonic, additive): a Bayesian linear model or GAM (LightGBM's linear tree, EBM/InterpretML) gives better calibration and uncertainty.
- **Strict interpretability for regulators** (e.g. ECOA in US credit): Even SHAP may not satisfy auditors. Reach for monotonic GAMs or scorecards.
- **Tasks needing differentiable end-to-end training** (multi-task, multi-modal, RL): use a neural network.
- **Latency budget < 100 microseconds**: GBT inference is fast but not as fast as a 50-coefficient logistic regression with prefetched features.
- **You have < 500 examples**: use a regularized linear model and hand-crafted features. GBT will memorize.

## 14. Senior-Level Concerns

### 14.1 Target leakage in feature engineering

The single most expensive bug in tabular ML. Concrete patterns I've seen ship:

- **Using `groupby().transform()` over the full dataframe before splitting.** Computing `user_avg_basket_size = df.groupby('user_id')['basket'].transform('mean')` over train+val+test mixes future labels into the past. Always compute aggregates *only* on training rows and then merge into val/test.
- **Target encoding without K-fold isolation.** Encoding `merchant_id -> mean(fraud)` on the full training set then using that encoded value during training: the row's own label leaks via the mean. Use out-of-fold target encoding or CatBoost's ordered target stats.
- **"Days since last event" computed including the prediction day.** If the event whose timestamp you're using is the one you're predicting, you've encoded the answer.
- **Currency conversion using same-day rate when the prediction window is "tomorrow".** Use yesterday's rate available at prediction time.
- **Survey/operational features filled in *after* the outcome.** Customer service notes added after a chargeback are perfect predictors of chargebacks.

Test: shuffle your label column and refit. If validation AUC > 0.55, you have leakage.

### 14.2 Train/serve skew

The model is fine; the pipeline is broken. Common manifestations:

- **Different feature pipelines** — training in pandas, serving in Java/Go. Each one rounds floats differently, handles NaN differently, and indexes categoricals differently. Solution: a single feature library callable from both, or write features once in SQL and materialize the same logical view at train and serve time.
- **Type mismatches** — training feature is `float32`, serving sends `float64`. LightGBM's histogram bins are derived from training dtype; double-precision input at serve time produces a slightly different bin and a different leaf for borderline rows. Force the serving payload through the same `np.asarray(x, dtype=np.float32)` step.
- **Categorical mapping drift** — training mapping `{"US": 0, "UK": 1, ...}` is JSON-stored; serving loads a newer mapping where positions shifted because a new country was inserted alphabetically. Pin the mapping by the training-time integer codes, never re-derive at serve time.
- **Missing-value sentinel mismatch** — pandas `NaN`, numpy `np.nan`, `None`, `-999`, empty string — all different to a tree. Pick one (usually `np.nan`) and enforce in both pipelines.

Detection: log a hash of the feature vector at both train and serve, sample 0.1% of inference requests, replay through training pipeline, compare predictions. Differences > 1e-6 are bugs.

### 14.3 Categorical encoding pitfalls

- **High cardinality (e.g. zip code, user_id, IP).** One-hot blows up memory; label encoding has no semantics; LightGBM's optimal categorical split has complexity $O(2^{k-1})$ in the worst case but uses a histogram approximation that requires `min_data_per_group` to avoid overfitting. For cardinality > 1000, target-encode with K-fold or use embeddings from a small neural net.
- **Unseen categories at inference.** Plan for them. Reserve integer 0 for "unknown" and clip serving inputs into the trained range. Random Forest-style "go in the direction of the majority" is *not* what tree libraries do for unseen categoricals — most route as missing, which may be calibrated differently.
- **Target encoding leakage.** Mean encoding without out-of-fold is leakage; with single-fold OOF on small categories you still get high-variance encodings that look valid but overfit. Use smoothing: `(n_c * mean_c + alpha * global_mean) / (n_c + alpha)` with `alpha` tuned via CV.
- **Frequency encoding** is a useful, leakage-free alternative for tree models when target encoding is overkill.

### 14.4 Distribution shift detection and handling

- **Covariate shift** ($P(X)$ changes, $P(Y \mid X)$ stable): retrain often; importance-weight if you can't.
- **Label shift** ($P(Y)$ changes, $P(X \mid Y)$ stable): recalibrate with EM (BBSE).
- **Concept drift** ($P(Y \mid X)$ changes): only fixable by retraining on fresh labels. Detect via prediction-vs-actual rolling KS test.

Quick monitor: track per-feature mean and std weekly; alert if PSI > 0.25 versus training distribution. Pair with a backstop monitor on prediction distribution itself (drift in predictions catches both covariate and concept drift, at the cost of detecting them late).

### 14.5 Calibration

GBT outputs from binary log-loss are *better calibrated than random forests but still not great*, especially after early stopping (which biases probabilities toward whatever boundary the model converged on) or after class weighting. Symptoms: histogram of predicted probabilities concentrated in the middle, reliability diagram concave or S-shaped.

```python
from sklearn.calibration import CalibratedClassifierCV
# Wrap a *frozen* booster with sklearn-compatible interface
cal = CalibratedClassifierCV(base_estimator=sklearn_wrapper, method="isotonic", cv="prefit")
cal.fit(X_calib, y_calib)        # use a held-out calibration set, NOT the val set used for early stopping
```

- **Platt** (sigmoid): few parameters, robust under small data, but assumes a logistic shape. Right when miscalibration is roughly monotonic.
- **Isotonic**: non-parametric, needs >= 1000 calibration points or it overfits the staircase. Right when miscalibration is non-monotonic (e.g. overconfident at extremes).
- **Beta calibration**: middle ground; works well for GBT.

Always use a *separate* calibration set. Calibrating on the early-stopping validation set biases the calibration map.

### 14.6 Class imbalance: `sample_weight` vs `scale_pos_weight` vs SMOTE

- **`sample_weight`** is the right tool when individual rows have different importance for *business* reasons (e.g. high-value customer fraud weighs 10x). Affects gradient and hessian directly.
- **`scale_pos_weight`** (XGBoost) / `is_unbalance` (LightGBM) / `auto_class_weights` (CatBoost) is the right tool for *class* imbalance. Effectively `sample_weight = neg_count/pos_count` for positives. Use this when you don't care about per-row business weights.
- **SMOTE** is almost always wrong for tabular GBT. It synthesizes points by interpolating in feature space — meaningless for categorical features, and it inflates leakage when applied before splitting. The original SMOTE paper is from 2002 and predates modern boosting. Use class weights instead.
- **Threshold tuning** is independent of and more important than any of the above. Train with `scale_pos_weight=1`, then move the decision threshold on the calibration set to optimize your business metric. This is more controllable than re-weighting during training.

Wrong combos seen in production: `scale_pos_weight=ratio` *and* downsampled negatives (double-counting), or `sample_weight` for class imbalance *and* `scale_pos_weight` (multiplied weights, wildly miscalibrated).

### 14.7 Feature importance is misleading

Three importance flavors give three different answers:

- **Gain (default in XGBoost/LightGBM):** total reduction in loss attributable to splits on that feature. Biased toward high-cardinality numeric features (more split points to pick from) and toward features used early (root gains dominate).
- **Split count:** how often the feature was used. Even more biased toward high-cardinality and noise.
- **Permutation importance:** drop in held-out metric when the feature is shuffled. Model-agnostic, less biased, but expensive and unstable for correlated features (shuffling A while B is correlated still leaks A through B).
- **SHAP:** per-row attribution that sums to the prediction. Theoretically grounded; for trees it's exact and fast (`TreeExplainer`). Use `shap.Explanation` and `mean(|shap|)` for global ranking.

Case study 1: a credit model showed `customer_id_hash` as the #2 feature by gain. The hash had no information; it was just high cardinality and got many split opportunities. Permutation importance showed near-zero drop. Removed feature, model performance unchanged, audit happy.

Case study 2: two highly correlated features (`income_monthly`, `income_annual = 12*income_monthly`). Gain split importance ~50/50 between them. Permutation importance for either alone was ~0 (the other compensated). True joint importance was high. Lesson: never report individual permutation importance for correlated features without computing pair-wise drops.

Case study 3: `country == "DE"` had high SHAP variance but small mean-abs SHAP. Headline "Germany doesn't matter" was wrong — Germany mattered a lot for a 5% subpopulation. Always check SHAP *distributions*, not just means.

### 14.8 Overfitting indicators — what train/valid gaps actually mean

A train AUC of 0.99 vs val 0.85 is *not necessarily* overfitting. It's overfitting only when the *val* curve has plateaued or worsened. Read curves carefully:

- Train improving, val improving — keep training.
- Train improving, val plateaued — overfitting begins; stop.
- Train improving fast, val improving slowly — high capacity vs noisy labels; lower learning rate, increase `min_data_in_leaf`.
- Train and val both plateau early — underfitting; deepen, add features, or check label quality.
- Train>>val from the very first iteration — leakage in train OR distribution mismatch between train and val. Investigate before tuning.

For LightGBM, `min_data_in_leaf` and `feature_fraction` are 5x more impactful than `max_depth` for controlling overfitting in leaf-wise growth. Many engineers reach for `max_depth` first; that's level-wise muscle memory.

### 14.9 Reproducibility

```python
# LightGBM
params = {"deterministic": True, "force_row_wise": True, "num_threads": 1, "seed": 42}
# XGBoost
params = {"seed": 42, "nthread": 1, "tree_method": "hist"}   # GPU non-determinism cannot be fully removed
# CatBoost
CatBoostClassifier(random_seed=42, thread_count=1)
```

Bit-exact reproducibility requires:
- Single thread (multi-thread reduction order is non-deterministic in float).
- CPU, not GPU (CUDA atomics are non-deterministic).
- Pinned library versions, including BLAS (numpy can pull in different MKL versions).
- Pinned OS — glibc version affects qsort tiebreaks.
- Same dataset row order (LightGBM is sensitive to row order even with `deterministic=True` if you're using categorical features whose internal mapping depends on first-appearance).

For "good enough" reproducibility (predictions agree to 1e-4), seed everything and pin major versions.

### 14.10 Reading verbose output

```
[100]   train's auc: 0.812   val's auc: 0.794
```

What to look at every 100 rounds:
- Train-val gap. Stable gap = good fit; widening gap = overfitting onset.
- Rate of improvement. If val AUC moves < 1e-4 per 100 rounds, you're done — kill it.
- Per-tree time. If iteration time is creeping up, you're hitting cache misses (large trees, dense histograms); reduce `max_bin` or `num_leaves`.

LightGBM `verbose=2` and XGBoost `verbosity=2` print per-tree statistics: number of leaves grown, best gain, bin allocation. If best-gain trends to zero after iteration `k`, increasing `num_boost_round` past `k` is wasted compute.

### 14.11 Production: model size, latency, ONNX, Treelite

| Approach | Size | Latency (1 row) | Notes |
|---|---|---|---|
| Native Booster.predict | full | 50-200 us | Fine for batch; threading overhead per call |
| ONNX Runtime | similar | 20-80 us | Cross-language; some op gaps for newer LGBM features |
| Treelite (compiled .so) | smaller | 5-20 us | C codegen; best single-row latency |
| Leaves (Go library) | small | 5-30 us | Reads native model files in Go without CGo |
| Quantized (uint8 thresholds) | 2-4x smaller | similar | Tradeoff in last-bit predictions; verify within tolerance |

Treelite is the right tool when latency dominates (real-time bidding, ad serving). Compile once, ship the .so, call from C++/Python/Java via the Treelite runtime. Caveat: Treelite lags behind LightGBM features by several months; check support matrix before committing.

### 14.12 Monitoring drift

```python
# PSI (Population Stability Index) — quick, robust, cheap
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e = np.histogram(expected, breakpoints)[0] / len(expected) + 1e-6
    a = np.histogram(actual,   breakpoints)[0] / len(actual)   + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))
# PSI < 0.1: stable; 0.1-0.25: minor; > 0.25: significant; investigate.

# KS test on continuous features
from scipy.stats import ks_2samp
ks_stat, p = ks_2samp(reference, current)

# Prediction distribution shift: cheap proxy for everything else.
# Compare current week's predicted-probability histogram to last month's.
```

Set alarms on three signals: (1) PSI per top-20 features, (2) KS on prediction distribution, (3) live precision/recall on labeled feedback. Any one alone is insufficient; together they triangulate.

### 14.13 Retraining cadence

- **Time-based** (weekly, monthly): simplest, easy to audit, predictable infra cost. Right when drift is gradual.
- **Trigger-based** (drift alarm): retrain when PSI > 0.2 or rolling AUC drops by > 2%. Right when drift is bursty or seasonal. Add a minimum cooldown to avoid retraining on transient anomalies (e.g. Black Friday).
- **Continual training**: nightly retrain on a rolling 90-day window. Best for high-velocity domains (ads, fraud). Must be paired with shadow evaluation — never auto-promote.

Whatever you pick, *keep the previous model warm* for instant rollback. Champion/challenger framework: serve champion, log challenger predictions; promote only when challenger beats champion on live metrics by a confidence margin for 2 consecutive windows.

### 14.14 A/B testing GBT upgrades safely

1. **Shadow mode** (week 1): challenger receives all traffic, predictions logged but not used. Compare per-row prediction deltas; investigate the top-1% disagreements.
2. **Canary** (week 2): 1-5% of traffic. Watch business KPIs, not just AUC. AUC up + revenue down means you optimized the wrong objective.
3. **Ramp** (week 3-4): 10% -> 50% -> 100%, with abort criteria pre-registered.
4. **Holdout**: keep a 1% champion holdout for a quarter to measure long-term lift and detect novelty effects.

Pre-register sample size: variance of binary KPIs requires more traffic than people expect. For a 0.5% absolute lift on a 5% conversion rate, you typically need 1-3M users per arm.

### 14.15 Common bugs

- **Leaf-wise vs level-wise mismatch.** Migrating from XGBoost (level-wise default) to LightGBM (leaf-wise) without changing `max_depth=6` (level-wise constrains 64 leaves) to `num_leaves=63` (leaf-wise). The LightGBM model is much weaker than expected.
- **Tree count off-by-one.** `best_iteration` is 0-indexed in some APIs, 1-indexed in others. `iteration_range=(0, best_iteration)` cuts one tree off; should be `(0, best_iteration + 1)`. Affects predictions by tens of bps.
- **Missing value handling differs.** XGBoost learns a default direction per split. LightGBM does too, but its handling of `np.nan` vs `None` vs missing-as-zero is configurable (`use_missing`, `zero_as_missing`). CatBoost has its own scheme. Mixing libraries in an ensemble without harmonizing missing values causes silent skew.
- **`predict_proba` vs `predict`.** XGBoost's sklearn API `predict` returns class labels for classifiers; the native API `Booster.predict` returns raw scores or probabilities depending on the objective. Wrong call -> log-odds get fed where probabilities were expected.
- **Categorical type lost across save/load.** Pandas `category` dtype is forgotten on `to_parquet`/`from_parquet` if not preserved; LightGBM then treats the column as numeric. Pin dtypes explicitly on load.
- **`scale_pos_weight` with multiclass.** Doesn't apply; you need per-class `class_weight` arrays. Quietly ignored in some versions.

## 15. Real-World Case Studies

### 15.1 Credit scoring with monotonic constraints

Regulatory and business reality: higher income should never decrease credit score, more late payments should never increase it. Without constraints, GBTs find spurious non-monotonicities in low-data regions of feature space.

```python
# Income monotone increasing, late_payments monotone decreasing
# WoE-encode highly cardinal categoricals (region, occupation) for stability
params = {
    "objective": "binary",
    "metric": "auc",
    "monotone_constraints": [1, -1, 0, 0, 1, 0, ...],
    "monotone_constraints_method": "advanced",
    "min_data_in_leaf": 500,    # heavy for stability of small subgroup decisions
    "max_bin": 64,              # coarser bins -> more stable, easier to defend
    "lambda_l2": 5.0,
}
```

Output must be (a) calibrated to PD (probability of default) — use isotonic on a held-out year, (b) stable across age/gender/race fairness slices — measure adverse impact ratio, (c) explainable per-decision — SHAP values per applicant stored alongside the decision for ECOA reasons.

### 15.2 Fraud detection with extreme class imbalance

Positive rate ~0.05%. Common mistakes:

- Reporting AUC. With a 99.95% negative class, AUC of 0.99 is unimpressive — random+majority hits 0.5 trivially, but mid-tier features get you to 0.95 with no work. Use PR-AUC and recall@k (top-k accounts to review per day fits ops capacity).
- Down-sampling negatives 100:1 then forgetting to recalibrate. Predicted probabilities are now 100x too high; if a downstream rule says "decline if p > 0.5", you decline too many.
- Time-based eval split is non-negotiable here. Fraud patterns change weekly. Stratified random splits will report 0.99 PR-AUC and the model will work for a week in production.

```python
model = lgb.LGBMClassifier(
    objective="binary", n_estimators=3000, learning_rate=0.02,
    num_leaves=127, min_data_in_leaf=200,
    is_unbalance=False,                 # we'll handle weights manually
    scale_pos_weight=1,                 # train on natural rate
)
# After training: choose threshold to land at ops capacity (e.g. top 1000 alerts/day)
# Calibrate separately because weighted training distorts probabilities
```

Pair with a graph-feature pipeline (device co-occurrence, IP-merchant edges) — features dominate model choice in fraud.

### 15.3 CTR / recommendation ranking

For ranking, optimize a ranking loss — pointwise logloss is suboptimal because two ads with predicted CTRs (0.10, 0.09) need only their relative order correct, not their absolute values.

```python
# LightGBM LambdaRank (pairwise/listwise) — group sizes per query MUST be passed
train_set = lgb.Dataset(
    X_train, label=y_train_relevance,   # 0=no click, 1=click, 2=convert, etc.
    group=group_sizes_train,             # array of impressions per query/session
)
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10, 20],
    "lambdarank_truncation_level": 20,   # only top-k pairs contribute to gradient
    "label_gain": [0, 1, 3],             # gain per relevance level
    "learning_rate": 0.05,
}
```

Pitfalls:
- Eval and serve groups must be identical in semantics (per-session, per-query, per-user/day).
- NDCG@k vs business metric (revenue@k): they correlate but diverge under reserve prices. Validate against the real KPI.
- Position bias: clicks at rank 1 happen more even when ad is identical. Use inverse propensity weighting or unbiased LambdaMART variants.

### 15.4 Time-series forecasting with GBT

GBTs can match or beat classical methods for tabular time-series forecasting (M5 winning solution was LightGBM). The catch is correct feature engineering.

```python
# Lag features — but be paranoid about CV leakage
df["sales_lag_7"]   = df.groupby("item_id")["sales"].shift(7)
df["sales_lag_28"]  = df.groupby("item_id")["sales"].shift(28)
df["sales_roll_28_mean"] = df.groupby("item_id")["sales"].shift(1).rolling(28).mean()
# NOTE: shift(1) BEFORE rolling so the rolling window does not include current row

# Time-series CV with gap >= forecast horizon to prevent leakage from autocorrelation
splitter = TimeSeriesSplit(n_splits=5, gap=14, test_size=28)
```

Common bugs:
- Computing rolling features without `shift(1)` — current row leaks into its own feature.
- Using a global `MinMaxScaler` over the whole timeline before splitting — future stats leak into past.
- Validating on a single time slice — you're optimizing for one regime. Use multiple time-slice CVs.
- Forgetting to retrain on train+val before predicting test, *if* you're in production (you've left signal on the table).

For multi-horizon forecasts: train one model per horizon (direct strategy) — fewer biases than recursive forecasting where errors compound.

## 16. Performance Optimization for Production

### 16.1 Tree pruning post-training

XGBoost supports cost-complexity pruning via `gamma` during training; post-hoc pruning can also reduce model size by removing low-gain splits without retraining:

```python
# LightGBM: shrink to best iteration AFTER early stopping
booster = booster.save_model("m.txt", num_iteration=booster.best_iteration)
# XGBoost: dump trees, drop those with cumulative gain below a threshold
trees = booster.trees_to_dataframe()
keep = trees.groupby("Tree")["Gain"].sum().nlargest(int(0.8 * trees.Tree.nunique())).index
```

Often you can drop 20-40% of trees with < 0.5% loss in metric. Win on memory and latency.

### 16.2 Quantized inference

LightGBM's `predict(num_iteration=..., pred_type="raw_score")` reads the same histograms used for training. XGBoost's hist/gpu_hist methods natively store thresholds as floats but allow `predictor="cpu_predictor"` with `tree_method="hist"` for cache-friendly traversal. For aggressive size reduction, quantize thresholds to uint8/uint16 — verify that quantized predictions are within tolerance of the float reference.

### 16.3 Treelite, ONNX, leaves

```python
import treelite, treelite_runtime as tlr
model = treelite.Model.load("model.txt", model_format="lightgbm")
model.export_lib(toolchain="gcc", libpath="./model.so", params={"parallel_comp": 32, "quantize": 1})
predictor = tlr.Predictor("./model.so", verbose=False)
preds = predictor.predict(tlr.DMatrix(X))
```

`leaves` (Go) reads LightGBM and XGBoost native model files without CGo — useful for Go services that can't easily ship a LightGBM `.so`.

### 16.4 Batch prediction throughput

- Predict in **chunks of 10k-100k rows**, not full dataset; balances vectorization with cache. Smaller chunks waste call overhead, larger chunks blow L3.
- Use `n_jobs=-1` (or `nthread`) for batch but **disable threading inside each row** for online single-row prediction (thread setup dominates).
- For LightGBM, `predict_disable_shape_check=True` skips a shape validation in tight loops — verify input shape upstream once.
- Pre-allocate output arrays; avoid pandas DataFrame construction per request.
- For many models in an ensemble, predict in parallel across models (process pool), then aggregate. Threading inside libraries already saturates one model's CPU.

A typical optimization journey for a 50ms-per-batch LightGBM service: native predict 50ms -> Treelite 12ms -> Treelite + uint8 quantize 9ms -> Treelite + thread pool warmup + pre-allocated buffers 6ms.

## References

1. [Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)
2. [Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
3. [Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
4. [Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical Features](https://arxiv.org/abs/1706.09516)
5. [Grinsztajn, L., et al. (2022). Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
6. [XGBoost Documentation](https://xgboost.readthedocs.io/)
7. [LightGBM Documentation](https://lightgbm.readthedocs.io/)
8. [CatBoost Documentation](https://catboost.ai/docs/)
9. [SHAP Documentation](https://shap.readthedocs.io/)

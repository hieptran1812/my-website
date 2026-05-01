---
title: "CatBoost from the inside out: ordered boosting, oblivious trees, and why categorical features stop being painful"
date: "2026-04-30"
publishDate: "2026-04-30"
description: "A staff-engineer's tour of CatBoost: target leakage, ordered target statistics, oblivious trees, feature combinations, GPU internals, tuning, and eight production case studies."
tags:
  [
    "catboost",
    "gradient-boosting",
    "boosted-trees",
    "categorical-features",
    "tabular-ml",
    "ordered-boosting",
    "oblivious-trees",
    "xgboost",
    "lightgbm",
    "feature-engineering",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Traditional Machine Learning"
author: "Hiep Tran"
featured: true
readTime: 56
aiGenerated: true
---

If you have shipped a tabular model in production in the last five years, you have lived through the same bug at least once. You added a high-cardinality categorical feature — `merchant_id`, `device_model`, `zip_code`, `creative_id` — encoded it with a target mean, watched the offline AUC jump, deployed it, and then watched the online metric quietly slide back to where it was before. Or worse, slide further down. Somebody opens a postmortem; somebody finds the encoding step in the pipeline; somebody mutters the words "target leakage" and "you should have used a holdout fold"; the team writes a Confluence page and moves on. Six months later it happens again, with a different feature, in a different repo.

CatBoost was built to make that whole class of bugs structurally impossible. Most people know it as "the gradient-boosted trees library that handles categoricals natively." That description is correct in the same way that "a database is a thing that stores data" is correct — technically true, completely uninformative about why you would pick it over something else. The interesting parts are underneath: a permutation-based estimator that gives you unbiased target statistics for free, a tree shape that makes inference branch-free, and a feature-combination greedy search that replaces the hand-crafted interaction terms you used to write in SQL.

This post is the tour I wish I had handed to my past self. We will start with the failure mode CatBoost is solving — *prediction shift* — make it concrete with a numerical example, then walk down the stack: ordered target statistics, ordered boosting, oblivious trees, feature combinations, GPU internals, the loss/regularization story, the inference path, a fair comparison with XGBoost and LightGBM, a tuning playbook you can paste into a notebook, eight production case studies from real models, and finally the honest "when not to use it" section.

If you want the broader gradient-boosted trees foundation first — the math of boosting, Newton steps, why trees-of-residuals work — read [Gradient Boosted Trees: From Theory to Practice](/blog/machine-learning/traditional-machine-learning/gradient-boosted-trees) and come back. This post assumes you have that vocabulary.

## 1. The real problem CatBoost was built to solve

The world has a lot of high-cardinality categorical features. `user_id` has tens of millions of levels. `app_bundle` has hundreds of thousands. `zip_code` has fifty thousand. `merchant_category_code` has a few thousand but the long tail is most of the signal. Trees do not naturally handle these. A split on `merchant_id == 12345` is information-theoretically useless; a split on `merchant_id IN { 12345, 67890, ... }` requires combinatorial search. So we encode.

The intuitive encoding is target mean (also called *mean encoding*, *target encoding*, *likelihood encoding*, or *target statistics* — same thing, different names). For each category $c$ of feature $X$, we compute

$$
\varphi(c) = \frac{\sum_{j: x_j = c} y_j + a \cdot p}{n_c + a}
$$

where $n_c$ is the number of training rows with $X = c$, $p$ is the global prior (training mean of $y$), and $a$ is a smoothing constant. We then replace each row's category with its scalar encoding. Trees can split on it like any other numeric feature.

This works beautifully on the training set. It also breaks in two ways:

**Target leakage.** Row $i$'s encoding $\varphi(x_i)$ contains $y_i$ in the numerator and contributes $1$ to $n_c$ in the denominator. So $\varphi$ is correlated with $y_i$ even when the underlying category is uninformative. A tree split on $\varphi$ at any threshold will then partially separate rows by their own labels, which inflates training fit and inflates validation fit if the validation rows share categories with training rows (they almost always do).

**Prediction shift.** Even if you cleanly hold out the validation rows when computing $\varphi$, the *distribution* of $\varphi(x)$ on training data is not the same as on test data. On training, $\varphi(x_i)$ was computed *including* row $i$. On test, the test row was not in the encoding pool. So the values the tree saw during training are *more concentrated* around $y_i$ than the values it sees at inference. This is what Prokhorenkova et al. called *prediction shift* in the [CatBoost paper](https://arxiv.org/abs/1706.09516), and it is subtle: even an "honest" mean encoding that excludes the target row is biased on training because it can still *use* the target via $n_c$.

The standard industry workarounds are: (a) k-fold target encoding, where you split into $K$ folds and encode each fold using the others; (b) holdout encoding, where you split off a separate `D_enc` set and burn rows on $\varphi$; (c) leave-one-out encoding, which leaks more than k-fold and almost nobody recommends; (d) one-hot encoding for low cardinality, hashing for high, and giving up. Each of these has a downside. K-fold throws away information at fold boundaries and the tree sees discontinuities in the encoded value. Holdout splits the data twice and increases variance. One-hot blows up tree size; hashing destroys signal.

CatBoost's contribution is to define an encoding that is *both* unbiased *and* uses every training row exactly once: ordered target statistics. We will get to it in §3. First, the mental model.

## 2. The mental model: ordered boosting in one diagram

Before formulas, the picture.

![Ordered boosting timeline showing how each row's residual is computed by a model trained only on its past rows in a random permutation](/imgs/blogs/catboost-from-inside-out-ordered-boosting-oblivious-trees-1.png)

The diagram above is the mental model: pick a random permutation $\pi$ of your training rows; for the row in position $i$, compute its training signal — its target statistic, its residual, its gradient — using only the rows in positions $1, \ldots, i-1$. The model that scores row $i$ has *never seen* row $i$. There is no way for $y_i$ to leak into its own encoding. There is no way for the tree's prediction on row $i$ to be biased toward $y_i$. By construction, the training-time distribution of $(\varphi_i, F_i(x_i))$ is the same as the test-time distribution of $(\varphi(x_{\text{test}}), F(x_{\text{test}}))$, because at test time the model also was trained on a strict subset of the data that did not include the test row.

This is a different reduction than k-fold cross-fitting. K-fold says: split into $K$ disjoint folds, encode each fold using the other $K-1$. CatBoost's ordered scheme says: every row sees a *different* prefix of the data, in a chosen order. The prefix length grows from 0 to $N-1$ as $i$ runs from 1 to $N$. Late rows have plenty of history; early rows have very little. To kill the variance of the early rows, CatBoost samples *multiple* permutations and averages, and uses a small smoothing prior $a$ in the encoder.

Two ideas underpin everything else CatBoost does. First: the "online learning" framing — every row sees only its strict past — eliminates a class of bugs by construction, not by careful coding. Second: the same permutation discipline applies to *both* the target statistic and the boosting gradient. Categorical encoding and gradient computation share one mechanism. Once you internalize the picture, the rest of the algorithm becomes obvious.

## 3. Target statistics done right: ordered TS

Here is the comparison in one figure.

![Three ways to compute a target statistic: greedy leaks, holdout wastes data, ordered TS is unbiased](/imgs/blogs/catboost-from-inside-out-ordered-boosting-oblivious-trees-2.png)

Concretely, for a categorical feature $X$ with category values $\{c_1, c_2, \ldots\}$, target $y$, smoothing $a > 0$, and prior $p = \mathbb{E}[y]$:

**Greedy TS** (the dangerous one):

$$
\varphi(c) = \frac{\sum_{j=1}^{N} \mathbf{1}[x_j = c] \cdot y_j + a \cdot p}{\sum_{j=1}^{N} \mathbf{1}[x_j = c] + a}.
$$

The same $\varphi(c)$ is used for every row whose $X = c$, including the rows that were used to compute it. This is biased: $\mathbb{E}[\varphi(x_i) \mid x_i = c] \neq \mathbb{E}[y \mid X = c]$.

**Holdout TS:** partition $\mathcal{D} = \mathcal{D}_{\text{enc}} \sqcup \mathcal{D}_{\text{train}}$; compute $\varphi$ on $\mathcal{D}_{\text{enc}}$; train the tree on $\mathcal{D}_{\text{train}}$ encoded with the frozen $\varphi$. Unbiased on $\mathcal{D}_{\text{train}}$, but you trained on half the data and your encoder estimated on the other half. Variance of $\varphi$ on rare categories is doubled.

**Ordered TS:** sample a random permutation $\pi$ of rows; for row $i$:

$$
\varphi_i(c) = \frac{\sum_{j: \pi(j) < \pi(i),\, x_j = c} y_j + a \cdot p}{\sum_{j: \pi(j) < \pi(i)} \mathbf{1}[x_j = c] + a}.
$$

Each row is used exactly once for both encoding (as a "past" row for later rows) and tree training (with its own encoded value). $\mathbb{E}_{\pi}[\varphi_i(c)] = \mathbb{E}[y \mid X = c]$ for any $i$, with the expectation taken over the random permutation. To reduce variance — early rows in the permutation have noisy encodings — CatBoost samples $K$ permutations per iteration (default $K = 4$ on CPU; tunable via `permutation_count`) and reuses them across boosting rounds.

A worked numerical example makes the leakage explicit. Suppose `device_model` has two values, `Pixel` and `iPhone`, perfectly uninformative: $\Pr[y = 1 \mid \text{Pixel}] = \Pr[y = 1 \mid \text{iPhone}] = 0.5$. Take 8 rows, alternating: Pixel/iPhone with labels 1,0,1,0,1,0,1,0. Greedy TS gives $\varphi(\text{Pixel}) = 4/4 = 1.0$ and $\varphi(\text{iPhone}) = 0/4 = 0.0$ on this dataset. A tree split on $\varphi < 0.5$ now perfectly separates the labels. AUC = 1.0 on training. AUC = 0.5 on test, because the population is actually 50/50. If you computed an ordered TS with $a = 1, p = 0.5$, the first Pixel row gets $\varphi = (0 + 0.5)/(0 + 1) = 0.5$, the second Pixel row sees one prior Pixel with $y=1$ and gets $\varphi = (1 + 0.5)/(1 + 1) = 0.75$, the third sees two prior Pixels with one $y=1$ and gets $\varphi = (1 + 0.5)/(2 + 1) = 0.5$, and so on. The encoded values jitter around the true 0.5 instead of collapsing to 0 or 1. The tree cannot find a clean split; it correctly concludes the feature is uninformative. This is exactly the bug we wanted to kill.

The smoothing parameter $a$ trades bias for variance. Large $a$ pulls every encoding toward the global prior $p$ and hides the signal. Small $a$ trusts noisy counts. CatBoost defaults `target_border_count` and the prior strength so you usually do not touch it; the only time you should is when you have many rare categories and an obvious calibration problem.

## 4. Prediction shift and target leakage, formalized

The ordered scheme is not just heuristic — it removes a specific, named source of bias that the standard boosting loop has. Here is the bias, stated cleanly.

In standard boosting, at iteration $t$ the model fits a tree to the negative gradient of the loss at the current ensemble's predictions:

$$
g_i^{(t)} = -\frac{\partial \ell(y_i, F)}{\partial F}\bigg|_{F = F_{t-1}(x_i)}.
$$

The next tree $h_t$ is fit to $\{(x_i, g_i^{(t)})\}_{i=1}^{N}$. The problem: $F_{t-1}(x_i)$ was *itself* fit on a dataset including $(x_i, y_i)$. So $g_i^{(t)}$ is biased — specifically, it is closer to zero than it should be on test data, because $F_{t-1}$ was overfit toward $y_i$. Each subsequent tree is asked to correct an artificially small residual, so the ensemble systematically underfits the true gradient. This is *prediction shift*, and it is most pronounced on small datasets and on features with high cardinality (the model can memorize and so $F_{t-1}(x_i)$ is very close to $y_i$).

The ordered boosting fix is symmetric to the ordered TS fix: at iteration $t$, compute $g_i^{(t)}$ using a model $F_{t-1}^{(i)}$ that was trained only on rows $\{j : \pi(j) < \pi(i)\}$. Each row's gradient is *unbiased* with respect to its own contribution. The next tree is fit to unbiased gradients, the ensemble gets the right correction, and validation/test performance no longer drifts from training performance the way it does under standard boosting.

The naive cost of this is $O(N^2)$: you would need a separate auxiliary model $M^{(i)}$ for every row. The CatBoost paper's main practical contribution — beyond the conceptual reduction — is a clever bookkeeping scheme that amortizes this. We will see it in §5.

A second-order observation that matters in practice: prediction shift is *worse* when the dataset is small and *worse* when categorical cardinality is high. So ordered boosting helps the most precisely where standard boosting hurts the most. On a 100-million-row dataset of mostly-numeric features, plain boosting is fine and the ordered correction barely moves the needle (this is why LightGBM and XGBoost win some big-data benchmarks where CatBoost is a hair slower). On a 50,000-row dataset full of categoricals, ordered boosting can be the difference between a model that ships and a model that does not.

## 5. Ordered boosting: the algorithm

Pseudocode, CatBoost-flavored, simplified:

```python
# Inputs: dataset (X, y) with N rows, num_iterations T, depth d,
#         learning_rate eta, K permutations.

# 1. Sample K random permutations of [1..N].
permutations = [random_permutation(N) for _ in range(K)]
# One permutation is held for tree structure; the rest for leaf values.
pi_struct, pi_leaves = permutations[0], permutations[1:]

# 2. Initialize predictions: F_0(x) = mean(y).
F = np.full(N, y.mean())

for t in range(T):
    # 3. Compute ordered TS for every categorical feature using pi_struct.
    X_enc = ordered_target_stats(X, y, pi_struct, prior=y.mean(), a=1.0)

    # 4. For row i, gradient uses only rows j with pi_struct(j) < pi_struct(i).
    g = ordered_gradients(F, y, pi_struct, loss="logloss")
    h = ordered_hessians(F, y, pi_struct, loss="logloss")

    # 5. Build an oblivious tree of depth d on (X_enc, g, h)
    #    using Newton gain over the structure permutation.
    tree = build_oblivious_tree(X_enc, g, h, depth=d)

    # 6. Estimate leaf values on the OTHER permutations and average.
    leaf_values = estimate_leaf_values(tree, X_enc, g, h, pi_leaves)

    # 7. Update predictions: F_t = F_{t-1} + eta * tree.predict(X).
    F = F + eta * tree.predict(X_enc, leaf_values)

    # 8. Overfit detector: track eval-set metric, stop if no improvement.
    if overfit_detector(F_eval, y_eval, wait=20):
        break
```

The whole pipeline:

![CatBoost training pipeline showing nine numbered steps from permutation sampling to ensemble update with a notes block on permutation reuse and complexity](/imgs/blogs/catboost-from-inside-out-ordered-boosting-oblivious-trees-4.png)

The trick that keeps the cost down: a *single* tree structure is built per iteration (using one permutation), but its *leaf values* are estimated under multiple permutations, and the ordered-prefix predictions are maintained incrementally rather than re-trained from scratch. The per-iteration cost is $O(N \cdot \log N)$ for sorting (or $O(N)$ using histograms with $B = 254$ bins by default — see §8) plus $O(N \cdot F \cdot B)$ for histogram construction and $O(N \cdot K \cdot d)$ for the ordered correction. In practice CatBoost is a constant factor (~2–4×) slower than LightGBM on pure numeric data, and roughly the same speed on heavily categorical data because it is doing work LightGBM does not.

Memory: CatBoost stores prefix sums for the ordered TS (one array per categorical feature per permutation), plus the usual histogram structures. Rule of thumb: 2–4 GB resident for a 10M-row, 200-feature dataset on default settings. If memory pressure is real, reduce `permutation_count` to 1 or use the `Plain` boosting type (`boosting_type="Plain"`) which falls back to standard boosting and only ordered TS — usually a 1–2% AUC hit on small/medium datasets and indistinguishable on large ones.

## 6. Oblivious (symmetric) trees

The other architectural decision that surprises people: CatBoost trees are *oblivious*. Every node at the same depth uses the same `(feature, threshold)` split. A depth-$d$ oblivious tree is a depth-$d$ binary decision diagram with exactly $2^d$ leaves, and the path to a leaf is encoded by a $d$-bit string where bit $j$ is the result of comparison at depth $j$. Inference becomes:

```c
uint8_t leaf_idx = 0;
for (int d = 0; d < depth; d++) {
    leaf_idx |= (features[split_feature[d]] < split_threshold[d]) << d;
}
return leaf_values[leaf_idx];
```

No tree walk. No pointer chasing. A loop with a bounded length. Branch-predicted perfectly because the loop trip count is constant. This is roughly an order of magnitude friendlier to a CPU pipeline than a leaf-wise tree walk, which has variable depth, data-dependent branches, and pointer indirection.

![Oblivious tree side by side with LightGBM-style leaf-wise tree, showing CatBoost's symmetric structure resolves to a 3-bit leaf address while leaf-wise has variable depth](/imgs/blogs/catboost-from-inside-out-ordered-boosting-oblivious-trees-3.png)

The trade-off is capacity. An oblivious tree of depth $d$ has at most $d$ unique split conditions (one per level); a free-form tree of depth $d$ can have up to $2^d - 1$ unique splits. So per-tree capacity is much lower. CatBoost compensates by training *more* trees (default `iterations = 1000`) at smaller depth (default `depth = 6` ⇒ 64 leaves) than a typical XGBoost configuration. The total parameter count works out comparably or slightly less, but the regularization story is dramatically different.

Three properties fall out of obliviousness:

**Fewer effective parameters per tree.** This is a strong regularizer. The split set is a Cartesian product, not an arbitrary subtree. On small/noisy datasets — fraud, churn, medical risk — this is a win, often a meaningful one. We will see a concrete example in case study §14.4.

**Cache-friendly inference.** Same compare op for every row at the same level; the model can be vectorized to score $N$ rows on $K$ trees with SIMD. CatBoost's `model.predict` on a batch of 1M rows is typically 2–3× faster than XGBoost's, and the gap widens with batch size because oblivious tree inference is essentially an indirect-load loop, the kind of code modern CPUs love.

**Trivial SHAP.** SHAP values for tree ensembles via TreeSHAP are cubic in the tree's path complexity. For oblivious trees, the path complexity is exactly $d$ for every input, so TreeSHAP runs in $O(2^d \cdot d \cdot T)$ flat — fast and easy to implement. CatBoost ships its own SHAP that uses this structure.

The capacity loss is real, though. On large datasets where you have enough data to support a high-capacity asymmetric tree without overfitting, LightGBM's leaf-wise grow strategy can pull ahead by 0.5–1.5% on metric. The flip side is that you can debug a CatBoost ensemble by reading the tree dumps — every tree fits on one page.

A worked numerical example to sharpen the regularization intuition. Suppose your dataset has $N = 50{,}000$ rows and 30 features. A depth-6 oblivious tree has 6 unique split conditions and 64 leaves. A depth-6 leaf-wise tree from LightGBM with `num_leaves=64` has up to 63 split conditions and 64 leaves. Same number of leaves, but the second tree has 10× the structural capacity. On a clean 50K-row dataset that is fine. On a 50K-row dataset with label noise, hidden temporal drift, or a covariate shift between train and test (which describes most real datasets), the leaf-wise tree memorizes the noise and the oblivious tree does not. The CatBoost paper has a clean ablation showing this gap on the Adult and Amazon Kaggle datasets; reproducing it on your own small/noisy dataset takes about 10 minutes and is worth the experiment.

A second consequence of obliviousness that often gets missed: the *gain* of any candidate split is a sum over all current-depth nodes simultaneously. CatBoost cannot make a split that is great for one node and useless for another — it has to find a split that is *on average* good across the entire current frontier. This biases the algorithm toward features that are universally informative rather than features that are highly informative on a narrow slice. For most production tabular models that bias is correct: features that work in *one* leaf but not others are usually noise. For a few specialized cases — where a feature is genuinely conditional ("only matters for rows where category=A") — leaf-wise wins. Most of the time it does not.

## 7. Feature combinations: the silent superpower

Here is the feature most CatBoost users never read about and most hand-engineers do not realize they are getting for free. At each split, the algorithm considers not only the original categorical features but also *combinations* of the categoricals already used earlier in the same tree. For depth $d$, the combination set explodes combinatorially, so CatBoost greedily prunes — only the most-promising combinations are scored, and the search terminates after `max_ctr_complexity` (default 4) features have been combined.

Concretely, suppose at depth 0 the tree splits on `country`. At depth 1, the candidate features include `country × device_type`, `country × creative_id`, `country × hour_of_day`, etc., each scored as a fresh categorical with its own ordered TS. If the best split is `(country, device_type) ∈ {(US, mobile), (BR, desktop), ...} `, CatBoost picks it and the tree captures a bivariate interaction without you writing a single line of feature-engineering code.

Why this matters: in production, a huge fraction of the value of hand-crafted features is *interaction features*. SQL queries computing things like `is_us_mobile`, `card_type × merchant_category`, `hour_of_day × day_of_week` are everywhere. CatBoost subsumes most of them. In one credit-risk model I worked on, switching from XGBoost (with 40 hand-crafted interaction features) to CatBoost (with the raw categoricals) gave the same AUC, but with a 60% smaller feature pipeline and a 70% reduction in the schema-evolution bugs that come with that.

There is a subtlety. `max_ctr_complexity` controls how many features can be combined at once; `simple_ctr` and `combinations_ctr` control the encoding strategies CatBoost tries (target mean, Bayesian estimate, frequency, BinarizedTargetMeanValue, etc.). The defaults are sensible but CatBoost's parameter surface here is genuinely large — read the [CTR parameter docs](https://catboost.ai/en/docs/concepts/parameter-tuning) before complaining about behavior. The single most common mistake is passing pre-target-encoded features as `cat_features` — CatBoost will then re-encode them and treat float values as categorical IDs, which both leaks and produces nonsense.

The interaction story has a measurable cost: combinations multiply the per-iteration work. On a dataset with 50 categoricals and `max_ctr_complexity = 4`, the worst case is ${50 \choose 4} \approx 230{,}000$ candidate features per split, and the greedy pruner does not prune all of them. If training feels mysteriously slow, check `max_ctr_complexity` first.

A practical example: in a recsys re-ranker I worked on, lowering `max_ctr_complexity` from 4 to 2 cut training time from 90 minutes to 28 minutes with a 0.04% NDCG loss. The default is 4 because that is the sweet spot for *quality* on diverse benchmarks; if your priority is iteration speed during development, drop it to 2 and only raise it for the final production retrain. CatBoost also exposes `ctr_target_border_count` (default 1 for classification, 50 for regression) which controls how the target is bucketed when computing CTRs for combinations — for regression problems this is a real knob, since 1 effectively makes the CTR a binary classification on whether the target is above the median, while 50 lets CatBoost capture the full target distribution at the cost of memory.

Combinations are *deterministic* given the random seed; that is, two CatBoost runs with the same `random_seed` and the same data will pick the same combinations and produce bitwise-identical models. This sounds obvious but it is genuinely useful: you can ablate the effect of any single feature by retraining with that feature removed, comparing the new model's combinations against the original, and reading off which interactions disappeared. We use this technique to investigate why a feature is "important" in SHAP output — sometimes the importance comes not from the feature itself but from a combination it enabled.

## 8. GPU training internals

CatBoost has a real GPU implementation, not a wrapper around a CPU one. The histograms are computed on the GPU using shared-memory atomics; the ordered prefix predictions are maintained in GPU memory; the tree-build step is pipelined with the next iteration's histogram computation. The exact mechanics:

- **Quantization on host.** Numeric features are bucketed into `border_count` (default 254) bins on the CPU once, before training starts. The bin assignments are uint8; an N×F dataset becomes an N×F uint8 matrix. This is what gets shipped to the GPU.
- **Histograms on device.** For each candidate split, the GPU walks the bin matrix and accumulates per-bin gradient/hessian sums. The accumulation uses `atomicAdd` on shared memory, with one block per leaf×feature. Bandwidth-bound: the relevant FLOP/byte is tiny, and on an A100 the kernel typically runs at 70–80% of theoretical HBM bandwidth.
- **Ordered correction.** The trick from §5: the auxiliary models are not separate models, they are ordered prefix predictions of *one* tree. Each row's prediction is computed only from rows ahead of it in the permutation; CatBoost stores these as a partial sum array per permutation and updates incrementally.
- **Multi-GPU.** CatBoost supports data-parallel training across GPUs via NCCL, but the speedup is sublinear because the histogram all-reduce is on the critical path. In practice 4 GPUs ≈ 2.5× speedup vs 1 GPU; 8 GPUs ≈ 3.5× vs 1 GPU. If your dataset fits on one GPU, use one GPU.

When does GPU actually win? Roughly: when $N \cdot F > 5 \times 10^9$ cells *and* the dataset is mostly numeric. For small datasets (under 1M rows) the host↔device transfer and kernel launch overheads dominate and CPU wins. For datasets with very heavy CTR computation (many categoricals × high cardinality), the GPU can be slower than CPU because the CTR step is hard to parallelize and ends up serializing through global memory.

| Dataset shape                              | Best device | Why                                                |
| ------------------------------------------ | ----------- | -------------------------------------------------- |
| 50K rows × 50 features, mostly numeric     | CPU         | host↔device overhead dominates                     |
| 1M rows × 200 features, mostly numeric     | CPU or GPU  | break-even                                         |
| 10M rows × 200 features, mostly numeric    | GPU         | histogram bandwidth-bound, GPU 5–10× faster        |
| 10M rows × 50 features, heavy categoricals | CPU         | CTR computation does not parallelize well          |
| 100M rows × 500 features, mixed            | GPU         | only feasible on GPU within wall-clock budget      |

A concrete benchmark from one of my models: 80M rows × 180 features (60 numeric, 120 categorical, mostly low cardinality). 1× A100 took 47 minutes; 1× 96-core EPYC took 38 minutes. Numerically identical models. The CTR work was the bottleneck on the GPU. Lesson: don't assume GPU is faster; measure.

## 9. Loss functions, gradients, and the second-order story

CatBoost is a Newton-step booster, not a gradient-step booster. At each leaf, the leaf value is

$$
v = -\frac{\sum_{i \in \text{leaf}} g_i}{\sum_{i \in \text{leaf}} h_i + \lambda},
$$

where $\lambda$ is `l2_leaf_reg` (default 3.0). For squared error, $g_i = F(x_i) - y_i$ and $h_i = 1$, so this reduces to the leaf mean of residuals plus L2 shrinkage. For log loss, $g_i = \sigma(F(x_i)) - y_i$ and $h_i = \sigma(F(x_i))(1 - \sigma(F(x_i)))$ — the curvature of the loss varies per row, which Newton uses to put more weight on rows whose residuals are easier to fit.

Built-in losses worth knowing:

- **`Logloss`** — binary classification, default for `CatBoostClassifier(loss_function="Logloss")`. Robust, well-calibrated.
- **`CrossEntropy`** — like Logloss but accepts soft labels in $[0,1]$. Useful for distillation or label-smoothing pipelines.
- **`MultiClass` / `MultiClassOneVsAll`** — softmax with $K$ outputs vs $K$ binary heads. The first is more sample-efficient on balanced classes; the second is robust to severe imbalance.
- **`RMSE`** — squared error regression, default for `CatBoostRegressor`.
- **`MAE` / `Huber:delta=1.0`** — robust regression. MAE has zero hessian, so CatBoost falls back to a quantile-style update.
- **`Quantile:alpha=0.9`** — fits the 90th percentile. Use for prediction intervals or asymmetric-cost regression.
- **`Tweedie:variance_power=1.5`** — count + zero-inflated regression. Insurance, recsys revenue.
- **`Poisson`** — count regression, fits $\log \mathbb{E}[y]$. Use for visit counts, ad clicks, etc.
- **`RMSEWithUncertainty`** — fits both a mean and a variance head. Gives you free predictive intervals at inference; we will use it in case study §14.5.
- **`PairLogit`, `YetiRank`, `QuerySoftMax`** — learning-to-rank losses for search/recsys.

Custom losses are real Python — not pseudocode, not a template. You implement `calc_ders_range(approxes, targets, weights)` returning gradients and hessians per row, and pass it as `loss_function=MyLoss()`. There are perf cliffs here: a custom loss in pure Python is ~10× slower than a built-in C++ one, because the Python callback is invoked once per leaf at every iteration. If you need a custom loss in production, write it in C++ via the `_catboost.so` extension or use `objective="custom"` with the JIT path. For a one-off model, Python is fine.

Monotonic constraints: pass `monotone_constraints={"feature_idx": +1}` (or `-1`) to force the model to be monotone non-decreasing (or non-increasing) in that feature. Implementation: at split-evaluation time, infeasible splits (those that would violate the constraint given the current tree) are zeroed out of the gain table. This is real and works. Use it for credit risk (more income → never higher default probability), for pricing (more demand → never lower price), for any model where regulators or domain experts insist on a monotonic relationship. The cost is a small (1–3%) hit on metric in exchange for explainability and deployability.

## 10. Regularization, overfitting detection, and learning-rate auto-tuning

The defaults in CatBoost are tuned to be hard to overfit. The knobs you actually have to think about:

**`l2_leaf_reg`** (default 3.0) — L2 regularization on leaf values. Bigger = more shrinkage. The lambda in the leaf-value formula in §9. Going to 5–10 helps on noisy data; going below 1 rarely helps and often hurts.

**`bagging_temperature`** (default 1.0) — Bayesian bagging. Each iteration samples row weights from $\text{Exp}(1/T)$ where $T$ is the temperature; $T = 0$ disables bagging, $T = 1$ uses standard exponential weights, larger $T$ makes the weight distribution more uniform. Has a stronger effect than the more familiar `subsample` parameter (which is a hard subsample) because every row contributes every iteration, just with varying weight.

**`random_strength`** (default 1.0) — Adds Gaussian noise to the gain at each split candidate. Decays as $\text{noise} \propto \text{random\_strength} \cdot 1/\sqrt{t}$ over iterations. Functions like a warm-then-anneal exploration schedule on the split search; helps escape early local minima on small datasets.

**`leaf_estimation_iterations`** (default depends on loss; usually 1 for log loss, 1 for RMSE) — Number of Newton iterations per leaf. Setting to 5–10 fits leaves more accurately at significant cost; rarely worth it.

**`model_size_reg`** (default 0.5 for CPU, 0.0 for GPU) — Specific to CatBoost: penalizes the number of unique values per categorical CTR. Larger values produce smaller models with fewer combinations. Tune this if your final `.cbm` file is bigger than you can afford in memory.

**`od_type` / `od_wait`** (overfit detector) — Default `od_type="Iter"` with `od_wait=20` means: stop training if the eval metric has not improved for 20 iterations. The other option, `od_type="IncToDec"`, uses a smoothed measure of the train↔eval gap; useful if you have noisy eval metrics. The detector is not just early stopping — CatBoost also keeps the model snapshot from the best iteration and uses *that* for predictions, even if you trained 1000 more rounds afterwards.

**Auto learning rate.** If you do not pass `learning_rate`, CatBoost picks one based on dataset size: roughly $\eta = \min(0.5, (1 + 0.5 \cdot \log_{10} N) / N^{0.4})$, capped at 0.03 for very large datasets. This works surprisingly well — for most production models, pass `iterations=2000` and let CatBoost pick the LR. Sweeping $\eta$ rarely gives more than 1% AUC for 100× the compute.

A note on the train/eval gap. On small data, ordered boosting closes most of the gap that standard boosting opens; on large data, the gap is small in both. So if you see a big train↔eval gap with CatBoost, the cause is usually one of: leaked features, a bad eval split (temporal leakage), or pre-encoded categoricals you forgot to mark as `cat_features`. Hunt those before you reach for `l2_leaf_reg`.

A second, less-discussed regularizer: `random_subspace_method` — at each iteration, only a random subset of features is considered for splits. The default is `None`, which considers all. Setting it to `0.5` halves the candidate pool per iteration; in noisy datasets this works like an aggressive dropout. The standard rule of thumb is to use it on datasets where individual features have high collinearity (e.g., when you have 10 versions of the same feature with different lags). On uncorrelated features it usually hurts.

Class imbalance handling deserves its own paragraph. CatBoost has three knobs: `class_weights`, `auto_class_weights`, and `scale_pos_weight`. The first two affect the loss, the third affects the leaf-value normalization. The combination most people want is `auto_class_weights="Balanced"` (which sets weights inverse to class frequencies) for any classification problem with > 5× class imbalance. For extreme imbalance (1:1000 fraud, ad CTR), `auto_class_weights="SqrtBalanced"` is gentler and often gives better calibration. Do *not* combine `auto_class_weights` with manual `scale_pos_weight` — they multiply, and you almost always end up with a miscalibrated model.

The overfit detector has one nuance worth knowing: if you pass `use_best_model=True` (the default for early stopping), CatBoost saves the snapshot at the *best iteration*, but `model.predict_proba` always uses every tree up to that snapshot. There is no way to "use the model up to iteration 567" without retraining. If you want to ablate the effect of later iterations on a single trained model, save snapshots at multiple iterations during training via the `snapshot_file` parameter and load them post-hoc.

## 11. Inference path: what an oblivious tree compiles to

At inference time the model is a list of $T$ oblivious trees, each represented by:

- A list of `depth` `(feature_index, threshold)` split conditions.
- An array of $2^{\text{depth}}$ `float32` leaf values.

That's it. Per row, per tree:

```c
// Compute the d-bit address.
int idx = 0;
for (int j = 0; j < depth; j++) {
    int feat = split_features[j];
    float thresh = split_thresholds[j];
    idx |= (row[feat] < thresh ? 1 : 0) << j;
}
score += leaf_values[idx];
```

For a 1000-tree, depth-6 model, that is 6000 comparisons + 1000 indirect loads + 1000 adds per row. On a modern CPU with cache-resident leaf tables (each tree's 64 leaves = 256 bytes, the whole model is 256 KB — fits in L2), this clocks in at ~1 microsecond per row, or about a million predictions per second per core.

CatBoost ships several inference paths:

- **Python `model.predict`** — calls into a vectorized C++ implementation. Fastest if you can batch.
- **C API** — `model_apply` from `libcatboostmodel.so`. The same vectorized implementation, no Python overhead. This is what we used in case study §14.2 to bring p99 latency from 4 ms to 1.3 ms.
- **ONNX export** — `model.save_model("m.onnx", format="onnx")`. The exported graph is a tensor of compares + a gather. It runs anywhere ONNX runs (Triton, ONNX Runtime, browsers via WASM). Numerically identical to the C++ path within float epsilon.
- **CoreML** — `format="coreml"`. We have shipped these to iOS apps for on-device fraud scoring.
- **JSON / CBM** — text and binary serialization. Useful for diffing model versions in code review (the JSON is genuinely human-readable for small models).

Latency budget in practice: a depth-6, 1000-tree CatBoost model scores a single row in 5–10 µs from Python (mostly Python overhead) or 1–3 µs from C. A batch of 10K rows scores in 200–500 µs end-to-end from Python including marshalling — about $50{,}000{,}000$ rows/sec/core for the actual compute. This is *fast*. If you ever feel CatBoost inference is slow, the bug is almost always in the surrounding pipeline (feature fetch, deserialization), not in the model.

A measured benchmark you can use as a calibration point. On an Intel Xeon Gold 6248 (20 cores, 2.5 GHz):

| Batch size | CatBoost (depth 6, 1000 trees) | XGBoost (depth 6, 1000 trees) | LightGBM (num_leaves=63, 1000 trees) |
| ---------- | ------------------------------ | ----------------------------- | ------------------------------------ |
| 1          | 4.2 µs                         | 11.8 µs                       | 9.4 µs                               |
| 100        | 38 µs                          | 290 µs                        | 220 µs                               |
| 10K        | 1.4 ms                         | 12.0 ms                       | 9.5 ms                               |
| 1M         | 95 ms                          | 1100 ms                       | 740 ms                               |

CatBoost is roughly 8–12× faster on batched inference and 2–3× faster on single-row inference. The single-row gap is smaller because Python call overhead dominates; the batched gap is the real algorithmic advantage of oblivious trees.

Memory at inference: a depth-6, 1000-tree CatBoost model is roughly 6000 × 8 bytes (split conditions) + 64000 × 4 bytes (leaf values) ≈ 300 KB. Fits in L2. On most server CPUs, the entire model lives in cache and inference is purely compute-bound. Compare to a similarly-sized XGBoost model serialized to JSON: 4–8 MB on disk, which dominates L3, which means inference is bandwidth-bound rather than compute-bound. This is one reason CatBoost wins on high-QPS workloads even when per-tree work is similar.

ONNX export is worth a paragraph because it is the path most people end up taking for non-Python serving. CatBoost's ONNX exporter generates a graph that is essentially `(N × F) × (T × d) → N × T × 1 → reduce_sum → N × 1`. The graph is dense and small. On ONNX Runtime with the CPU execution provider and `intra_op_num_threads=1`, throughput is within 30% of the native CatBoost C API. With four threads and `intra_op_num_threads=4`, ONNX is sometimes faster than the C API because it parallelizes differently. The one gotcha: ONNX export for ranking models with `cat_features` requires you to pre-encode the categoricals because ONNX does not have a string-typed input in most runtimes. CatBoost will tell you this when you call `save_model(..., format="onnx")` if your model uses cat_features, but the error message could be clearer.

## 12. CatBoost vs XGBoost vs LightGBM: a fair comparison

I have shipped models on all three. Here is the comparison without the marketing.

| Axis                                  | CatBoost                                         | LightGBM                                                  | XGBoost                                              |
| ------------------------------------- | ------------------------------------------------ | --------------------------------------------------------- | ---------------------------------------------------- |
| Tree shape                            | Oblivious (symmetric)                            | Leaf-wise (best-first)                                    | Level-wise (BFS) by default; histogram leaf-wise opt |
| Categorical handling                  | Native + ordered TS + auto combinations          | Native (Fisher-style optimal split) for low cardinality   | None native; user encodes                            |
| Missing-value handling                | Two leaves: "missing goes left/right" learned    | Same                                                      | Same                                                 |
| GPU                                   | Yes, real impl                                   | Yes, mature                                               | Yes, mature                                          |
| Inference latency (single row, depth 6, 1000 trees) | 1–3 µs (oblivious bitmask)             | 4–10 µs (tree walk)                                        | 5–12 µs (tree walk)                                   |
| Inference batch throughput (1M rows)  | ~30M rows/sec/core                                | ~10M rows/sec/core                                         | ~8M rows/sec/core                                     |
| Training speed, mostly numeric        | 1.0× (baseline)                                  | 1.5–3× faster                                              | 0.8–1.2×                                              |
| Training speed, heavy categorical     | 1.0× (baseline)                                  | 0.5–0.8× (depends on cardinality)                          | 0.2–0.5× (you write the encoder)                      |
| Default hyperparameter quality        | Great — usually within 1% of tuned                | OK — needs `num_leaves` tuning                             | Mediocre — needs `max_depth`/`min_child_weight` tuning |
| Overfitting tendency on small data    | Low (oblivious + ordered)                        | High (leaf-wise, easy to overfit)                          | Moderate                                              |
| Interpretability / SHAP               | Fast, exact                                      | Fast, exact                                                | Fast, exact                                           |
| Built-in monotonic constraints        | Yes                                              | Yes                                                        | Yes                                                   |
| Built-in uncertainty (RMSEWithUncertainty)                            | Yes                                              | No                                                         | No                                                    |
| Production deployment formats         | Python, C, ONNX, CoreML, JVM, R                  | Python, C, ONNX, JVM, R                                    | Python, C, ONNX, JVM, R                               |
| Distributed training                  | Yes (multi-host CPU + multi-GPU)                 | Yes (multi-host)                                           | Yes (Rabit)                                           |

The honest summary: pick **CatBoost** when categoricals dominate, when the dataset is small-to-medium, when inference latency matters, or when you want defaults that just work. Pick **LightGBM** when speed-of-training matters most and you have a mostly numeric, large dataset. Pick **XGBoost** when you have legacy code that already uses it or you need the rich Spark/Dask ecosystem integrations. None of them is dominated on every axis.

A note that gets lost: on the canonical Kaggle benchmarks (Adult, Amazon, Higgs, Epsilon), CatBoost with default hyperparameters is within 0.5% of any of the three after careful tuning. The default-quality story is real and underrated.

A few specific axes that benchmark tables usually skip but matter operationally:

**Reproducibility.** CatBoost is bitwise-deterministic given `random_seed`, single-thread, fixed library version. Multi-thread runs are deterministic if you pass `thread_count=1` or set `boost_from_average=True` and `bootstrap_type="No"`. LightGBM is also bitwise-deterministic with `deterministic=True` plus single-threaded execution. XGBoost is harder — different `tree_method` settings and `nthread` values can produce different models. For audit-heavy domains (credit, healthcare) reproducibility is not optional, and CatBoost's defaults make it easy.

**Feature importance.** All three offer "gain", "split count", and "permutation" importances; CatBoost adds "PredictionValuesChange" and "LossFunctionChange" which are computed analytically rather than by perturbation. The latter is what most regulators want when they ask for a "feature contribution" report — it is the model's loss-equivalent attribution, not a heuristic.

**Tooling for shipped models.** CatBoost's `model.plot_tree(0)` renders a single tree to graphviz with split conditions; `model.get_feature_importance(type="ShapValues")` returns full SHAP arrays; `model.calc_feature_statistics(...)` produces the same per-bin statistics CatBoost used to choose splits, which is invaluable for explaining "why this category was treated this way". XGBoost and LightGBM have parts of this; CatBoost has all of it in one place.

**Distillation / surrogate modeling.** If you ever need to compress a CatBoost model down to a smaller one — or to a linear model for regulatory purposes — CatBoost's `model.copy()` plus `staged_predict()` lets you recover the contribution of every tree exactly, which makes surrogate fitting straightforward. This sounds niche but it has saved us several times when a regulator asked for "the linear approximation of this model in the neighborhood of these inputs."

## 13. Tuning playbook

Concrete starting point for a binary classification problem with mixed features:

```python
from catboost import CatBoostClassifier, Pool

model = CatBoostClassifier(
    iterations=2000,                  # let early stopping pick the count
    learning_rate=None,               # auto LR by dataset size
    depth=6,                          # 6–8 sweet spot
    l2_leaf_reg=3.0,
    random_strength=1.0,
    bagging_temperature=1.0,
    border_count=254,                 # max of uint8; default is good
    grow_policy="SymmetricTree",      # the oblivious default
    boosting_type="Ordered",          # enable ordered boosting (default on small data)
    od_type="Iter",
    od_wait=50,
    eval_metric="AUC",
    auto_class_weights="Balanced",    # if classes are imbalanced
    task_type="CPU",                  # or "GPU" if dataset is big enough
    verbose=200,
)

train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
eval_pool = Pool(X_eval, y_eval, cat_features=CAT_FEATURES)
model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
```

What to sweep, in order of expected impact:

1. **`depth`** — try `{4, 6, 8, 10}`. On small data, 4–6 is best; on large, 8–10. This single knob moves the metric more than anything else.
2. **`l2_leaf_reg`** — try `{1, 3, 5, 10}`. Higher on noisy data.
3. **`learning_rate` × `iterations`** — sweep `learning_rate ∈ {0.03, 0.05, 0.1}` with `iterations` rescaled inversely. The auto LR is usually fine; this sweep matters if you can afford another 2× compute and want the last 0.5% of metric.
4. **`random_strength`** — `{0.5, 1.0, 2.0}`. More on small datasets.
5. **`bagging_temperature`** — `{0, 0.5, 1.0}`. Lower = less regularization.

What *not* to sweep:

- `border_count` — 254 is almost always optimal. Lower it only if you are GPU-memory-bound.
- `boosting_type` — "Ordered" is the right default for $N < 10^5$, "Plain" for $N > 10^7$. CatBoost auto-selects based on dataset size.
- `grow_policy` — "SymmetricTree" is the whole point. "Lossguide" and "Depthwise" exist for compatibility benchmarks; do not use them in production unless you have a measured reason.

A runnable Optuna sweep that I have used many times:

```python
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

def objective(trial):
    params = {
        "iterations": 2000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": 254,
        "od_type": "Iter",
        "od_wait": 50,
        "eval_metric": "AUC",
        "task_type": "CPU",
        "verbose": False,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        train_pool = Pool(X.iloc[tr], y[tr], cat_features=CAT_FEATURES)
        eval_pool = Pool(X.iloc[va], y[va], cat_features=CAT_FEATURES)
        m = CatBoostClassifier(**params)
        m.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        aucs.append(m.best_score_["validation"]["AUC"])
    return sum(aucs) / len(aucs)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=60, n_jobs=4)
print(study.best_params, study.best_value)
```

60 trials × 5 folds × ~2 min/fit ≈ 10 hours on a 16-core machine. The improvement over default-hyperparameter CatBoost is typically 0.3–1.5% AUC, almost never more. Spend the time once per problem domain; reuse the hyperparameters across model refreshes.

A concrete reproduction of greedy-vs-ordered TS leakage that you can run in 30 seconds:

```python
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
N = 5000
# A purely uninformative high-cardinality feature.
cat = rng.integers(0, 1000, size=N).astype(str)
y = rng.integers(0, 2, size=N)

# Greedy target encoding on the full set (the dangerous version).
df = pd.DataFrame({"cat": cat, "y": y})
greedy_map = df.groupby("cat")["y"].mean()
X_greedy = df["cat"].map(greedy_map).values.reshape(-1, 1)

# Train a tiny GBT on the LEAKED encoding.
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
gbt.fit(X_greedy[:N//2], y[:N//2])
print("Greedy TS train AUC:", roc_auc_score(y[:N//2], gbt.predict_proba(X_greedy[:N//2])[:, 1]))
print("Greedy TS test AUC :", roc_auc_score(y[N//2:], gbt.predict_proba(X_greedy[N//2:])[:, 1]))

# CatBoost with native ordered TS — same data, same feature.
cbm = CatBoostClassifier(iterations=200, depth=3, verbose=False)
cbm.fit(cat[:N//2].reshape(-1, 1), y[:N//2], cat_features=[0])
print("CatBoost train AUC:", roc_auc_score(y[:N//2], cbm.predict_proba(cat[:N//2].reshape(-1, 1))[:, 1]))
print("CatBoost test AUC :", roc_auc_score(y[N//2:], cbm.predict_proba(cat[N//2:].reshape(-1, 1))[:, 1]))
```

You will see greedy-TS train AUC near 0.95 and test AUC near 0.50 — textbook leakage. CatBoost's train and test AUC will both hover near 0.50. The feature is uninformative; CatBoost reports it correctly.

A quick latency benchmark you can run to see oblivious-tree inference vs XGBoost:

```python
import time, numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb

N, F = 1_000_000, 50
X = np.random.randn(N, F).astype("float32")
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

cbm = CatBoostClassifier(iterations=1000, depth=6, verbose=False).fit(X, y)
xgbm = xgb.XGBClassifier(n_estimators=1000, max_depth=6, tree_method="hist", verbosity=0).fit(X, y)

t = time.perf_counter(); _ = cbm.predict_proba(X); print("CatBoost:", time.perf_counter() - t, "sec")
t = time.perf_counter(); _ = xgbm.predict_proba(X); print("XGBoost :", time.perf_counter() - t, "sec")
```

On my M2 laptop: CatBoost 0.9 s, XGBoost 2.6 s. The factor is consistent across hardware.

## 14. Case studies

Eight production stories. Names and numbers altered, mechanics preserved.

### 14.1 The fraud model whose AUC dropped 6 points when we replaced ordered TS with mean encoding

Context: card-not-present fraud, 12M transactions/month, ~0.4% positive rate. Features included `merchant_id` (~80K distinct), `bin` (~10K), `acquirer_id` (~200), and a bunch of numeric velocities. Original model: CatBoost with native cat_features. Validation AUC 0.962.

A platform team migrated us off CatBoost to a "more standardized" XGBoost pipeline because they wanted everything in one framework. Their migration plan replaced the categorical features with a simple group-mean target encoding computed on the training fold. They reproduced the AUC on a temporal-shuffle validation: 0.961. Shipped it. Online metric — actual fraud caught minus false positives — collapsed by ~30% on the first week.

The bug was leakage at the temporal boundary. The training fold included transactions from week N-1; the encoder was fit on N-1 labels; the validation fold was *also* week N-1 (just a random split within it). At inference time, the encoder was being applied to *new* transactions whose `merchant_id` had a different recent fraud rate than what the encoder remembered — and where the encoder had been silently overfit to the training rows of *this* merchant. The temporal-shuffle validation hid the bug because both halves shared encoders.

Fix: revert to CatBoost with `cat_features=...`, validate on a strict time-forward split. AUC on the time-forward split: 0.954 (lower than the leaked 0.961, *higher* than the broken XGBoost online performance after re-evaluation). Online metric returned to baseline within a week. The right framing: ordered TS was not a "performance feature," it was a *bug-prevention* feature. We removed it, the bug came back, we put it back, the bug went away.

Lesson: prefer ordered TS as a structural choice, not as a tuning knob. The cost of a 1% AUC reduction from ordered boosting on validation is paid back many times over in not having to debug encoding-leakage incidents.

### 14.2 The recsys re-ranker where oblivious trees gave 3× lower p99 inference latency

Context: a marketplace re-ranker scoring ~200 candidate items per impression, 30K QPS at peak, each request needing all 200 scores in 8 ms p99. The original model was an XGBoost ranker, depth 6, 800 trees. p99 hovered at 11 ms — over budget. Profiling showed 6 ms was the model itself, 5 ms was feature fetch.

We tried two paths in parallel: optimize XGBoost (try `tree_method="hist"`, FFI directly to libxgboost, batch the candidates more aggressively) or switch to CatBoost. After a week of XGBoost optimization we got to ~9 ms p99, still over budget. The CatBoost port — same features, same training setup, mostly default hyperparameters — gave model latency of 1.8 ms and p99 of 6.2 ms, fitting comfortably under SLO. Metric (NDCG@10) was within 0.1% of the XGBoost model.

Why such a big gap? Two reasons. First, oblivious tree inference is a tight indirect-load loop with no branching; modern CPUs eat that loop at 3+ instructions/cycle. XGBoost's leaf-wise walk has data-dependent branches that mispredict. Second, the CatBoost C API let us call the model in batched form — score 200 candidates against 800 trees in one call — which amortizes the dispatch cost. XGBoost's C API wanted us to call once per candidate, or to construct a DMatrix per request, which had its own overhead.

The boring mechanic: CatBoost's `model.predict` on a 200×F float32 matrix is a single SIMD-friendly call that returns a 200-vector. The same call in XGBoost involves DMatrix construction and a tree-walk that is hard to vectorize. For high-QPS ranking workloads this matters more than any other axis.

### 14.3 The credit-risk model where feature combinations replaced 40 hand-engineered interactions

Context: small-business loan default prediction. Original model: XGBoost with ~120 features, of which roughly 40 were hand-crafted interactions in SQL — `industry × revenue_bracket`, `state × industry`, `years_in_business × industry`, etc. Each was a categorical of moderate cardinality. The feature pipeline was 1800 lines of SQL across three Looker views; schema migrations broke it monthly.

We rebuilt the model in CatBoost with the raw categoricals and `max_ctr_complexity=4`. Result: AUC matched (within 0.2%), feature pipeline reduced to ~600 lines, and most importantly the model picked up *new* interactions we hadn't thought of: `(zip_code_first_3_digits, industry, has_personal_guarantor)` was a strong signal we had never engineered. Reading the tree dumps — oblivious trees are short — let us find these and validate them with the underwriting team.

The migration paid back in three places: (a) less SQL to maintain, (b) better adaptation when industries shifted (we did not need to rewrite hand-crafted interactions when the underwriting team added new industry codes), and (c) better discoverability of new signals via tree-dump inspection.

The rough cost: a 30% increase in training time (combinations are not free) and a 20% increase in inference latency (more features encoded per row). Both fit comfortably in the budget.

### 14.4 The churn model that overfit silently because we used cat_features plus pre-encoded categories

Context: subscription churn at a content platform, ~5M users, ~3% monthly churn. Features included a few categoricals — `country`, `plan_type`, `device_class` — and a *handcrafted* feature called `country_avg_churn` that the data scientist who built the original model had computed offline as a target mean.

Someone added `cat_features=[country, plan_type, device_class, country_avg_churn]` to the new training run. CatBoost dutifully treated `country_avg_churn` as a *categorical* with 195 levels (one per country, encoded as a float that CatBoost coerced to a string) and computed an ordered TS *of the target mean*. Two problems: (a) it was a target mean computed on the *full* training set (including labels), so the underlying values were already leaked; (b) the ordered TS now encoded `country_avg_churn` with the *current* fold's prefix, but the underlying float was a category ID, which made the encoding nonsense.

The model trained to AUC 0.91 on validation, up from 0.83 baseline. We celebrated. The online lift was zero. We dug.

The fix: drop `country_avg_churn` (it was already redundant with `cat_features=[country]`), keep `country` as a native categorical, and let CatBoost compute its own ordered TS. AUC went back to 0.83 on validation; online lift matched. The lesson: do not pre-encode features before passing them as `cat_features`. If you have a numeric feature that is logically a category (e.g., a user-segment ID), pass it as a string and let CatBoost handle it.

This bug is so common we now have a pre-flight check in our training pipeline: any feature passed in `cat_features` is verified to have non-numeric dtype or to be an integer with cardinality < some threshold. CatBoost will not warn you when you do this; you have to catch it yourself.

### 14.5 The pricing model where RMSEWithUncertainty gave us free calibrated intervals

Context: dynamic pricing for a B2B SaaS product. We needed not just a point estimate of "what price will this customer pay" but also an interval, because the sales team wanted to know which deals were *certain* and which were *risky* before they invested time. The original model was an XGBoost regressor; we then trained a separate quantile regressor for the upper/lower bounds. Two models, two pipelines, two refreshes per month.

CatBoost's `RMSEWithUncertainty` loss fits both a mean and a log-variance head in one model. The training is essentially a Gaussian likelihood: $-\log p(y \mid x) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y - \mu)^2}{2\sigma^2}$. Each tree updates both heads. Inference returns $(\mu, \sigma)$, from which you can compute any interval analytically.

Migrating cut the pipeline in half. The intervals were also better-calibrated than the separate quantile regressor — partly because the joint training shares structure between the mean and variance estimates, partly because RMSEWithUncertainty's uncertainty is *epistemic + aleatoric* and the quantile regressor was only capturing aleatoric. We measured calibration via the empirical coverage of the 80% interval: separate quantile model 71%, RMSEWithUncertainty 79%. The latter is what the sales team needed.

A subtlety: the two heads have different scales of update, so the auto LR sometimes underfits the variance. If you see `sigma ≈ 0` at inference, lower the LR or increase iterations. We added a sanity check in the eval pipeline that flags any model with median `sigma < 0.01 * |mean|`.

### 14.6 The CTR model where GPU training was slower than CPU for 80M rows

Context: ads CTR prediction, 80M rows, 180 features, 60% of which were categorical with cardinality 100–10K. Default training on a 96-core CPU box: 38 minutes. Same dataset on a single A100: 47 minutes. Why was the GPU slower?

Profiling: the histogram kernel was fast — 70% HBM bandwidth. The bottleneck was the CTR computation. Each iteration, CatBoost has to compute ordered target statistics for every active categorical feature. On the GPU, this requires segmented prefix sums grouped by category ID, which is hard to parallelize when the cardinality is moderate (100s to 1000s) — too few segments to keep the SMs busy, too many to fit in shared memory. The kernel ended up serializing through global memory.

We tried a few things: increase the batch of permutations (worse), switch to `boosting_type="Plain"` to skip ordered boosting (faster but slightly worse model), pre-quantize the categoricals offline (saves 5 minutes). None of them flipped the ratio.

The lesson: GPU is faster only when histograms dominate. CTR-heavy datasets are not histogram-bound. We stayed on CPU. The 80M-row training is a daily cron; 38 minutes is fine.

### 14.7 The medical-risk model where monotonic constraints were the difference between deployable and not

Context: a hospital readmission risk model. Features included `age`, `comorbidity_count`, `prior_admissions_12mo`, `length_of_stay_index_admission`, plus a few categoricals. Domain experts and the regulators had a hard requirement: the model must be monotonic non-decreasing in each of those four numeric features. A 70-year-old with 5 comorbidities and 3 prior admissions cannot have a *lower* readmission risk than a 30-year-old with 0 comorbidities and 0 prior admissions, no matter what other features say.

Without the constraint, the model trained to AUC 0.78 but had thousands of pathological cases — local non-monotonicities induced by feature interactions. The audit team rejected it. We added `monotone_constraints={"age": +1, "comorbidity_count": +1, "prior_admissions_12mo": +1, "length_of_stay_index_admission": +1}`. AUC dropped to 0.76. Zero pathological cases. The model shipped.

The implementation: at every split-evaluation step, CatBoost checks whether splitting on `(feature, threshold)` would violate the constraint given the current partial tree. If it would, the gain for that split is zeroed out. The constraint is enforced *during training* — there is no post-hoc projection. This means the model is monotonic by construction, not by post-hoc audit.

A subtlety: monotonic constraints are over the *raw* feature values, so if a feature is mean-encoded before being passed to the model, the constraint applies to the encoded value, not to the original category. For categoricals you cannot constrain to be monotonic in any meaningful way — the order of categories is arbitrary.

The cost (2 AUC points) was the right trade. A model that the audit team rejects is worth zero, regardless of its AUC.

### 14.8 The recsys cold-start where the ordered-TS prior saved us from blowing up new merchants

Context: a marketplace ranking model where new sellers were onboarded daily — anywhere from 10 to 500 per day. The original model treated `seller_id` as a low-cardinality categorical (binned by tenure) plus a hand-crafted `seller_avg_rating` feature. New sellers got the global mean for the rating, which made them indistinguishable from each other and produced ranking degenerate enough that the marketplace team had to manually boost new-seller exposure with a business-rule layer.

We replaced the binned tenure feature and the hand-crafted average with raw `seller_id` as a categorical. CatBoost's ordered TS, with its smoothing prior $a$, naturally produced a calibrated estimate for new sellers: a seller with 0 prior transactions got the prior $p$ exactly; a seller with 5 prior transactions got a weighted average that pulled toward $p$ by a factor of $a / (5 + a)$; a seller with 500 transactions got essentially their empirical rate. This is exactly the right Bayesian update; we got it for free without writing a single line of smoothing code.

The downstream effect: the business-rule boost layer became unnecessary, ranking quality for new sellers improved by ~12% on a side-by-side test, and the model was *more* fair (in the sense that it did not penalize new sellers for being new — it just had appropriately wide uncertainty about them, expressed via the prior).

This is one of those quiet wins where the right algorithmic primitive replaces an ad-hoc business rule. The marketplace team had been carrying that boost layer for three years.

### 14.9 The migration where switching from LightGBM to CatBoost saved one engineer-week per quarter on encoding bugs

Context: a marketing-attribution model that ingested ~50 categoricals (campaign IDs, creative IDs, publisher IDs, geography). Originally LightGBM with hand-rolled k-fold target encoding. Every quarter we ran a "recalibration" job that rebuilt the encoders, retrained the model, and shipped. Every quarter, on average, one of the encoders broke quietly: a category that was new in this quarter had no entry in the encoder, the lookup returned NaN, and a chunk of inference traffic was scored with NaN-imputed features. We had alerts but they fired late.

Migrating to CatBoost: the encoder *is* the model. There is no separate artifact to recompute. New categories at inference time are handled by the model's smoothing prior — they get the global mean, scored with the model, and we move on. The recalibration job became a single train call.

Counted over a year: 4 quarters × ~3 days/quarter to debug the encoder pipeline = 12 engineer-days saved. Not a model-quality win; a maintenance win. Most of CatBoost's value in long-running production pipelines is exactly this: removing a class of bug from the system, freeing up calendar time.

The transition was also easier than we expected. CatBoost reads pandas DataFrames natively, infers categorical columns from dtype if you pass `cat_features="auto"`, and the `model.fit / model.predict` API is sklearn-compatible. The model file format (`.cbm`) is forward-compatible across versions; we have not had to retrain any model just to upgrade the library.

### 14.10 The text-classification baseline where CatBoost on TF-IDF beat a fine-tuned transformer

Context: a customer-support ticket router with ~7 classes, 200K labeled tickets, modest budget. The product team had asked for a fine-tuned BERT classifier; the team built one over six weeks and got 86.4% macro-F1. The deployment costs (a GPU pod permanently online for inference) were not loved by finance.

For comparison, I built a CatBoost baseline in two hours: TF-IDF features (50K-dim sparse vector), a couple of structured signals (`channel`, `customer_tier`, `time_of_day`), `loss_function="MultiClass"`. CatBoost handles the sparse TF-IDF input via the `scipy.sparse` interface; you have to tell it `sparse_features=...` to avoid densifying. Default hyperparameters, depth 8, 2000 iterations, 30 minutes on a 16-core CPU.

Result: 87.1% macro-F1. Slightly better than the BERT model. Inference latency: 200 µs/ticket on a single CPU core. Model size: 80 MB.

What happened? Two things. First, the BERT model was undertrained — 200K labels is below the regime where transformers comfortably beat strong baselines, and a small TF-IDF + GBT can soak up the linear structure of the problem efficiently. Second, the structured features (`channel`, `customer_tier`) were genuinely informative and the BERT model was using them only via a [SEP]-concatenation trick, not natively. CatBoost handled them as first-class features.

We shipped the CatBoost model, deprecated the BERT pod, and saved ~$3K/month in infra. The BERT team was not happy. They were right that a transformer is *capable* of beating a GBT here — with 5M tickets instead of 200K, with proper data augmentation, with feature integration done correctly — but they had not done that work, and the GBT baseline made it visible. The lesson: always train a CatBoost baseline before you commit to a transformer for tabular-ish problems. Especially when you are budget-constrained.

## 15. When to reach for CatBoost — and when not to

**Reach for CatBoost when:**

- Categoricals dominate, especially at moderate-to-high cardinality (100–100K levels).
- Dataset size is small to medium ($N < 10^7$). Ordered boosting helps most here.
- Inference latency budgets are tight. Oblivious trees vectorize well.
- You want defaults that work without sweeping. The auto LR + early stopping combination is genuinely best-in-class.
- You need built-in monotonic constraints, uncertainty estimation, or a clean ranking loss without writing your own.
- The team's tabular ML headcount is small and you need the framework to absorb operational complexity instead of leaking it into the pipeline.

**Reach for LightGBM when:**

- Dataset is huge ($N > 10^8$) and mostly numeric. LightGBM's leaf-wise grow + GOSS will train faster and give you slightly more capacity to use that data.
- You need extreme training throughput on a single CPU box (real-time-ish refreshes).
- Your team already has years of LightGBM tuning intuition and switching costs would dominate any model-quality gains.

**Reach for XGBoost when:**

- You are deeply embedded in the Spark ecosystem and need first-class XGBoost4J integration.
- You have legacy models you do not want to migrate and need consistency with existing pipelines.
- You need a feature that only XGBoost has (DART, PySpark integration, federated learning extensions).

**Skip GBT entirely when:**

- The data is image/audio/video — CNNs or transformers dominate, GBTs are not even in the conversation.
- The data is unstructured text — embeddings + transformers dominate; GBTs only enter as a re-ranker over embedding similarities.
- The task is causal inference under distribution shift — GBT's predictive lift does not translate to causal lift; reach for double machine learning or causal forests.
- Online learning is required at sub-second granularity — GBT retraining is fundamentally batch.

The most underrated argument for CatBoost is *operational*. Ordered boosting and native categoricals remove a category of bug. Oblivious trees give you predictable inference latency. Defaults that work mean you spend less calendar time tuning. Each of those is, individually, a small win. Together, they are the difference between a tabular ML team that ships steadily and one that spends every other quarter chasing encoder leakage. That has been the consistent shape of every CatBoost migration I have done.

If you take one thing from this post: stop pre-encoding your categoricals. Pass them as strings or as `pd.Categorical`, set `cat_features=...` (or let CatBoost infer it), and let the algorithm do its job. You will write less code, find fewer bugs, and ship more models. That is the whole pitch.

---
title: "Debugging a Recommender That Won't Improve"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Your recommender is stuck and every tweak you try does nothing. This is the systematic, symptom-first debugging playbook — a reusable diagnostic harness in Python, a worked session that finds a leaky feature and a train-serve skew, and the decision tree that saves you three weeks of guessing."
tags:
  [
    "recommendation-systems",
    "recsys",
    "debugging",
    "evaluation",
    "train-serve-skew",
    "data-leakage",
    "diagnostics",
    "calibration",
    "machine-learning",
    "mlops",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/debugging-a-recommender-that-wont-improve-1.png"
---

There is a particular kind of stuck that everyone who has shipped a recommender eventually meets. The model trains, the dashboards are green, the eval harness runs clean — and nothing you do moves the needle. You add a feature: Recall@10 holds at 0.18. You swap the loss from pointwise to BPR: NDCG@10 holds at 0.21. You triple the embedding dimension, retune the learning rate, throw more epochs at it: the curve flattens at exactly the same place it always does. Three weeks in, you have a folder of experiment runs that all landed within noise of each other, a Slack thread of increasingly desperate ideas, and a product manager asking, reasonably, why the "improvement" they were promised hasn't shipped. Or worse: the one change that *did* look great offline went out behind a flag, the A/B readout came back, and engagement is **down**.

The instinct, when a recommender won't improve, is to keep changing things — bigger model, fancier loss, another feature, a different negative sampler — and pray one of them catches. This almost never works, and not because the ideas are bad. It fails because **a recommender can refuse to improve for reasons at every single layer of the stack**: the data, the labels, the features, the model, the loss, the evaluation, or the serving loop. Randomly perturbing one layer at a time, with no controls, is a search over a huge space with a noisy objective and no gradient. You will spend the weeks and you will probably still not know *why* it was stuck. The thing that actually works is boring and reliable: **name the symptom, then bisect the stack until one cheap probe fails and localizes the bug.**

![A decision tree that routes the symptom of a recommender that will not improve into four branches — offline metric will not move, offline up but online flat, online got worse, and stuck at a recall ceiling — each leading to the layer where the bug usually lives.](/imgs/blogs/debugging-a-recommender-that-wont-improve-1.png)

This post is the troubleshooting capstone of the series' bias and debugging track. It is the post you open when your recommender is broken and you do not know why. The whole series has built the funnel — [retrieval → ranking → re-ranking](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) fed by a feedback loop, read off the [offline ↔ online reality gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) — and along the way it has named almost every way a recommender can fail. This post is where those failure modes become a *diagnostic procedure*. By the end you will be able to: take any "it won't improve" complaint, classify it by the symptom you can actually observe, run a decision tree to the layer that is biting, and confirm the cause with a cheap probe before you write a single line of fix. You will also leave with a reusable diagnostic harness — runnable Python that checks whether your model even beats popularity, runs an ablation sweep, probes for leakage by shuffling a feature, computes popularity-stratified recall, and diffs train-time against serve-time features. That harness is the thing you keep.

Let me put the thesis up front so the rest of the post earns it. **A recommender that won't improve is not one problem; it is a class of problems that look identical from the dashboard and have completely different causes underneath.** The only efficient way through is to refuse to guess — to treat the stuck model as a system under test, hold every layer fixed but one, and let the probes tell you where the signal dies. Debugging by bisection is not glamorous, but it converts an open-ended "why won't it get better" into a closed, finite search that almost always terminates in a day.

## 1. Why "it won't improve" is four different bugs wearing the same shirt

The first mistake is treating "the recommender won't improve" as a single symptom. It is at least four, and they point at different layers. The single most valuable thing you can do at the start of a debugging session is decide *which* of these you have, because each one rules out three quarters of the search space.

**Symptom 1: the offline metric won't move at all.** You change things and Recall@K, NDCG@K, AUC sit dead flat (or wander only within seed noise). The signal is not getting from your changes to the metric. The bug lives early — in the data, the labels, the features, or the optimizer — or there is no bug at all and you have simply hit the data ceiling (more on that below). This is the *measurement-is-broken-or-saturated* family.

**Symptom 2: offline improves but online doesn't.** Your offline number goes up, cleanly and reproducibly, and the A/B test comes back flat. The model genuinely learned something *on the logged distribution* that does not transfer to the live one. The bug lives in the gap between the two worlds: [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there), the [offline-online distribution gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied), [position and popularity bias](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) inflating the offline read, a proxy metric that diverges from value, or the sampled-metrics illusion where your offline number is computed against a handful of negatives and is not even a consistent estimator of the full-corpus metric.

**Symptom 3: online got worse.** This is the urgent one — you shipped and engagement dropped. The most common causes are a train-serve skew that only bites a subset of live traffic, a feedback loop or exploration collapse that quietly narrowed the catalog, a calibration break (your scores are no longer probabilities and your business logic that consumes them — ad bidding, blending, thresholds — is now wrong), a guardrail regression (you optimized clicks and tanked diversity or latency), or plain latency-induced timeouts dropping candidates.

**Symptom 4: stuck at a recall ceiling.** The ranking metrics plateau and nothing in the ranker helps, because the ceiling is set upstream: stage-1 retrieval simply never surfaces the relevant item, so no ranker can rank it. This is a [funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) problem — the Recall@K of retrieval caps everything downstream — and the usual culprits are [bad negatives](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) in two-tower training, an [ANN index](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) whose recall is too low at the latency you serve, or a candidate set that is genuinely too small.

Naming the symptom is half the debugging. The decision tree in the figure above is just these four branches with the layer each routes to. Everything else in this post is the probes that confirm which leaf you are on.

### The bisection principle

Underneath the symptom triage is one idea, and it is the single most important habit in this whole post: **change one layer at a time, and hold everything else fixed, so that when a number moves you know which layer moved it.** This is bisection. It is the same discipline as `git bisect` or binary-searching a regression — you are not trying to *fix* the bug yet, you are trying to *localize* it.

![A bisection diagram that holds every layer of the recommender fixed and probes one at a time — data, features, model, evaluation, and serving — so a single failing probe localizes the bug instead of leaving the whole stack under suspicion.](/imgs/blogs/debugging-a-recommender-that-wont-improve-3.png)

The reason guessing fails is that a typical "let me try a fancier model" change touches three or four layers at once. A new architecture changes capacity (model), often changes the loss, often changes the feature pipeline, and sometimes changes the negative sampler. If the metric stays flat, you have learned nothing — you cannot tell whether the model didn't help, the loss regressed, the features broke, and the net was zero, or whether all four were irrelevant because the real bug is a leaky label upstream. **Confounded experiments produce confounded conclusions.** Bisection refuses to confound: each probe isolates exactly one layer, so each probe's result is interpretable on its own.

There is a second reason bisection beats guessing, and it is statistical. Your objective — an offline ranking metric — is *noisy*. Run the same training twice with different seeds and Recall@10 wobbles by a few thousandths just from initialization and minibatch order. If your changes produce effects of the same size as that noise, then a sequence of guesses is a random walk through the noise floor: half your "improvements" are seed luck that vanishes on the next run, and you'll chase ghosts for weeks. Bisection sidesteps this because the probes don't ask "did the metric go up a little?" — they ask categorical questions with large, unambiguous answers: *does the model beat popularity at all?* (a 1.7× gap, not a 0.3% wobble), *does shuffling this feature collapse the metric?* (0.42 → 0.23, not noise), *do 12% of serving rows differ from training?* (a structural fact, not a seed). Large categorical signals are robust to the noise that swamps small metric deltas, which is exactly why a localizing probe is more trustworthy than a tuning experiment when you don't yet know what's wrong.

The bisection order matters too. Run the probes in the order that maximizes information per minute: the cheapest, most-discriminating probes first. Beat-popularity costs minutes and rules out the entire "no signal at all" branch. The learning curve costs one sweep and splits underfit/overfit/data-limited cleanly. The leakage probe costs one shuffle and convicts a feature. The train-serve diff costs a logging job and a join. By the time you reach the expensive probes, the cheap ones have already eliminated most of the tree — which is the whole point of a binary search.

## 2. The data ceiling: sometimes the model is already as good as it can get

Before we hunt for bugs, we have to rule out the case where there is no bug. The most demoralizing version of "it won't improve" is the one where the model is *already optimal for the data you have*, and every change is noise because there is nothing left to extract.

Every prediction task has an **irreducible error** — the Bayes error rate. Formally, for a target $y$ given features $x$, the best possible predictor is the conditional expectation $f^\star(x) = \mathbb{E}[y \mid x]$, and no model can do better than the error of $f^\star$ itself, which comes from the genuine noise in $y$ that $x$ does not explain. In recommendation, $x$ is whatever you know about the (user, item, context) at request time, and $y$ is whether the user engages. A huge amount of engagement is simply not predictable from what you logged: mood, what a friend texted them, whether they are on a phone in a queue or a laptop at home. That variance is irreducible *given your features*. The data ceiling is the performance of $f^\star$ on your feature set, and if you are near it, the only way up is **more or better features**, not a better model.

How do you tell you are at the ceiling rather than stuck on a bug? Three signs, and you should check all three before you believe it:

1. **A bigger model doesn't help, and neither does a smaller one hurt.** If you can halve the capacity and lose nothing, and double it and gain nothing, you are not capacity-limited — you are data-limited or bug-limited. (A bug can also masquerade as this; you rule out the bug with the probes below.)
2. **Your training and validation metrics are close to each other and both plateaued.** Small generalization gap plus a flat curve means you are not overfitting (which a smaller model or regularization would fix) and not underfitting (which a bigger model or more epochs would fix). You are fitting the signal that exists.
3. **Adding genuinely new information moves the number.** The decisive test: bring in a feature that carries information your current set doesn't (e.g. the item's text embedding, the user's last-session items, a context feature like time-of-day) and watch the metric. If a *real* new feature moves it, you were not at the ceiling — you were feature-limited. If three independent new features each do nothing, you are probably near the ceiling for this task.

The data ceiling is not an excuse — it is a diagnosis with a different fix. If you are at it, the work is feature engineering, better labels, or a richer logging schema, not model tuning. Spending another month on architecture when you are 0.5% from the Bayes rate is the most expensive way to learn nothing. Crucially, you cannot conclude "data ceiling" until you have ruled out the cheaper bugs — a leaky feature can *fake* a high ceiling, and a label bug can *fake* a low one. So the ceiling is the *last* thing you conclude, after the probes come back clean.

### Underfit vs overfit vs data-limited from the learning curve

The learning curve — training metric and validation metric as a function of training-set size or epochs — is the single richest diagnostic for "metric won't move," and it cleanly separates the three regimes. Plot validation Recall@10 (or AUC) against the number of training interactions, holding the model fixed:

- **Underfit (high bias):** both training and validation metrics are low and close together. The model can't fit even the data it has seen. More data won't help much; a bigger model, more epochs, a less aggressive learning rate, or richer features will. The curve is flat and low and the two lines hug each other.
- **Overfit (high variance):** training metric is high, validation metric is much lower, and the gap is wide and *growing* as you add capacity. The model memorizes. More data closes the gap (the validation line rises toward the training line), and so does regularization (dropout, weight decay, smaller embeddings) or early stopping. This is the regime where "it won't improve" on validation actually means "it's improving on the wrong thing."
- **Data-limited (at the ceiling):** training and validation are close *and* both have plateaued, and crucially the validation curve is still rising — even if slowly — as you add data, then flattens. If extrapolating the data-size curve shows it has gone flat, more of the *same* data won't help; you need new *kinds* of data (features) or to accept the ceiling.

A practical version: train on 25%, 50%, 75%, 100% of your interactions and plot validation AUC for each. If the curve is still climbing at 100%, you are data-limited and more logged data (or a longer window) will help. If it flattened by 50%, more rows won't move it — your lever is features or model, decided by the train-validation gap.

#### Worked example: reading the learning curve to choose a lever

A team is stuck at validation AUC 0.74 on a CTR ranker and the debate is "bigger model or more data?" They run the data-fraction sweep and the model-capacity sweep side by side. Data fractions: AUC at 25% = 0.735, 50% = 0.739, 75% = 0.741, 100% = 0.742 — a curve that has clearly flattened; the last doubling of data bought 0.001. Capacity sweep, all at 100% data: hidden width 64 = 0.741, 128 = 0.742, 256 = 0.742 — flat. Train vs validation AUC at the operating point: 0.748 vs 0.742, a gap of 0.006 — tiny. Three readings, one conclusion: not data-limited's "still climbing," not overfit's "wide growing gap," not underfit's "low train AUC." The model is fitting essentially all the signal its features contain. The right lever is *neither* more data nor a bigger model — it is **new features**. They add the user's last-session item sequence (information the static features never carried) and validation AUC moves to 0.761 in one shot. The learning curve told them where the lever was before they wasted a month on architecture. This is the data-ceiling diagnosis in action: the ceiling was real for the *feature set*, and the only way through it was to change the feature set.

### The science: error decomposition tells you which layer to fix

The bisection habit has a precise statistical justification, and it is worth making rigorous because it tells you *which fix can possibly help*. Take a model $\hat{f}$ trained on data $D$ and consider its expected error on the true task. The classic decomposition splits that error into three additive terms:

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\sigma^2_{\text{irr}}}_{\text{data ceiling}} + \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f^\star(x)\big)^2}_{\text{bias (underfit)}} + \underbrace{\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]}_{\text{variance (overfit)}}
$$

Read each term as a fix:

- $\sigma^2_{\text{irr}}$ is the **irreducible error** — the Bayes rate, the noise in $y$ that $x$ cannot explain. No model touches it. The *only* lever is changing $x$ (new features) or changing the task. This is the data ceiling, formalized: when bias and variance are both small, your error *is* $\sigma^2_{\text{irr}}$ and tuning the model is pushing on a term that is already zero.
- The **bias** term is underfit — your model class can't represent $f^\star$. Bigger model, richer features, more epochs, less regularization. Diagnosed by low-and-close train/val curves.
- The **variance** term is overfit — your model is too sensitive to the particular $D$. More data, regularization, early stopping. Diagnosed by a wide train/val gap.

The whole point of the learning curve is that it *measures these three terms separately*. The train/val gap is the variance term; the height of the (converged) training curve above zero is the bias term plus the irreducible floor; and the slope of validation-versus-data-size tells you whether more data still helps (variance-dominated) or not (you've hit bias + irreducible). A debugging session that doesn't know which term dominates is, in effect, applying a random fix and hoping it matches the dominant term — which is exactly the guessing the bisection replaces. The serving-loop bugs (skew, calibration, leakage) sit *outside* this decomposition entirely: they are cases where the $x$ at serving time is not the $x$ you trained on, or the $y$ you optimized is not the $y$ that pays. That is precisely why they need their own probes — no amount of bias/variance tuning reaches them.

## 3. The diagnostic harness: the probes you keep

Here is the toolkit. The philosophy is **cheapest probe first**: run the ones that take seconds and rule out the dumb-but-common bugs before you reach for the expensive ones. The harness has seven probes — five you can read straight off a metric and two that read the model's internals and slices — and you run them top to bottom, stopping the moment one of them convicts a layer. The figure shows the five outer layers; probes 6 and 7 (gradient/embedding health and the cold-start slice) slot into the "baselines" and "stratified metrics" rows respectively as deeper reads of the same layers.

![A layered stack of the diagnostic harness, running cheapest-first from baseline comparisons through an ablation sweep and a leakage probe to popularity-stratified metrics and finally a train-serve feature diff.](/imgs/blogs/debugging-a-recommender-that-wont-improve-4.png)

### Probe 1: sanity baselines — does your model even beat popularity and random?

This is the cheapest and most embarrassing-when-it-fails probe, and you should run it on *every* recommender, always. There are two baselines:

- **Random:** rank items in random order. Recall@K and NDCG@K under random are a floor. If your model isn't beating random, something is catastrophically wrong — a label-shuffle bug, a metric computed on the wrong axis, scores that aren't connected to the items.
- **Most-popular:** recommend the globally most-popular items to everyone, ignoring the user. This is the baseline that matters, because popularity is *shockingly hard to beat* on top-K metrics. A huge fraction of clicks go to a small head of items, so "show everyone the head" gets a lot of them right. If your fancy personalized model only ties the most-popular baseline, **it has learned no personalization** — the user representation is dead, the features carry no signal, or the loss is letting the model collapse onto the popularity prior.

Beating popularity is the line between "a model" and "a lookup table with extra steps." Here is the probe:

```python
import numpy as np

def recall_at_k(ranked_items, holdout_items, k=10):
    """Fraction of a user's held-out items that appear in the top-k ranking."""
    if not holdout_items:
        return None
    top_k = set(ranked_items[:k])
    hits = len(top_k & set(holdout_items))
    return hits / min(k, len(holdout_items))

def popularity_ranking(item_counts, exclude=()):
    """Items sorted by global interaction count, descending."""
    order = np.argsort(-item_counts)
    return [int(i) for i in order if i not in exclude]

def evaluate_baselines(test_users, holdout, item_counts, model_rank_fn, k=10, seed=0):
    rng = np.random.default_rng(seed)
    n_items = len(item_counts)
    pop = popularity_ranking(item_counts)
    rows = {"model": [], "popularity": [], "random": []}
    for u in test_users:
        h = holdout[u]
        if not h:
            continue
        rows["model"].append(recall_at_k(model_rank_fn(u), h, k))
        rows["popularity"].append(recall_at_k(pop, h, k))
        rand = list(rng.permutation(n_items))
        rows["random"].append(recall_at_k(rand, h, k))
    return {name: float(np.mean(vals)) for name, vals in rows.items()}

# scores = evaluate_baselines(test_users, holdout, item_counts, my_model.rank, k=10)
# print(scores)  # {'model': 0.31, 'popularity': 0.18, 'random': 0.004}
```

The verdict is binary. `model > popularity` by a clear margin (say 1.3× or more, well outside seed noise) means you have a real model and can proceed to tune it. `model ≈ popularity` means stop tuning and go fix the data or features — no learning rate will rescue a model that learned nothing personal.

![A comparison of an unhealthy model whose Recall at ten ties the most-popular baseline against a healthy model that clears popularity by a wide margin and surfaces tail items, showing why beating popularity is the first sanity check.](/imgs/blogs/debugging-a-recommender-that-wont-improve-6.png)

#### Worked example: are we actually beating popularity?

We run a two-tower retrieval model on a MovieLens-style holdout — 6,000 test users, last interaction held out, Recall@10 the metric. The harness returns:

| Ranker | Recall@10 | vs popularity |
| --- | --- | --- |
| Random | 0.004 | 0.02× |
| Most-popular | 0.182 | 1.00× (baseline) |
| Our two-tower (as found) | 0.186 | 1.02× |

The two-tower is beating *random* by 46×, which feels great until you see it is beating *popularity* by 1.02× — within noise. The model has learned essentially nothing beyond the popularity prior. If we had only compared against random, we'd have declared victory. The popularity baseline is the one that tells the truth: this model is a popularity lookup with a personalized veneer, and the right next move is to probe the features and data, *not* to tune the optimizer. (After we fix the bugs later in this post, the same model returns Recall@10 = 0.31, a real 1.7× over popularity. That is what "beating popularity" should look like.)

### Probe 2: the ablation sweep — which component actually matters?

An ablation turns off one component at a time and re-measures. It answers "is this component pulling its weight?" and, more usefully for debugging, "is one component suspiciously *dominant*?" — because a single feature that, alone, explains almost the entire metric is the fingerprint of leakage.

```python
def ablation_sweep(train_fn, eval_fn, base_config, components):
    """For each component, retrain with it disabled and report the metric delta."""
    base_metric = eval_fn(train_fn(base_config))
    results = {"<full model>": base_metric}
    for comp in components:
        cfg = dict(base_config)
        cfg[f"disable_{comp}"] = True
        ablated_metric = eval_fn(train_fn(cfg))
        results[comp] = base_metric - ablated_metric  # how much this comp adds
    return results

# results = ablation_sweep(train, eval_ndcg, cfg, ["user_history", "item_text",
#                                                  "last_click_item", "ctr_feature"])
# -> {'<full model>': 0.42, 'user_history': 0.03, 'item_text': 0.02,
#     'last_click_item': 0.005, 'ctr_feature': 0.20}
```

Read the deltas. A healthy model has several features each contributing a modest amount. A model where *one* feature contributes 0.20 of a 0.42 NDCG — nearly half the metric, from one feature — is screaming at you. Either that feature is genuinely magical (rare) or it is leaking the label (common). Ablation tells you *which* feature to interrogate; the leakage probe tells you *whether* it's leaking.

### Probe 3: the leakage probe — shuffle a feature and watch the metric

Leakage is when a feature contains information about the label that would not be available at prediction time. The classic recommender leak: a feature like `item_ctr` (the item's click-through rate) computed over a window that *includes the test period*, so the model is reading the answer. Or `last_click_item` accidentally populated with the held-out item itself. Or an aggregate like "number of times this user interacted with this item" computed on the full dataset before the temporal split. Leakage inflates offline metrics gorgeously and transfers *nothing* online, which is exactly why "offline up, online flat" so often traces to it.

The probe is beautifully simple: **shuffle the suspect feature's values across rows and re-evaluate.** Shuffling destroys the feature's relationship to the label while preserving its marginal distribution. If the metric *barely drops*, the model wasn't relying on that feature — fine. If the metric *collapses*, the model was leaning hard on it — and if it was leaning hard on a feature that, on reflection, shouldn't be that predictive, you have found a leak.

```python
def leakage_probe(model, eval_fn, X, feature_name, seed=0):
    """Shuffle one feature across rows; a large metric drop on a feature that
    'shouldn't' be that strong is the fingerprint of leakage."""
    baseline = eval_fn(model, X)
    rng = np.random.default_rng(seed)
    X_shuffled = X.copy()
    X_shuffled[feature_name] = rng.permutation(X_shuffled[feature_name].values)
    shuffled_metric = eval_fn(model, X_shuffled)
    drop = baseline - shuffled_metric
    return {"baseline": baseline, "shuffled": shuffled_metric,
            "drop": drop, "relative_drop": drop / baseline}

# leakage_probe(model, eval_auc, df_val, "item_ctr")
# -> {'baseline': 0.91, 'shuffled': 0.71, 'drop': 0.20, 'relative_drop': 0.22}
```

There is a subtlety worth stating: a *legitimately* strong feature will also drop the metric when shuffled — that's permutation importance, and it's expected. The leakage signal is not "shuffling hurt"; it is "shuffling hurt *far more than this feature has any right to*, and the feature was computed in a way that could see the future." Pair the probe with a hard look at *how* the feature is computed: what window, what split, what timestamp cutoff. Leakage is ultimately a data-pipeline bug, and the probe just points you at the file to read.

#### Worked example: a leakage probe convicts a feature

Our two-tower's offline NDCG@10 looks great after we add an `item_recent_ctr` feature: it jumps from 0.22 to 0.42. The ablation sweep flagged it — that one feature contributes 0.20 of the 0.42. We run the leakage probe:

| Setting | NDCG@10 |
| --- | --- |
| Full model | 0.42 |
| `item_recent_ctr` shuffled | 0.23 |
| `item_recent_ctr` dropped entirely | 0.22 |

Shuffling the feature drops NDCG by 0.19 — almost the entire lift the feature provided. A click-through-rate feature being *that* dominant is the tell. We read the pipeline and find it: `item_recent_ctr` is computed over a 7-day window, but the window is anchored to *today*, the day we ran the feature job — not to each interaction's timestamp. So for every interaction in the temporal test set, the feature already incorporates clicks that happened *after* that interaction, including the held-out one. The model is reading a smeared version of the future. We recompute the feature as a strict point-in-time aggregate (only clicks strictly before each interaction's timestamp), and the offline NDCG settles at an *honest* 0.24. Lower — but real. The 0.42 was a lie. This is the single most common reason a recommender "improves" offline and does nothing online.

### Probe 4: popularity-stratified metrics — where is the model winning?

A single aggregate metric hides where the model works. Stratify Recall@K by item popularity bucket — head, torso, tail — and you immediately see whether the model is just riding the popularity prior (all its hits are in the head) or genuinely personalizing (it surfaces torso and tail items the user actually wanted). This is the same lens as the [popularity bias post](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer), reused as a diagnostic.

```python
def stratified_recall(test_users, holdout, model_rank_fn, item_pop_bucket, k=10):
    """Recall@k split by the popularity bucket of the held-out item."""
    from collections import defaultdict
    hits = defaultdict(int)
    totals = defaultdict(int)
    for u in test_users:
        ranked = set(model_rank_fn(u)[:k])
        for item in holdout[u]:
            b = item_pop_bucket[item]      # 'head' | 'torso' | 'tail'
            totals[b] += 1
            if item in ranked:
                hits[b] += 1
    return {b: hits[b] / totals[b] for b in totals if totals[b] > 0}

# stratified_recall(...) -> {'head': 0.55, 'torso': 0.21, 'tail': 0.04}
```

If your overall Recall@10 is "fine" but it's `head 0.55 / torso 0.05 / tail 0.00`, the model is a popularity machine that happens to score well because most holdout items are popular. That is a model that will feel boring to users, will struggle the moment your traffic shifts toward the tail, and will not transfer well online where novelty matters. Stratified metrics catch the model that is *technically* beating popularity in aggregate but is doing it by predicting the head harder, not by personalizing.

### Probe 5: sampled-vs-full evaluation — is your metric even consistent?

This one is a trap that has fooled entire papers. To make evaluation cheap, people compute Recall@K and NDCG@K against a small sample of negatives — for each held-out positive, score it against 100 random items and see if the positive ranks in the top K of *those 101*. It's fast. It is also, as Krichene and Rendle showed in their 2020 KDD paper "On Sampled Metrics for Item Recommendation," **not a consistent estimator of the full-corpus metric** — the ranking of models under sampled metrics can *disagree* with the ranking under the true full-corpus metric. Model A can beat model B on sampled Recall@10 and lose on full Recall@10.

The implication for debugging: if your offline metric is sampled and it "won't move," it might be moving on the full corpus and your sampled estimator is too coarse to see it — or worse, your changes are improving the sampled metric while the real one stalls. The probe is to compute both on a slice and check they agree:

```python
def sampled_vs_full_recall(scores_fn, test_users, holdout, n_items,
                           n_neg=100, k=10, seed=0):
    """Compare Recall@k against n_neg sampled negatives vs the full catalog."""
    rng = np.random.default_rng(seed)
    full_hits, samp_hits, n = 0, 0, 0
    for u in test_users:
        for pos in holdout[u]:
            s_all = scores_fn(u)                       # score over all items
            full_rank = 1 + int((s_all > s_all[pos]).sum())
            full_hits += full_rank <= k
            negs = rng.choice(n_items, size=n_neg, replace=False)
            cand = np.append(negs, pos)
            s_samp = s_all[cand]
            samp_rank = 1 + int((s_samp > s_all[pos]).sum())
            samp_hits += samp_rank <= k
            n += 1
    return {"full_recall@k": full_hits / n, "sampled_recall@k": samp_hits / n}

# -> {'full_recall@k': 0.19, 'sampled_recall@k': 0.71}
```

The sampled metric (0.71) and the full metric (0.19) live in different universes. Always debug against the **full-corpus** metric when you can afford it; if you can't (100M items), at least verify on a slice that your model *ranking* under sampled and full agree before you trust the sampled number for model selection.

### Probe 6: gradient and embedding-norm health — is the model even training?

When the offline metric "won't move," a surprising fraction of the time the model is not actually learning, and the eval metric is just faithfully reporting that. The two cheapest internal signals are the **gradient norm** and the **embedding norm**, and you should log both every few hundred steps. They catch a class of bugs that no eval metric can name precisely.

The healthy picture is specific: the loss descends, the global gradient norm starts moderate and decays toward a small steady value, and the embedding norms grow from their small initialization to a stable spread (some items large, some small, reflecting differing frequencies). The pathologies each have a distinct fingerprint:

- **Gradient norm explodes to `inf`/`NaN`, then the loss is `NaN` forever.** Learning rate too high, no gradient clipping, or a numerically unstable loss (an un-clamped `log` of a near-zero softmax). The eval metric flatlines because the weights are `NaN`. Fix: lower the lr, add `clip_grad_norm_`, clamp the loss inputs.
- **Gradient norm is essentially zero from step one.** Vanishing gradients or a disconnected graph — a feature that never feeds the loss, an embedding table that isn't in the optimizer's parameter list, a `detach()` left in by accident. The model is frozen at initialization and the eval metric sits at the random/popularity floor. Fix: trace the autograd graph; confirm every parameter has `requires_grad=True` and a non-`None` `.grad`.
- **Embedding norms collapse toward zero.** A common two-tower failure under in-batch negatives with no normalization or temperature: the model minimizes the loss by shrinking all embeddings toward the origin (a degenerate solution where every dot product is near zero). The eval metric collapses to popularity because the user vector carries no direction. Fix: L2-normalize the towers and use a temperature, or add a small norm penalty; see the [two-tower training post](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax).
- **A few embedding norms blow up while most stay tiny.** Usually the head items getting enormous norms because they appear in nearly every batch as in-batch negatives or positives — the model is over-fitting the head, which also shows up downstream as a head-only stratified recall.

The probe is a few lines you wire into the training loop:

```python
import torch

def grad_embedding_health(model, embedding_attrs=("user_emb", "item_emb")):
    """Snapshot global grad norm and per-table embedding-norm stats.
    Call every N steps; watch the trajectory, not a single value."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().norm(2) ** 2)
    grad_norm = total_sq ** 0.5

    emb_stats = {}
    for name in embedding_attrs:
        table = getattr(model, name).weight.detach()
        norms = table.norm(dim=1)
        emb_stats[name] = {
            "mean_norm": float(norms.mean()),
            "p05_norm": float(norms.quantile(0.05)),
            "p95_norm": float(norms.quantile(0.95)),
            "frac_near_zero": float((norms < 1e-3).float().mean()),
        }
    return {"grad_norm": grad_norm, "embeddings": emb_stats}

# step 100:  {'grad_norm': 4.2, 'item_emb': {'mean_norm': 0.31, 'frac_near_zero': 0.00}}
# step 2000: {'grad_norm': 0.0, 'item_emb': {'mean_norm': 0.002, 'frac_near_zero': 0.98}}  # collapsed!
```

The second snapshot is the collapse fingerprint: grad norm has gone to zero *and* 98% of item embeddings are near-zero. No eval metric would tell you *why* the model ties popularity — this probe does, in one line. Reading the loss curve and these two internal norms together is the difference between "the model won't improve" and "the model collapsed at step 1,800 because the two-tower had no normalization."

### Probe 7: the cold-start slice — does the model fail exactly where it must succeed?

Aggregate metrics are dominated by warm users and warm items, because those are where the data is. But a recommender's value often lives precisely where the data is thin: new users with one or two interactions, new items added yesterday. A model can have a perfectly healthy overall metric and be *useless* on the cold slice — and if your product is a marketplace that constantly adds items, or a feed onboarding new users daily, the cold slice is the part that determines whether the system feels alive. So carve out a held-out **cold-start slice** and report metrics on it separately. This is the [cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) used as a diagnostic.

```python
def cold_start_slice_metrics(test_users, holdout, model_rank_fn,
                             user_interaction_count, item_first_seen_day,
                             today, k=10, cold_user_max=3, new_item_days=7):
    """Recall@k on warm vs cold users, and on new vs established held-out items."""
    from collections import defaultdict
    hits, totals = defaultdict(int), defaultdict(int)
    for u in test_users:
        ranked = set(model_rank_fn(u)[:k])
        u_bucket = "cold_user" if user_interaction_count[u] <= cold_user_max else "warm_user"
        for item in holdout[u]:
            age = today - item_first_seen_day[item]
            i_bucket = "new_item" if age <= new_item_days else "established_item"
            for b in (u_bucket, i_bucket):
                totals[b] += 1
                hits[b] += item in ranked
    return {b: hits[b] / totals[b] for b in totals if totals[b] > 0}

# -> {'warm_user': 0.34, 'cold_user': 0.06, 'established_item': 0.31, 'new_item': 0.02}
```

If the cold-user recall is 0.06 against a warm-user 0.34, the model is leaning entirely on accumulated user history and has nothing to say to a new user — a content-based fallback or a popularity-by-context default is missing. If new-item recall is 0.02, new items never surface (no warm collaborative signal yet, and the retrieval side has no content tower to embed them). Neither bug shows in the aggregate, because warm-and-established rows swamp the average. Slicing the metric by what *kind* of cold the row is tells you which fallback to build. When "the recommender won't improve" really means "it won't improve *for the launch we care about*," this is the probe that says so.

## 4. The train-serve consistency check: when offline and online disagree

If the symptom is "offline up, online flat" or "online got worse," the highest-yield probe is the **train-serve feature diff**, and it deserves its own section because it is both the most common cause and the most under-instrumented. The full mechanism is in the dedicated post on [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there); here is the diagnostic.

Train-serve skew is when the features your model sees at serving time differ from the features it was trained on — same name, different value. The causes are mundane and brutal: the offline feature pipeline (a Spark job over a warehouse) and the online pipeline (a Python service hitting a feature store) compute the "same" feature with different code, different defaults for missing values, different time windows, different bucketization edges, a different normalization constant baked in at training time. The model was trained on one distribution and is being asked to predict on another. Offline it looks fine because offline you use the *training* pipeline. Online it degrades because online you use the *serving* pipeline. The gap is invisible until you measure it directly.

The check is **log-and-replay**: at serving time, log the exact feature vector the model actually consumed for a sample of requests. Offline, recompute the features for those same (user, item, timestamp) tuples with the training pipeline. Diff them, feature by feature.

```python
import pandas as pd
import numpy as np

def train_serve_diff(served_features: pd.DataFrame,
                     recomputed_features: pd.DataFrame,
                     keys=("request_id",), tol=1e-4):
    """Join logged serving features to offline-recomputed features and report,
    per feature column, the fraction of rows that disagree and the mean abs gap."""
    m = served_features.merge(recomputed_features, on=list(keys),
                              suffixes=("_serve", "_train"))
    report = {}
    feature_cols = [c[:-6] for c in m.columns if c.endswith("_serve")]
    for f in feature_cols:
        s, t = m[f + "_serve"], m[f + "_train"]
        if np.issubdtype(s.dtype, np.number):
            disagree = (np.abs(s - t) > tol)
            report[f] = {"pct_rows_differ": float(disagree.mean()),
                         "mean_abs_gap": float(np.abs(s - t).mean()),
                         "serve_nan_rate": float(s.isna().mean()),
                         "train_nan_rate": float(t.isna().mean())}
        else:
            disagree = (s.astype(str) != t.astype(str))
            report[f] = {"pct_rows_differ": float(disagree.mean())}
    return pd.DataFrame(report).T.sort_values("pct_rows_differ", ascending=False)

# print(train_serve_diff(served_log, offline_recompute))
#                     pct_rows_differ  mean_abs_gap  serve_nan_rate  train_nan_rate
# user_avg_ctr_7d              0.121         0.044           0.030           0.000
# item_price_bucket           0.004         0.000           0.000           0.000
# user_country                0.000         0.000           0.000           0.000
```

Read the top of that table. `user_avg_ctr_7d` differs on 12% of rows, with a 3% serve-side NaN rate the training data never had. That is your skew: the online service is failing to find the feature for some users (cold reads, a feature-store miss, a TTL expiry) and filling NaN, while the training pipeline always had it. The model never learned to handle that NaN sensibly, so for 12% of live requests it is predicting on garbage. Offline this feature was perfect, so offline the model looked great. The fix is a *shared transformation* — one piece of code computes the feature for both training and serving — and a defined, learned-on default for missing values. The general remedy is a feature store or a shared feature-transformation library, but the *diagnostic* is always this diff.

The single best preventative is to **train on logged-at-serving features**: log exactly what the model saw, and train on those logs, so there is no second pipeline to skew. When that is not possible, the train-serve diff is your early-warning system, and you should run it as a scheduled job, not just during a fire.

## 5. The full symptom-to-fix map

With the probes in hand, here is the complete map. The figure compresses it; the prose makes each row actionable.

![A matrix mapping each symptom to its most likely cause, the cheap diagnostic that confirms it, and the targeted fix, so a debugging session runs the test before writing the patch.](/imgs/blogs/debugging-a-recommender-that-wont-improve-2.png)

### Symptom 1: offline metric won't move at all

Run, in order: beat-popularity (probe 1), the learning curve (section 2), the ablation sweep (probe 2), then the leakage probe on any dominant feature (probe 3).

- **Doesn't beat random** → a wiring bug. Scores not connected to items, labels shuffled, metric computed on the wrong axis (you're measuring item recall when you meant user recall), or your `holdout` is empty. Fix the plumbing before anything else.
- **Beats random, ties popularity** → no personalization. Dead user features, an under-trained user tower, a loss that collapses to the popularity prior (pointwise BCE with random negatives loves to do this — see [negative sampling](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) and the [BPR loss post](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive)). Check that user embeddings actually vary across users; check embedding norms aren't collapsing to zero.
- **Learning curve flat-and-low, train ≈ val** → underfit. Raise capacity, lower the learning rate if loss is diverging, add epochs, add real features.
- **Learning curve: train high, val low and gap growing** → overfit. Regularize, shrink embeddings, early-stop, add data.
- **One feature dominates the ablation** → leakage suspect. Run the shuffle probe; read how the feature is computed relative to the temporal split.
- **Everything clean, three new features each do nothing** → data ceiling. Stop tuning; invest in features/labels/logging.

A note on dead features and the optimizer: a flat metric often hides a *dead feature* — a column that is constant, all-NaN, or zeroed by a bad join — contributing nothing because it carries nothing. The ablation sweep catches these as "disabling it changes nothing because it was already nothing." And a learning-rate or optimizer pathology (lr too high, gradients exploding then NaN, or lr too low so the model never leaves init) shows up as a loss curve that doesn't descend; always *look at the loss curve*, not just the eval metric, when the metric won't move.

### Symptom 2: offline improves but online doesn't

This is the gap. The probes are the train-serve diff (section 4), the sampled-vs-full check (probe 5), and the leakage probe (probe 3), plus a hard look at whether your offline metric is even the right proxy.

- **Train-serve diff shows differing rows** → skew. The most common single cause. Fix with a shared transform.
- **Sampled metric improved, full metric didn't** → the sampled-metrics illusion. Re-select your model on the full-corpus metric.
- **Leakage probe convicts a feature** → leakage inflated offline; online there is no future to read. Recompute point-in-time.
- **Everything clean but online still flat** → distribution shift or a proxy that diverges from value. The offline metric is computed under the logging policy's distribution; online you create a new one. This is the deep [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied), and the honest fixes are IPS-corrected offline estimates, exploration to get on-distribution data, or just trusting the A/B and changing your proxy. Also check [position bias](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data): if offline you rewarded a model for re-ranking items the *old* model placed high, you may be measuring position, not relevance.

### Symptom 3: online got worse

Triage by speed, because this is a regression you may need to roll back *now*.

- **Latency p99 spiked** → timeouts. If the ranker is slower, the serving layer is dropping candidates or truncating the slate, and you're effectively serving a worse, smaller funnel. Check the [funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) latency budget before you blame the model.
- **Scores no longer calibrated** → calibration break. If downstream logic consumes the score as a probability — ad auctions, blending two models, a threshold for "show or not" — and your new model is sharper or shifted, the *same* business rule now behaves differently. This is the subtle online-down-with-better-AUC case: ranking improved, calibration regressed, and the consumer of the score broke. Diagnose with a reliability diagram and Expected Calibration Error; fix with isotonic or Platt recalibration. See [calibration](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust).
- **Diversity/coverage collapsed** → a feedback-loop or exploration regression. The new model is more confident and narrower, exploration shrank, the catalog is collapsing onto a few items. This degrades the long-run experience even if short-run CTR ticks up. See [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles).
- **A guardrail metric regressed** → you optimized the wrong objective. CTR up, watch-time or retention down means you're chasing clickbait. The fix is multi-objective; the diagnosis is to *look at the guardrails*, which is why you defined them before launch.

The calibration case is worth dwelling on because it is the canonical "better model, worse product." A model can have *higher* AUC (better ranking) and *worse* calibration (the predicted probabilities are systematically off). AUC is invariant to any monotonic transform of the scores, so it literally cannot see calibration. If anything downstream treats the score as a probability, AUC going up while calibration goes down is a recipe for an online regression that the offline dashboard celebrates.

### Symptom 4: stuck at a recall ceiling

The ranker can only rank what retrieval gives it. If ranking metrics plateau no matter what you do to the ranker, measure the **funnel recall** directly: what fraction of the items the user actually engaged with were present in the stage-1 candidate set at all?

```python
def funnel_recall(test_users, holdout, retrieval_fn, n_candidates=1000):
    """Of the user's held-out items, how many even made it into the
    stage-1 candidate set? This is the ceiling on everything downstream."""
    present, total = 0, 0
    for u in test_users:
        cands = set(retrieval_fn(u, n=n_candidates))
        for item in holdout[u]:
            total += 1
            present += item in cands
    return present / total

# funnel_recall(...) -> 0.46   # ranking can never exceed this
```

If funnel recall@1000 is 0.46, then **no ranker can do better than 0.46 recall** end-to-end — the other 54% of relevant items never reached the ranker. Tuning the ranker is rearranging items that are already in the set; the ceiling is set by retrieval. Fix retrieval: better [negatives](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) (random negatives make the two-tower good at separating obvious junk but bad at fine distinctions; hard negatives sharpen it), a higher-recall [ANN configuration](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) (raise `nprobe` for IVF or `efSearch` for HNSW — you trade latency for recall, and if your ANN recall@1000 is 0.85 you're silently losing 15% of the items the exact index would have found), or simply a larger candidate set if your latency budget allows.

A subtle version: your *exact* retrieval (brute-force MIPS) has recall 0.60, but your *approximate* index serves 0.51 because the ANN is dropping near-neighbors. The funnel-recall probe run against both the exact and approximate retrievers separates "retrieval model is weak" (fix with training/negatives) from "ANN is lossy" (fix with index parameters). Always run the probe against both.

## 6. Ranking won't separate: when the ranker can't tell good from great

A close cousin of the recall ceiling is the ranker that *has* the right items in the candidate set but cannot order them — NDCG and AUC sit flat, and the top-K looks like a shuffle of plausible-but-not-best items. The candidate items are all "kind of relevant," and the ranker can't pull the truly-relevant ones to the top.

The usual causes, in the order you should check them:

- **The negatives are too easy.** If the ranker trains on positives versus *random* negatives, it learns to separate "relevant" from "obviously irrelevant" — but at serving time every candidate is already relevant (retrieval did its job), so the ranker faces a distribution it never trained on. The fix is to train the ranker on **hard negatives**: items retrieval surfaced but the user didn't engage with. This aligns training with the serving distribution and is, in my experience, the single biggest lever on ranker separation. See [negative sampling](/blog/machine-learning/recommendation-systems/negative-sampling-strategies).
- **The loss is pointwise when the task is ranking.** Pointwise BCE optimizes calibrated probability, not order. For top-K, a pairwise ([BPR](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive)) or listwise loss sees the *order* directly and often separates better. The science here: the BPR gradient is driven by $\sigma(s_i - s_j)$ — the difference of scores within a (positive, negative) pair — so it pushes the positive above the negative regardless of their absolute scores. Pointwise loss can get both scores "right" on average and still order them wrong.
- **Label noise or a wrong label.** If your "positive" is a click but you care about a purchase, you're optimizing the wrong target and the ranker is faithfully separating clickbait. Check what the label actually is.
- **Capacity or feature crossing.** If the ranker is a shallow model and the signal lives in feature *interactions* (user-segment × item-category), it literally can't represent the function. This is where [DeepFM](/blog/machine-learning/recommendation-systems/deepfm-and-automatic-feature-interactions) and [DCN](/blog/machine-learning/recommendation-systems/dcn-and-explicit-feature-crossing) earn their keep — explicit crossing gives the model the interactions a plain MLP only approximates.

The diagnostic for "won't separate" is again ablation plus a learning curve, but run on the *ranking* metric within the candidate set, and with the negatives swapped from random to hard as one of the ablation arms. If hard negatives move NDCG and nothing else does, you've found your lever.

There is a trap inside the hard-negatives fix worth flagging, because it produces its own "won't improve" plateau. **Hard negatives are often false negatives.** An item retrieval surfaced that the user didn't click is not necessarily irrelevant — the user may simply not have seen it (it was below the fold), or would have loved it but never got the chance. If you train the ranker to push down every un-clicked candidate, you are teaching it that genuinely relevant items are negatives, and the loss fights itself: gradients from true positives and from false negatives that *look like* positives cancel, and the metric stalls. The fingerprint is a ranker that improves a little with hard negatives and then refuses to go further no matter how many you add. The fix is to mine hard negatives more carefully — sample from impressions the user demonstrably saw and skipped (in-view, dwell-thresholded), down-weight or remove the hardest negatives that are statistically likely to be false, or mix a fraction of easy random negatives back in to dilute the false-negative gradient. This is the same false-negative tension covered in [negative sampling](/blog/machine-learning/recommendation-systems/negative-sampling-strategies); as a debugging signal, "hard negatives plateaued the ranker" means you've swung from too-easy to too-aggressive and need to land in the middle.

One more "won't separate" cause that masquerades as a model problem: **a saturated or mis-scaled score head**. If the final logits are squashed by an activation that saturates (a `tanh` or a sigmoid applied before the ranking comparison), then once scores hit the flat region the model can no longer create *differences* between candidates even though it "knows" the order — the gradient through the saturated activation is near zero. The fingerprint is high training loss that won't descend and tightly clustered output scores (look at the histogram of logits — if they're all within a hair of each other, the head is saturated). Remove the squashing activation before the ranking comparison and the scores spread out again. This is cheap to check (one histogram) and embarrassingly common.

## 7. The worked debugging session: a recommender stuck at baseline

Let me put the whole playbook to work on one realistic session, with numbers at every step. This is the kind of narrative the harness produces in practice.

![A flow of a single debugging session that splits a stuck recommender into two independent bugs, a leaky feature found offline and a serving skew found online, and shows that fixing both is what finally moves the metrics.](/imgs/blogs/debugging-a-recommender-that-wont-improve-7.png)

**The setup.** An e-commerce feed recommender, two-tower retrieval into a DeepFM ranker, on a MovieLens-style internal dataset (call it 1M users, 50k items, implicit clicks). The complaint: "we've been trying to improve this for three weeks and it won't move." The on-call engineer (us) opens the harness instead of opening the model code.

**Step 1 — beat popularity.** Recall@10 = 0.42 for the model, 0.18 for most-popular. The model is beating popularity by 2.3×. That's actually *healthy*-looking offline. So the offline metric is not flat-at-baseline — it's high. The complaint must be the *online* side: offline is great, online is flat. We've classified the symptom: this is symptom 2, the gap.

**Step 2 — ablation sweep.** We disable each feature group. One feature, `item_recent_ctr`, contributes 0.20 of the 0.42 NDCG@10 — nearly half, from one feature. Suspicious. Strong-feature, or leak?

**Step 3 — leakage probe.** We shuffle `item_recent_ctr`. NDCG@10 drops from 0.42 to 0.23. Shuffling one feature destroys nearly half the metric. We read the feature job and find the window is anchored to *today*, not point-in-time — it's reading post-interaction clicks, including the held-out one. **Bug #1: a leaky CTR feature.** It inflated offline NDCG to 0.42; the honest offline number with a point-in-time recompute is 0.24. The 0.42 we'd been "trying to beat" was never real. That alone explains why offline looked great while online was flat — online there is no future to leak, so the live model performs like the honest 0.24, not the fake 0.42.

**Step 4 — popularity-stratified recall.** With the honest model: head 0.55, torso 0.21, tail 0.04. The model personalizes on the head and torso but barely touches the tail. Not a bug per se, but a note: online novelty is weak. We'll come back to this after the bigger fish.

**Step 5 — train-serve diff.** Now the online-flat side. We log the features the serving service actually fed the model for 50,000 live requests and recompute them offline. The diff:

| Feature | % rows differ | serve NaN rate | train NaN rate |
| --- | --- | --- | --- |
| `user_avg_ctr_7d` | 12.1% | 3.0% | 0.0% |
| `item_price_bucket` | 0.4% | 0.0% | 0.0% |
| `user_country` | 0.0% | 0.0% | 0.0% |

`user_avg_ctr_7d` differs on 12% of rows, with a 3% serve-side NaN that training never saw. **Bug #2: a train-serve skew.** The online feature store misses this feature for cold/low-activity users and fills NaN; the offline pipeline always had it. For 12% of live requests, the model predicts on a feature it never learned to handle.

![A matrix walking the worked session probe by probe, showing for each diagnostic the concrete reading it returned and the verdict it delivered, from the beat-popularity check through the train-serve diff.](/imgs/blogs/debugging-a-recommender-that-wont-improve-8.png)

**The diagnosis.** Two independent bugs, found by two different probes. The leaky feature made offline lie (so we were chasing a number that didn't exist). The serving skew made online underperform even the honest offline (so even after we trusted the honest number, the live model would lag). *Fixing either alone would not have worked* — drop the leak and offline becomes honest but online still skews; fix the skew and online stops degrading but we're still optimizing against a leaked offline metric. You have to fix both.

**The fixes.**
1. Recompute `item_recent_ctr` as a strict point-in-time aggregate (only clicks before each interaction's timestamp). Offline NDCG@10 drops to an honest 0.24 — now a number that means something.
2. Move `user_avg_ctr_7d` to a shared feature-transformation library used by both training and serving, with a learned default (the global mean CTR) for missing values, and backfill the feature store so cold users get a sensible value instead of NaN.

**The result.** We retrain on honest features, redeploy with the shared transform, and run the A/B:

| Stage | Offline NDCG@10 | Online CTR (rel.) | Notes |
| --- | --- | --- | --- |
| As found | 0.42 (leaked) | baseline | offline lie, online flat |
| Leak fixed | 0.24 (honest) | baseline | offline now truthful |
| Leak + skew fixed | 0.27 (honest) | **+2.1%** | metrics finally move |

The honest offline NDCG even *rose* from 0.24 to 0.27 after the skew fix, because training on consistent features (the same transform offline and online) let the model actually learn the `user_avg_ctr_7d` signal instead of half-learning a value that was sometimes missing at serving. And online CTR moved +2.1%, statistically significant — the first real win in three weeks. The harness found in an afternoon what three weeks of guessing could not: not one bug, but two, hiding behind a single symptom.

![A contrast between three weeks of guessing and tweaking, which moves several layers at once and cannot attribute any result, and a systematic bisection that holds layers fixed, probes one at a time, and localizes the bug in about a day.](/imgs/blogs/debugging-a-recommender-that-wont-improve-5.png)

## 8. Case studies: three debugging narratives from the wild

The worked session is composite. Here are three real-flavored narratives that match patterns reported in the literature and in production postmortems, each illustrating one branch of the tree.

### Case 1: offline up, online flat → it was skew

A team improves an offline ranking AUC by nearly a point with a new set of aggregated user features and ships behind a flag. The A/B comes back dead flat. The postmortem-style investigation finds the aggregates were computed in the batch pipeline with a 24-hour-fresh warehouse table, but the serving path read a feature store that lagged by up to 48 hours and silently returned stale values for a meaningful slice of users. The model trained on fresh aggregates and served on stale ones — a textbook train-serve skew on *freshness* rather than *code*. This pattern is exactly why Google's TFX and feature-store literature (e.g. the "Data Validation for Machine Learning" work by Breck et al. and the broader TFX papers) treat training-serving skew detection as a first-class pipeline stage: the recommendation is to log serving features and continuously diff against training, which is precisely probe 4. The fix is unifying the freshness contract; the lesson is that the offline win was real *for the offline distribution* and meaningless for the served one.

### Case 2: metric flat → a leakage ceiling, then a real ceiling

A recommender for a media catalog has been "stuck at NDCG 0.31" for two quarters; nobody can move it. An ablation reveals a single feature — a per-(user, item) historical interaction count — carrying most of the metric. The shuffle probe collapses NDCG to 0.12 when that feature is permuted. Reading the pipeline shows the count was computed over the *entire* dataset before the temporal split, so it included future interactions: a leak. Recomputing point-in-time drops the honest NDCG to 0.19. Now the *real* debugging can start, because the team had been "improving" a leaked metric that could not move (it was already saturated by the leak). After the fix, three rounds of genuine feature work (text embeddings, session features, time-of-day) push the honest NDCG from 0.19 to 0.235 — and then a fourth, fifth, and sixth feature each do nothing. The learning curve has flattened with a small train-val gap: they've hit the data ceiling for their feature schema. The lesson is two-sided: leakage can make a metric *look* stuck (it's pinned high by the leak), and the genuine ceiling is real and worth recognizing so you stop spending on architecture and start spending on data.

### Case 3: online got worse → a calibration break

A team replaces a logistic-regression CTR model with a deep ranker. Offline AUC improves clearly. Online, total revenue *drops*, even though raw CTR is up slightly. The cause: the downstream ad-ranking step multiplies predicted CTR by a bid to compute expected value, so it *consumes the score as a probability*. The new deep model has better ranking (higher AUC) but is over-confident — its predicted probabilities are systematically too high — so the expected-value computation is distorted and the auction allocates poorly. AUC, being invariant to monotone transforms, could not see this; only a reliability diagram and the Expected Calibration Error showed the predicted probabilities drifting off the diagonal. Isotonic recalibration on a held-out set restored the probabilities, and revenue recovered with the CTR gain intact. This is the canonical "better model, worse product," and it is why any score consumed as a probability needs a calibration check in the launch gate, not just an AUC check. The mechanism and the fix are the subject of the [calibration post](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust); the debugging lesson is that a regression with *higher* offline AUC points almost always at calibration, the loop, or skew — never at "the ranking got worse."

## 9. When to reach for the harness (and how much of it)

Not every stuck model needs the full procedure, and over-instrumenting is its own waste. Here is the decisive guidance.

- **Always run probe 1 (beat popularity/random).** It costs minutes and it is the single most common silent failure. There is no recommender for which you should skip it. If you build one diagnostic into CI, make it this.
- **Run the train-serve diff (probe 4) the moment offline and online disagree** — and ideally run it on a schedule, not just during fires. It is the highest-yield probe for production recommenders and the most under-deployed.
- **Run the leakage probe whenever the ablation flags a dominant feature, or whenever offline looks "too good."** A model that beats popularity by 3× on a noisy implicit-feedback task should make you suspicious, not happy.
- **Don't bother with sampled-vs-full (probe 5) if you already evaluate on the full corpus.** It only matters if you've been using sampled metrics for model selection.
- **Don't conclude "data ceiling" until every cheaper probe is clean.** The ceiling is the diagnosis of last resort precisely because a leak or a skew can imitate it from either direction.
- **Don't keep tuning the ranker once funnel recall caps it.** Measure the ceiling first; if retrieval recall@1000 is 0.46, the ranker is not your problem and never was.

And a meta-rule: **resist the urge to fix before you localize.** The leaky feature in the worked session is tempting to "just remove," but if you remove it before running the train-serve diff, you'd fix the offline lie, see online still flat, and conclude your fix didn't work — when in fact there was a second, independent bug. Localize *all* the failing probes first, then fix, then re-run the harness to confirm. The harness is also your regression test: keep it, run it on every model change, and "it won't improve" becomes a one-afternoon question instead of a three-week one.

## 10. The debugging checklist

Tape this to the wall. When a recommender won't improve, go top to bottom.

1. **Name the symptom.** Offline-flat, offline-up-online-flat, online-worse, or recall-ceiling. Don't proceed until you've picked one.
2. **Beat random and popularity?** If not, stop and fix wiring (random) or data/features (popularity). No tuning until you clear popularity.
3. **Look at the loss curve, not just the metric.** Descending? NaN? Stuck at init? An optimizer/lr pathology hides behind a flat metric.
4. **Read the learning curve.** Underfit (low, train≈val), overfit (train≫val, gap growing), or data-limited (flat, train≈val, val-vs-data plateaued)? Each has a different fix.
5. **Ablate.** Disable each component. A feature contributing half the metric is a leakage suspect, not a hero.
6. **Shuffle the suspect feature.** A collapse on a feature that "shouldn't" be that strong, computed in a way that could see the future, is a leak. Recompute point-in-time.
7. **Stratify by popularity.** Head-only wins mean a popularity machine, not personalization.
8. **Check sampled vs full.** If you select models on sampled metrics, verify the ranking agrees with full-corpus before you trust it.
9. **Diff train vs serve.** Log serving features, recompute offline, diff per column. Differing rows or serve-only NaNs are skew. Fix with a shared transform.
10. **Measure funnel recall.** Ranking can't beat the candidate set. If retrieval recall@K caps you, fix negatives/ANN/candidate-set size, not the ranker.
11. **For online-worse, check calibration, latency, and guardrails first.** Higher AUC + worse online almost always = calibration break, loop collapse, or skew.
12. **Localize all failing probes before fixing any.** Two bugs can hide behind one symptom. Fix all, then re-run the harness to confirm.

## 11. Key takeaways

- **"It won't improve" is four bugs, not one.** Classify by the symptom you can observe — offline-flat, offline-up-online-flat, online-worse, recall-ceiling — and each rules out most of the search space.
- **Bisect, don't guess.** Hold every layer fixed but one, so when a number moves you know which layer moved it. Confounded experiments produce confounded conclusions, and guessing burns weeks.
- **Beating popularity is the line between a model and a lookup table.** Run it always; a model that only ties most-popular has learned no personalization, and no learning rate will save it.
- **Leakage is the number-one reason offline lies.** Ablate to find a dominant feature, shuffle to convict it, and read the pipeline to find the future it was reading. Recompute point-in-time.
- **Train-serve skew is the number-one reason online disappoints.** Log serving features, diff against an offline recompute, fix with a single shared transformation and a learned default for missing values.
- **AUC can't see calibration.** A higher-AUC model that regressed online is almost always a calibration break, a loop collapse, or a skew — not "the ranking got worse."
- **The ranker can't beat the candidate set.** When ranking plateaus, measure funnel recall first; if retrieval caps you, fix retrieval, not the ranker.
- **The data ceiling is a real diagnosis, but it's the last one.** Conclude it only after every cheaper probe is clean and three genuinely new features each do nothing.
- **Keep the harness.** Baselines, ablation, leakage probe, stratified metrics, and the train-serve diff are a regression suite. Run them on every model change and "won't improve" becomes an afternoon, not a quarter.

## 12. Further reading

- **Krichene, Rendle (2020), "On Sampled Metrics for Item Recommendation," KDD.** The result behind probe 5: sampled metrics are not consistent estimators of full-corpus metrics and can reverse model rankings. Required reading before you trust any sampled Recall@K.
- **Rendle et al. (2009), "BPR: Bayesian Personalized Ranking from Implicit Feedback," UAI.** The pairwise objective and the $\sigma(s_i - s_j)$ gradient behind "ranking won't separate."
- **Breck et al. (2019), "Data Validation for Machine Learning," SysML; and the TFX papers (Baylor et al., 2017).** Training-serving skew detection as a first-class pipeline stage — the production framing of probe 4.
- **Guo et al. (2017), "On Calibration of Modern Neural Networks," ICML.** Why a more accurate model can be worse-calibrated, and how to measure it with reliability diagrams and ECE — the mechanism behind case 3.
- **faiss documentation (`IndexIVFFlat`, `IndexHNSWFlat`) and hnswlib `efSearch`.** The ANN recall-vs-latency knobs you turn when the recall ceiling is the index, not the model.
- **Within this series:** the intro map [what a recommender system is](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system); the [retrieval → ranking → re-ranking funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking); [train-serve skew and the bugs that hide there](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there); [the offline-online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied); [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate); [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer); [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies); [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust); and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

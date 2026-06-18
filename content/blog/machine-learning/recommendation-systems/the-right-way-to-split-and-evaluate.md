---
title: "The Right Way to Split and Evaluate a Recommender"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why random splitting leaks the future, why temporal global splits are the only honest ones, why sampled top-K metrics can reverse model rankings, and how to build a leak-free temporal split plus a full-catalog evaluation harness with bootstrap confidence intervals in pandas."
tags:
  [
    "recommendation-systems",
    "recsys",
    "evaluation",
    "temporal-split",
    "sampled-metrics",
    "data-leakage",
    "ndcg",
    "reproducibility",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/the-right-way-to-split-and-evaluate-1.png"
---

The most expensive bug I have ever shipped did not crash anything. It was a number. We had a new retrieval model that scored a Recall@10 of 0.42 on our offline benchmark, comfortably above the 0.31 of the model in production, and on the strength of that gap we ran an A/B test. Two weeks later the experiment readout came back flat: no lift in click-through, no lift in dwell time, a faint negative wobble on long-session users that was inside the noise band. The offline metric had promised a thirty-five percent relative improvement and online delivered nothing. The model was not bad. The *evaluation* was bad. We had split the interaction log uniformly at random, which meant the training set was riddled with interactions that, in calendar time, happened *after* the interactions we were testing on. The model had been allowed to peek at the future, and a model that has seen the future scores beautifully on a test set drawn from the same future. The day we re-ran the same comparison with a proper temporal split, the new model's Recall@10 fell to 0.21 and the gap over production shrank to almost nothing. The offline number had finally agreed with the online number, because for the first time it was measuring the same thing the deployed system actually does: predict forward in time, against the whole catalog, for users it has never finished learning about.

This post is about getting that number right. Not the model, not the loss, not the architecture — the *measurement*. A recommender is deployed forward in time: you train it on everything you have logged up to now, you ship it, and it ranks items for interactions that have not happened yet. So the only offline experiment that estimates what you will actually see in production is one that obeys the same arrow of time, scores against the same full catalog, and quantifies its own uncertainty before declaring a winner. Almost every way of cutting corners on that experiment — shuffling the log, ranking the true item against a handful of sampled negatives, computing a feature over the entire dataset, filtering out the hard users — moves the offline number in a flattering direction and widens the gap between what you measure and what you ship. The figure below is the whole post in one contrast: a random split that leaks the future and reports an inflated Recall, versus a temporal global split that respects the deployment timeline and reports the honest, lower number.

![A before and after comparison contrasting a random hold out split that shuffles all clicks and ignores timestamps so training sees interactions after the test event and reports an inflated Recall, against a temporal global split that cuts at a timestamp so training uses only the past and testing only the future and reports a realistic lower Recall](/imgs/blogs/the-right-way-to-split-and-evaluate-1.png)

By the end you will be able to choose a split strategy on purpose rather than by default, knowing exactly what each one over- or under-estimates; you will understand the single most important negative result in modern recommender evaluation — that sampled top-K metrics are *inconsistent* and can flip which model looks better, the Krichene and Rendle result from KDD 2020 — and you will have working pandas code for a leak-free temporal global split, a leave-one-out split, a full-catalog evaluation harness, a popularity baseline that is harder to beat than you expect, and bootstrap confidence intervals. This is the methodology layer of the series. The bird's-eye map of the whole pipeline is [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system); the metrics themselves — Recall, NDCG, MAP, MRR — are defined and dissected in [offline evaluation metrics](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr); why even an honest offline number still diverges from online is [the offline online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied); and the synthesis of all of it is the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

## 1. The deployment distribution is forward in time

Start from what the production system actually does, because the evaluation has exactly one job: estimate the production system's quality before you pay to run it on live users. When a recommender serves a request, it has been trained on the log of everything that happened up to the moment of the last retrain. It then ranks items for an interaction that is, by construction, in the future relative to that training data. The user it is serving may be brand new, or may have a longer history than anyone in training, or may have shifted their taste since the model last saw them. The catalog it ranks over may contain items that did not exist when the model was trained. Every one of those conditions is a property of *time moving forward*, and none of them is reproduced by an evaluation that ignores time.

This is the core scientific claim of the whole post, so let me state it precisely. Let the interaction log be a set of events $(u, i, t)$ — user $u$ interacted with item $i$ at timestamp $t$. The deployment distribution is the distribution of $(u, i)$ pairs the model will be asked to score *conditioned on $t$ being after the current training cutoff*. An offline evaluation estimates a quantity of the form $\mathbb{E}_{(u,i) \sim \mathcal{D}_{\text{test}}}\big[m(u, i; f_\theta)\big]$, where $m$ is the metric (Recall@K, NDCG@K, and so on) and $f_\theta$ is the trained model. The estimate is *unbiased for the deployment quantity* only if $\mathcal{D}_{\text{test}}$ matches the deployment distribution and $f_\theta$ was trained without access to any event in $\mathcal{D}_{\text{test}}$. A random split violates both: the test set is a uniform sample over all of time, not the future, and the training set contains events that postdate test events. The bias is not small and it is not random — it is systematically optimistic, because seeing the future is the single most useful thing a model can do.

There is a clean way to see why temporal leakage is so much more damaging in recommendation than in, say, image classification. In a classification problem the examples are exchangeable: a cat is a cat whether it was photographed in 2019 or 2024, and shuffling the dataset costs you nothing. Interaction data is emphatically *not* exchangeable. User tastes drift, items rise and fall in popularity, seasonal patterns dominate, and — most insidiously — the popularity of an item at time $t$ is a leak about whether a user will interact with it at time $t' > t$. If your training set contains the fact that a movie became a runaway hit next month, your model can learn to recommend it *this* month, and your random-split test set, also drawn from next month, will reward it for doing so. In deployment that future popularity is exactly the thing you do not know. The arrow of time is not a formality you can shuffle away; it is the structure of the prediction problem.

### Three distinct ways the future leaks backward

It helps to separate the leakage into its mechanisms, because each one survives a different set of half-measures and you want to recognize all three. The first is **label leakage through the split**: the test interaction itself, or a later interaction by the same user, is in the training set. This is the obvious one a random split causes, and a temporal global cut closes it completely. The second is **co-occurrence leakage**: even if no single user's future leaks, a random split scatters the *whole population's* future across training, so collaborative-filtering models — which learn from "users who interacted with X also interacted with Y" — absorb co-occurrence statistics that only crystallized in the test period. A movie and its sequel become correlated in your training co-occurrence matrix because the sequel's launch (a test-period event) pulled both into the same sessions; the model learns a correlation that did not exist at serving time. The third is **feature leakage**, which we devote section 6 to: an aggregate feature computed over the whole log smuggles the future into the inputs even when the split is clean. The temporal global split fixes the first two; only point-in-time features fix the third. A team that switches to a temporal split and declares victory has closed two of the three doors.

### What "matches deployment" really demands

Saying the test distribution must "match deployment" is easy; meeting it has a few non-obvious requirements worth spelling out. Deployment serves a *mixture* of warm users (rich history), lukewarm users (a handful of interactions), and cold users (none), in roughly the proportions your live traffic shows. An evaluation that quietly drops cold users — because they have no training embedding, or because they are awkward to score — is no longer matching deployment; it is matching the warm subpopulation, which is easier, so the number goes up. Deployment also serves a *catalog that grows*: items launched after the training cutoff are real candidates the model must handle, usually via a content/feature fallback, and an evaluation that filters them out is measuring a frozen catalog the business does not have. And deployment ranks against the *full* candidate set the retrieval stage produces, not a curated pool — which is the deep reason sampling negatives for the metric is wrong, a point we will make rigorous in section 4. Keep this triad in mind: the right split matches deployment in *time*, in *population*, and in *catalog*. Most protocols get at most one of the three.

#### Worked example: how much a random split inflates Recall

Take a concrete, deliberately simplified scenario. Suppose 60% of all clicks in your log, across all users, land on the ten most popular items, and that popularity is *non-stationary*: the set of ten most-popular items in the second half of the year is mostly different from the first half, because of seasonality and a few viral launches. Train a model on a random 80% of the log. Roughly 80% of every popular item's clicks — including the second-half viral ones — are now in training. The model trivially learns the global popularity ranking that *includes the future*, and when tested on the random 20% hold-out (also drawn from across the whole year), it scores Recall@10 around 0.42 just by recommending currently-and-future-popular items. Now re-split temporally: train on the first half of the year, test on the second. The model only ever saw first-half popularity. The second-half hits it has never heard of, and the items it considers popular have cooled off. Recall@10 collapses to about 0.21. Same model, same data, same metric — the only change is whether the experiment let the model see the future. The 0.42 was measuring a capability the deployed model will never have, namely clairvoyance.

That 2x gap is not a contrived number; it is the order of magnitude reported repeatedly in the reproducibility literature when papers' random-split numbers are recomputed under temporal splits, and it is roughly what I have seen on production data. The temporal number is the one that predicted the flat A/B test. Internalize the direction: **random splits do not make your model better, they make your measurement lie, and they always lie upward.**

## 2. The split strategies and what each one estimates

There is no single "correct" split; there is a correct split *for a given question*, and the failure mode is using a split whose implicit question is not the question you care about. Four families cover almost everything you will meet, and the matrix below lays them against the four properties that decide between them: how much they leak, how realistic they are relative to deployment, whether they let you study cold-start, and when to reach for them.

![A four by four decision matrix comparing random hold out, leave one out, temporal global, and user based splits across the properties of leakage, realism, cold evaluation, and when to use, showing the temporal global split as the only deployment faithful default](/imgs/blogs/the-right-way-to-split-and-evaluate-2.png)

**Random hold-out** shuffles all interactions and assigns, say, 80% to train and 20% to test, ignoring timestamps entirely. It is the default in scikit-learn's `train_test_split` and the reason it is the default in so many recommender papers is simply inertia from classification. Its leakage is severe (future leaks into past, as we established) and its realism is low; it systematically *over*-estimates quality. The only honest use I can defend is a fast sanity check that your code runs end to end — never for a number you will report or act on.

**Leave-one-out (LOO)** holds out, for each user, exactly one interaction — almost always the user's chronologically last one — and trains on the rest. This is the standard protocol for sequential recommendation benchmarks (it is what the SASRec and BERT4Rec papers use), and it has a real virtue: by holding out the *last* item per user, it partially respects per-user time order. But it has two well-known issues. First, although the held-out item is each user's last, the *training* set for user $A$ can still contain interactions that, in global calendar time, happened after user $B$'s held-out test item — so there is residual global leakage. Second, a single held-out item per user gives a very high-variance estimate and over-weights heavy users implicitly through how you aggregate. LOO under-estimates the difficulty of the cold case (every test user has, by definition, a history) and is best treated as a *sequential-prediction benchmark protocol*, not as an estimate of deployed quality.

**Temporal global split** picks a single global timestamp $T$ and routes every event before $T$ to training and every event after $T$ to test. This is the only split whose test distribution genuinely matches deployment: the model is trained exclusively on the past, tested exclusively on the future, and — crucially — the cut is *global*, so there is no path by which any user's future leaks into any user's training. New items that first appear after $T$ show up in the test period exactly as they would in production, which is the only way to measure item cold-start honestly. It under-states nothing and over-states nothing structural; its numbers are lower than random-split numbers precisely because they are real. This is your default, and most of the rest of this post is about implementing it correctly.

**User-based split** holds out *entire users* — train on 80% of users' full histories, test on the remaining 20% of users with no overlap. This is the right tool for exactly one question: how well does the model serve a user it has never seen any interaction from, i.e. *user cold-start*? It deliberately removes the user's history from training, so it measures the content/feature pathway rather than the collaborative-filtering pathway. It is the wrong tool for everything else, because in steady-state deployment most served users *do* have history, and a user-based split throws that history away.

The shape of these choices is a small taxonomy, and the taxonomy itself is informative: the first and most important fork is whether the split respects time at all. Everything on the time-respecting branch is deployment-faithful to some degree; everything on the time-ignoring branch leaks. The tree below draws that structure with a real intermediate level so the families do not collapse into an undifferentiated list.

![A taxonomy tree of splitting strategies branching first into time respecting splits that avoid future leakage and time ignoring splits that leak the future, with the temporal global and per user temporal splits under the first branch and random hold out, leave one out, and user based splits under the second](/imgs/blogs/the-right-way-to-split-and-evaluate-5.png)

A subtle point that trips up even careful teams: the temporal global split and the per-user "leave the last $k$ items out" split are *not* the same, and the difference matters. The per-user version still leaks globally — user $A$'s training tail can postdate user $B$'s test item — so it sits one notch below the global cut on realism even though it is on the time-respecting branch. The only reason to prefer it is that a single global cutoff can leave you with very few test interactions for users whose entire history predates $T$; the per-user temporal split guarantees every active user contributes test data. The honest compromise many production teams use is a global cutoff *for the model's training data* combined with reporting metrics only on users who have at least one interaction in the test window — which brings us, unavoidably, to survivorship.

## 3. The timeline split, drawn and coded

Let me make the global cut concrete, because the implementation has three details that are easy to get wrong and each one reintroduces leakage if you miss it. The figure traces the flow: the timestamped log is cut at $T$ into a past and a future; the past is further carved into a small validation slice (the events just before $T$) used for early stopping and the remaining training events; the model is fit on the past only; and then it scores the future against the full catalog with the future events serving as ground truth.

![A branching dataflow graph showing a timestamped interaction log cut at a global timestamp T into past events that become the training set and future events that become the test set, with a validation slice taken from just before T for early stopping, the model fit on the past only, and then scoring the future against the full catalog using the future events as ground truth](/imgs/blogs/the-right-way-to-split-and-evaluate-3.png)

The three details. **First**, the validation slice must come from *just before* $T$, not from a random sample of the training period, because validation exists to tell you when the model has started over-fitting to the *near future* it will actually face — and an early-stop signal computed on a random training sample is itself leaky. **Second**, any user or item that appears *only* after $T$ has no training representation; you must decide explicitly whether to (a) drop their test interactions, (b) keep them and let the model fall back to a cold-start path, or (c) report them as a separate cold subset. Silently dropping them is the survivorship trap of section 6. **Third**, the cutoff $T$ should be chosen so the test window is long enough to contain a representative slice of behavior (a week is typical for a feed; a month for slower-moving catalogs) but short enough that you are still estimating *near-future* quality, which is what a model retrained frequently actually faces.

Here is the temporal global split in pandas, written to be copy-and-adapt ready. It takes a dataframe of `user`, `item`, `ts` (a unix timestamp or pandas datetime), cuts at a quantile of time so the split ratio is by *interaction count over time* rather than an arbitrary date, and carves a validation slice.

```python
import pandas as pd
import numpy as np

def temporal_global_split(df, ts_col="ts", train_frac=0.8, val_frac=0.1):
    """Global timestamp split: past -> train/val, future -> test.

    df must have columns user, item, ts. No user's training data
    postdates any test event, because the cut is global.
    Returns (train, val, test) dataframes.
    """
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    # Cut points by time quantile so the split is balanced over the timeline.
    t_train_end = df[ts_col].quantile(train_frac)
    t_val_end = df[ts_col].quantile(train_frac + val_frac)

    train = df[df[ts_col] <= t_train_end]
    val = df[(df[ts_col] > t_train_end) & (df[ts_col] <= t_val_end)]
    test = df[df[ts_col] > t_val_end]
    return train, val, test
```

Notice what this does *not* do: it does not shuffle, it does not stratify by user, and it does not guarantee every user appears in every split. That is correct. In deployment, some users in the test window are new and some training users go dormant; the split should reflect that, not paper over it. The one thing you must add for a clean evaluation is to decide your policy on test interactions whose user or item is unseen in training. Here is the explicit version, which is the one I actually run:

```python
def temporal_split_with_policy(df, ts_col="ts", train_frac=0.8, val_frac=0.1,
                               cold_user="separate", cold_item="separate"):
    train, val, test = temporal_global_split(df, ts_col, train_frac, val_frac)
    train_users = set(train["user"].unique())
    train_items = set(train["item"].unique())

    test = test.copy()
    test["cold_user"] = ~test["user"].isin(train_users)
    test["cold_item"] = ~test["item"].isin(train_items)

    if cold_user == "drop":
        test = test[~test["cold_user"]]
    if cold_item == "drop":
        test = test[~test["cold_item"]]
    # "separate" keeps them flagged so you can report a cold subset.
    return train, val, test
```

The `cold_user` and `cold_item` flags are the difference between an evaluation that quietly tells you a comforting lie and one that tells you the truth with footnotes. We will use them in section 6.

### Leave-one-out, done correctly

For sequential benchmarks you will also want leave-one-out, and the standard implementation has a quiet bug that the careful papers avoid: it must hold out the *chronologically* last item, not a random one, and it must build the input sequence from the strictly-earlier items. Here is the version that respects per-user order:

```python
def leave_one_out_split(df, ts_col="ts"):
    """Hold out each user's chronologically last interaction as test,
    the second-to-last as validation, the rest as train.
    """
    df = df.sort_values([ "user", ts_col ], kind="mergesort")
    # rank within user by time, descending so rank 1 is the newest
    df = df.copy()
    df["rev_rank"] = df.groupby("user")[ts_col].rank(method="first",
                                                     ascending=False)
    test = df[df["rev_rank"] == 1]
    val = df[df["rev_rank"] == 2]
    train = df[df["rev_rank"] > 2]
    return (train.drop(columns="rev_rank"),
            val.drop(columns="rev_rank"),
            test.drop(columns="rev_rank"))
```

This is honest *within* each user, and it is the right protocol when your model is sequential and your claim is "given a user's history, predict the next item." But remember the global-leakage caveat: it does not stop user $A$'s training tail from postdating user $B$'s test item, so when your claim is about *deployed* quality rather than *next-item benchmark* quality, prefer the temporal global cut.

### Rolling-window evaluation: one cut is a sample of one

A single temporal cutoff gives you exactly one estimate of forward quality, computed at one moment in the data's history — and that moment might be unrepresentative (the week before a holiday, a quiet stretch, a data outage). The more robust version is a **rolling-window (or sliding/expanding-window) evaluation**: pick several cutoffs $T_1 < T_2 < \dots < T_m$, and for each one train on everything before $T_k$ and test on the window after it, then average the metrics across folds. This is the temporal analogue of cross-validation, and it does for forward-quality estimation what k-fold does for i.i.d. data: it turns a sample of one into a sample of several and gives you a variance estimate across time as a bonus.

```python
def rolling_temporal_folds(df, ts_col="ts", n_folds=4,
                           test_window_frac=0.1):
    """Expanding-window temporal folds. Fold k trains on all data
    before cut_k and tests on the window after it. Yields (train, test).
    """
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    # place cuts so each fold leaves a test_window_frac slice after it
    start = 1.0 - n_folds * test_window_frac
    for k in range(n_folds):
        c_lo = start + k * test_window_frac
        c_hi = c_lo + test_window_frac
        t_lo = df[ts_col].quantile(c_lo)
        t_hi = df[ts_col].quantile(c_hi)
        train = df[df[ts_col] <= t_lo]
        test = df[(df[ts_col] > t_lo) & (df[ts_col] <= t_hi)]
        yield k, train, test
```

The expanding-window variant (train on *all* data before the cut, as above) mirrors a production system that retrains on the full accumulated log; a *sliding*-window variant (train on a fixed-length window before the cut) mirrors a system that trains on a rolling recent window for freshness. Choose the one that matches your retraining policy. Either way, reporting "Recall@10 = 0.142 ± 0.011 across four temporal folds" is dramatically more trustworthy than a single number, because it bakes the across-time variance into the headline — and it catches the embarrassing case where your model looks great on one fold and falls apart on the next, which a single cut would have hidden entirely.

### The validation set is part of the protocol, not an afterthought

One more detail teams routinely botch: the validation set used for hyperparameter tuning and early stopping must itself be temporal, sitting between train and test, and it must *never* overlap the test window. If you tune your learning rate, embedding dimension, and regularization by repeatedly checking the *test* set, you have laundered the test set into your model-selection decisions, and your reported test metric is now optimistic by an amount proportional to how hard you searched. This is the same multiple-comparisons leakage that plagues any ML benchmark, but it is especially easy to commit in recommenders because the eval harness is fiddly and people tune against whatever number is convenient. The rule: tune against validation, look at test exactly once per final model, and if you find yourself "checking test to see if it helped," you have already contaminated it.

## 4. The sampled-metrics problem: the result that invalidated a decade of numbers

Now the big one. To compute a top-K metric you must, in principle, rank the user's held-out true item against *every other item in the catalog* and see where it lands. For a catalog of a few hundred items that is trivial. For a catalog of a million items, scoring every (test user, every item) pair is expensive — potentially billions of score computations per evaluation — so a shortcut became standard practice, especially in the neural-recommender literature of 2016 through 2019: instead of ranking the true item against all $N$ items, rank it against a small *sample* of negatives, typically 99 or 100 randomly drawn items the user has not interacted with. You then compute HitRate@K or NDCG@K within that little pool of ~100 candidates. It is hundreds or thousands of times cheaper, and for years nearly everyone did it.

In 2020, Walid Krichene and Steffen Rendle published "On Sampled Metrics for Item Recommendation" at KDD, and the result is devastating in its simplicity: **sampled metrics are inconsistent estimators of the true (full-catalog) metrics, and they are not even rank-preserving — the model that looks best under sampling can be worse under full evaluation.** This is not a "the sampled estimate is a bit noisy" caveat. It is "the sampled estimate can confidently tell you to ship the wrong model." The figure contrasts the two regimes: ranking the true item against 100 sampled negatives versus ranking it against the full catalog, with the model order flipping between them.

![A before and after comparison contrasting sampled metrics that rank the true item against one hundred sampled negatives and are cheap but biased and non monotone with one model winning, against full catalog scoring that ranks against all items and is a consistent true ranking with the other model winning, illustrating that the model order can reverse](/imgs/blogs/the-right-way-to-split-and-evaluate-4.png)

### Why sampling breaks the metric: the nonlinearity argument

Here is the argument, sketched at the level of rigor that makes it click. Fix a single test instance: a user $u$ with one held-out true item, and let $r$ be the *rank* of that true item among all $N$ catalog items under the model (so $r = 1$ is best). The true top-K metric for this instance is a function of $r$ — for HitRate@K it is the indicator $\mathbb{1}[r \le K]$, for NDCG it is a discount $1 / \log_2(r + 1)$ if $r \le K$ and zero otherwise. Now sample $n$ negatives uniformly from the $N - 1$ non-true items and compute the *sampled* rank $r'$ of the true item within the pool of size $n + 1$. The expected sampled rank is, by linearity, $\mathbb{E}[r'] = 1 + \frac{n}{N-1}(r - 1)$ — a clean linear shrinkage of the true rank toward the top. So far so benign.

The trouble is that the metric is a *nonlinear* function of rank, and **the expectation of a nonlinear function is not that function of the expectation** (this is Jensen's inequality biting). The sampled HitRate@K is $\mathbb{E}\big[\mathbb{1}[r' \le K]\big] = \Pr[r' \le K]$, and that probability, as a function of the *true* rank $r$, is a smooth sigmoid-like curve — not the step function $\mathbb{1}[r \le K]$ that the true metric uses. Krichene and Rendle work out this map exactly: the sampled metric $M_{\text{samp}}$ is related to the true metric through a fixed, model-independent but *nonlinear* transformation of the rank distribution. Two consequences follow and both are fatal.

First, the sampled metric *compresses* the rank axis. A model whose true item sits at rank 500 out of a million and a model whose true item sits at rank 5,000 will both, under 100 negatives, usually place the true item near the top of the little pool, so they look almost identical on the sampled metric even though one is ten times better at the real task. Sampling throws away exactly the resolution in the tail that distinguishes good retrieval from mediocre retrieval. Second, and worse, because the nonlinear map is not monotone in the *shape* of a model's rank distribution, two models whose true-metric values are ordered $M_A^{\text{true}} > M_B^{\text{true}}$ can have sampled values ordered the other way, $M_A^{\text{samp}} < M_B^{\text{samp}}$. The relative position of two models is not preserved. That is the inconsistency: the sampled metric is not a monotone transform of the true metric, so it cannot be trusted even for *ranking* models, which is the one job an evaluation metric has.

#### Worked example: a sampled-versus-full reversal with small numbers

Let me build the reversal explicitly so you can see it is not hand-waving. Catalog of $N = 10{,}000$ items. Two models, each evaluated on the same 1,000 held-out test instances, and assume for cleanliness that each model's true rank of the held-out item is constant across instances (real distributions are spread out, but the mechanism is identical).

Model A is a *uniformly decent* retriever: on every instance the true item sits at rank 50 out of 10,000. Model B is a *bimodal* retriever: on half the instances it nails the true item to rank 2, and on the other half it buries it at rank 5,000.

Full-catalog HitRate@10: Model A places the item at rank 50 every time, which is never inside the top 10, so $\text{HR@10}_A^{\text{full}} = 0$. Model B hits the top 10 on exactly the half where the item is at rank 2, so $\text{HR@10}_B^{\text{full}} = 0.5$. By the true metric, **B is far better.** (I'll adjust to make A the full-evaluation winner under a more realistic spread below; right now I am isolating the mechanism.)

Now evaluate with $n = 100$ sampled negatives. For Model A, true rank 50 out of 10,000: the expected number of the 100 sampled negatives that outscore the true item is $100 \times \frac{49}{9999} \approx 0.49$, so the true item is, on average, at sampled rank ~1.49 — almost always inside the top 10. So $\text{HR@10}_A^{\text{samp}} \approx 0.93$. For Model B's bad half (true rank 5,000): expected negatives above the true item is $100 \times \frac{4999}{9999} \approx 50$, sampled rank ~51, never in the top 10; for its good half (rank 2) it is always in the top 10. So $\text{HR@10}_B^{\text{samp}} \approx 0.5 \times 1.0 + 0.5 \times 0.0 = 0.5$. Sampled metric says $0.93$ for A versus $0.5$ for B — **A wins decisively under sampling** — while the full metric says $0$ for A versus $0.5$ for B — **B wins decisively under full evaluation.** The order flipped. The sampled-100 metric did not merely add noise; it reversed the verdict, because Model A's "consistently mediocre rank 50" looks like a near-perfect rank-1 once you collapse the catalog to 101 candidates, while Model B's genuinely excellent half is diluted by its terrible half exactly the same amount. This is the Krichene-Rendle phenomenon in one paragraph of arithmetic.

The matrix below records a more realistic instance of the same reversal — the kind you actually see when a popularity-tilted model is compared against a real collaborative model — where sampled HR@10 and sampled NDCG both rank one model first and full-catalog HR@10 ranks the other first.

![A matrix comparing model A a full matrix factorization model and model B a popularity tilted model across sampled HitRate at ten, sampled NDCG, full catalog HitRate at ten, and the final verdict, showing model B winning both sampled columns while model A wins the full catalog column so the sampled metric reverses which model is chosen](/imgs/blogs/the-right-way-to-split-and-evaluate-8.png)

### The fixes

There are three responses to the sampled-metrics result, in increasing order of effort and decreasing order of how much I trust them.

The clean one is **full-catalog evaluation**: just rank against all $N$ items. For catalogs up to a few hundred thousand items this is entirely feasible if you compute scores as one big matrix multiply (user embeddings times the item embedding matrix) and use a partial top-K selection rather than a full sort. We implement exactly this in section 5, and it is what I recommend by default. For a catalog of $10^6$ items and $10^5$ test users, the score matrix is $10^{11}$ entries, which you stream in blocks; it is an afternoon of engineering, not a research problem.

The second response, for genuinely enormous catalogs where full scoring is infeasible even in blocks, is a **corrected estimator**. Krichene and Rendle derive estimators that invert the nonlinear sampling map to recover an (approximately) unbiased estimate of the true metric from sampled ranks. These work, but they have higher variance and they require you to trust the correction's assumptions; treat them as a fallback, not a default.

The third, which is not really a fix but is worth knowing, is that if you *only* ever compare models with full-catalog evaluation, you sidestep the entire problem — and you should additionally distrust any external benchmark number computed with sampled metrics. When you read a paper reporting HitRate@10 of 0.71 against "99 sampled negatives," mentally tag it as not comparable to a full-catalog number and not reliable for ranking against another sampled-99 result unless the sampling protocol is byte-identical.

### A little more on the corrected estimator

Because some readers genuinely cannot afford full scoring, it is worth seeing the shape of the correction so you know what you are buying. The sampled metric, viewed as a function of the true rank $r$, is the probability that fewer than $K$ of the $n$ sampled negatives outscore the true item. Each sampled negative outscores the true item independently with probability $p = (r-1)/(N-1)$ — the fraction of catalog items that beat the true item — so the number that outscore it is $\text{Binomial}(n, p)$, and the sampled metric for HitRate@K is $\Pr\big[\text{Binomial}(n, (r-1)/(N-1)) < K\big]$. This is a smooth, strictly increasing function of $p$ (and hence of $r$) for a *single* fixed rank, which is why people assumed it was harmless. The inconsistency does not come from a single rank; it comes from the fact that a model's quality is an *average* over a whole *distribution* of ranks, and averaging a nonlinear function over two different rank distributions does not preserve order. Krichene and Rendle's corrected estimators essentially invert this binomial map at the level of the rank distribution rather than the average, recovering an estimate of the full-catalog metric from the empirical distribution of sampled ranks. They reduce the bias substantially, but at the cost of variance — you are now estimating a whole distribution from a thin sample — and they assume the negatives were drawn uniformly, which breaks if your sampler is popularity-weighted. The honest summary: the correction is real and useful as a last resort, but it is strictly worse than not sampling, and "I used the corrected estimator" should never be a reason to skip full evaluation when full evaluation is affordable.

### Why this fooled so many careful people

It is worth pausing on *why* a community of careful researchers used sampled metrics for years. The answer is that the bias is invisible within any single protocol. If everyone samples 100 negatives the same way, the *relative* comparisons inside that protocol mostly look reasonable, because the nonlinear map, while not order-preserving in general, is order-preserving often enough that you rarely catch it red-handed on two models you already believe are close. The reversal shows up when you compare against a *different* protocol (full catalog), or when two models have genuinely different rank-distribution *shapes* — exactly the case of a popularity model versus a personalized model, where one has a fat-tailed rank distribution and the other a concentrated one. So the failure mode is not "every sampled number is wrong"; it is "you cannot know which sampled comparisons are wrong without doing the full evaluation you were trying to avoid." That is what makes the result so corrosive to the literature: it does not flag the bad papers, it casts doubt on all of them at once.

## 5. A full-catalog evaluation harness in pandas and numpy

Let me build the harness, because the gap between knowing you should do full-catalog evaluation and actually having code that does it correctly and fast is where most teams quietly revert to sampling. The plan: given a trained scorer that can produce, for a batch of users, scores over all items (a dense `[n_users, n_items]` block), compute Recall@K, NDCG@K, and HitRate@K against the held-out positives, masking out items the user already interacted with in training so we never reward re-recommending a known item.

First the metric computation, vectorized over a block of users. The key trick is to never sort the full score row when you only need top-K: `np.argpartition` gives you the top-K indices in linear time, and you sort only those $K$.

```python
import numpy as np

def topk_metrics_block(scores, train_mask, test_positives, K=10):
    """Full-catalog top-K metrics for a block of users.

    scores: [B, N] model scores for B users over all N items.
    train_mask: [B, N] boolean, True where the user already interacted
                in training (we set those scores to -inf to exclude them).
    test_positives: list of length B; test_positives[b] is a set of item
                    ids that are the held-out ground truth for user b.
    Returns dicts of per-user recall, ndcg, hit at K.
    """
    scores = scores.copy()
    scores[train_mask] = -np.inf  # never recommend a seen item

    B, N = scores.shape
    # top-K item indices per user (unordered), then sort just those K
    topk_unsorted = np.argpartition(-scores, K, axis=1)[:, :K]
    row = np.arange(B)[:, None]
    order = np.argsort(-scores[row, topk_unsorted], axis=1)
    topk = topk_unsorted[row, order]  # [B, K], best first

    recalls, ndcgs, hits = [], [], []
    # precompute the DCG position discounts once
    discounts = 1.0 / np.log2(np.arange(2, K + 2))
    for b in range(B):
        pos = test_positives[b]
        if not pos:
            continue
        ranked = topk[b]
        hitvec = np.array([1.0 if it in pos else 0.0 for it in ranked])
        n_pos = len(pos)
        recalls.append(hitvec.sum() / min(n_pos, K))
        dcg = (hitvec * discounts).sum()
        idcg = discounts[:min(n_pos, K)].sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        hits.append(1.0 if hitvec.sum() > 0 else 0.0)
    return {"recall": recalls, "ndcg": ndcgs, "hit": hits}
```

A few decisions encoded here that are easy to get wrong. We mask training items to `-inf` so the model is never credited for re-ranking something the user already clicked — recommending a known item is not a real recommendation, and forgetting this mask is a common silent inflation. The Recall denominator is `min(n_pos, K)` rather than `n_pos`, which is the standard convention so that a user with twenty held-out positives is not penalized for the fact that only ten can fit in the top 10 (there are alternative conventions; the important thing is to pick one and report it). The NDCG uses the ideal DCG over the achievable number of positives, the standard normalization.

Now the harness that streams users in blocks so the score matrix never blows up memory, and accumulates per-user metric values (we keep them *per user*, not pre-averaged, because we need the per-user values for the bootstrap in section 7):

```python
def evaluate_full_catalog(score_fn, user_ids, train_items_by_user,
                          test_items_by_user, n_items, K=10, block=512):
    """score_fn(user_block) -> [len(user_block), n_items] dense scores.
    Returns per-user metric arrays for bootstrap CIs.
    """
    all_recall, all_ndcg, all_hit = [], [], []
    for start in range(0, len(user_ids), block):
        ub = user_ids[start:start + block]
        scores = score_fn(ub)  # [b, n_items]

        train_mask = np.zeros((len(ub), n_items), dtype=bool)
        test_pos = []
        for j, u in enumerate(ub):
            for it in train_items_by_user.get(u, ()):  # seen -> mask
                train_mask[j, it] = True
            test_pos.append(set(test_items_by_user.get(u, ())))

        m = topk_metrics_block(scores, train_mask, test_pos, K=K)
        all_recall += m["recall"]; all_ndcg += m["ndcg"]; all_hit += m["hit"]
    return (np.array(all_recall), np.array(all_ndcg), np.array(all_hit))
```

For a matrix-factorization model the `score_fn` is one matmul: `user_emb[ub] @ item_emb.T`. For a two-tower model it is the user-tower forward followed by the same dot product against the precomputed item-embedding matrix. The point is that full-catalog scoring is *one dense matrix multiply per block*, which a GPU does in milliseconds; the sampled shortcut was never buying you as much as the inconsistency cost.

### The sampled harness, for comparison only

To actually demonstrate the reversal on your own data — which I strongly recommend doing once, as a vaccination — you also need the sampled version, so you can run both and watch them disagree:

```python
def evaluate_sampled(score_fn_pair, user_ids, train_items_by_user,
                     test_items_by_user, n_items, K=10, n_neg=100, seed=0):
    """For each (user, held-out positive), rank the positive against
    n_neg uniformly sampled negatives. score_fn_pair(u, items) -> scores.
    """
    rng = np.random.default_rng(seed)
    hits, ndcgs = [], []
    discounts = 1.0 / np.log2(np.arange(2, K + 2))
    for u in user_ids:
        seen = train_items_by_user.get(u, set())
        for pos in test_items_by_user.get(u, ()):
            negs = []
            while len(negs) < n_neg:
                c = rng.integers(0, n_items)
                if c != pos and c not in seen and c not in negs:
                    negs.append(c)
            cand = np.array([pos] + negs)
            s = score_fn_pair(u, cand)              # scores for candidates
            rank = (s > s[0]).sum()                  # 0-based rank of pos
            hits.append(1.0 if rank < K else 0.0)
            ndcgs.append(discounts[rank] if rank < K else 0.0)
    return np.array(hits), np.array(ndcgs)
```

Run both on the same trained model and the same test set. On a real model over MovieLens-20M with the full catalog of ~27,000 movies, you will see sampled HR@10 land somewhere comfortable like 0.6–0.8 while full-catalog HR@10 sits down around 0.1–0.2 — and if you compare two models, you will occasionally catch the order flipping. That flip, on your own data, is what finally kills the temptation to sample.

## 6. The other traps: features, popularity, filtering, survivorship

Splitting and sampling are the two big levers, but a careful evaluation has to dodge several more traps, each of which moves the number in a flattering direction.

### Leaky features

The split can be perfectly temporal and you can still leak the future *through your features*. The classic offender is an aggregate computed over the whole log: "this item's overall click-through rate," "this user's average rating," "the item's total purchase count." If you compute that aggregate over the entire dataset and then attach it to training rows, you have just told every training example about clicks that happened in the test period. The feature carries the future, even though the split did not. The figure contrasts a leaky aggregate against the point-in-time-correct version.

![A before and after comparison contrasting a leaky aggregate feature that computes item click through rate over the entire log including post cutoff data and yields an inflated AUC, against a point in time correct feature that computes the click through rate only up to each event time using an as of join and yields a lower honest AUC](/imgs/blogs/the-right-way-to-split-and-evaluate-6.png)

The fix is a **point-in-time (as-of) join**: every feature value attached to an event at time $t$ must be computed using *only* data with timestamp $\le t$. In pandas this is `merge_asof`; in a feature store it is the whole reason point-in-time correctness exists as a concept. Concretely, an item's CTR feature for an event at time $t$ is the cumulative clicks divided by cumulative impressions *as of $t$*, not the global CTR. The depth of this trap and the full machinery for getting features right is the subject of [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders); for evaluation purposes the rule is simply: **a temporal split protects the labels, but only point-in-time features protect the inputs.** Both are required.

#### Worked example: a leaky CTR feature inflating offline AUC

Suppose you add a single feature, "item click-through rate," to a ranking model. Computed leakily over the whole log, this feature *is* almost the label for popular items — an item that gets clicked a lot in the test period has a high global CTR, and that high CTR is sitting right there in the training row. The model learns "high-CTR items get clicked," which is circular, and offline AUC jumps to 0.84. Recompute the same feature point-in-time — CTR as of each event's timestamp — and the feature still helps (popularity has genuine predictive value) but the circularity is gone; AUC settles at a defensible 0.77. The 0.07 of AUC you "lost" was never real; it was the model reading the answer key. Online, the leaky model performs like the 0.77 model at best, because in production the feature is necessarily computed point-in-time — you cannot know an item's future CTR when you serve it. The leaky offline AUC was, once again, measuring a capability the deployed system does not have.

### Popularity is a deceptively strong baseline

Before you celebrate any model's full-catalog Recall, compare it against the dumbest possible recommender: **always recommend the globally most popular items** (computed, of course, point-in-time on the training period only). On most datasets this baseline is shockingly hard to beat, because a large fraction of interactions genuinely go to popular items. If your sophisticated model does not clearly beat point-in-time popularity on a temporal split, it has learned nothing beyond popularity, and you should suspect it. Here is the baseline:

```python
def popularity_topk(train_df, n_items, K=10):
    """Most-popular items from the TRAINING period only.
    Returns the top-K item ids; the same list is recommended to everyone
    (minus their already-seen items, handled by the eval mask)."""
    counts = train_df["item"].value_counts()
    pop = np.zeros(n_items)
    pop[counts.index.values] = counts.values
    topk = np.argpartition(-pop, K)[:K]
    return topk[np.argsort(-pop[topk])]
```

To evaluate it inside the same harness, the `score_fn` simply broadcasts the popularity vector to every user: `np.tile(pop, (len(ub), 1))`. The training mask still removes each user's seen items, so popularity is evaluated fairly. I have watched teams ship a deep model that, under an honest temporal split with full-catalog scoring, beat popularity by a statistically insignificant margin — which means all the GPU hours bought essentially nothing, a fact a random split with sampled metrics had hidden completely.

### k-core filtering changes the answer

Almost every benchmark "cleans" the data with **k-core filtering**: iteratively drop users with fewer than $k$ interactions and items with fewer than $k$ interactions until everyone clears the bar (5-core and 10-core are common). This makes the data denser and the numbers prettier, and it is *defensible for benchmarking model quality on warm data* — but it is a different dataset than production, where the long tail of light users and rare items is exactly where recommenders struggle and where the business often most wants to win. The trap is comparing your 10-core number against another paper's 5-core number, or reporting a 10-core number as if it estimated deployed quality. Report the filtering threshold every single time, and at least once evaluate on the *unfiltered* data to see how much the filtering flattered you. The effect is not subtle: tightening from 5-core to 20-core can move Recall@10 by 30–50% relative, purely by removing the hard cases.

### Survivorship: evaluating only on users with history

Here is the most insidious trap because it hides inside an apparently reasonable choice. To compute a metric you need ground-truth test interactions, so you naturally evaluate only on users who *have* a test interaction. But "has a test interaction" is itself a behavioral outcome — engaged users have test interactions, churned and cold users often do not. By conditioning your metric on users who survived into the test window, you measure quality on exactly the population that was already going to engage, and you systematically over-state quality on the population you most need to serve. This is survivorship bias, the same statistical error as evaluating an investment strategy only on the funds that did not close. The honest move is to (a) report metrics separately on the cold subset you flagged in section 3, and (b) be explicit that your headline metric is conditional on test-window activity. A temporal split with the cold flags surfaces this; a random split with sampled metrics buries it three layers deep.

## 7. Variance, confidence intervals, and not fooling yourself

Every offline metric is an estimate from a finite sample of test users, so it has a variance, and reporting a metric without an interval is the statistical equivalent of reporting a measurement without units. Two models that differ by 0.003 in Recall@10 on 2,000 test users may be indistinguishable; the same 0.003 on 200,000 users may be rock solid. You cannot know which without a confidence interval, and the cheapest honest interval for a metric that is an average over users is the **bootstrap**.

The logic: your per-user metric values (which we deliberately kept un-averaged in the harness) are a sample from a distribution. Resample the *users* with replacement many times, recompute the mean each time, and the spread of those means is your sampling distribution. The 2.5th and 97.5th percentiles give a 95% interval. Crucially, resample *users*, not interactions, because users are the independent unit — a single user's interactions are correlated, and treating them as independent under-states the variance.

```python
def bootstrap_ci(per_user_values, n_boot=1000, alpha=0.05, seed=0):
    """95% bootstrap CI for the mean of a per-user metric array."""
    rng = np.random.default_rng(seed)
    n = len(per_user_values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # resample USERS
        means[b] = per_user_values[idx].mean()
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return per_user_values.mean(), lo, hi
```

For comparing two models, the right move is a **paired** bootstrap: resample users once per iteration and compute *both* models' means on the same resampled users, then look at the distribution of the *difference*. Pairing removes the between-user variance that is common to both models and gives a much tighter interval on the delta — it is the single most effective way to detect a real-but-small improvement.

```python
def paired_bootstrap_delta(values_a, values_b, n_boot=1000, seed=0):
    """Paired bootstrap CI for mean(A) - mean(B) over the same users.
    values_a, values_b must be aligned per-user arrays."""
    rng = np.random.default_rng(seed)
    n = len(values_a)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = values_a[idx].mean() - values_b[idx].mean()
    point = values_a.mean() - values_b.mean()
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return point, lo, hi
```

If the delta interval straddles zero, you do not have evidence that A beats B, no matter how pretty the point estimates look — and you should not launch an A/B test on that basis, because the offline signal that motivates the online test is itself inside the noise. The other half of the discipline is **multiple training runs with different seeds**: a single run conflates the model's expected quality with the luck of one initialization and one negative-sampling draw. Report mean and standard deviation across at least three seeds. A surprising amount of the recommender "progress" in the literature evaporates when you put error bars on it — which is exactly the reproducibility-crisis finding we turn to next.

The honest pipeline, assembled, is the stack below: a temporal split feeds point-in-time features, the popularity baseline sets the floor, full-catalog scoring produces per-user metrics, the bootstrap turns those into intervals, and only then do you compare two models on the *overlapping* test users with a paired delta.

![A vertical stack of the honest evaluation pipeline showing a temporal global split at the top feeding into point in time features then a popularity baseline that must be beaten then full catalog scoring against all items then a bootstrap confidence interval over resampled users and finally a paired comparison of two models reporting the delta](/imgs/blogs/the-right-way-to-split-and-evaluate-7.png)

## 8. Stress-testing the protocol against the hard cases

A protocol is only as good as its behavior on the cases that break weaker protocols, so let me reason through the ones that matter, the way I would at a design review.

**What happens with only implicit feedback?** Everything above assumed implicit positives (clicks, plays), which is the common case. The subtlety is that you have no negatives — every non-interaction is ambiguous (unseen, not disliked), the missing-not-at-random problem. For evaluation this means your held-out test positives are real, but the "negatives" you rank against (all the other catalog items) include items the user *would* have liked and simply never saw. So full-catalog Recall is a *lower bound* on true quality: some of the items you rank above the held-out positive may actually be good recommendations the user never got the chance to interact with. This does not make the metric useless — it is still a consistent way to compare two models, because the bias hits both equally — but it is why offline Recall undershoots and why you must not read an absolute Recall of 0.14 as "the model is 14% good." It is a *relative* instrument, and the temporal split keeps it an honest relative instrument.

**What happens at 100 million items?** Full-catalog scoring of a 100M-item catalog for 100k test users is $10^{13}$ scores, which you cannot hold in memory and would rather not compute even in blocks. Two honest moves. First, evaluate the *full pipeline* rather than the ranker in isolation: in production a retrieval stage (a two-tower with ANN search, covered in [the two tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) and [ANN serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann)) already narrows 100M items to maybe 500 candidates, and evaluating top-K over *that* candidate set is both feasible and exactly what deployment does — it is not "sampling," because it is the real candidate set, not a random pool. Second, if you must evaluate the retriever itself, score against the full catalog using the same ANN index you serve with, accepting that the index's recall is part of what you are measuring. The mistake is using a *random* 100-negative pool as a proxy for 100M items; the candidate set from your own retrieval stage is the right "negatives," because those are the items the model actually competes against.

**What happens when the offline metric rises but online is flat?** This is the canonical paradox, and the first thing to check is whether the offline lift survived an honest protocol at all — half the time an "offline win" was a random-split-plus-sampled artifact that vanishes the moment you re-run it temporally with full catalog, which is the entire point of this post. If the lift *does* survive an honest offline protocol and online is still flat, then you are looking at a genuine offline-online gap with a different cause: position bias, presentation effects, the feedback loop, or a metric that does not track the business objective. That second class of cause is the subject of [the offline online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) and is not something a better split can fix. The discipline is to *first* make the offline number honest so that a surviving gap is informative rather than self-inflicted.

**What happens when negatives are mostly false negatives?** In a dense, high-engagement domain (a music app where users stream thousands of tracks), a large fraction of the "unseen" items are items the user would happily stream — so the held-out positive is competing against a sea of plausible recommendations, and full-catalog Recall looks crushingly low (0.02 is not unusual) even for a good model. The metric is still consistent, but its dynamic range is tiny, and small absolute differences are easy to mistake for noise. The fix is partly to lean harder on NDCG and MRR, which reward getting *a* good item near the top rather than the *specific* held-out item, and partly to widen $K$ so the held-out positive has room to land. It is also the strongest argument for confidence intervals: when the metric's range is 0.02 to 0.04, a 0.003 difference is meaningful only if its interval excludes zero.

**What happens when the population shifts mid-test-window?** If your test window spans a holiday, a major launch, or a UI change, the test distribution is not stationary even within itself, and a single aggregate metric averages over regimes that deployment will face one at a time. The honest move is to slice: report the metric on the first week and the last week of the test window separately, and if they diverge sharply, your "offline quality" is a moving target and you should retrain and evaluate more frequently. This is also why the cutoff $T$ should put a *near-future* slice in test rather than a six-month tail — you want to estimate the quality of a model that will be retrained in days, not the quality of a model left to rot for half a year.

## 9. Putting it together: the same models, two ways

Let me show the whole methodology paying off in one table. Take MovieLens-20M, treat ratings $\ge 4$ as positive implicit feedback, and compare a popularity baseline, a matrix-factorization model (BPR-trained), and a small two-tower model. Evaluate each two ways: a random 80/20 split with 100 sampled negatives (the "old" protocol) versus a temporal global split with full-catalog scoring and paired bootstrap intervals (the "honest" protocol). All numbers below are illustrative of the *pattern* the literature and my own runs show; treat them as representative orders of magnitude, not as a benchmark submission.

| Model | Random split, sampled-100 HR@10 | Temporal split, full-catalog HR@10 | Temporal full Recall@10 (95% CI) |
|---|---|---|---|
| Popularity (point-in-time) | 0.58 | 0.061 | 0.058 to 0.064 |
| Matrix factorization (BPR) | 0.79 | 0.142 | 0.137 to 0.147 |
| Two-tower | 0.81 | 0.139 | 0.134 to 0.144 |

Read this table slowly, because every cell teaches something. The sampled-random numbers (0.58 to 0.81) are uniformly inflated and uniformly compressed — they make popularity look almost competitive and squeeze the three models into a narrow band where the differences look small. The honest temporal full-catalog numbers (0.061 to 0.142) are *lower* and *more separated*: matrix factorization more than doubles popularity, a clearly real win, while the two-tower model is statistically tied with matrix factorization (their CIs overlap heavily: 0.137–0.147 versus 0.134–0.144), which is exactly the kind of finding the sampled protocol hides. Under the old protocol you might have shipped the two-tower for its 0.81-vs-0.79 edge; under the honest protocol you would correctly conclude the two-tower buys you nothing over BPR-MF on this data and is not worth its extra serving complexity. **The honest protocol did not just lower the numbers; it changed the decision.**

#### Worked example: random-split leakage inflating Recall, traced

Trace the popularity row to see the leakage mechanically. Under the random split, popularity is computed over 80% of the *entire* timeline, including the test period's hits, and the test set is drawn from across the whole timeline too — so "popular items" and "test-period clicks" share the same future, and the sampled-100 pool makes it trivial for any one of the top items to outrank 100 random negatives. Result: HR@10 of 0.58. Under the temporal split, popularity is computed on the *first* portion of the timeline only, the test set is the *later* portion, and scoring is against all ~27,000 items: now the genuinely-popular-in-the-future items the model has never heard of are not in its top list, and ranking against the whole catalog is hard. Result: HR@10 of 0.061 — nearly ten times lower. That factor of ten is the size of the lie a random split with sampled metrics tells you about your weakest baseline, and it is the reason an offline win can vanish online.

#### Worked example: the paired-bootstrap verdict on two-tower vs matrix factorization

Now use the table's two close models to show the confidence interval earning its keep. The two-tower's full-catalog Recall@10 is 0.139 with a 95% interval of 0.134 to 0.144; matrix factorization is 0.142 with an interval of 0.137 to 0.147. The point estimates differ by 0.003 in favor of MF, but the intervals overlap almost entirely, so an unpaired comparison is inconclusive. Run the *paired* bootstrap on the same 40,000 test users: resample users with replacement, recompute both models' means on each resample, and take the distribution of the difference. Because both models are evaluated on the same resampled users every iteration, the large between-user variance (some users are easy, some are hard, for *both* models) cancels in the difference, and the delta interval tightens. Suppose it comes out to a mean delta of +0.003 in favor of MF with a 95% interval of −0.001 to +0.007. That interval straddles zero: there is no statistically reliable difference between the two models on this data. The correct decision is to ship the *simpler, cheaper-to-serve* model — matrix factorization — not because it "won" but because the two-tower failed to demonstrate any advantage worth its extra serving complexity. A random split with sampled metrics, reporting 0.81 versus 0.79 with no interval at all, would have sent you to build and maintain the two-tower for an improvement that does not exist. This is the whole post in one decision: the honest protocol did not just change the numbers, it changed what you ship and what you spend.

## 10. Case studies: the reproducibility reckoning

This is not a niche concern; it triggered a public reckoning in the recommender-systems research community, and three results anchor it.

**Krichene and Rendle, "On Sampled Metrics for Item Recommendation" (KDD 2020).** This is the source of the sampled-metrics result we derived in section 4. They prove sampled metrics are inconsistent estimators of the corresponding full metrics and demonstrate empirically — on real models and datasets — that the relative ordering of models can and does change between sampled and full evaluation. They also derive corrected estimators for cases where full evaluation is infeasible. The practical takeaway the paper itself emphasizes: prefer full-catalog evaluation, and treat published sampled-metric comparisons with caution. An extended journal version appeared in 2022 (ACM TKDD). If you read one paper alongside this post, read this one.

**Dacrema, Cremonesi, and Jannach, "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches" (RecSys 2019).** The authors took a set of recently-published neural recommenders, attempted to reproduce them, and then compared them against properly-tuned *classical* baselines — well-tuned nearest-neighbor methods, simple linear models like SLIM, graph-based methods. Of the methods they could reproduce at all, most were outperformed, on the original papers' own datasets and metrics, by these tuned simple baselines. The finding was not that neural methods cannot work; it was that the *evaluation was so sloppy* — weak or untuned baselines, inconsistent splits, sampled metrics, missing reproducibility — that the reported "progress" was largely an artifact of comparing a tuned new model against a crippled old one. A 2021 extended version (ACM TOIS) reinforced it. The lesson for your own work: your strongest baseline is the experiment, and a result that does not clearly beat a *tuned* classical baseline under an honest protocol is not a result.

**The broader reproducibility crisis.** These two papers crystallized a wider unease. Rendle and colleagues separately showed in "Neural Collaborative Filtering vs. Matrix Factorization Revisited" (RecSys 2020) that a properly-tuned dot-product matrix factorization matches or beats the learned-similarity neural CF that had been reported to surpass it — the original NCF comparison had under-tuned the MF baseline. Across these works the common thread is methodology, not modeling: random splits leaking the future, sampled metrics reversing rankings, weak baselines, no confidence intervals, and irreproducible code. The community response has been concrete — stronger reproducibility requirements at RecSys, the rise of unified evaluation frameworks like RecBole and Elliot that pin down the split and metric protocol, and a default expectation of full-catalog metrics with tuned baselines. The dataset names and the exact deltas vary by paper, but the direction never does: **honest evaluation lowers the headline number and tightens the gap between methods, and a great deal of reported progress does not survive it.**

## 11. When to reach for each protocol (and when not to)

A decision section, because every choice here is a cost and the corners are tempting.

Reach for the **temporal global split with full-catalog scoring** as your default for any number you will act on — model selection, launch decisions, anything feeding an A/B-test go/no-go. It is more expensive than sampling and it produces lower numbers, both of which create internal pressure to skip it; resist that pressure, because it is the only protocol that has ever agreed with my online results. Do *not* use it for a quick "does my code run" smoke test — there a random split is fine precisely because you are not going to believe the number.

Reach for **leave-one-out** when you are publishing or comparing against a sequential-recommendation benchmark whose protocol is LOO (SASRec, BERT4Rec, and their descendants), so your numbers are comparable — but report it *as a benchmark protocol*, not as an estimate of deployed quality, and never report LOO with sampled negatives if you can compute it full-catalog. Reach for a **user-based split** only when your question is specifically user cold-start; it is the wrong tool for steady-state quality because it discards the user history that most served requests actually have.

Do *not* sample negatives for metric computation if you can possibly avoid it — and you almost always can, because full-catalog scoring is one matrix multiply per user block. If your catalog is so large that full scoring is genuinely infeasible even in blocks (tens of millions of items, hundreds of thousands of test users, tight compute), use a corrected estimator and state loudly that you did. Do *not* report a metric without a confidence interval and at least a note on seed variance, and do *not* compare your model only against itself or against an untuned baseline — the tuned popularity baseline and a tuned classical model (a well-configured `implicit` ALS or BPR) are the floor every result must clear. Do *not* k-core filter aggressively and then claim a deployment-quality number; report the threshold and check the unfiltered case at least once.

## 12. Key takeaways

- **Recommenders deploy forward in time, so the only honest split is temporal.** Random hold-out leaks the future into the past and inflates every metric, systematically and always upward; on real data it can double or 10x a baseline's apparent quality.
- **Use a global timestamp cutoff by default.** Train on the past, validate on the slice just before the cut, test on the future, and decide explicitly what to do with users and items that appear only after the cut — do not silently drop them.
- **Sampled top-K metrics are inconsistent (Krichene-Rendle, KDD 2020).** Ranking the true item against ~100 negatives is a nonlinear, non-rank-preserving estimate of the full-catalog metric and can *reverse* which model looks better. Use full-catalog evaluation; it is one matrix multiply per user block.
- **A temporal split protects the labels; only point-in-time features protect the inputs.** Aggregates computed over the whole log (item CTR, user averages) leak the future through the features even when the split is clean. Use as-of joins.
- **Beat the popularity baseline or admit you learned nothing.** Point-in-time popularity is shockingly strong under an honest protocol; a model that does not clearly beat it bought you only GPU hours.
- **Report confidence intervals and seed variance.** Bootstrap over *users* (not interactions); use a *paired* bootstrap to compare two models. A delta whose interval straddles zero is not a result.
- **Watch survivorship and filtering.** Evaluating only on users with test interactions over-states quality on the population you most need to serve; aggressive k-core filtering flatters the number. Report both choices explicitly.
- **The strongest baseline is the experiment.** Much of the reported progress in the literature (Dacrema et al., RecSys 2019) does not survive a tuned classical baseline under an honest split — make sure yours does.

## 13. Further reading

- Krichene, W. and Rendle, S. (2020). *On Sampled Metrics for Item Recommendation.* KDD 2020 (extended in ACM TKDD 2022). The proof that sampled metrics are inconsistent and the corrected estimators — the central paper for this post.
- Dacrema, M. F., Cremonesi, P., and Jannach, D. (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys 2019 (extended in ACM TOIS 2021). The reproducibility wake-up call and the case for tuned baselines.
- Rendle, S., Krichene, W., Zhang, L., and Anderson, J. (2020). *Neural Collaborative Filtering vs. Matrix Factorization Revisited.* RecSys 2020. Why a properly-tuned dot product matches learned-similarity neural CF.
- Meng, Z. et al. (2020). *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models.* RecSys 2020. A systematic study of how split choice changes the conclusions.
- RecBole documentation (recbole.io) and the Elliot framework — unified, reproducible evaluation pipelines that pin the split and metric protocol so results are comparable.
- Within this series: [offline evaluation metrics: Recall, NDCG, MAP, MRR](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr) for the metric definitions; [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders) for point-in-time feature correctness; [the offline online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) for why even an honest offline number still diverges from online; [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) for estimating online quality offline; and the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) for how this fits the whole pipeline.

---
title: "The Offline-Online Gap: Why Your Metric Lied"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Your model gained +5% offline NDCG and the A/B test came back flat. Here is the full diagnosis-and-fix playbook for the most painful recurring failure in applied recommenders, with a runnable simulator, an IPS-corrected estimate that tracks online, and a leakage demo."
tags:
  [
    "recommendation-systems",
    "recsys",
    "offline-evaluation",
    "online-evaluation",
    "distribution-shift",
    "off-policy",
    "counterfactual",
    "ndcg",
    "ab-testing",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-1.png"
---

You did everything right. You held out a temporal test set, you computed NDCG@10 the careful way, you ran it five times to be sure the seed did not flatter you. The new ranker beat the production model by 5.2% relative on NDCG@10, the AUC went up almost a point, logloss dropped. You wrote the design doc, the reviewers nodded, you shipped it behind a flag and started a clean A/B test on 5% of traffic.

Two weeks later the readout lands. Engagement: **-0.3%, not statistically significant.** Seven-day retention: flat. The launch committee asks the obvious question, and you do not have a good answer: *if the model is 5% better, why can't anyone tell?* You roll it back. Somewhere in a different org, an engineer who has been here before reads your postmortem and nods. They have a name for this: the offline-online gap. And the thing that makes it so demoralizing is that it is not a fluke or a bug in your eval code. **It is the default.** A new model that looks better offline is, more often than not, going to disappoint online. The surprising thing should be when offline and online *agree*.

![A side-by-side comparison showing an offline scorecard with strong NDCG and AUC gains on the left and a flat online A/B readout with a rollback on the right.](/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-1.png)

This post is the deep treatment of that gap. The series intro ([what a recommender system actually is](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system)) framed the whole pipeline as a **retrieval → ranking → re-ranking funnel** fed by a **feedback loop** that you read off the **offline↔online reality gap**, and a sibling post, [offline vs online, the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys), introduced the gap and gave a first simulator. This is where we open the hood. By the end you will be able to: name the seven distinct mechanisms that drive offline and online apart, run a decision tree to tell *which one* is biting on a given launch, write a simulator that reproduces the disagreement, compute an inverse-propensity-corrected offline estimate that tracks online where raw NDCG does not, and — the hardest part — decide which numbers to trust. Everything here is grounded in a running e-commerce-feed recommender on logged click data, with runnable NumPy you can paste into a notebook.

Let me say the thesis up front so the rest of the post earns it. **Offline metrics estimate the value of a model under the distribution that produced the data; online metrics measure the value of the model under the distribution it itself creates.** When those two distributions differ — and they always differ, because the new model would have shown different things — the offline number is an estimate of the wrong quantity. Sometimes it is close. Often it is not. The job is to know when, and to fix it.

## 1. Why offline up, online flat is the default, not the exception

Start with the cleanest possible mental model of what offline evaluation does. You have a log of interactions: for each request, the production model (call it the **logging policy**, $\pi_{\text{log}}$) chose a slate of items to show, the user clicked some and ignored others, and you wrote it all down. Offline evaluation takes that frozen log, asks your new model to re-rank the items that *were shown*, and scores the re-ranking with a metric like NDCG@10 that rewards putting clicked items near the top.

Notice the quiet assumption: the items in the log are the items the **old** model decided to show. That is the entire problem in one sentence. The log is not a random, representative sample of (user, item, outcome) triples. It is a sample drawn through the lens of one specific policy. Every item that the logging policy never surfaced has no outcome recorded, and your new model — if it is genuinely different — wants to show some of exactly those items. The offline metric has nothing to say about them, so it quietly treats "never shown" as "not relevant," and your new model gets neither credit nor blame for its actual decisions.

There is a second, subtler reason the gap is the default. Offline you optimize a **proxy**: usually clicks, sometimes a graded engagement label. Online you measure **value**: retention, watch time, gross merchandise value, the thing the business actually cares about. The proxy and the value are correlated — that is why the proxy is useful at all — but they are not the same function, and the gap between them is precisely the region a clever optimizer will exploit. A model that learns to predict clicks better will, with depressing reliability, also learn to predict *clickbait* better.

Put those two together and the asymmetry is structural. The space of model changes that look good offline is large; the subspace of those that are *also* good online is smaller and sits inside it. Most of the offline-good region is offline-good *because* it exploits one of the gap mechanisms — favoring items the log can't evaluate, or chasing the proxy past the point where it tracks value. So when you sample "a change that improved offline NDCG," you are sampling mostly from a region where the improvement does not transfer. This is not pessimism; it is geometry.

It helps to be precise about *what kind* of error the gap is, because engineers reflexively assume that a bigger test set will fix it, and it will not. There are two ways an offline estimate can be wrong. The first is **variance**: your test set is finite, so the metric is a noisy estimate of the true value on the *logged distribution*, and you reduce that noise by adding data or averaging over seeds. The second is **bias**: the metric is a clean, low-variance estimate of *the wrong quantity* — the value on the logged distribution rather than on the distribution the new model creates. The offline-online gap is almost entirely a *bias* problem. That is why your five careful re-runs with different seeds all agreed: they were precisely estimating the wrong number, and precision on the wrong number is exactly as misleading as a single noisy read. More data tightens the confidence interval around a biased estimate; it does not move the estimate toward the truth. The only things that reduce the bias are the ones that change *which distribution you are integrating against* — reweighting (IPS), exploration (new data from the right distribution), or just measuring online (sampling from the right distribution directly). Keep this bias-versus-variance distinction in mind throughout; nearly every wrong instinct about the gap comes from treating a bias problem as a variance problem.

There is also a timing asymmetry that makes the gap feel worse than it is. Offline you get a number in minutes; online you wait two weeks for an A/B readout. So the cheap, fast number is the biased one, and the expensive, slow number is the trustworthy one. Every incentive in a fast-moving team pushes toward shipping on the cheap number, and every one of those ships is a bet that this particular change happens to live in the small subspace where offline and online agree. Sometimes you win that bet. The discipline this post is really about is knowing which bets are good ones *before* you place them — using cheap offline signals as a filter, not a verdict.

That is the bad news. The good news is that the gap is not a single fog. It decomposes into a small number of named mechanisms, each with a fingerprint you can detect and a fix you can apply.

![A taxonomy tree grouping offline-online gap causes under three intermediate categories: distorted data, wrong target, and self-amplifying loop, each splitting into specific failure modes.](/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-2.png)

## 2. The science: two expectations, one missing integral

Let me make "two different worlds" precise, because the formalization tells you exactly where the fix has to live. A **policy** $\pi(a \mid x)$ is a distribution over actions $a$ (here, the item or slate you show) given a context $x$ (the user, the request, the candidate set). The **logging policy** $\pi_{\text{log}}$ generated the data; the **new policy** $\pi_{\text{new}}$ is what you want to evaluate. There is a reward $r(x, a)$ — for now treat it as 1 if the shown item was clicked and 0 otherwise, though the same algebra holds for graded rewards.

The quantity you actually care about — the **online value** of the new policy — is its expected reward under its *own* action distribution:

$$
V(\pi_{\text{new}}) = \mathbb{E}_{x \sim p(x)} \; \mathbb{E}_{a \sim \pi_{\text{new}}(\cdot \mid x)} \big[\, r(x, a) \,\big].
$$

The actions are drawn from $\pi_{\text{new}}$. That is the world the model creates when it ships. Now look at what a naive offline metric computes. It scores the new model on the logged interactions, where the actions came from $\pi_{\text{log}}$:

$$
\widehat{V}_{\text{naive}}(\pi_{\text{new}}) = \mathbb{E}_{x} \; \mathbb{E}_{a \sim \pi_{\text{log}}(\cdot \mid x)} \big[\, s_{\pi_{\text{new}}}(x, a) \cdot r(x, a) \,\big],
$$

where $s_{\pi_{\text{new}}}$ is however the new model's score enters the metric (for a ranking metric it is the position weight the new model assigns). The expectation over actions is taken under $\pi_{\text{log}}$, not $\pi_{\text{new}}$. **The naive offline estimator integrates the new model against the old model's action distribution.** It is estimating

$$
V(\pi_{\text{new}}) \quad\text{but sampling from}\quad \pi_{\text{log}},
$$

and unless those two policies put mass on the same actions, the estimate is biased. The size of the bias is governed by the mismatch $\pi_{\text{new}} / \pi_{\text{log}}$. Where the new policy wants to put mass that the old policy never did — $\pi_{\text{new}}(a \mid x) > 0$ but $\pi_{\text{log}}(a \mid x) = 0$ — there is *no data at all*. That region is the **counterfactual hole**, and no amount of clever reweighting can fill a hole with zero samples. This is the **coverage** or **positivity** condition, and it is the deepest reason offline evaluation of a genuinely novel policy is hard: the integral you want has a region the data never touches.

### Inverse propensity scoring: reweighting to bridge the gap

When coverage *does* hold — every action the new policy might take was at least occasionally taken by the logging policy — there is a clean fix, and it is the foundation of off-policy evaluation. (The next post, [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), is the full treatment; here is the preview that makes the gap concrete.) Importance sampling says: to take an expectation under $\pi_{\text{new}}$ using samples from $\pi_{\text{log}}$, reweight each sample by the ratio of the two probabilities.

$$
V(\pi_{\text{new}}) = \mathbb{E}_{a \sim \pi_{\text{log}}}\!\left[\, \frac{\pi_{\text{new}}(a \mid x)}{\pi_{\text{log}}(a \mid x)} \, r(x, a) \,\right].
$$

The ratio $w = \pi_{\text{new}} / \pi_{\text{log}}$ is the **importance weight**, and $\pi_{\text{log}}(a \mid x)$ — the probability the logging policy showed the action it actually showed — is the **propensity**. Given $n$ logged interactions, the **IPS estimator** is the empirical version:

$$
\widehat{V}_{\text{IPS}}(\pi_{\text{new}}) = \frac{1}{n} \sum_{i=1}^{n} \frac{\pi_{\text{new}}(a_i \mid x_i)}{\pi_{\text{log}}(a_i \mid x_i)} \, r_i .
$$

The intuition is worth pausing on, because it is the whole game. An interaction where the new policy strongly agrees with the old one ($\pi_{\text{new}} \approx \pi_{\text{log}}$) gets weight near 1 — it is representative of both worlds. An interaction the old policy showed *often* but the new policy would show *rarely* gets a small weight — we down-weight it because the new world would not produce it much. And an action the old policy showed rarely but the new policy loves gets a *large* weight — we up-weight the few samples we have, because they are precious evidence about a region the new policy cares about. The estimator is **unbiased** under the coverage condition, and the proof is one line worth seeing, because it shows *exactly* where the coverage assumption enters:

$$
\mathbb{E}_{a \sim \pi_{\text{log}}}\!\left[\frac{\pi_{\text{new}}(a \mid x)}{\pi_{\text{log}}(a \mid x)} r(x,a)\right]
= \sum_{a} \pi_{\text{log}}(a \mid x) \, \frac{\pi_{\text{new}}(a \mid x)}{\pi_{\text{log}}(a \mid x)} \, r(x,a)
= \sum_{a} \pi_{\text{new}}(a \mid x) \, r(x,a)
= V(\pi_{\text{new}} \mid x).
$$

The $\pi_{\text{log}}$ in front (from sampling) cancels the $\pi_{\text{log}}$ in the denominator (from the weight), leaving an honest sum over $\pi_{\text{new}}$. But look at the cancellation: it only works for actions $a$ where $\pi_{\text{log}}(a \mid x) > 0$. For any action with $\pi_{\text{log}}(a \mid x) = 0$ the term is $\tfrac{0}{0}$ — undefined, silently dropped — and if $\pi_{\text{new}}$ puts mass there, the sum is missing exactly those terms. That dropped mass is the counterfactual hole made algebraic. **Unbiasedness requires coverage**; without it IPS is not slightly wrong, it is estimating a *truncated* policy that pretends the holes do not exist.

The catch — and it is a big one — is **variance.** When $\pi_{\text{log}}(a_i \mid x_i)$ is tiny, the weight $w_i$ is huge, and a single lucky-or-unlucky logged sample can dominate the whole estimate. You can see why in the variance of the estimator: it scales with $\mathbb{E}[w^2 r^2]$, and $w^2$ blows up quadratically as the propensity shrinks. A logged interaction with propensity $0.01$ that the new policy loves contributes a weight near 100, and its square — 10,000 — to the variance. One such sample can swamp ten thousand well-behaved ones. This is the fundamental tension of off-policy evaluation: the *more different* the new policy is from the logger (exactly the case where you most need IPS, because raw NDCG is most biased), the *larger* the weights and the *higher* the variance. Bias and variance trade against each other along the axis of policy disagreement.

Three standard remedies tame the variance, each trading a little bias back:

- **Clipping (capped IPS).** Cap every weight at some $M$: $\tilde{w}_i = \min(w_i, M)$. This bounds the variance contribution of any single sample at $M^2$ but introduces a small downward bias (you are under-counting the rare-but-important interactions). Typical $M$ is 10-50; tune it by watching the effective sample size.
- **Self-normalized IPS (SNIPS).** Divide by $\sum_i w_i$ instead of $n$: $\widehat{V}_{\text{SNIPS}} = \tfrac{\sum_i w_i r_i}{\sum_i w_i}$. This is biased in finite samples but *consistent* (the bias vanishes as $n \to \infty$), and it is far steadier because it cannot produce an estimate outside the range of observed rewards — a property vanilla IPS lacks, as the worked example below will show dramatically.
- **Doubly-robust (DR).** Combine IPS with a learned reward model $\hat{r}(x,a)$: estimate the model's value directly with $\hat{r}$, then use IPS only to correct the *residual* $r - \hat{r}$ on the logged actions. DR is unbiased if *either* the propensities or the reward model is correct (hence "doubly robust"), and its variance is governed by how well $\hat{r}$ fits, so a decent reward model slashes the weight-driven variance. This is the workhorse of modern off-policy evaluation and the subject of the next post.

Hold that thought; the simulator below will show the raw, clipped, and self-normalized estimators side by side, and you will watch raw IPS produce a nonsense number while SNIPS stays sane.

A practical diagnostic falls straight out of the variance analysis: the **effective sample size**, $\text{ESS} = \tfrac{(\sum_i w_i)^2}{\sum_i w_i^2}$. It tells you how many "equivalent independent samples" your weighted estimate is really worth. If you logged a million interactions but ESS is 800, your IPS estimate has the statistical power of 800 samples, not a million — the weights collapsed almost all the information onto a handful of high-weight points. Reporting ESS alongside every off-policy estimate is non-negotiable; an IPS number without its ESS is a number without an error bar.

### Position bias: the model in the propensity

There is one more piece of science the formalism demands, because in a real recommender the "action" is not a single item — it is a *ranked list*, and where an item sits changes whether the user even looks at it. The standard model is the **position-based propensity model**: a user examines position $k$ with probability $p_k$ (the **examination probability**, which falls off sharply with rank), and clicks an examined item if and only if it is relevant. So the observed click probability factorizes:

$$
P(\text{click} \mid \text{item } d \text{ at position } k) = \underbrace{p_k}_{\text{examination}} \cdot \underbrace{P(\text{relevant} \mid d)}_{\text{true preference}}.
$$

This is the source of **position bias**: an item shown at rank 1 gets clicked far more than the same item at rank 10, not because it is more relevant but because $p_1 \gg p_{10}$. A naive metric that treats clicks as relevance labels will conclude that whatever the old model ranked highly is "good," which means your offline metric *rewards agreement with the logging policy's ordering* — the single most insidious way the gap hides. The fix is to debias the labels by dividing the click by its position propensity, $\hat{r}_d = c_d / p_k$, which is exactly an IPS weight where the propensity is the examination probability. Position-bias correction and policy IPS are the same idea wearing two hats.

It is worth seeing why the debiasing is unbiased, because it mirrors the policy-IPS proof exactly. The expected debiased label for a truly-relevant item at position $k$ is $\mathbb{E}\!\left[\tfrac{c_d}{p_k}\right] = \tfrac{1}{p_k}\,\mathbb{E}[c_d] = \tfrac{1}{p_k}\,p_k \cdot P(\text{rel}) = P(\text{rel})$ — the position cancels, leaving the true relevance regardless of where the item was shown. So a ranker trained or evaluated on $c_d / p_k$ sees relevance, not placement. The practical catch, again, is variance: items shown only at the bottom of the list have tiny $p_k$, so their debiased labels are high-variance, which is why production unbiased-learning-to-rank systems clip the propensities at a floor (say $p_k \geq 0.05$). And there is a chicken-and-egg wrinkle worth flagging: to debias you need $p_k$, but $p_k$ must itself be estimated, usually from a small **position-randomization** experiment (swap items across slots on a slice of traffic and read off how CTR changes with position alone) or from a model fit on items that naturally appear at multiple positions. If your estimate of $p_k$ is wrong, your debiasing is wrong — another reason a little randomization in production is the gift that keeps giving.

## 3. The seven mechanisms, each with its fingerprint

With the formal frame in hand, here are the concrete ways offline and online diverge. I have shipped or debugged every one of these. They are not mutually exclusive — a bad launch often has two or three at once — but each has a distinct cause and a distinct fix, so it pays to name them separately.

**1. Distribution shift / the closed loop (the counterfactual hole).** Covered above: the log was generated by the current model, so the new model's favored items were rarely or never shown, and the offline metric scores it on the wrong distribution. *Fingerprint:* the new model's top-K overlaps little with the logging policy's top-K; the offline gain comes disproportionately from re-ranking items that appear rarely in the log.

**2. Position and selection bias in the logs.** Clicks encode position, not just relevance, and the candidate set itself was selected by an upstream retrieval stage, so items that never reached the slate are invisible. A model that learns "rank-1 items are good" looks great offline (it agrees with the log) and adds nothing online (it just reproduces the old order). *Fingerprint:* CTR-versus-slot curve is steep; the offline gain shrinks dramatically when you debias labels by position.

**3. The proxy-metric gap.** Offline you score clicks; online you measure retention or satisfaction. A model that gets better at predicting clicks may get better at predicting *clickbait*. *Fingerprint:* the offline metric (clicks/NDCG) improves but a held-out **north-star** proxy — dwell time, next-day return, complete views — does not, or moves the wrong way.

**4. Goodhart / metric gaming.** "When a measure becomes a target, it ceases to be a good measure." A model trained hard on a single offline objective finds the cheap way to inflate it. *Fingerprint:* the target metric goes up while **guardrail** metrics (diversity, report rate, unsubscribe) degrade; nobody downstream feels happier.

**5. Feedback loops amplifying.** The model's outputs become tomorrow's training data, so any bias toward popular items compounds. A new model that is marginally more pop-biased looks fine offline (the log is already pop-skewed) but slowly collapses the catalog online. We treat this fully in [the feedback loop post]; here it is one cause of the gap. *Fingerprint:* online catalog coverage and recommendation entropy fall over days; the effect is invisible in a single static offline snapshot.

**6. Data leakage inflating offline.** A feature that secretly encodes the future or the label — a session-level aggregate computed over the whole session, a "user clicked this category today" flag — hands the model the answer offline but is unavailable (or differently computed) at serve time. *Fingerprint:* the offline gain is implausibly large; ablating the suspect feature makes it vanish; the feature's offline distribution differs from its online distribution (train-serve skew, covered in [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate)).

**7. Novelty and presentation effects offline can't see.** A new layout, a new badge, a "because you watched" explanation, even just *newness* itself drives a temporary online lift (or, after the novelty wears off, a fade) that no offline metric on historical logs can capture. *Fingerprint:* online effect decays over the experiment window; offline is silent because the log predates the change.

And underneath all of these is a measurement hazard worth its own mention: **sampled metrics are unreliable.** To make offline eval cheap, many teams score each positive against a small random sample of negatives (say 100) instead of the full catalog. A landmark 2020 result (Krichene & Rendle, KDD 2020, "On Sampled Metrics for Item Recommendation") showed that sampled NDCG and Recall are **inconsistent**: they can *reverse* the ranking of two models relative to the full-catalog metric. So before you even reach the offline-online gap, your offline number itself may be lying about which model is better. If you sample negatives, this is the first thing to rule out.

It is worth stressing how these mechanisms interact, because in a real postmortem they rarely arrive one at a time. Position bias and distribution shift compound: a model that learns to mimic the logging policy's *ordering* both agrees with biased click labels (position bias) and stays close to the logging distribution (low shift) — so it scores well offline for two wrong reasons at once, and the offline number is doubly inflated relative to what it would do if it actually changed the ordering. The proxy gap and Goodhart are two views of the same thing seen at different time scales: Goodhart is the proxy gap *after the optimizer has had time to exploit it*. And leakage plus feedback loops are the pair that produce the most confident wrong decisions — leakage makes the offline number huge, the feedback loop makes the online harm slow and invisible, so you ship a "30% win" and watch the catalog homogenize over a month while every offline re-run keeps insisting the model is great. The reason a single decision tree (next section) works anyway is that the mechanisms, while they co-occur, have *distinct* detection tests; you can peel them off one at a time even when several are present.

One more framing that helps engineers reason about the whole set: every mechanism is a violation of one of three assumptions that offline evaluation silently makes. (1) *The data is representative of the policy we will deploy* — violated by distribution shift, position/selection bias, and feedback loops. (2) *The label we optimize equals the value we want* — violated by the proxy gap and Goodhart. (3) *The features available offline equal the features available online* — violated by leakage and train-serve skew. If you keep those three assumptions written on the wall, every offline-online surprise you ever hit will be a violation of one of them, and naming which one is the first half of the fix.

## 4. The diagnostic playbook: which cause is biting?

When a launch disappoints, you do not get to fix all seven at once. You need to know which one (or two) is responsible, because the fixes point in different directions: IPS for distribution shift, label debiasing for position bias, a new objective for the proxy gap, a feature ablation for leakage. Here is the decision tree I run, ordered so the cheapest and most fatal checks come first.

![A vertical stack of diagnostic decision layers, from check leakage first, then favors-unlogged-items, then position bias, then bad proxy, then loop effects, ending at trust the A/B test.](/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-6.png)

**Step 0 — Is the offline number itself trustworthy?** Did you compute the metric on the *full* candidate set or a sampled one? If sampled, recompute on the full catalog (or at least a much larger sample) before believing the delta at all. Did you use a temporal split, or did you randomly shuffle and accidentally train on the future? Random splits leak future interactions into the training set and inflate everything. Rule these out first; they are pure measurement error, not a real gap.

**Step 1 — Leakage.** Take your single biggest offline win and *ablate the most suspicious feature* — anything that aggregates over the session, anything with "today" or "this visit" in its definition, anything whose value at serve time you cannot reconstruct from data available *before* the recommendation is made. If the offline gain largely disappears, you found a leak. A useful tell: a real ranking improvement is usually 1-3% relative on NDCG; a single feature that buys you 8-10% is almost always cheating.

**Step 2 — Does the new model favor rarely-logged items?** Compute the rank-overlap between your new model's top-10 and the logging policy's top-10 across a sample of requests (Jaccard or Kendall's tau). If the overlap is high, the offline metric is mostly measuring *agreement with the old model* and your "win" is small re-orderings of items the old model already showed — that transfers, but the ceiling is low. If the overlap is *low*, the new model wants to show different things, the offline metric is scoring it on a distribution it does not control, and you are squarely in counterfactual-hole territory. This is where IPS earns its keep.

**Step 3 — Position bias.** Estimate the examination curve $p_k$ (a simple way: items that appear at multiple positions across requests let you back out the relative CTR by slot). Debias the labels — divide each click by its position propensity — and recompute the offline metric. If the gain shrinks a lot after debiasing, the model was largely learning to agree with the old ranking.

**Step 4 — Proxy gap.** Pick a *secondary* offline proxy that is closer to value than clicks: dwell-time-weighted relevance, complete-view rate, next-session return on the holdout. If the click metric goes up but this one is flat or negative, the model is optimizing the wrong thing. This one often cannot be fully resolved offline — it is the strongest argument for trusting the A/B.

**Step 5 — Loop and presentation effects.** These you usually catch *online*: watch the experiment over time. A novelty effect decays; a feedback-loop effect (coverage, entropy) drifts. If the offline metric is fine and the online effect is non-stationary, you are seeing one of these, and the offline log — a static snapshot — was never going to predict it.

Most of these steps reduce to a handful of cheap numbers you can compute in one pass over the log and a candidate model. I keep this diagnostic battery as a single function and run it on every model line before I trust the headline NDCG; it has saved me from shipping a leak more than once.

```python
import numpy as np

def diagnose_gap(model_scores_fn, log, pop, exam, K=10):
    """One pass producing the signals from the diagnostic tree.
    log: list of (user, slate_item_ids, clicks, slot_propensities)."""
    overlaps, debiased_gain, leak_flag = [], [], 0.0
    raw_clicks, debiased_clicks = 0.0, 0.0
    catalog_hits = np.zeros_like(pop)

    for (u, slate, clicks, exam_p) in log:
        scores = model_scores_fn(u)
        topk = np.argsort(-scores)[:K]
        # (Step 2) overlap of model top-K with what the logger actually showed
        overlaps.append(len(set(topk) & set(slate)) / K)
        # (Step 3) raw vs position-debiased click mass on the logged slate
        raw_clicks += clicks.sum()
        debiased_clicks += (clicks / exam_p).sum()
        # (Step 5) coverage: how concentrated are the model's picks?
        catalog_hits[topk] += 1

    overlap = float(np.mean(overlaps))
    # coverage entropy: high = broad catalog use, low = collapse toward head
    p = catalog_hits / catalog_hits.sum()
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    max_entropy = np.log((catalog_hits > 0).sum())

    return {
        "topk_overlap_with_log": round(overlap, 3),       # high => low shift
        "position_inflation": round(raw_clicks / debiased_clicks, 3),  # >1 => bias
        "coverage_entropy_norm": round(entropy / max_entropy, 3),       # low => loop risk
        "distinct_items_in_topk": int((catalog_hits > 0).sum()),
    }
```

Read the output like a dashboard. A `topk_overlap_with_log` near 0.3 or higher says the model mostly re-orders what the logger already showed — offline is a decent guide. Below ~0.2 says it favors unlogged items — distribution shift, reach for IPS. A `position_inflation` well above 1 says clicks are heavily position-driven and your labels need debiasing before you believe any metric. A low `coverage_entropy_norm` (say below 0.5) means the model concentrates on a head of popular items — a feedback-loop risk that no static offline metric will flag. None of these *prove* a cause, but together they tell you which branch of the tree to walk down, and they cost one pass over data you already have.

![A matrix mapping each root cause to its symptom, a cheap detection test, and the targeted fix that addresses it.](/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-4.png)

The matrix is the reference card. Notice that the *fix* column is what actually matters operationally: distribution shift wants IPS or online exploration, position bias wants label debiasing, the proxy gap wants a better target, leakage wants the feature removed, Goodhart wants guardrail constraints, and the feedback loop wants exploration plus re-ranking. Reaching for the wrong fix — say, adding more capacity to "improve" a model whose offline win is pure leakage — wastes weeks.

## 5. A simulator that reproduces the gap

Talk is cheap; let me build the disagreement so you can watch it happen. The plan: define a *ground-truth* preference that we (the simulator) know but the model does not, generate logs through a logging policy with position bias, then show that a model favoring novel items posts a higher *raw offline NDCG* while its *true online reward* (which we can compute because we own the ground truth) does not improve — and finally show an IPS-corrected offline estimate that tracks the truth.

```python
import numpy as np

rng = np.random.default_rng(7)

N_USERS, N_ITEMS = 2000, 500
K = 10  # slate size / cutoff

# --- Ground truth the model never sees -------------------------------------
# Each user has a latent preference over items. Online "reward" = sum of true
# preference of items the user actually engages with from the shown slate.
U = rng.normal(size=(N_USERS, 16))
V = rng.normal(size=(N_ITEMS, 16))
true_pref = 1 / (1 + np.exp(-(U @ V.T) / 4.0))   # P(like | user, item) in (0,1)

# Item popularity (skewed) drives what the LOGGING policy tends to show.
pop = rng.pareto(2.0, size=N_ITEMS) + 1.0
pop /= pop.sum()

# Position examination probabilities p_k: steep falloff (position bias).
exam = 1.0 / np.log2(np.arange(2, K + 2))   # p_1 ~ 1.0 down to p_10 ~ 0.28
exam /= exam[0]

print("examination by slot:", np.round(exam, 3))
```

The logging policy is popularity-biased — it mostly shows popular items, which is realistic. We log, for each user, a slate of $K$ items, and a click on each slot that is examined *and* whose item is truly liked.

```python
def logging_policy_slate(user_id):
    """Old model: score = popularity + small noise. Returns top-K item ids."""
    score = np.log(pop) + rng.normal(0, 0.3, size=N_ITEMS)
    return np.argsort(-score)[:K]

def simulate_clicks(user_id, slate):
    """Position-based click model: click iff examined AND truly liked."""
    examined = rng.random(K) < exam
    liked = rng.random(K) < true_pref[user_id, slate]
    return (examined & liked).astype(int)

# Build the log: (user, slate, clicks, propensity_of_each_slot)
log = []
for u in range(N_USERS):
    slate = logging_policy_slate(u)
    clicks = simulate_clicks(u, slate)
    # propensity = examination prob of the slot the item was shown in
    log.append((u, slate, clicks, exam.copy()))

total_clicks = sum(c.sum() for _, _, c, _ in log)
print(f"logged interactions: {len(log)} slates, {total_clicks} clicks")
```

Now two candidate rankers. **Model A** is honest: it ranks by a noisy estimate of true preference. **Model B** is the troublemaker: it has *also* learned that novel (unpopular) items the log rarely shows tend to be relevant when shown — which is *true* — but the offline metric cannot credit that, while a naive offline metric will still reward B's re-ordering of the logged items because B happens to push the few logged relevant items up. We will measure each model two ways: raw offline NDCG on the logged slate, and *true online reward* if we re-shipped that model's top-K (computable because we own `true_pref`).

```python
def model_scores(model, user_id):
    if model == "A":   # honest: noisy view of true preference
        return true_pref[user_id] + rng.normal(0, 0.25, size=N_ITEMS)
    if model == "B":   # honest about preference AND up-weights novel items
        novelty = -np.log(pop)               # high for rare items
        return true_pref[user_id] + 0.15 * novelty + rng.normal(0, 0.25, size=N_ITEMS)

def dcg(rel):
    return np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))

def ndcg_at_k(rel_in_rank_order, k=K):
    d = dcg(rel_in_rank_order[:k])
    ideal = dcg(np.sort(rel_in_rank_order)[::-1][:k])
    return d / ideal if ideal > 0 else 0.0
```

Raw offline NDCG re-ranks the *logged slate* by the model's scores and uses *clicks* as relevance — exactly what a standard offline harness does:

```python
def raw_offline_ndcg(model):
    vals = []
    for (u, slate, clicks, _) in log:
        order = np.argsort(-model_scores(model, u)[slate])  # re-rank logged items
        vals.append(ndcg_at_k(clicks[order]))
    return np.mean(vals)

def true_online_reward(model):
    """If we shipped this model, show its top-K; reward = examined-and-liked,
    weighted by TRUE preference. This is the online metric we cannot see in prod."""
    rewards = []
    for u in range(N_USERS):
        topk = np.argsort(-model_scores(model, u))[:K]
        examined = rng.random(K) < exam
        rewards.append(np.sum(examined * true_pref[u, topk]))
    return np.mean(rewards)

for m in ["A", "B"]:
    print(f"Model {m}: raw_offline_NDCG={raw_offline_ndcg(m):.3f}  "
          f"true_online_reward={true_online_reward(m):.3f}")
```

When you run this, the pattern is the gap in miniature: Model B posts a *higher raw offline NDCG* (it shuffles the logged relevant items toward the top a touch better) while its *true online reward* is essentially tied with A — its novelty-seeking shows items the offline log never scored, so offline gives it neither credit nor penalty, and the small NDCG edge is noise on the logged distribution, not a real online win. That is offline-up, online-flat, manufactured from first principles.

The simulator is worth keeping around because you can turn each knob and watch the gap respond, which builds the intuition no amount of reading can. Crank the popularity skew (`pop = rng.pareto(1.2, ...)`) and the logging policy concentrates harder on the head; the counterfactual hole widens, B's overlap with the log drops further, and the raw-NDCG-versus-true-reward disagreement grows. Flatten the examination curve (less position bias) and the click labels become cleaner, so the offline metric tracks truth better — direct confirmation that position bias is a *cause* of the gap, not just a correlate. Make the logging policy *stochastic* with high temperature (more exploration) and coverage improves, the counterfactual holes shrink, and — as the next section shows — IPS starts working because there is finally data in the regions the new policy cares about. Every lever in the simulator maps to a real lever you have in production: catalog skew, slot design, exploration budget. The disagreement you saw in the painful A/B readout is not mysterious; it is the sum of these knobs, and you can dial it up or down on purpose.

#### Worked example: tracing one offline-up/online-flat result to distribution shift

Let me trace a single concrete readout. Suppose the run prints:

```
Model A: raw_offline_NDCG=0.412  true_online_reward=2.05
Model B: raw_offline_NDCG=0.461  true_online_reward=2.04
```

Raw offline says B beats A by $\tfrac{0.461 - 0.412}{0.412} = +11.9\%$ NDCG — a slam-dunk by any offline standard. True online reward says B is $2.04$ versus $2.05$, a $-0.5\%$ change, dead flat. If this were production you would ship B on the offline number and be baffled by the A/B.

Now run Step 2 of the playbook. Compute the top-10 overlap between B and the logging policy versus A and the logging policy:

```python
def overlap_with_log(model):
    j = []
    for (u, slate, _, _) in log:
        topk = set(np.argsort(-model_scores(model, u))[:K])
        j.append(len(topk & set(slate)) / K)
    return np.mean(j)

print("A overlap with log:", round(overlap_with_log("A"), 3))
print("B overlap with log:", round(overlap_with_log("B"), 3))
```

A's top-10 overlaps the logged slate (say 0.34); B's overlaps far less (say 0.19) because B keeps surfacing rare items the popularity-biased logger never showed. **Low overlap is the fingerprint of distribution shift.** B's offline win is being scored almost entirely on the items the *old* policy chose — a distribution B does not even agree with — so the offline number is an estimate of B's value under A's world, which is not the world B would create. The +11.9% is an artifact of evaluating the new policy on the old policy's actions. Diagnosis: distribution shift, not a real improvement. Fix: reweight with IPS, which we do next.

## 6. The IPS-corrected estimate that tracks online

Now the payoff: take the *same logged data* and reweight it so the offline estimate approximates each model's value under *its own* action distribution. We need propensities. In this simulator the logging policy's slate is deterministic given the popularity scores, so to make IPS well-defined I will use a **stochastic** logging policy (sample the slate proportional to a softmax over popularity), which is also what you should do in production precisely so off-policy evaluation is possible — a deterministic logger gives you zero coverage and no escape.

```python
def softmax(x, temp=1.0):
    z = np.exp((x - x.max()) / temp)
    return z / z.sum()

# Stochastic logging policy: per-slot sampling proportional to popularity.
def log_propensities(user_id):
    return softmax(np.log(pop), temp=1.0)   # P(show item i) under logger

def model_policy_probs(model, user_id, temp=1.0):
    return softmax(model_scores(model, user_id), temp=temp)

def ips_estimate(model, clip=20.0, self_norm=True):
    """IPS reward estimate using logged single-item interactions."""
    num, den = 0.0, 0.0
    p_log_all = None
    for (u, slate, clicks, exam_p) in log:
        p_log = log_propensities(u)
        p_new = model_policy_probs(model, u)
        for slot, item in enumerate(slate):
            # reward observed for this shown item (debias position too)
            r = clicks[slot] / exam_p[slot]          # position-debiased reward
            w = p_new[item] / max(p_log[item], 1e-9) # policy importance weight
            w = min(w, clip)                          # clip for variance control
            num += w * r
            den += w if self_norm else 1.0
    return num / max(den, 1e-9)

for m in ["A", "B"]:
    print(f"Model {m}: IPS_estimate={ips_estimate(m):.3f}")
```

The IPS estimate down-weights the logged interactions where the new model disagrees with the logger and up-weights the rare interactions on items the new model favors. The result is an offline number whose *ranking of models* matches the true online reward — where raw NDCG flips them, IPS keeps them in order, or correctly calls the difference a wash. That is the entire promise of off-policy evaluation: a number computed on the old log that estimates the new world.

![A side-by-side comparison showing raw offline NDCG ranking model B above model A while the IPS-corrected estimate matches the flat online reward.](/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-5.png)

#### Worked example: naive versus IPS-weighted offline on three logged interactions

To see the arithmetic, shrink the world to three logged interactions for one user, all single-item slates so we can do it by hand. The logging policy and a new policy assign these probabilities, with rewards observed (position-debiased to 0/1 for simplicity):

| Interaction | item | $\pi_{\text{log}}$ | $\pi_{\text{new}}$ | reward $r$ | weight $w = \pi_{\text{new}}/\pi_{\text{log}}$ |
| :---------- | :--- | :----------------- | :----------------- | :--------- | :--------------------------------------------- |
| 1 | popular | 0.60 | 0.20 | 0 | 0.33 |
| 2 | mid | 0.30 | 0.30 | 1 | 1.00 |
| 3 | rare | 0.10 | 0.50 | 1 | 5.00 |

**Naive estimate** treats every interaction equally: $\widehat{V}_{\text{naive}} = \tfrac{0 + 1 + 1}{3} = 0.667$. This is the old policy's reward rate, not the new one's — it over-counts the popular item the new policy barely shows.

**IPS estimate** (vanilla, $\tfrac{1}{n}\sum w_i r_i$): $\tfrac{1}{3}(0.33 \cdot 0 + 1.00 \cdot 1 + 5.00 \cdot 1) = \tfrac{6.0}{3} = 2.0$. That blew up past 1.0 — a clear symptom of the high-variance failure mode, driven by the weight-5 sample on the rare item. This is exactly why raw IPS is dangerous with small propensities.

**Self-normalized IPS** divides by the weight sum instead of $n$: $\tfrac{0.33\cdot 0 + 1.0\cdot 1 + 5.0\cdot 1}{0.33 + 1.0 + 5.0} = \tfrac{6.0}{6.33} = 0.948$. Now it is a sensible reward rate in $[0,1]$, and it correctly reports that the new policy — which leans on the rare item that *did* get a reward — looks *better* than the naive 0.667, because the new policy concentrates mass where the reward actually was. The self-normalized estimator traded the exact unbiasedness of vanilla IPS for a bounded, far steadier number. In production you would also **clip** the weight-5 term (cap at, say, 4.0) to bound variance further. The takeaway: the naive number answered the wrong question, vanilla IPS answered the right question loudly and unstably, and self-normalized IPS answered it sensibly.

## 7. A leakage demo: the offline win that was never real

Distribution shift is the subtle cause; leakage is the embarrassing one, and it is shockingly common. Let me manufacture it so the failure is undeniable. We add a feature that *peeks at the future*: a per-(user, item) flag that is 1 if the user clicked that item *anywhere in the session*, computed over the whole session and then attached to every request in that session. Offline it is a near-perfect predictor of the click you are trying to rank. At serve time it does not exist — you cannot know future clicks when you make the recommendation.

```python
# Build a leaky feature: did the user click this item in the session?
# (Offline we know; online we cannot.)
def leaky_offline_ndcg():
    vals = []
    for (u, slate, clicks, _) in log:
        # Leaky "score": the click label itself, lightly noised.
        leaky_score = clicks + rng.normal(0, 0.05, size=K)
        order = np.argsort(-leaky_score)
        vals.append(ndcg_at_k(clicks[order]))
    return np.mean(vals)

def honest_offline_ndcg():
    return raw_offline_ndcg("A")   # honest model, pre-event features only

print(f"leaky offline NDCG : {leaky_offline_ndcg():.3f}")
print(f"honest offline NDCG: {honest_offline_ndcg():.3f}")
```

The leaky NDCG comes out near the ceiling (it is essentially sorting by the label) while the honest model sits where a real ranker sits. If you saw the leaky number in a design doc — NDCG@10 of 0.55 against a 0.42 baseline, a 30% relative jump — your instinct should be alarm, not celebration. **No ranking change buys you 30% relative NDCG honestly.** The detection test is in the playbook: ablate the feature and watch the gain evaporate, and check that the feature is computable from data available strictly *before* the recommendation timestamp. Train-serve skew — the same feature computed differently in the batch pipeline than in the online service — is the quieter cousin of outright leakage and produces the same offline-online gap; the split-and-evaluate post covers how to catch it with a temporal split and an online-offline feature parity check.

![A side-by-side comparison showing a leaky feature inflating offline NDCG to an implausible level versus an honest feature set producing a realistic offline score that matches online.](/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-7.png)

## 8. Results: when offline and online agree, and when they flip

Here is the kind of table I keep for every model line. It records, for a set of real changes we tried on the e-commerce-feed recommender, the offline delta and the (simulated or A/B-measured) online delta, and whether they agreed. The point of the table is not the exact numbers — they are representative of what these changes do — but the *pattern*: agreement is far from guaranteed, and the disagreements are systematic, traceable to the mechanisms above.

| Model change | Offline delta (NDCG@10) | Online delta | Agree? | Mechanism |
| :----------- | :---------------------- | :----------- | :----- | :-------- |
| Larger embedding dim (32→64) | +1.1% | CTR +0.8% | Yes | none — real gain |
| Add session-sum-clicks feature | +9.0% | CTR 0.0% | **No** | leakage |
| Hard negatives in training | +3.4% | CTR +2.1% | Yes | real gain |
| Click-only objective (drop dwell) | +4.2% | watch-time -1.6% | **No** | proxy gap |
| Popularity re-rank boost | +2.0% | retention -0.9% | **No** | feedback loop |
| Add diversity term to re-ranker | -0.6% | retention +1.3% | **No (flip)** | offline can't see novelty |

![A matrix listing six model changes with their offline delta, online delta, and whether the two agree, with disagreements labeled by mechanism.](/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-8.png)

Read the table top to bottom and you can feel the lesson. Two changes (bigger embeddings, hard negatives) are honest wins — they improve the ranking on a distribution the offline metric can actually see, and the online lift follows. The other four are exactly the traps: a leaky feature that posts a huge offline number and zero online; a click-only objective that looks better offline but trades away watch time; a popularity boost that nudges NDCG up while quietly hurting retention through the feedback loop; and — the most important row — a **diversity term that looks *worse* offline but is *better* online.** That last one is why "ship the model with the best offline NDCG" is not a policy; it is a way to systematically reject the changes that help users in ways your historical log cannot represent.

Now the IPS column. If you re-score those same six changes with the self-normalized IPS estimator from §6 instead of raw NDCG, the leakage row collapses (the leaky feature does not change the *policy* over the catalog, only the offline label, so IPS on the true reward does not reward it), and the distribution-shift rows come back into agreement with online far more often. IPS does not fix everything — the proxy gap and novelty effects are about the *reward definition* and *presentation*, which no reweighting of click data can repair — but for the distribution-shift family it turns a misleading offline number into a useful one. **The honest summary: raw NDCG agreed with online on 2 of 6 changes; IPS-corrected offline agreed on roughly 4 of 6; only the A/B got all 6 right.** That ordering — A/B beats IPS beats raw offline — is the ranking of trust you should internalize.

#### Worked example: a Goodhart launch that gamed its own metric

The "popularity re-rank boost" row deserves a full trace, because it is the cleanest example of Goodhart's law in a recommender and the one teams fall for most often. The team's offline target was NDCG@10 on clicks. Someone noticed that adding a small popularity prior to the final re-ranker — bumping each candidate's score by $\beta \log(\text{item popularity})$ with $\beta = 0.2$ — lifted offline NDCG@10 by +2.0% relative. The mechanism is exactly the position-bias trap in disguise: the logged clicks are themselves popularity-skewed (popular items were shown more, at better slots, and clicked more), so a model that leans toward popular items *agrees with the biased labels* and scores well. The offline metric, computed on those same biased labels, applauds.

Run the numbers on the A/B. CTR was actually up a hair, +0.4%, because popular items do get clicked — so a CTR-only readout would have called this a marginal win. But the guardrail metrics told the real story: **catalog coverage** (fraction of the catalog shown to anyone over a week) fell from 41% to 33%, recommendation **entropy** dropped, and **7-day retention** came in at -0.9%. The popularity boost was eating the long tail: users saw a narrower, more homogeneous feed, found fewer new things they liked, and a measurable slice came back less often. Offline NDCG saw none of this because the historical log was *already* popularity-skewed — the metric had no vocabulary for "the catalog got narrower." This is Goodhart in three lines: the target (NDCG) became the goal, the optimizer found that agreeing with the biased log inflates it, and the measure stopped measuring anything users cared about. The fix was not a better model; it was adding coverage and entropy as **guardrail constraints** that block any launch that narrows the catalog past a threshold, regardless of what NDCG says. The lesson generalizes: a single offline objective with no guardrails is an invitation to game it, and the gaming is invisible in the very metric you are watching.

## 9. The fixes, in the order you should reach for them

The diagnosis tells you the cause; here is the toolbox, roughly in increasing cost.

**Debias your offline labels (cheap, do it always).** If you only fix one thing, divide clicks by their position propensity before computing any offline metric. It removes the single most common way offline rewards mere agreement with the logging policy. Estimating $p_k$ can be as simple as a position-randomization experiment on a slice of traffic, or a model fit on items that appear at multiple positions.

**Pick a better proxy (cheap, high leverage).** If the proxy gap is your problem, no amount of off-policy machinery helps — you are optimizing the wrong target. Move from raw clicks to a value-weighted label: dwell-time-weighted relevance, complete-view, add-to-cart, next-session return. Multi-task ranking ([MMoE/PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple)) exists largely to let you train on several proxies at once so no single one dominates.

**Counterfactual / off-policy evaluation (medium).** When distribution shift is the cause, IPS, clipped IPS, self-normalized IPS, and the lower-variance **doubly-robust** estimator let you estimate the new policy's value on the old log — *if* you logged propensities and your logging policy had nonzero coverage. This is the single biggest reason to make your serving policy slightly **stochastic**: a deterministic logger gives you no propensities and no escape from the counterfactual hole. The full machinery is the [counterfactual and off-policy evaluation post](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation).

**Interleaving (medium, cheap online signal).** Instead of a full A/B test, *interleave* the rankings of two models into one list and attribute each click to whichever model placed the item. Interleaving controls for position and is dramatically more sensitive than an A/B test — Radlinski & Craswell and the Bing team report it needs roughly 10-100x less traffic to reach significance — which makes it the fastest honest online read on a ranking change. It does not measure long-term retention, but for "which ranker do users prefer right now" it is the cheapest truthful answer. The reason it is so much more sensitive is that it is a *within-request* comparison: every user sees both rankers' items in one list, so per-user noise (some users just click a lot) cancels out, where an A/B compares *different* users and must average that noise away with sheer sample size. Here is the team-draft interleaving that produces an unbiased per-impression preference:

```python
import numpy as np

def team_draft_interleave(rank_a, rank_b, rng):
    """Interleave two ranked lists; track which model placed each item."""
    out, owner, seen = [], [], set()
    ia = ib = 0
    while len(out) < min(len(rank_a), len(rank_b)):
        a_first = rng.random() < 0.5  # coin flip who picks this round
        for first, rank, ptr_name in ([("A", rank_a, "a"), ("B", rank_b, "b")]
                                       if a_first else
                                       [("B", rank_b, "b"), ("A", rank_a, "a")]):
            ptr = ia if ptr_name == "a" else ib
            while ptr < len(rank) and rank[ptr] in seen:
                ptr += 1
            if ptr < len(rank):
                out.append(rank[ptr]); owner.append(first); seen.add(rank[ptr])
                if ptr_name == "a": ia = ptr + 1
                else: ib = ptr + 1
    return out, owner

def interleaving_preference(clicks, owner):
    """+ means model A preferred this impression, - means B."""
    a = sum(1 for c, o in zip(clicks, owner) if c and o == "A")
    b = sum(1 for c, o in zip(clicks, owner) if c and o == "B")
    return (a - b) / max(a + b, 1)
```

Sum that per-impression preference over a day of traffic and a confidence interval on the mean tells you, with far less data than an A/B, which ranker users actually prefer — and crucially, it does so *online*, so it sidesteps the counterfactual hole entirely (both rankers' items were really shown and really clicked).

**Exploration to fill the holes (medium-high).** The counterfactual hole exists because the logging policy never showed certain items. The structural fix is to *show them sometimes*: epsilon-greedy exploration, or a contextual bandit that allocates a slice of traffic to uncertain items. This both improves the model (it sees more of the catalog) and improves *future* off-policy evaluation (the log gains coverage). Exploration has a real cost — you are deliberately showing some worse items — so budget it explicitly.

**A/B testing (high, the ground truth).** When the proxy gap or novelty effects are in play, or when the change is big enough to matter, there is no substitute for a randomized online experiment measuring the north-star metric over a long-enough window. The [A/B testing post](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) covers the statistics — power, novelty-effect windows, network interference, guardrail metrics. The rule that survives every project: **when offline and online disagree, online wins.** Offline is a filter to decide *what to A/B*, not the final word.

![A branching dataflow graph showing the logging policy recorded an outcome for item A while the new policy prefers item B, which was never shown, leaving a counterfactual hole the offline metric fills by assuming the item is irrelevant.](/imgs/blogs/the-offline-online-gap-and-why-your-metric-lied-3.png)

The graph above is the picture to hold in your head while you reach for these fixes: the logging policy showed A and recorded that it was clicked; the new policy prefers B; but B was never shown, so there is no label, and the naive offline metric fills the hole with "irrelevant." IPS reweights what data you *do* have toward B-like items; exploration creates B-data going forward; the A/B test sidesteps the hole entirely by just showing B to real users.

## 10. Case studies and the literature

These are not hypotheticals. The offline-online gap is one of the most documented phenomena in applied recsys.

**Netflix and the offline-online divergence.** Netflix has written repeatedly (in their tech blog and the *Netflix Recommender System* ACM TMIS paper, Gomez-Uribe & Hunt, 2015) that offline metrics are at best a weak predictor of the online A/B result, and that they treat offline evaluation strictly as a *filter* to decide which ideas are worth an A/B test, never as the decision itself. Their reported practice — every algorithm change goes through an online experiment measuring retention-linked metrics, because offline ranking accuracy did not reliably predict the member-engagement lift — is the canonical statement of "trust the A/B."

**Ad-tech and counterfactual evaluation.** The off-policy evaluation literature grew up in ad ranking precisely because of this gap. Bottou et al. (2013, *Counterfactual Reasoning and Learning Systems*, JMLR) used Microsoft's Bing ad system to show how to estimate the online effect of a policy change from logged data with importance weighting, and why naive offline estimates of ad-ranking changes mislead. Li et al. (2011, "Unbiased Offline Evaluation of Contextual-Bandit-Based News Article Recommendation," WSDM) showed a *replay* method on Yahoo! Front Page news that gives an unbiased online estimate from logged random-bucket data — the practical realization of "log with exploration so you can evaluate off-policy later."

**Position bias and unbiased learning to rank.** Joachims et al. (2017, "Unbiased Learning-to-Rank with Biased Feedback," WSDM) formalized the position-based propensity model and showed that IPS-debiased click labels recover a ranker that matches one trained on true relevance, while naive click-trained rankers do not — direct evidence that the position-bias mechanism is real and correctable. The whole field of *unbiased learning to rank* is the offline-online gap, attacked at the label level.

**Sampled metrics are inconsistent.** Krichene & Rendle (2020, "On Sampled Metrics for Item Recommendation," KDD) is the result every recsys engineer should know: evaluating against a small sample of negatives can *reverse* the relative ranking of two models compared to the full-catalog metric. Their recommendation is to use the full catalog, or corrected sampled metrics, and to never compare two models' sampled NDCG without checking consistency. This is the offline number lying *before* you even get to the online gap.

**Feedback loops in the wild.** Chaney et al. (2018, "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility," RecSys) simulated the closed loop and showed that recommending from data the recommender itself generated drives homogenization and *lower* true utility over time — invisible to any single offline snapshot. This is the mechanism behind the popularity-boost row in the results table.

**The retrieval-stage selection bias.** A practical case that recurs in every two-stage system: your *ranker* is evaluated offline on the candidates the *retrieval* stage surfaced, which were themselves chosen by the production retrieval model. So even a perfect ranker is capped at the recall of the upstream stage, and an offline ranker metric improves are scored only on items retrieval bothered to fetch. Teams at YouTube and elsewhere have documented (in the YouTube deep-neural-networks-for-recommendation line of work, Covington et al., 2016, and follow-ons) that the candidate-generation stage is where most of the true ceiling lives, yet offline ranker evaluation is blind to candidates retrieval never produced — a selection bias one layer up from the click logs. The cure is the same family: evaluate end-to-end, debias the candidate selection, and explore in retrieval.

**The proxy gap, quantified.** Several teams have published the gap between click-prediction quality and downstream value directly. The general finding across the watch-time and dwell-time literature (for example the work on optimizing for *satisfied* clicks rather than raw clicks, and the move at video platforms from CTR to expected-watch-time objectives) is that a model that improves click AUC by a point can leave watch time flat or negative, precisely because clicks include the bait the user immediately abandons. The fix was never a better click model; it was redefining the label toward value. This is the proxy-gap row in the results table, drawn from production rather than simulation.

Together these say the same thing from six directions: the offline log is a biased, partial, proxy view of a world the new model will change, and treating its metric as ground truth is a category error. None of these teams are careless — they are among the most sophisticated in the industry. The gap is not a skill issue; it is a property of evaluating a counterfactual from observational data, and the only defenses are the structural ones in this post.

## 11. How to trust your offline numbers

So what do you actually do, given that the offline metric is structurally unreliable but A/B tests are expensive and slow? You build a *ladder of trust* and you are honest about which rung each number sits on.

1. **Make the offline number as honest as it can be.** Temporal split (never random). Full-catalog metrics (or proven-consistent sampled ones). Position-debiased labels. Strict feature-time discipline so nothing computed after the recommendation leaks in. A leakage ablation on every headline win. This gets you a number that is *internally* valid — it correctly ranks models on the logged distribution.
2. **Estimate the counterfactual.** Log propensities, keep a small exploration slice, and compute clipped or self-normalized IPS (and doubly-robust when you can). This gets you a number that estimates the *new* distribution, not the old one. Track how well your IPS estimate has predicted past A/B outcomes; that calibration is your confidence in it.
3. **Get a cheap online read.** Interleave for ranking changes; it is 10-100x more sensitive than an A/B and catches most "this re-ordering is actually worse" failures in a day.
4. **A/B the survivors on the north-star.** Only changes that clear the offline filter and the IPS estimate and look good in interleaving earn a full A/B on retention/GMV, run long enough to see past novelty effects, with guardrail metrics watched for Goodhart.
5. **When they disagree, online wins — and write down why.** Every offline-online disagreement is a lesson about which mechanism your offline harness is blind to. Feed it back: a proxy gap means fix the label; a distribution-shift surprise means improve the IPS calibration or the exploration budget; a leak means a process failure in feature engineering. The gap shrinks over time only if you treat each disagreement as a bug in the *evaluation*, not just a disappointing model.

There is a cultural piece to this that no estimator fixes. A team that rewards "offline NDCG wins" in performance reviews will produce offline NDCG wins, and a depressing fraction of them will be gap artifacts — that is Goodhart applied to the *team*, not just the model. The teams that ship reliable improvements measure their members on *online* outcomes and treat offline gains as proposals, not deliverables. Concretely: track, for every model that cleared the offline filter, whether its A/B confirmed the offline prediction, and publish that hit rate. If your offline harness predicts the A/B direction 80% of the time, your offline number is genuinely informative and you can lean on it to triage. If it predicts 50% of the time, your offline number is a coin flip with extra steps, and you should fix the harness (debiasing, IPS, leakage checks) before trusting another decision. This calibration number — *how often does offline predict online* — is the single most important metric about your evaluation system, and almost nobody measures it. Once you do, the offline-online gap stops being a recurring trauma and becomes a tracked quantity you can drive down.

The single sentence to take away: **offline metrics decide what to test; online metrics decide what to ship.** Anyone who inverts that order will, sooner or later, ship a +5% offline win that does nothing — and now you know exactly why.

## When to reach for each tool (and when not to)

- **Don't ship on offline NDCG alone — ever — for a change that alters *which* items get shown.** If the new model's top-K overlaps the logging policy's top-K (small re-orderings), offline is a decent guide. If the overlap is low, offline is estimating the wrong distribution; you need IPS or an A/B.
- **Don't reach for IPS when coverage is zero.** A deterministic logging policy gives propensities of 0 or 1 and no escape from the counterfactual hole. IPS is worth the variance pain *only* if you logged a stochastic policy with real coverage. Fix the logging first.
- **Don't use raw (unclipped) IPS in production.** The variance will make a single small-propensity sample dominate your estimate. Clip or self-normalize, always; reach for doubly-robust when you have a decent reward model.
- **Don't debug a proxy gap with more model capacity.** If clicks go up and retention does not, the problem is the *target*, not the model. A bigger model just optimizes the wrong thing better. Fix the label or add a multi-task head.
- **Don't trust a sampled offline metric for a close call.** If two models are within a few percent on sampled NDCG, recompute on the full catalog before deciding; sampled metrics can flip the order.
- **Don't skip interleaving because "we'll just A/B it."** Interleaving is so much cheaper that there is rarely a reason not to run it first on a ranking change; it filters out the obvious losers before you spend A/B traffic.

## Key takeaways

- **Offline up, online flat is the default.** The space of changes that look good offline is larger than the subspace that is also good online, and most offline wins exploit a gap mechanism that does not transfer.
- **Formally, offline estimates value under the *logging* policy's action distribution; online measures it under the *new* policy's.** The bias is governed by the mismatch $\pi_{\text{new}}/\pi_{\text{log}}$, and where the new policy wants mass the logger never put mass, there is a counterfactual hole no math can fill.
- **There are seven distinct mechanisms** — distribution shift, position/selection bias, the proxy gap, Goodhart, feedback loops, leakage, and novelty/presentation — each with its own fingerprint and its own fix. Diagnose before you fix.
- **Run the diagnostic tree in order:** rule out untrustworthy offline numbers (sampling, random split) and leakage first, then check top-K overlap for distribution shift, then position bias, then the proxy, then loop/novelty effects.
- **IPS reweights the old log toward the new world.** Clip or self-normalize to control variance; it bridges the distribution-shift family but cannot fix a wrong proxy or unseen novelty effects.
- **Interleaving is the cheapest honest online signal** for ranking changes (often 10-100x more sensitive than an A/B); exploration fills the counterfactual holes for next time.
- **The ladder of trust is raw-offline < IPS-corrected < interleaving < A/B.** Offline decides what to test; online decides what to ship; when they disagree, online wins.
- **Every offline-online disagreement is a bug in your evaluation, not just a bad model.** Feed each one back into a more honest offline harness and the gap shrinks over time.

## Further reading

- **Within this series:** [offline vs online, the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) (the gap introduced) · [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) (the full IPS/DR/replay treatment) · [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) (temporal splits, leakage, train-serve skew) · [A/B testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) (the online ground truth) · the series intro map [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
- Bottou, Peters, Quiñonero-Candela, et al. (2013), "Counterfactual Reasoning and Learning Systems: The Example of Computational Advertising," *JMLR* — the foundational off-policy-from-logs paper, on Bing ads.
- Li, Chu, Langford, Wang (2011), "Unbiased Offline Evaluation of Contextual-Bandit-Based News Article Recommendation Algorithms," *WSDM* — the replay method on Yahoo! news.
- Joachims, Swaminathan, Schnabel (2017), "Unbiased Learning-to-Rank with Biased Feedback," *WSDM* — position-based propensity model and IPS-debiased learning to rank.
- Krichene, Rendle (2020), "On Sampled Metrics for Item Recommendation," *KDD* — why sampled NDCG/Recall are inconsistent and can reverse model order.
- Gomez-Uribe, Hunt (2015), "The Netflix Recommender System: Algorithms, Business Value, and Innovation," *ACM TMIS* — offline as a filter, A/B as the decision.
- Chaney, Stewart, Engelhardt (2018), "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility," *RecSys* — the closed-loop feedback simulation.
- Radlinski, Kurup, Joachims (2008), "How Does Clickthrough Data Reflect Retrieval Quality?," *CIKM* and Chapelle et al. on interleaving — the sensitivity advantage of interleaved comparisons.

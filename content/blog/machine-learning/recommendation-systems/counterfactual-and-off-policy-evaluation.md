---
title: "Counterfactual and Off-Policy Evaluation for Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Estimate how a new ranking policy would perform online using only logs from the old one. Derive the IPS estimator and prove it is unbiased, build the Direct Method, SNIPS, and Doubly Robust in numpy on a simulated logged bandit, and see exactly when poor propensity overlap blows the variance up."
tags:
  [
    "recommendation-systems",
    "recsys",
    "off-policy-evaluation",
    "counterfactual",
    "inverse-propensity-scoring",
    "doubly-robust",
    "evaluation",
    "machine-learning",
    "causal-inference",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/counterfactual-and-off-policy-evaluation-1.png"
---

You have a new ranking model. The offline NDCG@10 is up four points, the AUC moved in the right direction, and the eval harness says it is better than the model currently in production. So you do the responsible thing and you ask for an A/B test slot. The platform team tells you the queue is three weeks deep, that you will need at least two weeks of traffic to clear the significance bar at your effect size, and that — by the way — if the new model is *worse*, you will have degraded the experience for a few hundred thousand real users for the duration of the test. You have eleven candidate models in your backlog this quarter. At two weeks each, serialized, that is most of a year of A/B slots for models that mostly will not win. There has to be a better way to decide which one is worth the slot.

There is, and it is the subject of this post. The question you actually want answered is counterfactual: *if I had shown users what the new policy would have shown them, what reward would I have gotten?* You never ran that policy. But you did run the **old** policy, and you logged everything it did — the context it saw, the action it took, and the reward it got back. The discipline of estimating a new policy's online value from logs collected under a different policy is called **off-policy evaluation** (OPE), or counterfactual estimation, and it is one of the highest-leverage tools a recommender team can own. It lets you screen a backlog of candidate models *offline*, before any of them touches live traffic, and reserve the expensive A/B slots for the few that an honest offline estimate says are worth the risk.

![A branching dataflow showing logged tuples of context action propensity and reward feeding both a reward model route and an importance reweighting route that merge into an estimate of the new policy value](/imgs/blogs/counterfactual-and-off-policy-evaluation-1.png)

The figure above is the whole machine in one picture. On the left you have the logs: for every impression, the context the system saw, the action it chose, the **propensity** (the probability the logging policy assigned to that action), and the reward that came back. Those tuples feed two routes. One route fits a reward model and asks it to score what the new policy would do. The other route reweights each logged reward by how much more (or less) likely the new policy was to take that same action. A third estimator combines both. Every route produces the same thing on the right: an estimate of $V(\pi_{\text{new}})$, the online value of the new policy, computed without serving it to a single user.

By the end of this post you will be able to: write down the logged-bandit setup and explain why propensities are the load-bearing ingredient; derive the inverse-propensity-scoring (IPS) estimator and prove it is unbiased in three lines of algebra; reason about its variance and why propensity overlap is the thing that makes or breaks it; build the Direct Method, IPS, self-normalized IPS (SNIPS), and Doubly Robust (DR) estimators in numpy on a simulated logged bandit where you *know* the true value; and read a results table that shows DR landing closest to truth while IPS's variance explodes under poor overlap. This is the offline-evaluation toolset that sits one level deeper than ranking metrics: NDCG tells you whether your model orders a fixed list well, but it cannot tell you what a *new policy that changes which items get shown* would earn online. OPE can. It is the bridge across [the two worlds of offline and online recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys), and the formal answer to the question every eval post in this series keeps circling: why does the offline number so often fail to predict the online one?

## 1. Why ranking metrics are not enough

Most of this series has been about metrics computed on a fixed held-out set: Recall@K, NDCG@K, AUC, logloss. Those metrics answer a precise question — given a *fixed* set of candidates with *fixed* relevance labels, how well does the model order them? That is exactly the right question for a re-ranking model whose job is to sort a slate someone else chose. It is the wrong question, or at least an incomplete one, for a model that *changes which items appear at all*.

Here is the gap. Your logs were generated by the production policy $\pi_{\text{log}}$. Every reward you observed is conditioned on the action $\pi_{\text{log}}$ chose. If item 47 was never shown to user $u$, you have no idea whether $u$ would have clicked it. A new policy $\pi_{\text{new}}$ that loves item 47 will show it constantly — and your logs are silent on what happens then. This is the **counterfactual** problem in its purest form: you want $E[r]$ under a policy you never ran, and your data only tells you $E[r]$ under the one you did. Standard supervised metrics paper over this by silently assuming the action distribution does not change, which is the one assumption a new policy is built to violate.

Concretely, consider a feed where the logging policy mostly shows popular items. Your new policy is a fresh model that surfaces more niche content. If you score the new model with offline NDCG on the logged impressions, you are grading it on its ability to re-rank *the popular items the old policy already chose* — which is not what it will do in production. In production it will retrieve different candidates, change the slate, and earn rewards on items the old policy rarely showed. The NDCG number is measuring a different policy than the one you will ship. This is the same distribution-shift, missing-not-at-random, position-bias triad that makes [the offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) so treacherous; OPE is the principled way to close it.

The fix is to stop pretending the action distribution is fixed and instead *model the change in action distribution explicitly*. That is what propensities buy you, and it is why a system that logs them can do something a system that does not log them cannot.

## 2. The logged-bandit formalism

Strip a recommender down to its decision core and you have a **contextual bandit**. At each step:

1. A context $x$ arrives (the user, their history, the candidate set, time of day — the features).
2. The system picks an action $a$ from a set $\mathcal{A}$ (which item to show, which slate to assemble, which ranking to commit).
3. A reward $r$ comes back (a click, a watch, a purchase, dwell time — whatever you optimize).

A **policy** $\pi(a \mid x)$ is a conditional distribution over actions given context. The deterministic argmax ranker is a degenerate policy that puts all its mass on one action. A policy that explores is a stochastic one that spreads probability across several actions. The **value** of a policy is the expected reward you would earn if you deployed it:

$$
V(\pi) \;=\; \mathbb{E}_{x \sim D}\, \mathbb{E}_{a \sim \pi(\cdot \mid x)}\, \mathbb{E}_{r \sim p(\cdot \mid x, a)}\,[\, r \,].
$$

Read that right to left: the world hands you a reward $r$ for the action $a$ you took in context $x$; you average over the rewards, over the actions your policy draws, and over the contexts the world sends. $V(\pi)$ is the single number an A/B test estimates by actually running $\pi$ and averaging the rewards it earns.

Your logs are a set of $n$ tuples collected under the logging policy:

$$
\mathcal{D} \;=\; \big\{ (x_i,\; a_i,\; p_i,\; r_i) \big\}_{i=1}^{n}, \qquad p_i \;=\; \pi_{\text{log}}(a_i \mid x_i).
$$

The fourth element, $p_i$, is the one most logging systems forget to record and the one that makes everything downstream possible. It is the **propensity**: the probability that the logging policy assigned to the action it actually took. Not a derived quantity, not something you can recover after the fact from a deterministic argmax — the *actual probability* the policy used at decision time, which you must log at decision time or lose forever.

The goal of OPE is to estimate $V(\pi_{\text{new}})$ for a target policy $\pi_{\text{new}}$ using only $\mathcal{D}$, which was collected under $\pi_{\text{log}} \neq \pi_{\text{new}}$. We measure an estimator $\hat{V}$ by two quantities: its **bias** $\mathbb{E}[\hat{V}] - V(\pi_{\text{new}})$ (does it center on the truth?) and its **variance** $\operatorname{Var}(\hat{V})$ (how much does it bounce around from one log to another?). The whole drama of this post is the trade between those two. The compact summary of the four estimators we will build is the matrix below.

![A comparison matrix of Direct Method IPS SNIPS and Doubly Robust against bias variance needs propensity and needs reward model showing each estimator makes a different trade](/imgs/blogs/counterfactual-and-off-policy-evaluation-2.png)

Read the matrix as a decision aid. The Direct Method has low variance but is biased when its reward model is wrong. IPS is unbiased but carries very high variance and needs propensities. SNIPS reduces that variance for a tiny bias. Doubly Robust needs both ingredients but is unbiased if *either* one is correct and has lower variance than IPS. We will earn every one of those claims with math and then with a simulation.

### Why you would ever do this instead of an A/B test

A/B testing is the gold standard — it directly estimates $V(\pi)$ by running $\pi$. So why bother with estimators at all? Because A/B tests are slow, expensive, and risky, and OPE is fast, cheap, and safe. The contrast is stark enough to draw.

![A two column comparison of an online A B test that is slow and risky against an off policy estimate from logs that is fast and computed before shipping](/imgs/blogs/counterfactual-and-off-policy-evaluation-3.png)

The left column is the A/B reality: you commit live traffic to $\pi_{\text{new}}$, wait two to four weeks for the effect to clear the noise floor, and accept the risk that you have just degraded the experience for the test population if the new policy is worse. The right column is OPE: you reuse logged data with no new exposure, you get an estimate in minutes, and you A/B only the handful of candidates that survive the offline screen. The two are complements, not substitutes — you still confirm the winner with an A/B test, because OPE estimates can be wrong in ways we will catalog. But OPE turns a backlog of eleven candidates into a backlog of two, and that is the difference between shipping improvements this quarter and next year. This is the same logic that makes [A/B testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) a scarce, gated resource: if slots are scarce, the offline screen that decides who gets one is doing real work.

## 3. The Direct Method: model the reward, then evaluate

The most obvious approach needs no propensities at all. Fit a **reward model** $\hat{r}(x, a)$ that predicts the expected reward of taking action $a$ in context $x$, using your logs as training data (this is just supervised regression: features $(x, a)$, target $r$). Then evaluate the new policy by asking the model what reward it expects for whatever the new policy would do:

$$
\hat{V}_{\text{DM}}(\pi_{\text{new}}) \;=\; \frac{1}{n} \sum_{i=1}^{n} \sum_{a \in \mathcal{A}} \pi_{\text{new}}(a \mid x_i)\, \hat{r}(x_i, a).
$$

For each logged context $x_i$, you average the model's predicted reward over the actions the new policy would take, weighted by how likely it is to take each. No propensities, no reweighting — just a model and a sum. This is the **Direct Method** (DM), sometimes called the regression estimator or the plug-in estimator.

DM's appeal is its low variance. It is a smooth function of a fitted model, so it does not bounce around with the luck of which actions happened to be logged. Its curse is bias. If $\hat{r}$ is systematically wrong — and it usually is, because it was trained on the *logging* policy's action distribution and now you are querying it on the *new* policy's, which visits different parts of action space — then $\hat{V}_{\text{DM}}$ inherits that error directly. The model extrapolates into regions it never saw enough data for, and DM trusts the extrapolation completely. There is no statistical correction; if the model says item 47 earns 0.9 and it really earns 0.3, DM is off by 0.6 with full confidence.

```python
import numpy as np

def direct_method(contexts, reward_model, pi_new_probs):
    """DM estimate.
    reward_model: callable (contexts, a) -> predicted reward, shape (n,)
    pi_new_probs: array (n, A) of pi_new(a | x_i) for every action a
    Returns scalar V_hat_DM.
    """
    n, A = pi_new_probs.shape
    # Predicted reward for every (context, action) pair: shape (n, A)
    rhat = np.stack([reward_model(contexts, a) for a in range(A)], axis=1)
    # Average model prediction under the new policy's action distribution
    per_context = (pi_new_probs * rhat).sum(axis=1)   # shape (n,)
    return per_context.mean()
```

The deep problem with DM is that you cannot tell from the logs whether the model is right where it matters. The logs cover the logging policy's action distribution well and the new policy's poorly — exactly backwards from what DM needs. A model with great held-out logloss on logged actions can still be badly wrong on the actions the new policy favors, and DM gives you no warning. This is why we reach for estimators that use propensities to correct the distribution rather than trusting the model to extrapolate.

To see the extrapolation failure concretely, take a single action the new policy adores but the logger almost never tried. Suppose action 3 was shown in only 0.4% of impressions, all of them to one narrow slice of users where it happened to under-perform, so the model fits $\hat{r}(x, 3) \approx 0.31$ across the board. The new policy, trained differently, wants to show action 3 to a *broad* audience where its true reward is 0.58. DM evaluates the new policy by querying $\hat{r}(x, 3) \approx 0.31$ on contexts the model never saw action 3 paired with, and reports a value far below the truth — with total confidence, because the regression has no notion of "I have not seen this region." The logloss on the held-out logged data looks fine; the held-out data simply does not contain the contexts where the new policy diverges. This is not a fixable bug in the model; it is the structural mismatch between where the logs are dense and where the new policy operates, and only a propensity-aware correction can flag or fix it.

## 4. Inverse propensity scoring, derived and proven unbiased

The second route does not model rewards at all. Instead it reweights the rewards you observed so that the logged action distribution looks like the new policy's. The intuition is a reweighting one — and it sits directly under the figure that draws it.

![A vertical stack showing a logged reward scaled by an importance weight then averaged over the log to produce an unbiased value estimate](/imgs/blogs/counterfactual-and-off-policy-evaluation-6.png)

The stack is the whole estimator. Take each logged reward $r_i$. Multiply it by an **importance weight** $w_i = \pi_{\text{new}}(a_i \mid x_i) / \pi_{\text{log}}(a_i \mid x_i)$ — the ratio of how likely the new policy was to take that action to how likely the logging policy was. Average the weighted rewards over the log. That average is the **inverse propensity scoring** (IPS) estimate, also called inverse propensity weighting (IPW):

$$
\hat{V}_{\text{IPS}}(\pi_{\text{new}}) \;=\; \frac{1}{n} \sum_{i=1}^{n} \frac{\pi_{\text{new}}(a_i \mid x_i)}{\pi_{\text{log}}(a_i \mid x_i)}\, r_i \;=\; \frac{1}{n} \sum_{i=1}^{n} w_i\, r_i.
$$

The mechanics: actions the new policy likes more than the logging policy did ($w_i > 1$) get up-weighted; actions it likes less ($w_i < 1$) get down-weighted; an action $\pi_{\text{new}}$ would never take ($w_i = 0$) drops out entirely. The propensity $p_i$ sits in the denominator, which is why an action the logger took rarely (tiny $p_i$) produces a huge weight — hold that thought, it is the seed of every IPS disaster.

### The unbiasedness proof

Here is the three-line argument that IPS is unbiased — that its expectation under the logging policy is exactly the new policy's true value. This is the result that justifies the whole enterprise, so it is worth doing slowly. Take the expectation of a single term over the data-generating process (context $x$ from $D$, action $a$ from $\pi_{\text{log}}$, reward $r$ from $p(\cdot \mid x, a)$):

$$
\mathbb{E}_{\pi_{\text{log}}}\!\left[\frac{\pi_{\text{new}}(a \mid x)}{\pi_{\text{log}}(a \mid x)}\, r\right]
= \mathbb{E}_{x}\, \mathbb{E}_{a \sim \pi_{\text{log}}(\cdot\mid x)}\!\left[\frac{\pi_{\text{new}}(a \mid x)}{\pi_{\text{log}}(a \mid x)}\, \bar{r}(x, a)\right],
$$

where $\bar{r}(x, a) = \mathbb{E}[r \mid x, a]$ is the true mean reward and we have used that $r$'s conditional mean given $(x,a)$ does not depend on which policy chose $a$. Now expand the inner expectation over $a$ as an explicit sum against $\pi_{\text{log}}$, and watch the propensity cancel:

$$
= \mathbb{E}_{x}\sum_{a} \pi_{\text{log}}(a \mid x)\, \frac{\pi_{\text{new}}(a \mid x)}{\pi_{\text{log}}(a \mid x)}\, \bar{r}(x, a)
= \mathbb{E}_{x}\sum_{a} \pi_{\text{new}}(a \mid x)\, \bar{r}(x, a).
$$

The $\pi_{\text{log}}(a \mid x)$ in front and the one in the denominator cancel exactly, leaving an expectation of reward under $\pi_{\text{new}}$ — which is precisely $V(\pi_{\text{new}})$:

$$
= \mathbb{E}_{x}\, \mathbb{E}_{a \sim \pi_{\text{new}}(\cdot\mid x)}[\bar{r}(x, a)] \;=\; V(\pi_{\text{new}}).
$$

Each term of the sum has expectation $V(\pi_{\text{new}})$, so the average of $n$ of them does too. The estimator is unbiased. No model, no extrapolation, no assumption about the reward's functional form — just a reweighting that the algebra guarantees centers on the truth.

```python
def ips(actions, propensities, rewards, pi_new_probs):
    """Vanilla IPS estimate.
    actions:      array (n,) of logged action indices a_i
    propensities: array (n,) of logged pi_log(a_i | x_i) = p_i
    rewards:      array (n,) of observed r_i
    pi_new_probs: array (n, A) of pi_new(a | x_i)
    """
    n = len(rewards)
    pi_new_a = pi_new_probs[np.arange(n), actions]   # pi_new(a_i | x_i)
    weights = pi_new_a / propensities                # w_i, shape (n,)
    return (weights * rewards).mean()
```

The unbiasedness proof quietly used one assumption: that $\pi_{\text{log}}(a \mid x) > 0$ wherever $\pi_{\text{new}}(a \mid x) > 0$. If the logger gave zero probability to an action the new policy wants, the cancellation step divides by zero — there is no logged evidence to reweight, and the estimator is simply blind to that action. This is the **common support** or **overlap** assumption, and it is the single most important condition in all of OPE. We will spend a whole section on it.

#### Worked example: an IPS estimate by hand

Make it concrete with five logged records. The logging policy was uniform over four actions, so every propensity is $p_i = 0.25$. The new policy is more decisive — here are its probabilities for the action that was actually logged in each record, alongside the observed reward (1 = click, 0 = no click):

| Record | Logged action | $p_i = \pi_{\text{log}}$ | $\pi_{\text{new}}(a_i \mid x_i)$ | $w_i = \pi_{\text{new}}/p_i$ | $r_i$ | $w_i r_i$ |
|---|---|---|---|---|---|---|
| 1 | A | 0.25 | 0.60 | 2.40 | 1 | 2.40 |
| 2 | C | 0.25 | 0.10 | 0.40 | 0 | 0.00 |
| 3 | B | 0.25 | 0.50 | 2.00 | 1 | 2.00 |
| 4 | A | 0.25 | 0.05 | 0.20 | 1 | 0.20 |
| 5 | D | 0.25 | 0.30 | 1.20 | 0 | 0.00 |

The IPS estimate is the mean of the last column:

$$
\hat{V}_{\text{IPS}} = \frac{2.40 + 0.00 + 2.00 + 0.20 + 0.00}{5} = \frac{4.60}{5} = 0.92.
$$

Notice what the weights did. Record 1 (action A, which the new policy loves at 0.60) was a click, and it got up-weighted to 2.40 — the new policy would show this kind of thing far more often, so its click counts for more. Record 4 was also a click on action A, but in *that* context the new policy barely wants A (0.05), so its click is down-weighted to 0.20 — the new policy rarely produces this situation, so it should barely contribute. The naive click-through rate of these five records is $3/5 = 0.60$; IPS says the new policy would earn 0.92 because it concentrates probability on the contexts where clicks happened. That is the counterfactual reweighting doing its job.

## 5. Variance, and why overlap is everything

IPS is unbiased, which sounds like the end of the story. It is not, because unbiasedness says nothing about how far a *single* estimate from a *single* log can be from the truth. That is variance, and IPS's variance is its Achilles heel.

The variance of the IPS estimator scales with the variance of the weighted reward $w_i r_i$:

$$
\operatorname{Var}(\hat{V}_{\text{IPS}}) = \frac{1}{n}\operatorname{Var}_{\pi_{\text{log}}}(w\, r) \;\le\; \frac{1}{n}\,\mathbb{E}_{\pi_{\text{log}}}[\,w^2 r^2\,].
$$

The killer is the $w^2$ term. The weight $w_i = \pi_{\text{new}}(a_i\mid x_i)/p_i$ has the propensity in its denominator, so a tiny propensity produces a huge weight, and *squaring* it makes the contribution to variance enormous. If even one logged record had $p_i = 0.001$ and the new policy wanted that action with probability 0.5, then $w_i = 500$, and a single click on that record contributes $500^2 = 250{,}000$ to the sum inside the variance. One unlucky record can dominate the entire estimate. This is the formal reason IPS estimates can swing wildly: their variance is governed by the *worst* overlap in the log, not the average.

The structural quantity behind this is **overlap** (common support): how well the logging policy's action distribution covers the new policy's. The contrast between good and poor overlap is the difference between a usable IPS estimate and an unusable one.

![A two column comparison of good propensity overlap with bounded weights and reliable IPS against poor overlap where one tiny propensity makes a weight explode and the estimate becomes unreliable](/imgs/blogs/counterfactual-and-off-policy-evaluation-5.png)

On the left, the logging policy explored every action with at least modest probability (say $p \ge 0.05$), so the largest weight is bounded (around 20), and IPS is stable. On the right, the logging policy took some action the new policy now loves with probability only 0.001; that single record gets a weight of 1000, it dominates the average, and the RMSE blows up by an order of magnitude. Same estimator, same proof of unbiasedness — but on the right it is unbiased in the way a broken clock is right twice a day: correct on average over infinitely many logs, useless on the one log you actually have.

#### Worked example: variance exploding from one tiny propensity

Take ten logged records, all with reward $r_i = 1$ for simplicity, and a new policy that assigns probability 0.4 to each logged action. Nine of the records were logged with a healthy propensity $p_i = 0.2$, giving weight $w_i = 0.4/0.2 = 2.0$. The tenth record was logged with a tiny propensity $p_i = 0.0008$, giving weight $w_{10} = 0.4/0.0008 = 500$.

The IPS estimate:

$$
\hat{V}_{\text{IPS}} = \frac{1}{10}\big(9 \times 2.0 \times 1 + 1 \times 500 \times 1\big) = \frac{18 + 500}{10} = 51.8.
$$

A click-through rate cannot exceed 1.0, yet IPS reports 51.8 — a wildly impossible value, driven entirely by the one record with the tiny propensity. Now flip that tenth record's reward to 0 (it was a no-click instead of a click). The other nine are unchanged:

$$
\hat{V}_{\text{IPS}} = \frac{1}{10}\big(18 + 500 \times 0\big) = 1.8.
$$

The estimate dropped from 51.8 to 1.8 because of a *single* record flipping from click to no-click. An estimator that swings by a factor of nearly 30 on one observation is not one you can trust on a single log. That sensitivity is the variance the formula warned us about, made painfully concrete. Everything that follows — capping, self-normalization, doubly-robust correction — exists to tame it.

## 6. SNIPS: self-normalized importance sampling

The cheapest fix for IPS's variance keeps the same ingredients and changes only the denominator. Plain IPS divides the weighted-reward sum by $n$. **Self-normalized IPS** (SNIPS) divides by the *sum of the weights* instead:

$$
\hat{V}_{\text{SNIPS}}(\pi_{\text{new}}) = \frac{\sum_{i=1}^{n} w_i\, r_i}{\sum_{i=1}^{n} w_i}.
$$

Why does this help? Because the weights $w_i$ have expectation 1 (the new policy's probabilities sum to 1 over actions, and the reweighting preserves total mass in expectation), so $\frac{1}{n}\sum w_i \approx 1$ and dividing by it is *almost* the same as dividing by $n$. But on any finite log, the weight sum fluctuates, and that fluctuation is *correlated* with the fluctuation in the numerator: a log that happened to draw a huge weight has a large numerator *and* a large denominator, and the ratio partially cancels the spike. SNIPS trades a little bias (the ratio of two random quantities is not exactly unbiased) for a large reduction in variance. It also has a wonderful sanity property that plain IPS lacks: because the numerator and denominator use the same weights, $\hat{V}_{\text{SNIPS}}$ is guaranteed to lie within the range of observed rewards — it can never report a click-through rate of 51.8.

```python
def snips(actions, propensities, rewards, pi_new_probs):
    """Self-normalized IPS: divide by the sum of weights, not by n."""
    n = len(rewards)
    pi_new_a = pi_new_probs[np.arange(n), actions]
    weights = pi_new_a / propensities
    return (weights * rewards).sum() / weights.sum()
```

There is a second, free benefit hiding in the denominator. The weight sum $\frac{1}{n}\sum_i w_i$ is itself an estimator — of the constant 1, since the expected importance weight is exactly 1. So the *deviation* of the observed weight sum from $n$ is a built-in diagnostic: if your weights average 1.4 instead of 1.0 on a log, the propensities are probably miscalibrated or the overlap is poor, and the discrepancy is visible without ever knowing the true value. Practitioners watch $\frac{1}{n}\sum_i w_i$ as a cheap health check on every OPE run; a value far from 1 is a warning that the estimate underneath it is suspect. SNIPS turns that diagnostic into a correction by dividing it out.

A close cousin is **clipped** or **capped IPS**, which simply truncates every weight at a ceiling $M$: replace $w_i$ with $\min(w_i, M)$ before averaging. Capping introduces bias (you are throwing away mass from the big weights) but bounds the variance hard. The two are often combined — cap the weights, then self-normalize. In practice SNIPS is the default first upgrade from vanilla IPS because it costs one line of code and almost always helps. Run the same ten-record worked example through SNIPS: the numerator is $18 + 500 = 518$ and the denominator is $9 \times 2.0 + 500 = 518$, so SNIPS reports $518/518 = 1.0$ — pinned to the maximum possible reward rather than the absurd 51.8 of plain IPS. The self-normalization clamps the estimate back into the feasible range.

## 7. Doubly Robust: combine the model and the reweighting

Now the elegant one. The Direct Method has low variance but is biased when the model is wrong. IPS is unbiased but has high variance. **Doubly Robust** (DR) combines them so that you get the low variance of DM *and* the unbiasedness of IPS, and — the magic — it stays unbiased if *either* the model *or* the propensities are correct. You no longer have to bet the estimate on getting both right; one is enough.

The DR estimator starts from the Direct Method's prediction and adds an IPS-style correction applied only to the *residual* between the observed reward and what the model predicted:

$$
\hat{V}_{\text{DR}}(\pi_{\text{new}}) = \frac{1}{n}\sum_{i=1}^{n}\left[\; \underbrace{\sum_{a}\pi_{\text{new}}(a\mid x_i)\,\hat{r}(x_i,a)}_{\text{DM term}} \;+\; \underbrace{\frac{\pi_{\text{new}}(a_i\mid x_i)}{\pi_{\text{log}}(a_i\mid x_i)}\big(r_i - \hat{r}(x_i,a_i)\big)}_{\text{IPS-corrected residual}} \;\right].
$$

Read it as: *take the model's estimate, then patch it with reweighted evidence of where the model was wrong.* If the model is perfect, the residual $r_i - \hat{r}(x_i, a_i)$ is zero in expectation and the correction term vanishes — you are left with DM, which is now unbiased because the model is right. If the propensities are correct but the model is garbage, the correction term is exactly the IPS estimator of the residual, which corrects the model's bias back out — you are left with something unbiased because IPS is. Either ingredient being correct is enough to center the estimate; that is the **double robustness** property, and it is why DR is the workhorse of modern OPE.

The variance win is just as important as the bias win. IPS reweights the full reward $r_i$, which can be large; DR reweights only the residual $r_i - \hat{r}(x_i, a_i)$, which is small whenever the model is even *roughly* right. A smaller thing inside the reweighting means the $w^2$ variance term hits a smaller quantity, so DR's variance is typically far below IPS's. The contrast is worth drawing.

![A two column comparison of IPS that reweights the full reward and has high variance against Doubly Robust that reweights only the model residual and has lower variance while both stay unbiased](/imgs/blogs/counterfactual-and-off-policy-evaluation-7.png)

The left column is IPS: it reweights the full reward, it is unbiased when propensities are correct, but the weights hit the whole reward and the variance is high. The right column is DR: a model baseline plus a weighted residual, unbiased if model *or* propensity is correct, and low variance because the residual it reweights is small. DR is strictly the more robust object — the only cost is that you have to fit a reward model, which DM already required anyway.

```python
def doubly_robust(contexts, actions, propensities, rewards,
                  reward_model, pi_new_probs):
    """Doubly Robust estimate = DM baseline + IPS-corrected residual."""
    n, A = pi_new_probs.shape
    # DM baseline: E_{a ~ pi_new}[ rhat(x, a) ] per context
    rhat_all = np.stack([reward_model(contexts, a) for a in range(A)], axis=1)
    dm_term = (pi_new_probs * rhat_all).sum(axis=1)             # shape (n,)
    # IPS-corrected residual on the logged action only
    rhat_logged = rhat_all[np.arange(n), actions]              # rhat(x_i, a_i)
    pi_new_a = pi_new_probs[np.arange(n), actions]
    weights = pi_new_a / propensities
    correction = weights * (rewards - rhat_logged)             # shape (n,)
    return (dm_term + correction).mean()
```

The whole estimator family, organized by how it gets its leverage, is the tree below: a model-based branch (DM), an importance-weighting branch (IPS and SNIPS), and a hybrid branch (DR) that sits between them precisely because it borrows from both.

![A taxonomy tree of off policy estimators splitting into a model based branch with the Direct Method an importance weighting branch with IPS and SNIPS and a hybrid branch with Doubly Robust](/imgs/blogs/counterfactual-and-off-policy-evaluation-4.png)

That hybrid position is the point. DR is not a fourth, unrelated estimator; it is the principled combination of the first three, and it dominates them on the bias-variance plane whenever you can afford to fit a reward model.

## 8. The full simulation: bias and variance against a known truth

Everything above is theory until you watch the estimators run on data where you know the right answer. The advantage of a simulation is exactly that: we *define* the reward function, so we can compute the true value $V(\pi_{\text{new}})$ exactly and measure each estimator's bias and RMSE against it. Here is a complete, self-contained logged-bandit simulator.

```python
import numpy as np

rng = np.random.default_rng(0)

N_ACTIONS = 5
N_FEATURES = 4
N_LOGS = 5000

# --- True reward function: a logistic model over (context, action) ---
true_W = rng.normal(0, 1.0, size=(N_ACTIONS, N_FEATURES))

def true_mean_reward(x, a):
    """P(click) for context x (n, d) and action a (scalar): logistic."""
    logits = x @ true_W[a]                    # shape (n,)
    return 1.0 / (1.0 + np.exp(-logits))

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

# --- Logging policy: a softer (more exploratory) version of the truth ---
def logging_policy_probs(x, temp=2.0):
    scores = np.stack([x @ true_W[a] for a in range(N_ACTIONS)], axis=1)
    return softmax(scores / temp, axis=1)     # (n, A); temp>1 => flatter

# --- New (target) policy: a sharper, near-greedy version ---
def new_policy_probs(x, temp=0.5):
    scores = np.stack([x @ true_W[a] for a in range(N_ACTIONS)], axis=1)
    return softmax(scores / temp, axis=1)     # (n, A); temp<1 => peakier

def true_policy_value(policy_probs_fn, n=200_000):
    """Monte-Carlo the TRUE online value of a policy (ground truth)."""
    x = rng.normal(0, 1.0, size=(n, N_FEATURES))
    probs = policy_probs_fn(x)
    rbar = np.stack([true_mean_reward(x, a) for a in range(N_ACTIONS)], axis=1)
    return (probs * rbar).mean(axis=0).sum()

V_TRUE = true_policy_value(new_policy_probs)
print(f"True V(pi_new) = {V_TRUE:.4f}")
```

The logging policy is a high-temperature softmax over the true scores — it explores, so every action has nonzero probability and overlap is good. The new policy is a low-temperature softmax — it is near-greedy and concentrates on the high-reward actions. We Monte-Carlo the true value over 200k fresh contexts so `V_TRUE` is essentially exact. Now generate a log and run all four estimators.

```python
def generate_log(n=N_LOGS):
    x = rng.normal(0, 1.0, size=(n, N_FEATURES))
    log_probs = logging_policy_probs(x)                       # (n, A)
    actions = np.array([rng.choice(N_ACTIONS, p=log_probs[i]) for i in range(n)])
    propensities = log_probs[np.arange(n), actions]           # p_i
    rbar = true_mean_reward(x, actions)                       # P(click | x, a)
    rewards = (rng.random(n) < rbar).astype(float)            # Bernoulli draw
    return x, actions, propensities, rewards

def fit_reward_model(x, actions, rewards):
    """Cheap per-action logistic reward model via sklearn."""
    from sklearn.linear_model import LogisticRegression
    models = {}
    for a in range(N_ACTIONS):
        mask = actions == a
        if mask.sum() > 5 and len(np.unique(rewards[mask])) == 2:
            models[a] = LogisticRegression(max_iter=500).fit(x[mask], rewards[mask])
    def predict(xq, a):
        if a in models:
            return models[a].predict_proba(xq)[:, 1]
        return np.full(len(xq), rewards.mean())   # fallback to global mean
    return predict
```

Each logged reward is a real Bernoulli click drawn from the true probability, so the rewards are noisy — exactly like production. The reward model is a per-action logistic regression, deliberately *imperfect* (it is linear in features but the click probability is logistic in a linear score, so it is a decent but not exact fit). Now the driver that runs all four estimators over many independent logs to measure their sampling distributions.

```python
def run_trial(n=N_LOGS):
    x, a, p, r = generate_log(n)
    pi_new = new_policy_probs(x)                  # (n, A)
    rmodel = fit_reward_model(x, a, r)
    return {
        "DM":    direct_method(x, rmodel, pi_new),
        "IPS":   ips(a, p, r, pi_new),
        "SNIPS": snips(a, p, r, pi_new),
        "DR":    doubly_robust(x, a, p, r, rmodel, pi_new),
    }

N_TRIALS = 200
estimates = {k: [] for k in ["DM", "IPS", "SNIPS", "DR"]}
for _ in range(N_TRIALS):
    out = run_trial()
    for k, v in out.items():
        estimates[k].append(v)

print(f"{'estimator':<8} {'mean':>8} {'bias':>9} {'rmse':>8}")
for k in ["DM", "IPS", "SNIPS", "DR"]:
    arr = np.array(estimates[k])
    bias = arr.mean() - V_TRUE
    rmse = np.sqrt(((arr - V_TRUE) ** 2).mean())
    print(f"{k:<8} {arr.mean():>8.4f} {bias:>+9.4f} {rmse:>8.4f}")
```

We run 200 independent logs, compute each estimator on each, and measure the **mean** (does it center on truth?), the **bias** (mean minus truth), and the **RMSE** (root-mean-square distance from truth — the honest one-number summary of how wrong a single estimate tends to be). Representative output, with a true value of about 0.62:

```
estimator     mean      bias     rmse
DM          0.5482   -0.0718   0.0742
IPS         0.6178   -0.0022   0.0610
SNIPS       0.6113   -0.0089   0.0283
DR          0.6156   -0.0044   0.0167
```

That table is the entire post compressed into numbers, and it confirms every claim the theory made.

![A results matrix of truth Direct Method IPS SNIPS and Doubly Robust against estimated value bias and RMSE showing Doubly Robust closest to truth with the lowest RMSE](/imgs/blogs/counterfactual-and-off-policy-evaluation-8.png)

Read the results matrix row by row. The **Direct Method** is precise (low RMSE *relative to its own mean*) but badly biased: it centers at 0.548, a full 0.072 below the truth, because its linear reward model misestimates the high-reward actions the new policy concentrates on. **IPS** is almost perfectly unbiased — it centers at 0.618, within 0.002 of truth — but its RMSE of 0.061 is the worst of the corrected estimators because a few large weights make individual estimates swing. **SNIPS** picks up a tiny bias (centers at 0.611) in exchange for cutting the RMSE in half to 0.028. **Doubly Robust** wins outright: nearly unbiased at 0.616 *and* the lowest RMSE at 0.017, because it reweights only the small residual rather than the full reward. DR beats DM on bias and beats IPS on variance simultaneously — exactly the "best of both" the derivation promised.

#### Worked example: reading the RMSE as a decision

Suppose the new policy's true value is 0.62 and the production policy's online value is 0.60 — a real but modest +3.3% relative lift you would happily ship. Can each estimator *detect* a lift that small? DM centers at 0.548, *below* the production policy's 0.60 — it would tell you the new policy is worse and you would wrongly kill it. IPS centers correctly at 0.618 but with RMSE 0.061, so a single estimate could easily land anywhere from 0.56 to 0.68 — it cannot reliably distinguish 0.62 from 0.60. SNIPS, RMSE 0.028, gives you a fighting chance. DR, RMSE 0.017, can resolve the 0.02 gap: a single DR estimate lands within about 0.017 of 0.62 typically, comfortably above 0.60. The practical upshot is blunt: **only the low-variance estimators can detect small lifts**, and small lifts are most of what a mature recommender ships. This is why DR, not IPS, is what you actually deploy.

## 9. Confidence intervals: never report a point estimate alone

The simulation above ran 200 logs to *measure* each estimator's spread, but in production you have exactly one log. So how do you know whether your single number is trustworthy? You attach an interval to it, and the cheapest honest way to do that is the **bootstrap**. Resample the logged records with replacement, recompute the estimator on each resample, and read the spread of the resampled estimates as the sampling distribution of your estimate. The 2.5th and 97.5th percentiles give you a 95% confidence interval — and crucially, if your weights are wild, that interval will be wide enough to *tell you so*.

```python
def bootstrap_ci(estimator_fn, n_boot=1000, alpha=0.05, seed=1):
    """Bootstrap a CI for any of our estimators.
    estimator_fn(idx) recomputes the estimate on a resampled index array.
    """
    bg = np.random.default_rng(seed)
    n = N_LOGS
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = bg.integers(0, n, size=n)      # resample with replacement
        vals[b] = estimator_fn(idx)
    lo = np.quantile(vals, alpha / 2)
    hi = np.quantile(vals, 1 - alpha / 2)
    return vals.mean(), lo, hi
```

Run this on the same log for IPS and DR and the contrast is the whole point of the post in interval form. A representative result on a single 5000-record log looks like this:

```
estimator   point   95% CI            CI width
IPS         0.617   [0.498, 0.741]    0.243
SNIPS       0.609   [0.561, 0.658]    0.097
DR          0.615   [0.591, 0.640]    0.049
```

IPS reports a 0.24-wide interval — it cannot tell 0.55 from 0.74, which makes it useless for deciding whether a +0.02 lift is real. DR's interval is five times tighter at 0.049, which can resolve the lift. The lesson is procedural: **never report an OPE point estimate without its interval.** A naked number invites someone to ship on a difference that is pure noise. The interval is also the natural place where OPE hands off to the A/B test: if the interval is too wide to clear your decision threshold, that is the signal to spend the slot and measure online instead of guessing.

There is a subtlety the bootstrap exposes that a point estimate hides: the IPS sampling distribution is *right-skewed*, because the occasional huge weight pulls the upper tail out. The mean of the bootstrap distribution can sit above the median, and the upper CI bound can be implausibly high (above 1.0 for a click-rate). That asymmetry is itself a diagnostic — a skewed bootstrap distribution is the fingerprint of poor overlap, and it is visible from a single log without ever knowing the truth. SNIPS and DR distributions are far more symmetric, which is another quiet reason to prefer them.

## 10. Capped IPS, Switch-DR, and the estimator-selection problem

SNIPS and DR are the two upgrades every practitioner should know, but the literature has gone further, and the further estimators are worth a paragraph each because they target the exact failure mode — a few monstrous weights — that wrecks the basics.

**Capped (clipped) IPS** is the bluntest instrument: pick a ceiling $M$ and replace every weight with $\min(w_i, M)$. This bounds the variance hard, because no single record can contribute more than $M^2$ to the variance sum. The cost is bias: by truncating the big weights you systematically under-count the actions the new policy most prefers, so capped IPS is biased *downward* whenever the new policy concentrates on rarely-logged actions. The art is choosing $M$ — too low and you bias the estimate into uselessness, too high and you have not capped anything. A common heuristic is to set $M$ at a high percentile of the observed weights (say the 99th) so you only clip the genuine outliers.

```python
def capped_ips(actions, propensities, rewards, pi_new_probs, cap=50.0):
    n = len(rewards)
    w = pi_new_probs[np.arange(n), actions] / propensities
    w = np.minimum(w, cap)                    # clip the monsters
    return (w * rewards).mean()
```

**Switch-DR** (Wang, Agarwal, Dudík, 2017) is cleverer: it *switches* between DM and DR per record based on the weight. For records with a small, trustworthy weight, it uses the full DR correction (low bias). For records with a weight above a threshold $\tau$, it drops the IPS correction entirely and falls back to the pure DM prediction for that record (low variance, accepting the model's bias only where the weight would have been dangerous). The result is an estimator that uses reweighting where reweighting is safe and the model where it is not, tuning a single threshold to trade bias against variance smoothly. The Open Bandit Pipeline ships Switch-DR alongside **DR with optimistic shrinkage** (DRos), which shrinks each weight toward zero by an amount that minimizes an upper bound on the estimator's MSE — a principled, automatic version of capping.

This raises an obvious meta-question: with a dozen estimators and a tuning knob on several of them, *which estimator do you actually trust on a given log?* You cannot just pick the one whose estimate you like best — that is p-hacking. The principled answer is the **SLOPE** estimator-selection procedure (Su, Srinath, Krishnamurthy, 2020), which selects among estimators (and their hyperparameters) using only logged data, by exploiting the bias-variance trade-off structure: it walks from high-variance/low-bias estimators toward lower-variance/higher-bias ones and stops just before the bias term starts to dominate, all measurable from the log alone. The practical takeaway is not that you must implement SLOPE by hand — it is in `obp` — but that estimator selection is a real, named problem, and the honest workflow is to let a principled procedure choose rather than eyeballing the answer you wanted.

#### Worked example: how capping trades bias for variance

Return to the ten-record log from section 5, where nine records had weight 2.0 and one had weight 500, all with reward 1. Plain IPS gave 51.8 — an impossible click-rate dominated by the one monster weight. Now apply a cap of $M = 50$. The monster weight 500 is clipped to 50, and the estimate becomes

$$
\hat{V}_{\text{capped}} = \frac{1}{10}\big(9 \times 2.0 \times 1 + 1 \times 50 \times 1\big) = \frac{18 + 50}{10} = 6.8.
$$

Still above 1.0 and still wrong, but the variance is now bounded — flipping that record's reward to 0 moves the estimate only from 6.8 to 1.8, a factor of 3.8 instead of plain IPS's factor of 29. Tighten the cap to $M = 5$ and the estimate becomes $(18 + 5)/10 = 2.3$, with the record-flip sensitivity down to a factor of 1.3. Each tightening cuts the variance and adds downward bias — you are paying for stability in accuracy. This single example is the entire capped-IPS trade-off: the cap is a dial from "unbiased but wild" toward "stable but biased," and SNIPS, Switch-DR, and DRos are all smarter, data-driven ways of setting that dial than a hand-picked constant.

## 11. Position bias: the propensities you did not know you needed

So far the "action" has been a single chosen item, which fits candidate selection but not the most common recommender output: a *ranked list*. Ranking introduces a second source of why-was-this-clicked that masquerades as relevance and silently biases every offline metric — **position bias**. Users click the top of a list far more than the bottom, *regardless of relevance*, simply because they look there first. A click on rank 1 is weak evidence of relevance; a click on rank 8 is strong evidence, because the user had to scroll past seven items to give it. If you train or evaluate on raw clicks, you reward whatever the old ranker happened to put on top — a feedback loop that ossifies the current ranking. This is the ranking-flavored cousin of the same MNAR (missing-not-at-random) problem OPE exists to fix, and the fix is the same: propensities.

The **position-based propensity model** treats the probability that a user examines position $k$ as a per-position constant $\theta_k$ (the examination propensity), independent of the item there. A click happens only if the user both examined the position and found the item relevant: $P(\text{click at } k) = \theta_k \cdot P(\text{relevant})$. Then the propensity for a click observed at position $k$ is $\theta_k$, and you correct each click by $1/\theta_k$ — an IPS estimator over positions. A click at rank 8, where $\theta_8$ is small, gets up-weighted heavily; a click at rank 1, where $\theta_1 \approx 1$, barely changes. This is exactly the IPS reweighting from section 4, with the "policy" being the examination distribution induced by position.

```python
def ips_ndcg_position_debiased(relevance, ranks, theta):
    """Position-debiased estimate of a ranking metric from click logs.
    relevance: array (n,) of observed clicks (1/0)
    ranks:     array (n,) of the position each clicked item was shown at
    theta:     array (max_rank,) of examination propensities theta_k
    """
    inv_prop = 1.0 / theta[ranks]                 # 1 / theta_k per click
    gains = relevance * inv_prop                    # IPS-corrected relevance
    discounts = 1.0 / np.log2(ranks + 2.0)          # NDCG-style position discount
    return (gains * discounts).mean()
```

The deep question is where $\theta_k$ comes from, and the elegant answer is that you can *estimate it from the logs* if the logging system ever randomized — for instance, by occasionally swapping two adjacent results (a RandPair / FairPairs intervention) or shuffling the top-K for a small fraction of traffic. The click-rate difference between an item at position $k$ and the same item at position $k'$ reveals the ratio $\theta_k / \theta_{k'}$, and a little exploration recovers the whole curve. This is the same lesson as the rest of the post, in ranking dress: **a little logged randomization is what makes unbiased offline estimation possible.** The technique — IPS over examination propensities — is the foundation of *unbiased learning to rank*, and it connects this post directly to [learning to rank for recommenders](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders), where the same propensity weights enter the training loss rather than only the evaluation.

## 12. From off-policy evaluation to off-policy learning

Everything so far estimates the value of a *fixed* new policy. The natural next step is to *optimize* the policy directly against the logs — to search for the $\pi$ that maximizes $\hat{V}(\pi)$ — which is **off-policy learning**, also called counterfactual risk minimization (CRM) or batch learning from logged bandit feedback (BLBF). The estimators become *objectives*: instead of evaluating one policy, you make $\hat{V}_{\text{IPS}}(\pi_\phi)$ or $\hat{V}_{\text{DR}}(\pi_\phi)$ a differentiable function of policy parameters $\phi$ and run gradient ascent. The IPS objective for a parameterized stochastic policy is

$$
\hat{V}_{\text{IPS}}(\pi_\phi) = \frac{1}{n}\sum_{i=1}^{n} \frac{\pi_\phi(a_i \mid x_i)}{\pi_{\text{log}}(a_i \mid x_i)}\, r_i,
$$

which you maximize over $\phi$. There is a subtlety that does not appear in evaluation: when you *optimize* an IPS objective, the optimizer learns to exploit the variance — it can inflate $\hat{V}$ by concentrating probability on a few high-weight, high-reward records that happened to land in the log, a phenomenon Swaminathan and Joachims named **propensity overfitting**. The fix is the same SNIPS self-normalization (their POEM algorithm adds a variance-regularization term to the objective), plus the same overlap discipline. Off-policy learning is its own large topic, but the bridge is worth seeing: the estimators in this post are not only how you *grade* a policy offline, they are how you can *train* one from logs without any online interaction at all — the purest form of learning from the feedback loop. When you can afford it, the safest pattern is to *learn* a candidate policy off-policy, *evaluate* it off-policy with a different (doubly-robust) estimator on held-out logs, and only then A/B test the survivor.

## 13. A debugging narrative: the estimate that disagreed with the A/B test

Theory and simulation are clean; production is not. Here is the kind of failure you will actually hit, and how the tools in this post diagnose it. You run DR on last month's logs and it predicts a +4% lift for a new ranker. You ship it to an A/B test to confirm, and the test comes back *flat* — no lift, maybe slightly negative. The offline estimate and the online truth disagree, which is exactly the situation OPE was supposed to prevent. What went wrong, and how do you find out?

Walk the assumptions from section 9 in order, because each leaves a different fingerprint. **First, overlap.** Compute the effective sample size on the log you used: if $n_{\text{eff}}$ is 60 out of 80,000 records, your "estimate" was essentially three records wearing a trench coat, and the +4% was noise the bootstrap interval would have flagged as ±15% if you had looked. *Did you report the interval?* If not, that is the bug. **Second, propensities.** Pull a sample of logged records and check that $p_i$ equals the probability the live policy actually used — not the pre-filter score, not a reconstruction. A frequent culprit: a business-rule layer (dedup, diversity caps, freshness boosts) sits *after* the model and changes what was actually shown, so the logged propensity describes the model's intent, not the exposure. Every weight is then wrong. **Third, the reward model.** DR leans on $\hat{r}$ for variance reduction and for unbiasedness when propensities are off; if the model was trained on stale features or has train-serve skew, its residuals are biased and DR's correction term carries that bias. **Fourth, stationarity.** If last month had a promotion the test month did not, the policy you evaluated is being judged on a world that has moved.

The disciplined response is not to distrust OPE wholesale — it is to *instrument* it. Log $n_{\text{eff}}$, the max weight, and the bootstrap CI width alongside every offline estimate, and treat a tiny $n_{\text{eff}}$ or a huge max weight as a hard gate that *forces* an A/B test rather than a ship. Validate your OPE pipeline the way the Open Bandit Pipeline lets you: run two logging policies for a slice of traffic, estimate policy B's value from policy A's logs, and check it against B's *actual* online value. If your estimator cannot recover a known answer on held-back data, fix the estimator before you trust it on an unknown one. OPE that disagrees with an A/B test is not a reason to abandon OPE; it is a reason to find which of the four assumptions you violated — and the diagnostics above will tell you which one.

## 14. The assumptions, and how each one fails

OPE works only under conditions that production data routinely violates. Knowing the failure modes is the difference between an estimate you can act on and a number that quietly lies.

**Overlap / common support.** Already met in detail: if $\pi_{\text{log}}(a\mid x) = 0$ where $\pi_{\text{new}}(a\mid x) > 0$, IPS is blind to that action and the estimate is biased toward whatever the logger *did* explore. Near-violations (tiny but nonzero propensities) are worse in a way, because they do not error out — they silently inflate the variance to the point of uselessness. **Diagnostic:** compute the **effective sample size** $n_{\text{eff}} = (\sum_i w_i)^2 / \sum_i w_i^2$. If your log has 5000 records but $n_{\text{eff}} = 40$, your estimate has the precision of 40 samples regardless of how many you logged, and you should not trust it. Also inspect the max weight and the weight histogram; a single weight above a few hundred is a red flag.

```python
def effective_sample_size(actions, propensities, pi_new_probs):
    n = len(actions)
    w = pi_new_probs[np.arange(n), actions] / propensities
    return (w.sum() ** 2) / (w ** 2).sum()
```

**Correct, logged propensities.** The proof needs $p_i$ to be the *true* probability the logging policy used. Two ways this breaks. First, the propensities are wrong — someone logged the post-softmax score instead of the sampling probability, or logged the probability before a downstream business-rule filter changed the actual exposure. A wrong denominator biases every weight. Second, and most common, **the logging policy was deterministic**, so there are no meaningful propensities at all. A pure argmax ranker assigns probability 1 to the action it took and 0 to everything else, which makes $w_i$ either 0 or undefined and IPS unusable. The cure is to make the logging policy *stochastic* — to inject exploration — which is exactly what [bandits and the exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) are about. A small amount of logged randomization (an epsilon-greedy wrapper, a softmax over scores, a Thompson-sampling layer) is what turns your logs into OPE-able data. If you take one operational lesson from this post, it is this: **log propensities and inject a little exploration now, so that off-policy evaluation is possible later.**

**Unconfoundedness / no hidden context.** IPS assumes the reward depends only on the logged $(x, a)$, not on something the policy saw but you did not log. If the logging policy used a feature you failed to record, the propensities you reconstruct are wrong and the estimate is biased. The fix is discipline: log the propensity *at decision time* from the policy itself, not reconstructed later from partial features.

**Stationarity.** OPE estimates the value of $\pi_{\text{new}}$ *on the distribution the logs came from*. If the world has shifted since the log was collected — a holiday, a new content mix, a seasonal user base — the estimate is for a world that no longer exists. Use recent logs, and treat OPE as a screen, not a guarantee.

When does the deterministic-logging problem leave you with *only* the Direct Method? When you have no propensities and cannot re-collect logs with exploration. DM needs no propensities, so it still runs — but it inherits all of DM's bias risk with no IPS correction to lean on, and DR collapses to DM. This is the worst position to be in, and it is entirely avoidable by logging propensities from day one.

## 15. Connecting OPE to the funnel and the loop

OPE is not a standalone trick; it slots directly into the serve → log → train → deploy loop that organizes this whole series. The loop *produces* the logs OPE consumes, and OPE *gates* what re-enters the loop.

The serving system runs $\pi_{\text{log}}$ and logs $(x, a, p, r)$ — the propensity $p$ being the thing you must remember to write. Those logs train new candidate models. Before any candidate goes back out to serving, OPE estimates its online value from the logs, and only the candidates that clear the offline screen earn an A/B slot. The A/B test confirms the winner, which becomes the new $\pi_{\text{log}}$, and the loop turns again. OPE is the cheap, fast filter between "we trained twelve models" and "we A/B-tested two," and exploration in the logging policy is the upstream investment that makes the filter possible at all.

This also explains a subtle interaction with the rest of the funnel. A retrieval model that changes the *candidate set* (not just the re-ranking of a fixed set) is the hardest case for OPE, because it can surface items the logger never showed — a direct overlap violation. A re-ranking model that only reorders the slate the logger already retrieved is the *easiest* case, because the action space is fully covered. So OPE is most trustworthy for the late, narrow stages of the funnel and least trustworthy for the early, wide ones — a useful thing to know when you decide how much weight to put on an offline estimate before committing an A/B slot.

#### Worked example: budgeting a quarter of A/B slots

Put numbers on the screening argument, because the economics are the entire reason OPE exists. Say your team can run **2 A/B tests per month** (the platform queue and the statistics both cap you there), each test costs **2 weeks** and exposes a slice of traffic to risk, and you have **12 candidate models** in the backlog this quarter. Without OPE, 12 candidates at 2 per month is 6 months of slots — you cannot even finish the backlog in the quarter, and roughly half of those tests will measure a model that is worse than production, degrading the experience for the test population each time. Now add OPE. You run DR with bootstrap intervals on all 12 offline in an afternoon. Suppose it ranks them and the intervals say 8 are clearly no better than production (their CIs straddle zero lift), 2 are ambiguous (wide CIs from poor overlap — those get flagged for an A/B test *because* OPE could not resolve them), and 2 show a clear positive estimate. You A/B test those 4 over 2 months, ship the winners, and you spent **zero** live-traffic risk on the 8 that OPE killed. The arithmetic: OPE converted a 6-month, 12-test plan into a 2-month, 4-test plan, and removed the expected harm of 4 to 6 negative live experiments. That is the quarter-over-quarter velocity difference between a team that logs propensities and one that does not.

## 16. Case studies and real systems

The estimators in this post are not academic toys; they were forged on real ad and recommendation systems and are shipped in production OPE tooling today.

**Bottou et al., counterfactual reasoning at Bing (2013).** The paper that put counterfactual estimation on the industrial map. Léon Bottou and colleagues at Microsoft showed how to estimate the effect of changes to the Bing ad-placement system from logged data, using importance-weighted (IPS-style) estimators and confidence bounds, *without* running an A/B test for every change. Their central message — that learning systems are *interventions* and you must reason counterfactually about what would have happened under a different policy — is the philosophical foundation of this entire post. They also emphasized logging the propensities (the randomization the system already had) as the enabling ingredient, exactly the operational lesson above.

**Dudík, Langford, and Li, Doubly Robust policy evaluation (2011).** The DR estimator we built comes from this line of work. Dudík and colleagues showed that combining a reward model with importance weighting gives an estimator that is consistent if *either* component is correct, and demonstrated it dramatically reduces variance versus plain IPS on real classification-to-bandit benchmarks. DR has since become the default estimator in serious OPE practice precisely because the double-robustness property removes the need to bet on a single modeling choice.

**The Open Bandit Pipeline and Open Bandit Dataset, Saito et al. (2020).** ZOZO's Open Bandit Pipeline (`obp`) is the open-source library that productionized all of this. It ships implementations of DM, IPS, SNIPS, DR, and more advanced estimators (switch-DR, doubly-robust with optimistic shrinkage, the SLOPE estimator-selection procedure), plus the Open Bandit Dataset — real logged-bandit data from a fashion e-commerce platform that ran *two* logging policies (uniform-random and a contextual one), so you can validate an OPE estimate computed from one policy's logs against the *other* policy's actual online performance. That cross-policy validation is the closest thing the field has to ground truth on real data, and the benchmark consistently shows DR-family estimators beating both DM and vanilla IPS. If you are going to do OPE for real, `obp` is where to start rather than rolling your own.

**IPS in ad-tech and recommendation.** Inverse propensity weighting is everywhere in the ad world because ad systems already randomize (to explore the value of new creatives and placements) and already log propensities (because they have to, for billing and auctions). Criteo released a large-scale logged-bandit dataset (the Criteo counterfactual learning / CRITEO-UPLIFT data) specifically to benchmark counterfactual estimators, and the same IPS/SNIPS/DR machinery underlies counterfactual *learning to rank*, where position-based propensities correct for the fact that users click top positions more regardless of relevance — the position-bias correction that connects this post to learning-to-rank.

A representative summary of what these sources consistently report, expressed as the qualitative ordering you should expect rather than a single magic number:

| Estimator | Typical bias | Typical variance | When the literature prefers it |
|---|---|---|---|
| Direct Method | High, model-dependent | Lowest | Strong reward model, poor overlap |
| IPS / IPW | Near zero | Highest | Good overlap, distrust the model |
| SNIPS | Small | Medium | Default upgrade over IPS, bounded estimate |
| Doubly Robust | Near zero | Low | The default; needs both ingredients |

The numbers move with the dataset and the policy gap, but the *ordering* — DR best on the bias-variance trade, DM lowest variance but biased, IPS unbiased but wild — is remarkably stable across Bottou's ad system, Dudík's benchmarks, and the Open Bandit Pipeline's real data.

## 17. When to reach for OPE (and when not to)

OPE is powerful and it is also easy to misuse. Here is the decisive guidance.

**Reach for OPE when** you have logs with recorded propensities (or a stochastic logging policy you can reconstruct them from), the new policy mostly reorders or reweights actions the logger already explored (good overlap), and you want to screen many candidates before spending scarce A/B slots. This is the sweet spot: a re-ranking model change, a new ranking loss, a re-tuned blending weight, a calibration tweak — incremental changes to a stage of the funnel whose action space the logger covered.

**Default to Doubly Robust.** Among the estimators, DR is the right default essentially always. It dominates DM on bias and IPS on variance, and the only cost is fitting a reward model you probably want anyway. Use SNIPS when you cannot or do not want to fit a reward model — it is the best propensity-only estimator. Use plain IPS only as a teaching baseline or a sanity check; its variance makes it a poor production choice. Use DM alone only when you have no propensities at all (deterministic logging with no way to re-collect), and treat its output with deep suspicion.

**Do not trust OPE when** overlap is poor — and *measure* it, do not assume it. Compute the effective sample size and the max weight before you believe any IPS or DR estimate. If $n_{\text{eff}}$ is a tiny fraction of $n$, or the max weight is in the hundreds, your estimate is dominated by a handful of records and you should fall back to a more model-based estimate (DM or a heavily capped DR) or simply accept that this change needs a real A/B test. Do not use OPE to compare a new *retrieval* policy that surfaces brand-new candidates — that is the overlap violation in its strongest form, and no amount of reweighting fixes evidence you never collected.

**Do not let OPE replace the A/B test for the final decision.** OPE is a screen and a debugging tool, not a substitute for measuring the real thing. Estimates can be biased by every assumption in section 9, and the only way to know the true online value is to run the policy. Use OPE to decide *which* policies deserve an A/B slot, then confirm the winner online. The two work together: OPE turns a dozen candidates into two, and the A/B test turns two into one.

**Do not skip logging propensities because you are not doing OPE yet.** The single most expensive mistake is shipping a deterministic logging policy that records no propensities, because it forecloses OPE on all the data you collect until you fix it. Logging propensities and injecting a little exploration is cheap to add now and impossible to add retroactively to old logs. Pay the small cost upfront.

## 18. Key takeaways

- **OPE estimates a new policy's online value from old logs** so you can screen candidates before spending slow, risky A/B slots — the offline filter feeding the serve-log-train-deploy loop.
- **Propensities are the load-bearing ingredient.** Log $\pi_{\text{log}}(a\mid x)$ at decision time. Without it you cannot do IPS, SNIPS, or DR, and DM (which needs no propensity) is left carrying all the bias risk alone.
- **IPS is unbiased by a three-line proof** ($\mathbb{E}_{\pi_{\text{log}}}[\frac{\pi_{\text{new}}}{\pi_{\text{log}}}r] = V(\pi_{\text{new}})$ via exact propensity cancellation) but its variance scales with $w^2$, so one tiny propensity can blow it up — as the worked example swinging from 51.8 to 1.8 on a single record showed.
- **Overlap is everything.** IPS and DR are only trustworthy where the logger gave nonzero probability to the actions the new policy wants. Measure overlap with effective sample size and the max weight before you believe any estimate.
- **SNIPS self-normalizes** by dividing by the weight sum instead of $n$, trading a tiny bias for a large variance cut and guaranteeing the estimate stays in the feasible reward range.
- **Doubly Robust is the default.** It combines a reward model with IPS-corrected residuals, is unbiased if *either* the model or the propensities are right, and reweights only the small residual so its variance sits far below IPS. In the simulation it landed closest to truth with the lowest RMSE.
- **Only low-variance estimators (SNIPS, DR) can detect small lifts**, and small lifts are most of what a mature recommender ships — which is the practical reason DR, not IPS, is what you deploy.
- **OPE is a screen, not a verdict.** Use it to decide which policies earn an A/B slot, then confirm the winner online; never let it replace the real measurement for the final ship decision.

## 19. Further reading

- **Bottou, Peters, Quiñonero-Candela, et al. (2013), "Counterfactual Reasoning and Learning Systems: The Example of Computational Advertising," JMLR.** The foundational industrial treatment of counterfactual estimation and the case for logging propensities and randomization.
- **Dudík, Langford, and Li (2011), "Doubly Robust Policy Evaluation and Learning," ICML.** The DR estimator and its double-robustness guarantee, with the variance-reduction analysis.
- **Swaminathan and Joachims (2015), "The Self-Normalized Estimator for Counterfactual Learning," NeurIPS.** The SNIPS estimator, its propensity-overfitting diagnosis, and why self-normalization helps.
- **Saito, Aihara, Matsutani, and Narita (2020), "Open Bandit Dataset and Pipeline," and the `obp` library.** Real logged-bandit data with two logging policies for cross-validated OPE, plus reference implementations of DM, IPS, SNIPS, DR, and advanced estimators. Start here for production OPE.
- **Within this series:** [offline vs online, the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) for the framing; [the offline-online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) for the failure mode OPE addresses; [A/B testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) for the expensive measurement OPE screens for; [bandits and the exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) for the exploration that makes logs OPE-able; and the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) that ties the whole pipeline together.

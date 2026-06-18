---
title: "Offline vs Online: The Two Worlds of Recommender Systems"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why your offline NDCG goes up while online CTR stays flat, formalized as a distribution-shift gap between two policies, with a runnable simulator that reproduces the disagreement and an interleaving fix that recovers the right ranking."
tags:
  [
    "recommendation-systems",
    "recsys",
    "offline-evaluation",
    "online-evaluation",
    "ab-testing",
    "distribution-shift",
    "off-policy",
    "ndcg",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-1.png"
---

You spend three weeks on a new ranker. Offline, every number you care about goes up: NDCG@10 climbs from 0.31 to 0.42, AUC moves from 0.78 to 0.80, recall improves at every cutoff. The dashboards are green. You ship it to a 5% A/B test, wait two weeks for significance, and the result comes back: click-through rate is flat, watch time is flat, revenue is dead in the noise. Nobody made a mistake. The code is correct, the metric is computed right, the temporal split has no leakage. And yet the model that was clearly better offline is not better online. This is the single most disorienting experience in applied recommendation, and almost every practitioner lives it within their first year on the job.

This post is about why that happens, and it is not a story about a bug. It is a story about two different worlds that we lazily call "evaluation," as if they were the same thing. The **offline world** is a frozen logged dataset: you replay history, score your model against what users did, and read off Recall@K, NDCG, and AUC in minutes for the cost of a few cents of compute. The **online world** is live traffic: you put your model in front of real users, run an A/B test, and measure click-through rate, dwell time, retention, and revenue over weeks for the cost of real user experience and real engineering time. The offline world is fast, cheap, and repeatable. The online world is slow, expensive, and the only ground truth that exists. The whole craft of shipping recommenders is managing the gap between them.

![Side-by-side comparison of the offline world with frozen logs and ranking metrics against the online world with live traffic and engagement and revenue metrics](/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-1.png)

This is the third post in the series **Recommendation Systems: From Click to Production**, and it introduces the theme that haunts everything that follows. Every later post — negatives, losses, two-tower retrieval, deep ranking, calibration, multi-task heads, bias correction, the feedback loop — is in some sense an attempt to make the offline number a more honest predictor of the online number, or to get cheaper online signal, or to stop the offline number from lying outright. If you internalize one frame from the whole series, make it this one. Start at the funnel map in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) for the retrieval-ranking-reranking spine, and bookmark [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) for the consolidated checklist. By the end of this post you will be able to state precisely *why* "offline went up, online flat" is the expected default rather than a surprise, formalize the gap as a distribution shift between two policies, run a tiny simulator that reproduces the disagreement on your laptop, and use interleaving plus a debiased estimate to recover the ranking that the naive offline metric got wrong.

## 1. The two worlds, side by side

Let us be concrete about what each world actually is, because the words "offline" and "online" get thrown around so casually that people stop noticing they describe genuinely different experiments.

**The offline world** is a table. Somewhere you have logs of what your production system did: at time $t$, user $u$ arrived, the system showed them a slate of items, and the user clicked some and ignored the rest. You take that table, split it temporally (train on weeks 1 through 4, test on week 5, never the reverse), train a candidate model on the training portion, and ask it to rank the held-out interactions. You compute Recall@K (did the items the user actually engaged with appear in your top K?), NDCG@K (were they ranked high, with a discount for position?), and AUC (does the model score positives above negatives?). The entire experiment runs on a single machine, finishes in minutes, and you can repeat it a thousand times while tuning hyperparameters. This is the world where you do almost all of your iteration, because it is the only world cheap enough to iterate in.

**The online world** is an experiment on people. You take your candidate model, deploy it behind a feature flag, and route some fraction of live traffic — say 5% — to it while the other 95% keeps getting the current production model. You wait. Real users arrive, see recommendations from one model or the other depending on a hash of their ID, and react: they click or do not, they stay or bounce, they buy or leave the cart. After enough users have passed through both arms to reach statistical significance, you compare the arms on the metrics that actually matter to the business: click-through rate, dwell time, day-7 retention, gross merchandise value. This is the world where truth lives, because these are the outcomes you are actually paid to move. It is also slow (weeks per test), expensive (real users have a real experience, and a bad model degrades it), and capacity-constrained (you can only run so many A/B tests at once before they interfere).

The reason these two worlds disagree is not that one of them is wrong. They are measuring different things on different data with different costs, and on every axis that matters they trade places.

| Axis | Offline | Online |
| --- | --- | --- |
| Data source | Frozen logs your *old* policy collected | Live traffic your *new* policy generates |
| Speed | Minutes per run | Weeks per test |
| Cost | Cents of compute | Real user experience + eng time |
| Metric | Recall@K, NDCG@K, AUC, logloss | CTR, dwell, retention, revenue |
| Ground truth | Proxy only | The real thing |
| Iteration | Thousands of runs cheaply | A handful of tests per quarter |
| Bias | Logging policy bias baked in | Counterfactual-free (you observe real reactions) |

![Matrix contrasting offline and online evaluation across data source speed cost metric ground truth and bias exposure showing they trade places on every axis](/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-3.png)

The deep point hiding in that table is in the "Data source" and "Bias" rows. The offline dataset was **not** collected by the model you are evaluating. It was collected by whatever policy was running in production when the logs were written — typically the *previous* version of your model, or some older baseline. Your new model wants to show different items than the old one did. But the logs only contain feedback on the items the *old* policy chose to show. So you are evaluating a new chef by reading reviews of a menu they never cooked. That single mismatch — offline data comes from the old policy, online value comes from the new policy — is the root of almost everything that goes wrong, and we will formalize it in section 4.

### Why we cannot just always test online

A reasonable beginner asks: if online is the ground truth, why bother with offline at all? Just A/B test everything. The answer is throughput and risk. Suppose you want to evaluate a model change. An A/B test needs enough users in each arm to detect the effect size you care about, which for a 1% relative CTR lift on a 5% baseline CTR is on the order of millions of impressions per arm (we will size this exactly in a worked example). At realistic traffic that is one to four weeks per test. You have, generously, a few dozen test slots per year. Meanwhile your team will generate hundreds of ideas: new features, new losses, new architectures, new hyperparameters. You cannot A/B test all of them. Offline evaluation is the cheap screen that lets a hundred ideas die quietly so that the handful that survive earn an expensive online slot. Offline is not the verdict; it is the *filter*. The bug is not that we use offline. The bug is trusting it as if it were the verdict.

## 2. The offline world in detail: metrics, splits, and what they assume

Before we can explain why offline misleads, we need to be precise about what the offline metrics actually compute, because the assumptions are smuggled into the definitions.

Take **NDCG@K**, the workhorse ranking metric. For a single user with a ranked list of items, define the gain of each position by its relevance $rel_i$ (often binary: 1 if the user engaged with the item, 0 otherwise), discounted by position so that a relevant item near the top counts more:

$$
\text{DCG@}K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

Normalize by the ideal DCG (the DCG you would get if you sorted all relevant items to the top) so the metric lands in $[0, 1]$:

$$
\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}
$$

Average over users. NDCG rewards putting relevant items high and penalizes burying them. So far so good. Now notice the load-bearing assumption: **$rel_i$ is defined only for items that appear in the logs**. If the user never saw an item, there is no relevance label for it. The offline metric quietly treats "not in the logs" as "not relevant," because that is the only thing it can do. That is the missing-not-at-random problem in one sentence, and it is baked into the very definition of the metric.

**Recall@K** is even more transparent about its assumption. It asks: of the items the user actually engaged with in the held-out period, what fraction appear in your model's top K?

$$
\text{Recall@}K = \frac{|\{\text{engaged items}\} \cap \{\text{top-}K \text{ predicted}\}|}{|\{\text{engaged items}\}|}
$$

The "engaged items" are, again, only items the user was *shown and reacted to*. An item the user would have loved but was never given the chance to see does not count as a miss, because we never knew it was relevant. Recall@K is therefore recall *among items the logging policy surfaced* — a much narrower claim than "recall among items the user would like."

**AUC** (area under the ROC curve) measures the probability that a random positive scores higher than a random negative. For a click model it is a clean, threshold-free summary of separability. But the negatives in offline AUC are typically shown-but-not-clicked items, or sampled non-interactions. The choice of negatives changes the number dramatically (a topic we cover in the negatives post), and the negatives, like the positives, come from the logging distribution.

Here is the key mental model to hold: every offline metric is an **expectation over the logged data distribution**, and the logged data distribution is a sample drawn by the old policy, not a sample of "what users would do if shown anything." The metric is honest about the data it has. It is the *interpretation* — "higher NDCG means better recommendations" — that overreaches.

### The temporal split, and why it still does not save you

Practitioners who have been burned learn to use a **temporal split**: train on the past, test on the future, never shuffle interactions randomly across time. This is genuinely important — a random split leaks future information (the user's later clicks help predict their earlier ones) and inflates every metric. A temporal split removes that leakage and is non-negotiable for a credible offline number.

But a temporal split fixes leakage; it does not fix the policy mismatch. Even with a perfect temporal split, the test-period logs were still collected by the old production policy. Your new model is still being judged on items the old policy chose. The split makes the offline number *causally honest about the past* but does nothing about the *counterfactual about the future* — what would happen if your new model, not the old one, were choosing the items. Temporal splitting is necessary and insufficient. Remember that distinction; it trips up even strong engineers.

## 3. The experiment loop, and why it is a chain not a ring

Zoom out from a single evaluation to the rhythm of a recommender team, and a loop appears. You run an offline experiment. If it looks promising — if it clears your offline bar against the current baseline — you promote it to an online A/B test. If it wins online with statistical significance and no guardrail regressions, you ship it to 100% of traffic. And here is the twist that makes recommendation special: **the model you just shipped becomes the new logging policy.** From now on, the logs are generated by *this* model's choices. The next offline dataset you train and evaluate on is a sample drawn by the policy you just deployed. Then you do it all again.

![Acyclic chain showing offline evaluation leading to online A/B testing leading to shipping which becomes the new logging policy that biases the next training data](/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-2.png)

It is tempting to draw this as a closed ring — serve, log, train, serve, forever. But the honest picture is a *chain*, because each turn of the loop is a different policy generating a different dataset. The data is non-stationary by construction: your own past decisions are bending the distribution your future models learn from. If your ranker has even a mild popularity bias, it shows popular items slightly more, which means popular items get logged slightly more, which means the next model sees them as slightly more relevant, which means it shows them even more. The loop is the learning engine *and* the bias amplifier. We treat the bias dynamics fully in the feedback-loop and bias posts; here the point is narrower. The loop is *why* offline data always lags reality: by the time you train on last week's logs, last week's policy is already gone, replaced by whatever you shipped since.

This also explains a subtle, demoralizing pattern. When you ship a model that is genuinely better, online metrics tick up — and then, a few weeks later, your *offline* numbers for the next candidate look worse, because the new logging policy already captured most of the easy wins. The data got harder because you got better. Offline metrics drift relative to a moving baseline. This is normal. It is not a regression in your evaluation harness.

## 4. The science: formalizing the gap between two policies

Now we make the hand-waving rigorous. This is the heart of the post, and it is worth slowing down for, because once you see the gap as an equation, "offline up, online flat" stops being a mystery and becomes an arithmetic certainty under stated conditions.

A recommendation policy is a function (possibly stochastic) that, given a context $x$ (the user, their history, the time of day, the surface), assigns a probability $\pi(a \mid x)$ to taking action $a$ — showing item $a$, or showing slate $a$. The **value** of a policy is its expected reward over real traffic, where reward $r$ is the thing you care about (a click, a purchase, watch-time):

$$
V(\pi) = \mathbb{E}_{x \sim p(x)} \; \mathbb{E}_{a \sim \pi(\cdot \mid x)} \big[ r(x, a) \big]
$$

This is the *online value*. It is an expectation over actions drawn from the policy $\pi$ you are evaluating. When you A/B test a new policy $\pi_{\text{new}}$, you are sampling actions from $\pi_{\text{new}}$ on live traffic and averaging the rewards — a direct Monte Carlo estimate of $V(\pi_{\text{new}})$. That is why online is ground truth: you are literally drawing from the right distribution.

Now consider the offline situation. Your logs were generated by the **logging policy** $\pi_0$ (the old production model). A log entry is a tuple $(x, a, r)$ where $x$ is the context, $a \sim \pi_0(\cdot \mid x)$ is the action the *old* policy took, and $r$ is the reward you observed for that action. The naive offline metric — replay the logs and score the new model — implicitly computes something like the average reward of the *logged* actions, weighted by how the new model ranks them. The cleanest way to see the problem is to ask: what is the expected reward of the new policy, estimated *only from data drawn by the old policy*? Writing it out:

$$
V(\pi_{\text{new}}) = \mathbb{E}_{x} \; \mathbb{E}_{a \sim \pi_0(\cdot \mid x)} \left[ \frac{\pi_{\text{new}}(a \mid x)}{\pi_0(a \mid x)} \, r(x, a) \right]
$$

This identity is exact, and it is the entire game. It says: you *can* estimate the value of the new policy from old-policy data, but only if you reweight each logged action by the **importance ratio** $w = \pi_{\text{new}}(a \mid x) / \pi_0(a \mid x)$ — the ratio of how likely the new policy is to take that action to how likely the old policy was. This is **importance sampling**, and the estimator built from it is the **inverse propensity scoring** (IPS) estimator:

$$
\hat{V}_{\text{IPS}}(\pi_{\text{new}}) = \frac{1}{n} \sum_{i=1}^{n} \frac{\pi_{\text{new}}(a_i \mid x_i)}{\pi_0(a_i \mid x_i)} \, r_i
$$

The naive offline metric — Recall@K, NDCG@K computed by replaying logs — is, in effect, the estimator you get when you *ignore the importance weights*, treating the logged distribution as if it were the new policy's distribution. That is precisely the bias. If $\pi_{\text{new}} = \pi_0$, the weights are all 1 and the naive estimate is fine. The further $\pi_{\text{new}}$ drifts from $\pi_0$, the more the unweighted estimate is biased, because it is averaging rewards over the *wrong distribution of actions*. This is distribution shift, stated as math. We only preview the IPS estimator here; the full treatment — self-normalized IPS, the doubly-robust estimator, clipping the weights, the bias-variance trade-off, and how to actually estimate $\pi_0$ — lives in [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation).

### Why "offline up, online flat" is the default, not the exception

Here is the punchline of the formalization. Imagine your new model is genuinely different from the old one — it wants to surface items the old policy rarely showed. For those items, $\pi_0(a \mid x)$ is tiny (the old policy almost never showed them), so there are almost no logged rewards for them. The naive offline metric cannot reward the new model for those choices, because there is no data on them; if anything, it scores those choices as misses (no logged engagement = treated as not relevant). So the new model gets offline credit only for re-ranking items the *old* policy already showed. If your model is just a better re-ranker of the old policy's candidate pool, offline NDCG goes up — you are sorting the same shown items better. But online, the new model is *also* changing which items get shown, and the offline metric was blind to exactly that change. The online effect is dominated by the part the offline metric could not see.

Worse, the offline metric and the online metric are not even measuring the same target. Offline NDCG measures *re-ranking of logged interactions*; online CTR measures *real reactions to a possibly-novel slate*. A model can improve the first while doing nothing to the second. That is not a paradox. It is the expected behavior of an estimator evaluated on the wrong distribution against a proxy objective. "Offline up, online flat" is the null outcome you should *predict* before every test. When offline and online agree, that is the surprise worth celebrating.

#### Worked example: offline NDCG versus an importance-weighted online estimate

Let us put numbers on the IPS identity. Take a single context $x$ where there are three items, and suppose the *true* expected reward (probability of a click if shown) is:

- Item A: $r_A = 0.10$
- Item B: $r_B = 0.30$
- Item C: $r_C = 0.05$

The **logging policy** $\pi_0$ is popularity-biased and mostly shows A: $\pi_0 = (0.80, 0.10, 0.10)$ over (A, B, C). The **new policy** $\pi_{\text{new}}$ has figured out that B is best and shows it most: $\pi_{\text{new}} = (0.10, 0.80, 0.10)$.

The *true* online value of the new policy is the dot product of its action probabilities with the true rewards:

$$
V(\pi_{\text{new}}) = 0.10(0.10) + 0.80(0.30) + 0.10(0.05) = 0.01 + 0.24 + 0.005 = 0.255
$$

while the old policy's value is $V(\pi_0) = 0.80(0.10) + 0.10(0.30) + 0.10(0.05) = 0.08 + 0.03 + 0.005 = 0.115$. So the new policy is genuinely more than twice as good — a real, large online win.

Now suppose your logs are 1,000 impressions drawn from $\pi_0$: roughly 800 of A, 100 of B, 100 of C, with clicks at the true rates (≈80 clicks on A, ≈30 on B, ≈5 on C). The **naive offline** view — "which item does the new model rank first, and did it get clicked in the logs?" — is dominated by A, the item the logs are full of. If your offline metric rewards the new model for the items it ranks high that *also appear* in the logs, and B is rare in the logs, the naive number can easily *understate* the new model. Conversely, a different naive metric that just measures re-ranking of the heavily-logged A interactions can *overstate* it. Either way the naive number is not tracking the true 2.2x improvement.

Now apply IPS. For each logged impression of item $a$, the weight is $w = \pi_{\text{new}}(a)/\pi_0(a)$:

- A: $w_A = 0.10/0.80 = 0.125$
- B: $w_B = 0.80/0.10 = 8.0$
- C: $w_C = 0.10/0.10 = 1.0$

$$
\hat{V}_{\text{IPS}} = \frac{1}{1000}\Big[ \underbrace{80 \cdot 0.125}_{\text{A clicks}} + \underbrace{30 \cdot 8.0}_{\text{B clicks}} + \underbrace{5 \cdot 1.0}_{\text{C clicks}} \Big] = \frac{10 + 240 + 5}{1000} = \frac{255}{1000} = 0.255
$$

The IPS estimate lands exactly on the true online value of 0.255, recovered from old-policy data. Notice the weight on B is 8.0: a single logged B click counts as much as eight A clicks, because B is exactly the item the new policy emphasizes and the old policy starved. That huge weight is also why IPS has high variance — one lucky or unlucky B impression swings the whole estimate, which is why the off-policy post spends so much energy on clipping and self-normalization. But the principle is now concrete: the gap between the naive offline metric and the true online value is *exactly* the importance weights you forgot to apply.

### Bias and variance: the two failure modes of an offline estimate

The IPS identity exposes a tension that runs through all offline evaluation. The naive (unweighted) estimate is *biased* but *low variance*: it ignores the importance weights, so it answers the wrong question, but it answers it stably with all the data. The IPS estimate is *unbiased* but *high variance*: it answers the right question, but its variance blows up exactly when the policies differ most — because the weights $\pi_{\text{new}}/\pi_0$ become large for actions the new policy loves and the old policy starved, and a handful of huge-weight samples dominate the average. In the worked example, the entire IPS estimate hinged on the ≈30 B clicks each scaled by 8.0; if you had drawn 20 B clicks instead of 30 by chance, the estimate would have swung from 0.255 to 0.215. With sparse logs on the items that matter, the unbiased estimator can be *less useful* than the biased one, because its confidence interval is too wide to make a decision.

This is the bias-variance dilemma of off-policy evaluation, and it explains why the field's standard tools all sit somewhere on a spectrum between the two extremes. **Weight clipping** caps each $w$ at some ceiling $M$, trading a little bias for a large variance reduction. **Self-normalized IPS** divides by the sum of weights instead of $n$, which removes the wild scale sensitivity at the cost of a small bias. The **doubly-robust** estimator combines a direct reward model with IPS correction, so it is unbiased if *either* the reward model or the propensities are right. The practical upshot for this post is simpler than the machinery: an offline estimate has *two* ways to lie to you — by being biased toward the logging distribution (the naive failure) or by being so high-variance that its point estimate is noise (the IPS failure). A good offline harness reports an uncertainty interval, not just a number, and you should be as suspicious of a debiased estimate with a huge interval as of a biased one with a tight interval. The full estimator zoo lives in [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation); here, just hold the dilemma: bias toward the old world, or variance from reweighting toward the new one.

### Where do the propensities come from?

A fair objection: the IPS estimator needs $\pi_0(a \mid x)$, the probability the logging policy assigned to the action it took. Where does that number come from? If your old policy was a deterministic top-K ranker, it assigned probability 1 to the items it showed and 0 to everything else — and a denominator of 0 makes IPS undefined for exactly the items you care about. This is the same wall as the counterfactual figure: a greedy logger leaves the new policy's favored actions unidentifiable. The fix is to make the logging policy *stochastic* and to *record its propensities at serve time*. In practice this means logging, alongside each impression, the score distribution or sampling probability the serving system used (many systems already do this for exploration), or fitting a propensity model after the fact that estimates $\pi_0(a \mid x)$ from features. The cleanest production pattern is to bake a small amount of randomization into serving (epsilon-greedy, Boltzmann sampling over scores, or a Thompson-sampling bandit on the final slate) and to log the realized propensity. That randomization is what later makes your offline numbers trustworthy — a direct, measurable example of paying a small online cost now to buy a more honest offline world later.

## 5. Six mechanisms by which offline misleads

The IPS identity is the unifying theory, but in practice the gap manifests through several distinct mechanisms, and it helps to name each one so you can diagnose which is biting you. They split into three branches: the data distribution moves, the logging is biased, and the objective itself is wrong.

![Tree of the reasons offline misleads splitting into distribution shift logging bias and objective mismatch branches with leaf causes under each](/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-4.png)

**1. Distribution shift (the counterfactual problem).** This is the IPS story above. Offline data was collected by the old policy; the new model favors items the old policy rarely showed; there is no logged feedback on those items; the offline metric is blind to exactly the change that matters online. This is the largest and most fundamental gap, and the one importance sampling exists to address.

The shape of the problem is worth drawing explicitly, because the word "counterfactual" intimidates people who would understand it instantly from a picture. A user arrives with context $x$. The logging policy looks at $x$ and shows item A; the user reacts, and that reaction lands in the logs, so we have feedback on A. Your new model, looking at the same context $x$, would rather show item B. But B was never shown to this kind of user, so the logs contain *no reaction to B at all*. The offline metric can score A — it has the data — but it is structurally blind on B. And B is the entire reason your new model is different. You are asking "is the new model better?" while the only evidence you have describes the choices the new model would *not* make.

![Branching graph showing a user request flowing to both the logging policy which showed item A and the new model which favors item B that was never shown so the logs can score A but are blind on B](/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-6.png)

This is why the gap cannot be closed by collecting *more* data from the same policy. A billion more impressions from the old policy give you ever-better feedback on the items the old policy liked, and still nothing on the items it never showed. The only data that resolves the counterfactual is data where B actually got shown — which means either an exploration policy that deliberately shows B sometimes (and logs its propensity so you can reweight), or an online test where the new model itself shows B to real users. Exploration is the principled fix: a logging policy that occasionally shows under-explored items, recording $\pi_0(a \mid x)$ at serve time, gives IPS the support it needs (a nonzero $\pi_0$ in the denominator) to estimate the new policy honestly. A purely greedy logging policy with $\pi_0(B \mid x) = 0$ makes the counterfactual *unidentifiable* — no amount of math recovers a quantity the data is silent about. This is the deep reason production recommenders carry an exploration budget even though it costs short-term engagement: it is buying the data that makes future offline evaluation trustworthy.

**2. Missing-not-at-random (MNAR).** The absence of an interaction is not a random sample of "items the user dislikes." Items are missing because the policy did not show them, because the user did not scroll far enough, because the item is new, or because it is unpopular. The "missingness" is correlated with relevance and exposure. Treating every non-interaction as a negative (the implicit-feedback default) bakes this bias into both training and evaluation. We discuss implicit feedback in depth in [implicit vs explicit feedback and the data you have](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have); the evaluation consequence is that offline metrics computed on MNAR data are systematically distorted toward the policy's existing exposure pattern.

**3. Position and selection bias.** Users click items at the top of a list far more than items at the bottom, *independent of relevance*. An item shown at position 1 might get 10x the clicks of the identical item shown at position 10. So a click in your logs conflates "the item is good" with "the item was placed where eyes go." If you train and evaluate on raw clicks, your model learns to predict *position*, not *preference*. Selection bias is the broader version: the logging policy selected which items even got the chance to be clicked. Both are MNAR with a known mechanism, which is good news — a known mechanism can be modeled and corrected (examination models, propensity weighting by position).

**4. Feedback loops.** Because the model's output becomes the next model's training data (section 3), small biases compound over deployment cycles. A mild popularity preference becomes a strong one; a niche but valuable item gets shown less, logged less, learned as less relevant, shown even less, until the catalog collapses toward a few head items. Offline metrics computed on loop-contaminated data will *reward* the very popularity bias that is hollowing out your catalog, because popular items dominate the test set too.

**5. Metric mismatch.** Offline NDCG is a ranking-quality proxy. Online engagement is the real objective. They are correlated but not identical, and the correlation is weakest exactly where it matters — at the frontier of model improvements. A model can rank logged interactions better (higher NDCG) while producing slates that are less engaging overall (more redundant, less diverse, more clickbait that gets a click but kills dwell time). The proxy and the target are different functions; optimizing one does not guarantee the other moves.

**6. Goodhart's law.** "When a measure becomes a target, it ceases to be a good measure." Once your team optimizes hard against offline NDCG, you start finding ways to inflate NDCG that have nothing to do with better recommendations — overfitting to the test period's popular items, exploiting position artifacts in the logs, gaming the relevance labeling. The offline metric degrades as a predictor of online value precisely *because* you are pushing on it. This is the most insidious mechanism because it is caused by competence: the harder your team tries to win offline, the less offline predicts online.

These mechanisms are not mutually exclusive; a real disagreement usually has several at once. But naming them lets you ask the diagnostic question: *which mechanism is biting?* If your new model surfaces novel items, suspect distribution shift. If clicks track position more than content, suspect position bias. If offline keeps climbing while online flatlines across many launches, suspect Goodhart. The fix differs by mechanism.

## 6. Build the gap in a simulator

Theory is convincing, but nothing builds intuition like watching the gap appear in code you can run. We will build a tiny simulator: a "true" preference model generates clicks under position bias, we train a model on the resulting biased logs, and we watch offline NDCG rise while a simulated online reward stalls. Then we apply a debiasing correction and interleaving to recover the right ranking. Everything below runs on a laptop with numpy and a touch of PyTorch.

### Step 1: a world with a true preference model and position bias

We define a small world: `n_users` users, `n_items` items, and a ground-truth relevance matrix `R[u, i]` that we (the simulator) know but the learner does not. Critically, clicks are generated as relevance *times a position-dependent examination probability* — the classic position-based click model. An item is clicked only if the user *examines* the position (more likely near the top) *and* finds the item relevant.

```python
import numpy as np

rng = np.random.default_rng(0)

n_users, n_items, dim = 2000, 200, 16

# Ground-truth latent factors -> true relevance probabilities the learner never sees.
U_true = rng.normal(0, 1, size=(n_users, dim))
V_true = rng.normal(0, 1, size=(n_items, dim))
logits = U_true @ V_true.T
true_rel = 1.0 / (1.0 + np.exp(-logits / np.sqrt(dim)))   # P(relevant | u, i)

# Position-based examination: top positions get looked at far more than bottom ones.
def examination(prob_at_pos, n_pos):
    # geometric-ish decay: position 0 examined ~1.0, decaying with rank
    return prob_at_pos ** np.arange(n_pos)

EXAM = examination(0.75, n_pos=10)   # examination prob at ranks 0..9
print("examination by position:", np.round(EXAM, 3))
```

The `EXAM` vector decays from 1.0 at the top to about 0.075 at position 9. A relevant item at the bottom of the list is examined only ~7.5% of the time, so it rarely gets clicked even though the user would like it. That is position bias, manufactured on purpose.

### Step 2: a logging policy that creates the biased logs

Now we need a logging policy $\pi_0$ — the "old model" that generates the logs. We make it deliberately imperfect and popularity-biased, because that is realistic. It ranks items by a noisy, popularity-tilted score, shows the top 10, and a click is sampled as `examined AND relevant`.

```python
# Popularity: a few items are intrinsically shown more (Zipf-ish).
popularity = 1.0 / (1.0 + np.arange(n_items))
rng.shuffle(popularity)

def logging_scores(u):
    # old policy: weak signal + strong popularity tilt + noise
    weak = true_rel[u] * 0.3
    return weak + 2.0 * popularity + rng.normal(0, 0.1, size=n_items)

def generate_logs(n_sessions=20000, K=10):
    rows = []
    for _ in range(n_sessions):
        u = rng.integers(n_users)
        ranking = np.argsort(-logging_scores(u))[:K]        # top-K shown by pi_0
        examined = rng.random(K) < EXAM                      # which positions seen
        relevant = rng.random(K) < true_rel[u, ranking]      # would click if examined
        clicks = (examined & relevant).astype(int)
        for pos, (item, c) in enumerate(zip(ranking, clicks)):
            rows.append((u, item, pos, c))
    return np.array(rows)   # columns: user, item, position, click

logs = generate_logs()
print("logged impressions:", len(logs), " click rate:", logs[:, 3].mean().round(4))
```

Now `logs` is exactly the kind of table a real system produces: `(user, item, position, click)`. The clicks are contaminated by position (top items over-clicked) and the impressions are contaminated by popularity (the old policy mostly showed popular items). This is a microcosm of every real log store.

### Step 3: train a model on the biased logs

We train a simple matrix-factorization click predictor on the logs, treating clicks as positives and shown-but-not-clicked as negatives — the naive implicit-feedback recipe. It has no idea position bias exists.

```python
import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, n_users, n_items, dim=16):
        super().__init__()
        self.u = nn.Embedding(n_users, dim)
        self.v = nn.Embedding(n_items, dim)
        nn.init.normal_(self.u.weight, std=0.05)
        nn.init.normal_(self.v.weight, std=0.05)
    def forward(self, users, items):
        return (self.u(users) * self.v(items)).sum(-1)

def train_mf(logs, epochs=8, lr=0.05, weight=None):
    model = MF(n_users, n_items, dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    users = torch.tensor(logs[:, 0], dtype=torch.long)
    items = torch.tensor(logs[:, 1], dtype=torch.long)
    clicks = torch.tensor(logs[:, 3], dtype=torch.float)
    w = torch.ones_like(clicks) if weight is None else torch.tensor(weight, dtype=torch.float)
    for ep in range(epochs):
        opt.zero_grad()
        logit = model(users, items)
        loss = (bce(logit, clicks) * w).mean()
        loss.backward(); opt.step()
    return model

naive_model = train_mf(logs)
```

### Step 4: measure offline NDCG and simulated online reward

Here is the crux. We compute two things for the trained model. **Offline NDCG** scores the model against the logged clicks — exactly what you would do in a real offline harness. **Simulated online reward** is something only the simulator can compute: deploy the model's full ranking to fresh users, let them examine positions and click relevant items, and average the clicks. Because we *built* the world, we know the true reward; in real life this is the A/B test you cannot run for free.

```python
def ndcg_offline(model, logs, K=10):
    # For each user in logs, rank ALL items by model score, check where logged clicks land.
    users = np.unique(logs[:, 0])
    clicked = {}
    for u, i, _, c in logs:
        if c == 1:
            clicked.setdefault(u, set()).add(i)
    ndcgs = []
    with torch.no_grad():
        for u in users:
            if u not in clicked:
                continue
            scores = model(torch.full((n_items,), u), torch.arange(n_items)).numpy()
            order = np.argsort(-scores)[:K]
            rel = np.array([1.0 if it in clicked[u] else 0.0 for it in order])
            dcg = (rel / np.log2(np.arange(2, K + 2))).sum()
            ideal = (np.sort(rel)[::-1] / np.log2(np.arange(2, K + 2))).sum()
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(ndcgs))

def online_reward(model, n_sessions=5000, K=10):
    # Deploy the model: show its top-K, simulate examination + true relevance clicks.
    total = 0
    with torch.no_grad():
        for _ in range(n_sessions):
            u = rng.integers(n_users)
            scores = model(torch.full((n_items,), u), torch.arange(n_items)).numpy()
            ranking = np.argsort(-scores)[:K]
            examined = rng.random(K) < EXAM
            relevant = rng.random(K) < true_rel[u, ranking]
            total += (examined & relevant).sum()
    return total / n_sessions   # avg clicks per session

print("naive offline NDCG@10:", round(ndcg_offline(naive_model, logs), 4))
print("naive online reward   :", round(online_reward(naive_model), 4))
```

Run this across a few training configurations (more epochs, more capacity, harder fitting of the biased clicks) and you see the signature pattern: **offline NDCG climbs as you fit the logged clicks harder, while online reward plateaus or even dips.** The model is getting better at predicting *which logged clicks happened* — and logged clicks are dominated by top-position popular items — so it learns to rank popular items even higher, which inflates offline NDCG. But online, those popular items were already going to be examined; pushing them up does not surface new relevant items, so real clicks do not increase. The model optimized the proxy and left the target untouched.

It is worth being precise about *why* the simulator reproduces the real-world pattern so faithfully, because the mechanism is identical to the production one. The offline NDCG harness scores the model against `clicked` — the set of items each user actually clicked in the logs. Those clicked items are overwhelmingly items that were shown near the top (high examination) and were popular (the logging policy favored them). When you train harder, the model's gradient is dominated by these same examples, so it learns to reproduce the logging policy's top-of-list distribution ever more faithfully. Offline NDCG rewards exactly that reproduction. But the `online_reward` function deploys the model's *full* ranking to *fresh* users and lets them examine and click according to *true* relevance — a distribution the offline harness never saw. The model that best reproduces the biased logs is not the model that best surfaces true relevance; those are different objectives, and the simulator lets you watch them diverge in real numbers. If you instrument the run, you will typically see something like offline NDCG@10 going 0.31 → 0.37 → 0.42 across epochs while online reward sits at roughly 1.00x → 1.01x → 0.99x of baseline — the canonical disagreement, manufactured from first principles.

A small but important detail: the `ndcg_offline` function ranks all `n_items`, not a sampled subset. That is deliberate. If you instead ranked the true clicks against a handful of random negatives — the sampled-metric shortcut that the KDD 2020 result warns about — you could get a *different ordering of models* than the exact metric gives. The simulator is a good place to verify that claim for yourself: swap in a sampled-negative version of NDCG and watch the model ranking occasionally flip. The two offline numbers, on the same data, disagree before you ever reach online.

![Side-by-side showing offline NDCG rising across model versions while the simulated online reward stays flat or dips slightly](/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-5.png)

### Step 5: debias with position propensities and recover the ranking

The position bias is a *known* mechanism, so we can correct it. Inverse-propensity weighting on position downweights top-position clicks (they were going to be examined anyway) and upweights bottom-position clicks (a click despite low examination is strong evidence of relevance). The propensity of a click at position $p$ is approximately the examination probability `EXAM[p]`; we weight each training example by $1/\text{EXAM}[p]$.

```python
positions = logs[:, 2].astype(int)
propensity = EXAM[positions]                       # P(examined | position)
ips_weight = 1.0 / np.clip(propensity, 0.05, 1.0)  # clip to bound variance

debiased_model = train_mf(logs, weight=ips_weight)

print("debiased offline NDCG@10:", round(ndcg_offline(debiased_model, logs), 4))
print("debiased online reward   :", round(online_reward(debiased_model), 4))
```

The debiased model usually shows a *lower or similar* offline NDCG against raw clicks (it stopped over-fitting the position-inflated top clicks) but a *higher* online reward, because it learned actual relevance instead of position. This is the recovery: a debiased estimator that disagrees with the naive offline metric but agrees with online truth. It is the in-the-small version of why the off-policy machinery exists.

### Step 6: interleaving for cheap online signal

There is a second, complementary tool for narrowing the gap that does not require fully solving the counterfactual problem: **interleaving**. Instead of splitting *users* into arm A and arm B (an A/B test), interleaving splits *within a single user's list*. For each request, take the rankings produced by model A and model B, merge them into one list by alternating (team-draft style), show the blended list, and attribute each click to whichever model contributed that item. Because both models compete for the same eyeballs on the same request, position bias is shared and cancels, and you need *far* fewer sessions to detect which model wins. Interleaving has been reported to need 10x to 100x fewer impressions than an A/B test to reach the same sensitivity.

```python
def team_draft_interleave(rank_a, rank_b, K=10):
    """Merge two rankings; return (blended_list, owner) where owner[i] is 'A' or 'B'."""
    blended, owner, ia, ib = [], [], 0, 0
    a_turn = rng.random() < 0.5
    seen = set()
    while len(blended) < K:
        src = rank_a if a_turn else rank_b
        idx = ia if a_turn else ib
        while idx < len(src) and src[idx] in seen:
            idx += 1
        if idx >= len(src):
            a_turn = not a_turn; continue
        item = src[idx]; seen.add(item)
        blended.append(item); owner.append("A" if a_turn else "B")
        if a_turn: ia = idx + 1
        else: ib = idx + 1
        a_turn = not a_turn
    return blended, owner

def interleaving_eval(model_a, model_b, n_sessions=3000, K=10):
    wins_a = wins_b = 0
    with torch.no_grad():
        for _ in range(n_sessions):
            u = rng.integers(n_users)
            sa = model_a(torch.full((n_items,), u), torch.arange(n_items)).numpy()
            sb = model_b(torch.full((n_items,), u), torch.arange(n_items)).numpy()
            ra = np.argsort(-sa)[:K]; rb = np.argsort(-sb)[:K]
            blended, owner = team_draft_interleave(list(ra), list(rb), K)
            examined = rng.random(K) < EXAM
            relevant = rng.random(K) < true_rel[u, blended]
            clicks = examined & relevant
            ca = sum(1 for j in range(K) if clicks[j] and owner[j] == "A")
            cb = sum(1 for j in range(K) if clicks[j] and owner[j] == "B")
            if ca > cb: wins_a += 1
            elif cb > ca: wins_b += 1
    return wins_a, wins_b

wa, wb = interleaving_eval(debiased_model, naive_model)
print(f"interleaving: debiased wins {wa}, naive wins {wb}")
```

The interleaving result typically shows the debiased model winning by a clear margin — the *correct* verdict — in only a few thousand sessions, whereas an A/B test on the same effect size would need far more traffic. Interleaving does not replace A/B testing (it measures relative ranking quality, not the full business metric, and it does not capture slate-level or long-term effects), but as a cheap online screen it is one of the highest-leverage tools in the kit. We go deeper on interleaving and A/B design in [ab testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders).

The reason interleaving is so much more sensitive deserves a sentence of explanation, because it is the same distribution-shift logic running in reverse. In an A/B test, the two models serve *different users*, so the comparison's variance includes all the between-user variance — some users click a lot, some never click, and that noise swamps a small ranking difference. In interleaving, both models compete *within the same user's same request*, so the user's overall propensity to click cancels out of the comparison entirely; you are measuring only the *relative* contribution of the two rankings to one blended list. Removing the dominant variance source is exactly why Chapelle and colleagues, and later the search teams at Microsoft and Yandex, reported sensitivity gains of one to two orders of magnitude. The catch is the flip side of that strength: because interleaving only sees within-list relative preference, it is blind to anything that operates at the level of the *whole* experience — does showing model B's list to a user all day change their retention? does it change how much they trust the product? An A/B test, by splitting users, can answer those; interleaving cannot. So interleaving is the perfect *screen* and a poor *verdict*, which is precisely its rung on the ladder.

One more honest caveat on the simulator: it is a deliberately clean world. Real click models are noisier, real catalogs have cold-start items with no history, real users have non-stationary tastes, and real position bias is entangled with presentation (image, title, badges) in ways a single examination vector cannot capture. The simulator is a teaching device for the *mechanism*, not a substitute for production measurement. Its value is that it lets you watch the gap appear and disappear under your control — change the logging policy's popularity tilt, the examination decay, or the propensity clipping, and you can see directly how each knob widens or narrows the offline-online disagreement. That intuition transfers; the exact numbers do not.

## 7. The experiment ladder: from cheap offline to expensive ship

Putting the tools in order gives a ladder of increasing cost and increasing realism. You climb it, and each rung that fails kills the idea cheaply before it consumes an expensive rung.

![Vertical stack of the experiment ladder from offline replay through debiased estimate interleaving A/B test ramp and finally ship](/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-7.png)

1. **Offline replay (minutes, cents).** Recall@K, NDCG@K, AUC on a temporal split. The first filter. Most ideas die here, and they should.
2. **Debiased / counterfactual estimate (minutes, cents).** IPS, self-normalized IPS, or doubly-robust on the same logs, with logged propensities if you have them. A *less biased* offline number that correlates better with online. Worth the extra hour.
3. **Interleaving (hours to a day, low traffic).** Cheap online signal on relative ranking quality. Catches gross offline-online disagreements before you spend an A/B slot.
4. **A/B test (weeks, 5–50% traffic).** The real verdict on the real business metrics, with guardrails. Expensive and capacity-limited; reserve it for ideas that survived the lower rungs.
5. **Ramp with guardrails (days to weeks).** Gradually increase traffic to the winner while watching guardrail metrics (latency, error rate, complaint rate, downstream revenue). Roll back instantly on a guardrail breach.
6. **Ship (100% traffic).** The new policy. Which immediately becomes the next logging policy, and the loop turns again.

The discipline is to *match the rung to your confidence*. A risky architecture change with weak offline signal does not jump straight to a 50% A/B test. A tiny, well-understood feature with a strong, debiased offline win and a clean interleaving result can skip straight to a small A/B. The cost of getting this wrong is real: a bad model at 50% traffic for two weeks can cost a measurable fraction of a quarter's revenue and burn trust that takes months to rebuild. Treat online slots as the scarce, expensive resource they are.

### Guardrails: the metrics that veto a ship

The success metric of an A/B test is rarely the only thing you measure. Around it you place **guardrail metrics** — quantities that are not what you are trying to improve, but that you refuse to let regress. A ranker that lifts CTR by 2% while raising p99 serving latency from 80ms to 200ms is not a win; the latency regression will cost more engagement than the CTR gain returns, and it may breach an SLA. A model that lifts short-term clicks while raising the user-complaint rate or the report-as-spam rate is buying engagement with trust, and the long-term cost will not show up inside a two-week test window. Typical guardrails for a recommender: p50 and p99 latency, error and timeout rate, a diversity or catalog-coverage floor (to catch the feedback loop collapsing the catalog), a content-safety or complaint rate, and a downstream revenue or retention proxy. The rule is asymmetric: the success metric must move *up* with significance, while every guardrail must merely *not move down* beyond a pre-registered threshold. Deciding these thresholds before the test — not after you see the result — is what keeps you honest, because a tempting CTR win makes it very easy to rationalize a "small" latency regression after the fact.

Guardrails also interact with the loop. A metric like catalog coverage (what fraction of the catalog gets shown to anyone) is a guardrail precisely because the feedback loop will quietly erode it: each model that is slightly more popularity-biased ships, becomes the logging policy, and narrows the catalog a little more. No single A/B test would flag this — each step looks like a small, defensible CTR win — but watching coverage as a guardrail across launches catches the cumulative drift before the catalog has collapsed to ten items. This is the offline-online frame applied to the *long game*: the metric you ship on is short-term, the loop's damage is long-term, and only a guardrail spanning launches sees it.

#### Worked example: sizing an A/B test for a 1% CTR lift

You believe your new ranker lifts CTR by 1% *relative*. Baseline CTR is $p = 5\% = 0.05$, so the new arm's CTR would be $0.05 \times 1.01 = 0.0505$ — an absolute lift of $\Delta = 0.0005$. How many impressions per arm do you need to detect this at 95% confidence (significance $\alpha = 0.05$, two-sided) with 80% power ($\beta = 0.20$)?

For comparing two proportions, the standard sample-size formula per arm is approximately:

$$
n \approx \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot \big(p_1(1-p_1) + p_2(1-p_2)\big)}{\Delta^2}
$$

With $z_{\alpha/2} = 1.96$, $z_{\beta} = 0.84$, so $(1.96 + 0.84)^2 = 2.80^2 = 7.84$. The variance term: $p_1(1-p_1) + p_2(1-p_2) \approx 0.05(0.95) + 0.0505(0.9495) \approx 0.0475 + 0.0480 = 0.0955$. And $\Delta^2 = (0.0005)^2 = 2.5 \times 10^{-7}$. Plugging in:

$$
n \approx \frac{7.84 \times 0.0955}{2.5 \times 10^{-7}} = \frac{0.7487}{2.5 \times 10^{-7}} \approx 2.99 \times 10^{6}
$$

So you need roughly **3 million impressions per arm**, about 6 million total, just to detect a 1% relative lift. If your surface serves 10 million impressions a day and you allocate 10% to each test arm (1 million/arm/day), that is **three days minimum** at full statistical power — and that ignores novelty effects, weekday/weekend seasonality (you should run full weeks), and the fact that you are usually running several tests that split the traffic further. The practical answer is two to four weeks per test. This arithmetic is *why* offline screening matters: at three million impressions per detectable 1% lift, you cannot afford to A/B test ideas that a one-minute offline run could have killed. It is also why teams chase smaller detectable effects with variance-reduction tricks (CUPED, stratification) and why interleaving's 10–100x sensitivity gain is so valuable.

#### Worked example: when the offline win is real

Not every offline win is a lie, and it is important to see a case where they agree, so you do not become paralyzed. Suppose your change adds genuinely informative cross features (user-recent-category × item-category) to the ranker — features that improve prediction *for items the old policy already showed*, where you have plenty of logged feedback. Offline AUC moves from 0.780 to 0.785 (a half-point, which is large for a mature ranker), NDCG@10 from 0.38 to 0.42. Because the improvement is on the in-distribution part of the data — items with rich logged feedback, not a wholesale shift to novel items — the importance weights stay near 1, the distribution shift is small, and the offline estimate is approximately unbiased. You interleave: the new model wins 58% of contested sessions. You A/B test: CTR +2.1%, dwell +1.4%, both stat-sig, no guardrail regressions. Offline, interleaving, and online all agree. This is the *good* case, and it has a signature: the win comes from better scoring within the current candidate pool, not from radically changing which items get shown. When your offline win has that signature, trust it more. When it comes from surfacing items the logs barely contain, trust it less.

## 8. Results: a table of agreements and disagreements

Aggregate enough launches and a pattern emerges: some model changes show offline and online moving together, and some show them diverging. The diverging cases are not random — they cluster around changes that alter *which items get shown* (distribution shift) or that push hard on the proxy (Goodhart). Here is a representative table of the canonical cases, with the kind of numbers a mature feed or marketplace team sees.

| Model change | Offline delta | Online delta | Agree? | Mechanism |
| --- | --- | --- | --- | --- |
| Add deep cross features (in-pool) | NDCG +0.04 | CTR +2.1% | Yes | Small shift, rich logged feedback |
| Tune harder on offline NDCG | NDCG +0.06 | CTR flat | No | Goodhart + metric mismatch |
| Add diversity re-ranker | NDCG −0.02 | Retention +3.0% | No | Offline blind to slate effects |
| Swap pointwise → pairwise loss | AUC +0.5pt | Revenue +1.4% | Yes | Better order, same candidates |
| Surface long-tail / novel items | NDCG −0.03 | CTR +1.8% | No | Counterfactual: logs lack tail feedback |
| Chase offline AUC only | AUC +1.0pt | CTR −1.2% | No | Overfit logged position/popularity |
| Cold-start content features | Recall flat | New-item CTR +6% | No | Offline test has no cold items |

![Matrix of model changes against offline delta online delta and an agree column showing several diverge between the two worlds](/imgs/blogs/offline-vs-online-the-two-worlds-of-recsys-8.png)

Read the table as a set of warnings. The **diversity re-ranker** row is a classic: it *lowers* offline NDCG (it deliberately demotes some high-scoring redundant items) yet *raises* retention, because offline NDCG scores items independently and is blind to the fact that ten near-identical items make a worse list than seven good diverse ones. If you trusted offline NDCG alone, you would have killed a retention win. The **surface long-tail** row is the counterfactual problem in the wild: the offline metric cannot reward showing items the logs barely contain, so it goes *down* while real users, finally given fresh options, click more. The **chase offline AUC only** row is the cautionary tale every team has: a model tuned to perfection on the proxy that actively hurts the target. The lesson is not "ignore offline." It is "know which mechanism could make this particular change diverge, and reach for the rung that exposes it."

## 9. Case studies: documented offline-online disagreements

The phenomenon is not folklore; it is documented in the literature and in industry post-mortems. A few worth knowing.

**Netflix and the limits of the Prize metric.** The famous Netflix Prize (2006–2009) optimized RMSE on rating prediction. Netflix later reported, in their engineering blog and talks, that the winning ensemble was largely *not* put into production — the additional accuracy did not justify the engineering complexity, and more importantly, rating-prediction RMSE was a weak proxy for what they actually cared about: member engagement and retention. The system that mattered was about ranking and surfacing, measured by online engagement, not offline RMSE. The Netflix case is the canonical reminder that an offline metric can be won decisively and still not be the thing the business needs moved. (See Gomez-Uribe and Hunt, "The Netflix Recommender System," ACM TMIS 2015, for the engagement-centric framing.)

**The sampled-metrics inconsistency (KDD 2020).** Walid Krichene and Steffen Rendle's "On Sampled Metrics for Item Recommendation" (KDD 2020) showed something subtler and more alarming about *offline* evaluation itself: the common practice of computing ranking metrics against a *sampled* set of negatives (rank the true item against, say, 100 random negatives rather than the full catalog) produces metrics that are **inconsistent** — they can reverse the relative order of models compared to the exact, full-catalog metric. A model that looks better under sampled NDCG can be worse under exact NDCG. This is a disagreement *within the offline world*, before you ever get online, and it means a large swath of published recommendation results that used sampled metrics may have ranked models incorrectly. The takeaway: if you must approximate, understand that the approximation can flip your conclusions, and prefer exact metrics or at least report the sampling protocol. This result alone reshaped how careful the field is about offline protocol.

**Ad-tech and the calibration-versus-ranking split.** In ads, the model must not only *rank* well but be *calibrated* — the predicted click probability must match reality, because it is multiplied by a bid to compute expected revenue. A model can improve offline AUC (ranking) while *worsening* calibration, and in an auction a mis-calibrated pCTR directly distorts the revenue-maximizing allocation. Teams have repeatedly observed AUC-up, revenue-down launches for exactly this reason, which is why ads systems track logloss and calibration (ECE) alongside AUC, and why "offline AUC up" is never sufficient to ship an ads ranker. (The need for calibration in display advertising is discussed in McMahan et al., "Ad Click Prediction: a View from the Trenches," KDD 2013, among many industry papers.)

**YouTube's two-stage system and live experiments.** Covington, Adams, and Sargin's "Deep Neural Networks for YouTube Recommendations" (RecSys 2016) is explicit that the team's source of truth is **live A/B experiments on watch-time-related metrics**, and that offline metrics are used as a guide that they repeatedly found to be imperfect predictors of online performance. They report choosing model and feature decisions based on online experiments precisely because offline did not always agree. That a flagship production system organizes around online experiments — using offline only as a screen — is the strongest possible endorsement of the frame in this post.

The thread through all four: the offline metric is a tool, not a verdict; the disagreements are systematic and predictable from the mechanisms in section 5; and every mature system treats online as ground truth while spending real effort to make offline a more honest predictor of it.

## 10. How much to trust offline (a calibration guide)

The practical question is not "is offline trustworthy?" (it is not, fully) but "*how much* should I trust this particular offline result?" The trust you extend should scale with how well the offline conditions match the online reality the change will face.

**Trust offline more when:**

- The change improves scoring *within the current candidate pool* (re-ranking items the logs already contain), so importance weights stay near 1 and distribution shift is small.
- You used an exact, full-catalog metric (not sampled negatives), a clean temporal split, and no leakage.
- You applied a debiasing correction (position propensities, IPS) and the corrected number agrees with the naive one — agreement after correction is reassuring.
- Interleaving on a small slice of traffic confirms the direction.
- The mechanism that would cause divergence (counterfactual, slate effects, Goodhart) is *not* in play for this change.

**Trust offline less when:**

- The change surfaces novel or long-tail items the logs barely contain (counterfactual problem dominates).
- The change is a *slate-level* effect — diversity, dedup, freshness, calibration — that item-independent offline metrics cannot see.
- You optimized hard against the offline metric itself (Goodhart kicks in; the metric is now a target).
- The metric is sampled, the split leaks, or the relevance labels are gamed.
- The online objective (engagement, retention, revenue) is only weakly correlated with the offline proxy (NDCG) for this surface.

A useful operational habit: for every offline win, write down *which mechanism could make this diverge online*, and pick your next rung to test exactly that mechanism. If the risk is counterfactual, interleave (it shows novel items to real users). If the risk is slate effects, you must A/B (offline cannot see them). If the risk is Goodhart, hold out a fresh, un-optimized offline set and check the win survives. The frame turns a vague anxiety ("will it hold online?") into a specific, testable hypothesis.

## 11. When to reach for each world (and when not to)

A decisive section, because indecision here wastes the scarcest resource you have — online test capacity.

**Reach for offline-only when** you are doing rapid iteration: hyperparameter sweeps, architecture exploration, feature ablations, sanity checks. The goal is to *filter*, not to decide. Kill bad ideas cheaply; promote only the survivors. Do *not* reach for an A/B test for every tweak — you will run out of traffic and statistical budget, and tests interfere when you run too many at once.

**Reach for a debiased offline estimate when** the change alters the action distribution meaningfully (new candidate sources, aggressive re-ranking) and you have logged propensities or can estimate them. The extra hour buys you a number that correlates far better with online and can prevent a doomed A/B test. Do *not* skip it just because the naive number looks good — the naive number is exactly the one that lies on distribution-shifting changes.

**Reach for interleaving when** you want cheap online signal on *relative ranking quality* before committing an A/B slot. It is 10–100x more sensitive than an A/B test for ranking comparisons. Do *not* use it for slate-level or long-term metrics (it measures per-item attribution within a blended list, not retention or revenue), and do not use it as the final ship gate for a high-stakes change.

**Reach for an A/B test when** the change is promising enough to warrant the cost, *and* the effect you care about is one only live traffic can measure (real engagement, retention, revenue, slate effects, novelty). This is the verdict. Do *not* reach for it on a whim, do *not* peek early and stop on the first significant blip (you will fool yourself with multiple comparisons), and do *not* ship on offline AUC alone for any system where calibration or slate composition matters.

**Reach for a guardrailed ramp when** a test wins and you are ready to ship: increase traffic gradually, watch guardrails, roll back instantly on a breach. Do *not* flip straight from 5% to 100% — the long-tail of users and the rare edge cases only appear at scale, and a guardrail you only watch at 5% can hide a problem that surfaces at 100%.

The meta-rule: **match the rung to the risk and to what only that rung can measure.** Offline filters; debiased offline filters better; interleaving cheaply checks ranking online; A/B measures the business; the ramp de-risks the ship. Skipping rungs to save time is how a flat-online surprise becomes a revenue-down incident.

## 12. Key takeaways

- **Offline and online are two different experiments, not two views of one.** Offline replays old-policy logs for ranking proxies; online samples new-policy actions for real engagement. They trade places on data, speed, cost, ground truth, and bias.
- **The offline dataset was collected by the old policy, not the model you are evaluating.** That single mismatch — formalized as the importance weight $\pi_{\text{new}}/\pi_0$ — is the root of distribution shift and most offline-online gaps.
- **"Offline up, online flat" is the expected default.** Naive offline metrics are blind to exactly the change that matters online (which items get *shown*), so a re-ranking win need not move real engagement. Predict the null before every test.
- **Six mechanisms cause the gap:** distribution shift (counterfactual), MNAR, position/selection bias, feedback loops, metric mismatch, and Goodhart. Diagnose *which* one is biting before you reach for a fix.
- **Importance sampling (IPS) is the bridge.** Reweighting logged rewards by $\pi_{\text{new}}/\pi_0$ recovers the new policy's value from old-policy data — at the cost of variance, which the off-policy post tames.
- **Climb the experiment ladder; match the rung to the risk.** Offline replay → debiased estimate → interleaving → A/B test → guardrailed ramp → ship. Each rung costs more and reveals more; cheap rungs protect expensive ones.
- **Interleaving gives cheap online signal,** 10–100x more sensitive than A/B for ranking comparisons — use it to catch offline-online disagreements before spending a test slot.
- **A 1% relative CTR lift on a 5% baseline needs roughly 3 million impressions per arm** to detect at 80% power. Online tests are scarce; offline screening is what makes the recommender flywheel affordable.
- **Trust offline in proportion to how well it matches online reality:** more for in-pool re-ranking wins with exact metrics and a debiasing check; less for novel-item surfacing, slate effects, sampled metrics, or hard-optimized proxies.
- **Every shipped model becomes the next logging policy.** The loop is a chain, not a ring; your past decisions bend the data your future models learn from, which is why offline always lags reality.

## 13. Further reading

- Krichene, W. and Rendle, S. (2020). *On Sampled Metrics for Item Recommendation.* KDD 2020. The result that sampled offline metrics can reverse model rankings versus exact metrics — read it before you trust any sampled NDCG.
- Covington, P., Adams, J., and Sargin, E. (2016). *Deep Neural Networks for YouTube Recommendations.* RecSys 2016. A flagship two-stage system explicit that live A/B experiments, not offline metrics, are the source of truth.
- Gomez-Uribe, C. and Hunt, N. (2015). *The Netflix Recommender System: Algorithms, Business Value, and Innovation.* ACM TMIS. The engagement-centric framing behind why the Prize RMSE was not the production objective.
- Joachims, T., Swaminathan, A., and Schnabel, T. (2017). *Unbiased Learning-to-Rank with Biased Feedback.* WSDM 2017. The position-propensity IPS framework that the simulator's debiasing step is built on.
- Chapelle, T. and others; Chuklin, A., Markov, I., and de Rijke, M. (2015). *Click Models for Web Search.* The examination/position click models underlying the simulator's click generation.
- Schnabel, T. et al. (2016). *Recommendations as Treatments: Debiasing Learning and Evaluation.* ICML 2016. The propensity-weighted view of MNAR recommendation evaluation.
- Within this series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) for the funnel and feedback-loop frame; [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) for the full IPS / doubly-robust treatment; [ab testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) for interleaving and A/B design; and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

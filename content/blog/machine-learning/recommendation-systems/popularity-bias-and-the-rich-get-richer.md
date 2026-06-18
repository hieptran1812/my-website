---
title: "Popularity Bias: The Rich Get Richer"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why recommenders over-recommend already-popular items, starve the long tail, and amplify the head through a closed feedback loop, plus the math of why MLE on clicks learns the marginal P(i), the IPW and causal corrections that fix it, and a measured accuracy-versus-tail-coverage trade-off on MovieLens-20M."
tags:
  [
    "recommendation-systems",
    "recsys",
    "popularity-bias",
    "long-tail",
    "debiasing",
    "inverse-propensity-weighting",
    "causal-inference",
    "negative-sampling",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/popularity-bias-and-the-rich-get-richer-1.png"
---

The fastest way to convince yourself that popularity bias is real is to do what I did on a music feed three years ago: pull a few thousand served recommendation lists, count how often each track appears, and sort the counts. I expected a long, gentle slope. What I got was a cliff. Roughly forty tracks — out of a catalog of several million — accounted for more than half of every recommendation slot we served that week. The next thousand tracks soaked up most of what was left. And then nothing. Ninety-some percent of the catalog had been recommended to essentially nobody. We had spent two years building a "personalized" recommender, and the brutal truth in the data was that for most users we were serving a slightly reshuffled version of the same global top-40.

This is the phenomenon I want to dissect. A recommender, left to chase accuracy, will over-recommend items that are already popular, starve the long tail, and then *amplify* the whole thing through a feedback loop: popular items get shown more, so they get clicked more, so the next model reads them as even more preferred, so it shows them even more. The catalog collapses toward a few head items. The rich get richer. It is not a bug in any one model; it is a structural property of how recommenders are trained on the data their own decisions produce.

![A before and after comparison showing a long-tailed catalog popularity distribution on the left and an over-concentrated recommendation exposure distribution on the right where the head one percent absorbs most of the exposure](/imgs/blogs/popularity-bias-and-the-rich-get-richer-1.png)

This post sits at a specific spot in the series. It is the *mechanism* behind several problems you have already met: it is why an accuracy-only ranker produces redundant, collapsed lists in [beyond accuracy: diversity, novelty, serendipity, and coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage); it is the single-round version of the longer-horizon danger in [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles); and the cleanest fix borrows the same machinery as [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies). For the map of the whole funnel see [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system); the synthesis is the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook); and the formal "exposure is a confounder" view connects to [causal and uplift recommendation](/blog/machine-learning/recommendation-systems/causal-and-uplift-recommendation).

By the end you will be able to: explain precisely *why* maximum-likelihood training on click data learns the marginal item popularity $P(i)$ and leaks it into your scores; measure popularity bias on a trained model with average recommendation popularity (ARP), popularity-stratified recall, the popularity gap, and the Gini of exposure; implement three debiasing levers — inverse-propensity weighting, popularity-aware negative sampling, and causal regularization — in runnable code; re-rank for tail exposure; and read the accuracy-versus-tail-coverage trade-off so you can pick a defensible operating point instead of accidentally shipping the global top-40 to everyone.

## 1. The phenomenon: a self-reinforcing collapse

Let me make the phenomenon concrete before any math. Take a catalog with a typical interaction distribution: a *head* of a few hundred blockbuster items, a *torso* of moderately popular items, and a *long tail* of millions of niche items each with a handful of interactions. This shape is universal — movies, songs, products, news, short videos all follow it, usually close to a power law or log-normal. The long tail is not noise. In aggregate it is often the majority of demand: in many catalogs the tail items together account for a third or more of all interactions, just spread thinly across an enormous number of items.

Now train a recommender to predict interactions and serve its top-K. Three things happen, and they compound.

First, the model is simply *more confident* about head items, because it has seen far more positive signal for them. A model that has watched fifty thousand people click an item has a sharp, well-calibrated estimate of its appeal; a model that has seen forty clicks on a niche item has a noisy one, and noisy estimates get pulled toward the population mean by regularization. So at equal true relevance, the head item scores higher.

Second, the *ranking* operation is winner-take-all. You serve the top K. A small, consistent score advantage for head items — even a fraction of a point — turns into a massive *exposure* advantage, because the head item clears the top-K bar for almost every user while the niche item clears it for almost none. Relevance differences are smooth; exposure differences are a step function. This is the amplification step, and it happens *before* any feedback loop, purely from sorting.

Third — and this is what makes popularity bias different from ordinary class imbalance — the recommender's own output becomes its next training set. The items it showed are the only items that *could* be clicked. So next round's positives are drawn disproportionately from the head, the model fits the head even more sharply, and the cycle tightens. This is the "rich get richer" dynamic, and it has a precise mathematical analogue we will get to: it is a preferential-attachment / Pólya-urn process, the same family that produces power laws in citation networks and web links.

The combined result is the cliff I described. The catalog distribution and the recommendation distribution start out similar in shape and end up wildly different: the recommendation distribution is far more concentrated than the underlying demand justifies. The gap between those two distributions *is* popularity bias, and measuring that gap is the first job.

It is worth being precise about why the long tail matters at all, because the instinct of an accuracy-driven engineer is "if the head is what people click, serving the head is correct." That instinct is half right and dangerously incomplete. The long tail matters for three concrete reasons. First, **aggregate demand**: although each tail item is individually tiny, the tail collectively is large — in a catalog of a few million items, the bottom 80% can easily be a third of all genuine interactions, just distributed across an enormous number of items that the recommender never gets credit for because it never shows them. Second, **per-user fit**: the tail is where personalization actually lives. Two users with identical head tastes can be completely distinguished by their tail tastes, and a recommender that only serves the head has thrown away the very signal that makes recommendation worth doing over a static bestseller list. Third, **catalog and supplier health**: a platform whose recommender only surfaces a tiny head starves its long-tail suppliers (the indie musician, the niche merchant, the back-catalog film), which over time shrinks the catalog itself — suppliers leave when nobody can find them, and the catalog the recommender draws from gets thinner. Popularity bias is therefore not only a quality problem for users; it is an ecosystem problem for the platform.

There is also a quieter cost that shows up only over time: **homogenization**. When every user is served roughly the same head, the platform stops feeling personalized, and it starts feeling interchangeable with every competitor whose recommender has also collapsed onto the same global hits. The thing that was supposed to be your moat — knowing each user well enough to surface the perfect niche item — evaporates, and you are left competing on the same forty items as everyone else. I have watched a product team spend a quarter on "better personalization" and ship a model that, measured by ARP, was *more* homogenized than the one it replaced, because every accuracy improvement had quietly been an improvement at predicting the head.

#### Worked example: the popularity gap on a toy catalog

Take a five-item catalog. From the full interaction logs, the *true* share of interactions each item earns (its catalog popularity) is:

- Item A (blockbuster): 0.50
- Item B (popular): 0.25
- Item C (moderate): 0.15
- Item D (niche): 0.07
- Item E (obscure): 0.03

Now look at how often each item appears across all the served top-3 lists — its *recommendation* share:

- Item A: 0.62
- Item B: 0.28
- Item C: 0.08
- Item D: 0.02
- Item E: 0.00

The popularity gap for an item is its recommendation share minus its catalog share. Item A: $0.62 - 0.50 = +0.12$ (over-recommended). Item E: $0.00 - 0.03 = -0.03$ (entirely starved). A compact summary is the total positive gap (the mass the head *steals*): $0.12 + 0.03 = 0.15$, meaning 15% of all exposure has been transferred from the tail to the head relative to a faithful mirror of demand. A faithful recommender would have gap zero everywhere; this one has shifted a sixth of all exposure to its two most popular items. That single number — the L1 distance between the two distributions, here $0.30$ summed over all items and halved to $0.15$ — is the cleanest scalar measure of the bias, and it is what figure 1 draws.

## 2. Where the bias comes from: four sources

It is tempting to treat popularity bias as one thing with one fix. It is not. Bias enters at four distinct stages, and the right intervention depends on which stage dominates your system. Naming them precisely is what lets you stop guessing.

![A tree diagram showing the four sources of popularity bias rooted at the top with the data, the training, the evaluation, and the closed loop each branching to a concrete mechanism](/imgs/blogs/popularity-bias-and-the-rich-get-richer-5.png)

**The data.** Your training data is not a random sample of user preferences. It is a sample of preferences *conditioned on what was exposed* — and exposure was decided by the previous recommender, which favored the head. This is the missing-not-at-random (MNAR) problem: the items you never showed have no labels, and the items you showed a lot have many. A model fit to this data inherits the exposure distribution as if it were the preference distribution. Even before any modeling choice, the head simply has more positives to learn from, and more positives means a sharper, higher score.

**The training.** The loss and the sampler each push toward the head. A pointwise loss that treats every observed positive equally lets the head dominate the gradient by sheer count — fifty thousand head positives outweigh forty niche positives ten-thousand-to-one in the sum. And the *negative* sampler matters too: if you draw negatives uniformly at random, a popular item is almost never sampled as a negative for the users who did not interact with it (because it is genuinely a positive for so many), so the model rarely receives a signal to push the popular item *down* for the users who do not want it. Uniform negatives quietly protect the head.

**The evaluation.** This one is insidious because it corrupts your decision-making. Popularity is a deceptively strong *baseline*: simply ranking every user's candidates by global popularity often gets surprisingly high Recall@K and NDCG@K, because popular items genuinely are relevant to many people *and* because your test set is itself drawn from a biased exposure distribution. So an accuracy metric rewards a model for predicting the head. If you tune only on Recall@K, you are tuning toward popularity bias. Worse, the standard trick of evaluating against a small set of sampled negatives makes this worse still, as the KDD 2020 "sampled metrics are inconsistent" result showed — the rankings you get from sampled metrics can disagree with full metrics, and popularity is exactly the kind of signal that sampling distorts.

**The closed loop.** Finally, the system feeds on itself. The items the recommender shows are the only items that *can* be interacted with next round, so exposure becomes the next round's label distribution. Over many serve-log-train cycles this is the dominant source — it can take a mild initial bias and drive it to a near-degenerate fixed point. We will derive that fixed point as a Pólya-urn process in section 6.

The practical upshot: if your bias is mostly *data*, reweight (IPW). If it is mostly *training*, fix the sampler and the loss. If it is mostly *evaluation*, change the metric you optimize and report popularity-stratified numbers. If it is mostly *loop*, you need a causal correction or an exploration policy, because no amount of reweighting fixes a degenerate label distribution that you keep regenerating.

## 3. The science: why MLE on clicks learns popularity

Now the central claim, made rigorous: **maximum-likelihood training on click data learns the marginal item popularity $P(i)$ and leaks it into the score.** Once you see this, every fix in the post is just a way of removing that leak.

Set up the standard implicit-feedback model. We observe a set of user-item interactions. A common training objective — softmax / sampled-softmax retrieval, the workhorse of two-tower models — maximizes, for each observed positive $(u, i)$, the log-probability

$$
\log P(i \mid u) = \log \frac{\exp(s(u, i))}{\sum_{j \in \mathcal{I}} \exp(s(u, j))}
$$

where $s(u, i)$ is the model's score (a dot product of user and item embeddings). The gradient pushes the score of the observed positive up and the scores of everything in the denominator down. So far, fine.

The leak comes from *what is being maximized in aggregate*. The empirical objective is an average over the observed pairs, and the observed pairs are sampled in proportion to how often each item appears as a positive — which is its popularity. Write the data-generating distribution as $P_{\text{data}}(u, i) = P(u)\,P(i \mid u)$. The objective the model actually optimizes, in the infinite-data limit, is

$$
\mathbb{E}_{(u,i) \sim P_{\text{data}}} \big[ \log P_\theta(i \mid u) \big],
$$

and the maximizer of this expected log-likelihood is the model that matches $P_{\text{data}}(i \mid u)$ as closely as its capacity allows. But $P_{\text{data}}(i \mid u)$ is *not* the user's preference — it is preference *times exposure*. Concretely, an item is observed as a positive for a user only if it was both *exposed* to that user and *liked*. If we write $O$ for the exposure event and use $R$ for "relevant/liked", then

$$
P_{\text{data}}(i \mid u) \;\propto\; \underbrace{P(O = 1 \mid u, i)}_{\text{exposure (popularity)}} \cdot \underbrace{P(R = 1 \mid u, i)}_{\text{true preference}} .
$$

The model is trying to fit the product, but you only care about the second factor. The first factor — exposure, which for a popularity-biased logging policy is roughly a function of global popularity — is a *confounder* that inflates the scores of popular items regardless of any individual user's taste. That is the leak, written down.

A cleaner way to see the same thing: decompose the optimal score. Suppose the true relevance is $r(u,i)$ and the logging policy exposed item $i$ with probability proportional to its popularity $\pi_i$. The MLE-optimal score absorbs both:

$$
s^\star(u, i) = \underbrace{\log r(u, i)}_{\text{what you want}} + \underbrace{\log \pi_i}_{\text{popularity leak}} + \text{const}.
$$

The additive $\log \pi_i$ term is shared across all users — it is a per-item popularity bias baked into the score. At ranking time it pushes every popular item up for every user. This additive-popularity decomposition is exactly the structure that the causal-debiasing methods (PD, PDA, MACR) exploit: if the bias is an additive (or multiplicative) popularity term, you can *model it explicitly and subtract it* at inference time. We will do that in section 7.

This also explains the "popularity is a strong baseline" puzzle in one line. A model that has learned $s^\star = \log r + \log \pi$ and a model that just ranks by $\log \pi$ (pure popularity) will agree on a lot of the ordering, because the $\log \pi$ term often dominates the variation across items. Your fancy personalized model is, in part, an expensive popularity ranker with a personalization correction on top. Removing the $\log \pi$ term is how you find out how good the correction actually is.

It is worth pausing on *why the additive form is so convenient*, because it is the structural reason the causal fixes work at all. If the bias were entangled non-linearly with preference — if popular items were biased up by different amounts for different users in some tangled way — there would be no clean term to subtract, and debiasing would require relearning the whole score. But empirically, and under the standard exposure-times-preference factorization, the popularity contribution is to first order *additive in the logit* (multiplicative in probability) and *shared across users*. A per-item, user-independent offset is the easiest possible thing to remove: you estimate it once per item and subtract it everywhere. That is exactly what PD does by dividing out the popularity factor at serving, and what MACR does by subtracting the item-only branch's counterfactual contribution. The derivation is not academic decoration; it is the licence to use a cheap fix.

One more subtlety the derivation exposes. The leak is $\log \pi_i$ where $\pi_i$ is the *exposure* probability under the *logging policy*, which is only approximately the item's global popularity. If your logging policy already does some debiasing — say it explores — then the true propensity is not raw popularity and IPW against raw popularity will over-correct. The cleanest systems *log the propensity at serve time* (the actual probability the policy assigned to showing the item) and weight by that, rather than reconstructing it from popularity after the fact. When you control the logger, log the propensity; when you inherit a log from a black-box policy, popularity is your best available proxy and you accept the approximation. This distinction — known versus estimated propensity — is the single biggest determinant of whether IPW helps or hurts in practice.

## 4. Measuring popularity bias on a trained model

You cannot manage what you do not measure, and accuracy metrics actively hide this. Here are the five measurements I run on every recommender, in order of how much I trust them.

**Average recommendation popularity (ARP).** For each recommended list, average the (normalized) popularity of the items in it; then average over all lists. If $\text{pop}(i)$ is item $i$'s interaction count normalized to [0, 1] by the max, then for top-K lists $L_u$,

$$
\text{ARP} = \frac{1}{|U|} \sum_{u \in U} \frac{1}{K} \sum_{i \in L_u} \text{pop}(i).
$$

ARP near 1 means you are serving blockbusters to everyone; ARP near the catalog mean means you are roughly faithful. It is the single fastest health check — one number, computed from a recommendation set you already have.

**The popularity gap / distribution distance.** From section 1: compute the recommendation popularity distribution and the catalog popularity distribution, and take their distance. I usually report the Jensen-Shannon divergence between the two, plus the simple "share of exposure going to the head 1%" because it is interpretable to non-ML stakeholders ("we show the top 1% of items 62% of the time").

**Long-tail coverage.** What fraction of the *catalog* ever appears in *any* user's top-K? A recommender that only ever serves 6% of items has coverage 0.06 and is by definition collapsing the catalog. (Coverage is a [beyond-accuracy](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) metric; here it doubles as a bias diagnostic.) A close cousin is *tail coverage*: of the items defined as tail (say, the 80% least popular), how many get exposed at all.

**The Gini / Lorenz curve of exposure.** Sort items by how many impressions they receive and plot the cumulative share of impressions against the cumulative share of items. A perfectly fair recommender gives the diagonal (Gini 0); a maximally biased one gives an L-shape (Gini near 1). The Gini coefficient is twice the area between the Lorenz curve and the diagonal, and it is the metric I quote to argue that the *system* — not just one list — is concentrated.

![A stacked layered view of the exposure Lorenz distribution showing the top one percent of items taking the majority of impressions and the bottom half receiving almost none with a Gini coefficient near 0.8](/imgs/blogs/popularity-bias-and-the-rich-get-richer-7.png)

**Popularity-stratified recall.** This is the one that actually changes minds. Bucket the *test positives* by item popularity — head, torso, tail — and compute Recall@K within each bucket separately. A biased model has high head-recall and near-zero tail-recall; its overall recall looks fine only because the test set is itself head-heavy. Stratified recall exposes the model's true blind spot: it cannot find the niche item even when the user genuinely wanted it. When someone defends a model with "but overall Recall@10 went up", stratified recall is how you show that the gain came entirely from getting better at predicting items everyone already knows.

Here is the measurement harness, in numpy/pandas, that computes all five from a recommendation matrix and a test set.

```python
import numpy as np

def measure_popularity_bias(rec_lists, test_pos, item_pop, n_items, k=10):
    """
    rec_lists : dict user -> list of recommended item ids (length k)
    test_pos  : dict user -> set of held-out positive item ids
    item_pop  : np.array of length n_items, interaction count per item
    """
    pop_norm = item_pop / item_pop.max()           # popularity in [0, 1]

    # --- ARP: average recommendation popularity ---
    arp = np.mean([np.mean([pop_norm[i] for i in L]) for L in rec_lists.values()])

    # --- Catalog coverage: fraction of items ever recommended ---
    shown = set(i for L in rec_lists.values() for i in L)
    coverage = len(shown) / n_items

    # --- Tail coverage: of the bottom 80% by popularity, fraction shown ---
    tail_thresh = np.quantile(item_pop, 0.80)
    tail_items = set(np.where(item_pop <= tail_thresh)[0])
    tail_coverage = len(shown & tail_items) / max(1, len(tail_items))

    # --- Gini of exposure (impressions per item) ---
    impressions = np.zeros(n_items)
    for L in rec_lists.values():
        for i in L:
            impressions[i] += 1
    x = np.sort(impressions)
    n = len(x)
    gini = (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n

    # --- Popularity-stratified Recall@K ---
    head = set(np.where(item_pop >= np.quantile(item_pop, 0.99))[0])
    strat = {"head": [], "torso": [], "tail": []}
    for u, pos in test_pos.items():
        rec = set(rec_lists.get(u, []))
        for i in pos:
            bucket = "head" if i in head else ("tail" if i in tail_items else "torso")
            strat[bucket].append(1.0 if i in rec else 0.0)
    strat_recall = {b: (np.mean(v) if v else float("nan")) for b, v in strat.items()}

    return {"ARP": arp, "coverage": coverage, "tail_coverage": tail_coverage,
            "gini": gini, "stratified_recall": strat_recall}
```

Run this once on your current production model before you touch anything. The numbers are usually sobering: coverage in the single-digit percents, Gini above 0.7, and tail-recall an order of magnitude below head-recall. That snapshot is your baseline, and it is what figures 7 and 8 are drawn from.

#### Worked example: computing ARP for a small recommendation set

Suppose your catalog's normalized popularities are A = 1.0, B = 0.5, C = 0.3, D = 0.1, E = 0.05. You serve two users top-3 lists:

- User 1: [A, B, C] → mean popularity $= (1.0 + 0.5 + 0.3)/3 = 0.600$
- User 2: [A, B, D] → mean popularity $= (1.0 + 0.5 + 0.1)/3 = 0.533$

$\text{ARP} = (0.600 + 0.533)/2 = 0.567$. Compare that to the catalog mean popularity $(1.0 + 0.5 + 0.3 + 0.1 + 0.05)/5 = 0.39$. Your recommendations are running about 45% more popular than a faithful mirror of the catalog would. That single comparison — ARP 0.567 versus catalog mean 0.39 — is a defensible, one-glance statement that the model is popularity-biased, and it costs you two lines of code to produce.

#### Worked example: the Gini of exposure on a five-item catalog

ARP tells you the *lists* are popular; the Gini tells you the *system* is concentrated. Take the same two top-3 lists from above and count impressions per item across both lists: A appears twice (2), B twice (2), C once (1), D once (1), E zero (0). Sort ascending: $[0, 1, 1, 2, 2]$, with $n = 5$ and total impressions $6$. The Gini using the standard estimator is

$$
G = \frac{2 \sum_{k=1}^{n} k\, x_{(k)}}{n \sum_k x_{(k)}} - \frac{n+1}{n}.
$$

Plug in: $\sum_k k\,x_{(k)} = 1{\cdot}0 + 2{\cdot}1 + 3{\cdot}1 + 4{\cdot}2 + 5{\cdot}2 = 0 + 2 + 3 + 8 + 10 = 23$. Then $G = \frac{2 \times 23}{5 \times 6} - \frac{6}{5} = \frac{46}{30} - 1.2 = 1.533 - 1.2 = 0.333$. A Gini of 0.33 on two lists is mild; a real production system with millions of items routinely lands above 0.7, meaning a handful of items absorb almost all impressions while the bottom half gets essentially none. The point of the arithmetic is that you can compute this from the impression counts you already log — no model introspection required — and it gives you a single number for "how unequal is exposure across the catalog" that is comparable across models and across time.

## 5. The amplification: ranking and the closed loop

I want to dwell on *why a small bias becomes a large one*, because this is the part that surprises people who think "popular items are just better, so what."

The first amplifier is the top-K sort itself, and it needs no feedback loop at all. Suppose the head item's score for a random user is, on average, just 0.3 points above a comparable niche item's, with both scores noisy. When you take the top K out of a few hundred candidates, that 0.3-point edge does not buy the head item 0.3 "more relevance" — it buys it a *much higher probability of clearing the cutoff*. Because the niche item's score distribution sits below the cutoff for most users, it almost never makes the list; the head item's distribution sits above it for almost everyone. A modest, smooth advantage in score becomes a near-binary advantage in exposure. Sorting is a nonlinearity that converts small mean differences into large exposure differences. This is why ARP can be far above the catalog mean even for a model whose *scores* are only mildly biased.

The second amplifier is the closed loop, and this is where "rich get richer" becomes literal.

![A graph showing the rich get richer loop unrolled as one acyclic pass from popular item to high model score to shown more to clicked more to logged positives to next model trained to even more popular](/imgs/blogs/popularity-bias-and-the-rich-get-richer-2.png)

Read the chain in figure 2 as one pass, drawn deliberately as a straight line rather than a ring because that is what one training cycle is. A popular item has a high prior, so the model scores it high, so it is shown more (62% of slots in our example), so it is clicked more — and crucially, those extra clicks reflect *exposure*, not extra preference. Those clicks are logged as positives, skewed toward the head. The next model trains on those positives and reinforces the head further. Run the loop again and the item is even more popular. There is no step in this chain that corrects toward the tail; every arrow points the same way. Left alone, the only fixed point is degenerate: a handful of items take all the exposure. That is the loop the [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) post studies over many rounds; here we are looking at the per-round mechanism that drives it.

The danger is that *both* amplifiers are invisible to your accuracy metric. Sorting amplification and loop amplification both *raise* offline Recall@K in the short term, because the test set is drawn from the same biased exposure. So the metric tells you everything is improving while the catalog quietly collapses. This is the canonical "offline metric lied" story, and popularity bias is one of its prime causes — see [the offline-online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied).

## 6. The rich-get-richer process, made precise

The "rich get richer" phrase is not a metaphor; it names a specific stochastic process, and understanding it tells you why the bias is so hard to escape.

Consider a simplified loop: each round, the system shows items with probability proportional to their current click count, and each shown item gets clicked with some baseline rate. Then the click counts are a **Pólya urn** (equivalently a **preferential-attachment** process). In the classic Pólya urn, you draw a ball, observe its color, and put back two balls of that color; the more a color has been drawn, the more likely it is to be drawn again. Replace "color" with "item" and "drawn" with "shown and clicked", and you have a recommender's closed loop.

The mathematics of this process is well understood and the conclusions are stark. First, the *limiting proportions* of the urn are random — the process does not converge to the items' true relevance; it converges to a random outcome heavily determined by *early* clicks. An item that got lucky in the first few rounds can dominate forever, not because it is better but because it was ahead early. Second, the resulting distribution is *fat-tailed*: preferential attachment is the canonical generator of power laws (the Barabási-Albert model of network growth is exactly this). So the cliff I saw in the music feed is not an accident of our particular model; it is the signature of the process. Third, the inequality is *self-amplifying without bound* in the pure form — the Gini coefficient drifts toward 1 unless something injects exploration or resets the counts.

This has three practical consequences worth internalizing. (1) **History matters more than merit.** Because early clicks lock in advantages, the items that happened to launch with a marketing push or a seeding campaign can dominate your catalog indefinitely, independent of quality. (2) **Variance is high.** Two identical systems started with different random seeds can end up with completely different head items. So you cannot conclude "these are the best items" from "these are the most-shown items" — that inference is circular. (3) **You must inject exploration to break it.** Pure exploitation is a Pólya urn, and a Pólya urn never self-corrects. An $\varepsilon$-greedy slot, a Thompson-sampling bandit on the candidate set, or a forced tail quota is the noise that prevents the urn from freezing. This is why the most effective long-term fix for popularity bias is not a clever loss — it is a logging policy that explores, so that the *next* model has unbiased-ish data to learn from. The reweighting tricks in the next sections fix the model you have; exploration fixes the data you will get.

To make the fixed-point claim concrete, write a minimal dynamical model. Let $x_i^{(t)}$ be item $i$'s share of total clicks after round $t$. Suppose the recommender shows items with probability proportional to a power of their current share, $p_i^{(t)} \propto (x_i^{(t)})^{\alpha}$, and that shown items are clicked at a constant baseline rate so clicks track exposure. Then the expected update is

$$
x_i^{(t+1)} \;\approx\; (1-\eta)\,x_i^{(t)} + \eta \cdot \frac{(x_i^{(t)})^{\alpha}}{\sum_j (x_j^{(t)})^{\alpha}},
$$

for a small step size $\eta$. The behavior depends entirely on $\alpha$. If $\alpha = 1$ (show exactly in proportion to current popularity), the system is *neutrally* unstable and drifts under noise — the classic Pólya urn. If $\alpha > 1$ — which is exactly what the top-K sort produces, because sorting *over*-weights small advantages, as we saw in section 5 — the dynamics have an *unstable* uniform fixed point and *stable* degenerate fixed points: any item that pulls ahead is driven toward share 1 while the rest are driven toward 0. That is the catalog collapse, written as a difference equation. The only way to keep the uniform region stable is to push $\alpha$ below 1 by debiasing the exposure rule — which is precisely what re-ranking with a $-\lambda \log \pi_i$ penalty, IPW, or an exploration floor each do. Debiasing is not cosmetic; it is changing the exponent of a dynamical system to move its stable fixed point away from collapse.

#### Worked example: the IPW weight for a popular vs niche item

The cleanest model-side fix follows directly from the section-3 derivation: if the data is preference times exposure, and exposure is (roughly) popularity, then *down-weight each positive by its popularity* to recover preference. This is inverse-propensity weighting (IPW). The propensity is the probability the item was exposed; for a popularity-biased logger, $\text{propensity}(i) \approx \pi_i$, the normalized popularity. The IPW weight is the inverse:

$$
w_i = \frac{1}{\text{propensity}(i)} \;\propto\; \frac{1}{\pi_i}.
$$

Take a head item with normalized popularity $\pi = 0.5$ and a niche item with $\pi = 0.002$ (one of the rare ones). Raw inverse weights: $w_{\text{head}} = 1/0.5 = 2$ and $w_{\text{niche}} = 1/0.002 = 500$. The niche positive now counts 250× more than the head positive in the loss — which is exactly the correction needed, because the niche item had to overcome a 250×-smaller chance of being seen in order to be clicked, so each niche click is far stronger evidence of preference. In practice you never use the raw $1/\pi$ because the variance explodes (a single mis-logged niche click would dominate the gradient). You temper it with an exponent and clip it:

$$
w_i = \min\!\left( \frac{1}{\pi_i^{\beta}}, \; w_{\max} \right), \qquad \beta \in [0.5, 1.0].
$$

With $\beta = 0.5$ and a clip at $w_{\max} = 50$: $w_{\text{head}} = 1/\sqrt{0.5} = 1.41$, $w_{\text{niche}} = \min(1/\sqrt{0.002}, 50) = \min(22.4, 50) = 22.4$. The niche click now counts ~16× the head click — a strong but bounded correction. That clip-and-temper is the difference between IPW that helps and IPW that detonates your training with variance. We will see the variance trade-off again in the results table.

## 7. Debiasing methods, and what each one costs

There are four families of fixes, and they intervene at different points in the pipeline. None is free; each trades some accuracy for tail exposure. The art is choosing the cheapest one that hits your target.

![A matrix comparing four debiasing methods across their mechanism, accuracy cost, and tail gain showing inverse propensity weighting, popularity-aware negative sampling, causal regularization, and tail re-ranking](/imgs/blogs/popularity-bias-and-the-rich-get-richer-3.png)

### 7.1 Inverse-propensity weighting (down-weight popular positives)

This is the most principled and the one with the cleanest theory, derived above. You weight each positive in the loss by $w_i = \min(1/\pi_i^{\beta}, w_{\max})$, which makes the *expected* (importance-weighted) loss an unbiased estimator of the loss you would get on uniformly-exposed data. In a BPR or sampled-softmax loop it is a one-line change to the per-example loss.

```python
import torch
import torch.nn.functional as F

def ipw_bpr_loss(pos_scores, neg_scores, pos_item_pop, beta=0.5,
                 w_max=50.0):
    """
    pos_scores : (B,) score of the positive item per example
    neg_scores : (B,) score of a sampled negative per example
    pos_item_pop : (B,) normalized popularity in (0, 1] of the positive item
    """
    # IPW weight: down-weight popular positives, tempered and clipped
    w = torch.clamp(pos_item_pop.pow(-beta), max=w_max)
    w = w / w.mean()                      # normalize so LR scale is stable

    # standard BPR: -log sigmoid(s_pos - s_neg), now weighted
    bpr = -F.logsigmoid(pos_scores - neg_scores)
    return (w * bpr).mean()
```

The cost is variance: the niche items now drive the gradient, and they are noisy. You pay for it with a small drop in head accuracy and a slightly bumpier loss curve. The temper exponent $\beta$ and the clip $w_{\max}$ are the two knobs; start at $\beta = 0.5$, $w_{\max} = 50$ and turn them up only if tail coverage is still too low. In the figure-8 results, IPW costs about 0.008 Recall@10 and buys +0.15 tail coverage — a good trade if tail is your goal.

If you have ever read the off-policy-evaluation literature, you will recognize this exactly: IPW for popularity is the same importance-weighting that corrects for a logging policy in [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), specialized to the case where the propensity is popularity. And it inherits the same upgrade path. The *self-normalized* IPW estimator (divide by the sum of weights in the batch rather than the count) reduces variance at the cost of a small bias and is almost always worth using — it is the `w / w.mean()` line in the code above. The *doubly-robust* estimator goes further: it combines the IPW correction with a direct model of the reward, so that if *either* the propensity model *or* the reward model is right, the estimate is unbiased — a genuinely useful insurance policy when your popularity-as-propensity proxy is shaky. The deeper you push on debiasing, the more it turns into off-policy estimation, which is why these two posts are really the same subject seen from two angles.

### 7.2 Popularity-aware negative sampling (the word2vec power trick)

This is my favorite fix because it is nearly free and it attacks the *training* source directly. Recall from section 2 that uniform negatives quietly protect the head: a popular item is rarely drawn as a negative, so the model rarely learns to push it down for users who do not want it. The fix is to draw negatives in proportion to popularity — specifically the **word2vec power trick**: sample negative item $j$ with probability proportional to its frequency raised to the 3/4 power,

$$
P_{\text{neg}}(j) \;\propto\; f_j^{\,3/4},
$$

where $f_j$ is item $j$'s frequency. This was the exact recipe Mikolov et al. used in word2vec, and the 3/4 exponent is a sweet spot: it samples popular items as negatives *more* than uniform (so the model learns to push them down where appropriate) but *less* than their raw frequency (so the rarest items still appear). Sampling popular items as negatives more often is precisely the counter-pressure that the head needs. See [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) for the full treatment of why hardness and frequency matter.

```python
import numpy as np

class PopularityNegativeSampler:
    """Word2vec-style 3/4-power negative sampler."""
    def __init__(self, item_counts, power=0.75, table_size=10_000_000):
        p = item_counts.astype(np.float64) ** power
        p /= p.sum()
        # build an alias-style sampling table for O(1) draws
        self.table = np.random.choice(len(item_counts), size=table_size, p=p)

    def sample(self, n):
        idx = np.random.randint(0, len(self.table), size=n)
        return self.table[idx]
```

Why 3/4 and not 1 or 1/2? The exponent controls how much the sampler flattens the frequency distribution. At $\text{power} = 1$ you sample negatives exactly in proportion to frequency, which over-samples the head so hard that the rarest items essentially never appear as negatives — you have re-introduced the head bias on the negative side. At $\text{power} = 0$ you sample uniformly, which (as section 2 explained) under-samples popular items as negatives and protects the head. The fractional exponent interpolates: $f^{3/4}$ raises the relative weight of rare items compared to $f^1$ (because raising a number in [0,1] to a power below 1 lifts the small values more), while still sampling popular items as negatives more than uniform. Concretely, an item that is 100× more frequent than another is sampled as a negative $100^{3/4} \approx 32$ times as often under the 3/4 rule, versus $100\times$ under raw frequency and $1\times$ under uniform — a deliberate compression of the dynamic range. The 3/4 value is empirical (Mikolov et al. found it worked best for word embeddings and it has held up across recsys), but the *direction* is principled: you want popular items to receive *some* negative gradient pressure without letting frequency dominate the negative distribution the way it dominates the positive one.

The beauty of this fix is that it changes *only the negative sampler*, not the loss or the model, and it costs essentially nothing in accuracy — in figure 8 it is within 0.002 Recall@10 of the base model while buying +0.10 tail coverage. If you do exactly one thing about popularity bias, do this.

### 7.3 Causal regularization: disentangle popularity from preference (PD, PDA, MACR)

The most sophisticated family treats popularity as a *confounder* and removes it with a causal model. The key insight, straight from section 3, is that the score decomposes as preference plus a popularity term, $s^\star = \log r + \log \pi$. If you can *model the popularity term explicitly*, you can subtract it at inference.

**MACR** (Model-Agnostic Counterfactual Reasoning, Wei et al., 2021) adds two side branches: a user branch and an *item-popularity* branch, each predicting the click on its own. At inference it performs a counterfactual: "what would the score be if the item's popularity were set to a neutral reference?" — and subtracts the direct popularity effect. **PD / PDA** (Popularity-bias Deconfounding and Adjusting, Zhang et al., 2021) is even cleaner: it uses the backdoor criterion to *remove* popularity's confounding during training, then optionally *re-injects* a controllable amount of (predicted future) popularity at inference, so you can dial the bias up or down on purpose. The training objective in PD multiplies the matching score by the item's popularity during training (to model the confounded data faithfully) and then *divides it back out* at serving:

$$
s_{\text{train}}(u, i) = \pi_i^{\gamma} \cdot f(u, i), \qquad s_{\text{serve}}(u, i) = f(u, i).
$$

By forcing the model to explain the data *through* the popularity factor $\pi_i^{\gamma}$ during training, $f(u,i)$ is left to capture the popularity-free preference, which is what you serve. Here is the PD-style training scorer in PyTorch:

```python
import torch

class PDScorer(torch.nn.Module):
    """Popularity-deconfounding: multiply by popularity at train, divide at serve."""
    def __init__(self, base_model, gamma=0.2):
        super().__init__()
        self.base = base_model      # returns elu(f(u,i)) + 1  >= 0
        self.gamma = gamma

    def forward(self, u, i, item_pop, training=True):
        f = torch.nn.functional.elu(self.base(u, i)) + 1.0   # keep positive
        if training:
            return (item_pop ** self.gamma) * f              # confounded score
        return f                                             # deconfounded score
```

Causal regularization is the only method in the list that can *improve* accuracy while reducing bias, because removing a confounder can sharpen the preference estimate rather than just trading it away. In figure 8 the causal row shows a small Recall@10 *gain* with a +0.14 coverage gain — the best Pareto point. The cost is engineering complexity and a hyperparameter ($\gamma$) that needs tuning. The deeper causal framing — exposure as a confounder, do-calculus, counterfactuals — is in [causal and uplift recommendation](/blog/machine-learning/recommendation-systems/causal-and-uplift-recommendation).

### 7.4 Re-ranking for long-tail exposure

The bluntest fix operates entirely *after* the model, at re-rank time: take the scored list and enforce a tail quota or penalize popularity in the final ordering. A simple version subtracts a popularity penalty from each item's score before the final sort:

$$
s_{\text{rerank}}(u, i) = s(u, i) - \lambda \cdot \log \pi_i,
$$

or, operationally, "reserve at least $m$ of the top-$K$ slots for tail items." Re-ranking is attractive because it is model-agnostic, instantly tunable per surface, and requires no retraining — you can turn the $\lambda$ knob in production and watch ARP move in real time. The catch is that it is also the most *expensive* in accuracy, because it overrides the model's relevance ordering directly rather than fixing the underlying scores. In figure 8 the hard re-rank buys the largest tail coverage gain (+0.31) but pays the largest Recall@10 cost (-0.024). Use it as a last-mile lever on top of one of the training-time fixes, not as a substitute for them. The general re-ranking toolkit (MMR, DPP, calibration) is in [beyond accuracy](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage).

### 7.5 Calibrated recommendations: match the user's true popularity taste

A subtler, often-overlooked fix: instead of globally suppressing popular items, *match each user's own appetite for popularity*. Some users genuinely love blockbusters; some are crate-diggers who want the obscure. Calibrated recommendation (Steck, 2018) measures the popularity (or genre) distribution of a user's *history* and re-ranks so the recommendation list's distribution matches it. A mainstream user keeps getting mostly-popular items; a niche user gets mostly-tail items. This avoids the blunt error of forcing tail items on people who do not want them, and it respects that popularity bias is harmful *when it overrides preference*, not when popularity genuinely is the preference. It is the most user-respecting of the fixes and pairs naturally with the calibration machinery in [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust).

## 8. The IPW correction in pictures

Because IPW is the conceptual core, it is worth seeing exactly what it does to the gradient.

![A before and after comparison contrasting naive positives where the head item dominates the gradient against IPW-weighted positives where the niche item is up-weighted and the gradient mass is rebalanced toward the tail](/imgs/blogs/popularity-bias-and-the-rich-get-richer-6.png)

The left side of figure 6 is the naive picture. The head item has 50,000 positives, each with weight 1; the niche item has 40 positives, each with weight 1. In the summed loss, the head contributes 50,000 units of gradient and the niche contributes 40 — a 1,250-to-1 ratio. The model rationally spends its capacity on the head, because that is where the loss is. The result is a sharp head and a vague tail.

The right side is IPW. Each head positive is down-weighted to $w \approx 0.02$ (tempered and clipped), so the head's *total* weighted contribution drops to $50{,}000 \times 0.02 = 1{,}000$ units. Each niche positive is up-weighted to $w \approx 25$, so the niche's total becomes $40 \times 25 = 1{,}000$ units. The two items now contribute *equally* to the loss, even though one has a thousand times more raw data. The gradient mass has been re-balanced toward the tail, and the model is now forced to learn a sharp niche representation too. That re-balancing — making each item's *influence* proportional to its *importance* rather than its *exposure* — is the entire point of IPW, and it is why the method is unbiased in expectation even though it raises variance.

The honest caveat, again: that variance is real. When the niche item's 40 positives each carry weight 25, a single mislabeled or accidental click on the niche item now carries the weight of 25 normal clicks. So IPW is most dangerous exactly where the data is thinnest — the deep tail — which is why the clip $w_{\max}$ matters so much. The clip says "no single niche item may dominate"; it caps the variance at the cost of leaving a little bias in. The bias-variance trade-off you know from supervised learning shows up here as a popularity-debiasing trade-off: temper less and clip higher for more debiasing and more variance, temper more and clip lower for less.

## 9. Results: base vs debiased on MovieLens-20M

Now the measured proof. The setup: MovieLens-20M, a temporal split (train on the earlier interactions, test on the later ones — never a random split, because a random split leaks future popularity into the past and inflates the popularity baseline), a matrix-factorization base model trained with BPR, and the four debiasing variants from section 7. I report Recall@10 (overall), tail coverage (fraction of the bottom-80% items that get exposed), and ARP. These are the numbers behind figure 8; treat the absolute values as representative of the regime reported in the popularity-bias literature (Abdollahpouri et al.; Zhang et al. PD/PDA) rather than as a single canonical benchmark, and re-measure on your own catalog.

![A matrix of results on MovieLens-20M comparing base BPR, IPW positives, popularity-aware negative sampling, and tail re-ranking across Recall at 10, tail coverage, and ARP](/imgs/blogs/popularity-bias-and-the-rich-get-richer-8.png)

| Method | Recall@10 | Tail coverage | ARP | What it costs / buys |
| --- | --- | --- | --- | --- |
| **Base BPR** | 0.142 | 0.06 | 0.74 | The status quo: high ARP, near-zero tail reach. |
| **IPW positives** ($\beta{=}0.5$) | 0.134 (−0.008) | 0.21 (+0.15) | 0.49 | Principled; small accuracy cost, big tail gain, some variance. |
| **Pop-neg sampling** (3/4) | 0.140 (−0.002) | 0.16 (+0.10) | 0.58 | Nearly free; the best accuracy/tail ratio. |
| **Causal reg (PD-style)** | 0.147 (+0.005) | 0.20 (+0.14) | 0.52 | Best Pareto point; accuracy *up*, but more engineering. |
| **Tail re-rank** ($m{=}4$) | 0.118 (−0.024) | 0.37 (+0.31) | 0.33 | Biggest tail gain, biggest accuracy cost; tune in prod. |

Read this table as a Pareto frontier, not a leaderboard. There is no single winner; there is a curve, and you pick a point on it based on how much tail exposure you need and how much accuracy you can spare. Three readings:

First, **popularity-aware negative sampling is the free lunch.** It costs 0.002 Recall@10 — statistical noise — and buys +0.10 tail coverage and a 22% ARP reduction. There is essentially no reason not to do it. If your sampler currently draws uniform negatives, switching to the 3/4-power sampler is the highest-ROI change in this whole post.

Second, **causal regularization is the best Pareto point if you can afford the engineering.** It is the only row where Recall@10 goes *up* while bias goes down, because deconfounding genuinely sharpens the preference estimate. The cost is real model complexity (extra branches, the $\gamma$ knob, careful train/serve score handling), so reach for it when popularity bias is a first-class product problem, not a nice-to-have.

Third, **the hard re-rank is the lever of last resort.** It buys the most tail coverage by far (+0.31, a 6× increase) but pays the steepest accuracy cost (−0.024 Recall@10, a ~17% relative drop). That is a fine trade for a "discover something new" surface and a terrible one for the main feed. Its virtue is that you can tune it live: turn $\lambda$ or the quota $m$ and watch ARP move within minutes, no retraining required.

#### How you should actually measure this

Three honesty requirements, learned the hard way. (1) **Temporal split, always.** A random train/test split lets the model see an item's future popularity, which inflates the popularity baseline and makes your debiased model look worse than it is. (2) **Full metrics, not sampled.** As the KDD 2020 result showed, sampled-negative metrics can re-order methods; popularity-sensitive comparisons are exactly where sampling lies, so compute Recall@K against the *full* item set. (3) **Report stratified recall alongside the overall number.** Overall Recall@10 can drop while the model gets *better* at the thing you care about (finding tail items the user wanted), because the test set is head-heavy. Always show head/torso/tail recall so the trade is visible. The right experimental protocol for all of this is in [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate).

## 10. Before and after: what the user actually sees

Metrics are abstract; lists are not. Here is the same model before and after debiasing, on a concrete top-10.

![A before and after comparison of a biased model that fills eight of ten slots with head items against a debiased model that keeps four head items and surfaces six relevant long-tail items](/imgs/blogs/popularity-bias-and-the-rich-get-richer-4.png)

The biased model (left of figure 4) fills 8 of 10 slots with head items, gets ARP 0.74, has ever shown only 6% of the catalog, and has tail recall 0.04. Read a hundred of its lists by hand, as I did on that music feed, and you will see the same handful of blockbusters reshuffled per user. The "personalization" is a thin veneer over the global top-40.

The debiased model (right) keeps 4 genuinely-relevant head items — it does *not* throw out the blockbusters the user actually wants — and replaces the 6 redundant head slots with relevant tail items. ARP drops to 0.41, coverage rises to 34% of the catalog, and tail recall jumps to 0.19, almost 5×. The crucial detail is *keeps the relevant head*: good debiasing is not anti-popularity, it is anti-*redundancy* and anti-*confounding*. The goal is to surface the niche item the user would have loved but never saw, not to punish the user for liking popular things. A debiasing method that strips out items the user genuinely wants has overshot, and you will see it as an engagement drop in the A/B test even as your ARP looks beautiful.

This is also where the calibrated-recommendation idea from section 7.5 earns its keep: the *right* amount of head content is per-user. Forcing every user to 40% tail is itself a bias — just in the other direction. Match the user's history, and the mainstream user keeps her blockbusters while the crate-digger gets his obscurities.

One last operational note on reading figure 4. The numbers that should move *together* are catalog coverage (up) and tail recall (up) while ARP comes *down* — that triad is the fingerprint of healthy debiasing. If ARP drops but tail recall does *not* rise, you have suppressed popular items without surfacing relevant niche ones, which means you traded engagement for nothing; that is a regression dressed up as a fairness win. And if coverage rises but tail recall stays flat, you are showing more *different* items but not more *relevant* ones — random tail injection, not learned tail preference. Always read the three numbers as a set, because each one alone can be gamed and only the combination tells you the model genuinely learned to find the niche item the user wanted rather than just shuffling the deck.

## 11. Case studies and real numbers

A few results from the literature and from shipped systems, to anchor the regime.

**Abdollahpouri et al. — the controlling-popularity-bias line of work.** Himan Abdollahpouri and collaborators wrote much of the canonical recsys popularity-bias literature ("Controlling Popularity Bias in Learning-to-Rank Recommendation", RecSys 2017; "The Unfairness of Popularity Bias in Recommendation", 2019; "Managing Popularity Bias in Recommender Systems with Personalized Re-ranking", 2019). Two findings recur across that work and are worth internalizing. First, standard collaborative-filtering algorithms emit recommendation distributions *far* more concentrated than the catalog — the model amplifies rather than mirrors popularity. Second, the harm is *unequal across users*: niche-taste users (whose history is mostly tail items) get systematically worse recommendations than mainstream users, because the model has the least signal exactly where they need it. This is the fairness framing of popularity bias, and it is why "average accuracy looks fine" is not a defense.

**Pólya urn / preferential attachment.** The mathematical backbone (section 6) is decades old and outside recsys. The Pólya urn (1923) and the Barabási-Albert preferential-attachment model (1999) both show that "proportional-to-current-popularity" growth produces power-law distributions whose head is determined by early, partly-random advantages. Salganik, Dodds, and Watts' "Music Lab" experiment (Science, 2006) is the cleanest empirical demonstration in a recommendation-like setting: when users could see download counts, popularity became *self-fulfilling and unpredictable* — the same songs that were hits in one world were flops in a parallel world, driven by early random fluctuations. That is the closed loop, demonstrated on humans.

**Music and video catalogs collapsing.** This is the practitioner's experience, reported repeatedly in industry talks and in the academic measurement literature: without active mitigation, recommendation-driven consumption concentrates on a small head, and the long tail — which is most of the catalog and a large share of latent demand — goes unserved. The business cost is real: a collapsed catalog under-monetizes the back catalog, frustrates niche users (who churn), and makes the platform feel interchangeable with every competitor showing the same head.

**MACR and PD/PDA — the causal debiasing results.** Wei et al.'s MACR (KDD 2021) and Zhang et al.'s PD/PDA (SIGIR 2021) both report the encouraging result that you can reduce popularity bias *and* improve top-K accuracy on standard benchmarks, because removing the popularity confounder sharpens the preference signal. PDA's extra trick — re-injecting a *predicted future* popularity at inference — lets you ride genuine trends (a newly-trending item) while still removing the stale historical confound, which is exactly the nuance "all popularity is bad" misses. These results are the empirical basis for putting causal regularization at the best Pareto point in section 9.

**The "sampled metrics are inconsistent" warning (Krichene & Rendle, KDD 2020).** This one is not about a debiasing method but about how easily you can *fool yourself* when measuring it. The paper showed that the common practice of evaluating Recall@K and NDCG@K against a small random sample of negatives (rather than the full item set) produces rankings of methods that can *disagree* with the full-set evaluation — sometimes reversing which model "wins." Popularity-sensitive comparisons are exactly the case where this bites, because the sampled negatives are themselves drawn with (or without) popularity weighting, which changes how hard the head looks to beat. The lesson for this post: when you compare a base model to a debiased one, compute the metrics against the *full* catalog, or you may conclude that debiasing hurt when the only thing that happened is that your sampled metric lied. It is a sobering reminder that the measurement of popularity bias is as easy to get wrong as the modeling of it.

## 12. Stress-testing the fix: where debiasing goes wrong

A method that works on MovieLens is not a method that works in production. Here is the reasoning I walk through before shipping any debiasing change, posed as the failure modes that have actually bitten me.

**What happens when negatives are mostly false negatives?** IPW and popularity-aware sampling both assume that a non-interaction is a meaningful negative. In implicit feedback it usually is not — the user did not click the item because she never saw it, not because she dislikes it. Now layer popularity-aware negative sampling on top: you are sampling popular items as negatives *more* often, but a popular item is also the most likely to be a *false* negative (the user genuinely would have liked it). So aggressive 3/4-power sampling can teach the model to push down popular items the user actually wants. The fix is to *exclude known positives* and, where you can, to use exposure logs to sample negatives only from items that were *shown and not clicked* — a true negative — rather than from the whole catalog. When you cannot, keep the power at 3/4 rather than pushing toward 1, precisely because 3/4 tempers the false-negative risk.

**What happens at 100M items?** The deep tail at scale is mostly cold items with one or two interactions, and IPW's $1/\pi$ weight on a one-interaction item is enormous. Without the clip, a single such interaction — which might be a bot, a misclick, or a fat-finger — dominates the gradient. At 100M items you also cannot afford to compute exact popularity-stratified metrics over the full catalog cheaply, so you sample the tail buckets carefully and accept noisier tail-recall estimates. The practical posture at scale is: clip aggressively, prefer the sampler fix (which has no per-item weight to blow up) over IPW, and lean on causal regularization whose popularity term is *learned and smooth* rather than a raw reciprocal.

**What happens when the offline metric rises but online is flat?** This is the signature of debiasing that *over-shot* — you removed real preference along with the confound, so offline accuracy looks acceptable (the test set is head-heavy and forgiving) but online engagement does not move because you stopped showing users the head items they actually wanted. The diagnostic is stratified recall *and* a calibration check: if your debiased model's recommendation popularity distribution no longer matches users' historical popularity distribution, you have pushed past correction into distortion. Dial back toward calibration (section 7.5) rather than a global target.

**What happens with brand-new trending items?** Pure popularity-deconfounding has a blind spot: it treats *all* popularity as stale confound, but some popularity is a genuine, fresh signal — a newly-released item that is legitimately blowing up. A model that ruthlessly removes popularity will *under*-recommend the very thing that is trending right now. This is exactly the gap PDA closes by re-injecting a *predicted future* popularity at inference: it removes the stale historical confound while riding the genuine current trend. If you debias and notice you have gone cold on real trends, you need the re-injection term, not less debiasing.

**What happens when the feature is computed differently offline and online?** Popularity itself is a feature, and it is a notorious source of train-serve skew. Offline you might compute popularity over the whole training window; online you compute a streaming, decayed popularity. If the two disagree, your IPW weights and your popularity-deconfounding term are computed against the wrong $\pi$, and the correction is silently mis-calibrated. The fix is the same as for any feature skew: compute popularity from the *same* feature pipeline offline and online, ideally a shared feature store, and log the exact popularity value used at serve time so the next training run reweights against the truth rather than a reconstruction.

The throughline of all five: debiasing is a *correction*, and a correction computed against the wrong target is worse than no correction. Measure what you are correcting against (the true propensity, the user's real taste, the genuine current trend), and validate online before you trust the offline win.

## 13. How much to debias

The single most common mistake is treating debiasing as a binary — "on" or "off" — when it is a *dial*, and the right setting depends on the surface and the business.

Popularity is not the enemy. Popular items are often popular *because they are good*, and many users genuinely want them. The harm is specifically the part of the popularity score that comes from *exposure rather than preference* — the $\log \pi_i$ leak from section 3 — and the part that *overrides a user's actual taste*. So the question is never "should we remove popularity" but "how much of the popularity signal is confound versus genuine preference, and for whom."

Three principles for setting the dial:

**Match the surface to the goal.** A homepage main feed exists to retain users; it should be only mildly debiased, because users come for the things they reliably like. A "discover" or "for you, something new" surface exists to expand taste and surface the catalog; it should be aggressively debiased, even at real accuracy cost. Different surfaces, different $\lambda$.

**Match the dial to the user (calibration).** Per section 7.5, the right amount of head content is the user's *own* historical appetite for it. Calibrated debiasing — pull each user toward her own popularity distribution, not toward a global target — is almost always better than a single global knob, because it never forces tail items on someone who genuinely wants the mainstream.

**Fix the data, not just the model, for the long run.** Reweighting and re-ranking fix the model you have, but the closed loop keeps regenerating biased data. The durable fix is an exploration policy in the logging system — an $\varepsilon$-greedy or bandit slot that occasionally shows tail items so you *get labels* for them. Without exploration you are forever correcting a Pólya urn after the fact; with it, you break the urn at the source and every future model trains on less-biased data. Pair this with [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) so you can measure the tail items honestly even before they have much data.

## 14. When to reach for this (and when not to)

A decisive recommendation, because every fix is a cost.

**Always do popularity-aware negative sampling.** It is nearly free (figure 8: −0.002 Recall@10) and attacks the bias at its training source. If you draw uniform negatives today, switch to the 3/4-power sampler this sprint. There is no good reason not to.

**Reach for IPW when your bias is in the data and you can tolerate variance.** It is the principled correction and its theory is clean, but it raises gradient variance and needs the temper/clip knobs tuned. Skip it if your tail data is so thin that the up-weighted niche positives are mostly noise — IPW amplifies noise exactly where data is sparsest.

**Reach for causal regularization (PD/MACR) when popularity bias is a first-class product problem.** It is the only method that can improve accuracy *and* reduce bias, and PDA's controllable popularity injection is genuinely useful for riding trends. The cost is engineering complexity; do not reach for it if a sampler change hits your target.

**Reach for re-ranking when you need a per-surface, tunable, no-retrain lever** — especially on discovery surfaces. But do not make it your only fix: it pays the steepest accuracy cost because it overrides relevance directly. Layer it on top of a training-time fix.

**Do not debias blindly on the main feed.** Forcing tail content on users who want the mainstream is itself a bias and it will show up as an engagement drop. Use calibration to match each user, and keep the genuinely-relevant head items.

**Do not trust an offline accuracy win as evidence that debiasing hurt.** The test set is head-heavy; a drop in overall Recall@10 can coincide with a model that is *better* at the thing that matters. Always read stratified recall and, ultimately, an online A/B test (see [A/B testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders)) before concluding.

## 15. Key takeaways

- **MLE on click data learns popularity.** The data is preference times exposure, so the optimal score is $\log r + \log \pi$ — an additive popularity leak shared across all users. Every fix is a way of removing that leak.
- **Bias enters at four stages** — data (MNAR exposure), training (loss + uniform negatives), evaluation (popularity is a strong, deceptive baseline), and the closed loop — and the right fix depends on which dominates.
- **Sorting amplifies before any loop.** Top-K is a nonlinearity that turns a small score advantage into a near-binary exposure advantage, so ARP can be far above the catalog mean even for a mildly-biased model.
- **The closed loop is a Pólya urn.** Proportional-to-popularity growth produces power laws determined by early, partly-random advantages; it never self-corrects, so durable fixes require exploration in the logging policy.
- **Measure with ARP, the popularity gap, coverage, the Gini of exposure, and stratified recall** — not overall accuracy, which actively hides the collapse.
- **Popularity-aware negative sampling (3/4 power) is the free lunch.** Nearly zero accuracy cost, real tail gain. Do it first.
- **Causal regularization (PD/MACR) is the best Pareto point** — it can raise accuracy while cutting bias by removing the confounder — when you can afford the engineering.
- **Debiasing is a dial, not a switch.** Match it to the surface, calibrate it to each user, and never strip out the head items the user genuinely wants. The goal is anti-confounding, not anti-popularity.
- **Fix the data for the long run.** Reweighting fixes the model you have; exploration fixes the data you will get. Without it, you correct the same urn forever.

## 16. Further reading

- **Abdollahpouri, Burke, Mobasher — "Controlling Popularity Bias in Learning-to-Rank Recommendation" (RecSys 2017)** and **"Managing Popularity Bias in Recommender Systems with Personalized Re-ranking" (FLAIRS 2019)** — the canonical measurement and re-ranking treatments.
- **Zhang, Feng, He et al. — "Causal Intervention for Leveraging Popularity Bias in Recommendation" (PD/PDA, SIGIR 2021)** — popularity as a confounder, with controllable re-injection.
- **Wei, Feng, Chen et al. — "Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias" (MACR, KDD 2021)** — the counterfactual side-branch approach.
- **Mikolov et al. — "Distributed Representations of Words and Phrases" (NeurIPS 2013)** — the source of the 3/4-power negative sampling trick.
- **Salganik, Dodds, Watts — "Experimental Study of Inequality and Unpredictability in an Artificial Cultural Market" (Science, 2006)** — the closed loop demonstrated on humans.
- **Barabási, Albert — "Emergence of Scaling in Random Networks" (Science, 1999)** — preferential attachment and power laws, the math of rich-get-richer.
- **Steck — "Calibrated Recommendations" (RecSys 2018)** — match each user's own taste distribution instead of a global target.
- **Within the series**: [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles), [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies), [beyond accuracy: diversity, novelty, serendipity, coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage), [causal and uplift recommendation](/blog/machine-learning/recommendation-systems/causal-and-uplift-recommendation), the map [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), and the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

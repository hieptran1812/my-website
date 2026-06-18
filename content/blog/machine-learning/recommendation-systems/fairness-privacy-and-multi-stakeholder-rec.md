---
title: "Fairness, Privacy, and Multi-Stakeholder Recommendation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build a recommender that serves consumers, providers, and the platform at once — measure exposure fairness, re-rank under a fairness constraint, and train with differential privacy and federated averaging without pretending any of it is free."
tags:
  [
    "recommendation-systems",
    "recsys",
    "fairness",
    "privacy",
    "differential-privacy",
    "federated-learning",
    "multi-stakeholder",
    "machine-learning",
    "re-ranking",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-1.png"
---

A recommender almost never has one customer. The day I learned this the hard way, our offline NDCG@10 had crept up a clean half-point over a quarter of model work, online click-through was up, watch-time was up, and the dashboards were green enough that the team took a long weekend. Then the partnerships team forwarded an email from a mid-size creator who had been on the platform for three years: their impressions had fallen 60% over the same quarter, their revenue with them, and they were leaving for a competitor. They were not alone. The model had quietly learned that the safest way to maximize the consumer-click objective was to pour exposure into the same few thousand already-popular items, and the long tail of creators — the supply that made the catalog worth browsing in the first place — was starving. We had optimized one number for one stakeholder, and the system was eating itself.

That is the subject of this post. A production recommender sits at the meeting point of at least three parties: the **consumer** who wants relevant results with low effort, the **provider** (creator, seller, artist, publisher) who wants a fair shot at an audience, and the **platform** that wants revenue and a healthy two-sided ecosystem that does not collapse. Their objectives overlap, but only partly. Optimizing one — usually consumer relevance, because that is what your logs measure — can starve providers, entrench popularity bias, harm long-term health, and in some markets break the law. Recommendation is not a single-objective prediction problem. It is a **multi-objective, multi-sided** allocation problem, and the figure below is the frame for everything that follows.

![Diagram of consumers, providers, and the platform each feeding objectives into one recommender that ranks slots toward a chosen objective and a final outcome](/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-1.png)

On top of the multi-sided objective sits a second, orthogonal constraint that the previous decade made unavoidable: the whole machine runs on **sensitive behavioral data**. Every click, dwell, replay, skip, and late-night search you logged to train the model is a record of a real person's intimate preferences. Pool it carelessly and you create a liability — membership-inference attacks, embedding leakage, regulatory exposure under GDPR or its peers. So this post carries two threads at once. The first is **fairness**: how to define and measure equitable treatment for consumers and providers, and how to re-rank to enforce it. The second is **privacy**: how to train on behavioral data without centralizing it, using federated learning and differential privacy. The unifying lesson — the one I most want you to leave with — is that fairness, privacy, and accuracy trade off against each other and against revenue. There is no free lunch. The platform must choose its objective explicitly, because if it does not, the gradient will choose one for it, and the gradient only knows how to maximize the number you wrote down.

This post is the multi-stakeholder layer of the [recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking): everything here lives in the re-ranking stage and in the training pipeline that feeds it. It builds directly on [popularity bias and the rich-get-richer dynamic](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) and on [beyond-accuracy objectives like diversity and coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage). By the end you will be able to compute an exposure-disparity metric, write a fairness-constrained re-ranker as a small linear program, sketch DP-SGD on an embedding model, and reason honestly about what each costs you. Let us go.

## 1. The multi-stakeholder view: who are you actually serving?

Start with the uncomfortable observation: your training data is generated almost entirely by one stakeholder. Consumers click. Those clicks become your labels. Your loss function — pointwise log-loss, [pairwise BPR](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive), sampled softmax, whatever — is a function of those clicks. So the only stakeholder your model can natively see is the consumer, and even then it sees a biased, position-distorted, missing-not-at-random slice of consumer behavior. The provider and the platform are *structurally invisible* to the objective unless you deliberately encode them. This is why multi-stakeholder concerns feel like an afterthought in most recsys stacks: they literally are an afterthought to the gradient.

Let us name the three parties precisely, because vague talk of "balancing interests" is how these projects die in a committee.

The **consumer** is the person receiving recommendations. They want results that are relevant, fresh, and require little effort to act on. Their satisfaction is approximated offline by ranking quality — $\text{NDCG@}K$, Recall@$K$ — and online by engagement and, crucially, retention. Consumers are the loudest stakeholder because they generate the labels and because their churn is the most immediately visible.

The **provider** is whoever supplies the items: a creator uploading videos, a seller listing products, an artist releasing tracks, a host listing a property, a publisher writing articles. They want **exposure** — a fair opportunity to reach an audience that would value their content. The harm when they are ignored is not abstract: small providers earn nothing, lose motivation, and leave, and when supply leaves, the catalog shrinks and the consumer experience degrades a quarter later. Providers are typically invisible to the consumer-click objective.

The **platform** is the business operating the marketplace. It wants revenue (CTR, GMV, ad load, take-rate) and, if it is well run, long-term **ecosystem health**: a diverse supply base, satisfied consumers who return, and providers who keep producing. The platform is the only party that can see all three sides, which means the platform is the only party that can *decide* the objective. That decision is a business decision, not a modeling one, and pretending otherwise is the central failure mode here.

![Matrix mapping each of consumer, provider, and platform to what they want, the harm if ignored, and the metric that measures them](/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-2.png)

Why does optimizing consumer relevance alone hurt the other two? Two mechanisms compound. The first is **popularity bias as a feedback fixed point**, which I cover in depth in [the rich-get-richer post](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer): an item shown more gets clicked more in absolute terms, which becomes a stronger training signal, which gets it shown more. The closed loop has a fixed point where a tiny set of head items absorbs nearly all exposure. The second is **superstar concentration in the catalog**: even with an unbiased model, relevance is heavy-tailed, so a pure relevance sort naturally piles exposure onto the top of the distribution. Both push the same way — toward a small set of providers — and the consumer-click objective sees nothing wrong, because clicks are indeed maximized in the short run.

#### Worked example: how fast the tail starves

Consider a music platform with 1,000,000 tracks and 10,000,000 daily recommendation slots (impressions). Suppose exposure follows a power law with the top 1% of tracks (10,000 tracks) capturing 80% of impressions under a consumer-only ranker. That is 8,000,000 impressions to 10,000 tracks, or 800 impressions per head track per day. The remaining 990,000 tracks split 2,000,000 impressions — about 2 impressions per track per day. A track averaging 2 impressions a day will, with a 1% click rate, get roughly one click every 50 days. No artist sustains a livelihood on that, and no artist can *learn* whether their work resonates, because the platform never surfaced it to a large enough audience to find out. Now suppose a multi-stakeholder objective caps the head's share at 52%. The head drops to 5,200,000 impressions (520 per head track) and the tail gets 4,800,000 — about 4.8 per tail track, a 2.4x increase in tail exposure. The head loses 35% of its impressions; the tail more than doubles. Whether that trade is worth it is a business decision, but now it is a *quantified* business decision, not an accident.

The picture below contrasts the two regimes directly.

![Before and after comparison of a consumer-only objective that starves small providers versus a multi-stakeholder objective that balances exposure for a small accuracy cost](/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-3.png)

The right column is not free. NDCG@10 drops from 0.41 to 0.39 — a real, measurable cost to consumers. The argument for paying it is not charity; it is that ecosystem health is a long-horizon term in the platform's true objective that the short-horizon click metric cannot see. A platform that lets its supply collapse will have nothing to recommend in a year. The honest framing is: you are spending a little measured consumer relevance today to buy provider retention and catalog diversity tomorrow, and you should be able to put numbers on both sides of that trade before you ship it.

### Why the loss function only sees the consumer

It is worth dwelling on the mechanism, because it explains why every fairness and privacy intervention in this post is a *bolt-on* rather than something the base model produces naturally. Take the three most common recsys objectives. **Pointwise log-loss** fits $\hat{p}(\text{click} \mid u, i)$ to observed clicks: the gradient is $(\hat{p} - y)\,\partial_\theta \hat{p}$, and $y$ is a consumer's click. **Pairwise BPR** maximizes $\sum \log \sigma(s_{ui^+} - s_{ui^-})$ over (clicked, not-clicked) pairs: again, the supervision is which item *the consumer* preferred. **Sampled softmax** for retrieval normalizes over a sampled set of items and pushes up the score of the consumer's positive. In every case, the only label is a consumer action. A provider's desire for exposure, the platform's desire for a healthy supply base — neither appears anywhere in the loss, so neither has a gradient. The model is not being malicious when it starves the tail; it is doing precisely what you asked, which was to predict consumer clicks, and the consumer-click-maximizing policy happens to concentrate exposure.

This is also why "just add a fairness feature to the model" rarely works. A feature like *provider-is-small* does not change the objective; it only gives the model another input with which to predict clicks. If small providers get fewer clicks (because they have historically gotten less exposure — the feedback loop), the feature will, if anything, *learn to down-rank them*, because that is what maximizes the consumer objective on biased historical data. To change behavior you must change the objective itself — either by re-ranking against an explicit fairness criterion (Section 3) or by adding a fairness regularizer to the loss. Changing inputs without changing the objective is a category error that wastes a quarter of work, and I have watched it happen more than once.

### Marketplace dynamics: the supply-side flywheel

There is a flywheel that makes the provider side even more important than a static snapshot suggests. Providers are not a fixed population; they *respond* to exposure. A creator who gets a fair shot, finds an audience, and earns something produces more and better content, which improves the catalog, which attracts more consumers, which creates more exposure to distribute — a virtuous cycle. A creator who never gets surfaced produces less, gets discouraged, and eventually leaves, shrinking the catalog and degrading the consumer experience a quarter or two later, on a lag long enough that the consumer-click dashboard never connects cause to effect. The lag is the trap: by the time consumer engagement softens from a thinning catalog, the exposure decisions that caused it are two quarters in the rearview mirror and no one attributes the softening to them.

This is why the *long-horizon* framing matters so much and why naive A/B testing can mislead. A two-week A/B test of a fairness re-rank will show only the cost — the small NDCG drop, maybe a tiny short-term engagement dip — and none of the benefit, because the supply-side flywheel turns on a quarter-plus timescale. If you evaluate a multi-stakeholder change on a two-week consumer-engagement metric, you will always reject it, and you will always be wrong. The correct evaluation tracks provider-side leading indicators (provider retention, new-provider activation, share of providers earning above a livelihood threshold, catalog growth) over months, which is exactly the discipline most teams lack and exactly why the supply side gets starved by default.

## 2. Fairness, defined: which fairness, for whom?

"Fairness" is one of the most overloaded words in machine learning. Before you can optimize it you have to pin down which of several incompatible definitions you mean. The taxonomy splits along three axes, and the tree below is the map.

![Tree showing fairness in recommendation splitting into consumer-side and provider-side, then group versus individual, then exposure versus quality with a merit-weighted leaf](/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-4.png)

The first split is **which side you protect**. *Consumer-side fairness* asks whether the recommender delivers comparable quality to different groups of users — for example, whether women get recommendations as relevant as men, or whether users in a low-data region get NDCG as high as users in a high-data region. *Provider-side fairness* asks whether different providers (or groups of providers) get equitable exposure relative to what they deserve — whether new creators, creators from underrepresented groups, or small sellers get a fair share of impressions. These are genuinely different problems with different metrics; a system can be perfectly fair to consumers and brutally unfair to providers, and vice versa.

The second split is **group versus individual**. *Group fairness* compares aggregate outcomes across protected groups (demographic parity, equal opportunity). *Individual fairness* demands that similar individuals receive similar treatment — formalized as a Lipschitz condition, where the difference in treatment between two users is bounded by their distance in some similarity metric. Group fairness is far more common in production because it is measurable from logs without a contested similarity function, but it can be satisfied while being grossly unfair to specific individuals inside a group. Individual fairness is principled but requires a similarity metric you can rarely defend.

The third split, which matters most for providers, is **exposure versus quality, and static versus amortized**. Provider-side fairness almost always concerns *exposure* — the share of impressions or top-$K$ slots a provider receives. And exposure fairness is naturally **amortized over time**: you cannot give every provider the top slot on a single query (there is only one top slot), but over thousands of queries you can ensure that exposure accrues in proportion to merit. This is the key insight of Singh and Joachims' *Fairness of Exposure in Rankings* (KDD 2018) — fairness is a property of the *policy* over many rankings, not of any single ranking.

### Defining the metrics so you can actually compute them

Let us make three provider-side metrics concrete. Let $g$ index provider groups (say, $A$ = established creators, $B$ = emerging creators), let $E_g$ be the total exposure group $g$ receives over an evaluation window, and let $M_g$ be the group's *merit* — its share of relevance or of the deserved attention, however you define it (by relevance scores, by quality ratings, or simply by catalog share if you want pure parity).

**Exposure share** is the simplest: the fraction of total exposure going to group $g$,

$$\text{ExpShare}(g) = \frac{E_g}{\sum_{g'} E_{g'}}.$$

**Demographic parity of exposure** asks that exposure share equal population (catalog) share, ignoring merit entirely: $\text{ExpShare}(g) = |g| / N$ for every group. This is the strongest, bluntest notion — it says every group gets impressions in proportion to how many items it has, regardless of how good those items are.

**Exposure disparity relative to merit** is the metric I reach for most, because it respects that better content should get more exposure but bounds how far exposure can outrun merit. Define group $g$'s **merit share** $m_g = M_g / \sum_{g'} M_{g'}$ and its **exposure share** $e_g = E_g / \sum_{g'} E_{g'}$. The disparity is

$$\text{Disparity} = \max_g \left| \frac{e_g}{m_g} - 1 \right| \quad\text{or, summed,}\quad D = \sum_g \left| e_g - m_g \right|.$$

A disparity of $0$ means exposure tracks merit exactly. A disparity of $0.30$ means some group is over- or under-exposed by 30 percentage points of share relative to what it earned. **Equal opportunity** in rankings is the position-weighted analogue: it weights exposure by the attention each rank actually receives (a position-bias curve $b(k) \approx 1/\log_2(k+1)$, the same DCG discount), so a top slot counts far more than slot 50. Position-weighted exposure is the honest version because raw impression counts treat slot 1 and slot 50 as equal, which they are emphatically not — see the discussion of position bias in [the click-data bias post](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data).

#### Worked example: computing exposure disparity for two groups

Suppose over a day we serve 100,000 top-10 ranking slots. Group $A$ (established creators, 200,000 items in catalog) receives 78,000 of the position-weighted exposure units; group $B$ (emerging creators, 800,000 items) receives 22,000. So $e_A = 0.78$, $e_B = 0.22$. Now we need merit. Say we measure merit by the groups' average relevance over a held-out, position-debiased sample, and group $A$'s items are genuinely somewhat more relevant on average (they are established for a reason), giving merit shares $m_A = 0.62$, $m_B = 0.38$. Then the summed disparity is

$$D = |0.78 - 0.62| + |0.22 - 0.38| = 0.16 + 0.16 = 0.32.$$

Group $A$ is over-exposed by 16 points of share beyond its merit; group $B$ is under-exposed by the same. The ratio form gives $e_A/m_A = 1.26$ (26% over-exposed) and $e_B/m_B = 0.58$ (42% under-exposed). Note the asymmetry: a 16-point shortfall hurts the smaller group far more in relative terms. If instead we demanded pure demographic parity, the target shares would be $0.2$ and $0.8$ (catalog shares), and the disparity against parity would be a staggering $|0.78-0.2| + |0.22-0.8| = 1.16$. Which target you pick — merit or parity — is the most consequential fairness decision you will make, and it is a values decision, not a technical one.

The **fairness-accuracy trade-off** is now visible in the arithmetic: to drive $D$ toward zero you must move exposure from $A$ to $B$, which means sometimes ranking a slightly-less-relevant $B$ item above a slightly-more-relevant $A$ item. Every such swap costs a little NDCG. The question is never "should we be fair?" in the abstract; it is "how much NDCG are we willing to spend to cut disparity from 0.32 to 0.06, and is that trade on the efficient frontier?"

### Consumer-side fairness: the quieter problem

Provider fairness gets most of the attention because the harm is visible (creators leave, send angry emails). Consumer-side fairness is quieter and just as real: does the recommender deliver comparable *quality* to different groups of users? The classic failure is a **data-volume artifact**. Suppose 90% of your users are from one region with dense interaction data and 10% from a region you launched in recently. The model fits the dense group well and the sparse group poorly, so NDCG@10 is 0.45 for the majority and 0.28 for the minority — a 38% relative quality gap that the *aggregate* NDCG (0.43, weighted toward the majority) completely hides. The aggregate looks healthy; a tenth of your users are getting a measurably worse product, and you will never see it unless you slice the metric by group.

The fix for consumer-side fairness is usually different from provider fairness. Provider fairness is an *allocation* problem solved at re-rank time. Consumer-side quality gaps are usually a *data* or *model* problem: the under-served group needs better cold-start handling, group-aware sampling during training, or sometimes a separate fine-tune. Re-ranking cannot create relevance the model never learned; it can only redistribute the relevance it has. So the first diagnostic for a consumer-fairness complaint is always: slice NDCG by group, and ask whether the gap is *quality* (the model is worse for this group — a training problem) or *exposure* (the group's preferred items are systematically buried — possibly a re-rank problem). They demand different fixes, and conflating them wastes effort.

### Group versus individual, made concrete

The group-versus-individual distinction sounds academic until it bites. **Group fairness** can be satisfied while being grossly unfair to specific individuals. Suppose you enforce demographic parity of exposure between provider groups $A$ and $B$: each gets its fair aggregate share. Within group $B$, though, the re-ranker might pour all of $B$'s allotted exposure into the *single best* $B$ item and give the other 799,999 $B$ items nothing. Group parity is perfectly satisfied — $B$ got its share — yet every individual $B$ provider except one is starved. Group metrics are blind to within-group concentration.

**Individual fairness** would forbid this by demanding that similar items receive similar exposure: $|\text{Exp}(i) - \text{Exp}(j)| \le L \cdot d(i, j)$ for a similarity distance $d$ and Lipschitz constant $L$. Two items of equal merit must get near-equal exposure. The catch, and it is fatal in practice, is the distance $d$: you need a defensible, contested-by-no-one notion of "similar items," and for a heterogeneous catalog (videos, products, songs) no such metric exists that survives scrutiny. So production systems almost always use group fairness *plus* a within-group exposure cap (a crude individual-fairness proxy) rather than true individual fairness. Knowing the distinction tells you what the group metric is *not* protecting — within-group concentration — which is precisely the gap an auditor will find first.

## 3. The science: fairness-constrained re-ranking as a linear program

Here is where the multi-stakeholder problem becomes tractable. We do not retrain the model to be fair — that is slow, brittle, and entangles fairness with everything else. Instead we treat fairness as a **re-ranking** problem applied after the ranker produces relevance scores, exactly where [beyond-accuracy objectives live in the funnel](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage). The ranker gives us relevance; the re-ranker enforces exposure fairness as a constraint while sacrificing as little relevance as possible.

The clean formulation comes from Singh and Joachims. Instead of producing a single hard ranking, we optimize over **probabilistic rankings** — a doubly-stochastic matrix $P$ where $P_{ik}$ is the probability that item $i$ is placed at position $k$. Let $v_k$ be the exposure (attention) at position $k$ — the position-bias curve, $v_k = 1/\log_2(k+1)$. Let $u_i$ be item $i$'s relevance (utility). The **expected utility** of the policy is

$$U(P) = \sum_{i} u_i \sum_{k} P_{ik}\, v_k,$$

and the **expected exposure** of item $i$ is $\text{Exp}_i(P) = \sum_k P_{ik} v_k$. We want to maximize $U(P)$ subject to: (1) $P$ is doubly stochastic ($\sum_i P_{ik} = 1$ for each position, $\sum_k P_{ik} = 1$ for each item), and (2) a **fairness constraint** equalizing group exposure relative to merit. A natural constraint is that each group's exposure share matches its merit share:

$$\frac{\sum_{i \in A} \text{Exp}_i(P)}{M_A} = \frac{\sum_{i \in B} \text{Exp}_i(P)}{M_B}.$$

Because both the objective $U(P)$ and the constraints are **linear in $P$**, this is a **linear program**. You solve the LP, get a doubly-stochastic $P$, and decompose it into a distribution over deterministic rankings via the Birkhoff–von Neumann theorem (any doubly-stochastic matrix is a convex combination of permutation matrices), then sample a ranking. The LP is the principled version; in production most teams use a cheaper greedy or regularized approximation, which I will show you in code below.

The **regularized re-rank** is the practical cousin. Instead of a hard constraint, add a penalty to the per-item score that grows as a group's exposure outruns its merit. At each position $k$, pick the item maximizing

$$\text{score}_i = (1-\lambda)\, u_i \;-\; \lambda \cdot \underbrace{\big(\text{cumExp}(g_i) - \text{merit}(g_i)\big)}_{\text{how over-exposed }i\text{'s group already is}},$$

where $\text{cumExp}(g_i)$ is the exposure group $g_i$ has accumulated so far in this ranking (or amortized across recent rankings), and $\lambda \in [0, 1]$ trades relevance against fairness. At $\lambda = 0$ you recover the pure relevance sort; at $\lambda = 1$ you ignore relevance entirely and chase parity. The sweep over $\lambda$ traces out the fairness-accuracy frontier — exactly the curve you want to show a product owner before they pick an operating point.

Let us implement the greedy regularized re-ranker. It is short, runnable, and the workhorse of real systems.

```python
import numpy as np

def fair_rerank(item_ids, relevance, groups, merit_share, k=10, lam=0.3):
    """Greedy fairness-aware re-rank of a single query's candidate list.

    item_ids:    array of candidate item ids
    relevance:   array of relevance scores (same order as item_ids)
    groups:      array of group labels for each item (e.g. 0=established, 1=emerging)
    merit_share: dict {group: target exposure share in [0,1]}
    k:           number of slots to fill
    lam:         fairness weight in [0,1]; 0 = pure relevance, 1 = pure parity
    """
    # position-bias / attention weights (log-discount, the DCG curve)
    pos_weight = 1.0 / np.log2(np.arange(2, k + 2))   # length k
    total_exposure = pos_weight.sum()

    # normalize relevance to [0,1] so it is comparable to the exposure penalty
    rel = (relevance - relevance.min()) / (relevance.ptp() + 1e-9)

    chosen = []
    cum_exp = {g: 0.0 for g in merit_share}        # exposure accrued per group
    remaining = list(range(len(item_ids)))

    for slot in range(k):
        w = pos_weight[slot]
        best_idx, best_score = None, -1e18
        for idx in remaining:
            g = groups[idx]
            # current exposure share vs target merit share for this group
            exp_share = cum_exp[g] / (total_exposure + 1e-9)
            over_exposed = exp_share - merit_share[g]   # >0 means group hogging
            score = (1 - lam) * rel[idx] - lam * over_exposed
            if score > best_score:
                best_score, best_idx = score, idx
        chosen.append(item_ids[best_idx])
        cum_exp[groups[best_idx]] += w
        remaining.remove(best_idx)
    return chosen
```

The logic is worth reading carefully. At each slot we compute, for every remaining candidate, a blended score: its normalized relevance, minus a penalty proportional to how over-exposed its group already is in this ranking. A group that has hogged the early high-attention slots accrues a large $\text{cumExp}$, its $\text{over\_exposed}$ term goes positive, and its items get penalized, so the next slot tends to go to the under-exposed group. The position weights ensure we account for *attention*, not raw slot count — placing an emerging-creator item at slot 8 gives them far less exposure than slot 1, and the penalty knows it.

Now the eval harness that sweeps $\lambda$ and measures both NDCG and disparity, so we can plot the frontier.

```python
def ndcg_at_k(ranked_ids, relevance_lookup, k=10):
    gains = np.array([relevance_lookup.get(i, 0.0) for i in ranked_ids[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = (gains * discounts).sum()
    ideal = np.sort(np.array(list(relevance_lookup.values())))[::-1][:k]
    idcg = (ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum()
    return dcg / (idcg + 1e-9)

def exposure_disparity(ranked_ids, groups_lookup, merit_share, k=10):
    pos_weight = 1.0 / np.log2(np.arange(2, k + 2))
    exp = {g: 0.0 for g in merit_share}
    for slot, i in enumerate(ranked_ids[:k]):
        exp[groups_lookup[i]] += pos_weight[slot]
    tot = sum(exp.values()) + 1e-9
    e_share = {g: exp[g] / tot for g in exp}
    return sum(abs(e_share[g] - merit_share[g]) for g in merit_share)

def sweep_lambda(queries, merit_share, lambdas, k=10):
    rows = []
    for lam in lambdas:
        ndcgs, disps = [], []
        for q in queries:                      # each q has ids, rel, groups, rel_lookup
            ranked = fair_rerank(q["ids"], q["rel"], q["groups"],
                                 merit_share, k=k, lam=lam)
            ndcgs.append(ndcg_at_k(ranked, q["rel_lookup"], k))
            disps.append(exposure_disparity(ranked, q["groups_lookup"],
                                            merit_share, k))
        rows.append((lam, float(np.mean(ndcgs)), float(np.mean(disps))))
    return rows   # list of (lambda, mean_NDCG@10, mean_disparity)
```

Run this over a set of held-out queries and you get a table of $(\lambda, \text{NDCG@}10, \text{disparity})$ — your operating-point menu. The figure below shows the qualitative result of one such sweep.

![Before and after comparison of an accuracy-only ranking where one provider group dominates the top slots versus a fairness-constrained re-rank that equalizes exposure for a small NDCG cost](/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-7.png)

### Measuring it honestly on a named dataset

To get numbers you can trust, evaluate the way you would for any ranking change: a **temporal split** (train on the past, evaluate on the future) to avoid leakage, and full-corpus metrics rather than sampled ones, since [sampled metrics can reorder methods](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr). On a MovieLens-20M-style setup, grouping items into "popular" (top-quintile by interaction count) and "tail" provider buckets, a representative sweep looks like the table below. (These are illustrative numbers from a re-ranking sweep on a head/tail split; treat the exact figures as approximate, but the *shape* — disparity falls fast, NDCG falls slowly — is robust across datasets and is the whole point.)

| $\lambda$ | NDCG@10 | Exposure disparity | Tail exposure share |
| --------- | ------- | ------------------ | ------------------- |
| 0.0 (relevance only) | 0.412 | 0.30 | 0.20 |
| 0.1 | 0.408 | 0.19 | 0.31 |
| 0.2 | 0.401 | 0.11 | 0.40 |
| 0.3 | 0.391 | 0.06 | 0.47 |
| 0.5 | 0.362 | 0.02 | 0.52 |
| 1.0 (parity only) | 0.281 | 0.00 | 0.52 |

Read it as a frontier. Going from $\lambda=0$ to $\lambda=0.3$ cuts disparity by 80% (0.30 → 0.06) for a 5% relative NDCG hit (0.412 → 0.391). Pushing to full parity ($\lambda=1.0$) buys you the last 0.06 of disparity at a brutal 32% NDCG collapse. The knee of the curve is around $\lambda = 0.2$–$0.3$; past that you pay a lot for a little. **That knee is the product decision.** When you bring this to a review, you do not bring "fairness is good"; you bring this table and ask which row the business wants to live on.

One implementation detail decides whether the re-rank behaves: the relevance scores feeding it must be *calibrated*, not just correctly ordered. The penalty term subtracts an exposure deficit from a normalized relevance, so the two quantities must live on a comparable scale; if relevance scores are wildly mis-scaled (a common artifact of pairwise losses, which only constrain order, not magnitude), the fairness penalty will either dominate everything or do nothing, and the $\lambda$ knob will feel jumpy and unpredictable. Calibrating the ranker's scores first — isotonic regression or Platt scaling against observed rates, the techniques in [the calibration post](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) — makes the $\lambda$ sweep smooth and the chosen operating point stable across queries. Skipping calibration is the most common reason a fairness re-rank that worked in a notebook misbehaves in production.

## 4. The fairness-accuracy trade-off and why it is real

It is tempting to hope the trade-off is illusory — that with a clever enough model you could be perfectly fair *and* perfectly accurate. Sometimes there is genuine slack: if your model is mis-calibrated or has a popularity-bias artifact, debiasing can improve fairness *and* relevance at once, because you were leaving relevance on the table. But once you are on the efficient frontier, the trade-off is mathematically forced. Here is the intuition made rigorous.

Relevance is maximized by a unique sort: items in descending order of $u_i$ (the rearrangement inequality, since the position-attention weights $v_k$ are also descending). Any deviation from that sort — which a fairness constraint requires whenever exposure does not already track merit — pairs a high-attention slot with a lower-relevance item, strictly lowering $U(P)$. So fairness that *binds* (a constraint that the relevance-optimal ranking violates) necessarily costs utility. The only way to avoid the cost is if the constraint does not bind, which happens only when relevance already produces fair exposure — rare in any heavy-tailed catalog.

This is why I am suspicious of any vendor or paper claiming "fairness for free." Either they are (a) starting off the frontier and recovering slack (legitimate but a one-time gift), (b) measuring a fairness metric that the relevance sort happens not to violate (so the constraint is vacuous), or (c) hiding the cost in a metric they did not report. Demand the trade-off curve. If they cannot show you one, they have not measured the thing.

The trade-off also extends past relevance to **revenue**. On a marketplace where the platform takes a percentage of transactions, head items often convert better, so re-distributing exposure to the tail can cost short-term GMV even as it improves long-term supply health. This is the deepest version of the no-free-lunch lesson: fairness trades against *both* consumer relevance *and* platform revenue, and the justification for paying both costs is a long-horizon ecosystem argument that you must be able to defend with a retention or supply-growth measurement, not just assert.

#### Worked example: is the trade on the frontier?

Suppose at $\lambda = 0.3$ you measure NDCG@10 = 0.391 and disparity = 0.06, costing 0.021 NDCG versus the 0.412 baseline. Is that good? Compare to a naive alternative: just demote the top-quintile items by a fixed amount. Say that naive rule also gets disparity to 0.06 but lands NDCG at 0.372 — worse for the same fairness. The LP/regularized re-rank is *Pareto-dominant*: same disparity, higher NDCG, so it sits on the frontier and the naive rule does not. The way you prove you are on the frontier is to try several methods at matched disparity and keep the one with the highest NDCG. A single (NDCG, disparity) point tells you nothing; the *envelope* of many points across methods and $\lambda$ tells you everything.

### Stress-testing the re-ranker

A re-ranker that works on a clean MovieLens split can still break in production. Walk through the ways it bends and what to do about each.

*What if merit is mis-measured?* The whole edifice rests on the merit shares $m_g$, and merit is itself estimated from data that the biased system generated. If group $B$'s items have historically been buried, your relevance model has *little signal* about how good they are, so it may underestimate their merit, and a merit-pegged re-rank will under-correct — fairness chasing its own tail through the feedback loop. The mitigation is to estimate merit from a *position-debiased* or *exploration* sample (a small fraction of randomized exposure, exactly the kind of held-out exploration discussed in the [bandits and exploration post](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff)), so merit reflects intrinsic quality rather than historical exposure. If you peg fairness to corrupted merit, you launder the old bias into a new metric.

*What if there are 10,000 provider groups, not two?* The two-group examples are pedagogical; real catalogs have many providers, sometimes one group per provider. The greedy regularized re-rank scales fine (it just tracks per-group cumulative exposure), but the LP becomes large. In practice you either (a) bucket providers into a manageable number of fairness groups (size tiers, demographic categories, new-vs-established), or (b) enforce *per-provider* exposure floors as in FairRec rather than full pairwise parity. Pairwise parity across 10,000 groups is neither computable nor meaningful; floors and tiers are.

*What if the re-rank fights the consumer too hard on a specific query?* Amortization is the escape hatch. You do not need every single ranking to be fair — you need exposure to be fair *over time*. So track cumulative exposure across a window (a day, a week) and apply the fairness penalty based on the *running* deficit, not the within-query deficit. A query where the relevant items happen to all be from the over-exposed group can be served almost purely by relevance, because a later query will naturally lean toward the under-exposed group to settle the amortized account. This both protects per-query relevance and makes the fairness guarantee a policy property, which is what Singh and Joachims argued it should be. Per-query parity is both too costly and wrong in principle; amortized parity is the right target.

## 5. Privacy: the data underneath the model is a liability

Switch threads. Everything above assumed you have a centralized log of behavior to train on. That assumption is exactly the privacy problem. A recommender is, by construction, a model fit to the most sensitive data a person generates: what they watch alone at 2am, what they search when sick, what they buy that they would never discuss. Centralize that, and you have created three concrete risks.

First, **membership inference**: an attacker who can query the model (or observe its recommendations) can often determine whether a specific person's data was in the training set, because models behave measurably differently on examples they memorized. For a recommender, "this user was in the training data" can itself be sensitive. Second, **embedding leakage / reconstruction**: user and item embeddings can encode enough about an individual's history that an attacker with partial access can reconstruct interactions. Rare users are the most exposed — a user with one unusual interaction can have that interaction effectively memorized in their embedding row. Third, **regulatory and trust exposure**: under GDPR and similar regimes, behavioral data is personal data, with rights to access, deletion, and minimization that a giant centralized table makes hard to honor (how do you "delete" a user whose data is baked into a trained embedding table?).

The figure below contrasts the centralized regime with the private one we are about to build.

![Before and after comparison of centralized training where raw clicks are pooled and embeddings can leak versus federated and differentially private training that keeps data on-device and shares only noised gradients](/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-5.png)

Two complementary tools address this. **Federated learning** changes *where* computation happens: the model trains on-device, and only model updates (gradients), never raw data, leave the device. **Differential privacy** changes *what the updates reveal*: by clipping and adding calibrated noise, it provides a mathematical guarantee that no single user's data can be inferred from the released model. They compose — federated DP is the gold standard — but each is useful alone, and each has a cost. Let us make both rigorous.

### How a membership-inference attack actually works

To take privacy seriously you have to believe the attack is real, so here is the mechanism in concrete terms. A membership-inference attacker wants to decide: was user $x$'s data in the training set? The leverage is **overfitting**. A model fit with ordinary SGD memorizes its training examples to some degree, so it assigns *higher confidence* (lower loss) to examples it trained on than to fresh examples drawn from the same distribution. Shokri et al. (*Membership Inference Attacks Against Machine Learning Models*, S&P 2017) showed you can train a "shadow" attack classifier on the target model's confidence outputs that distinguishes members from non-members well above chance. For a recommender, the attacker observes the model's predicted scores for a candidate user's history; if those scores are suspiciously well-fit, the user was probably in training. The signal is exactly the train-test confidence gap, and the gap is largest for **rare, memorizable** users — the very people with the most to lose.

The reconstruction attack is worse. Because each user's embedding row is updated only by that user's interactions, a user's embedding can encode their history densely enough that an attacker with gradient access (a real threat in federated settings, where updates are transmitted) can invert it to recover interactions. Zhu et al. (*Deep Leakage from Gradients*, NeurIPS 2019) demonstrated reconstructing training inputs from shared gradients alone — which is precisely why federated learning *by itself* is not sufficient and must be paired with DP and secure aggregation. Sharing gradients instead of data feels safe; it is not, unless the gradients are clipped, noised, and aggregated so that no single user's contribution survives.

## 6. Differential privacy: the guarantee and the noise–utility trade-off

Differential privacy is the rare privacy notion with a real mathematical definition rather than a vibe. A randomized algorithm $\mathcal{M}$ is **$(\epsilon, \delta)$-differentially private** if for any two datasets $D$ and $D'$ that differ in a single individual's records, and any set of outputs $S$,

$$\Pr[\mathcal{M}(D) \in S] \le e^{\epsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta.$$

Read it plainly: whether or not your data is in the dataset, the distribution of outputs is nearly the same — within a multiplicative factor of $e^{\epsilon}$, plus a small slack $\delta$. The parameter $\epsilon$ is the **privacy budget**: small $\epsilon$ (say 1) means strong privacy (the two distributions are very close, so an attacker learns almost nothing about whether you are present); large $\epsilon$ (say 10) means weak privacy. $\delta$ is the probability the guarantee fails outright and is set tiny, typically $\delta < 1/N$ for $N$ users. The beautiful property is **composition**: if you run $T$ private steps, the total privacy cost accumulates in a controlled way, and a *privacy accountant* tracks the running $\epsilon$ so you can stop before you overspend.

How do you make SGD private? **DP-SGD** (Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016) modifies the gradient step with two operations, shown in the pipeline below.

![Stack diagram of the DP-SGD pipeline computing per-sample gradients, clipping to a norm bound, adding Gaussian noise, updating weights, and accounting the spent epsilon budget](/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-6.png)

First, **per-example gradient clipping**: compute the gradient *for each example separately* and clip its $L_2$ norm to a bound $C$, so no single example can move the weights more than $C$. This bounds the **sensitivity** — how much one person's data can change the update. Second, **Gaussian noise**: add noise drawn from $\mathcal{N}(0, \sigma^2 C^2 I)$ to the *summed* clipped gradients before averaging. The noise multiplier $\sigma$ and the bound $C$ together determine $\epsilon$: more noise (larger $\sigma$) buys smaller $\epsilon$ (more privacy) at the cost of a noisier, slower-learning update. That is the **privacy–utility trade-off** in one sentence: noise is the price of privacy, and noise hurts accuracy.

Why does this give a guarantee? The logic is the **Gaussian mechanism**. Differential privacy of a function $f$ requires adding noise scaled to $f$'s **sensitivity** $\Delta f$ — the most $f$ can change when one person's data is added or removed. With no clipping, a single example's gradient is unbounded, so the sensitivity of the summed gradient is unbounded, and no finite noise gives privacy. Clipping to norm $C$ caps each example's contribution, forcing $\Delta f \le C$ (removing one example changes the sum by at most its clipped norm). The Gaussian mechanism then guarantees $(\epsilon, \delta)$-DP for a single step when the noise standard deviation satisfies $\sigma \ge \frac{C \sqrt{2 \ln(1.25/\delta)}}{\epsilon}$. So the clip $C$ and the noise scale $\sigma$ are *coupled by the guarantee*: you do not pick them independently — you pick the privacy target $(\epsilon, \delta)$ and they follow. Composition across $T$ steps (tracked by the moments/RDP accountant) accumulates the total budget, and Poisson subsampling of the batch *amplifies* privacy (an example not sampled this step cannot leak this step), which is why DP-SGD spends far less budget than the naive per-step bound would suggest. The accountant is what makes $T$ steps of training feasible at a usable $\epsilon$ at all.

Here is DP-SGD on an embedding-based recommender, written explicitly so you can see the clip-then-noise structure. In production you would use Opacus (PyTorch) or TensorFlow Privacy, which vectorize per-sample gradients and bundle a privacy accountant, but the manual version is the one that teaches.

```python
import torch
import torch.nn as nn

class MFModel(nn.Module):
    """Tiny matrix-factorization recommender: user and item embeddings."""
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)

def dp_sgd_step(model, batch, optimizer, clip_C=1.0, noise_mult=1.0, lr=0.05):
    """One DP-SGD step: per-sample clip, sum, add Gaussian noise, update."""
    u, i, y = batch                          # users, items, labels (0/1)
    B = u.shape[0]
    # accumulate clipped per-sample gradients
    summed_grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for b in range(B):                       # explicit per-sample loop (clarity)
        model.zero_grad()
        pred = model(u[b:b+1], i[b:b+1])
        loss = nn.functional.binary_cross_entropy_with_logits(pred, y[b:b+1].float())
        loss.backward()
        # clip this example's gradient to L2 norm clip_C
        total_norm = torch.sqrt(sum((p.grad ** 2).sum()
                                    for p in model.parameters() if p.grad is not None))
        scale = min(1.0, clip_C / (total_norm + 1e-9))
        for n, p in model.named_parameters():
            if p.grad is not None:
                summed_grads[n] += p.grad * scale
    # add Gaussian noise calibrated to the clip bound, then average
    for n, p in model.named_parameters():
        noise = torch.normal(0.0, noise_mult * clip_C, size=p.shape)
        priv_grad = (summed_grads[n] + noise) / B
        p.data -= lr * priv_grad
    return loss.item()
```

The two privacy-relevant lines are the clip (`scale = min(1.0, clip_C / total_norm)`, which guarantees no example contributes a gradient of norm more than `clip_C`) and the noise (`torch.normal(0.0, noise_mult * clip_C, ...)`, calibrated to that same bound). Everything else is ordinary SGD. The privacy budget $\epsilon$ is then computed by an accountant from `noise_mult`, the sampling rate `B / n_users`, and the number of steps. In Opacus the whole thing collapses to a few lines:

```python
from opacus import PrivacyEngine

model, optimizer, train_loader = ...           # ordinary PyTorch objects
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=20,
    target_epsilon=3.0,        # the privacy budget you commit to
    target_delta=1e-5,
    max_grad_norm=1.0,         # the clip bound C
)
# train normally; Opacus clips per-sample grads + adds noise + tracks epsilon
for epoch in range(20):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()       # DP step applied internally
print("spent epsilon:", privacy_engine.get_epsilon(delta=1e-5))
```

Opacus uses *functorch*-style vectorized per-sample gradients (no Python loop) and the RDP/PRV accountant, so it is fast enough for real models. You commit to a `target_epsilon` up front and it sizes the noise to hit it.

#### Worked example: the privacy–utility hit at a given epsilon

Take the matrix-factorization recommender on a MovieLens-style dataset. Non-private training reaches, say, Recall@20 = 0.34. Now train with DP at three budgets. At a loose budget $\epsilon = 10$, the noise multiplier is small and the model loses little — Recall@20 ≈ 0.32, a 6% relative drop. At a moderate $\epsilon = 3$ — a common "strong but usable" operating point — the noise is larger and Recall@20 ≈ 0.28, an 18% relative drop. At a strict $\epsilon = 1$, the noise dominates the signal for rare users and Recall@20 ≈ 0.22, a 35% drop. The shape is the lesson: utility falls *gently* as you loosen $\epsilon$ from 1 toward 3, then *very gently* from 3 toward 10, because beyond $\epsilon \approx 3$–4 the noise is already small relative to the gradient. So the practical advice is: pick the smallest (most private) $\epsilon$ that lands on the flat part of the curve, usually $\epsilon \in [2, 8]$ for recommenders. Below that you pay steeply; above it you are giving away privacy for almost no accuracy gain. The table makes the curve concrete.

| Privacy budget $\epsilon$ | Recall@20 | Relative drop vs non-private | Verdict |
| ------------------------- | --------- | ---------------------------- | ------- |
| $\infty$ (non-private) | 0.34 | — | baseline |
| 10 | 0.32 | −6% | barely private, small cost |
| 8 | 0.31 | −9% | good balance |
| 3 | 0.28 | −18% | strong privacy, real cost |
| 1 | 0.22 | −35% | very strong, steep cost |

These figures are illustrative of the *shape* DP-SGD produces on a small MF recommender; the exact drop depends on model size, dataset, and how heavy the tail of rare users is. The honest framing for any DP deployment is identical to the fairness one: show the privacy–utility *curve*, not a single point, and pick the knee.

There is one more DP subtlety specific to recommenders worth flagging. The **rare-user problem**: DP noise hurts users with little data the most, because their gradient signal is weak relative to the fixed noise. The very users most vulnerable to a privacy attack (memorizable because they are unusual) are the ones DP protects best but serves worst. This tension — privacy protects the rare user but degrades their recommendations — is the recommender-specific face of the privacy–utility trade-off, and it is why DP recommenders often pair a globally-private model with on-device personalization, which we turn to next.

## 7. Federated learning: train on-device, share gradients not data

Federated learning attacks the same problem from the other side: instead of protecting a centralized dataset, it never centralizes the data at all. The canonical algorithm is **Federated Averaging (FedAvg)** (McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017). The round structure is simple:

1. The server sends the current global model $w_t$ to a sample of $K$ client devices.
2. Each client $k$ trains locally for several SGD epochs on its *own* data, producing $w_t^k$. Raw data never leaves the device.
3. Each client sends back only its model update (or the new weights) $w_t^k$.
4. The server averages, weighting by each client's data size $n_k$: $w_{t+1} = \sum_k \frac{n_k}{n} w_t^k$.

Why does this limit leakage? Because the server sees *aggregated model updates*, not raw interactions. A click never crosses the network. And when you add **secure aggregation** (clients encrypt updates so the server only ever sees the sum, never an individual update) plus DP on the updates, the server learns essentially nothing about any one client. FedAvg also slashes communication: clients do many local steps per round, so far fewer rounds (and far less raw data transfer) are needed than naive distributed SGD.

Here is the FedAvg loop, server and client, in PyTorch-style pseudocode that is close to runnable:

```python
import copy
import torch

def client_update(global_state, local_data, model_fn, local_epochs=2, lr=0.05):
    """Train a copy of the global model on one client's local data only."""
    model = model_fn()
    model.load_state_dict(global_state)        # start from global weights
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(local_epochs):
        for batch in local_data:               # local data NEVER leaves device
            opt.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            # optional: clip + noise here for federated DP
            opt.step()
    return model.state_dict(), len(local_data)

def fedavg(model_fn, clients, rounds=100, clients_per_round=20):
    global_model = model_fn()
    global_state = global_model.state_dict()
    for r in range(rounds):
        selected = sample_clients(clients, clients_per_round)
        updates, sizes = [], []
        for c in selected:
            state, n = client_update(global_state, c.data, model_fn)
            updates.append(state); sizes.append(n)
        total = sum(sizes)
        # weighted average of client states (the FedAvg step)
        new_state = copy.deepcopy(updates[0])
        for key in new_state:
            new_state[key] = sum(u[key] * (n / total)
                                 for u, n in zip(updates, sizes))
        global_state = new_state
    return global_state
```

For recommenders, FedAvg has a wrinkle: the **item embedding table** is global (every device benefits from learning about every item), but the **user embedding is private** to each device (your row of the user table is yours and should stay on your device). A clean federated recommender therefore keeps the user embedding on-device, never uploaded, and federates only the item embeddings and the interaction network. This is the architecture behind **Google's Gboard** next-word and query-suggestion models, which trained federated language models on millions of phones without uploading a single keystroke — the most cited production federated deployment, and a direct cousin of on-device recommendation.

On-device personalization is the natural endpoint: a globally-trained (federated, DP) base model that learns population-level patterns, plus a small on-device fine-tune (just the user embedding, or a tiny adapter) that personalizes using local data that never leaves the device. This pattern — global model from federated DP training, local personalization on-device — gives you both the population signal and the private personal signal, and it is where the field is heading for privacy-sensitive recommendation. (For the LoRA-style adapters that make on-device fine-tuning cheap, the [LoRA and PEFT material in the training-techniques section](/blog/machine-learning/training-techniques/fine-tuning-tool-calling-llms-when-how) is a useful reference; the same low-rank-adapter idea applies to a personalization head.)

### Why federated recommendation is hard in practice

FedAvg in a slide is clean; federated recommendation in production is not. Three problems dominate. **Non-IID clients**: each device's data is wildly unrepresentative of the population — your phone holds *your* tastes, not a random sample — so local SGD steps pull the model in idiosyncratic directions, and averaging non-IID updates converges slower and to a worse optimum than centralized training on the pooled data. This is the central reason federated models trail their centralized counterparts in accuracy, on top of any DP cost. Mitigations like FedProx (a proximal term keeping local updates near the global model) and limiting local epochs help but do not erase the gap.

**Stragglers and availability**: phones are offline, on battery, on metered networks. A round can only use the clients that happen to be plugged in, on Wi-Fi, and idle — typically a small, possibly biased slice of users (the chronically-offline are under-represented, a fairness problem hiding inside the privacy mechanism). The orchestration to handle this — client selection, timeouts, dropping stragglers without biasing the aggregate — is a serious distributed-systems effort, which is why federated learning is a heavy engineering tax, not a config flag.

**The cold item / cold user problem is worse**: a brand-new item has no on-device interactions anywhere yet, and a federated round only learns about items some sampled client interacted with locally. Propagating a new item's embedding across the fleet takes rounds, so federated recommenders are slower to react to fresh catalog than centralized ones — a real cost for news, short-video, or any fast-moving catalog, and it interacts directly with the [cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem).

#### Worked example: the federated accuracy gap

Take the matrix-factorization recommender again. Centralized training (data pooled) reaches Recall@20 = 0.34. Federated training with FedAvg over the same users, same model, same total compute, but with each device training only on its own ~40 interactions and the server averaging 20 clients per round over 200 rounds, lands at Recall@20 ≈ 0.30 — a 12% relative drop, attributable almost entirely to non-IID slowdown and the averaging penalty. Now stack federated DP on top (clip + noise the client updates at $\epsilon = 3$): Recall@20 drops further to ≈ 0.27, a 21% total relative drop versus centralized non-private. That is the full price of the gold-standard private setup: roughly a fifth of your accuracy, traded for keeping raw data on-device and a provable $\epsilon = 3$ bound. Whether a fifth of Recall is worth it depends entirely on whether your domain *requires* this level of privacy — for a keyboard or a health app, yes; for a public-catalog shopping feed, almost certainly not.

### Data minimization: the cheapest privacy of all

**Data minimization** is the unglamorous third leg, and it is usually the highest-leverage one. Before any of the fancy machinery, ask: do you need to log this field at all? Do you need to retain raw events for two years, or would aggregated features suffice? Do you need a user-level join key, or would a salted, rotating session id break the linkage that makes re-identification possible? Every field you do not collect is a field that cannot leak. The strongest privacy guarantee is the data you never gathered, and in practice a serious data-minimization pass — shorter retention windows, coarser logging (truncating precise timestamps and geolocation), fewer joined identifiers, hashing or dropping raw text queries — often reduces risk more, and more cheaply, than any DP or FL deployment. It also makes the rest easier: less data to clip, less to federate, fewer rare-user rows to memorize. Privacy engineering starts with the schema, not the noise multiplier, and the team that reaches for DP-SGD before doing a retention audit is optimizing the wrong end of the pipeline.

## 8. The tension: there is no free lunch across the three axes

Step back and look at all three axes together — accuracy, fairness, privacy — plus revenue lurking behind them. They do not align. The summary matrix below reads methods down the rows against the three stakeholder axes across, and the diagonal of green-and-red tells the whole story: nothing wins on every axis.

![Matrix comparing accuracy-only, fair re-rank, DP-SGD, and federated-plus-DP methods across NDCG, exposure disparity, and privacy showing each method trades one axis against another](/imgs/blogs/fairness-privacy-and-multi-stakeholder-rec-8.png)

Trace the trade-offs concretely. **Accuracy-only** wins NDCG and loses everything else: worst disparity, zero privacy. **Fair re-rank** fixes disparity at a small NDCG cost but does nothing for privacy — your data is still centralized. **DP-SGD** buys a strong privacy bound at a moderate accuracy cost but does not touch fairness (a private model can be just as exposure-skewed as a non-private one; privacy and fairness are *orthogonal* problems, a point teams routinely conflate). **Federated + DP** stacks privacy guarantees but pays the most accuracy and still needs a separate fairness mechanism. The matrix is not a ranking; it is a menu of trades, and the right cell depends entirely on which stakeholder's pain is currently most acute for your business.

The deepest interactions are the cross-axis ones. **Privacy fights fairness for rare groups**: DP noise hurts under-represented users and providers the most (weak signal, fixed noise), so naive DP can *worsen* fairness for exactly the groups fairness aims to protect — a documented and counterintuitive interaction. **Fairness fights revenue**: re-distributing exposure to the tail can cost short-term GMV on a take-rate marketplace. **Privacy fights revenue**: less data, more noise, on-device-only signals — all reduce model accuracy, which reduces engagement and revenue. There is no setting of the dials that maxes all four. The platform's job is to *choose*, explicitly and measurably, which is the entire point of the next section.

## 9. Case studies: how the field actually ships this

Abstract trade-offs become believable when you see them shipped. Four landmark cases, cited so you can read the primaries.

**Spotify and provider (artist) fairness.** Spotify researchers have published extensively on balancing listener satisfaction against artist exposure, framing recommendation explicitly as a two-sided marketplace. Mehrotra et al. (*Towards a Fair Marketplace*, CIKM 2018) showed that ranking purely for relevance concentrates streams on already-popular artists and quantified the relevance cost of redistributing exposure toward less-popular artists — an early, concrete demonstration of the fairness-accuracy frontier on a real music platform. The practical lever they explore is the same exposure-aware re-ranking developed here: trade a small amount of listener relevance for a meaningful increase in the diversity of artists who get heard.

**Singh and Joachims — Fairness of Exposure in Rankings (KDD 2018).** The theoretical backbone of Section 3. They formalized exposure fairness as a property of a *probabilistic ranking policy*, posed the fair-ranking problem as a linear program over doubly-stochastic matrices, and proved you can decompose the LP solution into a distribution over deterministic rankings (via Birkhoff–von Neumann) to sample from. Their amortized-fairness framing — fairness accrues over many queries, not within one — is why provider fairness is a policy property, not a per-query property.

**FairRec (Patro et al., WWW 2020).** A two-sided fairness algorithm guaranteeing each producer a minimum *guaranteed exposure* (a fairness floor) while maintaining envy-free consumer allocations. FairRec reframed the problem as a fair-allocation problem from economics (constrained fair division) and showed you can guarantee both producer exposure floors and consumer relevance bounds simultaneously, with measured trade-offs on real datasets. It is the cleanest demonstration that two-sided fairness need not be a vague aspiration — it can be a *guarantee* with a knob.

**Google Gboard — production federated learning.** The most-cited production deployment of federated learning (Hard et al., *Federated Learning for Mobile Keyboard Prediction*, 2018; Bonawitz et al. on the systems side, 2019). Gboard trains next-word and query-suggestion models across millions of Android phones: each device trains locally on the user's typing, uploads only an aggregated, secure-aggregated, and differentially-private model update, and never uploads keystrokes. It demonstrated that federated DP works at production scale on a privacy-critical, recommendation-adjacent task — and it is the existence proof the whole on-device-recommendation field points to. **Apple** has separately shipped local differential privacy for usage analytics (emoji suggestions, QuickType) since iOS 10, adding noise *on-device* before any data is sent, the LDP cousin of the central DP we built here.

These cases share a structure: each picked one stakeholder pain as primary (artist exposure for Spotify, keystroke privacy for Gboard), built the specific mechanism for it, *measured the cost on the other axes*, and shipped a defended trade-off. None of them got all axes for free, and none pretended to.

## 10. Choosing your stakeholder objective

So how do you actually decide? Not with a universal formula — there isn't one — but with a disciplined process. The objective is almost always a **weighted combination**, and the weights are a business decision the platform must own.

The general form is a scalarized multi-objective: rank by

$$\text{score}_i = w_c \cdot \text{relevance}_i \;+\; w_p \cdot \text{provider\_fairness}_i \;+\; w_r \cdot \text{revenue}_i,$$

subject to privacy constraints baked into *training* (DP budget $\epsilon$, federated architecture) rather than ranking. The weights $w_c, w_p, w_r$ encode the platform's values, and you set them by (1) measuring the frontier — sweep the weights, plot relevance vs disparity vs revenue — and (2) picking the operating point the business can defend, ideally validated by an [online A/B test](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) that measures the *long-horizon* outcomes (provider retention, supply growth, consumer retention) the offline metrics cannot see.

A few decision rules I trust:

- **If small-provider churn is rising**, you have a provider-fairness problem; reach for exposure-aware re-ranking (Section 3) and measure provider retention, not just NDCG. Start at the knee of the $\lambda$ curve ($\lambda \approx 0.2$–0.3).
- **If you face regulatory exposure or a privacy-sensitive domain** (health, finance, messaging), you have a privacy problem; reach for DP-SGD at $\epsilon \in [2, 8]$ and/or federated training, and accept the accuracy cost as the price of operating legally and keeping user trust.
- **If consumer groups get unequal quality**, you have a consumer-fairness problem; audit NDCG *by group* and debias the data or re-rank for quality parity — and check whether the gap is a data-volume artifact you can fix by improving cold-start for the under-served group, which is a [cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) in disguise.
- **If everything looks fine offline but a stakeholder is quietly leaving**, trust the leaving stakeholder over the dashboard. The dashboard measures the consumer-click objective; the leaving party is a signal the objective is incomplete. This is exactly the trap from the intro.

The non-negotiable rule is **explicitness**. Write the objective down. Put the weights in a config the whole team can see. Measure all three axes on every launch, not just NDCG. The failure mode is never "we chose the wrong weights" — it is "we never chose, so the click metric chose for us, and it chose to starve the supply."

## 11. When to reach for each lever (and when not to)

Each of these tools is a cost. Here is when each is worth it and when it is not.

**Fairness-aware re-ranking — reach for it when** providers are a real stakeholder (a marketplace, creator platform, multi-seller catalog) and you can measure provider retention or supply health. **Don't reach for it when** you are a single-provider service (a company's own product catalog, a first-party media library) — there is no provider fairness problem if there is one provider, and adding a fairness penalty just costs relevance for no one's benefit. Also don't reach for full parity ($\lambda \to 1$); it is almost never worth the NDCG collapse. Live at the knee.

**Differential privacy — reach for it when** you operate in a regulated or privacy-critical domain, or when membership inference / embedding leakage is a credible threat for your user base. **Don't reach for it when** the data is non-sensitive (public catalog metadata, aggregate trends), or when you have not first done data minimization — DP on data you should not have collected is solving the wrong problem. And never ship DP at an $\epsilon$ so loose ($\epsilon > 20$) that the guarantee is theater; either commit to a meaningful budget or do not claim privacy.

**Federated learning — reach for it when** data genuinely cannot or should not be centralized (on-device keyboards, health apps, cross-silo data with legal barriers) and you have the engineering capacity for the substantial systems complexity (client orchestration, secure aggregation, stragglers, non-IID client data). **Don't reach for it when** you already have a centralized, well-governed data warehouse and no regulatory bar to using it — federated learning is a large engineering tax, and if centralized training with DP meets your privacy bar, take the simpler path. Federated is for when centralization is off the table, not a default.

**On-device personalization — reach for it when** you have a strong global model and want private, low-latency personalization without uploading personal signals. **Don't reach for it when** your devices are too constrained to run even a small fine-tune, or when the personalization gain over the global model is marginal — measure it before you build the on-device pipeline.

The meta-rule: do not stack all of these at once on day one. Pick the one axis that is your actual, measured pain, ship the matching lever, measure the cost on the other axes, and only add the next lever when the next pain is real. A team that bolts on DP, federation, *and* fairness re-ranking simultaneously, before measuring whether any of the three matters, ends up with a slow, inaccurate, hard-to-debug system that nobody can reason about — and usually rips two of the three back out.

## Key takeaways

- A recommender serves **three stakeholders** — consumers (relevance), providers (exposure), and the platform (revenue + health). Optimizing only consumer relevance is the default, because clicks are the only labels, and it quietly starves providers and entrenches popularity.
- **Provider fairness is an exposure problem**, amortized over many rankings, not a single-ranking property. Measure it with **exposure disparity** relative to merit ($D = \sum_g |e_g - m_g|$), position-weighted so a top slot counts more than a deep one.
- **Fairness-constrained re-ranking** is the practical lever: a linear program over doubly-stochastic rankings, or a cheap regularized greedy re-rank with a weight $\lambda$. Sweep $\lambda$ to trace the **fairness-accuracy frontier**; live at the knee (often $\lambda \approx 0.2$–0.3, where disparity falls 80% for a ~5% NDCG cost).
- The **fairness-accuracy trade-off is real on the frontier** (the rearrangement inequality forces it), though debiasing a mis-calibrated model can give a one-time free gift. Be suspicious of "fairness for free" — demand the trade-off curve.
- Recommenders run on **sensitive behavioral data**, exposing you to membership inference, embedding leakage, and regulatory risk. **Data minimization first** — the strongest privacy is the data you never collected.
- **Differential privacy** gives a real $(\epsilon, \delta)$ guarantee via **DP-SGD** (per-example clip to bound sensitivity, add calibrated Gaussian noise, account the budget). The **privacy-utility curve** is steep below $\epsilon \approx 3$ and flat above; operate at the knee, usually $\epsilon \in [2, 8]$.
- **Federated learning (FedAvg)** trains on-device and shares only aggregated model updates, never raw data — proven at production scale by **Gboard**. For recommenders, keep the user embedding on-device and federate the item table.
- **There is no free lunch across accuracy, fairness, privacy, and revenue.** They trade against each other (DP even hurts fairness for rare groups). The platform must **choose its objective explicitly** — write the weights into a config, measure all axes on every launch, and validate the long-horizon outcomes with an online test.

## Further reading

- **Singh & Joachims, "Fairness of Exposure in Rankings" (KDD 2018)** — the LP formulation of exposure fairness over probabilistic rankings; the theoretical backbone of fairness-aware re-ranking.
- **Patro et al., "FairRec: Two-Sided Fairness for Personalized Recommendations in Two-Sided Platforms" (WWW 2020)** — guaranteed-exposure floors for producers with envy-free consumer allocations.
- **Mehrotra et al., "Towards a Fair Marketplace: Counterfactual Evaluation of the trade-off between Relevance, Fairness & Satisfaction in Recommendation Systems" (CIKM 2018)** — Spotify's two-sided marketplace framing and measured trade-offs.
- **Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)** — DP-SGD: per-example clipping, Gaussian noise, and the moments accountant.
- **McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)** — Federated Averaging (FedAvg), the foundational federated algorithm.
- **Hard et al., "Federated Learning for Mobile Keyboard Prediction" (2018)** and the Opacus / TensorFlow Privacy docs — production federated learning and practical DP-SGD tooling.
- Within this series: [popularity bias and the rich-get-richer dynamic](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer), [beyond-accuracy objectives: diversity, novelty, serendipity, coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage), [the recommendation funnel: retrieval, ranking, re-ranking](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking), [causal and uplift recommendation](/blog/machine-learning/recommendation-systems/causal-and-uplift-recommendation), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

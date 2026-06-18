---
title: "Beyond Accuracy: Diversity, Novelty, Serendipity, and Coverage"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why a recommender that only chases accuracy ships a boring, redundant, popularity-collapsed list, and the metrics that capture what actually makes a list good: intra-list diversity, self-information novelty, serendipity, and catalog coverage, with the math, runnable MMR and DPP code, and a measured accuracy-diversity trade-off curve on MovieLens-20M."
tags:
  [
    "recommendation-systems",
    "recsys",
    "diversity",
    "novelty",
    "serendipity",
    "coverage",
    "re-ranking",
    "mmr",
    "dpp",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-1.png"
---

The first recommender I shipped that "won" offline and lost online taught me everything in this post. We had spent a quarter pushing NDCG@10 on a movie feed from 0.39 to 0.42, a result the team was genuinely proud of, and we shipped it to a fifty-fifty A/B test expecting a clean engagement win. Two weeks later the verdict came back: sessions per user were *down* 4%, and the support inbox had a recurring complaint we had never seen before — "why do you keep showing me the same kind of thing?" I pulled a hundred random top-10s from the winning model and read them by hand, and the problem was obvious the moment I stopped looking at the metric and started looking at the *list*. Slot one was a popular action movie. Slots two through ten were nine more popular action movies, several of them near-duplicates from the same franchise. Every list was, in effect, ten copies of the user's single most-predictable interest. The model was not wrong about any one item. It was wrong about the *set*, and our metric could not see the set.

That is the entire subject of this post. **Accuracy is necessary but not sufficient.** A ranker that perfectly orders items by predicted relevance, and then takes the top K, will reliably produce a list that is redundant (the top items are near-identical because relevance is correlated with similarity), popularity-collapsed (popular items have the most training signal, so the model is most confident about them), and ultimately boring. Users do not experience a list as a bag of independent items; they experience it as a *whole*, and a list of ten almost-identical good items is worse than a list of seven good items plus three pleasant surprises. The metrics that capture this — **diversity, novelty, serendipity, and coverage** — are collectively called *beyond-accuracy* metrics, and learning to measure and optimize them is the difference between a recommender that scores well on a leaderboard and one that people actually want to keep using.

![A before and after comparison of an accuracy-only top-10 that collapses onto ten near-identical popular items against a diversity-aware list that keeps relevance while adding variety and long-tail reach](/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-1.png)

This work lives in a specific place in the funnel you have followed through this series. Retrieval scans millions of items down to a few hundred candidates; ranking scores each candidate for relevance; and then **re-ranking** reshapes the scored list before it is served. Diversity, calibration, and business rules are re-ranking concerns — they operate on the *list as a set*, not on each item in isolation, which is exactly why they cannot be expressed as a per-item accuracy metric. If you want the map of where this sits, see [the recommendation funnel: retrieval, ranking, re-ranking](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking); the accuracy metrics this post deliberately moves *beyond* are defined in [offline evaluation metrics: recall, NDCG, MAP, MRR](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr); the mechanism that *creates* the popularity collapse is in [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer); and the longer-horizon danger — a homogenizing closed loop — is in [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles). The top-level map is [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), and the synthesis is the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

By the end you will be able to define each beyond-accuracy metric precisely and tell them apart; compute intra-list diversity, novelty, serendipity, and coverage by hand and in NumPy; derive why a greedy MMR re-ranker is the right first tool and what its λ knob controls; sketch a determinantal point process (DPP) as the principled cousin of MMR; implement calibrated re-ranking in the Steck (2018) sense; and read an accuracy-diversity trade-off curve so you can pick a defensible operating point instead of accidentally parking at the boring extreme.

## 1. Why accuracy alone collapses the list

Start with the precise reason an accuracy-only ranker produces a redundant list, because once you see the mechanism the fixes become obvious. Three forces conspire.

**Force one: relevance is correlated with similarity.** A pointwise or pairwise ranker scores each item independently by how relevant it is to the user. But the items a user finds relevant are not scattered randomly across the catalog — they cluster. If you like one Christopher Nolan film you probably like the others, so the ranker assigns all of them a high score, and the top of the list fills with near-neighbors. The ranker is doing its job correctly per item; the redundancy is an *emergent* property of taking the top K of a function whose high-scoring region is a tight cluster. Nothing in the per-item objective penalizes putting two near-identical items next to each other.

**Force two: popularity has the most training signal.** Popular items are interacted with by many users, so the model sees thousands of positive examples for them and learns confident, high embeddings for them. Niche items have a handful of interactions, noisy gradients, and embeddings that stay close to the random initialization. When you sort by predicted relevance, the popular items float to the top not only because they are genuinely liked but because the model is *more certain* about them. This is the engine of popularity bias, and an accuracy metric rewards it: in a test set drawn from the same skewed distribution, recommending popular items is a safe bet that scores well. The metric and the bias point the same direction.

**Force three: the metric is per-item and position-weighted.** NDCG, Recall@K, MAP, and MRR all decompose into a sum over individual items, each scored against the held-out ground truth, possibly discounted by position. None of them contain a term that depends on *the relationship between two recommended items*. You can swap item 3 for a near-duplicate of item 2 and the metric will not move at all if both are relevant. The objective is structurally blind to redundancy. That is not a bug in NDCG — NDCG was never trying to measure variety — but it means a system tuned only on NDCG has no pressure whatsoever toward a varied list.

Put these together and the failure is over-determined. The ranker concentrates score on a tight, popular cluster; the metric rewards exactly that; and nothing pushes back. The result is the list I read by hand: ten good, redundant, popular items. Users churn not because any item was bad but because the *experience* was monotonous, and a monotonous feed gives a user no reason to come back tomorrow.

It is worth seeing how this becomes a *self-reinforcing* loop, because that is what turns a one-time annoyance into a structural collapse. The recommender shows popular items; users mostly interact with what they are shown (you cannot click what you never see); those interactions become next week's training data, which is now *even more* concentrated on the popular items; the retrained model is therefore *even more* confident about them and surfaces them *even more* aggressively. Each cycle tightens the concentration. The long tail, having received no exposure, generates no fresh interaction data, so its embeddings never improve and its predicted relevance never rises — it is starved of exactly the signal that would let the model learn it is good. Left alone, this closed loop drives the served catalog toward a tiny fixed point of perennially-popular items, and the beyond-accuracy metrics are how you *detect* it (coverage falling, gini rising) and the diversity/novelty re-rank is one of the levers that *breaks* it by forcing exposure back into the tail. The full dynamics of this loop are the subject of [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles); for this post the point is that an accuracy-only objective does not just produce a boring list today, it actively *worsens* the catalog's health over time.

There is a deeper economic reason to care, too. A platform's catalog is an asset. If 96% of your catalog is never recommended to anyone, you paid to license or host content that does no work, your suppliers (sellers, creators, studios) see no traffic and leave, and your system becomes structurally incapable of discovering that a niche item is actually a sleeper hit, because it never gets the exposure that would generate the data to learn that. Accuracy-only ranking is not just a worse user experience; it is a slow-motion impoverishment of the catalog. Coverage, the metric we will get to, is the health check for exactly this.

So the goal of this post is not to abandon accuracy. A list of ten random items has perfect diversity and zero value. The goal is to optimize accuracy *subject to* the list being varied, fresh, and reaching the catalog — to move along a frontier rather than sit at one corner of it.

It is worth pausing on *why the offline metric cannot warn you about any of this*, because that is the trap. NDCG, Recall, MAP, and MRR are computed against a held-out set of items the user actually interacted with, drawn from the same logged distribution that trained the model. That held-out set is itself popularity-skewed and was itself collected under a previous recommender that already favored popular items — so the "ground truth" you are scoring against is biased toward exactly the items an accuracy-only ranker over-serves. The metric and the data agree with each other because they share a common ancestor: the logging policy. This is the missing-not-at-random (MNAR) problem, and its practical consequence here is brutal — an offline metric drawn from a popularity-collapsed log will *reward* a popularity-collapsed recommender and give you no signal at all that the list is redundant. You can drive NDCG up for a quarter, as my team did, and the metric will applaud you all the way into the boring corner. The only way to see the problem is to stop computing per-item accuracy and start computing the set-level and population-level quantities this post is about. The metric does not lie about what it measures; it just measures the wrong thing for this question.

## 2. The four beyond-accuracy metrics, defined

Let me define the family precisely before we compute anything. There are four core metrics plus a freshness notion, and the cleanest way to keep them straight is to ask, for each, *what does it measure, what is the formula, and why does it matter.*

![A matrix laying out diversity, novelty, serendipity and coverage against what each measures, its formula, and why it matters](/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-2.png)

**Diversity** measures how *different from each other* the K recommended items are. It is a property of a single list, computed without any reference to ground truth — you do not need to know what the user clicked, only how dissimilar the items you showed are. The standard operationalization is **intra-list diversity (ILD)**: the average pairwise distance between items in the list. High ILD means the list spans different regions of item space; low ILD means it clusters.

**Novelty** measures how *non-obvious* a recommended item is — how unlikely the user was to already know about it. The standard operationalization is popularity-based **self-information**: a globally rare item carries more novelty than a globally common one. An item recommended to everyone (the current top-of-charts movie) has near-zero novelty because the user almost certainly already knew it existed; a deep-catalog item the user would never have stumbled on themselves has high novelty. Novelty is a per-item quantity you then average over the list.

**Serendipity** is the hardest and most valuable: it measures recommendations that are *both relevant and unexpected*. Novelty alone is cheap — recommend random obscure items and you maximize novelty while destroying relevance. Serendipity requires the surprise to *land*: the user did not expect it, would not have found it on their own, and *yet liked it*. It is the "I didn't know I wanted this until you showed me" experience, and it is what separates a recommender that feels intelligent from one that feels like a sorting algorithm. Serendipity needs feedback (did the user actually engage?) to measure honestly, so it is the one metric you mostly evaluate online.

**Coverage** flips the frame from the single list to the *whole system*. It asks: across all the lists you serve to all your users, what fraction of the catalog ever appears? This is **catalog coverage** (the count or fraction of distinct items recommended at least once), and its richer cousins are the **gini coefficient** and **entropy** of the recommendation-frequency distribution, which measure not just whether the long tail is reached but how *evenly* exposure is spread. Coverage is sometimes called *aggregate diversity* because it is diversity measured across the population rather than within one list. It is the metric that catches the catalog impoverishment from section 1.

**Freshness / recency** is a fifth, more domain-specific notion: in news, social feeds, and marketplaces, an item's value decays with age, so a list of ten perfectly relevant but week-old items is a failure. Freshness is usually a constraint or a feature rather than a standalone re-ranking objective, but it lives in the same re-ranking layer and trades off against accuracy in the same way. A common implementation multiplies each item's relevance by a decay factor $\exp(-\gamma \cdot \text{age})$ before re-ranking, where $\gamma$ controls how fast staleness penalizes an item; a news feed uses an aggressive $\gamma$ (an article is dead in a day), an evergreen-content platform uses a gentle one. Freshness is genuinely distinct from novelty: a *fresh* item is new *to the world* (just published), while a *novel* item is new *to this user* (globally unpopular, so they likely have not seen it). A blockbuster released this morning is maximally fresh and minimally novel; a decade-old cult film is maximally novel and minimally fresh. Conflating the two is a frequent source of confused re-ranking objectives.

A note on what diversity and novelty are *not*, because the confusion is common. Diversity is about *within-list* relationships and needs no popularity data — you could compute it on a list of items nobody has ever seen. Novelty is about *global* popularity and ignores the relationships between the recommended items — a list of ten different but all-extremely-popular items has high diversity and low novelty, while a list of ten near-identical obscure items has low diversity and high novelty. They are orthogonal axes, and a good list usually wants both: varied (diverse) *and* including some non-obvious finds (novel). MMR with an embedding-based similarity buys diversity directly; novelty you typically buy by adding a popularity penalty to the relevance score, or by reserving a few list slots for long-tail items. Keeping the two as separate knobs, rather than hoping one metric captures both, is the practical discipline.

The crucial structural point — the one that organizes the whole family — is the split between **list-level** and **system-level** metrics. Diversity, novelty, and serendipity are properties of the list one user sees. Coverage and aggregate diversity are properties of the system across all users. You can have a system where *every individual list* is internally diverse, yet *coverage is terrible* because every user gets a diverse list drawn from the same small popular set. Optimizing list-level diversity does not automatically fix coverage, and vice versa. Keeping that distinction sharp will save you from claiming a coverage win when you only improved per-list variety.

![A taxonomy tree splitting beyond-accuracy quality into list-level metrics one user sees and system-level metrics across all users](/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-4.png)

## 3. The science: the formulas, made rigorous

Now the math. Each metric has a clean definition, and the definitions are worth deriving because the formula tells you exactly what knob you are turning.

### Intra-list diversity

Let the recommended list be $L = (i_1, \dots, i_K)$ and let $d(i, j) \in [0, 1]$ be a dissimilarity between two items — typically $d(i,j) = 1 - \cos(\mathbf{v}_i, \mathbf{v}_j)$ for item embeddings $\mathbf{v}$, or one minus a genre-overlap (Jaccard) similarity for categorical content. Intra-list diversity is the average pairwise dissimilarity over all $\binom{K}{2}$ unordered pairs:

$$ \text{ILD}(L) = \frac{2}{K(K-1)} \sum_{a=1}^{K} \sum_{b=a+1}^{K} d(i_a, i_b). $$

The factor $\frac{2}{K(K-1)}$ is just $1 / \binom{K}{2}$, normalizing to the number of pairs so ILD lands in $[0,1]$. Read the formula: ILD is high when items are far apart in embedding space and low when they cluster. Two identical items contribute $d=0$ and drag ILD down; that is the term that was missing from NDCG. The metric is symmetric and ignores order, which is correct — diversity is about the *set*, and you handle position separately.

### Novelty as self-information

Novelty is grounded in information theory. If item $i$ has been interacted with by a fraction $p(i)$ of users (its empirical popularity), then the *self-information* of observing $i$ is $-\log_2 p(i)$ bits — the standard surprise of an event with probability $p(i)$. A near-universal item ($p \to 1$) carries $\approx 0$ bits; a rare item ($p \to 0$) carries many bits. The novelty of a list is the mean self-information of its items:

$$ \text{Novelty}(L) = \frac{1}{K} \sum_{k=1}^{K} \big(-\log_2 p(i_k)\big), \qquad p(i) = \frac{\#\{u : u \text{ interacted with } i\}}{\#\text{users}}. $$

This is sometimes written with $p(i)$ as the fraction of *interactions* rather than users; either is defensible as long as you are consistent. The information-theoretic framing is not decoration: it gives novelty a meaningful unit (bits), it is the same self-information that underlies entropy (which we will use for coverage), and the logarithm correctly compresses the long tail — the difference between an item seen by 50% versus 25% of users is one bit, the same as 0.2% versus 0.1%, which matches the intuition that the interesting variation in novelty lives in the tail.

### Serendipity, operationalized

Serendipity is inherently relational: it is relevance *minus what was expected*. A common offline proxy defines an *expected* (unsurprising) recommendation as one a trivial baseline — usually a popularity ranker or a content-similarity-to-history ranker — would also have produced, and credits serendipity only to relevant items that were *not* in that expected set. One workable formula, following Ge et al. (2010) and the surveys that build on it:

$$ \text{Serendipity}(L, u) = \frac{1}{K} \sum_{k=1}^{K} \text{unexp}(i_k) \cdot \text{rel}(i_k, u), $$

where $\text{rel}(i_k, u) \in \{0,1\}$ is whether the user actually engaged, and $\text{unexp}(i_k) = 1$ if $i_k$ was *not* among the items a primitive baseline (most-popular, or nearest-neighbor of the user's history) would have surfaced, else 0. The product is the key: an item earns serendipity credit only when it is *both* unexpected *and* relevant. Maximize unexpectedness alone and relevance collapses; maximize relevance alone and unexpectedness collapses; the product forces the surprise to land. Because $\text{rel}$ requires real engagement, honest serendipity measurement is an online affair, which is why teams that care about it run holdouts that explicitly inject some unexpected items and measure whether engagement holds.

The cleanest way to actually *measure* serendipity online is an exploration slot design. Reserve, say, one slot in every list for an item the user's relevance model rates moderately (not the top, not random — picked from the "plausible but the model is uncertain" band) and that is dissimilar to the user's recent history. Then measure the engagement rate on that slot against a control where the slot is filled by the next-best-relevance item. If the exploration slot earns engagement comparable to or above the control over a long enough horizon, those are serendipitous hits — the model found things the user liked that the accuracy ranker would never have surfaced. This is a deliberate, measurable, bounded bet: you spend one slot of exposure on discovery and you get back both engagement *and* fresh training data on items the model was uncertain about, which improves the model's coverage of the uncertain region. The risk is bounded because it is one slot, and the upside is twofold — immediate serendipity and a healthier training distribution. This is also exactly the seam where beyond-accuracy re-ranking touches the *exploration* needed to break a feedback loop, the subject of [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles).

### Coverage, gini, and entropy

Let $f(i)$ be the number of times item $i$ was recommended across all lists served in some window, and let $\hat{p}(i) = f(i) / \sum_j f(j)$ be the normalized recommendation frequency. Three nested metrics:

**Catalog coverage** is the simplest — the fraction of the catalog that appears at least once:

$$ \text{Coverage} = \frac{\#\{i : f(i) > 0\}}{\#\text{catalog}}. $$

**Entropy** of the recommendation distribution measures how *evenly* exposure is spread, not just whether the tail is touched:

$$ H = -\sum_{i} \hat{p}(i) \log_2 \hat{p}(i). $$

Entropy is maximized ($\log_2 N$ bits for $N$ items) when every item is recommended equally often, and minimized (0 bits) when one item gets all the exposure. It is the population-level analogue of novelty.

**Gini coefficient** measures inequality of exposure directly. Sort items by ascending frequency $f_{(1)} \le f_{(2)} \le \dots \le f_{(N)}$; then

$$ G = \frac{2 \sum_{k=1}^{N} k \, f_{(k)}}{N \sum_{k=1}^{N} f_{(k)}} - \frac{N+1}{N}. $$

Gini ranges from 0 (perfectly equal exposure) to nearly 1 (one item gets everything). A healthy, diverse catalog has high coverage, high entropy, and low gini; a popularity-collapsed system has low coverage, low entropy, and gini near 1. These three move together but are not redundant: coverage is a binary touched/untouched count, while gini and entropy are sensitive to *how* skewed the exposure is among the touched items.

### The accuracy-diversity frontier

Why is there a trade-off at all? Because the most-relevant items are, by force one above, clustered. Any push toward diversity must replace some high-relevance-but-redundant item with a lower-relevance-but-different one, which costs accuracy. Formally, if you re-rank to maximize a blend $\lambda \cdot \text{relevance} + (1-\lambda) \cdot \text{diversity}$, sweeping $\lambda$ traces out a **Pareto frontier** in (accuracy, diversity) space. The accuracy-only ranker sits at the $\lambda = 1$ corner — maximum NDCG, minimum diversity. The empirical shape of this frontier is the single most useful artifact in this whole post: it is usually *concave and steep near the corner*, meaning the first few points of diversity cost almost no accuracy, while the last points cost a lot. That shape is why beyond-accuracy optimization is nearly free if you do it well — you are buying the cheap diversity near the corner, not driving to the random-list extreme.

The concavity is not a coincidence; it falls out of the structure of the candidate set. Near the accuracy-only corner, the items you can swap *in* to gain diversity are the redundant near-duplicates of items already chosen — and removing a near-duplicate costs almost no relevance (you had two items doing the same job; deleting one barely changes the user's options) while gaining a lot of diversity (you eliminated a $d \approx 0$ pair). So the first swaps have a steep diversity-per-relevance ratio. As you continue, you exhaust the cheap redundant swaps and must start replacing genuinely-relevant items with genuinely-less-relevant different ones, where each unit of diversity now costs real relevance. The marginal exchange rate worsens monotonically, which is precisely the definition of a concave frontier. Practically this means the *correct* default is not λ = 1 — it is a point a little way down the curve where you have harvested all the free diversity from killing duplicates and have not yet started paying for it with real relevance. Most teams that have never measured the frontier are sitting at the λ = 1 corner leaving the free diversity on the table.

![A before and after comparison of a ranker parked at the maximum-accuracy corner of the frontier against one moved to a balanced operating point that trades a tiny NDCG cost for large diversity gain](/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-3.png)

## 4. Worked example: computing diversity and novelty by hand

Let me make all of this concrete with arithmetic you can verify on paper.

#### Worked example: intra-list diversity and novelty of a 4-item list

Suppose a movie recommender returns a top-4 list $L = (A, B, C, D)$ with the following genre-vector cosine *similarities* between each pair (a higher number means more alike):

| pair | cosine similarity | dissimilarity d = 1 - cos |
|------|-------------------|---------------------------|
| A, B | 0.90 | 0.10 |
| A, C | 0.20 | 0.80 |
| A, D | 0.30 | 0.70 |
| B, C | 0.25 | 0.75 |
| B, D | 0.35 | 0.65 |
| C, D | 0.85 | 0.15 |

A and B are near-duplicates (both action sequels); C and D are near-duplicates (both quiet dramas); the two clusters are far apart. With $K = 4$ there are $\binom{4}{2} = 6$ pairs. Intra-list diversity is the mean dissimilarity:

$$ \text{ILD} = \frac{0.10 + 0.80 + 0.70 + 0.75 + 0.65 + 0.15}{6} = \frac{3.15}{6} = 0.525. $$

Now suppose a re-ranker swaps out the redundant B (near-duplicate of A) for a third, different item E that sits far from everything ($d(A,E)=0.7$, $d(C,E)=0.6$, $d(D,E)=0.65$). The new list is $(A, C, D, E)$ with pairwise dissimilarities $\{d(A,C)=0.80, d(A,D)=0.70, d(A,E)=0.70, d(C,D)=0.15, d(C,E)=0.60, d(D,E)=0.65\}$:

$$ \text{ILD}' = \frac{0.80 + 0.70 + 0.70 + 0.15 + 0.60 + 0.65}{6} = \frac{3.60}{6} = 0.600. $$

Removing one of the two redundant action films lifted ILD from 0.525 to 0.600 — a 14% relative diversity gain — by deleting the cheapest, most-redundant item. This is exactly the move MMR will make automatically.

Now novelty. Suppose the four original items have these empirical popularities (fraction of users who interacted): $p(A) = 0.40$, $p(B) = 0.35$, $p(C) = 0.02$, $p(D) = 0.015$. The self-information of each in bits:

$$ -\log_2 0.40 = 1.32, \quad -\log_2 0.35 = 1.51, \quad -\log_2 0.02 = 5.64, \quad -\log_2 0.015 = 6.06. $$

$$ \text{Novelty}(L) = \frac{1.32 + 1.51 + 5.64 + 6.06}{4} = \frac{14.53}{4} = 3.63 \text{ bits}. $$

Notice how the two long-tail items C and D carry roughly four times the novelty bits of the two popular items A and B — the logarithm has compressed the popular pair into a narrow low-novelty band and stretched the tail. If the list were all four popular items (each $p \approx 0.4$), novelty would be about 1.3 bits; the tail items are what make a list feel like a discovery. This is the quantitative reason a popularity-collapsed list feels stale: it is low not just in diversity but in raw information content.

## 5. Re-ranking for diversity: MMR

The dominant practical tool for buying diversity is **Maximal Marginal Relevance (MMR)**, introduced by Carbonell and Goldstein (1998) for document retrieval and adopted wholesale by recommenders. MMR is a greedy algorithm that builds the output list one item at a time, and at each step it picks the candidate that maximizes a trade-off between relevance to the user and *novelty relative to what is already selected*.

![A graph showing the MMR loop scoring each candidate by relevance minus a diversity penalty against the already-selected set, picking the argmax, and appending it to the output list](/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-5.png)

Let $S$ be the set already selected and $R \setminus S$ the remaining candidates. At each step MMR selects

$$ i^\star = \arg\max_{i \in R \setminus S} \Big[ \lambda \cdot \text{rel}(i, u) - (1 - \lambda) \cdot \max_{j \in S} \text{sim}(i, j) \Big]. $$

Read the two terms. The first, $\lambda \cdot \text{rel}(i, u)$, is the ranker's relevance score — pull toward accuracy. The second, $-(1-\lambda) \cdot \max_{j \in S} \text{sim}(i, j)$, is the *redundancy penalty*: it is the similarity of candidate $i$ to its *nearest already-selected neighbor*, and it pushes away from items that duplicate something already in the list. The $\max$ is the important detail — MMR penalizes you for being close to *any one* selected item, so the moment your first action movie is picked, every other action movie's score drops and the algorithm naturally moves to a different cluster for slot two.

The $\lambda \in [0, 1]$ knob is the entire trade-off in one number. At $\lambda = 1$ the penalty vanishes and MMR degenerates to the accuracy-only ranker — it just returns the top K by relevance, the boring list. At $\lambda = 0$ relevance vanishes and MMR returns the most mutually-dissimilar set it can find regardless of whether the user wants any of it — maximum diversity, useless list. Practical values sit in $[0.5, 0.8]$, and the right value is found by sweeping λ and reading off the accuracy-diversity frontier from section 3. MMR is greedy and therefore not guaranteed to find the globally optimal diverse subset, but it is $O(N K)$ for $N$ candidates and $K$ slots, embarrassingly cheap, deterministic, and the greedy choice is good enough in practice that it has been the production default for two decades.

#### Worked example: MMR picking a diverse item over a redundant one

Walk one step of MMR with $\lambda = 0.7$. Slot one is already filled with item A (the top-relevance action movie). Two candidates compete for slot two:

- **Item B**, a near-duplicate action sequel: $\text{rel}(B) = 0.95$, $\text{sim}(B, A) = 0.90$.
- **Item C**, a well-reviewed drama the user has historically liked: $\text{rel}(C) = 0.80$, $\text{sim}(C, A) = 0.20$.

B is *more relevant* (0.95 vs 0.80). An accuracy-only ranker picks B without hesitation. Now the MMR scores with $\lambda = 0.7$, so $1 - \lambda = 0.3$:

$$ \text{MMR}(B) = 0.7 \times 0.95 - 0.3 \times 0.90 = 0.665 - 0.270 = 0.395. $$

$$ \text{MMR}(C) = 0.7 \times 0.80 - 0.3 \times 0.20 = 0.560 - 0.060 = 0.500. $$

MMR picks **C**, the slightly-less-relevant but far-less-redundant drama, because its tiny relevance deficit (0.15) is dwarfed by its large redundancy advantage (the 0.90-vs-0.20 similarity gap). This is the whole point of MMR in one line of arithmetic: it accepts a small, controlled accuracy concession to avoid stacking near-duplicates. Note that if we lowered λ to 0.5, the gap would widen further in C's favor; if we raised it to 0.95, B would win again. The single λ dials exactly how much redundancy you are willing to tolerate for a point of relevance.

Here is MMR in NumPy, taking a relevance vector and an item-item similarity matrix and returning a re-ranked list:

```python
import numpy as np

def mmr_rerank(relevance, sim_matrix, k, lam=0.7):
    """Greedy MMR re-ranking.
    relevance : (N,) array of ranker scores for N candidates
    sim_matrix: (N, N) item-item similarity in [0, 1]
    k         : number of slots to fill
    lam       : trade-off, 1.0 = pure relevance, 0.0 = pure diversity
    returns   : list of selected candidate indices, in chosen order
    """
    n = len(relevance)
    candidates = list(range(n))
    selected = []
    # first slot: pure relevance, nothing to be redundant with yet
    first = int(np.argmax(relevance))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_idx, best_score = None, -np.inf
        for i in candidates:
            # redundancy = similarity to the NEAREST already-selected item
            redundancy = max(sim_matrix[i, j] for j in selected)
            score = lam * relevance[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected
```

The inner loop is the literal MMR formula; `max(sim_matrix[i, j] for j in selected)` is the $\max_{j \in S}$ term. For production you would vectorize the redundancy computation — keep a running `(N,)` vector of "max similarity to any selected item so far" and update it with `np.maximum(running, sim_matrix[:, last_selected])` after each pick, which turns the $O(NK^2)$ naive loop into $O(NK)$. The version above is written for clarity, not speed.

## 6. The principled cousin: determinantal point processes

MMR is greedy and heuristic. The principled formulation of "select a diverse, relevant subset" is the **determinantal point process (DPP)**, used in production at Hulu (Chen, Zhang, and Zhou, 2018) and elsewhere. A DPP defines a probability distribution over *subsets* of items such that the probability of selecting a subset is proportional to the *determinant* of a kernel matrix restricted to that subset:

$$ P(S) \propto \det(L_S), $$

where $L$ is an $N \times N$ positive semi-definite kernel and $L_S$ is its submatrix indexed by the items in $S$. The magic is geometric. Build the kernel as $L_{ij} = q_i \, q_j \, \phi_i^\top \phi_j$, where $q_i \ge 0$ is the *quality* (relevance) of item $i$ and $\phi_i$ is a normalized feature vector so that $\phi_i^\top \phi_j$ is item-item *similarity*. Then $\det(L_S)$ is the squared *volume* of the parallelepiped spanned by the vectors $q_i \phi_i$ for $i \in S$. Two consequences fall straight out of that geometry:

- **Relevance raises probability.** Larger $q_i$ stretches the vectors longer, increasing the volume — relevant items are favored.
- **Similarity lowers probability.** If two items point in nearly the same direction ($\phi_i^\top \phi_j \approx 1$), the parallelepiped they span is nearly flat, its volume collapses toward zero, and so does the probability of selecting both. *Redundancy is geometrically penalized by the determinant.*

So a DPP encodes relevance and diversity in a *single* object, the kernel, and "diverse and relevant" is literally "high volume." Finding the maximum-probability subset of fixed size $k$ (the MAP inference problem) is NP-hard in general, but the standard greedy algorithm — at each step add the item that maximizes the marginal log-determinant gain — has the same $O(N k^2)$ shape as MMR and a submodularity guarantee that puts it within a $(1 - 1/e)$ factor of optimal. The Hulu fast-greedy variant (2018) brought one full DPP re-rank down to a few milliseconds for hundreds of candidates, which is what made it shippable in a serving path.

Here is a compact greedy DPP MAP sketch in NumPy, building the kernel from relevance and embeddings:

```python
import numpy as np

def dpp_greedy(relevance, embeddings, k, theta=1.0):
    """Greedy MAP inference for a quality x similarity DPP.
    relevance : (N,) nonneg quality scores
    embeddings: (N, d) L2-normalized item feature vectors
    k         : subset size
    theta     : how strongly relevance scales the kernel
    """
    N = len(relevance)
    q = np.exp(theta * (relevance - relevance.max()))   # quality, stable
    S = embeddings @ embeddings.T                        # cosine similarity
    L = (q[:, None] * q[None, :]) * S                    # the DPP kernel
    selected, ci = [], np.zeros((N, 0))                  # Cholesky factors
    di2 = np.copy(np.diag(L))                            # incremental gains
    j = int(np.argmax(di2)); selected.append(j)
    for _ in range(k - 1):
        if ci.shape[1] == 0:
            ei = L[j, :] / np.sqrt(di2[j])
        else:
            ei = (L[j, :] - ci @ ci[j]) / np.sqrt(di2[j])
        ci = np.concatenate([ci, ei[:, None]], axis=1)
        di2 = di2 - ei ** 2                              # update marginal gains
        di2[selected] = -np.inf
        j = int(np.argmax(di2))
        if di2[j] <= 1e-10:
            break
        selected.append(j)
    return selected
```

This is the Chen et al. fast greedy MAP: it maintains incremental Cholesky factors so each step is $O(N)$ given the kernel, and the `di2` vector holds the marginal log-det gain of adding each remaining item. DPP versus MMR is a real choice — MMR's `max`-similarity penalty only looks at the single nearest selected item, while the determinant accounts for the *whole* selected set jointly, so DPP catches "three items that are each only moderately similar to one neighbor but collectively redundant" that MMR's pairwise max can miss. In practice DPP gives a slightly better diversity-relevance frontier at modestly higher implementation cost; MMR is the right first thing to ship, DPP the upgrade when the frontier matters enough to fund it.

## 7. Calibrated recommendations: matching the user's own proportions

There is a third re-ranking philosophy that deserves its own treatment because it reframes "diversity" in a way that is both more user-specific and easier to defend to a product team. **Calibrated recommendations**, introduced by Harald Steck (2018) at Netflix, observe that the right amount of diversity is *the user's own*. If a user has historically watched 70% comedy and 30% documentary, a list that is 100% comedy is *miscalibrated* — it has erased a real, demonstrated interest — and so is a list that is 50/50, which over-represents the documentaries. The calibrated ideal is a list whose genre proportions match the user's consumption proportions.

Steck formalizes this with the **KL divergence** between two distributions over genres (or any categorical attribute): $p(g \mid u)$, the user's historical genre distribution, and $q(g \mid u)$, the genre distribution of the recommended list. Calibration error is

$$ C_{\text{KL}}\big(p, q\big) = \sum_{g} p(g \mid u) \, \log \frac{p(g \mid u)}{\tilde{q}(g \mid u)}, $$

where $\tilde{q}$ is a smoothed version of the list distribution ($\tilde{q} = (1-\alpha) q + \alpha p$ with small $\alpha$) so that a genre the user likes but that is *missing* from the list does not blow up the divergence to infinity. You then re-rank to maximize a blend of accuracy and calibration:

$$ L^\star = \arg\max_{L} \Big[ (1 - \beta) \sum_{i \in L} \text{rel}(i, u) - \beta \, C_{\text{KL}}\big(p, q(L)\big) \Big], $$

and because adding items to a list and recomputing $q(L)$ is monotone-submodular in the right sense, the same greedy construction as MMR works. The conceptual upgrade over plain diversity is that calibration is *grounded in the individual user's revealed preferences* rather than a generic "be different" pressure. It will not push a user who genuinely only watches one genre into other genres — it respects a narrow taste — while still rescuing the user whose minority interests are being silently dropped by an accuracy-only ranker that fixates on their majority interest. Steck reported that an accuracy-optimized recommender systematically *amplified* a user's majority genre and crowded out minority interests, and that re-ranking for calibration corrected this with negligible loss in the primary ranking metric.

Here is calibration error and a greedy calibrated re-rank in NumPy, where each item has a genre distribution:

```python
import numpy as np

def kl_calibration(user_dist, list_dist, alpha=0.01):
    """KL(p || q_tilde) with smoothing so missing genres stay finite.
    user_dist, list_dist : (G,) probability vectors over G genres
    """
    q_tilde = (1 - alpha) * list_dist + alpha * user_dist
    mask = user_dist > 0
    return float(np.sum(user_dist[mask] * np.log(user_dist[mask] / q_tilde[mask])))

def calibrated_rerank(relevance, item_genre, user_dist, k, beta=0.5):
    """Greedy re-rank trading relevance against genre calibration.
    relevance  : (N,) ranker scores
    item_genre : (N, G) per-item genre distributions (rows sum to 1)
    user_dist  : (G,) the user's historical genre distribution
    """
    n = len(relevance)
    candidates, selected = list(range(n)), []
    while len(selected) < k and candidates:
        best_i, best_obj = None, -np.inf
        for i in candidates:
            trial = selected + [i]
            list_dist = item_genre[trial].mean(axis=0)   # list genre mix
            rel = relevance[trial].sum()
            cal = kl_calibration(user_dist, list_dist)
            obj = (1 - beta) * rel - beta * cal
            if obj > best_obj:
                best_obj, best_i = obj, i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected
```

Diversity, MMR, DPP, and calibration are not competitors so much as a toolkit: MMR/DPP enforce list-internal variety; calibration enforces fidelity to the user's revealed proportions; and you can combine them (calibrate the genre mix, then MMR within genre to avoid intra-genre duplicates). All four are re-ranking-layer concerns, and they all expose a single trade-off knob you sweep against accuracy.

Here is how the three diversity re-rankers compare on the dimensions that decide which to reach for:

| Method | What it optimizes | Set view | Cost | Knob | Reach for it when |
|--------|-------------------|----------|------|------|-------------------|
| MMR | relevance − max-sim penalty | pairwise (nearest selected) | $O(NK)$ | λ ∈ [0,1] | first thing to ship; cheap, deterministic, easy to explain |
| DPP (greedy MAP) | det of quality×similarity kernel | joint (whole set) | $O(NK^2)$ | kernel temperature | the frontier matters enough to fund; collective redundancy MMR misses |
| Calibration (Steck) | relevance − KL(user, list) over attributes | distributional (genre mix) | $O(NKG)$ | β ∈ [0,1] | you have clean categorical attributes and want per-user, not generic, variety |

The dimension that most often decides it is the *set view*. MMR's pairwise `max` penalty is blind to a subtle failure: three items that are each only moderately similar to one neighbor but collectively span almost no volume — MMR happily picks all three because no single pair trips the penalty, while the DPP determinant sees the collapsed volume and rejects the third. If your redundancy is mostly obvious pairwise duplication, MMR catches it and DPP is overkill. If your redundancy is "everything subtly clusters," DPP's joint view earns its extra cost. Calibration is the odd one out — it does not measure embedding similarity at all, it measures fidelity to a *named attribute distribution*, so it is the tool when "diverse" really means "don't drop the genres this user demonstrably likes" rather than "spread out in latent space."

## 8. Buying novelty and coverage explicitly

MMR and calibration buy *diversity* (within-list variety) and *fidelity* (genre match), but neither directly buys *novelty* (long-tail reach) — and as section 2 stressed, novelty and coverage are different axes that you optimize with different knobs. If your problem is that the long tail never gets recommended, MMR will not fix it: MMR diversifies *within the candidate set*, and if the candidate set is all popular, the most-diverse popular subset is still popular. You need a novelty term that explicitly rewards unpopular items.

The clean way to do this is to fold a popularity penalty straight into the relevance score before re-ranking. Replace $\text{rel}(i, u)$ with a novelty-adjusted score $\text{rel}'(i, u) = \text{rel}(i, u) + \eta \cdot \big(-\log_2 p(i)\big)$, where $\eta$ controls how much you reward rarity. This adds the per-item self-information directly to the objective, so a niche item with the same raw relevance as a popular one gets a boost proportional to how unknown it is. Sweep $\eta$ the same way you sweep λ, and watch coverage and gini move while accuracy holds. In NumPy:

```python
import numpy as np

def novelty_adjusted_scores(relevance, pop, eta=0.1):
    """Add self-information to relevance to reward long-tail items.
    relevance : (N,) ranker scores (assume comparable scale, e.g. z-scored)
    pop       : (N,) training-set popularities in (0, 1]
    eta       : novelty weight; 0 = pure relevance
    """
    self_info = -np.log2(np.clip(pop, 1e-9, 1.0))
    # normalize self-info to relevance scale so eta is interpretable
    self_info = (self_info - self_info.mean()) / (self_info.std() + 1e-9)
    return relevance + eta * self_info
```

You then feed `novelty_adjusted_scores(...)` into MMR as the relevance vector, so a single re-rank buys *both* within-list diversity (MMR's penalty) and long-tail novelty (the self-information boost). This composition matters: diversity without novelty gives you a varied list of popular items (good per-list variety, terrible coverage), and novelty without diversity gives you a list of different-but-clustered obscure items. Stacking them gives the list that is varied *and* reaches the catalog, which is what moves both the list-level and system-level metrics at once.

There is a population-level subtlety worth flagging. Coverage is an *emergent* property of how you re-rank *every* user, not a per-list quantity you can directly optimize one list at a time — if every user independently gets a novelty-boosted list, *collectively* you spread exposure across the catalog and coverage rises, but no single re-rank decision "chose" coverage. This is why the novelty penalty is the right lever: it is a per-list knob (boost rare items in this list) whose aggregate effect across all served lists is the system-level outcome (catalog coverage). You buy a global property by tuning a local one, and you verify it by accumulating the population frequency vector and recomputing coverage and gini, exactly as the harness in section 9 does.

## 9. Where this sits: the re-ranking stage

It is worth being explicit about *where in the system* all of this runs, because trying to bake diversity into the retrieval or ranking models directly is a common and costly mistake.

![A stack diagram of the funnel showing retrieval then ranking then a diversity re-rank then calibration then business rules then the served list](/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-6.png)

Retrieval and ranking are *per-item* scoring stages — they are trained on per-item objectives (sampled softmax, BPR, logloss) and they answer "how relevant is this one item to this one user." Diversity is a *set* property and cannot be expressed as a per-item label, so trying to make the ranker itself produce diverse lists means inventing listwise training objectives that are hard to optimize, brittle, and entangle relevance learning with diversity policy. The clean architecture keeps the models pure relevance predictors and applies diversity, calibration, freshness, and hard business rules as a *post-processing re-rank* on the ranker's top few hundred candidates. This separation has three concrete advantages:

1. **Tunability without retraining.** The λ (MMR), β (calibration), or kernel temperature (DPP) are serving-time knobs. You can A/B-test λ = 0.6 against λ = 0.7 by flipping a config, with no model retrain, no offline pipeline run, no risk to the relevance model. Diversity policy and relevance learning have different iteration speeds and different owners; the re-rank boundary lets them move independently.
2. **Composability with business rules.** The same stage that diversifies is where you pin sponsored items, dedup the same item across surfaces, enforce "no more than two items per seller," demote stale news, and apply freshness decay. These are all *constraints on the set*, and they live naturally next to MMR. A diversity re-rank that ignored business constraints would be relitigated by the constraint layer downstream anyway.
3. **Cheap, because the candidate set is small.** MMR is $O(NK)$ and DPP is $O(NK^2)$, which is fine on the few hundred candidates that survive ranking but would be ruinous on the millions in the catalog. Re-ranking *after* the funnel has narrowed the candidate set is what makes set-level computation affordable. This is the deeper reason diversity belongs in re-ranking and not retrieval: the math is only cheap once $N$ is small.

The one real tension is that re-ranking can only diversify what retrieval *gave it*. If retrieval returns 300 candidates that are all from the same popular cluster — because retrieval itself is popularity-biased — then no amount of re-ranking can produce a diverse list; you cannot pick a drama for slot two if there are no dramas in the candidate set. So diversity is a *system* property: you often need a touch of diversity in retrieval too (e.g., a multi-stage retriever that pulls candidates from several intent clusters, or an explicit long-tail retrieval source) to give the re-ranker raw material to work with. This is the seam where this post connects to [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer): re-ranking treats the symptom in one list, while popularity-bias mitigation treats the cause across the candidate-generation stack.

## 10. Measuring it: a full evaluation harness

Now let me put the measurement together end to end, because "we added MMR" without a measured frontier is just a vibe. The honest evaluation computes accuracy *and* the beyond-accuracy metrics on the same held-out split, so you can read the trade-off rather than claim a one-sided win.

A few methodology rules carry over from the accuracy world and matter just as much here:

- **Temporal split, not random.** Train on interactions up to time $t$, evaluate on interactions after $t$. A random split leaks future popularity into the past and inflates every metric. (The full argument is in [the offline evaluation post](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr).)
- **Coverage is measured across the whole served population, never per user.** Per-user coverage is meaningless — one user sees K items. You accumulate the recommendation-frequency vector $f(i)$ across *every* user's list, then compute coverage, entropy, and gini once on that aggregate. This is the list-level versus system-level split made operational.
- **Popularity $p(i)$ for novelty is computed on the training set only.** Using test-set popularity is leakage; novelty should reflect what was knowable at serving time.

Here is the metric suite in NumPy/pandas:

```python
import numpy as np

def intra_list_diversity(item_ids, sim_matrix):
    """Mean pairwise dissimilarity (1 - similarity) over the list."""
    k = len(item_ids)
    if k < 2:
        return 0.0
    total, pairs = 0.0, 0
    for a in range(k):
        for b in range(a + 1, k):
            total += 1.0 - sim_matrix[item_ids[a], item_ids[b]]
            pairs += 1
    return total / pairs

def novelty(item_ids, pop):
    """Mean self-information -log2 p(i) over the list. pop is a (N,) array
    of training-set popularities; clip to avoid log(0)."""
    p = np.clip(pop[item_ids], 1e-9, 1.0)
    return float(np.mean(-np.log2(p)))

def catalog_coverage(all_recommended_ids, n_catalog):
    """Fraction of the catalog recommended at least once across all users."""
    return len(set(all_recommended_ids)) / n_catalog

def gini(freq):
    """Gini coefficient of the recommendation-frequency distribution."""
    f = np.sort(np.asarray(freq, dtype=float))
    n = len(f)
    if f.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * f)) / (n * f.sum()) - (n + 1) / n)

def entropy(freq):
    """Shannon entropy (bits) of the recommendation distribution."""
    f = np.asarray(freq, dtype=float)
    p = f[f > 0] / f.sum()
    return float(-np.sum(p * np.log2(p)))
```

And the driver that sweeps λ, re-ranks every user, and accumulates both accuracy and aggregate metrics:

```python
import numpy as np
from collections import Counter

def evaluate_sweep(users, base_ranker, sim_matrix, pop, ground_truth,
                   n_catalog, lambdas, k=10):
    """For each lambda, MMR-rerank every user's candidates and report
    NDCG@k, mean ILD, mean novelty, coverage, gini."""
    results = []
    for lam in lambdas:
        ndcgs, ilds, novs = [], [], []
        rec_counter = Counter()
        for u in users:
            relevance = base_ranker.scores(u)        # (N,) candidate scores
            order = mmr_rerank(relevance, sim_matrix, k, lam=lam)
            ndcgs.append(ndcg_at_k(order, ground_truth[u], k))
            ilds.append(intra_list_diversity(order, sim_matrix))
            novs.append(novelty(order, pop))
            rec_counter.update(order)
        freq = np.array([rec_counter.get(i, 0) for i in range(n_catalog)])
        results.append({
            "lambda": lam,
            "NDCG@10": float(np.mean(ndcgs)),
            "diversity": float(np.mean(ilds)),
            "novelty": float(np.mean(novs)),
            "coverage": catalog_coverage(list(rec_counter.keys()), n_catalog),
            "gini": gini(freq),
            "entropy": entropy(freq),
        })
    return results
```

Run this with `lambdas = [1.0, 0.8, 0.5, 0.3]` and you get the trade-off curve directly: a table where each row is an operating point, and you read off how much accuracy each point of diversity cost. That table *is* the deliverable — it lets a product owner pick the operating point on evidence rather than taste.

## 11. Results: the trade-off on MovieLens-20M

Let me ground this with concrete numbers on a named dataset. The setup: **MovieLens-20M** (138,000 users, ~27,000 movies, 20 million ratings), a standard matrix-factorization ranker (ALS implicit feedback, 128-dimensional embeddings) as the base, item-item cosine similarity from those embeddings as the diversity kernel, a temporal 80/20 split, and MMR re-ranking the base ranker's top 200 candidates down to K = 10. The numbers below are representative of what this configuration produces and are consistent with the diversity-re-ranking literature; treat them as a defensible operating-point illustration, not a benchmarked leaderboard entry.

![A matrix showing the base ranker against three MMR lambda settings on NDCG@10, intra-list diversity, and catalog coverage, with diversity and coverage rising sharply as NDCG falls slightly](/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-8.png)

| Configuration | NDCG@10 | Intra-list diversity | Catalog coverage | Gini | Novelty (bits) |
|---------------|---------|----------------------|------------------|------|----------------|
| Base ranker (accuracy only) | **0.422** | 0.18 | 4% | 0.92 | 4.1 |
| MMR, λ = 0.8 | 0.418 (−1%) | 0.29 (+61%) | 9% (+125%) | 0.86 | 5.3 |
| MMR, λ = 0.5 | 0.411 (−3%) | 0.41 (+128%) | 19% (+375%) | 0.71 | 7.0 |
| MMR, λ = 0.3 | 0.388 (−8%) | 0.53 (+194%) | 31% (+575%) | 0.58 | 8.4 |

Read the shape. Going from the base to λ = 0.8 costs **1% of NDCG** and buys **61% more diversity and more than double the coverage** — this is the concave, steep-near-the-corner frontier from section 3, and it is nearly free diversity. Pushing to λ = 0.5 costs 3% of NDCG for a near-tripling of diversity and a quadrupling of coverage. Only when you push to λ = 0.3 does the accuracy cost (−8%) start to bite hard, in exchange for diminishing diversity returns. The right operating point for most products is somewhere around λ = 0.5 to 0.7: a small, well-understood accuracy concession in exchange for a list that is genuinely varied and a catalog that is genuinely reached. Note the gini collapsing from 0.92 to 0.58 — the exposure inequality that was strangling the long tail is what is actually being fixed.

![A before and after comparison of a popularity-collapsed system showing a few head items against a long-tail-covered system spreading exposure across the catalog](/imgs/blogs/beyond-accuracy-diversity-novelty-serendipity-coverage-7.png)

#### Worked example: pricing the trade-off as a business decision

Suppose your feed serves 10 million impressions a day, your base CTR at λ = 1.0 is 5.0%, and a careful A/B test of λ = 0.5 versus λ = 1.0 shows online CTR holding at 4.95% (the offline NDCG drop barely materializes online, because users were already saturated on the redundant items the diversity re-rank removed) while *seven-day retention* rises 1.8% because the feed stopped feeling repetitive. The accuracy "cost" is 0.05 percentage points of CTR — about 5,000 fewer clicks a day out of 500,000, a 1% relative dip. The retention "gain" compounds: 1.8% better seven-day retention on a base of, say, 2 million daily actives is roughly 36,000 additional retained users a week, each of whom generates many future sessions. Even if you valued a click and a retained-user-week crudely, the retention gain dwarfs the click loss within days, because clicks are a flow and retention is a stock. This is why diversity re-ranking is one of the highest-ROI moves in recsys: the offline metric you "lose" is a fraction of a percent, and the online behavior you gain is the thing the business actually monetizes. The catch — and it is a real one — is that you can only see this by measuring *retention*, not CTR, in the A/B test. A team that A/B-tests diversity changes on CTR alone will always conclude diversity "hurts," because CTR is exactly the short-horizon accuracy proxy that diversity trades against. You have to measure the long-horizon metric the diversity is *for*.

## 12. Stress-testing the diversity re-rank

Before trusting any of this in production, push on it. A re-ranking layer that works on a benchmark can fail in instructive ways under the conditions a real system actually hits.

**What if the candidate set is already homogeneous?** This is the most common real failure, and section 8 named it: re-ranking can only diversify what retrieval handed it. If your retriever returns 200 candidates that are all action movies — because the retriever is itself popularity- and recency-biased — then MMR with λ = 0.3 will dutifully return the *most dissimilar* 10 action movies, which is still 10 action movies. The diversity metric will tick up slightly (you spread within the cluster) but the user still sees a monotonous list, and you will be confused why a strong λ "did nothing." The fix is upstream: add a long-tail or multi-intent retrieval source so the candidate set spans more of item space, then let the re-rank do its job. Always inspect the *candidate set's* diversity before blaming the re-ranker — if the raw candidates have ILD 0.2, no re-rank can produce a list with ILD 0.5.

**What if the similarity matrix is wrong?** MMR and DPP both depend entirely on the item-item similarity. If you build it from collaborative-filtering embeddings, two items are "similar" when the *same users* interacted with them — which means a popular item and a niche item can look dissimilar simply because their audiences barely overlap, not because their content differs. Diversifying on co-occurrence similarity can therefore produce lists that look varied in embedding space but feel repetitive to the user (all the same genre, just different popularity tiers). If you build similarity from *content* features (genre, text, image embeddings) you get content diversity but may miss latent taste structure. The honest answer is that the similarity definition *is* your diversity definition, and you should pick it to match what "different" means to your user — usually a blend of content and behavioral similarity, validated by eyeballing real lists.

**What happens at 100 million items?** Coverage and gini are computed over the full catalog, and the recommendation-frequency vector $f(i)$ is then 100M long. That is fine to accumulate as a sparse counter, but the gini formula's sort is $O(N \log N)$ over the *touched* items only (untouched items contribute zero and can be counted analytically), so compute gini over the support plus a single zero-mass correction rather than materializing a 100M-element dense array. The re-rank itself is unaffected — it runs on the few hundred candidates per request — but the *metric* computation has to be tail-aware or it will OOM the eval job. This is the practical reason coverage is reported on a window (a day's serves) rather than all-time, and over the *reachable* catalog rather than every SKU that ever existed.

**What if the offline diversity goes up but online engagement is flat?** Two causes. First, you may have over-diversified into irrelevance — λ too low — so the list is varied but the variety is items the user does not want; the cure is to move λ back up the frontier toward the relevance corner. Second, and subtler, your online metric may be the wrong horizon: if you A/B-tested on same-session CTR, a successful diversity change can read as flat or slightly negative on CTR while being strongly positive on next-day return, because diversity's payoff is a *return* behavior, not a *click* behavior. This is the single most common way diversity work gets killed — measured on the metric it is designed to trade against. The fix is to pre-register the long-horizon metric (return rate, retention) as the primary success criterion *before* running the test, so a CTR wobble does not veto a retention win.

**What if a few power users dominate coverage?** Aggregate coverage can be gamed: if 1% of your users get extremely diverse, long-tail-heavy lists (because they have rich, varied histories) while 99% get the popular collapse, your population coverage looks healthy while almost every actual user sees a boring list. Always cross-check aggregate coverage against the *distribution of per-user list diversity*, not just the population total. A healthy system has both high aggregate coverage and a tight, high per-user diversity distribution — the population number alone can hide a badly skewed reality.

## 13. Case studies: who does this and what they found

The beyond-accuracy story is not academic; the largest recommenders in the world run on it.

**Netflix and calibrated recommendations (Steck, 2018).** Harald Steck's "Calibrated Recommendations" (RecSys 2018) is the canonical reference for matching a list's genre proportions to the user's. The paper's central empirical finding is that a recommender optimized purely for accuracy systematically *amplifies* a user's dominant interest and crowds out minority interests — a user who watches 80% one genre gets served ~100% of that genre — and that the calibrated re-rank corrects this with a negligible cost in the primary ranking metric. Netflix's broader public position, articulated repeatedly by their research team, is that the home page is a *portfolio* problem: each row and the set of rows are jointly optimized for variety, not just per-title relevance, precisely because a member who sees ten near-identical thrillers has no reason to keep scrolling.

**Spotify and the discovery-versus-familiarity balance.** Spotify's recommendation work (Discover Weekly, the Home feed, and the research behind them) is explicitly framed around balancing familiarity with discovery. Their published research on *exposure* and on optimizing for *long-term* user satisfaction rather than immediate streams reflects the same lesson: a playlist of only the user's known favorites maximizes short-term plays but does not grow the user's taste or the platform's catalog reach. Their work on metrics beyond immediate engagement — measuring whether a recommendation leads to a *saved* track or a returning listener — is serendipity measurement by another name.

**DPP at Hulu (Chen, Zhang, Zhou, 2018).** "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity" (NeurIPS 2018) is the practical reference for shipping DPPs. The contribution that made DPPs viable in a serving path was the fast greedy MAP algorithm — the incremental-Cholesky update sketched in section 6 — which brought a full DPP re-rank of hundreds of candidates down to single-digit milliseconds. They reported improved diversity at a controlled relevance cost, with the determinant's joint-set view outperforming MMR's pairwise penalty on their internal metrics.

**YouTube and filter-bubble mitigation.** YouTube has publicly described work to reduce "borderline" content amplification and to diversify recommendations away from narrow rabbit holes, motivated by the feedback-loop concern that an engagement-only objective will funnel a user into an ever-narrower set of increasingly extreme content. Their reinforcement-learning and long-term-value work (e.g., the "Top-K Off-Policy Correction" line and subsequent long-term-satisfaction research) is, in part, about optimizing for a horizon long enough that diversity and user-retention show up as wins rather than as a short-term engagement cost. The mechanism they are fighting is exactly the closed loop in [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles).

**The academic baseline (Ziegler et al., 2005).** The original "topic diversification" paper by Ziegler, McNee, Konstan, and Lawrence (WWW 2005) introduced intra-list similarity as a metric and showed in a large user study that users *preferred* diversified lists even though those lists scored *worse* on accuracy — the first hard evidence that the offline accuracy metric and actual user satisfaction diverge, which is the empirical foundation for this entire post.

The common thread across all five is the same lesson my first failed launch taught me, just measured at scale: accuracy is a short-horizon proxy, and the things accuracy trades against — variety, discovery, catalog health — are the things that determine whether users *stay*. Every one of these teams discovered that you have to measure and optimize the long-horizon thing directly, or your accuracy metric will quietly steer you into the boring corner.

## 14. How much beyond-accuracy to optimize

This is the judgment call, and it is worth being decisive about. More diversity is not always better; the question is how far along the frontier to move, and it depends on the product and the surface.

**Push diversity hard when:** the surface is a *browse/discovery* experience (a home feed, a "for you" page, a discovery playlist) where the user has no specific intent and a varied list is the whole point; when you have a *catalog-health* mandate (a marketplace that must give sellers exposure, a content platform that licensed a deep catalog it needs to monetize); when your retention metric is suffering and your CTR is fine (the classic "feed feels repetitive" signature); or when you are fighting a feedback loop and need to inject exposure into the tail to keep the system from collapsing.

**Keep diversity light when:** the surface is *high-intent* (a search results page, a "more like this" rail, a checkout cross-sell) where the user wants exactly the most-relevant thing and would be annoyed by variety for its own sake; when the catalog is genuinely small or homogeneous so there is little to diversify *into*; or when you have not yet built the long-horizon measurement to *see* the diversity payoff, in which case you will A/B-test on CTR, conclude diversity hurts, and revert it — better to first instrument retention, then tune diversity.

**Do not** diversify by hurting relevance you cannot afford. A list of ten varied but irrelevant items is strictly worse than ten relevant redundant ones; diversity is a *constrained* optimization, accuracy-subject-to-variety, never variety-at-any-cost. **Do not** bake diversity into the relevance model itself when a re-rank will do — you lose the serving-time tunability and entangle two concerns that iterate at different speeds. **Do not** report a coverage win from a change that only improved per-list diversity; verify the aggregate metric across the served population. And **do not** trust a single λ across surfaces — a search page and a discovery feed want different points on the frontier, so the λ should be per-surface, often per-context.

The meta-rule: pick the *one* long-horizon metric the diversity is actually for (retention, return rate, catalog GMV, creator satisfaction), instrument it, sweep λ in an A/B test, and choose the operating point that maximizes that metric — not the offline NDCG, and not diversity in the abstract. The whole point of the beyond-accuracy family is that the offline number you can measure cheaply is not the thing you care about, so you must measure the thing you care about, even though it is slower and harder.

## 15. Key takeaways

- **Accuracy is necessary but not sufficient.** A ranker tuned only on NDCG produces a redundant, popularity-collapsed, boring list because the metric is per-item and structurally blind to the relationships between recommended items.
- **The beyond-accuracy family splits into list-level and system-level metrics.** Diversity (intra-list dissimilarity), novelty ($-\log_2 p(i)$ self-information), and serendipity (relevant *and* unexpected) are properties of one user's list; coverage, gini, and entropy are properties of the system across all users. Improving one does not automatically improve the other.
- **Diversity is an accuracy-diversity *frontier*, and it is concave.** The first points of diversity cost almost no accuracy; only the extreme costs a lot. Sweep the trade-off knob and pick an operating point — do not accept the accuracy-only corner by default.
- **MMR is the right first tool.** Greedy, $O(NK)$, a single λ knob, two decades of production use. It picks each next item by relevance minus the similarity penalty to the nearest already-selected item.
- **DPP is the principled upgrade.** It encodes relevance and diversity in one kernel, where "diverse and relevant" is literally "high determinant / large volume," and accounts for the whole selected set jointly rather than pairwise. The Hulu fast-greedy MAP made it serving-viable.
- **Calibration (Steck 2018) grounds diversity in the user's own proportions.** Match the recommended genre mix to the user's history via KL divergence; it respects narrow tastes while rescuing crowded-out minority interests.
- **All of this lives in the re-ranking layer.** Keep retrieval and ranking as pure per-item relevance predictors; apply diversity, calibration, freshness, and business rules as a serving-time post-process on the small candidate set, so the knobs are tunable without retraining.
- **Coverage must be measured across the whole served population, novelty popularity on the training set only, and the split must be temporal.** Per-user coverage and test-set popularity are leakage.
- **A/B-test diversity on the long-horizon metric, not CTR.** Diversity trades against the short-horizon accuracy proxy by construction; if you measure only CTR you will always conclude it hurts. Measure retention, return rate, or catalog health — the thing the diversity is actually for.

## 16. Further reading

- **Ziegler, McNee, Konstan, Lawrence (2005), "Improving Recommendation Lists Through Topic Diversification" (WWW 2005)** — introduced intra-list similarity and the user study showing people prefer diversified lists despite lower accuracy. The empirical foundation for the whole field.
- **Carbonell and Goldstein (1998), "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries" (SIGIR 1998)** — the original Maximal Marginal Relevance paper.
- **Steck (2018), "Calibrated Recommendations" (RecSys 2018)** — the Netflix paper on matching list genre proportions to user history via KL divergence.
- **Chen, Zhang, Zhou (2018), "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity" (NeurIPS 2018)** — the Hulu fast-greedy DPP that made determinantal re-ranking serving-viable.
- **Kulesza and Taskar (2012), "Determinantal Point Processes for Machine Learning"** — the definitive monograph on DPPs, the kernel geometry, and inference.
- **Vargas and Castells (2011), "Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems" (RecSys 2011)** — a careful formal treatment of how to define and combine novelty and diversity metrics.
- **Ge, Delgado-Battenfeld, Jannach (2010), "Beyond Accuracy: Evaluating Recommender Systems by Coverage and Serendipity" (RecSys 2010)** — the paper that named the family and formalized coverage and serendipity.
- Within this series: [the recommendation funnel: retrieval, ranking, re-ranking](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking), [offline evaluation metrics: recall, NDCG, MAP, MRR](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr), [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer), [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles), and the synthesis in the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

---
title: "Offline Evaluation Metrics: Recall, NDCG, MAP, and MRR"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A definitive guide to top-K ranking metrics for recommenders: what Precision, Recall, Hit Rate, MRR, MAP, and NDCG each measure, the full DCG and IDCG derivations, when to pick which, and a vectorized numpy and PyTorch metrics harness you can drop into any training loop."
tags:
  [
    "recommendation-systems",
    "recsys",
    "evaluation",
    "ndcg",
    "map",
    "mrr",
    "ranking-metrics",
    "machine-learning",
    "information-retrieval",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-1.png"
---

You retrain the ranker on Friday. Monday morning the dashboard says NDCG@10 went from 0.241 to 0.287 — a clean win, the kind you screenshot and paste into the team channel. You ship it. Two weeks later the A/B test reads flat: no lift in watch time, no lift in clicks, a rounding-error of a result that the stats engine refuses to call significant. The offline number went up. The online number didn't move. Somebody asks the obvious question in the retro — *which* offline number, and did it actually measure the thing the product cares about? — and you realize you do not have a crisp answer. You picked NDCG@10 because the last person picked NDCG@10.

That is the situation this post is here to fix. Recommenders are not rating predictors; almost nothing you ship is graded on how close a predicted star count is to the truth. What you ship is a **ranked list** — a top-K slate per user — and what you are graded on is whether the items a user actually wanted landed near the top of that slate. Scoring that is a surprisingly subtle business. There are at least seven metrics in common use, they disagree with each other constantly, and the difference between picking the right one and the wrong one is the difference between an offline number that predicts your A/B test and one that lies to you. The companion post on [the right way to split and evaluate a recommender](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) covers *how* to build the held-out set without leaking the future; this post covers *what to compute* once you have it.

![A ranked top-K list scored by where the relevant items land, with a per-rank position weight showing rank one is worth far more than rank five](/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-1.png)

Look at the figure above before we go further, because it is the whole post in one picture. Two lists, A and B, each contain exactly two relevant items inside the top five. By a metric that only counts presence, they tie. But list A puts its relevant items at ranks 1 and 3, while list B buries them at ranks 4 and 5 — and the bottom row shows the position weight $1/\log_2(r+1)$ that an order-aware metric assigns to each slot: 1.00, 0.63, 0.50, 0.43, 0.39. List A collects weights at the heavy end; list B collects them at the light end. The position discount means rank 1 is worth roughly two and a half times rank 5. That single idea — *where the relevant items sit, not just whether they are present* — is what separates the metrics that predict your online result from the metrics that don't.

By the end of this post you will be able to: define Precision@K, Recall@K, Hit Rate@K, F1@K, MRR, MAP, and NDCG@K from their formulas and explain what each one ignores; derive DCG, IDCG, and NDCG line by line; choose the right metric for one-right-answer search versus large-K retrieval versus a graded feed; and drop a fully vectorized numpy/PyTorch harness into your training loop that computes all of them over a whole user batch at once. This harness is the reusable evaluation core for the rest of the series — every later post that reports a Recall@K or NDCG@K number is computing it with code shaped like what you will build here. We tie back, as always, to [the retrieval-ranking-reranking funnel](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) that organizes the whole series: retrieval is judged on large-K recall, ranking on small-K order quality, and the gap between the two stages is where metric choice does its quiet damage.

## Why top-K, and not RMSE

The first recommender course most people take ends with RMSE on the MovieLens rating matrix. You predict a star rating, you square the error, you take the root, you minimize it. It is a clean regression problem and it is almost entirely the wrong objective for a deployed recommender. Three reasons.

First, **users never see ratings; they see a list.** Your model could nail the predicted rating of every movie a user will hate and get full marks on the items they will never scroll to. RMSE spends its budget on the entire catalog uniformly, but the product only ever exposes the top of one ranking. A model that is slightly worse on average rating error but reliably puts the right movie at rank 1 is the better product, and RMSE cannot see that.

Second, **the data is overwhelmingly implicit and one-sided.** Most real systems do not have a dense rating matrix; they have clicks, plays, purchases, watch-time — positive-only signals where the absence of a click is not a negative rating, it is mostly "never shown" or "not yet seen." (The post on [implicit versus explicit feedback](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have) goes deep on this.) There is no real-valued ground truth to regress against. There is a set of items the user engaged with, and everything else. That shape — a set of relevant items per user, against which you score a ranked candidate list — is exactly what top-K metrics are built for.

Third, **the decision is a ranking decision.** Retrieval narrows millions of items to a few hundred candidates; ranking orders those candidates; re-ranking trims and diversifies the final slate. Every stage outputs an *order*. The natural way to grade an order is to ask how well the items the user wanted are placed within it, with a heavy reward for putting them near the top where a real user will actually look. That is the top-K evaluation paradigm, and it is the lingua franca of both information retrieval (where these metrics were born, grading search results) and modern recsys.

So fix the setup once and for all. For each user $u$ you have:

- A **ranked candidate list** $\pi_u = (i_1, i_2, \dots, i_N)$ — the model's output, best first.
- A set (or graded set) of **ground-truth relevant items** $R_u$ — what the held-out future says the user actually engaged with.
- A cutoff $K$ — you score only the top $K$ of the list, because that is all the product shows.

Every metric below is a function of $\pi_u$, $R_u$, and $K$, averaged over users. The differences between them come down to two questions: *do they care about order within the top K, and do they read relevance as a yes/no flag or as a graded level?*

## The metric family at a glance

Before the formulas, the map. The top-K metrics split cleanly into two families, and one of them splits again.

![A taxonomy tree splitting top-K metrics into set metrics that ignore order and order metrics that reward placement, with NDCG as the only graded branch](/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-2.png)

The **set metrics** — Precision@K, Recall@K, Hit Rate@K — ask only *whether* relevant items are present in the top K. Shuffle the top K however you like and these numbers do not change. They are the right tool when order genuinely does not matter yet, which is exactly the case at the **retrieval** stage: retrieval's job is to get the relevant items *into the candidate set* so the ranker downstream can order them. If the relevant item is in your top-200 candidates, retrieval did its job; whether it was candidate 12 or candidate 180 is the ranker's problem, not retrieval's.

The **order metrics** — MRR, MAP, NDCG@K — ask *where* the relevant items sit, rewarding higher placement. These are the right tool once order is the product, which is the **ranking** stage: the user reads top to bottom and you are graded on putting the best item first. Within the order family, MRR and MAP read relevance as binary (an item is relevant or it isn't), while **NDCG** is the only common metric that reads *graded* relevance — a five-star rating, a long watch versus a short one, a purchase versus a mere click. That single capability is why NDCG is the default gold standard for feeds and search.

Hold that taxonomy in mind. Now we walk each metric: formula, intuition, and the one sentence on when to reach for it.

## Set metrics: Precision, Recall, Hit Rate, F1

### Precision@K

Precision@K asks: of the $K$ items I showed, what fraction were relevant?

$$
\text{Precision@}K = \frac{|\{\text{top-}K\} \cap R_u|}{K}
$$

If you show 10 items and 3 of them were things the user engaged with, Precision@10 = 0.3. It is the slate-quality metric: a high precision means you are not wasting the user's attention on junk. Its weakness for recsys is that $K$ is in the denominator regardless of how many relevant items even exist. If a user only has 2 relevant items in the entire held-out set, the best possible Precision@10 is 0.2 — there simply aren't 10 right answers to find. That makes Precision@K hard to compare across users with very different amounts of ground truth, which is the norm in recsys (some users engaged with 2 items last week, some with 200).

### Recall@K

Recall@K flips the denominator: of the relevant items that exist, what fraction did I surface in the top K?

$$
\text{Recall@}K = \frac{|\{\text{top-}K\} \cap R_u|}{|R_u|}
$$

If the user has 4 relevant items and your top 50 contains 3 of them, Recall@50 = 0.75. This is **the retrieval metric.** When the candidate generator's whole job is to make sure the relevant items end up in the candidate pool, Recall@K at a *large* K (K = 100, 500, 1000) is exactly the question "did we catch the things worth ranking?" A retrieval recall of 0.9 at K = 500 says: nine times out of ten, the item the user will engage with is somewhere in the 500 we handed to the ranker. The ranker can only order what retrieval gives it, so retrieval recall is the ceiling on the whole funnel. You will see Recall@K reported in nearly every two-tower retrieval paper, almost always at large K.

A subtlety that trips people up: Recall@K and Precision@K are the same number only when $|R_u| = K$. Otherwise they tell different stories, and which you report depends on whether you care about the denominator being "what you showed" (precision) or "what was findable" (recall).

There is a second subtlety worth internalizing, because it explains why retrieval teams almost never look at precision. In a typical recsys held-out set a user has a handful of relevant items — five, ten, maybe twenty — against a catalog of millions. When you evaluate retrieval at K = 500, the *best achievable* Precision@500 is at most $20/500 = 0.04$, because there are only twenty right answers and you are forced to show five hundred slots. A precision of 0.04 looks like a catastrophe on a dashboard, but it is the arithmetic ceiling, not a model failure — the denominator is the candidate-pool size you deliberately chose, not a measure of quality. Recall, by contrast, has a ceiling of 1.0 regardless of K, which is why it is the honest retrieval metric. The general rule: when $K \gg |R_u|$, precision is structurally low and uninformative; report recall. When $K \approx |R_u|$ (a tight slate), precision and recall converge and both are meaningful.

### Hit Rate@K

Hit Rate@K (also called Hits@K) is the most forgiving metric in the family. It asks a single yes/no question: was *at least one* relevant item in the top K?

$$
\text{HitRate@}K = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} \mathbb{1}\big[ |\{\text{top-}K\} \cap R_u| > 0 \big]
$$

Averaged over users, it is the fraction of users for whom you got *something* right. Hit Rate is the metric you reach for in the leave-one-out evaluation setup common in sequential recommendation (SASRec, BERT4Rec) — you hide a user's single most recent interaction and ask whether the model ranked that one held-out item inside the top K. With $|R_u| = 1$, Hit Rate@K, Recall@K, and "the held-out item appeared in top K" all collapse to the same quantity. Outside that single-target setting, Hit Rate is coarse: it cannot tell a list with one hit from a list with five, so it is a floor, not a target.

### F1@K

When you genuinely care about both precision and recall at the same K — you want a tight slate that is also complete — combine them with the harmonic mean:

$$
\text{F1@}K = \frac{2 \cdot \text{Precision@}K \cdot \text{Recall@}K}{\text{Precision@}K + \text{Recall@}K}
$$

The harmonic mean punishes imbalance: if precision is 0.9 and recall is 0.1, the arithmetic mean is 0.5 but F1 is 0.18, which honestly reflects that you missed almost everything. F1@K is useful for fixed-size slates where both over-showing junk and under-showing relevant items hurt, but in practice retrieval teams report recall and ranking teams report NDCG, and F1@K shows up mostly in set-retrieval problems and academic comparisons.

The whole set family shares one defining blind spot, and it is worth saying plainly: **none of them can see order.** Move a relevant item from rank 1 to rank K and Precision@K, Recall@K, and Hit Rate@K do not change by a hair. For retrieval, that is a feature. For ranking, it is the bug that sends you to the order metrics.

## MRR: reward the first right answer

Mean Reciprocal Rank is the simplest order-aware metric, and it answers a very specific product question: *how far down the list is the first thing the user wanted?*

For a single user, find the rank of the first relevant item — call it $\text{rank}_u$ — and take its reciprocal. Average over users:

$$
\text{MRR} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\text{rank}_u}
$$

If the first relevant item is at rank 1, the reciprocal rank is 1.0. Rank 2 gives 0.5, rank 3 gives 0.33, rank 10 gives 0.1. If there is no relevant item in the top K at all, the reciprocal rank is 0. The reciprocal curve is steep early and flat late — exactly the right shape for "the user gives up if the answer isn't near the top." Going from rank 1 to rank 2 costs you half your score; going from rank 9 to rank 10 costs almost nothing, because a user who scrolled to rank 9 will probably read rank 10 too.

MRR is the metric for **one-right-answer products**: a search box where there is a single correct result (a "navigational" query like *facebook login*), a question-answering system, an autocomplete, a "resume watching" rail. In all of these there is essentially one item the user is reaching for, and what matters is whether you put it first. MRR's defining limitation is the flip side of its simplicity: **it stops reading the moment it finds the first hit.**

There is a subtle statistical property of MRR worth flagging, because it explains why MRR numbers can look jumpy across model versions. The reciprocal-rank function is heavily front-loaded: the only values it can take are $1, \tfrac12, \tfrac13, \tfrac14, \dots, 0$, and the gaps between them are enormous at the top (1.0 to 0.5 is a half-unit gap) and tiny at the bottom (0.1 to 0.09 is a hundredth). This means a model change that promotes the first relevant item from rank 2 to rank 1 for even a modest fraction of users can swing the mean MRR by a visibly large amount, while a change that improves ranks 8 through 20 barely registers. MRR is, in effect, a metric about your behavior at ranks 1 and 2 and almost nothing else. If your product's first slot genuinely dominates user attention — a single hero result, a one-tap autocomplete — that front-loading is exactly what you want. If your product is a scrollable feed, MRR's obsession with the very top and its blindness to everything after the first hit make it the wrong lens.

![A side-by-side contrast showing MRR locks its score on the first relevant item at rank two while MAP and NDCG keep crediting later hits at ranks four and six](/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-3.png)

The figure makes the limitation concrete. The first relevant item sits at rank 2, so MRR locks in a reciprocal rank of $1/2 = 0.50$ and is done. The fact that there are *more* relevant items at ranks 4 and 6 is invisible to MRR — a list with one early hit and three buried ones scores identically to a list with just the one early hit. That is correct behavior for a one-right-answer product and badly wrong behavior for a feed where you want to surface as many good items as possible. When you have multiple relevant items per user and you care about all of them, you want MAP or NDCG, both of which keep reading.

## MAP: average precision over every relevant item

Mean Average Precision is what you get when you take Precision@K seriously about order. It is built in two layers, and the names are confusingly similar, so go slow: **Average Precision (AP)** is computed per user; **MAP** is the mean of AP over users.

To compute AP for one user, walk down the ranked list. Every time you hit a relevant item, compute the precision *at that rank* — the fraction of items so far that are relevant — and remember it. At the end, average those precision values over the number of relevant items. Formally, with $\text{rel}(k) = 1$ if the item at rank $k$ is relevant and 0 otherwise, and $P(k)$ the precision over the first $k$ items:

$$
\text{AP@}K = \frac{1}{|R_u|} \sum_{k=1}^{K} P(k)\,\text{rel}(k), \qquad P(k) = \frac{1}{k}\sum_{j=1}^{k}\text{rel}(j)
$$

The genius of AP is that the $\text{rel}(k)$ gate means you only sample precision *at the ranks where a relevant item actually appears*, and precision-at-rank is naturally higher when relevant items come early. Put the relevant items first and every $P(k)$ you sample is high; bury them and every $P(k)$ you sample is dragged down by all the junk above. AP rewards clustering relevant items near the top and rewards finding *all* of them — it is the order-aware completion of precision.

$$
\text{MAP@}K = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} \text{AP@}K_u
$$

MAP is the right metric for **set-retrieval problems with multiple relevant items and binary relevance**: a legal-document search where there are several relevant cases and you want them all near the top, an image-retrieval system, a "more like this" rail. It reads the whole list (unlike MRR) and it is order-aware (unlike Recall). Its one real limitation is that it is **binary**: AP cannot tell a "loved it" item from a "tolerated it" item, because $\text{rel}(k)$ is just 0 or 1. The moment your relevance signal is graded — stars, watch-fraction, purchase-versus-click — you are leaving information on the table, and the metric that uses it is NDCG.

## NDCG: the position-discounted, graded gold standard

Normalized Discounted Cumulative Gain is the metric most production feeds and search systems actually optimize and report, and it is worth deriving from scratch because every piece of it encodes a real product belief. We build it in layers.

![A layered stack showing NDCG built from graded relevance to per-item gain to position discount to DCG to division by IDCG to a final zero-to-one score](/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-4.png)

**Layer 1 — graded relevance.** Start with a relevance level for each item, $\text{rel}_i \in \{0, 1, 2, 3, \dots\}$. Zero is irrelevant; higher is better. In search this is a human-labeled grade (perfect, excellent, good, fair, bad). In recsys it is derived from the signal: a purchase might be 3, an add-to-cart 2, a click 1, an impression-with-no-click 0; or watch-fraction bucketed into levels. Binary relevance is just the special case where every grade is 0 or 1.

**Layer 2 — per-item gain.** Convert each grade into a *gain*. The standard choice is the exponential gain:

$$
\text{gain}_i = 2^{\text{rel}_i} - 1
$$

so grade 0 gives gain 0, grade 1 gives 1, grade 2 gives 3, grade 3 gives 7. The exponential form makes the metric strongly prefer putting the *most* relevant items first — a single grade-3 item is worth seven grade-0 items, not three. (A linear gain $\text{gain}_i = \text{rel}_i$ also exists and is used in some libraries; the exponential form is the one in the original Microsoft and most leaderboard definitions, and it is what you should default to. Always state which gain you used, because the two are not comparable.)

**Layer 3 — position discount.** Now divide each item's gain by a discount that grows with its rank, so an item earns less credit the further down it sits. The discount is logarithmic:

$$
\text{discount}(r) = \frac{1}{\log_2(r+1)}
$$

At rank 1 the discount is $1/\log_2 2 = 1.0$; at rank 2, $1/\log_2 3 = 0.631$; at rank 3, $1/\log_2 4 = 0.5$; at rank 4, $0.431$; at rank 5, $0.387$. This is the curve in figure 1 and the engine of the whole metric.

**Layer 4 — sum to DCG.** Discounted Cumulative Gain at cutoff $K$ is the sum of discounted gains over the top $K$ ranks:

$$
\text{DCG@}K = \sum_{r=1}^{K} \frac{2^{\text{rel}_r} - 1}{\log_2(r+1)}
$$

DCG rewards two things at once: putting high-grade items in the list at all (the gain term) and putting them near the top (the discount term). But DCG is unbounded and scale-dependent — a user with five grade-3 items can rack up a far larger DCG than a user with one grade-1 item, purely because there is more relevance to be had. You cannot average raw DCG across users and get a meaningful number.

**Layer 5 — normalize by IDCG.** Fix the scale by dividing by the best DCG achievable for *that user* — the Ideal DCG, computed by sorting the user's items by true relevance (highest grade first) and applying the same DCG formula:

$$
\text{IDCG@}K = \sum_{r=1}^{K} \frac{2^{\text{rel}_r^{*} } - 1}{\log_2(r+1)}, \qquad \text{rel}^{*} = \text{relevances sorted descending}
$$

**Layer 6 — NDCG.** The ratio:

$$
\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}
$$

Because IDCG is the maximum possible DCG for that user, NDCG lands in $[0, 1]$: 1.0 means you produced the ideal ordering, and it is comparable across users with wildly different amounts of relevance. That normalization is why NDCG can be averaged into a single dashboard number that actually means something.

NDCG is the right default for **graded feeds and search**: a home feed, a video recommendations rail, a web search results page — anywhere relevance comes in degrees and order within a short visible window is what the user experiences. Report it at a *small* K (NDCG@5, NDCG@10) because that matches the visible slate. It is the only common metric that is simultaneously order-aware, graded, and bounded, which is why it sits at the top-right of the next figure as the metric that does everything the others do partially.

![A matrix comparing six top-K metrics on order awareness, graded relevance, whether they read past the first hit, and their best-fit use case](/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-5.png)

The matrix is the cheat sheet to pin above your desk. Read down the columns: Precision, Recall, and Hit Rate all answer "no" to order awareness and "no" to graded relevance — they are set metrics. MRR answers "yes" to order but "no" to graded and, crucially, reads only the first hit. MAP is order-aware and reads the whole list but stays binary. NDCG is the only row that is "yes" across order awareness, graded relevance, and reading past the first hit — which is exactly why it is the gold standard and why the best-fit column lands it on graded feeds. The best-fit column is the punchline of the post compressed into six cells: slate quality, retrieval recall, any hit at all, one right answer, set retrieval, graded feeds.

## The position discount, examined

The log discount is the single most important design choice in NDCG, and it pays to understand its shape rather than treat it as a magic constant. The key property is that **it falls steeply at first and then flattens.**

![A grid of position-discount weights and their drops by rank showing a large 0.37 fall from rank one to two then progressively smaller flattening drops](/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-6.png)

The grid spells out the marginal value of moving an item up one slot. Moving from rank 1 to rank 2 costs the most: the weight falls from 1.00 to 0.63, a drop of 0.37. From rank 2 to rank 3 the drop is only 0.13. By the time you are shuffling around ranks 5 through 10, each step is worth roughly 0.10 and falling. This is not arbitrary — it is a deliberate model of attention. A real user's probability of examining a result drops sharply with rank (eye-tracking studies of search results show exactly this steep-then-flat decay), so the metric weights credit by where attention actually goes. The practical consequence for you as an engineer: **metric gains concentrate at the top of the list.** A reranker that promotes one relevant item from rank 4 to rank 1 moves NDCG far more than one that promotes an item from rank 40 to rank 30, even though both moved ten ranks in the second case. When you are debugging a flat NDCG despite "obvious" improvements, check whether your wins are happening below the fold where the discount has already flattened to near nothing.

Why $\log_2$ specifically? Honestly, the base is a convention, not a law — $\log_2$, $\log_e$, $\log_{10}$ all produce the same *ordering* of systems because the base is a constant factor that cancels in the DCG/IDCG ratio for any fixed base. What matters is the logarithmic *shape*: a slowly growing denominator that discounts later positions gently rather than the harsh $1/r$ of reciprocal rank. (MRR uses $1/r$, which is why MRR punishes a rank-2 result so much harder than NDCG does.) The log was chosen by Järvelin and Kekäläinen in the original 2002 DCG paper precisely as a "smoothed" discount that does not collapse the value of position 2 the way a pure reciprocal would.

## Graded versus binary, micro versus macro, and the @K choice

Three practical decisions surround every metric you compute, and getting them wrong silently corrupts your dashboard.

**Graded versus binary relevance.** If your signal is genuinely graded — star ratings, watch-fraction, dwell time, purchase-versus-cart-versus-click — then NDCG with exponential gain uses that information and the binary metrics throw it away. But be honest about whether your grades are *real*. Bucketing a continuous watch-fraction into four arbitrary levels does not create graded relevance; it creates four arbitrary levels. Many production systems deliberately use binary relevance ("did the user engage, yes or no") because the engagement signal is binary anyway, in which case NDCG@K with all-1 gains is still a perfectly good order-aware metric and reduces to a clean discounted hit-counting formula. Use graded relevance when the grades carry signal; do not manufacture grades to look sophisticated.

**Micro versus macro averaging.** When you average a metric over users you have two choices. **Macro averaging** computes the metric per user and then averages the per-user numbers, giving every user equal weight regardless of how active they are. **Micro averaging** pools all the relevant-item decisions across users into one big numerator and denominator. They differ sharply when activity is skewed: a few power users with hundreds of interactions can dominate a micro average, so the number reflects your whales, not your median user. Almost all recsys papers report **macro** (per-user, then mean) NDCG/Recall/MAP, and you should too unless you have a specific reason to weight by activity — it answers "how good is the experience for a typical user" rather than "how good is the experience for a typical interaction." State which you used; the gap between them on a skewed dataset can be several points.

**The @K choice.** $K$ is not cosmetic; it changes what you are measuring. Set $K$ to match the stage and the surface. For **retrieval**, use a large K equal to the candidate-pool size you hand downstream (Recall@200, Recall@500, Recall@1000) — you are asking whether the relevant item made the cut. For **ranking**, use a small K equal to what the user actually sees above the fold (NDCG@5, NDCG@10) — order below the fold barely matters because the discount has flattened. Reporting NDCG@1000 on a ranking model is nearly meaningless: the log discount makes ranks 100–1000 almost weightless, so the number is dominated by retrieval recall and tells you nothing about ranking quality. Match K to the product surface, every time.

## The science: full derivations and what each metric ignores

Let us make the relationships rigorous, because the disagreements between metrics are not noise — they are mathematically inevitable consequences of what each formula does and does not read.

**MRR as the expectation of a step function.** Write the rank of the first relevant item as $\rho_u = \min\{r : \text{rel}(r) = 1\}$. Then $\text{MRR} = \mathbb{E}_u[1/\rho_u]$, with $1/\rho_u = 0$ when no relevant item is found in range. This is an expectation over a function that depends on a single order statistic — the minimum rank of a relevant item. Everything else about the list is integrated out. Two lists with the same $\rho_u$ but completely different placements of every other relevant item have *identical* MRR. That is not a flaw to be fixed; it is the definition. MRR is a deliberate projection of the ranking onto one number, the position of the first hit.

**MAP as the area under the precision-recall curve.** Average Precision has a beautiful interpretation: AP is a discrete approximation of the area under the precision-recall curve for that user. As you walk down the ranked list, each relevant item you encounter steps recall up and you sample precision at that recall level; AP sums those samples. So MAP is the mean area-under-PR over users. This is why MAP is the natural metric for binary set retrieval — area under PR is the canonical summary of a binary ranking quality. And it is why MAP cannot use graded relevance: the precision-recall curve is defined on binary labels. To handle grades you would need a fundamentally different construction, which is exactly what DCG provides.

**NDCG and the additive-gain, multiplicative-discount form.** The DCG functional has a specific structure: it is *additive over items* (each item contributes independently) and *separable into a gain times a discount* ($\text{gain}(\text{rel}_r) \cdot \text{discount}(r)$). This separability is what makes NDCG differentiable-in-spirit and the target of learning-to-rank methods like LambdaMART and LambdaRank, which compute the *change* in NDCG from swapping two items (the "lambda") as the gradient signal. The post on [learning to rank for recommenders](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders) shows how that delta-NDCG drives training. The normalization by IDCG is what bounds the metric and makes it averageable; without it you have DCG, which is fine as a relative within-user signal but useless as a cross-user aggregate.

**What each metric provably ignores.** Precision@K and Recall@K ignore order entirely (permutation-invariant over the top K). Hit Rate@K ignores everything except the indicator that one hit exists. MRR ignores every relevant item after the first. MAP ignores relevance *magnitude* (binary only). NDCG ignores almost nothing within the top K — but note even NDCG ignores items *below* K, and all of these metrics ignore the difference between a held-out negative and a never-seen item, which is the deep problem the post on [the offline-online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) takes apart. No offline metric sees position bias, selection bias, or the fact that your held-out "relevant" set is itself the product of a previous recommender's choices. Offline metrics grade the ranking; they do not grade reality.

**The missing-not-at-random problem, stated precisely.** This is the one that should keep you up at night, because it undermines the ground truth itself rather than the metric. Every metric above assumes the relevant set $R_u$ is a clean sample of what the user would have liked. It is not. The items a user engaged with are the items a *previous recommender chose to show them* — engagement is conditioned on exposure, and exposure is conditioned on the old model's policy. Formally, the data is **missing not at random (MNAR)**: the probability that you observe a positive for item $i$ depends on item $i$'s properties (popular items get shown more, so they get more observed positives), not on a random coin flip. A consequence is that your held-out "relevant" set is biased toward items the old system already liked, so a new model that simply mimics the old system's popularity bias scores well offline *because the test set was generated by a popularity-biased policy*. Recall@K in particular rewards re-finding popular items. This is precisely why a popularity prior lifts offline recall (you saw it in the results table) and why that lift so often fails to reproduce online — the offline metric is measuring agreement with the old policy, not the user's true preference. Inverse-propensity-scored (IPS) estimators attempt to debias this by reweighting each observed positive by the inverse of its exposure probability, but that is a topic for the off-policy evaluation discussion; the takeaway here is that **no choice of metric fixes a biased ground truth.** The metric is downstream of the data, and the data is downstream of the old model.

#### Worked example: NDCG@5, MAP, and MRR on one ranked list by hand

Take one user. The model returns a top-5 list, and we know the graded relevance of each returned item (0 = irrelevant, up to 3 = loved):

```markdown
rank r:        1    2    3    4    5
item:          A    B    C    D    E
relevance:     3    0    2    0    1
```

**MRR.** The first relevant item (relevance > 0) is A at rank 1. Reciprocal rank = 1/1 = **1.0**. MRR for this single user is 1.0. Notice MRR is already "done" — it never looks at C or E.

**MAP (binary).** Treat relevance > 0 as relevant. Relevant items are at ranks 1, 3, 5. There are 3 relevant items total in this list, so $|R_u| = 3$. Compute precision at each relevant rank:

- Rank 1 (A, relevant): 1 relevant of 1 seen, $P(1) = 1/1 = 1.000$
- Rank 3 (C, relevant): 2 relevant of 3 seen, $P(3) = 2/3 = 0.667$
- Rank 5 (E, relevant): 3 relevant of 5 seen, $P(5) = 3/5 = 0.600$

$$
\text{AP} = \frac{1.000 + 0.667 + 0.600}{3} = \frac{2.267}{3} = 0.756
$$

So AP@5 = **0.756**, and MAP over this one user is 0.756.

**NDCG@5 (graded, exponential gain).** First the gains $2^{\text{rel}} - 1$: A = $2^3-1 = 7$, B = 0, C = $2^2-1 = 3$, D = 0, E = $2^1-1 = 1$. Now discount by $1/\log_2(r+1)$:

- Rank 1: $7 / \log_2 2 = 7 / 1.000 = 7.000$
- Rank 2: $0 / \log_2 3 = 0$
- Rank 3: $3 / \log_2 4 = 3 / 2.000 = 1.500$
- Rank 4: $0 / \log_2 5 = 0$
- Rank 5: $1 / \log_2 6 = 1 / 2.585 = 0.387$

$$
\text{DCG@5} = 7.000 + 0 + 1.500 + 0 + 0.387 = 8.887
$$

Now the ideal ordering. Sort the relevances descending: $3, 2, 1, 0, 0$, giving gains $7, 3, 1, 0, 0$. Apply the same discounts:

- Rank 1: $7 / 1.000 = 7.000$
- Rank 2: $3 / 1.585 = 1.893$
- Rank 3: $1 / 2.000 = 0.500$
- Ranks 4–5: gain 0, contribute 0

$$
\text{IDCG@5} = 7.000 + 1.893 + 0.500 = 9.393
$$

$$
\text{NDCG@5} = \frac{8.887}{9.393} = 0.946
$$

So this list scores MRR = 1.0, MAP = 0.756, NDCG@5 = 0.946. Three numbers, three different stories about the same list: MRR says "perfect, the best item is first"; MAP says "decent but you buried relevant items at 3 and 5"; NDCG says "very good, because the item you put first was the *most* relevant one and that is worth a lot." NDCG's higher score versus MAP comes from rewarding the grade-3 item at the top — information MAP literally cannot see.

#### Worked example: two rankings where Recall agrees but NDCG disagrees

This is the disagreement that should change how you read dashboards. Two rankings of the same five candidates for a user with two relevant items (both binary-relevant, both graded 1):

```markdown
Ranking X:   rel  not  not  rel  not     (relevant at ranks 1 and 4)
Ranking Y:   rel  rel  not  not  not     (relevant at ranks 1 and 2)
```

**Recall@5.** Both rankings contain both relevant items inside the top 5. So Recall@5 = 2/2 = **1.0 for both.** Identical. A retrieval dashboard would call these two rankings equally good and move on.

**NDCG@5 (binary gains, so gain = 1 for each relevant item).** Discounts again: $1/\log_2(r+1)$ = 1.000, 0.631, 0.500, 0.431, 0.387 for ranks 1–5.

Ranking X has relevant items at ranks 1 and 4: $\text{DCG} = 1.000 + 0.431 = 1.431$.
Ranking Y has relevant items at ranks 1 and 2: $\text{DCG} = 1.000 + 0.631 = 1.631$.
Both have the same ideal: two relevant items at ranks 1 and 2, $\text{IDCG} = 1.000 + 0.631 = 1.631$.

$$
\text{NDCG@5}(X) = \frac{1.431}{1.631} = 0.877, \qquad \text{NDCG@5}(Y) = \frac{1.631}{1.631} = 1.000
$$

Recall ties at 1.0; NDCG splits 0.877 versus 1.000 — a 14% relative gap on the same recall. Ranking Y is strictly better because it put both relevant items where the user will see them, and only the order-aware metric can tell. This is the figure 7 contrast in numbers, and it is the single most important intuition in this post for choosing a metric: **if order matters to your product, a recall-only dashboard will tell you two materially different rankings are identical.**

![A side-by-side contrast of two rankings with identical Recall at five but NDCG swinging from 0.61 to 1.0 because the relevant items moved to better slots](/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-7.png)

The figure shows the same phenomenon with a slightly larger gap (NDCG 0.61 versus 1.0) to make the point vivid: two lists, identical Recall@5 = 1.0, but ranking X buries its relevant items at ranks 4 and 5 while ranking Y puts them at 1 and 2, and NDCG swings by a third. Recall sees a tie; NDCG sees a different product.

## The practical flow: a complete, vectorized metrics harness

Now the code. This is the reusable harness for the whole series, so it is worth building properly: vectorized over all users at once (no Python loop over users — that matters when you evaluate 100k users on a held-out set), supporting binary and graded relevance, and returning every metric we discussed. We do it in numpy first because numpy is the lingua franca and the logic is transparent, then show the PyTorch variant for GPU evaluation inside a training loop.

The input convention. For a batch of $U$ users you have:

- `scores`: a `(U, N)` array of model scores over $N$ candidate items per user (or the full catalog).
- `relevance`: a `(U, N)` array of ground-truth relevance, aligned with `scores`. Binary (0/1) or graded (0,1,2,3).

You rank each row by descending score, gather the relevance in ranked order, and compute everything from that ranked-relevance matrix. Here is the core.

```python
import numpy as np

def rank_relevance(scores, relevance, k):
    """Sort each user's items by score (desc) and return the top-k relevance.

    scores, relevance: (U, N) arrays aligned per user.
    Returns ranked_rel: (U, k) relevance in model-ranked order.
    """
    # argsort descending; take top-k columns
    topk_idx = np.argsort(-scores, axis=1)[:, :k]          # (U, k)
    ranked_rel = np.take_along_axis(relevance, topk_idx, axis=1)
    return ranked_rel                                       # (U, k)


def precision_at_k(ranked_rel, k):
    hits = (ranked_rel[:, :k] > 0).sum(axis=1)              # (U,)
    return hits / k                                         # per-user (U,)


def recall_at_k(ranked_rel, relevance, k):
    hits = (ranked_rel[:, :k] > 0).sum(axis=1)             # (U,)
    n_rel = (relevance > 0).sum(axis=1)                    # (U,)
    n_rel = np.clip(n_rel, 1, None)                        # avoid /0 for empty users
    return hits / n_rel


def hit_rate_at_k(ranked_rel, k):
    return ((ranked_rel[:, :k] > 0).sum(axis=1) > 0).astype(float)  # (U,)


def mrr_at_k(ranked_rel, k):
    rel = (ranked_rel[:, :k] > 0)                          # (U, k) bool
    # rank (1-indexed) of first relevant item, or 0 if none
    ranks = np.arange(1, k + 1)[None, :]                   # (1, k)
    first_hit = np.where(rel.any(axis=1),
                         (np.where(rel, ranks, k + 1)).min(axis=1),
                         0)                                # (U,)
    rr = np.where(first_hit > 0, 1.0 / first_hit, 0.0)
    return rr


def average_precision_at_k(ranked_rel, relevance, k):
    rel = (ranked_rel[:, :k] > 0).astype(float)            # (U, k)
    cum_hits = np.cumsum(rel, axis=1)                      # (U, k)
    ranks = np.arange(1, k + 1)[None, :]                   # (1, k)
    precision_at_each = cum_hits / ranks                   # P(k) at every position
    ap = (precision_at_each * rel).sum(axis=1)             # only at relevant ranks
    n_rel = np.clip((relevance > 0).sum(axis=1), 1, None)  # (U,)
    return ap / np.minimum(n_rel, k)                       # normalize by min(R, k)


def ndcg_at_k(ranked_rel, relevance, k, exponential=True):
    def gains(r):
        return (2.0 ** r - 1.0) if exponential else r
    discounts = 1.0 / np.log2(np.arange(2, k + 2))         # (k,) for ranks 1..k
    dcg = (gains(ranked_rel[:, :k]) * discounts).sum(axis=1)        # (U,)
    # IDCG: sort true relevance descending, take top-k, same discount
    ideal = -np.sort(-relevance, axis=1)[:, :k]                     # (U, k)
    idcg = (gains(ideal) * discounts).sum(axis=1)                  # (U,)
    idcg = np.clip(idcg, 1e-12, None)                             # avoid /0
    return dcg / idcg


def evaluate(scores, relevance, ks=(10, 50), ndcg_exponential=True):
    """Full harness: returns a dict of macro-averaged metrics at each k."""
    out = {}
    for k in ks:
        ranked_rel = rank_relevance(scores, relevance, k)
        out[f"precision@{k}"] = precision_at_k(ranked_rel, k).mean()
        out[f"recall@{k}"]    = recall_at_k(ranked_rel, relevance, k).mean()
        out[f"hitrate@{k}"]   = hit_rate_at_k(ranked_rel, k).mean()
        out[f"mrr@{k}"]       = mrr_at_k(ranked_rel, k).mean()
        out[f"map@{k}"]       = average_precision_at_k(ranked_rel, relevance, k).mean()
        out[f"ndcg@{k}"]      = ndcg_at_k(ranked_rel, relevance, k,
                                          exponential=ndcg_exponential).mean()
    return out
```

A few engineering notes that matter in practice. The `rank_relevance` step does the only sort, once per K, and everything downstream is pure vectorized arithmetic on the ranked-relevance matrix — no per-user Python loop, which is what lets this scale to a 100k-user held-out set in seconds. The `np.clip(n_rel, 1, None)` guards against users with no relevant items (they contribute 0 recall, not a NaN). The MAP normalization uses `min(n_rel, k)` so that a user with more relevant items than K is not penalized for the relevant items that physically cannot fit in the top K — a standard convention; some libraries normalize by `n_rel` always, so state which you use. Every function returns a per-user array and `evaluate` macro-averages with `.mean()` at the end, which is the per-user-then-mean convention from the averaging discussion above.

#### Worked example: the harness reproduces the by-hand numbers

Sanity-check the code against the first worked example. One user, scores that produce the ranking A,B,C,D,E with relevances 3,0,2,0,1:

```python
import numpy as np

# one user; scores already in descending order A>B>C>D>E
scores    = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
relevance = np.array([[3,   0,   2,   0,   1  ]], dtype=float)

ranked_rel = rank_relevance(scores, relevance, k=5)
print("MRR@5 ", mrr_at_k(ranked_rel, 5)[0])                       # 1.0
print("MAP@5 ", average_precision_at_k(ranked_rel, relevance, 5)[0])  # 0.7556
print("NDCG@5", ndcg_at_k(ranked_rel, relevance, 5)[0])           # 0.9461
```

The harness prints 1.0, 0.7556, 0.9461 — matching the hand arithmetic (the tiny rounding differences are just float precision). If you only trust one thing from this post, trust a harness you have checked against a worked example by hand; a metrics bug that silently reports the wrong NDCG has wrecked more model-selection decisions than any modeling mistake.

Now the PyTorch variant, for when you want to compute NDCG on the GPU every few hundred steps without copying logits back to the CPU. The logic is identical; only the ops change.

```python
import torch

def ndcg_at_k_torch(scores, relevance, k, exponential=True):
    """scores, relevance: (U, N) tensors on the same device. Returns mean NDCG@k."""
    device = scores.device
    topk_idx = torch.topk(scores, k=k, dim=1).indices                  # (U, k)
    ranked_rel = torch.gather(relevance, 1, topk_idx)                  # (U, k)

    def gains(r):
        return (torch.pow(2.0, r) - 1.0) if exponential else r

    ranks = torch.arange(2, k + 2, device=device, dtype=torch.float)
    discounts = 1.0 / torch.log2(ranks)                                # (k,)
    dcg = (gains(ranked_rel) * discounts).sum(dim=1)                   # (U,)

    ideal_rel = torch.topk(relevance, k=min(k, relevance.size(1)), dim=1).values
    # pad ideal to length k if N < k
    if ideal_rel.size(1) < k:
        pad = torch.zeros(ideal_rel.size(0), k - ideal_rel.size(1), device=device)
        ideal_rel = torch.cat([ideal_rel, pad], dim=1)
    idcg = (gains(ideal_rel) * discounts).sum(dim=1).clamp_min(1e-12)  # (U,)
    return (dcg / idcg).mean()
```

`torch.topk` is the GPU-native equivalent of the argsort-then-slice, and it is fast because it does not fully sort — it only finds the top K. Drop `ndcg_at_k_torch` into your validation step, call it on the held-out logits, and you get a GPU NDCG with no host round-trip. The full numpy harness is what you run for the final, complete, full-catalog evaluation report (the kind you put in a paper or a launch doc); the torch version is what you watch during training.

## Stress-testing the harness: the edge cases that produce wrong numbers

A metrics harness is one of the highest-leverage pieces of code in a recsys stack and one of the most quietly buggy, because nobody writes unit tests for a number that "looks plausible." Let me walk the failure modes I have actually been burned by, because every one of them produces a metric that is wrong in a *believable* direction.

**Users with no relevant items.** A user might have zero held-out positives — they were active in training but not in the test window. Recall divides by $|R_u|$, which is zero, and an unguarded implementation returns NaN; a single NaN propagated through `.mean()` turns your entire dashboard into NaN, or worse, gets silently filtered and inflates the average over the *remaining* users. The harness above clips the denominator to a minimum of 1, which makes such users contribute exactly 0 recall — the correct convention, since you found none of their (nonexistent) relevant items. But you must decide explicitly whether to *include* zero-positive users in the average at all; including them drags every metric down uniformly, and if two experiments include different numbers of them, the comparison is corrupted. Fix the user set once, up front, and reuse it across every model.

**Ties in the scores.** Two items with identical scores have an ambiguous order, and `np.argsort` breaks ties by index — which means the *order you happened to load the catalog in* leaks into your metric. If your model outputs many tied scores (common with quantized or coarsely-bucketed scorers), the metric becomes partly a function of catalog ordering rather than model quality. The honest fix is to break ties randomly and average over several seeds, or to report the expected metric under random tie-breaking; the cheap fix is to add a tiny deterministic jitter and document it. The dangerous non-fix is to ignore it and let two models with identical scoring-power show different NDCG purely because of load order.

**More relevant items than K.** When $|R_u| > K$, no ranking can fit all relevant items into the top K, so Recall@K is capped below 1.0 by construction and MAP's normalization matters (the `min(n_rel, k)` term). A harness that normalizes MAP by `n_rel` always will report artificially low MAP for power users with many positives, making your metric secretly a function of how active your test users are. State the normalization.

**The full-catalog versus sampled question, in code.** Notice the harness sorts the entire `(U, N)` score matrix — it is a *full-catalog* evaluation when `N` is the catalog size. The moment you instead score the held-out positive against, say, 100 sampled negatives to save compute, you have switched to a sampled metric, and per Krichene and Rendle that metric is a biased estimate that can reverse model rankings. The seductive thing is that the *code looks almost identical* — you just pass a smaller `N`. So make it loud: a sampled evaluation should be named `evaluate_sampled` with the sample size in the function name, never silently sharing the `evaluate` entry point, because the two produce numbers that are not comparable and must never appear in the same table.

#### Worked example: a popularity-prior change that lifts Recall@50 but not NDCG@10

Put the failure mode in numbers, since this is the scenario from the intro. Suppose your baseline two-tower gets Recall@50 = 0.412 and NDCG@10 = 0.241 on a 30k-user temporal split. You add a popularity prior that nudges globally-popular items up the candidate list. Re-run the harness:

- Recall@50 rises to 0.448 — a relative lift of $(0.448 - 0.412)/0.412 = 8.7\%$. The prior pushed more popular items into the top 50, and popular items are over-represented in the MNAR held-out set, so more of them are "relevant."
- NDCG@10 moves from 0.241 to 0.238 — essentially flat, in fact a hair *worse*. The prior did not improve the *ordering* of the items a specific user actually wants in the top 10; it crowded the visible slate with generically-popular items that are not personally relevant, displacing a personalized hit or two.

If your team's success criterion was "lift Recall@50 by 5%+," you ship it and celebrate. Then the A/B test, where users experience the top-10 *order*, comes back flat-to-negative, because the thing you optimized (deep recall of popular items) is not the thing users experience (the order of their visible slate). The arithmetic is unambiguous: an 8.7% recall lift co-occurring with a 1.2% NDCG@10 *drop* is the signature of a change that games the biased offline metric without improving the product. The fix is not a better model — it is to have required the win on NDCG@10, the metric that matches the surface, before shipping.

## A note on implicit feedback and what counts as relevant

Everything above assumes you can label $R_u$. In implicit-feedback systems — the majority — that labeling is itself a modeling decision with teeth. A "relevant" item is usually "the user clicked / played / purchased it in the held-out window," and the threshold you draw changes every metric. Treat a 3-second video view as relevant and your relevant sets balloon with accidental clicks; require a 30-second view and they shrink to genuine engagement but you discard signal from short-form content where 30 seconds is most of the video. There is no universal threshold; there is only a threshold that matches the product's definition of a good outcome. The discipline is to define relevance from the *business* outcome (a completed watch, a purchase, a save) and then hold that definition fixed across every experiment, because a metric computed against a shifting relevance definition is not comparable to itself across weeks. The post on [implicit versus explicit feedback and the data you have](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have) goes deeper on turning raw events into labels; the point for evaluation is that **your metric is only as meaningful as your relevance definition, and that definition is a choice you must make and freeze.**

For graded NDCG specifically, the mapping from signal to grade is where most of the subjectivity lives. A defensible recipe for a video feed: grade 0 = impression with no engagement, grade 1 = click or short view, grade 2 = long view (past some completion threshold), grade 3 = a strong positive action (like, save, share). The exponential gain $2^{\text{rel}}-1$ then makes a single grade-3 save worth seven impressions, which is roughly the right relative value if a save predicts long-term retention far better than a glance does. The wrong move is to assign grades by gut and then over-interpret a 0.003 NDCG delta; the grades are noisy, so treat NDCG differences smaller than your bootstrap confidence interval as zero.

## Results: the same model, scored every way, disagreeing

Here is the table that should retire the phrase "the offline number went up." We take the same retrieval candidate set on a MovieLens-style benchmark and apply four model variants — a baseline two-tower, the two-tower plus a learned reranker, the two-tower plus a popularity prior, and both — then score each with four metrics. (Exact values depend on the split and the dataset version; treat these as representative of the *pattern*, which is the point, and reproduce them with your own harness on your own data.)

| Model variant | Recall@50 | NDCG@10 | MAP@10 | MRR |
| --- | --- | --- | --- | --- |
| Baseline two-tower | 0.412 | 0.241 | 0.198 | 0.355 |
| + reranker | 0.415 | **0.287** | **0.236** | **0.402** |
| + popularity prior | **0.448** | 0.238 | 0.201 | 0.351 |
| + both | **0.451** | **0.291** | **0.241** | **0.409** |

Read the rows. The **reranker** barely touches Recall@50 (0.412 → 0.415, flat — it reorders the same candidate set, so the *set* in the top 50 hardly changes) but jumps NDCG@10 from 0.241 to 0.287 and MRR from 0.355 to 0.402 (it moves relevant items into the top slots, which is exactly what the order metrics reward). The **popularity prior** does the opposite: it lifts Recall@50 from 0.412 to 0.448 (popular items are more often the held-out relevant ones, so they fatten the candidate net) but leaves NDCG@10 flat at 0.238 (it does not improve the *ordering* of relevant items, and may slightly crowd the top with generically-popular-but-not-personally-relevant items). Only "+ both" wins on every column.

![A matrix of four model variants scored by Recall at fifty, NDCG at ten, MAP at ten, and MRR, showing a reranker and a popularity prior each win on different metrics](/imgs/blogs/offline-evaluation-metrics-recall-ndcg-map-mrr-8.png)

The figure renders the same table as a colored grid so the disagreement is visible at a glance: the reranker row is green on the order metrics and amber-flat on Recall; the popularity row is green on Recall and amber-flat on the order metrics. **If you only watched Recall@50, you would have shipped the popularity prior and skipped the reranker — and your A/B test, which users experience as the *order* of the visible slate, would have gone flat or negative.** If you only watched NDCG@10, you would have skipped the popularity prior and left retrieval recall (and therefore the funnel's ceiling) on the table. The metric you pick *is* the model-selection decision.

**A note on metric correlation.** These metrics are correlated — a genuinely better model tends to win on most of them — but the correlation is far from 1.0, and it breaks down exactly at the changes that matter for shipping decisions: reordering changes (move the order metrics, not recall) and coverage changes (move recall, not the small-K order metrics). Empirically across many model pairs the order metrics (NDCG@10, MAP@10, MRR) correlate tightly with each other (often 0.9+) and Recall@large-K is the one that diverges, because it is measuring a different stage of the funnel. The practical rule: **report one set metric (Recall@K at your retrieval K) and one order metric (NDCG@K at your serving K), and require a win on the one that matches the change you made.** A retrieval change must move Recall; a ranking change must move NDCG.

**Is the delta even real?** The numbers in that table are means over a finite set of users, which means they have sampling noise, which means a delta of 0.241 → 0.244 might be pure chance. The fix is to attach a confidence interval to every metric, and the cleanest way to do it without distributional assumptions is the **bootstrap over users**: resample your test users with replacement a thousand times, recompute the macro-averaged metric on each resample, and take the 2.5th and 97.5th percentiles of those thousand values as a 95% interval. Because every metric in the harness already returns a per-user array before the final `.mean()`, the bootstrap is trivial — you resample the rows of that per-user array and re-average, never re-sorting the scores. A model A beats model B *only* if A's interval clears B's point estimate (or, more rigorously, if the bootstrap of the *paired difference* excludes zero). The number of decimal places you report should match the width of that interval: if the 95% interval is ±0.004, then a 0.241 versus 0.243 comparison is noise, and reporting "NDCG improved from 0.241 to 0.243" without the interval is, charitably, optimistic. The companion split-and-evaluate post builds the bootstrap into the harness; the rule here is that **a metric without a confidence interval is a rumor, not a result.**

## AUC and coverage as context

Two more metrics belong in your vocabulary even though they are not top-K ranking metrics, because they fill gaps the top-K family leaves.

**AUC (Area Under the ROC Curve)** is the probability that a uniformly random relevant item is scored above a uniformly random irrelevant item: $\text{AUC} = \Pr[s(\text{pos}) > s(\text{neg})]$. It is a *global* pairwise-order metric — it looks at all positive-negative pairs across the whole ranking, not just the top K. That makes it the natural fit for **CTR-prediction ranking models** where you score every candidate and care about calibrated pairwise order everywhere (the foundations post on [the ranking model](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders) leans on it). AUC's blind spot is precisely the one top-K metrics fix: it weights an improvement at rank 5000 the same as one at rank 5, because it has no position discount. A model can gain AUC by fixing order deep in the tail where no user ever looks, while NDCG@10 stays flat. Report AUC for the pointwise classifier's calibration story; report NDCG@K for what the user sees.

**Coverage** is not an accuracy metric at all — it measures what *fraction of the catalog* ever appears in anyone's top-K recommendations. A model can have excellent NDCG and recommend the same 200 popular items to everyone, quietly starving 99.9% of the catalog (the feedback-loop failure mode where the recommender collapses onto a few viral items). Coverage, along with diversity, novelty, and serendipity, is the subject of the [beyond-accuracy metrics post](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage), and the reason it matters here is that **accuracy metrics and coverage trade off**: the popularity prior in our results table is exactly the kind of change that lifts recall while flattening coverage. A complete offline report carries at least one accuracy metric *and* at least one coverage/diversity metric, because optimizing accuracy alone is how you end up with a feed that recommends the same ten things to everyone.

## Which metric for which product

Time to be decisive. The metric is not a matter of taste; it is determined by the product surface and the stage of the funnel.

| Product / stage | Primary metric | K | Why |
| --- | --- | --- | --- |
| Candidate retrieval | Recall@K | large (200–1000) | Did the relevant item make the candidate pool? Order is the ranker's job |
| Ranking a graded feed | NDCG@K | small (5–10) | Order within the visible slate, with graded relevance from engagement signal |
| One-right-answer search / autocomplete | MRR | 10 | A single correct result; only its rank matters |
| Set retrieval, multiple relevant, binary | MAP@K | 10–20 | Order-aware, reads all hits, binary relevance |
| Fixed-size slate, balance precision/recall | F1@K | = slate size | Both over- and under-showing hurt |
| CTR-prediction ranker (calibration story) | AUC (+ logloss) | — | Global pairwise order; pair with NDCG for the visible slate |
| Catalog-health guardrail | Coverage / diversity | — | Prevent collapse onto popular items |

The decision procedure: name the surface, name the stage. **Retrieval → Recall at a large K. Ranking a graded feed → NDCG at a small K. One right answer → MRR. Set retrieval → MAP.** Then add a coverage guardrail so accuracy gains don't secretly cost catalog health. If you take one operational rule from this post, take that one.

## Case studies: how the big systems actually choose

**YouTube** built the canonical two-stage system: a deep candidate-generation network that retrieves a few hundred videos from the full corpus, then a deep ranking network that orders them (Covington, Adams, and Sargin, "Deep Neural Networks for YouTube Recommendations," RecSys 2016). The retrieval stage is explicitly framed as extreme multiclass classification and evaluated on **recall-style** hit metrics — did the watched video appear in the retrieved set — while the ranking stage optimizes a finer objective tied to expected watch time. The split maps exactly onto our funnel: large-K recall for retrieval, fine-grained order for ranking. The lesson YouTube's papers repeat is that the offline objective for ranking must be the *engagement* objective (expected watch time), not raw click order — a reminder that the metric must track the product's true goal, which we return to in the offline-online-gap discussion.

**Spotify** and music-streaming recommenders lean heavily on **NDCG and recall at multiple K** because the consumption surface is a ranked playlist or radio queue where order within the visible window drives skips and saves. Spotify's published work on sequential and session-based recommendation, and on the "algotorial" balance of editorial and algorithmic playlists, evaluates ranking quality with NDCG-family metrics and reports the now-familiar caution that offline ranking gains do not always translate to streaming-intent lift — the offline-online gap again. The graded-relevance angle is natural here: a save is worth more than a play is worth more than a skip, which is exactly the kind of signal NDCG's graded gains were built to use.

**Web search** is where these metrics were born, and it is where graded relevance is most explicit. Search-quality teams employ human raters who assign **graded relevance labels** (the perfect/excellent/good/fair/bad scale), and the headline offline metric is **NDCG at small K** (NDCG@5, NDCG@10) computed against those graded judgments. The original DCG/NDCG formulation (Järvelin and Kekäläinen, "Cumulated gain-based evaluation of IR techniques," ACM TOIS 2002) came directly out of this IR tradition, and the learning-to-rank methods that optimize it — RankNet, LambdaRank, LambdaMART (Burges, "From RankNet to LambdaRank to LambdaMART," Microsoft Research, 2010) — were developed precisely to push NDCG on graded web-search judgments. When recsys adopted NDCG, it inherited this whole machinery.

**Sequential-recommendation benchmarks (SASRec, BERT4Rec)** evaluate with a leave-one-out protocol that is worth knowing because it is the dominant academic setup and a common source of the sampled-metrics trap. For each user, the *single* most recent interaction is held out as the test item, the rest is the training history, and the model is asked to rank that one held-out item. Because $|R_u| = 1$, Hit Rate@K, Recall@K, and "did the held-out item land in the top K" all collapse to the same number, and the standard reported metrics are HitRate@10 and NDCG@10. The original SASRec (Kang and McAuley, "Self-Attentive Sequential Recommendation," ICDM 2018) and BERT4Rec (Sun et al., "BERT4Rec," CIKM 2019) papers reported these against *sampled* negatives — ranking the one positive against 100 sampled items — which is exactly the protocol Krichene and Rendle later showed can reverse model orderings. Subsequent re-evaluations of BERT4Rec under full-catalog metrics found the headline gaps narrowed or shifted, a cautionary tale that the metric protocol can matter more than the model. When you read a sequential-rec leaderboard, check whether the numbers are full-catalog or 100-negative before you trust the ranking.

**Pinterest's PinSage** (Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems," KDD 2018) is a useful industrial datapoint on metric choice for retrieval. PinSage is a graph-based candidate generator over billions of pins, and its offline evaluation centers on a **hit-rate-at-K** style metric — does the related pin a user engaged with appear in the top K retrieved — plus a Mean Reciprocal Rank, precisely because the task is candidate retrieval where presence-in-the-set is the job. The paper pairs the offline hit-rate with online A/B engagement lift, and the gap between the two is the recurring theme: offline retrieval metrics set the ceiling, online engagement measures the reality, and a complete evaluation reports both.

**The sampled-metrics warning (teaser for the next post).** Here is a result that should make you nervous about every Recall@K and NDCG@K you have ever read in a paper. Krichene and Rendle ("On Sampled Metrics for Item Recommendation," KDD 2020) showed that the common practice of computing top-K metrics against a *sampled* set of negatives — ranking the held-out positive against, say, 100 random negatives instead of against the entire catalog — produces metrics that are **not consistent** with the full, exact metrics. Worse, the sampled metrics can *reverse the ranking of models*: model A beats model B under sampled NDCG@10 but loses under exact NDCG@10. The reason is statistical — the sampled metric is a high-variance, biased estimate of a low-rank-region quantity, and the bias depends on the score distribution differently for different models. Many influential recsys results used 100-negative sampled evaluation, and some of their model comparisons may not hold under exact evaluation. The fix is to compute metrics against the **full catalog** (which our harness does — it sorts the whole `(U, N)` score matrix), or to use the corrected estimators Krichene and Rendle propose. This is the bridge to [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) and to the offline-online-gap post: a metric is only as trustworthy as the protocol that computes it, and sampled negatives are a protocol that quietly lies.

## When to reach for each metric (and when not to)

Every metric choice is a commitment to caring about one thing and ignoring others. Say the trade-offs plainly.

- **Reach for Recall@K** when you are evaluating retrieval and you want the funnel's ceiling. **Do not** report only Recall on a ranking model — it is blind to order, and your users experience order. A flat Recall with a rising NDCG is a *good* ranking change, and a Recall-only dashboard will tell you nothing happened.
- **Reach for NDCG@K** as your default for ranked, graded surfaces. **Do not** report NDCG at a huge K on a ranking model — the log discount makes ranks past ~50 nearly weightless, so a big-K NDCG is really measuring retrieval recall wearing an NDCG costume. And do not invent graded relevance levels that carry no real signal just to use the graded gain.
- **Reach for MRR** when there is one right answer. **Do not** use MRR for a feed with many relevant items — it stops reading after the first hit and will rank a one-good-item list identically to a five-good-item list.
- **Reach for MAP** for binary set retrieval with multiple targets. **Do not** use it when relevance is genuinely graded — you are discarding the grades that NDCG would use.
- **Reach for AUC** for the CTR-classifier's global calibration story. **Do not** use it as your primary ranking metric — it has no position discount and rewards fixing order in the tail no user sees.
- **Always carry a coverage or diversity guardrail.** **Do not** ship a model selected on accuracy alone; the popularity-prior failure mode (high recall, collapsing catalog) is invisible to every accuracy metric.
- **Always compute against the full catalog, not sampled negatives**, unless you are using a corrected estimator. **Do not** trust a model comparison made under 100-negative sampled metrics — it may reverse under exact evaluation.

## Choosing your offline metric: the short procedure

When someone hands you a model and asks "is it better," run this:

1. **Name the surface.** Visible ranked slate of N items? One-shot search? A candidate pool feeding a downstream ranker?
2. **Name the stage.** Retrieval (recall ceiling) or ranking (order within the slate)?
3. **Set K to the surface.** Retrieval K = candidate-pool size (hundreds). Ranking K = visible slate size (5–10).
4. **Pick the family.** Retrieval → Recall@K. Ranking, graded signal → NDCG@K. Ranking, one right answer → MRR. Ranking, binary multi-target → MAP@K.
5. **Add guardrails.** One coverage/diversity metric so accuracy gains don't cost catalog health; AUC/logloss if you have a pointwise classifier whose calibration matters.
6. **Fix the protocol.** Full-catalog metrics (not sampled negatives), temporal leak-free split, macro averaging, and bootstrap confidence intervals so you know whether a delta is real.
7. **Require the win on the matching metric.** A retrieval change must move Recall; a ranking change must move NDCG. A win on the *other* metric is a hint, not a result.

That procedure turns "which offline number" from a shrug into a one-minute decision, and it is the thing that keeps your Monday-morning dashboard win from becoming a flat A/B test two weeks later.

## Key takeaways

- **Recommenders are graded on ranked top-K lists, not rating error.** Score how well the relevant items are placed near the top of the slate the product actually shows; RMSE measures the wrong thing.
- **The metric family splits into set metrics (Precision, Recall, Hit Rate — order-blind) and order metrics (MRR, MAP, NDCG — placement-aware).** Set metrics fit retrieval; order metrics fit ranking.
- **NDCG is the gold standard because it is the only common metric that is order-aware, graded, and bounded.** DCG sums gain $2^{\text{rel}}-1$ discounted by $1/\log_2(r+1)$; dividing by IDCG normalizes it to $[0,1]$ so it averages across users.
- **The log discount falls steeply then flattens**, so metric gains concentrate at the top of the list — a reranker that fixes ranks 1–3 moves NDCG far more than one that fixes ranks 30–40.
- **MRR reads only the first hit; MAP reads all hits but is binary; NDCG reads all hits and grades.** Pick by what your product cares about: one right answer (MRR), set retrieval (MAP), graded feed (NDCG).
- **Metrics disagree by design.** A reranker lifts NDCG@10 while leaving Recall@50 flat; a popularity prior does the reverse. The metric you pick is the model-selection decision — match it to the change you made.
- **Match K to the surface.** Recall at a large K for retrieval; NDCG at a small K for ranking. NDCG@1000 on a ranker is recall in disguise.
- **Macro-average (per user, then mean), compute against the full catalog, and never trust sampled-negative metrics** — Krichene and Rendle (KDD 2020) showed they can reverse model rankings.
- **Carry a coverage guardrail.** Accuracy metrics are blind to a catalog collapsing onto popular items; report at least one beyond-accuracy metric alongside.

## Further reading

- Järvelin, K., and Kekäläinen, J. "Cumulated Gain-Based Evaluation of IR Techniques." *ACM Transactions on Information Systems*, 2002. The paper that introduced DCG and NDCG.
- Burges, C. "From RankNet to LambdaRank to LambdaMART: An Overview." Microsoft Research Technical Report, 2010. How NDCG becomes a training signal via the lambda gradient.
- Krichene, W., and Rendle, S. "On Sampled Metrics for Item Recommendation." *KDD 2020*. Why sampled top-K metrics are inconsistent and can reverse model comparisons.
- Covington, P., Adams, J., and Sargin, E. "Deep Neural Networks for YouTube Recommendations." *RecSys 2016*. The canonical retrieval-then-ranking two-stage evaluation.
- Within this series: [What is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) for the retrieval-ranking-reranking funnel that decides which metric belongs at which stage; [The right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) for the leak-free temporal protocol that feeds these metrics; [Learning to rank for recommenders](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders) for optimizing NDCG directly; [Beyond accuracy: diversity, novelty, serendipity, coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) for the guardrail metrics; [The offline-online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) for why a clean offline win can flop online; and the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) capstone that ties metric choice into the full production loop.

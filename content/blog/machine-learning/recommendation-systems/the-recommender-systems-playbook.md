---
title: "The Recommender Systems Playbook"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The single page that ties the whole series together: the end-to-end mental model, the decision trees for model, loss, negatives and metric, a staged build order, a production checklist, the ten pitfalls, and a worked design walkthrough you can apply tomorrow."
tags:
  [
    "recommendation-systems",
    "recsys",
    "retrieval",
    "ranking",
    "two-tower",
    "learning-to-rank",
    "evaluation",
    "machine-learning",
    "playbook",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/the-recommender-systems-playbook-1.png"
---

You have been handed a product with two million items and a home feed that is blank. Or worse: a home feed that already exists, runs on a model someone left behind, and quietly recommends the same ten viral items to everyone while the dashboard insists "engagement is up." You have read the fifty-nine posts that came before this one, each going deep on a single piece: [collaborative filtering](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles), the [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), [sampled softmax](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval), [calibration](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust), the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied). Now you need the one page that puts it all in order and tells you what to build first, what to measure, and where it will break.

This is that page. It is the capstone of **Recommendation Systems: From Click to Production**, and its job is not to teach a new technique. Its job is to be the map you open when you are mid-decision: which retrieval model for this scale, which loss for this objective, which negative sampler, which metric to trust, when an LLM is worth the latency and when a boring two-tower plus a gradient-boosted ranker is the right call. Everything here links back to the post that derives it, so you can drop down a level whenever you need the math or the code.

The spine of the whole series, the thing every track hangs from, is one picture: a **retrieval → ranking → re-ranking funnel** fed by a **feedback loop** (serve → log → train → serve), read off the **offline↔online** reality gap. If you remember nothing else, remember that the model you train is trained on logs the previous model generated, and the metric you optimize offline is not the metric you get paid for online. The figure below is the full stack we are going to navigate.

![Diagram of the complete recommender stack showing raw events flowing into a feature store, then retrieval, ranking and re-ranking stages narrowing the catalog to about twenty served items, with served impressions logged back as the next training set](/imgs/blogs/the-recommender-systems-playbook-1.png)

By the end of this post you will be able to walk into a design review and, for a brand-new product, name the model, the loss, the negatives, the metric, the build order, and the three pitfalls that will bite first, with a reason for each. We will close with a full worked walkthrough that designs a recommender from scratch by applying every decision tree in sequence. Let us start with the mental model that all of it sits on.

## 1. The mental model: a funnel fed by a loop, read off a gap

Strip every recommender down and you find the same three-part skeleton. First, **the funnel narrows the catalog**. You cannot score a billion items with an expensive model in 100 milliseconds, so you stage the work: a cheap **retrieval** step turns $10^9$ items into roughly 500 candidates, an expensive **ranking** step turns 500 into about 50, and a **re-ranking** step applies diversity and business rules to land on the 20 a user sees. The [recommendation funnel post](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) derives why this staging is forced rather than chosen.

Here is the arithmetic, because it is the load-bearing fact of the whole field. Suppose your catalog has $N = 10^9$ items and your deep ranker costs about $1$ ms of compute per scored item (feature lookup, embedding gather, a few dense layers). Scoring the entire catalog per request costs

$$
N \times t_{\text{rank}} = 10^9 \times 1\,\text{ms} = 10^6\,\text{seconds} \approx 11.6\,\text{days}.
$$

That is not a tuning problem. It is off by eight orders of magnitude. A retrieval model that is a single dot product between a user vector and item vectors, served by [approximate nearest neighbor (ANN) search](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann), returns the top 500 in single-digit milliseconds. The ranker then scores only those 500. The funnel is what makes the latency budget close at all:

$$
10^9 \xrightarrow{\text{ANN, } \sim5\text{ms}} 500 \xrightarrow{\text{deep ranker}} 50 \xrightarrow{\text{re-rank}} 20.
$$

Second, **the loop feeds the funnel**. The system serves a list, logs which items were shown and which were clicked, and trains the next model on those logs. This is the engine of the whole thing and also its single biggest hazard. The data is **missing-not-at-random**: you only observe clicks on items the previous model chose to show. A great item that was never retrieved has no impressions, so no clicks, so the model learns it is uninteresting, so it stays unretrieved. That is the [feedback loop and filter bubble](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) failure mode, and it is why [popularity bias](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) is a self-reinforcing fixed point rather than a passing artifact.

Third, **you read all of it off the gap**. Your offline metric is computed on logged data shaped by the old policy; your online metric is what a live user does under the new policy. These two diverge for three reasons that every practitioner must internalize: distribution shift (the new model shows different items than the logs contain), missing-not-at-random feedback, and [position bias](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data) (higher slots get more clicks regardless of relevance). The [offline-online gap post](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) makes this rigorous; the [two worlds of recsys post](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) sets the frame. The practical consequence is blunt: **an offline win is a hypothesis, not a result.** The result is an [A/B test](/blog/machine-learning/recommendation-systems/ab-testing-recommenders).

Everything else in this series is a refinement of one of those three parts. Models and ANN serving make the funnel work. Losses, negatives, and bias corrections make the loop healthy. Metrics, splits, and off-policy estimators close the gap. Keep the funnel-loop-gap triple in your head and you can place any new paper or any new bug into exactly one slot.

It is worth pausing on *why* a recommender is not just a classifier with a big output layer, because that framing is the source of half the early mistakes. A classifier assumes a fixed label set, independent and identically distributed train and test data, and clean positive and negative labels. A recommender breaks all three. The label space is a high-cardinality ID space of millions of items, each appearing a handful of times. The train and test data are not from the same distribution — the training data is generated by the very model you are about to replace. And the feedback is positive-only and implicit: a click is a noisy positive, but the absence of a click is *not* a negative, because the user may simply never have been shown the item. The [data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders) post catalogs what you actually get to work with, and [embeddings the heart of recommenders](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders) explains why a learned embedding per ID — not one-hot encoding — is the only sane way to represent that sparse ID space. Hold those breaks in mind; they are the reason every downstream decision in this playbook looks the way it does.

#### Worked example: why RMSE is the wrong objective

A team trains a [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) model to predict 1-to-5 star ratings and proudly reports test RMSE dropping from $0.92$ to $0.88$. They ship it and top-K engagement does not move. Why? RMSE rewards predicting the absolute rating accurately across all items, including the thousands of mediocre items a user will never see. The product only shows the top 20. A model can shave RMSE by nailing the middle of the distribution while shuffling the order of the top items, which is the only order that matters. The fix is to optimize a ranking objective directly, which is the entire point of [framing the problem as ranking and retrieval](/blog/machine-learning/recommendation-systems/framing-the-problem-rating-ranking-retrieval) rather than rating prediction. Same data, same model class, different loss, and the online needle moves. The lesson generalizes: choose the objective to match what the funnel actually serves, not what is easiest to measure.

## 2. The master decision tree: which model?

The most common question in a design review is "what model should we use?" and the most common mistake is answering it before answering "for which stage?" Retrieval and ranking are different problems with different constraints, and the same architecture is rarely best at both. Decide the stage first. Then decide your data regime: do you have only warm IDs, or do you need to serve cold items with side features? Only at the leaves do specific architectures appear. The tree below is the one I actually use.

![Decision tree for choosing a recommender model that forks first on retrieval versus ranking stage, then on the data regime, and only reaches specific architectures such as iALS, two-tower, GBDT and SASRec at the leaves](/imgs/blogs/the-recommender-systems-playbook-2.png)

**Retrieval branch.** Your constraint is brutal: score the whole catalog in milliseconds. That forces a model whose item representation can be precomputed and indexed, which in practice means a dot-product or cosine score between a user embedding and item embeddings.

- **Start with a non-personalized popularity baseline and item-item collaborative filtering.** Yes, really. Popularity is the floor every model must beat, and [item-item CF](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles) ("users who interacted with X also interacted with Y") is shockingly strong, cheap, and interpretable. If you cannot beat it, you have a data problem, not a model problem.
- **For warm IDs at moderate scale, use [implicit matrix factorization (iALS) or BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr).** It is a few-line `implicit` call, trains in minutes, and gives you a real personalized retrieval baseline with learned user and item vectors. The [MF post](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) covers the workhorse; the [implicit-vs-explicit feedback post](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have) tells you which loss your data supports.
- **When you need cold-start coverage or have rich side features, move to the [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval).** Two encoders, one for the user/context and one for the item, project into a shared embedding space; the item tower can ingest text, category, and other [content features](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) so a brand-new item gets a sensible vector before it has any interactions. This is the default modern retrieval model. Training it well is a topic of its own, covered in [training two-tower negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax).
- **Reach for a [graph neural network](/blog/machine-learning/recommendation-systems/graph-neural-networks-for-recommendation) only when you have a real graph** (social edges, co-view, co-purchase) that a two-tower cannot capture, and you can afford the engineering. PinSage-style models lift recall when the graph is informative, but they are not a free upgrade.
- **Generative retrieval (semantic IDs, TIGER-style) is the frontier**, covered in [autoencoders and the road to generative retrieval](/blog/machine-learning/recommendation-systems/autoencoders-and-the-road-to-generative-retrieval) and [generative and conversational recommendation](/blog/machine-learning/recommendation-systems/generative-and-conversational-recommendation). It is promising and unstable; treat it as research unless you have a specific reason.

**Ranking branch.** Now you score hundreds of candidates with full features and can afford a heavy model.

- **Start with logistic regression or a [gradient-boosted decision tree (GBDT)](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations).** GBDTs are the unreasonably effective baseline on tabular features; they handle non-linearities and missing values, train fast, and are hard to beat without a lot of data.
- **Move to deep CTR models when you have many high-cardinality sparse features whose interactions matter.** [Wide and Deep](/blog/machine-learning/recommendation-systems/wide-and-deep-and-the-memorization-generalization-tradeoff) gives you memorization plus generalization, [DeepFM](/blog/machine-learning/recommendation-systems/deepfm-and-automatic-feature-interactions) learns second-order interactions automatically, and [DCN](/blog/machine-learning/recommendation-systems/dcn-and-explicit-feature-crossing) crosses features explicitly at bounded degree.
- **Use a [sequential model (SASRec, BERT4Rec)](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) when order carries signal** — sessions, "next item," recency-driven intent. The [sequential and session-based recommendation post](/blog/machine-learning/recommendation-systems/sequential-and-session-based-recommendation) frames when sequence beats a bag of features.
- **Use [multi-task MMoE or PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) when you must trade off several objectives** (click, like, watch-time, conversion) and a single weighted sum is causing seesaw, where lifting one metric drops another.

The honest default for a new product is: **popularity → iALS → two-tower for retrieval, GBDT → deep CTR for ranking.** Everything fancier is something you earn by hitting a wall that the default cannot clear.

A note on a model that newcomers reach for too early: [neural collaborative filtering](/blog/machine-learning/recommendation-systems/neural-collaborative-filtering-and-its-critique). Replacing the dot product of matrix factorization with a multi-layer perceptron over concatenated user and item embeddings *feels* like an upgrade, but the original NCF critique showed that a well-tuned dot product often matches or beats the MLP, and the dot product is what lets you serve retrieval with ANN. A learned MLP scoring function cannot be precomputed and indexed, so it cannot serve the first arrow of the funnel at scale. This is a recurring theme: **the retrieval model's scoring function must be a dot product (or another ANN-friendly metric) or you cannot serve it.** That single constraint rules out a lot of architectures for the retrieval stage and is why the two-tower shape — two independent encoders that meet only at a final dot product — dominates. The [factorization machines](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm) family sits in an interesting middle: it generalizes matrix factorization to arbitrary feature interactions and is a strong ranking baseline, while its second-order structure keeps it cheap.

The other axis the tree hides is **scale**. At a million items, iALS on a single machine is fine and a two-tower may be overkill. At a hundred million items, the embedding table alone is tens of gigabytes and you are forced into the [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores) territory — sharded embedding tables, parameter servers, and ANN indexes that do not fit in RAM. At a billion items, even building the ANN index is a distributed job. So "which model?" is really "which model *at my scale*?", and the answer drifts toward two-tower-plus-ANN as scale grows, because it is the only shape that stays servable when the catalog gets enormous.

## 3. The model family map: stage, condition, metric

Decision trees are great for "pick one path," but in a review you also need the at-a-glance comparison: for each family, what stage does it serve, under what condition does it earn its place, and which metric judges it. The matrix below is the cheat sheet. Note the column that matters most: **the key metric changes with the stage.** Retrieval is judged by Recall@K because its only job is to not lose the good items; ranking is judged by NDCG and AUC because its job is to order them.

![Matrix mapping model families including matrix factorization, two-tower, graph neural networks, DeepFM, DCN, SASRec, MMoE and LLM4Rec against their stage, the condition under which to use them, and their headline metric](/imgs/blogs/the-recommender-systems-playbook-3.png)

| Family | Stage | Use when | Key metric | Where it shines |
|---|---|---|---|---|
| Popularity / item-CF | Retrieval | Always, as the floor | Recall@100 | Baseline you must beat |
| MF / iALS / BPR | Retrieval | Warm IDs, moderate scale | Recall@100 | Fast, strong, interpretable |
| Two-tower | Retrieval | Cold-start, rich features, web scale | Recall@100 | The modern default |
| GNN (PinSage) | Retrieval | A real, informative graph exists | Recall@100 | Co-view / social signal |
| GBDT / LR | Ranking | Tabular features, limited data | AUC, logloss | Baseline ranker |
| DeepFM / DCN | Ranking | Many sparse cross-features | AUC, logloss | Learned feature interactions |
| SASRec / BERT4Rec | Both | Order carries signal | NDCG@10, HitRate@10 | Sessions, next-item |
| MMoE / PLE | Ranking | Multiple objectives, seesaw risk | Per-task AUC | Multi-objective trade-offs |
| LLM4Rec | Both | Text-rich, cold-start, reasoning | NDCG, cost | Zero-/few-shot, generative |

The thing this table is really teaching is that **stage dictates metric**, and metric dictates loss, and loss dictates negatives. That chain is the next two sections. But first, internalize one rule from this map: a model's "key metric" is the one you A/B against, and a retrieval model that improves NDCG but drops Recall@100 has gotten worse, because it has thrown away good candidates the ranker never gets a chance to see. The [recall ceiling](#section-9) — the cap that retrieval places on everything downstream — is the most under-appreciated number in the stack, and we will compute it shortly.

A second rule the table teaches by its structure: **the same model can appear in more than one stage, but it plays a different role and is judged by a different metric in each.** SASRec is the clearest example. As a *retrieval* model, it encodes the user's recent sequence into a vector and you do ANN over item embeddings — judged by Recall@K. As a *ranking* model, it scores a small candidate set with full cross-attention over the sequence and rich features — judged by NDCG@K. Same architecture, two jobs, two metrics. The mistake is to evaluate the retrieval use of a model with a ranking metric, or vice versa, and conclude the model is "good" or "bad" without specifying the job. Always say the stage and the metric together: "SASRec retrieval at Recall@500" or "SASRec ranking at NDCG@10," never just "SASRec is at 0.3."

#### Worked example: the seesaw a single metric hides

A team running a single click-prediction ranker decides to also optimize for "save" events by adding save labels to the same loss. Click AUC holds at $0.78$, but save rate online *drops* 4%. They are confused — they added the objective, so why did it get worse? This is the **seesaw effect**: a shared-bottom model forced to predict two correlated-but-different objectives finds a compromise representation that serves neither well, and the weaker-signal task (saves, which are rarer) loses. A single aggregate metric hides it because click AUC, the dominant task, looks fine. The fix is a [multi-task architecture (MMoE or PLE)](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) that gives each task its own expert mixture, so the tasks stop fighting over one bottleneck. After the switch, click AUC stays at $0.78$ and save rate recovers and lifts 3%. The lesson: when you have multiple objectives, track them *separately*, and reach for multi-task structure the moment you see one task's metric move the wrong way as you add another.

## 4. Which loss? Match the objective, not the fashion

The loss is not a free choice. It is determined by what you are optimizing and which stage you are in. There are four families, and confusing them is one of the most common ways a recommender underperforms. The [loss function landscape post](/blog/machine-learning/recommendation-systems/the-loss-function-landscape-for-recsys) covers all four in depth; here is the decision logic.

**Pointwise BCE for calibration and CTR.** When you need a probability you can *trust* — for ad bidding, for expected-value ranking, for any downstream that multiplies the score by a number — train pointwise binary cross-entropy on logged impressions with their true click labels. The model learns $P(\text{click} \mid u, i)$, and with proper [calibration](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) that probability means what it says. Pointwise is the right loss for the *ranking* stage precisely because the ranker scores logged impressions where you have real negatives (shown-but-not-clicked).

**Pairwise BPR for top-K from implicit feedback.** When you only have positives and want the top of the list to be right, optimize the *order*, not the absolute score. [Bayesian Personalized Ranking](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive) maximizes the probability that a positive item ranks above a sampled negative. For a user $u$, a positive item $i$, and a negative item $j$, BPR minimizes

$$
\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)} \ln \sigma\big(\hat{x}_{ui} - \hat{x}_{uj}\big) + \lambda \lVert \Theta \rVert^2,
$$

where $\hat{x}_{ui}$ is the predicted score and $\sigma$ is the sigmoid. The gradient depends on the *difference* $\hat{x}_{ui} - \hat{x}_{uj}$, so the model only ever learns "$i$ above $j$," which is exactly what top-K cares about. This is why pairwise beats pointwise for top-K from implicit data: the gradient sees the order, not the absolute value.

**Sampled softmax / InfoNCE for retrieval.** A two-tower model has to discriminate the one positive item from the *entire catalog*, which is a softmax over millions of classes — intractable to compute exactly. So you approximate it with a sampled softmax: treat the in-batch positives of other examples as negatives and normalize over the sample. The catch is that the negatives are not drawn uniformly; popular items appear in more batches, so they are over-represented as negatives. You correct for this with the **$\log Q$ correction**, subtracting the log-probability of sampling each negative from its logit:

$$
s'(u, i) = s(u, i) - \log Q(i),
$$

where $Q(i)$ is the sampling probability of item $i$. Without it, the model over-penalizes popular items and your retrieval quality collapses on exactly the items users want most. The [sampled softmax and contrastive losses post](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval) derives this carefully.

**Listwise / LambdaRank for direct NDCG.** When you can afford it and you want to optimize the ranking metric itself, listwise losses like LambdaRank weight each pairwise swap by how much it would change NDCG. The trick that makes LambdaRank work is elegant: NDCG is not differentiable (it depends on the discrete ranking), but you can define the *gradient* directly as the pairwise gradient scaled by $|\Delta\text{NDCG}|$, the change in NDCG that swapping the two items would cause. So a swap near the top of the list (which moves NDCG a lot) gets a big gradient, and a swap deep in the list (which barely moves NDCG) gets a small one. The model spends its capacity where the metric cares. This is the most direct path to a ranking metric and is covered in [learning to rank for recommenders](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders). It costs more per step and needs full slates, so it is a refinement, not a starting point.

Here is the table that maps loss to objective to negatives in one place — the cheat sheet for the loss decision:

| Loss | Objective it serves | Stage | Negatives needed |
|---|---|---|---|
| Pointwise BCE | Calibrated $P(\text{click})$ | Ranking | Logged shown-not-clicked |
| Pairwise BPR | Top-K order from implicit | Retrieval or ranking | Sampled (one per positive) |
| Sampled softmax / InfoNCE | Discriminate vs whole catalog | Retrieval | In-batch + $\log Q$ |
| Listwise / LambdaRank | Direct NDCG | Ranking | Full slate |
| WARP (lightfm) | Top-K with rank-aware sampling | Retrieval | Rejection-sampled hard | 

![Decision tree for choosing a loss and its negatives that forks first on the objective being a calibrated probability or top-K quality, then routes to pointwise BCE, sampled-softmax retrieval, or pairwise ranking loss, and finally to a mixed negative strategy](/imgs/blogs/the-recommender-systems-playbook-4.png)

The tree above ties loss to negatives, because the two are inseparable. A pointwise BCE on logged data uses the real shown-but-not-clicked negatives. A sampled-softmax retrieval loss needs *sampled* negatives because there are no logged negatives at retrieval — the catalog is the negative set. That brings us to the question that quietly determines retrieval quality more than the architecture does.

## 5. Which negatives? The decision that beats the architecture

Here is a fact that surprises people: for retrieval, the negative sampling strategy often moves Recall@K more than swapping the model. The [negative sampling strategies post](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) is the deep dive; this is the playbook.

- **In-batch negatives with the $\log Q$ correction** are the default. Free to compute (you already have the batch), and the correction handles the popularity skew in the sampling distribution. Start here.
- **Popularity-weighted (frequency) negatives** sample harder negatives than uniform because popular items are plausible-but-wrong more often, teaching the model finer distinctions. This is the word2vec trick applied to recsys.
- **Hard negatives** — items that score high but are actually wrong — are where the real gains live. Mine them from the model's own top-K mistakes. But they are dangerous: a "hard negative" is often a **false negative**, an item the user would have liked but never saw. Over-mining hard negatives teaches the model to suppress good items, which is the same failure as popularity bias from a different direction.
- **Mixed negatives** — a blend of in-batch, popularity, and a small fraction of hard negatives — is what production systems converge on. The mix is a hyperparameter you tune against full-catalog Recall@K, never against a sampled metric.

Why do negatives matter so much? Because retrieval is fundamentally a problem of *discrimination against a vast background*. The positive item must beat every other item in the catalog, and the model only learns to push down the negatives it actually sees during training. If those negatives are all easy (random items wildly unrelated to the user), the model learns a coarse boundary that separates "obviously relevant" from "obviously irrelevant" and never learns the fine distinctions among plausible items — which is exactly the regime the top-K lives in. Hard negatives teach the fine boundary. But the same property that makes a hard negative informative — it scores high — is what makes it likely to be a false negative, an item the user would have engaged with but never saw. This is the central tension of negative sampling, and there is no setting that escapes it; you manage it with the mix and validate empirically. The [training two-tower post](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) shows the actual sampler code.

#### Worked example: hard negatives that backfire

A team mines hard negatives aggressively to lift retrieval. On the *sampled* evaluation (rank the positive against 100 random negatives) Recall@10 jumps from $0.61$ to $0.68$, a clear win. They ship it. Online, add-to-cart drops 2%. What happened? Many of the mined hard negatives were false negatives — relevant items the user simply had not been shown — so the model learned to push down items users actually want. Worse, the sampled metric hid it: against 100 *random* negatives the model still looks great, because random negatives are easy. The lesson is twofold. First, mix in only a small fraction of hard negatives and validate on **full-catalog** Recall@K, the failure mode the [sampled-metric illusion](#section-9) section explains. Second, treat unobserved items as uncertain, not as confident negatives. A 7-point sampled-metric "win" that costs 2% of cart is a textbook offline-online gap.

## 6. Which metric? Never trust offline alone

You now have a model, a loss, and negatives. How do you know it is better? This is where most teams quietly go wrong, because they trust a single offline number computed the wrong way. The discipline has a strict order, and each metric answers a different question. The [offline evaluation metrics post](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr) defines them; the [right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) tells you how to compute them without lying to yourself.

- **Recall@K for retrieval.** Retrieval's only job is to not lose the good items. Recall@K asks: of the items the user actually engaged with, what fraction made it into the top-K candidate set? If retrieval has Recall@500 of 0.7, then 30% of relevant items are gone before ranking even starts. No ranker can recover them. This is the **recall ceiling**, and it caps the whole funnel.
- **NDCG@K for ranking.** Once you have the candidates, ranking is judged by how well it orders them, with higher positions weighted more. NDCG (Normalized Discounted Cumulative Gain) is the standard. Its discounted gain at cutoff $K$ is $\text{DCG@}K = \sum_{r=1}^{K} \frac{2^{g_r} - 1}{\log_2(r+1)}$, where $g_r$ is the gain (relevance) of the item at rank $r$, normalized by the ideal ordering. The $\log_2(r+1)$ in the denominator is why getting rank 1 right matters far more than rank 10.
- **Calibration (ECE) when the probability is used as a number.** If you bid on $P(\text{click})$ or rank by expected value, a miscalibrated score is a business bug, not just a metric blip. Expected Calibration Error bins predictions and measures the gap between predicted and observed click rate per bin. The [calibration post](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) covers isotonic and Platt scaling.
- **Beyond accuracy: diversity, novelty, coverage.** A list that is "accurate" but shows ten near-duplicates is a bad list. The [beyond-accuracy post](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) covers intra-list diversity, catalog coverage, and serendipity, all of which the re-ranking stage manages.
- **A/B test and interleaving for truth.** This is the only metric that is ground truth, because it is the only one measured under the new policy on live users. [Interleaving](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) gives a faster, lower-variance read for ranking comparisons; a full A/B test gives the business metric. Everything offline is a filter to decide *what to A/B test*, not a substitute for it.
- **Off-policy / counterfactual estimation before you ship.** Between offline metrics and a live test, [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) (IPS, SNIPS, doubly-robust) estimates online performance from logged data with importance weighting. It is how you avoid wasting A/B slots on candidates that offline metrics love but that will flop.

The order is not negotiable: offline metrics → off-policy estimate → A/B test. Skipping straight from offline NDCG to ship is how you get the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) story that opens half the war stories in this series.

A subtlety that trips up even experienced teams: **the metric must match the funnel stage you are changing.** If you improve retrieval, the right offline read is full-catalog Recall@K *of the retrieval stage in isolation*, holding the ranker fixed — not the end-to-end NDCG, because the ranker can mask or amplify a retrieval change in confusing ways. If you improve the ranker, hold retrieval fixed and read NDCG@K *on the candidate set*. Mixing the two is how you get a result that says "the system got better" without knowing which component to credit, which makes the next decision impossible. Decompose the funnel and measure each stage against its own metric, then measure the whole thing end to end as the final check. The [right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) post details how to construct the per-stage harness without leakage.

Here is the metric cheat sheet, by question answered:

| Metric | Question it answers | Stage | Trust level |
|---|---|---|---|
| Recall@K | Did retrieval keep the good items? | Retrieval | Offline (full catalog) |
| NDCG@K, MAP, MRR | Did ranking order them well? | Ranking | Offline (per-stage) |
| AUC, logloss | Is CTR prediction discriminative? | Ranking | Offline |
| ECE / reliability | Is the probability trustworthy? | Ranking | Offline |
| Coverage, Gini, novelty | Is the catalog being used? | Re-rank / loop | Offline + monitoring |
| IPS / SNIPS / DR | What will online do, roughly? | Whole funnel | Off-policy estimate |
| A/B lift, interleaving | What did online actually do? | Whole funnel | Ground truth |

Read top to bottom and you have the escalation ladder. You climb it: cheap offline filters first, off-policy estimate to narrow the field, and the expensive, slow, but only-true A/B test last.

## 7. When to reach for the advanced tools

A solid two-tower retrieval plus a deep ranker plus an A/B harness will get most products most of the way. The advanced tools — bandits, reinforcement learning, causal models, LLMs, generative retrieval — each solve a specific problem the standard stack cannot, at a specific cost. The matrix below is the "is it worth it?" table.

![Matrix mapping advanced recommender tools including bandits, reinforcement learning, causal uplift models, LLM4Rec and generative retrieval against the conditions under which each pays off and the main cost each one carries](/imgs/blogs/the-recommender-systems-playbook-8.png)

**Bandits and exploration.** When fresh items arrive constantly and you need to learn their value fast without burning the catalog, a [bandit](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) balances exploration and exploitation. The cost is tuning the exploration budget and managing regret. Reach for it when cold-start churn is high (news, marketplaces with daily new listings).

**Reinforcement learning.** When you care about long-horizon value — retention over weeks, not the next click — [RL for recommendation](/blog/machine-learning/recommendation-systems/reinforcement-learning-for-recommendation) optimizes the cumulative reward. The cost is severe: off-policy training risk, hard debugging, and the need for a reliable simulator or strong off-policy correction. Reach for it only when you can prove the greedy next-click objective is actively hurting long-term value.

**Causal and uplift.** When the question is "who should I *intervene* on" rather than "who will engage anyway," [causal and uplift recommendation](/blog/machine-learning/recommendation-systems/causal-and-uplift-recommendation) estimates the treatment effect of recommending. The cost is that it needs randomized exposure data to identify the effect. Reach for it for promotions, notifications, and anything where the recommendation itself changes behavior.

**LLM4Rec.** When items are text-rich and you need zero-shot reasoning over cold-start items or natural-language intent, [LLMs for recommendation](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec) and [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice) bring world knowledge the ID-based models lack. The cost is latency and serving expense, often 10x to 100x a deep ranker, which is why [distillation and compression](/blog/machine-learning/recommendation-systems/distillation-and-compression-for-recsys) is usually the bridge from an LLM teacher to a servable student. For the underlying mechanics, the [LLM fine-tuning techniques post](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) goes deeper on LoRA/PEFT than this series does, and the in-series [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice) post is the recsys-specific version.

**Generative retrieval.** Semantic-ID models that generate item IDs directly collapse the funnel into one model. Promising, but early and unstable, with painful index rebuilds. Watch it; do not bet a launch on it.

The meta-rule: **the advanced tool must beat the boring baseline on the metric you A/B, by enough to justify its cost.** A two-tower plus ranker that hits target is not a problem to solve with an LLM.

There is one advanced technique that is *not* a model but a bridge, and it deserves its own mention because it changes the cost calculus of everything above: [distillation and compression](/blog/machine-learning/recommendation-systems/distillation-and-compression-for-recsys). Distillation lets you train a heavy, slow, expensive teacher — an LLM, a large cross-attention ranker, an ensemble — and then transfer its knowledge into a small, fast student that fits the latency budget. This is how teams get the *accuracy* of a frontier model with the *serving cost* of a boring one. The pattern recurs: a large model finds the signal, distillation makes it servable. The same logic underlies [pretraining and finetuning recommenders](/blog/machine-learning/recommendation-systems/pretraining-and-finetuning-recommenders) — pretrain a representation on abundant data (all interactions, or text), then finetune the cheap task-specific head. When you are tempted by an LLM4Rec that cannot serve, the right question is rarely "can I serve the LLM?" (you usually cannot) but "can I distill the LLM into something I can serve?" (often yes). That reframing turns a budget-busting idea into a shippable one.

A final structural point on the advanced tools: most of them *require data the standard stack does not generate*. Causal and uplift models need randomized exposure. Off-policy RL needs logged action probabilities (the propensity of each shown item under the logging policy). Bandits need to log their own exploration decisions. If you have not been logging propensities and exploration flags from day one, you cannot adopt these tools later without first running a campaign to collect the data. This is why the build order puts exploration *before* the advanced tools and why the loop-hygiene stage logs propensities even before you have a model that uses them. **The data you need for tomorrow's model is generated by today's logging — so log richly even when you do not yet need it.**

## 8. The build order: what to ship first and why

You do not build the whole stack at once. You build it in stages, and you earn the right to each stage by beating the previous one on a temporal split. The order below is not arbitrary — each stage de-risks the next and gives you a baseline to measure against. The [building the training pipeline post](/blog/machine-learning/recommendation-systems/building-the-training-pipeline-for-recsys) covers the plumbing; this is the sequence.

![Layered diagram of the staged recommender build order from a popularity baseline through retrieval, ranking, multi-task heads, exploration, and finally loop hygiene for bias, skew and drift](/imgs/blogs/the-recommender-systems-playbook-5.png)

1. **Popularity + item-item CF baseline.** Ship this first, end to end, with a real serving path and a real eval harness. It establishes the floor (can the model beat showing everyone the top items?) and forces you to build logging, the [temporal split](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate), and the metric pipeline before you have a complicated model to blame. Most "the model won't improve" bugs are actually eval-harness bugs that this stage flushes out.
2. **MF / two-tower retrieval.** Add personalized candidate generation. Start with iALS for warm IDs; move to a two-tower when you need cold-start. Validate on full-catalog Recall@K against the popularity baseline. Stand up [ANN serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) (FAISS/HNSW/ScaNN) here and measure the recall-latency Pareto.
3. **A ranker.** Add a GBDT or deep CTR model that re-scores the retrieved candidates with full features. Now you have the two-stage funnel. Measure NDCG@K on the candidate set and, critically, check that retrieval recall is not the bottleneck (the recall ceiling).
4. **Multi-task heads.** When the product has multiple objectives, add [MMoE/PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) so click, like, and conversion share representation without seesawing. Do not start here; you need a working single-objective ranker first.
5. **Exploration.** Add a [bandit](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) and [diversity in re-ranking](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) to break the feedback loop and surface fresh items. This is also where you start collecting the randomized data that makes off-policy evaluation and causal models possible later.
6. **Loop hygiene.** Continuously: monitor [popularity bias](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) and [feedback loops](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles), correct [position bias](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data), guard against [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there), handle [delayed feedback](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution), and watch for drift. This is not a stage you finish; it is the ongoing tax of running a recommender.

The deepest mistake teams make is inverting this order: building a fancy sequential multi-task model before they have a working baseline and a trustworthy eval harness. You then cannot tell whether the model is good, because you have nothing reliable to compare it to and the harness itself is probably buggy. Build the boring thing first.

#### Worked example: the build order saves a quarter

A team is given six months to launch a recommender and decides to go straight to a state-of-the-art sequential multi-task model, because that is what the best papers use. Three months in, offline NDCG looks great, but online the model barely beats the simple popularity widget the old team had. They cannot tell why: there is no clean baseline to A/B against, the eval harness was written alongside the complex model and has a temporal-leakage bug nobody caught, and retrieval recall was never measured in isolation, so they do not know the ranker is starved of good candidates. They have burned a quarter and have an unexplainable result. Contrast the staged approach: by week three they have a popularity baseline live with a leak-free temporal harness; by week six iALS retrieval beats popularity by a measured margin; by week ten a GBDT ranker is live and they *know* its NDCG gain came from reordering, because recall was held fixed and measured. Every step is explainable, every gain is attributable, and the harness was stress-tested by the boring baseline before the complex model ever ran. Same six months, but the second team ships something defensible and the first ships a mystery. The build order is not bureaucracy; it is how you keep the result interpretable.

A note on what "loop hygiene" (stage 6) actually involves, because it is the stage most often skipped and most often the source of the 2 a.m. page. It is at least five distinct, ongoing jobs. First, **train-serve parity**: assert that every feature is computed by the same code offline and online, ideally by sharing the transformation library. Second, **bias monitoring**: track catalog coverage and the Gini coefficient of exposure over time; a rising Gini means the loop is collapsing the catalog. Third, **position-bias correction**: train on click data weighted by an inverse-propensity estimate of the position's examination probability, so the model does not learn "high slot = good." Fourth, **delayed-feedback handling**: do not treat a recent un-converted impression as a hard negative, because the conversion may simply not have arrived yet; use a delayed-feedback model or a conversion window. Fifth, **drift detection**: alarm when feature distributions or model scores shift, because a silent upstream data change can degrade the model for weeks before anyone notices in the business metric. None of these is a model; all of them are the tax of running a live recommender, and budgeting for them is the difference between a launch and a sustainable system.

## 9. The top ten pitfalls, organized {#section-9}

A recommender breaks in one of four places: the **data** you logged, the **eval** you computed, the **serving** path you shipped, or the **loop** the system runs in. When a metric drops and you do not know why, name the layer first — it turns a vague panic into a specific, fixable bug. The tree below organizes the classic failures by layer; the [debugging a recommender post](/blog/machine-learning/recommendation-systems/debugging-a-recommender-that-wont-improve) is the systematic triage guide.

![Decision tree organizing recommender failure modes into data, evaluation, serving, and loop layers, with cold start and leakage under data, the sampled-metric illusion under evaluation, train-serve skew under serving, and popularity and position bias under the loop](/imgs/blogs/the-recommender-systems-playbook-7.png)

Here are the ten pitfalls every practitioner should be able to name on sight, each linking to its dedicated post:

1. **The sampled-metric illusion.** Evaluating by ranking the positive against a small random sample of negatives produces a metric that does not even rank models consistently with the full-catalog metric. The KDD 2020 result "On Sampled Metrics for Item Recommendation" (Krichene and Rendle) showed that sampled Recall and NDCG can reverse the ordering of models. Always evaluate retrieval on the **full catalog**. (See [offline evaluation metrics](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr).)
2. **Train-serve skew.** A feature computed one way in the training pipeline and another way at serving silently halves precision. The model is fine; the inputs disagree. (See [train-serve skew and the bugs that hide there](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there).)
3. **The offline-online gap.** Offline NDCG rises, online engagement is flat or down, because offline data is distribution-shifted, missing-not-at-random, and position-biased. (See [the offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied).)
4. **Popularity bias.** The system over-recommends popular items, which accumulate more impressions and clicks, which makes them more popular — a self-reinforcing fixed point that shrinks the effective catalog. (See [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer).)
5. **Feedback loops and filter bubbles.** The model trains on its own outputs, narrowing what users see until the catalog collapses to a handful of items per user. (See [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles).)
6. **Position bias.** Items in higher slots get more clicks regardless of relevance, so naively training on raw clicks teaches the model that "high position = good." (See [position and selection bias in click data](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data).)
7. **Cold start.** New users and new items have no interaction history, so ID-only models produce garbage for them. Side features and content models are the fallback. (See [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem).)
8. **Delayed feedback.** Conversions arrive hours or days after the click, so a model trained on "labeled so far" mislabels recent positives as negatives. (See [delayed feedback and conversion attribution](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution).)
9. **Leakage.** A feature that encodes the future (or a random rather than temporal split) inflates offline metrics and evaporates online. (See [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate).)
10. **The recall ceiling.** Retrieval that drops good items caps everything downstream; no ranker can rank an item it never receives. Measure retrieval recall before you blame the ranker. (See [the two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) and the funnel arithmetic above.)

A useful way to hold these ten in your head is to notice they pair up by *which assumption they violate*. The sampled-metric illusion, leakage, and the recall ceiling are all violations of "my offline number means what I think it means." Train-serve skew and the offline-online gap are violations of "the model I tested is the model I shipped, evaluated as it will run." Popularity bias, feedback loops, and position bias are violations of "my data is a fair sample of preferences" — they are the three faces of missing-not-at-random. Cold start and delayed feedback are violations of "I have a label for this user-item pair right now." Group them this way and a new failure you have never seen before still slots into one of four buckets, which tells you where to look. The [debugging post](/blog/machine-learning/recommendation-systems/debugging-a-recommender-that-wont-improve) turns this into a step-by-step bisection: change one thing at a time, hold the rest fixed, and isolate which assumption broke.

#### Worked example: computing the recall ceiling

Say your two-tower retrieval has full-catalog Recall@500 of $0.72$ on a temporal split. That means for the average relevant item, there is a 72% chance it appears in the 500 candidates the ranker sees. Now suppose your ranker is *perfect* — it always puts every relevant candidate at the top. The best possible end-to-end Recall@20 is still bounded by retrieval: you cannot rank what you did not retrieve. If the relevant items that survive retrieval are a fraction $0.72$ of all relevant items, then even a perfect ranker yields at most $\text{Recall@}20 \le 0.72$, and in practice far less because the ranker is not perfect and 20 < 500. The actionable consequence: if your end-to-end metric is stuck and retrieval recall is $0.72$, spending a month on a fancier ranker is wasted; the leverage is in lifting retrieval recall. This single calculation reorders more roadmaps than any model upgrade.

## 10. The science block: the three laws that govern the stack

The series is "scientific" by mandate, so let us make the three load-bearing quantitative facts explicit and rigorous in one place. These are the laws you reason with, not memorize.

**Law 1: the funnel is forced by the latency budget.** With catalog size $N$, ranker cost $t_r$ per item, retrieval cost $t_q$ per query, and budget $B$, you cannot run the ranker on the whole catalog because $N \cdot t_r \gg B$. Staging with a retrieval set of size $M$ gives total cost $t_q + M \cdot t_r$, and you choose $M$ as large as the budget allows: $M \le (B - t_q)/t_r$. With $B = 100$ ms, $t_q = 5$ ms, $t_r = 0.1$ ms (a lean batched ranker), you get $M \le 950$, which is why production retrieval sets are typically a few hundred to ~1000. The funnel is not a design preference; it is the solution to an inequality.

**Law 2: ANN trades recall for latency on a smooth Pareto.** Exact maximum-inner-product search over $N$ item vectors is $O(N)$ per query — too slow at scale. ANN indexes (IVF, HNSW, IVF-PQ) bound the search and recover a fraction of the true neighbors. The relation is monotone: more probes / higher `efSearch` / less aggressive quantization buys recall at the cost of latency and memory. You pick a point on the curve, not a single setting. A representative point: an HNSW index on 10M 64-dim vectors might give 0.95 recall at p99 ~2 ms, while pushing to 0.99 recall costs 3-4x the latency. The [ANN serving post](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) maps the index-recall-latency-memory trade-off in full.

**Law 3: the offline-online gap is a distribution-shift identity.** Let $\pi_{\text{old}}$ be the logging policy and $\pi_{\text{new}}$ the policy you want to evaluate. Your offline metric averages reward over data drawn from $\pi_{\text{old}}$:

$$
\hat{V}_{\text{offline}}(\pi_{\text{new}}) = \frac{1}{n}\sum_{i} r_i \cdot \mathbb{1}[\pi_{\text{new}}\text{ would show item } i],
$$

but the quantity you are paid for is $V_{\text{online}}(\pi_{\text{new}}) = \mathbb{E}_{a \sim \pi_{\text{new}}}[r(a)]$. These differ whenever $\pi_{\text{new}} \ne \pi_{\text{old}}$, because the actions $\pi_{\text{new}}$ would take are under-represented (or absent) in the logs. The IPS correction reweights each logged reward by $\pi_{\text{new}}(a)/\pi_{\text{old}}(a)$ to get an unbiased estimate — *provided* $\pi_{\text{old}}$ had nonzero probability on every action $\pi_{\text{new}}$ takes (the support condition). When the old policy never explored an action, no estimator can recover its value, which is the deep reason exploration is not optional. This is the formal statement of why offline metrics lie and why off-policy evaluation, covered in [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), exists.

These three laws are why the stack looks the way it does. Law 1 gives you the funnel. Law 2 gives you ANN-served retrieval. Law 3 gives you the eval discipline and the mandate to explore.

There is a fourth quantitative fact worth stating because it governs the loop rather than the funnel: **popularity bias is a fixed point.** Model the system as a map from the current exposure distribution to the next one. An item shown more accumulates more clicks (in proportion to its exposure, by position bias and sheer impression count), which raises its training weight, which raises its score, which raises its exposure next round. Write $e_t(i)$ for item $i$'s exposure share at round $t$; the loop's update is roughly $e_{t+1}(i) \propto e_t(i) \cdot f(\text{clicks}_t(i))$ with $f$ increasing. A distribution where a few items have almost all the exposure is a stable fixed point of this map — small perturbations decay back to it — while the uniform-coverage distribution is unstable. That is the precise sense in which "the rich get richer" is not a metaphor but a dynamical property of the closed loop, derived in [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer). The only way out of a stable bad fixed point is an external force: exploration that injects exposure the loop would not give, and debiasing that down-weights popularity in training. Both are deliberate counter-pressures against a dynamics that otherwise collapses the catalog.

#### Worked example: reading Law 2 off a real Pareto

Suppose you index 50 million item vectors at 128 dimensions with FAISS HNSW. At `efSearch = 32` you measure full-catalog Recall@100 of $0.91$ at p99 latency of $1.4$ ms. You need higher recall, so you raise `efSearch` to $128$: Recall@100 climbs to $0.97$ but p99 rises to $4.8$ ms. Is the trade worth it? Your retrieval budget inside a 100 ms SLA is about 5 ms, so $4.8$ ms just fits — but it leaves almost no headroom for tail spikes. The decision is not "more recall is better"; it is "is the 6-point recall gain worth consuming your entire retrieval budget and risking SLA misses under load?" Often the answer is to take the middle point (`efSearch = 64`, say $0.95$ recall at $2.6$ ms) and spend the remaining budget on a slightly larger candidate set $M$, which lifts the recall *ceiling* more cheaply than chasing the last points of index recall. This is Law 1 and Law 2 traded against each other, and it is a real decision you make at every launch. The arithmetic, not intuition, tells you where to sit.

## 11. The practical flow: an end-to-end skeleton

Talk is cheap; here is the code that ties the series together into one runnable spine. We build, in order, a popularity baseline, a two-tower retrieval model with FAISS serving, a ranker, and an eval harness. This is the skeleton you fork for a new product. It uses the toolchain the series teaches: `implicit`, PyTorch, `faiss`, and pandas/numpy for metrics. (Treat it as a faithful skeleton, not a turnkey production system — each piece has its own dedicated post for the full version.)

**Stage 1 — the popularity and item-CF baseline.** Always ship this first.

```python
import numpy as np
import pandas as pd

# interactions: a DataFrame of (user_id, item_id, ts)
def temporal_split(df, frac=0.9):
    df = df.sort_values("ts")
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]

train, test = temporal_split(interactions)

# popularity baseline: most-interacted items in the train window
popular = train["item_id"].value_counts().index.to_numpy()

def recommend_popular(user_id, k=20):
    return popular[:k]  # same list for everyone; the floor to beat
```

**Stage 2 — implicit MF retrieval with `implicit`.** A real personalized baseline in minutes.

```python
import implicit
from scipy.sparse import coo_matrix

# build the user-item matrix (implicit confidence = interaction count)
u = train["user_id"].astype("category")
i = train["item_id"].astype("category")
mat = coo_matrix((np.ones(len(train)), (u.cat.codes, i.cat.codes))).tocsr()

model = implicit.als.AlternatingLeastSquares(
    factors=64, regularization=0.05, iterations=20
)
model.fit(mat)  # learns 64-dim user and item vectors

# retrieve top-K for a user via the learned factors
def recommend_als(user_code, k=500):
    ids, scores = model.recommend(user_code, mat[user_code], N=k)
    return ids
```

**Stage 3 — a two-tower model in PyTorch with a sampled-softmax loss.** For cold-start and side features.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        # in production the item tower also ingests content features
    def user_vec(self, u):  return F.normalize(self.user_emb(u), dim=-1)
    def item_vec(self, i):  return F.normalize(self.item_emb(i), dim=-1)

def in_batch_softmax_loss(u_vec, i_vec, log_q, temp=0.07):
    # logits: every user against every item in the batch
    logits = (u_vec @ i_vec.t()) / temp
    logits = logits - log_q.unsqueeze(0)        # the logQ correction
    labels = torch.arange(u_vec.size(0), device=u_vec.device)
    return F.cross_entropy(logits, labels)      # positive is on the diagonal
```

**Stage 4 — FAISS index build and serving.** Turn the item tower into a millisecond retriever.

```python
import faiss

item_vectors = model_item_vectors.astype("float32")  # (n_items, dim), L2-normalized
index = faiss.IndexHNSWFlat(item_vectors.shape[1], 32)  # M=32 neighbors
index.hnsw.efConstruction = 200
index.add(item_vectors)
index.hnsw.efSearch = 64                      # the recall-latency knob

def retrieve(user_vector, k=500):
    scores, ids = index.search(user_vector.reshape(1, -1), k)
    return ids[0]                              # top-K candidate item ids
```

**Stage 5 — a ranker over the retrieved candidates.** A simple deep CTR head; swap in DeepFM/DCN as needed.

```python
class Ranker(nn.Module):
    def __init__(self, n_feat, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):  # x: (n_candidates, n_feat) full features
        return self.net(x).squeeze(-1)        # logit per candidate

# train with pointwise BCE on logged impressions (real shown/clicked labels)
def bce_step(ranker, x, y, opt):
    logits = ranker(x)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

**Stage 6 — the eval harness.** Full-catalog Recall@K and NDCG@K, the metrics you actually trust.

```python
def recall_at_k(recommended, relevant, k):
    rec_k = set(recommended[:k])
    return len(rec_k & set(relevant)) / max(1, len(set(relevant)))

def ndcg_at_k(recommended, relevant, k):
    rel = set(relevant)
    dcg = sum(1.0 / np.log2(r + 2) for r, it in enumerate(recommended[:k]) if it in rel)
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(rel))))
    return dcg / ideal if ideal > 0 else 0.0

# evaluate end-to-end on the temporal test set, FULL catalog, no sampling
def evaluate(recommender, test_by_user, k=20):
    rks, nks = [], []
    for user, relevant in test_by_user.items():
        recs = recommender(user, k=k)
        rks.append(recall_at_k(recs, relevant, k))
        nks.append(ndcg_at_k(recs, relevant, k))
    return {"recall@%d" % k: np.mean(rks), "ndcg@%d" % k: np.mean(nks)}
```

That is the entire spine: baseline, retrieval, ANN serving, ranker, eval. Each stage has a dedicated post with the production-grade version — [building the training pipeline](/blog/machine-learning/recommendation-systems/building-the-training-pipeline-for-recsys), [embeddings the heart of recommenders](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders), [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores) — but this is enough to stand up a real funnel and measure it honestly.

## 12. The production checklist

Before any recommender goes live (or back to live after a change), run it past this checklist. Each item exists because skipping it has shipped a broken recommender before. The matrix below pairs each check with the pitfall it guards against.

![Matrix pairing each production checklist item including temporal eval, train-serve parity, calibration, off-policy evaluation, bias monitoring and A/B testing with the specific pitfall it guards against](/imgs/blogs/the-recommender-systems-playbook-6.png)

- [ ] **Temporal eval, not random split.** Split by time so the model is tested on the future, never the past. Guards leakage. ([the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate))
- [ ] **Full-catalog metrics, not sampled.** Compute Recall@K and NDCG@K against the whole catalog. Guards the sampled-metric illusion.
- [ ] **Beat the baseline.** Does the model beat popularity and item-CF on the temporal split? If not, stop and debug the data.
- [ ] **Negatives chosen on purpose.** In-batch + $\log Q$ at minimum; hard negatives only with full-catalog validation. Guards over-suppression of good items.
- [ ] **Calibration checked.** If the score is used as a probability, measure ECE and fit isotonic/Platt. Guards miscalibration. ([calibration](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust))
- [ ] **Train-serve parity.** The exact same feature code path offline and online; assert on it. Guards train-serve skew. ([train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there))
- [ ] **Feature store with point-in-time joins.** No future leakage in features. Guards leakage. ([feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores))
- [ ] **ANN within the latency budget.** Pick the recall-latency point and load-test p99. Guards the recall ceiling and SLA misses.
- [ ] **Off-policy estimate before A/B.** IPS/SNIPS to filter candidates worth a live slot. Guards the offline-online gap. ([off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation))
- [ ] **A/B test or interleaving.** The only ground truth. Never ship on offline metrics alone. ([A/B testing](/blog/machine-learning/recommendation-systems/ab-testing-recommenders))
- [ ] **Monitor bias, loops, drift.** Track catalog coverage, Gini, and per-segment metrics over time. Guards popularity bias and feedback loops.
- [ ] **Cold-start fallback wired.** A content-based or popularity path for new users/items. Guards cold start. ([the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem))

If you can check every box, you have a recommender that is not just accurate offline but defensible in production. The checklist is the difference between "the model looks good" and "the model is good," and those are not the same sentence.

## 13. Case studies: the playbook in the wild

The decisions above are not theoretical; they are what shipped recommenders actually do. Four real systems, each illustrating a different part of the playbook.

**YouTube: the two-stage funnel, by the book.** The canonical "Deep Neural Networks for YouTube Recommendations" (Covington, Adams, Sargin, RecSys 2016) is the funnel made concrete: a deep candidate-generation network retrieves a few hundred videos from a corpus of millions, then a separate deep ranking network scores them with far richer features. It is the reference architecture for retrieval → ranking, and a key lesson from it is that the two stages optimize different objectives — candidate generation for recall, ranking for a watch-time-weighted objective. The [YouTube case study](/blog/machine-learning/recommendation-systems/case-study-youtube-deep-retrieval-and-ranking) walks through both networks.

**Netflix and Spotify: recommendation as the product.** Netflix's long-running work (from the Netflix Prize through the personalized homepage and artwork) and Spotify's Discover Weekly show that the recommender *is* the surface, and that the objective is retention, not next-click. This is where multi-objective ranking, diversity, and long-horizon value stop being optional. The [Netflix and Spotify case study](/blog/machine-learning/recommendation-systems/case-study-netflix-and-spotify-recommendation-as-product) covers how product framing changes the metric.

**Pinterest, Instagram, TikTok: feed ranking and the graph.** Pinterest's PinSage brought graph neural networks to web-scale retrieval; TikTok's For You page is the modern exemplar of a tight feedback loop with heavy exploration and sequential modeling. These systems live and die by the loop hygiene of Section 8. The [feed-ranking case study](/blog/machine-learning/recommendation-systems/case-study-pinterest-instagram-tiktok-feed-ranking) is the deep dive.

**Amazon and Alibaba: e-commerce at scale.** Amazon's classic item-to-item collaborative filtering (Linden, Smith, York, 2003) is the proof that a simple, well-engineered baseline can power a multi-billion-dollar surface for years. Alibaba's DIN/DIEN brought attention over user behavior sequences to CTR prediction. The [e-commerce case study](/blog/machine-learning/recommendation-systems/case-study-amazon-alibaba-e-commerce-rec) shows the spectrum from "boring and effective" to "sequential and deep."

What unites all four is the lesson the playbook is built around: **the architecture is rarely the differentiator; the loop hygiene and the eval discipline are.** YouTube's funnel is simple in outline; what made it work was getting the candidate-generation objective right and serving it within budget. Amazon's item-to-item CF is two-decades-old math; what made it durable was the engineering and the relentless A/B testing around it. None of these systems won by having a cleverer loss function than their competitors. They won by building the boring infrastructure — logging, temporal eval, train-serve parity, exploration, A/B harnesses — that let them iterate safely on whatever model came next. That is precisely why the build order in Section 8 puts the baseline and the harness before the fancy model: the companies that ship the best recommenders are the ones that can *measure* a recommender the most honestly, and measurement is infrastructure, not modeling.

One more honest caveat on case-study numbers. Headline figures like "35% of Amazon purchases come from recommendations" or "75% of Netflix watch time is recommended" are widely cited but are dated, self-reported, and methodologically fuzzy — treat them as order-of-magnitude evidence that recommendation is the dominant surface, not as benchmarks to hit. When you read a paper's online lift number, ask three questions before believing it: was it an A/B test or an offline estimate, what was the baseline it lifted *over*, and over what population and time window. A "+10% CTR" against a random baseline is meaningless; a "+1% GMV" against a strong production model in a clean A/B test is enormous. The playbook teaches you to read those numbers skeptically because your own numbers will be read the same way.

#### Worked example: a before-after table you can reason about

Here is a representative (illustrative, order-of-magnitude) progression for a mid-size e-commerce recommender on a temporal split, showing how each build stage moves the needle. The point is the *shape* of the gains, not the exact figures, which depend entirely on your data.

| Stage | Model | Recall@500 | NDCG@20 | p99 latency | Notes |
|---|---|---|---|---|---|
| 1 | Popularity | 0.21 | 0.06 | 1 ms | the floor |
| 2 | iALS retrieval | 0.58 | 0.14 | 3 ms | personalization |
| 2b | Two-tower retrieval | 0.72 | 0.17 | 5 ms | cold-start coverage |
| 3 | + GBDT ranker | 0.72 | 0.29 | 28 ms | ranking lift, recall unchanged |
| 4 | + MMoE multi-task | 0.72 | 0.31 | 32 ms | conversion +, click flat |
| 5 | + exploration | 0.74 | 0.30 | 33 ms | coverage up, slight NDCG dip |

Read the table the way a practitioner does. The biggest single jump in Recall@500 is popularity → retrieval (0.21 → 0.72); that is the recall ceiling being lifted, and it is where the early effort belongs. The ranker (stage 3) doubles NDCG@20 (0.14 → 0.29) *without changing recall* — it reorders the same candidates. Multi-task (stage 4) trades a tiny NDCG gain for a conversion gain that a single-objective model could not get (the seesaw resolved). Exploration (stage 5) *lowers* NDCG slightly while *raising* coverage and recall — the deliberate cost of breaking the feedback loop. Every number here is the consequence of a decision from one of the earlier sections, which is exactly what a playbook should let you predict.

## 14. Worked walkthrough: design a recommender for a new product

Let us put the whole playbook to work on a fresh problem, applying each decision tree in order. The product: a **recipe app** with 500,000 recipes and 2 million users, launching a personalized "For You" feed. We have implicit feedback only (saves, cooks, views), a stream of new recipes added daily, and rich text/ingredient metadata per recipe. Budget: a 120 ms feed latency SLA.

**Step 1 — frame the problem (Section 1).** This is top-K retrieval-then-ranking, not rating prediction. We have implicit, positive-only feedback ([implicit vs explicit](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have)), missing-not-at-random, with daily new items (cold start matters a lot). The objective for launch is engagement (saves + cooks), with retention as the north star later.

**Step 2 — pick the models (Sections 2-3).** Retrieval: we have cold-start churn and rich content, so the model decision tree routes us to a **two-tower** model whose item tower ingests recipe text and ingredient features, with [content-based features](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) giving new recipes a vector on day one. We will *also* ship a popularity + item-CF baseline first (Step 6). Ranking: we have tabular + text features and modest scale, so start with a **GBDT**, plan a **DCN** upgrade once feature crosses prove valuable. Sequence (recent cooking history) is plausibly informative, so SASRec is a stage-3 candidate, not a launch requirement.

**Step 3 — pick the loss and negatives (Sections 4-5).** Retrieval trains with **sampled softmax + $\log Q$ correction**, negatives = **in-batch + a small fraction of hard negatives mined from top-K mistakes**, validated on full-catalog Recall@K. The ranker trains with **pointwise BCE** on logged impressions (real shown/clicked labels), since we want a calibrated save-probability we can later combine with cook-probability.

**Step 4 — pick the metrics (Section 6).** Retrieval: full-catalog **Recall@500** on a temporal split. Ranking: **NDCG@20**. We watch **calibration (ECE)** because we will eventually rank by a weighted sum of save and cook probabilities. Truth comes from an **A/B test**, with **interleaving** for faster ranker comparisons, and **off-policy IPS** to pre-filter candidates.

**Step 5 — decide on advanced tools (Section 7).** With daily new recipes, cold-start churn is high, so a **bandit** for exploring fresh recipes is on the roadmap (stage 5), and the randomized exposure it generates sets us up for **causal uplift** on push notifications later. An **LLM4Rec** path is tempting given rich recipe text, but its latency blows the 120 ms budget; we note it as a possible *teacher* to distill into the two-tower item encoder, not a serving model. Generative retrieval: not yet.

**Step 6 — build order (Section 8).** (1) Ship popularity + item-CF with a temporal-split eval harness and logging. (2) Add iALS, then the two-tower with content features; stand up FAISS HNSW serving and pick the recall-latency point that fits a ~5 ms retrieval budget inside the 120 ms SLA. (3) Add the GBDT ranker over ~500 candidates; verify retrieval recall is not the ceiling. (4) Add MMoE for save + cook once single-objective works. (5) Add the bandit + diversity re-ranking. (6) Stand up loop hygiene from day one: train-serve parity assertions, coverage/Gini monitoring, cold-start fallback.

**Step 7 — run the checklist (Section 12) and name the first three pitfalls.** Temporal eval, full-catalog metrics, beat-the-baseline, parity, calibration, off-policy, A/B — all wired. The three pitfalls that will bite this product first, in order: **cold start** (daily new recipes with no interactions — mitigated by the content-aware item tower and a content fallback), **popularity bias** (a few viral recipes will dominate — mitigated by exploration and coverage monitoring), and **train-serve skew** (ingredient features computed in batch vs at serving — mitigated by a shared feature path and parity asserts). Name them out loud in the design review, with the mitigation for each, and you have done the playbook's whole job.

That walkthrough used every section of this post in sequence. That is what the playbook is: not a model, but a *procedure* for getting from a blank feed to a defensible production recommender, with a reason for every choice and a known failure mode for every stage.

## 15. When to reach for this playbook, and when not to

A playbook is a default, and defaults have limits. Reach for the full funnel-and-loop machinery when you have a large catalog, real interaction volume, and recommendation is a meaningful surface. Do **not** over-build when:

- **Your catalog is small** (hundreds of items). A two-stage funnel is overkill; a single ranking pass over the whole catalog fits the latency budget, and you can skip ANN entirely.
- **You have almost no interaction data.** No amount of model sophistication beats a good content-based or popularity recommender when the matrix is empty. Build the [content/hybrid](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) path first and earn the collaborative model as data arrives.
- **A two-tower plus ranker already hits target.** Do not add a GNN, an LLM, or RL because they are interesting. Each is a serving and maintenance cost; pay it only when the boring stack provably stalls on the metric you A/B.
- **You cannot run an A/B test.** If you have no way to measure online truth, be deeply skeptical of any offline win, lean on conservative changes, and invest first in the measurement infrastructure — it is worth more than any model.

The most expensive mistake in this field is not picking the wrong model. It is building a sophisticated model on a broken eval harness, shipping an offline win that flops online, and not being able to tell why. The playbook's real value is that it forces the boring, load-bearing work — baselines, temporal splits, full-catalog metrics, train-serve parity, A/B discipline — *before* the fancy modeling, so that when you do reach for the fancy model, you can actually tell whether it worked.

## 16. Key takeaways: the ten rules

1. **Funnel, loop, gap.** Every recommender is a retrieval → ranking → re-ranking funnel, fed by a serve → log → train loop, read off the offline↔online gap. Place every problem into one of these three slots.
2. **Pick the stage before the model.** Retrieval and ranking are different problems with different metrics (Recall@K vs NDCG). The same architecture is rarely best at both.
3. **Match the loss to the objective.** Pointwise BCE for calibrated CTR, BPR/pairwise for top-K from implicit data, sampled softmax + $\log Q$ for retrieval, listwise for direct NDCG.
4. **Negatives beat architecture.** For retrieval, the negative sampling strategy often moves Recall@K more than the model does — and hard negatives are frequently false negatives, so mix carefully and validate full-catalog.
5. **Never trust offline alone.** Offline → off-policy estimate → A/B test, in that order. An offline win is a hypothesis; the A/B test is the result.
6. **Always evaluate on the full catalog with a temporal split.** Sampled metrics can reverse model rankings; random splits leak the future. Both lie.
7. **Beat the baseline first.** If you cannot beat popularity and item-CF, you have a data or harness problem, not a model problem.
8. **Build in stages.** Baseline → retrieval → ranker → multi-task → exploration → loop hygiene. Earn each layer by beating the previous one.
9. **The recall ceiling caps everything.** No ranker can rank an item retrieval never returned. Measure retrieval recall before blaming the ranker.
10. **Loop hygiene is forever.** Popularity bias, feedback loops, position bias, train-serve skew, delayed feedback, and drift never get "solved" — they get monitored and corrected continuously.

If you internalize those ten rules, you can navigate the whole field without re-reading every post — and when you do need the depth, the cross-links above drop you straight into the derivation, the code, or the war story. That is the whole point of a playbook: not to replace the fifty-nine posts that came before, but to be the index you open first.

## 17. Further reading

The seminal papers behind the playbook, the official docs for the tools, and the series posts that go deeper than this hub can.

- **Funnel and retrieval-ranking:** Covington, Adams, Sargin, "Deep Neural Networks for YouTube Recommendations," RecSys 2016 — the canonical two-stage funnel. See [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) and [the two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval).
- **Implicit feedback and BPR:** Hu, Koren, Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," 2008; Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback," 2009. See [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) and the [BPR deep dive](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive).
- **Sampled metrics warning:** Krichene and Rendle, "On Sampled Metrics for Item Recommendation," KDD 2020 — why you must evaluate on the full catalog. See [offline evaluation metrics](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr).
- **Deep ranking:** Cheng et al., "Wide & Deep Learning," 2016; Guo et al., "DeepFM," 2017; Wang et al., "Deep & Cross Network," 2017; Kang and McAuley, "SASRec," 2018. See [the ranking model](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) and [self-attention for sequences](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec).
- **Off-policy evaluation:** the IPS/SNIPS/doubly-robust literature for counterfactual estimation. See [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) and [the offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied).
- **Tools:** the FAISS documentation (Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs"), the `implicit` library docs, RecBole for unified benchmarking, and `peft` for LoRA-based LLM4Rec. See [ANN serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) and [finetuning LLMs for recommendation](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice).
- **The series map:** start at [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) for the full intro, and use this playbook as the index back into every track.

That is the whole field on one page. The funnel narrows the catalog, the loop feeds the funnel, and the gap is why you measure everything online. Pick the stage, then the model, then the loss, then the negatives, then the metric. Build the baseline first. Beat it. Watch the loop forever. Everything else is detail — and every detail has a post.

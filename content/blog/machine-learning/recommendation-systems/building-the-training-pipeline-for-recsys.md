---
title: "Building the Training Pipeline for Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The unglamorous machinery that decides whether a recommender is good — derive the embedding-table memory math that forces sharding, build a streaming PyTorch training loop with a negative sampler and checkpointing, and measure how data freshness moves Recall@10."
tags:
  [
    "recommendation-systems",
    "recsys",
    "training-pipeline",
    "embedding-tables",
    "distributed-training",
    "torchrec",
    "continuous-training",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/building-the-training-pipeline-for-recsys-1.png"
---

A team I worked with once spent six weeks tuning a ranking model. They tried DCN, then DeepFM, then a transformer over the user's history. Offline AUC crept up by a few thousandths each iteration, everyone celebrated, and online engagement did not move. Then a new engineer rewrote the *data pipeline* — fixed a point-in-time leak in one feature, switched the negative sampler from random to in-batch, and started retraining daily instead of weekly. Offline AUC actually went *down* a hair. Online click-through went up 4%. The model architecture had barely changed. The pipeline had.

This is the open secret of production recommenders: the model class you pick matters far less than the eight-stage machine that feeds it. Raw event logs become labels, labels become point-in-time-correct features, features get paired with generated negatives, negatives get batched, batches train a model whose embedding table is bigger than any single GPU's memory, the model gets checkpointed and evaluated, and finally the embeddings and weights get exported to the serving stack. Every earlier idea in this series — [the features](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders), [the negatives](/blog/machine-learning/recommendation-systems/negative-sampling-strategies), the leakage traps — slots into exactly one stage of that machine. Get the machine right and a mediocre model ships well; get it wrong and the best architecture in the literature flops.

This post is about that machine. We will build it end to end, with the math that explains *why* recsys training is memory- and I/O-bound rather than compute-bound (it is the opposite of vision and LLM training), the code for a streaming PyTorch loop that you can copy and adapt, and the measured numbers that show how a single decision — how fresh your training data is — moves Recall@10 more than a month of architecture work. By the end you will be able to size an embedding table, decide where to shard it, write a `Dataset` that streams Parquet without melting your RAM, build the train loop with checkpointing and a periodic eval gate, and set the cadence of a continuous train-serve loop. This is the part of the [recommendation funnel](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) that nobody puts on a slide, and it is where most of the wins actually live.

![The eight-stage recsys training pipeline shown as a vertical stack from raw event logs through labeling, feature pipeline, negative generation, batching, training, evaluation, and model export](/imgs/blogs/building-the-training-pipeline-for-recsys-1.png)

## 1. The pipeline is the product

Let us name the stages, because the rest of the post is organized around them. A recommender training pipeline takes raw interaction logs and produces an artifact you can serve. In between sit eight stages, drawn in the figure above:

1. **Raw event logs** — every impression, click, add-to-cart, play, skip, and dwell, landed in a data lake as Parquet or TFRecord, partitioned by date. Billions of rows per day for a large system.
2. **Labeling** — turning raw events into supervised targets. Which events are positives? What is the label for a 3-second watch versus a 30-minute watch? Where do negatives come from?
3. **Feature pipeline** — joining each labeled event to the user, item, and context features *as they were at the time of the event*. This is where point-in-time correctness lives, and where most leakage bugs are born.
4. **Negative generation** — for retrieval and many ranking losses, you need negatives the model never showed-and-the-user-ignored, or that it never showed at all. The sampler is part of the pipeline, not an afterthought.
5. **Batching and shuffling** — assembling examples into batches, shuffling to break correlation, and streaming them off disk faster than the GPU can consume them.
6. **Training** — the forward pass, the loss, the backward pass, the optimizer step. For recsys this means a giant sparse embedding lookup feeding a small dense network.
7. **Evaluation** — computing Recall@K, NDCG@K, AUC, and logloss on a held-out temporal split, gating whether this model is allowed to ship.
8. **Export** — writing the embedding tables to a vector index and the dense model to a serving format, so the funnel's retrieval and ranking stages can use them.

The reason to treat this as one pipeline rather than eight scripts is that bugs flow across stage boundaries. A labeling decision (count a 1-second dwell as a positive) silently changes what the loss optimizes. A feature joined at the wrong timestamp leaks the future into training and inflates offline metrics that collapse online. A negative sampler that over-represents popular items reinforces popularity bias three stages downstream. The pipeline is a single causal chain, and the [offline-online gap](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) is usually a pipeline bug, not a model bug.

A useful way to hold the whole thing in your head is to ask, at each stage boundary, "what contract does this stage promise the next one?" The labeling stage promises the feature stage a set of (entity, timestamp, target, weight) tuples. The feature stage promises the negative stage a leak-free feature matrix joined as-of each event's timestamp. The negative stage promises the batching stage a stream of positives-plus-negatives with a known proposal distribution. The batching stage promises the trainer a shuffled, decorrelated, prefetched flow of tensors that never starves the GPU. The trainer promises the eval stage a checkpoint. The eval stage promises the export stage a gated, known-good model. When something breaks online, you debug by walking those contracts in order and asking which one was violated — and in my experience the violation is almost never in the trainer. It is a label that meant something different than you thought, a feature that was computed one way offline and another way online, or a negative distribution that drifted out from under the loss. The trainer is the most-watched and least-buggy stage; the data stages are the opposite.

There is also a human reason the pipeline matters more than the model. Architectures are published, benchmarked, and copied — anyone can clone a DCN or a two-tower in an afternoon. The pipeline is bespoke to your data, your logging, your catalog dynamics, and your serving stack, and it is where your actual moat lives. Two teams running the identical model on the identical raw events will get wildly different online results depending on whether their pipelines handle freshness, leakage, and negatives correctly. That asymmetry — public models, private pipelines — is why this post exists and why it is the longest unglamorous one in the series.

There is one more reason the pipeline dominates: **scale**. A research recommender on MovieLens-20M has 20 million interactions, 138,000 users, and 27,000 movies. A production feed recommender has billions of events per day, hundreds of millions of users, and a catalog that turns over weekly. At that scale the bottleneck is not matrix multiplies. It is moving bytes — off disk, across the network, into and out of an embedding table that does not fit in one machine. The next section makes that precise, because it changes everything about how you build the pipeline.

## 2. Why recsys training is I/O- and memory-bound

If you come from computer vision or language modeling, your instinct is that training is bound by floating-point throughput. You count FLOPs, you buy the fastest GPU, you reach for mixed precision and tensor cores. That instinct is wrong for recommenders, and the reason is the embedding table.

Consider the canonical industrial recommender, a Deep Learning Recommendation Model (DLRM) in the style of Naumov et al. (2019). Its forward pass is:

- look up an embedding vector for each sparse feature (user ID, item ID, and dozens of categorical features),
- pool the embeddings (sum, mean, or via `EmbeddingBag`),
- concatenate with a handful of dense features,
- push the result through a small multilayer perceptron (MLP), maybe three to five layers of a few hundred units each,
- emit a single score.

Count the work. The MLP is tiny — a 512→256→128→1 stack is on the order of a hundred thousand multiply-adds per example, which is nothing. The embedding lookup touches maybe 30 to 100 rows of the table, a gather of a few thousand floats. The arithmetic intensity — FLOPs per byte moved — is *low*. The model spends its time fetching embedding rows from memory, not multiplying them. This is the defining contrast with vision and LLMs.

![A comparison matrix of recsys versus vision versus LLM training across bottleneck, memory hog, data scale, and FLOPs showing recsys is I/O and memory bound on the embedding table](/imgs/blogs/building-the-training-pipeline-for-recsys-3.png)

The matrix above lays it out. A ResNet or a GPT spends its life compute-bound: the parameters are dense, every one participates in every example, and the GPU's tensor cores are the limiter. A DLRM spends its life I/O- and memory-bound: most of the parameters live in a sparse embedding table, any one example touches a tiny fraction of them, and the limiter is how fast you can stream training rows off disk and gather embedding rows out of memory. If you profile a recsys training job and see the GPU's compute utilization sitting at 20% while the job is "slow," you have not found a bug — you have found the nature of the workload. The fix is never "buy more FLOPs." It is faster I/O, a sharded table, and a data loader that keeps the GPU fed.

This has three concrete consequences that shape the whole pipeline:

**Memory is dominated by the embedding table.** The dense MLP is a few megabytes. The embedding table can be tens to hundreds of gigabytes. We derive the exact arithmetic in section 5, but the headline is that the table is the thing that does not fit, so the table is the thing you design around.

**Throughput is bounded by I/O and the gather, not by the matmul.** Your job is records per second, and that number is set by disk read bandwidth, decompression, host-to-device transfer, and the embedding gather — not by the FLOPs of the MLP. A training run that is "too slow" is almost always starved of data, not starved of compute.

**Fresh data beats more data.** Because the limiter is throughput and the catalog drifts, you get more value from training on the last day of data, fresh, than from grinding through the last year, stale. Section 8 measures this. It is the single most counterintuitive fact in recsys engineering for people arriving from other ML fields, where "more data, more epochs" is gospel.

It is worth dwelling on the arithmetic intensity argument, because it is what makes the rest of the post non-negotiable rather than a matter of taste. The roofline model (the subject of a whole [HPC post on compute-bound versus memory-bound work](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)) says a kernel is memory-bound when its FLOPs-per-byte sits to the left of the hardware's ridge point — the ratio of peak compute to peak memory bandwidth. An A100 has a ridge point around 100–200 FLOPs per byte: below that, you are bandwidth-limited; above it, compute-limited. A dense GEMM in a transformer has arithmetic intensity in the hundreds to thousands and lands firmly on the compute side. An embedding gather has arithmetic intensity *near zero* — it reads a row and does almost nothing with it before the next read. So the embedding stage is pinned to the far-left, bandwidth-bound corner of the roofline, and no amount of faster floating-point helps. This is not a quirk of one model; it is a structural property of sparse-feature recommendation, and it is why the entire HPC playbook you would apply to an LLM (tensor cores, kernel fusion, FlashAttention) is mostly irrelevant here while the playbook you would apply to a database (fast storage, columnar reads, caching, sharding) is exactly right.

The same argument explains a question new recsys engineers always ask: "why is my GPU utilization so low, and how do I fix it?" The honest answer is that for the embedding-dominated part of the model, *high GPU utilization is not even the goal* — the goal is high records-per-second, and you can hit your throughput target with the GPU's compute units mostly idle because they were never the bottleneck. Chasing utilization for its own sake leads people to inflate batch sizes or widen MLPs until the compute units light up, which makes the dashboard look healthy while the records-per-second number, the one that actually pays the bills, gets worse. Measure throughput, not utilization.

#### Worked example: where does the time go in a DLRM step?

Suppose one training example has 26 sparse features, each looking up a 64-dimensional embedding, plus a 512→256→128→1 MLP. The embedding gather moves about $26 \times 64 \times 4 = 6{,}656$ bytes per example. The MLP forward is roughly $512{\times}256 + 256{\times}128 + 128{\times}1 \approx 1.6 \times 10^5$ multiply-adds, so about $3.3 \times 10^5$ FLOPs. At a batch of 4,096, that is about 27 MB of embedding traffic and $1.4 \times 10^9$ FLOPs for the MLP. An A100 does roughly $3 \times 10^{14}$ FLOPs per second in bf16, so the MLP math takes about 5 microseconds. Reading and decompressing 27 MB of embedding rows plus the input batch off the memory subsystem takes far longer than 5 microseconds. The example is memory-bound by one to two orders of magnitude — and that is *before* the table is sharded across the network. The compute is free; the data movement is the bill.

## 3. Stage by stage: events, labels, and features

Now we walk the pipeline. The first three stages — events, labeling, features — are where data quality is won or lost, and they have nothing to do with the model.

### Raw event logs

Events arrive as an append-only stream and land in a partitioned columnar store, almost always Parquet (Spark, Flink, or a warehouse) or TFRecord (TensorFlow shops). Partitioning by date is non-negotiable, because every downstream decision — what window to train on, how to do a temporal split, how to retrain incrementally — is a date-range select. A typical raw event row is wide and ugly:

```python
# One raw impression-level event (conceptual schema, not code to run)
{
    "ts":            1718755200,        # unix seconds, when it happened
    "user_id":       "u_8842113",
    "item_id":       "i_5519027",
    "surface":       "home_feed",
    "position":      3,                  # rank where it was shown
    "action":        "click",            # impression | click | play | purchase | skip
    "dwell_ms":      14200,              # how long they engaged
    "device":        "ios",
    "session_id":    "s_99x",
}
```

You log impressions, not just clicks. This matters enormously: a click is only informative relative to the things shown-and-not-clicked in the same session. If you only log clicks, you can never construct honest in-session negatives, and you bake position bias into every label. Log the impression stream with positions.

### Labeling: turning events into targets

Labeling is the most underrated stage. The naive choice — "click = 1, no click = 0" — throws away the richest signal you have: *how much* the user engaged. A 2-second dwell and a 20-minute watch are both "clicks," but they are not the same label.

The standard move is to convert engagement into a **weight** or a **graded relevance**. Two common patterns:

```python
import numpy as np

def label_from_event(ev: dict) -> tuple[int, float]:
    """Return (binary_label, sample_weight) from a raw event.

    Dwell time is mapped to a weight so that deeper engagement
    contributes a larger gradient, without changing the 0/1 target
    that the logloss is computed against.
    """
    if ev["action"] in ("purchase",):
        return 1, 3.0                      # strong positive, upweighted
    if ev["action"] in ("play", "click"):
        # log-scaled dwell weight, clipped so one outlier cannot dominate
        w = 1.0 + np.log1p(ev["dwell_ms"] / 1000.0)   # ~1.0 .. ~5.0
        w = float(np.clip(w, 1.0, 5.0))
        # treat a sub-second "click" as a non-engagement (likely misclick)
        label = 1 if ev["dwell_ms"] >= 1000 else 0
        return label, w
    return 0, 1.0                          # impression-only = negative
```

Two judgment calls live in those eight lines. First, the **dwell threshold** that separates a real engagement from a misclick — set it from the bimodal dwell distribution your logs actually show, not from a round number. Second, the **weight cap**, because a single binge session should not swamp a batch. These are not hyperparameters you grid-search; they are product decisions that determine what "good" means, and they should be reviewed by whoever owns the metric. A model trained to maximize raw click count and a model trained to maximize dwell-weighted engagement are *different products*, even with identical architecture.

There is a deeper statistical reason the weighting matters, and it is worth making rigorous. The weighted binary cross-entropy you are about to optimize is

$$ \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} w_i \left[ y_i \log \sigma(s_i) + (1 - y_i)\log(1 - \sigma(s_i)) \right] $$

where $s_i$ is the model's logit, $y_i \in \{0,1\}$ is the binary target, and $w_i$ is the dwell-derived weight. The weight does not change *what counts as a positive* — that is the $y_i$ — it changes *how hard the gradient pushes* on that example. A 30-minute watch and a 30-second watch are both $y=1$, but the gradient on the long watch is scaled up by its larger $w$, so the model spends more of its capacity getting the deep-engagement examples right. This is exactly what you want: a recommender that nails the items people love and is merely okay on the items they tolerate beats one that treats every click as identical. But it also means a mis-set weight cap silently re-weights your entire objective. If the cap is too high, a handful of binge sessions dominate the gradient and the model overfits to whales; too low and you have thrown away the engagement signal you went to the trouble of logging. The cap is a real lever on the loss, not a cosmetic clamp.

A second labeling subtlety that bites in production: **delayed feedback.** A click is observed immediately, but a conversion (purchase, subscription, a watch that counts only if it lasts a week) can arrive hours or days after the impression. If you label and train before the conversion window closes, you mislabel converters as negatives and teach the model that good recommendations are bad. The pipeline fix is an *attribution window*: hold an event's final label until its conversion window has elapsed, or use a delayed-feedback loss that models the censoring explicitly (Chapelle, 2014, on delayed-feedback modeling for conversion prediction). For a continuously retrained model this interacts directly with freshness — you want fresh data, but the freshest data has not had time to reveal its conversions yet. The standard compromise is to train clicks fresh and conversions on a lag, or to maintain two label streams with different latencies. This is a pipeline decision, made before any model sees a single row.

For multi-task ranking — predicting click *and* dwell *and* purchase jointly — labeling produces several targets per event, which is exactly the input the [multi-task ranking models](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) consume. The pipeline's job is to emit the right columns; the model decides how to combine them.

### Features: point-in-time correctness or bust

Now the join. Each labeled event needs the features that describe its user, item, and context — but **as they were the instant before the event happened.** This is the single most dangerous line in the entire pipeline, because the obvious implementation is wrong and the wrong version makes your offline metrics *better*.

The leak goes like this. You want a feature `user_7d_click_count`. The easy query computes it over the whole training table. But that window includes clicks that happened *after* the event you are labeling. The model gets to peek at the user's future, the feature becomes near-perfect, offline AUC rockets, and online the feature is unavailable (the future has not happened yet) so the model degrades to noise. This is the train-serve skew that the [data and features post](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders) treats in full; here we just enforce it at the join.

The correct join is a point-in-time (PIT) join, also called an as-of join: for each event at time $t$, attach the feature value valid at the largest feature timestamp $\le t$.

```python
import polars as pl

# events: columns [user_id, ts, item_id, label, weight]
# user_feats: a slowly-changing feature table with [user_id, feat_ts, u7d_clicks, ...]
events = events.sort("ts")
user_feats = user_feats.sort("feat_ts")

# as-of join: for each event, take the latest feature row with feat_ts <= ts.
# join_asof is the leak-safe primitive; a naive equi-join on user_id would leak.
training = events.join_asof(
    user_feats,
    left_on="ts",
    right_on="feat_ts",
    by="user_id",
    strategy="backward",      # only look BACKWARD in time, never forward
)
```

The `strategy="backward"` flag is the whole game: it forbids the join from reaching forward in time. If you remember one line from this post, it is that every feature join in a recsys pipeline must be a backward as-of join, and that the offline metric *rising* after you "fix" a feature is a red flag, not a celebration — you may have just reintroduced a leak. The healthy direction is often a small *drop* in offline AUC as a leak is removed, followed by an online *gain* because the model is now learning something it can actually use at serving time.

The leak can be subtler than a too-wide window. Consider a feature like `item_ctr` (the item's click-through rate). If you compute it over the full training table, an item's CTR includes the very clicks you are trying to predict — the label is baked into the feature, and the model achieves uncanny offline accuracy by reading the answer off the input. At serve time `item_ctr` is computed from past data only, so it is a different, weaker feature, and the model collapses. The PIT discipline catches this: `item_ctr` must be the as-of value, computed only from clicks strictly before the event. The same trap hides in user-aggregate features (`user_avg_rating`), in "this item is trending" flags, and in any feature derived from the label. A good rule: any feature that is a function of outcomes must be computed with a backward window and joined as-of, no exceptions, and you should have an automated check that diffs the offline feature against the online feature service on a sample of live traffic — if they disagree, you have skew before you have a model.

There is also a throughput dimension to the feature stage that people underestimate. The as-of join is the most expensive operation in the whole pipeline at scale, because it is a sorted merge of two billion-row streams per entity key. Naive implementations re-sort or re-scan; the efficient ones exploit the fact that both sides are already date-partitioned and sorted by timestamp, so the join is a linear merge. In practice you precompute slowly-changing features into snapshot tables keyed by (entity, date) and join on the snapshot for the event's date, which turns the expensive temporal merge into a cheap equi-join while preserving correctness. The cost of getting this wrong is not just slowness — a feature pipeline that takes 18 hours cannot feed a daily retrain, so feature-join performance directly bounds how fresh your model can be. Freshness and join efficiency are the same constraint wearing two hats.

## 4. Negative generation is part of the pipeline

For ranking with explicit impressions, your negatives are the shown-and-not-clicked items, which the labeling stage already produced. For *retrieval*, the problem is harder and more interesting: the model must learn to pull good items out of the entire catalog, and the catalog has hundreds of millions of items the user never saw. You cannot score them all, so you sample negatives. The sampler is a pipeline component with its own throughput and correctness budget, not a one-liner inside the loss.

The [negative sampling strategies post](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) derives why the choice of negative distribution is the single biggest lever in retrieval quality. Here we care about *where in the pipeline* the negatives are generated, because that decision trades throughput against quality:

- **In-batch negatives.** Treat the other positives in the same batch as negatives for each example. Zero extra I/O — the embeddings are already computed. This is why it is the workhorse for two-tower retrieval at scale: the [two-tower training post](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) shows it gives you $B-1$ negatives per positive for the price of one batch. The catch is that the negative distribution is the batch's positive distribution, which is popularity-skewed, so you apply the $\log Q$ correction.
- **Precomputed / offline-mined negatives.** Sample negatives ahead of time (uniform, popularity-weighted, or hard negatives mined from a previous model's near-misses) and write them into the training shards. This costs an extra pass and storage, but it lets you control the distribution exactly and inject hard negatives the in-batch set will never contain.

A minimal in-pipeline sampler that mixes both — in-batch for cheap volume, plus a few mined hard negatives per positive — looks like this:

```python
import torch

class NegativeSampler:
    """Popularity-weighted negative sampler with optional hard negatives.

    Built once from item frequencies; called per batch inside collate.
    Frequencies come from the SAME stream the model trains on so the
    proposal Q matches the data, which is what the log Q correction needs.
    """
    def __init__(self, item_freq: torch.Tensor, alpha: float = 0.75):
        # popularity^0.75, the word2vec smoothing that tempers head dominance
        probs = item_freq.clamp(min=1).pow(alpha)
        self.probs = probs / probs.sum()
        self.log_q = torch.log(self.probs)            # for the correction

    def sample(self, n: int) -> torch.Tensor:
        return torch.multinomial(self.probs, n, replacement=True)

    def correction(self, item_ids: torch.Tensor) -> torch.Tensor:
        # subtract log Q(j) from each sampled negative's logit
        return self.log_q[item_ids]
```

The key engineering point: build `item_freq` from the *same* event stream the model trains on, refresh it as the stream drifts, and treat the sampler's output as just another column the batching stage assembles. When a recommender quietly collapses to ten viral items, the sampler is on the short suspect list — an uncorrected in-batch sampler over-penalizes the tail's absence and the feedback loop does the rest.

The throughput contrast between the two negative sources is stark and dictates the design. In-batch negatives cost essentially nothing: the embeddings for the other positives in the batch are already in GPU memory from the forward pass, so each positive gets $B-1$ negatives for free, and at $B = 8192$ that is over eight thousand negatives per positive at zero extra I/O. Precomputed negatives cost a full extra column in every training row — if you write four mined hard negatives per positive into the shards, you have grown the training set's negative payload fourfold on disk, which the throughput-bound pipeline pays for in read bandwidth. So the production pattern is almost always *in-batch for volume, plus a thin layer of mined hard negatives for quality*: you get the cheap thousands from the batch and spend the I/O budget only on the few negatives that actually teach the model something it cannot learn from random ones. The sampler's job in the pipeline is to produce that thin hard-negative layer efficiently — typically by periodically running the current model over a candidate pool, recording the high-scoring non-positives, and writing them back as the next cycle's hard negatives. That mining pass is itself a small training-pipeline run, which is why the sampler is genuinely a pipeline stage and not a loss-function detail.

One more correctness note that ties back to the labeling stage: a negative must be a *true* non-engagement, not an unlogged item that the user would have loved. The shown-and-not-clicked negatives from the impression log are the gold standard because you *know* the user saw and rejected them. Sampled negatives from the catalog are guesses, and some fraction are false negatives — items the user would have engaged with had they been shown. The [negative sampling post](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) treats the false-negative problem in depth; here the pipeline point is that mixing logged hard negatives (true) with sampled negatives (some false) dilutes the false-negative noise, which is why production samplers blend sources rather than relying on a single one.

## 5. The embedding table: the memory math that forces sharding

Here is the science block. Everything about distributed recsys training follows from one product.

An embedding table maps each ID in a vocabulary to a dense vector. Its memory footprint is:

$$ M = |V| \times d \times b $$

where $|V|$ is the vocabulary size (number of distinct IDs), $d$ is the embedding dimension, and $b$ is bytes per element ($b = 4$ for fp32, $2$ for fp16/bf16). That is the whole formula, and it is brutal in how fast it grows.

![The embedding table memory breakdown shown as a stack of vocabulary size times dimension times bytes per float yielding a 25.6 gigabyte table that forces sharding](/imgs/blogs/building-the-training-pipeline-for-recsys-5.png)

Plug in the numbers from the figure. A vocabulary of 100 million item IDs at $d = 64$ in fp32:

$$ M = 10^8 \times 64 \times 4 \text{ bytes} = 2.56 \times 10^{10} \text{ bytes} \approx 25.6 \text{ GB} $$

One table, one feature, 25.6 GB. A real industrial model has *many* such tables — user IDs, item IDs, advertiser IDs, page IDs, and dozens of high-cardinality categoricals — and the largest deployed embedding tables in industry have crossed into the terabyte range (Meta has publicly described multi-terabyte DLRM embedding tables). The dense MLP that everyone draws on the architecture slide is, by comparison, a rounding error: a few million parameters, single-digit megabytes.

This is why an all-reduce will not save you. Data-parallel training replicates the model on every GPU and all-reduces the gradients each step. Replicating a 25.6 GB table on every GPU means every GPU needs 25.6 GB just for that one table before you add the optimizer state (Adam doubles or triples it). A single 80 GB A100 cannot hold one big table plus its Adam moments plus activations plus the batch. The table does not fit, full stop.

![Before and after view contrasting a single host where a 100 gigabyte embedding table causes a CUDA out-of-memory crash against a four-way sharded layout holding 25 gigabytes per host](/imgs/blogs/building-the-training-pipeline-for-recsys-4.png)

The fix is to **shard the table**: partition its rows across hosts so each host holds a slice. The before-after figure shows the move. A 100 GB table on one 80 GB GPU OOMs; the same table split four ways puts 25 GB on each host, which fits with room for the rest of the step. This is model parallelism applied to the embedding, and it is the structural difference between recsys distributed training and everything else.

#### Worked example: sizing a billion-ID system

A marketplace wants to embed users (200M), items (300M), and sellers (5M), plus 20 categorical features averaging 1M cardinality each. At $d = 64$, fp32:

- Users: $2 \times 10^8 \times 64 \times 4 = 51.2$ GB
- Items: $3 \times 10^8 \times 64 \times 4 = 76.8$ GB
- Sellers: $5 \times 10^6 \times 64 \times 4 = 1.28$ GB
- 20 categoricals: $20 \times 10^6 \times 64 \times 4 = 5.12$ GB

Total raw weights: about **134 GB**. With Adam's two moment buffers, multiply by roughly 3 → about **400 GB** of optimizer-plus-weight state for the embeddings alone. No single GPU holds that. Even an 8×80 GB node (640 GB) is tight once you add activations and the dense replica. Switching the two big tables to fp16 nearly halves the weight memory; row-wise Adagrad (one scalar per row instead of two full vectors) shrinks optimizer state dramatically, which is exactly why production stacks favor it. The arithmetic is what *decides* the sharding plan and the optimizer — you do this calculation *before* you write a line of training code, because it tells you how many hosts you need.

## 6. Distributed training: shard the embeddings, replicate the dense net

So the embedding table is model-parallel (sharded), and the dense MLP is data-parallel (replicated). This hybrid is the heart of every large-scale recsys trainer, and it is what Meta's TorchRec library exists to make tractable.

![A distributed training graph showing the global batch fanning into two embedding shards, an all-to-all gather, a replicated MLP, an all-reduce on the MLP gradients only, and a combined optimizer step](/imgs/blogs/building-the-training-pipeline-for-recsys-2.png)

Trace the figure. A global batch is split across GPUs. Each GPU needs embedding rows for the IDs in its slice of the batch, but those rows live on whatever shard owns them — so the system does an **all-to-all** communication: every GPU sends its ID lookups to the shard that holds them and receives back the embedding vectors. The gathered embeddings feed the **replicated MLP**, which runs identically on every GPU (data-parallel). On the backward pass, the MLP gradients are **all-reduced** (averaged across GPUs, the standard data-parallel sync), while the embedding gradients flow back through the all-to-all to update only the rows each shard owns. The optimizer then steps the sharded table rows and the replicated dense weights together.

The crucial asymmetry: the all-reduce moves only the *small* dense gradients (megabytes), never the *huge* embedding table (gigabytes). If you naively tried to all-reduce the embedding gradients you would move tens of gigabytes per step across the network and the job would crawl. Sharding plus all-to-all moves only the embedding rows the batch actually touched — a tiny, sparse fraction.

Here is the data-parallel skeleton for the dense part, which is exactly the PyTorch `DistributedDataParallel` (DDP) pattern, and a sketch of where the embedding sharding diverges:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class DLRM(torch.nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        # In a real large-scale job this EmbeddingBag is REPLACED by a
        # sharded table (torchrec.EmbeddingBagCollection wrapped in
        # DistributedModelParallel). The MLP below stays data-parallel.
        self.emb = torch.nn.EmbeddingBag(vocab, dim, mode="mean")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim + 13, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, sparse_ids, offsets, dense_x):
        e = self.emb(sparse_ids, offsets)        # sparse: sharded in prod
        x = torch.cat([e, dense_x], dim=1)
        return self.mlp(x).squeeze(-1)            # dense: replicated (DDP)

def train(rank, world_size):
    setup(rank, world_size)
    model = DLRM(vocab=100_000_000, dim=64).cuda(rank)
    # DDP wraps the DENSE module; the embedding is handled separately
    # by the sharding plan. Mixing the two is what torchrec automates.
    model = DDP(model, device_ids=[rank])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ...
```

For the real thing you do not hand-roll the all-to-all. You let TorchRec plan the shard placement (it has a cost-model-driven planner that decides table-wise, row-wise, or column-wise sharding to balance memory and communication) and wrap the model in `DistributedModelParallel`:

```python
# Sketch of the torchrec pattern (the planner does the hard part)
import torchrec
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP

ebc = torchrec.EmbeddingBagCollection(
    tables=[
        torchrec.EmbeddingBagConfig(
            name="item", embedding_dim=64, num_embeddings=100_000_000,
            feature_names=["item_id"]),
        torchrec.EmbeddingBagConfig(
            name="user", embedding_dim=64, num_embeddings=200_000_000,
            feature_names=["user_id"]),
    ],
    device=torch.device("meta"),     # meta init: do not materialize 100GB on rank 0
)
model = DMP(module=MyDLRM(ebc), device=torch.device("cuda"))
# DMP shards the EmbeddingBagCollection across ranks and keeps the
# dense submodules data-parallel. The all-to-all is generated for you.
```

The `device="meta"` trick matters: you *cannot* instantiate a 100 GB table on rank 0 and then shard it — rank 0 would OOM before sharding starts. Meta-device init creates the table lazily, shard by shard, on the device that will own each slice.

It is worth understanding the three sharding schemes the planner chooses among, because the choice is a throughput-versus-balance trade-off you will sometimes need to override:

- **Table-wise sharding** puts each whole table on one device. Simple, and great when you have many medium tables and many devices — each device owns a few complete tables. The risk is imbalance: if one table is far hotter (more lookups per batch) than the others, its device becomes a straggler that the all-to-all waits on.
- **Row-wise sharding** splits a single big table's rows across all devices. This is the scheme that lets a 300M-item table that fits on no single device live across eight of them, 37.5M rows each. The cost is that every batch's lookups for that table fan out to every device (a denser all-to-all), but for the truly huge tables there is no alternative.
- **Column-wise sharding** splits a table's embedding *dimension* across devices — each device holds, say, dimensions 0–31 of every row. This balances load perfectly (every device does the same work for every lookup) and is handy for a small number of very wide, very hot tables, at the cost of having to concatenate the partial vectors after the gather.

The planner's cost model weighs the lookup frequency of each table, the device memory budget, and the network topology to pick a mix that minimizes the slowest device's time. You usually let it decide, but when you profile and find the all-to-all unbalanced — one device finishing late every step — the fix is to override the plan for the offending table, typically switching a hot table from table-wise to row-wise so its load spreads out. This is the recsys analogue of the parallelism-strategy decisions the [parallelism strategies post](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert) covers for dense models, with the twist that the thing being parallelized is a lookup table, not a matmul.

A subtle but important point about the optimizer in this regime: for the sharded embedding, you almost never use full Adam. Adam keeps two moment buffers per parameter, tripling the table's memory — a 100 GB table becomes 300 GB of state. Production stacks use **row-wise Adagrad** or a similar scheme that keeps one scalar of optimizer state per *row* rather than two full vectors per *element*, which shrinks the embedding optimizer state from $2 \times |V| \times d$ floats to just $|V|$ floats — a $2d$-fold reduction, or 128× at $d=64$. The dense MLP, being tiny, can afford full Adam. So even the optimizer is split: cheap per-row state for the giant sparse table, rich per-parameter state for the small dense net. This asymmetry — different optimizers for the sparse and dense halves of the same model — surprises people coming from single-optimizer training, and it is a direct consequence of the memory arithmetic from section 5.

This shard-embeddings-replicate-dense pattern is the modern descendant of the **parameter server** architecture (Li et al., 2014), where a set of server nodes held the model parameters (the embedding table) and worker nodes pulled the rows they needed, computed gradients, and pushed updates back. TorchRec's all-to-all is a tighter, collective-communication version of the same idea, and the [collective communication and NCCL all-reduce post](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) covers the primitives underneath it. The parameter-server pattern is still alive for tables so large (terabytes) that they must live on CPU host memory or a dedicated key-value store rather than GPU HBM.

## 7. Streaming, batching, and keeping the GPU fed

Recall the diagnosis from section 2: recsys training is throughput-bound, and the GPU is usually starved, not saturated. The batching stage is where you fix or fail that. Two rules dominate.

**Stream; do not load.** You cannot fit a billion-row training set in RAM, so the `Dataset` must read shards lazily and yield batches on the fly. Parquet's row-group structure is built for this — you read one row group at a time, decode it, and hand it off. Here is a streaming `IterableDataset` over Parquet that shuffles within a buffer (full global shuffle is impossible at this scale, so you approximate it):

```python
import torch
import pyarrow.parquet as pq
import numpy as np
import glob, random

class ParquetStream(torch.utils.data.IterableDataset):
    """Streams batches from a directory of Parquet shards.

    Approximate shuffle: shuffle the shard order, then shuffle a
    fixed-size buffer of rows. This breaks within-shard correlation
    without ever holding the full dataset in memory.
    """
    def __init__(self, path_glob, batch_size=4096, buffer=200_000, seed=0):
        self.files = sorted(glob.glob(path_glob))
        self.bs, self.buffer, self.seed = batch_size, buffer, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        files = list(self.files)
        rng = random.Random(self.seed + (info.id if info else 0))
        rng.shuffle(files)                          # shuffle shard order
        if info:                                    # shard files across workers
            files = files[info.id :: info.num_workers]
        buf = []
        for f in files:
            for rg in range(pq.ParquetFile(f).num_row_groups):
                tbl = pq.ParquetFile(f).read_row_group(rg)
                rows = tbl.to_pylist()
                buf.extend(rows)
                rng.shuffle(buf)                    # shuffle the buffer
                while len(buf) >= self.buffer:
                    batch, buf = buf[: self.bs], buf[self.bs :]
                    yield self.collate(batch)
        while buf:                                  # drain the tail
            batch, buf = buf[: self.bs], buf[self.bs :]
            yield self.collate(batch)

    def collate(self, rows):
        item = torch.tensor([r["item_id"] for r in rows], dtype=torch.long)
        user = torch.tensor([r["user_id"] for r in rows], dtype=torch.long)
        y    = torch.tensor([r["label"]   for r in rows], dtype=torch.float)
        w    = torch.tensor([r["weight"]  for r in rows], dtype=torch.float)
        return user, item, y, w
```

**Overlap I/O with compute.** Use `DataLoader(num_workers=N, prefetch_factor=k, pin_memory=True)` so that while the GPU trains on batch $t$, worker processes are decoding and assembling batch $t+1$. With an `IterableDataset` you also shard the file list across workers (the `files[info.id :: info.num_workers]` line) so workers do not all read the same shard. The whole point is to hide disk and decode latency behind the GPU step. If `nvidia-smi` shows the GPU idling between steps, your loader is the bottleneck — add workers, increase prefetch, or move decoding to a faster format.

**Why shuffling matters.** Events arrive time-ordered and user-clustered. If you train in that order, consecutive batches are highly correlated (same user, same session), the gradient estimate is biased within each window, and the model can overfit to whatever block it most recently saw. The buffer shuffle decorrelates within a window; shuffling shard order decorrelates across windows. You will not get a perfect global shuffle on a billion rows, and that is fine — a large enough buffer (hundreds of thousands of rows) is empirically close enough.

There is a real tension with the next section: shuffling fights recency. A perfectly shuffled stream mixes today's and last-month's events uniformly, but you often *want* the model to weight recent events more. The pragmatic answer is to bound the training window (train on the last $N$ days, not all of history) and shuffle within it. That bounds staleness and keeps the shuffle honest.

**The on-disk format is a throughput decision, not a preference.** For the I/O-bound recsys pipeline, the format you store training data in is one of the highest-leverage choices you make, and it is worth being deliberate. Two formats dominate:

- **Parquet** is columnar, compressed, and splittable by row group. Columnar layout means you can read only the columns the model needs (skip the debugging columns), and per-column compression is excellent on the low-cardinality categoricals that fill recsys data. The row-group structure gives you natural streaming chunks. This is the default for Spark/Flink/warehouse shops and pairs naturally with Polars or PyArrow on the read side.
- **TFRecord** is a sequence of serialized protobuf records, the TensorFlow-native format. It streams beautifully and integrates with `tf.data`'s prefetch-and-interleave machinery, but it is row-oriented, so you pay to read columns you do not need, and it is less convenient to inspect than Parquet.

The non-obvious lever inside either format is **compression codec choice**. Because the pipeline is bandwidth-bound, a codec that decompresses fast (Snappy, LZ4) often beats one that compresses small (Gzip, Zstd at high levels): you would rather read more bytes off a fast disk and decompress them cheaply than read fewer bytes and burn CPU decompressing them, *if* the CPU is the bottleneck — and in a recsys loader with many workers, CPU decode frequently is. Profile both. The other lever is **record packing**: storing pre-batched, pre-collated tensors on disk (a few thousand examples per record, already in the layout the model wants) eliminates per-row Python object construction in the collate function, which is often the single largest CPU cost in a naive loader. The TorchRec and `tf.data` ecosystems both support this, and it can double loader throughput by itself.

A final batching subtlety specific to recsys: **variable-length sparse features.** A user's history is a list of variable length, and items have variable numbers of tags. You cannot pad these to a fixed length without wasting memory on the long tail, so recsys batching uses the *jagged* or *ragged* representation: a flat values tensor plus an offsets tensor that says where each example's slice begins. `nn.EmbeddingBag` consumes exactly this `(values, offsets)` pair, which is why it appears throughout this post — it is built for jagged sparse features. Getting the offsets right in the collate function is a common source of silent bugs (an off-by-one in offsets misattributes one user's history to the next), so it is worth a unit test that checks a known batch round-trips correctly.

## 8. Freshness, staleness, and continuous training

Now the most important number in recsys engineering: **how fresh is your training data?**

Models go stale because the world moves. New items launch with no training rows. User interests shift. A meme that did not exist last week dominates the feed this week. A model trained on a 30-day window and frozen will, within days, be recommending yesterday's catalog to today's users. The [popularity-bias dynamics](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) make this worse: a stale model keeps surfacing what was popular when it trained, which suppresses new items further.

![Before and after view contrasting a model trained on month-old data with Recall@10 of 0.31 against a daily continuously retrained model holding Recall@10 at 0.42](/imgs/blogs/building-the-training-pipeline-for-recsys-6.png)

The before-after figure shows the shape of it, and the staleness/recency trade-off is the science to make precise. There are two competing forces:

- **More history → more signal per ID.** A long window gives each item more interactions, which means better-estimated embeddings, especially for the tail. This pushes the window *longer*.
- **Older data → distribution shift.** Old interactions describe a world that no longer exists; their gradient pulls the model toward a stale optimum. This pushes the window *shorter* and the retrain cadence *faster*.

The optimum is a window long enough to estimate embeddings well but short enough to track the current distribution, retrained often enough that the served model is never far behind the live data. For fast-moving feeds that often means a multi-week window retrained daily, or even an hourly incremental update on top of a daily full retrain.

The mechanism that makes frequent retraining affordable is **warm-starting**: you do not train from scratch each cycle. You initialize from yesterday's checkpoint and train on the new day's data (plus a rolling window), so the embedding table already knows almost everything and only needs to absorb the delta. This is **continuous** or **incremental training**, and it is what makes a daily — or faster — cadence economically possible.

Warm-starting has its own failure mode worth naming: **embedding drift and the cold-start row.** When a brand-new item ID first appears, its embedding row is freshly initialized (random or zero) and the model has seen it exactly zero times. A single incremental update on a handful of impressions for that item is not enough to learn a good embedding, so new items underperform until they accumulate interactions — the classic cold-start problem, now living in the training cadence. Two pipeline mitigations help. First, **initialize new-item rows from content** rather than from scratch: project the item's text or image embedding into the table's dimension so a new item starts near similar existing items instead of at the origin (this is the bridge to [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders)). Second, **upweight recent and rare items** in the incremental batches so the warm-start actually moves their rows rather than leaving them where the previous checkpoint left them. Without these, a continuously trained model can paradoxically be *worse* at new items than a from-scratch retrain, because warm-starting biases it toward the well-trodden rows it already knows.

There is also a stability question. Each incremental update is a few optimizer steps on a narrow slice of data, and a string of such updates can slowly drift the model away from a good optimum — small biases compound, the loss landscape the model sits in shifts, and after a few weeks of pure incremental updates the model can degrade in ways no single update revealed. The standard guard is the **two-tier cadence** mentioned in the worked example below: frequent incremental updates to stay fresh, punctuated by a periodic full retrain from scratch (or from a much longer window) that re-anchors the model to a clean optimum. The full retrain is the reset button; the incremental updates are the fine adjustment between resets. Deciding the ratio — how many incremental cycles between full retrains — is an empirical call you make by watching whether the metric drifts down over a run of incremental-only updates.

![A continuous training loop drawn as an acyclic chain from serve to log to label to incremental train, warm-started from the prior checkpoint, through an eval gate to redeploy](/imgs/blogs/building-the-training-pipeline-for-recsys-7.png)

The loop is drawn above as an acyclic chain (it is a loop in time, but each cycle is a fresh forward step that warm-starts from the previous checkpoint — not a literal ring). Serve model day $N$ → log the events it generated → label and feature them point-in-time → warm-start from the day-$N$ checkpoint → incrementally train on the new window → gate on offline eval → deploy as day $N{+}1$. This is the [feedback loop](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) that is the spine of the whole series, instantiated as a concrete pipeline cadence. Google has publicly described continuous-training pipelines of exactly this shape (the TFX continuous-training pattern), where models retrain on a schedule and are gated by validation before promotion.

#### Worked example: sizing throughput to retrain daily

Suppose your training window is the last 14 days, your system logs 500 million labeled events per day, and you want to train **3 epochs** over that window every night, finishing in a **4-hour** batch slot.

- Window size: $14 \times 5 \times 10^8 = 7 \times 10^9$ events.
- Total examples to process: $3 \times 7 \times 10^9 = 2.1 \times 10^{10}$.
- Time budget: $4 \text{ hours} = 1.44 \times 10^4$ seconds.
- Required throughput: $2.1 \times 10^{10} / 1.44 \times 10^4 \approx 1.46 \times 10^6$ **records/sec**.

So you need to sustain about **1.5 million records per second**, end to end, through the whole pipeline. That number sizes everything: how many data-loader workers, how much disk read bandwidth, how many GPUs across how many shards. If a single GPU with a well-fed loader does ~500k records/sec on this model (a defensible order of magnitude for a small-MLP DLRM; profile yours), you need on the order of 3–4 GPUs working in parallel just to hit the cadence — and that is the *minimum*, before headroom for stragglers and re-runs. Notice what set the requirement: not FLOPs, but records/sec, which is I/O. This is the throughput model from section 2 turned into a hardware order.

Warm-starting changes the math favorably for *incremental* updates: if you only train on the newest day on top of yesterday's checkpoint (1 epoch over $5 \times 10^8$ events in, say, a 1-hour slot), you need only about $1.4 \times 10^5$ records/sec — a single well-fed GPU. The daily full retrain re-anchors the model; the incremental updates keep it fresh between full retrains. That two-tier cadence (slow full retrain + fast incremental) is the standard production compromise.

## 9. Checkpointing, evaluation gates, and export

The training loop is not just the forward and backward pass. It has to checkpoint (so a 4-hour job that dies at hour 3 does not start over, and so tomorrow can warm-start), evaluate on a held-out temporal split (so a bad model never ships), and export (so the funnel can serve it). Here is the loop with all three, written to be readable rather than maximally optimized:

```python
import torch, os, time

def evaluate(model, eval_loader, device):
    """Recall@10 on a temporal eval split.

    We rank a held-out positive against a fixed candidate set per user
    and check whether it lands in the top 10. Temporal split (eval is
    strictly AFTER train) is what keeps this honest -- no leakage.
    """
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for user, pos_item, cand_items in eval_loader:   # cand: [B, C]
            user = user.to(device)
            pos_item = pos_item.to(device)
            cand_items = cand_items.to(device)
            scores = model.score_candidates(user, cand_items)  # [B, C]
            topk = scores.topk(10, dim=1).indices             # [B, 10]
            # cand_items[:, 0] is the held-out positive by convention
            hits += (topk == 0).any(dim=1).sum().item()
            total += user.size(0)
    return hits / max(total, 1)

def train_one_day(model, train_loader, eval_loader, opt, device,
                  ckpt_dir, warm_start=None, eval_every=2000):
    if warm_start and os.path.exists(warm_start):
        state = torch.load(warm_start, map_location=device)
        model.load_state_dict(state["model"])      # WARM-START from yesterday
        opt.load_state_dict(state["opt"])
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    best_recall, step = 0.0, 0
    for user, item, y, w in train_loader:
        model.train()
        user, item, y, w = (t.to(device) for t in (user, item, y, w))
        logits = model(user, item)
        loss = (loss_fn(logits, y) * w).mean()      # dwell-weighted loss
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % eval_every == 0:
            recall = evaluate(model, eval_loader, device)
            if recall > best_recall:                # EVAL GATE: keep best
                best_recall = recall
                torch.save({"model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "recall@10": recall, "ts": time.time()},
                           os.path.join(ckpt_dir, "best.pt"))
    return best_recall
```

Three things are load-bearing in that loop. First, **warm-start** at the top — load yesterday's checkpoint so today only learns the delta. Second, the **dwell-weighted loss** (`loss_fn(...) * w`) — the labeling stage's weights show up here as the per-example gradient scale, which is the whole reason we computed them. Third, the **eval gate** — we only persist a checkpoint when held-out Recall@10 improves, and a promotion job would refuse to ship a model whose offline Recall@10 regressed against the live one. That gate is what stands between you and shipping a broken model at 3 a.m.

On evaluation honesty: the eval split must be **strictly later in time** than the train split (a temporal split, not a random split), or you leak the future and the gate becomes theater. And beware the [sampled-metrics trap](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys): Krichene and Rendle (KDD 2020) showed that ranking a positive against a small *sampled* candidate set can give a metric that disagrees with the full-catalog metric, even reversing model rankings. For a gate, prefer the full or a large fixed candidate set, and treat sampled-metric deltas with suspicion.

**Export** is the last stage and it splits in two, matching the funnel:

- The **embedding tables** (or the item-tower embeddings for a two-tower model) get written out and loaded into an [ANN index](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) (faiss, ScaNN, HNSW) for the retrieval stage. You export the item embeddings as a matrix, build the index offline, and atomically swap it into serving.
- The **dense ranking model** gets exported to a serving format (TorchScript, ONNX, or a TensorFlow SavedModel) and deployed behind the ranker. The [ONNX deep dive](/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving) covers that path in detail.

The export contract is where train-serve skew comes home to roost: the features computed at serving time must match, bit for bit, the features the model trained on. If the offline feature pipeline computes `user_7d_clicks` one way and the online feature service computes it another, the model sees a different input distribution at serve time than at train time, and precision quietly halves. The defense is a shared feature definition (a feature store, the subject of the large-scale-embedding-and-feature-store post that pairs with this one) so the same code path produces the feature in both worlds.

## 10. Profiling: finding what is actually slow

You will build this pipeline, run it, and find it slower than you hoped. Before you optimize anything, profile, because recsys training has a non-obvious cost structure and the thing you *think* is slow usually is not. The [profiling GPU workloads post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) is the deep reference; here is the recsys-specific checklist.

Wrap a few steps in the PyTorch profiler and look at where wall-clock goes:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=5),
    record_shapes=True, with_stack=True,
) as prof:
    for i, (user, item, y, w) in enumerate(train_loader):
        user, item, y, w = (t.cuda(non_blocking=True) for t in (user, item, y, w))
        logits = model(user, item)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, reduction="none") * w).mean()
        loss.backward(); opt.step(); opt.zero_grad()
        prof.step()
        if i >= 8:
            break
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
```

What you will typically see, and what each finding means:

- **High CPU time, GPU idle between steps → data loader is the bottleneck.** This is the most common recsys finding. The GPU is waiting for batches. Fix: more `num_workers`, higher `prefetch_factor`, a faster on-disk format, columnar reads instead of row-by-row decode, `pin_memory=True` for faster host-to-device copy.
- **`EmbeddingBag` / gather dominating CUDA time → embedding lookup is the bottleneck.** Expected for recsys. Fix where possible: fuse lookups, reduce embedding dimension on tail tables, use mixed precision on the table, or shard better so the all-to-all is balanced.
- **`all_reduce` or `all_to_all` dominating → communication-bound.** The dense all-reduce should be small; if it is large, you may be accidentally syncing the embedding (a sharding-plan bug). The all-to-all grows with batch size and the number of shards — check that the planner balanced your tables.
- **MLP matmuls dominating → you are genuinely compute-bound,** which for a small DLRM MLP almost never happens and means either your MLP is unusually large or everything else is already well-optimized.

The discipline is the same as the rest of the post: do the arithmetic and read the profile before you act. A recsys engineer who "optimizes" the MLP because that is what they learned to optimize for vision will spend a week and gain nothing, while the data loader sits at 60% of wall-clock time, unexamined.

## 11. Putting the numbers together: a results table

Here is a consolidated before-after on a realistic setup, the kind you would put in a design review. The dataset is a feed recommender's interaction logs (numbers are representative orders of magnitude for a small-MLP DLRM-style model; measure your own, but the *directions* and *ratios* are what generalize). The metric is Recall@10 on a temporal eval split; throughput is end-to-end records/sec on the training pipeline; memory is per-host embedding-table memory.

| Configuration | Embedding | Sharding | Per-host mem | Throughput | Recall@10 |
|---|---|---|---|---|---|
| Baseline, stale weekly | 64-dim fp32 | 1 host | 25.6 GB (tight) | 180k rec/s | 0.31 |
| + in-batch negatives | 64-dim fp32 | 1 host | 25.6 GB | 175k rec/s | 0.36 |
| + point-in-time features | 64-dim fp32 | 1 host | 25.6 GB | 170k rec/s | 0.38 |
| + 4-way table sharding | 64-dim fp32 | 4 shards | 6.4 GB | 540k rec/s | 0.38 |
| + daily continuous retrain | 64-dim fp32 | 4 shards | 6.4 GB | 540k rec/s | 0.42 |
| + 128-dim embeddings | 128-dim fp16 | 4 shards | 12.8 GB | 480k rec/s | 0.44 |

![A results matrix of embedding size and sharding configurations against per-host memory and throughput, locating the four-shard 64-dimension configuration as the Pareto sweet spot](/imgs/blogs/building-the-training-pipeline-for-recsys-8.png)

The matrix figure summarizes the throughput-memory Pareto frontier from the same runs. Read the table top to bottom and the lesson is exactly the post's thesis: the biggest single Recall@10 jump (0.38 → 0.42) came not from the model but from **retraining on fresh data daily** instead of weekly. The second-biggest controllable lever was the **negatives** (0.31 → 0.36). Sharding bought a 3× throughput improvement without changing accuracy — its job is to make the *other* improvements affordable at scale, not to lift the metric itself. Widening to 128-dim helped accuracy but cost throughput and doubled memory; whether it is worth it depends on whether you have the host budget. The Pareto sweet spot for most teams is the 4-shard, 64-dim, daily-retrain configuration: it fits comfortably, sustains high throughput, and captures most of the accuracy.

#### Worked example: the daily-retrain ROI

Take the jump from weekly-stale to daily-fresh: Recall@10 went 0.38 → 0.42, a relative lift of about 10.5%. Suppose Recall@10 maps roughly linearly to engagement in your A/B history, and engagement maps to revenue. A 10% relative engagement lift on a surface doing, say, \$50M annualized GMV through recommendations is on the order of a \$5M lift — for the cost of a few extra GPUs running a nightly retrain. Compare that to six weeks of an engineer's time tuning architecture for a 0.005 AUC gain that did not move online at all. This is why the pipeline is the product: the highest-ROI work in recsys is almost always in the data freshness and the negatives, not the model class. (Treat the dollar figure as illustrative — the *ratio* of pipeline-ROI to architecture-ROI is the durable lesson.)

## 12. Case studies and real systems

Four real systems, each illustrating one stage of the pipeline at industrial scale.

**Meta's DLRM and TorchRec.** Naumov et al. (2019, *Deep Learning Recommendation Model for Personalization and Recommendation Systems*) defined the DLRM architecture — sparse embeddings + dense MLP + feature interactions — and is the canonical reference for why recsys training is memory-bound. TorchRec (Ivchenko et al., the PyTorch domain library for recommendation) is the open-source embodiment of the hybrid parallelism in this post: it provides sharded `EmbeddingBagCollection`s, a cost-model-driven sharding planner, `DistributedModelParallel`, and fused optimizers, and it is what powers Meta's production models with embedding tables in the multi-terabyte range. If you are building this for real on PyTorch, TorchRec is the library to reach for rather than hand-rolling the all-to-all.

**The parameter server (Li et al., 2014).** *Scaling Distributed Machine Learning with the Parameter Server* (OSDI 2014) introduced the server/worker split where a sharded set of server nodes holds the parameters and workers push and pull updates asynchronously. This is the intellectual ancestor of every sharded-embedding trainer. Its asynchronous variant tolerates stragglers (a worker can push a slightly stale gradient), which matters when you have thousands of workers and one slow node should not stall the job. Modern collective-communication trainers (synchronous all-to-all) trade that staleness tolerance for cleaner convergence, but the parameter-server pattern persists for tables too large for GPU HBM.

**YouTube's two-tower retrieval (Covington et al., 2016; Yi et al., 2019).** YouTube's deep retrieval system trains a two-tower model with sampled-softmax over the full corpus and exports the item-tower embeddings to a nearest-neighbor index — exactly the export-to-ANN path in section 9. Yi et al. (2019, *Sampled-Softmax with In-Batch Negatives*) describes the in-batch negative + frequency-correction recipe at YouTube scale, which is the negative-generation stage made concrete on a system serving billions of users. The headline operational point: they retrain frequently because the corpus and user behavior drift, which is the continuous-training lesson from section 8 in production.

**Google's continuous training (TFX).** Google has publicly described continuous-training pipelines built on TFX where models retrain on a schedule, validate against the current production model, and promote only if they pass — the eval-gate + redeploy loop from section 9. The operational insight from these descriptions is that the *infrastructure around* training (data validation, the schema check that catches a feature whose distribution shifted, the model-validation gate) is as important as the training itself, because at a daily cadence a silent data bug ships before a human notices.

## 13. When to reach for this (and when not to)

This is heavy machinery. Most of it is overkill for most projects, and the discipline is knowing when you actually need it.

**You do NOT need sharded embedding tables if your vocabulary fits in one GPU.** Run the section-5 arithmetic first. If $|V| \times d \times b$ comfortably fits in a single GPU's memory with room for the optimizer state and activations — say, under ~20 GB of tables on an 80 GB card — then a plain `nn.Embedding` on one device is correct and far simpler. Sharding adds an all-to-all, a planner, and a class of bugs you do not want until the table forces them. A 10M-item catalog at 64-dim is 2.5 GB; do not shard it.

**You do NOT need continuous training if your catalog and users are stable.** A B2B recommender over a slowly-changing product set might be fine retraining weekly or monthly. Continuous training earns its complexity when freshness measurably moves the metric (section 8) — prove that with a stale-vs-fresh A/B before you build the daily pipeline.

**You DO need the point-in-time feature join, always.** This is the one piece of the pipeline that is non-negotiable at every scale, because the leak it prevents is silent and inflates exactly the metric you trust. Even a tiny prototype on MovieLens should use a temporal split and a backward as-of join. Skipping it is the most common way to ship a model that looked great offline and died online.

**You DO need to profile before optimizing, always.** The temptation to optimize the model is strong and almost always wrong for recsys. Profile, find the data loader or the embedding gather, fix *that*. The cheapest large win in this entire post is usually adding data-loader workers.

**Reach for TorchRec (or a parameter server) when the table genuinely does not fit; reach for a single-GPU `nn.EmbeddingBag` until then.** The decision is the memory arithmetic, not the fashion. The [capstone playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) collects these decision rules into one place.

## 14. Key takeaways

- **The pipeline is the product.** A recommender's quality is decided more by the eight-stage data pipeline — labels, point-in-time features, negatives, freshness — than by the model class. The highest-ROI work usually lives in the data, not the architecture.
- **Recsys training is I/O- and memory-bound, not FLOP-bound.** The embedding table dominates memory; the data loader and the gather dominate time. Profile for the loader before you touch the MLP.
- **Embedding memory is $|V| \times d \times b$.** 100M IDs at 64-dim fp32 is 25.6 GB for one table. Do this arithmetic *before* you write training code — it decides your sharding plan and your optimizer.
- **Shard the embeddings, replicate the dense net.** Tables are model-parallel (too big to replicate); the small MLP is data-parallel. All-reduce only the dense gradients; never all-reduce the table — use an all-to-all on the rows the batch touched.
- **Point-in-time correctness is non-negotiable.** Every feature join must look strictly backward in time (a backward as-of join). An offline metric *rising* after a feature "fix" is a red flag for a reintroduced leak.
- **Fresh data beats more data.** Bound the training window and retrain often, warm-starting from the previous checkpoint. Daily continuous training moved Recall@10 more than a month of architecture work in the results table.
- **Draw the feedback loop as an acyclic cadence.** Serve → log → label → warm-start → incremental train → eval gate → redeploy. Each cycle is a fresh forward step, not a closed ring.
- **Gate on a temporal-split metric, and beware sampled metrics.** Eval strictly later in time than train, prefer full or large fixed candidate sets, and never promote a model whose offline metric regressed against the live one.
- **Export is where train-serve skew bites.** The features computed at serving time must match training bit-for-bit; share the feature definition across both worlds.

## 15. Further reading

- Naumov et al., *Deep Learning Recommendation Model for Personalization and Recommendation Systems* (2019) — the canonical DLRM architecture and why recsys is memory-bound.
- Li et al., *Scaling Distributed Machine Learning with the Parameter Server* (OSDI 2014) — the server/worker pattern that underlies every sharded-embedding trainer.
- Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys 2019) — YouTube's in-batch negatives with the frequency correction, at production scale.
- Covington, Adams, Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016) — two-tower retrieval and the export-to-nearest-neighbor path.
- Krichene and Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020) — why sampled offline metrics can disagree with the full-catalog truth, and what to gate on instead.
- TorchRec documentation (pytorch.org/torchrec) — sharded `EmbeddingBagCollection`, the sharding planner, and `DistributedModelParallel`.
- Within this series: the [recommendation funnel map](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), the [data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders) (the feature stage in depth), [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) (the negative-generation stage), [training two-tower models with negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax), the [offline-vs-online gap](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys), and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
- Out of series: [collective communication and NCCL all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch), [profiling GPU workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck), and the [ONNX deep dive on serving formats](/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving).

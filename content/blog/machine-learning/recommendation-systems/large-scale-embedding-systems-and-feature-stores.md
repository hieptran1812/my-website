---
title: "Large-Scale Embedding Systems and Feature Stores"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build the production data and serving plane of a real recommender: a feature store that kills train-serve skew with one transform and point-in-time joins, an embedding service that shards a billion IDs and caches the hot ones, and a serving funnel whose per-stage latencies sum under a 40ms SLA — with runnable Feast-style joins, an LRU embedding cache, and a two-tower precompute-to-ANN serving path."
tags:
  [
    "recommendation-systems",
    "recsys",
    "feature-store",
    "embedding-serving",
    "train-serve-skew",
    "mlops",
    "torchrec",
    "feast",
    "ann-serving",
    "machine-learning",
    "serving-infrastructure",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/large-scale-embedding-systems-and-feature-stores-1.png"
---

You shipped the model. Offline NDCG@10 went from 0.31 to 0.38, the two-tower retrieval pulls clean candidates, the ranker is calibrated, the A/B test is queued. Then someone asks the only question that matters at 200,000 requests per second: *where does the model actually run, and is the model you serve the model you trained?*

That second clause is where recommenders go to die. I have watched a launch where offline metrics were flawless and online CTR dropped 6% on day one — because the `user_clicks_last_7d` feature was computed with a 7-day rolling window in the training pipeline but a 7-*calendar*-day window in the serving path, and the two disagreed by exactly the amount that mattered. I have watched an embedding table for 1.2 billion item IDs OOM a serving host at 03:00 because someone bumped the dimension from 32 to 64 and nobody did the arithmetic ($1.2\text{B} \times 64 \times 4\text{ bytes} = 307\text{ GB}$, which does not fit in a 128 GB box). I have watched a p99 latency budget blow through 40 ms because the feature fetch quietly went from a batched `mget` to 500 individual round-trips. None of these are modeling bugs. They are all *serving plane* bugs, and they are the subject of this post.

This is the systems-and-MLOps chapter of the series. Everything upstream — [the retrieval-ranking-reranking funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking), the [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), the [embeddings](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders) that hold 99% of the parameters — assumes a data and serving plane that *exists*. This post builds it. We will solve the two infrastructure problems that are genuinely unique to recommendation (gigantic embedding tables and consistent features across train and serve), then assemble the serving funnel, wire in real-time and streaming, and put numbers on all of it. The figure below is the whole plane in one frame; we will earn every box.

![Layered diagram of the production recommender serving plane showing request gateway, feature store, embedding service, ANN retrieval, model server, and re-rank stage stacked top to bottom with per-stage latency budgets](/imgs/blogs/large-scale-embedding-systems-and-feature-stores-1.png)

By the end you will be able to: design a feature store that makes train-serve skew structurally impossible; size and shard an embedding service for a billion IDs; compute an embedding-cache hit rate from a Zipfian access pattern and decide how much RAM to spend; lay out a serving funnel whose stage latencies provably sum under an SLA; and sketch every one of these in runnable-ish Python.

## 1. Why a recommender needs a serving plane at all

A classifier is easy to serve. You load 100 MB of weights, you accept a feature vector, you return a probability. One process, one box, done. A recommender is not that, and the reason is the two structural facts that define the whole field.

**Fact one: the model is mostly a lookup table.** In a deep recommender — DLRM, DeepFM, a two-tower — the dense MLP is a few million parameters. The *embedding tables* are hundreds of millions to tens of billions. For a feed with 1 billion items and 500 million users at $d = 64$, the embedding parameters alone are $(1\text{B} + 0.5\text{B}) \times 64 \approx 96\text{B}$ parameters, roughly 384 GB in fp32. No single accelerator holds that. The "model" is a distributed sparse data structure that happens to have a small neural network bolted on top. Serving it is a *storage and lookup* problem first, a compute problem second. (We dug into why the table dominates in [embeddings, the heart of recommenders](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders); here we serve it.)

**Fact two: the features are computed twice, and the two computations must agree.** A recommender's features are not static columns in a row. They are aggregates over event streams — `user_clicks_last_24h`, `item_ctr_7d`, `category_affinity`, `time_since_last_purchase`. During *training* you compute these as of the historical moment each label was generated, over months of logs. During *serving* you compute them right now, in single-digit milliseconds, from live state. If those two computations disagree by even a little, you get [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there): the model learns on one distribution and predicts on another, and your offline metric is measuring a model that does not exist in production.

Put those two facts together and you get a plane of cooperating systems, not a single server. The figure above shows five: a request gateway, a feature store (online side), an embedding service, an ANN retrieval service, a model server, and a re-rank stage. Each is a separate scaling problem with its own latency budget. The discipline of this post is that the *sum* of their latencies must clear the end-to-end SLA, and the *consistency* of the data flowing through them must hold from training to serving.

Here is the framing I use when I onboard someone to a recsys serving stack:

> The model is the cheap part. The two expensive, recommender-specific parts are (1) storing and looking up billions of embeddings under a latency budget, and (2) guaranteeing that a feature means the same thing offline and online. Solve those two and the rest is normal distributed-systems engineering.

The four concerns and their canonical fixes fit in one small grid, which is worth keeping in your head as the map for the rest of the post.

![Matrix mapping four serving-plane concerns — embedding scale, feature consistency, freshness, and latency — each to its problem cell and its solution cell](/imgs/blogs/large-scale-embedding-systems-and-feature-stores-3.png)

Embedding scale → shard, cache, quantize. Feature consistency → one transform behind a feature store. Freshness → streaming materialization. Latency → a per-stage budget. The next sections take them one at a time.

## 2. The feature store: one transform, two paths

Start with the consistency problem because it is the one that silently destroys models, and the one a feature store exists to solve.

### What a feature actually is in a recommender

A *feature* here is a value keyed by an *entity* (a user, an item, a (user, item) pair) and valid as of a *timestamp*. `user_ctr_7d` for `user=42` is not one number — it is a function of time. As of last Tuesday it was 0.11; as of right now it is 0.14. The entity key plus the timestamp uniquely identify the value. This is the single most important idea in the whole post, so let me say it plainly: **a recommender feature is a time series keyed by an entity, and every correct use of it must specify *as of when*.**

Why does that matter? Because training and serving ask for the feature at *different times* and for *different reasons*.

- **Training** wants the value *as of the label's timestamp* — the value the model would have seen if it had been live at that historical moment. If you accidentally use the *current* value (computed over data that includes the future relative to the label), you have leakage: the model trains on information it cannot have at serving time, offline metrics inflate, and online performance collapses. We dissected leakage in [train-serve skew and the bugs that hide there](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there) and in [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders); the fix lives here.
- **Serving** wants the value *as of now*, in single-digit milliseconds, for the entities in the current request.

A feature store is the system that answers both questions from *one feature definition*. You write the transform once; the store materializes it into an *offline store* (a columnar table, often on a warehouse or object storage, indexed by entity and timestamp) for point-in-time training joins, and into an *online store* (a key-value store like Redis, DynamoDB, or Bigtable, keyed by entity) for low-latency serving lookups. Same code, two destinations. That is the entire trick, and it is drawn here.

![Branching dataflow graph showing raw events going through one shared transform that fans out to an offline store for point-in-time training joins and an online store for low-latency serving lookups](/imgs/blogs/large-scale-embedding-systems-and-feature-stores-2.png)

### The point-in-time join (the leakage killer)

The offline side of a feature store is built around the *point-in-time correct join*, also called an *as-of join*. You have a set of training events (labels), each with an entity key and an event timestamp. For each event you want the feature value *as it was at or before that timestamp* — the latest value not in the future.

Naively joining on entity key alone is the classic leakage bug: you pick up the *newest* feature value, which may have been computed after the label. The point-in-time join instead takes, per event, the most recent feature row whose timestamp is $\le$ the event timestamp. Formally, for an event with entity $e$ at time $t$, the joined value is

$$
f^\star(e, t) = f\big(e,\; \max\{\tau \le t : (e, \tau) \in F\}\big),
$$

where $F$ is the materialized feature time series. The $\max\{\tau \le t\}$ is the whole game: it forbids the future. Many stores also enforce a *time-to-live* (TTL), so a feature older than, say, 7 days is treated as missing rather than stale.

Here is a point-in-time join in pandas. It is the same logic Feast runs at scale, just small enough to read:

```python
import pandas as pd

# Training events: one row per (entity, event_time, label).
events = pd.DataFrame({
    "user_id":    [1, 1, 2, 2, 3],
    "event_time": pd.to_datetime([
        "2026-06-01 10:00", "2026-06-05 09:00",
        "2026-06-02 12:00", "2026-06-06 08:00", "2026-06-03 15:00",
    ]),
    "label":      [1, 0, 1, 1, 0],
})

# Feature time series: user_ctr_7d, materialized at various timestamps.
features = pd.DataFrame({
    "user_id":       [1, 1, 1, 2, 2, 3],
    "feature_time":  pd.to_datetime([
        "2026-05-30", "2026-06-03", "2026-06-04",
        "2026-06-01", "2026-06-05", "2026-06-02",
    ]),
    "user_ctr_7d":   [0.10, 0.13, 0.15, 0.08, 0.12, 0.05],
})

# Sort both by time; merge_asof takes the latest feature row <= event_time.
events = events.sort_values("event_time")
features = features.sort_values("feature_time")

joined = pd.merge_asof(
    events,
    features,
    left_on="event_time",
    right_on="feature_time",
    by="user_id",                       # join within each entity
    direction="backward",               # only look at the past
    tolerance=pd.Timedelta(days=7),     # TTL: ignore features older than 7d
)
print(joined[["user_id", "event_time", "user_ctr_7d", "label"]])
```

`merge_asof` with `direction="backward"` is exactly $\max\{\tau \le t\}$, and `tolerance` is the TTL. Notice what happens to user 1's event at 06-01 10:00: it picks up the 05-30 value (0.10), *not* the 06-03 value (0.13), because 06-03 is in the future relative to that label. That single line is the difference between a model that generalizes and one that lied to you in offline eval. Feast's `get_historical_features` and Tecton's offline retrieval both implement this join; you almost never hand-roll it in production, but you must understand it to trust it.

### The online lookup (the latency killer)

The serving side is a key-value read. Given the entities in the request, fetch their *current* feature vectors fast. A Feast-style online API looks like this:

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Serving: get the latest feature values for the request's entities.
feature_vector = store.get_online_features(
    features=[
        "user_features:user_ctr_7d",
        "user_features:clicks_last_24h",
        "item_features:item_ctr_7d",
        "item_features:price_bucket",
    ],
    entity_rows=[
        {"user_id": 42, "item_id": 9981},
        {"user_id": 42, "item_id": 1234},
        # ... one row per candidate item for this user
    ],
).to_dict()
```

Internally this is a batched read from the online store (Redis/Dynamo/Bigtable). The performance rule that bites everyone: **batch the lookup.** If you have 500 candidate items and you do 500 separate `GET`s, you pay 500 network round-trips and your feature fetch alone blows the budget. A single `MGET` over 500 keys is one round-trip. The skeleton of a Redis-backed online store makes the point:

```python
import redis, json
r = redis.Redis(host="localhost", port=6379)

def online_lookup(user_id, item_ids):
    # One MGET, not N GETs. Keys are entity:feature_namespace.
    keys = [f"item:{i}" for i in item_ids] + [f"user:{user_id}"]
    raw = r.mget(keys)                          # single round-trip
    feats = [json.loads(v) if v else {} for v in raw]
    item_feats, user_feats = feats[:-1], feats[-1]
    # Assemble (user, item) feature rows for the ranker.
    return [{**user_feats, **itf} for itf in item_feats]
```

The materialization job (a batch or streaming process) writes `item:{id}` and `user:{id}` keys into Redis on a schedule; the serving path only ever reads. This separation — write offline/streaming, read online — is what keeps the read path predictable.

### Entity keys, feature views, and freshness lag

Three concepts make the store usable in practice, and they are exactly the primitives Feast and Tecton expose.

**Entity keys** are the join keys: `user_id`, `item_id`, and composite keys like `(user_id, category)` for cross features. The store indexes everything by entity key, and the online lookup is a point read on that key. Choosing the right entity granularity is a real design decision — too coarse (one row per user) and you cannot express per-(user, item) features; too fine (one row per (user, item, hour)) and the online store explodes in cardinality. The pragmatic answer is to keep *user-level* and *item-level* feature views separate and join them at request time (the `{**user_feats, **itf}` merge in the snippet above), so you store $|\mathcal{U}| + |\mathcal{I}|$ rows, not $|\mathcal{U}| \times |\mathcal{I}|$.

**Feature views** group features that share an entity key and a materialization schedule. A `user_features` view (entity `user_id`, refreshed hourly) holds `user_ctr_7d`, `clicks_last_24h`, `category_affinity`; an `item_features` view (entity `item_id`, refreshed hourly) holds `item_ctr_7d`, `price_bucket`, `age_days`. Grouping by schedule matters because the materialization job runs per view — a streaming `session_features` view (entity `user_id`, refreshed every 10 seconds) is a *different* job from the hourly `user_features` view even though both key on `user_id`. The serving lookup reads from whichever views the model needs and merges them.

**Freshness lag** is the gap between when an event happens and when its effect is visible in the online store. It is the one place a feature store does *not* perfectly eliminate skew: training joins read the offline store as-of the label, but serving reads the online store as-of *now minus the materialization lag*. If `clicks_last_24h` is materialized hourly, the serving value can be up to an hour stale relative to the live click stream, while the training value was point-in-time exact. The fix is to materialize fast-moving features as streaming (sub-10-second lag, so training and serving see effectively the same value) and to *simulate the serving lag at training time* for features you cannot stream — i.e., compute the training feature as-of `label_time - expected_serving_lag` so the model learns on the staleness it will actually face. This is a subtle, real source of residual skew that teams discover only after the obvious codepath-divergence skew is fixed.

### Why this kills skew

Train-serve skew has one root cause: the feature was computed by *different code* (or the same code over different inputs) in training versus serving. A feature store removes the possibility by construction: there is *one* transform definition, and both the offline materialization and the online materialization run it. The training join reads the offline store; the serving lookup reads the online store; both stores were populated by the same transform. The model trains on exactly the values it will see at inference, modulo freshness lag (which we address in §6). The before-and-after is stark.

![Two-column comparison showing ad-hoc features with two codepaths, future leakage, and skew on the left versus a feature store with one definition, point-in-time joins, and zero skew on the right](/imgs/blogs/large-scale-embedding-systems-and-feature-stores-4.png)

#### Worked example: the skew that halved precision

A real-shape incident. A team had `user_session_length` computed in training as the sum of event gaps over a session (capped at 30 minutes), but the serving path computed it as `now - session_start`, with no cap and no gap-merging. For active users the two agreed; for users who left a tab open, the serving value ran to hours while training never exceeded 30 minutes. The ranker had learned a strong negative coefficient on long sessions (long-tab users rarely click), so at serving time roughly 8% of requests got a feature value the model had *never* seen in training, and those users were systematically down-ranked into noise.

Measured impact: offline AUC was 0.792 (clean, because training was self-consistent). Online CTR on the affected slice dropped from a baseline 4.1% to 2.0% — a **51% relative drop** on 8% of traffic, which dragged the global CTR down 4.3% relative. Moving the transform into a feature store (one capped, gap-merged definition, materialized to both stores) brought the slice back to 4.0% and recovered the global metric within noise. No model change. The "fix" was deleting the second codepath. This is the single highest-ROI infrastructure investment in most recsys stacks, and it is why the feature store exists.

## 3. Sizing the embedding table (the memory wall)

Now the other unique problem: the table itself is enormous. Before you can serve it you must know how big it is, because the arithmetic decides everything downstream — how many shards, how much cache, whether you can afford fp32.

The size is brutally simple:

$$
\text{Memory} = (\,|\mathcal{U}| + |\mathcal{I}| + \textstyle\sum_c |\mathcal{C}_c|\,)\times d \times b,
$$

where $|\mathcal{U}|, |\mathcal{I}|$ are user and item cardinalities, $|\mathcal{C}_c|$ are the cardinalities of categorical features (cross features, device, geo, etc.), $d$ is the embedding dimension, and $b$ is bytes per parameter (4 for fp32, 2 for fp16, 1 for int8). The dense MLP is rounding error next to this.

#### Worked example: when 32→64 OOMs the host

Items $|\mathcal{I}| = 1.2\text{B}$, users $|\mathcal{U}| = 0.4\text{B}$, plus 200M of assorted categoricals. Total rows $\approx 1.8\text{B}$.

- At $d = 32$, fp32: $1.8\text{B} \times 32 \times 4 = 230\text{ GB}$.
- At $d = 64$, fp32: $1.8\text{B} \times 64 \times 4 = 461\text{ GB}$.

A 128 GB host holds neither. At $d = 32$ you need at least two shards (and want headroom, so call it three or four); at $d = 64$ you need four-plus. The "bump $d$ to 64 for a recall gain" decision is not a hyperparameter — it *doubles your serving fleet*. This is why teams reach for the compression tricks from [the embeddings post](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders) (hashing trick, QR embeddings, mixed dimensions) before they reach for more hosts. As a Pareto rule of thumb on MovieLens-20M-scale experiments and at production scale alike: int8 quantization for serving cuts memory $4\times$ for a Recall@10 drop usually under 1% relative; mixed dimensions (small $d$ for cold IDs, large for hot) recover most of a uniform-large-$d$ model at a fraction of the memory. The table below is the menu.

| Technique | Memory effect | Recall@10 effect (typical) | Where it lives | When to use |
|---|---|---|---|---|
| Uniform fp32 | baseline | baseline | training | small tables only |
| fp16 storage | $2\times$ smaller | $\approx$ neutral | train + serve | almost always |
| int8 quantized serving | $4\times$ smaller | $-0.5\%$ to $-1\%$ | serving only | large tables |
| Product quantization (PQ) | $8\times$–$16\times$ smaller | $-1\%$ to $-3\%$ | serving (codes) | billion-scale |
| Hashing trick | bound rows to a budget | $-1\%$ to $-2\%$ | train + serve | unbounded ID space |
| QR / mixed-dim | $2\times$–$4\times$ fewer params | $\approx$ neutral | train + serve | long-tail-heavy |

The memory wall is also a *bandwidth* wall. Even if the table fits, a lookup reads scattered rows from HBM or RAM; the throughput ceiling is set by how many random reads per second the memory subsystem sustains. We will not re-derive the roofline here, but keep it in mind: at billion scale you are usually memory-bandwidth-bound, not FLOP-bound, which is exactly why caching the hot IDs (§5) pays off so much — it converts random reads into local-RAM hits.

## 4. Serving the embedding table: sharding and the parameter server

If the table does not fit on one host, it lives across many. There are two production patterns.

**Pattern A — the distributed embedding table (in-training and TorchRec).** During training, frameworks like Meta's TorchRec and NVIDIA's HugeCTR shard the embedding tables across GPUs. The standard layout is *row-wise* or *table-wise* sharding: each GPU owns a contiguous block of IDs (row-wise) or whole tables (table-wise). A forward pass that needs IDs spread across shards does an *all-to-all* communication: each GPU sends the IDs it needs to the owners, the owners return the embeddings. TorchRec's `EmbeddingBagCollection` plus a `ShardingPlan` automates this; you declare the tables and the planner picks a sharding that balances memory and communication.

```python
import torch
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner

# Declare two big embedding tables (users, items) at d=64.
ebc = EmbeddingBagCollection(
    tables=[
        EmbeddingBagConfig(name="user", embedding_dim=64,
                           num_embeddings=400_000_000, feature_names=["user_id"]),
        EmbeddingBagConfig(name="item", embedding_dim=64,
                           num_embeddings=1_200_000_000, feature_names=["item_id"]),
    ],
    device=torch.device("meta"),   # lazy: planner decides real placement
)

# The planner shards across the available GPUs (row-wise by default at scale).
planner = EmbeddingShardingPlanner()
# In a real job: plan = planner.plan(ebc, topology); model = DistributedModelParallel(ebc, plan=plan)
model = DistributedModelParallel(ebc)   # wraps + shards the tables
```

The win is that an 8-GPU box holds an 8× larger table than one GPU, and the all-to-all overlaps with compute. The cost is communication: at billion scale the all-to-all can dominate step time, which is why TorchRec spends so much effort on fused collectives and why HugeCTR offers a hierarchical parameter server that spills cold IDs to host memory and SSD.

**Pattern B — the parameter server for serving.** At inference you usually do not want GPUs holding the table. You want a sharded, replicated key-value service: the embedding *service*. Each shard owns a hash range of IDs; a lookup hashes the ID to its shard and reads the vector. Shards are replicated for availability and read throughput. The model server (the dense MLP) is a separate fleet that *calls* the embedding service. This decoupling is the standard production layout because the two have different scaling curves: the embedding service scales with table size and read QPS; the model server scales with FLOPs and candidate count.

The layered view of a single lookup — request → cache → quantized codes → sharded table — is the next figure, and it is where caching and compression earn their keep.

![Stacked layers of the embedding serving path showing a lookup request entering a hot ID LRU cache, falling through to product-quantized codes, then to the sharded billion-ID table, and returning a vector](/imgs/blogs/large-scale-embedding-systems-and-feature-stores-6.png)

A clean sharded lookup with consistent hashing looks like this:

```python
import bisect, hashlib

class ShardedEmbeddingService:
    def __init__(self, shards):           # shards: list of backends
        self.shards = shards
        self.ring = sorted(
            (int(hashlib.md5(f"shard{i}".encode()).hexdigest(), 16), i)
            for i in range(len(shards))
        )

    def _shard_for(self, item_id):
        h = int(hashlib.md5(str(item_id).encode()).hexdigest(), 16)
        idx = bisect.bisect(self.ring, (h,)) % len(self.ring)
        return self.ring[idx][1]

    def lookup(self, item_ids):
        # Group IDs by shard, then one batched read per shard (fan-out).
        by_shard = {}
        for iid in item_ids:
            by_shard.setdefault(self._shard_for(iid), []).append(iid)
        out = {}
        for shard_idx, ids in by_shard.items():
            out.update(self.shards[shard_idx].batch_get(ids))   # one RPC/shard
        return [out[i] for i in item_ids]
```

The key engineering choices: **consistent hashing** so adding a shard moves only $1/N$ of the keys; **fan-out by shard** so a 500-ID lookup is at most $N$ RPCs, not 500; and **replication** behind each shard for read throughput and failover. Consistent hashing is the same idea that keeps distributed caches stable when nodes come and go.

### The throughput model: how many shards and replicas

Sizing the service is not guesswork; it falls out of a small throughput model. You need to satisfy a target lookup QPS, and each shard backend sustains some reads per second. The number of *shards* is set by memory (the table must fit across them), and the number of *replicas per shard* is set by throughput (the read load must fit across them):

$$
\#\text{shards} = \left\lceil \frac{\text{table bytes}}{\text{bytes per host}} \right\rceil, \qquad \#\text{replicas} = \left\lceil \frac{\text{peak lookup QPS} \times f_{\text{miss}}}{\text{reads/s per host}} \right\rceil,
$$

where $f_{\text{miss}}$ is the cache miss fraction — the cache (§5) is what makes the replica count tractable, because only the *misses* reach the shards. Concretely: a 461 GB table on 128 GB hosts needs $\lceil 461/100 \rceil = 5$ shards (leaving headroom). If peak is 200k recommendation requests/s, each touching 500 candidate embeddings, that is $10^8$ lookups/s pre-cache. At a 92% cache hit rate, only $8\times10^6$ lookups/s reach the shards; if each host serves $5\times10^5$ reads/s, you need $\lceil 8\times10^6 / 5\times10^5 \rceil = 16$ replicas per shard. Without the cache you would need 200 replicas per shard — a 12× larger fleet. **The cache does not just cut latency; it cuts the shard fleet by an order of magnitude.** That is the throughput argument for §5, and it is usually the dominant cost argument too: serving hosts are the bill, and the cache shrinks it.

One more sizing subtlety: the all-to-all in *training* (TorchRec) and the fan-out in *serving* scale differently. Training communication grows with the number of GPUs (more shards = more all-to-all volume), so there is a sweet spot beyond which adding GPUs slows the step. Serving fan-out grows with the number of shards a single request touches, which is why you want candidates *clustered* across few shards where possible (some systems co-shard related items). Do not assume the training sharding plan is the right serving layout — they optimize different costs.

## 5. The embedding cache: why a tiny cache wins big

Here is the most beautiful result in this whole post, and the one that lets a serving fleet survive a billion-ID table on commodity RAM: **embedding access is Zipfian, so a cache holding a small fraction of the IDs serves the large majority of lookups.**

### The science: Zipfian access and the hit-rate law

Item popularity follows a power law. If you rank items by access frequency, the $k$-th most popular item is accessed with probability proportional to $1/k^s$ for some exponent $s$ (often $s \approx 1$, sometimes higher for very head-heavy feeds). The normalized access probability of rank $k$ over a catalog of $N$ items is

$$
p(k) = \frac{1/k^{s}}{H_{N,s}}, \qquad H_{N,s} = \sum_{j=1}^{N} \frac{1}{j^{s}},
$$

where $H_{N,s}$ is the generalized harmonic number (the normalizer). An LRU cache of capacity $C$ holds, in steady state and to a good approximation under independent reference, the $C$ most popular IDs. Its hit rate is therefore the cumulative access mass of the top $C$ ranks:

$$
\text{hit-rate}(C) \approx \sum_{k=1}^{C} p(k) = \frac{H_{C,s}}{H_{N,s}}.
$$

For $s = 1$, $H_{N,1} \approx \ln N + \gamma$ (with $\gamma \approx 0.577$ the Euler–Mascheroni constant), so the hit rate is approximately

$$
\text{hit-rate}(C) \approx \frac{\ln C + \gamma}{\ln N + \gamma}.
$$

The logarithm is the magic. Because both numerator and denominator are logs, caching a *tiny* fraction of a huge catalog buys a *large* fraction of the hits.

#### Worked example: hit rate for a billion-ID Zipfian feed

Catalog $N = 10^9$, exponent $s = 1$. Cache the hottest $C = 10^7$ IDs — that is **1% of the catalog**.

$$
\text{hit-rate} \approx \frac{\ln(10^7) + 0.577}{\ln(10^9) + 0.577} = \frac{16.12 + 0.577}{20.72 + 0.577} = \frac{16.70}{21.30} \approx 0.784.
$$

So **1% of the IDs in cache serves ~78% of all lookups** under $s=1$. Push the cache to 5% ($C = 5\times10^7$): numerator $\ln(5\times10^7)+0.577 = 17.73+0.577 = 18.31$, hit rate $\approx 18.31/21.30 \approx 0.86$. Real feeds are often *more* head-heavy than $s=1$ (popular content dominates harder), pushing realistic hit rates for a 1% cache into the high-80s to low-90s. That is why I label the cache "hit ~92%" in the diagrams: it is a defensible production figure for a head-heavy feed, not a coincidence.

What does a 92% hit rate buy in latency? If a cache hit costs ~0.1 ms (local RAM) and a miss costs ~12 ms (shard RPC + table read), the expected lookup latency is

$$
\mathbb{E}[\text{latency}] = 0.92 \times 0.1 + 0.08 \times 12 = 0.092 + 0.96 \approx 1.05\text{ ms},
$$

versus 12 ms with no cache — an **11× reduction in mean embedding lookup latency**, and an even larger reduction in load on the sharded table (only 8% of reads reach it). The before/after is worth seeing as a picture.

![Two-column comparison contrasting a no-cache path where every lookup hits the sharded table at high tail latency against an LRU-cache path where the hot one percent of IDs is served from RAM at low latency](/imgs/blogs/large-scale-embedding-systems-and-feature-stores-7.png)

### The code: an LRU embedding cache with measured hit rate

```python
from collections import OrderedDict
import numpy as np

class LRUEmbeddingCache:
    def __init__(self, capacity, backend, dim=64):
        self.capacity = capacity
        self.backend = backend          # the sharded embedding service
        self.dim = dim
        self.store = OrderedDict()       # id -> vector, MRU at the end
        self.hits = 0
        self.misses = 0

    def get(self, item_ids):
        out, missing = {}, []
        for iid in item_ids:
            if iid in self.store:
                self.store.move_to_end(iid)      # mark most-recently-used
                out[iid] = self.store[iid]
                self.hits += 1
            else:
                missing.append(iid)
                self.misses += 1
        if missing:
            fetched = dict(zip(missing, self.backend.lookup(missing)))
            for iid, vec in fetched.items():
                self.store[iid] = vec
                self.store.move_to_end(iid)
                if len(self.store) > self.capacity:
                    self.store.popitem(last=False)  # evict LRU (front)
            out.update(fetched)
        return np.stack([out[i] for i in item_ids])

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
```

And a quick simulation that confirms the law on synthetic Zipfian traffic — the kind of check you run before you size the cache:

```python
import numpy as np

def zipf_ids(n_items, n_requests, s=1.0, seed=0):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_items + 1)
    p = (1.0 / ranks**s); p /= p.sum()
    return rng.choice(ranks, size=n_requests, p=p)

class FakeBackend:
    def lookup(self, ids): return [np.zeros(8) for _ in ids]

N, REQ = 100_000, 2_000_000
for frac in (0.005, 0.01, 0.05):
    cache = LRUEmbeddingCache(capacity=int(N*frac), backend=FakeBackend(), dim=8)
    for iid in zipf_ids(N, REQ, s=1.0):
        cache.get([int(iid)])
    print(f"cache={frac:.1%} of catalog -> hit rate {cache.hit_rate:.3f}")
```

On this $N = 10^5$, $s=1$ traffic you will see roughly: 0.5% → ~0.61, 1% → ~0.66, 5% → ~0.79. The absolute numbers are lower than the billion-scale example because the hit rate is $\ln C / \ln N$ and $\ln(10^5)$ is small, but the *shape* is identical: each cache-size doubling adds a roughly constant chunk of hit rate, and a tiny cache already wins most of the traffic. Run it at $s = 1.2$ and the hit rates jump — head-heaviness is your friend here.

The practical knobs: **cache the embedding *after* quantization** so each cached row is 1 byte/dim, not 4 (a 10M-ID, $d{=}64$ int8 cache is 640 MB, trivially RAM-resident); **warm the cache** on deploy by pre-loading the top-$C$ IDs so you do not eat a cold-start spike; and **co-locate the cache with the model server** so hits never touch the network. The combination of "Zipfian access + cheap quantized cache" is what makes billion-scale embedding serving economical.

## 6. Freshness, materialization, and streaming features

Consistency (§2) says train and serve must agree. *Freshness* says the value must also be recent enough to be useful. These pull against each other and against cost, and the trade-off is one of the defining tensions of a serving plane.

Features split by how fast they must update:

- **Batch features** — recompute daily or hourly. `item_ctr_30d`, `user_category_affinity`. A scheduled job (Spark, a warehouse query) reads the offline store, recomputes, and *materializes* the result into the online store. Staleness of hours is fine; these aggregates barely move.
- **Streaming / near-real-time features** — recompute within seconds. `clicks_last_5m`, `items_viewed_this_session`, `is_trending_now`. A stream processor (Flink, Kafka Streams, Spark Structured Streaming) consumes the event log and updates the online store continuously.
- **Request-time features** — computed in the serving path itself from request context. `hour_of_day`, `device`, `query_length`, `position`. Zero materialization lag; they are pure functions of the request.

The hard part is that a streaming feature must use the *same transform* as its batch backfill, or you reintroduce skew through the back door. This is the "kappa vs lambda" debate in feature engineering: a *lambda* architecture runs a batch path and a streaming path with two implementations (skew risk); a *kappa* architecture runs one streaming implementation and backfills by replaying the log through it (one transform, the feature-store ideal). Tecton and modern Feast lean kappa precisely to preserve the one-transform guarantee.

A streaming materialization, sketched in Spark Structured Streaming with the *same* windowing logic you would use for the batch backfill:

```python
from pyspark.sql import functions as F

# Read the click event stream from Kafka.
clicks = (spark.readStream.format("kafka")
          .option("subscribe", "click-events").load())

events = (clicks.selectExpr("CAST(value AS STRING) as json")
          .select(F.from_json("json", schema).alias("e")).select("e.*"))

# clicks_last_5m: a 5-minute sliding window, same transform used for backfill.
agg = (events
       .withWatermark("event_time", "10 minutes")
       .groupBy(F.window("event_time", "5 minutes", "1 minute"), "user_id")
       .agg(F.count("*").alias("clicks_last_5m")))

# Materialize to the online store (Redis) so serving reads the latest value.
def write_online(batch_df, batch_id):
    rows = batch_df.collect()
    pipe = redis_client.pipeline()
    for row in rows:
        pipe.set(f"user:{row.user_id}:clicks_5m", row.clicks_last_5m, ex=600)
    pipe.execute()

(agg.writeStream.foreachBatch(write_online)
    .outputMode("update").trigger(processingTime="10 seconds").start())
```

The `withWatermark` and `window` definitions are the transform; if the batch backfill uses the same 5-minute window and 10-minute watermark, the streaming and batch values agree, and the offline training join (which reads the batch-materialized history) sees the same feature the online store serves. That is kappa in one snippet.

### The freshness-vs-cost trade-off

Fresher features cost more: tighter triggers mean more compute, more writes, more pressure on the online store. The decision is per-feature, driven by how fast the underlying signal moves:

| Feature class | Update cadence | Staleness tolerance | Cost driver | Example |
|---|---|---|---|---|
| Static / slow | daily batch | hours–days | warehouse compute | item category, base price |
| Medium aggregate | hourly batch | ~1 hour | batch job frequency | item_ctr_7d, user_ctr_7d |
| Session / trending | streaming (seconds) | seconds | stream processor + writes | clicks_last_5m, trending |
| Request-time | per request | none | serving CPU | hour_of_day, position |

The mistake I see most is making *everything* streaming because "fresher is better." It is not better if the signal does not move — you pay stream-processing cost to recompute a 30-day CTR that changes by 0.0001 per minute. Spend freshness where the gradient is steep (session, trending, recent actions for sequence models) and batch the rest.

#### Worked example: the freshness budget for a session feature

A sequential model (SASRec-style, from [self-attention for sequences](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec)) consumes the user's last 50 interactions. If `recent_items` is materialized only hourly, a user who just watched three videos in a session sees recommendations based on stale context — the single highest-signal feature for a returning user is an hour out of date. Making `recent_items` streaming (sub-10-second freshness) is unambiguously worth the cost: in published sequential-rec deployments, real-time session features routinely move online engagement by mid-single-digit percentages, because they capture intent that batch features structurally cannot. Meanwhile `item_ctr_30d` for the same model can stay daily-batch with no measurable loss. Same model, two features, two cadences — that is the freshness budget done right.

## 7. Serving the two-tower: precompute, ANN, and the full path

Now we assemble the retrieval side, where the embedding service and the feature store meet the [ANN index](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann). The two-tower model is the canonical large-scale retriever, and its serving pattern is the reason it scales.

### The precompute trick

A two-tower model has a user tower $f_u$ and an item tower $f_v$; the score is $f_u(x_u)^\top f_v(x_v)$. The crucial property is that the item embedding $f_v(x_v)$ does *not* depend on the user. So you **precompute every item embedding offline**, build an ANN index over them, and at serving time you only run the user tower once and do a nearest-neighbor search. This converts an $O(N)$ scoring problem (score the user against every item) into an $O(\log N)$-ish ANN query. The serving recipe:

1. **Offline (batch, every few hours):** run the item tower over the catalog, write item embeddings to the embedding service, and (re)build the ANN index (Faiss `IndexHNSWFlat` or `IndexIVFPQ`, ScaNN, or hnswlib).
2. **Online (per request):** fetch the user's features from the online store, run the user tower to get the query embedding, query the ANN index for top-$K$ candidates, hand them to ranking.

```python
import numpy as np, faiss, torch

# --- OFFLINE: precompute item embeddings + build the ANN index ---
item_emb = item_tower(catalog_features).detach().numpy().astype("float32")  # (N, d)
faiss.normalize_L2(item_emb)                       # cosine via inner product
index = faiss.IndexHNSWFlat(item_emb.shape[1], 32) # M=32 neighbors/node
index.hnsw.efConstruction = 200
index.add(item_emb)                                # one-time build
faiss.write_index(index, "items.hnsw")             # ship to the retrieval service

# --- ONLINE: user tower -> ANN query -> candidates ---
def retrieve(user_id, k=500):
    feats = online_lookup(user_id, item_ids=[])     # user features from store
    q = user_tower(to_tensor(feats)).detach().numpy().astype("float32")[None]
    faiss.normalize_L2(q)
    index.hnsw.efSearch = 128                        # recall/latency knob
    scores, ids = index.search(q, k)                # top-k in ~single-digit ms
    return ids[0], scores[0]
```

`efSearch` is the recall-vs-latency dial we derived in the [ANN serving post](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann): higher `efSearch` visits more nodes, raises recall, costs latency. The index rebuild cadence is a freshness decision exactly like a feature's: rebuild too rarely and new items are invisible to retrieval (a cold-start problem); rebuild too often and you pay constant index-build cost. Hourly to a few-times-daily is typical, with new items pushed to a small "fresh" index that is merged on rebuild.

### Multi-source retrieval and request fan-out

Real systems do not retrieve from one index. A production feed merges several *candidate sources*, each with its own retriever: the two-tower ANN index (personalized), a "trending now" source (fresh, popular), a "followed creators" source (graph-based), a "similar to recently engaged" source (item-to-item ANN). Each source returns a few hundred candidates; the union (deduplicated) is the candidate set the ranker scores. This is a deliberate design: a single retriever has a single failure mode (a cold-started two-tower returns garbage for new users), and multiple sources give the ranker a diverse, robust pool.

The infrastructure consequence is *request fan-out*: the serving layer calls $k$ candidate sources in parallel, waits for all (or a quorum, with a deadline), deduplicates, and proceeds. Fan-out is where tail latency bites — you wait for the *slowest* source — so each source carries its own timeout, and a slow source is dropped rather than allowed to blow the budget. The pattern:

```python
import concurrent.futures as cf

SOURCES = {                                  # name -> retriever fn
    "two_tower": retrieve_two_tower,
    "trending":  retrieve_trending,
    "i2i":       retrieve_item_to_item,
    "followed":  retrieve_followed,
}

def multi_source_retrieve(user_id, per_source=300, deadline_ms=10):
    cands = {}
    with cf.ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
        futures = {ex.submit(fn, user_id, per_source): name
                   for name, fn in SOURCES.items()}
        for fut in cf.as_completed(futures, timeout=deadline_ms / 1000):
            name = futures[fut]
            try:
                for iid in fut.result():
                    cands.setdefault(iid, set()).add(name)   # track source(s)
            except (cf.TimeoutError, Exception):
                continue                       # drop a slow/failed source
    return cands                               # {item_id: {sources}} deduped
```

Tracking which source produced each candidate matters downstream: the re-rank stage caps per-source counts (`max_per_source=3` in the funnel snippet) so one prolific source cannot dominate the slate, and the source is a useful feature for the ranker itself ("retrieved by the trending source" is signal). The fan-out deadline is part of the latency budget: if retrieval is allotted 8 ms and you fan out to four sources, each source must return in well under 8 ms or be dropped.

### The full serving funnel

Put the pieces together and you get the funnel: request fans out to ANN retrieval and feature fetch, merges at the ranker, narrows through re-rank to a response. This is the infrastructure realization of the retrieval → ranking → re-ranking funnel that is the spine of the whole series.

![Branching serving funnel graph where a request fans out to ANN retrieval and feature fetch, both merging into the ranker, which feeds a re-rank stage and finally the response](/imgs/blogs/large-scale-embedding-systems-and-feature-stores-5.png)

```python
def serve_recommendation(user_id, k_retrieve=500, k_final=10):
    # 1. RETRIEVE: user tower -> ANN -> candidate item IDs.
    cand_ids, _ = retrieve(user_id, k=k_retrieve)

    # 2. FEATURE FETCH: one batched online lookup for all candidates.
    rows = online_lookup(user_id, item_ids=cand_ids.tolist())   # single MGET

    # 3. EMBEDDING FETCH: cached lookup for ranker's sparse features.
    emb = embedding_cache.get(cand_ids.tolist())                # ~92% from RAM

    # 4. RANK: model server scores all candidates in one batched forward.
    scores = ranker.score_batch(rows, emb)                      # GPU batch

    # 5. RE-RANK: diversity / business rules over the top of the ranked list.
    order = rerank_with_diversity(cand_ids, scores, max_per_source=3)
    return order[:k_final]
```

Three production-critical details hide in those five lines. **Batch everything**: one ANN query, one feature `MGET`, one cache `get`, one model forward — never a per-candidate loop. **Fan-out the independent stages**: retrieval and feature fetch for *user* features can start in parallel; only the *candidate* feature fetch must wait for retrieval. **Candidate caching**: for hot users or hot queries, cache the retrieved candidate set for a few seconds so repeated requests skip the ANN query entirely (a second Zipfian win, at the request level).

## 8. The latency budget: the math that keeps you under SLA

The single discipline that makes a serving funnel real is the *latency budget*. The end-to-end SLA (say, p99 $\le$ 40 ms for the recommendation call) is divided into per-stage budgets, and the design constraint is that the stages **sum** under the SLA:

$$
\sum_{\text{stages}} \text{latency}_{\text{stage}}^{(p99)} + \text{overhead} \;\le\; \text{SLA}.
$$

The subtlety, and the reason serving is hard, is that p99 latencies do *not* add the way means do — a chain of stages each at p99 is rarer than any single stage at p99, but tail amplification from fan-out (you wait for the slowest of $N$ parallel calls) pushes the *other* way. The honest engineering rule is to budget each stage at its p99, sum them, and leave 15–25% headroom for network, serialization, GC pauses, and tail amplification. If the naive sum already exceeds the SLA, no amount of luck saves you.

#### Worked example: budgeting retrieval → rank → re-rank under 40 ms

Target: p99 $\le$ 40 ms. Allocate:

| Stage | p99 budget | What it covers |
|---|---|---|
| Feature fetch (user + candidates) | 5 ms | one batched online-store `MGET` |
| ANN retrieval | 8 ms | user tower + HNSW query, top-500 |
| Ranking | 20 ms | model server scores 500 candidates, GPU batch |
| Re-rank | 4 ms | diversity / business rules |
| **Sum** | **37 ms** | |
| Headroom | 3 ms | serialization, network, tail amplification |
| **Total** | **40 ms** | meets SLA |

The arithmetic is unforgiving: 37 ms of work leaves 3 ms of headroom, which is *thin*. If ranking slips to 25 ms (a deeper model, or 1000 candidates instead of 500), the budget breaks and you must either cut $k_{\text{retrieve}}$, distill the ranker (see [distillation and compression](/blog/machine-learning/recommendation-systems/distillation-and-compression-for-recsys)), or batch on the GPU more aggressively. The budget is not a report you write after the fact — it is a *design constraint* you allocate up front and defend in code review. The funnel's stage-by-stage budget and the infra lever for each stage are the last figure.

![Matrix laying out four funnel stages — feature fetch, ANN retrieval, ranking, and re-rank — against their p99 latency budget cells and the infrastructure lever cells that defend each budget](/imgs/blogs/large-scale-embedding-systems-and-feature-stores-8.png)

The levers, stage by stage:

- **Feature fetch (5 ms):** batched `MGET`, co-located online store, connection pooling. The killer here is per-key round-trips; the fix is always batching.
- **ANN retrieval (8 ms):** tune `efSearch`/`nprobe` to the lowest value that meets the recall target; quantize the index (IVFPQ) to fit in RAM; shard the index for throughput.
- **Ranking (20 ms):** batch all candidates into one forward; serve on GPU with a model server (Triton, TorchServe); distill if the budget is tight; cache the embedding lookups (§5).
- **Re-rank (4 ms):** keep diversity logic (MMR, per-source caps) $O(K)$ over the already-ranked list; it touches only the top candidates, so it should be cheap.

The full plane's latency story is just the sum of these, which is exactly what figure 1 promised.

## 9. Online learning, continuous training, and monitoring

The serving plane is not static. The model that is fresh today is stale tomorrow, because the world shifts (new items, new users, seasonal taste) and because the recommender's own outputs reshape the data it next trains on — the [feedback loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) that runs through the whole series. Three mechanisms keep the plane alive.

**Continuous / incremental training.** Rather than retrain from scratch weekly, many production recommenders retrain incrementally: warm-start from yesterday's weights, train on the latest day (or hour) of logs, validate, and deploy. This is fed by the [training pipeline](/blog/machine-learning/recommendation-systems/building-the-training-pipeline-for-recsys): logs → labels → point-in-time feature join → train → eval → deploy. The serve→log→train→deploy chain is a loop in principle, but you build and reason about it as an acyclic pipeline that runs on a schedule, because treating it as a closed ring hides the validation gate that must sit between train and deploy.

**Online learning (true streaming updates).** Some systems update embeddings *continuously* from the live click stream — the item embedding for a freshly trending video should move within minutes, not at the next full retrain. This is incremental embedding update: stream the (user, item, label) events, run a few SGD steps on the affected rows, and push the updated rows to the embedding service. The risk is instability — a noisy hour can corrupt embeddings — so online updates are usually constrained (small learning rate, clipped, validated against a holdout) and reserved for the features where freshness pays (recent items, trending). Incremental embedding update is also how you serve new items without a full rebuild: initialize a new item's embedding from content features, then let online updates refine it as interactions arrive.

**Monitoring.** You cannot defend a serving plane you cannot see. The recsys-specific signals to alert on:

- **Feature drift.** The serving distribution of each feature versus its training distribution (population stability index, KL divergence). A spike means the world shifted or a transform broke — exactly the skew detector from [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there), run continuously.
- **Embedding staleness.** Age of the item embeddings in the ANN index and the embedding service. If the index is 12 hours old, new items are invisible to retrieval.
- **Model staleness.** Age of the deployed model and the gap between its training-data cutoff and now. Plot it; it correlates with slow metric decay.
- **Online↔offline divergence.** The gap between offline predicted CTR and online observed CTR (calibration drift; see [calibration](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust)). A widening gap is the earliest warning that something in the plane changed.
- **Cache hit rate and lookup latency.** A dropping cache hit rate (a content shift made the head less head-heavy) silently raises table load and tail latency before it shows up as an SLA breach.

A minimal feature-drift monitor, the kind you run on a sample of serving traffic:

```python
import numpy as np

def population_stability_index(expected, actual, bins=10):
    # PSI compares a training (expected) feature distribution to serving (actual).
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a_pct = np.histogram(actual,   edges)[0] / len(actual)   + 1e-6
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

# Rule of thumb: PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate (likely skew).
psi = population_stability_index(train_user_ctr, serve_user_ctr)
if psi > 0.25:
    alert(f"feature drift on user_ctr_7d: PSI={psi:.3f} (possible train-serve skew)")
```

PSI under 0.1 is stable, 0.1–0.25 is a watch, above 0.25 means the serving distribution has materially diverged from training — investigate for skew or genuine drift before it shows up as a CTR drop.

### Logging: the feature snapshot that makes everything verifiable

There is one logging discipline that pays for itself a hundred times over: **log the exact feature vector you served, alongside the prediction and the eventual label.** Do not log the entity IDs and recompute features later — recomputing is precisely how skew creeps back in (the recompute uses today's feature values, not the ones served). Log the served vector. This *feature snapshot* is the ground truth that lets you (1) train the next model on exactly what was served (no recompute skew), (2) audit any prediction by replaying its served features through the model, and (3) measure online↔offline divergence by comparing the served prediction to what the offline pipeline produces from the same snapshot. A serving request should emit a log row like:

```python
log_event = {
    "request_id": req_id,
    "user_id": user_id,
    "served_items": [iid for iid in slate],
    "served_features": served_feature_matrix.tolist(),  # the EXACT vectors used
    "served_scores": scores.tolist(),
    "model_version": MODEL_VERSION,
    "served_at": now_iso(),
    # label joined later from the click/conversion stream by request_id + item
}
```

Joining `served_at` plus `request_id` against the downstream click stream produces a perfectly point-in-time-correct training set — the features are the served snapshot, the label is the observed outcome. This closes the serve→log→train→deploy loop *without* recompute skew, and it is the single most important logging decision in a recsys serving plane. (It also feeds [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), which needs the served propensities that only this logging captures.)

#### Worked example: the cost of the cache, in dollars and hosts

Tie the throughput model (§4) to a budget. Peak 200k requests/s, 500 candidates each, billion-ID table at int8 (≈64 GB after quantization), hosts at 100 GB usable and 500k reads/s each.

- **Memory:** 64 GB fits on one host, but for replication and headroom call it 2 shards.
- **Without cache:** $10^8$ lookups/s reach the shards; at 500k reads/s/host that is 200 replicas per shard, ~400 hosts. At a representative \$2/hr/host that is roughly \$19,000/day just for embedding serving.
- **With a 1% int8 cache (640 MB, co-located, ~78% hit at $s{=}1$, ~92% on a head-heavy feed):** only 8% of lookups reach the shards, $8\times10^6$/s, needing 16 replicas/shard, ~32 hosts — about \$1,500/day.

That is a **~12× serving-cost reduction** (≈\$17,500/day saved) from 640 MB of RAM per host. The cache is not an optimization you do later; at billion scale it is the difference between a viable and an unaffordable serving fleet. The arithmetic is the whole argument: spend a little RAM, save an order of magnitude of hosts.

## 10. Case studies: how the big systems actually do this

The patterns above are not invented; they are the distilled practice of the teams that ship recommenders at the largest scale. Four worth knowing, with citations.

**Uber Michelangelo and Palette.** Uber's ML platform Michelangelo introduced *Palette*, an internal feature store, explicitly to solve train-serve consistency across hundreds of models. Palette stores features in both a batch store (Hive) and an online store (Cassandra/Redis), and crucially lets a feature be *defined once* and consumed by both training and serving — the one-transform guarantee that the modern open-source feature stores formalized. Uber's engineering writeups (Hermann and Del Balso, "Meet Michelangelo," 2017) are the canonical origin story for the feature-store pattern; the term "feature store" entered the mainstream from this work.

**Feast.** Feast is the open-source feature store that generalized the Michelangelo pattern. Its architecture is exactly §2: feature definitions in code, an offline store (BigQuery, Snowflake, file/Parquet) for point-in-time historical retrieval via `get_historical_features`, and an online store (Redis, DynamoDB, Datastore) for low-latency `get_online_features`. A *materialization* step moves features from offline to online on a schedule. Tecton (founded by the Michelangelo team) is the commercial evolution, adding managed streaming materialization and the kappa one-transform guarantee. If you want to read one system's docs to understand feature stores, read Feast's — it is the reference implementation of every idea in §2.

**Meta TorchRec and DLRM serving.** Meta's DLRM (Naumov et al., 2019) is the reference deep recommender, and TorchRec is the PyTorch library that makes its enormous embedding tables trainable and servable. TorchRec provides sharded embedding tables (row-wise, table-wise, column-wise, and table-row-wise hybrid sharding), a sharding *planner* that optimizes placement for memory and communication, fused embedding kernels, and quantization for inference. Meta has publicly described embedding tables in the *terabytes* and trillions of parameters, served via sharded parameter servers with caching and int8/int4 quantization — the exact stack of §3–§5. TorchRec is the production answer to "how do you serve a table that does not fit on any host."

**Pinterest and Instagram serving infra.** Pinterest's PinSage (Ying et al., KDD 2018) is a graph-based retriever, but the lesson for *this* post is its serving architecture: item (pin) embeddings are precomputed in a large offline MapReduce/GPU job and served from a key-value store, with ANN over the precomputed embeddings for candidate generation — the precompute-then-ANN pattern of §7 at billion-pin scale. Instagram and Facebook's feed/Explore systems similarly run a two-stage retrieve-then-rank serving funnel with a feature store underneath; their published systems work (e.g., the "Recommending items to more than a billion people" engineering posts) describes the same plane: precomputed embeddings, ANN retrieval, a feature service, and a ranking model server under a tight latency budget.

**YouTube two-tower retrieval.** The YouTube recommendation papers (Covington et al., "Deep Neural Networks for YouTube Recommendations," RecSys 2016, and the later sampled-softmax two-tower work, Yi et al., RecSys 2019) describe the canonical serving split this post is built around: a deep candidate-generation network produces user and video embeddings, video embeddings are precomputed and indexed for ANN nearest-neighbor retrieval, and a separate, heavier ranking network scores the few hundred retrieved candidates with the full feature set. The candidate-generation step is explicitly framed as an approximate nearest-neighbor problem over precomputed embeddings — exactly §7 — because exhaustively scoring a billion videos per request is impossible under the latency budget. YouTube's sampled-softmax training (the $\log Q$ correction we covered in [sampled softmax and contrastive losses](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval)) exists precisely to make this two-tower-for-ANN-serving architecture trainable at that scale.

The through-line: every one of these independently arrived at the same plane — one-transform feature store + sharded/cached embedding service + precompute-and-ANN retrieval + batched model-server ranking under a latency budget. When four teams at four companies converge on the same architecture under different constraints, that architecture is not a fashion; it is the shape the problem forces.

| System | Feature store | Embedding serving | Retrieval |
|---|---|---|---|
| Uber Michelangelo / Palette | Palette (Hive + Cassandra), one definition | model-specific | per-model |
| Feast / Tecton | offline (warehouse) + online (Redis/Dynamo), kappa | external | pairs with ANN |
| Meta TorchRec / DLRM | internal | sharded table + int8/int4 + cache, terabyte-scale | precompute + ANN |
| YouTube two-tower | internal feature pipeline | precomputed video embeddings | candidate-gen via ANN |
| Pinterest PinSage | feature service | precomputed pin embeddings in KV store | ANN over embeddings |

## 11. Building your serving plane (a decision walkthrough)

You will not build all of this on day one, and you should not. Here is the order I build a serving plane in, with the decision at each step.

**Step 1 — Decide if you even need a feature store.** If you have one model, a handful of features, and a small team, a feature store is overhead. A documented, *shared* transform module (imported by both the training job and the serving service) buys you 80% of the consistency guarantee for 5% of the operational cost. Reach for a real feature store (Feast/Tecton) when you have *multiple models sharing features*, *streaming features*, or *point-in-time joins you keep getting wrong*. The smell that says "you need one": a skew incident traced to two diverging codepaths. Until then, the shared-transform module is the right call — don't ship Tecton to serve five features.

**Step 2 — Size the embedding table before you pick hardware.** Run the §3 arithmetic. If the table fits on one host in fp16, you are done — serve it in-process, no embedding service, no shards. The distributed embedding service is a cost you take on *only* when the table exceeds a host; do not build a parameter server for a 10 GB table.

**Step 3 — Add the cache only when the table is sharded.** If lookups are in-process, there is no cache win. Once the table is across the network, an int8 LRU over the hot 1% (sized by the §5 calculation against your actual traffic's Zipf exponent) is the single highest-ROI serving optimization — measure your real hit rate with the simulator before committing RAM.

**Step 4 — Precompute and ANN for retrieval, always.** If you serve a two-tower (or any dot-product retriever), precompute item embeddings and build an ANN index. There is no scenario at scale where scoring the user against the whole catalog at request time is the right answer; the precompute trick is free recall-per-millisecond.

**Step 5 — Budget the funnel, then defend the budget.** Write down the per-stage p99 budget (§8) *before* you optimize. Profile against it. The budget tells you which stage to attack; without it you optimize the stage that is easiest, not the one that is over budget.

**Step 6 — Monitor drift and staleness from day one.** Even a minimal PSI monitor on your top features and an age metric on your model and index will catch the skew and staleness bugs that otherwise take a week of war-room to diagnose.

### Stress tests (the questions that find the cracks)

- **What at 100M items and a 4 ms feature budget?** Your feature fetch is now the bottleneck, not ranking. The fix is fewer, fatter keys (pre-join item features into one value per item so 500 candidates is one `MGET` of 500 fat keys, not 500 × per-feature keys) and aggressive co-location.
- **What if the cache hit rate craters?** A content shift (a viral event flattens the head) drops your hit rate, table load spikes, p99 breaches. Alert on hit rate, not just latency — it is the leading indicator.
- **What if a feature is computed differently offline and online?** This is the skew bug. The feature store makes it structurally impossible; the shared-transform module makes it a code-review catch; nothing else reliably prevents it. If you have neither, you *will* ship this bug.
- **What if the model says one thing offline and another online?** Check calibration drift (online↔offline divergence) first — it is almost always either skew (a feature differs) or staleness (the model trained on a different distribution). The plane's monitors are designed to tell these apart.
- **What if you need new items live in seconds?** Full ANN rebuilds are too slow; push new items to a small fresh index queried in parallel with the main index, and initialize their embeddings from content features (the cold-start bridge from [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem)).

## 12. Results: putting numbers on the plane

A consolidated before→after, the kind I would put in a design doc to justify the investment. These are representative production-scale figures (and consistent with the worked examples above and published systems numbers); treat the exact values as defensible orders of magnitude, not measured constants for your system.

**Serving latency p99 breakdown (funnel), with and without the optimizations:**

| Stage | Naive p99 | Optimized p99 | Lever |
|---|---|---|---|
| Feature fetch | 22 ms (500 GETs) | 5 ms (one MGET) | batch the lookup |
| ANN retrieval | 14 ms (exhaustive-ish) | 8 ms (HNSW, tuned ef) | ANN index + ef tuning |
| Embedding lookup | 12 ms (every miss) | 1 ms (92% cache) | LRU cache over hot IDs |
| Ranking | 28 ms (per-candidate) | 20 ms (GPU batch) | batched GPU forward |
| Re-rank | 4 ms | 4 ms | unchanged |
| **End-to-end p99** | **~80 ms (over SLA)** | **~38 ms (under SLA)** | the whole plane |

**Embedding-cache hit rate vs cache size (Zipfian, $s \approx 1$, $N = 10^9$):**

| Cache size | Fraction of catalog | Hit rate (approx) | Mean lookup latency |
|---|---|---|---|
| 1M IDs | 0.1% | ~0.66 | ~4.1 ms |
| 10M IDs | 1% | ~0.78 | ~2.7 ms |
| 50M IDs | 5% | ~0.86 | ~1.8 ms |
| 100M IDs | 10% | ~0.89 | ~1.4 ms |

(Mean lookup latency uses 0.1 ms hit / 12 ms miss; head-heavier traffic, $s > 1$, pushes every hit rate up.) The marginal return falls off a cliff after ~1%: going from 1% to 10% of the catalog (a 10× RAM spend) buys only 0.78 → 0.89 hit rate. That diminishing return *is* the Zipf log curve, and it tells you to stop at ~1–5% for most feeds.

**Feature store vs ad-hoc (consistency, over a year of operation, representative):**

| Metric | Ad-hoc features | Feature store |
|---|---|---|
| Train-serve skew incidents / year | 6–10 | 0–1 |
| Mean time to add a feature | days (rebuild two paths) | hours (one definition) |
| Feature reuse across models | low (copy-paste) | high (shared registry) |
| Point-in-time correctness | manual, error-prone | enforced by the join |
| Online CTR lost to skew (typical worst incident) | up to ~5% global | ~0 |

The story the tables tell together: batching and caching turn an over-SLA 80 ms funnel into a comfortable 38 ms; a 1% cache wins ~78% of lookups because access is Zipfian; and a feature store converts the chronic, expensive skew-incident drip into a near-zero baseline. None of these required a better *model*. They required a better *plane*.

How to measure this honestly, because the numbers above are only worth what your measurement methodology is worth. For **latency**, always report p99 (and p999 if you have an SLA tail), measured *under load*, after warm-up — a cold cache or an un-warmed JIT reports latency that does not exist in steady state. For the **cache hit rate**, measure it on *replayed production traffic*, not synthetic uniform traffic, because the Zipf exponent of your real traffic is the only thing that matters and it varies by surface (a homepage feed is more head-heavy than a search-results page). For **skew**, the honest test is a shadow comparison: log the served feature snapshot, recompute the same features through the offline pipeline, and diff them per feature — any nonzero diff is skew, and the PSI monitor only catches the *aggregate* version of what the per-request diff catches exactly. And for any **online lift** number, it must come from a controlled A/B test with a fixed-horizon analysis, not a before/after on calendar time, because seasonality and concurrent launches will fool a naive comparison every time. The plane is only as trustworthy as the measurement around it — and a serving plane that you cannot measure is one you cannot defend when CTR drops at 03:00 and the loop's serve→log→train→deploy chain hands you the bill.

## When to reach for this (and when not to)

- **Reach for a feature store** when multiple models share features, when you have streaming features, or when point-in-time correctness keeps biting you. **Don't** stand up Tecton to serve five features for one model — a shared transform module is enough until you feel the pain.
- **Reach for a sharded embedding service** when the table exceeds a single host even in fp16. **Don't** build a parameter server for a table that fits in RAM — serve it in-process and skip the network entirely.
- **Reach for an embedding cache** once lookups cross the network and traffic is Zipfian (it almost always is). **Don't** cache when lookups are already in-process, or when access is near-uniform (rare, but then the cache buys nothing).
- **Reach for precompute + ANN** for any dot-product retriever at scale — it is non-negotiable. **Don't** brute-force score the catalog at request time past a few thousand items.
- **Reach for streaming features** where the signal moves fast (recent actions, trending). **Don't** stream a 30-day aggregate that barely changes — you will pay stream-processing cost for nothing.
- **Reach for online/incremental training** when freshness drives metrics (new items, fast-moving catalogs). **Don't** ship continuous updates without a validation gate and clipping — a noisy hour can corrupt your embeddings.

## Key takeaways

1. **A recommender is a serving plane, not a model.** The two recommender-specific problems — gigantic embedding tables and consistent features across train and serve — define the architecture; the neural net is the cheap part.
2. **One transform, two paths.** A feature store makes train-serve skew structurally impossible by defining a feature once and materializing it to an offline store (point-in-time joins) and an online store (low-latency lookups). Skew is the highest-ROI bug to eliminate.
3. **The point-in-time join is the leakage killer.** Always take the latest feature value *at or before* the label's timestamp ($\max\{\tau \le t\}$) with a TTL; `merge_asof(direction="backward")` is the small-scale version of what Feast runs at scale.
4. **Size the table before you size the fleet.** $(\text{rows}) \times d \times b$ decides shards, cache, and precision. Bumping $d$ from 32 to 64 can double your serving fleet; reach for quantization and mixed-dim before more hosts.
5. **A 1% cache wins ~78% of lookups** because access is Zipfian — hit-rate $\approx \ln C / \ln N$. Cache after quantization, warm on deploy, co-locate with the model server.
6. **Precompute item embeddings and ANN them.** The two-tower's item tower is user-independent, so retrieval becomes a nearest-neighbor query — $O(N)$ scoring collapses to $O(\log N)$.
7. **The funnel must sum under the SLA.** Allocate a per-stage p99 budget up front, leave 15–25% headroom for tail amplification, and defend the budget in code review. Batch every stage.
8. **Spend freshness where the gradient is steep.** Stream session and trending features; batch slow aggregates. Use one transform for batch and streaming (kappa) or you reintroduce skew through the back door.
9. **Monitor drift and staleness, not just latency.** PSI on top features, age on model and index, online↔offline calibration gap, and cache hit rate are the leading indicators that catch skew and staleness before a CTR drop does.
10. **Build incrementally.** Shared transform → feature store; in-process table → sharded service → cache; precompute + ANN; budget; monitor. Each step is a cost you take on only when the previous one's pain is real.

## Further reading

- **Within this series:** [Building the training pipeline for recsys](/blog/machine-learning/recommendation-systems/building-the-training-pipeline-for-recsys) (the loop that feeds this plane), [Embeddings: the heart of recommenders](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders) (why the table dominates and how to compress it), [Train-serve skew and the bugs that hide there](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there) (the consistency failure this post structurally prevents), [Approximate nearest neighbor serving: Faiss, HNSW, ScaNN](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) (the retrieval index this plane queries), and the capstone [The recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
- Hermann, J. and Del Balso, M., "Meet Michelangelo: Uber's Machine Learning Platform" (Uber Engineering, 2017) and "Scaling Machine Learning at Uber with Michelangelo" — the origin of the feature-store pattern and Palette.
- Naumov et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (DLRM, 2019), and the TorchRec library documentation — sharded embedding tables, sharding planners, and quantized serving at terabyte scale.
- Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage, KDD 2018) — precompute-then-ANN serving at billion-item scale.
- Feast documentation (`feast.dev`) — the reference open-source feature store: offline/online stores, `get_historical_features`, `get_online_features`, and materialization. Tecton's docs for the managed, streaming-first evolution.
- Faiss documentation and wiki (`IndexHNSWFlat`, `IndexIVFPQ`) — building and tuning the ANN index that backs retrieval, including the recall-vs-latency knobs and product quantization for memory.

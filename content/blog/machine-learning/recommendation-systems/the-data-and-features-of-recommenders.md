---
title: "The Data and Features of Recommenders: IDs, Context, and Leakage"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A practitioner's guide to the features that actually drive a recommender — ID embeddings, context, and aggregates — and the leakage traps that make offline metrics lie before production exposes the truth."
tags:
  [
    "recommendation-systems",
    "recsys",
    "feature-engineering",
    "embeddings",
    "data-leakage",
    "feature-hashing",
    "train-serve-skew",
    "machine-learning",
    "criteo",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/the-data-and-features-of-recommenders-1.png"
---

A ranking model lands in the offline harness with an AUC of 0.81, up from the incumbent's 0.78. The team is thrilled. It ships behind a 5% A/B test, and within a day online CTR is *down* 2%. Nobody wrote a bug. The model did exactly what the data told it to: one of its strongest features was an item's click-through rate computed over the entire training period, including the days *after* the impression it was trying to predict. Offline, that feature could practically see the answer. Online, at the moment of serving, the future does not exist yet, so the feature collapses to noise. The model's whole apparent edge was leakage.

This post is about the part of a recommender that nobody puts on a slide but that decides whether the slide's numbers are real: the data and the features. We will build the full feature taxonomy — user, item, context, and cross features — and then spend most of our energy on the two things that separate a recommender that ships from one that embarrasses you: **ID embeddings**, which carry the overwhelming majority of the predictive signal, and **leakage**, the family of silent failures that make offline metrics lie. If you have read [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) you already know the retrieval-ranking-reranking funnel and the serve-log-train feedback loop. Features are what flow through that funnel, and the feedback loop is precisely what makes their leakage so dangerous: a leaky feature does not just inflate one offline number, it poisons the logs that train every future model.

By the end you will be able to draw the feature taxonomy from memory, explain why a user-ID embedding beats a stack of demographics, size an embedding table in gigabytes, compute the collision rate of the hashing trick from first principles, name the four leakage paths and how each one inflates offline AUC before collapsing online, and build a point-in-time-correct feature pipeline on a Criteo-style dataset that does not lie to you. The figure below is the map: four feature families, with ID embeddings sitting at the center of the signal.

![Tree diagram of the recommender feature taxonomy showing user, item, context, and cross feature families with ID embeddings highlighted as the dominant signal carriers](/imgs/blogs/the-data-and-features-of-recommenders-1.png)

## 1. The feature taxonomy: four families

Strip a recommender's input down and every feature it consumes belongs to one of four families. Knowing the family tells you the feature's cardinality, its density, how much signal it tends to carry, and — critically — how likely it is to leak. Let us walk them in order.

**User features** describe *who* is asking. The big one is the **user ID** itself, treated as a categorical with millions of values, each mapped to a learned embedding vector. Around it sit demographics (age bucket, country, registered-vs-guest), and — far more valuable than demographics — *behavioral history*: the user's recent clicks, plays, purchases, searches, the categories they linger on, their average session length, their activity level. A user with a thousand interactions is a different prediction problem from a user with three. Behavioral history is often pre-aggregated into a **user embedding** (the average of recently interacted item embeddings, or the output of a sequence model over the last-N items) so the model gets a dense summary of taste without re-reading the whole history at serving time.

**Item features** describe *what* could be shown. Again the headline is the **item ID** and its embedding. Then content attributes: category, brand, language, price bucket, the text of a title or description, an image embedding, tags. And then *item statistics* that summarize how the item has performed: its historical CTR, its number of impressions, its average dwell time, its age (hours since it entered the catalog). Item age matters enormously in feeds and news, where a six-hour-old article and a six-month-old one behave completely differently. Item statistics are also where leakage most loves to hide, because they are aggregates over interaction data and it is alarmingly easy to aggregate over the wrong time window.

**Context features** describe *the moment*. Time of day and day of week (people watch different things at 8am on a Tuesday and 11pm on a Saturday), device (phone vs TV vs desktop changes both intent and the slate size), location, network type, the search query if there is one, and — easy to forget but high-signal — the **position in the session**: is this the first item of a fresh session or the fortieth scroll? Position-in-session correlates with fatigue and with the kind of content that keeps someone going. Context features are usually low-cardinality and dense, which makes them cheap, but they are a classic source of *train-serve skew* because the way you compute "local hour" in a batch job and the way the serving system computes it from a request timestamp are rarely identical.

**Cross features** (also called interaction features) describe the *match* between this user and this item. The most important is implicit: the dot product of the user embedding and the item embedding, which a two-tower model computes directly. But explicit crosses matter too — "has this user interacted with this item's category before," "how many of this brand has the user bought," "the user's average rating for this genre." Crosses are where memorization lives. A linear model cannot learn that *this* user likes *this* category unless you hand it the cross; the famous Wide and Deep architecture exists precisely to let a wide linear part memorize sparse crosses while a deep part generalizes. We will preview that in the case studies.

Two orthogonal axes cut across all four families and you should internalize them because they decide how a feature is *encoded*:

- **Sparse categorical vs dense numeric.** A categorical (user ID, category, device) takes one of a discrete, often huge set of values and is encoded as an embedding lookup. A numeric (price, age in hours, CTR, count) is a real number and is encoded by normalization. Most of a recommender's *parameters* live in the categorical embedding tables; most of its *feature columns* are categorical too. The Criteo display-ads benchmark, the standard CTR dataset, has 13 numeric features and 26 categorical features — and the 26 categoricals contain tens of millions of distinct values between them, dwarfing the numerics in parameter count.
- **High cardinality vs low cardinality.** Device has maybe five values; user ID has hundreds of millions. The encoding strategy diverges sharply at scale: low-cardinality categoricals get a full embedding row each, while high-cardinality ones force you into the **hashing trick** or frequency thresholds, which we cover in section 3.

It helps to put a concrete row on the table. A single training example for a feed ranker might look like this before encoding: user ID `u_8831204`, item ID `v_55120098`, user's last-5 video IDs `[v_91, v_4421, v_77, v_5500, v_12]`, item category `cooking`, item age `4.2 hours`, item's 7-day CTR `0.031`, device `android_phone`, local hour `12`, day of week `tuesday`, session position `3`, user's 30-day activity `41 interactions`, label `clicked = 1`. Every field on that row maps to a family: the two IDs and the history sequence are user/item; category, age, and CTR are item; device, hour, day, and position are context. After encoding, the two IDs become two `d`-dim vectors, the category becomes another small embedding, the device/hour/day become tiny embeddings, and age/CTR/activity become normalized scalars. The model never sees `cooking` or `android_phone` as strings; it sees rows of a table, gathered by integer index. That gather-by-index is the recommender's defining computational primitive, and it is why a recommender's parameter count is dominated by tables, not by the dense network that consumes them.

One more distinction deserves emphasis because it changes how a feature behaves under sparsity. A **categorical that recurs** — the same user ID appearing in a thousand examples — accumulates a thousand gradient updates to its embedding row, so that row is well-estimated. A **categorical seen once** gets a single update and its row stays close to its random initialization; it carries no learned signal. This is the heart of the high-cardinality problem: it is not that there are many values, it is that *most of them are rare*, so most embedding rows never get enough signal to be worth their memory. The whole machinery of frequency thresholds, hashing, and OOV buckets in section 3 exists to spend embedding capacity only where there is enough data to learn something.

Hold this taxonomy. The rest of the post is about turning these four families into a vector a model can train on — without lying to yourself about how well it works.

## 2. From raw events to a feature vector

Before any of these features exist, you have logs. A recommender's raw data is a stream of *events*: impressions (item X was shown to user U at time T in context C), and interactions (user U clicked / played / purchased item X at time T). The feature pipeline's job is to turn that stream into a fixed-width vector per training example, with categoricals encoded as embeddings and numerics normalized. The figure below traces that flow.

![Stack diagram showing raw click and impression logs flowing into split categoricals and numerics, then embedding lookup, hashing, and normalization, merging into a concatenated feature vector fed to a ranking model](/imgs/blogs/the-data-and-features-of-recommenders-2.png)

The flow has five stops. First, **events are joined into labeled examples**: each impression becomes a row, and its label is whether an interaction followed within an attribution window (say, a click within 30 minutes). Second, the row's features **split by type**: categorical columns route to encoders, numeric columns to scalers. Third, **categoricals become vectors** — low-cardinality ones via a full embedding table, high-cardinality ones via the hashing trick into a smaller table. Fourth, **numerics are normalized** — almost always with a `log1p` first (counts and prices are heavy-tailed, and a raw count of 10,000 next to a count of 2 will dominate any gradient) and then a z-score or min-max. Fifth, **everything is concatenated** into one wide vector and handed to the model.

The very first stop — turning events into labeled examples — hides a decision that quietly shapes everything downstream: **what counts as a positive, and over what window.** A click is the obvious positive, but a click followed by an immediate bounce is a worse signal than a click followed by a five-minute watch. Many production systems learn against a *weighted* or *multi-task* label (predict click, and separately predict post-click dwell), but even a single-task CTR model has to choose an **attribution window**: a click within 30 minutes of the impression is a positive, anything later is not attributed to this impression. Set the window too short and you label genuine engagement as negative; set it too long and you attribute to this impression a click the user made for an unrelated reason. The window also interacts with leakage: if your attribution window is 30 minutes but you snapshot features at impression time, you are fine; if any feature is recomputed at *labeling* time (after the window closes), it has seen 30 minutes of future. The discipline is to fix two timestamps per example — the **impression time** (when features are snapshotted) and the **label time** (when the outcome is resolved) — and to guarantee that every feature reads only data before the impression time, never the label time. Getting this boundary explicit on every row is half the battle against leakage, because most leaks are exactly a feature accidentally reading from the label-time side of that boundary.

Two design decisions in this pipeline cause most production pain, and both are about *consistency*. The first is that the exact same transformation code must run offline (over a historical batch) and online (over a single request) — otherwise you get train-serve skew, the subject of section 7. The second is that every aggregate (every CTR, every count) must be computed using only information available *at or before* the example's timestamp — otherwise you get temporal leakage, the subject of section 6. The pipeline diagram looks innocent; the timestamps flowing through it are where careers are made and lost.

Here is the skeleton of such a pipeline in PyTorch and pandas, which we will fill in across the rest of the post. We use a Criteo-style schema: numeric features `I1..I13`, categorical features `C1..C26`, label `y`.

```python
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

NUM_COLS = [f"I{i}" for i in range(1, 14)]      # 13 dense numerics
CAT_COLS = [f"C{i}" for i in range(1, 27)]      # 26 sparse categoricals

def preprocess_numerics(df):
    # heavy-tailed counts: log1p then standardize. Criteo's classic trick
    # is log(x)^2 for x > 2; log1p is a clean, robust default.
    x = df[NUM_COLS].fillna(0.0).clip(lower=0).values.astype("float32")
    x = np.log1p(x)
    mu, sigma = x.mean(0), x.std(0) + 1e-6
    return (x - mu) / sigma, (mu, sigma)

class CriteoDataset(Dataset):
    def __init__(self, df, cat_index, num_stats):
        self.num = self._apply_num(df, num_stats)
        self.cat = self._apply_cat(df, cat_index)      # int64 indices
        self.y = df["y"].values.astype("float32")

    def _apply_num(self, df, stats):
        mu, sigma = stats
        x = np.log1p(df[NUM_COLS].fillna(0.0).clip(lower=0).values.astype("float32"))
        return ((x - mu) / sigma).astype("float32")

    def _apply_cat(self, df, cat_index):
        # cat_index maps each column's raw value -> integer id (built later)
        cols = [df[c].map(cat_index[c]).fillna(0).astype("int64").values
                for c in CAT_COLS]
        return np.stack(cols, axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.num[i]),
                torch.from_numpy(self.cat[i]),
                torch.tensor(self.y[i]))
```

Notice what is *not* in here yet: the categorical-to-integer mapping (`cat_index`), the embedding tables, the hashing for high-cardinality columns, and — most importantly — any aggregate feature. We add those deliberately, because each is a place leakage can enter. We start with the embeddings, because that is where the signal is.

## 3. The dominance of ID embeddings

Here is the single most counterintuitive fact about recommender features, and the one that most reliably surprises engineers coming from tabular ML: in a mature recommender, the **user-ID and item-ID embeddings carry most of the signal**, often dramatically more than every hand-crafted feature combined. You can ablate away demographics, content categories, even most context, and lose a point or two of AUC. Ablate the ID embeddings and the model is barely better than popularity. Why?

Because the ID embedding is a *learned, per-entity free parameter*. It is not a description of the user; it is whatever vector best predicts that user's behavior, discovered by gradient descent. Demographics say "25-34, US, Android." The ID embedding says "this specific person who, empirically, watches cooking videos at lunch and finishes thrillers on weekends." No demographic bucket captures individual taste; the ID embedding *is* individual taste, compressed to `d` numbers. The same holds for items: the item-ID embedding learns latent properties ("this is the kind of thing people who like X also like") that no content tag enumerates.

### What an embedding lookup actually is

It is worth being precise, because the mechanics explain both the power and the cost. An embedding lookup is *exactly* a matrix multiply of a one-hot vector by the embedding table. Let the vocabulary have $|V|$ entries and the embedding dimension be $d$. The table is a matrix $E \in \mathbb{R}^{|V| \times d}$. To embed entity $i$, form the one-hot vector $\mathbf{x}_i \in \{0,1\}^{|V|}$ with a single 1 in position $i$, and compute

$$
\mathbf{e}_i = \mathbf{x}_i^\top E
$$

which selects row $i$ of $E$. The figure below shows this concretely: the one-hot vector for item 2 picks out exactly row 2 of the table.

![Grid diagram showing a one-hot column vector for item 2 selecting a single row of an embedding table, illustrating that an embedding lookup equals one-hot times table](/imgs/blogs/the-data-and-features-of-recommenders-3.png)

In practice you never materialize the one-hot vector — that would be `$|V|$`-wide and almost all zeros. You store integer indices and gather rows directly; `torch.nn.Embedding` does exactly this `gather`, and its backward pass scatters gradients back only to the rows that were touched. But mathematically it is one-hot times table, and that framing matters because it explains why the embedding is a *linear* readout of identity: the model has a private, trainable slot for each entity, and the only thing tying entities together is the shared downstream network. For a deeper treatment of why these vectors organize themselves into meaningful geometry, see [embeddings, the heart of recommenders](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders).

### The memory cost

The power comes at a price you must be able to compute on a whiteboard. The table is $|V| \times d$ floats. At 4 bytes per float (fp32), the memory is

$$
M = |V| \cdot d \cdot 4 \ \text{bytes}.
$$

This is linear in both vocabulary size and dimension, and it is the dominant memory cost of an industrial recommender. Dense layers might be a few million parameters; embedding tables are billions.

#### Worked example: sizing the embedding table

Take a video platform with 200 million users and 50 million items, and pick $d = 64$. The user table is $2{\times}10^8 \cdot 64 \cdot 4 = 51.2$ GB. The item table is $5{\times}10^7 \cdot 64 \cdot 4 = 12.8$ GB. Together, 64 GB — for the embeddings alone, before optimizer state. With Adam you carry two extra moments per parameter, so the *training* footprint triples to roughly 192 GB, which already exceeds a single 80 GB GPU and forces sharding across hosts (this is exactly why systems like the open-source DLRM and FBGEMM exist: to shard and quantize these tables). Halve $d$ to 32 and you halve the table to 32 GB; this is the lever you reach for first when the table OOMs a host. The trade-off — how much AUC you give up by shrinking $d$ — is the subject of section 8.

### The high-cardinality problem and the hashing trick

A user-ID vocabulary of 200 million is large but bounded. The real trouble is *unbounded, high-churn* categoricals: a hashed cookie ID, an ad creative ID, a search query, a URL. These can run to billions, grow every second, and have a long tail of values seen exactly once. You cannot allocate a row per value — the table would be larger than the dataset and most rows would receive a single gradient step, learning nothing.

The standard fix is the **hashing trick** (feature hashing). Instead of a dictionary from value to index, you apply a hash function and take it modulo a fixed table size $m$:

```python
def hash_index(value: str, m: int) -> int:
    # deterministic, language-agnostic; mmh3 or built-in hash with a seed
    import hashlib
    h = int(hashlib.md5(value.encode()).hexdigest(), 16)
    return h % m
```

Now the table has a fixed $m$ rows no matter how many distinct values arrive. The cost is **collisions**: two different values hash to the same row and are forced to share an embedding. A small collision rate is usually harmless (rare values share a row and the model treats them as a fuzzy bucket), but a high collision rate is destructive — popular values collide and the model cannot distinguish them. So you need to be able to *predict* the collision rate, which is pure probability.

#### Worked example: computing the hashing collision rate

Suppose you have $n = 10^7$ distinct values and a table of size $m = 10^6$ (a 10:1 load factor). For a single value, the probability that it lands in a row already occupied by at least one other value is governed by the same math as the birthday problem. The probability that a *specific other* value avoids a given row is $1 - 1/m$. The probability that *all* $n-1$ other values avoid it is $(1 - 1/m)^{n-1}$, so the probability that the row collides is

$$
P(\text{collision for one value}) = 1 - \left(1 - \frac{1}{m}\right)^{n-1}.
$$

Plug in: $(1 - 10^{-6})^{10^7 - 1} \approx e^{-10^7/10^6} = e^{-10} \approx 4.5\times10^{-5}$, so the collision probability is about $1 - 4.5\times10^{-5} \approx 0.99995$. Almost every value collides! That is the wrong way to read it, though — at a 10:1 load factor *of course* rows are shared, on average ten values per row. The number you actually care about is the *expected number of collisions per row*, which is just the load factor $n/m = 10$. If instead you size $m = 10^8$ for $n = 10^7$ (a 0.1 load factor), the expected occupants per row is 0.1, and the probability a given value shares its row with any other is $1 - e^{-(n-1)/m} \approx 1 - e^{-0.1} \approx 0.095$, under 10%. The lesson: **collisions are a function of the load factor $n/m$, not of $n$ alone.** Size your hash table to the load factor you can tolerate. A common production rule of thumb is to keep the load factor at or below 1, then accept that the long tail shares rows (which is exactly what you want — rare values *should* be fuzzy).

A more refined approach mixes strategies. Keep a **full embedding** for the high-frequency head (the few thousand or million IDs that appear often enough to learn a dedicated vector) and **hash the tail** into a shared table. You decide head-vs-tail with a **frequency threshold**: any value seen fewer than, say, 10 times in the training window goes to a small set of **out-of-vocabulary (OOV) buckets** (often hashed into 1000-ish buckets so the tail still has some granularity). This bounds the table to (head size + OOV buckets) and guarantees that every dedicated row gets enough gradient signal to be meaningful. Here is the index-building step that realizes that policy:

```python
from collections import Counter

def build_cat_index(df, cat_cols, min_freq=10, n_oov=1000):
    """Map each column's value -> int id. Head gets dedicated ids;
    rare values fall into hashed OOV buckets. Reserve 0 for unseen."""
    cat_index = {}
    for c in cat_cols:
        counts = Counter(df[c].dropna())
        head = {v for v, k in counts.items() if k >= min_freq}
        mapping = {}
        next_id = 1 + n_oov                       # 0 = padding, 1..n_oov = OOV
        for v in sorted(head):
            mapping[v] = next_id
            next_id += 1
        # closure: rare or unseen -> deterministic OOV bucket
        def make_lookup(m, n_oov=n_oov):
            def lookup(v):
                if v in m:
                    return m[v]
                if pd.isna(v):
                    return 0
                return 1 + (hash(str(v)) % n_oov)
            return lookup
        cat_index[c] = make_lookup(mapping)
    return cat_index
```

Now the model. A `DeepFM`-style or plain MLP ranker that consumes the 13 numerics and 26 categoricals looks like this — note the per-column `nn.Embedding`, sized to each column's vocabulary, and the `EmbeddingBag` option for multi-valued fields like "user's last-N categories":

```python
import torch.nn as nn

class CTRRanker(nn.Module):
    def __init__(self, cat_cardinalities, d=32, n_num=13, hidden=(256, 128)):
        super().__init__()
        # one embedding table per categorical column, sized to its vocab
        self.embeds = nn.ModuleList(
            [nn.Embedding(card, d, padding_idx=0) for card in cat_cardinalities]
        )
        in_dim = n_num + d * len(cat_cardinalities)
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, num, cat):           # num: [B,13], cat: [B,26]
        emb = [self.embeds[j](cat[:, j]) for j in range(cat.size(1))]
        x = torch.cat([num] + emb, dim=1)  # [B, 13 + 26*d]
        return self.mlp(x).squeeze(-1)     # logits
```

The model is almost boilerplate. The signal lives in those embedding tables, and the danger lives in how we built the *features* that feed them.

### When the ID has no signal: cold start and content features

The dominance of ID embeddings has a sharp limit: an ID embedding is only as good as the gradients it has received, and a brand-new item or a brand-new user has received *zero*. A fresh item's ID row is still at its random initialization, so a model that relies on it will rank that item essentially at random — the cold-start problem, restated as a feature problem. This is precisely where **content features** earn their keep. A text embedding of the item's title (from `sentence-transformers`), an image embedding of its thumbnail, its category and brand — none of these require any interaction history, so they give the model a way to place a never-seen item near similar seen items in feature space. The practical recipe is to feed *both* the ID embedding and the content embedding, and let the model learn to weight them: lean on content while the ID row is untrained, lean on the ID once it has accumulated signal. This is also why content embeddings score "medium signal" but "helps cold-start" in the priority table later — for warm items they add little over the ID, but for the cold items they are the only signal there is. The series covers cold-start strategies in depth on their own; here the point is narrow: content features exist to cover the gap where ID embeddings have no data.

Now, time to confront the danger that hides in the engineered features directly.

## 4. Feature engineering for rec: counts, time, and sequences

Beyond raw IDs, the most useful engineered features in a recommender fall into three groups, and each carries a characteristic leakage hazard. Let us build them, and flag the hazard each time.

**Count and statistical features** summarize behavior with an aggregate: an item's CTR (clicks / impressions), an item's total impressions, a user's activity level (interactions in the last 7 days), a category's average dwell time. These are enormously useful — an item's empirical CTR is one of the single strongest features in display advertising — and enormously dangerous, because *every aggregate is computed over a set of rows, and if that set includes future rows or the current label row, it leaks.* We will dwell on this in sections 5 and 6.

There is also a quieter statistical hazard in count features that has nothing to do with leakage: **the bias-variance trade-off of a small-sample rate.** An item's empirical CTR is an estimate of its true click probability $p$ from $n$ Bernoulli trials. The variance of that estimate is $p(1-p)/n$, which blows up for small $n$. A brand-new item with 3 impressions and 1 click has an empirical CTR of $0.333$ with a standard error near $0.27$ — essentially noise. If you feed that raw $0.333$ to the model, you are feeding it a number that is wrong by a wide margin for exactly the items (new ones) where the prediction matters most. The fix is the same empirical-Bayes shrinkage as target encoding: pull the estimate toward a prior (the global or category mean) by an amount that decreases as $n$ grows. A count feature without smoothing is a high-variance feature that the model will either over-trust (and mispredict cold items) or learn to ignore (wasting the signal on warm items). Always smooth, and consider passing the *count itself* ($n$, log-scaled) as a companion feature so the model can learn how much to trust the rate.

**Time and recency features** capture *when*. The most important is **recency**: time since the user's last interaction, time since the item was created (item age), time since the user last saw this category. Recency is high-signal because freshness drives engagement in almost every feed. It is also the feature most often computed *wrong* with respect to the prediction time: "time since last interaction" must be measured relative to the impression timestamp, not relative to the data export time. If you compute `now() - last_interaction` in a batch job at export time, every row's recency is wrong by the gap between the impression and the export, and that error is *systematically different* offline and online.

**Sequence features** capture the user's *recent trajectory*: the last $N$ items they interacted with, as an ordered list of IDs (later pooled or fed to a sequence model like SASRec). Sequences are the richest behavioral signal and the backbone of modern session-based recommenders. The leakage trap here is subtle and lethal: the "last $N$ items" must be the last $N$ *strictly before* the current impression. It is shockingly easy to build the sequence by sorting a user's interactions and, for each one, taking the surrounding window — which includes the target itself and items after it. That is target leakage dressed up as a sequence.

The leak-free way to build sequence features is to walk each user's events in time order and, for each event, snapshot only the prefix that came before it. The code below does exactly that, producing a left-padded fixed-length history per row with no peeking forward:

```python
def build_history(df, user="user_id", item="item_id", ts="ts", N=20):
    """For each row, the last N item ids the user interacted with strictly
    before this row's timestamp. Left-padded with 0 (the padding id)."""
    df = df.sort_values([user, ts]).reset_index(drop=True)
    histories = []
    for _, grp in df.groupby(user, sort=False):
        seq = []                                  # grows as we walk forward
        for it in grp[item].tolist():
            window = seq[-N:]                     # strictly-before prefix
            padded = [0] * (N - len(window)) + window
            histories.append(padded)
            seq.append(it)                        # only AFTER snapshotting
        # note: the current item is appended only after its row is recorded,
        # so no row ever sees its own target in its history.
    return np.array(histories, dtype="int64")
```

The single load-bearing line is `seq.append(it)` happening *after* `histories.append(padded)`. Swap those two lines and every row's history contains the item the model is trying to predict — a flawless leak that pushes offline AUC toward 1.0 and is worthless online. The fixed-length left-padded array drops straight into an `nn.EmbeddingBag` (mean-pool the history embeddings) or a sequence model. This is the standard preprocessing behind SASRec and BERT4Rec, and the strict-prefix discipline is exactly what their published training setups enforce.

### Target encoding and its leakage danger

There is one feature-engineering technique so prone to leakage that it deserves its own treatment: **target encoding** (also called mean encoding or likelihood encoding). The idea is seductive: replace a high-cardinality categorical with the *mean of the label* for that category. So instead of embedding "device = Pixel 7," you replace it with "the average CTR of all Pixel 7 impressions." For a categorical with thousands of values and limited data, this can be a powerful, compact feature.

It is also a leakage machine, for two compounding reasons. First, **the naive computation includes the current row's own label in the mean.** If "category Z" appears 5 times with labels $\{1,0,1,1,0\}$, the mean is 0.6, but for the first row, the mean *that includes its own label* tells the model 60% of the answer for that row. With rare categories this is catastrophic: a category that appears once gets a target encoding exactly equal to its own label — a perfect leak. Second, even excluding the current row, **if you compute the encoding over the whole dataset and then split into train and test, the test rows informed the encoding**, which is temporal leakage.

The math makes the bias precise. Suppose a category appears $k$ times with true rate $p$. The naive leave-self-in mean for a given row is

$$
\hat{p}_{\text{leak}} = \frac{1}{k}\left( y_i + \sum_{j \neq i} y_j \right) = \frac{y_i}{k} + \frac{k-1}{k}\,\bar{y}_{-i}
$$

which is an explicit linear function of the row's own label $y_i$. For $k = 1$ it equals $y_i$ exactly; the feature *is* the label. The bias-variance read is also clean: target encoding trades the *variance* of a sparse one-hot (rare category, few examples) for the *bias* of a smoothed estimate, and the smoothing parameter controls the trade. A common smoothing is the empirical-Bayes shrink toward the global mean $\mu$:

$$
\hat{p} = \frac{n_c \, \bar{y}_c + \alpha\, \mu}{n_c + \alpha},
$$

where $n_c$ is the count for category $c$, $\bar{y}_c$ its mean label, and $\alpha$ a pseudo-count. As $n_c \to \infty$ the estimate trusts the category; as $n_c \to 0$ it falls back to the global mean, which is exactly the bias-variance dial you want.

The correct, leak-free way to compute target encoding is **out-of-fold (OOF)**: split the training data into $K$ folds, and for each fold compute the encoding using *only the other folds*. No row ever sees its own label. At inference (and for the test set), you use the encoding computed over the *entire* training set. Here is the OOF target encoder, with smoothing:

```python
from sklearn.model_selection import KFold

def oof_target_encode(df, col, target="y", n_splits=5, alpha=20.0, seed=0):
    """Leak-free out-of-fold target encoding with empirical-Bayes smoothing."""
    global_mean = df[target].mean()
    oof = np.full(len(df), global_mean, dtype="float32")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(df):
        tr = df.iloc[tr_idx]
        stats = tr.groupby(col)[target].agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)
        oof[val_idx] = df.iloc[val_idx][col].map(enc).fillna(global_mean).values
    # full-data encoding to apply at inference / on the test split
    full = df.groupby(col)[target].agg(["mean", "count"])
    full_enc = (full["mean"] * full["count"] + global_mean * alpha) / (full["count"] + alpha)
    return oof, full_enc.to_dict(), global_mean
```

This is correct for a *random* split. But a recommender is not a random-split problem — it is a *temporal* problem, and the OOF fold trick is necessary but not sufficient. To see why, we have to confront leakage head-on.

## 5. The leakage traps unique to rec

This is the section that justifies the post. Leakage is when information that would not be available at prediction time sneaks into the features, letting the model "cheat" offline. It is the single most common reason a recommender that looked great offline fails online, and recommenders are *unusually* prone to it because so many of their best features are aggregates over interaction data, and interaction data is timestamped, label-bearing, and re-logged in a feedback loop. There are four distinct paths, and the figure below lays them out converging on the model.

![Graph diagram showing four leakage paths — future information, aggregates including the label row, train-serve skew, and post-impression signals — converging on the training features and inflating the model AUC before the online metric collapses](/imgs/blogs/the-data-and-features-of-recommenders-6.png)

Let us take them one at a time, and for each, name the mechanism, show how it inflates offline metrics, and show how it collapses online.

**Leak 1 — Temporal leakage: using the future to predict the past.** This is the lifetime-CTR story from the intro. You compute an item's CTR over the entire dataset and attach it to every impression of that item, including impressions from before most of the clicks happened. Offline, that feature correlates strongly with the label because it literally aggregates the labels. The model leans on it, AUC jumps. Online, at serving time, "this item's CTR" can only be computed over the past, and for a brand-new item it is undefined. The feature the model trusted does not exist in the form it was trained on. **Symptom: large offline gain, flat-or-negative online, and the gap is worst for fresh items.**

**Leak 2 — Target leakage via aggregates that include the label row.** A finer-grained cousin: even if you window the aggregate to the past, if your join is sloppy and the aggregate for an impression *includes that impression's own outcome*, you leak the label directly. The OOF target encoder above fixes the random-split version of this; the timestamp-aware version requires that every aggregate use a strict `<` comparison on the impression timestamp, never `<=`, and never the export time. **Symptom: an implausibly strong single feature; ablating it tanks offline AUC by a suspiciously large margin (more than any honest feature should be worth).**

**Leak 3 — Train-serve skew: a feature computed differently offline vs online.** Even with perfect time discipline, if the *code* that computes a feature differs between the offline training pipeline (a Spark batch job over Parquet) and the online serving path (a streaming feature store with a 5-minute lag), the model trains on one distribution and serves on another. The classic case: offline you compute "items in cart in last hour" over a complete, sorted log; online your streaming store is 5 minutes behind, so the count is systematically low. The model learned coefficients for the offline distribution. **Symptom: offline metrics fine, online metrics quietly worse, and no single feature looks broken — the *whole* model is mildly miscalibrated.** We give this its own section (7) because it is the hardest to detect and the most common in real systems; see also [train-serve skew and the bugs that hide there](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there).

**Leak 4 — The label is in the features via post-impression signals.** The sneakiest. Some features are only *available after* the impression's outcome is known. "Number of comments on this item" or "total watch time of this item" keeps growing after the impression; if you snapshot it at export time, it encodes the success the model is trying to predict. Even "user's session length" leaks if measured over the whole session that contains the target. **Symptom: near-perfect offline AUC (0.95+ on a problem that should be 0.75), a feature whose importance dwarfs the IDs, and a model that is useless online because the feature is unknowable at serving.**

The unifying principle is one sentence, and it is worth memorizing: **a feature is valid only if it can be computed, identically, from information strictly available before the prediction timestamp, in both training and serving.** Everything in this post is an application of that rule. Violate it and the feedback loop will amplify the damage — your leaky model serves bad recommendations, which generate bad logs, which train the next leaky model.

That amplification deserves a beat, because it is what makes recommender leakage worse than leakage in a static prediction problem. In a one-shot Kaggle competition, a leak inflates your leaderboard score and that is the end of it. In a deployed recommender, the model's outputs *become the next training set's inputs*. A leaky model that overranks items with high (leaked) lifetime CTR will serve those items more, generating more impressions and clicks for them, which makes their lifetime CTR higher still, which makes the next model overrank them even more confidently. The leak does not just inflate one offline number; it seeds a self-reinforcing distortion of the catalog. So the cost of getting features wrong compounds over retraining cycles in a way that a single offline metric can never reveal. This is the deeper reason point-in-time correctness is non-negotiable: it is not just about one model's honesty, it is about keeping the feedback loop from learning a lie.

A useful way to *detect* leakage before it reaches production, beyond a clean temporal split, is the **feature-importance sniff test combined with an ablation**. Train the model, rank features by importance (permutation importance or the gradient-times-input on a held-out batch), and look at the top of the list. If a single non-ID feature dominates — especially an aggregate or a count — be suspicious. Then ablate it: retrain without it and measure the offline drop. An honest, valuable feature might be worth a point or two of AUC. A feature that costs you five or ten points of AUC when removed is not a great feature; it is a leak, because no single honest feature in a mature recommender is worth that much over the ID embeddings. The rule of thumb I use: **if removing one non-ID feature drops offline AUC by more than the gap between two consecutive serious model architectures, that feature is leaking until proven otherwise.**

#### Worked example: a leaky lifetime-CTR feature, then the fix

Let us make it concrete with numbers, because the *size* of the inflation is the thing to feel. Suppose our Criteo-style ranker without any CTR feature scores **AUC 0.760** on a proper temporal test set. Now we add `item_lifetime_ctr`, computed over the entire dataset and joined to every row. Offline AUC jumps to **0.810** — a 5-point gain, the kind that gets you promoted. We ship it. Online CTR drops **2.0%**, because for the most-served fresh items the feature was pure future information and at serving time it is undefined or stale. We roll back, humbled.

Now the fix: compute a **point-in-time CTR** — for each impression at time $t$, the CTR is clicks/impressions of that item over a trailing window $[t - 7\text{d}, t)$, strictly before $t$. Offline AUC is now **0.765** — only a 0.5-point gain over the no-CTR baseline, because the honest feature really is only mildly predictive (most of the lifetime-CTR "signal" was leakage). But this 0.5 point is *real*: shipped, online CTR rises **+1.4%**. The honest feature is worth less offline and worth *more* online. The figure below contrasts the two.

![Before and after diagram contrasting a leaky lifetime CTR feature that inflates offline AUC and collapses online against a point-in-time CTR feature that is honest offline and lifts online metrics](/imgs/blogs/the-data-and-features-of-recommenders-4.png)

This is the most important graph in the post. The leaky version has *higher offline AUC and lower online CTR*. If your evaluation rewards offline AUC, it will actively select for leakage. The defense is structural, not vigilant: build features point-in-time-correct by construction, and split temporally so that any residual leakage is at least measured. For the discipline of splitting, see [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate).

## 6. The point-in-time-correct feature pipeline

Now we build the feature pipeline that does not lie. The two pillars are a **temporal split** and **point-in-time aggregates**. Let us do both, on the Criteo-style dataset, and produce a `DataLoader` ready for the model from section 3.

### The temporal split, done right

A recommender must be evaluated the way it will be deployed: trained on the past, tested on the future. A random shuffle-split mixes future rows into training and is the most common cause of optimistic offline numbers. The rule is simple: pick a cutoff timestamp $t_{\text{split}}$, train on everything before it, test on everything at or after it. If you do feature aggregation, the test set's aggregates may use training-period data (the past relative to the test rows) but the training set's aggregates may *never* use test-period data.

There is a subtle refinement that catches teams who got the basic split right. If your features include aggregates over a window — say a 7-day trailing CTR — then a training example timestamped right at the cutoff has a window that reaches back into pure training data, fine. But the *test* examples just after the cutoff have windows that span the boundary, reaching back into the training period; that is correct and expected (at serving you also use the recent past). The danger is the reverse: a training example whose label resolution window (the attribution window from section 2) extends *past* the cutoff. Its label may have been influenced by test-period behavior. The clean fix is an **embargo gap**: leave a small buffer between the last training timestamp and the first test timestamp, at least as wide as your attribution window, so no training label depends on a moment that also feeds a test feature. The embargo is cheap insurance against the most easily-missed temporal leak. Skip it and you can have a perfectly temporal-looking split that still bleeds a little future into training.

```python
def temporal_split(df, ts_col="ts", frac_train=0.8):
    df = df.sort_values(ts_col).reset_index(drop=True)
    cut = df[ts_col].quantile(frac_train)
    train = df[df[ts_col] < cut].copy()
    test  = df[df[ts_col] >= cut].copy()
    return train, test, cut
```

### Point-in-time aggregates with a trailing window

The heart of an honest pipeline is computing each impression's aggregate features using only strictly-earlier events. The clean, scalable way is a **time-windowed rolling count** per key, evaluated as-of each impression's timestamp. Pandas makes the as-of join readable for moderate data; at scale you would do this in Spark with a windowed aggregation or precompute it in a streaming feature store keyed by `(item_id, time_bucket)`.

```python
def point_in_time_ctr(df, key="item_id", ts="ts",
                      click="y", window="7D"):
    """For each row, CTR of `key` over a trailing window strictly before ts.
    No row sees its own label (shift by one within key)."""
    df = df.sort_values([key, ts]).copy()
    # rolling sums over a time window, EXCLUDING the current row via shift
    g = df.groupby(key, group_keys=False)
    # cumulative within the window: rolling needs a DatetimeIndex per group
    def _agg(grp):
        grp = grp.set_index(ts)
        clicks = grp[click].shift(1).rolling(window).sum()      # strictly before
        impr   = grp[click].shift(1).rolling(window).count()
        ctr = (clicks + 1.0) / (impr + 2.0)                     # Laplace smoothing
        out = grp.copy()
        out["pit_ctr"] = ctr.fillna(grp[click].mean()).values   # prior for cold items
        out["pit_impr"] = impr.fillna(0).values
        return out.reset_index()
    return g.apply(_agg).reset_index(drop=True)
```

Three details earn their keep. The `shift(1)` makes the aggregate strictly *before* the current row, killing leak 2 (self-inclusion). The trailing `rolling(window)` kills leak 1 (lifetime / future) by bounding the aggregate to the recent past. The Laplace smoothing `(clicks+1)/(impr+2)` is the bias-variance dial from section 4: a brand-new item with zero prior impressions gets the global prior, not a divide-by-zero or a wild estimate from one observation. This is the same smoothing logic as the empirical-Bayes target encoder, now made *temporal*.

### Putting the pipeline together

Now assemble: temporal split, build the categorical index on *train only*, compute point-in-time aggregates, OOF target encode the high-cardinality columns on train, and wrap in a `DataLoader`.

```python
# 1. temporal split — the test set is strictly the future
train_df, test_df, cut = temporal_split(raw, ts_col="ts", frac_train=0.8)

# 2. point-in-time CTR (computed within each split independently, no cross-leak)
train_df = point_in_time_ctr(train_df, key="item_id", window="7D")
test_df  = point_in_time_ctr(test_df,  key="item_id", window="7D")

# 3. categorical index built on TRAIN ONLY (test values may be OOV — correct!)
cat_index = build_cat_index(train_df, CAT_COLS, min_freq=10, n_oov=1000)
cardinalities = [max(cat_index[c](v) for v in train_df[c].dropna().unique()) + 1
                 for c in CAT_COLS]

# 4. numeric stats fit on TRAIN, applied to both
_, num_stats = preprocess_numerics(train_df)

# 5. datasets + loaders
train_ds = CriteoDataset(train_df, cat_index, num_stats)
test_ds  = CriteoDataset(test_df,  cat_index, num_stats)
train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=8192, shuffle=False, num_workers=4)
```

Every step that touches the label or fits a statistic uses **train only**. The categorical index is built on train, so genuinely new items at test time become OOV — which is exactly what happens in production, where you serve items you have never trained on. Numeric normalization stats are fit on train. The aggregates are within-split and strictly-before. This pipeline cannot leak the four ways from section 5, and that is the entire point: leakage prevention is a *property of the pipeline's structure*, not a checklist you run afterward.

The training loop itself is unremarkable — that is the goal. All the intelligence went into the features.

```python
import torch.nn.functional as F

model = CTRRanker(cardinalities, d=32).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    for num, cat, y in train_loader:
        num, cat, y = num.cuda(), cat.cuda(), y.cuda()
        logits = model(num, cat)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
```

## 7. Train-serve skew, the hardest leak

Train-serve skew deserves its own section because it survives every check the others fail. Your timestamps are perfect, your aggregates are strictly-before, your split is temporal — and the model still degrades online, because the *feature value at serving time differs from the feature value the model trained on*, for the same conceptual feature. The figure below shows the mechanism.

![Before and after diagram contrasting an offline batch feature computed over a complete window against the same feature computed online from a lagging stream, showing how the mismatch halves precision in production](/imgs/blogs/the-data-and-features-of-recommenders-7.png)

There are three classic sources, and you will hit all three eventually:

**Different code paths.** Offline, "user's clicks in last 24h" is a Spark `groupBy.count` over a Parquet table. Online, it is a Redis counter incremented by a streaming job. The two implementations round time buckets differently, handle late-arriving events differently, and treat deletions differently. They disagree by a few percent on most rows and by a lot on the tail. The fix is a **single feature definition** consumed by both paths — a shared feature store (Feast, Tecton, or an in-house equivalent) where the transformation is defined once and materialized for both training and serving. If you cannot share code, you must at least share *test vectors*: log a sample of the actual online feature values and diff them against the offline computation on the same examples.

**Different freshness.** The online feature store has a materialization lag — counts are 5 to 15 minutes stale. Offline, you can compute the count as-of the exact impression time. So "items viewed in this session" is complete offline and truncated online. The model learned that a high session-view count predicts a click; online the count is systematically lower, so the model under-fires. The fix is to **train on the serving-time snapshot, not the complete history** — log the actual feature values served (point-in-time logging), and train on those logged values rather than recomputing from the raw event log. This is the single most effective anti-skew practice: *train on what you served.*

**Different value distributions for the same feature.** Even with identical code, if the offline data is sampled differently (e.g. negatives downsampled) and you do not correct for it, an aggregate computed over the sampled set differs from the full-traffic value online. The fix is to compute aggregates over the *full, unsampled* log and only downsample at the final training-example step.

#### Worked example: a 5-minute lag halving precision

Concretely: a feed ranker uses `session_item_views` (how many items the user has scrolled past this session) as a feature, and it is genuinely predictive — engaged users scroll a lot before clicking. Offline, computed as-of impression time, the feature ranges 0 to 200 with a mean of 30. Online, the streaming store lags 5 minutes; for an active scroller, 5 minutes is dozens of items, so the served value is systematically ~40% low, mean 18. The model's learned weight on this feature assumes mean 30; fed a mean-18 distribution, its calibration drifts and its precision@10 on the most-active (most-valuable) users drops about 40%. Nothing in the offline harness shows this — offline, the feature is correct. The only way to catch it is to **log served features and replay them**, then compare offline AUC computed on *recomputed* features versus on *logged* features. If those two AUCs diverge, you have skew. That diagnostic is worth building before you ship any aggregate feature.

The diagnostic itself is short. You log, alongside each impression, the exact feature vector the serving system used. Then offline you recompute the same features from the raw event log and compare. A per-feature distribution diff (a population stability index, or a simple mean/quantile diff) localizes which feature skews; an AUC diff between the two vector sets quantifies how much the skew costs.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def detect_skew(logged_feats, recomputed_feats, feature_names):
    """Compare features the serving system logged vs offline recomputation
    on the SAME examples. Large divergence => train-serve skew."""
    report = {}
    for j, name in enumerate(feature_names):
        a, b = logged_feats[:, j], recomputed_feats[:, j]
        # population stability index: how different are the two distributions
        bins = np.quantile(np.concatenate([a, b]), np.linspace(0, 1, 11))
        pa, _ = np.histogram(a, bins=bins); pa = pa / pa.sum() + 1e-6
        pb, _ = np.histogram(b, bins=bins); pb = pb / pb.sum() + 1e-6
        psi = float(np.sum((pa - pb) * np.log(pa / pb)))
        report[name] = {"mean_logged": float(a.mean()),
                        "mean_recomputed": float(b.mean()),
                        "psi": psi}                   # PSI > 0.2 is a red flag
    return report

# AUC diff: did the skew actually cost ranking quality?
auc_logged     = roc_auc_score(y, model_scores_on(logged_feats))
auc_recomputed = roc_auc_score(y, model_scores_on(recomputed_feats))
print(f"AUC gap from skew: {auc_recomputed - auc_logged:+.4f}")
```

A population stability index above roughly 0.2 on any feature is a strong skew signal; above 0.1 is worth investigating. If `auc_recomputed` is meaningfully higher than `auc_logged`, your offline numbers are computed on features cleaner than what you actually serve, and your online results will land below your offline expectations by roughly that gap. Build this before you ship; it is the single diagnostic that catches the leak no temporal split can find.

## 8. Results: leakage looks great offline, and the embedding-dim trade-off

Two measurement angles close the loop: quantifying how much leakage inflates offline metrics, and the embedding-dimension trade-off that governs your memory budget.

### Leakage inflation table

Here is a representative before-after on a Criteo-style temporal split — the kind of table you should produce for every aggregate feature, comparing the leaky and point-in-time versions on both offline AUC and (where you can A/B) online CTR. The numbers are illustrative of the *pattern*, which is the reproducible part; your exact deltas depend on your data.

| Feature variant | Offline test AUC | Online CTR (A/B) | Verdict |
|---|---|---|---|
| No CTR feature (baseline) | 0.760 | — | honest floor |
| `item_lifetime_ctr` (leaky) | 0.810 | -2.0% | leakage: offline up, online down |
| `item_pit_ctr_7d` (point-in-time) | 0.765 | +1.4% | honest: small offline gain, real online lift |
| `item_pit_ctr_7d` + OOF target enc on `C20` | 0.771 | +1.9% | honest stacking of valid features |
| Leaky target enc on `C20` (leave-self-in) | 0.844 | -3.1% | worst leak: looks best offline, hurts most |

Read the table top to bottom and the moral is unmistakable: **offline AUC is positively correlated with leakage.** The two rows with the highest offline AUC (0.810, 0.844) are the two leaky ones, and both *lost* online. The honest features gain little offline and win online. If your model-selection criterion is offline AUC and your features include un-audited aggregates, you are running a leakage-maximization procedure. The defense is the temporal-split-plus-point-in-time pipeline from section 6, plus the served-feature replay diagnostic from section 7.

### Embedding-dimension trade-off

The other measurement that drives real decisions is how AUC and memory scale with the embedding dimension $d$. AUC rises with $d$ but with sharply diminishing returns, while memory grows strictly linearly (section 3). So there is a Pareto knee — the smallest $d$ that captures most of the achievable AUC — and finding it is one of the highest-leverage tuning decisions because it sets your serving memory bill. The figure below shows a representative sweep.

![Matrix diagram showing embedding dimension against test AUC and table memory, with AUC rising with diminishing returns while memory grows linearly, marking d equals 64 as the Pareto knee](/imgs/blogs/the-data-and-features-of-recommenders-8.png)

#### Worked example: picking the embedding dimension

On our 200M-user, 50M-item platform, sweep $d \in \{8, 16, 32, 64, 128\}$ and you typically see test AUC climb 0.762, 0.774, 0.781, 0.784, 0.785, while table memory grows linearly 0.3, 0.6, 1.2, 2.4, 4.8 GB (for the item table at fp32; the user table scales the same way). Going from $d=8$ to $d=32$ buys +0.019 AUC for +0.9 GB — clearly worth it. Going from $d=64$ to $d=128$ buys +0.001 AUC for +2.4 GB — clearly not. The knee is around $d=64$: it captures essentially all the achievable AUC (0.784 of a 0.785 ceiling) at half the memory of $d=128$. In practice many shipped systems run $d=32$ or $d=64$, and reach for tricks like **mixed-dimension embeddings** (large $d$ for the popular head, small $d$ for the long tail) and **fp16 / int8 quantization** of the table at serving (halving or quartering memory at a fraction of a point of AUC) before they spend on a wider uniform dimension. The discipline is the same as everywhere in this post: measure the trade-off, find the knee, do not pay for the flat part of the curve.

Putting the feature types side by side clarifies which to invest in. The matrix below ranks the families on cardinality, density, signal, and leakage risk — the four properties that decide both how you encode a feature and how carefully you must audit it.

![Matrix diagram comparing feature families across cardinality, density, signal strength, and leakage risk, showing ID embeddings as highest signal and aggregate CTR features as highest leakage risk](/imgs/blogs/the-data-and-features-of-recommenders-5.png)

The pattern jumps out: the two highest-signal families (the ID embeddings) are the *lowest* leakage risk, because an ID is just an identity — it carries no future information. The highest-leakage family (aggregate CTR and target encodings) is only medium-to-high signal once you compute it honestly. So the prioritization writes itself: **invest first in ID embeddings, which give you the most signal for the least leakage risk**, and treat every aggregate as guilty until proven point-in-time-correct.

## 9. Case studies: how real systems build features

Three real systems, three lessons, all citable.

**Criteo display advertising — the canonical CTR feature set.** The Criteo Kaggle / Display Advertising Challenge dataset (2014) is the de facto benchmark for click-through-rate features and is what every "Criteo-style" reference in this post points to. It is a week of display-ad logs with **13 integer (count) features** and **26 categorical (hashed) features**, all anonymized, label = clicked or not. The categoricals are *already hashed* by Criteo — you receive opaque 32-bit hashes, not raw values, precisely because the real values (cookie IDs, site IDs) are unbounded high-cardinality. This is the industry's tacit admission that the hashing trick is the default, not the exception, for production ad features. Every winning solution on this dataset leans on the categorical embeddings (or factorization-machine cross terms over them); the 13 numerics, after `log1p`, are useful but secondary. It is the cleanest public illustration of the "categoricals dominate" thesis. (See Criteo's released dataset and the DLRM benchmark that uses it.)

**YouTube deep recommendations (Covington, Adams, Sargin, RecSys 2016).** The YouTube paper is the most-cited demonstration of feature engineering at industrial scale, and it makes the "IDs plus a few engineered features" recipe explicit. Their candidate-generation network embeds the user's **watch history** (averaged video-ID embeddings) and **search history** (averaged token embeddings), plus context: **geographic embedding**, **device**, **gender**, and crucially the **"age of the video"** feature and a feature for **time since last watch**. The paper is famous for one engineered-feature insight directly relevant to leakage: they feed **"example age"** (how old the training example is) as a feature so the model can learn the *time-dependence* of popularity, then *set it to zero at serving* — an explicit, honest handling of a feature that would otherwise leak the future. They also report that powering features like **previous impressions of a video to a user** (to demote repeatedly-shown-but-not-clicked items) measurably improved engagement. The lesson: the engineered features that matter most are recency and history, and the discipline that makes them safe is point-in-time correctness.

**Google Wide and Deep (Cheng et al., DLRS 2016) — cross features for memorization.** Wide and Deep, deployed on Google Play app recommendations, is the canonical demonstration of **cross features**. The "deep" side embeds the categoricals and generalizes; the "wide" side is a linear model over **hand-crafted cross-product features** like `AND(user_installed_app=netflix, impression_app=hulu)` — explicit user-item interaction terms that let the model *memorize* specific co-occurrences a deep model would smooth away. Google reported that the combined model lifted online app acquisitions over a deep-only baseline (the paper reports a +3.9% relative gain in app acquisitions in the live experiment versus the deep-only model, with the wide-only and deep-only as the comparison arms). The lesson for this post: cross features are a distinct, valuable family, and they are *memorization* features — which is exactly why they must be built point-in-time-correct, since a cross over interaction history is an aggregate and aggregates leak. We will go deep on this architecture later in the series; for now it is the canonical reason "cross features" sits in the taxonomy.

The through-line across all three: the signal lives in IDs, history, and recency; the danger lives in how you aggregate over interaction data with respect to time. Every system that ships at scale solves the same two problems — high-cardinality encoding (hashing / frequency thresholds) and leakage-free aggregation (point-in-time, example-age handling, train-on-what-you-served).

## 10. Which features are worth it

You have a finite feature budget — every feature is latency at serving, storage in the feature store, and a surface for skew. Here is the prioritization I would defend, roughly in order of signal-per-unit-risk:

| Feature | Signal | Cost / risk | Worth it? |
|---|---|---|---|
| User-ID embedding | Very high | Memory (table size) | Always — the core signal |
| Item-ID embedding | Very high | Memory (table size) | Always — the core signal |
| User history sequence (last-N) | High | Latency, point-in-time discipline | Yes — but build strictly-before |
| Item age / recency | High (feeds/news) | Skew (compute time) | Yes — set example-age to 0 at serve |
| Context (time, device, position) | Medium | Skew (offline-vs-online clocks) | Yes — cheap, share the definition |
| Point-in-time CTR / counts | Medium | High leakage risk | Yes, only if windowed + strictly-before |
| Content embeddings (text/image) | Medium (helps cold-start) | Compute, storage | For cold-start and new items |
| Demographics | Low | Privacy, low signal | Usually skip once IDs are learned |
| Lifetime CTR / leave-self-in target enc | Fake-high | Catastrophic leakage | Never |

The decisive rules: spend your first effort on the ID embeddings and getting their dimension right, because they give the most signal for the least risk. Add history and recency next, because they are the next-strongest signal — but only with point-in-time discipline. Treat every aggregate as a liability that must earn its place through a leakage audit, not an asset you sprinkle in. And demographics, despite being the first thing newcomers reach for, are usually the *least* valuable family once the IDs are learned, because a user-ID embedding subsumes "what a 25-34 US Android user likes" with "what *this* user likes."

There is also a cost no offline metric shows: **every feature is a serving-time liability.** A feature that must be fetched from a feature store at request time adds a network round trip; a feature computed from the request adds CPU; an aggregate adds a streaming pipeline you have to keep healthy at 3am. A model with 200 features and a model with 40 features that capture 98% of the signal are very different operational beasts. The 40-feature model is faster to serve, cheaper to store, has fewer skew surfaces, and is easier to debug when something drifts. So "which features are worth it" is not only an accuracy question; it is a question of whether the marginal AUC justifies a permanent operational tax. My default is to be ruthless: start from the IDs plus a handful of high-signal engineered features, add features one at a time with an A/B test and a skew audit, and delete any feature whose online lift does not clearly exceed its serving and maintenance cost. The most reliable recommenders I have shipped had *fewer* features than the prototypes that preceded them, not more — the prototype's extra features were either redundant with the IDs or quietly skewing, and removing them made the system both faster and more honest.

## 11. When to reach for which encoding (and when not to)

A short decision guide for the encoding choice, which is the question you will actually face per column:

- **Low-cardinality categorical (under ~10k values):** full `nn.Embedding`, one row per value, plus an OOV/padding row. No hashing, no thresholds. Cheap and exact.
- **High-cardinality, bounded, mostly-recurring (user ID, item ID, up to ~10^8):** full embedding table with a **frequency threshold** — dedicated rows for the head, OOV buckets for the tail. This is the workhorse.
- **Unbounded or extreme-cardinality (cookie ID, query, URL, 10^9+):** the **hashing trick**, sized to a load factor you can tolerate (section 3). Optionally double-hash (two hash functions into two tables, concatenated) to reduce the impact of any single collision.
- **High-cardinality categorical with limited data per value, and you want a compact scalar:** **out-of-fold, smoothed target encoding** — never naive target encoding. And only if you have verified it does not skew offline-vs-online.
- **Heavy-tailed numeric (count, price, dwell time):** `log1p` then standardize. Never feed a raw count to a neural net next to a normalized feature.
- **Don't** hash a low-cardinality column (you gain nothing and add collisions). **Don't** target-encode the user or item ID (the embedding already does this, learned, without leakage). **Don't** ship any aggregate without the point-in-time and served-feature-replay audits.

## 12. Key takeaways

- **The four feature families are user, item, context, and cross.** Each has a characteristic cardinality, density, signal, and leakage risk — learn to place any feature instantly.
- **ID embeddings carry most of the signal.** A user-ID or item-ID embedding is a learned per-entity free parameter that subsumes demographics; ablating it costs far more than ablating any engineered feature.
- **An embedding lookup is one-hot times the table**, costing $|V| \cdot d \cdot 4$ bytes — billions of parameters at industrial scale, so size it on a whiteboard before you train.
- **Collisions are a function of the load factor $n/m$, not $n$.** Size the hash table to the load factor you can tolerate; keep dedicated rows for the frequent head, hash the tail.
- **Leakage has four paths:** temporal (future info), self-inclusive aggregates, train-serve skew, and post-impression signals. Each inflates offline metrics and collapses online.
- **Offline AUC is positively correlated with leakage** if your features include un-audited aggregates — model selection on offline AUC becomes leakage maximization.
- **Build features point-in-time-correct by construction:** temporal split, strictly-before trailing-window aggregates, out-of-fold smoothed target encoding, and train on the features you actually served.
- **Train-serve skew is the hardest leak** because every other check passes; the cure is a single shared feature definition and logging served features for replay.
- **Pick the embedding dimension at the Pareto knee** — often $d = 32$ to $64$ — then quantize before you widen.

## 13. Further reading

- Covington, Adams, Sargin, *Deep Neural Networks for YouTube Recommendations*, RecSys 2016 — the canonical industrial feature set (watch/search history embeddings, example age, "train on what you serve").
- Cheng et al., *Wide & Deep Learning for Recommender Systems*, DLRS 2016 — cross features for memorization plus deep generalization, with online lift on Google Play.
- Weinberger et al., *Feature Hashing for Large Scale Multitask Learning*, ICML 2009 — the original analysis of the hashing trick and its collision behavior.
- Naumov et al., *Deep Learning Recommendation Model (DLRM)*, 2019, and the Criteo Display Advertising Challenge dataset — the standard CTR benchmark with 13 numeric + 26 hashed categorical features.
- Micci-Barreca, *A Preprocessing Scheme for High-Cardinality Categorical Attributes*, SIGKDD Explorations 2001 — the empirical-Bayes target-encoding smoothing this post uses.
- Within series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) · [embeddings, the heart of recommenders](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders) · [train-serve skew and the bugs that hide there](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there) · [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) · [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

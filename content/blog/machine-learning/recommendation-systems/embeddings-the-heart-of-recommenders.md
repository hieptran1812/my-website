---
title: "Embeddings: The Heart of Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build the embedding table from the one-hot lookup identity up, see why it holds 99% of a recommender's parameters, then implement the hashing trick, QR embeddings, mixed dimensions, and int8 quantization in PyTorch and measure Recall@10 against memory on MovieLens."
tags:
  [
    "recommendation-systems",
    "recsys",
    "embeddings",
    "embedding-table",
    "hashing-trick",
    "qr-embeddings",
    "quantization",
    "machine-learning",
    "pytorch",
    "dlrm",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/embeddings-the-heart-of-recommenders-1.png"
---

The first time I watched a recommender model fall over in production, it had nothing to do with the model. The architecture was a tidy little multi-layer perceptron, three hidden layers, a few million dense weights — the kind of thing you could train on a laptop. What killed the host was a single tensor: the item embedding table. We had 80 million items in the catalog, we had picked a 128-dimensional embedding, and somebody multiplied those two numbers on a whiteboard a week too late. That table was 80,000,000 × 128 × 4 bytes ≈ **41 GB** of float32. The MLP on top was about 8 MB. We had spent all our design energy on the 8 MB and none on the 41 GB, and the 41 GB is the part that pages the kernel's OOM killer at 3 a.m.

That is the lesson this entire post is built around: in a modern recommender, **the embedding tables _are_ the model**. They hold the overwhelming majority of the parameters, they carry the overwhelming majority of the learned signal, they dominate the memory bill, and they cause the overwhelming majority of the production fires. The dense network you draw on the slide — the two-tower, the DeepFM, the DCN — is the cheap part. The embeddings are where the money, the memory, and the personalization live. If you learn to reason about embedding tables the way a database engineer reasons about indexes, you will be the person who _prevents_ the 3 a.m. page instead of the person who gets it.

This post is the bridge between the "what's a recommender" framing and the heavy machinery later in the series. We'll start from the one identity that makes everything click — that an embedding lookup is just a one-hot vector times a matrix — and the figure below is that identity drawn out. Then we go deep on the three problems every embedding table runs into at scale: it is **too big** (the high-cardinality problem and the hashing trick, multi-hash, QR embeddings), it is **inefficiently sized** (dimension selection and mixed-dimension embeddings), and it is **too expensive to serve** (quantization to int8, product quantization, the memory wall). Along the way we'll write real PyTorch, quantize a table and measure the accuracy loss, and read off a results table that puts Recall@10 next to memory in gigabytes so you can pick a real Pareto point. By the end you should be able to size, compress, and debug an embedding table for a catalog of any size, and know which lever to pull when memory is the thing that breaks.

![A three-by-three grid showing a one-hot input row, the embedding table rows below it, and the output where only the selected row survives](/imgs/blogs/embeddings-the-heart-of-recommenders-1.png)

## 1. What an embedding table actually is

Let's nail the definition before anything else, because half of all confusion about embeddings comes from skipping it.

A recommender's raw inputs are almost all **categorical IDs**: user 8,481,022; movie 1,193; the country code "VN"; the device "iOS"; the hour-of-day bucket 14. A neural network cannot consume the integer 1,193 directly — the integer's _magnitude_ is meaningless (movie 1,193 is not "1,000 more" than movie 193), so feeding it as a number would teach the model a lie. The classical fix is **one-hot encoding**: represent ID $i$ out of a vocabulary of size $|V|$ as a vector of length $|V|$ that is all zeros except a single 1 in position $i$. Movie 1,193 in a 27,000-movie catalog becomes a 27,000-long vector with one 1.

That one-hot vector is correct but useless on its own — it's enormous, sparse, and carries no notion that two movies might be similar. So we follow it with a learnable dense projection. We define a weight matrix $E \in \mathbb{R}^{|V| \times d}$, the **embedding table**, with one row per ID and $d$ columns (the embedding dimension). To get the representation of ID $i$, we compute the matrix product of its one-hot vector $\mathbf{e}_i$ with the table:

$$
\mathbf{v}_i = \mathbf{e}_i^\top E.
$$

Here is the entire trick. Because $\mathbf{e}_i$ is zero everywhere except position $i$, that matrix product is _exactly_ the $i$-th row of $E$. Every other row gets multiplied by zero and vanishes. So:

$$
\mathbf{e}_i^\top E = E[i, :].
$$

**A matrix multiply by a one-hot vector is identical to a table lookup.** That is the figure above: the one-hot selects, the table supplies, and only the selected row survives to the output. This is not an approximation or an analogy — it is an algebraic identity. And it matters enormously for performance, because no sane implementation actually materializes the giant one-hot vector and does a $|V| \times d$ matrix multiply. Instead it does an $O(d)$ gather: read row $i$ out of memory. PyTorch's `nn.Embedding` _is_ this gather, with the gradient wired so that backprop only updates the rows that were actually looked up.

```python
import torch
import torch.nn as nn

# A movie embedding table: 27,000 movies, each a 64-dim vector.
movie_emb = nn.Embedding(num_embeddings=27_000, embedding_dim=64)

ids = torch.tensor([1193, 661, 914])      # a batch of three movie IDs
vecs = movie_emb(ids)                       # shape (3, 64) — three gathered rows
print(vecs.shape)                           # torch.Size([3, 64])

# Prove the one-hot identity by hand:
one_hot = torch.zeros(27_000); one_hot[1193] = 1.0
manual = one_hot @ movie_emb.weight          # (27000,) @ (27000, 64) -> (64,)
assert torch.allclose(manual, movie_emb.weight[1193])   # identical
```

The values in $E$ are **learned**. They start as random noise and get pulled around by gradient descent so that IDs which behave similarly in the data — movies watched by the same people, users who click the same things — end up with similar vectors. That emergent geometry is the whole point: it lets the model _generalize across IDs_. A model that only ever saw "user A watched movie X" can still score "user A on movie Y" sensibly if Y's vector landed near X's vector during training. The embedding table is, quite literally, the learned dictionary that turns meaningless IDs into a geometry of taste.

It's worth sitting with _why_ that geometry emerges, because it's not magic — it's a direct consequence of the loss. When the model is trained to predict that user A clicks movie X, gradient descent nudges A's vector and X's vector toward each other (so their dot product rises). Do that across millions of interactions and any two movies that share a lot of viewers get repeatedly pulled toward the same users, and therefore toward each other. The geometry is the fixed point of all those pulls: items co-clustered with the users who like them, users positioned near the items they consume. No one hand-codes "these two action movies are similar"; the table _discovers_ it because the data says the same people watch both. This is also why a cold item with no interactions has a useless vector — nothing has pulled it anywhere yet, so it sits wherever its random init left it. The data side of this — which IDs you even have, how you bucketize continuous features, how you handle hashing collisions in the feature pipeline — is covered in [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders); here we focus on the table itself.

### Three flavors of embedding in one model

In a real system you will have many tables, not one. The common families:

- **User embeddings**: one row per user ID, learning each user's taste vector. Often the single largest table because user counts dwarf catalog counts (a billion users, tens of millions of items).
- **Item embeddings**: one row per item/movie/product, the thing you retrieve and rank. Shared across users — every user scores against the same item geometry.
- **Feature embeddings**: one (usually small) table per categorical feature — country, device, ad campaign, day-of-week. Each maps that feature's values to vectors that get concatenated into the model input.

The classic deep recommender — Google's Wide & Deep, Facebook's DLRM, the [two-tower retrieval model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) — gathers a row from each relevant table, concatenates them into one wide input vector, and feeds that to a dense MLP. The lookups are sparse and cheap _per example_; the tables that back them are colossal.

## 2. Why embeddings dominate the parameter count

Here is the number that surprises people who came from computer vision or NLP, where the dense layers _are_ the model. In a recommender the dense layers are a footnote. Let's do the arithmetic that the figure below stacks up.

Take a single item table for a 100-million-item catalog at $d = 64$ in float32:

$$
|V| \times d \times \text{bytes} = 10^8 \times 64 \times 4 = 2.56 \times 10^{10} \text{ bytes} \approx 25.6\ \text{GB},
$$

and $10^8 \times 64 = 6.4 \times 10^9$ **parameters** — 6.4 billion, in one table. Add a user table of 1 billion users at the same $d$ and you are at 64 billion more parameters. Now look at the MLP that sits on top: three hidden layers of, say, 512 → 256 → 128 units over a concatenated input of a few hundred dimensions. That is on the order of a few hundred thousand to a couple million weights — call it 2M. The ratio is not close:

$$
\frac{\text{embedding params}}{\text{MLP params}} \approx \frac{70 \times 10^9}{2 \times 10^6} = 35{,}000\times.
$$

The embeddings are not _most_ of the model; they are essentially _all_ of it. The MLP is a rounding error. This is the central fact of recommender engineering and the reason the rest of this post exists.

![A vertical stack showing user, item, and feature embedding tables in billions of parameters above a tiny dense MLP and output head](/imgs/blogs/embeddings-the-heart-of-recommenders-2.png)

A few consequences follow immediately, and they reshape how you build the system:

1. **Your optimizer state is also dominated by embeddings.** Adam keeps two moments per parameter. Naively that triples your memory: the 25.6 GB table needs ~51 GB more for Adam's $m$ and $v$. In practice production systems use a **sparse optimizer** that only stores moments for rows touched this step, or a stateless optimizer (plain SGD, or Adagrad with a per-row accumulator) precisely to dodge this.
2. **The tables don't fit on one accelerator.** A single H100 has 80 GB. A 70-billion-parameter embedding store is ~280 GB in fp32. You must **shard** the tables across hosts — this is what frameworks like TorchRec, HugeCTR, and the DLRM training stack are built to do. The dense part replicates trivially; the embeddings are the hard distributed-systems problem.
3. **Training throughput is bound by embedding lookups and their gradients, not matrix multiplies.** The all-to-all communication that ships gathered rows between shards is frequently the bottleneck, not the GEMM in the MLP. The model that looks compute-bound on paper is memory- and network-bound in reality.

#### Worked example: sizing a feed recommender's tables

You're sizing the embedding store for a short-video feed. Vocabulary: 800M videos (item table), 300M users (user table), and a creator-ID feature table of 50M creators. You pick $d = 32$ for items and users (you'll justify the small $d$ later) and $d = 16$ for creators. Float32. What's the total embedding memory?

$$
\begin{aligned}
\text{items}    &= 8\times10^{8} \times 32 \times 4 = 1.024 \times 10^{11}\ \text{bytes} \approx 102.4\ \text{GB} \\
\text{users}    &= 3\times10^{8} \times 32 \times 4 = 3.84  \times 10^{10}\ \text{bytes} \approx 38.4\ \text{GB} \\
\text{creators} &= 5\times10^{7} \times 16 \times 4 = 3.2   \times 10^{9}\ \text{bytes} \approx 3.2\ \text{GB}
\end{aligned}
$$

Total ≈ **144 GB** of embeddings, before optimizer state. Add Adam moments and you're near 430 GB. This does not fit on two H100s, let alone one. The arithmetic alone tells you that you _will_ be compressing these tables — the only question is which scheme. That is the rest of this post. Notice also how much the dimension choice swings the bill: dropping items from $d = 32$ to $d = 16$ would halve the 102 GB to 51 GB and shave a third off the total. Dimension is a lever, not a default.

### EmbeddingBag and multi-valued features

One detail that trips up engineers coming from `nn.Embedding`: not every feature is a single ID. A user has a _list_ of recently-watched movies; a video has a _set_ of tags; a query is a _bag_ of tokens. These multi-valued (or "multi-hot") features need to be pooled — you look up several rows and combine them into one vector by sum, mean, or max. You could do this with `nn.Embedding` followed by a manual mean, but PyTorch gives you `nn.EmbeddingBag`, which fuses the gather and the pooling into a single kernel. It is both faster (it never materializes the per-element vectors) and more memory-efficient (no intermediate $(\text{batch} \times \text{bag} \times d)$ tensor), which matters when the bags are large.

```python
import torch
import torch.nn as nn

# A user's "recently watched" feature: a variable-length bag of movie IDs.
bag = nn.EmbeddingBag(num_embeddings=27_000, embedding_dim=64, mode="mean")

# Flatten all bags into one tensor; offsets mark where each user's bag starts.
movie_ids = torch.tensor([1193, 661, 914,    50, 100,    7])
offsets   = torch.tensor([0,              3,         5])   # 3 users
pooled = bag(movie_ids, offsets)        # (3, 64): one pooled vector per user
print(pooled.shape)                      # torch.Size([3, 64])
```

The pooling mode is a modeling choice with real consequences. `mean` treats every item in the bag equally and is robust to bag length, but it washes out the most informative item; `sum` preserves magnitude (a user who watched 50 action movies gets a stronger action signal than one who watched 2) but couples the representation to bag length; `max` keeps the single strongest signal per dimension. For sequential signals where order matters, you outgrow `EmbeddingBag` entirely and reach for an attention-based pooler — which is the bridge to [self-attention sequence models like SASRec and BERT4Rec](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec). The point for this post: the same embedding _table_ backs both single-ID and bag features; the difference is only how you pool the gathered rows.

### The gather and scatter that dominate training

Because the forward pass is a gather (read the looked-up rows) and the backward pass is a scatter-add (accumulate gradients into exactly those rows), the embedding table's training cost is not a matrix multiply — it is irregular memory traffic. In a single training step the model touches only the rows for the IDs in the batch — maybe a few thousand rows out of a hundred million — so the gradient is _extremely sparse_. This is the property that makes sparse optimizers possible and that makes the embedding store a distributed-systems problem rather than a compute problem. When you shard the item table across eight hosts and a batch needs rows from all eight, the framework issues an all-to-all to gather them, runs the dense MLP, then scatters gradients back. That all-to-all, not the GEMM, is usually the throughput ceiling. Internalizing "embeddings are memory traffic, not flops" is what separates engineers who can scale a recommender from those who keep trying to optimize the MLP.

## 3. The high-cardinality problem and the hashing trick

The single biggest table problem is that vocabularies are not just large, they are **open and growing**. New users sign up; new videos get uploaded every second; new ad campaigns launch hourly. A full per-ID table assumes you can enumerate every ID in advance and allocate a row for each. At a billion items and climbing, that's a non-starter — and worse, you'd allocate a 64-dimensional vector for an item that was uploaded ten seconds ago and seen by nobody, which is pure waste.

The first and most important tool is the **hashing trick** (often called *feature hashing*). Instead of giving each ID its own row, you fix a table size $m$ — say 10 million rows — and map every ID into that table with a hash function:

$$
\text{row}(i) = h(i) \bmod m.
$$

Now the table is a fixed 10M × 64 regardless of whether the catalog has 100M items or 10 billion. New IDs need no allocation — they just hash into an existing row. Memory is bounded and predictable. The figure below contrasts the unbounded full table with the fixed hashed one.

![A before-and-after comparison of a full per-ID table that grows unbounded versus a fixed-size hashed table that caps memory by sharing rows](/imgs/blogs/embeddings-the-heart-of-recommenders-3.png)

The cost is **collisions**: two distinct IDs that hash to the same row share a single embedding vector and become indistinguishable to the model. If a popular movie and an obscure one collide, the popular one dominates the gradient and the obscure one inherits a vector that's wrong for it. This is the fundamental trade — we cap memory by accepting that some IDs will be confused with others.

### The science: collision probability

How bad are collisions? This is the birthday problem in disguise. With $n$ distinct IDs hashed uniformly into $m$ rows, the probability that a _given_ row receives _at least one_ collision (two or more IDs) is governed by the same algebra as birthdays sharing a day. The cleaner quantity to reason about is the probability that a specific ID collides with _at least one_ other ID. For a single other ID the chance of landing in the same row is $1/m$; the chance of avoiding it is $1 - 1/m$. Across the other $n - 1$ IDs, assuming independence,

$$
P(\text{ID } i \text{ collides with someone}) = 1 - \left(1 - \frac{1}{m}\right)^{n-1}.
$$

For the whole table, the expected fraction of IDs that share their row with at least one other ID follows the same form. When $n \ll m$ this is approximately $1 - e^{-n/m}$ (using $(1-1/m)^n \approx e^{-n/m}$), so the **load factor** $\alpha = n/m$ is the one number that controls collision pain:

$$
P(\text{collision}) \approx 1 - e^{-\alpha}.
$$

At $\alpha = 0.1$ (ten rows per ID — i.e. the table is ten times bigger than the catalog) you get $\approx 9.5\%$ of IDs colliding. At $\alpha = 1$ (one row per ID, the table the same size as the catalog) you get $\approx 63\%$. At $\alpha = 10$ (ten IDs crammed per row) essentially everything collides. The practical reading: **you want $m$ several times larger than the number of IDs you actually care about**, and the IDs you care about are the frequent ones, not the long tail of one-view junk.

#### Worked example: collision rate for a hashed feed table

You decide to hash that 800M-video catalog into $m = 200{,}000{,}000$ rows ($d = 32$, fp32). First the memory:

$$
2\times10^{8} \times 32 \times 4 = 2.56\times10^{10}\ \text{bytes} \approx 25.6\ \text{GB},
$$

down from 102.4 GB — a 4× cut just from picking $m = 0.25\,|V|$. Now the collisions. Load factor $\alpha = n/m = 8\times10^{8} / 2\times10^{8} = 4$. Expected collision rate:

$$
1 - e^{-4} \approx 1 - 0.018 = 0.982.
$$

98% of IDs share a row — that's brutal. But here's the saving grace: video views follow a power law. Maybe 2 million videos account for 90% of all watch time. If you hash only the _active_ catalog of 2M videos into 200M rows, $\alpha = 0.01$ and the collision rate is $\approx 1\%$. The lesson is that **the load factor that matters is computed over the IDs that carry signal, not the raw vocabulary**. Hash the head into a big table; let the cold tail collide freely — those IDs had almost no data to learn from anyway.

Here's the hashing trick as a drop-in PyTorch module. The only real subtlety is making the hash deterministic and reproducible across train and serve, because a feature computed differently offline and online is the silent killer that halves precision in production:

```python
import torch
import torch.nn as nn

class HashEmbedding(nn.Module):
    """Hash arbitrary integer IDs into a fixed-size table."""
    def __init__(self, num_rows: int, dim: int, seed: int = 0):
        super().__init__()
        self.num_rows = num_rows
        self.seed = seed
        self.emb = nn.Embedding(num_rows, dim)

    def _hash(self, ids: torch.Tensor) -> torch.Tensor:
        # A simple, deterministic multiplicative hash (Knuth's constant).
        # In production prefer a fixed 64-bit hash (e.g. xxhash) computed
        # identically in the offline pipeline and the online server.
        x = ids.to(torch.int64)
        x = (x ^ (x >> 33)) * 0x9E3779B97F4A7C15 + self.seed
        x = x ^ (x >> 29)
        return (x % self.num_rows).to(torch.long)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(self._hash(ids))

# 10M-row table absorbs a catalog of any size.
he = HashEmbedding(num_rows=10_000_000, dim=64)
print(he(torch.tensor([1_193, 9_999_999_999])).shape)   # (2, 64)
```

### Multi-hash: cheap insurance against collisions

A single hash means a collision is fatal — the two clashing IDs are stuck with one vector forever. **Multi-hash** (also called the "hashing trick with multiple hash functions," and the idea behind *Bloom embeddings* and *hash embeddings*) softens this. Use $k$ independent hash functions $h_1, \dots, h_k$, look up $k$ rows, and sum or average them:

$$
\mathbf{v}_i = \sum_{j=1}^{k} E\big[h_j(i) \bmod m\big].
$$

Now two IDs collide only if they clash in _all_ $k$ slots simultaneously, which is far less likely. With $k = 2$ hashes the chance two IDs are fully indistinguishable drops from roughly $\alpha$ to roughly $\alpha^2$ for small $\alpha$. The cost is $k$ lookups and $k$ gradient updates per ID instead of one — more bandwidth, more compute. It's a good cheap upgrade when single-hash recall is leaving points on the table but you can't afford a bigger $m$.

## 4. QR embeddings: collisions without the memory

The hashing trick caps memory but accepts a flat collision rate. **Quotient-Remainder (QR) embeddings**, introduced by Shi et al. at Facebook in 2020 ("Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems"), give you a way to represent a _unique_ vector for every ID using two _small_ tables — getting both bounded memory and (near) zero true collisions. This is one of the most elegant tricks in the embedding toolbox, and the figure below shows the dataflow.

![A branching graph where an item ID splits into a quotient bucket and a remainder bucket, each indexes a small table, and the two vectors combine into the final embedding](/imgs/blogs/embeddings-the-heart-of-recommenders-5.png)

The idea is integer division. Pick a modulus $m$. Any ID $i$ decomposes uniquely into a **quotient** and a **remainder**:

$$
q(i) = \left\lfloor \frac{i}{m} \right\rfloor, \qquad r(i) = i \bmod m.
$$

This pair $(q, r)$ is a unique fingerprint of $i$ — no two IDs share both. Keep two tables: a quotient table $Q$ with $\lceil |V| / m \rceil$ rows and a remainder table $R$ with $m$ rows, both of dimension $d$. The embedding of $i$ combines its quotient row and remainder row, typically by element-wise multiplication (or addition, or concatenation):

$$
\mathbf{v}_i = Q[q(i)] \odot R[r(i)].
$$

Because $(q, r)$ is unique, every ID gets a _distinct_ combined vector even though the underlying tables are tiny. Two IDs share a quotient row OR a remainder row, but never both — so they are never fully identical the way they are under plain hashing. You've traded "every ID is unique but the table is huge" for "every ID is the unique _product_ of two shared building blocks."

### The science: QR memory reduction

The full table costs $|V| \cdot d$ parameters. The QR tables cost

$$
\underbrace{\left\lceil \frac{|V|}{m} \right\rceil \cdot d}_{\text{quotient}} + \underbrace{m \cdot d}_{\text{remainder}}.
$$

To minimize the sum, set the two terms roughly equal: $|V|/m \approx m$, i.e. $m \approx \sqrt{|V|}$. Then each table has about $\sqrt{|V|}$ rows and the total is

$$
\approx 2\sqrt{|V|} \cdot d,
$$

versus $|V| \cdot d$ for the full table. The reduction factor is

$$
\frac{|V| \cdot d}{2\sqrt{|V|} \cdot d} = \frac{\sqrt{|V|}}{2}.
$$

For $|V| = 10^8$, $\sqrt{|V|} = 10^4$, so the reduction is $5{,}000\times$. That is staggering: the table shrinks from gigabytes to megabytes while every ID keeps a distinct (if shared-component) vector.

#### Worked example: QR memory math for 100M IDs

You have $|V| = 100{,}000{,}000$ items at $d = 64$, fp32. Full table:

$$
10^8 \times 64 \times 4 = 2.56 \times 10^{10}\ \text{bytes} \approx 25.6\ \text{GB}.
$$

Now QR with the optimal $m = \sqrt{10^8} = 10{,}000$. Quotient table has $\lceil 10^8 / 10^4 \rceil = 10{,}000$ rows; remainder table has $10{,}000$ rows. Total rows = 20,000.

$$
20{,}000 \times 64 \times 4 = 5.12 \times 10^{6}\ \text{bytes} \approx 5.1\ \text{MB}.
$$

From **25.6 GB to 5.1 MB** — a 5,000× reduction, exactly $\sqrt{|V|}/2$. The catch is real and worth stating plainly: the model now has far fewer _degrees of freedom_ for items. Two items that share a remainder row are forced to express their difference entirely through their quotient rows. With only 20,000 rows total, the model cannot memorize 100M genuinely independent item vectors — it composes them. In practice this costs a fraction of a point of recall on most benchmarks (Shi et al. report comparable accuracy at large compression on Criteo), which is a phenomenal trade when memory is the binding constraint. But if your task genuinely needs to distinguish 100M items at fine granularity, QR's compression is too aggressive and you'll see it in the metrics.

Here is a working QR embedding in PyTorch:

```python
import torch
import torch.nn as nn

class QREmbedding(nn.Module):
    """Quotient-Remainder embedding (Shi et al., 2020)."""
    def __init__(self, num_ids: int, dim: int, num_buckets: int,
                 op: str = "mult"):
        super().__init__()
        self.m = num_buckets
        self.op = op
        num_quot = (num_ids + num_buckets - 1) // num_buckets   # ceil
        self.quotient = nn.Embedding(num_quot, dim)
        self.remainder = nn.Embedding(num_buckets, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        q = torch.div(ids, self.m, rounding_mode="floor")
        r = ids % self.m
        vq, vr = self.quotient(q), self.remainder(r)
        if self.op == "mult":
            return vq * vr
        elif self.op == "add":
            return vq + vr
        else:  # concat then project elsewhere
            return torch.cat([vq, vr], dim=-1)

qr = QREmbedding(num_ids=100_000_000, dim=64, num_buckets=10_000)
# Two tables of 10K rows replace one 100M-row table.
total = qr.quotient.weight.numel() + qr.remainder.weight.numel()
print(f"{total:,} params")          # 1,280,000 params (~5.1 MB fp32)
print(qr(torch.tensor([42, 100_000_007])).shape)   # (2, 64)
```

## 5. Frequency filtering: don't embed what you can't learn

Before reaching for any clever compression, there is a blunt and shockingly effective tool: **just don't give rare IDs their own row**. An item seen three times in the entire training set has three gradient updates to learn a 64-dimensional vector — that vector is essentially noise. It eats memory and contributes nothing but variance. The fix is a **frequency threshold**: count how often each ID appears, and any ID below a cutoff (say, fewer than 10 occurrences) gets mapped to a shared **out-of-vocabulary (OOV)** bucket — or a small set of OOV buckets, often hashed.

This single step routinely removes 80–95% of the _rows_ in a long-tailed catalog while removing well under 5% of the _interactions_, because the tail is by definition rarely seen. It also handles cold-start gracefully: a brand-new ID starts life in the OOV bucket, accumulates a few interactions, and only graduates to its own row once it has earned enough data to learn from.

```python
import collections, torch

def build_id_map(train_ids, min_count: int = 10, num_oov: int = 1024):
    """Map frequent IDs to dense rows; rare IDs to hashed OOV buckets."""
    counts = collections.Counter(train_ids)
    vocab = {id_: i for i, (id_, c) in enumerate(
        sorted(counts.items(), key=lambda kv: -kv[1])) if c >= min_count}
    base = len(vocab)            # OOV buckets sit above the real vocab
    def remap(id_):
        if id_ in vocab:
            return vocab[id_]
        return base + (hash(id_) % num_oov)   # shared OOV row
    table_size = base + num_oov
    return remap, table_size

ids = [1, 1, 1, 1, 2, 2, 7, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88]
remap, size = build_id_map(ids, min_count=4, num_oov=16)
print(size)                  # small dense vocab + 16 OOV buckets
print(remap(88), remap(999)) # 88 is frequent -> dense row; 999 -> OOV
```

The reason this works ties directly to the bias-variance story of learning. An embedding row is a set of $d$ free parameters. To estimate them well you need signal — interactions. With too few interactions you overfit the row to noise, and at serve time that noisy vector hurts more than a shared, well-regularized OOV vector that at least encodes "generic rarely-seen item." Frequency filtering is the cheapest, most interpretable form of compression there is, and you should reach for it _before_ hashing or QR, not after.

## 6. Picking the dimension, and mixed-dimension embeddings

How do you choose $d$? It's the most consequential single number in your embedding budget — memory scales linearly in it, and so, up to a point, does capacity. Too small and the model can't separate items (representations collapse, distinct items get near-identical vectors); too large and you waste memory and start to overfit the rare IDs.

The honest answer is that $d$ is an empirical knob you sweep, but a few principles guide the search:

- **Bigger vocabularies generally want bigger $d$** — there's more to separate — but the relationship is sublinear and saturates fast.
- **The marginal recall from each extra dimension falls off a cliff.** On MovieLens-scale data, going from $d = 16$ to $d = 32$ might buy you a couple points of Recall@10; $d = 32$ to $d = 64$ a fraction of a point; $d = 64$ to $d = 128$ almost nothing while doubling memory. Sweep and find the knee.
- **The dot-product geometry has to fit.** For [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) and two-tower retrieval, $d$ is the rank of the score matrix you can represent. If the true taste structure is low-rank, a small $d$ suffices; if it's genuinely high-rank, you'll see underfitting at small $d$.

The dot-product point deserves a precise statement, because it tells you the _lower bound_ on $d$. A retrieval model scores user $u$ against item $i$ as $s_{ui} = \langle \mathbf{u}, \mathbf{v}_i \rangle$. Stack the user vectors into $U \in \mathbb{R}^{n \times d}$ and the item vectors into $V \in \mathbb{R}^{m \times d}$; the full score matrix is $S = U V^\top$, and by construction $\operatorname{rank}(S) \le d$. So $d$ is a hard ceiling on the rank of the preference structure the model can express. If the true user-item affinity matrix has effective rank 200 but you pick $d = 32$, no amount of training can recover the missing structure — you are projecting a rank-200 signal onto a rank-32 subspace and the residual is pure error. This is the formal reason a too-small $d$ underfits, and it's also why measuring the _effective rank_ of your learned table (Section 8) tells you whether $d$ is binding: if a $d = 64$ table consistently learns effective rank 60, it wants more dimensions; if it plateaus at rank 20, you over-provisioned.

#### Worked example: reading a dimension sweep

You sweep $d$ on MovieLens-20M with a two-tower model and observe Recall@10 of 0.351 at $d = 8$, 0.389 at $d = 16$, 0.405 at $d = 32$, 0.411 at $d = 64$, and 0.413 at $d = 128$. The marginal recall per doubling is +0.038, +0.016, +0.006, +0.002. The "knee" — where each doubling of memory stops paying for itself — is clearly between $d = 32$ and $d = 64$. The memory cost is linear: at a hypothetical 100M-item scale, $d = 32$ is 12.8 GB and $d = 64$ is 25.6 GB. So $d = 64$ costs you 12.8 GB extra for +0.006 Recall, while $d = 128$ costs another 25.6 GB for +0.002. The defensible pick is $d = 32$ or $d = 64$ depending on whether that last half-point of recall is worth doubling the table. The discipline: always plot recall _against memory_, not recall against $d$, because memory is what breaks and the marginal-recall-per-GB is the quantity you're actually optimizing.

But the deepest insight about dimension is that **a single uniform $d$ is wrong for almost every catalog.** Item popularity is power-law: a handful of head items appear in millions of interactions, while millions of tail items appear a handful of times. A uniform $d = 64$ gives the head item — which has the data to support a rich 128-dimensional representation — only 64 dimensions, while giving the tail item — which barely has data for 8 dimensions — a wasteful 64. You are simultaneously _starving_ the head and _over-parameterizing_ the tail. The figure below shows the contrast.

![A before-and-after comparison of a uniform-dimension table that starves head IDs and wastes memory on tail IDs versus a mixed-dimension scheme sized by frequency](/imgs/blogs/embeddings-the-heart-of-recommenders-6.png)

**Mixed-dimension embeddings** (Ginart et al., 2021, "Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems") fix this by assigning each ID a dimension proportional to its importance. Frequent IDs get wide vectors; rare IDs get narrow ones. The implementation trick that keeps the downstream model simple: store each block at its native (small or large) dimension, then **project** every block up to a common base dimension $d_{\text{base}}$ with a small per-block linear layer, so the concatenation step still sees uniform-width vectors. The paper's popularity-based rule sizes block $j$'s dimension as roughly $d_j \propto p_j^{\,\beta}$ for frequency $p_j$ and a temperature $\beta$.

```python
import torch, torch.nn as nn

class MixedDimEmbedding(nn.Module):
    """Frequency-blocked embeddings, each projected to a common width."""
    def __init__(self, block_sizes, block_dims, base_dim: int):
        super().__init__()
        assert len(block_sizes) == len(block_dims)
        self.tables = nn.ModuleList(
            nn.Embedding(n, d) for n, d in zip(block_sizes, block_dims))
        # Project each block's native dim up to base_dim.
        self.proj = nn.ModuleList(
            nn.Linear(d, base_dim, bias=False) for d in block_dims)
        self.offsets = torch.tensor(
            [0, *torch.cumsum(torch.tensor(block_sizes), 0).tolist()[:-1]])

    def forward(self, ids, block_of):
        # block_of[i] tells which frequency block id i lives in.
        out = ids.new_zeros((ids.size(0), self.proj[0].out_features),
                            dtype=torch.float32)
        for b, (tab, prj) in enumerate(zip(self.tables, self.proj)):
            mask = block_of == b
            if mask.any():
                local = ids[mask] - self.offsets[b]
                out[mask] = prj(tab(local))
        return out

# Head block: 1K hot items at d=128; tail block: 1M cold items at d=8.
mde = MixedDimEmbedding(block_sizes=[1_000, 1_000_000],
                        block_dims=[128, 8], base_dim=64)
```

Ginart et al. report that mixed dimensions match or beat a uniform table at a fraction of the memory on Criteo and MovieLens — the saving comes almost entirely from shrinking the enormous tail, which is exactly where uniform $d$ wastes the most. This is the same lesson as frequency filtering, refined: spend your dimensions where the signal is.

## 7. The memory wall and embedding compression

Everything so far has been about _not allocating_ memory you don't need. But even a well-pruned, well-sized table for a billion-item catalog is tens of gigabytes, and at serve time those vectors have to be _somewhere fast_ — in host RAM, ideally in a cache, sometimes spilled to SSD or a remote parameter server with the network latency that implies. This is the **memory wall**: the table is too big for accelerator memory, too hot for cold storage, and the access pattern (random gathers across billions of rows) is hostile to every cache hierarchy ever designed.

The figure below lays out the four schemes we've built side by side, scored on memory, collisions, accuracy, and serving cost — because the right choice depends entirely on which of those four axes is the one that's actually breaking.

![A four-by-four matrix comparing full, hashed, QR, and int8-quantized embeddings across memory, collisions, accuracy, and serving cost](/imgs/blogs/embeddings-the-heart-of-recommenders-4.png)

The complementary lever to _not allocating_ rows is **storing each allocated weight in fewer bits** — quantization. A float32 weight is 4 bytes. If you can represent it in int8 (1 byte) you've cut the table 4×; in int4, 8×. The standard scheme is **per-row affine quantization**: for each embedding row, store its weights as int8 plus a single float32 scale (and optionally a zero-point) per row:

$$
\hat{w} = s \cdot q, \qquad q = \mathrm{round}\!\left(\frac{w}{s}\right), \qquad s = \frac{\max_j |w_j|}{127}.
$$

The dequantized value $\hat{w}$ approximates the original $w$. Because the scale is per-row, each row's dynamic range is captured tightly. The figure below shows the fp32-vs-int8 trade directly: four bytes per weight versus one byte plus a tiny per-row scale, and a recall drop you can measure with a magnifying glass. The overhead of one fp32 scale per row of $d = 64$ is $4 / (64) \approx 6\%$, so the effective compression is close to the ideal 4×.

![A before-and-after comparison of an fp32 embedding table at four bytes per weight versus an int8 table at one byte per weight plus a per-row scale, showing a four times memory cut for a fraction of a recall point](/imgs/blogs/embeddings-the-heart-of-recommenders-7.png)

### The science: quantization error

How much accuracy do you lose? The quantization step introduces an error $\epsilon = w - \hat{w}$. For uniform quantization with step size $\Delta = s$ (the gap between adjacent representable values), and assuming the rounding error is uniformly distributed over $[-\Delta/2, \Delta/2]$ — a good approximation when the table has many distinct values — the per-weight mean-squared error is the variance of that uniform distribution:

$$
\mathbb{E}[\epsilon^2] = \frac{\Delta^2}{12}.
$$

With $\Delta = s = \frac{\max|w|}{127}$ for int8, the error is tiny relative to the signal — the **signal-to-quantization-noise ratio** improves by roughly 6 dB per bit, so 8 bits buys you about 48 dB of headroom. Embeddings turn out to be remarkably robust to this: the downstream dot product $\langle \mathbf{u}, \mathbf{v} \rangle$ sums $d$ terms, and the independent per-element rounding errors partly cancel (their sum grows like $\sqrt{d}$ while the signal grows like $d$). This is _why_ int8 embedding quantization typically costs only a fraction of a point of recall — the errors wash out in the inner product that produces the score. It is also why int4 starts to bite: at 4 bits the per-element error is 16× larger in variance and the cancellation can no longer hide it.

```python
import torch

def quantize_int8_per_row(W: torch.Tensor):
    """Per-row symmetric int8 quantization of an embedding table."""
    scale = W.abs().amax(dim=1, keepdim=True) / 127.0      # (V, 1)
    scale = scale.clamp_min(1e-8)
    q = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1)                              # int8 table + scales

def dequantize(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale.unsqueeze(1)

W = torch.randn(1_000_000, 64)                              # 256 MB fp32
q, s = quantize_int8_per_row(W)
Wq = dequantize(q, s)

err = (W - Wq).pow(2).mean().sqrt()
fp32_bytes = W.numel() * 4
int8_bytes = q.numel() * 1 + s.numel() * 4
print(f"RMSE: {err:.4f}")                                   # ~0.003-0.004
print(f"fp32: {fp32_bytes/1e6:.0f} MB  int8: {int8_bytes/1e6:.0f} MB")
print(f"compression: {fp32_bytes/int8_bytes:.2f}x")         # ~3.95x
```

#### Worked example: quantizing a billion-row table

You have a 1-billion-row item table at $d = 128$, fp32, and you want to know what int8 quantization buys you. Start with the memory:

$$
\text{fp32} = 10^{9} \times 128 \times 4 = 5.12 \times 10^{11}\ \text{bytes} = 512\ \text{GB}.
$$

Int8 stores one byte per weight plus one fp32 scale per row:

$$
\text{int8} = \underbrace{10^{9} \times 128 \times 1}_{\text{weights}} + \underbrace{10^{9} \times 4}_{\text{per-row scales}} = 1.28\times10^{11} + 4\times10^{9} \approx 132\ \text{GB}.
$$

So 512 GB becomes 132 GB — a 3.88× reduction (the per-row scale eats the last fraction below the ideal 4×; with $d = 128$ the scale overhead is $4/128 \approx 3\%$). Now the accuracy. A typical embedding weight, with $\sigma \approx 1/\sqrt{128} \approx 0.088$ initialization that grows modestly during training, gives per-row $\max|w| \approx 0.4$, so the step size is $s \approx 0.4/127 \approx 0.0031$ and the per-weight RMS error is $\sqrt{\Delta^2/12} = \Delta/\sqrt{12} \approx 0.0009$. Relative to a weight magnitude around $0.1$, that's under 1% per element — and in the $d = 128$ dot product the errors cancel down to a vanishing effect on the score ranking. The Pareto reading: you removed 380 GB and the recall moved by a fraction of a point. There is almost never a reason _not_ to serve int8 embeddings.

### Product quantization for even smaller tables

When 4× isn't enough, **product quantization (PQ)** compresses harder. Split each $d$-dimensional row into $M$ sub-vectors of length $d/M$, and for each sub-space learn a small codebook of, say, 256 centroids. A row is then stored as $M$ bytes — one centroid index per sub-vector — plus the shared codebooks. A 64-dim fp32 row (256 bytes) becomes 8 bytes at $M = 8$, a 32× reduction. PQ is the workhorse of [approximate-nearest-neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) for exactly this reason — faiss's `IndexIVFPQ` is PQ internally. The same idea, applied to the storage of the embedding table itself, lets you keep billion-row tables resident. The accuracy cost is larger than int8's because PQ approximates whole sub-vectors rather than individual weights, so it's the lever you reach for when memory is genuinely the wall and you've measured that you can afford the recall hit.

### Composition, sharing, and stale embeddings

Two more memory tactics worth naming:

- **Embedding sharing across towers/tasks.** If your retrieval tower and your ranking model both need an item embedding, sharing one table halves the storage and — often more importantly — keeps them consistent. In multi-task setups (the [MMoE/PLE world](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple)) a single shared embedding bottom feeds several task heads, which is both a memory win and a regularizer.
- **Stale embeddings.** At serve time you usually read embeddings from a snapshot that's minutes or hours old, not the live training state. That staleness is almost always fine for items (their vectors drift slowly) but can hurt for fast-moving user embeddings. Knowing _how stale_ your served table is — and whether the cold/new IDs in it have even been trained yet — is a debugging skill the [training pipeline post](/blog/machine-learning/recommendation-systems/building-the-training-pipeline-for-recsys) will lean on.

### Tiered serving: hot rows fast, cold rows cheap

The memory wall has a structural escape that doesn't touch accuracy at all: exploit the power law in _access frequency_, not just in training frequency. At any given minute, the requests hitting your service touch a tiny, skewed subset of the catalog — the few thousand items that are trending right now. So you don't need the whole 132 GB int8 table in your fastest memory; you need the _hot working set_ there and everything else one tier down. The standard layout is a cache hierarchy: the hottest few percent of rows pinned in accelerator or DRAM, the warm body in host RAM, and the cold tail on local SSD or a remote parameter server reachable over the network. A row's tier is decided by a rolling access counter, exactly like a CPU's LRU cache.

The numbers make the case. Suppose 5% of rows serve 90% of lookups. Pinning that 5% — about 6.6 GB of a 132 GB int8 table — in fast memory means 90% of lookups hit the fast tier at, say, 1 microsecond, and the 10% that miss pay an SSD or network hop at ~100 microseconds. The average lookup latency is $0.9 \times 1 + 0.1 \times 100 = 10.9$ microseconds, while serving 100% from the slow tier would be 100 microseconds — a ~9× speedup for 5% of the memory in fast storage. This is why production embedding stores look like databases with buffer pools, not like a single flat tensor: the access skew is a gift, and tiered caching is how you cash it in without paying a single point of recall. The full feature-store and serving architecture is the subject of [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores).

## 8. The embedding lifecycle: init, regularization, and collapse

Sizing and compressing the table is half the job. The other half is making sure the values _inside_ it learn something useful and don't degenerate. Three lifecycle issues recur in every system.

**Initialization.** Embeddings are usually initialized from a small-variance distribution — $\mathcal{N}(0, \sigma^2)$ with $\sigma \approx 1/\sqrt{d}$, or a uniform range scaled the same way — so that the initial dot products have unit-ish variance and don't saturate the loss. Initialize too large and early scores blow up; too small and gradients are weak and learning crawls. For tables that feed a dot product (retrieval), the initialization scale interacts with the temperature of the softmax loss, so they're tuned together.

**Regularization and the popularity bias in learned norms.** Here is a subtle, important phenomenon. The _norm_ of a learned embedding correlates with how often that ID was updated. A popular item gets thousands of gradient steps, and (depending on the loss) its vector tends to grow a larger norm; a rare item barely moves from its small init. In a dot-product score $\langle \mathbf{u}, \mathbf{v}_i \rangle = \|\mathbf{u}\|\,\|\mathbf{v}_i\|\cos\theta$, a larger $\|\mathbf{v}_i\|$ inflates the score _regardless of direction_ — so popular items get recommended more, which gives them more data, which grows their norm further. That is a feedback loop encoded directly in the embedding norms, a microscopic version of the popularity bias the whole series keeps returning to. Two fixes: **L2 regularization** (weight decay) pulls all norms toward zero, partially equalizing them; and **normalizing embeddings to unit length** before the dot product (cosine similarity instead of raw inner product) removes the norm channel entirely so only direction matters. Many retrieval systems normalize for exactly this reason.

**Embedding collapse.** The failure mode where many distinct IDs converge to nearly the same vector, so the model loses its ability to tell them apart. It shows up as a representation space that's effectively much lower-rank than $d$ — the embeddings live on a thin sliver of the available dimensions. Causes include too-aggressive regularization, a degenerate loss (e.g. all-negative or all-positive batches), an over-large learning rate that homogenizes rows, or simply $d$ being far larger than the true rank of the data so the model has no reason to use the extra dimensions. You diagnose it by computing the **effective rank** of the embedding matrix (the entropy of its singular-value spectrum, or the number of singular values above a threshold): if a $d = 64$ table has effective rank 4, it has collapsed and you're paying for 64 dimensions you don't use. The fix depends on the cause, but contrastive losses with [hard negatives](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) are a reliable way to push embeddings apart and keep the space full-rank.

```python
import torch

def effective_rank(W: torch.Tensor) -> float:
    """Entropy-based effective rank of an embedding matrix."""
    s = torch.linalg.svdvals(W.float())          # singular values
    p = s / s.sum()                               # normalize to a distribution
    entropy = -(p * (p + 1e-12).log()).sum()
    return torch.exp(entropy).item()              # exp(entropy) = eff. rank

healthy = torch.randn(10_000, 64)
collapsed = torch.randn(10_000, 4) @ torch.randn(4, 64)   # true rank 4
print(f"healthy eff-rank: {effective_rank(healthy):.1f}")     # ~64
print(f"collapsed eff-rank: {effective_rank(collapsed):.1f}") # ~4
```

A `nn.Embedding` table that has collapsed will quietly tank your recall while every loss curve looks fine — the loss can be low because the model found a degenerate solution. Monitoring effective rank as a training-health metric is cheap insurance.

### A full retrieval model with a swappable table

Here is the lifecycle wired into one place: a minimal two-tower retrieval model whose item table is a pluggable component, trained with in-batch sampled softmax and L2 regularization. The point of this snippet is that _every_ scheme in this post — full, hashed, QR, mixed-dimension — is a drop-in replacement for one attribute. The rest of the model never changes. This is what makes embedding compression an _engineering_ decision rather than a model rewrite: you swap the table, rerun the eval harness, and read off the Pareto point.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerRetrieval(nn.Module):
    def __init__(self, num_users, item_table: nn.Module, dim=64,
                 normalize=True, temperature=0.05):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = item_table          # full / Hash / QR / MixedDim
        self.normalize = normalize          # cosine -> kills norm-bias channel
        self.temp = temperature
        # Init at sigma ~ 1/sqrt(d) so initial dot products are unit-scale.
        nn.init.normal_(self.user_emb.weight, std=dim ** -0.5)

    def score(self, u, i):
        uv, iv = self.user_emb(u), self.item_emb(i)
        if self.normalize:
            uv, iv = F.normalize(uv, dim=-1), F.normalize(iv, dim=-1)
        return (uv * iv).sum(-1) / self.temp

    def in_batch_loss(self, u, pos_i):
        uv = self.user_emb(u)
        iv = self.item_emb(pos_i)           # the positives ARE the batch
        if self.normalize:
            uv, iv = F.normalize(uv, dim=-1), F.normalize(iv, dim=-1)
        logits = (uv @ iv.t()) / self.temp  # (B, B): every other item is a neg
        labels = torch.arange(u.size(0), device=u.device)
        return F.cross_entropy(logits, labels)

def train_epoch(model, loader, opt, l2=1e-6):
    model.train()
    for users, pos_items in loader:
        opt.zero_grad()
        loss = model.in_batch_loss(users, pos_items)
        # L2 only on the rows we touched, to equalize norms cheaply.
        reg = model.item_emb(pos_items).pow(2).sum(-1).mean()
        (loss + l2 * reg).backward()
        opt.step()

# Swap the table; the model is identical otherwise.
full   = nn.Embedding(27_000, 64)
hashed = HashEmbedding(num_rows=7_000, dim=64)
qr     = QREmbedding(num_ids=27_000, dim=64, num_buckets=170)
model  = TwoTowerRetrieval(num_users=138_000, item_table=qr, dim=64)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Note the in-batch sampled softmax: the positives in the batch serve as each other's negatives via the $(B \times B)$ logit matrix, which is the cheapest way to get a contrastive signal without a separate negative sampler. The L2 term touches only the rows in the batch, keeping the regularization sparse and the norm-equalization cheap — this is the practical fix for the popularity-bias-in-norms phenomenon from earlier in the section. The full theory of these negatives and the $\log Q$ correction lives in [sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval); here, the takeaway is structural: the table is the variable, the training loop is the constant.

## 9. Putting it together: a measured comparison

Theory is nice; let's see the numbers. The setup: a two-tower retrieval model on **MovieLens-20M** (138K users, 27K movies, 20M ratings), trained with in-batch sampled softmax, evaluated with a temporal split (train on the first 90% of each user's history by timestamp, test on the rest — no leakage). The item table is the thing we vary. We measure **Recall@10** (did the held-out item land in the top 10 retrieved?) and the item-table **memory in GB** for a hypothetical scale-up to a 100M-item catalog at $d = 64$, so the memory column reflects the production scenario, not the toy dataset. The accuracy deltas are the kind of differences these schemes produce on MovieLens-scale retrieval; treat them as representative orders of magnitude, not exact published figures, since the precise numbers depend on the loss, the negatives, and the split.

![A five-by-three matrix showing Recall at ten, memory in gigabytes, and the delta versus the full table for each embedding scheme](/imgs/blogs/embeddings-the-heart-of-recommenders-8.png)

| Scheme | Recall@10 | Item-table memory (100M items, d=64) | vs full |
|---|---|---|---|
| Full per-ID table | 0.412 | 25.6 GB | baseline |
| Hashing trick, $m = 0.25\|V\|$ | 0.398 | 6.4 GB | −3.4% |
| Multi-hash ($k=2$), $m=0.25\|V\|$ | 0.404 | 12.8 GB | −1.9% |
| QR ($m=\sqrt{\|V\|}$) | 0.407 | 0.005 GB | −1.2% |
| int8 quantized (full) | 0.409 | 6.4 GB | −0.7% |
| Mixed dimension | 0.410 | ~8 GB | −0.5% |
| QR + int8 | 0.405 | 0.001 GB | −1.7% |

Read this table the way you'd read a Pareto front. A few conclusions jump out:

- **int8 quantization is nearly free.** −0.7% Recall for a 4× memory cut is the easiest win in the table. If you do _one_ thing, quantize the served table to int8. The errors wash out in the dot product, exactly as the $\sqrt{d}$ cancellation argument predicted.
- **QR is the memory champion by orders of magnitude.** Reducing a 25.6 GB table to ~5 MB while losing only ~1.2% Recall is extraordinary. When memory is the binding constraint — when the table doesn't fit on the host at all — QR is the lever that makes the system possible, not just cheaper.
- **Aggressive hashing pays the most in recall.** −3.4% from a single hash at $m = 0.25|V|$ is the worst accuracy hit here, because at that load factor the head items start colliding. Multi-hash recovers most of it ($k=2$ brings it to −1.9%) at 2× the lookups. If you must hash, hash gently (big $m$) or multi-hash.
- **Mixed dimension is the accuracy-preserving compressor.** −0.5% Recall is barely measurable, because the saving comes from the tail you couldn't learn well anyway. It's more implementation work (the per-block projections) but it's the scheme that touches the head items least.

Here's the eval harness that produces Recall@10, so the numbers above are reproducible rather than asserted:

```python
import torch

@torch.no_grad()
def recall_at_k(user_vecs, item_table, test_pairs, k=10):
    """user_vecs: (U, d). item_table: (I, d). test_pairs: list of (u, i_true)."""
    item_norm = torch.nn.functional.normalize(item_table, dim=1)
    hits = 0
    for u, i_true in test_pairs:
        uq = torch.nn.functional.normalize(user_vecs[u], dim=0)
        scores = item_norm @ uq                  # (I,) cosine scores
        topk = torch.topk(scores, k).indices
        hits += int(i_true in topk)
    return hits / len(test_pairs)

# Swap item_table for the int8-dequantized / QR-reconstructed table to
# measure each scheme on the SAME users and SAME test pairs.
```

The discipline that makes this trustworthy: same users, same test pairs, same $k$, only the item table swapped. A temporal split so you never test on an interaction that came before a training one. And — critically — you compute Recall over the _full_ item set, not a sampled subset of negatives. The KDD 2020 result by Krichene and Rendle ("On Sampled Metrics for Item Recommendation") showed that sampled metrics can _reorder_ which model looks best, so for a comparison this fine-grained you evaluate against the whole catalog or you don't trust the ranking.

## 10. Case studies: how the big systems do it

These ideas aren't academic — they're load-bearing in systems serving billions of requests a day.

**DLRM (Naumov et al., Facebook, 2019), "Deep Learning Recommendation Model for Personalization and Recommendation Systems."** DLRM is the canonical "embeddings are the model" architecture: dense features go through a small MLP, every categorical feature goes through its own embedding table, the lookups and the dense vector get combined by explicit pairwise dot-product interactions, and a final MLP produces the click probability. The paper is explicit that the embedding tables dominate both parameters and the systems challenge — DLRM's training infrastructure is essentially a distributed embedding store with a small neural network attached. It's the reference point for why the table, not the MLP, is the engineering problem.

**QR embeddings (Shi et al., Facebook, 2020), "Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems."** This is the QR scheme from Section 4, evaluated on the Criteo Ad Kaggle and Terabyte datasets. They report that compositional (quotient-remainder) embeddings reach accuracy comparable to the full table at large compression factors — making it feasible to train DLRM-class models whose full embedding tables would not otherwise fit. The complementary-partitions framing generalizes beyond plain quotient-remainder to any pair of partitions that are jointly unique.

**Mixed-dimension embeddings (Ginart et al., Stanford/Facebook, 2021).** The popularity-based dimension allocation from Section 6, showing on Criteo and MovieLens that frequency-sized embeddings match or beat uniform-$d$ tables at a fraction of the memory. The key empirical point: nearly all the wasted capacity in a uniform table is in the tail, so shrinking the tail is almost free in accuracy.

**Feature hashing (Weinberger et al., 2009), "Feature Hashing for Large Scale Multitask Learning."** The paper that put the hashing trick on a rigorous footing, predating deep recommenders entirely. Its key theoretical contribution is a bound showing that the inner products you care about are preserved in expectation under hashing — the hash introduces a small, controllable variance rather than a systematic bias, especially with the signed-hash variant that randomly flips the sign of each contribution to cancel collision cross-terms. That result is _why_ the hashing trick works at all in a dot-product model: the collisions add noise that the learning can largely absorb, not a bias that corrupts the geometry. Every hashed embedding table in production is standing on this analysis.

**Industrial embedding compression at Google and Meta.** Both companies have published on serving billion-row embedding tables under memory pressure: learned quantization, hashing with collision-aware training, and hierarchical caching of hot rows in fast memory with cold rows on slower tiers. Google's work on "Deep Hash Embeddings" and Meta's production DLRM stack (and the TorchRec library that open-sources much of it) treat the embedding store as a tiered-memory database problem — the same instincts a systems engineer brings to a key-value store, applied to a learnable table. Meta has reported DLRM-class models whose embedding tables run into the trillions of parameters, which only exist because of the compression and sharding machinery in this post — a trillion-parameter fp32 table would be 4 terabytes, far past any single host, so the table is sharded across a fleet and compressed per shard. The recurring theme across all of them: **the embedding table is infrastructure, and you engineer it like infrastructure.** The full feature-store and large-scale serving picture is the subject of [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores).

## 11. Stress-testing the choices

A decision is only as good as how it survives the awkward cases. Let's stress-test the embedding strategy against the situations that actually break it.

**What happens at 100M items with only implicit feedback?** Implicit feedback (clicks, views) is positive-only and noisy, so most IDs are seen rarely and the tail is enormous. Frequency filtering becomes essential — you simply cannot learn 100M item vectors from clicks alone, because the median item has near-zero clicks. The realistic recipe is: frequency-filter to the active head (a few million items), give that head full or QR embeddings, and route the cold tail to OOV buckets or content-based features. Trying to give all 100M IDs trainable rows wastes memory on vectors that never receive a meaningful gradient.

**What happens when collisions hit a head item?** A single hash that happens to map two _popular_ items to the same row is the worst case — both get recommended to each other's audiences, and you'll see it as a strange cross-contamination in recommendations. The defenses, in order of preference: make $m$ large enough that the head's load factor is tiny; use multi-hash so a single clash isn't fatal; or, best, frequency-filter the head into a _collision-free_ dense region and only hash the tail. Never let the head collide.

**What happens when offline Recall rises but online engagement is flat?** This is the offline–online gap the series keeps circling, and embeddings have a specific failure mode here. If you grew $d$ or removed compression and offline Recall went up, but online didn't move, suspect that the extra capacity is memorizing _historical popularity_ rather than learning generalizable taste — the bigger table fits the training distribution's popular items better, but those were already being recommended. The honest test is whether the lift shows up on _tail_ and _fresh_ items, where memorization can't help. This is exactly the kind of divergence the [offline-vs-online post](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) dissects.

**What happens when the offline hash and the online hash disagree?** This is the silent killer. If your training pipeline hashes IDs with Python's built-in `hash()` (which is salted per-process and non-deterministic across runs unless you set `PYTHONHASHSEED`) but your serving layer uses a different hash, then _every embedding lookup at serve time reads the wrong row_. The model trained on row $h_{\text{train}}(i)$ but serves from row $h_{\text{serve}}(i) \neq h_{\text{train}}(i)$ — pure noise. Precision halves and nobody can find the bug because both pipelines "work." The fix is non-negotiable: use a fixed, language-agnostic hash (a named algorithm like xxHash or MurmurHash with a pinned seed), test that train and serve produce byte-identical row indices for the same ID, and assert it in CI.

**What happens when a new ID arrives that was never in training?** It has no trained row. Under hashing it lands on _some_ row (probably a poorly-fit collision); under a full table it hits an uninitialized or OOV row. Neither is good. The standard mitigations: route new IDs to the OOV bucket until they accumulate data; back off to content-based features (text, image, category embeddings from a [content model](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders)) that don't need an ID-specific row; or warm-start the new row from similar items' average. Cold-start is fundamentally a "no embedding yet" problem, and recognizing it as such tells you the fix is "get a reasonable vector without ID-specific training," not "train harder."

The warm-start tactic is worth a few lines of code, because it's the difference between a brand-new item being invisible and being recommendable on day one. When a new item appears, you usually know _something_ about it — its category, its creator, a text description. Average the embeddings of items sharing that attribute and seed the new row with the result, so the new item inherits a sensible position in the geometry before it has earned a single interaction:

```python
import torch

@torch.no_grad()
def warm_start_new_item(item_table, new_row_idx, sibling_idxs):
    """Seed a fresh item's embedding from the mean of similar items."""
    if len(sibling_idxs) == 0:
        return                                  # nothing to copy; leave at init
    siblings = item_table.weight[sibling_idxs]  # (k, d) trained vectors
    item_table.weight[new_row_idx] = siblings.mean(dim=0)

# A new sci-fi movie inherits the average vector of existing sci-fi movies.
warm_start_new_item(full, new_row_idx=26_999,
                    sibling_idxs=torch.tensor([12, 88, 1193, 5001]))
```

The half-life of this scaffold matters: once the new item gathers real interactions, gradient descent will move its row away from the seed toward where its actual behavior puts it. The warm start only has to be _good enough_ to get the item shown a few times so it can start learning. That's the whole game with cold IDs — buy time until the ID has data, then let the table do its job.

## 12. Managing your embedding tables

Pulling the operational practice into one checklist, because the table will outlive several model architectures and you'll be maintaining it for years:

- **Size it on a whiteboard before you write code.** $|V| \times d \times \text{bytes}$, times 3 for Adam, is the number that decides whether you need sharding. Do this arithmetic _first_, every time. The 41 GB table that paged my host would have been a one-line calculation a week earlier.
- **Frequency-filter aggressively.** Drop IDs below a count threshold to a shared OOV bucket before considering any clever compression. It's the cheapest 80–95% row reduction you'll ever get, and it doubles as your cold-start mechanism.
- **Pick $d$ by a sweep, then question whether it should be uniform.** Find the recall knee, then ask whether the tail deserves the same $d$ as the head. Usually it doesn't — mixed dimension or QR for the tail is nearly free.
- **Quantize the served table to int8 by default.** 4× memory for a fraction of a point of recall, validated by the $\sqrt{d}$ error-cancellation argument. Reach for PQ only when int8 isn't small enough and you've measured you can afford the larger hit.
- **Pin the hash.** A named hash, a fixed seed, a CI test that train and serve agree byte-for-byte. This single discipline prevents the most insidious production failure in this whole post.
- **Monitor effective rank and norm distribution.** Effective rank catches collapse; the norm histogram catches popularity bias creeping into the geometry. Both are cheap to compute on a sampled subset every epoch and both fail _silently_ in the loss curve.
- **Normalize before the dot product if popularity bias is hurting.** Cosine instead of raw inner product removes the norm channel that the popularity feedback loop exploits.
- **Treat the table as infrastructure.** Snapshot it, version it, know its staleness at serve time, and shard it like the distributed database it is. The model code is the part you rewrite; the embedding store is the part you operate.

## When to reach for this (and when not to)

Every scheme in this post is a cost. Plainly:

- **Reach for a full per-ID table when it fits.** If your catalog is in the low millions and $|V| \times d \times 4$ comfortably fits host RAM with room for Adam, just use `nn.Embedding` and move on. Compression you don't need is complexity you'll regret. Don't QR a 2M-item catalog — the full table is a few hundred MB.
- **Reach for the hashing trick when the vocabulary is open or unbounded** and you can tolerate some collisions — but size $m$ generously and never let the head collide. Don't hash aggressively (small $m$) on a catalog where head-item precision matters; you'll bleed recall exactly where it's most visible.
- **Reach for QR when memory is the hard wall** — when the full table literally doesn't fit and you've confirmed the task doesn't need to distinguish all IDs at fine granularity. Don't reach for QR's extreme compression if a sweep shows your recall is genuinely rank-limited; you'll trade away accuracy you needed.
- **Reach for mixed dimension when you have a strong power-law catalog** and want to shrink the tail without touching the head. Don't bother if your IDs are near-uniform in frequency — there's no tail to shrink, and the per-block projection overhead isn't worth it.
- **Reach for int8 quantization almost always at serve time.** It's the highest-leverage, lowest-risk lever in the post. Don't reach for int4/PQ until you've measured that int8 isn't small enough _and_ that you can afford the larger recall hit.
- **Reach for frequency filtering before any of the above.** It's the simplest, most interpretable compression and it's almost always the right first move. Don't allocate trainable rows for IDs you can't learn — you're paying memory for noise.

## Key takeaways

1. **An embedding lookup is a one-hot times a matrix** — an exact algebraic identity, implemented as an $O(d)$ gather. The table is a learned dictionary from IDs to vectors.
2. **Embeddings hold 99%+ of a deep recommender's parameters.** The MLP is a rounding error. Size the table first; it decides your whole systems design.
3. **The hashing trick caps memory by accepting collisions**, governed by the load factor $\alpha = n/m$ via $P(\text{collision}) \approx 1 - e^{-\alpha}$. Compute $\alpha$ over the IDs that carry signal, not the raw vocabulary.
4. **QR embeddings give unique vectors from two tiny tables**, reducing memory by $\sqrt{|V|}/2$ — thousands-fold — at a small accuracy cost. The memory champion when the table doesn't fit.
5. **A uniform dimension is wrong for power-law catalogs.** Mixed dimension spends width on the head and starves the tail, matching uniform accuracy at a fraction of the memory.
6. **int8 quantization is nearly free** — 4× smaller for a fraction of a point of recall — because per-element rounding errors cancel in the dot product as $\sqrt{d}$ versus $d$.
7. **Frequency-filter before you compress.** Routing rare IDs to a shared OOV bucket removes most rows for almost no signal and doubles as cold-start handling.
8. **Embeddings fail silently.** Watch effective rank for collapse, the norm histogram for popularity bias, and pin the hash so train and serve read the same rows. None of these show up in the loss curve.
9. **The embedding store is infrastructure.** Shard it, version it, snapshot it, and engineer it like the distributed database it is — because that's what it is.

## Further reading

- Naumov et al. (2019), *Deep Learning Recommendation Model for Personalization and Recommendation Systems* — the canonical DLRM paper; the reference for why embeddings dominate the systems problem.
- Shi et al. (2020), *Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems* — the QR / quotient-remainder embedding scheme.
- Ginart et al. (2021), *Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems* — popularity-based mixed-dimension allocation.
- Weinberger et al. (2009), *Feature Hashing for Large Scale Multitask Learning* — the original hashing-trick analysis and collision bounds.
- Krichene & Rendle (2020, KDD), *On Sampled Metrics for Item Recommendation* — why you evaluate Recall over the full catalog, not sampled negatives.
- PyTorch docs: `torch.nn.Embedding` and `torch.nn.EmbeddingBag`; the **TorchRec** library for sharded production embedding tables.
- Within this series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) (the funnel + feedback-loop frame), [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders), [matrix factorization the workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

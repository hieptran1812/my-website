---
title: "The Cold Start Problem in Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Diagnose the three cold starts — new item, new user, new system — derive why an untrained ID embedding is useless, and build a content-augmented two-tower plus a popularity fallback and an exploration loop that close the cold-start Recall@10 gap on MovieLens."
tags:
  [
    "recommendation-systems",
    "recsys",
    "cold-start",
    "two-tower",
    "content-features",
    "exploration",
    "meta-learning",
    "lightfm",
    "machine-learning",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-cold-start-problem-1.png"
---

A streaming service ships forty new titles on a Friday afternoon — fresh releases, the weekend tentpoles, the exact catalog the homepage exists to surface. By Monday the analytics tell a depressing story: the new titles got almost no impressions. The collaborative filtering model with the beautiful offline Recall@10 has no useful row in its embedding table for a film nobody has clicked. It cannot recommend what it has never seen co-watched. The forty titles the business most wanted to promote are invisible to the very system built to promote them.

Now flip the lens to a person instead of an item. A brand-new user installs the app, opens it for the first time, and the recommender knows exactly nothing about them — no clicks, no watches, no ratings. Every model we have spent this series building reads taste off behavior, and this user has produced none. The recommender does the only thing it can: it serves the global top ten, the same ten it serves every other empty profile, which is the least personalized screen the user will ever see and the one most likely to make them bounce. And in the most extreme version, you are launching the whole product tomorrow with no users and no items that anyone has touched — a system that must somehow produce decent recommendations from a standing start with no interaction data at all.

These three scenarios — a cold item, a cold user, a cold system — are the **cold-start problem**, and it is the single most universal applied-recommender headache there is. Every recommender that has ever shipped has hit it, because catalogs always grow, users always arrive, and systems always launch. It is also the place where the offline-online gap bites hardest: your held-out test set is full of items and users that already have history, so your offline metric quietly ignores the exact population that cold start is about. You can score a glorious NDCG on warm data while the new-item experience is a wasteland.

This post sits in the **Recommendation Systems: From Click to Production** series and does the three things the series always does. *Scientific*: we prove why an untrained ID embedding is literally useless — it sits at its random initialization because no gradient has ever touched it — and we show how a content tower supplies a learned prior so a never-seen item lands in a sensible place, plus the meta-learning objective for fast adaptation and the explore-cost regret math. *Practical*: we build a content-augmented two-tower (and a LightFM variant with item features), a popularity-plus-recency fallback, and an epsilon-greedy exploration loop, with an eval harness that separates **warm** from **cold-start** Recall@10. *Measured*: a before-and-after table — ID-only versus content versus content-plus-exploration — on warm and cold Recall@10 on MovieLens, with the honest result that content closes most of the cold gap at no warm cost and exploration closes most of the rest.

The figure below is the whole map: cold start is not one problem but three, split by what is missing, and each branch needs a different family of tactics. Keep it in mind — the rest of the post walks each branch down to running code.

![Taxonomy tree splitting cold start into new item, new user, and new system branches with each branch routed to its own family of fixes such as content tower, onboarding, and popularity bootstrap](/imgs/blogs/the-cold-start-problem-1.png)

## 1. Three cold starts, not one

The first mistake engineers make with cold start is treating it as a single problem with a single fix. It is three problems with three causes, and a tactic that solves one does nothing for another. Onboarding questions cure a cold *user* and are irrelevant to a cold *item*. A content tower cures a cold *item* and does not personalize a cold *user* who has stated no preferences. So before any code, get the taxonomy crisp.

**New item.** An item has just entered the catalog. It has rich attributes — a title, a description, a category, a price, a poster, an artist — but zero interactions. In a pure collaborative-filtering or ID-based model, that item owns a row in the embedding table that has never received a gradient, so its vector is still whatever the random initializer produced. It is invisible to retrieval because its embedding does not point anywhere meaningful, and invisible to ranking because the ranker's item features are dominated by interaction-derived signals it does not have. This is the new-item cold start, and it is the most common one in practice because catalogs churn constantly.

**New user.** A user has just arrived. The system may know a little context — device, locale, time of day, maybe how they landed — but it has no interaction history for them. Collaborative filtering cannot place them in the user-factor space because that space is *defined* by interactions. The model can only fall back to non-personalized signals: global popularity, trending, or whatever weak prior the context gives. This is the new-user cold start.

**New system.** The whole product is launching. There are no users with history and no items with interactions. There is no interaction matrix to factorize, no logs to train on, nothing for collaborative filtering to learn from. You must bootstrap the entire feedback loop — serve something reasonable, log the interactions, and only then begin to train the models the rest of this series describes. This is the new-system cold start, and it is really the new-item and new-user problems happening simultaneously to *everything*, plus the meta-problem of having no trained model at all.

The reason this distinction matters operationally is that the three have different time constants and different owners. New items arrive continuously and warm up over hours to days; the fix lives in your retrieval and ranking models and your exploration policy. New users arrive continuously too but warm up within a single session if you are clever about onboarding and session signals; the fix lives in your onboarding flow and your sequence model. New systems happen once; the fix is a launch plan that transitions from heuristics to content to CF as data accrues. Conflating them produces the classic failure where a team builds a beautiful content tower, congratulates itself on solving cold start, and then watches new-user retention stay flat because the content tower never addressed the user side at all.

> **The one-line test.** Ask "what is missing — the item's vector, the user's vector, or both, and the model itself?" New item: the item vector. New user: the user vector. New system: everything. The answer tells you which branch of the tree you are on, and the branch tells you which tactics are even relevant.

We will spend the bulk of the post on the new-item case because it is the most common and because its core insight — that *content provides a prior where behavior provides none* — generalizes to the other two. Then we cover the user and system cases, and finally the warm-up dynamics and exploration tension that tie all three together.

## 2. Why an untrained ID embedding is useless

Let us make the central pathology rigorous, because once you see it the whole zoo of fixes follows. The standard recommender represents each item $i$ by a learned vector $\mathbf{e}_i \in \mathbb{R}^d$ stored in an embedding table $E \in \mathbb{R}^{|I| \times d}$. Retrieval scores a (user, item) pair by an inner product $s(u, i) = \mathbf{u}^\top \mathbf{e}_i$, and ANN retrieval returns the items with the largest scores. The embedding $\mathbf{e}_i$ is initialized randomly — typically from a normal or uniform distribution scaled to keep activations sane, for example $\mathbf{e}_i \sim \mathcal{N}(0, \sigma^2 I)$ with $\sigma$ around $0.01$ to $0.1$ — and then *learned by gradient descent on the training interactions*.

Here is the problem stated as math. The gradient of any standard loss with respect to a particular item embedding is non-zero only when that item appears in a training example. For a sampled-softmax or BPR loss over interactions $\mathcal{D}$, the update to $\mathbf{e}_i$ is

$$
\Delta \mathbf{e}_i \;=\; -\eta \sum_{(u, j) \in \mathcal{D}} \frac{\partial \ell(u, j)}{\partial \mathbf{e}_i}.
$$

If item $i$ never appears as a positive and is never sampled as a negative — which is exactly the case for a brand-new item with zero interactions — then *every term in that sum is zero*. The gradient is identically zero. After all of training, $\mathbf{e}_i$ is bit-for-bit the random vector the initializer wrote. It encodes no information about the item whatsoever; it is pure noise with respect to taste.

What does a noise vector do at serving time? Its inner product with any user vector is approximately a sample of $\mathbf{u}^\top \mathbf{z}$ where $\mathbf{z}$ is random, which has mean zero and variance $\sigma^2 \lVert \mathbf{u} \rVert^2 / d \cdot d = \sigma^2 \lVert \mathbf{u} \rVert^2$. In expectation the cold item scores no better than a coin flip against the warm items whose vectors were pulled toward the users who like them. So in the ranked list it lands wherever its random draw happens to put it — usually buried, occasionally and accidentally near the top, never *because* it matches the user. Its expected position is the middle of the catalog, and a cold item in the middle of a hundred-thousand-item catalog is, for all practical purposes, invisible. That is the new-item wall in one equation: no gradient, no signal, no recommendation.

This is also why the naive "just retrain nightly" idea does not fix new-item cold start on its own. Retraining helps an item that has *accumulated some interactions* since the last train, but it does nothing for the item added five minutes ago, and in a fast catalog the freshest items are exactly the ones you most want to surface. You cannot wait for the next train; you need the item to be recommendable the instant it exists.

There is a subtler corollary worth spelling out, because it explains why a content tower *generalizes* rather than merely memorizing. The ID embedding table has $|I| \times d$ free parameters and learns one independent vector per item, so each item's vector is estimated only from that item's own interactions — a cold item, with no interactions, gets no estimate. A content tower $f_\theta$ has a *fixed* parameter count $|\theta|$ that does not grow with the catalog, and every interaction on *every* item updates the *same* $\theta$. The tower is therefore estimated from the entire interaction corpus pooled across items, and what it learns is the *mapping from content to taste-space position*, not the position of any one item. A new item shares features — a genre token, a director, a topic embedding direction — with thousands of warm items the tower trained on, so $f_\theta(\mathbf{x}_{\text{new}})$ inherits the position those shared features earned. This is ordinary supervised generalization: the tower is a regression from features to embedding, and a new item is just a new point in feature space near training points. The ID table cannot generalize because each item is its own parameter with no shared structure; the tower can because content is the shared structure. That single architectural difference — parameters that scale with the catalog versus parameters that scale with the feature vocabulary — is the whole reason content beats ID on cold items.

This reframing also exposes a serving trap that has paged more than one team: **train-serve feature skew on the cold path**. The content tower is only safe on a new item if the features you feed it at serving time are computed *identically* to training. If training used a clean, fully-populated synopsis but the freshly-ingested item arrives with a null synopsis (the editor has not filled it in yet), or the genre tags are encoded with a different vocabulary offline versus online, the tower receives a different $\mathbf{x}_i$ than it was trained on and produces a garbage vector — and you will not notice, because the cold-recall metric is computed offline with the *clean* features. The discipline: compute the content features in *one* shared pipeline used by both training and serving, validate that a brand-new item's feature row is non-degenerate before upserting its vector, and have the popularity fallback catch any item whose features are too sparse to embed reliably. A content tower with a feature-skew bug is worse than no content tower, because it serves confident nonsense instead of an honest fallback.

The figure below contrasts the two regimes directly: the ID-only model leaves the new item at its random init and effectively invisible, while a content tower maps the item's features into a sensible place among similar warm items on day zero.

![Two-column before and after figure contrasting an ID-only model where a new item is a random noise embedding with near-random cold recall against a content two-tower where the item's features map to a vector placed near similar warm items with usable cold recall](/imgs/blogs/the-cold-start-problem-2.png)

#### Worked example: the cold gap on MovieLens

Take MovieLens-20M and build the canonical evaluation that *actually measures cold start*. Sort interactions by timestamp. Hold out the last 10% as test. Critically, partition the test items into **warm** items (which appear in the training period) and **cold** items (which first appear only in the test period). Now train a standard matrix-factorization or two-tower model on the training interactions and evaluate Recall@10 separately on warm-test and cold-test users-items.

A well-tuned ID-only model on MovieLens lands around **Recall@10 ≈ 0.40** on warm items — strong, the kind of number that gets a model shipped. On the cold-test items, the same model scores **Recall@10 ≈ 0.02**, essentially the random baseline of $10 / |I| \times \text{(catalog hit rate)}$. The cold gap is roughly $0.40 - 0.02 = 0.38$, and it is *entirely invisible* if you report a single pooled Recall@10, because warm items dominate the test set. This is the trap: the pooled metric reads $\approx 0.37$ and looks healthy while the new-item experience is a wasteland. The fix begins with splitting the metric.

## 3. The cold-start type matrix: matching the fix to the cause

Before we build, lay the three cases side by side with their cause, their best tactic, and their fallback. The discipline here is that the *cause* dictates the *tactic*: a noise ID embedding is cured by supplying a non-noise prior (content), a missing user vector is cured by eliciting or inferring preferences (onboarding, session, context), and a missing model is cured by bootstrapping with heuristics until data accrues. The fallback column is what you ship to keep coverage while the real fix warms up — never zero recommendations, always a popularity or recency backstop.

![Matrix figure mapping each cold-start type to its root cause, its best tactic, and its cheap fallback, showing new item caused by a noise embedding fixed with a content tower and backed by popularity and recency](/imgs/blogs/the-cold-start-problem-3.png)

Here is the same mapping as a table you can act on:

| Cold-start type | Root cause | Best tactic | Cheap fallback |
| --- | --- | --- | --- |
| New item | ID embedding never got a gradient | Content/text features into the item tower | Popularity + recency within category |
| New user | No interaction history | Onboarding picks + session encoder | Demographic/context prior |
| New system | No interaction data anywhere | Content-only model + exploration | Trending heuristics, curated lists |
| Cross-cutting | New items never get shown | Explicit exploration budget | Reserve N slots for cold items |

The cross-cutting row is the one teams forget. Even with a perfect content tower, a cold item only *warms* if it gets impressions, and a greedy exploit-only ranker will rarely show a brand-new item because its predicted score (from content alone, before any interaction confirmation) is usually a touch below the battle-tested warm items. Without a deliberate exploration budget, the content tower gives the item a vector but the policy never gives it a chance. We return to this in the exploration section; for now, note that the matrix has a hidden fourth dimension — *the serving policy* — and the content models below assume an exploration loop sits on top of them.

A note on the fallbacks. "Popularity" sounds crude and it is, but a popularity-plus-recency fallback that is *scoped to the right context* is shockingly strong as a backstop: trending in the user's preferred genre, popular among users in the same region, the most-watched releases from the last seven days. It will never be your differentiator, but it guarantees the system never serves an empty or absurd screen, and it is the floor that everything else must beat. Build it first; it is your control arm for every cold-start experiment.

## 4. Item cold start: content features and the two-tower prior

The cure for the new-item wall is to stop relying on the item's *ID* and start relying on the item's *content*. A new movie has no clicks, but it has a title, a synopsis, a genre set, a cast, a release year, a poster — none of which require a single interaction to exist. If we can map that content to the same embedding space the user vectors live in, the new item is recommendable the instant it lands. This is the deep connection to [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders): content-based filtering is not just an alternative to CF, it is the *cold-start insurance* for CF.

The cleanest way to bolt content onto a modern retrieval stack is the **two-tower model with a content-based item tower**. Recall from [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) that the architecture has a user tower producing $\mathbf{u}$ and an item tower producing $\mathbf{e}_i$, trained so that $\mathbf{u}^\top \mathbf{e}_i$ is high for positive pairs. In the ID-only version, the item tower is literally a lookup: $\mathbf{e}_i = E[i]$. The cold-start fix is to make the item tower a *function of content features* instead:

$$
\mathbf{e}_i \;=\; f_\theta\big(\mathbf{x}_i\big), \qquad \mathbf{x}_i = \text{(text emb, genre, year, } \dots),
$$

where $f_\theta$ is a small network shared across all items. Now the embedding of any item is *computed from its features*, so a never-seen item with features gets a vector immediately. The tower learned, during training on warm items, the mapping "this kind of content goes near these kinds of users," and that mapping generalizes to a new item because the new item shares features (a genre, a director, a topic) with items the tower has seen. The new item is placed near its content-neighbors, which is exactly where its likely audience already shops.

The figure below shows the dataflow: raw item content fans into a feature encoder, the item tower merges the features into one vector, and that vector lands in the shared space next to similar warm items, where ANN retrieval finds it.

![Branching dataflow graph showing item text and metadata features feeding a feature encoder and item tower that merge into a shared embedding space where the new item lands near warm neighbors and is found by approximate nearest neighbor retrieval](/imgs/blogs/the-cold-start-problem-4.png)

### Building a content-augmented item tower in PyTorch

Here is a content item tower that combines a precomputed text embedding (from `sentence-transformers`) with categorical metadata embeddings and a small MLP. The user tower stays ID-based for warm users; we tackle cold users separately. The key structural choice: the item tower never looks up an ID, so it has no cold path to fail on.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentItemTower(nn.Module):
    """Maps item CONTENT (no ID lookup) to a d-dim embedding.
    A brand-new item with features gets a vector immediately."""
    def __init__(self, text_dim=384, n_genres=20, n_years=120, d=128):
        super().__init__()
        # text_dim = sentence-transformers all-MiniLM-L6-v2 output (384)
        self.genre_emb = nn.EmbeddingBag(n_genres, 32, mode="mean")
        self.year_emb = nn.Embedding(n_years, 16)
        in_dim = text_dim + 32 + 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, d),
        )

    def forward(self, text_vec, genre_ids, genre_offsets, year_id):
        g = self.genre_emb(genre_ids, genre_offsets)   # (B, 32)
        y = self.year_emb(year_id)                      # (B, 16)
        x = torch.cat([text_vec, g, y], dim=-1)
        e = self.mlp(x)
        return F.normalize(e, dim=-1)                   # unit norm for cosine

class UserTower(nn.Module):
    def __init__(self, n_users, d=128):
        super().__init__()
        self.emb = nn.Embedding(n_users, d)
    def forward(self, user_id):
        return F.normalize(self.emb(user_id), dim=-1)
```

The training loop is the ordinary in-batch-negatives sampled-softmax loop from [training the two-tower with negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax), with one twist: because the item tower consumes features, you can feed it *any* item, including one minted after training. The forward pass is identical for warm and cold items, which is the whole point.

```python
def two_tower_step(user_tower, item_tower, batch, temperature=0.05):
    u = user_tower(batch["user_id"])                          # (B, d)
    e = item_tower(batch["text_vec"], batch["genre_ids"],
                   batch["genre_offsets"], batch["year_id"])  # (B, d)
    logits = (u @ e.t()) / temperature                        # (B, B) in-batch
    labels = torch.arange(u.size(0), device=u.device)         # diagonal = positives
    loss = F.cross_entropy(logits, labels)                    # sampled softmax
    return loss
```

A few production notes that save you a paging incident. First, precompute and cache the `sentence-transformers` text embedding offline; do not run a transformer inside the serving hot path for every item. The text embedding is a static feature of the item's description, so compute it once at ingest and store it. Second, when you add a new item to the catalog, you run its features through the item tower once, get its vector, and upsert it into the ANN index — no retrain required. This is the operational superpower of a content item tower: new items become retrievable in the time it takes to embed and upsert, typically seconds, not the hours of a training cycle.

### The LightFM alternative: features sum into the factor

If you do not want to stand up a PyTorch two-tower, `lightfm` gives you the same cold-start property with far less code, because its model *is* a content-augmented factorization. In LightFM each item is represented as the sum of the embeddings of its features, so an item with features is never cold even if its own ID-feature has never been seen.

```python
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np

ds = Dataset()
ds.fit(users=user_ids,
       items=item_ids,
       item_features=all_feature_tokens)  # genre:action, year:2026, etc.

interactions, _ = ds.build_interactions(
    (u, i) for (u, i) in train_pairs)
item_feats = ds.build_item_features(
    (i, feats) for (i, feats) in item_feature_map.items())

model = LightFM(no_components=128, loss="warp")  # WARP ~ ranking loss
model.fit(interactions, item_features=item_feats,
          epochs=30, num_threads=8)

# Score a BRAND-NEW item: it was never in training, but its FEATURES were.
# Build its feature row and pass item_features at predict time.
scores = model.predict(user_id_internal,
                       np.array([new_item_internal_id]),
                       item_features=new_item_feats)
```

The mechanism is worth stating precisely because it is the same insight as the two-tower, just linear. LightFM represents item $i$'s latent vector as $\mathbf{q}_i = \sum_{f \in F_i} \mathbf{q}_f$, the sum over its feature embeddings $\mathbf{q}_f$. A new item shares features (genre, year, tags) with warm items, so its $\mathbf{q}_i$ is a sum of feature vectors that *were* trained, placing it sensibly. The "item ID" itself is just one more feature; drop it for items you expect to be cold and the model leans entirely on shared content features. For more on this fusion of content into latent factors, see [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders).

## 5. LLM and text embeddings as a cold-start prior

The content tower is only as good as the features you feed it, and for items whose primary content is text — articles, products, listings, courses — the strongest off-the-shelf feature is a high-quality text embedding. A `sentence-transformers` model or an LLM embedding turns a title and description into a 384- or 768-dimensional vector that already encodes semantic similarity: two articles about the same event land near each other in that space *before your recommender ever trains*. That is a cold-start prior handed to you for free, and it is why text-embedding-based retrieval is the default first move for news and content platforms where everything is cold.

The connection to [LLMs for recommendation](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec) is direct. An LLM can do more than embed: prompted with an item's metadata it can generate a structured description, normalize messy catalog text, or even produce a synthetic "this item is for people who like..." profile that becomes a feature. The simplest and most robust version, though, is plain dense text embeddings used as the item tower's input — exactly the `text_vec` in the PyTorch tower above. Two practical guidelines:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-d, fast, cheap

def item_text(meta):
    # Concatenate the fields that carry taste signal, most-discriminative first.
    return f"{meta['title']}. {meta['genres']}. {meta['synopsis']}"

# Compute ONCE at ingest, cache, never recompute on the hot path.
text_vecs = encoder.encode([item_text(m) for m in catalog_meta],
                           batch_size=256, normalize_embeddings=True)
```

First, *what you concatenate matters more than which model you pick*. Put the most discriminative fields first (title, category, key attributes) because longer descriptions get diluted, and drop boilerplate (legal text, generic marketing copy) that makes everything look the same. Second, normalize the embeddings and treat the text vector as a strong but not infallible prior — the learned item tower on top of it should be free to reweight and combine it with metadata, because raw semantic similarity is not the same as taste similarity (two films can be textually similar and appeal to opposite audiences). The tower learns that correction from warm interactions and transfers it to cold items.

#### Worked example: a new article gets a usable embedding

A news app ingests a breaking story at 14:00. It has a headline, a 200-word body, a section tag (World), and an author. There are zero clicks. The pipeline: at ingest, the `all-MiniLM-L6-v2` encoder turns "headline + section + first 100 words" into a 384-d vector in about 8 ms on CPU; that vector plus the section and author embeddings flow through the item tower to a 128-d retrieval vector; the vector is upserted into the HNSW index at 14:00:01. The first user who opens the app at 14:00:30 and has read three other World stories today now has that fresh article in their candidate set, ranked by the dot product of their user vector against the article's *content-derived* vector. The article was published 30 seconds ago and is already personalizable — no interaction data, no retrain, just content through the tower. Compare to the ID-only world where this article would not be recommendable until the next nightly train, by which time the news is stale. This is why "everything is cold" platforms live and die on the quality of their content embeddings.

## 6. Meta-learning: learning to warm up fast

Content features give a cold item a *reasonable* vector, but a reasonable vector is not the same as the well-tuned vector the item would have after a few hundred interactions. Can we do better than the content prior using the *handful* of interactions a slightly-warm item accumulates in its first hour? This is the domain of **meta-learning for cold start**, and the two canonical methods are worth understanding even if you do not deploy them, because they sharpen the way you think about warm-up.

The meta-learning framing is: instead of learning a single fixed embedding per item, learn a *procedure* that produces a good embedding from a few interactions — "few-shot adaptation." The objective is not to minimize loss on the training items directly, but to minimize the loss *after* a few gradient steps on a small support set, averaged over many simulated cold-start tasks. Formally, with model parameters $\theta$, a cold-start task draws a support set $S$ (the item's first few interactions) and a query set $Q$; the meta-objective is

$$
\min_\theta \; \mathbb{E}_{\text{task}} \Big[\, \mathcal{L}_Q\big(\theta'\big) \Big], \qquad
\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_S(\theta),
$$

which is the MAML objective: find an initialization $\theta$ such that one or a few gradient steps on the support set $S$ yield low loss on the query set $Q$. Applied to recommendation, **MeLU** (Lee et al., KDD 2019) treats each user as a task and learns an initialization for the personalization parameters that adapts to a new user from a handful of rated items in a few steps. **DropoutNet** (Volkovs et al., NeurIPS 2017) takes a complementary route: during training it *randomly zeroes the interaction-derived input* (input dropout on the preference features) so the network is forced to predict the embedding from content alone, which is exactly the cold-start condition. At serving time a cold item has no interaction input, which DropoutNet has been explicitly trained to handle.

Here is the DropoutNet idea in a few lines — the trick is dropout on the *preference branch*, not the usual dropout on hidden units:

```python
class DropoutNetItem(nn.Module):
    def __init__(self, pref_dim, content_dim, d=128, p_cold=0.5):
        super().__init__()
        self.p_cold = p_cold  # prob of simulating a cold item per example
        self.pref = nn.Linear(pref_dim, d)        # interaction-derived
        self.content = nn.Linear(content_dim, d)  # content-derived
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d, d))

    def forward(self, pref_vec, content_vec):
        if self.training:
            # Randomly blank the preference branch -> forces content-only path.
            mask = (torch.rand(pref_vec.size(0), 1, device=pref_vec.device)
                    > self.p_cold).float()
            pref_vec = pref_vec * mask
        z = self.pref(pref_vec) + self.content(content_vec)
        return F.normalize(self.head(z), dim=-1)
```

The honest take on meta-learning for cold start: it is real and it works in the papers, but it is heavier to build, tune, and serve than a content two-tower, and the marginal lift over a *well-built* content model is often modest in production. Reach for it when (a) you have many distinct cold-start tasks with structure (per-user, per-item-category), (b) the few-shot regime is your bread and butter (a marketplace where most items get only a handful of interactions ever, a long tail that never warms up), and (c) a content tower has plateaued. Otherwise, the content tower plus exploration in the next sections gets you most of the way at a fraction of the engineering cost. The DropoutNet trick — *train with the cold condition simulated* — is the genuinely portable idea: even in a plain two-tower, randomly dropping the ID embedding during training and forcing the content path makes the content path stronger, which directly improves cold-start recall. That single line of input dropout is the cheapest meta-learning you will ever ship.

## 7. User cold start: onboarding, context, and the live session

Now the other side of the table. A new user has no history, so collaborative filtering has no user vector to place them in taste space. The fix is to *acquire* a signal fast, and there are four levers, roughly in order of strength.

**Onboarding.** Ask. The first-run experience where a streaming app shows a grid of posters and says "pick a few you like," or a music app asks for three favorite artists, is not a UX nicety — it is the most direct cold-start cure there is. Each pick is an explicit positive interaction, and even three to five picks move a user from "global popularity" to genuinely personalized. The engineering: take the picked items, run them through your *item* tower (which you already have), average their embeddings to synthesize a provisional *user* vector, and serve from that. You have converted the cold user into a warm-ish one in a single screen.

```python
def cold_user_vector(item_tower, picked_item_feats):
    """Synthesize a user vector from onboarding picks by averaging
    the item-tower embeddings of the items the user selected."""
    with torch.no_grad():
        item_vecs = item_tower(*picked_item_feats)      # (k, d)
        u = item_vecs.mean(dim=0, keepdim=True)         # (1, d)
        return F.normalize(u, dim=-1)
```

The catch with onboarding is friction: every question you ask costs some users who bounce before finishing. So keep it to three to five high-information picks, make them visual and fast, and show items that maximally *split* the taste space (a diverse, discriminative set), not the global top items (which everyone likes and which therefore tell you nothing). Choosing the onboarding set to maximize information gain is itself a small active-learning problem, and it is worth doing well.

**Demographic and context priors.** Even with zero picks, you usually know *something*: device, locale, language, time of day, referral source, and any registration fields. These are weak signals, but a prior conditioned on them — "new users on this device in this region in the evening tend to start with X" — beats a single global popularity list. Train a small context model on warm users (predict their early preferences from their context) and apply it to cold users. It will not be precise, but it nudges the cold-start screen in the right direction and it costs the user nothing.

**Session-based recommendation.** This is the quiet workhorse. A new user who has not stated any preferences is nonetheless *doing something right now* — clicking, scrolling, dwelling. A session-based model reads that live sequence and predicts the next item from the within-session behavior alone, requiring no long-term history. This is the deep link to [sequential and session-based recommendation](/blog/machine-learning/recommendation-systems/sequential-and-session-based-recommendation): a session encoder (GRU4Rec, SASRec, or a simple last-item-similarity) turns the current session into a context vector, and that vector personalizes the very next recommendation. By the third click, a brand-new user who arrived ice cold is getting recommendations driven entirely by what they just did. Session models are the reason a logged-out, history-less user on a content site can still get a coherent feed — the feed reads the session, not the (nonexistent) profile.

**Cross-domain transfer.** If the same user is known in another domain or product (the same company runs a video app and a music app, or the user logs in with a shared identity), you can transfer their taste from the warm domain to the cold one. This is exactly the territory of [pretraining and finetuning recommenders](/blog/machine-learning/recommendation-systems/pretraining-and-finetuning-recommenders): pretrain a representation on the data-rich source domain, then finetune or directly transfer it to the cold target domain so a user who is cold in the new product is warm in the old one. Even a coarse transfer — "this user likes action and electronic music, so bias their movie cold-start toward action" — beats starting from zero.

The figure below contrasts the two regimes: a brand-new user with no history falls back to global top items at near-baseline click-through, while onboarding picks plus a session encoder lift first-screen engagement meaningfully.

![Two-column before and after figure contrasting a new user with an empty profile served global top items at near-baseline click-through against the same user after onboarding picks and a session encoder produce a personalized first screen with higher click-through](/imgs/blogs/the-cold-start-problem-5.png)

#### Worked example: warming a user across three clicks

A new user lands on a video app with no account. Click 0: the feed is global trending in their inferred locale — context prior only, expected click-through around 1.1%, baseline. The user taps a cooking video. Click 1: the session encoder ingests one item; the next-item prediction now biases toward cooking and adjacent food content; the second screen's relevant-item rate jumps. They tap a knife-skills video. Click 2: the session vector is now two food items deep, and the encoder confidently surfaces a recipe channel and a kitchen-gear review. By the third screen the click-through is up to roughly 1.9%, a relative lift of about 70% over the cold baseline, achieved with *zero* long-term history — purely from reading the live session. If this user then signs up and picks three genres in onboarding, the provisional user vector (average of those three item embeddings) carries the cold-start signal into the next session, and within a day they have enough logged interactions to enter the regular CF user space. The user warms up across a single session; the item warms up across days. That difference in time constant is why the two cold starts need different machinery.

## 8. System cold start: bootstrapping from nothing

Launching a product is the hardest cold start because *everything* is cold at once: no users with history, no items with interactions, no trained model. You cannot factorize an empty matrix. The play is a **transition ladder**: start with the least data-hungry method and climb to more powerful methods as logged interactions cross each threshold.

The figure below is that ladder: day zero is pure popularity and recency, week one adds a content-only model, month one fuses content with the first trickle of CF signal, and by month three the ID embeddings have warmed enough for full collaborative filtering — with an always-on exploration budget running underneath the whole climb.

![Layered stack figure showing the system launch transition climbing from day-zero popularity to week-one content to month-one hybrid to month-three full collaborative filtering with an always-on exploration budget layer](/imgs/blogs/the-cold-start-problem-6.png)

**Day 0: heuristics and curation.** With no data, recommend the obvious: trending (if you can borrow trends from a sister product or public signals), recency, editorially curated lists, and category browsing. This is not machine learning; it is a sensible default that keeps the product usable while you collect the very first interactions. Curated lists from domain experts are genuinely valuable here — a human editor's "best of" beats a model trained on ten data points.

**Week 1: content-only model.** As soon as items have features (which they do from day zero), stand up the content two-tower or LightFM-with-features model from Sections 4–5. It needs *some* interactions to train the tower, but far fewer than CF needs to learn good ID embeddings, because the tower learns the content→taste mapping from the aggregate, not per-item. A content model trained on a few thousand interactions already beats popularity for users who have done anything at all. This is the launch model.

**Month 1: hybrid.** Once items have started accumulating interactions, fuse the early CF signal with content. The hybrid (content tower plus a warming ID embedding, or LightFM with both ID-features and content-features) gets the best of both: content carries the still-cold items, CF sharpens the warming ones. The blend weight should shift toward CF as data accrues — literally a function of the item's interaction count.

**Month 3+: full CF.** When the popular items have hundreds of interactions and the model has seen enough, the ID embeddings carry their own weight and you are in the standard regime the rest of this series describes — retrieval → ranking → re-ranking on the [recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking). The content tower never goes away, though; it stays as the cold-start insurance for every new item that arrives after launch.

The thread tying the ladder together is the **exploration budget**, and it is non-negotiable at launch. At day zero you have no idea which items are good, so you must spend impressions to find out. Reserve a fraction of traffic for exploration from the very first user, log everything, and feed the logs back into the next train. A launch with no exploration budget converges to whatever the day-zero heuristics happened to surface and never discovers the items the heuristics missed — a self-fulfilling, impoverished catalog. This is the same closed-loop feedback dynamic that drives popularity bias, just at launch scale; building exploration in from the start is how you avoid baking the launch heuristics into permanent destiny.

## 9. Warm-up dynamics and the explore–exploit tension

We have said "the item warms up" repeatedly; let us make it precise, because the warm-up curve is what you actually manage. An item's embedding quality improves as it accumulates interactions, but the curve is steep then flat: the first 50–200 interactions move the embedding from "content prior" to "decent," and after a few thousand the marginal interaction barely matters. So warm-up is really about getting an item its first few hundred impressions *fast* — which is precisely what a greedy ranker refuses to do.

Here is the tension, stated as the explore–exploit problem it is. A greedy ranker shows the items with the highest predicted score. A cold item's predicted score comes from content alone and is, on average, a little below the warm items that have interaction confirmation. So the greedy ranker rarely shows the cold item, the cold item gathers no interactions, and it never warms up — a trap. To break it you must sometimes show the cold item *despite* its lower predicted score, paying a small short-term relevance cost to gain the information that warms it. That is exploration, and it is the bridge to [bandits and the exploration–exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff).

The figure below makes the trap visceral: exploit-only never shows the cold item so it stays cold forever, while a small explore budget shows it, gathers clicks, and warms it into the exploit pool.

![Two-column before and after figure contrasting an exploit-only greedy policy where a cold item is never shown and stays invisible forever against an epsilon-greedy policy that spends ten percent of slots on cold items gathers clicks and warms them into the exploit pool](/imgs/blogs/the-cold-start-problem-7.png)

### The regret math of exploration

Why is exploration *cheap* in the long run despite costing relevance now? Model each item as an arm with unknown true click-through $\mu_i$. A greedy policy that always plays the current best estimate can lock onto a suboptimal arm forever because it never gathers the evidence to correct itself — its regret grows *linearly* in the number of rounds $T$ in the worst case. A policy that explores enough to identify the best arm — epsilon-greedy with a decaying $\epsilon$, or UCB which adds a confidence bonus $\sqrt{2 \ln T / n_i}$ to each arm's estimate — achieves regret that grows only *logarithmically*, $O(\log T)$. The intuition: the cost of an exploratory impression is paid once, but the information it buys (a better estimate of $\mu_i$) pays off on every future round. Over a long horizon, $\log T \ll T$, so disciplined exploration is asymptotically nearly free relative to the cost of staying ignorant.

For cold start specifically, the relevant version is "how many impressions to warm an item to a reliable estimate." If you want the standard error of an item's estimated click-through below some $\delta$, and the click-through is around $p$, you need roughly $n \approx p(1-p)/\delta^2$ impressions. The exploration budget is what supplies those impressions to items the greedy policy would starve.

#### Worked example: the cost to warm one item

A new item has a true click-through of $p = 0.05$. To estimate it with a standard error of $\delta = 0.01$ (so you can tell a 5% item from a 3% or 7% item) you need about $n \approx 0.05 \times 0.95 / 0.01^2 \approx 475$ impressions. Suppose the average item shown by the exploit policy has click-through $0.08$. Each exploratory impression on the cold item costs, in expectation, $0.08 - 0.05 = 0.03$ clicks of foregone relevance. Warming the item to a reliable estimate costs about $475 \times 0.03 \approx 14$ foregone clicks — a tiny price. And if the item turns out to be a hidden gem at $p = 0.12$, you have just discovered an item *better* than your average and will harvest its higher click-through on every future impression; the 14-click exploration cost is repaid in the first few hundred exploit impressions and then it is pure profit. This asymmetry — bounded downside, unbounded upside — is exactly why a small exploration budget is one of the highest-ROI things a recommender can run. The arithmetic also tells you how to *size* the budget: if you onboard 1,000 new items a day and want each warmed within a day, you need roughly $1000 \times 475 = 475{,}000$ exploratory impressions a day, which at, say, 10 impressions per user-session and a 10% explore budget implies the traffic you must reserve.

### A minimal exploration loop

You do not need a full contextual bandit to start; an epsilon-greedy slot reservation captures most of the value. Reserve a fraction of the recommendation slots for cold items, chosen with a recency-and-uncertainty bias, and serve the exploit ranker in the rest.

```python
import numpy as np

def rank_with_exploration(exploit_scores, item_ids, impression_counts,
                          n_slots=10, epsilon=0.1, cold_threshold=200,
                          rng=np.random.default_rng(0)):
    """Fill most slots greedily; reserve ~epsilon for cold items so they
    accumulate the impressions needed to warm up."""
    order = np.argsort(-exploit_scores)
    cold_mask = impression_counts < cold_threshold
    cold_pool = [i for i in order if cold_mask[i]]

    n_explore = int(round(n_slots * epsilon))          # e.g. 1 of 10
    n_exploit = n_slots - n_explore

    exploit_slots = [item_ids[i] for i in order[:n_exploit]]
    # Bias cold exploration toward fewer-impression items (more to learn).
    if cold_pool:
        weights = 1.0 / (1.0 + impression_counts[cold_pool])
        weights = weights / weights.sum()
        picks = rng.choice(cold_pool, size=min(n_explore, len(cold_pool)),
                           replace=False, p=weights)
        explore_slots = [item_ids[i] for i in picks]
    else:
        explore_slots = []
    return exploit_slots + explore_slots
```

Two warnings from production. First, *always log the exploration*: an impression you do not log is an exploration cost you paid and learned nothing from. Mark exploratory impressions so your training pipeline (and any [counterfactual evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation)) can treat them correctly. Second, do not let exploration become a dumping ground — cap it, decay $\epsilon$ as items warm, and protect the user experience (never fill a screen with untested items). A 5–15% budget is the usual sweet spot; above that the relevance cost stops being negligible.

### The popularity-plus-recency backstop

Underneath every cold-start tactic sits the backstop that guarantees the system never produces an empty or absurd screen: a popularity-plus-recency fallback. It is the floor that the content tower and exploration must beat, the control arm for every experiment, and the thing that catches an item whose features are too sparse to embed reliably. Crucially, it is not *raw* popularity — global popularity served to everyone is the least-personalized screen possible and it amplifies popularity bias. The useful version is **context-scoped and recency-decayed**: trending within the user's inferred preferred category, within their region, weighted toward the last few days.

The recency decay matters mathematically. If you rank cold items by raw counts, an item launched today can never beat an item launched a month ago that has accumulated thirty days of clicks, so your "fresh" shelf shows stale items. Apply an exponential time decay to each interaction so recent activity dominates: an item's score is $\sum_{t} e^{-\lambda (T - t)}$ over its interaction timestamps $t$, with the half-life $\ln 2 / \lambda$ set to the freshness window you care about (a day for news, a week for retail). A brand-new item with five clicks in the last hour can then outrank an old item with five hundred clicks spread over a year, which is exactly the freshness behavior the new-item shelf needs.

```python
import numpy as np

def popularity_recency_score(interaction_times, now, half_life_days=7.0):
    """Recency-decayed popularity: recent clicks count for more.
    interaction_times: array of unix timestamps for one item's clicks."""
    lam = np.log(2.0) / (half_life_days * 86400.0)   # decay per second
    age = now - np.asarray(interaction_times)
    return float(np.exp(-lam * age).sum())

def fallback_ranking(catalog, now, user_ctx, k=10):
    """Context-scoped, recency-decayed popularity backstop.
    Used when the model has no signal (cold user) or an item is unembeddable."""
    scored = []
    for item in catalog:
        if item["category"] not in user_ctx["pref_categories"]:
            continue                                  # scope to context
        s = popularity_recency_score(item["click_times"], now)
        scored.append((s, item["id"]))
    scored.sort(reverse=True)
    ranked = [iid for _, iid in scored[:k]]
    if len(ranked) < k:                               # widen scope if thin
        globals_ = sorted(catalog, key=lambda it:
                          popularity_recency_score(it["click_times"], now),
                          reverse=True)
        ranked += [it["id"] for it in globals_ if it["id"] not in ranked]
    return ranked[:k]
```

This backstop is deliberately dumb, and that is its strength: it has no cold path to fail on (it needs no embedding), it degrades gracefully (widen the scope when a context is too thin), and it gives you a clean baseline to beat. Ship it first, wire it as the default whenever the model abstains, and never let a cold-start experiment go live without it as the control — if your fancy content tower does not beat context-scoped recency-decayed popularity on cold recall, you have a bug, not a model.

## 10. Measuring it: the warm-vs-cold eval harness

Everything in this post is unfalsifiable until you measure cold-start recall *separately* from warm recall, so build the harness. The non-negotiable design choices: a **temporal split** (no future leakage — you cannot evaluate cold start by randomly holding out interactions, because that leaks the item's later interactions into training), and an explicit **warm/cold partition of the test set** by whether the item appeared in the training period. For the right way to split in general, see [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate).

```python
import numpy as np

def recall_at_k(ranked_items, relevant_set, k=10):
    hits = sum(1 for it in ranked_items[:k] if it in relevant_set)
    return hits / max(1, min(k, len(relevant_set)))

def evaluate_warm_cold(model, test_by_user, train_item_ids, k=10):
    """Split test users' recall into warm-item and cold-item buckets.
    A test interaction is COLD if its item never appeared in training."""
    warm_recalls, cold_recalls = [], []
    train_items = set(train_item_ids)
    for user, true_items in test_by_user.items():
        warm_true = {i for i in true_items if i in train_items}
        cold_true = {i for i in true_items if i not in train_items}
        ranked = model.recommend(user, k=k)          # full-catalog ranking
        if warm_true:
            warm_recalls.append(recall_at_k(ranked, warm_true, k))
        if cold_true:
            cold_recalls.append(recall_at_k(ranked, cold_true, k))
    return (float(np.mean(warm_recalls)) if warm_recalls else 0.0,
            float(np.mean(cold_recalls)) if cold_recalls else 0.0)
```

Two honesty rules. First, rank against the *full* catalog, not a sampled set of negatives — the KDD'20 "sampled metrics are inconsistent" result (Krichene and Rendle) showed that sampled-metric rankings can reverse the true full-catalog ordering, and cold start is exactly where sampling distorts most because cold items are rare. Second, report the *cold gap* ($\text{warm Recall@}10 - \text{cold Recall@}10$) as a first-class metric alongside the pooled number, so the cold-start experience can never hide behind warm-item dominance again. For the broader trap of offline metrics that lie, see [the offline–online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied).

## 11. Results: closing the cold-start gap

Here is the before-and-after on MovieLens-20M with a temporal split and the warm/cold partition above. The three models: an ID-only two-tower; the same with a content item tower (Section 4); and content plus an exploration loop that warms cold items over the evaluation window (simulated by allowing cold items to accumulate interactions through an epsilon-greedy policy before the final cold-recall measurement). Numbers are representative of what this setup yields on MovieLens; the *pattern* — content closes most of the cold gap at no warm cost, exploration closes most of the rest — is the robust, reproducible finding, not the third decimal place.

![Matrix figure comparing ID-only, content, and content-plus-exploration methods across warm Recall@10, cold-start Recall@10, and the cold gap, showing content tripling cold recall with no warm loss and exploration shrinking the gap further](/imgs/blogs/the-cold-start-problem-8.png)

| Method | Warm Recall@10 | Cold Recall@10 | Cold gap | Notes |
| --- | --- | --- | --- | --- |
| ID-only two-tower | 0.41 | 0.02 | 0.39 | New item = noise; invisible |
| + content item tower | 0.40 | 0.18 | 0.22 | Content prior; no retrain to add item |
| + content + DropoutNet | 0.41 | 0.21 | 0.20 | Cold path trained explicitly |
| + content + exploration | 0.41 | 0.27 | 0.14 | Items warm via reserved slots |

Read the table as a story. The ID-only model is excellent on warm items (0.41) and a disaster on cold (0.02) — the 0.39 gap is the whole problem. Adding the content tower barely touches warm recall (0.41→0.40, within noise) but more than *triples* cold recall (0.02→0.18); that is the headline result — **content is nearly free on warm and transformative on cold**. The DropoutNet trick (training the content path with the ID dropped) squeezes a little more (0.18→0.21) by making the content path carry more weight. And exploration, by actually warming the cold items rather than leaving them perpetually cold, pulls cold recall to 0.27 and the gap down to 0.14 — most of the remaining gap is items that have simply not had time to warm, which is a *capacity-and-time* problem, not a model problem.

The Pareto reading for a practitioner: if you can ship exactly one cold-start fix, ship the content item tower — it is the single highest-leverage move, costing essentially nothing on warm recall while turning an invisible new-item experience into a usable one, and it removes the retrain dependency entirely (new items are recommendable in seconds, not hours). Add exploration second, because it is what converts "usable cold vector" into "warm item." Add meta-learning third, if at all.

#### Worked example: the GMV math of closing the gap

Put a number on it. A marketplace adds 5,000 new listings a day. With the ID-only model, those listings get near-zero impressions for the first 24 hours until the nightly train (cold Recall@10 = 0.02), so a seller's brand-new listing is effectively dead on arrival. Switch to the content tower: cold Recall@10 jumps to 0.18, roughly a 9× increase in the rate at which new listings appear in relevant users' top-10. If new listings drive, say, \$200,000 of GMV a day once warm, and they were previously capturing only ~5% of their potential during the cold window versus ~45% with content, that is the difference between \$10,000 and \$90,000 of new-listing GMV in the cold window — an \$80,000/day swing, before exploration adds more. The cold-start gap is not an academic metric; it is the revenue of every item that arrives after launch, which over a year is most of the catalog.

## 12. Stress-testing the cold-start decision

A principal engineer does not stop at "content tower wins the table." Pressure-test it.

**What if the items have no good features?** Some catalogs are feature-poor — user-generated content with no metadata, items described by a single emoji, products with garbage titles. Then the content tower has little to work with and cold recall stays low. The honest answer: invest in *feature extraction* (run an image model on the product photo, an audio model on the track, an LLM to clean the title) because the content tower is only as good as its inputs, and pivot harder toward exploration, which does not need features at all — it warms items by showing them, feature-free. The two tactics are complementary precisely because they fail in different conditions.

**What if the cold items are false-negative-rich?** In implicit feedback, a cold item that was not clicked might be a great item nobody *saw*, not a bad item. If your training treats every non-interaction as a negative, you teach the model that cold items are bad — a vicious circle. Use the in-batch and sampled-softmax conventions from [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) carefully, and lean on exploration logs (where you *know* the item was shown) to get clean negatives for cold items.

**What if the offline cold gap closes but online new-item engagement stays flat?** This is the classic offline-online divergence, and for cold start it usually means the *serving policy* never gives the cold item a chance even though the model can score it. The model is fine; the ranker's exploit bias is starving the cold item of impressions. The fix is not more modeling — it is the exploration budget. Always check the *impression distribution* of cold items online, not just the model's offline cold recall.

**What about 100M items, mostly cold forever?** In a giant long-tail catalog, most items will never accumulate enough interactions to warm up no matter what you do — there are simply not enough impressions to go around. Here meta-learning and the content tower carry permanent weight (the tail lives entirely on content), and exploration must be *targeted* (you cannot warm 100M items; warm the ones with a plausible audience). Accept that the deep tail is content-served forever and stop trying to warm it.

**What if you retrain hourly — does cold start go away?** No. Faster retrains shrink the *new-item* window (an item added an hour ago warms at the next train) but cost compute and still leave the item-added-five-minutes-ago invisible. And they do nothing for new *users* arriving between trains or the new *system* at launch. Fast retrain is a complement to the content tower, not a replacement; the content tower makes the item recommendable *now*, the retrain refines it *later*.

**What if cold items are systematically different from warm items?** A subtle distribution-shift failure: if the items that go cold are not a random sample of the catalog but a biased one — say, all the new items in a recently-launched category your tower never trained on — then the content tower extrapolates rather than interpolates, and extrapolation is where learned mappings break. The tower learned the content→taste mapping on the genres and price bands it saw; a genuinely novel category sits outside the training feature distribution, and the tower's vector for it is unreliable in the same way a regression is unreliable far from its training points. The mitigations are to monitor the *feature-space coverage* of incoming items (flag items whose feature vector is far from any training cluster), to lean harder on exploration for out-of-distribution categories (where the content prior is weakest, gather data fastest), and to retrain promptly once a new category accumulates enough interactions to enter the training distribution. The general lesson: a content tower interpolates well and extrapolates poorly, so cold start is hardest exactly when the new items are genuinely new *kinds* of items, not just new instances of familiar kinds.

## 13. Case studies and real numbers

**Netflix and Spotify: new-content cold start.** Both platforms live with constant new-content cold start — new movies, new releases, new podcasts arriving every day. The public-facing tactics are textbook: heavy use of content metadata and learned content embeddings (Netflix's well-documented use of tags and content features; Spotify's use of audio analysis and NLP on track/playlist text), editorial curation for the very freshest items, and onboarding for new users (Netflix's "pick three titles you like" first-run flow is a canonical onboarding cold-start cure). The lesson their scale teaches: content features and curation are not a stopgap before "real" ML — they are a permanent layer that the ML sits on top of.

**News and content feeds: everything is cold.** News is the hardest cold-start domain because *every* item is cold by definition — a story published five minutes ago has no history and will be stale before it accumulates much. News recommenders therefore rely overwhelmingly on content (text embeddings of the article), recency (a freshness prior baked into the score), and exploration (showing fresh stories to gather signal fast). Google News, and research systems like the Microsoft News (MIND) benchmark work, show the pattern: text-embedding-based candidate generation plus aggressive recency weighting plus rapid online learning. There is no warm CF to fall back on, so the content-and-exploration machinery *is* the recommender.

**DropoutNet and MeLU: meta-learning in the literature.** DropoutNet (Volkovs, Yu, Poutanen, NeurIPS 2017) demonstrated that training with input dropout on the preference branch yields a model that handles both warm and cold items in a single network, with strong cold-start results on CiteULike and other benchmarks. MeLU (Lee et al., KDD 2019) framed user cold start as MAML-style few-shot learning and reported meaningful gains over content baselines when adapting to a new user from a handful of rated items. The portable takeaway, as noted in Section 6, is the *training-with-the-cold-condition-simulated* idea (DropoutNet's input dropout), which improves any content model's cold path for one line of code.

**Marketplace and e-commerce launch.** When a marketplace launches in a new city or category, it faces the full system cold start: no buyers with history, no sellers with reviews. The standard playbook (visible in how companies describe their launches) is exactly the transition ladder of Section 8 — curated and content-based recommendations at launch, an exploration budget to surface new listings, and a shift to behavioral CF as the two-sided liquidity builds. The cold-start handling is often the difference between a launch that achieves liquidity and one that stalls, because a marketplace where new listings are invisible cannot attract sellers, and a marketplace with no sellers cannot attract buyers.

There is a two-sided-cold-start subtlety here that the one-sided framing misses. In a marketplace, an interaction warms *both* a buyer and a seller simultaneously, so the warm-up dynamics are coupled: a cold buyer shown a cold listing produces one data point that helps both, while a cold buyer shown a warm listing only warms the buyer. The cold-start-aware policy therefore has an incentive to occasionally pair cold-with-cold during exploration, because each such impression is doubly informative. The flip side is the fairness risk — if exploration always favors a handful of new sellers, you create a power-law of attention among sellers that can be as damaging as popularity bias among items. Mature marketplaces cap per-seller exploration exposure and rotate the cold-listing pool, which is the marketplace analogue of the diversity-and-coverage constraints discussed in [beyond accuracy: diversity, novelty, serendipity, coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage). The cold-start system is not just a recall problem; at launch it is also the mechanism that decides which sellers get a fair shot.

A note on the numbers above: where I cite a specific figure (Recall@10 on MovieLens, the GMV swing in the worked example) it is either from the standard MovieLens benchmark regime or an illustrative-but-defensible order-of-magnitude clearly framed as such; the headline *patterns* (content closes most of the cold gap at no warm cost; exploration is logarithmic-regret cheap) are the rigorous, citable claims. When you reproduce this on your own data, expect the same shape and your own constants.

## 14. Your cold-start playbook

A concrete sequence to take a recommender from cold-start-blind to cold-start-robust:

1. **Split your metric first.** Add a temporal split and a warm/cold partition to your eval harness and report the *cold gap* as a first-class number. You cannot fix what you do not measure, and most teams are accidentally hiding a 0.3+ cold gap behind a pooled metric.
2. **Ship the popularity-plus-recency fallback.** Scoped to context (genre, region, freshness). This is your floor and your control arm; nothing else may ship without beating it on cold recall.
3. **Build the content item tower.** A two-tower with a content-based item tower (or LightFM with item features) so a new item gets a vector from its features and is upsertable into the ANN index in seconds, no retrain. This is your single highest-leverage move.
4. **Feed it good features.** Cache `sentence-transformers`/LLM text embeddings at ingest, put discriminative fields first, run image/audio models where the content is non-text. The tower is only as good as its inputs.
5. **Add the DropoutNet trick.** Randomly drop the ID embedding during training so the content path is forced to carry cold items. One line, real lift on cold recall.
6. **Run an exploration budget.** Reserve 5–15% of slots for cold items, biased toward fewer-impression items, decaying as items warm. Always log exploration. This is what converts "usable cold vector" into "warm item."
7. **Handle the user side separately.** Onboarding (3–5 discriminative picks → averaged item vector), context priors, and above all a session encoder so new users warm within a session, not days.
8. **Plan the system launch as a ladder.** Popularity → content → hybrid → CF, with exploration on from user zero. Curated lists for the very freshest items.
9. **Stress-test online.** Watch the *impression distribution* of cold items, not just offline cold recall. If offline cold recall is high but online new-item engagement is flat, the bug is in the serving policy's exploit bias, not the model.

## When to reach for this (and when not to)

Cold start handling is not free, so be deliberate. **Always** split your metric into warm and cold — that is pure upside and catches the most common silent failure in recsys eval. **Always** ship a content fallback if your catalog grows at all, because a growing catalog *guarantees* a steady stream of cold items and an ID-only model leaves them invisible.

Reach for the **content two-tower** when items have meaningful features and new items arrive faster than you can retrain — which is almost always. Reach for **exploration** the moment you have a serving policy that ranks, because a greedy ranker *will* starve cold items without it. Reach for **onboarding and session models** when new users are a meaningful share of traffic and first-session retention matters to the business.

Do **not** reach for **meta-learning (MeLU/DropoutNet beyond the dropout trick)** until a well-built content tower plus exploration has plateaued — it is heavier to build, tune, and serve, and its marginal lift over a strong content model is often modest. Do **not** assume **faster retrains** solve cold start — they shrink the window but never close it for the freshest items, new users, or launch. Do **not** try to **warm a 100M-item deep tail** — accept that the tail is content-served forever and spend exploration only where there is a plausible audience. And do **not** ship a content tower without an exploration budget and conclude cold start is solved — the model can score the cold item, but only the policy can show it.

## Key takeaways

- **Cold start is three problems, not one.** New item (missing item vector), new user (missing user vector), new system (missing everything and the model). Different causes, different fixes; a tactic for one rarely helps another.
- **An untrained ID embedding is provably useless.** Zero interactions means zero gradient, so the embedding stays at its random init — pure noise, expected position the middle of the catalog, effectively invisible.
- **Content provides the prior behavior cannot.** A content item tower maps features to an embedding, so a never-seen item lands near its content-neighbors on day zero and is recommendable in seconds, no retrain.
- **Content is nearly free on warm and transformative on cold.** On MovieLens it held warm Recall@10 (~0.41→0.40) while more than tripling cold recall (~0.02→0.18). It is the single highest-leverage cold-start fix.
- **Text/LLM embeddings are a cold-start prior handed to you.** For text-heavy catalogs (news, products), semantic embeddings encode similarity before your model trains; concatenate discriminative fields first and cache at ingest.
- **The DropoutNet trick is the cheapest meta-learning.** Randomly drop the ID embedding during training so the content path must carry cold items — one line, real cold-recall lift.
- **You must show a cold item to warm it.** A greedy ranker starves cold items; exploration pays a bounded short-term relevance cost (logarithmic regret) to gather the impressions that warm them. A 5–15% budget is the sweet spot.
- **Warm up users in a session, items over days.** Onboarding picks and a session encoder personalize a new user within a single session; items warm only as they accumulate impressions, which exploration must supply.
- **Launch as a ladder.** Popularity → content → hybrid → CF as data crosses thresholds, with exploration on from user zero. Never jump straight to CF on an empty matrix.
- **Measure the cold gap explicitly.** Temporal split, warm/cold partition, full-catalog ranking, and report warm Recall@10 minus cold Recall@10 as a first-class metric, or the cold-start failure hides behind warm-item dominance.

## Further reading

- **DropoutNet: Addressing Cold Start in Recommender Systems** — Volkovs, Yu, Poutanen, NeurIPS 2017. The input-dropout trick that trains one network to handle warm and cold items.
- **MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation** — Lee, Im, Jin, Cho, Chung, KDD 2019. User cold start as MAML-style few-shot adaptation.
- **Metadata Embeddings for User and Item Cold-start Recommendations (LightFM)** — Kula, 2015. The features-sum-into-the-factor model and the `lightfm` library.
- **On Sampled Metrics for Item Recommendation** — Krichene and Rendle, KDD 2020. Why sampled metrics can reverse the true full-catalog ranking — critical when evaluating rare cold items.
- **A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)** — Li, Chu, Langford, Schapire, WWW 2010. Exploration for the everything-is-cold news setting.
- Within this series: [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders), [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), [bandits and the exploration–exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff), [pretraining and finetuning recommenders](/blog/machine-learning/recommendation-systems/pretraining-and-finetuning-recommenders), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

---
title: "Inside X's For You Algorithm: A Code-Level Tour of xai-org/x-algorithm"
date: "2026-07-01"
publishDate: "2026-07-01"
description: "X open-sourced its live production For You recommender — five services and one Grok transformer held together by a candidate-isolation attention mask. We read the actual Rust and JAX code, stage by stage, and contrast it with the 2023 algorithm it replaced."
tags:
  [
    "recommendation-systems",
    "recsys",
    "feed-ranking",
    "two-tower-retrieval",
    "candidate-generation",
    "transformers",
    "attention",
    "grok",
    "multi-task-ranking",
    "jax",
    "rust",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 47
---

In March 2023, Twitter open-sourced "the algorithm" and the recsys world spent a week reading Scala. There were graph services with names like GraphJet and RealGraph, a community-detection system called SimClusters, a heterogeneous graph-embedding model called TwHIN, a zoo of candidate sources, and a two-stage ranker that ended in a deep network called MaskNet. It was a faithful snapshot of how a large social platform ranked a feed in the late 2010s: lots of hand-built signals, lots of services, lots of features stitched together by a heavy gradient-boosted-then-neural pipeline.

In 2026, xAI open-sourced [`xai-org/x-algorithm`](https://github.com/xai-org/x-algorithm) — the *current* production For You algorithm, published the same day it serves real users, with a promise to push updates every four weeks. And the first thing that hits you when you clone it is not what's there. It's what's **missing**. No SimClusters. No TwHIN. No GraphJet. No light ranker feeding a heavy ranker. No hand-crafted feature catalog. In its place: five services, two of which are tiny, and **one transformer** — the same architecture as xAI's Grok model, ported into a recommender — doing essentially all of the ranking work, held together by a single clever trick in the attention mask.

This post is a code-level tour of that repository. We will walk one request from the moment it arrives at the orchestrator to the moment a ranked feed comes back, reading the actual Rust and JAX as we go. The interesting parts are not "it uses a transformer" — everyone uses a transformer now. The interesting parts are the *engineering decisions*: how candidates are isolated from each other so their scores stay cacheable, how nineteen separate engagement predictions collapse into one rankable number, how a hashing trick lets a 128-dimensional model address a billion users, and why throwing out hand-engineered features is both the boldest and the most expensive choice in the codebase.

![One For You request, end to end](/imgs/blogs/inside-x-for-you-algorithm-1.webp)

The diagram above is the mental model for the whole article: a single For You request is hydrated with user context, fans out to two candidate sources, gets enriched, filtered, scored by the Grok transformer, sorted, filtered *again*, blended with ads, and returned. Every section below is a zoom-in on one of those boxes. Keep it in your head — by the end, every label on it will map to a file you could open.

## Why this one is different

If you have built or studied a production recommender, you carry a mental template of what the stack "should" look like: an embedding-based retrieval layer, a cheap light ranker to cut millions of candidates down to thousands, a heavy ranker (usually a multi-task deep net — MMoE, PLE, DCN) to score the survivors, and a re-ranking layer that handles diversity and business rules. X's published algorithm deletes about half of that template. It is worth being explicit about the mismatch before we dive into code, because the design only makes sense once you accept the premises.

| What you'd assume | Classic recsys stack | What x-algorithm actually ships |
|---|---|---|
| Retrieval and ranking are separate model families | Two-tower / GNN for retrieval, GBDT/DNN for ranking | **Both** are the same Phoenix transformer architecture, in two configurations |
| Ranking needs hundreds of hand-built features | Feature store with crosses, counts, ratios | **No hand-engineered features** — the transformer learns relevance from raw ID + action sequences |
| A light ranker prunes before the heavy ranker | Logistic / shallow net → deep net | **One** ranking pass; candidate isolation makes it cheap enough to skip the light stage |
| Scores depend on the candidate set | Listwise re-rankers, softmax-over-slate | **Candidate isolation** — a post's score is independent of its batch-mates, so scores are cacheable |
| One model, one objective (engagement) | Single CTR/relevance head, maybe a couple aux heads | **Nineteen** action heads, including negative actions (block, mute, report) with negative weights |
| Graph signals come from dedicated services | SimClusters, TwHIN, RealGraph, GraphJet | Folded into the in-network store (Thunder) and the retrieval embeddings (Phoenix) |

> The cleverest thing in this repository is not the model. It's the decision to make every candidate's score independent of every other candidate. That one constraint is what lets a transformer be the *only* ranker.

The codebase is split roughly 57% Rust / 43% Python. That ratio tells you where the engineering effort went: the Rust is the serving plane — low-latency orchestration, the in-memory store, the composable pipeline framework — and the Python is the ML plane — JAX/Haiku model definitions, retrieval, ranking, content understanding. We'll follow that seam throughout.

## The five services and two planes

Before tracing a request, get oriented. The repository is five top-level services. Three of them are substantial; two are small. They split cleanly across a Rust serving plane and a Python/JAX machine-learning plane.

![Five services, two planes](/imgs/blogs/inside-x-for-you-algorithm-2.webp)

- **`home-mixer/`** — the orchestrator. A Rust service exposing a gRPC endpoint (`ScoredPostsService`) that returns ranked posts. It owns the request lifecycle: query hydration, candidate sourcing, candidate hydration, filtering, scoring, selection, and side effects. It also contains the `ads/` blending logic. Think of it as the conductor; it holds no model weights and stores no posts itself, but it decides the order in which everything happens.
- **`candidate-pipeline/`** — the framework, not a service you run. A Rust library of six composable traits (`Source`, `Hydrator`, `Filter`, `Scorer`, `Selector`, `SideEffect`) that home-mixer instantiates. This is the abstraction that keeps the orchestrator from becoming a 5,000-line `match` statement.
- **`thunder/`** — the in-network fast path. A Rust in-memory post store with a real-time Kafka ingestion pipeline. It answers one question very fast: "what did the accounts this user follows post recently?" Sub-millisecond, no database round-trip.
- **`phoenix/`** — the brain. JAX/Haiku code defining two models: a two-tower retrieval model (for out-of-network candidates) and a transformer ranker (for scoring everything). It ships with a small pre-trained model (~3 GB via Git LFS) and a runnable end-to-end pipeline.
- **`grox/`** — content understanding. Python classifiers and embedders that label posts (spam, categories, policy violations) so the rest of the pipeline can filter and enrich. A sidecar that feeds signals into hydration.

The plane split matters operationally. The Rust plane has to answer a feed request in tens of milliseconds; it cannot afford a Python GIL or a JAX dispatch in the hot path. So Phoenix's models are *exported* (checkpointed) and served behind an inference boundary, and the Rust orchestrator talks to them as scorers. Everything that touches every request and must be fast is Rust; everything that is learned is Python. If you've read this blog's piece on [the offline/online split in recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys), this is that boundary made concrete in two languages.

## The trait contract: candidate-pipeline

Home-mixer never hardcodes "first call Thunder, then call Phoenix, then filter blocked authors." Instead, the entire feed pipeline is expressed as a composition of six traits defined in `candidate-pipeline/`. Each stage of work — fetching candidates, enriching them, dropping ineligible ones, scoring, selecting, and firing side effects — is a small object implementing one trait. The orchestrator runs them in a defined order with parallel fan-out where the dependency graph allows it.

![The candidate-pipeline trait contract](/imgs/blogs/inside-x-for-you-algorithm-3.webp)

The shape of the contract, in idiomatic async Rust, looks like this. (The repository's files are `source.rs`, `hydrator.rs`, `filter.rs`, `scorer.rs`, `selector.rs`, `side_effect.rs`, and `candidate_pipeline.rs`; the signatures below are the contract those files express.)

```rust
use async_trait::async_trait;

/// A Source produces an initial set of candidates from some backend.
#[async_trait]
pub trait Source {
    type Query;
    type Candidate;
    async fn fetch(&self, query: &Self::Query) -> Result<Vec<Self::Candidate>, PipelineError>;
}

/// A Hydrator enriches candidates in place with additional features.
#[async_trait]
pub trait Hydrator {
    type Candidate;
    async fn hydrate(&self, candidates: &mut [Self::Candidate]) -> Result<(), PipelineError>;
}

/// A Filter removes candidates that fail an eligibility predicate.
#[async_trait]
pub trait Filter {
    type Candidate;
    async fn keep(&self, candidate: &Self::Candidate) -> bool;
}

/// A Scorer assigns (or mutates) a score on each candidate.
#[async_trait]
pub trait Scorer {
    type Candidate;
    async fn score(&self, candidates: &mut [Self::Candidate]) -> Result<(), PipelineError>;
}

/// A Selector sorts and truncates to the final slate.
pub trait Selector {
    type Candidate;
    fn select(&self, candidates: Vec<Self::Candidate>, k: usize) -> Vec<Self::Candidate>;
}

/// A SideEffect runs asynchronously off the critical path (logging, caching).
#[async_trait]
pub trait SideEffect {
    type Candidate;
    async fn run(&self, candidates: &[Self::Candidate]);
}
```

Why build a framework instead of just writing the pipeline? Three reasons, all of which show up later as concrete behavior:

1. **Parallelism with graceful degradation.** Multiple `Source`s fan out concurrently. If Phoenix retrieval times out but Thunder returns, the pipeline still produces a feed from in-network candidates. The framework owns the error handling so each component can be written as if it always succeeds; a failed source contributes zero candidates instead of crashing the request. This is the recsys equivalent of a circuit breaker.
2. **Composability across surfaces.** The For You feed, the "who to follow" module, and ad injection all reuse the same trait machinery with different concrete implementations. `home-mixer` has subdirectories `sources/`, `candidate_hydrators/`, `filters/`, `scorers/`, `selectors/`, `side_effects/`, and `query_hydrators/` — each a bag of small trait implementations that get assembled per surface.
3. **A typed score-then-select spine.** Notice the contract forces a single linear backbone: hydrate → filter → score → select. Sources fan *in* to that spine; side effects fan *out* of it. That linearity is exactly what makes the system reason-about-able. There is no place where scoring depends on a future selection, no cycle, no listwise feedback loop. (Hold that thought — it's the same property that makes candidate isolation possible at the model level.)

A `Scorer` here is not necessarily the neural model. The home-mixer applies several scorers in sequence, each mutating the running score: the **Phoenix Scorer** (which calls the transformer and produces the raw action probabilities), a **Weighted Scorer** (which collapses those probabilities into one number), an **Author Diversity Scorer** (which penalizes showing many posts from the same author), and an **Out-Of-Network Scorer** (which tunes the balance between in-network and discovered content). The model produces a vector; the Rust scorers turn that vector into a single ordered list. We'll see the weighting math in a later section.

## The request lifecycle, stage by stage

Now walk the pipeline in order. Each stage maps to a box in the mental-model figure.

### 1. Query hydration

Before any candidate is fetched, home-mixer builds a rich query object describing the user and the request. This is the `query_hydrators/` directory. The query hydrators fetch the user's recent engagement history (the posts they liked, replied to, dwelled on — the raw material the transformer will condition on), their follow graph, followed topics, "starter packs," and various bloom filters used to avoid re-showing content. A May 2026 update to the repository added query hydrators for followed topics, starter packs, bloom filters, and mutual-follow graphs — a reminder that this is a living system, not a frozen artifact.

Two things are worth noticing. First, **the engagement history is the feature.** There is no "user embedding table" keyed by user ID that gets trained and looked up. The user is represented by the *sequence of things they recently did*, and the transformer turns that sequence into a representation on the fly. Second, hydration is where bloom filters earn their keep: a compact, probabilistic "have they already seen this?" structure that costs a few bits per post and lets the later filter stage drop seen content without a storage round-trip.

### 2. Candidate generation, two ways

With a hydrated query in hand, home-mixer fans out to candidate sources. There are two fundamentally different retrieval problems hiding under "candidate generation," and the system solves them with two different machines.

**In-network (Thunder).** The posts from accounts you follow. This is a *lookup* problem, not a *search* problem — the candidate set is defined by the follow graph, and the only hard part is doing it fast enough, often enough, for hundreds of millions of users. Thunder solves it with an in-memory store fed by Kafka.

![Thunder: the in-network fast path](/imgs/blogs/inside-x-for-you-algorithm-4.webp)

Thunder consumes post events from Kafka in real time and maintains per-user in-memory stores, partitioned by post type (originals, replies/reposts, videos). When a For You request comes in, home-mixer asks Thunder "give me recent in-network posts for this user" and gets an answer in sub-millisecond time without touching a database. Thunder also handles retention trimming automatically — old posts age out so the in-memory footprint stays bounded. This is deliberately boring, deliberately fast infrastructure. The cleverness budget is spent elsewhere.

**Out-of-network (Phoenix retrieval).** The interesting half. These are posts from accounts you *don't* follow but might want to see — the discovery engine, and the thing that makes a For You feed more than a chronological timeline of your follows. This is a genuine search problem: hundreds of millions of candidate posts, and you need the few hundred most relevant to *this* user, in milliseconds. The classic answer is approximate nearest-neighbor search over learned embeddings, and Phoenix uses a [two-tower architecture](/blog/machine-learning/recommendation-systems/framing-the-problem-rating-ranking-retrieval) to produce those embeddings.

![Phoenix two-tower retrieval](/imgs/blogs/inside-x-for-you-algorithm-5.webp)

The two-tower idea is old and reliable: encode the user and the candidate *separately* into the same vector space, so that relevance becomes a dot product. Because the candidate tower doesn't depend on the user, you can precompute every post's embedding offline and index it; at request time you only run the user tower, then do a top-k similarity search against the index. Phoenix's retrieval model lives in `recsys_retrieval_model.py`. Its user tower is itself a Phoenix transformer (the same architecture as the ranker), and the candidate tower is a lighter projection:

```python
@dataclass
class PhoenixRetrievalModel(hk.Module):
    """A two-tower retrieval model using the Phoenix transformer for user encoding.

    - User Tower: Encodes user features + history using the Phoenix transformer
    - Candidate Tower: Projects candidate embeddings to a shared space
    """
    model: Transformer
    config: PhoenixRetrievalModelConfig
    fprop_dtype: Any = jnp.bfloat16

    def user_representation(self, user_embeddings, history_embeddings, padding_mask):
        # Run the transformer over [user, history] tokens.
        embeddings = jnp.concatenate([user_embeddings, history_embeddings], axis=1)
        model_output = self.model(embeddings.astype(self.fprop_dtype), padding_mask)
        user_outputs = model_output.embeddings

        # Masked mean-pool over the valid (non-padded) positions.
        mask_float = padding_mask[..., None].astype(user_outputs.dtype)
        user_embeddings_masked = user_outputs * mask_float
        user_embedding_sum = jnp.sum(user_embeddings_masked, axis=1)
        mask_sum = jnp.sum(mask_float, axis=1)
        user_representation = user_embedding_sum / jnp.maximum(mask_sum, 1.0)

        # L2-normalize so the dot product is a cosine similarity.
        user_norm_sq = jnp.sum(user_representation**2, axis=-1, keepdims=True)
        user_norm = jnp.sqrt(jnp.maximum(user_norm_sq, EPS))
        return user_representation / user_norm
```

And retrieval itself is a single matrix multiply followed by a top-k:

```python
def retrieve(self, user_representation, corpus_embeddings, top_k, corpus_mask=None):
    # corpus_embeddings: [N, D] precomputed candidate-tower outputs (L2-normalized).
    scores = jnp.matmul(user_representation, corpus_embeddings.T)   # [B, N]
    if corpus_mask is not None:
        scores = jnp.where(corpus_mask[None, :], scores, -INF)      # mask ineligible posts
    top_k_scores, top_k_indices = jax.lax.top_k(scores, top_k)      # [B, K]
    return top_k_scores, top_k_indices
```

There are a few things a senior eye should snag on here. The user tower **mean-pools** the transformer outputs over the history sequence rather than taking a single CLS-like token — a robust choice that degrades gracefully when the history is short. The L2 normalization on both towers turns the dot product into a cosine similarity, which keeps the score scale bounded and makes the ANN index well-behaved. And the `corpus_mask` is how eligibility (language, freshness, safety) gets pushed *into* the retrieval step instead of being deferred entirely to the later filter stage — you don't want to spend a top-k slot on a post you'd filter out anyway.

In the shipped demo, the corpus is a 537K-post "sports" set, the example user history is three sports posts, and the pipeline retrieves the top 200 by dot product. That's a toy, but it's the *same code path* the production system runs at a vastly larger corpus size behind an ANN index. (If you want the theory of how that top-k is made fast at billions of vectors, this blog's piece on [ANN serving with FAISS, HNSW, and ScaNN](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) is the companion read.)

A May 2026 update added more candidate sources beyond these two: ads, "who to follow," a Phoenix mixture-of-experts retrieval variant, topics, and prompts. But the architecture is unchanged — they're all `Source` implementations feeding the same spine.

### 3. Candidate hydration

Thunder and Phoenix return post IDs and a little metadata. Before scoring, those candidates need to be *hydrated* into full feature-bearing objects: the post's author, engagement counts, video duration, the viewer's subscription status, brand-safety signals, language, mutual-follow scores. This is the `candidate_hydrators/` directory, and the May 2026 update expanded it (engagement counts, brand safety, language detection, mutual-follow scores). Hydration is also where Grox's content-understanding labels get attached — but we'll come back to Grox.

Hydration is mostly an I/O fan-out problem: each hydrator hits a different backend, and the framework runs them concurrently. The interesting design choice is *what doesn't get hydrated*. There is no giant catalog of cross-features ("author-category × viewer-language × time-of-day") being computed here, because the model doesn't consume them. Hydration fetches raw facts; the transformer is responsible for finding the interactions. That is the "no hand-engineered features" philosophy showing up as an absence in a directory listing.

### 4. The ranking model

Everything so far has been plumbing. Here is the brain. Phoenix's ranking model (`recsys_model.py`) is a transformer that takes the user, their history, and a batch of candidate posts, and predicts — for each candidate — the probability of nineteen different engagement actions. The transformer itself (`grok.py`) is ported from xAI's Grok-1 open-source release: RMSNorm, rotary position embeddings, grouped-query attention, a `tanh` soft-cap on attention logits. The recommender adapts that backbone in two ways that matter: how the input sequence is assembled, and how the attention mask is built.

#### Assembling the sequence

The ranker packs three different things into one transformer sequence: the user, their interaction history, and the candidate posts to be scored.

![Assembling the ranking sequence](/imgs/blogs/inside-x-for-you-algorithm-6.webp)

```python
@dataclass
class PhoenixModel(hk.Module):
    """A transformer-based recommendation model for ranking candidates."""
    model: Transformer
    config: PhoenixModelConfig
    fprop_dtype: Any = jnp.bfloat16

    def __call__(self, user_emb, history_emb, candidate_emb,
                 user_mask, history_mask, candidate_mask, positions):
        # Concatenate the three segments into one sequence along the time axis.
        embeddings = jnp.concatenate(
            [user_emb, history_emb, candidate_emb], axis=1
        )
        padding_mask = jnp.concatenate(
            [user_mask, history_mask, candidate_mask], axis=1
        )
        # Everything before this index is "context"; everything after is a candidate.
        candidate_start_offset = user_mask.shape[1] + history_mask.shape[1]

        model_output = self.model(
            embeddings, padding_mask,
            candidate_start_offset=candidate_start_offset, positions=positions,
        )
        out = layer_norm(model_output.embeddings)
        candidate_out = out[:, candidate_start_offset:, :]   # only the candidate tokens
        ...
```

The single integer `candidate_start_offset = user_mask.shape[1] + history_mask.shape[1]` is doing a lot of work. It marks the boundary between "context" (user + history) and "things being scored" (candidates). The mini model uses a history sequence length of up to 127 tokens and a candidate sequence length of up to 64 — so you can score up to 64 posts in a single forward pass, all conditioned on the same user context, with one prefix computed once.

Each token isn't a word; it's a *post* (or the user, or a candidate), represented by concatenating several embeddings — post ID, author ID, action type, and surface — then projecting down. Which brings us to the embedding trick.

#### Hash-based embeddings

A platform has billions of posts and hundreds of millions of users and authors. A standard embedding table with one row per entity would be enormous and mostly cold. Phoenix instead uses **hash-based embeddings**: each entity is hashed (with two hash functions in the mini config) into a fixed-size table, and the looked-up vectors are combined and projected. The mini model declares a 1M-row vocabulary for users, items, and authors, with two hashes per entity.

```python
# Each entity contributes `num_*_hashes` embedding rows; reshape and project down to D.
B = user_embeddings.shape[0]
user_embedding = user_embeddings.reshape((B, 1, num_user_hashes * D))
proj_mat_1 = hk.get_parameter(
    "proj_mat_1", [num_user_hashes * D, D], dtype=jnp.float32, init=init_fn
)
user_embedding = jnp.dot(user_embedding.astype(proj_mat_1.dtype), proj_mat_1)
# History and candidate tokens concatenate post/author/action/surface embeddings,
# then project through their own learned matrices in exactly the same way.
```

Two hashes per entity is the well-known *hashing trick* with a collision-mitigation twist: a single hash function will occasionally collide two unrelated posts into the same row, but the probability that *both* hashes collide for the same pair is much lower, and the learned projection can separate entities that share one row but not the other. It's a memory-for-accuracy trade that lets a 128-dimensional model address an effectively unbounded ID space with a fixed table. The cost is a small, bounded amount of identity blur; the benefit is that you never have to grow the table when a new user signs up.

#### Candidate isolation: the one idea that matters

Here is the design decision that makes the whole thing work. In a normal transformer, every token attends to every earlier token (causal) or every token (bidirectional). If you naively packed 64 candidate posts into one sequence and let them attend to each other, then candidate C's representation — and therefore its score — would depend on *which other candidates happened to be in the batch*. Score post C alongside a batch of cat videos and it looks one way; score it alongside a batch of breaking news and it looks another. That's a disaster for a ranker: scores become non-deterministic, un-cacheable, and dependent on retrieval randomness.

Phoenix forbids it with a custom attention mask. From `grok.py`:

```python
def make_recsys_attn_mask(
    seq_len: int,
    candidate_start_offset: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Create attention mask for recommendation system inference.
    Positions 0 to candidate_start_offset-1 use causal attention; candidates
    attend to user+history and themselves only — not to other candidates."""

    causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=dtype))
    # Zero out the entire candidate-by-candidate block...
    attn_mask = causal_mask.at[:, :, candidate_start_offset:, candidate_start_offset:].set(0)
    # ...then restore the diagonal so each candidate can still attend to itself.
    candidate_indices = jnp.arange(candidate_start_offset, seq_len)
    attn_mask = attn_mask.at[:, :, candidate_indices, candidate_indices].set(1)
    return attn_mask
```

Read it slowly, because it's three lines and the whole architecture hangs on them. Start with a standard lower-triangular causal mask (`jnp.tril`). Then take the bottom-right block — rows and columns from `candidate_start_offset` onward, i.e. candidate-attending-to-candidate — and **zero it entirely**. Finally, set the diagonal of that block back to 1 so each candidate can still attend to *itself*. The net effect: a candidate token can attend to the user, the full history, and itself, but it is blind to every other candidate in the batch.

![The candidate-isolation attention mask](/imgs/blogs/inside-x-for-you-algorithm-7.webp)

The figure shows the mask on a tiny example — one user token `U`, two history tokens `H1 H2`, three candidates `C1 C2 C3`. Green cells are "attend"; the red cells are the isolation zeros. The top-left region is ordinary causal attention over the context. The bottom-left region (candidates → context) is all green: candidates see everything before them. The bottom-right block (candidates → candidates) is red everywhere *except* the diagonal: `C1` cannot see `C2` or `C3`, `C2` cannot see `C1` or `C3`, and so on.

The consequence is profound and worth stating as a theorem: **a candidate's representation depends only on the user, the history, and itself — never on its batch-mates.** Therefore its score is identical no matter what else you score it alongside.

<figure class="blog-anim">
<svg viewBox="0 0 760 400" role="img" aria-label="A candidate's score stays at 0.73 when its batch-mates are swapped, because candidates cannot attend to each other" style="width:100%;height:auto;max-width:840px">
<style>
.x8-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.x8-ctx{fill:var(--surface,#f3f4f6);stroke:var(--text-secondary,#6b7280);stroke-width:1.5}
.x8-c{fill:var(--accent,#6366f1);opacity:.18;stroke:var(--accent,#6366f1);stroke-width:2.5}
.x8-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.x8-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.x8-acc{font:700 18px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.x8-edge{stroke:var(--accent,#6366f1);stroke-width:2.5;fill:none;marker-end:url(#x8-ah)}
.x8-no{stroke:var(--text-secondary,#9ca3af);stroke-width:1.5;stroke-dasharray:5 5;fill:none}
@keyframes x8-fa{0%,42%{opacity:1}52%,92%{opacity:0}100%{opacity:1}}
@keyframes x8-fb{0%,42%{opacity:0}52%,92%{opacity:1}100%{opacity:0}}
.x8-A{animation:x8-fa 9s ease-in-out infinite}
.x8-B{animation:x8-fb 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.x8-A{animation:none;opacity:1}.x8-B{animation:none;opacity:0}}
</style>
<defs>
<marker id="x8-ah" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="var(--accent,#6366f1)"/></marker>
</defs>
<text class="x8-acc x8-A" x="380" y="34">round 1:  batch = { A, C, D }</text>
<text class="x8-acc x8-B" x="380" y="34">round 2:  batch = { X, C, Y }</text>
<rect class="x8-ctx" x="40" y="120" width="180" height="170" rx="10"/>
<text class="x8-lbl" x="130" y="196">user</text>
<text class="x8-lbl" x="130" y="224">+ history</text>
<text class="x8-sub" x="130" y="252">(shared prefix)</text>
<rect class="x8-box" x="450" y="84" width="190" height="58" rx="10"/>
<rect class="x8-c"   x="450" y="176" width="190" height="58" rx="10"/>
<rect class="x8-box" x="450" y="268" width="190" height="58" rx="10"/>
<text class="x8-lbl" x="545" y="210">C — this post</text>
<text class="x8-lbl x8-A" x="545" y="119">neighbor A</text>
<text class="x8-lbl x8-A" x="545" y="303">neighbor D</text>
<text class="x8-lbl x8-B" x="545" y="119">neighbor X</text>
<text class="x8-lbl x8-B" x="545" y="303">neighbor Y</text>
<path class="x8-edge" d="M450,205 L222,205"/>
<text class="x8-sub" x="336" y="194">attends</text>
<path class="x8-no" d="M545,176 L545,142"/>
<path class="x8-no" d="M545,234 L545,268"/>
<text class="x8-sub" x="575" y="164">no attention</text>
<text class="x8-sub" x="575" y="252">no attention</text>
<rect class="x8-c" x="40" y="330" width="280" height="50" rx="10"/>
<text class="x8-acc" x="180" y="362">score(C) = 0.73  (unchanged)</text>
</svg>
<figcaption>Swap C's batch-mates between rounds; because the mask blocks candidate-to-candidate attention, C reaches only the shared user+history prefix, so its score is identical in both rounds.</figcaption>
</figure>

Why is that worth contorting the attention mask for? Several payoffs, all of which are why the system can get away with a single ranking pass:

- **Cacheable scores.** If a post's score for a given user is independent of the batch, you can compute it once and cache it. Re-ranking, pagination, and re-requests reuse the cached score instead of re-running the model. (See this blog's piece on [calibration and trustworthy predictions](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) for why batch-stable scores are also easier to calibrate.)
- **Deterministic ranking.** The same post in the same context always gets the same score. No flicker between refreshes caused by which other candidates happened to be retrieved.
- **Embarrassingly parallel scoring.** Because candidates don't interact, you can split a large candidate set across forward passes arbitrarily — 64 here, 64 there — and concatenate the results. There is no "correct batch" you must preserve.
- **No light ranker needed.** The classic reason for a cheap light ranker is to avoid running the expensive heavy ranker on millions of candidates. But the expensive part of a transformer ranker is encoding the *context* (user + history), and candidate isolation means that context is encoded *once* and shared across all candidates in the batch. Each additional candidate is nearly free — it's one more token attending to a fixed prefix. That economics is what lets X collapse two ranking stages into one.

The price is that the model gives up listwise reasoning. It cannot, within a single forward pass, decide "I'll rank C lower because B is so similar to it." Diversity and de-duplication therefore have to be handled *outside* the model — which is exactly what the Author Diversity Scorer and the conversation-dedup filter do. The architecture trades model-level slate optimization for engineering-level cacheability, and then patches the lost diversity back in with cheap Rust scorers. That is a very deliberate, very senior trade.

#### Nineteen actions, one score

The ranker doesn't predict "relevance." It predicts a vector of nineteen action probabilities for each candidate, then a separate Rust scorer collapses that vector into one number. The model's output head looks like this:

```python
        # candidate_out: [B, num_candidates, D] — the transformer's candidate tokens.
        unembeddings = self._get_unembedding()            # [D, num_discrete_actions]
        logits = jnp.dot(candidate_out, unembeddings)      # [B, num_candidates, A]

        continuous_mat = self._get_continuous_head()       # [D, num_continuous_actions]
        continuous_logits = jnp.dot(candidate_out, continuous_mat)
        continuous_preds = jax.nn.sigmoid(continuous_logits)

        return RecsysModelOutput(logits=logits, continuous_preds=continuous_preds)
```

The output shape is `[B, num_candidates, num_actions]` — for every candidate, a prediction per action. The action indices map to a proto enum; the README documents several: `1=favorite`, `4=reply`, `5=quote`, `6=repost`, `11=dwell`, `13=video view`. The full set spans nineteen actions and crucially includes **negative** ones — `not_interested`, `block_author`, `mute_author`, `report` — alongside positive engagement and continuous signals like dwell time. This is exactly the [multi-task / multi-objective ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) pattern, but instead of MMoE or PLE experts, the "shared bottom" is just the Grok transformer and the "task heads" are linear projections off the candidate tokens.

The collapse from nineteen probabilities to one rankable score is the Weighted Scorer, and it's deliberately simple:

```python
# Final score = Σ_i  weight_i * P(action_i)
#   positive actions (favorite, reply, repost, dwell, ...) -> positive weights
#   negative actions (block, mute, report, not_interested) -> negative weights
def weighted_score(action_probs: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights[a] * p for a, p in action_probs.items())
```

<figure class="blog-anim">
<svg viewBox="0 0 760 440" role="img" aria-label="Nineteen predicted engagement probabilities, positive ones adding and negative ones subtracting, collapse into one weighted final score" style="width:100%;height:auto;max-width:840px">
<style>
.x9-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1}
.x9-pos{fill:#16a34a}
.x9-neg{fill:#dc2626}
.x9-fin{fill:var(--accent,#6366f1)}
.x9-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.x9-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.x9-fina{font:700 18px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
.x9-bar{transform-box:fill-box;transform-origin:left center}
@keyframes x9-grow{0%{transform:scaleX(0)}18%{transform:scaleX(0)}70%{transform:scaleX(1)}100%{transform:scaleX(1)}}
.x9-g{animation:x9-grow 9s ease-out infinite}
.x9-d1{animation-delay:.3s}.x9-d2{animation-delay:.6s}.x9-d3{animation-delay:.9s}.x9-d4{animation-delay:1.2s}.x9-d5{animation-delay:1.5s}.x9-d6{animation-delay:1.8s}.x9-d7{animation-delay:2.1s}
@keyframes x9-final{0%,55%{transform:scaleX(0)}88%{transform:scaleX(1)}100%{transform:scaleX(1)}}
.x9-fg{animation:x9-final 9s ease-out infinite}
@media (prefers-reduced-motion:reduce){.x9-g,.x9-fg{animation:none;transform:scaleX(1)}}
</style>
<text class="x9-lbl" x="24" y="36">19 predicted actions  →  weighted sum</text>
<text class="x9-sub" x="24" y="60">positive weights add · negative weights subtract</text>
<text class="x9-lbl" x="24" y="100">favorite</text><text class="x9-sub" x="430" y="100">P=.42 · w=+1.0</text>
<rect class="x9-track" x="170" y="86" width="240" height="18" rx="4"/><rect class="x9-pos x9-bar x9-g x9-d1" x="170" y="86" width="100" height="18" rx="4"/>
<text class="x9-lbl" x="24" y="134">reply</text><text class="x9-sub" x="430" y="134">P=.18 · w=+3.0</text>
<rect class="x9-track" x="170" y="120" width="240" height="18" rx="4"/><rect class="x9-pos x9-bar x9-g x9-d2" x="170" y="120" width="130" height="18" rx="4"/>
<text class="x9-lbl" x="24" y="168">repost</text><text class="x9-sub" x="430" y="168">P=.12 · w=+2.0</text>
<rect class="x9-track" x="170" y="154" width="240" height="18" rx="4"/><rect class="x9-pos x9-bar x9-g x9-d3" x="170" y="154" width="70" height="18" rx="4"/>
<text class="x9-lbl" x="24" y="202">dwell</text><text class="x9-sub" x="430" y="202">P=.55 · w=+0.5</text>
<rect class="x9-track" x="170" y="188" width="240" height="18" rx="4"/><rect class="x9-pos x9-bar x9-g x9-d4" x="170" y="188" width="150" height="18" rx="4"/>
<text class="x9-lbl" x="24" y="236">video view</text><text class="x9-sub" x="430" y="236">P=.30 · w=+1.0</text>
<rect class="x9-track" x="170" y="222" width="240" height="18" rx="4"/><rect class="x9-pos x9-bar x9-g x9-d5" x="170" y="222" width="86" height="18" rx="4"/>
<text class="x9-lbl" x="24" y="270">mute author</text><text class="x9-sub" x="430" y="270">P=.04 · w=−8.0</text>
<rect class="x9-track" x="170" y="256" width="240" height="18" rx="4"/><rect class="x9-neg x9-bar x9-g x9-d6" x="170" y="256" width="30" height="18" rx="4"/>
<text class="x9-lbl" x="24" y="304">report</text><text class="x9-sub" x="430" y="304">P=.02 · w=−20</text>
<rect class="x9-track" x="170" y="290" width="240" height="18" rx="4"/><rect class="x9-neg x9-bar x9-g x9-d7" x="170" y="290" width="18" height="18" rx="4"/>
<line x1="24" y1="340" x2="600" y2="340" stroke="var(--border,#d1d5db)" stroke-width="1.5"/>
<text class="x9-fina" x="24" y="380">final score  =  Σ wᵢ · P(actionᵢ)</text>
<rect class="x9-track" x="24" y="396" width="576" height="26" rx="6"/><rect class="x9-fin x9-bar x9-fg" x="24" y="396" width="372" height="26" rx="6"/>
<text class="x9-fina" x="612" y="416">0.62</text>
</svg>
<figcaption>Each engagement head outputs a probability; the ranker multiplies by a per-action weight and sums — favorite/reply/dwell push the score up, mute/report push it down — into one number the feed sorts on.</figcaption>
</figure>

The weights themselves are not learned by the model; they're a *product* decision, set and tuned by the people who own the feed. That separation is the whole point. The model's job is to predict, as accurately as possible, "if we show this user this post, how likely is each of these nineteen things?" The product's job is to decide how much a reply is worth relative to a like, and how hard a predicted "report" should bury a post. Want to make the feed favor long-form reading? Raise the dwell weight. Want to suppress rage-bait? Make the predicted-block and predicted-report weights more negative. You can change the *values* of the feed without retraining the model — you just retune the weight vector. The negative actions are the part most platforms keep quiet about: the model explicitly predicts the probability that you'll *block*, *mute*, or *report* a post, and those predictions actively push content down.

### 5. Filter, select, filter again

The model produces scores; now the pipeline has to turn scores into a slate. This happens in three movements, and there are filters on *both* sides of the sort.

![Filter, select, filter again](/imgs/blogs/inside-x-for-you-algorithm-10.webp)

**Pre-scoring filters** run before the model, to avoid spending compute on candidates that can't be shown: duplicates, posts that are too old, the user's own posts, posts from blocked authors, posts matching muted keywords, previously-seen content (the bloom filters from hydration), and ineligible subscription content. The order matters — these are cheap predicates that shrink the candidate set before the expensive scoring pass.

**Selection** sorts the surviving candidates by their weighted score and takes the top *k*. This is the `Selector` trait; it's pure Rust, no model. Before the sort, the Author Diversity Scorer has already nudged scores down for authors who already appear, and the OON Scorer has tuned the in-network/out-of-network balance — so "sort by score" already encodes diversity and the network-mix policy.

**Post-selection filters** run *after* the sort, on the much smaller selected slate, to catch things that are expensive or only make sense to check on the final list: visibility validation (deleted, spam, violence, gore — much of it sourced from Grox), and conversation de-duplication (don't show three replies from the same thread). Doing these after selection is an optimization: you only pay the cost on the handful of posts you're about to show, not on every candidate.

Then ads are blended in (`home-mixer/ads/`), and the ranked, ad-injected feed is returned over the gRPC `ScoredPostsService`. That's the full lifecycle.

### 6. Grox: the content-understanding sidecar

One service we've referenced but not opened: `grox/`. Grox is the content-understanding pipeline — a Python service of classifiers and embedders running on a task-execution engine. Its workloads include spam detection, post categorization, and policy enforcement (the repo calls this PTOS — policy/trust-and-safety scoring). Grox doesn't rank anything; it *labels* posts so that the ranking pipeline can filter and enrich them. Its outputs show up in two places we've already seen: as brand-safety and category signals during candidate hydration, and as the visibility signals (spam, violence, gore) the post-selection filters key on.

Architecturally, Grox is the acknowledgment that "what is this post about, and is it safe?" is a different problem from "will this user engage with it?" — and that the first question is better answered by dedicated content models than by asking the engagement ranker to also be a safety classifier. Keeping them separate means trust-and-safety policy can change (and be audited) independently of the recommendation objective. That is a healthier separation than the older approach of cramming safety features into the ranker's input and hoping the weights behave.

## The serving economics of one ranking pass

The boldest line in the architecture — "no light ranker" — only survives contact with production if the numbers work. They do, and the reason is worth making quantitative, because it's the single most transferable idea in the repository.

A normal transformer scoring `n` candidates packed with a context of length `c` pays attention cost on the order of $L \cdot (c + n)^2 \cdot d$ — the whole sequence attends to the whole sequence, every layer. If you double the candidates, that term grows quadratically. That quadratic is exactly why nobody runs a transformer ranker on a million candidates, and why the light ranker exists: to cut the candidate set down to a size the heavy quadratic can afford.

Candidate isolation changes the exponent. Because the candidate-by-candidate block of the attention matrix is masked to zero, the cost decomposes into two pieces that grow very differently:

$$\text{cost} \approx \underbrace{L \cdot c^2 \cdot d}_{\text{prefix encode (once)}} \;+\; \underbrace{n \cdot L \cdot c \cdot d}_{\text{per-candidate, linear in } n}$$

The first term — encoding the user-and-history prefix — is paid **once** per request and shared across every candidate in the batch. The second term is *linear* in the number of candidates: each candidate is one extra row attending to the fixed `c`-length prefix plus itself, not to the other candidates. Quadratic-in-`n` became linear-in-`n`, and the constant on the linear term is small.

Put real (illustrative) numbers on the mini model: `c = 128` context tokens, `d = 128`, `L = 4` layers. The prefix encode is some fixed cost `P`. The marginal cost of one more candidate is roughly $L \cdot (c \cdot d + d^2) \cdot 2 \approx 4 \cdot (128{\cdot}128 + 128^2) \cdot 2 \approx 2.6 \times 10^{5}$ FLOPs — a rounding error next to a modern accelerator's throughput. Scoring 200 candidates costs "one prefix encode, plus 200 cheap rows," not "a 328-token sequence attending to itself." That is the entire justification for deleting the light ranker: retrieval (a dot-product top-k that's effectively free per candidate behind an ANN index) prunes millions down to hundreds, and then a *single* transformer pass scores those hundreds at near-linear cost.

| Stage | Classic two-stage ranker | x-algorithm |
|---|---|---|
| Prune millions → thousands | Light ranker (cheap model, all candidates) | **Two-tower retrieval** (ANN dot-product) |
| Score thousands → hundreds | (still light ranker) | — (skipped) |
| Score hundreds → final | Heavy ranker, quadratic in slate | **One transformer**, linear in candidates via isolation |
| Re-encode context per candidate? | Often yes (no prefix sharing) | **No** — prefix encoded once, shared |

The lesson generalizes far beyond X: if you can make your candidates independent of each other, you can share the expensive context encode across all of them and turn a quadratic ranker into a linear one. The cost of that trick is the listwise reasoning you give up — but if your model wasn't doing subtle slate optimization anyway, you were paying the quadratic for nothing.

## The Grok transformer underneath

It's worth pulling the camera back to the transformer itself for a moment, because "ported from Grok-1" hides some specific choices that a recommender doesn't obviously need but inherits anyway. From `grok.py`:

```python
@dataclass
class TransformerConfig:
    emb_size: int
    key_size: int
    num_q_heads: int       # query heads
    num_kv_heads: int      # key/value heads (fewer than q heads => grouped-query attention)
    num_layers: int
    widening_factor: float = 4.0
    attn_output_multiplier: float = 1.0
```

The mini model shipped with the repo sets these small: 128-dimensional embeddings, 4 transformer layers, 4 attention heads with a 32-wide key size, a widening factor of 2, and the 127/64 history/candidate sequence lengths we saw. The forward pass runs in `bfloat16`. A few inherited Grok-isms:

- **Grouped-query attention.** Separate `num_q_heads` and `num_kv_heads` means the model can have more query heads than key/value heads, sharing K/V projections across groups. In an LLM this saves KV-cache memory at long context; in the recommender, where the "context" is the candidate-isolation prefix, it's a cheap way to keep more query expressivity than KV cost.
- **Rotary position embeddings (RoPE).** Position is injected by rotating query/key vectors rather than adding a position vector. For a recommender this is a slightly funny inheritance — the history is a sequence with a meaningful order (recency), but the candidates are a *set*, not a sequence. The candidate-isolation mask means candidate positions don't interact with each other anyway, so RoPE on the candidate block is mostly harmless.
- **`tanh` soft-cap on attention logits.** The attention scores are passed through a `tanh` that clamps them to a maximum magnitude (30.0 in the Grok-1 lineage) before the softmax. This is a numerical-stability guard against a single attention logit blowing up and saturating the softmax — a real failure mode when you train in low precision. It's the kind of detail you only add after watching a training run diverge.
- **RMSNorm** instead of LayerNorm — the now-standard cheaper normalization that skips the mean-centering step.

None of these are recsys innovations. They're a battle-tested LLM transformer borrowed wholesale, with the *only* recsys-specific surgery being the input-assembly and the attention mask. That reuse is itself a thesis: that a strong sequence model plus the right masking is a better ranker than a bespoke recsys architecture with hand-built features. We'll see in the case studies how much that thesis costs.

## What the architecture implies about training

The open-source release is an *inference* artifact — exported checkpoints and a runnable pipeline — so the repository doesn't ship the training loop. But the architecture constrains how a model shaped like this *must* be trained, and those constraints are worth spelling out because they're where most of the real difficulty lives. Everything below is standard practice for models of this shape, not a claim about X's exact recipe; read it as "what training this architecture requires," and cross-reference this blog's deeper treatments where noted.

**The retrieval tower is trained contrastively.** A two-tower model learns by pulling a user's embedding toward posts they engaged with and pushing it away from posts they didn't. The cheap, scalable way to get negatives is *in-batch negatives*: within a training batch, each user's positive post is everyone else's negative, so a batch of size `B` yields `B−1` negatives per example for free. The loss is a [sampled softmax / contrastive objective](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval) over those negatives. The subtlety that bites every team: in-batch negatives are sampled in proportion to popularity, so popular posts appear as negatives too often and get unfairly suppressed — the fix is the *logQ correction*, subtracting a popularity term from the logits. If you take one thing from the retrieval side, it's that [negative sampling strategy](/blog/machine-learning/recommendation-systems/negative-sampling-strategies), not architecture, is what usually decides whether two-tower retrieval works.

**The ranker is trained as nineteen binary problems at once.** Each action head is a probability, supervised by whether that action actually happened on that (user, post) impression — a multi-label setup with one binary cross-entropy per head, summed. The labels are the same nineteen actions the model predicts, logged from real impressions. Two hard parts dominate. First, **delayed and rare feedback**: a "favorite" might arrive seconds later, a "report" essentially never, so the negative heads are trained on extreme class imbalance and need careful loss weighting or focal-style down-weighting of easy negatives. Second, **position and selection bias**: the model is trained on logged impressions, which were themselves chosen by a previous version of the model, so the training distribution is not the distribution you'd see under a random policy. This is the [feedback-loop problem](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles), and it's why serious recsys teams invest in [off-policy correction and counterfactual evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation).

**Candidate isolation has to hold at train time too.** This is the easy-to-miss constraint. If the model is trained with full candidate-to-candidate attention and then served with the isolation mask, train/serve skew silently wrecks it — the served representations come from a different computation than the trained ones. So the same `make_recsys_attn_mask` must be applied during training. The payoff is that you can train on many candidates per user impression efficiently (shared prefix, just like serving) and the scores you train are exactly the scores you serve.

**The weights are not trained with the model.** Recall the Weighted Scorer's per-action weights live in the Rust serving plane, not in the model. They're tuned by online experimentation — A/B tests measuring downstream feed health — not by gradient descent. That means the model and the value function are on different release cadences: you can retrain the model weekly and retune the action weights daily, independently. It also means evaluating "is the new model better?" is genuinely hard, because the metric the model optimizes (per-action log-loss) is not the metric the product cares about (long-term engagement under a fixed weight vector). This gap between offline model metrics and online outcomes is the perennial recsys trap, covered in [offline vs online evaluation](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

## Running it yourself

The repository is not just a code dump — it runs. Phoenix ships a small pre-trained model and an end-to-end pipeline you can execute:

```shell
# Phoenix uses uv for dependency + venv management.
cd phoenix
uv run run_pipeline.py --artifacts_dir artifacts/oss-phoenix-artifacts
```

This loads the exported checkpoints, takes an example user history (three sports posts), retrieves the top-200 candidates by dot-product similarity from a 537K-post sports corpus, then ranks them by predicted engagement. You can poke at it:

```shell
# Edit example_sequence.json to change the simulated user's interaction history,
# then re-run with a deeper retrieval and a longer display list.
uv run run_pipeline.py \
  --artifacts_dir artifacts/oss-phoenix-artifacts \
  --top_k_retrieval 500 \
  --top_k_display 50
```

The separate entry points `run_retrieval.py` and `run_ranker.py` let you exercise the two stages in isolation, and `runners.py` holds the glue. The action indices in the output map straight to the proto enum (`1=favorite`, `4=reply`, `5=quote`, `6=repost`, `11=dwell`, `13=video view`), so you can read off *why* a post ranked where it did. There are unit tests (`test_recsys_model.py`, `test_recsys_retrieval_model.py`) that double as the clearest documentation of the expected tensor shapes — when in doubt about what `candidate_start_offset` does to a real array, read the test.

The fact that you can edit `example_sequence.json`, swap in your own "user," and watch the ranking change is the most educational thing in the repo. It turns "candidate isolation" and "weighted multi-action scoring" from concepts into a thing you can perturb and observe.

## What the code reveals

A codebase is a sequence of decisions, and the interesting ones are rarely announced in the README. Here are eight design choices the source makes — each small, each a window into how the system thinks. Read these the way you'd read a postmortem: what was the choice, why is it non-obvious, and what does it cost?

### 1. `candidate_start_offset` is the entire architecture in one integer

Most of the architecture's cleverness reduces to a single index passed through the model: `candidate_start_offset = user_len + history_len`. It defines where "context" ends and "things being scored" begins, and it's the argument to `make_recsys_attn_mask`. Get it wrong by one and you either leak a candidate into the context (it would attend bidirectionally to other candidates) or push a history token into the candidate block (it would lose its history attention and get isolated). The lesson is architectural minimalism: rather than a special candidate-encoder module, the design expresses "candidates are isolated" as a boundary index plus a mask edit. It's the kind of move that looks trivial and is load-bearing — the cacheability of every score in the system depends on that integer being computed correctly. If you're auditing a port of this model, that boundary is the first thing to test.

### 2. Two hashes per entity is a deliberate collision budget

The mini config declares a 1M-row vocabulary with two hashes per user/item/author. A platform has far more than a million of each, so collisions are *guaranteed* — two posts will share a hash row. The design accepts that. The bet is that (a) double-hashing makes a *total* collision (both hashes the same for two entities) rare, and (b) the learned projection can disambiguate entities that collide on one hash but not the other. The cost is a bounded amount of identity blur, worst for the rarest entities (a brand-new author with no engagement history is mostly defined by collisions). The benefit is a fixed memory budget that never grows with the user base — you never have to resize the table or handle "new ID" as a special case. It's the recsys version of accepting a known, bounded error to buy operational simplicity. The cold-start tax this implies is real; see [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) for why new entities are always the hardest.

### 3. The model predicts how much you'll hate a post

It is easy to skim past the action list, but look at what's in it: `not_interested`, `block_author`, `mute_author`, `report`. The ranker spends prediction capacity learning the probability that showing you a post will make you *block its author*. Those heads get negative weights in the Weighted Scorer, so a post the model thinks you're likely to report is actively pushed down. This is the part of feed ranking that platforms rarely make legible, and here it is in nineteen-way output and a weight vector. The non-obvious consequence: the quality of negative-action prediction is gated by how often those actions actually happen, which is very rarely — a "report" is orders of magnitude less frequent than a "favorite." Predicting rare events well is hard, and a mis-calibrated negative head can either fail to suppress genuinely bad content or over-suppress merely unpopular content. The weights on these heads are some of the most consequential, least-observable numbers in the whole system.

### 4. No light ranker is a bet on prefix sharing

Every textbook recsys pipeline has a light ranker because running the heavy model on millions of candidates is too expensive. X deletes the light ranker. The justification is structural: candidate isolation means the costly work — encoding the user-and-history prefix through the transformer — happens *once* per request and is shared across all candidates in the batch. Each candidate adds one token attending to a fixed, already-computed prefix; the marginal cost per candidate is tiny. So the "heavy" ranker is heavy in its prefix and cheap in its candidates, which inverts the usual economics. The cost of this bet: it only holds while the candidate set is small enough that "tiny per candidate" times "number of candidates" stays under budget. Retrieval still has to do the heavy pruning (millions → hundreds) — the light ranker's job didn't vanish, it moved into the two-tower retrieval and the pre-scoring filters. If retrieval ever returns too many candidates, the missing light ranker becomes a latency cliff.

### 5. `bfloat16` everywhere, including the things you'd keep in fp32

The forward pass declares `fprop_dtype = jnp.bfloat16`. The projection matrices are created in `float32` and the activations are cast as they flow through, but the bulk of the compute is bf16. For an LLM this is unremarkable; for a *ranker* it's a small statement that the system tolerates the precision loss in exchange for throughput. The interesting tension is with the score-caching design: if scores are cached and reused, bf16 rounding noise in a score is frozen into the cache, so two posts whose true scores differ by less than the bf16 granularity can have their order fixed by rounding. In practice the weighted-sum step and the product weights swamp that noise, but it's a reminder that "deterministic and cacheable" and "low precision" are quietly in tension, and the design has chosen which one wins.

### 6. The shipped model is a faithful miniature, not the real one

The open-sourced checkpoint is explicitly a *mini* model: 128-dim, 4 layers, 4 heads, a 1M vocab, a 537K sports corpus. The production model is larger on every axis. This is the right thing to ship — it's small enough to run on a laptop, it exercises the exact same code paths, and it doesn't leak the production weights or the full corpus. But it changes how you should read benchmarks: the *architecture* is production-faithful, the *capacity* is not. Conclusions about "how good is X's ranking" can't be drawn from the mini model; conclusions about "how does X's ranking *work*" absolutely can. When a repo ships a miniature, always check which of those two questions it's built to answer — this one answers the second, cleanly, and is honest that it doesn't answer the first.

### 7. Diversity lives outside the model, on purpose

Candidate isolation costs the model its ability to reason about a slate as a whole — it can't say "C and B are near-duplicates, so demote one." The system recovers that with the Author Diversity Scorer (a Rust scorer that penalizes repeated authors before the sort) and the conversation-dedup filter (a post-selection filter that collapses near-duplicate threads). The non-obvious lesson is that this is a *feature*, not a workaround: slate-level concerns like diversity and de-duplication change far more often than the engagement model does, and they're easier to reason about, audit, and tune as explicit Rust rules than as emergent behavior of a listwise model. By pushing diversity out of the model, X made it cheap to change — at the cost of the model never learning subtle, content-dependent diversity that a true listwise ranker might. This blog's piece on [diversity, novelty, and serendipity](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) is the argument for why that tradeoff is rarely free.

### 8. "No hand-engineered features" is the boldest and most expensive line

The README's proudest claim is that the system "relies entirely on the Grok-based transformer to learn relevance" — no hand-engineered features. This is genuinely radical. A decade of recsys lore is about features: ratios, counts, time-decays, crosses, the feature store that serves them, the pipeline that backfills them. X throws it out and feeds the model raw IDs and action sequences, trusting the transformer to discover the interactions. The benefit is enormous pipeline simplification: no feature store, no training/serving skew from feature computation, no feature catalog to maintain. The cost is equally enormous and easy to under-state: the model must be *large enough and trained on enough data* to rediscover, from raw sequences, everything those hand-built features used to encode for free. That's a trade only an organization with X's data scale and compute can make — and it's why this architecture is a fascinating thing to study but a dangerous thing to copy if you don't have a billion daily interactions to train on. The hand-features were never just complexity; they were a sample-efficiency prior, and deleting them moves the cost from engineering to data.

### 9. One transformer class serves two different models

`recsys_retrieval_model.py` and `recsys_model.py` both import and instantiate the *same* `Transformer` class from `grok.py`. The retrieval model wraps it in a two-tower harness and mean-pools its outputs into a user embedding; the ranking model wraps it in the candidate-isolation harness and reads off nineteen action heads. The non-obvious benefit is engineering leverage: one transformer implementation to optimize, one set of kernels to tune, one numerical-stability story (the `tanh` soft-cap, the RMSNorm, the bf16 casting) to get right — and it pays off in two places. When you fix a bug or land a speedup in `grok.py`, both retrieval and ranking inherit it. The cost is coupling: a change that helps ranking but hurts retrieval can't be made in the shared class without a config knob. That the team accepted that coupling tells you they value a single, well-understood transformer over two specialized ones — the same minimalism that produced the three-line attention mask. It's a recurring philosophy in this codebase: fewer, sharper primitives over many bespoke ones.

### 10. Side effects never block the feed

The sixth trait, `SideEffect`, is the one that does no work the user waits on — logging the served slate for training data, writing scores to the cache, updating the seen-content bloom filters. It's defined as `async fn run(&self, candidates: &[Candidate])` with no return value the pipeline awaits on the critical path. That signature is a promise: whatever a side effect does, the feed response does not wait for it. The reason this matters is training data. The single most valuable byproduct of serving a feed is the logged (impression, action) pairs that become tomorrow's training set — and you must log *exactly* what was shown, with the scores that produced it, or your training distribution drifts from your serving distribution. By making logging a first-class but off-critical-path trait, the design guarantees the training log is complete without ever letting it add latency to a user's feed. It's a small structural choice with an outsized consequence: the quality of the next model depends on a side effect that, by contract, can never slow down the current one.

## Old algorithm, new algorithm

Step back and compare the 2026 architecture to the 2023 one it replaced. The contrast is the cleanest way to see what's actually new.

![2023 Twitter algorithm vs 2026 X algorithm](/imgs/blogs/inside-x-for-you-algorithm-11.webp)

The 2023 stack was a *constellation* of specialized systems. Candidate generation pulled from a zoo of sources (UTEG, the user-tweet-entity graph; FRS, the follow-recommendation service; CR, content recommender; and more). Embeddings came from SimClusters (sparse community embeddings via matrix factorization on the follow graph) and TwHIN (dense heterogeneous-graph embeddings). Graph services like GraphJet and RealGraph computed real-time interaction graphs and tie strengths. And ranking was a two-stage affair: a cheap light ranker pruned, then a heavy ranker — a deep network in the MaskNet family — scored the survivors over a catalog of hand-built features.

The 2026 stack collapses most of that:

| Concern | 2023 the-algorithm | 2026 x-algorithm |
|---|---|---|
| In-network candidates | Tweet mixer over follow graph + RealGraph | **Thunder** in-memory store fed by Kafka |
| Out-of-network candidates | UTEG, FRS, CR, SimClusters neighbors | **Phoenix two-tower** retrieval (one model) |
| Graph / community signals | SimClusters, TwHIN, GraphJet, RealGraph | Folded into retrieval embeddings + in-network store |
| Light ranking | Logistic / shallow model | **Removed** — retrieval + filters do the pruning |
| Heavy ranking | MaskNet-family DNN over hand-built features | **One Grok transformer**, no hand-built features |
| Score determinism | Batch- and feature-pipeline-dependent | **Candidate isolation** → cacheable, batch-independent |
| Objective | Engagement prediction with a few heads | **19 action heads**, including negative actions |
| Diversity | Mixed into ranking + heuristics | Explicit Rust scorers outside the model |

What changed is not "Twitter discovered transformers." It's a *consolidation thesis*: that a single sequence model, given enough scale and the right input representation, can subsume the work of a dozen specialized graph and embedding systems and a two-stage ranker. Whether that thesis is *correct* — whether one transformer truly matches a hand-tuned constellation on every slice of the distribution — is exactly the kind of question the open-sourcing invites the community to probe. What's not in doubt is that it's a dramatically simpler system to operate, and that simplicity is itself a competitive advantage: fewer services, fewer pipelines, fewer places for training/serving skew to hide.

## How to read this codebase if you're new to recsys

If you're coming to this repository to *learn* recommendation systems rather than to reverse-engineer X specifically, here's the reading order I'd suggest, mapped to concepts:

1. **Start with `phoenix/run_pipeline.py` and the README.** Run it. Watch a retrieval-then-rank flow end to end on the toy corpus. This grounds everything else.
2. **Read `recsys_retrieval_model.py` for the two-tower pattern.** It's the cleanest production example of [retrieval as a dot product](/blog/machine-learning/recommendation-systems/framing-the-problem-rating-ranking-retrieval) you'll find — user tower, candidate tower, L2-norm, top-k. Compare it to the [SASRec/BERT4Rec sequence models](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) this blog covers; the user tower is a close cousin.
3. **Read `make_recsys_attn_mask` in `grok.py`.** Three lines, and the most novel idea in the repo. If you understand why those three lines matter, you understand the whole architecture.
4. **Read `recsys_model.py` for multi-task output.** The nineteen heads and the discrete-plus-continuous split are textbook [multi-objective ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) without the MMoE/PLE machinery.
5. **Skim the Rust `candidate-pipeline/` traits.** You don't need to read all the Rust, but understanding the six-trait contract tells you how a real feed-serving system is structured around a model, which is the part recsys courses always skip.

The thing the repo teaches that no course does: a recommender is mostly *not* the model. It's the orchestration, the in-network fast path, the filters on both sides of the sort, the content-understanding sidecar, and the framework that lets all of it fail gracefully. The model is one box in eleven.

## When to borrow this design — and when not to

This architecture is beautiful, and most teams should not copy it wholesale. Here's the honest decision boundary.

**Reach for this design when:**

- **You have web-scale interaction data.** "No hand-engineered features" only works if the model can rediscover those features from raw sequences, which requires a *lot* of data. If you have billions of daily interactions, the transformer can learn what hand-features used to encode. If you have millions, the hand-features are a sample-efficiency prior you can't afford to throw away.
- **Score cacheability is worth an architectural constraint.** If your read pattern involves heavy pagination, re-requests, or re-ranking, candidate isolation's batch-independent scores pay for themselves. Borrow the isolation mask even if you borrow nothing else.
- **You want to tune feed values without retraining.** The "model predicts probabilities, product owns the weights" split is broadly applicable and worth stealing. It lets non-ML stakeholders adjust the feed's behavior through a weight vector instead of a training run.
- **You're consolidating a sprawl of legacy recsys services.** If you're staring at your own zoo of graph services and a two-stage ranker, the consolidation thesis is at least worth prototyping. One sequence model is a lot less to operate.

**Skip it when:**

- **Your data is small or sparse.** Below web scale, [feature engineering and simpler models](/blog/machine-learning/recommendation-systems/dcn-and-explicit-feature-crossing) (factorization machines, DCN, gradient-boosted trees) will beat a feature-free transformer that's starved for data. The transformer's appetite is the constraint.
- **You need true listwise / slate optimization in the model.** Candidate isolation explicitly forbids candidates from seeing each other. If your problem genuinely requires the model to reason about a whole slate — complementary items in a shopping cart, a diversified news front page where the model itself must balance topics — isolation is the wrong tool, and a listwise ranker is worth its cost.
- **You can't afford a large transformer in the serving path.** The prefix-sharing economics only help if you can run the transformer at all within latency budget. A small team on modest hardware may get more relevance per dollar from a two-tower retrieval plus a gradient-boosted ranker.
- **Interpretability is a hard requirement.** A feature-free transformer is a black box; "why did this rank here?" is answerable only through the nineteen action probabilities, not through inspectable features. In regulated or high-trust settings, the older feature-based rankers are easier to explain and audit.

The deepest lesson of `xai-org/x-algorithm` isn't any single technique. It's a worldview: that scale plus a strong sequence model plus the *right* structural constraints (candidate isolation, multi-action heads, a clean trait spine) can replace a decade of accumulated recsys machinery — for an organization that has the scale to pay for it. Reading the code is the best way to decide whether that worldview applies to *your* problem, and the fact that you *can* read the code, line by line, is the genuinely unprecedented part.

## Further reading

- [`xai-org/x-algorithm`](https://github.com/xai-org/x-algorithm) — the repository itself; start with `phoenix/README.md` and `run_pipeline.py`.
- [Grok-1 open release](https://github.com/xai-org/grok-1) — the transformer `phoenix/grok.py` is ported from.
- [`twitter/the-algorithm`](https://github.com/twitter/the-algorithm) — the 2023 system, for the before/after comparison.
- [Framing the problem: rating, ranking, retrieval](/blog/machine-learning/recommendation-systems/framing-the-problem-rating-ranking-retrieval) — the retrieval/ranking split this whole pipeline embodies.
- [Self-attention for sequences: SASRec and BERT4Rec](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) — the sequence-model lineage of the user tower.
- [Multi-task and multi-objective ranking: MMoE and PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) — the multi-head scoring pattern, with the expert machinery X chose to skip.
- [Approximate nearest-neighbor serving: FAISS, HNSW, ScaNN](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) — how the two-tower top-k is made fast at scale.

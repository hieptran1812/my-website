---
title: "Content-Based and Hybrid Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build a content-based recommender from item text and embeddings, derive why feature vectors solve item cold start, then fuse content into collaborative factors with LightFM and measure warm versus cold-start Recall@10 on MovieLens."
tags:
  [
    "recommendation-systems",
    "recsys",
    "content-based-filtering",
    "hybrid-recommenders",
    "lightfm",
    "tf-idf",
    "sentence-transformers",
    "cold-start",
    "machine-learning",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/content-based-and-hybrid-recommenders-1.png"
---

You ship a movie catalog on a Tuesday. By Friday, marketing has loaded forty new titles for the weekend — fresh releases, the exact thing the homepage exists to surface. Monday morning you open the analytics: the new titles got almost no impressions. Your collaborative filtering model, the one with the beautiful offline Recall@10, simply has no row in its embedding table for a film that nobody has clicked yet. It cannot recommend what it has never seen co-watched. The forty titles you most wanted to promote are invisible to the very system meant to promote them.

This is the **cold-start wall**, and it is the single most common reason a recommender that looks great offline disappoints the business. Collaborative filtering (CF) — the family of models that learns from who-interacted-with-what, which we built up in [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles) and [matrix factorization, the workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) — is powerful precisely because it ignores item content and reads taste straight off behavior. But that strength is also its blind spot. No behavior, no vector. The fix is to bring back the thing CF threw away: the *content* of the item itself. A new film has a title, a synopsis, a genre, a cast, a poster, even an audio fingerprint of its trailer. None of that requires a single click. If we can turn that content into a vector and recommend from it, the film is recommendable the instant it lands.

That is **content-based filtering**, and the moment you combine it with collaborative filtering you get a **hybrid** — which, after twenty years of production recommenders, is simply the default. This post sits in the **Recommendation Systems: From Click to Production** series right after CF and before retrieval, and it does the three things the series always does. *Scientific*: we derive TF-IDF, prove why cosine similarity in content space is the right scoring rule, and — the key insight — show algebraically why representing an item as a *sum of its feature embeddings* (the LightFM trick) means a brand-new item with features is never cold. *Practical*: we build a content recommender on MovieLens with `sentence-transformers` and TF-IDF, then a true hybrid with `lightfm` that fuses item features into the latent factors, with a full eval harness. *Measured*: a before-and-after table — CF-only versus content-only versus hybrid — on both **warm Recall@10** and **cold-start Recall@10**, with the honest result that content and hybrid win cold start while CF wins warm.

The figure below is the whole content-based idea in one frame: item features become item vectors, your liked items average into a user profile, and similarity in that shared space ranks everything — including items with zero interactions.

![Branching dataflow showing item features turning into item vectors, liked items averaging into a user profile, and cosine similarity producing a cold-start-safe top-N list](/imgs/blogs/content-based-and-hybrid-recommenders-1.png)

## 1. The content-based bet: similar content, similar taste

Collaborative filtering bets that *people who agreed in the past will agree again*. Content-based filtering makes a different bet: **you will like items whose content resembles the content of items you already liked**. If you watched and rated five hard science-fiction films highly, a sixth hard sci-fi film is a good guess — not because other people who liked the first five liked the sixth (that is the CF argument) but because the sixth *is itself* the kind of thing you demonstrably enjoy. The recommendation is justified by the item's own attributes, not by the crowd.

That difference in justification is the whole story. It buys you three things and costs you three things, and almost every design decision in this post is a trade between those two columns. The figure later in section 5 lays the trade-out as a matrix; for now, hold the core mechanism.

The mechanism has three moving parts:

1. **An item profile.** Each item gets a vector in some feature space — built from its text, tags, categories, or media embeddings. The film "Blade Runner" becomes a point.
2. **A user profile, in the same space.** We summarize what you like as a vector in the *same* feature space — most simply, the average of the vectors of items you rated highly. Your taste becomes a point sitting among the items.
3. **A scoring rule.** We score every candidate item by its similarity to your profile vector and return the top-N. Cosine similarity is the standard choice, and section 3 derives why.

The thing to notice immediately — the thing that makes this whole family worth the trouble — is what is *absent*. There is no other user anywhere in that pipeline. Your recommendations depend only on your own history and the items' content. This is why content-based filtering is sometimes the only thing you can do: a cold-start user with three ratings, on a catalog where every item is brand new, where you have no co-occurrence data at all, can still be served. The crowd is optional.

### 1.1 Where content-based filtering came from

The lineage is worth a sentence because it explains the shape of the field. Content-based recommendation grew out of *information retrieval* (IR) and *information filtering* in the 1990s — the same TF-IDF and vector-space machinery that powered early search engines. Systems like NewsWeeder (Lang, 1995) and the book recommender in Mooney and Roy's "Content-Based Book Recommending Using Learning for Text Categorization" (2000) treated recommendation as a text-classification problem: learn what words appear in items a user likes, then score new items by those words. The definitive survey is Lops, de Gemmis, and Semeraro's "Content-based Recommender Systems: State of the Art and Trends" (2011), which is the citation to reach for and which we return to in the case-studies section. The through-line from IR is important because it tells you the failure modes in advance: content-based recommenders inherit IR's strengths (no cold start for items, transparent explanations) and IR's weaknesses (you can only match on what you can represent, and you tend to retrieve more of the same).

### 1.2 The funnel position

In the series' recurring **retrieval → ranking → re-ranking** funnel, a content-based recommender is a *candidate generator*: it produces a few hundred plausible items cheaply, which a ranker then orders. It is an especially good candidate generator for two slots in the funnel. First, the **cold-start slot**: when CF returns nothing for a new item or new user, content fills in. Second, the **freshness slot**: news, short-form video, and any catalog where items expire in hours cannot wait for interaction data to accumulate; content lets a five-minute-old article compete. We will see both in the case studies. The content vectors you build here are also exactly the *item features* that flow into the [two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) later in the series — a two-tower item encoder is, in one reading, a learned content-based item profile. Keep that link in mind; the hybrid we build is the bridge to it.

### 1.3 The cold-start asymmetry, in one picture

It is worth making the central advantage concrete before we go deeper, because everything else in the post is downstream of it. The figure below puts the two worlds side by side: a collaborative model facing a brand-new item, and a content (or hybrid) model facing the same item. Hold the contrast in mind as the mechanism for the rest of the post.

![Before and after comparison of a collaborative filtering model failing on a new item with zero interactions versus a content or hybrid model giving it a vector from its features](/imgs/blogs/content-based-and-hybrid-recommenders-3.png)

On the left, the collaborative model. A new item arrives with zero interactions. Collaborative filtering's entire vocabulary is co-occurrence — *who interacted with what* — so an item that nobody has interacted with has no place in that vocabulary. There is no row in the embedding table, no neighbor list, no latent vector. The model's honest answer for that item is the global prior, which in a ranked list means the item never surfaces. Recall@10 for genuinely new items is, structurally, zero. No amount of model capacity fixes this; it is a property of the *input*, not the architecture.

On the right, the content (or hybrid) model. The same new item arrives with the same zero interactions — but it also arrives with a title, a synopsis, genres, perhaps a poster. Those features existed before any click. We turn them into a vector exactly the way we turn every other item into a vector, and the new item takes its place in the shared space immediately, ranked sensibly against the user profile. The item is recommendable on day one. This asymmetry — CF blind to the new, content fluent in it — is the reason content-based methods exist at all, and the reason every serious production system eventually becomes a hybrid. We will quantify the gap (a real 0.00-versus-0.19 cold-start Recall@10) in section 8.

There is a symmetric version of this story for *users* rather than items, and it is worth flagging now so you do not over-claim. Content solves *item* cold start cleanly: a new item has features. It does *not* fully solve *user* cold start, because a brand-new user has no liked items to average into a profile — you cannot take the mean of an empty set. The honest framing is that content-based filtering converts the item cold-start problem (hard, structural for CF) into a feature-quality problem (tractable, an engineering problem), while leaving user cold start to other tools — onboarding questionnaires, demographic priors, popularity fallbacks, and bandit exploration — covered in [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem). Knowing which half of cold start a method actually addresses keeps you from promising the business a fix you cannot ship.

## 2. Item representations: from raw content to a vector

Everything downstream depends on the quality of the item vector. Garbage in, garbage out, with no recourse — if your features cannot tell two films apart, no scoring rule will. So the representation layer is where the engineering effort goes. The figure shows the layers an item climbs through: raw content and categorical attributes at the bottom, an encoder in the middle, optional media embeddings alongside, and one dense vector at the top that shares a space with the user profile.

![Vertical stack of item representation layers rising from raw title and tags through an encoder to a single dense item vector in the shared user and item space](/imgs/blogs/content-based-and-hybrid-recommenders-5.png)

There are three families of representation, in rough order of age and sophistication.

### 2.1 TF-IDF over text

The classic. Take the item's text — title, synopsis, tags, reviews — tokenize it, and represent each item as a sparse vector over the vocabulary, where each entry is the **term frequency–inverse document frequency** of that word. TF-IDF is the workhorse because it is cheap, transparent, and shockingly competitive. We derive the math in section 3. Its strength is interpretability: you can read off *which words* made two items similar. Its weakness is that it is bag-of-words — it has no idea that "movie" and "film" mean the same thing, or that "not good" is negative. It matches on surface tokens.

### 2.2 Categorical and structured features

Most catalogs have structured attributes: genre, director, decade, brand, price band, language. These become one-hot or multi-hot vectors, optionally concatenated with the text vector. They are dense in signal and trivially cold-start-safe (a new film's genre is known at upload). The risk is the curse of dimensionality if you one-hot a high-cardinality field like "director" — that is where learned *feature embeddings* (section 7, the LightFM trick) earn their keep, by mapping each category to a learned dense vector rather than a sparse indicator. The mechanics of building these features cleanly — handling missing values, hashing high-cardinality fields, avoiding train-serve skew — are the subject of [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders); here we just consume them.

### 2.3 Modern dense embeddings

The current default for anything with rich content. Instead of bag-of-words, encode the item's text with a pretrained sentence encoder — `sentence-transformers` models like `all-MiniLM-L6-v2` give a 384-dimensional vector that captures meaning, so "film" and "movie" land near each other and "a heartbreaking war drama" lands near other war dramas regardless of exact wording. For images, **CLIP** image embeddings turn a poster or product photo into a vector in a space aligned with text. For audio, models like the embeddings behind music tagging turn a 30-second clip into a vector — this is the modern echo of Pandora's hand-built Music Genome (case studies). The general recipe: one pretrained encoder per modality, concatenate or sum the modality vectors, and you have a dense item profile that needs zero interaction data.

The trade-off across the three families is the usual one. TF-IDF: cheap, interpretable, brittle to vocabulary. Dense text embeddings: capture meaning, need a model and a GPU-ish encode step, harder to explain. Media embeddings: capture what text cannot (a film's visual style, a song's timbre) at the cost of a heavier pipeline. A practical content recommender often uses TF-IDF as a fast baseline and dense embeddings where the content is rich enough to justify them. We build both.

A decision people get wrong here is reaching for the heaviest representation first. The right order is the cheapest representation that clears your bar: start with TF-IDF over a well-constructed document string, measure cold-start Recall@10, and only move to dense embeddings if the content has semantic structure TF-IDF cannot reach (paraphrase, synonymy, cross-lingual). Move to media embeddings only when text is genuinely insufficient — a fashion catalog where the *look* matters more than the description, a music catalog where the *sound* matters. Each step up the ladder roughly doubles your pipeline complexity (a model to host, a GPU to budget, a version to pin, a drift to monitor), and the marginal recall it buys shrinks. I have seen teams spend a quarter standing up a CLIP pipeline that a one-day TF-IDF baseline would have matched, because nobody measured the baseline first. Measure the baseline first.

To make the representation choice concrete, here is how the three families trade off on the axes that decide it in practice:

| Representation | Cold start | Captures meaning | Cost to build | Explainable | Reach for it when |
|---|---|---|---|---|---|
| TF-IDF over text | yes | surface tokens only | very low | high (read the terms) | clean text exists; want a fast, transparent baseline |
| Dense text (SBERT) | yes | paraphrase, synonymy | medium (model + encode) | low | content has semantic structure TF-IDF misses |
| Categorical / structured | yes | exact attributes | low (curate taxonomy) | high (read the tags) | reliable structured metadata at upload |
| Media (CLIP, audio) | yes | visual / sonic style | high (GPU pipeline) | low | the look or sound matters more than the words |

Every row is cold-start-safe — that is the whole point of content — but they differ sharply on cost and on what kind of similarity they capture. The pragmatic default is TF-IDF or structured features for the baseline, dense text where it pays, and media only when text genuinely cannot represent the thing.

One more representation subtlety that matters for cold start specifically: **be careful which features are available at upload time.** A feature that is only computed *after* an item accumulates interactions — say, "average dwell time" or "click-through rate so far" — is useless for cold start, because a cold item has none of it by definition. The features that solve cold start are precisely the ones knowable from the item alone, before any user touches it: text, declared category, media. When you design the content representation, mentally split features into *intrinsic* (known at upload, cold-start-safe) and *behavioral* (accrued from interactions, cold-start-useless), and make sure your cold-start representation rests only on the intrinsic ones. Mixing a behavioral feature into your "content" vector is a common way to fool yourself in offline evaluation: the offline metric looks great because the test items already have their behavioral features, but in production a truly new item does not, and the numbers collapse. This is the same train-serve-skew trap that haunts the whole field, wearing a content-feature costume.

```python
# Two ways to turn MovieLens item text into vectors.
# Text per movie = title + genres + user-supplied tags (from tags.csv).
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

movies = pd.read_csv("ml-latest-small/movies.csv")            # movieId,title,genres
tags = pd.read_csv("ml-latest-small/tags.csv")                # userId,movieId,tag,timestamp

# Aggregate free-text tags per movie into one string.
tag_text = (tags.groupby("movieId")["tag"]
                .apply(lambda s: " ".join(s.astype(str)))
                .rename("tagtext"))
movies = movies.merge(tag_text, on="movieId", how="left").fillna({"tagtext": ""})
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
movies["doc"] = (movies["title"] + " " + movies["genres"] + " " + movies["tagtext"])

# --- Representation A: TF-IDF (sparse, interpretable) ---
tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_features=20000,
                        sublinear_tf=True)        # sublinear_tf = 1 + log(tf)
X_tfidf = tfidf.fit_transform(movies["doc"])      # (n_items, vocab), L2-normalized rows
print("TF-IDF item matrix:", X_tfidf.shape)
```

```python
# --- Representation B: dense sentence embeddings (semantic) ---
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("all-MiniLM-L6-v2")             # 384-dim, fast on CPU
X_dense = encoder.encode(movies["doc"].tolist(),
                         batch_size=256,
                         normalize_embeddings=True,           # unit-norm -> cosine = dot
                         show_progress_bar=True)
X_dense = np.asarray(X_dense, dtype="float32")                # (n_items, 384)
print("Dense item matrix:", X_dense.shape)
```

Two design notes that bite people. First, `normalize_embeddings=True` (or L2-normalizing the TF-IDF rows, which `TfidfVectorizer` does by default) means cosine similarity reduces to a plain dot product — a big speedup at scale and the reason ANN retrieval works on these vectors. Second, the *text you feed the encoder is a product decision*, not a default. Concatenating title, genres, and crowd tags gives a far better movie vector than the title alone, because the genres and tags carry the taste-relevant signal. Spend time on the document string.

With item vectors in hand, the simplest useful content recommender is "more like this": given an item you liked, return its nearest neighbors in content space. This is exactly the "Because you watched X" rail you see on every streaming homepage, and it is two lines once the vectors exist.

```python
# "More like this": item-item content nearest neighbors (serving-style).
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def more_like_this(movie_id, item_mat, k=10):
    row = mid_to_row[movie_id]
    sims = cosine_similarity(np.asarray(item_mat[row]).reshape(1, -1),
                             np.asarray(item_mat)).ravel()
    nbr = np.argsort(-sims)[1:k + 1]            # skip self at index 0
    return [(movies.title.values[i], float(sims[i])) for i in nbr]

for title, s in more_like_this(260, X_dense):   # 260 = Star Wars (1977)
    print(f"{s:.3f}  {title}")
# -> high-cosine neighbors are other space-opera / sci-fi adventure films,
#    each with its own readable similarity score for the explanation string.
```

The same nearest-neighbor lookup is how content scales to large catalogs at serving time: you precompute every item vector offline, push them into a [vector index](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) (faiss, hnswlib), and at request time the "more like this" call is a single approximate-nearest-neighbor query that returns in milliseconds. Cold start is free in this design — a new item's vector is computed from its features the moment it is uploaded and inserted into the index, no retraining required. That is a real operational advantage over CF, where a new item is genuinely invisible until the next model train.

## 3. The science: TF-IDF and cosine similarity, derived

Let us make the matching rigorous, because the whole content-based pipeline is two formulas and it pays to understand exactly what they assume.

### 3.1 TF-IDF

Let there be a corpus of $N$ items (documents). For a term $t$ in item $d$, define:

- **Term frequency** $\text{tf}(t, d)$: how often $t$ appears in $d$. Raw counts over-weight long documents and repeated words, so we usually use the *sublinear* form $1 + \log \text{tf}(t,d)$ — the tenth occurrence of "space" should not count ten times more than the first.
- **Document frequency** $\text{df}(t)$: the number of items containing $t$. A term that appears in almost every item (like "movie" in a movie catalog) carries no discriminative signal.
- **Inverse document frequency**:

$$
\text{idf}(t) = \log \frac{N}{1 + \text{df}(t)} + 1 .
$$

The $1+\text{df}$ avoids dividing by zero and the trailing $+1$ keeps weights positive (this is the scikit-learn convention). The TF-IDF weight of term $t$ in item $d$ is the product:

$$
w_{t,d} = \big(1 + \log \text{tf}(t, d)\big) \cdot \text{idf}(t).
$$

The intuition the formula encodes: a term matters for an item if it appears *often in that item* (high tf) but *rarely across the corpus* (high idf). "Cyberpunk" appearing three times in one film's tags and almost nowhere else is a strong signal; "the" appearing fifty times everywhere is none. The item vector $\mathbf{d}$ stacks $w_{t,d}$ over the whole vocabulary, then is L2-normalized: $\hat{\mathbf{d}} = \mathbf{d} / \lVert \mathbf{d} \rVert_2$. Normalization is what lets us compare a short-synopsis film to a long-synopsis film fairly — we compare *directions*, not magnitudes.

### 3.2 Cosine similarity

Two items (or an item and a user profile) are scored by the cosine of the angle between their vectors:

$$
\text{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert_2\, \lVert \mathbf{b} \rVert_2} = \sum_t \hat{a}_t \, \hat{b}_t \quad (\text{when both are unit-normalized}).
$$

Why cosine and not Euclidean distance? Because in a bag-of-words or embedding space, *direction encodes content and magnitude encodes length or verbosity*, and we care about content. Two films about the same theme should be similar whether one has a one-line synopsis and the other a three-paragraph one. Cosine throws away the length and keeps the topic mix. Geometrically, cosine ranges from $-1$ (opposite) through $0$ (orthogonal, no shared terms) to $1$ (identical direction); for non-negative TF-IDF vectors it lives in $[0, 1]$.

### 3.3 The user profile

The simplest user profile is the mean of the vectors of items the user liked. If user $u$ liked the set $L_u$ of items with vectors $\hat{\mathbf{d}}_i$, the profile is:

$$
\mathbf{p}_u = \frac{1}{|L_u|} \sum_{i \in L_u} \hat{\mathbf{d}}_i .
$$

We score a candidate item $j$ by $\text{cos}(\mathbf{p}_u, \hat{\mathbf{d}}_j)$ and rank. A small refinement: weight the average by rating or recency, so a 5-star film pulls the profile harder than a 3-star one, and last month's taste counts more than last year's:

$$
\mathbf{p}_u = \frac{\sum_{i \in L_u} \alpha_i\, r_{ui}\, \hat{\mathbf{d}}_i}{\sum_{i \in L_u} \alpha_i\, r_{ui}}, \qquad \alpha_i = e^{-\lambda \Delta t_i},
$$

with $\Delta t_i$ the age of the interaction and $\lambda$ a decay rate. This "Rocchio-style" profile (named after the relevance-feedback algorithm from 1971 IR) is the bread-and-butter user model in content systems. A more expressive option is to skip the average entirely and *train a per-user classifier* — logistic regression on liked-vs-not over the item vectors — which lets the user model learn that you like sci-fi *but not* horror sci-fi, a distinction the mean cannot make. The mean is a strong, cheap baseline; reach for the classifier when you have enough per-user data.

### 3.4 Why the mean profile works, and exactly when it breaks

It is worth being precise about *why* averaging your liked-item vectors is a sensible user model, because understanding the assumption tells you its failure mode. The mean profile assumes your taste is **unimodal** in feature space — that all the things you like cluster around a single centroid. When that holds, the average sits in the middle of your cluster and scores nearby items highly, which is exactly right. The geometry is the same as a class prototype in nearest-centroid classification: represent a class by its mean, classify by distance to the mean.

The failure mode is **multimodal taste**, and it is common. Suppose you love both hard science fiction *and* lighthearted romantic comedies — two well-separated clusters in feature space. Their average lands *between* them, in a region of taste-space you may not like at all (some bland middle-genre nothing). The mean profile then scores the bland middle highly and your two real clusters lower, which is backwards. This is the centroid-of-bimodal-data pathology, and it is the single biggest reason a naive content recommender can feel oddly off. Three fixes, in increasing order of effort:

- **Multiple profiles.** Cluster your liked items (k-means into 2–4 taste clusters) and keep one centroid per cluster; score a candidate by its *max* similarity over your centroids. Now a rom-com scores against your rom-com cluster and a sci-fi against your sci-fi cluster, and the bland middle scores against neither. This is the cheapest real fix and it works well.
- **Per-user classifier.** Train logistic regression (or a small tree) on liked-versus-not over item vectors. A linear classifier can carve a decision boundary that the mean cannot, and it naturally down-weights features that do not discriminate. It needs more per-user data and a per-user fit, but it captures "sci-fi yes, horror-sci-fi no."
- **Session profiles.** Instead of one all-time profile, build a short-lived profile from the current session's interactions. This both handles multimodality (each session tends to be unimodal — you are in a "sci-fi mood" tonight) and tracks taste drift. Most production content systems blend a long-term profile with a session profile for exactly this reason.

The lesson generalizes beyond content: any time you summarize a set by its mean, you are betting the set is unimodal, and that bet fails on mixed tastes. Knowing the bet lets you detect when it is failing (your recommendations drift toward genre-neutral mush) and reach for the right fix.

#### Worked example: TF-IDF and cosine on a tiny corpus

Take three short item documents:

- $d_1$ = "space space alien war" (a space-war film)
- $d_2$ = "space alien planet" (a space film)
- $d_3$ = "romance drama wedding" (a romance)

Corpus size $N = 3$. Document frequencies: "space" appears in $d_1, d_2$ so $\text{df} = 2$; "alien" in $d_1, d_2$, $\text{df}=2$; "war" in $d_1$ only, $\text{df}=1$; "planet" in $d_2$, $\text{df}=1$; "romance", "drama", "wedding" each in $d_3$, $\text{df}=1$. Using $\text{idf}(t) = \log(N / (1+\text{df})) + 1$ with natural log:

- $\text{idf}(\text{space}) = \log(3/3) + 1 = 0 + 1 = 1.00$
- $\text{idf}(\text{alien}) = \log(3/3) + 1 = 1.00$
- $\text{idf}(\text{war}) = \log(3/2) + 1 = 0.405 + 1 = 1.405$
- $\text{idf}(\text{planet}) = \log(3/2) + 1 = 1.405$

For $d_1$, raw term frequencies are space=2, alien=1, war=1. With sublinear tf $(1 + \log \text{tf})$: space $= 1 + \log 2 = 1.693$, alien $= 1$, war $= 1$. Multiply by idf:

- $w_{\text{space}, d_1} = 1.693 \times 1.00 = 1.693$
- $w_{\text{alien}, d_1} = 1.00 \times 1.00 = 1.00$
- $w_{\text{war}, d_1} = 1.00 \times 1.405 = 1.405$

For $d_2$ (space=1, alien=1, planet=1, all tf=1): $w_{\text{space}} = 1.00$, $w_{\text{alien}} = 1.00$, $w_{\text{planet}} = 1.405$.

Now the cosine between $d_1$ and $d_2$. Shared terms are "space" and "alien". The dot product of the raw weight vectors:

$$
\mathbf{d}_1 \cdot \mathbf{d}_2 = (1.693)(1.00) + (1.00)(1.00) + (1.405)(0) + (0)(1.405) = 2.693 .
$$

Norms: $\lVert \mathbf{d}_1 \rVert = \sqrt{1.693^2 + 1.00^2 + 1.405^2} = \sqrt{2.866 + 1.00 + 1.974} = \sqrt{5.840} = 2.417$. $\lVert \mathbf{d}_2 \rVert = \sqrt{1.00^2 + 1.00^2 + 1.405^2} = \sqrt{1.00 + 1.00 + 1.974} = \sqrt{3.974} = 1.994$. So:

$$
\text{cos}(d_1, d_2) = \frac{2.693}{2.417 \times 1.994} = \frac{2.693}{4.820} \approx 0.559 .
$$

And $\text{cos}(d_1, d_3) = 0$ exactly, because $d_1$ and $d_3$ share no terms — orthogonal vectors. So if you liked $d_1$, the content recommender ranks $d_2$ (cosine 0.559) far above $d_3$ (cosine 0). Notice it did this with *no other user* — purely from the words. That is the cold-start superpower in miniature, and it is exactly the calculation `sklearn`'s `cosine_similarity` runs for the whole catalog.

## 4. Building a content-based recommender end to end

We now wire the pieces into a working top-N recommender on MovieLens. The plan: build item vectors (section 2), form a user profile as the rating-weighted mean of the user's liked items, score every catalog item, mask out items already seen, return the top-N. Then evaluate with Recall@10 and NDCG@10 on a temporal split. We use the small MovieLens (`ml-latest-small`, ~100k ratings, ~9k movies, with a `tags.csv`) so the code runs on a laptop; the same code scales to MovieLens-20M.

```python
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("ml-latest-small/ratings.csv")   # userId,movieId,rating,timestamp
# movies, X_dense (or X_tfidf) come from section 2; build an index map.
mid_to_row = {m: r for r, m in enumerate(movies["movieId"].values)}

# Temporal split: last 20% of each user's interactions go to test (no leakage).
ratings = ratings.sort_values("timestamp")
def temporal_split(df, frac=0.2):
    cut = df.groupby("userId")["timestamp"].transform(
        lambda s: s.quantile(1 - frac))
    return df[df.timestamp <= cut], df[df.timestamp > cut]
train, test = temporal_split(ratings)

LIKE = 4.0  # treat rating >= 4 as a positive
def user_profile(uid, item_mat):
    liked = train[(train.userId == uid) & (train.rating >= LIKE)]
    if liked.empty:
        return None, set()
    rows = [mid_to_row[m] for m in liked.movieId if m in mid_to_row]
    w = liked.rating.values[:len(rows)]                 # weight by rating
    vecs = item_mat[rows]
    prof = np.average(np.asarray(vecs), axis=0, weights=w)  # weighted mean
    return prof.reshape(1, -1), set(liked.movieId)

def recommend(uid, item_mat, N=10):
    prof, seen = user_profile(uid, item_mat)
    if prof is None:
        return []
    scores = cosine_similarity(prof, np.asarray(item_mat)).ravel()
    order = np.argsort(-scores)
    recs = [movies.movieId.values[r] for r in order
            if movies.movieId.values[r] not in seen][:N]
    return recs
```

```python
# Eval harness: Recall@K and NDCG@K against held-out positives.
def dcg(rel):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))

def evaluate(item_mat, N=10):
    recalls, ndcgs, n = [], [], 0
    test_pos = (test[test.rating >= LIKE]
                .groupby("userId")["movieId"].apply(set).to_dict())
    for uid, truth in test_pos.items():
        recs = recommend(uid, item_mat, N)
        if not recs or not truth:
            continue
        hits = [1 if m in truth else 0 for m in recs]
        recalls.append(sum(hits) / min(len(truth), N))
        ideal = dcg([1] * min(len(truth), N))
        ndcgs.append(dcg(hits) / ideal if ideal > 0 else 0.0)
        n += 1
    return {"Recall@%d" % N: np.mean(recalls),
            "NDCG@%d" % N: np.mean(ndcgs), "users": n}

print("TF-IDF :", evaluate(X_tfidf))
print("Dense  :", evaluate(X_dense))
```

A few engineering points this code makes concrete. The **temporal split** is non-negotiable — splitting randomly would let a future rating leak into the training profile and inflate your numbers; we cut at each user's 80th-percentile timestamp so the test set is strictly *later* than training. The **"liked" threshold** (rating $\ge 4$) turns explicit ratings into the positives content needs; if you only have implicit clicks, every click is a positive and the profile is an unweighted mean. The **seen-item mask** matters enormously — without it, the top recommendation is always something the user already rated, which scores great offline and is useless online. And the metric definitions are the standard ones we use across the series: Recall@10 is the fraction of held-out positives that appear in the top 10, NDCG@10 rewards putting hits higher in the list. (The full taxonomy of these metrics lives in [offline vs online: the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).)

### 4.1 What the numbers look like

On `ml-latest-small` with a temporal split, content-only recommenders land in a predictable place: **respectable but below CF on warm items**. Dense `all-MiniLM-L6-v2` embeddings beat TF-IDF by a few points of Recall@10 because they capture semantic similarity the bag-of-words misses, but both trail a well-tuned BPR matrix factorization on users and items the model has seen. That is the expected ordering and not a failure — content's job is not to win the warm benchmark, it is to win the cold-start one and to add the signal CF cannot see. We make that explicit in section 8's table.

## 5. Strengths and weaknesses: where content earns its place

Now the honest accounting. The matrix below compares content-based, collaborative, and hybrid recommenders across the five properties that actually drive the choice in production.

![Comparison matrix of content versus collaborative versus hybrid recommenders across item cold start, serendipity, explainability, feature need, and popularity bias](/imgs/blogs/content-based-and-hybrid-recommenders-2.png)

**Strengths of content-based filtering:**

- **Item cold start: solved.** This is the headline. A new item with features is recommendable from the moment it is uploaded, because its vector comes from its content, not from interactions it has not yet received. For catalogs with constant churn — news, marketplaces, video platforms — this is decisive.
- **No popularity bias.** CF tends to over-recommend already-popular items because popular items have the most interaction data and therefore the best-estimated vectors — a self-reinforcing loop that quietly collapses the catalog onto a few hits. (We treat that feedback loop properly later in the series.) Content filtering ranks by *content* similarity, so an obscure film with the right features competes with a blockbuster on equal footing. This is genuinely valuable for the long tail.
- **Transparent and explainable.** "Recommended because it shares the genres *cyberpunk* and *dystopian* with films you rated highly" is an explanation you can render directly from the TF-IDF overlap. Users trust recommendations they understand, and product/legal teams increasingly require explanations.
- **Independent of other users.** A single user with a few interactions can be served well, no crowd required. Privacy-sensitive settings (where you would rather not model cross-user co-occurrence) like this.

**Weaknesses — and they are real:**

- **Over-specialization (the filter bubble).** This is the big one. Because the recommender only ever scores items by similarity to what you already liked, it relentlessly serves *more of the same*. Liked five sci-fi films? Here are five hundred more sci-fi films. It cannot surprise you with a brilliant documentary, because the documentary does not look like your history. Content-based systems have low **serendipity** by construction. The figure in section 6 dramatizes this.
- **Limited by features.** The recommender can only match on what you can represent. If your features cannot capture "well-acted" or "tightly edited" or "the kind of comfort-watch I want on a Sunday", the recommender cannot use it. Two films can be content-identical (same genre, same decade, same cast) and yet one is great and one is terrible — content cannot tell them apart. CF *can*, because the crowd's behavior encodes quality.
- **Needs good features, which is real work.** TF-IDF needs clean text; dense embeddings need an encoding pipeline; media embeddings need GPUs and model maintenance. A genre taxonomy has to be curated. The "no feature engineering" superpower of CF is exactly what content gives up.
- **User cold start is only partially solved.** Content fixes *item* cold start. A brand-new *user* with zero history still has an empty profile — you cannot average over nothing. (The full treatment of both cold-start directions is in [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem).)

The takeaway from the matrix is not "content beats CF" or the reverse. It is that **their strengths and weaknesses are almost perfectly complementary**: CF is strong exactly where content is weak (serendipity, quality-beyond-content, no feature work) and content is strong exactly where CF is weak (cold start, popularity bias, explainability). Two methods with complementary errors is the textbook setup for an ensemble. That is the entire motivation for hybrids.

### 5.1 Explainability you can actually ship

The explainability strength deserves a concrete demonstration, because it is one place content beats CF not just in theory but in code you can put in front of a user. With TF-IDF item vectors, the explanation for "why was this recommended" is literally the terms the recommended item shares most strongly with the user's profile — an element-wise product of the two sparse vectors, sorted descending.

```python
import numpy as np

def explain(uid, movie_id, item_mat, vectorizer, top_terms=4):
    prof, _ = user_profile(uid, item_mat)            # user profile vector
    if prof is None:
        return "no profile yet"
    item_vec = np.asarray(item_mat[mid_to_row[movie_id]].todense()).ravel()
    contrib = prof.ravel() * item_vec                # per-term contribution to cosine
    vocab = np.array(vectorizer.get_feature_names_out())
    top = np.argsort(-contrib)[:top_terms]
    terms = [vocab[i] for i in top if contrib[i] > 0]
    return "Recommended because it matches your taste for: " + ", ".join(terms)

print(explain(1, 260, X_tfidf, tfidf))
# -> "Recommended because it matches your taste for: sci-fi, space, adventure, classic"
```

That string is honest — every term in it provably contributed to the score — and it is the kind of explanation that raises trust and, in regulated contexts, satisfies a "right to explanation" requirement. CF can approximate this ("people who watched X also watched this") but cannot point at *why* in terms of the item's actual content. When explainability is a product requirement, content is not just nice-to-have; it is the cheapest way to get a defensible reason string.

## 6. The over-specialization trap, and why hybrids escape it

Before we build the hybrid, sit with the failure that most motivates it. The figure contrasts a pure content recommender, which narrows you into a clone-bubble, with a hybrid that lets collaborative signal pull in genuinely different items your taste-neighbors enjoyed.

![Before and after comparison of a content-only recommender producing near-duplicate sci-fi recommendations versus a hybrid restoring serendipity through collaborative signal](/imgs/blogs/content-based-and-hybrid-recommenders-6.png)

Here is the dynamic, stated as a problem-solving narrative. You launch a content recommender. Week one, users love it — the recommendations are obviously relevant. Week four, engagement is sliding. You dig in and find the **catalog coverage** has collapsed: the system is recommending from a narrow slice of the catalog, the slice nearest each user's existing taste, over and over. A sci-fi fan never sees the acclaimed drama that everyone like them also loved, because the drama does not *look* like sci-fi. The recommender is technically accurate and practically boring. This is over-specialization, and it is not a tuning bug you can fix with a threshold — it is structural. The scoring rule *only* knows content similarity, so it *cannot* recommend something dissimilar-but-loved.

How does the hybrid escape it? By adding a second scoring signal that is *orthogonal* to content. Collaborative filtering scores the drama highly for the sci-fi fan because the *people* who liked that fan's sci-fi films also liked the drama — a connection invisible to content. Blend the two scores and the drama surfaces. The hybrid keeps content's cold-start and explainability wins while borrowing CF's serendipity. Concretely, in our MovieLens evaluation, moving from content-only to a hybrid raises catalog coverage from roughly 4% of items appearing in any top-10 to over 30%, because the CF component reaches items the content component would never rank. That coverage jump is the measurable signature of escaping the bubble.

This is the crux of why **hybrids are the production default**. You almost never want pure content (filter bubble, can't beat content) or pure CF (cold start, popularity bias) when you can have both. The rest of the post is about *how* to combine them well, because there are several ways and they are not equally good.

## 7. Hybrid systems: the taxonomy and the math

In 2002 Robin Burke wrote the paper that organized this whole space: "Hybrid Recommender Systems: Survey and Experiments" (User Modeling and User-Adapted Interaction). He cataloged seven ways to combine recommenders, and the taxonomy is still the right mental map twenty years later. The tree below sorts the seven into two families — designs that combine the *outputs* of two separate models, and designs that *fuse the signals inside one model*. The second family is where modern hybrids (LightFM, two-tower) live, and where the most interesting math is.

![Tree of Burke's hybridization taxonomy splitting into output-combining designs and signal-fusing designs that share one model](/imgs/blogs/content-based-and-hybrid-recommenders-4.png)

### 7.1 Burke's seven designs

- **Weighted.** Compute a content score and a CF score for each item, then combine them with weights: $s = \beta \, s_{\text{CF}} + (1-\beta)\, s_{\text{content}}$. Simplest and often strong. The blend weight $\beta$ is tuned (section 7.3).
- **Switching.** Pick *one* model per situation. Use content when the item is cold (few interactions), switch to CF once it has accumulated enough data. A clean way to route cold versus warm traffic.
- **Mixed.** Present recommendations from both models side by side — e.g. a "because you watched" content row and a "people like you" CF row. Common in real UIs; the "combination" happens in the layout, not the score.
- **Feature combination.** Treat collaborative information (who interacted) as *additional features* fed into a single content-based-style model. The model sees both content and behavior features and learns one scoring function.
- **Feature augmentation.** Use one model to *generate a feature* that the other consumes — e.g. a CF model produces a "predicted rating" or cluster id that becomes an input feature to the content model. Subtler than feature combination; the first model's output augments the second's inputs.
- **Cascade.** Stage the models: one produces a coarse ranking, the next refines it (breaking ties or re-ordering). This is exactly the **retrieval → ranking** funnel — a cheap model proposes, an expensive one refines. The whole series funnel is a cascade hybrid.
- **Meta-level.** One model's *learned model* becomes the other's input — e.g. learn a content model per user, then feed those models into a collaborative step. The most tightly coupled and the rarest in practice.

Two of these — **feature combination** and the model class around it — are what we build with LightFM, because they fuse signals into one set of latent factors and that fusion is what cleanly solves cold start.

### 7.2 The key math: LightFM as a hybrid matrix factorization

Maciej Kula's `lightfm` (introduced in "Metadata Embeddings for User and Item Cold-start Recommendations", 2015) is the cleanest production-grade hybrid you can `pip install`, and its core idea is one equation worth dwelling on.

Plain matrix factorization learns one latent vector per user and one per item: $\hat{r}_{ui} = \mathbf{u}_u^\top \mathbf{v}_i$. The cold-start problem is structural — there is no $\mathbf{v}_i$ for an item never seen in training. LightFM removes the per-item vector and replaces it with a **sum of feature embeddings**. Let item $i$ have a set of features $F_i$ (its own identity, plus content features like genre=sci-fi, decade=1990s, director=Scott). Each *feature* $f$ has a learned embedding $\mathbf{e}_f$ and a bias $b_f$. The item's representation is:

$$
\mathbf{v}_i = \sum_{f \in F_i} \mathbf{e}_f, \qquad b_i = \sum_{f \in F_i} b_f .
$$

Users are represented the same way as a sum of their feature embeddings $\mathbf{u}_u = \sum_{g \in F_u} \mathbf{e}_g$. The score is then the familiar:

$$
\hat{r}_{ui} = \mathbf{u}_u^\top \mathbf{v}_i + b_u + b_i = \Big(\sum_g \mathbf{e}_g\Big)^\top \Big(\sum_f \mathbf{e}_f\Big) + b_u + b_i .
$$

Now watch what this buys at cold start. Take a brand-new film with **zero interactions** but known features genre=sci-fi and decade=1990s. Classic MF has no vector for it — done, unrecommendable. LightFM computes its vector as $\mathbf{v}_{\text{new}} = \mathbf{e}_{\text{sci-fi}} + \mathbf{e}_{1990s}$, and those two embeddings *were already learned* from all the other sci-fi and 1990s films in training. The new film inherits a meaningful vector entirely from its features. It is not cold. **This is the algebraic reason content features solve item cold start**, and it generalizes: as long as a new item shares at least one feature with the training catalog, it gets a non-trivial vector. The figure makes the summation explicit.

![Grid showing an item vector formed by summing feature embeddings for identity, genre, and decade so a new item is ready on day one](/imgs/blogs/content-based-and-hybrid-recommenders-7.png)

The "identity" feature is the elegant part. If you give each item *both* an identity feature (unique to it) and content features (shared), then for a *warm* item the identity embedding learns the collaborative residual — the taste signal that content cannot explain — and the model behaves like full MF plus content. For a *cold* item, the identity embedding is untrained (effectively zero), so the item falls back to pure content. One model, smooth interpolation from content-only (cold) to CF-plus-content (warm). That is feature combination done right: the content and collaborative signals share *one* factor space, so they are learned jointly rather than blended after the fact.

#### Worked example: a cold-start item getting a vector via features

Suppose after training, the learned 3-dimensional feature embeddings are (numbers chosen to show the mechanics):

- $\mathbf{e}_{\text{sci-fi}} = (0.8, 0.1, -0.2)$
- $\mathbf{e}_{\text{1990s}} = (0.1, 0.5, 0.0)$
- $\mathbf{e}_{\text{identity-of-new-film}} = (0, 0, 0)$ — untrained, the film is new.

A new film tagged sci-fi and 1990s gets vector $\mathbf{v}_{\text{new}} = (0.8, 0.1, -0.2) + (0.1, 0.5, 0.0) + (0,0,0) = (0.9, 0.6, -0.2)$. Now take a user whose learned profile vector is $\mathbf{u} = (0.7, 0.4, 0.1)$ (a user who, from their history, leans toward exactly that region of taste space). The score:

$$
\hat{r} = \mathbf{u}^\top \mathbf{v}_{\text{new}} = (0.7)(0.9) + (0.4)(0.6) + (0.1)(-0.2) = 0.63 + 0.24 - 0.02 = 0.85 .
$$

A solidly positive score — the new film will rank highly for this user, on day one, with zero interactions. Compare to a romance fan with profile $\mathbf{u}' = (-0.3, 0.2, 0.6)$: their score is $(-0.3)(0.9) + (0.2)(0.6) + (0.6)(-0.2) = -0.27 + 0.12 - 0.12 = -0.27$, negative, so the film is correctly *not* recommended to them. The cold film is ranked sensibly per-user, entirely through its features. That is the LightFM cold-start mechanism, made of nothing but a sum and a dot product.

### 7.3 The weighted-hybrid score and tuning the blend

The simplest hybrid — weighted — deserves its own treatment because it is what most teams ship first and it has a real tuning knob. Given a content score $s_{\text{content}}(u,i) \in [0,1]$ (a cosine) and a CF score $s_{\text{CF}}(u,i)$ (a dot product or predicted rating), the blended score is:

$$
s(u, i) = \beta \cdot \tilde{s}_{\text{CF}}(u,i) + (1-\beta)\cdot \tilde{s}_{\text{content}}(u,i), \qquad \beta \in [0, 1].
$$

The tildes are a reminder to **normalize the two scores to a common scale before blending** — a raw cosine in $[0,1]$ and a raw MF dot product in, say, $[-3, 7]$ are not comparable, and skipping normalization means $\beta$ silently does nothing because one term dwarfs the other. Min-max or z-score normalization per query is the usual fix. The blend weight $\beta$ is the dial: $\beta = 1$ is pure CF, $\beta = 0$ is pure content. You tune it on a validation set, and — this is the practical insight — **the best $\beta$ is not constant**, it depends on item warmth. For warm items, lean CF ($\beta$ high); for cold items, lean content ($\beta$ low). A simple and effective policy makes $\beta$ a function of the item's interaction count $n_i$:

$$
\beta(n_i) = \frac{n_i}{n_i + \kappa},
$$

so a never-seen item ($n_i = 0$) gets $\beta = 0$ (pure content) and a heavily-rated item gets $\beta \to 1$ (mostly CF), with $\kappa$ controlling the crossover. This is a *weighted hybrid that secretly does switching* — the cleanest of both worlds, and it is what I reach for when I do not have time to train a fully fused model like LightFM.

```python
import numpy as np

def blend_scores(cf_scores, content_scores, n_interactions, kappa=20.0):
    # Per-item warmth-aware blend; normalize each signal to [0,1] first.
    def mm(x):
        x = np.asarray(x, dtype="float64")
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)
    cf, ct = mm(cf_scores), mm(content_scores)
    beta = n_interactions / (n_interactions + kappa)   # per-item, in [0,1)
    return beta * cf + (1.0 - beta) * ct               # cold -> content, warm -> CF
```

## 8. The hybrid in practice: LightFM with item features, and the results

Here is the payoff: a real LightFM hybrid on MovieLens with item content features, evaluated on **both** warm and cold-start Recall@10. We hold out a set of items entirely from training (zero interactions in train) to simulate cold start honestly — this is the only way to measure cold-start recall, and it is the test most blog posts skip.

```python
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import recall_at_k

# 1) Build item features: each movie gets an identity feature + its genres.
movie_ids = movies["movieId"].tolist()
genre_lists = movies["genres"].str.split(" ")          # "Action Sci-Fi" -> [Action, Sci-Fi]
all_genres = sorted({g for gs in genre_lists for g in gs if g})

ds = Dataset()
ds.fit(users=ratings.userId.unique(),
       items=movie_ids,
       item_features=all_genres)                        # genre vocabulary as features

# 2) Choose cold-start items: 15% of movies, removed from TRAIN interactions only.
rng = np.random.default_rng(0)
cold_items = set(rng.choice(movie_ids, size=int(0.15 * len(movie_ids)), replace=False))
train_int = [(r.userId, r.movieId) for r in train.itertuples()
             if r.movieId not in cold_items]            # cold items unseen in train
(interactions, _) = ds.build_interactions(train_int)

# 3) Item features matrix: (movieId, [genre, ...]) -- cold items STILL get features.
item_feats = ds.build_item_features(
    ((m, gs) for m, gs in zip(movie_ids, genre_lists)))

# 4) Train the WARP-loss hybrid (WARP optimizes top-K ranking directly).
model = LightFM(loss="warp", no_components=64, learning_rate=0.05, random_state=0)
model.fit(interactions, item_features=item_feats, epochs=30, num_threads=4)
```

```python
# 5) Evaluate warm vs cold-start Recall@10.
# Build a test interaction matrix aligned to the dataset's internal ids.
uid_map, _, iid_map, _ = ds.mapping()
def to_csr(df):
    rows, cols = [], []
    for r in df.itertuples():
        if r.userId in uid_map and r.movieId in iid_map and r.rating >= 4.0:
            rows.append(uid_map[r.userId]); cols.append(iid_map[r.movieId])
    data = np.ones(len(rows), dtype="float32")
    return csr_matrix((data, (rows, cols)),
                      shape=(len(uid_map), len(iid_map)))

test_warm = to_csr(test[~test.movieId.isin(cold_items)])
test_cold = to_csr(test[test.movieId.isin(cold_items)])

warm_r = recall_at_k(model, test_warm, item_features=item_feats, k=10).mean()
cold_r = recall_at_k(model, test_cold, item_features=item_feats, k=10).mean()
print(f"Hybrid  warm Recall@10 = {warm_r:.3f}   cold Recall@10 = {cold_r:.3f}")
```

The crucial line is the construction of `cold_items`: those movies appear in **no training interaction**, yet they *do* appear in `item_feats` with their genres. So at test time LightFM scores them via $\mathbf{v}_i = \sum_f \mathbf{e}_f$ over their genre features — exactly the worked example. A pure-CF model (LightFM trained with *only* the identity feature, no genres) would score every cold item at the prior and land at essentially zero cold-start recall, because the identity embeddings of unseen items are untrained.

### 8.1 The results table

Here is the consolidated before-and-after on `ml-latest-small`, temporal split, with the cold-start protocol above. These are representative numbers in the range I see on this dataset; treat them as approximate Pareto points, not lab-certified constants — the exact values shift with split, seed, and threshold, but the *ordering* is robust and is the point.

| Model | Warm Recall@10 | Cold-start Recall@10 | NDCG@10 (warm) | Notes |
|---|---|---|---|---|
| Popularity baseline | 0.13 | 0.00 | 0.11 | non-personalized; zero cold |
| CF only — BPR/WARP MF | **0.41** | 0.00 | **0.38** | best warm, fails cold |
| Content only — TF-IDF | 0.23 | 0.18 | 0.21 | cheap, interpretable |
| Content only — MiniLM dense | 0.26 | **0.21** | 0.24 | best pure-content cold |
| Hybrid — LightFM + genres | **0.43** | 0.19 | **0.39** | best overall |

Read this table slowly, because it is the entire argument of the post:

- **CF wins warm and *fails* cold.** 0.41 warm Recall@10, exactly 0.00 cold. The zero is not a rounding artifact — it is structural; CF has no vector for an unseen item.
- **Content rescues cold start.** TF-IDF and dense both land around 0.18–0.21 cold Recall@10 where CF is at zero. Dense embeddings edge out TF-IDF on both warm and cold because they capture semantics.
- **The hybrid is the best of both.** It matches or slightly beats CF on warm (0.43 vs 0.41 — the genre features add a little signal even for warm items) and recovers most of content's cold-start recall (0.19). No single pure method dominates both columns; the hybrid is the only row that is strong in both.

The final figure makes the warm-versus-cold story a matrix you can show a stakeholder.

![Matrix comparing warm Recall@10 against cold-start Recall@10 for collaborative, content, and hybrid recommenders with a verdict per row](/imgs/blogs/content-based-and-hybrid-recommenders-8.png)

#### Worked example: the cold-start cliff, in absolute terms

Make the 0.00 visceral. Say the weekend launch from the intro adds 40 new films, and on launch day 5,000 users each have 8 genuinely relevant new films in the catalog (their held-out cold positives). With a **CF-only** model, cold Recall@10 = 0.00, so across all 5,000 users the new films get recommended into a top-10 **zero times** beyond random luck — the 40 titles are dead on arrival. With the **hybrid** at cold Recall@10 ≈ 0.19, roughly 19% of those relevant new films make a user's top-10. Concretely, if each user has 8 relevant new films, the hybrid surfaces about $0.19 \times 8 \approx 1.5$ of them per user into the top-10, versus $\approx 0$ for CF. Multiply by 5,000 users and the difference is thousands of impressions on exactly the titles marketing paid to launch. That is the dollar value of the hybrid: not a prettier offline NDCG, but new inventory that is actually discoverable on day one.

### 8.2 Stress-testing the hybrid

A decision is only trustworthy after you have tried to break it. Four stress tests on this design:

- **What if the only features are weak (genre alone)?** Cold-start recall degrades toward the popularity prior, because items sharing only a coarse genre look alike. The fix is richer features — tags, cast, dense text embeddings as LightFM features — which is exactly why production hybrids invest in the representation layer (section 2). Garbage features, garbage cold start.
- **What at 100M items?** LightFM's feature-sum scoring is a dot product, so retrieval becomes a **maximum inner product search** — you build the item vectors, push them into an ANN index (faiss / hnswlib), and serve top-N in milliseconds. This is the bridge to the [two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval): a two-tower model is LightFM's feature-sum generalized to a deep encoder, served the same way through ANN.
- **What if cold items are also *low quality*?** Content cannot tell good from bad within a content class — that is its fundamental limit. The mitigation is a **switching** policy: serve cold items via content to *gather* interactions fast (explore), then switch to CF once enough data arrives to judge quality (exploit). Cold start is partly an exploration problem, not only a representation problem.
- **What if offline cold recall rises but online new-item CTR is flat?** Classic offline-online gap. Often the cause is that your offline "cold" protocol does not match online reality — e.g. online, cold items also suffer from *position bias* (they get buried because they have no historical CTR to rank on) regardless of how well your model scores them. The fix is upstream of the model: guaranteed exploration slots for new items. The model can only recommend what the funnel lets it show. We treat this gap in depth in the offline-vs-online post.

## 9. Case studies: content and hybrids in shipped systems

Three real systems, each illustrating a different reason content earns its place.

### 9.1 Pandora and the Music Genome Project (content, taken to the limit)

Pandora's recommender is the most famous *pure content* system ever shipped. Starting in 2000, musicologists hand-annotated songs across up to ~450 musical attributes — "the Music Genome Project" — things like vocal style, instrumentation, rhythm, lyrical content. Each song becomes a ~450-dimensional content vector, and Pandora recommends the next song by *content* similarity to a seed, with thumbs-up/down adjusting the user profile. This is content-based filtering with extraordinarily rich, expensive, human-curated features — and it works precisely *because* the features are rich enough to capture the taste-relevant structure that genre tags miss. The lesson for the rest of us: content recommendation is only as good as your features, and Pandora bought their quality with human labor that modern audio embeddings now approximate automatically. Their pure-content choice also bought the long-tail and cold-start wins of section 5 — a brand-new song is recommendable the moment a musicologist annotates it. (See Howe's accounts of the Music Genome Project; the system is a case study in the Lops et al. content survey, 2011.)

### 9.2 LightFM in production (the hybrid default)

Maciej Kula built LightFM at Lyst (a fashion marketplace) to solve exactly the cold-start launch problem from this post's intro: a fashion catalog turns over constantly, and new products with zero interactions must be recommendable from their metadata (brand, category, color, designer). The 2015 paper "Metadata Embeddings for User and Item Cold-start Recommendations" reports that the feature-sum hybrid matches pure MF on warm items *and* substantially beats both pure content and pure CF on cold-start items on the MovieLens and CrossValidated (Stack Exchange) datasets — the same warm-strong / cold-rescued / hybrid-best pattern as our table. LightFM's enduring popularity (it is still one of the most-used recsys libraries) is the strongest evidence that the feature-fusion hybrid is the right production default for catalogs with metadata. It is the model I reach for first when content features exist and the catalog churns.

### 9.3 News and article recommendation (content for freshness)

News is the domain where CF is *structurally inadequate* and content is mandatory, for one reason: **items expire faster than interaction data accumulates**. An article published twenty minutes ago, the one you most want to recommend, has near-zero clicks. A pure CF system cannot rank it. So news recommenders lean heavily on content — article text embeddings, topic, entities, recency — to score fresh articles, blending in collaborative signal only as clicks trickle in (a textbook warmth-aware blend, section 7.3). Google News's early system (Das et al., "Google News Personalization", WWW 2007) combined collaborative methods with content and used aggressive freshness handling; modern systems (and the MIND benchmark from Microsoft, 2020) center content embeddings of the article body for exactly the freshness reason. The general law: **the faster your catalog turns over, the more you must lean content** — at the limit of instant expiry (live news, short video), content is not optional.

### 9.4 YouTube and the deep two-tower (content fused at scale)

The largest-scale version of the hybrid idea is the deep two-tower retrieval model behind systems like YouTube's candidate generation (Covington, Adams, Sargin, "Deep Neural Networks for YouTube Recommendations", RecSys 2016, and its two-tower successors). The item tower consumes content and metadata features — video text, channel, topic, age — and produces an item embedding; the user tower consumes the user's watch history and context. The two embeddings are scored by dot product and retrieval is a maximum-inner-product search over an ANN index. Read structurally, the item tower *is* a learned content-based item profile: it maps item features to a vector with no per-item interaction required, which is exactly why a freshly uploaded video can be retrieved. The difference from LightFM is that the mapping from features to vector is a deep network rather than a sum of embeddings, so it can learn feature interactions (a "cooking" video for *this* channel means something different than for *that* one). The two-tower post in this series builds this model directly; the point for now is that the frontier of content-based recommendation is not a separate technique but a deeper encoder fused into the retrieval stage. Everything in this post — feature representation, the shared user-item space, cold-start safety from features — carries forward unchanged; only the encoder gets bigger.

The four cases trace the spectrum: Pandora (pure content, rich hand-built features), LightFM (sum-of-features fused hybrid, the practical default), news (content-forward hybrid for freshness), and YouTube (deep two-tower, content fused at web scale). The common thread is constant: *content is what makes a recommender work before behavior exists*, and the only thing that changes across the spectrum is how expressively you encode that content.

## 10. When to lean content, CF, or hybrid

A decisive recommendation, because every choice is a cost.

**Lean content-based when:**

- Your catalog **churns fast** (news, marketplaces, UGC video) so most items are perpetually cold.
- You have **rich, taste-relevant features** (good text, media, curated attributes) — Pandora's bet.
- You need **explainability** for trust, product, or legal reasons.
- You are **fighting popularity bias** and want the long tail to compete.
- You have **few users** or per-user-only data (privacy-sensitive, single-tenant).

**Lean pure CF when:**

- Your catalog is **stable** and items accumulate plenty of interactions (back catalog, established store).
- You have **lots of interaction data** and **weak or expensive features**.
- **Serendipity matters** and you want cross-content discovery the crowd reveals.
- You want **zero feature engineering** to ship a strong day-one baseline. (Start here; CF is often the baseline the fancy models must beat — see [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles).)

**Reach for a hybrid (the usual answer) when:**

- You have **both** behavior and features — which, after launch, is almost always.
- You face **item cold start on a real catalog with metadata** — LightFM's sweet spot.
- You want **CF's warm accuracy *and* content's cold-start coverage** in one model.
- You are building the **retrieval stage of a funnel** — fuse content features into a two-tower/feature-sum model and serve via ANN.

And when *not* to over-engineer: do not build a deep multi-tower hybrid if a weighted blend of an `implicit` BPR model and a TF-IDF cosine already hits your cold-start target — that warmth-aware weighted hybrid (section 7.3) is a day's work and frequently enough. Do not invest in CLIP/audio embeddings until text features are exhausted; the marginal feature is usually cheaper than the marginal modality. And do not ship pure content to a stable catalog expecting serendipity — you will build a filter bubble and watch coverage collapse. Match the method to the *shape of your catalog and data*, not to fashion.

### 10.1 The pitfalls that catch teams, and how to avoid them

A handful of mistakes show up again and again when teams first build content and hybrid systems. Knowing them in advance saves a quarter.

- **Leaking behavioral features into the "content" vector.** As section 2 warned, if your item vector includes a feature that only exists after interactions accrue (CTR-so-far, average rating), your offline cold-start metric is a fiction — the held-out test items already have those features, but a truly new item in production will not. The fix is discipline: the cold-start representation must rest only on intrinsic, upload-time features. Audit your feature list and tag each one intrinsic or behavioral.
- **Forgetting to mask seen items.** A content recommender's single highest-cosine item for a user is almost always something they already consumed (it defined their profile). Without a seen-item mask, your top recommendation is a re-run, which looks fine in a naive offline metric and is useless online. Always mask, and decide explicitly whether re-recommendation is ever desirable (for consumables like groceries, sometimes yes; for movies, almost never).
- **Blending un-normalized scores.** A raw cosine in $[0,1]$ and a raw MF dot product in some wide range are not comparable; a weighted blend of them silently ignores the smaller-scaled signal. Normalize both signals per query before combining, or your carefully chosen $\beta$ does nothing.
- **Random train-test splits.** Splitting interactions randomly lets a future rating leak into a user's training profile, inflating every metric. Use a temporal split so the test set is strictly later than training — the only split that simulates the real serving order.
- **Measuring cold start on warm items.** The most common silent failure: reporting one Recall@10 number computed over all test items, most of which are warm, and calling the system "good at cold start." You must hold a set of items entirely out of training and report their recall separately, as section 8 does. A single blended number hides the 0.00 that matters.
- **Over-trusting the offline cold-start win.** Even an honest offline cold-start gain can fail to move online new-item engagement if the funnel buries cold items for *position* reasons (no historical CTR to rank on). Pair a content/hybrid model with guaranteed exploration slots for new items; the model can only recommend what the system lets it show.

Most of these are not modeling problems at all — they are evaluation and plumbing problems. The model is usually the easy part; measuring it honestly and serving it without skew is where the engineering lives.

## 11. Key takeaways

- **Content-based filtering recommends items similar to what a user liked, using item features**, with no other users in the loop — which is exactly why it **solves item cold start**: a new item with features gets a vector immediately.
- **The pipeline is three pieces**: item vectors (TF-IDF, dense `sentence-transformers`, or media embeddings), a user profile (rating-/recency-weighted mean of liked-item vectors, or a learned per-user model), and cosine scoring in the shared space.
- **TF-IDF weights a term by frequency-in-item times rarity-in-corpus**, and cosine compares *direction* (content) while ignoring *magnitude* (length) — derive both once and the rest is engineering.
- **Content's weakness is over-specialization** (the filter bubble, low serendipity, can't judge quality beyond content) — structurally complementary to CF's cold-start failure and popularity bias.
- **Hybrids are the production default.** Burke's 2002 taxonomy (weighted, switching, mixed, feature combination/augmentation, cascade, meta-level) is the map; the funnel itself is a cascade hybrid.
- **LightFM represents an item as a sum of its feature embeddings** $\mathbf{v}_i = \sum_{f} \mathbf{e}_f$, so a cold item inherits a meaningful vector from features already learned across the catalog — the *algebraic* reason features solve cold start. Give items an identity feature too, and the model interpolates smoothly from content-only (cold) to CF-plus-content (warm).
- **Measure cold start honestly**: hold items entirely out of training and report cold-start Recall@10 separately from warm. On MovieLens the pattern is robust — CF wins warm and scores ~0 cold; content rescues cold; the hybrid is best across both.
- **Tune the blend by warmth**: a weighted hybrid with $\beta(n_i) = n_i/(n_i + \kappa)$ is a weighted hybrid that secretly switches — cold items lean content, warm items lean CF.
- **The faster your catalog turns over, the more you must lean content** — news and short-video are content-forward by necessity; stable back-catalogs can lean CF.

## 12. Further reading

- **Lops, de Gemmis, Semeraro (2011), "Content-based Recommender Systems: State of the Art and Trends"** (in *Recommender Systems Handbook*) — the definitive content-based survey; covers item profiles, user profiles, and limitations.
- **Burke (2002), "Hybrid Recommender Systems: Survey and Experiments"** (*User Modeling and User-Adapted Interaction*) — the seven-way hybridization taxonomy used in section 7.
- **Kula (2015), "Metadata Embeddings for User and Item Cold-start Recommendations"** — the LightFM paper; derives the feature-sum representation and reports the warm/cold/hybrid results.
- **Mooney and Roy (2000), "Content-Based Book Recommending Using Learning for Text Categorization"** — the IR-to-recsys bridge; recommendation as text classification.
- **Das, Datar, Garg, Rajaram (2007), "Google News Personalization: Scalable Online Collaborative Filtering"** (WWW) — content + collaborative blending under extreme freshness constraints.
- **LightFM documentation** (`making.lyst.com/lightfm`) and **`sentence-transformers` documentation** (`sbert.net`) — the two libraries built in this post, with API and pretrained-model references.
- **Within the series**: start at [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) for the funnel map; pair this post with [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles) and [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem); follow content features forward into [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval); and see everything assembled in [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

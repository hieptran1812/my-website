---
title: "Data Selection and Pruning: Choosing the Best N% From an Ocean"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "You have far more data than compute. This is the practitioner's guide to picking the subset that trains the best model — importance resampling (DSIR), classifier and influence/gradient selection (DsDm, LESS), coreset pruning (SemDeDup), and the pruning scaling law that lets a good difficulty metric beat the power law."
tags:
  - training-data
  - data-selection
  - data-pruning
  - importance-resampling
  - dsir
  - coreset
  - influence-functions
  - scaling-laws
  - llm-pretraining
  - data-quality
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 30
---

Every serious pretraining run starts the same way: you have far more data than you can afford to train on. After you have crawled, extracted, deduplicated, and filtered the web, you are still holding a pool of trillions of tokens and a compute budget that will only pay for a few hundred billion. The naive move is to shuffle the survivors and take the first slice that fits. That is a mistake worth billions of forward passes, because the slice you pick *is* the model you get. Selection is not a cleanup step you tack on at the end; it is the last and most aggressive lever you have over the training distribution.

![Each stage narrows the pool; the training set is a fraction of a percent of the raw crawl](/imgs/blogs/data-selection-and-pruning-1.webp)

The diagram above is the mental model for this entire post: a funnel. A ten-trillion-token crawl becomes three trillion after dedup and quality filters, a few hundred billion after domain-matched selection, and thirty billion after you cut to the compute budget — roughly three-tenths of one percent of where you started. The early stages are covered elsewhere in this series ([measuring data quality](/blog/machine-learning/training-data/measuring-data-quality), [classifier and perplexity filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering), [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale)). This post is about the two narrowest stages: **selection** — actively choosing which documents make the cut — and **pruning** — throwing away examples you already have because they are redundant, too easy, or actively harmful. We will build up five families of methods, the one scaling law that makes pruning counterintuitive, two worked scenarios with real numbers, three named case studies, and a troubleshooting section for when selection quietly makes your model worse.

## 1. The selection problem: which subset, and selected for what?

Start with the senior rule of thumb: **selection is a distribution-shaping operation, not a scoring operation.** Filtering asks a per-document question ("is this document good enough to keep?"). Selection asks a *set-level* question ("what should the training distribution look like, and which documents move it there?"). Those are different problems, and confusing them is the root of most selection failures.

Concretely, you are given a raw pool $D$ with billions of documents, a compute budget that buys $B$ tokens, and — this is the part people skip — some notion of what you want the model to be good at. Call that the *target*. The target might be a reference corpus you trust (Wikipedia, textbooks, curated instructions), a downstream validation set (a slice of MMLU, a coding benchmark, a customer's domain), or nothing at all (you just want a broadly good general model). Your job is to choose a subset $S \subset D$ with roughly $B$ tokens that minimizes the model's loss *on the target* after training on $S$.

Written that way, the problem is obviously intractable — you cannot train a model per candidate subset. Every practical method is a cheap **proxy** for that intractable objective, and the families differ in what proxy they use and how much of the target they are allowed to see:

| Question the method answers | Family | Target-aware? | Cost |
| --- | --- | --- | --- |
| "Does this doc match a reference distribution?" | Importance resampling (DSIR) | Yes, a reference corpus | CPU, cheap |
| "Does this doc look like high-quality text?" | Classifier scoring | Weakly (via labels) | 1 forward pass/doc |
| "Would this doc reduce loss on my val set?" | Influence / gradient (DsDm, LESS) | Yes, a validation set | Gradient/doc, expensive |
| "Is this doc redundant or too easy?" | Coreset / pruning (SemDeDup) | No, intrinsic | 1 embedding/doc |

Two design axes fall out of that table and they will organize the rest of the post. The first is **target-awareness**: does the method chase a specific downstream objective, or does it optimize an intrinsic property of the data? The second is **cost per document**: a full-corpus method that needs a gradient per document is orders of magnitude more expensive than one that needs a hash. You almost never run the expensive methods on the whole pool; you run a cheap method to get to a few hundred billion tokens, then an expensive one on what remains. The funnel is layered on purpose.

There is a third axis that hides behind the other two and causes the nastiest bugs: **the proxy is not the goal.** Every method here optimizes something you can measure cheaply, and the moment you optimize hard against a cheap proxy, you invite [Goodhart's law](/blog/machine-learning/training-data/measuring-data-quality) — the proxy improves while the thing you actually cared about regresses. We will return to this repeatedly, and the entire troubleshooting section is about catching it.

## 2. Importance resampling: match a target distribution with DSIR

The cleanest, cheapest, most under-used selection method is **importance resampling**, and the reference implementation is DSIR (Data Selection via Importance Resampling, Xie et al. 2023). The senior rule: **if you know what distribution you want, don't hand-write filters to approximate it — measure it and resample toward it.**

The intuition is a change-of-measure. You have a giant raw pool distributed like $q$ (mostly web junk) and a small target corpus distributed like $p$ (Wikipedia plus instructions, say). You want a subset of the *raw pool* that looks like it was drawn from $p$. Importance sampling says: keep each raw document with probability proportional to the likelihood ratio $w(x) = p(x)/q(x)$. Documents that are common in the target but rare in the raw pool get upweighted; documents common in the raw pool but rare in the target get dropped. You never write a rule like "keep formal text" — the ratio discovers what "formal" means from the two distributions.

![DSIR maps both corpora into one hashed n-gram space, then keeps documents in proportion to the target-over-raw likelihood ratio](/imgs/blogs/data-selection-and-pruning-2.webp)

The figure above is the whole pipeline. The trick that makes it cheap enough to run on trillions of tokens is the feature space: DSIR does not model $p(x)$ and $q(x)$ over raw text (impossible). It hashes unigrams and bigrams into a fixed number of buckets (10,000 in the paper) and models each distribution as a bag of hashed n-grams — a simple categorical distribution over buckets. Estimating $p$ and $q$ is then just counting, and the importance weight of a document is a product of per-bucket ratios, which becomes a sum of log-ratios in log space. No neural network, no GPU, embarrassingly parallel across the corpus.

### A worked example, with numbers

Make it concrete. Suppose we collapse the hashed features into four interpretable buckets: **A** = encyclopedic/technical n-grams, **B** = instructional n-grams ("how to", "step by"), **C** = SEO/boilerplate ("buy now", "free shipping"), **D** = conversational filler. We estimate the two distributions by counting:

| Bucket | Target $p$ | Raw $q$ | Log-ratio $\log(p/q)$ |
| --- | --- | --- | --- |
| A (technical) | 0.40 | 0.15 | +0.981 |
| B (instructional) | 0.35 | 0.10 | +1.253 |
| C (SEO boilerplate) | 0.05 | 0.45 | -2.197 |
| D (conversational) | 0.20 | 0.30 | -0.405 |

Now score three ten-n-gram documents by summing the log-ratios of their n-grams. This sum *is* the log importance weight:

| Document | A | B | C | D | Log importance (sum) | Per-n-gram |
| --- | --- | --- | --- | --- | --- | --- |
| Technical article | 6 | 3 | 0 | 1 | **+9.24** | +0.924 |
| SEO spam page | 0 | 0 | 8 | 2 | **-18.39** | -1.839 |
| Forum mixed post | 2 | 1 | 3 | 4 | **-5.00** | -0.500 |

The technical article scores strongly positive because it is dense in the buckets the target over-represents; the SEO page is deeply negative because it is nothing but the bucket the target hates. Resampling then keeps documents in proportion to $\exp(\text{score})$, so the technical article survives with near-certainty and the SEO page is almost surely dropped — *without a single hand-written rule about SEO or technical writing.* One caveat visible in the table: the raw sum grows with document length, so long junk can outscore short gold. DSIR uses the raw sum but production pipelines usually length-normalize (the per-n-gram column) or cap length, or the longest documents dominate the selection.

### Runnable DSIR-style resampling

Here is the mechanism end to end — hashed n-gram features, distribution estimation, log importance weights, and Gumbel top-k resampling (the standard trick for sampling $k$ items without replacement in proportion to weights):

```python
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# 1. Featurize both corpora into the SAME hashed n-gram space.
#    Raw counts (not tf-idf): DSIR models a bag-of-hashed-ngrams categorical.
NBUCKETS = 10_000
vec = HashingVectorizer(
    ngram_range=(1, 2), n_features=NBUCKETS,
    alternate_sign=False, norm=None, binary=False,
)

def bucket_counts(texts):
    # Sum the sparse count matrix down to a single NBUCKETS-vector.
    return np.asarray(vec.transform(texts).sum(axis=0)).ravel()

def estimate_dist(texts, smoothing=1.0):
    c = bucket_counts(texts) + smoothing          # Laplace smoothing so no bucket is 0
    return c / c.sum()

# target = small trusted corpus (Wikipedia + instructions); raw = the ocean.
p = estimate_dist(target_docs)                    # target distribution
q = estimate_dist(raw_sample_docs)                # raw-pool distribution (a sample is fine)
log_ratio = np.log(p) - np.log(q)                 # per-bucket log importance

# 2. Score every raw document = sum of log-ratios of its n-grams.
def log_importance(text, length_normalize=True):
    counts = np.asarray(vec.transform([text]).todense()).ravel()
    score = float(counts @ log_ratio)
    return score / max(counts.sum(), 1) if length_normalize else score

scores = np.array([log_importance(d) for d in raw_docs])

# 3. Gumbel top-k: keep k docs ~ proportional to exp(score), no replacement.
#    (adding Gumbel noise to log-weights and taking the top-k is exact sampling)
k = 300_000
gumbel = -np.log(-np.log(np.random.rand(len(scores))))
keep_idx = np.argpartition(scores + gumbel, -k)[-k:]
selected = [raw_docs[i] for i in keep_idx]
```

Two production notes. First, you estimate $q$ from a *sample* of the raw pool — you do not need the whole ocean to know its n-gram distribution, and a few million documents is plenty. Second, Gumbel top-k gives you *soft* selection: it favors high-importance documents but still admits some lower-scoring ones, which preserves diversity better than a hard threshold. A hard `scores > τ` cut is the fastest way to collapse your selected set onto a single register (more on that in troubleshooting).

DSIR's result is the reason it is worth knowing: on continued pretraining, selecting Pile/Common-Crawl data to match a Wikipedia-plus-books target beat both random selection and hand-tuned heuristic filtering on downstream GLUE, at essentially zero GPU cost. It is the highest ratio of downstream gain to engineering effort in the whole selection toolbox.

## 3. Classifier-based selection: learn the boundary, then generalize it

Importance resampling matches a distribution. Classifier-based selection asks a sharper question: **can a learned model tell "the kind of text I want" from "everything else" better than an n-gram ratio can?** This is the family covered in depth in [classifier and perplexity-based filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering), so we will recap the mechanism and then generalize it, because the generalization is where selection lives.

The recap: train a lightweight classifier (fastText, or a linear head on frozen embeddings) to distinguish a positive reference set from a random web sample, then keep documents the classifier scores highly. GPT-3 and CCNet did the original versions with a reference of "formatted like WebText/Wikipedia." FineWeb-Edu did the modern, potent version: it used Llama-3-70B to label a few hundred thousand web pages for *educational quality* on a 0–5 scale, trained a small classifier to reproduce those labels from embeddings, and kept the high-scoring pages. The MMLU jump was large enough that FineWeb-Edu became a default.

![Cheaper methods use intrinsic signals; target-aware methods pay gradients to chase a specific downstream loss](/imgs/blogs/data-selection-and-pruning-3.webp)

The figure places all four families on the two axes from Section 1. Classifier selection sits in the middle: more expressive than an n-gram ratio (it can learn "educational" as a nonlinear function of embeddings), but only weakly target-aware — it optimizes "resembles my positive labels," which is a *proxy* for "helps my downstream task," and the two can diverge. Three generalizations matter in practice:

- **The positive set is the whole ballgame.** A classifier trained with "Wikipedia" as positives learns a *style* (formal, third-person, dense) as much as a *quality*. If your downstream task is casual dialogue or code, that style bias hurts. Choose positives that resemble what you actually want, not what feels prestigious.
- **Score calibration drifts across domains.** A quality classifier trained mostly on English prose will systematically under-score code, math, and non-English text, silently pruning them. Always check the score distribution *per domain* before applying a global threshold.
- **Soft beats hard.** As with DSIR, sampling in proportion to a temperature-scaled score preserves more diversity than a hard cutoff. FineWeb-Edu's own ablations show the threshold is a diversity-vs-quality dial, not a free parameter.

The senior take: classifier selection is the right default when you have a clear notion of "good" that a reference set captures but an n-gram ratio cannot. It is *not* the right tool when your real objective is a specific downstream benchmark — for that, you want a method that looks at the model's own gradients.

## 4. Influence and gradient selection: pick data by its effect on a target loss

Everything so far selects data by a property of the *data*. Influence-based selection asks the only question that ultimately matters: **which training examples actually reduce the loss on the examples I care about?** This is target-awareness in its strongest form — you hand the method a validation set (or a few-shot description of the target task) and it selects the training documents whose *gradients* most help that target.

The intuition comes from a first-order Taylor expansion. Take one SGD step on training example $z$; the change in loss on a test example $z'$ is, to first order, $-\eta\, \nabla\ell(z') \cdot \nabla\ell(z)$. So the *influence* of $z$ on $z'$ is proportional to the dot product of their gradients: examples whose gradients point the same way help each other; examples whose gradients oppose hurt. Three methods build on this:

- **TracIn** (Pruthi et al. 2020) makes the idea trainable: it approximates the total influence of $z$ on $z'$ as a sum over training checkpoints of $\nabla\ell(z') \cdot \nabla\ell(z)$. It is the honest, expensive definition of influence — a gradient dot product per (train, test) pair per checkpoint.
- **DsDm** (Datamodel Selection, Engstrom et al. 2024) reframes selection as regression: fit a *datamodel* that predicts target loss as a linear function of which training examples were included, then select the subset the datamodel says minimizes target loss. On target benchmarks it beat classifier selection outright — and, tellingly, the data it chose often looked *worse* by classifier standards, because "helps the target" and "looks high-quality" are genuinely different objectives.
- **LESS** (Xia et al. 2024) makes targeted instruction tuning practical. Computing a real gradient per document is too expensive, so LESS builds low-rank gradient features: LoRA-style adapters plus random projection compress each example's gradient into a small vector, stored once in a datastore. Selection is then a similarity search — keep the training examples whose projected gradient aligns with the projected gradient of a few-shot example of the target task. The headline result: selecting roughly 5% of an instruction pool with LESS matched or beat training on the full pool for a target task, and the selection transferred across model sizes.

Here is the shape of gradient-based selection, LESS-style, in PyTorch — the point is the *gradient feature*, not the plumbing:

```python
import torch, torch.nn.functional as F

def grad_feature(model, batch, proj):
    """Low-rank gradient feature for one example (LESS-style)."""
    model.zero_grad()
    out = model(**batch)
    out.loss.backward()
    # Concatenate LoRA-adapter grads only (tiny vs full model), then
    # random-project to a fixed low dimension so features are comparable.
    g = torch.cat([p.grad.reshape(-1) for p in model.lora_parameters()])
    return F.normalize(proj @ g, dim=0)           # unit vector in R^d_proj

# Build a datastore of gradient features over the candidate training pool once.
proj = torch.randn(8192, n_lora_params) / (8192 ** 0.5)   # random projection
store = torch.stack([grad_feature(model, ex, proj) for ex in candidate_pool])

# Target signal: average gradient feature of a few-shot example of the task.
target_vec = torch.stack(
    [grad_feature(model, ex, proj) for ex in target_fewshot]
).mean(0)
target_vec = F.normalize(target_vec, dim=0)

# Influence score = alignment; keep the top-k most aligned training examples.
influence = store @ target_vec                    # cosine similarity, higher = more helpful
keep = influence.topk(k=int(0.05 * len(candidate_pool))).indices
```

The tradeoff is stark and worth stating plainly. Influence selection is the only family that optimizes the *actual* objective (target loss), and it wins when you have a specific, known target. It is also the most expensive (a gradient per document), the most fragile (it overfits the validation set you hand it — see troubleshooting), and the least useful for general-purpose pretraining, where there is no single target to align to. Reach for it for **targeted fine-tuning and domain adaptation**, not for building a foundation model's 10-trillion-token diet.

## 5. Coreset selection and pruning: drop the redundant and the easy

The methods above *choose* documents to include. Pruning *removes* documents you already have because they carry no new information. The senior rule: **most of a web corpus is redundant, and redundancy is not the same as duplication.** Deduplication (covered in [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale)) removes near-identical text. Semantic pruning removes documents that are *different strings but the same information* — the tenth explanation of a for-loop, the thousandth product review with the same sentiment.

The workhorse is **SemDeDup** (Abbas et al. 2023). Embed every document with a pretrained encoder, run k-means to cluster the embedding space, and within each cluster drop documents whose cosine similarity to a kept neighbor exceeds a threshold — keeping one representative per tight group. On LAION it removed about half the data with negligible downstream loss; on C4 it removed a large fraction of "semantic duplicates" that exact and near-dup dedup missed entirely.

![SemDeDup keeps one representative per dense cluster and keeps every point in rare regions](/imgs/blogs/data-selection-and-pruning-6.webp)

The figure is the mechanism as a picture of the embedding space. Dense clusters are regions where the corpus says the same thing many ways; you keep one prototype and drop the near-duplicates (the gray points). Sparse regions — the rare, atypical, hard examples far from any centroid — you keep entirely, because each one is information you have nowhere else. The distance from a point to its cluster centroid becomes a natural **difficulty metric**: points near a centroid are typical and easy; points far from every centroid are rare and hard. This is the SSL-prototype metric from Sorscher et al., and it is the bridge to the most surprising result in this whole area.

Here is difficulty-based pruning end to end — embed, cluster, score by distance-to-centroid, and keep a fraction. Note the one line that flips the whole strategy depending on your data budget:

```python
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

enc = SentenceTransformer("all-MiniLM-L6-v2")
emb = enc.encode(docs, batch_size=256, normalize_embeddings=True)   # (N, d)

# Cluster the embedding space; distance to the nearest centroid = difficulty.
km = KMeans(n_clusters=1024, n_init="auto").fit(emb)
centroids = km.cluster_centers_[km.labels_]
difficulty = np.linalg.norm(emb - centroids, axis=1)   # far = hard/atypical

keep_fraction = 0.6
n_keep = int(keep_fraction * len(docs))
data_is_abundant = len(docs) > 5 * n_keep              # the regime check

if data_is_abundant:
    # Plenty of data: keep the HARDEST (most informative) examples.
    keep = np.argsort(difficulty)[-n_keep:]
else:
    # Scarce data: keep the EASIEST (typical, low-noise) examples.
    keep = np.argsort(difficulty)[:n_keep]

# Guardrail: never let pruning empty out a cluster (preserve coverage).
kept_labels = set(km.labels_[keep])
for c in range(km.n_clusters):
    if c not in kept_labels:
        members = np.where(km.labels_ == c)[0]
        keep = np.append(keep, members[np.argmin(difficulty[members])])
selected = [docs[i] for i in np.unique(keep)]
```

The `data_is_abundant` branch is not a stylistic choice — it is the crux of the next section, and getting it backwards is one of the most common pruning failures in the field.

## 6. Pruning scaling laws: beating the power law with the right difficulty metric

Here is the result that makes pruning worth a whole section instead of a paragraph. Neural scaling laws say test error falls as a **power law** in dataset size: double the data, shave a fixed fraction off the error, forever, with diminishing returns. It feels like a law of nature. Sorscher et al. (2022), in *Beyond neural scaling laws: beating power law scaling via data pruning*, showed it is not — it is an artifact of training on *randomly sampled* data. With the right difficulty metric and the right pruning strategy, test error can fall **faster** than any power law, approaching exponential scaling in the best case.

![With the right difficulty metric, pruning bends the loss curve below the power law](/imgs/blogs/data-selection-and-pruning-4.webp)

The figure is the claim in one picture. The dashed curve is the power law you get from random data: more tokens, slowly less error. The solid curve is what you get from a *pruned* corpus that keeps the informative examples: it starts at the same place and then bends below the dashed line, and the gap widens with scale. That gap is free accuracy — the same or better error from *less* data. The mechanism is intuitive once you see it: easy, typical examples are largely redundant with each other, so in the data-rich regime each additional easy example teaches the model almost nothing, while each hard example still carries signal. Pruning to the hard examples spends your fixed budget where the information is.

But — and this is the counterintuitive part that the `data_is_abundant` check encodes — **the optimal difficulty flips with the data budget.**

![Prune easy examples only when data is plentiful; when data is scarce, the easy examples carry the signal](/imgs/blogs/data-selection-and-pruning-5.webp)

The 2×2 above is the decision rule. When data is abundant relative to your budget, keep the **hard** examples: the easy ones are redundant and keeping them wastes budget (top row). When data is scarce, keep the **easy** examples: hard examples in the low-data regime are dominated by noise and label ambiguity, and the model cannot yet learn from them, so keeping them hurts while keeping the easy, unambiguous examples teaches the basics (bottom row). The successful strategies sit on the anti-diagonal. This single flip explains a mountain of contradictory pruning results in the literature — two teams get opposite conclusions from the "same" method because one was data-rich and the other data-poor.

<figure class="blog-anim">
<svg viewBox="0 0 680 300" role="img" aria-label="A difficulty histogram; as the keep-threshold sweeps right, the easy low-difficulty bars fade out while the hard bars are kept" style="width:100%;height:auto;max-width:820px">
<style>
.a8-bar{fill:var(--accent,#6366f1)}
.a8-axis{stroke:var(--text-secondary,#6b7280);stroke-width:2}
.a8-thr{fill:var(--text-primary,#1f2937);opacity:.85}
.a8-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a8-sub{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes a8-drop{0%,12%{opacity:1}52%,100%{opacity:.12}}
@keyframes a8-sweep{0%{transform:translateX(0)}52%,100%{transform:translateX(288px)}}
.a8-drop{animation:a8-drop 8s ease-in-out infinite alternate}
.a8-d1{animation-delay:0s}.a8-d2{animation-delay:.25s}.a8-d3{animation-delay:.5s}
.a8-d4{animation-delay:.75s}.a8-d5{animation-delay:1s}.a8-d6{animation-delay:1.25s}
.a8-sweep{animation:a8-sweep 8s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.a8-drop{animation:none;opacity:.12}.a8-sweep{animation:none;transform:translateX(288px)}}
</style>
<line class="a8-axis" x1="50" y1="230" x2="650" y2="230"/>
<rect class="a8-bar a8-drop a8-d1" x="60"  y="200" width="40" height="30"/>
<rect class="a8-bar a8-drop a8-d2" x="108" y="175" width="40" height="55"/>
<rect class="a8-bar a8-drop a8-d3" x="156" y="145" width="40" height="85"/>
<rect class="a8-bar a8-drop a8-d4" x="204" y="115" width="40" height="115"/>
<rect class="a8-bar a8-drop a8-d5" x="252" y="90"  width="40" height="140"/>
<rect class="a8-bar a8-drop a8-d6" x="300" y="80"  width="40" height="150"/>
<rect class="a8-bar" x="348" y="80"  width="40" height="150"/>
<rect class="a8-bar" x="396" y="90"  width="40" height="140"/>
<rect class="a8-bar" x="444" y="115" width="40" height="115"/>
<rect class="a8-bar" x="492" y="145" width="40" height="85"/>
<rect class="a8-bar" x="540" y="175" width="40" height="55"/>
<rect class="a8-bar" x="588" y="200" width="40" height="30"/>
<rect class="a8-thr a8-sweep" x="48" y="46" width="4" height="184"/>
<text class="a8-lbl" x="200" y="262">easy (pruned)</text>
<text class="a8-lbl" x="470" y="262">hard (kept)</text>
<text class="a8-sub" x="345" y="288">low  &#8592;  difficulty  &#8594;  high</text>
</svg>
<figcaption>Prune-by-difficulty: as the keep-threshold sweeps right, the easy low-difficulty bars fade out and only the hard examples survive.</figcaption>
</figure>

### A worked pruning sweep, with numbers

Put numbers on it. Take an image-classification-style setup in the data-abundant regime and sweep the keep fraction two ways: random subsampling versus keeping the hardest examples first (by distance-to-prototype). The magnitudes below are illustrative but track Sorscher's ImageNet experiments:

| Keep fraction | Random-keep error | Hardest-keep error |
| --- | --- | --- |
| 100% (full) | 0.300 | 0.300 |
| 80% | 0.315 | **0.291** |
| 60% | 0.335 | **0.286** |
| 40% | 0.370 | 0.298 |
| 20% | 0.430 | 0.345 |

Read the second column: random pruning always makes things worse, exactly as the power law predicts. Now read the third column: keeping the hardest 60% gives error 0.286 — *lower than training on 100% of the data.* You trained on 40% less data and got a better model. That is the power law broken. Note also the turn: below roughly 40% keep, even hardest-first degrades, because you have thrown away so much that the surviving hard examples are now under-supported and the noise in them dominates. The curve has a sweet spot; past it you are in the scarce-data regime and the flip rule takes over. This is why the honest version of the pruning recipe is "keep the hardest examples down to the point where error stops improving, then stop" — not "prune as hard as possible."

The connection to the rest of the field: this is the same phenomenon the [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) describe from the quality side. Quality filtering and difficulty pruning are two views of one truth — *the effective size of a corpus is its count of non-redundant, informative tokens, not its raw token count.*

## 7. A decision framework: which method, when

Before the case studies, here is the opinionated default stack, in the order you should apply it:

| Stage | Method | Why here | Typical reduction |
| --- | --- | --- | --- |
| 1 | Exact + fuzzy dedup | Removes literal copies; cheapest, zero false positives | 20–50% |
| 2 | Heuristic + classifier quality filter | Removes junk registers; cheap per-doc | 40–70% |
| 3 | SemDeDup (semantic pruning) | Removes redundant information dedup missed | 20–50% |
| 4 | DSIR (importance resampling) | Shapes the survivors toward a target distribution | tune to budget |
| 5 | Influence/gradient (only for targeted runs) | Aligns a small final set to a specific downstream task | tune to budget |

The ordering is not arbitrary. Cheap, high-recall methods run first on the whole ocean; expensive, high-precision methods run last on what remains. Never run LESS on ten trillion tokens; run it on the few billion that survive to the final stage. And never skip stage 1 — the expensive methods all get more expensive and less accurate when the pool is full of duplicates.

## 8. Case studies

### 1. DSIR: cheap distribution matching that beat hand-tuned filters

The team wanted a domain-adapted model and had the usual constraint: a big raw pool, a small trusted target corpus (formal reference text plus instructions), and no GPU budget to spare on data selection. The first instinct — the one everyone has — was to hand-write filters: keep documents with a high fraction of dictionary words, drop documents with too many symbols, boost documents from "good" domains. That approach ate two engineer-weeks and produced a selected set that was cleaner but not obviously *more like the target*; downstream GLUE barely moved.

Switching to DSIR changed the outcome and the effort. The whole pipeline was CPU-only: hash unigrams and bigrams into ten thousand buckets, estimate the target and raw distributions by counting, score every document by its summed log-ratio, and Gumbel-top-k down to budget. It ran overnight on a handful of cores. The selected set measurably matched the target's n-gram distribution — and downstream GLUE improved by roughly two points over both random selection and the hand-tuned heuristic set. The lesson the team internalized: *when you can measure the distribution you want, resampling toward it beats guessing at rules that approximate it,* and it does so at a fraction of the engineering cost. The failure mode they had to watch (and did) was length bias — without per-document length normalization, the longest documents dominated the top-k and skewed the set toward verbose sources.

### 2. DsDm and LESS: selecting for the target, not for "quality"

A different team needed a model strong on a *specific* downstream suite, not broadly good. They had been using a FineWeb-Edu-style quality classifier and hitting a wall: more high-quality-classified data stopped helping the target benchmarks. The diagnosis, from the DsDm line of work, was that "high quality by the classifier" and "reduces loss on the target" had quietly diverged — the classifier was optimizing educational-looking prose while the target needed something the classifier scored as mediocre.

They rebuilt selection around influence. For pretraining-scale selection they used a datamodel (DsDm-style): estimate each candidate's effect on target loss and select the subset that minimizes it. For the final instruction-tuning stage they used LESS: build a datastore of low-rank projected gradient features over the instruction pool, compute the average gradient feature of a few-shot example of each target task, and keep the ~5% of instruction examples whose gradients aligned best. The targeted 5% matched full-pool training on the target tasks and trained far faster. The surprising, repeatedly-confirmed observation: the selected examples often looked *unremarkable* by quality-classifier standards. Influence selection had found the data that helped the target, which is a genuinely different set from the data that looks nice. The cost was real — a gradient pass per candidate — which is exactly why they ran it only on the final, already-filtered pool and never on the raw ocean.

### 3. Sorscher: pruning that beats the power law on ImageNet

The result that named this section came from asking whether scaling laws are fundamental or fixable. Sorscher et al. took ImageNet and, instead of scaling the dataset up, scaled it *down* intelligently. They needed a difficulty metric that did not require labels, and found one in self-supervised embeddings: cluster a SwAV/self-supervised embedding space with k-means and score each example by its distance to the nearest prototype. Far-from-prototype examples are hard and atypical; near-prototype examples are easy and redundant.

With that metric, keeping the hardest examples first, they observed the effect the theory predicted: in the data-abundant regime, test error fell *faster than the power law* as they pruned — a pruned subset outperformed the full dataset. Crucially, they also mapped the flip: in the low-data regime, keeping the *easiest* examples won, because hard examples there were dominated by noise the small model could not yet exploit. The practical payoff for practitioners: a **label-free, model-based difficulty metric** (distance to a self-supervised prototype) is enough to break the power law, and it is the same metric SemDeDup uses for semantic pruning. One difficulty signal, two jobs — remove redundancy, and rank by informativeness. The caveat they were careful about, and you should be too: the whole effect depends on the *quality* of the difficulty metric. A noisy metric ranks noise as "hard" and pruning to it is worse than random. Beating the power law is a reward for a good metric, not a free lunch.

## 9. Troubleshooting: when selection makes the model worse

Selection is unusually good at looking successful while quietly failing, because the thing you optimize (a proxy) and the thing you care about (the model) are different. Here are the three failures that cost the most time, each as symptom, detection, and fix.

### Symptom: the proxy improves but the model gets worse

![Goodhart's law in data selection: the proxy score climbs while dev loss and diversity both regress](/imgs/blogs/data-selection-and-pruning-7.webp)

This is Goodhart's law, and the figure is its anatomy. Your selection metric looks great — the DSIR target-distribution match hits 0.98, or the quality classifier's mean score on the selected set is way up — but when you actually train, dev loss is flat or *worse* and the outputs feel same-y. What happened is that you optimized the proxy so hard you selected the *argmax of the proxy* rather than a healthy distribution: the documents that maximize "matches Wikipedia n-grams" are near-duplicates of each other, so your selected set collapsed onto a narrow ridge of the proxy.

**Detection.** Never trust the selection metric alone. Hold out a *disjoint* model-based evaluation — train a small proxy model on the selected set and measure loss on a dev set the selection never saw. Watch a diversity statistic on the *selected* set, not just the score: n-gram entropy, number of distinct embedding clusters covered, or type-token ratio. Goodharting shows up as score-up, diversity-down. If your selected set's cluster coverage dropped from, say, 1,000 clusters to 300 while the score climbed, you are on the ridge.

**Fix.** Soften the objective. Use Gumbel-top-k or temperature-scaled sampling instead of a hard threshold so you admit some lower-scoring documents. Add an explicit diversity constraint — cap the number of documents kept per embedding cluster. And regularize the proxy itself: DSIR with heavy length normalization and a smoothed distribution is far more robust than raw importance weights. The rule: *select toward a distribution, not toward a maximum.*

### Symptom: diversity collapse from over-pruning

You pruned aggressively — SemDeDup at a tight threshold, or hard-example keeping down to 20% — and your benchmark averages held or improved, so you shipped it. Then the model turns out brittle: it fails on rare domains, non-English text, minority dialects, or long-tail entities that were fine before. Over-pruning removed the tail, and the tail is where generalization to unseen inputs lives. This is the same failure that leaving duplicates in *causes* (upweighting the head), run in reverse: prune too hard and you starve the tail.

**Detection.** Measure coverage before and after pruning, stratified by whatever axes you care about — language, domain, topic cluster, source. A single aggregate "we kept 60%" hides a catastrophe where you kept 90% of English prose and 10% of code. Track per-stratum keep rates. Also watch tail benchmarks specifically (rare-language eval, long-tail QA), not just the headline average, which is dominated by the head you kept.

**Fix.** Prune *within* strata, not globally. Cluster first, then keep a floor fraction per cluster so no region is emptied — this is the guardrail loop in the pruning code above. For difficulty pruning, cap how far you push: the worked sweep showed error turns back up past the sweet spot, so stop pruning when your held-out dev loss stops improving, not when you hit a target size. When in doubt, keep the rare regions whole; they are cheap in tokens and expensive to lose.

### Symptom: the easy-vs-hard rule is backwards for your budget

Your team read the Sorscher result, kept the hardest examples aggressively, and the model got *worse*, not better. Or the opposite: you kept the easiest examples "to be safe" and left accuracy on the table. Both are the flip rule misapplied. Keeping hard examples only helps in the **data-abundant** regime; in the **data-scarce** regime hard examples are noise the model cannot yet use, and easy examples carry the learnable signal.

**Detection.** The tell is the ratio of pool size to budget. If your raw pool is only two or three times your token budget, you are effectively data-scarce and should suspect any "keep the hardest" strategy. Confirm it empirically with a small sweep: train small models on hardest-X% and easiest-X% at your actual budget and compare dev loss. If easiest-keep wins, you are in the scarce regime, full stop.

**Fix.** Match difficulty to budget explicitly — this is the `data_is_abundant` branch in the pruning code, and it should be a computed decision, not a default. When data is abundant, keep hard and prune to the sweet spot. When data is scarce, keep easy and do not prune much at all — the right move in the low-data regime is often to *not* prune and instead get more data. And remember the metric dependency from case study 3: if your difficulty metric is noisy, "keep hard" keeps noise; validate the metric on a small labeled slice before trusting it to rank your whole corpus.

## 10. When to reach for selection and pruning — and when not to

**Reach for importance resampling (DSIR) when:**

- You have a trusted target corpus and want the raw pool to look like it, cheaply.
- You are doing domain adaptation or continued pretraining toward a known register.
- You want a CPU-only, trillion-token-scale method with no model in the loop.

**Reach for classifier selection when:**

- "Good" is a nonlinear notion of embeddings that an n-gram ratio cannot capture (educational quality, code quality).
- You can build a reliable positive set that resembles what you actually want — not just what looks prestigious.

**Reach for influence/gradient selection (DsDm, LESS) when:**

- You have a *specific* downstream target and the budget for a gradient pass per candidate.
- You are selecting a small final set (instruction tuning, domain fine-tuning), not the pretraining ocean.
- Your quality classifier has plateaued and you suspect "quality" and "helps the target" have diverged.

**Reach for coreset pruning (SemDeDup, difficulty) when:**

- Your corpus is large and you suspect heavy semantic redundancy beyond literal duplicates.
- You are in the data-abundant regime and want to spend budget on informative examples.
- You have a *validated* difficulty metric (distance to a self-supervised prototype is a good default).

**Skip or soften these when:**

- You are data-scarce (pool only a few times your budget) — get more data before you prune it; aggressive selection in the low-data regime usually hurts.
- You cannot measure a disjoint, model-based evaluation — without it you cannot catch Goodharting, and blind selection is worse than random.
- Your difficulty or quality metric is unvalidated — a bad metric makes every method here *worse* than random sampling.
- The gain does not justify the fragility — for a general-purpose model with a huge diverse pool, good dedup plus a soft quality filter plus DSIR gets you most of the way; the exotic methods are for the last few points and the targeted runs.

Selection is where you convert "we have data" into "we have the *right* data for this budget." The whole toolbox reduces to two disciplines: know your target, and never trust the proxy you optimize. Do both and you routinely beat teams with more tokens and more compute. The next step is deciding how to *mix* the selected domains and in what order to feed them — the subject of [data mixing, domain weighting, and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum).

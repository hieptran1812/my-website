---
title: "Data quality is a scaling axis: filtering, dedup, and the quality-quantity tradeoff"
date: "2026-06-15"
description: "Learn to treat data curation as a multiplicative compute factor: how filtering and deduplication shift the loss-vs-compute curve, and why the optimal filter aggressiveness depends on your compute budget."
tags: ["scaling-laws", "data-quality", "data-curation", "deduplication", "data-filtering", "quality-quantity-tradeoff", "fineweb", "dclm", "semdedup", "large-language-models", "pretraining-data", "compute-optimal"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

Most of the scaling-laws literature treats the training corpus as a single scalar: a token count $D$ that you plug into a loss law and forget about. [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) tells you to feed about 20 tokens per parameter; [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) tells you how many times you can re-read those tokens before the gains run out. Both of them quietly assume every token is worth the same as every other token. That assumption is false, and the gap between "a token" and "a *good* token" turns out to be one of the largest, cheapest levers in the whole pretraining stack. A model trained on a carefully filtered and deduplicated corpus reaches the same loss as a model trained on raw web text using roughly **six times less compute** — that is the headline DataComp-LM number, and it dwarfs most architecture tweaks. The diagram below is the mental model for the entire post: filtering and deduplication take a raw web dump and produce a curated corpus, and the effect of that curation is to shift the loss-versus-compute curve down — equivalent to a multiplicative factor on your compute budget — with one crucial nuance that the rest of the article is about.

![A branching diagram showing a raw Common Crawl dump flowing through a quality filter and a deduplication stage into a curated corpus, which then both shifts the loss curve down for a large compute saving and raises the question of compute-dependent filter aggressiveness](/imgs/blogs/data-quality-scaling-laws-1.png)

The nuance, in one sentence: the *optimal* amount of filtering is not a fixed property of your dataset, it is a function of how much compute you have relative to how much data you can get. This is the central result of Goyal et al. 2024 ("Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic," arXiv:2404.07177), and it is the reason a naive "always keep only the top 10% highest-quality data" rule will quietly sabotage a large training run. The same filter that gives you a +7.5% accuracy boost at small scale will *lose* to unfiltered Common Crawl once you have enough compute to over-repeat the curated set. So data quality is genuinely a scaling axis — it has its own curve, its own crossover points, and its own compute-dependent optimum — and treating it as a one-time preprocessing decision is a category error.

> [!important] The one number to remember: about 6.6x less compute from good data
> - **Data quality is a multiplicative compute factor.** DCLM-Baseline reaches Llama-3-8B-class quality on the 53-task average with about **6.6x less training compute** than the comparison, purely by curating the corpus. This is a larger lever than most architecture changes.
> - **Deduplication is the highest-ROI step.** Lee et al. 2021 found C4 contained a 61-word string repeated over **60,000 times**; deduplicating cuts verbatim memorization roughly **10x**, fixes a >4% train-test overlap that was inflating eval numbers, and reaches the same or better perplexity in fewer steps.
> - **Semantic dedup goes further than exact dedup.** SemDeDup removes about **50% of LAION with minimal quality loss (~2x faster training)** by clustering embeddings and dropping semantic near-duplicates, not just byte-identical ones.
> - **The optimal filter aggressiveness depends on compute (the QQT).** At low compute, hard filtering wins (+7.5% accuracy in Goyal's CLIP setting); at high compute, the small filtered set gets over-repeated, its utility decays, and unfiltered diverse data wins. Filter hard when data is abundant relative to compute; loosen toward diversity when you would otherwise re-epoch a tiny set many times.
> - **Repeated data has a utility half-life.** Each repetition of a curated subset halves its marginal utility on a fixed schedule, $b_{k+1} = b \cdot (1/2)^{k/\tau}$; combining pools multiplies the effective half-life, which is the formal reason diversity beats repetition at scale.
> - **Per-dump dedup beat global dedup** in the FineWeb work, because global dedup removes so many near-duplicates that it ends up upsampling the low-quality remainder — a direct echo of the quality-quantity tradeoff.
> - **The practical recipe:** dedup aggressively (it is nearly free and only helps), then set filter strictness from your compute-to-data ratio, not from a fixed quality threshold.

## Why "more data" is the wrong frame

The right way to feel this result is to notice how the standard scaling-law story sets you up to ignore it. The [Kaplan](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) and Chinchilla laws are written in terms of $N$ (parameters) and $D$ (tokens), and they are spectacularly successful at predicting loss from those two numbers. Their very success makes it easy to believe that $D$ is the whole data story — that the corpus is fungible, and your only job is to get the count high enough. Here is the assumption-versus-reality table a senior data engineer keeps in their head.

| Question | The naive "token count is everything" view | The reality once you measure |
|---|---|---|
| Are all tokens worth the same? | Yes — $D$ is a scalar, count is the lever | No — a good token can be worth several noisy ones; quality shifts the curve |
| Does removing data ever help? | No — more tokens is strictly better | Yes — deduping and filtering routinely *raise* quality at *lower* token counts |
| Is the best corpus the biggest corpus? | Yes — keep everything you can crawl | No — RefinedWeb and FineWeb beat curated corpora using filtered web alone |
| Is "filter to the top X%" a fixed recipe? | Yes — pick a quality threshold once | No — optimal aggressiveness depends on compute (the QQT) |
| Does repeating data preserve its value? | Roughly, up to the [4-epoch rule](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) | Its marginal utility decays with a half-life; small sets decay fast |
| What is the biggest cheap win in pretraining? | A bigger model or more steps | Curation — up to ~6.6x effective compute, before you touch the model |

The two rows that matter most are the first and the last. If a good token is worth several noisy ones, then the effective token count $D$ in the loss law is not what you crawled — it is a *quality-weighted* count, and curation is how you raise the weight. And if the optimal filter strictness depends on compute, then there is no single "clean version" of your dataset to compute once and reuse forever; the right corpus for a 1B-token run is a different corpus than the right one for a 1T-token run.

> If you take one thing from this post: stop thinking of your corpus as a fixed pile of $D$ tokens. Think of it as a *quality distribution* you get to reshape, and the right reshaping depends on how much compute you are going to pour through it.

This reframing has a clean mathematical home. Recall the Chinchilla parametric loss law,

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

where $E$ is the irreducible entropy floor, $N$ is parameters, $D$ is tokens, and $A, B, \alpha, \beta$ are fitted constants. The finite-data penalty $B/D^{\beta}$ is the term curation attacks. There are two ways to read what good data does to it. The first reading: curation raises the *effective* $D$ at fixed crawled size, because cleaner tokens carry more signal per token, so $B/D^{\beta}$ falls faster than the raw count suggests. The second reading, which the data-constrained-scaling work makes precise, is that curation changes the constants $B$ and $\beta$ themselves — a better corpus is a different scaling regime, not just a different point on the same curve. Both readings agree on the empirical fact: at any fixed compute, a better corpus sits on a lower loss curve. The disagreement is only about whether you model it as a shift along the $D$ axis or a change in the law's shape, and for practical budgeting the shift-along-$D$ picture is the more useful one.

### A short history of how data quality became a first-class axis

It is worth tracing the lineage, because data quality went from "preprocessing hygiene" to "a scaling axis with its own laws" in about three years, and the order of discoveries explains why the field's intuitions are still catching up.

The first wave was deduplication as a *correctness* fix. In 2021, Lee et al. ("Deduplicating Training Data Makes Language Models Better," arXiv:2107.06499) showed that the standard web corpora were riddled with duplicates — not just spam, but the same passages crawled from mirror sites, syndicated articles, boilerplate, and templated pages. Their finding that C4 contained a single 61-word string repeated more than 60,000 times became the canonical example. The framing at the time was about memorization and eval contamination: dedup made models leak training data less and made benchmark numbers honest. The efficiency win — same perplexity in fewer steps — was almost a side effect.

The second wave generalized dedup from bytes to *semantics*. Exact-string and MinHash methods only catch documents that are literally or near-literally identical. But two product reviews that say the same thing in different words, or two stock-photo captions describing the same scene, are duplicates in every sense that matters to a learner, and exact methods miss them entirely. SemDeDup (Abbas et al. 2023, arXiv:2303.09540) embedded every example, clustered with k-means, and dropped semantic near-duplicates by cosine threshold — removing about half of LAION with minimal quality loss. D4 (Tirumala et al. 2023, arXiv:2308.12284) pushed further, combining semantic dedup with explicit diversification, and made the striking claim that *intelligently selected* repeated data can beat the baseline.

The third wave was *model-based filtering*, where you train a small classifier to predict whether a document is high quality and keep only the documents it likes. RefinedWeb (Penedo et al. 2023, arXiv:2306.01116) and then FineWeb / FineWeb-Edu (Penedo et al. 2024, arXiv:2406.17557) showed that filtered web text *alone* — no books, no curated academic corpora — could match or beat the hand-assembled mixtures everyone had been using. DataComp-LM (Li, Fang et al. 2024, arXiv:2406.11794) turned the whole thing into a benchmark and found model-based filtering dominant, with DCLM-Baseline hitting Llama-3-8B-class results at roughly 6.6x less compute.

The fourth wave, and the conceptual punchline, was Goyal et al. 2024 noticing that none of the above had a *fixed* answer. The best filter at one compute budget is the wrong filter at another. That is the quality-quantity tradeoff, and it is what turns "data quality" from a checklist of preprocessing steps into a genuine scaling axis with a compute-dependent optimum.

## 1. The quality-quantity tradeoff: the crossover that breaks naive filtering

**Senior rule of thumb: a filter is not a property of your data, it is a joint property of your data and your compute budget. Re-derive its aggressiveness every time the budget changes.** This is the single most counterintuitive idea in data curation, and the figure below is the whole argument in one picture.

![A line chart with two loss-versus-compute curves crossing: a solid curve for a heavily filtered small high-quality dataset that starts lower then rises as it is over-repeated, and a dashed curve for unfiltered Common Crawl that starts higher but keeps descending, with the crossover marked and annotations explaining low-compute and high-compute regimes](/imgs/blogs/data-quality-scaling-laws-2.png)

Read the chart carefully because it overturns the obvious intuition. The solid curve is a model trained on a small, aggressively filtered, high-quality subset — say the top 10% of Common Crawl by a quality classifier. The dashed curve is a model trained on the full unfiltered crawl. At low compute (the left side), the filtered curve is *below* the unfiltered one: with little compute, you can only afford to see a limited number of tokens, so seeing high-quality ones is strictly better, and the high-quality subset is large enough that you see each token roughly once. This is the regime where filtering looks like free money. In Goyal's CLIP/DataComp experiments, at a 32M-sample compute budget, Top-10% filtering delivered about +7.5% accuracy at the 128M evaluation point.

Now follow the curves to the right. As compute grows, you run more iterations, and the small filtered subset gets *repeated*. The first repetition is nearly free, the second less so, and by the time you have over-repeated a small set dozens of times its marginal utility has decayed to almost nothing. Meanwhile the unfiltered crawl is so large that you never run out of fresh tokens, so it keeps descending. The two curves cross. Past the crossover — Goyal reports this happening above roughly 450M iterations in their setting, where unfiltered Common Crawl beats LAION-filtered data — the *unfiltered* corpus wins, because diversity beats repetition once you have enough compute to exhaust the small set.

The way this works mechanistically is that filtering trades quantity for quality, and the value of that trade flips sign as compute grows. The yellow crossover dot is the budget at which the trade breaks even. Below it, filter hard. Above it, the filter is actively hurting you, and you should loosen toward diversity.

### The utility half-life: why repetition decays

To make the crossover quantitative, Goyal models the *utility* of a data pool as something that decays with repetition. Let $b$ be the marginal utility (the loss reduction per token) of a pool on its first pass. On the $k$-th repetition, the utility is

$$b_{k+1} = b \cdot \left(\tfrac{1}{2}\right)^{k/\tau},$$

where $\tau$ is the **utility half-life** — the number of repetitions over which utility halves. A small, aggressively filtered pool has a short half-life: it is high quality, so its first pass is very useful, but there is not much of it, so it decays fast under repetition. A large diverse pool has a long half-life. The next figure makes the decay concrete.

![A line chart showing marginal utility decaying as the number of repetitions increases: a steep solid curve for a small curated pool with a short half-life, and a shallower dashed curve for combined pools with a longer effective half-life, with annotations giving the decay law and noting that gains are nearly free up to about four epochs and collapse past about sixteen](/imgs/blogs/data-quality-scaling-laws-5.png)

The mechanism here connects directly to the [4-epoch rule from data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws). Muennighoff et al. found that up to about 4 epochs of repetition is essentially free (an 8.7B model trained for 4 epochs over 44B unique tokens ended only 0.5% higher in validation loss than the all-unique run), with diminishing returns out to about 16 epochs and a hard collapse past that. The utility half-life is the same phenomenon viewed through a different lens: the 4-epoch "free" region is where $(1/2)^{k/\tau}$ is still close to 1, and the 16-epoch collapse is where it has fallen near zero.

The deepest part of Goyal's model is the claim that *combining pools multiplies the effective half-life*. If you have two high-quality pools each with half-life $\tau$, training on their union does not just give you twice the tokens — it gives you a pool whose utility decays more slowly, because the model can extract signal from one pool while the other is "resting." This is the formal reason a diverse mixture beats a single curated source at scale: not because the average token is better, but because the aggregate decays slower under the repetition that large compute forces on you. This is also why the high-compute side of the crossover favors unfiltered data: the unfiltered crawl is, in effect, an enormous combination of pools with a very long effective half-life.

### Working the crossover numerically

Let me put numbers on this so the tradeoff stops being abstract. Suppose you have crawled 100B tokens. A top-10% quality filter leaves you 10B high-quality tokens; the unfiltered set is the full 100B. Say (using round numbers for illustration, in the spirit of Goyal's setting) the high-quality tokens have first-pass utility $b = 1.0$ and a half-life of $\tau = 4$ repetitions, while the unfiltered tokens have first-pass utility $b = 0.6$ but a half-life of $\tau = 40$.

At a small budget of 10B training tokens, the filtered run sees its 10B-token pool exactly once: total utility $\approx 10\text{B} \times 1.0 = 10$ utility-units. The unfiltered run also processes 10B tokens, all unique, at utility 0.6: total $\approx 6$ utility-units. Filtering wins, 10 to 6 — and this matches the qualitative +7.5% boost.

Now scale the budget to 200B training tokens. The unfiltered run sees its 100B pool twice; even with mild decay it accumulates roughly $100\text{B} \times 0.6 \times (1 + 0.97) \approx 118$ utility-units. The filtered run must see its 10B pool 20 times. With $\tau = 4$, the utility of the 20th pass is $1.0 \times (1/2)^{19/4} \approx 0.037$ — almost nothing. Summing the geometric-ish series of 20 passes gives roughly $10\text{B} \times 1.0 \times \sum_{k=0}^{19}(1/2)^{k/4} \approx 10\text{B} \times 5.8 = 58$ utility-units. Now the unfiltered run wins, 118 to 58. The curves have crossed somewhere between 10B and 200B of compute, exactly as the figure shows. The numbers are illustrative, but the structure — early filtering win, late filtering loss, a crossover whose location depends on pool sizes and half-lives — is the real result.

The practical reading: **the crossover compute is the quantity to estimate.** If your planned training compute is well below it, filter hard. If it is above, loosen the filter or expand the pool. And critically, the crossover moves: a bigger crawl pushes it right (more unique high-quality tokens before you must repeat), and a larger model at fixed tokens pushes it left.

### Where these numbers come from: the CLIP / DataComp setting

A fair question at this point is why a post about *language-model* data quality keeps citing a paper whose headline experiments are about CLIP. Goyal et al. ran their study in the DataComp benchmark, which trains CLIP image-text models on filtered subsets of a web-scale image-text pool (LAION and Common Crawl image-text pairs), measured by zero-shot accuracy on tasks like ImageNet. The +7.5% and the >450M-iteration crossover are CLIP/DataComp numbers, and it is honest to flag that.

But the *mechanism* — utility decaying under repetition, with a half-life that depends on pool size and quality — is modality-agnostic. It is a statement about how much signal a fixed set of examples can give a learner before the learner has extracted everything, and that is true whether the examples are image-text pairs or text documents. The language-model literature confirms the same shape from the other direction: the [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) work measured the repetition-decay curve directly for LLMs and found the 4-epoch-free, 16-epoch-collapse pattern that the half-life model predicts. So the safe way to use Goyal is: take the *qualitative law* (filtering aggressiveness must depend on compute; small high-quality sets decay fast) as established across modalities, and take the *specific numbers* (+7.5%, 450M iterations) as the CLIP-setting evidence for it. The language-model crossover exists; its exact location depends on your corpus and model, which is precisely why the decision procedure in section 5 computes it from your own numbers rather than quoting a universal constant.

One more nuance from Goyal worth carrying forward: they found that the quality of a filter is itself a function of compute. A filter that looks excellent when you evaluate the small filtered set at low compute can look mediocre at high compute, because what you are really measuring is "filtered-set utility *at this number of repetitions*," not some intrinsic, compute-free quality. This is why "what is the best filter?" is a malformed question. The well-formed question is "what is the best filter *for my compute budget and my crawl size?*" — and the answer changes as those change.

## 2. Deduplication: the highest-ROI curation step

**Senior rule of thumb: deduplicate before you do anything else, because unlike filtering it has essentially no downside and three independent upsides.** Filtering trades quantity for quality and can backfire (that is the whole point of section 1). Deduplication, in contrast, removes tokens that were never adding information in the first place, so it improves quality *and* honesty *and* efficiency at once, with no compute-dependent crossover to worry about. The before-and-after below shows what it buys.

![A before-and-after comparison: the left column shows a raw C4 corpus with a 61-word string repeated over sixty thousand times, high verbatim memorization, over four percent train-test overlap, and baseline perplexity; the right column shows a deduplicated corpus with about ten times less memorization, a clean train-test split, honest eval numbers, and the same or better perplexity in fewer steps](/imgs/blogs/data-quality-scaling-laws-3.png)

Lee et al. 2021 is the foundational paper, and its examples are worth internalizing because they show how bad the problem is in corpora everyone trusted. C4 — the cleaned Common Crawl that powered T5 and many later models — contained a 61-word string repeated more than 60,000 times. It contained a more-than-4% overlap between training and test sets, which means a chunk of every benchmark a C4-trained model was evaluated on had literally been in its training data. And it contained the long tail of web duplication: syndicated news, mirror sites, SEO templates, legal boilerplate, and auto-generated pages.

Deduplicating this corpus did three things. First, it cut **verbatim memorization roughly 10x** — a deduplicated model regurgitates training text about ten times less often, which matters for privacy, copyright, and the general sanity of not having your model parrot whole web pages. Second, it **fixed the train-test contamination**, making evaluation numbers honest; a model that has seen the test set during training is not measuring generalization, and the >4% overlap meant a meaningful fraction of reported scores were inflated. Third — and this is the efficiency win that makes dedup a scaling lever rather than just hygiene — the deduplicated model reached the **same or better perplexity in fewer training steps**, because every duplicate pass had been spending compute teaching the model something it already knew.

### Exact-substring and MinHash near-duplicate detection

Lee et al. used two complementary methods, and understanding the difference matters because they catch different failure modes.

**ExactSubstr** finds verbatim repeated spans. It builds a suffix array over the entire corpus and identifies any substring of at least a threshold length (they used 50 tokens) that appears more than once, then removes the duplicate occurrences. This is exact and surgical: it catches the 61-word-string-repeated-60,000-times case perfectly, and it operates at the substring level so it can excise a duplicated paragraph from an otherwise-unique document without throwing the whole document away. The cost is that it only catches *byte-identical* spans; change one word and the match breaks.

**NearDup (MinHash)** finds near-duplicate *documents*. It computes MinHash signatures — a locality-sensitive hash where the probability two documents collide equals their Jaccard similarity over n-gram shingles — and clusters documents whose estimated similarity exceeds a threshold, keeping one representative per cluster. This catches documents that are 95% identical but differ in a header, a date, or a few edited words, which ExactSubstr would miss. The cost is that it works at document granularity, so it is coarser; it cannot remove a single duplicated paragraph the way ExactSubstr can.

The two are used together because they are complementary: ExactSubstr for precise span-level verbatim removal, MinHash for fuzzy document-level near-duplicate clustering. A representative pipeline runs MinHash first to cluster and drop near-dup documents, then ExactSubstr to clean up repeated spans within the survivors.

Here is the shape of a MinHash dedup in pseudocode-but-runnable form, using the `datasketch` library that most of these pipelines lean on:

```python
from datasketch import MinHash, MinHashLSH

def shingles(text, k=5):
    tokens = text.split()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}

# num_perm controls the accuracy/cost tradeoff; 128 is the common default.
lsh = MinHashLSH(threshold=0.8, num_perm=128)
signatures = {}

for doc_id, text in corpus:                      # corpus: iterable of (id, str)
    m = MinHash(num_perm=128)
    for sh in shingles(text, k=5):
        m.update(sh.encode("utf-8"))
    # Query before insert so the first doc in each near-dup cluster survives.
    if not lsh.query(m):                          # no near-duplicate seen yet
        lsh.insert(doc_id, m)
        signatures[doc_id] = m                    # keep this representative
    # else: drop doc_id as a near-duplicate of an earlier document
```

The `threshold=0.8` is the Jaccard-similarity cutoff: documents sharing 80% of their 5-gram shingles are treated as duplicates. Tighten it (higher threshold) to remove only very close duplicates; loosen it to remove more aggressively. At web scale you do not hold the whole `lsh` index in one process — you bucket by MinHash band across a cluster of machines — but the logic is exactly this.

### Deduplication at web scale: the engineering reality

The pseudocode above hides the part that makes dedup hard in practice: scale. Common Crawl is petabytes; you cannot hold a global similarity index in one process, and you cannot afford an all-pairs comparison ($O(n^2)$ on trillions of documents is a non-starter). The production answer is **banded LSH on a distributed cluster.** You split each document's MinHash signature into $b$ bands of $r$ rows each; two documents become *candidates* if they collide in at least one band; you then verify candidates with the full Jaccard estimate. The band/row split tunes the similarity threshold: the probability that two documents with Jaccard similarity $s$ collide in at least one band is $1 - (1 - s^r)^b$, an S-curve whose steepness and midpoint you set by choosing $b$ and $r$. This turns a quadratic comparison into a near-linear bucket-and-verify, distributable across hundreds of machines because each band is an independent hash bucket.

ExactSubstr has its own scaling trick: the suffix array. Building a suffix array over a concatenated corpus lets you find every repeated substring of length $\ge k$ in roughly $O(n \log n)$ time and linear space, rather than comparing every span to every other span. The Lee et al. release used exactly this, and it is why ExactSubstr is tractable on hundreds of billions of tokens. The practical upshot: both dedup methods have near-linear algorithms, which is what makes "dedup first, always" affordable advice rather than a counsel of perfection. The cost is real — a full dedup pass over a trillion-token corpus is a meaningful compute and storage job — but it is a one-time cost amortized across every model you ever train on that corpus, and it is dwarfed by the training compute it saves.

There is a subtle correctness trap in distributed dedup worth flagging: **order-dependence.** When you keep "the first document in each near-duplicate cluster" (as the MinHash pseudocode does), the *first* depends on the order you process documents, and across a sharded cluster that order is nondeterministic. Two runs can keep different representatives, and worse, a naive implementation can keep zero or two representatives of a cluster split across shard boundaries. The fix is to make the keep-decision deterministic — e.g., keep the document with the lexicographically smallest ID in each cluster — so the result is reproducible regardless of sharding. This is the kind of bug that does not show up in a small test and silently corrupts a web-scale run.

### Second-order optimization: per-dump versus global deduplication

The non-obvious gotcha, and one of the most important practical findings in the whole area, comes from the FineWeb work: **per-dump deduplication beat global deduplication.** The intuition is subtle and worth dwelling on because it is a direct echo of the quality-quantity tradeoff.

Common Crawl publishes a new "dump" (snapshot of the web) every month or two. The obvious thing is to deduplicate *globally* across all dumps at once, so that a page crawled in 2019 and again in 2023 is counted once. The FineWeb team tried this and found it made models *worse*. The reason: global dedup is so effective at removing near-duplicates that it disproportionately strips out the *high-quality* content (good articles get mirrored and re-crawled across dumps, so they have many near-duplicates and get aggressively collapsed), while the *low-quality* spam that is unique to each dump survives. Global dedup therefore ends up upsampling the junk. Per-dump dedup — deduplicating within each snapshot but allowing cross-snapshot repeats — preserves a healthier quality distribution. This is the same lesson as the QQT crossover: maximally aggressive removal is not maximally good, because removal interacts with the quality distribution in ways that can backfire.

## 3. From exact dedup to semantic clustering: the SemDeDup ladder

**Senior rule of thumb: byte-level deduplication is the floor, not the ceiling — the duplicates that hurt most are semantic, and you need embeddings to find them.** Exact and MinHash methods catch documents that are literally similar. But the web is full of content that is *semantically* redundant without being textually similar: ten thousand near-identical product descriptions written by different sellers, stock-photo captions describing the same scene, paraphrased news, and templated content with the slots filled differently. To a learner these are duplicates — the eleven-thousandth product description teaches nothing the first thousand did not — but no n-gram method will catch them. The matrix below lays out the full ladder of deduplication methods.

![A four-row comparison matrix of deduplication methods: ExactSubstr matching identical spans at substring granularity removing a few percent for ten times less memorization, MinHash NearDup matching near-duplicate documents removing tens of percent and fixing test overlap, SemDeDup matching semantic near-duplicates at embedding-cluster granularity removing about half the data for two times faster training, and D4 combining dedup with diversification removing about twenty percent effective for a two percent average gain](/imgs/blogs/data-quality-scaling-laws-6.png)

**SemDeDup** (Abbas et al. 2023) is the semantic step. The recipe is: embed every example with a pretrained encoder, cluster the embeddings with k-means, and within each cluster remove examples whose cosine similarity to a kept representative exceeds a threshold. Because it operates in embedding space, it catches paraphrases and semantic restatements that n-gram methods miss entirely. On LAION (the large image-text dataset behind many open CLIP and diffusion models), SemDeDup removed about **50% of the data with minimal performance loss, roughly doubling training speed.** Half the dataset was, in a meaningful sense, redundant — and you could only see that in embedding space.

The mechanism is worth stating precisely. Two examples that are semantic near-duplicates land close together in a good embedding space; k-means groups nearby points into clusters; within a cluster, the cosine threshold says "anything this similar to a point we are keeping is redundant, drop it." The threshold is the knob: a tight threshold removes only very close semantic dups (conservative), a loose one removes more (aggressive). Here is the core of it:

```python
import numpy as np
from sklearn.cluster import KMeans

# embeddings: (N, d) float32 array, L2-normalized so dot product = cosine.
def semdedup(embeddings, n_clusters=11000, sim_threshold=0.95):
    labels = KMeans(n_clusters=n_clusters, n_init=1).fit_predict(embeddings)
    keep = np.ones(len(embeddings), dtype=bool)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        E = embeddings[idx]                       # points in this cluster
        sims = E @ E.T                            # cosine sims (normalized)
        np.fill_diagonal(sims, 0.0)
        # Drop the second-and-later member of each near-duplicate pair.
        for i in range(len(idx)):
            if keep[idx[i]] and (sims[i] > sim_threshold).any():
                dups = idx[np.where(sims[i] > sim_threshold)[0]]
                keep[dups[dups > idx[i]]] = False
    return keep                                   # boolean mask of survivors
```

The `sim_threshold=0.95` is the cosine cutoff; lowering it toward 0.9 removes more aggressively. The number of clusters trades accuracy for cost — more clusters mean smaller, tighter clusters and more precise dedup, at the cost of more k-means compute. In practice you run this on GPU with FAISS rather than scikit-learn, but the logic is identical.

### D4: dedup plus diversification, and the "intelligent repetition" result

**D4** (Tirumala et al. 2023, "Document De-Duplication and Diversification") goes one step past SemDeDup by combining semantic dedup with explicit *diversification*. After removing near-duplicates, D4 also prunes examples that sit in the densest regions of embedding space — not because they are duplicates of any single example, but because that region of the space is over-represented. The result is a corpus that is both deduplicated and more uniformly spread across the embedding space.

On a 6.7B-parameter model, D4 delivered about **20% training efficiency** and a **+2% average improvement across 16 downstream tasks** versus the baseline. The headline conceptual claim — and the one that connects D4 back to the quality-quantity tradeoff — is that *repeating intelligently selected data beats training on the baseline.* That sounds paradoxical given everything section 1 said about repetition decaying utility, but the resolution is exactly the half-life argument: a D4-selected pool is more diverse, so it has a longer effective half-life, so its repeated passes retain more utility than repeated passes over the un-pruned baseline. D4 is, in effect, engineering the data pool to have a longer half-life on purpose.

### Second-order optimization: dedup interacts with everything downstream

The gotcha for the whole dedup ladder: deduplication changes the token count, which changes your [Chinchilla allocation](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) and your epoch count. If you planned a 1T-token run on a 1T-token corpus and then SemDeDup removes 40%, you now have 600B unique tokens — which means either a smaller compute-optimal model, or repeating to ~1.7 epochs to hit your token budget. Dedup is not a free preprocessing step you can bolt on without re-planning; it shifts you along the data axis, and you have to re-run the allocation math. The good news is that the math from data-constrained scaling tells you exactly how: with 600B unique tokens, you are still well inside the ~4-epoch free zone for most budgets, so the dedup is almost certainly a net win even after accounting for the repetition it forces.

## 4. Model-based filtering: how curated web beats curated corpora

**Senior rule of thumb: a small classifier that scores documents for quality is worth more than a hand-assembled mixture of "good" sources — let the model tell you what good data looks like.** The most surprising result of the 2023–2024 data wave is that filtered web text *alone* beats the carefully curated mixtures (web + books + Wikipedia + academic papers + code) that everyone had assumed were necessary. The pipeline that produces such a corpus is the subject of the next figure.

![A horizontal pipeline showing web text flowing through six stages: Common Crawl WARC dumps, a URL filter for blocklists and spam, fastText language identification keeping English above a threshold, heuristic filters using Gopher and C4 rules, per-dump MinHash deduplication, and a model-based edu-quality classifier, producing a trainable corpus of 1.3 trillion educational tokens with large MMLU and ARC gains](/imgs/blogs/data-quality-scaling-laws-4.png)

Walk the stages left to right, because the order is load-bearing. **URL filtering** comes first and is the cheapest: a blocklist drops adult, gambling, and known-spam domains before you spend any compute parsing their content. **Language identification** (a fastText classifier) keeps documents above a confidence threshold for the target language — FineWeb keeps English above about 0.65 confidence. **Heuristic filters** apply the Gopher and C4 quality rules: drop documents with too few words, too high a symbol-to-word ratio, too many bullet points, missing punctuation, and the other tell-tale signs of machine-generated or boilerplate text. **Deduplication** (per-dump MinHash, per section 2's finding) removes near-duplicates. And finally **model-based classification** scores each surviving document and keeps the highest-quality ones.

RefinedWeb (Penedo et al. 2023) established that this pipeline, applied to Common Crawl alone, produces a corpus competitive with the curated mixtures used to train the best models of the era. FineWeb (2024) refined it and released a 15T-token corpus. The most striking variant is **FineWeb-Edu**, where the model-based filter is trained specifically to recognize *educational* content — text that explains things clearly, the way a textbook or a good tutorial does. Filtering the 15T FineWeb down to about **1.3T educational tokens** produced large gains on knowledge-and-reasoning benchmarks like MMLU and ARC, far out of proportion to the token count. A 1.3T-token educational corpus beat much larger general corpora on exactly the evals you care about for a capable assistant.

### DataComp-LM and the 6.6x compute factor

DataComp-LM (DCLM, Li/Fang et al. 2024) turned data curation into a controlled benchmark: fix the model architecture, the training recipe, and the compute, and let competitors vary only the data filtering. This is the experimental design that lets you *measure* data quality as a scaling factor, because everything else is held constant. The dominant approach was model-based filtering, and the headline result is the cleanest single number in this whole post: **DCLM-Baseline trained a 7B model to 64% MMLU on 2.6T tokens, matching Llama-3-8B on the 53-task average with about 6.6x less compute.**

Sit with that number. Llama-3-8B is a strong model trained on 15T tokens. A 7B model trained on a *better-filtered* 2.6T-token corpus matched it on the broad task average while using roughly one-sixth the training compute. The entire difference is data curation. No new architecture, no new optimizer, no new objective — just a better classifier deciding which web documents to keep. This is why "data quality is a scaling axis" is not a slogan: it is a 6.6x multiplicative factor on compute, measured under controlled conditions, larger than almost any architectural intervention you could name.

### How the quality classifier actually works

It is worth being concrete about what "model-based filtering" means mechanically, because the phrase makes it sound more mysterious than it is. The classifier is small and cheap — often a fastText linear model over n-gram features, or a small transformer — and it outputs a single quality score per document. The art is entirely in how you build its *training labels*, because the classifier learns to recognize whatever you define as good.

The cheapest labeling trick, used by GPT-3's original filter and many since, is **contrastive labeling against a reference corpus.** Treat a known-good corpus (Wikipedia, books, WebText) as positive examples and a random sample of raw Common Crawl as negatives, then train a binary classifier to separate them. The classifier learns "what makes a document look like Wikipedia rather than random web text," and you keep the documents it scores highly. This requires no human annotation at all — the labels come for free from the choice of reference corpus.

FineWeb-Edu took a more deliberate route: they used a large language model to *rate* a sample of documents on an educational-value scale (0 to 5, "how useful is this for learning?"), then trained a small classifier to reproduce those ratings, then ran the cheap classifier over the whole 15T-token corpus. The expensive LLM annotates a few hundred thousand documents; the cheap classifier scales that judgment to trillions. This LLM-as-annotator-then-distill-to-a-cheap-filter pattern is now the dominant recipe, because it lets you define "quality" with a rich natural-language rubric while keeping the per-document inference cost low enough for web scale.

```python
import fasttext

# Build a labeled file: each line is "__label__keep <text>" or
# "__label__drop <text>". Labels come from an LLM rubric or a reference corpus.
# fastText trains a linear model over word/char n-grams in seconds-to-minutes.
model = fasttext.train_supervised(
    input="quality_labels.txt",
    epoch=5, lr=0.5, wordNgrams=2, dim=256,
)

def keep(text, threshold=0.85):
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    # Keep documents the classifier scores as "keep" above the threshold.
    return labels[0] == "__label__keep" and probs[0] >= threshold
```

The `threshold` here is the QQT knob in disguise: raise it to keep a smaller, higher-quality set (aggressive); lower it to keep more (loose). Everything section 1 said about choosing aggressiveness from compute applies directly to this one number.

### Second-order optimization: the filter can overfit to the eval

The gotcha with model-based filtering, and the reason to be a little paranoid about that 6.6x: a quality classifier trained to recognize "good" documents can learn to recognize *documents that look like your benchmark*, which quietly contaminates your evaluation. If your edu-quality classifier was trained on examples that resemble MMLU questions, it will preferentially keep MMLU-like text, and your MMLU score will rise for reasons that are partly contamination rather than partly capability. The defenses are decontamination (explicitly removing benchmark text from the training corpus, which dedup helps with) and evaluating on held-out tasks the classifier could not have targeted. The DCLM and FineWeb teams take this seriously precisely because the filter is powerful enough to game the metric if you are not careful. Powerful levers cut both ways.

There is a subtler version of this failure that is worth naming, because it is easy to miss: the classifier can be *correct* about quality and still hurt you by collapsing diversity. A filter that keeps only the cleanest, most textbook-like prose will produce a corpus that is high-quality but narrow — it strips out the messy, conversational, long-tail text that teaches a model robustness to real-world inputs. Models trained on over-sanitized corpora can score well on knowledge benchmarks and then stumble on the kind of informal, noisy prompts real users actually type. The fix is the same diversification principle D4 uses: filter for quality, but cap how much of any one stylistic region you keep, so the corpus stays spread across the space rather than collapsing onto a single "clean" mode. Quality and diversity are different axes, and a filter that maximizes the first can quietly destroy the second.

## 5. Compute-dependent filtering in practice: the decision matrix

**Senior rule of thumb: choose your filter aggressiveness from the ratio of compute to available data, not from a fixed quality threshold — and when in doubt, the failure mode of over-filtering at high compute is worse than under-filtering.** Section 1 gave you the crossover; this section turns it into a decision you can actually make. The two variables that matter are how aggressive your filter is and how much data you have relative to compute. The matrix below crosses them.

![A two-by-two decision grid crossing filter aggressiveness against data abundance relative to compute: aggressive filtering with abundant data is a win giving the plus seven point five percent boost, aggressive filtering with scarce data is a trap because the tiny set must be re-epoched and its utility decays, loose filtering with abundant data is wasteful because junk is kept when you could afford to cut, and loose filtering with scarce data is a win because diversity beats repetition at high compute](/imgs/blogs/data-quality-scaling-laws-7.png)

Read the four quadrants as a recipe. **Aggressive filter, data abundant relative to compute (top-left, WIN):** you have far more high-quality tokens than you can train on, so you see them roughly once and waste no compute on noise — this is Goyal's +7.5% regime, and you should filter hard. **Aggressive filter, data scarce relative to compute (top-right, TRAP):** you do not have enough high-quality tokens, so you are forced to re-epoch the tiny set, its utility decays past the 4-to-16-epoch horizon, and you would have been better off keeping more data. **Loose filter, data abundant (bottom-left, WASTEFUL):** you are leaving junk in the corpus when you could afford to cut it and still fill your budget with clean tokens — a missed opportunity, though less catastrophic than the trap. **Loose filter, data scarce (bottom-right, WIN):** at high compute relative to data, diversity beats repetition, unfiltered Common Crawl overtakes the filtered set, and loosening is correct.

The asymmetry is the practical takeaway: the top-right TRAP is the dangerous quadrant. Over-filtering when you are compute-rich relative to your data is the mistake that the naive "always keep the top 10%" rule walks you straight into, and it is exactly what happened in Goyal's high-compute experiments where the filtered set lost to unfiltered data. If you are unsure which regime you are in, lean toward less aggressive filtering, because under-filtering wastes a little compute on noise while over-filtering forces destructive repetition.

### A concrete decision procedure

Here is how to actually pick a filter strictness for a planned run, as a checklist:

1. **Estimate your training token budget** $D_{\text{train}}$ from your compute and Chinchilla allocation: $D_{\text{train}} \approx C / (6N)$.
2. **Estimate available unique tokens at each strictness.** A top-10% filter on a 100B crawl gives ~10B; top-30% gives ~30B; unfiltered gives 100B.
3. **Compute the implied epoch count** $E = D_{\text{train}} / D_{\text{unique}}$ for each strictness.
4. **Apply the epoch rule.** If $E \le 4$, the repetition is essentially free and you can afford that strictness. If $4 < E \le 16$, you are paying for repetition but may still come out ahead if the quality lift is large. If $E > 16$, the utility has collapsed — this strictness is too aggressive for your budget; loosen it.
5. **Pick the strictest filter that keeps $E \le 4$** (or modestly above, if the quality lift justifies it). This puts you in the top-left WIN quadrant by construction.

Worked example: you plan a 7B model on $C = 2.5 \times 10^{23}$ FLOPs, so $D_{\text{train}} \approx C/(6N) = 2.5\times10^{23} / (6 \times 7\times10^9) \approx 6\text{T}$ tokens. Your crawl yields 100B tokens raw. Top-10% gives 10B unique, implying $E = 600$ epochs — catastrophically over the 16-epoch wall, deep in the TRAP quadrant. Even unfiltered (100B unique) implies $E = 60$ epochs, still past the wall. The honest conclusion: your *crawl is too small for your compute*, and the right move is not to filter harder but to **get more data** — crawl more, or accept that 6T tokens of training will require expanding the pool well beyond 100B unique. Filtering cannot manufacture unique tokens; at high compute the binding constraint is the size of the clean pool, which is precisely why the high-compute regime favors keeping everything.

Flip the example: a 1B model on $C = 1.2\times10^{20}$ FLOPs gives $D_{\text{train}} \approx 1.2\times10^{20}/(6\times10^9) = 20\text{B}$ tokens. Now top-10% (10B unique) implies $E = 2$ epochs — comfortably inside the free zone — and you should filter hard, banking the +7.5%-style quality lift. Same crawl, same filter, opposite decision, purely because the compute changed. That is the quality-quantity tradeoff in one paragraph.

## 6. Putting it on the curve: data quality in the loss law

**Senior rule of thumb: model data quality as a shift along the effective-data axis, and you can budget it with the same loss law you already use for tokens.** We can now make the "multiplicative compute factor" claim precise enough to plug into the Chinchilla machinery from the [compute-optimal post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling).

Start from the parametric loss law $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$. Suppose curation makes each token worth a factor $q > 1$ more in effective signal, so the effective data is $D_{\text{eff}} = q \cdot D$. The data penalty becomes $B / (qD)^{\beta} = q^{-\beta} \cdot B/D^{\beta}$. To get the same loss with raw data that you get with curated data, you would need to multiply your raw token count — and hence your compute, since $C \approx 6ND$ — by $q$. So a quality multiplier $q$ on the data axis is, to first order, a multiplicative factor $q$ on compute. The DCLM result of 6.6x less compute corresponds to an effective quality multiplier of roughly $q \approx 6.6$ on that benchmark.

This is a first-order picture, and the honest caveat is the same one [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) raises: the multiplier is not constant. It depends on where you are on the curve, because quality and repetition interact. At low compute (each token seen once), curation's quality lift is fully realized and $q$ is large. At high compute (curated set repeated many times), the utility half-life eats into the lift and the *effective* $q$ shrinks — which is just the QQT crossover restated in loss-law terms. So the clean statement is: **data quality is a multiplicative compute factor whose magnitude is largest at low compute and decays toward 1 (or below) as you over-repeat the curated set.**

There is a precision-adjacent corollary worth flagging for anyone who also reads the [precision scaling laws](/blog/machine-learning/scaling-laws/precision-scaling-laws) post: just as over-training a model makes it more fragile under post-training quantization, over-repeating a small high-quality corpus makes the marginal token actively harmful. Both are cases where "more of a good thing" crosses a threshold and starts to hurt, and both are invisible if you only look at the headline scalar (token count, or training data volume) without modeling the interaction. The general lesson across the scaling-laws series is that the simple laws are first-order approximations, and the second-order interactions — repetition decay, quantization fragility, compute-dependent filtering — are where the real engineering judgment lives.

### Folding quality into the data-constrained law

We can be more precise by borrowing the effective-token machinery from the [data-constrained scaling law](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws). Muennighoff et al. replace the raw token count $D$ with an *effective* count $D'$ that accounts for the diminishing value of repeated passes:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{(D')^{\beta}}, \qquad D' = U_D + U_D \cdot R_D^{*}\left(1 - e^{-R_D / R_D^{*}}\right),$$

where $U_D$ is the number of *unique* tokens, $R_D = \max(D/U_D - 1, 0)$ is the number of repeats beyond the first pass, and $R_D^{*} \approx 15.4$ is a fitted decay constant (the source of the ~16-epoch horizon). The key structural fact is that the repetition term *saturates* at $U_D \cdot R_D^{*}$: no matter how many times you re-read your corpus, the effective token count cannot exceed $U_D (1 + R_D^{*}) \approx 16.4 \, U_D$. There is a hard ceiling on how far repetition can carry a fixed unique pool.

Now layer data quality on top. Curation does two things to this equation. First, it raises the *per-token quality*, which we can model as a multiplier $q$ on the effective count: $D'_{\text{curated}} = q \cdot D'$. Second — and this is the QQT interaction — aggressive filtering *shrinks* $U_D$, which lowers the saturation ceiling. So filtering harder raises $q$ but lowers the cap $U_D(1 + R_D^{*})$. At low compute you are nowhere near the cap, so the $q$ gain dominates and filtering wins. At high compute you slam into the (now lower) cap, the saturation term flatlines, and the larger-$U_D$ unfiltered corpus — with its higher ceiling — overtakes you. The crossover in the figure is precisely the compute at which the filtered corpus hits its saturation ceiling. This is the same story the half-life told, now expressed in the data-constrained law's own variables, and it shows why the two framings (utility half-life, effective-token saturation) are the same phenomenon.

A concrete instantiation: take $U_D = 10\text{B}$ for a top-10% filter with quality multiplier $q = 1.5$, versus $U_D = 100\text{B}$ unfiltered with $q = 1.0$. The filtered corpus saturates at effective $1.5 \times 16.4 \times 10\text{B} = 246\text{B}$; the unfiltered one saturates at $1.0 \times 16.4 \times 100\text{B} = 1{,}640\text{B}$. Below ~250B of effective compute-driven tokens, the filtered corpus is climbing toward its higher *initial slope* (the $q = 1.5$ advantage) and wins; above it, the filtered corpus is flat at its ceiling while the unfiltered one is still climbing toward a ceiling almost 7x higher, and it wins. The exact numbers are illustrative, but the structure — two saturating curves with different slopes and different ceilings, crossing once — is exactly the QQT figure, derived from the data-constrained law rather than asserted.

### A comparison table of the levers

To make the relative magnitudes concrete, here is how the main data-quality levers stack up, with the numbers from the research collected in one place.

| Lever | Mechanism | Reported effect | Compute-dependent? |
|---|---|---|---|
| Exact-substring dedup | Suffix array, remove repeated 50-token spans | ~10x less memorization, fewer steps to target perplexity | No — always helps |
| MinHash NearDup | LSH on n-gram shingles, drop near-dup docs | Fixes >4% train-test overlap, same/better perplexity | No — always helps |
| SemDeDup | Embed, k-means, cosine-threshold removal | ~50% of LAION removed, ~2x faster | Mildly — removal rate can be tuned |
| D4 | SemDeDup + density-based diversification | ~20% efficiency, +2% avg over 16 tasks at 6.7B | Mildly — longer half-life helps at scale |
| Heuristic filters | Gopher/C4 rules (length, symbols, punctuation) | Removes boilerplate; foundation of RefinedWeb | No — cheap and safe |
| Model-based filter | Classifier scores doc quality, keep top X% | DCLM ~6.6x less compute; FineWeb-Edu MMLU/ARC gains | **Yes — this is the QQT lever** |
| Per-dump vs global dedup | Dedup within snapshot, not across | Per-dump beat global (global upsamples junk) | Yes — interacts with quality distribution |
| Quality/quantity filter strictness | Set top-X% threshold | +7.5% at low compute; *loses* at high compute | **Yes — the crossover** |

The pattern is clear: dedup methods are unconditionally good (no compute-dependence, just remove the redundant tokens), while *filtering* strictness is the lever that the quality-quantity tradeoff governs. That is the cleanest practical division of the field: dedup first, always; filter second, by compute.

## 7. The timeline: three years of data-curation scaling

The progression from "dedup as hygiene" to "filtering as a compute-aware scaling axis" happened fast, and seeing it laid out chronologically makes the conceptual arc obvious. The timeline below marks the key results.

![A horizontal timeline of data-curation results from 2021 to 2024: Lee et al. deduplication in 2021 giving about ten times less memorization, SemDeDup in 2023 removing fifty percent for two times speedup, RefinedWeb in 2023 showing filtered web beats curated corpora, D4 in 2023 combining dedup and diversification for a two percent average gain, Goyal's quality-quantity tradeoff in 2024 establishing that filtering must depend on compute, and DCLM with FineWeb-Edu in 2024 reaching six point six times less compute through model-based filtering](/imgs/blogs/data-quality-scaling-laws-8.png)

The shape of the arc is worth narrating. **2021 (Lee et al.):** deduplication arrives as a memorization-and-contamination fix, with the efficiency win almost incidental. **2023 (SemDeDup, RefinedWeb, D4):** the field generalizes — dedup goes semantic, filtered web is shown to beat curated mixtures, and diversification is added to dedup. This is the year data quality became a respectable research axis rather than a preprocessing footnote. **2024 (Goyal QQT, DCLM, FineWeb-Edu):** the axis gets its scaling law. Goyal shows the optimum is compute-dependent; DCLM measures the compute factor at 6.6x under controlled conditions; FineWeb-Edu shows targeted filtering produces outsized gains on the benchmarks that matter. In three years, "clean your data" became "data quality is a scaling axis with a compute-dependent optimum, worth more than most architecture changes."

## Case studies from production

The principles above are easiest to internalize through specific incidents — the symptom, the wrong first guess, the actual cause, the fix, and the lesson. These are drawn from the published results and the kinds of failures teams hit when they apply (or ignore) them.

### 1. The 61-word string that ate C4

**Symptom:** A model trained on C4 would, given an innocuous prompt, occasionally emit a specific 61-word block of text verbatim, and its benchmark scores looked suspiciously strong on certain tasks. **Wrong first hypothesis:** the model had simply memorized a common phrase; nothing unusual. **Actual cause:** that exact 61-word string appeared in C4 more than 60,000 times — it was duplicated boilerplate that the cleaning pipeline had not caught — so the model saw it tens of thousands of times and learned it as a near-deterministic output. The strong benchmark scores were partly the >4% train-test overlap leaking test items into training. **Fix:** ExactSubstr deduplication removed the repeated spans, and decontamination removed the test overlap; the model's verbatim regurgitation dropped roughly 10x and the benchmark numbers became honest (and, on the contaminated tasks, lower). **Lesson:** corpora everyone trusts contain pathological duplicates, and they inflate both memorization and eval scores. Dedup before you measure anything.

### 2. The top-10% filter that lost a large run

**Symptom:** A team validated a top-10% quality filter on a small pilot run (it gave a clear accuracy boost), then scaled up to a much larger compute budget — and the big run *underperformed* the unfiltered baseline. **Wrong first hypothesis:** a bug in the large-scale data pipeline, or a learning-rate misconfiguration at scale. **Actual cause:** the QQT crossover. The pilot was in the low-compute regime where the small high-quality set was seen roughly once (filtering wins); the large run was past the crossover, where the 10% set got over-repeated, its utility decayed, and unfiltered diversity would have won. The filter's value had flipped sign with compute, exactly as Goyal predicts. **Fix:** loosen the filter for the large run (or expand the high-quality pool) so the implied epoch count stayed inside the free zone. **Lesson:** never validate a filter at one compute scale and deploy it at another. Aggressiveness must be re-derived from the compute-to-data ratio.

### 3. Global dedup that upsampled the junk

**Symptom:** A curation team deduplicated their multi-dump Common Crawl corpus globally (across all snapshots), expecting cleaner data and better models — but models trained on the globally-deduped corpus were *worse* than per-dump-deduped baselines. **Wrong first hypothesis:** global dedup removed too much data and the model was now data-starved. **Actual cause:** global dedup preferentially collapsed high-quality content (good articles are mirrored across dumps, so they have many near-duplicates) while leaving per-dump-unique spam intact, shifting the quality distribution toward junk. The token count was fine; the *quality mix* had degraded. **Fix:** switch to per-dump deduplication, deduping within each snapshot but allowing cross-snapshot repeats. **Lesson:** dedup interacts with the quality distribution. Maximally aggressive removal can upsample exactly the content you wanted to remove. This is the QQT logic applied to dedup granularity.

### 4. SemDeDup that broke the Chinchilla allocation

**Symptom:** A team applied SemDeDup to their corpus, removing about 40% of it, then trained at their originally-planned token budget — and the model came out smaller-than-intended in effective capability. **Wrong first hypothesis:** SemDeDup removed useful data and hurt quality. **Actual cause:** removing 40% of the corpus shrank the unique-token pool, which pushed the run from ~1 epoch to ~1.7 epochs and, more importantly, meant the [compute-optimal model size](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) for the new (smaller) corpus was different from the one they had provisioned. They had not re-run the allocation after dedup. **Fix:** re-derive $N_{\text{opt}}$ and the epoch count from the post-dedup token count; the run was still net-positive because it stayed well inside the 4-epoch free zone, but the model size needed adjusting. **Lesson:** dedup moves you along the data axis. Always re-run the Chinchilla allocation after a curation step that changes the token count.

### 5. The filter that learned to recognize the benchmark

**Symptom:** A model-based quality filter produced a corpus that gave a large MMLU jump, but the same model's real-world performance on novel questions did not improve proportionally. **Wrong first hypothesis:** MMLU is a good proxy and the model genuinely got much smarter. **Actual cause:** the quality classifier had been trained on examples that resembled MMLU-style multiple-choice text, so it preferentially kept MMLU-like documents — partially contaminating the eval. The MMLU gain was inflated by the filter targeting the benchmark's distribution. **Fix:** decontaminate the corpus against the benchmark, retrain the filter on quality signals uncorrelated with the eval format, and validate on held-out tasks. **Lesson:** a filter powerful enough to lift a metric is powerful enough to game it. Treat large single-benchmark gains from a new filter as a contamination hypothesis until proven otherwise.

### 6. FineWeb-Edu's outsized gains from a narrow filter

**Symptom (a positive one):** Filtering 15T tokens of FineWeb down to 1.3T "educational" tokens produced MMLU and ARC gains far larger than the 8.6x reduction in token count would suggest. **The puzzle:** how does *less* data give *more* capability on knowledge tasks? **Actual cause:** the educational filter concentrated exactly the kind of clear, explanatory, factual text that knowledge-and-reasoning benchmarks reward, while discarding the vast low-signal remainder. The effective quality multiplier $q$ on the relevant subspace of capabilities was very high, even though the total token count dropped. **The nuance:** this works *because* 1.3T tokens is still plenty to stay inside the free-repetition zone for the target model size — push the filter harder, to 130B tokens, and you would hit the QQT trap. **Lesson:** targeted filtering can produce enormous gains on specific capabilities, but only while the filtered pool stays large enough to avoid over-repetition. The same filter that wins at 1.3T would lose at 130B.

### 7. The pilot that under-filtered and wasted compute

**Symptom:** A team, having been burned by the QQT crossover before (case 2), over-corrected and ran a large model on almost-unfiltered Common Crawl — and it underperformed a competitor's filtered run at the same compute. **Wrong first hypothesis:** the competitor had a secret architecture advantage. **Actual cause:** they were in the bottom-left WASTEFUL quadrant. They had plenty of compute-to-data headroom (a large crawl, a moderate model) and could have afforded aggressive filtering while still seeing each clean token roughly once — but they left the junk in and spent compute learning from it. The crossover protects you from over-filtering at high compute, not from under-filtering at low compute. **Fix:** apply a moderate model-based filter sized so the clean pool still implied ≤4 epochs. **Lesson:** the QQT cuts both ways. Avoiding the high-compute trap does not mean never filtering; it means filtering as hard as your compute-to-data ratio allows.

### 8. Code data as a free effective-token multiplier

**Symptom (an opportunity):** A team training a natural-language model was data-constrained — not enough unique high-quality NL tokens for their compute budget, staring down a high epoch count. **The move:** mix in code. The [data-constrained scaling work](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) found that adding code to a text corpus acts like roughly a 2x effective-token multiplier even for natural-language evaluation, and up to 50% code showed no NL deterioration. **Actual mechanism:** code is a large, diverse, high-quality source with a long utility half-life, so it extends the effective unique-token pool and pushes the QQT crossover to the right — letting the team train longer before repetition decay bites. **Fix:** blend up to ~50% code into the mixture, expanding the effective pool. **Lesson:** when you are data-constrained relative to compute, expanding the pool with a diverse high-quality source (code, multilingual, domain text) is often better than filtering harder — it lengthens the effective half-life rather than shortening it.

### 9. The decontamination step that "lost" benchmark points

**Symptom:** After adding aggressive train-test decontamination to a curation pipeline, a model's headline benchmark scores *dropped* by a few points, and stakeholders worried the pipeline change had hurt the model. **Wrong first hypothesis:** decontamination removed useful training data and degraded the model. **Actual cause:** the previous scores had been inflated by test-set leakage (the >4%-overlap problem from case 1). Removing the leaked test items lowered the reported numbers because those numbers had been partly measuring memorization of the test set, not generalization. The model was no worse; the measurement got honest. **Fix:** none needed for the model — but communicate clearly that the lower numbers are the *correct* ones and the previous numbers were contaminated. **Lesson:** a curation improvement that lowers your benchmark score may be exposing prior contamination rather than hurting the model. Always pair dedup with decontamination, and expect honest numbers to sometimes be lower numbers.

### 10. The semantic duplicates that exact dedup couldn't see

**Symptom:** A team ran thorough ExactSubstr and MinHash dedup on a product-review corpus, confirmed near-zero byte-level duplication, and still found the model producing generic, repetitive review text and showing little gain from the last several hundred billion tokens. **Wrong first hypothesis:** the corpus was clean (the dedup metrics said so) and the model had simply saturated. **Actual cause:** the corpus was full of *semantic* duplicates — tens of thousands of reviews saying "great product, fast shipping, would buy again" in slightly different words — which n-gram dedup cannot catch because no long span is byte-identical. The model was effectively over-repeating the same few semantic templates. **Fix:** SemDeDup in embedding space removed about half the corpus as semantic near-duplicates, and training speed roughly doubled with no quality loss. **Lesson:** byte-level dedup metrics reading "clean" does not mean the corpus is diverse. Semantic redundancy is invisible to n-gram methods and requires embedding-space dedup to find and remove.

### 11. The over-sanitized corpus that failed on real users

**Symptom:** A team trained on a corpus aggressively filtered for "high-quality, well-written" text and beat the baseline on every academic benchmark — then deployed the model and watched it stumble on the messy, informal, typo-laden prompts real users actually sent. **Wrong first hypothesis:** the model needed more instruction tuning or a better system prompt. **Actual cause:** the quality filter had been *correct* about quality but catastrophic for diversity. By keeping only clean, textbook-like prose, it stripped out the conversational, ungrammatical, code-switched, long-tail text that teaches a model robustness to real-world input distribution. The model had never seen text that looked like how people actually type. **Fix:** re-filter with a diversity cap — keep high-quality text but limit how much of any one stylistic cluster survives, preserving the messy long tail. **Lesson:** quality and diversity are orthogonal axes. A filter that maximizes quality can destroy diversity, and a model trained on an over-sanitized corpus is brittle exactly where it meets reality. Filter for quality, but protect the spread.

### 12. The filter validated at the wrong scale, in reverse

**Symptom:** A small startup, reading the DCLM 6.6x result, applied the most aggressive published filter to their modest corpus and small compute budget — and got *worse* results than a light filter, the opposite of what they expected from "filtering is good." **Wrong first hypothesis:** they had a bug in the filtering code or the filter did not transfer to their domain. **Actual cause:** the aggressive DCLM filter was tuned for a setting with an enormous crawl and large compute, where you can afford to throw away most of the data and still have plenty of unique tokens. The startup had a small crawl; the aggressive filter cut it down so far that they were forced into many epochs over a tiny set — the QQT trap — while the light filter kept enough unique tokens to stay near one epoch. **Fix:** scale the filter strictness down to match their corpus size, keeping the implied epoch count near 4. **Lesson:** published filters come tuned for the publisher's compute-to-data ratio, not yours. Copying a strictness threshold without recomputing it for your own crawl and budget is the same mistake as case 2, just in the other direction — and it bites small teams hardest, because their crawl-to-compute ratio is least like a frontier lab's.

## What this means in practice

Strip away the papers and here is the operating procedure for treating data quality as a scaling axis.

**Deduplicate first, always, and aggressively.** Dedup has no compute-dependent crossover — it only removes tokens that were not adding information. Run MinHash NearDup at the document level and ExactSubstr at the span level; add SemDeDup in embedding space if you can afford the embedding pass, because byte-level dedup misses the semantic duplicates that hurt most. Expect ~10x less memorization, honest eval numbers, and the same or better perplexity in fewer steps. The only thing to remember is to re-run your [Chinchilla allocation](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) afterward, because dedup changes your token count.

**Filter second, and set strictness from compute, not from a fixed threshold.** This is where the quality-quantity tradeoff lives. Compute your training token budget $D_{\text{train}} \approx C/(6N)$, estimate the unique-token pool at each filter strictness, and pick the strictest filter that keeps the implied epoch count $E = D_{\text{train}}/D_{\text{unique}}$ at or near 4. If even the unfiltered pool forces $E > 16$, your problem is not the filter — your *crawl is too small for your compute*, and you should expand the pool (more crawling, code, multilingual, domain data) rather than filter harder.

**Prefer per-dump dedup over global dedup**, because global dedup upsamples the low-quality remainder. More generally, watch for any curation step that interacts with the quality distribution in a way that backfires — maximally aggressive is rarely optimal.

**Treat data quality as a measured compute factor, not a vibe.** Under controlled conditions (DCLM) it is worth about 6.6x compute — larger than most architecture changes. But the multiplier is largest at low compute and decays as you over-repeat, so the number you can bank depends on your regime. Budget it the way you budget tokens: as a shift along the effective-data axis in the loss law, with the [data-constrained repetition math](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) telling you how much of the lift survives the epochs your compute forces.

**Be paranoid about contamination.** A filter powerful enough to lift a benchmark is powerful enough to game it. Decontaminate against your evals, validate on held-out tasks, and treat large single-benchmark gains from a new filter as a contamination hypothesis until proven otherwise. Expect honest numbers to sometimes be lower than contaminated ones.

The throughline of this whole series is that the simple scaling laws are first-order approximations and the engineering judgment lives in the second-order interactions. [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) gives you the parameter-token split; [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) tells you how repetition decays it; [precision scaling](/blog/machine-learning/scaling-laws/precision-scaling-laws) tells you how low-bit training bends the curve; and data-quality scaling tells you that the tokens themselves are a reshapeable distribution whose optimal shape depends on your compute. The unifying move in every case is the same: do not treat the headline scalar as the whole story. A token is not a token. A good token, fed once into a compute-rich run, is worth several noisy ones — and the entire art of data curation is knowing exactly how many.

## Further reading

- Goyal et al. 2024, "Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic" — https://arxiv.org/abs/2404.07177
- Lee et al. 2021, "Deduplicating Training Data Makes Language Models Better" — https://arxiv.org/abs/2107.06499
- Abbas et al. 2023, "SemDeDup: Data-efficient learning at web-scale through semantic deduplication" — https://arxiv.org/abs/2303.09540
- Tirumala et al. 2023, "D4: Improving LLM Pretraining via Document De-Duplication and Diversification" — https://arxiv.org/abs/2308.12284
- Li, Fang et al. 2024, "DataComp-LM: In search of the next generation of training sets for language models" — https://arxiv.org/abs/2406.11794
- Penedo et al. 2023, "The RefinedWeb Dataset for Falcon LLM" — https://arxiv.org/abs/2306.01116
- Penedo et al. 2024, "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" — https://arxiv.org/abs/2406.17557
- Muennighoff et al. 2023, "Scaling Data-Constrained Language Models" — https://arxiv.org/abs/2305.16264

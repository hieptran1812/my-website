---
title: "Deduplication at Scale: The Highest-ROI Step in Data Curation"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Why removing duplicates is the single most cost-effective thing you can do to a training corpus, and exactly how to do it — exact hashing, MinHash+LSH, suffix arrays, and semantic dedup — with the banding math, runnable code, and the failure modes that bite at petabyte scale."
tags:
  - training-data
  - deduplication
  - minhash
  - locality-sensitive-hashing
  - semantic-deduplication
  - data-pipeline
  - llm-pretraining
  - data-quality
  - fineweb
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 27
---

If you can only afford to run one cleaning stage over your training corpus, run deduplication. Not language ID, not the quality classifier, not the PII scrubber — dedup. It is the step with the best ratio of model-quality gained to engineering-hours spent, and it is the one most teams under-invest in because "we already removed exact duplicates" feels like enough. It is not enough. The web is a hall of mirrors: the same article syndicated across a hundred domains, the same Stack Overflow answer scraped into a dozen content farms, the same license header pasted into a million repositories, the same Common Crawl page captured in snapshot after snapshot. Left in, those copies quietly sabotage the model in four different ways at once.

![Why leaving duplicates in the corpus upweights, memorizes, and leaks the same document N times over](/imgs/blogs/deduplication-at-scale-1.webp)

This post is the practitioner's version: what "duplicate" actually means at each granularity, the four families of dedup you should know (exact, near-duplicate via MinHash+LSH, exact-substring via suffix arrays, and semantic via embeddings), the banding math you need to tune locality-sensitive hashing correctly, runnable code, and — because this is where people lose weeks — the failure modes that only show up at scale. It sits downstream of [text extraction](/blog/machine-learning/training-data/sourcing-and-collecting-training-data) and pairs closely with [decontamination](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage), which reuses the exact same overlap-detection machinery for a different purpose.

## Why deduplication is the highest-ROI step

Start with the mechanism, because "duplicates are bad" is not an argument a staff engineer accepts. A document that appears N times in the corpus does four measurable kinds of damage.

**It multiplies memorization.** The probability that a model reproduces a training sequence verbatim scales sharply with how many times that sequence appeared. Carlini and colleagues showed that sequences duplicated even a handful of times become far more extractable than singletons, and the relationship is roughly super-linear: a string seen a thousand times is memorized with near-certainty. Memorized training data is a privacy liability (it can be extracted with the right prompt) and a sign the model is spending capacity on rote copying instead of generalization.

**It leaks eval into train.** Benchmarks live on the web. If your corpus contains many copies of a page, the odds that one of them overlaps a test set climb, and duplicates make the overlap harder to catch because you have to find *all* copies to remove the leak. This is the bridge to [benchmark decontamination](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) — dedup and decontam are the same string-matching problem pointed at two different targets.

**It wastes compute.** Every duplicate token you train on is a token you are not spending on new information. If 15% of your corpus is redundant, you are burning roughly 15% of your pretraining budget re-teaching the model things it already saw — or, at a fixed token budget, you are training on 15% less unique signal than your token count claims.

**It upweights whatever the duplicate says — usually junk.** This is the subtle one. Deduplication is implicitly a *reweighting* of your data distribution. The documents that get duplicated the most are rarely your best content; they are boilerplate, SEO spam, auto-generated pages, and license text. Leaving duplicates in silently upweights exactly the material you would most like to downweight. Dedup is not just hygiene; it is distribution shaping.

Make that concrete. Imagine a 100-billion-token corpus where a particular 500-token SEO template — "Buy cheap X online, best prices, free shipping…" — appears, lightly varied, five million times. That is 2.5 billion tokens, 2.5% of your entire corpus, spent on a single garbage template. At a Chinchilla-style budget that is more tokens than you will spend on all of, say, high-quality mathematics. The model sees that template more often than it sees most real knowledge, so it learns to produce it fluently — capacity and compute both burned on the worst content you have, purely because it was duplicated. Deduplication is the only stage that fixes this: it converts "seen five million times" back into "seen once," restoring the natural frequency the template deserves. Every other cleaning stage either keeps a document or drops it; dedup is the one that fixes *how many times* the survivors are counted.

The empirical payoff matches the theory. When Lee et al. deduplicated C4 and other corpora in *Deduplicating Training Data Makes Language Models Better*, they found train sets riddled with near-duplicates (one 61-word sequence appeared over 60,000 times in C4), models emitted up to 10× less memorized text after dedup, and — critically — validation perplexity improved or held while training on *less* data. That is the definition of a free lunch: smaller corpus, faster training, better model.

## Exact deduplication: the cheap floor

Always do exact dedup first. It is O(n), embarrassingly parallel, and has zero false positives, so it costs almost nothing and never removes something it shouldn't. There are two granularities.

**Document-level exact dedup** hashes the normalized bytes of each document and keeps one representative per hash.

```python
import hashlib

def doc_key(text: str) -> str:
    # Normalize lightly so trivially-different copies collide:
    # lowercase, collapse whitespace. Do NOT strip punctuation here —
    # that is a near-dup concern, not exact.
    norm = " ".join(text.lower().split())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()

seen = set()
def keep_exact(text: str) -> bool:
    k = doc_key(text)
    if k in seen:
        return False
    seen.add(k)
    return True
```

The `seen` set is the scaling problem: at a trillion documents you cannot hold every 16-byte key in RAM on one box. In practice you either shard by hash prefix across workers (all keys with prefix `0x00…` go to worker 0, etc., so each worker's set is 1/K the size) or use a disk-backed key-value store. A Bloom filter is tempting but introduces false positives — i.e., it will occasionally *drop a unique document* — so reserve it for cases where losing a tiny fraction is acceptable.

How hard you normalize before hashing is itself a design choice that quietly sets the boundary between "exact" and "near" dedup. Lowercasing and whitespace-collapsing (as above) makes trivially-reformatted copies collide, which is almost always what you want. Go further — stripping punctuation, normalizing Unicode, removing HTML entities — and you catch more copies but start merging documents that differ in meaningful ways (code where punctuation is semantic, or two prices that differ only by a digit). The rule of thumb: normalize just enough that *presentation* differences collide but *content* differences do not, and push everything fuzzier than that into the MinHash stage where you have a tunable threshold instead of a hard hash equality. Getting this boundary wrong is a common source of "why did exact dedup remove that?" surprises.

**Line- and paragraph-level exact dedup** targets a different beast: shared blocks *inside* otherwise-distinct documents. Navigation menus, cookie banners, and boilerplate footers survive document-level dedup because the surrounding article differs, but they are pure repetition. C4's original recipe removed any three-sentence span that appeared more than once across the corpus. This is powerful and dangerous: it will happily delete a legitimately common sentence ("Thanks for reading!") from every document that contains it. Do line-level dedup *after* good [boilerplate extraction](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal), and prefer removing whole repeated blocks over individual repeated lines.

Exact dedup catches byte-identical copies. It does nothing for the far more common case: documents that are 98% identical but differ in a timestamp, an ad slot, or one edited word. For that you need near-duplicate detection.

## Near-duplicate detection: shingling to MinHash

The near-dup question is "are these two documents *similar enough*?", and the standard similarity measure is **Jaccard**: represent each document as a set of overlapping k-word shingles, and compute the ratio of shared shingles to total distinct shingles.

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Two documents that differ by one word out of a hundred share almost all their shingles, so their Jaccard is near 1. The problem is that computing exact Jaccard between every pair of documents is O(n²) pairwise and each document's shingle set is huge. MinHash solves the first half; LSH (next section) solves the second.

![Shingling to MinHash: signatures agree at a fraction of positions equal in expectation to the Jaccard overlap](/imgs/blogs/deduplication-at-scale-2.webp)

**MinHash** compresses a document's shingle set into a short fixed-length signature with a magic property: the probability that two documents' signatures agree at a given position equals their Jaccard similarity. Concretely, pick `num_perm` independent hash functions; for each, the signature entry is the *minimum* hash value over all the document's shingles. Because the minimum is equally likely to come from any shingle, two sets agree on that minimum exactly when the min-hashing shingle is in their intersection — which happens with probability J(A,B). Average over `num_perm` positions and you get an unbiased Jaccard estimate whose error shrinks like 1/√(num_perm). The figure above walks a tiny example: two sentences differing by one word produce five-entry signatures that match in three positions, estimating a Jaccard of 0.60 — exactly the true value on that toy set.

```python
from datasketch import MinHash

def shingles(text: str, k: int = 5):
    words = text.lower().split()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, k=5):
        m.update(sh.encode("utf-8"))
    return m

a = minhash("the quick brown fox jumps over the lazy dog every morning")
b = minhash("the quick brown fox jumps over the lazy dog each morning")
print(a.jaccard(b))   # ~0.8, estimated from 128-position signatures
```

Two knobs matter. `k` (shingle size) sets the granularity: small k (3) catches short shared phrases and is noisier; large k (7–10) demands longer verbatim runs to register as similar. Five-word word-shingles are a reasonable default for prose. `num_perm` trades accuracy for cost: 128 permutations is standard, 256 for tighter estimates, and the signature is what you store per document, so it also sets your memory footprint.

## LSH banding: finding candidates without O(n²)

MinHash gives you cheap pairwise similarity *estimates*, but you still cannot afford to compare all n² pairs. **Locality-sensitive hashing** turns similarity search into a hash-bucket lookup: it hashes similar documents into the same bucket with high probability and dissimilar ones into the same bucket with low probability, so you only ever compare documents that already collided.

![LSH banding: split the signature into b bands of r rows and bucket each band separately](/imgs/blogs/deduplication-at-scale-3.webp)

The trick is **banding**. Split each `num_perm`-length signature into `b` bands of `r` rows each (so `num_perm = b × r`). Hash each band independently into its own bucket table. Two documents become a *candidate pair* if they collide in **at least one** band. Because a whole band must match — all r of its rows identical — a single band matching is strong evidence, and requiring only *one* of b bands to match gives you many independent chances to catch a true near-duplicate.

<figure class="blog-anim">
<svg viewBox="0 0 760 420" role="img" aria-label="Four documents slide from a left rail into LSH buckets; two near-duplicates land in the same bucket, which lights up and is flagged as a candidate pair" style="width:100%;height:auto;max-width:820px">
<style>
.dz-bucket{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.dz-doc{stroke:var(--border,#d1d5db);stroke-width:1.5}
.dz-neu{fill:var(--surface,#f3f4f6)}
.dz-dup{fill:var(--accent,#6366f1)}
.dz-lblN{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.dz-lblD{font:600 15px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.dz-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.dz-hi{fill:var(--accent,#6366f1);stroke:var(--accent,#6366f1);stroke-width:2}
.dz-tag{font:700 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
@keyframes dz-flow{0%{transform:translate(0,0);opacity:0}8%{opacity:1}45%{transform:translate(var(--dx),var(--dy))}90%{transform:translate(var(--dx),var(--dy));opacity:1}100%{transform:translate(var(--dx),var(--dy));opacity:0}}
@keyframes dz-lite{0%,45%{opacity:0}55%,90%{opacity:1}100%{opacity:0}}
.dz-flow{animation:dz-flow 9s ease-in-out infinite}
.dz-lite{animation:dz-lite 9s ease-in-out infinite;opacity:0}
@media (prefers-reduced-motion:reduce){.dz-flow{animation:none;transform:translate(var(--dx),var(--dy));opacity:1}.dz-lite{animation:none;opacity:1}}
</style>
<text class="dz-cap" x="105" y="20">documents (MinHash-banded)</text>
<text class="dz-cap" x="625" y="20">LSH buckets</text>
<rect class="dz-bucket" x="520" y="30" width="210" height="90" rx="10"/>
<rect class="dz-bucket" x="520" y="170" width="210" height="110" rx="10"/>
<rect class="dz-bucket" x="520" y="320" width="210" height="90" rx="10"/>
<text class="dz-cap" x="625" y="52">bucket A</text>
<text class="dz-cap" x="625" y="162">bucket B</text>
<text class="dz-cap" x="625" y="342">bucket C</text>
<rect class="dz-hi dz-lite" x="514" y="164" width="222" height="122" rx="12" fill-opacity="0.18"/>
<text class="dz-tag dz-lite" x="625" y="150">collision, candidate pair</text>
<g class="dz-flow" style="--dx:512px;--dy:22px">
<rect class="dz-doc dz-neu" x="24" y="44" width="150" height="44" rx="8"/>
<text class="dz-lblN" x="99" y="72">D1</text>
</g>
<g class="dz-flow" style="--dx:512px;--dy:80px">
<rect class="dz-doc dz-dup" x="24" y="120" width="150" height="44" rx="8"/>
<text class="dz-lblD" x="99" y="148">D2</text>
</g>
<g class="dz-flow" style="--dx:512px;--dy:52px">
<rect class="dz-doc dz-dup" x="24" y="196" width="150" height="44" rx="8"/>
<text class="dz-lblD" x="99" y="224">D3</text>
</g>
<g class="dz-flow" style="--dx:512px;--dy:94px">
<rect class="dz-doc dz-neu" x="24" y="272" width="150" height="44" rx="8"/>
<text class="dz-lblN" x="99" y="300">D4</text>
</g>
</svg>
<figcaption>Each document is placed by its band hashes; D2 and D3 share a band, so they land in the same bucket, which lights up and is emitted as a candidate pair. D1 and D4 land alone and are never compared.</figcaption>
</figure>

The probability that two documents with true Jaccard `s` become a candidate pair has a clean closed form:

$$P(\text{candidate}) = 1 - (1 - s^{\,r})^{b}$$

Read it piece by piece: `s^r` is the chance all r rows of one band match; `(1 - s^r)` is the chance a given band does *not* match; raised to the `b` is the chance *no* band matches; one minus that is the chance at least one band matches. This function is an **S-curve** in `s`: near-flat for low similarity, a steep rise through a threshold, and near-flat at 1 for high similarity. The location of the steep part is what you tune.

## Worked scenario: choosing b and r for a target threshold

Say you want to flag pairs with Jaccard ≥ 0.8 and ignore the rest, using `num_perm = 128`. The knee of the S-curve — the similarity where candidacy probability crosses one-half — is approximated by:

$$t \approx \left(\tfrac{1}{b}\right)^{1/r}$$

Push `r` up (more rows per band) and the band-match test gets stricter, moving the knee *right* (higher threshold, fewer candidates). Push `b` up (more bands) and you get more chances to match, moving the knee *left*. They fight, constrained by `b × r = num_perm`. Let's tabulate a few splits of 128:

![The banding S-curve puts a sharp probability knee at the target Jaccard threshold, tuned by b and r](/imgs/blogs/deduplication-at-scale-4.webp)

| b (bands) | r (rows) | approx knee `(1/b)^(1/r)` | behavior |
| --- | --- | --- | --- |
| 32 | 4 | ~0.42 | knee far left — floods you with candidates, high recall, expensive verify |
| 16 | 8 | ~0.71 | balanced |
| 11 | 11 | ~0.80 | knee near your nominal target |
| 9 | 13 | ~0.84 | what datasketch actually picks for threshold 0.8 |
| 8 | 16 | ~0.88 | knee right — misses borderline near-dups, cheap |

You do not have to compute the split by hand — `datasketch` solves the optimization for you. But here is the subtlety that bites in practice, and it is worth internalizing before you trust any LSH library blindly. When you ask for `threshold=0.8` with `num_perm=128`, datasketch does **not** hand you the split whose 0.5-crossing sits exactly at 0.8. It minimizes a *weighted* sum of false positives and false negatives across the entire S-curve, and for these inputs it chooses **b = 9, r = 13** — whose actual knee sits around **0.844**, not 0.8. The nominal "threshold" you passed is a target, not a guarantee about where the effective cutoff lands.

The consequence is concrete and easy to miss: a pair with true Jaccard 0.81 — which you *believed* you asked to catch when you typed `0.8` — falls just below the effective knee and is **silently dropped from the candidate set**. It is never even compared, so no log line tells you it slipped through. Whenever a dedup pass leaves behind near-duplicates you know are there, this off-by-a-knee gap is the first thing to check. Here is the pattern end to end, with the real numbers a quick datasketch run produces:

```python
from datasketch import MinHash, MinHashLSH

# LSH picks b, r internally to place the S-curve knee near threshold=0.8
lsh = MinHashLSH(threshold=0.8, num_perm=128)
print(lsh.b, lsh.r)   # -> 9 13   (effective knee ~0.844, NOT 0.80)

docs = {
    "d1": "the quick brown fox jumps over the lazy dog every single morning",
    "d2": "the quick brown fox jumps over the lazy dog every morning today",  # J(d1,d2) ~ 0.945
    "d3": "a completely unrelated sentence about b-tree database indexing",
    "d4": "a completely unrelated sentence about b-tree database indexing",   # exact dup of d3
    "d5": "quarterly revenue rose twelve percent driven by cloud services",
    "d6": "the quick brown fox jumps over the very lazy dog each morning",    # J(d1,d6) ~ 0.81
}

sigs = {}
for name, text in docs.items():
    m = MinHash(num_perm=128)
    for sh in shingles(text, k=5):
        m.update(sh.encode("utf-8"))
    sigs[name] = m
    lsh.insert(name, m)

# Union-Find over candidate pairs -> duplicate clusters, keep 1 representative each
kept, dropped = [], []
for name in docs:
    dups = set(lsh.query(sigs[name])) - {name}
    if any(d in kept for d in dups):
        dropped.append(name)      # a cluster-mate is already kept
    else:
        kept.append(name)

print("kept:   ", kept)     # ['d1', 'd3', 'd5', 'd6']
print("dropped:", dropped)  # ['d2', 'd4']
```

Look at what survived. `d2` (Jaccard 0.945 with `d1`) and the exact copy `d4` are correctly dropped. But `d6` — a genuine near-duplicate of `d1` at Jaccard ~0.81 — **is kept**, because 0.81 sits below the effective knee (~0.844) that datasketch's `b=9, r=13` actually implements. That is the missed-pair failure in the flesh: you asked for 0.8, the library gave you ~0.844, and everything in the 0.80–0.84 band leaks through. If catching those matters, lower the nominal threshold (e.g., request 0.7 to *effectively* cover 0.8), bump `num_perm` so more `b, r` splits are available, or add a cheap exact-Jaccard verify over each bucket's members to recover the borderline pairs the S-curve dropped.

That is the whole near-dup pipeline in miniature: shingle, MinHash, insert into LSH, query for candidates, cluster, keep one representative. At scale you replace the in-memory `MinHashLSH` with a sharded version (each band table partitioned across workers) and the Union-Find with a distributed connected-components pass, but the logic is identical.

**SimHash** is the common alternative to MinHash+LSH, favored by some large crawls (it was Google's original near-dup method). Instead of min-of-permutations it builds a single b-bit fingerprint by summing sign-weighted feature hashes; two documents are near-duplicates if their fingerprints differ in only a few bits (small Hamming distance). It is more compact (one 64-bit integer per document) and the "differ in ≤ k bits" query is cheap, but it estimates cosine-style overlap rather than Jaccard and is a little coarser. For most text corpora MinHash+LSH is the better-understood default; reach for SimHash when signature storage is the binding constraint.

## Exact-substring deduplication with suffix arrays

MinHash asks "are these two *documents* similar?" It is blind to a different, very common duplicate: a long verbatim passage that appears inside many otherwise-different documents — a quoted press release, a boilerplate legal disclaimer, a memorized poem embedded in a hundred distinct blog posts. Document-level Jaccard misses these because the surrounding text differs; you need substring-level exact matching.

The tool is a **suffix array**. Build a suffix array over the concatenation of the entire corpus and you can find, in O(n log n), every substring of at least some length (Lee et al. used 50 tokens) that occurs more than once, then remove all but one occurrence of each. This is the `ExactSubstr` half of *Deduplicating Training Data Makes LMs Better*, and Google's open-source `deduplicate-text-datasets` implements it in Rust because the memory and IO discipline required to build a suffix array over hundreds of gigabytes is not something you want to do in Python.

Substring dedup is precise (it removes only genuinely repeated spans, so false-positive risk is low) and complementary to near-dup dedup — one operates on whole-document similarity, the other on repeated fragments. Serious pipelines run both.

Two practical notes. First, decide what to do with a repeated span: you can delete every copy but one, or delete *all* copies (safer for something like a leaked answer key, riskier because it can leave a document with a hole in it). Most pipelines keep the first occurrence and cut the rest, then drop any document left too short to be useful. Second, the reason this is a Rust tool and not a pandas one-liner is IO discipline: building a suffix array over hundreds of gigabytes means external-memory sorting of billions of suffixes, and a naive in-memory implementation will exhaust RAM on a corpus a thousandth the size of what you actually need to process. Treat substring dedup as a batch job you run once per corpus build, not something you iterate on interactively.

## Semantic deduplication: catching paraphrase

All of the above are *lexical*: they detect shared tokens. They are defeated by paraphrase — two documents that say the same thing in different words share few shingles and hash to different buckets. As synthetic data and machine translation flood the web, semantic near-duplicates matter more.

**SemDeDup** (Abbas et al.) handles them: embed every document with a sentence/text encoder, cluster the embeddings (k-means into many clusters), and within each cluster drop documents whose cosine similarity to a kept neighbor exceeds a threshold. It found large fractions of web and image-text corpora to be *semantically* redundant even after lexical dedup, and removing them preserved or improved quality while shrinking the data.

```python
# Sketch — SemDeDup within one cluster
import numpy as np

def semdedup_cluster(embeds: np.ndarray, ids: list[str], thresh: float = 0.95):
    # embeds: L2-normalized, shape [n, d]; keep a doc unless it is
    # near-identical to one already kept.
    kept_idx = []
    for i in range(len(ids)):
        if not kept_idx:
            kept_idx.append(i); continue
        sims = embeds[i] @ embeds[kept_idx].T   # cosine (normalized)
        if sims.max() < thresh:
            kept_idx.append(i)
    return [ids[i] for i in kept_idx]
```

Why does this matter more every year? Because the fraction of the web that is machine-generated is climbing, and generated text is *lexically* diverse while being *semantically* repetitive — ten LLM rewrites of the same fact share almost no five-word shingles but carry the same information. MinHash sees ten distinct documents; SemDeDup sees one idea ten times. As synthetic data and machine translation fill crawls, lexical dedup increasingly under-counts true redundancy, and semantic dedup is the only tool that closes the gap. The same technique also transfers directly to multimodal data — SemDeDup was in fact demonstrated on image-text corpora like LAION, where captions differ but images are near-identical.

Semantic dedup is the most powerful and the most dangerous of the four. It is expensive (you must embed the whole corpus) and its false-positive risk is high: two genuinely different documents on the same narrow topic can look "semantically identical" to an embedding model and get collapsed, silently destroying real diversity. Use a high similarity threshold, cluster finely, and — as always — validate the effect with an [ablation](/blog/machine-learning/training-data/measuring-data-quality) rather than trusting the dedup rate.

## Four methods, four duplicate shapes

These are not competitors; they are a stack. Each catches a different shape of duplicate at a different cost and false-positive risk, and a mature pipeline layers them cheapest-first: exact document, exact line/block, MinHash+LSH near-dup, suffix-array substring, and (optionally) semantic.

![Each dedup method catches a different duplicate shape at a different cost and false-positive risk; they compose, not compete](/imgs/blogs/deduplication-at-scale-5.webp)

The ordering is deliberate. Exact dedup is free and risk-free, so it runs first and shrinks the input to everything downstream. Near-dup and substring dedup do the heavy lifting on lexical redundancy. Semantic dedup, being both expensive and risky, runs last on an already-cleaned corpus if at all. Run them in the other order and you pay the expensive stages on data the cheap stages would have removed for nothing.

## Dedup scope: from one page to every snapshot

Independent of *method*, you must decide *scope*: what population of documents are you deduplicating against?

![Dedup scope ranges from within-document to within-corpus to global cross-snapshot, changing cost and duplicate definition](/imgs/blogs/deduplication-at-scale-6.webp)

**Within-document** removes repeated lines inside a single page (the boilerplate case). **Within-corpus** — really *within one snapshot or crawl* — removes documents that duplicate each other in that batch. **Global** deduplicates across the entire dataset at once, including across all Common Crawl snapshots, so a page captured in the January and February crawls is recognized as the same page.

Global sounds strictly better — why keep the February copy of a January page? — but it is where the most expensive and most surprising failure in this whole space lives.

## Case study: FineWeb and the global-vs-local surprise

The instinct that "more dedup is always better" is wrong, and the FineWeb team demonstrated it convincingly. Building a 15-trillion-token corpus from roughly one hundred Common Crawl snapshots, they first did the obvious thing: MinHash-dedup **globally** across all snapshots at once. The result was a corpus that performed *worse* on their ablation benchmarks than a version deduplicated **per-snapshot** (locally), despite the global version being more aggressively deduplicated.

The mechanism is worth internalizing. Global dedup keeps one copy of each repeated page and throws away the rest — but the copies it throws away are concentrated in the *older* snapshots (a page first seen in 2014 and re-crawled every year loses all but one instance). What survives global dedup in the old snapshots is the *residue*: the pages that were unique to that crawl, which skew toward lower-quality, un-syndicated, long-tail content. Meanwhile the frequently-recrawled pages — often the *higher*-quality, widely-referenced ones — are the very pages global dedup thins out most. So aggressive global dedup can invert the quality gradient of your corpus, upweighting the dregs. FineWeb's fix was to deduplicate **within each snapshot independently** and largely skip cross-snapshot dedup, which kept the good repeated pages appearing at a natural frequency and produced a materially better model.

The lesson generalizes past FineWeb: deduplication changes your data distribution, and "remove more duplicates" is not the same as "improve the distribution." Which duplicates you remove — and from which part of the corpus — determines whether dedup helps or hurts. The only way to know is to ablate the choice, not to assume.

This complements the earlier lexical-dedup case studies. Lee et al. established that near-dup and substring dedup *reduce memorization and leakage and improve perplexity*; RefinedWeb and C4 showed dedup is one of the biggest single levers on downstream quality; FineWeb added the crucial caveat that scope and aggressiveness are tunable knobs with a real optimum, not a "max it out" slider.

## How much do real pipelines actually remove?

The numbers surprise people the first time they see them, so calibrate your expectations. Deduplication is not a 2% trim — it is often the largest single reduction in the whole pipeline.

- **C4** (the T5 corpus): the original construction removed any duplicated three-sentence span across the corpus, and Lee et al. later found that *even after* that, near-duplicate content remained widespread — with individual sequences repeated tens of thousands of times (that 61-word span appearing over 60,000 times).
- **RefinedWeb** (Falcon): the team reported that stringent deduplication — exact plus fuzzy MinHash — removed roughly **half** of what survived their quality filters. Half. Dedup, not quality filtering, was the stage that most shrank the corpus, and RefinedWeb's headline result was that a purely web corpus this aggressively deduplicated could match curated corpora.
- **The Pile and other repositories**: cross-source dedup routinely finds that "distinct" sources overlap heavily (the same book in Books3 and in a web scrape, the same paper in arXiv and in a crawl), so global dedup removes a double-digit percentage of what naive concatenation produced.

The operational lesson: budget for dedup to remove a *large* fraction, plan storage and compute for a pass over the full corpus that emits maybe half of it, and never quote your corpus size until *after* deduplication — a pre-dedup token count is marketing, not signal.

## Engineering it at scale

Everything above is easy on a laptop and hard at a petabyte. The three things that break:

**Memory.** You cannot hold every MinHash signature and every LSH bucket table in one machine's RAM. Shard the LSH band tables by band hash across workers, so each worker owns a slice of the bucket space; documents are routed to workers by their band hashes, collisions are detected locally, and candidate pairs are emitted to a global connected-components job. Signatures themselves are streamed from and to object storage, never fully resident.

Put numbers on it. A 128-permutation MinHash signature stored as 32-bit integers is 512 bytes per document. At one billion documents that is 512 GB of signatures alone — already too large for a single node's RAM, before you add the LSH band tables (which, at b bands, hold roughly b entries per document). This is why signature width is a real cost lever: dropping to `num_perm=64` halves your signature storage at the price of a noisier Jaccard estimate, and using 16-bit hashes instead of 32-bit halves it again. At ten billion documents you are firmly in "shard everything, stream everything, hold nothing globally" territory, and the difference between a dedup job that finishes overnight and one that never finishes is almost always whether the band tables were partitioned correctly across workers.

**Throughput and IO.** Dedup is IO-bound, not compute-bound — you are reading the entire corpus and writing a filtered copy. Store the corpus in a columnar/streamable format (Parquet or WebDataset shards), process shard-parallel with a framework like Spark or Ray Data, and keep the MinHash computation vectorized. The signature step (shingling + hashing) is the CPU hotspot; everything else is moving bytes.

A useful mental model: dedup is a *map-heavy, shuffle-once* job. The map phase (read shard, shingle, MinHash, emit band-keys) is perfectly parallel and scales linearly with workers. The single unavoidable shuffle is grouping documents by band-key so collisions can be detected, and that shuffle is where your cluster spends its network budget — so anything that shrinks the keys (fewer bands, narrower hashes) or pre-removes hot boilerplate (fewer documents per key) pays off twice, once in CPU and once in network. Size the job by the shuffle, not by the map, and you will estimate wall-clock far more accurately than a per-document benchmark suggests.

**Cross-shard coordination.** The one operation that resists sharding is the final "which documents are in the same duplicate cluster" step, because a cluster can span shards. This is a distributed connected-components / Union-Find over the candidate-pair graph. Do it as an explicit graph job (label propagation or iterative Union-Find), keep one representative per component, and — importantly — make the choice of representative *deterministic* (e.g., lowest document ID) so the pipeline is reproducible across reruns. Non-deterministic representative selection makes your corpus un-reproducible and your ablations un-comparable.

A pragmatic default stack for a web-scale text corpus: exact document + line dedup in a first Spark/Ray pass; MinHash (num_perm=128, k=5) + sharded LSH at threshold ~0.8 **per snapshot**; optional suffix-array substring dedup with the Rust tool; and semantic dedup only if you have evidence of paraphrase redundancy and the budget to ablate it.

## Troubleshooting: the failures that bite

![An over-aggressive similarity threshold collapses genuinely distinct documents, deleting real information as false positives](/imgs/blogs/deduplication-at-scale-7.webp)

**Symptom: the corpus shrank far more than expected and eval got worse.** *Cause:* over-aggressive near-dup — threshold too low, so genuinely distinct documents that share boilerplate (nav bars, license text, common headers) get collapsed into one. The figure above shows the classic version: two distinct tutorials that share a navigation menu and an MIT license header score 0.55 Jaccard on their *raw* text and get merged. *Fix:* strip boilerplate **before** dedup so Jaccard is computed on real content, and raise the threshold to ~0.8. Verify by sampling "duplicate" clusters and reading them — if cluster-mates are not actually near-duplicates, your threshold or your extraction is wrong.

**Symptom: dedup rate swings wildly when you tweak parameters.** *Cause:* the S-curve is steep by design, so small changes in `b`, `r`, or the threshold move the knee and change what counts as a duplicate. `num_perm` too low also makes the Jaccard estimate noisy, adding randomness on top. *Fix:* pin `num_perm` at 128–256 for stable estimates, choose `b, r` from the target-threshold math above rather than by trial and error, and treat the threshold as a hyperparameter you ablate once and then freeze.

**Symptom: the LSH job OOMs or a few workers run 100× longer than the rest.** *Cause:* hot buckets — a boilerplate band shared by millions of documents all hash to one bucket, and that worker tries to materialize an enormous candidate set. *Fix:* cap candidate-set size per bucket, remove exact-duplicate boilerplate first (so the hot bucket never forms), and shard by band hash so hot buckets are at least spread across the band tables. Monitor per-worker candidate counts, not just job completion.

**Symptom: dedup results differ between two runs on the same data.** *Cause:* non-deterministic representative selection in the clustering step, or non-deterministic hash seeds. *Fix:* fix all hash seeds, and pick cluster representatives deterministically (lowest ID / earliest crawl). Reproducibility here is not optional — it is what lets you attribute an eval change to a data change instead of to noise.

**Symptom: you deduplicated hard but the model still memorizes and still leaks benchmarks.** *Cause:* lexical dedup does not catch paraphrase or near-identical-with-heavy-edits, and it does not catch benchmark contamination that survives as scattered fragments. *Fix:* add substring dedup for long repeated spans, consider semantic dedup for paraphrase, and run a dedicated [decontamination pass](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) against your eval sets — dedup and decontam are related but not substitutes.

**Symptom: global dedup made the corpus "cleaner" but the model got worse (the FineWeb trap).** *Cause:* cross-snapshot global dedup thinned out frequently-recrawled high-quality pages and left the low-quality residue, inverting your quality gradient. *Fix:* deduplicate within each snapshot independently, be conservative about cross-snapshot dedup, and — the recurring theme — ablate the scope choice on a small proxy model instead of assuming more dedup is better.

## A default recipe you can start from

If you want a starting point to adapt rather than a menu, here is a defensible default for a web-scale text corpus, cheapest stage first:

1. **Exact document dedup** — normalize (lowercase, collapse whitespace), BLAKE2b hash, keep one per hash, sharded by hash prefix. Always on; zero false positives.
2. **Exact line/block dedup** — remove repeated multi-line blocks, but only *after* boilerplate extraction, and prefer whole-block removal over single lines.
3. **Near-duplicate MinHash + LSH** — `num_perm=128`, `k=5` word-shingles, request `threshold≈0.8` (remembering the effective knee lands higher — lower the request if you need to catch the 0.80–0.84 band), run **per snapshot**, sharded band tables, deterministic representative per cluster.
4. **Suffix-array substring dedup** — optional but recommended for prose/code; use the Rust `deduplicate-text-datasets` tool, minimum match length ~50 tokens.
5. **Semantic dedup** — only if you have measured paraphrase redundancy and can afford to embed the corpus; high similarity threshold, fine clustering, and *ablate the effect* before trusting it.

Two rules sit on top of all five. First, **ablate the aggressiveness** — the FineWeb result proves the optimum is not "maximum," so verify each scope/threshold choice on a small proxy model against real evals, exactly as covered in [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality). Second, **make it reproducible** — fixed seeds, deterministic representatives — so that when the model changes, you can tell whether the data or the noise moved.

## The through-line

Deduplication is the highest-ROI curation step because it does four good things at once — less memorization, less leakage, less wasted compute, better distribution — with a well-understood toolkit and modest engineering. But it is not a slider you crank to maximum. The method (exact, near-dup, substring, semantic), the parameters (`k`, `num_perm`, `b`, `r`, threshold), and above all the *scope* (within-document, within-snapshot, global) are choices with a real optimum, and the FineWeb result is the cautionary proof that the naive "dedup everything against everything" can make your model worse. Do exact dedup always, near-dup and substring dedup as the workhorses, semantic dedup with care — and let a small-model [ablation](/blog/machine-learning/training-data/measuring-data-quality), not the dedup rate, tell you when you have it right. Next in the pipeline is making sure none of your eval sets leaked in: [decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage), which reuses everything you just built.

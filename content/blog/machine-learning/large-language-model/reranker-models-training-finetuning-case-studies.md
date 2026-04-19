---
title: "Reranker Models: A Complete Guide to Training, Fine-tuning, and Real-World Case Studies"
publishDate: "2026-04-18"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "reranker",
    "reranking",
    "cross-encoder",
    "RAG",
    "LLM",
    "semantic-search",
    "information-retrieval",
    "listwise-ranking",
    "NLP",
    "AI",
  ]
date: "2026-04-18"
author: "Hiep Tran"
featured: true
aiGenerated: true
excerpt: "A deep dive into reranker models — what they are, why they matter, how cross-encoders differ from bi-encoders, how to train and fine-tune them, and how production teams at Cohere, BAAI, Jina, and Airbnb shipped them. Covers pointwise/pairwise/listwise objectives, distillation, hard-negative mining, latency budgets, LLM-as-reranker, and concrete recipes."
---

# Reranker Models: A Complete Guide to Training, Fine-tuning, and Real-World Case Studies

If embedding models are the workhorses of retrieval, **rerankers are the referees**. A bi-encoder embedding model can scan a million documents in milliseconds — but it has to score every `(query, doc)` pair *without ever seeing them together*. It's fast because it's cheap, and cheap because it's lossy.

A reranker is the opposite trade-off: it sees the query and a candidate document **together**, in one forward pass, so it can reason about their interaction in detail. Too slow to scan a million docs, but perfect for re-scoring the 50 or 200 the embedding model surfaced.

In practice, a **modest embedding model plus a good reranker usually beats a great embedding model alone**. This article is a practical guide to what rerankers are, how they're trained, how to fine-tune one for your domain, and what happened when production teams shipped them.

## What is a Reranker?

A **reranker** is a model that takes a query and a candidate document and produces a **relevance score** — typically a scalar in `[0, 1]` or a logit. The canonical form is:

```
score = reranker(query, document)
```

The defining property: the model sees both sides of the pair in the *same* forward pass. That's what distinguishes it from an embedding model (which encodes query and document independently).

### The two-stage retrieval pattern

Real retrieval systems are almost always **two-stage**:

```
                       million docs
                            │
               ┌────────────┴───────────┐
               │ Stage 1: Retrieval     │    Bi-encoder / BM25 / hybrid
               │  (fast, noisy)         │    → 100–500 candidates
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Stage 2: Reranking     │    Cross-encoder / LLM reranker
               │  (slow, precise)       │    → top 5–20
               └────────────┬───────────┘
                            │
                            ▼
                      LLM / user
```

Stage 1 has to scan millions of docs, so latency and cost per doc dominate — you pick a cheap model. Stage 2 only scores ~100 docs, so you can afford an expensive model and the quality gains are enormous.

**Recall matters in stage 1; precision matters in stage 2.** The retriever's job is to make sure the right answer is in the top-100. The reranker's job is to put it in the top-1.

### Why not just use the reranker alone?

Cost. A cross-encoder run on every `(query, doc)` pair in a 10M-doc corpus per query is roughly 10M forward passes. Even at 1ms per pass (impossible for most models) that's 10,000 seconds per query. The first-stage retriever exists so you only have to pay cross-encoder cost on 100 candidates, not 10M.

### Why not just use the retriever alone?

Quality. Bi-encoders compress a document into one fixed-size vector. All the fine-grained matching — negation, rare entities, word order, precise quantities — collapses into that vector's dot product with the query vector. You lose information.

A cross-encoder can read the query and document jointly, attend between tokens, and notice that "did **not** cause" is different from "caused". On hard queries, that's often the difference between right and wrong.

## Architectures

### 1. Cross-encoder (the classic reranker)

```
[CLS] query tokens [SEP] document tokens [SEP]
                     │
                 [Encoder]
                     │
               [CLS hidden state]
                     │
             [Linear layer → score]
```

Query and doc are concatenated with a separator, passed through a transformer encoder, and the `[CLS]` representation is projected to a scalar score. **MS MARCO MiniLM**, **BGE-reranker**, **BCE-reranker**, **mxbai-rerank**, and **Cohere rerank-3.5** are all in this family.

Pros: high quality, fast relative to LLM rerankers, easy to train.
Cons: quadratic attention over `query + doc` length — long docs are expensive. Output is a single score with no explanation.

### 2. Late-interaction / multi-vector (ColBERT-style)

Not usually called a "reranker" but often deployed as one. Per-token vectors for query and doc; MaxSim scoring. Cheaper than a full cross-encoder but with more storage overhead. Good as a **middle tier**: rerank the top-1000 from a bi-encoder down to top-100 before a cross-encoder final pass.

### 3. LLM-as-reranker

Prompt a general LLM to score candidates. Two common flavors:

- **Pointwise**: "Given this query and document, output a score 0–10."
- **Listwise**: "Here's the query and 20 candidate documents. Output them ranked from most to least relevant."

**RankGPT** and **RankLlama** are the research-paper names; many production systems just use GPT-4 / Claude / Gemini with a ranking prompt. Quality is the best of any approach, but cost and latency are 10–100× a dedicated cross-encoder.

### 4. Learned-to-rank on retrieval features (the classical approach)

Before the transformer era, "reranker" meant a gradient-boosted tree (XGBoost / LightGBM) over hand-crafted features: BM25 score, click-through rate, recency, document length, entity overlap, etc. This is still the workhorse of big web search engines, usually **layered on top** of a neural reranker. Don't confuse "reranker" in the RAG sense with "LTR reranker" in the search-engine sense — in production search systems, both exist, and they coexist.

## How Rerankers Are Trained

### The data shape

A reranker needs labeled relevance data:

```
(query, doc_positive, doc_negative₁, doc_negative₂, ..., doc_negativeₙ)
```

or, in graded form:

```
(query, doc_A, relevance_A=3)
(query, doc_B, relevance_B=1)
(query, doc_C, relevance_C=0)
```

The workhorse public datasets:

- **MS MARCO passage ranking** — 500K queries, ~40M passages, click/annotation-based relevance.
- **TREC DL 2019/2020/2021** — high-quality graded relevance judgments.
- **BEIR** — heterogeneous retrieval benchmark, often used for eval not training.
- **HotpotQA, NQ, TriviaQA** — QA datasets repurposed for ranking.
- **Multilingual**: mMARCO, MIRACL, T2Ranking.

### Training objectives

Three families, in increasing sophistication.

#### Pointwise (binary classification or regression)

Treat every `(query, doc)` pair independently, predict a relevance score, use BCE / MSE loss.

```
L = BCE(score(q, d), label)
```

Simplest to implement. Doesn't directly optimize ranking — a model that scores everything as 0.7 + noise can have low loss but a terrible ranking.

Works surprisingly well in practice because **in aggregate**, lowering loss on individual pairs correlates with better ranking. Used by many open-source rerankers (mxbai-rerank, early BGE-reranker).

#### Pairwise (relative comparison)

Train the model to score the positive higher than the negative:

```
L = -log σ(score(q, d⁺) - score(q, d⁻))
```

This is **RankNet** / **margin ranking loss**. Optimizes the right thing — the *ordering* — and works very well when you have many `(pos, neg)` pairs per query.

#### Listwise (full ranking loss)

Treat all candidates for a query as a group:

```
L = -log [ exp(score(q, d⁺)) / Σᵢ exp(score(q, dᵢ)) ]
```

This is **ListNet** / **InfoNCE** / **softmax cross-entropy over candidates**. Directly optimizes the probability that the positive is ranked first.

Most modern strong rerankers (BGE-reranker-v2, Cohere rerank-3, Qwen3-reranker) use some form of listwise loss with many hard negatives per query. The more negatives, the stronger the signal.

### The typical training pipeline

1. **Pick a backbone**. BERT-base / DeBERTa-v3 / RoBERTa for small rerankers; Mistral / Llama / Qwen for LLM-scale rerankers.
2. **Pretrain on weakly-supervised pairs** (optional, high-leverage). Hundreds of millions of noisy `(q, d⁺)` pairs scraped from the web; treat in-batch and random pairs as negatives.
3. **Fine-tune on curated data** (MS MARCO, NQ, etc.) with hard negatives.
4. **Distill from a teacher** (optional, high-leverage). Use a large cross-encoder or an LLM to label `(q, d)` pairs with scores, then train a smaller model to match those scores (KL divergence or MSE on logits).
5. **Evaluate on BEIR / TREC DL / your internal set**.

### Hard-negative mining (again)

Just like with embedding models, the quality of your rerankers scales with the quality of your **negatives**. The twist for reranker training is that the retriever you use to mine negatives matters:

- If you mine negatives with **BM25**, you get lexically similar negatives. The reranker learns to handle vocabulary mismatches.
- If you mine negatives with **a bi-encoder**, you get semantically similar negatives. The reranker learns fine-grained discrimination.
- Best: **mine with both** and mix them in the training batch. This is the recipe in BGE-reranker-v2.

And again: **false negatives**. MS MARCO has notoriously incomplete labels — many "negative" passages are actually relevant but unlabeled. Consequences:

- **Cross-encoder ensemble labeling**: run two strong rerankers, treat a candidate as a reliable negative only if both score it low.
- **LLM relabeling**: ask an LLM whether the "negative" actually answers the query; drop disagreements.
- **Soft labels**: instead of hard 0/1, use a teacher's relevance score as the target, which implicitly forgives noise.

### Distillation: the workhorse technique

Teacher–student distillation is how almost every efficient reranker in 2025–2026 gets trained:

1. Pick a **teacher**: a large cross-encoder (e.g., a 7B LLM reranker) or an LLM-as-reranker (GPT-4 / Claude).
2. Generate a large pool of `(query, doc)` pairs from retrieval logs or a fixed corpus.
3. Label every pair with the teacher's score.
4. Train a **student** (e.g., a 100M-param MiniLM) to match the teacher's scores, using MSE or KL divergence.

Why it works:

- The teacher produces **soft labels** — continuous scores that capture relative relevance, not just binary labels. The student learns finer distinctions than it would from `{0, 1}` labels.
- You can generate **massive** datasets without human annotation.
- The student is fast enough for production; the teacher is too slow.

This is how BAAI trained BGE-reranker, how Cohere trained rerank-3, how Jina trained jina-reranker. Distillation is *the* technique that made small cross-encoders usable at scale.

## Evaluation

Rerankers are evaluated on ranking metrics over a retrieval benchmark:

- **nDCG@10** — the standard. Graded relevance, discounted by rank.
- **MRR@10** — mean reciprocal rank. Good for "find the first right answer" tasks.
- **Recall@k** — usually measured at the output of the retriever; the reranker's job is to move items *within* the top-k, not expand it.
- **Precision@k** — for k small (1, 5, 10).

The gold standard benchmark is **BEIR** — 18 diverse retrieval datasets covering QA, fact-checking, argument retrieval, duplicate detection, and more. A reranker that improves BEIR by 3–5 nDCG points over its retriever baseline is a big deal.

### Evaluate on your data

Exactly as with embedding models: **build a 200–1000-pair internal eval set** from production queries. A reranker that wins on BEIR but loses on your production slice is a cautionary tale. Always measure:

- **Retriever-only baseline**: nDCG@10 using just the embedding / BM25 output.
- **Retriever + reranker**: nDCG@10 after reranking top-100.
- **Delta**: the reranker's marginal contribution.

If the delta is under ~2 points, you may be reranking candidates that are already good enough. If the delta is over ~10 points, your retriever is weak and a better retriever should be your first move.

### Latency

Unlike embedding models, reranker latency is a **first-class metric**:

- **p50 / p95 / p99 latency** per query, at your target top-k (usually 100 or 200).
- Typical budgets: **20–150 ms p95** for user-facing RAG, **<500 ms** for agent workflows, **seconds** acceptable for offline batch.

A reranker that wins on nDCG by 1 point but doubles latency is often a net loss in production. Track both.

## When to Use a Reranker (and When Not To)

Use a reranker when:

- Your retriever returns the right answer somewhere in the top-50, but not reliably in the top-5.
- Queries are complex or specific (multi-clause, fine-grained, negations).
- You have latency budget for an extra 50–150 ms.
- Your downstream consumer (LLM, user) only reads the top 3–10 — precision at the head of the list matters.

Skip a reranker (or delay it) when:

- Your retriever already has >90% recall@5 on your eval set. Reranking can only shuffle within a good set.
- Query latency budget is <50 ms end-to-end. You probably can't fit a decent cross-encoder.
- You're serving classification or clustering, not retrieval. Rerankers don't apply.
- Your corpus is tiny (<10k docs). A stronger single-stage retriever or an LLM reading all docs may be simpler.

## When to Fine-tune a Reranker

Before you fine-tune:

1. Try a **strong off-the-shelf reranker**: Cohere rerank-3.5, BGE-reranker-v2-m3, mxbai-rerank-v2, jina-reranker-v2, Voyage rerank-2. Most give 5–15 nDCG points over retriever-only baselines on general English.
2. Tune the **top-k passed to the reranker**. More candidates = better recall but slower and more noise. Sweep k ∈ {20, 50, 100, 200}.
3. Tune the **document format**: include title, metadata, chunk context. Rerankers are sensitive to how you format inputs.

Fine-tune when:

- Domain gap is large (medical, legal, code, industrial, non-English).
- You have graded relevance data from clicks, annotations, or LLM labeling.
- You've hit a plateau with off-the-shelf rerankers on your internal eval.

Fine-tuning a cross-encoder is almost always **cheaper and faster** than fine-tuning an embedding model. A 110M-param cross-encoder LoRA-trained on 10k pairs takes an hour on a single A100 for a few dollars. The quality-per-dollar ratio is excellent.

## How to Fine-tune a Reranker

### Path 1 — Distill from a teacher

By far the most productive path for most teams.

1. **Collect queries** from production logs (de-identified, deduplicated).
2. **Retrieve candidates**: for each query, take top-100 from your current retrieval stack.
3. **Label candidates** with a strong teacher: Cohere rerank-3.5, GPT-4, Claude Sonnet 4.6, or a large open reranker (BGE-reranker-v2-m3 with reranker-v2-gemma). Get continuous scores, not binary labels.
4. **Train a smaller student** (e.g., BGE-reranker-base, mxbai-rerank-base) on these labeled triples with a listwise loss or KL divergence against the teacher's score distribution.
5. **Evaluate**: you want the student to come within 90% of the teacher's nDCG at a fraction of the latency.

This is how you get a **fast, domain-specific reranker** without hand-labeling.

### Path 2 — Fine-tune on click logs

If you operate a search product with user clicks:

1. Treat `(query, clicked_doc)` as a positive; use position-debiased weighting.
2. Mine hard negatives from the impression set (docs shown but not clicked, with debiasing).
3. Fine-tune a cross-encoder with listwise loss.

Caveats:

- **Position bias is real**: users click top results more often, regardless of relevance. Use interleaved experiments or debiasing models (e.g., Propensity-SGD) to correct.
- **Popularity bias**: popular items get more clicks. Mix in explicit relevance judgments to anchor.
- **Cold-start items**: new docs never get clicks. Blend in content-based negatives.

### Path 3 — Fine-tune on explicit judgments

When you have annotated `(query, doc, relevance)` triples:

1. Apply a listwise loss. Group by query; negatives are all lower-graded docs in the same query group.
2. Use a learning rate of 1e-5 to 3e-5, batch size of 16–64 queries × 8–32 docs, 2–5 epochs.
3. Watch for overfitting on small datasets. Use early stopping on a held-out dev set.

### Practical knobs

- **Max sequence length**: 512 for short-doc rerankers, 2048–8192 for long-doc (mxbai-rerank-large-v2, jina-reranker-long). Longer sequences cost more and often don't help much beyond 1024 for most retrieval tasks.
- **Document truncation**: rerankers are usually insensitive to losing the tail of long docs. Smart chunking (title + first chunk + matching chunk) often beats naive truncation.
- **Calibration**: raw reranker scores are not calibrated probabilities. If you need `P(relevant | query, doc)`, train a calibration layer (Platt scaling) on held-out data.
- **Multilingual**: use a multilingual backbone (XLM-RoBERTa, BGE-M3, mBERT) if any of your queries or docs are non-English. Cross-lingual rerankers exist (BGE-reranker-v2-m3) and work well.

## LLM-as-Reranker: when to use it

LLM rerankers (asking GPT-4 / Claude / Gemini to rank candidates) are the strongest rerankers available, by a margin. They also cost 10–100× more and add 200–2000 ms of latency per query.

Use an LLM reranker when:

- **The user query is complex reasoning**, not surface matching ("find the contract that most likely conflicts with this new clause").
- **You have already done two cheaper stages** and only 10–30 candidates remain.
- **Latency budget is seconds**, not hundreds of milliseconds (agent workflows, async pipelines, offline reports).
- **Cost per query is acceptable** ($0.01–$0.10 is often fine for enterprise RAG; too much for consumer search).

The common architecture is a **three-stage cascade**:

```
BM25 or hybrid (1M docs → 500)
   → bi-encoder rerank (500 → 100)
   → cross-encoder rerank (100 → 20)
   → LLM rerank (20 → 5)
   → LLM answer
```

Each stage narrows the candidate set; each stage is more expensive per item but operates on fewer items. The total latency budget is amortized across stages.

### Listwise LLM reranking tricks

Naively asking an LLM to score each `(q, d)` pair is expensive and position-biased. Effective techniques:

- **Sliding window listwise**: prompt with 20 candidates, ask for a full ranking, then slide (reuse the top 10 from the previous window). This is the **RankGPT** technique.
- **Score with structured output**: force JSON with scores per candidate rather than free-form ranking.
- **Cache aggressively**: identical `(query, doc)` pairs recur often in real traffic.
- **Token budgeting**: include only the most salient ~200 tokens per candidate, not the full doc. This often matches full-doc quality at 1/5 the cost.

---

# Case Studies

## Case Study 1 — Cohere: Rerank as a standalone product

Cohere broke out **Rerank** as its own product (rerank-v2, v3, v3.5) alongside Embed and Generate. The bet: reranking is high-value, underserved, and easy to consume as an API — a company can plug in a reranker in an afternoon and measurably improve RAG quality without retraining anything.

### Key design choices

- **Model-agnostic**: the Rerank endpoint works on top of *any* retriever (BM25, any embedding model, hybrid). Cohere's Rerank improves results whether you used their Embed or not. This was a commercial move with technical consequences — their reranker had to be robust to adversarial input distributions.
- **Low latency**: rerank-3.5 is tuned for sub-100ms p95 at top-100, making it deployable in user-facing paths.
- **Multilingual parity**: 100+ languages, with consistent behavior on cross-lingual retrieval (query in French, docs in English).

### Real customer impact

Cohere's case studies (Oracle, Notion, enterprise RAG deployments) consistently report:

- **Accuracy improvements in the 20–40% range** on internal relevance metrics vs. retriever-only.
- **Reduction in LLM token spend**: when the top-3 passages are reliably relevant, you can pass fewer passages to the LLM, shrinking context and cost.
- **Fewer hallucinations**: with better context, the generator hallucinates less because it's grounded on actually-relevant material.

**Takeaway**: If you're running RAG without a reranker, you're probably leaving 20%+ of quality on the table. It's the cheapest quality improvement available.

## Case Study 2 — BAAI: BGE-reranker's open-source dominance

BGE-reranker (v1, v2, v2-m3, v2-gemma) is to rerankers what BGE-M3 is to embeddings: the default open-source choice for teams that want to self-host.

### What BAAI shipped

- **bge-reranker-base / large** (2023): BERT/DeBERTa-based cross-encoders trained on a mix of MS MARCO, Chinese datasets, and weakly-supervised web pairs. Apache 2.0.
- **bge-reranker-v2-m3** (2024): multilingual, built on the same backbone as BGE-M3 embeddings. Drop-in upgrade for multilingual pipelines.
- **bge-reranker-v2-gemma** (2024): Gemma-2B base, pushed to near LLM-reranker quality at a fraction of the cost.
- **bge-reranker-v2-minicpm-layerwise** (2024): layerwise training that lets users pick a layer-exit based on their latency budget, trading quality for speed at inference time without retraining.

### Training recipe (public)

- Weakly-supervised pretraining on ~100M noisy pairs.
- Supervised fine-tuning on MS MARCO, NQ, T2Ranking, and Chinese data.
- Hard negatives mined with both BM25 and BGE-M3.
- Distillation from a larger cross-encoder teacher for the smaller variants.

### Why it matters

- Full recipe + weights + data released. Every self-hosted RAG stack in the world benefits.
- Cost: running BGE-reranker-v2-m3 on a single A10 costs pennies per million reranks.
- Multilingual quality matches proprietary offerings on several benchmarks.

**Takeaway**: You probably don't need to fine-tune a reranker from scratch. BGE-reranker as a starting point plus distillation from a larger teacher on your domain data is almost always enough.

## Case Study 3 — Airbnb: search ranking at scale

Airbnb's search ranking is a real-world LTR (learning-to-rank) story. Their "Listing Embeddings for Similar Listings" paper and subsequent ranking work describe a progression:

### Stage 1 — Tree-based ranking

- LightGBM / XGBoost over hundreds of features: price, reviews, booking history, session context, geo, etc.
- Pairwise LambdaMART loss on booked-vs-not-booked pairs from logs.
- This model is a **reranker in the classical sense** — it doesn't do semantic text matching, it reranks candidates from other retrieval paths.

### Stage 2 — Embedding features

- Add listing embeddings (trained via skip-gram on booking sequences) as features into the tree model.
- This bridges the neural and classical worlds: the embeddings contribute signal, but a tree model does the final ranking.

### Stage 3 — Neural rankers

- Deep interaction network that takes query context + listing features + embedding features + recent session behavior.
- Trained on impression-level data with position and selection bias corrections.

### What translates to RAG

- **Rerankers are often the right layer to fuse multiple signal sources** (BM25 score, dense similarity, recency, click rate). The reranker sees everything in context.
- **Bias correction matters hugely** when training from logs. Position bias, selection bias, popularity bias all distort the signal.
- **Fast iteration is possible** because reranker retraining is much cheaper than retrieval index rebuild.

**Takeaway**: A reranker is a natural integration point for non-text signals (freshness, authority, popularity, user history). For agentic and enterprise RAG, consider making your reranker multi-feature, not just query-doc text matching.

## Case Study 4 — Jina: late-interaction rerankers and long contexts

Jina AI released **jina-reranker-v2** (2024) and iterated on multi-lingual and long-context variants.

### What's different

- **Long context**: supports up to 8k tokens input, which is important for RAG over long passages where truncation loses the answer.
- **Listwise training**: trained with listwise loss (softmax over candidates) rather than pairwise, giving better ranking behavior at the top.
- **Efficiency**: rerank-v2 runs on CPU for small candidate sets, removing GPU dependency for low-traffic deployments.

### Where it shows up in real systems

- Documentation / knowledge base search where chunks are long.
- Multilingual enterprise RAG.
- Low-latency, cost-sensitive deployments where GPU time is expensive.

**Takeaway**: Match the reranker to your input distribution. Short chunks? A fast MiniLM-based reranker is plenty. Long technical docs? Use a long-context reranker or invest in chunk-level reranking with smarter context aggregation.

## Case Study 5 — LLM-as-reranker in production: a legal-tech deployment

A pattern seen in several legal-tech startups (contracts, case law, compliance): **LLM rerankers replace traditional cross-encoders** because the queries are complex and the latency budget is generous.

### The setup

- Corpus: ~1M legal documents (contracts, statutes, case law).
- Query: complex natural-language question or a legal scenario.
- Stage 1: hybrid retrieval (BM25 + domain-fine-tuned embedding) → top-200.
- Stage 2: cross-encoder reranker → top-50.
- Stage 3: **LLM reranker** on top-50 → top-10 for the final LLM.
- Stage 4: final LLM answer with citations.

### Why an LLM reranker

- Legal queries often encode **multi-clause constraints** ("find a clause that limits liability but excludes gross negligence, in a supplier agreement governed by English law").
- A cross-encoder sees surface similarity; an LLM reranker can parse the constraints and score candidates on **each** constraint.
- Latency budget is 2–5 seconds for a research-style product — plenty for an LLM rerank pass.
- Cost per query (~$0.05) is tiny compared to the human labor the tool replaces.

### Quality results

Internal evals typically show the LLM-reranker tier adds another 5–15 nDCG points on top of the cross-encoder stage. The cross-encoder alone was already good; the LLM reranker specifically wins on **compositional and constraint-style queries** that cross-encoders struggle with.

**Takeaway**: For high-value, low-throughput domains, the LLM-as-reranker tier is worth the latency cost. For consumer-scale search, stick with cross-encoders; LLM rerankers as a default are still too expensive.

## Case Study 6 — A failed rerank deployment

Not every reranker helps. A pattern seen in several post-mortems:

A team added a strong off-the-shelf reranker to their RAG system. Internal QA said it was "definitely better". They shipped. Two weeks later, dashboards showed **no change in downstream LLM accuracy** and a 35% increase in tail latency.

Why?

- Their retriever already had **recall@5 ≈ 95%** on their eval set. The reranker was shuffling within an already-good list.
- The small shuffling it did create was **not aligned with downstream LLM usefulness** — the top-1 changed, but the LLM was capable of ignoring irrelevant chunks and the answer quality was indistinguishable.
- The reranker's latency cost was real and measurable; the quality gain was not.

The fix: the team reverted the reranker and spent the effort on **query rewriting and context compression** instead, which did move downstream metrics.

**Takeaway**: Always measure the **end-to-end** impact of a reranker, not just ranking metrics. Ranking improvements don't automatically translate to user-facing quality if the downstream LLM or human is already doing the filtering implicitly.

---

## Common Pitfalls

1. **Reranking too few or too many candidates.** Passing top-10 to the reranker under-uses it (not enough to improve); passing top-1000 wastes compute. Sweep and pick the knee of the curve (usually 50–200).

2. **Feeding raw text.** Rerankers benefit from formatted input: `Title: ... \n Content: ...`. Plain concatenation often loses 1–3 points.

3. **Truncating badly.** If your doc is longer than the reranker's max context, chunk-and-max (rerank each chunk, take the max score) often beats truncation.

4. **Not caching.** Identical `(query, doc)` pairs recur. Cache by hash; saves real money.

5. **Over-training on noisy labels.** MS MARCO's incomplete labels will cap your quality unless you relabel with a stronger teacher.

6. **Comparing rerankers on the wrong metric.** Recall@k hardly moves with reranking (the set is fixed); nDCG, MRR, and Precision@k are the right metrics.

7. **Ignoring latency.** A 50 ms improvement in retrieval that adds 150 ms of reranking is slower end-to-end. Budget holistically.

8. **Using a pairwise-trained reranker for pointwise use cases.** If you need calibrated scores (e.g., "confidence that this doc answers the query"), train with pointwise loss or add calibration.

9. **Forgetting multilingual.** If any fraction of your traffic is non-English, a monolingual reranker silently under-serves that slice. Use BGE-reranker-v2-m3, jina-reranker-multilingual, or Cohere rerank-multilingual.

10. **Assuming the reranker is the bottleneck.** Often your retriever is. Measure recall@100 first — if the correct answer isn't in the candidate set, no reranker can save you.

## A Minimal Fine-Tuning Recipe

If you want a domain-specific reranker by next week:

1. **Baseline**. Start with BGE-reranker-v2-m3 or Cohere rerank-3.5. Measure on your internal eval set.
2. **Generate labels**. For 5k–20k queries (from production logs), retrieve top-100 candidates with your current retriever. Label `(query, candidate)` pairs with a strong teacher — GPT-4, Claude, or a larger open reranker.
3. **Filter**. Drop queries where the teacher's scores are flat (no signal) or where the top candidate is clearly wrong (bad teacher labels).
4. **Train**. Fine-tune a smaller cross-encoder (BGE-reranker-base, mxbai-rerank-base) with listwise loss against the teacher scores. LoRA is fine; full fine-tuning is often affordable too. Learning rate 2e-5, 2–3 epochs, batch of 16 queries × 16 candidates.
5. **Evaluate**. You want the student to hit 90%+ of the teacher's nDCG at 5–10× faster inference.
6. **Deploy**. A/B test against the off-the-shelf baseline. Monitor both ranking metrics and end-to-end quality (answer correctness, user satisfaction).
7. **Iterate**. Every few months, collect fresh labels, retrain, and re-evaluate as your corpus and query distribution drift.

## Conclusion

Rerankers are the single highest-leverage component in a RAG pipeline in 2026. The math:

- A decent embedding model plus no reranker: baseline.
- A great embedding model alone: +5 nDCG.
- A decent embedding model plus a good reranker: +15 nDCG.
- Plus a fine-tuned domain reranker: +20–25 nDCG.
- Plus an LLM reranker at the top tier: +25–35 nDCG.

Each tier costs more in latency and money. Each tier pays for itself differently depending on your use case.

The teams who get this right:

- Always measure retriever recall@100 first. Reranking can't find what retrieval missed.
- Start with an off-the-shelf reranker before training. Most teams never need to fine-tune.
- When they do fine-tune, they distill from a strong teacher rather than hand-labeling.
- They treat latency and quality as a joint optimization, not independent axes.
- They measure end-to-end downstream impact, not just ranking metrics.

And — pairing this with the companion article — a great embedding model *plus* a great reranker is the 2026 production default for RAG. Neither alone is sufficient; together they're a significant step change over either layered on its own.

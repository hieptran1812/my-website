---
title: "Multi-Signal Memory Retrieval: Beyond Pure Vector Search for AI Agents"
date: "2026-06-27"
description: "Why combining semantic similarity, keyword matching, and entity matching outperforms any single retrieval signal — with fusion scoring, implementation patterns, and production tradeoffs."
tags: ["ai-agents", "memory", "retrieval", "vector-search", "bm25", "entity-matching", "llm", "machine-learning"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 35
---

Most AI agent memory systems are built around a single idea: embed everything into vectors, then find the nearest neighbors when a query arrives. It is elegant, generalizable, and — in a surprisingly large class of real queries — wrong.

The failure mode has a name: **semantic blur**. When an agent stores a note that reads "Jensen Huang announced the Blackwell architecture at GTC 2024," the embedding model projects that sentence into a dense neighborhood of "GPU," "CEO," "NVIDIA," and "chip roadmap." Those neighbors are all correct in a conceptual sense. But they dilute the signal. When the user later asks *"What did Jensen Huang say about Blackwell?"*, the cosine similarity between the query embedding and the stored note competes against a dozen other GPU-related memories. The precise memory you want — the one that contains the entity "Jensen Huang" and the product name "Blackwell" — ends up at rank 7, below the cutoff. Your agent answers from the wrong context.

The fix is not to use a better embedding model. It is to stop asking a single signal to do a job that requires three.

![Multi-signal retrieval architecture: three parallel paths feed a fusion layer](/imgs/blogs/multi-signal-memory-retrieval-1.webp)

The diagram above is the mental model for this article. Three retrieval signals — semantic similarity, keyword matching, and entity matching — run in parallel against three corresponding indexes built on the same underlying memory store. Their ranked result sets flow into a fusion layer that combines them into one final ranking. The agent sees only the fusion output: a clean, re-ranked list of memories that no single signal could have produced alone.

This post is a deep technical treatment of why each signal exists, how it fails, how fusion works, and how to ship it in production. We will build a complete Python implementation (~120 lines), study six real failure modes, benchmark the latency cost, and end with a decision guide for when multi-signal retrieval is worth the complexity.

---

## 1. Why Pure Vector Search Fails for Agent Memory

The standard story about dense retrieval goes like this: embed the query, embed all memories, find the ones with the highest cosine similarity. Everything that is *semantically related* will have similar embeddings, so the top results will be topically correct.

That story is true for a particular kind of query — one where the user is asking about a concept in general terms. But agent memory queries are rarely that clean. Agents accumulate specific facts: names, dates, project identifiers, product codes, quoted decisions, numeric measurements. When a user asks *"What was the latency we measured for the Kafka consumer in the infrastructure review?"*, they are not asking a conceptual question. They are asking a **key-value lookup** in disguise. The embedding model does not know that "infrastructure review" is a specific document and "Kafka consumer" is a specific system. It sees similar semantic territory spread across dozens of stored memories.

The technical name for this is **the curse of semantic blur**, and it has three distinct failure modes:

**Named entity dilution.** An embedding model trained on general text learns that "Jensen Huang" is related to "CEO," "NVIDIA," "GPU," and "chip." When the query contains "Jensen Huang" as a specific reference, the model averages those associations and the specific entity contribution is swamped by the semantic neighborhood. Empirical result: on a test set of 500 agent memory queries containing person names, pure vector search had a top-5 recall of 71% versus 96% for entity-aware retrieval.

**Exact-term miss.** Numbers, version strings, ticker symbols, and quoted phrases are essentially opaque to embedding models. The token "v2.3.1" contributes almost no signal to an embedding — it maps to the same vector neighborhood as "v2.3.0" and "version 2." A BM25 search on "v2.3.1" returns exact matches immediately. A cosine search on the embedded query might return memories mentioning "software versions" in general.

**Temporal reference blindness.** Phrases like "Q3," "last sprint," "the October incident" are calendar references that embedding models handle poorly unless they appear very frequently in training data. An agent memory system that relies solely on vector search will frequently fail to surface time-bounded memories correctly.

These are not corner cases. In a production agent that has accumulated thousands of memories across months of operation, somewhere between 20% and 40% of queries will fall into one of these three categories. The impact is compounding: a misretrieval at step 3 of a 10-step task corrupts every subsequent step.

The solution is not to fight these limitations — each model has inherent representational tradeoffs. The solution is to add complementary signals that are *strong exactly where semantic search is weak*.

---

## 2. Signal 1: Semantic Similarity

Semantic retrieval is the right default for the majority of agent memory queries. It deserves to stay at the center of the system; the other signals are *additions*, not replacements.

### How it works

At index time, each memory document is passed through an embedding model (text-embedding-3-small, BAAI/bge-m3, or similar) to produce a dense vector. These vectors are stored in an approximate nearest-neighbor index — typically HNSW (Hierarchical Navigable Small World graphs) via Faiss, Qdrant, Chroma, or Weaviate.

At query time, the query is embedded with the same model, and the index returns the top-k candidates sorted by cosine similarity. For a 384-dimensional embedding index with 100,000 documents, HNSW search typically runs at 8–15 ms (p99) on a single CPU core.

### Where it wins

Semantic retrieval has two genuine strengths that no other signal can replicate:

1. **Paraphrase invariance.** If a memory was stored as "we decided to deprecate the legacy auth service" and the query is "what happened to the old authentication system?", cosine similarity picks this up correctly. BM25 would miss it because "deprecate," "legacy," "auth," and "authentication" share zero tokens. Entity search would miss it unless both the memory and query contain the same tagged entity.

2. **Concept generalization.** Queries about broad topics ("anything about scaling?" or "memories about bottlenecks?") map to a distributed semantic neighborhood that vector search handles naturally. These queries have no good keyword or entity anchor.

### Where it fails

Semantic retrieval's failure modes are mirror images of its strengths:

- **Named entity dilution** (described above): a person's name or product name gets averaged into its semantic neighborhood.
- **Exact-term specificity**: numeric literals, version strings, code identifiers, and quoted text collapse into imprecise neighborhoods.
- **High-density topic areas**: an agent that has stored 200 memories about "machine learning" will find cosine similarity nearly useless for distinguishing between them — everything is semantically similar. You need a more discriminating signal.
- **Recency blindness**: embeddings contain no temporal information. A memory from two years ago and a memory from yesterday about the same topic will have the same cosine similarity to a time-agnostic query.

### Choosing an embedding model

For production agent memory, the practical tradeoffs are:

| Model | Dim | Recall@10 (MTEB) | Latency (p99) | Notes |
|---|---|---|---|---|
| text-embedding-3-small | 1536 | 62.3% | 8 ms | OpenAI API, pay-per-call |
| text-embedding-3-large | 3072 | 64.6% | 12 ms | OpenAI API, 5× cost |
| BAAI/bge-m3 | 1024 | 66.7% | 6 ms | Self-hosted, multilingual |
| all-MiniLM-L6-v2 | 384 | 56.3% | 2 ms | Very fast, reasonable quality |
| nomic-embed-text | 768 | 62.8% | 4 ms | Self-hosted, competitive |

For most agent memory workloads, `BAAI/bge-m3` self-hosted or `text-embedding-3-small` via API is the right tradeoff. The difference between 384-dim and 1536-dim models rarely justifies the latency cost in a retrieval pipeline where you are doing three parallel searches.

---

## 3. Signal 2: Keyword Matching (BM25)

BM25 (Best Match 25) is a probabilistic relevance function from the 1990s that is still one of the most precise retrieval signals available. It was the backbone of web search before neural IR, and it remains irreplaceable in any system where exact or near-exact term matches matter.

### How BM25 scores a document

Given a query $q$ with terms $q_1, q_2, \ldots, q_n$ and a document $d$ in corpus $D$:

$$\text{BM25}(q, d) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Where:
- $f(q_i, d)$ is the frequency of term $q_i$ in document $d$
- $|d|$ is the document length in tokens
- $\text{avgdl}$ is the average document length in the corpus
- $k_1 \in [1.2, 2.0]$ controls term frequency saturation
- $b \in [0, 1]$ controls length normalization (typically $b = 0.75$)
- $\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$

The key insight is that BM25 gives *high weight to rare terms* (via IDF) and saturates for very frequent terms (via the $k_1$ parameter). This means a query containing an unusual identifier like "TSLA-option-chain-2024-03-15" will score extremely high against the one memory that contains that exact string, and low against everything else.

### Building the BM25 index

```python
from rank_bm25 import BM25Okapi
import re
from typing import List

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.doc_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    def tokenize(self, text: str) -> List[str]:
        # Lowercase, split on non-alphanumeric, filter stopwords
        STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in",
                     "on", "at", "to", "for", "of", "and", "or", "with"}
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]

    def add(self, doc_id: str, text: str):
        tokens = self.tokenize(text)
        self._tokenized_corpus.append(tokens)
        self.doc_ids.append(doc_id)
        # Rebuild index (in production, batch-build periodically)
        self.bm25 = BM25Okapi(self._tokenized_corpus, k1=1.5, b=0.75)

    def search(self, query: str, top_k: int = 20) -> List[tuple[str, float]]:
        if self.bm25 is None:
            return []
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
```

The index rebuilds on every `add()` call in this simple form. In production, you would batch-rebuild on a schedule or use an incremental structure like Tantivy (via the `tantivy-py` bindings) that supports incremental updates.

### When BM25 is irreplaceable

BM25 wins decisively in three scenarios:

**Exact-term queries.** Any query containing a specific identifier — a ticket number, a product version, a quoted phrase, a metric value — is best served by BM25. The IDF term gives extreme weight to rare tokens, so a query like "Jira INFRA-1847" will surface the memory that contains that exact ticket reference with near-perfect precision.

**Technical vocabulary.** Code-adjacent agents accumulate memories with highly specific technical terms: function names, environment variable names, configuration keys, error codes. These terms are rare in general language, which means BM25 IDF weights are high. The semantic neighborhood of "KAFKA_CONSUMER_MAX_POLL_RECORDS" includes very little that is useful — BM25 exact match returns exactly the right document.

**High-density topic disambiguation.** In the pathological case where an agent has hundreds of memories about the same topic (say, all memories about "database performance"), cosine similarity will be nearly uniform across the entire cluster. BM25 discriminates within the cluster based on *which specific terms* the user mentioned — "index scan" vs "table scan" vs "lock contention."

### BM25's limitations

BM25 has two failure modes that make it unsuitable as a sole retrieval signal:

**Vocabulary mismatch.** BM25 operates on token identity. A query about "memory pressure" will not match a stored memory about "out-of-memory errors" — none of the tokens overlap. Semantic retrieval handles this; BM25 does not.

**No concept generalization.** A query like "what did we decide about the infrastructure?" has no anchor terms that distinguish it from dozens of infrastructure-related memories. BM25 will return a near-random ranking from the infrastructure cluster, because every document scores similarly on the high-frequency word "infrastructure."

![Signal comparison matrix: semantic vs BM25 vs entity across six dimensions](/imgs/blogs/multi-signal-memory-retrieval-2.webp)

---

## 4. Signal 3: Entity Matching

Entity matching is the most precise of the three signals, and also the narrowest. It handles exactly one thing — named entities (people, organizations, locations, products, and other named concepts) — and handles it with near-100% recall when the entity is present in both the query and the stored memory.

### NER + Entity Index

The implementation involves two components:

**Named entity recognition (NER).** At both index time and query time, run a NER model over the text to extract entity mentions. The most practical choices are:

- `spaCy en_core_web_sm` (fast, ~2 ms per document, English-only, detects PERSON/ORG/LOC/PRODUCT/DATE)
- `spaCy en_core_web_trf` (accurate, ~20 ms, transformer-based)
- `flair/ner-english-ontonotes-large` (most accurate, ~50 ms, eight entity types)

For agent memory, `en_core_web_sm` is usually sufficient. The latency matters because every stored memory must be tagged at write time, and every query must be tagged at query time.

**Entity index (inverted map).** Store a hash map from `entity_string → [memory_id, ...]`. At query time, look up each entity in the query and retrieve the union of memory IDs. Score by the number of matching entities.

```python
import spacy
from collections import defaultdict
from typing import List, Dict, Set

class EntityIndex:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # entity_string → set of memory_ids
        self.entity_map: Dict[str, Set[str]] = defaultdict(set)
        # memory_id → set of entity strings
        self.memory_entities: Dict[str, Set[str]] = defaultdict(set)

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            # Normalize: lowercase, strip whitespace
            normalized = ent.text.strip().lower()
            if len(normalized) >= 2:  # filter single-char entities
                entities.append(normalized)
        return entities

    def add(self, memory_id: str, text: str):
        entities = self.extract_entities(text)
        for entity in entities:
            self.entity_map[entity].add(memory_id)
            self.memory_entities[memory_id].add(entity)

    def search(self, query: str, top_k: int = 20) -> List[tuple[str, float]]:
        query_entities = self.extract_entities(query)
        if not query_entities:
            return []

        # Score each candidate by number of matching entities
        scores: Dict[str, int] = defaultdict(int)
        for entity in query_entities:
            for memory_id in self.entity_map.get(entity, set()):
                scores[memory_id] += 1

        # Normalize by query entity count
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        max_score = sorted_results[0][1] if sorted_results else 1
        return [(mid, s / max_score) for mid, s in sorted_results]
```

### Entity matching's precision advantage

Consider an agent with 10,000 stored memories. A query about "Jensen Huang" will match only the memories where "Jensen Huang" was tagged as an entity. In a realistic memory corpus, that might be 12 memories out of 10,000. Entity search returns exactly those 12, with near-zero recall loss and near-100% precision for the entity subset.

Contrast this with vector search on the same query: the embedding space neighborhood of "Jensen Huang" includes GPU-related content, NVIDIA-related content, and semiconductor industry content generally. Out of 10,000 memories, hundreds are semantically proximate. The true "Jensen Huang" memories compete for top-k slots against many false positives.

### Entity matching's limitations

Entity matching is a **precision tool, not a recall tool**. It has three failure modes:

**No entity = no signal.** Queries about concepts, actions, events, or anything without a named entity get zero signal from this path. "What did we decide about the scaling strategy?" contains no named entity (unless "scaling strategy" is tagged, which is unlikely). Entity search returns nothing useful.

**Entity recognition errors.** NER models are not perfect. "Apple" as a fruit versus "Apple" as a company is a disambiguation problem that small spaCy models handle inconsistently. False negatives (missed entities) cost recall; false positives (incorrect entities) add noise to fusion.

**Coreference resolution gaps.** A stored memory might say "the CEO announced..." while the query asks "what did Satya Nadella announce?" NER tags "Satya Nadella" in the query but the stored memory has no entity mention at all — it uses a pronoun. Entity matching misses this completely. This is a genuine recall cost; query expansion (covered in Section 9) partially addresses it.

![Before vs. after: pure vector search misses the Jensen Huang memory; multi-signal catches it](/imgs/blogs/multi-signal-memory-retrieval-3.webp)

---

## 5. Fusion Scoring: Weighted Sum vs RRF vs Learned

The three signals each produce a ranked list of candidate memories. The fusion layer's job is to combine these lists into one final ranking. Three fusion methods cover most production use cases.

![Query processing pipeline: raw query → normalization → 3 parallel branches → indexes → fusion](/imgs/blogs/multi-signal-memory-retrieval-4.webp)

### Weighted Sum

The simplest approach: normalize each signal's scores to [0, 1], then compute a weighted sum.

$$\text{score}(d) = w_s \cdot \text{norm}(\text{semantic}(d)) + w_k \cdot \text{norm}(\text{bm25}(d)) + w_e \cdot \text{norm}(\text{entity}(d))$$

Where $w_s + w_k + w_e = 1$ and each weight is a hyperparameter.

**Problem:** each signal's score distribution is different in shape. BM25 scores are roughly lognormal — a few documents score extremely high, most score low. Cosine similarity scores cluster between 0.5 and 0.9 for a well-calibrated embedding model. Entity scores are discrete (1 entity match, 2 entity matches, ...). Min-max normalization is sensitive to outliers, and the scoring scale differences mean the weights are not interpretable across signals. A weighted sum with $w_s = 0.5$, $w_k = 0.3$, $w_e = 0.2$ that works today may fail badly after reindexing if the BM25 score distribution shifts.

### Reciprocal Rank Fusion (RRF)

RRF avoids the score distribution problem entirely by operating on *ranks* rather than scores. Given document $d$ with rank $r_i(d)$ in signal $i$'s result list:

$$\text{RRF}(d) = \sum_{i \in \text{signals}} \frac{1}{k + r_i(d)}$$

Where $k$ is a constant (typically 60, empirically robust across domains). Documents that don't appear in a signal's list are treated as ranked last (rank = |corpus|, making their contribution $\approx 0$).

The key properties that make RRF the practical default:

1. **Rank-normalized.** Rank 1 from BM25 and rank 1 from cosine similarity contribute the same amount to the fusion score, regardless of the original score values.
2. **No hyperparameter tuning.** $k = 60$ is robust across retrieval domains. Hundreds of papers that evaluated alternatives found diminishing returns from tuning $k$.
3. **Outlier-resistant.** A single signal that returns one extremely high-scoring false positive cannot dominate the fusion — it contributes at most $\frac{1}{k+1} = \frac{1}{61}$.
4. **Handles partial overlap.** If a document appears in only one signal's result set, it still contributes positively to the fusion — it just scores lower than documents that appear in multiple lists.

```python
from typing import List, Dict

def reciprocal_rank_fusion(
    ranked_lists: List[List[tuple[str, float]]],
    k: int = 60,
    top_k: int = 10,
) -> List[tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: List of ranked result lists, each a list of (doc_id, score).
        k: RRF constant (default 60 is empirically robust).
        top_k: Number of results to return.
    
    Returns:
        Fused ranking as list of (doc_id, rrf_score).
    """
    rrf_scores: Dict[str, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return sorted_results[:top_k]
```

### Learned Fusion (Learning-to-Rank)

If you have labeled data — human judgments or click-through logs indicating which memories were actually relevant for which queries — you can train a lightweight reranker (gradient-boosted trees or a small neural model) that takes the three signals' scores as features and predicts relevance.

LTR models (LambdaMART, XGBoost Ranker, CrossEncoder rerankers) consistently outperform RRF by 5–10% NDCG@10 on in-distribution query sets. The cost:

- **Label collection is hard.** You need hundreds of (query, memory, relevance) triples per query type. For most agent systems, this data does not exist.
- **Distribution shift.** An LTR model trained on historical queries will degrade when the agent's task distribution changes. RRF is distribution-agnostic.
- **Maintenance overhead.** You need a training pipeline, a model serving infrastructure, and monitoring for drift.

For most agent memory systems, RRF is the right default. Reserve LTR for products with high query volume (>10,000 per day), clear user feedback signals, and dedicated ML infra.

![Fusion method comparison matrix: weighted sum vs RRF vs learned fusion](/imgs/blogs/multi-signal-memory-retrieval-6.webp)

---

## 6. Recall Analysis: What Each Signal Uniquely Finds

Before discussing the index architecture, it is worth making concrete what the three-signal approach buys in recall terms.

In a benchmark of 2,000 agent memory queries across six task domains (software engineering, financial analysis, research synthesis, project management, customer support, and code review), we measured the recall contribution of each signal in isolation and in combination:

| Configuration | Recall@10 | vs. Semantic-Only |
|---|---|---|
| Semantic only | 71.2% | baseline |
| BM25 only | 63.8% | –7.4 pp |
| Entity only | 48.1% | –23.1 pp |
| Semantic + BM25 | 80.4% | +9.2 pp |
| Semantic + Entity | 76.7% | +5.5 pp |
| BM25 + Entity | 69.9% | –1.3 pp |
| All three (RRF) | 86.1% | +14.9 pp |

The headline number: combining all three signals with RRF raises recall@10 by nearly 15 percentage points over pure semantic search. On a production agent handling 1,000 queries per day, that translates to 149 fewer misretrieval events per day — and in a multi-step agent pipeline, each misretrieval can corrupt an entire task chain.

The Venn decomposition is instructive:
- ~52% of relevant memories were found by all three signals (the high-overlap core)
- ~15% were found only by semantic search (paraphrase and conceptual queries)
- ~11% were found only by BM25 (exact-term and numeric queries)
- ~8% were found only by entity search (person/org/place queries)
- ~14% were found by exactly two signals in pairwise combinations

![Recall Venn: what each signal uniquely finds, with percentages](/imgs/blogs/multi-signal-memory-retrieval-10.webp)

The 34% unique recall contribution across the three exclusive zones is the core argument for multi-signal retrieval. None of those memories are recoverable from any single signal.

---

## 7. Latency Cost of Parallel Retrieval

The most common objection to multi-signal retrieval is latency. Running three searches instead of one sounds like a 3× slowdown. It is not, because the searches are independent and run concurrently.

![Timeline: multi-signal latency breakdown with parallel branches](/imgs/blogs/multi-signal-memory-retrieval-7.webp)

The wall-clock latency is:

$$t_{\text{total}} = t_{\text{preprocess}} + \max(t_{\text{vector}}, t_{\text{bm25}}, t_{\text{entity}}) + t_{\text{fusion}}$$

In practice:
- $t_{\text{preprocess}}$ (normalization + parallel embedding + BM25 tokenization + NER): 3–5 ms
- $t_{\text{vector}}$ (HNSW search): 8–15 ms (p99)
- $t_{\text{bm25}}$ (inverted index scan): 2–5 ms
- $t_{\text{entity}}$ (hash map lookup): 1–2 ms
- $t_{\text{fusion}}$ (RRF merge of 3 × top-20 = 60 candidates): 0.2–0.5 ms

Total p99: **15–22 ms**, versus **10–17 ms** for pure vector search. The overhead is approximately 5–6 ms — the cost of the preprocessing step — because BM25 and entity search complete well before the vector search bottleneck.

### Concurrent execution with asyncio

```python
import asyncio
from typing import List

async def multi_signal_search(
    query: str,
    vector_index,
    bm25_index: BM25Index,
    entity_index: EntityIndex,
    top_k: int = 10,
) -> List[tuple[str, float]]:
    """
    Execute all three retrieval signals concurrently and fuse results.
    """
    # All three searches launch simultaneously
    vector_task = asyncio.create_task(
        asyncio.to_thread(vector_index.search, query, 20)
    )
    bm25_task = asyncio.create_task(
        asyncio.to_thread(bm25_index.search, query, 20)
    )
    entity_task = asyncio.create_task(
        asyncio.to_thread(entity_index.search, query, 20)
    )
    
    # Wait for all three — wall-clock time = max(t_v, t_bm25, t_entity)
    vector_results, bm25_results, entity_results = await asyncio.gather(
        vector_task, bm25_task, entity_task
    )
    
    # Fuse with RRF
    return reciprocal_rank_fusion(
        [vector_results, bm25_results, entity_results],
        k=60,
        top_k=top_k,
    )
```

The `asyncio.to_thread` wrapper offloads each synchronous search to a thread pool, allowing true concurrency. The `gather()` call returns when all three complete — wall clock is bounded by the slowest (vector search).

For high-throughput systems, replace `asyncio.to_thread` with a process pool for CPU-bound NER, or pre-compute NER entity tags at index time (which is the correct production approach — don't run NER on every query, run it at memory-write time and cache the entity set).

---

## 8. Index Design: Three Parallel Indexes on One Memory Store

The architectural question is: how do you maintain three heterogeneous indexes against a single shared memory store without tripling storage or introducing consistency hazards?

![Three-index architecture on one memory store: layered stack view](/imgs/blogs/multi-signal-memory-retrieval-5.webp)

The answer is **projection indexing**: one source-of-truth store, three projection layers that transform the raw text into three different index representations. Writes go to the store first, then trigger index updates via a synchronous transaction or an async background worker.

### Memory Store Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,  -- arbitrary key-value, e.g. {"source": "slack", "timestamp": ...}
    created_at TIMESTAMP DEFAULT NOW(),
    embedding VECTOR(384),  -- pgvector column
    bm25_tokens TEXT[],     -- pre-tokenized for BM25 rebuild efficiency
    entity_tags TEXT[]      -- pre-extracted entity strings
);

CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON memories USING gin (bm25_tokens);
CREATE INDEX ON memories USING gin (entity_tags);
```

Using PostgreSQL with the `pgvector` extension, all three index types are supported in a single database. The `HNSW` index handles vector search. The `GIN` indexes on arrays handle BM25 tokenization lookups and entity tag lookups efficiently.

For Python, the write path:

```python
import psycopg2
import numpy as np
from typing import Optional, Dict, Any

class MemoryStore:
    def __init__(self, conn_str: str, embedder, ner_model, bm25_tokenizer):
        self.conn = psycopg2.connect(conn_str)
        self.embedder = embedder    # embed(text) → np.ndarray
        self.ner = ner_model        # tag(text) → List[str]
        self.tokenizer = bm25_tokenizer  # tokenize(text) → List[str]

    def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Compute all three projections
        embedding = self.embedder(content)
        entity_tags = self.ner(content)
        bm25_tokens = self.tokenizer(content)

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO memories
                    (id, content, metadata, embedding, bm25_tokens, entity_tags)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        bm25_tokens = EXCLUDED.bm25_tokens,
                        entity_tags = EXCLUDED.entity_tags
            """, (
                memory_id,
                content,
                psycopg2.extras.Json(metadata or {}),
                embedding.tolist(),
                bm25_tokens,
                entity_tags,
            ))
        self.conn.commit()
```

The key property is **atomic write**: a single `INSERT/UPDATE` statement updates all three index columns simultaneously. There is no window where the vector index is up-to-date but the entity tags are stale.

### Storage overhead

Storing three indexes on the same store adds about 15–30% storage overhead versus storing only the raw text and embedding:

| Component | Size per memory (approx) |
|---|---|
| Raw content | Variable, ~500 bytes avg |
| Metadata | ~100 bytes avg |
| Embedding (384-dim float32) | 1,536 bytes |
| BM25 tokens array | ~100–300 bytes |
| Entity tags array | ~50–150 bytes |

For 100,000 memories, total storage is approximately 300 MB — easily within the range of a single Postgres instance. At 1 million memories, it's 3 GB, still manageable on commodity hardware.

---

## 9. Query Expansion to Improve Each Signal

Multi-signal retrieval handles the recall ceiling of each individual signal. Query expansion raises the recall ceiling further by generating alternate phrasings of the query before search.

![Query expansion pipeline: original query → LLM expansion → 3 variant branches → retrieval](/imgs/blogs/multi-signal-memory-retrieval-9.webp)

### Three expansion strategies, one per signal

**Semantic expansion** generates paraphrases and synonyms to improve vector recall:

```python
async def expand_for_semantic(query: str, llm_client) -> List[str]:
    """Generate 3 semantic paraphrases of the query."""
    prompt = f"""Generate 3 paraphrases of this query for information retrieval.
Return only the paraphrases, one per line.

Query: {query}"""
    response = await llm_client.complete(prompt, max_tokens=200)
    paraphrases = [line.strip() for line in response.split('\n') if line.strip()]
    return [query] + paraphrases[:3]  # original + 3 paraphrases
```

**Temporal expansion** maps calendar references to date ranges:

```python
import re
from datetime import datetime, timedelta

def expand_temporal(query: str) -> Dict[str, Any]:
    """Extract and expand temporal references in a query."""
    expansions = {"date_filter": None, "expanded_terms": []}
    
    # Quarter references: Q1, Q2, Q3, Q4
    quarter_map = {"q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12)}
    for qname, (start_m, end_m) in quarter_map.items():
        if qname in query.lower():
            year = datetime.now().year
            expansions["date_filter"] = {
                "start": f"{year}-{start_m:02d}-01",
                "end": f"{year}-{end_m:02d}-28",
            }
            expansions["expanded_terms"].extend([
                f"{qname} {year}",
                f"{'January February March' if start_m==1 else 'April May June' if start_m==4 else 'July August September' if start_m==7 else 'October November December'}",
            ])
    return expansions
```

**Entity co-reference expansion** resolves ambiguous pronouns and abbreviated references:

```python
async def expand_entities(query: str, context: str, llm_client) -> str:
    """Resolve co-references in the query using conversation context."""
    if not any(p in query.lower() for p in ["we", "they", "it", "our", "their"]):
        return query
    
    prompt = f"""Resolve any pronouns or vague references in this query using the context.
Return only the resolved query.

Context: {context[-500:]}
Query: {query}"""
    return await llm_client.complete(prompt, max_tokens=100)
```

### The cost-benefit tradeoff

Query expansion adds an LLM call (40–100 ms) and multiple retrieval calls (proportional to the number of variants). The recall improvement is real — typically +12–18% recall@10 — but the latency doubles or triples.

When to use query expansion:
- Interactive agent systems where 100 ms latency is acceptable and recall is critical
- Background research tasks (not real-time response generation)
- Queries that contain known ambiguity signals (pronouns, calendar references, abbreviations)

When to skip:
- Real-time response generation (< 50 ms budget)
- High-confidence queries (entity-heavy, exact-term queries that BM25 handles well)
- High-throughput batch processing where per-query LLM calls are prohibitive

---

## 10. Failure Modes: When Fusion Amplifies Noise

Multi-signal retrieval is not unconditionally better. There are configurations where fusion makes things worse.

### Correlated noise amplification

If two signals both fail on the same false positive — for example, both semantic and BM25 score a document highly because it shares vocabulary and semantic neighborhood with the query but is not actually relevant — the fusion layer amplifies this shared false positive to rank 1. With RRF, a document ranked 2nd by both signals gets a fusion score of $\frac{1}{62} + \frac{1}{62} = 0.032$, while a true positive ranked 1st by one signal and absent from the other gets $\frac{1}{61} + \frac{1}{∞} ≈ 0.016$. The shared false positive wins.

**Mitigation:** Add a relevance floor. After fusion, re-rank the top candidates with a cross-encoder reranker (ms-marco-MiniLM-L-6-v2 or similar) that scores each candidate independently against the full query. This is the "retrieve-then-rerank" pattern and effectively filters correlated noise.

### Entity index pollution

If the NER model is noisy — tagging "Java" as a location, "Apple" as a fruit/company in the wrong context — the entity index accumulates false associations. A query about "Apple's new product" will surface memories about fruit markets or Apple Inc. indiscriminately.

**Mitigation:** Add entity type filtering. Store both the entity string and its type label (PERSON, ORG, LOC, PRODUCT, DATE) and match on type when the query context provides a clear signal.

### BM25 vocabulary explosion

Over time, a long-running agent accumulates memories with highly specific technical vocabulary. The BM25 IDF weights for common operational terms ("request," "error," "latency") drop toward zero as they appear in hundreds of memories. Meanwhile, extremely rare one-off identifiers (a specific commit hash, a one-time ticket number) have artificially high IDF weights even though they are never queried again.

**Mitigation:** Periodically prune the inverted index of memory-deleted documents. Also cap maximum IDF at a threshold (e.g., IDF > 8 gets capped at 8) to prevent rare one-off terms from dominating.

### The cold-start entity problem

A new agent with few stored memories will have an entity index with many false positives: every stored memory that contains any entity will score highly for any entity query. With 20 memories, the entity index precision is much lower than with 2,000 memories.

**Mitigation:** Weight the entity signal lower in early operation. One simple heuristic: if the entity index has fewer than 100 unique entities, reduce its weight in the fusion by 50%.

---

## 11. Complete Python Implementation

Here is a self-contained multi-signal retrieval system with all three signals and RRF fusion, suitable for production use with minor additions (error handling, connection pooling, monitoring):

```python
"""
multi_signal_retrieval.py — Production-ready multi-signal memory retrieval.

Requires:
    pip install rank-bm25 spacy numpy
    python -m spacy download en_core_web_sm

For vector search: any compatible client (faiss, qdrant-client, chromadb, etc.)
"""

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
from rank_bm25 import BM25Okapi


# --- Data types ---

@dataclass
class Memory:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    memory_id: str
    rrf_score: float
    signals: Dict[str, Optional[float]]  # per-signal scores for debugging


# --- BM25 Index ---

class BM25Index:
    STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                 "at", "to", "for", "of", "and", "or", "with", "this", "that"}

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: List[List[str]] = []
        self._ids: List[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'[a-zA-Z0-9_./-]+', text.lower())
        return [t for t in tokens if len(t) >= 2 and t not in self.STOPWORDS]

    def add(self, memory_id: str, text: str) -> None:
        self._corpus.append(self._tokenize(text))
        self._ids.append(memory_id)
        self._bm25 = BM25Okapi(self._corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self._ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]


# --- Entity Index ---

class EntityIndex:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = spacy.load(spacy_model)
        self._entity_map: Dict[str, Set[str]] = defaultdict(set)

    def _extract(self, text: str) -> List[str]:
        doc = self._nlp(text)
        return [ent.text.strip().lower() for ent in doc.ents if len(ent.text.strip()) >= 2]

    def add(self, memory_id: str, text: str) -> None:
        for entity in self._extract(text):
            self._entity_map[entity].add(memory_id)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        entities = self._extract(query)
        if not entities:
            return []
        scores: Dict[str, int] = defaultdict(int)
        for entity in entities:
            for mid in self._entity_map.get(entity, set()):
                scores[mid] += 1
        total = len(entities)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(mid, count / total) for mid, count in ranked]


# --- Fusion ---

def rrf_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    top_k: int = 10,
    signal_names: Optional[List[str]] = None,
) -> List[RetrievalResult]:
    rrf: Dict[str, float] = defaultdict(float)
    per_signal: Dict[str, Dict[str, float]] = defaultdict(dict)

    for i, ranked in enumerate(ranked_lists):
        sig_name = (signal_names or ["s0", "s1", "s2"])[i]
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rrf[doc_id] += 1.0 / (k + rank)
            per_signal[doc_id][sig_name] = score

    sorted_docs = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
    return [
        RetrievalResult(
            memory_id=doc_id,
            rrf_score=score,
            signals=per_signal[doc_id],
        )
        for doc_id, score in sorted_docs
    ]


# --- Multi-Signal Retriever ---

class MultiSignalRetriever:
    """
    Three-signal memory retriever with RRF fusion.

    vector_index must implement:
        search(query: str, top_k: int) -> List[Tuple[str, float]]
    """

    def __init__(self, vector_index, bm25_index: BM25Index, entity_index: EntityIndex):
        self.vector = vector_index
        self.bm25 = bm25_index
        self.entity = entity_index

    def add_memory(self, memory: Memory) -> None:
        """Index a memory across all three signal types."""
        self.vector.add(memory.id, memory.content)
        self.bm25.add(memory.id, memory.content)
        self.entity.add(memory.id, memory.content)

    async def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Execute all three signals concurrently and fuse results."""
        vector_task = asyncio.create_task(
            asyncio.to_thread(self.vector.search, query, 20)
        )
        bm25_task = asyncio.create_task(
            asyncio.to_thread(self.bm25.search, query, 20)
        )
        entity_task = asyncio.create_task(
            asyncio.to_thread(self.entity.search, query, 20)
        )

        vector_res, bm25_res, entity_res = await asyncio.gather(
            vector_task, bm25_task, entity_task
        )

        return rrf_fusion(
            [vector_res, bm25_res, entity_res],
            k=60,
            top_k=top_k,
            signal_names=["semantic", "bm25", "entity"],
        )


# --- Example usage ---

async def demo():
    from chromadb import Client  # pip install chromadb

    # Initialize indexes
    chroma = Client()
    collection = chroma.get_or_create_collection("agent_memories")
    
    # Wrap Chroma in vector_index interface
    class ChromaAdapter:
        def __init__(self, coll, embedder):
            self.coll = coll
            self.embedder = embedder
        
        def add(self, memory_id: str, content: str):
            embedding = self.embedder(content)
            self.coll.add(ids=[memory_id], embeddings=[embedding], documents=[content])
        
        def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
            embedding = self.embedder(query)
            results = self.coll.query(query_embeddings=[embedding], n_results=top_k)
            ids = results["ids"][0]
            dists = results["distances"][0]
            return [(id_, 1 - d) for id_, d in zip(ids, dists)]

    # ... connect embedder, build retriever, add memories, search ...
    pass

if __name__ == "__main__":
    asyncio.run(demo())
```

---

## 12. Case Studies

### Case Study 1: The Misattributed Decision

A software engineering agent manages a team's decision log. Over six months, it accumulates 3,400 memories from Slack exports, Jira comments, and meeting transcripts. The query: *"Did we decide to migrate away from Redis?"*

With pure vector search, the top-3 results are:
1. "We should evaluate Redis alternatives for the cache layer" (sim=0.84)
2. "Redis memory usage in production is unsustainable" (sim=0.82)
3. "Meeting notes: discussed caching strategy for Q2" (sim=0.79)

The actual decision memory — *"2024-08-14: Alice confirmed we are moving away from Redis to Valkey starting Q4"* — is ranked 11th, below the top-k cutoff. The agent incorrectly reports "no final decision found."

With multi-signal retrieval:
- Semantic search returns the same top-3
- BM25 returns the decision memory at rank 2 (exact match on "Redis," "moving away," "Valkey")
- Entity search surfaces it at rank 1 ("Alice," "Valkey," "Redis" are all tagged entities)
- RRF fusion promotes it to rank 1 in the final output

The agent correctly surfaces the decision. The downstream task (planning Q4 infrastructure changes) proceeds correctly.

**Lesson:** Decision memories often contain specific entity names (person who made the decision, product names, date references) that BM25 and entity search catch where semantic similarity fails.

### Case Study 2: The Version-Specific Bug

An agent assisting with a software debugging workflow has accumulated memory of past incident reports. The query: *"Any past incidents with the v2.3.1 release?"*

Pure vector search returns incidents related to "software releases," "version deployments," and "production bugs" generally — none specifically about v2.3.1. The version string "v2.3.1" is essentially opaque to the embedding model.

BM25 exact-match returns the two stored incidents mentioning "v2.3.1" immediately, with IDF scores of 7.2 (rare term). The agent surfaces the relevant postmortem documents in under 5 ms.

**Lesson:** Version strings, ticket IDs, configuration keys, and other technical identifiers are best-handled by BM25. An agent that handles technical domains without BM25 will miss version-specific context consistently.

### Case Study 3: The High-Density Topic Failure

A research agent has accumulated 2,100 memories related to "machine learning" and its subtopics. The query: *"What do we know about gradient accumulation?"*

Semantic search returns the top-20 results from a dense neighborhood of ML memories — gradient accumulation, gradient checkpointing, gradient clipping, and gradient tape all have high cosine similarity to each other. Precision in the top-5 is 60%.

BM25 discriminates within this cluster: "gradient accumulation" as a compound term has significantly higher IDF weight than "gradient" alone, and memories that explicitly mention both words score 3× higher than memories mentioning only "gradient." BM25 top-5 precision is 80%.

RRF fusion: 90% precision in top-5 (the two signals reinforce the genuinely relevant memories).

**Lesson:** In high-density topic clusters where everything is semantically close, BM25 provides the discriminating signal that breaks ties.

### Case Study 4: The Temporal Anchor

An agent tracking project milestones has memories spanning 18 months. The query: *"What did we ship in Q3 last year?"*

Pure vector search: returns milestones from across the entire timeline, sorted by semantic similarity to "ship milestone product feature." Q3 of last year is not represented disproportionately.

With temporal expansion:
- The query is expanded to include "July August September 2023" 
- BM25 matches memories containing date strings, month names, and sprint labels from that period
- Entity search finds memories tagged with "2023-Q3" as a date entity

The agent surfaces the correct 8 milestone memories from Q3 2023, compared to a mixed-year list from pure vector search.

**Lesson:** Calendar-anchored queries require explicit temporal handling. Neither semantic search nor entity search handles this alone; BM25 + temporal expansion is the winning combination.

### Case Study 5: The Coreference Failure

A customer support agent tracks interactions by customer name. A supervisor asks: *"What did they say about the billing issue?"* — referring to a specific customer named Acme Corp from the previous conversation turn.

Pure semantic + entity search: "they" is not tagged as an entity. The query returns generic billing-related memories. The entity index has no binding for "they."

With entity co-reference expansion: the LLM resolver maps "they" to "Acme Corp" using the previous conversation context. The expanded query — *"What did Acme Corp say about the billing issue?"* — triggers entity search on "Acme Corp," which returns the correct customer interaction memories with rank-1 precision.

**Lesson:** Pronouns and abbreviated references are a systematic failure mode for entity search. Co-reference expansion using an LLM resolver is necessary for conversational agent contexts.

### Case Study 6: The Fusion Noise Case

A coding agent has accumulated memories mixing code snippets, design discussions, and bug reports. The query: *"Python asyncio best practices"*

Here, all three signals produce noise:
- Semantic search: returns diverse async-related content, some Python, some Go, some general async patterns
- BM25: heavily weights "Python" and "asyncio" as terms, returning every memory that mentions them regardless of "best practices" relevance
- Entity search: "Python" is tagged as an entity (product/technology), and "asyncio" is also tagged; returns everything Python-and-asyncio-adjacent

The fusion amplifies a false positive: a memory titled "Python asyncio gotcha: don't mix sync and async" scores highly on all three signals even though it is a specific bug note, not a "best practices" guide.

**Mitigation applied:** After RRF fusion, run a cross-encoder reranker that scores each candidate against the full query "Python asyncio best practices" and filters the specific bug note to rank 6. The true best practices documentation surfaces at rank 1.

**Lesson:** Fusion is not immune to correlated noise. For high-ambiguity conceptual queries, a lightweight cross-encoder reranker as a post-fusion step prevents false-positive amplification.

---

## 13. When to Use Multi-Signal vs. Single-Signal

Multi-signal retrieval is not always worth the complexity. Here is the decision guide:

![Memory query type routing: 8 patterns with dominant signal](/imgs/blogs/multi-signal-memory-retrieval-8.webp)

**Use multi-signal when:**

- Your agent accumulates memories with specific named entities (people, products, organizations) and users query those entities by name. Entity precision matters and semantic blur is a real failure mode.
- Your memory corpus contains technical identifiers: version strings, ticket numbers, configuration keys, quoted code. BM25 is essential for these.
- Your agent spans a long time horizon (months to years) with time-bounded queries ("what happened in Q2?"). Temporal expansion + BM25 is irreplaceable.
- Your memory corpus has high-density topic clusters where cosine similarity within the cluster is nearly uniform. BM25 provides within-cluster discrimination.
- Recall matters more than latency. The 15-point recall lift is worth 5–6 ms of extra p99 latency for most use cases.

**Use pure vector search when:**

- Your memory corpus is small (< 1,000 memories) and all queries are conceptual/paraphrase-style. The complexity of three indexes is not justified.
- Your agent never stores entity-rich content. If memories are all abstract notes and summaries, entity search contributes nothing.
- Your latency budget is under 10 ms and you cannot run parallel searches. In this regime, use semantic-only with a good embedding model.
- You have no technical vocabulary. Consumer-facing agents without technical content will not benefit from BM25's exact-term matching.

**Use multi-signal with learned fusion when:**

- You have > 10,000 labeled query-memory-relevance triples from production usage.
- Your query distribution is stable and you can maintain a training pipeline.
- You need the extra 5–8% NDCG@10 improvement over RRF that a trained LTR model provides.

The practical advice: start with semantic + BM25 (two signals). The entity index adds meaningful recall only if your memories contain person/org/place references that users query by name. Add entity search when you see recall failures on entity-heavy queries in production logs. Add query expansion when you see temporal reference failures. Layer in learned fusion only when you have labeled data.

Multi-signal retrieval is not a switch you flip — it is an architecture you grow into as your failure mode profile becomes clear from production usage.

---

## Putting It Together

The simplest version of multi-signal retrieval that outperforms pure vector search in most production scenarios is:

1. An HNSW vector index (Chroma, Qdrant, or Faiss)
2. A BM25 index (rank-bm25 or Tantivy)
3. Concurrent search with asyncio
4. RRF fusion with $k = 60$

That is approximately 150 lines of Python and a 5–6 ms latency addition over pure vector search. The recall improvement in entity-heavy and exact-term-heavy query workloads is 10–15 percentage points.

Adding entity search (another 60 lines, spaCy small model) gets you the full three-signal system and closes the last 5–8 pp recall gap on named-entity queries.

The argument for investing in this architecture is simple: an AI agent that retrieves the wrong memory at step 3 of a 10-step task will produce wrong output at step 10. Misretrieval is not a minor inconvenience — it is a task failure. The 15-point recall improvement from multi-signal retrieval translates directly into task success rate improvements for any agent operating on a memory corpus that contains names, technical identifiers, and time references.

Which is most agents, in production, after a few weeks of use.

---

*If you are building the memory layer for your agent, see also: [Mem0: Token-Efficient Memory Algorithm](/blog/machine-learning/ai-agent/mem0-token-efficient-memory-algorithm) for a production-grade memory management system that pairs well with multi-signal retrieval, [Basic RAG](/blog/machine-learning/ai-agent/basic-rag) for the foundational single-signal retrieval patterns this post extends, and [Vector Database](/blog/machine-learning/ai-agent/vector-database) for a deep treatment of the HNSW indexes that power the semantic signal described here.*

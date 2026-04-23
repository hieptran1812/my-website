---
title: "HNSW: The Algorithm Behind Fast Vector Search"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  [
    "hnsw",
    "vector-search",
    "approximate-nearest-neighbor",
    "similarity-search",
    "ai-agent",
    "rag",
    "embeddings",
    "data-structures",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "HNSW (Hierarchical Navigable Small World) is the dominant algorithm for fast approximate nearest neighbor search — powering vector databases like Pinecone, Weaviate, Qdrant, and pgvector. This guide explains the algorithm from first principles, with intuition, math, implementation, and real-world case studies."
---

## The Problem: Finding Needles in a Billion-Dimensional Haystack

![HNSW layered graph structure with greedy descent query path](/imgs/blogs/hnsw-algorithm-diagram.png)

You have a database of 100 million vectors, each with 768 dimensions (say, embeddings from a text encoder). A user sends a query. You need to find the 10 most similar vectors — **in under 10 milliseconds**.

**Brute force** computes the distance between the query and every vector in the database:

$$\text{Time} = N \times d = 100{,}000{,}000 \times 768 = 76.8 \text{ billion operations}$$

Even at 1 TFLOP/s, that's ~77ms — too slow for interactive applications. And this scales linearly: 1 billion vectors = 770ms per query.

**The core challenge**: We need a data structure that trades a tiny amount of accuracy for a massive speedup — returning the **approximate** nearest neighbors (not guaranteed exact) in sublinear time.

HNSW (Hierarchical Navigable Small World) solves this by building a multi-layer graph that supports greedy traversal from coarse to fine resolution, achieving **logarithmic** search complexity with >95% recall.

## Building Intuition: The Airport Analogy

Before diving into the algorithm, let's build intuition with an analogy.

**Imagine you want to travel from Tokyo to a small village in rural France.**

```
Layer 3 (few nodes, long-range connections):
  Major international hubs only
  Tokyo ────────────────── Paris ────────────── New York

Layer 2 (more nodes, medium-range connections):
  Regional airports
  Tokyo ── Seoul ── Paris ── Lyon ── New York ── Chicago

Layer 1 (many nodes, short-range connections):
  Small airports and train stations
  Tokyo ── Seoul ── ... ── Paris ── Lyon ── Grenoble ── ... ── village_station

Layer 0 (all nodes, local connections):
  Every location, connected to nearby locations
  ... ── Grenoble ── small_town_A ── small_town_B ── target_village
```

**The search process**:
1. **Start at the top layer**: Only major hubs exist. Greedily jump from Tokyo to the hub closest to your destination → Paris (few hops, long distance each)
2. **Drop to layer 2**: More nodes available. From Paris, find the closest regional node → Lyon
3. **Drop to layer 1**: Even more nodes. From Lyon, find the closest small station → Grenoble
4. **Drop to layer 0**: All locations. From Grenoble, walk through nearby villages to your exact target

This hierarchical approach is fast because:
- Top layers let you cross large distances in few hops (coarse navigation)
- Bottom layers give you precision (fine navigation)
- You never waste time exploring irrelevant regions (you're already near the target when you reach layer 0)

**HNSW works exactly like this, but with vectors instead of cities.**

## The Two Building Blocks

HNSW combines two ideas: **Skip Lists** (hierarchical structure) and **Navigable Small World graphs** (efficient greedy routing). Let's understand each.

### Building Block 1: Skip Lists

A skip list is a probabilistic data structure for ordered data that supports $O(\log n)$ search:

```
Level 3:  1 ─────────────────────────────── 50 ─────────── 99
Level 2:  1 ────── 15 ────── 35 ────── 50 ────── 75 ── 99
Level 1:  1 ── 8 ── 15 ── 22 ── 35 ── 42 ── 50 ── 63 ── 75 ── 88 ── 99
Level 0:  1  3  5  8  11 15 18 22 27 35 38 42 47 50 55 63 68 75 80 88 93 99
```

To find 42:
1. Start at level 3: 1 → 50 (overshoot!) → back to 1, drop down
2. Level 2: 1 → 15 → 35 → 50 (overshoot!) → back to 35, drop down
3. Level 1: 35 → 42 ✓

Each element appears at level 0 (always), level 1 with probability $p$, level 2 with probability $p^2$, etc. This creates exponentially fewer elements at higher levels.

**HNSW borrows this hierarchy**: each vector has a randomly assigned maximum level. Most vectors exist only at level 0. A few exist at levels 0-1. Very few exist at levels 0-2. And so on.

### Building Block 2: Navigable Small World (NSW) Graphs

An NSW graph connects each vector to its neighbors such that **greedy search** — always moving to the neighbor closest to the target — finds a good approximate nearest neighbor.

```
Greedy search on a single-layer NSW graph:

Query: Q (marked with ✕)

  A ─── B ─── C ─── D
  │     │     │     │
  E ─── F ─── G ─── H
  │     │     │     │
  I ─── J ─── K ─── L
              │     │
              ✕ Q   M

Start at a random entry point (say A).
Step 1: A's neighbors = {B, E}. E is closer to Q → move to E
Step 2: E's neighbors = {A, F, I}. I is closer to Q → move to I
Step 3: I's neighbors = {E, J}. J is closer to Q → move to J
Step 4: J's neighbors = {I, F, K}. K is closer to Q → move to K
Step 5: K's neighbors = {J, G, L, Q_nearest}. → found!
```

**Problem with single-layer NSW**: At the start, the search makes large jumps (good). But if the entry point is far from the target, many hops are needed. The search complexity is $O(N^{1/d} \log N)$ — not great for high-dimensional data.

**HNSW's solution**: Add the skip list hierarchy. Long-range connections exist naturally at higher layers (because fewer nodes are present, so neighbors are farther apart). Short-range connections exist at layer 0.

## The HNSW Algorithm

### Structure

HNSW is a multi-layer graph where:

- **Layer 0** contains **all** $N$ vectors, each connected to its $M$ nearest neighbors
- **Layer 1** contains a **subset** of vectors (each vector's level is drawn from an exponential distribution)
- **Layer 2** contains an even smaller subset
- ...and so on

Each vector at level $l$ is connected to up to $M$ neighbors **at that same level**. The connections at higher levels are naturally longer-range (because fewer nodes exist, so "nearest" neighbors are farther in the original space).

```
Layer 2 (very sparse):
  ●₁ ─────────────────── ●₅₀ ───────────────── ●₉₉
  (long-range jumps)

Layer 1 (sparse):
  ●₁ ──── ●₁₅ ──── ●₃₅ ──── ●₅₀ ──── ●₇₅ ──── ●₉₉
  (medium-range connections)

Layer 0 (all nodes):
  ●₁ ● ● ● ●₈ ● ● ●₁₅ ● ● ● ● ●₂₇ ● ●₃₅ ● ● ● ●₅₀ ● ● ● ●₆₃ ● ●₇₅ ● ● ●₈₈ ● ●₉₉
  (short-range connections, all vectors present)
```

### Level Assignment

Each new vector is assigned a maximum level drawn from an exponential distribution:

$$l = \lfloor -\ln(\text{uniform}(0, 1)) \times m_L \rfloor$$

Where $m_L = 1 / \ln(M)$ is the level multiplier. This ensures:
- ~63% of nodes exist only at level 0
- ~23% exist at levels 0-1
- ~9% exist at levels 0-2
- ~3% exist at levels 0-3
- ...exponential decay

### Search Algorithm

```python
def hnsw_search(graph, query, K, ef):
    """
    Search for K nearest neighbors of query.
    
    graph: the HNSW graph
    query: query vector
    K: number of neighbors to return
    ef: beam width (controls accuracy vs speed trade-off)
    
    Returns: K approximate nearest neighbors
    """
    # Start at the entry point (a node at the top layer)
    current = graph.entry_point
    
    # Phase 1: Greedy search through upper layers (layer L down to layer 1)
    # At each layer, find the single closest node via greedy traversal
    for layer in range(graph.max_level, 0, -1):
        changed = True
        while changed:
            changed = False
            for neighbor in graph.get_neighbors(current, layer):
                if distance(query, neighbor) < distance(query, current):
                    current = neighbor
                    changed = True
    
    # Phase 2: Beam search at layer 0
    # Use a priority queue (beam) of size ef for thorough exploration
    candidates = MinHeap()   # candidates to explore (closest first)
    results = MaxHeap()      # best results so far (farthest first, capped at ef)
    visited = set()
    
    candidates.push(current, distance(query, current))
    results.push(current, distance(query, current))
    visited.add(current)
    
    while len(candidates) > 0:
        # Get closest unexplored candidate
        closest_candidate = candidates.pop()
        
        # Stop if closest candidate is farther than the farthest result
        if distance(query, closest_candidate) > results.peek_max():
            break
        
        # Explore this candidate's neighbors at layer 0
        for neighbor in graph.get_neighbors(closest_candidate, layer=0):
            if neighbor not in visited:
                visited.add(neighbor)
                
                dist = distance(query, neighbor)
                
                # Add to results if better than worst current result
                if dist < results.peek_max() or len(results) < ef:
                    candidates.push(neighbor, dist)
                    results.push(neighbor, dist)
                    
                    if len(results) > ef:
                        results.pop_max()  # keep only ef best
    
    # Return the K closest from our ef candidates
    return results.get_top_k(K)
```

**The `ef` parameter** (expansion factor) is the key accuracy-speed trade-off:
- `ef = K` (minimum): Fast but less accurate. The beam is too narrow to explore all good candidates.
- `ef = 100-500`: Good balance. The beam explores broadly enough to find most true nearest neighbors.
- `ef = N`: Equivalent to brute force (always finds exact nearest neighbors but defeats the purpose).

### Insertion Algorithm

When a new vector is inserted:

1. **Assign a level** $l$ from the exponential distribution
2. **Search for neighbors** at each layer (from top to $l$) using the search algorithm
3. **Connect** the new node to its $M$ nearest neighbors at each layer
4. **Prune** connections if any node exceeds the maximum degree $M_\text{max}$

```python
def hnsw_insert(graph, new_vector):
    """Insert a new vector into the HNSW graph."""
    # Step 1: Assign level
    level = floor(-log(random.uniform(0, 1)) * mL)
    
    # Step 2: Find entry point via greedy search from top
    entry = graph.entry_point
    for l in range(graph.max_level, level, -1):
        entry = greedy_search_single(graph, new_vector, entry, l)
    
    # Step 3: For each layer from level down to 0, find and connect neighbors
    for l in range(min(level, graph.max_level), -1, -1):
        # Find ef_construction nearest neighbors at this layer
        neighbors = search_layer(graph, new_vector, entry, l, ef=ef_construction)
        
        # Select M best neighbors (using heuristic or simple closest)
        selected = select_neighbors(new_vector, neighbors, M)
        
        # Add bidirectional edges
        for neighbor in selected:
            graph.add_edge(new_vector, neighbor, l)
            graph.add_edge(neighbor, new_vector, l)
            
            # Prune neighbor if it now has too many connections
            if graph.degree(neighbor, l) > M_max:
                graph.prune(neighbor, l, M_max)
    
    # Step 4: Update entry point if new node has higher level
    if level > graph.max_level:
        graph.entry_point = new_vector
        graph.max_level = level
```

### Neighbor Selection Heuristic

A critical detail that significantly affects quality. Two approaches:

**Simple**: Select the $M$ closest vectors. Fast but can create poor graphs in clustered data — all neighbors might be in the same cluster, with no "bridges" to other clusters.

**Heuristic (recommended)**: Prefer diverse neighbors. When selecting the next neighbor, reject candidates that are closer to an already-selected neighbor than to the new vector. This creates "bridge" edges between clusters:

```python
def select_neighbors_heuristic(query, candidates, M):
    """Select M neighbors with diversity heuristic."""
    selected = []
    remaining = sorted(candidates, key=lambda c: distance(query, c))
    
    while len(selected) < M and len(remaining) > 0:
        closest = remaining.pop(0)
        
        # Check if this candidate is closer to any already-selected neighbor
        # than to the query itself
        is_useful = True
        for s in selected:
            if distance(closest, s) < distance(closest, query):
                is_useful = False  # too close to an existing neighbor
                break
        
        if is_useful:
            selected.append(closest)
    
    return selected
```

**Why this matters**: Without the heuristic, clusters become internally well-connected but isolated from each other. Queries that need to traverse between clusters will fail (greedy search gets stuck in the wrong cluster). The heuristic forces "bridge" edges that connect clusters, maintaining navigability.

## Key Parameters and Their Effects

| Parameter | Typical Value | Effect of Increase | Trade-off |
|-----------|--------------|-------------------|-----------|
| $M$ (max edges per node) | 16-64 | Better recall, slower search, more memory | Recall vs memory/speed |
| $M_\text{max}$ (max edges at layer 0) | $2 \times M$ | Better recall, more memory | Usually set to 2M |
| $ef_\text{construction}$ | 100-500 | Better graph quality, slower build | Build time vs search quality |
| $ef_\text{search}$ | 50-500 | Better recall, slower search | Accuracy vs latency |
| $m_L$ (level multiplier) | $1/\ln(M)$ | More/fewer layers | Derived from M |

### Parameter Tuning Guide

```
Starting point:
  M = 16 (good for most datasets)
  ef_construction = 200 (invest in build quality)
  ef_search = 100 (adjust based on recall needs)

For higher accuracy (>99% recall):
  M = 32-64
  ef_construction = 400-800
  ef_search = 200-500

For lower memory:
  M = 8-12 (each edge = 4 bytes × 2 directions = 8 bytes per edge)
  Saves ~50% memory vs M=32

For higher dimensions (>256):
  M = 32-48 (higher dimensions need more connections for navigability)
  ef_construction = 400+
```

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Search | $O(\log N)$ average (with high probability) | $O(ef \times M)$ working memory |
| Insert | $O(\log N)$ average | — |
| Build ($N$ insertions) | $O(N \log N)$ | $O(N \times M \times \text{avg\_layers})$ |
| Memory per vector | — | $O(M \times d + M \times \text{pointer})$ |

### Memory Calculation

For $N$ vectors of dimension $d$, with parameter $M$:

$$\text{Memory} \approx N \times d \times 4\text{B} + N \times M \times 2 \times 8\text{B} + \text{overhead}$$

The first term is the vectors themselves (float32). The second term is the graph edges (bidirectional, 8 bytes per pointer/ID).

**Example**: 10M vectors, $d=768$, $M=16$:
- Vectors: $10M \times 768 \times 4 = 30.7$ GB
- Graph: $10M \times 16 \times 2 \times 8 = 2.56$ GB
- **Total: ~33.3 GB** (fits in RAM on a single machine)

## HNSW vs Other ANN Algorithms

| Algorithm | Search Time | Build Time | Memory | Recall@10 | Supports Updates? |
|-----------|------------|-----------|--------|-----------|-------------------|
| **HNSW** | ~1ms | Hours | High (vectors + graph) | 95-99%+ | Yes (insert/delete) |
| **IVF-PQ** | ~0.5-2ms | Hours | Low (compressed) | 85-95% | Partial (rebuild clusters) |
| **ScaNN** | ~0.3-1ms | Hours | Medium | 95-99% | Limited |
| **Annoy** | ~1-5ms | Hours | Medium (trees) | 80-95% | No (immutable) |
| **Brute Force** | ~100ms+ | None | Low (just vectors) | 100% | Yes |
| **LSH** | ~2-10ms | Minutes | Medium | 70-90% | Yes |

**Why HNSW dominates in practice**:
1. **Best recall-speed trade-off**: At 95%+ recall, HNSW is consistently the fastest or near-fastest algorithm across benchmarks (ANN Benchmarks)
2. **Dynamic updates**: Supports incremental insertion without rebuilding the entire index. Critical for production systems where data changes continuously.
3. **Simple tuning**: Only 2-3 parameters to tune (M, ef_construction, ef_search), with clear effects.
4. **No training required**: Unlike IVF (needs to train cluster centroids) or PQ (needs to train product quantization), HNSW is training-free.

**When HNSW is NOT the best choice**:
- **Memory-constrained**: HNSW stores full vectors + graph in memory. IVF-PQ with product quantization can compress vectors 10-50x, fitting much larger datasets in RAM.
- **Billion-scale datasets**: At 1B+ vectors, HNSW's memory becomes prohibitive on a single machine. Disk-based indexes (DiskANN) or distributed approaches (Milvus, Pinecone) are needed.
- **Very high insertion rates**: Each insertion requires searching the graph, which can be slow. Batch insertion with periodic index rebuilding may be faster.

## HNSW in Vector Databases

Every major vector database uses HNSW as a core indexing algorithm:

| Database | HNSW Implementation | Storage | Filtering | Managed? |
|----------|-------------------|---------|-----------|----------|
| **Pinecone** | Custom (proprietary) | Distributed cloud | Metadata filtering | Yes (cloud-only) |
| **Weaviate** | Custom Go implementation | Disk + memory | GraphQL filters | Self-hosted + cloud |
| **Qdrant** | Custom Rust implementation | Disk (mmap) + memory | Rich payload filters | Self-hosted + cloud |
| **Milvus** | Knowhere (C++) | Distributed | Attribute filtering | Self-hosted + cloud |
| **pgvector** | `hnsw` index type | PostgreSQL storage | SQL WHERE clauses | Via PostgreSQL |
| **ChromaDB** | hnswlib (C++) | In-memory / persistent | Metadata filtering | Self-hosted |
| **FAISS** | hnswlib-based | In-memory | Post-filtering | Library only |

### pgvector Example

```sql
-- Create a table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)  -- 768-dimensional vector
);

-- Create HNSW index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Search for nearest neighbors
SET hnsw.ef_search = 100;  -- accuracy-speed trade-off

SELECT id, content, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

### Qdrant Example

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

# Create collection with HNSW configuration
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    ),
    hnsw_config={
        "m": 16,
        "ef_construct": 200,
        "full_scan_threshold": 10000,  # use brute force below this size
    },
)

# Search with metadata filtering
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=10,
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "technical"}},
            {"key": "date", "range": {"gte": "2026-01-01"}},
        ]
    },
)
```

## The Filtering Problem

A major challenge in production: **combining vector search with metadata filters**. You want the 10 nearest neighbors WHERE category="technical" AND date > 2026-01-01.

### Three Approaches

**Pre-filtering**: Filter first, then search only within filtered results.
- Problem: If the filter is very selective (1% of data), the HNSW graph is effectively destroyed — most graph edges point to filtered-out nodes. Search quality collapses.

**Post-filtering**: Search the full index for $K \times \text{oversampling}$ results, then filter.
- Problem: If the filter is very selective, you need to oversample enormously. If only 1% of data matches the filter, you need ~1000x oversampling to get 10 results.

**In-graph filtering (Qdrant, Weaviate approach)**: During HNSW traversal, skip nodes that don't match the filter but continue exploring their neighbors.
- Best approach in practice — maintains graph navigability while respecting filters
- Some quality degradation with very selective filters, but much better than pre/post-filtering

```
Standard HNSW search:
  A → B → C → D → E → F → [result]
  All nodes explored

Filtered HNSW search (filter: category="technical"):
  A(tech) → B(marketing, SKIP) → explore B's neighbors anyway
  → C(tech) → D(tech) → E(finance, SKIP) → F(tech) → [result]
  Skipped nodes don't enter results but their neighbors are still explored
```

## Case Studies

### Case Study 1: Spotify — Music Recommendation at Scale

**Problem**: Spotify has 100M+ tracks, each represented as a 256-dimensional embedding (capturing audio features, listener behavior, and metadata). When a user listens to a song, Spotify needs to find similar tracks for the "Radio" feature — in real-time.

**Solution**: HNSW-based approximate nearest neighbor search on track embeddings.

**Architecture**:
```
User plays "Bohemian Rhapsody" by Queen
    ↓
[Get track embedding] → 256-dim vector
    ↓
[HNSW search on 100M track index] → Top 500 similar tracks
    ↓
[Rerank with user-specific model] → Remove tracks already heard
    ↓
[Serve top 50 as Radio playlist]

Search latency: <5ms for 500 candidates from 100M tracks
```

**Why HNSW**: Dynamic updates are critical — Spotify adds ~60,000 new tracks per day. HNSW supports incremental insertion without rebuilding the index. IVF-PQ would require periodic re-clustering.

**Tuning**: $M=32$ (higher than typical because audio embeddings benefit from more connections), $ef_\text{search}=200$ (quality matters more than latency for recommendations — users don't notice 5ms vs 2ms).

### Case Study 2: RAG Pipeline for Enterprise Search

**Problem**: A company has 5 million internal documents (wikis, Slack messages, Jira tickets, Google Docs). Employees ask natural language questions and need relevant documents retrieved in <100ms for an LLM to answer.

**Solution**: Document embeddings + HNSW index + metadata filtering.

**Architecture**:
```
Offline (indexing):
  Documents → [Chunk into 512-token passages] → 20M passages
  Each passage → [Embedding model (e5-large)] → 1024-dim vector
  Vectors → [HNSW index in Qdrant] with metadata (source, date, author, team)

Online (query):
  User: "What's our policy on remote work for contractors?"
    ↓
  [Embed query] → 1024-dim vector
    ↓
  [HNSW search, filter: source IN ('wiki', 'policy-docs'), limit 20]
  Latency: ~8ms for 20 results from 20M vectors
    ↓
  [Rerank with cross-encoder] → Top 5 passages
    ↓
  [LLM generates answer with citations]
```

**Key decisions**:
- **HNSW parameters**: $M=16$, $ef_\text{construction}=200$, $ef_\text{search}=128$ — 97% recall is sufficient because the reranker corrects most HNSW misses
- **Filtering**: In-graph filtering by source type and team — employees only see documents they have access to
- **Memory**: 20M × 1024 × 4B = 82GB for vectors + ~5GB for graph = ~87GB. Fits on a single machine with 128GB RAM

### Case Study 3: E-Commerce Visual Search (ASOS)

**Problem**: Users take a photo of clothing they like and want to find similar items in a catalog of 85,000+ products, each with multiple images (~500K total images).

**Solution**: Visual embeddings + HNSW with product-level deduplication.

**Architecture**:
```
User uploads photo
    ↓
[CLIP/SigLIP visual encoder] → 512-dim embedding
    ↓
[HNSW search on 500K image index] → Top 100 similar images
    ↓
[Deduplicate by product_id] → Top 20 unique products
    ↓
[Filter by availability, size, price range]
    ↓
[Display to user]

Search latency: <3ms (500K vectors is small — fits entirely in L3 cache)
```

**Challenge**: The same dress photographed from different angles produces different embeddings. The index must return the **product**, not just the most similar single image. Solution: search for top 100 images (overcollect), then deduplicate by product ID to get 20 unique products.

### Case Study 4: Anomaly Detection in Cybersecurity

**Problem**: A security platform processes 50M network event logs per day. Each log is embedded as a 128-dim vector capturing network behavior patterns. The system must detect anomalous events (potential attacks) by finding events that are far from any cluster of normal behavior.

**Solution**: Maintain an HNSW index of "normal" behavior embeddings. For each new event, query the index — if the nearest neighbor distance exceeds a threshold, flag as anomalous.

```
New network event → [Embed to 128-dim]
    ↓
[HNSW search: find 5 nearest neighbors in "normal" index]
    ↓
Average distance to 5 nearest neighbors = d_avg
    ↓
if d_avg > threshold: FLAG AS ANOMALY
else: NORMAL (add to index for future reference)
```

**Why HNSW works well here**: The index grows continuously (50M new events/day). HNSW's dynamic insertion support is critical — IVF-PQ would need daily rebuilding. The $O(\log N)$ search time scales well as the index grows to billions of events over months.

**Tuning**: $M=8$ (low, to save memory at billion-scale), $ef_\text{search}=50$ (anomaly detection is tolerant of approximate results — a true anomaly is far from ALL neighbors, so missing the absolute nearest one doesn't matter).

### Case Study 5: GitHub Copilot — Code Search at Scale

**Problem**: GitHub indexes billions of code snippets for Copilot's retrieval-augmented generation. When a developer types code, Copilot retrieves similar code patterns from the index to improve suggestions.

**Solution**: Code embeddings (from a fine-tuned code encoder) indexed with HNSW, distributed across many machines.

**Challenges at GitHub scale**:
- **Billions of vectors**: Can't fit on a single machine. Solution: shard the index across 100+ machines, each holding a partition of the data with its own HNSW index. Query all shards in parallel, merge results.
- **Freshly pushed code**: New code is pushed to GitHub every second. Solution: a small "hot" HNSW index for recent code (rebuilt hourly) + a large "cold" index for historical code (rebuilt daily).
- **Latency requirements**: <20ms end-to-end including network. Solution: keep HNSW graphs in memory (not memory-mapped from disk), use $ef_\text{search}=64$ (speed over perfect recall, since the LLM is tolerant of imperfect retrieval).

## Common Pitfalls

### 1. Forgetting to Normalize Vectors

HNSW with cosine distance assumes vectors are **unit-normalized**. If you forget to normalize, L2 distance is computed instead, producing wrong results for cosine similarity use cases.

```python
# WRONG — vectors not normalized
index.add(embedding)

# RIGHT — normalize before inserting
embedding = embedding / np.linalg.norm(embedding)
index.add(embedding)

# Or use the database's built-in cosine distance operator
```

### 2. Setting ef_search Too Low

The default `ef_search` in many libraries is low (e.g., 40 in hnswlib). This sacrifices significant recall for speed. Always benchmark recall at your target latency:

```python
# Benchmark recall at different ef_search values
for ef in [10, 20, 50, 100, 200, 500]:
    index.set_ef(ef)
    recall = compute_recall(index, queries, ground_truth, k=10)
    latency = measure_latency(index, queries, k=10)
    print(f"ef={ef}: recall@10={recall:.3f}, latency={latency:.1f}ms")

# Typical output:
# ef=10:  recall@10=0.72, latency=0.3ms
# ef=50:  recall@10=0.93, latency=0.8ms  
# ef=100: recall@10=0.97, latency=1.5ms  ← good balance
# ef=200: recall@10=0.99, latency=3.0ms
# ef=500: recall@10=0.999, latency=7.5ms
```

### 3. Not Accounting for Memory

HNSW keeps the full vectors + graph in memory. At 100M × 768-dim float32, that's **~307 GB just for vectors**. Plan your infrastructure accordingly or use quantized vectors (float16 halves memory, int8 quarters it).

### 4. Ignoring Build Time

Building an HNSW index for 100M vectors can take **hours**. Plan for this in your pipeline — don't try to build the index during a deployment.

## Interview Questions and Answers

### Q: What is HNSW and why is it the dominant ANN algorithm?

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor search algorithm. It builds a multi-layer graph where higher layers have fewer nodes with long-range connections (for coarse navigation) and lower layers have all nodes with short-range connections (for fine navigation). Search starts at the top layer, greedily descends to find a good entry point, then performs a beam search at layer 0.

It dominates because: (1) best recall-speed trade-off on ANN benchmarks (95%+ recall at <1ms for millions of vectors), (2) supports dynamic insertion/deletion without rebuilding, (3) simple to tune (2-3 parameters), (4) no training phase needed (unlike IVF-PQ which needs cluster centroids).

### Q: Explain the search algorithm step by step.

**Phase 1 — Coarse search (layers L to 1)**: Starting from the entry point at the top layer, perform greedy traversal — always move to the neighbor closest to the query. At each layer, the closest node found becomes the entry point for the next layer down. This quickly narrows to the right "region" of the space.

**Phase 2 — Fine search (layer 0)**: Perform beam search with width `ef`. Maintain a priority queue of candidates and a results set. Explore candidates in order of distance to query. For each candidate, check all its neighbors — add them to candidates/results if closer than the current worst result. Stop when the closest remaining candidate is farther than the farthest result. Return the top K from the ef results.

The `ef` parameter controls the accuracy-speed trade-off: higher ef explores more candidates (better recall, slower), lower ef explores fewer (worse recall, faster).

### Q: How does HNSW handle the insertion of new vectors?

1. Assign a random level $l$ from an exponential distribution ($l = \lfloor -\ln(\text{rand}) \times m_L \rfloor$). Most nodes get level 0, fewer get level 1, even fewer get level 2, etc.
2. Search the graph for the closest node at each layer from top down to level $l$ (greedy search, same as the search algorithm).
3. At each layer from $l$ down to 0, find the $M$ best neighbors using the construction search (with `ef_construction` beam width).
4. Add bidirectional edges between the new node and its selected neighbors.
5. If any existing node now exceeds $M_\text{max}$ connections, prune its least useful edges.
6. If the new node's level exceeds the current max level, it becomes the new entry point.

### Q: What is the neighbor selection heuristic and why does it matter?

The simple approach selects the $M$ closest vectors as neighbors. The heuristic approach adds diversity: when selecting the next neighbor, it rejects candidates that are closer to an already-selected neighbor than to the new vector.

This matters because without diversity, nodes in the same cluster all connect to each other but have no "bridge" edges to other clusters. Greedy search gets stuck in the wrong cluster. The heuristic forces bridge connections, maintaining global navigability. This can improve recall by 5-15% on clustered datasets with no speed penalty.

### Q: How do you handle metadata filtering with HNSW?

Three approaches: **pre-filtering** (filter data, then search — breaks graph navigability with selective filters), **post-filtering** (search, then filter — requires massive oversampling), and **in-graph filtering** (during traversal, skip non-matching nodes but continue exploring their neighbors).

In-graph filtering is the best production approach (used by Qdrant, Weaviate). It maintains graph navigability because the algorithm still traverses filtered-out nodes' edges — it just doesn't include them in results. Quality degrades with very selective filters (<1% match rate) because too many nodes are skipped, but it's much better than pre/post-filtering.

### Q: Calculate the memory requirements for 50M vectors of dimension 1024 with M=16.

Vectors: $50M \times 1024 \times 4\text{B (float32)} = 204.8$ GB

Graph edges: Average node degree ≈ $M \times \text{avg\_layers} \approx 16 \times 1.2 = 19.2$ edges per node. Each edge is a 4-byte integer ID. Bidirectional: $50M \times 19.2 \times 2 \times 4\text{B} = 7.68$ GB.

Overhead (level info, metadata pointers): ~$50M \times 32\text{B} = 1.6$ GB.

**Total: ~214 GB.** This requires a machine with at least 256GB RAM for comfortable operation. Alternatives: use float16 vectors (halves to ~110GB), product quantization (compress vectors 4-8x), or distribute across multiple machines.

### Q: When would you NOT use HNSW?

1. **Memory-constrained environments**: HNSW requires all vectors + graph in memory. For 1B+ vectors at high dimensions, this can exceed 1TB. Use IVF-PQ (product quantization compresses vectors 10-50x) or DiskANN (SSD-based).
2. **Exact nearest neighbor required**: HNSW is approximate. If you need guaranteed exact results (legal, financial applications), use brute force or exact tree-based methods on smaller datasets.
3. **Very small datasets** (<10K vectors): Brute force is faster because the overhead of graph traversal exceeds the benefit. Most databases automatically fall back to brute force below a threshold.
4. **Write-heavy workloads**: Frequent bulk deletions degrade graph quality (orphaned edges). Periodic rebuilding may be needed, which is expensive.
5. **Extremely high dimensions** (>2000): The "curse of dimensionality" means distances between vectors converge, making nearest neighbor less meaningful. Dimensionality reduction before indexing often helps.

## References

1. Malkov, Y., & Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Using Hierarchical Navigable Small World Graphs." IEEE TPAMI, 2020.
2. [hnswlib — C++ Header-Only HNSW Implementation](https://github.com/nmslib/hnswlib)
3. [ANN Benchmarks — Algorithm Comparison](http://ann-benchmarks.com/)
4. [Pinecone — Understanding HNSW](https://www.pinecone.io/learn/hnsw/)
5. [Qdrant — HNSW Index Documentation](https://qdrant.tech/documentation/concepts/indexing/)
6. [pgvector — HNSW Index](https://github.com/pgvector/pgvector)
7. Jayaram Subramanya, S., et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS 2019.
8. Johnson, J., Douze, M., & Jégou, H. "Billion-Scale Similarity Search with GPUs (FAISS)." IEEE TBD, 2021.

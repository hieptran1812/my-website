---
title: "Vector Databases: Algorithms, Architecture, and Comparison"
publishDate: "2026-03-12"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  [
    "vector-database",
    "similarity-search",
    "ai-agent",
    "rag",
    "embeddings",
  ]
date: "2026-03-12"
author: "Hiep Tran"
featured: false
image: ""
excerpt: "A comprehensive guide to vector databases — the backbone of modern AI applications. Covering indexing algorithms (HNSW, IVF, PQ), distance metrics, and an in-depth comparison of Pinecone, Weaviate, Milvus, Qdrant, Chroma, and more."
---

## What is a Vector Database?

A vector database is a specialized database designed to store, index, and query high-dimensional vector embeddings efficiently. Unlike traditional databases that operate on structured rows and columns, vector databases are optimized for similarity search, that means finding the closest vectors to a given query vector.

Vector databases are the backbone of modern AI applications, including:

- **Retrieval-Augmented Generation (RAG)** — grounding LLM responses with relevant documents
- **Semantic search** — finding results by meaning rather than keywords
- **Recommendation systems** — matching users to similar items
- **Image/audio retrieval** — finding similar media based on embeddings
- **Anomaly detection** — identifying outliers in high-dimensional space

## How It Works

```
Raw Data → Embedding Model → Vector [0.12, -0.34, 0.78, ...] → Vector Database
                                                                      ↓
Query → Embedding Model → Query Vector → Similarity Search → Top-K Results
```

## Embedding Representations

Before diving into algorithms, it's important to understand what vectors represent.

An embedding is a dense, fixed-length numerical representation of data (text, images, audio) produced by a neural network. Popular embedding models include:

| Data Type  | Models                                                     | Dimensions |
| ---------- | ---------------------------------------------------------- | ---------- |
| Text       | OpenAI `text-embedding-3-large`, Cohere `embed-v4`, BGE-M3 | 768–3072   |
| Images     | CLIP, SigLIP, DINOv2                                       | 512–1024   |
| Audio      | Whisper, CLAP                                              | 512–768    |
| Multimodal | ImageBind, ONE-PEACE                                       | 1024       |

## Distance Metrics

The choice of distance metric directly affects retrieval quality. The three most common metrics:

### Euclidean Distance (L2)

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

- Measures the straight-line distance between two points
- Best for: dense embeddings where magnitude matters
- Smaller = more similar

### Cosine Similarity

$$\text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$$

- Measures the angle between two vectors, ignoring magnitude
- Best for: normalized text embeddings
- Larger = more similar (range: -1 to 1)

### Inner Product (Dot Product)

$$\text{IP}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} a_i \cdot b_i$$

- Combines direction and magnitude
- Best for: Maximum Inner Product Search (MIPS), recommendation systems
- Larger = more similar

### When to Use Which?

| Metric      | Use When                                             | Common Models                     |
| ----------- | ---------------------------------------------------- | --------------------------------- |
| Cosine      | Embeddings are normalized or you want direction-only | Most text embedding models        |
| L2          | Magnitude matters, raw embeddings                    | Image embeddings, scientific data |
| Dot Product | Embeddings encode relevance via magnitude            | Recommendation, MIPS tasks        |

## 4. Indexing Algorithms

The core challenge: **exact nearest neighbor search** in high dimensions is $O(n \cdot d)$ — too slow for millions of vectors. Vector databases use **Approximate Nearest Neighbor (ANN)** algorithms that trade a small amount of accuracy for massive speed gains.

### 4.1 Flat Index (Brute Force)

- Compares the query against every vector in the database
- **100% recall** — always finds the true nearest neighbor
- Time complexity: $O(n \cdot d)$
- Only practical for small datasets (< 50K vectors)

### 4.2 IVF (Inverted File Index)

**Idea**: Partition the vector space into clusters using k-means, then only search the closest clusters at query time.

**Training Phase**:

1. Run k-means to create `nlist` centroids (e.g., 1024)
2. Assign each vector to its nearest centroid

**Query Phase**:

1. Find the `nprobe` closest centroids to the query vector
2. Search only vectors in those clusters

```
Parameters:
- nlist: number of clusters (higher = more partitions, faster search)
- nprobe: number of clusters to search (higher = better recall, slower)
```

**Tradeoffs**:

- ✅ Simple, well-understood, works well with quantization (IVF-PQ)
- ❌ Requires training, cluster boundaries can miss relevant vectors
- Recall vs Speed: tunable via `nprobe`

### 4.3 HNSW (Hierarchical Navigable Small World)

The most popular ANN algorithm in production vector databases.

**Idea**: Build a multi-layer graph where each layer is a navigable small-world network. Higher layers have fewer nodes and longer edges (for fast coarse navigation), lower layers have more nodes and shorter edges (for precise local search).

**Construction**:

1. Each new vector is assigned a random maximum layer $l$ (exponential distribution)
2. Starting from the top layer, greedily navigate to the closest node
3. At layer $l$ and below, connect the new node to its `M` nearest neighbors

**Query**:

1. Start at the top layer entry point
2. Greedily descend through layers
3. At the bottom layer, perform a beam search with width `ef`

```
Parameters:
- M: max edges per node (higher = better recall, more memory)
- ef_construction: beam width during build (higher = better graph quality)
- ef_search: beam width during query (higher = better recall, slower)
```

**Tradeoffs**:

- ✅ Excellent recall-speed tradeoff, no training required, incremental inserts
- ❌ High memory usage ($O(n \cdot M \cdot d)$), not ideal for billion-scale
- State of the art for most use cases under ~100M vectors

### 4.4 Product Quantization (PQ)

**Idea**: Compress vectors by splitting them into subvectors and quantizing each independently.

**Process**:

1. Split each $d$-dimensional vector into $m$ subvectors of $d/m$ dimensions
2. Train a k-means codebook (typically $k = 256$) for each subvector
3. Replace each subvector with its nearest centroid ID (1 byte)

**Result**: A 768-dim float32 vector (3072 bytes) → 96 bytes with $m = 96$ subvectors

**Tradeoffs**:

- ✅ Massive compression (10-50x), enables billion-scale search
- ❌ Lossy — reduces recall, requires training
- Often combined with IVF → **IVF-PQ** (the workhorse of billion-scale search)

### 4.5 Scalar Quantization (SQ)

Simpler than PQ: quantize each dimension from float32 to int8 (or int4).

- 4x compression with minimal recall loss
- No training required
- Often used as a first optimization step

### 4.6 DiskANN

**Idea**: Store the graph index on SSD instead of RAM, enabling billion-scale search on commodity hardware.

**Key innovations**:

- Vamana graph (single-layer, degree-bounded graph optimized for SSD)
- PQ-compressed vectors in RAM for distance estimation
- Full vectors on SSD for re-ranking

**Tradeoffs**:

- ✅ Billion-scale on a single machine with 64GB RAM + SSD
- ❌ Higher latency than pure in-memory solutions
- Used by Microsoft Bing, Azure AI Search

### 4.7 ScaNN (Google)

Google's ANN library combining:

- Anisotropic vector quantization (better than standard PQ)
- Asymmetric hashing
- Reordering for final precision

### Algorithm Comparison

| Algorithm | Recall@10 | QPS (1M vectors) | Memory               | Build Time | Incremental Insert |
| --------- | --------- | ---------------- | -------------------- | ---------- | ------------------ |
| Flat      | 100%      | ~1K              | $O(n \cdot d)$       | None       | ✅                 |
| IVF       | 95-99%    | ~10K             | $O(n \cdot d)$       | Minutes    | ❌ (retrain)       |
| HNSW      | 98-99.9%  | ~10K-50K         | $O(n \cdot (d + M))$ | Minutes    | ✅                 |
| IVF-PQ    | 90-95%    | ~50K-100K        | $O(n \cdot m)$       | Minutes    | ❌                 |
| DiskANN   | 95-99%    | ~5K-10K          | $O(n \cdot m)$ RAM   | Hours      | Partial            |

## 5. Vector Database Comparison

### 5.1 Overview of Major Vector Databases

| Database          | Type             | Language    | Open Source     | Cloud Managed     | First Release       |
| ----------------- | ---------------- | ----------- | --------------- | ----------------- | ------------------- |
| **Pinecone**      | Purpose-built    | C++/Rust    | ❌              | ✅                | 2021                |
| **Weaviate**      | Purpose-built    | Go          | ✅ (BSD-3)      | ✅                | 2019                |
| **Milvus**        | Purpose-built    | Go/C++      | ✅ (Apache 2.0) | ✅ (Zilliz)       | 2019                |
| **Qdrant**        | Purpose-built    | Rust        | ✅ (Apache 2.0) | ✅                | 2021                |
| **Chroma**        | Purpose-built    | Python/Rust | ✅ (Apache 2.0) | ✅                | 2022                |
| **pgvector**      | Extension        | C           | ✅ (PostgreSQL) | ✅ (via PG hosts) | 2021                |
| **Elasticsearch** | General + vector | Java        | ✅ (SSPL)       | ✅                | 2010 (vector: 2022) |
| **LanceDB**       | Purpose-built    | Rust        | ✅ (Apache 2.0) | ✅                | 2023                |

### 5.2 Detailed Comparison

#### Pinecone

- **Pros**: Fully managed, zero-ops, extremely easy to get started, strong enterprise features, serverless pricing model
- **Cons**: Proprietary (no self-hosting), can be expensive at scale, limited control over indexing
- **Index**: Proprietary (likely IVF-PQ + graph-based)
- **Max Dimensions**: 20,000
- **Filtering**: Metadata filtering with single-stage approach
- **Best For**: Teams that want managed infrastructure with minimal DevOps

#### Weaviate

- **Pros**: Built-in vectorization modules (OpenAI, Cohere, HuggingFace), GraphQL API, hybrid search (BM25 + vector), multi-tenancy
- **Cons**: Higher memory consumption, Go ecosystem less familiar for ML teams
- **Index**: HNSW (with PQ compression), Flat
- **Max Dimensions**: 65,535
- **Filtering**: Pre-filtering with inverted index (efficient)
- **Best For**: Applications needing built-in embedding generation and hybrid search

#### Milvus

- **Pros**: Battle-tested at massive scale (billions of vectors), GPU indexing support, rich index options (IVF, HNSW, DiskANN, ScaNN), strong distributed architecture
- **Cons**: Complex deployment (requires etcd, MinIO, Pulsar), steeper learning curve
- **Index**: HNSW, IVF-Flat, IVF-PQ, IVF-SQ8, DiskANN, GPU indexes, ScaNN
- **Max Dimensions**: 32,768
- **Filtering**: Attribute filtering with partitioning
- **Best For**: Large-scale production deployments requiring flexibility and GPU acceleration

#### Qdrant

- **Pros**: Written in Rust (fast, memory-efficient), excellent filtering performance, payload indexing, quantization options (scalar, product, binary), gRPC API
- **Cons**: Smaller community than Milvus/Weaviate, fewer built-in integrations
- **Index**: HNSW (with quantization variants)
- **Max Dimensions**: 65,535
- **Filtering**: Advanced filtering with payload indexes (very efficient)
- **Best For**: Performance-critical applications with complex filtering requirements

#### Chroma

- **Pros**: Extremely easy to use, great Python API, embedded mode (no server needed), perfect for prototyping, LangChain/LlamaIndex native integration
- **Cons**: Not yet mature for large-scale production, limited distributed capabilities
- **Index**: HNSW (via hnswlib)
- **Max Dimensions**: Limited by memory
- **Filtering**: Metadata filtering
- **Best For**: Prototyping, small-to-medium RAG applications, local development

#### pgvector

- **Pros**: Use existing PostgreSQL infrastructure, SQL interface, ACID transactions, joins with relational data, familiar tooling
- **Cons**: Slower than purpose-built solutions, limited to single-node, HNSW build can be slow
- **Index**: IVFFlat, HNSW
- **Max Dimensions**: 2,000 (HNSW), 16,000 (IVFFlat)
- **Filtering**: Full SQL WHERE clauses — the most flexible filtering
- **Best For**: Teams already using PostgreSQL who don't want another database

#### LanceDB

- **Pros**: Serverless (no server process needed), built on Lance columnar format, disk-based (low memory), multi-modal support, versioning
- **Cons**: Newer project, smaller community, fewer production references
- **Index**: IVF-PQ, Flat (DiskANN in progress)
- **Max Dimensions**: No hard limit
- **Filtering**: SQL-like filtering with Lance
- **Best For**: Cost-sensitive applications, multi-modal data, embedded/edge deployments

### 5.3 Performance Benchmarks

Based on the [ANN Benchmarks](https://ann-benchmarks.com/) and various community benchmarks:

| Database | QPS (1M, 768d, recall@95%) | Latency p99 | Memory/1M vectors | Filtering Overhead |
| -------- | -------------------------- | ----------- | ----------------- | ------------------ |
| Pinecone | ~5K-10K                    | ~10ms       | Managed           | Low                |
| Weaviate | ~3K-8K                     | ~15ms       | ~4-6 GB           | Medium             |
| Milvus   | ~5K-15K                    | ~8ms        | ~3-5 GB           | Medium             |
| Qdrant   | ~8K-20K                    | ~5ms        | ~2-4 GB           | Low                |
| Chroma   | ~1K-3K                     | ~20ms       | ~4-6 GB           | Medium             |
| pgvector | ~500-2K                    | ~30ms       | ~4-5 GB           | High (SQL)         |
| LanceDB  | ~2K-5K                     | ~15ms       | ~1-2 GB (disk)    | Low                |

> **Note**: Benchmarks vary significantly based on hardware, dataset, dimensionality, and configuration. Always benchmark with your own data and queries.

### 5.4 Feature Comparison Matrix

| Feature            | Pinecone             | Weaviate             | Milvus                   | Qdrant               | Chroma     | pgvector         |
| ------------------ | -------------------- | -------------------- | ------------------------ | -------------------- | ---------- | ---------------- |
| Hybrid Search      | ✅                   | ✅                   | ✅                       | ✅ (sparse vectors)  | ❌         | ❌               |
| Multi-tenancy      | ✅                   | ✅                   | ✅                       | ✅                   | ✅         | Via schemas      |
| GPU Index          | ❌                   | ❌                   | ✅                       | ❌                   | ❌         | ❌               |
| Disk-based Index   | ✅                   | ❌                   | ✅ (DiskANN)             | ❌                   | ❌         | ❌               |
| Built-in Embedding | ✅                   | ✅                   | ❌                       | ❌                   | ✅         | ❌               |
| RBAC               | ✅                   | ✅                   | ✅                       | ✅                   | ❌         | ✅ (PostgreSQL)  |
| Backup/Restore     | ✅                   | ✅                   | ✅                       | ✅                   | ❌         | ✅ (pg_dump)     |
| Replication        | ✅                   | ✅                   | ✅                       | ✅                   | ❌         | ✅ (PG replicas) |
| SDKs               | Python, JS, Go, Java | Python, JS, Go, Java | Python, JS, Go, Java, C# | Python, JS, Go, Rust | Python, JS | Any PG driver    |

## 6. Choosing the Right Vector Database

### Decision Framework

```
Start here:
│
├─ Prototyping / Small scale (< 100K vectors)?
│   └─ → Chroma (easiest) or LanceDB (serverless)
│
├─ Already using PostgreSQL?
│   └─ → pgvector (add extension, no new infra)
│
├─ Need managed service, minimal ops?
│   └─ → Pinecone (fully managed) or Weaviate Cloud
│
├─ High performance + complex filtering?
│   └─ → Qdrant (best filter performance)
│
├─ Massive scale (billions) + GPU?
│   └─ → Milvus (most scalable, GPU indexes)
│
├─ Need hybrid search (keyword + vector)?
│   └─ → Weaviate or Milvus
│
└─ Budget-constrained, disk-based?
    └─ → LanceDB (serverless, disk-native)
```

### Common RAG Stack Patterns

| Stack           | Vector DB                  | Why                        |
| --------------- | -------------------------- | -------------------------- |
| Quick prototype | Chroma                     | Zero config, embedded mode |
| Startup MVP     | Qdrant / Weaviate Cloud    | Managed, good free tier    |
| Enterprise      | Pinecone / Milvus (Zilliz) | SLAs, compliance, scale    |
| PostgreSQL shop | pgvector                   | No new infrastructure      |
| Cost-optimized  | LanceDB + S3               | Serverless, pay-per-query  |

## 7. Best Practices

1. **Normalize embeddings** if using cosine similarity — many models output unnormalized vectors
2. **Benchmark with your data** — synthetic benchmarks don't reflect real workloads
3. **Use metadata filtering** to reduce the search space before ANN search
4. **Choose dimensions wisely** — 768-1024 is the sweet spot; higher dims have diminishing returns
5. **Quantize for scale** — scalar quantization (int8) gives 4x compression with minimal recall loss
6. **Hybrid search** improves recall for keyword-sensitive queries (e.g., product names, codes)
7. **Re-rank results** with a cross-encoder for higher precision in RAG applications
8. **Monitor recall** — set up evaluation pipelines to track retrieval quality over time
9. **Batch inserts** — upsert in batches of 100-1000 for optimal throughput
10. **Use namespaces/collections** to isolate different data types or tenants

## Deep Dive: How Filtering Actually Works

Filtering is the hidden complexity of vector databases. You rarely want "the 10 nearest vectors" — you want "the 10 nearest vectors WHERE category='technical' AND date > 2025-01-01." This is deceptively hard.

### The Three Strategies

**Pre-filtering**: Apply metadata filter first, then run ANN search on the filtered subset.

```
All vectors (10M) → [Filter: category='technical'] → 500K vectors → [HNSW search] → Top 10
```

Problem: If the filter is very selective (e.g., 0.1% of data), the HNSW graph is effectively destroyed — most edges point to filtered-out nodes. The search degenerates to near-random. Recall can drop from 95% to 30%.

**Post-filtering**: Run ANN search on the full index, then filter results.

```
All vectors (10M) → [HNSW search, fetch 1000 candidates] → [Filter: category='technical'] → Top 10
```

Problem: If the filter is selective, you need enormous oversampling. If only 1% of data matches, fetching 1000 candidates yields ~10 matches — barely enough. Fetching 10,000 candidates is slow and wasteful.

**In-graph filtering** (Qdrant, Weaviate approach): During HNSW traversal, check the filter at each node. Skip non-matching nodes but still explore their neighbors.

```
HNSW traversal:
  Visit node A (category='technical') → ADD to results
  Visit node B (category='marketing') → SKIP, but explore B's neighbors
  Visit node C (category='technical') → ADD to results
  ...
```

This is the best general approach because it maintains graph navigability while respecting filters. Qdrant indexes metadata in **payload indexes** (like B-tree indexes for JSON fields) for fast filter evaluation at each hop.

### Hybrid Search: Combining Sparse and Dense

Hybrid search combines traditional keyword search (BM25/TF-IDF — sparse vectors) with semantic search (dense embeddings). This handles cases where one alone fails:

```
Query: "error code E-4502 in payment module"

Dense search alone:   Finds semantically similar documents about payment errors
                      but misses the exact error code "E-4502"

Sparse search alone:  Finds documents containing "E-4502" literally
                      but misses paraphrased descriptions

Hybrid:               Finds both — documents about E-4502 AND semantically
                      related payment error documentation
```

Fusion strategies:
- **Reciprocal Rank Fusion (RRF)**: $\text{score}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$ where $k = 60$ typically
- **Weighted linear combination**: $\text{score} = \alpha \cdot \text{dense\_score} + (1-\alpha) \cdot \text{sparse\_score}$
- **Learned fusion**: Train a model to combine scores (most complex, best results)

Weaviate, Milvus, and Pinecone all support hybrid search natively. pgvector can achieve it by combining `tsvector` (full-text search) with `vector` (ANN search) in the same query.

## Production Architecture Patterns

### Pattern 1: Single-Node (< 10M vectors)

```
Application → Qdrant (single instance, HNSW in memory)
              └── Vectors: in RAM
              └── Metadata: payload indexes
              └── Persistence: WAL + snapshots to disk
```

Sufficient for most RAG applications. A machine with 64GB RAM can hold ~10M vectors at 768 dimensions with HNSW overhead.

### Pattern 2: Distributed (10M - 1B vectors)

```
Application → Load Balancer → Milvus cluster
                                ├── Query Nodes (HNSW in memory, handle searches)
                                ├── Data Nodes (handle inserts, segment management)
                                ├── Index Nodes (build HNSW/IVF indexes offline)
                                ├── etcd (metadata coordination)
                                ├── MinIO/S3 (segment storage)
                                └── Pulsar/Kafka (streaming log)
```

Milvus shards data across multiple nodes. Each shard has its own HNSW index. Queries are scatter-gathered across shards and merged.

### Pattern 3: Serverless / Embedded

```
Application process
  └── LanceDB (embedded, no server)
      └── Lance files on S3
      └── IVF-PQ index loaded on demand
      └── Query: fetch only needed segments from S3
```

LanceDB and Chroma (embedded mode) run inside the application process — no network hops, no server management. Ideal for edge deployments and cost-sensitive applications.

### Pattern 4: Existing PostgreSQL + pgvector

```
Application → PostgreSQL (existing)
              ├── Relational tables (users, orders, products)
              ├── Vector columns (product_embedding vector(768))
              ├── HNSW index on vector columns
              └── SQL joins between relational and vector data

SELECT p.name, p.price, p.embedding <=> $1 AS distance
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE c.name = 'electronics'
  AND p.price < 500
ORDER BY p.embedding <=> $1
LIMIT 10;
```

The killer feature: **SQL joins**. No other vector database can join vector similarity results with relational data in a single query. Critical for e-commerce, content management, and any application with complex data relationships.

## Common Failure Modes in Production

### 1. Recall Degradation After Updates

**Symptom**: Search quality slowly degrades over weeks/months as data is inserted and deleted.

**Cause**: HNSW graph quality degrades with updates. Deleted vectors leave "holes" in the graph (orphaned edges). Inserted vectors get suboptimal connections (they connect to whatever's nearest at insertion time, not what's globally optimal).

**Fix**: Periodic re-indexing. Most databases support background reindexing. Schedule weekly or monthly depending on update volume.

### 2. Memory OOM on Large Indexes

**Symptom**: Database crashes or slows dramatically as data grows.

**Cause**: HNSW keeps full vectors + graph in memory. 10M × 768d × 4B = 30GB + 5GB graph = 35GB. Add metadata, temporary buffers, and OS needs — you're at 50GB+ easily.

**Fixes**: 
- Enable scalar quantization (int8 → 4x compression)
- Enable product quantization (further compression at some recall cost)
- Use disk-based storage (Qdrant's mmap mode, LanceDB)
- Switch to IVF-PQ for billion-scale (lower memory but lower recall)

### 3. Cold Start Latency

**Symptom**: First queries after restart are very slow (seconds instead of milliseconds).

**Cause**: HNSW index must be loaded from disk to memory. For a 30GB index, this takes 10-30 seconds depending on disk speed.

**Fix**: Pre-warm the index at startup. Keep the database running (don't restart for deployments). Use rolling deployments with multiple replicas.

### 4. Embedding Model Drift

**Symptom**: Recall drops after updating the embedding model, even though the new model is "better."

**Cause**: The new model produces embeddings in a **different space** than the old model. Old vectors and new vectors are incompatible — searching with a new-model query vector against old-model document vectors produces garbage results.

**Fix**: Re-embed ALL existing documents when changing the embedding model. There's no shortcut. Plan for this in your architecture by storing raw text alongside vectors so you can re-embed.

### 5. Filtering Destroys Recall

**Symptom**: Recall is 95% without filters, 50% with a selective filter (matching <5% of data).

**Cause**: Pre-filtering or naive filtering breaks HNSW graph navigability. The graph was built on the full dataset — removing 95% of nodes disconnects the remaining nodes.

**Fix**: Use in-graph filtering (Qdrant, Weaviate). If recall is still poor with very selective filters, create separate indexes per filter value (e.g., one HNSW index per category). This is called **partitioned indexing**.

## Interview Questions and Answers

### Q: What is a vector database and why can't you just use PostgreSQL with an array column?

A vector database is specialized storage for high-dimensional embeddings that supports efficient approximate nearest neighbor (ANN) search. You can't just store vectors as PostgreSQL arrays because: (1) ANN requires specialized index structures (HNSW, IVF) that general databases don't provide — brute-force search on arrays is $O(N \times d)$, infeasible for millions of vectors. (2) Distance computations need SIMD-optimized kernels operating on aligned memory, not generic array operations. (3) Vector databases optimize memory layout for sequential vector scanning, which is critical for cache efficiency.

That said, **pgvector** adds exactly these capabilities to PostgreSQL. For teams already on PostgreSQL with < 10M vectors, pgvector is a strong choice — you get HNSW indexing, cosine/L2/inner product distance, and the full power of SQL joins and transactions.

### Q: Explain the trade-offs between HNSW, IVF, and IVF-PQ. When would you use each?

**HNSW**: Graph-based. Best recall-speed trade-off for datasets up to ~100M vectors. Supports dynamic inserts. High memory usage (stores full vectors + graph in RAM). Use for: most applications under 100M vectors where memory is available.

**IVF**: Partition-based. Clusters vectors via k-means, then searches only nearby clusters. Lower memory than HNSW (no graph overhead), but requires training (the k-means step) and can miss vectors at cluster boundaries. Use for: simple deployments where HNSW's memory is too expensive, or as a building block for IVF-PQ.

**IVF-PQ**: IVF + product quantization. Compresses vectors 10-50x by splitting into subvectors and quantizing each. Dramatically reduces memory — a 768-dim float32 vector (3KB) becomes ~96 bytes. Trade-off: recall drops 5-10% vs HNSW. Use for: billion-scale datasets where full vectors don't fit in RAM (e.g., 1B × 768d = 3TB as float32, but only ~96GB with IVF-PQ).

| Scenario | Best Algorithm |
|----------|---------------|
| <10M vectors, need best recall | HNSW |
| 10M-100M vectors, enough RAM | HNSW with scalar quantization |
| 100M-1B vectors, limited RAM | IVF-PQ |
| 1B+ vectors on single machine | DiskANN |
| Need to avoid training step | HNSW (no training) |

### Q: How does metadata filtering interact with ANN search? What are the challenges?

The fundamental challenge: ANN indexes (HNSW, IVF) are built on the **full dataset**. When you filter, you're asking "find nearest neighbors within a subset" — but the index structure doesn't know about the subset.

Three approaches with trade-offs:

**Pre-filter → search**: Fast if filter is broad (matching >50% of data), but recall collapses with selective filters because the HNSW graph becomes disconnected.

**Search → post-filter**: Works well if filter matches >10% of data (oversample by 10x). Impractical with very selective filters (need 1000x oversampling).

**In-graph filter** (best): During HNSW traversal, check filter at each node. Skip non-matching nodes but continue exploring their neighbors. Maintains graph navigability. Qdrant and Weaviate implement this. Recall degrades gracefully even with selective filters.

Production tip: For frequently-used filter values (e.g., "category=electronics"), create separate HNSW indexes per filter value (partitioned indexing). This gives full recall within each partition at the cost of more memory.

### Q: You need to build a RAG system for 50M documents. Walk through your vector database architecture.

**Step 1 — Chunking**: Split documents into ~512-token passages with 50-token overlap. 50M docs × ~5 passages each = 250M passages.

**Step 2 — Embedding**: Use a high-quality model (e.g., `text-embedding-3-large`, 3072-dim, or `e5-large`, 1024-dim). At 1024-dim, storage = 250M × 1024 × 4B = 1TB. Too large for a single machine in HNSW.

**Step 3 — Choose database**: 250M vectors at 1024-dim → need distributed or compressed. Options:
- **Qdrant cluster** (3 nodes, 128GB each) with scalar quantization (250GB compressed) + replication
- **Milvus cluster** with IVF-PQ (25GB compressed) if recall trade-off is acceptable
- **Pinecone** if fully managed is preferred (serverless plan handles this scale)

**Step 4 — Index configuration**: 
- HNSW: M=16, ef_construction=200, ef_search=128
- Scalar quantization: int8 (4x compression, ~2% recall loss)
- Metadata: index on `source`, `date`, `department` fields for filtering

**Step 5 — Query pipeline**:
```
User query → embed → HNSW search (ef=128, top 50) → rerank with cross-encoder → top 5 → LLM
```

**Step 6 — Monitoring**: Track recall on a labeled evaluation set weekly. Alert if recall drops below 90%.

### Q: Compare Qdrant and pgvector for a production application.

| Aspect | Qdrant | pgvector |
|--------|--------|----------|
| Performance (1M, 768d) | ~15K QPS, ~5ms p99 | ~1.5K QPS, ~30ms p99 |
| Max practical scale | ~100M per node, distributed for more | ~5-10M per node (single PostgreSQL) |
| Filtering | Payload indexes, in-graph filtering (fast) | SQL WHERE (flexible but slower — requires two-phase) |
| Joins with relational data | Not possible | Native SQL joins (killer feature) |
| Operational complexity | Separate service to manage | Part of existing PostgreSQL |
| Hybrid search | Sparse vector support | Combine `tsvector` + `vector` in SQL |
| Quantization | Scalar, product, binary | None built-in (quantize before insert) |
| Transactions | No ACID | Full ACID |

**Choose Qdrant when**: Performance matters (10x faster), you need complex vector filtering, scale exceeds 5-10M vectors, or you need built-in quantization.

**Choose pgvector when**: You already run PostgreSQL, need SQL joins between vector results and relational tables, need ACID transactions, or your dataset is < 5M vectors and simplicity is valued over performance.

### Q: What happens when you change your embedding model? How do you handle migration?

This is a common production challenge. When you upgrade from model A to model B:

1. **Vectors from model A and model B are incompatible** — they live in different embedding spaces. You cannot search with a model-B query against model-A documents.

2. **You must re-embed ALL existing documents** with model B. For 50M documents, this can take hours-days depending on the model and hardware.

**Migration strategy**:
- Store raw text alongside vectors (always — this is your escape hatch)
- Create a new collection/index with model-B embeddings
- Run re-embedding as a batch job (parallel, resumable)
- Blue-green switch: route queries to the new collection once it's fully populated
- Keep the old collection as a rollback option until verified

**Avoiding this problem**: Choose a well-established embedding model (OpenAI, Cohere, e5-large) and don't change it unnecessarily. The quality difference between top-tier models is small — the migration cost is real.

### Q: How would you evaluate and monitor a vector database in production?

**Evaluation (before deployment)**:
1. Create a labeled evaluation dataset: 200-500 queries with known-relevant documents
2. Measure recall@K: what fraction of relevant documents appear in the top K results?
3. Measure latency: p50, p95, p99 at expected QPS
4. Measure throughput: maximum QPS before latency degrades
5. Test with representative filters (not just unfiltered)

**Monitoring (in production)**:
```
Key metrics:
  1. Search latency (p50, p95, p99) — alert if p99 > 100ms
  2. Recall on evaluation set — run weekly, alert if < target
  3. Index memory usage — alert if > 80% of available RAM
  4. Insert throughput — track for capacity planning
  5. Filter selectivity — track the cardinality of common filters
  6. Error rate — failed searches, timeouts
```

**Common anti-pattern**: Measuring only latency but not recall. A vector database can return results in 1ms with 50% recall (garbage results, fast). Always measure both together.

## References

1. [ANN Benchmarks](https://ann-benchmarks.com/)
2. [Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
3. [Jégou, H., Douze, M., & Schmid, C. (2011). Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202)
4. [Subramanya, S. J., et al. (2019). DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
5. [Guo, R., et al. (2020). Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)](https://arxiv.org/abs/1908.10396)
6. [Pinecone Documentation](https://docs.pinecone.io/)
7. [Weaviate Documentation](https://weaviate.io/developers/weaviate)
8. [Milvus Documentation](https://milvus.io/docs)
9. [Qdrant Documentation](https://qdrant.tech/documentation/)
10. [Chroma Documentation](https://docs.trychroma.com/)
11. [pgvector GitHub](https://github.com/pgvector/pgvector)
12. [LanceDB Documentation](https://lancedb.github.io/lancedb/)

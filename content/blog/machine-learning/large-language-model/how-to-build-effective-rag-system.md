---
title: "How to Build an Effective RAG System: A Complete Guide"
publishDate: "2026-03-16"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "RAG",
    "LLM",
    "retrieval-augmented-generation",
    "vector-database",
    "embeddings",
    "NLP",
    "AI",
    "information-retrieval",
  ]
date: "2026-03-16"
author: "Hiep Tran"
featured: true
aiGenerated: true
excerpt: "A comprehensive, practical guide to building production-ready RAG systems. Covers every stage from document ingestion and chunking strategies to retrieval, reranking, generation, and evaluation — with best practices, advanced techniques, and common pitfalls."
---

# How to Build an Effective RAG System: A Complete Guide

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an architecture pattern that enhances Large Language Models (LLMs) by giving them access to external knowledge at inference time. Instead of relying solely on what the model memorized during training, a RAG system **retrieves** relevant documents from a knowledge base and **feeds them into the prompt** so the LLM can generate grounded, accurate, and up-to-date responses.

The core idea is simple:

```
User Question → Retrieve relevant documents → Augment prompt with documents → Generate answer
```

### Why RAG Matters

| Problem with vanilla LLMs | How RAG solves it |
|---|---|
| Knowledge cutoff — the model doesn't know recent events | Retrieves from an up-to-date knowledge base |
| Hallucination — the model confidently makes things up | Grounds answers in real source documents |
| No private data access — the model can't see your internal docs | Connects to your proprietary knowledge base |
| Expensive retraining — updating knowledge requires fine-tuning | Just update the document store, no retraining |
| No source attribution — you can't verify where the answer came from | Every answer can cite the exact source chunk |

### RAG vs. Fine-tuning

A common question: *"Should I fine-tune or use RAG?"*

- **Fine-tuning** bakes knowledge into the model weights. Good for teaching the model a style, format, or domain-specific reasoning patterns.
- **RAG** keeps knowledge external and retrievable. Good for factual Q&A, documentation search, and any case where the knowledge changes frequently.

In practice, the best systems often **combine both**: fine-tune a model to be good at following retrieval context, then use RAG to supply that context.

---

## The RAG Pipeline: End-to-End Architecture

A production RAG system has two main pipelines:

### Indexing Pipeline (Offline)

This runs when you ingest new documents:

```
Raw Documents
    ↓
[1] Document Loading
    ↓
[2] Preprocessing & Cleaning
    ↓
[3] Chunking
    ↓
[4] Embedding
    ↓
[5] Indexing into Vector Store
```

### Query Pipeline (Online)

This runs when a user asks a question:

```
User Query
    ↓
[1] Query Understanding & Transformation
    ↓
[2] Retrieval (Vector Search + Optional Keyword Search)
    ↓
[3] Reranking
    ↓
[4] Context Assembly
    ↓
[5] Generation (LLM)
    ↓
[6] Post-processing & Citation
    ↓
Answer with Sources
```

Let's go through each stage in detail.

---

## Stage 1: Document Loading & Preprocessing

### Document Loading

Your knowledge base can contain many formats. You need parsers for each:

| Format | Recommended Tools | Notes |
|---|---|---|
| PDF | `PyMuPDF`, `pdfplumber`, `Unstructured` | PDFs are notoriously tricky — tables, images, multi-column layouts |
| HTML | `BeautifulSoup`, `Trafilatura` | Strip boilerplate, keep meaningful content |
| Markdown | Native parsing | Preserve heading hierarchy for metadata |
| Word/PPTX | `python-docx`, `python-pptx`, `Unstructured` | Extract text + structure |
| Code | Tree-sitter parsers | Preserve function/class boundaries |
| Images/Scans | `Tesseract OCR`, multimodal models | Consider using vision LLMs for complex layouts |

**Best practice**: Use a unified document loading library like [Unstructured](https://github.com/Unstructured-IO/unstructured) or [LlamaIndex's document loaders](https://docs.llamaindex.ai/) to handle multiple formats consistently.

### Preprocessing & Cleaning

Before chunking, clean your documents:

1. **Remove boilerplate**: Headers, footers, page numbers, navigation menus
2. **Normalize whitespace**: Collapse multiple newlines, fix encoding issues
3. **Extract metadata**: Title, author, date, section headers, URL — this metadata is crucial for filtering and citation later
4. **Handle tables**: Convert to markdown or structured text so the LLM can interpret them
5. **Deduplicate**: Remove duplicate or near-duplicate documents using MinHash or SimHash

```python
# Example: Simple preprocessing pipeline
def preprocess_document(doc: str, metadata: dict) -> dict:
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', doc)
    text = re.sub(r' {2,}', ' ', text)

    # Extract section structure
    sections = extract_sections_by_headers(text)

    # Clean each section
    cleaned_sections = []
    for section in sections:
        cleaned = remove_boilerplate(section)
        if len(cleaned.strip()) > 50:  # Skip near-empty sections
            cleaned_sections.append(cleaned)

    return {
        "text": "\n\n".join(cleaned_sections),
        "metadata": metadata,
        "sections": cleaned_sections
    }
```

---

## Stage 2: Chunking — The Most Critical Step

Chunking is how you split documents into smaller pieces for retrieval. **This is arguably the most impactful decision in your entire RAG pipeline.** Bad chunking leads to retrieving irrelevant or incomplete information, which cascades into poor answers.

### Why Chunk Size Matters

- **Too small** (< 100 tokens): Chunks lack context. The retriever finds a matching sentence, but the LLM doesn't have enough surrounding information to generate a good answer.
- **Too large** (> 1000 tokens): Chunks contain too much irrelevant information. The embedding becomes a "blurry average" of many topics, reducing retrieval precision. Also wastes context window space.
- **Sweet spot**: 200–500 tokens for most use cases, but this depends heavily on your data and query patterns.

### Chunking Strategies

#### 1. Fixed-Size Chunking

The simplest approach: split text every N tokens with some overlap.

```python
def fixed_size_chunk(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
```

**Pros**: Simple, predictable chunk sizes.
**Cons**: Splits mid-sentence, mid-paragraph, or mid-thought. Ignores document structure.

#### 2. Recursive Character Splitting

Tries to split on natural boundaries (paragraphs → sentences → words) and falls back to smaller separators only when chunks are too large.

```python
# LangChain's RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)
```

**Pros**: Respects natural text boundaries.
**Cons**: Still doesn't understand semantic meaning.

#### 3. Semantic Chunking

Uses embeddings to detect topic boundaries. Adjacent sentences with similar embeddings stay together; when embedding similarity drops (indicating a topic shift), a chunk boundary is created.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # Split when similarity drops below 95th percentile
)
chunks = splitter.create_documents([document])
```

**Pros**: Chunks are semantically coherent.
**Cons**: Slower (requires embedding computation), variable chunk sizes, sensitive to threshold tuning.

#### 4. Document Structure-Aware Chunking

Uses the document's own structure (headings, sections, code blocks) to define chunk boundaries.

```python
def structure_aware_chunk(markdown_text: str, max_chunk_size: int = 500) -> list[dict]:
    """Split by markdown headers, keeping hierarchy metadata."""
    chunks = []
    current_headers = {}

    for section in split_by_headers(markdown_text):
        # Track header hierarchy
        current_headers[section.level] = section.header
        # Clear lower-level headers
        for level in list(current_headers):
            if level > section.level:
                del current_headers[level]

        # If section is too long, sub-chunk with recursive splitting
        if len(section.content.split()) > max_chunk_size:
            sub_chunks = recursive_split(section.content, max_chunk_size)
            for sc in sub_chunks:
                chunks.append({
                    "text": sc,
                    "headers": dict(current_headers),
                    "source_section": section.header
                })
        else:
            chunks.append({
                "text": section.content,
                "headers": dict(current_headers),
                "source_section": section.header
            })

    return chunks
```

**Pros**: Preserves document structure, enables section-level filtering.
**Cons**: Requires format-specific parsers.

#### 5. Agentic Chunking (LLM-Based)

Uses an LLM to decide chunk boundaries based on semantic understanding.

```python
def agentic_chunk(text: str, llm) -> list[str]:
    """Use an LLM to identify natural chunk boundaries."""
    prompt = """Given the following text, identify the natural topical boundaries
    where the text should be split into self-contained chunks.
    Return the split points as line numbers.

    Text:
    {text}"""

    boundaries = llm.invoke(prompt.format(text=text))
    return split_at_boundaries(text, boundaries)
```

**Pros**: Most semantically coherent chunks.
**Cons**: Expensive, slow, non-deterministic. Best used for high-value, static document collections.

#### 6. Parent-Child Chunking (Small-to-Big)

Create two levels: **small chunks** for precise retrieval, **parent chunks** for comprehensive context.

```python
# Index small chunks for retrieval
small_chunks = split(document, chunk_size=200)

# Map each small chunk to its parent (larger context window)
parent_chunks = split(document, chunk_size=800)

# At retrieval time:
# 1. Search against small chunks (precise matching)
# 2. Return the parent chunk (more context for the LLM)
```

**Pros**: Best of both worlds — precise retrieval + rich context.
**Cons**: More complex indexing and storage.

### Chunking Best Practices

1. **Always use overlap** (10–20% of chunk size) to avoid losing context at boundaries
2. **Preserve metadata** with every chunk: source document, page number, section header, URL
3. **Add contextual headers**: Prepend the section title or document title to each chunk so it makes sense in isolation
4. **Test empirically**: There is no universally optimal chunk size. Test different sizes with your actual queries and measure retrieval quality
5. **Consider your query patterns**: If users ask detailed questions, use smaller chunks. If they ask broad questions, use larger chunks or parent-child retrieval

---

## Stage 3: Embedding

Embedding converts text chunks into dense vector representations that capture semantic meaning. Similar meanings → similar vectors → retrievable by vector search.

### Choosing an Embedding Model

| Model | Dimensions | Max Tokens | Highlights |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 | 8191 | Strong general-purpose, supports dimension reduction |
| `text-embedding-3-small` (OpenAI) | 1536 | 8191 | Good balance of cost and quality |
| `voyage-3-large` (Voyage AI) | 1024 | 32000 | Excellent for code and long documents |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | Best open-source option, runs locally |
| `nomic-embed-text` | 768 | 8192 | Open source, long context |
| `Cohere embed-v4` | 1024 | 512 | Built-in retrieval optimization |
| `GTE-Qwen2` | 768–1536 | 8192 | Strong multilingual support |

### How to Choose

1. **Check the MTEB leaderboard** for benchmark scores on retrieval tasks
2. **Match your domain**: Some models are better at code, legal, medical, etc.
3. **Consider latency and cost**: Smaller dimensions = faster search, less storage
4. **Max token limit**: Must be >= your chunk size
5. **Multilingual needs**: If your documents span languages, use a multilingual model

### Embedding Best Practices

```python
from openai import OpenAI

client = OpenAI()

def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed a batch of texts efficiently."""
    # Batch for efficiency (API handles up to ~2048 texts per call)
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]
```

**Key tips**:
- **Batch your embedding calls** — don't embed one chunk at a time
- **Normalize vectors** if your vector DB doesn't do it automatically
- **Use the same model** for indexing and querying (this is critical — mixing models produces garbage results)
- **Consider instruction-prefixed embeddings**: Some models (like E5, GTE) expect a prefix like `"query: "` or `"passage: "` to differentiate query vs. document embeddings

---

## Stage 4: Vector Storage & Indexing

### Choosing a Vector Database

| Database | Type | Best For |
|---|---|---|
| **Pinecone** | Managed cloud | Production-ready with zero ops |
| **Weaviate** | Self-hosted / cloud | Hybrid search (vector + keyword) out of the box |
| **Qdrant** | Self-hosted / cloud | High performance, rich filtering |
| **Milvus / Zilliz** | Self-hosted / cloud | Billion-scale datasets |
| **Chroma** | Embedded | Prototyping and small-scale apps |
| **pgvector** | Postgres extension | When you already use Postgres |
| **FAISS** | In-memory library | Research and offline experiments |

For a deeper dive on vector databases, see my article on [Vector Databases: Algorithms, Architecture, and Comparison](/blog/machine-learning/ai-agent/vector-database).

### Indexing Strategy

```python
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

client = qdrant_client.QdrantClient(url="http://localhost:6333")

# Create collection with appropriate settings
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(
        size=1536,           # Must match your embedding model dimensions
        distance=Distance.COSINE
    )
)

# Index chunks with metadata
points = [
    PointStruct(
        id=i,
        vector=chunk_embedding,
        payload={
            "text": chunk.text,
            "source": chunk.metadata["source"],
            "page": chunk.metadata.get("page"),
            "section": chunk.metadata.get("section"),
            "date": chunk.metadata.get("date"),
            "url": chunk.metadata.get("url"),
        }
    )
    for i, (chunk, chunk_embedding) in enumerate(zip(chunks, embeddings))
]

client.upsert(collection_name="knowledge_base", points=points)
```

### Metadata: Your Secret Weapon

Store rich metadata with every chunk. It enables **filtered retrieval**, which dramatically improves precision:

```python
# Instead of searching ALL chunks, filter by metadata first
results = client.search(
    collection_name="knowledge_base",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="source", match=MatchValue(value="technical-docs")),
            FieldCondition(key="date", range=Range(gte="2025-01-01")),
        ]
    ),
    limit=10
)
```

---

## Stage 5: Retrieval — Finding the Right Context

Retrieval is where most RAG systems succeed or fail. The LLM can only generate a good answer if it receives the right context.

### Retrieval Methods

#### 1. Dense Retrieval (Vector Search)

The standard RAG approach: embed the query, search for nearest neighbors.

```python
query_embedding = embed_texts(["What is the capital of France?"])[0]
results = vector_db.search(query_embedding, top_k=10)
```

**Strength**: Understands semantics — "car" matches "automobile".
**Weakness**: Can miss exact keyword matches — searching for "error code 0x8007" might not find chunks containing that exact string.

#### 2. Sparse Retrieval (Keyword Search)

Traditional BM25/TF-IDF based search that matches on exact tokens.

```python
from rank_bm25 import BM25Okapi

tokenized_corpus = [chunk.split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

query_tokens = "error code 0x8007".split()
scores = bm25.get_scores(query_tokens)
```

**Strength**: Excellent at exact matches, acronyms, error codes, proper nouns.
**Weakness**: No semantic understanding — "car" doesn't match "automobile".

#### 3. Hybrid Retrieval (Best of Both Worlds)

Combine dense + sparse retrieval and merge results. This is the **recommended approach for production systems**.

```python
def hybrid_search(query: str, alpha: float = 0.7, top_k: int = 10):
    """
    Combine vector search and BM25.
    alpha: weight for vector search (1.0 = pure vector, 0.0 = pure BM25)
    """
    # Dense retrieval
    query_embedding = embed(query)
    dense_results = vector_db.search(query_embedding, top_k=top_k * 2)

    # Sparse retrieval
    sparse_results = bm25_search(query, top_k=top_k * 2)

    # Reciprocal Rank Fusion (RRF) to merge results
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        weights=[alpha, 1 - alpha]
    )

    return fused[:top_k]

def reciprocal_rank_fusion(result_lists: list, weights: list, k: int = 60) -> list:
    """Merge multiple ranked lists using RRF."""
    scores = {}
    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results):
            doc_id = doc.id
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += weight * (1 / (k + rank + 1))

    # Sort by fused score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Many vector databases (Weaviate, Qdrant, Pinecone) support hybrid search natively.

#### 4. Multi-Query Retrieval

A single query might not capture all aspects of the user's question. Generate multiple query variations and retrieve for each:

```python
def multi_query_retrieve(original_query: str, llm, retriever, top_k: int = 10):
    """Generate multiple query perspectives and retrieve for each."""
    prompt = f"""Generate 3 different versions of the following question to help
    retrieve relevant documents. Each version should approach the question
    from a different angle.

    Original question: {original_query}

    Provide 3 alternative questions, one per line:"""

    alternative_queries = llm.invoke(prompt).split("\n")

    all_results = set()
    for query in [original_query] + alternative_queries:
        results = retriever.search(query, top_k=top_k)
        all_results.update(results)

    return list(all_results)[:top_k]
```

#### 5. HyDE (Hypothetical Document Embedding)

Instead of embedding the query directly, ask the LLM to generate a hypothetical answer, then embed *that* to search.

```python
def hyde_retrieve(query: str, llm, retriever, top_k: int = 10):
    """Generate a hypothetical document and use it for retrieval."""
    prompt = f"""Write a short passage that would answer the following question.
    Don't worry about accuracy — just write what a good answer would look like.

    Question: {query}

    Passage:"""

    hypothetical_doc = llm.invoke(prompt)

    # Embed the hypothetical document (which is closer to document space
    # than a short question would be)
    hyde_embedding = embed(hypothetical_doc)

    return retriever.search_by_vector(hyde_embedding, top_k=top_k)
```

**Why this works**: Queries are short and often in "question space" while documents are in "answer space". HyDE bridges this gap by converting the query into answer-like text before embedding.

#### 6. Contextual Retrieval (Anthropic's Approach)

Prepend document-level context to each chunk before embedding, so the chunk makes sense in isolation:

```python
def add_contextual_prefix(chunk: str, document: str, llm) -> str:
    """Use an LLM to generate context that situates the chunk within the document."""
    prompt = f"""<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

    context = llm.invoke(prompt)
    return f"{context}\n\n{chunk}"
```

This technique can **reduce retrieval failures by ~50%** according to Anthropic's experiments.

### Retrieval Best Practices

1. **Retrieve more, rerank later**: Fetch top-20 to top-50 candidates, then rerank to top-5
2. **Use hybrid search** unless you have a specific reason not to
3. **Filter by metadata** when possible — it's faster and more precise than searching everything
4. **Track retrieval metrics**: Monitor what percentage of queries return relevant results
5. **Set a similarity threshold**: Don't include results below a minimum relevance score

---

## Stage 6: Reranking — Quality Over Quantity

The initial retrieval cast a wide net. Now we need to **rerank** the candidates to put the most relevant ones first. This step alone can improve answer quality by 10-20%.

### Why Rerank?

- **Bi-encoder** (embedding search) compresses documents into fixed-size vectors — fast but lossy
- **Cross-encoder** (reranker) takes the full query + document text and computes a precise relevance score — slow but much more accurate

### Reranking Options

#### 1. Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank(query: str, documents: list[str], top_k: int = 5) -> list:
    """Rerank documents using a cross-encoder."""
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)

    # Sort by score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

#### 2. Cohere Rerank API

```python
import cohere

co = cohere.Client("your-api-key")

results = co.rerank(
    model="rerank-v3.5",
    query="What is the capital of France?",
    documents=candidate_documents,
    top_n=5
)
```

#### 3. LLM-Based Reranking

Use the LLM itself to judge relevance (expensive but powerful for complex queries):

```python
def llm_rerank(query: str, documents: list[str], llm) -> list[str]:
    """Use an LLM to judge document relevance."""
    prompt = f"""Given the question: "{query}"

Rate each document's relevance on a scale of 1-10 and explain briefly.

{chr(10).join(f"Document {i+1}: {doc[:200]}..." for i, doc in enumerate(documents))}

Return a JSON list of [doc_index, score] pairs, sorted by score descending."""

    result = llm.invoke(prompt)
    rankings = parse_json(result)
    return [documents[idx] for idx, _ in rankings]
```

### Reranking Best Practices

1. **Retrieve broadly, rerank narrowly**: Fetch 20-50 candidates, rerank to top 3-5
2. **Use cross-encoders for cost-effective reranking** — they're much cheaper than LLM-based reranking
3. **Consider latency**: Cross-encoder reranking adds 50-200ms. For real-time apps, keep candidate count manageable
4. **Chain rerankers**: Vector search → Cohere/cross-encoder rerank → LLM relevance filter

---

## Stage 7: Context Assembly & Prompt Engineering

You've retrieved and reranked the best chunks. Now you need to assemble them into a prompt that helps the LLM generate the best possible answer.

### Prompt Template

```python
RAG_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite your sources by referencing the [Source N] tags
- Be concise and direct

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

def assemble_context(chunks: list[dict], max_tokens: int = 3000) -> str:
    """Assemble retrieved chunks into a formatted context string."""
    context_parts = []
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        chunk_tokens = len(chunk_text.split()) * 1.3  # Rough token estimate

        if total_tokens + chunk_tokens > max_tokens:
            break

        source_info = chunk.get("source", "Unknown")
        context_parts.append(f"[Source {i+1}] ({source_info})\n{chunk_text}")
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts)
```

### Context Ordering

Research shows that LLMs pay more attention to the **beginning and end** of the context (the "lost in the middle" problem). Place your most relevant chunks first and last:

```python
def order_context_for_attention(chunks: list, relevance_scores: list) -> list:
    """Order chunks to maximize LLM attention — best at start and end."""
    ranked = sorted(zip(chunks, relevance_scores), key=lambda x: x[1], reverse=True)

    if len(ranked) <= 2:
        return [c for c, _ in ranked]

    # Best chunks at positions 1 and N, lesser ones in the middle
    result = []
    for i, (chunk, _) in enumerate(ranked):
        if i % 2 == 0:
            result.insert(0, chunk)  # Prepend (goes to start)
        else:
            result.append(chunk)     # Append (goes to end)

    return result
```

### Key Prompt Engineering Tips

1. **Be explicit about behavior**: Tell the LLM to say "I don't know" when context is insufficient
2. **Add source citation instructions**: "[Source 1] says..." format makes verification easy
3. **Set boundaries**: "Answer based ONLY on the provided context" prevents hallucination
4. **Include metadata**: Adding dates, authors, and section titles helps the LLM reason about recency and authority
5. **Control output format**: If you need structured output (JSON, bullet points), specify it in the prompt

---

## Stage 8: Generation

### Model Selection

| Model | Best For |
|---|---|
| Claude (Sonnet/Opus) | Long-context reasoning, nuanced answers, safety |
| GPT-4o | General-purpose, fast |
| GPT-4o-mini | Cost-sensitive applications |
| Llama 3 (70B+) | Self-hosted, privacy-sensitive |
| Mistral Large | European data sovereignty requirements |
| Gemini 1.5 Pro | Extremely long context windows |

### Streaming Responses

For a good UX, always stream responses:

```python
from anthropic import Anthropic

client = Anthropic()

def generate_answer(context: str, question: str):
    """Generate a streaming RAG answer."""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": RAG_PROMPT.format(context=context, question=question)
        }]
    ) as stream:
        for text in stream.text_stream:
            yield text
```

### Handling "I Don't Know"

A critical feature: the system should **refuse to answer** when the retrieved context doesn't contain relevant information.

```python
SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based on the
provided context. If the context does not contain sufficient information to
answer the question, respond with:

"I don't have enough information in my knowledge base to answer this question.
Here's what I found that might be related: [brief summary of what was retrieved]"

Never make up information that isn't in the context."""
```

---

## Stage 9: Evaluation — Measuring RAG Quality

You can't improve what you can't measure. RAG evaluation has three levels:

### Level 1: Retrieval Evaluation

*"Did we find the right documents?"*

| Metric | What It Measures | How to Compute |
|---|---|---|
| **Recall@K** | What fraction of relevant docs are in the top-K results | `(relevant ∩ retrieved) / relevant` |
| **Precision@K** | What fraction of top-K results are relevant | `(relevant ∩ retrieved) / K` |
| **MRR** (Mean Reciprocal Rank) | How high the first relevant doc ranks | `1 / rank_of_first_relevant` |
| **NDCG** | Relevance quality of the full ranking | Accounts for position and graded relevance |
| **Hit Rate** | Did at least one relevant doc appear in top-K | Binary: 1 if any relevant doc found, else 0 |

```python
def evaluate_retrieval(queries: list[dict], retriever, k: int = 10):
    """Evaluate retrieval quality against ground truth."""
    metrics = {"recall": [], "precision": [], "mrr": [], "hit_rate": []}

    for item in queries:
        query = item["question"]
        relevant_ids = set(item["relevant_doc_ids"])

        results = retriever.search(query, top_k=k)
        retrieved_ids = [r.id for r in results]

        # Recall@K
        found = set(retrieved_ids) & relevant_ids
        metrics["recall"].append(len(found) / len(relevant_ids))

        # Precision@K
        metrics["precision"].append(len(found) / k)

        # MRR
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                metrics["mrr"].append(1 / rank)
                break
        else:
            metrics["mrr"].append(0)

        # Hit Rate
        metrics["hit_rate"].append(1 if found else 0)

    return {k: sum(v) / len(v) for k, v in metrics.items()}
```

### Level 2: Generation Evaluation

*"Is the generated answer correct and faithful to the context?"*

#### LLM-as-Judge

Use a strong LLM to evaluate answer quality:

```python
EVAL_PROMPT = """You are evaluating a RAG system's answer. Score each dimension from 1-5.

**Question**: {question}
**Retrieved Context**: {context}
**Generated Answer**: {answer}
**Ground Truth Answer**: {ground_truth}

Evaluate:
1. **Faithfulness** (1-5): Does the answer only use information from the context? No hallucination?
2. **Relevance** (1-5): Does the answer address the question?
3. **Completeness** (1-5): Does the answer cover all important points from the context?
4. **Conciseness** (1-5): Is the answer free of unnecessary information?

Return a JSON object with scores and brief justifications."""
```

#### RAGAS Framework

[RAGAS](https://github.com/explodinggradients/ragas) is a popular framework for RAG evaluation:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Is the answer grounded in the context?
    answer_relevancy,    # Is the answer relevant to the question?
    context_precision,   # Are retrieved contexts relevant?
    context_recall,      # Are all relevant contexts retrieved?
)

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(results)
```

### Level 3: End-to-End Evaluation

*"Are users happy with the system?"*

- **User feedback**: Thumbs up/down on answers
- **Click-through rate on citations**: Are users verifying sources?
- **Query abandonment rate**: Do users rephrase or give up?
- **Time to answer**: Is the system fast enough?
- **A/B testing**: Compare different configurations with real users

---

## Advanced Techniques

### 1. Query Routing

Not all queries need RAG. Route queries to the right pipeline:

```python
def route_query(query: str, llm) -> str:
    """Determine the best pipeline for a query."""
    prompt = f"""Classify this query into one of these categories:
    - RETRIEVAL: Needs specific factual information from documents
    - CONVERSATIONAL: General chat, greetings, or simple questions
    - ANALYTICAL: Needs reasoning across multiple documents
    - CODE: Needs code generation or debugging help

    Query: {query}
    Category:"""

    category = llm.invoke(prompt).strip()

    if category == "CONVERSATIONAL":
        return "direct_llm"       # No retrieval needed
    elif category == "ANALYTICAL":
        return "multi_step_rag"   # Retrieve → Reason → Retrieve again
    elif category == "CODE":
        return "code_rag"         # Specialized code retrieval
    else:
        return "standard_rag"
```

### 2. Corrective RAG (CRAG)

After retrieval, evaluate whether the results are actually relevant. If not, fall back to web search or rephrase the query:

```python
def corrective_rag(query: str, retriever, llm):
    """Self-correcting RAG pipeline."""
    # Step 1: Initial retrieval
    docs = retriever.search(query, top_k=5)

    # Step 2: Grade each document's relevance
    relevant_docs = []
    for doc in docs:
        grade = llm.invoke(f"Is this document relevant to '{query}'? "
                          f"Document: {doc.text[:500]}... Answer YES or NO.")
        if "YES" in grade.upper():
            relevant_docs.append(doc)

    # Step 3: If insufficient relevant docs, try query transformation
    if len(relevant_docs) < 2:
        rewritten_query = llm.invoke(
            f"Rewrite this query to be more specific: {query}"
        )
        additional_docs = retriever.search(rewritten_query, top_k=5)
        relevant_docs.extend(additional_docs)

    # Step 4: Generate answer with validated context
    return generate_answer(relevant_docs, query)
```

### 3. Self-RAG

The model decides at each generation step whether to retrieve, and after generating, checks its own output for hallucination:

```
Generate → [Decide: need more info?] → Retrieve → Generate → [Check: is this faithful?] → Output
```

### 4. Agentic RAG

Use an AI agent that can iteratively search, reason, and refine:

```python
def agentic_rag(query: str, tools: dict, llm, max_iterations: int = 5):
    """Agent that iteratively retrieves and reasons."""
    context = []
    thought_process = []

    for i in range(max_iterations):
        # Agent decides next action
        action = llm.invoke(f"""
        Question: {query}
        Current context: {context}
        Previous thoughts: {thought_process}

        What should I do next? Options:
        1. SEARCH(query) - search the knowledge base
        2. ANSWER(text) - provide the final answer
        3. CLARIFY(question) - ask a follow-up question to the knowledge base
        """)

        if action.startswith("ANSWER"):
            return action.replace("ANSWER(", "").rstrip(")")
        elif action.startswith("SEARCH"):
            search_query = extract_query(action)
            results = tools["search"](search_query)
            context.extend(results)
            thought_process.append(f"Searched for: {search_query}, found {len(results)} results")
        elif action.startswith("CLARIFY"):
            clarify_query = extract_query(action)
            results = tools["search"](clarify_query)
            context.extend(results)

    return generate_answer(context, query)
```

### 5. Knowledge Graph-Enhanced RAG (GraphRAG)

Combine vector retrieval with knowledge graph traversal for multi-hop reasoning:

```
Query: "What products does the CEO of Company X's subsidiary manage?"

Vector retrieval alone might miss this because the answer spans multiple documents.

With GraphRAG:
1. Entity extraction: "Company X" → find CEO → find subsidiary → find products
2. Graph traversal connects the dots across documents
3. Combine graph results with vector results for comprehensive context
```

### 6. Multi-Modal RAG

Extend RAG to handle images, tables, and diagrams:

```python
def multimodal_rag(query: str, retriever, vision_llm):
    """RAG that handles text, images, and tables."""
    # Retrieve text chunks
    text_results = retriever.search_text(query, top_k=5)

    # Retrieve relevant images/diagrams
    image_results = retriever.search_images(query, top_k=3)

    # Use a vision model to interpret images
    image_descriptions = []
    for img in image_results:
        description = vision_llm.describe(img.image_data, query)
        image_descriptions.append(description)

    # Combine all context
    full_context = format_context(text_results, image_descriptions)
    return generate_answer(full_context, query)
```

---

## Common Pitfalls and How to Avoid Them

### 1. "Garbage In, Garbage Out" — Poor Document Quality

**Symptom**: The system retrieves chunks but answers are still wrong.
**Cause**: Source documents contain errors, are outdated, or are poorly formatted.
**Fix**: Invest in document quality. Clean, deduplicate, and validate your knowledge base. Remove outdated documents or add date metadata for recency filtering.

### 2. The Lost-in-the-Middle Problem

**Symptom**: The answer ignores relevant information that was in the middle of the context.
**Cause**: LLMs attend more to the beginning and end of long contexts.
**Fix**: Place the most relevant chunks at the start and end. Keep total context length reasonable (don't stuff 20 chunks when 5 will do).

### 3. Chunk Boundary Artifacts

**Symptom**: Retrieved chunks cut off mid-sentence or miss critical context.
**Cause**: Naive chunking that splits on character count.
**Fix**: Use structure-aware or semantic chunking. Always include overlap. Consider parent-child chunking.

### 4. Embedding/Query Mismatch

**Symptom**: Relevant documents exist but aren't retrieved.
**Cause**: Query phrasing is very different from document phrasing.
**Fix**: Use HyDE, multi-query retrieval, or query expansion. Add contextual prefixes to chunks.

### 5. Over-Retrieval

**Symptom**: The answer contains irrelevant information or is confusing.
**Cause**: Too many chunks retrieved, some irrelevant.
**Fix**: Use reranking. Set similarity thresholds. Reduce top-K after reranking.

### 6. Hallucination Despite Retrieval

**Symptom**: The answer includes information not in any retrieved chunk.
**Cause**: The LLM ignores the context and generates from its own knowledge.
**Fix**: Stronger system prompts. Use CRAG to validate. Fine-tune the model on context-following behavior. Consider smaller, more instruction-following models.

### 7. Latency Issues

**Symptom**: The system takes too long to respond.
**Cause**: Multiple sequential API calls (embed → search → rerank → generate).
**Fix**:
- Cache frequent queries and their results
- Use smaller/faster embedding models
- Limit reranking to top-20 candidates
- Stream the LLM response
- Pre-compute embeddings for known query patterns

---

## Production Checklist

Before deploying your RAG system, go through this checklist:

### Data Pipeline
- [ ] Document ingestion handles all required formats
- [ ] Chunking strategy is tested and optimized for your data
- [ ] Metadata is extracted and stored with every chunk
- [ ] Incremental updates work (add/update/delete documents)
- [ ] Deduplication is in place

### Retrieval
- [ ] Hybrid search (vector + keyword) is configured
- [ ] Reranking is enabled
- [ ] Metadata filters are available for common query patterns
- [ ] Similarity threshold is set to filter out irrelevant results
- [ ] Retrieval metrics are tracked (recall, precision, MRR)

### Generation
- [ ] System prompt instructs the LLM to stay faithful to context
- [ ] "I don't know" behavior is implemented and tested
- [ ] Source citations are included in responses
- [ ] Response streaming is enabled
- [ ] Output is validated (no PII leakage, appropriate tone)

### Evaluation
- [ ] Evaluation dataset exists with ground truth Q&A pairs
- [ ] Retrieval and generation metrics are measured regularly
- [ ] User feedback loop is in place
- [ ] A/B testing infrastructure is ready

### Operations
- [ ] Vector database is backed up and scalable
- [ ] Embedding model versioning is tracked
- [ ] Monitoring and alerting for API failures, latency spikes
- [ ] Rate limiting and authentication for the RAG API
- [ ] Cost monitoring for LLM and embedding API calls

---

## Putting It All Together: A Minimal Working Example

Here's a complete, minimal RAG system using OpenAI and Chroma:

```python
"""Minimal RAG system — ~60 lines of core logic."""

import chromadb
from openai import OpenAI

# Initialize
openai_client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_rag")

# --- Indexing ---

def index_documents(documents: list[dict]):
    """Index documents with embeddings and metadata."""
    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [{"source": doc["source"]} for doc in documents]

    # Embed all texts
    response = openai_client.embeddings.create(
        input=texts, model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]

    # Store in Chroma
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

# --- Querying ---

def ask(question: str, top_k: int = 5) -> str:
    """Ask a question and get a RAG-powered answer."""
    # 1. Embed the question
    q_embedding = openai_client.embeddings.create(
        input=[question], model="text-embedding-3-small"
    ).data[0].embedding

    # 2. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k
    )

    # 3. Assemble context
    context = "\n\n---\n\n".join(
        f"[Source: {meta['source']}]\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    # 4. Generate answer
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context. "
             "Cite sources. Say 'I don't know' if the context is insufficient."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    return response.choices[0].message.content

# --- Usage ---

# Index some documents
index_documents([
    {"id": "1", "text": "RAG combines retrieval with generation...", "source": "rag-intro.pdf"},
    {"id": "2", "text": "Vector databases store embeddings...", "source": "vector-db.pdf"},
])

# Ask a question
answer = ask("What is RAG?")
print(answer)
```

---

## Real-Life Use Cases

RAG is not just a research concept — it powers production systems at companies of every size. Below are concrete use cases grouped by data source, including several that pull from structured databases rather than just documents.

### Use Case 1: Customer Support Chatbot (Document-Based)

**Company type**: SaaS, e-commerce, telecom
**Data source**: Help center articles, product docs, past support tickets

A support chatbot uses RAG to answer customer questions by retrieving relevant knowledge base articles. When a customer asks *"How do I reset my password on mobile?"*, the system retrieves the specific help article and generates a step-by-step answer.

**Why RAG over fine-tuning**: Product documentation changes frequently (new features, updated UI). RAG lets you update the knowledge base without retraining the model.

**Architecture**:
```
Customer message
    → Query rewriting (handle typos, abbreviations)
    → Hybrid search over help center chunks
    → Rerank top results
    → Generate answer with source links
    → If confidence is low → escalate to human agent
```

**Real-world example**: Klarna's AI assistant handles 2/3 of customer service chats using a RAG-like system over their support documentation.

---

### Use Case 2: Internal Knowledge Assistant (Enterprise Docs + Database)

**Company type**: Any enterprise
**Data source**: Confluence pages, Notion docs, Slack threads, Google Drive, HR policies, **SQL databases** (employee directory, project tracker, OKRs)

Employees ask questions like:
- *"What is our parental leave policy?"* → retrieves from HR policy documents
- *"Who is the tech lead for Project Atlas?"* → queries the **project database**
- *"How many open headcount do we have in the engineering org?"* → queries the **HR database**

**The database challenge**: Not all knowledge lives in documents. Organizational data (people, projects, budgets, OKRs) lives in databases. The RAG system needs to handle both.

**Architecture with Text-to-SQL**:

```python
def hybrid_knowledge_assistant(query: str, llm, doc_retriever, db_connection):
    """Route between document retrieval and database queries."""

    # Step 1: Classify query type
    route = llm.invoke(f"""Classify this employee question:
    - DOCUMENT: Needs policy, process, or knowledge base info
    - DATABASE: Needs structured data (people, projects, numbers, dates)
    - BOTH: Needs info from documents AND structured data

    Question: {query}
    Category:""").strip()

    context_parts = []

    # Step 2: Retrieve from documents if needed
    if route in ("DOCUMENT", "BOTH"):
        doc_results = doc_retriever.search(query, top_k=5)
        context_parts.append(format_doc_results(doc_results))

    # Step 3: Query database if needed
    if route in ("DATABASE", "BOTH"):
        # Generate SQL from natural language
        sql = text_to_sql(query, db_connection.schema, llm)

        # Safety check: only allow SELECT queries
        if not sql.strip().upper().startswith("SELECT"):
            context_parts.append("Database query blocked for safety.")
        else:
            results = db_connection.execute(sql)
            context_parts.append(format_db_results(results, sql))

    # Step 4: Generate final answer
    context = "\n\n".join(context_parts)
    return generate_answer(context, query, llm)


def text_to_sql(query: str, schema: str, llm) -> str:
    """Convert natural language to SQL."""
    prompt = f"""Given this database schema:
{schema}

Convert this question to a SQL SELECT query. Return ONLY the SQL, nothing else.
Question: {query}
SQL:"""
    return llm.invoke(prompt).strip()
```

**Key considerations for database RAG**:
- **Schema awareness**: The LLM needs to know your table/column names and relationships
- **Safety**: Never execute generated INSERT/UPDATE/DELETE — only SELECT
- **Caching**: Cache frequent queries (e.g., "who is the CEO" doesn't change daily)
- **Fallback**: If Text-to-SQL fails, show the user a helpful error rather than wrong data

---

### Use Case 3: E-commerce Product Advisor (Product Database + Reviews)

**Company type**: E-commerce, marketplace
**Data source**: **Product catalog database** (SQL/NoSQL), customer reviews, product descriptions

Customers ask:
- *"What's the best laptop under $1000 for video editing?"* → queries the **product database** with filters, retrieves reviews
- *"Does the Samsung Galaxy S24 have wireless charging?"* → retrieves from product specs
- *"Compare the MacBook Air M3 vs Dell XPS 15"* → queries **two products from the database**, retrieves relevant reviews for both

**Architecture**:

```python
def product_advisor(query: str, llm, product_db, review_retriever):
    """RAG over structured product data + unstructured reviews."""

    # Step 1: Extract structured filters from the query
    filters = llm.invoke(f"""Extract product search filters from this query.
    Return JSON with optional fields: category, min_price, max_price,
    brand, use_case, features.

    Query: {query}
    Filters:""")

    filters = parse_json(filters)

    # Step 2: Query product database with extracted filters
    products = product_db.search(
        category=filters.get("category"),
        min_price=filters.get("min_price"),
        max_price=filters.get("max_price"),
        brand=filters.get("brand"),
        limit=10
    )

    # Step 3: Retrieve relevant reviews for matched products
    product_ids = [p.id for p in products]
    reviews = review_retriever.search(
        query=query,
        filter={"product_id": {"$in": product_ids}},
        top_k=10
    )

    # Step 4: Assemble context from both sources
    context = format_product_context(products, reviews)

    # Step 5: Generate recommendation
    return generate_answer(context, query, llm,
        system="You are a helpful product advisor. Recommend products based on "
               "the user's needs. Always mention price, key specs, and cite "
               "customer reviews when relevant.")
```

**Why this is powerful**: The system combines structured data (exact prices, specs, availability from the database) with unstructured data (sentiment and real experiences from reviews) to give advice that neither source could provide alone.

---

### Use Case 4: Healthcare Clinical Decision Support (EHR Database + Medical Literature)

**Company type**: Hospital, healthtech
**Data source**: **Electronic Health Records (EHR) database** (patient history, lab results, medications), medical literature, clinical guidelines

A physician asks:
- *"What are the drug interactions for this patient's current medications?"* → queries the **patient medication database**, retrieves from drug interaction literature
- *"What's the recommended treatment protocol for stage 2 hypertension in a diabetic patient?"* → retrieves from clinical guidelines, cross-references with **patient lab values from the database**

**Architecture**:

```python
def clinical_decision_support(query: str, patient_id: str, llm, ehr_db, literature_retriever):
    """RAG combining patient data with medical knowledge."""
    context_parts = []

    # Step 1: Retrieve relevant patient data from EHR database
    patient_data = ehr_db.get_patient_summary(patient_id)
    medications = ehr_db.get_current_medications(patient_id)
    recent_labs = ehr_db.get_recent_lab_results(patient_id, days=90)

    context_parts.append(f"""PATIENT CONTEXT:
- Age: {patient_data.age}, Sex: {patient_data.sex}
- Conditions: {', '.join(patient_data.active_conditions)}
- Current medications: {', '.join(m.name + ' ' + m.dose for m in medications)}
- Recent labs: {format_labs(recent_labs)}""")

    # Step 2: Retrieve relevant medical literature
    # Include patient context in the retrieval query for relevance
    enriched_query = f"{query} Patient has: {', '.join(patient_data.active_conditions)}"
    literature = literature_retriever.search(enriched_query, top_k=5)
    context_parts.append(format_literature(literature))

    # Step 3: Retrieve clinical guidelines
    guidelines = literature_retriever.search(
        query,
        filter={"document_type": "clinical-guideline"},
        top_k=3
    )
    context_parts.append(format_guidelines(guidelines))

    # Step 4: Generate with strong safety guardrails
    return generate_answer(
        "\n\n".join(context_parts), query, llm,
        system="You are a clinical decision support tool. Provide evidence-based "
               "suggestions citing specific guidelines and studies. ALWAYS note that "
               "final clinical decisions must be made by the treating physician. "
               "Flag any potential drug interactions or contraindications.")
```

**Critical requirements**:
- HIPAA compliance for all data handling
- Never present AI suggestions as definitive medical advice
- Audit trail for every query and response
- Physician-in-the-loop — the system assists, never decides

---

### Use Case 5: Financial Analytics Assistant (Market Database + Reports)

**Company type**: Investment firm, bank, fintech
**Data source**: **Market data database** (stock prices, financial statements, economic indicators), analyst reports, SEC filings, earnings call transcripts

Analysts ask:
- *"What was Apple's gross margin trend over the last 8 quarters?"* → queries the **financial database**, generates chart
- *"Summarize the key risks mentioned in Tesla's latest 10-K filing"* → retrieves from SEC filings
- *"How did semiconductor companies perform after the last Fed rate hike?"* → queries **market database** for historical prices + retrieves analyst commentary

**Architecture**:

```python
def financial_assistant(query: str, llm, market_db, report_retriever):
    """RAG combining market data with financial reports."""
    context_parts = []

    # Step 1: Determine if structured data is needed
    needs_data = llm.invoke(
        f"Does this question need numerical market/financial data? YES or NO: {query}"
    ).strip()

    if "YES" in needs_data.upper():
        # Generate and execute a safe database query
        sql = text_to_sql(query, market_db.schema, llm)

        # Validate: only allow SELECT on approved tables
        if is_safe_financial_query(sql):
            data = market_db.execute(sql)
            context_parts.append(f"DATABASE RESULTS:\n{format_table(data)}\n"
                                f"Query used: {sql}")

    # Step 2: Retrieve relevant reports and filings
    reports = report_retriever.search(query, top_k=5)
    context_parts.append(format_reports(reports))

    # Step 3: Generate analysis
    return generate_answer(
        "\n\n".join(context_parts), query, llm,
        system="You are a financial analyst assistant. When presenting data, "
               "be precise with numbers and dates. Cite sources for all claims. "
               "Note that past performance does not guarantee future results.")
```

---

### Use Case 6: Developer Documentation & Codebase Q&A (Code + Docs + Logs Database)

**Company type**: Any engineering team
**Data source**: Codebase (GitHub), internal documentation, **CI/CD logs database**, **incident/on-call database**, Slack engineering channels

Engineers ask:
- *"How does the authentication middleware work?"* → retrieves from codebase and architecture docs
- *"Why did the payment service crash last Tuesday?"* → queries the **incident database** + retrieves post-mortem docs + queries **error log database**
- *"What's the P99 latency for the /api/users endpoint this week?"* → queries the **metrics database**

**Architecture**:

```python
def engineering_assistant(query: str, llm, code_retriever, doc_retriever, ops_db):
    """RAG for engineering teams combining code, docs, and operational data."""

    route = classify_engineering_query(query, llm)
    context_parts = []

    if route in ("CODE", "ARCHITECTURE"):
        # Search codebase with AST-aware chunking
        code_results = code_retriever.search(query, top_k=5)
        context_parts.append(format_code_results(code_results))

    if route in ("INCIDENT", "DEBUGGING"):
        # Query incident database
        incidents = ops_db.query("""
            SELECT title, severity, root_cause, resolution, created_at
            FROM incidents
            WHERE ts_rank(search_vector, plainto_tsquery(%s)) > 0.1
            ORDER BY created_at DESC LIMIT 5
        """, [query])
        context_parts.append(format_incidents(incidents))

        # Query error logs
        error_logs = ops_db.query("""
            SELECT timestamp, service, error_message, stack_trace
            FROM error_logs
            WHERE message_embedding <-> %s < 0.3
            ORDER BY timestamp DESC LIMIT 10
        """, [embed(query)])
        context_parts.append(format_logs(error_logs))

    if route in ("DOCS", "ARCHITECTURE", "PROCESS"):
        doc_results = doc_retriever.search(query, top_k=5)
        context_parts.append(format_doc_results(doc_results))

    if route == "METRICS":
        # Generate and execute metrics query
        metrics_sql = text_to_sql(query, ops_db.metrics_schema, llm)
        if is_safe_query(metrics_sql):
            data = ops_db.execute(metrics_sql)
            context_parts.append(f"METRICS:\n{format_table(data)}")

    return generate_answer("\n\n".join(context_parts), query, llm)
```

---

### Use Case 7: Legal Contract Analysis (Contract Database + Case Law)

**Company type**: Law firm, legal tech, corporate legal department
**Data source**: **Contract database** (clause metadata, party names, dates, contract values), case law documents, regulatory texts

Lawyers ask:
- *"Find all contracts with Company X that have a non-compete clause expiring in 2026"* → queries the **contract metadata database** + retrieves the actual clause text
- *"What are the indemnification obligations in our MSA with Acme Corp?"* → retrieves from the specific contract document
- *"How have courts interpreted 'material adverse change' clauses in M&A deals?"* → retrieves from case law

```python
def legal_assistant(query: str, llm, contract_db, case_law_retriever):
    """RAG for legal analysis combining contract DB with case law."""
    context_parts = []

    # Query structured contract metadata
    contracts = contract_db.query("""
        SELECT c.id, c.title, c.counterparty, c.effective_date, c.expiry_date,
               c.contract_value, cl.clause_type, cl.clause_text
        FROM contracts c
        JOIN clauses cl ON c.id = cl.contract_id
        WHERE c.search_vector @@ plainto_tsquery(%s)
        ORDER BY c.effective_date DESC
        LIMIT 10
    """, [query])
    context_parts.append(format_contracts(contracts))

    # Retrieve relevant case law
    case_law = case_law_retriever.search(query, top_k=5)
    context_parts.append(format_case_law(case_law))

    return generate_answer("\n\n".join(context_parts), query, llm,
        system="You are a legal research assistant. Cite specific contract "
               "clauses and case references. Note that this is not legal advice.")
```

---

### Summary: Choosing the Right Data Source Pattern

| Pattern | When to Use | Example |
|---|---|---|
| **Document-only RAG** | Knowledge is in unstructured text | Support docs, policies, research papers |
| **Database-only RAG** (Text-to-SQL) | Knowledge is in structured tables | Metrics, inventory, financial data |
| **Hybrid RAG** (Docs + Database) | Need both facts and context | Product advice (specs DB + reviews), clinical decision support (EHR + literature) |
| **Multi-source RAG** (Docs + DB + APIs) | Complex enterprise assistants | Engineering assistant (code + docs + logs + metrics) |

**Key principle**: The best RAG systems don't just search documents — they **orchestrate across multiple data sources**, choosing the right retrieval strategy for each query. The routing layer (deciding which source to query) is just as important as the retrieval itself.

---

## Conclusion

Building an effective RAG system is not about any single technique — it's about getting every stage right and understanding how they interact:

1. **Clean, well-structured documents** form the foundation
2. **Smart chunking** determines what the retriever can find
3. **Good embeddings** capture the right semantics
4. **Hybrid retrieval + reranking** ensures the best context reaches the LLM
5. **Careful prompt engineering** guides the LLM to generate faithful, cited answers
6. **Continuous evaluation** catches regressions and drives improvement

Start simple (the minimal example above), measure everything, and iteratively improve each stage. The best RAG systems are built incrementally, not designed perfectly upfront.

## References

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) — The original RAG paper
- Anthropic, "Introducing Contextual Retrieval" (2024)
- LangChain Documentation — RAG tutorials and integrations
- LlamaIndex Documentation — Data framework for RAG applications
- RAGAS — Evaluation framework for RAG systems
- MTEB Leaderboard — Embedding model benchmarks

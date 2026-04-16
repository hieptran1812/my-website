---
title: "Basic RAG: Retrieval-Augmented Generation from Scratch"
publishDate: "2026-04-16"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  [
    "rag",
    "retrieval-augmented-generation",
    "ai-agent",
    "llm",
    "embeddings",
    "vector-database",
  ]
date: "2026-04-16"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A comprehensive guide to Retrieval-Augmented Generation (RAG) — how it works, why it matters, and how to build a basic RAG pipeline from scratch using embeddings, vector stores, and large language models."
---

## Introduction

Large Language Models (LLMs) like GPT-4, Claude, and Llama are remarkably capable, but they share a fundamental limitation: their knowledge is frozen at training time. They can hallucinate facts, lack domain-specific knowledge, and cannot access your private data. **Retrieval-Augmented Generation (RAG)** solves these problems by giving LLMs the ability to look things up before answering.

RAG was introduced by [Lewis et al. (2020)](https://arxiv.org/abs/2005.11401) at Facebook AI Research. The core idea is simple: instead of relying solely on parametric memory (model weights), augment the generation process with non-parametric memory (an external knowledge base) at inference time.

This article covers the fundamentals of RAG — the architecture, each component in detail, and how to build a basic pipeline from scratch.

## Why RAG?

Before diving into the architecture, let's understand the problems RAG addresses:

| Problem                | How RAG Helps                                             |
| ---------------------- | --------------------------------------------------------- |
| **Knowledge cutoff**   | Retrieves up-to-date information from external sources    |
| **Hallucination**      | Grounds responses in real documents, reducing fabrication |
| **Domain specificity** | Enables use of private/internal knowledge bases           |
| **Transparency**       | Retrieved sources can be cited, making answers verifiable |
| **Cost efficiency**    | Cheaper than fine-tuning a model on new data              |

RAG gives you the best of both worlds: the reasoning power of LLMs combined with the accuracy and recency of a search engine over your own data.

## RAG Architecture Overview

A basic RAG pipeline consists of two main phases:

### 1. Indexing Phase (Offline)

This is the data preparation step, done ahead of time:

```
Documents → Chunking → Embedding → Vector Store
```

1. **Load** raw documents (PDFs, web pages, markdown, databases)
2. **Chunk** documents into smaller, semantically meaningful pieces
3. **Embed** each chunk into a dense vector using an embedding model
4. **Store** vectors in a vector database for fast similarity search

### 2. Query Phase (Online)

This happens at inference time when a user asks a question:

```
Query → Embed → Retrieve → Augment Prompt → Generate
```

1. **Embed** the user query using the same embedding model
2. **Retrieve** the top-k most relevant chunks from the vector store
3. **Augment** the LLM prompt with the retrieved context
4. **Generate** the final answer using the LLM

## Deep Dive into Each Component

### Document Loading

The first step is ingesting your knowledge base. Common document sources include:

- **Unstructured text**: TXT, Markdown, HTML
- **Documents**: PDF, DOCX, PPTX
- **Structured data**: CSV, JSON, SQL databases
- **Web content**: Web scraping, APIs

Libraries like [LangChain](https://github.com/langchain-ai/langchain), [LlamaIndex](https://github.com/run-llama/llama_index), and [Unstructured](https://github.com/Unstructured-IO/unstructured) provide document loaders for all of these formats.

### Chunking Strategies

Chunking is one of the most impactful decisions in a RAG pipeline. The goal is to split documents into pieces that are:

- **Small enough** to be specific and relevant
- **Large enough** to retain sufficient context
- **Semantically coherent** — not cutting mid-sentence or mid-idea

Common chunking strategies:

#### Fixed-Size Chunking

Split text into fixed-size windows (e.g., 512 tokens) with overlap:

```python
def fixed_size_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
```

**Pros**: Simple, predictable chunk sizes.
**Cons**: May split sentences or paragraphs awkwardly.

#### Recursive Character Splitting

Split by hierarchy of separators (`\n\n` → `\n` → `. ` → ` `), falling back to finer-grained splits only when needed:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document_text)
```

This is the most commonly used strategy and a good default.

#### Semantic Chunking

Group sentences by semantic similarity using embeddings. Adjacent sentences that are semantically similar stay together; breakpoints are inserted where the topic shifts.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

chunker = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)
chunks = chunker.split_text(document_text)
```

**Pros**: Produces more semantically coherent chunks.
**Cons**: Slower and more expensive (requires embedding calls during chunking).

### Embedding Models

Embedding models convert text into dense numerical vectors that capture semantic meaning. Similar texts produce vectors that are close together in the embedding space.

Popular embedding models:

| Model                                    | Dimensions | Provider    | Notes                          |
| ---------------------------------------- | ---------- | ----------- | ------------------------------ |
| `text-embedding-3-small`                 | 1536       | OpenAI      | Good price/performance ratio   |
| `text-embedding-3-large`                 | 3072       | OpenAI      | Higher quality, more expensive |
| `voyage-3`                               | 1024       | Voyage AI   | Strong retrieval performance   |
| `BAAI/bge-large-en-v1.5`                 | 1024       | Open source | Top open-source option         |
| `sentence-transformers/all-MiniLM-L6-v2` | 384        | Open source | Lightweight, fast              |

Example using OpenAI:

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Embed a chunk
vector = get_embedding("RAG combines retrieval with generation.")
print(f"Vector dimension: {len(vector)}")  # 1536
```

**Key principle**: Use the **same embedding model** for both indexing and querying. Mixing models will produce incompatible vector spaces.

### Vector Stores

Vector stores (or vector databases) index and search over embeddings using approximate nearest neighbor (ANN) algorithms. See my [article on vector databases](/blog/machine-learning/ai-agent/vector-database) for a deep comparison.

Quick options to get started:

- **Chroma** — Simple, in-process, great for prototyping
- **FAISS** — Facebook's library, fast and battle-tested
- **Pinecone** — Managed cloud service, zero infrastructure
- **Qdrant** — Open source, rich filtering capabilities
- **Weaviate** — Open source, hybrid search support

Example using Chroma:

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize
ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
client = chromadb.Client()
collection = client.create_collection("my_docs", embedding_function=ef)

# Index chunks
collection.add(
    documents=["chunk 1 text", "chunk 2 text", "chunk 3 text"],
    ids=["id1", "id2", "id3"],
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
)

# Query
results = collection.query(query_texts=["What is RAG?"], n_results=3)
```

### Retrieval

Retrieval is the process of finding the most relevant chunks for a given query. The standard approach is **dense retrieval** — embedding the query and performing a nearest-neighbor search in the vector store.

#### Similarity Metrics

Common distance/similarity functions:

- **Cosine similarity**: Measures the angle between vectors. Most widely used.
- **Euclidean distance (L2)**: Measures straight-line distance. Sensitive to magnitude.
- **Dot product**: Similar to cosine when vectors are normalized.

#### Hybrid Search

Combining dense retrieval (semantic) with sparse retrieval (keyword-based, e.g., BM25) often improves results:

```python
# Pseudocode for hybrid search
dense_results = vector_store.similarity_search(query, k=10)
sparse_results = bm25_index.search(query, k=10)

# Reciprocal Rank Fusion (RRF) to merge results
final_results = reciprocal_rank_fusion(dense_results, sparse_results)
```

Hybrid search helps when:

- Queries contain specific keywords, names, or codes
- The corpus has highly technical or domain-specific terminology

### Prompt Augmentation

Once you have retrieved relevant chunks, you inject them into the LLM prompt as context:

```python
def build_rag_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    return f"""Answer the question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
```

**Best practices for the prompt**:

- Instruct the model to only use the provided context
- Ask it to say "I don't know" when context is insufficient
- Include source references for traceability
- Keep context within the model's context window limit

### Generation

Finally, pass the augmented prompt to the LLM:

```python
from openai import OpenAI

client = OpenAI()

def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # Lower temperature for factual answers
    )
    return response.choices[0].message.content
```

Use a **low temperature** (0.0–0.3) for factual question-answering to reduce hallucination.

## Putting It All Together

Here's a minimal end-to-end RAG pipeline:

```python
import chromadb
from openai import OpenAI

# --- Setup ---
openai_client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("knowledge_base")

# --- Indexing ---
def embed(text: str) -> list[float]:
    resp = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding

documents = [
    "RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLM generation.",
    "Vector databases store embeddings and support fast similarity search using ANN algorithms.",
    "Chunking splits documents into smaller pieces for more precise retrieval.",
    "The embedding model converts text into dense vectors that capture semantic meaning.",
]

for i, doc in enumerate(documents):
    collection.add(
        documents=[doc],
        embeddings=[embed(doc)],
        ids=[f"doc_{i}"]
    )

# --- Query ---
def rag_query(question: str, top_k: int = 3) -> str:
    # Retrieve
    results = collection.query(query_embeddings=[embed(question)], n_results=top_k)
    retrieved = results["documents"][0]

    # Augment
    context = "\n\n".join(retrieved)
    prompt = f"""Answer based on the context below. If unsure, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

    # Generate
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# --- Use it ---
answer = rag_query("What is RAG?")
print(answer)
```

## Evaluation

Evaluating RAG systems requires measuring both **retrieval quality** and **generation quality**:

### Retrieval Metrics

- **Recall@k**: What fraction of relevant documents appear in the top-k results?
- **Precision@k**: What fraction of top-k results are actually relevant?
- **MRR (Mean Reciprocal Rank)**: How high does the first relevant result rank?
- **NDCG**: Considers the position of relevant results (higher is better).

### Generation Metrics

- **Faithfulness**: Does the answer stick to the retrieved context (no hallucination)?
- **Relevance**: Does the answer actually address the question?
- **Completeness**: Does the answer cover all aspects of the question?

Frameworks like [RAGAS](https://github.com/explodinggradients/ragas) automate RAG evaluation:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(result)
```

## Common Pitfalls and How to Avoid Them

### 1. Poor Chunking

**Symptom**: Retrieved chunks are irrelevant or lack context.
**Fix**: Experiment with chunk sizes (500–1500 tokens), add overlap, try semantic chunking.

### 2. Wrong Embedding Model

**Symptom**: Semantically similar queries don't retrieve relevant results.
**Fix**: Use embedding models specifically trained for retrieval (e.g., `text-embedding-3-small`, `bge-large`). Benchmark on your domain.

### 3. Insufficient Context in Prompt

**Symptom**: Model gives generic or incorrect answers despite good retrieval.
**Fix**: Include more retrieved chunks (increase k), provide clearer instructions, add metadata (source, date) to context.

### 4. Context Window Overflow

**Symptom**: Errors or truncated responses with many/large chunks.
**Fix**: Limit total context size, use reranking to select only the best chunks, or use models with larger context windows.

### 5. No Reranking

**Symptom**: The best answer is retrieved but buried at position 8 out of 10.
**Fix**: Add a cross-encoder reranker (e.g., Cohere Rerank, `bge-reranker`) to re-score and re-order retrieved chunks before passing to the LLM.

## Beyond Basic RAG

Once you have a working basic pipeline, consider these advanced techniques:

- **Query transformation**: Rewrite or expand queries for better retrieval (HyDE, multi-query)
- **Reranking**: Use cross-encoders to re-score retrieved results
- **Parent-child chunking**: Retrieve small chunks but pass their parent (larger) chunk to the LLM
- **Metadata filtering**: Pre-filter documents by date, source, or category before vector search
- **Agentic RAG**: Let an LLM agent decide when and how to retrieve, iterating if needed
- **GraphRAG**: Combine knowledge graphs with vector retrieval for better reasoning over relationships

## Conclusion

RAG is one of the most practical and impactful patterns in the LLM ecosystem. It gives your models access to fresh, private, and domain-specific knowledge without the cost and complexity of fine-tuning.

The basic pipeline is straightforward: **chunk → embed → store → retrieve → augment → generate**. Start simple, measure with proper evaluation metrics, and iterate on the components that matter most for your use case — usually chunking strategy and retrieval quality.

## References

- Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAGAS — RAG Assessment](https://docs.ragas.io/)
- [Pinecone Learning Center — RAG](https://www.pinecone.io/learn/retrieval-augmented-generation/)

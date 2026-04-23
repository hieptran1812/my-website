---
title: "Advanced RAG for Financial Documents: Tables, Multimodal Layouts, and Beyond"
publishDate: "2026-04-16"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  [
    "rag",
    "retrieval-augmented-generation",
    "ai-agent",
    "llm",
    "financial-documents",
    "multimodal",
    "table-extraction",
    "document-ai",
  ]
date: "2026-04-16"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A deep dive into building production-grade RAG pipelines for financial documents — handling complex tables, multi-column layouts, charts, scanned PDFs, and mixed modalities with advanced parsing, chunking, and retrieval strategies."
---

## Introduction

![Financial-document RAG pipeline: PDF ingest -> element extract -> multi-vector index -> hybrid retrieve -> structured context](/imgs/blogs/advanced-rag-financial-documents-diagram.png)

Financial documents are among the hardest document types to process with RAG. Annual reports (10-K/10-Q filings), earnings call transcripts, prospectuses, balance sheets, and research reports are packed with:

- **Sophisticated tables** — nested headers, merged cells, multi-level row hierarchies, footnotes
- **Diverse layouts** — multi-column text, sidebars, callout boxes, headers/footers
- **Charts and figures** — bar charts, line graphs, pie charts with embedded data
- **Mixed modalities** — scanned pages alongside digital text, watermarks, logos
- **Domain-specific conventions** — GAAP/IFRS terminology, ticker symbols, fiscal calendars

A naive RAG pipeline that treats documents as flat text will **silently destroy** the structural and numerical information that matters most in finance. This article covers battle-tested techniques for building advanced RAG pipelines that handle these challenges.

If you're new to RAG, start with my [Basic RAG article](/blog/machine-learning/ai-agent/basic-rag) for foundational concepts.

## The Problem with Basic RAG on Financial Documents

Let's look at what goes wrong when you apply a standard text-based RAG pipeline to an annual report:

### Table Destruction

Consider a simple balance sheet:

|                  |     FY2025 |     FY2024 | Change (%) |
| ---------------- | ---------: | ---------: | ---------: |
| Total Revenue    | USD 142.3B | USD 131.0B |      +8.6% |
| Operating Income |  USD 48.7B |  USD 42.1B |     +15.7% |
| Net Income       |  USD 39.2B |  USD 33.8B |     +16.0% |
| Diluted EPS      |  USD 12.41 |  USD 10.69 |     +16.1% |

When a basic text splitter processes this table, it often:

1. **Separates headers from data rows** — "FY2025" ends up in a different chunk than "USD 142.3B"
2. **Loses column alignment** — the model can't tell which number belongs to which year
3. **Breaks merged cells** — multi-level headers become gibberish
4. **Drops footnotes** — critical qualifiers like "excluding discontinued operations" vanish

### Layout Confusion

A typical 10-K filing has:

- Two-column layouts in risk factor sections
- Tables that span full pages
- Footnotes referenced across pages
- Headers and page numbers interleaved with content

Standard PDF-to-text extraction merges columns incorrectly, producing interleaved gibberish like:

```
Risk Factor 1: Market         We compete in a highly
conditions may adversely      fragmented market with
affect our operations.        numerous participants.
```

### Chart Blindness

Basic text extraction completely ignores charts, graphs, and diagrams. In financial documents, these often contain information not stated elsewhere in the text — trend visualizations, segment breakdowns, geographic distributions.

## Architecture for Financial Document RAG

A production-grade pipeline for financial documents requires specialized components at every stage:

```
                    ┌─────────────────────────────────────┐
                    │        Document Ingestion            │
                    │  ┌─────────┐  ┌─────────┐  ┌─────┐ │
                    │  │  OCR    │  │ Layout   │  │Table│ │
                    │  │ Engine  │  │ Detector │  │Parse│ │
                    │  └────┬────┘  └────┬─────┘  └──┬──┘ │
                    │       └────────┬───┘───────────┘    │
                    │           Unified Document           │
                    │           Representation             │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      Multimodal Chunking             │
                    │  ┌────────┐ ┌───────┐ ┌──────────┐  │
                    │  │ Text   │ │ Table │ │  Visual  │  │
                    │  │ Chunks │ │Chunks │ │  Chunks  │  │
                    │  └───┬────┘ └───┬───┘ └────┬─────┘  │
                    └──────┼──────────┼──────────┼────────┘
                           │          │          │
                    ┌──────▼──────────▼──────────▼────────┐
                    │      Multi-Vector Indexing            │
                    │  ┌──────────┐  ┌──────────────────┐  │
                    │  │ Text     │  │ Table/Image      │  │
                    │  │ Embeddings│  │ Summaries+Embeds │  │
                    │  └──────────┘  └──────────────────┘  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      Hybrid Retrieval + Reranking    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      Augmented Generation            │
                    │      (with structured context)       │
                    └─────────────────────────────────────┘
```

Let's dive into each layer.

## Document Ingestion: Layout-Aware Parsing

The ingestion layer must understand the **visual structure** of a document, not just extract raw text.

### Layout Detection Models

Modern document AI models detect and classify regions on a page — text blocks, tables, figures, headers, footers, page numbers — before extracting content from each.

Key tools:

| Tool                            | Approach                  | Table Support | Open Source | Notes                          |
| ------------------------------- | ------------------------- | ------------- | ----------- | ------------------------------ |
| **Unstructured**                | Hybrid (rule + ML)        | Excellent     | Yes         | Best general-purpose parser    |
| **Azure Document Intelligence** | Cloud vision model        | Excellent     | No          | Strong on complex tables       |
| **Amazon Textract**             | Cloud vision model        | Good          | No          | AWS-native                     |
| **Docling** (IBM)               | Layout model + OCR        | Excellent     | Yes         | Strong table understanding     |
| **Marker**                      | Layout model + heuristics | Good          | Yes         | Fast, converts PDF to Markdown |
| **LlamaParse**                  | Multimodal LLM-based      | Excellent     | No          | Uses vision models for parsing |
| **Table Transformer (TATR)**    | Deep learning             | Excellent     | Yes         | Microsoft's DETR-based model   |

### Using Unstructured for Layout-Aware Parsing

[Unstructured](https://github.com/Unstructured-IO/unstructured) is the most popular open-source option. It detects element types (Title, NarrativeText, Table, Image, Header, Footer) and preserves structure:

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="annual_report_2025.pdf",
    strategy="hi_res",              # Use layout detection model
    infer_table_structure=True,     # Extract table structure as HTML
    extract_images_in_pdf=True,     # Extract embedded images
    extract_image_block_types=["Image", "Table"],
    extract_image_block_output_dir="./extracted_images",
)

# Each element has a type and metadata
for el in elements:
    print(f"Type: {type(el).__name__}")
    print(f"Text: {el.text[:100]}...")
    if hasattr(el.metadata, "text_as_html"):
        print(f"HTML: {el.metadata.text_as_html[:200]}...")
    print("---")
```

Output for a table element:

```
Type: Table
Text: Total Revenue 142.3 131.0 +8.6% Operating Income 48.7 42.1 ...
HTML: <table><thead><tr><th></th><th>FY2025</th><th>FY2024</th>...
---
```

The `text_as_html` field preserves the table structure as HTML — this is critical for downstream processing.

### Docling for High-Fidelity Table Extraction

[Docling](https://github.com/DS4SD/docling) from IBM excels at understanding complex table structures, including:

- Spanning cells (rowspan/colspan)
- Multi-level headers
- Hierarchical row labels

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("10k_filing.pdf")

# Export to structured format
doc = result.document

for table in doc.tables:
    # Get as pandas DataFrame — preserves structure perfectly
    df = table.export_to_dataframe()
    print(df)

    # Or as Markdown
    md = table.export_to_markdown()
    print(md)
```

### Handling Scanned Financial Documents

Many financial documents — especially older filings, signed agreements, and faxed reports — are scanned images without selectable text.

Pipeline for scanned docs:

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="scanned_report.pdf",
    strategy="hi_res",
    ocr_languages=["eng"],
    # Use Tesseract or a cloud OCR engine
    hi_res_model_name="yolox",     # Layout detection model
    infer_table_structure=True,
)
```

For higher accuracy on scanned tables, use **Azure Document Intelligence**:

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(
    endpoint="https://<resource>.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<key>")
)

with open("scanned_financial_statement.pdf", "rb") as f:
    poller = client.begin_analyze_document("prebuilt-layout", body=f)

result = poller.result()

for table in result.tables:
    print(f"Table: {table.row_count} rows x {table.column_count} cols")
    for cell in table.cells:
        print(f"  [{cell.row_index},{cell.column_index}] = {cell.content}")
```

## Multimodal Chunking Strategies

Financial documents require **element-aware chunking** — different strategies for text, tables, and images.

### Principle: Chunk by Element Type

Never split across element boundaries. A table should always be one chunk (or, if very large, split by logical sections within the table). A paragraph should not be merged with an adjacent table.

```python
from unstructured.chunking.title import chunk_by_title

chunks = chunk_by_title(
    elements,
    max_characters=1500,
    new_after_n_chars=1000,
    combine_text_under_n_chars=200,
    multipage_sections=True,
)
```

### Table Chunking: Preserve Structure

Tables need special treatment. Here's a strategy that preserves structure while making tables searchable:

```python
def chunk_table_element(table_element) -> dict:
    """Create a rich chunk from a table element."""

    # 1. Keep the raw HTML for faithful reproduction
    html = table_element.metadata.text_as_html

    # 2. Create a Markdown version for embedding
    markdown = html_table_to_markdown(html)

    # 3. Generate a natural language summary for better retrieval
    summary = summarize_table_with_llm(html)

    # 4. Extract key-value pairs for structured queries
    kvs = extract_key_values(html)

    return {
        "type": "table",
        "html": html,
        "markdown": markdown,
        "summary": summary,       # "Revenue grew 8.6% YoY to USD 142.3B..."
        "key_values": kvs,        # {"Total Revenue FY2025": "USD 142.3B", ...}
        "page": table_element.metadata.page_number,
        "source": table_element.metadata.filename,
    }
```

The **summary** is the key innovation here: embedding a natural language description of a table retrieves far better than embedding raw table text.

### Table Summarization Prompt

```python
def summarize_table_with_llm(table_html: str) -> str:
    prompt = f"""You are a financial analyst. Summarize the following table
in 2-4 sentences. Include all key figures, trends, and notable changes.
Mention specific numbers.

Table (HTML):
{table_html}

Summary:"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content
```

Example output:

> "The company's consolidated income statement shows total revenue of USD 142.3B in FY2025, up 8.6% from USD 131.0B in FY2024. Operating income grew faster at 15.7% to USD 48.7B, suggesting margin expansion. Diluted EPS increased 16.1% to USD 12.41, reflecting strong bottom-line growth."

### Visual/Chart Chunking

For charts and figures, use a vision-language model to generate textual descriptions:

```python
import base64
from openai import OpenAI

client = OpenAI()

def describe_financial_chart(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Describe this financial chart in detail. Include:
1. Chart type (bar, line, pie, etc.)
2. All axis labels and values
3. All data points and trends
4. Key takeaways
Be precise with numbers."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                }
            ]
        }],
        temperature=0.1,
    )
    return response.choices[0].message.content
```

## Multi-Vector Indexing

A single embedding per chunk is insufficient for financial documents. Use **multi-vector retrieval** — store multiple representations of each chunk and retrieve based on the best match.

### The Multi-Vector Strategy

```
Original Content          Indexed Representations
─────────────────         ───────────────────────
                     ┌──→ Text embedding (of summary)
Complex Table ───────┼──→ Key-value pairs (structured)
                     └──→ Raw content (stored, not embedded)

                     ┌──→ Text embedding (of description)
Chart/Figure ────────┤
                     └──→ Image embedding (CLIP/SigLIP)

                     ┌──→ Text embedding (of text)
Narrative Text ──────┤
                     └──→ Metadata (section, page, entities)
```

### Implementation with LangChain's MultiVectorRetriever

```python
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import uuid

# Vectorstore for summaries/descriptions (searchable)
vectorstore = Chroma(
    collection_name="financial_docs",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
)

# Docstore for full original content (returned after retrieval)
docstore = InMemoryByteStore()

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=docstore,
    id_key="doc_id",
)

# Index a table: embed the summary, store the full HTML
for table_chunk in table_chunks:
    doc_id = str(uuid.uuid4())

    # Add summary to vectorstore (this is what gets searched)
    retriever.vectorstore.add_documents(
        [Document(
            page_content=table_chunk["summary"],
            metadata={"doc_id": doc_id, "type": "table", "page": table_chunk["page"]}
        )]
    )

    # Store full content in docstore (this is what gets returned)
    retriever.docstore.mset([(doc_id, table_chunk["html"])])
```

**Why this works**: Users ask questions in natural language ("What was the revenue growth?"), which matches better against summaries than raw table content. But the LLM receives the full, structured table for accurate answer generation.

### Combining Text and Image Embeddings

For charts and figures, you can use both text embeddings (of the description) and image embeddings (via CLIP) to enable retrieval from both modalities:

```python
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_image_embedding(image_path: str) -> list[float]:
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze().tolist()

def get_text_embedding_clip(text: str) -> list[float]:
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    return embedding.squeeze().tolist()
```

This enables queries like "show me the revenue trend chart" to match against actual chart images, not just text descriptions.

## Advanced Retrieval for Financial Queries

### Metadata Filtering

Financial queries are often scoped — by year, company, filing type, or section. Pre-filter using metadata before vector search:

```python
# "What was Apple's gross margin in the 2024 10-K?"
results = vectorstore.similarity_search(
    query="gross margin",
    k=10,
    filter={
        "company": "AAPL",
        "filing_type": "10-K",
        "fiscal_year": 2024,
    }
)
```

### Structured Metadata Schema for Financial Docs

```python
metadata_schema = {
    "company": str,           # Ticker or name
    "filing_type": str,       # 10-K, 10-Q, 8-K, earnings transcript
    "fiscal_year": int,
    "fiscal_quarter": int,    # 1-4, None for annual
    "section": str,           # "MD&A", "Risk Factors", "Financial Statements"
    "element_type": str,      # "text", "table", "chart", "footnote"
    "page_number": int,
    "table_title": str,       # For tables: "Consolidated Balance Sheet"
    "date_filed": str,        # ISO date
}
```

### Query Decomposition for Complex Financial Questions

Financial analysts often ask multi-part questions:

> "Compare Apple's and Microsoft's revenue growth and operating margins over the last 3 years."

A single retrieval won't capture all the needed data. Decompose the query into sub-queries:

```python
def decompose_financial_query(query: str) -> list[str]:
    prompt = f"""Decompose this financial question into independent sub-queries
that can each be answered from a single document section or table.

Question: {query}

Return a JSON array of sub-queries."""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)["sub_queries"]

# Result:
# [
#   "Apple total revenue FY2022 FY2023 FY2024 FY2025",
#   "Microsoft total revenue FY2022 FY2023 FY2024 FY2025",
#   "Apple operating income and operating margin FY2022-FY2025",
#   "Microsoft operating income and operating margin FY2022-FY2025"
# ]
```

Then retrieve for each sub-query, deduplicate, and merge:

```python
def multi_query_retrieve(sub_queries: list[str], k_per_query: int = 5) -> list:
    all_results = []
    seen_ids = set()

    for sq in sub_queries:
        results = retriever.invoke(sq)
        for doc in results[:k_per_query]:
            doc_id = doc.metadata.get("doc_id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append(doc)

    return all_results
```

### Cross-Encoder Reranking

After retrieving candidates from multiple sub-queries, rerank to surface the most relevant chunks:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank(query: str, documents: list, top_k: int = 10) -> list:
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(documents, scores), key=lambda x: x[1], reverse=True
    )
    return [doc for doc, score in scored_docs[:top_k]]
```

For financial documents, consider using **Cohere Rerank** or training a domain-specific reranker on financial QA pairs.

## Structured Context Injection

When passing retrieved financial content to the LLM, structure matters. Don't just concatenate chunks — organize them.

### Financial RAG Prompt Template

```python
def build_financial_rag_prompt(
    query: str,
    text_chunks: list[str],
    table_chunks: list[dict],
    chart_descriptions: list[str],
) -> str:

    prompt_parts = [
        "You are a senior financial analyst. Answer the question using ONLY "
        "the provided context. Cite specific numbers. If data is insufficient, "
        "state what's missing.\n"
    ]

    # Add text context
    if text_chunks:
        prompt_parts.append("## Narrative Context")
        for i, chunk in enumerate(text_chunks, 1):
            prompt_parts.append(f"[Text {i}]\n{chunk}\n")

    # Add tables with structure preserved
    if table_chunks:
        prompt_parts.append("## Financial Tables")
        for i, table in enumerate(table_chunks, 1):
            title = table.get("table_title", f"Table {i}")
            # Pass HTML for best structure preservation
            prompt_parts.append(f"### {title}\n{table['html']}\n")

    # Add chart descriptions
    if chart_descriptions:
        prompt_parts.append("## Charts and Figures")
        for i, desc in enumerate(chart_descriptions, 1):
            prompt_parts.append(f"[Figure {i}] {desc}\n")

    prompt_parts.append(f"## Question\n{query}")
    prompt_parts.append(
        "\n## Instructions\n"
        "- Use exact figures from the tables when available\n"
        "- Perform calculations if needed (show your work)\n"
        "- Reference the source (e.g., 'per Table 2' or 'Text 3')\n"
        "- If comparing periods, compute the change and percentage"
    )

    return "\n\n".join(prompt_parts)
```

### Why Pass Tables as HTML?

LLMs understand HTML table structure far better than Markdown or plain text for complex tables:

```html
<!-- LLM can correctly parse this -->
<table>
  <thead>
    <tr>
      <th rowspan="2">Segment</th>
      <th colspan="2">FY2025</th>
      <th colspan="2">FY2024</th>
    </tr>
    <tr>
      <th>Revenue</th>
      <th>% of Total</th>
      <th>Revenue</th>
      <th>% of Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Cloud</td>
      <td>USD 63.4B</td>
      <td>44.6%</td>
      <td>USD 52.1B</td>
      <td>39.8%</td>
    </tr>
    <tr>
      <td>Enterprise</td>
      <td>USD 48.2B</td>
      <td>33.9%</td>
      <td>USD 47.3B</td>
      <td>36.1%</td>
    </tr>
    <tr>
      <td>Consumer</td>
      <td>USD 30.7B</td>
      <td>21.5%</td>
      <td>USD 31.6B</td>
      <td>24.1%</td>
    </tr>
  </tbody>
</table>
```

Multi-level headers with `rowspan`/`colspan` are unambiguous in HTML but lossy in Markdown.

## Handling Specific Financial Document Types

### SEC Filings (10-K, 10-Q)

SEC filings have a predictable structure. Use section-level chunking:

```python
SEC_10K_SECTIONS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}

def tag_sec_section(element, elements_context) -> str:
    """Determine which SEC section an element belongs to."""
    # Use heading detection + regex matching on Item numbers
    for prev_el in reversed(elements_context):
        if "Item" in prev_el.text:
            for code, name in SEC_10K_SECTIONS.items():
                if f"Item {code}" in prev_el.text:
                    return f"Item {code}: {name}"
    return "Unknown"
```

### Earnings Call Transcripts

Transcripts are conversational but contain critical forward-looking statements:

```python
def chunk_earnings_transcript(text: str) -> list[dict]:
    """Split transcript by speaker turns."""
    import re

    turns = re.split(r'\n([\w\s]+(?:CEO|CFO|COO|Analyst|Director).*?):',text)

    chunks = []
    for i in range(1, len(turns), 2):
        speaker = turns[i].strip()
        content = turns[i + 1].strip() if i + 1 < len(turns) else ""

        # Tag speaker role
        role = "management" if any(t in speaker for t in ["CEO", "CFO", "COO"]) else "analyst"

        chunks.append({
            "speaker": speaker,
            "role": role,
            "content": content,
            "type": "transcript_turn",
        })

    return chunks
```

### Fund Fact Sheets and Prospectuses

These are highly visual documents with:

- Performance tables (1Y, 3Y, 5Y, 10Y returns)
- Pie charts (asset allocation, sector breakdown)
- Risk metrics (Sharpe ratio, standard deviation, max drawdown)

Use the multimodal pipeline: image extraction → vision model description → multi-vector indexing.

## End-to-End Example: 10-K Filing RAG

Here's a complete pipeline for processing a 10-K filing:

```python
import chromadb
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import uuid, json

openai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./financial_vectorstore")

# ============================================================
# STEP 1: Parse with layout awareness
# ============================================================
elements = partition_pdf(
    filename="apple_10k_2025.pdf",
    strategy="hi_res",
    infer_table_structure=True,
    extract_images_in_pdf=True,
    extract_image_block_output_dir="./extracted_images",
)

# ============================================================
# STEP 2: Separate element types
# ============================================================
text_elements = [el for el in elements if el.category in ("NarrativeText", "Title")]
table_elements = [el for el in elements if el.category == "Table"]
image_paths = [
    f"./extracted_images/{f}"
    for f in os.listdir("./extracted_images") if f.endswith(".png")
]

# ============================================================
# STEP 3: Summarize tables and images
# ============================================================
def summarize(content: str, content_type: str = "table") -> str:
    prompt = f"Summarize this financial {content_type} in 2-4 sentences with key figures:\n\n{content}"
    resp = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.1
    )
    return resp.choices[0].message.content

table_summaries = [summarize(el.metadata.text_as_html, "table") for el in table_elements]

# ============================================================
# STEP 4: Build multi-vector index
# ============================================================
collection = chroma_client.get_or_create_collection("apple_10k")

def embed(text: str) -> list[float]:
    resp = openai_client.embeddings.create(input=text, model="text-embedding-3-large")
    return resp.data[0].embedding

# Store for original content (keyed by ID)
original_content = {}

# Index text chunks
text_chunks = chunk_by_title(text_elements, max_characters=1500)
for chunk in text_chunks:
    doc_id = str(uuid.uuid4())
    original_content[doc_id] = chunk.text
    collection.add(
        ids=[doc_id],
        embeddings=[embed(chunk.text)],
        documents=[chunk.text],
        metadatas=[{"type": "text", "page": chunk.metadata.page_number or 0}],
    )

# Index table summaries (search) → link to full HTML (retrieve)
for i, (summary, table_el) in enumerate(zip(table_summaries, table_elements)):
    doc_id = str(uuid.uuid4())
    original_content[doc_id] = table_el.metadata.text_as_html
    collection.add(
        ids=[doc_id],
        embeddings=[embed(summary)],
        documents=[summary],
        metadatas=[{"type": "table", "page": table_el.metadata.page_number or 0}],
    )

# ============================================================
# STEP 5: Query
# ============================================================
def financial_rag_query(question: str, k: int = 8) -> str:
    # Retrieve
    results = collection.query(query_embeddings=[embed(question)], n_results=k)

    # Separate text and table results
    text_context = []
    table_context = []

    for doc_id, metadata, document in zip(
        results["ids"][0], results["metadatas"][0], results["documents"][0]
    ):
        original = original_content.get(doc_id, document)
        if metadata["type"] == "table":
            table_context.append(original)
        else:
            text_context.append(original)

    # Build structured prompt
    context = ""
    if text_context:
        context += "## Text Context\n" + "\n---\n".join(text_context) + "\n\n"
    if table_context:
        context += "## Financial Tables\n" + "\n---\n".join(table_context) + "\n\n"

    prompt = f"""You are a senior financial analyst. Answer using ONLY the provided context.
Cite specific numbers and their sources. Show calculations when needed.

{context}

Question: {question}

Answer:"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content

# ============================================================
# STEP 6: Use it
# ============================================================
answer = financial_rag_query(
    "What was Apple's services segment revenue and how did it compare to the prior year?"
)
print(answer)
```

## Evaluation for Financial RAG

Standard RAG evaluation metrics apply, but financial RAG needs additional domain-specific checks:

### Numerical Accuracy

The most critical metric for financial RAG. Evaluate whether numbers in the generated answer exactly match the source documents:

```python
import re

def numerical_accuracy(generated: str, reference: str) -> float:
    """Check if all numbers in the reference appear in the generated answer."""
    ref_numbers = set(re.findall(r'\$?[\d,]+\.?\d*[BMK%]?', reference))
    gen_numbers = set(re.findall(r'\$?[\d,]+\.?\d*[BMK%]?', generated))

    if not ref_numbers:
        return 1.0

    matched = ref_numbers.intersection(gen_numbers)
    return len(matched) / len(ref_numbers)
```

### Table Retrieval Accuracy

Measure whether the correct table was retrieved for a given question:

```python
def table_retrieval_accuracy(queries_with_labels: list[dict], retriever) -> float:
    """
    queries_with_labels: [{"query": "...", "expected_table_id": "..."}]
    """
    correct = 0
    for item in queries_with_labels:
        results = retriever.invoke(item["query"])
        retrieved_ids = [r.metadata.get("doc_id") for r in results[:3]]
        if item["expected_table_id"] in retrieved_ids:
            correct += 1
    return correct / len(queries_with_labels)
```

### Financial QA Benchmark

Build an eval set with questions, expected answers, and source passages from your financial documents:

```python
eval_set = [
    {
        "question": "What was the year-over-year change in operating income?",
        "expected_answer": "Operating income increased 15.7% from USD 42.1B to USD 48.7B",
        "source_table": "consolidated_income_statement",
        "requires_calculation": True,
    },
    # ... more examples
]
```

## Common Pitfalls in Financial Document RAG

### 1. Footnote Loss

Financial tables are meaningless without their footnotes ("\*Adjusted for one-time charges", "Excludes discontinued operations"). Always extract and link footnotes to their parent tables.

### 2. Currency and Unit Confusion

"Revenue of 142.3" — is that millions, billions, or thousands? Always preserve the unit context from the table header ("In millions, except per share data").

### 3. Fiscal Year vs. Calendar Year

A company's FY2025 might end in September 2025, not December. Include fiscal year end dates in metadata.

### 4. Table Title Separation

Table titles are often a separate element from the table itself. Merge them during chunking:

```python
def merge_table_with_title(elements: list) -> list:
    merged = []
    for i, el in enumerate(elements):
        if el.category == "Table" and i > 0 and elements[i-1].category == "Title":
            el.metadata.table_title = elements[i-1].text
        merged.append(el)
    return merged
```

### 5. Not Validating Extracted Numbers

After extraction, always validate that key figures (totals, subtotals) are arithmetically consistent. If "Total Revenue" is the sum of segment revenues, verify it.

## Conclusion

Building RAG for financial documents is substantially harder than general-purpose RAG. The combination of complex tables, multi-column layouts, charts, and domain-specific conventions means each component of the pipeline needs specialized handling.

**Key takeaways**:

1. **Use layout-aware parsers** (Unstructured, Docling, Azure Document Intelligence) — not basic PDF-to-text
2. **Preserve table structure** as HTML throughout the pipeline
3. **Summarize tables and charts** for embedding — raw table text embeds poorly
4. **Multi-vector retrieval** — search on summaries, return original structured content
5. **Structured prompts** — separate text, tables, and charts in the context
6. **Evaluate numerical accuracy** — the metric that matters most in finance
7. **Preserve metadata** — company, filing type, fiscal year, section, units, footnotes

Start with the end-to-end example above, measure with domain-specific evaluation, and iterate on the components where your pipeline is weakest.

## References

- Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Huang, L., et al. (2023). [FinBen: A Holistic Financial Benchmark for Large Language Models](https://arxiv.org/abs/2402.12659)
- [Unstructured Documentation](https://docs.unstructured.io/)
- [Docling — IBM Document Understanding](https://github.com/DS4SD/docling)
- [Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- [LangChain Multi-Vector Retriever](https://python.langchain.com/docs/how_to/multi_vector/)
- [LlamaIndex — SEC Filing Processing](https://docs.llamaindex.ai/)
- [RAGAS — RAG Evaluation](https://docs.ragas.io/)

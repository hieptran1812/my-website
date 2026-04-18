---
title: "Graph RAG: A Complete Guide with Real-World Case Studies"
publishDate: "2026-04-18"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "Graph-RAG",
    "GraphRAG",
    "RAG",
    "LLM",
    "knowledge-graph",
    "retrieval-augmented-generation",
    "vector-database",
    "NLP",
    "AI",
    "information-retrieval",
  ]
date: "2026-04-18"
author: "Hiep Tran"
featured: true
aiGenerated: true
excerpt: "A deep dive into Graph RAG — what it is, why it matters, how to build one from scratch, and how teams at Microsoft, LinkedIn, Siemens, and others deployed it in production. Covers architecture, indexing pipelines, community detection, global vs. local search, hybrid retrieval, evaluation, cost trade-offs, and lessons from real case studies."
---

# Graph RAG: A Complete Guide with Real-World Case Studies

Vector-based RAG is very good at one thing: finding chunks that *look* like the question. But when the question requires connecting dots across many documents — "*summarize the themes in this 10,000-document corpus*", "*trace how this entity is related to that one*", "*which regulations apply to this customer profile across all contracts*" — vector search falls on its face. The top-*k* chunks are semantically similar, but individually none of them answers the question.

**Graph RAG** is the response to that failure mode. Instead of treating a corpus as an unstructured bag of chunks, Graph RAG extracts **entities and relationships** into a **knowledge graph**, uses the graph structure to organize retrieval, and feeds graph-aware context to the LLM.

This article walks through what Graph RAG is, when to use it, how to build one end-to-end, and what actually happened when real teams shipped it.

## What is Graph RAG?

**Graph RAG** is a retrieval pattern where the knowledge base is represented (at least partially) as a **graph of entities and relationships**, and retrieval uses graph traversal, community structure, or subgraph extraction — not just vector similarity — to build the context passed to the LLM.

At a high level:

```
Documents → Entity/Relation Extraction → Knowledge Graph
                                             │
                                             ▼
                         Communities / Subgraphs / Paths
                                             │
User Question ──► Graph + Vector Retrieval ──► Context ──► LLM ──► Answer
```

The graph gives you three things vanilla RAG lacks:

1. **Multi-hop reasoning**: follow edges to connect facts that live in different documents.
2. **Global structure**: communities and hierarchies reveal themes across a whole corpus.
3. **Explicit relationships**: `(Acme, acquired, Beta, 2024)` is a first-class fact, not a sentence buried in a PDF.

### Graph RAG vs. Vector RAG

| Capability | Vector RAG | Graph RAG |
|---|---|---|
| "Find a chunk that mentions X" | Excellent | Good |
| "Summarize the main themes across 10,000 docs" | Poor — top-k chunks are redundant | Strong — community summaries |
| "How is Alice connected to this contract?" | Weak — depends on co-mention | Strong — multi-hop traversal |
| "What are the downstream effects of change X?" | Weak | Strong — forward edges |
| Indexing cost | Low (embed + upsert) | **High** (LLM-based extraction) |
| Query cost | Low | Medium-to-high (multi-stage) |
| Freshness / updates | Easy (upsert chunks) | Harder (entity reconciliation) |
| Explainability | Cite chunks | Cite chunks **+ subgraph paths** |

Graph RAG is not a replacement — it's a layer. In practice, the best systems are **hybrid**: vector search for lexical/semantic hits, graph retrieval for structural queries, fused at the reranker.

### When Graph RAG is worth it

Graph RAG pays for itself when **at least one** of the following is true:

- The corpus has strong **entity structure**: people, companies, drugs, genes, contracts, products, incidents.
- Users ask **global** or **multi-hop** questions, not just lookup questions.
- **Explainability** matters — auditors, clinicians, compliance officers want the chain of evidence.
- The corpus is **large but thematically coherent** (research archive, company wiki, legal corpus) and users ask "what's in here?"-style questions.

If users mostly ask "what does the manual say about error code E47?", stay with vector RAG.

## The Core Idea: From Text to Graph

A knowledge graph has three primitives:

- **Nodes (entities)**: `Person`, `Company`, `Drug`, `Gene`, `Contract`, `Event`, `Concept`.
- **Edges (relations)**: `works_at`, `acquired`, `treats`, `inhibits`, `governs`, `caused_by`.
- **Properties**: `Person.name`, `Event.date`, `Relation.evidence` (the source chunk).

Every fact in the graph should be traceable back to a **source chunk** — this is what makes Graph RAG citable.

### Two schools of Graph RAG

There are two broad design choices for where the graph comes from:

1. **Schema-first (ontology-driven)**. You define the entity and relation types up front. The extractor is constrained to that schema. Best for **bounded domains** (finance, healthcare, legal) where the ontology is well-understood. Higher precision, less recall, easier to evaluate.

2. **Schema-free (open extraction)**. The extractor proposes entities and relations as it reads. Microsoft's **GraphRAG** is in this camp. Best for **open domains** and exploratory corpora. Higher recall, more noise, needs community-level summarization to be useful.

Production systems often combine both: a fixed schema for the "important" types, plus open extraction for long-tail concepts.

## Architecture: End-to-End Graph RAG

A full Graph RAG system has two pipelines — **indexing** (offline, expensive) and **query** (online) — plus a **maintenance** loop.

### The indexing pipeline

```
┌─────────────┐   ┌──────────┐   ┌───────────────────┐   ┌────────────────┐
│  Documents  │──►│  Chunk   │──►│ Entity & Relation │──►│  Canonicalize  │
└─────────────┘   └──────────┘   │    Extraction     │   │  & Deduplicate │
                                 └───────────────────┘   └────────┬───────┘
                                                                  │
                                                                  ▼
┌─────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│ Global Summary  │◄──│ Community Summary  │◄──│  Community Detection │
└─────────────────┘   └────────────────────┘   └──────────────────────┘
                                                                  │
                                                                  ▼
                                                  ┌──────────────────────┐
                                                  │ Graph DB + Vector DB │
                                                  │ (entities, edges,    │
                                                  │  chunks, summaries)  │
                                                  └──────────────────────┘
```

### The query pipeline

```
User Query
    │
    ├──► Query classifier: local | global | hybrid
    │
    ├──► Local: entity linking → subgraph extraction → chunks → rerank → LLM
    │
    ├──► Global: map over community summaries → reduce → LLM
    │
    └──► Hybrid: vector top-k + graph expansion → fuse → rerank → LLM
```

Let's walk through each stage.

## Stage 1 — Chunking and Preprocessing

Chunking for Graph RAG is a bit different from vanilla RAG.

- **Larger chunks** (600–1,200 tokens) are usually better because the extractor needs enough context to identify relations, not just entities. A 200-token chunk might mention "Acme" and "the merger" but miss the verb that connects them.
- **Semantic boundaries** matter: split on headings, paragraphs, or topic shifts, not at arbitrary token counts. A relation split across two chunks is a relation the extractor will miss.
- **Keep document metadata**: `doc_id`, `section`, `page`, `timestamp`, `author`. These become **properties on the edge** and are essential for citation and freshness filtering.

A common mistake is to reuse the same chunks for extraction and for final context. You want **extraction chunks** (larger, for LLM extraction) and **retrieval chunks** (smaller, for returning to the reader LLM). They can be linked by offset so that once you find the right extraction chunk via the graph, you return the tight retrieval chunk to the LLM.

## Stage 2 — Entity and Relation Extraction

This is where most of the cost and most of the quality live. Three approaches:

### 2.1. Prompt-based LLM extraction

Feed the chunk to an LLM and ask for structured output. This is what Microsoft GraphRAG, LlamaIndex PropertyGraphIndex, and most production systems do today.

Example prompt skeleton:

```
You are extracting a knowledge graph from the text below.

Extract entities with types: [Person, Organization, Product, Location, Event, Concept]
Extract relations with types: [works_at, acquired, located_in, caused, partnered_with, ...]

For each entity, return: {name, type, description}
For each relation, return: {source, target, type, description, evidence_span}

Return as JSON. Only include facts explicitly supported by the text.

TEXT:
<chunk>
```

Practical tips that actually matter:

- **Force JSON output** with a schema (structured outputs, function calling, or a JSON-mode model). Free-form extraction is unusable at scale.
- **Ask for evidence spans**: for every extracted relation, return the exact substring that supports it. This becomes your citation and your quality check.
- **Include descriptions, not just names**. "Jordan" by itself is useless; "Jordan, a retired NBA player" lets you disambiguate later.
- **Batch sparsely**. Putting 5 chunks in one prompt to save money sounds clever until the model starts hallucinating cross-chunk relations that don't exist.

### 2.2. Fine-tuned extractors

For narrow domains at high volume, a fine-tuned smaller model (say, 7B–14B) on your ontology will beat prompting a frontier model on **cost per document** by 20–50x, with comparable or better precision. Worth it if you're extracting millions of documents and your schema is stable.

### 2.3. Hybrid: NER + RE + LLM

The classical NLP stack — NER tagger + relation extractor — is still competitive for well-defined relations (UMLS concepts, GLEIF entities, drug-drug interactions). Use it as the first pass; use the LLM to fill gaps and extract long-tail relations.

### The precision/recall knob

Your biggest design knob is **temperature × thoroughness**:

- **High-recall extraction**: "extract every plausible relation." Produces a dense graph, many false positives, good for exploratory search.
- **High-precision extraction**: "only extract relations explicitly stated." Sparse graph, trustworthy edges, good for compliance.

Most teams start too permissive, drown in noise, then dial back. Start strict, loosen deliberately.

## Stage 3 — Canonicalization (Entity Resolution)

This is the single most underestimated step. Raw extraction gives you:

```
"OpenAI", "OpenAI, Inc.", "Open AI", "openai"
"Sam Altman", "S. Altman", "Sam A."
"GPT-4", "GPT 4", "gpt4", "GPT-4 Turbo" (?)
```

If you skip canonicalization, your graph is **broken**: the same entity appears 5 times, with 1/5 the edges each, and no single node has a complete picture.

Approaches, in increasing sophistication:

1. **Normalize surface form**: lowercase, strip punctuation, trim legal suffixes (`Inc.`, `Ltd.`, `GmbH`).
2. **Embedding similarity**: embed `name + type + description`, cluster with threshold. Cheap, catches spelling variants and paraphrases.
3. **LLM-as-a-judge**: for ambiguous pairs, ask an LLM "are these the same entity?" with both descriptions. Expensive but decisive.
4. **External linking**: match to Wikidata / UMLS / GLEIF / internal master data. The gold standard, but only works where you have an authoritative registry.

A practical recipe that works in production:

```
for each candidate entity e:
    1. normalize(e.name)
    2. find top-5 existing entities by embedding(name + type + description)
    3. if top-1 similarity > 0.92 and types match → merge
    4. if 0.80 < sim < 0.92 → LLM disambiguation
    5. else → create new entity
```

Canonicalization must happen for **relations too**. `acquired`, `bought`, `purchased`, `took over` are the same relation. Map them to a controlled vocabulary at write time.

## Stage 4 — Community Detection and Hierarchical Summarization

This is the insight that made Microsoft's GraphRAG paper go viral. Once you have a graph, run a **community detection algorithm** (Leiden, Louvain, label propagation) to cluster densely-connected nodes. Then — and this is the key move — **summarize each community with an LLM**.

```
Graph
  ├── Community A (50 entities, ~200 edges) → LLM summary: "This cluster concerns
  │                                           the company's 2024 acquisitions in
  │                                           the energy sector, led by..."
  ├── Community B → LLM summary: "..."
  └── ...

Communities at level 0 → cluster again → level 1 communities → summaries
                      → cluster again → level 2 communities → top-level themes
```

You now have, for free, a **multi-resolution table of contents** for your corpus. At query time:

- **Local query** ("who is Alice's manager?") → go directly to the node.
- **Global query** ("what are the main risk themes in our contracts?") → retrieve community summaries at the right level and let the LLM map-reduce over them.

Vanilla RAG literally cannot answer the second question, because no single chunk contains the answer. Graph RAG's community summaries are **synthetic chunks** that do.

### Picking a level

- Level 0: ~50–200 entities per community. Use for focused topic queries.
- Level 1: ~500–2,000 entities per community. Use for cross-cutting themes.
- Level 2+: a handful of communities covering the whole corpus. Use for "what's in this corpus?"

For a typical 100k-document corpus, 3 levels is plenty. More levels = more indexing cost, diminishing returns.

## Stage 5 — Storage

You need **two stores** (sometimes fused):

| Store | Contents | Typical tech |
|---|---|---|
| **Graph store** | Nodes, edges, properties, community membership | Neo4j, Memgraph, NebulaGraph, TigerGraph, Kùzu, FalkorDB, or a plain Postgres + `ltree` / recursive CTE for small graphs |
| **Vector store** | Chunk embeddings, entity description embeddings, community summary embeddings | pgvector, Qdrant, Weaviate, Milvus, LanceDB |

Some new tools (Neo4j + vector index, Weaviate + graph module, Kùzu) fuse the two. For corpora under ~10M nodes, Postgres + pgvector is often enough and avoids the operational cost of a dedicated graph DB.

**What you store in the vector index matters**:

- Chunk embeddings — for lexical/semantic top-k
- Entity embeddings (`name + type + description`) — for entity linking
- Relation description embeddings — for "what relations are relevant to this query?"
- Community summary embeddings — for global search

All four are queried at different stages.

## Stage 6 — Retrieval Strategies

This is where Graph RAG systems diverge most. There are three canonical retrieval modes.

### 6.1. Local search (entity-centric)

The user's query mentions specific entities, and the answer lives in their neighborhood.

```
1. Extract entities from the query (NER + embedding match against entity index)
2. Pull the 1- or 2-hop subgraph around those entities
3. Collect associated chunks (from the edges' evidence spans)
4. Rerank chunks by query relevance
5. Compose context: entities + their attributes + relations + chunks
6. Feed to LLM
```

This is the mode that replaces "top-k vector search" for entity-rich queries.

### 6.2. Global search (community-centric)

The user's query is about themes, patterns, or aggregates. No specific entity anchor.

```
1. Retrieve top-N community summaries at the appropriate level
   (by embedding similarity between query and summary)
2. For each summary, ask the LLM: "given this summary, what's a partial answer
   to the query? Return a point score 0-100."
3. Aggregate partial answers, weighted by score
4. Ask the LLM to produce the final synthesized answer from the aggregates
```

This is the **map-reduce** pattern from the GraphRAG paper. It's expensive (many LLM calls per query) but it's the only way to answer true global questions over large corpora.

### 6.3. Hybrid / DRIFT search

Most real queries are in between. Microsoft's **DRIFT** and similar hybrid strategies:

```
1. Vector top-k to find candidate chunks
2. Extract entities from those chunks → graph anchors
3. Expand via graph (1-hop neighbors + edges)
4. Collect community summaries for those neighborhoods
5. Fuse: chunks + local subgraph + community context
6. Rerank all candidates
7. Feed the top to the LLM
```

In production systems the query classifier is often itself an LLM call — cheap, ~100 tokens — that decides local / global / hybrid. You can skip it and always do hybrid at the cost of latency.

### Ranking and fusion

When you have N candidate evidence items from M different retrievers (vector, graph, community, keyword), you need **rank fusion**. The workhorses:

- **Reciprocal Rank Fusion (RRF)**: cheap, no tuning, good default.
- **Learned reranker**: cross-encoder (e.g., bge-reranker) over `(query, evidence)` pairs. Better quality, adds latency.
- **LLM-as-reranker**: highest quality, highest cost. Worth it for final top-20 → top-5.

## Stage 7 — Generation

Graph RAG changes the prompt shape. You're not just pasting chunks — you're pasting **structured context**:

```
## Entities
- Alice Chen (VP Engineering): Joined 2019, reports to CEO...
- Project Orion: Cloud platform, launched 2024, owner = Alice Chen

## Relations (with evidence)
- (Alice Chen) --[owns]--> (Project Orion) [src: doc_42, p.3]
- (Project Orion) --[depends_on]--> (AWS Aurora) [src: doc_88, p.1]

## Supporting chunks
[chunk 1, doc_42 p.3]: "...Alice took over ownership of Orion in Q2 2024..."
[chunk 2, doc_88 p.1]: "...Orion runs on Aurora with a Redis sidecar..."

## Community context
This project is part of the "Platform Modernization" initiative covering 12 projects...

## Question
<user's query>

Ground your answer in the entities and chunks above. Cite using [doc_id, page].
```

Give the LLM the schema it's operating over. If it knows that an edge has a `date` property and a `confidence` field, it will use them in the answer. Without structure, it'll ignore half of the information you retrieved.

## Evaluation

Evaluating Graph RAG is **harder** than vanilla RAG because the quality depends on the graph **and** the retrieval **and** the generation. Evaluate each layer:

### Graph quality

- **Entity precision/recall**: sample N chunks, have humans annotate, compare.
- **Relation precision**: per-edge, does the evidence span actually support the relation?
- **Coverage**: what fraction of named entities in the corpus are in the graph?
- **Canonicalization quality**: number of duplicate entities, measured by human spot-check or external linking.

### Retrieval quality

- **Recall@k** on a labeled set of question → relevant chunks/subgraphs.
- For graph queries: **path correctness** — did the returned subgraph contain the true path?

### End-to-end quality

- **Faithfulness**: does the answer only claim things supported by the retrieved context? (Use an LLM judge or a claim-by-claim NLI check.)
- **Completeness**: for global queries, how many of the ground-truth themes did the answer cover?
- **Citation correctness**: when the answer cites `[doc_42]`, does that doc actually support the claim?

The **RAGAS**, **TruLens**, and **Arize Phoenix** libraries cover most of these. For global queries, the GraphRAG paper's "comprehensiveness + diversity" LLM-judge pairwise comparison is the de-facto benchmark, though it's noisy and expensive.

## Cost: What Graph RAG Actually Costs

Vanilla RAG indexing cost is roughly `$/1M tokens × corpus_size × embedding_factor`. Graph RAG adds:

| Component | Rough cost (vs vanilla RAG indexing) |
|---|---|
| Entity/relation extraction | 10–50× (one LLM call per chunk) |
| Canonicalization (LLM-in-the-loop) | 0.5–3× |
| Community detection | Negligible (CPU) |
| Community summarization | 1–5× (one LLM call per community × levels) |
| **Total indexing** | **15–60× vanilla RAG** |

Query-time, hybrid search adds maybe 1.5–3× latency and 2–5× cost vs. vanilla RAG, mostly from reranking and the additional LLM orchestration.

The punchline: **Graph RAG is a capital-expenditure pattern**. You pay a lot to build the index, then amortize over many queries. It makes sense when:

- the corpus is relatively stable,
- queries are frequent and high-value,
- the extra quality justifies the cost.

For a corpus that changes hourly and serves 100 queries/day, vanilla RAG wins. For a corpus that serves 10,000 queries/day over a stable archive, Graph RAG pays for itself quickly.

### Freshness and incremental updates

Keeping the graph fresh is genuinely hard:

- **Append**: easy — extract, link to existing entities, add edges.
- **Update** (doc revised): hard — you may need to retract old edges.
- **Delete**: hardest — garbage-collect edges whose only evidence was the deleted doc; rebuild communities if the deletion is topologically significant.

Most production systems do **periodic rebuilds** (nightly/weekly) for community structure and **incremental updates** for entities/edges, with a "fresh" vector index layered on top for bleeding-edge documents.

---

# Case Studies

Now for the interesting part: what actually happened when teams shipped Graph RAG.

## Case Study 1 — Microsoft: GraphRAG on the Podcast Transcripts & VIINA Corpora

Microsoft Research introduced **GraphRAG** (Edge et al., 2024) with a public paper and open-source implementation. The core experiment compared Graph RAG to vanilla vector RAG on two corpora:

- **VIINA** — a ~1M-token news dataset on the Russia-Ukraine conflict.
- **Podcast transcripts** — a ~1M-token collection of technology podcast episodes.

The benchmark was **global sensemaking questions** — "what are the main themes?", "summarize the key actors and their relationships" — evaluated head-to-head by LLM judges against a naive RAG baseline.

### What they built

- Schema-free LLM extraction of entities, relations, and claims.
- Hierarchical Leiden community detection → LLM-generated community summaries at 3 levels.
- Query-time: **local** (entity neighborhood) and **global** (map-reduce over community summaries).

### What they found

On global sensemaking questions, Graph RAG produced answers that were judged **more comprehensive and more diverse** than vanilla RAG by LLM judges roughly 70–80% of the time. On naive lookup questions, the two were comparable.

### The honest limitations

The paper and subsequent analyses were candid about trade-offs:

- **Indexing cost was ~25–50× vanilla RAG**, dominated by extraction and summarization.
- **Freshness was poor** — incremental updates required careful engineering outside the paper's scope.
- **Lookup accuracy was not improved** — for "when did X happen" questions, the graph added complexity without benefit.

The paper was influential because it quantified what practitioners already suspected: **vanilla RAG is structurally incapable of answering global questions**, and a graph-based index is a principled fix.

**Takeaway**: Graph RAG shines on **global queries** over **stable corpora**. If your users don't ask global questions, the index cost isn't justified.

## Case Study 2 — LinkedIn: Customer Service Ticket Assistant

LinkedIn's engineering team published a paper (Xu et al., "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering", 2024) describing a production deployment for customer service.

### The problem

Customer service agents handle thousands of tickets daily. Historical tickets are a goldmine, but they're noisy, duplicative, and filled with conversational fluff. Vanilla RAG over raw tickets returned **semantically similar but often irrelevant** tickets, because the surface language of two tickets can match while the underlying issue is completely different.

### What they built

LinkedIn's team constructed a **knowledge graph of past tickets** with nodes for:

- **Issue** (the customer's actual problem)
- **Root cause**
- **Resolution**
- **Customer attributes** (product, segment, tenure)

Edges connected issues to their causes and resolutions. Extraction was LLM-based with a fixed schema.

At query time, they:

1. Extracted structured features from the new ticket.
2. Retrieved the **subgraph** of matching issue–cause–resolution paths.
3. Fed both the subgraph and the top-k raw tickets to the LLM.

### Reported results

The paper reports a **meaningful reduction in per-issue resolution time** (roughly 28.6% in one internal measurement) versus the vanilla-RAG baseline, driven primarily by more accurate retrieval of *analogous* past cases rather than just *similar-looking* ones.

### What to steal from this design

- **Fixed, shallow schema**: Issue → Cause → Resolution is three node types and two edge types. Don't over-engineer the ontology.
- **Structured ticket summary as the "chunk"**: they summarized each ticket into a canonical form before extraction, which cut noise.
- **Hybrid retrieval**: the graph gave them analogical matches; vector search caught the long tail. Neither alone was sufficient.

**Takeaway**: A small, domain-specific schema beats a big generic one. Three node types with high precision outperform twenty types with noisy extraction.

## Case Study 3 — Siemens / Industrial Knowledge Graphs

Siemens has been public about using knowledge graphs for **industrial documentation** — product manuals, service bulletins, incident reports, parts catalogs. Technicians servicing a turbine or MRI machine can't wait through a lossy RAG query; they need the **exact** bolt, the **exact** torque, the **exact** prior incident.

### The problem

- Documentation spans tens of thousands of PDFs across product lines and decades.
- A single machine has a part hierarchy (machine → subsystem → assembly → component → fastener) that vanilla RAG can't navigate.
- Safety and regulatory requirements demand **auditable citations** — the answer must point to a specific manual, page, and revision.

### What they built

A combined approach that ingests structured BOM data (from CAD/PLM systems) alongside unstructured documents, linking extracted entities to the canonical product ontology. Queries traverse the part hierarchy plus cross-references to incident and service data, then fall back to text chunks for the prose.

### What mattered

- **The ontology was not invented** — it came from the PLM system. The graph was a **union** of the structured master data and LLM-extracted facts.
- **Edges carry revisions**: a relation is valid for a specific machine revision, which lets the system answer "what is the torque spec for serial number X, built in 2019?" without conflating it with the 2023 revision.
- **Every edge has a source and confidence**: structured master data = 1.0; LLM-extracted = 0.6–0.9. The LLM is told to prefer high-confidence edges.

**Takeaway**: When you already have a master data source, the graph should **extend** it, not replace it. LLM extraction fills gaps around the structured spine.

## Case Study 4 — NVIDIA / Healthcare & Biomedical Research

NVIDIA has written extensively about biomedical Graph RAG, and many pharma and academic groups have published similar pipelines. The pattern is remarkably consistent:

### The problem

A biomedical researcher asks: *"What genes are linked to both Parkinson's disease and mitochondrial dysfunction, and which of them are targets of currently approved drugs?"*

This is a three-hop query with entity typing at each hop. Vanilla RAG over PubMed abstracts returns abstracts that mention *any* of the terms; almost none answer the question. The user has to read 50 abstracts to find the 2 that do.

### What the graph looks like

- Nodes: `Disease`, `Gene`, `Protein`, `Drug`, `Pathway`, `Phenotype`.
- Edges: `associated_with`, `inhibits`, `upregulates`, `treats`, `contraindicated_with`.
- Canonicalized to **UMLS / MeSH / ChEMBL / Open Targets** identifiers — this is non-negotiable. "TP53" and "p53" and "tumor protein p53" must collapse to one node.

### What they built

- Extraction via a biomedical-fine-tuned model (BioBERT-lineage or a fine-tuned 7B LLM).
- **Forced linking** to UMLS at extraction time: if an entity can't be linked, it either goes to a "staging" queue for human review or is dropped (depending on policy).
- Query-time: entity linking on the query, then Cypher/openCypher traversal over the graph, with text chunks attached as evidence.

### What typically shows up in results

- **Meaningful recall improvements** on multi-hop biomedical questions versus vector-only baselines, often in the 15–35 percentage-point range depending on the benchmark.
- **Latency improvements** because the graph prunes candidates before the LLM sees them.
- **Easier regulatory review**: every claim in the answer cites a specific paper via the edge that supports it.

**Takeaway**: In regulated / high-stakes domains, the graph isn't just a retrieval aid — it's the **audit log**. The edges *are* the explanation.

## Case Study 5 — Bloomberg / Financial Entity Intelligence

Bloomberg and similar financial data providers have long operated entity graphs (tickers, issuers, instruments, events, filings). Graph RAG has become a natural extension.

### The problem

An analyst asks: *"What's the exposure of our credit portfolio to suppliers of Company X that have been downgraded in the last 90 days?"*

This requires:

1. Resolving Company X to a LEI.
2. Traversing a supplier graph (derived from filings, news, structured data).
3. Filtering suppliers by rating action edges with date properties.
4. Joining back to the internal portfolio.

### Architecture

- The graph is populated from **filings, news, press releases, earnings calls**, with LLM extraction adding edges that structured feeds miss.
- Entity IDs are **LEI / ticker / ISIN** — no ambiguity allowed.
- Every edge has a timestamp, source URL, and confidence.
- Queries combine Cypher (structural) with vector search (news prose).

### Lessons

- **Time is a first-class property**: `(Company A) --[supplier_of { start: 2022-01, end: 2024-09 }]--> (Company B)`. A supplier relationship that ended last quarter should not show up in today's answer.
- **Contradiction is normal**: two news sources disagree. The graph stores both with provenance; the LLM is told to flag the conflict rather than pick a winner.
- **Real-time is possible but expensive**: streaming extraction from news wires means pipeline engineering, not just model selection.

**Takeaway**: Temporal edges and strict entity identity are what make financial Graph RAG work. The LLM is actually the *least* interesting part.

## Case Study 6 — A "Graph RAG that failed" story

Not every Graph RAG project succeeds. A pattern we see often in post-mortems:

A team took a 50,000-document internal wiki and built a full GraphRAG-style pipeline: LLM extraction, Leiden communities, hybrid search. It took 8 weeks and meaningful LLM-spend.

On evaluation, it was roughly tied with vanilla RAG and substantially more expensive.

Why?

- **Users mostly asked lookup questions**. "How do I request access to X?" "What's the on-call rotation?" The graph added nothing.
- **The wiki was shallow**: pages were short, entities were ad-hoc, relationships were implicit. The extractor produced a fluffy graph with lots of nodes and weak edges.
- **Canonicalization was skipped**. "Access Request Process" and "ARP" and "the access flow" became three nodes. The graph was fragmented and the community summaries were muddled.

The lesson: **Graph RAG is not a silver bullet**. If your corpus doesn't have strong latent structure, or if users don't ask structural questions, the index cost is pure waste. Vanilla RAG with a good reranker is often the right answer.

Before building Graph RAG, do the cheap experiment: sample 50 real user queries, classify them (lookup / multi-hop / global), and estimate what fraction would *benefit* from a graph. If it's under ~30%, think hard.

---

## Common Pitfalls and How to Avoid Them

1. **Skipping canonicalization.** Symptom: your graph has "GPT-4", "GPT 4", and "gpt4" as three separate nodes. Fix: normalize + embedding-cluster + LLM disambiguation from day one.

2. **Over-engineering the ontology.** 40 node types and 100 edge types sound impressive but the extractor won't respect them consistently. Start with 5 node types and 10 edge types; grow as you learn.

3. **Uniform chunks for extraction and retrieval.** Extraction needs context (larger chunks), retrieval needs precision (smaller chunks). Use both, linked by offsets.

4. **No evidence spans.** If every edge doesn't have a source chunk and character range, you cannot cite, cannot audit, and cannot debug. Make evidence a hard requirement.

5. **Trusting the extractor.** Run human spot-checks on a sample of 100 extractions every week. Extraction quality drifts silently when prompts or models change.

6. **One-shot ingestion.** Treating the graph as write-once-read-many breaks the moment documents are updated. Build the update/delete paths on day one, even if minimal.

7. **No query classifier.** Blindly running hybrid retrieval for every query wastes latency and money on lookup queries that vector RAG handles in 50ms.

8. **Evaluating only end-to-end.** When the system is wrong, you can't tell if the graph is wrong, the retrieval is wrong, or the LLM is hallucinating. Layered evaluation (graph → retrieval → answer) is the only way to debug at scale.

9. **Ignoring time.** Most real-world graphs are temporal. An edge without a timestamp is a landmine — eventually something changes and your answers are wrong in ways that are hard to notice.

10. **Using Graph RAG when vanilla RAG would do.** This is the biggest failure mode. Do the cheap evaluation first.

## A Minimal Graph RAG Recipe

If you're building a first version and want something that works end-to-end in a week:

1. **Schema**: 3–5 node types, 5–10 edge types, relevant to your domain. Don't overthink.
2. **Chunk** at 800 tokens with 100-token overlap on paragraph boundaries.
3. **Extract** with a frontier LLM in JSON mode, requiring `evidence_span` on every relation. Start with temperature 0 and a strict prompt.
4. **Canonicalize** by normalizing names + embedding clustering at similarity 0.90 + type match. Skip LLM disambiguation in v1.
5. **Store** in Postgres with `pgvector` + recursive CTEs for traversal. Graduate to Neo4j/Kùzu only when you feel pain.
6. **Communities**: run Leiden at 2 levels; summarize each with an LLM.
7. **Query**: LLM classifier → `local` / `global` / `hybrid`. For `local`, entity-link + 1-hop + chunks. For `global`, map-reduce over community summaries. For `hybrid`, vector top-k + 1-hop expansion + RRF fusion.
8. **Generate** with structured context (entities, relations, chunks, community context) and enforce citations.
9. **Evaluate** on 50 hand-labeled questions across lookup / multi-hop / global. Beat vanilla RAG **end to end** before scaling.

That's a working Graph RAG. Everything else — entity resolution to Wikidata, streaming extraction, fine-tuned extractors, temporal edges, DRIFT — is iteration on that spine.

## Conclusion

Graph RAG is not magic and it's not free. It's a principled answer to a specific question: *how do you let an LLM reason over the structure of a large corpus, not just the surface?*

Where it works — stable corpora with entity structure, global or multi-hop queries, regulated/auditable settings — it is transformative. Where it doesn't — shallow corpora, lookup-only queries, high-churn documents — it's an expensive mistake.

The teams that succeed with Graph RAG share a few habits:

- They start from a clear, small ontology tied to real user needs.
- They invest in canonicalization and evidence spans before anything fancy.
- They evaluate layer-by-layer, not just end-to-end.
- They treat the graph as a living system with updates, audits, and freshness budgets.
- They know when **not** to use it.

If you're deciding whether to build one, the most valuable thing you can do is spend a day classifying 100 real queries. That answer tells you more than any benchmark.

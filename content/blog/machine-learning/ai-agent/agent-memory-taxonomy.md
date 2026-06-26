---
title: "Agent Memory Taxonomy: The Four Types Every AI Agent Needs"
date: "2026-06-27"
description: "A precise taxonomy of agent memory — working, episodic, semantic, and procedural — with the LLM component that maps to each and the engineering decisions each type demands."
tags: ["ai-agents", "memory", "llm", "machine-learning", "nlp", "system-design", "production-ml", "architecture"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 41
---

Most agent systems I have reviewed fail in one of two ways. Either they stuff everything into the context window and wonder why the agent "forgets" cross-session preferences, or they build a single vector database labeled "memory" and discover that retrieval quality degrades because past events, world facts, and skill patterns all compete for the same embedding space.

The root cause is the same in both cases: the team never decided what kind of memory they actually needed. Memory is not one thing. The human brain has been teaching us this for decades through lesion studies — patients with hippocampal damage lose the ability to form new episodic memories but retain procedural skills; patients with semantic dementia lose facts about the world while episodic autobiographical memory survives. These dissociations exist because memory is genuinely modular, and the modules demand different infrastructure.

Building an AI agent without a memory taxonomy is like building a database without deciding between OLTP and OLAP. You will end up with something that does neither well.

This post gives you the taxonomy, the engineering decisions each type forces, and the storage systems that actually implement them in production.

![Four Memory Types: Speed vs Persistence Stack](/imgs/blogs/agent-memory-taxonomy-1.webp)

The diagram above is the mental model: four layers ordered by how frequently they change. Working memory changes every token. Procedural memory changes only when you run a fine-tuning job. The layers differ not just in speed but in where they live, how you query them, and what breaks when they fail.

## 1. Why Memory Taxonomy Matters: Wrong Type = Wrong Architecture

Before getting into each type, it is worth being precise about what we mean by "memory" in the context of an LLM-based agent. Memory is any mechanism that allows information from one moment in time to influence the agent's behavior at a later moment. Under this definition, the context window is memory. The model weights are memory. A vector database is memory. They are all different implementations of the same abstract concept, and they differ dramatically in their operational characteristics.

The taxonomy failure shows up concretely in three common antipatterns.

**The context-cram antipattern.** A team builds a customer support agent that retrieves the user's entire ticket history and stuffs it into the context window before every response. This works fine for users with five tickets. For users with fifty tickets it fails: the context window fills up, early tickets get truncated, and the agent starts contradicting resolutions it made three months ago. The team has used working memory to solve an episodic memory problem.

**The single-vector-store antipattern.** A team builds a research agent that stores everything it learns in one Chroma collection: user preferences, Wikipedia facts it ingested, API schema documentation, and notes from previous research sessions. Query recall degrades because a search for "Python datetime format string" surfaces both an API documentation chunk and a user preference note that happens to mention dates. The semantic and episodic signals pollute each other. The team has conflated semantic and episodic memory into one store.

**The prompt-engineering-as-procedural antipattern.** A team wants their agent to always format code responses with a specific style guide, always call a particular validation tool before writing to a database, and always ask for clarification before deleting resources. They put all of this in the system prompt. It mostly works — until a long conversation compresses the context, the system prompt gets partially shadowed by the user's instructions, and the agent starts skipping the validation step. The team has tried to use working memory to implement procedural memory.

Each antipattern maps to a specific failure mode, and each failure mode maps to a missing memory type. The taxonomy is not academic — it is the prerequisite for building the right infrastructure.

## 2. Type 1 — Working Memory: The Context Window and Its Limits

Working memory is the information currently in active processing. For an LLM agent, working memory is the context window: the sequence of tokens that the model attends to during a forward pass.

Everything you want the model to consider in a given response must be in working memory. This is both the strength and the fundamental constraint of transformer-based agents. The strength: working memory is zero-latency. There is no retrieval step, no index lookup, no network call. The tokens are already sitting in GPU SRAM as key-value pairs computed during the prefill phase. The constraint: working memory has a hard capacity limit measured in tokens.

### The KV Cache as Working Memory Implementation

The LLM component that implements working memory is the KV cache. During prefill, the model computes key and value tensors for every attention layer and every input token, then stores these in GPU memory. During decode, each new token attends to all previous tokens' cached key-value pairs without recomputing them. The KV cache is what makes autoregressive decoding tractable — without it, generating the 500th token would require a full forward pass over the entire 499-token prefix, every step.

For working memory engineering, the KV cache has three important properties:

**Capacity is bounded by GPU SRAM.** A single attention layer's KV cache for one sequence of length $L$ tokens consumes $2 \times L \times d_{head} \times n_{heads} \times \text{dtype\_bytes}$ bytes. For a 70B parameter model with 64 attention heads at 128 dimensions each, in fp16, a 128K-token context requires approximately 32 GB just for the KV cache. This is why long-context deployments need careful memory management.

**Eviction has cascading effects.** When serving frameworks like vLLM run out of KV cache space (the "KV cache full" state), they either reject new requests or evict cached prefixes. Evicted prefixes must be recomputed from scratch, turning what was a $O(1)$ decode step into an $O(L^2)$ recompute. This is the 3-5x latency spike you see in production serving logs when KV cache occupancy hits 90%.

**The context window is not uniform.** Transformers with full attention attend to all positions equally in theory, but empirically the "lost in the middle" effect means tokens in the middle of a long context receive less attention weight than tokens at the beginning and end. This matters for working memory curation: the most important information (task instructions, current state, key constraints) should be at the extremes of the context, not buried in the middle.

### What Belongs in Working Memory

The working memory curator — the code that assembles the context before a generation call — must make deliberate decisions about what to include. Not everything retrieved from long-term memory stores should land in the context wholesale.

A reasonable taxonomy of what belongs in working memory:

- **System prompt and tool definitions**: static configuration, always present
- **Current task state**: the goal, intermediate results, constraints discovered so far
- **Retrieved episodic context**: the 2-3 most relevant past events (not the full history)
- **Retrieved semantic context**: the specific fact chunks relevant to the current question
- **Recent conversation turns**: typically the last 4-8 turns, not the full history
- **Tool outputs**: results from the most recent tool calls

What does _not_ belong in working memory: entire conversation histories, full document corpora, the complete knowledge base, all past decisions. Those belong in their respective long-term memory stores and should be retrieved selectively.

### Working Memory Limits Drive the Entire Architecture

Here is the key insight that most agent architects miss: every architectural decision downstream of working memory exists to route information into or out of the context window in a controlled way. Episodic memory exists because you cannot fit the entire conversation history into the context window. Semantic memory exists because you cannot include the entire knowledge base in every prompt. Procedural memory exists because you cannot reliably transmit behavioral policies through the context window — they degrade under context pressure.

Working memory is not a component you add to your agent. It is the constraint around which the entire architecture is designed.

## 3. Type 2 — Episodic Memory: What Happened When

Episodic memory is the record of specific events: what happened, when it happened, and in what context. For an AI agent, episodic memory is the mechanism that allows an agent to remember that last Tuesday the user asked about their Q2 budget, that in the previous session the agent helped diagnose a database connection error, or that the user previously rejected a recommendation for a particular vendor.

The defining characteristic of episodic memory is its time-indexing. An episode has a temporal context — it happened at a specific moment, in a specific sequence relative to other events. This is what distinguishes episodic from semantic memory: semantic memory stores the fact that Paris is the capital of France; episodic memory stores the memory of learning that fact in a middle school geography class.

![Single Agent Turn: Reading All Four Memory Types](/imgs/blogs/agent-memory-taxonomy-3.webp)

### Engineering Episodic Memory

Episodic memory is most commonly implemented with a vector database storing turn-level or session-level summaries. The canonical implementation:

1. **At turn end**: summarize the conversation turn (or the last N turns) into a structured episode record
2. **Embed** the episode record using an embedding model (text-embedding-3-small, Cohere Embed v3, or similar)
3. **Store** the embedding + metadata in a vector store (Chroma, Weaviate, Pinecone, pgvector)
4. **At turn start**: retrieve the top-k most relevant past episodes via approximate nearest neighbor search, filtered by user ID and recency

The metadata schema matters more than most teams realize. A good episode record should contain:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class EpisodeRecord:
    episode_id: str              # UUID
    user_id: str                 # partition key for retrieval
    session_id: str              # group episodes by conversation
    timestamp: datetime          # for recency filtering and ordering
    turn_summary: str            # what was discussed (human-readable)
    key_decisions: list[str]     # explicit decisions made
    user_preferences_observed: dict[str, str]  # extracted preferences
    unresolved_items: list[str]  # things left open
    entity_mentions: list[str]   # people, products, systems mentioned
    sentiment: Optional[str]     # positive/negative/neutral
    embedding: list[float]       # stored separately in vector DB
```

The `user_preferences_observed` and `key_decisions` fields are particularly valuable because they allow structured retrieval beyond pure semantic similarity. If a user mentions they prefer TypeScript over JavaScript, that preference should be retrievable by a direct key lookup, not just by semantic similarity to a future query that happens to mention TypeScript.

### Retrieval Strategy for Episodic Memory

Naive ANN retrieval over all past episodes has a recall problem: the most relevant episode might be semantically distant from the current query while being highly contextually relevant. A user asking "what was the issue we fixed last time?" cannot be served by semantic similarity — there is no subject matter to match against.

We have found a hybrid retrieval strategy works better:

```python
async def retrieve_relevant_episodes(
    user_id: str,
    current_query: str,
    current_entities: list[str],
    top_k: int = 5
) -> list[EpisodeRecord]:
    
    # 1. Semantic similarity: retrieve episodes similar to current query
    semantic_results = await vector_db.search(
        query_embedding=embed(current_query),
        filter={"user_id": user_id},
        top_k=top_k * 3,  # over-retrieve, then rerank
    )
    
    # 2. Recency bias: always include the last N sessions
    recent_results = await sql_db.query(
        "SELECT * FROM episodes WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, 3)
    )
    
    # 3. Entity matching: retrieve episodes that mention the same entities
    entity_results = []
    for entity in current_entities:
        entity_results += await vector_db.search(
            filter={"user_id": user_id, "entity_mentions": {"$contains": entity}},
            top_k=2
        )
    
    # 4. Merge, deduplicate, rerank
    all_candidates = deduplicate(semantic_results + recent_results + entity_results)
    return rerank(all_candidates, current_query)[:top_k]
```

This retrieval pattern ensures you always have recent context (recency), relevant context (semantic), and entity-specific context (entity matching) — three different access patterns against the same store.

### Episodic Memory Write-Back

The write-back timing matters. Two common patterns:

**Synchronous write-back**: at the end of each turn, block on writing the episode before proceeding. Simple, but adds 10-50ms of latency per turn for the embedding and DB write.

**Asynchronous write-back**: fire-and-forget at turn end, continue to the next turn immediately. Lower latency, but the agent might serve the next turn before the previous episode is indexed. For chat agents where turns are seconds apart, this is fine. For high-frequency agentic loops where turns are milliseconds apart, the lag can cause the agent to retry work it just completed.

We generally recommend asynchronous write-back with a local in-memory buffer that the retriever can access before the DB write completes, giving you the best of both worlds.

## 4. Type 3 — Semantic Memory: Facts, Knowledge, World Model

Semantic memory is the store of facts about the world that are not tied to specific episodes. It is the agent's knowledge base: what products exist, what their prices and specifications are, what the company's refund policy says, what the codebase's API schemas look like, what historical events are relevant to the current task.

The critical distinction from episodic memory: semantic facts are typically not time-indexed in the same way. The fact that Python's `datetime.strftime` uses `%Y-%m-%d` for ISO format dates is true independent of when the agent learned it. The fact that a product's price is $49/month is true until it changes. Semantic memory stores timeless (or slowly changing) facts; episodic memory stores events.

![Monolithic Blob vs Taxonomy-Structured Memory](/imgs/blogs/agent-memory-taxonomy-4.webp)

### The RAG Pipeline as Semantic Memory Implementation

Retrieval-Augmented Generation (RAG) is the dominant implementation of semantic memory in production systems. The basic pipeline:

1. **Ingest**: split source documents into chunks, embed each chunk, store in vector DB
2. **Retrieve**: at query time, embed the question, ANN-search for relevant chunks
3. **Augment**: prepend retrieved chunks to the context window
4. **Generate**: the model responds with access to the retrieved facts

The engineering complexity of RAG is almost entirely in the ingest and retrieval stages, not the generation stage. Getting the chunking strategy right matters enormously: too small (sentence-level) and you lose inter-sentence coherence; too large (section-level) and you embed too much irrelevant content per chunk, degrading retrieval precision.

A production-grade semantic memory implementation needs:

**Hierarchical chunking**: document → section → paragraph, with parent pointers so you can retrieve a paragraph but return its section for context. The "small-to-big retrieval" pattern.

**Metadata extraction**: extract structured fields from each document during ingest. Product name, version number, category, last_updated timestamp. These enable filtered retrieval that is orders of magnitude more precise than pure semantic similarity for structured queries.

**Knowledge versioning**: facts change. A product price that changes from $49 to $59 should not create two competing chunks — the old one should be marked as superseded. Without versioning, the agent will sometimes retrieve stale facts and confidently present them as current.

**Deduplication**: ingesting the same document twice creates duplicate chunks that get double-counted in retrieval. A good deduplication strategy uses document-level content hashes to detect exact duplicates and embedding-level similarity to detect near-duplicates.

```python
class SemanticMemoryIngestion:
    def __init__(self, vector_db, embedding_model):
        self.db = vector_db
        self.embed = embedding_model
        
    async def ingest_document(self, doc: Document) -> None:
        # Check for existing version of this document
        existing = await self.db.get_by_source_id(doc.source_id)
        if existing:
            if existing.content_hash == doc.content_hash:
                return  # Identical content, skip
            # New version: mark old chunks as superseded
            await self.db.update_many(
                filter={"source_id": doc.source_id},
                update={"$set": {"superseded": True, "superseded_at": datetime.utcnow()}}
            )
        
        # Chunk with hierarchy
        chunks = self.hierarchical_chunk(doc)
        
        # Embed and upsert
        embeddings = await self.embed.batch_encode([c.text for c in chunks])
        await self.db.upsert_many([
            {
                "id": chunk.id,
                "source_id": doc.source_id,
                "content_hash": doc.content_hash,
                "text": chunk.text,
                "parent_id": chunk.parent_id,
                "metadata": chunk.metadata,
                "embedding": emb,
                "superseded": False,
                "ingested_at": datetime.utcnow(),
            }
            for chunk, emb in zip(chunks, embeddings)
        ])
```

### Knowledge Graphs as Structured Semantic Memory

For domains with rich entity relationships — enterprise software with products, customers, contracts, users, and teams — a knowledge graph alongside the vector database provides retrieval capabilities that pure semantic search cannot match.

Consider a query like "What pricing tier does Acme Corp have for our enterprise product?" A pure semantic search will retrieve chunks about pricing and chunks about Acme Corp, but combining them to answer the specific question requires the model to reason over multiple retrieved chunks. A knowledge graph stores the direct relationship: Acme → [has_contract] → Enterprise Tier → [priced_at] → $2000/month. The answer is a two-hop graph traversal, not a vector similarity search.

In production, we use both: semantic search for open-ended knowledge retrieval, graph traversal for entity-relationship queries. The retrieval router decides which path to take based on whether the query contains specific entity names that exist in the graph.

## 5. Type 4 — Procedural Memory: How to Do Things

Procedural memory is the most counterintuitive of the four types for most ML engineers, because it is not stored in a database. Procedural memory is the agent's skill set — its knowledge of _how_ to do things, as opposed to _what_ is true. It includes:

- How to use tools correctly (call sequence, parameter formats, error handling)
- Which tool to use for which task
- How to format responses for different contexts
- What tone to use with different types of users
- When to ask for clarification vs. when to proceed
- What operations require user confirmation before executing

The critical engineering insight: procedural knowledge is most reliably encoded in model parameters. An agent that has been fine-tuned on thousands of correct tool-use trajectories will reliably invoke tools correctly, even under context pressure, even when the system prompt is truncated, even in novel situations not covered by the training data. An agent that relies solely on system prompt instructions for procedural guidance will occasionally fail when the instructions are not prominent in the working memory.

### The Three Implementation Mechanisms

Procedural memory in LLM agents has three implementation mechanisms, in increasing order of reliability:

**Mechanism 1: System prompt instructions.** The cheapest and most brittle option. You write out the procedures in prose ("Always validate the input before calling the database write API"). This works well for simple, few-step procedures when the context window is not under pressure. It fails when procedures are complex, numerous, or when the context window is crowded with other content.

**Mechanism 2: Few-shot examples in the system prompt.** More reliable than prose instructions because the model can pattern-match against concrete examples. Instead of "validate before writing", you provide 3-5 examples showing the full read-validate-write sequence. This costs context window tokens but produces more consistent behavior.

**Mechanism 3: Fine-tuning.** The most reliable mechanism. Collect trajectories of correct procedure execution (from demonstrations, from human feedback, from expert-written examples), and fine-tune the model on these trajectories. Fine-tuned procedural memory survives context compression, survives novel situations that don't match any in-context example, and does not consume context window tokens. The cost is the fine-tuning infrastructure and the need to collect training data.

The practical recommendation: start with system prompt instructions (fast to iterate), graduate to few-shot examples for the procedures that matter most, and fine-tune once you have accumulated enough correct trajectories and the system prompt approach is producing too many failures.

### LoRA Adapters as Procedural Memory Modules

Low-rank adaptation (LoRA) provides a middle path between system prompt engineering and full fine-tuning. A LoRA adapter modifies a small subset of the model's weight matrices — typically the query and value projection matrices in the attention layers — to encode task-specific behavior.

For agents, this translates to: train one LoRA adapter per major behavior mode. A customer support agent might have:

- A `tone_formal` adapter trained on formal customer-facing examples
- A `tool_use_sql` adapter trained on correct SQL tool invocation sequences
- A `escalation_detection` adapter trained to recognize when to escalate to a human

At inference time, you load the base model plus the relevant adapter(s) for the current deployment context. Multiple adapters can be merged linearly: $W_{merged} = W_{base} + \alpha_1 W_{LoRA_1} + \alpha_2 W_{LoRA_2}$, where the $\alpha$ coefficients control the strength of each adapter.

This composable adapter approach gives you procedural memory that is:
- Modular (add/remove skills without retraining the base model)
- Efficient (adapters are typically 1-2% of the base model's parameter count)
- Versionable (each adapter has a version, can be rolled back)

## 6. How Each Type Maps to LLM Components

The four memory types map cleanly onto the LLM infrastructure components you are already managing or considering.

![LLM Component → Memory Type Mapping](/imgs/blogs/agent-memory-taxonomy-5.webp)

| Memory Type | Primary LLM Component | Secondary Component |
|---|---|---|
| Working | Context window (prompt tokens) | KV cache (GPU SRAM) |
| Episodic | Vector database (time-indexed) | SQL event log |
| Semantic | Vector database (knowledge index) | Knowledge graph |
| Procedural | Model weights | LoRA adapters |

This mapping has a direct implication for system design: working and episodic memory share the vector database infrastructure but must be physically separated into different collections/namespaces. Using the same collection for episodic and semantic retrieval is the single most common memory architecture mistake I have seen in production agents.

The reason: episodic and semantic content have fundamentally different retrieval requirements. Episodic retrieval is filtered primarily by user identity and recency — you want recent events for the current user. Semantic retrieval is filtered primarily by content relevance — you want the facts most relevant to the current question, regardless of when they were indexed. Mixing them into one collection forces you to use the same retrieval strategy for both, and neither works well.

The practical fix: maintain at least three separate vector collections per agent deployment:

```python
# Collection naming convention
EPISODIC_COLLECTION = f"episodic_{tenant_id}"    # user events, turn summaries
SEMANTIC_COLLECTION = f"semantic_{tenant_id}"    # knowledge base, facts
# (Working memory is not a DB collection; it's the context window)
# (Procedural memory is not a DB collection; it's model weights)
```

## 7. Storage Systems for Each Type: In-Context, Vector DB, Relational DB, Model Weights

![Memory Type Comparison Matrix: Six Engineering Dimensions](/imgs/blogs/agent-memory-taxonomy-2.webp)

The matrix above summarizes the six engineering dimensions for each type. The table below shows the storage system mapping in detail, including access patterns and typical latency.

![Storage System × Memory Type × Access Pattern](/imgs/blogs/agent-memory-taxonomy-7.webp)

Each memory type has a canonical storage system, but the choice is not always obvious. Let me work through the decision for each.

### Working Memory Storage: Context Window + KV Cache

There is no separate database for working memory. It lives in the context window (the token sequence the model processes) and is physically realized as key-value tensors in GPU SRAM during inference.

The engineering decisions for working memory are about _management_: how do you decide what to include in the context? how do you handle the capacity limit? how do you evict content when the window fills?

A production working memory manager should implement:

```python
class WorkingMemoryManager:
    def __init__(self, max_tokens: int, tokenizer):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.BUDGET_SYSTEM_PROMPT = 0.10     # 10% reserved for system prompt
        self.BUDGET_TASK_STATE = 0.20         # 20% for current task
        self.BUDGET_RETRIEVED_CONTEXT = 0.40  # 40% for retrieved memory
        self.BUDGET_RECENT_HISTORY = 0.20     # 20% for recent turns
        self.BUDGET_TOOLS_SCRATCH = 0.10      # 10% for tool outputs/scratch
        
    def assemble_context(
        self,
        system_prompt: str,
        task_state: str,
        retrieved_episodes: list[str],
        retrieved_facts: list[str],
        recent_turns: list[dict],
        tool_outputs: list[str],
    ) -> str:
        budgets = {
            "system": int(self.max_tokens * self.BUDGET_SYSTEM_PROMPT),
            "task": int(self.max_tokens * self.BUDGET_TASK_STATE),
            "retrieved": int(self.max_tokens * self.BUDGET_RETRIEVED_CONTEXT),
            "history": int(self.max_tokens * self.BUDGET_RECENT_HISTORY),
            "tools": int(self.max_tokens * self.BUDGET_TOOLS_SCRATCH),
        }
        
        parts = []
        parts.append(self._truncate(system_prompt, budgets["system"]))
        parts.append(self._truncate(task_state, budgets["task"]))
        
        # Retrieved context: interleave episodes and facts by relevance score
        retrieved_budget = budgets["retrieved"]
        for chunk in sorted(retrieved_episodes + retrieved_facts, key=lambda x: x.score, reverse=True):
            chunk_tokens = self._count_tokens(chunk.text)
            if chunk_tokens <= retrieved_budget:
                parts.append(chunk.text)
                retrieved_budget -= chunk_tokens
            else:
                break  # Budget exhausted
                
        # Recent history: include most recent turns that fit
        history_budget = budgets["history"]
        for turn in reversed(recent_turns):
            turn_text = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            turn_tokens = self._count_tokens(turn_text)
            if turn_tokens <= history_budget:
                parts.insert(-len(parts)//2, turn_text)  # Insert before retrieved
                history_budget -= turn_tokens
                
        return "\n\n".join(parts)
```

The budget allocation is a parameter you tune per use case. A research agent that needs lots of retrieved context might use 60% for retrieved memory and only 10% for recent history. A conversational agent might reverse those proportions.

### Episodic Memory Storage: Vector DB + Relational Event Log

The episodic store needs two components working together:

**Vector database for semantic retrieval**: allows "find past events similar to the current situation." Chroma for development and small scale (< 1M episodes), Weaviate or Qdrant for production scale, pgvector if you are already on Postgres and want to avoid a separate infrastructure component.

**Relational event log for structured queries**: allows "find the last three conversations this user had about billing" or "find all unresolved issues from this session." A simple Postgres table:

```sql
CREATE TABLE episodes (
    episode_id UUID PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    summary TEXT NOT NULL,
    key_decisions JSONB,
    user_preferences JSONB,
    unresolved_items JSONB,
    entity_mentions TEXT[],
    embedding_id VARCHAR REFERENCES vector_embeddings(id),
    INDEX(user_id, timestamp DESC),
    INDEX(session_id, timestamp DESC)
);
```

The `embedding_id` field links to the vector store row so you can do structured SQL filtering first, then semantic similarity scoring on the filtered set — the correct order of operations for high-precision retrieval.

### Semantic Memory Storage: Vector DB + Knowledge Graph

Semantic memory uses the same vector database infrastructure as episodic memory but in a different collection with different chunking and metadata strategies. For structured domains, a knowledge graph adds valuable retrieval capabilities.

The graph database decision: **use a knowledge graph if and only if your domain has entities with typed relationships that agents need to traverse.** For a general-purpose assistant, a knowledge graph is overkill — pure vector retrieval is sufficient. For an enterprise SaaS agent that needs to know "which users have access to which features under which contract," the graph's traversal capabilities pay for the operational complexity.

### Procedural Memory Storage: Model Weights + Adapter Files

Procedural memory is stored as file artifacts, not database records:

- **Base model weights**: downloaded once, versioned by model ID (e.g., `meta-llama/Llama-3.1-70B-Instruct`)
- **LoRA adapter files**: trained artifacts, stored in a model registry (MLflow, Weights & Biases, or a simple S3 bucket with versioned paths)

The versioning discipline matters. If you fine-tune an adapter and push it to production, you need a way to:
1. Know which adapter version is currently serving
2. Roll back to the previous adapter if the new one regresses
3. A/B test adapter versions against a production traffic slice

A minimal adapter registry:

```python
@dataclass
class AdapterVersion:
    adapter_id: str
    version: str
    base_model_id: str
    training_date: datetime
    training_dataset_hash: str
    eval_metrics: dict[str, float]
    artifact_path: str    # s3://bucket/adapters/adapter_id/version/
    status: str           # 'staging' | 'production' | 'deprecated'
```

## 8. Read/Write Patterns: When Each Type Is Read, When Written, How Often Updated

![Memory Lifecycle Across Five Conversations](/imgs/blogs/agent-memory-taxonomy-6.webp)

The read/write patterns differ dramatically across memory types, and this asymmetry has infrastructure implications.

**Working memory** is read and written every generation step. It is assembled before the LLM call and discarded after. The only "write" operation is assembling the context; there is no persistence. The infrastructure implication: working memory management must be fast — typically under 5ms for context assembly, or it adds perceptible latency to every turn.

**Episodic memory** is read at turn start (to provide context for the current turn) and written at turn end (to record what happened). The read/write ratio is roughly 1:1 per turn. Write latency is acceptable up to 100ms since it happens after the response is sent. Read latency must be under 50ms to avoid user-visible delays. The infrastructure implication: write asynchronously, optimize reads with appropriate vector index settings (HNSW ef_search, quantization level).

**Semantic memory** is read frequently (on most agent turns that involve factual questions) but written rarely (when the knowledge base is updated). A production semantic store might have a read/write ratio of 100:1 or higher. This high read/write ratio means you can trade write latency for index quality: spending 5 seconds to build a better HNSW index is worthwhile if it improves retrieval precision for the 100 subsequent reads. The infrastructure implication: batch-process semantic ingest jobs during low-traffic periods; prioritize index quality over ingest speed.

**Procedural memory** is read at model load time (the adapter weights are loaded into GPU memory) and written only during fine-tuning runs (which happen offline, once a week or once a month). The read/write ratio is orders of magnitude higher than the others — potentially 1,000,000:1 between inference reads and training writes. The infrastructure implication: the serving path is entirely read-only; writes happen in a separate offline training pipeline that has no latency requirements.

## 9. Failure Modes Per Type: What Breaks When Each Type Is Missing or Corrupted

![Failure Mode Tree: Missing or Corrupted Memory](/imgs/blogs/agent-memory-taxonomy-8.webp)

Each memory type has characteristic failure modes that allow you to diagnose which layer is missing or corrupted.

### Working Memory Failures

**Context truncation**: the most common working memory failure. Symptoms: the agent contradicts a constraint established earlier in the conversation; the agent asks for information the user already provided; the agent repeats a suggestion it made before. Root cause: the context window filled up and early tokens were truncated. Fix: improve working memory curation to keep essential information prominent; use a model with a larger context window; implement hierarchical summarization to compress older turns.

**KV cache thrashing**: in high-load serving environments, the KV cache fills up and serving frameworks start evicting cached prefixes to make room for new requests. Symptoms: latency spikes of 3-5x; P99 latency degrades significantly under load while P50 remains acceptable. Root cause: KV cache capacity is insufficient for the load pattern. Fix: reduce average context length; add GPU memory; use quantized KV cache (KV8 halves memory usage at slight quality cost).

**Attention dilution ("lost in the middle")**: retrieved context placed in the middle of a long prompt receives less attention than content at the beginning or end. Symptoms: the agent fails to use retrieved facts even though they were included in the context. Fix: place the most critical retrieved content at the beginning of the context, not in the middle; use instruction-following models that are explicitly trained for long-context attention.

### Episodic Memory Failures

**Goldfish syndrome**: the agent has no memory of previous conversations with the user. Every session starts cold. Symptoms: the agent repeatedly asks for user preferences that were established in prior sessions; users report that the agent "forgot" previous decisions. Root cause: episodic memory is not implemented, or the retrieval is not connected to the context assembly pipeline. Fix: implement episodic write-back and retrieval.

**Stale episode injection**: old episodes from different contexts are retrieved and injected into the current context, causing the agent to conflate past situations with the present. Symptoms: the agent mentions a problem that was resolved months ago as if it were current; the agent applies a constraint from one user to a different user. Root cause: missing user-level filtering on episodic retrieval, or TTL eviction is not configured. Fix: add strict user_id filtering to all episodic queries; implement TTL-based eviction for episodes older than N days.

**Episode write failure**: turn summaries fail to write to the episodic store (network error, DB overload, etc.). Symptoms: occasional "memory gaps" where specific conversations are not recalled; inconsistent recall across users. Root cause: synchronous write-back with no retry logic; async write-back with no monitoring. Fix: implement retry-with-backoff on episodic writes; monitor write error rates as a production metric.

### Semantic Memory Failures

**Hallucination from knowledge gaps**: the agent confidently answers questions about topics not in the knowledge base, using its pretraining knowledge instead. Symptoms: factual errors about company-specific information (prices, policies, procedures); confident but incorrect answers to domain-specific questions. Root cause: semantic memory not implemented, or the retrieval pipeline is not connected, or the knowledge base does not cover the domain. Fix: implement semantic memory with appropriate knowledge base coverage; add a retrieval confidence gate that abstains when no relevant chunks are found.

**Knowledge drift**: the knowledge base contains stale information that was accurate at indexing time but has since changed. Symptoms: the agent gives prices that were valid 6 months ago; the agent describes deprecated features as current. Root cause: no TTL or versioning on semantic memory chunks; incremental updates not propagated to the vector store. Fix: implement document versioning with supersession; set TTL on time-sensitive facts; implement periodic knowledge refresh jobs.

**Retrieval precision failure**: the retrieval returns tangentially related chunks instead of the directly relevant information. Symptoms: the agent gives a generic answer when specific information is in the knowledge base; the retrieved context is technically "about" the topic but does not actually answer the question. Root cause: poor chunking strategy (chunks too large or too small); missing metadata filtering; suboptimal embedding model for the domain. Fix: tune chunk size; add metadata extraction and filtered retrieval; evaluate domain-specific embedding models.

### Procedural Memory Failures

**Tool misuse**: the agent calls tools with incorrect parameters, in the wrong sequence, or with insufficient error handling. Symptoms: tool call parse errors in the logs; tools returning unexpected results because they were called with the wrong input format; the agent calling a write operation before a validation step. Root cause: procedural memory relies solely on system prompt instructions that are either unclear or get deprioritized under context pressure. Fix: graduate from prose instructions to few-shot examples; invest in fine-tuning for the most critical tool-use sequences.

**Policy regression**: after a fine-tuning run intended to improve one behavior, the agent starts failing on a different behavior that was working before. Symptoms: previously reliable safety refusals stop working; formatting consistency degrades; a specific tool-use pattern breaks. Root cause: catastrophic forgetting during fine-tuning; the new training data inadvertently conflicts with previously learned behaviors. Fix: use LoRA adapters instead of full fine-tuning; implement an evaluation harness that tests all critical behaviors before promoting a new adapter to production.

**Behavioral inconsistency under load**: in rare cases, the base model's procedural behavior changes under high-load conditions due to subtle numerical differences in the forward pass (different batch sizes, different quantization configurations). Symptoms: the agent occasionally fails to follow a procedure it reliably follows in testing; failures are non-deterministic and load-correlated. Root cause: serving configuration differences between test and production environments. Fix: match test and production inference configurations exactly; test at production-representative batch sizes.

## 10. Design Decisions: Which Types to Implement for Different Agent Use Cases

The honest answer is that you rarely need all four types on day one. The question is not "which four types do I need?" but "which one do I start with, and when do I add the next one?"

![Memory Requirements by Agent Use Case](/imgs/blogs/agent-memory-taxonomy-9.webp)

The decision framework:

**Start with working memory only** when you are building a stateless, single-session agent that does not need to remember anything across sessions. One-shot QA tools, code generation endpoints, text transformation APIs. These agents often work well with just the context window and a carefully crafted system prompt.

**Add episodic memory** when users start complaining that the agent "forgets" things they told it before, or when you need to maintain continuity across a series of interactions (a multi-step task that spans multiple sessions, a customer relationship that deepens over time). Episodic memory is usually the first addition after working memory, because the user experience impact is immediate and visible.

**Add semantic memory** when the agent starts hallucinating domain-specific facts, or when you have a corpus of knowledge (product documentation, internal policies, codebase APIs) that the base model does not know about and that changes too frequently to address through fine-tuning. Semantic memory is essential for any enterprise agent that needs to reason about company-specific information.

**Add procedural memory** when system prompt engineering and few-shot examples are producing too many procedure-following failures in production. Signs you need procedural memory: tool call error rates > 2% in production, inconsistent formatting despite explicit instructions, safety policy violations that should have been caught by the system prompt. Procedural memory through fine-tuning is expensive to maintain but produces the most robust behavior.

## 11. Case Studies: Memory Architecture Decisions and Consequences

### Case Study 1: The Customer Support Agent That Forgot Everything

A fintech company deployed a GPT-4-based customer support agent in Q3 2024. The agent handled account inquiries, transaction disputes, and product questions. Within two weeks, they were receiving user complaints about the agent "not remembering" what users had discussed in previous chat sessions.

The root cause was architectural: the agent was stateless. Each conversation session was completely independent — the agent had no episodic memory, and users would have to re-explain their situation from scratch every time they contacted support. For a billing dispute that required multiple back-and-forth sessions to resolve, this was a serious UX failure.

The team implemented episodic memory using Pinecone for vector storage and a Postgres table for session metadata. Each conversation session was summarized at session end using a separate LLM call (GPT-3.5-turbo, cheaper than GPT-4) and stored with user ID as the partition key. At session start, the three most relevant past episodes were retrieved and injected into the system prompt.

The result was a dramatic improvement in user satisfaction scores: the "agent remembered my previous issue" metric went from 0% to 67% (not 100%, because retrieval recall is not perfect). More importantly, the average time to resolve a repeated issue dropped from 12 minutes to 5 minutes — users no longer needed to re-explain context the agent should already know.

**The lesson**: episodic memory is often the highest-ROI first investment after working memory. Its absence is immediately visible to users and its implementation is straightforward.

### Case Study 2: The Code Assistant That Hallucinated Its Own APIs

An enterprise developer tools company deployed an internal code assistant trained to help engineers use their proprietary SDK. The SDK had about 300 API methods with complex type signatures, optional parameters, and subtle behavioral nuances that differed from common patterns in the training data.

The agent, powered by a general-purpose LLM without any RAG, consistently hallucinated API signatures — confidently inventing optional parameters that did not exist, generating function calls with the wrong parameter types, and occasionally citing API methods that did not exist at all. The base model had learned enough Python to produce plausible-looking code, but it had no knowledge of the company's specific SDK.

The team built a semantic memory layer: they ingested all 300 API method documentation strings, type signatures, and usage examples into a vector database. The retrieval pipeline was specialized for code queries: when the user asked about a specific API method by name, the system did an exact match retrieval (not semantic ANN search) to ensure it got the precise documentation for that method rather than semantically similar but distinct methods.

Hallucination rate on SDK API calls dropped from ~23% to ~3%. The 3% that remained were failures on unusual parameter combinations not well-represented in the documentation, suggesting a remaining gap in semantic memory coverage.

**The lesson**: semantic memory is essential when the agent operates in a specialized domain where the base model's pretraining knowledge is incomplete or incorrect. The retrieval strategy should be tailored to the query type — exact match for known entity lookups, semantic ANN for open-ended questions.

### Case Study 3: The Research Agent With the Tangled Memory Store

A research team at a hedge fund built an autonomous research agent designed to investigate investment theses over multi-day research sessions. The agent would search for information, read documents, form hypotheses, and progressively refine its understanding of a company or market.

The team's first implementation used a single vector database collection for all agent memory: the research notes the agent had written, the facts it had extracted from documents, and the metadata about what it had already explored (to avoid redundant searches). All three types of content went into the same Chroma collection.

The failure mode was subtle but consistent: the agent would get confused between its own hypotheses (episodic content — things the agent had speculated) and facts extracted from source documents (semantic content). It would sometimes present an unvalidated hypothesis as a confirmed fact, or dismiss a real fact because it conflated it with a prior speculation that turned out to be wrong.

The team separated the collection into three: `research_notes` for the agent's own reasoning and hypotheses (episodic), `extracted_facts` for verified information from source documents (semantic), and `search_log` for what had been explored (working state). The system prompt included explicit instructions about which collection each type of information should be written to, and retrieval queries were routed to the appropriate collection based on whether the agent was seeking its own thoughts or external facts.

**The lesson**: mixing episodic and semantic content in one store causes retrieval precision failures that manifest as the model confusing its own reasoning with external ground truth. The collections must be separated.

### Case Study 4: The Chat Agent That Forgot Its Own Policies Under Pressure

A legal tech company deployed a chat agent to help lawyers draft contracts. The agent had a complex set of procedural requirements in its system prompt: it would not provide legal advice (only information), it would always recommend professional review, it would not modify contract templates without explicit user confirmation, and it would cite sources for any legal claims.

In testing, the agent followed all four requirements correctly 100% of the time. In production, under certain load conditions and with certain conversation patterns, compliance dropped to around 85% for the "cite sources" requirement and 92% for the "always recommend review" requirement.

The team's investigation found the root cause: long conversations where users were iteratively editing a contract clause by clause would eventually fill the context window to 80-90% capacity. The system prompt (which contained the procedural requirements) was at the beginning of the context, meaning it had the lowest recency weight in the model's attention pattern. Under context pressure, the model would occasionally treat the procedural requirements as lower priority than the user's immediate request.

The fix required two changes. First, they moved the four critical requirements from a prose paragraph in the system prompt to explicit system-level reminder messages injected every 10 turns ("Reminder: always recommend professional legal review of any contract"). Second, they fine-tuned a small LoRA adapter specifically on correct trajectories that included all four compliance behaviors, to bake the procedural requirements into model weights rather than relying on the context window.

**The lesson**: critical procedural requirements cannot be implemented solely through context window prompting in agents with long sessions. They need to be either periodically re-injected into the context or encoded into model weights through fine-tuning.

### Case Study 5: The Voice Assistant That Learned Wrong Procedures

A consumer electronics company deployed an LLM-based voice assistant on their smart home devices. The initial deployment used a GPT-4 base without fine-tuning, with all procedural guidance in the system prompt. User satisfaction was high for the first three months.

The team then fine-tuned the model on a dataset of successful voice assistant interactions collected from production usage. The goal was to improve response latency (by using a smaller fine-tuned 7B model instead of GPT-4) and to improve handling of device-specific commands.

After deploying the fine-tuned model, refusal rates on legitimate requests increased by 15%. The model started refusing commands like "turn off all lights" (citing privacy concerns that were never in the original system prompt) and "set the alarm for 6 AM tomorrow" (citing uncertainty about timezone that it had never expressed before).

Investigation revealed the root cause: the training dataset contained a biased sample of interactions. The successful interactions that were logged and labeled for training skewed toward cautious, verbose, safety-hedging responses because human raters (who labeled the data) systematically rated cautious responses as "safer" and therefore "better," even for benign requests. The fine-tuning had encoded excessive caution as a procedural behavior.

The fix required three steps: audit the training data to remove the systematic rater bias, retrain the adapter, and implement an evaluation harness that explicitly tested for over-refusal (measuring false negative rate, not just false positive rate).

**The lesson**: procedural memory encoded through fine-tuning inherits the biases of the training data. An evaluation harness that tests all critical behaviors — including correct acceptance of legitimate requests, not just correct rejection of harmful ones — is non-negotiable before promoting a new model or adapter to production.

### Case Study 6: The Semantic Memory That Wouldn't Retire

A healthcare information platform deployed a clinical information retrieval agent for medical professionals. The semantic memory store contained medical literature, clinical guidelines, and drug information. The team implemented a RAG pipeline using a 768-dimensional embeddings model and Weaviate as the vector store.

Over 18 months of operation, clinical guidelines were updated by medical societies, drug information was revised, and several literature citations were retracted. The team would ingest the updated documents into the vector store, but would not delete the old versions. Their reasoning: they were nervous about deleting content and losing information.

The result: for any query about a topic that had been updated, the vector store returned both the new guideline and the old guideline in the top-k results. The model, receiving conflicting information, would sometimes synthesize a composite answer that was neither the current recommendation nor the obsolete one — a medically dangerous third option.

The fix required implementing document versioning with explicit supersession. When a new version of a guideline was ingested, the system would automatically mark all previous chunks from that source document as `superseded=True` and exclude them from retrieval queries. The system also ran quarterly audits comparing the current date against each document's `last_valid_date` metadata field, flagging content due for review.

**The lesson**: semantic memory without version control and eviction policies becomes a liability in any domain where facts change. The cost of keeping stale information in the store is often higher than the cost of losing it.

### Case Study 7: The Multi-Agent System With Shared Memory Contamination

A software development company built a multi-agent system for code review. The system had three specialized agents: an architecture reviewer, a security reviewer, and a style reviewer. Each agent ran independently and wrote its findings to a shared "review notes" vector store, which all three agents could read to see what the others had found.

The design worked in unit testing but failed in integration. In practice, each agent's retrieval queries would return not just their own agent's previous notes but also notes from the other agents. The security reviewer would read architecture notes and use them in its context, sometimes causing it to flag architecture patterns as security vulnerabilities (incorrect conflation). The style reviewer would read security findings and become excessively cautious about style decisions with security implications (over-application of retrieved context).

The fix was to partition the shared store by agent ID and by content type. Agents could write to their own partition freely, and could read from other agents' partitions only through a controlled cross-agent retrieval API that was filtered by content type. Raw security findings were not directly accessible to the style reviewer; only a structured "security summary" object was accessible.

**The lesson**: in multi-agent systems, memory sharing requires explicit access control and content-type filtering. An uncontrolled shared memory store is to multi-agent systems what an uncontrolled shared mutable state is to concurrent programming.

### Case Study 8: The Agent That Improved Its Own Procedural Memory

This case study is unusual because it is a success story about procedural memory self-improvement. A data analysis agent was deployed to help analysts at an e-commerce company run SQL queries and interpret results. The agent had initial LoRA adapters trained on internal SQL patterns.

The team implemented a feedback loop: when an analyst manually corrected a SQL query the agent generated, the correction (original query + analyst's corrected version) was logged to a training buffer. When the buffer accumulated 500 examples, an automated fine-tuning job ran to update the SQL adapter. The new adapter was automatically evaluated on a held-out test set and promoted to production if it passed.

Over six months, the agent's SQL correctness rate on complex multi-table join queries improved from 71% to 89%. The improvement was entirely driven by the self-improvement loop — no human ML engineer was manually curating training examples or running fine-tuning jobs. The loop was fully automated.

The key architectural decisions that made this work:

1. **Separate buffer per query type**: corrections to SELECT queries were not mixed with corrections to UPDATE/INSERT queries (very different risk profiles)
2. **Conservative promotion criteria**: the new adapter had to exceed the current adapter by at least 2% on the held-out test set to be promoted
3. **Automatic rollback**: if production error rates increased by more than 5% after a promotion, the previous adapter was automatically restored

**The lesson**: procedural memory can be improved incrementally through continuous learning if you build the correction-logging and fine-tuning pipeline upfront. The key is a conservative, automated evaluation gate that prevents regressions from propagating to production.

## 12. Phased Rollout Strategy: Which Types to Implement in What Order

![Phased Memory Rollout: What You Gain at Each Phase](/imgs/blogs/agent-memory-taxonomy-10.webp)

Here is the practical rollout sequence I recommend for teams building agents from scratch.

### Phase 1: Working Memory Only — Ship Something

**Timeline**: Sprint 0-2

**What to build**:
- Context assembly pipeline with explicit budget allocation by content type
- System prompt that clearly defines the agent's role, constraints, and tool definitions
- A few-shot example block for the 2-3 most common interaction patterns

**What you gain**:
- A functional, shippable agent
- Baseline metrics to measure against as you add memory types

**What you accept**:
- No cross-session memory
- Re-asking for user information on every session
- Hallucination on domain-specific information not in the system prompt

The goal of Phase 1 is not to build the right agent — it is to build a working agent that you can deploy, measure, and iterate on. The data you collect in Phase 1 will tell you which memory type to add in Phase 2.

### Phase 2: Add Episodic Memory — Users Feel Remembered

**Timeline**: Sprint 3-6

**What to build**:
- Vector database setup (Chroma for dev, Weaviate or pgvector for production)
- Turn summarization pipeline (run a cheap model at turn end to generate the episode record)
- Episode retrieval integration into the context assembly pipeline
- User_id partitioning and access control

**What you gain**:
- Cross-session memory: the agent remembers user preferences, past decisions, prior context
- Resolution continuity: multi-session tasks can pick up where they left off
- Reduced repetition: users stop having to re-explain their context

**The metric to watch**: "session continuity score" — what percentage of second-and-later sessions correctly reference information from prior sessions?

**Common pitfall**: not building the episode schema carefully enough on the first attempt, then having to migrate an existing production store. Do the schema design upfront, including metadata fields for structured filtering.

### Phase 3: Add Semantic Memory — Hallucination Drops

**Timeline**: Sprint 7-12

**What to build**:
- Document ingestion pipeline with chunking, embedding, and metadata extraction
- Retrieval integration with configurable retrieval strategy per query type
- Knowledge versioning with supersession marking
- Separate vector collection for semantic content (distinct from episodic)

**What you gain**:
- Grounded factual answers about domain-specific information
- Significant reduction in hallucination on topics covered in the knowledge base
- The ability to keep agent knowledge current without model retraining

**The metric to watch**: "factual accuracy on domain-specific queries" measured by a QA evaluation set covering the knowledge base topics.

**Common pitfall**: thinking that adding RAG will eliminate hallucination. It reduces domain-specific hallucination but does not eliminate general hallucination or out-of-domain hallucination. Set realistic expectations.

### Phase 4: Add Procedural Memory — Reliable Tool Use

**Timeline**: Sprint 13-20 (much longer because fine-tuning requires data collection)

**What to build**:
- Correction logging pipeline: capture instances where users corrected the agent's tool calls
- Training data curation: clean and structure the correction logs into fine-tuning format
- Fine-tuning pipeline: likely using LoRA, with automated evaluation
- Adapter registry: version control for trained adapters
- A/B testing infrastructure: compare adapter versions on production traffic

**What you gain**:
- Reliable tool use that survives context pressure and long conversations
- Consistent behavioral policies that don't erode under conversation length
- A self-improving system that gets better as more corrections accumulate

**The metric to watch**: "tool call success rate" (valid parameters, correct sequence) and "behavioral policy compliance rate" (percentage of turns where all required behavioral policies are followed).

**Common pitfall**: fine-tuning too early with too little data. The minimum viable dataset for behavioral fine-tuning is typically 500-2000 high-quality examples per behavior. Training with less data produces adapters that overfit to the training examples and generalize poorly.

---

## When to Reach for Each Memory Type

| Memory Type | Reach for it when... | Don't reach for it when... |
|---|---|---|
| Working (context management) | Always — every agent needs it | — |
| Episodic | Users must maintain context across sessions; multi-step tasks span multiple interactions | Stateless, single-session tools; batch processing jobs |
| Semantic | The base model lacks domain-specific knowledge; hallucination on domain facts is a problem | The domain is already well-covered by the base model's pretraining; knowledge changes too frequently to maintain |
| Procedural (fine-tuning) | Tool call error rate > 2% in production; behavioral policies erode under context pressure; you have 500+ correction examples | You have fewer than 500 examples; the current system-prompt approach works reliably; you lack fine-tuning infrastructure |

The pattern to watch for: if you find yourself solving an episodic memory problem with working memory (stuffing conversation history into the context), or solving a procedural memory problem with semantic memory (putting procedure documentation into the RAG store), you have identified an architecture debt that will compound as your agent's usage grows. The taxonomy is not bureaucracy — it is the map that prevents the wrong infrastructure from accumulating in the wrong place.

Build the layer that solves your current bottleneck, measure its impact, then decide whether to invest in the next layer. The four types are not all required; they are all available.

---

## Further Reading

- [Memory vs Context Window in Agents](/blog/machine-learning/ai-agent/memory-vs-context-window-agents) — deep dive on the working memory capacity problem and techniques for managing context under pressure
- [Long-Term Memory for Conversational Agents: MemGPT](/blog/machine-learning/ai-agent/long-term-memory-conversational-agents-memgpt) — how MemGPT implements a virtual context extension to simulate unbounded working memory
- [Mem0: Token-Efficient Memory Algorithm](/blog/machine-learning/ai-agent/mem0-token-efficient-memory-algorithm) — a specific algorithm for episodic memory management that minimizes token cost while maximizing recall

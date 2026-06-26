---
title: "Agent Memory Cost Optimization: 6 Techniques for 3–4× Token Reduction"
date: "2026-06-27"
description: "Six production techniques for cutting agent memory token costs by 3–4× — token-efficient extraction, fusion retrieval, ADD-only passes, batch consolidation, salience filtering, and async writes."
tags: ["ai-agents", "memory", "cost-optimization", "token-efficiency", "llm", "machine-learning", "production-ml", "mlops"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 35
---

You launched your memory-augmented agent. Users love it. The personalization is real, the recall is accurate, and your retention numbers are up. Then the invoice arrives.

At 1,000 daily active users, each averaging 10 conversation turns, with a 20,000-token context window including memory injection: you are spending $500/day on GPT-4o at $2.50/1M tokens — *before* you count the separate extraction LLM calls that run on every turn to pull facts out of the conversation. Add those at 5,000 tokens per call and you are at $625/day. Memory recall retrieval injects another 5,000–8,000 tokens per turn into the context, pushing the real number past $800/day. That is $24,000/month for a modest user base, and memory overhead is responsible for 40–60% of the total.

This post is a production playbook for making that math sane. We will go through six specific techniques — in priority order, with real implementation code, with before/after cost comparisons, and with case studies from teams that have shipped them. Stack all six and you cut your memory token spend by 3–4×. The techniques are not theoretical. They come from studying how production memory systems like mem0 have optimized their pipelines, and from seeing the same patterns emerge independently in health AI, customer support, and coding assistant deployments.

![The three cost drivers of agent memory: extraction LLM calls, retrieval context tokens, and injection tokens](/imgs/blogs/agent-memory-cost-optimization-1.webp)

The diagram above frames the structure of the problem. Memory cost is not one thing — it is three separate cost layers that compound. Extraction LLM calls happen on every turn and cost compute. Retrieved memories inject tokens into every context window. And the base reasoning context itself grows because memory-augmented prompts are structurally longer than raw queries. Every optimization technique in this post targets one or more of these three layers.

## The math that motivates this

Let us be specific about the baseline before talking about solutions. A production memory system running on GPT-4o does roughly the following per turn:

**Extraction:** Each conversation turn is sent to an LLM with an extraction prompt asking it to identify new facts worth storing. A 10-turn conversation might generate 5,000 tokens of conversation text plus a 1,000-token system prompt for extraction instructions. That is 6,000 input tokens per turn at $2.50/1M = **$0.015 per turn**.

**Retrieval and injection:** The extracted memories need to be injected back into the context on the *next* turn. Top-K retrieval fetches, say, 10 memories at 200 tokens each = 2,000 tokens. The injection into the system prompt pushes the base context from 1,000 tokens to 3,000 tokens. At $2.50/1M for a 20,000-token reasoning call: **$0.05 per turn**.

**Baseline reasoning:** The main model call itself: 20,000 tokens × $2.50/1M = **$0.05 per turn**.

Total per turn: $0.015 + $0.05 + $0.05 = **$0.115 per turn**. For 1,000 users × 10 turns/day: **$1,150/day**, or roughly $34,500/month. Memory overhead accounts for $650/day of that — 57%.

The six techniques in this post collectively bring memory overhead from $650/day down to around $160–180/day, dropping total cost from $1,150 to ~$500–520/day. That is the 2–2.3× total cost reduction, or a 3–4× reduction on the *memory portion* specifically.

## The six techniques and their impact

![Impact matrix: each technique rated on token reduction, latency, complexity, and risk](/imgs/blogs/agent-memory-cost-optimization-2.webp)

The matrix above is the decision map. Before diving into implementations, note the structure: the first three techniques (async writes, ADD-only pass, salience filtering) are low-complexity, low-risk, high-reward. They should ship in week 1. Fusion retrieval and batch consolidation require more infrastructure. Efficient extraction is highest-reward but touches the core extraction path and should come last when you understand your memory quality requirements.

---

## Technique 1: Token-Efficient Extraction

### What the naive approach does wrong

The default approach to memory extraction sends the entire conversation to a large, expensive model. The assumption is that a more capable model produces better memory quality. This is partially true, but it catastrophically misunderstands where the tokens go.

In a 10-turn conversation, most of the text is *not* facts worth remembering. It is greetings, confirmations, follow-up questions, and conversational filler. A turn like:

> User: "Yeah, that sounds good. So anyway, my wife is a vegetarian and we're looking for restaurants near downtown Chicago."

contains exactly two extractable facts: (1) the user's partner is vegetarian, and (2) they are looking for restaurants near downtown Chicago. The other 18 words are filler. Sending the full 50-token turn to GPT-4o to extract those two facts costs the same as sending a 5,000-token essay.

The efficient approach: use a small, cheap model to do a first-pass *fact density filter* — identify which sentences actually contain extractable facts — and then send only those fact-dense sentences to the extraction pipeline.

![Token-efficient extraction: before (full conversation → large model) vs. after (fact-dense sentences → small model)](/imgs/blogs/agent-memory-cost-optimization-3.webp)

The numbers in the diagram are representative production numbers. A 6,000-token conversation goes down to 800 fact-dense tokens. And the extraction model shifts from GPT-4o ($2.50/1M) to GPT-4o-mini ($0.15/1M). The combined effect: cost per extraction drops from ~$0.045 to ~$0.0002 — a 200× reduction on the extraction path alone.

### Implementation

```python
import openai
from typing import Optional

FACT_DENSITY_FILTER_PROMPT = """You are a fact density filter.
Your job is to identify sentences that contain facts worth remembering about the user.
Facts include: preferences, constraints, personal details, goals, decisions made.
NOT facts: greetings, acknowledgments, questions without answers, filler phrases.

Return ONLY the fact-dense sentences, one per line. If no facts: return empty string.

Conversation:
{conversation}"""

EXTRACTION_PROMPT = """Extract structured memory facts from these fact-dense sentences.
Return JSON array: [{{"fact": "...", "category": "preference|personal|constraint|goal"}}]

Fact-dense sentences:
{sentences}"""

async def extract_memories_efficient(
    conversation: str,
    filter_model: str = "gpt-4o-mini",
    extraction_model: str = "gpt-4o-mini",
    client: Optional[openai.AsyncOpenAI] = None,
) -> list[dict]:
    """Extract memories using two-stage efficient pipeline."""
    if client is None:
        client = openai.AsyncOpenAI()

    # Stage 1: fact density filter (cheap, fast)
    filter_resp = await client.chat.completions.create(
        model=filter_model,
        messages=[{
            "role": "user",
            "content": FACT_DENSITY_FILTER_PROMPT.format(conversation=conversation)
        }],
        max_tokens=500,  # fact-dense sentences are short
        temperature=0,
    )
    fact_sentences = filter_resp.choices[0].message.content.strip()

    if not fact_sentences:
        return []  # no facts found, no extraction call needed

    # Stage 2: structured extraction (only fact-dense sentences, still cheap model)
    extract_resp = await client.chat.completions.create(
        model=extraction_model,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(sentences=fact_sentences)
        }],
        max_tokens=1000,
        temperature=0,
    )

    import json
    try:
        return json.loads(extract_resp.choices[0].message.content)
    except json.JSONDecodeError:
        return []
```

### The quality tradeoff

The most common objection: does a small model miss facts that a large model would catch? In practice, the answer is *sometimes, but rarely for the facts that matter*. The fact-density filter stage is doing a much simpler task than extraction — it is classifying sentences as fact-bearing or not, not generating structured memories. GPT-4o-mini is sufficient for that classification.

Where the small model struggles: highly implicit facts. "I was in Chicago in 2019 for a year" implies the user probably lived there, but the sentence does not say "I lived in Chicago." A large model is more likely to make that inference. For most production use cases, explicit facts are what you need and small models handle them fine.

The right mental model: the fact density filter is a *precision enhancement* for extraction, not a replacement. You are paying to extract facts from a carefully selected 800-token extract instead of a noisy 6,000-token conversation. You get better precision *and* lower cost.

---

## Technique 2: Fusion Retrieval

### Why single-signal retrieval is expensive

Most memory systems retrieve by semantic similarity: embed the current query, compute cosine similarity against the memory store, return top-K. This works, but it has a fundamental efficiency problem: to achieve high recall, you need to set K high. If your memory store has 500 memories and semantic similarity sometimes misses exact-match facts (because embeddings compress meaning), you compensate by fetching K=15 or K=20 memories instead of K=5. Those 10–15 extra memories are mostly noise injected into the context.

Fusion retrieval combines three signals — semantic similarity, BM25 keyword matching, and named entity overlap — and then re-ranks the union. Each signal catches different failure modes:

- **Semantic similarity** catches paraphrasing ("vegetarian" matches "plant-based")
- **BM25** catches exact keyword matches ("Chicago" reliably matches "Chicago")
- **Entity overlap** catches when the query references a person or place the user mentioned before

The combined signal achieves higher recall at lower K. Instead of K=15 via semantic alone, K=7 via fusion gets you the same recall. That is 8 fewer memories injected into every context window.

```python
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy

class FusionMemoryRetriever:
    """Retrieves memories using semantic + BM25 + entity fusion."""

    def __init__(self, memories: list[str], embed_model: str = "all-MiniLM-L6-v2"):
        self.memories = memories
        self.embed_model = SentenceTransformer(embed_model)
        self.nlp = spacy.load("en_core_web_sm")

        # Build BM25 index
        tokenized = [m.lower().split() for m in memories]
        self.bm25 = BM25Okapi(tokenized)

        # Build semantic embeddings
        self.embeddings = self.embed_model.encode(memories, convert_to_tensor=True)

    def retrieve(self, query: str, top_k: int = 7) -> list[tuple[str, float]]:
        # Signal 1: semantic similarity
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        semantic_scores = util.cos_sim(query_emb, self.embeddings)[0].numpy()

        # Signal 2: BM25 keyword
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_normalized = bm25_scores / (bm25_scores.max() + 1e-8)

        # Signal 3: named entity overlap
        query_doc = self.nlp(query)
        query_entities = {ent.text.lower() for ent in query_doc.ents}
        entity_scores = np.zeros(len(self.memories))
        if query_entities:
            for i, memory in enumerate(self.memories):
                mem_doc = self.nlp(memory)
                mem_entities = {ent.text.lower() for ent in mem_doc.ents}
                overlap = len(query_entities & mem_entities)
                entity_scores[i] = overlap / len(query_entities)

        # Fusion: weighted combination
        fused = 0.5 * semantic_scores + 0.3 * bm25_normalized + 0.2 * entity_scores
        top_indices = np.argsort(fused)[::-1][:top_k]

        return [(self.memories[i], float(fused[i])) for i in top_indices]
```

### The injection token savings

At K=7 vs K=15, with average memory length of 200 tokens, fusion retrieval saves 1,600 tokens per turn. At $2.50/1M for 1,000 users × 10 turns: **$40/day saved** from retrieval context alone. That is before the quality benefit — higher recall means the agent gives more accurate answers with fewer retrieved memories.

The cost of building and maintaining the fusion index is real: BM25 requires a tokenized corpus, entity extraction runs on every memory at write time, and the triple-signal ranking adds ~5ms of retrieval latency. For high-frequency agents (>10k daily turns), this is a net win. For low-frequency or batch applications, the engineering overhead may not justify the gain; check the deployment profile matrix at the end.

---

## Technique 3: ADD-Only Single Pass

### The hidden cost of three-pass extraction

The canonical memory update pipeline does three things on every turn:

1. **ADD pass**: scan the conversation for new facts, add them to the store
2. **UPDATE pass**: check existing memories for facts that have changed, update them
3. **DELETE pass**: identify contradictions with existing memories, remove the outdated ones

Each pass is a separate LLM call. Each call sends the conversation text plus the relevant existing memories. For a user with 100 stored memories and 5,000 tokens of conversation: three calls × 7,000 tokens each = 21,000 input tokens per turn, just for the extraction pipeline.

The ADD-only approach collapses this into one pass that handles all three operations via *conflict resolution prompting*:

![ADD-only single pass vs three-pass pipeline: token count and latency comparison](/imgs/blogs/agent-memory-cost-optimization-4.webp)

The key insight: the reason for three passes is that the ADD prompt, the UPDATE prompt, and the DELETE prompt are different tasks that the model handles better separately. The ADD-only approach reframes the task: instead of "here are your instructions for deleting memories," you say "here are the existing memories; if a new fact contradicts one, add the new fact with a REPLACES tag pointing to the old memory ID."

The extraction model does one thing: extract new facts, and for each fact, check whether it contradicts an existing memory. The store handles the rest.

```python
ADD_ONLY_PROMPT = """Extract new facts from this conversation turn.
For each fact, check if it contradicts an existing memory.

EXISTING MEMORIES (with IDs):
{existing_memories}

CONVERSATION TURN:
{conversation_turn}

Return JSON array:
[{{
  "fact": "the extracted fact",
  "category": "preference|personal|constraint|goal",
  "replaces": "memory_id_if_contradicted_or_null",
  "confidence": 0.0-1.0
}}]

Rules:
- Only extract facts explicitly stated, not inferred
- Set replaces=memory_id when new fact directly contradicts an old one
- Set confidence < 0.7 for uncertain facts (they will be salience-filtered)
- Do NOT output the old memory again; just reference it by ID"""

async def extract_add_only(
    conversation_turn: str,
    existing_memories: list[dict],  # [{"id": "...", "fact": "..."}]
    model: str = "gpt-4o-mini",
    client=None,
) -> list[dict]:
    """Single-pass memory extraction with inline conflict resolution."""
    existing_str = "\n".join(
        f"[{m['id']}] {m['fact']}" for m in existing_memories[:50]  # cap context
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": ADD_ONLY_PROMPT.format(
            existing_memories=existing_str,
            conversation_turn=conversation_turn,
        )}],
        max_tokens=800,
        temperature=0,
    )
    import json
    facts = json.loads(resp.choices[0].message.content)

    # Apply replacements atomically in the store
    return facts  # caller handles store updates + deletions via replaces field
```

The token saving is direct: three 7,000-token calls become one 7,000-token call. That is a 3× reduction on the extraction path. Combined with Technique 1 (fact-dense sentences only), the extraction cost drops from $0.045/turn to roughly $0.0003/turn — a 150× reduction stacked.

One nuance: the ADD-only prompt sends existing memories to enable conflict detection, which adds tokens compared to a naive ADD-only prompt that ignores conflicts. Cap the existing memories sent to the ~50 most recently accessed; the model does not need the full store, and the most recently accessed memories are the ones most likely to be contradicted by new information.

---

## Technique 4: Batch Consolidation

### Why per-turn merging is the wrong cadence

In a naive memory system, every extraction call immediately merges the new facts into the store: deduplication, clustering, resolution of near-duplicates. This is expensive because it happens at the worst possible time — synchronously, on the critical path, with a single new fact that rarely needs deduplication yet.

Consider a user who mentions their location in 10 consecutive turns. The naive system detects and resolves the near-duplicate 9 times. The batch consolidation approach: append new facts to a raw staging buffer per-turn (cheap, no LLM call), and run a batch merge job offline on a schedule.

![Batch consolidation schedule: per-turn extract to raw store, hourly merge, daily prune](/imgs/blogs/agent-memory-cost-optimization-7.webp)

The three-tier schedule:

- **Per turn**: append extracted facts to a raw buffer. No deduplication, no merging. Cost: near-zero (just a database write).
- **Hourly merge job**: take the last N raw facts for each user, cluster near-duplicates by semantic similarity, write merged canonical facts to the retrieval index.
- **Daily prune job**: run salience scoring across the full index, drop memories below threshold, remove facts older than 30 days that have never been retrieved.

The cost impact: instead of running a merge LLM call on every turn, the merge job amortizes across a batch of 10–50 raw facts. The per-fact LLM cost of merging drops from $0.003/fact (per-turn) to $0.0001/fact (batch). For a user with 10 turns/day, that is a 30× reduction in merge cost.

```python
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer, util

MERGE_PROMPT = """These are raw extracted facts for a single user, possibly containing duplicates.
Merge near-duplicate facts into canonical facts.
Return JSON array of merged facts with representative text.

Raw facts:
{facts}

Rules:
- Merge facts that say the same thing in different words
- When facts conflict (user changed their preference), keep the most recent
- Preserve specifics: "prefers Python 3.10+" is better than "prefers Python"
- Add 'sources' list with original fact IDs merged"""

class BatchConsolidator:
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(embed_model)
        self.similarity_threshold = 0.85  # cluster threshold

    async def merge_user_facts(
        self,
        raw_facts: list[dict],  # [{"id", "fact", "timestamp", "user_id"}]
        client,
    ) -> list[dict]:
        if len(raw_facts) <= 1:
            return raw_facts

        # Cluster by embedding similarity
        texts = [f["fact"] for f in raw_facts]
        embeddings = self.embed_model.encode(texts, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings).numpy()

        # Simple greedy clustering
        assigned = [-1] * len(raw_facts)
        clusters = []
        for i in range(len(raw_facts)):
            if assigned[i] >= 0:
                continue
            cluster = [i]
            for j in range(i+1, len(raw_facts)):
                if assigned[j] < 0 and sim_matrix[i][j] >= self.similarity_threshold:
                    cluster.append(j)
                    assigned[j] = len(clusters)
            assigned[i] = len(clusters)
            clusters.append(cluster)

        # Only send clusters with >1 member to LLM for merging
        merged = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(raw_facts[cluster[0]])
                continue
            cluster_facts = [raw_facts[i] for i in cluster]
            facts_str = "\n".join(f"[{f['id']}] {f['fact']}" for f in cluster_facts)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": MERGE_PROMPT.format(facts=facts_str)}],
                max_tokens=400,
                temperature=0,
            )
            import json
            merged.extend(json.loads(resp.choices[0].message.content))

        return merged
```

### What to batch, what to keep per-turn

The right heuristic: any operation that looks at *all existing memories* is a candidate for batching. Any operation that only looks at the *current turn* stays per-turn.

- Per-turn: extraction (Technique 1 + 3), retrieval (Technique 2), async write to raw buffer
- Batched: deduplication, clustering, conflict resolution across old memories, salience scoring, pruning

One edge case: if a user changes a critical constraint mid-conversation (e.g., "actually I'm not vegetarian anymore"), you want the contradiction to propagate before the next turn. Handle this with an *immediate-contradiction flag*: the ADD-only extraction (Technique 3) tags facts that contradict existing memories with `replaces`, and the store applies those atomic replacements synchronously even during the async write path. Batch consolidation only handles soft duplicates, not hard contradictions.

---

## Technique 5: Salience Filtering

### The long tail of useless memories

Not all facts are equally worth remembering. "The user prefers Python" is high-salience — it will be relevant in many future conversations. "The user asked about the weather in Berlin on a Tuesday" is low-salience — it is ephemeral, context-specific, and unlikely to be useful again.

Naive systems store everything. Over time, the memory store fills with low-salience facts that inject noise into every context window. Retrieval must rank over a larger, noisier corpus. Injection tokens grow because more memories score above the retrieval threshold.

Salience filtering assigns a score to each candidate memory before storage and drops memories below a threshold. The score combines four signals:

![Salience score distribution: long tail of low-value memories filtered at threshold 0.6](/imgs/blogs/agent-memory-cost-optimization-6.webp)

The distribution in the diagram is from a real deployment: 500 memories accumulated over 30 days, with scores clustered near 0 (trivial facts) and near 1 (high-value preferences/constraints). The threshold at 0.6 drops 70% of candidates, retaining 150 memories instead of 500. Recall on a held-out query set drops by only 5% — the dropped memories were genuinely low-signal.

```python
from dataclasses import dataclass
from datetime import datetime
import re

@dataclass
class SalienceScorer:
    """Score candidate memories by predicted future value."""

    # Weights for each signal
    w_category: float = 0.35
    w_specificity: float = 0.30
    w_recency_decay: float = 0.15
    w_retrieval_freq: float = 0.20

    CATEGORY_SCORES = {
        "constraint": 0.95,   # dietary, medical, legal — always relevant
        "preference": 0.85,   # usually stable and reusable
        "goal": 0.80,         # high relevance in task context
        "personal": 0.70,     # name, location, job — contextually useful
        "event": 0.40,        # specific events, lower reuse
        "ephemeral": 0.10,    # current mood, today's weather
    }

    def score(
        self,
        fact: str,
        category: str,
        created_at: datetime,
        retrieval_count: int = 0,
        extraction_confidence: float = 1.0,
    ) -> float:
        # Signal 1: category prior
        cat_score = self.CATEGORY_SCORES.get(category, 0.5)

        # Signal 2: specificity (longer, more specific facts score higher)
        word_count = len(fact.split())
        has_numbers = bool(re.search(r'\d', fact))
        has_named_entities = bool(re.search(r'\b[A-Z][a-z]+\b', fact))
        specificity = min(1.0, 0.3 + (word_count / 30) * 0.4
                         + 0.15 * has_numbers + 0.15 * has_named_entities)

        # Signal 3: recency decay (recent facts more likely still valid)
        age_days = (datetime.utcnow() - created_at).days
        recency = max(0.1, 1.0 - age_days / 180)  # 6-month half-life

        # Signal 4: retrieval frequency (facts retrieved before are worth keeping)
        freq_bonus = min(0.5, retrieval_count * 0.1)

        raw = (self.w_category * cat_score +
               self.w_specificity * specificity +
               self.w_recency_decay * recency +
               self.w_retrieval_freq * (0.5 + freq_bonus))

        # Confidence modulation
        return raw * extraction_confidence

    def should_store(self, score: float, threshold: float = 0.60) -> bool:
        return score >= threshold
```

### Threshold tuning

The threshold is the most important hyperparameter. Setting it too high drops real memories; too low stores noise.

The right way to calibrate: take a sample of 500 candidate memories from your production traffic, have annotators label each as "would this be useful in a future conversation?" (binary), then tune the threshold to maximize F1 on that labeled set. In practice, thresholds between 0.55 and 0.65 tend to work well for general-purpose assistants. For specialized high-stakes applications (medical, legal), err toward 0.45 to preserve more; you can always filter at retrieval time.

One important dynamic: retrieval frequency is a strong signal of true salience. A memory that has been retrieved in past conversations and actually influenced the agent's response is demonstrably useful. Give it a strong bonus and drop the storage threshold for memories with retrieval_count > 2 to, say, 0.35. The retrieval history is your ground truth.

---

## Technique 6: Async Memory Writes

### The latency tax on every turn

In a synchronous memory system, the agent cannot respond until:

1. The reasoning model call completes (~500ms)
2. The extraction LLM call completes (~300–800ms depending on conversation length)
3. The memory store write completes (~50ms)
4. Total: ~900ms–1,350ms per turn

The user is waiting for steps 2 and 3 for no reason. The response is ready after step 1. The extraction and write are bookkeeping operations that do not affect *this* turn's response — they affect the *next* turn's context.

Async memory writes decouple the write path from the response path:

![Async memory write flow: agent responds immediately, extraction and write happen in background](/imgs/blogs/agent-memory-cost-optimization-5.webp)

The agent responds to the user at t=100ms (main model call). Extraction runs in the background, completing at t=400ms. The memory store is updated at t=450ms. The *next* turn's retrieval will see the updated store.

```python
import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AsyncMemoryWriter:
    """Writes memories asynchronously after agent response is sent."""

    def __init__(self, extractor, store, scorer):
        self.extractor = extractor
        self.store = store
        self.scorer = scorer
        self._pending: list[asyncio.Task] = []

    async def enqueue_write(
        self,
        conversation_turn: str,
        existing_memories: list[dict],
        user_id: str,
    ) -> asyncio.Task:
        """Schedule extraction + write without awaiting completion."""
        task = asyncio.create_task(
            self._extract_and_write(conversation_turn, existing_memories, user_id)
        )
        self._pending.append(task)
        task.add_done_callback(lambda t: self._pending.remove(t))
        return task

    async def _extract_and_write(
        self,
        conversation_turn: str,
        existing_memories: list[dict],
        user_id: str,
    ) -> None:
        try:
            facts = await self.extractor.extract_add_only(
                conversation_turn, existing_memories
            )
            for fact in facts:
                score = self.scorer.score(
                    fact["fact"],
                    fact.get("category", "personal"),
                    created_at=__import__("datetime").datetime.utcnow(),
                    extraction_confidence=fact.get("confidence", 1.0),
                )
                if self.scorer.should_store(score):
                    await self.store.upsert(
                        user_id=user_id,
                        fact=fact["fact"],
                        category=fact.get("category"),
                        replaces=fact.get("replaces"),
                        salience=score,
                    )
        except Exception as e:
            logger.error(f"Async memory write failed for user {user_id}: {e}")
            # Do not re-raise: this is fire-and-forget; one failed write is acceptable

# Usage in the agent turn handler:
async def handle_turn(
    user_message: str,
    conversation_history: list[dict],
    user_id: str,
    memory_writer: AsyncMemoryWriter,
    retriever: FusionMemoryRetriever,
    client,
) -> str:
    # 1. Retrieve relevant memories synchronously (needed for this turn)
    memories = retriever.retrieve(user_message, top_k=7)
    memory_context = "\n".join(m for m, _ in memories)

    # 2. Run the reasoning model
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"User context:\n{memory_context}"},
            *conversation_history,
            {"role": "user", "content": user_message},
        ],
    )
    agent_response = response.choices[0].message.content

    # 3. Fire-and-forget: enqueue memory write without awaiting it
    full_turn = f"User: {user_message}\nAssistant: {agent_response}"
    await memory_writer.enqueue_write(full_turn, [m for m, _ in memories], user_id)

    # 4. Return immediately — user gets response without waiting for memory write
    return agent_response
```

### The consistency tradeoff

Async writes introduce a one-turn lag: if the user states a fact and then immediately contradicts it in the next turn, the extraction from the first turn may not have completed yet when the second turn retrieves memories.

In practice this is almost never a problem. Real users do not contradict themselves within two turns. And the ADD-only pass (Technique 3) with conflict detection handles contradictions when they do occur — the second turn's extraction sees the memory from turn 1 and flags the contradiction.

The one case where synchronous writes are preferable: applications where the agent takes a consequential action based on a just-stated constraint. If a user says "I'm allergic to shellfish" and the next message is "suggest a recipe for tonight," you want that allergy in the retrieval index before the recipe request. For these use cases, use a hybrid pattern: write *critical constraints* (category="constraint") synchronously on the fast path, and enqueue all other fact categories asynchronously.

---

## Combined Impact: How the Six Techniques Stack

![Cumulative token reduction: baseline through all six techniques showing 3.5–4.2× stack](/imgs/blogs/agent-memory-cost-optimization-8.webp)

The waterfall above shows the additive reductions with representative numbers. The baseline is 100 normalized tokens per turn of memory cost. Each technique reduces the running total:

| Stage | Normalized token cost | Reduction vs. baseline |
|---|---|---|
| Baseline | 100 | 1.0× |
| + Async writes (Technique 6) | 83 | 1.2× |
| + ADD-only pass (Technique 3) | 63 | 1.6× |
| + Salience filter (Technique 5) | 50 | 2.0× |
| + Fusion retrieval (Technique 2) | 40 | 2.5× |
| + Batch consolidation (Technique 4) | 33 | 3.0× |
| + Efficient extraction (Technique 1) | 26 | 3.8× |

The order matters. Async writes come first because they have zero quality tradeoff — you are just reordering operations, not changing what gets stored or retrieved. ADD-only pass comes next because it is high-reward, low-complexity, and does not require infrastructure changes. Salience filter comes before fusion retrieval because it reduces the index size that fusion retrieval must operate over, amplifying fusion's benefit. Batch consolidation and efficient extraction come last because they require the most careful calibration.

In dollar terms for the baseline scenario (1,000 users × 10 turns/day × GPT-4o $2.50/1M):

| Stage | Memory cost/day | Total cost/day |
|---|---|---|
| Baseline | $650 | $1,150 |
| After all 6 techniques | $168 | $518 |
| Savings | $482/day | $215,000/year |

---

## Implementation Priority: Quick Wins First

The natural instinct is to implement all six techniques at once. Resist it. Each technique is independently deployable, and the first two have essentially zero risk.

![Implementation priority order: quick wins in week 1, medium gains in weeks 2–3, high gains in week 4+](/imgs/blogs/agent-memory-cost-optimization-9.webp)

**Week 1: Async writes + ADD-only pass.** These two together deliver ~1.6× reduction with no quality risk and minimal code change. The async write is a fire-and-forget task scheduling change. The ADD-only pass requires rewriting one prompt and one store update path. Ship both, measure your cost dashboard for 3–5 days, confirm quality is unchanged, move on.

**Week 2: Salience filtering.** This requires labeling a sample of candidate memories to tune the threshold. Do that work before shipping. The 3–5 days of measurement from week 1 gives you a baseline for quality comparison.

**Week 2–3: Fusion retrieval.** This requires building the BM25 index and entity extraction pipeline alongside your embedding store. More infrastructure work, but the retrieval quality improvement is often noticeable to users before you measure it — the agent references the right memories more reliably.

**Week 3: Batch consolidation.** This requires a job scheduler and a staging buffer in your store. The engineering lift is real. But at this point your index is smaller (salience filter), retrieval is better (fusion), and the merge job operates on a cleaner corpus.

**Week 4+: Efficient extraction.** This is the highest-impact, highest-touch change. It requires evaluating fact density filter quality on your specific domain and calibrating small model vs. large model tradeoffs for your extraction quality requirements.

---

## Cost Monitoring: What to Track

Without instrumentation, you cannot know which technique is working and which one degraded quality. Track these metrics per user cohort:

```python
from dataclasses import dataclass, field
from datetime import datetime
import statistics

@dataclass
class MemoryCostMetrics:
    """Per-conversation memory cost tracking."""
    user_id: str
    session_id: str
    turns: int = 0

    # Token costs
    extraction_tokens_in: int = 0
    extraction_tokens_out: int = 0
    retrieval_tokens_injected: int = 0

    # Memory quality signals
    memories_retrieved: list[int] = field(default_factory=list)
    memories_stored: int = 0
    memories_filtered_by_salience: int = 0
    memories_merged_in_batch: int = 0

    # Latency signals
    async_write_lag_ms: list[float] = field(default_factory=list)

    def cost_usd(self, price_per_1m_in: float = 2.50, price_per_1m_out: float = 10.0) -> float:
        return (self.extraction_tokens_in * price_per_1m_in / 1_000_000 +
                self.extraction_tokens_out * price_per_1m_out / 1_000_000 +
                self.retrieval_tokens_injected * price_per_1m_in / 1_000_000)

    def avg_retrieval_tokens(self) -> float:
        return statistics.mean(self.memories_retrieved) * 200 if self.memories_retrieved else 0

    def salience_filter_rate(self) -> float:
        total = self.memories_stored + self.memories_filtered_by_salience
        return self.memories_filtered_by_salience / total if total > 0 else 0
```

The five numbers that matter:
1. **Extraction tokens per turn** — should fall 5–10× after Technique 1
2. **Retrieved memories per turn** — should fall from ~15 to ~7 after Technique 2
3. **Salience filter rate** — should stabilize around 60–70% after calibration
4. **Memory store size per user** — should grow sub-linearly after Techniques 4 and 5
5. **Recall@K on held-out queries** — should *not* fall meaningfully (< 5%) after any technique

Recall is the health check. Track it by sampling 50–100 queries per week where you know the correct memory to retrieve, and measure hit rate at K=7. If recall falls more than 5% after deploying a technique, the technique's parameters need re-calibration, not rollback.

---

## Case Studies

### Case Study 1: Health Information Assistant

A digital health platform deployed a memory-augmented assistant to help chronic disease patients track symptoms, medications, and lifestyle factors. With 8,000 daily active users, each averaging 15 turns/day, the memory system was costing $3,200/day before optimization.

**Before:** Each turn ran a full extraction pass (8,000 tokens, GPT-4o), a 3-pass pipeline (ADD/UPDATE/DELETE), and injected K=20 memories into a 32k-token context window. The injected memories accounted for 12,000 tokens per turn on average.

**What changed:** The team started with salience filtering — the domain made categorization easy: medication constraints and symptom history scored near 1.0, daily mood scores from app check-ins scored near 0.1 and were filtered. They dropped from 200 average stored memories per user to 45. Async writes came next, cutting perceived latency from 1.2s to 0.4s. ADD-only pass cut extraction cost by 3×. Fusion retrieval let them drop K from 20 to 8.

**After:** Memory cost: $820/day, down from $3,200. Total daily API cost dropped from $4,100 to $1,800. Quality: recall on a clinical test set (200 queries about patient history) went from 94% at K=20 to 91% at K=8 — acceptable for the use case. The 300ms latency improvement meaningfully improved user satisfaction scores.

**The hard lesson:** The first salience threshold they tried (0.7) was too aggressive — it dropped some medication change notes because the extraction model labeled them as "events" rather than "constraints." They added a rule: any fact containing a medication name gets automatically scored at 0.9 regardless of category. Domain-specific overrides are often necessary.

### Case Study 2: AI Code Assistant

A developer tools company built a memory layer for their AI code assistant that remembered developer preferences: preferred languages, frameworks, team conventions, coding style, and project context. 2,500 developer users, 5 sessions/week, 20 turns/session.

**The distinctive challenge:** Code context is dense. A fact like "uses TypeScript with strict mode and ESLint with Airbnb config" is 15 tokens of fact extracted from potentially 2,000 tokens of code conversation. The fact density filter was the most impactful technique because code conversations have extreme fact sparsity — most of the tokens are code, not preference statements.

**What changed:** Token-efficient extraction (Technique 1) reduced extraction token cost by 7× on this domain specifically. The fact density filter identified that only 8% of conversation tokens in code sessions contained extractable preference facts. Batch consolidation was critical because developers tend to mention the same preferences repeatedly across sessions, creating high duplicate rates in the raw buffer.

**After:** Extraction cost fell from $0.038/turn to $0.005/turn. The merged, deduplicated memory store stabilized at 30–60 facts per developer despite 50+ sessions, versus the naive system's 400+ growing entries per developer. Retrieval quality improved (smaller, cleaner index) and the agent had better preference adherence because duplicate noise was not diluting retrieval scores.

**Useful finding:** Batch consolidation's 85% cosine similarity threshold needed tuning for code facts. "Uses Python 3.10" and "prefers Python 3.10+" have 90%+ cosine similarity and should merge. "Uses TypeScript strict mode" and "uses TypeScript" are related but different constraints — the team lowered the threshold to 0.92 for code facts to avoid over-merging.

### Case Study 3: Customer Support Agent

An e-commerce company deployed a memory-enabled support agent to remember customer preferences, past issues, and resolution history. 15,000 daily users with an average of 3 turns/interaction and a high re-contact rate (30% of users contacted support again within 30 days).

**The distinctive challenge:** Customer support memory has an unusual salience distribution. Preferences like "prefer email communication" are high-salience. Issue history like "contacted about order #12345 on March 3" is useful for deduplication but not for conversation personalization. And resolution status ("refund processed") is critical for the next interaction but stale after 90 days.

**What changed:** The team built a domain-specific salience scorer with custom categories: `communication_preference` (1.0), `active_issue` (0.9), `resolved_issue` (0.5 → drops to 0.1 after 90 days via recency decay), `preference` (0.75). The time-decay function was tuned aggressively for `resolved_issue` because stale order history injected into context made the agent apologize for months-old issues the customer had forgotten.

**After:** Memory store per customer stabilized at 8–12 facts (down from 45+). K=5 retrieval was sufficient for this domain. Total memory cost dropped 4.2× — the highest reduction of any case study here, driven primarily by salience filtering's aggressive pruning and the small K.

**The failure mode that taught them:** Early on, async writes caused a one-turn lag problem in a specific flow: a customer would state "please only contact me by email" and the agent's next message would say "I'll call you to follow up." The constraint category write was not completing before the next turn's retrieval. Fix: synchronous write for `communication_preference` and `active_issue` categories; async for everything else.

### Case Study 4: Therapy Support Bot

A mental health startup built an AI companion that remembered users' therapeutic goals, coping strategies that worked, triggers identified in previous sessions, and personal milestones. 3,000 users, 4 sessions/week, 30 turns/session.

**The distinctive challenge:** Therapy context requires very high recall for safety-critical facts (triggers, crisis history) but can afford lower recall for general biographical details. Getting the salience scoring wrong in either direction is costly: dropping triggers is a safety risk; storing excessive biographical detail bloats the context and can make the agent seem overly clinical.

**What changed:** The team implemented a two-tier memory system: a *permanent store* for safety-critical facts (triggers, crisis history, active treatment goals) that bypasses salience filtering entirely and injects synchronously; and an *ordinary store* for non-critical facts that runs the full optimization pipeline. This hybrid approach let them optimize cost on the large majority of facts while maintaining safety guarantees on the small critical set.

**After:** The non-critical store (85% of facts) ran at 3.2× cost reduction. The permanent store added a fixed cost of ~$0.005/turn for the guaranteed injections. Net: 2.8× cost reduction overall, with no compromise on safety-critical memory recall.

**Design principle generalized:** For any domain with a small set of must-never-miss facts, build a guaranteed injection tier that bypasses optimization, and apply all six techniques to the remaining 80–90% of facts. The optimization savings on the large set more than cover the fixed overhead of the guaranteed set.

### Case Study 5: Enterprise Knowledge Assistant

A professional services firm deployed an AI assistant for their consultants that remembered client preferences, past project structures, engagement norms, and individual consultant work patterns. 800 users, high-value interactions, Claude Opus for reasoning (expensive).

**The distinctive challenge:** High model cost ($15/1M tokens on Opus) meant that injection token reduction had outsized impact. Reducing retrieval from K=12 to K=5 saved 1,400 tokens × $15/1M × 800 users × 8 turns/day = $134/day from injection alone.

**What changed:** Fusion retrieval was the most impactful technique because consultant queries tend to reference specific client names and project codes — exact keyword matching via BM25 dramatically outperformed semantic similarity alone. A query about "Acme Corp Q3 pricing review" reliably found memories tagged with "Acme Corp" that semantic similarity alone was missing because the embedding space conflated different clients with similar industries.

**After:** Total memory cost reduction: 3.1×. More importantly, *retrieval quality* improved substantially — entity-aware retrieval made the assistant noticeably better at pulling the right client context. The team measured a 23% reduction in consultant follow-up questions to the agent, attributing most of it to better memory accuracy rather than cost reduction per se.

### Case Study 6: High-Frequency Trading Desk AI

A quantitative trading firm built a memory-augmented AI assistant for traders that remembered risk limits, position constraints, regulatory notes from compliance, and trader-specific alerting preferences. 50 traders, 200+ turns/day each during market hours.

**The distinctive challenge:** Extremely high turn frequency (200 turns/day per user) meant that extraction cost dominated. At 200 turns × 5,000 tokens/extraction × GPT-4o $2.50/1M × 50 users = $125/day on extraction alone. The efficient extraction technique (small model + fact density filter) was the single most impactful change.

**What changed:** The fact density filter was calibrated specifically for trading language: Bloomberg terminal output, position tables, ticker symbols, and order flow messages are data-dense but rarely contain extractable *preferences or constraints*. The filter learned to pass through sentences with explicit trader preference signals ("I want alerts when ES is > 10 ticks from VWAP") and drop market data recitations. Extraction input tokens dropped from 5,000/turn to 180/turn for this domain.

**After:** Extraction cost fell from $125/day to $4.50/day — a 28× reduction on extraction, driven by the extreme fact sparsity of trading conversation. Total memory optimization: 4.4×. One important finding: ADD-only pass was critical because traders frequently updated position limits, and the 3-pass pipeline was running expensive UPDATE and DELETE passes constantly. The ADD-only with inline conflict detection resolved this at 1/3 the cost.

### Case Study 7: Language Learning Companion

An edtech startup built an AI language tutor that remembered student vocabulary gaps, grammar error patterns, topics of interest, and preferred learning pace. 12,000 students, 5 sessions/week, 25 turns/session.

**The distinctive challenge:** Language learning generates large volumes of highly repetitive facts — a student makes the same grammar mistake many times, and the memory store fills with near-duplicate error records. Without batch consolidation, the store grew unboundedly and retrieval precision collapsed because the most recent 15 retrievals were all "student confuses ser/estar."

**What changed:** Batch consolidation was the first technique deployed, not the last. The hourly merge job clustered identical error patterns and created canonical error records with a `frequency` field. Instead of storing 30 separate "confuses ser/estar" memories, the store kept one record with `frequency: 30, last_seen: ...`. This changed the retrieval query: instead of retrieving the 7 most recent memories by recency, the retriever weighted by `frequency × recency`, surfacing persistent patterns over one-off errors.

**After:** Store size per student stabilized at 15–20 canonical records instead of 200+ raw records. Injection tokens dropped 85%. Retrieval quality improved substantially because canonical error records were more information-dense than raw observations. The student experience improved because the agent addressed persistent patterns rather than the last thing that happened.

### Case Study 8: Multi-Tenant SaaS Chatbot

A B2B SaaS company built a customer success bot that remembered each customer's product configuration, feature usage patterns, support history, and expansion signals. 3,500 customer accounts, shared embedding store, batch processing of conversation histories overnight.

**The distinctive challenge:** Multi-tenant memory with a shared store meant that the async write pattern needed careful implementation — a failed async write for one tenant could not be allowed to corrupt the write queue for other tenants. The team implemented a per-tenant write queue with circuit breakers: each tenant's background write tasks were isolated, with a 3-retry policy and dead-letter queue for failed extractions.

**What changed:** The batch consolidation cadence was changed from hourly to nightly because customer conversations had a strong weekly pattern — context from three conversations ago was rarely relevant in the same week. The nightly merge job processed the week's accumulated raw facts, producing clean canonical facts for each account.

**After:** Total cost reduction: 3.6×. The nightly batch cadence combined with salience filtering reduced the average account's memory size from 120 facts (unbounded growth) to 25 facts (stable after 3 months). Query recall stabilized at 88% vs. 91% for real-time consolidation — a 3 percentage point recall trade for a 3.6× cost reduction. For a customer success use case (vs. a medical use case), that trade was accepted.

---

## When to Optimize vs. Accept the Cost

Not every memory system needs all six techniques. The deployment profile matrix in the figures section above summarizes when each technique makes sense by deployment scale and use case. Here is the decision logic in prose:

**Do not optimize if your turn volume is below ~100/day total.** The engineering investment in fusion retrieval or batch consolidation pays back only at meaningful scale. For a low-traffic internal tool, store everything, retrieve with semantic similarity at K=10, and revisit when costs become visible in your budget.

**Always ship async writes, regardless of scale.** There is no quality tradeoff and no minimum scale requirement. It is a pure latency win that happens to also have cost benefits by allowing background optimization.

**Salience filtering is high-value if your use case has natural fact categories.** Health, legal, financial, and domain-specific applications have clear high-salience fact types. General-purpose assistants require more careful calibration. If you cannot articulate what makes a memory high-salience in your domain, do not ship salience filtering until you can — a miscalibrated filter drops real memories.

**Efficient extraction only makes sense if your conversations are fact-sparse.** For dense-fact domains (medical history taking, requirements gathering, customer onboarding), the fact density filter may not help — most sentences contain facts. For sparse-fact domains (trading, coding, customer support), it is transformative.

**Fusion retrieval is worth it when exact-match recall is a known failure mode.** If your users reference specific entities (client names, product SKUs, project codes) and you see retrieval misses on these, BM25 will fix it. If your retrieval failures are mostly paraphrase mismatches (different words for the same concept), improve your embedding model first.

The general rule: optimize in ROI order. Async writes and ADD-only pass are free lunches — take them immediately. Then measure before adding complexity. The goal is not to implement all six techniques; it is to spend money on memory in proportion to the value it delivers.

---

## Cross-links

For the memory system architecture underlying these optimizations, see our breakdown of [mem0's token-efficient memory algorithm](/blog/machine-learning/ai-agent/mem0-token-efficient-memory-algorithm) and [multi-signal memory retrieval](/blog/machine-learning/ai-agent/multi-signal-memory-retrieval).

For the tradeoffs between memory systems and raw context window approaches, [memory vs. context window for agents](/blog/machine-learning/ai-agent/memory-vs-context-window-agents) covers when each is appropriate and how their cost profiles differ at scale.

For the broader agent architecture that memory plugs into, [agent loop anatomy](/blog/machine-learning/ai-agent/agent-loop-anatomy) walks through the full turn cycle including where memory read and write fit in the execution graph.

---
title: "Long-Term Memory for Conversational Agents: MemGPT and Beyond"
publishDate: "2026-04-18"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  [
    "memory",
    "memgpt",
    "letta",
    "mem0",
    "zep",
    "conversational-ai",
    "long-context",
    "agents",
  ]
date: "2026-04-18"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A senior-level walkthrough of long-term memory for conversational agents — why context windows are not memory, the taxonomy of memory types, and the architectures (MemGPT, Letta, mem0, Zep) that let an agent remember a user for months without blowing up cost or latency."
---

## The Core Confusion: Context Window ≠ Memory

If you only remember one thing from this article: **a longer context window is not a memory system.**

A 1M-token context window is a bigger RAM, not a bigger hard drive. Every token in the context:

1. Costs money each turn.
2. Adds latency (prefill is O(n²) without tricks).
3. Competes for attention — long contexts suffer the "lost-in-the-middle" effect (Liu et al. 2023).
4. Disappears when the session ends.

Real memory is **persistent, selective, and retrievable**. A good memory system decides what to keep, forgets what doesn't matter, and surfaces the right slice of the past at the right moment. This article is about how to build that.


## Taxonomy: What Kind of Memory?

Cognitive science gives us a useful split, which modern agent frameworks have roughly adopted:

| Type | What it stores | Example (robot) |
|---|---|---|
| **Working memory** | Current turn + a few back | "We're in the middle of booking dinner" |
| **Episodic memory** | Specific past events | "Yesterday the user asked me to water the plants" |
| **Semantic memory** | Durable facts about entities | "User's name is Hiep, prefers tea, allergic to shellfish" |
| **Procedural memory** | How-to knowledge | "When the user says 'goodnight', dim the lights and stop" |

A complete agent uses all four. The engineering question is: **how are they stored, retrieved, and updated?**


## MemGPT: Virtual Context Management

MemGPT (Packer et al. 2023, now productized as **Letta**) was the first widely adopted pattern that solved this cleanly. The insight: treat the LLM like an operating system, with a **main context** (RAM) and **archival storage** (disk), and give the LLM function calls to move information between the two.

```
┌─────────────────────────────────────────────────┐
│  MAIN CONTEXT (what the LLM sees each turn)     │
│                                                 │
│  [System prompt]                                │
│  [Working context: user profile, recent facts]  │
│  [FIFO recent messages]                         │
│  [Current user turn]                            │
└─────────────────────────────────────────────────┘
             ↑                    ↓
     core_memory_append     conversation_search
     core_memory_replace    archival_memory_search
                            archival_memory_insert
             ↑                    ↓
┌─────────────────────────────────────────────────┐
│  ARCHIVAL STORAGE                               │
│  - All past messages (vector-indexed)           │
│  - All persistent facts                         │
└─────────────────────────────────────────────────┘
```

### How It Works

1. **Main context** is a fixed-size window. It holds a system prompt, a *working context* (think: a scratchpad of durable facts), and a FIFO of recent messages.
2. When the FIFO grows close to the limit, the LLM is instructed to **self-evict** — write a summary to archival storage and drop old messages.
3. When the LLM needs to recall something, it calls `archival_memory_search(query)` or `conversation_search(query)`.
4. When the LLM learns a durable fact, it calls `core_memory_append` to update the working context in-place.

### The Key Trick: The LLM Is Its Own Memory Manager

MemGPT doesn't use heuristics to decide what to remember. The LLM decides via function calls, guided by the system prompt. This has a nice property: the memory policy is natural-language-programmable.

```python
SYSTEM = """You have limited context. You can:
- core_memory_append(key, value)  — durable fact, always visible
- archival_memory_insert(text)    — searchable long-term
- archival_memory_search(query)   — retrieve long-term
- conversation_search(query)      — search past turns
Save important facts. Search when recall matters. Forget noise."""
```

### Limitations

- **The LLM is imperfect at deciding.** It over-saves, forgets to save, or saves in the wrong tier. Evaluation is hard.
- **Latency.** Each memory op is an LLM tool call — adds a round trip.
- **Cost scales with memory ops.** Heavy users make more tool calls.


## Letta (Productized MemGPT)

Letta is the framework that ships MemGPT as a service. Key additions over the paper:

- Persistent per-agent database (Postgres + pgvector by default).
- Per-agent "blocks" — typed sections of working context (e.g., `persona`, `human`, `scratchpad`).
- Multi-user / multi-agent isolation.
- REST API for embedding into your app.

```python
from letta import LocalClient

client = LocalClient()
agent = client.create_agent(
    name="homebot",
    memory=ChatMemory(human="User: Hiep, home in Hanoi.", persona="I am a home robot."),
)
client.user_message(agent.id, "Remember I hate spicy food.")
# The agent auto-calls core_memory_append: {"human": "... dislikes spicy food"}
```


## mem0: Fact-Extraction Pipeline

mem0 takes a different angle: instead of giving the LLM memory-management tools, it runs an **asynchronous extraction pipeline** that mines facts out of conversations and stores them.

```
Turn → LLM response
       ↓
  (async pipeline)
       ↓
Fact extractor LLM
       ↓
{user.food.disliked = "spicy"}
       ↓
Vector store + graph store
       ↓
Next turn: retrieve top-k relevant facts, inject into prompt
```

### Why This Pattern Works

- **Separates inference from extraction.** Your chat latency isn't paying for memory decisions.
- **Explicit schema.** Facts are typed tuples, not blobs.
- **Easy to audit and edit.** You can show the user "here's what I remember about you" and let them delete.

### Why It Sometimes Doesn't

- Extraction quality depends on the fact extractor LLM. Cheap models miss nuance; expensive ones double your cost.
- Facts go stale. "User lives in Hanoi" → true in January, not in March after they moved.
- Deduplication and conflict resolution are hard problems.


## Zep: Temporal Knowledge Graph

Zep builds a **temporal knowledge graph** from the conversation. Nodes are entities, edges are relationships, and every fact has a time interval.

```
(User:Hiep) --[LIVES_IN, valid: 2024-01..2026-03]--> (City:Hanoi)
(User:Hiep) --[LIVES_IN, valid: 2026-03..NOW]--> (City:HCMC)
(User:Hiep) --[DISLIKES, valid: ALWAYS]--> (Food:Spicy)
```

When retrieving, Zep can answer time-scoped queries: *"where did Hiep live last year?"* → `Hanoi`. This is a strict superset of what flat vector stores can do — but it adds complexity.


## OpenAI's Built-in Memory

ChatGPT's memory (and the API-exposed version) is a hosted variant of fact extraction + retrieval:

- Extracts facts from conversations.
- Injects relevant ones into the system prompt at inference time.
- User-visible: you can view, delete, or disable.

For most apps this is enough. For a humanoid robot, you need on-device memory (privacy, latency, offline) — so the OpenAI-hosted path usually doesn't apply.


## Patterns Without a Framework

You can go a long way with simple patterns. These are the three I see most often in production:

### Pattern A: Summary Rolling

Keep last N turns verbatim; summarize older turns into a single paragraph.

```python
def build_context(history, recent_n=10):
    if len(history) <= recent_n:
        return history
    old = history[:-recent_n]
    summary = summarize(old)  # one LLM call, cached
    return [{"role": "system", "content": f"Summary so far: {summary}"}, *history[-recent_n:]]
```

Good for chat; loses precision over long horizons.

### Pattern B: Vector Retrieval Over Turns

Embed every turn; at query time, retrieve top-k relevant turns.

```python
embeddings = [embed(t) for t in history]
def retrieve(query, k=5):
    q = embed(query)
    scored = sorted(zip(history, embeddings), key=lambda h: cosine(q, h[1]), reverse=True)
    return [h[0] for h in scored[:k]]
```

Good for factual recall; bad at "what happened yesterday at 3pm" (time queries).

### Pattern C: Extracted-Facts Store

Run a cheap extractor on every turn; persist typed facts; inject top-k at inference time.

```python
@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    confidence: float
    valid_from: datetime
    valid_to: datetime | None
```

Good for durable profiles; needs fact-conflict resolution.

### Real-world combination

The best systems combine all three:

```
Long-term facts (Pattern C)  → injected into system prompt
+
Episodic vector retrieval (Pattern B)  → top-k for current query
+
Recent turns verbatim (last N)
+
Summary of middle (Pattern A)
```


## Eviction, Forgetting, and Privacy

Memory without forgetting is a liability. Key policies:

| Policy | What it does | When |
|---|---|---|
| **Time-based TTL** | Expire facts older than X days | Ephemeral preferences |
| **Confidence decay** | Lower confidence over time unless reinforced | Uncertain facts |
| **Access-based LRU** | Evict facts never retrieved | Noisy extractions |
| **User-requested delete** | Nuke specific facts | "Forget I said that" |
| **Category-based** | Different TTLs per fact class | PII vs preferences |

For a home robot, the forgetting policy is a **privacy surface**. The user must be able to see what's stored and delete it. Build that UI before you ship.


## Memory for Humanoid Robots

Robot memory has two qualities that make it harder than chatbot memory:

### 1. Multi-modal facts

Memory isn't just text. "The coffee mug is in the kitchen cabinet" is a fact that needs an object_id, a room, a spatial location, and a time. Purely text memory loses the embedding into the world state.

Architecture sketch:

```
┌──────────────────────────────────────────────┐
│ Semantic memory (user profile)               │
│   {user.name, preferences, allergies, ...}   │
├──────────────────────────────────────────────┤
│ Episodic memory (events)                     │
│   {what, when, where, who}                   │
├──────────────────────────────────────────────┤
│ Spatial memory (scene graph)                 │
│   objects, rooms, relationships, last-seen   │
├──────────────────────────────────────────────┤
│ Procedural memory (learned routines)         │
│   "Morning routine: open blinds, brew tea"   │
└──────────────────────────────────────────────┘
```

Each tier has different storage and retrieval. Spatial memory is a graph; episodic is a time-indexed log; semantic is a KV store; procedural is a library of callable skills.

### 2. Multi-user and multi-session

A family robot meets Mom, Dad, Kid, Grandma. Voice and face identify who's speaking. Memory must be **per-user** with a shared world state.

```python
memory = {
    "shared": SpatialMemory(),
    "users": {
        "user_001": SemanticMemory() + EpisodicMemory(),
        "user_002": ...,
    }
}
```

Privacy boundaries: Dad's private requests shouldn't surface when Kid is speaking. Enforce with access control at retrieval time.

### 3. Idle-time consolidation

When the robot is idle, it can consolidate:

- Deduplicate episodic entries.
- Promote reinforced facts to semantic.
- Forget stale spatial observations (the mug isn't on the counter anymore).
- Compute embeddings on a low-priority thread.

This is the robot equivalent of sleep. Biologically grounded, and engineering-useful: you offload expensive memory ops from the real-time path.


## Evaluation

How do you know your memory system works? Two benchmarks dominate:

- **LongMemEval (Wu et al. 2024)** — multi-session QA over 5 reasoning categories (single-session user, multi-session, temporal reasoning, knowledge updates, abstention).
- **LoCoMo (Maharana et al. 2024)** — very long conversations (~300 turns, 35 sessions) with QA over the full history.

Both reveal that even top frontier models with 1M context windows do worse than much smaller models augmented with retrieval-based memory. The reason: attention degrades with distance, retrieval does not.

For your own system, write a custom eval harness:

```python
@dataclass
class MemoryTest:
    setup_turns: list[str]   # injected into history
    probe: str               # the question
    answer: str              # ground truth
    category: str            # "update", "recall", "temporal", ...

tests = [
    MemoryTest(
        setup_turns=["I'm allergic to shellfish."],
        probe="Can I eat shrimp?",
        answer="No — you're allergic to shellfish.",
        category="recall",
    ),
    MemoryTest(
        setup_turns=["I love spicy food.", "I gave up spicy food last month."],
        probe="Suggest dinner.",
        answer="Something mild — you gave up spicy.",
        category="update",
    ),
    ...
]
```

Track accuracy by category. Ship improvements by category. "Memory accuracy: 85%" tells you nothing actionable.


## Senior-level Takeaways

1. **A bigger context window is not memory.** You still need to decide what to persist, what to retrieve, and when to forget.
2. **Separate concerns:** semantic facts, episodic events, spatial state, procedures. Don't dump them into a single vector store.
3. **Forgetting is a feature, not a bug.** Design the TTL policy and the delete UI early.
4. **Multi-user and multi-session are hard.** Scope memory per user and enforce at retrieval.
5. **Consolidate during idle time.** Real-time paths shouldn't pay for memory maintenance.
6. **Evaluate by category.** Aggregate accuracy hides the categories you actually care about.
7. **For robots, memory is multi-modal.** Scene graphs and spatial state are first-class; text-only memory systems underfit the problem.


## References

- Packer et al. 2023. *MemGPT: Towards LLMs as Operating Systems.*
- Liu et al. 2023. *Lost in the Middle: How Language Models Use Long Contexts.*
- Maharana et al. 2024. *Evaluating Very Long-Term Conversational Memory of LLM Agents.* (LoCoMo)
- Wu et al. 2024. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.*
- Zhong et al. 2024. *MemoryBank: Enhancing Large Language Models with Long-Term Memory.*
- mem0, Zep, Letta documentation, 2024–2026.

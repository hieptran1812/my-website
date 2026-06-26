---
title: "Context Window Management: Keeping Agents Sharp Across Long Conversations"
date: "2026-06-27"
description: "Practical strategies for managing context window pressure in production agents — compression, sliding windows, summarization, token budgets, and intelligent pruning."
tags: ["ai-agents", "context-window", "memory", "llm", "machine-learning", "nlp", "production-ml", "optimization"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 41
---

Every long-running agent eventually hits the same wall. The conversation starts clean — a crisp system prompt, a few tool definitions, a single user message. Twenty turns later, the context window holds the accumulated weight of every tool call result, every reasoning chain, every retrieved document, and every assistant turn that came before. The model hasn't gotten dumber. But it responds as if it has: it forgets facts stated ten turns ago, repeats answers it already gave, loses track of the user's original intent, and costs ten times more per call than it did at turn one.

This is context pressure, and it's the most commonly underestimated problem in production agent systems.

The diagram below shows the mental model: every production agent context splits into six named slots, each competing for a fixed budget. The system prompt and tool definitions are static and potentially large. Conversation history is dynamic and grows without bound. Retrieval results spike on certain queries. The response budget must always be protected. Understanding which slots you can control — and how — is the starting point for every strategy in this post.

![Context window anatomy — six named slots each with a token budget](/imgs/blogs/context-window-management-1.webp)

We'll go deep on six strategies: sliding window, summarization, token budget allocation, importance scoring, hierarchical compression, and prefix caching. Then we'll cover the lost-in-the-middle problem (why position matters as much as presence), a ~80-line Python implementation, and eight case studies from production.

## 1. The Context Pressure Problem: Why Agents Degrade After N Turns

A typical GPT-4-class call at turn one costs roughly $0.01–0.05 depending on your system prompt size. By turn twenty in an unmanaged agent, that same call can cost $0.30–0.50. By turn fifty, if your model supports it, you're approaching $1.00 per call — and the model's effective accuracy on information from turns five through fifteen has likely dropped to the 45–65% range.

The degradation has two independent causes:

**Cost explosion.** Every token in the context window is billed on input. A system prompt of 6,000 tokens and tool definitions of 4,000 tokens already use 10,000 tokens before the conversation starts. Each turn adds more. A 50-turn conversation with a typical assistant that uses tools aggressively can accumulate 80,000–120,000 tokens of history. At $3 per million input tokens (GPT-4o pricing), that's $0.24–$0.36 just for the history portion of a single call. Multiply by the number of calls in a session, and you have a serious unit-economics problem.

**Quality degradation.** This is subtler and more insidious. Two distinct mechanisms drive it:

*The lost-in-the-middle effect.* Large language models have a strong recency bias and a primacy bias, but they dramatically underweight information from the middle of a long context. A document or conversation turn at position 15 in a 50-turn context may as well not exist for most retrieval and reasoning tasks. We'll look at the empirical evidence in detail in section 10.

*Attention dilution.* Transformer attention is computed over all tokens in the context. As the context grows, the model's effective "attention budget" per token shrinks. Important signals that were crisp at short contexts become diluted as the total length grows. The model still attends to everything technically, but the gradient of attention over relevant vs. irrelevant tokens flattens.

The combined effect: an agent that was accurate and cheap at turn one becomes expensive and unreliable at turn thirty. The curve is not linear — it's a cliff. The figure below shows what that growth trajectory looks like without intervention:

![Context growth across 10 turns — no management leads to overflow](/imgs/blogs/context-window-management-2.webp)

There's a simpler framing that I use with engineers who haven't worked with long-context agents before: **a context window is not a database**. You cannot query it selectively. You cannot index it. Every token you put in is re-read, re-attended to, and re-billed on every subsequent call. The only way to manage cost and quality over a long session is to actively manage what stays in the window.

## 2. What Fills the Context Window: Anatomy of an Agent Context

Before we can manage a context window, we need to know what's in it. A production agent context typically contains six classes of content, and they have very different characteristics.

### System prompt

The system prompt is the foundational instruction set: who the agent is, what it can do, how it should behave, what format to use for outputs. Well-engineered system prompts for production assistants run from 2,000 to 12,000 tokens. They're written by engineers, not by the conversation — they change rarely (maybe once a day when you deploy a new version) and they're identical across all calls within a session.

This makes them excellent candidates for **prefix caching**, which we'll cover in strategy 6. For now, just note that the system prompt is a fixed cost per call that can be dramatically reduced with caching but cannot be compressed.

### Tool definitions

Every tool the agent can call requires a JSON schema: name, description, parameter types, and parameter descriptions. A well-documented tool takes 200–800 tokens to describe. An agent with 20 tools can easily use 8,000 tokens just for tool definitions, before the conversation starts. Like the system prompt, tool definitions are static within a session and prefix-cacheable.

### Conversation history

This is the primary pressure source. Every user turn and every assistant turn accumulates here. What makes history particularly dangerous is that tool calls are part of the history — and tool call results are often very large. A tool that returns a JSON payload of database query results can add 5,000–20,000 tokens in a single turn. An agent that makes ten tool calls in a session can easily add 30,000–100,000 tokens to history just from those results.

History is the slot that all six management strategies primarily target.

### Retrieved documents (RAG context)

If your agent uses retrieval-augmented generation, each turn may inject a new set of retrieved chunks. A typical retrieval setup returns 3–10 chunks of 200–1,000 tokens each, meaning 600–10,000 tokens per turn. These chunks are often redundant across turns (the same document retrieved multiple times), stale (retrieved for an earlier part of the conversation), or simply less relevant than what the agent now needs.

RAG context management is a sub-discipline of its own. The key insight is that retrieved chunks should not simply pile up in history — they should be evaluated for continued relevance on each turn and pruned aggressively.

### Scratchpad / chain-of-thought

Many agent frameworks give the model a "scratchpad" space to reason before committing to a tool call or a response. Chain-of-thought reasoning, planning, and self-verification all live here. A typical reasoning trace runs 200–1,000 tokens. Since scratchpad content is intermediate — it's the work, not the result — it can often be trimmed aggressively once the reasoning step is committed. Many frameworks strip scratchpad content before adding the turn to history.

### Response budget

The response budget is the space reserved for the model's output tokens. This is often overlooked in context accounting, but it's non-negotiable: if you fill the context window completely with input tokens, the model cannot generate any output. Production agents should reserve at least 512 tokens for short responses and 2,048–4,096 tokens for responses that may include tool calls with large parameter payloads.

A common production failure mode: the context budget accounting doesn't include the response budget, context fills to the model's hard limit, and the model either refuses to respond or silently truncates its output mid-sentence.

The figure below compares what an unmanaged context looks like at turn 15 versus a well-managed context:

![Unmanaged 128k tokens vs managed 8k tokens — 16x cost difference](/imgs/blogs/context-window-management-3.webp)

## 3. Strategy 1 — Sliding Window: Keep Last K Turns, Drop the Rest

The sliding window is the simplest possible context management strategy: keep only the most recent K conversation turns, discarding everything older. Its appeal is that it requires zero additional LLM calls, zero complexity, and is completely deterministic.

### How it works

Maintain a list of message dicts. Before constructing the context for each call, trim the message list to the last K items. Optionally, always keep the first turn (to preserve any task framing that happened there). Append the current user message.

```python
def sliding_window_messages(history: list[dict], k: int = 8) -> list[dict]:
    """Keep the last k turns plus any pinned messages."""
    if len(history) <= k:
        return history
    # Always keep turn 0 if it's a detailed task framing
    pinned = [history[0]] if history and history[0].get("pinned") else []
    recent = history[-k:]
    # De-duplicate if pinned is already in recent
    seen_ids = {m.get("id") for m in recent}
    pinned = [m for m in pinned if m.get("id") not in seen_ids]
    return pinned + recent
```

### When to use it

Sliding window is the right default for sessions where:

- The conversation is short to medium (under 20 turns)
- Each turn is mostly self-contained (the user is not building on specific things said 15 turns ago)
- You need the absolute minimum latency overhead (zero)
- You're doing early prototyping and want to defer complexity

It's the wrong choice when:

- Users refer back to early parts of the conversation ("as we discussed at the start, my budget is $500")
- The agent is executing a long multi-step task where early tool call results are still needed
- You're building a support agent where the issue description is in turn 1 and the diagnosis is still ongoing at turn 20

### The K selection problem

Choosing K is a balancing act. Too small (K=3) and you lose context required for basic coherence. Too large (K=15) and you lose most of the cost benefit. A practical approach: set K based on your average turn token size and your budget target. If the average turn is 800 tokens and your total history budget is 6,400 tokens, set K=8.

A smarter approach: track the running average token count per turn and adjust K dynamically. If turns are unusually dense (tool results, retrieved documents), shrink K automatically.

### Token counting matters

A naive implementation counts "turns" as the unit. A better implementation counts tokens. Two turns might be 200 tokens each, or one might be 12,000 tokens (a large tool result). Implement sliding window in token space, not turn count space:

```python
def sliding_window_by_tokens(
    history: list[dict],
    max_tokens: int,
    token_counter: Callable[[dict], int]
) -> list[dict]:
    """Keep as many recent turns as fit within max_tokens."""
    selected = []
    total = 0
    for msg in reversed(history):
        cost = token_counter(msg)
        if total + cost > max_tokens:
            break
        selected.insert(0, msg)
        total += cost
    return selected
```

### The real tradeoff

Sliding window achieves excellent token savings (60–80%) at zero latency cost and minimal implementation complexity (~20 lines). Its fatal flaw is irreversibility: once you drop a turn, it's gone. If the user mentions something critical at turn 2 that only becomes relevant at turn 22, a K=8 sliding window will have deleted it.

This is the fundamental tension that every more complex strategy attempts to resolve: can we drop tokens while preserving the information they carried?

## 4. Strategy 2 — Summarization: Compress Old Turns into a Summary

Summarization addresses the fundamental flaw of the sliding window: instead of dropping old turns entirely, we compress them into a summary that preserves their semantic content at a fraction of the token cost.

### How it works

Maintain a rolling summary. When the conversation history exceeds a threshold, invoke the LLM to summarize the oldest N turns and replace those turns with the summary. The summary is inserted as a system or assistant message with a marker like `[Conversation summary from turns 1–10]`.

```python
async def maybe_compress_history(
    history: list[dict],
    token_limit: int,
    summary_threshold: int,
    llm_client: Any,
    token_counter: Callable[[dict], int]
) -> list[dict]:
    """Summarize oldest turns if history exceeds threshold."""
    total_tokens = sum(token_counter(m) for m in history)
    if total_tokens < summary_threshold:
        return history
    
    # Find how many old turns to summarize
    # Keep at least the last 5 turns verbatim
    summarize_count = max(0, len(history) - 5)
    if summarize_count < 3:
        return history  # Not enough old turns to bother
    
    to_summarize = history[:summarize_count]
    to_keep = history[summarize_count:]
    
    # Build summarization prompt
    turns_text = "\n\n".join(
        f"[{m['role'].upper()}]: {m['content'][:2000]}"
        for m in to_summarize
    )
    summary_prompt = f"""Summarize the following conversation turns in 3-5 sentences. 
Focus on: (1) the user's original goal and any constraints they stated, 
(2) key facts established, (3) decisions made, (4) tool call results that are 
still relevant. Omit greetings, clarifications, and intermediate reasoning.

TURNS TO SUMMARIZE:
{turns_text}

SUMMARY:"""
    
    response = await llm_client.complete(summary_prompt, max_tokens=400)
    summary_msg = {
        "role": "system",
        "content": f"[Summary of turns 1–{summarize_count}]: {response.text}",
        "is_summary": True,
        "summarized_turn_count": summarize_count,
    }
    
    return [summary_msg] + to_keep
```

### Summarization quality

The quality of the summary is the critical variable. A bad summary loses information that the model cannot recover. A few things that determine summarization quality:

**What to include in the summarization prompt.** The default "summarize this conversation" prompt produces generic summaries that preserve sentiment but lose specifics. You want a structured prompt that extracts: (a) the user's stated goals and constraints, (b) key entities introduced (people, files, systems, numbers), (c) decisions committed to, (d) tool call results still relevant, and (e) any explicit user preferences expressed.

**Summarization model selection.** You don't need the full model to summarize. A cheaper, faster model (GPT-4o-mini, Claude Haiku) can produce excellent conversation summaries at 3–10x lower cost. The trade-off: the summary is generated at compression time, so you pay the summarization cost once and amortize it across all future turns.

**Lossy by construction.** Summarization is lossy. The model choosing what's important and what to omit is making a judgment call that may not align with what the future conversation needs. If the user asks "what was the exact JSON schema you showed me earlier?", a summary that only says "JSON schema was discussed" is useless. Use summarization for high-level context preservation, not verbatim recall.

### When to trigger summarization

Don't summarize too early (you lose the benefit of verbatim context when the conversation is still short) or too late (the context is already full and you have nowhere to fit the summary call). A good heuristic: trigger when total history tokens exceed 70% of the history budget. This leaves room for the summary message itself and several more verbatim turns.

### Latency cost

Summarization adds one full LLM call to the turn where it triggers. For a user-facing agent, this can add 200–600 ms to a single turn. Two mitigation strategies: (1) trigger summarization asynchronously while the current turn is being processed, completing before the *next* turn needs it; (2) trigger it during low-activity periods (immediately after a tool call result arrives, while the user is reading).

## 5. Strategy 3 — Token Budget Allocation: Assign Fixed Budgets per Slot

Token budget allocation treats the context window as a finite resource divided into explicitly managed partitions. Rather than letting any single slot grow unboundedly, each slot gets a fixed maximum.

### The partition map

A typical production partition for a 32k-token context:

| Slot | Budget | Notes |
|------|--------|-------|
| System prompt | 8,000 tok | Hard cap; if your prompt exceeds this, it needs to be shorter |
| Tool definitions | 6,000 tok | Hard cap; prune tool list to essential tools only |
| Conversation history | 10,000 tok | Managed via sliding window or summarization within this budget |
| RAG retrieved docs | 6,000 tok | Top-N chunks by relevance score within this budget |
| Scratchpad | 2,000 tok | Trimmed aggressively after each step |
| Response budget | 4,000 tok | Always reserved; never compressed |
| **Total** | **36,000 tok** | Intentional 4k headroom for response-size variance |

The discipline comes from treating these as hard limits enforced in code, not soft guidelines. When a slot exceeds its budget, the context manager trims it before building the final context payload.

### Implementation

```python
from dataclasses import dataclass, field
from typing import Optional
import tiktoken

@dataclass
class ContextBudget:
    system_prompt: int = 8000
    tools: int = 6000
    history: int = 10000
    retrieved_docs: int = 6000
    scratchpad: int = 2000
    response: int = 4000
    
    @property
    def total(self) -> int:
        return (self.system_prompt + self.tools + self.history + 
                self.retrieved_docs + self.scratchpad + self.response)

class BudgetEnforcingContextManager:
    def __init__(self, budget: ContextBudget, model: str = "gpt-4o"):
        self.budget = budget
        self.enc = tiktoken.encoding_for_model(model)
        self.history: list[dict] = []
        self.retrieved_docs: list[str] = []
    
    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def count_messages_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += self.count_tokens(m.get("content", ""))
            total += 4  # overhead per message (role, delimiters)
        return total
    
    def trim_history_to_budget(self) -> list[dict]:
        """Trim history from oldest first until under budget."""
        trimmed = list(self.history)
        while (self.count_messages_tokens(trimmed) > self.budget.history 
               and len(trimmed) > 1):
            # Never drop the first message if it's a task framing
            start = 1 if trimmed[0].get("is_task_framing") else 0
            if start >= len(trimmed):
                break
            trimmed.pop(start)
        return trimmed
    
    def trim_docs_to_budget(self, docs: list[str]) -> list[str]:
        """Keep top-k docs by relevance that fit in budget."""
        selected = []
        total = 0
        for doc in docs:  # Assumes docs pre-sorted by relevance score
            cost = self.count_tokens(doc)
            if total + cost > self.budget.retrieved_docs:
                break
            selected.append(doc)
            total += cost
        return selected
    
    def build_context(
        self,
        system_prompt: str,
        tool_definitions: list[dict],
        current_message: str,
        retrieved_docs: Optional[list[str]] = None,
    ) -> dict:
        """Build the full context payload within all budget constraints."""
        history = self.trim_history_to_budget()
        docs = self.trim_docs_to_budget(retrieved_docs or [])
        
        # Inject retrieved docs as a system message if any
        context_messages = []
        if docs:
            doc_text = "\n\n---\n\n".join(docs)
            context_messages.append({
                "role": "system",
                "content": f"Retrieved context:\n{doc_text}"
            })
        
        context_messages.extend(history)
        context_messages.append({"role": "user", "content": current_message})
        
        return {
            "system": system_prompt,
            "tools": tool_definitions,
            "messages": context_messages,
            "max_tokens": self.budget.response,
        }
    
    def add_turn(self, role: str, content: str):
        """Add a completed turn to history."""
        self.history.append({"role": role, "content": content})
```

### Why explicit budgets beat implicit hoping

Without explicit budgets, context overflow happens in unpredictable ways: sometimes the LLM API truncates silently, sometimes it raises a context length error mid-session, sometimes it completes but with degraded accuracy because critical early context was pushed to the lost-in-the-middle zone. Explicit budgets make the failure mode predictable and controllable: you decide what gets trimmed and when, rather than the API deciding for you at the worst possible moment.

A secondary benefit: explicit budgets make context costs auditable. You can log the token count per slot per turn and build observability dashboards that catch cost explosions before they become billing surprises.

## 6. Strategy 4 — Importance Scoring: Keep High-Value Messages, Drop Low-Value

Importance scoring is the first strategy that uses semantic understanding to decide what to keep. Rather than dropping the oldest messages or enforcing arbitrary slot budgets, we score each message for its likely future relevance and drop the lowest-scoring ones first.

### Scoring signals

A useful importance score combines several signals:

**Recency.** All else equal, recent turns are more likely to be relevant. Simple time-decay: `recency_score = 1 / (turn_age + 1)`.

**Message type.** Some message types are structurally more important:
- User messages containing explicit goals, constraints, or preferences: high
- Assistant messages that commit to a decision or action: high
- Tool call results (especially the first result for a tool): high
- Follow-up clarification exchanges: medium
- Greeting exchanges, "got it" acknowledgments: low

**Explicit signals.** Users often telegraph importance: "remember this for later", "this is the key constraint", "I need you to keep this in mind". A simple keyword scan can extract these.

**Tool call diversity.** If the same tool was called ten times with similar inputs, we only need one representative result in history. Deduplicate tool results by tool name and semantic similarity.

**Reference patterns.** If a message has been referenced by a subsequent assistant turn (the assistant said "as I mentioned earlier..." or "following from the constraint you gave..."), it's been surfaced as important by the model itself.

### A minimal scoring implementation

```python
import re
from dataclasses import dataclass

@dataclass
class MessageScore:
    message_id: str
    recency_score: float  # [0, 1] — 1 is most recent
    type_score: float     # [0, 1] — 1 is highest type priority
    signal_score: float   # [0, 1] — explicit importance signals
    total: float          # weighted sum

IMPORTANCE_SIGNALS = [
    r"remember this",
    r"keep this in mind",
    r"this is important",
    r"critical constraint",
    r"my budget is",
    r"my deadline is",
    r"don't forget",
]

def score_message(
    msg: dict,
    turn_index: int,
    total_turns: int,
) -> MessageScore:
    recency = (turn_index + 1) / total_turns  # newer = higher
    
    # Type scoring
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if msg.get("is_summary"):
        type_score = 0.9  # summaries are pre-distilled, high value
    elif role == "user":
        type_score = 0.7
    elif role == "tool":
        type_score = 0.6
    elif role == "assistant" and "I will" in content or "decided" in content.lower():
        type_score = 0.8  # decision commitment
    else:
        type_score = 0.4
    
    # Explicit signal scoring
    signal_score = 0.0
    content_lower = content.lower()
    for pattern in IMPORTANCE_SIGNALS:
        if re.search(pattern, content_lower):
            signal_score = min(1.0, signal_score + 0.3)
    
    total = 0.4 * recency + 0.4 * type_score + 0.2 * signal_score
    return MessageScore(
        message_id=msg.get("id", str(turn_index)),
        recency_score=recency,
        type_score=type_score,
        signal_score=signal_score,
        total=total,
    )

def importance_scored_trim(
    history: list[dict],
    max_tokens: int,
    token_counter: Callable,
    min_keep: int = 3,  # always keep at least this many recent turns
) -> list[dict]:
    if sum(token_counter(m) for m in history) <= max_tokens:
        return history
    
    n = len(history)
    scores = [
        (i, score_message(m, i, n))
        for i, m in enumerate(history)
    ]
    
    # Always keep the last min_keep turns regardless of score
    protected = {n - 1 - i for i in range(min(min_keep, n))}
    
    # Sort unprotected by score ascending (drop lowest first)
    trimmable = sorted(
        [(i, s) for i, s in scores if i not in protected],
        key=lambda x: x[1].total
    )
    
    drop_indices = set()
    current_tokens = sum(token_counter(m) for m in history)
    
    for idx, _score in trimmable:
        if current_tokens <= max_tokens:
            break
        current_tokens -= token_counter(history[idx])
        drop_indices.add(idx)
    
    return [m for i, m in enumerate(history) if i not in drop_indices]
```

### Latency and accuracy tradeoffs

Importance scoring adds 50–200 ms per trim operation depending on the number of messages and the complexity of the scoring function. It significantly outperforms pure sliding window on tasks where early-session information matters (goal recall, constraint recall), and it outperforms pure summarization on verbatim-recall tasks (what exactly did the user say about their deadline?).

The cost is implementation complexity: you need message IDs, a scoring function that actually discriminates well, and a minimum-keep floor to prevent pathological drops. It also requires testing — a buggy scoring function can silently drop critical messages with no runtime error.

## 7. Strategy 5 — Hierarchical Compression: Rolling Summaries + Detailed Recent

Hierarchical compression is the most sophisticated strategy: it maintains multiple tiers of context fidelity at the same time, like a cache hierarchy. The most recent turns are kept verbatim. Older turns are compressed into rolling summaries. The oldest turns are compressed into a high-level abstract.

![Hierarchical compression timeline — three tiers of fidelity](/imgs/blogs/context-window-management-7.webp)

### The three-tier model

**Tier 1: Recent (verbatim).** The last 3–5 turns are always kept in full. This ensures the model has maximum coherence with the immediate conversation state. When a new turn arrives and bumps a turn out of the recent tier, it gets absorbed into the rolling summary.

**Tier 2: Rolling summary.** A rolling paragraph-level summary of the last 5–20 turns (excluding the verbatim tier). When a turn leaves the recent tier, a mini-summarization call compresses it and updates the rolling summary incrementally. This is less expensive than full batch summarization because we're only processing one new turn at a time, not re-summarizing everything.

**Tier 3: Abstract.** A high-level abstract of the entire session up to some earlier checkpoint. Updated infrequently (every 20–30 turns), this captures the overarching goal, major decisions, and key constraints. It's the "executive summary" that gives the model session-level grounding.

### Incremental rolling summary update

The key to making this work efficiently is incremental updates. Rather than re-summarizing all mid-tier turns from scratch when a new turn arrives, we update the existing summary:

```python
async def update_rolling_summary(
    existing_summary: str,
    new_turn: dict,
    llm_client: Any,
    max_summary_tokens: int = 400,
) -> str:
    """Incrementally update the rolling summary with one new turn."""
    new_content = new_turn["content"][:1000]  # Truncate very long turns
    role = new_turn["role"]
    
    update_prompt = f"""You are maintaining a running summary of a conversation.

CURRENT SUMMARY (covers all prior turns):
{existing_summary}

NEW TURN TO INTEGRATE:
[{role.upper()}]: {new_content}

Update the summary to include the new turn. Keep the total summary under 
{max_summary_tokens // 4} words. Focus on: goals, constraints, decisions, 
key facts, and tool results that remain relevant. Drop any information 
from the prior summary that is now superseded or no longer relevant.

UPDATED SUMMARY:"""
    
    response = await llm_client.complete(update_prompt, max_tokens=max_summary_tokens)
    return response.text.strip()
```

### Cost analysis

A three-tier hierarchical system for a 100-turn conversation:

- Verbatim recent (5 turns): ~3,000 tokens
- Rolling summary: ~600 tokens
- Abstract: ~200 tokens
- Total context used for history: ~3,800 tokens

Versus unmanaged: ~80,000 tokens. That's a 21x reduction, with far better quality preservation than a sliding window of the same size.

The cost: 95 incremental summarization mini-calls over the course of the session (one per turn that ages out of the verbatim tier), each using a cheap fast model at perhaps 300–500 tokens input + 400 tokens output. At $0.15/million input tokens for a mini model, 95 calls × 900 tokens = 85,500 tokens = about $0.013 in extra LLM calls for the entire session. Well worth it for a 21x context reduction.

## 8. Strategy 6 — Prefix Caching: Cache Static System Prompt to Reduce Cost

Prefix caching is qualitatively different from the other five strategies: it doesn't reduce the logical information in the context, it reduces the cost and latency of transmitting the static prefix. Every major inference provider now supports it: Anthropic's prompt caching, OpenAI's automatic prompt caching on GPT-4o, and vLLM's prefix KV cache for self-hosted deployments.

![Prefix caching: full re-send vs cached delta per turn](/imgs/blogs/context-window-management-8.webp)

### How prefix caching works

The transformer's attention mechanism requires computing key-value (KV) pairs for every token in the input. For a 10,000-token system prompt that's sent on every call, the provider normally re-computes all 10,000 KV pairs from scratch. Prefix caching stores the computed KV pairs for the static prefix in memory. On subsequent calls, if the prefix is identical (same tokens, same position), the cached KV pairs are reused — no recomputation.

For the user, this means:
- **Cost reduction**: Cached tokens are billed at 10–50% of the normal input token rate. On Claude, cached tokens are $0.30/million vs $3.00/million for normal input — a 10x discount. On GPT-4o, cached tokens are $1.25/million vs $2.50/million.
- **Latency reduction**: The time-to-first-token (TTFT) drops significantly because the prefill computation for cached tokens is skipped. For a 10,000-token system prompt, this can save 100–400 ms of TTFT on a typical inference cluster.

### What qualifies as a cacheable prefix

The critical constraint: the cached prefix must be **identical** across calls — same tokens in the same order. This means:

- System prompts that include dynamic content (current date, user ID, session context) break prefix caching. Move all dynamic content *out of the system prompt* into the conversation messages.
- Tool definitions that are reordered or partially included break prefix caching. Always send all tools in a fixed order.
- Some providers require the prefix to be a minimum length (e.g., Anthropic requires ≥ 1,024 tokens for a cache write to be created).

### Implementation for Anthropic's prompt caching

```python
# Claude API prompt caching
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """..."""  # Your full, static system prompt

def call_with_caching(
    history: list[dict],
    user_message: str,
    tool_definitions: list[dict],
) -> str:
    # The cache_control block marks where the cacheable prefix ends.
    # Everything before this breakpoint gets cached.
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # Cache this block
            }
        ],
        tools=[
            {
                **tool,
                "cache_control": {"type": "ephemeral"},  # Cache tool defs too
            }
            for tool in tool_definitions
        ] if tool_definitions else [],
        messages=[
            *history,
            {"role": "user", "content": user_message},
        ],
    )
    
    # Check cache hit/miss in response metadata
    usage = response.usage
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Cache read tokens: {usage.cache_read_input_tokens}")
    print(f"Cache write tokens: {usage.cache_creation_input_tokens}")
    
    return response.content[0].text
```

### Prefix caching is orthogonal to the other strategies

Every other strategy reduces the number of tokens in the context. Prefix caching reduces the cost of the tokens that can't be removed. They compose: a well-managed context uses sliding window or hierarchical compression to manage history, and prefix caching to handle the unchangeable static prefix. Together, the cost reduction can be 85–95% compared to an unmanaged context with no caching.

## 9. Measuring Context Quality: What Metrics Tell You Your Context Is Degrading

You can't manage what you don't measure. Context quality degradation is subtle — the model doesn't throw an error when its effective context quality drops, it just gives worse answers. Here's what to instrument.

### Primary metrics

**Early-mention recall rate.** Design evaluation tasks where critical information is stated in turn 1 (user constraints, deadlines, budget) and tested at turn 20. Measure the fraction of time the model correctly uses that early information when it's directly relevant. A healthy agent should score above 85%. An unmanaged agent at turn 20 often scores 50–60%.

**Token cost per call (P50/P95).** Track input tokens per call over the session length. The P50 tells you your typical cost; the P95 tells you your outlier cost. An unmanaged agent's P95 will be 20–50x its P50 by turn 30. A well-managed agent's P95 should be at most 2–3x its P50.

**Response coherence score.** Track model outputs for references to "as I mentioned earlier" type phrases and verify they're referencing information that's actually in the current context window. A model that hallucinates a prior statement that was dropped from the window is a context quality failure.

**Latency (TTFT and E2E).** Context window size directly affects time-to-first-token on most inference backends. Monitoring TTFT over session length will show you context pressure in milliseconds before it shows up in billing.

### Secondary metrics

**Context utilization ratio.** `(history tokens) / (history budget)`. If this is consistently above 0.9, you're running hot and a spike will overflow. Target: 0.6–0.75.

**Summarization frequency.** How often does the summarization trigger per session? Too frequent (every 3 turns) suggests your summary model is generating long summaries or your verbatim budget is too small. Too infrequent (every 50 turns) suggests your threshold is set too high and you're paying for context you don't need.

**Tool result truncation rate.** What fraction of tool results are being truncated before insertion? A high truncation rate may indicate your tools are returning unnecessarily verbose payloads that could be pruned at the source.

## 10. The Lost-in-the-Middle Problem: Why Position Matters, Not Just Presence

This problem deserves its own section because it's counterintuitive. Most engineers understand that too many tokens is a problem. Fewer understand that *position* within the context window is as important as *presence*.

The empirical finding, from Liu et al. (2023) "Lost in the Middle: How Language Models Use Long Contexts," is stark: when you ask a language model to retrieve information from a long context, performance follows a U-shaped curve over document position. Information at the beginning of the context and at the end of the context is retrieved with high accuracy. Information in the middle — anywhere from 10% to 80% of the total context length — is retrieved with dramatically lower accuracy.

For a 20,000-token context, that means roughly 14,000 tokens of content have degraded retrieval accuracy just because of where they sit. This effect has been replicated across model sizes and model families.

![Lost-in-the-middle U-curve: accuracy vs position in context window](/imgs/blogs/context-window-management-6.webp)

### Why this happens

The technical explanation involves attention patterns. In autoregressive transformers, early tokens attend forward to later tokens (causally), creating a primacy effect where the model can't "forget" early content. Recent tokens have high attention weights from the current generation step (recency effect). But middle tokens are neither anchored to the output nor anchored to the beginning — they receive systematically lower attention weights in the layers that drive retrieval.

This is not a bug that will be fixed with scale. Larger models show the same U-shape, just with a higher baseline accuracy at the valley. The shape is structural.

### Practical implications for context management

**Never bury critical context in the middle.** The most dangerous thing you can do to a long-running agent is accumulate 50 turns of history and then put the user's original goal statement in position 10 of that 50. Position it first (always anchor the task framing at the top of the context) and most recently (repeat key constraints in summary form near the current turn).

**This is why hierarchical compression wins on quality.** A three-tier hierarchical context puts the abstract at the top (primacy zone), the rolling summary next (still primacy-adjacent), and the full verbatim recent turns at the bottom (recency zone). There is no middle. Every piece of information is either in the primacy zone or the recency zone, never buried in between.

**This is why sliding window can work better than it "should."** Even though a sliding window drops old information entirely, the information it *keeps* is concentrated in the recency zone where the model pays full attention. A 6,000-token sliding window can outperform a 20,000-token unmanaged context on recent-turn coherence, because the 6,000-token window is all recency zone.

**RAG position matters.** When you inject retrieved documents, put the most relevant one *first* in the injected context block, not last. Most RAG implementations order by relevance score descending and then inject in that order — leading to the most relevant document landing in the primacy zone. But if you're injecting multiple chunks, the second-most-relevant chunk may end up in the middle of a long injection block. Consider truncating to 2–3 chunks rather than 10 to keep all retrieved content in the primacy zone.

## 11. Implementation: Python Context Manager with Budget Enforcement

Here's the full implementation that ties together budget allocation, sliding window, importance scoring, and prefix caching into a single production-ready context manager (~80 lines of core logic):

```python
"""
context_manager.py — Production context manager for LLM agents.

Combines: token budget allocation, importance-scored history trimming,
tool result truncation, and prefix cache compatibility.
"""
from __future__ import annotations
import tiktoken
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class ContextBudgets:
    """Token budgets for each context slot."""
    system: int = 8_000      # Hard cap; set to None to skip enforcement
    tools: int = 6_000       # Hard cap
    history: int = 10_000    # Managed via importance-scored trimming
    retrieved: int = 6_000   # Top-N chunks by relevance within budget
    scratchpad: int = 2_000  # Trimmed aggressively post-step
    response: int = 4_000    # Always reserved; never compressed
    
    @property
    def total(self) -> int:
        return sum(filter(None, [
            self.system, self.tools, self.history,
            self.retrieved, self.scratchpad, self.response
        ]))

class ContextManager:
    def __init__(
        self,
        budgets: ContextBudgets,
        model: str = "gpt-4o",
        tool_result_max_tokens: int = 1_000,
    ):
        self.budgets = budgets
        self.enc = tiktoken.encoding_for_model(model)
        self.history: list[dict] = []
        self.tool_result_max = tool_result_max_tokens
    
    # ─── Token counting ────────────────────────────────────────────
    
    def count(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def count_msg(self, msg: dict) -> int:
        return self.count(msg.get("content", "")) + 4  # 4-token overhead
    
    def count_msgs(self, msgs: list[dict]) -> int:
        return sum(self.count_msg(m) for m in msgs)
    
    # ─── History management ─────────────────────────────────────────
    
    def add_turn(self, role: str, content: str, **meta):
        """Add a completed turn to history with metadata."""
        self.history.append({
            "role": role,
            "content": content,
            "turn_idx": len(self.history),
            **meta
        })
    
    def add_tool_result(self, tool_name: str, result: str):
        """Add a tool result, truncating if over budget."""
        truncated = self._truncate_tool_result(tool_name, result)
        self.add_turn("tool", truncated, tool_name=tool_name)
    
    def _truncate_tool_result(self, tool_name: str, result: str) -> str:
        tokens = self.count(result)
        if tokens <= self.tool_result_max:
            return result
        # Keep the first `tool_result_max` tokens worth of content
        encoded = self.enc.encode(result)[:self.tool_result_max]
        truncated = self.enc.decode(encoded)
        return truncated + f"\n\n[...truncated from {tokens} to {self.tool_result_max} tokens]"
    
    def _score_message(self, msg: dict, n: int) -> float:
        """Simple importance score [0, 1]. Higher = more important to keep."""
        idx = msg.get("turn_idx", 0)
        recency = (idx + 1) / max(n, 1)  # [0, 1], newer = higher
        
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Type-based score
        if msg.get("is_summary"):
            type_s = 0.95
        elif msg.get("is_task_framing"):
            type_s = 1.0   # Always keep
        elif role == "user":
            type_s = 0.7
        elif role == "assistant":
            type_s = 0.5
        elif role == "tool":
            type_s = 0.4
        else:
            type_s = 0.3
        
        # Explicit importance signals
        importance_keywords = [
            "remember", "keep in mind", "important", "critical",
            "must", "required", "deadline", "budget", "constraint",
        ]
        signal_s = 0.0
        lower = content.lower()
        for kw in importance_keywords:
            if kw in lower:
                signal_s = min(1.0, signal_s + 0.25)
        
        return 0.35 * recency + 0.45 * type_s + 0.20 * signal_s
    
    def _trim_history(self, min_keep: int = 3) -> list[dict]:
        """Importance-scored trim to fit within history budget."""
        msgs = list(self.history)
        if self.count_msgs(msgs) <= self.budgets.history:
            return msgs
        
        n = len(msgs)
        # Protect the last `min_keep` turns unconditionally
        protected = {n - 1 - i for i in range(min(min_keep, n))}
        
        scored = [
            (i, self._score_message(m, n))
            for i, m in enumerate(msgs)
            if i not in protected
        ]
        # Sort by score ascending so we drop lowest-value first
        scored.sort(key=lambda x: x[1])
        
        drop = set()
        current = self.count_msgs(msgs)
        for idx, _score in scored:
            if current <= self.budgets.history:
                break
            current -= self.count_msg(msgs[idx])
            drop.add(idx)
        
        return [m for i, m in enumerate(msgs) if i not in drop]
    
    # ─── Context assembly ───────────────────────────────────────────
    
    def build(
        self,
        user_message: str,
        retrieved_docs: Optional[list[str]] = None,
    ) -> list[dict]:
        """Build the trimmed message list for the next API call."""
        history = self._trim_history()
        
        messages: list[dict] = []
        
        # Inject retrieved docs above history (primacy zone)
        if retrieved_docs:
            doc_tokens = 0
            selected_docs = []
            for doc in retrieved_docs:  # Assumes sorted by relevance desc
                t = self.count(doc)
                if doc_tokens + t > self.budgets.retrieved:
                    break
                selected_docs.append(doc)
                doc_tokens += t
            if selected_docs:
                messages.append({
                    "role": "system",
                    "content": "Retrieved context:\n\n" + "\n\n---\n\n".join(selected_docs),
                })
        
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages
    
    def usage_report(self) -> dict:
        """Current token usage by slot."""
        return {
            "history_tokens": self.count_msgs(self.history),
            "history_budget": self.budgets.history,
            "history_utilization": self.count_msgs(self.history) / self.budgets.history,
            "turn_count": len(self.history),
        }
```

### Usage example

```python
# Initialize
budgets = ContextBudgets(history=8_000, retrieved=4_000, response=2_048)
ctx = ContextManager(budgets=budgets, model="gpt-4o")

# Add task framing as a pinned message
ctx.add_turn("user", "My budget is $500 and deadline is Friday. Help me plan.", 
             is_task_framing=True)
ctx.add_turn("assistant", "Got it — $500 budget, Friday deadline. Let's start...")

# Turn 10: Add a large tool result (will be truncated to 1000 tokens)
ctx.add_tool_result("database_query", large_json_payload)

# Build context for next call
messages = ctx.build(
    user_message="What were the results of the last query?",
    retrieved_docs=[doc1, doc2, doc3],  # Pre-sorted by relevance
)

# Check utilization
print(ctx.usage_report())
# {'history_tokens': 4821, 'history_budget': 8000, 'history_utilization': 0.60, 'turn_count': 12}
```

## 12. Case Studies: Context Mismanagement in Production

### Case study 1: The customer support bot that forgot the ticket

A B2B SaaS company deployed a customer support agent backed by GPT-4o. The agent opened each support ticket by reading the ticket description, then interacted with the customer over 20–30 turns to diagnose and resolve the issue. Initial quality metrics looked good.

After six weeks, the support team started flagging a pattern: customers who had long conversations were complaining that the agent "forgot" what their issue was. Investigation revealed the root cause: the agent's context management used a simple K=8 sliding window. By turn 20, the ticket description from turn 1 had been completely evicted from context. The agent was reasoning purely from the last 8 turns — helpful for immediate back-and-forth but blind to the original problem statement.

The fix: mark the ticket description message with `is_task_framing=True` and exempt it from sliding window eviction. A second fix: inject a compressed version of the ticket description into the system prompt as a permanent anchor, placing it in the primacy zone where the model reliably attends to it. This reduced "forgot the original issue" complaints by 94% and required only a day of engineering work.

**Lesson:** Always protect task-framing messages from eviction. The user's original goal is the most important piece of information in the entire session.

### Case study 2: The coding agent with 50k-token tool results

A developer productivity startup shipped a coding agent that could execute shell commands, read files, and run tests. Their early agents worked beautifully on small codebases. When they expanded to large monorepos, something broke: the agent would work well for a few turns, then start making contradictory changes, ignoring prior decisions, and hallucinating file contents.

The culprit was tool result size. When the agent executed `find . -name "*.py" | head -200`, it received a list of 200 Python file paths. When it then read a large Python file, it could receive 10,000–30,000 tokens of source code. By turn 5, the context was full, and the provider was silently truncating the middle of the conversation to fit within the 128k limit.

The fix had two parts: (1) add a hard `tool_result_max_tokens=2000` cap that truncated all tool results before inserting them into history, with a clear annotation showing the truncation; (2) implement result summarization for `read_file` results — extract function signatures, class definitions, and docstrings, then summarize inline function bodies. This reduced typical `read_file` results from 8,000 tokens to 400 tokens with no loss in the agent's ability to reason about the code structure.

**Lesson:** Tool results are the most dangerous source of context explosion. Instrument their sizes. Set hard caps. For large structured results, summarize rather than truncate blindly.

### Case study 3: The research agent that degraded at turn 30

A research assistant agent was designed for long, open-ended research sessions: the user provides a research question, and the agent explores it over many turns, using web search and document retrieval. The agent worked well in testing (which used sessions of 10–15 turns) but degraded in production where users ran 40–60-turn sessions.

The degradation was subtle: the agent would correctly summarize recent findings, but would contradict positions it had taken earlier in the session ("actually, studies show X" when it had said exactly the opposite at turn 8). This inconsistency was driving users away from the product.

Root cause: the agent used summarization for context management, but the summarization prompt was generic ("summarize this conversation"). The generic summary captured what was *said*, not what was *established*. By turn 30, the rolling summary had conflicting claims coexisting because the summarizer didn't resolve contradictions — it just reported both.

The fix: redesign the summarization prompt to produce a structured "session state" rather than a narrative summary:

```
RESEARCH SESSION STATE:
- Question: [original research question]
- Established facts (high confidence): [bullet list]
- Tentative findings (medium confidence): [bullet list]  
- Contradictions/uncertainties: [bullet list]
- Sources consulted: [list with conclusions]
- Next steps planned: [bullet list]
```

This structured summary forced the LLM to resolve contradictions explicitly during summarization rather than passing them through. Consistency scores recovered to near-baseline performance even at turn 50.

**Lesson:** Generic summarization prompts pass through contradictions and noise. For knowledge-building tasks, use structured state summaries that force explicit conflict resolution.

### Case study 4: The pipeline that paid $40,000/month for prefix tokens

A multi-agent automation pipeline at a mid-size company used GPT-4o with a detailed 12,000-token system prompt describing the agent's persona, tools, and operational procedures. The system prompt was carefully crafted and stable — it was updated at most once per week.

The team had implemented context management for history (hierarchical compression, well-tuned), but hadn't noticed one thing: OpenAI's automatic prefix caching requires the cacheable prefix to be exactly the same. They were injecting a `{current_datetime}` variable into their system prompt at the start of every turn. Because the datetime changed every second, no two calls had the same prefix, and caching never fired.

This wasn't visible in any quality metric — the agent worked perfectly. But when they ran a billing analysis, they found that 40% of their monthly LLM bill ($40K out of $100K) was attributable to the 12,000-token system prompt being re-billed on every single call.

The fix: move the datetime out of the system prompt into the first user message. The system prompt became static; prefix caching activated immediately. Within one billing cycle, the system prompt cost dropped to near zero ($300 that month vs. $40,000 prior).

**Lesson:** Audit your system prompts for dynamic content that breaks prefix caching. Even a single timestamp injection can cost you tens of thousands of dollars per month.

### Case study 5: The analyst agent that buried its own findings

A financial analysis agent built on Claude was used to research investment opportunities. The agent would spend 15–20 turns gathering data (news, filings, financial statements) via tool calls, then synthesize a report.

Users reported that the final synthesis was often inconsistent with data the agent had collected earlier. In one memorable example, the agent collected a company's Q3 earnings (showing a significant miss) at turn 8, then at turn 25 wrote "earnings have been strong" in its synthesis.

The diagnosis: the agent used a sliding window with K=12. By turn 25, the Q3 earnings tool result from turn 8 had been evicted. The synthesis was based only on the most recent 12 turns of data — and the most recent turns happened to contain more positive news.

The fix: implement importance scoring with a domain-specific rule for financial agents — any tool result from a financial data source gets a high importance score and is never evicted until the session ends. A simple rule: if `msg.role == "tool" and msg.tool_name in FINANCIAL_DATA_TOOLS: score = 1.0`.

The more general lesson: domain context determines which message types are critical. A financial agent's important messages are tool results. A writing assistant's important messages are user style preferences. Build your importance scoring with your domain in mind.

**Lesson:** Generic importance scoring misses domain-specific critical information. Build domain knowledge into your scoring function.

### Case study 6: The vector database query agent with redundant RAG

A developer-facing documentation agent used RAG to retrieve relevant documentation chunks on each turn. The agent was using a simple "always retrieve top 10 chunks per turn and inject them all" strategy, which meant each turn added 3,000–5,000 tokens of retrieved content to the context.

By turn 5, the context contained 15,000–25,000 tokens of retrieved documentation — much of it the same sections retrieved multiple times (the same core API docs appeared in nearly every query). The agent was spending 70% of its context budget on redundant retrieved content.

Two fixes were applied. First: deduplicate retrieved chunks by chunk ID before injection — if a chunk was already injected in a recent turn, don't inject it again. Second: reduce from top-10 to top-3 chunks per turn, accepting a slight accuracy reduction in exchange for an 8x reduction in RAG context cost. The accuracy reduction turned out to be negligible because the top-3 chunks captured the relevant content the vast majority of the time.

**Lesson:** RAG context accumulates duplicates aggressively. Deduplicate chunk injection and be conservative with k — you rarely need more than 3–5 chunks for well-formed queries.

### Case study 7: The customer onboarding agent that was too good at forgetting

A fintech company built a customer onboarding agent that helped users fill out account opening forms. The agent used aggressive context management — hierarchical compression with a 5,000-token total budget — which worked well for most onboarding sessions.

However, the compliance team flagged an issue: the agent was occasionally asking users for information they had already provided. In one case, a user entered their Social Security Number in turn 3, the SSN was compressed into a rolling summary by turn 15, and by turn 22 the agent asked for it again.

This is the over-compression failure mode: the rolling summary said "user's identity verified" without capturing the actual SSN (which it shouldn't, for security reasons), but also without marking the entity as "already collected." The agent didn't know whether it needed to collect the SSN or not.

The fix: implement an explicit "form state" data structure separate from the context window. Every piece of structured data collected during onboarding (SSN, address, employment status, etc.) was stored in a dedicated state object and injected as a compact system message on every turn: "COLLECTED SO FAR: {state}". This state message was exempt from context trimming and always present in the primacy zone.

**Lesson:** For task-completion agents, structured state lives outside the context window. Don't rely on the model to infer what's been done from compressed history. Maintain a separate state object and inject it explicitly.

### Case study 8: The multi-session agent with stale context

An enterprise CRM integration agent was designed to help sales representatives prepare for customer calls. Before each call, the agent would load the recent history from the CRM (last 3 interactions), the account details, and then allow the rep to ask questions.

The system was working well until a bug was discovered: sessions from different customers were sharing context. Due to a session management bug, the context from one customer's session (rep A talking to customer X) was being loaded at the start of a different customer's session (rep B talking to customer Y). Because both sessions used the same context ID, the cached context was reused.

The agent would confidently say things like "as we discussed last time, their primary pain point is X" about the wrong customer. The bug was in the session management layer, not the context management layer — but it was enabled by the implicit assumption that context was always fresh and customer-specific.

**Lesson:** In multi-tenant agent systems, always include an explicit session/customer identifier in the context and validate it before each call. Never reuse context across customer boundaries. Make the session ID part of the prefix so that cache mismatches are caught early.

## 13. When to Use Each Strategy

The decision isn't which strategy is best — it's which strategy fits your agent's conversation shape, quality requirements, and operational constraints.

![Context strategy decision flow — pick based on session length and quality needs](/imgs/blogs/context-window-management-4.webp)

![Strategy comparison matrix — six strategies across five dimensions](/imgs/blogs/context-window-management-5.webp)

![Context pruning decision tree — keep, summarize, or drop per message](/imgs/blogs/context-window-management-9.webp)

![Agent profiles and recommended strategies by archetype](/imgs/blogs/context-window-management-10.webp)

### Decision framework

**Use sliding window when:** your sessions are short (< 20 turns), turns are mostly self-contained, and you need minimum implementation overhead. It's the right default for quick prototyping. Upgrade when you start seeing "the agent forgot" complaints.

**Use summarization when:** sessions are medium-length (20–50 turns), quality is more important than latency, and the conversation doesn't require verbatim recall of specific early statements. Pair it with a structured summarization prompt, not a generic one.

**Use token budget allocation when:** you have well-defined slot boundaries and want deterministic, auditable context accounting. It's the foundation layer — combine it with sliding window or summarization within the history slot for best results.

**Use importance scoring when:** your agent handles domain-specific high-value messages (financial data, user constraints, task objectives) that must survive aggressive history trimming. You need to define what "important" means in your domain. Don't use it as a black box.

**Use hierarchical compression when:** sessions are long (50+ turns), quality is critical (research agents, task automation), and you can afford the extra LLM calls for incremental summarization. It's the gold standard for quality-preserving context management at high turn counts.

**Use prefix caching when:** you have a large, stable system prompt and/or tool definitions. It's the highest ROI optimization in this list — often a simple API change that immediately cuts your input token bill by 30–60%. There's almost no reason not to use it.

**Stack strategies.** These are not mutually exclusive. The right production setup for most agents:

1. Prefix caching for the static prefix (always)
2. Token budget allocation as the accounting layer (always)
3. Sliding window or summarization within the history budget (choose based on quality needs)
4. Importance scoring on top of sliding window for domain-critical messages (if needed)
5. Hierarchical compression for long-running high-stakes agents (if session length demands it)

The diagram below shows what each agent archetype maps to:

The patterns summarized:

| Session Type | Turn Count | Primary Strategy | Secondary | Anti-pattern to Avoid |
|---|---|---|---|---|
| Stateless QA | 1–3 | Token budgets | Prefix cache | Over-engineering |
| Customer support | 5–20 | Sliding window K=8 | Importance scoring | No management at all |
| Code assistant | 10–40 | Summarization + RAG dedup | Prefix cache | Raw tool results |
| Research agent | 20–100 | Hierarchical compression | Token budget | Sliding window alone |
| Task automation | 50–500 | Hierarchical + external memory | Vector store offload | Any in-window-only approach |
| Data analysis | 10–50 | Importance scoring (keep results) | Summarize reasoning | Dropping tool results |
| Long-context RAG | 1–5 deep | RAG re-ranking + dedup | Budget (docs > history) | 10-chunk injection |

### When NOT to manage context

There are two scenarios where active context management is counterproductive:

**Short sessions that fit naturally.** If your p99 session is 15 turns and your model has 32k context, you fit comfortably without management. Adding complexity for no benefit is its own failure mode.

**Summarization for verbatim-recall tasks.** If your agent's primary job is to help users draft, review, and revise specific text (legal contracts, code snippets, documentation), lossy summarization of prior drafts is dangerous. Use structured state storage for the versioned artifacts instead, keep the context lean, and let users explicitly reference which version they want to continue from.

---

Context management is not optional for production agents. It's as fundamental as error handling: you can build a prototype without it, but you cannot ship a reliable, economical product without it. The strategies in this post are incremental — start with prefix caching and token budgets, add sliding window, graduate to summarization and hierarchical compression as your sessions grow longer. Every increment reduces cost and improves the accuracy of information that matters.

The six strategies are a toolkit, not a ladder. Pick the ones that match your agent's conversation shape. Measure. Iterate. Keep the window sharp.

---

*Related reading:*
- [Agent Memory Taxonomy](/blog/machine-learning/ai-agent/agent-memory-taxonomy) — how context window management fits into the broader memory architecture of autonomous agents
- [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) — a complementary perspective on what to put in the context, not just how to manage it
- [Agent Loop Anatomy](/blog/machine-learning/ai-agent/agent-loop-anatomy) — the full request/response/tool cycle that context management slots into

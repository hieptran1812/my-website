---
title: "Cost Optimization for Agents: Cutting LLM Spend Without Cutting Capability"
date: "2026-06-27"
description: "Practical techniques for reducing AI agent costs — model routing, prompt compression, caching, batching, memory efficiency, and the measurement framework to know if optimizations actually work."
tags: ["ai-agents", "cost-optimization", "llm", "production-ml", "machine-learning", "mlops", "token-efficiency"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

You built an agent that works. Users love it. Then the invoice arrives.

A single-turn GPT-4o call costs roughly $0.003 per 1,000 input tokens. A 10-step research agent, each step carrying a 3,000-token context, burns $0.09 before it finishes its first task. At 10,000 tasks per day you are paying $900 daily — $27,000 a month — for one agent workflow. Add memory retrieval, tool calls, retries, and the inevitable "let me think step by step" output verbosity, and the real number is often 3–5× that estimate.

This is the agent cost problem in a sentence: agents are programs that call LLMs in a loop, and each iteration of that loop multiplies the baseline cost. Optimizing a single LLM call gets you nowhere if you do it without fixing the loop.

The diagram below is the mental model: every agent task decomposes into a stack of cost layers, each with its own optimization lever. You need to work all six layers simultaneously.

![Agent cost anatomy showing six cost layers from input tokens to compute](/imgs/blogs/cost-optimization-for-agents-1.webp)

The good news is that the optimization space is well-mapped. Teams that systematically apply model routing, prompt compression, caching, and memory tiering consistently achieve 80–90% cost reduction without meaningful quality loss. The bad news is that most teams apply one technique at random, declare victory at 20% savings, and then watch costs creep back up every time the product evolves.

This post is the complete playbook. We start with the cost anatomy, work through each optimization layer, build the measurement framework to know whether things are actually improving, and close with case studies of both wins and regressions.

## 1. The Agent Cost Problem: Why Agents Cost 10–100× More Than Single Calls

Before we optimize, we need to be precise about *why* agents are expensive. There are four structural reasons.

**The loop multiplier.** Every tool-use step in a ReAct-style agent is a full LLM call. A 10-step workflow makes 10 calls. But it is worse than a 10× multiplier, because each call carries the *entire conversation history* in its input — so the input tokens grow at each step. The first step might have 1,200 input tokens. By step 10, after injecting tool results, reasoning traces, and prior turns, you might be at 8,000 input tokens. The total input across 10 steps is often 40–50k tokens, not 10×1,200 = 12k.

**Output token inflation.** Output tokens cost 3–5× more than input tokens on most models (GPT-4o: $2.50/1M input, $10/1M output). Agents are verbose by design — chain-of-thought, tool-use schemas, intermediate reasoning, and formatted final answers. A 10-step agent that produces 300 output tokens per step generates 3,000 output tokens total: $0.030 in output cost alone, before input.

**Tool call overhead.** Every function call adds a round-trip: the model serializes the call as output tokens (JSON), the result is injected as input tokens on the next turn. A single search tool call typically adds 200–500 tokens of overhead. Five tool calls per task add 1,000–2,500 tokens of overhead — roughly 10–25% of total token cost in a typical research agent.

**Retry amplification.** When tools fail, agents retry. When the model produces malformed output, the orchestrator retries. When a guardrail fires, the agent reruns the step with a corrected prompt. Each retry doubles the cost of that step. A retry rate of 15% (realistic for tools that call external APIs) amplifies the mean cost of every step by 15%.

The cumulative effect: a well-designed 10-step agent typically costs 20–40× a single comparable LLM call. A poorly-designed one — long system prompt, all tools loaded, raw history, no caching — can hit 100×.

### The Production Reality Check

Here is a concrete number to calibrate against. In 2024, production teams shipping ReAct-style research agents reported median costs of $0.08–$0.25 per task completion for GPT-4-class models. At 10k tasks/day, that is $800–$2,500/day or $24k–$75k/month. For teams that did not apply systematic optimization, costs doubled every time they added a new tool or extended the agent's scope.

The benchmark you want to target: $0.01–$0.05 per successful task for a medium-complexity research or coding agent using GPT-4o-class quality. That is achievable with the stack in this post.

## 2. Cost Anatomy: Decomposing the Six Layers

Understanding where each dollar goes is prerequisite to cutting it. We can decompose agent cost into six layers with different characteristics.

The six layers, roughly in order of magnitude for a typical research agent:

**Layer 1: Input tokens (40–60% of cost).** The dominant cost driver. Every step carries: system prompt, prior conversation history, tool schemas for all available tools, retrieved context from memory or RAG, and the current user request. The system prompt alone is often 800–1,500 tokens. Load 12 tool schemas at ~200 tokens each and you have added 2,400 tokens before a single user word is processed.

**Layer 2: Output tokens (15–25% of cost).** Priced at 3–5× input tokens, output cost is highly dependent on model verbosity. Chain-of-thought models (o1, o3, Claude 3.7 Sonnet "extended thinking") can produce 500–2,000 tokens of reasoning before the actual answer. Every word of that reasoning costs money.

**Layer 3: Tool call overhead (10–20% of cost).** Function call serialization (output tokens), result injection (input tokens next turn), and particularly retries. On tool-heavy agents (search + code execution + database), this layer can exceed 20%.

**Layer 4: Embedding calls (5–15% of cost).** RAG retrieval, semantic cache lookups, and memory indexing each require embedding calls. text-embedding-3-small costs $0.02/1M tokens — cheap per call, but adds up at 3–5 embedding calls per agent step. At 10k tasks/day with 5 embeddings of 500 tokens each: 10k × 5 × 500 = 25M tokens/day = $0.50/day — negligible. But for heavier RAG setups with larger documents, this can reach 15% of total cost.

**Layer 5: Storage and vector DB (5–10% of cost).** Chroma, Weaviate, Pinecone, pgvector — each query has a compute cost. For high-frequency agents, vector DB costs are small but non-zero. The bigger issue is write cost: ingesting agent outputs and memories into a vector store adds latency and cost that teams forget to account for in their per-task cost estimates.

**Layer 6: Compute and infrastructure (5–10% of cost).** Container runtime, orchestration overhead, cold-start latency, and function compute. Often underestimated because it comes from a different budget line than LLM API spend.

### The 80/20 of Cost Cutting

Given this anatomy, the leverage points are clear:

1. Input tokens: cut the context aggressively, load tools selectively
2. Model pricing: use a cheaper model for simple steps
3. Cache hits: serve frequent requests without any LLM call at all

These three alone can cut 80% of total cost. Everything else is marginal optimization.

## 3. Model Routing: Using Cheaper Models for Simple Steps

Model routing is the single highest-leverage optimization in the agent cost toolkit. The core insight: not all agent steps need the same model. Summarizing a tool result is simpler than generating a SQL query. Extracting named entities from retrieved text is simpler than planning a multi-step research strategy.

![Model routing flow showing complexity classifier directing 70% of steps to cheap models](/imgs/blogs/cost-optimization-for-agents-2.webp)

The math makes the case directly. GPT-4o costs $2.50/1M input tokens. GPT-4o-mini costs $0.15/1M input tokens — a 16.7× cost difference. If 70% of your agent steps are genuinely "simple" (reformatting, extraction, summarization, simple Q&A), and you route those steps to GPT-4o-mini, your blended cost per million input tokens drops from $2.50 to:

```
0.70 × $0.15 + 0.20 × $2.50 + 0.10 × $15.00 = $0.105 + $0.50 + $1.50 = $2.11/1M
```

But only if the routing is to a frontier model for the hard 10%. If your hardest tasks go to $2.50/1M models, the blended rate is $0.105 + $0.50 + $0.25 = $0.855/1M — a 66% cost reduction.

### Building a Complexity Classifier

The classifier itself needs to be fast and cheap. Three approaches in order of engineering effort:

**Rule-based routing (1 day to implement).** Define step types in your agent's state machine. "Extract entities from text" → cheap model. "Write and debug Python code" → mid-tier. "Reason over conflicting information" → expensive. Works well for structured workflows but breaks on novel task types.

```python
from enum import Enum

class StepComplexity(Enum):
    SIMPLE = "simple"    # GPT-4o-mini
    MEDIUM = "medium"    # GPT-4o
    HARD = "hard"        # GPT-4.1 / o3

STEP_TYPE_TO_COMPLEXITY = {
    "summarize": StepComplexity.SIMPLE,
    "extract_entities": StepComplexity.SIMPLE,
    "reformat": StepComplexity.SIMPLE,
    "answer_factual": StepComplexity.MEDIUM,
    "write_code": StepComplexity.MEDIUM,
    "debug_complex": StepComplexity.HARD,
    "plan_multi_step": StepComplexity.HARD,
    "reason_conflict": StepComplexity.HARD,
}

MODEL_FOR_COMPLEXITY = {
    StepComplexity.SIMPLE: "gpt-4o-mini",
    StepComplexity.MEDIUM: "gpt-4o",
    StepComplexity.HARD: "gpt-4.1",
}

def route_step(step_type: str) -> str:
    complexity = STEP_TYPE_TO_COMPLEXITY.get(step_type, StepComplexity.MEDIUM)
    return MODEL_FOR_COMPLEXITY[complexity]
```

**Heuristic scoring (2–3 days).** Score each step on multiple signals: prompt length (longer → harder), number of constraints in the instruction, whether the step requires tool use, and whether the output will be inspected by a human or only by the next agent step. Weighted sum gives a complexity score.

```python
def score_step_complexity(
    prompt: str,
    requires_tool_use: bool,
    output_inspected_by_human: bool,
    max_retries_allowed: int,
) -> StepComplexity:
    score = 0
    
    # Prompt length heuristic
    token_estimate = len(prompt.split()) * 1.3
    if token_estimate > 2000:
        score += 2
    elif token_estimate > 500:
        score += 1
    
    # Constraint density: count imperative verbs and "must"/"should" clauses
    constraint_words = ["must", "should", "exactly", "precisely", "ensure", "verify"]
    constraints = sum(1 for w in constraint_words if w in prompt.lower())
    score += min(constraints, 3)
    
    # Tool use and human inspection
    if requires_tool_use:
        score += 1
    if output_inspected_by_human:
        score += 2
    
    if score <= 2:
        return StepComplexity.SIMPLE
    elif score <= 5:
        return StepComplexity.MEDIUM
    else:
        return StepComplexity.HARD
```

**LLM-as-classifier (1 week but highest accuracy).** Use a cheap model (GPT-4o-mini at $0.15/1M) to classify each step before routing. The classification call costs ~$0.0001 and breaks even after routing a single step correctly. The classifier prompt:

```python
CLASSIFIER_PROMPT = """You are a task complexity classifier for an AI agent system.
Given a task description, classify it as SIMPLE, MEDIUM, or HARD.

SIMPLE: Summarization, formatting, entity extraction, straightforward Q&A with clear answer
MEDIUM: Code writing, moderate reasoning, multi-step instructions without ambiguity  
HARD: Complex reasoning over conflicting information, multi-step planning, debugging hard bugs

Task: {task_description}

Respond with exactly one word: SIMPLE, MEDIUM, or HARD"""

async def classify_step(task: str, client) -> StepComplexity:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # Use cheap model for classification
        messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(task_description=task)}],
        max_tokens=10,
        temperature=0,
    )
    label = response.choices[0].message.content.strip().upper()
    return StepComplexity[label] if label in StepComplexity.__members__ else StepComplexity.MEDIUM
```

### Fallback Escalation

Start with a cheap model and escalate on failure. This pattern is safer than pre-classification when you have a new task type:

```python
async def execute_with_escalation(
    prompt: str,
    client,
    quality_check_fn,
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
) -> str:
    for model in models:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.choices[0].message.content
        
        if quality_check_fn(result):
            return result
        
        # Log the escalation for future classifier training
        logger.info(f"Model {model} failed quality check, escalating to next tier")
    
    return result  # Return best attempt from most expensive model
```

### Routing in Practice: What Step Types Look Like

In a typical research-and-summarize agent, we found the following distribution after auditing 5,000 production tasks:

| Step type | Typical frequency | Recommended tier | Cost vs. naive |
|---|---|---|---|
| Summarize tool result | 35% | Mini | −94% |
| Extract key facts | 20% | Mini | −94% |
| Reformat for next step | 10% | Mini | −94% |
| Answer mid-complexity Q | 15% | Mid | −83% |
| Write/debug code | 12% | Mid | −83% |
| Plan complex strategy | 5% | Frontier | 0% |
| Reason over ambiguity | 3% | Frontier | 0% |

Weighted average cost reduction: ~75% for this workflow. Your distribution will vary, but the principle holds: most steps are genuinely simple.

## 4. Prompt Compression: Removing Redundant Context

Even after routing, the input token count for each step is still the dominant cost driver. The next attack surface is the context itself: how much of what's in the prompt is actually needed for this step to succeed?

![Prompt compression before vs after showing 81% token reduction](/imgs/blogs/cost-optimization-for-agents-3.webp)

The typical uncompressed agent context looks like this for step N of a 10-step task:

- Full system prompt with all instructions: 1,200 tokens
- All 12 tool schemas (many never used on this step): 2,800 tokens
- Raw conversation history for turns 1 through N-1: 4,500 tokens
- Current tool result (raw JSON): 1,700 tokens
- Total: ~10,200 tokens → $0.031/step

The compressed equivalent:

- System prompt (keep only the core mission and step-relevant instructions): 600 tokens
- Only the 3 tools relevant to this step's type: 700 tokens
- History summary for turns 1 through N-1: 400 tokens
- Tool result (key fields extracted, formatted as plain text): 100 tokens
- Total: ~1,800 tokens → $0.005/step — an 82% reduction

### System Prompt Compression

Most system prompts contain sections that are only relevant for specific step types: the section on "how to write SQL queries" is irrelevant when the agent is in a summarization step. Use structured system prompts with sections and inject only the relevant sections per step type:

```python
SYSTEM_PROMPT_SECTIONS = {
    "core": """You are a research agent. Your goal is to answer user questions 
    accurately using available tools. Always cite sources.""",
    
    "sql_query": """When writing SQL queries:
    - Use parameterized queries only
    - Explain the query in a comment above it
    - Return at most 100 rows unless asked otherwise""",
    
    "code_execution": """When writing and executing code:
    - Write idiomatic Python
    - Handle exceptions explicitly  
    - Use type hints in function signatures""",
    
    "summarization": """When summarizing:
    - Lead with the key finding
    - Use bullet points for lists of facts
    - Keep to under 200 words unless asked for more""",
}

def build_system_prompt(step_type: str) -> str:
    sections = [SYSTEM_PROMPT_SECTIONS["core"]]
    
    if step_type in SYSTEM_PROMPT_SECTIONS:
        sections.append(SYSTEM_PROMPT_SECTIONS[step_type])
    
    return "\n\n".join(sections)
```

### Tool Schema Gating

Loading all tool schemas every step is the most common prompt waste. Instead, maintain a "tool relevance" mapping and only include schemas for tools that could plausibly be called in the current step:

```python
STEP_TYPE_TO_RELEVANT_TOOLS = {
    "research": ["web_search", "retrieve_document", "get_citations"],
    "code": ["python_executor", "file_read", "file_write"],
    "data_analysis": ["sql_query", "python_executor", "get_chart"],
    "summarize": [],  # No tools needed for pure summarization
}

def build_tool_schemas(step_type: str, all_tools: list[dict]) -> list[dict]:
    relevant_names = STEP_TYPE_TO_RELEVANT_TOOLS.get(step_type, [t["name"] for t in all_tools])
    return [t for t in all_tools if t["name"] in relevant_names]
```

For a 12-tool agent with 3 tools per step type on average, this cuts tool-schema tokens by 75%.

### History Summarization

Raw conversation history is the most token-expensive part of a long-running agent context. The fix is to compress old turns into a rolling summary when the history exceeds a token budget.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConversationMemory:
    rolling_summary: Optional[str] = None
    summary_token_count: int = 0
    recent_turns: list[dict] = None  # Keep last N turns verbatim
    key_facts: list[str] = None  # Extracted important facts
    
    def __post_init__(self):
        if self.recent_turns is None:
            self.recent_turns = []
        if self.key_facts is None:
            self.key_facts = []

async def compress_history(
    history: list[dict],
    token_budget: int,
    client,
    keep_last_n: int = 3,
) -> ConversationMemory:
    """Compress old history into a summary, keeping recent turns verbatim."""
    
    if len(history) <= keep_last_n:
        return ConversationMemory(recent_turns=history)
    
    # Split: old turns to summarize, recent turns to keep
    old_turns = history[:-keep_last_n]
    recent_turns = history[-keep_last_n:]
    
    # Extract key facts first (faster, cheaper than full summary)
    facts_prompt = f"""Extract the 5 most important facts from this conversation so far.
    Format as a JSON array of strings. Be concise.
    
    Conversation:
    {format_turns(old_turns)}"""
    
    facts_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": facts_prompt}],
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    
    import json
    key_facts = json.loads(facts_response.choices[0].message.content).get("facts", [])
    
    # Create summary
    summary_prompt = f"""Summarize this conversation history in 2-3 sentences, 
    focusing on what the agent learned and what was decided.
    
    Conversation:
    {format_turns(old_turns)}"""
    
    summary_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=200,
    )
    
    return ConversationMemory(
        rolling_summary=summary_response.choices[0].message.content,
        recent_turns=recent_turns,
        key_facts=key_facts,
    )

def build_context_from_memory(memory: ConversationMemory) -> list[dict]:
    """Build a compact context from compressed memory."""
    messages = []
    
    if memory.rolling_summary or memory.key_facts:
        context_msg = "Context from earlier in this conversation:\n"
        if memory.rolling_summary:
            context_msg += f"Summary: {memory.rolling_summary}\n"
        if memory.key_facts:
            context_msg += "Key facts established:\n" + "\n".join(f"- {f}" for f in memory.key_facts)
        
        messages.append({"role": "system", "content": context_msg})
    
    messages.extend(memory.recent_turns)
    return messages
```

### Structured Tool Result Compression

Raw tool results are often deeply nested JSON that the LLM will summarize before using anyway. Extract only the relevant fields:

```python
def compress_tool_result(tool_name: str, raw_result: dict) -> str:
    """Extract key fields from tool results, discarding metadata."""
    
    extractors = {
        "web_search": lambda r: "\n".join([
            f"- [{item['title']}]: {item['snippet']}"
            for item in r.get("results", [])[:5]
        ]),
        "sql_query": lambda r: f"Rows returned: {len(r.get('rows', []))}\n" + 
                               "\n".join(str(row) for row in r.get("rows", [])[:20]),
        "python_executor": lambda r: (
            f"Output:\n{r.get('stdout', '')[:500]}\n"
            f"{'Error: ' + r.get('stderr', '')[:200] if r.get('stderr') else ''}"
        ),
    }
    
    extractor = extractors.get(tool_name)
    if extractor:
        try:
            return extractor(raw_result)
        except Exception:
            pass
    
    # Fallback: truncate JSON to key fields
    import json
    return json.dumps(raw_result, indent=2)[:1000] + "..."
```

A well-tuned compression stack — prompt gating + history summarization + tool result extraction — reliably cuts total input tokens by 70–82%, translating directly to the same reduction in input token costs.

## 5. Caching: Semantic, Tool-Result, and Prefix

Caching is cost reduction with zero quality loss when done right: a cache hit returns the exact same (or semantically equivalent) result without touching the LLM. The challenge is choosing the right caching strategy for each query type.

![Caching strategies matrix showing hit rates and cost savings by strategy type](/imgs/blogs/cost-optimization-for-agents-4.webp)

### Exact Match Caching

The simplest cache: hash the full input, return the stored output on collision. Works well for:
- FAQ-style agent deployments where users ask the same questions repeatedly
- Agent tools that make deterministic external API calls (same input → same output)
- Background summarization jobs that process the same documents repeatedly

```python
import hashlib
import json
from typing import Any, Optional
import redis

class ExactMatchCache:
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    def _hash_key(self, messages: list[dict], model: str) -> str:
        content = json.dumps({"messages": messages, "model": model}, sort_keys=True)
        return f"llm:exact:{hashlib.sha256(content.encode()).hexdigest()}"
    
    async def get(self, messages: list[dict], model: str) -> Optional[str]:
        key = self._hash_key(messages, model)
        result = self.redis.get(key)
        return result.decode() if result else None
    
    async def set(self, messages: list[dict], model: str, response: str) -> None:
        key = self._hash_key(messages, model)
        self.redis.setex(key, self.ttl, response.encode())
```

Typical hit rate: 5–15%. Low, but each hit saves 100% of the LLM call cost.

### Semantic Caching

Semantic caching uses embedding similarity to serve cached responses for paraphrase-equivalent queries. If a user asks "What is the weather in Hanoi today?" and the cache contains a response for "Hanoi current weather conditions?" from 30 minutes ago, a semantic cache returns the cached answer.

```python
import numpy as np
from openai import AsyncOpenAI

class SemanticCache:
    def __init__(
        self,
        vector_store,  # e.g., Chroma, Weaviate, or pgvector
        openai_client: AsyncOpenAI,
        similarity_threshold: float = 0.92,
        ttl_minutes: int = 60,
    ):
        self.store = vector_store
        self.client = openai_client
        self.threshold = similarity_threshold
        self.ttl_minutes = ttl_minutes
    
    async def _embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    
    async def get(self, query: str) -> Optional[dict]:
        query_embedding = await self._embed(query)
        
        results = self.store.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        
        if not results["distances"][0]:
            return None
        
        distance = results["distances"][0][0]
        similarity = 1 - distance  # cosine distance → similarity
        
        if similarity < self.threshold:
            return None
        
        metadata = results["metadatas"][0][0]
        
        # Check TTL
        import time
        age_minutes = (time.time() - metadata["created_at"]) / 60
        if age_minutes > self.ttl_minutes:
            return None
        
        return {
            "response": results["documents"][0][0],
            "similarity": similarity,
            "age_minutes": age_minutes,
        }
    
    async def set(self, query: str, response: str) -> None:
        import time
        embedding = await self._embed(query)
        
        self.store.add(
            documents=[response],
            embeddings=[embedding],
            ids=[f"cache-{hashlib.md5(query.encode()).hexdigest()}"],
            metadatas=[{"query": query, "created_at": time.time()}],
        )
```

**Choosing the similarity threshold.** Too low (< 0.88) and you will return wrong answers for similar-but-different queries. Too high (> 0.97) and you will miss most cache-eligible requests. The sweet spot is 0.90–0.94 for most deployment contexts. Validate the threshold on a sample of your real traffic:

```python
async def validate_cache_threshold(
    sample_queries: list[tuple[str, str]],  # (query, expected_response)
    cache: SemanticCache,
    thresholds: list[float] = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96],
):
    """Measure false positive rate at each threshold."""
    for threshold in thresholds:
        cache.threshold = threshold
        false_positives = 0
        hits = 0
        
        for query, expected in sample_queries:
            result = await cache.get(query)
            if result:
                hits += 1
                if not is_semantically_equivalent(result["response"], expected):
                    false_positives += 1
        
        hit_rate = hits / len(sample_queries)
        fp_rate = false_positives / max(hits, 1)
        print(f"Threshold {threshold}: hit_rate={hit_rate:.2%}, false_positive_rate={fp_rate:.2%}")
```

### Tool Result Caching

Tool results are often the most cache-friendly part of an agent workflow. A web search for "Python asyncio documentation" returns the same URLs whether called at 2pm or 3pm. A database query for "total sales in Q1 2024" is deterministic and the answer does not change.

```python
import hashlib
import json
from typing import Callable, Any

class ToolResultCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def cached_tool(
        self,
        fn: Callable,
        tool_name: str,
        ttl_seconds: int = 3600,
        cache_condition: Callable[[Any], bool] = None,
    ):
        """Decorator to add caching to any tool function."""
        async def wrapper(**kwargs):
            # Create cache key from tool name + sorted kwargs
            cache_key = f"tool:{tool_name}:{hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()}"
            
            # Check cache
            cached = self.redis.get(cache_key)
            if cached:
                import json as _json
                return _json.loads(cached)
            
            # Call the real tool
            result = await fn(**kwargs)
            
            # Cache if condition is met (e.g., successful result, non-empty)
            should_cache = cache_condition(result) if cache_condition else True
            if should_cache:
                self.redis.setex(cache_key, ttl_seconds, json.dumps(result))
            
            return result
        
        return wrapper

# Usage
tool_cache = ToolResultCache(redis_client)

@tool_cache.cached_tool(
    tool_name="web_search",
    ttl_seconds=1800,  # 30 minutes
    cache_condition=lambda r: len(r.get("results", [])) > 0,
)
async def web_search(query: str, max_results: int = 5) -> dict:
    # Real search implementation
    ...
```

Tool result caching typically achieves 30–70% hit rates for research agents (users ask similar questions) and nearly 0% for coding agents (queries are unique). Know your workload.

### Prompt Prefix Caching (Provider-Level)

OpenAI, Anthropic, and Google all support prompt prefix caching at the API level: if the first N tokens of your request match a recently-seen prefix, those tokens are charged at 50–90% discount (OpenAI: 50% off, Anthropic: 90% off). This is the easiest optimization to enable — it requires zero application code changes if you structure your prompts correctly.

The rules:
1. The system prompt must be first and identical across requests
2. Any tools/functions must come next, in a consistent order
3. The cache activates after ~1,024 tokens (OpenAI) or 1,024 tokens (Anthropic)
4. The prefix must appear in at least 2 requests within a short time window

For multi-tenant agents where all users share the same system prompt (1,200 tokens + 12 tool schemas = ~4,000 tokens), prefix caching alone saves 50% of those 4,000 tokens = 2,000 tokens per request × your request volume.

```python
# Ensure consistent prefix ordering for prefix cache to activate
def build_messages_for_prefix_cache(
    system_prompt: str,
    tools: list[dict],  # Keep order STABLE across requests
    history: list[dict],
    current_message: str,
) -> tuple[list[dict], list[dict]]:
    messages = [
        {"role": "system", "content": system_prompt},  # Static prefix: max cache hits
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": current_message})
    
    # Tools stay in consistent order (alphabetical)
    stable_tools = sorted(tools, key=lambda t: t["name"])
    
    return messages, stable_tools
```

## 6. Batching: Combining Multiple Requests

Batching is the most overlooked cost optimization. OpenAI's Batch API (and similar offerings from Anthropic and others) charges 50% of the standard rate for requests that can be deferred by up to 24 hours. For non-urgent agent tasks — nightly analysis, background summarization, offline report generation — batching alone cuts LLM cost in half.

```python
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

class DeferredBatchQueue:
    def __init__(
        self,
        client,
        max_batch_size: int = 50,
        max_wait_seconds: int = 300,  # 5 minutes
        use_batch_api: bool = True,
    ):
        self.client = client
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_seconds
        self.use_batch_api = use_batch_api
        
        self._queue: list[dict] = []
        self._results: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._flush_task = None
    
    async def submit(
        self,
        request_id: str,
        messages: list[dict],
        model: str,
        urgent: bool = False,
    ) -> str:
        """Submit a request. Non-urgent goes to batch queue; urgent goes direct."""
        if urgent:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content
        
        # Queue for batching
        future = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            self._queue.append({
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model, "messages": messages},
            })
            self._results[request_id] = future
            
            if len(self._queue) >= self.max_batch_size:
                asyncio.create_task(self._flush())
        
        # Start timer for first item in queue
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(
                self._flush_after_delay(self.max_wait)
            )
        
        return await future
    
    async def _flush_after_delay(self, delay: float):
        await asyncio.sleep(delay)
        await self._flush()
    
    async def _flush(self):
        async with self._lock:
            if not self._queue:
                return
            batch = self._queue[:self.max_batch_size]
            self._queue = self._queue[self.max_batch_size:]
        
        if self.use_batch_api:
            await self._submit_batch_api(batch)
        else:
            await self._submit_concurrent(batch)
    
    async def _submit_batch_api(self, batch: list[dict]):
        """Use OpenAI Batch API for 50% cost reduction on non-urgent work."""
        import json
        import tempfile
        
        # Create JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        # Upload and create batch job
        with open(temp_path, 'rb') as f:
            batch_file = await self.client.files.create(file=f, purpose="batch")
        
        batch_job = await self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        
        # Poll for completion (in production, use a background job instead)
        while batch_job.status not in ("completed", "failed"):
            await asyncio.sleep(30)
            batch_job = await self.client.batches.retrieve(batch_job.id)
        
        if batch_job.status == "completed":
            # Retrieve and distribute results
            results_file = await self.client.files.content(batch_job.output_file_id)
            for line in results_file.text.splitlines():
                result = json.loads(line)
                request_id = result["custom_id"]
                if request_id in self._results:
                    self._results[request_id].set_result(
                        result["response"]["body"]["choices"][0]["message"]["content"]
                    )
```

### Dynamic Batch Size Tuning

Batch size directly affects latency. Larger batches achieve better throughput (and savings from reduced API overhead) but increase the average wait time before a batch is submitted. The right batch size depends on your request volume and your latency budget.

For a deployment receiving 10 requests per second with a 5-minute deferred latency budget:
- In 5 minutes, you accumulate 10 × 300 = 3,000 requests
- You could submit one batch of 3,000 or split into 60 batches of 50
- OpenAI's Batch API has no per-batch overhead, so 1 batch of 3,000 is equivalent to 60 batches of 50 in cost terms

For low-volume deployments (< 100 requests/hour), the timer trigger matters more than the size trigger: set a 5-minute max-wait and submit whatever is in the queue.

```python
class AdaptiveBatchScheduler:
    def __init__(
        self,
        target_latency_seconds: float = 300,  # 5 minutes
        min_batch_size: int = 5,
        max_batch_size: int = 1000,
        request_rate_estimate: float = 1.0,  # requests per second
    ):
        self.target_latency = target_latency_seconds
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.rate_estimate = request_rate_estimate
    
    def optimal_batch_size(self) -> int:
        """Calculate batch size that achieves target_latency at current request rate."""
        estimated_batch = int(self.rate_estimate * self.target_latency)
        return max(self.min_batch_size, min(estimated_batch, self.max_batch_size))
    
    def update_rate_estimate(self, recent_requests_per_second: float) -> None:
        """Exponential moving average of request rate."""
        alpha = 0.2  # Smoothing factor
        self.rate_estimate = alpha * recent_requests_per_second + (1 - alpha) * self.rate_estimate
```

### When to Batch vs When to Serve Immediately

| Workload type | Batch eligible? | Rationale |
|---|---|---|
| Nightly report generation | Yes | User waits until morning anyway |
| Background summarization of ingested docs | Yes | No user waiting |
| Async email drafting (user reviews later) | Yes | Hours of acceptable delay |
| User-facing chat response | No | < 3s latency required |
| Agent planning step | No | Blocks the rest of the workflow |
| Code execution output | No | Time-sensitive, user watching |
| Real-time data analysis | No | Data has TTL |

For a typical SaaS product, 20–40% of total LLM calls are eligible for batching. At 50% batch pricing, that is a 10–20% reduction in total LLM spend with zero quality impact.

## 7. Memory Efficiency: Tiered Memory Architecture

Memory is where many agents bleed tokens. An agent that naively appends every turn to its context grows its input token count linearly with task duration. A 50-turn customer support session that started at 500 tokens/turn ends at 25,000+ tokens/turn — a 50× cost multiplier over the session.

![Tiered memory showing hot in-context, warm vector store, and cold archive with 1000x cost gradient](/imgs/blogs/cost-optimization-for-agents-7.webp)

The solution is tiered memory: not everything that happened needs to stay in the expensive hot tier (the LLM context window). Structure memory as:

**Hot tier (in-context): 0 ms latency, $0.003/1k tokens per call.** This is your most expensive storage. Reserve it for: the last 3 turns, the current task state, active tool results, and explicitly retrieved memories. Budget: 2,000–4,000 tokens maximum.

**Warm tier (vector store): 50–200 ms latency, $0.0001/query.** Past session summaries, semantic memories, cross-task knowledge. Query cost is negligible; the overhead is in the retrieval latency and the tokens injected when you stage results into hot context.

**Cold tier (archive / DB): 500ms–2s latency, $0.0000002/1k tokens storage.** Full conversation logs, audit trails, raw tool results. Access is expensive in latency but nearly free in cost. Used for compliance, debugging, and rare reference lookups.

### Implementing the Eviction Policy

```python
import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class MemoryEntry:
    importance_score: float = field(compare=True)
    created_at: float = field(compare=False)
    content: str = field(compare=False)
    metadata: dict = field(default_factory=dict, compare=False)

class TieredMemoryManager:
    def __init__(
        self,
        hot_token_budget: int = 3000,
        vector_store = None,
        archive_db = None,
        summarizer = None,  # LLM client for summarization before eviction
    ):
        self.hot_budget = hot_token_budget
        self.vector_store = vector_store
        self.archive_db = archive_db
        self.summarizer = summarizer
        
        # Hot tier: priority queue (lowest importance evicted first)
        self._hot_tier: list[MemoryEntry] = []
        self._hot_token_count: int = 0
    
    def _estimate_tokens(self, text: str) -> int:
        return len(text.split()) * 1.3  # rough estimate
    
    def _importance_score(self, content: str, metadata: dict) -> float:
        """Score: recency + explicit importance flags + access frequency."""
        import time
        recency = 1.0 / (1 + (time.time() - metadata.get("created_at", time.time())) / 3600)
        explicit = metadata.get("importance", 0.5)
        accesses = metadata.get("access_count", 0) / 10
        return recency * 0.5 + explicit * 0.3 + accesses * 0.2
    
    async def add_to_hot(self, content: str, metadata: dict) -> None:
        """Add memory, evicting low-importance entries if over budget."""
        tokens = self._estimate_tokens(content)
        import time
        
        entry = MemoryEntry(
            importance_score=-self._importance_score(content, metadata),  # negate for min-heap
            created_at=time.time(),
            content=content,
            metadata=metadata,
        )
        
        heapq.heappush(self._hot_tier, entry)
        self._hot_token_count += tokens
        
        # Evict if over budget
        while self._hot_token_count > self.hot_budget and self._hot_tier:
            evicted = heapq.heappop(self._hot_tier)
            evicted_tokens = self._estimate_tokens(evicted.content)
            self._hot_token_count -= evicted_tokens
            
            # Summarize before evicting to warm tier
            if self.summarizer and len(evicted.content) > 200:
                summary = await self._summarize(evicted.content)
                await self.vector_store.add(summary, evicted.metadata)
            else:
                await self.vector_store.add(evicted.content, evicted.metadata)
            
            # Archive full content
            if self.archive_db:
                await self.archive_db.store(evicted.content, evicted.metadata)
    
    async def _summarize(self, content: str) -> str:
        """Compress content before evicting to warm tier."""
        response = await self.summarizer.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Summarize this in one sentence, preserving key facts:\n\n{content}"
            }],
            max_tokens=100,
        )
        return response.choices[0].message.content
    
    async def retrieve_from_warm(self, query: str, top_k: int = 3) -> list[str]:
        """Query warm tier and stage relevant memories to hot context."""
        results = await self.vector_store.search(query, top_k=top_k, min_similarity=0.85)
        return [r.content for r in results]
    
    def get_hot_context(self) -> list[str]:
        """Return all hot-tier contents sorted by importance (highest first)."""
        return [entry.content for entry in sorted(self._hot_tier, key=lambda e: e.importance_score)]
```

### Memory Efficiency in Practice

For a customer support agent handling 50-turn conversations, implementing tiered memory typically reduces per-turn input token cost by 60–70% compared to naive append-all approaches. The warm-tier retrieval adds 50–150ms per turn but saves $0.015–$0.020 per turn in token cost — a clear win at any reasonable latency budget.

## 8. Tool Call Efficiency: Reducing Unnecessary Tool Calls

Tool calls are a surprising cost driver in agents with unreliable external APIs. A tool that fails 20% of the time and retries twice doubles the expected cost of every call to that tool. Reducing unnecessary and redundant tool calls directly cuts both cost and latency.

```python
import asyncio
from functools import wraps

class ToolCallOptimizer:
    """Wrapper that adds deduplication, caching, and rate limiting to tools."""
    
    def __init__(self, tool_result_cache: ToolResultCache):
        self.cache = tool_result_cache
        self._in_flight: dict[str, asyncio.Future] = {}
    
    async def call(
        self,
        tool_name: str,
        tool_fn,
        deduplicate: bool = True,
        **kwargs
    ) -> Any:
        """Call a tool with deduplication (prevent concurrent identical calls)."""
        import json, hashlib
        
        call_key = f"{tool_name}:{hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()}"
        
        # Check cache first
        cached = await self.cache.get_tool_result(tool_name, kwargs)
        if cached is not None:
            return cached
        
        # Deduplication: if identical call is already in-flight, wait for it
        if deduplicate and call_key in self._in_flight:
            return await self._in_flight[call_key]
        
        future = asyncio.get_event_loop().create_future()
        if deduplicate:
            self._in_flight[call_key] = future
        
        try:
            result = await tool_fn(**kwargs)
            await self.cache.set_tool_result(tool_name, kwargs, result)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._in_flight.pop(call_key, None)
```

**Parallel tool calls.** When the agent plans multiple tool calls that do not depend on each other, execute them in parallel. A ReAct agent that calls `web_search("climate change 2024")` and `get_citations("IPCC AR6")` serially wastes the latency of whichever call finishes first. Parallel execution cuts latency in half and does not change token cost at all.

```python
async def execute_parallel_tools(
    tool_calls: list[dict],
    tool_registry: dict[str, Callable],
    optimizer: ToolCallOptimizer,
) -> list[Any]:
    """Execute all tool calls in parallel where dependencies allow."""
    
    tasks = []
    for call in tool_calls:
        tool_fn = tool_registry[call["name"]]
        task = optimizer.call(
            tool_name=call["name"],
            tool_fn=tool_fn,
            **call["arguments"]
        )
        tasks.append(task)
    
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**Prefetching predictable tool calls.** For agents that follow predictable patterns — a research agent that almost always searches before summarizing — prefetch the likely-needed tool results before the model requests them. This turns latency into cost: you may occasionally prefetch a result that is not needed (wasted tool call), but you avoid the user-perceived latency of waiting for the search.

## 9. Context Window Management: Keeping Only What Matters

Context window management is the intersection of cost and capability. Too little context and the agent loses track of what it is doing. Too much context and cost balloons without quality improvement.

For the full context window management story, see [Context Window Management for Agents](/blog/machine-learning/ai-agent/context-window-management). The cost-relevant summary:

**The token budget allocation framework.** Divide the context window into explicit budgets:

```python
@dataclass
class ContextBudget:
    system_prompt_tokens: int = 600       # Compressed, step-relevant only
    tool_schemas_tokens: int = 700        # Active tools only (3–5 max)
    working_memory_tokens: int = 1200     # Current task state, recent facts
    conversation_history_tokens: int = 800  # Last 3 turns + summary
    retrieved_context_tokens: int = 600    # From warm-tier memory query
    tool_result_tokens: int = 400          # Compressed result of last call
    response_reserve_tokens: int = 500     # Space for the model's output
    # Total: ~4,800 tokens per step vs ~10,000 uncompressed

    @property
    def total(self) -> int:
        return (self.system_prompt_tokens + self.tool_schemas_tokens +
                self.working_memory_tokens + self.conversation_history_tokens +
                self.retrieved_context_tokens + self.tool_result_tokens +
                self.response_reserve_tokens)
```

**Sliding window for long conversations.** When history exceeds budget, keep the most recent N turns plus a rolling summary. The summary lives in the system message prefix (maximizing prefix cache utilization) and is updated every K turns:

```python
SUMMARY_UPDATE_EVERY_K_TURNS = 5

async def manage_context_window(
    full_history: list[dict],
    budget: ContextBudget,
    summarizer,
    current_turn_index: int,
) -> list[dict]:
    """Returns a context-window-safe message list."""
    
    recent_turns = full_history[-3:]  # Always keep last 3 verbatim
    
    if len(full_history) > 3:
        old_turns = full_history[:-3]
        
        # Summarize every K turns
        if current_turn_index % SUMMARY_UPDATE_EVERY_K_TURNS == 0:
            summary = await summarize_turns(old_turns, summarizer)
            return [
                {"role": "system", "content": f"Earlier in this conversation: {summary}"},
                *recent_turns
            ]
    
    return full_history
```

Related: see [Circuit Breakers and Cost Caps](/blog/machine-learning/ai-agent/circuit-breakers-and-cost-caps) for how to enforce hard cost limits at runtime so a runaway context explosion does not bankrupt you before you notice.

## 10. Measuring Cost Efficiency: The Metrics That Matter

None of the optimizations above are safe to deploy blind. You need a measurement framework that tells you whether you actually saved money and whether quality held.

![Cost measurement framework showing five stages from instrumentation to regression testing](/imgs/blogs/cost-optimization-for-agents-8.webp)

### The Primary Metric: Cost Per Successful Task

Do not optimize for raw cost-per-task. Optimize for **cost-per-successful-task**: `total_cost / successful_completions`.

A naive optimization that cuts cost 50% but drops success rate from 90% to 60% makes things worse:

- Before: $1.00/task × 90% success = $1.11/successful-task
- After naive: $0.50/task × 60% success = $0.83/successful-task

The cost per successful task improved 25%, but the quality drop means customers are getting half as many successful answers. A 2× improvement in efficiency came with a 33% drop in value delivered.

The correct metric catches this immediately. Always run quality evaluation (success rate) in parallel with cost measurement.

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class TaskCostMetrics:
    task_id: str
    agent_type: str
    model_used: str
    
    # Token costs
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    
    # Derived costs ($ USD)
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    tool_cost_usd: float = 0.0  # API costs for external tools
    total_cost_usd: float = 0.0
    
    # Quality
    success: bool = False
    quality_score: Optional[float] = None  # 0.0 – 1.0
    retry_count: int = 0
    step_count: int = 0
    
    # Derived efficiency
    @property
    def cost_per_successful_task(self) -> float:
        if self.success:
            return self.total_cost_usd
        return float('inf')  # Failed task has infinite cost-per-success
    
    @property
    def effective_cost_per_quality_unit(self) -> Optional[float]:
        if self.quality_score and self.quality_score > 0:
            return self.total_cost_usd / self.quality_score
        return None

class CostInstrumentor:
    """OpenTelemetry-compatible cost tracking for agent steps."""
    
    MODEL_PRICING = {
        "gpt-4o": {"input": 2.50e-6, "output": 10.0e-6},
        "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6},
        "gpt-4.1": {"input": 2.00e-6, "output": 8.00e-6},
        "text-embedding-3-small": {"input": 0.02e-6},
    }
    
    def __init__(self, metrics_backend):
        self.metrics = metrics_backend
    
    def record_llm_call(
        self,
        task_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        step_index: int,
        agent_type: str,
    ) -> float:
        pricing = self.MODEL_PRICING.get(model, {"input": 2.50e-6, "output": 10.0e-6})
        cost = (input_tokens * pricing["input"] + 
                output_tokens * pricing.get("output", pricing["input"] * 4))
        
        self.metrics.counter("llm.cost_usd").add(cost, {
            "task_id": task_id,
            "model": model,
            "step_index": step_index,
            "agent_type": agent_type,
        })
        self.metrics.histogram("llm.input_tokens").record(input_tokens, {
            "model": model,
            "agent_type": agent_type,
        })
        
        return cost
    
    def record_task_completion(
        self,
        task_id: str,
        total_cost: float,
        success: bool,
        quality_score: Optional[float],
    ) -> None:
        self.metrics.gauge("agent.cost_per_task").set(total_cost, {
            "task_id": task_id,
            "success": str(success),
        })
        
        if success and quality_score is not None:
            efficiency = total_cost / max(quality_score, 0.01)
            self.metrics.gauge("agent.cost_per_quality_unit").set(efficiency)
```

### Secondary Metrics

| Metric | What it catches | Alert threshold |
|---|---|---|
| Input tokens p95 per step | Context window creep | >50% increase vs baseline |
| Output tokens p95 per step | Verbosity regression | >30% increase vs baseline |
| Tool retry rate | Tool reliability regression | >20% retries |
| Cache hit rate | Cache degradation | <5% drop over 7 days |
| Cost p99 / p50 ratio | Tail cost outliers | >5× ratio |
| Routing accuracy | Classifier degradation | <85% accuracy on validation set |

### Cost Attribution by Agent Type

Multi-agent systems complicate cost attribution. When an orchestrator spawns three subagents — a research agent, a summarizer, and a validator — the cost of a single user task is split across four agents, each with its own model, its own context, its own tool calls. Naive per-API-call logging gives you aggregate spend but tells you nothing about which agent type, which task category, or which user segment is driving cost.

The solution: structured tagging at every LLM call.

```python
from dataclasses import dataclass
from typing import Optional
import contextvars

# Thread-local (or async-context-local) cost context
_cost_context: contextvars.ContextVar = contextvars.ContextVar('cost_context', default=None)

@dataclass
class CostContext:
    task_id: str
    user_id: str
    agent_type: str
    parent_agent_type: Optional[str] = None
    session_id: Optional[str] = None
    task_category: Optional[str] = None  # e.g. "research", "code", "support"

def set_cost_context(ctx: CostContext):
    _cost_context.set(ctx)

def get_cost_context() -> Optional[CostContext]:
    return _cost_context.get()

# Instrumented LLM call wrapper
async def llm_call_with_attribution(
    client,
    messages: list[dict],
    model: str,
    instrumentor: CostInstrumentor,
    **kwargs,
):
    ctx = get_cost_context()
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    
    usage = response.usage
    if ctx:
        instrumentor.record_llm_call(
            task_id=ctx.task_id,
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            step_index=kwargs.get("step_index", 0),
            agent_type=ctx.agent_type,
        )
        
        # Also record parent attribution for orchestrator cost rollup
        if ctx.parent_agent_type:
            instrumentor.record_attribution(
                task_id=ctx.task_id,
                from_agent=ctx.parent_agent_type,
                to_agent=ctx.agent_type,
                cost_usd=instrumentor.calculate_cost(model, usage.prompt_tokens, usage.completion_tokens),
            )
    
    return response
```

With proper attribution, you can answer questions like:
- "The research subagent accounts for 68% of total cost for research-type tasks"
- "The validator agent costs $0.02/task and only catches 3% of real errors — is it worth keeping?"
- "User segment 'power users' generates 4× the cost of 'casual users' but only 2× the revenue"

These insights are only possible with structured cost tagging from the start.

### A/B Testing Cost Optimizations

Before deploying any optimization to 100% of traffic, run a shadow evaluation:

```python
async def ab_test_optimization(
    requests: list[AgentRequest],
    baseline_agent: Agent,
    optimized_agent: Agent,
    evaluator: AgentEvaluator,
    traffic_split: float = 0.1,  # 10% to optimized
) -> dict:
    """Run A/B test on a sample of traffic."""
    
    baseline_results = []
    optimized_results = []
    
    for req in requests:
        if random.random() < traffic_split:
            result = await optimized_agent.run(req)
            optimized_results.append(result)
        else:
            result = await baseline_agent.run(req)
            baseline_results.append(result)
    
    # Compare key metrics
    baseline_cost = sum(r.total_cost for r in baseline_results) / len(baseline_results)
    optimized_cost = sum(r.total_cost for r in optimized_results) / len(optimized_results)
    
    baseline_success = sum(r.success for r in baseline_results) / len(baseline_results)
    optimized_success = sum(r.success for r in optimized_results) / len(optimized_results)
    
    return {
        "cost_reduction": 1 - optimized_cost / baseline_cost,
        "success_rate_delta": optimized_success - baseline_success,
        "cost_per_success_baseline": baseline_cost / max(baseline_success, 0.01),
        "cost_per_success_optimized": optimized_cost / max(optimized_success, 0.01),
    }
```

## 11. Cost Regression Testing: Detecting When Changes Increase Cost

Every time you change a prompt, add a tool, upgrade a model, or modify the agent's logic, you risk a cost regression. A cost regression test suite catches these automatically.

![Cost regression test matrix showing different change types with alert thresholds](/imgs/blogs/cost-optimization-for-agents-9.webp)

### Building the Test Suite

```python
import pytest
from typing import NamedTuple

class CostBaseline(NamedTuple):
    scenario_name: str
    expected_input_tokens_p50: float
    expected_cost_per_task_p50: float
    tolerance: float  # e.g. 0.15 = alert if 15% above baseline

# Pre-measured baselines from production
COST_BASELINES = [
    CostBaseline("simple_qa", 1800, 0.0054, 0.15),
    CostBaseline("research_task", 4200, 0.0126, 0.20),
    CostBaseline("code_generation", 3500, 0.0105, 0.20),
    CostBaseline("long_form_analysis", 8000, 0.0240, 0.25),
]

@pytest.mark.asyncio
@pytest.mark.parametrize("baseline", COST_BASELINES)
async def test_cost_regression(baseline: CostBaseline, agent, test_scenarios, instrumentor):
    """Fail CI if cost exceeds baseline by more than tolerance."""
    
    scenarios = test_scenarios[baseline.scenario_name]
    total_cost = 0.0
    total_input_tokens = 0
    
    for scenario in scenarios[:20]:  # Sample 20 scenarios per type
        result = await agent.run(scenario.request)
        total_cost += result.cost_usd
        total_input_tokens += result.input_tokens
    
    mean_cost = total_cost / len(scenarios[:20])
    mean_tokens = total_input_tokens / len(scenarios[:20])
    
    cost_threshold = baseline.expected_cost_per_task_p50 * (1 + baseline.tolerance)
    token_threshold = baseline.expected_input_tokens_p50 * (1 + baseline.tolerance)
    
    assert mean_cost <= cost_threshold, (
        f"{baseline.scenario_name}: mean cost ${mean_cost:.4f} exceeds threshold "
        f"${cost_threshold:.4f} ({baseline.tolerance:.0%} above baseline {baseline.expected_cost_per_task_p50:.4f})"
    )
    assert mean_tokens <= token_threshold, (
        f"{baseline.scenario_name}: mean input tokens {mean_tokens:.0f} exceeds threshold "
        f"{token_threshold:.0f}"
    )
```

### Integrating Into CI

```yaml
# .github/workflows/cost-regression.yml
name: Cost Regression Tests

on:
  pull_request:
    paths:
      - 'agents/**'
      - 'prompts/**'
      - 'tools/**'

jobs:
  cost-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run cost regression tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }}
          COST_REGRESSION_MODE: "true"
        run: |
          pytest tests/cost_regression/ \
            --tb=short \
            -v \
            --cost-baseline-file=cost_baselines.json \
            --fail-on-cost-regression
      
      - name: Post cost delta to PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const report = require('./cost_regression_report.json');
            const body = `## Cost Regression Report
            
            | Scenario | Baseline | Current | Delta |
            |---|---|---|---|
            ${report.results.map(r => 
              `| ${r.scenario} | $${r.baseline.toFixed(4)} | $${r.current.toFixed(4)} | ${r.delta > 0 ? '+' : ''}${(r.delta * 100).toFixed(1)}% |`
            ).join('\n')}`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body
            });
```

## 12. Case Studies: Real-World Cost Optimization Wins and Regressions

### Case Study 1: Research Agent Cuts Cost 87% With Tiered Routing

A B2B SaaS company was running a competitive intelligence agent that researched companies, products, and market trends. Baseline cost: $0.31 per research task, running 3,000 tasks/day = $930/day.

The team audited 500 production tasks and found:
- 68% of steps were "retrieve and summarize" (simple)
- 22% were "synthesize across multiple sources" (medium)  
- 10% were "reason about contradictory evidence" (hard)

They implemented three-tier routing (GPT-4o-mini / GPT-4o / GPT-4.1) and added prompt compression (gating tool schemas, summarizing history every 5 turns). No caching in this phase.

Result after 2 weeks:
- Cost per task: $0.31 → $0.04 (87% reduction)
- Success rate: 91% → 90% (1% drop, within acceptable range)
- Daily spend: $930 → $120

The 1% quality drop was entirely in the "retrieve and summarize" steps, where GPT-4o-mini occasionally missed a nuance that GPT-4o would have caught. The team added a confidence check: if the mini model output confidence (estimated via log-probs or a self-grade prompt) was below 0.85, it escalated. This recovered the quality to 90.5%.

### Case Study 2: Customer Support Agent Regresses on History Compression

A customer support agent used rolling conversation summaries to compress history. Baseline: $0.08/conversation, 90% resolution rate.

After compressing history every 3 turns (aggressive setting), cost dropped to $0.035/conversation. But resolution rate dropped from 90% to 76%.

Root cause: the summary was losing the customer's account history details ("I called last week about X") that were referenced 8–12 turns later in long conversations. The compression was overaggressive: it summarized facts that sounded less important ("customer mentioned calling last week") but turned out to be critical for resolution.

Fix: tag turns that contain customer account references with high importance before summarizing, and always include those turns verbatim. New result: $0.042/conversation (48% reduction), 89% resolution rate.

Lesson: history compression needs domain-specific importance tagging, not generic summarization. Do not measure only cost savings — always measure task-type-specific quality metrics.

### Case Study 3: Semantic Cache Saves 40% on Document Analysis Agent

A legal document analysis agent processed the same set of ~500 standard contract templates repeatedly, with different user queries about each document. The team added semantic caching with a 0.92 cosine similarity threshold.

Cache hit rate: 38% in week 1, stabilizing at 42% by week 4 (as the cache warmed up with real queries).

Cost impact:
- Before: $0.18/query
- After: $0.18 × (1 - 0.42) = $0.10/query (44% reduction)
- With embedding call overhead: net ~40% reduction

The cache also reduced latency significantly — a cache hit returned in 30ms vs. 2.5s for a full LLM call, which improved user-perceived responsiveness.

One unexpected issue: queries about regulatory changes (e.g., "does this contract comply with the new GDPR amendment from March 2025?") were serving stale cached responses. Fixed with a time-aware cache key: queries that contain date references or the word "recent/current/latest/new" skip the cache entirely.

### Case Study 4: Model Downgrade Causes Reasoning Failure in Planning Agent

A logistics planning agent was routing all steps to GPT-4o-mini to save costs. The 80% cost savings looked great on paper. But within two weeks, customer escalations spiked.

The root cause: the planning step — choosing the optimal route given a complex set of constraints (delivery windows, vehicle capacities, traffic, temperature requirements) — was legitimately hard. GPT-4o-mini was producing valid-looking plans that violated temperature constraints for temperature-sensitive cargo. The plans parsed correctly and passed schema validation but were wrong.

The quality check had been "output is valid JSON matching the plan schema" — a syntactic check, not a semantic one. The real quality metric should have been "plan satisfies all hard constraints."

Fix:
1. Added a constraint-verification step using a deterministic solver (not LLM) to check hard constraints. This cost ~$0.0001 per plan (cheap) and caught violations.
2. Routed all planning steps back to GPT-4o.
3. Added a test suite with constraint-heavy scenarios to the cost regression test.

Final cost: up from the 80%-off low, but still 45% cheaper than baseline because summarization and data extraction steps stayed on mini.

### Case Study 5: Tool Result Caching Eliminates Duplicate API Costs

A market data agent was calling a financial data API (charged per call at $0.05/call) to fetch stock prices. In a multi-turn analysis session, the same ticker would be fetched 3–8 times in different steps: once to get current price, once to check recent trend, once to compare to industry peers.

After adding tool result caching with a 60-second TTL for real-time prices (prices do not change in 60 seconds in a meaningful way for most analyses) and 24-hour TTL for fundamental data (P/E ratios, revenue), API call costs dropped from $1.20/session to $0.15/session — an 88% reduction in that cost layer.

The key insight: the LLM does not know that it asked for AAPL's price two steps ago. It will ask again. The cache layer does know, and serves the cached result without the $0.05 API charge.

### Case Study 6: Batching Cuts Nightly Report Costs 50%

A content platform generated daily performance summaries for 10,000 publishers every night. Each summary was a multi-step agent task (fetch metrics, compare to benchmarks, draft narrative) costing $0.12/publisher = $1,200/night.

The team identified that all 10,000 tasks were submitted in a 2-hour window (publisher dashboards update at midnight) and did not need to complete until 8am — 6 hours of acceptable delay.

After implementing OpenAI's Batch API:
- Cost: $1,200/night → $600/night (50% reduction, as advertised)
- Completion time: 2–4 hours (within the 8am deadline)
- Quality: identical (same prompts, same model, just queued)

The only engineering effort was wrapping the existing task submission in a batch queue and adding a polling loop to collect results. Total implementation time: 3 days.

### Case Study 7: Context Window Explosion From Injecting Raw Error Messages

A code-generation and execution agent started concatenating full Python tracebacks into the agent's context when code execution failed. A complex debugging session with 5 errors each producing 400-line tracebacks injected 2,000 lines × ~8 tokens/line = 16,000 tokens into a single step's context.

Cost per debugging session: $0.48 (instead of $0.08 for a normal session). The team had no alerting on p99 cost, only on mean cost — so the regression was invisible in aggregate but was hitting a significant fraction of debugging users.

Fix:
1. Truncate tracebacks to the last 30 lines + the first 10 lines (where the error is, and where it propagates)
2. Compress error messages: strip ANSI codes, line numbers for library-internal frames, and repeated stack frames
3. Add per-session cost cap: if a session exceeds $0.20, switch to a less expensive model for remaining steps

Related: [Circuit Breakers and Cost Caps](/blog/machine-learning/ai-agent/circuit-breakers-and-cost-caps) covers the pattern of hard cost limits at the session level.

### Case Study 8: Memory Tiering Saves 62% for Long-Running Research Agents

A competitive intelligence agent ran 60–90 minute research sessions, accumulating 200–300 conversation turns before completing a deep-dive report. Without tiered memory, by turn 100, the context was 40,000+ tokens per step — at $0.10/step × 100 steps beyond turn 100, that was $10 in the tail of the session.

After implementing tiered memory with hot tier capped at 3,000 tokens (rolling summary + recent 3 turns + retrieved warm-tier memories):
- Average cost per session: $4.20 → $1.60 (62% reduction)
- Session completion quality (rated by human reviewers): 88% → 87% (1% drop)
- The 1% quality drop was from very specific facts mentioned early in the session that were not retrieved by the warm-tier query at the relevant step

The team added "anchor facts" — facts explicitly marked by the user as high-importance early in the session are stored in a separate always-included memory slot (200 tokens reserved), not subject to eviction. This recovered quality to 88.5%.

## 13. The Cost-Quality Tradeoff: Finding the Pareto-Optimal Point

Every optimization lives on a tradeoff frontier. Cut costs carelessly and quality falls; cut them intelligently and you can cut 90% while quality barely moves. The key is to measure cost-per-success, not cost in isolation.

![Cost-quality tradeoff showing naive vs Pareto-optimal optimization paths](/imgs/blogs/cost-optimization-for-agents-10.webp)

### The Pareto-Optimal Optimization Process

The sequence matters. Each optimization should be evaluated independently, in order of lowest quality risk to highest:

**Phase 1 (zero quality risk): Quick wins**
1. Enable prompt prefix caching (API-level, zero code changes)
2. Implement tool result caching for deterministic tools (TTL: 30–60 minutes)
3. Enable batching for non-urgent background tasks
4. Deduplicate identical tool calls within a single agent step

**Phase 2 (low quality risk): Structure optimizations**  
1. Gate tool schemas — load only relevant tools per step type
2. Implement three-tier model routing with conservative classifier thresholds
3. Compress tool results (keep key fields, truncate verbose JSON)

**Phase 3 (medium quality risk): Context optimizations**
1. Summarize conversation history with keep-last-3-verbatim strategy
2. Compress system prompt with step-type-aware section injection
3. Move to tiered memory with high eviction thresholds

**Phase 4 (requires careful validation): Aggressive compression**
1. Semantic caching with threshold tuning
2. Aggressive history compression (summarize every 2 turns)
3. Route more steps to cheaper models

After each phase, run the quality evaluation and confirm cost-per-successful-task improved. Stop at the phase where cost-per-success starts to degrade.

### The Floor You Must Not Cross

Define a quality floor before you optimize. For most production agents, this is:
- Success rate >= 88% (or whatever your baseline was before optimization)
- p95 latency <= your SLA
- User satisfaction score >= 4.0 / 5.0 (if measurable)

Any optimization that pushes any of these below floor is unacceptable, regardless of the cost savings it achieves.

### When Optimization Hurts Capability: Red Flags

Certain optimization patterns reliably damage quality:

**Too aggressive history compression.** If conversation turns reference earlier turns by content ("as I mentioned earlier, the revenue was $12M"), and those turns are compressed away, the reference breaks. Keep enough history verbatim to cover the reference distance in your typical conversations.

**Routing complex tasks to cheap models.** The simplest failure mode: a task looks syntactically simple (short prompt, no tools) but is semantically hard (requires domain expertise, subtle reasoning). Rule-based routers misclassify these. LLM-based classifiers handle them better because they read the content, not just the structure.

**Caching time-sensitive queries.** A semantic cache match for "What is the current Bitcoin price?" should not return a 2-hour-old cached response. Detect time-sensitive queries (keywords: current, now, today, recent, latest, live) and bypass the cache.

**Semantic cache threshold too low.** At threshold 0.85, "What is Apple's stock price?" and "What is Apple's quarterly revenue?" may score 0.87 similarity and return the same cached response — completely wrong. Validate with representative queries from your domain.

## Production Anti-Patterns: What Not to Do

We have spent most of this post describing what to do. Equally valuable is understanding what goes wrong in practice. These anti-patterns recur across almost every team that ships agents at scale.

**Anti-pattern 1: Optimizing the wrong metric.** The most common mistake. Teams focus on $/call or $/token, not $/successful-task. We have seen teams celebrate 70% cost reduction while their quality monitoring (if they had it) showed success rate dropping from 91% to 55%. On a per-successful-task basis, they were paying *more* than before the "optimization."

The fix is simple: define cost-per-successful-task before you start, measure it continuously, and refuse to ship any optimization that does not improve it.

**Anti-pattern 2: Adding caching as an afterthought.** Teams implement semantic caching after the fact, then discover the cache is causing intermittent quality bugs that are hard to reproduce (because they only trigger on cache hits, not cache misses). Debugging a semantic cache miss is especially tricky: the user sees a wrong answer, the logs show a cache hit, but the hit came from a slightly different query 3 hours ago.

Design your caching layer with observability from day one: log every cache hit with the original query, the cached query, the similarity score, and whether the returned response was correct (rated by downstream quality signals). This is how you tune the threshold with real data.

**Anti-pattern 3: Routing without escalation.** Teams implement routing (cheap model for simple tasks) without adding an escalation path. When the cheap model fails — wrong answer, malformed output, violated constraint — the agent either returns the bad answer or fails the task entirely. There is no path to "try the more expensive model."

Always implement the escalation pattern (shown in the routing section above). The cost of escalation is the cost of the expensive call; the cost of not escalating is a failed task that cannot be recovered from.

**Anti-pattern 4: Tool schema loading by habit.** The most universal waste: loading all N tools in every step because the original agent was "designed that way." Every tool schema is 150–300 tokens. 12 tools = 1,800–3,600 tokens of schema, loaded whether or not the current step could ever call any of them.

Audit your agent's tool use across 100 production sessions. You will almost certainly find that 80%+ of steps use at most 2–3 tools. The remaining tools are never called but are paying the schema-loading tax on every step.

**Anti-pattern 5: No per-session cost caps.** An agent without a cost cap is an agent that can bankrupt you. A single adversarial input, a retry loop bug, or an unexpected edge case can cause an agent to spin indefinitely, generating thousands of LLM calls before anyone notices. We have seen production incidents where a single malformed user input caused 800 LLM calls before the session was manually killed.

Implement per-session cost caps at the infrastructure level, not in the agent logic. The agent loop itself may be the thing that is broken, so you cannot rely on the agent to self-limit. Use a middleware layer or a watchdog process that kills sessions exceeding a cost threshold. See [Circuit Breakers and Cost Caps](/blog/machine-learning/ai-agent/circuit-breakers-and-cost-caps) for the full pattern.

**Anti-pattern 6: Silent cache TTL expiration.** A cache with a TTL that is too long serves stale answers. A cache with a TTL that is too short has a low hit rate. Neither failure mode is obvious from looking at average cache hit rates, because they both look like "low hit rate." The difference: TTL-too-long should surface in quality metrics (wrong answers for time-sensitive queries); TTL-too-short surfaces in cost metrics (low hit rate, high per-task cost).

Monitor both. Build dashboards that show cache hit rate, cache miss rate, and the correlation between cache hits and answer quality scores (from user feedback or LLM-as-judge). Tune TTL separately for different tool types: stock prices (60 seconds), company fundamentals (24 hours), historical documents (indefinite).

**Anti-pattern 7: Compression that loses referential integrity.** The most subtle quality failure in history compression: the compressed summary uses pronouns or references that become ambiguous once the original text is gone. "She rejected the proposal" in a summary, but the original text makes clear "she" is the CFO of the target company — and the agent's next decision depends on knowing the CFO's role.

Write summaries in full-reference form: "CFO Jane Smith rejected the acquisition proposal citing valuation concerns." No pronouns, no implicit references. This adds tokens to summaries but eliminates referential-ambiguity bugs.

**Anti-pattern 8: Optimizing production before optimizing evals.** Every optimization in this post requires an evaluation harness to be safe. If you do not have a way to measure whether optimization X degraded quality, you should not deploy optimization X. Build your evaluation harness (test scenarios, quality metrics, regression tests) before you ship to production.

This is counterintuitive when you are under cost pressure: the most urgent thing feels like deploying the cost cut immediately. But a cost cut deployed without evals will eventually cause a quality regression you can't detect until customer support is flooded. Build evals first, cost cuts second.

## When to Reach for Each Optimization

**Start here (implement on day 1):**
- Prompt prefix caching (zero engineering effort, 40–60% savings on shared prefix)
- Tool schema gating (1 day, 10–20% input token reduction)
- Tool result caching with short TTL (1 day, 10–25% savings)

**Do these in your first sprint:**
- Three-tier model routing with rule-based classifier (3–5 days, 60–80% savings)
- History summarization with keep-last-3 (3 days, 30–50% history token savings)
- Batch API for non-urgent tasks (3 days, 50% savings on eligible tasks)

**Do these after you have production traffic to calibrate on:**
- Semantic caching with threshold tuning (requires real query distribution)
- Tiered memory with importance-based eviction (requires per-domain importance heuristics)
- LLM-based complexity classifier (requires labeled training data from production)

**Never do these:**
- Remove context that the agent genuinely needs to complete its task
- Route tasks that require creativity, complex reasoning, or multi-step planning to cheap models without extensive testing
- Cache without TTL (stale data is often worse than no cache)
- Optimize for cost without measuring quality in parallel

## Building a Cost-Aware Development Culture

The techniques above are engineering solutions. But the biggest driver of long-term cost discipline is not a technical pattern — it is building cost visibility into your development workflow so that developers feel the impact of their prompt changes before they hit production.

**Show cost in the development loop.** When a developer writes a new prompt, they should see an estimated token count and cost immediately. A simple CLI tool that runs the prompt against a sample of representative inputs and reports mean tokens + estimated cost creates the feedback loop that prevents casual 500-token bloat in system prompts.

```bash
$ agent-cost-check --prompt prompts/research_agent.txt \
                   --sample test_cases/research_samples.json \
                   --compare baseline
                   
Prompt size: 1,247 tokens (+23 vs baseline of 1,224)
Sample input mean: 3,891 tokens/step (+47 vs baseline of 3,844)
Estimated cost per task: $0.0117 (+1.2% vs baseline of $0.0116)
All within thresholds. ✓
```

**Cost visibility in code review.** When a PR modifies a prompt file, the CI cost regression check should post its results directly to the PR as a comment (shown in the regression testing section above). Reviewers should be expected to look at cost impact the same way they look at performance impact.

**Cost budgets per team.** In larger organizations, allocate monthly LLM spend budgets to teams. A team running a research agent gets a budget; a team running a code-gen agent gets a different budget. When a team approaches its budget, they own the problem. This creates natural incentives for cost-conscious engineering that no centralized "cost optimization initiative" can replicate.

**Postmortems for cost regressions, not just reliability regressions.** When a deployment causes a 2× spike in LLM spend, hold a postmortem. What was the root cause? What was missed in the regression tests? What early signal was available that was ignored? Treating cost regressions with the same seriousness as reliability incidents trains the team to think about cost as a first-class production concern.

**The developer experience investment.** All of this — cost checking in the dev loop, PR cost comments, team budgets, cost postmortems — requires a small investment in internal tooling. Teams that make this investment consistently outperform teams that do not in the long run. The cost-oblivious team ships fast and optimizes later; the cost-aware team ships almost as fast but avoids the debt of rearchitecting a system that has grown to 10× its originally budgeted cost.

## Connecting the Pieces: Agent Observability and Memory

This post focused on the optimization mechanics. Two companion pieces complete the picture:

- [Agent Observability and Tracing](/blog/machine-learning/ai-agent/agent-observability-and-tracing) covers how to instrument your agent with the span-level tracing that makes cost attribution actually accurate. Without good instrumentation, you are guessing at where cost comes from.
- [Agent Memory Cost Optimization](/blog/machine-learning/ai-agent/agent-memory-cost-optimization) goes deeper on the tiered memory architecture — specifically the warm-tier retrieval patterns and how to tune the importance scoring for different agent types.
- [Circuit Breakers and Cost Caps](/blog/machine-learning/ai-agent/circuit-breakers-and-cost-caps) covers runtime enforcement: what to do when a session is heading toward runaway cost, and how to implement per-session and per-tenant spend limits without breaking the user experience.
- [Context Window Management for Agents](/blog/machine-learning/ai-agent/context-window-management) covers the full context management story, including windowing strategies for long-horizon tasks that go beyond what we covered here.

## Evolving the Stack as Models Change

One important caveat: the specific numbers in this post (model prices, hit rates, savings percentages) shift as the model landscape evolves. GPT-4o at $2.50/1M input tokens in 2024 may be $0.50/1M in 2026. Claude 3.5 Sonnet at $3.00/1M may be succeeded by a model twice as capable for the same price.

What does not change is the structure of the optimization problem:

1. **The loop multiplier always applies.** Whether the baseline cost is $2.50 or $0.25 per 1M tokens, a 10-step agent still costs 10–40× a single call before optimization.

2. **Model tier gaps persist.** There will always be a cost ratio of 5–20× between the cheapest usable model and the frontier model. The routing insight remains: you do not need the frontier model for 70% of steps.

3. **Redundancy persists.** Human language is naturally redundant. Unless you change how prompts are written at a structural level, raw context will always contain more tokens than strictly necessary.

4. **Caching always works.** As long as users ask similar questions, caching saves money. As long as tools produce deterministic outputs, tool result caching saves money.

What will change: the optimal threshold values (similarity threshold for semantic cache, confidence threshold for routing escalation), the specific models at each tier, and the exact cost savings percentages. Plan your cost optimization infrastructure with configurable thresholds and model references so that as prices change, you update configuration, not code.

The investment in measurement infrastructure — the `CostInstrumentor`, the cost regression test suite, the A/B testing framework — pays dividends regardless of what the model landscape looks like. The specific numbers will become obsolete; the ability to measure and optimize will not.

## Summary: The Cost Optimization Stack

The complete agent cost optimization stack, in order of implementation priority:

1. **Instrument first.** You cannot optimize what you cannot attribute. Cost-per-successful-task is your north-star metric.

2. **Enable prefix caching.** Zero code changes, 40–60% savings on shared system prompts.

3. **Gate tool schemas.** Load only the tools relevant to the current step type. Cuts tool-schema tokens by 60–80%.

4. **Implement model routing.** Three tiers (cheap/mid/frontier) with a simple rule-based or heuristic classifier. Single highest-leverage optimization — 60–80% cost reduction for typical research/analysis agents.

5. **Compress history.** Rolling summary + verbatim last 3 turns. 30–50% reduction in history tokens.

6. **Cache tool results.** Deterministic tools with short TTL. 10–25% savings with zero quality risk.

7. **Add semantic caching.** For query-heavy deployments where paraphrase-equivalent queries are common. 20–40% additional savings.

8. **Implement tiered memory.** For long-running sessions (> 20 turns). 40–70% reduction in per-turn context cost.

9. **Run cost regression tests in CI.** Every prompt change, model change, and tool change should trigger a cost regression check before deployment.

10. **Measure cost-per-successful-task at every step.** If quality drops faster than cost drops, stop and re-evaluate.

Applied in full, this stack consistently achieves 80–90% cost reduction on the agents we have seen in production. The math compounds: routing × compression × caching × memory all multiply together. Getting from $0.31/task to $0.04/task is not one big win — it is five 35–40% wins stacked on top of each other.

The single most important thing to take away from this post: measure cost-per-successful-task from day one. Before you enable prefix caching, before you implement routing, before you compress a single prompt — set up the instrumentation so you can see whether what you are doing is actually working. Every optimization technique above has a failure mode that is invisible in aggregate cost metrics but visible in quality metrics. The teams that ship sustainable, low-cost agents are the teams that track both at every step and refuse to ship an optimization that only improves one of them.

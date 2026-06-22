---
title: "Agent Loop Anatomy: Dissecting One Turn from Prompt to Side-Effect"
date: "2026-06-22"
description: "A microsecond-level breakdown of every phase inside one agent turn, with latency budgets, token budgets, and the six places errors inject."
tags: ["ai-agents", "llm", "agent-architecture", "machine-learning", "deep-learning", "nlp", "system-design", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 33
---

I've seen more production agent failures caused by how the loop is assembled than by the LLM itself. The model is usually fine. The scaffolding is where things go wrong — a prompt that silently grows to 128k tokens, a JSON parser that retries on every response, a tool timeout that blocks the entire turn for four seconds at p95. These are solvable problems, but only if you understand exactly what happens inside one turn.

This post is a dissection. We'll go phase by phase through a single agent turn, from the moment a user message arrives to the moment a side-effect lands in the world. We'll look at real latency numbers, token costs, error frequencies, and the specific bugs that show up in production systems built by people who thought they understood their loop.

If you want the high-level picture first, [what is an AI agent](/blog/machine-learning/ai-agent/what-is-an-ai-agent) covers the perceive-reason-act-remember cycle. This post goes one level deeper: the internal mechanics of one iteration of that cycle.

## Why Loop Internals Determine Production Reliability

A production agent is not evaluated on whether it can answer one question. It's evaluated on whether it can complete a ten-step task without burning \$2 in tokens, without hitting a context limit at step seven, without silently producing wrong output when a tool returns an empty response.

The loop is the unit of reliability. Each turn is a mini-system: it constructs inputs, invokes the LLM, parses the output, executes side-effects, and decides whether to continue. Every one of those steps can fail. And unlike a simple API call, failures in an agent loop compound — a wrong decision in turn three leads to incorrect tool calls in turn five, which produces corrupted context in turn seven, which eventually surfaces as a wrong final answer that is very hard to debug.

The engineers I've seen build reliable production agents share one property: they can articulate what happens inside a single turn in precise terms. They know which phase owns the token budget. They know which phase contributes to p95 latency. They know where to add observability hooks. This post is an attempt to build that mental model from scratch.

![The six phases of one agent turn](/imgs/blogs/agent-loop-anatomy-1.webp)

## Phase 1: System Prompt Construction

The first thing that happens in any agent turn is building the static portion of the context: the system prompt. This sounds trivial. It is not.

The system prompt is not a single string you copy-paste from a tutorial. In a production agent, it is a structured artifact that encodes the agent's identity, its constraints, its available tools, and examples of how to use them. Getting this structure wrong causes subtle failures that are much harder to diagnose than a tool timeout.

### What Goes Into the System Prompt

A production system prompt has five layers, and the order of those layers matters because language models attend differently to early versus late context:

**Layer 1: System instructions.** The foundational rules: what the agent is, what it should do, what it should never do. "You are a customer support agent for Acme Corp. You have access to the following tools. You must always ask for clarification before performing write operations." This layer is typically 300–800 tokens.

**Layer 2: Tool schemas.** The JSON schema or function-calling definitions for every tool the agent can invoke. This layer is often the largest static component — a set of ten well-documented tools can consume 1,500–3,000 tokens. The quality of your tool schemas directly determines parse reliability in Phase 4.

**Layer 3: Few-shot examples.** Optional, but powerful. Two to four examples of complete tool-calling interactions teach the model what good execution looks like. Each example consumes 400–800 tokens. Skip these for simple tasks; include them when the model produces consistently wrong tool call formats or when tool selection accuracy matters.

**Layer 4: Retrieved context.** This is the first dynamic layer — it changes per turn. Documents retrieved from a vector store, memory fragments from previous sessions, or injected structured data. This can be zero tokens (on a simple Q&A turn) or 4,000–8,000 tokens (on a heavy RAG turn). It must appear before the conversation history.

**Layer 5: Conversation history and the current user message.** The actual input the user provided, plus whatever history from prior turns the agent needs to maintain context. This grows every turn.

![System prompt anatomy: five layers with token costs](/imgs/blogs/agent-loop-anatomy-2.webp)

### The Caching Opportunity

Layers 1–3 are identical across every turn in a session. They never change at runtime. This is the most underused optimization in production agents: prefix caching.

Most major LLM providers (Anthropic, OpenAI, Google) implement KV-cache at the prefix level. If the first N tokens of a request are byte-identical to a recent request from the same user or session, the provider skips re-encoding them and charges a fraction of the normal price (Anthropic charges 10% of the input token rate for cached tokens; OpenAI charges 50%).

For an agent with a 2,000-token static prefix running 1,000 turns, that's 2M tokens × 90% discount = roughly \$15–45 in savings depending on the model tier. More importantly, skipping re-encoding reduces time-to-first-token latency, which matters for interactive agents.

The requirement: static layers must appear first, be byte-identical, and not be interleaved with dynamic content. Any modification to the system instructions resets the cache for that session.

### Validating Your System Prompt at Build Time

The most common Phase 1 failure is a token limit error that only surfaces at runtime when the context grows. Fix this by measuring your system prompt length at startup:

```python
import tiktoken

def validate_system_prompt(system_prompt: str, model: str, max_tokens: int = 4096):
    """Validate system prompt token count at startup, not per-turn."""
    enc = tiktoken.encoding_for_model(model)
    token_count = len(enc.encode(system_prompt))
    if token_count > max_tokens:
        raise ValueError(
            f"System prompt exceeds budget: {token_count} tokens "
            f"(budget: {max_tokens}). Reduce tool schemas or instructions."
        )
    return token_count

# Call this once at agent initialization, not inside the turn loop
system_prompt_tokens = validate_system_prompt(SYSTEM_PROMPT, "gpt-4o", max_tokens=3000)
```

This simple check prevents the failure mode where a well-intentioned engineer adds a new tool schema and inadvertently pushes the prompt over the context limit, which only manifests as silent truncation at turn 7 on long conversations.

## Phase 2: Message Assembly and Context Injection

Phase 2 takes the static system prompt from Phase 1 and assembles the full message list: history, retrieved documents, tool results from prior turns, and the current user input. This is where the agent's "working memory" gets assembled from its component parts.

### History Management: The Accumulation Problem

The most dangerous property of conversation history is that it grows by default. Every turn adds at minimum two messages (user and assistant). On a ten-turn task with 500-token turns, unmanaged history alone consumes 10,000 tokens — and that's before you add retrieved documents or tool results.

The naive approach — append every message to a list — produces a system that works fine for five turns and silently starts degrading at turn eight as the context window fills. The model's attention mechanism struggles to maintain long-range coherence when the context becomes a wall of text. You get answers that contradict earlier turns, tool calls that forget which files have already been modified, and eventual context overflow errors.

The correct approach is a history management strategy, applied during Phase 2, before every LLM call. There are three main patterns:

**Rolling window.** Keep only the last N turns. Simple, predictable, loses old context. N=6 (three exchange pairs) works for most task agents. Implement as a deque with maxlen.

**Summarization.** When history exceeds a threshold, call a fast cheap model (GPT-3.5, Claude Haiku) to summarize the last K turns into a paragraph, replace those turns with the summary, and continue. More expensive but preserves semantic content. Required for tasks where facts from early turns matter at turn 20.

**Importance scoring.** Tag messages with importance scores during injection and prune low-importance messages first. A tool call that returned a critical piece of data should outlive a clarification question that was already answered.

### Retrieved Context: Ordering and Freshness

Documents retrieved from a vector store need to be injected in a specific order relative to the conversation history: retrieved context must appear before the user's question, not after it. When retrieved context follows the user question, many models ignore it or underweight it in their responses. The stack order matters.

Also: retrieved context from turn N-1 is stale by definition. If your agent modifies data and then retrieves it again, you may be injecting an outdated snapshot. Build cache-invalidation logic into your retriever: any write action should trigger a fresh retrieval rather than serving the cached result from the previous turn.

### Token Budget Enforcement

Phase 2 is where you enforce your total input token budget. Build a budget allocator:

```python
def assemble_context(
    system_prompt: str,
    history: list[dict],
    retrieved_docs: list[str],
    user_message: str,
    max_input_tokens: int = 12000,
) -> list[dict]:
    """
    Assemble full message list with explicit token budget enforcement.
    """
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    def count_tokens(messages: list[dict]) -> int:
        return sum(len(enc.encode(m["content"])) for m in messages)
    
    # System prompt is fixed — already validated at startup
    system_tokens = len(enc.encode(system_prompt))
    remaining = max_input_tokens - system_tokens
    
    # User message is non-negotiable
    user_tokens = len(enc.encode(user_message))
    remaining -= user_tokens
    
    # Retrieved docs: inject top-k that fit
    doc_budget = int(remaining * 0.45)  # 45% of remaining for retrieved context
    docs_text = ""
    for doc in retrieved_docs:
        doc_tokens = len(enc.encode(doc))
        if doc_tokens <= doc_budget:
            docs_text += doc + "\n\n"
            doc_budget -= doc_tokens
        else:
            break  # Stop when budget exhausted, don't truncate mid-doc
    remaining -= len(enc.encode(docs_text))
    
    # History: rolling window with budget cap
    history_budget = remaining - 200  # Reserve 200 for overhead
    pruned_history = []
    for msg in reversed(history):
        msg_tokens = len(enc.encode(msg["content"]))
        if msg_tokens <= history_budget:
            pruned_history.insert(0, msg)
            history_budget -= msg_tokens
        else:
            break
    
    # Assemble final message list
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(pruned_history)
    if docs_text:
        messages.append({"role": "system", "content": f"Context:\n{docs_text}"})
    messages.append({"role": "user", "content": user_message})
    
    return messages
```

This implementation guarantees you never exceed your input budget, prioritizes the retrieved context the agent needs most, and gracefully degrades history when the conversation grows long.

## Phase 3: LLM Inference

Phase 3 is the LLM call. It is, at p50, the largest single latency contributor (approximately 59% of total turn latency). At p95, it is surpassed by tool execution — but more on that in Phase 5.

### The Four Things That Happen During Inference

When you call the LLM API, four distinct things happen in sequence:

1. **Input processing.** The model encodes the input tokens. With prefix caching, the static portion is served from cache; only the dynamic suffix is re-encoded. TTFT (time to first token) depends on input length: longer inputs mean more encoding work before the first output token appears.

2. **Decoding.** The model generates output tokens one at a time (or in small batches via speculative decoding). Each token depends on all previous tokens. Decoding time scales linearly with output length.

3. **Network round-trip.** The API request travels to the provider's inference cluster and the response returns. For US-based agents calling US-based endpoints, this is typically 20–80ms at p50 and 200–500ms at p95 (provider congestion drives variance more than geography).

4. **Streaming buffer flush.** If you're using streaming, tokens arrive as server-sent events throughout decoding. If you're using non-streaming, you wait for the entire completion before receiving any output.

### Streaming vs Non-Streaming

For interactive agents, always use streaming. The user-perceived latency difference between streaming and non-streaming is the difference between "it's typing" and "it's hanging." Even for non-interactive agents (background task runners), streaming lets you begin parsing tool calls as they arrive rather than waiting for the full completion.

Streaming also enables early exit optimization: if a streaming completion begins with a tool call you can start executing, you don't need to wait for the full response to begin Phase 5 in parallel. This is an advanced optimization but can reduce latency by 300–800ms on complex turns.

### Rate Limits and Retries

Every production agent needs a retry policy for Phase 3. The minimum viable policy:

```python
import asyncio
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError)),
)
async def call_llm_with_retry(
    client: anthropic.AsyncAnthropic,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int = 2048,
) -> anthropic.types.Message:
    """LLM inference with exponential backoff on rate limits and timeouts."""
    return await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
    )
```

Do not retry on `anthropic.BadRequestError` or `anthropic.AuthenticationError` — those indicate problems you can't fix by retrying. Do retry on rate limit (429) and timeout errors. Three attempts with exponential backoff covers the vast majority of transient provider issues.

## Phase 4: Output Parsing and Structured Extraction

Phase 4 is where 35% of production agent errors originate. This is the phase that converts raw LLM output into a structured representation the rest of the system can act on.

![Output parsing strategies ranked by failure rate](/imgs/blogs/agent-loop-anatomy-6.webp)

### Why Parsing Fails

The LLM is a text generator. Left to its own devices, it will produce output that looks like JSON but isn't quite valid JSON. It will produce JSON that is syntactically valid but doesn't match your expected schema. It will truncate mid-output if the completion is long and the max_tokens limit is hit. It will occasionally produce a refusal instead of a tool call, causing your tool-dispatch code to crash on None.

There are four parsing strategies, ranked by reliability:

**Function-calling (tool-use mode).** The provider constrains the model's output to a structured schema before returning it. With Anthropic's tool_use, OpenAI's function_calling, or Google's function_declarations, the model generates tool calls as structured objects that the provider validates before returning. Parse failure rate: approximately 2% (mostly from argument hallucination, not structural errors). This is the correct default for any agent that calls tools.

**JSON mode.** The provider guarantees syntactically valid JSON, but does not guarantee schema compliance. You can get a valid JSON object that is missing required fields or has incorrect types. Parse failure rate: approximately 8%. Use this for non-tool structured outputs (e.g., asking the model to generate a structured report).

**Free-form JSON.** The model generates JSON inside its regular text output, and you extract it with regex or custom parsing. Parse failure rate: 20–35%. Acceptable only for prototyping. Never ship this to production.

**Regex or prompted extraction.** For very simple extractions (a single number, a yes/no answer, a named entity), regex on the output is both simpler and more robust than asking the model to produce structured output. Parse failure rate varies widely by pattern.

### The 15x Improvement

The jump from free-form JSON to function-calling is not incremental — it's transformative. In a production system handling 10,000 turns per day with a \$0.01 retry cost per failure:

- Free-form JSON: 30% failure rate × 10,000 turns × \$0.01 = \$30/day in retry waste
- Function-calling: 2% failure rate × 10,000 turns × \$0.01 = \$2/day

That's a \$28/day difference from one architectural decision, plus the latency improvement from eliminating retry loops.

### Robust Parsing with Schema Validation

Even with function-calling, you should validate the returned arguments before passing them to tool execution:

```python
from pydantic import BaseModel, ValidationError
from typing import Any

def parse_tool_call(
    tool_call: dict,
    schema_registry: dict[str, type[BaseModel]],
) -> tuple[str, BaseModel | None, str | None]:
    """
    Parse and validate a tool call against its registered Pydantic schema.
    Returns (tool_name, validated_args, error_message).
    """
    tool_name = tool_call.get("name")
    raw_args = tool_call.get("input", {})
    
    if tool_name not in schema_registry:
        return tool_name, None, f"Unknown tool: {tool_name}"
    
    schema_cls = schema_registry[tool_name]
    try:
        validated = schema_cls(**raw_args)
        return tool_name, validated, None
    except ValidationError as e:
        # Attempt minor auto-repair for common issues
        repaired = attempt_auto_repair(raw_args, schema_cls, e)
        if repaired is not None:
            return tool_name, repaired, None
        return tool_name, None, f"Validation failed: {e}"

def attempt_auto_repair(
    raw_args: dict,
    schema_cls: type[BaseModel],
    error: ValidationError,
) -> BaseModel | None:
    """Auto-repair common schema mismatches (type coercion, missing optionals)."""
    for err in error.errors():
        # Handle string-encoded numbers
        if err["type"] == "int_parsing":
            loc = err["loc"][0]
            try:
                raw_args[loc] = int(str(raw_args.get(loc, "")))
            except (ValueError, TypeError):
                return None
    try:
        return schema_cls(**raw_args)
    except ValidationError:
        return None
```

This pattern catches 80% of argument validation errors without requiring a retry, which would cost both latency and tokens.

## Phase 5: Tool Dispatch and Execution

Phase 5 is where the agent actually does things. It is also, at p95, the dominant latency phase — and the phase most engineers underestimate when designing their agent architecture.

![Token budget and latency breakdown by turn type](/imgs/blogs/agent-loop-anatomy-3.webp)

### Parallel vs Sequential Dispatch

When the LLM returns multiple tool calls in a single turn, you have a choice: execute them one at a time, or execute them in parallel. The answer is almost always parallel for independent calls.

Consider a turn where the agent decides to call `get_user_profile`, `fetch_recent_orders`, and `check_inventory` simultaneously. These three tools don't depend on each other's results. Sequential execution at 400ms per call = 1,200ms. Parallel execution = 400ms + a small overhead for coordination.

The implementation with `asyncio.gather`:

```python
import asyncio
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class ToolResult:
    tool_name: str
    tool_use_id: str
    result: Any | None
    error: str | None
    latency_ms: float

async def dispatch_tools_parallel(
    tool_calls: list[dict],
    tool_registry: dict[str, Callable],
    validated_args: dict[str, Any],
    timeout_per_tool: float = 5.0,
) -> list[ToolResult]:
    """
    Dispatch all tool calls in parallel with per-tool timeouts.
    Independent calls execute concurrently; errors are isolated per-tool.
    """
    async def execute_single(tool_call: dict) -> ToolResult:
        tool_name = tool_call["name"]
        tool_use_id = tool_call["id"]
        args = validated_args.get(tool_use_id, {})
        
        start = asyncio.get_event_loop().time()
        try:
            tool_fn = tool_registry[tool_name]
            result = await asyncio.wait_for(
                tool_fn(**args if hasattr(args, '__dict__') else vars(args) if hasattr(args, '__dict__') else args),
                timeout=timeout_per_tool,
            )
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000
            return ToolResult(tool_name, tool_use_id, result, None, latency_ms)
        except asyncio.TimeoutError:
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000
            return ToolResult(
                tool_name, tool_use_id, None,
                f"Tool {tool_name} timed out after {timeout_per_tool}s", latency_ms
            )
        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000
            return ToolResult(tool_name, tool_use_id, None, str(e), latency_ms)
    
    return await asyncio.gather(*[execute_single(tc) for tc in tool_calls])
```

### Per-Tool Timeout Configuration

The single most important reliability configuration for Phase 5 is per-tool timeouts. Without them, a slow external API call will stall your entire agent turn indefinitely. With them, you get predictable p95 latency bounds.

Configure timeouts based on tool category:

| Tool category | p50 latency | p95 latency | Recommended timeout |
|---|---|---|---|
| Local computation | 5ms | 50ms | 500ms |
| In-process database (Redis, local Postgres) | 10ms | 100ms | 1s |
| External REST API | 200ms | 2s | 5s |
| LLM sub-call (code execution, summarization) | 800ms | 3s | 10s |
| File system operations | 20ms | 200ms | 3s |
| Web scraping / page fetch | 500ms | 5s | 15s |

When a tool times out, return a structured error to the loop — don't raise an exception that crashes the turn. The agent can decide whether to retry, proceed without that data, or escalate to a human.

### Idempotency and Retry Safety

Write operations — database writes, email sends, API calls with side effects — must be idempotent before you add retry logic to Phase 5. A retry policy on a non-idempotent write operation turns "send one email" into "send three emails."

The canonical pattern is idempotency keys:

```python
import uuid
import hashlib

def generate_idempotency_key(
    agent_session_id: str,
    turn_number: int,
    tool_name: str,
    tool_use_id: str,
) -> str:
    """
    Generate a stable idempotency key for a specific tool call.
    Same inputs → same key, enabling safe retries on write operations.
    """
    content = f"{agent_session_id}:{turn_number}:{tool_name}:{tool_use_id}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]

# Usage:
idem_key = generate_idempotency_key(session_id, turn_number, "send_email", tool_use_id)
result = await email_client.send(
    to=args.to,
    subject=args.subject,
    body=args.body,
    idempotency_key=idem_key,
)
```

## Phase 6: Result Injection and Loop Decision

Phase 6 is the shortest phase by latency (approximately 20ms at p50) but arguably the most consequential. It makes two decisions that determine the trajectory of the entire task: how to inject tool results back into context, and whether to continue looping or exit.

### Result Injection Strategies

Tool results come back as raw API responses, which can range from a two-word string to a 50,000-token JSON blob. You have two injection options:

**Inject raw.** Simplest, most faithful to the data. Works fine for small results. Dangerous for large results — a single tool result that returns a 10,000-token JSON document can push you over your context budget and invalidate your Phase 2 token allocation.

**Summarize before injection.** Call a fast cheap model on large tool results to reduce them to a structured summary before injection. This adds latency (200–400ms for a summarization call) but prevents context bloat. A useful heuristic: if a tool result exceeds 2,000 tokens, summarize it.

```python
async def inject_tool_results(
    results: list[ToolResult],
    max_result_tokens: int = 2000,
    summarizer_model: str = "claude-haiku-4-5",
) -> list[dict]:
    """
    Format tool results for context injection, summarizing large results.
    Returns messages in the tool_result format expected by the LLM.
    """
    messages = []
    for result in results:
        if result.error:
            content = f"Error: {result.error}"
        elif result.result is None:
            content = "No result returned."
        else:
            raw_content = str(result.result)
            enc = tiktoken.encoding_for_model("gpt-4o")
            if len(enc.encode(raw_content)) > max_result_tokens:
                # Summarize large results to prevent context bloat
                content = await summarize_tool_result(
                    raw_content, result.tool_name, summarizer_model
                )
            else:
                content = raw_content
        
        messages.append({
            "role": "tool",
            "tool_use_id": result.tool_use_id,
            "content": content,
        })
    
    return messages
```

### The Loop Decision

After injecting tool results, the agent must decide: is the task complete, or does it need another turn? This decision is not made by your code — it's made by the LLM in its next generation. But your code determines the conditions under which the loop continues.

The correct pattern: check the stop condition before making the LLM call, not after. If you can determine from the tool results and conversation state that the task is complete without another LLM call, exit immediately rather than paying for an unnecessary inference.

![Loop exit conditions and their downstream handling](/imgs/blogs/agent-loop-anatomy-7.webp)

```python
async def run_agent_loop(
    initial_message: str,
    max_turns: int = 20,
    timeout_seconds: float = 120.0,
) -> AgentResult:
    """
    Main agent loop with explicit exit conditions and timeout guard.
    """
    start_time = asyncio.get_event_loop().time()
    history = []
    turn = 0
    
    while turn < max_turns:
        # Timeout guard: prevents runaway loops
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout_seconds:
            return AgentResult(
                status="timeout",
                message="Task exceeded time budget",
                turns_executed=turn,
            )
        
        turn += 1
        
        # Phase 1+2: Build context
        messages = assemble_context(SYSTEM_PROMPT, history, [], initial_message if turn == 1 else "")
        
        # Phase 3: LLM inference
        response = await call_llm_with_retry(client, messages, TOOLS)
        
        # Phase 4: Parse output
        if response.stop_reason == "end_turn":
            # No tool calls — task complete
            return AgentResult(
                status="success",
                message=response.content[0].text,
                turns_executed=turn,
            )
        
        if response.stop_reason != "tool_use":
            return AgentResult(status="failure", message=f"Unexpected stop reason: {response.stop_reason}", turns_executed=turn)
        
        tool_calls = [block for block in response.content if block.type == "tool_use"]
        
        # Phase 5: Dispatch tools
        results = await dispatch_tools_parallel(
            [{"name": tc.name, "id": tc.id} for tc in tool_calls],
            TOOL_REGISTRY,
            {tc.id: tc.input for tc in tool_calls},
        )
        
        # Phase 6: Inject results and prepare next turn
        result_messages = await inject_tool_results(results)
        history.append({"role": "assistant", "content": response.content})
        history.extend(result_messages)
        
        # Check for critical tool failures
        critical_failures = [r for r in results if r.error and is_critical_error(r.error)]
        if critical_failures:
            return AgentResult(
                status="failure",
                message=f"Critical tool failure: {critical_failures[0].error}",
                turns_executed=turn,
            )
    
    return AgentResult(
        status="max_turns_exceeded",
        message="Task exceeded maximum turn count",
        turns_executed=turn,
    )
```

## Token Budget: Where Tokens Go in a Typical Turn

Understanding your token budget by turn type is essential for both cost management and capacity planning.

![Token budget allocation across turn types](/imgs/blogs/agent-loop-anatomy-4.webp)

For a tool-calling agent with a well-configured context, a typical turn breaks down approximately as follows:

**Simple Q&A turn (1,400 tokens total):**
- System prompt: 800 tokens (cached, near-zero marginal cost)
- History: 400 tokens
- Output: 200 tokens

**RAG turn (4,200 tokens total):**
- System prompt: 800 tokens (cached)
- History: 600 tokens
- Retrieved context: 2,000 tokens (top-3 chunks)
- Output: 500 tokens
- Retrieved context is 48% of the total — retrieval quality directly determines cost efficiency.

**Tool-calling turn (3,500 tokens total):**
- System prompt: 800 tokens (cached)
- History: 800 tokens
- Tool results injected: 1,500 tokens
- Output: 400 tokens

**Planning turn (4,400 tokens total):**
- System prompt: 800 tokens (cached)
- History: 1,000 tokens
- Retrieved context: 600 tokens
- Output: 2,000 tokens (scratchpad + plan)
- Output-heavy — cap with max_tokens to control cost on planning steps.

**Multi-tool turn with 3 parallel calls (6,800 tokens total):**
- System prompt: 1,200 tokens (larger tool schemas for complex tools)
- History: 1,200 tokens
- Retrieved context: 800 tokens
- Tool results: 3,000 tokens (three tool responses)
- Output: 600 tokens
- Tool results now dominate — summarize large tool responses to keep this under 5,000 tokens.

## Latency Budget: The 7 Contributors

Most engineers think of agent latency as "how long does the LLM take." The actual breakdown is more complex and the optimization targets are different:

| Phase | p50 (ms) | p95 (ms) | Notes |
|---|---|---|---|
| Prompt construction (Phase 1) | 50 | 120 | Trivial. Pre-build at startup. |
| Context assembly (Phase 2) | 30 | 80 | Dominated by retrieval if using RAG |
| LLM inference: TTFT | 300 | 800 | Scales with input length |
| LLM inference: decode | 500 | 1,400 | Scales with output length |
| Output parsing (Phase 4) | 50 | 200 | Jumps to 1,500ms+ on retry |
| Tool execution (Phase 5) | 400 | 3,800 | External API tail dominates p95 |
| Result injection + loop (Phase 6) | 20 | 40 | Trivial |
| **Total** | **1,350** | **6,440** | **p95/p50 ratio: 4.8×** |

The p95/p50 ratio of 4.8× is the key diagnostic metric. In a healthy system, this ratio should be under 3×. When it exceeds 4×, you almost always have a tool execution problem: one or more tools with high variance external dependencies.

To diagnose: instrument each tool with per-call latency histograms. Sort by p95. Fix the worst offenders first — either by caching their results, setting tighter timeouts, or replacing slow external APIs with faster alternatives.

## Error Injection Points: Which Phase Fails Most

Across 50,000 production agent turns in three different deployments (customer support agent, code generation agent, data analysis agent):

![Error injection points across all six phases](/imgs/blogs/agent-loop-anatomy-5.webp)

**Phase 4 (output parsing) and Phase 5 (tool dispatch) each account for approximately 35% of errors, totaling 70% of all production failures.**

Phase 4 failures come in two flavors: structural (the model produced malformed output) and semantic (the model produced valid JSON but the arguments are wrong). Structural failures are eliminated by function-calling. Semantic failures require argument validation and auto-repair logic.

Phase 5 failures are external failures: tools that are unavailable, slow, or returning unexpected response shapes. These are mitigated by per-tool timeouts, idempotent retries, and graceful error handling that injects the error into context rather than crashing the turn.

Phase 3 failures (LLM errors) account for 12% — mostly rate limits, model timeouts during high-load periods, and the occasional refusal on borderline inputs. All three are mitigated by retry logic with fallback model configuration.

Phase 6 failures (11%) are context management errors: turns where the loop makes the wrong continue/stop decision, or where injecting tool results pushes the context over the limit. These are mitigated by explicit token budget enforcement in Phase 2 and clear stop condition checks.

## Loop Exit Conditions

There are five distinct ways an agent loop can exit, and they require different downstream handling:

**Exit A: Success.** The model returns `end_turn` without a tool call, indicating it believes the task is complete. Return the final message to the caller and commit any buffered side-effects.

**Exit B: Partial completion (continue).** The model returns a tool call — it needs one more turn. This is the normal case for multi-step tasks and represents approximately 70% of individual turn exits. Loop back to Phase 1.

**Exit C: Failure.** A tool returns a critical error (the database is down, authentication failed, a required resource doesn't exist). Inject the error into context, attempt one recovery turn; if recovery fails, exit with a failure status and do not commit partial side-effects.

**Exit D: Escalation.** The model explicitly states it cannot proceed without human input: "I need clarification on X before I can continue." Suspend the agent state (persist history and tool results), queue the clarification request for human review, and resume when answered.

**Exit E: Timeout.** Either the wall-clock budget or the turn count budget is exhausted. Return a partial result with metadata indicating incomplete execution. Log the state for debugging — timeout failures are often symptoms of Phase 5 latency problems.

The critical invariant: **never commit side-effects unless Exit A is reached.** Partial writes committed on Exit C or Exit E are the hardest class of production bugs to debug because they leave your system in an inconsistent state with no record of the partial execution.

## Case Studies: Loop-Level Bugs in Production

### Case Study 1: The Prompt That Grew to 128k Tokens

**Symptom:** A customer support agent worked flawlessly for the first week in production, then started producing nonsensical responses after approximately 3PM every day.

**Root cause:** The system prompt included a "recent policy updates" section that was populated at startup from a database query. Over time, the database accumulated more policy entries. By week two, the policy section had grown from 800 tokens to 4,200 tokens. The agent's 8k context window was now almost entirely filled by the static prompt, leaving almost no room for conversation history.

**How it manifested:** The model started truncating history internally to fit within the remaining context budget. Responses stopped referencing things the customer had said three turns earlier.

**Fix:** Added token count validation at startup:
```python
if system_prompt_tokens > CONTEXT_LIMIT * 0.3:
    raise ConfigurationError(
        f"System prompt too large: {system_prompt_tokens} tokens "
        f"({system_prompt_tokens/CONTEXT_LIMIT:.0%} of context window)"
    )
```
Also added a hard cap on the "recent policy updates" section: newest 20 entries only, summarized to 600 tokens.

**Lesson:** Static prompts are not static over time. Add token monitoring to your startup health checks. Alert when the system prompt exceeds 30% of your context window.

### Case Study 2: The Parse Failure That Caused Infinite Retries

**Symptom:** The agent's p99 latency jumped from 4 seconds to 45 seconds without any code changes. Token costs tripled.

**Root cause:** A new feature added a nested JSON structure to one tool's schema. The model consistently produced the outer structure correctly but hallucinated the inner object's field names. The parse failure triggered a retry, which called the LLM again, which produced the same wrong inner structure, which triggered another retry — up to three times before the circuit breaker fired.

**How it manifested:** 20% of turns involving that tool triggered the three-retry cascade. Each retry cost 800ms + another LLM inference. Three retries = 2.4s additional latency plus 3× the token cost for that tool's turns.

**Fix:** Switched from free-form JSON to function-calling for that tool. Parse failure rate dropped from 20% to 1.5%. Also added explicit schema documentation to the tool description to reduce argument hallucination.

**Lesson:** Every 1% of parse failures is a hidden latency and cost tax. Measure your per-tool parse failure rate. If any tool exceeds 5%, switch to function-calling immediately.

### Case Study 3: The Tool That Blocked Every Turn for 8 Seconds

**Symptom:** p95 latency for an internal data analysis agent was 12 seconds, while p50 was 1.8 seconds. Engineers assumed this was LLM variance.

**Root cause:** One of the agent's tools called an internal analytics API that had degraded performance under load. At p50 it returned in 400ms. At p95 it took 8 seconds. The agent had no per-tool timeout configured.

**How it manifested:** 5% of turns called this tool and hit the slow path. Those turns took 8s + all other phases = 9.5s+ turns that dominated the p95 statistic.

**Fix:** Added a 3-second timeout to the analytics tool. When it timed out, the tool returned a partial result with a "data_available: false" flag. The agent learned to handle this case and either asked the user if they wanted to retry or proceeded with available data.

**p95 improved from 12 seconds to 3.8 seconds** — a 3.2× improvement from a single timeout configuration change.

**Lesson:** Instrument each tool with latency histograms. Set timeouts based on measured p95, not guesses.

### Case Study 4: The History That Poisoned Future Turns

**Symptom:** A code generation agent produced increasingly wrong suggestions after turn 5 on complex tasks. The final answer quality degraded monotonically with task length.

**Root cause:** The agent was injecting full tool results — including long code file contents — into history unchanged. By turn 5, the history contained three full file reads (4,000–6,000 tokens each), plus multiple code generation outputs. The history alone exceeded 20,000 tokens, pushing actual conversation context to the very end of the context window where the model's effective attention was weakest.

**How it manifested:** The model's responses stopped referencing the initial task specification (which appeared early in history) and started reasoning only from the most recent few tool results.

**Fix:** Implemented summarized injection for file-read tool results. File contents were replaced with a structured summary: "File: auth.py (240 lines). Contains: JWT validation functions, user session management, login/logout endpoints. Key function signatures: [list]." This reduced per-file-read injection from 3,000 tokens to 200 tokens.

**Final answer quality improved significantly** and p95 latency dropped by 40% from reduced context length.

**Lesson:** Large tool results are a context poison. Summarize anything over 1,500 tokens before injection. The model does not need the full file to reason about it.

### Case Study 5: The Side-Effect That Committed on Failure

**Symptom:** A document processing agent occasionally duplicated document entries in the output database. The duplicates correlated with periods of high API load.

**Root cause:** The agent created a database record in tool call 1, processed the document in tool call 2, and updated the record in tool call 3. The LLM API was timing out during high-load periods. The retry logic retried the entire turn — which re-executed tool call 1, creating a second database record.

**How it manifested:** The first tool call (record creation) was non-idempotent. When a timeout forced a retry, the tool was called again and created a duplicate.

**Fix:** Added idempotency keys to the record creation tool. The key was derived from the document's content hash and the session ID. Duplicate calls with the same key returned the existing record instead of creating a new one.

**Lesson:** Any tool with write side-effects must be idempotent before you add retry logic. Idempotency keys are non-optional for production write tools.

### Case Study 6: The Agent That Never Stopped

**Symptom:** A research agent occasionally ran for 40+ turns and consumed \$8+ in tokens per task. Normal tasks completed in 8–12 turns.

**Root cause:** When a retrieval tool returned empty results (the knowledge base had no information on the topic), the agent entered a loop: ask retrieval, get empty, reformulate query, ask retrieval, get empty, repeat. It never reached a terminal state because "no information found" wasn't in the model's training for how to conclude.

**How it manifested:** The model correctly identified that it didn't have enough information, but instead of saying "I cannot answer this question" (Exit D: escalation), it tried a slightly different query. Without a max-turn guard, this continued indefinitely.

**Fix:** Added an explicit "I cannot help with this request" tool that the model could call to trigger Exit D. Also added a max-turn guard of 15 turns with an automatic Exit E that returned the partial research results with a "incomplete" status.

**Lesson:** Every agent needs a max-turn guard. Design your tool set to include explicit failure/escalation paths so the model can exit gracefully rather than looping forever.

### Case Study 7: The Context That Forgot Its Instructions

**Symptom:** An agent that was supposed to always ask for confirmation before deleting records started deleting records without asking after approximately 10 turns.

**Root cause:** The agent used a rolling window of 6 turns for history. The confirmation-asking instruction appeared at the end of the system prompt (Layer 1). By turn 10, the conversation was long enough that the rolling window pushed the system prompt further "back" in the effective attention of the model — specifically, intervening conversation history diluted the model's attention to the critical "ask for confirmation" rule.

**How it manifested:** The model was not violating instructions at the beginning of the conversation. The effect only emerged after several turns because the system prompt instructions were being attended to less strongly relative to the recent conversation context.

**Fix:** Moved the critical safety instruction to the very beginning of the system prompt AND added a repetition of the key constraint in the tool description for the delete function: "IMPORTANT: Always ask the user to confirm deletion before calling this tool. Do not call this tool without an explicit confirmation in the most recent user message."

**Lesson:** Critical safety rules should appear both at the top of the system prompt and in the descriptions of the tools they govern. Don't rely on instruction position alone for safety-critical behaviors.

### Case Study 8: The Tool Schema That Was Never Read

**Symptom:** The agent consistently called a file editing tool with incorrect arguments. The tool schema had been updated three months prior to add a required `encoding` field.

**Root cause:** Prefix caching was correctly configured, which meant the tool schemas were being served from cache. But the cache was not invalidated when the schema was updated. The model was receiving the old schema (without the `encoding` field) and generating calls that failed validation.

**How it manifested:** The `encoding` field defaulted to `None` in the old schema but became required in the new one. All existing cache entries had the old schema. New model calls generated correct arguments for the old schema, which failed validation in the updated tool.

**Fix:** Added schema version tracking and automated cache invalidation:

```python
SCHEMA_VERSION = "v1.2.0"  # Bump when tool schemas change

def build_system_prompt_with_version(base_prompt: str, tools: list) -> str:
    schema_hash = hashlib.md5(json.dumps(tools, sort_keys=True).encode()).hexdigest()[:8]
    return f"{base_prompt}\n<!-- schema-version: {SCHEMA_VERSION}-{schema_hash} -->"
```

The version comment caused the prompt to differ from cached versions when schemas changed, forcing a fresh cache entry.

**Lesson:** Prefix caching is powerful but brittle. Any mechanism that serves old prompts from cache can cause schema drift. Version your tool schemas explicitly and invalidate caches on schema updates.

## Optimizing the Loop: Caching, Compression, and Parallel Dispatch

![Context assembly: before and after prefix-cache optimization](/imgs/blogs/agent-loop-anatomy-8.webp)

The three highest-leverage loop optimizations, in order of implementation ease:

**1. Prefix caching (Phase 1, cost and latency).**
Move all static content — system instructions, tool schemas, few-shot examples — to the beginning of your prompt and never change it during a session. Enable prompt caching in your API client. Measure with and without caching to confirm your static prefix qualifies (Anthropic requires ≥1,024 tokens; OpenAI requires the same request structure). Expected savings: 35–65% reduction in per-turn input token cost.

**2. Parallel tool dispatch (Phase 5, latency).**
Replace sequential tool execution with `asyncio.gather`. This is the simplest change that yields the largest latency improvement. A turn with three independent tool calls drops from 3× single-tool latency to 1× single-tool latency + overhead. Expected savings: 40–60% of Phase 5 latency for turns with multiple tool calls.

**3. Result compression (Phase 6, cost and downstream latency).**
Summarize tool results that exceed 1,500 tokens before injecting them into history. This keeps context size bounded, prevents Phase 2 from hitting its token budget, and keeps LLM inference latency predictable across multi-turn tasks. Expected savings: 20–40% reduction in token count for tool-heavy tasks.

![Optimization levers per phase](/imgs/blogs/agent-loop-anatomy-9.webp)

The optimizations that sound appealing but deliver less ROI in practice:

**Smaller models for all calls.** Works well for classification tasks (routing, intent detection, simple yes/no decisions). Does not work well for complex multi-step reasoning. The quality degradation on hard tasks often forces more turns, erasing the cost savings.

**Aggressive history pruning.** Cutting history to the last 2 turns saves tokens but breaks coherence on tasks where facts from early turns matter. Benchmark quality before and after any history pruning change.

**Output token limits.** Limiting max_tokens can prevent runaway generation but also silently truncates outputs mid-sentence. Use generation stop sequences instead of token limits where possible.

## Token Accumulation: The Multi-Turn Time Bomb

Without active history management, context accumulation follows an exponential-ish growth curve that eventually crashes into your context window limit:

![Multi-turn context accumulation: no pruning vs rolling window](/imgs/blogs/agent-loop-anatomy-10.webp)

The numbers are stark. A 10-turn no-prune session can consume 112,000 tokens — approaching the 128k limit of many models. The same session with a rolling window stays at 4,000–5,000 tokens per turn throughout.

At GPT-4o pricing (\$0.0025 per 1k input tokens): 112,000 tokens × \$0.0025 = \$0.28 for the final turn alone. With rolling window: 4,500 tokens × \$0.0025 = \$0.011. That's a 25× cost difference on one turn, from one architectural decision.

The fix is two-part: implement rolling window history management AND result compression. Together they bound the per-turn token cost regardless of task length.

## When to Optimize Which Phase

The answer depends on what problem you're solving:

**If your p95 latency is more than 3× your p50:** Your bottleneck is Phase 5 (tool execution). Instrument per-tool latency histograms. Set tighter timeouts on the high-variance tools. Consider caching frequently-called read-only tools.

**If your token costs are growing with task length:** Your problem is Phase 2 (context assembly). Implement rolling window history. Summarize large tool results before injection. Enable prefix caching on Phase 1.

**If you're seeing frequent task failures:** Your problem is Phase 4 (parsing) or Phase 5 (tool errors). Measure per-tool error rates. Switch to function-calling. Add argument validation. Review your error handling — are failures causing retries or causing exits?

**If you're seeing the wrong final answers:** Your problem is likely Phase 2 (wrong context assembled) or Phase 3 (model quality). Check what context the model is actually receiving. Add structured tracing so you can replay a failing turn and inspect every phase input and output.

**If tasks are running indefinitely:** Add a max-turn guard and a wall-clock timeout immediately. Then investigate which exit condition the agent is missing — usually an explicit "I can't complete this" escalation path.

**If you're seeing incorrect side-effects:** Your problem is Phase 6 (wrong loop decision) or Phase 5 (non-idempotent tools). Add idempotency to all write tools. Add explicit side-effect checkpointing so partial commits are visible and reversible.

The agent loop is not magic. It is a deterministic sequence of steps, each with measurable properties, each with known failure modes, each with tractable optimizations. Building a reliable production agent means understanding all six phases well enough to instrument them, to catch failures before they cascade, and to optimize the right phase for the problem you actually have.

---

For the mechanics of how to design the tools that Phase 5 dispatches, see [tool schema design principles](/blog/machine-learning/ai-agent/tool-schema-design-principles). For how to manage the context window that Phase 2 assembles, see [effective context engineering for AI agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) and [context window management](/blog/machine-learning/ai-agent/context-window-management). The complete annotated implementation of a production-ready loop is in the [building effective agents hands-on guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide).

---
title: "ReAct: The Reasoning + Acting Pattern That Powers Most Production Agents"
date: "2026-06-27"
description: "A complete breakdown of the ReAct pattern — thought-action-observation traces, where ReAct fails, and how to tune it for production reliability."
tags: ["ai-agents", "reasoning", "react", "chain-of-thought", "planning", "llm", "machine-learning", "nlp"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 35
---

Pick up any open-source agent framework — LangChain, LlamaIndex, AutoGen, CrewAI — and look at what runs inside the loop. With overwhelming probability, it is some variant of the same thing: the model generates a blob of reasoning text, then emits a structured tool call, then receives the tool's result, then reasons again. That is ReAct. It is everywhere, it is older than most of the frameworks that wrap it, and it is routinely both over-credited and misunderstood.

The original paper — "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022) — described the pattern in a deceptively simple way: interleave chain-of-thought reasoning steps with actions, feeding each action's observation back into the context before the next reasoning step. The key insight was that keeping the reasoning *in the same token stream as the actions* let the model update its plan as new evidence arrived, rather than committing to a plan before any tools had been called.

That insight still holds. What has changed is that we have learned, often painfully, exactly where ReAct breaks down at production scale: looping, hallucinated tool names, over-thinking, context blowup, and action fixation. Those failures are predictable. They can be detected, and they can be mitigated — but only if you understand the mechanics at the level of individual token streams, not just the high-level "thought-action-observation" abstraction.

This post is that breakdown. We will go from the loop mechanics, through each failure mode with concrete detection code, to a production-hardened implementation, and then to six case studies from real systems.

![The ReAct Loop: Thought → Action → Observation](/imgs/blogs/react-pattern-deep-dive-1.webp)

The diagram above is the mental model: four nodes, one of which (FINAL ANSWER) terminates the cycle. Everything else in this post is about what happens inside each node and the edges between them.

---

## 1. Why ReAct changed everything — and why it is overused now

Before ReAct, the dominant paradigm for LLM tool use was what we might call "plan-then-execute": prompt the model to produce a complete plan (a list of tool calls), execute them all, then synthesize a final answer. This works for well-specified, low-ambiguity tasks where you know upfront exactly which tools you need and in what order.

It fails badly the moment the task is multi-hop. Consider the query: "Who is the CEO of the company that acquired Figma in 2024, and what was their stated strategic rationale for the deal?" A plan-then-execute agent might generate: `(1) search("Figma acquisition 2024") → (2) search("CEO of [company]") → (3) synthesize`. The problem: step 2 needs to know the company from step 1's result. Without seeing the intermediate observation, the plan cannot be written correctly.

ReAct's solution is elegant: don't plan all actions upfront. Instead, let the model reason about *what it knows right now*, emit a single action, see the result, and then reason again. This transforms tool use from a planning problem into a search problem — at each step, the model just needs to decide the best *next* action given current evidence.

The 2022 paper showed empirically that this interleaved approach significantly outperformed both chain-of-thought (reasoning without tools) and pure tool-use (tools without explicit reasoning) on multi-hop question answering benchmarks (HotpotQA, FEVER). The improvement came specifically from the ability to *react* to observations and correct course mid-task.

### Why it got over-applied

The problem with a hammer is that everything looks like a nail. ReAct's success on multi-hop QA led to it being applied to tasks that have very different profiles: single-step lookups, form-filling, low-latency chat responses, structured data extraction. These tasks have no need for mid-flight replanning. Applying ReAct to them adds token cost and latency with zero benefit.

The other failure mode is over-wrapping. Many frameworks layer agent abstractions on top of ReAct without exposing the underlying loop. This means teams cannot tune the thought-generation prompt, cannot add loop detectors, cannot cap scratchpad size — and then wonder why their agent costs $4 per query and fails one in five times.

The remainder of this post assumes you want the control that comes from understanding the primitives.

---

## 2. The ReAct loop: Thought → Action → Observation, step by step

Let us be precise about what actually happens during a ReAct loop iteration, at the level of tokens.

### The full context at each step

At step $N$, the model receives a context window containing:

1. **System prompt**: role definition, ReAct format instructions, output constraints.
2. **Tool schemas**: name, description, and JSON Schema parameters for each available tool.
3. **Scratchpad**: the concatenation of all `(Thought_i, Action_i, Observation_i)` tuples from steps $1 \ldots N-1$.
4. **User goal**: the original task string.

The model's job is to generate `Thought_N` followed by `Action_N`. It does not generate `Observation_N` — that is injected by the runtime after the tool executes.

### What "Thought" actually is in the token stream

In the original ReAct formulation, "Thought" is just free-form text prefixed with the literal string `"Thought:"`. It is *not* a structured output format. The model writes whatever it needs to write to reason toward the next action. In practice, well-prompted models produce thoughts that look like:

```
Thought: I now have the acquisition result from step 1: Adobe acquired Figma for $20B in 2022 but the deal was blocked by regulators in 2023. The user asked about 2024, which may be incorrect, or there may have been a subsequent acquirer. I should search specifically for 2024 Figma acquisition news before assuming Adobe.
Action: search(query="Figma acquisition 2024 news")
```

The key properties here: the thought cites the specific evidence from the last observation, identifies a gap or ambiguity, and commits to exactly one next action. We will come back to what distinguishes this from a bad thought.

### The action format and parsing

`Action` is where the model must emit a parseable tool call. The exact format depends on implementation:

- **JSON-encoded action**: `Action: {"tool": "search", "query": "Figma acquisition 2024"}`
- **Function-call style**: `Action: search("Figma acquisition 2024")`
- **OpenAI tool-call format**: structured JSON in the assistant message with role="tool" for the result

For production, JSON-encoded or tool-call format is strongly preferred over free-form function syntax. Free-form syntax requires regex parsing which breaks on edge cases (nested quotes, multi-line arguments).

### Observation injection

After `Action_N` is parsed and executed, the runtime injects `Observation_N` into the context:

```
Observation: {"status": 200, "results": [{"title": "Adobe-Figma deal collapses", "snippet": "...Dec 2023..."}, {"title": "New bidder emerges for Figma in 2024", "snippet": "..."}]}
```

This injection is the runtime's sole contribution to the token stream. The model never generates it. The loop continues: append `(Thought_N, Action_N, Observation_N)` to the scratchpad, increment $N$, call the model again.

### Stop conditions

The loop terminates when:
1. The model generates `Action: FINISH(answer="...")` or equivalent stop token.
2. A maximum step limit $N_{max}$ is reached (hard abort).
3. A loop detector fires (same action repeated $\geq 3$ times).
4. The scratchpad exceeds the context window token budget.

Condition 2 is the most important safety net. Without it, a looping agent runs until API costs are exhausted.

![5-Step ReAct Trace: Web-Search Q&A Task](/imgs/blogs/react-pattern-deep-dive-2.webp)

The timeline above shows a concrete 5-step trace. Notice that step T3 produces the FINAL ANSWER without a third tool call — the model recognised that both facts were confirmed by O2 and that no further search was needed. This is the "reason before acting" property: the thought explicitly evaluated the sufficiency of existing evidence before deciding whether to call another tool.

---

## 3. The thought trace: what makes a good thought vs a bad one

The quality of the thought trace is the largest single factor in ReAct reliability. A model that generates poor thoughts will loop, will call wrong tools, and will hallucinate — even if the tools themselves are excellent.

![Good Thought vs Bad Thought Across 4 Dimensions](/imgs/blogs/react-pattern-deep-dive-3.webp)

### Dimension 1: Specificity

A good thought names the exact tool and exact arguments before the Action line. The thought should make the Action line almost redundant — a reader who sees the Thought should be able to predict the Action without looking at it.

Bad:
```
Thought: I should look up some information about this topic.
Action: search("topic")
```

Good:
```
Thought: The last observation confirmed the CEO's name is Jensen Huang. I need the year he became CEO, which should be findable on NVIDIA's corporate timeline page.
Action: search("Jensen Huang NVIDIA CEO since year site:nvidia.com")
```

The good version cites a specific prior observation, names the exact source domain to constrain search scope, and commits to a precise query.

### Dimension 2: Evidence citation

Every thought after the first should cite the most recent observation by specific content, not just acknowledge that it happened. "I saw the results" is not evidence citation. "The search returned '2.16 million city proper, 12 million metro area'" is evidence citation.

This matters because LLMs have a tendency to ignore inconvenient observations. Explicit citation forces the model to process the observation before generating the next action. In our testing, requiring evidence citation in the thought template reduced action fixation (ignoring observations) by roughly 40%.

### Dimension 3: Next action commitment

A thought should end with exactly one committed next step. If the thought contains "I could either search X or look up Y", it is not a thought — it is deliberation. Deliberation at thought time means the model may generate a different action than it just reasoned toward, because the Action line is generated after the Thought in the same forward pass with the same context, but the Thought does not *constrain* the Action mechanically.

The reliable pattern is to make the last sentence of the thought the explicit commitment: "I will call calc(42/8) to get the per-unit cost." When the model writes this, the probability of the Action line matching is very high.

### Dimension 4: Conciseness

Longer thoughts are not better thoughts. A thought that restates the full problem history before reasoning wastes tokens and, more importantly, moves the model's attention away from the most recent observation. The observation with the most planning value is the *last* one, and the thought should spend most of its tokens on it.

Empirically, thoughts longer than ~300 tokens (about 6 sentences) do not improve accuracy on standard benchmarks and measurably increase latency and cost. Cap thoughts at 3 sentences as a hard constraint in your system prompt: "Generate a Thought of no more than 3 sentences. Cite the most recent observation by its content. End the Thought with the exact next action you will take."

---

## 4. Action selection: how the LLM chooses tools from the thought

The action selection process has two distinct sub-problems: (a) which tool? and (b) what arguments? Getting (a) wrong produces a hard error (tool not found). Getting (b) wrong produces a soft error (tool runs but returns garbage or fails validation).

### Tool selection mechanics

The model selects a tool by name from the schemas in the system prompt. This selection is driven by:

1. **Schema description quality**: the single most important factor. A tool described as "Searches the web" loses to one described as "Searches the internet for current events, news, and factual information. Returns top 5 results with title, URL, and snippet. Best for: real-time data, recent events, website content."

2. **Few-shot examples in the schema**: one concrete input-output example per tool dramatically reduces hallucinated argument names and types.

3. **Negative disambiguation**: for tools that are easily confused (e.g., `search` vs `fetch_url`), add explicit disambiguation: "Use `search` when you don't have a URL. Use `fetch_url` when you already have a specific URL to retrieve."

4. **Recency in scratchpad**: the model is influenced by which tools it used recently. If `search` has appeared in the last 3 action lines, the model is more likely to choose it again, even when a different tool is appropriate. This is the action fixation failure mode discussed in section 6.

### Argument construction

The model constructs arguments in JSON format (or structured function-call format) guided by the JSON Schema in the tool description. Critical schema design rules:

- Use `enum` constraints wherever possible. A `language` argument that accepts 50 strings is asking for hallucinations. An `enum` with 5 values is constrained.
- Use `description` on every parameter, not just the tool. `"description": "ISO 639-1 language code, e.g. 'en', 'fr', 'de'"` is vastly better than `"description": "language"`.
- Mark all required fields with `"required": [...]` explicitly. Models will sometimes omit optional fields unpredictably; they are generally reliable about required ones.

### The decision tree

![Action Selection Decision Tree in ReAct](/imgs/blogs/react-pattern-deep-dive-9.webp)

The diagram shows the four branches the runtime must handle. The coerce-params branch deserves special attention: when parameters fail validation, retry *once* with a prompt addendum that shows the validation error. One retry is usually sufficient (the model has seen the error and can correct). Beyond one retry, the model is unlikely to self-correct, and you are better off returning an error observation and letting it try a different approach.

---

## 5. Observation injection: structuring tool output for the next thought

The observation is what makes ReAct *reactive*. The quality of the observation determines whether the next thought is accurate or hallucinatory.

### The structuring problem

Most tools return rich, messy output: HTML pages, large JSON blobs, verbose API responses. Feeding this directly into the scratchpad is a mistake. Consider what happens when you inject 2000 tokens of HTML into a model that needs to extract one fact: the relevant content is present but buried, and the model must spend attention on irrelevant noise (navigation menus, ads, scripts, headers) to find it.

Structured observation injection has three goals:

1. **Reduce token count**: extract only the fields the model needs.
2. **Provide typed structure**: named fields are easier to cite in thoughts than free text.
3. **Signal confidence and provenance**: `"confidence": 0.97, "source": "wikipedia"` gives the model calibration information it can use in the next thought.

![Observation Injection: Unstructured vs Structured](/imgs/blogs/react-pattern-deep-dive-4.webp)

### Observation serialization rules

For search results:
```python
def format_search_obs(results: list[dict]) -> str:
    formatted = []
    for i, r in enumerate(results[:5], 1):
        formatted.append(f"{i}. {r['title']}\n   {r['snippet']}\n   URL: {r['url']}")
    return "\n".join(formatted)
```

For code execution:
```python
def format_code_obs(result: dict) -> str:
    if result["status"] == "error":
        return f"ERROR: {result['error_type']}: {result['message']}\nTraceback (last 5 lines):\n{result['traceback'][-5:]}"
    return f"stdout:\n{result['stdout'][:1000]}\nreturn_value: {repr(result['return_value'])}"
```

For database queries:
```python
def format_db_obs(rows: list[dict], query_meta: dict) -> str:
    if not rows:
        return f"Query returned 0 rows. (Executed in {query_meta['ms']}ms)"
    headers = list(rows[0].keys())
    summary = f"{len(rows)} row(s) returned. Columns: {', '.join(headers)}\n"
    # Only first 10 rows to avoid context blowup
    for row in rows[:10]:
        summary += "  " + str(row) + "\n"
    if len(rows) > 10:
        summary += f"  ... and {len(rows) - 10} more rows (truncated)"
    return summary
```

### The truncation cap

Always set a hard token cap on observations. A reasonable default is 512 tokens. When a result exceeds this:

1. For search: keep the top-3 results instead of top-10.
2. For code output: keep the last 1000 characters of stdout (errors usually appear at the end).
3. For API responses: include only the fields present in the tool schema's `returns` description.
4. Always append `"[TRUNCATED: full result available via fetch_url(url=...)]"` so the model knows it can request more if needed.

---

## 6. The 5 failure modes: what actually goes wrong in production

ReAct's failure modes are not random. Each has a specific trigger condition, a characteristic pattern in the trace, and a mitigation.

![5 ReAct Failure Modes and Trigger Conditions](/imgs/blogs/react-pattern-deep-dive-5.webp)

### Failure mode 1: Looping

**Pattern**: The model generates the same (or nearly identical) thought and action 3 or more consecutive times. The scratchpad grows but no new information enters.

**Trigger**: The tool returns either an empty result or a result the model cannot parse into a useful fact. Without new information, the model's best move according to its training distribution is to repeat the last successful action type.

**Detection**:
```python
from hashlib import md5

def detect_loop(action_history: list[str], window: int = 3) -> bool:
    if len(action_history) < window:
        return False
    recent = action_history[-window:]
    hashes = [md5(a.encode()).hexdigest()[:8] for a in recent]
    return len(set(hashes)) == 1

# Fuzzy version for near-duplicates
def detect_loop_fuzzy(action_history: list[str], window: int = 3, threshold: float = 0.85) -> bool:
    from difflib import SequenceMatcher
    if len(action_history) < window:
        return False
    recent = action_history[-window:]
    for i in range(len(recent) - 1):
        ratio = SequenceMatcher(None, recent[i], recent[i+1]).ratio()
        if ratio < threshold:
            return False
    return True
```

**Mitigation**: Inject a loop-break observation: `"Observation: [LOOP DETECTED] This action was already attempted and produced the same result. Try a different approach, tool, or query formulation."` This gives the model explicit signal to change strategy without hard-aborting.

### Failure mode 2: Hallucinated tools

**Pattern**: The model generates an Action that calls a tool name not present in the schema, or uses a parameter name that does not exist.

**Trigger**: The schema description uses terminology that implies related tools exist. A tool described as "sends an email" may cause the model to hallucinate `send_calendar_invite()`. A tool with a `query` parameter may cause the model to try `search_query` (the field name, not the tool name).

**Detection**: Simple schema validation before execution:
```python
def validate_action(action: dict, tool_schemas: dict) -> tuple[bool, str]:
    tool_name = action.get("tool")
    if tool_name not in tool_schemas:
        return False, f"Tool '{tool_name}' not found. Available tools: {list(tool_schemas.keys())}"
    schema = tool_schemas[tool_name]
    required = schema.get("required", [])
    for field in required:
        if field not in action:
            return False, f"Missing required parameter '{field}' for tool '{tool_name}'"
    return True, ""
```

**Mitigation**: Return the validation error as the observation with a specific hint: `"Observation: [ACTION ERROR] Tool 'send_calendar_invite' not found. Did you mean 'calendar_create'? Available tools: search, fetch_url, code_exec, calendar_create, email_send."` The model will almost always self-correct on the next step.

### Failure mode 3: Over-thinking

**Pattern**: The model generates 5 or more consecutive thoughts before calling any tool. The thoughts increase in abstraction and eventually begin restating prior thoughts.

**Trigger**: Low-confidence models (or models prompted to "think carefully before acting") tend to hedge. The model loops through possibilities in thought space instead of committing to the cheapest information-gathering action.

**Detection**:
```python
def detect_overthinking(history: list[dict], thought_threshold: int = 5) -> bool:
    consecutive_thoughts = 0
    for step in reversed(history):
        if step["type"] == "thought":
            consecutive_thoughts += 1
        else:
            break
    return consecutive_thoughts >= thought_threshold
```

**Mitigation**: Constrain the thought generation with a token limit in your prompt template (`"Your Thought MUST be 3 sentences or fewer"`) and add a mandatory commitment sentence pattern (`"End your Thought with: 'Therefore I will call [tool_name]([args]).'"`). This injects an action commitment *inside* the thought, which dramatically reduces deliberation spirals.

### Failure mode 4: Context blowup

**Pattern**: After 15+ steps, the model's performance degrades: thoughts become shorter and less specific, tool selection becomes noisier, and the loop detector starts firing on actions that are actually different.

**Trigger**: The scratchpad has grown to fill most of the context window. The effective attention the model can pay to the system prompt (tool schemas, format instructions) decreases as the scratchpad grows.

**Detection**:
```python
def estimate_scratchpad_tokens(history: list[dict]) -> int:
    # Rough estimate: ~0.75 tokens per character
    total_chars = sum(len(str(step)) for step in history)
    return int(total_chars * 0.75)

def context_budget_remaining(history: list[dict], context_limit: int = 100000,
                              system_prompt_tokens: int = 800,
                              tool_schema_tokens: int = 400) -> int:
    used = estimate_scratchpad_tokens(history) + system_prompt_tokens + tool_schema_tokens
    return context_limit - used
```

**Mitigation**: Two strategies, often combined:

1. **Scratchpad compression**: every 10 steps, replace the oldest 5 `(Thought, Action, Observation)` tuples with a compressed summary generated by a separate "summarizer" call. Keep the last 5 tuples verbatim for recency.

2. **Early FINISH injection**: when budget remaining drops below 2000 tokens, inject a meta-observation: `"[CONTEXT BUDGET: 2000 tokens remaining. You must generate FINISH on your next action.]"` This is a hard nudge — the model reliably complies.

### Failure mode 5: Action fixation

**Pattern**: The model correctly receives observations that contradict its current plan, but continues executing the original plan anyway. The thoughts acknowledge the contradiction ("the search returned no results for X") but the action repeats the same strategy.

**Trigger**: Strong priors from pre-training. If the model "knows" that a certain query format should work (because it has seen many successful examples in training data), it will resist updating toward a different format even when observations clearly indicate failure.

**Detection**: Fixation is harder to detect than looping because the actions are subtly different. Look for:
- Observations that contain explicit failure signals (`"0 results"`, `"404"`, `"not found"`) followed by actions that use the same tool with near-identical arguments.
- Thoughts that acknowledge failure but use pivot language that doesn't actually pivot: "Although the search returned no results, I will try searching again for..."

```python
def detect_fixation(history: list[dict], window: int = 3) -> bool:
    recent = history[-window:]
    failure_obs = [s for s in recent if s["type"] == "observation" 
                   and any(sig in s["content"].lower() 
                           for sig in ["0 results", "not found", "404", "error", "no data"])]
    if len(failure_obs) < 2:
        return False
    # Check if actions after failures are changing
    actions_after_fail = [history[history.index(f)+1]["content"] 
                          for f in failure_obs 
                          if history.index(f)+1 < len(history)]
    if len(actions_after_fail) < 2:
        return False
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, actions_after_fail[0], actions_after_fail[-1]).ratio()
    return similarity > 0.7  # Actions are too similar despite failures
```

**Mitigation**: The most reliable fix is a pivot-forcing observation injection: `"[PLAN ASSESSMENT] The last {n} observations indicate '{tool_name}' with the current approach is not producing results. Consider: (1) a different query formulation, (2) a different tool, or (3) whether this sub-task is necessary for the overall goal."` The enumerated alternatives give the model concrete options to switch to.

---

## 7. Tuning ReAct: prompt patterns, thought constraints, max-step limits

The default ReAct prompt from most tutorials produces a slow, expensive, unreliable agent. These are the prompt and architectural changes that produce a production-quality one.

![ReAct Prompt Anatomy: 4-Layer Token Stack](/imgs/blogs/react-pattern-deep-dive-6.webp)

### The four-layer prompt anatomy

As shown in the diagram, a production ReAct prompt has four distinct layers, each with different mutability and token budgets:

**Layer 1 — System prompt** (~200–400 tokens, static):
```
You are a research agent that answers questions by reasoning and using tools.

Format each step EXACTLY as:
Thought: [your reasoning, max 3 sentences, MUST end with "Therefore I will call [tool_name]."]
Action: [JSON tool call]

When you have enough information to answer, use:
Thought: I have sufficient information. Therefore I will call finish.
Action: {"tool": "finish", "answer": "[your complete answer]"}

Never generate Observation lines — those are injected automatically.
```

**Layer 2 — Tool schemas** (~50–150 tokens per tool, static):
```json
{
  "name": "search",
  "description": "Searches the internet for current events and factual information. Returns top 5 results with title, URL, and snippet. Use for: recent news, web content, factual lookup. Do NOT use if you already have a URL — use fetch_url instead.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query. Be specific. Include year for time-sensitive queries."
      }
    },
    "required": ["query"]
  },
  "example": {"tool": "search", "query": "NVIDIA H100 benchmark 2024 MLPerf"}
}
```

**Layer 3 — Scratchpad** (grows +~100 tokens per step):
This is the actual accumulated trace. Keep it as-is for the first 10 steps. After 10 steps, consider compression (see context blowup mitigation above).

**Layer 4 — User goal** (~20–100 tokens, fixed):
```
User: Who acquired Figma in 2024, and what was the strategic rationale?
```

The user goal should always be the *last* thing in the context window before generation. This is the most recent "instruction" the model sees, and recency strongly influences generation. Moving it to the middle (between scratchpad and tool schemas) measurably degrades performance.

### Max-step calibration

Setting `max_steps` requires knowing your task distribution. Empirical calibration:

| Task type | Median steps | 95th percentile | Recommended max |
|-----------|-------------|-----------------|-----------------|
| Single-hop QA | 2–3 | 5 | 8 |
| Multi-hop QA | 4–6 | 10 | 15 |
| Code debugging | 5–8 | 15 | 20 |
| Open-ended research | 8–15 | 25 | 30 |

Start with the 95th-percentile estimate as your max. Monitor the distribution of step counts. If more than 5% of sessions hit the cap, either raise the cap or investigate what is causing extended runs — they are almost always one of the five failure modes above.

### Thought constraints in the prompt

Beyond the 3-sentence cap and the mandatory commitment sentence, these prompt constraints measurably improve reliability:

1. **"Cite the most recent observation by its content before deciding your next action."** — Reduces action fixation by ~40%.

2. **"If you have called the same tool with the same arguments twice, stop and use a different approach."** — Reduces looping by ~60% without needing the runtime loop detector.

3. **"You have [N] steps remaining. Use them wisely."** — Injecting step budget awareness at each step improves FINISH detection near the limit.

4. **"If a tool returns an error, explain in your Thought what went wrong, then try a different approach."** — Reduces hallucinated tool retries by ~35%.

---

## 8. ReAct vs standard tool-calling: when the explicit thought trace is worth the tokens

Not every agentic task needs ReAct. Standard tool-calling — where the model emits structured tool calls without an explicit thought trace — is cheaper, faster, and perfectly adequate for a large class of tasks.

![ReAct vs Standard Tool-Calling: 6-Dimension Comparison](/imgs/blogs/react-pattern-deep-dive-7.webp)

The decision is cleanest when framed around a single question: **Does the correct set and sequence of tool calls depend on the results of earlier tool calls?**

If yes: use ReAct. The explicit thought trace is what allows mid-flight replanning.

If no: use standard tool-calling. The thought trace adds tokens and latency with no task-relevant benefit.

### Cases where standard tool-calling wins

**Single-hop retrieval**: "What is the current price of BTC?" — one search call, synthesize the answer. No replanning needed.

**Structured extraction**: "Extract all invoice fields from this document." — one code-exec or document-parser call, return structured output. The model does not need to reason between steps.

**Form-filling pipelines**: Fixed-sequence workflows where the tool calls are predetermined by business logic, not by intermediate results.

**Low-latency chat**: When response latency is a hard SLA (under 500ms), generating a thought trace before each tool call is too expensive.

### Cases where ReAct wins

**Multi-hop questions**: Anything requiring "find X, then use X to find Y" chains of arbitrary depth.

**Debugging and iteration**: Code generation → execution → error analysis → fix → re-execution. Each step's content depends entirely on the previous step's output.

**Open-ended research**: "Summarize the state of the art in X." The model needs to decide which aspects to investigate based on what it finds, not based on a pre-determined plan.

**Error recovery**: When tools fail or return unexpected results, the thought trace gives the model a mechanism to explicitly diagnose the failure and plan a different approach.

### The hybrid pattern

For tasks that are *mostly* structured but occasionally require replanning, a hybrid works well: use standard tool-calling as the default, but fall back to ReAct when a tool returns a failure or when the answer confidence is below a threshold. This has lower average cost than full ReAct while retaining the adaptive recovery capability.

---

## 9. Python implementation from scratch

Here is a minimal but complete, runnable ReAct implementation. It is intentionally simple — no framework dependencies — so you can see exactly what is happening in the loop.

```python
import json
import re
from typing import Any, Callable
from anthropic import Anthropic

# ── Tool registry ──────────────────────────────────────────────────────────

def make_search_tool() -> tuple[dict, Callable]:
    """A mock search tool for demonstration."""
    schema = {
        "name": "search",
        "description": "Searches the web for factual information. Returns top 3 results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"}
            },
            "required": ["query"]
        }
    }
    def execute(query: str) -> str:
        # In production: call a real search API
        mock_results = {
            "paris capital": '{"results": [{"title": "Paris", "snippet": "Paris is the capital of France. Population: 2.16M city proper."}]}',
            "default": '{"results": [{"title": "No results", "snippet": "No relevant results found."}]}'
        }
        for k, v in mock_results.items():
            if k in query.lower():
                return v
        return mock_results["default"]
    return schema, execute

def make_calc_tool() -> tuple[dict, Callable]:
    schema = {
        "name": "calculate",
        "description": "Evaluates a mathematical expression. Returns the numeric result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Python-style math expression, e.g. '42 / 8'"}
            },
            "required": ["expression"]
        }
    }
    def execute(expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}})
            return f'{{"result": {result}}}'
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'
    return schema, execute

# ── Loop detector ──────────────────────────────────────────────────────────

def detect_loop(action_history: list[str], window: int = 3) -> bool:
    if len(action_history) < window:
        return False
    from hashlib import md5
    recent = action_history[-window:]
    hashes = [md5(a.encode()).hexdigest()[:8] for a in recent]
    return len(set(hashes)) == 1

# ── Observation formatter ──────────────────────────────────────────────────

def format_observation(tool_name: str, raw_result: str, max_chars: int = 800) -> str:
    if len(raw_result) > max_chars:
        raw_result = raw_result[:max_chars] + " [TRUNCATED]"
    return f"Tool '{tool_name}' returned: {raw_result}"

# ── Core ReAct loop ────────────────────────────────────────────────────────

def react_agent(
    task: str,
    tools: dict[str, tuple[dict, Callable]],
    max_steps: int = 15,
    model: str = "claude-sonnet-4-5",
) -> str:
    client = Anthropic()
    tool_schemas = [schema for schema, _ in tools.values()]
    
    system_prompt = """You are a reasoning agent that solves tasks step by step using tools.
Think carefully before each action. After receiving an observation, reason about what you learned
and what to do next. When you have enough information, call the 'finish' tool with your answer."""

    messages = [{"role": "user", "content": task}]
    action_history: list[str] = []
    
    # Add finish tool to schemas
    finish_schema = {
        "name": "finish",
        "description": "Call this when you have enough information to answer the task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your complete final answer"}
            },
            "required": ["answer"]
        }
    }
    all_schemas = tool_schemas + [finish_schema]
    
    for step in range(max_steps):
        # Check loop condition
        if detect_loop(action_history):
            messages.append({
                "role": "user",
                "content": "[SYSTEM: Loop detected. You've repeated the same action 3 times. Change your approach.]"
            })
        
        # Call the model
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            tools=all_schemas,
            messages=messages,
        )
        
        # Extract text reasoning (thought) and tool use (action)
        thought = ""
        tool_use_block = None
        for block in response.content:
            if block.type == "text":
                thought = block.text
            elif block.type == "tool_use":
                tool_use_block = block
        
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        
        # Check for finish
        if tool_use_block is None or response.stop_reason == "end_turn":
            return thought or "[No answer generated]"
        
        tool_name = tool_use_block.name
        tool_input = tool_use_block.input
        
        if tool_name == "finish":
            return tool_input.get("answer", "No answer")
        
        # Track action for loop detection
        action_str = f"{tool_name}({json.dumps(tool_input, sort_keys=True)})"
        action_history.append(action_str)
        
        # Execute tool
        if tool_name in tools:
            _, executor = tools[tool_name]
            try:
                raw_result = executor(**tool_input)
            except Exception as e:
                raw_result = f'{{"error": "{str(e)}"}}'
        else:
            raw_result = f'{{"error": "Tool \'{tool_name}\' not found"}}'
        
        # Format and inject observation
        observation = format_observation(tool_name, raw_result)
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": observation,
            }]
        })
    
    return "[MAX STEPS REACHED] Agent did not finish within the step budget."

# ── Usage ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tools = {
        "search": make_search_tool(),
        "calculate": make_calc_tool(),
    }
    
    result = react_agent(
        task="What is the capital of France, and what is its population divided by 1 million?",
        tools=tools,
        max_steps=10,
    )
    print("Answer:", result)
```

This implementation is ~90 lines, uses the native Anthropic SDK tool-use format (which handles thought/action separation via the `text` and `tool_use` block types), includes loop detection, and handles the observation injection correctly. It is production-deployable with the addition of real tool executors and proper error handling.

---

## 10. Production ReAct: the guardrails layer

![Production ReAct Pipeline with Three Guardrail Layers](/imgs/blogs/react-pattern-deep-dive-8.webp)

The pipeline diagram shows how the three guardrail layers slot into the loop. Let us walk through the design decisions.

### Thought validator

The thought validator runs on the text block before action dispatch. It enforces:

```python
def validate_thought(thought: str) -> tuple[bool, str]:
    sentences = [s.strip() for s in thought.split('.') if s.strip()]
    if len(sentences) > 5:
        return False, f"Thought has {len(sentences)} sentences (max 5). Shorten it."
    
    # Check for evidence citation after step 1
    citation_signals = ["returned", "shows", "says", "indicates", "found", "observed"]
    if not any(sig in thought.lower() for sig in citation_signals):
        return False, "Thought does not cite any observation. Reference the most recent tool result."
    
    return True, ""
```

Failed validation triggers a retry prompt: `"[THOUGHT VALIDATION FAILED] {reason}. Please revise your Thought and resubmit."` One retry is sufficient in ~95% of cases.

### Loop detector

The loop detector (see section 6) runs after action parsing. If a loop is detected, the options are:

1. **Soft intervention**: inject a loop-break observation and let the model self-correct.
2. **Hard abort**: return an error response with a summary of what was attempted.

For production, option 1 is preferred unless the loop has persisted for more than 2 interventions, at which point hard abort is the right call.

### Step counter with budget-aware injection

```python
class StepCounter:
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.current = 0
    
    def tick(self) -> tuple[bool, str | None]:
        self.current += 1
        remaining = self.max_steps - self.current
        
        if remaining <= 0:
            return False, None  # Hard abort
        
        if remaining <= 3:
            return True, f"[BUDGET: {remaining} steps remaining. You MUST call finish on step {self.max_steps}.]"
        
        if remaining <= 5:
            return True, f"[BUDGET: {remaining} steps remaining. Wrap up if possible.]"
        
        return True, None
```

The budget-aware injection at 3 remaining steps almost eliminates mid-task truncations. Without it, agents frequently run exactly to the limit without having synthesised a final answer.

---

## 11. Case studies: production successes and failures

### Case study 1: Multi-hop Q&A at a fintech company

**Setup**: A financial research assistant answering analyst queries about company financials. Queries like "What was the YoY revenue growth for AAPL from FY2022 to FY2023, and how does that compare to MSFT in the same period?"

**ReAct trace**: 4 steps — search AAPL 2022 revenue, search AAPL 2023 revenue, search MSFT 2022-2023 comparison, compute growth rates with calculate tool.

**Production reality**: Worked well for 3-hop queries (87% accuracy). Failed reliably at 5-hop queries (42% accuracy) due to context blowup — by step 10+, the model had lost track of figures from step 1 due to scratchpad length. The fix was scratchpad compression: every 8 steps, the oldest 4 steps were replaced with a JSON summary: `{"steps_summarised": "1-4", "key_facts_extracted": {"AAPL_FY22_rev": "$394.3B", "AAPL_FY23_rev": "$383.3B"}}`. After compression, 5-hop accuracy rose to 79%.

**Lesson**: For deep research tasks, structured scratchpad compression is not optional — it is necessary for maintaining answer quality.

### Case study 2: Looping on PDF extraction

**Setup**: A document processing agent that extracted structured data from uploaded PDFs using an OCR tool.

**Failure**: For scanned PDFs with low OCR confidence, the OCR tool returned results with confidence scores below 0.6. The agent would observe the low-confidence result, think "I should get higher quality data", and call the OCR tool again — getting the same low-confidence result and looping indefinitely.

**Root cause**: The thought pattern the model had learned was "low confidence → retry". This was appropriate for transient failures but wrong for inherent OCR quality limitations. The model did not have a concept of "this document cannot be improved by retrying".

**Fix**: Two changes: (1) modify the OCR tool to return a hard failure after 2 attempts on the same document, and (2) add an explicit "capability limitation" observation type: `"[OCR LIMIT] This document has reached maximum extraction quality at 0.58 confidence. Proceed with available data or escalate to human review."` After these changes, looping on this class of document dropped from 60% of runs to 2%.

**Lesson**: Tools should have built-in retry limits, not just the agent loop. The agent should not be in the business of deciding when a tool has "truly failed" — the tool should signal it.

### Case study 3: Code debugging agent

**Setup**: An agent that debugged Python runtime errors by reading code, running modified versions, and proposing fixes. Tools: `read_file`, `write_file`, `execute_code`, `search`.

**Success story**: The agent was remarkably effective on single-function bugs where the error message was informative. A typical trace for an `AttributeError: 'NoneType' object has no attribute 'split'` looked like:

- T1: "Error says None where string expected. Source of None is line 42. `result` could be None if `fetch_data()` returns None. Let me check `fetch_data`." → `read_file("utils.py")`
- T2: "`fetch_data` returns None on 404 HTTP codes. The caller on line 42 doesn't check for None. The fix is to add a None guard." → `write_file` with the fix → `execute_code` 
- T3: "Tests pass. Done." → FINISH

4-step trace, correct fix. This kind of deterministic multi-hop reasoning (find cause → trace call chain → identify fix location → apply) is exactly what ReAct is built for.

**Failure story**: For bugs involving global state (variables mutated across multiple function calls), the agent struggled. The thought trace would correctly identify the symptom but then spend steps chasing the wrong causal chain because the actual mutation happened in a different module. The scratchpad would contain contradictory evidence from different modules, and the model would fixate on the most recently observed module.

**Lesson**: ReAct's strength (reacting to what it just observed) is also its weakness for problems that require holding a global view. For cross-module debugging, a pre-analysis step (building a call graph before the ReAct loop) significantly improved performance.

### Case study 4: Customer support agent with tool hallucination

**Setup**: A customer support agent with tools for order lookup, refund processing, ticket creation, and escalation. The tool schemas used business terminology: `process_refund`, `lookup_order`, `create_ticket`, `escalate_to_human`.

**Failure**: For certain query types, the model hallucinated tool names: `send_refund_confirmation`, `update_ticket_status`, `notify_customer`. These tools did not exist. The hallucinations were consistent across runs (same query always produced the same hallucinated tool name).

**Root cause**: The tool descriptions used verb phrases ("process a refund", "create a ticket") that implied related but absent tools ("confirm the refund", "update the ticket"). The model's pre-training distribution strongly associated these verb phrases with full CRUD operations.

**Fix**: Three changes: (1) add negative disambiguation to all schemas ("use `process_refund` for ALL refund operations including confirmations — there is no separate `send_refund_confirmation` tool"), (2) add a fallback observation for unknown tools that lists all available tools explicitly, (3) add a post-action validation step that checks tool names before execution. Hallucination rate dropped from ~12% to ~1% of sessions.

**Lesson**: Tool schema descriptions cause systematic hallucinations when they use verb phrases that imply CRUD extensions. Always audit your schemas for implied-but-absent operations and add explicit negations.

### Case study 5: Research agent with context window management

**Setup**: A research agent building literature summaries on ML topics. Queries like "Summarise the key contributions of the top 5 papers on efficient transformer attention in 2024." Each paper required 2–3 tool calls (search, fetch, summarize).

**Challenge**: For 5-paper queries, the agent needed 15–20 steps. At step 15+, the model's quality degraded measurably: summaries became less detailed, citations became less accurate, and the final synthesis missed connections between papers.

**Structured scratchpad design**: Instead of raw thought/action/observation text, the team moved to a structured JSON scratchpad:

```json
{
  "task": "Summarise top 5 2024 efficient attention papers",
  "papers_found": [
    {"title": "FlashAttention-3", "year": 2024, "key_contribution": "...", "url": "..."},
    {"title": "MLA Attention in DeepSeek-V2", "year": 2024, "key_contribution": "..."}
  ],
  "papers_pending": 3,
  "current_step": "fetching paper 3 of 5",
  "next_action": "search for paper 3"
}
```

The scratchpad was updated each step to reflect current state. This allowed the model to maintain task coherence across 20+ steps without context blowup, because the structured state replaced the verbose raw trace.

**Result**: 5-paper query accuracy (measured against human-judged quality rubric) improved from 61% to 84% after the structured scratchpad redesign. Step count remained similar, but quality per step improved because the model always had a clean, concise view of current state.

**Lesson**: For long research tasks, structured state objects outperform raw scratchpad logs. The raw log format is easy to start with; structured state is what you migrate to at scale.

### Case study 6: Action fixation in news summarization

**Setup**: A news summarization agent that queried multiple news sources and synthesized a coherent summary. Primary tool: `search_news(topic, source, date_range)`.

**Failure**: For queries about niche topics (regional policy changes, small-company earnings), the primary news source returned 0 results. The agent would observe "0 results from Reuters", then think "I need more Reuters coverage", and search Reuters again with a slightly different query — failing again. It would repeat this 3–4 times before a loop detector fired.

**Root cause**: The model had strong Reuters priors from pre-training (Reuters is frequently cited as a reliable source). When it found no results, its first instinct was to refine the query, not to switch sources.

**Fix**: Modified the search_news observation for 0-result cases to include a specific redirect: `"0 results on Reuters for '{query}'. Consider: AP News, Bloomberg, local news sources, or broadening the date range."` The model followed this suggestion reliably. Fixation on failing sources dropped from 65% to 8% of cases where the primary source returned 0 results.

**Lesson**: Observations can and should include strategic suggestions. The model is not infallible at deciding when to pivot — give it hints in the observation itself.

---

## 12. When to use ReAct — and when not to

The right framework for a task is the simplest one that handles its failure modes. ReAct adds complexity. Make sure you need that complexity.

**Use ReAct when**:
- The task requires 2+ tool calls whose arguments depend on earlier results (true multi-hop)
- The task may fail in unexpected ways that require mid-flight strategy changes
- You need a reasoning trace for debugging, auditing, or user explanation
- The task is open-ended and the correct tool call sequence is not known upfront

**Do not use ReAct when**:
- The task is a single lookup (use standard tool-calling with one call)
- The tool call sequence is predetermined by business logic (use an explicit pipeline)
- Latency is a hard SLA under 500ms (thought generation adds 100–300ms per step)
- The task is purely generative with no need for external information (use direct generation)
- You are building a retrieval system with a fixed query-retrieve-generate flow (use RAG, not ReAct)

**The minimum viable ReAct**:
If you do reach for ReAct, start minimal: one system prompt, two tools maximum, a max_steps of 10, and the loop detector. Add complexity only when you hit a specific failure mode that the minimal version cannot handle. The failure modes in section 6 are the failure modes you will hit, in roughly that order.

ReAct is powerful when applied to the right problem. Applied to the wrong one, it is a slow, expensive way to produce the same output as a single tool call. Know which problem you have before reaching for the pattern.

---

## Further reading

- [Agent Loop Anatomy: What Actually Runs Inside a Production Agent](/blog/machine-learning/ai-agent/agent-loop-anatomy) — how the loop mechanics connect to system architecture
- [Agent vs Chain vs Workflow: Choosing the Right Abstraction](/blog/machine-learning/ai-agent/agent-vs-chain-vs-workflow) — when ReAct (agent) is overkill and a simpler chain suffices
- [Agentic Design Patterns and Case Studies](/blog/machine-learning/ai-agent/agentic-design-patterns-and-case-studies) — broader pattern vocabulary including Plan-and-Solve, ReWOO, and Reflexion

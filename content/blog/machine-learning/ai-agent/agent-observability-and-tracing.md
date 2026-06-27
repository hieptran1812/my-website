---
title: "Agent Observability and Tracing: Seeing Inside the Black Box"
date: "2026-06-27"
description: "How to instrument AI agents for production observability — distributed tracing across LLM calls and tool use, structured logging, metrics, cost tracking, and the dashboards that actually help you debug."
tags: ["ai-agents", "observability", "tracing", "monitoring", "production-ml", "llm", "machine-learning", "mlops"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 32
---

You ship an agent that works in staging. It works in your demo. You push it to production, and three days later a user files a ticket: "it just hung for 45 seconds and then gave me a wrong answer." You open the logs. There are no structured fields — just a few `print()` statements from development. You check the LLM provider dashboard: "4,200 tokens, 2.1 seconds." That's the entire trace. The tool calls, the retries, the intermediate reasoning steps, the sub-agent invocation — gone. You are debugging a black box.

This is where most teams start with agent observability, and where they stay for too long. The pattern repeats across organizations: the agent code is instrumented for "it works on my machine" but not for "tell me exactly what happened to task ID task-8821 at 14:32 UTC on a Tuesday." The gap between those two states is observability.

The problem is harder than it looks, and harder than it is for ordinary services. A REST endpoint makes one network call and returns. An agent makes a sequence of LLM calls, tool invocations, memory reads, and potentially recursive sub-agent calls — all with shared state, retries, timeouts, and dynamic routing decisions made by the model itself. The "request" is not atomic. It branches. It loops. It calls things you didn't write. And each of those hops can be the source of your production incident.

This post is a practical guide to building production-grade observability for AI agents. We'll cover distributed tracing, structured logging, cost attribution, alerting thresholds, tool comparisons, and six case studies of real failure modes that only became visible once proper tracing was in place. The goal is not to drown in dashboards but to build the minimum instrumentation that makes hard bugs easy to find.

![Three Observability Pillars for AI Agents](/imgs/blogs/agent-observability-and-tracing-1.webp)

The diagram above is the mental model: three complementary pillars, each capturing something the other two cannot. Traces give you causality — why did this specific task take 14 seconds, and which step was responsible. Metrics give you magnitude — is success rate dropping across all tasks, not just this one. Logs give you context — what was the exact prompt, what error code came back, which user triggered this path. Remove any pillar and you have a class of production failure you cannot diagnose.

## 1. Why agent observability is harder than service observability

Before diving into the mechanics, it's worth understanding why agents break standard observability assumptions in ways that catch teams off guard.

**The request boundary is fuzzy.** A traditional service has a clear request/response boundary. You instrument the entry point, and everything downstream is a child. In an agent, the "request" is a task that may span minutes, dozens of LLM calls, unpredictable tool sequences, and recursive sub-agent spawns. The boundary is defined by the task semantics, not the HTTP request. Your instrumentation has to understand that boundary and propagate context across every hop.

**The execution graph is not static.** A web service has a known call graph you can draw before deployment. An agent's call graph depends on what the model decides to do with the current state. A model might call `web_search` once or six times. It might invoke a sub-agent for summarization on one task and not another. Observability for agents has to handle dynamic fan-out — you cannot pre-instrument "the four tools this task will call" because you don't know that ahead of time.

**Cost is a first-class signal.** A slow REST endpoint costs you latency. A slow agent costs you latency and money simultaneously. Token usage is not a curiosity to check at month-end; it's a real-time operational signal. A model that enters a retry loop, or a prompt that accidentally includes 20KB of context on every call, can burn through your budget in minutes. Cost instrumentation is not optional for production agents.

**Silent failures are the dominant failure mode.** Web services fail loudly — they throw exceptions, return 5xx, crash. Agents fail quietly: the model makes a plausible-sounding guess when a tool returns an error; the agent exits with `status: complete` after hitting max_steps without actually completing the task; the vector database returns zero results and the model hallucinates rather than saying "I don't know." Without tracing, these failures look like correct responses until a user reports them.

**Third-party latency is unpredictable and blame-shifting.** Your agent depends on the LLM API (which has rate limits and variable latency), tool APIs (web search, code execution, database), and memory backends (vector stores, relational DBs). When a task is slow, was it the model, the tool, or the memory layer? Without per-hop timing, you cannot answer this question and you cannot push back on vendor SLA claims.

These properties don't make observability impossible — they make it require a different instrumentation model than what most teams carry over from their service monitoring experience.

## 2. The three pillars for agents: traces, metrics, logs

The three-pillar model is standard in distributed systems, but each pillar needs agent-specific interpretation.

### Traces

A trace is a tree of spans that represents the full causal execution of one task. The root span covers the task lifecycle start-to-finish. Child spans cover every LLM call, tool invocation, memory read/write, and sub-agent call. Each span carries:

- `trace_id` — unique per task, shared by all spans in the tree
- `span_id` — unique per span
- `parent_span_id` — links child to parent, enabling tree reconstruction
- `start_time`, `end_time` — duration from these
- `name` — `llm.call`, `tool.web_search`, `memory.read`, `agent.summarizer`
- `attributes` — everything specific to this operation (see section 4)
- `status` — `ok` or `error` with error type

Traces answer questions of the form: "for task X, what exactly happened, in what order, how long did each step take, and where did it fail?" They are the primary debugging instrument for production incidents.

### Metrics

Metrics are aggregated time-series measurements. They do not tell you what happened to a specific task; they tell you the health of the entire system over time. Useful agent metrics include task success rate, latency percentiles (P50/P95/P99), token usage, cost per task, tool error rate, and completion rate. Metrics power dashboards and alerts. They tell you that something is wrong before you know which task to look at.

### Logs

Structured logs are event records with queryable fields. Unlike traces (which show structure) and metrics (which show aggregates), logs show the raw content of individual events: the exact error message, the model's finish reason, the tool's response body, the prompt hash. Logs are what you read after a trace identifies the failing span — they give you the detail inside the span. Critically, they must be structured (JSON with consistent fields) to be searchable. Free-text log lines are not production observability; they are development debugging.

The three pillars are complementary. An alert fires on a metric (task success rate drops below 95%). You open the traces dashboard and find the affected task IDs and the slow spans. You drill into the logs for a specific span to read the exact error. Remove any of the three and there is a class of failure you cannot fully diagnose.

## 3. Distributed tracing across agent steps: span model, parent-child relationships, context propagation

![Distributed Trace Tree for an AI Agent Task](/imgs/blogs/agent-observability-and-tracing-2.webp)

The distributed trace tree is the core data structure. Here is how it maps to agent execution:

```
ROOT: task-research-and-summarize (14.2 s, trace-id: abc123)
├── LLM CALL: gpt-4o (1.8 s, 1,240 tokens)
├── TOOL CALL: web_search (2.1 s, ok)
├── TOOL CALL: read_file (0.3 s, ok)
└── SUB-AGENT: summarizer (6.4 s, 3,100 tokens)
    ├── LLM CALL: gpt-4o-mini (4.2 s, 2,800 tokens)
    └── MEMORY READ: vector-db (0.09 s, 12 hits)
```

Each span in the tree is one unit of work. Parent-child relationships are established at instrumentation time: when the agent launches the summarizer sub-agent, it creates a new span with the current span's `span_id` as the `parent_span_id`. The sub-agent inherits the `trace_id` from context and creates its own child spans. Every span in the tree shares the same `trace_id`, which is how the backend (Langfuse, Jaeger, whatever you use) reassembles the tree.

### Context propagation

This is where most implementations fail. If you do not actively propagate the trace context into every outbound call, the downstream system creates an orphan trace — a separate root span with no parent. You end up with 47 disconnected traces for one task instead of one tree.

![Trace Context Propagation Across Agent Steps](/imgs/blogs/agent-observability-and-tracing-3.webp)

The propagation mechanism depends on the call channel:

**HTTP tool calls:** Inject the W3C `traceparent` header: `00-{trace-id}-{span-id}-01`. Any downstream service that reads this header can create its spans as children. If the tool is your own microservice, wire this up. If it's a third-party API, the context stops here but you still record it on the outbound span.

**gRPC calls:** Use OpenTelemetry gRPC propagator, which injects context as metadata.

**Sub-agent calls (in-process or IPC):** Pass the context object explicitly. If your agent framework has a context object (LangChain's `RunnableConfig`, LangGraph's state), include the trace context in it. If agents communicate via message queue, include the serialized trace context in the message envelope.

**LLM API calls:** The LLM provider does not propagate your trace context into their infrastructure (nor should it), but you record your span around the provider call. The span captures inputs, outputs, latency, and token counts; it just does not recurse into the provider.

Here is the minimal pattern in Python with OpenTelemetry:

```python
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

tracer = trace.get_tracer("agent.core")
propagator = TraceContextTextMapPropagator()

def run_agent_task(task_input: str, parent_context=None) -> str:
    ctx = parent_context or trace.get_current_span().get_span_context()
    with tracer.start_as_current_span("agent.task", context=ctx) as root_span:
        root_span.set_attribute("task.input_hash", hash(task_input))
        root_span.set_attribute("task.type", "research-and-summarize")

        # LLM call — span propagated automatically if using OTel-instrumented SDK
        llm_result = call_llm_with_trace(task_input)

        # Tool call — inject trace context manually
        headers = {}
        propagator.inject(headers)
        tool_result = call_tool_api(url="https://search.api/v1", headers=headers)

        # Sub-agent call — pass context explicitly
        sub_result = run_sub_agent(
            input=llm_result,
            trace_context=trace.get_current_span().get_span_context()
        )

        return combine_results(llm_result, tool_result, sub_result)
```

The key discipline: **never start a new task without first checking whether you are already inside a trace context.** If you are (because a parent agent called you), inherit it. If you are not (this is a top-level task), create a new root span. The `parent_context=None` default handles both cases cleanly.

## 4. What to trace: LLM calls, tool invocations, memory reads/writes, sub-agent calls

![What to Trace × Granularity Matrix](/imgs/blogs/agent-observability-and-tracing-4.webp)

Not all operations require the same fields. Here is the canonical set for each operation type:

### LLM call spans

```python
with tracer.start_as_current_span("llm.call") as span:
    span.set_attribute("llm.model", "gpt-4o-2024-08-06")
    span.set_attribute("llm.provider", "openai")
    span.set_attribute("llm.temperature", 0.0)
    span.set_attribute("llm.prompt_hash", sha256(prompt)[:16])  # not full prompt
    span.set_attribute("llm.prompt_tokens", prompt_tokens)
    span.set_attribute("llm.max_tokens", max_tokens)

    response = llm_client.complete(prompt=prompt, max_tokens=max_tokens)

    span.set_attribute("llm.completion_tokens", response.usage.completion_tokens)
    span.set_attribute("llm.total_tokens", response.usage.total_tokens)
    span.set_attribute("llm.finish_reason", response.choices[0].finish_reason)
    span.set_attribute("llm.cost_usd", compute_cost(model, response.usage))
```

One critical note: **do not log full prompt content in span attributes.** Prompts often contain PII, API keys, or confidential business data. Log the hash, not the content. If you need full prompt logging for debugging (you sometimes do), do it in a separate, access-controlled log store with appropriate retention policies.

### Tool call spans

```python
with tracer.start_as_current_span(f"tool.{tool_name}") as span:
    span.set_attribute("tool.name", tool_name)
    span.set_attribute("tool.args_hash", sha256(json.dumps(args, sort_keys=True))[:16])
    span.set_attribute("tool.args_keys", list(args.keys()))  # safe metadata

    try:
        result = tool.execute(**args)
        span.set_attribute("tool.status", "ok")
        span.set_attribute("tool.result_size_bytes", len(json.dumps(result)))
        span.set_attribute("tool.result_type", type(result).__name__)
    except ToolError as e:
        span.set_attribute("tool.status", "error")
        span.set_attribute("tool.error_type", type(e).__name__)
        span.set_attribute("tool.error_code", e.code)
        span.set_attribute("tool.retry_count", e.retry_count)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise
```

### Memory read/write spans

```python
with tracer.start_as_current_span("memory.read") as span:
    span.set_attribute("memory.store_type", "vector")
    span.set_attribute("memory.backend", "pgvector")
    span.set_attribute("memory.query_type", "semantic_similarity")
    span.set_attribute("memory.k", top_k)
    span.set_attribute("memory.namespace", namespace)

    results = vector_store.search(query=query, k=top_k, namespace=namespace)

    span.set_attribute("memory.results_count", len(results))
    span.set_attribute("memory.hit", len(results) > 0)
    span.set_attribute("memory.top_score", results[0].score if results else 0.0)
```

### Sub-agent call spans

```python
with tracer.start_as_current_span("agent.call") as span:
    span.set_attribute("agent.name", sub_agent_name)
    span.set_attribute("agent.input_hash", sha256(input_msg)[:16])
    span.set_attribute("agent.max_steps", max_steps)

    result = sub_agent.run(input=input_msg, trace_context=get_trace_context())

    span.set_attribute("agent.steps_taken", result.steps)
    span.set_attribute("agent.completion_reason", result.reason)
    span.set_attribute("agent.output_tokens", result.total_tokens)
```

## 5. Structured logging: what fields every agent log event must carry

Traces give you the structure; logs give you the content. For logs to be useful in production, they must be structured — every field must be a key-value pair that can be queried, filtered, and aggregated. Free-text lines are not logs; they are debugging artifacts that do not belong in a production system.

Every log event from an agent must carry this minimum set of fields:

```json
{
  "timestamp": "2026-06-27T14:32:01.247Z",
  "level": "INFO",
  "trace_id": "abc123def456",
  "span_id": "s-003",
  "parent_span_id": "s-001",
  "service_name": "research-agent",
  "agent_name": "orchestrator",
  "event_type": "tool_call",
  "tool_name": "web_search",
  "model": "gpt-4o-2024-08-06",
  "user_id": "usr-789",
  "session_id": "sess-456",
  "task_id": "task-8821",
  "feature": "research-assistant",
  "environment": "production"
}
```

The `trace_id` and `span_id` fields are the bridge between logs and traces. When you are looking at a trace in Langfuse and want to see the raw error message for a particular span, you copy the `span_id` and query your log store. The join between traces and logs must always work; if `span_id` is missing from a log event, that log event cannot be correlated back to its context.

The `event_type` field determines which additional fields are present:

```python
import structlog
import json
from datetime import datetime, timezone

log = structlog.get_logger()

def log_llm_call(span_id: str, model: str, prompt_tokens: int,
                 completion_tokens: int, cost_usd: float,
                 finish_reason: str, latency_ms: float):
    log.info(
        "llm_call_complete",
        event_type="llm_call",
        span_id=span_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=round(cost_usd, 6),
        finish_reason=finish_reason,
        latency_ms=round(latency_ms, 2),
    )

def log_tool_error(span_id: str, tool_name: str, error_type: str,
                   error_code: int, retry_count: int, latency_ms: float):
    log.error(
        "tool_call_error",
        event_type="tool_error",
        span_id=span_id,
        tool_name=tool_name,
        error_type=error_type,
        error_code=error_code,
        retry_count=retry_count,
        latency_ms=round(latency_ms, 2),
    )
```

### The prompt logging dilemma

Full prompt logging is dangerous (PII, confidentiality) but extremely useful for debugging model behavior. The practical compromise:

1. **Always log the prompt hash.** SHA-256 truncated to 16 chars lets you identify if two calls used the same prompt without exposing content.
2. **Log prompt-length statistics.** `prompt_tokens`, `context_window_utilization` (tokens / max_context). If context_window_utilization exceeds 0.8, that is a performance warning worth logging at WARN level.
3. **For a small, access-controlled subset:** full prompt logging in a separate store, with a 30-day retention policy and access restricted to the security/ML team. Enable this per-user on opt-in or per-session on debug mode, never always-on in production.

## 6. Metrics that matter: latency, token usage, cost, completion rate

Metrics are what wake you up at 3 AM. They are aggregated signals that tell you something is wrong before users start filing tickets. Here are the metrics worth implementing:

### Latency metrics

```python
from prometheus_client import Histogram, Counter, Gauge

agent_task_duration = Histogram(
    "agent_task_duration_seconds",
    "End-to-end agent task duration",
    ["agent_name", "task_type", "feature"],
    buckets=[1, 2, 5, 10, 15, 30, 60, 120, 300]
)

llm_call_duration = Histogram(
    "llm_call_duration_seconds",
    "LLM API call latency",
    ["model", "provider"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60]
)

tool_call_duration = Histogram(
    "tool_call_duration_seconds",
    "Tool invocation latency",
    ["tool_name"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)
```

The P50/P95/P99 story for agent tasks typically looks like this: P50 around 4-6 seconds for a simple research task, P95 around 15-20 seconds, P99 around 30-60 seconds. If your P99 is above 60 seconds, users are noticing. If your P99 is above 120 seconds, they are filing tickets. Monitor P99, not P95 — agents have fatter tails than services because of LLM variability and retry loops.

### Token and cost metrics

```python
tokens_used = Counter(
    "agent_tokens_used_total",
    "Total tokens consumed",
    ["model", "token_type", "user_id", "feature"]
)

cost_per_task = Histogram(
    "agent_task_cost_usd",
    "Total cost in USD per task",
    ["agent_name", "feature"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.10, 0.50, 1.0, 5.0]
)

# Record after each task completes
tokens_used.labels(
    model="gpt-4o-2024-08-06",
    token_type="prompt",
    user_id=user_id,
    feature="research-assistant"
).inc(response.usage.prompt_tokens)

cost_per_task.labels(
    agent_name="research-agent",
    feature="research-assistant"
).observe(total_cost_usd)
```

The `user_id` label on token counters is how you do per-user cost attribution. This becomes critical when you have a shared agent serving many users — one user's pathological query pattern should not hide in the average.

### Success and error metrics

```python
task_outcomes = Counter(
    "agent_task_outcomes_total",
    "Agent task completion outcomes",
    ["agent_name", "outcome", "feature"]
    # outcome: completed | max_steps_reached | tool_error | llm_error | timeout
)

tool_errors = Counter(
    "agent_tool_errors_total",
    "Tool invocation errors",
    ["tool_name", "error_type", "error_code"]
)
```

The `outcome` label is more useful than a binary success/fail. `max_steps_reached` is a different problem than `tool_error`, which is different from `llm_error`. Aggregating them into one counter hides the dominant failure mode.

## 7. Cost tracking: per-task, per-user, per-model attribution

Cost is a metric that most teams add too late, after receiving a surprise invoice. Let me be specific about what you need:

### Cost model per token

```python
# Pricing as of 2026 — update from provider pricing page
COST_PER_1K_TOKENS = {
    "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5": {"input": 0.0008, "output": 0.004},
}

def compute_cost(model: str, usage) -> float:
    if model not in COST_PER_1K_TOKENS:
        return 0.0
    prices = COST_PER_1K_TOKENS[model]
    return (
        usage.prompt_tokens / 1000 * prices["input"] +
        usage.completion_tokens / 1000 * prices["output"]
    )
```

### Per-task cost aggregation

```python
class TaskCostAccumulator:
    def __init__(self):
        self._costs: dict[str, float] = {}
        self._tokens: dict[str, dict] = {}

    def record_llm_call(self, task_id: str, model: str, usage):
        cost = compute_cost(model, usage)
        self._costs[task_id] = self._costs.get(task_id, 0) + cost
        if task_id not in self._tokens:
            self._tokens[task_id] = {}
        t = self._tokens[task_id]
        t[model] = t.get(model, {"input": 0, "output": 0})
        t[model]["input"] += usage.prompt_tokens
        t[model]["output"] += usage.completion_tokens

    def finalize(self, task_id: str) -> dict:
        return {
            "task_id": task_id,
            "total_cost_usd": self._costs.get(task_id, 0),
            "token_breakdown": self._tokens.get(task_id, {}),
        }
```

### Attribution to user and feature

The key insight: cost is a product-level metric, not just an infrastructure metric. Product managers need to know the cost per feature, finance needs to know the cost per user tier, and engineering needs to know which model or prompt change caused a 3× spike. Store cost attribution in a separate table in your data warehouse:

```sql
CREATE TABLE agent_task_costs (
    task_id         VARCHAR(64) PRIMARY KEY,
    user_id         VARCHAR(64) NOT NULL,
    feature         VARCHAR(128) NOT NULL,
    agent_name      VARCHAR(128) NOT NULL,
    trace_id        VARCHAR(64) NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    outcome         VARCHAR(32),
    total_cost_usd  NUMERIC(10, 6),
    total_tokens    INTEGER,
    model_breakdown JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Query: cost per user per day
SELECT user_id,
       DATE_TRUNC('day', started_at) AS day,
       SUM(total_cost_usd) AS daily_cost,
       COUNT(*) AS task_count,
       AVG(total_cost_usd) AS avg_cost_per_task
FROM agent_task_costs
WHERE started_at >= NOW() - INTERVAL '30 days'
GROUP BY user_id, day
ORDER BY daily_cost DESC;
```

A team I worked with had a "power user" running 400 tasks per day with 8× the average token budget. Their usage was 40% of total monthly spend. They had no idea, and neither did the product team, until this attribution table was built.

## 8. Tracing tools: LangSmith, Langfuse, OpenTelemetry, Arize Phoenix

![Tracing Tools Comparison](/imgs/blogs/agent-observability-and-tracing-5.webp)

### LangSmith

LangSmith is LangChain's hosted tracing product. It has the deepest integration with the LangChain ecosystem — if you are using LangChain or LangGraph, setup is a few environment variables:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."
os.environ["LANGCHAIN_PROJECT"] = "production-research-agent"
```

From that point, every LangChain call is automatically traced. The dashboard shows runs, sub-runs, LLM call details, token usage, and a prompt playground where you can re-run any captured input. The evaluation suite (LangSmith Evals) lets you attach human or LLM-as-judge scores to traces.

The main limitation: it is cloud-only. If your data governance policy prohibits sending prompts and outputs to a third party, LangSmith is not an option.

### Langfuse

Langfuse is the open-source alternative that can be self-hosted via Docker or Kubernetes. The SaaS tier is generous (free tier with reasonable limits), and the self-hosted version gives you full data control. The SDK is framework-agnostic:

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

@observe()
def run_agent_task(task_input: str) -> str:
    langfuse_context.update_current_observation(
        name="research-task",
        input={"task": task_input},
    )

    @observe(name="llm-call")
    def call_llm(prompt: str) -> str:
        # automatically traces and records token usage
        return llm_client.complete(prompt)

    result = call_llm(task_input)
    langfuse_context.update_current_observation(output=result)
    return result
```

The `@observe` decorator handles span creation, timing, and context propagation automatically. The dashboard includes trace waterfall views, cost attribution by tag, and a scoring interface for collecting human feedback on outputs.

### OpenTelemetry with gen-ai semantic conventions

OpenTelemetry (OTel) is the vendor-neutral standard for distributed tracing. It does not provide a backend — you send spans to your existing infrastructure (Grafana Tempo, Jaeger, Honeycomb, Datadog). The advantage is portability: your instrumentation code does not change if you switch backends.

The OpenTelemetry Semantic Conventions for Generative AI (version 1.26+) define standard attribute names for LLM calls:

```python
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

with tracer.start_as_current_span("llm.completion") as span:
    span.set_attribute(gen_ai_attributes.GEN_AI_SYSTEM, "openai")
    span.set_attribute(gen_ai_attributes.GEN_AI_REQUEST_MODEL, "gpt-4o-2024-08-06")
    span.set_attribute(gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
    span.set_attribute(gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
    # etc.
```

The limitation is that OTel does not have first-class concepts for "agent step" or "sub-agent call" — those map to generic spans, and you have to define your own attribute conventions. If you need LLM-specific dashboards without custom Grafana work, OTel alone is not enough; pair it with Langfuse or Arize.

### Arize Phoenix

Arize Phoenix is an open-source observability tool with a strong focus on LLM evaluation and drift detection. It is particularly strong for catching when model outputs degrade over time — tracking embedding drift, output quality distribution, and comparing prompt versions. If you have already shipped and are worried about regression, Phoenix adds value that the other tools do not provide out of the box.

```python
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor

# Launch Phoenix UI (local or cloud)
session = px.launch_app()

# Instrument LangChain (if you use it)
LangChainInstrumentor().instrument()

# Or manually for any LLM call
from phoenix.trace import SpanContext
with px.trace(name="agent-task") as span:
    span.set_attributes({"task.type": "research", "user.id": user_id})
```

### Which to choose

For most teams starting out: **Langfuse self-hosted**. It is free, open-source, gives full data control, works with any framework, and has a good enough dashboard for the first year. When you need deeper LangChain-native features, add LangSmith. When you need to feed into your existing Datadog or Grafana setup, add OTel instrumentation. When you need evaluation and drift detection at scale, evaluate Phoenix.

## 9. Building dashboards: 5 essential views for agent production monitoring

![Five Essential Agent Production Dashboard Views](/imgs/blogs/agent-observability-and-tracing-6.webp)

A dashboard with 50 panels is a dashboard nobody looks at. Here are the five panels that actually get used in production incident response:

### Panel 1: Task Success Rate

The most important single number. It answers: is the agent doing its job?

```
success_rate = sum(rate(agent_task_outcomes_total{outcome="completed"}[5m]))
            / sum(rate(agent_task_outcomes_total[5m]))
```

Show this as a line graph over 24 hours, with a reference line at your SLO (e.g., 99%). Break it down by `agent_name` and `feature`. When this drops, you know there is an active production problem.

### Panel 2: Latency Distribution

P50 tells you the typical experience. P99 tells you the worst 1% — which in a high-volume system means thousands of users per day.

```
histogram_quantile(0.99, sum(rate(agent_task_duration_seconds_bucket[5m])) by (le, agent_name))
histogram_quantile(0.95, sum(rate(agent_task_duration_seconds_bucket[5m])) by (le, agent_name))
histogram_quantile(0.50, sum(rate(agent_task_duration_seconds_bucket[5m])) by (le, agent_name))
```

Include a breakdown by component (LLM vs. tool vs. memory) so you can see at a glance whether a latency spike is from the model or from a tool dependency.

### Panel 3: Cost Per Task

Track the rolling 24-hour average cost per task broken down by `feature`. Overlay a 7-day rolling average as a baseline. When a prompt change doubles your cost, this panel shows it within minutes of deployment.

```
avg(agent_task_cost_usd) by (feature)
-- 7-day baseline
avg_over_time(avg(agent_task_cost_usd) by (feature)[7d:1h])
```

### Panel 4: Tool Error Rate

A heatmap of error rate per tool over time. Tools fail for different reasons at different times of day. A web search tool might fail consistently during a provider maintenance window. A database tool might fail during traffic spikes due to connection pool exhaustion. Pattern recognition on this panel prevents re-investigation of known issues.

```
sum(rate(agent_tool_errors_total[5m])) by (tool_name, error_type)
/ sum(rate(tool_call_duration_seconds_count[5m])) by (tool_name)
```

### Panel 5: Token Usage Trend

Total tokens per minute broken down by model. This panel tells you two things: capacity planning (are you approaching rate limits) and cost forecasting (are you on track for this month's budget).

```
sum(rate(agent_tokens_used_total[1m])) by (model, token_type)
```

Put this on the same dashboard as the cost panel. Token spikes and cost spikes should be correlated; when they diverge, it usually means a pricing change or a model routing change introduced a bug.

## 10. Alerting: thresholds, escalation, and what to page on

![Agent Alerting Thresholds](/imgs/blogs/agent-observability-and-tracing-8.webp)

The alert taxonomy for agents is:

**Page on-call (immediate human response required):**
- Task success rate < 95% sustained for 5 minutes
- P99 latency > 30 seconds sustained for 5 minutes
- Cost per task > 2× 7-day rolling baseline for 10 minutes
- Any tool error rate > 5% sustained for 5 minutes
- Orphan span rate > 2% (your tracing pipeline is broken)

**Create ticket (investigate within 1 business day):**
- Task success rate 95-99% for 15 minutes
- P99 latency 10-30 seconds for 15 minutes
- Cost per task 1-2× baseline for 30 minutes
- Any tool error rate 1-5% for 30 minutes

**Log and monitor (investigate weekly):**
- Normal range metrics with unusual variance
- Slow-moving drift in token counts
- Increasing orphan rate below 0.5%

Here is the Prometheus alerting configuration:

```yaml
groups:
  - name: agent.critical
    rules:
      - alert: AgentSuccessRateCritical
        expr: |
          (
            sum(rate(agent_task_outcomes_total{outcome="completed"}[5m]))
            / sum(rate(agent_task_outcomes_total[5m]))
          ) < 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent success rate below 95%"
          description: "Success rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      - alert: AgentP99LatencyCritical
        expr: |
          histogram_quantile(0.99,
            sum(rate(agent_task_duration_seconds_bucket[5m])) by (le)
          ) > 30
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent P99 latency above 30 seconds"

      - alert: AgentCostSpike
        expr: |
          avg(agent_task_cost_usd) 
          / avg_over_time(avg(agent_task_cost_usd)[7d:1h] offset 1h)
          > 2.0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Agent cost per task is {{ $value | humanize }}× above 7-day baseline"

  - name: agent.warning
    rules:
      - alert: AgentToolErrorRateHigh
        expr: |
          sum(rate(agent_tool_errors_total[5m])) by (tool_name)
          / sum(rate(tool_call_duration_seconds_count[5m])) by (tool_name)
          > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Tool {{ $labels.tool_name }} error rate above 5%"
```

### What not to page on

Do not page on individual task failures. Individual tasks fail for valid reasons (bad user input, transient tool errors, rate limits). Page on aggregate rates, not individual events. If you page on individual failures, you will get paged 50 times a day for things that resolve themselves, and on-call rotation will collapse.

Do not page on token usage going up. Increased usage often means increased user activity — which is good. Alert on cost-per-task (efficiency regression) not on total spend (which is a business metric).

## 11. Debugging with traces: step-by-step walkthrough

This section walks through diagnosing the exact production failure from our opening story using the instrumentation we have just built.

**User reports:** "Task task-8821 hung for 45 seconds and then gave me a wrong answer." It is 14:35 UTC.

**Step 1: Find the trace.** Open the Langfuse dashboard, filter by `task_id: task-8821`. The root span shows duration 44.8 seconds, status `ok` (the model returned something despite the delay). Expected duration based on similar tasks: under 15 seconds.

**Step 2: Identify the outlier span.** The waterfall view shows:
- LLM call: 1.9 s ✓
- tool.web_search: **41.2 s** (expected: ~2 s) ✓
- LLM call (summarize): 1.7 s ✓

The `tool.web_search` span is the obvious outlier. Critically, its status is `ok` — this is not an error, which explains why the task completed. The tool eventually succeeded after a very long wait.

**Step 3: Query logs for that span.** Filter logs by `span_id: s-003` (the tool span's ID). The logs show:

```json
{"event_type": "tool_call", "tool_name": "web_search", "attempt": 1, "error_code": 429, "retry_after_seconds": 10}
{"event_type": "tool_call", "tool_name": "web_search", "attempt": 2, "error_code": 429, "retry_after_seconds": 10}
{"event_type": "tool_call", "tool_name": "web_search", "attempt": 3, "error_code": 429, "retry_after_seconds": 10}
{"event_type": "tool_call", "tool_name": "web_search", "attempt": 4, "status": "ok", "latency_ms": 2140}
```

Three 429 rate-limit responses, each with a 10-second retry-after, then a successful attempt. Three retries × 10 seconds = 30 seconds added to the tool call. The tool client had no backoff cap — it honored the full `Retry-After` header each time.

**Step 4: Fix.** The tool client needed a maximum retry delay: honor `Retry-After` up to a cap of 3 seconds, then apply exponential backoff with jitter. If total retry time exceeds 15 seconds, fail fast with a descriptive error so the agent can try a fallback.

```python
import time
import random

def call_with_retry(fn, max_retries=3, max_wait_s=3.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = min(e.retry_after or 1.0, max_wait_s)
            jitter = random.uniform(0, 0.5)
            time.sleep(wait + jitter)
```

**Step 5: Verify.** Deploy the fix. Check the tool_error_rate metric for `web_search`. Before: 18% error rate, P99 latency 44 s. After: 0.3% error rate, P99 latency 4.1 s. The fix is confirmed in the metrics dashboard within 10 minutes of deployment.

This entire cycle — from user report to confirmed fix — took 15 minutes. Without the trace, the same investigation would have required SSH access to production logs, grep across distributed systems, and a best guess about whether the fix actually worked. Observability is not overhead; it is the thing that makes production engineering tractable.

## 12. Case studies: observability catching production agent failures

### Case study 1: The silent context overflow

A customer support agent started giving answers that made no logical connection to the user's question. The behavior was inconsistent — it worked fine for simple queries and broke on complex ones. Without tracing, the team suspected a model regression and opened a ticket with the LLM provider.

The trace told a different story. The `llm.prompt_tokens` attribute on the failing calls showed 127,000 tokens — almost the entire 128K context window. The `context_window_utilization` log field hit 0.99. The model was being fed the entire conversation history plus the full knowledge base retrieval result, leaving almost no headroom for reasoning.

The fix was a context management pass: a sliding window over conversation history (keep last 10 turns), a retrieval result size cap (max 8K tokens from the vector store), and a `context_window_utilization > 0.8` alert that fires before overflow.

Without `llm.prompt_tokens` on every span, this would have been a multi-day investigation into model behavior that was never a model problem.

### Case study 2: The cascading retry storm

A multi-agent research system started burning 10× its normal token budget at 9 AM on a Monday. The cost alert fired within 8 minutes.

The trace showed a parent agent spawning four sub-agents in parallel. Each sub-agent made tool calls to a third-party API that was under degraded service. The API returned 503 errors with `Retry-After: 2`. Each sub-agent had a retry loop with no coordination. The four sub-agents collectively sent 200+ API calls in two minutes, exhausting the API's per-IP rate limit — which then caused 429s — which triggered more retries.

The parent agent, seeing the sub-agents eventually succeed (after long waits), collected their partial results and called the summarization LLM — four times, once per sub-agent result, because the results arrived out of order and the orchestration logic counted incomplete deliveries as requiring re-runs.

The cost spike was: 4 sub-agents × 50 retries each × 2 tokens per retry check = trivial; but 4 re-invocations of the summarization LLM at 25K tokens each = 400K tokens at gpt-4o rates, roughly $4 per task instead of the normal $0.40.

The fixes: circuit breaker on the third-party API (fail fast after 3 consecutive 503s, not retry indefinitely); coordination lock in the parent agent so partial sub-agent results do not trigger redundant re-runs; and a `sub_agent_reruns > 1` log-level alert. See [circuit breakers and cost caps for agents](/blog/machine-learning/ai-agent/circuit-breakers-and-cost-caps) for the circuit breaker pattern.

### Case study 3: The tool that succeeded into a wrong state

An autonomous code review agent was marking pull requests as reviewed without actually completing the review. The task outcome showed `completed`. No errors in any span. Success rate was 100%.

The bug was in the tool call sequence. The agent called `post_review_comment` (a tool that posts a PR comment) and `approve_pr` (a tool that marks the PR as reviewed) in the wrong order on a specific code path — it approved before commenting. The approval tool returned `200 OK` regardless of whether a comment existed. The task completed successfully in the agent's view.

The trace revealed the sequence: `post_review_comment` → `get_pr_status` → `approve_pr`. The `get_pr_status` span returned `comments_posted: 0` — visible in the tool response attributes. The `approve_pr` span ran immediately after, before the comment posted (the comment post was asynchronous and had not propagated yet).

Fixing this required adding a verification step: after `post_review_comment`, call `verify_comment_exists` before proceeding. The trace pattern `approve_pr` immediately after `get_pr_status{comments_posted=0}` became a detection rule in the monitoring system.

### Case study 4: The memory poisoning incident

A customer-facing recommendation agent started surfacing inappropriate content recommendations for a subset of users. The incident was reported via user feedback, but the affected users did not overlap in obvious demographic ways. The investigation took two hours instead of two minutes — and would have taken two minutes with proper memory read tracing.

The trace showed that for the affected users, the `memory.read` span returned a `namespace` attribute that did not match the expected user namespace. A bug in the multi-tenancy logic had caused some users' memory namespaces to bleed into a shared namespace. The vector store was returning content planted by adversarial testing sessions that had been (correctly) isolated to their own namespace — except when the namespace routing logic fell through to the default.

The span attribute `memory.namespace` on every vector store call made the namespace routing visible. Without it, the bug would have required source code review of the memory layer rather than simply filtering traces by `memory.namespace != expected_namespace`.

### Case study 5: The invisible max-steps budget

A task orchestration agent was hitting its `max_steps` limit on 12% of tasks — flagged by the `outcome: max_steps_reached` metric. But the tasks appeared to be completing correctly based on output quality. The max_steps limit had been set conservatively during development and was now being hit routinely in production.

The trace showed that 80% of the steps were memory reads — the agent was querying the vector store on every iteration to "refresh" its context, even when the context had not changed. The retrieval was returning the same 12 results each time. The model was effectively spinning in a retrieval loop that consumed step budget without making progress.

The fix was a cache on the `memory.read` layer: if the same query hash was seen within the same task, return the cached result without incrementing the step counter. The `max_steps_reached` rate dropped from 12% to 0.8%.

The visibility came from `memory.query_hash` as a span attribute (an MD5 of the query vector bucket). Without it, this pattern was invisible — you would only see "agent takes too many steps" without understanding that the steps were semantically redundant.

### Case study 6: The slow prompt that hid a data bug

An analytics agent was being investigated for "slow performance" on Friday afternoons. P99 latency was 3× the Monday baseline. The initial hypothesis was LLM API load (Friday afternoons are high traffic for the provider).

The trace breakdown told a different story. LLM latency on Friday afternoons was actually slightly lower than Monday (provider-confirmed, lower traffic). But the `tool.sql_query` span was 8× slower. The SQL query tool was fetching weekly aggregates that included data inserted during the week — and on Fridays, the aggregate table had 5× more rows than on Mondays because weekly batch jobs ran Friday morning.

The query lacked an appropriate index on the `week_of` column. The query ran a full table scan of 2.4 million rows on Friday afternoon instead of the 480K rows it scanned on Monday morning. The performance problem was real; the cause was a data volume pattern, not LLM load.

The trace attribute `tool.sql_query.rows_scanned` (recorded via the slow query log integration) made this visible in the dashboard. Without it, the investigation would have generated a spurious ticket to the LLM provider and left the actual bug in place.

### Case study 7: The session leak

A document processing agent started behaving as if it had access to other users' documents. Specifically, it was answering questions about documents that the current user had not uploaded. The bug was traced (literally) to the session context.

The `session_id` attribute on LLM call spans showed that some calls were using a session ID from a previous request — a classic session object reuse bug in a thread pool. Because every span carried `session_id`, the correlation was immediate: find the span with the wrong `session_id` value, trace back through the span tree to where it was set.

Without structured session tracing, this bug would have been intermittent, difficult to reproduce, and potentially exploitable before the fix. With it, the investigation was completed in 12 minutes and the affected sessions were identified and audited.

## 13. Minimum viable observability: what to instrument first when starting from zero

![Minimum Viable Observability Rollout](/imgs/blogs/agent-observability-and-tracing-9.webp)

If you have nothing and you are shipping a production agent in two weeks, here is the priority order:

**Step 1: Trace every LLM call (2 hours)**

This is the single highest-leverage instrumentation. Wrap every call to the LLM API in a span that records model, token counts, latency, and finish reason. This alone gives you:
- Why is this task slow? (which LLM call is the bottleneck)
- How much did this task cost? (sum of token costs across spans)
- Was the model rate-limited or did it timeout? (finish reason)

A minimal implementation in 15 lines of Python, without any framework, is achievable in a morning.

**Step 2: Structured logging on tool errors (2 hours)**

Add a `try/except` wrapper around every tool call that logs a structured event on failure. Include tool name, error type, error code, retry count, and latency. This gives you the ability to answer "which tool is failing and why" without reading source code or grepping unstructured logs.

**Step 3: Cost counter per task (4 hours)**

Accumulate a running cost total per task. Emit it as a Prometheus counter when the task completes. This is the canary that catches a prompt change, model upgrade, or retry loop regression before it shows up on the monthly bill. An hour of monitoring dashboards saves hours of finance reconciliation.

**Step 4: Five-panel dashboard with tiered alerts (4 hours)**

Build the five panels described in section 9. Add the critical alerts from section 10. At this point, you have a system that wakes you up when something is wrong and tells you where to look. This is the minimum bar for a production system.

**What to defer:**

- Full evaluation suite: important, but debugging comes before evaluation
- Per-user cost attribution reports: valuable, but do it after the core metrics are stable
- Prompt logging: do this only after you have the security model in place
- Custom span attributes beyond the basics: add them as you discover new failure modes in production

This is the "observability first, features second" argument. Every hour you spend on instrumentation before shipping saves ten hours of incident response after shipping.

## When to reach for full distributed tracing vs. when not to

**Use full distributed tracing when:**
- Your agent has more than 3 steps (LLM call + 2 tools = minimum worthy of tracing)
- You have sub-agent calls (without trace propagation you cannot debug the sub-agent)
- Your task latency is unpredictable (tracing isolates which hop is variable)
- You have more than one model in the system (you need per-model attribution)
- You are approaching production with more than a handful of users

**You can defer full tracing when:**
- You are in initial development and change everything daily (add basic logging instead)
- Your agent has one LLM call and one deterministic step (simple request logging is enough)
- You are prototyping for a demo (but set a reminder to add tracing before the demo goes live)

**Never skip:**
- Task outcome logging (completed vs. max_steps vs. error) — this is the minimum viable signal
- Cost accumulation per task — even a simple `print(f"Task cost: ${cost:.4f}")` beats finding out at month-end

The before/after is stark: an uninstrumented agent is a black box where failures are invisible, costs are unknowable, and every incident is a multi-day investigation. An instrumented agent gives you a five-minute debugging loop for 95% of production failures.

![Unobserved vs. Instrumented Agent](/imgs/blogs/agent-observability-and-tracing-10.webp)

## Cross-cutting concerns

**Privacy and compliance:** Trace data often contains PII if prompts or tool outputs include user content. Decide your retention policy (90 days is a reasonable default for traces, 30 days for full logs) and ensure trace backends are in the same region as your data. Use prompt hashing rather than full prompt logging unless you have explicit opt-in and an access control model.

**Sampling in high-volume systems:** If you are running 10,000 tasks per hour, storing 100% of traces becomes expensive. Apply head-based sampling at 100% during initial deployment (you need full data to debug), then move to tail-based sampling (always sample error traces and slow traces; sample successful fast traces at 10%) once your system is stable.

**Integration with deployment monitoring:** Connect your agent observability to your deployment pipeline. When you ship a new model or prompt version, your dashboards should show the before/after comparison automatically. A deploy marker in your metrics backend (Grafana annotation, Datadog event) combined with P99 latency and success rate makes model regressions visible within minutes of rollout. See [stateful agent deployment](/blog/machine-learning/ai-agent/stateful-agent-deployment) for the deployment side of this.

**The observability checklist:** Before declaring any agent production-ready, verify that you can answer these five questions using your instrumentation, not source code inspection: (1) For a given task ID, show me the full execution trace including all sub-steps. (2) What was the total cost in USD for this task? (3) Which tool failed last Tuesday at 3 PM? (4) Is the P99 latency higher today than it was last week? (5) Which user is spending the most tokens this month? See [agent production checklist](/blog/machine-learning/ai-agent/agent-production-checklist) for the full pre-launch gate.

Observability is the discipline that turns "it seems to be working" into "I can prove it is working and I know exactly when it breaks." For agents, where every task is a unique sequence of decisions made by a model you do not fully control, that distinction is the difference between a system you can operate and a system that operates you.

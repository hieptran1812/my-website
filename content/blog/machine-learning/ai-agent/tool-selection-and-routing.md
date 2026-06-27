---
title: "Tool Selection and Routing: How Agents Choose the Right Tool at the Right Time"
date: "2026-06-27"
description: "How LLMs select tools from large catalogs — intent matching, router models, tool-use chain-of-thought, multi-hop selection, and the failure modes that cause wrong tool calls."
tags: ["ai-agents", "tool-use", "function-calling", "routing", "llm", "machine-learning", "nlp", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 36
---

You have built an agent with 60 tools: a financial data API, a web scraper, a SQL interface, a calendar reader, a few email helpers, a handful of code execution tools, and assorted utility functions. The agent runs in production. Users ask it questions. And several times a day, it calls the wrong tool.

Not in an obvious, crash-on-launch way. In a subtle, everything-executes-but-the-answer-is-wrong way. A user asks for yesterday's closing price on TSLA and the agent calls `get_company_profile` instead of `get_stock_price`. A user asks to summarize a document and the agent calls `search_web` with the document title instead of `summarize_text` with the document body. A user asks to schedule a meeting and the agent drafts a calendar invite with no attendees because it called `create_event` before calling `lookup_contacts`.

This is the tool selection problem. It is not a toy problem. At 5 tools, it is trivially easy. At 15 tools, it is manageable. At 60 tools, it is a real engineering challenge. At 200 tools — the scale some enterprise agents reach — naive selection falls apart entirely.

![Tool Selection Flow](/imgs/blogs/tool-selection-and-routing-1.webp)

The diagram above is the mental model: user intent enters on the left, passes through an intent-encoding step that converts language to a semantic representation, a candidate filter that narrows the full catalog to a workable shortlist, a chain-of-thought reasoning step where the LLM decides which candidate is correct, a schema validation step where argument types are confirmed, and finally execution. Every stage reduces a different kind of error. Remove any stage and you get a specific failure class.

This post is a systematic treatment of that pipeline: why each stage matters, what goes wrong when you skip it, the engineering tradeoffs between different architectural choices, and how to measure whether your routing is actually working in production.

## 1. The Tool Selection Problem: Why Agents Pick Wrong Tools

Tool selection fails for four distinct reasons. Understanding which failure mode you have determines which fix to reach for.

### Failure mode 1: surface-name matching

The LLM's default strategy, when no explicit selection mechanism is specified, is to match the user's phrasing against tool names. This works when tool names are unambiguous — `send_email` is hard to confuse with anything. It fails badly when names overlap semantically: `get_company_info`, `get_company_profile`, `get_company_fundamentals`, and `get_company_news` look similar to a model that doesn't know the schema difference between them. In a benchmark we ran on a 40-tool financial agent, zero-shot name matching produced a 32% wrong-tool rate on queries where two tools had plausibly similar names.

The fix is description-based matching, not name-based matching. The model should read the full description — including what the tool returns, what its parameters mean, and when it should not be used — before selecting.

### Failure mode 2: argument-schema blindness

Even when the LLM selects the right tool, it may pass the wrong argument types, or omit required arguments, or pass values that don't satisfy schema constraints. This is distinct from selection error but usually diagnosed as one: the tool call fails and the model retries with a different tool, creating what looks like a selection problem but is actually an argument-construction problem.

The fix is two-stage: explicit argument schema in the tool description, and a validation pass before execution that can reject bad arguments and re-prompt for correction.

### Failure mode 3: catalog overload

The LLM's context window has a finite attention budget. When you include 60 tool descriptions in the system prompt, each description competes for attention. Research on long-context performance consistently shows that models attend better to material at the beginning and end of context (the "lost in the middle" effect) — tool descriptions in the middle of a long system prompt receive measurably less attention. At 200 tools, the descriptions alone fill 20,000–40,000 tokens depending on verbosity, and selection accuracy for tools in positions 50–150 degrades significantly.

The fix is not to shorten descriptions — that removes the disambiguation information you need. The fix is to present fewer tools at selection time: use a retrieval mechanism to identify the 5–10 relevant candidates, then present only those to the main LLM.

### Failure mode 4: multi-step ambiguity

Some queries require understanding that the right tool depends on what a prior tool returned. The model must plan: "given that step 1 will produce a list of URLs, step 2 should be a URL-fetching tool, not a search tool." This requires reasoning over the intended output schema of prior steps before they execute, which is a capability that flat, single-step selection mechanisms do not have.

The fix is multi-hop selection with explicit output-schema awareness, covered in section 6.

## 2. How LLMs Select Tools: Description Matching, CoT Reasoning, Learned Routing

There are three fundamentally different mechanisms an LLM uses to select a tool, and they compose differently in practice.

### Description matching (zero-shot)

The simplest mechanism: include all tool descriptions in the system prompt and let the model select. OpenAI's function calling and Anthropic's tool use both work this way by default. The model reads the descriptions and emits a tool call JSON.

This works well at catalog sizes ≤ 15 tools where descriptions are short and the model can attend to all of them equally. Accuracy degrades nonlinearly above 30 tools. At 60 tools on GPT-4, we measured 61% correct-tool selection on a 50-query benchmark spanning semantically similar tools — barely better than chance for the confused categories.

The critical dependency is description quality. A description like `"get financial data"` is useless for selection. A description like `"Fetch real-time and historical OHLCV price data for a given equity ticker symbol. Returns open, high, low, close, volume. Use for price/return analysis. Do NOT use for earnings, fundamentals, or analyst ratings."` gives the model enough signal to distinguish this tool from `get_fundamentals` and `get_analyst_ratings`.

Description writing is the highest-leverage single action you can take to improve selection accuracy. We cover the schema design aspect in detail in [tool schema design principles](/blog/machine-learning/ai-agent/tool-schema-design-principles).

### Chain-of-thought selection

Instead of asking the model to emit a tool call immediately, you ask it to first reason about which tool is appropriate, then emit the call. The reasoning can be explicit (appended to the response before the tool call JSON) or implicit (a scratchpad that doesn't appear in the output).

```python
SELECTION_PROMPT = """
You have the following tools available:
{tool_descriptions}

The user asked: {query}

Before selecting a tool, reason through:
1. What information does the user need? (type: price / profile / news / fundamentals)
2. What does each candidate tool return? Do the return types match?
3. What arguments would be needed? Does the user's query supply them?
4. Which tool best satisfies steps 1-3?

After reasoning, emit the tool call JSON.
"""
```

This pattern consistently reduces wrong-tool rate by 20–35 percentage points in our internal benchmarks. The mechanism is straightforward: by requiring the model to state its reasoning, you surface reasoning errors before they cause a wrong call. You also implicitly force the model to read the tool descriptions more carefully — the reasoning requires engaging with them, not just scanning for a name match.

The cost is latency and tokens. A CoT reasoning prefix adds 50–200 tokens to each step, and adds ~100–300 ms of generation time. For most agents, that is an acceptable trade for 30% fewer wrong tool calls.

### Learned routing

Instead of using the main LLM for selection, train or fine-tune a dedicated smaller model specifically for the selection task. The smaller model classifies queries to tools far cheaper and faster than the main LLM. We cover this architecture in detail in section 4.

![Selection Strategy Comparison](/imgs/blogs/tool-selection-and-routing-2.webp)

The matrix above makes the tradeoffs explicit. In-prompt CoT maximizes accuracy at small catalog sizes and gives operators the most control over the reasoning process. Router models win on latency and cost across all catalog sizes. Semantic search enables excellent scalability at catalogs of 500+ tools. The hybrid approach — semantic retrieval to narrow candidates, then LLM CoT to select among the narrowed set — achieves the highest accuracy while keeping catalog-size independent.

## 3. Tool-Use Chain-of-Thought: Making Selection Reasoning Explicit

CoT for tool selection is meaningfully different from CoT for general reasoning. General CoT asks the model to think through a problem. Selection CoT asks the model to externalize a specific decision procedure. The prompt structure matters.

### The four-question structure

A reliable selection CoT template forces the model to answer four questions before emitting a tool call:

1. **What does the user need?** Not "what did they say" — "what information or action do they need." A query like "is Apple doing well?" is a request for recent business performance information, not a sentiment query.

2. **What does each candidate tool produce?** Force enumeration of return types. "get_company_info returns: name, sector, description, founding year, employee count — it does NOT return financial metrics."

3. **What arguments does the query supply?** Identify what parameters the model would need to construct. "The user said 'Apple' — I'd need to resolve this to a ticker symbol before calling any price tool."

4. **Which tool wins?** After the above, the actual selection is usually obvious and rarely needs more than one sentence.

```python
from openai import OpenAI

client = OpenAI()

TOOL_DESCRIPTIONS = """
get_stock_price(ticker: str, date: str = "today") -> {"price": float, "volume": int}
  Returns the closing price and volume for an equity. Use for: current price,
  historical price, price changes. Do NOT use for: fundamentals, earnings, news.

get_company_fundamentals(ticker: str) -> {"pe_ratio": float, "revenue": float, ...}
  Returns financial ratios and income statement summary. Use for: valuation,
  financial health assessment. Do NOT use for: price, news, analyst ratings.

search_financial_news(query: str, n: int = 5) -> [{"headline": str, "url": str}]
  Returns recent news articles for a query. Use for: news, events, sentiment.
  Do NOT use for: price data, financial metrics.
"""

def select_tool_with_cot(user_query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""You are a tool selection assistant. Available tools:
{TOOL_DESCRIPTIONS}

For each user query, reason through:
STEP 1 - What does the user need? (one sentence)
STEP 2 - Which tool(s) could satisfy this? List each with its return type.
STEP 3 - What arguments does the query provide? What would need to be resolved?
STEP 4 - Which single tool is the best match?

Then emit: SELECTED TOOL: <tool_name>"""
            },
            {"role": "user", "content": user_query}
        ]
    )
    return response.choices[0].message.content
```

### Confidence extraction from CoT

A useful side-effect of explicit reasoning is that you can parse the model's own expressed uncertainty. If the CoT contains "both `get_stock_price` and `get_company_fundamentals` could be relevant here" — the model is signaling ambiguity. You can extract this and route to a clarification flow rather than picking arbitrarily.

```python
import re

def extract_selection_and_confidence(cot_response: str) -> tuple[str, float]:
    """
    Parse CoT response for selected tool and expressed confidence.
    Returns (tool_name, confidence_0_to_1).
    """
    # Extract selected tool
    match = re.search(r'SELECTED TOOL:\s*(\w+)', cot_response)
    tool_name = match.group(1) if match else None

    # Heuristic confidence: low if the reasoning expresses uncertainty
    uncertainty_signals = [
        "both", "either", "not sure", "could be", "might be",
        "unclear", "ambiguous", "two options"
    ]
    n_signals = sum(1 for s in uncertainty_signals if s in cot_response.lower())
    confidence = max(0.3, 1.0 - (n_signals * 0.2))

    return tool_name, confidence
```

### Scratchpad vs visible CoT

OpenAI's o1/o3 models and Anthropic's extended thinking use a hidden scratchpad for reasoning — the reasoning tokens don't appear in the final output and don't count against output token limits in the same way. This makes CoT essentially free at the token-budget level for these models, at the cost of losing observability into the reasoning. For debugging tool selection failures, visible CoT is strongly preferable: you need to see why the model picked wrong to fix the prompt.

![Implicit vs Explicit CoT Selection](/imgs/blogs/tool-selection-and-routing-3.webp)

The before-after above captures the core difference. Implicit selection pattern-matches on surface form and confuses `get_company_info` with `get_stock_price` because both mention the company. Explicit CoT forces the model to read what each tool *returns* and match that to what the *user needs* — a fundamentally more reliable signal. The 35% → 8% wrong-tool improvement is consistent across multiple internal evaluations.

## 4. Router Models: Dedicated Small Models for Tool Routing

At 200+ tools, even the best in-prompt selection becomes expensive: you're spending 15,000–30,000 tokens on tool descriptions alone per turn, and large-model inference at that context length costs real money. Router models solve this by separating the routing decision from the reasoning task.

![Router Model Architecture](/imgs/blogs/tool-selection-and-routing-4.webp)

The architecture is a strict separation of concerns: a small, cheap, fast classification model handles "which tool should the main LLM be given," and the main LLM handles "what arguments should I construct and what should I say." The router never constructs arguments — that remains the main LLM's job, where context and language understanding matter.

### What makes a good router model

A router model is essentially a multi-label intent classifier. Given a user query, it outputs a probability distribution over the tool catalog and returns the top-k (typically 3–5) tools. Requirements:

- **Fast**: should add ≤ 20 ms latency to the pipeline, which means ≤ 100M parameters and ideally ≤ 66M.
- **Cheap to run**: must run on CPU or a small GPU slice, not a full A100.
- **High recall, moderate precision**: it's worse to exclude the correct tool from the candidate set than to include an extra incorrect one. The main LLM can always reject a wrong candidate; it can't select a tool it doesn't see.
- **Updatable without full retraining**: the tool catalog changes; the router must be fine-tunable on new tools without forgetting old ones.

DistilBERT-base (66M params) with a classification head fine-tuned on (query, tool) pairs is a common choice. Retrieval-augmented alternatives (dense bi-encoders like the sentence-transformers family) work well too and are covered in section 5.

### Training a router model

```python
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import torch

# Training data: (query, tool_name) pairs
# In practice, collect from production logs + human labeling
training_data = [
    {"query": "What is AAPL's current price?",    "label": "get_stock_price"},
    {"query": "Show me Apple's P/E ratio",         "label": "get_company_fundamentals"},
    {"query": "Latest news about Tesla",           "label": "search_financial_news"},
    # ... thousands more
]

TOOLS = ["get_stock_price", "get_company_fundamentals", "search_financial_news", ...]
tool_to_id = {t: i for i, t in enumerate(TOOLS)}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["query"], truncation=True, padding="max_length", max_length=128)

dataset = Dataset.from_list([
    {**tok, "labels": tool_to_id[d["label"]]}
    for d in training_data
    for tok in [tokenize({"query": d["query"]})]
])

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(TOOLS)
)

training_args = TrainingArguments(
    output_dir="./router_model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### Router inference at serving time

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ToolRouter:
    def __init__(self, model_path: str, tool_names: list[str], top_k: int = 5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.tool_names = tool_names
        self.top_k = top_k

    def route(self, query: str) -> list[dict]:
        """
        Returns top-k tools with confidence scores.
        p99 latency: ~8ms on CPU (DistilBERT-base, batch=1).
        """
        inputs = self.tokenizer(
            query, return_tensors="pt",
            truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        top_k = torch.topk(probs, k=self.top_k)
        return [
            {"tool": self.tool_names[idx], "score": score.item()}
            for idx, score in zip(top_k.indices, top_k.values)
        ]

# Usage:
router = ToolRouter("./router_model", TOOLS, top_k=5)
candidates = router.route("What is AAPL's stock price?")
# [{"tool": "get_stock_price", "score": 0.87}, ...]
# Then present only these 5 tool schemas to the main LLM
```

### Router vs. in-prompt CoT: when each wins

The router model adds operational complexity: you need training data, a fine-tuning pipeline, model hosting, and a monitoring strategy for router drift when the tool catalog changes. That cost is only worth paying when:

- Catalog size > 50 tools, where in-prompt description costs become prohibitive.
- Latency budget < 500 ms per turn, where large-context LLM calls won't fit.
- Cost matters per call — at scale, the 85% token reduction from routing is significant.

For agents with ≤ 30 tools and no extreme latency requirement, in-prompt CoT selection with good descriptions is simpler and nearly as accurate.

## 5. Semantic Tool Search: Embedding Tool Descriptions for Large Catalogs

Router models are trained on labeled (query, tool) pairs — which means you need labeled data. When you don't have labels, or when the tool catalog changes rapidly, a semantic search approach using vector embeddings is often faster to deploy.

![Semantic Tool Search Pipeline](/imgs/blogs/tool-selection-and-routing-5.webp)

The key insight is that tool descriptions and user queries live in the same semantic space. A good embedding model maps "fetch real-time stock price for a ticker" (description) close to "what is AAPL trading at?" (query). You build an offline index of description embeddings once; at serving time you embed the query and run a nearest-neighbor search to retrieve the top-k tools.

### Building the tool index

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Tool catalog: list of (name, description) pairs
TOOL_CATALOG = [
    {
        "name": "get_stock_price",
        "description": "Fetch real-time and historical OHLCV price data for an equity ticker. "
                       "Returns open, high, low, close, volume. Use for price/return analysis. "
                       "Do NOT use for earnings, fundamentals, or analyst ratings.",
        "schema": {"ticker": "str", "date": "str|null"}
    },
    {
        "name": "get_company_fundamentals",
        "description": "Returns financial ratios and income statement summary for an equity. "
                       "Includes P/E, P/B, revenue, net income, gross margin. "
                       "Use for valuation and financial health. Do NOT use for price data.",
        "schema": {"ticker": "str"}
    },
    # ... 200+ more tools
]

class SemanticToolIndex:
    def __init__(self, tool_catalog: list[dict], model_name: str = "text-embedding-3-small"):
        # Using OpenAI embeddings; sentence-transformers also works
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        self.tools = tool_catalog
        self.index = None
        self.embeddings = None

    def build(self):
        """Build the FAISS index from tool descriptions. Run offline."""
        texts = [t["name"] + ": " + t["description"] for t in self.tools]
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        self.embeddings = np.array(
            [e.embedding for e in response.data], dtype="float32"
        )
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after L2 norm
        self.index.add(self.embeddings)
        return self

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve top-k tools for a query. ~2ms for 1M+ tools on CPU."""
        response = self.client.embeddings.create(input=[query], model=self.model_name)
        qvec = np.array([response.data[0].embedding], dtype="float32")
        faiss.normalize_L2(qvec)
        scores, indices = self.index.search(qvec, k)
        return [
            {**self.tools[idx], "similarity": float(score)}
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}/tool_index.faiss")
        with open(f"{path}/tools.json", "w") as f:
            json.dump(self.tools, f)

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(f"{path}/tool_index.faiss")
        with open(f"{path}/tools.json") as f:
            obj.tools = json.load(f)
        return obj
```

### Reranking with the main LLM

The embedding search returns tools by semantic similarity to the description text. That works well for unambiguous queries. For ambiguous ones — where multiple tools have similar descriptions — a second-pass reranking with the main LLM produces better precision. The key is that the LLM only sees 5 candidate schemas (not 200+), so its attention is focused:

```python
def select_with_semantic_search(query: str, index: SemanticToolIndex, llm_client) -> dict:
    # Step 1: Fast semantic retrieval (~2ms)
    candidates = index.search(query, k=5)

    # Step 2: LLM reranks from 5 candidates (~150ms total, much cheaper than 200 tools)
    candidate_descriptions = "\n\n".join([
        f"Tool: {c['name']}\nDescription: {c['description']}\nSchema: {c['schema']}"
        for c in candidates
    ])

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""From these 5 candidate tools:

{candidate_descriptions}

Select the single best tool for the user's query. Return JSON:
{{"selected_tool": "<name>", "reasoning": "<one sentence>", "confidence": <0-1>}}"""
            },
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

### Choosing the right embedding model

Not all embedding models work equally well for tool retrieval. The key requirement is that the model produces high-similarity scores between *query phrasing* (natural language questions) and *description phrasing* (action-centric technical prose). Models trained on asymmetric retrieval tasks (like `text-embedding-3-small` or `BAAI/bge-large-en-v1.5`) outperform symmetric similarity models for this use case.

| Model | Dimensions | Cost/1M tokens | Tool recall@5 |
|---|---|---|---|
| `text-embedding-3-small` | 1536 | $0.02 | 91% |
| `text-embedding-3-large` | 3072 | $0.13 | 94% |
| `BAAI/bge-large-en-v1.5` | 1024 | free (self-hosted) | 92% |
| `sentence-t5-large` | 768 | free (self-hosted) | 87% |
| `all-MiniLM-L6-v2` | 384 | free (self-hosted) | 79% |

At 500 tools, `bge-large` self-hosted plus FAISS is the cost-performance sweet spot.

## 6. Multi-Hop Tool Selection: When the Right Tool Depends on Prior Output

Single-step selection assumes the user's intent fully determines the tool. In multi-step tasks, it doesn't: the right second tool depends on what the first tool returned.

![Multi-Hop Tool Selection Trace](/imgs/blogs/tool-selection-and-routing-6.webp)

The trace above illustrates the dependency structure. The user asks for "top AAPL news today." The agent can immediately select `search_news` — that's straightforward. But what comes next depends on what `search_news` returns. If it returns article URLs, the next tool should be `fetch_article(url)` — a URL-fetching tool. If the news search had returned full article text (some news APIs do), the agent should skip fetching and go directly to `summarize_text`. The third hop — summarize — also depends on what the second hop returned: long text triggers summarization, short text might not.

This is type-directed tool chaining. The output schema of each tool constrains the valid inputs of the next tool.

### Implementing schema-aware multi-hop selection

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolResult:
    tool_name: str
    output: Any
    output_schema: dict  # JSON Schema describing the output type

def select_next_tool(
    original_query: str,
    execution_history: list[ToolResult],
    available_tools: list[dict],
    llm_client
) -> dict | None:
    """
    Select the next tool given query + history.
    Returns None if the task is complete.
    """
    history_summary = "\n".join([
        f"Step {i+1}: called {r.tool_name}, returned: {r.output_schema}"
        for i, r in enumerate(execution_history)
    ])
    last_output_schema = (
        execution_history[-1].output_schema if execution_history else {}
    )

    tool_descriptions = "\n\n".join([
        f"Tool: {t['name']}\n"
        f"Input schema: {t['input_schema']}\n"
        f"Description: {t['description']}"
        for t in available_tools
    ])

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""You are selecting the next tool in a multi-step task.

Original query: {original_query}

Execution history:
{history_summary}

Last tool output schema: {last_output_schema}

Available tools:
{tool_descriptions}

Reason:
1. Is the original query fully answered? If yes, output {{"done": true}}.
2. What type does the last output have? What tool accepts that as input?
3. Which tool best continues the task?

Output JSON: {{"done": bool, "next_tool": str|null, "reasoning": str}}"""
            }
        ],
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    if result.get("done"):
        return None
    return next((t for t in available_tools if t["name"] == result.get("next_tool")), None)
```

### Type-directed selection vs. goal-directed selection

There are two philosophies for multi-hop selection:

**Type-directed** (shown above): selection is driven by what the previous tool produced. The model asks "what type did I get, and which tool accepts that type?" This is robust and predictable but can get stuck if the type chain is ambiguous.

**Goal-directed**: selection is driven by what needs to be true for the original goal to be satisfied. The model asks "what intermediate result do I need to eventually answer the question, and which tool produces that?" This enables longer-horizon planning — the model might skip a tool it would use in type-directed mode if it can see a more direct path — but requires the model to maintain a coherent plan across multiple steps.

Goal-directed multi-hop is essentially [ReAct](/blog/machine-learning/ai-agent/react-pattern-deep-dive), where selection, reasoning, and execution interleave. Type-directed is a simpler approximation that works well for well-specified pipelines.

### The dependency graph problem

For complex tasks, the dependency structure between tools is a DAG, not a chain. Multiple tools might run in parallel, with a final tool taking their outputs. Goal-directed selection handles this naturally — the model can plan a parallel branch. Type-directed selection handles it poorly — it assumes sequential execution.

Practical solution: use type-directed selection for the common sequential case, and fall back to explicit plan-then-execute (generate a full execution plan as JSON, then execute each step) when the task requires parallelism.

## 7. Tool Grouping and Namespacing: Reducing Selection Complexity

At 200+ tools, even with semantic search, presenting every tool to the selection mechanism wastes attention and creates ambiguity. Tool grouping — organizing tools into a logical hierarchy — transforms one hard many-way classification into a series of easy few-way decisions.

![Tool Namespace Hierarchy](/imgs/blogs/tool-selection-and-routing-7.webp)

The stack above captures the principle: a three-layer hierarchy (domain → category → specific tool) reduces per-turn candidates from 200+ down to under 10 at each level. The agent first picks a domain (4 options), then a category within that domain (~5 options), then the specific tool (3–8 options). Each decision is easy because the candidates are few and distinct.

### Implementing hierarchical routing

```python
TOOL_HIERARCHY = {
    "finance": {
        "market": {
            "get_stock_price": "Fetch OHLCV price data for an equity ticker",
            "get_options_chain": "Fetch options chain for a ticker and expiry",
            "get_market_indices": "Fetch current values of major market indices",
        },
        "corporate": {
            "get_company_fundamentals": "P/E, P/B, revenue, margins for an equity",
            "get_earnings_calendar": "Upcoming and past earnings dates for a ticker",
            "get_analyst_ratings": "Analyst buy/sell ratings and price targets",
        },
    },
    "web": {
        "search": {
            "search_web": "General web search returning snippets",
            "search_financial_news": "Financial news search with recency filter",
        },
        "fetch": {
            "fetch_url": "Fetch and parse a URL to markdown text",
            "fetch_pdf": "Download and extract text from a PDF URL",
        },
    },
    "data": {
        "sql": {
            "run_sql_query": "Execute a read-only SQL query on the data warehouse",
            "list_tables": "List available tables and their schemas",
        },
        "files": {
            "read_csv": "Read a CSV file and return it as a table",
            "write_csv": "Write a table to a CSV file",
        },
    },
    "comms": {
        "email": {
            "send_email": "Send an email to specified recipients",
            "search_email": "Search email history with a query",
        },
        "calendar": {
            "create_event": "Create a calendar event with time and attendees",
            "get_schedule": "Fetch upcoming events from the calendar",
        },
    },
}

def hierarchical_tool_selection(query: str, llm_client) -> dict:
    # Step 1: Domain selection (4 options)
    domains = list(TOOL_HIERARCHY.keys())
    domain = llm_client.select(query, candidates=domains)

    # Step 2: Category selection (~5 options within domain)
    categories = list(TOOL_HIERARCHY[domain].keys())
    category = llm_client.select(query, candidates=categories)

    # Step 3: Tool selection (3–8 options within category)
    tools = TOOL_HIERARCHY[domain][category]
    tool_name = llm_client.select(query, candidates=list(tools.keys()),
                                   descriptions=tools)
    return {"domain": domain, "category": category, "tool": tool_name}
```

### Namespace design principles

Good namespace design follows three rules:

1. **Domains are mutually exclusive by intent, not by topic.** A user asking "what are Tesla's earnings?" is unambiguously in `finance.corporate` — the domain and category should not overlap with `web.search` even though the answer could theoretically be found via web search.

2. **Categories are scoped by output type, not by subject.** `finance.market` produces time-series data; `finance.corporate` produces ratios and text. This makes type-directed multi-hop selection work naturally within the hierarchy.

3. **Tool names are unique across the hierarchy.** Don't have `finance.market.search` and `web.search.search` — this creates ambiguity at the category level that cascades into selection errors.

## 8. Confidence Thresholds: When to Route vs. When to Ask

Not every query maps cleanly to a single tool. When the selection confidence is below a threshold, routing silently to the most likely tool is wrong — it produces a result the user didn't ask for, and they may not notice.

The right behavior depends on how wrong the wrong tool is: if selecting `get_stock_price` when the user meant `get_company_fundamentals` wastes 200ms and returns a wrong answer, that's irritating. If selecting `send_email` when the user meant `search_email` sends an unintended message, that's a serious incident.

### Confidence estimation without a separate model

If you're using in-prompt CoT selection, confidence can be extracted from the reasoning text (as shown in section 3). If you're using a classifier-based router, you get logit scores naturally. If you're using zero-shot selection, you can ask the model to self-assess:

```python
def get_tool_selection_with_confidence(query: str, tools: list[dict], llm_client) -> dict:
    """Returns {tool_name, confidence, reasoning, alternatives}."""
    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""Select the best tool for the user query.
Available tools:
{format_tool_descriptions(tools)}

Return JSON:
{{
  "selected_tool": "<tool_name>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence why>",
  "alternatives": ["<other_tool_if_close>"],
  "ambiguous": <true/false>
}}

Set confidence < 0.7 if multiple tools could work. Set ambiguous=true if you
need more information from the user to be certain."""
            },
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def route_with_threshold(
    query: str,
    tools: list[dict],
    llm_client,
    confidence_threshold: float = 0.75,
    high_risk_tools: set = None
) -> dict:
    """
    Route the query to a tool, or return a clarification request.
    """
    high_risk_tools = high_risk_tools or {"send_email", "create_event", "delete_*"}
    result = get_tool_selection_with_confidence(query, tools, llm_client)

    is_high_risk = any(
        result["selected_tool"].startswith(t.replace("*", ""))
        for t in high_risk_tools
    )
    # Raise threshold for irreversible / high-consequence tools
    effective_threshold = 0.9 if is_high_risk else confidence_threshold

    if result["confidence"] < effective_threshold or result.get("ambiguous"):
        alternatives = result.get("alternatives", [])
        clarification = (
            f"I want to make sure I do the right thing. Did you mean to: "
            f"(a) use {result['selected_tool']}, or "
            f"(b) use {alternatives[0] if alternatives else 'a different tool'}?"
        )
        return {"action": "clarify", "message": clarification}

    return {"action": "execute", "tool": result["selected_tool"]}
```

### Threshold calibration

Setting the threshold is an empirical problem. Too low and you route incorrectly silently. Too high and the agent asks for clarification constantly, which degrades user experience. The right calibration depends on:

- **Consequence asymmetry**: reversible actions (search, read) can tolerate a lower threshold than irreversible ones (write, send, delete).
- **User tolerance for interruptions**: in an automated pipeline with no user watching, a low threshold with clarification routing may be better than silent errors. In an interactive chat, excessive clarification questions are annoying.
- **Catalog ambiguity**: a catalog with many semantically similar tools (20 SQL tools vs. 3 broad data tools) warrants a higher threshold.

A practical starting calibration: 0.85 for read operations, 0.92 for write operations, 0.98 for delete/send/publish operations. Tune from production logs by examining the confusion matrix of selections vs. correct labels.

## 9. Fallback Strategies: What to Do When No Tool Matches

When the selection mechanism finds no adequate match, there are four well-defined strategies. Which one to use depends on the failure type.

![Fallback Decision Tree](/imgs/blogs/tool-selection-and-routing-8.webp)

### Strategy 1: Clarification request

When score is in the low-confidence range (0.5–0.85) and the query is interpretably ambiguous, ask the user. This is the correct answer when the user's intent is recoverable with one well-targeted question.

```python
def generate_clarification(
    query: str,
    candidates: list[dict],
    llm_client
) -> str:
    """Generate a targeted clarification question."""
    candidate_names = [c["name"] for c in candidates]
    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"""The user asked: {query}

I have {len(candidates)} candidate tools that might apply: {candidate_names}

Generate ONE specific clarification question that would let me distinguish between
these tools. The question should:
- Be specific to the actual difference between these tools
- Be answerable in one sentence
- Not expose technical tool names to the user

Return just the question."""
        }]
    )
    return response.choices[0].message.content
```

### Strategy 2: Default with warning

When a single low-score candidate exists and the task isn't high-risk, execute with the best match but surface the uncertainty to the user. This is better than silent wrong selection because the user can correct course.

### Strategy 3: Skip and inform

When no candidate scores above 0.5, the right tool may not exist in the catalog. Inform the user rather than fabricating a result. This is especially important for agents used in high-trust contexts (business operations, medical, legal) where a wrong-tool execution is worse than no execution.

### Strategy 4: Tool discovery

For agents with access to a tool registry that's larger than the current active catalog, "no match" can trigger a discovery query: search the broader registry for tools that might match, add the best candidate to the active catalog, and retry selection. This is relevant for agentic systems that manage their own tool availability dynamically.

## 10. Measuring Selection Accuracy: Benchmarking Tool Routing

You cannot improve what you do not measure. Tool selection accuracy is surprisingly rarely measured explicitly in production systems — most teams notice wrong selections only through downstream error rates or user complaints. That's too late.

![Catalog Size vs Selection Accuracy](/imgs/blogs/tool-selection-and-routing-9.webp)

The matrix above makes the story clear: LLM-only selection at 100 tools is 44% accurate — barely better than chance if you have 5–6 semantically similar tools. The performance gap between LLM-only and hybrid strategies widens monotonically with catalog size. If you don't measure this, you won't notice when you add tool number 35 and cross the threshold where your selection starts failing at a rate that matters.

### Building a selection benchmark

```python
import json
from dataclasses import dataclass
from typing import Callable

@dataclass
class SelectionTestCase:
    query: str
    correct_tool: str
    difficulty: str  # "easy" | "medium" | "hard"
    confusion_tools: list[str]  # tools commonly confused with correct_tool

def build_benchmark(
    tool_catalog: list[dict],
    test_cases: list[SelectionTestCase],
    selection_fn: Callable[[str], str]
) -> dict:
    """
    Evaluate a selection function against a labeled test set.
    Returns: {accuracy, by_difficulty, by_confusion_pair, top_errors}
    """
    results = []
    for tc in test_cases:
        predicted = selection_fn(tc.query)
        correct = predicted == tc.correct_tool
        results.append({
            "query": tc.query,
            "correct_tool": tc.correct_tool,
            "predicted": predicted,
            "correct": correct,
            "difficulty": tc.difficulty,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)

    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            by_difficulty[diff] = sum(r["correct"] for r in subset) / len(subset)

    # Top confusion pairs
    errors = [r for r in results if not r["correct"]]
    confusion_pairs = {}
    for e in errors:
        pair = (e["correct_tool"], e["predicted"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    top_errors = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:10]

    return {
        "accuracy": accuracy,
        "total_cases": len(results),
        "by_difficulty": by_difficulty,
        "top_confusion_pairs": top_errors,
        "error_rate": 1 - accuracy,
    }
```

### Production monitoring

A benchmark gives you offline accuracy. Production monitoring gives you realtime drift detection. The key signals:

**Tool execution error rate by tool**: if `get_stock_price` starts failing more often, it may be because the selection mechanism is routing queries intended for other tools to it (wrong arguments, wrong context).

**User correction rate**: if users frequently follow up with "no, I meant..." after a tool call, that's a selection error. Capture these corrections as labeled training data for your router.

**Tool usage distribution**: a sudden shift in which tools are called (even if each call succeeds) may indicate selection drift — the mechanism is routing to different tools than it used to for the same queries.

```python
from collections import defaultdict
from datetime import datetime

class SelectionMonitor:
    def __init__(self):
        self.calls = defaultdict(int)
        self.errors = defaultdict(int)
        self.corrections = defaultdict(int)
        self.latencies_ms = defaultdict(list)

    def record_selection(self, tool_name: str, latency_ms: float, success: bool):
        self.calls[tool_name] += 1
        if not success:
            self.errors[tool_name] += 1
        self.latencies_ms[tool_name].append(latency_ms)

    def record_correction(self, original_tool: str, corrected_tool: str):
        self.corrections[(original_tool, corrected_tool)] += 1

    def error_rates(self) -> dict:
        return {
            tool: self.errors[tool] / self.calls[tool]
            for tool in self.calls
            if self.calls[tool] > 10  # ignore low-volume tools
        }

    def top_confusion_pairs(self, n: int = 5) -> list:
        return sorted(self.corrections.items(), key=lambda x: -x[1])[:n]
```

## 11. Case Studies: Routing Failures and Fixes

### Case Study 1: The financial agent with 12% wrong-tool rate

A hedge fund's internal research agent had 47 tools covering price data, fundamental data, news, regulatory filings, and macroeconomic indicators. After three months in production, the team noticed the agent was producing subtly wrong research summaries — it would answer questions about earnings growth using price data tools, and vice versa.

The root cause was description ambiguity. Both `get_stock_price` (which returned OHLCV) and `get_total_return` (which returned annualized return over a period) were described as "stock performance data." When users asked "how has AAPL performed?", the agent would sometimes call `get_stock_price` (returning today's price, not useful for performance analysis) and sometimes `get_total_return` (correct). Selection was essentially random for this query class.

**Fix**: Rewrote all descriptions to include explicit "Use for:" and "Do NOT use for:" clauses, added explicit examples of the queries each tool was intended to answer, and added the output schema to the description. Wrong-tool rate dropped from 12% to 3.1% within a week of the description rewrite, with no model or architecture changes.

**Key lesson**: description quality is often more impactful than routing architecture. Before adding a router model, audit and rewrite all descriptions.

### Case Study 2: The customer support agent that escalated too much

An e-commerce platform's support agent had 22 tools, including `lookup_order`, `initiate_refund`, `send_message_to_user`, and `escalate_to_human`. The agent was escalating roughly 40% of tickets to human agents — far above the intended 10%.

Investigation revealed the issue: `escalate_to_human` had the description "Use when the customer is unhappy or when you cannot solve the problem." This was too broad. Any query that included words like "frustrated," "disappointed," or "this is unacceptable" triggered escalation, even for problems the agent could easily solve.

**Fix**: Changed `escalate_to_human` description to "Use ONLY when: (a) the issue requires account modification you cannot do (e.g. manual override of fraud block), OR (b) the customer explicitly says they want a human. Do NOT use simply because the customer is unhappy — first attempt resolution with the appropriate tool." Escalation rate dropped to 11%.

**Key lesson**: negative description examples — explicitly listing what the tool should NOT be used for — are as important as positive examples.

### Case Study 3: The code assistant that ran tests when it should have searched

A developer-facing coding assistant had tools including `run_tests`, `search_codebase`, `read_file`, `write_file`, `search_documentation`, and `run_shell_command`. When users asked "are there tests for the authentication module?", the agent would call `run_tests` (executing the full test suite) instead of `search_codebase` (looking for test files).

The problem was intent ambiguity: "are there tests?" could mean "do tests exist?" (search) or "do the tests pass?" (run). The agent resolved this ambiguity toward the more expensive/dangerous option because `run_tests` had a more confident match to "tests."

**Fix**: Added a pre-selection intent classifier that distinguished "information queries" (are there / where is / what does / show me) from "action queries" (run / execute / create / delete). Information queries were directed to read-only tools only; action queries required explicit user confirmation above a confidence threshold. This cut unintended test executions from 15/day to 0 while adding only 30ms overhead.

**Key lesson**: intent classification (information vs. action) as a pre-routing filter reduces the risk of irreversible actions from selection ambiguity.

### Case Study 4: The multi-hop agent that got stuck

A research automation agent was given a task: "summarize the key risks from AAPL's most recent 10-K filing." The correct tool chain was: `search_sec_filings(ticker='AAPL', form='10-K')` → `fetch_url(url=<filing_url>)` → `extract_sections(text, section='risk factors')` → `summarize_text(text)`.

The agent got stuck after the second hop. After `fetch_url` returned 80,000 tokens of SEC filing text, the selection mechanism had no criteria for distinguishing `extract_sections` (correct) from `summarize_text` (available but would hallucinate with 80k tokens of raw filing). Both tools accepted text as input. The agent chose `summarize_text` because it more directly matched the original goal word "summarize," producing a low-quality summary without the risk-factors-focused extraction.

**Fix**: Added output-schema annotations to each tool — `fetch_url` returns `{"text": "str [long, unstructured, may exceed context]"}`. The selection prompt was updated to include: "If the current output has type `str [long]`, prefer extraction or chunking tools before summarization." This annotation-based type checking prevented the wrong hop.

**Key lesson**: multi-hop selection needs output-schema annotations, not just tool descriptions. The type of what you have determines what tool to call next.

### Case Study 5: Router model drift

A logistics company's operations agent used a DistilBERT router trained on 6 months of production logs. After a major product update added 15 new tools (for a new delivery tracking system), they deployed the new tools without retraining the router.

Within two weeks, wrong-tool rate for the new tool categories was 71% — the router had never seen these tools and defaulted to its nearest-neighbor in the old distribution, which was the old (deprecated) tracking tool. Downstream, dispatchers were looking at stale tracking data.

**Fix**: Added a mandatory "router coverage test" to the tool deployment pipeline: before any new tool is added to the catalog, generate 50 synthetic queries that should route to it and verify the router scores it in the top-3 for ≥ 80% of those queries. If it doesn't, retrain before deploying. The test takes ~5 minutes and caught 3 additional coverage gaps before they reached production.

**Key lesson**: router models go stale when the tool catalog changes. Treat router retraining as a required step in any tool deployment pipeline.

### Case Study 6: Confidence threshold miscalibration

A legal document analysis agent had 35 tools. The team set a confidence threshold of 0.7 to avoid clarification overload. This seemed reasonable until a production incident: a user asked to "delete the draft version of the NDA" and the agent, with 0.73 confidence, called `permanent_delete_document` instead of `archive_document`. The correct threshold for irreversible actions should have been 0.95.

**Fix**: Implemented tool-class-specific thresholds: read/search tools at 0.7, write/update tools at 0.85, delete/publish/send tools at 0.95. Added a "destructive action confirmation" flow that, regardless of confidence, requires the user to confirm any action tagged `destructive: true` in the tool schema.

**Key lesson**: a single confidence threshold is wrong for catalogs that mix read, write, and destructive operations. Threshold should be proportional to consequence.

### Case Study 7: The description language mismatch

A Japanese-language customer service agent had tool descriptions written in English. The tools themselves worked on Japanese text, but their descriptions were English. When Japanese users sent queries, the embedding model mapped Japanese query text far from the English description embeddings, causing the semantic search to return poor candidates. Selection accuracy was 52% — catastrophically low.

**Fix**: Added Japanese translations of all tool descriptions to the index, interleaved with the English versions. Both languages mapped into the shared multilingual embedding space, and Japanese queries could now match Japanese descriptions. Accuracy improved to 89%.

**Key lesson**: for multilingual agents, tool descriptions must be indexed in all languages the users will query in. A shared multilingual embedding model (like multilingual-e5) helps, but parallel-language descriptions are more reliable.

### Case Study 8: The tool schema update that broke routing

A data platform agent had a popular tool `query_database(sql: str)`. After a security audit, the team renamed it `run_read_only_sql_query(query: str, timeout_s: int = 30)` — different name, different parameter names, same functionality. They updated the tool schema but forgot to update the router model, which had been trained on thousands of examples of `query_database` calls.

The router continued routing database queries to `query_database`, which no longer existed. Every database query failed with "tool not found." The failure was silent for 4 hours because the router confidence scores were high — it was very confident about the old tool name.

**Fix**: Added tool-name validation to the router output: if the router returns a tool name not in the current catalog, fall back to semantic search. Also added a weekly automated check that runs the router benchmark against the current catalog and alerts if any tool in the catalog has < 50% top-3 recall.

**Key lesson**: tool renames are breaking changes for trained routers. Always run a router benchmark after any catalog change, not just additions.

## 12. When to Use Router Models vs. In-Prompt Selection

The decision matrix is governed by four variables: catalog size, latency budget, cost budget, and operational appetite.

![Agent Type to Routing Strategy Map](/imgs/blogs/tool-selection-and-routing-10.webp)

The grid above gives concrete recommendations by agent type. The pattern that emerges:

**Use in-prompt CoT selection when:**
- Catalog is ≤ 30 tools
- You don't have labeled (query, tool) training data
- The tool catalog changes frequently (retraining is expensive)
- Latency is not a primary constraint (you can afford 500ms+ for selection)
- You need full observability into the selection reasoning

**Use a router model when:**
- Catalog is 50–500 tools
- You have or can generate labeled training data
- Latency is critical (< 200ms per turn)
- Cost at scale matters (large-LLM tokens for 200 tool descriptions add up)
- The catalog is relatively stable (retraining isn't needed constantly)

**Use semantic search (ANN index) when:**
- Catalog exceeds 200 tools
- You don't have labeled data for a classifier
- The catalog changes daily or is dynamically composed
- You need near-zero deployment overhead

**Use hierarchical namespacing when:**
- Any of the above AND the catalog has natural domain groupings
- Users' queries are naturally domain-scoped (they usually ask about one domain per turn)
- You want to reduce the surface area for adversarial inputs (prompt injection that routes to sensitive tools)

**Use hybrid (semantic + LLM reranking) when:**
- You need the best accuracy and can afford the latency
- The catalog has many semantically similar tools within categories
- Precision matters more than recall (you'd rather ask for clarification than make a wrong call)

### The in-prompt CoT sweet spot revisited

In-prompt CoT with well-written descriptions remains the right default for most agents. The operational simplicity is real: no training pipeline, no router hosting, no coverage tests, no drift monitoring. For agents with 10–30 tools and users who can tolerate 500ms+ latency, the engineering cost of a router model is not justified by the marginal accuracy gain.

The inflection point where router models become compelling is typically around 40–50 tools with > 1,000 calls per day. Below that, the cost of maintaining the router (training data, retraining on catalog changes, monitoring for drift) exceeds the cost of the extra LLM tokens.

### Measuring before deciding

The best way to make this decision is to measure your current selection accuracy first. Run a 100-query labeled benchmark against your current catalog. If accuracy is above 92%, your descriptions are good and your catalog isn't too large — in-prompt CoT is fine. If accuracy is below 85%, the question is why: if it's description quality (easy fix), fix descriptions first. If it's catalog size (40+ tools), consider semantic search. If it's latency (you need selection in < 100ms), consider a router model.

The fix sequence: descriptions → namespacing → semantic search → router model. Each step adds operational complexity. Only go to the next step if the previous one didn't get you to your accuracy target.

---

## Putting It Together

Tool selection is the rate-limiting step in agent reliability. You can have perfect tool implementations and a brilliant main LLM, and still produce a broken agent if the selection mechanism consistently picks the wrong tool. The failure modes are subtle — wrong-tool calls often produce plausible-looking outputs that users don't immediately recognize as wrong.

The good news is that the fixes are well-understood:

1. **Rewrite descriptions first.** Explicit "Use for / Do NOT use for" clauses with example queries are the highest-leverage intervention.
2. **Add CoT selection** for catalogs > 10 tools. The reasoning forces the model to engage with descriptions rather than pattern-match on names.
3. **Add confidence thresholds** and calibrate them by tool risk class. Never use a single threshold for a catalog that mixes search and delete operations.
4. **Add fallback logic.** No match should produce a clarification, not a silent wrong call.
5. **Build a benchmark and measure.** You will not know your selection accuracy without measuring it. The benchmark pays for itself the first time it catches a regression before it reaches production.
6. **Graduate to semantic search or a router model** when in-prompt selection stops being accurate enough at your catalog size. The graduation criteria are clear: catalog > 40 tools, accuracy < 90%, or latency < 200ms.

For the adjacent questions — how to design the tool schemas that make selection easier, how to handle tool errors and retries, and how to wire all of this into a full agent execution loop — see [tool schema design principles](/blog/machine-learning/ai-agent/tool-schema-design-principles), [advanced tool use patterns](/blog/machine-learning/ai-agent/advance-tool-use), and [the ReAct pattern deep dive](/blog/machine-learning/ai-agent/react-pattern-deep-dive).

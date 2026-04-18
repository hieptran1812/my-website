---
title: "Effective Context Engineering for AI Agents"
publishDate: "2026-03-15"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  ["ai-agent", "context-engineering", "llm", "anthropic", "prompt-engineering"]
date: "2026-03-15"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Context engineering is the art of strategically managing the limited token budget available to LLMs when building AI agents. This article breaks down techniques like compaction, structured note-taking, and sub-agent architectures that help agents work effectively over long horizons."
---

## Introduction

If you have ever built something with an LLM, you have probably experienced this frustration: your agent works brilliantly on short tasks, but falls apart when conversations get long. It forgets instructions, loses track of goals, or starts hallucinating because its context window is overloaded with irrelevant information.

This is the core problem **context engineering** solves. Anthropic defines it as:

> "The set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference."

While prompt engineering focuses on writing a great system prompt, context engineering is the broader discipline of **strategically curating all the information that enters an LLM's limited attention budget at each step**. It requires what the Anthropic team calls **"thinking in context"** — considering the holistic state available to the LLM at any given time and what potential behaviors that state might yield.

This article is my notes and breakdown of [Anthropic's guide on effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents), with additional examples and explanations to make each concept easy to grasp.

## From Prompt Engineering to Context Engineering

Before we dive in, it is important to understand how context engineering evolves beyond prompt engineering:

| Aspect              | Prompt Engineering           | Context Engineering                                                           |
| ------------------- | ---------------------------- | ----------------------------------------------------------------------------- |
| **Scope**           | Crafting the system prompt   | Managing the entire context state                                             |
| **Focus**           | One-time instruction writing | Continuous, cyclic curation across turns                                      |
| **Components**      | System prompt text           | System prompt + tools + MCP + external data + message history + model outputs |
| **Nature**          | Discrete optimization        | Iterative, loop-based refinement                                              |
| **When it matters** | Single-turn tasks            | Multi-turn agents operating in loops                                          |

When agents operate in loops, they generate accumulating data that could be relevant at future turns. Context engineering becomes the discipline of determining **which information enters the limited context window** from this expanding universe of possibilities. It is not just about what you write upfront — it is about what the model sees at every single inference step.

## What Is Context, Exactly?

**Context = all tokens provided to an LLM during a single inference call.** This includes:

- **System prompt** (your instructions and background information)
- **Tool definitions** (function schemas the model can call, including MCP tools)
- **Message history** (the full conversation so far — user messages, assistant responses, tool calls and results)
- **Retrieved data** (documents, search results, database records pulled in via RAG or tools)
- **Model's own previous outputs** (its prior reasoning, summaries, and notes)

Think of context like a desk. You can only fit so many papers on it. If you pile everything on, you will not be able to find the one document you actually need. Context engineering is about keeping the desk organized — ensuring the right papers are on top when the model needs them.

## Why Does This Matter? Context Rot and Attention Scarcity

Here is the key insight: **as context grows, model performance degrades**. This is called **context rot**.

Research on "needle-in-a-haystack" benchmarks shows that when you hide a specific fact deep inside a long context, models become less accurate at recalling it as the total token count increases. As Anthropic puts it:

> "As the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases."

This means context must be treated as a **finite resource with diminishing marginal returns**. More tokens does not always mean better performance — in fact, it often means worse.

### The Architectural Reason: Why Transformers Struggle with Long Contexts

LLMs use the **transformer architecture**, where every token attends to every other token. For `n` tokens, that creates `n²` pairwise relationships. This design creates a natural tension between context size and attention focus:

1. **Attention gets diluted** — Like humans with limited working memory, LLMs have an "attention budget." Each token's share of that budget is spread thinner as context grows. With 1,000 tokens, each token gets a reasonable share. With 100,000 tokens, the model must distribute attention across 100x more information.

2. **Training distribution mismatch** — During training, models see shorter sequences far more often than longer ones. This means the model develops strong attention patterns for short-range dependencies but has fewer specialized parameters for context-wide dependencies. The model literally has less experience reasoning over long contexts.

3. **Position encoding degradation** — Techniques like position encoding interpolation allow models to handle longer sequences than they were originally trained on. But this comes at a cost: position understanding degrades. The model knows that two tokens are "far apart" but loses precision about exactly how far.

4. **No hard cliff, but a gradient** — The model does not suddenly break at some token count. Instead, there is a gradual performance degradation. The model remains capable but shows reduced precision for information retrieval and long-range reasoning compared to shorter contexts.

**Analogy**: Imagine you are in a meeting with 5 people vs. 500 people. In the small meeting, you catch every detail — who said what, the nuances, the subtext. In the large one, you catch the gist but miss specifics. You might confuse who raised which point, or lose track of earlier discussions. LLMs work the same way.

**The practical implication**: Even as context windows grow to millions of tokens, **context pollution and information relevance concerns persist at all sizes**. A 1M-token context window does not solve context engineering — it makes it more important.

### Measuring Context Rot in Your System

Context rot is not a belief — it is measurable. Three tests worth adding to CI for any serious agent:

**Test 1: Needle-in-haystack with varying context sizes.**
Insert a specific fact ("the user's account number is 47293") at the same position in context windows of varying size (2K, 8K, 32K, 100K). Ask the agent a question that requires retrieving that fact. Plot recall accuracy as a function of context size. A healthy agent shows gradual degradation; a steep cliff around a specific size usually means position encoding trouble or training mismatch. Trade-off: this test is cheap to run and tells you your *maximum usable* context, which is typically well below the model's advertised limit.

**Test 2: Position-sensitivity test.**
Same needle, same total context size, but placed at start / 25% / 50% / 75% / end. Models typically recall best at start and end, worst in the middle (the "lost in the middle" effect). The *slope* of accuracy across positions tells you how forgiving your layout can be. Trade-off: if the slope is steep, you must be disciplined about placing critical info at start or end — you can't rely on the middle at all.

**Test 3: Distractor injection.**
Add realistic but irrelevant content between the needle and the query. If recall drops significantly with distractors that a human would ignore, your agent is brittle to natural context bloat. Trade-off: this catches a failure mode that pure needle-in-haystack misses — real systems don't have clean contexts.

All three take ~100–500 model calls to run, so you can afford them at release cadence. Skipping them is common; catching a rot regression in production rather than CI is expensive. The investment ratio (hours to set up vs. weeks of production debugging avoided) is strongly favorable.

## The Anatomy of Effective Context

### 1. System Prompts: Finding the Right Altitude

The system prompt is your agent's "operating manual." The Anthropic team describes the challenge as finding the **"right altitude"** — a spectrum between two failure modes:

```
Brittle ◄────────────── Optimal Zone ──────────────► Vague

Complex hardcoded          Specific enough to         High-level guidance
logic covering every       guide behavior, flexible   that assumes shared
edge case. Fragments       enough to provide strong   context the model
when reality deviates      heuristics for edge        doesn't actually
from the script.           cases.                     have.
```

Here is what each extreme looks like in practice:

| Problem            | Example                                                                                                                                                                                                                        | Why It Fails                                                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Over-specific**  | "If the user says 'refund' and their order is < 30 days old and the item is undamaged and they have not had a refund in the last 90 days, then approve. If the order is 30-45 days old, escalate to tier 2. If 45-60 days..."  | Brittle. Breaks on any edge case not listed. What happens at exactly 30 days? What if the item is partially damaged? The model cannot generalize.                                         |
| **Under-specific** | "Handle refund requests appropriately."                                                                                                                                                                                        | Vague. The model guesses what "appropriate" means based on its training data. Different models, or even different samples from the same model, may interpret this completely differently. |
| **Just right**     | "Process refund requests by verifying order date and item condition. Approve refunds for orders within 30 days with undamaged items. Escalate unclear cases to human review. When in doubt, prioritize customer satisfaction." | Clear rules for common cases + flexible heuristics for edge cases + a fallback principle for ambiguity.                                                                                   |

**Best practices for system prompts:**

**Organize into clear sections** using XML tags or Markdown headers. This helps the model parse your instructions and find relevant guidance:

```xml
<background_information>
You are a customer support agent for TechCo, an electronics retailer.
Our return policy allows refunds within 30 days for undamaged items.
We prioritize customer satisfaction and have a "when in doubt, approve" culture.
Our target resolution time is under 5 minutes per ticket.
</background_information>

<instructions>
1. Greet the customer warmly and identify their issue
2. Look up their order using the order_lookup tool
3. For returns: verify order date and item condition against our policy
4. For exchanges: check inventory availability before promising
5. Escalate to human review if the situation involves:
   - Orders over $500
   - Repeat return customers (3+ returns in 90 days)
   - Items showing signs of intentional damage
6. Always confirm the resolution with the customer before closing
</instructions>

<tool_guidance>
- Use order_lookup for any order-related queries. Always look up before making claims about order status.
- Use inventory_check before promising exchanges or replacements.
- Use escalate_to_human when the case meets escalation criteria. Include a brief summary of the issue.
</tool_guidance>

<output_format>
- Respond in a friendly, professional tone
- Use bullet points for action items
- Always end with "Is there anything else I can help with?"
- Never reveal internal policy details or escalation criteria to the customer
</output_format>
```

**Start minimal, then iterate.** Anthropic recommends:

> "It's best to start by testing a minimal prompt with the best model available to see how it performs on your task, and then add clear instructions and examples to improve performance based on failure modes found during initial testing."

Do not write a 3,000-word system prompt on day one. Start with 200 words, test it, see where the model fails, and add specific guidance for those failure modes.

**The minimal information test**: Ask yourself, "Could a smart person with no domain knowledge follow these instructions and produce the right output?" If yes, your prompt is likely at the right altitude.

### 2. Tools: Your Agent's Hands and Eyes

Tools let agents interact with the world: searching databases, calling APIs, reading files, executing code, etc. **Poorly designed tools are one of the most common causes of agent failure** — and one of the most overlooked aspects of context engineering.

Remember: tool definitions live in the context window too. Every tool schema, description, and parameter takes up tokens. A bloated tool set with 50 overlapping tools does not just confuse the model — it wastes precious context space.

**The five principles of good tool design:**

#### Principle 1: Token-Efficient Returns

Every token a tool returns consumes context budget. Design tools to return only what the agent needs.

```python
# BAD: Returns entire user object (2000+ tokens)
def get_user(user_id: str) -> dict:
    return database.users.find(user_id)  # Returns 50+ fields

# GOOD: Returns only requested fields
def get_user(user_id: str, fields: list[str]) -> dict:
    user = database.users.find(user_id)
    return {k: user[k] for k in fields if k in user}

# Even better for common patterns: purpose-specific tools
def get_user_contact_info(user_id: str) -> dict:
    """Returns name, email, and phone for a user."""
    user = database.users.find(user_id)
    return {"name": user["name"], "email": user["email"], "phone": user["phone"]}
```

#### Principle 2: Clear, Single Purpose

Each tool should do one thing well. If you find yourself writing "or" in a tool description, you probably need two tools.

```python
# BAD: Swiss-army-knife tool
def manage_order(action: str, order_id: str, data: dict):
    """Create, update, cancel, or refund an order."""
    if action == "create": ...
    elif action == "update": ...
    elif action == "cancel": ...
    elif action == "refund": ...

# GOOD: Separate tools with clear purposes
def create_order(items: list, customer_id: str): ...
def cancel_order(order_id: str, reason: str): ...
def process_refund(order_id: str, amount: float): ...
```

#### Principle 3: Minimal Overlap

If two tools can accomplish the same task, the model has to decide between them — and it might choose wrong.

```
# BAD: Which one should the model use?
- search_products(query: str)        # "Search products by name or description"
- find_products(keywords: list[str]) # "Find products matching keywords"
- lookup_products(category: str)     # "Look up products in a category"

# GOOD: One tool with flexible parameters
- search_products(query: str, category: str | None = None)
  # "Search products by natural language query. Optionally filter by category."
```

#### Principle 4: Descriptive, Unambiguous Parameters

The model reads parameter descriptions to decide what values to pass. Vague descriptions lead to wrong inputs.

```json
{
  "name": "lookup_order",
  "description": "Look up a customer order by order ID or customer email. Returns order status, items, dates, and shipping info. Use this whenever a customer asks about an order.",
  "parameters": {
    "order_id": {
      "type": "string",
      "description": "The unique order identifier, formatted as ORD-XXXXX (e.g., ORD-12345). Preferred over email when available."
    },
    "customer_email": {
      "type": "string",
      "description": "Customer's email address. Use this only when order_id is not available. Will return the most recent order for this email."
    }
  }
}
```

#### Principle 5: Robust Error Handling

Tools should return clear, actionable error messages — not stack traces or cryptic codes.

```python
# BAD: Raw exception
def lookup_order(order_id: str):
    return database.orders.find(order_id)  # Throws KeyError if not found

# GOOD: Clear error message the model can act on
def lookup_order(order_id: str):
    order = database.orders.find(order_id)
    if order is None:
        return {"error": f"No order found with ID '{order_id}'. Please verify the order ID with the customer."}
    return order
```

**The litmus test from Anthropic:**

> "If a human engineer can't definitively say which tool should be used in a given situation, an AI agent can't be expected to do better."

#### Tool Count Trade-offs — When to Split, When to Merge

The *number* of tools in scope is as important as any individual tool's design. A worked comparison:

| Toolbox shape | Context cost | Accuracy | Maintenance | When it fits |
| --- | --- | --- | --- | --- |
| 50 flat tools (everything) | ~8–15KB of tool defs | Drops sharply — models over-pick or mis-pick from similar-sounding tools | High — any tool-naming drift causes confusion | Rare; usually indicates missing progressive disclosure |
| 15 curated tools | ~3–5KB | High — models choose well at this size | Medium — requires someone to curate | Most production agents |
| 5 atomic tools | ~1KB | Highest per-tool, but agent does more composition work | Low per tool, but composition work can hurt | Simple single-domain agents |

Concrete heuristic: **if >15 tools are in scope at a single step, accuracy starts falling noticeably**. Remedies, in order of preference:

1. **Stage-aware tool sets.** Expose only tools relevant to the current phase ("while reading code, no write tools are available").
2. **Retrieval-selected tools.** Embed tool descriptions; retrieve top-K tools matching the current user intent.
3. **Hierarchical tools.** A meta-tool that lists available sub-tools for a category ("`search_*` family: search_code, search_docs, search_issues").
4. **Sub-agent split.** If two tool categories rarely co-occur (writing code vs. writing SQL), give each its own agent with its own tool slice.

Trade-offs of each: stage-aware tool sets require predictable stages; retrieval-selected adds one model call per step; hierarchical adds one level of indirection the model must learn; sub-agent split adds coordination cost. Pick the cheapest fix that solves your specific bloat.

Anti-pattern: merging tools to "reduce count" at the cost of action clarity. `manage_everything(action, payload)` hides the selection problem from the agent's tool-choice step and pushes it into argument construction where errors are harder to catch.

### 3. Examples: The Pictures Worth a Thousand Words

Instead of writing a 500-word description of how your agent should behave, show it a few examples. As Anthropic puts it:

> "For an LLM, examples are the 'pictures' worth a thousand words."

The key insight is to **avoid creating a laundry list of edge cases**. Instead, curate a set of **diverse, canonical examples** that effectively portray the expected behavior across the main categories.

**What makes a good example set:**

1. **Coverage of main categories** — If your agent handles returns, exchanges, and inquiries, include one example of each
2. **Demonstration of tool usage** — Show when and how to call tools
3. **Demonstration of edge case handling** — Show how to handle ambiguity
4. **Correct tone and format** — The model mirrors the style of your examples

```xml
<examples>

<example name="standard_return">
<user_message>I want to return my headphones, order #ORD-98765</user_message>
<agent_response>
I'd be happy to help you with your return! Let me look up your order.

[Calls lookup_order(order_id="ORD-98765")]
[Result: Wireless headphones, ordered March 1st, $79.99, delivered March 5th]

I found your order for wireless headphones placed on March 1st. Since this is within our 30-day return window, I can process this for you right away.

I've initiated the return. Here's what to expect:
- You'll receive a prepaid shipping label at your email within 24 hours
- Once we receive the item, your refund of $79.99 will be processed within 3-5 business days

Is there anything else I can help with?
</agent_response>
</example>

<example name="ambiguous_escalation">
<user_message>I bought a laptop 6 weeks ago and the screen just cracked. I didn't drop it though, it just happened. I want a full refund.</user_message>
<agent_response>
I'm sorry to hear about the issue with your laptop screen! Let me look into this for you.

[Calls lookup_order(customer_email="...")]
[Result: Laptop, ordered Feb 1st, $1,299, delivered Feb 5th]

I can see your laptop was purchased about 6 weeks ago. Since this is outside our standard 30-day return window and involves potential hardware damage, I want to make sure we handle this properly for you.

I'm going to connect you with a specialist who can review this as a potential warranty or defect case, since a screen cracking without impact could indicate a manufacturing issue.

[Calls escalate_to_human(summary="Customer reports laptop screen cracked without physical impact, 6 weeks after purchase. Potential manufacturing defect. Order value $1,299.")]

A specialist will be with you shortly. They'll be able to review whether this qualifies for a warranty replacement. Is there anything else I can help with in the meantime?
</agent_response>
</example>

<example name="multi_tool_exchange">
<user_message>Can I exchange my blue jacket for a red one in size medium?</user_message>
<agent_response>
I'd love to help you exchange your jacket! Let me check on that.

[Calls lookup_order(customer_email="...")]
[Result: Blue jacket size L, ordered March 10th, $120]

[Calls inventory_check(product="jacket", color="red", size="medium")]
[Result: In stock, 3 available]

Great news! I found your blue jacket order and we do have the red jacket in medium in stock. Since your original order is within our 30-day window, I can set up this exchange right away.

Here's the plan:
- I'll send you a prepaid return label for the blue jacket
- Once it ships back, we'll send out the red medium jacket
- No additional charges since it's the same price

Is there anything else I can help with?
</agent_response>
</example>

</examples>
```

Notice how each example demonstrates different behaviors: a straightforward approval, an escalation with reasoning, and multi-tool usage. Three examples communicate more than pages of written rules.

## Context Retrieval: Just-in-Time vs. Pre-loaded

A fundamental design decision for any agent is **when** and **how** to load information into context. The field is converging on a simple but powerful agent definition:

> "LLMs autonomously using tools in a loop."

This loop-based nature makes context retrieval strategy critical.

### Pre-loaded (Traditional RAG)

Load relevant documents into context before the model starts reasoning.

```
User query → Embedding search → Top-K docs → LLM (with docs in context)
```

**How it works**: You pre-compute embeddings for your document corpus. When a query comes in, you find the K most similar documents and stuff them into the context alongside the query.

**Pros**: Fast (single inference call), predictable latency, simple architecture
**Cons**: May load irrelevant docs (wasting tokens), retrieval quality depends entirely on embedding similarity, cannot discover information it did not know to search for, results may be stale

### Just-in-Time (Agentic Search)

The agent decides what to load and when, using tools dynamically.

```
User query → LLM reasons about what it needs → Calls search tool →
Gets results → Reasons about gaps → Calls another tool → ... → Final answer
```

**How it works**: Instead of pre-loading context, the agent maintains **lightweight identifiers** — file paths, stored queries, web links — and uses these references to dynamically load data into context at runtime when it determines the information is needed.

**Pros**: Only loads what is actually needed, adapts dynamically to evolving requirements, can discover unexpected information, stays current
**Cons**: Slower (multiple inference calls), agent might search poorly without guidance, can waste context on failed searches

Anthropic describes this as:

> "This approach mirrors human cognition: we generally don't memorize entire corpuses of information, but rather introduce external organization and indexing systems like file systems, inboxes, and bookmarks to retrieve relevant information on demand."

### Real-world Example: Claude Code's Just-in-Time Approach

Claude Code demonstrates the just-in-time approach for complex data analysis and coding tasks:

1. **Instead of loading an entire dataset** into context, it writes targeted queries:
   - `head -20 data.csv` to see the structure and column names
   - `wc -l data.csv` to understand the scale
   - `tail -5 data.csv` to check for formatting issues at the end

2. **It builds understanding incrementally**, storing intermediate results and forming hypotheses that guide further exploration

3. **It uses `glob` and `grep`** to find relevant files rather than loading entire directory trees. For example, to understand how authentication works in a codebase, it might:
   - `glob("**/auth*.{js,ts}")` to find auth-related files
   - `grep("jwt|token|session", "src/**")` to find token handling code
   - Then read only the specific files that matter

This mirrors how experienced developers work: you do not memorize an entire codebase. You keep a mental map of where things are and look them up when needed.

### Progressive Disclosure: How Agents Explore Efficiently

When agents navigate codebases or data autonomously, they benefit from **progressive disclosure** — learning about the environment incrementally through metadata signals before committing context budget to full content.

Consider how much you can learn without reading file contents:

| Signal                  | What It Tells You         | Example                                                                              |
| ----------------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| **File name**           | Purpose and domain        | `test_utils.py` in `tests/` vs. in `src/core_logic/` signals very different purposes |
| **Directory structure** | Domain architecture       | `src/auth/`, `src/payments/`, `src/notifications/` reveals bounded contexts          |
| **File size**           | Complexity level          | A 5,000-line file suggests complexity; a 50-line file suggests a utility or config   |
| **Timestamps**          | Relevance to active work  | Recently modified files are more likely relevant                                     |
| **Naming conventions**  | Architectural patterns    | `*_handler.py`, `*_service.py`, `*_repository.py` reveal layered architecture        |
| **Import statements**   | Dependencies and coupling | First few lines reveal what a module depends on without reading the body             |

Agents can **assemble understanding layer by layer**, maintaining only what is necessary in working memory:

```
Step 1: ls src/              → See top-level modules (5 tokens)
Step 2: ls src/auth/         → See auth file names (10 tokens)
Step 3: head -5 src/auth/middleware.js → See imports and first function (20 tokens)
Step 4: Read full middleware.js         → Only if actually needed (500 tokens)
```

This progressive approach is far more token-efficient than loading everything at once.

### Hybrid Strategy (Best of Both Worlds)

In practice, the best approach is often a **hybrid**: load critical context upfront for speed, then let the agent explore further as needed.

```
┌───────────────────────────────────────────────────┐
│              Agent Context Window                  │
│                                                    │
│  [Pre-loaded at start]     [Loaded on demand]      │
│  ┌──────────────────┐     ┌──────────────────┐    │
│  │ System prompt     │     │ File contents    │    │
│  │ CLAUDE.md config  │     │ Search results   │    │
│  │ User profile      │     │ API responses    │    │
│  │ Core schema info  │     │ Database queries │    │
│  │ Recent errors     │     │ Stack traces     │    │
│  └──────────────────┘     └──────────────────┘    │
│                                                    │
│  Fast startup, always       Loaded only when       │
│  available                  the agent determines   │
│                             it's needed             │
└───────────────────────────────────────────────────┘
```

**Claude Code uses this hybrid approach:**

- **Pre-loaded**: `CLAUDE.md` files are dropped into context at startup, providing project-specific conventions, architecture notes, and common commands
- **Just-in-time**: `glob` and `grep` primitives enable on-demand file retrieval, bypassing stale indexing and complex syntax tree issues

This avoids two common pitfalls:

1. **Stale indexes** — Pre-computed embeddings become outdated as code changes. Grep always searches the current state.
2. **Index granularity** — Embedding-based retrieval might return an entire file when only one function matters. Agentic search can target exactly the right lines.

### Trade-offs: Speed vs. Accuracy

The choice between pre-loaded and just-in-time involves a clear trade-off:

```
Pre-loaded (RAG)                    Just-in-Time (Agentic)
─────────────────                   ─────────────────────
Fast ◄──────────────────────────────────────────────► Slow
Lower accuracy ◄────────────────────────────► Higher accuracy
Simple ◄────────────────────────────────────────► Complex
Predictable ◄──────────────────────────────► Adaptive
```

Without proper guidance, agentic search can waste context through tool misuse or dead-end chasing. The solution: **give agents clear instructions about what tools to use and when**, along with strategies for efficient exploration (e.g., "start with directory structure before reading files").

### Pre-loaded vs Just-in-Time — A Decision Framework

The choice is governed by three orthogonal axes: *latency sensitivity*, *data staleness tolerance*, and *information shape predictability*. Four concrete cases:

**Customer support agent → Pre-loaded.**
Latency: user is waiting — sub-2s target. Staleness: policies change quarterly, safe to cache. Shape: policies and escalation rules are known and small. Architecture: load policy docs, escalation tree, and tone guidelines at system prompt time. Tool calls are reserved for user-specific lookups (order status). Trade-off: first deployment costs the whole policy corpus in every prompt (mitigated by cache); when policies change, you redeploy.

**Coding agent in an unfamiliar repo → Just-in-time.**
Latency: user tolerates 10–30s for a thoughtful response. Staleness: code changes by the hour — cached indexes rot. Shape: unpredictable; the agent doesn't know what files matter until it explores. Architecture: `glob`/`grep`/`read_file` primitives, no pre-loaded code. Trade-off: more round-trips; agent occasionally explores dead-ends; the payoff is always-current knowledge.

**Product RAG chatbot → Hybrid.**
Latency: sub-3s target. Staleness: catalog updates daily — a stale embedding is usually close enough. Shape: questions generally map to a small K of relevant docs. Architecture: pre-computed embeddings retrieved at question time, plus a fallback just-in-time web search if embeddings score low. Trade-off: two retrieval paths = two failure modes; operational complexity worth it only at volume.

**Trading bot → Pure JIT with real-time tools.**
Latency: sub-500ms; every ms costs P&L. Staleness: zero tolerance — stale price is wrong price. Architecture: no pre-loaded data; every prompt pulls fresh market state via specialized low-latency tools. Trade-off: the tool latency is the floor of your response time; everything else is optimization around that fact.

**The quick triage:** *real-time data* and *unpredictable information needs* both push toward JIT. *Small, stable reference material* and *hard latency constraints* push toward pre-loaded. Hybrid is the default for most agents, and the art is deciding which piece of data goes in which layer.

## Managing Long-Horizon Tasks

When agents run for **tens of minutes to hours** — codebase migrations, research projects, multi-step workflows — the context window becomes a critical bottleneck. The generated tokens, tool results, and conversation history accumulate until they exceed the window or degrade performance long before reaching the limit.

Here are three battle-tested strategies, each addressing a different aspect of the problem.

### Strategy 1: Compaction

**What it does**: When the conversation nears the context limit, summarize the history and start a fresh window with the summary. As Anthropic notes:

> "Compaction typically serves as the first lever in context engineering to drive better long-term coherence."

**How it works in practice (Claude Code):**

```
Turn 1-50: Normal conversation, context fills up
│
│  Agent: "I've analyzed auth.js, here are the 47 JWT references..."
│  Agent: "Middleware.js updated, 12 endpoints migrated..."
│  Agent: "Config.js now uses OAuth2 provider settings..."
│  [Tool results: hundreds of lines of code, diffs, test outputs]
│
▼
Compaction trigger: Context approaching limit
│
▼
Summarizer processes entire message history:
"Here's what happened so far:
  - Objective: Migrate auth from JWT to OAuth2
  - We've updated 3 of 7 service files
  - Files completed: auth.js, middleware.js, config.js
  - Remaining: user.js, admin.js, api.js, tests.js
  - Key decisions:
    - Using PKCE flow for mobile clients
    - Refresh tokens stored in httpOnly cookies
    - Session duration: 1 hour access, 7 day refresh
  - Unresolved: admin.js has a custom auth decorator
    that needs special handling
  - Bug found: middleware.js line 142 has a race condition
    in token refresh — fix pending"
│
▼
Turn 51+: Fresh context with:
  - Original system prompt
  - Compressed summary above
  - 5 most recently accessed files (full content)
  - Current task state
```

**The art of compaction** is choosing what to keep versus what to discard. This is a recall vs. precision trade-off:

| Keep (High Recall)                | Discard (High Precision)                       |
| --------------------------------- | ---------------------------------------------- |
| Current objective and sub-goals   | Raw tool outputs already processed             |
| Key decisions and their reasoning | Verbose error logs from resolved issues        |
| Files modified and their state    | Exploratory dead ends                          |
| Unresolved bugs and blockers      | Intermediate reasoning that led to conclusions |
| Architectural decisions           | Code snippets already written to files         |

**Anthropic's recommendation**: Start aggressive on recall (keep more), then iterate to improve precision (discard noise) based on what causes problems.

**The lightest form of compaction** — and the safest starting point — is **tool result clearing**: once a tool has been called deep in the message history, why would the agent need to see the raw result again? The model's summary of that result (in its response) is usually sufficient. This technique was recently launched as a feature on the Claude Developer Platform.

**Example of tool result clearing:**

```
Before clearing (3000+ tokens):
  Turn 12: [tool_call: grep "auth" src/**]
  Turn 12: [tool_result: 847 lines of grep output...]  ← 3000 tokens
  Turn 12: [assistant: "I found 23 auth references across 8 files. The main
            auth logic is in src/auth/middleware.js and src/auth/token.js..."]

After clearing (saves ~3000 tokens):
  Turn 12: [tool_call: grep "auth" src/**]
  Turn 12: [tool_result: <cleared>]  ← 0 tokens
  Turn 12: [assistant: "I found 23 auth references across 8 files..."]  ← preserved
```

The model's summary remains, preserving the semantic content while reclaiming thousands of tokens. This is low-risk because the model already processed and summarized the data.

#### Compaction Failure Modes — Three Real Issues

Compaction is lossy by design. Three specific ways it silently damages agents:

**(1) The summarizer drops the task goal.**
Symptom: after compaction, the agent continues executing but in a subtly wrong direction — it picks up the most recent sub-goal and treats it as the root goal. Root cause: summarization prompts that emphasize "what happened" over "what are we trying to do." Mitigation: make the original goal + user intent a non-summarizable anchor that gets prepended verbatim to every compacted context. Trade-off: ~50–200 tokens of fixed overhead per turn post-compaction; worth every one of them.

**(2) Summarizer hallucinates facts that were never in the original.**
Symptom: the agent later references "decisions" that no one made, or "facts" not present in the pre-compaction context. Root cause: LLM summarizers under-specify — they smooth over gaps with plausible-sounding content. Mitigation: summarizer prompts that require each claim to cite a step or tool call ID; drop claims without citations; cheap verifier pass that samples claims and checks the pre-compaction history. Trade-off: lower compression ratio (you're not allowed to generalize across events), but higher fidelity.

**(3) Compaction triggered mid-important-reasoning.**
Symptom: the agent was five steps into a tight chain of deductions when context hit the threshold; post-compaction, the intermediate reasoning is summarized and the conclusion is lost. Root cause: threshold-based triggers ignore local reasoning state. Mitigation: add "reasoning-checkpoint" markers — the agent can emit a structured "don't compact past here yet" token, or compaction is disallowed inside tool-call loops. Trade-off: context pressure returns sooner; occasionally the agent has to stop mid-task. Better than silent data loss.

Aggressive vs conservative trade-off: **aggressive compaction (keep 20% of history) means more loss but more headroom. Conservative (keep 60%) preserves fidelity but context pressure returns in 50 turns.** Tune on your specific task's reasoning depth. Coding tasks typically want conservative (long chains of interdependent deductions); retrieval-style chatbots tolerate aggressive.

### Strategy 2: Structured Note-Taking (Agentic Memory)

**What it does**: The agent regularly writes notes to files **outside the context window** and reads them back when needed. This provides persistent memory with minimal context overhead.

This is like giving the agent a notebook. Instead of trying to remember everything in working memory (the context window), it writes things down and refers back to them. As Anthropic describes it:

> "Agents can assemble understanding layer by layer, maintaining only what's necessary in working memory and leveraging note-taking strategies for additional persistence."

**Why this is powerful**: After context resets (due to compaction or new conversations), the agent can read its own notes and resume exactly where it left off. This enables coherence across summarization steps that would be impossible when keeping all information in context.

> "This coherence across summarization steps enables long-horizon strategies that would be impossible when keeping all the information in the LLM's context window alone."

**Example: Claude Code's To-Do Lists**

During complex multi-file refactoring, Claude Code maintains structured notes:

```markdown
# Migration Progress - Auth System

## Status: In Progress (3/7 files complete)

## Completed

- [x] auth.js - Replaced JWT verification with OAuth2 token validation
- [x] middleware.js - Updated auth middleware to use new token format
- [x] config.js - Added OAuth2 provider settings (Google, GitHub)

## Remaining

- [ ] user.js - 8 JWT references, needs token refresh logic
- [ ] admin.js - Custom auth decorator (needs special handling, see note below)
- [ ] api.js - 15 endpoint auth checks to update
- [ ] tests/ - All auth tests need rewriting

## Key Decisions

- Using PKCE flow for mobile clients (more secure for public clients)
- Refresh tokens in httpOnly cookies (prevents XSS token theft)
- Access token TTL: 1 hour, Refresh token TTL: 7 days

## Blockers

- admin.js has a custom @require_admin decorator that bypasses standard
  middleware. Need to understand all code paths before modifying.

## Notes

- middleware.js line 142: potential race condition in concurrent token
  refresh. Added mutex but needs load testing.
```

**Example: Claude Playing Pokemon**

Anthropic demonstrated an agent playing Pokemon that maintained remarkably precise notes across thousands of game steps. The agent tracked things like:

> "For the last 1,234 steps I've been training my Pokémon in Route 1, Pikachu has gained 8 levels toward the target of 10."

Its notes looked something like this:

```markdown
## Current Objective

Train Pikachu to level 25 in Route 1 (currently level 17, +8 levels in 1,234 steps)

## Map Notes

- Route 1: Grass patches at coordinates (12,5), (15,8), (20,3)
- Viridian City: Poke Center at north entrance, Mart at south
- Route 2: Leads to Viridian Forest, higher level Pokemon
- Pewter City: First gym, Brock uses Rock-type Pokemon

## Battle Strategy

- Pikachu: Lead with Thunder Shock against water/flying types
- Switch to Charmander for grass types
- Heal at 30% HP, don't risk fainting
- Avoid Geodude encounters (Thunder Shock ineffective)

## Completed Objectives

- [x] Get starter Pokemon (Charmander)
- [x] Deliver Oak's Parcel
- [x] Get Pokedex
- [x] Catch Pikachu in Viridian Forest

## Inventory

- Poke Balls: 7
- Potions: 3
- Antidote: 1
```

The remarkable thing: the agent developed this note-taking structure **without being explicitly told how to structure its memory**. It created region maps, tracked achievements, and maintained combat strategies on its own. After context resets, reading its own notes allowed it to **seamlessly continue multi-hour gameplay** without losing track of progress.

**How to implement note-taking in your agents:**

```python
from pathlib import Path
import json
from datetime import datetime

class AgentMemory:
    """Simple file-based memory system for agent persistence."""

    def __init__(self, memory_dir: str = "./agent_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

    def save_note(self, key: str, content: str, metadata: dict = None):
        """Save a note that persists across context resets."""
        note = {
            "content": content,
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        (self.memory_dir / f"{key}.json").write_text(
            json.dumps(note, indent=2)
        )

    def read_note(self, key: str) -> str | None:
        """Read a previously saved note."""
        path = self.memory_dir / f"{key}.json"
        if path.exists():
            note = json.loads(path.read_text())
            return note["content"]
        return None

    def list_notes(self) -> list[str]:
        """List all available note keys."""
        return [p.stem for p in self.memory_dir.glob("*.json")]

    def append_to_note(self, key: str, new_content: str):
        """Append to an existing note (useful for logs and progress)."""
        existing = self.read_note(key) or ""
        self.save_note(key, existing + "\n" + new_content)


# Tool definitions for the agent
tools = [
    {
        "name": "save_note",
        "description": (
            "Save a note to persistent memory. Use this to track progress, "
            "decisions, blockers, and anything you'll need after a context "
            "reset. Keys should be descriptive (e.g., 'migration_progress', "
            "'architecture_decisions')."
        ),
        "parameters": {
            "key": {
                "type": "string",
                "description": "A descriptive key for this note"
            },
            "content": {
                "type": "string",
                "description": "The note content in markdown format"
            }
        }
    },
    {
        "name": "read_note",
        "description": (
            "Read a previously saved note. Use this at the start of a "
            "resumed session or when you need to recall a decision or status."
        ),
        "parameters": {
            "key": {
                "type": "string",
                "description": "The key of the note to read"
            }
        }
    },
    {
        "name": "list_notes",
        "description": (
            "List all saved note keys. Use this to see what information "
            "is available from previous sessions."
        ),
        "parameters": {}
    }
]
```

**Anthropic's memory tool**: With the Sonnet 4.5 launch, Anthropic released a **memory tool** in public beta on the Claude Developer Platform — a file-based system for storing and consulting information outside the context window. This enables agents to build knowledge bases over time, maintain project state across sessions, and reference previous work without context saturation.

### Strategy 3: Sub-Agent Architectures

**What it does**: Instead of one agent maintaining state across an entire project, a **lead agent** coordinates specialized **sub-agents**, each operating with its own clean context window.

```
┌──────────────────────────────────────────┐
│              Lead Agent                   │
│  "Migrate auth system to OAuth2"          │
│                                           │
│  High-level plan:                         │
│  1. Understand current auth architecture  │
│  2. Design new OAuth2 flow                │
│  3. Implement changes file by file        │
│  4. Update and run tests                  │
│                                           │
│  Context: Clean, focused on coordination  │
│  Sees: Summaries from sub-agents          │
└───────┬──────────┬──────────┬─────────────┘
        │          │          │
   ┌────▼────┐ ┌───▼────┐ ┌──▼──────┐
   │ Search  │ │ Code   │ │ Test    │
   │ Agent   │ │ Agent  │ │ Agent   │
   │         │ │        │ │         │
   │ Explores│ │ Reads  │ │ Runs    │
   │ entire  │ │ files, │ │ tests,  │
   │ codebase│ │ writes │ │ reads   │
   │ for auth│ │ new    │ │ errors, │
   │ patterns│ │ OAuth2 │ │ fixes   │
   │         │ │ code   │ │ issues  │
   │         │ │        │ │         │
   │ Context:│ │Context:│ │Context: │
   │ 40K tok │ │ 30K tok│ │ 25K tok │
   └────┬────┘ └───┬────┘ └───┬─────┘
        │          │           │
   Summary     Summary    Summary
   (1.5K tok)  (1.2K tok) (1K tok)
   from 40K    from 30K   from 25K
   of work     of work    of work
```

**Why this works:**

1. **Clean context per task** — Each sub-agent starts with a fresh context window focused entirely on its specific task. No distracting history from other tasks polluting attention.

2. **Massive compression** — A sub-agent might explore 40,000 tokens of code and search results but return a 1,500-token summary. The lead agent gets the insights without the noise. This is a **~25x compression ratio**.

3. **Parallel execution** — Multiple sub-agents can run simultaneously, searching different parts of the codebase or tackling different files at the same time. This can dramatically reduce wall-clock time.

4. **Failure isolation** — If a sub-agent goes down a dead end or runs out of context, it does not corrupt the lead agent's state. The lead agent can simply spawn a new sub-agent with adjusted instructions.

5. **Separation of concerns** — Detailed search context remains isolated within sub-agents. The lead agent focuses purely on high-level synthesis and coordination.

**Example: Multi-agent research system**

```python
import asyncio

async def research_task(question: str):
    """Lead agent coordinates specialized sub-agents."""
    lead_agent = Agent(system_prompt="""
        You are a research coordinator. Your job:
        1. Break the question into focused sub-questions
        2. Delegate each to a research agent
        3. Synthesize findings into a coherent answer

        You will receive summaries from sub-agents. Focus on
        identifying connections, contradictions, and gaps.
    """)

    # Step 1: Lead agent plans the research
    sub_questions = await lead_agent.plan(question)
    # e.g., ["What are the current approaches to X?",
    #         "What are the known limitations?",
    #         "What recent papers propose solutions?"]

    # Step 2: Sub-agents research in parallel
    findings = await asyncio.gather(*[
        research_sub_agent(sq) for sq in sub_questions
    ])
    # Each returns ~1-2K token summary from ~20-50K of exploration

    # Step 3: Lead agent synthesizes
    return await lead_agent.synthesize(question, findings)


async def research_sub_agent(question: str) -> str:
    """Sub-agent with its own clean context window."""
    agent = Agent(
        system_prompt=f"""
            Research this specific question thoroughly: {question}

            Use your tools to search for relevant information.
            When done, provide a structured summary including:
            - Key findings (with sources)
            - Confidence level for each finding
            - Open questions or gaps
        """,
        tools=[web_search, read_paper, extract_data]
    )
    # Agent explores extensively within its own clean context
    result = await agent.run()
    # Returns condensed summary — the only thing the lead agent sees
    return result.summary
```

Anthropic's own research showed that this multi-agent approach produced **substantial improvements over single-agent systems** on complex research tasks, as described in their article ["How we built our multi-agent research system"](https://www.anthropic.com/engineering).

#### Sub-Agent Overhead — When It Doesn't Pay

Sub-agents are not a free speed-up. A worked numerical example with 3 subtasks:

```
Scenario A — sub-agent split:
  3 × sub-agent prompt overhead (persona, tools, task):  ~3 × 3K = 9K tokens
  3 × sub-agent exploration/work:                        ~3 × 10K = 30K tokens
  3 × sub-agent summary output:                          ~3 × 500 = 1.5K tokens
  Lead agent: receives 1.5K summaries + plan + synthesis: ~5K tokens
  Total:                                                  ~45.5K tokens
  Wall time (parallel): max(10K sub-agent) ≈ 1 agent's latency

Scenario B — single agent:
  Full task in one context:                               ~30K tokens total
  Wall time: sum of all three subtasks (sequential) or
    parallel tool calls within one agent: variable
```

Sub-agents *cost more* in tokens but *win on wall-time* when parallelism is real. Break-even math:

- **Sub-agent wins** when: subtask size ≥ 5K tokens, compression ratio (raw work → summary) ≥ 5×, and latency matters more than cost.
- **Sub-agent loses** when: subtasks are small (< 2K tokens each — prompt overhead dominates), highly coupled (each needs the others' outputs, killing parallelism), or the handoff prompts are themselves expensive (if you're passing 2K of context per sub-agent, you're not really isolating context).

Three concrete losing cases:

- **Coding assistant debugging a single file.** 200-line file, one goal, tight reasoning loop. Splitting into sub-agents destroys coherence and doesn't parallelize meaningfully.
- **Short-form content generation.** Draft a tweet, draft a caption, draft a headline — each is 100 tokens of output. The overhead of spinning up three sub-agents costs more than the work saved.
- **Tightly interdependent pipeline stages.** Stage B needs A's full output, C needs B's — no parallelism available; the sub-agent cost has no speedup to amortize against.

The heuristic: **sub-agents pay off when each can absorb ≥ 5K tokens of context and return a ≤ 1K summary, AND their work is parallel.** Miss either condition and a single agent with good compaction wins.

### Choosing the Right Strategy

| Strategy        | Best For                                       | Strengths                                                                          | Limitations                                               |
| --------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Compaction**  | Long conversations with lots of back-and-forth | Maintains conversational flow, simple to implement, first lever to pull            | Lossy — subtle context may be lost in summarization       |
| **Note-taking** | Iterative development with clear milestones    | Precise persistence, agent controls what to remember, survives full context resets | Requires good note-taking habits (via prompt engineering) |
| **Sub-agents**  | Complex tasks with parallelizable subtasks     | Clean contexts, parallel execution, failure isolation, massive compression         | Higher latency per delegation, more complex orchestration |

In practice, **the best agents combine all three**:

- **Sub-agents** handle focused work with clean context
- Each sub-agent uses **note-taking** for its own persistence
- **Compaction** keeps individual context windows manageable within long sub-tasks

The Anthropic team's advice:

> "Do the simplest thing that works will likely remain our best advice for teams building agents on top of Claude."

Start with compaction. Add note-taking when you need milestone persistence. Add sub-agents when tasks are complex enough to benefit from parallelism and isolation.

### Combining Strategies — Three Real-World Stacks

Most production agents combine all three strategies, but the *composition* is specific to the task shape.

**Stack 1: Claude Code.**
- Pre-loaded: `CLAUDE.md` in context at startup (project conventions, frequently-used commands).
- Just-in-time: `glob`/`grep`/`read_file` tools for code retrieval as needed.
- Tool result clearing: raw grep/read outputs dropped from context once the assistant has summarized them.
- Agent scratchpad: a todo list the model maintains and consults across long sessions.
- *No sub-agents* — single agent with deep tool access.
- Why this composition: coding requires coherent codebase-wide reasoning, so sub-agents would lose fidelity. The bottleneck is context hygiene (compaction + clearing), not parallelism. Trade-off accepted: long tasks eventually hit the context ceiling; the user can start a new session with notes as hand-off.

**Stack 2: Anthropic's research agent.**
- Pre-loaded: tool definitions and a small orchestrator persona at the lead agent.
- Sub-agents (orchestrator-worker): 3–5 parallel subagents per round, each with clean context, task-specific search tools, and a structured output contract.
- Compaction in the lead: as findings accumulate, the lead compacts older rounds into summary state.
- *No shared scratchpad* — each subagent's summary is the handoff.
- Why this composition: research has separable sub-questions AND benefits from parallelism AND subagent work is verbose (context-bounding matters). Trade-off accepted: ~15× token cost vs single agent; worth it for the quality and latency wins.

**Stack 3: Enterprise customer support supervisor.**
- Pre-loaded: policies, user profile, recent interaction history at each specialist.
- Supervisor agent (cheap model): classifies intent, routes to specialist.
- Per-specialist clean context: specialist agent is spun up per request, doesn't see other specialists' work.
- Persistent user state in external store (Postgres), retrieved into each specialist's prompt.
- Tool result clearing aggressive on API results (specialist summarizes, raw results dropped).
- Why this composition: volume is huge (cost per turn matters), domains are partitioned cleanly (specialists don't need each other's context), and user state outlives the agent loop (Postgres is the source of truth). Trade-off accepted: specialist context is rebuilt each request; no cross-specialist learning without persisting it externally.

The meta-lesson: the strategy composition is a function of the task shape, not a best-practices checklist. A stack that wins for research would bankrupt a high-volume chatbot.

## Putting It All Together: A Context Engineering Checklist

When building an agent, work through these questions:

### System Prompt

- [ ] Is it at the right altitude? (Not too brittle, not too vague)
- [ ] Is it organized into clear sections with XML tags or Markdown headers?
- [ ] Have you tested with a minimal version first and iterated based on failures?
- [ ] Could someone with no domain knowledge follow the instructions?

### Tools

- [ ] Does each tool have a single, clear purpose?
- [ ] Are tool returns token-efficient (only returning what is needed)?
- [ ] Is there no overlap between tools?
- [ ] Are parameter descriptions specific and unambiguous?
- [ ] Can a human engineer always pick the right tool for a given situation?
- [ ] Do tools return clear error messages?

### Context Retrieval

- [ ] Are you loading only what the agent needs, when it needs it?
- [ ] Is critical context pre-loaded for fast startup?
- [ ] Can the agent explore for additional context via tools?
- [ ] Are you leveraging metadata signals (file names, sizes, timestamps) before loading full content?
- [ ] Is the hybrid approach appropriate for your use case?

### Long-Horizon Management

- [ ] Is compaction configured with appropriate recall/precision balance?
- [ ] Are old tool results being cleared to reclaim token space?
- [ ] Does the agent have note-taking tools for milestone persistence?
- [ ] For complex tasks, have you considered sub-agent delegation?
- [ ] After context resets, can the agent recover its state from notes?

## Key Takeaways

1. **Context is a finite, precious resource.** Treat every token like it costs attention — because it does. Even million-token context windows suffer from context rot.

2. **Context rot is real and architectural.** The transformer's n² attention mechanism means longer contexts inherently dilute attention. Less but more relevant context often outperforms more but noisier context.

3. **System prompts need the right altitude.** Specific enough to guide behavior, flexible enough to handle edge cases. Start minimal and iterate based on failure modes.

4. **Design tools for token efficiency.** Return only what the agent needs, not everything you have. Keep tool sets small, non-overlapping, and well-documented.

5. **Prefer just-in-time over pre-loaded context** when accuracy matters more than speed. Let agents explore progressively using metadata signals before committing to full content loading.

6. **For long tasks, layer your strategies.** Start with compaction, add note-taking for milestone persistence, introduce sub-agents for parallelizable complexity. Each addresses a different aspect of the context management challenge.

7. **The guiding principle** from Anthropic:

> "Find the smallest set of high-signal tokens that maximize the likelihood of your desired outcome."

As models become more capable, smarter models require less prescriptive engineering, allowing agents to operate with more autonomy. But even as capabilities scale, treating context as a precious, finite resource will remain central to building reliable, effective agents.

## References

- [Effective Context Engineering for AI Agents - Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [How We Built Our Multi-Agent Research System - Anthropic](https://www.anthropic.com/engineering)
- [Claude Developer Platform - Memory and Context Management Cookbook](https://docs.anthropic.com/)

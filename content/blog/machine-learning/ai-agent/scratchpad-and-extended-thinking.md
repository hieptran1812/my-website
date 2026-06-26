---
title: "Scratchpads and Extended Thinking: The Internal Monologue of Production Agents"
date: "2026-06-27"
description: "How agent scratchpads, chain-of-thought traces, and extended thinking tokens improve reasoning quality — and the token cost you pay for each."
tags: ["ai-agents", "reasoning", "chain-of-thought", "extended-thinking", "llm", "machine-learning", "nlp", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 35
---

Every production agent I have debugged fails in one of two ways. The first is factual: the model asserts something false, and there is no intermediate record to show why. The second is structural: the model generates a plausible-looking output that is wrong by composition — each sentence correct, the chain of inferences broken. Both failure modes share a root cause. The model was asked to go directly from prompt to answer without being given space to think.

A scratchpad is that space. It is a region of tokens — sometimes visible in the message thread, sometimes hidden inside the model — where the agent reasons before committing to a final response. The idea is old. Before transformers, it was called "working memory" in cognitive architectures. What has changed is that we now have two distinct implementations, each with a radically different cost profile, latency curve, and production failure mode. Choosing the wrong one in a latency-sensitive pipeline can blow through your token budget; choosing the wrong one in a high-stakes reasoning task can produce confident wrong answers.

![Three scratchpad positions: no scratchpad, external CoT, internal thinking tokens](/imgs/blogs/scratchpad-and-extended-thinking-1.webp)

The diagram above is the mental model. Three positions, each a different tradeoff. No scratchpad is fastest and cheapest but brittle the moment a task requires more than one inference step. External chain-of-thought puts the reasoning trace directly into the message history where you can read, audit, and constrain it. Internal thinking tokens — the approach used by Claude's extended thinking and OpenAI's o1/o3 family — give the model a hidden scratchpad with the full budget of the context window, at the price of opacity and cost.

This post covers all three positions in depth: how each works mechanically, when each is the right choice, the API details for extended thinking, what actually helps inside a scratchpad versus what wastes tokens, and six production case studies where the wrong choice had measurable consequences.

## 1. The scratchpad concept: trading tokens for reasoning quality

The case for scratchpads rests on a single empirical fact: transformer models trained on next-token prediction are significantly better at retrieving the correct answer when the answer is the last thing generated in a valid reasoning chain than when it is the first.

This was formalized in the original chain-of-thought paper (Wei et al., 2022), which showed that prompting GPT-3 with "Let's think step by step" before the answer raised performance on the GSM8K grade-school math benchmark from roughly 18% to 48% with zero additional training. The mechanism is straightforward: correct answers to multi-step problems are rare tokens in the training distribution when treated as direct completions. They are common tokens when they follow a complete reasoning trace. The model is effectively moving from a hard sampling problem (find the rare correct answer token) to an easier one (find the next correct reasoning step, then the next, until the answer is just a summary of what came before).

The tradeoff is cost. Every token in the reasoning trace is a billable token. A direct "what is 17 × 23?" gets answered in three tokens. With chain-of-thought: "17 × 20 = 340, 17 × 3 = 51, 340 + 51 = 391" adds forty tokens of overhead. For a batch of ten thousand arithmetic queries per day, that is four hundred thousand extra tokens — meaningful money at scale.

The more important tradeoff is between the three scratchpad positions, because they differ not just in token count but in latency architecture, failure mode topology, and debuggability. Let us go through each one.

## 2. External scratchpad: visible CoT in agent message history

The simplest scratchpad is a piece of text that the model writes into the current conversation before writing the final answer. You prompt for it; the model emits it; it lives in the `messages` array alongside everything else.

### How it works

The canonical external CoT prompt is a system instruction: `"Before answering, think step by step. Write your reasoning between <thinking> and </thinking> tags, then give your final answer."` The model then produces something like:

```text
<thinking>
The user wants to know the capital city of the country whose name is an anagram of "RAIN".
Anagram of RAIN: IRAN, RANI, NAÏR, NAIRU...
IRAN is a country. Capital of Iran: Tehran.
</thinking>
Tehran.
```

Everything between the tags is the external scratchpad. You can parse it, log it, validate it, or inject feedback into it. You can also truncate it if it grows too long, or replace it with a summary once it exceeds your context budget.

### Variants

There are three main prompt patterns:

**Zero-shot CoT.** You add one instruction: "Think step by step." The model decides what to write. Zero setup cost; useful when you do not have labeled examples for the specific domain. The downside is that the model's interpretation of "think step by step" is inconsistent across task types — sometimes it writes a brief bullet list, sometimes a full paragraph of hedging, sometimes it writes out the wrong reasoning chain very confidently.

**Few-shot CoT.** You provide two to five (question, reasoning trace, answer) triples as examples. The model then imitates the format and quality of reasoning you demonstrated. This is significantly more accurate than zero-shot on structured tasks (math, code, multi-hop reasoning). The downside is maintenance: you need domain-appropriate examples, and a bad example teaches the wrong reasoning pattern.

**Automatic CoT.** You cluster your questions by type, then use the LLM to auto-generate reasoning traces for representative questions in each cluster, filter out the ones that produced wrong answers, and use the survivors as few-shot examples. This gives you few-shot quality without the manual authoring. The trade is an offline pipeline cost and a corpus that can go stale as your query distribution shifts.

![CoT trace flowing from raw prompt through decompose, evaluate, ground, and execute action](/imgs/blogs/scratchpad-and-extended-thinking-2.webp)

### What goes in the scratchpad that actually helps

Not all scratchpad content is equally useful. Based on empirical studies and production debugging:

**Sub-goal decomposition helps.** Writing "I need to: (1) find X, (2) verify X against Y, (3) format the result" before doing any of those steps measurably reduces errors compared to doing them inline. The model essentially creates a contract with itself about what it is going to do.

**Explicit falsification helps.** Writing "if the date is before 2020, this approach fails because..." before checking the date forces the model to consider its own preconditions. This dramatically reduces errors where the model applies the right reasoning to the wrong case.

**Intermediate numerical bookkeeping helps.** Writing "running total: 42" at each accumulation step reduces arithmetic errors by an order of magnitude compared to doing multi-step math without intermediate notation.

**Venting uncertainty helps a little.** "I am not sure whether X or Y applies here; I will proceed with X and note the caveat" is better than proceeding silently. But it does not eliminate the error if the uncertain choice is wrong.

**Social self-congratulation wastes tokens.** "That's a great question! Let me think about this carefully..." is pure noise. "In conclusion, as I have shown above..." is pure noise. If you are paying for the tokens, strip these out in preprocessing.

**Repeating the question wastes tokens.** The model does not need to restate the prompt to itself before answering. This is a CoT antipattern induced by some training distributions.

## 3. Internal scratchpad: thinking tokens

Claude's extended thinking and OpenAI's o1/o3 family implement a fundamentally different scratchpad: the model reasons in a hidden internal state, and you only see the final answer. The thinking tokens are generated and consumed by the model but not returned to you (or returned in a redacted form).

![CoT variants compared across accuracy, token cost, latency, and engineering control](/imgs/blogs/scratchpad-and-extended-thinking-3.webp)

### Why hide the scratchpad?

The original motivation is alignment and honesty: a visible scratchpad creates an incentive for the model to produce "reasoning theater" — writing a plausible-looking reasoning trace that does not actually reflect how the model computed the answer. If the scratchpad is hidden, there is no audience for theater; the model is incentivized to use the space for genuinely useful computation.

There is also a practical benefit for multi-step planning: internal thinking tokens can represent partial states and hypothesis trees that would be confusing if visible. A hidden token budget is effectively a working memory that the model can fill, erase, and refill without the content cluttering the conversation context.

### What is actually in the thinking tokens?

For Claude, extended thinking content is labeled `type: "thinking"` in the streaming response and is redacted (replaced with a hash) in the final response for privacy and to prevent prompt injection via thinking content. For OpenAI o1/o3, the thinking tokens are simply not returned.

From leaked examples, model output examples where thinking was accidentally exposed, and Anthropic's own description of the mechanism: the thinking trace is essentially a free-form internal monologue. It is not structured JSON; it does not follow a fixed schema. The model writes hypotheses, tests them, discards them, restarts, tries alternative formulations. It is closer to a mathematician filling a whiteboard than to a structured algorithm.

The key implication for production: you cannot inspect, validate, or constrain the content of the thinking trace. You can only observe the output and set a token budget. If the thinking goes wrong — if the model gets stuck in an unproductive loop — you will only see that the output is wrong, not why.

## 4. CoT prompting variants in depth

The three CoT variants differ enough that choosing between them is a real engineering decision, not a stylistic preference.

### Zero-shot CoT

```python
# Python, anthropic SDK 0.40+
import anthropic

client = anthropic.Anthropic()

def zero_shot_cot(question: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system="You are a careful reasoning assistant. Before answering, "
               "think through the problem step by step between <thinking> "
               "and </thinking> tags.",
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

# Usage
result = zero_shot_cot(
    "A store has 37 items at $12.50 each. A customer buys 3. "
    "They pay with $50. What is the change?"
)
print(result)
# <thinking>
# Items bought: 3, price per item: $12.50
# Total cost: 3 × $12.50 = $37.50
# Change: $50.00 - $37.50 = $12.50
# </thinking>
# The change is $12.50.
```

Token overhead for this example: roughly 60 thinking tokens + 10 answer tokens = 70 total, versus 5 tokens for a direct answer. For 10,000 queries per day at Claude claude-sonnet-4-6 pricing ($3/MTok output), that is an extra $0.0018/day — negligible. For an enterprise batch of 10M queries, it matters.

### Few-shot CoT

```python
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "A train travels 60 mph for 2.5 hours. How far?"
    },
    {
        "role": "assistant", 
        "content": """<thinking>
Distance = speed × time
Speed: 60 mph
Time: 2.5 hours
Distance: 60 × 2.5 = 150 miles
</thinking>
The train travels 150 miles."""
    },
    {
        "role": "user",
        "content": "A recipe calls for 2.5 cups flour per batch. "
                   "You need 7 batches. How much flour?"
    },
    {
        "role": "assistant",
        "content": """<thinking>
Flour per batch: 2.5 cups
Batches: 7
Total flour: 2.5 × 7 = 17.5 cups
</thinking>
You need 17.5 cups of flour."""
    }
]

def few_shot_cot(question: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system="You are a careful reasoning assistant. Always think step by "
               "step between <thinking> and </thinking> tags before answering.",
        messages=[*FEW_SHOT_EXAMPLES, {"role": "user", "content": question}]
    )
    return response.content[0].text
```

Few-shot CoT adds the example token count to every call. The two-example set above adds roughly 200 tokens per request in context. For queries where accuracy is critical (customer-facing financial calculations, medical triage) this overhead pays for itself immediately.

### Automatic CoT

Auto-CoT requires an offline step:

```python
import random
from typing import list

def generate_auto_cot_examples(
    questions: list[str],
    n_clusters: int = 8,
    n_per_cluster: int = 1
) -> list[dict]:
    """
    Generate few-shot CoT examples by clustering questions and
    auto-generating reasoning traces for representative questions.
    
    Steps:
    1. Embed all questions
    2. K-means cluster into n_clusters groups  
    3. Pick one representative per cluster (centroid nearest)
    4. Generate CoT trace for each; keep if answer is verifiable-correct
    5. Return as few-shot examples
    """
    # Simplified: in production, use sentence-transformers + sklearn KMeans
    sampled = random.sample(questions, min(n_clusters * n_per_cluster, len(questions)))
    
    examples = []
    for q in sampled:
        trace = zero_shot_cot(q)  # generate the trace
        # In production: verify the answer against ground truth
        # Here we trust the model for illustration
        examples.append({"role": "user", "content": q})
        examples.append({"role": "assistant", "content": trace})
    
    return examples
```

The key empirical finding from Zhang et al. (2022): auto-CoT matches few-shot CoT performance on most benchmarks with no manual example authoring. It degrades when the auto-generated traces contain errors — so the filtering step (only keep traces where the answer is verifiable-correct) is not optional.

## 5. Extended thinking APIs: how to use them

Claude's extended thinking API gives you a first-class way to configure the internal scratchpad at the call level.

![Extended thinking API call timeline showing thinking tokens then final answer](/imgs/blogs/scratchpad-and-extended-thinking-4.webp)

### Basic usage

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-5",  # extended thinking requires claude-3-7-sonnet-20250219+
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # how many tokens to spend on thinking
    },
    messages=[{
        "role": "user",
        "content": "A company has 3 product lines. Product A has 40% margin "
                   "and $2M revenue. Product B has 25% margin and $5M revenue. "
                   "Product C has 60% margin and $800k revenue. "
                   "If you could only keep two product lines and had to maximize "
                   "total gross profit, which two do you keep and why?"
    }]
)

# Inspect content blocks
for block in response.content:
    if block.type == "thinking":
        print(f"Thinking tokens: {len(block.thinking)} chars (redacted hash: {block.signature[:16]}...)")
    elif block.type == "text":
        print(f"Answer: {block.text}")
```

The `budget_tokens` parameter is the most important lever. It sets the maximum number of tokens the model can use for internal reasoning. The model can use fewer tokens if it converges early; it cannot exceed the budget.

### Streaming extended thinking

For production agents where latency matters, streaming is mandatory — you want to start processing the answer as soon as the first answer token arrives, not wait for the full response:

```python
import anthropic
from anthropic import MessageStreamManager

client = anthropic.Anthropic()

def stream_extended_thinking(
    question: str, 
    budget_tokens: int = 8000
):
    thinking_tokens_count = 0
    answer_parts = []
    
    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=budget_tokens + 2000,  # budget + answer headroom
        thinking={
            "type": "enabled",
            "budget_tokens": budget_tokens
        },
        messages=[{"role": "user", "content": question}]
    ) as stream:
        for event in stream:
            if hasattr(event, 'type'):
                if event.type == "content_block_start":
                    if hasattr(event.content_block, 'type'):
                        block_type = event.content_block.type
                        print(f"\n[{block_type} block starting]")
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, 'type'):
                        if delta.type == "thinking_delta":
                            # Thinking tokens: increment counter, don't display
                            thinking_tokens_count += len(delta.thinking.split())
                        elif delta.type == "text_delta":
                            # Answer tokens: stream to output
                            print(delta.text, end="", flush=True)
                            answer_parts.append(delta.text)
    
    print(f"\n\n[Thinking tokens used: ~{thinking_tokens_count}]")
    return "".join(answer_parts)

# Usage
answer = stream_extended_thinking(
    "Design a schema for a multi-tenant SaaS application that needs to "
    "support 10k tenants with strong data isolation, efficient cross-tenant "
    "analytics, and sub-50ms query latency for single-tenant operations.",
    budget_tokens=12000
)
```

### The cost model

Extended thinking has a specific cost structure that catches engineers off-guard:

- **Thinking tokens are billed as output tokens** at the full output token rate, not the cheaper input rate.
- **The budget is a ceiling, not a guarantee.** The model can converge early and stop spending.
- **Thinking tokens do not appear in context on the next call** — they are ephemeral unless you explicitly store and re-inject them.

Concrete numbers at claude-3-7-sonnet pricing (approximate as of 2026):
- Input tokens: $3/MTok
- Output tokens: $15/MTok

For a query with 1k input tokens, `budget_tokens=16000`, and an answer of 500 tokens:
- Without extended thinking: `1000 × $3/1M + 500 × $15/1M = $0.003 + $0.0075 = $0.0105`
- With extended thinking (12k thinking, 500 answer): `1000 × $3/1M + 12500 × $15/1M = $0.003 + $0.1875 = $0.1905`

That is a 18× cost multiplier for a single call. For batch pipelines running 100k queries per day, this is the difference between a $1,050/day inference bill and a $19,050/day inference bill. Choosing the right `budget_tokens` value is an engineering optimization problem, not a knob to leave at the default.

### Setting budget_tokens

The empirical pattern I have seen across production deployments:

| Task type | Recommended budget_tokens | Reasoning |
|---|---|---|
| Simple fact retrieval | 0 (disable) | No benefit |
| Single-step math | 1,000–2,000 | Brief trace suffices |
| Multi-step math (GSM8K level) | 4,000–8,000 | Sufficient for most problems |
| Competition math (MATH level) | 8,000–16,000 | Harder problems need space |
| Complex planning (multi-tool) | 8,000–16,000 | Sub-goal exploration |
| Architectural design decisions | 16,000–32,000 | Trade-off enumeration |
| Rarely: extreme reasoning | 32,000–64,000 | Watch for overthinking |

Never set `budget_tokens` above 64,000 without benchmarking. The overthinking failure mode (Section 9) becomes significant above that threshold.

## 6. What actually helps vs. what wastes tokens

![Before CoT vs with CoT: wrong answer vs correct step-by-step derivation](/imgs/blogs/scratchpad-and-extended-thinking-5.webp)

Understanding what the scratchpad should contain lets you write better CoT prompts and set better budgets.

### High-value scratchpad content

**Explicit constraint propagation.** When a problem has multiple constraints (a scheduling problem, a resource allocation problem), writing each constraint explicitly and checking whether each candidate solution satisfies each one is worth 3–5× the token cost. The model without a scratchpad tends to satisfy the most salient constraints and silently violate the less obvious ones.

**Dead-end acknowledgment.** "Approach A fails because X. Try approach B." This is worth tokens. It prevents the model from cycling back to the same failed approach later in the trace (a failure mode visible in long extended thinking traces).

**Partial state bookkeeping.** In a multi-hop retrieval chain (question → entity extraction → search → result parsing → synthesized answer), writing the current entity and search result at each step reduces dropped context. Without this, the model can arrive at step 4 having lost track of what it extracted in step 1.

**Explicit uncertainty flagging.** "I am not sure whether the question is asking for X or Y. I will assume X." This is worth tokens because it creates a checkpoint: if the final answer is wrong, you can check the assumption. If you are logging scratchpad traces, these flagged uncertainties are where your debugging starts.

### Low-value scratchpad content

**Restating the question.** The model does not need to reread the question to itself. This wastes 50–200 tokens per call.

**Rehearsal confidence checks.** "I am now ready to compute the answer." Pure noise. 20 tokens wasted.

**Narrative connectives.** "Furthermore, it is worth noting that..." Zero information content. These appear in scratchpads trained on academic text.

**Premature conclusions.** "Therefore, the answer is X" placed halfway through a trace that then contradicts it. This pattern (write conclusion early, then reason backward) is a known failure mode of certain fine-tuning distributions and produces wrong answers more reliably than no CoT at all.

**Over-qualifying factual statements.** "It is generally believed that, in most cases, water typically boils at approximately 100°C under normal conditions." This is hedging theater. In a reasoning trace it costs tokens and adds no information.

The practical implication: if you are paying for external CoT tokens and can postprocess the trace, stripping connector phrases and restated questions can cut your CoT token overhead by 20–30% with no accuracy loss.

## 7. Scratchpad in the ReAct loop

The [ReAct pattern](/blog/machine-learning/ai-agent/react-pattern-deep-dive) — Reason, Act, Observe — has a natural home for a scratchpad between each Act and the next Reason. The scratchpad in a ReAct loop serves a different purpose than a scratchpad for a single-shot problem: it is accumulating state across multiple tool calls, not just decomposing a single question.

```python
import anthropic
import json
from typing import Callable

client = anthropic.Anthropic()

def react_agent_with_scratchpad(
    task: str,
    tools: list[dict],
    tool_fns: dict[str, Callable],
    max_iterations: int = 10,
    scratchpad_budget: int = 4000
) -> str:
    """
    ReAct agent with external scratchpad accumulated across turns.
    
    The scratchpad is stored in a separate system slot and updated after
    each observation. This prevents it from being lost to context pressure
    while keeping it inspectable.
    """
    
    scratchpad = []  # accumulates reasoning across turns
    messages = []
    
    system_prompt = """You are a careful agent. At each step:
1. Write your current reasoning in <thinking>...</thinking> tags
2. Decide: either call a tool OR give a final answer
3. Never re-derive facts already in your scratchpad summary

Current scratchpad summary:
{scratchpad_summary}"""
    
    for iteration in range(max_iterations):
        # Build scratchpad summary for injection
        scratchpad_summary = "\n".join(scratchpad[-5:]) if scratchpad else "Empty — this is the first step."
        
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=scratchpad_budget + 1000,
            system=system_prompt.format(scratchpad_summary=scratchpad_summary),
            tools=tools,
            messages=[
                {"role": "user", "content": task},
                *messages
            ]
        )
        
        # Extract thinking
        thinking_text = ""
        answer_text = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                # Parse thinking tags
                if "<thinking>" in block.text:
                    start = block.text.index("<thinking>") + len("<thinking>")
                    end = block.text.index("</thinking>")
                    thinking_text = block.text[start:end].strip()
                    answer_text = block.text[block.text.index("</thinking>") + len("</thinking>"):].strip()
                else:
                    answer_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(block)
        
        # Update scratchpad
        if thinking_text:
            scratchpad.append(f"[Turn {iteration+1}] {thinking_text[:500]}")
        
        # If no tool calls, we have a final answer
        if not tool_calls:
            return answer_text
        
        # Execute tools and add to messages
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        
        for tc in tool_calls:
            fn = tool_fns.get(tc.name)
            if fn:
                result = fn(**tc.input)
            else:
                result = f"Error: tool {tc.name} not found"
            
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": str(result)
            })
            
            # Add observation to scratchpad
            scratchpad.append(
                f"[Observation: {tc.name}({tc.input})] → {str(result)[:200]}"
            )
        
        messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached without final answer."
```

The key design decision: the scratchpad is injected into the `system` prompt as a compressed summary, not appended to `messages`. This means:
1. It does not grow the `messages` list with every turn (preventing quadratic context growth).
2. It is explicitly managed — you control what gets compressed and what gets dropped.
3. It is inspectable at every turn for debugging.

The alternative — letting the full reasoning trace accumulate in the `messages` array — works for short tasks but causes two failure modes on long tasks: context window exhaustion (the conversation gets longer than the model's context limit) and attention dilution (important early facts become hard for the model to attend to when they are buried under many later turns).

## 8. Persisting scratchpad state across multi-turn conversations

The most underengineered part of production agent systems is scratchpad persistence. Most teams discover this the hard way: the agent works fine on five-turn conversations in development, fails on fifteen-turn conversations in production.

![Multi-turn scratchpad persistence across context boundaries](/imgs/blogs/scratchpad-and-extended-thinking-8.webp)

The problem is architectural. Each API call is stateless. The model has no memory of previous turns except what you put in the `messages` array. If you naively append every turn to `messages`, you eventually hit the context window limit. If you truncate `messages` when they get too long, you lose the early scratchpad content that informed the current reasoning state.

### The summarize-and-inject pattern

The standard solution is a two-stage pipeline:

**Stage 1: Compress.** When the conversation history exceeds a threshold (typically 40–60% of the context window), trigger a compression call that asks the model to summarize the reasoning state so far into a structured working memory block:

```python
def compress_scratchpad(messages: list[dict], current_task: str) -> str:
    """
    Compress the conversation history into a working memory block.
    Preserves: decisions made, facts established, current state, open questions.
    Discards: intermediate calculations already resolved, tool call raw outputs.
    """
    
    # Serialize the relevant parts
    history_text = "\n".join([
        f"[{m['role'].upper()}]: {m['content'][:500]}"
        for m in messages
        if isinstance(m.get('content'), str)
    ])
    
    response = client.messages.create(
        model="claude-haiku-4-5",  # Use cheaper model for compression
        max_tokens=600,
        system="""Extract working memory from this conversation. Output a JSON block with:
- facts_established: list of facts that are now settled
- decisions_made: list of decisions and their rationale  
- current_state: where the agent is in the task
- open_questions: what still needs to be resolved
Be extremely concise. Each item max 20 words.""",
        messages=[{
            "role": "user",
            "content": f"Task: {current_task}\n\nHistory:\n{history_text}"
        }]
    )
    
    return response.content[0].text  # ~500 tokens of structured memory
```

**Stage 2: Inject.** On the next turn, inject the compressed working memory into the system prompt:

```python
def next_turn_with_memory(
    user_message: str,
    working_memory: str,
    current_task: str
) -> str:
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=f"""You are a careful agent continuing a multi-turn task.

TASK: {current_task}

WORKING MEMORY (established facts and decisions — treat as ground truth):
{working_memory}

Do not re-derive facts already in working memory. Continue from the current state.""",
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text
```

This approach keeps the context window manageable across arbitrarily long conversations, at the cost of one extra API call per compression trigger. In practice, using `claude-haiku-4-5` for compression and `claude-opus-4-5` for reasoning keeps the compression call to ~$0.001, negligible against the reasoning call cost.

### When to trigger compression

A simple heuristic: count tokens in the `messages` array after each turn. When the count exceeds 40% of the model's context window, compress. For claude-3-7-sonnet-20250219 with a 200k context window, this is 80,000 tokens — a long conversation. For GPT-4o with an 8k context window, it triggers after a few turns.

A more sophisticated approach: track which content in the messages array has been referenced in the most recent N turns. Content that has not been referenced in the last 5 turns is a compression candidate — it is either resolved or forgotten by the model already.

## 9. The overthinking failure mode

Extended thinking and long CoT traces share a failure mode that does not exist for short direct completions: overthinking. The model thinks its way into a wrong answer by considering too many alternatives, second-guessing correct intermediate conclusions, and eventually choosing a more "sophisticated" wrong path over a simpler correct one.

![Thinking length vs accuracy: rapid gain then plateau then degradation](/imgs/blogs/scratchpad-and-extended-thinking-6.webp)

The empirical picture (based on published results from OpenAI's o1 and Anthropic's extended thinking studies, plus internal benchmarking):

- **0–500 thinking tokens:** Accuracy near the no-CoT baseline for most tasks. The reasoning trace is too short to actually decompose complex problems.
- **500–4,000 thinking tokens:** Rapid accuracy improvement. This is the sweet spot for most reasoning tasks — enough space to decompose and verify, not enough to drift.
- **4,000–16,000 thinking tokens:** Diminishing returns on most tasks. The model has found the answer; additional thinking tokens are spent on unnecessary elaboration or alternative approaches.
- **16,000–64,000 thinking tokens:** Plateau. Accuracy stable or very slightly declining. For extremely hard tasks (competition math, complex software architecture), some tasks show continued improvement.
- **64,000+ thinking tokens:** Measurable degradation on tasks where the optimal answer is not the most sophisticated one. The model overthinks "obvious" cases and produces the more elaborate wrong answer.

### Why overthinking happens

The mechanism is related to the model's training distribution. Most training examples reward longer, more detailed reasoning traces (because human raters often prefer more thorough answers). This creates a bias: given unlimited thinking budget, the model tends to keep thinking even when it has found the right answer, because stopping early feels like insufficient effort.

The practical consequence: the model writes "Therefore X is correct" at step 12 of its trace, then continues writing "Although, one might argue that Y..." and eventually arrives at "On reflection, Y seems more defensible" — despite X being the correct answer. This pattern appears most often in:

- Questions where the obvious answer is correct but sounds too simple (mathematical trick questions)
- Ethical dilemmas where the model keeps finding more nuanced considerations
- Architectural decisions where the simple solution is actually right

### Mitigations

**Set a tighter budget.** The most effective mitigation is simply not giving the model enough tokens to overthink. For most production tasks, 4,000–8,000 budget tokens is sufficient. If you see correct answers at 4,000 tokens being replaced by wrong answers at 16,000 tokens, you have an overthinking problem — cut the budget.

**Early stopping signals.** Some teams add a system prompt instruction: "If you are confident in your answer, stop reasoning and give it. Do not keep generating alternatives once you have a working solution." This works but is not reliable — the model may decide it is not confident when it should be.

**Benchmark before deploying.** If you are using extended thinking in production, you must benchmark the accuracy curve at multiple budget sizes on held-out examples before setting the production budget. The accuracy-budget relationship is task-specific enough that general rules do not substitute for measurement.

## 10. Measuring scratchpad quality: does more thinking equal better answers?

The answer is: it depends on the task, and measuring it is non-trivial.

![External scratchpad vs internal thinking tokens compared across six dimensions](/imgs/blogs/scratchpad-and-extended-thinking-7.webp)

### Accuracy metrics

For tasks with ground-truth correct answers (math, code, logical reasoning), measuring scratchpad quality is straightforward: does adding the scratchpad increase the fraction of correct final answers?

```python
import statistics
from typing import Callable

def benchmark_scratchpad(
    questions: list[dict],  # [{"question": ..., "answer": ...}]
    model: str,
    budgets: list[int],
    n_trials: int = 3
) -> dict[int, float]:
    """
    Measure accuracy at each budget_tokens setting.
    Returns {budget: accuracy} dict.
    """
    results = {}
    
    for budget in budgets:
        correct = []
        
        for item in questions:
            for _ in range(n_trials):
                if budget == 0:
                    # No thinking
                    response = client.messages.create(
                        model=model,
                        max_tokens=500,
                        messages=[{"role": "user", "content": item["question"]}]
                    )
                    answer = response.content[0].text
                else:
                    response = client.messages.create(
                        model=model,
                        max_tokens=budget + 500,
                        thinking={"type": "enabled", "budget_tokens": budget},
                        messages=[{"role": "user", "content": item["question"]}]
                    )
                    answer = next(
                        (b.text for b in response.content if b.type == "text"),
                        ""
                    )
                
                is_correct = check_answer(answer, item["answer"])
                correct.append(int(is_correct))
        
        results[budget] = statistics.mean(correct)
        print(f"Budget {budget}: accuracy={results[budget]:.3f}")
    
    return results

def check_answer(model_answer: str, expected: str) -> bool:
    """Exact match or number extraction — extend for your task type."""
    # Simplified: in production, use a judge model or task-specific parser
    return expected.lower() in model_answer.lower()
```

### Scratchpad quality metrics

For tasks without clean ground truth (open-ended planning, architectural decisions), you need proxy metrics:

**Reasoning coherence.** Does each step follow logically from the previous one? A simple proxy: extract sentences from the CoT, embed them, and measure average cosine similarity to the previous sentence. A coherent trace has moderate similarity (0.4–0.7) throughout. Very high similarity (>0.8) suggests repetition. Very low similarity (<0.3) suggests disconnected reasoning.

**Constraint satisfaction.** How many of the explicitly stated constraints in the problem appear as explicit checklist items in the CoT? A higher fraction suggests more systematic reasoning.

**Dead-end ratio.** What fraction of the CoT contains explicit dead-end acknowledgments ("approach X fails because...")? A moderate dead-end ratio (5–15%) indicates the model is actually exploring the space. Zero dead-ends on a hard problem suggests shallow reasoning. High dead-end ratio (>30%) suggests the model is flailing.

**Answer stability.** Run the same prompt 5 times with the same budget. Compute the variance of the final answers. High-quality reasoning produces consistent answers. High variance at a given budget indicates the reasoning is not actually grounding the answer.

## 11. Case studies

### Case study 1: The legal contract analysis that needed 16,000 thinking tokens

A legal tech startup built a contract review agent to flag non-standard clauses. Initial deployment used zero-shot CoT with a 4,000 token thinking budget. Accuracy on their internal benchmark: 73%. After increasing to 16,000 tokens, accuracy rose to 89%.

The root cause: contract analysis requires simultaneously tracking multiple clause hierarchies, cross-references, and conditional dependencies. The 4,000 token budget was enough to analyze each clause in isolation but not enough to maintain the full clause graph while analyzing any individual one. The model would correctly identify that clause 12 was non-standard in isolation, then fail to notice that clause 3 had a carve-out that made clause 12 standard after all.

At 16,000 tokens, the model had enough space to build a mental clause map before analyzing individual clauses. The overhead: 12 additional seconds of latency (attorneys were fine with 15-second async responses) and a 4× cost increase. For a service billing per review at $50/document, the increased inference cost of $0.15/document was irrelevant.

Lesson: when the problem requires maintaining a global data structure (a graph, a table, a hierarchy) while performing local analysis, the thinking budget needs to be proportional to the graph size.

### Case study 2: The customer service bot that got slower and worse

An e-commerce company added extended thinking to their customer service chatbot after reading that it improved accuracy. Thinking budget: 8,000 tokens. The results were worse: accuracy declined 4 percentage points, and average response time went from 1.2 seconds to 4.8 seconds.

The investigation: the chatbot was handling queries like "Where is my order?" and "What is your return policy?" — single-hop factual retrieval from a known database. There was no reasoning to do. The extended thinking tokens were being spent on:

1. Restating the query to itself
2. Considering and rejecting alternative interpretations of the query
3. Writing out confidence qualifiers
4. Choosing between two essentially identical phrasings of the correct answer

The model's accuracy on these tasks was already near ceiling (91%) without thinking. The 4% decline was due to overthinking: the model occasionally second-guessed a clearly correct answer and switched to a wrong one.

Fix: disable extended thinking for intent classification and factual lookup, keep it enabled for the escalation path (complex multi-step complaints, refund policy edge cases). Net result: latency back to 1.2 seconds for 80% of queries; accuracy +11% on the complex 20%.

Lesson: always measure the accuracy-budget curve before deploying. The right budget for most production tasks is much lower than you think.

### Case study 3: The financial model that needed external scratchpad, not extended thinking

A hedge fund built an agent to analyze earnings call transcripts and extract three signals: management tone shift, guidance changes, and unusual phrasing patterns. They initially deployed with extended thinking. Two problems emerged:

1. **Auditability.** The fund's compliance team required a documented reasoning trace for every position taken. Extended thinking does not provide this — the trace is redacted.
2. **Consistency.** Running the same transcript twice with the same budget gave different signal outputs about 30% of the time. The hidden thinking trace was converging to different local optima.

They rebuilt with external CoT: the scratchpad was written into the assistant message and stored alongside the output. The reasoning trace became part of the audit log. Consistency improved to 95% (the remaining 5% variance was acceptable and traceable to legitimate ambiguity in the source text).

The cost trade-off: external CoT at 1,500 tokens per document versus extended thinking at 8,000 thinking tokens per document. The cheaper approach was also more compliant.

Lesson: in regulated industries or anywhere reasoning traces are legally required, external scratchpad beats extended thinking. The quality gap is usually smaller than the auditability gap is large.

### Case study 4: The code review agent that learned to stop early

A platform engineering team deployed a code review agent with a 16,000 token thinking budget. It produced excellent reviews — thorough, well-reasoned, technically accurate. After three months, they ran a cost analysis: the agent was spending $0.80 per review, 70% of which was thinking token costs.

They ran a budget sweep experiment: 1,000 / 2,000 / 4,000 / 8,000 / 16,000 token budgets on a held-out sample of 500 diffs. Results:

| Budget | Accuracy vs. 16k | Cost per review |
|---|---|---|
| 1,000 | −18% | $0.12 |
| 2,000 | −9% | $0.18 |
| 4,000 | −3% | $0.29 |
| 8,000 | −1% | $0.46 |
| 16,000 | baseline | $0.80 |

At 4,000 tokens, they recovered 97% of the accuracy at 36% of the cost. The remaining 3% accuracy gap was on large refactors (>500 lines changed) — they kept 8,000 tokens for those, which the agent classified automatically by diff size.

Lesson: the cost-accuracy trade-off is rarely linear. Benchmark your specific task at multiple budget levels. The Pareto optimal point is usually at a lower budget than the "use as much as possible" heuristic suggests.

### Case study 5: The multi-step research agent with scratchpad loss

A research automation agent ran 10–20 tool calls per task (web search, document parsing, entity extraction, synthesis). It worked correctly in testing (5-turn sessions) and failed in production (15–20 turn sessions).

The failure mode: at turn 12, the agent would repeat a search it had already done at turn 3, then come to the opposite conclusion about the same entity. Investigation revealed the cause: by turn 12, the early turns were so far back in context that the model was effectively ignoring them. The scratchpad content from turns 1–5 was in the messages array but outside the model's effective attention window.

The fix: after every 5 turns, trigger a scratchpad compression call (using `claude-haiku-4-5` to save cost) that extracted the current working state into a 400-token JSON block. This block was injected into the system prompt on every subsequent turn. The messages array was pruned to the last 5 turns plus tool results.

After the fix: 95% task success rate on 20-turn sessions. Before: 41%. The compression call added $0.002 per triggered compression, negligible against the $0.20+ per research task cost.

Lesson: scratchpad persistence is not optional for multi-turn agents. The default behavior (let messages accumulate) is only correct for tasks under ~8 turns. Plan the compression architecture from the start.

### Case study 6: The math tutoring bot and the overthinking wall

A math tutoring platform used extended thinking to generate step-by-step solution explanations. At 8,000 thinking tokens, the solutions were excellent. They upgraded to a newer, larger model and kept the same 8,000 token budget. Accuracy on their benchmark dropped 6%.

Investigation: the newer model had a longer "settling time" — it spent more tokens on preamble and alternative consideration before committing to a solution path. At 8,000 tokens, the model was hitting the budget ceiling before completing its reasoning trace, then being forced to generate a final answer based on an incomplete trace. This produced confident wrong answers.

The fix: increase budget to 12,000 tokens for the new model. Accuracy exceeded the old model at 8,000 tokens. The lesson is non-obvious: upgrading to a stronger model can *require* a larger thinking budget to realize its benefits, because the stronger model uses more tokens to explore the problem space before converging.

A useful diagnostic: look at the distribution of how much of the thinking budget the model actually uses. If more than 20% of requests are consuming > 95% of the budget, you have a budget-constrained problem, not an overthinking problem. Increase the budget. If fewer than 5% of requests exceed 50% of the budget, you are probably over-provisioning.

## 12. When to use scratchpads — and when not to

![8 task categories rated for scratchpad benefit](/imgs/blogs/scratchpad-and-extended-thinking-10.webp)

The decision tree below consolidates the rules of thumb from all the preceding sections:

![Scratchpad strategy decision tree: task complexity → latency → cost → strategy](/imgs/blogs/scratchpad-and-extended-thinking-9.webp)

### Use a scratchpad when

**The task requires more than one inference step.** If the correct answer cannot be looked up in a single retrieval operation, a scratchpad helps. Multi-step math, code generation, multi-hop Q&A, planning tasks: all benefit.

**Errors compound.** Any task where an early error propagates to make later steps wrong is a scratchpad candidate. The scratchpad creates checkpoints where errors can be caught.

**You need auditability.** If you need to explain why the agent gave a particular output, external CoT is the only option. The reasoning trace is your audit log.

**The task has hidden constraints.** Legal documents, complex requirements specs, architectural designs: tasks where the full constraint set is not explicit in the question benefit from a scratchpad because the model can enumerate constraints explicitly.

### Do not use a scratchpad when

**Single-hop factual retrieval.** "What is the capital of France?" has no benefit from a scratchpad. The overhead is pure cost with no accuracy return.

**Latency is under 500ms.** External CoT adds latency proportional to token count. Extended thinking adds latency proportional to `budget_tokens`. If your SLA is under 500ms, neither is viable at useful budget levels.

**You cannot absorb the token cost.** If a 5× token multiplier makes your unit economics unworkable, disable the scratchpad and invest in better retrieval, better prompting, or better fine-tuned models instead. Scratchpads are not the only path to accuracy.

**The task is classification into a small label set.** Sentiment analysis, intent detection, toxicity filtering: for tasks with ≤20 labels and a well-prompted model, zero-shot accuracy is typically within 2 percentage points of CoT accuracy. The overhead is not justified.

**You are running a stateless microservice at high QPS.** Extended thinking + streaming + stateless HTTP is possible, but the latency budget is hard to meet. At 10,000 QPS with a 2-second SLA, you cannot afford 8,000 thinking tokens per request at current inference speeds.

### The meta-rule

Measure first. Every "use scratchpad" or "don't use scratchpad" recommendation in this post is a default derived from empirical patterns. Your task might be an exception. The cost of running a budget sweep on 200 held-out examples is 30 minutes and $10 in API costs. The cost of deploying with the wrong budget in production is much higher.

Use the decision tree as a starting hypothesis. Benchmark it. Adjust. The right scratchpad strategy is the one you measured, not the one you assumed.

---

## Cross-links

For the broader agent architecture context, see [Agent Loop Anatomy](/blog/machine-learning/ai-agent/agent-loop-anatomy) — the scratchpad is one component of the larger observe-plan-act cycle. For how the ReAct loop integrates reasoning and acting, see [ReAct Pattern Deep Dive](/blog/machine-learning/ai-agent/react-pattern-deep-dive). For agents that use the scratchpad to evaluate and revise their own outputs, see [Reflection and Self-Critique Agents](/blog/machine-learning/ai-agent/reflection-and-self-critique-agents).
